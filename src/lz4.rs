use anyhow::Result;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct LZ4BlockDescriptor {
    pub compressed_size: u32,
    pub uncompressed_size: u32,
    pub compressed_offset: u32,
    pub output_offset: u32,
    pub is_compressed: bool,
}

#[derive(Debug, Clone)]
pub struct LZ4CompressedFrame {
    pub uncompressed_size: usize,
    pub block_size: usize,
    pub blocks: Vec<LZ4BlockDescriptor>,
    pub payload: Arc<[u8]>,
    pub total_compressed_bytes: usize,
    pub reported_content_size: Option<usize>,
    pub uses_block_checksum: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum LZ4Error {
    #[error("Malformed stream")]
    MalformedStream,
    #[error("Output overflow")]
    OutputOverflow,
    #[error("Unsupported frame: {0}")]
    UnsupportedFrame(String),
}

const LZ4_MAX_MATCH_LENGTH: usize = 0xFFFF + 4;
const LZ4_MIN_MATCH: usize = 4;
const LZ4_MAX_OFFSET: usize = 0xFFFF;

pub struct LZ4Decompressor;

impl LZ4Decompressor {
    pub fn new() -> Self {
        Self
    }

    pub fn decompress(
        &self,
        frame: &LZ4CompressedFrame,
        concurrency: Option<usize>,
    ) -> Result<Vec<u8>> {
        if frame.blocks.len() < 2 || concurrency == Some(1) {
            self.decompress_sequential(frame)
        } else {
            self.decompress_parallel(frame, concurrency)
        }
    }

    fn decompress_sequential(&self, frame: &LZ4CompressedFrame) -> Result<Vec<u8>> {
        let mut output = vec![0u8; frame.uncompressed_size];

        for block in &frame.blocks {
            let block_input = &frame.payload[block.compressed_offset as usize..]
                [..block.compressed_size as usize];
            let block_output =
                &mut output[block.output_offset as usize..][..block.uncompressed_size as usize];

            if block.is_compressed {
                self.decompress_block(block_input, block_output)?;
            } else {
                block_output.copy_from_slice(block_input);
            }
        }

        Ok(output)
    }

    fn decompress_parallel(
        &self,
        frame: &LZ4CompressedFrame,
        concurrency: Option<usize>,
    ) -> Result<Vec<u8>> {
        use rayon::prelude::*;

        let payload = &frame.payload;

        // Set up parallel processing
        let concurrency = concurrency.unwrap_or_else(|| num_cpus::get());
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(concurrency)
            .build_global();

        // Process blocks in parallel and collect results
        let block_results: Result<Vec<(usize, Vec<u8>)>, anyhow::Error> = frame
            .blocks
            .par_iter()
            .enumerate()
            .map(|(block_idx, block)| {
                let block_input =
                    &payload[block.compressed_offset as usize..][..block.compressed_size as usize];

                let block_output = if block.is_compressed {
                    let mut output = vec![0u8; block.uncompressed_size as usize];
                    self.decompress_block(block_input, &mut output)?;
                    output
                } else {
                    block_input.to_vec()
                };

                Ok((block_idx, block_output))
            })
            .collect();

        let block_results = block_results?;

        // Combine results into final output
        let mut output = vec![0u8; frame.uncompressed_size];
        for (block_idx, block_data) in block_results {
            let block = &frame.blocks[block_idx];
            let output_start = block.output_offset as usize;
            let output_end = output_start + block.uncompressed_size as usize;
            output[output_start..output_end].copy_from_slice(&block_data);
        }

        Ok(output)
    }

    fn decompress_block(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let mut src = 0;
        let mut dst = 0;
        let input_len = input.len();
        let output_len = output.len();

        while src < input_len {
            let token = input[src];
            src += 1;

            // Decode literal length
            let mut literal_length = (token >> 4) as usize;
            if literal_length == 15 {
                literal_length += self.read_length(&input[src..])?;
                src += self.length_bytes(literal_length - 15);
            }

            // Copy literals
            if src + literal_length > input_len {
                return Err(LZ4Error::MalformedStream.into());
            }
            if dst + literal_length > output_len {
                return Err(LZ4Error::OutputOverflow.into());
            }

            output[dst..dst + literal_length].copy_from_slice(&input[src..src + literal_length]);
            src += literal_length;
            dst += literal_length;

            if src >= input_len {
                break;
            }

            // Decode match
            if src + 2 > input_len {
                return Err(LZ4Error::MalformedStream.into());
            }

            let offset = input[src] as usize | ((input[src + 1] as usize) << 8);
            src += 2;

            if offset == 0 || offset > dst {
                return Err(LZ4Error::MalformedStream.into());
            }

            let mut match_length = ((token & 0x0F) as usize) + LZ4_MIN_MATCH;
            if (token & 0x0F) == 0x0F {
                match_length += self.read_length(&input[src..])?;
                src += self.length_bytes(match_length - LZ4_MIN_MATCH - 15);
            }

            if dst + match_length > output_len {
                return Err(LZ4Error::OutputOverflow.into());
            }

            // Copy match - use temporary buffer to avoid borrowing issues
            let src_start = dst - offset;
            let temp = output[src_start..src_start + match_length].to_vec();
            output[dst..dst + match_length].copy_from_slice(&temp);
            dst += match_length;
        }

        Ok(())
    }

    fn read_length(&self, input: &[u8]) -> Result<usize> {
        let mut length = 0;
        let mut i = 0;

        while i < input.len() {
            let value = input[i] as usize;
            i += 1;
            length += value;
            if value != 255 {
                break;
            }
        }

        Ok(length)
    }

    fn length_bytes(&self, length: usize) -> usize {
        let mut bytes = 0;
        let mut remaining = length;

        while remaining >= 255 {
            bytes += 1;
            remaining -= 255;
        }
        if remaining > 0 {
            bytes += 1;
        }

        bytes
    }

    pub fn measure_decompressed_size(input: &[u8]) -> Result<usize> {
        let mut src = 0;
        let mut produced = 0;
        let input_len = input.len();

        while src < input_len {
            let token = input[src];
            src += 1;

            let mut literal_length = (token >> 4) as usize;
            if literal_length == 15 {
                let mut i = src;
                while i < input_len {
                    let value = input[i] as usize;
                    i += 1;
                    literal_length += value;
                    if value != 255 {
                        break;
                    }
                }
                src = i;
            }

            if src + literal_length > input_len {
                return Err(LZ4Error::MalformedStream.into());
            }

            src += literal_length;
            produced += literal_length;

            if src >= input_len {
                break;
            }

            if src + 2 > input_len {
                return Err(LZ4Error::MalformedStream.into());
            }

            let offset = input[src] as usize | ((input[src + 1] as usize) << 8);
            src += 2;

            if offset == 0 || offset > produced {
                return Err(LZ4Error::MalformedStream.into());
            }

            let mut match_length = ((token & 0x0F) as usize) + LZ4_MIN_MATCH;
            if (token & 0x0F) == 0x0F {
                let mut i = src;
                while i < input_len {
                    let value = input[i] as usize;
                    i += 1;
                    match_length += value;
                    if value != 255 {
                        break;
                    }
                }
                src = i;
            }

            produced += match_length;
        }

        Ok(produced)
    }

    pub fn compress(&self, data: &[u8], block_size: usize) -> Result<LZ4CompressedFrame> {
        let mut blocks = Vec::new();
        let mut payload = Vec::new();
        payload.reserve(data.len() / 2);

        let mut input_offset = 0;
        let mut output_offset = 0;

        while input_offset < data.len() {
            let chunk_size = block_size.min(data.len() - input_offset);
            let chunk = &data[input_offset..input_offset + chunk_size];
            let compressed = self.compress_block(chunk)?;

            blocks.push(LZ4BlockDescriptor {
                compressed_size: compressed.len() as u32,
                uncompressed_size: chunk_size as u32,
                compressed_offset: payload.len() as u32,
                output_offset: output_offset as u32,
                is_compressed: true,
            });

            payload.extend_from_slice(&compressed);
            input_offset += chunk_size;
            output_offset += chunk_size;
        }

        let total_compressed_bytes = payload.len();
        Ok(LZ4CompressedFrame {
            uncompressed_size: data.len(),
            block_size,
            blocks,
            payload: Arc::from(payload),
            total_compressed_bytes,
            reported_content_size: Some(data.len()),
            uses_block_checksum: false,
        })
    }

    fn compress_block(&self, input: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::new();
        output.reserve(input.len() / 2);

        let hash_size = 1 << 16;
        let mut hash_table = vec![-1i32; hash_size];

        let mut anchor = 0;
        let mut index = 0;
        let match_limit = input.len().saturating_sub(LZ4_MIN_MATCH);

        while index <= match_limit {
            if index + LZ4_MIN_MATCH > input.len() {
                break;
            }

            let seq = self.read_u32_at(input, index);
            let h = ((seq.wrapping_mul(2654435761)) >> (32 - 16)) as usize;
            let candidate = hash_table[h] as usize;
            hash_table[h] = index as i32;

            if candidate != usize::MAX
                && index.saturating_sub(candidate) <= LZ4_MAX_OFFSET
                && self.read_u32_at(input, candidate) == seq
            {
                let mut match_index = candidate + LZ4_MIN_MATCH;
                let mut current_index = index + LZ4_MIN_MATCH;

                while current_index < input.len()
                    && match_index < input.len()
                    && input[current_index] == input[match_index]
                {
                    current_index += 1;
                    match_index += 1;
                }

                let literal_length = index - anchor;
                let mut token = (literal_length.min(15) as u8) << 4;
                let match_length = current_index - index;
                let match_run = match_length.saturating_sub(LZ4_MIN_MATCH);

                if match_run < 15 {
                    token |= match_run as u8;
                } else {
                    token |= 0x0F;
                }

                output.push(token);

                // Write literal length
                if literal_length >= 15 {
                    self.write_length(&mut output, literal_length - 15)?;
                }

                // Write literals
                output.extend_from_slice(&input[anchor..index]);

                // Write match offset
                let offset = index - candidate;
                output.push(offset as u8);
                output.push((offset >> 8) as u8);

                // Write match length
                if match_run >= 15 {
                    self.write_length(&mut output, match_run - 15)?;
                }

                anchor = current_index;
                index = current_index;
            } else {
                index += 1;
            }
        }

        // Handle remaining literals
        if anchor < input.len() {
            let literal_length = input.len() - anchor;
            let token = (literal_length.min(15) as u8) << 4;
            output.push(token);

            if literal_length >= 15 {
                self.write_length(&mut output, literal_length - 15)?;
            }

            output.extend_from_slice(&input[anchor..]);
        }

        Ok(output)
    }

    fn read_u32_at(&self, data: &[u8], offset: usize) -> u32 {
        if offset + 4 <= data.len() {
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        } else {
            let mut bytes = [0u8; 4];
            let copy_len = (data.len() - offset).min(4);
            bytes[..copy_len].copy_from_slice(&data[offset..offset + copy_len]);
            u32::from_le_bytes(bytes)
        }
    }

    fn write_length(&self, output: &mut Vec<u8>, mut length: usize) -> Result<()> {
        while length >= 255 {
            output.push(255);
            length -= 255;
        }
        if length > 0 {
            output.push(length as u8);
        }
        Ok(())
    }

    pub fn compress_to_frame(&self, data: &[u8], block_size: usize) -> Result<Vec<u8>> {
        let frame = self.compress(data, block_size)?;

        let mut output = Vec::new();

        // LZ4 Frame Magic Number
        output.extend_from_slice(&[0x04, 0x22, 0x4D, 0x18]);

        // Frame Descriptor
        let mut descriptor = 0x40u8; // Version 01, Block Independence
        if frame.uses_block_checksum {
            descriptor |= 0x10;
        }
        if frame.reported_content_size.is_some() {
            descriptor |= 0x08;
        }
        output.push(descriptor);

        // Content Size (if present)
        if let Some(content_size) = frame.reported_content_size {
            output.extend_from_slice(&content_size.to_le_bytes());
        }

        // Blocks
        for block in &frame.blocks {
            let block_size = if block.is_compressed {
                block.compressed_size
            } else {
                block.uncompressed_size | 0x80000000
            };

            output.extend_from_slice(&block_size.to_le_bytes());
            output.extend_from_slice(
                &frame.payload[block.compressed_offset as usize..]
                    [..block.compressed_size as usize],
            );
        }

        // End Mark
        output.extend_from_slice(&[0, 0, 0, 0]);

        Ok(output)
    }
}
