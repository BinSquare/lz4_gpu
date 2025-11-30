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

const LZ4_MIN_MATCH: usize = 4;

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
        // Pre-allocate output buffer once
        let mut output = vec![0u8; frame.uncompressed_size];

        // Process blocks sequentially with direct buffer access
        for block in &frame.blocks {
            let block_input = &frame.payload[block.compressed_offset as usize..]
                [..block.compressed_size as usize];
            let block_output =
                &mut output[block.output_offset as usize..][..block.uncompressed_size as usize];

            if block.is_compressed {
                self.decompress_block(block_input, block_output)?;
            } else {
                // Direct copy for uncompressed blocks
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

        // Process blocks in parallel and collect individual results
        // This avoids the borrowing issue by creating separate output buffers
        let block_results: Result<Vec<(usize, Vec<u8>)>, _> = frame
            .blocks
            .par_iter()
            .enumerate()
            .map(|(block_idx, block)| {
                let block_input =
                    &payload[block.compressed_offset as usize..][..block.compressed_size as usize];

                let block_output = if block.is_compressed {
                    let mut output = vec![0u8; block.uncompressed_size as usize];
                    self.decompress_block(block_input, &mut output)
                        .map_err(|e| anyhow::anyhow!("Decompression failed: {}", e))?;
                    output
                } else {
                    // Direct copy for uncompressed blocks
                    block_input.to_vec()
                };

                Ok::<(usize, Vec<u8>), anyhow::Error>((block_idx, block_output))
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

    #[inline(always)]
    fn decompress_block(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let mut src = 0;
        let mut dst = 0;
        let input_len = input.len();
        let output_len = output.len();

        // Pre-validate that we have enough space in both buffers
        if input_len == 0 || output_len == 0 {
            return Ok(());
        }

        // Prefetch the beginning of input and output for better cache performance
        // Using standard Rust prefetch hints when available
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                input.as_ptr() as *const i8,
                std::arch::x86_64::_MM_HINT_T0,
            );
            std::arch::x86_64::_mm_prefetch(
                output.as_ptr() as *const i8,
                std::arch::x86_64::_MM_HINT_T0,
            );
        }

        while src < input_len && dst < output_len {
            // Fast path: Unrolled literal and match processing
            if src + 5 <= input_len && dst + 16 <= output_len {
                let token = input[src];
                src += 1;

                // Process literals with fast path optimization
                let literal_length_nibble = (token >> 4) as usize;
                if literal_length_nibble < 15 {
                    // Fast path for short literals
                    let literal_end = src + literal_length_nibble;
                    if literal_end <= input_len {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                input.as_ptr().add(src),
                                output.as_mut_ptr().add(dst),
                                literal_length_nibble,
                            );
                        }
                        src = literal_end;
                        dst += literal_length_nibble;
                    } else {
                        return Err(LZ4Error::MalformedStream.into());
                    }
                } else {
                    // Extended literal length
                    let mut literal_length = literal_length_nibble;
                    while src < input_len {
                        let byte = input[src];
                        src += 1;
                        literal_length += byte as usize;
                        if byte != 255 {
                            break;
                        }
                    }

                    // Bounds check for extended literals
                    if src + literal_length > input_len || dst + literal_length > output_len {
                        return Err(LZ4Error::MalformedStream.into());
                    }

                    // Fast copy for extended literals
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            input.as_ptr().add(src),
                            output.as_mut_ptr().add(dst),
                            literal_length,
                        );
                    }
                    src += literal_length;
                    dst += literal_length;
                }

                // Early exit if we're at end of input
                if src >= input_len {
                    break;
                }

                // Process matches with fast path optimization
                if src + 2 <= input_len {
                    let offset = input[src] as usize | ((input[src + 1] as usize) << 8);
                    src += 2;

                    // Validate offset early
                    if offset == 0 || dst < offset {
                        return Err(LZ4Error::MalformedStream.into());
                    }

                    // Process match length with fast path
                    let match_length_nibble = (token & 0x0F) as usize;
                    let mut match_length = match_length_nibble + LZ4_MIN_MATCH;
                    if match_length_nibble == 15 {
                        // Extended match length
                        while src < input_len {
                            let byte = input[src];
                            src += 1;
                            match_length += byte as usize;
                            if byte != 255 {
                                break;
                            }
                        }
                    }

                    // Bounds check for match
                    if dst + match_length > output_len {
                        return Err(LZ4Error::OutputOverflow.into());
                    }

                    // Fast match copy with overlap handling
                    let match_src = dst - offset;
                    self.copy_match_overlap(output, match_src, dst, match_length);
                    dst += match_length;
                } else {
                    return Err(LZ4Error::MalformedStream.into());
                }
            } else {
                // Slow path for edge cases (near buffer boundaries)
                if src >= input_len {
                    break;
                }

                let token = input[src];
                src += 1;

                // Decode literal length (high nibble) - simplified
                let mut literal_length = (token >> 4) as usize;
                if literal_length == 15 {
                    while src < input_len {
                        let byte = input[src];
                        src += 1;
                        literal_length += byte as usize;
                        if byte != 255 {
                            break;
                        }
                    }
                }

                // Copy literals with bounds checking
                if literal_length > 0 {
                    if src + literal_length > input_len || dst + literal_length > output_len {
                        return Err(LZ4Error::MalformedStream.into());
                    }
                    output[dst..dst + literal_length]
                        .copy_from_slice(&input[src..src + literal_length]);
                    src += literal_length;
                    dst += literal_length;
                }

                // Early exit
                if src >= input_len {
                    break;
                }

                // Decode match - simplified
                if src + 2 > input_len {
                    return Err(LZ4Error::MalformedStream.into());
                }

                let offset = input[src] as usize | ((input[src + 1] as usize) << 8);
                src += 2;

                // Validate offset
                if offset == 0 || dst < offset {
                    return Err(LZ4Error::MalformedStream.into());
                }

                // Decode match length (low nibble) - simplified
                let mut match_length = ((token & 0x0F) as usize) + LZ4_MIN_MATCH;
                if (token & 0x0F) == 0x0F {
                    while src < input_len {
                        let byte = input[src];
                        src += 1;
                        match_length += byte as usize;
                        if byte != 255 {
                            break;
                        }
                    }
                }

                // Check output bounds
                if dst + match_length > output_len {
                    return Err(LZ4Error::OutputOverflow.into());
                }

                // Copy match with overlap handling
                let match_src = dst - offset;
                self.copy_match_overlap(output, match_src, dst, match_length);
                dst += match_length;
            }
        }

        Ok(())
    }

    /// Copy match data handling overlap efficiently
    fn copy_match_overlap(&self, output: &mut [u8], src: usize, dst: usize, length: usize) {
        // Only use non-overlapping copy when it is actually non-overlapping.
        // LZ4 matches often overlap (self-referential), so we must be careful.
        if src + length <= dst {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    output.as_ptr().add(src),
                    output.as_mut_ptr().add(dst),
                    length,
                );
            }
        } else {
            // Overlap case - byte-by-byte copy to handle self-reference correctly
            for i in 0..length {
                output[dst + i] = output[src + i];
            }
        }
    }

    fn read_length_with_count(&self, input: &[u8]) -> Result<(usize, usize)> {
        let mut length = 0;
        let mut i = 0;

        while i < input.len() {
            let value = input[i] as usize;
            length += value;
            i += 1;
            if value != 255 {
                break;
            }
        }

        Ok((length, i))
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
}
