use crate::lz4::{LZ4BlockDescriptor, LZ4CompressedFrame, LZ4Error};
use anyhow::{Context, Result};
use lz4_flex::block::decompress_into;
use std::io::Read;
use xxhash_rust::xxh32::{xxh32, Xxh32};

const FRAME_MAGIC: u32 = 0x184D2204;
const SKIPPABLE_MASK: u32 = 0xFFFFFFF0;
const SKIPPABLE_MAGIC: u32 = 0x184D2A50;
const MAX_BLOCKS: usize = 1_000_000; // Prevent runaway allocation on malformed inputs

pub struct LZ4FrameParser;

impl LZ4FrameParser {
    /// Measure uncompressed size of a compressed block without emitting bytes.
    pub fn measure_block_size(block_data: &[u8], block_max: usize) -> Result<usize> {
        let mut src = 0usize;
        let src_end = block_data.len();
        let mut dst: usize = 0;

        while src < src_end {
            let token = block_data[src];
            src += 1;

            let mut literal_len = (token >> 4) as usize;
            if literal_len == 15 {
                loop {
                    if src >= src_end {
                        return Err(LZ4Error::MalformedStream.into());
                    }
                    let b = block_data[src] as usize;
                    src += 1;
                    literal_len = literal_len
                        .checked_add(b)
                        .ok_or_else(|| anyhow::anyhow!("Literal length overflow"))?;
                    if b != 255 {
                        break;
                    }
                }
            }

            if src.checked_add(literal_len).filter(|v| *v <= src_end).is_none() {
                return Err(LZ4Error::MalformedStream.into());
            }
            dst = dst
                .checked_add(literal_len)
                .ok_or_else(|| anyhow::anyhow!("Literal output overflow"))?;
            if dst > block_max {
                return Err(LZ4Error::OutputOverflow.into());
            }
            src += literal_len;

            if src >= src_end {
                break;
            }

            if src + 2 > src_end {
                return Err(LZ4Error::MalformedStream.into());
            }
            let offset = (block_data[src] as usize) | ((block_data[src + 1] as usize) << 8);
            src += 2;
            if offset == 0 || offset > dst {
                return Err(LZ4Error::MalformedStream.into());
            }

            let mut match_len = (token & 0x0F) as usize + 4;
            if (token & 0x0F) == 0x0F {
                loop {
                    if src >= src_end {
                        return Err(LZ4Error::MalformedStream.into());
                    }
                    let b = block_data[src] as usize;
                    src += 1;
                    match_len = match_len
                        .checked_add(b)
                        .ok_or_else(|| anyhow::anyhow!("Match length overflow"))?;
                    if b != 255 {
                        break;
                    }
                }
            }

            dst = dst
                .checked_add(match_len)
                .ok_or_else(|| anyhow::anyhow!("Match output overflow"))?;
            if dst > block_max {
                return Err(LZ4Error::OutputOverflow.into());
            }
        }

        Ok(dst)
    }

    pub fn parse_file(path: &str) -> Result<ParsedFrame> {
        let data = std::fs::read(path).with_context(|| format!("Failed to read file: {}", path))?;

        let frame = Self::parse(&data)?;
        let file_size = data.len();
        let file_path = std::path::Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(ParsedFrame {
            frame,
            file_size,
            file_path,
        })
    }

    /// Parse file using direct I/O optimized for unified memory systems
    /// This path mirrors `parse_file` but is kept for compatibility with callers that
    /// expect a separate entry point. It currently performs a straightforward read.
    pub fn parse_file_direct_io(path: &str) -> Result<ParsedFrame> {
        let data = std::fs::read(path)
            .with_context(|| format!("Failed to read file: {}", path))?;

        let frame = Self::parse(&data)?;
        let file_size = data.len();
        let file_path = std::path::Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(ParsedFrame {
            frame,
            file_size,
            file_path,
        })
    }

    pub fn parse(data: &[u8]) -> Result<LZ4CompressedFrame> {
        let mut cursor = 0;

        let magic = Self::read_u32(data, &mut cursor)?;

        if (magic & SKIPPABLE_MASK) == SKIPPABLE_MAGIC {
            return Err(LZ4Error::UnsupportedFrame(
                "Skippable frames are not supported".to_string(),
            )
            .into());
        }

        if magic != FRAME_MAGIC {
            return Err(
                LZ4Error::UnsupportedFrame(format!("Unexpected magic {:08X}", magic)).into(),
            );
        }

        let header_start = cursor;

        // Read FLG and BD
        let flg = Self::read_u8(data, &mut cursor)?;
        let bd = Self::read_u8(data, &mut cursor)?;

        let version = (flg >> 6) & 0x03;
        if version != 1 {
            return Err(LZ4Error::UnsupportedFrame(format!(
                "Unsupported frame version {}",
                version
            ))
            .into());
        }

        if (flg & 0x01) != 0 {
            return Err(LZ4Error::UnsupportedFrame("Reserved FLG bit is set".to_string()).into());
        }

        let block_independence = ((flg >> 5) & 0x01) == 0x01;
        if !block_independence {
            return Err(LZ4Error::UnsupportedFrame(
                "Dependent blocks are not supported".to_string(),
            )
            .into());
        }

        let block_checksum_flag = ((flg >> 4) & 0x01) == 0x01;
        let content_size_flag = ((flg >> 3) & 0x01) == 0x01;
        let content_checksum_flag = ((flg >> 2) & 0x01) == 0x01;
        let dictionary_id_flag = ((flg >> 1) & 0x01) == 0x01;

        let block_max_id = (bd >> 4) & 0x07;
        let block_size = match block_max_id {
            4 => 64 * 1024,
            5 => 256 * 1024,
            6 => 1 * 1024 * 1024,
            7 => 4 * 1024 * 1024,
            _ => {
                return Err(LZ4Error::UnsupportedFrame(format!(
                    "Unsupported block size ID {}",
                    block_max_id
                ))
                .into())
            }
        };

        let reported_content_size = if content_size_flag {
            Some(Self::read_u64(data, &mut cursor)? as usize)
        } else {
            None
        };

        if dictionary_id_flag {
            Self::read_u32(data, &mut cursor)?; // Skip dictionary ID
        }

        // Validate header checksum (covers everything from FLG through last header field)
        let header_end = cursor;
        let stored_header_checksum = Self::read_u8(data, &mut cursor)?;
        let computed_header_checksum =
            Self::compute_header_checksum(&data[header_start..header_end]);
        if stored_header_checksum != computed_header_checksum {
            return Err(LZ4Error::ChecksumMismatch(format!(
                "Header checksum mismatch (expected {:02X}, got {:02X})",
                computed_header_checksum, stored_header_checksum
            ))
            .into());
        }

        let mut blocks = Vec::new();
        let mut payload = Vec::new();
        let mut output_offset: u64 = 0;
        let mut total_uncompressed: usize = 0;
        let mut content_hasher = content_checksum_flag.then(|| Xxh32::new(0));

        let max_blocks_from_content = reported_content_size.map(|content_size| {
            // Round up to the number of full blocks needed for the declared size.
            ((content_size + (block_size - 1)) / block_size).max(1)
        });

        loop {
            let block_header = Self::read_u32(data, &mut cursor)?;
            if block_header == 0 {
                break;
            }

            let is_compressed = (block_header & 0x80000000) == 0;
            let stored_size = (block_header & 0x7FFFFFFF) as usize;

            if stored_size == 0 {
                return Err(LZ4Error::MalformedStream.into());
            }

            if stored_size > block_size {
                return Err(LZ4Error::UnsupportedFrame(
                    "Block size exceeds negotiated maximum".to_string(),
                )
                .into());
            }

            if cursor + stored_size > data.len() {
                return Err(LZ4Error::MalformedStream.into());
            }

            let block_start = cursor;
            let block_data = &data[block_start..block_start + stored_size];

            // Validate compressed and uncompressed checksums as we go to avoid silent corruption.
            let uncompressed_size = if is_compressed {
                if content_checksum_flag {
                    let mut scratch = vec![0u8; block_size];
                    let decompressed_len =
                        decompress_into(block_data, &mut scratch).map_err(|e| {
                            anyhow::anyhow!("lz4_flex decompression failed while parsing: {}", e)
                        })?;

                    if let Some(hasher) = content_hasher.as_mut() {
                        hasher.update(&scratch[..decompressed_len]);
                    }

                    decompressed_len
                } else {
                    Self::measure_block_size(block_data, block_size)?
                }
            } else {
                if let Some(hasher) = content_hasher.as_mut() {
                    hasher.update(block_data);
                }
                stored_size
            };

            if uncompressed_size > block_size {
                return Err(LZ4Error::MalformedStream.into());
            }

            let compressed_offset = payload.len();
            payload
                .len()
                .checked_add(stored_size)
                .ok_or_else(|| anyhow::anyhow!("Compressed payload size overflow"))?;
            let new_output_offset = output_offset
                .checked_add(uncompressed_size as u64)
                .ok_or_else(|| anyhow::anyhow!("Output size overflow"))?;

            let descriptor = LZ4BlockDescriptor {
                compressed_size: stored_size as u32,
                uncompressed_size: uncompressed_size as u32,
                compressed_offset: compressed_offset as u64,
                output_offset,
                is_compressed,
            };

            blocks.push(descriptor);
            payload.extend_from_slice(block_data);
            cursor += stored_size;
            output_offset = new_output_offset;
            total_uncompressed = total_uncompressed
                .checked_add(uncompressed_size)
                .ok_or_else(|| anyhow::anyhow!("Total uncompressed size overflow"))?;

            if let Some(max_blocks) = max_blocks_from_content {
                if blocks.len() > max_blocks {
                    return Err(LZ4Error::UnsupportedFrame(
                        "Block count exceeds declared content size".to_string(),
                    )
                    .into());
                }
            }

            if blocks.len() > MAX_BLOCKS {
                return Err(LZ4Error::UnsupportedFrame(format!(
                    "Too many blocks (>{})",
                    MAX_BLOCKS
                ))
                .into());
            }

            if block_checksum_flag {
                let expected = Self::read_u32(data, &mut cursor)?;
                let actual = xxh32(block_data, 0);
                if expected != actual {
                    return Err(LZ4Error::ChecksumMismatch(format!(
                        "Block checksum mismatch at block {} (expected {:08X}, got {:08X})",
                        blocks.len() - 1,
                        expected,
                        actual
                    ))
                    .into());
                }
            }
        }

        let stored_content_checksum = if content_checksum_flag {
            Some(Self::read_u32(data, &mut cursor)?)
        } else {
            None
        };

        if let (Some(expected), Some(hasher)) = (stored_content_checksum, content_hasher) {
            let actual = hasher.digest();
            if expected != actual {
                return Err(LZ4Error::ChecksumMismatch(format!(
                    "Content checksum mismatch (expected {:08X}, got {:08X})",
                    expected, actual
                ))
                .into());
            }
        }

        if let Some(reported) = reported_content_size {
            if reported != total_uncompressed {
                return Err(LZ4Error::UnsupportedFrame(format!(
                    "Content size mismatch: header {} vs blocks {}",
                    reported, total_uncompressed
                ))
                .into());
            }
        }

        Ok(LZ4CompressedFrame {
            uncompressed_size: total_uncompressed,
            block_size,
            blocks,
            payload: payload.into(),
            total_compressed_bytes: data.len(),
            reported_content_size,
            uses_block_checksum: block_checksum_flag,
        })
    }

    pub fn compute_header_checksum(header_bytes: &[u8]) -> u8 {
        (xxh32(header_bytes, 0) >> 8) as u8
    }

    fn read_u8(data: &[u8], cursor: &mut usize) -> Result<u8> {
        if *cursor >= data.len() {
            return Err(LZ4Error::MalformedStream.into());
        }
        let value = data[*cursor];
        *cursor += 1;
        Ok(value)
    }

    fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32> {
        if *cursor + 4 > data.len() {
            return Err(LZ4Error::MalformedStream.into());
        }
        let value = u32::from_le_bytes([
            data[*cursor],
            data[*cursor + 1],
            data[*cursor + 2],
            data[*cursor + 3],
        ]);
        *cursor += 4;
        Ok(value)
    }

    fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64> {
        if *cursor + 8 > data.len() {
            return Err(LZ4Error::MalformedStream.into());
        }
        let value = u64::from_le_bytes([
            data[*cursor],
            data[*cursor + 1],
            data[*cursor + 2],
            data[*cursor + 3],
            data[*cursor + 4],
            data[*cursor + 5],
            data[*cursor + 6],
            data[*cursor + 7],
        ]);
        *cursor += 8;
        Ok(value)
    }
}

#[derive(Debug)]
pub struct ParsedFrame {
    pub frame: LZ4CompressedFrame,
    pub file_size: usize,
    pub file_path: String,
}

/// Streaming LZ4 frame reader that yields batches of independent blocks without
/// holding the entire compressed payload in memory.
pub struct LZ4FrameStream<R: std::io::Read> {
    reader: std::io::BufReader<R>,
    block_size: usize,
    block_checksum_flag: bool,
    content_checksum_flag: bool,
    reported_content_size: Option<usize>,
    max_batch_blocks: usize,
    content_hasher: Option<Xxh32>,
    scratch: Vec<u8>,
    blocks_seen: usize,
    total_uncompressed: usize,
    done: bool,
}

impl LZ4FrameStream<std::fs::File> {
    /// Create a streaming reader from a file path.
    pub fn from_file(path: &str, max_batch_blocks: usize) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open LZ4 file for streaming: {}", path))?;
        Self::new(file, max_batch_blocks)
    }
}

impl<R: std::io::Read> LZ4FrameStream<R> {
    pub fn new(reader: R, max_batch_blocks: usize) -> Result<Self> {
        let mut reader = std::io::BufReader::new(reader);
        let mut header_bytes = Vec::new();

        let magic = Self::read_u32_reader(&mut reader)?;
        if (magic & SKIPPABLE_MASK) == SKIPPABLE_MAGIC {
            return Err(LZ4Error::UnsupportedFrame(
                "Skippable frames are not supported".to_string(),
            )
            .into());
        }
        if magic != FRAME_MAGIC {
            return Err(
                LZ4Error::UnsupportedFrame(format!("Unexpected magic {:08X}", magic)).into(),
            );
        }

        let flg = Self::read_u8_reader(&mut reader)?;
        let bd = Self::read_u8_reader(&mut reader)?;
        header_bytes.push(flg);
        header_bytes.push(bd);

        let version = (flg >> 6) & 0x03;
        if version != 1 {
            return Err(LZ4Error::UnsupportedFrame(format!(
                "Unsupported frame version {}",
                version
            ))
            .into());
        }
        if (flg & 0x01) != 0 {
            return Err(LZ4Error::UnsupportedFrame("Reserved FLG bit is set".to_string()).into());
        }

        let block_independence = ((flg >> 5) & 0x01) == 0x01;
        if !block_independence {
            return Err(LZ4Error::UnsupportedFrame(
                "Dependent blocks are not supported".to_string(),
            )
            .into());
        }

        let block_checksum_flag = ((flg >> 4) & 0x01) == 0x01;
        let content_size_flag = ((flg >> 3) & 0x01) == 0x01;
        let content_checksum_flag = ((flg >> 2) & 0x01) == 0x01;
        let dictionary_id_flag = ((flg >> 1) & 0x01) == 0x01;

        let block_max_id = (bd >> 4) & 0x07;
        let block_size = match block_max_id {
            4 => 64 * 1024,
            5 => 256 * 1024,
            6 => 1 * 1024 * 1024,
            7 => 4 * 1024 * 1024,
            _ => {
                return Err(LZ4Error::UnsupportedFrame(format!(
                    "Unsupported block size ID {}",
                    block_max_id
                ))
                .into())
            }
        };

        let reported_content_size = if content_size_flag {
            let val = Self::read_u64_reader(&mut reader)? as usize;
            header_bytes.extend_from_slice(&(val as u64).to_le_bytes());
            Some(val)
        } else {
            None
        };

        if dictionary_id_flag {
            let id = Self::read_u32_reader(&mut reader)?;
            header_bytes.extend_from_slice(&id.to_le_bytes());
        }

        let stored_header_checksum = Self::read_u8_reader(&mut reader)?;
        let computed_header_checksum =
            LZ4FrameParser::compute_header_checksum(&header_bytes);
        if stored_header_checksum != computed_header_checksum {
            return Err(LZ4Error::ChecksumMismatch(format!(
                "Header checksum mismatch (expected {:02X}, got {:02X})",
                computed_header_checksum, stored_header_checksum
            ))
            .into());
        }

        Ok(Self {
            reader,
            block_size,
            block_checksum_flag,
            content_checksum_flag,
            reported_content_size,
            max_batch_blocks: max_batch_blocks.max(1),
            content_hasher: content_checksum_flag.then(|| Xxh32::new(0)),
            scratch: vec![0u8; block_size],
            blocks_seen: 0,
            total_uncompressed: 0,
            done: false,
        })
    }

    /// Return the next batch of blocks as a standalone frame, or None when the
    /// stream is exhausted.
    pub fn next_batch(&mut self) -> Result<Option<LZ4CompressedFrame>> {
        if self.done {
            return Ok(None);
        }

        let mut blocks = Vec::with_capacity(self.max_batch_blocks);
        let mut payload = Vec::with_capacity(self.block_size * self.max_batch_blocks);
        let mut batch_uncompressed: usize = 0;

        loop {
            let header = Self::read_u32_reader(&mut self.reader)?;
            if header == 0 {
                self.finish_checks()?;
                self.done = true;
                break;
            }

            let (block_data, is_compressed) = self.read_block(header)?;
            let uncompressed_size = self.validate_and_size_block(&block_data, is_compressed)?;

            self.append_block(
                &mut blocks,
                &mut payload,
                &mut batch_uncompressed,
                block_data,
                uncompressed_size,
                is_compressed,
            )?;

            if blocks.len() >= self.max_batch_blocks {
                break;
            }
        }

        if blocks.is_empty() {
            return Ok(None);
        }

        let payload_len = payload.len();

        Ok(Some(LZ4CompressedFrame {
            uncompressed_size: batch_uncompressed,
            block_size: self.block_size,
            blocks,
            payload: payload.into(),
            total_compressed_bytes: payload_len,
            reported_content_size: Some(batch_uncompressed),
            uses_block_checksum: self.block_checksum_flag,
        }))
    }

    pub fn reported_content_size(&self) -> Option<usize> {
        self.reported_content_size
    }

    pub fn total_uncompressed(&self) -> usize {
        self.total_uncompressed
    }

    /// Measure uncompressed size of a compressed block without emitting bytes.
    fn measure_block_size(block_data: &[u8], block_max: usize) -> Result<usize> {
        LZ4FrameParser::measure_block_size(block_data, block_max)
    }

    fn read_block(&mut self, block_header: u32) -> Result<(Vec<u8>, bool)> {
        let is_compressed = (block_header & 0x80000000) == 0;
        let stored_size = (block_header & 0x7FFF_FFFF) as usize;
        if stored_size == 0 {
            return Err(LZ4Error::MalformedStream.into());
        }
        if stored_size > self.block_size {
            return Err(LZ4Error::UnsupportedFrame(
                "Block size exceeds negotiated maximum".to_string(),
            )
            .into());
        }

        let mut block_data = vec![0u8; stored_size];
        self.reader
            .read_exact(&mut block_data)
            .context("Failed to read block payload")?;

        if self.block_checksum_flag {
            let expected = Self::read_u32_reader(&mut self.reader)?;
            let actual = xxh32(&block_data, 0);
            if expected != actual {
                return Err(LZ4Error::ChecksumMismatch(format!(
                    "Block checksum mismatch (expected {:08X}, got {:08X})",
                    expected, actual
                ))
                .into());
            }
        }

        Ok((block_data, is_compressed))
    }

    fn validate_and_size_block(
        &mut self,
        block_data: &[u8],
        is_compressed: bool,
    ) -> Result<usize> {
        let size = if is_compressed {
            if self.content_checksum_flag {
                let len = decompress_into(block_data, &mut self.scratch).map_err(|e| {
                    anyhow::anyhow!("lz4_flex decompression failed while sizing block: {}", e)
                })?;
                if len > self.block_size {
                    return Err(LZ4Error::OutputOverflow.into());
                }
                if let Some(hasher) = self.content_hasher.as_mut() {
                    hasher.update(&self.scratch[..len]);
                }
                len
            } else {
                Self::measure_block_size(block_data, self.block_size)?
            }
        } else {
            if let Some(hasher) = self.content_hasher.as_mut() {
                hasher.update(block_data);
            }
            block_data.len()
        };

        Ok(size)
    }

    fn append_block(
        &mut self,
        blocks: &mut Vec<LZ4BlockDescriptor>,
        payload: &mut Vec<u8>,
        batch_uncompressed: &mut usize,
        block_data: Vec<u8>,
        uncompressed_size: usize,
        is_compressed: bool,
    ) -> Result<()> {
        let stored_size = block_data.len();
        let compressed_offset = payload.len();
        payload
            .len()
            .checked_add(stored_size)
            .ok_or_else(|| anyhow::anyhow!("Compressed payload size overflow"))?;
        let new_batch_uncompressed = batch_uncompressed
            .checked_add(uncompressed_size)
            .ok_or_else(|| anyhow::anyhow!("Output size overflow"))?;

        let descriptor = LZ4BlockDescriptor {
            compressed_size: stored_size as u32,
            uncompressed_size: uncompressed_size as u32,
            compressed_offset: compressed_offset as u64,
            output_offset: *batch_uncompressed as u64,
            is_compressed,
        };

        payload.extend_from_slice(&block_data);

        blocks.push(descriptor);
        *batch_uncompressed = new_batch_uncompressed;
        self.blocks_seen += 1;
        self.total_uncompressed = self
            .total_uncompressed
            .checked_add(uncompressed_size)
            .ok_or_else(|| anyhow::anyhow!("Total uncompressed size overflow"))?;

        if self.blocks_seen > MAX_BLOCKS {
            return Err(LZ4Error::UnsupportedFrame(format!(
                "Too many blocks (>{})",
                MAX_BLOCKS
            ))
            .into());
        }

        Ok(())
    }

    fn finish_checks(&mut self) -> Result<()> {
        if let Some(expected_checksum) = self.content_checksum_flag.then(|| {
            Self::read_u32_reader(&mut self.reader)
        }) {
            let expected_checksum = expected_checksum?;
            if let Some(hasher) = self.content_hasher.take() {
                let actual = hasher.digest();
                if actual != expected_checksum {
                    return Err(LZ4Error::ChecksumMismatch(format!(
                        "Content checksum mismatch (expected {:08X}, got {:08X})",
                        expected_checksum, actual
                    ))
                    .into());
                }
            }
        }

        if let Some(expected) = self.reported_content_size {
            if expected != self.total_uncompressed {
                return Err(LZ4Error::UnsupportedFrame(format!(
                    "Content size mismatch: header {} vs decoded {}",
                    expected, self.total_uncompressed
                ))
                .into());
            }
        }

        Ok(())
    }

    fn read_u8_reader(reader: &mut std::io::BufReader<R>) -> Result<u8> {
        let mut buf = [0u8; 1];
        reader
            .read_exact(&mut buf)
            .context("Failed to read u8 from stream")?;
        Ok(buf[0])
    }

    fn read_u32_reader(reader: &mut std::io::BufReader<R>) -> Result<u32> {
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .context("Failed to read u32 from stream")?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64_reader(reader: &mut std::io::BufReader<R>) -> Result<u64> {
        let mut buf = [0u8; 8];
        reader
            .read_exact(&mut buf)
            .context("Failed to read u64 from stream")?;
        Ok(u64::from_le_bytes(buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_frame(
        flg: u8,
        block_data: &[u8],
        block_checksum: Option<u32>,
        content_checksum: Option<u32>,
    ) -> Vec<u8> {
        let mut frame = Vec::new();
        let mut header_bytes = Vec::new();

        // Magic
        frame.extend_from_slice(&FRAME_MAGIC.to_le_bytes());

        // FLG and BD (64KB block size)
        header_bytes.push(flg);
        header_bytes.push(0x40);
        frame.push(flg);
        frame.push(0x40);

        // Header checksum
        let header_checksum = LZ4FrameParser::compute_header_checksum(&header_bytes);
        frame.push(header_checksum);

        // Block header (uncompressed block)
        let block_header = 0x8000_0000u32 | (block_data.len() as u32);
        frame.extend_from_slice(&block_header.to_le_bytes());

        // Block payload
        frame.extend_from_slice(block_data);

        // Optional block checksum
        if let Some(cs) = block_checksum {
            frame.extend_from_slice(&cs.to_le_bytes());
        }

        // End marker
        frame.extend_from_slice(&0u32.to_le_bytes());

        // Optional content checksum
        if let Some(cs) = content_checksum {
            frame.extend_from_slice(&cs.to_le_bytes());
        }

        frame
    }

    #[test]
    fn block_checksum_mismatch_is_rejected() {
        let data = b"hello";
        let flg = 0x70; // version=1, block independence, block checksum
        let bad_block_checksum = xxh32(data, 0) ^ 0xFFFF;
        let frame = build_test_frame(flg, data, Some(bad_block_checksum), None);

        let err = LZ4FrameParser::parse(&frame).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Block checksum mismatch"));
    }

    #[test]
    fn content_checksum_mismatch_is_rejected() {
        let data = b"world";
        let flg = 0x64; // version=1, block independence, content checksum
        let bad_content_checksum = xxh32(data, 0) ^ 0x1234;
        let frame = build_test_frame(flg, data, None, Some(bad_content_checksum));

        let err = LZ4FrameParser::parse(&frame).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Content checksum mismatch"));
    }

    #[test]
    fn header_checksum_mismatch_is_rejected() {
        let flg = 0x60; // version=1, block independence

        let mut frame = Vec::new();
        frame.extend_from_slice(&FRAME_MAGIC.to_le_bytes());
        frame.push(flg);
        frame.push(0x40);
        // Wrong header checksum
        frame.push(0xFF);

        // End marker (no blocks)
        frame.extend_from_slice(&0u32.to_le_bytes());

        let err = LZ4FrameParser::parse(&frame).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Header checksum mismatch"));
    }

    #[test]
    fn streaming_reader_yields_batches() -> Result<()> {
        let flg = 0x60; // version=1, block independence
        let payload = b"stream me!";
        let frame_bytes = build_test_frame(flg, payload, None, None);

        let mut stream = LZ4FrameStream::new(std::io::Cursor::new(frame_bytes), 2)?;
        let mut batches = Vec::new();
        while let Some(batch) = stream.next_batch()? {
            batches.push(batch);
        }

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].blocks.len(), 1);

        let cpu = crate::lz4::LZ4Decompressor::new();
        let decoded = cpu.decompress(&batches[0], Some(1))?;
        assert_eq!(decoded, payload);

        Ok(())
    }
}
