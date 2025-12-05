use crate::lz4::{LZ4BlockDescriptor, LZ4CompressedFrame, LZ4Error};
use anyhow::{Context, Result};
use lz4_flex::block::decompress_into;
use xxhash_rust::xxh32::{xxh32, Xxh32};

const FRAME_MAGIC: u32 = 0x184D2204;
const SKIPPABLE_MASK: u32 = 0xFFFFFFF0;
const SKIPPABLE_MAGIC: u32 = 0x184D2A50;

pub struct LZ4FrameParser;

impl LZ4FrameParser {
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

        Self::read_u8(data, &mut cursor)?; // Skip reserved byte

        let mut blocks = Vec::new();
        let mut payload = Vec::new();
        let mut output_offset = 0;
        let mut total_uncompressed = 0;
        let mut content_hasher = content_checksum_flag.then(|| Xxh32::new(0));

        loop {
            let block_header = Self::read_u32(data, &mut cursor)?;
            if block_header == 0 {
                break;
            }

            let is_compressed = (block_header & 0x80000000) == 0;
            let stored_size = (block_header & 0x7FFFFFFF) as usize;

            if cursor + stored_size > data.len() {
                return Err(LZ4Error::MalformedStream.into());
            }

            let block_start = cursor;
            let block_data = &data[block_start..block_start + stored_size];

            // Validate compressed and uncompressed checksums as we go to avoid silent corruption.
            let uncompressed_size = if is_compressed {
                let mut scratch = vec![0u8; block_size];
                let decompressed_len = decompress_into(block_data, &mut scratch).map_err(|e| {
                    anyhow::anyhow!("lz4_flex decompression failed while parsing: {}", e)
                })?;

                if let Some(hasher) = content_hasher.as_mut() {
                    hasher.update(&scratch[..decompressed_len]);
                }

                decompressed_len
            } else {
                if let Some(hasher) = content_hasher.as_mut() {
                    hasher.update(block_data);
                }
                stored_size
            };

            if uncompressed_size > block_size {
                return Err(LZ4Error::MalformedStream.into());
            }

            let descriptor = LZ4BlockDescriptor {
                compressed_size: stored_size as u32,
                uncompressed_size: uncompressed_size as u32,
                compressed_offset: payload.len() as u32,
                output_offset: output_offset as u32,
                is_compressed,
            };

            blocks.push(descriptor);
            payload.extend_from_slice(block_data);
            cursor += stored_size;
            output_offset += uncompressed_size;
            total_uncompressed += uncompressed_size;

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

        // Magic
        frame.extend_from_slice(&FRAME_MAGIC.to_le_bytes());

        // FLG and BD (64KB block size)
        frame.push(flg);
        frame.push(0x40);

        // Header checksum placeholder (parser currently skips)
        frame.push(0);

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
}
