use anyhow::Result;
use lz4_flex::block::decompress_into;
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
        decompress_into(input, output)
            .map_err(|e| anyhow::anyhow!("lz4_flex decompression failed: {}", e))?;
        Ok(())
    }

    pub fn measure_decompressed_size(input: &[u8], max_uncompressed_size: usize) -> Result<usize> {
        let mut scratch = vec![0u8; max_uncompressed_size];
        let decompressed_len = decompress_into(input, &mut scratch)
            .map_err(|e| anyhow::anyhow!("lz4_flex decompression failed: {}", e))?;

        Ok(decompressed_len)
    }
}
