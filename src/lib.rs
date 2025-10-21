use anyhow::Result;

pub mod benchmark;
pub mod gpu;
pub mod lz4;
pub mod lz4_parser;
pub mod quantized;

// Re-export main types
pub use benchmark::{BenchmarkResult, CompressionDemo};
pub use gpu::{GPUDecompressor, GPUDevice};
pub use lz4::{LZ4BlockDescriptor, LZ4CompressedFrame, LZ4Decompressor};
pub use lz4_parser::{LZ4FrameParser, ParsedFrame};
pub use quantized::{QuantizedBlocks, QuantizedDecompressor};

/// Main decompressor that can use both CPU and GPU backends
pub struct Decompressor {
    cpu_decompressor: LZ4Decompressor,
    gpu_decompressor: Option<GPUDecompressor>,
}

impl Decompressor {
    pub fn new() -> Result<Self> {
        let cpu_decompressor = LZ4Decompressor::new();
        let gpu_decompressor = GPUDecompressor::new().ok();

        Ok(Self {
            cpu_decompressor,
            gpu_decompressor,
        })
    }

    pub fn decompress_cpu(
        &self,
        frame: &LZ4CompressedFrame,
        concurrency: Option<usize>,
    ) -> Result<Vec<u8>> {
        self.cpu_decompressor.decompress(frame, concurrency)
    }

    pub fn compress_cpu(&self, data: &[u8], block_size: usize) -> Result<LZ4CompressedFrame> {
        self.cpu_decompressor.compress(data, block_size)
    }

    pub fn compress_to_frame(&self, data: &[u8], block_size: usize) -> Result<Vec<u8>> {
        self.cpu_decompressor.compress_to_frame(data, block_size)
    }

    pub async fn decompress_gpu(&self, frame: &LZ4CompressedFrame) -> Result<Vec<u8>> {
        match &self.gpu_decompressor {
            Some(gpu) => gpu.decompress(frame).await,
            None => Err(anyhow::anyhow!("GPU decompressor not available")),
        }
    }

    pub fn has_gpu(&self) -> bool {
        self.gpu_decompressor.is_some()
    }
}
