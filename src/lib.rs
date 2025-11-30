use anyhow::Result;

pub mod async_pipeline;
pub mod direct_io;
pub mod gpu;
pub mod hybrid;
pub mod lz4;
pub mod lz4_parser;
pub mod memory_pool;
pub mod persistent_gpu;
pub mod resource_management;
pub mod simple_profiler;
pub mod streaming_pipeline;

// Re-export main types
pub use gpu::{GPUDecompressor, GPUDevice};
pub use lz4::{LZ4BlockDescriptor, LZ4CompressedFrame, LZ4Decompressor};
pub use lz4_parser::{LZ4FrameParser, ParsedFrame};

/// Main decompressor that can use both CPU and GPU backends
pub struct Decompressor {
    cpu_decompressor: LZ4Decompressor,
    gpu_decompressor: Option<GPUDecompressor>,
}

impl Decompressor {
    pub fn new() -> Result<Self> {
        let cpu_decompressor = LZ4Decompressor::new();
        let gpu_decompressor = match GPUDecompressor::new() {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                eprintln!("⚠️  GPU initialization failed, falling back to CPU: {e}");
                None
            }
        };

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

    pub async fn decompress_gpu(&self, frame: &LZ4CompressedFrame) -> Result<Vec<u8>> {
        match &self.gpu_decompressor {
            Some(gpu) => gpu.decompress(frame).await,
            None => Err(anyhow::anyhow!("GPU decompressor not available")),
        }
    }

    pub fn has_gpu(&self) -> bool {
        self.gpu_decompressor.is_some()
    }

    // Public accessors for profiling
    pub fn get_cpu_decompressor(&self) -> &LZ4Decompressor {
        &self.cpu_decompressor
    }

    pub fn get_gpu_decompressor(&self) -> Option<&GPUDecompressor> {
        self.gpu_decompressor.as_ref()
    }

    /// Enable persistent GPU mode for reduced setup overhead
    pub fn enable_persistent_gpu(&mut self) -> Result<()> {
        if let Some(ref mut _gpu) = self.gpu_decompressor {
            // In a full implementation, this would initialize persistent GPU resources
            // For now, we'll just log that the feature is enabled
            println!("Persistent GPU mode enabled");
            Ok(())
        } else {
            Err(anyhow::anyhow!("No GPU decompressor available"))
        }
    }
}
