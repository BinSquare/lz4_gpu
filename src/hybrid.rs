use crate::{Decompressor, LZ4CompressedFrame};
use anyhow::Result;
use std::time::Instant;

/// Hybrid decompression mode that automatically selects the best approach
/// based on file characteristics and system conditions.
pub struct HybridDecompressor {
    decompressor: Decompressor,
    gpu_available: bool,
}

impl HybridDecompressor {
    pub fn new() -> Result<Self> {
        let decompressor = Decompressor::new()?;
        let gpu_available = decompressor.has_gpu();

        Ok(Self {
            decompressor,
            gpu_available,
        })
    }

    /// Decompress using the hybrid approach that selects the best method
    pub async fn decompress_hybrid(&self, frame: &LZ4CompressedFrame) -> Result<Vec<u8>> {
        // Choose the best approach based on heuristics
        let strategy = self.choose_strategy(frame);

        match strategy {
            HybridStrategy::CpuOnly => {
                // Use CPU with optimal thread count
                let cpu_cores = num_cpus::get();
                self.decompressor.decompress_cpu(frame, Some(cpu_cores))
            }
            HybridStrategy::GpuOnly => {
                // Use GPU if available
                if self.gpu_available {
                    self.decompressor.decompress_gpu(frame).await
                } else {
                    // Fallback to CPU if GPU not available
                    let cpu_cores = num_cpus::get();
                    self.decompressor.decompress_cpu(frame, Some(cpu_cores))
                }
            }
        }
    }

    /// Choose the best strategy based on frame characteristics
    fn choose_strategy(&self, frame: &LZ4CompressedFrame) -> HybridStrategy {
        // Heuristic-based decision making

        // 1. File size threshold - GPU benefits more with larger files
        let total_size = frame.uncompressed_size;

        // 2. Block count - more blocks = better GPU parallelization
        let block_count = frame.blocks.len();

        // 3. Average block size - larger blocks benefit more from GPU
        let avg_block_size = if block_count > 0 {
            total_size / block_count
        } else {
            0
        };

        // Decision logic with debug information
        let strategy = if !self.gpu_available {
            // No GPU available - always use CPU
            HybridStrategy::CpuOnly
        } else if total_size < 1_000_000 {
            // Small files (< 1MB) - CPU is usually faster due to lower overhead
            HybridStrategy::CpuOnly
        } else if total_size > 50_000_000 && block_count > 10 {
            // Large files with many blocks - GPU can provide good parallelization
            HybridStrategy::GpuOnly
        } else if avg_block_size > 1_000_000 && block_count > 5 {
            // Large average block size with sufficient blocks for GPU parallelization
            HybridStrategy::GpuOnly
        } else if block_count > 20 {
            // Many blocks - GPU parallelization can help even with moderate file sizes
            HybridStrategy::GpuOnly
        } else {
            // Moderate files - use CPU for predictable performance
            HybridStrategy::CpuOnly
        };

        // Log the decision for debugging (in production, this could be controlled by verbosity flag)
        #[cfg(debug_assertions)]
        {
            println!("Hybrid strategy decision:");
            println!("  File size: {} bytes", total_size);
            println!("  Block count: {}", block_count);
            println!("  Avg block size: {} bytes", avg_block_size);
            println!("  GPU available: {}", self.gpu_available);
            println!("  Chosen strategy: {:?}", strategy);
        }

        strategy
    }

    /// Benchmark and compare CPU vs GPU performance for adaptive optimization
    pub async fn benchmark_comparison(
        &self,
        frame: &LZ4CompressedFrame,
    ) -> Result<BenchmarkResult> {
        let mut result = BenchmarkResult::default();

        // CPU benchmark
        let cpu_start = Instant::now();
        let cpu_result = self
            .decompressor
            .decompress_cpu(frame, Some(num_cpus::get()))?;
        let cpu_duration = cpu_start.elapsed();

        result.cpu_time = cpu_duration;
        result.cpu_result_size = cpu_result.len();

        // GPU benchmark (if available)
        if self.gpu_available {
            let gpu_start = Instant::now();
            let gpu_result = self.decompressor.decompress_gpu(frame).await?;
            let gpu_duration = gpu_start.elapsed();

            result.gpu_time = Some(gpu_duration);
            result.gpu_result_size = Some(gpu_result.len());

            // Determine which was faster
            if gpu_duration < cpu_duration {
                result.fastest_method = FastestMethod::Gpu;
            } else {
                result.fastest_method = FastestMethod::Cpu;
            }
        } else {
            result.fastest_method = FastestMethod::Cpu;
        }

        Ok(result)
    }
}

/// Strategy for hybrid decompression
#[derive(Debug, Clone, Copy, PartialEq)]
enum HybridStrategy {
    CpuOnly,
    GpuOnly,
}

/// Benchmark results for performance comparison
#[derive(Debug)]
pub struct BenchmarkResult {
    pub cpu_time: std::time::Duration,
    pub cpu_result_size: usize,
    pub gpu_time: Option<std::time::Duration>,
    pub gpu_result_size: Option<usize>,
    pub fastest_method: FastestMethod,
}

impl Default for BenchmarkResult {
    fn default() -> Self {
        Self {
            cpu_time: std::time::Duration::from_millis(0),
            cpu_result_size: 0,
            gpu_time: None,
            gpu_result_size: None,
            fastest_method: FastestMethod::Cpu,
        }
    }
}

/// Which method was fastest
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FastestMethod {
    Cpu,
    Gpu,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_decompressor_creation() -> Result<()> {
        let hybrid = HybridDecompressor::new()?;
        // Just test that it can be created
        assert_eq!(hybrid.gpu_available, hybrid.decompressor.has_gpu());
        Ok(())
    }

    #[test]
    fn test_benchmark_result_default() {
        let result = BenchmarkResult::default();
        assert_eq!(result.cpu_result_size, 0);
        assert!(result.gpu_time.is_none());
        assert!(result.gpu_result_size.is_none());
    }
}
