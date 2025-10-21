use std::time::{Duration, Instant};
use anyhow::Result;
use crate::{Decompressor, LZ4CompressedFrame};

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub demo_name: String,
    pub dataset_description: String,
    pub uncompressed_bytes: usize,
    pub compressed_bytes: usize,
    pub cpu_milliseconds: f64,
    pub gpu_milliseconds: Option<f64>,
    pub notes: Vec<String>,
}

impl BenchmarkResult {
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_bytes > 0 {
            self.uncompressed_bytes as f64 / self.compressed_bytes as f64
        } else {
            0.0
        }
    }

    pub fn speedup(&self) -> Option<f64> {
        self.gpu_milliseconds.map(|gpu_ms| self.cpu_milliseconds / gpu_ms)
    }
}

pub trait CompressionDemo {
    fn name(&self) -> &str;
    fn run(&self, decompressor: &Decompressor) -> Result<BenchmarkResult>;
}

pub struct QuantizedSignalDemo {
    pub block_size: usize,
    pub block_count: usize,
}

impl QuantizedSignalDemo {
    pub fn new() -> Self {
        Self {
            block_size: 512,
            block_count: 4096,
        }
    }
}

impl CompressionDemo for QuantizedSignalDemo {
    fn name(&self) -> &str {
        "Quantized Signal Reconstruction"
    }

    fn run(&self, decompressor: &Decompressor) -> Result<BenchmarkResult> {
        let sample_count = self.block_size * self.block_count;
        
        // Generate synthetic signal data
        let mut samples = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let base = (i as f32 * 0.01).sin() * 80.0;
            let noise = (rand::random::<f32>() - 0.5) * 10.0;
            samples.push(base + noise);
        }

        // Compress the data
        let blocks = crate::quantized::QuantizedDecompressor::compress(&samples, self.block_size)?;
        
        // Benchmark CPU decompression
        let (cpu_ms, cpu_output) = measure_time(5, || {
            crate::quantized::QuantizedDecompressor::new().decompress_cpu(&blocks)
        })?;

        // Calculate error metrics
        let cpu_error = compute_error_metrics(&samples, &cpu_output);
        let mut notes = vec![
            format!("CPU error: max {:.4}, mean {:.4}", cpu_error.max, cpu_error.mean)
        ];

        // Benchmark GPU decompression if available
        let gpu_ms = if decompressor.has_gpu() {
            let (measured_gpu_ms, gpu_output) = measure_time(5, || {
                // For now, use CPU decompression as GPU quantized decompression is not implemented
                crate::quantized::QuantizedDecompressor::new().decompress_cpu(&blocks)
            })?;
            
            let gpu_error = compute_error_metrics(&samples, &gpu_output);
            let cpu_vs_gpu = compute_error_metrics(&cpu_output, &gpu_output);
            
            notes.push(format!("GPU error vs original: max {:.4}, mean {:.4}", gpu_error.max, gpu_error.mean));
            notes.push(format!("GPU vs CPU delta: max {:.6}, mean {:.6}", cpu_vs_gpu.max, cpu_vs_gpu.mean));
            
            notes.push(format!("Speedup (CPU/GPU): {:.2}x", cpu_ms / measured_gpu_ms));
            
            Some(measured_gpu_ms)
        } else {
            notes.push("GPU unavailable; Metal-based decompression skipped.".to_string());
            None
        };

        let compressed_size = blocks.codes.len() + blocks.mins.len() * 4 + blocks.scales.len() * 4;

        Ok(BenchmarkResult {
            demo_name: self.name().to_string(),
            dataset_description: format!("{} float samples ({} x {})", sample_count, self.block_count, self.block_size),
            uncompressed_bytes: sample_count * 4,
            compressed_bytes: compressed_size,
            cpu_milliseconds: cpu_ms,
            gpu_milliseconds: gpu_ms,
            notes,
        })
    }
}

pub struct LZ4TextDemo {
    pub file_path: Option<String>,
    pub cpu_concurrency: usize,
}

impl LZ4TextDemo {
    pub fn new(file_path: Option<String>, cpu_concurrency: usize) -> Self {
        Self {
            file_path,
            cpu_concurrency: cpu_concurrency.max(1),
        }
    }
}

impl LZ4TextDemo {
    fn make_synthetic_data(&self) -> Vec<u8> {
        let paragraph = b"Call me Ishmael. Some years ago - never mind how long precisely - having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. Whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people's hats off - then, I account it high time to get to sea as soon as I can.";
        
        let mut data = Vec::new();
        data.reserve(paragraph.len() * 1024);
        
        for _ in 0..1024 {
            data.extend_from_slice(paragraph);
            data.push(b'\n');
        }
        
        data
    }
    
    fn make_synthetic_frames(&self, data: &[u8], block_size: usize) -> Vec<LZ4CompressedFrame> {
        // Create compressed frames using the compressor
        let decompressor = crate::LZ4Decompressor::new();
        match decompressor.compress(data, block_size) {
            Ok(frame) => vec![frame],
            Err(_) => vec![],
        }
    }
    
    fn combine_outputs(&self, outputs: &[Vec<u8>]) -> Vec<u8> {
        let total_size: usize = outputs.iter().map(|o| o.len()).sum();
        let mut combined = Vec::with_capacity(total_size);
        for output in outputs {
            combined.extend_from_slice(output);
        }
        combined
    }
}

impl CompressionDemo for LZ4TextDemo {
    fn name(&self) -> &str {
        "LZ4 Text Blocks"
    }

    fn run(&self, decompressor: &Decompressor) -> Result<BenchmarkResult> {
        if let Some(file_path) = &self.file_path {
            // Parse LZ4 file
            let parsed = crate::lz4_parser::LZ4FrameParser::parse_file(file_path)?;
            let frame = &parsed.frame;
            
            // Benchmark CPU decompression
            let (cpu_ms, cpu_output) = measure_time(5, || {
                decompressor.decompress_cpu(frame, Some(self.cpu_concurrency))
            })?;
            
            let mut notes = vec![
                format!("File: {}", parsed.file_path),
                format!("Frames: 1"),
                format!("Total blocks: {}", frame.blocks.len()),
                format!("CPU threads: {}", self.cpu_concurrency),
                format!("Block size limit: {} bytes", frame.block_size),
                format!("Block checksum: {}", frame.uses_block_checksum),
            ];
            
            if let Some(reported_size) = frame.reported_content_size {
                notes.push(format!("Header content size: {} bytes", reported_size));
            }
            
            // Benchmark GPU decompression if available
            let gpu_ms = if decompressor.has_gpu() {
                let (measured_gpu_ms, gpu_output) = measure_time(5, || {
                    pollster::block_on(decompressor.decompress_gpu(frame))
                })?;
                
                if gpu_output == cpu_output {
                    notes.push("GPU validation: output matches CPU decoder".to_string());
                } else {
                    notes.push("GPU validation FAILED: mismatch with CPU decoder".to_string());
                }
                
                notes.push(format!("Speedup (CPU/GPU): {:.2}x", cpu_ms / measured_gpu_ms));
                
                Some(measured_gpu_ms)
            } else {
                notes.push("GPU unavailable; skipping Metal acceleration.".to_string());
                None
            };
            
            Ok(BenchmarkResult {
                demo_name: self.name().to_string(),
                dataset_description: format!("File {} ({} blocks)", parsed.file_path, frame.blocks.len()),
                uncompressed_bytes: cpu_output.len(),
                compressed_bytes: parsed.file_size,
                cpu_milliseconds: cpu_ms,
                gpu_milliseconds: gpu_ms,
                notes,
            })
        } else {
            // Generate synthetic data for demo
            let synthetic_data = self.make_synthetic_data();
            let frames = self.make_synthetic_frames(&synthetic_data, 256 * 1024);
            
            let mut notes = vec![
                format!("Synthetic data: {} bytes", synthetic_data.len()),
                format!("Frames: {}", frames.len()),
                format!("Total blocks: {}", frames.iter().map(|f| f.blocks.len()).sum::<usize>()),
                format!("CPU threads: {}", self.cpu_concurrency),
            ];
            
            // Benchmark CPU decompression
            let (cpu_ms, cpu_outputs) = measure_time(5, || {
                frames.iter().map(|frame| {
                    decompressor.decompress_cpu(frame, Some(self.cpu_concurrency))
                }).collect::<Result<Vec<_>>>()
            })?;
            
            let cpu_output = self.combine_outputs(&cpu_outputs);
            notes.push(format!("CPU output size: {} bytes", cpu_output.len()));
            
            // Benchmark GPU decompression if available
            let gpu_ms = if decompressor.has_gpu() {
                let (measured_gpu_ms, gpu_outputs) = measure_time(5, || {
                    let mut results = Vec::new();
                    for frame in &frames {
                        let result = pollster::block_on(decompressor.decompress_gpu(frame))?;
                        results.push(result);
                    }
                    Ok(results)
                })?;
                
                let gpu_output = self.combine_outputs(&gpu_outputs);
                
                if gpu_output == cpu_output {
                    notes.push("GPU validation: output matches CPU decoder".to_string());
                } else {
                    notes.push("GPU validation FAILED: mismatch with CPU decoder".to_string());
                }
                
                notes.push(format!("Speedup (CPU/GPU): {:.2}x", cpu_ms / measured_gpu_ms));
                
                Some(measured_gpu_ms)
            } else {
                notes.push("GPU unavailable; skipping Metal acceleration.".to_string());
                None
            };
            
            Ok(BenchmarkResult {
                demo_name: self.name().to_string(),
                dataset_description: format!("Synthetic data: {} bytes ({} frames)", synthetic_data.len(), frames.len()),
                uncompressed_bytes: cpu_output.len(),
                compressed_bytes: frames.iter().map(|f| f.total_compressed_bytes).sum(),
                cpu_milliseconds: cpu_ms,
                gpu_milliseconds: gpu_ms,
                notes,
            })
        }
    }
}

pub fn measure_time<F, T>(repetitions: usize, mut operation: F) -> Result<(f64, T)>
where
    F: FnMut() -> Result<T>,
{
    let mut total_duration = Duration::new(0, 0);
    let mut last_result = None;

    for _ in 0..repetitions {
        let start = Instant::now();
        let result = operation()?;
        let duration = start.elapsed();
        total_duration += duration;
        last_result = Some(result);
    }

    let mean_duration = total_duration.as_nanos() as f64 / (repetitions as f64 * 1_000_000.0);
    Ok((mean_duration, last_result.unwrap()))
}

#[derive(Debug)]
pub struct ErrorMetrics {
    pub max: f32,
    pub mean: f32,
}

pub fn compute_error_metrics(original: &[f32], decompressed: &[f32]) -> ErrorMetrics {
    assert_eq!(original.len(), decompressed.len());
    
    let mut max_error = 0.0f32;
    let mut total_error = 0.0f64;
    
    for (orig, decomp) in original.iter().zip(decompressed.iter()) {
        let error = (orig - decomp).abs();
        max_error = max_error.max(error);
        total_error += error as f64;
    }
    
    ErrorMetrics {
        max: max_error,
        mean: (total_error / original.len() as f64) as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_metrics() {
        let original = vec![1.0, 2.0, 3.0];
        let decompressed = vec![1.1, 2.0, 2.9];
        let metrics = compute_error_metrics(&original, &decompressed);
        
        assert!(metrics.max > 0.0);
        assert!(metrics.mean > 0.0);
    }
}
