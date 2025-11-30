use std::time::{Duration, Instant};
use std::collections::HashMap;
use anyhow::Result;

pub struct Profiler {
    timers: HashMap<String, Vec<Duration>>,
    start_times: HashMap<String, Instant>,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            timers: HashMap::new(),
            start_times: HashMap::new(),
        }
    }

    pub fn start(&mut self, name: &str) {
        self.start_times.insert(name.to_string(), Instant::now());
    }

    pub fn stop(&mut self, name: &str) {
        if let Some(start_time) = self.start_times.remove(name) {
            let elapsed = start_time.elapsed();
            self.timers.entry(name.to_string()).or_insert_with(Vec::new).push(elapsed);
        }
    }

    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Profiling Report ===\n");
        
        for (name, durations) in &self.timers {
            if durations.is_empty() {
                continue;
            }
            
            let total: Duration = durations.iter().sum();
            let count = durations.len();
            let avg = total / count as u32;
            let min = durations.iter().min().unwrap();
            let max = durations.iter().max().unwrap();
            
            report.push_str(&format!(
                "{}: count={}, total={:.2?}, avg={:.2?}, min={:.2?}, max={:.2?}\n",
                name, count, total, avg, min, max
            ));
        }
        
        report
    }

    pub fn clear(&mut self) {
        self.timers.clear();
        self.start_times.clear();
    }
}

pub struct ProfilingDecompressor {
    profiler: Profiler,
}

impl ProfilingDecompressor {
    pub fn new() -> Self {
        Self {
            profiler: Profiler::new(),
        }
    }

    pub fn get_profiler(&mut self) -> &mut Profiler {
        &mut self.profiler
    }

    pub async fn profile_gpu_decompression(
        &mut self,
        gpu_decompressor: &crate::GPUDecompressor,
        frame: &crate::LZ4CompressedFrame,
    ) -> Result<Vec<u8>> {
        self.profiler.start("gpu_decompression");
        let result = gpu_decompressor.decompress(frame).await;
        self.profiler.stop("gpu_decompression");
        result.map_err(|e| anyhow::anyhow!("GPU decompression failed: {}", e))
    }

    pub fn profile_cpu_decompression(
        &mut self,
        cpu_decompressor: &crate::LZ4Decompressor,
        frame: &crate::LZ4CompressedFrame,
        concurrency: Option<usize>,
    ) -> Result<Vec<u8>> {
        self.profiler.start("cpu_decompression_total");
        let result = cpu_decompressor.decompress(frame, concurrency);
        self.profiler.stop("cpu_decompression_total");
        result.map_err(|e| anyhow::anyhow!("CPU decompression failed: {}", e))
    }

    pub async fn profile_gpu_vs_cpu(
        &mut self,
        decompressor: &crate::Decompressor,
        frame: &crate::LZ4CompressedFrame,
        concurrency: Option<usize>,
    ) -> Result<ProfileResult> {
        let mut result = ProfileResult::default();
        
        // CPU profiling
        self.profiler.start("cpu_decompression");
        let cpu_result = decompressor.cpu_decompressor.decompress(frame, concurrency)?;
        self.profiler.stop("cpu_decompression");
        
        // GPU profiling (if available)
        if let Some(ref gpu) = decompressor.gpu_decompressor {
            self.profiler.start("gpu_decompression");
            let gpu_result = gpu.decompress(frame).await?;
            self.profiler.stop("gpu_decompression");
            
            // Verify results are the same
            if cpu_result != gpu_result {
                return Err(anyhow::anyhow!("CPU and GPU results don't match!"));
            }
            
            result.gpu_available = true;
            result.gpu_result_size = gpu_result.len();
        } else {
            result.gpu_available = false;
        }
        
        result.cpu_result_size = cpu_result.len();
        Ok(result)
    }
}

#[derive(Default, Debug)]
pub struct ProfileResult {
    pub cpu_result_size: usize,
    pub gpu_result_size: usize,
    pub gpu_available: bool,
}