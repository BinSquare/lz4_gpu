//! Simple performance profiling for lz4_gpu 
//! This module provides simplified performance testing without complex GPU context management

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Simple timing profiler
pub struct SimpleProfiler {
    timings: HashMap<String, Vec<Duration>>,
}

impl SimpleProfiler {
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
        }
    }

    pub fn start_timer(&mut self, name: &str) -> TimerGuard<'_> {
        TimerGuard::new(name.to_string(), self)
    }

    pub fn record_time(&mut self, name: &str, duration: Duration) {
        self.timings
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }

    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Performance Report ===\n");

        for (name, durations) in &self.timings {
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
}

/// Timer guard for automatic timing
pub struct TimerGuard<'a> {
    name: String,
    start: Instant,
    profiler: &'a mut SimpleProfiler,
}

impl<'a> TimerGuard<'a> {
    pub fn new(name: String, profiler: &'a mut SimpleProfiler) -> Self {
        Self {
            name,
            start: Instant::now(),
            profiler,
        }
    }
}

impl<'a> Drop for TimerGuard<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.profiler.record_time(&self.name, duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_profiler() {
        let mut profiler = SimpleProfiler::new();

        // Record some timings
        profiler.record_time("test_op", Duration::from_millis(100));
        profiler.record_time("test_op", Duration::from_millis(150));
        profiler.record_time("test_op", Duration::from_millis(120));

        let report = profiler.report();
        assert!(report.contains("test_op"));
        assert!(report.contains("count=3"));
        println!("{}", report);
    }

    #[test]
    fn test_timer_guard() {
        let mut profiler = SimpleProfiler::new();

        {
            let _guard = profiler.start_timer("guarded_op");
            // Simulate some work
            std::thread::sleep(Duration::from_millis(10));
        }

        let report = profiler.report();
        assert!(report.contains("guarded_op"));
        assert!(report.contains("count=1"));
        println!("{}", report);
    }
}
