//! Resource management and cleanup for GPU operations
//! This module provides safe resource handling with automatic cleanup

use std::any::Any;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu::*;

/// Resource guard for automatic cleanup
pub struct ResourceGuard {
    cleanup_fn: Option<Box<dyn FnOnce() + Send>>,
    created_at: Instant,
    resource: Option<Box<dyn Any + Send>>,
}

impl ResourceGuard {
    /// Create a new resource guard
    pub fn new<T: Send + 'static>(resource: T, cleanup_fn: impl FnOnce() + Send + 'static) -> Self {
        Self {
            cleanup_fn: Some(Box::new(cleanup_fn)),
            created_at: Instant::now(),
            resource: Some(Box::new(resource)),
        }
    }

    /// Get resource age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get a reference to the guarded resource if the type matches
    pub fn resource<T: 'static>(&self) -> Option<&T> {
        self.resource
            .as_ref()
            .and_then(|res| res.downcast_ref::<T>())
    }

    /// Release the resource without running the cleanup function
    pub fn release(mut self) -> Option<Box<dyn Any + Send>> {
        self.cleanup_fn.take();
        self.resource.take()
    }
}

impl Drop for ResourceGuard {
    fn drop(&mut self) {
        if let Some(cleanup_fn) = self.cleanup_fn.take() {
            cleanup_fn();
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub allocated_bytes: u64,
    pub used_bytes: u64,
    pub buffer_count: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub oldest_buffer_age: Duration,
}

impl Default for MemoryPoolStats {
    fn default() -> Self {
        Self {
            allocated_bytes: 0,
            used_bytes: 0,
            buffer_count: 0,
            allocation_count: 0,
            deallocation_count: 0,
            oldest_buffer_age: Duration::from_secs(0),
        }
    }
}

/// GPU memory pool with automatic cleanup
/// Note: This is a simplified implementation that tracks buffer usage patterns
/// Actual buffer management should be handled by the GPU device
pub struct GPUMemoryPool {
    buffer_sizes: Arc<Mutex<Vec<BufferInfo>>>,
    stats: Arc<Mutex<MemoryPoolStats>>,
    max_age: Duration,
    max_buffers: usize,
}

/// Buffer information for tracking purposes
#[derive(Debug, Clone)]
struct BufferInfo {
    created_at: Instant,
}

impl BufferInfo {
    /// Create a new buffer info entry
    pub fn new() -> Self {
        Self { created_at: Instant::now() }
    }

    /// Get buffer age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Check if buffer is too old
    pub fn is_too_old(&self, max_age: Duration) -> bool {
        self.age() > max_age
    }
}

impl GPUMemoryPool {
    /// Create a new GPU memory pool
    pub fn new() -> Self {
        Self {
            buffer_sizes: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(MemoryPoolStats::default())),
            max_age: Duration::from_secs(60), // 1 minute max age
            max_buffers: 128,                 // Maximum number of buffers to keep
        }
    }

    /// Create a new GPU memory pool with custom settings
    pub fn with_settings(max_age: Duration, max_buffers: usize) -> Self {
        Self {
            buffer_sizes: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(MemoryPoolStats::default())),
            max_age,
            max_buffers,
        }
    }

    /// Record buffer allocation for statistics tracking
    pub fn record_buffer_allocation(&self, size: u64, _usage: BufferUsages) {
        let mut buffer_sizes = self.buffer_sizes.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        // Add buffer info for tracking
        buffer_sizes.push(BufferInfo::new());

        // Update stats
        stats.allocation_count += 1;
        stats.allocated_bytes += size;
        stats.buffer_count += 1;
        stats.used_bytes += size;

        // Trim old buffers if pool is too large
        if buffer_sizes.len() > self.max_buffers {
            buffer_sizes.retain(|entry| !entry.is_too_old(self.max_age));
        }
    }

    /// Record buffer usage for statistics tracking
    pub fn record_buffer_usage(&self, size: u64) {
        let mut stats = self.stats.lock().unwrap();
        stats.used_bytes += size;
    }

    /// Record buffer return for statistics tracking
    pub fn record_buffer_return(&self, size: u64) {
        let mut stats = self.stats.lock().unwrap();
        if stats.used_bytes >= size {
            stats.used_bytes -= size;
        }
    }

    /// Get memory pool statistics
    pub fn get_stats(&self) -> MemoryPoolStats {
        let buffer_sizes = self.buffer_sizes.lock().unwrap();
        let stats = self.stats.lock().unwrap();

        let oldest_buffer_age = buffer_sizes
            .iter()
            .map(|entry| entry.age())
            .max()
            .unwrap_or(Duration::from_secs(0));

        MemoryPoolStats {
            allocated_bytes: stats.allocated_bytes,
            used_bytes: stats.used_bytes,
            buffer_count: stats.buffer_count,
            allocation_count: stats.allocation_count,
            deallocation_count: stats.deallocation_count,
            oldest_buffer_age,
        }
    }

    /// Trim old buffers from the pool
    pub fn trim_old_buffers(&self) {
        let mut buffer_sizes = self.buffer_sizes.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let old_count = buffer_sizes.len();
        buffer_sizes.retain(|entry| !entry.is_too_old(self.max_age));

        let removed_count = old_count - buffer_sizes.len();
        if removed_count > 0 {
            stats.deallocation_count += removed_count;
        }
    }

    /// Clear all buffers from the pool
    pub fn clear(&self) {
        let mut buffer_sizes = self.buffer_sizes.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let buffer_count = buffer_sizes.len();
        buffer_sizes.clear();

        stats.buffer_count = 0;
        stats.allocated_bytes = 0;
        stats.used_bytes = 0;
        stats.deallocation_count += buffer_count;
    }

    /// Get buffer efficiency ratio
    pub fn efficiency_ratio(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        if stats.allocated_bytes > 0 {
            (stats.used_bytes as f64) / (stats.allocated_bytes as f64)
        } else {
            1.0
        }
    }
}

impl Default for GPUMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn test_resource_guard() -> Result<()> {
        // Test resource guard creation and cleanup
        let resource = Arc::new(42i32);
        let guard = ResourceGuard::new(resource.clone(), || {
            println!("Resource cleaned up");
        });

        assert!(guard.resource::<Arc<i32>>().is_some());
        assert_eq!(**guard.resource::<Arc<i32>>().unwrap(), 42);

        // Test release without cleanup
        let released = guard.release();
        assert!(released.unwrap().downcast::<Arc<i32>>().ok().is_some());

        Ok(())
    }

    #[test]
    fn test_memory_pool() -> Result<()> {
        // Test memory pool creation
        let pool = GPUMemoryPool::new();
        assert_eq!(pool.get_stats().buffer_count, 0);
        assert_eq!(pool.get_stats().allocated_bytes, 0);

        // Test with custom settings
        let pool_custom = GPUMemoryPool::with_settings(Duration::from_secs(30), 64);
        assert_eq!(pool_custom.max_age, Duration::from_secs(30));
        assert_eq!(pool_custom.max_buffers, 64);

        Ok(())
    }

    #[test]
    fn test_trim_and_clear() -> Result<()> {
        let pool = GPUMemoryPool::new();

        // Test trim (no-op since pool is empty)
        pool.trim_old_buffers();
        assert_eq!(pool.get_stats().buffer_count, 0);

        // Test clear (no-op since pool is empty)
        pool.clear();
        assert_eq!(pool.get_stats().buffer_count, 0);

        Ok(())
    }
}
