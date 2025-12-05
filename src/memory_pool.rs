use std::collections::HashMap;
use wgpu::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferKey {
    size: u64,
    usage: BufferUsages,
}

pub struct GPUMemoryPool {
    pool: HashMap<BufferKey, Vec<Buffer>>,
    stats: MemoryPoolStats,
}

// Hard limits to prevent unbounded growth in long-running processes.
const MAX_POOL_BUFFERS: usize = 32;
const MAX_POOL_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB of pooled buffers

#[derive(Debug, Default)]
pub struct MemoryPoolStats {
    pub allocated_count: usize,
    pub reused_count: usize,
    pub total_size_bytes: u64,
}

impl GPUMemoryPool {
    pub fn new() -> Self {
        Self {
            pool: HashMap::new(),
            stats: MemoryPoolStats::default(),
        }
    }

    pub fn get_buffer(&mut self, size: u64, usage: BufferUsages) -> Option<Buffer> {
        let key = BufferKey { size, usage };

        if let Some(buffer_list) = self.pool.get_mut(&key) {
            if let Some(buffer) = buffer_list.pop() {
                self.stats.reused_count += 1;
                return Some(buffer);
            }
        }

        None // No buffer available in pool
    }

    pub fn allocate_new_buffer(
        &mut self,
        device: &Device,
        size: u64,
        usage: BufferUsages,
    ) -> Buffer {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Pooled GPU Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });

        self.stats.allocated_count += 1;
        self.stats.total_size_bytes += size;

        buffer
    }

    pub fn return_buffer(&mut self, buffer: Buffer, size: u64, usage: BufferUsages) {
        let key = BufferKey { size, usage };
        self.pool.entry(key).or_insert_with(Vec::new).push(buffer);
        self.enforce_limits();
    }

    pub fn get_stats(&self) -> &MemoryPoolStats {
        &self.stats
    }

    pub fn clear(&mut self) {
        self.pool.clear();
    }

    pub fn trim(&mut self) {
        // Remove empty lists to save memory
        self.pool.retain(|_, list| !list.is_empty());
    }

    fn enforce_limits(&mut self) {
        loop {
            let (buffer_count, total_bytes) = self.current_usage();
            if buffer_count <= MAX_POOL_BUFFERS && total_bytes <= MAX_POOL_BYTES {
                break;
            }

            // Evict from the largest buffers first to free space quickly.
            if let Some((_, list)) = self
                .pool
                .iter_mut()
                .max_by_key(|(key, list)| (key.size, list.len()))
            {
                let _ = list.pop();
            } else {
                break;
            }

            self.trim();
        }
    }

    fn current_usage(&self) -> (usize, u64) {
        let mut buffer_count = 0;
        let mut total_bytes = 0;

        for (key, list) in &self.pool {
            let count = list.len() as u64;
            buffer_count += count as usize;
            total_bytes += key.size * count;
        }

        (buffer_count, total_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let mut pool = GPUMemoryPool::new();
        assert!(pool.get_buffer(1024, BufferUsages::MAP_READ).is_none());
        pool.trim();
    }
}
