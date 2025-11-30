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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        // This test would need a real GPU device, so it's just for compilation verification
    }
}
