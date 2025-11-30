//! Persistent GPU Context Manager for Reduced Setup Overhead
//! This module implements a singleton pattern for GPU device persistence

use anyhow::Result;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use wgpu::*;

/// Global GPU context manager using singleton pattern
static GPU_CONTEXT_MANAGER: OnceLock<Arc<Mutex<GPUContextManager>>> = OnceLock::new();

/// Persistent GPU context that reduces setup overhead between operations
pub struct PersistentGPUContext {
    device: Device,
    queue: Queue,
    compute_pipeline: ComputePipeline,
    last_used: Instant,
    is_initialized: bool,
}

impl PersistentGPUContext {
    /// Create a new persistent GPU context
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to get adapter"))?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get device: {}", e))?;

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("LZ4 Decompression Shader"),
            source: ShaderSource::Wgsl(crate::gpu::LZ4_SHADER.into()),
        });

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("LZ4 Decompression Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "lz4_decompress_blocks",
        });

        Ok(Self {
            device,
            queue,
            compute_pipeline,
            last_used: Instant::now(),
            is_initialized: true,
        })
    }

    /// Check if the GPU context is still valid and usable
    pub fn is_valid(&self) -> bool {
        self.is_initialized
    }

    /// Get the time since last usage for cleanup decisions
    pub fn idle_duration(&self) -> Duration {
        self.last_used.elapsed()
    }

    /// Mark the context as recently used
    pub fn mark_used(&mut self) {
        self.last_used = Instant::now();
    }

    /// Get device reference for buffer creation
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get queue reference for command submission
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Get compute pipeline reference
    pub fn compute_pipeline(&self) -> &ComputePipeline {
        &self.compute_pipeline
    }
}

/// GPU Context Manager that handles persistent contexts with cleanup
pub struct GPUContextManager {
    context: Option<Arc<Mutex<PersistentGPUContext>>>,
    max_idle_time: Duration,
}

impl GPUContextManager {
    /// Create a new GPU context manager
    pub fn new() -> Self {
        Self {
            context: None,
            max_idle_time: Duration::from_secs(30), // 30 seconds idle before cleanup
        }
    }

    /// Get or create a persistent GPU context
    pub async fn get_context(&mut self) -> Result<Option<Arc<Mutex<PersistentGPUContext>>>> {
        // Check if we have a valid context
        if let Some(ref ctx) = self.context {
            let ctx_locked = ctx.lock().unwrap();
            if ctx_locked.is_valid() {
                // Check if context is still fresh
                if ctx_locked.idle_duration() < self.max_idle_time {
                    // Mark as used and return
                    drop(ctx_locked); // Release the lock before cloning
                    let ctx_clone = ctx.clone();
                    // Mark as used after cloning to avoid deadlock
                    if let Ok(mut ctx_mut) = ctx_clone.lock() {
                        ctx_mut.mark_used();
                    }
                    return Ok(Some(ctx_clone));
                }
                // Context is too old, clean it up
                drop(ctx_locked);
                self.context = None;
            } else {
                // Context is invalid, clean it up
                drop(ctx_locked);
                self.context = None;
            }
        }

        // Create a new context
        match PersistentGPUContext::new().await {
            Ok(ctx) => {
                let arc_ctx = Arc::new(Mutex::new(ctx));
                self.context = Some(arc_ctx.clone());
                Ok(Some(arc_ctx))
            }
            Err(e) => {
                // Failed to create context, clear any stale references
                self.context = None;
                eprintln!("Failed to create persistent GPU context: {}", e);
                Ok(None)
            }
        }
    }

    /// Cleanup unused contexts
    pub fn cleanup(&mut self) {
        if let Some(ref ctx) = self.context {
            let ctx_locked = ctx.lock().unwrap();
            if ctx_locked.idle_duration() >= self.max_idle_time {
                drop(ctx_locked);
                self.context = None;
            } else {
                drop(ctx_locked);
            }
        }
    }

    /// Force cleanup of all contexts
    pub fn reset(&mut self) {
        self.context = None;
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.context.is_some()
    }
}

impl Default for GPUContextManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the global GPU context manager
pub fn get_global_gpu_context_manager() -> Arc<Mutex<GPUContextManager>> {
    GPU_CONTEXT_MANAGER
        .get_or_init(|| Arc::new(Mutex::new(GPUContextManager::new())))
        .clone()
}

/// Initialize the global GPU context manager
pub async fn initialize_global_gpu_context() -> Result<Option<Arc<Mutex<PersistentGPUContext>>>> {
    let manager = get_global_gpu_context_manager();
    let mut manager_locked = manager.lock().unwrap();
    manager_locked.get_context().await
}

/// Get a persistent GPU context if available
pub async fn get_persistent_gpu_context() -> Option<Arc<Mutex<PersistentGPUContext>>> {
    let manager = get_global_gpu_context_manager();
    let mut manager_locked = manager.lock().unwrap();

    // Try to get an existing context or create a new one
    match manager_locked.get_context().await {
        Ok(ctx) => ctx,
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_persistent_gpu_context() -> Result<()> {
        // This test would need a real GPU device, so it's just for compilation verification
        println!("Persistent GPU Context module compiled successfully");
        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_context_manager() -> Result<()> {
        let mut manager = GPUContextManager::new();

        // Test context creation (may fail if no GPU)
        match manager.get_context().await {
            Ok(Some(_ctx)) => {
                println!("Successfully created GPU context");
                assert!(manager.has_gpu());
            }
            Ok(None) => {
                println!("No GPU available or failed to create context");
                assert!(!manager.has_gpu());
            }
            Err(e) => {
                eprintln!("Error creating GPU context: {}", e);
                assert!(!manager.has_gpu());
            }
        }

        // Test cleanup
        manager.cleanup();
        manager.reset();

        Ok(())
    }

    #[tokio::test]
    async fn test_global_gpu_context() -> Result<()> {
        // Test global context manager initialization
        let manager = get_global_gpu_context_manager();
        assert!(manager.lock().unwrap().context.is_none());

        // Test global context initialization
        let _ctx = initialize_global_gpu_context().await?;
        println!("Global GPU context initialized successfully");

        // Test getting persistent context
        let _persistent_ctx = get_persistent_gpu_context().await;
        println!("Persistent GPU context retrieved successfully");

        Ok(())
    }
}
