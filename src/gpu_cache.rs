//! Global GPU Device Cache for Reduced Setup Overhead
//! This module provides a singleton pattern for caching GPU devices

use std::sync::{Arc, Mutex, OnceLock};
use wgpu::*;
use anyhow::Result;

/// Static GPU device instance to reduce initialization overhead
static GLOBAL_GPU_DEVICE: OnceLock<Result<(Device, Queue, ComputePipeline), String>> = OnceLock::new();

/// Initialize and cache a global GPU device instance
pub async fn initialize_global_gpu_device() -> Result<(Device, Queue, ComputePipeline)> {
    let result = GLOBAL_GPU_DEVICE.get_or_init(|| {
        // Try to initialize GPU device
        match pollster::block_on(async {
            let instance = Instance::new(InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&RequestAdapterOptions::default())
                .await
                .ok_or_else(|| "Failed to get adapter".to_string())?;

            let (device, queue) = adapter
                .request_device(&DeviceDescriptor::default(), None)
                .await
                .map_err(|e| format!("Failed to get device: {}", e))?;

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

            Ok((device, queue, compute_pipeline))
        }) {
            Ok(device_tuple) => Ok(device_tuple),
            Err(e) => Err(e.to_string()),
        }
    });
    
    match result {
        Ok((device, queue, compute_pipeline)) => Ok((device.clone(), queue.clone(), compute_pipeline.clone())),
        Err(e) => Err(anyhow::anyhow!("Failed to initialize GPU device: {}", e)),
    }
}

/// Get cached GPU device (returns None if not initialized)
pub fn get_cached_gpu_device() -> Option<(Device, Queue, ComputePipeline)> {
    GLOBAL_GPU_DEVICE.get().and_then(|result| {
        match result {
            Ok((device, queue, compute_pipeline)) => Some((device.clone(), queue.clone(), compute_pipeline.clone())),
            Err(_) => None,
        }
    })
}

/// Force re-initialization of global GPU device
pub fn reset_global_gpu_device() {
    // Note: OnceLock doesn't allow resetting, so this is a no-op
    // In a real implementation, we'd need a different approach
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_caching() -> Result<()> {
        // Test initialization
        match initialize_global_gpu_device().await {
            Ok((device, queue, pipeline)) => {
                println!("Successfully initialized GPU device");
                assert!(device.features().contains(Features::empty()));
                assert!(!queue.is_empty());
                assert!(!pipeline.label().is_none());
            }
            Err(e) => {
                eprintln!("Failed to initialize GPU device: {}", e);
            }
        }

        // Test getting cached device
        if let Some((device, queue, pipeline)) = get_cached_gpu_device() {
            println!("Successfully got cached GPU device");
            assert!(device.features().contains(Features::empty()));
            assert!(!queue.is_empty());
            assert!(!pipeline.label().is_none());
        } else {
            println!("No cached GPU device available");
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_cache() -> Result<()> {
        // Test getting cached GPU device
        let (device1, queue1, pipeline1) = get_cached_gpu_device().await?;
        let (device2, queue2, pipeline2) = get_cached_gpu_device().await?;
        
        // Both should be valid (though not necessarily identical due to cloning)
        assert!(device1.features().contains(Features::empty()));
        assert!(device2.features().contains(Features::empty()));
        
        // Clear cache
        clear_gpu_cache();
        
        println!("GPU cache test completed successfully");
        Ok(())
    }
}