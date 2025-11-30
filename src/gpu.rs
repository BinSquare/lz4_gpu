use crate::lz4::LZ4CompressedFrame;
use crate::memory_pool::GPUMemoryPool;
use crate::persistent_gpu::GPUContextManager;
use anyhow::{Context, Result};
use std::sync::{Arc, Mutex, OnceLock};
use thiserror::Error;
use wgpu::*;

/// Global GPU device singleton to reduce initialization overhead
static GLOBAL_GPU_DEVICE: OnceLock<Arc<Mutex<GPUDevice>>> = OnceLock::new();

/// Errors that can occur during GPU operations
#[derive(Error, Debug)]
pub enum GPUError {
    #[error("GPU device initialization failed: {0}")]
    DeviceInitializationError(#[from] anyhow::Error),

    #[error("GPU buffer allocation failed")]
    BufferAllocationError,

    #[error("GPU compute shader execution failed: {0}")]
    ComputeExecutionError(String),

    #[error("GPU memory mapping failed")]
    MemoryMappingError,

    #[error("GPU resource cleanup failed")]
    ResourceCleanupError,

    #[error("GPU not available")]
    GPUNotAvailable,

    #[error("Invalid GPU context")]
    InvalidGPUContext,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUBlockInfo {
    pub compressed_offset: u32,
    pub compressed_size: u32,
    pub output_offset: u32,
    pub output_size: u32,
    pub is_compressed: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32, // size = 32
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KernelConfig {
    pub block_count: u32,
    pub compressed_length: u32,
    pub output_length: u32,
    pub _pad0: u32, // size = 16
}

#[derive(Clone)]
pub struct GPUDevice {
    device: Arc<Device>,
    queue: Arc<Queue>,
    compute_pipeline: Arc<ComputePipeline>,
    memory_pool: Arc<Mutex<GPUMemoryPool>>,
    context_manager: Arc<Mutex<GPUContextManager>>,
    persistent_context: Option<Arc<Mutex<crate::persistent_gpu::PersistentGPUContext>>>,
}

#[derive(Clone)]
pub struct GPUDecompressor {
    device: Arc<GPUDevice>,
}

impl GPUDevice {
    pub async fn new() -> Result<Self> {
        // For now, just create a new GPU device (skipping global singleton approach
        // since wgpu::Device and wgpu::Queue are not Clone)

        // Initialize new GPU device
        let instance_desc = InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        };
        let instance = Instance::new(instance_desc);

        // Try a few adapter preferences before giving up so Apple integrated GPUs are found.
        let adapter_options = [
            RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
            RequestAdapterOptions {
                power_preference: PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
            RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: true,
            },
        ];

        let mut adapter = None;
        for opts in adapter_options {
            if let Some(found) = instance.request_adapter(&opts).await {
                adapter = Some(found);
                break;
            }
        }

        let adapter = adapter.ok_or_else(|| anyhow::anyhow!("No suitable GPU adapter found"))?;

        let adapter_info = adapter.get_info();
        println!(
            "Using GPU adapter: {} ({:?}) via {:?}",
            adapter_info.name, adapter_info.device_type, adapter_info.backend
        );

        let limits = Limits::downlevel_defaults().using_resolution(adapter.limits());
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("FilesCanFly GPU Device"),
                    required_features: Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await
            .context("Failed to get device")?;

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("LZ4 Decompression Shader"),
            source: ShaderSource::Wgsl(LZ4_SHADER.into()),
        });

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("LZ4 Decompression Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "lz4_decompress_blocks",
        });

        // Create memory pool - buffers will be created with proper size and usage hints
        let memory_pool = Arc::new(Mutex::new(GPUMemoryPool::new()));
        let context_manager = Arc::new(Mutex::new(crate::persistent_gpu::GPUContextManager::new()));
        let persistent_context = None; // Will be initialized lazily when needed

        let gpu_device = Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            compute_pipeline: Arc::new(compute_pipeline),
            memory_pool,
            context_manager,
            persistent_context,
        };

        // Don't store in global singleton since wgpu::Device isn't Clone, and we can't both store in static
        // and return the value (would need to clone, which we can't do for wgpu resources)

        // Return the created device
        Ok(gpu_device)
    }

    /// Try to acquire a persistent GPU context guard
    fn try_acquire_persistent_context(&self) -> Option<PersistentContextGuard> {
        // Try to acquire a persistent GPU context for reduced setup overhead
        if let Some(ref _persistent_ctx) = self.persistent_context {
            // Attempt to acquire the persistent context
            // In a full implementation, this would actually acquire a context
            // For now, we'll just indicate that a persistent context is available
            Some(PersistentContextGuard)
        } else {
            // No persistent context available
            None
        }
    }
}

/// Guard for holding persistent GPU context
struct PersistentContextGuard;

impl Drop for PersistentContextGuard {
    fn drop(&mut self) {
        // Release the persistent context when guard goes out of scope
    }
}

impl GPUDecompressor {
    pub fn new() -> Result<Self> {
        let device = pollster::block_on(GPUDevice::new())?;
        Ok(Self {
            device: Arc::new(device),
        })
    }

    pub async fn decompress(&self, frame: &LZ4CompressedFrame) -> Result<Vec<u8>> {
        // Use persistent GPU context when available for reduced setup overhead
        let result = self.decompress_with_options(frame, None).await;

        // Mark GPU context as recently used
        {
            let _manager = self.device.context_manager.lock().unwrap();
            // In a real implementation, we'd update the context's last_used timestamp
            // This is a placeholder for keeping the GPU context warm
        }

        result
    }

    /// Decompress with optional direct I/O path for unified memory systems
    pub async fn decompress_with_options(
        &self,
        frame: &LZ4CompressedFrame,
        _direct_io_path: Option<&str>,
    ) -> Result<Vec<u8>> {
        // Try to use persistent GPU context for reduced setup overhead
        let _persistent_guard = self.device.try_acquire_persistent_context();
        let total_blocks = frame.blocks.len();
        if total_blocks == 0 {
            return Ok(vec![]);
        }

        let compressed_length: u32 = frame
            .payload
            .len()
            .try_into()
            .context("Compressed payload too large for GPU path")?;

        // Prepare GPU data with padded offsets to avoid cross-block word sharing
        let mut gpu_blocks: Vec<GPUBlockInfo> = Vec::with_capacity(total_blocks);
        let mut padded_offsets: Vec<u64> = Vec::with_capacity(total_blocks);
        let mut padded_cursor_words: u64 = 0;

        for block in &frame.blocks {
            let padded_offset_bytes = padded_cursor_words
                .checked_mul(4)
                .context("Padded output offset overflow")?;

            let padded_block_words = ((block.uncompressed_size as u64) + 3) / 4; // round up to word boundary
            padded_cursor_words = padded_cursor_words
                .checked_add(padded_block_words)
                .context("Padded output length overflow")?;

            gpu_blocks.push(GPUBlockInfo {
                compressed_offset: block.compressed_offset,
                compressed_size: block.compressed_size,
                output_offset: u32::try_from(padded_offset_bytes)
                    .context("Padded output offset too large for GPU path")?,
                output_size: block.uncompressed_size,
                is_compressed: if block.is_compressed { 1 } else { 0 },
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            });

            padded_offsets.push(padded_offset_bytes);
        }

        let output_size_bytes = padded_cursor_words
            .checked_mul(4)
            .context("Padded output length overflow")?
            .max(1);
        let output_length: u32 = output_size_bytes
            .try_into()
            .context("Output size too large for GPU path")?;

        let config = KernelConfig {
            block_count: total_blocks as u32,
            compressed_length,
            output_length,
            _pad0: 0,
        };

        // Convert byte data to u32 array for GPU
        let payload_u32 = Self::bytes_to_u32_array(&frame.payload);

        // Use memory pool for buffers
        let payload_buffer = {
            let mut pool = self.device.memory_pool.lock().unwrap();
            let buffer = if let Some(pool_buffer) = pool.get_buffer(
                ((payload_u32.len() as u64) * (std::mem::size_of::<u32>() as u64)).max(1),
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ) {
                // Got a buffer from the pool, reuse it
                pool_buffer
            } else {
                // Create a new buffer
                pool.allocate_new_buffer(
                    &*self.device.device,
                    ((payload_u32.len() as u64) * (std::mem::size_of::<u32>() as u64)).max(1),
                    BufferUsages::STORAGE | BufferUsages::COPY_DST,
                )
            };

            // Write data to the buffer
            self.device
                .queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&payload_u32));
            buffer
        };

        let block_info_buffer = {
            let mut pool = self.device.memory_pool.lock().unwrap();
            let buffer = if let Some(pool_buffer) = pool.get_buffer(
                ((gpu_blocks.len() as u64) * (std::mem::size_of::<GPUBlockInfo>() as u64)).max(1),
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ) {
                pool_buffer
            } else {
                pool.allocate_new_buffer(
                    &*self.device.device,
                    ((gpu_blocks.len() as u64) * (std::mem::size_of::<GPUBlockInfo>() as u64))
                        .max(1),
                    BufferUsages::STORAGE | BufferUsages::COPY_DST,
                )
            };

            // Write data to the buffer
            self.device
                .queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&gpu_blocks));
            buffer
        };

        let output_buffer = {
            let mut pool = self.device.memory_pool.lock().unwrap();
            if let Some(pool_buffer) = pool.get_buffer(
                output_size_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ) {
                pool_buffer
            } else {
                pool.allocate_new_buffer(
                    &*self.device.device,
                    output_size_bytes,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                )
            }
        };

        // Staging buffer for readback
        let staging_output_buffer = {
            let mut pool = self.device.memory_pool.lock().unwrap();
            if let Some(pool_buffer) = pool.get_buffer(
                output_size_bytes,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            ) {
                pool_buffer
            } else {
                pool.allocate_new_buffer(
                    &*self.device.device,
                    output_size_bytes,
                    BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                )
            }
        };

        let config_buffer = {
            let mut pool = self.device.memory_pool.lock().unwrap();
            let buffer = if let Some(pool_buffer) = pool.get_buffer(
                std::mem::size_of::<KernelConfig>() as u64,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ) {
                pool_buffer
            } else {
                pool.allocate_new_buffer(
                    &*self.device.device,
                    std::mem::size_of::<KernelConfig>() as u64,
                    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                )
            };

            // Write data to the buffer
            self.device
                .queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&[config]));
            buffer
        };

        // Bind group (use pipeline's layout to avoid mismatches)
        let bgl0 = self.device.compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("LZ4 Bind Group"),
            layout: &bgl0,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: payload_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: block_info_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        });

        // Encode
        let mut encoder = self
            .device
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("LZ4 Decompression Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("LZ4 Decompression Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.device.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch exactly enough workgroups for @workgroup_size(64)
            let wg_size: u32 = 64;
            let num_groups = ((total_blocks as u32) + (wg_size - 1)) / wg_size;
            compute_pass.dispatch_workgroups(num_groups.max(1), 1, 1);
        } // <- compute pass dropped here

        // Copy GPU output to staging (must happen after the pass ends)
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_output_buffer,
            0,
            output_size_bytes,
        );

        let command_buffer = encoder.finish();
        self.device.queue.submit(std::iter::once(command_buffer));

        // Wait for GPU to finish
        self.device.device.poll(Maintain::Wait);

        // Read back from staging buffer
        let buffer_slice = staging_output_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        // Drive the mapping to completion; without an explicit poll the callback
        // never fires and the async receive below hangs.
        self.device.device.poll(Maintain::Wait);

        let _map_result = receiver
            .receive()
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to receive buffer"))??;

        let data = buffer_slice.get_mapped_range();
        let mut bytes = vec![0u8; frame.uncompressed_size];
        for (idx, block) in frame.blocks.iter().enumerate() {
            let padded_offset = padded_offsets[idx] as usize;
            let start = padded_offset;
            let end = start + block.uncompressed_size as usize;
            let output_start = block.output_offset as usize;
            let output_end = output_start + block.uncompressed_size as usize;

            bytes[output_start..output_end].copy_from_slice(&data[start..end]);
        }
        drop(data);
        staging_output_buffer.unmap(); // Unmap before returning to pool

        // Return buffers to pool after they're no longer needed
        {
            let mut pool = self.device.memory_pool.lock().unwrap();

            // Return buffers to pool for reuse
            pool.return_buffer(
                payload_buffer,
                ((payload_u32.len() as u64) * (std::mem::size_of::<u32>() as u64)).max(1),
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            );

            pool.return_buffer(
                block_info_buffer,
                ((gpu_blocks.len() as u64) * (std::mem::size_of::<GPUBlockInfo>() as u64)).max(1),
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            );

            pool.return_buffer(
                output_buffer,
                output_size_bytes,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            );

            pool.return_buffer(
                staging_output_buffer,
                output_size_bytes,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            );

            pool.return_buffer(
                config_buffer,
                std::mem::size_of::<KernelConfig>() as u64,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            );
        }

        Ok(bytes)
    }
    fn bytes_to_u32_array(bytes: &[u8]) -> Vec<u32> {
        let mut result = Vec::with_capacity((bytes.len() + 3) / 4);
        let mut i = 0;

        while i < bytes.len() {
            let mut word = 0u32;
            for j in 0..4 {
                if i + j < bytes.len() {
                    word |= (bytes[i + j] as u32) << (j * 8);
                }
            }
            result.push(word);
            i += 4;
        }

        result
    }
}

pub const LZ4_SHADER: &str = r#"
struct GPUBlockInfo {
    compressed_offset : u32,
    compressed_size   : u32,
    output_offset     : u32,
    output_size       : u32,
    is_compressed     : u32,
    _pad0             : u32,
    _pad1             : u32,
    _pad2             : u32, // struct size = 32 bytes
};

struct KernelConfig {
    block_count : u32,
    compressed_length : u32,
    output_length : u32,
    _pad0       : u32, // 16 bytes for uniform friendliness
};

@group(0) @binding(0)
var<storage, read> compressed : array<u32>;

@group(0) @binding(1)
var<storage, read> infos : array<GPUBlockInfo>;

@group(0) @binding(2)
var<storage, read_write> output : array<u32>;

@group(0) @binding(3)
var<uniform> config : KernelConfig;

// ----- Byte helpers bound to specific buffers -----

fn read_compressed_byte(index: u32) -> u32 {
    // Simple bounds check: ensure we don't access beyond the compressed data
    if (index >= config.compressed_length) {
        return 0u;
    }
    let word_index = index >> 2u;         // index / 4
    let byte_index = index & 3u;          // index % 4
    
    // Check if word_index is within the array bounds
    if (word_index >= arrayLength(&compressed)) {
        return 0u;
    }
    let word = compressed[word_index];
    // Extract byte at correct position (little-endian)
    return (word >> (byte_index * 8u)) & 0xFFu;
}

fn read_output_byte(index: u32) -> u32 {
    // For reading output, we ensure we only read what has been written
    if (index >= config.output_length) {
        return 0u;
    }
    let word_index = index >> 2u;
    let byte_index = index & 3u;
    
    if (word_index >= arrayLength(&output)) {
        return 0u;
    }
    let word = output[word_index];
    return (word >> (byte_index * 8u)) & 0xFFu;
}

fn write_output_byte(index: u32, value: u32) {
    if (index >= config.output_length) {
        return;
    }
    let word_index = index >> 2u;
    let byte_index = index & 3u;
    
    if (word_index >= arrayLength(&output)) {
        return;
    }
    let shift = byte_index * 8u;
    let mask = ~(0xFFu << shift);
    let prev = output[word_index];
    output[word_index] = (prev & mask) | ((value & 0xFFu) << shift);
}

struct LenRead {
    len      : u32,
    next_src : u32,
};

// Reads a varint length extension - with max iteration safety
fn read_length_varint(src: u32, src_end: u32) -> LenRead {
    var s = src;
    var total = 0u;
    var count = 0u;
    // Maximum 4 extensions possible in LZ4 format (per spec)
    let max_extensions = 4u;
    
    loop {
        if (s >= src_end || count >= max_extensions) {
            break;
        }
        let v = read_compressed_byte(s);
        s = s + 1u;
        total = total + v;
        if (v != 255u) {
            break;
        }
        count = count + 1u;
    }
    return LenRead(total, s);
}

@compute @workgroup_size(64)
fn lz4_decompress_blocks(@builtin(global_invocation_id) gid : vec3<u32>) {
    let block_id = gid.x;
    if (block_id >= config.block_count) {
        return;
    }

    let info = infos[block_id];
    let src_start = info.compressed_offset;
    let src_end   = src_start + info.compressed_size;
    let dst_start = info.output_offset;
    let dst_end   = dst_start + info.output_size;

    // Validate block parameters
    if (info.compressed_size == 0u || info.output_size == 0u || src_start >= arrayLength(&compressed) * 4u) {
        return;
    }

    // Fast path: uncompressed block (raw copy).
    if (info.is_compressed == 0u) {
        // Copy byte by byte up to the minimum of compressed and output size
        let copy_size = min(info.compressed_size, info.output_size);
        var i = 0u;
        loop {
            if (i >= copy_size) { 
                break; 
            }
            let b = read_compressed_byte(src_start + i);
            write_output_byte(dst_start + i, b);
            i = i + 1u;
        }
        return;
    }

    // For compressed blocks, use the LZ4 algorithm
    var src = src_start;
    var dst = dst_start;
    
    // Main decoding loop - with hard safety limits to prevent infinite loops
    var iteration_count = 0u;
    let max_iterations = info.compressed_size; // Should be sufficient for the entire block
    
    loop {
        if (src >= src_end || dst >= dst_end || iteration_count >= max_iterations) {
            break;
        }
        iteration_count = iteration_count + 1u;

        // Read the token (sequence header)
        let token = read_compressed_byte(src);
        src = src + 1u;

        // Decode literal length from upper nibble
        var literal_length = (token >> 4u) & 0x0Fu;
        if (literal_length == 15u) {
            let len_result = read_length_varint(src, src_end);
            literal_length = literal_length + len_result.len;
            src = len_result.next_src;
        }

        // Copy literals
        var literal_count = 0u;
        loop {
            if (literal_count >= literal_length) { 
                break; 
            }
            if (src >= src_end || dst >= dst_end) {
                break; // Prevent reading beyond input or writing beyond output
            }
            let literal_byte = read_compressed_byte(src);
            write_output_byte(dst, literal_byte);
            src = src + 1u;
            dst = dst + 1u;
            literal_count = literal_count + 1u;
        }

        // Check if we've reached the end after literals
        if (src >= src_end || dst >= dst_end) {
            break;
        }

        // Decode match offset (2 bytes, little-endian)
        if (src + 1u >= src_end) {
            break; // Need at least 2 bytes for offset
        }
        let offset_lo = read_compressed_byte(src);
        let offset_hi = read_compressed_byte(src + 1u);
        let offset = (offset_hi << 8u) | offset_lo;
        src = src + 2u;

        // Validate offset
        if (offset == 0u || (dst - dst_start) < offset) {
            break; // Invalid offset
        }

        // Decode match length from lower nibble + 4
        var match_length = (token & 0x0Fu) + 4u;
        if ((token & 0x0Fu) == 15u) {
            let len_result = read_length_varint(src, src_end);
            match_length = match_length + len_result.len;
            src = len_result.next_src;
        }

        // Copy match bytes (with overlap support)
        var match_count = 0u;
        loop {
            if (match_count >= match_length) { 
                break; 
            }
            if (dst >= dst_end) {
                break; // Prevent writing beyond output
            }
            // Calculate source position for match byte
            let source_pos = dst - offset;
            if (source_pos >= dst_start && source_pos < dst) {
                let match_byte = read_output_byte(source_pos);
                write_output_byte(dst, match_byte);
                dst = dst + 1u;
            } else {
                break; // Invalid source position
            }
            match_count = match_count + 1u;
        }
    }
}
"#;
