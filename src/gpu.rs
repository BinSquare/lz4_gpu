use crate::lz4::{LZ4BlockDescriptor, LZ4CompressedFrame};
use crate::lz4_parser::LZ4FrameStream;
use crate::memory_pool::GPUMemoryPool;
use anyhow::{Context, Result};
use std::io::Write;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use wgpu::*;

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
    adapter_info: AdapterInfo,
    adapter_limits: Limits,
}

#[derive(Clone)]
pub struct GPUDecompressor {
    device: Arc<GPUDevice>,
}

struct DispatchBuffers {
    payload_buffer: Buffer,
    block_info_buffer: Buffer,
    output_buffer: Buffer,
    staging_output_buffer: Buffer,
    config_buffer: Buffer,
    status_buffer: Buffer,
    status_staging: Buffer,
    payload_size: u64,
    block_info_size: u64,
    output_size_bytes: u64,
    config_size: u64,
    status_size: u64,
    status_readback_size: u64,
    payload_usage: BufferUsages,
    block_info_usage: BufferUsages,
    output_usage: BufferUsages,
    staging_output_usage: BufferUsages,
    config_usage: BufferUsages,
    status_usage: BufferUsages,
    status_readback_usage: BufferUsages,
}

impl DispatchBuffers {
    fn return_to_pool(self, pool: &mut GPUMemoryPool) {
        pool.return_buffer(self.payload_buffer, self.payload_size, self.payload_usage);
        pool.return_buffer(
            self.block_info_buffer,
            self.block_info_size,
            self.block_info_usage,
        );
        pool.return_buffer(
            self.output_buffer,
            self.output_size_bytes,
            self.output_usage,
        );
        pool.return_buffer(
            self.staging_output_buffer,
            self.output_size_bytes,
            self.staging_output_usage,
        );
        pool.return_buffer(self.config_buffer, self.config_size, self.config_usage);
        pool.return_buffer(self.status_buffer, self.status_size, self.status_usage);
        pool.return_buffer(
            self.status_staging,
            self.status_readback_size,
            self.status_readback_usage,
        );
    }
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
                    label: Some("GPU Device"),
                    required_features: Features::empty(),
                    required_limits: limits.clone(),
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

        let gpu_device = Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            compute_pipeline: Arc::new(compute_pipeline),
            memory_pool,
            adapter_info,
            adapter_limits: limits,
        };

        // Don't store in global singleton since wgpu::Device isn't Clone, and we can't both store in static
        // and return the value (would need to clone, which we can't do for wgpu resources)

        // Return the created device
        Ok(gpu_device)
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
        let result = self.decompress_with_options(frame, None).await;

        result
    }

    /// Decompress the entire frame and stream the output directly to a writer without holding it all in RAM.
    pub async fn decompress_to_writer<W: Write + Send>(
        &self,
        frame: &LZ4CompressedFrame,
        writer: &mut W,
    ) -> Result<()> {
        self.decompress_with_options_to_writer(frame, None, writer)
            .await
    }

    /// Stream decompressed output block-by-block to a writer to avoid holding the full output in RAM.
    pub async fn decompress_streaming_to_writer<W: Write + Send>(
        &self,
        frame: &LZ4CompressedFrame,
        writer: &mut W,
    ) -> Result<()> {
        for block in &frame.blocks {
            let start = block.compressed_offset as usize;
            let end = start + block.compressed_size as usize;
            let block_slice = &frame.payload[start..end];

            let block_frame = LZ4CompressedFrame {
                uncompressed_size: block.uncompressed_size as usize,
                block_size: frame.block_size,
                blocks: vec![LZ4BlockDescriptor {
                    compressed_size: block.compressed_size,
                    uncompressed_size: block.uncompressed_size,
                    compressed_offset: 0,
                    output_offset: 0,
                    is_compressed: block.is_compressed,
                }],
                payload: Arc::from(block_slice.to_vec()),
                total_compressed_bytes: block_slice.len(),
                reported_content_size: Some(block.uncompressed_size as usize),
                uses_block_checksum: frame.uses_block_checksum,
            };

            let bytes = self.decompress(&block_frame).await?;
            writer.write_all(&bytes)?;
        }
        Ok(())
    }

    /// Split a frame into subframes that fit within GPU per-dispatch limits.
    /// Keeps compressed payload and padded output under 4GiB for WGSL/u32 indexing.
    pub fn split_frame_for_gpu(
        frame: &LZ4CompressedFrame,
        max_batch_bytes: usize,
    ) -> Result<Vec<LZ4CompressedFrame>> {
        if frame.blocks.is_empty() {
            return Ok(vec![frame.clone()]);
        }

        let mut batches = Vec::new();
        let mut current_blocks = Vec::new();
        let mut current_payload = Vec::new();
        let mut current_uncompressed: usize = 0;
        let mut output_cursor: u64 = 0;
        let mut padded_cursor_words: u64 = 0;

        for block in &frame.blocks {
            let _padded_offset_bytes = padded_cursor_words
                .checked_mul(4)
                .context("Padded output offset overflow")?;
            let padded_block_words = ((block.uncompressed_size as u64) + 3) / 4;
            let new_padded_cursor = padded_cursor_words
                .checked_add(padded_block_words)
                .context("Padded output length overflow")?;

            let block_start = block.compressed_offset as usize;
            let block_end = block_start + block.compressed_size as usize;
            let block_bytes = &frame.payload[block_start..block_end];

            let new_payload_len = current_payload
                .len()
                .checked_add(block_bytes.len())
                .ok_or_else(|| anyhow::anyhow!("Compressed payload size overflow"))?;
            let new_output_bytes = new_padded_cursor
                .checked_mul(4)
                .ok_or_else(|| anyhow::anyhow!("Padded output length overflow"))?;

            let would_exceed =
                new_payload_len > max_batch_bytes || new_output_bytes > max_batch_bytes as u64;

            if would_exceed && !current_blocks.is_empty() {
                batches.push(Self::make_subframe(
                    &current_blocks,
                    &current_payload,
                    current_uncompressed,
                    frame.block_size,
                    frame.uses_block_checksum,
                )?);
                current_blocks.clear();
                current_payload.clear();
                current_uncompressed = 0;
                output_cursor = 0;
                padded_cursor_words = 0;
            }

            let offset_in_batch = current_payload.len() as u64;
            current_payload.extend_from_slice(block_bytes);
            current_blocks.push(LZ4BlockDescriptor {
                compressed_size: block.compressed_size,
                uncompressed_size: block.uncompressed_size,
                compressed_offset: offset_in_batch,
                output_offset: output_cursor,
                is_compressed: block.is_compressed,
            });

            output_cursor = output_cursor
                .checked_add(block.uncompressed_size as u64)
                .ok_or_else(|| anyhow::anyhow!("Output offset overflow"))?;
            padded_cursor_words = new_padded_cursor;
            current_uncompressed = current_uncompressed
                .checked_add(block.uncompressed_size as usize)
                .ok_or_else(|| anyhow::anyhow!("Batch uncompressed overflow"))?;
        }

        if !current_blocks.is_empty() {
            batches.push(Self::make_subframe(
                &current_blocks,
                &current_payload,
                current_uncompressed,
                frame.block_size,
                frame.uses_block_checksum,
            )?);
        }

        Ok(batches)
    }

    fn make_subframe(
        blocks: &[LZ4BlockDescriptor],
        payload: &[u8],
        uncompressed_size: usize,
        block_size: usize,
        uses_block_checksum: bool,
    ) -> Result<LZ4CompressedFrame> {
        Ok(LZ4CompressedFrame {
            uncompressed_size,
            block_size,
            blocks: blocks.to_vec(),
            payload: Arc::from(payload.to_vec()),
            total_compressed_bytes: payload.len(),
            reported_content_size: Some(uncompressed_size),
            uses_block_checksum,
        })
    }

    /// Decompress with optional direct I/O path for unified memory systems
    pub async fn decompress_with_options(
        &self,
        frame: &LZ4CompressedFrame,
        _direct_io_path: Option<&str>,
    ) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(frame.uncompressed_size);
        self.decompress_with_options_to_writer(frame, _direct_io_path, &mut output)
            .await?;
        Ok(output)
    }

    async fn decompress_with_options_to_writer<W: Write + Send>(
        &self,
        frame: &LZ4CompressedFrame,
        _direct_io_path: Option<&str>,
        writer: &mut W,
    ) -> Result<()> {
        let total_blocks = frame.blocks.len();
        if total_blocks == 0 {
            return Ok(());
        }

        // Optional GPU dispatch/profile info for visibility. Enable with FCF_GPU_PROFILE=1.
        if std::env::var("FCF_GPU_PROFILE").is_ok() {
            let wg_size: u32 = 1;
            let num_groups = ((total_blocks as u32) + (wg_size - 1)) / wg_size;
            println!(
                "[gpu-profile] adapter={} ({:?}, {:?}), limits: max_workgroups_xyz={:?}, max_wg_size_xyz={:?}, max_total_wg_size={}, max_storage_buffer_binding_size={}, max_buffer_size={}",
                self.device.adapter_info.name,
                self.device.adapter_info.device_type,
                self.device.adapter_info.backend,
                self.device.adapter_limits.max_compute_workgroups_per_dimension,
                [
                    self.device.adapter_limits.max_compute_workgroup_size_x,
                    self.device.adapter_limits.max_compute_workgroup_size_y,
                    self.device.adapter_limits.max_compute_workgroup_size_z,
                ],
                self.device.adapter_limits.max_compute_invocations_per_workgroup,
                self.device.adapter_limits.max_storage_buffer_binding_size,
                self.device.adapter_limits.max_buffer_size,
            );
            println!(
                "[gpu-profile] dispatch blocks={}, workgroup_size={}, workgroups={}x1x1",
                total_blocks, wg_size, num_groups
            );
        }

        let compressed_length: u32 = frame
            .payload
            .len()
            .try_into()
            .context("GPU path does not support compressed payloads larger than 4GiB")?;

        let (gpu_blocks, padded_offsets, output_size_bytes, output_length) =
            self.build_block_metadata(frame)?;
        let buffers = self.prepare_dispatch_buffers(
            frame,
            compressed_length,
            output_size_bytes,
            output_length,
            &gpu_blocks,
        )?;

        self.submit_dispatch(total_blocks, &buffers)?;

        if let Some(err) = self.read_statuses(frame, &buffers).await? {
            let mut pool = self.device.memory_pool.lock().unwrap();
            buffers.return_to_pool(&mut pool);
            return Err(err);
        }

        self.write_output_to_writer(frame, &padded_offsets, &buffers, writer)
            .await?;

        let mut pool = self.device.memory_pool.lock().unwrap();
        buffers.return_to_pool(&mut pool);
        Ok(())
    }

    fn lock_pool(&self) -> Result<std::sync::MutexGuard<'_, GPUMemoryPool>> {
        self.device
            .memory_pool
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU memory pool lock poisoned"))
    }

    /// Stream a file through the GPU in batches to avoid holding the entire compressed payload in memory.
    pub async fn decompress_file_streaming_to_writer<W: Write + Send>(
        &self,
        path: &str,
        writer: &mut W,
        max_batch_blocks: usize,
    ) -> Result<()> {
        let mut stream = LZ4FrameStream::from_file(path, max_batch_blocks)?;
        while let Some(frame) = stream.next_batch()? {
            self.decompress_to_writer(&frame, writer).await?;
        }
        Ok(())
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

    fn build_block_metadata(
        &self,
        frame: &LZ4CompressedFrame,
    ) -> Result<(Vec<GPUBlockInfo>, Vec<u64>, u64, u32)> {
        let mut gpu_blocks: Vec<GPUBlockInfo> = Vec::with_capacity(frame.blocks.len());
        let mut padded_offsets: Vec<u64> = Vec::with_capacity(frame.blocks.len());
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
                compressed_offset: u32::try_from(block.compressed_offset)
                    .context("GPU path only supports per-batch compressed offsets up to 4GiB")?,
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
            .context("GPU path does not support outputs larger than 4GiB")?;

        Ok((gpu_blocks, padded_offsets, output_size_bytes, output_length))
    }

    fn prepare_dispatch_buffers(
        &self,
        frame: &LZ4CompressedFrame,
        compressed_length: u32,
        output_size_bytes: u64,
        output_length: u32,
        gpu_blocks: &[GPUBlockInfo],
    ) -> Result<DispatchBuffers> {
        let total_blocks = frame.blocks.len();
        let config = KernelConfig {
            block_count: total_blocks as u32,
            compressed_length,
            output_length,
            _pad0: 0,
        };
        let status_init = vec![0u32; total_blocks];

        let payload_u32 = Self::bytes_to_u32_array(&frame.payload);
        let payload_usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let payload_size =
            ((payload_u32.len() as u64) * (std::mem::size_of::<u32>() as u64)).max(1);
        let payload_buffer = {
            let mut pool = self.lock_pool()?;
            let buffer = if let Some(pool_buffer) = pool.get_buffer(payload_size, payload_usage) {
                pool_buffer
            } else {
                pool.allocate_new_buffer(&*self.device.device, payload_size, payload_usage)
            };

            self.device
                .queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&payload_u32));
            buffer
        };

        let block_info_usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let block_info_size =
            ((gpu_blocks.len() as u64) * (std::mem::size_of::<GPUBlockInfo>() as u64)).max(1);
        let block_info_buffer = {
            let mut pool = self.lock_pool()?;
            let buffer = if let Some(pool_buffer) = pool.get_buffer(block_info_size, block_info_usage)
            {
                pool_buffer
            } else {
                pool.allocate_new_buffer(&*self.device.device, block_info_size, block_info_usage)
            };

            self.device
                .queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(gpu_blocks));
            buffer
        };

        let output_usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let output_buffer = {
            let mut pool = self.lock_pool()?;
            if let Some(pool_buffer) = pool.get_buffer(output_size_bytes, output_usage) {
                pool_buffer
            } else {
                pool.allocate_new_buffer(&*self.device.device, output_size_bytes, output_usage)
            }
        };

        let staging_output_usage = BufferUsages::MAP_READ | BufferUsages::COPY_DST;
        let staging_output_buffer = {
            let mut pool = self.lock_pool()?;
            if let Some(pool_buffer) = pool.get_buffer(output_size_bytes, staging_output_usage) {
                pool_buffer
            } else {
                pool.allocate_new_buffer(
                    &*self.device.device,
                    output_size_bytes,
                    staging_output_usage,
                )
            }
        };

        let config_usage = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
        let config_size = std::mem::size_of::<KernelConfig>() as u64;
        let config_buffer = {
            let mut pool = self.lock_pool()?;
            let buffer = if let Some(pool_buffer) = pool.get_buffer(config_size, config_usage) {
                pool_buffer
            } else {
                pool.allocate_new_buffer(&*self.device.device, config_size, config_usage)
            };

            self.device
                .queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&[config]));
            buffer
        };

        let status_usage =
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let status_size = (status_init.len() as u64 * std::mem::size_of::<u32>() as u64).max(4);
        let status_buffer = {
            let mut pool = self.lock_pool()?;
            let buffer = if let Some(pool_buffer) = pool.get_buffer(status_size, status_usage) {
                pool_buffer
            } else {
                pool.allocate_new_buffer(&*self.device.device, status_size, status_usage)
            };

            self.device
                .queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&status_init));
            buffer
        };

        let status_readback_usage = BufferUsages::MAP_READ | BufferUsages::COPY_DST;
        let status_readback_size = (status_init.len() as u64 * std::mem::size_of::<u32>() as u64)
            .max(4);
        let status_staging = {
            let mut pool = self.lock_pool()?;
            if let Some(pool_buffer) =
                pool.get_buffer(status_readback_size, status_readback_usage)
            {
                pool_buffer
            } else {
                pool.allocate_new_buffer(
                    &*self.device.device,
                    status_readback_size,
                    status_readback_usage,
                )
            }
        };

        Ok(DispatchBuffers {
            payload_buffer,
            block_info_buffer,
            output_buffer,
            staging_output_buffer,
            config_buffer,
            status_buffer,
            status_staging,
            payload_size,
            block_info_size,
            output_size_bytes,
            config_size,
            status_size,
            status_readback_size,
            payload_usage,
            block_info_usage,
            output_usage,
            staging_output_usage,
            config_usage,
            status_usage,
            status_readback_usage,
        })
    }

    fn submit_dispatch(&self, total_blocks: usize, buffers: &DispatchBuffers) -> Result<()> {
        let bgl0 = self.device.compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.device.create_bind_group(&BindGroupDescriptor {
            label: Some("LZ4 Bind Group"),
            layout: &bgl0,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffers.payload_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buffers.block_info_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffers.output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buffers.config_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: buffers.status_buffer.as_entire_binding(),
                },
            ],
        });

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
            let wg_size: u32 = 1;
            let num_groups = ((total_blocks as u32) + (wg_size - 1)) / wg_size;
            compute_pass.dispatch_workgroups(num_groups.max(1), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &buffers.output_buffer,
            0,
            &buffers.staging_output_buffer,
            0,
            buffers.output_size_bytes,
        );

        let command_buffer = encoder.finish();
        self.device.queue.submit(std::iter::once(command_buffer));
        self.device.device.poll(Maintain::Wait);
        Ok(())
    }

    async fn read_statuses(
        &self,
        frame: &LZ4CompressedFrame,
        buffers: &DispatchBuffers,
    ) -> Result<Option<anyhow::Error>> {
        let mut status_encoder = self
            .device
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Status Readback Encoder"),
            });
        status_encoder.copy_buffer_to_buffer(
            &buffers.status_buffer,
            0,
            &buffers.status_staging,
            0,
            buffers.status_readback_size,
        );
        let status_cb = status_encoder.finish();
        self.device.queue.submit(std::iter::once(status_cb));

        let status_slice = buffers.status_staging.slice(..);
        let (status_tx, status_rx) = futures_intrusive::channel::shared::oneshot_channel();
        status_slice.map_async(MapMode::Read, move |res| {
            let _ = status_tx.send(res);
        });
        self.device.device.poll(Maintain::Wait);
        let _ = status_rx
            .receive()
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to receive status buffer map"))??;

        let status_data = status_slice.get_mapped_range();
        let mut gpu_error: Option<(usize, u32)> = None;
        for (idx, chunk) in status_data.chunks(4).enumerate() {
            let val = u32::from_le_bytes(chunk.try_into().unwrap_or_default());
            if val != 0 {
                gpu_error = Some((idx, val));
                break;
            }
        }
        drop(status_data);
        let _ = status_slice;
        buffers.status_staging.unmap();

        if let Some((idx, packed)) = gpu_error {
            let code = packed & 0xFF;
            let detail = packed >> 8;
            let block_debug = frame.blocks.get(idx).map(|b| {
                let start = b.compressed_offset as usize;
                let end = start.saturating_add(b.compressed_size as usize).min(frame.payload.len());
                let slice = &frame.payload[start..end];
                let token = slice.get(0).copied().unwrap_or(0);
                let off_lo = slice.get(1).copied().unwrap_or(0);
                let off_hi = slice.get(2).copied().unwrap_or(0);
                format!(
                    "compressed_size={}, output_size={}, is_compressed={}, first_token=0x{:02X}, first_offset=0x{:02X}{:02X}",
                    b.compressed_size,
                    b.uncompressed_size,
                    b.is_compressed,
                    token,
                    off_hi,
                    off_lo
                )
            });

            let err = anyhow::anyhow!(
                "GPU decompression failed for block {} (status code {}, detail {}){}",
                idx,
                code,
                detail,
                block_debug
                    .as_ref()
                    .map(|s| format!(", block={}", s))
                    .unwrap_or_default()
            );
            return Ok(Some(err));
        }

        Ok(None)
    }

    async fn write_output_to_writer<W: Write + Send>(
        &self,
        frame: &LZ4CompressedFrame,
        padded_offsets: &[u64],
        buffers: &DispatchBuffers,
        writer: &mut W,
    ) -> Result<()> {
        let buffer_slice = buffers.staging_output_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        self.device.device.poll(Maintain::Wait);

        let _map_result = receiver
            .receive()
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to receive buffer"))??;

        let data = buffer_slice.get_mapped_range();
        for (idx, block) in frame.blocks.iter().enumerate() {
            let padded_offset = padded_offsets[idx] as usize;
            let start = padded_offset;
            let end = start + block.uncompressed_size as usize;
            writer.write_all(&data[start..end])?;
        }
        drop(data);
        buffers.staging_output_buffer.unmap();
        Ok(())
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

@group(0) @binding(4)
var<storage, read_write> statuses : array<u32>;

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

fn set_error(block: u32, code: u32, detail: u32) {
    if (block < arrayLength(&statuses)) {
        // Pack detail in the upper bits for debugging: [detail:24][code:8]
        statuses[block] = (detail << 8u) | (code & 0xFFu);
    }
}

struct LenRead {
    len      : u32,
    next_src : u32,
};

// Reads a varint length extension - with max iteration safety
fn read_length_varint(src: u32, src_end: u32, max_len: u32) -> LenRead {
    var s = src;
    var total = 0u;
    
    loop {
        if (s >= src_end) {
            break;
        }
        let v = read_compressed_byte(s);
        s = s + 1u;
        total = total + v;
        if (total > max_len) {
            // Stop early; caller will treat oversized length as an error.
            break;
        }
        if (v != 255u) {
            break;
        }
    }
    return LenRead(total, s);
}

// Workgroup size 1 keeps one invocation per block, ensuring small block counts
// still fully parallelize instead of wasting lanes in a large workgroup.
@compute @workgroup_size(1)
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

    // Initialize status to success for this block (so stale data doesn't leak).
    set_error(block_id, 0u, 0u);

    // Validate block parameters
    if (info.compressed_size == 0u || info.output_size == 0u || src_start >= arrayLength(&compressed) * 4u) {
        set_error(block_id, 1u, 0u);
        return;
    }

    if ((src_start + info.compressed_size) > config.compressed_length) {
        set_error(block_id, 2u, 0u);
        return;
    }

    if ((dst_start + info.output_size) > config.output_length) {
        set_error(block_id, 3u, 0u);
        return;
    }

    // Uncompressed blocks must match output size.
    if (info.is_compressed == 0u && info.compressed_size != info.output_size) {
        set_error(block_id, 4u, 0u);
        return;
    }

    // Clear destination region to avoid stale data when decoding aborts early.
    var zi = 0u;
    loop {
        if (zi >= info.output_size) { break; }
        write_output_byte(dst_start + zi, 0u);
        zi = zi + 1u;
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
            let len_result = read_length_varint(src, src_end, info.output_size);
            literal_length = literal_length + len_result.len;
            src = len_result.next_src;
        }

        // Copy literals
        if (src + literal_length > src_end) {
            set_error(block_id, 5u, literal_length);
            return;
        }
        if (dst + literal_length > dst_end) {
            set_error(block_id, 5u, literal_length);
            return;
        }
        var literal_count = 0u;
        loop {
            if (literal_count >= literal_length) { 
                break; 
            }
            let literal_byte = read_compressed_byte(src);
            write_output_byte(dst, literal_byte);
            src = src + 1u;
            dst = dst + 1u;
            literal_count = literal_count + 1u;
        }

        // If we've consumed all input after literals, block is complete.
        if (src >= src_end) {
            break;
        }

        // Decode match offset (2 bytes, little-endian)
        if (src + 1u >= src_end) {
            set_error(block_id, 5u, 0u); // Need at least 2 bytes for offset
            return;
        }
        let offset_lo = read_compressed_byte(src);
        let offset_hi = read_compressed_byte(src + 1u);
        let offset = (offset_hi << 8u) | offset_lo;
        src = src + 2u;

        // Validate offset
        let written = dst - dst_start;
        if (offset == 0u || offset > written) {
            // Record where in the compressed stream we hit the bad offset (relative to block start).
            set_error(block_id, 6u, src - src_start);
            return; // Invalid offset
        }

        // Decode match length from lower nibble + 4
        var match_length = (token & 0x0Fu) + 4u;
        if ((token & 0x0Fu) == 15u) {
            let len_result = read_length_varint(src, src_end, info.output_size);
            match_length = match_length + len_result.len;
            src = len_result.next_src;
        }

        if (dst + match_length > dst_end) {
            set_error(block_id, 5u, match_length);
            return;
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
            if (source_pos < dst_start || source_pos >= dst) {
                // Invalid source position; bail with error.
                break;
            }
            let match_byte = read_output_byte(source_pos);
            write_output_byte(dst, match_byte);
            dst = dst + 1u;
            match_count = match_count + 1u;
        }

        // If we didn't complete the match copy, mark error.
        if (match_count < match_length) {
            set_error(block_id, 6u, src - src_start);
            return;
        }
    }

    // Success path: status already zeroed at start.
}
"#;
