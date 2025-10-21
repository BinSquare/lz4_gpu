use crate::lz4::LZ4CompressedFrame;
use anyhow::{Context, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::*;

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
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32, // size = 16
}

pub struct GPUDevice {
    device: Device,
    queue: Queue,
    compute_pipeline: ComputePipeline,
}

pub struct GPUDecompressor {
    device: Arc<GPUDevice>,
}

impl GPUDevice {
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .context("Failed to get adapter")?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
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

        Ok(Self {
            device,
            queue,
            compute_pipeline,
        })
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
        let total_blocks = frame.blocks.len();
        if total_blocks == 0 {
            return Ok(vec![]);
        }

        // Prepare GPU data
        let gpu_blocks: Vec<GPUBlockInfo> = frame
            .blocks
            .iter()
            .map(|block| GPUBlockInfo {
                compressed_offset: block.compressed_offset,
                compressed_size: block.compressed_size,
                output_offset: block.output_offset,
                output_size: block.uncompressed_size,
                is_compressed: if block.is_compressed { 1 } else { 0 },
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            })
            .collect();

        let config = KernelConfig {
            block_count: total_blocks as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        // Convert byte data to u32 array for GPU
        let payload_u32 = Self::bytes_to_u32_array(&frame.payload);
        let output_size_u32 = (frame.uncompressed_size + 3) / 4; // round up to u32 boundary
        let output_size_bytes = (output_size_u32 * 4) as u64;

        // Buffers
        let payload_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Payload Buffer"),
                    contents: bytemuck::cast_slice(&payload_u32),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                });

        let block_info_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Block Info Buffer"),
                    contents: bytemuck::cast_slice(&gpu_blocks),
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                });

        let output_buffer = self.device.device.create_buffer(&BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for readback
        let staging_output_buffer = self.device.device.create_buffer(&BufferDescriptor {
            label: Some("LZ4 Staging Output Buffer"),
            size: output_size_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let config_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Config Buffer"),
                    contents: bytemuck::cast_slice(&[config]),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

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
            compute_pass.dispatch_workgroups(num_groups, 1, 1);
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

        // Read back from staging buffer
        let buffer_slice = staging_output_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        self.device.device.poll(Maintain::Wait);
        receiver
            .receive()
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to receive buffer"))??;

        let data = buffer_slice.get_mapped_range();
        let u32_result: &[u32] = bytemuck::cast_slice(&data);
        let bytes = Self::u32_array_to_bytes(u32_result, frame.uncompressed_size);
        drop(data);
        staging_output_buffer.unmap();

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

    fn u32_array_to_bytes(u32_array: &[u32], target_size: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(target_size);

        for &word in u32_array {
            for i in 0..4 {
                if result.len() < target_size {
                    result.push((word >> (i * 8)) as u8);
                }
            }
        }

        result.truncate(target_size);
        result
    }
}

const LZ4_SHADER: &str = r#"
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
    _pad0       : u32,
    _pad1       : u32,
    _pad2       : u32, // 16 bytes for uniform friendliness
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
    let word_index = index >> 2u;         // index / 4
    let byte_index = index & 3u;          // index % 4
    let word = compressed[word_index];
    return (word >> (byte_index * 8u)) & 0xFFu;
}

fn read_output_byte(index: u32) -> u32 {
    let word_index = index >> 2u;
    let byte_index = index & 3u;
    let word = output[word_index];
    return (word >> (byte_index * 8u)) & 0xFFu;
}

fn write_output_byte(index: u32, value: u32) {
    let word_index = index >> 2u;
    let byte_index = index & 3u;
    let shift = byte_index * 8u;
    let mask = ~(0xFFu << shift);
    let prev = output[word_index];
    output[word_index] = (prev & mask) | ((value & 0xFFu) << shift);
}

struct LenRead {
    len      : u32,
    next_src : u32,
};

// Reads a varint length extension per LZ4 rules: keep adding 255 until a byte < 255.
fn read_length_varint(src: u32, src_end: u32) -> LenRead {
    var s = src;
    var total = 0u;
    loop {
        if (s >= src_end) {
            break; // truncated; return what we have
        }
        let v = read_compressed_byte(s);
        s = s + 1u;
        total = total + v;
        if (v != 255u) {
            break;
        }
    }
    return LenRead(total, s);
}

@compute @workgroup_size(64)
fn lz4_decompress_blocks(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid = gid.x;
    if (tid >= config.block_count) {
        return;
    }

    let info = infos[tid];
    let src_start = info.compressed_offset;
    let src_end   = src_start + info.compressed_size;
    let dst_start = info.output_offset;
    let dst_end   = dst_start + info.output_size;

    if (info.compressed_size == 0u || info.output_size == 0u) {
        return;
    }

    // Fast path: uncompressed block (raw copy).
    if (info.is_compressed == 0u) {
        let count = min(info.compressed_size, info.output_size);
        var i = 0u;
        loop {
            if (i >= count) { break; }
            let b = read_compressed_byte(src_start + i);
            write_output_byte(dst_start + i, b);
            i = i + 1u;
        }
        return;
    }

    // LZ4 block decompression.
    var src = src_start;
    var dst = dst_start;

    loop {
        if (src >= src_end || dst >= dst_end) {
            break;
        }

        // Read token
        let token = read_compressed_byte(src);
        src = src + 1u;

        // Literal length (high nibble)
        var lit_len = token >> 4u;
        if (lit_len == 15u) {
            let lr = read_length_varint(src, src_end);
            lit_len = lit_len + lr.len;
            src = lr.next_src;
        }

        // Bounds check for literals
        if (src + lit_len > src_end) {
            // Truncated input; bail safely.
            break;
        }

        // Copy literals
        var i = 0u;
        loop {
            if (i >= lit_len || dst >= dst_end) { break; }
            let b = read_compressed_byte(src);
            write_output_byte(dst, b);
            src = src + 1u;
            dst = dst + 1u;
            i = i + 1u;
        }

        // End of sequences: valid to finish after literals
        if (src >= src_end) {
            break;
        }

        // Need 2 bytes for match offset
        if (src + 1u >= src_end) {
            break; // Truncated
        }

        // Read 2-byte little-endian offset
        let off_lo = read_compressed_byte(src);
        let off_hi = read_compressed_byte(src + 1u);
        let offset = (off_hi << 8u) | off_lo;
        src = src + 2u;

        // Invalid offset (zero) or before beginning of this block's output
        if (offset == 0u) {
            break;
        }
        let produced = dst - dst_start;
        if (produced < offset) {
            break; // would read before dst_start
        }

        // Match length (low nibble) with base +4
        var match_len = (token & 0x0Fu) + 4u;
        if ((token & 0x0Fu) == 0x0Fu) {
            let lr2 = read_length_varint(src, src_end);
            match_len = match_len + lr2.len;
            src = lr2.next_src;
        }

        // Copy match (overlap-safe: read from dst - offset each step)
        var j = 0u;
        loop {
            if (j >= match_len || dst >= dst_end) { break; }
            let b = read_output_byte(dst - offset);
            write_output_byte(dst, b);
            dst = dst + 1u;
            j = j + 1u;
        }
    }
}
"#;
