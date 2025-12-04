# lz4_gpu (Rust)

Proof of concept work for LZ4 frame decompression with GPU acceleration. GPU is built on `wgpu` (Metal/Vulkan/DX12); CPU uses Rayon for block-parallel decode. This project focuses on decompression only.

## Features
- LZ4 frame parsing and validation (independent blocks only)
- CPU decompression with configurable thread count
- GPU decompression via WGSL compute shader and a pooled buffer allocator
- Direct I/O fast path optimized for Apple silicon unified memory

## Quick start
```bash
cargo build --release

# Decompress to stdout
cargo run --release -- test_data/lz4_compressed/samba.txt.lz4

# Write output and compare CPU vs GPU timings
cargo run --release -- test_data/lz4_compressed/samba.txt.lz4 --compare -o /tmp/out.txt

# Force CPU or adjust threads
cargo run --release -- test_data/lz4_compressed/samba.txt.lz4 --disable-gpu --cpu-threads 8
```

## Library use
```rust
use files_can_fly_rust::{Decompressor, LZ4FrameParser};

let parsed = LZ4FrameParser::parse_file("input.lz4")?;
let decompressor = Decompressor::new()?;
let cpu_bytes = decompressor.decompress_cpu(&parsed.frame, Some(8))?;
if decompressor.has_gpu() {
    let gpu_bytes = decompressor.decompress_gpu(&parsed.frame).await?;
}
```

## Notes for GPU runs
- GPU wins only when there are many blocks; small block counts favor CPU.
- Ensure Metal/Vulkan/DX12 is available; initialization failures fall back to CPU with a log message.
- Workgroup size is tuned for one block per invocation to avoid idle lanes on small files.

## Project layout
- `src/lz4.rs` – CPU decoder
- `src/gpu.rs` – GPU pipeline and WGSL shader
- `src/lz4_parser.rs` – LZ4 frame parser
- `src/main.rs` – CLI entry point
