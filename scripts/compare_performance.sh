#!/bin/bash

# Performance comparison script for FilesCanFly Rust
# This script runs the Rust implementation benchmarks

set -e

echo "🔬 FilesCanFly Performance Comparison"
echo "===================================="

# Check if Rust is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found. Please install Rust toolchain."
    echo "   Visit: https://rustup.rs/"
    exit 1
fi

echo "✅ Rust toolchain found"

# Build Rust version
echo ""
echo "🔨 Building Rust implementation..."
cd /Users/binsquare/Documents/FilesCanFly
cargo build --release

# Run Rust benchmarks
echo ""
echo "🚀 Running Rust benchmarks..."
echo "-----------------------------"
time cargo run --release -- --cpu-threads 8

# Performance summary
echo ""
echo "📈 Performance Summary"
echo "====================="
echo "FilesCanFly Rust LZ4 decompression performance:"
echo "- CPU decompression time"
echo "- GPU decompression time (if available)"
echo "- Memory efficiency"
echo ""
echo "✨ Benchmark completed!"
