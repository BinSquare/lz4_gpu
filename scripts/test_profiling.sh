#!/bin/bash
# Test script for profiling functionality

echo "Testing FilesCanFly profiling capabilities..."

cd /Users/binsquare/Documents/FilesCanFly

echo ""
echo "1. Testing help functionality..."
cargo run --release -- --help

echo ""
echo "2. Testing basic decompression..."
time cargo run --release -- test_data/lz4_compressed/sao.bin.lz4 -o /tmp/test_output.txt

echo ""
echo "3. Testing profiling (GPU)..."
time cargo run --release -- --profile test_data/lz4_compressed/sao.bin.lz4 -o /tmp/prof_output.txt

echo ""
echo "4. Testing CPU-only mode..."
time cargo run --release -- test_data/lz4_compressed/sao.bin.lz4 --disable-gpu -o /tmp/cpu_output.txt

echo ""
echo "5. Testing CPU profiling (may fail due to compatibility)..."
time cargo run --release -- --profile test_data/lz4_compressed/sao.bin.lz4 --disable-gpu -o /tmp/cpu_prof_output.txt

echo ""
echo "Tests completed!"