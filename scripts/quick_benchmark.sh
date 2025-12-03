#!/bin/bash
# Quick performance benchmark focusing on key metrics

echo "🚀 Quick FilesCanFly Performance Benchmark"
echo "======================================="
echo ""

cd /Users/binsquare/Documents/FilesCanFly

# Test files sorted by size (larger first for better GPU utilization)
TEST_FILES=(
    "test_data/lz4_compressed/dickens.txt.lz4"  # 5.2MB compressed
    "test_data/lz4_compressed/sao.bin.lz4"      # 1.2MB compressed  
)

echo "Testing $(echo ${TEST_FILES[@]} | wc -w) representative files..."
echo ""

# Warmup to ensure fair comparison
echo "🔥 Warming up..."
cargo run --release -- test_data/lz4_compressed/sao.bin.lz4 --disable-gpu -o /tmp/warmup.txt >/dev/null 2>&1
echo ""

# Performance comparison for each file
for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "📄 Testing: $filename"
        echo "----------------------------------------"
        
        # Standard LZ4 tool (baseline)
        echo "⏱️  Standard LZ4:"
        time lz4 -d "$file" -c >/dev/null 2>&1
        
        # FilesCanFly CPU
        echo "⏱️  FilesCanFly CPU:"
        time cargo run --release -- "$file" --disable-gpu -o /tmp/cpu_test.txt >/dev/null 2>&1
        
        # FilesCanFly GPU
        echo "⏱️  FilesCanFly GPU:"
        time cargo run --release -- "$file" -o /tmp/gpu_test.txt >/dev/null 2>&1
        
        echo ""
        
        # Clean up test files
        rm -f /tmp/cpu_test.txt /tmp/gpu_test.txt /tmp/warmup.txt
    fi
done

echo "✅ Quick performance benchmark completed!"
echo ""
echo "💡 Performance Analysis:"
echo "  • GPU performance improves with larger files due to setup overhead"
echo "  • CPU performance scales with core count and file size"
echo "  • Standard LZ4 is highly optimized for CPU-only workloads"
echo "  • FilesCanFly shows benefits with GPU acceleration for suitable workloads"