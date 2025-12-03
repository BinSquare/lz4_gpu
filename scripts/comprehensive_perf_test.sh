#!/bin/bash
# Comprehensive performance test for FilesCanFly

echo "🚀 FilesCanFly Comprehensive Performance Test"
echo "=========================================="
echo ""

cd /Users/binsquare/Documents/FilesCanFly

# Test files sorted by size
TEST_FILES=(
    "test_data/lz4_compressed/xray.img.lz4"     # 8.0MB
    "test_data/lz4_compressed/samba.txt.lz4"    # 11MB  
    "test_data/lz4_compressed/mr.img.lz4"       # 10MB
    "test_data/lz4_compressed/dickens.txt.lz4"  # 5.2MB
    "test_data/lz4_compressed/sao.bin.lz4"      # 1.2MB
)

echo "Testing $(echo ${TEST_FILES[@]} | wc -w) files with both CPU and GPU..."
echo ""

# Warmup
echo "🔥 Warming up..."
cargo run --release -- test_data/lz4_compressed/sao.bin.lz4 --disable-gpu -o /tmp/warmup.txt >/dev/null 2>&1
echo ""

# Performance comparison
echo "📊 Performance Comparison Results:"
echo "================================"

for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        filesize=$(du -h "$file" | awk '{print $1}')
        
        echo ""
        echo "📄 $filename ($filesize)"
        echo "------------------------"
        
        # CPU Test
        echo "CPU:"
        time cargo run --release -- "$file" --disable-gpu -o /tmp/cpu_${filename}.txt >/dev/null 2>&1
        
        # GPU Test
        echo "GPU:"
        time cargo run --release -- "$file" -o /tmp/gpu_${filename}.txt >/dev/null 2>&1
        
        # Clean up
        rm -f /tmp/cpu_${filename}.txt /tmp/gpu_${filename}.txt
    fi
done

echo ""
echo "✅ Performance test completed!"
echo ""
echo "💡 Notes:"
echo "  • Times shown are wall-clock time for complete decompression"
echo "  • GPU performance varies with file size and block count"
echo "  • Larger files typically benefit more from GPU acceleration"
echo "  • Performance also depends on GPU hardware capabilities"