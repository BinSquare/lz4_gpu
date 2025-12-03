#!/bin/bash
# Simple performance benchmark for FilesCanFly

echo "🚀 FilesCanFly Performance Benchmark"
echo "====================================="
echo ""

cd /Users/binsquare/Documents/FilesCanFly

# Test files sorted by size (larger first for better GPU utilization)
TEST_FILES=(
    "test_data/lz4_compressed/xray.img.lz4"
    "test_data/lz4_compressed/samba.txt.lz4" 
    "test_data/lz4_compressed/mr.img.lz4"
    "test_data/lz4_compressed/xml.xml.lz4"
    "test_data/lz4_compressed/dickens.txt.lz4"
    "test_data/lz4_compressed/reymont.txt.lz4"
    "test_data/lz4_compressed/webster.txt.lz4"
    "test_data/lz4_compressed/sao.bin.lz4"
    "test_data/lz4_compressed/osdb.bin.lz4"
    "test_data/lz4_compressed/nci.txt.lz4"
    "test_data/lz4_compressed/mozilla.dat.lz4"
    "test_data/lz4_compressed/ooffice.dll.lz4"
)

echo "Testing $(ls test_data/lz4_compressed/*.lz4 | wc -l) files..."
echo ""

# Warmup
echo "🔥 Warming up..."
cargo run --release -- test_data/lz4_compressed/dickens.txt.lz4 --disable-gpu -o /tmp/warmup.txt >/dev/null 2>&1

echo "📊 Performance Results:"
echo "======================"

for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        filesize=$(du -h "$file" | cut -f1)
        
        echo ""
        echo "📄 $filename ($filesize):"
        echo "   ----------------------------------------"
        
        # CPU test
        cpu_time=$(cargo run --release -- "$file" --disable-gpu -o /tmp/cpu_test.txt 2>/dev/null | grep "CPU decompression completed" | awk '{print $4}')
        if [ -n "$cpu_time" ]; then
            echo "   CPU: ${cpu_time}"
        else
            echo "   CPU: Failed"
        fi
        
        # GPU test  
        gpu_time=$(cargo run --release -- "$file" -o /tmp/gpu_test.txt 2>/dev/null | grep "GPU decompression completed" | awk '{print $4}')
        if [ -n "$gpu_time" ]; then
            echo "   GPU: ${gpu_time}"
        else
            echo "   GPU: Not available or failed"
        fi
        
        # Clean up
        rm -f /tmp/cpu_test.txt /tmp/gpu_test.txt /tmp/warmup.txt
    fi
done

echo ""
echo "✅ Benchmark completed!"