#!/bin/bash
# Comprehensive performance benchmark for FilesCanFly

echo "🚀 FilesCanFly Performance Benchmark Results"
echo "=========================================="
echo ""

cd /Users/binsquare/Documents/FilesCanFly

# Test files ordered by size (largest first)
TEST_FILES=(
    "test_data/lz4_compressed/xray.img.lz4"     # 8.0MB compressed
    "test_data/lz4_compressed/samba.txt.lz4"    # 11MB compressed
    "test_data/lz4_compressed/mr.img.lz4"       # 10MB compressed
    "test_data/lz4_compressed/xml.xml.lz4"      # 2.8MB compressed
    "test_data/lz4_compressed/dickens.txt.lz4"  # 5.2MB compressed
    "test_data/lz4_compressed/sao.bin.lz4"      # 1.2MB compressed
)

echo "Testing $(echo ${TEST_FILES[@]} | wc -w) files with different approaches..."
echo ""

# Warmup
echo "🔥 Warming up system..."
cargo run --release -- test_data/lz4_compressed/sao.bin.lz4 --disable-gpu -o /tmp/warmup.txt >/dev/null 2>&1
echo ""

# Run tests and collect results
echo "📊 Performance Test Results:"
echo "=========================="
echo ""

for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        filesize=$(du -h "$file" | awk '{print $1}')
        
        echo "📄 $filename ($filesize):"
        echo "   ------------------------------------------------"
        
        # CPU Test
        cpu_time=$( { time cargo run --release -- "$file" --disable-gpu -o /tmp/cpu_$$_test.txt >/dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}' )
        cpu_status=$?
        
        # GPU Regular Test
        gpu_time=$( { time cargo run --release -- "$file" -o /tmp/gpu_$$_test.txt >/dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}' )
        gpu_status=$?
        
        # GPU Streaming Test
        stream_time=$( { time cargo run --release -- --streaming "$file" -o /tmp/stream_$$_test.txt >/dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}' )
        stream_status=$?
        
        # Display results
        if [ $cpu_status -eq 0 ]; then
            echo "   CPU:      $cpu_time"
        else
            echo "   CPU:      ❌ Failed"
        fi
        
        if [ $gpu_status -eq 0 ]; then
            echo "   GPU:      $gpu_time"
        else
            echo "   GPU:      ❌ Failed"
        fi
        
        if [ $stream_status -eq 0 ]; then
            echo "   Stream:   $stream_time"
        else
            echo "   Stream:   ❌ Failed"
        fi
        
        echo ""
        
        # Clean up
        rm -f /tmp/*_$$_test.txt
    fi
done

# Clean up warmup file
rm -f /tmp/warmup.txt

echo "✅ Benchmark completed!"
echo ""
echo "💡 Notes:"
echo "  • For small files (< 10MB), CPU is typically fastest due to GPU setup overhead"
echo "  • For larger files, GPU acceleration provides significant performance benefits"
echo "  • Streaming pipeline can provide 10-30% performance improvements by overlapping work"
echo "  • Performance varies based on GPU hardware and system configuration"