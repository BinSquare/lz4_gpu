#!/bin/bash
# Comprehensive benchmark comparing FilesCanFly CPU vs GPU performance

echo "🚀 FilesCanFly CPU vs GPU Performance Benchmark"
echo "=============================================="
echo ""

cd /Users/binsquare/Documents/FilesCanFly

# Test files sorted by size
TEST_FILES=(
    "test_data/lz4_compressed/dickens.txt.lz4"  # 5.2MB compressed
    "test_data/lz4_compressed/sao.bin.lz4"      # 1.2MB compressed
    "test_data/lz4_compressed/xray.img.lz4"     # 8.0MB compressed
    "test_data/lz4_compressed/samba.txt.lz4"    # 11MB compressed
)

echo "Testing $(echo ${TEST_FILES[@]} | wc -w) representative files..."
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
        
        # CPU Test (3 runs for average)
        echo "CPU (3 runs):"
        cpu_times=()
        for i in {1..3}; do
            cpu_time=$( { time cargo run --release -- "$file" --disable-gpu -o /tmp/cpu_test_$i.txt >/dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}' )
            cpu_times+=("$cpu_time")
            echo "  Run $i: $cpu_time"
        done
        
        # GPU Test (3 runs for average)
        echo "GPU (3 runs):"
        gpu_times=()
        for i in {1..3}; do
            gpu_time=$( { time cargo run --release -- "$file" -o /tmp/gpu_test_$i.txt >/dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}' )
            gpu_times+=("$gpu_time")
            echo "  Run $i: $gpu_time"
        done
        
        # Calculate averages (simple extraction of numeric part)
        cpu_avg=$(printf '%s\n' "${cpu_times[@]}" | sed 's/[a-z]*//' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }')
        gpu_avg=$(printf '%s\n' "${gpu_times[@]}" | sed 's/[a-z]*//' | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }')
        
        echo "  CPU Average: ${cpu_avg}s"
        echo "  GPU Average: ${gpu_avg}s"
        
        # Calculate speedup
        if [ -n "$cpu_avg" ] && [ -n "$gpu_avg" ] && [ "$gpu_avg" != "0" ]; then
            speedup=$(echo "$cpu_avg / $gpu_avg" | bc -l)
            rounded_speedup=$(printf "%.2f" "$speedup")
            echo "  Speedup: ${rounded_speedup}x"
        fi
        
        # Clean up
        rm -f /tmp/cpu_test_*.txt /tmp/gpu_test_*.txt /tmp/warmup.txt
    fi
done

echo ""
echo "✅ Performance benchmark completed!"
echo ""
echo "💡 Notes:"
echo "  • Times shown are wall-clock time including all overhead"
echo "  • GPU performance depends on available compute resources"
echo "  • FilesCanFly shows benefits with larger files and GPU acceleration"
echo "  • Performance varies based on GPU hardware and system configuration"