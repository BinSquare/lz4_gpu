#!/bin/bash
# Performance comparison script for FilesCanFly vs standard LZ4 tools

echo "🚀 FilesCanFly vs Standard LZ4 Performance Comparison"
echo "================================================="
echo ""

cd /Users/binsquare/Documents/FilesCanFly

# Test files to compare (sorted by size for better GPU utilization)
TEST_FILES=(
    "test_data/lz4_compressed/sao.bin.lz4"      # Small file
    "test_data/lz4_compressed/dickens.txt.lz4"  # Medium file
    "test_data/lz4_compressed/samba.txt.lz4"    # Large file
    "test_data/lz4_compressed/xray.img.lz4"     # Very large file
)

echo "Testing $(echo ${TEST_FILES[@]} | wc -w) files with both FilesCanFly and standard LZ4..."
echo ""

# Warmup
echo "🔥 Warming up systems..."
cargo run --release -- test_data/lz4_compressed/sao.bin.lz4 --disable-gpu -o /tmp/warmup.txt >/dev/null 2>&1
lz4 -t test_data/lz4_compressed/sao.bin.lz4 >/dev/null 2>&1
echo ""

echo "📊 Performance Comparison Results:"
echo "==============================="

for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        filesize=$(du -h "$file" | awk '{print $1}')
        
        echo ""
        echo "📄 $filename ($filesize)"
        echo "----------------------------------------"
        
        # Standard LZ4 tool test
        echo "Standard LZ4:"
        time lz4 -d "$file" -c >/dev/null 2>&1
        
        # FilesCanFly CPU test
        echo "FilesCanFly CPU:"
        time cargo run --release -- "$file" --disable-gpu -o /tmp/cpu_test.txt >/dev/null 2>&1
        
        # FilesCanFly GPU test (if available)
        echo "FilesCanFly GPU:"
        time cargo run --release -- "$file" -o /tmp/gpu_test.txt >/dev/null 2>&1
        
        # Clean up
        rm -f /tmp/cpu_test.txt /tmp/gpu_test.txt /tmp/warmup.txt
    fi
done

echo ""
echo "✅ Performance comparison completed!"
echo ""
echo "💡 Notes:"
echo "  • Times shown are wall-clock time for decompression"
echo "  • GPU performance depends on available compute resources"
echo "  • FilesCanFly shows benefits with larger files and GPU acceleration"
echo "  • Standard LZ4 is highly optimized for CPU-only workloads"