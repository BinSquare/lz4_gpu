#!/bin/bash
# Simple performance test for FilesCanFly

echo "🚀 FilesCanFly Quick Performance Test"
echo "===================================="
echo ""

cd /Users/binsquare/Documents/FilesCanFly

# Define representative test files
TEST_FILES=(
    "test_data/lz4_compressed/dickens.txt.lz4"  # Medium file
    "test_data/lz4_compressed/sao.bin.lz4"      # Small file  
    "test_data/lz4_compressed/xray.img.lz4"     # Large file
)

echo "Testing representative files..."
echo ""

for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "📄 Testing: $filename"
        echo "   $(ls -lh "$file" | awk '{print $5}') compressed → $(ls -lh "${file%.lz4}" 2>/dev/null | awk '{print $5}' || echo '???') uncompressed"
        
        # CPU test
        echo "   CPU test:"
        time cargo run --release -- "$file" --disable-gpu -o /tmp/cpu_test.txt >/dev/null 2>&1
        
        # GPU test
        echo "   GPU test:"
        time cargo run --release -- "$file" -o /tmp/gpu_test.txt >/dev/null 2>&1
        
        echo ""
        rm -f /tmp/cpu_test.txt /tmp/gpu_test.txt
    fi
done

echo "✅ Performance test completed!"