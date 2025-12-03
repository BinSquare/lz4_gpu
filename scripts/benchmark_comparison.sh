#!/bin/bash
# Benchmark script to compare FilesCanFly against standard LZ4 tool

cd /Users/binsquare/Documents/FilesCanFly

echo "🔬 FilesCanFly vs Standard LZ4 Benchmark"
echo "======================================"

echo ""
echo "📊 Testing FilesCanFly Decompression Performance:"
echo "----------------------------------------------"
time cargo run --release

echo ""
echo "📊 Testing Standard LZ4 Tool Performance:"
echo "----------------------------------------"
echo "Testing dickens.txt.lz4 (10MB text file):"
time lz4 -b1 -i3 -d test_data/lz4_compressed/dickens.txt.lz4

echo ""
echo "Testing sao.bin.lz4 (7MB binary file):"
time lz4 -b1 -i3 -d test_data/lz4_compressed/sao.bin.lz4

echo ""
echo "Testing webster.txt.lz4 (0.9MB dictionary-like file):"
time lz4 -b1 -i3 -d test_data/lz4_compressed/webster.txt.lz4

echo ""
echo "📊 Individual File Compression Ratios:"
echo "--------------------------------------"
for file in test_data/lz4_compressed/*.lz4; do
    if [ -f "$file" ]; then
        original=$(basename "$file" .lz4)
        original_path="test_data/$original"
        if [ -f "$original_path" ]; then
            orig_size=$(stat -f%z "$original_path" 2>/dev/null || stat -c%s "$original_path" 2>/dev/null)
            comp_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
            ratio=$(echo "scale=2; $orig_size / $comp_size" | bc)
            printf "%-20s: %8d -> %8d bytes (ratio: %.2fx)\n" "$(basename "$original")" $orig_size $comp_size $ratio
        fi
    fi
done

echo ""
echo "✅ Benchmark completed! Compare FilesCanFly's performance against standard LZ4."