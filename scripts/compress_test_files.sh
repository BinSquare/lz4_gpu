#!/bin/bash
# Compress test files with standard LZ4 tool

cd /Users/binsquare/Documents/FilesCanFly

echo "Creating compressed versions of test files..."

# Create directory for compressed files
mkdir -p test_data/lz4_compressed

# Loop through each test file and compress it
for file in test_data/*.txt test_data/*.dat test_data/*.img test_data/*.dll test_data/*.bin xml; do
  if [ -f "$file" ]; then
    echo "Compressing $file..."
    lz4 -1 "$file" "test_data/lz4_compressed/$(basename "$file").lz4"
  fi
done

# Handle all the specific files
lz4 -1 test_data/dickens.txt test_data/lz4_compressed/dickens.txt.lz4
lz4 -1 test_data/mozilla.dat test_data/lz4_compressed/mozilla.dat.lz4
lz4 -1 test_data/mr.img test_data/lz4_compressed/mr.img.lz4
lz4 -1 test_data/nci.txt test_data/lz4_compressed/nci.txt.lz4
lz4 -1 test_data/ooffice.dll test_data/lz4_compressed/ooffice.dll.lz4
lz4 -1 test_data/osdb.bin test_data/lz4_compressed/osdb.bin.lz4
lz4 -1 test_data/reymont.txt test_data/lz4_compressed/reymont.txt.lz4
lz4 -1 test_data/samba.txt test_data/lz4_compressed/samba.txt.lz4
lz4 -1 test_data/sao.bin test_data/lz4_compressed/sao.bin.lz4
lz4 -1 test_data/webster.txt test_data/lz4_compressed/webster.txt.lz4
lz4 -1 test_data/xml.xml test_data/lz4_compressed/xml.xml.lz4
lz4 -1 test_data/xray.img test_data/lz4_compressed/xray.img.lz4

echo "All test files compressed successfully!"
echo "Compressed files are in test_data/lz4_compressed/"