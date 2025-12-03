# FilesCanFly Test Data

This directory contains test files for benchmarking FilesCanFly's LZ4 decompression performance.

## File Structure

- `test_data/*.txt|*.dat|*.img|*.dll|*.bin|*.xml` - Original uncompressed test files
- `test_data/lz4_compressed/*.lz4` - LZ4 compressed versions of test files

## Test Files (Similar to Silesia Corpus)

These files represent different types of data commonly used in compression benchmarking:

1. **dickens.txt** (10.0 MB) - Novel-like English text
2. **mozilla.dat** (5.0 MB) - Executable-like binary data  
3. **mr.img** (10.0 MB) - Image-like data (medical imaging)
4. **nci.txt** (5.0 MB) - Text similar to chemical database
5. **ooffice.dll** (6.0 MB) - DLL-like executable data
6. **osdb.bin** (10.1 MB) - Database-like structured binary
7. **reymont.txt** (7.0 MB) - Text in other language style
8. **samba.txt** (21.0 MB) - Source code-like text
9. **sao.bin** (7.1 MB) - Star catalog-like binary data
10. **webster.txt** (0.9 MB) - Dictionary-like content
11. **xml.xml** (4.5 MB) - XML structured data
12. **xray.img** (8.0 MB) - X-ray image-like data

## Benchmarking

To benchmark FilesCanFly against standard LZ4 tools:

```bash
# Test FilesCanFly decompression performance
cargo run --release

# Test standard LZ4 decompression performance
lz4 -b1 -d test_data/lz4_compressed/dickens.txt.lz4

# Use lzbench for comprehensive comparison
lzbench -elz4 test_data/dickens.txt
```