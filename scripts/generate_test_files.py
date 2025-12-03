#!/usr/bin/env python3
"""
Generate standard test files for LZ4 benchmarking.
This script creates various types of files similar to the Silesia corpus.
"""

import os
import random
import string
import struct
import numpy as np
from pathlib import Path

def create_text_file(file_path, size_mb=1):
    """Create a text file with realistic content."""
    # Generate text similar to English novels
    words = [
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", 
        "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
        "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy",
        "did", "man", "men", "run", "too", "any", "big", "eat", "him", "job",
        "lot", "put", "say", "she", "try", "use", "way", "will", "word", "work",
        "back", "good", "have", "here", "home", "know", "life", "long", "make",
        "much", "must", "over", "take", "tell", "went", "with", "after", "again",
        "could", "every", "first", "found", "great", "house", "large", "learn",
        "never", "place", "small", "sound", "spell", "still", "study", "their",
        "there", "these", "thing", "think", "those", "three", "water", "where",
        "which", "world", "would", "write", "years", "about", "above", "across",
        "added", "after", "again", "against", "along", "also", "always", "among",
        "another", "around", "because", "before", "below", "between", "called",
        "cannot", "change", "different", "example", "first", "followed", "found",
        "great", "group", "hand", "helped", "important", "large", "later", "left",
        "like", "long", "made", "main", "make", "many", "may", "mean", "men",
        "might", "more", "most", "move", "much", "must", "named", "never",
        "next", "number", "often", "old", "part", "play", "read", "right",
        "said", "same", "saw", "say", "school", "set", "should", "show", "small",
        "some", "sometimes", "still", "such", "take", "tell", "than", "that",
        "them", "then", "there", "these", "they", "thing", "think", "this",
        "those", "thought", "three", "through", "time", "together", "too",
        "took", "toward", "two", "under", "until", "upon", "use", "very",
        "want", "water", "way", "well", "went", "were", "what", "when",
        "where", "which", "while", "with", "without", "work", "world", "would",
        "write", "year", "young"
    ]
    
    content = []
    target_size = size_mb * 1024 * 1024  # MB to bytes
    
    while len(' '.join(content)) < target_size:
        sentence_length = random.randint(5, 15)
        sentence = ' '.join([random.choice(words) for _ in range(sentence_length)])
        # Add proper sentence structure
        sentence = sentence[0].upper() + sentence[1:] + '. '
        content.append(sentence)
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(content)[:target_size])
    print(f"Created text file: {file_path} ({os.path.getsize(file_path)} bytes)")


def create_dictionary_file(file_path, size_mb=1):
    """Create a dictionary-like file with sorted entries."""
    # Generate content similar to a dictionary
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    words = []
    
    for letter in alphabet:
        for i in range(500):  # Generate ~500 words per letter
            word_length = random.randint(4, 10)
            word = letter + ''.join(random.choices(string.ascii_lowercase, k=word_length-1))
            definition_length = random.randint(10, 50)
            definition = ' '.join(random.choices(alphabet * 3, k=definition_length))
            words.append(f"{word.upper()}: {definition}")
    
    content = '\n'.join(words)
    target_size = size_mb * 1024 * 1024  # MB to bytes
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content[:target_size])
    print(f"Created dictionary file: {file_path} ({os.path.getsize(file_path)} bytes)")


def create_binary_file(file_path, size_mb=1):
    """Create a binary file with structured data."""
    target_size = size_mb * 1024 * 1024  # MB to bytes
    
    # Create structured binary data similar to databases
    with open(file_path, 'wb') as f:
        record_size = 128  # bytes per record
        num_records = target_size // record_size
        
        for i in range(num_records):
            # Create a structured record
            # ID (int32)
            f.write(struct.pack('<I', i))
            # Name (32 bytes)
            name = f"record_{i:08d}".encode('ascii')[:31].ljust(31, b'\x00') + b'\x00'
            f.write(name)
            # Timestamp (int64)
            f.write(struct.pack('<Q', 1000000 + i))
            # Value (float32)
            f.write(struct.pack('<f', 3.14159 * i))
            # Flags (int32)
            f.write(struct.pack('<I', random.randint(0, 0xFFFF)))
            # Padding to reach record size
            padding = record_size - 51  # already used 51 bytes
            f.write(b'\x00' * padding)
    
    print(f"Created binary file: {file_path} ({os.path.getsize(file_path)} bytes)")


def create_xml_file(file_path, size_mb=1):
    """Create an XML file with structured content."""
    target_size = size_mb * 1024 * 1024  # MB to bytes
    
    elements = [
        "item", "product", "user", "record", "data", "entry", 
        "object", "element", "node", "value", "field", "property"
    ]
    attributes = [
        "id", "name", "value", "type", "class", "ref", 
        "status", "timestamp", "version", "category"
    ]
    
    content = ['<?xml version="1.0" encoding="UTF-8"?>\n<root>\n']
    
    while len(''.join(content)) < target_size * 0.9:  # Leave some space for closing
        element = random.choice(elements)
        attr1 = random.choice(attributes)
        attr2 = random.choice(attributes)
        
        # Create an XML element with attributes
        xml_line = f'  <{element} {attr1}="{random.randint(1, 10000)}" {attr2}="{random.choice(["active", "inactive", "pending"])}">'
        xml_line += f'{"".join(random.choices(string.ascii_letters + " ", k=random.randint(10, 50)))}</{element}>\n'
        content.append(xml_line)
    
    content.append('</root>\n')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(''.join(content)[:target_size])
    
    print(f"Created XML file: {file_path} ({os.path.getsize(file_path)} bytes)")


def create_executable_like_file(file_path, size_mb=1):
    """Create a file that looks like executable code."""
    target_size = size_mb * 1024 * 1024  # MB to bytes
    
    # Generate pseudo-assembly-like content with repeated patterns
    opcodes = ["MOV", "ADD", "SUB", "JMP", "CMP", "CALL", "RET", "PUSH", "POP"]
    registers = ["RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "RBP", "RSP"]
    
    content = []
    while len(''.join(content)) < target_size:
        opcode = random.choice(opcodes)
        reg1 = random.choice(registers)
        reg2 = random.choice(registers)
        # Create assembly-like instruction
        line = f"{opcode} {reg1}, {reg2}\n"
        if random.random() > 0.7:  # Add some comments
            line += f"; This is a comment about {opcode}\n"
        content.append(line)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(''.join(content)[:target_size])
    
    print(f"Created executable-like file: {file_path} ({os.path.getsize(file_path)} bytes)")


def create_image_like_file(file_path, size_mb=1):
    """Create image-like binary data with patterns."""
    target_size = size_mb * 1024 * 1024  # MB to bytes
    
    # Create image-like data with spatial correlation
    with open(file_path, 'wb') as f:
        # Write a simple header
        f.write(b"IMG_HDR")  # Simple header
        f.write(struct.pack('<I', target_size))  # Size info
        
        # Generate pixel-like data with some correlation
        last_value = 128
        for i in range(target_size - 9):  # Account for header
            # Create value with some correlation to previous value
            change = random.randint(-10, 10)
            new_value = max(0, min(255, last_value + change))
            f.write(struct.pack('B', new_value))
            last_value = new_value
    
    print(f"Created image-like file: {file_path} ({os.path.getsize(file_path)} bytes)")


def main():
    """Generate standard test files for LZ4 benchmarking."""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    print("Generating standard test files for LZ4 benchmarking...")
    
    # Create different types of files similar to Silesia corpus
    create_text_file(test_dir / "dickens.txt", size_mb=10)  # Novel-like text
    create_executable_like_file(test_dir / "mozilla.dat", size_mb=5)  # Executable-like
    create_image_like_file(test_dir / "mr.img", size_mb=10)  # Image-like (medical)
    create_text_file(test_dir / "nci.txt", size_mb=5)  # Chemical database like
    create_executable_like_file(test_dir / "ooffice.dll", size_mb=6)  # DLL-like
    create_binary_file(test_dir / "osdb.bin", size_mb=10)  # Database-like
    create_text_file(test_dir / "reymont.txt", size_mb=7)  # Text in other language
    create_text_file(test_dir / "samba.txt", size_mb=21)  # Source code like
    create_binary_file(test_dir / "sao.bin", size_mb=7)  # Star catalog like
    create_dictionary_file(test_dir / "webster.txt", size_mb=41)  # Dictionary
    create_image_like_file(test_dir / "xray.img", size_mb=8)  # X-ray image
    create_xml_file(test_dir / "xml.xml", size_mb=5)  # XML data
    
    print(f"\nAll test files created in {test_dir}/")
    print("Files created:")
    for file_path in sorted(test_dir.glob("*")):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  {file_path.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()