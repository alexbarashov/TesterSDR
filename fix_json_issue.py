#!/usr/bin/env python3
"""
Fix JSON serialization issue in beacon_tester_web.py
"""
import re

def fix_json_serialization():
    """Fix numpy int64 JSON serialization issue"""
    file_path = 'beacon406/beacon_tester_web.py'

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix the problematic lines
    fixes = [
        ("'total_written': iq_ring_buffer.total_written if iq_ring_buffer else 0,",
         "'total_written': int(iq_ring_buffer.total_written) if iq_ring_buffer else 0,"),
        ("'capacity': iq_ring_buffer.capacity if iq_ring_buffer else 0,",
         "'capacity': int(iq_ring_buffer.capacity) if iq_ring_buffer else 0,"),
        ("'write_pos': iq_ring_buffer.write_pos if iq_ring_buffer else 0,",
         "'write_pos': int(iq_ring_buffer.write_pos) if iq_ring_buffer else 0,")
    ]

    modified = False
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            modified = True
            print(f"Fixed: {old[:50]}...")

    if modified:
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✓ JSON serialization issue fixed!")
    else:
        print("× No fixes needed or patterns not found")

if __name__ == "__main__":
    fix_json_serialization()