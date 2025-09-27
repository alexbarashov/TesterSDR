#!/usr/bin/env python3
"""
Fix remaining JSON serialization issues in beacon_tester_web.py
"""

def fix_remaining_json_serialization():
    """Fix remaining numpy int64 JSON serialization issues"""
    file_path = 'beacon406/beacon_tester_web.py'

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix the remaining problematic lines
    fixes = [
        ("'sample_counter': sample_counter,",
         "'sample_counter': int(sample_counter),"),
        ("'bp_sample_counter': bp_sample_counter,",
         "'bp_sample_counter': int(bp_sample_counter),"),
        ("'pulse_start_abs': pulse_start_abs",
         "'pulse_start_abs': int(pulse_start_abs)")
    ]

    modified = False
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new)
            modified = True
            print(f"Fixed: {old[:30]}...")

    if modified:
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("OK Additional JSON serialization issues fixed!")
    else:
        print("× No additional fixes needed or patterns not found")

if __name__ == "__main__":
    fix_remaining_json_serialization()