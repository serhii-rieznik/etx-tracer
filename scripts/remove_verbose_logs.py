#!/usr/bin/env python3
"""
Remove verbose log::info and log::debug calls from RHI files, keeping only warnings and errors.
"""

import re
import sys
from pathlib import Path

def remove_verbose_logs(content: str) -> str:
    """
    Remove log::info and log::debug calls while preserving log::warning and log::error.
    """
    # Pattern to match log::info and log::debug calls
    # This matches: log::info( or log::debug(
    # We need to be careful about multi-line calls and nested parentheses

    lines = content.split('\n')
    result_lines = []

    for line in lines:
        # Remove log::info and log::debug calls
        # Simple approach: remove lines that start with log::info or log::debug
        # More complex calls might span multiple lines, but this covers most cases

        stripped = line.strip()

        # Skip lines that are just log::info or log::debug calls
        if (stripped.startswith('log::info(') or
            stripped.startswith('log::debug(') or
            stripped.startswith('} else log::info(') or  # Handle } else log::info( patterns
            stripped.startswith('} else log::debug(')):
            continue

        # Handle cases where log call is at the end of a line with other code
        # Look for log::info( and log::debug( anywhere in the line
        if 'log::info(' in line or 'log::debug(' in line:
            # This is more complex - we need to remove just the log call
            # For now, let's use a regex to remove these patterns
            # Remove log::info(...) and log::debug(...)
            line = re.sub(r'log::info\([^)]*\);?', '', line)
            line = re.sub(r'log::debug\([^)]*\);?', '', line)

            # Clean up any resulting empty lines or trailing whitespace
            line = line.rstrip()
            if not line.strip():
                continue

        result_lines.append(line)

    # Remove excessive empty lines
    cleaned_lines = []
    prev_empty = False

    for line in result_lines:
        is_empty = not line.strip()
        if not (is_empty and prev_empty):
            cleaned_lines.append(line)
        prev_empty = is_empty

    return '\n'.join(cleaned_lines).rstrip() + '\n'

def process_file(file_path: Path) -> bool:
    """Process a single C++ file to remove verbose logs."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Remove verbose logs
        cleaned_content = remove_verbose_logs(content)

        # Write back only if changed
        if cleaned_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"Processed: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_verbose_logs.py <file_or_directory> [file_or_directory ...]")
        print("Examples:")
        print("  python remove_verbose_logs.py sources/etx/rhi/vulkan/vk_device.cxx")
        print("  python remove_verbose_logs.py sources/etx/rhi/")
        sys.exit(1)

    cpp_extensions = {'.cpp', '.cxx', '.cc', '.c++', '.h', '.hpp', '.hxx', '.h++'}

    processed_count = 0
    changed_count = 0

    for arg in sys.argv[1:]:
        path = Path(arg)

        if path.is_file():
            if path.suffix.lower() in cpp_extensions:
                processed_count += 1
                if process_file(path):
                    changed_count += 1
            else:
                print(f"Skipping non-C++ file: {path}")

        elif path.is_dir():
            for file_path in path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in cpp_extensions:
                    processed_count += 1
                    if process_file(file_path):
                        changed_count += 1
        else:
            # Try glob pattern
            import glob
            matches = glob.glob(str(path))
            for match in matches:
                file_path = Path(match)
                if file_path.is_file() and file_path.suffix.lower() in cpp_extensions:
                    processed_count += 1
                    if process_file(file_path):
                        changed_count += 1

    print(f"\nSummary: Processed {processed_count} files, changed {changed_count} files")

if __name__ == '__main__':
    main()