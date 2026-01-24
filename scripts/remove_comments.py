#!/usr/bin/env python3
"""
Smart C++ comment remover that preserves strings and code structure.
"""

import re
import sys
from pathlib import Path

def remove_comments_from_cpp(content: str) -> str:
    """
    Remove C++ comments while preserving strings and code structure.
    Uses a more robust approach to handle strings correctly.
    """
    # Split into tokens while preserving structure
    # This is still simplified but handles strings better

    result = []
    i = 0
    in_multiline_comment = False

    while i < len(content):
        # Handle multiline comments
        if in_multiline_comment:
            # Look for end of multiline comment
            end_pos = content.find('*/', i)
            if end_pos != -1:
                # Skip the entire multiline comment
                i = end_pos + 2
                in_multiline_comment = False
            else:
                # End of file reached while in multiline comment
                break
            continue

        # Check for start of multiline comment
        if i < len(content) - 1 and content[i:i+2] == '/*':
            in_multiline_comment = True
            i += 2
            continue

        # Check for single-line comment (but not in strings)
        if i < len(content) - 1 and content[i:i+2] == '//':
            # Find end of line
            line_end = content.find('\n', i)
            if line_end == -1:
                # End of file
                break
            # Skip to end of line
            i = line_end
            continue

        # Handle string literals more carefully
        if content[i] in ['"', "'"]:
            quote_char = content[i]
            result.append(quote_char)
            i += 1

            # Find the matching closing quote
            while i < len(content):
                if content[i] == quote_char:
                    # Check if it's escaped
                    escape_count = 0
                    j = i - 1
                    while j >= 0 and content[j] == '\\':
                        escape_count += 1
                        j -= 1

                    if escape_count % 2 == 0:  # Not escaped
                        result.append(quote_char)
                        i += 1
                        break

                result.append(content[i])
                i += 1
            continue

        # Regular character
        result.append(content[i])
        i += 1

    # Join and clean up
    cleaned = ''.join(result)

    # Remove excessive empty lines
    lines = cleaned.split('\n')
    result_lines = []

    prev_empty = False
    for line in lines:
        is_empty = not line.strip()
        if not (is_empty and prev_empty):
            result_lines.append(line)
        prev_empty = is_empty

    # Remove trailing empty lines
    while result_lines and not result_lines[-1].strip():
        result_lines.pop()

    return '\n'.join(result_lines) + '\n'

def process_file(file_path: Path) -> bool:
    """Process a single C++ file to remove comments."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Remove comments
        cleaned_content = remove_comments_from_cpp(content)

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
        print("Usage: python remove_comments.py <file_or_directory> [file_or_directory ...]")
        print("Examples:")
        print("  python remove_comments.py file.cpp")
        print("  python remove_comments.py sources/etx/rhi/")
        print("  python remove_comments.py *.hxx *.cxx")
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