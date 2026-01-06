#!/usr/bin/env python3
"""
Custom boolean comparison style checker for C++ code.
Enforces the following rules:
- Use 'condition == false' instead of '!condition' for boolean variables
- Use 'ptr != nullptr' instead of '!ptr' for pointer variables
- Use 'func() == false' instead of '!func()' for function calls
- Use 'condition' instead of 'condition == true'
"""

import re
import sys


def check_boolean_style(content):
    """Check for boolean comparison style violations"""
    issues = []

    # Pattern for !boolean_variable (should be boolean_var == false)
    not_bool_pattern = r"!\s*([a-zA-Z_][a-zA-Z0-9_]*)"

    # Pattern for condition == true (should be just condition)
    true_compare_pattern = r"(\w+)\s*==\s*true"

    for line_num, line in enumerate(content.split("\n"), 1):
        # Skip preprocessor directives
        stripped_line = line.strip()
        if stripped_line.startswith("#"):
            continue

        # Check for !boolean_variable usage
        if re.search(not_bool_pattern, line):
            match = re.search(not_bool_pattern, line)
            if match:
                var_name = match.group(1)
                # Check if it's a pointer check (!ptr) - should use != nullptr
                pointer_names = [
                    "file",
                    "ptr",
                    "data",
                    "buffer",
                    "context",
                    "handle",
                    "stream",
                ]
                is_pointer_check = (
                    var_name in pointer_names  # Common pointer variable names
                    or var_name.endswith("_ptr")  # Variables ending with _ptr
                    or var_name.startswith("p_")  # Variables starting with p_
                )

                is_function_call = bool(re.search(r"!\s*\w+\s*\(", line))

                if is_pointer_check:
                    # Pointer check - suggest using != nullptr
                    issues.append(
                        f"Line {line_num}: Use '{var_name} != nullptr' instead of '!{var_name}'"
                    )
                elif is_function_call:
                    # Function call - suggest using == false
                    issues.append(
                        f"Line {line_num}: Use '{var_name}() == false' instead of '!{var_name}()'"
                    )
                else:
                    # Boolean variable - suggest using == false
                    issues.append(
                        f"Line {line_num}: Use '{var_name} == false' instead of '!{var_name}'"
                    )

        # Check for == true usage
        if re.search(true_compare_pattern, line):
            match = re.search(true_compare_pattern, line)
            if match:
                var_name = match.group(1)
                issues.append(
                    f"Line {line_num}: Use '{var_name}' instead of '{var_name} == true'"
                )

    return issues


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/custom_boolean_check.py <file>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        with open(filename, "r") as f:
            content = f.read()

        issues = check_boolean_style(content)
        for issue in issues:
            print(f"{filename}:{issue}")

        if not issues:
            print(f"{filename}: No boolean style violations found")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
