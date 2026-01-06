#!/usr/bin/env python3
"""
Batch boolean comparison style checker for all C++ source files.
Scans the entire sources/ directory (excluding thirdparty) and reports all style violations.
Console shows progress with pass/fail indicators.
Log file 'boolean_style_check_results.txt' contains ONLY files with violations (no clean files).
"""

import os
import sys
import subprocess
from pathlib import Path


def find_cpp_files():
    """Find all C++ source files in sources/ directory, excluding thirdparty"""
    sources_dir = Path("sources")
    cpp_files = []

    if not sources_dir.exists():
        print("Error: sources/ directory not found")
        return []

    # Extensions to check
    extensions = [".cxx", ".cpp", ".cc", ".hxx", ".hpp", ".h"]

    for ext in extensions:
        # Find all files with this extension
        for file_path in sources_dir.rglob(f"*{ext}"):
            # Skip thirdparty directory
            if "thirdparty" not in str(file_path):
                cpp_files.append(file_path)

    return sorted(cpp_files)


def run_boolean_check(file_path):
    """Run the boolean style checker on a single file and return violations"""
    try:
        result = subprocess.run(
            [sys.executable, "scripts/custom_boolean_check.py", str(file_path)],
            capture_output=True,
            text=True,
            cwd=".",
        )

        violations = []
        for line in result.stdout.strip().split("\n"):
            if line.strip() and "No boolean style violations found" not in line:
                violations.append(line)

        return violations

    except Exception as e:
        return [f"Error checking {file_path}: {e}"]


def log_and_print(message, log_file=None):
    """Print to console and optionally write to log file"""
    print(message)
    if log_file:
        with open(log_file, "a") as f:
            f.write(message + "\n")


def main():
    # Log file will be created only if violations are found
    log_file = None  # Will be set when first violation is found

    log_and_print("Checking boolean comparison style across all source files...", None)
    log_and_print("=" * 70, None)

    cpp_files = find_cpp_files()

    if not cpp_files:
        log_and_print("No C++ source files found in sources/ directory", log_file)
        return

    log_and_print(f"Found {len(cpp_files)} C++ source files to check", log_file)
    log_and_print("", log_file)

    total_violations = 0
    files_with_violations = 0
    files_checked = 0

    for file_path in cpp_files:
        files_checked += 1
        print(f"[{files_checked:2d}/{len(cpp_files):2d}] Checking {file_path}")

        violations = run_boolean_check(file_path)

        if violations:
            # Initialize log file on first violation
            if log_file is None:
                log_file = "boolean_style_check_results.txt"
                with open(log_file, "w") as f:
                    f.write("Boolean Comparison Style Check Results\n")
                    f.write("=" * 70 + "\n")
                    f.write(f"Generated: {Path.cwd()}\n\n")

            files_with_violations += 1
            total_violations += len(violations)

            # Log detailed info to file
            log_and_print(
                f"[{files_checked:2d}/{len(cpp_files):2d}] Checking {file_path}",
                log_file,
            )
            log_and_print(f"  [FAIL] {len(violations)} violations found:", log_file)
            for violation in violations:
                log_and_print(f"     {violation}", log_file)
            log_and_print("", log_file)

            # Console summary only
            print(f"  [FAIL] {len(violations)} violations found:")
        else:
            print("  [PASS] No violations")

    # Summary
    if total_violations == 0:
        print()
        print("SUCCESS: All files pass boolean comparison style checks!")
        return 0
    else:
        # Only show summary when there are violations
        print()
        log_and_print("=" * 70, log_file)
        log_and_print("SUMMARY", log_file)
        log_and_print(f"   Files checked: {files_checked}", log_file)
        log_and_print(f"   Files with violations: {files_with_violations}", log_file)
        log_and_print(f"   Total violations: {total_violations}", log_file)
        log_and_print("", log_file)
        log_and_print(f"Results saved to: {log_file}", log_file)
        warning_msg = "WARNING: Style violations found. Consider fixing them for better code consistency."
        log_and_print(warning_msg, log_file)
        return 1


if __name__ == "__main__":
    sys.exit(main())
