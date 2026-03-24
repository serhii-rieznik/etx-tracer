#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


RETURN_SEMANTIC_LINE_RE = re.compile(
    r"^(?P<indent>\s*):\s*(?P<semantic>[A-Za-z_][A-Za-z0-9_]*)(?P<suffix>(?:\s*\{)?(?:\s*//.*)?)$"
)


def fix_hlsl_return_semantics(text: str) -> str:
    has_trailing_newline = text.endswith(("\n", "\r"))
    lines = text.splitlines()
    fixed_lines: list[str] = []

    index = 0
    while index < len(lines):
        line = lines[index]
        if index + 1 < len(lines) and _should_join_return_semantic(line, lines[index + 1]):
            fixed_lines.append(f"{line.rstrip()} {lines[index + 1].lstrip()}")
            index += 2
            continue

        fixed_lines.append(line)
        index += 1

    fixed = "\n".join(fixed_lines)
    if has_trailing_newline:
        fixed += "\n"

    return fixed


def process_file(path: Path) -> bool:
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    original = raw.decode("utf-8")
    fixed = fix_hlsl_return_semantics(original)
    if fixed == original:
        return False

    path.write_bytes(fixed.replace("\n", newline).encode("utf-8"))
    return True


def _should_join_return_semantic(signature_line: str, semantic_line: str) -> bool:
    if not signature_line.rstrip().endswith(")"):
        return False

    return RETURN_SEMANTIC_LINE_RE.match(semantic_line) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix HLSL function return semantics after clang-format.")
    parser.add_argument("paths", nargs="+", help="HLSL files to rewrite in place.")
    args = parser.parse_args()

    for path_str in args.paths:
        path = Path(path_str)
        if path.suffix.lower() != ".hlsl":
            continue
        process_file(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
