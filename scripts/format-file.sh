#!/bin/sh

clang-format -verbose -style=file -i "$1"

case "$1" in
  *.hlsl|*.HLSL)
    python3 "$(dirname "$0")/fix_hlsl_semantics.py" "$1"
    ;;
esac
