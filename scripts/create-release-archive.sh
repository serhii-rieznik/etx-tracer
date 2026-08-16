#!/bin/bash

# Script to create release archive for etx-tracer
# Creates a zip containing bin/ contents (except assets_testing, lib, and tmp) plus blender plugin in blender/ folder

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Creating release archive for etx-tracer..."
echo "Project root: $PROJECT_ROOT"
echo "Excluding: assets_testing/, lib/, tmp/"

# Create temporary directory for packaging
TEMP_DIR=$(mktemp -d)
RELEASE_DIR="$TEMP_DIR/etx-tracer-release"

echo "Using temporary directory: $TEMP_DIR"

# Create release directory
mkdir -p "$RELEASE_DIR"

# Copy only necessary directories from bin/
echo "Copying bin/ contents (including assets and runtime shaders)..."
cd "$PROJECT_ROOT/bin"
for item in *; do
    if [[ "$item" == "assets" || "$item" == "fonts" || "$item" == "spectrum" ||
          "$item" == "shaders" || "$item" == "interop" || "$item" == "access" ]]; then
        echo "  Copying: $item"
        cp -r "$item" "$RELEASE_DIR/"
    fi
done

# Copy necessary files (executables, libraries, configs)
echo "Copying executables, libraries, and config files..."
for item in *.exe *.dylib *.dll *.json; do
    if [[ -f "$item" ]]; then
        echo "  Copying: $item"
        cp "$item" "$RELEASE_DIR/"
    fi
done

# Also copy raytracer binary (may not have extension on macOS)
if [[ -f "raytracer" ]]; then
    echo "  Copying: raytracer"
    cp "raytracer" "$RELEASE_DIR/"
fi

# Create zip of blender plugin
echo "Creating blender plugin archive..."
cd "$PROJECT_ROOT"
mkdir -p "$RELEASE_DIR/blender"
BLENDER_ZIP="$RELEASE_DIR/blender/etx_tracer_exporter.zip"
cd blender
zip -r "$BLENDER_ZIP" etx_tracer_exporter/
echo "  Created: blender/etx_tracer_exporter.zip"

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
else
    PLATFORM="unknown"
fi

# Create final release archive
cd "$TEMP_DIR"
ARCHIVE_NAME="etx-tracer-$PLATFORM-$(date +%Y%m%d-%H%M%S).zip"
echo "Creating final archive: $ARCHIVE_NAME"
zip -r "$ARCHIVE_NAME" etx-tracer-release/

# Move archive to project root
mv "$ARCHIVE_NAME" "$PROJECT_ROOT/"
echo "Archive created: $PROJECT_ROOT/$ARCHIVE_NAME"

# Cleanup
rm -rf "$TEMP_DIR"

echo "Release archive creation complete!"
echo "Archive: $PROJECT_ROOT/$ARCHIVE_NAME"
echo ""
echo "Contents:"
unzip -l "$PROJECT_ROOT/$ARCHIVE_NAME" | head -20
