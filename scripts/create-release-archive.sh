#!/bin/bash

# Script to create release archive for etx-tracer
# Creates a zip containing runtime files plus the Blender plugin.

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Creating release archive for etx-tracer..."
echo "Project root: $PROJECT_ROOT"
echo "Excluding: assets_testing/, lib/, tmp/"

# Create temporary directory for packaging
TEMP_DIR=$(mktemp -d)
RELEASE_DIR="$TEMP_DIR/etx-tracer-release"
trap 'rm -rf "$TEMP_DIR"' EXIT

echo "Using temporary directory: $TEMP_DIR"

# Create release directory
mkdir -p "$RELEASE_DIR"

if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
    APP_BUNDLE="$PROJECT_ROOT/bin/ETX Tracer.app"
    if [[ ! -d "$APP_BUNDLE" ]]; then
        echo "Error: ETX Tracer.app was not built"
        exit 1
    fi
    if ! codesign --verify --deep --strict "$APP_BUNDLE"; then
        echo "Error: ETX Tracer.app has an invalid code signature"
        exit 1
    fi
    echo "Copying signed macOS application bundle..."
    ditto "$APP_BUNDLE" "$RELEASE_DIR/ETX Tracer.app"
else
    PLATFORM="linux"
    echo "Copying runtime data and compiled shader package..."
    cd "$PROJECT_ROOT/bin"
    for item in *; do
        if [[ "$item" == "assets" || "$item" == "fonts" || "$item" == "spectrum" ||
              "$item" == "shaders.etxpack" ]]; then
            echo "  Copying: $item"
            cp -r "$item" "$RELEASE_DIR/"
        fi
    done
    if [[ ! -f "$RELEASE_DIR/shaders.etxpack" ]]; then
        echo "Error: shaders.etxpack was not generated"
        exit 1
    fi
    if [[ ! -f "$PROJECT_ROOT/bin/raytracer" ]]; then
        echo "Error: raytracer was not built"
        exit 1
    fi
    cp "$PROJECT_ROOT/bin/raytracer" "$RELEASE_DIR/"
fi

# Create zip of blender plugin
echo "Creating blender plugin archive..."
cd "$PROJECT_ROOT"
mkdir -p "$RELEASE_DIR/blender"
BLENDER_ZIP="$RELEASE_DIR/blender/etx_tracer_exporter.zip"
cd blender
zip -r "$BLENDER_ZIP" etx_tracer_exporter/ -x '*/__pycache__/*' '*/.DS_Store' '*.pyc'
echo "  Created: blender/etx_tracer_exporter.zip"

# Create final release archive
cd "$TEMP_DIR"
ARCHIVE_NAME="etx-tracer-$PLATFORM-$(date +%Y%m%d-%H%M%S).zip"
echo "Creating final archive: $ARCHIVE_NAME"
if [[ "$PLATFORM" == "macos" ]]; then
    ditto -c -k --sequesterRsrc --keepParent etx-tracer-release "$ARCHIVE_NAME"
else
    zip -r "$ARCHIVE_NAME" etx-tracer-release/
fi

# Move archive to project root
mv "$ARCHIVE_NAME" "$PROJECT_ROOT/"
echo "Archive created: $PROJECT_ROOT/$ARCHIVE_NAME"

echo "Release archive creation complete!"
echo "Archive: $PROJECT_ROOT/$ARCHIVE_NAME"
echo ""
echo "Contents:"
unzip -l "$PROJECT_ROOT/$ARCHIVE_NAME" | head -20
