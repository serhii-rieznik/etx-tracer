# Vendored DXC Layout

This folder is used by `cmake/FindDXC.cmake` to auto-discover a locally vendored
DirectX Shader Compiler (DXC) package.

## Required package structure

For each platform, provide a folder that contains:

- `include/dxc/dxcapi.h` (required)
- DXC runtime library (required):
  - Windows: `lib/dxcompiler.dll` or `bin/dxcompiler.dll`
  - macOS: `lib/libdxcompiler.dylib` or `bin/libdxcompiler.dylib`
  - Linux: `lib/libdxcompiler.so` or `bin/libdxcompiler.so`
- `bin/dxc` (optional, `dxc.exe` on Windows)
- sidecar DXIL library (optional, copied if present):
  - Windows: `dxil.dll`
  - macOS: `libdxil.dylib`
  - Linux: `libdxil.so`

## Recommended folder names

Use one of these names under `thirdparty/dxc/`:

- macOS (Apple Silicon): `macos-arm64`
- macOS (Intel): `macos-x64`
- Windows x64: `windows-x64`
- Linux x64: `linux-x64`

Other aliases also supported by CMake:

- macOS: `macos`, `darwin`, `darwin-arm64`
- Windows: `windows`, `win`, `win-x64`
- Linux: `linux`

## Example layouts

### macOS arm64

```text
thirdparty/dxc/macos-arm64/
  include/dxc/dxcapi.h
  lib/libdxcompiler.dylib
  lib/libdxil.dylib            # optional
  bin/dxc                      # optional
```

### Windows x64

```text
thirdparty/dxc/windows-x64/
  include/dxc/dxcapi.h
  bin/dxcompiler.dll
  bin/dxil.dll                 # optional
  bin/dxc.exe                  # optional
```

### Linux x64

```text
thirdparty/dxc/linux-x64/
  include/dxc/dxcapi.h
  lib/libdxcompiler.so
  lib/libdxil.so               # optional
  bin/dxc                      # optional
```

## Build workflow

1. Build or obtain DXC for your platform.
2. Copy headers and binaries into one folder under `thirdparty/dxc/` as above.
3. Configure/build with CMake.

If CMake still does not find DXC, set `DXC_PATH` to your package root:

```bash
cmake -S . -B build -DDXC_PATH=/absolute/path/to/dxc/package
```

## Distribution behavior

When DXC is found, the `raytracer` post-build step copies:

- DXC runtime library
- DXC executable (if present)
- DXIL sidecar library (if present)

into the project `bin/` folder for distribution.
