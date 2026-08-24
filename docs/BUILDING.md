# Building etx-tracer

This description will be updated during the development process.

## Requirements

Most of external libraries will be located directly in the source code, to reduce a number of dependencies and make building faster.
These libraries and tools you have to install by yourself:
- CMake
- [Intel Embree](https://www.embree.org/) for CPU ray-tracing
- [DirectX Shader Compiler (DXC)](https://github.com/microsoft/DirectXShaderCompiler) for HLSL-to-SPIR-V compilation

## Building for Windows
- download and install the latest release of Intel Embree from [GitHub](https://github.com/embree/embree/releases);
  - add environment variable `EMBREE_LOCATION` pointing to the Embree installation folder provide this parameter to CMake (i.e `cmake -DEMBREE_LOCATION=path/to/embree`);
  - copy embree binaries (embree4.dll and other required dlls) to the `bin` folder in the root directory of `etx-tracer`

### DXC (DirectX Shader Compiler)
DXC is required at build time for compiling HLSL shaders. CMake will automatically:
1. Check if DXC is already installed on your system
2. If not found on Windows, download a DXC release from GitHub
3. Copy DXC binaries to the developer `bin` folder for shader-package generation

You can also install DXC manually using:
- WinGet: `winget install -e --id Microsoft.DirectXShaderCompiler`
- Chocolatey: `choco install directxshadercompiler`
- Or download directly from [GitHub releases](https://github.com/microsoft/DirectXShaderCompiler/releases)

If DXC is installed in a custom location, set `DXC_PATH` to the installation prefix (the folder that contains `bin/`, `lib/`, and `include/`).

You can also vendor DXC directly in this repo. CMake now auto-prefers `thirdparty/dxc` with platform folders, for example:
- `thirdparty/dxc/macos-arm64/include/dxc/dxcapi.h`
- `thirdparty/dxc/macos-arm64/lib/libdxcompiler.dylib`
- optional: `thirdparty/dxc/macos-arm64/lib/libdxil.dylib`
- optional: `thirdparty/dxc/macos-arm64/bin/dxc`

Other recognized folder names include:
- macOS: `macos`, `darwin`, `macos-arm64`, `darwin-arm64`
- Windows: `windows`, `win`, `windows-x64`, `win-x64`
- Linux: `linux`, `linux-x64`

When DXC is found (system, `DXC_PATH`, or vendored path), the build copies required DXC binaries to `bin/` for development. Release archives exclude DXC.

After that generating and building a project should be as simple as creating a folder for build files and calling CMake, something like:
```cmake
cmake -G "Visual Studio 17 2022"  ..
or 
cmake -G "Visual Studio 17 2022" -DEMBREE_LOCATION=path/to/embree ..
```

## Building for macOS 
Required shader build setup:
- Install/build DXC locally and either:
  - place it under `thirdparty/dxc/<platform-folder>/` (auto-discovered), or
  - set `DXC_PATH` to its installation prefix.

Build the native application with:

```sh
cmake --build build --config Release --target raytracer_app
```

The target generates and verifies the Metal shader package before copying it into `ETX Tracer.app/Contents/Resources`.

## Production shader packages

Release builds use `shaders.etxpack` instead of distributing the `shaders`, `interop`, and `access` source trees. The package contains an indexed set of backend binaries for every renderer variant reachable from the application:

- Metal uses compiled `.metallib` payloads and stored binding metadata.
- Vulkan uses SPIR-V payloads.
- Each index and payload has an integrity hash.
- Package publication uses a temporary file followed by an atomic replacement.
- Release runtime compilation is disabled. A missing, corrupt, or incomplete package is a startup/rendering error.

The package excludes HLSL text, include files, source paths, and DXC. Compiled GPU binaries remain inspectable like other executable code; the package is an IP-exposure reduction mechanism, not encryption.

The `raytracer` Debug and RelWithDebInfo targets compile the copied runtime shader sources and ignore any existing package. Shader edits therefore take effect after rebuilding and restarting the application. Production packages are generated only by the `raytracer_shader_package` target; release builds and CI must build both targets:

```sh
cmake --build build --config Release --target raytracer raytracer_shader_package
```

On macOS, the package builder uses `xcrun metal` and `xcrun metallib`. Its content-addressed cache is stored under the CMake build directory and includes the installed Metal compiler version and deployment target in its key. The shared shader cache also includes a fingerprint of the loaded DXC library. Shader or compiler changes rebuild affected binaries; unchanged package rebuilds reuse them.

`scripts/create-release-archive.sh` packages the signed macOS application bundle. The Windows archive script packages the executable, runtime libraries, data, and shader package while excluding DXC and shader sources.

For a direct diagnostic build:

```sh
bin/raytracer --build-shader-package /path/to/shaders.etxpack --shader-backend metal
bin/raytracer.exe --build-shader-package C:\path\to\shaders.etxpack --shader-backend vulkan
```

The CMake package target reads directly from `sources/etx/render`; direct diagnostic commands default to the development data folder and accept
`--shader-source-root <directory>` when an explicit source tree is required. Package generation bypasses any existing package so every successful build compiles or
cache-validates the current source dependencies.

## Built-in dependencies
These libraries are included into the source code in `thirdparty` folder:
- [enkits](https://github.com/dougbinks/enkiTS) - A permissively licensed C and C++ Task Scheduler for creating parallel programs
- [imgui](https://github.com/ocornut/imgui) - Dear ImGui: Bloat-free Graphical User interface for C++ with minimal dependencies
- [jansson](https://github.com/akheron/jansson) - C library for encoding, decoding and manipulating JSON data
- [mikktspace](https://github.com/mmikk/MikkTSpace) - a common standard for tangent space used in baking tools to produce normal maps.
- [sokol_app, sokol_gfx, sokol_imgui](https://github.com/floooh/sokol) - minimal cross-platform standalone C headers
- [stb_image](https://github.com/nothings/stb) - stb single-file public domain libraries for C/C++
- [tinyexr](https://github.com/syoyo/tinyexr) - tiny OpenEXR image loader/saver library
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) - tiny but powerful single file wavefront obj loader
- [nanovdb](https://developer.nvidia.com/nanovdb) - the library for loading volumetric data
