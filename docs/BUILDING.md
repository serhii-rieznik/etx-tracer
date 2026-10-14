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
DXC is required at runtime for on-demand HLSL shader compilation. CMake will automatically:
1. Check if DXC is already installed on your system
2. If not found on Windows, download a DXC release from GitHub
3. Copy DXC binaries to `bin` for runtime shader compilation

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

When DXC is found (system, `DXC_PATH`, or vendored path), the build copies required DXC binaries to `bin/`. Release archives include the DXC runtime and shader sources.

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

The target bundles shader sources in `ETX Tracer.app/Contents/Resources` and DXC in `Contents/Frameworks`. Metal libraries and pipelines compile on demand and use the runtime disk cache.

## Runtime shaders

All configurations compile only requested shader entry points. Build the Windows application with:

```sh
cmake --build build --config Release --target raytracer
```

The build copies the `shaders`, `interop`, and `access` trees beside the executable. It never compiles an inventory of shader variants or reads `shaders.etxpack`. Shader edits take effect after rebuilding or running the `raytracer_runtime_shaders` target and reloading shaders in the application.

The runtime caches preprocessed sources, compiled entry points, and backend pipelines separately. Source dependency timestamps and sizes validate the preprocessing cache. Compiled entry-point keys retain global declarations, resource bindings, type layouts, and all overloads of referenced function names; comments, whitespace, line directives, and unreferenced top-level function definitions do not invalidate binaries. The same analysis removes those function definitions before DXC compilation. Nested scopes remain intact. Debug-information builds retain the complete source and location-sensitive hash.

Binary cache entries are content-addressed, include the loaded DXC library fingerprint and compilation options, and validate payload integrity. Reverting source edits reuses prior entries. Shader compiler revisions must bump the cache version when code-generation options or SPIRV-Cross translation change.

Windows releases distribute shader sources and DXC. macOS bundles include sources and DXC, and use cached Metal libraries and pipeline archives. The first request for an uncached variant incurs compilation; subsequent requests reuse validated caches. CPU-only commands initialize DXC only when they request a shader.

`--gpu-kernel-timings` enables GPU dispatch timestamps. `--gpu-kernel-statistics` separately requests Vulkan pipeline executable statistics when the driver supports them, recording subgroup size, registers, and memory statistics in the console. Statistics capture can increase pipeline preparation cost. Nsight Graphics provides sampled runtime occupancy and stall analysis.

Headless Vulkan renders expose frame boundaries through `VK_EXT_frame_boundary` when a profiling tool supplies that extension. Nsight can then capture `--render` runs automatically. On Windows, omit Nsight's `--platform` option: Qt consumes it as a platform-plugin override and can prevent Nsight from starting.

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
