# Building etx-tracer

This description will be updated during the development process.

## Requirements

Most of external libraries will be located directly in the source code, to reduce a number of dependencies and make building faster.
These libraries and tools you have to install by yourself:
- CMake
- [Intel Embree](https://www.embree.org/) for CPU ray-tracing
- [DirectX Shader Compiler (DXC)](https://github.com/microsoft/DirectXShaderCompiler) for HLSL-to-SPIR-V compilation

## Building for Windows
Windows is the only one platform, which is completely supported at the moment.
- download and install the latest release of Intel Embree from [GitHub](https://github.com/embree/embree/releases);
  - add environment variable `EMBREE_LOCATION` pointing to the Embree installation folder provide this parameter to CMake (i.e `cmake -DEMBREE_LOCATION=path/to/embree`);
  - copy embree binaries (embree4.dll and other required dlls) to the `bin` folder in the root directory of `etx-tracer`

### DXC (DirectX Shader Compiler)
DXC is required for compiling HLSL shaders to SPIR-V for Vulkan. CMake will automatically:
1. Check if DXC is already installed on your system
2. If not found on Windows, download a DXC release from GitHub
3. Copy DXC binaries to the `bin` folder for runtime usage

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

When DXC is found (system, `DXC_PATH`, or vendored path), the build copies required DXC binaries to `bin/` for distribution.

After that generating and building a project should be as simple as creating a folder for build files and calling CMake, something like:
```cmake
cmake -G "Visual Studio 17 2022"  ..
or 
cmake -G "Visual Studio 17 2022" -DEMBREE_LOCATION=path/to/embree ..
```

## Building for macOS 
Currently macOS platform is not completely supported, however there are steps towards it.

Required extra step for shaders:
- Install/build DXC locally and either:
  - place it under `thirdparty/dxc/<platform-folder>/` (auto-discovered), or
  - set `DXC_PATH` to its installation prefix.

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
