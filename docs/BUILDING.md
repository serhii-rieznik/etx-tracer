# Building etx-tracer

This description will be updated during the development process.

## Requirements
Most of external libraries will be located directly in the source code, to reduce a number of dependencies and make building faster.
But this libraries and tools you have to install by yourself:
- CMake
- [Intel Embree](https://www.embree.org/) for CPU ray-tracing
- CUDA
- OptiX
- optionally OpenVDB if you want to load .vdb files with volumetric data.

## Building for Windows
Windows is the only one platform, which is supported at the moment.
- download and install the latest release of Intel Embree from [GitHub](https://github.com/embree/embree/releases)
- copy embree binaries (embree3.dll and tbb12.dll) to "bin" folder in the root directory of etx-tracer;
- and then building should be as simple as creating a folder for build files and calling CMake, something like:
```
cmake -G "Visual Studio 17 2022"  ..
```

## Built-in dependencies
These libraries are included into the source code in `thirdparty` folder:
- [enkits](https://github.com/dougbinks/enkiTS) - A permissively licensed C and C++ Task Scheduler for creating parallel programs
- [glm](https://github.com/g-truc/glm) - OpenGL Mathematics
- [imgui](https://github.com/ocornut/imgui) - Dear ImGui: Bloat-free Graphical User interface for C++ with minimal dependencies
- [jansson](https://github.com/akheron/jansson) - C library for encoding, decoding and manipulating JSON data
- [mikktspace](https://github.com/mmikk/MikkTSpace) - A common standard for tangent space used in baking tools to produce normal maps.
- [sokol_app, sokol_gfx, sokol_imgui](https://github.com/floooh/sokol) - minimal cross-platform standalone C headers
- [stb_image](https://github.com/nothings/stb) - stb single-file public domain libraries for C/C++
- [tinyexr](https://github.com/syoyo/tinyexr) - Tiny OpenEXR image loader/saver library
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) - Tiny but powerful single file wavefront obj loader


