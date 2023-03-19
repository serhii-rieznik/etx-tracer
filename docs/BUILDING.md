# Building etx-tracer

This description will be updated during the development process.

## Requirements

Most of external libraries will be located directly in the source code, to reduce a number of dependencies and make building faster.
These libraries and tools you have to install by yourself:
- CMake
- [Intel Embree](https://www.embree.org/) for CPU ray-tracing

#### For GPU ray tracing which is now being updated, you need:
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [OptiX](https://developer.nvidia.com/designworks/optix/download)

## Building for Windows
Windows is the only one platform, which is completely supported at the moment.
- download and install the latest release of Intel Embree from [GitHub](https://github.com/embree/embree/releases);
  - add environment variable `EMBREE_LOCATION` pointing to the Embree installation folder provide this parameter to CMake (i.e `cmake -DEMBREE_LOCATION=path/to/embree`);
  - copy embree binaries (embree4.dll and other required dlls) to the `bin` folder in the root directory of `etx-tracer`

Optionally:
  - download and install CUDA and OptiX;
  - add environment variable `OptiX_INSTALL_DIR` pointing to the OptiX installation folder;
    
If OptiX and/or CUDA is not available - solution will still be created, but GPU rendering will not be available. Provide `DISABLE_GPU` option to CMake to force disable GPU support

After that generating and building a project should be as simple as creating a folder for build files and calling CMake, something like:
```cmake
cmake -G "Visual Studio 17 2022"  ..
or 
cmake -G "Visual Studio 17 2022" -DEMBREE_LOCATION=path/to/embree ..
```

## Building for macOS 
Currently macOS platform is not completely supported, however there are steps towards it.

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

