# Building etx-tracer

This description will be updated during the development process.

## Requirements
Most of external libraries will be located directly in the source code, to reduce a number of dependencies and make building faster.
But this libraries and tools you have to install by yourself:
- CMake
- CUDA
- OptiX
- optionally OpenVDB if you want to load .vdb files with volumetric data.

## Building for Windows
Windows is the only one platform, which is supported at the moment.
Building should be as simple as creating a folder for build files and calling CMake, something like:
```
cmake -G "Visual Studio 17 2022"  ..
```

