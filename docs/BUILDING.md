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
- install latest release of Intel Embree from [GitHub](https://github.com/embree/embree/releases)
- copy embree binaries (embree3.dll and tbb12.dll) to "bin" folder in the root directory of etx-tracer;
- and then building should be as simple as creating a folder for build files and calling CMake, something like:
```
cmake -G "Visual Studio 17 2022"  ..
```


