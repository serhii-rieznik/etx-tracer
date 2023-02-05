#pragma once

#include <stdint.h>

namespace etx {

enum class CUDACompileTarget : uint32_t {
  PTX,
  Library,
};

constexpr bool kCUDADebugBuild = false;

bool compile_cuda(CUDACompileTarget target, const char* path_to_file, const char* output_to_file, uint32_t arch);

}  // namespace etx
