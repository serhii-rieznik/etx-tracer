#pragma once

#include <stdint.h>

namespace etx {

enum class CUDACompileTarget : uint32_t {
  PTX,
  Library,
};

bool compile_cuda(CUDACompileTarget target, const char* path_to_file, const char* output_to_file, const char* options);

}  // namespace etx
