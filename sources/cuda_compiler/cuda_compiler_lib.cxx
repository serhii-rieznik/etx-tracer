#include "cuda_compiler_lib.hxx"

#include <etx/log/log.hxx>
#include <stdio.h>

#define WIN32_LEAN_AND_MEAN 1
#include <Windows.h>

#include <nvrtc.h>

#include <vector>

namespace etx {

bool rtc_compile(CUDACompileTarget target, const char* filename, const uint32_t arch, const char* output_file) {
  nvrtcProgram program = {};

  auto rtc_call_impl = [&program](const char* expr, nvrtcResult result) -> bool {
    if (result == NVRTC_SUCCESS)
      return true;

    const char* err = nvrtcGetErrorString(result);
    log::error("Call to %s failed with `%s`", expr, err);
    uint64_t log_size = 0;
    if (nvrtcGetProgramLogSize(program, &log_size) == NVRTC_SUCCESS) {
      std::vector<char> log(log_size + 1llu);
      if (nvrtcGetProgramLog(program, log.data()) == NVRTC_SUCCESS) {
        log::error("%s", log.data());
      }
    }

    return false;
  };
#define rtc_call(expr) rtc_call_impl(#expr, expr)

  FILE* source = fopen(filename, "rb");
  if (source == nullptr)
    return false;

  fseek(source, 0, SEEK_END);
  uint64_t file_size = ftell(source);
  fseek(source, 0, SEEK_SET);

  std::vector<char> source_data(file_size + 1llu);
  uint64_t bytes_read = fread(source_data.data(), 1llu, file_size, source);
  fclose(source);

  if (bytes_read != file_size) {
    return false;
  }

  log::info("Compiling CUDA: %s...", filename);
  log::info(" - creating program");
  if (rtc_call(nvrtcCreateProgram(&program, source_data.data(), nullptr, 0, nullptr, nullptr)) == false) {
    return false;
  }

  char arch_ptx[] = "--gpu-architecture=compute_XX";
  arch_ptx[sizeof(arch_ptx) - 3] = char(48 + arch / 10);
  arch_ptx[sizeof(arch_ptx) - 2] = char(48 + arch % 10);
  char arch_bin[] = "--gpu-architecture=sm_XX";
  arch_bin[sizeof(arch_bin) - 3] = char(48 + arch / 10);
  arch_bin[sizeof(arch_bin) - 2] = char(48 + arch % 10);

  std::vector<const char*> option_set = {
    "--device-as-default-execution-space",
    "--use_fast_math",
    "--std=c++17",
    (target == CUDACompileTarget::PTX) ? arch_ptx : arch_bin,
    "-I" ETX_INCLUDES,
    "-I" ETX_OPTIX_INCLUDES,
    "-I" ETX_CUDA_INCLUDES,
    "-I" ETX_CUDA_INCLUDES "/cuda/std",
  };

  if (kCUDADebugBuild) {
    option_set.emplace_back("--device-debug");
  } else {
    option_set.emplace_back("--dopt=on");
    option_set.emplace_back("-DNDEBUG");
  }

  log::info(" - compiling program...");
  if (rtc_call(nvrtcCompileProgram(program, static_cast<int32_t>(option_set.size()), option_set.data())) == false) {
    rtc_call(nvrtcDestroyProgram(&program));
    return false;
  }
  log::info(" - getting compiled data...");

  bool success = false;

  if (target == CUDACompileTarget::PTX) {
    uint64_t ptx_size = 0;
    rtc_call(nvrtcGetPTXSize(program, &ptx_size));
    std::vector<char> ptx_data(ptx_size);
    if (rtc_call(nvrtcGetPTX(program, ptx_data.data()))) {
      FILE* f_out = fopen(output_file, "wb");
      if (f_out != nullptr) {
        uint64_t bytes_written = fwrite(ptx_data.data(), 1llu, ptx_size, f_out);
        fclose(f_out);
        success = (bytes_written == ptx_size);
      }
    }
  } else if (target == CUDACompileTarget::Library) {
    uint64_t cubin_size = 0;
    rtc_call(nvrtcGetCUBINSize(program, &cubin_size));
    std::vector<char> cubin_data(cubin_size + 1024llu);
    if (rtc_call(nvrtcGetCUBIN(program, cubin_data.data()))) {
      FILE* f_out = fopen(output_file, "wb");
      if (f_out != nullptr) {
        uint64_t bytes_written = fwrite(cubin_data.data(), 1llu, cubin_size, f_out);
        fclose(f_out);
        success = (bytes_written == cubin_size);
      }
    }
  }
  log::info("Compilation finished: %s", success ? "success" : "fail");
  rtc_call(nvrtcDestroyProgram(&program));
  return success;
}

bool compile_cuda(CUDACompileTarget target, const char* path_to_file, const char* output_to_file, const uint32_t arch) {
  return rtc_compile(target, path_to_file, arch, output_to_file);
}

}  // namespace etx
