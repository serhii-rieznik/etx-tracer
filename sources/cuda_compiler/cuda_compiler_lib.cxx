#include "cuda_compiler_lib.hxx"

#include <etx/log/log.hxx>
#include <stdio.h>

#define WIN32_LEAN_AND_MEAN 1
#include <Windows.h>

#include <nvrtc.h>

#include <vector>

namespace etx {

bool rtc_compile(CUDACompileTarget target, const char* filename, const char* options, const char* output_file) {
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

  if (rtc_call(nvrtcCreateProgram(&program, source_data.data(), nullptr, 0, nullptr, nullptr)) == false) {
    return false;
  }

  std::vector<const char*> option_set = {
    "--device-as-default-execution-space",
    "--use_fast_math",
    "--std=c++17",
    (target == CUDACompileTarget::PTX) ? "--gpu-architecture=compute_86" : "--gpu-architecture=sm_86",
    "-I" ETX_INCLUDES,
    "-I" ETX_OPTIX_INCLUDES,
    "-I" ETX_CUDA_INCLUDES,
    "-I" ETX_CUDA_INCLUDES "/cuda/std",
  };

  if (kCUDADebugBuild) {
    option_set.emplace_back("--device-debug");
    option_set.emplace_back("--dopt=off");
  } else {
    option_set.emplace_back("--generate-line-info");
    option_set.emplace_back("--dopt=on");
    option_set.emplace_back("-DNDEBUG");
  }

  if (rtc_call(nvrtcCompileProgram(program, static_cast<int32_t>(option_set.size()), option_set.data())) == false) {
    rtc_call(nvrtcDestroyProgram(&program));
    return false;
  }

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

  rtc_call(nvrtcDestroyProgram(&program));
  return success;
}

bool compile_cuda(CUDACompileTarget target, const char* path_to_file, const char* output_to_file, const char* options) {
  return rtc_compile(target, path_to_file, options, output_to_file);
  auto con = GetStdHandle(STD_OUTPUT_HANDLE);

  static char out_ptx_file[4096] = {};
  int len = snprintf(out_ptx_file, sizeof(out_ptx_file), "%s", path_to_file);
  while ((len > 0) && (out_ptx_file[len] != '.')) {
    --len;
  }
  if (len > 0) {
    snprintf(out_ptx_file + len, sizeof(out_ptx_file) - len, "%s\0", target == CUDACompileTarget::PTX ? ".ptx" : ".fatbin");
  }
  const char* target_file = output_to_file == nullptr ? out_ptx_file : output_to_file;

  const char* debug_flags = nullptr;
  if (kCUDADebugBuild) {
    debug_flags = (target == CUDACompileTarget::PTX)    //
                    ? "--device-debug --source-in-ptx"  //
                    : "--device-debug";                 //
  } else {
    debug_flags = (target == CUDACompileTarget::PTX)                                                 //
                    ? "--device-debug --source-in-ptx --dopt on --define-macro NDEBUG --optimize 3"  //
                    : "--device-debug --dopt on --define-macro NDEBUG --optimize 3";                 //
  }

  static char command_line[4096] = {};
  snprintf(command_line, sizeof(command_line),
    "%s \"%s\" --output-file \"%s\" -I\"%s\" -I\"%s\" --std c++17 --expt-relaxed-constexpr --compiler-bindir \"%s\" %s %s %s",  //
    ETX_CUDA_COMPILER, path_to_file, target_file, ETX_OPTIX_INCLUDES, ETX_INCLUDES, ETX_MSBUILD_PATH, debug_flags,              //
    (target == CUDACompileTarget::PTX) ? "--ptx" : "--fatbin", options);                                                        //

  static char command_line_info[4096] = {};
  int j = 0;
  for (int i = 0; command_line[i] != 0; ++i, ++j) {
    if ((command_line[i] == ' ') && (command_line[i + 1] == '-')) {
      command_line_info[j++] = '\n';
      command_line_info[j++] = ' ';
      command_line_info[j] = ' ';
    } else {
      command_line_info[j] = command_line[i];
    }
  }
  command_line_info[j] = 0;

  SetConsoleTextAttribute(con, FOREGROUND_RED | FOREGROUND_GREEN);
  puts(command_line_info);
  SetConsoleTextAttribute(con, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);

  STARTUPINFO startup_info = {
    .cb = sizeof(STARTUPINFO),
  };
  PROCESS_INFORMATION process_info = {};
  if (CreateProcess(nullptr, command_line, nullptr, nullptr, true, NORMAL_PRIORITY_CLASS, nullptr, nullptr, &startup_info, &process_info) == FALSE) {
    SetConsoleTextAttribute(con, FOREGROUND_RED);
    printf("\nFailed to compile %s (failed to create a process)\n", path_to_file);
    SetConsoleTextAttribute(con, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    return false;
  }

  if (WaitForSingleObject(process_info.hProcess, 1000 * 3600) != WAIT_OBJECT_0) {
    TerminateProcess(process_info.hProcess, 0xffffffff);
    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    SetConsoleTextAttribute(con, FOREGROUND_RED);
    printf("\nFailed to compile %s (failed to wait for a process)\n", path_to_file);
    SetConsoleTextAttribute(con, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    return false;
  }

  DWORD code = 0;
  if (GetExitCodeProcess(process_info.hProcess, &code) == FALSE) {
    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    SetConsoleTextAttribute(con, FOREGROUND_RED);
    printf("\nFailed to compile %s (failed to get an exit code)\n", path_to_file);
    SetConsoleTextAttribute(con, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    return false;
  }

  CloseHandle(process_info.hThread);
  CloseHandle(process_info.hProcess);

  if (code != 0) {
    SetConsoleTextAttribute(con, FOREGROUND_RED);
    printf("\nFailed to compile %s (exit code = %d)\n", path_to_file, code);
    SetConsoleTextAttribute(con, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    return false;
  }

  SetConsoleTextAttribute(con, FOREGROUND_GREEN);
  printf("\nCompiled %s\n", path_to_file);
  SetConsoleTextAttribute(con, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
  return true;
}

}  // namespace etx
