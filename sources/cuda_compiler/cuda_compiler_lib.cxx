#include "cuda_compiler_lib.hxx"

#include <stdio.h>

#define WIN32_LEAN_AND_MEAN 1
#include <Windows.h>

namespace etx {

bool compile_nvcc_file(const char* path_to_file, const char* output_to_file) {
  auto con = GetStdHandle(STD_OUTPUT_HANDLE);

  static char out_ptx_file[4096] = {};
  int len = snprintf(out_ptx_file, sizeof(out_ptx_file), "%s", path_to_file);
  while ((len > 0) && (out_ptx_file[len] != '.')) {
    --len;
  }
  if (len > 0) {
    snprintf(out_ptx_file + len, sizeof(out_ptx_file) - len, "%s\0", ".ptx");
  }

  const char* target_file = output_to_file == nullptr ? out_ptx_file : output_to_file;

#if defined(NDEBUG) || defined(_NDEBUG)
  const char* debug_flags = "--define-macro NDEBUG --optimize 3";
#else
  const char* debug_flags = "--debug --device-debug --source-in-ptx";
#endif

  static char command_line[4096] = {};
  snprintf(command_line, sizeof(command_line),
    "%s \"%s\" --ptx --output-file \"%s\" -I\"%s\" -I\"%s\" --compiler-bindir \"%s\" -allow-unsupported-compiler %s",  //
    ETX_CUDA_COMPILER, path_to_file, target_file, ETX_OPTIX_INCLUDES, ETX_INCLUDES, ETX_MSBUILD_PATH, debug_flags);    //

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
