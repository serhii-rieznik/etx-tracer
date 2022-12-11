#include <cuda_compiler_lib.hxx>

#include <stdio.h>
#include <conio.h>

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    printf("Usage cuda-compiler input-file <output-file>\n");
    return 1;
  }
  const char* output_file = argc >= 3 ? argv[2] : nullptr;

  int result = 0;
  do {
    result = etx::compile_cuda(etx::CUDACompileTarget::PTX, argv[1], output_file, 61) ? 0 : 1;
    printf("Press [R] to retry or any other key to exit.\n");

    int option = _getch();
    if ((option != 'R') && (option != 'r')) {
      break;
    }

  } while (true);

  return result;
}
