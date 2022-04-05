#include <cuda_compiler_lib.hxx>

#include <stdio.h>

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    printf("Usage cuda-compiler input-file <output-file>\n");
    return 1;
  }

  const char* output_file = argc >= 3 ? argv[2] : nullptr;

  int result = etx::compile_nvcc_file(argv[1], output_file) ? 0 : 1;
  printf("Press enter to continue...\n");
  (void)getc(stdin);
  return result;
}
