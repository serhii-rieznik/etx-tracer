add_subdirectory(etx)

if (ETX_GPU_RENDER)
  add_subdirectory(cuda_compiler)
endif()

add_subdirectory(raytracer)
