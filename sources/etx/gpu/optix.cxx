#include <etx/log/log.hxx>
#include <etx/gpu/optix.hxx>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define ETX_CUDA_CALL_FAILED(F) ((F) != cudaSuccess)
#define ETX_OPTIX_CALL_FAILED(F) ((F) != OPTIX_SUCCESS)

namespace etx {

struct GPUBufferOptixImpl {
  GPUBufferOptixImpl(const GPUBuffer::Descriptor& desc) {
  }

  ~GPUBufferOptixImpl() {
  }
};

struct GPUOptixImplData {
  CUcontext cuda_context = {};
  CUstream main_stream = {};

  void* optix_handle = nullptr;
  OptixDeviceContext optix = {};

  ObjectIndexPool<GPUBufferOptixImpl> buffer_pool;

  GPUOptixImplData() {
    buffer_pool.init(1024u);

    cudaFree(nullptr);

    if (init_cuda() == false) {
      log::error("Failed to initialize CUDA");
      return;
    }

    init_optix();
  }

  ~GPUOptixImplData() {
    cleanup_optix();
    cleanup_cuda();

    buffer_pool.cleanup();
  }

  bool init_cuda() {
    int device_count = 0;
    if ETX_CUDA_CALL_FAILED (cudaGetDeviceCount(&device_count))
      return false;

    if (device_count == 0)
      return false;

    int device_id = 0;
    if ETX_CUDA_CALL_FAILED (cudaSetDevice(device_id))
      return false;

    if ETX_CUDA_CALL_FAILED (cudaStreamCreate(&main_stream))
      return false;

    cudaDeviceProp device_props = {};
    if ETX_CUDA_CALL_FAILED (cudaGetDeviceProperties(&device_props, device_id))
      return false;

    if (cuCtxGetCurrent(&cuda_context) != CUDA_SUCCESS)
      return false;

    return true;
  }

  void cleanup_cuda() {
    cudaStreamDestroy(main_stream);
  }

  bool init_optix() {
    OptixDeviceContextOptions options = {};
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    options.logCallbackLevel = 3;
    options.logCallbackFunction = [](unsigned int level, const char* tag, const char* message, void* cbdata) {
      log::info("OptiX: [%s] %s\n", tag, message);
    };

    if ETX_OPTIX_CALL_FAILED (optixInitWithHandle(&optix_handle))
      return false;

    if ETX_OPTIX_CALL_FAILED (optixDeviceContextCreate(cuda_context, &options, &optix))
      return false;

    if ETX_OPTIX_CALL_FAILED (optixDeviceContextSetLogCallback(optix, options.logCallbackFunction, nullptr, 4))
      return false;

    return true;
  }

  void cleanup_optix() {
    optixDeviceContextDestroy(optix);
    optixUninitWithHandle(optix_handle);
  }
};

ETX_PIMPL_IMPLEMENT_ALL(GPUOptixImpl, Data)

GPUBuffer GPUOptixImpl::create_buffer(const GPUBuffer::Descriptor& desc) {
  return {_private->buffer_pool.alloc(desc)};
}

void GPUOptixImpl::destroy_buffer(GPUBuffer buffer) {
  _private->buffer_pool.free(buffer.handle);
}

}  // namespace etx
