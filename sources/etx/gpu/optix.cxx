#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/gpu/optix.hxx>

#include <json.hpp>

#include <cuda_compiler/cuda_compiler_lib.hxx>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <atomic>
#include <unordered_map>

namespace etx {

constexpr static uint64_t kMaxEntryPoints = 8;
static CUstream shared_cuda_stream = {};

struct PipelineDesc {
  const char* name = nullptr;
  const char* code = nullptr;
  const char* source = nullptr;
  GPUPipeline::EntryPoint entry_points[kMaxEntryPoints] = {};
  uint32_t entry_point_count = 0;
  CUDACompileTarget target = CUDACompileTarget::PTX;
  bool completed = false;
};

CUstream cuda_stream() {
  return shared_cuda_stream;
}

const char* cuda_result(CUresult result);

bool check_cuda_call_failed(CUresult result, const char* expr, const char* file, uint32_t line) {
  if (result == CUDA_SUCCESS)
    return false;

  log::error("CUDA call %s failed with %s at %s [%u]", expr, cuda_result(result), file, line);
  return true;
}

#define ETX_CUDA_CALL(expr)                                  \
  do {                                                       \
    check_cuda_call_failed(expr, #expr, __FILE__, __LINE__); \
  } while (0)

#define ETX_CUDA_FAILED(expr) check_cuda_call_failed(expr, #expr, __FILE__, __LINE__)
#define ETX_CUDA_SUCCEED(expr) (check_cuda_call_failed(expr, #expr, __FILE__, __LINE__) == false)

struct GPUBufferOptixImpl;
struct GPUPipelineOptixImpl;
struct GPUAccelerationStructureImpl;

struct GPUOptixImplData {
  constexpr static const uint64_t kSharedBufferSize = 16llu * 1024llu * 1024llu + 2llu * 2160llu * 3840llu * 16llu;

  CUdevice cuda_device = {};
  CUcontext cuda_context = {};
  CUstream main_stream = {};
  cudaDeviceProp device_properties = {};

  void* optix_handle = nullptr;
  OptixDeviceContext optix = {};
  OptixDenoiser denoiser = {};
  device_pointer_t denoiser_state = {};
  device_pointer_t denoiser_scratch = {};
  OptixDenoiserSizes denoiser_sizes = {};
  uint2 denoiser_image_size = {};
  bool denoiser_setup = false;

  ObjectIndexPool<GPUBufferOptixImpl> buffer_pool;
  ObjectIndexPool<GPUPipelineOptixImpl> pipeline_pool;
  ObjectIndexPool<GPUAccelerationStructureImpl> accelearaion_structure_pool;

  GPUBuffer shared_buffer = {};
  std::atomic<uint64_t> shared_buffer_offset = {};
  uint32_t cuda_arch = 0u;

  GPUOptixImplData() {
    buffer_pool.init(1024u);
    pipeline_pool.init(1024u);
    accelearaion_structure_pool.init(16u);

    cudaFree(nullptr);

    if (init_cuda() == false) {
      log::error("Failed to initialize CUDA");
      return;
    }

    if (init_optix()) {
      init_denoiser();
    }
  }

  ~GPUOptixImplData() {
    cleanup_denoiser();
    cleanup_optix();
    cleanup_cuda();
    accelearaion_structure_pool.cleanup();
    pipeline_pool.cleanup();
    buffer_pool.cleanup();
  }

  bool init_cuda() {
    int device_count = 0;
    if (cuda_call_failed(cudaGetDeviceCount(&device_count)))
      return false;

    if (device_count == 0)
      return false;

    int device_id = 0;
    if (cuda_call_failed(cudaSetDevice(device_id)))
      return false;

    if (cuda_call_failed(cuDeviceGet(&cuda_device, device_id)))
      return false;

    int compute_major = 0;
    cuDeviceGetAttribute(&compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda_device);

    int compute_minor = 0;
    cuDeviceGetAttribute(&compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda_device);

    uint64_t device_memory = 0llu;
    cuDeviceTotalMem(&device_memory, cuda_device);

    char device_name[128] = {};
    cuDeviceGetName(device_name, sizeof(device_name), cuda_device);
    log::info("Using CUDA device: %s (compute: %d.%d, %llu Mb)", device_name, compute_major, compute_minor, device_memory / 1024 / 1024);

    if (cuda_call_failed(cudaStreamCreate(&main_stream)))
      return false;

    shared_cuda_stream = main_stream;

    if (cuda_call_failed(cudaGetDeviceProperties(&device_properties, device_id)))
      return false;

    if (cuCtxGetCurrent(&cuda_context) != CUDA_SUCCESS)
      return false;

    cuda_arch = 10 * device_properties.major + device_properties.minor;

    return true;
  }

  void cleanup_cuda() {
    cudaStreamDestroy(main_stream);
  }

  bool init_optix() {
    OptixDeviceContextOptions options = {};
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    options.logCallbackLevel = 4;
    options.logCallbackData = this;
    options.logCallbackFunction = [](unsigned int level, const char* tag, const char* message, void* cbdata) {
      auto self = reinterpret_cast<GPUOptixImplData*>(cbdata);
      if (level <= 2) {
        log::error("OptiX: [%s] %s", tag, message);
        self->report_error();
      } else if (level <= 3) {
        log::warning("OptiX: [%s] %s", tag, message);
      } else {
        log::info("OptiX: [%s] %s", tag, message);
      }
    };

    if (optix_call_failed(optixInitWithHandle(&optix_handle)))
      return false;

    if (optix_call_failed(optixDeviceContextCreate(cuda_context, &options, &optix)))
      return false;

    if (optix_call_failed(optixDeviceContextSetLogCallback(optix, options.logCallbackFunction, nullptr, 4)))
      return false;

    return true;
  }

  void cleanup_optix() {
    optixDeviceContextDestroy(optix);
    optixUninitWithHandle(optix_handle);
  }

  void init_denoiser() {
    OptixDenoiserOptions options = {};
    if (optix_call_failed(optixDenoiserCreate(optix, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &denoiser))) {
      log::error("Failed to create denoiser");
      return;
    }
  }

  void cleanup_denoiser() {
    if (optixDenoiserDestroy(denoiser)) {
      log::error("Failed to destroy denoiser");
    }
  }

  void report_error() {
#if (ETX_DEBUG)
    ETX_DEBUG_BREAK();
#endif
    cudaStreamDestroy(main_stream);
    main_stream = {};
  }

  bool invalid_state() const {
    return main_stream == nullptr;
  }

  bool cuda_call_failed(cudaError result) {
    if (result == cudaError::cudaSuccess)
      return false;

    report_error();
    return true;
  }

  bool cuda_call_failed(CUresult result) {
    if (result == CUresult::CUDA_SUCCESS)
      return false;

    report_error();
    return true;
  }

  bool optix_call_failed(OptixResult result) {
    if (result == OptixResult::OPTIX_SUCCESS)
      return false;

    report_error();
    return true;
  }

  bool launch_optix(GPUPipelineOptixImpl& pipeline, const char* function, uint32_t dim_x, uint32_t dim_y, device_pointer_t params, uint64_t params_size);
  bool launch_cuda(GPUPipelineOptixImpl& pipeline, const char* function, uint32_t dim_x, uint32_t dim_y, device_pointer_t params, uint64_t params_size);

  device_pointer_t upload_to_shared_buffer(device_pointer_t ptr, void* data, uint64_t size);
  bool compile(const char* file, bool force, GPUPipeline::Descriptor& desc, std::vector<uint8_t>& binary);
};

#define ETX_OPTIX_INCLUDES 1

/****************************************************
 *
 * #include "optix_buffer.hxx"
 *
 ***************************************************/

#if !defined(ETX_OPTIX_INCLUDES)
#error This file should not be included
#endif

struct GPUBufferOptixImpl {
  GPUBufferOptixImpl() = default;

  GPUBufferOptixImpl(GPUOptixImplData* device, const GPUBuffer::Descriptor& desc) {
    capacity = align_up(desc.size, 16llu);

    if (capacity == 0) {
      log::warning("Allocating CUDA empty buffer (size: %llu)", capacity);
      return;
    }

    if (device->cuda_call_failed(cuMemAlloc(&device_ptr, capacity))) {
      log::error("Failed to create CUDA buffer with size: %llu", capacity);
      return;
    }

    if (desc.data != nullptr) {
      if (device->cuda_call_failed(cuMemcpyHtoD(device_ptr, desc.data, desc.size)))
        log::error("Failed to copy content to CUDA buffer %p from %p with size %llu", device_ptr, desc.data, desc.size);
    }
  }

  ~GPUBufferOptixImpl() {
    ETX_ASSERT(device_ptr == 0u);
  }

  void release(GPUOptixImplData* device) {
    if (device->cuda_call_failed(cuMemFree(device_ptr))) {
      log::error("Failed to free CUDA buffer: %p", device_ptr);
    }
    device_ptr = 0llu;
  }

  device_pointer_t device_pointer() const {
    return device_ptr;
  }

  device_pointer_t device_ptr = 0llu;
  uint64_t capacity = 0;
};

struct GPUPipelineOptixImpl {
  GPUPipelineOptixImpl(GPUOptixImplData* device, const GPUPipeline::Descriptor& desc) {
    if (create_module(device, desc) == false) {
      log::error("Failed to create OptiX module");
      return;
    }
    create_pipeline(device, desc);
  }

  ~GPUPipelineOptixImpl() {
    ETX_ASSERT(cuda.cuda_module == nullptr);
    ETX_ASSERT(optix_module == nullptr);
    ETX_ASSERT(pipeline == nullptr);
  }

  bool create_optix_module(GPUOptixImplData* device, const GPUPipeline::Descriptor& desc) {
    constexpr uint32_t debug_build = kCUDADebugBuild ? 1u : 0u;
    constexpr uint32_t optimization_level = debug_build * OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 + (1 - debug_build) * OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    constexpr uint32_t debug_level = debug_build * OPTIX_COMPILE_DEBUG_LEVEL_FULL + (1 - debug_build) * OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
    constexpr uint64_t kMaxLogSize = 64llu * 1024llu;

    std::vector<char> compile_log(kMaxLogSize);

    OptixModuleCompileOptions module_options = {
      .maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
      .optLevel = OptixCompileOptimizationLevel(optimization_level),
      .debugLevel = OptixCompileDebugLevel(debug_level),
    };

    pipeline_options = {
      .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
      .numPayloadValues = kOptiXPayloadSize,
      .numAttributeValues = 2,
      .exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_USER,
      .pipelineLaunchParamsVariableName = "global",
    };

    uint64_t compile_log_size = kMaxLogSize;
    if (device->optix_call_failed(optixModuleCreateFromPTX(device->optix, &module_options, &pipeline_options,  //
          reinterpret_cast<const char*>(desc.data), desc.data_size, compile_log.data(), &compile_log_size, &optix_module))) {
      log::error("optixModuleCreateFromPTX failed");
      if (compile_log_size > 1) {
        log::error(compile_log.data());
      }
      release(device);
      return false;
    }

    if (compile_log_size > 1) {
      log::warning(compile_log.data());
    }

    for (uint64_t i = 0; i < desc.entry_point_count; ++i) {
      OptixProgramGroupDesc program_desc = {
        .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
      };
      program_desc.raygen.module = optix_module;
      program_desc.raygen.entryFunctionName = desc.entry_points[i];
      OptixProgramGroupOptions program_options = {};
      compile_log_size = kMaxLogSize;
      if (device->optix_call_failed(optixProgramGroupCreate(device->optix, &program_desc, 1, &program_options, compile_log.data(), &compile_log_size, entry_points + i))) {
        log::error("optixProgramGroupCreate failed");
        if (compile_log_size > 1) {
          log::error(compile_log.data());
        }
        release(device);
        return false;
      }

      if (compile_log_size > 1) {
        log::warning(compile_log.data());
      }
    }

    {
      OptixProgramGroupDesc program_desc = {
        .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
      };
      program_desc.hitgroup.moduleCH = optix_module;
      program_desc.hitgroup.entryFunctionNameCH = "__closesthit__main";
      program_desc.hitgroup.moduleAH = optix_module;
      program_desc.hitgroup.entryFunctionNameAH = "__anyhit__main";

      compile_log_size = kMaxLogSize;
      OptixProgramGroupOptions program_options = {};
      if (device->optix_call_failed(optixProgramGroupCreate(device->optix, &program_desc, 1, &program_options, compile_log.data(), &compile_log_size, &main_hit_program_group))) {
        log::error("optixProgramGroupCreate failed");
        if (compile_log_size > 1) {
          log::error(compile_log.data());
        }
        release(device);
        return false;
      }

      if (compile_log_size > 1) {
        log::warning(compile_log.data());
      }
    }

    {
      OptixProgramGroupDesc program_desc = {
        .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
      };
      program_desc.miss.module = optix_module;
      program_desc.miss.entryFunctionName = "__miss__main";

      compile_log_size = kMaxLogSize;
      OptixProgramGroupOptions program_options = {};
      if (device->optix_call_failed(optixProgramGroupCreate(device->optix, &program_desc, 1, &program_options, compile_log.data(), &compile_log_size, &main_miss_program_group))) {
        log::error("optixProgramGroupCreate failed");
        if (compile_log_size > 1) {
          log::error(compile_log.data());
        }
        release(device);
        return false;
      }

      if (compile_log_size > 1) {
        log::warning(compile_log.data());
      }
    }

    return true;
  }

  bool create_cuda_module(GPUOptixImplData* device, const GPUPipeline::Descriptor& desc) {
    ETX_ASSERT(cuda.cuda_module == nullptr);
    return ETX_CUDA_SUCCEED(cuModuleLoadData(&cuda.cuda_module, desc.data));
  }

  bool create_module(GPUOptixImplData* device, const GPUPipeline::Descriptor& desc) {
    if (device->invalid_state())
      return false;

    target = CUDACompileTarget(desc.compile_options);

    switch (target) {
      case CUDACompileTarget::PTX:
        return create_optix_module(device, desc);

      case CUDACompileTarget::Library:
        return create_cuda_module(device, desc);

      default:
        break;
    }

    return false;
  }

  void release(GPUOptixImplData* device) {
    if (main_hit_program_group != nullptr) {
      if (device->optix_call_failed(optixProgramGroupDestroy(main_hit_program_group)))
        log::error("optixProgramGroupDestroy failed");
      main_hit_program_group = {};
    }
    if (main_miss_program_group != nullptr) {
      if (device->optix_call_failed(optixProgramGroupDestroy(main_miss_program_group)))
        log::error("optixProgramGroupDestroy failed");
      main_miss_program_group = {};
    }

    for (uint32_t i = 0; i < descriptor.entry_point_count; ++i) {
      if (entry_points[i] != nullptr) {
        if (device->optix_call_failed(optixProgramGroupDestroy(entry_points[i])))
          log::error("optixProgramGroupDestroy failed");
      }
      entry_points[i] = {};
    }

    if (pipeline != nullptr) {
      if (device->optix_call_failed(optixPipelineDestroy(pipeline)))
        log::error("Failed to destroy OptiX pipeline");
      pipeline = {};
    }

    if (optix_module != nullptr) {
      if (device->optix_call_failed(optixModuleDestroy(optix_module)))
        log::error("Failed to destroy OptiX module");
      optix_module = {};
    }

    if (cuda.cuda_module != nullptr) {
      ETX_CUDA_CALL(cuModuleUnload(cuda.cuda_module));
      cuda.cuda_module = nullptr;
    }

    descriptor = {};
  }

  bool create_cuda_pipeline(GPUOptixImplData* device) {
    return true;
  }

  bool create_optix_pipeline(GPUOptixImplData* device) {
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) ProgramGroupRecord {
      alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE] = {};
      void* dummy = nullptr;
    };

    OptixPipelineLinkOptions link_options = {
      .maxTraceDepth = kOptiXMaxTraceDepth,
    };

    OptixProgramGroup all_program_groups[kOptiXMaxEntryPoints + 2] = {};
    for (uint32_t i = 0; i < descriptor.entry_point_count; ++i) {
      all_program_groups[i] = entry_points[i];
    }
    all_program_groups[descriptor.entry_point_count + 0] = main_hit_program_group;
    all_program_groups[descriptor.entry_point_count + 1] = main_miss_program_group;

    char compile_log[2048] = {};
    size_t compile_log_size = 0;
    if (device->optix_call_failed(optixPipelineCreate(device->optix, &pipeline_options, &link_options,  //
          all_program_groups, descriptor.entry_point_count + 2, compile_log, &compile_log_size, &pipeline))) {
      return false;
    }

    if (pipeline != nullptr) {
      if (device->optix_call_failed(optixPipelineSetStackSize(pipeline, 0u, 0u, 1u << 14u, link_options.maxTraceDepth)))
        return false;
    }

    for (uint32_t i = 0; i < descriptor.entry_point_count; ++i) {
      ProgramGroupRecord raygen_record = {};
      if (device->optix_call_failed(optixSbtRecordPackHeader(entry_points[i], &raygen_record)))
        return false;

      ProgramGroupRecord hit_record = {};
      if (device->optix_call_failed(optixSbtRecordPackHeader(main_hit_program_group, &hit_record)))
        return false;

      ProgramGroupRecord miss_record = {};
      if (device->optix_call_failed(optixSbtRecordPackHeader(main_miss_program_group, &miss_record)))
        return false;

      shader_binding_tables[i].raygenRecord = device->upload_to_shared_buffer(0, &raygen_record, sizeof(ProgramGroupRecord));
      shader_binding_tables[i].hitgroupRecordBase = device->upload_to_shared_buffer(0, &hit_record, sizeof(ProgramGroupRecord));
      shader_binding_tables[i].hitgroupRecordStrideInBytes = sizeof(ProgramGroupRecord);
      shader_binding_tables[i].hitgroupRecordCount = 1;
      shader_binding_tables[i].missRecordBase = device->upload_to_shared_buffer(0, &miss_record, sizeof(ProgramGroupRecord));
      shader_binding_tables[i].missRecordStrideInBytes = sizeof(ProgramGroupRecord);
      shader_binding_tables[i].missRecordCount = 1;
    }

    return true;
  }

  bool create_pipeline(GPUOptixImplData* device, const GPUPipeline::Descriptor& desc) {
    if (device->invalid_state())
      return false;

    descriptor = desc;
    switch (CUDACompileTarget(descriptor.compile_options)) {
      case CUDACompileTarget::PTX:
        return create_optix_pipeline(device);
      case CUDACompileTarget::Library: {
        return create_cuda_pipeline(device);
        default:
          break;
      }
    }
    return false;
  }

  GPUPipeline::Descriptor descriptor = {};
  OptixPipelineCompileOptions pipeline_options = {};
  OptixModule optix_module;

  OptixProgramGroup main_hit_program_group = {};
  OptixProgramGroup main_miss_program_group = {};
  OptixProgramGroup entry_points[kMaxEntryPoints] = {};
  OptixShaderBindingTable shader_binding_tables[kMaxEntryPoints] = {};

  OptixPipeline pipeline = {};

  struct CUDAFunction {
    CUfunction func = {};
    uint32_t min_grid_size = 0u;
    uint32_t max_block_size = 0u;
  };

  struct {
    CUmodule cuda_module = {};
    std::unordered_map<uint32_t, CUDAFunction> functions = {};
  } cuda = {};

  struct {
    std::unordered_map<uint32_t, uint32_t> functions = {};
  } optix = {};

  CUDACompileTarget target = CUDACompileTarget::PTX;
};

/*
 * #include "optix_pipeline.hxx"
 */

struct GPUAccelerationStructureImpl {
  GPUAccelerationStructureImpl(GPUOptixImplData* device, const GPUAccelerationStructure::Descriptor& desc) {
    if ((desc.vertex_count == 0) || (desc.triangle_count == 0)) {
      return;
    }

    CUdeviceptr vertex_buffer = device->buffer_pool.get(desc.vertex_buffer.handle).device_pointer();
    uint32_t triangle_array_flags = 0;

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.indexBuffer = device->buffer_pool.get(desc.index_buffer.handle).device_pointer();
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = desc.index_buffer_stride;
    build_input.triangleArray.numIndexTriplets = desc.triangle_count;
    build_input.triangleArray.vertexBuffers = &vertex_buffer;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = desc.vertex_buffer_stride;
    build_input.triangleArray.numVertices = desc.vertex_count;
    build_input.triangleArray.flags = &triangle_array_flags;
    build_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions build_options = {};
    build_options.motionOptions.numKeys = 1;
    build_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    build_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes memory_usage = {};
    if (device->optix_call_failed(optixAccelComputeMemoryUsage(device->optix, &build_options, &build_input, 1, &memory_usage))) {
      log::error("optixAccelComputeMemoryUsage failed");
      return;
    }

    void* temp_buffer = {};
    cudaMalloc(&temp_buffer, memory_usage.tempSizeInBytes);

    void* output_buffer = {};
    cudaMalloc(&output_buffer, memory_usage.outputSizeInBytes);

    OptixAccelEmitDesc emit_desc = {};
    emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    cudaMalloc(&reinterpret_cast<void*&>(emit_desc.result), sizeof(uint64_t));

    if (device->optix_call_failed(optixAccelBuild(device->optix, device->main_stream, &build_options, &build_input, 1, reinterpret_cast<CUdeviceptr>(temp_buffer),
          memory_usage.tempSizeInBytes, reinterpret_cast<CUdeviceptr>(output_buffer), memory_usage.outputSizeInBytes, &traversable, &emit_desc, 1))) {
      log::error("optixAccelBuild failed");
      return;
    }

    if (device->cuda_call_failed(cudaDeviceSynchronize())) {
      log::error("cudaDeviceSynchronize failed");
      return;
    }

    uint64_t compact_size = 0;
    cudaMemcpy(&compact_size, reinterpret_cast<void*>(emit_desc.result), sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(reinterpret_cast<void*>(emit_desc.result));

    cudaMalloc(&final_buffer, compact_size);
    if (device->optix_call_failed(optixAccelCompact(device->optix, device->main_stream, traversable, reinterpret_cast<CUdeviceptr>(final_buffer), compact_size, &traversable))) {
      log::error("optixAccelCompact failed");
      return;
    }

    cudaFree(temp_buffer);
    cudaFree(output_buffer);
  }

  GPUAccelerationStructureImpl() {
    cudaFree(final_buffer);
  }

  OptixTraversableHandle traversable = {};
  void* final_buffer = {};
};

#undef ETX_OPTIX_INCLUDES

device_pointer_t GPUOptixImplData::upload_to_shared_buffer(device_pointer_t ptr, void* data, uint64_t size) {
  if (invalid_state())
    return 0;

  auto& object = buffer_pool.get(shared_buffer.handle);

  if (ptr == 0) {
    uint64_t offset = shared_buffer_offset.fetch_add(size);
    ETX_CRITICAL(offset + size <= GPUOptixImplData::kSharedBufferSize);
    ptr = object.device_ptr + offset;
  }

  if (data != nullptr) {
    cudaMemcpy(reinterpret_cast<void*>(ptr), data, size, cudaMemcpyHostToDevice);
  }

  return ptr;
}

bool GPUOptixImplData::launch_optix(GPUPipelineOptixImpl& pipeline, const char* function, uint32_t dim_x, uint32_t dim_y, device_pointer_t params, uint64_t params_size) {
  if (invalid_state()) {
    return false;
  }

  if (dim_x * dim_y == 0) {
    return true;
  }

  uint32_t entry_point_index = kInvalidHandle;

  uint32_t func_hash = fnv1a32(function);
  auto f = pipeline.optix.functions.find(func_hash);
  if (f == pipeline.optix.functions.end()) {
    for (uint32_t index = 0; index < pipeline.descriptor.entry_point_count; ++index) {
      constexpr uint64_t kPrefixOffset = sizeof("_raygen__");
      if (strcmp(pipeline.descriptor.entry_points[index] + kPrefixOffset, function) == 0) {
        entry_point_index = index;
        break;
      }
    }
    pipeline.optix.functions[func_hash] = entry_point_index;
  } else {
    entry_point_index = f->second;
  }

  if (entry_point_index == kInvalidHandle) {
    return false;
  }

  auto result = optixLaunch(pipeline.pipeline, main_stream, params, params_size, pipeline.shader_binding_tables + entry_point_index, dim_x, dim_y, 1);
  return (result == OPTIX_SUCCESS);
}

bool GPUOptixImplData::launch_cuda(GPUPipelineOptixImpl& pipeline, const char* function, uint32_t dim_x, uint32_t dim_y, device_pointer_t params, uint64_t params_size) {
  if (invalid_state()) {
    return false;
  }

  if (dim_x * dim_y == 0) {
    return true;
  }

  GPUPipelineOptixImpl::CUDAFunction func = {};
  uint32_t func_hash = fnv1a32(function);
  auto f = pipeline.cuda.functions.find(func_hash);
  if (f == pipeline.cuda.functions.end()) {
    if (ETX_CUDA_FAILED(cuModuleGetFunction(&func.func, pipeline.cuda.cuda_module, function)))
      return false;

    int min_grid_size = 0;
    int max_block_size = 0;
    if (ETX_CUDA_FAILED(cuOccupancyMaxPotentialBlockSize(&min_grid_size, &max_block_size, func.func, nullptr, 0, 0)))
      return false;

    func.min_grid_size = uint32_t(min_grid_size);
    func.max_block_size = uint32_t(max_block_size);
    pipeline.cuda.functions[func_hash] = func;
  } else {
    func = f->second;
  }

  if ((func.func == nullptr) || (func.max_block_size == 0) || (func.min_grid_size == 0)) {
    return false;
  }

  void* call_params[] = {&params};

  uint2 block_size = {
    (dim_x > 1u) ? func.max_block_size : 1u,
    (dim_y > 1u) ? func.max_block_size : 1u,
  };

  // HACKS!
  if (dim_y < block_size.y) {
    block_size.y = dim_y;
    block_size.x = (dim_x > 1u) ? (func.max_block_size + block_size.y - 1) / block_size.y : 1u;
  }

  uint32_t scale_dim = 0;
  uint32_t* bptr = &block_size.x;
  while (block_size.x * block_size.y > uint32_t(device_properties.maxThreadsPerBlock)) {
    if (bptr[scale_dim] > 1) {
      bptr[scale_dim] = bptr[scale_dim] / 2;
    } else if (bptr[1u - scale_dim] > 1) {
      bptr[1u - scale_dim] = bptr[1u - scale_dim] / 2;
    } else {
      break;
    }
    scale_dim = 1u - scale_dim;
  }

  uint2 grid_size = {
    (dim_x + block_size.x - 1) / block_size.x,
    (dim_y + block_size.y - 1) / block_size.y,
  };

  CUresult call_result = cuLaunchKernel(func.func,  //
    grid_size.x, grid_size.y, 1u,                   //
    block_size.x, block_size.y, 1u,                 //
    0u, main_stream, reinterpret_cast<void**>(&call_params), nullptr);

  return ETX_CUDA_SUCCEED(call_result);
}

inline PipelineDesc parse_file(const nlohmann::json& json, const char* filename, char buffer[], uint64_t buffer_size) {
  return {};
  /*
  if (json_is_object(json) == false) {
    return {};
  }

  int buffer_pos = 0;
  PipelineDesc result = {};

  const char* key = nullptr;
  json_t* value = nullptr;
  json_object_foreach(json, key, value) {
    if (strcmp(key, "class") == 0) {
      auto cls = json_is_string(value) ? json_string_value(value) : "optix";
      if (strcmp(cls, "CUDA") == 0) {
        result.target = CUDACompileTarget::Library;
      }
    } else if (strcmp(key, "name") == 0) {
      result.name = json_is_string(value) ? json_string_value(value) : "undefined";
    } else if (strcmp(key, "source") == 0) {
      if (json_is_string(value) == false) {
        return {};
      }
      int pos_0 = buffer_pos;
      result.source = buffer + buffer_pos;
      buffer_pos += 1 + snprintf(buffer + buffer_pos, buffer_size - buffer_pos, "%s", filename);
      int pos_1 = buffer_pos;
      while ((pos_1 > pos_0) && (buffer[pos_1] != '/') && (buffer[pos_1] != '\\')) {
        --pos_1;
      }
      buffer[pos_1] = 0;
      buffer_pos += 1 + snprintf(buffer + pos_1, buffer_size - pos_1, "/%s", json_string_value(value));
    } else if (strcmp(key, "code") == 0) {
      if (json_is_string(value) == false) {
        return {};
      }
      int pos_0 = buffer_pos;
      result.code = buffer + buffer_pos;
      buffer_pos += 1 + snprintf(buffer + buffer_pos, buffer_size - buffer_pos, "%s", filename);
      int pos_1 = buffer_pos;
      while ((pos_1 > pos_0) && (buffer[pos_1] != '/') && (buffer[pos_1] != '\\')) {
        --pos_1;
      }
      buffer[pos_1] = 0;
      buffer_pos += 1 + snprintf(buffer + pos_1, buffer_size - pos_1, "/%s", json_string_value(value));
    } else if (strcmp(key, "entry-points") == 0) {
      if (json_is_array(value) == false) {
        return {};
      }

      int index = 0;
      json_t* entry_point_name = nullptr;
      json_array_foreach(value, index, entry_point_name) {
        if (index >= kMaxEntryPoints) {
          break;
        }

        if (json_is_string(entry_point_name)) {
          snprintf(result.entry_points[result.entry_point_count], sizeof(GPUPipeline::EntryPoint), "__raygen__%s", json_string_value(entry_point_name));
          result.entry_point_count += 1;
        }
      }
    }
  }

  result.completed = true;
  return result;
  */
}

bool GPUOptixImplData::compile(const char* json_filename, bool force, GPUPipeline::Descriptor& desc, std::vector<uint8_t>& binary_data) {
  return false;
  /*
  json_error_t error = {};
  json_t* json_data = json_load_file(json_filename, 0, &error);
  if (json_data == nullptr) {
    log::error("Erorr parsing json `%s`:[%d, %d] %s", json_filename, error.line, error.column, error.text);
    return {};
  }

  char info_buffer[2048] = {};
  auto ps_desc = parse_file(json_data, json_filename, info_buffer, sizeof(info_buffer));
  if (ps_desc.completed == false) {
    json_delete(json_data);
    log::error("Failed to load pipeline from json: `%s`", json_filename);
    return {};
  }

  bool has_output_file = false;
  if ((ps_desc.source != nullptr) && (ps_desc.source[0] != 0)) {
    if (auto file = fopen(ps_desc.source, "rb")) {
      has_output_file = true;
      fclose(file);
    }
  }

  if ((ps_desc.code != nullptr) && ((has_output_file == false) || force)) {
    if (compile_cuda(ps_desc.target, ps_desc.code, ps_desc.source, cuda_arch) == false) {
      return false;
    }
  }

  if ((ps_desc.source != nullptr) && (ps_desc.source[0] != 0)) {
    if (auto file = fopen(ps_desc.source, "rb")) {
      fclose(file);
    } else {
      return false;
    }
  }

  // std::vector<uint8_t> binary_data;
  if (load_binary_file(ps_desc.source, binary_data) == false) {
    return false;
  }

  desc.data = binary_data.data();
  desc.data_size = static_cast<uint32_t>(binary_data.size());
  memcpy(desc.entry_points, ps_desc.entry_points, ps_desc.entry_point_count * sizeof(GPUPipeline::EntryPoint));
  desc.entry_point_count = ps_desc.entry_point_count;
  desc.compile_options = uint32_t(ps_desc.target);
  return true;
  */
}

/***********************************************
 *
 * GPUOptixImpl
 *
 **********************************************/

GPUOptixImpl::GPUOptixImpl() {
  ETX_PIMPL_CREATE(GPUOptixImpl, Data);
  _private->shared_buffer = {_private->buffer_pool.alloc(_private, GPUBuffer::Descriptor{GPUOptixImplData::kSharedBufferSize, nullptr})};
}

GPUOptixImpl::~GPUOptixImpl() {
  auto& buffer = _private->buffer_pool.get(_private->shared_buffer.handle);
  buffer.release(_private);
  _private->buffer_pool.free(_private->shared_buffer.handle);
  ETX_PIMPL_DESTROY(GPUOptixImpl, Data);
}

bool GPUOptixImpl::rendering_enabled() {
  return (_private->main_stream != nullptr) && (_private->optix != nullptr);
}

GPUBuffer GPUOptixImpl::create_buffer(const GPUBuffer::Descriptor& desc) {
  return {_private->buffer_pool.alloc(_private, desc)};
}

void GPUOptixImpl::destroy_buffer(GPUBuffer buffer) {
  if (buffer.handle == kInvalidHandle)
    return;

  if (_private->cuda_call_failed(cudaDeviceSynchronize())) {
    log::error("Failed to synchronize device before the deletion of a buffer.");
  }

  auto& object = _private->buffer_pool.get(buffer.handle);
  object.release(_private);
  _private->buffer_pool.free(buffer.handle);
}

device_pointer_t GPUOptixImpl::get_buffer_device_pointer(GPUBuffer buffer) const {
  if (buffer.handle == kInvalidHandle)
    return 0;

  auto& object = _private->buffer_pool.get(buffer.handle);
  return object.device_ptr;
}

device_pointer_t GPUOptixImpl::upload_to_shared_buffer(device_pointer_t ptr, void* data, uint64_t size) {
  return _private->upload_to_shared_buffer(ptr, data, size);
}

device_pointer_t GPUOptixImpl::copy_to_buffer(GPUBuffer buffer, const void* src, uint64_t offset, uint64_t size) {
  if (_private->invalid_state())
    return 0;

  auto& object = _private->buffer_pool.get(buffer.handle);
  ETX_ASSERT(offset + size <= object.capacity);

  CUdeviceptr ptr_base = {};
  size_t ptr_size = {};
  device_pointer_t device_ptr = object.device_ptr + offset;
  CUresult query_result = cuMemGetAddressRange_v2(&ptr_base, &ptr_size, device_ptr);
  if (query_result != CUresult::CUDA_SUCCESS) {
    log::error("Failed to get mem info: %p", device_ptr);
  }

  if (_private->cuda_call_failed(cuMemcpyHtoD(device_ptr, src, size)))
    log::error("Failed to copy from buffer %p (%llu, %llu)", object.device_ptr, offset, size);

  return device_ptr;
}

void GPUOptixImpl::copy_from_buffer(GPUBuffer buffer, void* dst, uint64_t offset, uint64_t size) {
  if (_private->invalid_state())
    return;

  auto& object = _private->buffer_pool.get(buffer.handle);

  CUdeviceptr ptr_base = {};
  size_t ptr_size = {};
  CUdeviceptr device_ptr = object.device_ptr + offset;
  CUresult query_result = cuMemGetAddressRange_v2(&ptr_base, &ptr_size, device_ptr);
  if (query_result != CUresult::CUDA_SUCCESS) {
    log::error("Failed to get mem info: %p", device_ptr);
  }

  if (_private->cuda_call_failed(cuMemcpyDtoH(dst, device_ptr, size)))
    log::error("Failed to copy from buffer %p (%llu, %llu)", object.device_ptr, offset, size);
}

void GPUOptixImpl::clear_buffer(GPUBuffer buffer) {
  if (_private->invalid_state())
    return;

  auto& object = _private->buffer_pool.get(buffer.handle);
  cuMemsetD32(object.device_pointer(), 0u, object.capacity / sizeof(uint32_t));
}

GPUPipeline GPUOptixImpl::create_pipeline(const GPUPipeline::Descriptor& desc) {
  return {_private->pipeline_pool.alloc(_private, desc)};
}

void GPUOptixImpl::destroy_pipeline(GPUPipeline pipeline) {
  if (pipeline.handle == kInvalidHandle)
    return;

  if (_private->cuda_call_failed(cudaDeviceSynchronize())) {
    log::error("Failed to synchronize device before the deletion of a pipeline.");
  }

  auto& object = _private->pipeline_pool.get(pipeline.handle);
  object.release(_private);
  _private->pipeline_pool.free(pipeline.handle);
}

GPUPipeline GPUOptixImpl::create_pipeline_from_file(const char* json_filename, bool force_recompile) {
  GPUPipeline::Descriptor desc = {};
  std::vector<uint8_t> binary_data;
  if (_private->compile(json_filename, force_recompile, desc, binary_data) == false) {
    log::error("Failed to create pipeline from file: %s", json_filename);
    return {};
  }
  return {_private->pipeline_pool.alloc(_private, desc)};
}

void GPUOptixImpl::create_pipeline_from_files(TaskScheduler& scheduler, uint64_t file_count, const char* files[], GPUPipeline pipelines[], bool force) {
  if (file_count == 0)
    return;

  std::vector<GPUPipeline::Descriptor> desc(file_count);
  std::vector<std::vector<uint8_t>> binary_data(file_count);

  std::atomic<bool> success = {true};
  scheduler.execute(uint32_t(file_count), [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    for (uint32_t i = begin; i < end; ++i) {
      if (_private->compile(files[i], force, desc[i], binary_data[i]) == false) {
        log::error("Failed to create pipeline from file: %s", files[i]);
        success = false;
      }
    }
  });

  for (uint64_t i = 0; success && (i < file_count); ++i) {
    pipelines[i] = {_private->pipeline_pool.alloc(_private, desc[i])};
  }
}

bool GPUOptixImpl::launch(GPUPipeline pipeline, const char* function, uint32_t dim_x, uint32_t dim_y, device_pointer_t params, uint64_t params_size) {
  if (_private->invalid_state() || (pipeline.handle == kInvalidHandle) || (dim_x * dim_y == 0)) {
    return false;
  }

  auto& pp = _private->pipeline_pool.get(pipeline.handle);
  if (pp.target == CUDACompileTarget::Library) {
    return _private->launch_cuda(pp, function, dim_x, dim_y, params, params_size);
  } else if (pp.target == CUDACompileTarget::PTX) {
    return _private->launch_optix(pp, function, dim_x, dim_y, params, params_size);
  }

  return false;
}

GPUAccelerationStructure GPUOptixImpl::create_acceleration_structure(const GPUAccelerationStructure::Descriptor& desc) {
  return {_private->accelearaion_structure_pool.alloc(_private, desc)};
}

device_pointer_t GPUOptixImpl::get_acceleration_structure_device_pointer(GPUAccelerationStructure acc) {
  if (acc.handle == kInvalidHandle)
    return 0;

  const auto& object = _private->accelearaion_structure_pool.get(acc.handle);
  return object.traversable;
}

void GPUOptixImpl::destroy_acceleration_structure(GPUAccelerationStructure acc) {
  if (acc.handle == kInvalidHandle)
    return;

  if (_private->cuda_call_failed(cudaDeviceSynchronize())) {
    log::error("Failed to synchronize device before the deletion of an acceleration structure.");
  }
  _private->accelearaion_structure_pool.free(acc.handle);
}

void GPUOptixImpl::setup_denoiser(uint32_t dim_x, uint32_t dim_y) {
  _private->denoiser_setup = false;
  if (_private->denoiser == nullptr)
    return;

  if (_private->optix_call_failed(optixDenoiserComputeMemoryResources(_private->denoiser, dim_x, dim_y, &_private->denoiser_sizes))) {
    log::error("Failed to compute memory resources for denoiser");
    return;
  }

  _private->denoiser_image_size = {dim_x, dim_y};
  _private->denoiser_state = upload_to_shared_buffer(_private->denoiser_state, nullptr, _private->denoiser_sizes.stateSizeInBytes);
  _private->denoiser_scratch = upload_to_shared_buffer(_private->denoiser_scratch, nullptr, _private->denoiser_sizes.withOverlapScratchSizeInBytes);

  auto invoke = optixDenoiserSetup(_private->denoiser, _private->main_stream, dim_x, dim_y, _private->denoiser_state, _private->denoiser_sizes.stateSizeInBytes,
    _private->denoiser_scratch, _private->denoiser_sizes.withOverlapScratchSizeInBytes);

  if (_private->optix_call_failed(invoke)) {
    log::error("Failed to setup denoiser");
    return;
  }
  _private->denoiser_setup = true;
}

bool GPUOptixImpl::denoise(GPUBuffer input, GPUBuffer output) {
  if (_private->denoiser_setup == false)
    return false;

  OptixImage2D input_image = {
    .data = get_buffer_device_pointer(input),
    .width = _private->denoiser_image_size.x,
    .height = _private->denoiser_image_size.y,
    .rowStrideInBytes = static_cast<uint32_t>(_private->denoiser_image_size.x * sizeof(float4)),
    .format = OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT4,
  };

  OptixImage2D ouput_image = {
    .data = get_buffer_device_pointer(output),
    .width = _private->denoiser_image_size.x,
    .height = _private->denoiser_image_size.y,
    .rowStrideInBytes = static_cast<uint32_t>(_private->denoiser_image_size.x * sizeof(float4)),
    .format = OptixPixelFormat::OPTIX_PIXEL_FORMAT_FLOAT4,
  };

  OptixDenoiserParams params = {};
  OptixDenoiserGuideLayer guide_layer = {};
  OptixDenoiserLayer guide_layers[1] = {{
    .input = input_image,
    .output = ouput_image,
  }};

  auto invoke = optixDenoiserInvoke(_private->denoiser, _private->main_stream, &params, _private->denoiser_state, _private->denoiser_sizes.stateSizeInBytes, &guide_layer,
    guide_layers, 1, 0, 0, _private->denoiser_scratch, _private->denoiser_sizes.withOverlapScratchSizeInBytes);

  if (_private->optix_call_failed(invoke)) {
    log::error("Failed to invoke denoiser");
    return false;
  }

  return true;
}

const char* cuda_result(CUresult result) {
#define CASE(X) \
  case X:       \
    return #X

  switch (result) {
    CASE(CUDA_SUCCESS);
    CASE(CUDA_ERROR_INVALID_VALUE);
    CASE(CUDA_ERROR_OUT_OF_MEMORY);
    CASE(CUDA_ERROR_NOT_INITIALIZED);
    CASE(CUDA_ERROR_DEINITIALIZED);
    CASE(CUDA_ERROR_PROFILER_DISABLED);
    CASE(CUDA_ERROR_PROFILER_NOT_INITIALIZED);
    CASE(CUDA_ERROR_PROFILER_ALREADY_STARTED);
    CASE(CUDA_ERROR_PROFILER_ALREADY_STOPPED);
    CASE(CUDA_ERROR_STUB_LIBRARY);
    // CASE(CUDA_ERROR_DEVICE_UNAVAILABLE);
    CASE(CUDA_ERROR_NO_DEVICE);
    CASE(CUDA_ERROR_INVALID_DEVICE);
    CASE(CUDA_ERROR_DEVICE_NOT_LICENSED);
    CASE(CUDA_ERROR_INVALID_IMAGE);
    CASE(CUDA_ERROR_INVALID_CONTEXT);
    CASE(CUDA_ERROR_CONTEXT_ALREADY_CURRENT);
    CASE(CUDA_ERROR_MAP_FAILED);
    CASE(CUDA_ERROR_UNMAP_FAILED);
    CASE(CUDA_ERROR_ARRAY_IS_MAPPED);
    CASE(CUDA_ERROR_ALREADY_MAPPED);
    CASE(CUDA_ERROR_NO_BINARY_FOR_GPU);
    CASE(CUDA_ERROR_ALREADY_ACQUIRED);
    CASE(CUDA_ERROR_NOT_MAPPED);
    CASE(CUDA_ERROR_NOT_MAPPED_AS_ARRAY);
    CASE(CUDA_ERROR_NOT_MAPPED_AS_POINTER);
    CASE(CUDA_ERROR_ECC_UNCORRECTABLE);
    CASE(CUDA_ERROR_UNSUPPORTED_LIMIT);
    CASE(CUDA_ERROR_CONTEXT_ALREADY_IN_USE);
    CASE(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
    CASE(CUDA_ERROR_INVALID_PTX);
    CASE(CUDA_ERROR_INVALID_GRAPHICS_CONTEXT);
    CASE(CUDA_ERROR_NVLINK_UNCORRECTABLE);
    CASE(CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
    CASE(CUDA_ERROR_UNSUPPORTED_PTX_VERSION);
    CASE(CUDA_ERROR_JIT_COMPILATION_DISABLED);
    CASE(CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY);
    CASE(CUDA_ERROR_INVALID_SOURCE);
    CASE(CUDA_ERROR_FILE_NOT_FOUND);
    CASE(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND);
    CASE(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED);
    CASE(CUDA_ERROR_OPERATING_SYSTEM);
    CASE(CUDA_ERROR_INVALID_HANDLE);
    CASE(CUDA_ERROR_ILLEGAL_STATE);
    CASE(CUDA_ERROR_NOT_FOUND);
    CASE(CUDA_ERROR_NOT_READY);
    CASE(CUDA_ERROR_ILLEGAL_ADDRESS);
    CASE(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
    CASE(CUDA_ERROR_LAUNCH_TIMEOUT);
    CASE(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING);
    CASE(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
    CASE(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
    CASE(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE);
    CASE(CUDA_ERROR_CONTEXT_IS_DESTROYED);
    CASE(CUDA_ERROR_ASSERT);
    CASE(CUDA_ERROR_TOO_MANY_PEERS);
    CASE(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED);
    CASE(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED);
    CASE(CUDA_ERROR_HARDWARE_STACK_ERROR);
    CASE(CUDA_ERROR_ILLEGAL_INSTRUCTION);
    CASE(CUDA_ERROR_MISALIGNED_ADDRESS);
    CASE(CUDA_ERROR_INVALID_ADDRESS_SPACE);
    CASE(CUDA_ERROR_INVALID_PC);
    CASE(CUDA_ERROR_LAUNCH_FAILED);
    CASE(CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE);
    CASE(CUDA_ERROR_NOT_PERMITTED);
    CASE(CUDA_ERROR_NOT_SUPPORTED);
    CASE(CUDA_ERROR_SYSTEM_NOT_READY);
    CASE(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH);
    CASE(CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE);
    CASE(CUDA_ERROR_MPS_CONNECTION_FAILED);
    CASE(CUDA_ERROR_MPS_RPC_FAILURE);
    CASE(CUDA_ERROR_MPS_SERVER_NOT_READY);
    CASE(CUDA_ERROR_MPS_MAX_CLIENTS_REACHED);
    CASE(CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED);
    CASE(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    CASE(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
    CASE(CUDA_ERROR_STREAM_CAPTURE_MERGE);
    CASE(CUDA_ERROR_STREAM_CAPTURE_UNMATCHED);
    CASE(CUDA_ERROR_STREAM_CAPTURE_UNJOINED);
    CASE(CUDA_ERROR_STREAM_CAPTURE_ISOLATION);
    CASE(CUDA_ERROR_STREAM_CAPTURE_IMPLICIT);
    CASE(CUDA_ERROR_CAPTURED_EVENT);
    CASE(CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD);
    CASE(CUDA_ERROR_TIMEOUT);
    CASE(CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE);
    CASE(CUDA_ERROR_EXTERNAL_DEVICE);
    CASE(CUDA_ERROR_UNKNOWN);
    default:
      break;
  }
  return "Unknown CUDA error";
#undef CASE
}

}  // namespace etx
