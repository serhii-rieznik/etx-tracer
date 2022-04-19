#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/log/log.hxx>
#include <etx/gpu/optix.hxx>

#include <jansson.h>

#include <cuda_compiler/cuda_compiler_lib.hxx>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define ETX_CUDA_CALL_FAILED(F) ((F) != cudaSuccess)
#define ETX_OPTIX_CALL_FAILED(F) ((F) != OPTIX_SUCCESS)

namespace etx {

struct GPUBufferOptixImpl;
struct GPUPipelineOptixImpl;

struct GPUOptixImplData {
  CUcontext cuda_context = {};
  CUstream main_stream = {};

  void* optix_handle = nullptr;
  OptixDeviceContext optix = {};

  ObjectIndexPool<GPUBufferOptixImpl> buffer_pool;
  ObjectIndexPool<GPUPipelineOptixImpl> pipeline_pool;

  GPUBuffer system_buffer_handle = {};

  GPUOptixImplData() {
    buffer_pool.init(1024u);
    pipeline_pool.init(1024u);

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

    pipeline_pool.cleanup();
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
      log::info("OptiX: [%s] %s", tag, message);
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

  GPUBufferOptixImpl& system_buffer() {
    return buffer_pool.get(system_buffer_handle.handle);
  }
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

  GPUBufferOptixImpl(const GPUBuffer::Descriptor& desc) {
    capacity = align_up(desc.size, 16u);

    if ETX_CUDA_CALL_FAILED (cudaMalloc(&device_ptr, capacity)) {
      log::error("Failed to create CUDA buffer with size: %llu", capacity);
      return;
    }

    if (desc.data != nullptr) {
      if ETX_CUDA_CALL_FAILED (cudaMemcpy(device_ptr, desc.data, desc.size, cudaMemcpyKind::cudaMemcpyHostToDevice))
        log::error("Failed to copy content to CUDA buffer %p from %p with size %llu", device_ptr, desc.data, desc.size);
    }
  }

  ~GPUBufferOptixImpl() {
    release();
  }

  void release() {
    if ETX_CUDA_CALL_FAILED (cudaFree(device_ptr)) {
      log::error("Failed to free CUDA buffer: %p", device_ptr);
    }
    device_ptr = nullptr;
  }

  CUdeviceptr upload(void* data, uint64_t size) {
    ETX_CRITICAL(upload_offset + size <= capacity);
    auto result = device_ptr + upload_offset;
    upload_offset += size;
    return reinterpret_cast<CUdeviceptr>(result);
  }

  uint8_t* device_ptr = nullptr;
  uint64_t capacity = 0;
  uint64_t upload_offset = 0;
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
    destroy_module();
  }

  bool create_module(GPUOptixImplData* device, const GPUPipeline::Descriptor& desc) {
    OptixModuleCompileOptions module_options = {
      .maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
      .optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
      .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
    };
    pipeline_options = {
      .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
      .numPayloadValues = static_cast<int>(desc.payload_count & 0x000000ff),
      .numAttributeValues = 2,
      .exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_USER,
      .pipelineLaunchParamsVariableName = "input",
    };

    char compile_log[4096] = {};
    size_t compile_log_size = sizeof(compile_log);
    if ETX_OPTIX_CALL_FAILED (optixModuleCreateFromPTX(device->optix, &module_options, &pipeline_options,  //
                                reinterpret_cast<const char*>(desc.data), desc.data_size, compile_log, &compile_log_size, &optix_module)) {
      log::error("optixModuleCreateFromPTX failed");
      if (compile_log_size > 1) {
        log::error(compile_log);
      }
      destroy_module();
      return false;
    }

    if (compile_log_size > 1) {
      log::warning(compile_log);
    }

    ETX_ASSERT(desc.raygen != nullptr);
    {
      OptixProgramGroupDesc program_desc = {};
      program_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      program_desc.raygen.module = optix_module;
      program_desc.raygen.entryFunctionName = desc.raygen;
      OptixProgramGroupOptions program_options = {};
      if ETX_OPTIX_CALL_FAILED (optixProgramGroupCreate(device->optix, &program_desc, 1, &program_options, compile_log, &compile_log_size, &raygen)) {
        log::error("optixProgramGroupCreate failed");
        if (compile_log_size > 1) {
          log::error(compile_log);
        }
        destroy_module();
        return false;
      }

      if (compile_log_size > 1) {
        log::warning(compile_log);
      }
    }

    group_count = desc.entry_count;

    for (uint64_t i = 0; i < desc.entry_count; ++i) {
      auto& entry = desc.entries[i];
      auto& group = groups[i];

      if ((entry.closest_hit != nullptr) || (entry.any_hit != nullptr)) {
        OptixProgramGroupDesc program_desc = {};
        program_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (entry.closest_hit != nullptr) {
          program_desc.hitgroup.moduleCH = optix_module;
          program_desc.hitgroup.entryFunctionNameCH = entry.closest_hit;
        }
        if (entry.any_hit != nullptr) {
          program_desc.hitgroup.moduleAH = optix_module;
          program_desc.hitgroup.entryFunctionNameAH = entry.any_hit;
        }
        OptixProgramGroupOptions program_options = {};

        if ETX_OPTIX_CALL_FAILED (optixProgramGroupCreate(device->optix, &program_desc, 1, &program_options, compile_log, &compile_log_size, &group.hit)) {
          log::error("optixProgramGroupCreate failed");
          if (compile_log_size > 1) {
            log::error(compile_log);
          }
          destroy_module();
          return false;
        }

        if (compile_log_size > 1) {
          log::warning(compile_log);
        }
      }

      if (entry.miss != nullptr) {
        OptixProgramGroupDesc program_desc = {};
        program_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        program_desc.miss.module = optix_module;
        program_desc.miss.entryFunctionName = entry.miss;
        OptixProgramGroupOptions program_options = {};
        if ETX_OPTIX_CALL_FAILED (optixProgramGroupCreate(device->optix, &program_desc, 1, &program_options, compile_log, &compile_log_size, &group.miss)) {
          log::error("optixProgramGroupCreate failed");
          if (compile_log_size > 1) {
            log::error(compile_log);
          }
          destroy_module();
          return false;
        }

        if (compile_log_size > 1) {
          log::warning(compile_log);
        }
      }

      if (entry.exception != nullptr) {
        OptixProgramGroupDesc program_desc = {};
        program_desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        program_desc.miss.module = optix_module;
        program_desc.miss.entryFunctionName = entry.exception;
        OptixProgramGroupOptions program_options = {};
        if ETX_OPTIX_CALL_FAILED (optixProgramGroupCreate(device->optix, &program_desc, 1, &program_options, compile_log, &compile_log_size, &group.exception)) {
          log::error("optixProgramGroupCreate failed");
          if (compile_log_size > 1) {
            log::error(compile_log);
          }
          destroy_module();
          return false;
        }

        if (compile_log_size > 1) {
          log::warning(compile_log);
        }
      }
    }

    return true;
  }

  void destroy_module() {
    if (raygen != nullptr) {
      if ETX_OPTIX_CALL_FAILED (optixProgramGroupDestroy(raygen))
        log::error("optixProgramGroupDestroy failed");
      raygen = {};
    }

    for (uint32_t i = 0; i < group_count; ++i) {
      if (groups[i].hit != nullptr) {
        if ETX_OPTIX_CALL_FAILED (optixProgramGroupDestroy(groups[i].hit))
          log::error("optixProgramGroupDestroy failed");
      }
      if (groups[i].miss != nullptr) {
        if ETX_OPTIX_CALL_FAILED (optixProgramGroupDestroy(groups[i].miss))
          log::error("optixProgramGroupDestroy failed");
      }
      if (groups[i].exception != nullptr) {
        if ETX_OPTIX_CALL_FAILED (optixProgramGroupDestroy(groups[i].exception))
          log::error("optixProgramGroupDestroy failed");
      }
      groups[i] = {};
    }
    group_count = 0;

    if (optix_module != nullptr) {
      if ETX_OPTIX_CALL_FAILED (optixModuleDestroy(optix_module))
        log::error("Failed to destroy OptiX module");
      optix_module = {};
    }
  }

  bool create_pipeline(GPUOptixImplData* device, const GPUPipeline::Descriptor& desc) {
    OptixPipelineLinkOptions link_options = {
      .maxTraceDepth = desc.max_trace_depth,
    };

    OptixProgramGroup program_groups[kMaxModuleGroups * 3] = {};
    ProgramGroupRecord hit_records[kMaxModuleGroups * 3] = {};
    ProgramGroupRecord miss_records[kMaxModuleGroups * 3] = {};

    ProgramGroupRecord raygen_record = {};
    program_groups[0] = raygen;
    if ETX_OPTIX_CALL_FAILED (optixSbtRecordPackHeader(raygen, &raygen_record))
      return false;

    uint32_t program_group_count = 1;

    uint32_t hit_record_count = 0;
    uint32_t miss_record_count = 0;
    for (uint32_t i = 0; i < group_count; ++i) {
      if (groups[i].hit != nullptr) {
        program_groups[program_group_count] = groups[i].hit;
        if ETX_OPTIX_CALL_FAILED (optixSbtRecordPackHeader(program_groups[program_group_count], hit_records + hit_record_count))
          return false;
        ++program_group_count;
        ++hit_record_count;
      }
      if (groups[i].miss != nullptr) {
        program_groups[program_group_count] = groups[i].miss;
        if ETX_OPTIX_CALL_FAILED (optixSbtRecordPackHeader(program_groups[program_group_count], miss_records + miss_record_count))
          return false;
        ++program_group_count;
        ++miss_record_count;
      }
    }

    char compile_log[2048] = {};
    size_t compile_log_size = 0;
    if ETX_OPTIX_CALL_FAILED (optixPipelineCreate(device->optix, &pipeline_options, &link_options, program_groups, program_group_count, compile_log, &compile_log_size, &pipeline))
      return false;

    if (pipeline != nullptr) {
      if ETX_OPTIX_CALL_FAILED (optixPipelineSetStackSize(pipeline, 0u, 0u, 1u << 14u, link_options.maxTraceDepth))
        return false;
    }

    auto& system_buffer = device->system_buffer();

    shader_binding_table.raygenRecord = system_buffer.upload(&raygen_record, sizeof(ProgramGroupRecord));

    if (hit_record_count > 0) {
      shader_binding_table.hitgroupRecordBase = system_buffer.upload(hit_records, sizeof(ProgramGroupRecord) * hit_record_count);
      shader_binding_table.hitgroupRecordStrideInBytes = sizeof(ProgramGroupRecord);
      shader_binding_table.hitgroupRecordCount = hit_record_count;
    }
    if (miss_record_count > 0) {
      shader_binding_table.missRecordBase = system_buffer.upload(miss_records, sizeof(ProgramGroupRecord) * miss_record_count);
      shader_binding_table.missRecordStrideInBytes = sizeof(ProgramGroupRecord);
      shader_binding_table.missRecordCount = miss_record_count;
    }

    return true;
  }

  struct ModuleGroups {
    OptixProgramGroup hit = {};
    OptixProgramGroup miss = {};
    OptixProgramGroup exception = {};
  };

  struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) ProgramGroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* dummy = nullptr;
  };

  constexpr static uint64_t kMaxModuleGroups = 8;

  OptixPipelineCompileOptions pipeline_options = {};
  OptixModule optix_module;

  OptixProgramGroup raygen = {};
  ModuleGroups groups[kMaxModuleGroups] = {};
  uint32_t group_count = 0;

  OptixPipeline pipeline = {};
  OptixShaderBindingTable shader_binding_table = {};
};

//
#include "optix_pipeline.hxx"

#undef ETX_OPTIX_INCLUDES

ETX_PIMPL_IMPLEMENT_ALL(GPUOptixImpl, Data)

GPUBuffer GPUOptixImpl::create_buffer(const GPUBuffer::Descriptor& desc) {
  return {_private->buffer_pool.alloc(desc)};
}

void GPUOptixImpl::destroy_buffer(GPUBuffer buffer) {
  if (buffer.handle == 0)
    return;

  _private->buffer_pool.free(buffer.handle);
}

uint64_t GPUOptixImpl::get_buffer_device_handle(GPUBuffer buffer) const {
  auto& object = _private->buffer_pool.get(buffer.handle);
  return reinterpret_cast<uint64_t>(object.device_ptr);
}

void GPUOptixImpl::copy_from_buffer(GPUBuffer buffer, void* dst, uint64_t offset, uint64_t size) {
  auto& object = _private->buffer_pool.get(buffer.handle);
  if ETX_CUDA_CALL_FAILED (cudaMemcpy(dst, reinterpret_cast<const uint8_t*>(object.device_ptr) + offset, size, cudaMemcpyDeviceToHost))
    log::error("Failed to copy from buffer %p (%llu, %llu)", object.device_ptr, offset, size);
}

GPUPipeline GPUOptixImpl::create_pipeline(const GPUPipeline::Descriptor& desc) {
  return {_private->pipeline_pool.alloc(_private, desc)};
}

void GPUOptixImpl::destroy_pipeline(GPUPipeline buffer) {
  if (buffer.handle == 0)
    return;

  _private->pipeline_pool.free(buffer.handle);
}

struct PipelineDesc {
  const char* name = nullptr;
  const char* code = nullptr;
  const char* source = nullptr;
  const char* raygen = nullptr;
  GPUPipeline::Entry modules[GPUPipelineOptixImpl::kMaxModuleGroups] = {};
  uint32_t module_count = 0;
  uint32_t max_trace_depth = 1;
  uint32_t payload_size = 0;
  bool completed = false;
};

inline PipelineDesc parse_file(json_t* json, const char* filename, char buffer[], uint64_t buffer_size) {
  if (json_is_object(json) == false) {
    return {};
  }

  int buffer_pos = 0;
  PipelineDesc result = {};

  const char* key = nullptr;
  json_t* value = nullptr;
  json_object_foreach(json, key, value) {
    if (strcmp(key, "name") == 0) {
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
    } else if (strcmp(key, "max_trace_depth") == 0) {
      result.max_trace_depth = static_cast<uint32_t>(json_is_number(value) ? json_number_value(value) : result.max_trace_depth);
    } else if (strcmp(key, "payload_size") == 0) {
      result.payload_size = static_cast<uint32_t>(json_is_number(value) ? json_number_value(value) : result.payload_size);
    } else if (strcmp(key, "raygen") == 0) {
      if (json_is_string(value) == false) {
        return {};
      }
      result.raygen = buffer + buffer_pos;
      buffer_pos += 1 + snprintf(buffer + buffer_pos, buffer_size - buffer_pos, "__raygen__%s", json_string_value(value));
    } else if (strcmp(key, "modules") == 0) {
      if (json_is_array(value) == false) {
        return {};
      }

      int index = 0;
      json_t* module_value = nullptr;
      json_array_foreach(value, index, module_value) {
        if (index >= GPUPipelineOptixImpl::kMaxModuleGroups) {
          break;
        }

        if (json_is_object(module_value) == false) {
          continue;
        }

        const char* program_name = nullptr;
        json_t* program_value = nullptr;
        json_object_foreach(module_value, program_name, program_value) {
          if (json_is_string(program_value) == false) {
            continue;
          }

          if (strcmp(program_name, "name") == 0) {
            // TODO : add name?
          } else if (strcmp(program_name, "closest_hit") == 0) {
            result.modules[index].closest_hit = buffer + buffer_pos;
            buffer_pos += 1 + snprintf(buffer + buffer_pos, buffer_size - buffer_pos, "__closesthit__%s", json_string_value(program_value));
          } else if (strcmp(program_name, "any_hit") == 0) {
            result.modules[index].any_hit = buffer + buffer_pos;
            buffer_pos += 1 + snprintf(buffer + buffer_pos, buffer_size - buffer_pos, "__anyhit__%s", json_string_value(program_value));
          } else if (strcmp(program_name, "miss") == 0) {
            result.modules[index].miss = buffer + buffer_pos;
            buffer_pos += 1 + snprintf(buffer + buffer_pos, buffer_size - buffer_pos, "__miss__%s", json_string_value(program_value));
          } else if (strcmp(program_name, "exception") == 0) {
            result.modules[index].exception = buffer + buffer_pos;
            buffer_pos += 1 + snprintf(buffer + buffer_pos, buffer_size - buffer_pos, "__exception__%s", json_string_value(program_value));
          }
        }

        result.module_count += 1;
      }
    }
  }

  result.completed = true;
  return result;
}

GPUPipeline GPUOptixImpl::create_pipeline_from_file(const char* json_filename, bool force_recompile) {
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

  bool has_ptx_file = false;
  if (auto file = fopen(ps_desc.source, "rb")) {
    has_ptx_file = true;
    fclose(file);
  }

  if ((ps_desc.code != nullptr) && ((has_ptx_file == false) || force_recompile)) {
    if (compile_nvcc_file(ps_desc.code, ps_desc.source) == false) {
      return {};
    }
  }

  std::vector<uint8_t> ptx;
  if (load_binary_file(ps_desc.source, ptx) == false) {
    return {};
  }

  GPUPipeline::Descriptor desc;
  desc.data = ptx.data();
  desc.data_size = static_cast<uint32_t>(ptx.size());
  desc.raygen = ps_desc.raygen;
  desc.entries = ps_desc.modules;
  desc.entry_count = ps_desc.module_count;
  desc.payload_count = ps_desc.payload_size;
  desc.max_trace_depth = ps_desc.max_trace_depth;
  return create_pipeline(desc);
}

bool GPUOptixImpl::launch(GPUPipeline pipeline, uint32_t dim_x, uint32_t dim_y, device_pointer_t params, uint64_t params_size) {
  if ((pipeline.handle == 0) || (dim_x * dim_y == 0)) {
    return false;
  }

  const auto& pp = _private->pipeline_pool.get(pipeline.handle);
  auto result = optixLaunch(pp.pipeline, _private->main_stream, params, params_size, &pp.shader_binding_table, dim_x, dim_y, 1);
  return (result == OPTIX_SUCCESS);
}

}  // namespace etx
