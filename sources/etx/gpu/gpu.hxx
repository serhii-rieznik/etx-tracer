#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/pool.hxx>

namespace etx {

using device_pointer_t = uint64_t;

struct GPUBuffer {
  struct Descriptor {
    uint64_t size = 0;
    uint8_t* data = nullptr;
  };
  uint32_t handle = 0;
};

struct GPUPipeline {
  struct Entry {
    const char* closest_hit = nullptr;
    const char* any_hit = nullptr;
    const char* miss = nullptr;
    const char* exception = nullptr;
  };

  struct Descriptor {
    uint8_t* data = nullptr;
    uint64_t data_size = 0;

    const char* raygen = nullptr;

    Entry* entries = nullptr;
    uint32_t entry_count = 0;

    uint32_t payload_count = 0;
    uint32_t max_trace_depth = 2;
  };

  uint32_t handle = 0;
};

struct GPUAccelerationStructure {
  struct Descriptor {
    GPUBuffer vertex_buffer = {};
    uint32_t vertex_buffer_stride = 0;
    uint32_t vertex_count = 0;

    GPUBuffer index_buffer = {};
    uint32_t index_buffer_stride = 0;
    uint32_t triangle_count = 0;
  };

  uint32_t handle = 0;
};

struct GPUDevice {
  GPUDevice() = default;
  virtual ~GPUDevice() = default;

  virtual GPUBuffer create_buffer(const GPUBuffer::Descriptor&) = 0;
  virtual void destroy_buffer(GPUBuffer) = 0;
  virtual device_pointer_t get_buffer_device_pointer(GPUBuffer) const = 0;
  virtual device_pointer_t upload_to_shared_buffer(device_pointer_t ptr, void* data, uint64_t size) = 0;
  virtual void copy_from_buffer(GPUBuffer, void* dst, uint64_t offset, uint64_t size) = 0;

  virtual GPUPipeline create_pipeline(const GPUPipeline::Descriptor&) = 0;
  virtual GPUPipeline create_pipeline_from_file(const char* filename, bool force_recompile) = 0;
  virtual void destroy_pipeline(GPUPipeline) = 0;

  virtual GPUAccelerationStructure create_acceleration_structure(const GPUAccelerationStructure::Descriptor&) = 0;
  virtual void destroy_acceleration_structure(GPUAccelerationStructure) = 0;

  virtual bool launch(GPUPipeline, uint32_t dim_x, uint32_t dim_y, device_pointer_t params, uint64_t params_size) = 0;

  static GPUDevice* create_optix_device();
  static void free_device(GPUDevice*);
};

}  // namespace etx
