#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/host/pool.hxx>

namespace etx {

struct GPUBuffer {
  struct Descriptor {
    uint64_t size = 0;
    uint8_t* data = nullptr;
  };
  uint32_t handle = 0;
};

struct GPUDevice {
  GPUDevice() = default;
  virtual ~GPUDevice() = default;

  virtual GPUBuffer create_buffer(const GPUBuffer::Descriptor&) = 0;
  virtual void destroy_buffer(GPUBuffer) = 0;

  static GPUDevice* create_optix_device();
  static void free_device(GPUDevice*);
};

}  // namespace etx
