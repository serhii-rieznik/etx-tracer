#include <etx/gpu/gpu.hxx>

namespace etx {

struct GPUDummyDevice : public GPUDevice {
  bool rendering_enabled() override {
    return false;
  }
  GPUBuffer create_buffer(const GPUBuffer::Descriptor&) override {
    return {};
  }
  device_pointer_t get_buffer_device_pointer(GPUBuffer) const override {
    return {};
  }
  device_pointer_t upload_to_shared_buffer(device_pointer_t ptr, void* data, uint64_t size) override {
    return {};
  }
  device_pointer_t copy_to_buffer(GPUBuffer, const void* src, uint64_t offset, uint64_t size) override {
    return {};
  }
  void copy_from_buffer(GPUBuffer, void* dst, uint64_t offset, uint64_t size) override {
  }
  void destroy_buffer(GPUBuffer) override {
  }
  GPUPipeline create_pipeline(const GPUPipeline::Descriptor&) override {
    return {};
  }
  GPUPipeline create_pipeline_from_file(const char* filename, bool force_recompile) override {
    return {};
  }
  void destroy_pipeline(GPUPipeline) override {
  }
  GPUAccelerationStructure create_acceleration_structure(const GPUAccelerationStructure::Descriptor&) override {
    return {};
  }
  device_pointer_t get_acceleration_structure_device_pointer(GPUAccelerationStructure) override {
    return {};
  }
  void destroy_acceleration_structure(GPUAccelerationStructure) override {
  }
  bool launch(GPUPipeline, uint32_t dim_x, uint32_t dim_y, device_pointer_t params, uint64_t params_size) override {
    return false;
  }
};

GPUDevice* GPUDevice::create_optix_device() {
  static GPUDummyDevice device;
  return &device;
}

void GPUDevice::free_device(GPUDevice*) {
}

}  // namespace etx
