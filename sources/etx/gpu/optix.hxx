#pragma once

#include <etx/gpu/gpu.hxx>

namespace etx {

struct GPUOptixImpl : public GPUDevice {
  GPUOptixImpl();
  ~GPUOptixImpl() override;

  GPUBuffer create_buffer(const GPUBuffer::Descriptor& desc) override;
  void destroy_buffer(GPUBuffer) override;
  device_pointer_t get_buffer_device_pointer(GPUBuffer) const override;
  device_pointer_t upload_to_shared_buffer(device_pointer_t ptr, void* data, uint64_t size) override;
  void copy_from_buffer(GPUBuffer, void* dst, uint64_t offset, uint64_t size) override;

  GPUPipeline create_pipeline(const GPUPipeline::Descriptor&) override;
  GPUPipeline create_pipeline_from_file(const char*, bool) override;
  void destroy_pipeline(GPUPipeline) override;

  GPUAccelerationStructure create_acceleration_structure(const GPUAccelerationStructure::Descriptor&) override;
  void destroy_acceleration_structure(GPUAccelerationStructure) override;

  bool launch(GPUPipeline, uint32_t dim_x, uint32_t dim_y, device_pointer_t params, uint64_t params_size) override;

  ETX_PIMPL_DECLARE(GPUOptixImpl, Data, 2048);
};

}  // namespace etx