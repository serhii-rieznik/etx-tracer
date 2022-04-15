#pragma once

#include <etx/gpu/gpu.hxx>

namespace etx {

struct GPUOptixImpl : public GPUDevice {
  GPUOptixImpl();
  ~GPUOptixImpl() override;

  GPUBuffer create_buffer(const GPUBuffer::Descriptor& desc) override;
  void destroy_buffer(GPUBuffer) override;
  uint64_t get_buffer_device_handle(GPUBuffer) const override;
  void copy_from_buffer(GPUBuffer, void* dst, uint64_t offset, uint64_t size) override;

  GPUPipeline create_pipeline(const GPUPipeline::Descriptor&) override;
  GPUPipeline create_pipeline_from_file(const char*, bool) override;
  void destroy_pipeline(GPUPipeline) override;

  ETX_PIMPL_DECLARE(GPUOptixImpl, Data, 2048);
};

}  // namespace etx