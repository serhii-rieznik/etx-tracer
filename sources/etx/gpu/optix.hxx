#pragma once

#include <etx/gpu/gpu.hxx>

namespace etx {

struct GPUOptixImpl : public GPUDevice {
  GPUOptixImpl();
  ~GPUOptixImpl() override;

  GPUBuffer create_buffer(const GPUBuffer::Descriptor& desc) override;
  void destroy_buffer(GPUBuffer) override;

  ETX_PIMPL_DECLARE(GPUOptixImpl, Data, 2048);
};

}  // namespace etx