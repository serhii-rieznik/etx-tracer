#include <etx/rhi/rhi.hxx>
#include <etx/rhi/vulkan/vk_rhi.hxx>
#include <etx/rhi/metal/mt_rhi.hxx>

#include <etx/core/log.hxx>

namespace etx {

RHIContext* RHIContext::create(const RHIInitInfo& info) {
  switch (info.backend) {
    case RHIBackend::Vulkan:
      return new VKContext(info);
    case RHIBackend::Metal:
      return new MTContext();
    default:
      log::error("Unsupported RHI backend: {}", static_cast<uint32_t>(info.backend));
      return nullptr;
  }
}

void RHIContext::release(RHIContext* context) {
  delete context;
}

}  // namespace etx
