#include <etx/rhi/rhi.hxx>
#include <etx/rhi/vulkan/vk_rhi.hxx>
#include <etx/rhi/metal/mt_rhi.hxx>

#include <etx/core/log.hxx>

namespace etx {

RHIContext* create_rhi_context(const RHIInitInfo& info) {
  switch (info.backend) {
    case RHIBackend::Vulkan: {
      auto context = new VKContext();
      return context;
    }
    case RHIBackend::Metal: {
      auto context = new MTContext();
      return context;
    }
    default:
      log::error("Unsupported RHI backend: {}", static_cast<uint32_t>(info.backend));
      return nullptr;
  }
}

void destroy_rhi_context(RHIContext* context) {
  if (context == nullptr) {
    return;
  }

  delete context;
}

}  // namespace etx
