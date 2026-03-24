#include <etx/rhi/rhi.hxx>

#include <etx/core/log.hxx>
#include <utility>

namespace etx {

#if defined(ETX_PLATFORM_WINDOWS)
void create_vulkan_context(RHIContext& context, const RHIInitInfo& info);
#elif defined(ETX_PLATFORM_APPLE)
void create_vulkan_context(RHIContext& context, const RHIInitInfo& info);
#endif

RHIContext RHIContext::create(const RHIInitInfo& info) {
  RHIContext context = {};

#if defined(ETX_PLATFORM_WINDOWS)
  if (info.backend != RHIBackend::Vulkan) {
    log::warning("Requested backend %u on Windows; forcing Vulkan", static_cast<uint32_t>(info.backend));
  }
  create_vulkan_context(context, info);
#elif defined(ETX_PLATFORM_APPLE)
  if (info.backend != RHIBackend::Vulkan) {
    log::warning("Requested backend %u on Apple platform during MoltenVK phase; forcing Vulkan", static_cast<uint32_t>(info.backend));
  }
  create_vulkan_context(context, info);
#else
  (void)info;
  log::error("Unsupported platform for RHIContext::create");
#endif

  return context;
}

RHIContext::~RHIContext() {
  if (_impl != nullptr) {
    destroy_backend();
  }
}

RHIContext::RHIContext(RHIContext&& other) noexcept {
  move_from(std::move(other));
}

RHIContext& RHIContext::operator=(RHIContext&& other) noexcept {
  if (this != &other) {
    if (_impl != nullptr) {
      destroy_backend();
    }
    move_from(std::move(other));
  }
  return *this;
}

}  // namespace etx
