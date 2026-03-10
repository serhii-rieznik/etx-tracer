#include "headless_render_context.hxx"

namespace etx {

void HeadlessRenderContext::init() {
  RHIInitInfo info = {
    .backend = RHIBackend::Vulkan,
    .enable_validation = ETX_DEBUG,
    .headless = true,
  };
#if defined(ETX_PLATFORM_APPLE)
  info.backend = RHIBackend::Metal;
#endif

  _context = RHIContext::create(info);
  if (_context.valid()) {
    _context.initialize_headless();
  }
}

void HeadlessRenderContext::cleanup() {
  if (_context.valid()) {
    _context.wait_idle();
    _context = {};
  }
}

void HeadlessRenderContext::begin_frame() {
  if (_context.valid()) {
    _context.begin_frame();
  }
}

void HeadlessRenderContext::end_frame() {
  if (_context.valid()) {
    _context.end_frame();
  }
}

}  // namespace etx
