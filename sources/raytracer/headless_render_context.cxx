#include "headless_render_context.hxx"
namespace etx {
namespace {

RHIBackend select_default_backend() {
#if ETX_PLATFORM_APPLE
  return RHIBackend::Metal;
#else
  return RHIBackend::Vulkan;
#endif
}

}  // namespace

void HeadlessRenderContext::init() {
  RHIInitInfo info = {
    .backend = select_default_backend(),
    .enable_validation = ETX_DEBUG,
    .headless = true,
  };

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
