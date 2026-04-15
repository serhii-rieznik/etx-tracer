#include "runtime_output.hxx"

namespace etx {
namespace {

RHITextureFormat default_offscreen_output_format() {
#if ETX_PLATFORM_WINDOWS
  return RHITextureFormat::B8G8R8A8_UNORM;
#else
  return RHITextureFormat::R8G8B8A8_UNORM;
#endif
}

}  // namespace

bool RuntimeOutput::init(RHIContext& rhi, const RuntimeOutputConfig& config) {
  shutdown(rhi);

  _config = config;
  if (_config.color_format == RHITextureFormat::Undefined) {
    _config.color_format = default_offscreen_output_format();
  }

  if (_config.mode == RuntimeMode::Desktop) {
    rhi.create_swapchain(_config.native_window, _config.width, _config.height);
    _output_format = rhi.get_swapchain_format();
    _extent = rhi.get_swapchain_extent();
    return (_output_format != RHITextureFormat::Undefined) && (_extent.width > 0u) && (_extent.height > 0u);
  }

  rhi.initialize_headless();
  _output_format = _config.color_format;
  _extent = {_config.width, _config.height};
  return recreate_offscreen_texture(rhi, _config.width, _config.height);
}

void RuntimeOutput::shutdown(RHIContext& rhi) {
  destroy_offscreen_texture(rhi);
  if (_config.mode == RuntimeMode::Desktop) {
    rhi.destroy_swapchain();
  }
  _config = {};
  _output_format = RHITextureFormat::Undefined;
  _extent = {};
}

void RuntimeOutput::begin_frame(RHIContext& rhi) {
  rhi.begin_frame();
}

RuntimeOutputTarget RuntimeOutput::acquire_target(RHIContext& rhi) {
  RuntimeOutputTarget result = {};

  if (_config.mode == RuntimeMode::Desktop) {
    result.texture = rhi.get_current_swapchain_texture();
    const RHIExtent2D runtime_extent = rhi.get_swapchain_extent();
    result.width = runtime_extent.width;
    result.height = runtime_extent.height;
    _output_format = rhi.get_swapchain_format();
    _extent = runtime_extent;
  } else {
    result.texture = _offscreen_texture;
    result.width = _extent.width;
    result.height = _extent.height;
  }

  return result;
}

void RuntimeOutput::submit_frame(RHIContext& rhi, RHICommandBuffer cmd) {
  if (cmd.valid() == false) {
    rhi.end_frame();
    return;
  }

  rhi.submit_frame_command_buffer(cmd);
  if (_config.mode == RuntimeMode::Desktop) {
    rhi.present();
  } else {
    rhi.end_frame();
  }
}

void RuntimeOutput::resize(RHIContext& rhi, uint32_t width, uint32_t height) {
  _config.width = width;
  _config.height = height;

  if ((_config.mode == RuntimeMode::Desktop) && (_config.native_window != nullptr)) {
    rhi.resize_swapchain(width, height);
    _output_format = rhi.get_swapchain_format();
    _extent = rhi.get_swapchain_extent();
    return;
  }

  _extent = {width, height};
  recreate_offscreen_texture(rhi, width, height);
}

void RuntimeOutput::destroy_offscreen_texture(RHIContext& rhi) {
  if (_offscreen_texture.valid()) {
    rhi.device().destroy_texture(_offscreen_texture);
    _offscreen_texture = {};
  }
}

bool RuntimeOutput::recreate_offscreen_texture(RHIContext& rhi, uint32_t width, uint32_t height) {
  destroy_offscreen_texture(rhi);

  if ((width == 0u) || (height == 0u) || (_output_format == RHITextureFormat::Undefined)) {
    return false;
  }

  RHITextureDesc desc = {};
  desc.width = width;
  desc.height = height;
  desc.format = _output_format;
  desc.usage = RHITextureUsage::ColorAttachment | RHITextureUsage::TransferSrc | RHITextureUsage::Sampled;

  const RHICreateBindlessResult create_result = rhi.device().create_texture(desc);
  if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
    return false;
  }

  _offscreen_texture = create_result.handle;
  return true;
}

}  // namespace etx
