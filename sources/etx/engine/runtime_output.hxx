#pragma once

#include <etx/rhi/rhi.hxx>

namespace etx {

enum class RuntimeMode : uint32_t {
  Desktop,
  Headless,
  Streaming,
};

struct RuntimeOutputConfig {
  RuntimeMode mode = RuntimeMode::Desktop;
  uint32_t width = 0u;
  uint32_t height = 0u;
  const void* native_window = nullptr;
  RHITextureFormat color_format = RHITextureFormat::B8G8R8A8_UNORM;
};

struct RuntimeOutputTarget {
  RHITexture texture = {};
  uint32_t width = 0u;
  uint32_t height = 0u;

  bool valid() const {
    return texture.valid();
  }
};

struct RuntimeOutput {
  bool init(RHIContext& rhi, const RuntimeOutputConfig& config);
  void shutdown(RHIContext& rhi);

  void begin_frame(RHIContext& rhi);
  RuntimeOutputTarget acquire_target(RHIContext& rhi);
  void submit_frame(RHIContext& rhi, RHICommandBuffer cmd);

  void resize(RHIContext& rhi, uint32_t width, uint32_t height);

  RuntimeMode mode() const {
    return _config.mode;
  }

  bool imgui_supported() const {
    return (_config.mode == RuntimeMode::Desktop);
  }

  RHITextureFormat output_format() const {
    return _output_format;
  }

  RHIExtent2D extent() const {
    return _extent;
  }

  RHITexture offscreen_texture() const {
    return _offscreen_texture;
  }

 private:
  void destroy_offscreen_texture(RHIContext& rhi);
  bool recreate_offscreen_texture(RHIContext& rhi, uint32_t width, uint32_t height);

 private:
  RuntimeOutputConfig _config = {};
  RHITexture _offscreen_texture = {};
  RHITextureFormat _output_format = RHITextureFormat::Undefined;
  RHIExtent2D _extent = {};
};

}  // namespace etx
