#pragma once

#include <etx/render/interop/interop.hxx>

#include <etx/core/pimpl.hxx>
#include <etx/engine/runtime_output.hxx>
#include <etx/rhi/rhi_types.hxx>
#include <etx/rhi/rhi.hxx>
#include "options.hxx"
#include <functional>
#include <vector>

namespace etx {

struct RHIImGui;
enum class RHIImGuiTheme;
struct TaskScheduler;
struct SceneRepresentation;
struct Renderer;

struct RenderContextConfig {
  RuntimeMode mode = RuntimeMode::Desktop;
  uint32_t width = 1600u;
  uint32_t height = 900u;
  const void* native_window = nullptr;
  bool enable_imgui = true;
  float dpi_scale = 1.0f;
};

struct RenderContext {
  struct PresentationViewport {
    int32_t x = 0;
    int32_t y = 0;
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t display_width = 0u;
    uint32_t display_height = 0u;
    bool valid = false;
  };

  struct FrameData {
    float dt = 0.0f;
    uint32_t sample_count = 0;
    ViewParameters view_parameters = {};
  };

  RenderContext(TaskScheduler& s);
  ~RenderContext();

  bool valid() const;
  void init(const RenderContextConfig& config = {});
  void cleanup();
  void cleanup(bool device_already_idle);

  void start_frame(Renderer* renderer, SceneRepresentation& scene, const FrameData&);
  void end_frame();
  void set_presentation_viewport(const PresentationViewport& viewport);

  RHIImGui& rhi_ui();
  RHIContext& get_context();
  RHIDevice& get_device();
  RHITextureFormat get_swapchain_format();
  RHITextureFormat get_depth_format();
  RuntimeMode runtime_mode() const;
  bool imgui_enabled() const;
  bool capture_output_png(std::vector<uint8_t>& png_data, uint32_t& width, uint32_t& height);
  void set_ui_theme(RHIImGuiTheme theme);

  void set_reference_image(const char*);
  void set_reference_image(const float4 data[], const uint2 dimensions);

 public:
  ETX_DECLARE_PIMPL(RenderContext, 2048);
};

}  // namespace etx
