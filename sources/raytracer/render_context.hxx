#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/base.hxx>
#include <etx/rhi/rhi_types.hxx>
#include <etx/rhi/rhi.hxx>

#include "options.hxx"

#include <functional>

namespace etx {

struct RHIImGui;
struct TaskScheduler;
struct SceneRepresentation;
struct Renderer;

struct RenderContext {
  struct FrameData {
    float dt = 0.0f;
    uint32_t sample_count = 0;
    ViewParameters view_parameters = {};
  };

  RenderContext(TaskScheduler& s);
  ~RenderContext();

  void init();
  void cleanup();

  void start_frame(Renderer* renderer, SceneRepresentation& scene, const FrameData&);
  void end_frame();

  RHICommandBuffer* current_command_buffer();
  RHIImGui& rhi_ui();
  RHIContext* get_context();
  RHIDevice* get_device();
  RHITextureFormat get_swapchain_format();
  RHITextureFormat get_depth_format();

 public:
  ETX_DECLARE_PIMPL(RenderContext, 1024);
};

}  // namespace etx
