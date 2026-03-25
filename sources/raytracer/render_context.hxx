#pragma once

#include <etx/render/interop/interop.hxx>

#include <etx/core/pimpl.hxx>
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

  bool valid() const;
  void init();
  void cleanup();

  void start_frame(Renderer* renderer, SceneRepresentation& scene, const FrameData&);
  void end_frame();

  RHIImGui& rhi_ui();
  RHIContext& get_context();
  RHIDevice& get_device();
  RHITextureFormat get_swapchain_format();
  RHITextureFormat get_depth_format();

  void set_reference_image(const char*);
  void set_reference_image(const float4 data[], const uint2 dimensions);

 public:
  ETX_DECLARE_PIMPL(RenderContext, 2048);
};

}  // namespace etx
