#pragma once

#if !defined(WIN32_LEAN_AND_MEAN)
# define WIN32_LEAN_AND_MEAN 1
#endif

#include <etx/render/interop/interop.hxx>

#include <etx/core/core.hxx>
#include <etx/core/handle.hxx>
#include <etx/core/environment.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/math.hxx>

#include <etx/rhi/rhi.hxx>
#include <etx/rhi/rhi_imgui.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/engine/camera_controller.hxx>
#include "envmap.hxx"
#include "ocean.hxx"

#include <sokol_app.h>

#include <cstdint>

namespace etx {

struct PlaygroundApp {
  PlaygroundApp()
    : _camera_controller(_camera) {
  }
  ~PlaygroundApp() = default;

  void init();
  void frame();
  void cleanup();
  void process_event(const sapp_event* e);

 private:
  void recreate_scene_targets(uint32_t width, uint32_t height);
  void sync_scene_targets_to_swapchain_extent();
  bool recreate_tonemap_pipeline();
  void sync_swapchain_dependent_resources();

  RHIContext _rhi;
  RHIImGui _imgui;
  RHIPipeline _pipeline;
  RHIPipeline _compute_pipeline;
  RHIPipeline _tonemap_pipeline;
  RHIBindlessHandle _vertex_buffer;
  RHIBindlessHandle _index_buffer;
  RHIBindlessHandle _test_storage_texture;
  RHITexture _scene_opaque_color_buffer;
  RHITexture _scene_color_buffer;
  RHITexture _depth_buffer;

  EnvMap _envmap;
  Ocean _ocean;

  uint32_t _width = 0;
  uint32_t _height = 0;
  uint32_t _render_width = 0;
  uint32_t _render_height = 0;
  float _time = 0.0f;
  float _tonemap_exposure = 1.0f;
  bool _scene_opaque_color_initialized = false;
  bool _scene_color_initialized = false;
  RHITextureFormat _swapchain_color_format = RHITextureFormat::Undefined;

  Camera _camera = {};
  CameraController _camera_controller;
};

}  // namespace etx
