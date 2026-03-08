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
#include <etx/render/shared/scattering.hxx>

#include <etx/rhi/rhi.hxx>
#include <etx/rhi/rhi_imgui.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/engine/camera_controller.hxx>
#include "envmap.hxx"
#include "ocean.hxx"

#include <sokol_app.h>

#include <cstdint>
#include <string>

namespace etx {

struct PlaygroundApp {
  static constexpr uint32_t k_gpu_timing_query_count = 12u;
  static constexpr uint32_t k_gpu_timing_segment_count = k_gpu_timing_query_count - 1u;
  static constexpr uint32_t k_gpu_timing_pending_frame_count = 8u;

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
  bool ensure_sky_scattering_context();
  bool create_sun_sky_textures();
  bool record_regenerate_sun_sky_textures(RHICommandBuffer cmd);
  bool recreate_sun_sprite_pipeline();
  bool recreate_base_pipeline();
  bool apply_scene_msaa_sample_count(uint32_t sample_count);
  bool apply_ocean_patch_resolution(uint32_t patch_resolution);
  bool apply_ocean_fft_resolution(uint32_t fft_resolution);
  void draw_sun_sprite(RHICommandBuffer cmd, const float4x4& view_proj, uint32_t render_width, uint32_t render_height, float vertical_fov_radians);
  void poll_gpu_timing_results();
  void enqueue_gpu_timing_request(RHICommandBuffer cmd);
  void draw_gpu_timing_window();
  void destroy_ocean_obj_export_buffers();
  bool record_ocean_obj_export_capture(RHICommandBuffer cmd);
  void finalize_ocean_obj_export_capture();

  RHIContext _rhi;
  RHIImGui _imgui;
  RHIPipeline _pipeline;
  RHIPipeline _compute_pipeline;
  RHIPipeline _tonemap_pipeline;
  RHIPipeline _sun_sprite_pipeline;
  RHIBindlessHandle _vertex_buffer;
  RHIBindlessHandle _index_buffer;
  RHIBindlessHandle _test_storage_texture;
  RHITexture _scene_opaque_color_buffer;
  RHITexture _scene_color_buffer;
  RHITexture _depth_buffer;
  RHITexture _scene_opaque_color_msaa_buffer;
  RHITexture _scene_color_msaa_buffer;
  RHITexture _depth_msaa_buffer;
  RHITexture _ocean_wave_thickness_min_buffer;
  RHITexture _ocean_wave_thickness_max_buffer;
  RHIResourceState _ocean_wave_thickness_min_state = RHIResourceState::Undefined;
  RHIResourceState _ocean_wave_thickness_max_state = RHIResourceState::Undefined;
  RHITexture _ocean_foam_history_buffer[2] = {};
  RHIResourceState _ocean_foam_history_state[2] = {RHIResourceState::Undefined, RHIResourceState::Undefined};
  uint32_t _ocean_foam_history_write_index = 0u;
  RHITexture _generated_sky_envmap_texture;
  RHIResourceState _generated_sky_envmap_texture_state = RHIResourceState::Undefined;
  RHITexture _generated_sun_texture;
  RHIResourceState _generated_sun_texture_state = RHIResourceState::Undefined;
  scattering::GpuContext _sky_scattering = {};

  EnvMap _envmap;
  Ocean _ocean;

  uint32_t _width = 0;
  uint32_t _height = 0;
  uint32_t _render_width = 0;
  uint32_t _render_height = 0;
  float _time = 0.0f;
  float _tonemap_exposure = 1.0f;
  float _fps_counter_accumulated_time = 0.0f;
  float _fps_display = 0.0f;
  float _frame_time_ms_display = 0.0f;
  uint32_t _fps_counter_accumulated_frames = 0u;
  uint32_t _scene_msaa_sample_count = 4u;
  uint32_t _pending_scene_msaa_sample_count = 4u;
  bool _scene_msaa_recreate_requested = false;
  uint32_t _pending_ocean_patch_resolution = 128u;
  bool _ocean_patch_resolution_recreate_requested = false;
  uint32_t _pending_ocean_fft_resolution = 256u;
  bool _ocean_fft_resolution_recreate_requested = false;
  bool _simulation_paused = false;
  int32_t _ocean_obj_export_size_m = 100;
  struct OceanObjExportState {
    bool request_next_frame = false;
    bool capture_recorded = false;
    bool last_result_valid = false;
    bool last_result_success = false;
    uint32_t serial = 0u;
    uint32_t fft_resolution = 0u;
    uint32_t grid_resolution = 1001u;
    float area_size_m = 100.0f;
    float3 center = {0.0f, 0.0f, 0.0f};
    float cascade_lengths[Ocean::k_cascade_count] = {0.0f, 0.0f, 0.0f};
    float cascade_weights[Ocean::k_cascade_count] = {0.0f, 0.0f, 0.0f};
    RHIBindlessHandle displacement_readback_buffers[Ocean::k_cascade_count] = {};
    std::string pending_output_path = {};
    std::string last_output_path = {};
    std::string last_error = {};
  } _ocean_obj_export = {};
  struct GpuTimingPendingFrame {
    bool valid = false;
    RHICommandBuffer command_buffer = {};
    uint64_t frame_index = 0u;
  };
  GpuTimingPendingFrame _gpu_timing_pending_frames[k_gpu_timing_pending_frame_count] = {};
  uint32_t _gpu_timing_pending_read_index = 0u;
  uint32_t _gpu_timing_pending_write_index = 0u;
  uint32_t _gpu_timing_pending_count = 0u;
  uint64_t _gpu_timing_last_ticks[k_gpu_timing_query_count] = {};
  float _gpu_timing_last_segment_ms[k_gpu_timing_segment_count] = {};
  float _gpu_timing_last_total_ms = 0.0f;
  uint64_t _gpu_timing_frame_counter = 0u;
  uint64_t _gpu_timing_last_ready_frame_index = 0u;
  double _gpu_timestamp_period_ns = 0.0;
  bool _gpu_timing_supported = false;
  bool _gpu_timing_have_results = false;
  bool _gpu_timing_last_poll_not_ready = false;
  bool _scene_opaque_color_initialized = false;
  bool _scene_color_initialized = false;
  bool _scene_opaque_color_msaa_initialized = false;
  bool _scene_color_msaa_initialized = false;
  bool _sun_sky_dirty = false;
  RHITextureFormat _swapchain_color_format = RHITextureFormat::Undefined;

  Camera _camera = {};
  CameraController _camera_controller;
};

}  // namespace etx
