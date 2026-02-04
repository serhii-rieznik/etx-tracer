#pragma once

#include <etx/core/core.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/rhi_imgui.hxx>

#include <sokol_app_new.h>

namespace etx {

struct RHITestApplication {
  RHITestApplication();
  ~RHITestApplication();

  void init();
  void frame();
  void cleanup();
  void process_event(const sapp_event*);
  void memtest(int32_t line, const char*);

 private:
  bool create_rendering_resources();
  bool create_noise_texture();
  bool compile_shader_to_spirv(const std::string& file_path, const std::string& entry_point, RHIShaderStage stage, std::vector<uint8_t>& out_spirv);

 private:
  RHIContext* rhi_context = nullptr;
  RHICommandBuffer command_buffer = {};
  RHIImGui imgui = {};

  RHITexture noise_texture = {};
  RHIBuffer vertex_buffer = {};
  RHIBuffer index_buffer = {};
  uint32_t vertex_buffer_index = 0;
  uint32_t index_buffer_index = 0;
  RHIShader vertex_shader = {};
  RHIShader fragment_shader = {};
  RHIShader compute_shader = {};
  RHIPipeline graphics_pipeline = {};
  RHIPipeline compute_pipeline = {};

  bool initialized = false;
  float time = 0.0f;
  uint32_t frame_counter = 0;

  struct {
    float noise_speed = 1.0f;
    float noise_scale = 1.0f;
    float color[3] = {1.0f, 1.0f, 1.0f};
    bool show_demo_window = false;
  } ui_params;
};

}  // namespace etx