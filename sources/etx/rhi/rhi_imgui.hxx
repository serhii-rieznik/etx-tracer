#pragma once

#include <etx/rhi/rhi_types.hxx>

#include <imgui.h>
#include <vector>
#include <memory>

struct ImDrawData;
struct ImFontAtlas;

struct sapp_event;

namespace etx {

struct RHIContext;
class RHICommandBuffer;

struct RHIImGuiDesc {
  uint32_t max_vertices = 65536;
  RHITextureFormat color_format = RHITextureFormat::B8G8R8A8_SRGB;
  RHITextureFormat depth_format = RHITextureFormat::Undefined;
  uint32_t sample_count = 1;
  std::string ini_filename = {};
  bool no_default_font = false;
  bool write_alpha_channel = false;
};

struct RHIImGuiFrameDesc {
  uint32_t width = 0;
  uint32_t height = 0;
  double delta_time = 0.0;
  float dpi_scale = 1.0f;
};

struct RHIImGui {
  RHIImGui();
  ~RHIImGui();

  RHIResult setup(RHIContext* context, const etx::RHIImGuiDesc& desc);
  void shutdown();

  void new_frame(const RHIImGuiFrameDesc& desc);
  void render(RHICommandBuffer* command_buffer);

  bool handle_event(const sapp_event* event);

  RHIBindlessHandle get_font_texture() const {
    return _font_texture;
  }
  RHIPipeline get_pipeline() const {
    return _pipeline;
  }

 private:
  float _cur_dpi_scale = 1.0f;

  static ImGuiKey map_keycode(uint32_t key_code);
  static void update_modifiers(uint32_t modifiers);

  struct VertexBuffer {
    std::vector<uint8_t> data;
    RHIBindlessHandle buffer = {};
    bool dirty = false;
  };

  struct IndexBuffer {
    std::vector<uint8_t> data;
    RHIBindlessHandle buffer = {};
    bool dirty = false;
  };

  RHIContext* _context = nullptr;
  RHIImGuiDesc _desc = {};

  VertexBuffer _vertices[kRHIMaxFrames] = {};
  IndexBuffer _indices[kRHIMaxFrames] = {};
  RHIBindlessHandle _font_texture = {};
  RHIPipeline _pipeline = {};

  bool _initialized = false;

  RHIResult create_resources();
  void destroy_resources();
  RHIResult update_buffers(const ImDrawData* draw_data);
  void render_draw_data(RHICommandBuffer* command_buffer, const ImDrawData* draw_data);

  RHIResult create_font_texture();
  RHIResult create_pipeline();
};

}  // namespace etx