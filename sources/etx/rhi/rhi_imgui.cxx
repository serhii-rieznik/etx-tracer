#include <etx/rhi/rhi_imgui.hxx>

#include <imgui.h>
#include <sokol_app_new.h>

#include <etx/core/log.hxx>
#include <etx/core/environment.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

namespace etx {

RHIImGui::RHIImGui() = default;

RHIImGui::~RHIImGui() {
  ETX_CRITICAL(_initialized == false);
}

RHIResult RHIImGui::setup(RHIContext* context, const RHIImGuiDesc& desc) {
  if (_initialized) {
    shutdown();
  }

  _context = context;
  _desc = desc;

  // Initialize ImGui context if not already done
  if (ImGui::GetCurrentContext() == nullptr) {
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = _desc.ini_filename.c_str();
    io.BackendRendererName = "etx-rhi-imgui";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
  }

  // Update DPI scale before creating resources (needed for style)
  _cur_dpi_scale = (sapp_dpi_scale() > 0.0f) ? sapp_dpi_scale() : 1.0f;

  _desc.no_default_font = true;

  auto result = create_resources();
  if (result != RHIResult::Success) {
    log::error("Failed to create RHI ImGui resources: {}", static_cast<uint32_t>(result));
    return result;
  }

  _initialized = true;
  return RHIResult::Success;
}

void RHIImGui::shutdown() {
  if (_initialized == false) {
    return;
  }
  ImGui::SaveIniSettingsToDisk(_desc.ini_filename.c_str());
  destroy_resources();
  _initialized = false;
  _context = nullptr;
}

void RHIImGui::new_frame(const RHIImGuiFrameDesc& desc) {
  if (_initialized == false) {
    return;
  }

  bool dpi_changed = (desc.dpi_scale > 0.0f) && (desc.dpi_scale != _cur_dpi_scale);
  _cur_dpi_scale = (desc.dpi_scale > 0.0f) ? desc.dpi_scale : 1.0f;

  if (dpi_changed) {
    create_font_texture();
  }

  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(static_cast<float>(desc.width) / _cur_dpi_scale, static_cast<float>(desc.height) / _cur_dpi_scale);
  io.DeltaTime = static_cast<float>(desc.delta_time);
  io.DisplayFramebufferScale = ImVec2(_cur_dpi_scale, _cur_dpi_scale);

  ImGui::NewFrame();
}

void RHIImGui::render(RHICommandBuffer* command_buffer) {
  if (_initialized == false || command_buffer == nullptr) {
    return;
  }

  ImGui::Render();
  ImDrawData* draw_data = ImGui::GetDrawData();
  if (draw_data == nullptr || draw_data->TotalVtxCount == 0) {
    return;
  }

  // Update buffers with draw data
  if (update_buffers(draw_data) != RHIResult::Success) {
    return;
  }

  // Render the draw data
  render_draw_data(command_buffer, draw_data);
}

RHIResult RHIImGui::create_resources() {
  RHIBufferDesc vb_desc = {
    .size = _desc.max_vertices * sizeof(ImDrawVert),
    .usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst,
    .host_visible = true,
  };
  RHIBufferDesc ib_desc = {
    .size = _desc.max_vertices * 3 * sizeof(ImDrawIdx),
    .usage = RHIBufferUsage::Index | RHIBufferUsage::TransferDst,
    .host_visible = true,
  };

  for (uint32_t i = 0; i < kRHIMaxFrames; ++i) {
    auto vb_result = _context->get_device()->create_buffer(vb_desc);
    if (vb_result.result != RHIResult::Success) {
      return vb_result.result;
    }
    auto ib_result = _context->get_device()->create_buffer(ib_desc);
    if (ib_result.result != RHIResult::Success) {
      return ib_result.result;
    }
    _vertices[i].buffer = vb_result.handle;
    _vertices[i].data.resize(vb_desc.size);
    _indices[i].buffer = ib_result.handle;
    _indices[i].data.resize(ib_desc.size);
  }

  auto font_result = create_font_texture();
  if (font_result != RHIResult::Success) {
    return font_result;
  }

  auto pipeline_result = create_pipeline();
  if (pipeline_result != RHIResult::Success) {
    return pipeline_result;
  }

  return RHIResult::Success;
}

void RHIImGui::destroy_resources() {
  auto device = _context ? _context->get_device() : nullptr;
  if (device == nullptr) {
    return;
  }

  for (uint32_t i = 0; i < kRHIMaxFrames; ++i) {
    device->destroy_buffer(_vertices[i].buffer);
    _vertices[i].buffer = {};
    device->destroy_buffer(_indices[i].buffer);
    _indices[i].buffer = {};
  }

  device->destroy_texture(_font_texture);
  _font_texture = {};

  device->destroy_pipeline(_pipeline);
  _pipeline = {};
}

RHIResult RHIImGui::create_font_texture() {
  auto device = _context ? _context->get_device() : nullptr;
  if (device != nullptr) {
    if (_font_texture != 0) {
      device->destroy_texture(_font_texture);
      _font_texture = {};
    }
  }

  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->Clear();

  ImFontConfig font_config = {};
  font_config.OversampleH = 4;
  font_config.OversampleV = 4;

  char font_file[1024] = {};
  env().file_in_data("fonts/ubuntu.ttf", font_file, sizeof(font_file));
  float font_size = 14.0f;

  auto font = io.Fonts->AddFontFromFileTTF(font_file, font_size * _cur_dpi_scale, &font_config, nullptr);
  if (font == nullptr) {
    font = io.Fonts->AddFontDefault(&font_config);
  }

  if (font != nullptr) {
    font->Scale = 1.0f / _cur_dpi_scale;
  }

  // Get font texture data
  unsigned char* pixels = nullptr;
  int width = 0, height = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

  if (pixels == nullptr || width == 0 || height == 0) {
    return RHIResult::InvalidArgument;
  }

  // Create texture
  RHITextureDesc tex_desc = {
    .width = static_cast<uint32_t>(width),
    .height = static_cast<uint32_t>(height),
    .format = RHITextureFormat::R8G8B8A8_UNORM,
    .usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst,
    .host_visible = false,
  };

  auto tex_result = _context->get_device()->create_texture(tex_desc);
  if (tex_result.result != RHIResult::Success) {
    return tex_result.result;
  }
  _font_texture = tex_result.handle;

  // Upload texture data
  auto update_result = _context->get_device()->update_texture(_font_texture, pixels, 0, 0);
  if (update_result != RHIResult::Success) {
    _context->get_device()->destroy_texture(_font_texture);
    _font_texture = {};
    return update_result;
  }

  // Create sampler
  RHISamplerDesc sampler_desc = {};
  sampler_desc.min_filter = RHISamplerFilter::Linear;
  sampler_desc.mag_filter = RHISamplerFilter::Linear;
  sampler_desc.mipmap_mode = RHISamplerMipmapMode::Linear;
  sampler_desc.address_mode_u = RHISamplerAddressMode::ClampToEdge;
  sampler_desc.address_mode_v = RHISamplerAddressMode::ClampToEdge;
  sampler_desc.address_mode_w = RHISamplerAddressMode::ClampToEdge;

  auto sampler_result = _context->get_device()->create_sampler(sampler_desc);
  if (sampler_result.result != RHIResult::Success) {
    _context->get_device()->destroy_texture(_font_texture);
    _font_texture = {};
    return sampler_result.result;
  }

  io.Fonts->TexID = static_cast<ImTextureID>(_font_texture);
  return RHIResult::Success;
}

RHIResult RHIImGui::create_pipeline() {
  auto device = _context->get_device();
  auto compiler = ShaderCompiler::get_global_instance();
  if (compiler == nullptr) {
    log::error("Shader compiler not available");
    return RHIResult::InvalidArgument;
  }

  std::string error_message;
  std::string hlsl_source = compiler->read_file_content("./shaders/imgui.hlsl", error_message);
  if (hlsl_source.empty()) {
    log::error("Failed to read imgui.hlsl: {}", error_message);
    return RHIResult::InvalidArgument;
  }

  auto vs_result = compiler->compile_hlsl_to_spirv(hlsl_source, "vs_main", RHIShaderStage::Vertex, "imgui.hlsl");
  if (vs_result.result != RHIResult::Success) {
    log::error("Failed to compile imgui vertex shader: {}", vs_result.error_message);
    return vs_result.result;
  }

  auto ps_result = compiler->compile_hlsl_to_spirv(hlsl_source, "ps_main", RHIShaderStage::Fragment, "imgui.hlsl");
  if (ps_result.result != RHIResult::Success) {
    log::error("Failed to compile imgui fragment shader: {}", ps_result.error_message);
    return ps_result.result;
  }

  RHIGraphicsPipelineDesc pipeline_desc = {
    .vertex_shader =
      {
        .spirv_data = vs_result.spirv_data.data(),
        .spirv_size = vs_result.spirv_data.size(),
        .stage = RHIShaderStage::Vertex,
        .entry_point = "vs_main",
      },
    .fragment_shader =
      {
        .spirv_data = ps_result.spirv_data.data(),
        .spirv_size = ps_result.spirv_data.size(),
        .stage = RHIShaderStage::Fragment,
        .entry_point = "ps_main",
      },
    .blend =
      {
        .src_color_blend_factor = RHIBlendFactor::SrcAlpha,
        .dst_color_blend_factor = RHIBlendFactor::OneMinusSrcAlpha,
        .color_blend_op = RHIBlendOp::Add,
        .src_alpha_blend_factor = RHIBlendFactor::OneMinusSrcAlpha,
        .dst_alpha_blend_factor = RHIBlendFactor::Zero,
        .alpha_blend_op = RHIBlendOp::Add,
        .blend_enable = true,
      },
    .color_attachment_count = 1,
    .color_formats = {_desc.color_format},
    .depth_format = _desc.depth_format,
  };

  auto pipeline_result = device->create_graphics_pipeline(pipeline_desc);
  if (pipeline_result.result != RHIResult::Success) {
    log::error("Failed to create imgui graphics pipeline");
    return pipeline_result.result;
  }

  _pipeline = pipeline_result.handle;
  return RHIResult::Success;
}

RHIResult RHIImGui::update_buffers(const ImDrawData* draw_data) {
  uint32_t total_vertices = draw_data->TotalVtxCount;
  uint32_t total_indices = draw_data->TotalIdxCount;

  if (total_vertices == 0 || total_indices == 0) {
    return RHIResult::Success;
  }

  const uint32_t frame_index = _context->get_current_frame_index();
  auto& vertices = _vertices[frame_index];
  auto& indices = _indices[frame_index];

  size_t required_vb_size = total_vertices * sizeof(ImDrawVert);
  size_t required_ib_size = total_indices * sizeof(ImDrawIdx);

  auto device = _context->get_device();

  if (required_vb_size > vertices.data.size()) {
    if (vertices.buffer) {
      device->destroy_buffer(vertices.buffer);
    }

    size_t new_size = required_vb_size + (required_vb_size / 2);
    RHIBufferDesc desc = {new_size, RHIBufferUsage::Storage | RHIBufferUsage::TransferDst, true};
    auto res = device->create_buffer(desc);
    if (res.result != RHIResult::Success)
      return res.result;

    vertices.buffer = res.handle;
    vertices.data.resize(new_size);
  }

  if (required_ib_size > indices.data.size()) {
    if (indices.buffer) {
      device->destroy_buffer(indices.buffer);
    }

    size_t new_size = required_ib_size + (required_ib_size / 2);
    RHIBufferDesc desc = {new_size, RHIBufferUsage::Index | RHIBufferUsage::TransferDst, true};
    auto res = device->create_buffer(desc);
    if (res.result != RHIResult::Success)
      return res.result;
    indices.buffer = res.handle;
    indices.data.resize(new_size);
  }

  size_t vb_offset = 0;
  size_t ib_offset = 0;
  for (int i = 0; i < draw_data->CmdListsCount; ++i) {
    const ImDrawList* cmd_list = draw_data->CmdLists[i];
    memcpy(vertices.data.data() + vb_offset, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
    vb_offset += cmd_list->VtxBuffer.Size * sizeof(ImDrawVert);
    memcpy(indices.data.data() + ib_offset, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
    ib_offset += cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx);
  }

  device->update_buffer(vertices.buffer, vertices.data.data(), required_vb_size);
  device->update_buffer(indices.buffer, indices.data.data(), required_ib_size);

  return RHIResult::Success;
}

struct ImGuiPushConstants {
  float scale[2];
  float translate[2];
  uint32_t vertex_buffer_index;
  uint32_t texture_index;
  uint32_t sampler_index;
  uint32_t padding;
};

void RHIImGui::render_draw_data(RHICommandBuffer* command_buffer, const ImDrawData* draw_data) {
  command_buffer->set_pipeline(_pipeline);

  const uint32_t frame_index = _context->get_current_frame_index();
  auto& vertices = _vertices[frame_index];
  auto& indices = _indices[frame_index];

  float L = draw_data->DisplayPos.x;
  float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
  float T = draw_data->DisplayPos.y;
  float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;

  ImGuiPushConstants pc = {};
  pc.scale[0] = 2.0f / (R - L);
  pc.scale[1] = 2.0f / (B - T);
  pc.translate[0] = (R + L) / (L - R);
  pc.translate[1] = (T + B) / (T - B);
  pc.vertex_buffer_index = get_bindless_descriptor_index(vertices.buffer);
  pc.sampler_index = _context->get_sampler_index(RHISamplerType::LinearRepeat);

  float fb_width = draw_data->DisplaySize.x * draw_data->FramebufferScale.x;
  float fb_height = draw_data->DisplaySize.y * draw_data->FramebufferScale.y;

  RHIViewport viewport = {0.0f, 0.0f, fb_width, fb_height, 0.0f, 1.0f};
  command_buffer->set_viewport(viewport);

  uint32_t vertex_offset = 0;
  uint32_t index_offset = 0;
  for (int cmd_list_idx = 0; cmd_list_idx < draw_data->CmdListsCount; ++cmd_list_idx) {
    const ImDrawList* cmd_list = draw_data->CmdLists[cmd_list_idx];
    for (int cmd_idx = 0; cmd_idx < cmd_list->CmdBuffer.Size; ++cmd_idx) {
      const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_idx];
      if (pcmd->UserCallback != nullptr)
        continue;

      ImVec2 clip_min = {(pcmd->ClipRect.x - draw_data->DisplayPos.x) * draw_data->FramebufferScale.x,
        (pcmd->ClipRect.y - draw_data->DisplayPos.y) * draw_data->FramebufferScale.y};
      ImVec2 clip_max = {(pcmd->ClipRect.z - draw_data->DisplayPos.x) * draw_data->FramebufferScale.x,
        (pcmd->ClipRect.w - draw_data->DisplayPos.y) * draw_data->FramebufferScale.y};

      if (clip_min.x < 0.0f)
        clip_min.x = 0.0f;
      if (clip_min.y < 0.0f)
        clip_min.y = 0.0f;
      if (clip_max.x > fb_width)
        clip_max.x = fb_width;
      if (clip_max.y > fb_height)
        clip_max.y = fb_height;

      if ((clip_max.x <= clip_min.x) || (clip_max.y <= clip_min.y))
        continue;

      RHIRect scissor = {
        static_cast<int32_t>(clip_min.x),
        static_cast<int32_t>(clip_min.y),
        static_cast<uint32_t>(clip_max.x - clip_min.x),
        static_cast<uint32_t>(clip_max.y - clip_min.y),
      };
      RHIIndexedDrawDesc draw_desc = {
        .index_count = pcmd->ElemCount,
        .instance_count = 1,
        .first_index = pcmd->IdxOffset + index_offset,
        .vertex_offset = pcmd->VtxOffset + vertex_offset,
        .index_type = sizeof(ImDrawIdx) == 2 ? RHIIndexType::UInt16 : RHIIndexType::UInt32,
      };
      pc.texture_index = get_bindless_descriptor_index(static_cast<RHIBindlessHandle>(pcmd->GetTexID()));

      command_buffer->set_scissor(scissor);
      command_buffer->push_constants(&pc, sizeof(ImGuiPushConstants));
      command_buffer->draw_indexed(draw_desc, indices.buffer);
    }
    vertex_offset += cmd_list->VtxBuffer.Size;
    index_offset += cmd_list->IdxBuffer.Size;
  }
}

bool RHIImGui::handle_event(const sapp_event* event) {
  if (_initialized == false) {
    return false;
  }

  ImGuiIO& io = ImGui::GetIO();
  update_modifiers(event->modifiers);

  switch (event->type) {
    case SAPP_EVENTTYPE_MOUSE_DOWN:
      io.AddMousePosEvent(event->mouse_x / _cur_dpi_scale, event->mouse_y / _cur_dpi_scale);
      io.AddMouseButtonEvent(static_cast<int>(event->mouse_button), true);
      return io.WantCaptureMouse;
    case SAPP_EVENTTYPE_MOUSE_UP:
      io.AddMousePosEvent(event->mouse_x / _cur_dpi_scale, event->mouse_y / _cur_dpi_scale);
      io.AddMouseButtonEvent(static_cast<int>(event->mouse_button), false);
      return io.WantCaptureMouse;
    case SAPP_EVENTTYPE_MOUSE_MOVE:
      io.AddMousePosEvent(event->mouse_x / _cur_dpi_scale, event->mouse_y / _cur_dpi_scale);
      return io.WantCaptureMouse;
    case SAPP_EVENTTYPE_MOUSE_SCROLL:
      io.AddMouseWheelEvent(event->scroll_x, event->scroll_y);
      return io.WantCaptureMouse;
    case SAPP_EVENTTYPE_KEY_DOWN:
      io.AddKeyEvent(map_keycode(event->key_code), true);
      return io.WantCaptureKeyboard;
    case SAPP_EVENTTYPE_KEY_UP:
      io.AddKeyEvent(map_keycode(event->key_code), false);
      return io.WantCaptureKeyboard;
    case SAPP_EVENTTYPE_CHAR:
      io.AddInputCharacter(event->char_code);
      return io.WantCaptureKeyboard;
    case SAPP_EVENTTYPE_FOCUSED:
      io.AddFocusEvent(true);
      break;
    case SAPP_EVENTTYPE_UNFOCUSED:
      io.AddFocusEvent(false);
      break;
    default:
      break;
  }

  return false;
}

ImGuiKey RHIImGui::map_keycode(uint32_t key_code) {
  switch (key_code) {
    case SAPP_KEYCODE_SPACE:
      return ImGuiKey_Space;
    case SAPP_KEYCODE_APOSTROPHE:
      return ImGuiKey_Apostrophe;
    case SAPP_KEYCODE_COMMA:
      return ImGuiKey_Comma;
    case SAPP_KEYCODE_MINUS:
      return ImGuiKey_Minus;
    case SAPP_KEYCODE_PERIOD:
      return ImGuiKey_Period;
    case SAPP_KEYCODE_SLASH:
      return ImGuiKey_Slash;
    case SAPP_KEYCODE_0:
      return ImGuiKey_0;
    case SAPP_KEYCODE_1:
      return ImGuiKey_1;
    case SAPP_KEYCODE_2:
      return ImGuiKey_2;
    case SAPP_KEYCODE_3:
      return ImGuiKey_3;
    case SAPP_KEYCODE_4:
      return ImGuiKey_4;
    case SAPP_KEYCODE_5:
      return ImGuiKey_5;
    case SAPP_KEYCODE_6:
      return ImGuiKey_6;
    case SAPP_KEYCODE_7:
      return ImGuiKey_7;
    case SAPP_KEYCODE_8:
      return ImGuiKey_8;
    case SAPP_KEYCODE_9:
      return ImGuiKey_9;
    case SAPP_KEYCODE_SEMICOLON:
      return ImGuiKey_Semicolon;
    case SAPP_KEYCODE_EQUAL:
      return ImGuiKey_Equal;
    case SAPP_KEYCODE_A:
      return ImGuiKey_A;
    case SAPP_KEYCODE_B:
      return ImGuiKey_B;
    case SAPP_KEYCODE_C:
      return ImGuiKey_C;
    case SAPP_KEYCODE_D:
      return ImGuiKey_D;
    case SAPP_KEYCODE_E:
      return ImGuiKey_E;
    case SAPP_KEYCODE_F:
      return ImGuiKey_F;
    case SAPP_KEYCODE_G:
      return ImGuiKey_G;
    case SAPP_KEYCODE_H:
      return ImGuiKey_H;
    case SAPP_KEYCODE_I:
      return ImGuiKey_I;
    case SAPP_KEYCODE_J:
      return ImGuiKey_J;
    case SAPP_KEYCODE_K:
      return ImGuiKey_K;
    case SAPP_KEYCODE_L:
      return ImGuiKey_L;
    case SAPP_KEYCODE_M:
      return ImGuiKey_M;
    case SAPP_KEYCODE_N:
      return ImGuiKey_N;
    case SAPP_KEYCODE_O:
      return ImGuiKey_O;
    case SAPP_KEYCODE_P:
      return ImGuiKey_P;
    case SAPP_KEYCODE_Q:
      return ImGuiKey_Q;
    case SAPP_KEYCODE_R:
      return ImGuiKey_R;
    case SAPP_KEYCODE_S:
      return ImGuiKey_S;
    case SAPP_KEYCODE_T:
      return ImGuiKey_T;
    case SAPP_KEYCODE_U:
      return ImGuiKey_U;
    case SAPP_KEYCODE_V:
      return ImGuiKey_V;
    case SAPP_KEYCODE_W:
      return ImGuiKey_W;
    case SAPP_KEYCODE_X:
      return ImGuiKey_X;
    case SAPP_KEYCODE_Y:
      return ImGuiKey_Y;
    case SAPP_KEYCODE_Z:
      return ImGuiKey_Z;
    case SAPP_KEYCODE_LEFT_BRACKET:
      return ImGuiKey_LeftBracket;
    case SAPP_KEYCODE_BACKSLASH:
      return ImGuiKey_Backslash;
    case SAPP_KEYCODE_RIGHT_BRACKET:
      return ImGuiKey_RightBracket;
    case SAPP_KEYCODE_GRAVE_ACCENT:
      return ImGuiKey_GraveAccent;
    case SAPP_KEYCODE_ESCAPE:
      return ImGuiKey_Escape;
    case SAPP_KEYCODE_ENTER:
      return ImGuiKey_Enter;
    case SAPP_KEYCODE_TAB:
      return ImGuiKey_Tab;
    case SAPP_KEYCODE_BACKSPACE:
      return ImGuiKey_Backspace;
    case SAPP_KEYCODE_INSERT:
      return ImGuiKey_Insert;
    case SAPP_KEYCODE_DELETE:
      return ImGuiKey_Delete;
    case SAPP_KEYCODE_RIGHT:
      return ImGuiKey_RightArrow;
    case SAPP_KEYCODE_LEFT:
      return ImGuiKey_LeftArrow;
    case SAPP_KEYCODE_DOWN:
      return ImGuiKey_DownArrow;
    case SAPP_KEYCODE_UP:
      return ImGuiKey_UpArrow;
    case SAPP_KEYCODE_PAGE_UP:
      return ImGuiKey_PageUp;
    case SAPP_KEYCODE_PAGE_DOWN:
      return ImGuiKey_PageDown;
    case SAPP_KEYCODE_HOME:
      return ImGuiKey_Home;
    case SAPP_KEYCODE_END:
      return ImGuiKey_End;
    case SAPP_KEYCODE_CAPS_LOCK:
      return ImGuiKey_CapsLock;
    case SAPP_KEYCODE_SCROLL_LOCK:
      return ImGuiKey_ScrollLock;
    case SAPP_KEYCODE_NUM_LOCK:
      return ImGuiKey_NumLock;
    case SAPP_KEYCODE_PRINT_SCREEN:
      return ImGuiKey_PrintScreen;
    case SAPP_KEYCODE_PAUSE:
      return ImGuiKey_Pause;
    case SAPP_KEYCODE_F1:
      return ImGuiKey_F1;
    case SAPP_KEYCODE_F2:
      return ImGuiKey_F2;
    case SAPP_KEYCODE_F3:
      return ImGuiKey_F3;
    case SAPP_KEYCODE_F4:
      return ImGuiKey_F4;
    case SAPP_KEYCODE_F5:
      return ImGuiKey_F5;
    case SAPP_KEYCODE_F6:
      return ImGuiKey_F6;
    case SAPP_KEYCODE_F7:
      return ImGuiKey_F7;
    case SAPP_KEYCODE_F8:
      return ImGuiKey_F8;
    case SAPP_KEYCODE_F9:
      return ImGuiKey_F9;
    case SAPP_KEYCODE_F10:
      return ImGuiKey_F10;
    case SAPP_KEYCODE_F11:
      return ImGuiKey_F11;
    case SAPP_KEYCODE_F12:
      return ImGuiKey_F12;
    case SAPP_KEYCODE_KP_0:
      return ImGuiKey_Keypad0;
    case SAPP_KEYCODE_KP_1:
      return ImGuiKey_Keypad1;
    case SAPP_KEYCODE_KP_2:
      return ImGuiKey_Keypad2;
    case SAPP_KEYCODE_KP_3:
      return ImGuiKey_Keypad3;
    case SAPP_KEYCODE_KP_4:
      return ImGuiKey_Keypad4;
    case SAPP_KEYCODE_KP_5:
      return ImGuiKey_Keypad5;
    case SAPP_KEYCODE_KP_6:
      return ImGuiKey_Keypad6;
    case SAPP_KEYCODE_KP_7:
      return ImGuiKey_Keypad7;
    case SAPP_KEYCODE_KP_8:
      return ImGuiKey_Keypad8;
    case SAPP_KEYCODE_KP_9:
      return ImGuiKey_Keypad9;
    case SAPP_KEYCODE_KP_DECIMAL:
      return ImGuiKey_KeypadDecimal;
    case SAPP_KEYCODE_KP_DIVIDE:
      return ImGuiKey_KeypadDivide;
    case SAPP_KEYCODE_KP_MULTIPLY:
      return ImGuiKey_KeypadMultiply;
    case SAPP_KEYCODE_KP_SUBTRACT:
      return ImGuiKey_KeypadSubtract;
    case SAPP_KEYCODE_KP_ADD:
      return ImGuiKey_KeypadAdd;
    case SAPP_KEYCODE_KP_ENTER:
      return ImGuiKey_KeypadEnter;
    case SAPP_KEYCODE_KP_EQUAL:
      return ImGuiKey_KeypadEqual;
    case SAPP_KEYCODE_LEFT_SHIFT:
      return ImGuiKey_LeftShift;
    case SAPP_KEYCODE_LEFT_CONTROL:
      return ImGuiKey_LeftCtrl;
    case SAPP_KEYCODE_LEFT_ALT:
      return ImGuiKey_LeftAlt;
    case SAPP_KEYCODE_LEFT_SUPER:
      return ImGuiKey_LeftSuper;
    case SAPP_KEYCODE_RIGHT_SHIFT:
      return ImGuiKey_RightShift;
    case SAPP_KEYCODE_RIGHT_CONTROL:
      return ImGuiKey_RightCtrl;
    case SAPP_KEYCODE_RIGHT_ALT:
      return ImGuiKey_RightAlt;
    case SAPP_KEYCODE_RIGHT_SUPER:
      return ImGuiKey_RightSuper;
    case SAPP_KEYCODE_MENU:
      return ImGuiKey_Menu;
    default:
      return ImGuiKey_None;
  }
}

void RHIImGui::update_modifiers(uint32_t modifiers) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddKeyEvent(ImGuiMod_Ctrl, (modifiers & SAPP_MODIFIER_CTRL) != 0);
  io.AddKeyEvent(ImGuiMod_Shift, (modifiers & SAPP_MODIFIER_SHIFT) != 0);
  io.AddKeyEvent(ImGuiMod_Alt, (modifiers & SAPP_MODIFIER_ALT) != 0);
  io.AddKeyEvent(ImGuiMod_Super, (modifiers & SAPP_MODIFIER_SUPER) != 0);
}

}  // namespace etx
