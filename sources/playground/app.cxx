#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>

#include "app.hxx"

#include <imgui.h>

#include <cmath>
#include <vector>
#include <string>
#include <cstring>

#include <etx/render/interop/geometry.hxx>

#if defined(ETX_PLATFORM_WINDOWS)
# define WIN32_LEAN_AND_MEAN 1
# include <Windows.h>
#endif

namespace etx {

static constexpr float k_cascade_length_ratio_min = 2.0f;
static constexpr float k_cascade_length_min[Ocean::k_cascade_count] = {20.0f, 10.0f, 5.0f};
static constexpr float k_cascade_length_max = 4000.0f;

static bool enforce_cascade_length_constraints_ui(OceanParameters& parameters) {
  bool changed = false;
  float length_0 = min(k_cascade_length_max, max(k_cascade_length_min[0], parameters.cascade_lengths[0]));
  float max_length_1 = min(k_cascade_length_max / k_cascade_length_ratio_min, length_0 / k_cascade_length_ratio_min);
  float length_1 = min(max_length_1, max(k_cascade_length_min[1], parameters.cascade_lengths[1]));
  float max_length_2 = min((k_cascade_length_max / k_cascade_length_ratio_min) / k_cascade_length_ratio_min, length_1 / k_cascade_length_ratio_min);
  float length_2 = min(max_length_2, max(k_cascade_length_min[2], parameters.cascade_lengths[2]));

  if (fabsf(parameters.cascade_lengths[0] - length_0) > 1.0e-6f) {
    parameters.cascade_lengths[0] = length_0;
    changed = true;
  }
  if (fabsf(parameters.cascade_lengths[1] - length_1) > 1.0e-6f) {
    parameters.cascade_lengths[1] = length_1;
    changed = true;
  }
  if (fabsf(parameters.cascade_lengths[2] - length_2) > 1.0e-6f) {
    parameters.cascade_lengths[2] = length_2;
    changed = true;
  }
  return changed;
}

static void draw_ocean_control_panel(Ocean& ocean) {
  OceanParameters& parameters = ocean.parameters();
  bool spectrum_parameters_changed = false;

  ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Once);
  ImGui::SetNextWindowSize(ImVec2(430.0f, 720.0f), ImGuiCond_Once);
  ImGui::Begin("Infinite Ocean 3D");

  ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
  if (ImGui::CollapsingHeader("Geometry Stats")) {
    ImGui::BulletText("Patch resolution: %u", ocean.patch_resolution());
    ImGui::BulletText("Clipmap levels: %u", ocean.clipmap_levels());
    ImGui::BulletText("Base patch size: %.2f", ocean.base_patch_size());
    ImGui::BulletText("Base spacing: %.3f m/vertex", ocean.base_vertex_spacing());
    ImGui::BulletText("Coverage radius: %.1f km", ocean.coverage_radius() * 0.001f);
    ImGui::BulletText("FFT resolution: %u", ocean.fft_resolution());
    ImGui::BulletText("Instance count: %u", ocean.instance_count());
  }

  ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
  if (ImGui::CollapsingHeader("Simulation")) {
    ImGui::SliderFloat("Time scale", &parameters.time_scale, 0.0f, 4.0f, "%.3f");
    ImGui::SliderFloat("LOD focus max dist (m)", &parameters.lod_forward_bias, 0.0f, 2000.0f, "%.1f");
    if (ImGui::SliderFloat("Wind dir X", &parameters.wind_direction_x, -1.0f, 1.0f, "%.3f")) {
      spectrum_parameters_changed = true;
    }
    if (ImGui::SliderFloat("Wind dir Y", &parameters.wind_direction_y, -1.0f, 1.0f, "%.3f")) {
      spectrum_parameters_changed = true;
    }
    float wind_length_sq = (parameters.wind_direction_x * parameters.wind_direction_x) + (parameters.wind_direction_y * parameters.wind_direction_y);
    if (wind_length_sq < 1e-6f) {
      parameters.wind_direction_x = 1.0f;
      parameters.wind_direction_y = 0.0f;
    }
    if (ImGui::Button("Normalize wind direction")) {
      float wind_length = sqrtf((parameters.wind_direction_x * parameters.wind_direction_x) + (parameters.wind_direction_y * parameters.wind_direction_y));
      if (wind_length > 1e-6f) {
        parameters.wind_direction_x /= wind_length;
        parameters.wind_direction_y /= wind_length;
        spectrum_parameters_changed = true;
      }
    }
    if (ImGui::SliderFloat("Wind speed", &parameters.wind_speed, 0.0f, 60.0f, "%.3f")) {
      spectrum_parameters_changed = true;
    }
    if (ImGui::SliderFloat("Water depth (m)", &parameters.water_depth, 1.0f, 4000.0f, "%.3f")) {
      spectrum_parameters_changed = true;
    }
    if (ImGui::SliderFloat("JONSWAP gamma", &parameters.jonswap_gamma, 1.0f, 8.0f, "%.3f")) {
      spectrum_parameters_changed = true;
    }
    if (ImGui::SliderFloat("Directional spread", &parameters.directional_spread, 1.0f, 64.0f, "%.3f")) {
      spectrum_parameters_changed = true;
    }
    bool significant_wave_height_enable = parameters.significant_wave_height_enable;
    if (ImGui::Checkbox("Global sea state (Hs)", &significant_wave_height_enable)) {
      parameters.significant_wave_height_enable = significant_wave_height_enable;
      spectrum_parameters_changed = true;
    }
    if (parameters.significant_wave_height_enable) {
      if (ImGui::SliderFloat("Significant wave height Hs (m)", &parameters.significant_wave_height, 0.0f, 20.0f, "%.3f")) {
        spectrum_parameters_changed = true;
      }
    }
    ImGui::SliderFloat("Choppiness", &parameters.choppiness, 0.0f, 3.0f, "%.3f");
    bool spectral_band_limit_enable = parameters.spectral_band_limit_enable;
    if (ImGui::Checkbox("Spectral band-limit", &spectral_band_limit_enable)) {
      parameters.spectral_band_limit_enable = spectral_band_limit_enable;
      spectrum_parameters_changed = true;
    }
    if (ImGui::Button("Regenerate spectrum")) {
      ocean.force_regenerate_spectrum();
    }
  }

  ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
  if (ImGui::CollapsingHeader("Cascade Spectrum")) {
    ImGui::Text("Length constraints: L0 >= %.1fxL1 >= %.1fxL2", k_cascade_length_ratio_min, k_cascade_length_ratio_min);
    if (parameters.significant_wave_height_enable) {
      ImGui::TextUnformatted("Per-cascade RMS is solved from spectrum-band energy.");
    } else {
      ImGui::TextUnformatted("Height slider is absolute RMS in meters.");
    }
    for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
      ImGui::PushID(static_cast<int>(i));
      ImGui::Text("Cascade %u", i);
      if (ImGui::SliderFloat("Length (m)", &parameters.cascade_lengths[i], k_cascade_length_min[2], k_cascade_length_max, "%.3f")) {
        spectrum_parameters_changed = true;
      }
      if (parameters.significant_wave_height_enable) {
        ImGui::Text("Solved RMS (m): %.3f", ocean.resolved_cascade_rms(i));
      } else {
        if (ImGui::SliderFloat("Height RMS (m)", &parameters.cascade_amplitudes[i], 0.0f, 20.0f, "%.3f")) {
          spectrum_parameters_changed = true;
        }
      }
      ImGui::Separator();
      ImGui::PopID();
    }
  }
  if (enforce_cascade_length_constraints_ui(parameters)) {
    spectrum_parameters_changed = true;
  }

  ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
  if (ImGui::CollapsingHeader("Debug Overrides")) {
    ImGui::TextUnformatted("Artistic/debug controls. Keep disabled for physical runs.");
    bool debug_cascade_overrides_enable = parameters.debug_cascade_overrides_enable;
    if (ImGui::Checkbox("Enable cascade debug overrides", &debug_cascade_overrides_enable)) {
      parameters.debug_cascade_overrides_enable = debug_cascade_overrides_enable;
    }
    if (parameters.debug_cascade_overrides_enable == false) {
      ImGui::TextUnformatted("Neutral physical mixing is active.");
    }
    ImGui::TextUnformatted("Render weights are energy-normalized when combined.");
    ImGui::BeginDisabled(parameters.debug_cascade_overrides_enable == false);
    for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
      ImGui::PushID(static_cast<int>(i + 100));
      bool cascade_enable = parameters.cascade_enable[i];
      if (ImGui::Checkbox("Enabled", &cascade_enable)) {
        parameters.cascade_enable[i] = cascade_enable;
      }
      ImGui::SliderFloat("Render weight", &parameters.cascade_render_weight[i], 0.0f, 4.0f, "%.3f");
      ImGui::Separator();
      ImGui::PopID();
    }
    const char* cascade_solo_items[] = {"All cascades", "Cascade 0", "Cascade 1", "Cascade 2"};
    int32_t solo_selector = parameters.solo_cascade + 1;
    if (ImGui::Combo("Solo cascade", &solo_selector, cascade_solo_items, 4)) {
      parameters.solo_cascade = solo_selector - 1;
    }
    ImGui::EndDisabled();
  }

  ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
  if (ImGui::CollapsingHeader("Rendering Debug")) {
    bool lock_lods = parameters.lock_lods;
    if (ImGui::Checkbox("Lock LODs", &lock_lods)) {
      parameters.lock_lods = lock_lods;
    }
    ImGui::SliderFloat("Transition cells (LOD)", &parameters.stitch_transition_cells, 1.0f, 16.0f, "%.3f");
    ImGui::SliderFloat("Mip color mix", &parameters.mip_color_mix, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Mip color enable", &parameters.mip_color_enable, 0.0f, 1.0f, "%.3f");
    const char* normal_visualize_items[] = {"Off", "Combined", "Cascade 0", "Cascade 1", "Cascade 2"};
    int normal_visualize_mode = parameters.normal_map_visualize_mode;
    if (ImGui::Combo("Normal map view", &normal_visualize_mode, normal_visualize_items, 5)) {
      parameters.normal_map_visualize_mode = max(0, min(normal_visualize_mode, 4));
    }
    bool normal_map_shading_enable = parameters.normal_map_shading_enable;
    if (ImGui::Checkbox("Apply normal map in shading", &normal_map_shading_enable)) {
      parameters.normal_map_shading_enable = normal_map_shading_enable;
    }
    ImGui::SliderFloat("Normal map scale", &parameters.normal_map_scale, 0.0f, 3.0f, "%.3f");
    bool wireframe_enable = parameters.wireframe_enable;
    if (ImGui::Checkbox("Wireframe", &wireframe_enable)) {
      parameters.wireframe_enable = wireframe_enable;
    }
    ImGui::ColorEdit3("Wireframe color", &parameters.wireframe_color.x);
  }

  if (spectrum_parameters_changed) {
    ocean.force_regenerate_spectrum();
  }

  ImGui::End();
}

void PlaygroundApp::init() {
  ETX_PROFILER_SCOPE();

  _width = static_cast<uint32_t>(sapp_width());
  _height = static_cast<uint32_t>(sapp_height());

  RHIInitInfo info = {
#if defined(ETX_PLATFORM_APPLE)
    .backend = RHIBackend::Metal,
#else
    .backend = RHIBackend::Vulkan,
#endif
    .enable_validation = true,
  };

  _rhi = RHIContext::create(info);
  _rhi.create_swapchain(sapp_win32_get_hwnd(), _width, _height);
  RHIImGuiDesc imgui_desc = {
    .color_format = _rhi.get_swapchain_format(),
    .depth_format = RHITextureFormat::D32_FLOAT,
  };
  _imgui.setup(_rhi, imgui_desc);

  ShaderCompiler::instance().initialize();

  std::vector<Vertex> vertices = {{{-0.5f, -0.5f, 0.0f}, {}, {}, {}, {0.0f, 0.0f}}, {{0.5f, -0.5f, 0.0f}, {}, {}, {}, {1.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.0f}, {}, {}, {}, {0.0f, 1.0f}}, {{0.5f, 0.5f, 0.0f}, {}, {}, {}, {1.0f, 1.0f}}};

  RHIBufferDesc vb_desc = {
    .size = vertices.size() * sizeof(Vertex),
    .usage = RHIBufferUsage::Vertex,
    .host_visible = true,
  };

  auto vb_result = _rhi.device().create_buffer(vb_desc);
  if (vb_result.result == RHIResult::Success) {
    _vertex_buffer = vb_result.handle;
    _rhi.device().update_buffer(_vertex_buffer, vertices.data(), vb_desc.size);
  }

  std::vector<uint32_t> indices = {0, 1, 2, 2, 1, 3};
  RHIBufferDesc ib_desc = {
    .size = indices.size() * sizeof(uint32_t),
    .usage = RHIBufferUsage::Index,
    .host_visible = true,
  };

  auto ib_result = _rhi.device().create_buffer(ib_desc);
  if (ib_result.result == RHIResult::Success) {
    _index_buffer = ib_result.handle;
    _rhi.device().update_buffer(_index_buffer, indices.data(), ib_desc.size);
  }

  RHITextureDesc depth_desc = {};
  depth_desc.width = _width;
  depth_desc.height = _height;
  depth_desc.format = RHITextureFormat::D32_FLOAT;
  depth_desc.usage = RHITextureUsage::DepthAttachment;
  auto depth_result = _rhi.device().create_texture(depth_desc);
  if (depth_result.result == RHIResult::Success) {
    _depth_buffer = depth_result.handle;
  }

  _envmap.setup(_rhi);
  _ocean.init(_rhi);

  std::string shader_source = env().file_in_data("playground/shaders/base.hlsl");

  ShaderCompiler::ShaderEntryPoint vs = {"VSMain", RHIShaderStage::Vertex};
  ShaderCompiler::ShaderEntryPoint ps = {"PSMain", RHIShaderStage::Fragment};

  auto compilation = ShaderCompiler::instance().compile(shader_source, {vs, ps});

  if (compilation.result == RHIResult::Success) {
    RHIGraphicsPipelineDesc p_desc = {};
    p_desc.vertex_shader.stage = RHIShaderStage::Vertex;
    p_desc.vertex_shader.entry_point = "VSMain";
    p_desc.vertex_shader.spirv_data = compilation.binaries[0].spirv_data;
    p_desc.vertex_shader.spirv_size = compilation.binaries[0].spirv_size;

    p_desc.fragment_shader.stage = RHIShaderStage::Fragment;
    p_desc.fragment_shader.entry_point = "PSMain";
    p_desc.fragment_shader.spirv_data = compilation.binaries[1].spirv_data;
    p_desc.fragment_shader.spirv_size = compilation.binaries[1].spirv_size;

    p_desc.rasterization.depth_clamp_enable = false;
    p_desc.rasterization.rasterizer_discard_enable = false;

    p_desc.depth_state.depth_test_enable = true;
    p_desc.depth_state.depth_write_enable = true;

    p_desc.blend.blend_enable = false;

    p_desc.primitive_topology = RHIPrimitiveTopology::TriangleList;

    p_desc.color_attachment_count = 1;
    p_desc.color_formats[0] = _rhi.get_swapchain_format();
    p_desc.depth_format = RHITextureFormat::D32_FLOAT;

    auto p_result = _rhi.device().create_graphics_pipeline(p_desc);
    if (p_result.result == RHIResult::Success) {
      _pipeline = p_result.handle;
    } else {
      log::error("Failed to create graphics pipeline: %u", p_result.result);
    }
  } else {
    log::error("Failed to compile shaders:\n%s", compilation.error_message.c_str());
  }

  build_camera(_camera, {5.0f, 5.0f, 5.0f}, normalize(float3{0.0f, 0.0f, 0.0f} - float3{5.0f, 5.0f, 5.0f}), kWorldUp, {_width, _height}, 45.0f);

  // Compute test Foundation
  RHITextureDesc st_desc = {};
  st_desc.width = 256;
  st_desc.height = 256;
  st_desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
  st_desc.usage = RHITextureUsage::Sampled | RHITextureUsage::Storage;
  auto st_result = _rhi.device().create_texture(st_desc);
  if (st_result.result == RHIResult::Success) {
    _test_storage_texture = st_result.handle;
  }

  std::string compute_source = env().file_in_data("playground/shaders/compute_test.hlsl");
  auto c_compilation = ShaderCompiler::instance().compile(compute_source, {{"CSMain", RHIShaderStage::Compute}});
  if (c_compilation.result == RHIResult::Success) {
    RHIComputePipelineDesc cp_desc = {};
    cp_desc.compute_shader.stage = RHIShaderStage::Compute;
    cp_desc.compute_shader.entry_point = "CSMain";
    cp_desc.compute_shader.spirv_data = c_compilation.binaries[0].spirv_data;
    cp_desc.compute_shader.spirv_size = c_compilation.binaries[0].spirv_size;
    auto cp_result = _rhi.device().create_compute_pipeline(cp_desc);
    if (cp_result.result == RHIResult::Success) {
      _compute_pipeline = cp_result.handle;
    }
  }
}

void PlaygroundApp::cleanup() {
  ETX_PROFILER_SCOPE();
  _rhi.wait_idle();
  _imgui.shutdown();
  _envmap.cleanup(_rhi);
  _ocean.cleanup(_rhi);
  if (_pipeline.valid()) {
    _rhi.device().destroy_pipeline(_pipeline);
  }
  if (_compute_pipeline.valid()) {
    _rhi.device().destroy_pipeline(_compute_pipeline);
  }
  if (_vertex_buffer.valid()) {
    _rhi.device().destroy_buffer(_vertex_buffer);
  }
  if (_index_buffer.valid()) {
    _rhi.device().destroy_buffer(_index_buffer);
  }
  if (_test_storage_texture.valid()) {
    _rhi.device().destroy_texture(_test_storage_texture);
  }
  _rhi.destroy_swapchain();
  ShaderCompiler::instance().shutdown();
}

void PlaygroundApp::frame() {
  ETX_PROFILER_SCOPE();

  uint32_t w = static_cast<uint32_t>(sapp_width());
  uint32_t h = static_cast<uint32_t>(sapp_height());

  if ((w != _width) || (h != _height)) {
    _width = w;
    _height = h;
    _rhi.wait_idle();
    _rhi.resize_swapchain(_width, _height);

    if (_depth_buffer.valid()) {
      _rhi.device().destroy_texture(_depth_buffer);
    }
    RHITextureDesc depth_desc = {};
    depth_desc.width = _width;
    depth_desc.height = _height;
    depth_desc.format = RHITextureFormat::D32_FLOAT;
    depth_desc.usage = RHITextureUsage::DepthAttachment;
    auto depth_result = _rhi.device().create_texture(depth_desc);
    if (depth_result.result == RHIResult::Success) {
      _depth_buffer = depth_result.handle;
    }
  }

  _rhi.begin_frame();
  RHITexture swapchain_texture = _rhi.get_current_swapchain_texture();

  RHICommandBuffer cmd = _rhi.get_command_buffer();
  _rhi.command_buffer_begin(cmd);

  auto t = sapp_frame_duration();
  _time += static_cast<float>(t);

  // Compute test dispatch
  if (_compute_pipeline.valid() && _test_storage_texture.valid()) {
    _rhi.cmd_texture_barrier(cmd, _test_storage_texture, RHIResourceState::Undefined, RHIResourceState::General);
    _rhi.cmd_set_pipeline(cmd, _compute_pipeline);

    struct ComputePushConstants {
      uint32_t outputTextureIndex;
      uint32_t width;
      uint32_t height;
      float time;
    } cpc;
    cpc.outputTextureIndex = get_bindless_descriptor_index(_test_storage_texture);
    cpc.width = 256;
    cpc.height = 256;
    cpc.time = _time;
    _rhi.cmd_push_constants(cmd, &cpc, sizeof(cpc), 0);
    _rhi.cmd_dispatch(cmd, {16, 16, 1});

    _rhi.cmd_texture_barrier(cmd, _test_storage_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
  }

  _camera_controller.update(t);
  _camera.film_size = {_width, _height};

  float4x4 view = look_at(_camera.position, _camera.position + _camera.direction, kWorldUp);
  float ocean_far_plane = _ocean.valid() ? (_ocean.coverage_radius() * 1.25f) : 8192.0f;
  float far_plane = max(8192.0f, ocean_far_plane);
  float4x4 proj = perspective(kPi / 4.0f, _width, _height, 1.0f, far_plane);
  float4x4 view_proj = proj * view;

  if (_ocean.valid()) {
    _ocean.update(_rhi, cmd, _time, _camera.position, _camera.direction, kPi / 4.0f, _width, _height);
  }

  float clear_color[4] = {0.1f, 0.2f, 0.3f, 1.0f};
  RHIResourceState final_state = RHIResourceState::ColorAttachment;
  _rhi.cmd_begin_render_pass(cmd, 1, &swapchain_texture, clear_color, _depth_buffer, &final_state, RHIResourceState::DepthAttachment);

  // Envmap background — drawn before geometry, depth write disabled
  if (_envmap.valid()) {
    _rhi.cmd_set_viewport(cmd, {0.0f, static_cast<float>(_height), static_cast<float>(_width), -static_cast<float>(_height), 0.0f, 1.0f});
    _rhi.cmd_set_scissor(cmd, {0, 0, _width, _height});
    _envmap.draw(_rhi, cmd, inverse(view_proj));
  }

  if (_ocean.valid()) {
    _rhi.cmd_set_viewport(cmd, {0.0f, static_cast<float>(_height), static_cast<float>(_width), -static_cast<float>(_height), 0.0f, 1.0f});
    _rhi.cmd_set_scissor(cmd, {0, 0, _width, _height});
    _ocean.draw(_rhi, cmd, view_proj, _camera.position, _envmap.texture());
  }

  if (_pipeline.valid()) {
    _rhi.cmd_set_viewport(cmd, {0.0f, static_cast<float>(_height), static_cast<float>(_width), -static_cast<float>(_height), 0.0f, 1.0f});
    _rhi.cmd_set_scissor(cmd, {0, 0, _width, _height});
    _rhi.cmd_set_pipeline(cmd, _pipeline);

    struct PushConstants {
      uint32_t vb_index;
      uint32_t test_texture_index;
      uint32_t test_sampler_index;
      uint32_t padding;
      float4x4 mvp;
    };

    PushConstants pc = {};
    pc.vb_index = get_bindless_descriptor_index(_vertex_buffer);
    pc.test_texture_index = get_bindless_descriptor_index(_test_storage_texture);
    pc.test_sampler_index = _rhi.get_sampler_index(RHISamplerType::LinearRepeat);

    RHIIndexedDrawDesc draw_desc = {
      .index_count = 6,
      .instance_count = 1,
      .index_type = RHIIndexType::UInt32,
    };

    // First plane: vertical, rotating around Y at origin
    float4 q1 = {0.0f, sinf(_time * 0.5f), 0.0f, cosf(_time * 0.5f)};
    float4x4 world1 = transform_matrix(float3{0.0f, 0.0f, 0.0f}, q1, float3{2.0f, 2.0f, 2.0f});
    pc.mvp = view_proj * world1;
    _rhi.cmd_push_constants(cmd, &pc, sizeof(PushConstants), 0);
    _rhi.cmd_draw_indexed(cmd, draw_desc, _index_buffer);

    // Second plane: horizontal floor at origin (rotated 90° around X)
    float4 q2 = {sinf(kPi / 4.0f), 0.0f, 0.0f, cosf(kPi / 4.0f)};
    float4x4 world2 = transform_matrix(float3{0.0f, 0.0f, 0.0f}, q2, float3{4.0f, 4.0f, 4.0f});
    pc.mvp = view_proj * world2;
    _rhi.cmd_push_constants(cmd, &pc, sizeof(PushConstants), 0);
    _rhi.cmd_draw_indexed(cmd, draw_desc, _index_buffer);
  }

  // GUI
  RHIImGuiFrameDesc imgui_frame_desc = {
    .width = _width,
    .height = _height,
    .delta_time = sapp_frame_duration(),
    .dpi_scale = sapp_dpi_scale(),
  };
  _imgui.new_frame(imgui_frame_desc);
  draw_ocean_control_panel(_ocean);

  _imgui.render(cmd);

  _rhi.cmd_end_render_pass(cmd);
  _rhi.command_buffer_end(cmd);

  _rhi.submit_frame_command_buffer(cmd);
  _rhi.present();
}

void PlaygroundApp::process_event(const sapp_event* e) {
  ETX_PROFILER_SCOPE();
  bool event_handled = _imgui.handle_event(e);
  bool imgui_wants_mouse = ImGui::GetIO().WantCaptureMouse;
  bool imgui_wants_keyboard = ImGui::GetIO().WantCaptureKeyboard;
  if ((event_handled == false) && (imgui_wants_mouse == false) && (imgui_wants_keyboard == false)) {
    _camera_controller.handle_event(e);
  }
}

}  // namespace etx
