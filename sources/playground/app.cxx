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

static constexpr RHITextureFormat k_scene_color_format = RHITextureFormat::R32G32B32A32_FLOAT;
static constexpr RHITextureFormat k_scene_depth_format = RHITextureFormat::D32_FLOAT;

static constexpr float k_cascade_length_ratio_min = 2.0f;
static constexpr float k_cascade_length_min[Ocean::k_cascade_count] = {20.0f, 10.0f, 5.0f};
static constexpr float k_cascade_length_max = 4000.0f;

static bool texture_format_is_srgb(RHITextureFormat format) {
  return (format == RHITextureFormat::R8G8B8A8_SRGB) || (format == RHITextureFormat::B8G8R8A8_SRGB);
}

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
    if (ImGui::SliderFloat("Spectrum water depth (m)", &parameters.water_depth, 1.0f, 4000.0f, "%.3f")) {
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
  if (ImGui::CollapsingHeader("Physical Water Shading")) {
    ImGui::TextUnformatted("Phase 1: parameterized optics scaffold (Fresnel/absorption controls).");
    bool physical_render_mode = parameters.physical_render_mode;
    if (ImGui::Checkbox("Physical shading mode", &physical_render_mode)) {
      parameters.physical_render_mode = physical_render_mode;
    }
    ImGui::SliderFloat("Water IOR", &parameters.water_ior, 1.0f, 1.6f, "%.4f");
    ImGui::SliderFloat3("Absorption coeff RGB (1/m)", &parameters.absorption_coeff_rgb.x, 0.0f, 2.0f, "%.4f");
    ImGui::SliderFloat3("Scattering coeff RGB (1/m)", &parameters.scattering_coeff_rgb.x, 0.0f, 2.0f, "%.4f");
    ImGui::SliderFloat("Env reflection intensity", &parameters.env_reflection_intensity, 0.0f, 4.0f, "%.3f");
    ImGui::SliderFloat("Optical depth (m)", &parameters.optical_depth_m, 0.0f, 100.0f, "%.3f");
    ImGui::SliderFloat("Unresolved slope roughness", &parameters.unresolved_slope_roughness, 0.0f, 0.5f, "%.4f");
    ImGui::SliderFloat("Specular AA strength", &parameters.specular_aa_strength, 0.0f, 4.0f, "%.3f");
    ImGui::SliderFloat("Refraction distortion scale", &parameters.refract_distortion_scale, 0.0f, 0.25f, "%.4f");
    bool sun_lighting_enable = parameters.sun_lighting_enable;
    if (ImGui::Checkbox("Enable sun lighting", &sun_lighting_enable)) {
      parameters.sun_lighting_enable = sun_lighting_enable;
    }
    ImGui::SliderFloat3("Sun direction", &parameters.sun_direction.x, -1.0f, 1.0f, "%.3f");
    ImGui::SliderFloat3("Sun radiance RGB", &parameters.sun_radiance.x, 0.0f, 64.0f, "%.3f");
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
    const char* surface_normal_visualize_items[] = {
      "Off", "Combined", "Cascade 0", "Cascade 1", "Cascade 2", "Ref error", "Mip drift", "Foldover risk"};
    int surface_normal_visualize_mode = parameters.surface_normal_visualize_mode;
    if (ImGui::Combo("Surface normal view", &surface_normal_visualize_mode, surface_normal_visualize_items, 8)) {
      parameters.surface_normal_visualize_mode = max(0, min(surface_normal_visualize_mode, 7));
    }
    ImGui::TextUnformatted("Debug heatmaps: Ref error (deg), Mip drift (deg), Foldover risk (magenta).");
    ImGui::TextUnformatted("Numeric stats require GPU readback support (not wired yet).");
    bool surface_normal_shading_enable = parameters.surface_normal_shading_enable;
    if (ImGui::Checkbox("Apply surface normal in shading", &surface_normal_shading_enable)) {
      parameters.surface_normal_shading_enable = surface_normal_shading_enable;
    }
    ImGui::SliderFloat("Surface normal strength", &parameters.surface_normal_strength, 0.0f, 1.0f, "%.3f");
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

void PlaygroundApp::recreate_scene_targets(uint32_t width, uint32_t height) {
  if ((width == 0u) || (height == 0u)) {
    return;
  }

  _render_width = width;
  _render_height = height;
  _scene_opaque_color_initialized = false;
  _scene_color_initialized = false;

  if (_depth_buffer.valid()) {
    _rhi.device().destroy_texture(_depth_buffer);
    _depth_buffer = {};
  }
  if (_scene_opaque_color_buffer.valid()) {
    _rhi.device().destroy_texture(_scene_opaque_color_buffer);
    _scene_opaque_color_buffer = {};
  }
  if (_scene_color_buffer.valid()) {
    _rhi.device().destroy_texture(_scene_color_buffer);
    _scene_color_buffer = {};
  }

  RHITextureDesc scene_color_desc = {};
  scene_color_desc.width = width;
  scene_color_desc.height = height;
  scene_color_desc.format = k_scene_color_format;
  scene_color_desc.usage = RHITextureUsage::ColorAttachment | RHITextureUsage::Sampled;
  auto scene_opaque_color_result = _rhi.device().create_texture(scene_color_desc);
  if (scene_opaque_color_result.result == RHIResult::Success) {
    _scene_opaque_color_buffer = scene_opaque_color_result.handle;
  } else {
    log::error("Failed to recreate scene opaque HDR color texture: %u", scene_opaque_color_result.result);
  }

  auto scene_color_result = _rhi.device().create_texture(scene_color_desc);
  if (scene_color_result.result == RHIResult::Success) {
    _scene_color_buffer = scene_color_result.handle;
  } else {
    log::error("Failed to recreate scene final HDR color texture: %u", scene_color_result.result);
  }

  RHITextureDesc depth_desc = {};
  depth_desc.width = width;
  depth_desc.height = height;
  depth_desc.format = k_scene_depth_format;
  depth_desc.usage = RHITextureUsage::DepthAttachment | RHITextureUsage::Sampled;
  auto depth_result = _rhi.device().create_texture(depth_desc);
  if (depth_result.result == RHIResult::Success) {
    _depth_buffer = depth_result.handle;
  } else {
    log::error("Failed to recreate depth texture: %u", depth_result.result);
  }
}

void PlaygroundApp::sync_scene_targets_to_swapchain_extent() {
  const RHIExtent2D swapchain_extent = _rhi.get_swapchain_extent();
  if ((swapchain_extent.width == 0u) || (swapchain_extent.height == 0u)) {
    return;
  }

  const bool extent_changed = ((_render_width != swapchain_extent.width) || (_render_height != swapchain_extent.height));
  const bool missing_scene_opaque_color = (_scene_opaque_color_buffer.valid() == false);
  const bool missing_scene_color = (_scene_color_buffer.valid() == false);
  const bool missing_depth = (_depth_buffer.valid() == false);

  if ((extent_changed == false) && (missing_scene_opaque_color == false) && (missing_scene_color == false) && (missing_depth == false)) {
    return;
  }

  recreate_scene_targets(swapchain_extent.width, swapchain_extent.height);
}

bool PlaygroundApp::recreate_tonemap_pipeline() {
  if (_tonemap_pipeline.valid()) {
    _rhi.device().destroy_pipeline(_tonemap_pipeline);
    _tonemap_pipeline = {};
  }

  ShaderCompiler::ShaderEntryPoint vs = {"VSMain", RHIShaderStage::Vertex};
  ShaderCompiler::ShaderEntryPoint ps = {"PSMain", RHIShaderStage::Fragment};

  std::string tonemap_shader_source = env().file_in_data("playground/shaders/tonemap.hlsl");
  auto tonemap_compilation = ShaderCompiler::instance().compile(tonemap_shader_source, {vs, ps});
  if (tonemap_compilation.result == RHIResult::Success) {
    RHIGraphicsPipelineDesc tonemap_desc = {};
    tonemap_desc.vertex_shader.stage = RHIShaderStage::Vertex;
    tonemap_desc.vertex_shader.entry_point = "VSMain";
    tonemap_desc.vertex_shader.spirv_data = tonemap_compilation.binaries[0].spirv_data;
    tonemap_desc.vertex_shader.spirv_size = tonemap_compilation.binaries[0].spirv_size;

    tonemap_desc.fragment_shader.stage = RHIShaderStage::Fragment;
    tonemap_desc.fragment_shader.entry_point = "PSMain";
    tonemap_desc.fragment_shader.spirv_data = tonemap_compilation.binaries[1].spirv_data;
    tonemap_desc.fragment_shader.spirv_size = tonemap_compilation.binaries[1].spirv_size;

    tonemap_desc.rasterization.depth_clamp_enable = false;
    tonemap_desc.rasterization.rasterizer_discard_enable = false;
    tonemap_desc.depth_state.depth_test_enable = false;
    tonemap_desc.depth_state.depth_write_enable = false;
    tonemap_desc.blend.blend_enable = false;
    tonemap_desc.primitive_topology = RHIPrimitiveTopology::TriangleList;
    tonemap_desc.color_attachment_count = 1;
    tonemap_desc.color_formats[0] = _rhi.get_swapchain_format();
    tonemap_desc.depth_format = k_scene_depth_format;

    auto tonemap_result = _rhi.device().create_graphics_pipeline(tonemap_desc);
    if (tonemap_result.result == RHIResult::Success) {
      _tonemap_pipeline = tonemap_result.handle;
      return true;
    }

    log::error("Failed to create tonemap pipeline: %u", tonemap_result.result);
    return false;
  }

  log::error("Failed to compile tonemap shaders:\n%s", tonemap_compilation.error_message.c_str());
  return false;
}

void PlaygroundApp::sync_swapchain_dependent_resources() {
  const RHITextureFormat current_swapchain_format = _rhi.get_swapchain_format();
  if (current_swapchain_format == RHITextureFormat::Undefined) {
    return;
  }

  const bool format_changed = (_swapchain_color_format != current_swapchain_format);
  const bool initial_setup = (_swapchain_color_format == RHITextureFormat::Undefined);
  if ((format_changed == false) && (initial_setup == false)) {
    return;
  }

  RHIImGuiDesc imgui_desc = {
    .color_format = current_swapchain_format,
    .depth_format = k_scene_depth_format,
  };
  RHIResult imgui_result = _imgui.setup(_rhi, imgui_desc);
  if (imgui_result != RHIResult::Success) {
    log::error("Failed to (re)create ImGui resources for swapchain format: %u", static_cast<uint32_t>(imgui_result));
  }

  recreate_tonemap_pipeline();
  _swapchain_color_format = current_swapchain_format;
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

  ShaderCompiler::instance().initialize();
  sync_swapchain_dependent_resources();

  std::vector<Vertex> vertices = {
    // +X
    {{0.5f, -0.5f, -0.5f}, {}, {}, {}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.5f}, {}, {}, {}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {}, {}, {}, {0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {}, {}, {}, {1.0f, 1.0f}},
    // -X
    {{-0.5f, -0.5f, 0.5f}, {}, {}, {}, {0.0f, 0.0f}},
    {{-0.5f, -0.5f, -0.5f}, {}, {}, {}, {1.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.5f}, {}, {}, {}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {}, {}, {}, {1.0f, 1.0f}},
    // +Y
    {{-0.5f, 0.5f, -0.5f}, {}, {}, {}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {}, {}, {}, {1.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.5f}, {}, {}, {}, {0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {}, {}, {}, {1.0f, 1.0f}},
    // -Y
    {{-0.5f, -0.5f, 0.5f}, {}, {}, {}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.5f}, {}, {}, {}, {1.0f, 0.0f}},
    {{-0.5f, -0.5f, -0.5f}, {}, {}, {}, {0.0f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {}, {}, {}, {1.0f, 1.0f}},
    // +Z
    {{-0.5f, -0.5f, 0.5f}, {}, {}, {}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.5f}, {}, {}, {}, {1.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.5f}, {}, {}, {}, {0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {}, {}, {}, {1.0f, 1.0f}},
    // -Z
    {{0.5f, -0.5f, -0.5f}, {}, {}, {}, {0.0f, 0.0f}},
    {{-0.5f, -0.5f, -0.5f}, {}, {}, {}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {}, {}, {}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {}, {}, {}, {1.0f, 1.0f}},
  };

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

  std::vector<uint32_t> indices = {
    0, 1, 2, 2, 1, 3,      // +X
    4, 5, 6, 6, 5, 7,      // -X
    8, 9, 10, 10, 9, 11,   // +Y
    12, 13, 14, 14, 13, 15, // -Y
    16, 17, 18, 18, 17, 19, // +Z
    20, 21, 22, 22, 21, 23, // -Z
  };
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

  sync_scene_targets_to_swapchain_extent();

  _envmap.setup(_rhi, k_scene_color_format, k_scene_depth_format);
  _ocean.init(_rhi, k_scene_color_format, k_scene_depth_format);

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
    p_desc.color_formats[0] = k_scene_color_format;
    p_desc.depth_format = k_scene_depth_format;

    auto p_result = _rhi.device().create_graphics_pipeline(p_desc);
    if (p_result.result == RHIResult::Success) {
      _pipeline = p_result.handle;
    } else {
      log::error("Failed to create graphics pipeline: %u", p_result.result);
    }
  } else {
    log::error("Failed to compile shaders:\n%s", compilation.error_message.c_str());
  }

  uint32_t camera_width = (_render_width > 0u) ? _render_width : _width;
  uint32_t camera_height = (_render_height > 0u) ? _render_height : _height;
  build_camera(_camera, {5.0f, 5.0f, 5.0f}, normalize(float3{0.0f, 0.0f, 0.0f} - float3{5.0f, 5.0f, 5.0f}), kWorldUp, {camera_width, camera_height}, 45.0f);
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
  if (_tonemap_pipeline.valid()) {
    _rhi.device().destroy_pipeline(_tonemap_pipeline);
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
  if (_scene_opaque_color_buffer.valid()) {
    _rhi.device().destroy_texture(_scene_opaque_color_buffer);
  }
  if (_scene_color_buffer.valid()) {
    _rhi.device().destroy_texture(_scene_color_buffer);
  }
  if (_depth_buffer.valid()) {
    _rhi.device().destroy_texture(_depth_buffer);
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
    _rhi.resize_swapchain(_width, _height);
    sync_scene_targets_to_swapchain_extent();
  }

  _rhi.begin_frame();
  sync_swapchain_dependent_resources();
  sync_scene_targets_to_swapchain_extent();

  const RHIExtent2D swapchain_extent = _rhi.get_swapchain_extent();
  if ((swapchain_extent.width == 0u) || (swapchain_extent.height == 0u)) {
    return;
  }

  const uint32_t render_width = (_render_width > 0u) ? _render_width : swapchain_extent.width;
  const uint32_t render_height = (_render_height > 0u) ? _render_height : swapchain_extent.height;
  RHITexture swapchain_texture = _rhi.get_current_swapchain_texture();

  RHICommandBuffer cmd = _rhi.get_command_buffer();
  _rhi.command_buffer_begin(cmd);

  auto t = sapp_frame_duration();
  _time += static_cast<float>(t);

  _camera_controller.update(t);
  _camera.film_size = {render_width, render_height};

  float4x4 view = look_at(_camera.position, _camera.position + _camera.direction, kWorldUp);
  float ocean_far_plane = _ocean.valid() ? (_ocean.coverage_radius() * 1.25f) : 8192.0f;
  float far_plane = max(8192.0f, ocean_far_plane);
  float4x4 proj = perspective(kPi / 4.0f, render_width, render_height, 1.0f, far_plane);
  float4x4 view_proj = proj * view;
  float4x4 inv_view_proj = inverse(view_proj);

  if (_ocean.valid()) {
    _ocean.update(_rhi, cmd, _time, _camera.position, _camera.direction, kPi / 4.0f, render_width, render_height);
  }

  bool hdr_path_available = _scene_opaque_color_buffer.valid() && _scene_color_buffer.valid() && _tonemap_pipeline.valid();

  float scene_clear_color[4] = {0.1f, 0.2f, 0.3f, 1.0f};
  auto set_viewport_and_scissor = [&]() {
    _rhi.cmd_set_viewport(cmd, {0.0f, static_cast<float>(render_height), static_cast<float>(render_width), -static_cast<float>(render_height), 0.0f, 1.0f});
    _rhi.cmd_set_scissor(cmd, {0, 0, render_width, render_height});
  };

  auto draw_env_and_buoy = [&]() {
    if (_envmap.valid()) {
      set_viewport_and_scissor();
      _envmap.draw(_rhi, cmd, inv_view_proj);
    }

    if (_pipeline.valid()) {
      set_viewport_and_scissor();
      _rhi.cmd_set_pipeline(cmd, _pipeline);

      struct PushConstants {
        uint32_t vb_index;
        uint32_t test_texture_index;
        uint32_t test_sampler_index;
        uint32_t ocean_sampler_index;
        uint4 disp_map_index_01;
        float4 cascade_len_weight_01;
        float4 buoy_center_draft;
        float4x4 vp;
      };

      PushConstants pc = {};
      pc.vb_index = get_bindless_descriptor_index(_vertex_buffer);
      pc.test_texture_index = 0u;
      pc.test_sampler_index = 0u;
      pc.ocean_sampler_index = _rhi.get_sampler_index(RHISamplerType::LinearRepeat);
      pc.disp_map_index_01 = {0u, 0u, 0u, 0u};
      pc.cascade_len_weight_01 = {1.0f, 1.0f, 0.0f, 0.0f};
      if (_ocean.valid()) {
        float cascade_weights[3] = {0.0f, 0.0f, 0.0f};
        _ocean.effective_cascade_weights(cascade_weights);
        const OceanParameters& ocean_parameters = _ocean.parameters();
        float rigid_weight_0 = max(0.0f, cascade_weights[0]);
        float rigid_weight_1 = max(0.0f, cascade_weights[1]);
        float rigid_weight_sum = rigid_weight_0 + rigid_weight_1;
        if (rigid_weight_sum > 1.0e-6f) {
          rigid_weight_0 /= rigid_weight_sum;
          rigid_weight_1 /= rigid_weight_sum;
        } else {
          rigid_weight_0 = 1.0f;
          rigid_weight_1 = 0.0f;
        }
        pc.cascade_len_weight_01 = {ocean_parameters.cascade_lengths[0], ocean_parameters.cascade_lengths[1], rigid_weight_0, rigid_weight_1};

        for (uint32_t i = 0u; i < 2u; ++i) {
          RHITexture disp_tex = _ocean.displacement_texture(i);
          if (disp_tex.valid()) {
            if (i == 0u) {
              pc.disp_map_index_01.x = get_bindless_descriptor_index(disp_tex);
            } else {
              pc.disp_map_index_01.y = get_bindless_descriptor_index(disp_tex);
            }
          }
        }
      }

      float buoy_center_x = sinf(_time * 0.17f) * 2.0f;
      float buoy_center_z = cosf(_time * 0.13f) * 1.4f;
      float buoy_draft = 0.18f;
      float buoy_sample_radius_scale = 0.9f;
      pc.buoy_center_draft = {buoy_center_x, buoy_center_z, buoy_draft, buoy_sample_radius_scale};
      pc.vp = view_proj;

      RHIIndexedDrawDesc draw_desc = {
        .index_count = 36,
        .instance_count = 1,
        .index_type = RHIIndexType::UInt32,
      };
      _rhi.cmd_push_constants(cmd, &pc, sizeof(PushConstants), 0);
      _rhi.cmd_draw_indexed(cmd, draw_desc, _index_buffer);
    }
  };

  if (hdr_path_available) {
    RHIResourceState opaque_old_state = _scene_opaque_color_initialized ? RHIResourceState::ShaderReadOnly : RHIResourceState::Undefined;
    _rhi.cmd_texture_barrier(cmd, _scene_opaque_color_buffer, opaque_old_state, RHIResourceState::ColorAttachment);
    _scene_opaque_color_initialized = true;
    RHIResourceState opaque_final_state = RHIResourceState::ShaderReadOnly;
    _rhi.cmd_begin_render_pass(cmd, 1, &_scene_opaque_color_buffer, scene_clear_color, _depth_buffer, &opaque_final_state, RHIResourceState::DepthAttachment);
    draw_env_and_buoy();
    _rhi.cmd_end_render_pass(cmd);

    RHIResourceState final_old_state = _scene_color_initialized ? RHIResourceState::ShaderReadOnly : RHIResourceState::Undefined;
    _rhi.cmd_texture_barrier(cmd, _scene_color_buffer, final_old_state, RHIResourceState::ColorAttachment);
    _scene_color_initialized = true;
    RHIResourceState final_scene_state = RHIResourceState::ShaderReadOnly;
    _rhi.cmd_begin_render_pass(cmd, 1, &_scene_color_buffer, scene_clear_color, _depth_buffer, &final_scene_state, RHIResourceState::DepthAttachment);
    draw_env_and_buoy();
    if (_ocean.valid()) {
      set_viewport_and_scissor();
      _ocean.draw(_rhi, cmd, view_proj, inv_view_proj, _camera.position, _envmap.texture(), _scene_opaque_color_buffer, render_width, render_height);
    }
  } else {
    RHITexture scene_color_target = swapchain_texture;
    RHIResourceState scene_final_state = RHIResourceState::ColorAttachment;
    _rhi.cmd_begin_render_pass(cmd, 1, &scene_color_target, scene_clear_color, _depth_buffer, &scene_final_state, RHIResourceState::DepthAttachment);
    draw_env_and_buoy();
    if (_ocean.valid()) {
      set_viewport_and_scissor();
      _ocean.draw(_rhi, cmd, view_proj, inv_view_proj, _camera.position, _envmap.texture(), {}, render_width, render_height);
    }
  }

  // GUI
  RHIImGuiFrameDesc imgui_frame_desc = {
    .width = render_width,
    .height = render_height,
    .delta_time = sapp_frame_duration(),
    .dpi_scale = sapp_dpi_scale(),
  };
  _imgui.new_frame(imgui_frame_desc);
  draw_ocean_control_panel(_ocean);

  if (hdr_path_available) {
    _rhi.cmd_end_render_pass(cmd);

    float present_clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    RHIResourceState swapchain_final_state = RHIResourceState::ColorAttachment;
    _rhi.cmd_begin_render_pass(cmd, 1, &swapchain_texture, present_clear_color, _depth_buffer, &swapchain_final_state, RHIResourceState::DepthAttachment);

    _rhi.cmd_set_viewport(cmd, {0.0f, static_cast<float>(render_height), static_cast<float>(render_width), -static_cast<float>(render_height), 0.0f, 1.0f});
    _rhi.cmd_set_scissor(cmd, {0, 0, render_width, render_height});
    _rhi.cmd_set_pipeline(cmd, _tonemap_pipeline);

    struct TonemapPushConstants {
      uint32_t hdr_texture_index;
      uint32_t sampler_index;
      float exposure;
      uint32_t output_gamma_encode;
    } tonemap_pc = {};

    tonemap_pc.hdr_texture_index = get_bindless_descriptor_index(_scene_color_buffer);
    tonemap_pc.sampler_index = _rhi.get_sampler_index(RHISamplerType::LinearClamp);
    tonemap_pc.exposure = _tonemap_exposure;
    tonemap_pc.output_gamma_encode = texture_format_is_srgb(_rhi.get_swapchain_format()) ? 0u : 1u;
    _rhi.cmd_push_constants(cmd, &tonemap_pc, sizeof(TonemapPushConstants), 0);
    _rhi.cmd_draw(cmd, {.vertex_count = 3, .instance_count = 1});

    _imgui.render(cmd);
    _rhi.cmd_end_render_pass(cmd);
  } else {
    _imgui.render(cmd);
    _rhi.cmd_end_render_pass(cmd);
  }

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
