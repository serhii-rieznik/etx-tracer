#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>

#include "app.hxx"

#include <imgui.h>

#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <etx/render/interop/geometry.hxx>
#include <etx/render/shared/scattering.hxx>

#if defined(ETX_PLATFORM_WINDOWS)
# define WIN32_LEAN_AND_MEAN 1
# include <Windows.h>
#endif

namespace etx {
namespace {

RHIBackend select_default_backend() {
#if ETX_PLATFORM_APPLE
  return RHIBackend::Metal;
#else
  return RHIBackend::Vulkan;
#endif
}

const char* backend_name(RHIBackend backend) {
  switch (backend) {
    case RHIBackend::Vulkan:
      return "Vulkan";
    case RHIBackend::Metal:
      return "Metal";
    default:
      return "Unknown";
  }
}

}  // namespace

bool PlaygroundApp::initialized() const {
  return _rhi.valid();
}

bool PlaygroundApp::recreate_headless_present_target(uint32_t width, uint32_t height) {
  if (_headless_present_texture.valid()) {
    _rhi.device().destroy_texture(_headless_present_texture);
    _headless_present_texture = {};
  }

  if ((width == 0u) || (height == 0u)) {
    return false;
  }

  RHITextureDesc desc = {};
  desc.width = width;
  desc.height = height;
  desc.format = _rhi.get_swapchain_format();
  desc.usage = RHITextureUsage::ColorAttachment | RHITextureUsage::TransferSrc;
  auto result = _rhi.device().create_texture(desc);
  if (result.result != RHIResult::Success) {
    log::error("Playground headless: failed to create present target: %u", static_cast<uint32_t>(result.result));
    return false;
  }

  _headless_present_texture = result.handle;
  return true;
}

static constexpr RHITextureFormat k_scene_color_format = RHITextureFormat::R16G16B16A16_FLOAT;
static constexpr RHITextureFormat k_scene_depth_format = RHITextureFormat::D32_FLOAT;
static constexpr uint32_t k_default_scene_msaa_sample_count = 4u;

static uint32_t default_scene_msaa_sample_count(RHIBackend backend) {
  return (backend == RHIBackend::Metal) ? 1u : k_default_scene_msaa_sample_count;
}

static constexpr float k_cascade_length_min[Ocean::k_cascade_count] = {20.0f, 10.0f, 5.0f};
static constexpr float k_cascade_length_max = 4000.0f;
static constexpr uint2 k_sky_envmap_dimensions = {1024u, 512u};
static constexpr uint2 k_sun_sprite_dimensions = {256u, 256u};
static constexpr float k_sun_angular_size_radians = 0.53f * (kPi / 180.0f);
static constexpr float k_sun_temperature_kelvin = 5800.0f;
static constexpr int32_t k_ocean_obj_export_default_size_m = 100;
static constexpr uint32_t k_ocean_obj_export_grid_resolution = 1001u;
enum GpuTimingQueryIndex : uint32_t {
  k_gpu_timing_query_frame_begin = 0u,
  k_gpu_timing_query_after_sun_sky_regen = 1u,
  k_gpu_timing_query_after_ocean_update = 2u,
  k_gpu_timing_query_after_resource_prep = 3u,
  k_gpu_timing_query_after_opaque_pass = 4u,
  k_gpu_timing_query_after_opaque_resolve = 5u,
  k_gpu_timing_query_after_scene_pass = 6u,
  k_gpu_timing_query_after_scene_resolve = 7u,
  k_gpu_timing_query_after_tonemap = 8u,
  k_gpu_timing_query_after_imgui_render = 9u,
  k_gpu_timing_query_after_final_pass = 10u,
  k_gpu_timing_query_frame_end = 11u,
};

static float sun_disk_solid_angle_sr(float angular_size_radians) {
  float half_angle = max(0.5f * angular_size_radians, 0.0f);
  float cos_half_angle = cosf(half_angle);
  return max(2.0f * kPi * (1.0f - cos_half_angle), 1.0e-8f);
}

static SpectralDistribution sync_ocean_sun_spectrum(OceanParameters& parameters) {
  float sun_brightness = max(0.0f, parameters.sun_brightness);
  SpectralDistribution sun_emission_spectrum = SpectralDistribution::from_normalized_black_body(k_sun_temperature_kelvin, sun_brightness);
  const float3& integrated_rgb = sun_emission_spectrum.integrated();
  parameters.sun_radiance = {max(0.0f, integrated_rgb.x), max(0.0f, integrated_rgb.y), max(0.0f, integrated_rgb.z)};
  return sun_emission_spectrum;
}

static float sun_elevation_degrees_from_direction(const float3& direction) {
  float3 dir = normalize(direction);
  float y = min(1.0f, max(-1.0f, dir.y));
  return asinf(y) * (180.0f / kPi);
}

static float sun_azimuth_degrees_from_direction(const float3& direction) {
  float3 dir = normalize(direction);
  return atan2f(dir.z, dir.x) * (180.0f / kPi);
}

static float3 direction_with_sun_elevation_degrees(const float3& direction, float elevation_degrees) {
  float3 dir = normalize(direction);
  float2 xz = {dir.x, dir.z};
  float xz_len = length(xz);
  if (xz_len < 1.0e-6f) {
    xz = {1.0f, 0.0f};
    xz_len = 1.0f;
  }
  xz /= xz_len;

  float elevation_radians = elevation_degrees * (kPi / 180.0f);
  float cos_elevation = cosf(elevation_radians);
  float sin_elevation = sinf(elevation_radians);
  return normalize(float3{xz.x * cos_elevation, sin_elevation, xz.y * cos_elevation});
}

static float3 direction_with_sun_azimuth_degrees(const float3& direction, float azimuth_degrees) {
  float3 dir = normalize(direction);
  float elevation_radians = asinf(min(1.0f, max(-1.0f, dir.y)));
  float cos_elevation = cosf(elevation_radians);
  float sin_elevation = sinf(elevation_radians);
  float azimuth_radians = azimuth_degrees * (kPi / 180.0f);

  float x = cosf(azimuth_radians) * cos_elevation;
  float z = sinf(azimuth_radians) * cos_elevation;
  return normalize(float3{x, sin_elevation, z});
}

static bool texture_format_is_srgb(RHITextureFormat format) {
  return (format == RHITextureFormat::R8G8B8A8_SRGB) || (format == RHITextureFormat::B8G8R8A8_SRGB);
}

struct OceanExportDisplacementTexel {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 0.0f;
};

static_assert(sizeof(OceanExportDisplacementTexel) == 16u, "Ocean export texel layout must match R32G32B32A32_FLOAT");

struct OceanExportTextureData {
  uint32_t width = 0u;
  uint32_t height = 0u;
  std::vector<OceanExportDisplacementTexel> texels = {};
};

struct OceanExportVertex {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

static float wrap_repeat_coordinate(float value) {
  float wrapped = value - floorf(value);
  if (wrapped < 0.0f) {
    wrapped += 1.0f;
  }
  return wrapped;
}

static uint32_t wrap_repeat_index(int32_t index, uint32_t dimension) {
  if (dimension == 0u) {
    return 0u;
  }

  int32_t dim = static_cast<int32_t>(dimension);
  int32_t wrapped = index % dim;
  if (wrapped < 0) {
    wrapped += dim;
  }
  return static_cast<uint32_t>(wrapped);
}

static float3 sample_ocean_export_displacement_texture(const OceanExportTextureData& texture_data, const float2& uv) {
  if ((texture_data.width == 0u) || (texture_data.height == 0u) || texture_data.texels.empty()) {
    return {0.0f, 0.0f, 0.0f};
  }

  float u = wrap_repeat_coordinate(uv.x);
  float v = wrap_repeat_coordinate(uv.y);
  float x = (u * static_cast<float>(texture_data.width)) - 0.5f;
  float y = (v * static_cast<float>(texture_data.height)) - 0.5f;

  int32_t x0 = static_cast<int32_t>(floorf(x));
  int32_t y0 = static_cast<int32_t>(floorf(y));
  int32_t x1 = x0 + 1;
  int32_t y1 = y0 + 1;

  float tx = x - floorf(x);
  float ty = y - floorf(y);

  uint32_t ix0 = wrap_repeat_index(x0, texture_data.width);
  uint32_t iy0 = wrap_repeat_index(y0, texture_data.height);
  uint32_t ix1 = wrap_repeat_index(x1, texture_data.width);
  uint32_t iy1 = wrap_repeat_index(y1, texture_data.height);

  const OceanExportDisplacementTexel& s00 = texture_data.texels[iy0 * texture_data.width + ix0];
  const OceanExportDisplacementTexel& s10 = texture_data.texels[iy0 * texture_data.width + ix1];
  const OceanExportDisplacementTexel& s01 = texture_data.texels[iy1 * texture_data.width + ix0];
  const OceanExportDisplacementTexel& s11 = texture_data.texels[iy1 * texture_data.width + ix1];

  float3 a = {lerp(s00.x, s10.x, tx), lerp(s00.y, s10.y, tx), lerp(s00.z, s10.z, tx)};
  float3 b = {lerp(s01.x, s11.x, tx), lerp(s01.y, s11.y, tx), lerp(s01.z, s11.z, tx)};
  return {lerp(a.x, b.x, ty), lerp(a.y, b.y, ty), lerp(a.z, b.z, ty)};
}

static float3 sample_ocean_export_displacement_field(const OceanExportTextureData texture_data[Ocean::k_cascade_count], const float cascade_lengths[Ocean::k_cascade_count],
  const float cascade_weights[Ocean::k_cascade_count], const float2& world_xz) {
  float3 displacement = {0.0f, 0.0f, 0.0f};
  for (uint32_t i = 0u; i < Ocean::k_cascade_count; ++i) {
    if (cascade_weights[i] <= 0.0f) {
      continue;
    }
    if ((texture_data[i].width == 0u) || (texture_data[i].height == 0u)) {
      continue;
    }
    float cascade_length = max(cascade_lengths[i], 1.0e-6f);
    float2 uv = {world_xz.x / cascade_length, world_xz.y / cascade_length};
    float3 sample = sample_ocean_export_displacement_texture(texture_data[i], uv);
    displacement += sample * cascade_weights[i];
  }
  return displacement;
}

static bool write_ocean_surface_obj_file(const char* file_name, const OceanExportTextureData texture_data[Ocean::k_cascade_count],
  const float cascade_lengths[Ocean::k_cascade_count], const float cascade_weights[Ocean::k_cascade_count], float area_size_m, uint32_t grid_resolution, const float3& center,
  std::string& out_error) {
  if ((file_name == nullptr) || (file_name[0] == '\0')) {
    out_error = "Invalid output path";
    return false;
  }
  if (grid_resolution < 2u) {
    out_error = "Grid resolution must be at least 2";
    return false;
  }

  FILE* file = fopen(file_name, "wb");
  if (file == nullptr) {
    out_error = "Failed to open OBJ file for writing";
    return false;
  }

  setvbuf(file, nullptr, _IOFBF, 1u << 20u);

  const uint64_t vertex_count_u64 = static_cast<uint64_t>(grid_resolution) * static_cast<uint64_t>(grid_resolution);
  if (vertex_count_u64 > static_cast<uint64_t>(0xFFFFFFFFu)) {
    fclose(file);
    out_error = "Grid resolution is too large";
    return false;
  }

  std::vector<OceanExportVertex> vertices = {};
  vertices.resize(static_cast<size_t>(vertex_count_u64));

  float half_extent = 0.5f * area_size_m;
  float denom = static_cast<float>(grid_resolution - 1u);
  for (uint32_t z = 0u; z < grid_resolution; ++z) {
    float z_lerp = static_cast<float>(z) / denom;
    float world_z = center.z + lerp(-half_extent, half_extent, z_lerp);
    for (uint32_t x = 0u; x < grid_resolution; ++x) {
      float x_lerp = static_cast<float>(x) / denom;
      float world_x = center.x + lerp(-half_extent, half_extent, x_lerp);
      float2 world_xz = {world_x, world_z};
      float3 displacement = sample_ocean_export_displacement_field(texture_data, cascade_lengths, cascade_weights, world_xz);
      OceanExportVertex& vertex = vertices[static_cast<size_t>(z) * static_cast<size_t>(grid_resolution) + x];
      vertex.x = world_x + displacement.x;
      vertex.y = displacement.y;
      vertex.z = world_z + displacement.z;
    }
  }

  std::vector<OceanExportVertex> normals = {};
  normals.resize(static_cast<size_t>(vertex_count_u64));
  for (uint32_t z = 0u; z < grid_resolution; ++z) {
    uint32_t z_prev = (z > 0u) ? (z - 1u) : z;
    uint32_t z_next = ((z + 1u) < grid_resolution) ? (z + 1u) : z;
    for (uint32_t x = 0u; x < grid_resolution; ++x) {
      uint32_t x_prev = (x > 0u) ? (x - 1u) : x;
      uint32_t x_next = ((x + 1u) < grid_resolution) ? (x + 1u) : x;

      const OceanExportVertex& p_left = vertices[static_cast<size_t>(z) * static_cast<size_t>(grid_resolution) + x_prev];
      const OceanExportVertex& p_right = vertices[static_cast<size_t>(z) * static_cast<size_t>(grid_resolution) + x_next];
      const OceanExportVertex& p_down = vertices[static_cast<size_t>(z_prev) * static_cast<size_t>(grid_resolution) + x];
      const OceanExportVertex& p_up = vertices[static_cast<size_t>(z_next) * static_cast<size_t>(grid_resolution) + x];

      float3 tangent_x = {p_right.x - p_left.x, p_right.y - p_left.y, p_right.z - p_left.z};
      float3 tangent_z = {p_up.x - p_down.x, p_up.y - p_down.y, p_up.z - p_down.z};
      float3 normal = cross(tangent_z, tangent_x);
      float normal_len2 = dot(normal, normal);
      if (normal_len2 > 1.0e-20f) {
        normal *= (1.0f / sqrtf(normal_len2));
      } else {
        normal = {0.0f, 1.0f, 0.0f};
      }
      if (normal.y < 0.0f) {
        normal = -normal;
      }

      OceanExportVertex& out_normal = normals[static_cast<size_t>(z) * static_cast<size_t>(grid_resolution) + x];
      out_normal.x = normal.x;
      out_normal.y = normal.y;
      out_normal.z = normal.z;
    }
  }

  fprintf(file, "# ETX playground ocean export\n");
  fprintf(file, "# area_size_m %.6f\n", area_size_m);
  fprintf(file, "# grid_resolution %u\n", grid_resolution);
  fprintf(file, "# vertex_count %llu\n", static_cast<unsigned long long>(vertex_count_u64));
  fprintf(file, "# triangle_count %llu\n", static_cast<unsigned long long>(2u * static_cast<uint64_t>(grid_resolution - 1u) * static_cast<uint64_t>(grid_resolution - 1u)));

  for (uint64_t i = 0u; i < vertex_count_u64; ++i) {
    const OceanExportVertex& vertex = vertices[static_cast<size_t>(i)];
    fprintf(file, "v %.9g %.9g %.9g\n", vertex.x, vertex.y, vertex.z);
  }
  for (uint64_t i = 0u; i < vertex_count_u64; ++i) {
    const OceanExportVertex& normal = normals[static_cast<size_t>(i)];
    fprintf(file, "vn %.9g %.9g %.9g\n", normal.x, normal.y, normal.z);
  }

  for (uint32_t z = 0u; (z + 1u) < grid_resolution; ++z) {
    for (uint32_t x = 0u; (x + 1u) < grid_resolution; ++x) {
      uint32_t v00 = (z * grid_resolution) + x + 1u;
      uint32_t v10 = v00 + 1u;
      uint32_t v01 = ((z + 1u) * grid_resolution) + x + 1u;
      uint32_t v11 = v01 + 1u;
      fprintf(file, "f %u//%u %u//%u %u//%u\n", v00, v00, v01, v01, v11, v11);
      fprintf(file, "f %u//%u %u//%u %u//%u\n", v00, v00, v11, v11, v10, v10);
    }
  }

  fclose(file);
  return true;
}

struct PlaygroundUiChanges {
  bool sky_parameters_changed = false;
  bool msaa_sample_count_changed = false;
  uint32_t msaa_sample_count = 4u;
  bool patch_resolution_changed = false;
  uint32_t patch_resolution = 128u;
  bool fft_resolution_changed = false;
  uint32_t fft_resolution = 256u;
  bool simulation_paused_changed = false;
  bool simulation_paused = false;
  bool ocean_obj_export_size_changed = false;
  int32_t ocean_obj_export_size_m = k_ocean_obj_export_default_size_m;
  bool ocean_obj_export_requested = false;
};

static PlaygroundUiChanges draw_ocean_control_panel(Ocean& ocean, uint32_t current_msaa_sample_count, bool simulation_paused, bool ocean_obj_export_request_pending,
  bool ocean_obj_export_last_result_valid, bool ocean_obj_export_last_result_success, int32_t ocean_obj_export_size_m, const char* ocean_obj_export_status) {
  OceanParameters& parameters = ocean.parameters();
  bool spectrum_parameters_changed = false;
  bool sky_parameters_changed = false;
  PlaygroundUiChanges result = {};
  result.msaa_sample_count = current_msaa_sample_count;
  result.patch_resolution = ocean.patch_resolution();
  result.fft_resolution = ocean.fft_resolution();
  result.simulation_paused = simulation_paused;
  result.ocean_obj_export_size_m = ocean_obj_export_size_m;

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
    bool ui_simulation_paused = result.simulation_paused;
    if (ImGui::Checkbox("Pause simulation", &ui_simulation_paused)) {
      result.simulation_paused = ui_simulation_paused;
      result.simulation_paused_changed = true;
    }
    ImGui::TextUnformatted("OBJ export freezes simulation and captures the next frame.");
    if (ocean_obj_export_request_pending) {
      ImGui::TextUnformatted("OBJ export: scheduled for next frame");
    }
    int32_t export_size_m = result.ocean_obj_export_size_m;
    if (ImGui::InputInt("Export size (m)", &export_size_m)) {
      export_size_m = max(export_size_m, 1);
      result.ocean_obj_export_size_m = export_size_m;
      result.ocean_obj_export_size_changed = true;
    }
    ImGui::Text("OBJ export target: %d x %d m, %u x %u vertices (~%.2f M tris)", result.ocean_obj_export_size_m, result.ocean_obj_export_size_m, k_ocean_obj_export_grid_resolution,
      k_ocean_obj_export_grid_resolution,
      (2.0f * static_cast<float>(k_ocean_obj_export_grid_resolution - 1u) * static_cast<float>(k_ocean_obj_export_grid_resolution - 1u)) / 1000000.0f);
    if (ImGui::Button("Export Water Mesh OBJ (Next Frame)")) {
      result.ocean_obj_export_requested = true;
      result.simulation_paused = true;
      result.simulation_paused_changed = true;
    }
    if ((ocean_obj_export_last_result_valid) && (ocean_obj_export_status != nullptr) && (ocean_obj_export_status[0] != '\0')) {
      if (ocean_obj_export_last_result_success) {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.45f, 1.0f), "Last export: %s", ocean_obj_export_status);
      } else {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "Last export failed: %s", ocean_obj_export_status);
      }
    }
    ImGui::Separator();
    ImGui::SliderFloat("Time scale", &parameters.time_scale, 0.0f, 4.0f, "%.3f");
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
    if (ImGui::SliderFloat("Spectrum water depth (m)", &parameters.water_depth, 1.0f, 2000.0f, "%.3f")) {
      spectrum_parameters_changed = true;
    }
    if (ImGui::SliderFloat("JONSWAP gamma", &parameters.jonswap_gamma, 1.0f, 8.0f, "%.3f")) {
      spectrum_parameters_changed = true;
    }
    if (ImGui::SliderFloat("Directional spread", &parameters.directional_spread, 0.0f, 64.0f, "%.3f")) {
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
    ImGui::DragFloat("Choppiness", &parameters.choppiness, 0.01f, 0.0f, 0.0f, "%.3f");
    if (ImGui::DragFloat("Cascade detail boost", &parameters.cascade_detail_boost, 0.01f, 0.1f, 8.0f, "%.3f")) {
      spectrum_parameters_changed = true;
    }
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
    if (parameters.significant_wave_height_enable) {
      ImGui::TextUnformatted("Per-cascade RMS is solved from spectrum-band energy.");
    } else {
      ImGui::TextUnformatted("Height slider is absolute RMS in meters.");
    }
    for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
      ImGui::PushID(static_cast<int>(i));
      ImGui::Text("Cascade %u", i);
      if (ImGui::SliderFloat("Length (m)", &parameters.cascade_lengths[i], k_cascade_length_min[i], k_cascade_length_max, "%.3f")) {
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
    ImGui::TextUnformatted("Render weights are combined additively.");
    ImGui::BeginDisabled(parameters.debug_cascade_overrides_enable == false);
    for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
      ImGui::PushID(static_cast<int>(i + 100));
      bool cascade_enable = parameters.cascade_enable[i];
      if (ImGui::Checkbox("Enabled", &cascade_enable)) {
        parameters.cascade_enable[i] = cascade_enable;
      }
      ImGui::DragFloat("Render weight", &parameters.cascade_render_weight[i], 0.01f, 0.0f, 32.0f, "%.3f");
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
    ImGui::SliderFloat("Wave thickness path scale", &parameters.wave_thickness_path_scale, 0.0f, 4.0f, "%.3f");
    bool foam_enable = parameters.foam_enable;
    if (ImGui::Checkbox("Enable aeration / foam", &foam_enable)) {
      parameters.foam_enable = foam_enable;
    }
    ImGui::SliderFloat("Foam strength", &parameters.foam_strength, 0.0f, 4.0f, "%.3f");
    ImGui::SliderFloat("Foam slope start", &parameters.foam_slope_start, 0.0f, 16.0f, "%.3f");
    ImGui::SliderFloat("Foam slope end", &parameters.foam_slope_end, 0.0f, 32.0f, "%.3f");
    ImGui::SliderFloat("Foam thickness start (m)", &parameters.foam_thickness_start_m, 0.0f, 2.0f, "%.3f");
    ImGui::SliderFloat("Foam thickness end (m)", &parameters.foam_thickness_end_m, 0.0f, 4.0f, "%.3f");
    ImGui::SliderFloat("Foam surface coverage", &parameters.foam_surface_coverage, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Foam spec suppression", &parameters.foam_specular_suppression, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Foam diffuse gain", &parameters.foam_diffuse_gain, 0.0f, 4.0f, "%.3f");
    ImGui::SliderFloat("Foam backlight gain", &parameters.foam_backlight_gain, 0.0f, 4.0f, "%.3f");
    ImGui::SliderFloat("Foam aeration scatter", &parameters.foam_aeration_scatter_scale, 0.0f, 32.0f, "%.3f");
    ImGui::SliderFloat("Foam aeration absorb", &parameters.foam_aeration_absorption_scale, 0.0f, 4.0f, "%.3f");
    ImGui::ColorEdit3("Foam albedo", &parameters.foam_albedo.x);
    ImGui::SliderFloat("Foam detail scale 1", &parameters.foam_detail_scale_1, 0.01f, 2.0f, "%.4f");
    ImGui::SliderFloat("Foam detail scale 2", &parameters.foam_detail_scale_2, 0.01f, 4.0f, "%.4f");
    ImGui::SliderFloat("Foam detail mix", &parameters.foam_detail_mix, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Foam detail contrast", &parameters.foam_detail_contrast, 0.0f, 4.0f, "%.3f");
    ImGui::SliderFloat("Foam alpha threshold", &parameters.foam_alpha_threshold, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Foam detail scroll", &parameters.foam_detail_scroll_speed, 0.0f, 2.0f, "%.3f");
    bool sun_lighting_enable = parameters.sun_lighting_enable;
    if (ImGui::Checkbox("Enable sun lighting", &sun_lighting_enable)) {
      parameters.sun_lighting_enable = sun_lighting_enable;
    }
    float sun_azimuth_degrees = sun_azimuth_degrees_from_direction(parameters.sun_direction);
    if (ImGui::SliderFloat("Sun azimuth (deg)", &sun_azimuth_degrees, -180.0f, 180.0f, "%.2f")) {
      parameters.sun_direction = direction_with_sun_azimuth_degrees(parameters.sun_direction, sun_azimuth_degrees);
      sky_parameters_changed = true;
    }
    float sun_elevation_degrees = sun_elevation_degrees_from_direction(parameters.sun_direction);
    if (ImGui::SliderFloat("Sun elevation (deg)", &sun_elevation_degrees, -10.0f, 89.0f, "%.2f")) {
      parameters.sun_direction = direction_with_sun_elevation_degrees(parameters.sun_direction, sun_elevation_degrees);
      sky_parameters_changed = true;
    }
    ImGui::Text("Sun spectrum: normalized black body %.0f K", k_sun_temperature_kelvin);
    if (ImGui::SliderFloat("Sun brightness", &parameters.sun_brightness, 0.0f, 64.0f, "%.3f")) {
      sync_ocean_sun_spectrum(parameters);
      sky_parameters_changed = true;
    }
    ImGui::Text("Derived sun RGB: %.3f %.3f %.3f", parameters.sun_radiance.x, parameters.sun_radiance.y, parameters.sun_radiance.z);
  }

  ImGui::SetNextItemOpen(false, ImGuiCond_FirstUseEver);
  if (ImGui::CollapsingHeader("Rendering Debug")) {
    const char* patch_resolution_items[] = {"64", "128", "256"};
    uint32_t current_patch_resolution = ocean.patch_resolution();
    int patch_resolution_index = (current_patch_resolution == 64u) ? 0 : ((current_patch_resolution == 256u) ? 2 : 1);
    if (ImGui::Combo("Patch resolution", &patch_resolution_index, patch_resolution_items, 3)) {
      result.patch_resolution = (patch_resolution_index == 0) ? 64u : ((patch_resolution_index == 1) ? 128u : 256u);
      result.patch_resolution_changed = (result.patch_resolution != current_patch_resolution);
    }
    ImGui::TextUnformatted("Changing patch resolution recreates ocean mesh/resources.");

    const char* fft_resolution_items[] = {"128", "256", "512"};
    uint32_t current_fft_resolution = ocean.fft_resolution();
    int fft_resolution_index = (current_fft_resolution == 128u) ? 0 : ((current_fft_resolution == 512u) ? 2 : 1);
    if (ImGui::Combo("FFT resolution", &fft_resolution_index, fft_resolution_items, 3)) {
      result.fft_resolution = (fft_resolution_index == 0) ? 128u : ((fft_resolution_index == 1) ? 256u : 512u);
      result.fft_resolution_changed = (result.fft_resolution != current_fft_resolution);
    }
    ImGui::TextUnformatted("Changing FFT resolution recreates ocean sim textures and can be expensive.");

    const char* msaa_items[] = {"1x (Off)", "2x", "4x"};
    int msaa_index = (current_msaa_sample_count == 1u) ? 0 : ((current_msaa_sample_count == 2u) ? 1 : 2);
    if (ImGui::Combo("MSAA samples", &msaa_index, msaa_items, 3)) {
      result.msaa_sample_count = (msaa_index == 0) ? 1u : ((msaa_index == 1) ? 2u : 4u);
      result.msaa_sample_count_changed = (result.msaa_sample_count != current_msaa_sample_count);
    }
    ImGui::TextUnformatted("Changing MSAA recreates scene render targets and graphics pipelines.");
    bool lock_lods = parameters.lock_lods;
    if (ImGui::Checkbox("Lock LODs", &lock_lods)) {
      parameters.lock_lods = lock_lods;
    }
    ImGui::SliderFloat("Transition cells (LOD)", &parameters.stitch_transition_cells, 1.0f, 16.0f, "%.3f");
    ImGui::SliderFloat("Mip color mix", &parameters.mip_color_mix, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Mip color enable", &parameters.mip_color_enable, 0.0f, 1.0f, "%.3f");
    const char* surface_normal_visualize_items[] = {"Off", "Combined", "Cascade 0", "Cascade 1", "Cascade 2", "Ref error", "Mip drift", "Foldover risk"};
    int surface_normal_visualize_mode = parameters.surface_normal_visualize_mode;
    if (ImGui::Combo("Surface normal view", &surface_normal_visualize_mode, surface_normal_visualize_items, 8)) {
      parameters.surface_normal_visualize_mode = max(0, min(surface_normal_visualize_mode, 7));
    }
    const char* water_debug_visualize_items[] = {"Off", "Wave thickness", "Foam mask"};
    int water_debug_visualize_mode = parameters.water_debug_visualize_mode;
    if (ImGui::Combo("Water debug view", &water_debug_visualize_mode, water_debug_visualize_items, 3)) {
      parameters.water_debug_visualize_mode = max(0, min(water_debug_visualize_mode, 2));
    }
    ImGui::SliderFloat("Wave thickness debug max (m)", &parameters.wave_thickness_debug_max_m, 0.05f, 20.0f, "%.3f");
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
  result.sky_parameters_changed = sky_parameters_changed;
  return result;
}

static void draw_performance_overlay(float frame_time_ms, float fps, uint32_t msaa_sample_count) {
  ImGuiWindowFlags flags =
    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
  const ImGuiIO& io = ImGui::GetIO();
  ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 12.0f, 12.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
  ImGui::SetNextWindowBgAlpha(0.75f);
  if (ImGui::Begin("Performance", nullptr, flags)) {
    ImGui::Text("FPS: %.1f", fps);
    ImGui::Text("Frame: %.2f ms", frame_time_ms);
    ImGui::Text("MSAA: %ux", msaa_sample_count);
    ImGui::TextUnformatted("V-Sync: ON");
  }
  ImGui::End();
}

void PlaygroundApp::poll_gpu_timing_results() {
  _gpu_timing_last_poll_not_ready = false;

  if (_gpu_timing_supported == false) {
    return;
  }

  while (_gpu_timing_pending_count > 0u) {
    GpuTimingPendingFrame& pending = _gpu_timing_pending_frames[_gpu_timing_pending_read_index];
    if (pending.valid == false) {
      _gpu_timing_pending_read_index = (_gpu_timing_pending_read_index + 1u) % k_gpu_timing_pending_frame_count;
      _gpu_timing_pending_count -= 1u;
      continue;
    }

    uint64_t timestamp_ticks[k_gpu_timing_query_count] = {};
    RHIResult read_result = _rhi.read_timestamps(pending.command_buffer, 0u, k_gpu_timing_query_count, timestamp_ticks);
    if (read_result == RHIResult::NotReady) {
      _gpu_timing_last_poll_not_ready = true;
      break;
    }

    pending.valid = false;
    if (pending.command_buffer.valid()) {
      // Keep Vulkan command buffer/query-pool destruction tied to the frame pool cleanup after fence wait.
      _rhi.command_buffer_reset(pending.command_buffer);
      pending.command_buffer = {};
    }
    _gpu_timing_pending_read_index = (_gpu_timing_pending_read_index + 1u) % k_gpu_timing_pending_frame_count;
    _gpu_timing_pending_count -= 1u;

    if (read_result != RHIResult::Success) {
      continue;
    }

    for (uint32_t i = 0u; i < k_gpu_timing_query_count; ++i) {
      _gpu_timing_last_ticks[i] = timestamp_ticks[i];
    }

    const double tick_to_ms = _gpu_timestamp_period_ns * 1.0e-6;
    for (uint32_t i = 0u; i < k_gpu_timing_segment_count; ++i) {
      uint64_t begin_tick = timestamp_ticks[i];
      uint64_t end_tick = timestamp_ticks[i + 1u];
      uint64_t delta_tick = (end_tick >= begin_tick) ? (end_tick - begin_tick) : 0u;
      _gpu_timing_last_segment_ms[i] = static_cast<float>(static_cast<double>(delta_tick) * tick_to_ms);
    }

    uint64_t total_begin_tick = timestamp_ticks[k_gpu_timing_query_frame_begin];
    uint64_t total_end_tick = timestamp_ticks[k_gpu_timing_query_frame_end];
    uint64_t total_delta_tick = (total_end_tick >= total_begin_tick) ? (total_end_tick - total_begin_tick) : 0u;
    _gpu_timing_last_total_ms = static_cast<float>(static_cast<double>(total_delta_tick) * tick_to_ms);
    _gpu_timing_last_ready_frame_index = pending.frame_index;
    _gpu_timing_have_results = true;

    break;
  }
}

void PlaygroundApp::enqueue_gpu_timing_request(RHICommandBuffer cmd) {
  if (_gpu_timing_supported == false) {
    return;
  }

  GpuTimingPendingFrame& pending = _gpu_timing_pending_frames[_gpu_timing_pending_write_index];
  if (pending.valid) {
    return;
  }

  pending.valid = true;
  pending.command_buffer = cmd;
  pending.frame_index = _gpu_timing_frame_counter;

  _gpu_timing_pending_write_index = (_gpu_timing_pending_write_index + 1u) % k_gpu_timing_pending_frame_count;
  if (_gpu_timing_pending_count < k_gpu_timing_pending_frame_count) {
    _gpu_timing_pending_count += 1u;
  }

  _gpu_timing_frame_counter += 1u;
}

void PlaygroundApp::draw_gpu_timing_window() {
  ImGui::SetNextWindowPos(ImVec2(12.0f, 744.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(420.0f, 310.0f), ImGuiCond_FirstUseEver);
  if (ImGui::Begin("GPU Timings")) {
    if (_gpu_timing_supported == false) {
      ImGui::TextUnformatted("GPU timestamps are not supported by the active RHI/backend.");
      ImGui::End();
      return;
    }

    uint64_t result_age = 0u;
    if ((_gpu_timing_have_results) && (_gpu_timing_frame_counter >= _gpu_timing_last_ready_frame_index)) {
      result_age = _gpu_timing_frame_counter - _gpu_timing_last_ready_frame_index;
    }

    ImGui::Text("Timestamp period: %.3f ns", _gpu_timestamp_period_ns);
    ImGui::Text("Pending GPU frames: %u", _gpu_timing_pending_count);
    ImGui::Text("Latest ready frame age: %llu", static_cast<unsigned long long>(result_age));
    ImGui::Text("Polling status: %s", (_gpu_timing_last_poll_not_ready ? "waiting (non-blocking)" : "ready/idle"));
    ImGui::Separator();

    if (_gpu_timing_have_results == false) {
      ImGui::TextUnformatted("No GPU timing results yet.");
      ImGui::End();
      return;
    }

    static const char* k_segment_labels[k_gpu_timing_segment_count] = {
      "Frame Start -> Sun/Sky Regen",
      "Sun/Sky Regen -> Ocean Update",
      "Ocean Update -> Resource Prep",
      "Resource Prep -> Opaque Pass End",
      "Opaque Pass End -> Opaque Resolve End",
      "Opaque Resolve End -> Scene Pass End",
      "Scene Pass End -> Scene Resolve End",
      "Scene Resolve End -> Tonemap End",
      "Tonemap End -> ImGui Render End",
      "ImGui Render End -> Final Pass End",
      "Final Pass End -> Frame End Marker",
    };

    ImGui::Text("GPU Total (markers): %.3f ms", _gpu_timing_last_total_ms);
    ImGui::Separator();
    for (uint32_t i = 0u; i < k_gpu_timing_segment_count; ++i) {
      ImGui::Text("%s: %.3f ms", k_segment_labels[i], _gpu_timing_last_segment_ms[i]);
    }
  }
  ImGui::End();
}

void PlaygroundApp::destroy_ocean_obj_export_buffers() {
  for (uint32_t i = 0u; i < Ocean::k_cascade_count; ++i) {
    if (_ocean_obj_export.displacement_readback_buffers[i].valid()) {
      _rhi.device().destroy_buffer(_ocean_obj_export.displacement_readback_buffers[i]);
      _ocean_obj_export.displacement_readback_buffers[i] = {};
    }
  }
  _ocean_obj_export.capture_recorded = false;
  _ocean_obj_export.fft_resolution = 0u;
  _ocean_obj_export.pending_output_path.clear();
}

bool PlaygroundApp::record_ocean_obj_export_capture(RHICommandBuffer cmd) {
  _ocean_obj_export.last_result_valid = false;
  _ocean_obj_export.last_result_success = false;
  _ocean_obj_export.last_output_path.clear();
  _ocean_obj_export.last_error.clear();

  destroy_ocean_obj_export_buffers();

  if (_ocean.valid() == false) {
    _ocean_obj_export.last_result_valid = true;
    _ocean_obj_export.last_result_success = false;
    _ocean_obj_export.last_error = "Ocean renderer is not initialized";
    return false;
  }
  if (cmd.valid() == false) {
    _ocean_obj_export.last_result_valid = true;
    _ocean_obj_export.last_result_success = false;
    _ocean_obj_export.last_error = "Invalid command buffer for export capture";
    return false;
  }

  _ocean_obj_export.area_size_m = static_cast<float>(max(_ocean_obj_export_size_m, 1));
  _ocean_obj_export.grid_resolution = k_ocean_obj_export_grid_resolution;
  _ocean_obj_export.center = _camera.position;
  _ocean_obj_export.fft_resolution = _ocean.fft_resolution();
  if (_ocean_obj_export.fft_resolution == 0u) {
    _ocean_obj_export.last_result_valid = true;
    _ocean_obj_export.last_result_success = false;
    _ocean_obj_export.last_error = "FFT resolution is zero";
    return false;
  }

  const OceanParameters& ocean_parameters = _ocean.parameters();
  for (uint32_t i = 0u; i < Ocean::k_cascade_count; ++i) {
    _ocean_obj_export.cascade_lengths[i] = ocean_parameters.cascade_lengths[i];
    _ocean_obj_export.cascade_weights[i] = 0.0f;
  }
  _ocean.effective_cascade_weights(_ocean_obj_export.cascade_weights);

  char relative_path_buffer[256] = {};
  uint32_t export_serial = _ocean_obj_export.serial + 1u;
  _ocean_obj_export.serial = export_serial;
  snprintf(relative_path_buffer, sizeof(relative_path_buffer), "ocean_exports/ocean_surface_%04u.obj", export_serial);

  char output_path_buffer[2048] = {};
  env().file_in_tmp(relative_path_buffer, output_path_buffer, sizeof(output_path_buffer));
  _ocean_obj_export.pending_output_path = output_path_buffer;

  uint64_t texel_count = static_cast<uint64_t>(_ocean_obj_export.fft_resolution) * static_cast<uint64_t>(_ocean_obj_export.fft_resolution);
  uint64_t readback_buffer_size = texel_count * sizeof(OceanExportDisplacementTexel);
  RHIBufferDesc readback_buffer_desc = {};
  readback_buffer_desc.size = readback_buffer_size;
  readback_buffer_desc.usage = RHIBufferUsage::TransferDst;
  readback_buffer_desc.host_visible = true;

  for (uint32_t i = 0u; i < Ocean::k_cascade_count; ++i) {
    RHITexture displacement_texture = _ocean.displacement_texture(i);
    if (displacement_texture.valid() == false) {
      _ocean_obj_export.last_result_valid = true;
      _ocean_obj_export.last_result_success = false;
      _ocean_obj_export.last_error = "Missing ocean displacement texture";
      destroy_ocean_obj_export_buffers();
      return false;
    }

    auto create_result = _rhi.device().create_buffer(readback_buffer_desc);
    if (create_result.result != RHIResult::Success) {
      _ocean_obj_export.last_result_valid = true;
      _ocean_obj_export.last_result_success = false;
      _ocean_obj_export.last_error = "Failed to allocate readback buffer";
      destroy_ocean_obj_export_buffers();
      return false;
    }
    _ocean_obj_export.displacement_readback_buffers[i] = create_result.handle;

    _rhi.cmd_texture_barrier(cmd, displacement_texture, RHIResourceState::ShaderReadOnly, RHIResourceState::TransferSrc);
    _rhi.cmd_copy_texture_to_buffer(cmd, displacement_texture, _ocean_obj_export.displacement_readback_buffers[i], _ocean_obj_export.fft_resolution,
      _ocean_obj_export.fft_resolution, 0u);
    _rhi.cmd_texture_barrier(cmd, displacement_texture, RHIResourceState::TransferSrc, RHIResourceState::ShaderReadOnly);
  }

  _ocean_obj_export.capture_recorded = true;
  log::info("Ocean OBJ export scheduled: %s", _ocean_obj_export.pending_output_path.c_str());
  return true;
}

void PlaygroundApp::finalize_ocean_obj_export_capture() {
  if (_ocean_obj_export.capture_recorded == false) {
    return;
  }

  RHIResult wait_result = _rhi.wait_idle();
  if (wait_result != RHIResult::Success) {
    _ocean_obj_export.last_result_valid = true;
    _ocean_obj_export.last_result_success = false;
    _ocean_obj_export.last_error = "GPU wait_idle failed before export readback";
    destroy_ocean_obj_export_buffers();
    return;
  }

  if (_ocean_obj_export.fft_resolution == 0u) {
    _ocean_obj_export.last_result_valid = true;
    _ocean_obj_export.last_result_success = false;
    _ocean_obj_export.last_error = "Invalid FFT resolution for export";
    destroy_ocean_obj_export_buffers();
    return;
  }

  const uint64_t texel_count = static_cast<uint64_t>(_ocean_obj_export.fft_resolution) * static_cast<uint64_t>(_ocean_obj_export.fft_resolution);
  const uint64_t readback_size = texel_count * sizeof(OceanExportDisplacementTexel);
  OceanExportTextureData texture_data[Ocean::k_cascade_count] = {};

  for (uint32_t i = 0u; i < Ocean::k_cascade_count; ++i) {
    if (_ocean_obj_export.displacement_readback_buffers[i].valid() == false) {
      _ocean_obj_export.last_result_valid = true;
      _ocean_obj_export.last_result_success = false;
      _ocean_obj_export.last_error = "Missing readback buffer during export finalize";
      destroy_ocean_obj_export_buffers();
      return;
    }

    texture_data[i].width = _ocean_obj_export.fft_resolution;
    texture_data[i].height = _ocean_obj_export.fft_resolution;
    texture_data[i].texels.resize(static_cast<size_t>(texel_count));

    RHIResult read_result = _rhi.device().read_buffer(_ocean_obj_export.displacement_readback_buffers[i], texture_data[i].texels.data(), readback_size, 0u);
    if (read_result != RHIResult::Success) {
      _ocean_obj_export.last_result_valid = true;
      _ocean_obj_export.last_result_success = false;
      _ocean_obj_export.last_error = "Failed to read displacement buffer";
      destroy_ocean_obj_export_buffers();
      return;
    }
  }

  std::filesystem::path output_path = _ocean_obj_export.pending_output_path;
  std::error_code filesystem_error = {};
  std::filesystem::path output_parent = output_path.parent_path();
  if (output_parent.empty() == false) {
    std::filesystem::create_directories(output_parent, filesystem_error);
    if (filesystem_error.value() != 0) {
      _ocean_obj_export.last_result_valid = true;
      _ocean_obj_export.last_result_success = false;
      _ocean_obj_export.last_error = "Failed to create output directory";
      destroy_ocean_obj_export_buffers();
      return;
    }
  }

  std::string write_error = {};
  bool write_success = write_ocean_surface_obj_file(_ocean_obj_export.pending_output_path.c_str(), texture_data, _ocean_obj_export.cascade_lengths,
    _ocean_obj_export.cascade_weights, _ocean_obj_export.area_size_m, _ocean_obj_export.grid_resolution, _ocean_obj_export.center, write_error);
  _ocean_obj_export.last_result_valid = true;
  _ocean_obj_export.last_result_success = write_success;

  if (write_success) {
    _ocean_obj_export.last_output_path = _ocean_obj_export.pending_output_path;
    _ocean_obj_export.last_error.clear();
    log::info("Ocean OBJ export completed: %s", env().to_project_relative(_ocean_obj_export.last_output_path).c_str());
  } else {
    _ocean_obj_export.last_output_path.clear();
    _ocean_obj_export.last_error = write_error;
    log::error("Ocean OBJ export failed: %s", write_error.c_str());
  }

  destroy_ocean_obj_export_buffers();
}

void PlaygroundApp::recreate_scene_targets(uint32_t width, uint32_t height) {
  if ((width == 0u) || (height == 0u)) {
    return;
  }

  _render_width = width;
  _render_height = height;
  _scene_opaque_color_initialized = false;
  _scene_color_initialized = false;
  _scene_opaque_color_msaa_initialized = false;
  _scene_color_msaa_initialized = false;

  if (_depth_buffer.valid()) {
    _rhi.device().destroy_texture(_depth_buffer);
    _depth_buffer = {};
  }
  if (_depth_msaa_buffer.valid()) {
    _rhi.device().destroy_texture(_depth_msaa_buffer);
    _depth_msaa_buffer = {};
  }
  if (_scene_opaque_color_buffer.valid()) {
    _rhi.device().destroy_texture(_scene_opaque_color_buffer);
    _scene_opaque_color_buffer = {};
  }
  if (_scene_opaque_color_msaa_buffer.valid()) {
    _rhi.device().destroy_texture(_scene_opaque_color_msaa_buffer);
    _scene_opaque_color_msaa_buffer = {};
  }
  if (_scene_color_buffer.valid()) {
    _rhi.device().destroy_texture(_scene_color_buffer);
    _scene_color_buffer = {};
  }
  if (_scene_color_msaa_buffer.valid()) {
    _rhi.device().destroy_texture(_scene_color_msaa_buffer);
    _scene_color_msaa_buffer = {};
  }
  if (_ocean_wave_thickness_min_buffer.valid()) {
    _rhi.device().destroy_texture(_ocean_wave_thickness_min_buffer);
    _ocean_wave_thickness_min_buffer = {};
  }
  if (_ocean_wave_thickness_max_buffer.valid()) {
    _rhi.device().destroy_texture(_ocean_wave_thickness_max_buffer);
    _ocean_wave_thickness_max_buffer = {};
  }
  _ocean_wave_thickness_min_state = RHIResourceState::Undefined;
  _ocean_wave_thickness_max_state = RHIResourceState::Undefined;
  for (uint32_t i = 0u; i < 2u; ++i) {
    if (_ocean_foam_history_buffer[i].valid()) {
      _rhi.device().destroy_texture(_ocean_foam_history_buffer[i]);
      _ocean_foam_history_buffer[i] = {};
    }
    _ocean_foam_history_state[i] = RHIResourceState::Undefined;
  }
  _ocean_foam_history_write_index = 0u;

  RHITextureDesc scene_color_desc = {};
  scene_color_desc.width = width;
  scene_color_desc.height = height;
  scene_color_desc.format = k_scene_color_format;
  scene_color_desc.usage = RHITextureUsage::ColorAttachment | RHITextureUsage::Sampled | RHITextureUsage::TransferDst;
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

  RHITextureDesc ocean_wave_thickness_desc = {};
  ocean_wave_thickness_desc.width = width;
  ocean_wave_thickness_desc.height = height;
  ocean_wave_thickness_desc.format = RHITextureFormat::R32_FLOAT;
  ocean_wave_thickness_desc.usage = RHITextureUsage::ColorAttachment | RHITextureUsage::Sampled;
  auto wave_thickness_min_result = _rhi.device().create_texture(ocean_wave_thickness_desc);
  if (wave_thickness_min_result.result == RHIResult::Success) {
    _ocean_wave_thickness_min_buffer = wave_thickness_min_result.handle;
  } else {
    log::error("Failed to recreate ocean wave thickness MIN texture: %u", wave_thickness_min_result.result);
  }
  auto wave_thickness_max_result = _rhi.device().create_texture(ocean_wave_thickness_desc);
  if (wave_thickness_max_result.result == RHIResult::Success) {
    _ocean_wave_thickness_max_buffer = wave_thickness_max_result.handle;
  } else {
    log::error("Failed to recreate ocean wave thickness MAX texture: %u", wave_thickness_max_result.result);
  }

  RHITextureDesc ocean_foam_history_desc = {};
  ocean_foam_history_desc.width = width;
  ocean_foam_history_desc.height = height;
  ocean_foam_history_desc.format = RHITextureFormat::R8_UNORM;
  ocean_foam_history_desc.usage = RHITextureUsage::ColorAttachment | RHITextureUsage::Sampled;
  for (uint32_t i = 0u; i < 2u; ++i) {
    auto foam_history_result = _rhi.device().create_texture(ocean_foam_history_desc);
    if (foam_history_result.result == RHIResult::Success) {
      _ocean_foam_history_buffer[i] = foam_history_result.handle;
    } else {
      log::error("Failed to recreate ocean foam history texture[%u]: %u", i, foam_history_result.result);
    }
    _ocean_foam_history_state[i] = RHIResourceState::Undefined;
  }
  _ocean_foam_history_write_index = 0u;

  if (_scene_msaa_sample_count > 1u) {
    RHITextureDesc scene_color_msaa_desc = scene_color_desc;
    scene_color_msaa_desc.sample_count = _scene_msaa_sample_count;
    scene_color_msaa_desc.usage = RHITextureUsage::ColorAttachment | RHITextureUsage::TransferSrc;

    auto scene_opaque_color_msaa_result = _rhi.device().create_texture(scene_color_msaa_desc);
    if (scene_opaque_color_msaa_result.result == RHIResult::Success) {
      _scene_opaque_color_msaa_buffer = scene_opaque_color_msaa_result.handle;
    } else {
      log::error("Failed to recreate scene opaque HDR MSAA color texture: %u", scene_opaque_color_msaa_result.result);
    }

    auto scene_color_msaa_result = _rhi.device().create_texture(scene_color_msaa_desc);
    if (scene_color_msaa_result.result == RHIResult::Success) {
      _scene_color_msaa_buffer = scene_color_msaa_result.handle;
    } else {
      log::error("Failed to recreate scene final HDR MSAA color texture: %u", scene_color_msaa_result.result);
    }
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

  if (_scene_msaa_sample_count > 1u) {
    RHITextureDesc depth_msaa_desc = depth_desc;
    depth_msaa_desc.sample_count = _scene_msaa_sample_count;
    depth_msaa_desc.usage = RHITextureUsage::DepthAttachment;
    auto depth_msaa_result = _rhi.device().create_texture(depth_msaa_desc);
    if (depth_msaa_result.result == RHIResult::Success) {
      _depth_msaa_buffer = depth_msaa_result.handle;
    } else {
      log::error("Failed to recreate MSAA depth texture: %u", depth_msaa_result.result);
    }
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
  const bool missing_ocean_wave_thickness_min = (_ocean_wave_thickness_min_buffer.valid() == false);
  const bool missing_ocean_wave_thickness_max = (_ocean_wave_thickness_max_buffer.valid() == false);
  const bool missing_ocean_foam_history_0 = (_ocean_foam_history_buffer[0].valid() == false);
  const bool missing_ocean_foam_history_1 = (_ocean_foam_history_buffer[1].valid() == false);
  const bool msaa_enabled = (_scene_msaa_sample_count > 1u);
  const bool missing_scene_opaque_color_msaa = msaa_enabled && (_scene_opaque_color_msaa_buffer.valid() == false);
  const bool missing_scene_color_msaa = msaa_enabled && (_scene_color_msaa_buffer.valid() == false);
  const bool missing_depth_msaa = msaa_enabled && (_depth_msaa_buffer.valid() == false);

  if ((extent_changed == false) && (missing_scene_opaque_color == false) && (missing_scene_color == false) && (missing_depth == false) &&
      (missing_ocean_wave_thickness_min == false) && (missing_ocean_wave_thickness_max == false) && (missing_ocean_foam_history_0 == false) &&
      (missing_ocean_foam_history_1 == false) && (missing_scene_opaque_color_msaa == false) && (missing_scene_color_msaa == false) && (missing_depth_msaa == false)) {
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
  auto tonemap_compilation = ShaderCompiler::instance().compile(tonemap_shader_source, {vs, ps}, {}, _rhi.backend());
  if (tonemap_compilation.result == RHIResult::Success) {
    RHIGraphicsPipelineDesc tonemap_desc = {};
    tonemap_desc.vertex_shader.stage = RHIShaderStage::Vertex;
    tonemap_desc.vertex_shader.entry_point = "VSMain";
    tonemap_desc.vertex_shader.spirv_data = tonemap_compilation.binaries[0].spirv_data;
    tonemap_desc.vertex_shader.spirv_size = tonemap_compilation.binaries[0].spirv_size;
    tonemap_desc.vertex_shader.backend = tonemap_compilation.binaries[0].backend;
    tonemap_desc.vertex_shader.format = tonemap_compilation.binaries[0].format;

    tonemap_desc.fragment_shader.stage = RHIShaderStage::Fragment;
    tonemap_desc.fragment_shader.entry_point = "PSMain";
    tonemap_desc.fragment_shader.spirv_data = tonemap_compilation.binaries[1].spirv_data;
    tonemap_desc.fragment_shader.spirv_size = tonemap_compilation.binaries[1].spirv_size;
    tonemap_desc.fragment_shader.backend = tonemap_compilation.binaries[1].backend;
    tonemap_desc.fragment_shader.format = tonemap_compilation.binaries[1].format;

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

  if (_headless == false) {
    RHIImGuiDesc imgui_desc = {
      .color_format = current_swapchain_format,
      .depth_format = k_scene_depth_format,
    };
    RHIResult imgui_result = _imgui.setup(_rhi, imgui_desc);
    if (imgui_result != RHIResult::Success) {
      log::error("Failed to (re)create ImGui resources for swapchain format: %u", static_cast<uint32_t>(imgui_result));
    }
  }

  recreate_tonemap_pipeline();
  _swapchain_color_format = current_swapchain_format;
}

bool PlaygroundApp::ensure_sky_scattering_context() {
  if (_sky_scattering.initialized) {
    return true;
  }

  if (scattering::gpu_init(_rhi, _sky_scattering) == false) {
    log::warning("Playground atmosphere startup: failed to initialize GPU atmosphere path");
    return false;
  }

  if (scattering::gpu_precompute_optical_depth(_rhi, _sky_scattering) == false) {
    log::warning("Playground atmosphere startup: failed to precompute GPU optical depth");
    scattering::gpu_cleanup(_rhi, _sky_scattering);
    return false;
  }

  return true;
}

bool PlaygroundApp::create_sun_sky_textures() {
  if (ensure_sky_scattering_context() == false) {
    return false;
  }

  if (_generated_sky_envmap_texture.valid()) {
    _rhi.device().destroy_texture(_generated_sky_envmap_texture);
    _generated_sky_envmap_texture = {};
    _generated_sky_envmap_texture_state = RHIResourceState::Undefined;
  }
  if (_generated_sun_texture.valid()) {
    _rhi.device().destroy_texture(_generated_sun_texture);
    _generated_sun_texture = {};
    _generated_sun_texture_state = RHIResourceState::Undefined;
  }

  RHITextureDesc sky_desc = {};
  sky_desc.width = k_sky_envmap_dimensions.x;
  sky_desc.height = k_sky_envmap_dimensions.y;
  sky_desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
  sky_desc.usage = RHITextureUsage::Sampled | RHITextureUsage::Storage | RHITextureUsage::TransferSrc;
  auto sky_result = _rhi.device().create_texture(sky_desc);
  if ((sky_result.result != RHIResult::Success) || (sky_result.handle.valid() == false)) {
    log::warning("Playground atmosphere startup: failed to create generated sky texture (%u)", static_cast<uint32_t>(sky_result.result));
    return false;
  }
  _generated_sky_envmap_texture = sky_result.handle;
  _generated_sky_envmap_texture_state = RHIResourceState::Undefined;

  RHITextureDesc sun_desc = {};
  sun_desc.width = k_sun_sprite_dimensions.x;
  sun_desc.height = k_sun_sprite_dimensions.y;
  sun_desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
  sun_desc.usage = RHITextureUsage::Sampled | RHITextureUsage::Storage | RHITextureUsage::TransferSrc;
  auto sun_result = _rhi.device().create_texture(sun_desc);
  if ((sun_result.result != RHIResult::Success) || (sun_result.handle.valid() == false)) {
    log::warning("Playground atmosphere startup: failed to create generated sun texture (%u)", static_cast<uint32_t>(sun_result.result));
    _rhi.device().destroy_texture(_generated_sky_envmap_texture);
    _generated_sky_envmap_texture = {};
    _generated_sky_envmap_texture_state = RHIResourceState::Undefined;
    return false;
  }
  _generated_sun_texture = sun_result.handle;
  _generated_sun_texture_state = RHIResourceState::Undefined;

  return true;
}

bool PlaygroundApp::record_regenerate_sun_sky_textures(RHICommandBuffer cmd) {
  if (ensure_sky_scattering_context() == false) {
    return false;
  }
  if ((_generated_sky_envmap_texture.valid() == false) || (_generated_sun_texture.valid() == false)) {
    if (create_sun_sky_textures() == false) {
      return false;
    }
  }

  OceanParameters& ocean_parameters = _ocean.parameters();
  ocean_parameters.sun_direction = normalize(ocean_parameters.sun_direction);
  SpectralDistribution sun_emission_spectrum = sync_ocean_sun_spectrum(ocean_parameters);

  scattering::Parameters sky_parameters = {};
  std::vector<scattering::LightSource> sky_light_sources;
  sky_light_sources.push_back({
    sun_emission_spectrum,
    ocean_parameters.sun_direction,
    k_sun_angular_size_radians,
    1.0f,
  });

  if (scattering::gpu_record_generate_sky_raw(_rhi, cmd, _sky_scattering, sky_parameters, k_sky_envmap_dimensions, sky_light_sources, _generated_sky_envmap_texture,
        _generated_sky_envmap_texture_state) == false) {
    return false;
  }
  if (scattering::gpu_record_generate_sun(_rhi, cmd, _sky_scattering, sky_parameters, k_sun_sprite_dimensions, ocean_parameters.sun_direction, k_sun_angular_size_radians,
        _generated_sun_texture, _generated_sun_texture_state) == false) {
    return false;
  }

  if (_envmap.valid()) {
    _envmap.set_texture_state(_generated_sky_envmap_texture_state);
  }

  return true;
}

bool PlaygroundApp::recreate_sun_sprite_pipeline() {
  if (_sun_sprite_pipeline.valid()) {
    _rhi.device().destroy_pipeline(_sun_sprite_pipeline);
    _sun_sprite_pipeline = {};
  }

  ShaderCompiler::ShaderEntryPoint vs = {"VSMain", RHIShaderStage::Vertex};
  ShaderCompiler::ShaderEntryPoint ps = {"PSMain", RHIShaderStage::Fragment};
  std::string shader_source = env().file_in_data("playground/shaders/sun_sprite.hlsl");
  auto compilation = ShaderCompiler::instance().compile(shader_source, {vs, ps}, {}, _rhi.backend());
  if (compilation.result != RHIResult::Success) {
    log::error("Failed to compile sun sprite shader:\n%s", compilation.error_message.c_str());
    return false;
  }

  RHIGraphicsPipelineDesc p_desc = {};
  p_desc.vertex_shader.stage = RHIShaderStage::Vertex;
  p_desc.vertex_shader.entry_point = "VSMain";
  p_desc.vertex_shader.spirv_data = compilation.binaries[0].spirv_data;
  p_desc.vertex_shader.spirv_size = compilation.binaries[0].spirv_size;
  p_desc.vertex_shader.backend = compilation.binaries[0].backend;
  p_desc.vertex_shader.format = compilation.binaries[0].format;
  p_desc.fragment_shader.stage = RHIShaderStage::Fragment;
  p_desc.fragment_shader.entry_point = "PSMain";
  p_desc.fragment_shader.spirv_data = compilation.binaries[1].spirv_data;
  p_desc.fragment_shader.spirv_size = compilation.binaries[1].spirv_size;
  p_desc.fragment_shader.backend = compilation.binaries[1].backend;
  p_desc.fragment_shader.format = compilation.binaries[1].format;
  p_desc.depth_state.depth_test_enable = false;
  p_desc.depth_state.depth_write_enable = false;
  p_desc.blend.blend_enable = true;
  p_desc.blend.src_color_blend_factor = RHIBlendFactor::One;
  p_desc.blend.dst_color_blend_factor = RHIBlendFactor::One;
  p_desc.blend.color_blend_op = RHIBlendOp::Add;
  p_desc.blend.src_alpha_blend_factor = RHIBlendFactor::One;
  p_desc.blend.dst_alpha_blend_factor = RHIBlendFactor::One;
  p_desc.blend.alpha_blend_op = RHIBlendOp::Add;
  p_desc.primitive_topology = RHIPrimitiveTopology::TriangleList;
  p_desc.color_attachment_count = 1;
  p_desc.color_formats[0] = k_scene_color_format;
  p_desc.depth_format = k_scene_depth_format;
  p_desc.sample_count = _scene_msaa_sample_count;

  auto pipeline_result = _rhi.device().create_graphics_pipeline(p_desc);
  if (pipeline_result.result != RHIResult::Success) {
    log::error("Failed to create sun sprite pipeline: %u", static_cast<uint32_t>(pipeline_result.result));
    return false;
  }

  _sun_sprite_pipeline = pipeline_result.handle;
  return true;
}

bool PlaygroundApp::recreate_base_pipeline() {
  if (_pipeline.valid()) {
    _rhi.device().destroy_pipeline(_pipeline);
    _pipeline = {};
  }

  std::string shader_source = env().file_in_data("playground/shaders/base.hlsl");
  ShaderCompiler::ShaderEntryPoint vs = {"VSMain", RHIShaderStage::Vertex};
  ShaderCompiler::ShaderEntryPoint ps = {"PSMain", RHIShaderStage::Fragment};
  auto compilation = ShaderCompiler::instance().compile(shader_source, {vs, ps}, {}, _rhi.backend());

  if (compilation.result != RHIResult::Success) {
    log::error("Failed to compile shaders:\n%s", compilation.error_message.c_str());
    return false;
  }

  RHIGraphicsPipelineDesc p_desc = {};
  p_desc.vertex_shader.stage = RHIShaderStage::Vertex;
  p_desc.vertex_shader.entry_point = "VSMain";
  p_desc.vertex_shader.spirv_data = compilation.binaries[0].spirv_data;
  p_desc.vertex_shader.spirv_size = compilation.binaries[0].spirv_size;
  p_desc.vertex_shader.backend = compilation.binaries[0].backend;
  p_desc.vertex_shader.format = compilation.binaries[0].format;

  p_desc.fragment_shader.stage = RHIShaderStage::Fragment;
  p_desc.fragment_shader.entry_point = "PSMain";
  p_desc.fragment_shader.spirv_data = compilation.binaries[1].spirv_data;
  p_desc.fragment_shader.spirv_size = compilation.binaries[1].spirv_size;
  p_desc.fragment_shader.backend = compilation.binaries[1].backend;
  p_desc.fragment_shader.format = compilation.binaries[1].format;

  p_desc.rasterization.depth_clamp_enable = false;
  p_desc.rasterization.rasterizer_discard_enable = false;
  p_desc.depth_state.depth_test_enable = true;
  p_desc.depth_state.depth_write_enable = true;
  p_desc.blend.blend_enable = false;
  p_desc.primitive_topology = RHIPrimitiveTopology::TriangleList;
  p_desc.color_attachment_count = 1;
  p_desc.color_formats[0] = k_scene_color_format;
  p_desc.depth_format = k_scene_depth_format;
  p_desc.sample_count = _scene_msaa_sample_count;

  auto p_result = _rhi.device().create_graphics_pipeline(p_desc);
  if (p_result.result != RHIResult::Success) {
    log::error("Failed to create graphics pipeline: %u", p_result.result);
    return false;
  }

  _pipeline = p_result.handle;
  return true;
}

bool PlaygroundApp::apply_scene_msaa_sample_count(uint32_t sample_count) {
  if (_scene_msaa_sample_count == sample_count) {
    _scene_msaa_recreate_requested = false;
    _pending_scene_msaa_sample_count = sample_count;
    return true;
  }

  if ((sample_count != 1u) && (sample_count != 2u) && (sample_count != 4u)) {
    log::warning("Playground: unsupported MSAA sample count request %u (supported UI values: 1, 2, 4)", sample_count);
    return false;
  }

  if (_rhi.wait_idle() != RHIResult::Success) {
    log::warning("Playground: wait_idle failed before MSAA sample count recreation");
  }

  for (uint32_t i = 0u; i < k_gpu_timing_pending_frame_count; ++i) {
    GpuTimingPendingFrame& pending = _gpu_timing_pending_frames[i];
    if (pending.valid && pending.command_buffer.valid()) {
      _rhi.command_buffer_reset(pending.command_buffer);
      _rhi.destroy_command_buffer(pending.command_buffer);
    }
    pending.valid = false;
    pending.command_buffer = {};
    pending.frame_index = 0u;
  }
  _gpu_timing_pending_count = 0u;
  _gpu_timing_pending_read_index = 0u;
  _gpu_timing_pending_write_index = 0u;

  _scene_msaa_sample_count = sample_count;
  _pending_scene_msaa_sample_count = sample_count;
  _scene_msaa_recreate_requested = false;
  log::info("Playground: applying MSAA sample count = %u", _scene_msaa_sample_count);

  const RHIExtent2D swapchain_extent = _rhi.get_swapchain_extent();
  if ((swapchain_extent.width > 0u) && (swapchain_extent.height > 0u)) {
    recreate_scene_targets(swapchain_extent.width, swapchain_extent.height);
  }

  bool base_pipeline_ok = recreate_base_pipeline();
  bool sun_pipeline_ok = recreate_sun_sprite_pipeline();

  bool envmap_ok = false;
  if (_generated_sky_envmap_texture.valid()) {
    envmap_ok = _envmap.setup_with_texture(_rhi, k_scene_color_format, k_scene_depth_format, _generated_sky_envmap_texture, _generated_sky_envmap_texture_state, true,
      _scene_msaa_sample_count);
    if (envmap_ok == false) {
      log::warning("Playground: failed to recreate generated-sky envmap pipeline for MSAA=%u, falling back to HDR envmap", _scene_msaa_sample_count);
    }
  }
  if (envmap_ok == false) {
    envmap_ok = _envmap.setup(_rhi, k_scene_color_format, k_scene_depth_format, _scene_msaa_sample_count);
  }

  _ocean.cleanup(_rhi);
  _ocean.init(_rhi, k_scene_color_format, k_scene_depth_format, _scene_msaa_sample_count);
  sync_ocean_sun_spectrum(_ocean.parameters());

  if ((base_pipeline_ok == false) || (sun_pipeline_ok == false) || (envmap_ok == false)) {
    log::warning("Playground: one or more resources failed to recreate after MSAA sample count change");
  }
  log::info("Playground: MSAA sample count applied = %u (HDR format = R16G16B16A16_FLOAT)", _scene_msaa_sample_count);

  return (base_pipeline_ok && sun_pipeline_ok && envmap_ok);
}

bool PlaygroundApp::apply_ocean_patch_resolution(uint32_t patch_resolution) {
  if (_ocean.patch_resolution() == patch_resolution) {
    _pending_ocean_patch_resolution = patch_resolution;
    _ocean_patch_resolution_recreate_requested = false;
    return true;
  }

  if ((patch_resolution != 64u) && (patch_resolution != 128u) && (patch_resolution != 256u)) {
    log::warning("Playground: unsupported ocean patch resolution request %u (supported UI values: 64, 128, 256)", patch_resolution);
    return false;
  }

  if (_rhi.wait_idle() != RHIResult::Success) {
    log::warning("Playground: wait_idle failed before ocean patch resolution recreation");
  }

  for (uint32_t i = 0u; i < k_gpu_timing_pending_frame_count; ++i) {
    GpuTimingPendingFrame& pending = _gpu_timing_pending_frames[i];
    if ((pending.valid) && pending.command_buffer.valid()) {
      _rhi.command_buffer_reset(pending.command_buffer);
      _rhi.destroy_command_buffer(pending.command_buffer);
    }
    pending.valid = false;
    pending.command_buffer = {};
    pending.frame_index = 0u;
  }
  _gpu_timing_pending_count = 0u;
  _gpu_timing_pending_read_index = 0u;
  _gpu_timing_pending_write_index = 0u;

  _pending_ocean_patch_resolution = patch_resolution;
  _ocean_patch_resolution_recreate_requested = false;

  _ocean.cleanup(_rhi);
  _ocean.set_patch_resolution(patch_resolution);
  _ocean.init(_rhi, k_scene_color_format, k_scene_depth_format, _scene_msaa_sample_count);
  sync_ocean_sun_spectrum(_ocean.parameters());

  log::info("Playground: ocean patch resolution applied = %u", patch_resolution);
  return true;
}

bool PlaygroundApp::apply_ocean_fft_resolution(uint32_t fft_resolution) {
  if (_ocean.fft_resolution() == fft_resolution) {
    _pending_ocean_fft_resolution = fft_resolution;
    _ocean_fft_resolution_recreate_requested = false;
    return true;
  }

  if ((fft_resolution != 128u) && (fft_resolution != 256u) && (fft_resolution != 512u)) {
    log::warning("Playground: unsupported ocean FFT resolution request %u (supported UI values: 128, 256, 512)", fft_resolution);
    return false;
  }

  if (_rhi.wait_idle() != RHIResult::Success) {
    log::warning("Playground: wait_idle failed before ocean FFT resolution recreation");
  }

  for (uint32_t i = 0u; i < k_gpu_timing_pending_frame_count; ++i) {
    GpuTimingPendingFrame& pending = _gpu_timing_pending_frames[i];
    if ((pending.valid) && pending.command_buffer.valid()) {
      _rhi.command_buffer_reset(pending.command_buffer);
      _rhi.destroy_command_buffer(pending.command_buffer);
    }
    pending.valid = false;
    pending.command_buffer = {};
    pending.frame_index = 0u;
  }
  _gpu_timing_pending_count = 0u;
  _gpu_timing_pending_read_index = 0u;
  _gpu_timing_pending_write_index = 0u;

  _pending_ocean_fft_resolution = fft_resolution;
  _ocean_fft_resolution_recreate_requested = false;

  _ocean.cleanup(_rhi);
  _ocean.set_fft_resolution(fft_resolution);
  _ocean.init(_rhi, k_scene_color_format, k_scene_depth_format, _scene_msaa_sample_count);
  sync_ocean_sun_spectrum(_ocean.parameters());

  log::info("Playground: ocean FFT resolution applied = %u", fft_resolution);
  return true;
}

void PlaygroundApp::draw_sun_sprite(RHICommandBuffer cmd, const float4x4& view_proj, uint32_t render_width, uint32_t render_height, float vertical_fov_radians) {
  if (_sun_sprite_pipeline.valid() == false) {
    return;
  }
  if (_generated_sun_texture.valid() == false) {
    return;
  }
  if (_ocean.parameters().sun_lighting_enable == false) {
    return;
  }
  if ((render_width == 0u) || (render_height == 0u)) {
    return;
  }

  const float3 sun_direction = normalize(_ocean.parameters().sun_direction);
  const float3 sun_world_position = _camera.position + (sun_direction * 1024.0f);
  const float4 sun_clip = view_proj * float4{sun_world_position.x, sun_world_position.y, sun_world_position.z, 1.0f};
  if (sun_clip.w <= 1.0e-6f) {
    return;
  }

  const float2 center_ndc = {sun_clip.x / sun_clip.w, sun_clip.y / sun_clip.w};
  if ((fabsf(center_ndc.x) > 1.5f) || (fabsf(center_ndc.y) > 1.5f)) {
    return;
  }

  const float aspect = float(render_width) / float(render_height);
  if (aspect <= 0.0f) {
    return;
  }

  const float half_height_ndc = tanf(0.5f * k_sun_angular_size_radians) / tanf(0.5f * vertical_fov_radians);
  const float half_width_ndc = half_height_ndc / aspect;

  struct PushConstants {
    uint32_t texture_index;
    uint32_t sampler_index;
    float2 center_ndc;
    float2 half_size_ndc;
    float2 pad0;
    float4 tint;
  };

  PushConstants pc = {};
  pc.texture_index = get_bindless_descriptor_index(_generated_sun_texture);
  pc.sampler_index = _rhi.get_sampler_index(RHISamplerType::LinearClamp);
  pc.center_ndc = center_ndc;
  pc.half_size_ndc = {half_width_ndc, half_height_ndc};
  float sun_solid_angle = sun_disk_solid_angle_sr(k_sun_angular_size_radians);
  float sun_disk_radiance_scale = 1.0f / sun_solid_angle;
  pc.tint = {_ocean.parameters().sun_radiance.x * sun_disk_radiance_scale, _ocean.parameters().sun_radiance.y * sun_disk_radiance_scale,
    _ocean.parameters().sun_radiance.z * sun_disk_radiance_scale, 1.0f};

  _rhi.cmd_set_pipeline(cmd, _sun_sprite_pipeline);
  _rhi.cmd_push_constants(cmd, &pc, sizeof(pc));
  _rhi.cmd_draw(cmd, {.vertex_count = 6, .instance_count = 1});
}

void PlaygroundApp::init_internal(uint32_t width, uint32_t height, const void* native_window, bool headless) {
  _headless = headless;
  _width = width;
  _height = height;

  RHIInitInfo info = {
    .backend = select_default_backend(),
    .enable_validation = true,
    .headless = headless,
  };

  _rhi = RHIContext::create(info);
  if (_rhi.valid() == false) {
    log::error("Playground: failed to create RHI context");
    return;
  }

  if (_headless) {
    _rhi.initialize_headless();
    _rhi.resize_swapchain(_width, _height);
  } else {
    _rhi.create_swapchain(native_window, _width, _height);
  }

  const RHICapabilities capabilities = _rhi.capabilities();
  log::info("Playground RHI backend: %s (swapchain=%u, bindless=%u, timestamps=%u, ray_tracing=%u)", backend_name(info.backend),
    static_cast<uint32_t>(capabilities.supports_swapchain), static_cast<uint32_t>(capabilities.supports_bindless), static_cast<uint32_t>(capabilities.supports_timestamps),
    static_cast<uint32_t>(capabilities.supports_ray_tracing));
  _scene_msaa_sample_count = default_scene_msaa_sample_count(info.backend);
  _pending_scene_msaa_sample_count = _scene_msaa_sample_count;
  _scene_msaa_recreate_requested = false;
  _pending_ocean_patch_resolution = _ocean.patch_resolution();
  _ocean_patch_resolution_recreate_requested = false;
  _pending_ocean_fft_resolution = _ocean.fft_resolution();
  _ocean_fft_resolution_recreate_requested = false;
  _gpu_timing_supported = _rhi.supports_timestamps();
  if (_gpu_timing_supported) {
    _gpu_timestamp_period_ns = _rhi.timestamp_period_ns();
  } else {
    _gpu_timestamp_period_ns = 0.0;
  }

  ShaderCompiler::instance().initialize();
  sync_swapchain_dependent_resources();
  if (_headless) {
    recreate_headless_present_target(_width, _height);
  }

  bool generated_sky_envmap_valid = create_sun_sky_textures();
  if (generated_sky_envmap_valid == false) {
    log::warning("Playground atmosphere startup: failed to create GPU sun/sky textures");
  }

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
    0,
    1,
    2,
    2,
    1,
    3,  // +X
    4,
    5,
    6,
    6,
    5,
    7,  // -X
    8,
    9,
    10,
    10,
    9,
    11,  // +Y
    12,
    13,
    14,
    14,
    13,
    15,  // -Y
    16,
    17,
    18,
    18,
    17,
    19,  // +Z
    20,
    21,
    22,
    22,
    21,
    23,  // -Z
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

  if (generated_sky_envmap_valid) {
    if (_envmap.setup_with_texture(_rhi, k_scene_color_format, k_scene_depth_format, _generated_sky_envmap_texture, _generated_sky_envmap_texture_state, true,
          _scene_msaa_sample_count) == false) {
      log::warning("Playground atmosphere startup: failed to create envmap from generated sky, falling back to HDR envmap");
      if (_generated_sky_envmap_texture.valid()) {
        _rhi.device().destroy_texture(_generated_sky_envmap_texture);
        _generated_sky_envmap_texture = {};
        _generated_sky_envmap_texture_state = RHIResourceState::Undefined;
      }
      _envmap.setup(_rhi, k_scene_color_format, k_scene_depth_format, _scene_msaa_sample_count);
    }
  } else {
    _envmap.setup(_rhi, k_scene_color_format, k_scene_depth_format, _scene_msaa_sample_count);
  }
  _ocean.init(_rhi, k_scene_color_format, k_scene_depth_format, _scene_msaa_sample_count);
  sync_ocean_sun_spectrum(_ocean.parameters());
  recreate_sun_sprite_pipeline();
  _sun_sky_dirty = generated_sky_envmap_valid;

  recreate_base_pipeline();

  uint32_t camera_width = (_render_width > 0u) ? _render_width : _width;
  uint32_t camera_height = (_render_height > 0u) ? _render_height : _height;
  build_camera(_camera, {5.0f, 5.0f, 5.0f}, normalize(float3{0.0f, 0.0f, 0.0f} - float3{5.0f, 5.0f, 5.0f}), kWorldUp, {camera_width, camera_height}, 45.0f);
}

void PlaygroundApp::init() {
  ETX_PROFILER_SCOPE();

  const uint32_t width = static_cast<uint32_t>(sapp_width());
  const uint32_t height = static_cast<uint32_t>(sapp_height());
  const void* native_window = nullptr;
#if ETX_PLATFORM_WINDOWS
  native_window = sapp_win32_get_hwnd();
#elif defined(__APPLE__)
  native_window = sapp_macos_get_window();
#endif
  init_internal(width, height, native_window, false);
}

void PlaygroundApp::init_headless(uint32_t width, uint32_t height) {
  ETX_PROFILER_SCOPE();
  init_internal(width, height, nullptr, true);
}

void PlaygroundApp::frame_headless(float delta_time, float dpi_scale) {
  _headless_frame_delta_time = delta_time;
  _headless_dpi_scale = dpi_scale;
  frame();
}

void PlaygroundApp::cleanup() {
  ETX_PROFILER_SCOPE();
  if (_rhi.valid() == false) {
    ShaderCompiler::instance().shutdown();
    return;
  }
  _rhi.wait_idle();
  destroy_ocean_obj_export_buffers();
  for (uint32_t i = 0u; i < k_gpu_timing_pending_frame_count; ++i) {
    GpuTimingPendingFrame& pending = _gpu_timing_pending_frames[i];
    if (pending.valid && pending.command_buffer.valid()) {
      _rhi.command_buffer_reset(pending.command_buffer);
      _rhi.destroy_command_buffer(pending.command_buffer);
    }
    pending.valid = false;
    pending.command_buffer = {};
    pending.frame_index = 0u;
  }
  _gpu_timing_pending_count = 0u;
  _gpu_timing_pending_read_index = 0u;
  _gpu_timing_pending_write_index = 0u;
  _imgui.shutdown();
  _envmap.cleanup(_rhi);
  if (_generated_sky_envmap_texture.valid()) {
    _rhi.device().destroy_texture(_generated_sky_envmap_texture);
    _generated_sky_envmap_texture = {};
    _generated_sky_envmap_texture_state = RHIResourceState::Undefined;
  }
  if (_generated_sun_texture.valid()) {
    _rhi.device().destroy_texture(_generated_sun_texture);
    _generated_sun_texture = {};
    _generated_sun_texture_state = RHIResourceState::Undefined;
  }
  if (_sky_scattering.initialized) {
    scattering::gpu_cleanup(_rhi, _sky_scattering);
  }
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
  if (_sun_sprite_pipeline.valid()) {
    _rhi.device().destroy_pipeline(_sun_sprite_pipeline);
    _sun_sprite_pipeline = {};
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
  if (_scene_opaque_color_msaa_buffer.valid()) {
    _rhi.device().destroy_texture(_scene_opaque_color_msaa_buffer);
  }
  if (_scene_color_buffer.valid()) {
    _rhi.device().destroy_texture(_scene_color_buffer);
  }
  if (_scene_color_msaa_buffer.valid()) {
    _rhi.device().destroy_texture(_scene_color_msaa_buffer);
  }
  if (_ocean_wave_thickness_min_buffer.valid()) {
    _rhi.device().destroy_texture(_ocean_wave_thickness_min_buffer);
    _ocean_wave_thickness_min_buffer = {};
    _ocean_wave_thickness_min_state = RHIResourceState::Undefined;
  }
  if (_ocean_wave_thickness_max_buffer.valid()) {
    _rhi.device().destroy_texture(_ocean_wave_thickness_max_buffer);
    _ocean_wave_thickness_max_buffer = {};
    _ocean_wave_thickness_max_state = RHIResourceState::Undefined;
  }
  for (uint32_t i = 0u; i < 2u; ++i) {
    if (_ocean_foam_history_buffer[i].valid()) {
      _rhi.device().destroy_texture(_ocean_foam_history_buffer[i]);
      _ocean_foam_history_buffer[i] = {};
    }
    _ocean_foam_history_state[i] = RHIResourceState::Undefined;
  }
  if (_depth_buffer.valid()) {
    _rhi.device().destroy_texture(_depth_buffer);
  }
  if (_depth_msaa_buffer.valid()) {
    _rhi.device().destroy_texture(_depth_msaa_buffer);
  }
  if (_headless_present_texture.valid()) {
    _rhi.device().destroy_texture(_headless_present_texture);
    _headless_present_texture = {};
  }
  _rhi.destroy_swapchain();
  ShaderCompiler::instance().shutdown();
}

void PlaygroundApp::frame() {
  ETX_PROFILER_SCOPE();
  if (_rhi.valid() == false) {
    return;
  }
  static_assert(k_gpu_timing_query_count <= 64u, "Playground GPU timing query count exceeds Vulkan command-buffer query pool capacity.");

  uint32_t w = _headless ? _width : static_cast<uint32_t>(sapp_width());
  uint32_t h = _headless ? _height : static_cast<uint32_t>(sapp_height());

  if ((w != _width) || (h != _height)) {
    _width = w;
    _height = h;
    _rhi.resize_swapchain(_width, _height);
    sync_scene_targets_to_swapchain_extent();
    if (_headless) {
      recreate_headless_present_target(_width, _height);
    }
  }

  _rhi.begin_frame();
  sync_swapchain_dependent_resources();
  sync_scene_targets_to_swapchain_extent();
  poll_gpu_timing_results();
  if (_scene_msaa_recreate_requested) {
    apply_scene_msaa_sample_count(_pending_scene_msaa_sample_count);
    sync_scene_targets_to_swapchain_extent();
  }
  if (_ocean_patch_resolution_recreate_requested) {
    apply_ocean_patch_resolution(_pending_ocean_patch_resolution);
  }
  if (_ocean_fft_resolution_recreate_requested) {
    apply_ocean_fft_resolution(_pending_ocean_fft_resolution);
  }

  const RHIExtent2D swapchain_extent = _rhi.get_swapchain_extent();
  if ((swapchain_extent.width == 0u) || (swapchain_extent.height == 0u)) {
    return;
  }

  const uint32_t render_width = (_render_width > 0u) ? _render_width : swapchain_extent.width;
  const uint32_t render_height = (_render_height > 0u) ? _render_height : swapchain_extent.height;
  RHITexture swapchain_texture = _headless ? _headless_present_texture : _rhi.get_current_swapchain_texture();
  if (swapchain_texture.valid() == false) {
    return;
  }

  RHICommandBuffer cmd = _rhi.get_command_buffer();
  _rhi.command_buffer_begin(cmd);

  bool record_ocean_obj_export_this_frame = false;
  if (_ocean_obj_export.request_next_frame) {
    _ocean_obj_export.request_next_frame = false;
    _simulation_paused = true;
    if (_ocean.valid()) {
      record_ocean_obj_export_this_frame = true;
    } else {
      _ocean_obj_export.last_result_valid = true;
      _ocean_obj_export.last_result_success = false;
      _ocean_obj_export.last_output_path.clear();
      _ocean_obj_export.last_error = "Ocean renderer is not initialized";
    }
  }

  auto write_gpu_timestamp = [&](uint32_t query_index) {
    if (_gpu_timing_supported) {
      _rhi.cmd_write_timestamp(cmd, query_index, RHITimestampStage::AllCommands);
    }
  };
  if (_gpu_timing_supported) {
    _rhi.cmd_reset_timestamps(cmd, 0u, k_gpu_timing_query_count);
    write_gpu_timestamp(k_gpu_timing_query_frame_begin);
  }

  if (_sun_sky_dirty) {
    bool regenerated = record_regenerate_sun_sky_textures(cmd);
    if (regenerated == false) {
      log::warning("Playground atmosphere: failed to regenerate sun/sky textures");
    }
    _sun_sky_dirty = false;
  }
  write_gpu_timestamp(k_gpu_timing_query_after_sun_sky_regen);

  float frame_delta_time = _headless ? _headless_frame_delta_time : static_cast<float>(sapp_frame_duration());
  if (_simulation_paused == false) {
    _time += frame_delta_time;
  }

  if (frame_delta_time > 0.0f) {
    _fps_counter_accumulated_time += frame_delta_time;
    _fps_counter_accumulated_frames += 1u;
  }
  if ((_fps_counter_accumulated_time >= 0.25f) && (_fps_counter_accumulated_frames > 0u)) {
    _fps_display = static_cast<float>(_fps_counter_accumulated_frames) / _fps_counter_accumulated_time;
    _frame_time_ms_display = (_fps_counter_accumulated_time * 1000.0f) / static_cast<float>(_fps_counter_accumulated_frames);
    _fps_counter_accumulated_time = 0.0f;
    _fps_counter_accumulated_frames = 0u;
  }

  _camera_controller.update(frame_delta_time);
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
  if (record_ocean_obj_export_this_frame) {
    record_ocean_obj_export_capture(cmd);
  }
  write_gpu_timestamp(k_gpu_timing_query_after_ocean_update);

  if (_envmap.valid()) {
    _envmap.prepare_texture_for_sampling(_rhi, cmd);
    if ((_generated_sky_envmap_texture.valid()) && (_envmap.texture() == _generated_sky_envmap_texture)) {
      _generated_sky_envmap_texture_state = _envmap.texture_state();
    }
  }
  if ((_generated_sun_texture.valid()) && (_generated_sun_texture_state != RHIResourceState::ShaderReadOnly)) {
    _rhi.cmd_texture_barrier(cmd, _generated_sun_texture, _generated_sun_texture_state, RHIResourceState::ShaderReadOnly);
    _generated_sun_texture_state = RHIResourceState::ShaderReadOnly;
  }
  write_gpu_timestamp(k_gpu_timing_query_after_resource_prep);

  const bool msaa_enabled = (_scene_msaa_sample_count > 1u);
  bool hdr_path_available = (_scene_opaque_color_buffer.valid() && _scene_color_buffer.valid() && _tonemap_pipeline.valid());
  if (msaa_enabled) {
    hdr_path_available = hdr_path_available && _scene_opaque_color_msaa_buffer.valid() && _scene_color_msaa_buffer.valid() && _depth_msaa_buffer.valid();
  }

  float scene_clear_color[4] = {0.1f, 0.2f, 0.3f, 1.0f};
  auto set_viewport_and_scissor = [&]() {
    _rhi.cmd_set_viewport(cmd, {0.0f, static_cast<float>(render_height), static_cast<float>(render_width), -static_cast<float>(render_height), 0.0f, 1.0f});
    _rhi.cmd_set_scissor(cmd, {0, 0, render_width, render_height});
  };

  auto draw_env_and_buoy = [&]() {
    if (_envmap.valid()) {
      set_viewport_and_scissor();
      _envmap.draw(_rhi, cmd, inv_view_proj);
      draw_sun_sprite(cmd, view_proj, render_width, render_height, kPi / 4.0f);
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

  auto render_ocean_wave_thickness_prepass = [&]() {
    if ((_ocean.valid() == false) || (_ocean_wave_thickness_min_buffer.valid() == false) || (_ocean_wave_thickness_max_buffer.valid() == false)) {
      return;
    }

    _rhi.cmd_texture_barrier(cmd, _ocean_wave_thickness_min_buffer, _ocean_wave_thickness_min_state, RHIResourceState::ColorAttachment);
    _ocean_wave_thickness_min_state = RHIResourceState::ColorAttachment;
    float wave_thickness_min_clear[4] = {kMaxFloat, 0.0f, 0.0f, 0.0f};
    RHIResourceState wave_thickness_min_final_state = RHIResourceState::ShaderReadOnly;
    _rhi.cmd_begin_render_pass(cmd, 1, &_ocean_wave_thickness_min_buffer, wave_thickness_min_clear, {}, &wave_thickness_min_final_state);
    set_viewport_and_scissor();
    _ocean.draw_wave_thickness_prepass(_rhi, cmd, view_proj, inv_view_proj, _camera.position, false, render_width, render_height);
    _rhi.cmd_end_render_pass(cmd);
    _ocean_wave_thickness_min_state = wave_thickness_min_final_state;

    _rhi.cmd_texture_barrier(cmd, _ocean_wave_thickness_max_buffer, _ocean_wave_thickness_max_state, RHIResourceState::ColorAttachment);
    _ocean_wave_thickness_max_state = RHIResourceState::ColorAttachment;
    float wave_thickness_max_clear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    RHIResourceState wave_thickness_max_final_state = RHIResourceState::ShaderReadOnly;
    _rhi.cmd_begin_render_pass(cmd, 1, &_ocean_wave_thickness_max_buffer, wave_thickness_max_clear, {}, &wave_thickness_max_final_state);
    set_viewport_and_scissor();
    _ocean.draw_wave_thickness_prepass(_rhi, cmd, view_proj, inv_view_proj, _camera.position, true, render_width, render_height);
    _rhi.cmd_end_render_pass(cmd);
    _ocean_wave_thickness_max_state = wave_thickness_max_final_state;
  };

  render_ocean_wave_thickness_prepass();

  RHITexture ocean_foam_history_shading_texture = {};

  if (hdr_path_available && msaa_enabled) {
    RHIResourceState opaque_msaa_old_state = _scene_opaque_color_msaa_initialized ? RHIResourceState::TransferSrc : RHIResourceState::Undefined;
    _rhi.cmd_texture_barrier(cmd, _scene_opaque_color_msaa_buffer, opaque_msaa_old_state, RHIResourceState::ColorAttachment);
    _scene_opaque_color_msaa_initialized = true;

    RHIResourceState opaque_resolve_old_state = _scene_opaque_color_initialized ? RHIResourceState::ShaderReadOnly : RHIResourceState::Undefined;
    _rhi.cmd_texture_barrier(cmd, _scene_opaque_color_buffer, opaque_resolve_old_state, RHIResourceState::TransferDst);

    RHIResourceState opaque_msaa_final_state = RHIResourceState::ColorAttachment;
    _rhi.cmd_begin_render_pass(cmd, 1, &_scene_opaque_color_msaa_buffer, scene_clear_color, _depth_msaa_buffer, &opaque_msaa_final_state, RHIResourceState::DepthAttachment);
    draw_env_and_buoy();
    _rhi.cmd_end_render_pass(cmd);
    write_gpu_timestamp(k_gpu_timing_query_after_opaque_pass);

    _rhi.cmd_texture_barrier(cmd, _scene_opaque_color_msaa_buffer, RHIResourceState::ColorAttachment, RHIResourceState::TransferSrc);
    _rhi.cmd_resolve_texture(cmd, _scene_opaque_color_msaa_buffer, _scene_opaque_color_buffer, render_width, render_height);
    _rhi.cmd_texture_barrier(cmd, _scene_opaque_color_buffer, RHIResourceState::TransferDst, RHIResourceState::ShaderReadOnly);
    _scene_opaque_color_initialized = true;
    write_gpu_timestamp(k_gpu_timing_query_after_opaque_resolve);

    RHIResourceState final_msaa_old_state = _scene_color_msaa_initialized ? RHIResourceState::TransferSrc : RHIResourceState::Undefined;
    _rhi.cmd_texture_barrier(cmd, _scene_color_msaa_buffer, final_msaa_old_state, RHIResourceState::ColorAttachment);
    _scene_color_msaa_initialized = true;

    RHIResourceState final_resolve_old_state = _scene_color_initialized ? RHIResourceState::ShaderReadOnly : RHIResourceState::Undefined;
    _rhi.cmd_texture_barrier(cmd, _scene_color_buffer, final_resolve_old_state, RHIResourceState::TransferDst);

    RHIResourceState final_msaa_scene_state = RHIResourceState::ColorAttachment;
    _rhi.cmd_begin_render_pass(cmd, 1, &_scene_color_msaa_buffer, scene_clear_color, _depth_msaa_buffer, &final_msaa_scene_state, RHIResourceState::DepthAttachment);
    draw_env_and_buoy();
    if (_ocean.valid()) {
      set_viewport_and_scissor();
      _ocean.draw(_rhi, cmd, view_proj, inv_view_proj, _camera.position, _envmap.texture(), _envmap.uses_equal_area_mapping(), _scene_opaque_color_buffer,
        _ocean_wave_thickness_min_buffer, _ocean_wave_thickness_max_buffer, ocean_foam_history_shading_texture, render_width, render_height);
    }
    _rhi.cmd_end_render_pass(cmd);
    write_gpu_timestamp(k_gpu_timing_query_after_scene_pass);

    _rhi.cmd_texture_barrier(cmd, _scene_color_msaa_buffer, RHIResourceState::ColorAttachment, RHIResourceState::TransferSrc);
    _rhi.cmd_resolve_texture(cmd, _scene_color_msaa_buffer, _scene_color_buffer, render_width, render_height);
    _rhi.cmd_texture_barrier(cmd, _scene_color_buffer, RHIResourceState::TransferDst, RHIResourceState::ShaderReadOnly);
    _scene_color_initialized = true;
    write_gpu_timestamp(k_gpu_timing_query_after_scene_resolve);
  } else if (hdr_path_available) {
    RHIResourceState opaque_old_state = _scene_opaque_color_initialized ? RHIResourceState::ShaderReadOnly : RHIResourceState::Undefined;
    _rhi.cmd_texture_barrier(cmd, _scene_opaque_color_buffer, opaque_old_state, RHIResourceState::ColorAttachment);
    _scene_opaque_color_initialized = true;
    RHIResourceState opaque_final_state = RHIResourceState::ShaderReadOnly;
    _rhi.cmd_begin_render_pass(cmd, 1, &_scene_opaque_color_buffer, scene_clear_color, _depth_buffer, &opaque_final_state, RHIResourceState::DepthAttachment);
    draw_env_and_buoy();
    _rhi.cmd_end_render_pass(cmd);
    write_gpu_timestamp(k_gpu_timing_query_after_opaque_pass);
    write_gpu_timestamp(k_gpu_timing_query_after_opaque_resolve);

    RHIResourceState final_old_state = _scene_color_initialized ? RHIResourceState::ShaderReadOnly : RHIResourceState::Undefined;
    _rhi.cmd_texture_barrier(cmd, _scene_color_buffer, final_old_state, RHIResourceState::ColorAttachment);
    _scene_color_initialized = true;
    RHIResourceState final_scene_state = RHIResourceState::ShaderReadOnly;
    _rhi.cmd_begin_render_pass(cmd, 1, &_scene_color_buffer, scene_clear_color, _depth_buffer, &final_scene_state, RHIResourceState::DepthAttachment);
    draw_env_and_buoy();
    if (_ocean.valid()) {
      set_viewport_and_scissor();
      _ocean.draw(_rhi, cmd, view_proj, inv_view_proj, _camera.position, _envmap.texture(), _envmap.uses_equal_area_mapping(), _scene_opaque_color_buffer,
        _ocean_wave_thickness_min_buffer, _ocean_wave_thickness_max_buffer, ocean_foam_history_shading_texture, render_width, render_height);
    }
    _rhi.cmd_end_render_pass(cmd);
    write_gpu_timestamp(k_gpu_timing_query_after_scene_pass);
    write_gpu_timestamp(k_gpu_timing_query_after_scene_resolve);
  } else {
    RHITexture scene_color_target = swapchain_texture;
    RHIResourceState scene_final_state = RHIResourceState::ColorAttachment;
    _rhi.cmd_begin_render_pass(cmd, 1, &scene_color_target, scene_clear_color, _depth_buffer, &scene_final_state, RHIResourceState::DepthAttachment);
    draw_env_and_buoy();
    if (_ocean.valid()) {
      set_viewport_and_scissor();
      _ocean.draw(_rhi, cmd, view_proj, inv_view_proj, _camera.position, _envmap.texture(), _envmap.uses_equal_area_mapping(), {}, _ocean_wave_thickness_min_buffer,
        _ocean_wave_thickness_max_buffer, ocean_foam_history_shading_texture, render_width, render_height);
    }
    write_gpu_timestamp(k_gpu_timing_query_after_opaque_pass);
    write_gpu_timestamp(k_gpu_timing_query_after_opaque_resolve);
    write_gpu_timestamp(k_gpu_timing_query_after_scene_pass);
    write_gpu_timestamp(k_gpu_timing_query_after_scene_resolve);
  }

  const bool imgui_available = (_headless == false) && _imgui.initialized();
  if (imgui_available) {
    RHIImGuiFrameDesc imgui_frame_desc = {
      .width = render_width,
      .height = render_height,
      .delta_time = _headless ? _headless_frame_delta_time : static_cast<float>(sapp_frame_duration()),
      .dpi_scale = _headless ? _headless_dpi_scale : sapp_dpi_scale(),
    };
    _imgui.new_frame(imgui_frame_desc);
    const char* ocean_obj_export_status = nullptr;
    if (_ocean_obj_export.last_result_valid) {
      if (_ocean_obj_export.last_result_success) {
        ocean_obj_export_status = _ocean_obj_export.last_output_path.c_str();
      } else {
        ocean_obj_export_status = _ocean_obj_export.last_error.c_str();
      }
    }
    PlaygroundUiChanges ui_changes = draw_ocean_control_panel(_ocean, _scene_msaa_sample_count, _simulation_paused, _ocean_obj_export.request_next_frame,
      _ocean_obj_export.last_result_valid, _ocean_obj_export.last_result_success, _ocean_obj_export_size_m, ocean_obj_export_status);
    if (ui_changes.sky_parameters_changed) {
      _sun_sky_dirty = true;
    }
    if (ui_changes.simulation_paused_changed) {
      _simulation_paused = ui_changes.simulation_paused;
    }
    if (ui_changes.ocean_obj_export_size_changed) {
      _ocean_obj_export_size_m = max(ui_changes.ocean_obj_export_size_m, 1);
    }
    if (ui_changes.ocean_obj_export_requested) {
      _ocean_obj_export.request_next_frame = true;
    }
    if (ui_changes.msaa_sample_count_changed) {
      _pending_scene_msaa_sample_count = ui_changes.msaa_sample_count;
      _scene_msaa_recreate_requested = true;
    }
    if (ui_changes.patch_resolution_changed) {
      _pending_ocean_patch_resolution = ui_changes.patch_resolution;
      _ocean_patch_resolution_recreate_requested = true;
    }
    if (ui_changes.fft_resolution_changed) {
      _pending_ocean_fft_resolution = ui_changes.fft_resolution;
      _ocean_fft_resolution_recreate_requested = true;
    }
    draw_performance_overlay(_frame_time_ms_display, _fps_display, _scene_msaa_sample_count);
    draw_gpu_timing_window();
  }

  if (hdr_path_available) {
    float present_clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    RHIResourceState swapchain_final_state = RHIResourceState::ColorAttachment;
    _rhi.cmd_begin_render_pass(cmd, 1, &swapchain_texture, present_clear_color, {}, &swapchain_final_state);

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
    write_gpu_timestamp(k_gpu_timing_query_after_tonemap);

    if (imgui_available) {
      _imgui.render(cmd);
    }
    write_gpu_timestamp(k_gpu_timing_query_after_imgui_render);
    _rhi.cmd_end_render_pass(cmd);
    write_gpu_timestamp(k_gpu_timing_query_after_final_pass);
  } else {
    write_gpu_timestamp(k_gpu_timing_query_after_tonemap);
    if (imgui_available) {
      _imgui.render(cmd);
    }
    write_gpu_timestamp(k_gpu_timing_query_after_imgui_render);
    _rhi.cmd_end_render_pass(cmd);
    write_gpu_timestamp(k_gpu_timing_query_after_final_pass);
  }

  write_gpu_timestamp(k_gpu_timing_query_frame_end);
  _rhi.command_buffer_end(cmd);

  _rhi.submit_frame_command_buffer(cmd);
  enqueue_gpu_timing_request(cmd);
  if (_ocean_obj_export.capture_recorded) {
    finalize_ocean_obj_export_capture();
  }
  if (_headless == false) {
    _rhi.present();
  }
}

void PlaygroundApp::process_event(const sapp_event* e) {
  ETX_PROFILER_SCOPE();
  const bool imgui_available = _imgui.initialized();
  bool event_handled = imgui_available && _imgui.handle_event(e);
  bool imgui_wants_mouse = imgui_available && ImGui::GetIO().WantCaptureMouse;
  bool imgui_wants_keyboard = imgui_available && ImGui::GetIO().WantCaptureKeyboard;
  if ((event_handled == false) && (imgui_wants_mouse == false) && (imgui_wants_keyboard == false)) {
    _camera_controller.handle_event(e);
  }
}

}  // namespace etx
