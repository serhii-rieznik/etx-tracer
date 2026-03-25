#include "ocean.hxx"

#include <etx/render/interop/geometry.hxx>
#include <etx/render/host/image_loaders.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/core.hxx>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>

namespace etx {

static constexpr uint32_t k_max_clipmap_levels = 16;
static constexpr uint32_t k_max_clipmap_instances = 16u + (12u * (k_max_clipmap_levels - 1u));
static constexpr uint32_t k_unculled_lod_levels = 1u;
static constexpr float k_frustum_cull_guard = 1.1f;
static constexpr float k_planar_patch_radius_scale = 1.41421356237f;
static constexpr float k_min_projected_cell_pixels = 5.0f;
static constexpr float k_max_wave_steepness = 0.85f;
static constexpr float k_surface_tension_over_density = 7.28e-5f;
static constexpr uint32_t k_cascade_rng_seed[Ocean::k_cascade_count] = {0x9e3779b9u, 0x7f4a7c15u, 0x94d049bbu};
static constexpr float k_cascade_length_ratio_min = 2.0f;
static constexpr float k_cascade_length_min[Ocean::k_cascade_count] = {20.0f, 10.0f, 5.0f};
static constexpr float k_cascade_length_max = 4000.0f;

static uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
  if (divisor == 0u) {
    return 0u;
  }
  return (value + divisor - 1u) / divisor;
}

static bool is_power_of_two_u32(uint32_t value) {
  if (value == 0u) {
    return false;
  }
  return ((value & (value - 1u)) == 0u);
}

static uint32_t integer_log2_u32(uint32_t value) {
  uint32_t result = 0u;
  while (value > 1u) {
    value >>= 1u;
    ++result;
  }
  return result;
}

static uint32_t calculate_mip_levels(uint32_t width, uint32_t height) {
  uint32_t mip_levels = 1u;
  while ((width > 1u) || (height > 1u)) {
    width = (width > 1u) ? (width / 2u) : 1u;
    height = (height > 1u) ? (height / 2u) : 1u;
    ++mip_levels;
  }
  return mip_levels;
}

static uint32_t hash_u32(uint32_t x) {
  x ^= (x >> 16u);
  x *= 0x7feb352du;
  x ^= (x >> 15u);
  x *= 0x846ca68bu;
  x ^= (x >> 16u);
  return x;
}

static float hash_to_unit_float(uint32_t x) {
  uint32_t bits = hash_u32(x);
  return static_cast<float>(bits & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

static void save_debug_texture_r8_pgm(const char* file_name, uint32_t width, uint32_t height, const uint8_t* pixels) {
  if ((file_name == nullptr) || (pixels == nullptr) || (width == 0u) || (height == 0u)) {
    return;
  }

  FILE* file = fopen(file_name, "wb");
  if (file == nullptr) {
    log::error("Failed to open debug texture file: %s", file_name);
    return;
  }

  int header_size = fprintf(file, "P5\n%u %u\n255\n", width, height);
  if (header_size <= 0) {
    fclose(file);
    log::error("Failed to write PGM header: %s", file_name);
    return;
  }

  size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
  size_t written = fwrite(pixels, 1u, pixel_count, file);
  fclose(file);
  if (written != pixel_count) {
    log::error("Failed to write PGM pixels: %s", file_name);
  }
}

static void save_debug_texture_preview_ppm(const char* file_name, uint32_t width, uint32_t height, const uint8_t* pixels) {
  if ((file_name == nullptr) || (pixels == nullptr) || (width == 0u) || (height == 0u)) {
    return;
  }

  std::vector<uint8_t> rgb_pixels(width * height * 3u, 255u);
  for (uint32_t i = 0u; i < (width * height); ++i) {
    uint8_t v = pixels[i];
    uint8_t rim = (v > 180u) ? 255u : 0u;
    rgb_pixels[(i * 3u) + 0u] = static_cast<uint8_t>(min(255u, static_cast<uint32_t>(v) + 30u));
    rgb_pixels[(i * 3u) + 1u] = static_cast<uint8_t>(min(255u, static_cast<uint32_t>(v) + 10u));
    rgb_pixels[(i * 3u) + 2u] = static_cast<uint8_t>(min(255u, (static_cast<uint32_t>(v) / 2u) + (static_cast<uint32_t>(rim) / 3u)));
  }

  FILE* file = fopen(file_name, "wb");
  if (file == nullptr) {
    log::error("Failed to open debug preview texture file: %s", file_name);
    return;
  }

  int header_size = fprintf(file, "P6\n%u %u\n255\n", width, height);
  if (header_size <= 0) {
    fclose(file);
    log::error("Failed to write PPM header: %s", file_name);
    return;
  }

  size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
  size_t written = fwrite(rgb_pixels.data(), 1u, pixel_count, file);
  fclose(file);
  if (written != pixel_count) {
    log::error("Failed to write PPM pixels: %s", file_name);
  }
}

static bool create_r8_texture_from_pixels(RHIContext& rhi, uint32_t width, uint32_t height, const uint8_t* pixels, RHITexture& out_texture, const char* debug_name) {
  if ((pixels == nullptr) || (width == 0u) || (height == 0u)) {
    log::error("Failed to create %s texture: invalid source pixels", (debug_name != nullptr) ? debug_name : "ocean detail");
    return false;
  }

  RHITextureDesc tex_desc = {};
  tex_desc.width = width;
  tex_desc.height = height;
  tex_desc.format = RHITextureFormat::R8_UNORM;
  tex_desc.usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst;
  auto tex_result = rhi.device().create_texture(tex_desc);
  if (tex_result.result != RHIResult::Success) {
    log::error("Failed to create %s texture", (debug_name != nullptr) ? debug_name : "ocean detail");
    return false;
  }

  out_texture = tex_result.handle;
  if (rhi.device().update_texture(out_texture, pixels) != RHIResult::Success) {
    log::error("Failed to upload %s texture", (debug_name != nullptr) ? debug_name : "ocean detail");
    rhi.device().destroy_texture(out_texture);
    out_texture = {};
    return false;
  }

  return true;
}

static bool file_exists_on_disk(const char* file_name) {
  if (file_name == nullptr) {
    return false;
  }

  FILE* file = fopen(file_name, "rb");
  if (file == nullptr) {
    return false;
  }
  fclose(file);
  return true;
}

static bool load_r8_texture_from_file(RHIContext& rhi, const char* project_relative_path, RHITexture& out_texture, const char* debug_name) {
  if (project_relative_path == nullptr) {
    log::error("Failed to load %s texture: invalid path", (debug_name != nullptr) ? debug_name : "ocean detail");
    return false;
  }

  std::vector<uint8_t> image_data;
  uint2 image_dims = {};
  Image::Format image_format = Image::Format::Undefined;
  std::string loaded_from_path;
  std::string candidate_paths[2] = {};
  uint32_t candidate_count = 0u;
  candidate_paths[candidate_count++] = project_relative_path;
  if (((strncmp(project_relative_path, "bin/", 4u) == 0) || (strncmp(project_relative_path, "bin\\", 4u) == 0)) && (candidate_count < 2u)) {
    candidate_paths[candidate_count++] = std::string(project_relative_path + 4);
  }

  for (uint32_t i = 0u; i < candidate_count; ++i) {
    std::string resolved_path = env().resolve_to_absolute(candidate_paths[i]);
    if (file_exists_on_disk(resolved_path.c_str()) == false) {
      continue;
    }
    image_format = load_data(resolved_path.c_str(), image_data, image_dims);
    if (image_format != Image::Format::Undefined) {
      loaded_from_path = candidate_paths[i];
      break;
    }
  }
  if (image_format == Image::Format::Undefined) {
    log::error("Failed to load %s texture from %s", (debug_name != nullptr) ? debug_name : "ocean detail", project_relative_path);
    return false;
  }
  if (image_format != Image::Format::RGBA8) {
    log::error("Unsupported %s texture format for %s (expected RGBA8-compatible image)", (debug_name != nullptr) ? debug_name : "ocean detail", loaded_from_path.c_str());
    return false;
  }
  if ((image_dims.x == 0u) || (image_dims.y == 0u)) {
    log::error("Invalid %s texture dimensions for %s", (debug_name != nullptr) ? debug_name : "ocean detail", loaded_from_path.c_str());
    return false;
  }

  size_t pixel_count = static_cast<size_t>(image_dims.x) * static_cast<size_t>(image_dims.y);
  if (image_data.size() < (pixel_count * 4u)) {
    log::error("Invalid %s texture data size for %s", (debug_name != nullptr) ? debug_name : "ocean detail", loaded_from_path.c_str());
    return false;
  }

  bool alpha_varies = false;
  for (size_t i = 0u; i < pixel_count; ++i) {
    if (image_data[(i * 4u) + 3u] != 255u) {
      alpha_varies = true;
      break;
    }
  }

  std::vector<uint8_t> r8_pixels(pixel_count, 0u);
  for (size_t i = 0u; i < pixel_count; ++i) {
    const uint8_t r = image_data[(i * 4u) + 0u];
    const uint8_t g = image_data[(i * 4u) + 1u];
    const uint8_t b = image_data[(i * 4u) + 2u];
    const uint8_t a = image_data[(i * 4u) + 3u];
    if (alpha_varies) {
      r8_pixels[i] = a;
    } else {
      uint32_t luma = (static_cast<uint32_t>(r) * 54u) + (static_cast<uint32_t>(g) * 183u) + (static_cast<uint32_t>(b) * 19u);
      r8_pixels[i] = static_cast<uint8_t>(min(luma >> 8u, 255u));
    }
  }

  return create_r8_texture_from_pixels(rhi, image_dims.x, image_dims.y, r8_pixels.data(), out_texture, debug_name);
}

struct ClipmapInstance {
  float x_offset;
  float z_offset;
  float scale;
  float level;
};

struct OceanRenderSettings {
  uint32_t envmap_index;
  uint32_t scene_color_index;
  uint32_t env_sampler_index;
  uint32_t scene_sampler_index;
  float stitch_transition_cells;
  float mip_color_mix;
  float mip_color_enable;
  float _padding0;
  float4 cascade_lengths;
  float4 cascade_weights;
  float4 clipmap_center_offset;
  float4 debug_view;
  float4 surface_normal_controls;
  float4 wireframe_color;
  float4 screen_size;
  float4x4 inv_view_proj;
  uint32_t surface_deriv_u_index[4];
  uint32_t surface_deriv_v_index[4];
  uint32_t slope_metric_index[4];
  float4 water_optics_0;
  float4 water_optics_1;
  float4 water_absorption;
  float4 water_scattering;
  float4 sun_direction_enable;
  float4 sun_radiance;
  float4x4 prev_view_proj;
  uint32_t wave_thickness_min_index;
  uint32_t wave_thickness_max_index;
  uint32_t wave_thickness_sampler_index;
  uint32_t _padding1;
  uint32_t foam_history_index;
  uint32_t foam_detail_index;
  uint32_t foam_sampler_index;
  uint32_t aeration_detail_index;
  float4 wave_thickness_controls;
  float4 foam_controls_0;
  float4 foam_controls_1;
  float4 foam_controls_2;
  float4 foam_color;
  float4 foam_temporal_controls;
  float4 foam_detail_controls;
  float4 foam_flow_controls;
};

static_assert(sizeof(OceanRenderSettings) == 576u, "OceanRenderSettings layout must match ocean.hlsl");

static bool patch_visible_in_frustum(const float3& patch_center, float patch_radius, const float3& camera_position, const float3& camera_right, const float3& camera_up,
  const float3& camera_forward, float tan_half_fov, float aspect, float cull_guard_scale) {
  float3 rel = patch_center - camera_position;
  float view_x = dot(rel, camera_right);
  float view_y = dot(rel, camera_up);
  float view_z = dot(rel, camera_forward);

  if (view_z < -patch_radius) {
    return false;
  }

  float depth = max(view_z + patch_radius, 0.0f);
  float frustum_half_h = depth * tan_half_fov * cull_guard_scale;
  float frustum_half_w = frustum_half_h * aspect;

  if (fabsf(view_x) > (frustum_half_w + patch_radius)) {
    return false;
  }
  if (fabsf(view_y) > (frustum_half_h + patch_radius)) {
    return false;
  }
  return true;
}

static uint32_t calculate_min_active_clipmap_level(float camera_height, float base_patch_size, uint32_t patch_resolution, float fov, uint32_t viewport_height,
  uint32_t clipmap_levels) {
  if ((clipmap_levels == 0u) || (patch_resolution == 0u) || (viewport_height == 0u)) {
    return 0u;
  }

  float tan_half_fov = tanf(0.5f * fov);
  if (tan_half_fov <= 1.0e-6f) {
    return 0u;
  }

  float base_cell_size = base_patch_size / static_cast<float>(patch_resolution);
  float surface_distance = max(fabsf(camera_height), 1.0e-3f);
  float focal_pixels_y = (0.5f * static_cast<float>(viewport_height)) / tan_half_fov;
  float cell_pixels = (base_cell_size * focal_pixels_y) / surface_distance;

  uint32_t min_active_level = 0u;
  while (((min_active_level + 1u) < clipmap_levels) && (cell_pixels < k_min_projected_cell_pixels)) {
    cell_pixels *= 2.0f;
    ++min_active_level;
  }

  return min_active_level;
}

static float dispersion_omega(float k_len, float water_depth) {
  if (k_len <= 1.0e-6f) {
    return 0.0f;
  }
  float safe_depth = max(water_depth, 1.0e-3f);
  float gravity = 9.81f;
  float kh = k_len * safe_depth;
  float restoring = (gravity * k_len) + (k_surface_tension_over_density * k_len * k_len * k_len);
  return sqrtf(max(restoring * tanhf(kh), 0.0f));
}

static float dispersion_domega_dk(float k_len, float water_depth, float omega) {
  if ((k_len <= 1.0e-6f) || (omega <= 1.0e-6f)) {
    return 0.0f;
  }
  float safe_depth = max(water_depth, 1.0e-3f);
  float gravity = 9.81f;
  float kh = k_len * safe_depth;
  float tanh_kh = tanhf(kh);
  float sech_kh = 1.0f / coshf(kh);
  float sech2_kh = sech_kh * sech_kh;
  float k2 = k_len * k_len;
  float restoring = (gravity * k_len) + (k_surface_tension_over_density * k_len * k2);
  float restoring_dk = gravity + (3.0f * k_surface_tension_over_density * k2);
  float omega_sq_dk = (restoring_dk * tanh_kh) + (restoring * safe_depth * sech2_kh);
  return 0.5f * omega_sq_dk / omega;
}

static float jonswap_frequency_spectrum(float omega, float wind_speed, float jonswap_gamma) {
  if (omega <= 1.0e-6f) {
    return 0.0f;
  }
  float safe_wind_speed = max(wind_speed, 0.1f);
  float gravity = 9.81f;
  float omega_p = (0.877f * gravity) / safe_wind_speed;
  float sigma = (omega <= omega_p) ? 0.07f : 0.09f;
  float alpha = 0.0081f;
  float omega_ratio = omega_p / omega;
  float omega_ratio4 = omega_ratio * omega_ratio * omega_ratio * omega_ratio;
  float omega5 = omega * omega * omega * omega * omega;
  float pm = alpha * gravity * gravity * expf(-1.25f * omega_ratio4) / max(omega5, 1.0e-9f);

  float delta = omega - omega_p;
  float sigma_omega_p = max(sigma * omega_p, 1.0e-6f);
  float exponent = -(delta * delta) / (2.0f * sigma_omega_p * sigma_omega_p);
  float gamma_peak = powf(max(jonswap_gamma, 1.0f), expf(exponent));
  return pm * gamma_peak;
}

static float directional_spreading_weight(const float2& k, const float2& wind_dir, float directional_spread) {
  float k_len = length(k);
  if (k_len <= 1.0e-6f) {
    return 0.0f;
  }
  float spread = max(directional_spread, 0.0f);
  float2 k_normalized = k / k_len;
  float k_dot_w = dot(k_normalized, wind_dir);
  float forward = powf(max(0.0f, k_dot_w), spread);
  float backward = 0.0 * powf(max(0.0f, -k_dot_w), spread);
  return forward + backward;
}

static float smooth_step(float edge0, float edge1, float x) {
  float delta = edge1 - edge0;
  if (fabsf(delta) <= 1.0e-8f) {
    return (x >= edge1) ? 1.0f : 0.0f;
  }
  float t = (x - edge0) / delta;
  t = min(1.0f, max(0.0f, t));
  return t * t * (3.0f - (2.0f * t));
}

static float ocean_wave_spectrum(const float2& k, const float2& wind_dir, float wind_speed, float water_depth, float jonswap_gamma, float directional_spread,
  float amplitude_gain) {
  float k_len = length(k);
  if (k_len <= 1.0e-6f) {
    return 0.0f;
  }

  float omega = dispersion_omega(k_len, water_depth);
  if (omega <= 1.0e-6f) {
    return 0.0f;
  }
  float d_omega_d_k = dispersion_domega_dk(k_len, water_depth, omega);
  if (d_omega_d_k <= 0.0f) {
    return 0.0f;
  }

  float s_omega = jonswap_frequency_spectrum(omega, wind_speed, jonswap_gamma);
  if (s_omega <= 0.0f) {
    return 0.0f;
  }

  float directional = directional_spreading_weight(k, wind_dir, directional_spread);
  if (directional <= 0.0f) {
    return 0.0f;
  }

  float kh = k_len * max(water_depth, 1.0e-3f);
  float depth_attenuation = tanhf(kh);
  depth_attenuation *= depth_attenuation;

  float spectral_density_k = s_omega * (d_omega_d_k / max(k_len, 1.0e-6f)) * directional * depth_attenuation;
  float gravity = 9.81f;
  float omega_p = (0.877f * gravity) / max(wind_speed, 0.1f);
  float omega_ratio = omega / max(omega_p, 1.0e-6f);
  float capillary_gate = smooth_step(2.0f, 5.0f, omega_ratio);
  float gravity_restoring = gravity * k_len;
  float capillary_restoring = k_surface_tension_over_density * k_len * k_len * k_len;
  float capillary_boost = sqrtf((gravity_restoring + capillary_restoring) / max(gravity_restoring, 1.0e-6f));
  capillary_boost = min(capillary_boost, 2.0f);
  float micro_scale_m = 0.004f;
  float micro_damp = 1.0f / (1.0f + powf(k_len * micro_scale_m, 4.0f));
  float short_wave_boost = 1.0f + ((capillary_boost - 1.0f) * capillary_gate * micro_damp);
  spectral_density_k *= short_wave_boost;
  return amplitude_gain * spectral_density_k;
}

static float spectrum_band_weight(float k_len, float k_min, float k_max, float k_min_soft, float k_max_soft) {
  float weight = 1.0f;
  if (k_min_soft > 0.0f) {
    weight *= smooth_step(k_min - k_min_soft, k_min + k_min_soft, k_len);
  } else if (k_len < k_min) {
    return 0.0f;
  }

  if (k_max_soft > 0.0f) {
    weight *= (1.0f - smooth_step(k_max - k_max_soft, k_max + k_max_soft, k_len));
  } else if (k_len > k_max) {
    return 0.0f;
  }
  return weight;
}

static float estimate_base_rms_height(uint32_t resolution, float cascade_length, const float2& wind_dir, float wind_speed, float water_depth, float jonswap_gamma,
  float directional_spread, float k_min, float k_max, float k_min_soft, float k_max_soft) {
  if ((resolution == 0u) || (cascade_length <= 1.0e-6f)) {
    return 0.0f;
  }

  double sum_power = 0.0;
  float n_half = 0.5f * static_cast<float>(resolution);
  float inv_length = (2.0f * kPi) / cascade_length;
  for (uint32_t y = 0; y < resolution; ++y) {
    float n_y = static_cast<float>(y) - n_half;
    for (uint32_t x = 0; x < resolution; ++x) {
      float n_x = static_cast<float>(x) - n_half;
      float2 k = {inv_length * n_x, inv_length * n_y};
      float k_len = length(k);
      float band_weight = spectrum_band_weight(k_len, k_min, k_max, k_min_soft, k_max_soft);
      if (band_weight <= 0.0f) {
        continue;
      }

      float p_k = ocean_wave_spectrum(k, wind_dir, wind_speed, water_depth, jonswap_gamma, directional_spread, 1.0f);
      float p_minus_k = ocean_wave_spectrum(-k, wind_dir, wind_speed, water_depth, jonswap_gamma, directional_spread, 1.0f);
      float expected_ht_power = 0.5f * (p_k + p_minus_k) * band_weight;
      sum_power += static_cast<double>(expected_ht_power);
    }
  }

  double n2 = static_cast<double>(resolution) * static_cast<double>(resolution);
  double variance = sum_power / (n2 * n2);
  if (variance <= 0.0) {
    return 0.0f;
  }
  return static_cast<float>(sqrt(variance));
}

static void calculate_cascade_band_limits(const OceanParameters& parameters, float* out_k_min, float* out_k_max, float* out_k_min_soft, float* out_k_max_soft) {
  constexpr float k_unbounded_max = 1.0e30f;
  float k_step[Ocean::k_cascade_count] = {0.0f, 0.0f, 0.0f};
  for (uint32_t c = 0; c < Ocean::k_cascade_count; ++c) {
    float cascade_length = (parameters.cascade_lengths[c] > 1.0e-3f) ? parameters.cascade_lengths[c] : 1.0e-3f;
    k_step[c] = (2.0f * kPi) / cascade_length;
    out_k_min[c] = 0.0f;
    out_k_max[c] = k_unbounded_max;
    out_k_min_soft[c] = 0.0f;
    out_k_max_soft[c] = 0.0f;
  }

  if (parameters.spectral_band_limit_enable == false) {
    return;
  }

  uint32_t sorted_indices[Ocean::k_cascade_count] = {0u, 1u, 2u};
  for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
    for (uint32_t j = i + 1u; j < Ocean::k_cascade_count; ++j) {
      if (k_step[sorted_indices[j]] < k_step[sorted_indices[i]]) {
        uint32_t tmp = sorted_indices[i];
        sorted_indices[i] = sorted_indices[j];
        sorted_indices[j] = tmp;
      }
    }
  }

  float k_boundaries[Ocean::k_cascade_count - 1u] = {0.0f, 0.0f};
  float k_boundary_soft[Ocean::k_cascade_count - 1u] = {0.0f, 0.0f};
  for (uint32_t b = 0; b < (Ocean::k_cascade_count - 1u); ++b) {
    float k0 = k_step[sorted_indices[b]];
    float k1 = k_step[sorted_indices[b + 1u]];
    k_boundaries[b] = 0.5f * (k0 + k1);
    float raw_soft = 0.5f * min(k0, k1);
    float max_soft = 0.5f * max(0.0f, k1 - k0);
    k_boundary_soft[b] = min(raw_soft, max_soft);
  }

  // Keep adjacent soft transitions disjoint so neighboring band gates remain complementary.
  for (uint32_t b = 0; (b + 1u) < (Ocean::k_cascade_count - 1u); ++b) {
    float available_span = max(0.0f, k_boundaries[b + 1u] - k_boundaries[b]);
    float target_span = available_span * 0.98f;
    float combined_soft = k_boundary_soft[b] + k_boundary_soft[b + 1u];
    if (combined_soft > target_span) {
      float soft_scale = (combined_soft > 1.0e-8f) ? (target_span / combined_soft) : 0.0f;
      k_boundary_soft[b] *= soft_scale;
      k_boundary_soft[b + 1u] *= soft_scale;
    }
  }

  for (uint32_t rank = 0; rank < Ocean::k_cascade_count; ++rank) {
    uint32_t cascade_index = sorted_indices[rank];
    float k_min = (rank > 0u) ? k_boundaries[rank - 1u] : 0.0f;
    float k_max = (rank + 1u < Ocean::k_cascade_count) ? k_boundaries[rank] : k_unbounded_max;

    if (rank > 0u) {
      out_k_min_soft[cascade_index] = k_boundary_soft[rank - 1u];
    }
    if (rank + 1u < Ocean::k_cascade_count) {
      out_k_max_soft[cascade_index] = k_boundary_soft[rank];
    }

    out_k_min[cascade_index] = k_min;
    out_k_max[cascade_index] = k_max;
    if ((k_max < k_unbounded_max) && (out_k_max[cascade_index] <= out_k_min[cascade_index])) {
      out_k_max[cascade_index] = out_k_min[cascade_index] + 1.0e-6f;
    }
  }
}

static bool spectrum_parameters_changed(const OceanParameters& current, const OceanParameters& previous) {
  constexpr float epsilon = 1e-6f;
  if (current.spectral_band_limit_enable != previous.spectral_band_limit_enable) {
    return true;
  }
  if (fabsf(current.wind_direction_x - previous.wind_direction_x) > epsilon) {
    return true;
  }
  if (fabsf(current.wind_direction_y - previous.wind_direction_y) > epsilon) {
    return true;
  }
  if (fabsf(current.wind_speed - previous.wind_speed) > epsilon) {
    return true;
  }
  if (fabsf(current.water_depth - previous.water_depth) > epsilon) {
    return true;
  }
  if (fabsf(current.jonswap_gamma - previous.jonswap_gamma) > epsilon) {
    return true;
  }
  if (fabsf(current.directional_spread - previous.directional_spread) > epsilon) {
    return true;
  }
  if (current.significant_wave_height_enable != previous.significant_wave_height_enable) {
    return true;
  }
  if (fabsf(current.significant_wave_height - previous.significant_wave_height) > epsilon) {
    return true;
  }
  if (fabsf(current.cascade_detail_boost - previous.cascade_detail_boost) > epsilon) {
    return true;
  }
  for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
    if (fabsf(current.cascade_lengths[i] - previous.cascade_lengths[i]) > epsilon) {
      return true;
    }
    if (fabsf(current.cascade_amplitudes[i] - previous.cascade_amplitudes[i]) > epsilon) {
      return true;
    }
  }
  return false;
}

static void apply_cascade_detail_boost(const OceanParameters& parameters, float* io_rms) {
  if (io_rms == nullptr) {
    return;
  }

  float detail_boost = max(parameters.cascade_detail_boost, 0.0f);
  if (fabsf(detail_boost - 1.0f) <= 1.0e-6f) {
    return;
  }

  float min_length = 1.0e30f;
  float max_length = 0.0f;
  for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
    float cascade_length = max(parameters.cascade_lengths[i], 1.0e-3f);
    min_length = min(min_length, cascade_length);
    max_length = max(max_length, cascade_length);
  }

  float log_range = logf(max(max_length, 1.0e-3f)) - logf(max(min_length, 1.0e-3f));
  if (fabsf(log_range) <= 1.0e-6f) {
    for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
      io_rms[i] *= detail_boost;
    }
    return;
  }

  float pre_boost_rms_sq = 0.0f;
  float post_boost_rms_sq = 0.0f;
  float boosted_rms[Ocean::k_cascade_count] = {0.0f, 0.0f, 0.0f};
  for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
    float cascade_length = max(parameters.cascade_lengths[i], 1.0e-3f);
    float rank_t = (logf(max_length) - logf(cascade_length)) / log_range;
    rank_t = min(max(rank_t, 0.0f), 1.0f);
    float cascade_boost = powf(detail_boost, rank_t);
    float base_rms = max(io_rms[i], 0.0f);
    boosted_rms[i] = base_rms * cascade_boost;
    pre_boost_rms_sq += (base_rms * base_rms);
    post_boost_rms_sq += (boosted_rms[i] * boosted_rms[i]);
  }

  float preserve_total_scale = 1.0f;
  if ((parameters.significant_wave_height_enable) && (pre_boost_rms_sq > 1.0e-12f) && (post_boost_rms_sq > 1.0e-12f)) {
    preserve_total_scale = sqrtf(pre_boost_rms_sq / post_boost_rms_sq);
  }

  for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
    io_rms[i] = boosted_rms[i] * preserve_total_scale;
  }
}

static void calculate_effective_cascade_weights(const OceanParameters& parameters, float* out_weights) {
  bool use_neutral_physical_mixing = (parameters.significant_wave_height_enable) && (parameters.debug_cascade_overrides_enable == false);
  if (use_neutral_physical_mixing) {
    for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
      out_weights[i] = 1.0f;
    }
    return;
  }

  for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
    float weight = max(0.0f, parameters.cascade_render_weight[i]);
    if (parameters.cascade_enable[i] == false) {
      weight = 0.0f;
    }
    if ((parameters.solo_cascade >= 0) && (parameters.solo_cascade != static_cast<int32_t>(i))) {
      weight = 0.0f;
    }
    out_weights[i] = weight;
  }
}

static bool enforce_cascade_length_constraints(OceanParameters& parameters) {
  bool changed = false;

  for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
    float length = min(k_cascade_length_max, max(k_cascade_length_min[i], parameters.cascade_lengths[i]));
    if (fabsf(parameters.cascade_lengths[i] - length) > 1.0e-6f) {
      parameters.cascade_lengths[i] = length;
      changed = true;
    }
  }

  return changed;
}

static void calculate_target_cascade_rms(uint32_t fft_resolution, const OceanParameters& parameters, float* out_rms) {
  if (parameters.significant_wave_height_enable == false) {
    for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
      out_rms[i] = max(0.0f, parameters.cascade_amplitudes[i]);
    }
    apply_cascade_detail_boost(parameters, out_rms);
    return;
  }

  float2 wind_dir = {parameters.wind_direction_x, parameters.wind_direction_y};
  float wind_dir_len_sq = dot(wind_dir, wind_dir);
  if (wind_dir_len_sq < 1.0e-6f) {
    wind_dir = {1.0f, 0.0f};
  } else {
    wind_dir = wind_dir / sqrtf(wind_dir_len_sq);
  }

  float cascade_k_min[Ocean::k_cascade_count] = {0.0f, 0.0f, 0.0f};
  float cascade_k_max[Ocean::k_cascade_count] = {1.0e30f, 1.0e30f, 1.0e30f};
  float cascade_k_min_soft[Ocean::k_cascade_count] = {0.0f, 0.0f, 0.0f};
  float cascade_k_max_soft[Ocean::k_cascade_count] = {0.0f, 0.0f, 0.0f};
  calculate_cascade_band_limits(parameters, cascade_k_min, cascade_k_max, cascade_k_min_soft, cascade_k_max_soft);

  float base_cascade_rms[Ocean::k_cascade_count] = {0.0f, 0.0f, 0.0f};
  float base_total_rms_sq = 0.0f;
  for (uint32_t c = 0; c < Ocean::k_cascade_count; ++c) {
    float base_rms = estimate_base_rms_height(fft_resolution, parameters.cascade_lengths[c], wind_dir, parameters.wind_speed, parameters.water_depth, parameters.jonswap_gamma,
      parameters.directional_spread, cascade_k_min[c], cascade_k_max[c], cascade_k_min_soft[c], cascade_k_max_soft[c]);
    if ((std::isfinite(base_rms) == false) || (base_rms <= 1.0e-6f)) {
      base_rms = estimate_base_rms_height(fft_resolution, parameters.cascade_lengths[c], wind_dir, parameters.wind_speed, parameters.water_depth, parameters.jonswap_gamma,
        parameters.directional_spread, 0.0f, 1.0e30f, 0.0f, 0.0f);
    }
    if ((std::isfinite(base_rms) == false) || (base_rms <= 1.0e-6f)) {
      base_rms = 0.0f;
    }
    base_cascade_rms[c] = base_rms;
    base_total_rms_sq += (base_rms * base_rms);
  }

  if (base_total_rms_sq <= 1.0e-12f) {
    for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
      out_rms[i] = 0.0f;
    }
    return;
  }

  float target_sigma = max(0.0f, parameters.significant_wave_height) * 0.25f;
  float rms_scale = target_sigma / sqrtf(base_total_rms_sq);
  for (uint32_t i = 0; i < Ocean::k_cascade_count; ++i) {
    out_rms[i] = base_cascade_rms[i] * rms_scale;
  }
  apply_cascade_detail_boost(parameters, out_rms);
}

void Ocean::init(RHIContext& rhi, RHITextureFormat color_format, RHITextureFormat depth_format, uint32_t sample_count) {
  const int M = static_cast<int>(_patch_resolution);
  ETX_ASSERT(_clipmap_levels <= k_max_clipmap_levels);
  _h0_generated = false;
  _resources_initialized = false;
  _has_previous_spectrum_parameters = false;

  std::vector<Vertex> vertices;
  vertices.reserve(static_cast<size_t>((M + 1) * (M + 1)));
  for (int y = 0; y <= M; ++y) {
    for (int x = 0; x <= M; ++x) {
      float px = static_cast<float>(x) / static_cast<float>(M);
      float pz = static_cast<float>(y) / static_cast<float>(M);

      Vertex v = {};
      v.pos = {px, 0.0f, pz};
      v.nrm = {0.0f, 1.0f, 0.0f};
      v.tan = {1.0f, 0.0f, 0.0f};
      v.btn = {0.0f, 0.0f, 1.0f};
      v.tex = {px, pz};
      vertices.push_back(v);
    }
  }

  std::vector<uint32_t> indices;
  indices.reserve(static_cast<size_t>(M * M * 6));
  for (int y = 0; y < M; ++y) {
    for (int x = 0; x < M; ++x) {
      uint32_t tl = static_cast<uint32_t>(y * (M + 1) + x);
      uint32_t tr = tl + 1;
      uint32_t bl = static_cast<uint32_t>((y + 1) * (M + 1) + x);
      uint32_t br = bl + 1;

      indices.push_back(tl);
      indices.push_back(bl);
      indices.push_back(tr);

      indices.push_back(tr);
      indices.push_back(bl);
      indices.push_back(br);
    }
  }
  _index_count = static_cast<uint32_t>(indices.size());

  std::vector<ClipmapInstance> instances;
  for (uint32_t l = 0; l < _clipmap_levels; ++l) {
    float scale = _base_patch_size * static_cast<float>(1 << l);
    for (int x = -2; x <= 1; ++x) {
      for (int z = -2; z <= 1; ++z) {
        if ((l > 0) && (x >= -1) && (x <= 0) && (z >= -1) && (z <= 0)) {
          continue;
        }
        instances.push_back({static_cast<float>(x) * scale, static_cast<float>(z) * scale, scale, static_cast<float>(l)});
      }
    }
  }
  _instance_capacity = static_cast<uint32_t>(instances.size());
  _instance_count = _instance_capacity;

  RHIBufferDesc vb_desc = {
    .size = vertices.size() * sizeof(Vertex),
    .usage = RHIBufferUsage::Vertex,
    .host_visible = true,
  };
  auto vb_result = rhi.device().create_buffer(vb_desc);
  if (vb_result.result == RHIResult::Success) {
    _vertex_buffer = vb_result.handle;
    rhi.device().update_buffer(_vertex_buffer, vertices.data(), vb_desc.size);
  } else {
    log::error("Ocean: Failed to create vertex buffer: %u", vb_result.result);
  }

  RHIBufferDesc ib_desc = {
    .size = indices.size() * sizeof(uint32_t),
    .usage = RHIBufferUsage::Index,
    .host_visible = true,
  };
  auto ib_result = rhi.device().create_buffer(ib_desc);
  if (ib_result.result == RHIResult::Success) {
    _index_buffer = ib_result.handle;
    rhi.device().update_buffer(_index_buffer, indices.data(), ib_desc.size);
  } else {
    log::error("Ocean: Failed to create index buffer: %u", ib_result.result);
  }

  RHIBufferDesc inst_desc = {
    .size = instances.size() * sizeof(ClipmapInstance),
    .usage = RHIBufferUsage::Vertex | RHIBufferUsage::Storage,
    .host_visible = true,
  };
  for (uint32_t i = 0; i < kRHIMaxFrames; ++i) {
    auto inst_result = rhi.device().create_buffer(inst_desc);
    if (inst_result.result == RHIResult::Success) {
      _instance_buffer[i] = inst_result.handle;
      rhi.device().update_buffer(_instance_buffer[i], instances.data(), inst_desc.size);
    } else {
      log::error("Ocean: Failed to create instance buffer[%u]: %u", i, inst_result.result);
    }
  }

  RHIBufferDesc settings_desc = {
    .size = sizeof(OceanRenderSettings),
    .usage = RHIBufferUsage::Storage,
    .host_visible = true,
  };
  for (uint32_t i = 0; i < kRHIMaxFrames; ++i) {
    auto settings_result = rhi.device().create_buffer(settings_desc);
    if (settings_result.result == RHIResult::Success) {
      _settings_buffer[i] = settings_result.handle;
    } else {
      log::error("Ocean: Failed to create settings buffer[%u]: %u", i, settings_result.result);
    }
  }

  std::string shader_source = env().file_in_data("playground/shaders/ocean.hlsl");
  ShaderCompiler::ShaderEntryPoint vs = {"VSMain", RHIShaderStage::Vertex};
  ShaderCompiler::ShaderEntryPoint ps = {"PSMain", RHIShaderStage::Fragment};
  ShaderCompiler::ShaderEntryPoint ps_thickness = {"PSThickness", RHIShaderStage::Fragment};
  ShaderCompiler::ShaderEntryPoint ps_foam_history = {"PSFoamHistory", RHIShaderStage::Fragment};
  auto compilation = ShaderCompiler::instance().compile(shader_source, {vs, ps, ps_thickness, ps_foam_history});

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
    p_desc.depth_state.depth_compare_op = RHICompareOp::Less;

    p_desc.blend.blend_enable = false;
    p_desc.primitive_topology = RHIPrimitiveTopology::TriangleList;

    p_desc.color_attachment_count = 1;
    p_desc.color_formats[0] = color_format;
    p_desc.depth_format = depth_format;
    p_desc.sample_count = sample_count;

    auto p_result = rhi.device().create_graphics_pipeline(p_desc);
    if (p_result.result != RHIResult::Success) {
      log::error("Failed to create ocean graphics pipeline");
    }
    _pipeline = p_result.handle;

    RHIGraphicsPipelineDesc wire_desc = p_desc;
    wire_desc.primitive_topology = RHIPrimitiveTopology::TriangleList;
    wire_desc.depth_state.depth_write_enable = false;
    wire_desc.rasterization.wireframe_enable = true;
    wire_desc.rasterization.line_width = 1.0f;
    auto wire_result = rhi.device().create_graphics_pipeline(wire_desc);
    if (wire_result.result != RHIResult::Success) {
      log::error("Failed to create ocean wireframe pipeline");
    }
    _wire_pipeline = wire_result.handle;

    RHIGraphicsPipelineDesc thickness_desc = p_desc;
    thickness_desc.fragment_shader.entry_point = "PSThickness";
    thickness_desc.fragment_shader.spirv_data = compilation.binaries[2].spirv_data;
    thickness_desc.fragment_shader.spirv_size = compilation.binaries[2].spirv_size;
    thickness_desc.depth_state.depth_test_enable = false;
    thickness_desc.depth_state.depth_write_enable = false;
    thickness_desc.depth_format = RHITextureFormat::Undefined;
    thickness_desc.sample_count = 1u;
    thickness_desc.color_formats[0] = RHITextureFormat::R32_FLOAT;
    thickness_desc.blend.blend_enable = true;
    thickness_desc.blend.src_color_blend_factor = RHIBlendFactor::One;
    thickness_desc.blend.dst_color_blend_factor = RHIBlendFactor::One;
    thickness_desc.blend.src_alpha_blend_factor = RHIBlendFactor::One;
    thickness_desc.blend.dst_alpha_blend_factor = RHIBlendFactor::One;
    thickness_desc.blend.color_blend_op = RHIBlendOp::Min;
    thickness_desc.blend.alpha_blend_op = RHIBlendOp::Min;

    auto thickness_min_result = rhi.device().create_graphics_pipeline(thickness_desc);
    if (thickness_min_result.result != RHIResult::Success) {
      log::error("Failed to create ocean thickness MIN pipeline");
    }
    _thickness_min_pipeline = thickness_min_result.handle;

    thickness_desc.blend.color_blend_op = RHIBlendOp::Max;
    thickness_desc.blend.alpha_blend_op = RHIBlendOp::Max;
    auto thickness_max_result = rhi.device().create_graphics_pipeline(thickness_desc);
    if (thickness_max_result.result != RHIResult::Success) {
      log::error("Failed to create ocean thickness MAX pipeline");
    }
    _thickness_max_pipeline = thickness_max_result.handle;

    RHIGraphicsPipelineDesc foam_history_desc = p_desc;
    foam_history_desc.fragment_shader.entry_point = "PSFoamHistory";
    foam_history_desc.fragment_shader.spirv_data = compilation.binaries[3].spirv_data;
    foam_history_desc.fragment_shader.spirv_size = compilation.binaries[3].spirv_size;
    foam_history_desc.depth_state.depth_test_enable = false;
    foam_history_desc.depth_state.depth_write_enable = false;
    foam_history_desc.depth_format = RHITextureFormat::Undefined;
    foam_history_desc.sample_count = 1u;
    foam_history_desc.color_formats[0] = RHITextureFormat::R8_UNORM;
    foam_history_desc.blend.blend_enable = true;
    foam_history_desc.blend.src_color_blend_factor = RHIBlendFactor::One;
    foam_history_desc.blend.dst_color_blend_factor = RHIBlendFactor::One;
    foam_history_desc.blend.src_alpha_blend_factor = RHIBlendFactor::One;
    foam_history_desc.blend.dst_alpha_blend_factor = RHIBlendFactor::One;
    foam_history_desc.blend.color_blend_op = RHIBlendOp::Max;
    foam_history_desc.blend.alpha_blend_op = RHIBlendOp::Max;
    auto foam_history_result = rhi.device().create_graphics_pipeline(foam_history_desc);
    if (foam_history_result.result != RHIResult::Success) {
      log::error("Failed to create ocean foam history pipeline");
    }
    _foam_history_pipeline = foam_history_result.handle;

    // Compute Pipeline
    std::string compute_source = env().file_in_data("playground/shaders/ocean_compute.hlsl");
    auto cs_compilation = ShaderCompiler::instance().compile(compute_source, {{"GenerateH0", RHIShaderStage::Compute}, {"UpdateSpectrum", RHIShaderStage::Compute}});
    if (cs_compilation.result != RHIResult::Success) {
      log::error("Failed to compile ocean compute shader:\n%s", cs_compilation.error_message.c_str());
    } else {
      RHIComputePipelineDesc h0_desc = {};
      h0_desc.compute_shader.stage = RHIShaderStage::Compute;
      h0_desc.compute_shader.entry_point = "GenerateH0";
      h0_desc.compute_shader.spirv_data = cs_compilation.binaries[0].spirv_data;
      h0_desc.compute_shader.spirv_size = cs_compilation.binaries[0].spirv_size;
      auto h0_result = rhi.device().create_compute_pipeline(h0_desc);
      if (h0_result.result != RHIResult::Success) {
        log::error("Failed to create ocean GenerateH0 pipeline");
      }
      _h0_pipeline = h0_result.handle;

      RHIComputePipelineDesc update_desc = {};
      update_desc.compute_shader.stage = RHIShaderStage::Compute;
      update_desc.compute_shader.entry_point = "UpdateSpectrum";
      update_desc.compute_shader.spirv_data = cs_compilation.binaries[1].spirv_data;
      update_desc.compute_shader.spirv_size = cs_compilation.binaries[1].spirv_size;
      auto update_result = rhi.device().create_compute_pipeline(update_desc);
      if (update_result.result != RHIResult::Success) {
        log::error("Failed to create ocean UpdateSpectrum pipeline");
      }
      _update_spectrum_pipeline = update_result.handle;

      std::string fft_source = env().file_in_data("playground/shaders/ocean_fft.hlsl");
      auto fft_compilation = ShaderCompiler::instance().compile(fft_source, {{"FFTMain", RHIShaderStage::Compute}});
      if (fft_compilation.result != RHIResult::Success) {
        log::error("Failed to compile ocean FFT shader:\n%s", fft_compilation.error_message.c_str());
      } else {
        RHIComputePipelineDesc fft_desc = {};
        fft_desc.compute_shader.stage = RHIShaderStage::Compute;
        fft_desc.compute_shader.entry_point = "FFTMain";
        fft_desc.compute_shader.spirv_data = fft_compilation.binaries[0].spirv_data;
        fft_desc.compute_shader.spirv_size = fft_compilation.binaries[0].spirv_size;
        auto fft_result = rhi.device().create_compute_pipeline(fft_desc);
        if (fft_result.result != RHIResult::Success) {
          log::error("Failed to create ocean FFT pipeline");
        }
        _fft_pipeline = fft_result.handle;
      }

      std::string assemble_source = env().file_in_data("playground/shaders/ocean_assemble.hlsl");
      auto assemble_compilation = ShaderCompiler::instance().compile(assemble_source, {{"AssembleMain", RHIShaderStage::Compute}});
      if (assemble_compilation.result != RHIResult::Success) {
        log::error("Failed to compile ocean Assemble shader:\n%s", assemble_compilation.error_message.c_str());
      } else {
        RHIComputePipelineDesc assemble_desc = {};
        assemble_desc.compute_shader.stage = RHIShaderStage::Compute;
        assemble_desc.compute_shader.entry_point = "AssembleMain";
        assemble_desc.compute_shader.spirv_data = assemble_compilation.binaries[0].spirv_data;
        assemble_desc.compute_shader.spirv_size = assemble_compilation.binaries[0].spirv_size;
        auto assemble_result = rhi.device().create_compute_pipeline(assemble_desc);
        if (assemble_result.result != RHIResult::Success) {
          log::error("Failed to create ocean Assemble pipeline");
        }
        _assemble_pipeline = assemble_result.handle;
      }
    }

    const uint32_t TESS = _fft_resolution;
    RHITextureDesc tex_desc = {};
    tex_desc.width = TESS;
    tex_desc.height = TESS;
    tex_desc.array_layers = 1;
    tex_desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
    tex_desc.usage = RHITextureUsage::Sampled | RHITextureUsage::Storage | RHITextureUsage::TransferSrc;

    RHITextureDesc surface_derivative_desc = tex_desc;
#if defined(ETX_PLATFORM_APPLE)
    surface_derivative_desc.mip_levels = 1;
    surface_derivative_desc.usage = RHITextureUsage::Sampled | RHITextureUsage::Storage;
#else
    surface_derivative_desc.mip_levels = calculate_mip_levels(tex_desc.width, tex_desc.height);
    surface_derivative_desc.usage =
      RHITextureUsage::Sampled | RHITextureUsage::Storage | RHITextureUsage::ColorAttachment | RHITextureUsage::TransferSrc | RHITextureUsage::TransferDst;
#endif

    for (uint32_t i = 0; i < k_cascade_count; ++i) {
      _displacement_map[i] = rhi.device().create_texture(tex_desc).handle;
      _surface_derivative_u_map[i] = rhi.device().create_texture(surface_derivative_desc).handle;
      _surface_derivative_v_map[i] = rhi.device().create_texture(surface_derivative_desc).handle;
      _slope_metric_map[i] = rhi.device().create_texture(surface_derivative_desc).handle;
      _h0_texture[i] = rhi.device().create_texture(tex_desc).handle;
      _ht_texture[i] = rhi.device().create_texture(tex_desc).handle;
      _dxdz_texture[i] = rhi.device().create_texture(tex_desc).handle;
      _deriv_spec_0_texture[i] = rhi.device().create_texture(tex_desc).handle;
      _deriv_spec_1_texture[i] = rhi.device().create_texture(tex_desc).handle;
      _deriv_spec_2_texture[i] = rhi.device().create_texture(tex_desc).handle;
      _ht_pingpong[i] = rhi.device().create_texture(tex_desc).handle;
      _dxdz_pingpong[i] = rhi.device().create_texture(tex_desc).handle;
      _deriv_spec_0_pingpong[i] = rhi.device().create_texture(tex_desc).handle;
      _deriv_spec_1_pingpong[i] = rhi.device().create_texture(tex_desc).handle;
      _deriv_spec_2_pingpong[i] = rhi.device().create_texture(tex_desc).handle;
    }

    if (load_r8_texture_from_file(rhi, "bin/playground/textures/foam.png", _foam_detail_texture, "ocean foam detail") == false) {
      log::error("Ocean foam texture will be disabled until bin/playground/textures/foam.png is available");
    }
    if (load_r8_texture_from_file(rhi, "bin/playground/textures/aeration.png", _aeration_detail_texture, "ocean aeration detail") == false) {
      log::error("Ocean aeration texture will be disabled until bin/playground/textures/aeration.png is available");
    }
  } else {
    log::error("Ocean: Failed to compile shaders:\n%s", compilation.error_message.c_str());
  }
}

void Ocean::cleanup(RHIContext& rhi) {
  _has_previous_view_proj = false;
  _last_update_time = 0.0f;
  if (_pipeline.valid()) {
    rhi.device().destroy_pipeline(_pipeline);
  }
  if (_wire_pipeline.valid()) {
    rhi.device().destroy_pipeline(_wire_pipeline);
  }
  if (_thickness_min_pipeline.valid()) {
    rhi.device().destroy_pipeline(_thickness_min_pipeline);
  }
  if (_thickness_max_pipeline.valid()) {
    rhi.device().destroy_pipeline(_thickness_max_pipeline);
  }
  if (_foam_history_pipeline.valid()) {
    rhi.device().destroy_pipeline(_foam_history_pipeline);
  }
  if (_h0_pipeline.valid()) {
    rhi.device().destroy_pipeline(_h0_pipeline);
  }
  if (_update_spectrum_pipeline.valid()) {
    rhi.device().destroy_pipeline(_update_spectrum_pipeline);
  }
  if (_fft_pipeline.valid()) {
    rhi.device().destroy_pipeline(_fft_pipeline);
  }
  if (_assemble_pipeline.valid()) {
    rhi.device().destroy_pipeline(_assemble_pipeline);
  }
  if (_vertex_buffer.valid()) {
    rhi.device().destroy_buffer(_vertex_buffer);
  }
  if (_index_buffer.valid()) {
    rhi.device().destroy_buffer(_index_buffer);
  }
  if (_foam_detail_texture.valid()) {
    rhi.device().destroy_texture(_foam_detail_texture);
  }
  if (_aeration_detail_texture.valid()) {
    rhi.device().destroy_texture(_aeration_detail_texture);
  }
  for (uint32_t i = 0; i < kRHIMaxFrames; ++i) {
    if (_instance_buffer[i].valid()) {
      rhi.device().destroy_buffer(_instance_buffer[i]);
    }
  }
  for (uint32_t i = 0; i < kRHIMaxFrames; ++i) {
    if (_settings_buffer[i].valid()) {
      rhi.device().destroy_buffer(_settings_buffer[i]);
    }
  }
  for (uint32_t i = 0; i < k_cascade_count; ++i) {
    if (_displacement_map[i].valid())
      rhi.device().destroy_texture(_displacement_map[i]);
    if (_surface_derivative_u_map[i].valid())
      rhi.device().destroy_texture(_surface_derivative_u_map[i]);
    if (_surface_derivative_v_map[i].valid())
      rhi.device().destroy_texture(_surface_derivative_v_map[i]);
    if (_slope_metric_map[i].valid())
      rhi.device().destroy_texture(_slope_metric_map[i]);
    if (_h0_texture[i].valid())
      rhi.device().destroy_texture(_h0_texture[i]);
    if (_ht_texture[i].valid())
      rhi.device().destroy_texture(_ht_texture[i]);
    if (_dxdz_texture[i].valid())
      rhi.device().destroy_texture(_dxdz_texture[i]);
    if (_deriv_spec_0_texture[i].valid())
      rhi.device().destroy_texture(_deriv_spec_0_texture[i]);
    if (_deriv_spec_1_texture[i].valid())
      rhi.device().destroy_texture(_deriv_spec_1_texture[i]);
    if (_deriv_spec_2_texture[i].valid())
      rhi.device().destroy_texture(_deriv_spec_2_texture[i]);
    if (_ht_pingpong[i].valid())
      rhi.device().destroy_texture(_ht_pingpong[i]);
    if (_dxdz_pingpong[i].valid())
      rhi.device().destroy_texture(_dxdz_pingpong[i]);
    if (_deriv_spec_0_pingpong[i].valid())
      rhi.device().destroy_texture(_deriv_spec_0_pingpong[i]);
    if (_deriv_spec_1_pingpong[i].valid())
      rhi.device().destroy_texture(_deriv_spec_1_pingpong[i]);
    if (_deriv_spec_2_pingpong[i].valid())
      rhi.device().destroy_texture(_deriv_spec_2_pingpong[i]);
  }
  _has_previous_spectrum_parameters = false;
  _h0_generated = false;
  _resources_initialized = false;
  _instance_capacity = 0;
  _instance_count = 0;
  for (uint32_t i = 0; i < k_cascade_count; ++i) {
    _resolved_cascade_rms[i] = 0.0f;
  }
}

void Ocean::update(RHIContext& rhi, RHICommandBuffer cmd, float time, const float3& camera_position, const float3& camera_direction, float fov, uint32_t viewport_width,
  uint32_t viewport_height) {
  _last_update_time = time * _parameters.time_scale;
  if ((valid() == false) || (_h0_pipeline.valid() == false) || (_displacement_map[0].valid() == false)) {
    return;
  }
  enforce_cascade_length_constraints(_parameters);

  float effective_cascade_weight[k_cascade_count] = {0.0f, 0.0f, 0.0f};
  calculate_effective_cascade_weights(_parameters, effective_cascade_weight);
  float target_cascade_rms[k_cascade_count] = {0.0f, 0.0f, 0.0f};
  calculate_target_cascade_rms(_fft_resolution, _parameters, target_cascade_rms);
  for (uint32_t i = 0; i < k_cascade_count; ++i) {
    _resolved_cascade_rms[i] = target_cascade_rms[i];
  }

  float weighted_height_rms_sq = 0.0f;
  for (uint32_t i = 0; i < k_cascade_count; ++i) {
    float cascade_rms = target_cascade_rms[i] * effective_cascade_weight[i];
    weighted_height_rms_sq += cascade_rms * cascade_rms;
  }
  float weighted_height_rms = sqrtf(max(0.0f, weighted_height_rms_sq));
  float vertical_displacement_margin = max(0.25f, 3.0f * weighted_height_rms);
  float horizontal_displacement_margin = fabsf(_parameters.choppiness) * vertical_displacement_margin;

  uint32_t frame_index = rhi.get_current_frame_index();
  RHIBindlessHandle frame_instance_buffer = _instance_buffer[frame_index];
  if ((frame_instance_buffer.valid()) && (_instance_capacity > 0u) && (_parameters.lock_lods == false)) {
    float3 view_forward = camera_direction;
    float forward_len_sq = dot(view_forward, view_forward);
    if (forward_len_sq > 1.0e-6f) {
      view_forward = normalize(view_forward);
    } else {
      view_forward = {0.0f, 0.0f, -1.0f};
    }

    float base_grid_size = _base_patch_size / static_cast<float>(_patch_resolution);
    float center_x = camera_position.x;
    float center_z = camera_position.z;
    float snapped_center_x = floorf(center_x / base_grid_size) * base_grid_size;
    float snapped_center_z = floorf(center_z / base_grid_size) * base_grid_size;
    float snapped_camera_x = floorf(camera_position.x / base_grid_size) * base_grid_size;
    float snapped_camera_z = floorf(camera_position.z / base_grid_size) * base_grid_size;
    _clipmap_center_offset = {snapped_center_x - snapped_camera_x, snapped_center_z - snapped_camera_z};

    float3 view_right = cross(view_forward, float3{0.0f, 1.0f, 0.0f});
    float right_len_sq = dot(view_right, view_right);
    if (right_len_sq > 1.0e-6f) {
      view_right = normalize(view_right);
    } else {
      view_right = {1.0f, 0.0f, 0.0f};
    }
    float3 view_up = normalize(cross(view_right, view_forward));

    float aspect = (viewport_height > 0u) ? (static_cast<float>(viewport_width) / static_cast<float>(viewport_height)) : 1.0f;
    float tan_half_fov = tanf(0.5f * fov);
    uint32_t min_active_lod = calculate_min_active_clipmap_level(camera_position.y, _base_patch_size, _patch_resolution, fov, viewport_height, _clipmap_levels);

    ClipmapInstance visible_instances[k_max_clipmap_instances] = {};
    uint32_t visible_instance_count = 0;
    for (uint32_t l = min_active_lod; l < _clipmap_levels; ++l) {
      float scale = _base_patch_size * static_cast<float>(1u << l);
      float half_scale = 0.5f * scale;
      float planar_radius = (half_scale + horizontal_displacement_margin) * k_planar_patch_radius_scale;
      float patch_radius = sqrtf((planar_radius * planar_radius) + (vertical_displacement_margin * vertical_displacement_margin));
      bool keep_full_ring = (l < (min_active_lod + k_unculled_lod_levels));
      for (int x = -2; x <= 1; ++x) {
        for (int z = -2; z <= 1; ++z) {
          if ((l > min_active_lod) && (x >= -1) && (x <= 0) && (z >= -1) && (z <= 0)) {
            continue;
          }
          float world_center_x = ((static_cast<float>(x) + 0.5f) * scale) + snapped_camera_x + _clipmap_center_offset.x;
          float world_center_z = ((static_cast<float>(z) + 0.5f) * scale) + snapped_camera_z + _clipmap_center_offset.y;
          float3 patch_center = {world_center_x, 0.0f, world_center_z};
          bool patch_visible = keep_full_ring;
          if (patch_visible == false) {
            patch_visible = patch_visible_in_frustum(patch_center, patch_radius, camera_position, view_right, view_up, view_forward, tan_half_fov, aspect, k_frustum_cull_guard);
          }
          if ((patch_visible) && (visible_instance_count < _instance_capacity)) {
            visible_instances[visible_instance_count++] = {static_cast<float>(x) * scale, static_cast<float>(z) * scale, scale, static_cast<float>(l)};
          }
        }
      }
    }

    if (visible_instance_count == 0u) {
      float scale = _base_patch_size * static_cast<float>(1u << min_active_lod);
      float level = static_cast<float>(min_active_lod);
      visible_instances[0] = {-1.0f * scale, -1.0f * scale, scale, level};
      visible_instances[1] = {0.0f * scale, -1.0f * scale, scale, level};
      visible_instances[2] = {-1.0f * scale, 0.0f * scale, scale, level};
      visible_instances[3] = {0.0f * scale, 0.0f * scale, scale, level};
      visible_instance_count = 4u;
    }

    _instance_count = visible_instance_count;
    size_t visible_data_size = static_cast<size_t>(visible_instance_count) * sizeof(ClipmapInstance);
    rhi.device().update_buffer(frame_instance_buffer, visible_instances, visible_data_size);
  }

  if ((_has_previous_spectrum_parameters == false) || spectrum_parameters_changed(_parameters, _previous_spectrum_parameters)) {
    _h0_generated = false;
    _previous_spectrum_parameters = _parameters;
    _has_previous_spectrum_parameters = true;
  }

  const int TESS = static_cast<int>(_fft_resolution);
  static bool fft_resolution_error_reported = false;
  if (is_power_of_two_u32(_fft_resolution) == false) {
    if (fft_resolution_error_reported == false) {
      log::error("Ocean: FFT resolution must be power-of-two, got %u", _fft_resolution);
      fft_resolution_error_reported = true;
    }
    return;
  }
  if (fft_resolution_error_reported) {
    fft_resolution_error_reported = false;
  }
  uint32_t fft_log2_n = integer_log2_u32(_fft_resolution);
  uint32_t dispatch_group_count_x = ceil_div_u32(_fft_resolution, 8u);
  uint32_t dispatch_group_count_y = ceil_div_u32(_fft_resolution, 8u);
  if (_resources_initialized == false) {
    for (uint32_t i = 0; i < k_cascade_count; ++i) {
      rhi.cmd_texture_barrier(cmd, _h0_texture[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _ht_texture[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _dxdz_texture[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_0_texture[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_1_texture[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_2_texture[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _ht_pingpong[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _dxdz_pingpong[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_0_pingpong[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_1_pingpong[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_2_pingpong[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _displacement_map[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _surface_derivative_u_map[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _surface_derivative_v_map[i], RHIResourceState::Undefined, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _slope_metric_map[i], RHIResourceState::Undefined, RHIResourceState::General);
    }
    _resources_initialized = true;
  } else if (_assemble_pipeline.valid()) {
    for (uint32_t i = 0; i < k_cascade_count; ++i) {
      rhi.cmd_texture_barrier(cmd, _displacement_map[i], RHIResourceState::ShaderReadOnly, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _surface_derivative_u_map[i], RHIResourceState::ShaderReadOnly, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _surface_derivative_v_map[i], RHIResourceState::ShaderReadOnly, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _slope_metric_map[i], RHIResourceState::ShaderReadOnly, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_0_texture[i], RHIResourceState::General, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_1_texture[i], RHIResourceState::General, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_2_texture[i], RHIResourceState::General, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_0_pingpong[i], RHIResourceState::General, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_1_pingpong[i], RHIResourceState::General, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_2_pingpong[i], RHIResourceState::General, RHIResourceState::General);
    }
  }

  if ((_h0_generated == false) && _h0_pipeline.valid()) {
    rhi.cmd_set_pipeline(cmd, _h0_pipeline);

    struct OceanComputePushConstants {
      uint32_t outH0Index;
      uint32_t inH0Index;
      uint32_t outHtIndex;    // ht_Dx_Dz
      uint32_t outDxDzIndex;  // to store Choppy X and Z
      uint32_t outDerivSpec0Index;
      uint32_t outDerivSpec1Index;
      uint32_t outDerivSpec2Index;
      uint32_t N;
      uint32_t seed;
      float L;
      float A;
      float windDirX;
      float windDirY;
      float windSpeed;
      float waterDepth;
      float jonswapGamma;
      float directionalSpread;
      float time;
      float kMin;
      float kMax;
      float kMinSoft;
      float kMaxSoft;
      float choppiness;
    };

    float cascade_k_min[k_cascade_count] = {0.0f, 0.0f, 0.0f};
    float cascade_k_max[k_cascade_count] = {1.0e30f, 1.0e30f, 1.0e30f};
    float cascade_k_min_soft[k_cascade_count] = {0.0f, 0.0f, 0.0f};
    float cascade_k_max_soft[k_cascade_count] = {0.0f, 0.0f, 0.0f};
    calculate_cascade_band_limits(_parameters, cascade_k_min, cascade_k_max, cascade_k_min_soft, cascade_k_max_soft);

    float2 wind_dir = {_parameters.wind_direction_x, _parameters.wind_direction_y};
    float wind_dir_len_sq = dot(wind_dir, wind_dir);
    if (wind_dir_len_sq < 1.0e-6f) {
      wind_dir = {1.0f, 0.0f};
    } else {
      wind_dir = wind_dir / sqrtf(wind_dir_len_sq);
    }

    static constexpr float k_max_spectrum_amplitude = 1.0e16f;
    float cascade_spectrum_amplitude[k_cascade_count] = {0.0f, 0.0f, 0.0f};
    bool spectrum_amplitude_clamped = false;
    float max_unclamped_spectrum_amplitude = 0.0f;
    for (uint32_t c = 0; c < k_cascade_count; ++c) {
      float target_rms_height = target_cascade_rms[c];
      if (target_rms_height <= 0.0f) {
        cascade_spectrum_amplitude[c] = 0.0f;
        continue;
      }

      float base_rms_height = estimate_base_rms_height(_fft_resolution, _parameters.cascade_lengths[c], wind_dir, _parameters.wind_speed, _parameters.water_depth,
        _parameters.jonswap_gamma, _parameters.directional_spread, cascade_k_min[c], cascade_k_max[c], cascade_k_min_soft[c], cascade_k_max_soft[c]);
      if ((std::isfinite(base_rms_height) == false) || (base_rms_height <= 1.0e-6f)) {
        base_rms_height = estimate_base_rms_height(_fft_resolution, _parameters.cascade_lengths[c], wind_dir, _parameters.wind_speed, _parameters.water_depth,
          _parameters.jonswap_gamma, _parameters.directional_spread, 0.0f, 1.0e30f, 0.0f, 0.0f);
      }
      if ((std::isfinite(base_rms_height) == false) || (base_rms_height <= 1.0e-6f)) {
        base_rms_height = 1.0e-3f;
      }

      float amplitude_scale = target_rms_height / base_rms_height;
      if (std::isfinite(amplitude_scale) == false) {
        amplitude_scale = 0.0f;
      }
      float spectrum_amplitude = amplitude_scale * amplitude_scale;
      if (std::isfinite(spectrum_amplitude) == false) {
        spectrum_amplitude = 0.0f;
      }
      max_unclamped_spectrum_amplitude = max(max_unclamped_spectrum_amplitude, spectrum_amplitude);
      if (spectrum_amplitude > k_max_spectrum_amplitude) {
        spectrum_amplitude_clamped = true;
      }
      cascade_spectrum_amplitude[c] = min(spectrum_amplitude, k_max_spectrum_amplitude);
    }

    if (spectrum_amplitude_clamped) {
      log::warning("Ocean: spectrum amplitude clamped (max requested %.3e, cap %.3e). Consider raising FFT resolution for large Hs.", max_unclamped_spectrum_amplitude,
        k_max_spectrum_amplitude);
    }

    for (uint32_t c = 0; c < k_cascade_count; ++c) {
      OceanComputePushConstants pc = {};
      pc.outH0Index = get_bindless_descriptor_index(_h0_texture[c]);
      pc.outDerivSpec0Index = 0u;
      pc.outDerivSpec1Index = 0u;
      pc.outDerivSpec2Index = 0u;
      pc.N = TESS;
      pc.seed = k_cascade_rng_seed[c];
      pc.L = _parameters.cascade_lengths[c];
      pc.A = cascade_spectrum_amplitude[c];
      pc.windDirX = _parameters.wind_direction_x;
      pc.windDirY = _parameters.wind_direction_y;
      pc.windSpeed = _parameters.wind_speed;
      pc.waterDepth = _parameters.water_depth;
      pc.jonswapGamma = _parameters.jonswap_gamma;
      pc.directionalSpread = _parameters.directional_spread;
      pc.kMin = cascade_k_min[c];
      pc.kMax = cascade_k_max[c];
      pc.kMinSoft = cascade_k_min_soft[c];
      pc.kMaxSoft = cascade_k_max_soft[c];
      pc.choppiness = _parameters.choppiness;
      rhi.cmd_push_constants(cmd, &pc, sizeof(OceanComputePushConstants), 0);

      RHIDispatchDesc dispatch = {};
      dispatch.group_count_x = dispatch_group_count_x;
      dispatch.group_count_y = dispatch_group_count_y;
      dispatch.group_count_z = 1;
      rhi.cmd_dispatch(cmd, dispatch);
      rhi.cmd_texture_barrier(cmd, _h0_texture[c], RHIResourceState::General, RHIResourceState::General);
    }
    _h0_generated = true;
  }

  if (_update_spectrum_pipeline.valid()) {
    rhi.cmd_set_pipeline(cmd, _update_spectrum_pipeline);

    struct OceanComputePushConstants {
      uint32_t outH0Index;
      uint32_t inH0Index;
      uint32_t outHtIndex;    // ht_Dx_Dz
      uint32_t outDxDzIndex;  // to store Choppy X and Z
      uint32_t outDerivSpec0Index;
      uint32_t outDerivSpec1Index;
      uint32_t outDerivSpec2Index;
      uint32_t N;
      uint32_t seed;
      float L;
      float A;
      float windDirX;
      float windDirY;
      float windSpeed;
      float waterDepth;
      float jonswapGamma;
      float directionalSpread;
      float time;
      float kMin;
      float kMax;
      float kMinSoft;
      float kMaxSoft;
      float choppiness;
    };

    float scaled_time = time * _parameters.time_scale;
    for (uint32_t c = 0; c < k_cascade_count; ++c) {
      OceanComputePushConstants pc = {};
      pc.inH0Index = get_bindless_descriptor_index(_h0_texture[c]);
      pc.outHtIndex = get_bindless_descriptor_index(_ht_texture[c]);
      pc.outDxDzIndex = get_bindless_descriptor_index(_dxdz_texture[c]);
      pc.outDerivSpec0Index = get_bindless_descriptor_index(_deriv_spec_0_texture[c]);
      pc.outDerivSpec1Index = get_bindless_descriptor_index(_deriv_spec_1_texture[c]);
      pc.outDerivSpec2Index = get_bindless_descriptor_index(_deriv_spec_2_texture[c]);
      pc.N = TESS;
      pc.seed = 0u;
      pc.L = _parameters.cascade_lengths[c];
      pc.windDirX = _parameters.wind_direction_x;
      pc.windDirY = _parameters.wind_direction_y;
      pc.windSpeed = _parameters.wind_speed;
      pc.waterDepth = _parameters.water_depth;
      pc.jonswapGamma = _parameters.jonswap_gamma;
      pc.directionalSpread = _parameters.directional_spread;
      pc.time = scaled_time;
      pc.kMin = 0.0f;
      pc.kMax = 0.0f;
      pc.kMinSoft = 0.0f;
      pc.kMaxSoft = 0.0f;
      pc.choppiness = _parameters.choppiness;
      rhi.cmd_push_constants(cmd, &pc, sizeof(OceanComputePushConstants), 0);

      RHIDispatchDesc dispatch = {};
      dispatch.group_count_x = dispatch_group_count_x;
      dispatch.group_count_y = dispatch_group_count_y;
      dispatch.group_count_z = 1;
      rhi.cmd_dispatch(cmd, dispatch);
      rhi.cmd_texture_barrier(cmd, _ht_texture[c], RHIResourceState::General, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _dxdz_texture[c], RHIResourceState::General, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_0_texture[c], RHIResourceState::General, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_1_texture[c], RHIResourceState::General, RHIResourceState::General);
      rhi.cmd_texture_barrier(cmd, _deriv_spec_2_texture[c], RHIResourceState::General, RHIResourceState::General);
    }
  }

  if (_fft_pipeline.valid()) {
    rhi.cmd_set_pipeline(cmd, _fft_pipeline);

    struct FFTPushConstants {
      uint32_t inTexIndex;
      uint32_t outTexIndex;
      uint32_t N;
      uint32_t pass;
      uint32_t direction;  // 0 for horizontal, 1 for vertical
      uint32_t log2N;
    };

    RHIDispatchDesc dispatch = {};
    dispatch.group_count_x = dispatch_group_count_x;
    dispatch.group_count_y = dispatch_group_count_y;
    dispatch.group_count_z = 1;

    auto run_fft_2d = [&](RHITexture src_tex, RHITexture ping_tex, FFTPushConstants& io_fft_pc) -> bool {
      bool pingpong_local = false;
      io_fft_pc.direction = 0;
      for (uint32_t i = 0; i < io_fft_pc.log2N; ++i) {
        io_fft_pc.pass = i;
        io_fft_pc.inTexIndex = get_bindless_descriptor_index((pingpong_local == false) ? src_tex : ping_tex);
        io_fft_pc.outTexIndex = get_bindless_descriptor_index((pingpong_local == false) ? ping_tex : src_tex);
        pingpong_local = (pingpong_local == false);
        rhi.cmd_push_constants(cmd, &io_fft_pc, sizeof(FFTPushConstants), 0);
        rhi.cmd_dispatch(cmd, dispatch);
        rhi.cmd_texture_barrier(cmd, src_tex, RHIResourceState::General, RHIResourceState::General);
        rhi.cmd_texture_barrier(cmd, ping_tex, RHIResourceState::General, RHIResourceState::General);
      }

      io_fft_pc.direction = 1;
      for (uint32_t i = 0; i < io_fft_pc.log2N; ++i) {
        io_fft_pc.pass = i;
        io_fft_pc.inTexIndex = get_bindless_descriptor_index((pingpong_local == false) ? src_tex : ping_tex);
        io_fft_pc.outTexIndex = get_bindless_descriptor_index((pingpong_local == false) ? ping_tex : src_tex);
        pingpong_local = (pingpong_local == false);
        rhi.cmd_push_constants(cmd, &io_fft_pc, sizeof(FFTPushConstants), 0);
        rhi.cmd_dispatch(cmd, dispatch);
        rhi.cmd_texture_barrier(cmd, src_tex, RHIResourceState::General, RHIResourceState::General);
        rhi.cmd_texture_barrier(cmd, ping_tex, RHIResourceState::General, RHIResourceState::General);
      }

      return pingpong_local;
    };

    for (uint32_t c = 0; c < k_cascade_count; ++c) {
      rhi.cmd_set_pipeline(cmd, _fft_pipeline);
      FFTPushConstants fft_pc = {};
      fft_pc.N = TESS;
      fft_pc.log2N = fft_log2_n;

      bool pingpong = run_fft_2d(_ht_texture[c], _ht_pingpong[c], fft_pc);
      bool pingpong_dxdz = run_fft_2d(_dxdz_texture[c], _dxdz_pingpong[c], fft_pc);
      bool pingpong_deriv_0 = run_fft_2d(_deriv_spec_0_texture[c], _deriv_spec_0_pingpong[c], fft_pc);
      bool pingpong_deriv_1 = run_fft_2d(_deriv_spec_1_texture[c], _deriv_spec_1_pingpong[c], fft_pc);
      bool pingpong_deriv_2 = run_fft_2d(_deriv_spec_2_texture[c], _deriv_spec_2_pingpong[c], fft_pc);

      // Assemble Displacements and Surface Derivatives
      if (_assemble_pipeline.valid()) {
        rhi.cmd_set_pipeline(cmd, _assemble_pipeline);
        struct AssemblePushConstants {
          uint32_t inHtIndex;
          uint32_t inDxDzIndex;
          uint32_t inDerivSpec0Index;
          uint32_t inDerivSpec1Index;
          uint32_t inDerivSpec2Index;
          uint32_t outDispIndex;
          uint32_t outDerivUIndex;
          uint32_t outDerivVIndex;
          uint32_t outSlopeMetricIndex;
          uint32_t N;
          float lambda;
        } assemble_pc = {};

        // Final FFT results are in the ping-pong toggled texture
        assemble_pc.inHtIndex = get_bindless_descriptor_index((pingpong == false) ? _ht_texture[c] : _ht_pingpong[c]);
        assemble_pc.inDxDzIndex = get_bindless_descriptor_index((pingpong_dxdz == false) ? _dxdz_texture[c] : _dxdz_pingpong[c]);
        assemble_pc.inDerivSpec0Index = get_bindless_descriptor_index((pingpong_deriv_0 == false) ? _deriv_spec_0_texture[c] : _deriv_spec_0_pingpong[c]);
        assemble_pc.inDerivSpec1Index = get_bindless_descriptor_index((pingpong_deriv_1 == false) ? _deriv_spec_1_texture[c] : _deriv_spec_1_pingpong[c]);
        assemble_pc.inDerivSpec2Index = get_bindless_descriptor_index((pingpong_deriv_2 == false) ? _deriv_spec_2_texture[c] : _deriv_spec_2_pingpong[c]);
        assemble_pc.outDispIndex = get_bindless_descriptor_index(_displacement_map[c]);
        assemble_pc.outDerivUIndex = get_bindless_descriptor_index(_surface_derivative_u_map[c]);
        assemble_pc.outDerivVIndex = get_bindless_descriptor_index(_surface_derivative_v_map[c]);
        assemble_pc.outSlopeMetricIndex = get_bindless_descriptor_index(_slope_metric_map[c]);
        assemble_pc.N = TESS;
        assemble_pc.lambda = -_parameters.choppiness;

        rhi.cmd_push_constants(cmd, &assemble_pc, sizeof(AssemblePushConstants), 0);
        rhi.cmd_dispatch(cmd, dispatch);
        rhi.cmd_texture_barrier(cmd, _displacement_map[c], RHIResourceState::General, RHIResourceState::ShaderReadOnly);
#if defined(ETX_PLATFORM_APPLE)
        rhi.cmd_texture_barrier(cmd, _surface_derivative_u_map[c], RHIResourceState::General, RHIResourceState::ShaderReadOnly);
        rhi.cmd_texture_barrier(cmd, _surface_derivative_v_map[c], RHIResourceState::General, RHIResourceState::ShaderReadOnly);
        rhi.cmd_texture_barrier(cmd, _slope_metric_map[c], RHIResourceState::General, RHIResourceState::ShaderReadOnly);
#else
        rhi.cmd_generate_mipmaps(cmd, _surface_derivative_u_map[c]);
        rhi.cmd_generate_mipmaps(cmd, _surface_derivative_v_map[c]);
        rhi.cmd_generate_mipmaps(cmd, _slope_metric_map[c]);
#endif
      }
    }
  }
}

void Ocean::prepare_render_draw_state(RHIContext& rhi, RHICommandBuffer cmd, const float4x4& view_proj, const float4x4& inv_view_proj, const float3& camera_position,
  RHITexture envmap_texture, bool envmap_equal_area_mapping, RHITexture scene_opaque_color_texture, RHITexture wave_thickness_min_texture, RHITexture wave_thickness_max_texture,
  RHITexture foam_history_texture, uint32_t viewport_width, uint32_t viewport_height) {
  uint32_t frame_index = rhi.get_current_frame_index();
  RHIBindlessHandle frame_instance_buffer = _instance_buffer[frame_index];
  RHIBindlessHandle frame_settings_buffer = _settings_buffer[frame_index];

  struct PushConstants {
    uint32_t vb_index;
    uint32_t sampler_index;
    uint32_t instance_buffer_index;
    uint32_t settings_buffer_index;
    uint32_t disp_map_index[4];
    float4 camera_pos;
    float4x4 vp_matrix;
  } pc = {};

  OceanRenderSettings render_settings = {};
  render_settings.envmap_index = envmap_texture.valid() ? get_bindless_descriptor_index(envmap_texture) : 0;
  render_settings.scene_color_index = scene_opaque_color_texture.valid() ? get_bindless_descriptor_index(scene_opaque_color_texture) : 0;
  render_settings.env_sampler_index = rhi.get_sampler_index(RHISamplerType::LinearRepeatUClampV);
  render_settings.scene_sampler_index = rhi.get_sampler_index(RHISamplerType::LinearClamp);
  render_settings.stitch_transition_cells = _parameters.stitch_transition_cells;
  render_settings.mip_color_mix = _parameters.mip_color_mix;
  render_settings.mip_color_enable = _parameters.mip_color_enable;
  render_settings._padding0 = envmap_equal_area_mapping ? 1.0f : 0.0f;
  render_settings.cascade_lengths = {_parameters.cascade_lengths[0], _parameters.cascade_lengths[1], _parameters.cascade_lengths[2], 0.0f};
  float cascade_weight[3] = {0.0f, 0.0f, 0.0f};
  calculate_effective_cascade_weights(_parameters, cascade_weight);
  render_settings.cascade_weights = {cascade_weight[0], cascade_weight[1], cascade_weight[2], 0.0f};
  render_settings.clipmap_center_offset = {_clipmap_center_offset.x, _clipmap_center_offset.y, 0.0f, 0.0f};
  int32_t surface_normal_visualize_mode = max(0, min(_parameters.surface_normal_visualize_mode, 7));
  bool draw_wireframe = _parameters.wireframe_enable;
  if ((draw_wireframe) && (_wire_pipeline.valid() == false)) {
    draw_wireframe = false;
  }
  render_settings.debug_view = {draw_wireframe ? 1.0f : 0.0f, static_cast<float>(_patch_resolution), 0.0f, static_cast<float>(_clipmap_levels - 1)};
  float surface_normal_strength = min(max(0.0f, _parameters.surface_normal_strength), 1.0f);
  render_settings.surface_normal_controls = {_parameters.surface_normal_shading_enable ? 1.0f : 0.0f, surface_normal_strength, static_cast<float>(surface_normal_visualize_mode),
    0.0f};
  render_settings.wireframe_color = {_parameters.wireframe_color.x, _parameters.wireframe_color.y, _parameters.wireframe_color.z, 1.0f};
  float inv_w = (viewport_width > 0u) ? (1.0f / static_cast<float>(viewport_width)) : 0.0f;
  float inv_h = (viewport_height > 0u) ? (1.0f / static_cast<float>(viewport_height)) : 0.0f;
  render_settings.screen_size = {static_cast<float>(viewport_width), static_cast<float>(viewport_height), inv_w, inv_h};
  render_settings.inv_view_proj = inv_view_proj;
  for (uint32_t i = 0u; i < k_cascade_count; ++i) {
    render_settings.surface_deriv_u_index[i] = get_bindless_descriptor_index(_surface_derivative_u_map[i]);
    render_settings.surface_deriv_v_index[i] = get_bindless_descriptor_index(_surface_derivative_v_map[i]);
    render_settings.slope_metric_index[i] = get_bindless_descriptor_index(_slope_metric_map[i]);
  }
  render_settings.water_optics_0 = {_parameters.water_ior, _parameters.env_reflection_intensity, _parameters.refract_distortion_scale,
    _parameters.physical_render_mode ? 1.0f : 0.0f};
  render_settings.water_optics_1 = {max(0.0f, _parameters.optical_depth_m), min(max(_parameters.unresolved_slope_roughness, 0.0f), 1.0f),
    max(_parameters.specular_aa_strength, 0.0f), 0.0f};
  render_settings.water_absorption = {max(0.0f, _parameters.absorption_coeff_rgb.x), max(0.0f, _parameters.absorption_coeff_rgb.y), max(0.0f, _parameters.absorption_coeff_rgb.z),
    0.0f};
  render_settings.water_scattering = {max(0.0f, _parameters.scattering_coeff_rgb.x), max(0.0f, _parameters.scattering_coeff_rgb.y), max(0.0f, _parameters.scattering_coeff_rgb.z),
    0.0f};
  render_settings.sun_direction_enable = {_parameters.sun_direction.x, _parameters.sun_direction.y, _parameters.sun_direction.z, _parameters.sun_lighting_enable ? 1.0f : 0.0f};
  render_settings.sun_radiance = {_parameters.sun_radiance.x, _parameters.sun_radiance.y, _parameters.sun_radiance.z, 0.0f};
  if (_has_previous_view_proj) {
    render_settings.prev_view_proj = _previous_view_proj;
  } else {
    render_settings.prev_view_proj = view_proj;
  }
  int32_t water_debug_visualize_mode = max(0, min(_parameters.water_debug_visualize_mode, 2));
  render_settings.wave_thickness_min_index = wave_thickness_min_texture.valid() ? get_bindless_descriptor_index(wave_thickness_min_texture) : 0u;
  render_settings.wave_thickness_max_index = wave_thickness_max_texture.valid() ? get_bindless_descriptor_index(wave_thickness_max_texture) : 0u;
  render_settings.wave_thickness_sampler_index = rhi.get_sampler_index(RHISamplerType::NearestClamp);
  render_settings._padding1 = 0u;
  render_settings.foam_history_index = foam_history_texture.valid() ? get_bindless_descriptor_index(foam_history_texture) : 0u;
  render_settings.foam_detail_index = _foam_detail_texture.valid() ? get_bindless_descriptor_index(_foam_detail_texture) : 0u;
  render_settings.foam_sampler_index = rhi.get_sampler_index(RHISamplerType::LinearClamp);
  render_settings.aeration_detail_index = _aeration_detail_texture.valid() ? get_bindless_descriptor_index(_aeration_detail_texture) : 0u;
  render_settings.wave_thickness_controls = {max(_parameters.wave_thickness_path_scale, 0.0f), static_cast<float>(water_debug_visualize_mode),
    max(_parameters.wave_thickness_debug_max_m, 1.0e-3f), 0.0f};
  render_settings.foam_controls_0 = {_parameters.foam_enable ? 1.0f : 0.0f, max(_parameters.foam_strength, 0.0f), max(_parameters.foam_slope_start, 0.0f),
    max(_parameters.foam_slope_end, 0.0f)};
  render_settings.foam_controls_1 = {max(_parameters.foam_thickness_start_m, 0.0f), max(_parameters.foam_thickness_end_m, 0.0f),
    min(max(_parameters.foam_surface_coverage, 0.0f), 1.0f), min(max(_parameters.foam_specular_suppression, 0.0f), 1.0f)};
  render_settings.foam_controls_2 = {max(_parameters.foam_diffuse_gain, 0.0f), max(_parameters.foam_backlight_gain, 0.0f), max(_parameters.foam_aeration_scatter_scale, 0.0f),
    max(_parameters.foam_aeration_absorption_scale, 0.0f)};
  render_settings.foam_color = {min(max(_parameters.foam_albedo.x, 0.0f), 4.0f), min(max(_parameters.foam_albedo.y, 0.0f), 4.0f), min(max(_parameters.foam_albedo.z, 0.0f), 4.0f),
    min(max(_parameters.foam_alpha_threshold, 0.0f), 1.0f)};
  render_settings.foam_temporal_controls = {min(max(_parameters.foam_history_decay, 0.0f), 1.0f), max(_parameters.foam_history_gain, 0.0f),
    max(_parameters.foam_history_bias, 0.0f), 0.0f};
  render_settings.foam_detail_controls = {max(_parameters.foam_detail_scale_1, 1.0e-4f), max(_parameters.foam_detail_scale_2, 1.0e-4f),
    min(max(_parameters.foam_detail_mix, 0.0f), 1.0f), max(_parameters.foam_detail_contrast, 0.0f)};
  float2 wind_dir = {_parameters.wind_direction_x, _parameters.wind_direction_y};
  float wind_dir_len2 = dot(wind_dir, wind_dir);
  if (wind_dir_len2 > 1.0e-8f) {
    wind_dir /= sqrtf(wind_dir_len2);
  } else {
    wind_dir = {1.0f, 0.0f};
  }
  render_settings.foam_flow_controls = {wind_dir.x, wind_dir.y, _last_update_time, max(_parameters.foam_detail_scroll_speed, 0.0f)};
  rhi.device().update_buffer(frame_settings_buffer, &render_settings, sizeof(OceanRenderSettings));

  pc.vb_index = get_bindless_descriptor_index(_vertex_buffer);
  pc.sampler_index = rhi.get_sampler_index(RHISamplerType::LinearRepeat);
  pc.instance_buffer_index = get_bindless_descriptor_index(frame_instance_buffer);
  pc.settings_buffer_index = get_bindless_descriptor_index(frame_settings_buffer);
  for (uint32_t i = 0; i < k_cascade_count; ++i) {
    pc.disp_map_index[i] = get_bindless_descriptor_index(_displacement_map[i]);
  }
  pc.camera_pos = {camera_position.x, camera_position.y, camera_position.z, 0.0f};
  pc.vp_matrix = view_proj;

  rhi.cmd_push_constants(cmd, &pc, sizeof(PushConstants), 0);
}

void Ocean::draw_wave_thickness_prepass(RHIContext& rhi, RHICommandBuffer cmd, const float4x4& view_proj, const float4x4& inv_view_proj, const float3& camera_position,
  bool max_blend_pass, uint32_t viewport_width, uint32_t viewport_height) {
  uint32_t frame_index = rhi.get_current_frame_index();
  RHIBindlessHandle frame_instance_buffer = _instance_buffer[frame_index];
  RHIBindlessHandle frame_settings_buffer = _settings_buffer[frame_index];
  if ((_vertex_buffer.valid() == false) || (_index_buffer.valid() == false) || (frame_settings_buffer.valid() == false) || (frame_instance_buffer.valid() == false)) {
    return;
  }

  RHIPipeline pipeline = _thickness_min_pipeline;
  if (max_blend_pass) {
    pipeline = _thickness_max_pipeline;
  }
  if (pipeline.valid() == false) {
    return;
  }

  rhi.cmd_set_pipeline(cmd, pipeline);
  prepare_render_draw_state(rhi, cmd, view_proj, inv_view_proj, camera_position, {}, false, {}, {}, {}, {}, viewport_width, viewport_height);

  RHIIndexedDrawDesc draw = {
    .index_count = _index_count,
    .instance_count = _instance_count,
    .index_type = RHIIndexType::UInt32,
  };
  rhi.cmd_draw_indexed(cmd, draw, _index_buffer);
}

void Ocean::draw_foam_history_prepass(RHIContext& rhi, RHICommandBuffer cmd, const float4x4& view_proj, const float4x4& inv_view_proj, const float3& camera_position,
  RHITexture wave_thickness_min_texture, RHITexture wave_thickness_max_texture, RHITexture prev_foam_history_texture, uint32_t viewport_width, uint32_t viewport_height) {
  uint32_t frame_index = rhi.get_current_frame_index();
  RHIBindlessHandle frame_instance_buffer = _instance_buffer[frame_index];
  RHIBindlessHandle frame_settings_buffer = _settings_buffer[frame_index];
  if ((_vertex_buffer.valid() == false) || (_index_buffer.valid() == false) || (frame_settings_buffer.valid() == false) || (frame_instance_buffer.valid() == false)) {
    return;
  }
  if (_foam_history_pipeline.valid() == false) {
    return;
  }

  rhi.cmd_set_pipeline(cmd, _foam_history_pipeline);
  prepare_render_draw_state(rhi, cmd, view_proj, inv_view_proj, camera_position, {}, false, {}, wave_thickness_min_texture, wave_thickness_max_texture, prev_foam_history_texture,
    viewport_width, viewport_height);

  RHIIndexedDrawDesc draw = {
    .index_count = _index_count,
    .instance_count = _instance_count,
    .index_type = RHIIndexType::UInt32,
  };
  rhi.cmd_draw_indexed(cmd, draw, _index_buffer);
}

void Ocean::draw(RHIContext& rhi, RHICommandBuffer cmd, const float4x4& view_proj, const float4x4& inv_view_proj, const float3& camera_position, RHITexture envmap_texture,
  bool envmap_equal_area_mapping, RHITexture scene_opaque_color_texture, RHITexture wave_thickness_min_texture, RHITexture wave_thickness_max_texture,
  RHITexture foam_history_texture, uint32_t viewport_width, uint32_t viewport_height) {
  uint32_t frame_index = rhi.get_current_frame_index();
  RHIBindlessHandle frame_instance_buffer = _instance_buffer[frame_index];
  RHIBindlessHandle frame_settings_buffer = _settings_buffer[frame_index];
  if ((valid() == false) || (_vertex_buffer.valid() == false) || (_index_buffer.valid() == false) || (frame_settings_buffer.valid() == false) ||
      (frame_instance_buffer.valid() == false)) {
    return;
  }
  bool draw_wireframe = _parameters.wireframe_enable;
  if ((draw_wireframe) && (_wire_pipeline.valid() == false)) {
    draw_wireframe = false;
  }
  rhi.cmd_set_pipeline(cmd, draw_wireframe ? _wire_pipeline : _pipeline);
  prepare_render_draw_state(rhi, cmd, view_proj, inv_view_proj, camera_position, envmap_texture, envmap_equal_area_mapping, scene_opaque_color_texture, wave_thickness_min_texture,
    wave_thickness_max_texture, foam_history_texture, viewport_width, viewport_height);

  RHIIndexedDrawDesc draw = {
    .index_count = _index_count,
    .instance_count = _instance_count,
    .index_type = RHIIndexType::UInt32,
  };
  rhi.cmd_draw_indexed(cmd, draw, _index_buffer);
  _previous_view_proj = view_proj;
  _has_previous_view_proj = true;
}

void Ocean::effective_cascade_weights(float* out_weights) const {
  if (out_weights == nullptr) {
    return;
  }
  calculate_effective_cascade_weights(_parameters, out_weights);
}

}  // namespace etx
