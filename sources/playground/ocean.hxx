#pragma once

#include <etx/render/interop/interop.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/render/shared/base.hxx>
#include <cstdint>

namespace etx {

struct OceanParameters {
  float wind_direction_x = 1.0f;
  float wind_direction_y = 0.0f;
  float wind_speed = 10.5f;
  float water_depth = 700.0f;
  float jonswap_gamma = 3.2f;
  float directional_spread = 4.0f;
  bool significant_wave_height_enable = true;
  float significant_wave_height = 2.8f;
  bool debug_cascade_overrides_enable = false;
  float time_scale = 1.0f;
  float choppiness = 1.35f;
  float lod_forward_bias = 0.0f;
  bool lock_lods = false;
  bool spectral_band_limit_enable = true;
  float stitch_transition_cells = 8.0f;
  float mip_color_mix = 0.35f;
  float mip_color_enable = 0.0f;
  int32_t surface_normal_visualize_mode = 0;
  bool surface_normal_shading_enable = true;
  float surface_normal_strength = 1.0f;
  bool physical_render_mode = true;
  float water_ior = 1.333f;
  float3 absorption_coeff_rgb = {0.28f, 0.06f, 0.02f};
  float3 scattering_coeff_rgb = {0.01f, 0.03f, 0.06f};
  float env_reflection_intensity = 1.0f;
  float optical_depth_m = 8.0f;
  float unresolved_slope_roughness = 0.04f;
  float specular_aa_strength = 0.35f;
  float refract_distortion_scale = 0.02f;
  bool sun_lighting_enable = true;
  float3 sun_direction = {0.35f, 0.65f, 0.67f};
  float3 sun_radiance = {12.0f, 11.5f, 10.5f};
  bool wireframe_enable = false;
  float3 wireframe_color = {0.05f, 0.05f, 0.05f};
  int32_t solo_cascade = -1;
  bool cascade_enable[3] = {true, true, true};
  float cascade_render_weight[3] = {1.0f, 1.0f, 1.0f};
  float cascade_lengths[3] = {320.0f, 80.0f, 20.0f};
  float cascade_amplitudes[3] = {0.50f, 0.18f, 0.06f};
};

struct Ocean {
  static constexpr uint32_t k_cascade_count = 3;

  void init(RHIContext& rhi, RHITextureFormat color_format, RHITextureFormat depth_format);
  void cleanup(RHIContext& rhi);
  void update(RHIContext& rhi, RHICommandBuffer cmd, float time, const float3& camera_position, const float3& camera_direction, float fov, uint32_t viewport_width,
    uint32_t viewport_height);
  void draw(RHIContext& rhi, RHICommandBuffer cmd, const float4x4& view_proj, const float4x4& inv_view_proj, const float3& camera_position, RHITexture envmap_texture,
    RHITexture scene_opaque_color_texture, uint32_t viewport_width, uint32_t viewport_height);

  OceanParameters& parameters() {
    return _parameters;
  }

  const OceanParameters& parameters() const {
    return _parameters;
  }

  void force_regenerate_spectrum() {
    _h0_generated = false;
  }

  uint32_t patch_resolution() const {
    return _patch_resolution;
  }

  uint32_t clipmap_levels() const {
    return _clipmap_levels;
  }

  float base_patch_size() const {
    return _base_patch_size;
  }

  uint32_t fft_resolution() const {
    return _fft_resolution;
  }

  float base_vertex_spacing() const {
    return _base_patch_size / static_cast<float>(_patch_resolution);
  }

  float coverage_radius() const {
    return _base_patch_size * static_cast<float>(1u << _clipmap_levels);
  }

  uint32_t instance_count() const {
    return _instance_count;
  }

  float resolved_cascade_rms(uint32_t index) const {
    return (index < k_cascade_count) ? _resolved_cascade_rms[index] : 0.0f;
  }

  void effective_cascade_weights(float* out_weights) const;

  RHITexture displacement_texture(uint32_t index) const {
    return (index < k_cascade_count) ? _displacement_map[index] : RHITexture{};
  }

  bool valid() const {
    return _pipeline.valid();
  }

 private:
  RHIBindlessHandle _vertex_buffer;
  RHIBindlessHandle _index_buffer;
  RHIBindlessHandle _instance_buffer[kRHIMaxFrames] = {};
  RHIBindlessHandle _settings_buffer[kRHIMaxFrames] = {};
  RHIPipeline _pipeline;
  RHIPipeline _wire_pipeline;
  RHIPipeline _h0_pipeline;
  RHIPipeline _update_spectrum_pipeline;
  RHIPipeline _fft_pipeline;
  RHIPipeline _assemble_pipeline;
  RHITexture _displacement_map[3];
  RHITexture _surface_derivative_u_map[3];
  RHITexture _surface_derivative_v_map[3];
  RHITexture _slope_metric_map[3];
  RHITexture _h0_texture[3];
  RHITexture _ht_texture[3];
  RHITexture _dxdz_texture[3];
  RHITexture _ht_pingpong[3];
  RHITexture _dxdz_pingpong[3];
  uint32_t _index_count = 0;
  uint32_t _instance_count = 0;
  uint32_t _instance_capacity = 0;
  uint32_t _patch_resolution = 128;
  uint32_t _clipmap_levels = 11;
  uint32_t _fft_resolution = 256;
  float _base_patch_size = 10.0f;
  float2 _clipmap_center_offset = {0.0f, 0.0f};
  OceanParameters _parameters = {};
  OceanParameters _previous_spectrum_parameters = {};
  bool _has_previous_spectrum_parameters = false;
  bool _h0_generated = false;
  bool _resources_initialized = false;
  float _resolved_cascade_rms[k_cascade_count] = {0.0f, 0.0f, 0.0f};
};

}  // namespace etx
