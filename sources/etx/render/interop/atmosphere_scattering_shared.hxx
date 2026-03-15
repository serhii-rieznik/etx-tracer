#pragma once

#include "interop.hxx"

ETX_STATIC_CONST float kScatteringPlanetRadius = 6371e+3f;
ETX_STATIC_CONST float kScatteringAtmosphereRadius = 120e+3f;
ETX_STATIC_CONST float kScatteringOuterSphereRadius = kScatteringPlanetRadius + kScatteringAtmosphereRadius;
ETX_STATIC_CONST float kScatteringDeltaDensity = 0.01f;
ETX_STATIC_CONST float kScatteringRayleighDensityScale = 7994.0f;
ETX_STATIC_CONST float kScatteringMieDensityScale = 1200.0f;
ETX_STATIC_CONST uint32_t kScatteringOpticalDepthMaxSteps = (1u << 14u);
ETX_STATIC_CONST uint32_t kAtmosphereSkyGpuLightStride = 32u;
ETX_STATIC_CONST uint32_t kAtmosphereSkyGpuLightDirectionOffset = 0u;
ETX_STATIC_CONST uint32_t kAtmosphereSkyGpuLightAngularSizeOffset = 12u;
ETX_STATIC_CONST uint32_t kAtmosphereSkyGpuLightIntensityScaleOffset = 16u;
ETX_STATIC_CONST uint32_t kAtmosphereSkyGpuLightEmissionSpectrumIndexOffset = 20u;

struct ScatteringDensityAndDerivative {
  float3 density ETX_INIT({});
  float3 derivative ETX_INIT({});
};

struct ETX_ALIGNED AtmosphereSkyGpuParameters {
  float altitude ETX_INIT(0.0f);
  float anisotropy ETX_INIT(0.0f);
  float rayleigh_scale ETX_INIT(0.0f);
  float mie_scale ETX_INIT(0.0f);
  float ozone_scale ETX_INIT(0.0f);
  uint32_t pad0 ETX_INIT(0u);
  uint32_t pad1 ETX_INIT(0u);
  uint32_t pad2 ETX_INIT(0u);
};

struct AtmosphereSkyPassFlags {
  enum : uint32_t {
    PrimaryScattering = 1u << 0u,
    SecondaryScattering = 1u << 1u,
  };
};

struct ETX_ALIGNED AtmosphereSkyGpuLight {
  float3 direction ETX_INIT({});
  float angular_size ETX_INIT(0.0f);
  float intensity_scale ETX_INIT(0.0f);
  uint32_t emission_spectrum_index ETX_INIT(0u);
  uint32_t pad0 ETX_INIT(0u);
  uint32_t pad1 ETX_INIT(0u);
};

struct AtmosphereSkyPushConstants {
  uint32_t output_texture_index ETX_INIT(0u);
  uint32_t optical_depth_texture_index ETX_INIT(0u);
  uint32_t optical_depth_sampler_index ETX_INIT(0u);
  uint32_t aux_texture_index ETX_INIT(0u);
  uint32_t lights_buffer_index ETX_INIT(0u);
  uint32_t spectra_buffer_index ETX_INIT(0u);
  uint32_t width ETX_INIT(0u);
  uint32_t height ETX_INIT(0u);
  uint32_t light_count ETX_INIT(0u);
  uint32_t pass_flags ETX_INIT(0u);
  float average_color_x ETX_INIT(0.0f);
  float average_color_y ETX_INIT(0.0f);
  float average_color_z ETX_INIT(0.0f);
  float pad0 ETX_INIT(0.0f);
  float atmosphere_altitude ETX_INIT(0.0f);
  float atmosphere_anisotropy ETX_INIT(0.0f);
  float atmosphere_rayleigh_scale ETX_INIT(0.0f);
  float atmosphere_mie_scale ETX_INIT(0.0f);
  float atmosphere_ozone_scale ETX_INIT(0.0f);
  uint32_t atmosphere_pad0 ETX_INIT(0u);
  uint32_t atmosphere_pad1 ETX_INIT(0u);
  uint32_t atmosphere_pad2 ETX_INIT(0u);
};

struct ETX_ALIGNED AtmosphereSunPushConstants {
  uint32_t output_texture_index ETX_INIT(0u);
  uint32_t width ETX_INIT(0u);
  uint32_t height ETX_INIT(0u);
  uint32_t pad0 ETX_INIT(0u);
  float3 light_direction ETX_INIT({});
  float angular_size ETX_INIT(0.0f);
  float atmosphere_altitude ETX_INIT(0.0f);
  float atmosphere_rayleigh_scale ETX_INIT(0.0f);
  float atmosphere_mie_scale ETX_INIT(0.0f);
  float atmosphere_ozone_scale ETX_INIT(0.0f);
};

ETX_SHARED_INLINE float scattering_distance_to_sphere(ETX_IN(float3, ray_origin), ETX_IN(float3, ray_direction), ETX_IN(float3, center), float radius) {
  float3 e = ray_origin - center;
  float b = dot(ray_direction, e);
  float d = (b * b) - dot(e, e) + (radius * radius);
  if (d < 0.0f) {
    return 0.0f;
  }

  d = ETX_STD sqrt(d);
  float a0 = -b - d;
  float a1 = -b + d;
  return (a0 < 0.0f) ? (((a1 < 0.0f) ? 0.0f : a1)) : a0;
}

ETX_SHARED_INLINE float2 scattering_precomputed_params_to_uv(ETX_IN(float2, params)) {
  float x = params.x * 0.5f + 0.5f;
  float u = x * x;
  float h = params.y / kScatteringAtmosphereRadius;
  float v = ETX_STD sqrt(saturate(h));
  return float2(u, v);
}

ETX_SHARED_INLINE float2 scattering_uv_to_precomputed_params(ETX_IN(float2, uv)) {
  float h = (uv.y * uv.y) * kScatteringAtmosphereRadius;
  float n_dot_l = ETX_STD sqrt(uv.x) * 2.0f - 1.0f;
  return float2(n_dot_l, h);
}

ETX_SHARED_INLINE float scattering_rayleigh(float wavelength_nm) {
  float l = wavelength_nm / 100.0f;
  float l2 = l * l;
  float l4 = l2 * l2;
  return 1.169939f / (l4 * 100.0f);
}

ETX_SHARED_INLINE float scattering_mie(float wavelength_nm) {
  float scale = 0.3954608f * (kPi * kPi * kPi);
  return scale / (wavelength_nm * wavelength_nm);
}

ETX_SHARED_INLINE float scattering_ozone_absorption(float wavelength_nm) {
  float x = wavelength_nm;
  float x2 = x * x;
  float x3 = x2 * x;
  float x4 = x2 * x2;
  float x5 = x4 * x;
  float x6 = x4 * x2;
  float base = -1.109902e-15f * x6 + 3.950001e-12f * x5 - 5.784719e-09f * x4 + 4.460262e-06f * x3 - 1.909367e-03f * x2 + 4.303677e-01f * x - 3.992226e+01f;
  float na = 6.022140857f;
  float concentration = 41.58e-6f;
  return (base > 0.0f) ? (base * na * concentration) : 0.0f;
}

ETX_SHARED_INLINE float scattering_phase_rayleigh(float l_dot_v) {
  return (3.0f / 4.0f) * (1.0f + l_dot_v * l_dot_v) * (1.0f / kDoublePi);
}

ETX_SHARED_INLINE float scattering_phase_mie(float l_dot_v, float anisotropy) {
  float g = anisotropy;
  float temp = 1.0f + g * g - 2.0f * g * l_dot_v;
  return (3.0f / 2.0f) * ((1.0f - g * g) * (1.0f + l_dot_v * l_dot_v)) / ((2.0f + g * g) * temp * ETX_STD sqrt(temp)) * (1.0f / kDoublePi);
}

ETX_SHARED_INLINE float2 scattering_optical_depth_precomputed_uv_from_position(ETX_IN(float3, position), ETX_IN(float3, light_direction)) {
  float position_len = length(position);
  float n_dot_l = 0.0f;
  if ((position_len > kRayEpsilon)) {
    n_dot_l = dot(position / position_len, light_direction);
  }
  return scattering_precomputed_params_to_uv(float2(n_dot_l, position_len - kScatteringPlanetRadius));
}

ETX_SHARED_INLINE float scattering_distance_to_atmosphere_or_planet(ETX_IN(float3, origin), ETX_IN(float3, direction)) {
  float to_space = scattering_distance_to_sphere(origin, direction, float3(0.0f, 0.0f, 0.0f), kScatteringOuterSphereRadius);
  float to_planet = scattering_distance_to_sphere(origin, direction, float3(0.0f, 0.0f, 0.0f), kScatteringPlanetRadius);
  return (to_planet > 0.0f) ? to_planet : to_space;
}

ETX_SHARED_INLINE float3 scattering_sky_direction_equal_area(ETX_IN(float2, uv)) {
  float phi = uv.x * kDoublePi - kPi;
  float v_mapped = uv.y * 2.0f - 1.0f;
  float theta = ETX_STD asin(max(-1.0f, min(1.0f, -v_mapped)));
  float cos_phi = ETX_STD cos(phi);
  float sin_phi = ETX_STD sin(phi);
  float cos_theta = ETX_STD cos(theta);
  float sin_theta = ETX_STD sin(theta);
  return float3(cos_phi * cos_theta, sin_theta, sin_phi * cos_theta);
}

ETX_SHARED_INLINE float scattering_sky_average_weight(ETX_IN(float3, direction)) {
  return max(0.0f, direction.y);
}

ETX_SHARED_INLINE float3 scattering_sky_apply_approx_multiple_scattering(ETX_IN(float3, rgb), ETX_IN(float3, average_color)) {
  return rgb + kDoublePi * average_color * rgb + average_color;
}

ETX_SHARED_INLINE float3 scattering_sky_approx_multiple_scattering_only(ETX_IN(float3, rgb), ETX_IN(float3, average_color)) {
  return kDoublePi * average_color * rgb + average_color;
}

ETX_SHARED_INLINE float scattering_sun_disk_darkening(float u, float v) {
  float radial = max(0.0f, 1.0f - (u * u + v * v));
  return 1.0f - 0.6f * (1.0f - radial);
}

ETX_SHARED_INLINE AtmosphereSkyGpuParameters scattering_make_gpu_parameters(float altitude, float anisotropy, float rayleigh_scale, float mie_scale, float ozone_scale) {
  ETX_ZERO_INIT(AtmosphereSkyGpuParameters, result);
  result.altitude = altitude;
  result.anisotropy = anisotropy;
  result.rayleigh_scale = rayleigh_scale;
  result.mie_scale = mie_scale;
  result.ozone_scale = ozone_scale;
  return result;
}

ETX_SHARED_INLINE ScatteringDensityAndDerivative scattering_density_and_derivative_direct(float height_above_surface) {
  float h = max(0.0f, height_above_surface);
  float x = h / 1000.0f;
  float x2 = x * x;
  float x3 = x2 * x;
  float x4 = x2 * x2;
  float x5 = x4 * x;
  float x6 = x3 * x3;

  float rayleigh_density = ETX_STD exp(-h / kScatteringRayleighDensityScale);
  float mie_density = ETX_STD exp(-h / kScatteringMieDensityScale);
  float rayleigh_derivative = -rayleigh_density / kScatteringRayleighDensityScale;
  float mie_derivative = -mie_density / kScatteringMieDensityScale;

  float f = 3.759384e-08f * x6 - 1.067250e-05f * x5 + 1.080311e-03f * x4 - 4.851181e-02f * x3 + 9.185432e-01f * x2 - 4.886021e+00f * x + 7.900478e+00f;
  float df = 6.0f * 3.759384e-08f * x5 - 5.0f * 1.067250e-05f * x4 + 4.0f * 1.080311e-03f * x3 - 3.0f * 4.851181e-02f * x2 + 2.0f * 9.185432e-01f * x - 4.886021e+00f;

  ETX_STATIC_CONST float kOzoneScale = 1.0f / 30.8491249f;
  ETX_STATIC_CONST float kDxDh = 1.0f / 1000.0f;
  float ozone_density = max(0.0f, f * kOzoneScale);
  float ozone_derivative = max(0.0f, df) * kDxDh * kOzoneScale;

  ScatteringDensityAndDerivative result;
  result.density = float3(rayleigh_density, mie_density, ozone_density);
  result.derivative = float3(rayleigh_derivative, mie_derivative, ozone_derivative);
  return result;
}

ETX_SHARED_INLINE float3 scattering_density_direct(float height_above_surface) {
  return scattering_density_and_derivative_direct(height_above_surface).density;
}

ETX_SHARED_INLINE float3 scattering_density_derivative_direct(float height_above_surface) {
  return scattering_density_and_derivative_direct(height_above_surface).derivative;
}

ETX_SHARED_INLINE float scattering_calculate_step_size_direct(float current_distance, float total_distance, ETX_IN(float3, origin), ETX_IN(float3, direction)) {
  float3 position = origin + direction * current_distance;
  float position_len = length(position);
  float height = position_len - kScatteringPlanetRadius;
  float3 directional_derivative = scattering_density_derivative_direct(height);

  float directional_factor = 0.0f;
  if ((position_len > kRayEpsilon)) {
    float3 normalized_position = position / position_len;
    directional_factor = dot(normalized_position, direction);
  }
  directional_derivative = directional_derivative * directional_factor;

  float l0 = ETX_STD log(max(kRayEpsilon, (1.0f + directional_derivative.x) / kScatteringDeltaDensity)) * kScatteringRayleighDensityScale;
  float l1 = ETX_STD log(max(kRayEpsilon, (1.0f + directional_derivative.y) / kScatteringDeltaDensity)) * kScatteringMieDensityScale;
  float calculated = ETX_STD sqrt(kScatteringDeltaDensity * ((l0 * l0) + (l1 * l1)));

  return min(total_distance - current_distance, max(kRayEpsilon, calculated));
}

ETX_SHARED_INLINE float3 scattering_optical_length_direct(ETX_IN(float3, origin), ETX_IN(float3, direction), float total_distance) {
  float3 result = float3(0.0f, 0.0f, 0.0f);
  float t = 0.0f;

  uint32_t steps = 0u;
  while ((t < total_distance) && (steps < kScatteringOpticalDepthMaxSteps)) {
    float dt = scattering_calculate_step_size_direct(t, total_distance, origin, direction);
    float3 p = origin + direction * (t + 0.5f * dt);
    t += dt;
    float height_above_surface = length(p) - kScatteringPlanetRadius;
    result += dt * scattering_density_direct(height_above_surface);
    steps += 1u;
  }

  return result;
}
