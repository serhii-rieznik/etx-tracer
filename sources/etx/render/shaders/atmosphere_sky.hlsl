#include "bindless.hlsl"

#include <interop/gpu_abi_constants.hxx>
#include <interop/spectrum.hxx>
#include <interop/atmosphere_scattering_shared.hxx>

struct SpectrumAccessGPUSharedContext {
  ByteAddressBuffer buffer;
};

float3 spectrum_access_shared_gpu_integrated(SpectrumAccessGPUSharedContext context, uint spectrum_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride;
  return asfloat(context.buffer.Load3(base_offset + kSpectralDistributionIntegratedOffset));
}

uint spectrum_access_shared_gpu_entry_count(SpectrumAccessGPUSharedContext context, uint spectrum_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride;
  return context.buffer.Load(base_offset + kSpectralDistributionEntryCountOffset);
}

float spectrum_access_shared_gpu_entry_wavelength(SpectrumAccessGPUSharedContext context, uint spectrum_index, uint entry_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride + kSpectralDistributionEntriesOffset + entry_index * kSpectralDistributionEntryStride;
  return asfloat(context.buffer.Load(base_offset + 0u));
}

float spectrum_access_shared_gpu_entry_power(SpectrumAccessGPUSharedContext context, uint spectrum_index, uint entry_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride + kSpectralDistributionEntriesOffset + entry_index * kSpectralDistributionEntryStride;
  return asfloat(context.buffer.Load(base_offset + 4u));
}

#define ETX_SPECTRUM_ACCESS_SHARED_CONTEXT_TYPE SpectrumAccessGPUSharedContext
#define ETX_SPECTRUM_ACCESS_SHARED_INTEGRATED(context, spectrum_index) spectrum_access_shared_gpu_integrated(context, spectrum_index)
#define ETX_SPECTRUM_ACCESS_SHARED_ENTRY_COUNT(context, spectrum_index) spectrum_access_shared_gpu_entry_count(context, spectrum_index)
#define ETX_SPECTRUM_ACCESS_SHARED_ENTRY_WAVELENGTH(context, spectrum_index, entry_index) spectrum_access_shared_gpu_entry_wavelength(context, spectrum_index, entry_index)
#define ETX_SPECTRUM_ACCESS_SHARED_ENTRY_POWER(context, spectrum_index, entry_index) spectrum_access_shared_gpu_entry_power(context, spectrum_index, entry_index)
#include <interop/spectrum_access_shared.hxx>
#undef ETX_SPECTRUM_ACCESS_SHARED_ENTRY_POWER
#undef ETX_SPECTRUM_ACCESS_SHARED_ENTRY_WAVELENGTH
#undef ETX_SPECTRUM_ACCESS_SHARED_ENTRY_COUNT
#undef ETX_SPECTRUM_ACCESS_SHARED_INTEGRATED
#undef ETX_SPECTRUM_ACCESS_SHARED_CONTEXT_TYPE

[[vk::push_constant]] AtmosphereSkyPushConstants constants;

groupshared float4 g_group_sum[64];

AtmosphereSkyGpuLight load_atmosphere_sky_light(ByteAddressBuffer buffer, uint light_index) {
  uint base_offset = light_index * kAtmosphereSkyGpuLightStride;
  AtmosphereSkyGpuLight result = (AtmosphereSkyGpuLight)0;
  result.direction = asfloat(buffer.Load3(base_offset + kAtmosphereSkyGpuLightDirectionOffset));
  result.angular_size = asfloat(buffer.Load(base_offset + kAtmosphereSkyGpuLightAngularSizeOffset));
  result.intensity_scale = asfloat(buffer.Load(base_offset + kAtmosphereSkyGpuLightIntensityScaleOffset));
  result.emission_spectrum_index = buffer.Load(base_offset + kAtmosphereSkyGpuLightEmissionSpectrumIndexOffset);
  return result;
}

float3 atmosphere_sky_sample_optical_depth_lut(uint texture_index, uint sampler_index, float3 position, float3 light_direction) {
  float2 uv = scattering_optical_depth_precomputed_uv_from_position(position, light_direction);
  float4 value = bindless_textures[NonUniformResourceIndex(texture_index)].SampleLevel(bindless_samplers[NonUniformResourceIndex(sampler_index)], uv, 0.0f);
  return value.xyz;
}

float atmosphere_sky_transmittance(float optical_depth) {
  return exp(-optical_depth);
}

float atmosphere_sky_spectral_step_value(float wavelength, float emission_power, float3 total_optical_path, float3 density_scale, float3 current_density, float phase_r, float phase_m,
  float dt) {
  float r = scattering_rayleigh(wavelength);
  float m = scattering_mie(wavelength);
  float o = scattering_ozone_absorption(wavelength);

  float transmittance_r = atmosphere_sky_transmittance(r * total_optical_path.x);
  float transmittance_m = atmosphere_sky_transmittance(m * total_optical_path.y);
  float transmittance_o = atmosphere_sky_transmittance(o * total_optical_path.z);
  float total_transmittance = transmittance_r * transmittance_m * transmittance_o;

  float scattering_coeff = phase_r * r * density_scale.x * current_density.x + phase_m * m * density_scale.y * current_density.y;
  return total_transmittance * dt * scattering_coeff * emission_power;
}

float3 atmosphere_sky_integrate_light_step_xyz(SpectrumAccessGPUSharedContext spectra_context, AtmosphereSkyGpuLight light, float3 total_optical_path, float3 density_scale,
  float3 current_density, float phase_r, float phase_m, float dt) {
  uint entry_count = spectrum_access_shared_gpu_entry_count(spectra_context, light.emission_spectrum_index);
  if (entry_count < 2u) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  float3 xyz_result = float3(0.0f, 0.0f, 0.0f);

  float prev_wavelength = spectrum_access_shared_gpu_entry_wavelength(spectra_context, light.emission_spectrum_index, 0u);
  float prev_emission = spectrum_access_shared_gpu_entry_power(spectra_context, light.emission_spectrum_index, 0u) * light.intensity_scale;
  float prev_value =
    atmosphere_sky_spectral_step_value(prev_wavelength, prev_emission, total_optical_path, density_scale, current_density, phase_r, phase_m, dt);
  float3 prev_xyz = spectral_response_to_xyz(spectral_response_make(prev_wavelength, prev_value));

  for (uint entry_index = 1u; entry_index < entry_count; ++entry_index) {
    float wavelength = spectrum_access_shared_gpu_entry_wavelength(spectra_context, light.emission_spectrum_index, entry_index);
    float emission = spectrum_access_shared_gpu_entry_power(spectra_context, light.emission_spectrum_index, entry_index) * light.intensity_scale;
    float value = atmosphere_sky_spectral_step_value(wavelength, emission, total_optical_path, density_scale, current_density, phase_r, phase_m, dt);
    float3 xyz = spectral_response_to_xyz(spectral_response_make(wavelength, value));

    xyz_result += (wavelength - prev_wavelength) * (prev_xyz + 0.5f * (xyz - prev_xyz));
    prev_wavelength = wavelength;
    prev_xyz = xyz;
  }

  return xyz_result;
}

float3 atmosphere_sky_radiance_xyz(AtmosphereSkyPushConstants constants, float3 view_direction) {
  if (constants.light_count == 0u) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  float3 origin = float3(0.0f, kScatteringPlanetRadius + constants.atmosphere_altitude, 0.0f);
  float3 density_scale = float3(constants.atmosphere_rayleigh_scale, constants.atmosphere_mie_scale, constants.atmosphere_ozone_scale);

  float3 view_optical_path = float3(0.0f, 0.0f, 0.0f);
  float3 xyz_result = float3(0.0f, 0.0f, 0.0f);
  float to_space = scattering_distance_to_atmosphere_or_planet(origin, view_direction);
  float t = 0.0f;

  ByteAddressBuffer lights_buffer = bindless_buffers[NonUniformResourceIndex(constants.lights_buffer_index)];
  SpectrumAccessGPUSharedContext spectra_context;
  spectra_context.buffer = bindless_buffers[NonUniformResourceIndex(constants.spectra_buffer_index)];

  while (t < to_space) {
    float dt = scattering_calculate_step_size_direct(t, to_space, origin, view_direction);
    float3 p = origin + view_direction * (t + 0.5f * dt);
    t += dt;

    float height_above_surface = length(p) - kScatteringPlanetRadius;
    if (height_above_surface < -kScatteringRayleighDensityScale) {
      break;
    }

    float3 current_density = scattering_density_direct(height_above_surface);
    view_optical_path += dt * density_scale * current_density;

    for (uint light_index = 0u; light_index < constants.light_count; ++light_index) {
      AtmosphereSkyGpuLight light = load_atmosphere_sky_light(lights_buffer, light_index);
      float3 light_optical_path = density_scale * atmosphere_sky_sample_optical_depth_lut(constants.optical_depth_texture_index, constants.optical_depth_sampler_index, p, light.direction);
      float3 total_optical_path = view_optical_path + light_optical_path;

      float l_dot_v = dot(light.direction, view_direction);
      float phase_r = scattering_phase_rayleigh(l_dot_v);
      float phase_m = scattering_phase_mie(l_dot_v, constants.atmosphere_anisotropy);

      xyz_result += atmosphere_sky_integrate_light_step_xyz(spectra_context, light, total_optical_path, density_scale, current_density, phase_r, phase_m, dt);
    }
  }

  return xyz_result;
}

[numthreads(8, 8, 1)]
void sky_raw_main(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
  float3 rgb = float3(0.0f, 0.0f, 0.0f);
  float weight = 0.0f;

  if ((dtid.x < constants.width) && (dtid.y < constants.height)) {
    float2 uv = float2((float(dtid.x) + 0.5f) / float(constants.width), (float(dtid.y) + 0.5f) / float(constants.height));
    float3 direction = scattering_sky_direction_equal_area(uv);
    float3 xyz = atmosphere_sky_radiance_xyz(constants, direction);
    rgb = max(float3(0.0f, 0.0f, 0.0f), spectral_xyz_to_rgb(xyz));
    weight = scattering_sky_average_weight(direction);

    bindless_storage_textures[NonUniformResourceIndex(constants.output_texture_index)][int2(dtid.xy)] = float4(rgb, 1.0f);
  }

  float3 weighted_rgb = rgb * weight;
  g_group_sum[gi] = float4(weighted_rgb, weight);
  GroupMemoryBarrierWithGroupSync();

  for (uint stride = 32u; stride > 0u; stride >>= 1u) {
    if (gi < stride) {
      g_group_sum[gi] += g_group_sum[gi + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if ((gi == 0u) && (constants.aux_texture_index != kInvalidIndex)) {
    bindless_storage_textures[NonUniformResourceIndex(constants.aux_texture_index)][int2(gid.xy)] = g_group_sum[0];
  }
}

[numthreads(8, 8, 1)]
void sky_finalize_main(uint3 dtid : SV_DispatchThreadID) {
  if ((dtid.x >= constants.width) || (dtid.y >= constants.height)) {
    return;
  }

  float4 current = bindless_storage_textures[NonUniformResourceIndex(constants.output_texture_index)][int2(dtid.xy)];
  float3 average_color = float3(constants.average_color_x, constants.average_color_y, constants.average_color_z);
  float3 rgb = scattering_sky_apply_approx_multiple_scattering(current.xyz, average_color);
  bindless_storage_textures[NonUniformResourceIndex(constants.output_texture_index)][int2(dtid.xy)] = float4(rgb, current.w);
}
