#pragma once

#include <etx/render/access/spectrum_access_cpu.hxx>
#include <etx/render/shared/density_grid.hxx>
#include <etx/render/access/medium_access_shared.hxx>

namespace etx {

struct Scene;

struct MediumAccessCPUContext {
  const Scene* scene = nullptr;
  const Medium* direct_medium = nullptr;
};

ETX_SHARED_INLINE MediumAccessCPUContext make_medium_access_cpu_context(const Scene& scene, const Medium* direct_medium = nullptr) {
  MediumAccessCPUContext result = {};
  result.scene = &scene;
  result.direct_medium = direct_medium;
  return result;
}

ETX_SHARED_INLINE MediumAccess medium_access_cpu_make(ETX_IN(Medium, medium), uint32_t medium_index = kInvalidIndex) {
  MediumAccess result = {};
  DensityGrid density_grid = {};
  density_grid.density = medium.density_view;
  result.grid = density_grid.to_shared_grid(medium.grid, static_cast<uint32_t>(medium.density_view.count));
  result.bounds_min = medium.bounds.p_min;
  result.bounds_max = medium.bounds.p_max;
  result.medium_index = medium_index;
  result.medium_class = static_cast<uint32_t>(medium.cls);
  result.absorption_spectrum_index = medium.absorption_index;
  result.scattering_spectrum_index = medium.scattering_index;
  return result;
}

ETX_SHARED_INLINE bool medium_access_try_load(ETX_IN(MediumAccessCPUContext, context), uint32_t medium_index, ETX_OUT(MediumAccess, access)) {
  access = {};
  if ((context.scene == nullptr) || (medium_index >= context.scene->mediums.count)) {
    return false;
  }

  access = medium_access_cpu_make(context.scene->mediums[medium_index], medium_index);
  return true;
}

ETX_SHARED_INLINE const Medium* medium_access_cpu_resolve_medium(ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access)) {
  if ((context.scene != nullptr) && (access.medium_index < context.scene->mediums.count)) {
    return &context.scene->mediums[access.medium_index];
  }

  return context.direct_medium;
}

ETX_SHARED_INLINE bool medium_access_has_grid_data(ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access)) {
  const Medium* medium = medium_access_cpu_resolve_medium(context, access);
  if (medium != nullptr) {
    return medium->has_grid_data();
  }

  return medium_density_shared_has_grid_data(access.grid.type, access.grid.dimensions, access.grid.density_count);
}

ETX_SHARED_INLINE float medium_access_sample_density(ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access), ETX_IN(float3, local_coord)) {
  const Medium* medium = medium_access_cpu_resolve_medium(context, access);
  if (medium != nullptr) {
    BoundingBox bounds = medium_access_bounds(access);
    return medium->sample_density(local_coord, bounds);
  }

  if (access.grid.type == MediumGridType::NoiseFunction) {
    return medium_density_shared_sample_noise(local_coord, access.bounds_min, access.bounds_max, access.grid.noise_type, access.grid.noise_scale,
      access.grid.noise_octaves, access.grid.noise_lacunarity, access.grid.noise_persistence, access.grid.noise_seed, access.grid.noise_offset,
      access.grid.noise_enable_border_fade, access.grid.noise_border_fade_distance);
  }

  return 0.0f;
}

ETX_SHARED_INLINE bool medium_access_can_sample_spectrum(ETX_IN(MediumAccessCPUContext, context), uint32_t spectrum_index) {
  if (context.scene == nullptr) {
    return false;
  }

  SpectrumAccessCPUContext spectrum_context = make_spectrum_access_cpu_context(context.scene->spectrums.a, static_cast<uint32_t>(context.scene->spectrums.count));
  return spectrum_access_can_evaluate(spectrum_context, spectrum_index);
}

ETX_SHARED_INLINE float3 medium_access_load_absorption_integrated(ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access)) {
  if (medium_access_can_sample_spectrum(context, access.absorption_spectrum_index) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  SpectrumAccessCPUContext spectrum_context = make_spectrum_access_cpu_context(context.scene->spectrums.a, static_cast<uint32_t>(context.scene->spectrums.count));
  return spectrum_access_load_integrated(spectrum_context, access.absorption_spectrum_index);
}

ETX_SHARED_INLINE float3 medium_access_load_scattering_integrated(ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access)) {
  if (medium_access_can_sample_spectrum(context, access.scattering_spectrum_index) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  SpectrumAccessCPUContext spectrum_context = make_spectrum_access_cpu_context(context.scene->spectrums.a, static_cast<uint32_t>(context.scene->spectrums.count));
  return spectrum_access_load_integrated(spectrum_context, access.scattering_spectrum_index);
}

ETX_SHARED_INLINE float3 medium_access_load_extinction_integrated(ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access)) {
  return medium_access_load_absorption_integrated(context, access) + medium_access_load_scattering_integrated(context, access);
}

ETX_SHARED_INLINE SpectralResponse medium_access_load_absorption_spectral(
  ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access), ETX_IN(SpectralQuery, spect)) {
  if (medium_access_can_sample_spectrum(context, access.absorption_spectrum_index) == false) {
    return SpectralResponse{spect, 0.0f};
  }

  SpectrumAccessCPUContext spectrum_context = make_spectrum_access_cpu_context(context.scene->spectrums.a, static_cast<uint32_t>(context.scene->spectrums.count));
  const ::SpectralResponse response = spectrum_access_evaluate(spectrum_context, access.absorption_spectrum_index, static_cast<const ::SpectralQuery&>(spect));
  SpectralQuery response_query = {response.wavelength, response.flags};
  return ::spectral_response_is_spectral(response) ? SpectralResponse{response_query, response.value} : SpectralResponse{response_query, response.integrated};
}

ETX_SHARED_INLINE SpectralResponse medium_access_load_scattering_spectral(
  ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access), ETX_IN(SpectralQuery, spect)) {
  if (medium_access_can_sample_spectrum(context, access.scattering_spectrum_index) == false) {
    return SpectralResponse{spect, 0.0f};
  }

  SpectrumAccessCPUContext spectrum_context = make_spectrum_access_cpu_context(context.scene->spectrums.a, static_cast<uint32_t>(context.scene->spectrums.count));
  const ::SpectralResponse response = spectrum_access_evaluate(spectrum_context, access.scattering_spectrum_index, static_cast<const ::SpectralQuery&>(spect));
  SpectralQuery response_query = {response.wavelength, response.flags};
  return ::spectral_response_is_spectral(response) ? SpectralResponse{response_query, response.value} : SpectralResponse{response_query, response.integrated};
}

ETX_SHARED_INLINE SpectralResponse medium_access_load_extinction_spectral(
  ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access), ETX_IN(SpectralQuery, spect)) {
  return medium_access_load_absorption_spectral(context, access, spect) + medium_access_load_scattering_spectral(context, access, spect);
}

}  // namespace etx
