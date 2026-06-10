#pragma once

#include <etx/render/access/medium_access_shared.hxx>
#include <etx/render/access/spectrum_access_cpu.hxx>

struct MediumAccessCPUContext {
  const Scene* scene = nullptr;
};

ETX_SHARED_INLINE MediumAccessCPUContext make_medium_access_cpu_context(const Scene& scene) {
  MediumAccessCPUContext result = {};
  result.scene = &scene;
  return result;
}

ETX_SHARED_INLINE bool medium_access_can_load(ETX_IN(MediumAccessCPUContext, context), uint32_t medium_index) {
  return medium_index < context.scene->mediums.count;
}

ETX_SHARED_INLINE const Medium& medium_access_cpu_medium(ETX_IN(MediumAccessCPUContext, context), uint32_t medium_index) {
  ETX_ASSERT(medium_index < static_cast<uint32_t>(context.scene->mediums.count));
  return context.scene->mediums[medium_index];
}

ETX_SHARED_INLINE MediumAccess medium_access_cpu_make(ETX_IN(Medium, medium), uint32_t medium_index) {
  MediumAccess result = {};
  DensityGrid density_grid = {};
  density_grid.density = medium.density_view;
  result.grid = density_grid.to_shared_grid(medium.grid, static_cast<uint32_t>(medium.density_view.count));
  result.bounds_min = medium.bounds.p_min;
  result.bounds_max = medium.bounds.p_max;
  result.medium_index = medium_index;
  result.density_payload_descriptor_index = kInvalidIndex;
  result.medium_class = static_cast<uint32_t>(medium.cls);
  result.absorption_spectrum_index = medium.absorption_index;
  result.scattering_spectrum_index = medium.scattering_index;
  return result;
}

ETX_SHARED_INLINE bool medium_access_try_load(ETX_IN(MediumAccessCPUContext, context), uint32_t medium_index, ETX_OUT(MediumAccess, access)) {
  access = {};
  if (medium_access_can_load(context, medium_index) == false) {
    return false;
  }

  access = medium_access_cpu_make(medium_access_cpu_medium(context, medium_index), medium_index);
  return true;
}

ETX_SHARED_INLINE bool medium_access_has_grid_data(ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access)) {
  (void)context;
  return medium_access_has_grid_data(access);
}

ETX_SHARED_INLINE float medium_access_sample_density(ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access), ETX_IN(float3, local_coord)) {
  if (medium_access_can_load(context, access.medium_index) == false) {
    return 0.0f;
  }

  const Medium& medium = medium_access_cpu_medium(context, access.medium_index);
  if ((medium.grid.type == MediumGridType::Texture3D) && (medium.grid.density_image_index != kInvalidIndex) && (medium.grid.density_image_index < context.scene->images.count)) {
    float3 uvw = {};
    if (medium_density_shared_texture_uvw(local_coord, medium.grid.dimensions, uvw) == false) {
      return 0.0f;
    }

    const Image& density_image = context.scene->images[medium.grid.density_image_index];
    if ((density_image.format == Image::Format::R32F) && (density_image.isize.x == medium.grid.dimensions.x) && (density_image.isize.y == medium.grid.dimensions.y) &&
        (density_image.isize.z == medium.grid.dimensions.z)) {
      const float value = density_image.evaluate_r32f_fast_3d(uvw);
      return medium_density_shared_apply_shape(value, medium.grid.noise_power, medium.grid.noise_sharpness);
    }
  }

  return medium.sample_density(local_coord, medium.bounds);
}

ETX_SHARED_INLINE bool medium_access_can_sample_spectrum(ETX_IN(MediumAccessCPUContext, context), uint32_t spectrum_index) {
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

ETX_SHARED_INLINE SpectralResponse medium_access_load_absorption_spectral(ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access), ETX_IN(SpectralQuery, spect)) {
  if (medium_access_can_sample_spectrum(context, access.absorption_spectrum_index) == false) {
    return SpectralResponse{spect, 0.0f};
  }

  SpectrumAccessCPUContext spectrum_context = make_spectrum_access_cpu_context(context.scene->spectrums.a, static_cast<uint32_t>(context.scene->spectrums.count));
  const ::SpectralResponse response = spectrum_access_evaluate(spectrum_context, access.absorption_spectrum_index, static_cast<const ::SpectralQuery&>(spect));
  SpectralQuery response_query = {response.wavelength, response.flags};
  return ::spectral_response_is_spectral(response) ? SpectralResponse{response_query, response.value} : SpectralResponse{response_query, response.integrated};
}

ETX_SHARED_INLINE SpectralResponse medium_access_load_scattering_spectral(ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access), ETX_IN(SpectralQuery, spect)) {
  if (medium_access_can_sample_spectrum(context, access.scattering_spectrum_index) == false) {
    return SpectralResponse{spect, 0.0f};
  }

  SpectrumAccessCPUContext spectrum_context = make_spectrum_access_cpu_context(context.scene->spectrums.a, static_cast<uint32_t>(context.scene->spectrums.count));
  const ::SpectralResponse response = spectrum_access_evaluate(spectrum_context, access.scattering_spectrum_index, static_cast<const ::SpectralQuery&>(spect));
  SpectralQuery response_query = {response.wavelength, response.flags};
  return ::spectral_response_is_spectral(response) ? SpectralResponse{response_query, response.value} : SpectralResponse{response_query, response.integrated};
}

ETX_SHARED_INLINE SpectralResponse medium_access_load_extinction_spectral(ETX_IN(MediumAccessCPUContext, context), ETX_IN(MediumAccess, access), ETX_IN(SpectralQuery, spect)) {
  return medium_access_load_absorption_spectral(context, access, spect) + medium_access_load_scattering_spectral(context, access, spect);
}
