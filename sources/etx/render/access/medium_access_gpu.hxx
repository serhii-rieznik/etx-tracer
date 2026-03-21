#pragma once

#include <interop/medium_density_shared.hxx>
#include <access/medium_access_shared.hxx>
#include <access/spectrum_access_gpu.hxx>
#include <interop/scene_gpu_access_shared.hxx>

struct MediumAccessGPUContext {
  uint mediums_descriptor_index;
  uint spectrums_descriptor_index;
};

MediumAccessGPUContext make_medium_access_gpu_context(uint mediums_descriptor_index, uint spectrums_descriptor_index) {
  MediumAccessGPUContext result;
  result.mediums_descriptor_index = mediums_descriptor_index;
  result.spectrums_descriptor_index = spectrums_descriptor_index;
  return result;
}

bool medium_access_try_get_desc_offset(MediumAccessGPUContext context, uint medium_index, out uint medium_desc_offset) {
  medium_desc_offset = kInvalidIndex;
  if ((medium_index == kInvalidIndex) || (context.mediums_descriptor_index == kInvalidIndex)) {
    return false;
  }

  ByteAddressBuffer medium_blob = bindless_buffers[NonUniformResourceIndex(context.mediums_descriptor_index)];
  uint medium_count = medium_blob.Load(kMediumBlobHeaderMediumCountOffset);
  if (medium_index >= medium_count) {
    return false;
  }

  uint mediums_offset = medium_blob.Load(kMediumBlobHeaderMediumsOffset);
  if (mediums_offset == kInvalidIndex) {
    return false;
  }

  medium_desc_offset = mediums_offset + medium_index * kMediumStride;
  return true;
}

void medium_access_gpu_load(MediumAccessGPUContext context, ByteAddressBuffer medium_blob, uint medium_desc_offset, uint medium_index, out MediumAccess access) {
  access = ETX_ZERO(MediumAccess);
  access.grid.dimensions = medium_blob.Load3(medium_desc_offset + kMediumGridDimensionsOffset);
  access.grid.type = medium_blob.Load(medium_desc_offset + kMediumGridTypeOffset);
  access.grid.noise_type = medium_blob.Load(medium_desc_offset + kMediumGridNoiseTypeOffset);
  access.grid.density_data_offset = medium_blob.Load(medium_desc_offset + kMediumGridDensityDataOffsetOffset);
  access.grid.density_count = medium_blob.Load(medium_desc_offset + kMediumGridDensityCountOffset);
  access.grid.noise_seed = medium_blob.Load(medium_desc_offset + kMediumGridNoiseSeedOffset);
  access.grid.noise_offset = asfloat(medium_blob.Load3(medium_desc_offset + kMediumGridNoiseOffsetOffset));
  access.grid.noise_enable_border_fade = medium_blob.Load(medium_desc_offset + kMediumGridNoiseEnableBorderFadeOffset);
  access.grid.noise_octaves = medium_blob.Load(medium_desc_offset + kMediumGridNoiseOctavesOffset);
  access.grid.noise_scale = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoiseScaleOffset));
  access.grid.noise_lacunarity = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoiseLacunarityOffset));
  access.grid.noise_persistence = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoisePersistenceOffset));
  access.grid.noise_power = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoisePowerOffset));
  access.grid.noise_sharpness = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoiseSharpnessOffset));
  access.grid.noise_border_fade_distance = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoiseBorderFadeDistanceOffset));
  access.grid.density_data_chunk_index = medium_blob.Load(medium_desc_offset + kMediumGridDensityDataChunkIndexOffset);
  access.bounds_min = asfloat(medium_blob.Load3(medium_desc_offset + kMediumBoundsMinOffset));
  access.bounds_max = asfloat(medium_blob.Load3(medium_desc_offset + kMediumBoundsMaxOffset));
  access.medium_index = medium_index;
  access.density_payload_descriptor_index = kInvalidIndex;
  uint data_chunk_count = medium_blob.Load(kMediumBlobHeaderDataChunkCountOffset);
  uint data_chunk_indices_offset = medium_blob.Load(kMediumBlobHeaderDataChunkIndicesOffset);
  if ((access.grid.density_data_chunk_index != kInvalidIndex) && (access.grid.density_data_chunk_index < data_chunk_count) && (data_chunk_indices_offset != kInvalidIndex)) {
    access.density_payload_descriptor_index = medium_blob.Load(data_chunk_indices_offset + access.grid.density_data_chunk_index * 4u);
  }
  access.medium_class = load_u16(medium_blob, medium_desc_offset + kMediumClassOffset);
  access.absorption_spectrum_index = medium_blob.Load(medium_desc_offset + kMediumAbsorptionIndexOffset);
  access.scattering_spectrum_index = medium_blob.Load(medium_desc_offset + kMediumScatteringIndexOffset);
  access.phase_function_g = asfloat(medium_blob.Load(medium_desc_offset + kMediumPhaseFunctionGOffset));
  access.enable_explicit_connections = medium_blob.Load(medium_desc_offset + kMediumEnableExplicitConnectionsOffset);
}

bool medium_access_try_load(MediumAccessGPUContext context, uint medium_index, out MediumAccess access) {
  access = ETX_ZERO(MediumAccess);
  uint medium_desc_offset = kInvalidIndex;
  if (medium_access_try_get_desc_offset(context, medium_index, medium_desc_offset) == false) {
    return false;
  }

  ByteAddressBuffer medium_blob = bindless_buffers[NonUniformResourceIndex(context.mediums_descriptor_index)];
  medium_access_gpu_load(context, medium_blob, medium_desc_offset, medium_index, access);
  return true;
}

bool medium_access_has_grid_data(MediumAccessGPUContext context, MediumAccess access) {
  (void)context;
  if (medium_access_has_grid_data(access) == false) {
    return false;
  }

  if (access.grid.type == MediumGridType::NoiseFunction) {
    return true;
  }

  return (access.grid.density_data_offset != kInvalidIndex) && (access.density_payload_descriptor_index != kInvalidIndex);
}

struct MediumTextureSampleContext {
  ByteAddressBuffer payload_buffer;
  uint density_offset;
  uint density_count;
};

float medium_texture_sample_density(MediumTextureSampleContext context, uint3 dimensions, uint x, uint y, uint z) {
  uint index = x + y * dimensions.x + z * dimensions.x * dimensions.y;
  if (index >= context.density_count) {
    return 0.0f;
  }

  return asfloat(context.payload_buffer.Load(context.density_offset + index * 4u));
}

#include <interop/medium_texture_sample_shared.hxx>

float medium_access_sample_texture_3d(MediumAccessGPUContext context, MediumAccess access, float3 local_coord) {
  (void)context;
  if ((access.grid.density_count == 0u) || (access.grid.density_data_offset == kInvalidIndex) || (access.density_payload_descriptor_index == kInvalidIndex)) {
    return 0.0f;
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(access.density_payload_descriptor_index)];
  MediumTextureSampleContext sample_context = {payload_buffer, access.grid.density_data_offset, access.grid.density_count};
  return medium_texture_sample_shared_3d(sample_context, local_coord, access.grid.dimensions);
}

float medium_access_sample_noise(MediumAccessGPUContext context, MediumAccess access, float3 local_coord) {
  (void)context;
  return medium_density_shared_sample_noise(local_coord, access.bounds_min, access.bounds_max, access.grid.noise_type, access.grid.noise_scale, access.grid.noise_octaves,
    access.grid.noise_lacunarity, access.grid.noise_persistence, access.grid.noise_seed, access.grid.noise_offset, access.grid.noise_enable_border_fade,
    access.grid.noise_border_fade_distance);
}

float medium_access_sample_density(MediumAccessGPUContext context, MediumAccess access, float3 local_coord) {
  float value = 0.0f;
  if (access.grid.type == MediumGridType::NoiseFunction) {
    value = medium_access_sample_noise(context, access, local_coord);
  } else if (access.grid.type == MediumGridType::Texture3D) {
    value = medium_access_sample_texture_3d(context, access, local_coord);
  }

  return medium_density_shared_apply_shape(value, access.grid.noise_power, access.grid.noise_sharpness);
}

bool medium_access_can_sample_spectrum(MediumAccessGPUContext context, uint spectrum_index) {
  return scene_gpu_can_sample_spectrum(context.spectrums_descriptor_index, spectrum_index);
}

float3 medium_access_load_absorption_integrated(MediumAccessGPUContext context, MediumAccess access) {
  if (medium_access_can_sample_spectrum(context, access.absorption_spectrum_index) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(context.spectrums_descriptor_index)];
  SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, context.spectrums_descriptor_index);
  return spectrum_access_load_integrated(spectrum_context, access.absorption_spectrum_index);
}

float3 medium_access_load_scattering_integrated(MediumAccessGPUContext context, MediumAccess access) {
  if (medium_access_can_sample_spectrum(context, access.scattering_spectrum_index) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(context.spectrums_descriptor_index)];
  SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, context.spectrums_descriptor_index);
  return spectrum_access_load_integrated(spectrum_context, access.scattering_spectrum_index);
}

float3 medium_access_load_extinction_integrated(MediumAccessGPUContext context, MediumAccess access) {
  return medium_access_load_absorption_integrated(context, access) + medium_access_load_scattering_integrated(context, access);
}

SpectralResponse medium_access_load_absorption_spectral(MediumAccessGPUContext context, MediumAccess access, SpectralQuery spect) {
  if (medium_access_can_sample_spectrum(context, access.absorption_spectrum_index) == false) {
    return spectral_response_zero(spect);
  }

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(context.spectrums_descriptor_index)];
  SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, context.spectrums_descriptor_index);
  return spectrum_access_evaluate(spectrum_context, access.absorption_spectrum_index, spect);
}

SpectralResponse medium_access_load_scattering_spectral(MediumAccessGPUContext context, MediumAccess access, SpectralQuery spect) {
  if (medium_access_can_sample_spectrum(context, access.scattering_spectrum_index) == false) {
    return spectral_response_zero(spect);
  }

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(context.spectrums_descriptor_index)];
  SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, context.spectrums_descriptor_index);
  return spectrum_access_evaluate(spectrum_context, access.scattering_spectrum_index, spect);
}

SpectralResponse medium_access_load_extinction_spectral(MediumAccessGPUContext context, MediumAccess access, SpectralQuery spect) {
  return spectral_response_add(medium_access_load_absorption_spectral(context, access, spect), medium_access_load_scattering_spectral(context, access, spect));
}
