#pragma once

#include <access/medium_access_shared.hxx>
#include <access/spectrum_access_gpu.hxx>

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
  return medium_density_shared_has_grid_data(access.grid.type, access.grid.dimensions, access.grid.density_count);
}

bool medium_access_can_sample_spectrum(MediumAccessGPUContext context, uint spectrum_index) {
  return scene_resource_shared_can_sample_spectrum(context.spectrums_descriptor_index, spectrum_index);
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
