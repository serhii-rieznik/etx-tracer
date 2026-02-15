#pragma once

#include "bindless.hlsl"

#include <interop/geometry.hxx>
#include <interop/gpu_abi_constants.hxx>
#include <interop/image.hxx>
#include <interop/material.hxx>
#include <interop/projection.hxx>
#include <interop/distribution.hxx>
#include <interop/sampler_policy.hxx>
#include <interop/sampler.hxx>
#include <interop/medium_density_shared.hxx>

float rnd01(inout uint state) {
  return sampler_next_random(state);
}

uint load_u8(ByteAddressBuffer buffer, uint byte_offset) {
  uint word = buffer.Load(byte_offset & ~3u);
  uint shift = (byte_offset & 3u) * 8u;
  return (word >> shift) & 0xffu;
}

uint load_u16(ByteAddressBuffer buffer, uint byte_offset) {
  uint word = buffer.Load(byte_offset & ~3u);
  uint shift = (byte_offset & 2u) * 8u;
  return (word >> shift) & 0xffffu;
}

uint blue_noise_table_index(uint2 pixel, uint sample_index, uint dimension) {
  uint x = pixel.x & (kSamplerBlueNoiseTileSize - 1u);
  uint y = pixel.y & (kSamplerBlueNoiseTileSize - 1u);
  uint wrapped_sample = sample_index & (kSamplerBlueNoiseSampleCount - 1u);
  uint wrapped_dimension = dimension & (kSamplerBlueNoiseDimensionCount - 1u);
  return (((wrapped_sample * kSamplerBlueNoiseDimensionCount) + wrapped_dimension) * kSamplerBlueNoiseTileSize + y) * kSamplerBlueNoiseTileSize + x;
}

float sample_blue_noise_value(uint2 pixel, uint sample_index, uint dimension) {
  if (constants.blue_noise_buffer_index == kInvalidIndex) {
    return 0.5f;
  }

  ByteAddressBuffer blue_noise_table = bindless_buffers[NonUniformResourceIndex(constants.blue_noise_buffer_index)];
  uint index = blue_noise_table_index(pixel, sample_index, dimension);
  uint value = load_u8(blue_noise_table, index);
  return (float(value) + 0.5f) * (1.0f / float(kSamplerBlueNoiseSampleCount));
}

bool sample_use_blue_noise_primary(uint current_sample, uint stream) {
  SamplerPolicy policy;
  policy.enable_blue_noise = (constants.blue_noise_buffer_index == kInvalidIndex) ? 0u : 1u;
  policy.blue_noise_sample_limit = kSamplerBlueNoiseSampleCount;
  policy.blue_noise_dimension_limit = kSamplerBlueNoiseDimensionCount;
  policy.direct_camera_only = 1u;
  return sampler_use_blue_noise(policy, kSamplerPathSourceCamera, 1u, current_sample, stream);
}

float2 sample_primary_hybrid_2d(uint2 pixel, uint current_sample, uint stream, inout uint seed) {
  if (sample_use_blue_noise_primary(current_sample, stream)) {
    uint dimension_base = sampler_stream_dimension_base(stream);
    return float2(sample_blue_noise_value(pixel, current_sample, dimension_base + 0u), sample_blue_noise_value(pixel, current_sample, dimension_base + 1u));
  }

  return float2(rnd01(seed), rnd01(seed));
}

float3 load_float3(ByteAddressBuffer buffer, uint index) {
  return asfloat(buffer.Load3(index * 12u));
}

float2 load_float2(ByteAddressBuffer buffer, uint index) {
  return asfloat(buffer.Load2(index * 8u));
}

float3 load_float3_at_offset(ByteAddressBuffer buffer, uint byte_offset) {
  return asfloat(buffer.Load3(byte_offset));
}

struct TriangleData {
  uint3 i;
  uint material_index;
  float3 geo_n;
  uint emitter_index;
};

TriangleData load_triangle(ByteAddressBuffer buffer, uint triangle_index) {
  const uint base_offset = triangle_index * kTriangleStride;
  uint4 a = buffer.Load4(base_offset + 0u);
  uint4 b = buffer.Load4(base_offset + 16u);

  TriangleData result;
  result.i = a.xyz;
  result.material_index = a.w;
  result.geo_n = asfloat(b.xyz);
  result.emitter_index = b.w;
  return result;
}

uint load_material_scattering_spectrum_index(ByteAddressBuffer buffer, uint material_index) {
  uint base_offset = material_index * kMaterialStride;
  return buffer.Load(base_offset + kMaterialScatteringSpectrumIndexOffset);
}

uint load_material_scattering_image_index(ByteAddressBuffer buffer, uint material_index) {
  uint base_offset = material_index * kMaterialStride;
  return buffer.Load(base_offset + kMaterialScatteringImageIndexOffset);
}

uint load_material_class(ByteAddressBuffer buffer, uint material_index) {
  uint base_offset = material_index * kMaterialStride;
  return buffer.Load(base_offset + kMaterialClassOffset);
}

uint load_material_int_medium(ByteAddressBuffer buffer, uint material_index) {
  uint base_offset = material_index * kMaterialStride;
  return buffer.Load(base_offset + kMaterialIntMediumOffset);
}

uint load_material_ext_medium(ByteAddressBuffer buffer, uint material_index) {
  uint base_offset = material_index * kMaterialStride;
  return buffer.Load(base_offset + kMaterialExtMediumOffset);
}

float load_material_opacity(ByteAddressBuffer buffer, uint material_index) {
  uint base_offset = material_index * kMaterialStride;
  return asfloat(buffer.Load(base_offset + kMaterialOpacityOffset));
}

uint load_emitter_class(ByteAddressBuffer buffer, uint emitter_index) {
  uint base_offset = emitter_index * kEmitterStride;
  return buffer.Load(base_offset + kEmitterClassOffset);
}

uint load_emitter_profile_index(ByteAddressBuffer buffer, uint emitter_index) {
  uint base_offset = emitter_index * kEmitterStride;
  return buffer.Load(base_offset + kEmitterProfileOffset);
}

uint load_emitter_emission_spectrum_index(ByteAddressBuffer buffer, uint emitter_profile_index) {
  uint base_offset = emitter_profile_index * kEmitterProfileStride;
  return buffer.Load(base_offset + kEmitterProfileEmissionSpectrumIndexOffset);
}

uint load_emitter_emission_image_index(ByteAddressBuffer buffer, uint emitter_profile_index) {
  uint base_offset = emitter_profile_index * kEmitterProfileStride;
  return buffer.Load(base_offset + kEmitterProfileEmissionImageIndexOffset);
}

uint load_emitter_profile_class(ByteAddressBuffer buffer, uint emitter_profile_index) {
  uint base_offset = emitter_profile_index * kEmitterProfileStride;
  return buffer.Load(base_offset + kEmitterProfileClassOffset);
}

uint load_emitter_profile_meta(ByteAddressBuffer buffer, uint emitter_profile_index) {
  uint base_offset = emitter_profile_index * kEmitterProfileStride;
  return buffer.Load(base_offset + kEmitterProfileMetaOffset);
}

float3 load_emitter_profile_direction(ByteAddressBuffer buffer, uint emitter_profile_index) {
  uint base_offset = emitter_profile_index * kEmitterProfileStride;
  return load_float3_at_offset(buffer, base_offset + kEmitterProfileDirectionalDirectionOffset);
}

float load_emitter_profile_angular_size_cosine(ByteAddressBuffer buffer, uint emitter_profile_index) {
  uint base_offset = emitter_profile_index * kEmitterProfileStride;
  return asfloat(buffer.Load(base_offset + kEmitterProfileDirectionalAngularSizeCosineOffset));
}

float3 load_spectrum_integrated_value(ByteAddressBuffer buffer, uint spectrum_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride;
  return asfloat(buffer.Load3(base_offset + kSpectralDistributionIntegratedOffset));
}

uint load_spectrum_entry_count(ByteAddressBuffer buffer, uint spectrum_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride;
  return buffer.Load(base_offset + kSpectralDistributionEntryCountOffset);
}

float load_spectrum_entry_wavelength(ByteAddressBuffer buffer, uint spectrum_index, uint entry_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride + kSpectralDistributionEntriesOffset + entry_index * kSpectralDistributionEntryStride;
  return asfloat(buffer.Load(base_offset + 0u));
}

float load_spectrum_entry_power(ByteAddressBuffer buffer, uint spectrum_index, uint entry_index) {
  uint base_offset = spectrum_index * kSpectralDistributionStride + kSpectralDistributionEntriesOffset + entry_index * kSpectralDistributionEntryStride;
  return asfloat(buffer.Load(base_offset + 4u));
}

SpectralResponse load_spectrum_response(ByteAddressBuffer buffer, uint spectrum_index, SpectralQuery spect) {
  if (spectral_query_is_spectral(spect) == false) {
    return spectral_response_make(spect, load_spectrum_integrated_value(buffer, spectrum_index));
  }

  uint entry_count = load_spectrum_entry_count(buffer, spectrum_index);
  if (entry_count == 0u) {
    return spectral_response_make(spect, 0.0f);
  }

  uint begin = 0u;
  uint end = entry_count;
  while ((end - begin) > 1u) {
    uint middle = begin + (end - begin) / 2u;
    float middle_wavelength = load_spectrum_entry_wavelength(buffer, spectrum_index, middle);
    if (middle_wavelength > spect.wavelength) {
      end = middle;
    } else {
      begin = middle;
    }
  }

  uint i = begin;
  if (i >= entry_count) {
    return spectral_response_make(spect, 0.0f);
  }

  float wi = load_spectrum_entry_wavelength(buffer, spectrum_index, i);
  if ((i == 0u) && (spect.wavelength < wi)) {
    return spectral_response_make(spect, 0.0f);
  }

  if (((i + 1u) == entry_count) && (spect.wavelength > wi)) {
    return spectral_response_make(spect, 0.0f);
  }

  uint j = min(i + 1u, entry_count - 1u);
  float wj = load_spectrum_entry_wavelength(buffer, spectrum_index, j);
  float pi = load_spectrum_entry_power(buffer, spectrum_index, i);
  float pj = load_spectrum_entry_power(buffer, spectrum_index, j);
  float t = (i == j) ? 0.0f : (spect.wavelength - wi) / (wj - wi);
  return spectral_response_make(spect, lerp(pi, pj, t));
}

float3 load_camera_position(ByteAddressBuffer camera_buffer) {
  return asfloat(camera_buffer.Load3(kCameraPositionOffset));
}

uint load_camera_class(ByteAddressBuffer camera_buffer) {
  return camera_buffer.Load(kCameraClassOffset);
}

float3 load_camera_direction(ByteAddressBuffer camera_buffer) {
  return asfloat(camera_buffer.Load3(kCameraDirectionOffset));
}

float load_camera_aspect(ByteAddressBuffer camera_buffer) {
  return asfloat(camera_buffer.Load(kCameraAspectOffset));
}

float3 load_camera_side(ByteAddressBuffer camera_buffer) {
  return asfloat(camera_buffer.Load3(kCameraSideOffset));
}

float load_camera_tan_half_fov(ByteAddressBuffer camera_buffer) {
  return asfloat(camera_buffer.Load(kCameraTanHalfFovOffset));
}

float3 load_camera_up(ByteAddressBuffer camera_buffer) {
  return asfloat(camera_buffer.Load3(kCameraUpOffset));
}

uint2 load_camera_film_size(ByteAddressBuffer camera_buffer) {
  return camera_buffer.Load2(kCameraFilmSizeOffset);
}

float load_camera_lens_radius(ByteAddressBuffer camera_buffer) {
  return asfloat(camera_buffer.Load(kCameraLensRadiusOffset));
}

float load_camera_focal_distance(ByteAddressBuffer camera_buffer) {
  return asfloat(camera_buffer.Load(kCameraFocalDistanceOffset));
}

float load_camera_clip_near(ByteAddressBuffer camera_buffer) {
  return asfloat(camera_buffer.Load(kCameraClipNearOffset));
}

float load_camera_clip_far(ByteAddressBuffer camera_buffer) {
  return asfloat(camera_buffer.Load(kCameraClipFarOffset));
}

uint load_camera_lens_image(ByteAddressBuffer camera_buffer) {
  return camera_buffer.Load(kCameraLensImageOffset);
}

uint load_camera_medium_index(ByteAddressBuffer camera_buffer) {
  return camera_buffer.Load(kCameraMediumIndexOffset);
}

uint load_scene_options_samples() {
  if (constants.scene.scene_options == kInvalidIndex) {
    return 1u;
  }

  ByteAddressBuffer scene_options_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_options)];
  uint result = scene_options_buffer.Load(kSceneOptionsSamplesOffset);
  if (result == 0u) {
    return 1u;
  }
  return result;
}

uint load_scene_options_properties_flags() {
  if (constants.scene.scene_options == kInvalidIndex) {
    return 0u;
  }

  ByteAddressBuffer scene_options_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_options)];
  return scene_options_buffer.Load(kSceneOptionsPropertiesFlagsOffset);
}

bool scene_uses_spectral_mode() {
  return (load_scene_options_properties_flags() & (1u << SceneProperty::Spectral)) != 0u;
}

uint load_scene_globals_environment_emitter_count(ByteAddressBuffer scene_globals) {
  return scene_globals.Load(kSceneGlobalsEnvironmentEmitterCountOffset);
}

uint load_scene_globals_environment_emitter(ByteAddressBuffer scene_globals, uint index) {
  return scene_globals.Load(kSceneGlobalsEnvironmentEmittersOffset + index * 4u);
}

uint load_scene_globals_emitter_profile_count(ByteAddressBuffer scene_globals) {
  return scene_globals.Load(kSceneGlobalsEmitterProfileCountOffset);
}

uint load_scene_globals_emitter_instance_count(ByteAddressBuffer scene_globals) {
  return scene_globals.Load(kSceneGlobalsEmitterInstanceCountOffset);
}

bool has_material_spectrum_buffers() {
  return (constants.scene.materials != kInvalidIndex) && (constants.scene.spectrums != kInvalidIndex);
}

bool try_load_material_scattering_state(uint material_index, out uint scattering_spectrum_index, out uint scattering_image_index) {
  scattering_spectrum_index = kInvalidIndex;
  scattering_image_index = kInvalidIndex;
  if (has_material_spectrum_buffers() == false) {
    return false;
  }

  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.materials)];
  scattering_spectrum_index = load_material_scattering_spectrum_index(material_buffer, material_index);
  scattering_image_index = load_material_scattering_image_index(material_buffer, material_index);
  return scattering_spectrum_index != kInvalidIndex;
}

uint image_blob_image_count(ByteAddressBuffer image_blob) {
  return image_blob.Load(kImageBlobHeaderImageCountOffset);
}

uint image_blob_images_offset(ByteAddressBuffer image_blob) {
  return image_blob.Load(kImageBlobHeaderImagesOffset);
}

uint image_blob_data_chunk_count(ByteAddressBuffer image_blob) {
  return image_blob.Load(kImageBlobHeaderDataChunkCountOffset);
}

uint image_blob_data_chunk_indices_offset(ByteAddressBuffer image_blob) {
  return image_blob.Load(kImageBlobHeaderDataChunkIndicesOffset);
}

uint image_desc_base_offset(ByteAddressBuffer image_blob, uint image_index) {
  uint images_offset = image_blob_images_offset(image_blob);
  return images_offset + image_index * kImageDescStride;
}

uint image_desc_format(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescFormatOffset);
}

uint2 image_desc_size(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load2(image_desc_offset + kImageDescISizeOffset);
}

float2 image_desc_fsize(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return asfloat(image_blob.Load2(image_desc_offset + kImageDescFSizeOffset));
}

float2 image_desc_uv_offset(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return asfloat(image_blob.Load2(image_desc_offset + kImageDescOffsetOffset));
}

float2 image_desc_uv_scale(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return asfloat(image_blob.Load2(image_desc_offset + kImageDescScaleOffset));
}

uint image_desc_options(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescOptionsOffset);
}

uint image_desc_pixel_data_offset(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescPixelDataOffset);
}

uint image_desc_x_distribution_entries_offset(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescXDistributionEntriesOffset);
}

uint image_desc_y_distribution_entries_offset(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescYDistributionEntriesOffset);
}

uint image_desc_x_entries_stride(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescXEntriesStrideOffset);
}

uint image_desc_x_distribution_count(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescXDistributionCountOffset);
}

uint image_desc_y_entries_count(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescYEntriesCountOffset);
}

uint image_desc_pixel_data_stride(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescPixelDataStrideOffset);
}

uint image_desc_pixel_data_chunk_index(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescPixelDataChunkIndexOffset);
}

uint image_desc_x_distribution_chunk_index(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescXDistributionChunkIndexOffset);
}

uint image_desc_y_distribution_chunk_index(ByteAddressBuffer image_blob, uint image_desc_offset) {
  return image_blob.Load(image_desc_offset + kImageDescYDistributionChunkIndexOffset);
}

struct ImageDescAccess {
  uint desc_offset;
  uint format;
  uint2 size;
  float2 fsize;
  float2 uv_offset;
  float2 uv_scale;
  uint options;
  uint pixel_data_offset;
  uint x_distribution_entries_offset;
  uint y_distribution_entries_offset;
  uint x_entries_stride;
  uint x_distribution_count;
  uint y_entries_count;
  uint pixel_data_stride;
  uint pixel_data_chunk_index;
  uint x_distribution_chunk_index;
  uint y_distribution_chunk_index;
};

ImageDescAccess load_image_desc_access(ByteAddressBuffer image_blob, uint image_desc_offset) {
  ImageDescAccess result;
  result.desc_offset = image_desc_offset;
  result.format = image_desc_format(image_blob, image_desc_offset);
  result.size = image_desc_size(image_blob, image_desc_offset);
  result.fsize = image_desc_fsize(image_blob, image_desc_offset);
  result.uv_offset = image_desc_uv_offset(image_blob, image_desc_offset);
  result.uv_scale = image_desc_uv_scale(image_blob, image_desc_offset);
  result.options = image_desc_options(image_blob, image_desc_offset);
  result.pixel_data_offset = image_desc_pixel_data_offset(image_blob, image_desc_offset);
  result.x_distribution_entries_offset = image_desc_x_distribution_entries_offset(image_blob, image_desc_offset);
  result.y_distribution_entries_offset = image_desc_y_distribution_entries_offset(image_blob, image_desc_offset);
  result.x_entries_stride = image_desc_x_entries_stride(image_blob, image_desc_offset);
  result.x_distribution_count = image_desc_x_distribution_count(image_blob, image_desc_offset);
  result.y_entries_count = image_desc_y_entries_count(image_blob, image_desc_offset);
  result.pixel_data_stride = image_desc_pixel_data_stride(image_blob, image_desc_offset);
  result.pixel_data_chunk_index = image_desc_pixel_data_chunk_index(image_blob, image_desc_offset);
  result.x_distribution_chunk_index = image_desc_x_distribution_chunk_index(image_blob, image_desc_offset);
  result.y_distribution_chunk_index = image_desc_y_distribution_chunk_index(image_blob, image_desc_offset);
  return result;
}

bool try_load_image_desc_access(ByteAddressBuffer image_blob, uint image_index, out ImageDescAccess image_access) {
  image_access = (ImageDescAccess)0;
  uint image_count = image_blob_image_count(image_blob);
  if (image_index >= image_count) {
    return false;
  }

  uint image_desc_offset = image_desc_base_offset(image_blob, image_index);
  image_access = load_image_desc_access(image_blob, image_desc_offset);
  return true;
}

uint image_chunk_descriptor_index(ByteAddressBuffer image_blob, uint chunk_index) {
  uint data_chunk_count = image_blob_data_chunk_count(image_blob);
  uint data_chunk_indices_offset = image_blob_data_chunk_indices_offset(image_blob);

  if ((chunk_index == kInvalidIndex) || (chunk_index >= data_chunk_count) || (data_chunk_indices_offset == kInvalidIndex)) {
    return kInvalidIndex;
  }

  return image_blob.Load(data_chunk_indices_offset + chunk_index * 4u);
}

uint medium_blob_medium_count(ByteAddressBuffer medium_blob) {
  return medium_blob.Load(kMediumBlobHeaderMediumCountOffset);
}

uint medium_blob_mediums_offset(ByteAddressBuffer medium_blob) {
  return medium_blob.Load(kMediumBlobHeaderMediumsOffset);
}

uint medium_desc_base_offset(ByteAddressBuffer medium_blob, uint medium_index) {
  uint mediums_offset = medium_blob_mediums_offset(medium_blob);
  return mediums_offset + medium_index * kMediumStride;
}

uint medium_blob_data_chunk_count(ByteAddressBuffer medium_blob) {
  return medium_blob.Load(kMediumBlobHeaderDataChunkCountOffset);
}

uint medium_blob_data_chunk_indices_offset(ByteAddressBuffer medium_blob) {
  return medium_blob.Load(kMediumBlobHeaderDataChunkIndicesOffset);
}

uint medium_chunk_descriptor_index(ByteAddressBuffer medium_blob, uint chunk_index) {
  uint data_chunk_count = medium_blob_data_chunk_count(medium_blob);
  uint data_chunk_indices_offset = medium_blob_data_chunk_indices_offset(medium_blob);

  if ((chunk_index == kInvalidIndex) || (chunk_index >= data_chunk_count) || (data_chunk_indices_offset == kInvalidIndex)) {
    return kInvalidIndex;
  }

  return medium_blob.Load(data_chunk_indices_offset + chunk_index * 4u);
}

struct MediumBlobAccess {
  MediumDensitySharedGrid grid;
  float3 bounds_min;
  float3 bounds_max;
  uint density_payload_descriptor_index;
  uint medium_class;
  uint absorption_spectrum_index;
  uint scattering_spectrum_index;
};

MediumBlobAccess load_medium_blob_access(ByteAddressBuffer medium_blob, uint medium_desc_offset) {
  MediumBlobAccess result;
  result.grid.dimensions = medium_blob.Load3(medium_desc_offset + kMediumGridDimensionsOffset);
  result.grid.type = medium_blob.Load(medium_desc_offset + kMediumGridTypeOffset);
  result.grid.noise_type = medium_blob.Load(medium_desc_offset + kMediumGridNoiseTypeOffset);
  result.grid.density_data_offset = medium_blob.Load(medium_desc_offset + kMediumGridDensityDataOffsetOffset);
  result.grid.density_count = medium_blob.Load(medium_desc_offset + kMediumGridDensityCountOffset);
  result.grid.noise_seed = medium_blob.Load(medium_desc_offset + kMediumGridNoiseSeedOffset);
  result.grid.noise_offset = asfloat(medium_blob.Load3(medium_desc_offset + kMediumGridNoiseOffsetOffset));
  result.grid.noise_enable_border_fade = medium_blob.Load(medium_desc_offset + kMediumGridNoiseEnableBorderFadeOffset);
  result.grid.noise_octaves = medium_blob.Load(medium_desc_offset + kMediumGridNoiseOctavesOffset);
  result.grid.noise_scale = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoiseScaleOffset));
  result.grid.noise_lacunarity = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoiseLacunarityOffset));
  result.grid.noise_persistence = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoisePersistenceOffset));
  result.grid.noise_power = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoisePowerOffset));
  result.grid.noise_sharpness = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoiseSharpnessOffset));
  result.grid.noise_border_fade_distance = asfloat(medium_blob.Load(medium_desc_offset + kMediumGridNoiseBorderFadeDistanceOffset));
  result.grid.density_data_chunk_index = medium_blob.Load(medium_desc_offset + kMediumGridDensityDataChunkIndexOffset);
  result.bounds_min = asfloat(medium_blob.Load3(medium_desc_offset + kMediumBoundsMinOffset));
  result.bounds_max = asfloat(medium_blob.Load3(medium_desc_offset + kMediumBoundsMaxOffset));
  result.absorption_spectrum_index = medium_blob.Load(medium_desc_offset + kMediumAbsorptionIndexOffset);
  result.scattering_spectrum_index = medium_blob.Load(medium_desc_offset + kMediumScatteringIndexOffset);
  result.medium_class = load_u16(medium_blob, medium_desc_offset + kMediumClassOffset);
  result.density_payload_descriptor_index = medium_chunk_descriptor_index(medium_blob, result.grid.density_data_chunk_index);
  return result;
}

bool try_get_medium_desc_offset(uint medium_index, out uint medium_desc_offset) {
  medium_desc_offset = kInvalidIndex;
  if ((medium_index == kInvalidIndex) || (constants.scene.mediums == kInvalidIndex)) {
    return false;
  }

  ByteAddressBuffer medium_blob = bindless_buffers[NonUniformResourceIndex(constants.scene.mediums)];
  uint medium_count = medium_blob_medium_count(medium_blob);
  if (medium_index >= medium_count) {
    return false;
  }

  uint mediums_offset = medium_blob_mediums_offset(medium_blob);
  if (mediums_offset == kInvalidIndex) {
    return false;
  }

  medium_desc_offset = medium_desc_base_offset(medium_blob, medium_index);
  return true;
}

float medium_texture_density_value(ByteAddressBuffer payload_buffer, uint density_offset, uint density_count, uint3 dimensions, uint x, uint y, uint z) {
  uint index = x + y * dimensions.x + z * dimensions.x * dimensions.y;
  if (index >= density_count) {
    return 0.0f;
  }
  return asfloat(payload_buffer.Load(density_offset + index * 4u));
}

float medium_sample_texture_3d(ByteAddressBuffer medium_blob, MediumBlobAccess medium_access, float3 local_coord) {
  uint3 dimensions = medium_access.grid.dimensions;
  MediumDensitySharedTextureSample3D sample = (MediumDensitySharedTextureSample3D)0;
  if (medium_density_shared_prepare_texture_sample_3d(local_coord, dimensions, sample) == false) {
    return 0.0f;
  }

  if ((medium_access.grid.density_count == 0u) || (medium_access.grid.density_data_offset == kInvalidIndex) || (medium_access.density_payload_descriptor_index == kInvalidIndex)) {
    return 0.0f;
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(medium_access.density_payload_descriptor_index)];
  float d000 = medium_texture_density_value(payload_buffer, medium_access.grid.density_data_offset, medium_access.grid.density_count, dimensions, sample.ix, sample.iy, sample.iz);
  float d001 = medium_texture_density_value(payload_buffer, medium_access.grid.density_data_offset, medium_access.grid.density_count, dimensions, sample.nx, sample.iy, sample.iz);
  float d010 = medium_texture_density_value(payload_buffer, medium_access.grid.density_data_offset, medium_access.grid.density_count, dimensions, sample.ix, sample.ny, sample.iz);
  float d011 = medium_texture_density_value(payload_buffer, medium_access.grid.density_data_offset, medium_access.grid.density_count, dimensions, sample.nx, sample.ny, sample.iz);
  float d100 = medium_texture_density_value(payload_buffer, medium_access.grid.density_data_offset, medium_access.grid.density_count, dimensions, sample.ix, sample.iy, sample.nz);
  float d101 = medium_texture_density_value(payload_buffer, medium_access.grid.density_data_offset, medium_access.grid.density_count, dimensions, sample.nx, sample.iy, sample.nz);
  float d110 = medium_texture_density_value(payload_buffer, medium_access.grid.density_data_offset, medium_access.grid.density_count, dimensions, sample.ix, sample.ny, sample.nz);
  float d111 = medium_texture_density_value(payload_buffer, medium_access.grid.density_data_offset, medium_access.grid.density_count, dimensions, sample.nx, sample.ny, sample.nz);
  return medium_density_shared_trilerp(d000, d001, d010, d011, d100, d101, d110, d111, sample.dx, sample.dy, sample.dz);
}

float medium_sample_noise(MediumBlobAccess medium_access, float3 local_coord) {
  return medium_density_shared_sample_noise(local_coord, medium_access.bounds_min, medium_access.bounds_max, medium_access.grid.noise_type, medium_access.grid.noise_scale,
    medium_access.grid.noise_octaves, medium_access.grid.noise_lacunarity, medium_access.grid.noise_persistence, medium_access.grid.noise_seed,
    medium_access.grid.noise_offset, medium_access.grid.noise_enable_border_fade, medium_access.grid.noise_border_fade_distance);
}

float medium_sample_density(ByteAddressBuffer medium_blob, MediumBlobAccess medium_access, float3 local_coord) {
  float value = 0.0f;
  if (medium_access.grid.type == MediumGridType::NoiseFunction) {
    value = medium_sample_noise(medium_access, local_coord);
  } else if (medium_access.grid.type == MediumGridType::Texture3D) {
    value = medium_sample_texture_3d(medium_blob, medium_access, local_coord);
  }

  return medium_density_shared_apply_shape(value, medium_access.grid.noise_power, medium_access.grid.noise_sharpness);
}

bool medium_has_grid_data(MediumBlobAccess medium_access) {
  if (medium_density_shared_has_grid_data(medium_access.grid.type, medium_access.grid.dimensions, medium_access.grid.density_count) == false) {
    return false;
  }

  if (medium_access.grid.type == MediumGridType::NoiseFunction) {
    return true;
  }

  return (medium_access.grid.density_data_offset != kInvalidIndex) && (medium_access.density_payload_descriptor_index != kInvalidIndex);
}

struct MediumTransmittanceGPUSharedContext {
  ByteAddressBuffer medium_blob;
  MediumBlobAccess medium_access;
  uint seed;
};

float medium_transmittance_gpu_shared_rnd(inout MediumTransmittanceGPUSharedContext context) {
  return rnd01(context.seed);
}

float medium_transmittance_gpu_shared_density(inout MediumTransmittanceGPUSharedContext context, float3 local_pos) {
  return medium_sample_density(context.medium_blob, context.medium_access, local_pos);
}

#define ETX_MEDIUM_SHARED_CONTEXT_TYPE MediumTransmittanceGPUSharedContext
#define ETX_MEDIUM_SHARED_RND(context) medium_transmittance_gpu_shared_rnd(context)
#define ETX_MEDIUM_SHARED_DENSITY(context, local_pos) medium_transmittance_gpu_shared_density(context, local_pos)
#include <interop/medium_transmittance_shared.hxx>
#undef ETX_MEDIUM_SHARED_DENSITY
#undef ETX_MEDIUM_SHARED_RND
#undef ETX_MEDIUM_SHARED_CONTEXT_TYPE

float3 medium_segment_transmittance_integrated(uint medium_index, float3 origin, float3 direction, float distance, inout uint seed) {
  if ((distance <= 0.0f) || (constants.scene.spectrums == kInvalidIndex)) {
    return float3(1.0f, 1.0f, 1.0f);
  }

  uint medium_desc_offset = kInvalidIndex;
  if (try_get_medium_desc_offset(medium_index, medium_desc_offset) == false) {
    return float3(1.0f, 1.0f, 1.0f);
  }

  ByteAddressBuffer medium_blob = bindless_buffers[NonUniformResourceIndex(constants.scene.mediums)];
  MediumBlobAccess medium_access = load_medium_blob_access(medium_blob, medium_desc_offset);
  uint medium_class = medium_access.medium_class;
  if ((medium_class != Medium::Homogeneous) && (medium_class != Medium::Heterogeneous)) {
    return float3(1.0f, 1.0f, 1.0f);
  }
  uint absorption_index = medium_access.absorption_spectrum_index;
  uint scattering_index = medium_access.scattering_spectrum_index;

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  float3 extinction = float3(0.0f, 0.0f, 0.0f);
  if (absorption_index != kInvalidIndex) {
    extinction += load_spectrum_integrated_value(spectrum_buffer, absorption_index);
  }
  if (scattering_index != kInvalidIndex) {
    extinction += load_spectrum_integrated_value(spectrum_buffer, scattering_index);
  }

  if (medium_class == Medium::Homogeneous) {
    return medium_shared_transmittance_homogeneous_integrated(extinction, distance);
  }
  if (medium_has_grid_data(medium_access) == false) {
    return float3(1.0f, 1.0f, 1.0f);
  }

  MediumTransmittanceGPUSharedContext context;
  context.medium_blob = medium_blob;
  context.medium_access = medium_access;
  context.seed = seed;
  float3 transmittance = medium_shared_transmittance_heterogeneous_integrated(
    extinction, origin, direction, distance, medium_access.bounds_min, medium_access.bounds_max, context);
  seed = context.seed;
  return transmittance;
}

SpectralResponse medium_segment_transmittance_spectral(uint medium_index, float3 origin, float3 direction, float distance, SpectralQuery spect, inout uint seed) {
  SpectralResponse one = spectral_response_make(spect, 1.0f);
  if ((distance <= 0.0f) || (constants.scene.spectrums == kInvalidIndex)) {
    return one;
  }

  uint medium_desc_offset = kInvalidIndex;
  if (try_get_medium_desc_offset(medium_index, medium_desc_offset) == false) {
    return one;
  }

  ByteAddressBuffer medium_blob = bindless_buffers[NonUniformResourceIndex(constants.scene.mediums)];
  MediumBlobAccess medium_access = load_medium_blob_access(medium_blob, medium_desc_offset);
  uint medium_class = medium_access.medium_class;
  if ((medium_class != Medium::Homogeneous) && (medium_class != Medium::Heterogeneous)) {
    return one;
  }
  uint absorption_index = medium_access.absorption_spectrum_index;
  uint scattering_index = medium_access.scattering_spectrum_index;

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  SpectralResponse extinction = spectral_response_make(spect, 0.0f);
  if (absorption_index != kInvalidIndex) {
    extinction = spectral_response_add(extinction, load_spectrum_response(spectrum_buffer, absorption_index, spect));
  }
  if (scattering_index != kInvalidIndex) {
    extinction = spectral_response_add(extinction, load_spectrum_response(spectrum_buffer, scattering_index, spect));
  }

  if (medium_class == Medium::Homogeneous) {
    return medium_shared_transmittance_homogeneous_spectral(extinction, distance);
  }
  if (medium_has_grid_data(medium_access) == false) {
    return one;
  }

  MediumTransmittanceGPUSharedContext context;
  context.medium_blob = medium_blob;
  context.medium_access = medium_access;
  context.seed = seed;
  SpectralResponse transmittance = medium_shared_transmittance_heterogeneous_spectral(
    extinction, origin, direction, distance, medium_access.bounds_min, medium_access.bounds_max, context, spect);
  seed = context.seed;
  return transmittance;
}

float4 evaluate_image(uint image_index, float2 uv);
bool try_load_emitter_scene_state(out uint emitter_instance_count, out uint emitter_profile_count);

DistributionEntry load_distribution_entry(ByteAddressBuffer payload_buffer, uint byte_offset) {
  uint4 raw = payload_buffer.Load4(byte_offset);
  DistributionEntry result;
  result.value = asfloat(raw.x);
  result.pdf = asfloat(raw.y);
  result.cdf = asfloat(raw.z);
  result.reference = raw.w;
  return result;
}

uint sample_distribution(ByteAddressBuffer payload_buffer, uint entries_base_offset, uint count, float rnd, out float pdf) {
  if (count == 0u) {
    pdf = 0.0f;
    return kInvalidIndex;
  }

  DistributionSearchRange search = distribution_search_begin(count);
  while (distribution_search_active(search)) {
    uint middle = distribution_search_middle(search);
    DistributionEntry middle_entry = load_distribution_entry(payload_buffer, entries_base_offset + middle * kDistributionEntryStride);
    distribution_search_update(search, middle, middle_entry.cdf, rnd);
  }

  DistributionEntry result = load_distribution_entry(payload_buffer, entries_base_offset + search.begin * kDistributionEntryStride);
  pdf = result.pdf;
  return search.begin;
}

float2 sample_image_uv(uint image_index, float2 rnd, out float image_pdf, out uint2 location, out float4 eval) {
  image_pdf = 0.0f;
  location = uint2(0u, 0u);
  eval = float4(1.0f, 1.0f, 1.0f, 1.0f);

  if (constants.scene.images == kInvalidIndex) {
    return rnd;
  }

  ByteAddressBuffer image_blob = bindless_buffers[NonUniformResourceIndex(constants.scene.images)];
  ImageDescAccess image_access = (ImageDescAccess)0;
  if (try_load_image_desc_access(image_blob, image_index, image_access) == false) {
    return rnd;
  }

  if ((image_access.fsize.x <= 0.0f) || (image_access.fsize.y <= 0.0f) || (image_access.x_entries_stride == 0u) || (image_access.x_distribution_count == 0u) ||
      (image_access.y_entries_count == 0u) || (image_access.x_distribution_entries_offset == kInvalidIndex) || (image_access.y_distribution_entries_offset == kInvalidIndex) ||
      (image_access.x_distribution_chunk_index == kInvalidIndex) || (image_access.y_distribution_chunk_index == kInvalidIndex)) {
    float2 fallback_uv = rnd;
    eval = evaluate_image(image_index, fallback_uv);
    return fallback_uv;
  }

  uint x_payload_descriptor_index = image_chunk_descriptor_index(image_blob, image_access.x_distribution_chunk_index);
  uint y_payload_descriptor_index = image_chunk_descriptor_index(image_blob, image_access.y_distribution_chunk_index);
  if ((x_payload_descriptor_index == kInvalidIndex) || (y_payload_descriptor_index == kInvalidIndex)) {
    float2 fallback_uv = rnd;
    eval = evaluate_image(image_index, fallback_uv);
    return fallback_uv;
  }

  ByteAddressBuffer x_payload_buffer = bindless_buffers[NonUniformResourceIndex(x_payload_descriptor_index)];
  ByteAddressBuffer y_payload_buffer = bindless_buffers[NonUniformResourceIndex(y_payload_descriptor_index)];

  uint y_count = (image_access.y_entries_count > 0u) ? (image_access.y_entries_count - 1u) : 0u;
  uint x_count = (image_access.x_entries_stride > 0u) ? (image_access.x_entries_stride - 1u) : 0u;
  if ((y_count == 0u) || (x_count == 0u)) {
    float2 fallback_uv = rnd;
    eval = evaluate_image(image_index, fallback_uv);
    return fallback_uv;
  }

  float y_pdf = 0.0f;
  location.y = sample_distribution(y_payload_buffer, image_access.y_distribution_entries_offset, y_count, rnd.y, y_pdf);
  if ((location.y == kInvalidIndex) || (location.y >= image_access.x_distribution_count)) {
    float2 fallback_uv = rnd;
    eval = evaluate_image(image_index, fallback_uv);
    return fallback_uv;
  }

  uint row_base_offset = image_access.x_distribution_entries_offset + location.y * image_access.x_entries_stride * kDistributionEntryStride;
  float x_pdf = 0.0f;
  location.x = sample_distribution(x_payload_buffer, row_base_offset, x_count, rnd.x, x_pdf);
  if (location.x == kInvalidIndex) {
    float2 fallback_uv = rnd;
    eval = evaluate_image(image_index, fallback_uv);
    return fallback_uv;
  }

  DistributionEntry x0 = load_distribution_entry(x_payload_buffer, row_base_offset + location.x * kDistributionEntryStride);
  DistributionEntry x1 = load_distribution_entry(x_payload_buffer, row_base_offset + min(location.x + 1u, x_count - 1u) * kDistributionEntryStride);
  DistributionEntry y0 = load_distribution_entry(y_payload_buffer, image_access.y_distribution_entries_offset + location.y * kDistributionEntryStride);
  DistributionEntry y1 = load_distribution_entry(y_payload_buffer, image_access.y_distribution_entries_offset + min(location.y + 1u, y_count - 1u) * kDistributionEntryStride);

  float2 uv = image_sample_uv_from_distribution(rnd, location, image_access.fsize, x0.cdf, x1.cdf, y0.cdf, y1.cdf);
  eval = evaluate_image(image_index, uv);
  image_pdf = x_pdf * y_pdf;
  return uv;
}

float4 load_image_pixel(ByteAddressBuffer payload_buffer, uint format, uint byte_offset) {
  if (format == (uint)Image::Format::RGBA32F) {
    return asfloat(payload_buffer.Load4(byte_offset));
  }

  if (format == (uint)Image::Format::RGBA8) {
    uint packed_rgba = payload_buffer.Load(byte_offset);
    float4 result = float4(
      float((packed_rgba >> 0u) & 0xFFu),
      float((packed_rgba >> 8u) & 0xFFu),
      float((packed_rgba >> 16u) & 0xFFu),
      float((packed_rgba >> 24u) & 0xFFu));
    return result * (1.0f / 255.0f);
  }

  return float4(1.0f, 1.0f, 1.0f, 1.0f);
}

float4 evaluate_image(uint image_index, float2 uv) {
  if (constants.scene.images == kInvalidIndex) {
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
  }

  ByteAddressBuffer image_blob = bindless_buffers[NonUniformResourceIndex(constants.scene.images)];
  ImageDescAccess image_access = (ImageDescAccess)0;
  if (try_load_image_desc_access(image_blob, image_index, image_access) == false) {
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
  }

  if ((image_access.pixel_data_offset == kInvalidIndex) || (image_access.pixel_data_stride == 0u) || (image_access.size.x == 0u) || (image_access.size.y == 0u)) {
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
  }

  uint payload_descriptor_index = image_chunk_descriptor_index(image_blob, image_access.pixel_data_chunk_index);
  if (payload_descriptor_index == kInvalidIndex) {
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(payload_descriptor_index)];

  float2 image_uv = uv * image_access.fsize;
  float x0 = image_tex_coord_u(image_uv.x, image_access.fsize.x, image_access.options);
  float y0 = image_tex_coord_v(image_uv.y, image_access.fsize.y, image_access.options);

  float dx = x0 - floor(x0);
  float dy = y0 - floor(y0);

  uint row_0 = clamp(uint(y0), 0u, image_access.size.y - 1u);
  uint row_1 = clamp(row_0 + 1u, 0u, image_access.size.y - 1u);
  uint col_0 = clamp(uint(x0), 0u, image_access.size.x - 1u);
  uint col_1 = clamp(col_0 + 1u, 0u, image_access.size.x - 1u);

  uint pixel_offset_00 = image_access.pixel_data_offset + ((row_0 * image_access.size.x + col_0) * image_access.pixel_data_stride);
  uint pixel_offset_01 = image_access.pixel_data_offset + ((row_0 * image_access.size.x + col_1) * image_access.pixel_data_stride);
  uint pixel_offset_10 = image_access.pixel_data_offset + ((row_1 * image_access.size.x + col_0) * image_access.pixel_data_stride);
  uint pixel_offset_11 = image_access.pixel_data_offset + ((row_1 * image_access.size.x + col_1) * image_access.pixel_data_stride);

  float4 p00 = load_image_pixel(payload_buffer, image_access.format, pixel_offset_00);
  float4 p01 = load_image_pixel(payload_buffer, image_access.format, pixel_offset_01);
  float4 p10 = load_image_pixel(payload_buffer, image_access.format, pixel_offset_10);
  float4 p11 = load_image_pixel(payload_buffer, image_access.format, pixel_offset_11);

  return p00 * (1.0f - dx) * (1.0f - dy) + p01 * dx * (1.0f - dy) + p10 * (1.0f - dx) * dy + p11 * dx * dy;
}

bool image_has_alpha_channel(uint image_index) {
  if (constants.scene.images == kInvalidIndex) {
    return false;
  }

  ByteAddressBuffer image_blob = bindless_buffers[NonUniformResourceIndex(constants.scene.images)];
  ImageDescAccess image_access = (ImageDescAccess)0;
  if (try_load_image_desc_access(image_blob, image_index, image_access) == false) {
    return false;
  }

  return (image_access.options & Image::HasAlphaChannel) != 0u;
}

float3 make_barycentrics(float2 bary) {
  return float3(1.0f - bary.x - bary.y, bary.x, bary.y);
}

struct SurfacePoint {
  float3 barycentrics;
  Vertex vertex;
  float3 geo_normal;
};

SurfacePoint load_surface_point(ByteAddressBuffer position_buffer, ByteAddressBuffer normal_buffer, ByteAddressBuffer tangent_buffer, ByteAddressBuffer bitangent_buffer,
  ByteAddressBuffer texcoord_buffer, bool has_surface_frame, bool has_texcoords, TriangleData tri, float2 bary, float3 ray_dir) {
  SurfacePoint result;
  result.barycentrics = make_barycentrics(bary);

  float3 p0 = load_float3(position_buffer, tri.i.x);
  float3 p1 = load_float3(position_buffer, tri.i.y);
  float3 p2 = load_float3(position_buffer, tri.i.z);
  result.vertex.pos = p0 * result.barycentrics.x + p1 * result.barycentrics.y + p2 * result.barycentrics.z;

  float3 n0 = load_float3(normal_buffer, tri.i.x);
  float3 n1 = load_float3(normal_buffer, tri.i.y);
  float3 n2 = load_float3(normal_buffer, tri.i.z);
  result.vertex.nrm = normalize(n0 * result.barycentrics.x + n1 * result.barycentrics.y + n2 * result.barycentrics.z);

  if (has_surface_frame) {
    float3 t0 = load_float3(tangent_buffer, tri.i.x);
    float3 t1 = load_float3(tangent_buffer, tri.i.y);
    float3 t2 = load_float3(tangent_buffer, tri.i.z);
    result.vertex.tan = t0 * result.barycentrics.x + t1 * result.barycentrics.y + t2 * result.barycentrics.z;

    float3 b0 = load_float3(bitangent_buffer, tri.i.x);
    float3 b1 = load_float3(bitangent_buffer, tri.i.y);
    float3 b2 = load_float3(bitangent_buffer, tri.i.z);
    result.vertex.btn = b0 * result.barycentrics.x + b1 * result.barycentrics.y + b2 * result.barycentrics.z;
  } else {
    result.vertex.tan = float3(0.0f, 0.0f, 0.0f);
    result.vertex.btn = float3(0.0f, 0.0f, 0.0f);
  }

  if (has_texcoords) {
    float2 t0 = load_float2(texcoord_buffer, tri.i.x);
    float2 t1 = load_float2(texcoord_buffer, tri.i.y);
    float2 t2 = load_float2(texcoord_buffer, tri.i.z);
    result.vertex.tex = t0 * result.barycentrics.x + t1 * result.barycentrics.y + t2 * result.barycentrics.z;
  } else {
    result.vertex.tex = float2(0.0f, 0.0f);
  }

  result.geo_normal = normalize(tri.geo_n);
  if (dot(result.geo_normal, ray_dir) > 0.0f) {
    result.geo_normal = -result.geo_normal;
  }
  if (dot(result.vertex.nrm, result.geo_normal) < 0.0f) {
    result.vertex.nrm = -result.vertex.nrm;
  }

  return result;
}

float2 interpolate_uv_from_barycentrics(ByteAddressBuffer texcoord_buffer, TriangleData tri, float3 bc) {
  float2 t0 = load_float2(texcoord_buffer, tri.i.x);
  float2 t1 = load_float2(texcoord_buffer, tri.i.y);
  float2 t2 = load_float2(texcoord_buffer, tri.i.z);
  return t0 * bc.x + t1 * bc.y + t2 * bc.z;
}

float2 interpolate_uv(ByteAddressBuffer texcoord_buffer, TriangleData tri, float2 bary) {
  return interpolate_uv_from_barycentrics(texcoord_buffer, tri, make_barycentrics(bary));
}

bool alpha_test_pass(uint material_index, float2 uv, inout uint seed) {
  if (constants.scene.materials == kInvalidIndex) {
    return false;
  }

  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.materials)];
  uint material_class = load_material_class(material_buffer, material_index);
  if ((material_class == MaterialClass::Void) || (material_class == MaterialClass::Boundary)) {
    return true;
  }

  float material_alpha = load_material_opacity(material_buffer, material_index);
  float alpha_diffuse = 1.0f;
  uint scattering_image_index = load_material_scattering_image_index(material_buffer, material_index);
  if ((scattering_image_index != kInvalidIndex) && image_has_alpha_channel(scattering_image_index)) {
    alpha_diffuse = evaluate_image(scattering_image_index, uv).w;
  }

  float alpha_test_value = alpha_diffuse * material_alpha;
  return alpha_test_value <= rnd01(seed);
}

float3 default_ao_shading(float3 hit_normal, float ao) {
  float n_dot_up = saturate(dot(hit_normal, float3(0.0f, 1.0f, 0.0f)));
  float3 base = lerp(float3(0.35f, 0.37f, 0.42f), float3(0.85f, 0.87f, 0.9f), n_dot_up);
  return base * ao;
}

SpectralResponse evaluate_material_scattering_spectral(uint material_index, float2 uv, float ao, SpectralQuery spect, SpectralResponse fallback_value) {
  uint scattering_spectrum_index = kInvalidIndex;
  uint scattering_image_index = kInvalidIndex;
  if (try_load_material_scattering_state(material_index, scattering_spectrum_index, scattering_image_index) == false) {
    return fallback_value;
  }

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  SpectralResponse result = load_spectrum_response(spectrum_buffer, scattering_spectrum_index, spect);
  if ((scattering_image_index != kInvalidIndex) && (constants.scene.images != kInvalidIndex)) {
    float4 image_eval = evaluate_image(scattering_image_index, uv);
    result = spectral_response_apply_rgb_scale(spect, result, image_eval.xyz);
  }

  result = spectral_response_mul(result, ao);
  return spectral_response_clamp_non_negative(result);
}

float3 apply_image_integrated_or_fallback(uint material_index, float2 uv, float ao, float3 fallback_color) {
  uint scattering_spectrum_index = kInvalidIndex;
  uint scattering_image_index = kInvalidIndex;
  if (try_load_material_scattering_state(material_index, scattering_spectrum_index, scattering_image_index) == false) {
    return fallback_color;
  }

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  float3 scattering_integrated = load_spectrum_integrated_value(spectrum_buffer, scattering_spectrum_index);
  float3 result = spectral_rgb_clamp_non_negative(scattering_integrated);

  if ((scattering_image_index != kInvalidIndex) && (constants.scene.images != kInvalidIndex)) {
    float4 image_eval = evaluate_image(scattering_image_index, uv);
    result *= image_eval.xyz;
  }

  return spectral_rgb_clamp_non_negative(result) * ao;
}

bool try_load_emitter_scene_state(out uint emitter_instance_count, out uint emitter_profile_count) {
  emitter_instance_count = 0u;
  emitter_profile_count = 0u;

  if ((constants.scene.emitter_instances == kInvalidIndex) || (constants.scene.emitter_profiles == kInvalidIndex) || (constants.scene.spectrums == kInvalidIndex) ||
      (constants.scene.scene_globals == kInvalidIndex)) {
    return false;
  }

  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  emitter_instance_count = load_scene_globals_emitter_instance_count(scene_globals);
  emitter_profile_count = load_scene_globals_emitter_profile_count(scene_globals);
  return true;
}

struct EmitterEmissionAccess {
  uint emitter_class;
  uint emitter_profile_index;
  uint emitter_profile_class;
  uint emitter_profile_meta;
  uint emission_spectrum_index;
  uint emission_image_index;
  float3 emitter_direction;
  float emitter_angular_size_cosine;
};

bool try_load_emitter_emission_access(uint emitter_index, out EmitterEmissionAccess access) {
  access = (EmitterEmissionAccess)0;
  access.emitter_class = EmitterClass::Area;
  access.emitter_profile_index = kInvalidIndex;
  access.emitter_profile_class = EmitterClass::Area;
  access.emitter_profile_meta = 0u;
  access.emission_spectrum_index = kInvalidIndex;
  access.emission_image_index = kInvalidIndex;
  access.emitter_direction = float3(0.0f, 0.0f, 1.0f);
  access.emitter_angular_size_cosine = -1.0f;

  uint emitter_instance_count = 0u;
  uint emitter_profile_count = 0u;
  if (try_load_emitter_scene_state(emitter_instance_count, emitter_profile_count) == false) {
    return false;
  }
  if (emitter_index >= emitter_instance_count) {
    return false;
  }

  ByteAddressBuffer emitter_instance_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_instances)];
  ByteAddressBuffer emitter_profile_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_profiles)];

  access.emitter_class = load_emitter_class(emitter_instance_buffer, emitter_index);
  access.emitter_profile_index = load_emitter_profile_index(emitter_instance_buffer, emitter_index);
  if (access.emitter_profile_index >= emitter_profile_count) {
    return false;
  }

  access.emission_spectrum_index = load_emitter_emission_spectrum_index(emitter_profile_buffer, access.emitter_profile_index);
  access.emission_image_index = load_emitter_emission_image_index(emitter_profile_buffer, access.emitter_profile_index);
  if (access.emission_spectrum_index == kInvalidIndex) {
    return false;
  }

  access.emitter_profile_class = load_emitter_profile_class(emitter_profile_buffer, access.emitter_profile_index);
  access.emitter_profile_meta = load_emitter_profile_meta(emitter_profile_buffer, access.emitter_profile_index);
  access.emitter_direction = load_emitter_profile_direction(emitter_profile_buffer, access.emitter_profile_index);
  access.emitter_angular_size_cosine = load_emitter_profile_angular_size_cosine(emitter_profile_buffer, access.emitter_profile_index);
  return true;
}

bool emitter_emission_access_accepts_direction(EmitterEmissionAccess access, float3 direction) {
  return dot(normalize(direction), normalize(access.emitter_direction)) >= access.emitter_angular_size_cosine;
}

bool try_load_local_emission_access(uint emitter_index, out EmitterEmissionAccess access) {
  if (try_load_emitter_emission_access(emitter_index, access) == false) {
    return false;
  }

  return access.emitter_class == EmitterClass::Area;
}

bool try_load_distant_emission_access(uint emitter_index, float3 direction, out EmitterEmissionAccess access) {
  if (try_load_emitter_emission_access(emitter_index, access) == false) {
    return false;
  }
  if (access.emitter_class == EmitterClass::Area) {
    return false;
  }

  if ((access.emitter_class == EmitterClass::Directional) && (access.emitter_profile_class == EmitterClass::Directional)) {
    if (emitter_emission_access_accepts_direction(access, direction) == false) {
      return false;
    }
  }

  return true;
}

float3 evaluate_emission_integrated_source(uint emission_spectrum_index, uint emission_image_index, float2 uv) {
  if ((constants.scene.spectrums == kInvalidIndex) || (emission_spectrum_index == kInvalidIndex)) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  float3 result = spectral_rgb_clamp_non_negative(load_spectrum_integrated_value(spectrum_buffer, emission_spectrum_index));
  if ((emission_image_index != kInvalidIndex) && (constants.scene.images != kInvalidIndex)) {
    result *= evaluate_image(emission_image_index, uv).xyz;
  }
  return spectral_rgb_clamp_non_negative(result);
}

SpectralResponse evaluate_emission_spectral_source(uint emission_spectrum_index, uint emission_image_index, float2 uv, SpectralQuery spect) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  if ((constants.scene.spectrums == kInvalidIndex) || (emission_spectrum_index == kInvalidIndex)) {
    return zero_value;
  }

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  SpectralResponse result = load_spectrum_response(spectrum_buffer, emission_spectrum_index, spect);
  if ((emission_image_index != kInvalidIndex) && (constants.scene.images != kInvalidIndex)) {
    float4 image_eval = evaluate_image(emission_image_index, uv);
    result = spectral_response_apply_rgb_scale(spect, result, image_eval.xyz);
  }
  return spectral_response_clamp_non_negative(result);
}

float3 evaluate_local_emission_integrated(uint emitter_index, float2 uv) {
  EmitterEmissionAccess access = (EmitterEmissionAccess)0;
  if (try_load_local_emission_access(emitter_index, access) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  return evaluate_emission_integrated_source(access.emission_spectrum_index, access.emission_image_index, uv);
}

SpectralResponse evaluate_local_emission_spectral(uint emitter_index, float2 uv, SpectralQuery spect) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  EmitterEmissionAccess access = (EmitterEmissionAccess)0;
  if (try_load_local_emission_access(emitter_index, access) == false) {
    return zero_value;
  }

  return evaluate_emission_spectral_source(access.emission_spectrum_index, access.emission_image_index, uv, spect);
}

uint environment_projection_mode(uint emitter_profile_meta) {
  bool is_atmosphere = (emitter_profile_meta & EmitterProfileMeta::Atmosphere) != 0u;
  return is_atmosphere ? Projection::EqualArea : Projection::Equirectangular;
}

float2 environment_emission_uv(uint emitter_class, uint emitter_profile_meta, uint emission_image_index, float3 direction) {
  if (emitter_class != EmitterClass::Environment) {
    return float2(0.5f, 0.5f);
  }

  float2 image_offset = float2(0.0f, 0.0f);
  float image_u_scale = 1.0f;
  if ((constants.scene.images != kInvalidIndex) && (emission_image_index != kInvalidIndex)) {
    ByteAddressBuffer image_blob = bindless_buffers[NonUniformResourceIndex(constants.scene.images)];
    ImageDescAccess image_access = (ImageDescAccess)0;
    if (try_load_image_desc_access(image_blob, emission_image_index, image_access)) {
      image_offset = image_access.uv_offset;
      image_u_scale = image_access.uv_scale.x;
    }
  }

  uint projection_mode = environment_projection_mode(emitter_profile_meta);
  return direction_to_uv(normalize(direction), image_offset, image_u_scale, projection_mode);
}

float3 evaluate_distant_emission_integrated(uint emitter_index, float3 direction) {
  EmitterEmissionAccess access = (EmitterEmissionAccess)0;
  if (try_load_distant_emission_access(emitter_index, direction, access) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  float2 uv = environment_emission_uv(access.emitter_class, access.emitter_profile_meta, access.emission_image_index, direction);
  return evaluate_emission_integrated_source(access.emission_spectrum_index, access.emission_image_index, uv);
}

SpectralResponse evaluate_distant_emission_spectral(uint emitter_index, float3 direction, SpectralQuery spect) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  EmitterEmissionAccess access = (EmitterEmissionAccess)0;
  if (try_load_distant_emission_access(emitter_index, direction, access) == false) {
    return zero_value;
  }

  float2 uv = environment_emission_uv(access.emitter_class, access.emitter_profile_meta, access.emission_image_index, direction);
  return evaluate_emission_spectral_source(access.emission_spectrum_index, access.emission_image_index, uv, spect);
}

bool try_select_environment_emitter_random(inout uint seed, out uint emitter_index, out uint emitter_count) {
  emitter_index = kInvalidIndex;
  emitter_count = 0u;

  if (constants.scene.scene_globals == kInvalidIndex) {
    return false;
  }

  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  uint emitter_instance_count = load_scene_globals_emitter_instance_count(scene_globals);
  emitter_count = min(load_scene_globals_environment_emitter_count(scene_globals), SceneLimits::MaxEnvironmentEmitters);
  if (emitter_count == 0u) {
    return false;
  }

  uint selected = min(uint(rnd01(seed) * float(emitter_count)), emitter_count - 1u);
  emitter_index = load_scene_globals_environment_emitter(scene_globals, selected);
  if (emitter_index >= emitter_instance_count) {
    emitter_index = kInvalidIndex;
    return false;
  }

  return true;
}

float3 sample_distant_emission_integrated_random(float3 direction, inout uint seed) {
  uint emitter_index = kInvalidIndex;
  uint emitter_count = 0u;
  if (try_select_environment_emitter_random(seed, emitter_index, emitter_count) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  float3 sample_value = evaluate_distant_emission_integrated(emitter_index, direction);
  return sample_value * float(emitter_count);
}

SpectralResponse sample_distant_emission_spectral_random(float3 direction, SpectralQuery spect, inout uint seed) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  uint emitter_index = kInvalidIndex;
  uint emitter_count = 0u;
  if (try_select_environment_emitter_random(seed, emitter_index, emitter_count) == false) {
    return zero_value;
  }

  SpectralResponse sample_value = evaluate_distant_emission_spectral(emitter_index, direction, spect);
  return spectral_response_mul(sample_value, float(emitter_count));
}

