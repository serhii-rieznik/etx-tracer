#pragma once

#include "bindless.hlsl"

#include <interop/geometry.hxx>
#include <interop/gpu_abi_constants.hxx>
#include <interop/hit_policy.hxx>
#include <interop/image.hxx>
#include <interop/image_filter_shared.hxx>
#include <interop/material.hxx>
#include <interop/material_scattering_shared.hxx>
#include <interop/projection.hxx>
#include <interop/camera_shared.hxx>
#include <interop/camera_film_shared.hxx>
#include <interop/scene_resource_shared.hxx>
#include <interop/distribution.hxx>
#include <interop/sampler_policy.hxx>
#include <interop/sampler.hxx>
#include <interop/medium_density_shared.hxx>
#include <interop/surface_point_shared.hxx>
#include <interop/scene_math_shared.hxx>

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

struct GPUABIAccessSharedContext {
  ByteAddressBuffer buffer;
};

uint gpu_abi_access_shared_load_u32(GPUABIAccessSharedContext context, uint byte_offset) {
  return context.buffer.Load(byte_offset);
}

float gpu_abi_access_shared_load_f32(GPUABIAccessSharedContext context, uint byte_offset) {
  return asfloat(context.buffer.Load(byte_offset));
}

float3 gpu_abi_access_shared_load_f32x3(GPUABIAccessSharedContext context, uint byte_offset) {
  return load_float3_at_offset(context.buffer, byte_offset);
}

uint2 gpu_abi_access_shared_load_u32x2(GPUABIAccessSharedContext context, uint byte_offset) {
  return context.buffer.Load2(byte_offset);
}

#define ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE GPUABIAccessSharedContext
#define ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, byte_offset) gpu_abi_access_shared_load_u32(context, byte_offset)
#define ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32(context, byte_offset) gpu_abi_access_shared_load_f32(context, byte_offset)
#define ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32X3(context, byte_offset) gpu_abi_access_shared_load_f32x3(context, byte_offset)
#define ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32X2(context, byte_offset) gpu_abi_access_shared_load_u32x2(context, byte_offset)
#include <interop/gpu_abi_access_shared.hxx>
#undef ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32X2
#undef ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32X3
#undef ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32
#undef ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32
#undef ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE

GPUABIAccessSharedContext make_gpu_abi_access_shared_context(ByteAddressBuffer buffer) {
  GPUABIAccessSharedContext context;
  context.buffer = buffer;
  return context;
}

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

float3 load_spectrum_integrated_value(ByteAddressBuffer buffer, uint spectrum_index) {
  SpectrumAccessGPUSharedContext context = {buffer};
  return spectrum_access_shared_integrated(context, spectrum_index);
}

SpectralResponse load_spectrum_response(ByteAddressBuffer buffer, uint spectrum_index, SpectralQuery spect) {
  SpectrumAccessGPUSharedContext context = {buffer};
  return spectrum_access_shared_query(context, spectrum_index, spect);
}

Camera load_camera(ByteAddressBuffer camera_buffer) {
  GPUABIAccessSharedContext context = make_gpu_abi_access_shared_context(camera_buffer);
  ETX_ZERO_INIT(Camera, camera);
  camera.position = gpu_abi_access_shared_camera_position(context);
  camera.cls = gpu_abi_access_shared_camera_class(context);
  camera.direction = gpu_abi_access_shared_camera_direction(context);
  camera.aspect = gpu_abi_access_shared_camera_aspect(context);
  camera.side = gpu_abi_access_shared_camera_side(context);
  camera.tan_half_fov = gpu_abi_access_shared_camera_tan_half_fov(context);
  camera.up = gpu_abi_access_shared_camera_up(context);
  camera.lens_radius = gpu_abi_access_shared_camera_lens_radius(context);
  camera.focal_distance = gpu_abi_access_shared_camera_focal_distance(context);
  camera.clip_near = gpu_abi_access_shared_camera_clip_near(context);
  camera.clip_far = gpu_abi_access_shared_camera_clip_far(context);
  camera.lens_image = gpu_abi_access_shared_camera_lens_image(context);
  camera.medium_index = gpu_abi_access_shared_camera_medium_index(context);
  return camera;
}

struct SceneOptionsGPUSharedContext {
  uint scene_options_descriptor_index;
};

bool scene_options_shared_gpu_has_data(SceneOptionsGPUSharedContext context) {
  return context.scene_options_descriptor_index != kInvalidIndex;
}

uint scene_options_shared_gpu_load_u32(SceneOptionsGPUSharedContext context, uint byte_offset) {
  ByteAddressBuffer scene_options_buffer = bindless_buffers[NonUniformResourceIndex(context.scene_options_descriptor_index)];
  return scene_options_buffer.Load(byte_offset);
}

#define ETX_SCENE_OPTIONS_SHARED_CONTEXT_TYPE SceneOptionsGPUSharedContext
#define ETX_SCENE_OPTIONS_SHARED_HAS_DATA(context) scene_options_shared_gpu_has_data(context)
#define ETX_SCENE_OPTIONS_SHARED_LOAD_U32(context, byte_offset) scene_options_shared_gpu_load_u32(context, byte_offset)
#include <interop/scene_options_shared.hxx>
#undef ETX_SCENE_OPTIONS_SHARED_LOAD_U32
#undef ETX_SCENE_OPTIONS_SHARED_HAS_DATA
#undef ETX_SCENE_OPTIONS_SHARED_CONTEXT_TYPE

SceneOptionsGPUSharedContext make_scene_options_gpu_shared_context(uint scene_options_descriptor_index) {
  SceneOptionsGPUSharedContext context;
  context.scene_options_descriptor_index = scene_options_descriptor_index;
  return context;
}

uint load_scene_options_samples() {
  SceneOptionsGPUSharedContext context = make_scene_options_gpu_shared_context(constants.scene.scene_options);
  return scene_options_shared_samples(context);
}

uint load_scene_options_properties_flags() {
  SceneOptionsGPUSharedContext context = make_scene_options_gpu_shared_context(constants.scene.scene_options);
  return scene_options_shared_properties_flags(context);
}

bool scene_uses_spectral_mode() {
  SceneOptionsGPUSharedContext context = make_scene_options_gpu_shared_context(constants.scene.scene_options);
  return scene_options_shared_uses_spectral_mode(context);
}

struct SceneGlobalsGPUSharedContext {
  ByteAddressBuffer scene_globals;
};

uint scene_globals_gpu_shared_load_u32(SceneGlobalsGPUSharedContext context, uint byte_offset) {
  return context.scene_globals.Load(byte_offset);
}

float scene_globals_gpu_shared_load_f32(SceneGlobalsGPUSharedContext context, uint byte_offset) {
  return asfloat(context.scene_globals.Load(byte_offset));
}

#define ETX_SCENE_GLOBALS_SHARED_CONTEXT_TYPE SceneGlobalsGPUSharedContext
#define ETX_SCENE_GLOBALS_SHARED_LOAD_U32(context, byte_offset) scene_globals_gpu_shared_load_u32(context, byte_offset)
#define ETX_SCENE_GLOBALS_SHARED_LOAD_F32(context, byte_offset) scene_globals_gpu_shared_load_f32(context, byte_offset)
#include <interop/scene_globals_shared.hxx>
#undef ETX_SCENE_GLOBALS_SHARED_LOAD_F32
#undef ETX_SCENE_GLOBALS_SHARED_LOAD_U32
#undef ETX_SCENE_GLOBALS_SHARED_CONTEXT_TYPE

SceneGlobalsGPUSharedContext make_scene_globals_gpu_shared_context(ByteAddressBuffer scene_globals) {
  SceneGlobalsGPUSharedContext context;
  context.scene_globals = scene_globals;
  return context;
}

bool has_material_spectrum_buffers() {
  return scene_resource_shared_has_material_spectrum_buffers(constants.scene.materials, constants.scene.spectrums);
}

bool material_scattering_access_shared_gpu_has_required_scene_buffers(uint context) {
  return has_material_spectrum_buffers();
}

uint material_scattering_access_shared_gpu_load_scattering_spectrum_index(uint context, uint material_index) {
  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.materials)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(material_buffer);
  return gpu_abi_access_shared_material_scattering_spectrum_index(access_context, material_index);
}

uint material_scattering_access_shared_gpu_load_scattering_image_index(uint context, uint material_index) {
  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.materials)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(material_buffer);
  return gpu_abi_access_shared_material_scattering_image_index(access_context, material_index);
}

#define ETX_MATERIAL_SCATTERING_ACCESS_SHARED_CONTEXT_TYPE uint
#define ETX_MATERIAL_SCATTERING_ACCESS_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) material_scattering_access_shared_gpu_has_required_scene_buffers(context)
#define ETX_MATERIAL_SCATTERING_ACCESS_SHARED_LOAD_SCATTERING_SPECTRUM_INDEX(context, material_index) \
  material_scattering_access_shared_gpu_load_scattering_spectrum_index(context, material_index)
#define ETX_MATERIAL_SCATTERING_ACCESS_SHARED_LOAD_SCATTERING_IMAGE_INDEX(context, material_index) \
  material_scattering_access_shared_gpu_load_scattering_image_index(context, material_index)
#include <interop/material_scattering_access_shared.hxx>
#undef ETX_MATERIAL_SCATTERING_ACCESS_SHARED_LOAD_SCATTERING_IMAGE_INDEX
#undef ETX_MATERIAL_SCATTERING_ACCESS_SHARED_LOAD_SCATTERING_SPECTRUM_INDEX
#undef ETX_MATERIAL_SCATTERING_ACCESS_SHARED_HAS_REQUIRED_SCENE_BUFFERS
#undef ETX_MATERIAL_SCATTERING_ACCESS_SHARED_CONTEXT_TYPE

bool try_load_material_scattering_state(uint material_index, out uint scattering_spectrum_index, out uint scattering_image_index) {
  uint context = 0u;
  return material_scattering_access_shared_try_load(context, material_index, scattering_spectrum_index, scattering_image_index);
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

#define ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE ByteAddressBuffer
#define ETX_IMAGE_BLOB_ACCESS_SHARED_DESC_TYPE ImageDescAccess
#define ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, byte_offset) context.Load(byte_offset)
#define ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32X2(context, byte_offset) context.Load2(byte_offset)
#define ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_F32X2(context, byte_offset) asfloat(context.Load2(byte_offset))
#include <interop/image_blob_access_shared.hxx>
#undef ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_F32X2
#undef ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32X2
#undef ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32
#undef ETX_IMAGE_BLOB_ACCESS_SHARED_DESC_TYPE
#undef ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE

struct ImageSceneAccessGPUContext {
  uint images_descriptor_index;
};

bool image_scene_access_shared_gpu_has_images(ImageSceneAccessGPUContext context) {
  return scene_resource_shared_is_available(context.images_descriptor_index);
}

bool image_scene_access_shared_gpu_load_desc(ImageSceneAccessGPUContext context, uint image_index, out ImageDescAccess image_access) {
  ByteAddressBuffer image_blob = bindless_buffers[NonUniformResourceIndex(context.images_descriptor_index)];
  return image_blob_access_shared_try_load_desc(image_blob, image_index, image_access);
}

uint image_scene_access_shared_gpu_chunk_descriptor(ImageSceneAccessGPUContext context, uint chunk_index) {
  ByteAddressBuffer image_blob = bindless_buffers[NonUniformResourceIndex(context.images_descriptor_index)];
  return image_blob_access_shared_chunk_descriptor_index(image_blob, chunk_index);
}

#define ETX_IMAGE_SCENE_ACCESS_SHARED_CONTEXT_TYPE ImageSceneAccessGPUContext
#define ETX_IMAGE_SCENE_ACCESS_SHARED_DESC_TYPE ImageDescAccess
#define ETX_IMAGE_SCENE_ACCESS_SHARED_HAS_IMAGES(context) image_scene_access_shared_gpu_has_images(context)
#define ETX_IMAGE_SCENE_ACCESS_SHARED_LOAD_DESC(context, image_index, image_access) image_scene_access_shared_gpu_load_desc(context, image_index, image_access)
#define ETX_IMAGE_SCENE_ACCESS_SHARED_CHUNK_DESCRIPTOR(context, chunk_index) image_scene_access_shared_gpu_chunk_descriptor(context, chunk_index)
#include <interop/image_scene_access_shared.hxx>
#undef ETX_IMAGE_SCENE_ACCESS_SHARED_CHUNK_DESCRIPTOR
#undef ETX_IMAGE_SCENE_ACCESS_SHARED_LOAD_DESC
#undef ETX_IMAGE_SCENE_ACCESS_SHARED_HAS_IMAGES
#undef ETX_IMAGE_SCENE_ACCESS_SHARED_DESC_TYPE
#undef ETX_IMAGE_SCENE_ACCESS_SHARED_CONTEXT_TYPE

#define ETX_MEDIUM_BLOB_ACCESS_SHARED_CONTEXT_TYPE ByteAddressBuffer
#define ETX_MEDIUM_BLOB_ACCESS_SHARED_LOAD_U32(context, byte_offset) context.Load(byte_offset)
#include <interop/medium_blob_access_shared.hxx>
#undef ETX_MEDIUM_BLOB_ACCESS_SHARED_LOAD_U32
#undef ETX_MEDIUM_BLOB_ACCESS_SHARED_CONTEXT_TYPE

struct MediumBlobAccess {
  MediumDensitySharedGrid grid;
  float3 bounds_min;
  float3 bounds_max;
  uint density_payload_descriptor_index;
  uint medium_class;
  uint absorption_spectrum_index;
  uint scattering_spectrum_index;
};

uint medium_extinction_shared_gpu_load_absorption_index(uint context, MediumBlobAccess medium_access) {
  return medium_access.absorption_spectrum_index;
}

uint medium_extinction_shared_gpu_load_scattering_index(uint context, MediumBlobAccess medium_access) {
  return medium_access.scattering_spectrum_index;
}

bool medium_extinction_shared_gpu_can_sample_spectrum(uint context, uint spectrum_index) {
  return scene_resource_shared_can_sample_spectrum(constants.scene.spectrums, spectrum_index);
}

float3 medium_extinction_shared_gpu_load_spectrum_integrated(uint context, uint spectrum_index) {
  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  return load_spectrum_integrated_value(spectrum_buffer, spectrum_index);
}

SpectralResponse medium_extinction_shared_gpu_load_spectrum_spectral(uint context, uint spectrum_index, SpectralQuery spect) {
  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  return load_spectrum_response(spectrum_buffer, spectrum_index, spect);
}

#define ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE uint
#define ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE MediumBlobAccess
#define ETX_MEDIUM_EXTINCTION_SHARED_LOAD_ABSORPTION_INDEX(context, medium_access) medium_extinction_shared_gpu_load_absorption_index(context, medium_access)
#define ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SCATTERING_INDEX(context, medium_access) medium_extinction_shared_gpu_load_scattering_index(context, medium_access)
#define ETX_MEDIUM_EXTINCTION_SHARED_CAN_SAMPLE_SPECTRUM(context, spectrum_index) medium_extinction_shared_gpu_can_sample_spectrum(context, spectrum_index)
#define ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_INTEGRATED(context, spectrum_index) medium_extinction_shared_gpu_load_spectrum_integrated(context, spectrum_index)
#define ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_SPECTRAL(context, spectrum_index, spect) \
  medium_extinction_shared_gpu_load_spectrum_spectral(context, spectrum_index, spect)
#define ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_RESPONSE_TYPE SpectralResponse
#define ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_QUERY_TYPE SpectralQuery
#define ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ZERO(spect) spectral_response_zero(spect)
#define ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ADD(a, b) spectral_response_add(a, b)
#include <interop/medium_extinction_shared.hxx>
#undef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ADD
#undef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ZERO
#undef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_QUERY_TYPE
#undef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_RESPONSE_TYPE
#undef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_SPECTRAL
#undef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_INTEGRATED
#undef ETX_MEDIUM_EXTINCTION_SHARED_CAN_SAMPLE_SPECTRUM
#undef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SCATTERING_INDEX
#undef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_ABSORPTION_INDEX
#undef ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE
#undef ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE

#define ETX_MEDIUM_ACCESS_SHARED_CONTEXT_TYPE ByteAddressBuffer
#define ETX_MEDIUM_ACCESS_SHARED_ACCESS_TYPE MediumBlobAccess
#define ETX_MEDIUM_ACCESS_SHARED_LOAD_U32(context, byte_offset) context.Load(byte_offset)
#define ETX_MEDIUM_ACCESS_SHARED_LOAD_U16(context, byte_offset) load_u16(context, byte_offset)
#define ETX_MEDIUM_ACCESS_SHARED_LOAD_U32X3(context, byte_offset) context.Load3(byte_offset)
#define ETX_MEDIUM_ACCESS_SHARED_LOAD_F32(context, byte_offset) asfloat(context.Load(byte_offset))
#define ETX_MEDIUM_ACCESS_SHARED_LOAD_F32X3(context, byte_offset) asfloat(context.Load3(byte_offset))
#define ETX_MEDIUM_ACCESS_SHARED_CHUNK_DESCRIPTOR_INDEX(context, chunk_index) medium_blob_access_shared_chunk_descriptor_index(context, chunk_index)
#include <interop/medium_access_shared.hxx>
#undef ETX_MEDIUM_ACCESS_SHARED_CHUNK_DESCRIPTOR_INDEX
#undef ETX_MEDIUM_ACCESS_SHARED_LOAD_F32X3
#undef ETX_MEDIUM_ACCESS_SHARED_LOAD_F32
#undef ETX_MEDIUM_ACCESS_SHARED_LOAD_U32X3
#undef ETX_MEDIUM_ACCESS_SHARED_LOAD_U16
#undef ETX_MEDIUM_ACCESS_SHARED_LOAD_U32
#undef ETX_MEDIUM_ACCESS_SHARED_ACCESS_TYPE
#undef ETX_MEDIUM_ACCESS_SHARED_CONTEXT_TYPE

MediumBlobAccess load_medium_blob_access(ByteAddressBuffer medium_blob, uint medium_desc_offset) {
  ETX_ZERO_INIT(MediumBlobAccess, result);
  medium_access_shared_load(medium_blob, medium_desc_offset, result);
  return result;
}

bool try_get_medium_desc_offset(uint medium_index, out uint medium_desc_offset) {
  medium_desc_offset = kInvalidIndex;
  if ((medium_index == kInvalidIndex) || (constants.scene.mediums == kInvalidIndex)) {
    return false;
  }

  ByteAddressBuffer medium_blob = bindless_buffers[NonUniformResourceIndex(constants.scene.mediums)];
  return medium_blob_access_shared_try_get_desc_offset(medium_blob, medium_index, medium_desc_offset);
}

struct MediumTextureSampleGPUSharedContext {
  ByteAddressBuffer payload_buffer;
  uint density_offset;
  uint density_count;
};

float medium_texture_sample_gpu_shared_density(MediumTextureSampleGPUSharedContext context, uint3 dimensions, uint x, uint y, uint z) {
  uint index = x + y * dimensions.x + z * dimensions.x * dimensions.y;
  if (index >= context.density_count) {
    return 0.0f;
  }
  return asfloat(context.payload_buffer.Load(context.density_offset + index * 4u));
}

#define ETX_MEDIUM_TEXTURE_SAMPLE_SHARED_CONTEXT_TYPE MediumTextureSampleGPUSharedContext
#define ETX_MEDIUM_TEXTURE_SAMPLE_SHARED_DENSITY(context, dimensions, x, y, z) medium_texture_sample_gpu_shared_density(context, dimensions, x, y, z)
#include <interop/medium_texture_sample_shared.hxx>
#undef ETX_MEDIUM_TEXTURE_SAMPLE_SHARED_DENSITY
#undef ETX_MEDIUM_TEXTURE_SAMPLE_SHARED_CONTEXT_TYPE

float medium_sample_texture_3d(MediumBlobAccess medium_access, float3 local_coord) {
  if ((medium_access.grid.density_count == 0u) || (medium_access.grid.density_data_offset == kInvalidIndex) || (medium_access.density_payload_descriptor_index == kInvalidIndex)) {
    return 0.0f;
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(medium_access.density_payload_descriptor_index)];
  MediumTextureSampleGPUSharedContext context = {payload_buffer, medium_access.grid.density_data_offset, medium_access.grid.density_count};
  return medium_texture_sample_shared_3d(context, local_coord, medium_access.grid.dimensions);
}

float medium_sample_noise(MediumBlobAccess medium_access, float3 local_coord) {
  return medium_density_shared_sample_noise(local_coord, medium_access.bounds_min, medium_access.bounds_max, medium_access.grid.noise_type, medium_access.grid.noise_scale,
    medium_access.grid.noise_octaves, medium_access.grid.noise_lacunarity, medium_access.grid.noise_persistence, medium_access.grid.noise_seed,
    medium_access.grid.noise_offset, medium_access.grid.noise_enable_border_fade, medium_access.grid.noise_border_fade_distance);
}

struct MediumGridPolicyGPUSharedContext {
  MediumBlobAccess medium_access;
};

MediumDensitySharedGrid medium_grid_policy_gpu_shared_grid(MediumGridPolicyGPUSharedContext context) {
  return context.medium_access.grid;
}

float medium_grid_policy_gpu_shared_sample_noise(MediumGridPolicyGPUSharedContext context, float3 local_coord) {
  return medium_sample_noise(context.medium_access, local_coord);
}

float medium_grid_policy_gpu_shared_sample_texture(MediumGridPolicyGPUSharedContext context, float3 local_coord) {
  return medium_sample_texture_3d(context.medium_access, local_coord);
}

bool medium_grid_policy_gpu_shared_texture_ready(MediumGridPolicyGPUSharedContext context) {
  return (context.medium_access.grid.density_data_offset != kInvalidIndex) && (context.medium_access.density_payload_descriptor_index != kInvalidIndex);
}

#define ETX_MEDIUM_GRID_POLICY_SHARED_CONTEXT_TYPE MediumGridPolicyGPUSharedContext
#define ETX_MEDIUM_GRID_POLICY_SHARED_GRID(context) medium_grid_policy_gpu_shared_grid(context)
#define ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_NOISE(context, local_coord) medium_grid_policy_gpu_shared_sample_noise(context, local_coord)
#define ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_TEXTURE(context, local_coord) medium_grid_policy_gpu_shared_sample_texture(context, local_coord)
#define ETX_MEDIUM_GRID_POLICY_SHARED_TEXTURE_READY(context) medium_grid_policy_gpu_shared_texture_ready(context)
#include <interop/medium_grid_policy_shared.hxx>
#undef ETX_MEDIUM_GRID_POLICY_SHARED_TEXTURE_READY
#undef ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_TEXTURE
#undef ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_NOISE
#undef ETX_MEDIUM_GRID_POLICY_SHARED_GRID
#undef ETX_MEDIUM_GRID_POLICY_SHARED_CONTEXT_TYPE

float medium_sample_density(MediumBlobAccess medium_access, float3 local_coord) {
  MediumGridPolicyGPUSharedContext context = {medium_access};
  return medium_grid_policy_shared_sample_density(context, local_coord);
}

bool medium_has_grid_data(MediumBlobAccess medium_access) {
  MediumGridPolicyGPUSharedContext context = {medium_access};
  return medium_grid_policy_shared_has_grid_data(context);
}

struct MediumDensityRandomGPUSharedContext {
  MediumBlobAccess medium_access;
  uint seed;
};

float medium_density_random_gpu_shared_rnd(inout MediumDensityRandomGPUSharedContext context) {
  return rnd01(context.seed);
}

float medium_density_random_gpu_shared_density(inout MediumDensityRandomGPUSharedContext context, float3 local_pos) {
  return medium_sample_density(context.medium_access, local_pos);
}

uint medium_sample_shared_gpu_load_medium_class(MediumDensityRandomGPUSharedContext context) {
  return context.medium_access.medium_class;
}

bool medium_sample_shared_gpu_has_grid_data(MediumDensityRandomGPUSharedContext context) {
  return medium_has_grid_data(context.medium_access);
}

float3 medium_sample_shared_gpu_bounds_min(MediumDensityRandomGPUSharedContext context) {
  return context.medium_access.bounds_min;
}

float3 medium_sample_shared_gpu_bounds_max(MediumDensityRandomGPUSharedContext context) {
  return context.medium_access.bounds_max;
}

#define ETX_MEDIUM_SHARED_CONTEXT_TYPE MediumDensityRandomGPUSharedContext
#define ETX_MEDIUM_SHARED_RND(context) medium_density_random_gpu_shared_rnd(context)
#define ETX_MEDIUM_SHARED_DENSITY(context, local_pos) medium_density_random_gpu_shared_density(context, local_pos)
#include <interop/medium_transmittance_shared.hxx>
#undef ETX_MEDIUM_SHARED_DENSITY
#undef ETX_MEDIUM_SHARED_RND
#undef ETX_MEDIUM_SHARED_CONTEXT_TYPE

#define ETX_MEDIUM_SAMPLE_SHARED_CONTEXT_TYPE MediumDensityRandomGPUSharedContext
#define ETX_MEDIUM_SAMPLE_SHARED_RND(context) medium_density_random_gpu_shared_rnd(context)
#define ETX_MEDIUM_SAMPLE_SHARED_DENSITY(context, local_pos) medium_density_random_gpu_shared_density(context, local_pos)
#define ETX_MEDIUM_SAMPLE_SHARED_LOAD_MEDIUM_CLASS(context) medium_sample_shared_gpu_load_medium_class(context)
#define ETX_MEDIUM_SAMPLE_SHARED_HAS_GRID_DATA(context) medium_sample_shared_gpu_has_grid_data(context)
#define ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MIN(context) medium_sample_shared_gpu_bounds_min(context)
#define ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MAX(context) medium_sample_shared_gpu_bounds_max(context)
#include <interop/medium_sample_shared.hxx>
#undef ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MAX
#undef ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MIN
#undef ETX_MEDIUM_SAMPLE_SHARED_HAS_GRID_DATA
#undef ETX_MEDIUM_SAMPLE_SHARED_LOAD_MEDIUM_CLASS
#undef ETX_MEDIUM_SAMPLE_SHARED_DENSITY
#undef ETX_MEDIUM_SAMPLE_SHARED_RND
#undef ETX_MEDIUM_SAMPLE_SHARED_CONTEXT_TYPE

MediumSample sample_medium_gpu(
  MediumBlobAccess medium_access, SpectralQuery spect, SpectralResponse throughput, SpectralResponse scattering_value, SpectralResponse absorption_value, float3 pos, float3 w_i,
  float max_t, inout uint seed) {
  MediumDensityRandomGPUSharedContext context;
  context.medium_access = medium_access;
  context.seed = seed;
  MediumSample result = medium_sample_shared_sample(context, spect, throughput, scattering_value, absorption_value, pos, w_i, max_t);
  seed = context.seed;
  return result;
}

struct MediumSegmentTransmittanceGPUSharedContext {
  uint seed;
};

bool medium_segment_transmittance_shared_gpu_has_required_scene_buffers(MediumSegmentTransmittanceGPUSharedContext context) {
  return scene_resource_shared_has_medium_spectrum_buffers(constants.scene.mediums, constants.scene.spectrums);
}

bool medium_segment_transmittance_shared_gpu_try_load_access(MediumSegmentTransmittanceGPUSharedContext context, uint medium_index, out MediumBlobAccess medium_access) {
  medium_access = ETX_ZERO(MediumBlobAccess);
  uint medium_desc_offset = kInvalidIndex;
  if (try_get_medium_desc_offset(medium_index, medium_desc_offset) == false) {
    return false;
  }

  ByteAddressBuffer medium_blob = bindless_buffers[NonUniformResourceIndex(constants.scene.mediums)];
  medium_access = load_medium_blob_access(medium_blob, medium_desc_offset);
  return true;
}

uint medium_segment_transmittance_shared_gpu_load_medium_class(MediumSegmentTransmittanceGPUSharedContext context, MediumBlobAccess medium_access) {
  return medium_access.medium_class;
}

bool medium_segment_transmittance_shared_gpu_has_grid_data(MediumSegmentTransmittanceGPUSharedContext context, MediumBlobAccess medium_access) {
  return medium_has_grid_data(medium_access);
}

float3 medium_segment_transmittance_shared_gpu_load_extinction_integrated(MediumSegmentTransmittanceGPUSharedContext context, MediumBlobAccess medium_access) {
  return medium_extinction_shared_load_integrated(0u, medium_access);
}

SpectralResponse medium_segment_transmittance_shared_gpu_load_extinction_spectral(
  MediumSegmentTransmittanceGPUSharedContext context, MediumBlobAccess medium_access, SpectralQuery spect) {
  return medium_extinction_shared_load_spectral(0u, medium_access, spect);
}

float3 medium_segment_transmittance_shared_gpu_transmittance_homogeneous_integrated(
  MediumSegmentTransmittanceGPUSharedContext context, MediumBlobAccess medium_access, float3 extinction, float distance) {
  return medium_shared_transmittance_homogeneous_integrated(extinction, distance);
}

SpectralResponse medium_segment_transmittance_shared_gpu_transmittance_homogeneous_spectral(
  MediumSegmentTransmittanceGPUSharedContext context, MediumBlobAccess medium_access, SpectralResponse extinction, float distance, SpectralQuery spect) {
  return medium_shared_transmittance_homogeneous_spectral(extinction, distance);
}

float3 medium_segment_transmittance_shared_gpu_transmittance_heterogeneous_integrated(
  inout MediumSegmentTransmittanceGPUSharedContext context, MediumBlobAccess medium_access, float3 extinction, float3 origin, float3 direction, float distance) {
  MediumDensityRandomGPUSharedContext medium_context;
  medium_context.medium_access = medium_access;
  medium_context.seed = context.seed;
  float3 transmittance =
    medium_shared_transmittance_heterogeneous_integrated(extinction, origin, direction, distance, medium_access.bounds_min, medium_access.bounds_max, medium_context);
  context.seed = medium_context.seed;
  return transmittance;
}

SpectralResponse medium_segment_transmittance_shared_gpu_transmittance_heterogeneous_spectral(
  inout MediumSegmentTransmittanceGPUSharedContext context, MediumBlobAccess medium_access, SpectralResponse extinction, float3 origin, float3 direction, float distance,
  SpectralQuery spect) {
  MediumDensityRandomGPUSharedContext medium_context;
  medium_context.medium_access = medium_access;
  medium_context.seed = context.seed;
  SpectralResponse transmittance =
    medium_shared_transmittance_heterogeneous_spectral(extinction, origin, direction, distance, medium_access.bounds_min, medium_access.bounds_max, medium_context, spect);
  context.seed = medium_context.seed;
  return transmittance;
}

SpectralResponse medium_segment_transmittance_shared_gpu_spectral_one(SpectralQuery spect) {
  return spectral_response_make(spect, 1.0f);
}

#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_CONTEXT_TYPE MediumSegmentTransmittanceGPUSharedContext
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_ACCESS_TYPE MediumBlobAccess
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_RESPONSE_TYPE SpectralResponse
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_QUERY_TYPE SpectralQuery
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) medium_segment_transmittance_shared_gpu_has_required_scene_buffers(context)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRY_LOAD_ACCESS(context, medium_index, medium_access) \
  medium_segment_transmittance_shared_gpu_try_load_access(context, medium_index, medium_access)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_MEDIUM_CLASS(context, medium_access) \
  medium_segment_transmittance_shared_gpu_load_medium_class(context, medium_access)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_GRID_DATA(context, medium_access) medium_segment_transmittance_shared_gpu_has_grid_data(context, medium_access)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_INTEGRATED(context, medium_access) \
  medium_segment_transmittance_shared_gpu_load_extinction_integrated(context, medium_access)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_SPECTRAL(context, medium_access, spect) \
  medium_segment_transmittance_shared_gpu_load_extinction_spectral(context, medium_access, spect)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_INTEGRATED(context, medium_access, extinction, distance) \
  medium_segment_transmittance_shared_gpu_transmittance_homogeneous_integrated(context, medium_access, extinction, distance)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_SPECTRAL(context, medium_access, extinction, distance, spect) \
  medium_segment_transmittance_shared_gpu_transmittance_homogeneous_spectral(context, medium_access, extinction, distance, spect)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_INTEGRATED(context, medium_access, extinction, origin, direction, distance) \
  medium_segment_transmittance_shared_gpu_transmittance_heterogeneous_integrated(context, medium_access, extinction, origin, direction, distance)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_SPECTRAL(context, medium_access, extinction, origin, direction, distance, spect) \
  medium_segment_transmittance_shared_gpu_transmittance_heterogeneous_spectral(context, medium_access, extinction, origin, direction, distance, spect)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_ONE(spect) medium_segment_transmittance_shared_gpu_spectral_one(spect)
#include <interop/medium_segment_transmittance_shared.hxx>
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_ONE
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_SPECTRAL
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_INTEGRATED
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_SPECTRAL
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_INTEGRATED
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_SPECTRAL
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_INTEGRATED
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_GRID_DATA
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_MEDIUM_CLASS
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRY_LOAD_ACCESS
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_REQUIRED_SCENE_BUFFERS
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_QUERY_TYPE
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_RESPONSE_TYPE
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_ACCESS_TYPE
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_CONTEXT_TYPE

float3 medium_segment_transmittance_integrated(uint medium_index, float3 origin, float3 direction, float distance, inout uint seed) {
  MediumSegmentTransmittanceGPUSharedContext context;
  context.seed = seed;
  float3 transmittance = medium_segment_transmittance_shared_integrated(context, medium_index, origin, direction, distance);
  seed = context.seed;
  return transmittance;
}

SpectralResponse medium_segment_transmittance_spectral(uint medium_index, float3 origin, float3 direction, float distance, SpectralQuery spect, inout uint seed) {
  MediumSegmentTransmittanceGPUSharedContext context;
  context.seed = seed;
  SpectralResponse transmittance = medium_segment_transmittance_shared_spectral(context, medium_index, origin, direction, distance, spect);
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

struct DistributionSharedGPUContext {
  ByteAddressBuffer payload_buffer;
  uint entries_base_offset;
};

DistributionSharedGPUContext make_distribution_shared_gpu_context(ByteAddressBuffer payload_buffer, uint entries_base_offset) {
  DistributionSharedGPUContext context;
  context.payload_buffer = payload_buffer;
  context.entries_base_offset = entries_base_offset;
  return context;
}

float distribution_shared_gpu_cdf(DistributionSharedGPUContext context, uint index) {
  DistributionEntry entry = load_distribution_entry(context.payload_buffer, context.entries_base_offset + index * kDistributionEntryStride);
  return entry.cdf;
}

float distribution_shared_gpu_pdf(DistributionSharedGPUContext context, uint index) {
  DistributionEntry entry = load_distribution_entry(context.payload_buffer, context.entries_base_offset + index * kDistributionEntryStride);
  return entry.pdf;
}

#define ETX_DISTRIBUTION_SHARED_CONTEXT_TYPE DistributionSharedGPUContext
#define ETX_DISTRIBUTION_SHARED_CDF(context, index) distribution_shared_gpu_cdf(context, index)
#define ETX_DISTRIBUTION_SHARED_PDF(context, index) distribution_shared_gpu_pdf(context, index)
#include <interop/distribution_sample_shared.hxx>
#undef ETX_DISTRIBUTION_SHARED_PDF
#undef ETX_DISTRIBUTION_SHARED_CDF
#undef ETX_DISTRIBUTION_SHARED_CONTEXT_TYPE

uint sample_distribution(ByteAddressBuffer payload_buffer, uint entries_base_offset, uint count, float rnd, out float pdf) {
  DistributionSharedGPUContext context = make_distribution_shared_gpu_context(payload_buffer, entries_base_offset);
  return distribution_shared_sample(context, count, rnd, pdf);
}

struct ImageSampleSharedGPUContext {
  ByteAddressBuffer x_payload_buffer;
  ByteAddressBuffer y_payload_buffer;
  uint x_distribution_entries_offset;
  uint y_distribution_entries_offset;
  uint x_entries_stride;
  uint x_distribution_count;
  uint y_count;
  float2 fsize;
};

ImageSampleSharedGPUContext make_image_sample_shared_gpu_context(
  ByteAddressBuffer x_payload_buffer, ByteAddressBuffer y_payload_buffer, ImageDescAccess image_access, uint y_count) {
  ImageSampleSharedGPUContext context;
  context.x_payload_buffer = x_payload_buffer;
  context.y_payload_buffer = y_payload_buffer;
  context.x_distribution_entries_offset = image_access.x_distribution_entries_offset;
  context.y_distribution_entries_offset = image_access.y_distribution_entries_offset;
  context.x_entries_stride = image_access.x_entries_stride;
  context.x_distribution_count = image_access.x_distribution_count;
  context.y_count = y_count;
  context.fsize = image_access.fsize;
  return context;
}

uint image_sample_shared_gpu_y_count(ImageSampleSharedGPUContext context) {
  return context.y_count;
}

uint image_sample_shared_gpu_x_count(ImageSampleSharedGPUContext context, uint y_index) {
  if ((y_index >= context.x_distribution_count) || (context.x_entries_stride == 0u)) {
    return 0u;
  }

  return context.x_entries_stride - 1u;
}

float2 image_sample_shared_gpu_image_fsize(ImageSampleSharedGPUContext context) {
  return context.fsize;
}

uint image_sample_shared_gpu_sample_y(inout ImageSampleSharedGPUContext context, float rnd, out float pdf) {
  return sample_distribution(context.y_payload_buffer, context.y_distribution_entries_offset, context.y_count, rnd, pdf);
}

uint image_sample_shared_gpu_sample_x(inout ImageSampleSharedGPUContext context, uint y_index, float rnd, out float pdf) {
  uint row_base_offset = context.x_distribution_entries_offset + y_index * context.x_entries_stride * kDistributionEntryStride;
  uint x_count = image_sample_shared_gpu_x_count(context, y_index);
  return sample_distribution(context.x_payload_buffer, row_base_offset, x_count, rnd, pdf);
}

float image_sample_shared_gpu_cdf_y(ImageSampleSharedGPUContext context, uint y_index) {
  uint y_offset = context.y_distribution_entries_offset + y_index * kDistributionEntryStride;
  DistributionEntry entry = load_distribution_entry(context.y_payload_buffer, y_offset);
  return entry.cdf;
}

float image_sample_shared_gpu_cdf_x(ImageSampleSharedGPUContext context, uint y_index, uint x_index) {
  uint row_base_offset = context.x_distribution_entries_offset + y_index * context.x_entries_stride * kDistributionEntryStride;
  uint x_offset = row_base_offset + x_index * kDistributionEntryStride;
  DistributionEntry entry = load_distribution_entry(context.x_payload_buffer, x_offset);
  return entry.cdf;
}

#define ETX_IMAGE_SAMPLE_SHARED_CONTEXT_TYPE ImageSampleSharedGPUContext
#define ETX_IMAGE_SAMPLE_SHARED_Y_COUNT(context) image_sample_shared_gpu_y_count(context)
#define ETX_IMAGE_SAMPLE_SHARED_X_COUNT(context, y_index) image_sample_shared_gpu_x_count(context, y_index)
#define ETX_IMAGE_SAMPLE_SHARED_IMAGE_FSIZE(context) image_sample_shared_gpu_image_fsize(context)
#define ETX_IMAGE_SAMPLE_SHARED_SAMPLE_Y(context, rnd, pdf) image_sample_shared_gpu_sample_y(context, rnd, pdf)
#define ETX_IMAGE_SAMPLE_SHARED_SAMPLE_X(context, y_index, rnd, pdf) image_sample_shared_gpu_sample_x(context, y_index, rnd, pdf)
#define ETX_IMAGE_SAMPLE_SHARED_CDF_Y(context, y_index) image_sample_shared_gpu_cdf_y(context, y_index)
#define ETX_IMAGE_SAMPLE_SHARED_CDF_X(context, y_index, x_index) image_sample_shared_gpu_cdf_x(context, y_index, x_index)
#include <interop/image_sample_shared.hxx>
#undef ETX_IMAGE_SAMPLE_SHARED_CDF_X
#undef ETX_IMAGE_SAMPLE_SHARED_CDF_Y
#undef ETX_IMAGE_SAMPLE_SHARED_SAMPLE_X
#undef ETX_IMAGE_SAMPLE_SHARED_SAMPLE_Y
#undef ETX_IMAGE_SAMPLE_SHARED_IMAGE_FSIZE
#undef ETX_IMAGE_SAMPLE_SHARED_X_COUNT
#undef ETX_IMAGE_SAMPLE_SHARED_Y_COUNT
#undef ETX_IMAGE_SAMPLE_SHARED_CONTEXT_TYPE

float2 sample_image_uv_fallback(uint image_index, float2 fallback_uv, out float4 eval) {
  eval = evaluate_image(image_index, fallback_uv);
  return fallback_uv;
}

float2 sample_image_uv(uint image_index, float2 rnd, out float image_pdf, out uint2 location, out float4 eval) {
  image_pdf = 0.0f;
  location = uint2(0u, 0u);
  eval = float4(1.0f, 1.0f, 1.0f, 1.0f);

  ImageSceneAccessGPUContext access_context = {constants.scene.images};
  ETX_ZERO_INIT(ImageDescAccess, image_access);
  uint x_payload_descriptor_index = kInvalidIndex;
  uint y_payload_descriptor_index = kInvalidIndex;
  uint y_count = 0u;
  if (image_scene_access_shared_try_load_distribution_payload_descriptors(
        access_context, image_index, image_access, x_payload_descriptor_index, y_payload_descriptor_index, y_count) == false) {
    return sample_image_uv_fallback(image_index, rnd, eval);
  }

  ByteAddressBuffer x_payload_buffer = bindless_buffers[NonUniformResourceIndex(x_payload_descriptor_index)];
  ByteAddressBuffer y_payload_buffer = bindless_buffers[NonUniformResourceIndex(y_payload_descriptor_index)];

  ImageSampleSharedGPUContext sample_context = make_image_sample_shared_gpu_context(x_payload_buffer, y_payload_buffer, image_access, y_count);

  float2 uv = rnd;
  bool sampled = image_sample_shared_distribution(sample_context, rnd, image_pdf, location, uv);
  if (sampled == false) {
    return sample_image_uv_fallback(image_index, rnd, eval);
  }

  eval = evaluate_image(image_index, uv);
  return uv;
}

struct CameraLensSampleSharedGPUContext {
  uint images_descriptor_index;
};

bool camera_lens_sample_shared_gpu_try_sample_image_uv(
  CameraLensSampleSharedGPUContext context, uint lens_image, float2 rnd, out float2 image_uv) {
  image_uv = float2(0.0f, 0.0f);

  ImageSceneAccessGPUContext access_context = {context.images_descriptor_index};
  ETX_ZERO_INIT(ImageDescAccess, image_access);
  if (image_scene_access_shared_try_load_desc(access_context, lens_image, image_access) == false) {
    return false;
  }

  float image_pdf = 0.0f;
  uint2 image_location = uint2(0u, 0u);
  float4 image_eval = float4(1.0f, 1.0f, 1.0f, 1.0f);
  image_uv = sample_image_uv(lens_image, rnd, image_pdf, image_location, image_eval);
  return true;
}

#define ETX_CAMERA_LENS_SAMPLE_SHARED_CONTEXT_TYPE CameraLensSampleSharedGPUContext
#define ETX_CAMERA_LENS_SAMPLE_SHARED_TRY_SAMPLE_IMAGE_UV(context, lens_image, rnd, image_uv) \
  camera_lens_sample_shared_gpu_try_sample_image_uv(context, lens_image, rnd, image_uv)
#include <interop/camera_lens_sample_shared.hxx>
#undef ETX_CAMERA_LENS_SAMPLE_SHARED_TRY_SAMPLE_IMAGE_UV
#undef ETX_CAMERA_LENS_SAMPLE_SHARED_CONTEXT_TYPE

#define ETX_CAMERA_PRIMARY_RAY_SHARED_CONTEXT_TYPE CameraLensSampleSharedGPUContext
#include <interop/camera_primary_ray_shared.hxx>
#undef ETX_CAMERA_PRIMARY_RAY_SHARED_CONTEXT_TYPE

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
  ImageSceneAccessGPUContext access_context = {constants.scene.images};
  ETX_ZERO_INIT(ImageDescAccess, image_access);
  uint payload_descriptor_index = kInvalidIndex;
  if (image_scene_access_shared_try_load_pixel_payload_descriptor(access_context, image_index, image_access, payload_descriptor_index) == false) {
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(payload_descriptor_index)];
  ImageFilterSharedAddress sample = image_filter_shared_address(uv, image_access.fsize, image_access.size, image_access.options);

  uint pixel_offset_00 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_01 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);
  uint pixel_offset_10 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_11 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);

  float4 p00 = load_image_pixel(payload_buffer, image_access.format, pixel_offset_00);
  float4 p01 = load_image_pixel(payload_buffer, image_access.format, pixel_offset_01);
  float4 p10 = load_image_pixel(payload_buffer, image_access.format, pixel_offset_10);
  float4 p11 = load_image_pixel(payload_buffer, image_access.format, pixel_offset_11);

  return image_filter_shared_bilinear(p00, p01, p10, p11, sample.dx, sample.dy);
}

bool image_evaluate_shared_gpu_try_evaluate_image_rgba(uint context, uint image_index, float2 uv, out float image_pdf, out float4 image_value) {
  image_pdf = 0.0f;
  image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);

  ImageSceneAccessGPUContext access_context = {constants.scene.images};
  ETX_ZERO_INIT(ImageDescAccess, image_access);
  if (image_scene_access_shared_try_load_desc(access_context, image_index, image_access) == false) {
    return false;
  }

  image_value = evaluate_image(image_index, uv);
  return true;
}

#define ETX_IMAGE_EVALUATE_SHARED_CONTEXT_TYPE uint
#define ETX_IMAGE_EVALUATE_SHARED_TRY_EVALUATE_IMAGE_RGBA(context, image_index, uv, image_pdf, image_value) \
  image_evaluate_shared_gpu_try_evaluate_image_rgba(context, image_index, uv, image_pdf, image_value)
#include <interop/image_evaluate_shared.hxx>
#undef ETX_IMAGE_EVALUATE_SHARED_TRY_EVALUATE_IMAGE_RGBA
#undef ETX_IMAGE_EVALUATE_SHARED_CONTEXT_TYPE

bool image_has_alpha_channel(uint image_index) {
  ImageSceneAccessGPUContext access_context = {constants.scene.images};
  return image_scene_access_shared_has_alpha(access_context, image_index);
}

struct AlphaTestSharedGPUContext {
  uint material_index;
  float2 uv;
  uint seed;
};

uint alpha_test_shared_gpu_material_class(AlphaTestSharedGPUContext context) {
  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.materials)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(material_buffer);
  return gpu_abi_access_shared_material_class(access_context, context.material_index);
}

float alpha_test_shared_gpu_material_opacity(AlphaTestSharedGPUContext context) {
  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.materials)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(material_buffer);
  return gpu_abi_access_shared_material_opacity(access_context, context.material_index);
}

uint alpha_test_shared_gpu_scattering_image_index(AlphaTestSharedGPUContext context) {
  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.materials)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(material_buffer);
  return gpu_abi_access_shared_material_scattering_image_index(access_context, context.material_index);
}

bool alpha_test_shared_gpu_image_has_alpha(AlphaTestSharedGPUContext context, uint image_index) {
  return image_has_alpha_channel(image_index);
}

float alpha_test_shared_gpu_evaluate_alpha(AlphaTestSharedGPUContext context, uint image_index) {
  return image_evaluate_shared_sample_channel_or_default(0u, image_index, 3u, context.uv, 1.0f);
}

float alpha_test_shared_gpu_rnd(inout AlphaTestSharedGPUContext context) {
  return rnd01(context.seed);
}

#define ETX_ALPHA_TEST_SHARED_CONTEXT_TYPE AlphaTestSharedGPUContext
#define ETX_ALPHA_TEST_SHARED_MATERIAL_CLASS(context) alpha_test_shared_gpu_material_class(context)
#define ETX_ALPHA_TEST_SHARED_MATERIAL_OPACITY(context) alpha_test_shared_gpu_material_opacity(context)
#define ETX_ALPHA_TEST_SHARED_SCATTERING_IMAGE_INDEX(context) alpha_test_shared_gpu_scattering_image_index(context)
#define ETX_ALPHA_TEST_SHARED_IMAGE_HAS_ALPHA(context, image_index) alpha_test_shared_gpu_image_has_alpha(context, image_index)
#define ETX_ALPHA_TEST_SHARED_EVALUATE_ALPHA(context, image_index) alpha_test_shared_gpu_evaluate_alpha(context, image_index)
#define ETX_ALPHA_TEST_SHARED_RND(context) alpha_test_shared_gpu_rnd(context)
#include <interop/alpha_test_shared.hxx>
#undef ETX_ALPHA_TEST_SHARED_RND
#undef ETX_ALPHA_TEST_SHARED_EVALUATE_ALPHA
#undef ETX_ALPHA_TEST_SHARED_IMAGE_HAS_ALPHA
#undef ETX_ALPHA_TEST_SHARED_SCATTERING_IMAGE_INDEX
#undef ETX_ALPHA_TEST_SHARED_MATERIAL_OPACITY
#undef ETX_ALPHA_TEST_SHARED_MATERIAL_CLASS
#undef ETX_ALPHA_TEST_SHARED_CONTEXT_TYPE

bool alpha_test_access_shared_gpu_has_required_scene_buffers(uint context) {
  return scene_resource_shared_is_available(constants.scene.materials);
}

AlphaTestSharedGPUContext alpha_test_access_shared_gpu_make_alpha_context(uint context, uint material_index, float2 uv, uint seed) {
  ETX_ZERO_INIT(AlphaTestSharedGPUContext, alpha_context);
  alpha_context.material_index = material_index;
  alpha_context.uv = uv;
  alpha_context.seed = seed;
  return alpha_context;
}

#define ETX_ALPHA_TEST_ACCESS_SHARED_CONTEXT_TYPE uint
#define ETX_ALPHA_TEST_ACCESS_SHARED_ALPHA_CONTEXT_TYPE AlphaTestSharedGPUContext
#define ETX_ALPHA_TEST_ACCESS_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) alpha_test_access_shared_gpu_has_required_scene_buffers(context)
#define ETX_ALPHA_TEST_ACCESS_SHARED_MAKE_ALPHA_CONTEXT(context, material_index, uv, seed) alpha_test_access_shared_gpu_make_alpha_context(context, material_index, uv, seed)
#define ETX_ALPHA_TEST_ACCESS_SHARED_PASS(alpha_context) alpha_test_shared_pass(alpha_context)
#define ETX_ALPHA_TEST_ACCESS_SHARED_ALPHA_CONTEXT_SEED(alpha_context) alpha_context.seed
#include <interop/alpha_test_access_shared.hxx>
#undef ETX_ALPHA_TEST_ACCESS_SHARED_ALPHA_CONTEXT_SEED
#undef ETX_ALPHA_TEST_ACCESS_SHARED_PASS
#undef ETX_ALPHA_TEST_ACCESS_SHARED_MAKE_ALPHA_CONTEXT
#undef ETX_ALPHA_TEST_ACCESS_SHARED_HAS_REQUIRED_SCENE_BUFFERS
#undef ETX_ALPHA_TEST_ACCESS_SHARED_ALPHA_CONTEXT_TYPE
#undef ETX_ALPHA_TEST_ACCESS_SHARED_CONTEXT_TYPE

struct SurfacePoint {
  float3 barycentrics;
  Vertex vertex;
  float3 geo_normal;
};

SurfacePoint load_surface_point(ByteAddressBuffer position_buffer, ByteAddressBuffer normal_buffer, ByteAddressBuffer tangent_buffer, ByteAddressBuffer bitangent_buffer,
  ByteAddressBuffer texcoord_buffer, bool has_surface_frame, bool has_texcoords, TriangleData tri, float2 bary, float3 ray_dir) {
  SurfacePoint result;
  result.barycentrics = surface_point_shared_barycentrics(bary);

  float3 p0 = load_float3(position_buffer, tri.i.x);
  float3 p1 = load_float3(position_buffer, tri.i.y);
  float3 p2 = load_float3(position_buffer, tri.i.z);

  float3 n0 = load_float3(normal_buffer, tri.i.x);
  float3 n1 = load_float3(normal_buffer, tri.i.y);
  float3 n2 = load_float3(normal_buffer, tri.i.z);

  float3 tangent_0 = float3(0.0f, 0.0f, 0.0f);
  float3 tangent_1 = float3(0.0f, 0.0f, 0.0f);
  float3 tangent_2 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_0 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_1 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_2 = float3(0.0f, 0.0f, 0.0f);
  if (has_surface_frame) {
    tangent_0 = load_float3(tangent_buffer, tri.i.x);
    tangent_1 = load_float3(tangent_buffer, tri.i.y);
    tangent_2 = load_float3(tangent_buffer, tri.i.z);
    bitangent_0 = load_float3(bitangent_buffer, tri.i.x);
    bitangent_1 = load_float3(bitangent_buffer, tri.i.y);
    bitangent_2 = load_float3(bitangent_buffer, tri.i.z);
  }

  float2 texcoord_0 = float2(0.0f, 0.0f);
  float2 texcoord_1 = float2(0.0f, 0.0f);
  float2 texcoord_2 = float2(0.0f, 0.0f);
  if (has_texcoords) {
    texcoord_0 = load_float2(texcoord_buffer, tri.i.x);
    texcoord_1 = load_float2(texcoord_buffer, tri.i.y);
    texcoord_2 = load_float2(texcoord_buffer, tri.i.z);
  }

  surface_point_shared_interpolate_vertex(p0, p1, p2, n0, n1, n2, tangent_0, tangent_1, tangent_2, bitangent_0, bitangent_1, bitangent_2, texcoord_0, texcoord_1,
    texcoord_2, result.barycentrics, has_surface_frame, has_texcoords, result.vertex);

  result.geo_normal = surface_point_shared_orient_geo_normal(tri.geo_n, ray_dir);
  result.vertex.nrm = surface_point_shared_orient_shading_normal(result.vertex.nrm, result.geo_normal);

  return result;
}

float2 interpolate_uv_from_barycentrics(ByteAddressBuffer texcoord_buffer, TriangleData tri, float3 bc) {
  float2 t0 = load_float2(texcoord_buffer, tri.i.x);
  float2 t1 = load_float2(texcoord_buffer, tri.i.y);
  float2 t2 = load_float2(texcoord_buffer, tri.i.z);
  return surface_point_shared_lerp_float2(t0, t1, t2, bc);
}

float2 interpolate_uv(ByteAddressBuffer texcoord_buffer, TriangleData tri, float2 bary) {
  return interpolate_uv_from_barycentrics(texcoord_buffer, tri, surface_point_shared_barycentrics(bary));
}

bool alpha_test_pass(uint material_index, float2 uv, inout uint seed) {
  uint context = 0u;
  return alpha_test_access_shared_pass(context, material_index, uv, seed);
}

bool material_scattering_evaluate_shared_gpu_try_load_state(uint context, uint material_index, out uint scattering_spectrum_index, out uint scattering_image_index) {
  return try_load_material_scattering_state(material_index, scattering_spectrum_index, scattering_image_index);
}

float3 material_scattering_evaluate_shared_gpu_load_spectrum_integrated(uint context, uint scattering_spectrum_index) {
  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  return load_spectrum_integrated_value(spectrum_buffer, scattering_spectrum_index);
}

SpectralResponse material_scattering_evaluate_shared_gpu_load_spectrum_spectral(uint context, uint scattering_spectrum_index, SpectralQuery spect) {
  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  return load_spectrum_response(spectrum_buffer, scattering_spectrum_index, spect);
}

bool material_scattering_evaluate_shared_gpu_can_apply_image(uint context, uint scattering_image_index) {
  return scene_resource_shared_can_apply_image(constants.scene.images, scattering_image_index);
}

float3 material_scattering_evaluate_shared_gpu_evaluate_image_rgb(uint context, uint scattering_image_index, float2 uv) {
  return evaluate_image(scattering_image_index, uv).xyz;
}

#define ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CONTEXT_TYPE uint
#define ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_TRY_LOAD_STATE(context, material_index, scattering_spectrum_index, scattering_image_index) \
  material_scattering_evaluate_shared_gpu_try_load_state(context, material_index, scattering_spectrum_index, scattering_image_index)
#define ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_LOAD_SPECTRUM_INTEGRATED(context, scattering_spectrum_index) \
  material_scattering_evaluate_shared_gpu_load_spectrum_integrated(context, scattering_spectrum_index)
#define ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_LOAD_SPECTRUM_SPECTRAL(context, scattering_spectrum_index, spect) \
  material_scattering_evaluate_shared_gpu_load_spectrum_spectral(context, scattering_spectrum_index, spect)
#define ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CAN_APPLY_IMAGE(context, scattering_image_index) \
  material_scattering_evaluate_shared_gpu_can_apply_image(context, scattering_image_index)
#define ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_EVALUATE_IMAGE_RGB(context, scattering_image_index, uv) \
  material_scattering_evaluate_shared_gpu_evaluate_image_rgb(context, scattering_image_index, uv)
#include <interop/material_scattering_evaluate_shared.hxx>
#undef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_EVALUATE_IMAGE_RGB
#undef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CAN_APPLY_IMAGE
#undef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_LOAD_SPECTRUM_SPECTRAL
#undef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_LOAD_SPECTRUM_INTEGRATED
#undef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_TRY_LOAD_STATE
#undef ETX_MATERIAL_SCATTERING_EVALUATE_SHARED_CONTEXT_TYPE

SpectralResponse evaluate_material_scattering_spectral(uint material_index, float2 uv, float ao, SpectralQuery spect, SpectralResponse fallback_value) {
  uint context = 0u;
  return material_scattering_evaluate_shared_spectral(context, material_index, uv, ao, spect, fallback_value);
}

float3 apply_image_integrated_or_fallback(uint material_index, float2 uv, float ao, float3 fallback_color) {
  uint context = 0u;
  return material_scattering_evaluate_shared_integrated_or_fallback(context, material_index, uv, ao, fallback_color);
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

bool emitter_emission_shared_gpu_has_required_scene_buffers(uint context) {
  return scene_resource_shared_has_emitter_buffers(
    constants.scene.emitter_instances, constants.scene.emitter_profiles, constants.scene.spectrums, constants.scene.scene_globals);
}

uint emitter_emission_shared_gpu_load_emitter_instance_count(uint context) {
  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGlobalsGPUSharedContext globals_context = make_scene_globals_gpu_shared_context(scene_globals);
  return scene_globals_shared_emitter_instance_count(globals_context);
}

uint emitter_emission_shared_gpu_load_emitter_profile_count(uint context) {
  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGlobalsGPUSharedContext globals_context = make_scene_globals_gpu_shared_context(scene_globals);
  return scene_globals_shared_emitter_profile_count(globals_context);
}

uint emitter_emission_shared_gpu_load_instance_class(uint context, uint emitter_index) {
  ByteAddressBuffer emitter_instance_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_instances)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(emitter_instance_buffer);
  return gpu_abi_access_shared_emitter_class(access_context, emitter_index);
}

uint emitter_emission_shared_gpu_load_instance_profile_index(uint context, uint emitter_index) {
  ByteAddressBuffer emitter_instance_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_instances)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(emitter_instance_buffer);
  return gpu_abi_access_shared_emitter_profile_index(access_context, emitter_index);
}

uint emitter_emission_shared_gpu_load_profile_emission_spectrum_index(uint context, uint emitter_profile_index) {
  ByteAddressBuffer emitter_profile_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_profiles)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(emitter_profile_buffer);
  return gpu_abi_access_shared_emitter_emission_spectrum_index(access_context, emitter_profile_index);
}

uint emitter_emission_shared_gpu_load_profile_emission_image_index(uint context, uint emitter_profile_index) {
  ByteAddressBuffer emitter_profile_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_profiles)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(emitter_profile_buffer);
  return gpu_abi_access_shared_emitter_emission_image_index(access_context, emitter_profile_index);
}

uint emitter_emission_shared_gpu_load_profile_class(uint context, uint emitter_profile_index) {
  ByteAddressBuffer emitter_profile_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_profiles)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(emitter_profile_buffer);
  return gpu_abi_access_shared_emitter_profile_class(access_context, emitter_profile_index);
}

uint emitter_emission_shared_gpu_load_profile_meta(uint context, uint emitter_profile_index) {
  ByteAddressBuffer emitter_profile_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_profiles)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(emitter_profile_buffer);
  return gpu_abi_access_shared_emitter_profile_meta(access_context, emitter_profile_index);
}

float3 emitter_emission_shared_gpu_load_profile_direction(uint context, uint emitter_profile_index) {
  ByteAddressBuffer emitter_profile_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_profiles)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(emitter_profile_buffer);
  return gpu_abi_access_shared_emitter_profile_direction(access_context, emitter_profile_index);
}

float emitter_emission_shared_gpu_load_profile_angular_size_cosine(uint context, uint emitter_profile_index) {
  ByteAddressBuffer emitter_profile_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_profiles)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(emitter_profile_buffer);
  return gpu_abi_access_shared_emitter_profile_angular_size_cosine(access_context, emitter_profile_index);
}

#define ETX_EMITTER_EMISSION_SHARED_CONTEXT_TYPE uint
#define ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE EmitterEmissionAccess
#define ETX_EMITTER_EMISSION_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) emitter_emission_shared_gpu_has_required_scene_buffers(context)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_INSTANCE_COUNT(context) emitter_emission_shared_gpu_load_emitter_instance_count(context)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_PROFILE_COUNT(context) emitter_emission_shared_gpu_load_emitter_profile_count(context)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_CLASS(context, emitter_index) emitter_emission_shared_gpu_load_instance_class(context, emitter_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_PROFILE_INDEX(context, emitter_index) emitter_emission_shared_gpu_load_instance_profile_index(context, emitter_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_SPECTRUM_INDEX(context, emitter_profile_index) \
  emitter_emission_shared_gpu_load_profile_emission_spectrum_index(context, emitter_profile_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_IMAGE_INDEX(context, emitter_profile_index) \
  emitter_emission_shared_gpu_load_profile_emission_image_index(context, emitter_profile_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_CLASS(context, emitter_profile_index) emitter_emission_shared_gpu_load_profile_class(context, emitter_profile_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_META(context, emitter_profile_index) emitter_emission_shared_gpu_load_profile_meta(context, emitter_profile_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_DIRECTION(context, emitter_profile_index) emitter_emission_shared_gpu_load_profile_direction(context, emitter_profile_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_ANGULAR_SIZE_COSINE(context, emitter_profile_index) \
  emitter_emission_shared_gpu_load_profile_angular_size_cosine(context, emitter_profile_index)
#include <interop/emitter_emission_shared.hxx>
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_ANGULAR_SIZE_COSINE
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_DIRECTION
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_META
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_CLASS
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_IMAGE_INDEX
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_SPECTRUM_INDEX
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_PROFILE_INDEX
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_CLASS
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_PROFILE_COUNT
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_INSTANCE_COUNT
#undef ETX_EMITTER_EMISSION_SHARED_HAS_REQUIRED_SCENE_BUFFERS
#undef ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE
#undef ETX_EMITTER_EMISSION_SHARED_CONTEXT_TYPE

bool try_load_emitter_scene_state(out uint emitter_instance_count, out uint emitter_profile_count) {
  uint context = 0u;
  return emitter_emission_shared_try_load_scene_state(context, emitter_instance_count, emitter_profile_count);
}

bool try_load_emitter_emission_access(uint emitter_index, out EmitterEmissionAccess access) {
  uint context = 0u;
  return emitter_emission_shared_try_load_access(context, emitter_index, access);
}

bool try_load_local_emission_access(uint emitter_index, out EmitterEmissionAccess access) {
  uint context = 0u;
  return emitter_emission_shared_try_load_local_access(context, emitter_index, access);
}

bool try_load_distant_emission_access(uint emitter_index, float3 direction, out EmitterEmissionAccess access) {
  uint context = 0u;
  return emitter_emission_shared_try_load_distant_access(context, emitter_index, direction, access);
}

bool emission_source_shared_gpu_can_sample_spectrum(uint context, uint emission_spectrum_index) {
  return scene_resource_shared_can_sample_spectrum(constants.scene.spectrums, emission_spectrum_index);
}

float3 emission_source_shared_gpu_load_spectrum_integrated(uint context, uint emission_spectrum_index) {
  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  return load_spectrum_integrated_value(spectrum_buffer, emission_spectrum_index);
}

SpectralResponse emission_source_shared_gpu_load_spectrum_spectral(uint context, uint emission_spectrum_index, SpectralQuery spect) {
  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  return load_spectrum_response(spectrum_buffer, emission_spectrum_index, spect);
}

bool emission_source_shared_gpu_can_apply_image(uint context, uint emission_image_index) {
  return scene_resource_shared_can_apply_image(constants.scene.images, emission_image_index);
}

float3 emission_source_shared_gpu_evaluate_image_rgb(uint context, uint emission_image_index, float2 uv) {
  float4 image_value = image_evaluate_shared_sample_whole_or_default(context, emission_image_index, uv, float4(1.0f, 1.0f, 1.0f, 1.0f));
  return image_value.xyz;
}

#define ETX_EMISSION_SOURCE_SHARED_CONTEXT_TYPE uint
#define ETX_EMISSION_SOURCE_SHARED_CAN_SAMPLE_SPECTRUM(context, emission_spectrum_index) emission_source_shared_gpu_can_sample_spectrum(context, emission_spectrum_index)
#define ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_INTEGRATED(context, emission_spectrum_index) \
  emission_source_shared_gpu_load_spectrum_integrated(context, emission_spectrum_index)
#define ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_SPECTRAL(context, emission_spectrum_index, spect) \
  emission_source_shared_gpu_load_spectrum_spectral(context, emission_spectrum_index, spect)
#define ETX_EMISSION_SOURCE_SHARED_CAN_APPLY_IMAGE(context, emission_image_index) emission_source_shared_gpu_can_apply_image(context, emission_image_index)
#define ETX_EMISSION_SOURCE_SHARED_EVALUATE_IMAGE_RGB(context, emission_image_index, uv) emission_source_shared_gpu_evaluate_image_rgb(context, emission_image_index, uv)
#include <interop/emission_source_shared.hxx>
#undef ETX_EMISSION_SOURCE_SHARED_EVALUATE_IMAGE_RGB
#undef ETX_EMISSION_SOURCE_SHARED_CAN_APPLY_IMAGE
#undef ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_SPECTRAL
#undef ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_INTEGRATED
#undef ETX_EMISSION_SOURCE_SHARED_CAN_SAMPLE_SPECTRUM
#undef ETX_EMISSION_SOURCE_SHARED_CONTEXT_TYPE

float3 evaluate_emission_integrated_source(uint emission_spectrum_index, uint emission_image_index, float2 uv) {
  uint context = 0u;
  return emission_source_shared_evaluate_integrated(context, emission_spectrum_index, emission_image_index, uv);
}

SpectralResponse evaluate_emission_spectral_source(uint emission_spectrum_index, uint emission_image_index, float2 uv, SpectralQuery spect) {
  uint context = 0u;
  return emission_source_shared_evaluate_spectral(context, emission_spectrum_index, emission_image_index, uv, spect);
}

float3 evaluate_local_emission_integrated(uint emitter_index, float2 uv) {
  ETX_ZERO_INIT(EmitterEmissionAccess, access);
  if (try_load_local_emission_access(emitter_index, access) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  return evaluate_emission_integrated_source(access.emission_spectrum_index, access.emission_image_index, uv);
}

SpectralResponse evaluate_local_emission_spectral(uint emitter_index, float2 uv, SpectralQuery spect) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  ETX_ZERO_INIT(EmitterEmissionAccess, access);
  if (try_load_local_emission_access(emitter_index, access) == false) {
    return zero_value;
  }

  return evaluate_emission_spectral_source(access.emission_spectrum_index, access.emission_image_index, uv, spect);
}

bool environment_emission_uv_shared_gpu_is_environment_class(uint emitter_class) {
  return emitter_class == EmitterClass::Environment;
}

bool environment_emission_uv_shared_gpu_is_directional_class(uint emitter_class) {
  return emitter_class == EmitterClass::Directional;
}

bool environment_emission_uv_shared_gpu_try_load_image_params(uint context, uint emission_image_index, out float2 image_offset, out float image_u_scale) {
  image_offset = float2(0.0f, 0.0f);
  image_u_scale = 1.0f;
  ImageSceneAccessGPUContext access_context = {constants.scene.images};
  ETX_ZERO_INIT(ImageDescAccess, image_access);
  if (image_scene_access_shared_try_load_desc(access_context, emission_image_index, image_access) == false) {
    return false;
  }

  image_offset = image_access.uv_offset;
  image_u_scale = image_access.uv_scale.x;
  return true;
}

#define ETX_ENVIRONMENT_EMISSION_UV_SHARED_CONTEXT_TYPE uint
#define ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_ENVIRONMENT_CLASS(emitter_class) environment_emission_uv_shared_gpu_is_environment_class(emitter_class)
#define ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_DIRECTIONAL_CLASS(emitter_class) environment_emission_uv_shared_gpu_is_directional_class(emitter_class)
#define ETX_ENVIRONMENT_EMISSION_UV_SHARED_TRY_LOAD_IMAGE_PARAMS(context, emission_image_index, image_offset, image_u_scale) \
  environment_emission_uv_shared_gpu_try_load_image_params(context, emission_image_index, image_offset, image_u_scale)
#include <interop/environment_emission_uv_shared.hxx>
#undef ETX_ENVIRONMENT_EMISSION_UV_SHARED_TRY_LOAD_IMAGE_PARAMS
#undef ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_DIRECTIONAL_CLASS
#undef ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_ENVIRONMENT_CLASS
#undef ETX_ENVIRONMENT_EMISSION_UV_SHARED_CONTEXT_TYPE

float2 environment_emission_uv(uint emitter_class, uint emitter_profile_meta, uint emission_image_index, float3 emitter_direction, float emitter_angular_size_cosine, float3 direction) {
  uint context = 0u;
  return environment_emission_uv_shared(
    context, emitter_class, emitter_profile_meta, emission_image_index, emitter_direction, emitter_angular_size_cosine, direction);
}

float3 evaluate_distant_emission_integrated(uint emitter_index, float3 direction) {
  ETX_ZERO_INIT(EmitterEmissionAccess, access);
  if (try_load_distant_emission_access(emitter_index, direction, access) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  float2 uv = environment_emission_uv(
    access.emitter_class, access.emitter_profile_meta, access.emission_image_index, access.emitter_direction, access.emitter_angular_size_cosine, direction);
  return evaluate_emission_integrated_source(access.emission_spectrum_index, access.emission_image_index, uv);
}

SpectralResponse evaluate_distant_emission_spectral(uint emitter_index, float3 direction, SpectralQuery spect) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  ETX_ZERO_INIT(EmitterEmissionAccess, access);
  if (try_load_distant_emission_access(emitter_index, direction, access) == false) {
    return zero_value;
  }

  float2 uv = environment_emission_uv(
    access.emitter_class, access.emitter_profile_meta, access.emission_image_index, access.emitter_direction, access.emitter_angular_size_cosine, direction);
  return evaluate_emission_spectral_source(access.emission_spectrum_index, access.emission_image_index, uv, spect);
}

struct EnvironmentEmitterSelectGPUSharedContext {
  uint seed;
};

bool environment_emitter_select_shared_gpu_has_scene_globals(ETX_IN(EnvironmentEmitterSelectGPUSharedContext, context)) {
  return constants.scene.scene_globals != kInvalidIndex;
}

uint environment_emitter_select_shared_gpu_load_emitter_instance_count(ETX_IN(EnvironmentEmitterSelectGPUSharedContext, context)) {
  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGlobalsGPUSharedContext globals_context = make_scene_globals_gpu_shared_context(scene_globals);
  return scene_globals_shared_emitter_instance_count(globals_context);
}

uint environment_emitter_select_shared_gpu_load_environment_emitter_count(ETX_IN(EnvironmentEmitterSelectGPUSharedContext, context)) {
  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGlobalsGPUSharedContext globals_context = make_scene_globals_gpu_shared_context(scene_globals);
  return scene_globals_shared_environment_emitter_count(globals_context);
}

uint environment_emitter_select_shared_gpu_load_environment_emitter(ETX_IN(EnvironmentEmitterSelectGPUSharedContext, context), uint index) {
  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGlobalsGPUSharedContext globals_context = make_scene_globals_gpu_shared_context(scene_globals);
  return scene_globals_shared_environment_emitter(globals_context, index);
}

uint environment_emitter_select_shared_gpu_max_count(ETX_IN(EnvironmentEmitterSelectGPUSharedContext, context)) {
  return SceneLimits::MaxEnvironmentEmitters;
}

float environment_emitter_select_shared_gpu_rnd(ETX_INOUT(EnvironmentEmitterSelectGPUSharedContext, context)) {
  return rnd01(context.seed);
}

#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_CONTEXT_TYPE EnvironmentEmitterSelectGPUSharedContext
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_HAS_SCENE_GLOBALS(context) environment_emitter_select_shared_gpu_has_scene_globals(context)
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_EMITTER_INSTANCE_COUNT(context) environment_emitter_select_shared_gpu_load_emitter_instance_count(context)
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER_COUNT(context) environment_emitter_select_shared_gpu_load_environment_emitter_count(context)
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER(context, index) environment_emitter_select_shared_gpu_load_environment_emitter(context, index)
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_MAX_COUNT(context) environment_emitter_select_shared_gpu_max_count(context)
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_RND(context) environment_emitter_select_shared_gpu_rnd(context)
#include <interop/environment_emitter_select_shared.hxx>
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_RND
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_MAX_COUNT
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER_COUNT
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_EMITTER_INSTANCE_COUNT
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_HAS_SCENE_GLOBALS
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_CONTEXT_TYPE

bool try_select_environment_emitter_random(inout uint seed, out uint emitter_index, out uint emitter_count) {
  EnvironmentEmitterSelectGPUSharedContext context = {seed};
  bool result = environment_emitter_select_shared_try_select_random(context, emitter_index, emitter_count);
  seed = context.seed;
  return result;
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

