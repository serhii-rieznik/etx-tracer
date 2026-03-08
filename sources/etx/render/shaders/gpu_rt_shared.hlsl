#pragma once

#include "bindless.hlsl"

#include <access/spectrum_access_gpu.hxx>
#include <interop/geometry.hxx>
#include <interop/gpu_abi_constants.hxx>
#include <interop/gpu_abi_access_shared.hxx>
#include <interop/hit_policy.hxx>
#include <interop/image.hxx>
#include <interop/image_filter_shared.hxx>
#include <interop/material.hxx>
#include <interop/material_scattering_shared.hxx>
#include <interop/projection.hxx>
#include <interop/camera_shared.hxx>
#include <interop/camera_film_shared.hxx>
#include <interop/scene_options_shared.hxx>
#include <interop/scene_globals_shared.hxx>
#include <interop/scene_resource_shared.hxx>
#include <access/image_access_gpu.hxx>
#include <access/image_evaluate_gpu.hxx>
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

#include <access/material_access_gpu.hxx>

float3 load_spectrum_integrated_value(ByteAddressBuffer buffer, uint spectrum_index) {
  SpectrumAccessGPUContext context = make_spectrum_access_gpu_context(buffer, 0u);
  return spectrum_access_load_integrated(context, spectrum_index);
}

SpectralResponse load_spectrum_response(ByteAddressBuffer buffer, uint spectrum_index, SpectralQuery spect) {
  SpectrumAccessGPUContext context = make_spectrum_access_gpu_context(buffer, 0u);
  return spectrum_access_evaluate(context, spectrum_index, spect);
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

#include <access/medium_access_gpu.hxx>

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

float medium_sample_texture_3d(MediumAccess medium_access, float3 local_coord) {
  if ((medium_access.grid.density_count == 0u) || (medium_access.grid.density_data_offset == kInvalidIndex) || (medium_access.density_payload_descriptor_index == kInvalidIndex)) {
    return 0.0f;
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(medium_access.density_payload_descriptor_index)];
  MediumTextureSampleContext context = {payload_buffer, medium_access.grid.density_data_offset, medium_access.grid.density_count};
  return medium_texture_sample_shared_3d(context, local_coord, medium_access.grid.dimensions);
}

float medium_sample_noise(MediumAccess medium_access, float3 local_coord) {
  return medium_density_shared_sample_noise(local_coord, medium_access.bounds_min, medium_access.bounds_max, medium_access.grid.noise_type, medium_access.grid.noise_scale,
    medium_access.grid.noise_octaves, medium_access.grid.noise_lacunarity, medium_access.grid.noise_persistence, medium_access.grid.noise_seed,
    medium_access.grid.noise_offset, medium_access.grid.noise_enable_border_fade, medium_access.grid.noise_border_fade_distance);
}

float medium_sample_density(MediumAccess medium_access, float3 local_coord) {
  float value = 0.0f;
  if (medium_access.grid.type == MediumGridType::NoiseFunction) {
    value = medium_sample_noise(medium_access, local_coord);
  } else if (medium_access.grid.type == MediumGridType::Texture3D) {
    value = medium_sample_texture_3d(medium_access, local_coord);
  }

  return medium_density_shared_apply_shape(value, medium_access.grid.noise_power, medium_access.grid.noise_sharpness);
}

bool medium_has_grid_data(MediumAccess medium_access) {
  if (medium_density_shared_has_grid_data(medium_access.grid.type, medium_access.grid.dimensions, medium_access.grid.density_count) == false) {
    return false;
  }

  if (medium_access.grid.type == MediumGridType::NoiseFunction) {
    return true;
  }

  return (medium_access.grid.density_data_offset != kInvalidIndex) && (medium_access.density_payload_descriptor_index != kInvalidIndex);
}

struct MediumSharedContext {
  MediumAccess medium_access;
  uint seed;
  uint medium_class;
  uint has_grid_data;
  float3 bounds_min;
  float3 bounds_max;
};

float medium_shared_rnd(inout MediumSharedContext context) {
  return rnd01(context.seed);
}

float medium_shared_density(inout MediumSharedContext context, float3 local_pos) {
  return medium_sample_density(context.medium_access, local_pos);
}

#include <interop/medium_transmittance_shared.hxx>
#include <interop/medium_sample_shared.hxx>

MediumSample sample_medium_gpu(
  MediumAccess medium_access, SpectralQuery spect, SpectralResponse throughput, SpectralResponse scattering_value, SpectralResponse absorption_value, float3 pos, float3 w_i,
  float max_t, inout uint seed) {
  MediumSharedContext context;
  context.medium_access = medium_access;
  context.seed = seed;
  context.medium_class = medium_access.medium_class;
  context.has_grid_data = medium_has_grid_data(medium_access) ? 1u : 0u;
  context.bounds_min = medium_access.bounds_min;
  context.bounds_max = medium_access.bounds_max;
  MediumSample result = medium_sample_shared_sample(context, spect, throughput, scattering_value, absorption_value, pos, w_i, max_t);
  seed = context.seed;
  return result;
}

float3 medium_segment_transmittance_integrated(uint medium_index, float3 origin, float3 direction, float distance, inout uint seed) {
  float3 one = float3(1.0f, 1.0f, 1.0f);
  if ((distance <= 0.0f) || (scene_resource_shared_has_medium_spectrum_buffers(constants.scene.mediums, constants.scene.spectrums) == false)) {
    return one;
  }

  MediumAccessGPUContext access_context = make_medium_access_gpu_context(constants.scene.mediums, constants.scene.spectrums);
  ETX_ZERO_INIT(MediumAccess, medium_access);
  if (medium_access_try_load(access_context, medium_index, medium_access) == false) {
    return one;
  }
  if (medium_access_supported_class(medium_access.medium_class) == false) {
    return one;
  }

  float3 extinction = medium_access_load_extinction_integrated(access_context, medium_access);
  if (medium_access.medium_class == Medium::Homogeneous) {
    return medium_shared_transmittance_homogeneous_integrated(extinction, distance);
  }
  if (medium_has_grid_data(medium_access) == false) {
    return one;
  }

  MediumSharedContext medium_context;
  medium_context.medium_access = medium_access;
  medium_context.seed = seed;
  medium_context.medium_class = medium_access.medium_class;
  medium_context.has_grid_data = 1u;
  medium_context.bounds_min = medium_access.bounds_min;
  medium_context.bounds_max = medium_access.bounds_max;
  float3 transmittance =
    medium_shared_transmittance_heterogeneous_integrated(extinction, origin, direction, distance, medium_access.bounds_min, medium_access.bounds_max, medium_context);
  seed = medium_context.seed;
  return transmittance;
}

SpectralResponse medium_segment_transmittance_spectral(uint medium_index, float3 origin, float3 direction, float distance, SpectralQuery spect, inout uint seed) {
  SpectralResponse one = spectral_response_make(spect, 1.0f);
  if ((distance <= 0.0f) || (scene_resource_shared_has_medium_spectrum_buffers(constants.scene.mediums, constants.scene.spectrums) == false)) {
    return one;
  }

  MediumAccessGPUContext access_context = make_medium_access_gpu_context(constants.scene.mediums, constants.scene.spectrums);
  ETX_ZERO_INIT(MediumAccess, medium_access);
  if (medium_access_try_load(access_context, medium_index, medium_access) == false) {
    return one;
  }
  if (medium_access_supported_class(medium_access.medium_class) == false) {
    return one;
  }

  SpectralResponse extinction = medium_access_load_extinction_spectral(access_context, medium_access, spect);
  if (medium_access.medium_class == Medium::Homogeneous) {
    return medium_shared_transmittance_homogeneous_spectral(extinction, distance);
  }
  if (medium_has_grid_data(medium_access) == false) {
    return one;
  }

  MediumSharedContext medium_context;
  medium_context.medium_access = medium_access;
  medium_context.seed = seed;
  medium_context.medium_class = medium_access.medium_class;
  medium_context.has_grid_data = 1u;
  medium_context.bounds_min = medium_access.bounds_min;
  medium_context.bounds_max = medium_access.bounds_max;
  SpectralResponse transmittance =
    medium_shared_transmittance_heterogeneous_spectral(extinction, origin, direction, distance, medium_access.bounds_min, medium_access.bounds_max, medium_context, spect);
  seed = medium_context.seed;
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

struct DistributionGPUContext {
  ByteAddressBuffer payload_buffer;
  uint entries_base_offset;
};

DistributionGPUContext make_distribution_gpu_context(ByteAddressBuffer payload_buffer, uint entries_base_offset) {
  DistributionGPUContext context;
  context.payload_buffer = payload_buffer;
  context.entries_base_offset = entries_base_offset;
  return context;
}

float distribution_gpu_cdf(DistributionGPUContext context, uint index) {
  DistributionEntry entry = load_distribution_entry(context.payload_buffer, context.entries_base_offset + index * kDistributionEntryStride);
  return entry.cdf;
}

float distribution_gpu_pdf(DistributionGPUContext context, uint index) {
  DistributionEntry entry = load_distribution_entry(context.payload_buffer, context.entries_base_offset + index * kDistributionEntryStride);
  return entry.pdf;
}

uint sample_distribution(ByteAddressBuffer payload_buffer, uint entries_base_offset, uint count, float rnd, out float pdf) {
  if (count == 0u) {
    pdf = 0.0f;
    return kInvalidIndex;
  }

  DistributionGPUContext context = make_distribution_gpu_context(payload_buffer, entries_base_offset);
  DistributionSearchRange search = distribution_search_begin(count);
  while (distribution_search_active(search)) {
    uint middle = distribution_search_middle(search);
    float middle_cdf = distribution_gpu_cdf(context, middle);
    distribution_search_update(search, middle, middle_cdf, rnd);
  }

  pdf = distribution_gpu_pdf(context, search.begin);
  return search.begin;
}

struct ImageSampleGPUContext {
  ByteAddressBuffer x_payload_buffer;
  ByteAddressBuffer y_payload_buffer;
  uint x_distribution_entries_offset;
  uint y_distribution_entries_offset;
  uint x_entries_stride;
  uint x_distribution_count;
  uint y_count;
  float2 fsize;
};

ImageSampleGPUContext make_image_sample_gpu_context(
  ByteAddressBuffer x_payload_buffer, ByteAddressBuffer y_payload_buffer, ImageAccessGPUDesc image_access, uint y_count) {
  ImageSampleGPUContext context;
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

uint image_sample_gpu_y_count(ImageSampleGPUContext context) {
  return context.y_count;
}

uint image_sample_gpu_x_count(ImageSampleGPUContext context, uint y_index) {
  if ((y_index >= context.x_distribution_count) || (context.x_entries_stride == 0u)) {
    return 0u;
  }

  return context.x_entries_stride - 1u;
}

float2 image_sample_gpu_image_fsize(ImageSampleGPUContext context) {
  return context.fsize;
}

uint image_sample_gpu_sample_y(inout ImageSampleGPUContext context, float rnd, out float pdf) {
  return sample_distribution(context.y_payload_buffer, context.y_distribution_entries_offset, context.y_count, rnd, pdf);
}

uint image_sample_gpu_sample_x(inout ImageSampleGPUContext context, uint y_index, float rnd, out float pdf) {
  uint row_base_offset = context.x_distribution_entries_offset + y_index * context.x_entries_stride * kDistributionEntryStride;
  uint x_count = image_sample_gpu_x_count(context, y_index);
  return sample_distribution(context.x_payload_buffer, row_base_offset, x_count, rnd, pdf);
}

float image_sample_gpu_cdf_y(ImageSampleGPUContext context, uint y_index) {
  uint y_offset = context.y_distribution_entries_offset + y_index * kDistributionEntryStride;
  DistributionEntry entry = load_distribution_entry(context.y_payload_buffer, y_offset);
  return entry.cdf;
}

float image_sample_gpu_cdf_x(ImageSampleGPUContext context, uint y_index, uint x_index) {
  uint row_base_offset = context.x_distribution_entries_offset + y_index * context.x_entries_stride * kDistributionEntryStride;
  uint x_offset = row_base_offset + x_index * kDistributionEntryStride;
  DistributionEntry entry = load_distribution_entry(context.x_payload_buffer, x_offset);
  return entry.cdf;
}

bool image_sample_distribution_gpu(inout ImageSampleGPUContext context, float2 rnd, out float image_pdf, out uint2 location, out float2 uv) {
  image_pdf = 0.0f;
  location = uint2(0u, 0u);
  uv = rnd;

  uint y_count = image_sample_gpu_y_count(context);
  if (y_count == 0u) {
    return false;
  }

  float y_pdf = 0.0f;
  location.y = image_sample_gpu_sample_y(context, rnd.y, y_pdf);
  if ((location.y == kInvalidIndex) || (location.y >= y_count)) {
    return false;
  }

  uint x_count = image_sample_gpu_x_count(context, location.y);
  if (x_count == 0u) {
    return false;
  }

  float x_pdf = 0.0f;
  location.x = image_sample_gpu_sample_x(context, location.y, rnd.x, x_pdf);
  if ((location.x == kInvalidIndex) || (location.x >= x_count)) {
    return false;
  }

  uint x1_index = min(location.x + 1u, x_count);
  uint y1_index = min(location.y + 1u, y_count);

  float x0_cdf = image_sample_gpu_cdf_x(context, location.y, location.x);
  float x1_cdf = image_sample_gpu_cdf_x(context, location.y, x1_index);
  float y0_cdf = image_sample_gpu_cdf_y(context, location.y);
  float y1_cdf = image_sample_gpu_cdf_y(context, y1_index);

  uv = image_sample_uv_from_distribution(rnd, location, image_sample_gpu_image_fsize(context), x0_cdf, x1_cdf, y0_cdf, y1_cdf);
  image_pdf = x_pdf * y_pdf;
  return true;
}

float2 sample_image_uv_fallback(uint image_index, float2 fallback_uv, out float4 eval) {
  eval = evaluate_image(image_index, fallback_uv);
  return fallback_uv;
}

float2 sample_image_uv(uint image_index, float2 rnd, out float image_pdf, out uint2 location, out float4 eval) {
  image_pdf = 0.0f;
  location = uint2(0u, 0u);
  eval = float4(1.0f, 1.0f, 1.0f, 1.0f);

  ImageAccessGPUContext access_context = {constants.scene.images};
  ETX_ZERO_INIT(ImageAccessGPUDesc, image_access);
  uint x_payload_descriptor_index = kInvalidIndex;
  uint y_payload_descriptor_index = kInvalidIndex;
  uint y_count = 0u;
  if (image_access_try_load_distribution_payloads(access_context, image_index, image_access, x_payload_descriptor_index, y_payload_descriptor_index, y_count) == false) {
    return sample_image_uv_fallback(image_index, rnd, eval);
  }

  ByteAddressBuffer x_payload_buffer = bindless_buffers[NonUniformResourceIndex(x_payload_descriptor_index)];
  ByteAddressBuffer y_payload_buffer = bindless_buffers[NonUniformResourceIndex(y_payload_descriptor_index)];

  ImageSampleGPUContext sample_context = make_image_sample_gpu_context(x_payload_buffer, y_payload_buffer, image_access, y_count);

  float2 uv = rnd;
  bool sampled = image_sample_distribution_gpu(sample_context, rnd, image_pdf, location, uv);
  if (sampled == false) {
    return sample_image_uv_fallback(image_index, rnd, eval);
  }

  eval = evaluate_image(image_index, uv);
  return uv;
}

bool camera_lens_sampling_enabled(float lens_radius, float focal_distance) {
  return (lens_radius > kEpsilon) && (focal_distance > kEpsilon);
}

float2 camera_primary_uv(uint2 pixel, uint2 film_size) {
  return camera_shared_flip_y(camera_shared_center_uv(pixel, film_size));
}

float2 camera_sample_lens_uv(Camera camera, float2 sensor_sample_rnd) {
  if (camera_lens_sampling_enabled(camera.lens_radius, camera.focal_distance) == false) {
    return float2(0.0f, 0.0f);
  }

  if (camera.lens_image == kInvalidIndex) {
    return sample_disk(sensor_sample_rnd);
  }

  ImageAccessGPUContext access_context = {constants.scene.images};
  ETX_ZERO_INIT(ImageAccessGPUDesc, image_access);
  if (image_access_try_load(access_context, camera.lens_image, image_access) == false) {
    return sample_disk(sensor_sample_rnd);
  }

  float image_pdf = 0.0f;
  uint2 image_location = uint2(0u, 0u);
  float4 image_eval = float4(1.0f, 1.0f, 1.0f, 1.0f);
  float2 image_uv = sample_image_uv(camera.lens_image, sensor_sample_rnd, image_pdf, image_location, image_eval);
  return image_uv * 2.0f - 1.0f;
}

float3 camera_lens_point(Camera camera, float2 sensor_sample_rnd) {
  float2 sensor_sample = camera_sample_lens_uv(camera, sensor_sample_rnd) * camera.lens_radius;
  return camera_film_shared_lens_point(camera, sensor_sample);
}

Ray camera_generate_primary_ray(Camera camera, float2 uv, float2 sensor_sample_rnd) {
  float2 sensor_sample = camera_sample_lens_uv(camera, sensor_sample_rnd);
  return camera_generate_ray(camera, uv, sensor_sample);
}

float4 evaluate_image(uint image_index, float2 uv) {
  ImageEvaluateGPUContext context = make_image_evaluate_gpu_context(constants.scene.images);
  return image_evaluate_gpu_rgba(context, image_index, uv);
}

bool image_has_alpha_channel(uint image_index) {
  ImageAccessGPUContext access_context = {constants.scene.images};
  return image_access_has_alpha(access_context, image_index);
}

struct AlphaTestContext {
  MaterialAccess material_access;
  float2 uv;
  uint seed;
};

uint alpha_test_material_class(AlphaTestContext context) {
  return context.material_access.material_class;
}

float alpha_test_material_opacity(AlphaTestContext context) {
  return context.material_access.opacity;
}

uint alpha_test_scattering_image_index(AlphaTestContext context) {
  return context.material_access.scattering_image_index;
}

bool alpha_test_image_has_alpha(AlphaTestContext context, uint image_index) {
  return image_has_alpha_channel(image_index);
}

float alpha_test_evaluate_alpha(AlphaTestContext context, uint image_index) {
  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(constants.scene.images);
  return image_evaluate_sample_channel_or_default(image_context, image_index, 3u, context.uv, 1.0f);
}

float alpha_test_rnd(inout AlphaTestContext context) {
  return rnd01(context.seed);
}

#include <interop/alpha_test_shared.hxx>

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
  if (scene_resource_shared_is_available(constants.scene.materials) == false) {
    return false;
  }

  AlphaTestContext alpha_context = ETX_ZERO(AlphaTestContext);
  MaterialAccessGPUContext material_context = {constants.scene.materials};
  material_access_try_load(material_context, material_index, alpha_context.material_access);
  alpha_context.uv = uv;
  alpha_context.seed = seed;
  bool result = alpha_test_shared_pass(alpha_context);
  seed = alpha_context.seed;
  return result;
}

SpectralResponse evaluate_material_scattering_spectral(uint material_index, float2 uv, float ao, SpectralQuery spect, SpectralResponse fallback_value) {
  MaterialAccessGPUContext material_context = {constants.scene.materials};
  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, constants.scene.spectrums);
  ImageAccessGPUContext image_access_context = {constants.scene.images};
  ImageEvaluateGPUContext image_evaluate_context = make_image_evaluate_gpu_context(constants.scene.images);
  return material_access_evaluate_scattering_spectral(
    material_context, spectrum_context, image_access_context, image_evaluate_context, material_index, uv, ao, spect, fallback_value);
}

float3 apply_image_integrated_or_fallback(uint material_index, float2 uv, float ao, float3 fallback_color) {
  MaterialAccessGPUContext material_context = {constants.scene.materials};
  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, constants.scene.spectrums);
  ImageAccessGPUContext image_access_context = {constants.scene.images};
  ImageEvaluateGPUContext image_evaluate_context = make_image_evaluate_gpu_context(constants.scene.images);
  return material_access_evaluate_scattering_integrated_or_fallback(
    material_context, spectrum_context, image_access_context, image_evaluate_context, material_index, uv, ao, fallback_color);
}

#include <access/emitter_access_gpu.hxx>

EmitterAccessGPUContext make_scene_emitter_access_gpu_context() {
  return make_emitter_access_gpu_context(
    constants.scene.emitter_instances, constants.scene.emitter_profiles, constants.scene.spectrums, constants.scene.images, constants.scene.scene_globals);
}

bool try_load_emitter_scene_state(out uint emitter_instance_count, out uint emitter_profile_count) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  return emitter_access_try_load_scene_state(context, emitter_instance_count, emitter_profile_count);
}

bool try_load_emitter_emission_access(uint emitter_index, out EmitterAccess access) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  return emitter_access_try_load(context, emitter_index, access);
}

bool try_load_local_emission_access(uint emitter_index, out EmitterAccess access) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  return emitter_access_try_load_local(context, emitter_index, access);
}

bool try_load_distant_emission_access(uint emitter_index, float3 direction, out EmitterAccess access) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  return emitter_access_try_load_distant(context, emitter_index, direction, access);
}

float3 evaluate_emission_integrated_source(uint emission_spectrum_index, uint emission_image_index, float2 uv) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  return emitter_access_evaluate_integrated_source(context, emission_spectrum_index, emission_image_index, uv);
}

SpectralResponse evaluate_emission_spectral_source(uint emission_spectrum_index, uint emission_image_index, float2 uv, SpectralQuery spect) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  return emitter_access_evaluate_spectral_source(context, emission_spectrum_index, emission_image_index, uv, spect);
}

float3 evaluate_local_emission_integrated(uint emitter_index, float2 uv) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  ETX_ZERO_INIT(EmitterAccess, access);
  if (emitter_access_try_load_local(context, emitter_index, access) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  return emitter_access_evaluate_integrated_source(context, access.emission_spectrum_index, access.emission_image_index, uv);
}

SpectralResponse evaluate_local_emission_spectral(uint emitter_index, float2 uv, SpectralQuery spect) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  ETX_ZERO_INIT(EmitterAccess, access);
  if (emitter_access_try_load_local(context, emitter_index, access) == false) {
    return zero_value;
  }

  return emitter_access_evaluate_spectral_source(context, access.emission_spectrum_index, access.emission_image_index, uv, spect);
}

float3 evaluate_distant_emission_integrated(uint emitter_index, float3 direction) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  ETX_ZERO_INIT(EmitterAccess, access);
  if (emitter_access_try_load_distant(context, emitter_index, direction, access) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  float2 uv = emitter_access_environment_uv(context, access, direction);
  return emitter_access_evaluate_integrated_source(context, access.emission_spectrum_index, access.emission_image_index, uv);
}

SpectralResponse evaluate_distant_emission_spectral(uint emitter_index, float3 direction, SpectralQuery spect) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  ETX_ZERO_INIT(EmitterAccess, access);
  if (emitter_access_try_load_distant(context, emitter_index, direction, access) == false) {
    return zero_value;
  }

  float2 uv = emitter_access_environment_uv(context, access, direction);
  return emitter_access_evaluate_spectral_source(context, access.emission_spectrum_index, access.emission_image_index, uv, spect);
}

uint load_environment_emitter_instance_count() {
  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGlobalsGPUSharedContext globals_context = make_scene_globals_gpu_shared_context(scene_globals);
  return scene_globals_shared_emitter_instance_count(globals_context);
}

uint load_environment_emitter_count() {
  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGlobalsGPUSharedContext globals_context = make_scene_globals_gpu_shared_context(scene_globals);
  return scene_globals_shared_environment_emitter_count(globals_context);
}

uint load_environment_emitter(uint index) {
  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGlobalsGPUSharedContext globals_context = make_scene_globals_gpu_shared_context(scene_globals);
  return scene_globals_shared_environment_emitter(globals_context, index);
}

bool try_select_environment_emitter_random(inout uint seed, out uint emitter_index, out uint emitter_count) {
  emitter_index = kInvalidIndex;
  emitter_count = 0u;

  if (constants.scene.scene_globals == kInvalidIndex) {
    return false;
  }

  uint emitter_instance_count = load_environment_emitter_instance_count();
  emitter_count = min(load_environment_emitter_count(), SceneLimits::MaxEnvironmentEmitters);
  if (emitter_count == 0u) {
    return false;
  }

  uint selected = uint(rnd01(seed) * float(emitter_count));
  if (selected >= emitter_count) {
    selected = emitter_count - 1u;
  }

  emitter_index = load_environment_emitter(selected);
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

