#pragma once

#if !defined(ETX_GPU_RT_BINDLESS_ALREADY_INCLUDED)
# include "bindless.hlsl"
#endif

#include <access/spectrum_access_gpu.hxx>
#include <interop/geometry.hxx>
#include <interop/gpu_abi_constants.hxx>
#include <interop/gpu_abi_access_shared.hxx>
#include <interop/hit_policy.hxx>
#include <interop/image.hxx>
#include <interop/image_filter_shared.hxx>
#include <interop/material.hxx>
#include <interop/projection.hxx>
#include <interop/distribution.hxx>
#include <interop/camera_shared.hxx>
#include <interop/camera_film_shared.hxx>
#include <interop/scene_gpu_access_shared.hxx>
#include <access/image_access_gpu.hxx>
#include <access/image_evaluate_gpu.hxx>
#include <access/image_sample_gpu.hxx>
#include <access/bsdf_resource_gpu.hxx>
#include <interop/sampler_policy.hxx>
#include <interop/sampler.hxx>
#include <interop/bsdf_dispatch_shared.hxx>
#include <interop/medium_density_shared.hxx>
#include <interop/medium_phase_shared.hxx>
#include <interop/surface_point_shared.hxx>
#include <interop/scene_math_shared.hxx>
#include <interop/diffraction_transport_shared.hxx>

static const uint kSceneStrategyDirectHit = 1u << 0u;
static const uint kSceneStrategyConnectToLight = 1u << 1u;
static const uint kSceneStrategyConnectToCamera = 1u << 2u;
static const uint kSceneStrategyConnectVertices = 1u << 3u;
static const uint kSceneStrategyMergeVertices = 1u << 4u;
static const uint kScenePathModePathTracing = 0u;
static const uint kScenePathModeLightTracing = 1u;
static const uint kScenePathModeBDPTFast = 2u;
static const uint kScenePathModeBDPTFull = 3u;
static const uint kScenePathModeVCM = 4u;
static const uint kSceneLightSamplingUniform = 0u;
static const uint kSceneLightSamplingFromDistribution = 1u;
static const uint kSceneLightSamplingRISUniform = 2u;
static const uint kSceneLightSamplingRISFromDistribution = 3u;

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

uint load_scene_options_random_seed() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return options.random_seed;
}

uint scene_random_seed(uint value_0, uint value_1) {
  return sampler_random_seed(value_0, value_1 ^ load_scene_options_random_seed());
}

float sample_blue_noise_value(uint2 pixel, uint sample_index, uint dimension) {
  if (constants.blue_noise_buffer_index == kInvalidIndex) {
    return 0.5f;
  }

  ByteAddressBuffer blue_noise_table = bindless_buffers[NonUniformResourceIndex(constants.blue_noise_buffer_index)];
  uint index = blue_noise_table_index(pixel, sample_index ^ load_scene_options_random_seed(), dimension);
  return asfloat(blue_noise_table.Load(index * 4u));
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

BSDFResourceContext make_scene_bsdf_resource_gpu_context() {
  return make_bsdf_resource_gpu_context(constants.scene.images, constants.scene.spectrums, constants.scene.energy_compensation_interfaces, constants.scene.scene_globals);
}

bool try_load_material_full(uint material_index, out Material material) {
  MaterialAccessGPUContext material_context = {constants.scene.materials};
  return material_access_try_load_full(material_context, material_index, material);
}

BSDFData make_surface_bsdf_data(Vertex vertex, SpectralQuery spect, uint medium_index, float3 incoming_direction) {
  return bsdf_data_make(vertex, spect, medium_index, PathSource::Camera, incoming_direction);
}

Sampler make_bsdf_sampler(uint seed) {
  Sampler result = ETX_ZERO(Sampler);
  result.seed = seed;
  return result;
}

Camera load_camera(ByteAddressBuffer camera_buffer) {
  return gpu_abi_load_camera(camera_buffer);
}

uint load_scene_options_samples() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return options.samples;
}

uint load_scene_options_min_path_length() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return options.min_path_length;
}

uint load_scene_options_properties_flags() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return options.properties_flags;
}

uint load_scene_options_max_path_length() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return options.max_path_length;
}

uint load_scene_options_random_path_termination() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return options.random_path_termination;
}

uint load_scene_options_strategy_flags() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return options.strategy_flags;
}

uint load_scene_options_light_sampling() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return options.light_sampling;
}

uint load_scene_options_path_mode() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return options.path_mode;
}

bool scene_uses_spectral_mode() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return scene_gpu_uses_spectral_mode(options);
}

bool scene_has_diffraction_grating() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return scene_gpu_has_diffraction_grating(options);
}

bool scene_diffraction_contribution_enabled(SpectralQuery spect, bool contains_diffraction) {
  bool partition = diffraction_transport_partition_enabled(scene_uses_spectral_mode(), scene_has_diffraction_grating());
  return diffraction_transport_contribution_enabled(partition, spect, contains_diffraction);
}

bool scene_multiple_importance_sampling_enabled() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return (options.properties_flags & (1u << SceneProperty::MultipleImportanceSampling)) != 0u;
}

bool scene_strategy_enabled(uint flag) {
  return (load_scene_options_strategy_flags() & flag) != 0u;
}

bool scene_path_mode_is_path_tracing() {
  return load_scene_options_path_mode() == kScenePathModePathTracing;
}

bool scene_path_mode_is_light_tracing() {
  return load_scene_options_path_mode() == kScenePathModeLightTracing;
}

bool scene_path_mode_uses_bdpt_fast() {
  return load_scene_options_path_mode() == kScenePathModeBDPTFast;
}

bool scene_path_mode_is_bdpt_full() {
  return load_scene_options_path_mode() == kScenePathModeBDPTFull;
}

#include <access/medium_access_gpu.hxx>

struct MediumSharedContext {
  MediumAccessGPUContext access_context;
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
  return medium_access_sample_density(context.access_context, context.medium_access, local_pos);
}

#include <interop/medium_transmittance_shared.hxx>
#include <interop/medium_sample_shared.hxx>

MediumSample sample_medium_gpu(MediumAccess medium_access, SpectralQuery spect, SpectralResponse throughput, SpectralResponse scattering_value, SpectralResponse absorption_value,
  float3 pos, float3 w_i, float max_t, inout uint seed) {
  MediumSharedContext context;
  context.access_context = make_medium_access_gpu_context(constants.scene.mediums, constants.scene.images, constants.scene.spectrums);
  context.medium_access = medium_access;
  context.seed = seed;
  context.medium_class = medium_access.medium_class;
  context.has_grid_data = medium_access_has_grid_data(context.access_context, medium_access) ? 1u : 0u;
  context.bounds_min = medium_access.bounds_min;
  context.bounds_max = medium_access.bounds_max;
  MediumSample result = medium_sample_shared_sample(context, spect, throughput, scattering_value, absorption_value, pos, w_i, max_t);
  seed = context.seed;
  return result;
}

float3 medium_segment_transmittance_integrated(uint medium_index, float3 origin, float3 direction, float distance, inout uint seed) {
  float3 one = float3(1.0f, 1.0f, 1.0f);
  if ((distance <= 0.0f) || (scene_gpu_has_medium_spectrum_buffers(constants.scene.mediums, constants.scene.spectrums) == false)) {
    return one;
  }

  MediumAccessGPUContext access_context = make_medium_access_gpu_context(constants.scene.mediums, constants.scene.images, constants.scene.spectrums);
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
  if (medium_access_has_grid_data(access_context, medium_access) == false) {
    return one;
  }

  MediumSharedContext medium_context;
  medium_context.access_context = access_context;
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
  if ((distance <= 0.0f) || (scene_gpu_has_medium_spectrum_buffers(constants.scene.mediums, constants.scene.spectrums) == false)) {
    return one;
  }

  MediumAccessGPUContext access_context = make_medium_access_gpu_context(constants.scene.mediums, constants.scene.images, constants.scene.spectrums);
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
  if (medium_access_has_grid_data(access_context, medium_access) == false) {
    return one;
  }

  MediumSharedContext medium_context;
  medium_context.access_context = access_context;
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

bool try_load_medium_access(uint medium_index, out MediumAccess medium_access) {
  MediumAccessGPUContext access_context = make_medium_access_gpu_context(constants.scene.mediums, constants.scene.images, constants.scene.spectrums);
  return medium_access_try_load(access_context, medium_index, medium_access);
}

SpectralResponse gpu_medium_scattering(MediumAccess medium_access, SpectralQuery spect) {
  MediumAccessGPUContext access_context = make_medium_access_gpu_context(constants.scene.mediums, constants.scene.images, constants.scene.spectrums);
  return medium_access_load_scattering_spectral(access_context, medium_access, spect);
}

SpectralResponse gpu_medium_absorption(MediumAccess medium_access, SpectralQuery spect) {
  MediumAccessGPUContext access_context = make_medium_access_gpu_context(constants.scene.mediums, constants.scene.images, constants.scene.spectrums);
  return medium_access_load_absorption_spectral(access_context, medium_access, spect);
}

float gpu_medium_phase_function(MediumAccess medium_access, float3 incoming_direction, float3 outgoing_direction) {
  return medium_phase_shared_henyey_greenstein(incoming_direction, outgoing_direction, medium_access.phase_function_g);
}

float3 gpu_medium_sample_phase_function(MediumAccess medium_access, float2 sample_random, float3 incoming_direction) {
  return medium_phase_shared_sample_henyey_greenstein(incoming_direction, medium_access.phase_function_g, sample_random);
}

MediumSample gpu_sample_medium(MediumAccess medium_access, SpectralQuery spect, SpectralResponse throughput, float3 pos, float3 incoming_direction, float max_t, inout uint seed) {
  SpectralResponse scattering_value = gpu_medium_scattering(medium_access, spect);
  SpectralResponse absorption_value = gpu_medium_absorption(medium_access, spect);
  return sample_medium_gpu(medium_access, spect, throughput, scattering_value, absorption_value, pos, incoming_direction, max_t, seed);
}

bool camera_lens_sampling_enabled(float lens_radius, float focal_distance) {
  return (lens_radius > kEpsilon) && (focal_distance > kEpsilon);
}

bool gpu_random_continue(uint path_length, uint start_path_length, float eta_scale, inout uint seed, inout SpectralResponse throughput) {
  float max_throughput = spectral_response_maximum(throughput);
  if (max_throughput == 0.0f) {
    return false;
  }

  if (path_length < start_path_length) {
    return true;
  }

  float continuation = max_throughput * eta_scale * eta_scale;
  if (isfinite(continuation) == false) {
    return false;
  }

  float probability = clamp(continuation, 0.01f, 1.0f);
  if (rnd01(seed) > probability) {
    return false;
  }

  throughput = spectral_response_mul(throughput, 1.0f / probability);
  return true;
}

bool gpu_path_tracing_neutral_eta(float eta) {
  const float eta_delta = abs(eta - 1.0f);
  return eta_delta <= (16.0f * kEpsilon);
}

float2 camera_primary_uv(uint2 pixel, uint2 film_size) {
  return camera_shared_flip_y(camera_shared_center_uv(pixel, film_size));
}

float2 camera_sample_film_uv(uint2 pixel, uint2 film_size, float2 uv_sample) {
  float2 uv = camera_primary_uv(pixel, film_size);
  if (constants.sample_index == 0u) {
    return uv;
  }

  SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  float2 jitter = uv_sample * 2.0f - 1.0f;
  if (scene_globals_data.pixel_filter_image_index != kInvalidIndex) {
    ImageSampleGPUContext sample_context = make_image_sample_gpu_context(constants.scene.images);
    ImageSampleAccess image_sample = image_sample_access_default(uv_sample);
    if (image_sample_try_sample(sample_context, scene_globals_data.pixel_filter_image_index, uv_sample, image_sample)) {
      jitter = image_sample.uv * 2.0f - 1.0f;
    }
  }

  float2 filtered_uv = float2((float(pixel.x) + 0.5f + scene_globals_data.pixel_filter_radius * jitter.x) / float(film_size.x) * 2.0f - 1.0f,
    (float(pixel.y) + 0.5f + scene_globals_data.pixel_filter_radius * jitter.y) / float(film_size.y) * 2.0f - 1.0f);
  return camera_shared_flip_y(filtered_uv);
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

  ImageSampleGPUContext sample_context = make_image_sample_gpu_context(constants.scene.images);
  ImageSampleAccess image_sample = image_sample_access_default(sensor_sample_rnd);
  if (image_sample_try_sample(sample_context, camera.lens_image, sensor_sample_rnd, image_sample) == false) {
    return sensor_sample_rnd * 2.0f - 1.0f;
  }

  return image_sample.uv * 2.0f - 1.0f;
}

float3 camera_lens_point(Camera camera, float2 sensor_sample_rnd) {
  float2 sensor_sample = camera_sample_lens_uv(camera, sensor_sample_rnd) * camera.lens_radius;
  return camera_film_shared_lens_point(camera, sensor_sample);
}

Ray camera_generate_primary_ray(Camera camera, float2 uv, float2 sensor_sample_rnd) {
  float2 sensor_sample = camera_sample_lens_uv(camera, sensor_sample_rnd);
  return camera_generate_ray(camera, uv, sensor_sample);
}

SpectralResponse load_scene_spectrum_or_zero(uint spectrum_index, SpectralQuery spect) {
  if (scene_gpu_can_sample_spectrum(constants.scene.spectrums, spectrum_index) == false) {
    return spectral_response_zero(spect);
  }

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
  SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, constants.scene.spectrums);
  return spectrum_access_evaluate(spectrum_context, spectrum_index, spect);
}

SpectralImage make_spectral_image(uint spectrum_index, uint image_index) {
  SpectralImage result = ETX_ZERO(SpectralImage);
  result.spectrum_index = spectrum_index;
  result.image_index = image_index;
  return result;
}

SpectralResponse apply_image(SpectralQuery spect, SpectralImage img, float2 uv, out float image_pdf) {
  image_pdf = 0.0f;

  SpectralResponse result = load_scene_spectrum_or_zero(img.spectrum_index, spect);
  if (img.image_index == kInvalidIndex) {
    return result;
  }

  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(constants.scene.images);
  float4 image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  if (image_evaluate_try_rgba(image_context, img.image_index, uv, image_pdf, image_value) == false) {
    return result;
  }

  return spectral_response_apply_rgb_scale(spect, result, image_value.xyz);
}

SpectralResponse apply_image(SpectralQuery spect, SpectralImage img, float2 uv) {
  float image_pdf = 0.0f;
  return apply_image(spect, img, uv, image_pdf);
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

struct TraceSurfaceResult {
  uint hit;
  uint medium_index;
  uint triangle_index;
  uint emitter_index;
  float hit_t;
  TriangleData tri;
  SurfacePoint surface_point;
  Material material;
  SpectralResponse transmittance;
};

SurfacePoint load_surface_point(ByteAddressBuffer position_buffer, ByteAddressBuffer normal_buffer, ByteAddressBuffer tangent_buffer, ByteAddressBuffer bitangent_buffer,
  ByteAddressBuffer texcoord_buffer, bool has_surface_frame, bool has_texcoords, TriangleData tri, float2 bary, float3 ray_dir) {
  SurfacePoint result;
  result.barycentrics = barycentrics(bary);

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

  surface_point_shared_interpolate_vertex(p0, p1, p2, n0, n1, n2, tangent_0, tangent_1, tangent_2, bitangent_0, bitangent_1, bitangent_2, texcoord_0, texcoord_1, texcoord_2,
    result.barycentrics, has_surface_frame, has_texcoords, result.vertex);
  result.vertex.nrm = scene_math_shared_orient_normals_to_hemisphere(result.vertex.nrm, tri.geo_n, ray_dir);

  result.geo_normal = surface_point_shared_orient_geo_normal(tri.geo_n, ray_dir);

  return result;
}

float2 interpolate_uv_from_barycentrics(ByteAddressBuffer texcoord_buffer, TriangleData tri, float3 bc) {
  float2 t0 = load_float2(texcoord_buffer, tri.i.x);
  float2 t1 = load_float2(texcoord_buffer, tri.i.y);
  float2 t2 = load_float2(texcoord_buffer, tri.i.z);
  return t0 * bc.x + t1 * bc.y + t2 * bc.z;
}

float2 interpolate_uv(ByteAddressBuffer texcoord_buffer, TriangleData tri, float2 bary) {
  return interpolate_uv_from_barycentrics(texcoord_buffer, tri, barycentrics(bary));
}

bool alpha_test_pass(uint material_index, float2 uv, inout uint seed) {
  if (scene_gpu_has_descriptor(constants.scene.materials) == false) {
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

#include <access/emitter_access_gpu.hxx>

EmitterAccessGPUContext make_scene_emitter_access_gpu_context() {
  return make_emitter_access_gpu_context(constants.scene.emitter_instances, constants.scene.emitter_profiles, constants.scene.spectrums, constants.scene.images,
    constants.scene.scene_globals);
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

[noinline] bool try_load_emitter_instance(uint emitter_index, out GPUEmitterInstanceABIData emitter_instance) {
  emitter_instance = (GPUEmitterInstanceABIData)0;
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  uint emitter_instance_count = 0u;
  uint emitter_profile_count = 0u;
  if (emitter_access_try_load_scene_state(context, emitter_instance_count, emitter_profile_count) == false) {
    return false;
  }
  (void)emitter_profile_count;
  if (emitter_index >= emitter_instance_count) {
    return false;
  }

  ByteAddressBuffer emitter_instance_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_instances)];
  emitter_instance = gpu_abi_load_emitter_instance(emitter_instance_buffer, emitter_index);
  return true;
}

  [noinline] bool try_load_emitter_profile(uint emitter_profile_index, out GPUEmitterProfileABIData emitter_profile) {
  emitter_profile = (GPUEmitterProfileABIData)0;
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  uint emitter_instance_count = 0u;
  uint emitter_profile_count = 0u;
  if (emitter_access_try_load_scene_state(context, emitter_instance_count, emitter_profile_count) == false) {
    return false;
  }
  (void)emitter_instance_count;
  if (emitter_profile_index >= emitter_profile_count) {
    return false;
  }

  ByteAddressBuffer emitter_profile_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitter_profiles)];
  emitter_profile = gpu_abi_load_emitter_profile(emitter_profile_buffer, emitter_profile_index);
  return true;
}

bool try_load_distribution_entry(uint entry_index, out DistributionEntry entry) {
  entry = (DistributionEntry)0;
  if (constants.scene.emitters_distribution == kInvalidIndex) {
    return false;
  }

  ByteAddressBuffer distribution_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.emitters_distribution)];
  uint base_offset = entry_index * kDistributionEntryStride;
  entry.value = asfloat(distribution_buffer.Load(base_offset + 0u));
  entry.pdf = asfloat(distribution_buffer.Load(base_offset + 4u));
  entry.cdf = asfloat(distribution_buffer.Load(base_offset + 8u));
  entry.reference = distribution_buffer.Load(base_offset + 12u);
  return true;
}

[noinline] uint emitter_distribution_entry_count() {
  if (constants.scene.emitters_distribution == kInvalidIndex) {
    return 0u;
  }

  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(scene_globals);
  return scene_globals_data.active_emitter_count;
}

bool emitter_distribution_has_values() {
  return emitter_distribution_entry_count() > 0u;
}

[noinline] uint sample_emitter_distribution(inout uint seed, out float pdf_sample) {
  pdf_sample = 0.0f;
  uint entry_count = emitter_distribution_entry_count();
  if (entry_count == 0u) {
    return kInvalidIndex;
  }

  float rnd = rnd01(seed);
  DistributionSearchRange search = distribution_search_begin(entry_count);
  while (distribution_search_active(search)) {
    uint middle = distribution_search_middle(search);
    DistributionEntry middle_entry = (DistributionEntry)0;
    try_load_distribution_entry(middle, middle_entry);
    distribution_search_update(search, middle, middle_entry.cdf, rnd);
  }

  DistributionEntry selected_entry = (DistributionEntry)0;
  if (try_load_distribution_entry(search.begin, selected_entry) == false) {
    return kInvalidIndex;
  }

  pdf_sample = selected_entry.pdf;
  return selected_entry.reference;
}

  [noinline] float emitter_discrete_pdf(uint emitter_index) {
  uint entry_count = emitter_distribution_entry_count();
  for (uint entry_index = 0u; entry_index < entry_count; ++entry_index) {
    DistributionEntry entry = (DistributionEntry)0;
    if (try_load_distribution_entry(entry_index, entry) == false) {
      return 0.0f;
    }
    if (entry.reference == emitter_index) {
      return entry.pdf;
    }
  }

  return 0.0f;
}

float3 evaluate_emission_integrated_source(uint emission_spectrum_index, uint emission_image_index, float2 uv) {
  SpectralQuery integrated_query = spectral_query_sample();
  SpectralImage emission = make_spectral_image(emission_spectrum_index, emission_image_index);
  return apply_image(integrated_query, emission, uv).integrated;
}

SpectralResponse evaluate_emission_spectral_source(uint emission_spectrum_index, uint emission_image_index, float2 uv, SpectralQuery spect) {
  SpectralImage emission = make_spectral_image(emission_spectrum_index, emission_image_index);
  return apply_image(spect, emission, uv);
}

float3 evaluate_local_emission_integrated(uint emitter_index, float2 uv) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  ETX_ZERO_INIT(EmitterAccess, access);
  if (emitter_access_try_load_local(context, emitter_index, access) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  return evaluate_emission_integrated_source(access.emission_spectrum_index, access.emission_image_index, uv);
}

SpectralResponse evaluate_local_emission_spectral(uint emitter_index, float2 uv, SpectralQuery spect) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  ETX_ZERO_INIT(EmitterAccess, access);
  if (emitter_access_try_load_local(context, emitter_index, access) == false) {
    return zero_value;
  }

  return evaluate_emission_spectral_source(access.emission_spectrum_index, access.emission_image_index, uv, spect);
}

float3 evaluate_distant_emission_integrated(uint emitter_index, float3 direction) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  ETX_ZERO_INIT(EmitterAccess, access);
  if (emitter_access_try_load_distant(context, emitter_index, direction, access) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  float2 uv = emitter_access_environment_uv(context, access, direction);
  return evaluate_emission_integrated_source(access.emission_spectrum_index, access.emission_image_index, uv);
}

SpectralResponse evaluate_distant_emission_spectral(uint emitter_index, float3 direction, SpectralQuery spect) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  ETX_ZERO_INIT(EmitterAccess, access);
  if (emitter_access_try_load_distant(context, emitter_index, direction, access) == false) {
    return zero_value;
  }

  float2 uv = emitter_access_environment_uv(context, access, direction);
  return evaluate_emission_spectral_source(access.emission_spectrum_index, access.emission_image_index, uv, spect);
}

[noinline] float3 evaluate_distant_emission_integrated_all(float3 direction) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  uint emitter_instance_count = 0u;
  uint emitter_count = 0u;
  if (emitter_access_try_load_environment_state(context, emitter_instance_count, emitter_count) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }
  (void)emitter_instance_count;

  float3 result = float3(0.0f, 0.0f, 0.0f);

  for (uint i = 0u; i < emitter_count; ++i) {
    uint emitter_index = kInvalidIndex;
    if (emitter_access_try_load_environment_emitter(context, i, emitter_index)) {
      result += evaluate_distant_emission_integrated(emitter_index, direction);
    }
  }

  return result;
}

  [noinline] bool trace_surface_path(RaytracingAccelerationStructure as, RayDesc ray, SpectralQuery spect, inout uint medium_index, inout uint seed,
    out TraceSurfaceResult result) {
  result.hit = 0u;
  result.medium_index = medium_index;
  result.triangle_index = kInvalidIndex;
  result.emitter_index = kInvalidIndex;
  result.hit_t = ray.TMax;
  result.tri = (TriangleData)0;
  result.surface_point = (SurfacePoint)0;
  result.material = (Material)0;
  result.transmittance = spectral_response_make(spect, 1.0f);

  const bool has_geometry_buffers = (constants.scene.triangles != kInvalidIndex) && (constants.scene.vertex_positions != kInvalidIndex) &&
                                    (constants.scene.vertex_normals != kInvalidIndex) && (constants.scene.scene_globals != kInvalidIndex);
  if (has_geometry_buffers == false) {
    return false;
  }

  ByteAddressBuffer triangle_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)];
  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(scene_globals);
  const bool has_material_buffer = constants.scene.materials != kInvalidIndex;
  const bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;
  uint vertex_count = scene_globals_data.vertex_count;
  uint triangle_count = scene_globals_data.triangle_count;

  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> q;
  q.TraceRayInline(as, RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);

  float medium_segment_start_t = ray.TMin;
  uint ray_medium_index = medium_index;
  while (q.Proceed()) {
    if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
      continue;
    }

    uint candidate_triangle_index = q.CandidatePrimitiveIndex();
    if (candidate_triangle_index >= triangle_count) {
      continue;
    }

    float candidate_t = q.CandidateTriangleRayT();
    if (candidate_t > medium_segment_start_t) {
      float segment_distance = candidate_t - medium_segment_start_t;
      float3 segment_origin = ray.Origin + ray.Direction * medium_segment_start_t;
      SpectralResponse segment_transmittance = medium_segment_transmittance_spectral(ray_medium_index, segment_origin, ray.Direction, segment_distance, spect, seed);
      result.transmittance = spectral_response_mul(result.transmittance, segment_transmittance);
      medium_segment_start_t = candidate_t;
    }

    TriangleData tri = load_triangle(triangle_buffer, candidate_triangle_index);
    bool valid_indices = (tri.i.x < vertex_count) && (tri.i.y < vertex_count) && (tri.i.z < vertex_count);
    if (valid_indices == false) {
      continue;
    }

    float2 candidate_bary = q.CandidateTriangleBarycentrics();
    float2 candidate_uv = float2(0.0f, 0.0f);
    if (has_texcoords) {
      ByteAddressBuffer texcoord_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)];
      candidate_uv = interpolate_uv(texcoord_buffer, tri, candidate_bary);
    }

    MaterialAccess material_access = ETX_ZERO(MaterialAccess);
    if (has_material_buffer) {
      MaterialAccessGPUContext material_context = {constants.scene.materials};
      material_access_try_load(material_context, tri.material_index, material_access);
    }

    bool alpha_rejected = alpha_test_pass(tri.material_index, candidate_uv, seed);
    bool entering_surface = dot(tri.geo_n, ray.Direction) < 0.0f;
    HitPolicyDecision hit_policy = hit_policy_evaluate(HitPolicyMode::SkipBoundaryWithMediumTransition, material_access.material_class, alpha_rejected, entering_surface,
      material_access.int_medium_index, material_access.ext_medium_index);
    if (hit_policy.action == HitPolicyAction::Ignore) {
      continue;
    }

    if (hit_policy.action == HitPolicyAction::TransitionMedium) {
      ray_medium_index = hit_policy.medium_index;
      continue;
    }

    if (hit_policy.action == HitPolicyAction::CommitSurface) {
      q.CommitNonOpaqueTriangleHit();
    }
  }

  float medium_segment_end_t = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? q.CommittedRayT() : ray.TMax;
  if (medium_segment_end_t > medium_segment_start_t) {
    float segment_distance = medium_segment_end_t - medium_segment_start_t;
    float3 segment_origin = ray.Origin + ray.Direction * medium_segment_start_t;
    SpectralResponse segment_transmittance = medium_segment_transmittance_spectral(ray_medium_index, segment_origin, ray.Direction, segment_distance, spect, seed);
    result.transmittance = spectral_response_mul(result.transmittance, segment_transmittance);
  }

  medium_index = ray_medium_index;
  result.medium_index = ray_medium_index;

  if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
    return false;
  }

  const bool has_surface_frame_buffers = (constants.scene.vertex_tangents != kInvalidIndex) && (constants.scene.vertex_bitangents != kInvalidIndex);
  uint tangent_buffer_index = constants.scene.vertex_positions;
  uint bitangent_buffer_index = constants.scene.vertex_positions;
  uint texcoord_buffer_index = constants.scene.vertex_positions;
  if (has_surface_frame_buffers) {
    tangent_buffer_index = constants.scene.vertex_tangents;
    bitangent_buffer_index = constants.scene.vertex_bitangents;
  }
  if (has_texcoords) {
    texcoord_buffer_index = constants.scene.vertex_texcoords;
  }

  ByteAddressBuffer position_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)];
  ByteAddressBuffer normal_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)];
  ByteAddressBuffer tangent_buffer = bindless_buffers[NonUniformResourceIndex(tangent_buffer_index)];
  ByteAddressBuffer bitangent_buffer = bindless_buffers[NonUniformResourceIndex(bitangent_buffer_index)];
  ByteAddressBuffer texcoord_buffer = bindless_buffers[NonUniformResourceIndex(texcoord_buffer_index)];

  result.triangle_index = q.CommittedPrimitiveIndex();
  result.hit_t = q.CommittedRayT();
  result.tri = load_triangle(triangle_buffer, result.triangle_index);
  float2 bary = q.CommittedTriangleBarycentrics();
  result.surface_point = load_surface_point(position_buffer, normal_buffer, tangent_buffer, bitangent_buffer, texcoord_buffer, has_surface_frame_buffers, has_texcoords, result.tri,
    bary, ray.Direction);
  result.emitter_index = result.tri.emitter_index;
  try_load_material_full(result.tri.material_index, result.material);
  result.hit = 1u;
  return true;
}

#if 0

[noinline] bool trace_transmittance_to_point(
  RaytracingAccelerationStructure as, float3 origin, float3 target, SpectralQuery spect, uint medium_index, inout uint seed, out SpectralResponse transmittance) {
  transmittance = spectral_response_make(spect, 1.0f);
  float3 delta = target - origin;
  float distance = length(delta);
  if (distance <= kRayEpsilon) {
    return true;
  }

  RayDesc ray = (RayDesc)0;
  ray.Origin = origin;
  ray.Direction = delta / distance;
  ray.TMin = kRayEpsilon;
  ray.TMax = max(ray.TMin, distance - kRayEpsilon);

  TraceSurfaceResult trace_result = (TraceSurfaceResult)0;
  uint transmittance_medium = medium_index;
  bool found_surface = trace_surface_path(as, ray, spect, transmittance_medium, seed, trace_result);
  transmittance = trace_result.transmittance;
  return found_surface == false;
}

[noinline] float3 surface_shading_position(TraceSurfaceResult surface_hit, float3 outgoing_direction) {
  ByteAddressBuffer position_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)];
  ByteAddressBuffer normal_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)];
  float3 p0 = load_float3(position_buffer, surface_hit.tri.i.x);
  float3 p1 = load_float3(position_buffer, surface_hit.tri.i.y);
  float3 p2 = load_float3(position_buffer, surface_hit.tri.i.z);
  float3 n0 = load_float3(normal_buffer, surface_hit.tri.i.x);
  float3 n1 = load_float3(normal_buffer, surface_hit.tri.i.y);
  float3 n2 = load_float3(normal_buffer, surface_hit.tri.i.z);
  return scene_math_shared_shading_pos(
    p0, p1, p2, n0, n1, n2, surface_hit.tri.geo_n, surface_hit.surface_point.barycentrics, outgoing_direction);
}

[noinline] BSDFEval gpu_evaluate_material_bsdf(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  return bsdf_evaluate(context, data, outgoing_direction, material, sampler);
}

[noinline] SpectralResponse evaluate_direct_light(
  RaytracingAccelerationStructure as, TraceSurfaceResult surface_hit, float3 incoming_direction, uint medium_index, SpectralQuery spect, GPUEmitterSample emitter_sample,
  inout uint seed) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  if (emitter_sample.pdf_dir <= 0.0f) {
    return zero_value;
  }

  BSDFResourceContext bsdf_context = make_scene_bsdf_resource_gpu_context();
  BSDFData bsdf_data = make_surface_bsdf_data(surface_hit.surface_point.vertex, spect, medium_index, incoming_direction);
  Sampler bsdf_sampler = make_bsdf_sampler(seed);
  BSDFEval bsdf_eval = gpu_evaluate_material_bsdf(bsdf_context, bsdf_data, emitter_sample.direction, surface_hit.material, bsdf_sampler);
  seed = bsdf_sampler.seed;
  if (bsdf_eval_valid(bsdf_eval) == false) {
    return zero_value;
  }

  SpectralResponse transmittance = spectral_response_make(spect, 1.0f);
  float3 shadow_origin = surface_shading_position(surface_hit, emitter_sample.direction);
  if (trace_transmittance_to_point(as, shadow_origin, emitter_sample.origin, spect, medium_index, seed, transmittance) == false) {
    return zero_value;
  }

  bool no_weight = (scene_multiple_importance_sampling_enabled() == false) || (emitter_sample.is_delta != 0u);
  float weight = no_weight ? 1.0f : power_heuristic(emitter_sample.pdf_dir * emitter_sample.pdf_sample, bsdf_eval.pdf);
  float scale = weight / max(kEpsilon, emitter_sample.pdf_dir * emitter_sample.pdf_sample);
  return spectral_response_mul(spectral_response_mul(spectral_response_mul(bsdf_eval.bsdf, emitter_sample.value), transmittance), scale);
}

#endif

[noinline] SpectralResponse gpu_evaluate_local_emission_spectral(uint emitter_index, float2 uv, SpectralQuery spect) {
  return evaluate_local_emission_spectral(emitter_index, uv, spect);
}

  [noinline] SpectralResponse gpu_evaluate_distant_emission_spectral_all(float3 direction, SpectralQuery spect) {
  SpectralResponse result = spectral_response_zero(spect);
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  uint emitter_instance_count = 0u;
  uint emitter_count = 0u;
  if (emitter_access_try_load_environment_state(context, emitter_instance_count, emitter_count) == false) {
    return result;
  }
  (void)emitter_instance_count;

  for (uint i = 0u; i < emitter_count; ++i) {
    uint emitter_index = kInvalidIndex;
    if (emitter_access_try_load_environment_emitter(context, i, emitter_index)) {
      result = spectral_response_add(result, evaluate_distant_emission_spectral(emitter_index, direction, spect));
    }
  }

  return result;
}

bool gpu_bsdf_sample_supported_class(uint material_class) {
  switch (material_class) {
    case MaterialClass::Diffuse:
    case MaterialClass::Translucent:
    case MaterialClass::Plastic:
    case MaterialClass::Conductor:
    case MaterialClass::Dielectric:
    case MaterialClass::Thinfilm:
    case MaterialClass::Mirror:
    case MaterialClass::Boundary:
    case MaterialClass::Velvet:
    case MaterialClass::Void:
    case MaterialClass::DiffractionGrating:
      return true;

    // TODO(OpenPBR GPU parity): OpenPBR is deliberately unsupported on the production GPU path for now.
    // Do not re-enable this until sample/evaluate/pdf/reverse-pdf/albedo compile and pass CPU parity tests.
    case MaterialClass::OpenPBR:
    default: {
      return false;
    }
  }
}

bool gpu_valid_direction(float3 direction) {
  float direction_length_sq = dot(direction, direction);
  bool finite_components = all(isfinite(direction));
  return finite_components && (direction_length_sq > 0.25f) && (direction_length_sq < 4.0f);
}

bool gpu_valid_spectral_response(SpectralResponse value) {
  float3 rgb = spectral_response_to_rgb(value);
  return all(isfinite(rgb));
}

[noinline] BSDFSample gpu_diffuse_bsdf_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  return bsdf_diffuse_sample(context, data, material, sampler);
}

[noinline] BSDFSample gpu_plastic_bsdf_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  return bsdf_plastic_sample(context, data, material, sampler);
}

[noinline] BSDFSample gpu_thinfilm_bsdf_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  return bsdf_thinfilm_sample(context, data, material, sampler);
}

[noinline] BSDFSample gpu_sample_material_bsdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  if (material.cls == MaterialClass::Diffuse) {
    return bsdf_sample(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::Translucent) {
    return bsdf_sample(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::Plastic) {
    return gpu_plastic_bsdf_sample(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::Conductor) {
    return bsdf_sample(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::Dielectric) {
    return bsdf_sample(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::Thinfilm) {
    return gpu_thinfilm_bsdf_sample(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::Mirror) {
    return bsdf_sample(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::Boundary) {
    return bsdf_sample(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::Velvet) {
    return bsdf_sample(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::OpenPBR) {
    // TODO(OpenPBR GPU parity): keep OpenPBR off the production GPU sampler until the shader size and parity blockers are fixed.
    return bsdf_sample_zero(data.spectrum_sample);
  }
  if (material.cls == MaterialClass::Void) {
    return bsdf_sample(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::DiffractionGrating) {
    return bsdf_sample(context, data, material, sampler);
  }

  return bsdf_sample_zero(data.spectrum_sample);
}
