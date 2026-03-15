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
#include <interop/projection.hxx>
#include <interop/camera_shared.hxx>
#include <interop/camera_film_shared.hxx>
#include <interop/scene_gpu_access_shared.hxx>
#include <access/image_access_gpu.hxx>
#include <access/image_evaluate_gpu.hxx>
#include <access/image_sample_gpu.hxx>
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

Camera load_camera(ByteAddressBuffer camera_buffer) {
  return gpu_abi_load_camera(camera_buffer);
}

uint load_scene_options_samples() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return options.samples;
}

uint load_scene_options_properties_flags() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return options.properties_flags;
}

bool scene_uses_spectral_mode() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return scene_gpu_uses_spectral_mode(options);
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

MediumSample sample_medium_gpu(
  MediumAccess medium_access, SpectralQuery spect, SpectralResponse throughput, SpectralResponse scattering_value, SpectralResponse absorption_value, float3 pos, float3 w_i,
  float max_t, inout uint seed) {
  MediumSharedContext context;
  context.access_context = make_medium_access_gpu_context(constants.scene.mediums, constants.scene.spectrums);
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
  return make_emitter_access_gpu_context(
    constants.scene.emitter_instances, constants.scene.emitter_profiles, constants.scene.spectrums, constants.scene.images, constants.scene.scene_globals);
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

bool try_select_environment_emitter_random(inout uint seed, out uint emitter_index, out uint emitter_count) {
  emitter_index = kInvalidIndex;
  emitter_count = 0u;

  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  uint emitter_instance_count = 0u;
  if (emitter_access_try_load_environment_state(context, emitter_instance_count, emitter_count) == false) {
    return false;
  }
  (void)emitter_instance_count;

  uint selected = uint(rnd01(seed) * float(emitter_count));
  if (selected >= emitter_count) {
    selected = emitter_count - 1u;
  }

  if (emitter_access_try_load_environment_emitter(context, selected, emitter_index) == false) {
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

float3 evaluate_distant_emission_integrated_all(float3 direction) {
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

SpectralResponse evaluate_distant_emission_spectral_all(float3 direction, SpectralQuery spect) {
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

