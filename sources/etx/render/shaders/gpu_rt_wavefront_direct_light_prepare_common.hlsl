#pragma once

#include "gpu_rt_wavefront_storage.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>
#include <access/material_access_gpu.hxx>
#include <interop/bsdf_core_shared.hxx>
#include <interop/bsdf_resource_shared.hxx>
#include <interop/material.hxx>
#include <interop/scene_gpu_access_shared.hxx>

struct WavefrontDirectLightPathState {
  SpectralQuery spect;
  float2 film_uv;
  float last_emitter_pdf;
  uint sampler_seed;
};

struct WavefrontDirectLightPrepareInput {
  GPUWavefrontResources resources;
  uint task_index;
  uint path_index;
  uint camera_path_length;
  WavefrontDirectLightPathState state;
  GPUWavefrontHit hit;
  GPUWavefrontPathVertex current_vertex;
#if ETX_WAVEFRONT_PATH_TRACING_ONLY == 0
  GPUWavefrontPathVertex previous_vertex;
#endif
  GPUWavefrontDirectLightSample sample_value;
  Material material;
};

WavefrontDirectLightPathState wavefront_load_direct_light_path_state(uint descriptor_index, uint path_index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = path_index * kGPUWavefrontPathStateStride;
  WavefrontDirectLightPathState result = (WavefrontDirectLightPathState)0;
  result.spect.wavelength = asfloat(buffer.Load(base_offset + kGPUWavefrontPathStateSpectralQueryOffset + 0u));
  result.spect.flags = buffer.Load(base_offset + kGPUWavefrontPathStateSpectralQueryOffset + 4u);
  result.film_uv = wavefront_load_float2(buffer, base_offset + kGPUWavefrontPathStateFilmUvOffset);
  result.last_emitter_pdf = asfloat(buffer.Load(base_offset + kGPUWavefrontPathStateLastEmitterPdfOffset));
  result.sampler_seed = buffer.Load(base_offset + kGPUWavefrontPathStateSamplerSeedOffset);
  return result;
}

uint wavefront_load_camera_path_length(uint descriptor_index, uint path_index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  return buffer.Load(path_index * kGPUWavefrontPathMetaStride + kGPUWavefrontPathMetaCameraPathLengthOffset);
}

BSDFResourceContext wavefront_make_scene_bsdf_resource_gpu_context() {
  return make_bsdf_resource_gpu_context(constants.scene.images, constants.scene.spectrums, constants.scene.energy_compensation_interfaces, constants.scene.scene_globals);
}

bool wavefront_try_load_material_full(uint material_index, out Material material) {
  MaterialAccessGPUContext material_context = {constants.scene.materials};
  return material_access_try_load_full(material_context, material_index, material);
}

BSDFData wavefront_make_surface_bsdf_data(Vertex vertex, SpectralQuery spect, uint medium_index, float3 incoming_direction) {
  (void)medium_index;
  return bsdf_data_make(vertex, spect, kInvalidIndex, PathSource::Camera, incoming_direction);
}

Sampler wavefront_make_bsdf_sampler(uint seed) {
  Sampler result = ETX_ZERO(Sampler);
  result.seed = seed;
  return result;
}

bool wavefront_scene_multiple_importance_sampling_enabled() {
  SceneGPUSharedOptions options = scene_gpu_load_options(constants.scene.scene_options);
  return (options.properties_flags & (1u << SceneProperty::MultipleImportanceSampling)) != 0u;
}

bool wavefront_valid_spectral_response(SpectralResponse value) {
  return isfinite(value.value) && all(isfinite(value.integrated));
}

float wavefront_direct_light_sampling_pdf(GPUWavefrontDirectLightSample sample_value) {
  return sample_value.pdf_dir * sample_value.pdf_sample;
}

float wavefront_direct_light_emitter_sample_pdf(GPUWavefrontDirectLightSample sample_value) {
  if ((sample_value.flags & GPUWavefrontDirectLightSampleFlags::Distant) != 0u) {
    float directional_pdf = ((sample_value.flags & GPUWavefrontDirectLightSampleFlags::Delta) != 0u) ? 1.0f : sample_value.pdf_dir;
    return sample_value.pdf_sample * directional_pdf;
  }

  return sample_value.pdf_sample * sample_value.pdf_area;
}

float wavefront_direct_light_from_emitter_pdf(GPUWavefrontDirectLightSample sample_value, GPUWavefrontPathVertex current_vertex) {
  if ((sample_value.flags & GPUWavefrontDirectLightSampleFlags::Distant) != 0u) {
    float3 w_o = normalize(sample_value.origin - current_vertex.position);
    float cosine_term = wavefront_path_vertex_is_surface(current_vertex) ? abs(dot(current_vertex.geo_normal, w_o)) : 1.0f;
    return sample_value.pdf_area * cosine_term;
  }

  TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], sample_value.triangle_index);
  Material emitter_material = (Material)0;
  if (wavefront_try_load_material_full(tri.material_index, emitter_material) == false) {
    return 0.0f;
  }

  float3 w_o = current_vertex.position - sample_value.origin;
  float distance_squared = dot(w_o, w_o);
  if (distance_squared <= kEpsilon) {
    return 0.0f;
  }

  w_o *= rsqrt(distance_squared);
  float exponent = scene_math_shared_collimation_to_exponent(emitter_material.emission_collimation);
  float pdf_dir = pow(max(0.0f, dot(sample_value.normal, w_o)), exponent) * kInvPi;
  return wavefront_convert_solid_angle_pdf_to_area(pdf_dir, sample_value.origin, current_vertex.position, wavefront_path_vertex_is_surface(current_vertex), current_vertex.normal);
}

float wavefront_direct_light_mis_camera(uint camera_path_length, GPUWavefrontPathVertex current_vertex, GPUWavefrontPathVertex previous_vertex, float current_backward_pdf,
  float previous_backward_pdf) {
  float result_accumulated = 0.0f;
  float previous_connectible = wavefront_path_vertex_connectible(previous_vertex) ? 1.0f : 0.0f;
  if (camera_path_length > 1u) {
    float r1 = wavefront_safe_div(previous_backward_pdf, previous_vertex.pdf_from_prev);
    float previous_mis_connectible = wavefront_path_vertex_mis_connectible(previous_vertex) ? 1.0f : 0.0f;
    result_accumulated = r1 * (previous_mis_connectible + previous_vertex.pdf_history);
  }
  float r0 = wavefront_safe_div(current_backward_pdf, current_vertex.pdf_from_prev);
  result_accumulated = r0 * (previous_connectible + result_accumulated);
  return result_accumulated;
}

float wavefront_direct_light_weight(WavefrontDirectLightPrepareInput input_value, ETX_IN(BSDFEval, bsdf_eval), inout Sampler sampler) {
  if (wavefront_scene_multiple_importance_sampling_enabled() == false) {
    return 1.0f;
  }

  float sampling_pdf = wavefront_direct_light_sampling_pdf(input_value.sample_value);
  bool sampled_light_is_delta = (input_value.sample_value.flags & GPUWavefrontDirectLightSampleFlags::Delta) != 0u;
#if ETX_WAVEFRONT_PATH_TRACING_ONLY
  float direct_pdf = sampled_light_is_delta ? 0.0f : bsdf_eval.pdf;
  return power_heuristic(sampling_pdf, direct_pdf);
#else
  if (scene_path_mode_is_path_tracing()) {
    float direct_pdf = sampled_light_is_delta ? 0.0f : bsdf_eval.pdf;
    return power_heuristic(sampling_pdf, direct_pdf);
  }

  float p_sample = wavefront_direct_light_emitter_sample_pdf(input_value.sample_value);
  float p_fwd = input_value.previous_vertex.pdf_from_prev * input_value.current_vertex.pdf_from_prev;
  float p_connection = p_fwd * p_sample;
  bool sampled_light_is_surface = (input_value.sample_value.flags & GPUWavefrontDirectLightSampleFlags::Distant) == 0u;
  float p_bsdf_sample = sampled_light_is_surface ? wavefront_convert_solid_angle_pdf_to_area(bsdf_eval.pdf, input_value.current_vertex.position, input_value.sample_value.origin,
                                                     true, input_value.sample_value.normal)
                                                 : bsdf_eval.pdf;
  float p_direct = 0.0f;
  if (sampled_light_is_delta == false) {
    p_direct = p_fwd * p_bsdf_sample;
  }

  BSDFData reverse_data =
    wavefront_make_surface_bsdf_data(input_value.hit.vertex, input_value.state.spect, input_value.current_vertex.medium_index, -input_value.sample_value.direction);
  reverse_data.path_source = PathSource::Light;
  float3 previous_direction = normalize(input_value.previous_vertex.position - input_value.current_vertex.position);
  float reverse_pdf = wavefront_direct_light_stage_bsdf_pdf(wavefront_make_scene_bsdf_resource_gpu_context(), reverse_data, previous_direction, input_value.material, sampler);
  if (scene_path_mode_is_vcm()) {
    float w_light = sampled_light_is_delta ? 0.0f : wavefront_safe_div(bsdf_eval.pdf, sampling_pdf);
    float camera_factor = wavefront_path_vertex_is_surface(input_value.current_vertex) ? abs(dot(input_value.sample_value.direction, input_value.current_vertex.geo_normal)) : 1.0f;
    float emitter_cosine = abs(dot(input_value.sample_value.direction, input_value.sample_value.normal));
    float density_ratio = wavefront_safe_div(input_value.sample_value.pdf_dir * emitter_cosine, input_value.sample_value.pdf_dir_out * camera_factor);
    float w_camera = (density_ratio > 0.0f)
                       ? wavefront_safe_div(constants.vcm_vm_weight + input_value.current_vertex.forward_pdf + input_value.current_vertex.reverse_pdf * reverse_pdf, density_ratio)
                       : 0.0f;
    return 1.0f / (1.0f + w_light + w_camera);
  }
  float z_prev_backward_pdf = wavefront_convert_solid_angle_pdf_to_area(reverse_pdf, input_value.current_vertex.position, input_value.previous_vertex.position,
    wavefront_path_vertex_is_surface(input_value.previous_vertex), input_value.previous_vertex.normal);
  float p_bck = input_value.previous_vertex.pdf_history * ((input_value.camera_path_length > 1u) ? z_prev_backward_pdf : 1.0f);
  float from_emitter = wavefront_direct_light_from_emitter_pdf(input_value.sample_value, input_value.current_vertex);
  if (scene_path_mode_is_bdpt_full()) {
    float w_camera = wavefront_direct_light_mis_camera(input_value.camera_path_length, input_value.current_vertex, input_value.previous_vertex, from_emitter, z_prev_backward_pdf);
    float w_light = sampled_light_is_delta ? 0.0f : wavefront_safe_div(p_bsdf_sample, p_sample);
    return 1.0f / (1.0f + w_camera + w_light);
  }

  float p_light_path = p_sample * from_emitter * p_bck;

  return balance_heuristic(p_connection, p_direct, p_light_path);
#endif
}

bool wavefront_load_direct_light_prepare_input(uint dispatch_index, out WavefrontDirectLightPrepareInput input_value) {
  input_value = (WavefrontDirectLightPrepareInput)0;
  input_value.resources = wavefront_load_resources();
  if ((input_value.resources.direct_light_sample_buffer == kInvalidIndex) || (input_value.resources.direct_light_task_buffer == kInvalidIndex)) {
    return false;
  }

#if ETX_ENABLE_WORK_QUEUES
  const uint material_queue_count = wavefront_material_queue_count(input_value.resources, true, constants.work_queue_index);
  if (dispatch_index >= material_queue_count) {
    return false;
  }
  dispatch_index = wavefront_material_queue_load(input_value.resources, true, constants.work_queue_index, dispatch_index);
#endif
  input_value.task_index = dispatch_index;

  uint queue_descriptor = wavefront_queue_current_descriptor(true);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return false;
  }

  input_value.path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  input_value.state = wavefront_load_direct_light_path_state(input_value.resources.camera_state_buffer, input_value.path_index);
  input_value.camera_path_length = wavefront_load_camera_path_length(input_value.resources.path_meta_buffer, input_value.path_index);
  input_value.hit = wavefront_load_hit(input_value.resources.camera_hit_buffer, input_value.path_index);
  input_value.sample_value = wavefront_load_direct_light_sample(input_value.resources.direct_light_sample_buffer, dispatch_index);
  if (scene_path_mode_is_light_tracing()) {
    return false;
  }
  if ((wavefront_hit_valid(input_value.hit) == false) || wavefront_hit_is_miss(input_value.hit) || (input_value.camera_path_length == 0u)) {
    return false;
  }
  if ((input_value.sample_value.flags & GPUWavefrontDirectLightSampleFlags::Valid) == 0u) {
    return false;
  }
  input_value.current_vertex =
    wavefront_load_path_vertex(input_value.resources.camera_vertex_buffer, wavefront_camera_vertex_slot(input_value.path_index, input_value.camera_path_length));
  if ((wavefront_path_vertex_valid(input_value.current_vertex) == false) || (wavefront_path_vertex_connectible(input_value.current_vertex) == false)) {
    return false;
  }
#if ETX_WAVEFRONT_PATH_TRACING_ONLY == 0
  input_value.previous_vertex =
    wavefront_load_path_vertex(input_value.resources.camera_vertex_buffer, wavefront_camera_vertex_slot(input_value.path_index, input_value.camera_path_length - 1u));
  if (wavefront_path_vertex_valid(input_value.previous_vertex) == false) {
    return false;
  }
#endif
  if (wavefront_try_load_material_full(input_value.current_vertex.material_index, input_value.material) == false) {
    return false;
  }
  return true;
}

void wavefront_store_direct_light_prepare_sampler_seed(ETX_IN(WavefrontDirectLightPrepareInput, input_value), uint sampler_seed) {
  GPUWavefrontPathState state = wavefront_load_path_state(input_value.resources.camera_state_buffer, input_value.path_index);
  if (wavefront_path_state_valid(state) == false) {
    return;
  }

  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(input_value.resources.camera_state_buffer);
  uint base_offset = input_value.path_index * kGPUWavefrontPathStateStride;
  buffer.Store(base_offset + kGPUWavefrontPathStateSamplerSeedOffset, sampler_seed);
}

void wavefront_store_direct_light_prepare_task(uint dispatch_index, ETX_IN(WavefrontDirectLightPrepareInput, input_value), ETX_IN(BSDFEval, bsdf_eval),
  ETX_INOUT(Sampler, sampler)) {
  wavefront_store_direct_light_prepare_sampler_seed(input_value, sampler.seed);

  if ((bsdf_eval_valid(bsdf_eval) == false) || (wavefront_valid_spectral_response(bsdf_eval.bsdf) == false)) {
    return;
  }

  float sampling_pdf = wavefront_direct_light_sampling_pdf(input_value.sample_value);
  if (sampling_pdf <= 0.0f) {
    return;
  }

  float mis_weight = wavefront_direct_light_weight(input_value, bsdf_eval, sampler);
  SpectralResponse contribution = spectral_response_mul(spectral_response_mul(input_value.current_vertex.throughput, bsdf_eval.bsdf),
    spectral_response_mul(input_value.sample_value.value, mis_weight / sampling_pdf));
  if (wavefront_valid_spectral_response(contribution) == false) {
    return;
  }

  float3 shadow_origin = wavefront_surface_shading_position(input_value.hit, input_value.sample_value.direction);
  float3 shadow_delta = input_value.sample_value.origin - shadow_origin;
  float shadow_distance = length(shadow_delta);
  if (shadow_distance <= kRayEpsilon) {
    return;
  }

  GPUWavefrontDirectLightTask task = (GPUWavefrontDirectLightTask)0;
  task.shadow_ray.o = shadow_origin;
  task.shadow_ray.d = shadow_delta / shadow_distance;
  task.shadow_ray.min_t = kRayEpsilon;
  task.shadow_ray.max_t = shadow_distance;
  task.shadow_target = input_value.sample_value.origin;
  task.contribution = contribution;
  task.mis_weight = mis_weight;
  task.pixel_index = input_value.current_vertex.pixel_index;
  task.medium_index = ((bsdf_eval.properties & BSDFSample::MediumChanged) != 0u) ? bsdf_eval.medium_index : input_value.current_vertex.medium_index;
  task.flags = 1u;
  task.path_index = input_value.path_index;
  task.sampler_seed = sampler.seed;
  wavefront_store_direct_light_task(input_value.resources.direct_light_task_buffer, dispatch_index, task);
  wavefront_shadow_queue_append(input_value.resources, kGPUWavefrontShadowQueueDirectLight, dispatch_index);
}
