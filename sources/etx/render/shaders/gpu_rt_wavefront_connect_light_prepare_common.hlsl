#pragma once

#include "gpu_rt_wavefront_common.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>
#include <interop/bsdf_dispatch_shared.hxx>

struct WavefrontConnectLightPrepareInput {
  GPUWavefrontResources resources;
  uint task_index;
  uint path_index;
  uint light_vertex_length;
  GPUWavefrontPathMeta path_meta;
  GPUWavefrontPathVertex camera_vertex;
  GPUWavefrontPathVertex camera_previous_vertex;
  GPUWavefrontPathVertex light_vertex;
  GPUWavefrontPathVertex light_previous_vertex;
  Material camera_material;
  Material light_material;
};

Vertex wavefront_make_connect_vertex(float3 position, float3 normal, float2 texcoord) {
  Vertex result = (Vertex)0;
  result.pos = position;
  result.nrm = normal;
  result.tex = texcoord;
  return result;
}

float wavefront_connect_light_mis_camera(GPUWavefrontPathVertex current_vertex, GPUWavefrontPathVertex previous_vertex, float current_backward_pdf, float previous_backward_pdf) {
  float result_accumulated = 0.0f;
  float r1 = wavefront_safe_div(previous_backward_pdf, previous_vertex.pdf_from_prev);
  float previous_mis_connectible = ((previous_vertex.flags & GPUWavefrontVertexFlags::Mis_connectible) != 0u) ? 1.0f : 0.0f;
  float previous_connectible = ((previous_vertex.flags & GPUWavefrontVertexFlags::Connectible) != 0u) ? 1.0f : 0.0f;
  result_accumulated = r1 * (previous_mis_connectible + previous_vertex.pdf_history);
  float r0 = wavefront_safe_div(current_backward_pdf, current_vertex.pdf_from_prev);
  result_accumulated = r0 * (previous_connectible + result_accumulated);
  return result_accumulated;
}

float wavefront_connect_light_mis_light(GPUWavefrontPathVertex current_vertex, GPUWavefrontPathVertex previous_vertex, float current_backward_pdf, float previous_backward_pdf) {
  float result_accumulated = 0.0f;
  float r1 = wavefront_safe_div(previous_backward_pdf, previous_vertex.pdf_from_prev);
  float previous_mis_connectible = ((previous_vertex.flags & GPUWavefrontVertexFlags::Mis_connectible) != 0u) ? 1.0f : 0.0f;
  float previous_connectible = ((previous_vertex.flags & GPUWavefrontVertexFlags::Connectible) != 0u) ? 1.0f : 0.0f;
  result_accumulated = r1 * (previous_mis_connectible + previous_vertex.pdf_history);
  float r0 = wavefront_safe_div(current_backward_pdf, current_vertex.pdf_from_prev);
  result_accumulated = r0 * (previous_connectible + result_accumulated);
  return result_accumulated;
}

float wavefront_connect_light_weight(WavefrontConnectLightPrepareInput input_value, float z_curr_pdf, float z_prev_pdf, float y_curr_pdf, float y_prev_pdf) {
  if (scene_multiple_importance_sampling_enabled() == false) {
    return 1.0f;
  }

  float w_camera = wavefront_connect_light_mis_camera(input_value.camera_vertex, input_value.camera_previous_vertex, z_curr_pdf, z_prev_pdf);
  float w_light = wavefront_connect_light_mis_light(input_value.light_vertex, input_value.light_previous_vertex, y_curr_pdf, y_prev_pdf);
  return 1.0f / (1.0f + w_camera + w_light);
}

void wavefront_clear_connect_light_task(uint dispatch_index) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.connect_light_task_buffer != kInvalidIndex) {
    GPUWavefrontConnectLightTask empty_task = (GPUWavefrontConnectLightTask)0;
    empty_task.medium_index = kInvalidIndex;
    wavefront_store_connect_light_task(resources.connect_light_task_buffer, dispatch_index, empty_task);
  }
}

bool wavefront_load_connect_light_prepare_input(uint dispatch_index, out WavefrontConnectLightPrepareInput input_value) {
  input_value = (WavefrontConnectLightPrepareInput)0;
  input_value.resources = wavefront_load_resources();
  if ((input_value.resources.connect_light_task_buffer == kInvalidIndex) || (input_value.resources.connect_light_result_buffer == kInvalidIndex)) {
    return false;
  }

  const uint vertex_stride = input_value.resources.fixed_max_bounces + 1u;
  input_value.task_index = dispatch_index;
  input_value.path_index = dispatch_index / vertex_stride;
  input_value.light_vertex_length = dispatch_index % vertex_stride;

  if ((input_value.path_index >= input_value.resources.path_capacity) || (input_value.light_vertex_length == 0u)) {
    return false;
  }

  input_value.path_meta = wavefront_load_path_meta(input_value.resources.path_meta_buffer, input_value.path_index);
  if ((scene_strategy_enabled(kSceneStrategyConnectVertices) == false) || (input_value.path_meta.camera_path_length == 0u) ||
      (input_value.light_vertex_length > input_value.path_meta.light_path_length)) {
    return false;
  }

  uint target_path_length = input_value.path_meta.camera_path_length + input_value.light_vertex_length + 1u;
  if ((target_path_length < load_scene_options_min_path_length()) || (target_path_length > load_scene_options_max_path_length())) {
    return false;
  }

  input_value.camera_vertex =
    wavefront_load_path_vertex(input_value.resources.camera_vertex_buffer, wavefront_vertex_slot(input_value.path_index, input_value.path_meta.camera_path_length));
  input_value.camera_previous_vertex =
    wavefront_load_path_vertex(input_value.resources.camera_vertex_buffer, wavefront_vertex_slot(input_value.path_index, input_value.path_meta.camera_path_length - 1u));
  input_value.light_vertex = wavefront_load_path_vertex(input_value.resources.light_vertex_buffer, wavefront_vertex_slot(input_value.path_index, input_value.light_vertex_length));
  input_value.light_previous_vertex =
    wavefront_load_path_vertex(input_value.resources.light_vertex_buffer, wavefront_vertex_slot(input_value.path_index, input_value.light_vertex_length - 1u));

  if ((wavefront_path_vertex_valid(input_value.camera_vertex) == false) || (wavefront_path_vertex_valid(input_value.camera_previous_vertex) == false) ||
      (wavefront_path_vertex_valid(input_value.light_vertex) == false) || (wavefront_path_vertex_valid(input_value.light_previous_vertex) == false) ||
      (wavefront_path_vertex_connectible(input_value.camera_vertex) == false) || (wavefront_path_vertex_connectible(input_value.light_vertex) == false)) {
    return false;
  }

  if ((try_load_material_full(input_value.camera_vertex.material_index, input_value.camera_material) == false) ||
      (try_load_material_full(input_value.light_vertex.material_index, input_value.light_material) == false)) {
    return false;
  }

  return true;
}

void wavefront_store_connect_light_prepare_task(uint dispatch_index, WavefrontConnectLightPrepareInput input_value, ETX_IN(BSDFEval, camera_eval)) {
  if (bsdf_eval_valid(camera_eval) == false) {
    return;
  }

  float3 direction_to_camera = input_value.camera_vertex.position - input_value.light_vertex.position;
  float distance_squared = dot(direction_to_camera, direction_to_camera);
  if (distance_squared <= kRayEpsilon * kRayEpsilon) {
    return;
  }

  float inv_distance = rsqrt(distance_squared);
  direction_to_camera *= inv_distance;
  float geometry_term = inv_distance * inv_distance;
  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = input_value.camera_vertex.throughput.wavelength;
  spect.flags = input_value.camera_vertex.throughput.flags;

  Sampler light_sampler = make_bsdf_sampler(scene_random_seed(input_value.task_index, constants.sample_index ^ (constants.path_iteration + 17u)));
  BSDFData light_data = bsdf_data_make(wavefront_make_connect_vertex(input_value.light_vertex.position, input_value.light_vertex.normal, input_value.light_vertex.texcoord), spect,
    input_value.light_vertex.medium_index, PathSource::Light, input_value.light_vertex.w_i);
  BSDFEval light_eval = bsdf_evaluate(make_scene_bsdf_resource_gpu_context(), light_data, direction_to_camera, input_value.light_material, light_sampler);
  if (bsdf_eval_valid(light_eval) == false) {
    return;
  }

  SpectralResponse connection = spectral_response_mul(input_value.light_vertex.throughput, spectral_response_mul(light_eval.bsdf, camera_eval.bsdf));
  if (spectral_response_is_zero(connection)) {
    return;
  }

  BSDFData camera_reverse_data =
    bsdf_data_make(wavefront_make_connect_vertex(input_value.camera_vertex.position, input_value.camera_vertex.normal, input_value.camera_vertex.texcoord), spect,
      input_value.camera_vertex.medium_index, PathSource::Camera, direction_to_camera);
  Sampler camera_reverse_sampler = make_bsdf_sampler(scene_random_seed(input_value.task_index, (constants.sample_index + 1u) ^ (constants.path_iteration + 31u)));
  float3 camera_prev_direction = normalize(input_value.camera_previous_vertex.position - input_value.camera_vertex.position);
  float z_prev_pdf_dir = wavefront_connect_light_stage_camera_bsdf_pdf(make_scene_bsdf_resource_gpu_context(), camera_reverse_data, camera_prev_direction,
    input_value.camera_material, camera_reverse_sampler);
  float z_prev_pdf = wavefront_convert_solid_angle_pdf_to_area(z_prev_pdf_dir, input_value.camera_vertex.position, input_value.camera_previous_vertex.position,
    wavefront_path_vertex_is_surface(input_value.camera_previous_vertex), input_value.camera_previous_vertex.normal);

  float y_curr_pdf = wavefront_convert_solid_angle_pdf_to_area(camera_eval.pdf, input_value.camera_vertex.position, input_value.light_vertex.position,
    wavefront_path_vertex_is_surface(input_value.light_vertex), input_value.light_vertex.normal);

  BSDFData light_reverse_data = bsdf_data_make(wavefront_make_connect_vertex(input_value.light_vertex.position, input_value.light_vertex.normal, input_value.light_vertex.texcoord),
    spect, input_value.light_vertex.medium_index, PathSource::Light, -direction_to_camera);
  Sampler light_reverse_sampler = make_bsdf_sampler(scene_random_seed(input_value.task_index, (constants.sample_index + 3u) ^ (constants.path_iteration + 43u)));
  float3 light_prev_direction = normalize(input_value.light_previous_vertex.position - input_value.light_vertex.position);
  float y_prev_pdf_dir = bsdf_pdf(make_scene_bsdf_resource_gpu_context(), light_reverse_data, light_prev_direction, input_value.light_material, light_reverse_sampler);
  float y_prev_pdf = wavefront_vertex_to_vertex_area_pdf(y_prev_pdf_dir, input_value.light_vertex, input_value.light_previous_vertex);

  float z_curr_pdf = wavefront_convert_solid_angle_pdf_to_area(light_eval.pdf, input_value.light_vertex.position, input_value.camera_vertex.position,
    wavefront_path_vertex_is_surface(input_value.camera_vertex), input_value.camera_vertex.normal);

  float weight = wavefront_connect_light_weight(input_value, z_curr_pdf, z_prev_pdf, y_curr_pdf, y_prev_pdf);
  SpectralResponse contribution = spectral_response_mul(input_value.camera_vertex.throughput, spectral_response_mul(connection, weight * geometry_term));
  if (gpu_valid_spectral_response(contribution) == false) {
    return;
  }

  float light_sign = (dot(input_value.light_vertex.geo_normal, direction_to_camera) >= 0.0f) ? 1.0f : -1.0f;
  float3 shadow_origin = offset_ray(input_value.light_vertex.position, input_value.light_vertex.geo_normal * light_sign);
  float3 shadow_delta = input_value.camera_vertex.position - shadow_origin;
  float shadow_distance = length(shadow_delta);
  if (shadow_distance <= kRayEpsilon) {
    return;
  }

  GPUWavefrontConnectLightTask task = (GPUWavefrontConnectLightTask)0;
  task.shadow_ray.o = shadow_origin;
  task.shadow_ray.d = shadow_delta / shadow_distance;
  task.shadow_ray.min_t = kRayEpsilon;
  task.shadow_ray.max_t = shadow_distance;
  task.shadow_target = input_value.camera_vertex.position;
  task.contribution = contribution;
  task.mis_weight = weight;
  task.pixel_index = input_value.camera_vertex.pixel_index;
  task.medium_index = input_value.light_vertex.medium_index;
  task.flags = 1u;
  task.path_index = input_value.path_index;
  task.sampler_seed = 0u;
  wavefront_store_connect_light_task(input_value.resources.connect_light_task_buffer, dispatch_index, task);
}
