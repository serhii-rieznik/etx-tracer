#pragma once

#include "gpu_rt_wavefront_common.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>
#include <interop/bsdf_dispatch_shared.hxx>

struct WavefrontConnectLightPrepareInput {
  GPUWavefrontResources resources;
  uint task_index;
  uint storage_index;
  uint path_index;
  uint light_vertex_length;
  GPUWavefrontPathMeta path_meta;
  GPUWavefrontPathVertex camera_vertex;
  GPUWavefrontPathVertex camera_previous_vertex;
  GPUWavefrontPathVertex light_vertex;
  GPUWavefrontPathVertex light_previous_vertex;
  Material camera_material;
  Material light_material;
  uint camera_sampler_seed;
  uint light_vertex_index;
  uint previous_light_vertex_index;
};

Vertex wavefront_make_connect_vertex(float3 position, float3 normal, float2 texcoord) {
  Vertex result = (Vertex)0;
  result.pos = position;
  result.nrm = normal;
  result.tex = texcoord;
  return result;
}

Vertex wavefront_make_connect_path_vertex(GPUWavefrontPathVertex path_vertex) {
  if (path_vertex.triangle_index == kInvalidIndex) {
    return wavefront_make_connect_vertex(path_vertex.position, path_vertex.normal, path_vertex.texcoord);
  }

  TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], path_vertex.triangle_index);
  Vertex result = wavefront_interpolate_vertex(tri, barycentrics(path_vertex.barycentric));
  if (path_vertex.instance_index != kInvalidIndex) {
    result = scene_instance_transform_vertex(load_scene_instance(path_vertex.instance_index), result);
  }

  float frame_handedness = dot(cross(result.nrm, result.tan), result.btn) >= 0.0f ? 1.0f : -1.0f;
  result.pos = path_vertex.position;
  result.nrm = path_vertex.normal;
  result.tex = path_vertex.texcoord;
  scene_math_shared_build_sampling_frame_with_handedness(result.nrm, result.tan, result.btn, frame_handedness, result.nrm, result.tan, result.btn);
  return result;
}

bool wavefront_connect_light_try_load_vertex_medium(GPUWavefrontPathVertex vertex, out MediumAccess medium_access) {
  medium_access = (MediumAccess)0;
  if (wavefront_try_load_medium(vertex.medium_index, medium_access)) {
    return true;
  }

  if (wavefront_path_vertex_is_subsurface(vertex)) {
    medium_access.phase_function_g = 0.0f;
    return true;
  }

  return false;
}

BSDFEval wavefront_connect_light_medium_eval(SpectralQuery spect, GPUWavefrontPathVertex vertex, float3 incoming_direction, float3 outgoing_direction) {
  BSDFEval result = bsdf_eval_zero(spect);
  MediumAccess medium_access = (MediumAccess)0;
  if (wavefront_connect_light_try_load_vertex_medium(vertex, medium_access) == false) {
    return result;
  }

  float phase_value = gpu_medium_phase_function(medium_access, incoming_direction, outgoing_direction);
  result.func = spectral_response_make(spect, phase_value);
  result.bsdf = result.func;
  result.pdf = phase_value;
  result.eta = 1.0f;
  result.properties = BSDFSample::Diffuse;
  result.medium_index = vertex.medium_index;
  return result;
}

float wavefront_connect_light_medium_pdf(GPUWavefrontPathVertex vertex, float3 incoming_direction, float3 outgoing_direction) {
  MediumAccess medium_access = (MediumAccess)0;
  if (wavefront_connect_light_try_load_vertex_medium(vertex, medium_access) == false) {
    return 0.0f;
  }
  return gpu_medium_phase_function(medium_access, incoming_direction, outgoing_direction);
}

bool wavefront_connect_light_stage_matches_vertex(GPUWavefrontPathVertex vertex, uint material_class) {
#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
  if (wavefront_path_vertex_is_medium(vertex)) {
    return true;
  }
#endif
  return (wavefront_path_vertex_is_surface(vertex) && wavefront_connect_light_stage_matches_material(material_class));
}

float wavefront_connect_light_mis_camera(GPUWavefrontPathVertex current_vertex, GPUWavefrontPathVertex previous_vertex, float current_backward_pdf, float previous_backward_pdf) {
  float result_accumulated = 0.0f;
  float previous_connectible = ((previous_vertex.flags & GPUWavefrontVertexFlags::Connectible) != 0u) ? 1.0f : 0.0f;
  if (current_vertex.path_length > 1u) {
    float r1 = wavefront_safe_div(previous_backward_pdf, previous_vertex.pdf_from_prev);
    float previous_mis_connectible = ((previous_vertex.flags & GPUWavefrontVertexFlags::Mis_connectible) != 0u) ? 1.0f : 0.0f;
    result_accumulated = r1 * (previous_mis_connectible + previous_vertex.pdf_history);
  }
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

float wavefront_connect_light_vertex_to_vertex_area_pdf(float pdf_dir, GPUWavefrontPathVertex from_vertex, GPUWavefrontPathVertex to_vertex) {
  if (wavefront_path_vertex_is_infinite_emitter(to_vertex)) {
    return pdf_dir;
  }
  return wavefront_convert_solid_angle_pdf_to_area(pdf_dir, from_vertex.position, to_vertex.position, wavefront_path_vertex_is_surface(to_vertex), to_vertex.normal);
}

float3 wavefront_connect_light_shadow_origin(GPUWavefrontPathVertex vertex, float3 outgoing_direction) {
  if (wavefront_path_vertex_is_medium(vertex)) {
    return vertex.position;
  }

  if (vertex.triangle_index == kInvalidIndex) {
    float sign_value = (dot(vertex.geo_normal, outgoing_direction) >= 0.0f) ? 1.0f : -1.0f;
    return offset_ray(vertex.position, vertex.geo_normal * sign_value);
  }

  TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], vertex.triangle_index);
  ByteAddressBuffer position_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)];
  ByteAddressBuffer normal_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)];

  float3 p0 = load_float3(position_buffer, tri.i.x);
  float3 p1 = load_float3(position_buffer, tri.i.y);
  float3 p2 = load_float3(position_buffer, tri.i.z);
  float3 n0 = load_float3(normal_buffer, tri.i.x);
  float3 n1 = load_float3(normal_buffer, tri.i.y);
  float3 n2 = load_float3(normal_buffer, tri.i.z);
  const GPUSceneInstanceData instance = load_scene_instance(vertex.instance_index);
  const float orientation = (instance.flags & 1u) != 0u ? -1.0f : 1.0f;
  p0 = scene_instance_transform_point(instance, p0);
  p1 = scene_instance_transform_point(instance, p1);
  p2 = scene_instance_transform_point(instance, p2);
  n0 = scene_instance_transform_normal(instance, n0) * orientation;
  n1 = scene_instance_transform_normal(instance, n1) * orientation;
  n2 = scene_instance_transform_normal(instance, n2) * orientation;
  const float3 geo_normal = scene_instance_transform_geometric_normal(instance, tri.geo_n);
  return scene_math_shared_shading_pos(p0, p1, p2, n0, n1, n2, geo_normal, barycentrics(vertex.barycentric), outgoing_direction);
}

float wavefront_connect_light_weight(WavefrontConnectLightPrepareInput input_value, float z_curr_pdf, float z_prev_pdf, float y_curr_pdf, float y_prev_pdf) {
  if (scene_multiple_importance_sampling_enabled() == false) {
    return 1.0f;
  }

  float w_camera = wavefront_connect_light_mis_camera(input_value.camera_vertex, input_value.camera_previous_vertex, z_curr_pdf, z_prev_pdf);
  float w_light = wavefront_connect_light_mis_light(input_value.light_vertex, input_value.light_previous_vertex, y_curr_pdf, y_prev_pdf);
  return 1.0f / (1.0f + w_camera + w_light);
}

bool wavefront_load_connect_light_prepare_input(uint dispatch_index, uint batch_index, out WavefrontConnectLightPrepareInput input_value) {
  input_value = (WavefrontConnectLightPrepareInput)0;
  input_value.resources = wavefront_load_resources();
  if (input_value.resources.connect_light_task_buffer == kInvalidIndex) {
    return false;
  }
  if ((constants.dispatch_item_count != 0u) && (batch_index >= constants.dispatch_item_count)) {
    return false;
  }

  const uint vertex_stride = input_value.resources.max_path_length + 1u;
  const uint queue_descriptor = wavefront_queue_current_descriptor(true);
  const uint queue_count = wavefront_queue_count(queue_descriptor);
  const uint queue_index = dispatch_index;
  if (queue_index >= queue_count) {
    return false;
  }

  input_value.light_vertex_length = constants.connect_light_vertex_length - batch_index;
  input_value.task_index = queue_index * vertex_stride + input_value.light_vertex_length;
  input_value.storage_index = batch_index * input_value.resources.path_capacity + queue_index;
  input_value.path_index = wavefront_queue_load(queue_descriptor, queue_index);

  if (input_value.light_vertex_length == 0u) {
    return false;
  }

  input_value.path_meta = wavefront_load_path_meta(input_value.resources.path_meta_buffer, input_value.path_index);
  if ((scene_strategy_enabled(kSceneStrategyConnectVertices) == false) || (input_value.path_meta.camera_path_length == 0u) ||
      (input_value.light_vertex_length > input_value.path_meta.light_path_length)) {
    return false;
  }
  const uint target_path_length = input_value.path_meta.camera_path_length + input_value.light_vertex_length + 1u;
  if ((target_path_length < load_scene_options_min_path_length()) || (target_path_length > load_scene_options_max_path_length())) {
    return false;
  }

  if (input_value.resources.camera_state_buffer != kInvalidIndex) {
    GPUWavefrontPathState camera_state = wavefront_load_path_state(input_value.resources.camera_state_buffer, input_value.path_index);
    if ((wavefront_path_state_valid(camera_state) == false) || (camera_state.path_length != input_value.path_meta.camera_path_length)) {
      return false;
    }
    input_value.camera_sampler_seed = camera_state.sampler_seed;
  }

  input_value.camera_vertex =
    wavefront_load_path_vertex(input_value.resources.camera_vertex_buffer, wavefront_camera_vertex_slot(input_value.path_index, input_value.path_meta.camera_path_length));
  input_value.camera_previous_vertex =
    wavefront_load_path_vertex(input_value.resources.camera_vertex_buffer, wavefront_camera_vertex_slot(input_value.path_index, input_value.path_meta.camera_path_length - 1u));
  wavefront_load_connect_light_candidate_indices(input_value.resources.connect_light_task_buffer, input_value.storage_index, input_value.light_vertex_index,
    input_value.previous_light_vertex_index);
  if ((input_value.light_vertex_index == kInvalidIndex) || (input_value.previous_light_vertex_index == kInvalidIndex)) {
    return false;
  }
  input_value.light_vertex = wavefront_load_path_vertex(input_value.resources.light_vertex_buffer, input_value.light_vertex_index);
  input_value.light_previous_vertex = wavefront_load_path_vertex(input_value.resources.light_vertex_buffer, input_value.previous_light_vertex_index);

  if ((wavefront_path_vertex_valid(input_value.camera_vertex) == false) || (wavefront_path_vertex_valid(input_value.camera_previous_vertex) == false) ||
      (wavefront_path_vertex_valid(input_value.light_vertex) == false) || (wavefront_path_vertex_valid(input_value.light_previous_vertex) == false) ||
      (wavefront_path_vertex_connectible(input_value.camera_vertex) == false) || (wavefront_path_vertex_connectible(input_value.light_vertex) == false)) {
    return false;
  }

  if (wavefront_path_vertex_is_surface(input_value.camera_vertex) && (try_load_material_full(input_value.camera_vertex.material_index, input_value.camera_material) == false)) {
    return false;
  }
  if (wavefront_path_vertex_is_surface(input_value.light_vertex) && (try_load_material_full(input_value.light_vertex.material_index, input_value.light_material) == false)) {
    return false;
  }

  return true;
}

#if ETX_CONNECT_LIGHT_CAMERA_PREPARE_STAGE
void wavefront_store_connect_light_camera_task(WavefrontConnectLightPrepareInput input_value, ETX_IN(BSDFEval, camera_eval)) {
  if (bsdf_eval_valid(camera_eval) == false) {
    return;
  }

  float3 direction_to_light = input_value.light_vertex.position - input_value.camera_vertex.position;
  const float distance_squared = dot(direction_to_light, direction_to_light);
  if (distance_squared <= kInvMaxHalf) {
    return;
  }

  direction_to_light *= rsqrt(distance_squared);
  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = input_value.camera_vertex.throughput.wavelength;
  spect.flags = input_value.camera_vertex.throughput.flags;

  const float3 direction_to_camera = -direction_to_light;
  float3 camera_prev_direction = normalize(input_value.camera_previous_vertex.position - input_value.camera_vertex.position);
  float z_prev_pdf_dir = 0.0f;
  if (wavefront_path_vertex_is_medium(input_value.camera_vertex)) {
    z_prev_pdf_dir = wavefront_connect_light_medium_pdf(input_value.camera_vertex, direction_to_camera, camera_prev_direction);
  } else {
    BSDFData camera_reverse_data = bsdf_data_make(wavefront_make_connect_path_vertex(input_value.camera_vertex), spect, kInvalidIndex, PathSource::Camera, direction_to_camera);
    Sampler camera_reverse_sampler = make_bsdf_sampler(scene_random_seed(input_value.task_index, (constants.sample_index + 1u) ^ (constants.path_iteration + 31u)));
    z_prev_pdf_dir = wavefront_connect_light_stage_camera_bsdf_pdf(make_scene_bsdf_resource_gpu_context(), camera_reverse_data, camera_prev_direction, input_value.camera_material,
      camera_reverse_sampler);
  }
  float z_prev_pdf = wavefront_convert_solid_angle_pdf_to_area(z_prev_pdf_dir, input_value.camera_vertex.position, input_value.camera_previous_vertex.position,
    wavefront_path_vertex_is_surface(input_value.camera_previous_vertex), input_value.camera_previous_vertex.normal);

  GPUWavefrontConnectLightCandidate candidate = (GPUWavefrontConnectLightCandidate)0;
  candidate.camera_contribution = camera_eval.bsdf;
  candidate.camera_pdf = camera_eval.pdf;
  candidate.camera_reverse_area_pdf = z_prev_pdf;
  candidate.camera_reverse_direction_pdf = z_prev_pdf_dir;
  candidate.light_vertex_index = input_value.light_vertex_index;
  candidate.previous_light_vertex_index = input_value.previous_light_vertex_index;
  candidate.flags = GPUWavefrontConnectLightTaskFlags::CameraPrepared;
  candidate.sampler_seed = input_value.camera_sampler_seed;
  candidate.light_material_class = input_value.light_material.cls;
  wavefront_store_connect_light_candidate(input_value.resources.connect_light_task_buffer, input_value.storage_index, candidate);
}
#endif
#if ETX_CONNECT_LIGHT_RESOLVE_STAGE
void wavefront_resolve_connect_light_prepare_task(uint dispatch_index, uint batch_index) {
  const GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.connect_light_task_buffer == kInvalidIndex) || (dispatch_index >= resources.path_capacity) ||
      ((constants.dispatch_item_count != 0u) && (batch_index >= constants.dispatch_item_count))) {
    return;
  }

  const uint storage_index = batch_index * resources.path_capacity + dispatch_index;
  GPUWavefrontConnectLightCandidate candidate = wavefront_load_connect_light_candidate(resources.connect_light_task_buffer, storage_index);
  if (candidate.flags != GPUWavefrontConnectLightTaskFlags::CameraPrepared) {
    return;
  }
  if (wavefront_connect_light_stage_matches_material(candidate.light_material_class) == false) {
    return;
  }

  WavefrontConnectLightPrepareInput input_value = (WavefrontConnectLightPrepareInput)0;
  if (wavefront_load_connect_light_prepare_input(dispatch_index, batch_index, input_value) == false) {
    return;
  }
  if (wavefront_connect_light_stage_matches_vertex(input_value.light_vertex, input_value.light_material.cls) == false) {
    return;
  }

  float3 direction_to_camera = input_value.camera_vertex.position - input_value.light_vertex.position;
  float distance_squared = dot(direction_to_camera, direction_to_camera);
  if (distance_squared <= kInvMaxHalf) {
    return;
  }

  float inv_distance = rsqrt(distance_squared);
  direction_to_camera *= inv_distance;
  float geometry_term = inv_distance * inv_distance;
  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = input_value.camera_vertex.throughput.wavelength;
  spect.flags = input_value.camera_vertex.throughput.flags;

  BSDFEval light_eval = (BSDFEval)0;
  if (wavefront_path_vertex_is_medium(input_value.light_vertex)) {
    light_eval = wavefront_connect_light_medium_eval(spect, input_value.light_vertex, input_value.light_vertex.w_i, direction_to_camera);
  } else {
    Sampler light_sampler = make_bsdf_sampler(scene_random_seed(input_value.task_index, constants.sample_index ^ (constants.path_iteration + 17u)));
    BSDFData light_data = bsdf_data_make(wavefront_make_connect_path_vertex(input_value.light_vertex), spect, kInvalidIndex, PathSource::Light, input_value.light_vertex.w_i);
    light_eval = wavefront_connect_light_stage_light_bsdf_eval(make_scene_bsdf_resource_gpu_context(), light_data, direction_to_camera, input_value.light_material, light_sampler);
  }
  if (bsdf_eval_valid(light_eval) == false) {
    return;
  }
  if (wavefront_path_vertex_is_surface(input_value.light_vertex)) {
    float shading_fix = bsdf_fix_shading_normal(input_value.light_vertex.geo_normal, input_value.light_vertex.normal, input_value.light_vertex.w_i, direction_to_camera);
    light_eval.bsdf = spectral_response_mul(light_eval.bsdf, shading_fix);
  }

  SpectralResponse connection = spectral_response_mul(input_value.light_vertex.throughput, spectral_response_mul(light_eval.bsdf, candidate.camera_contribution));
  if (spectral_response_is_zero(connection)) {
    return;
  }

  float z_prev_pdf = candidate.camera_reverse_area_pdf;
  float z_prev_pdf_dir = candidate.camera_reverse_direction_pdf;
  float y_curr_pdf = wavefront_convert_solid_angle_pdf_to_area(candidate.camera_pdf, input_value.camera_vertex.position, input_value.light_vertex.position,
    wavefront_path_vertex_is_surface(input_value.light_vertex), input_value.light_vertex.normal);

  float3 light_prev_direction = normalize(input_value.light_previous_vertex.position - input_value.light_vertex.position);
  float y_prev_pdf_dir = 0.0f;
  if (wavefront_path_vertex_is_medium(input_value.light_vertex)) {
    y_prev_pdf_dir = wavefront_connect_light_medium_pdf(input_value.light_vertex, -direction_to_camera, light_prev_direction);
  } else {
    BSDFData light_reverse_data = bsdf_data_make(wavefront_make_connect_path_vertex(input_value.light_vertex), spect, kInvalidIndex, PathSource::Light, -direction_to_camera);
    Sampler light_reverse_sampler = make_bsdf_sampler(scene_random_seed(input_value.task_index, (constants.sample_index + 3u) ^ (constants.path_iteration + 43u)));
    y_prev_pdf_dir = wavefront_connect_light_stage_light_bsdf_pdf(make_scene_bsdf_resource_gpu_context(), light_reverse_data, light_prev_direction, input_value.light_material,
      light_reverse_sampler);
  }
  float y_prev_pdf = wavefront_connect_light_vertex_to_vertex_area_pdf(y_prev_pdf_dir, input_value.light_vertex, input_value.light_previous_vertex);

  float z_curr_pdf = wavefront_convert_solid_angle_pdf_to_area(light_eval.pdf, input_value.light_vertex.position, input_value.camera_vertex.position,
    wavefront_path_vertex_is_surface(input_value.camera_vertex), input_value.camera_vertex.normal);

  float weight = 1.0f;
  if (scene_multiple_importance_sampling_enabled()) {
    if (scene_path_mode_is_vcm()) {
      float vm_pair = (wavefront_path_vertex_is_medium(input_value.camera_vertex) || wavefront_path_vertex_is_medium(input_value.light_vertex)) ? 0.0f : constants.vcm_vm_weight;
      float w_light = y_curr_pdf * (vm_pair + input_value.light_vertex.forward_pdf + input_value.light_vertex.reverse_pdf * y_prev_pdf_dir);
      float w_camera = z_curr_pdf * (vm_pair + input_value.camera_vertex.forward_pdf + input_value.camera_vertex.reverse_pdf * z_prev_pdf_dir);
      weight = 1.0f / (1.0f + w_light + w_camera);
    } else {
      weight = wavefront_connect_light_weight(input_value, z_curr_pdf, z_prev_pdf, y_curr_pdf, y_prev_pdf);
    }
  }
  SpectralResponse contribution = spectral_response_mul(input_value.camera_vertex.throughput, spectral_response_mul(connection, weight * geometry_term));
  if (gpu_valid_spectral_response(contribution) == false) {
    return;
  }

  float3 shadow_origin = wavefront_connect_light_shadow_origin(input_value.light_vertex, direction_to_camera);
  float3 shadow_delta = input_value.camera_vertex.position - shadow_origin;
  if (dot(shadow_delta, shadow_delta) <= (kRayEpsilon * kRayEpsilon)) {
    return;
  }

  GPUWavefrontConnectLightTask task = (GPUWavefrontConnectLightTask)0;
  task.shadow_origin = shadow_origin;
  task.shadow_target = input_value.camera_vertex.position;
  task.contribution = contribution;
  task.pixel_index = input_value.camera_vertex.pixel_index;
  task.medium_index = input_value.light_vertex.medium_index;
  task.sampler_seed = sampler_random_seed(candidate.sampler_seed, candidate.light_vertex_index);
  task.inline_medium_extinction = input_value.light_vertex.inline_medium_extinction;
  task.inline_medium_flags = input_value.light_vertex.inline_medium_flags;
  wavefront_store_connect_light_task(input_value.resources.connect_light_task_buffer, input_value.storage_index, task);
  wavefront_shadow_queue_append(input_value.resources, kGPUWavefrontShadowQueueConnectLight, input_value.storage_index);
}
#endif
