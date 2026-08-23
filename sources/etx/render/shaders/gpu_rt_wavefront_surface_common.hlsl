#pragma once

#include "gpu_rt_wavefront_common.hlsl"

BSDFEval wavefront_evaluate_material_bsdf(BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return bsdf_evaluate(make_scene_bsdf_resource_gpu_context(), data, outgoing_direction, material, sampler);
}

uint wavefront_environment_emitter_count() {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  uint emitter_instance_count = 0u;
  uint emitter_count = 0u;
  if (emitter_access_try_load_environment_state(context, emitter_instance_count, emitter_count) == false) {
    return 0u;
  }
  (void)emitter_instance_count;
  return emitter_count;
}

bool wavefront_environment_emitter_index(uint local_index, out uint emitter_index) {
  emitter_index = kInvalidIndex;
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  uint emitter_instance_count = 0u;
  uint emitter_count = 0u;
  if (emitter_access_try_load_environment_state(context, emitter_instance_count, emitter_count) == false) {
    return false;
  }
  (void)emitter_instance_count;
  if (local_index >= emitter_count) {
    return false;
  }
  return emitter_access_try_load_environment_emitter(context, local_index, emitter_index);
}

float wavefront_vertex_to_vertex_area_pdf(float pdf_dir, GPUWavefrontPathVertex from_vertex, GPUWavefrontPathVertex to_vertex) {
  if (wavefront_path_vertex_is_infinite_emitter(to_vertex)) {
    return pdf_dir;
  }
  return wavefront_convert_solid_angle_pdf_to_area(pdf_dir, from_vertex.position, to_vertex.position, wavefront_path_vertex_is_surface(to_vertex), to_vertex.normal);
}

float wavefront_distant_emitter_sample_pdf(uint emitter_index, float3 in_direction) {
  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  GPUEmitterProfileABIData emitter_profile = (GPUEmitterProfileABIData)0;
  if (try_load_emitter_instance(emitter_index, emitter_instance) == false) {
    return 0.0f;
  }
  if (try_load_emitter_profile(emitter_instance.emitter_profile_index, emitter_profile) == false) {
    return 0.0f;
  }

  const float discrete_pdf = emitter_discrete_pdf(emitter_index);
  if (emitter_instance.emitter_class == EmitterClass::Directional) {
    float cosine_threshold = emitter_profile.emitter_angular_size_cosine;
    if (cosine_threshold <= 0.0f) {
      cosine_threshold = 1.0f;
    }
    if (emitter_access_accepts_direction(in_direction, emitter_profile.emitter_direction, cosine_threshold) == false) {
      return 0.0f;
    }
    return discrete_pdf;
  }
  if (emitter_instance.emitter_class != EmitterClass::Environment) {
    return 0.0f;
  }

  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  EmitterAccess access = (EmitterAccess)0;
  if (emitter_access_try_load_distant(context, emitter_index, in_direction, access) == false) {
    return 0.0f;
  }

  float2 uv = emitter_access_environment_uv(context, access, in_direction);
  float image_pdf = 0.0f;
  float4 image_value = float4(0.0f, 0.0f, 0.0f, 0.0f);
  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(constants.scene.images);
  if (image_evaluate_try_rgba(image_context, access.emission_image_index, uv, image_pdf, image_value) == false) {
    return 0.0f;
  }

  bool is_atmosphere = (access.emitter_profile_meta & EmitterProfileMeta::Atmosphere) != 0u;
  uint projection = projection_environment_mode(is_atmosphere);
  return discrete_pdf * projection_environment_image_pdf_to_solid_angle(image_pdf, uv, projection);
}

float wavefront_distant_emitter_area_pdf(float3 emission_direction, GPUWavefrontPathVertex target_vertex) {
  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  float result = 1.0f / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius);
  if (wavefront_path_vertex_is_surface(target_vertex)) {
    result *= abs(dot(emission_direction, target_vertex.geo_normal));
  }
  return result;
}

float wavefront_direct_hit_weight(uint camera_path_length, float current_pdf_from_prev, float previous_pdf_from_prev, float previous_pdf_history, bool previous_connectible,
  bool previous_mis_connectible, float p_sample, float p_from);

float2 wavefront_environment_emitter_pdf(float3 in_direction, GPUWavefrontPathVertex target_vertex) {
  uint environment_count = wavefront_environment_emitter_count();
  if (environment_count == 0u) {
    return float2(0.0f, 0.0f);
  }

  float pdf_dir = 0.0f;
  for (uint local_index = 0u; local_index < environment_count; ++local_index) {
    uint emitter_index = kInvalidIndex;
    if (wavefront_environment_emitter_index(local_index, emitter_index) == false) {
      continue;
    }
    pdf_dir += wavefront_distant_emitter_sample_pdf(emitter_index, in_direction);
  }

  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  float pdf_area = 1.0f / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius);
  if (wavefront_path_vertex_is_surface(target_vertex)) {
    pdf_area *= abs(dot(target_vertex.geo_normal, in_direction));
  }
  pdf_dir /= float(environment_count);
  return float2(pdf_area, pdf_dir);
}

SpectralResponse wavefront_environment_direct_hit_emitter_radiance(uint emitter_index, float3 direction, SpectralQuery spect, bool directly_visible, out float local_pdf_dir) {
  local_pdf_dir = 0.0f;

  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  GPUEmitterProfileABIData emitter_profile = (GPUEmitterProfileABIData)0;
  if ((try_load_emitter_instance(emitter_index, emitter_instance) == false) || (try_load_emitter_profile(emitter_instance.emitter_profile_index, emitter_profile) == false)) {
    return spectral_response_zero(spect);
  }

  if (emitter_instance.emitter_class == EmitterClass::Directional) {
    float directional_cosine = dot(direction, emitter_profile.emitter_direction);
    if ((directly_visible == false) || (emitter_profile.emitter_angular_size_cosine >= 1.0f) || (directional_cosine < emitter_profile.emitter_angular_size_cosine)) {
      return spectral_response_zero(spect);
    }

    local_pdf_dir = 1.0f;
    float sin_half_angle = sqrt(max(0.0f, 1.0f - (emitter_profile.emitter_angular_size_cosine * emitter_profile.emitter_angular_size_cosine)));
    float equivalent_disk_size = (emitter_profile.emitter_angular_size_cosine > kEpsilon) ? (2.0f * (sin_half_angle / emitter_profile.emitter_angular_size_cosine)) : 0.0f;
    float2 uv = disk_uv(emitter_profile.emitter_direction, direction, equivalent_disk_size, emitter_profile.emitter_angular_size_cosine);
    SpectralResponse value = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, uv, spect);
    SpectralResponse spectrum_value = load_scene_spectrum_or_zero(emitter_profile.emission_spectrum_index, spect);
    if (spectral_response_is_zero(spectrum_value)) {
      return spectral_response_zero(spect);
    }

    float normalization = kDoublePi * (1.0f - emitter_profile.emitter_angular_size_cosine);
    return spectral_response_div(value, spectral_response_mul(spectrum_value, normalization));
  }

  local_pdf_dir = wavefront_distant_emitter_sample_pdf(emitter_index, direction) / max(kEpsilon, emitter_discrete_pdf(emitter_index));
  return evaluate_distant_emission_spectral(emitter_index, direction, spect);
}

SpectralResponse wavefront_compute_environment_direct_hit_contribution(SpectralQuery spect, GPUWavefrontPathState state, GPUWavefrontPathVertex previous_vertex) {
  bool directly_visible = state.path_length <= 1u;
  SpectralResponse accumulated = spectral_response_zero(spect);
  uint environment_count = wavefront_environment_emitter_count();
  for (uint local_index = 0u; local_index < environment_count; ++local_index) {
    uint emitter_index = kInvalidIndex;
    if (wavefront_environment_emitter_index(local_index, emitter_index) == false) {
      continue;
    }

    float local_pdf_dir = 0.0f;
    SpectralResponse value = wavefront_environment_direct_hit_emitter_radiance(emitter_index, state.ray.d, spect, directly_visible, local_pdf_dir);
    if (spectral_response_is_zero(value)) {
      continue;
    }

    float this_weight = 1.0f;
    if (scene_multiple_importance_sampling_enabled() && scene_path_mode_is_path_tracing() && wavefront_path_vertex_connectible(previous_vertex) && (state.path_length > 1u)) {
      float this_p_connect = local_pdf_dir * emitter_discrete_pdf(emitter_index);
      this_weight = power_heuristic(previous_vertex.sampled_bsdf_pdf, this_p_connect);
    }

    accumulated = spectral_response_add(accumulated, spectral_response_mul(value, this_weight));
  }

  if (spectral_response_is_zero(accumulated)) {
    return accumulated;
  }

  float mis_weight = 1.0f;
  if (scene_multiple_importance_sampling_enabled() && (state.path_length > 1u) && (scene_path_mode_is_path_tracing() == false)) {
    float2 emitter_pdfs = wavefront_environment_emitter_pdf(state.ray.d, previous_vertex);
    if (scene_path_mode_is_vcm()) {
      mis_weight = 1.0f / (1.0f + state.forward_pdf * emitter_pdfs.y + state.reverse_pdf * emitter_pdfs.x * emitter_pdfs.y);
    } else {
      mis_weight = wavefront_direct_hit_weight(state.path_length, state.sampled_bsdf_pdf, previous_vertex.pdf_from_prev, previous_vertex.pdf_history,
        wavefront_path_vertex_connectible(previous_vertex), wavefront_path_vertex_mis_connectible(previous_vertex), emitter_pdfs.y, emitter_pdfs.x);
    }
  }

  return spectral_response_mul(spectral_response_mul(state.throughput, accumulated), mis_weight);
}

bool wavefront_path_state_refractive_depth_limit_reached(GPUWavefrontPathState state) {
  return (state.flags & GPUWavefrontPathFlags::Depth_limit_reached_while_refractive) != 0u;
}

bool wavefront_path_tracing_neutral_depth_exceeded(GPUWavefrontPathState state, uint path_length) {
  return scene_path_mode_is_path_tracing() && (path_length > load_scene_options_max_path_length()) && gpu_path_tracing_neutral_eta(state.eta);
}

bool wavefront_path_tracing_should_terminate_neutral_depth(GPUWavefrontPathState state, uint path_length) {
  return wavefront_path_tracing_neutral_depth_exceeded(state, path_length) && (wavefront_path_state_refractive_depth_limit_reached(state) == false);
}

void wavefront_path_tracing_update_refractive_depth(inout GPUWavefrontPathState state, uint path_length) {
  if (scene_path_mode_is_path_tracing() && (path_length > load_scene_options_max_path_length()) && (gpu_path_tracing_neutral_eta(state.eta) == false)) {
    state.flags |= GPUWavefrontPathFlags::Depth_limit_reached_while_refractive;
  }
}

uint wavefront_path_state_persistent_flags(GPUWavefrontPathState state) {
  return state.flags & GPUWavefrontPathFlags::Depth_limit_reached_while_refractive;
}

SpectralResponse wavefront_evaluate_local_direct_hit_radiance(uint emitter_index, SpectralQuery spect, float3 source_position, float3 target_position, float2 uv,
  bool directly_visible, out float pdf_area, out float pdf_dir, out float pdf_dir_out) {
  pdf_area = 0.0f;
  pdf_dir = 0.0f;
  pdf_dir_out = 0.0f;

  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  GPUEmitterProfileABIData emitter_profile = (GPUEmitterProfileABIData)0;
  if ((try_load_emitter_instance(emitter_index, emitter_instance) == false) || (try_load_emitter_profile(emitter_instance.emitter_profile_index, emitter_profile) == false)) {
    return spectral_response_zero(spect);
  }
  if (emitter_instance.emitter_class != EmitterClass::Area) {
    return spectral_response_zero(spect);
  }

  TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], emitter_instance.triangle_index);
  const float3 geo_normal = scene_instance_transform_geometric_normal(load_scene_instance(emitter_instance.instance_index), tri.geo_n);
  Material material = (Material)0;
  if (try_load_material_full(tri.material_index, material) == false) {
    return spectral_response_zero(spect);
  }

  float3 target_delta = target_position - source_position;
  if (dot(geo_normal, target_delta) >= 0.0f) {
    return spectral_response_zero(spect);
  }

  pdf_area = (emitter_instance.triangle_area > 0.0f) ? (1.0f / emitter_instance.triangle_area) : 0.0f;
  if (pdf_area <= 0.0f) {
    return spectral_response_zero(spect);
  }

  float3 dp = source_position - target_position;
  float distance_squared = dot(dp, dp);
  if (distance_squared > 0.0f) {
    float cos_t = abs(dot(dp, geo_normal)) / sqrt(distance_squared);
    float exponent = scene_math_shared_collimation_to_exponent(material.emission_collimation);
    float cos_tx = directly_visible ? cos_t : pow(cos_t, exponent);
    if (cos_tx > kEpsilon) {
      pdf_dir = pdf_area * distance_squared / cos_tx;
      pdf_dir_out = pdf_area * cos_tx * kInvPi;
    }
  }

  return evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, uv, spect);
}

float wavefront_local_direct_hit_pdf_from_emitter(uint emitter_index, SpectralQuery spect, GPUWavefrontPathVertex emitter_vertex, GPUWavefrontPathVertex target_vertex) {
  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  GPUEmitterProfileABIData emitter_profile = (GPUEmitterProfileABIData)0;
  if ((try_load_emitter_instance(emitter_index, emitter_instance) == false) || (try_load_emitter_profile(emitter_instance.emitter_profile_index, emitter_profile) == false)) {
    return 0.0f;
  }
  if (emitter_instance.emitter_class != EmitterClass::Area) {
    return 0.0f;
  }

  TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], emitter_instance.triangle_index);
  Material material = (Material)0;
  if (try_load_material_full(tri.material_index, material) == false) {
    return 0.0f;
  }

  float pdf_area = 0.0f;
  float pdf_dir = 0.0f;
  float pdf_dir_out = 0.0f;
  float3 w_o = normalize(target_vertex.position - emitter_vertex.position);
  float exponent = scene_math_shared_collimation_to_exponent(material.emission_collimation);
  float cos_t = max(0.0f, dot(emitter_vertex.normal, w_o));
  pdf_dir = pow(cos_t, exponent) * kInvPi;
  if (pdf_dir <= 0.0f) {
    return 0.0f;
  }

  pdf_area = (emitter_instance.triangle_area > 0.0f) ? (1.0f / emitter_instance.triangle_area) : 0.0f;
  pdf_dir_out = pdf_dir * pdf_area;
  (void)pdf_dir_out;
  return wavefront_convert_solid_angle_pdf_to_area(pdf_dir, emitter_vertex.position, target_vertex.position, wavefront_path_vertex_is_surface(target_vertex), target_vertex.normal);
}

float wavefront_direct_hit_weight(uint camera_path_length, float current_pdf_from_prev, float previous_pdf_from_prev, float previous_pdf_history, bool previous_connectible,
  bool previous_mis_connectible, float p_sample, float p_from) {
  if (scene_path_mode_is_bdpt_full()) {
    float result_accumulated = 0.0f;
    if (camera_path_length > 1u) {
      float r1 = wavefront_safe_div(p_from, previous_pdf_from_prev);
      result_accumulated = r1 * ((previous_mis_connectible ? 1.0f : 0.0f) + previous_pdf_history);
    }
    float r0 = wavefront_safe_div(p_sample, current_pdf_from_prev);
    result_accumulated = r0 * ((previous_connectible ? 1.0f : 0.0f) + result_accumulated);
    return 1.0f / (1.0f + result_accumulated);
  }

  if (scene_path_mode_uses_bdpt_fast()) {
    float to_emitter_direct = previous_pdf_from_prev * current_pdf_from_prev;
    float to_emitter_connect = previous_connectible ? (previous_pdf_from_prev * p_sample) : 0.0f;
    float p_from_light = p_from * p_sample;
    float p_ratio = previous_pdf_history;
    return balance_heuristic(to_emitter_direct, to_emitter_connect, p_ratio * p_from_light);
  }

  float p_sample_value = previous_connectible ? p_sample : 0.0f;
  return power_heuristic(previous_pdf_from_prev, p_sample_value);
}

SpectralResponse wavefront_compute_local_direct_hit_contribution(SpectralQuery spect, uint camera_path_length, GPUWavefrontPathVertex current_vertex,
  GPUWavefrontPathVertex previous_vertex, uint emitter_index) {
  if ((camera_path_length < load_scene_options_min_path_length()) || (camera_path_length > load_scene_options_max_path_length())) {
    return spectral_response_zero(spect);
  }

  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  if (try_load_emitter_instance(emitter_index, emitter_instance) == false) {
    return spectral_response_zero(spect);
  }
  if (emitter_instance.emitter_class != EmitterClass::Area) {
    return spectral_response_zero(spect);
  }

  bool directly_visible = camera_path_length <= 1u;
  float pdf_area = 0.0f;
  float pdf_dir = 0.0f;
  float pdf_dir_out = 0.0f;
  SpectralResponse emitter_value = wavefront_evaluate_local_direct_hit_radiance(emitter_index, spect, previous_vertex.position, current_vertex.position, current_vertex.texcoord,
    directly_visible, pdf_area, pdf_dir, pdf_dir_out);
  (void)pdf_dir_out;
  if (pdf_dir <= 0.0f) {
    return spectral_response_zero(spect);
  }

  float mis_weight = 1.0f;
  if ((scene_multiple_importance_sampling_enabled()) && (camera_path_length > 1u)) {
    bool previous_connectible = wavefront_path_vertex_connectible(previous_vertex);
    if (scene_path_mode_is_vcm()) {
      float emitter_sample_pdf = emitter_discrete_pdf(emitter_index);
      float w_camera = current_vertex.forward_pdf * pdf_area * emitter_sample_pdf + current_vertex.reverse_pdf * pdf_dir_out * emitter_sample_pdf;
      mis_weight = 1.0f / (1.0f + w_camera);
    } else if (scene_path_mode_is_path_tracing()) {
      float p_connect = emitter_discrete_pdf(emitter_index) * pdf_dir;
      mis_weight = previous_connectible ? power_heuristic(previous_vertex.sampled_bsdf_pdf, p_connect) : 1.0f;
    } else {
      float p_sample = emitter_discrete_pdf(emitter_index) * pdf_area;
      float p_from = wavefront_local_direct_hit_pdf_from_emitter(emitter_index, spect, current_vertex, previous_vertex);
      mis_weight = wavefront_direct_hit_weight(camera_path_length, current_vertex.pdf_from_prev, previous_vertex.pdf_from_prev, previous_vertex.pdf_history, previous_connectible,
        wavefront_path_vertex_mis_connectible(previous_vertex), p_sample, p_from);
    }
  }

  return spectral_response_mul(spectral_response_mul(emitter_value, current_vertex.throughput), mis_weight);
}

void wavefront_surface_precompute_camera_mis(bool current_connectible, uint path_length, inout GPUWavefrontPathMeta meta, inout GPUWavefrontPathVertex previous_vertex) {
  previous_vertex.pdf_ratio = wavefront_safe_div(previous_vertex.pdf_from_next, previous_vertex.pdf_from_prev);
  previous_vertex.pdf_history = meta.camera_mis_history;

  if (scene_path_mode_uses_bdpt_fast()) {
    if (path_length == 1u) {
      previous_vertex.pdf_accumulated = current_connectible ? meta.camera_mis_history : 0.0f;
    } else {
      previous_vertex.pdf_accumulated = meta.camera_mis_history * previous_vertex.pdf_ratio;
    }
  } else if (path_length > 1u) {
    float previous_mis_connectible = wavefront_path_vertex_mis_connectible(previous_vertex) ? 1.0f : 0.0f;
    previous_vertex.pdf_accumulated = previous_vertex.pdf_ratio * (previous_mis_connectible + meta.camera_mis_history);
  }
  meta.camera_mis_history = previous_vertex.pdf_accumulated;
}

float wavefront_medium_connect_camera_weight(Camera camera, CameraFilmSampleShared camera_sample, MediumAccess medium_access, GPUWavefrontResources resources, uint path_index,
  GPUWavefrontPathMeta path_meta, GPUWavefrontPathVertex current_vertex, GPUWavefrontPathVertex previous_vertex) {
  if ((scene_multiple_importance_sampling_enabled() == false) || scene_path_mode_is_light_tracing()) {
    return 1.0f;
  }

  float current_from_camera_dir = camera_shared_film_pdf_out(camera, current_vertex.position);
  float current_from_camera = wavefront_convert_solid_angle_pdf_to_area(current_from_camera_dir, camera_sample.position, current_vertex.position, false, float3(0.0f, 0.0f, 0.0f));

  if (scene_path_mode_uses_bdpt_fast()) {
    if (path_meta.light_path_length == 0u) {
      return 1.0f;
    }

    GPUWavefrontPathVertex emitter_root = (GPUWavefrontPathVertex)0;
    GPUWavefrontPathVertex first_light_vertex = (GPUWavefrontPathVertex)0;
    if (path_meta.light_path_length == 1u) {
      emitter_root = previous_vertex;
      first_light_vertex = current_vertex;
    } else if (resources.fast_light_endpoint_buffer != kInvalidIndex) {
      const GPUWavefrontFastLightEndpoint endpoint = wavefront_load_fast_light_endpoint(resources.fast_light_endpoint_buffer, path_index);
      emitter_root.pdf_from_prev = endpoint.emitter_pdf_from_prev;
      emitter_root.pdf_from_next = endpoint.emitter_pdf_from_next;
      emitter_root.flags = endpoint.emitter_flags;
      first_light_vertex.flags = endpoint.first_vertex_flags;
    } else {
      emitter_root = wavefront_load_path_vertex(resources.light_vertex_buffer, wavefront_light_vertex_slot(path_index, 0u));
      if (path_meta.light_path_length >= 1u) {
        first_light_vertex = wavefront_load_path_vertex(resources.light_vertex_buffer, wavefront_light_vertex_slot(path_index, 1u));
      }
    }
    if (wavefront_path_vertex_valid(emitter_root) == false) {
      return 1.0f;
    }

    float previous_from_current_dir = gpu_medium_phase_function(medium_access, -camera_sample.direction, current_vertex.w_i);
    float previous_from_current = wavefront_path_vertex_is_infinite_emitter(previous_vertex)
                                    ? previous_from_current_dir
                                    : wavefront_convert_solid_angle_pdf_to_area(previous_from_current_dir, current_vertex.position, previous_vertex.position,
                                        wavefront_path_vertex_is_surface(previous_vertex), previous_vertex.normal);

    float p_sample = emitter_root.pdf_from_prev;
    float p_light = previous_vertex.pdf_from_prev * current_vertex.pdf_from_prev;
    float p_bck = current_from_camera * previous_vertex.pdf_history;
    float p_direct = previous_from_current;
    if (path_meta.light_path_length > 1u) {
      p_direct = emitter_root.pdf_from_next;
      p_bck *= previous_from_current;
      p_light *= p_sample;
    }

    float p_camera_direct = ((emitter_root.flags & GPUWavefrontVertexFlags::Mis_connectible) != 0u) ? (p_bck * p_direct) : 0.0f;
    float p_camera_connect = ((first_light_vertex.flags & GPUWavefrontVertexFlags::Connectible) != 0u) ? (p_bck * p_sample) : 0.0f;
    return balance_heuristic(p_light, p_camera_direct, p_camera_connect);
  }

  float reverse_phase_pdf = gpu_medium_phase_function(medium_access, -camera_sample.direction, current_vertex.w_i);
  float w_light = current_from_camera * (current_vertex.forward_pdf + current_vertex.reverse_pdf * reverse_phase_pdf);
  return 1.0f / (1.0f + w_light);
}

void wavefront_store_medium_connect_camera_task(uint dispatch_index, uint path_index, GPUWavefrontResources resources, inout GPUWavefrontPathState state,
  GPUWavefrontPathMeta path_meta, GPUWavefrontPathVertex current_vertex, GPUWavefrontPathVertex previous_vertex) {
  if ((resources.connect_camera_task_buffer == kInvalidIndex) || (constants.camera_buffer_index == kInvalidIndex) ||
      (scene_strategy_enabled(kSceneStrategyConnectToCamera) == false) || (wavefront_path_vertex_connectible(current_vertex) == false)) {
    return;
  }

  if (wavefront_medium_explicit_connections_enabled(current_vertex.medium_index) == false) {
    return;
  }

  uint target_path_length = path_meta.light_path_length + 1u;
  if ((target_path_length < load_scene_options_min_path_length()) || (target_path_length > load_scene_options_max_path_length())) {
    return;
  }

  MediumAccess medium_access = (MediumAccess)0;
  if (wavefront_try_load_medium(current_vertex.medium_index, medium_access) == false) {
    return;
  }

  Camera camera = load_camera(bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)]);
  if (camera.cls == Camera::Class::Equirectangular) {
    return;
  }

  uint seed = state.sampler_seed;
  float2 lens_rnd = float2(0.0f, 0.0f);
  if (camera_lens_sampling_enabled(camera.lens_radius, camera.focal_distance)) {
    lens_rnd = float2(rnd01(seed), rnd01(seed));
  }
  float2 sensor_sample = camera_sample_lens_uv(camera, lens_rnd) * camera.lens_radius;
  float3 lens_point = camera_film_shared_lens_point(camera, sensor_sample);
  CameraFilmSampleShared camera_sample = camera_film_shared_evaluate(camera, current_vertex.position, lens_point);
  state.sampler_seed = seed;
  if ((camera_sample.pdf_dir <= 0.0f) || (camera_sample.weight <= 0.0f)) {
    return;
  }

  uint pixel_index = 0u;
  if (wavefront_camera_ndc_to_pixel_index(camera, camera_sample.uv, pixel_index) == false) {
    return;
  }

  float phase_value = gpu_medium_phase_function(medium_access, current_vertex.w_i, camera_sample.direction);
  if (phase_value <= 0.0f) {
    return;
  }

  float mis_weight = wavefront_medium_connect_camera_weight(camera, camera_sample, medium_access, resources, path_index, path_meta, current_vertex, previous_vertex);
  SpectralResponse contribution = spectral_response_mul(current_vertex.throughput, phase_value * camera_sample.weight * mis_weight);
  if (gpu_valid_spectral_response(contribution) == false) {
    return;
  }

  float len = length(camera_sample.position - current_vertex.position);
  float direction_scale = camera_shared_clip_direction_scale(camera, camera_sample.direction);
  float near_extent = (camera.clip_near > 0.0f) ? (camera.clip_near / direction_scale) : 0.0f;
  float3 clip_pos = current_vertex.position + camera_sample.direction * max(0.0f, len - near_extent);
  float3 shadow_delta = clip_pos - current_vertex.position;
  float shadow_distance = length(shadow_delta);
  if (shadow_distance <= kRayEpsilon) {
    return;
  }

  GPUWavefrontConnectCameraTask task = (GPUWavefrontConnectCameraTask)0;
  task.shadow_ray.o = current_vertex.position;
  task.shadow_ray.d = shadow_delta / shadow_distance;
  task.shadow_ray.min_t = kRayEpsilon;
  task.shadow_ray.max_t = shadow_distance;
  task.shadow_target = clip_pos;
  task.contribution = contribution;
  task.mis_weight = mis_weight;
  task.pixel_index = pixel_index;
  task.medium_index = current_vertex.medium_index;
  task.flags = 1u;
  task.path_index = path_index;
  task.sampler_seed = state.sampler_seed;
  wavefront_store_connect_camera_task(resources.connect_camera_task_buffer, dispatch_index, task);
  wavefront_shadow_queue_append(resources, kGPUWavefrontShadowQueueConnectCamera, dispatch_index);
}

void wavefront_surface_classify(bool from_camera, uint dispatch_index) {
  uint queue_descriptor = wavefront_queue_current_descriptor(from_camera);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  GPUWavefrontResources resources = wavefront_load_resources();
  uint path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  uint state_descriptor = from_camera ? resources.camera_state_buffer : resources.light_state_buffer;
  uint hit_descriptor = from_camera ? resources.camera_hit_buffer : resources.light_hit_buffer;
  if ((from_camera == false) && (resources.connect_camera_task_buffer != kInvalidIndex)) {
    GPUWavefrontConnectCameraTask empty_task = (GPUWavefrontConnectCameraTask)0;
    empty_task.medium_index = kInvalidIndex;
    wavefront_store_connect_camera_task(resources.connect_camera_task_buffer, dispatch_index, empty_task);
  }
  GPUWavefrontPathState state = wavefront_load_path_state(state_descriptor, path_index);
  GPUWavefrontHit hit = wavefront_load_hit(hit_descriptor, path_index);
  if ((wavefront_path_state_valid(state) == false) || (wavefront_hit_valid(hit) == false)) {
    return;
  }

  state.throughput = spectral_response_mul(state.throughput, hit.transmittance);
  wavefront_path_tracing_update_refractive_depth(state, state.path_length);
  const bool path_tracing_neutral_depth_exceeded = wavefront_path_tracing_neutral_depth_exceeded(state, state.path_length);
  const bool path_tracing_terminate_neutral_depth = wavefront_path_tracing_should_terminate_neutral_depth(state, state.path_length);
  if (wavefront_hit_is_miss(hit)) {
    if (path_tracing_terminate_neutral_depth) {
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }

    const bool direct_hit_path_length_enabled =
      scene_path_mode_is_path_tracing() || ((state.path_length >= load_scene_options_min_path_length()) && (state.path_length <= load_scene_options_max_path_length()));
    if (from_camera && (scene_path_mode_is_light_tracing() == false) && scene_strategy_enabled(kSceneStrategyDirectHit) && direct_hit_path_length_enabled) {
      GPUWavefrontPathVertex previous_vertex = wavefront_load_path_vertex(resources.camera_vertex_buffer, wavefront_camera_vertex_slot(path_index, state.path_length - 1u));
      if (wavefront_path_vertex_valid(previous_vertex)) {
        if (scene_path_mode_is_path_tracing() == false) {
          GPUWavefrontPathMeta meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
          wavefront_surface_precompute_camera_mis(true, state.path_length, meta, previous_vertex);
        }
        SpectralResponse contribution = wavefront_compute_environment_direct_hit_contribution(state.spect, state, previous_vertex);
        if (spectral_response_is_zero(contribution) == false) {
          wavefront_film_add(state.pixel_index, wavefront_spectral_estimate(contribution, state.spect));
        }
      }
    }
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  if (wavefront_hit_is_medium(hit)) {
    wavefront_write_medium_vertex(from_camera, path_index, state, hit);
    wavefront_write_path_meta(from_camera, path_index, state);

    uint vertex_descriptor = from_camera ? resources.camera_vertex_buffer : resources.light_vertex_buffer;
    GPUWavefrontPathVertex current_vertex = wavefront_load_path_vertex(vertex_descriptor, wavefront_path_vertex_slot(from_camera, path_index, state.path_length));
    GPUWavefrontPathVertex previous_vertex = wavefront_load_path_vertex(vertex_descriptor, wavefront_path_vertex_slot(from_camera, path_index, state.path_length - 1u));
    if ((wavefront_path_vertex_valid(current_vertex) == false) || (wavefront_path_vertex_valid(previous_vertex) == false)) {
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }

    const bool subsurface_medium_vertex = wavefront_path_vertex_is_subsurface(current_vertex);
    current_vertex.flags &= ~(GPUWavefrontVertexFlags::Connectible | GPUWavefrontVertexFlags::Mis_connectible | GPUWavefrontVertexFlags::Delta);
    current_vertex.flags |= GPUWavefrontVertexFlags::Connectible;
    if (wavefront_path_vertex_connectible(previous_vertex)) {
      current_vertex.flags |= GPUWavefrontVertexFlags::Mis_connectible;
    }

    current_vertex.pdf_from_prev = wavefront_vertex_to_vertex_area_pdf(state.sampled_bsdf_pdf, previous_vertex, current_vertex);
    current_vertex.forward_pdf = state.forward_pdf * hit.hit_t * hit.hit_t;
    current_vertex.reverse_pdf = state.reverse_pdf;
    current_vertex.d_vm = state.d_vm;
    if ((from_camera == false) && (state.path_length == 1u) && (previous_vertex.emitter_index != kInvalidIndex)) {
      GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
      if (try_load_emitter_instance(previous_vertex.emitter_index, emitter_instance) && (emitter_instance.emitter_class != EmitterClass::Area)) {
        previous_vertex.pdf_from_prev = wavefront_distant_emitter_sample_pdf(previous_vertex.emitter_index, -previous_vertex.w_i);
        current_vertex.pdf_from_prev = wavefront_distant_emitter_area_pdf(previous_vertex.w_i, current_vertex);
        wavefront_store_path_vertex(vertex_descriptor, wavefront_path_vertex_slot(from_camera, path_index, state.path_length - 1u), previous_vertex);
      }
    }
    MediumAccess medium_access = (MediumAccess)0;
    if (subsurface_medium_vertex) {
      GPUWavefrontSubsurfaceState subsurface_state = wavefront_load_subsurface_state(wavefront_subsurface_state_buffer(resources, from_camera), path_index);
      medium_access.phase_function_g = subsurface_state.phase_function_g;
    } else {
      if (wavefront_try_load_medium(current_vertex.medium_index, medium_access) == false) {
        state.flags = 0u;
        wavefront_store_path_state(state_descriptor, path_index, state);
        return;
      }
    }

    float2 sample_random = float2(rnd01(state.sampler_seed), rnd01(state.sampler_seed));
    float2 connection_random = float2(rnd01(state.sampler_seed), rnd01(state.sampler_seed));
    float2 support_random = float2(rnd01(state.sampler_seed), rnd01(state.sampler_seed));
    if (from_camera && (state.path_length == 1u) && (constants.sample_index < 256u)) {
      sample_random = float2(sample_blue_noise_value(state.pixel, constants.sample_index, 0u), sample_blue_noise_value(state.pixel, constants.sample_index, 1u));
      connection_random = float2(sample_blue_noise_value(state.pixel, constants.sample_index, 2u), sample_blue_noise_value(state.pixel, constants.sample_index, 3u));
      support_random = float2(sample_blue_noise_value(state.pixel, constants.sample_index, 4u), sample_blue_noise_value(state.pixel, constants.sample_index, 5u));
    }
    state.film_uv = connection_random;
    state.last_emitter_pdf = support_random.y;
    float3 sampled_direction = gpu_medium_sample_phase_function(medium_access, sample_random, current_vertex.w_i);
    float phase_pdf = gpu_medium_phase_function(medium_access, current_vertex.w_i, sampled_direction);
    if ((gpu_valid_direction(sampled_direction) == false) || (phase_pdf <= 0.0f)) {
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }

    float current_d_vcm = current_vertex.forward_pdf;
    float current_d_vc = current_vertex.reverse_pdf;
    float current_d_vm = current_vertex.d_vm;
    float reverse_phase_pdf = gpu_medium_phase_function(medium_access, sampled_direction, current_vertex.w_i);
    previous_vertex.pdf_from_next = wavefront_vertex_to_vertex_area_pdf(reverse_phase_pdf, current_vertex, previous_vertex);

    GPUWavefrontPathMeta path_meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
    if (from_camera) {
      wavefront_surface_precompute_camera_mis(wavefront_path_vertex_connectible(current_vertex), state.path_length, path_meta, previous_vertex);
    } else {
      previous_vertex.pdf_ratio = wavefront_safe_div(previous_vertex.pdf_from_next, previous_vertex.pdf_from_prev);
      previous_vertex.pdf_history = path_meta.light_mis_history;
      if (scene_path_mode_uses_bdpt_fast()) {
        float scale = (state.path_length > 1u) ? previous_vertex.pdf_ratio : 1.0f;
        previous_vertex.pdf_accumulated = path_meta.light_mis_history * scale;
      } else {
        float previous_mis_connectible = wavefront_path_vertex_mis_connectible(previous_vertex) ? 1.0f : 0.0f;
        previous_vertex.pdf_accumulated = previous_vertex.pdf_ratio * (previous_mis_connectible + path_meta.light_mis_history);
      }
      path_meta.light_mis_history = previous_vertex.pdf_accumulated;
    }

    current_vertex.sampled_bsdf_pdf = phase_pdf;
    state.forward_pdf = wavefront_safe_div(1.0f, phase_pdf);
    state.reverse_pdf = wavefront_safe_div((current_d_vc * reverse_phase_pdf) + current_d_vcm, phase_pdf);
    state.d_vm = scene_path_mode_is_vcm() ? wavefront_safe_div(current_d_vm * reverse_phase_pdf, phase_pdf) : 0.0f;
    state.sampled_bsdf_pdf = phase_pdf;
    state.ray.o = current_vertex.position;
    state.ray.d = normalize(sampled_direction);
    state.ray.min_t = kRayEpsilon;
    state.ray.max_t = kMaxFloat;

    wavefront_store_path_vertex(vertex_descriptor, wavefront_path_vertex_slot(from_camera, path_index, state.path_length - 1u), previous_vertex);
    wavefront_store_path_vertex(vertex_descriptor, wavefront_path_vertex_slot(from_camera, path_index, state.path_length), current_vertex);
    if ((from_camera == false) && (state.path_length == 1u) && (resources.fast_light_endpoint_buffer != kInvalidIndex)) {
      wavefront_store_fast_light_endpoint(resources.fast_light_endpoint_buffer, path_index, previous_vertex, current_vertex);
    }
    wavefront_store_path_meta(resources.path_meta_buffer, path_index, path_meta);

    if ((from_camera == false) && (subsurface_medium_vertex == false)) {
      wavefront_store_medium_connect_camera_task(dispatch_index, path_index, resources, state, path_meta, current_vertex, previous_vertex);
    }

    state.reserved0 = GPUWavefrontPendingContinuationFlags::Prepared | GPUWavefrontPendingContinuationFlags::Continue;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  wavefront_write_vertex(from_camera, path_index, state, hit);
  wavefront_write_path_meta(from_camera, path_index, state);
  uint vertex_descriptor = from_camera ? resources.camera_vertex_buffer : resources.light_vertex_buffer;
  GPUWavefrontPathVertex current_vertex = wavefront_load_path_vertex(vertex_descriptor, wavefront_path_vertex_slot(from_camera, path_index, state.path_length));
  GPUWavefrontPathVertex previous_vertex = wavefront_load_path_vertex(vertex_descriptor, wavefront_path_vertex_slot(from_camera, path_index, state.path_length - 1u));
  if (wavefront_path_vertex_valid(previous_vertex)) {
    current_vertex.pdf_from_prev = wavefront_vertex_to_vertex_area_pdf(state.sampled_bsdf_pdf, previous_vertex, current_vertex);
    float cos_to_prev = abs(dot(hit.vertex.nrm, -state.ray.d));
    if (cos_to_prev > 0.0f) {
      current_vertex.forward_pdf = wavefront_safe_div(state.forward_pdf * hit.hit_t * hit.hit_t, cos_to_prev);
      current_vertex.reverse_pdf = wavefront_safe_div(state.reverse_pdf, cos_to_prev);
      current_vertex.d_vm = wavefront_safe_div(state.d_vm, cos_to_prev);
      state.forward_pdf = current_vertex.forward_pdf;
      state.reverse_pdf = current_vertex.reverse_pdf;
      state.d_vm = current_vertex.d_vm;
    }
    if ((from_camera == false) && (state.path_length == 1u) && (previous_vertex.emitter_index != kInvalidIndex)) {
      GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
      if (try_load_emitter_instance(previous_vertex.emitter_index, emitter_instance) && (emitter_instance.emitter_class != EmitterClass::Area)) {
        previous_vertex.pdf_from_prev = wavefront_distant_emitter_sample_pdf(previous_vertex.emitter_index, -previous_vertex.w_i);
        if (scene_path_mode_is_vcm() && (cos_to_prev > 0.0f)) {
          // CPU VCM's first distant-emitter segment has no finite-distance
          // squared Jacobian; only its surface cosine is folded into d_vcm.
          current_vertex.forward_pdf = wavefront_safe_div(state.forward_pdf, hit.hit_t * hit.hit_t);
          state.forward_pdf = current_vertex.forward_pdf;
        }
        if ((emitter_instance.emitter_class == EmitterClass::Directional) || (emitter_instance.emitter_class == EmitterClass::Environment)) {
          current_vertex.pdf_from_prev = wavefront_distant_emitter_area_pdf(previous_vertex.w_i, current_vertex);
        }
        wavefront_store_path_vertex(vertex_descriptor, wavefront_path_vertex_slot(from_camera, path_index, state.path_length - 1u), previous_vertex);
      }
    }
    wavefront_store_path_vertex(vertex_descriptor, wavefront_path_vertex_slot(from_camera, path_index, state.path_length), current_vertex);
  }

  if (path_tracing_neutral_depth_exceeded) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

#if ETX_ENABLE_WORK_QUEUES
  const uint material_queue_index = wavefront_material_queue_index_from_material(hit.material_index);
  if (material_queue_index != kInvalidIndex) {
    wavefront_material_queue_append(resources, from_camera, material_queue_index, dispatch_index);
  }
#endif

  state.reserved0 = 0u;
  wavefront_store_path_state(state_descriptor, path_index, state);
}
