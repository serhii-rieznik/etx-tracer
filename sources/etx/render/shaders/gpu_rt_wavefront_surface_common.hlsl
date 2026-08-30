#pragma once

#include "gpu_rt_wavefront_common.hlsl"

#if ETX_UPBP
bool wavefront_upbp_append_medium_vertex(bool from_camera, uint path_index, GPUWavefrontPathState state, GPUWavefrontPathVertex current_vertex, float3 sampled_direction,
  float phase_pdf, float reverse_phase_pdf) {
  GPUUPBPResources resources = upbp_load_resources(wavefront_load_resources());
  GPUUPBPPathState path_state = upbp_load_path_state(resources.path_state_buffer, upbp_path_state_index(resources, from_camera, path_index));
  if ((path_state.flags & GPUUPBPPathStateFlags::Valid) == 0u) {
    return false;
  }
  GPUUPBPSegment segment = upbp_load_segment(resources.segment_buffer, path_state.current_segment_index);
  GPUUPBPVertex vertex = (GPUUPBPVertex)0;
  vertex.throughput = upbp_pack_spectral_response(state.throughput);
  vertex.outgoing_throughput = upbp_pack_spectral_response(spectral_response_zero(state.spect));
  vertex.position = current_vertex.position;
  vertex.sampled_direction = sampled_direction;
  vertex.medium_index = current_vertex.medium_index;
  vertex.w_i = current_vertex.w_i;
  vertex.incident_medium_index = current_vertex.medium_index;
  vertex.outgoing_medium_index = current_vertex.medium_index;
  vertex.material_index = kInvalidIndex;
  vertex.triangle_index = kInvalidIndex;
  vertex.instance_index = kInvalidIndex;
  vertex.scatter_pdf_forward = phase_pdf;
  vertex.scatter_pdf_reverse = reverse_phase_pdf;
  vertex.log_medium_event_density = segment.log_terminal_event_density;
  vertex.eta = state.eta;
  vertex.emitter_index = kInvalidIndex;
  vertex.flags = GPUUPBPVertexFlags::Medium;
  const bool subsurface_vertex = wavefront_path_vertex_is_subsurface(current_vertex);
  if (subsurface_vertex == false) {
    vertex.flags |= GPUUPBPVertexFlags::Connectible | GPUUPBPVertexFlags::DensityConnectible;
  } else {
    const GPUWavefrontSubsurfaceState subsurface_state = wavefront_load_subsurface_state(wavefront_subsurface_state_buffer(wavefront_load_resources(), from_camera), path_index);
    if ((subsurface_state.flags & GPUWavefrontSubsurfaceFlags::InlineMedium) != 0u) {
      vertex.medium_index = upbp_inline_medium_key(subsurface_state.material_index);
      vertex.inline_scattering = upbp_pack_spectral_response(subsurface_state.scattering);
      vertex.inline_extinction = upbp_pack_spectral_response(subsurface_state.extinction);
      vertex.inline_phase_function_g = subsurface_state.phase_function_g;
      vertex.flags |= GPUUPBPVertexFlags::InlineMedium;
    }
  }
  if (upbp_append_physical_vertex(resources, from_camera, path_index, path_state, vertex) == false) {
    if (upbp_terminate_degenerate_arrival(resources, from_camera, path_index, path_state)) {
      return false;
    }
    if (upbp_mark_failed_path(resources, from_camera, path_index, GPUUPBPPathFailure::AppendMediumVertex)) {
      RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(resources.counter_buffer);
      counters.Store(GPUUPBPCounterIndex::FirstFailureDetail0 * 4u, path_state.recursive_state.failure);
      counters.Store(GPUUPBPCounterIndex::FirstFailureDetail1 * 4u, path_state.recursive_state.failure_vertex_index);
      const GPUUPBPRecursiveWeights weights = path_state.recursive_state.weights;
      const uint finite_mask = (isfinite(weights.log_d_shared) ? 1u : 0u) | (isfinite(weights.log_d_bpt) ? 2u : 0u) | (isfinite(weights.log_d_pde) ? 4u : 0u) |
                               (isfinite(weights.log_ray_sample_forward_pdf_inverse) ? 8u : 0u) |
                               (isfinite(weights.log_ray_sample_reverse_pdf_inverse) ? 16u : 0u) | (isfinite(weights.log_ray_sample_forward_ratio) ? 32u : 0u) |
                               (isfinite(weights.log_ray_sample_reverse_ratio) ? 64u : 0u);
      counters.Store(GPUUPBPCounterIndex::FirstFailureDetail2 * 4u, finite_mask);
      counters.Store(GPUUPBPCounterIndex::FirstFailureDetail3 * 4u, asuint(segment.distance));
    }
    return false;
  }
  return true;
}
#endif

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
    mis_weight = 1.0f / (1.0f + state.forward_pdf * emitter_pdfs.y + state.reverse_pdf * emitter_pdfs.x * emitter_pdfs.y);
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
  (void)directly_visible;
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
    const float cosine = max(0.0f, dot(dp, geo_normal)) / sqrt(distance_squared);
    if (cosine > kEpsilon) {
      const float exponent = scene_math_shared_collimation_to_exponent(material.emission_collimation);
      pdf_dir = pdf_area * distance_squared / cosine;
      pdf_dir_out = pdf_area * scene_math_shared_collimated_direction_pdf(cosine, exponent);
      const float emission_scale = scene_math_shared_collimated_emission_scale(cosine, exponent);
      return spectral_response_mul(evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, uv, spect), emission_scale);
    }
  }

  return spectral_response_zero(spect);
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
    if (scene_path_mode_is_path_tracing()) {
      float p_connect = emitter_discrete_pdf(emitter_index) * pdf_dir;
      mis_weight = previous_connectible ? power_heuristic(previous_vertex.sampled_bsdf_pdf, p_connect) : 1.0f;
    } else {
      float emitter_sample_pdf = emitter_discrete_pdf(emitter_index);
      float w_camera = current_vertex.forward_pdf * pdf_area * emitter_sample_pdf + current_vertex.reverse_pdf * pdf_dir_out * emitter_sample_pdf;
      mis_weight = 1.0f / (1.0f + w_camera);
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

float wavefront_medium_connect_camera_weight(Camera camera, CameraFilmSampleShared camera_sample, MediumAccess medium_access, GPUWavefrontPathMeta path_meta,
  GPUWavefrontPathVertex current_vertex) {
  if ((scene_multiple_importance_sampling_enabled() == false) || scene_path_mode_is_light_tracing()) {
    return 1.0f;
  }

  float current_from_camera_dir = camera_shared_film_pdf_out(camera, current_vertex.position);
  float current_from_camera = wavefront_convert_solid_angle_pdf_to_area(current_from_camera_dir, camera_sample.position, current_vertex.position, false, float3(0.0f, 0.0f, 0.0f));

  float reverse_phase_pdf = gpu_medium_phase_function(medium_access, -camera_sample.direction, current_vertex.w_i);
  float adjacent_connection = current_vertex.forward_pdf;
  if (scene_path_mode_uses_bdpt_fast() && (path_meta.light_path_length != 1u)) {
    adjacent_connection = 0.0f;
  }
  float w_light = current_from_camera * (adjacent_connection + current_vertex.reverse_pdf * reverse_phase_pdf);
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
#if ETX_UPBP
  GPUUPBPResources upbp_resources = (GPUUPBPResources)0;
  GPUUPBPPathState upbp_path_state = (GPUUPBPPathState)0;
  const bool upbp = scene_path_mode_is_upbp();
  if (upbp) {
    upbp_resources = upbp_load_resources(resources);
    upbp_path_state = upbp_load_path_state(upbp_resources.path_state_buffer, upbp_path_state_index(upbp_resources, false, path_index));
    if (((upbp_path_state.flags & GPUUPBPPathStateFlags::Valid) == 0u) || (upbp_path_state.path_length == 0u)) {
      return;
    }
    seed = upbp_deterministic_seed(upbp_path_state.global_path_index, 1u, upbp_path_state.path_length + 1u, kUPBPRandomDomainFilmConnection);
  }
#else
  const bool upbp = false;
#endif
  float2 lens_rnd = float2(0.0f, 0.0f);
  if (camera_lens_sampling_enabled(camera.lens_radius, camera.focal_distance)) {
    lens_rnd = float2(rnd01(seed), rnd01(seed));
  }
  float2 sensor_sample = camera_sample_lens_uv(camera, lens_rnd) * camera.lens_radius;
  float3 lens_point = camera_film_shared_lens_point(camera, sensor_sample);
  CameraFilmSampleShared camera_sample = camera_film_shared_evaluate(camera, current_vertex.position, lens_point);
  if (upbp == false) {
    state.sampler_seed = seed;
  }
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

  float mis_weight = wavefront_medium_connect_camera_weight(camera, camera_sample, medium_access, path_meta, current_vertex);
  float scattering_pdf_reverse = 0.0f;
  float camera_area_density = 0.0f;
#if ETX_UPBP
  if (upbp) {
    mis_weight = 1.0f;
    scattering_pdf_reverse = gpu_medium_phase_function(medium_access, -camera_sample.direction, current_vertex.w_i);
    camera_area_density = wavefront_convert_solid_angle_pdf_to_area(camera_sample.pdf_dir_out, camera_sample.position, current_vertex.position, false, float3(0.0f, 0.0f, 0.0f));
  }
#endif
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
  task.mis_weight = upbp ? phase_value : mis_weight;
  task.upbp_scattering_pdf_reverse_bits = asuint(scattering_pdf_reverse);
  task.upbp_camera_area_density_bits = asuint(camera_area_density);
  task.pixel_index = pixel_index;
  task.medium_index = current_vertex.medium_index;
  task.inline_medium_extinction = current_vertex.inline_medium_extinction;
  task.inline_medium_flags = current_vertex.inline_medium_flags;
  task.flags = GPUWavefrontPointConnectionTaskFlags::Ready | GPUWavefrontPointConnectionTaskFlags::SourceMedium;
  task.path_index = path_index;
#if ETX_UPBP
  if (upbp) {
    task.sampler_seed = upbp_deterministic_seed(upbp_path_state.global_path_index, 1u, upbp_path_state.path_length + 1u, kUPBPRandomDomainIntersectionTraversal);
    task.upbp_auxiliary1_bits = upbp_deterministic_seed(upbp_path_state.global_path_index, 1u, upbp_path_state.path_length + 1u, kUPBPRandomDomainConnectionTransmittance);
  } else
#endif
  {
    task.sampler_seed = state.sampler_seed;
  }
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
    if (from_camera && (scene_path_mode_is_light_tracing() == false) && (scene_path_mode_is_upbp() == false) && scene_strategy_enabled(kSceneStrategyDirectHit) &&
        direct_hit_path_length_enabled) {
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

    const bool upbp_path = scene_path_mode_is_upbp();
    float2 sample_random = float2(rnd01(state.sampler_seed), rnd01(state.sampler_seed));
    float2 connection_random = (float2)0;
    float2 support_random = (float2)0;
    if (upbp_path == false) {
      connection_random = float2(rnd01(state.sampler_seed), rnd01(state.sampler_seed));
      support_random = float2(rnd01(state.sampler_seed), rnd01(state.sampler_seed));
    }
    if ((upbp_path == false) && from_camera && (state.path_length == 1u)) {
      const bool use_blue_noise_bsdf = sample_use_blue_noise_primary(constants.sample_index, kSamplerStreamBSDF);
      const bool use_blue_noise_connection = sample_use_blue_noise_primary(constants.sample_index, kSamplerStreamConnection);
      const bool use_blue_noise_support = sample_use_blue_noise_primary(constants.sample_index, kSamplerStreamSupport);
      uint2 sample_pixel = state.pixel;
      if (use_blue_noise_bsdf || use_blue_noise_connection || use_blue_noise_support) {
        sample_pixel = sample_blue_noise_translated_pixel(state.pixel);
      }
      if (use_blue_noise_bsdf) {
        const uint bsdf_dimension = sampler_stream_dimension_base(kSamplerStreamBSDF);
        sample_random = float2(sample_blue_noise_value_at_translated_pixel(sample_pixel, constants.sample_index, bsdf_dimension + 0u),
          sample_blue_noise_value_at_translated_pixel(sample_pixel, constants.sample_index, bsdf_dimension + 1u));
      }
      if (use_blue_noise_connection) {
        const uint connection_dimension = sampler_stream_dimension_base(kSamplerStreamConnection);
        connection_random = float2(sample_blue_noise_value_at_translated_pixel(sample_pixel, constants.sample_index, connection_dimension + 0u),
          sample_blue_noise_value_at_translated_pixel(sample_pixel, constants.sample_index, connection_dimension + 1u));
      }
      if (use_blue_noise_support) {
        const uint support_dimension = sampler_stream_dimension_base(kSamplerStreamSupport);
        support_random = float2(sample_blue_noise_value_at_translated_pixel(sample_pixel, constants.sample_index, support_dimension + 0u),
          sample_blue_noise_value_at_translated_pixel(sample_pixel, constants.sample_index, support_dimension + 1u));
      }
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
#if ETX_UPBP
    if (scene_path_mode_is_upbp() &&
        (wavefront_upbp_append_medium_vertex(from_camera, path_index, state, current_vertex, sampled_direction, phase_pdf, reverse_phase_pdf) == false)) {
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }
#endif
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
    float connection_source = current_d_vcm;
    if (scene_path_mode_uses_bdpt_fast()) {
      connection_source = (state.path_length == 1u) ? current_d_vcm : 0.0f;
    }
    state.reverse_pdf = wavefront_safe_div((current_d_vc * reverse_phase_pdf) + connection_source, phase_pdf);
    state.d_vm = scene_path_mode_is_vcm() ? wavefront_safe_div(current_d_vm * reverse_phase_pdf, phase_pdf) : 0.0f;
    state.sampled_bsdf_pdf = phase_pdf;
    state.ray.o = current_vertex.position;
    state.ray.d = normalize(sampled_direction);
    state.ray.min_t = scene_path_mode_is_upbp() ? 0.0f : kRayEpsilon;
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
        if ((scene_path_mode_is_bdpt_full() || scene_path_mode_uses_bdpt_fast()) && (cos_to_prev > 0.0f)) {
          // A first distant-emitter segment has no finite-distance squared
          // Jacobian; only its surface cosine is folded into d_vcm.
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
