#pragma once

#include "gpu_rt_wavefront_common.hlsl"

float wavefront_surface_shading_pdf_environment(float3 direction, bool target_is_surface, float3 target_geo_normal) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  uint emitter_instance_count = 0u;
  uint environment_count = 0u;
  if (emitter_access_try_load_environment_state(context, emitter_instance_count, environment_count) == false) {
    return 0.0f;
  }
  (void)emitter_instance_count;

  float pdf_dir = 0.0f;
  for (uint i = 0u; i < environment_count; ++i) {
    uint emitter_index = kInvalidIndex;
    if (emitter_access_try_load_environment_emitter(context, i, emitter_index)) {
      pdf_dir += emitter_discrete_pdf(emitter_index) * gpu_distant_emission_area_pdf(emitter_index);
    }
  }

  if (environment_count == 0u) {
    return 0.0f;
  }

  float normal_factor = target_is_surface ? abs(dot(target_geo_normal, direction)) : 1.0f;
  return normal_factor * (pdf_dir / float(environment_count));
}

bool wavefront_trace_surface_path_compact(RayDesc ray, SpectralQuery spect, inout uint medium_index, inout uint seed, out TraceSurfaceResult result);
bool wavefront_trace_path_state(bool from_camera, uint path_index, RayDesc ray, SpectralQuery spect, SpectralResponse throughput, inout uint medium_index, inout uint seed,
  out TraceSurfaceResult result);
bool wavefront_trace_closest_surface_or_boundary(RayDesc ray, inout uint seed, out TraceSurfaceResult result, out bool boundary_hit, out uint boundary_medium_index);
SurfacePoint wavefront_load_surface_point_compact(TriangleData tri, float2 bary, float3 ray_dir, uint instance_index);
float3 wavefront_trace_surface_shading_position(TraceSurfaceResult surface_hit, float3 outgoing_direction);

bool wavefront_subsurface_state_active(GPUWavefrontResources resources, bool from_camera, uint path_index, out GPUWavefrontSubsurfaceState state) {
  state = (GPUWavefrontSubsurfaceState)0;
  uint state_buffer = wavefront_subsurface_state_buffer(resources, from_camera);
  if (state_buffer == kInvalidIndex) {
    return false;
  }

  state = wavefront_load_subsurface_state(state_buffer, path_index);
  return (state.flags & GPUWavefrontSubsurfaceFlags::Active) != 0u;
}

bool wavefront_trace_transmittance_to_point(float3 origin, float3 target, SpectralQuery spect, uint medium_index, inout uint seed, out SpectralResponse transmittance) {
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
  const float t_max_epsilon = max(kRayEpsilon, distance * kRayEpsilon);
  ray.TMax = max(ray.TMin, distance - t_max_epsilon);

  bool has_geometry_buffers = (constants.scene.triangles != kInvalidIndex) && (constants.scene.vertex_positions != kInvalidIndex) &&
                              (constants.scene.vertex_normals != kInvalidIndex) && (constants.scene.scene_globals != kInvalidIndex);
  if (has_geometry_buffers == false) {
    return true;
  }

  SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  uint vertex_count = scene_globals_data.vertex_count;
  uint triangle_count = scene_globals_data.triangle_count;
  bool has_material_buffer = constants.scene.materials != kInvalidIndex;
  bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;

  const uint kBoundaryCapacity = 62u;
  float boundary_t[kBoundaryCapacity];
  uint boundary_medium[kBoundaryCapacity];
  uint boundary_count = 0u;

  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> ray_query;
  ray_query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(constants.as_index)], RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);

  while (ray_query.Proceed()) {
    if (ray_query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
      continue;
    }

    const uint candidate_instance_index = ray_query.CandidateInstanceID();
    uint candidate_triangle_index = scene_instance_triangle_index(ray_query.CandidatePrimitiveIndex(), candidate_instance_index);
    if (candidate_triangle_index >= triangle_count) {
      continue;
    }

    TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], candidate_triangle_index);
    bool valid_indices = (tri.i.x < vertex_count) && (tri.i.y < vertex_count) && (tri.i.z < vertex_count);
    if (valid_indices == false) {
      continue;
    }

    float2 candidate_bary = ray_query.CandidateTriangleBarycentrics();
    float2 candidate_uv = float2(0.0f, 0.0f);
    if (has_texcoords) {
      candidate_uv = interpolate_uv(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri, candidate_bary);
    }

    MaterialAccess material_access = ETX_ZERO(MaterialAccess);
    if (has_material_buffer) {
      MaterialAccessGPUContext material_context = {constants.scene.materials};
      material_access_try_load(material_context, tri.material_index, material_access);
    }

    bool alpha_rejected = alpha_test_pass(tri.material_index, candidate_uv, seed);
    const float3 candidate_geo_normal = scene_instance_transform_geometric_normal(load_scene_instance(candidate_instance_index), tri.geo_n);
    bool entering_surface = dot(candidate_geo_normal, ray.Direction) < 0.0f;
    HitPolicyDecision hit_policy = hit_policy_evaluate(HitPolicyMode::MediumTransmittance, material_access.material_class, alpha_rejected, entering_surface,
      material_access.int_medium_index, material_access.ext_medium_index);
    if (hit_policy.action == HitPolicyAction::Ignore) {
      continue;
    }

    if (hit_policy.action == HitPolicyAction::Occlude) {
      return false;
    }

    if (hit_policy.action == HitPolicyAction::TransitionMedium) {
      if (boundary_count >= kBoundaryCapacity) {
        return false;
      }

      boundary_t[boundary_count] = ray_query.CandidateTriangleRayT();
      boundary_medium[boundary_count] = hit_policy.medium_index;
      boundary_count += 1u;
    }
  }

  for (uint i = 0u; i < boundary_count; ++i) {
    for (uint j = i + 1u; j < boundary_count; ++j) {
      if (boundary_t[i] > boundary_t[j]) {
        float swap_t = boundary_t[i];
        boundary_t[i] = boundary_t[j];
        boundary_t[j] = swap_t;

        uint swap_medium = boundary_medium[i];
        boundary_medium[i] = boundary_medium[j];
        boundary_medium[j] = swap_medium;
      }
    }
  }

  float current_t = 0.0f;
  uint current_medium_index = medium_index;
  for (uint boundary_index = 0u; boundary_index < boundary_count; ++boundary_index) {
    float segment_end_t = boundary_t[boundary_index];
    float segment_distance = max(0.0f, segment_end_t - current_t);
    float3 segment_origin = origin + ray.Direction * current_t;
    SpectralResponse segment_transmittance = medium_segment_transmittance_spectral(current_medium_index, segment_origin, ray.Direction, segment_distance, spect, seed);
    transmittance = spectral_response_mul(transmittance, segment_transmittance);
    current_medium_index = boundary_medium[boundary_index];
    current_t = segment_end_t;
  }

  float final_segment_distance = max(0.0f, ray.TMax - current_t);
  float3 final_segment_origin = origin + ray.Direction * current_t;
  SpectralResponse final_segment_transmittance =
    medium_segment_transmittance_spectral(current_medium_index, final_segment_origin, ray.Direction, final_segment_distance, spect, seed);
  transmittance = spectral_response_mul(transmittance, final_segment_transmittance);
  return true;
}

#if ETX_UPBP
bool wavefront_trace_closest_surface_or_boundary_with_origin_retry(RayDesc ray, inout uint seed, out TraceSurfaceResult result, out bool boundary_hit,
  out uint boundary_medium_index);

bool wavefront_upbp_trace_connection_to_point(float3 origin, float3 target, SpectralQuery spect, uint medium_index, SpectralResponse inline_extinction, uint inline_flags,
  bool source_is_medium, inout uint intersection_seed, inout uint medium_seed, out bool visible, out GPUUPBPConnectionInterval connection, out uint failure) {
  visible = false;
  failure = GPUUPBPConnectionTrackingFailure::None;
  connection = (GPUUPBPConnectionInterval)0;
  connection.weight = spectral_response_make(spect, 1.0f);

  float3 delta = target - origin;
  const float distance = length(delta);
  if (distance <= kRayEpsilon) {
    visible = true;
    return true;
  }
  const float3 direction = delta / distance;
  float3 current_origin = origin;
  uint current_medium_index = medium_index;
  SpectralResponse current_inline_extinction = inline_extinction;
  uint current_inline_flags = inline_flags;
  uint boundary_count = 0u;
  bool retry_from_source = source_is_medium;
  for (;;) {
    const float target_distance = dot(target - current_origin, direction);
    const float t_max_epsilon = max(kRayEpsilon, distance * kRayEpsilon);
    if (target_distance <= t_max_epsilon) {
      visible = true;
      return true;
    }

    RayDesc ray = (RayDesc)0;
    ray.Origin = current_origin;
    ray.Direction = direction;
    ray.TMin = retry_from_source ? 0.0f : kRayEpsilon;
    ray.TMax = max(ray.TMin, target_distance - t_max_epsilon);

    TraceSurfaceResult trace_result = (TraceSurfaceResult)0;
    bool boundary_hit = false;
    uint boundary_medium = kInvalidIndex;
    const bool found_hit = retry_from_source ? wavefront_trace_closest_surface_or_boundary_with_origin_retry(ray, intersection_seed, trace_result, boundary_hit, boundary_medium)
                                             : wavefront_trace_closest_surface_or_boundary(ray, intersection_seed, trace_result, boundary_hit, boundary_medium);
    const float interval_distance = found_hit ? trace_result.hit_t : ray.TMax;
    if (isfinite(interval_distance) == false) {
      failure = GPUUPBPConnectionTrackingFailure::InvalidIntervalDistance;
      return false;
    }
    if (interval_distance <= 0.0f) {
      if (found_hit == false) {
        failure = GPUUPBPConnectionTrackingFailure::InvalidIntervalDistance;
        return false;
      }
      if (boundary_hit == false) {
        return true;
      }
      boundary_count += 1u;
      GPUUPBPResources upbp_resources = upbp_load_resources(wavefront_load_resources());
      if (boundary_count > upbp_resources.iteration.maximum_boundary_count) {
        failure = GPUUPBPConnectionTrackingFailure::BoundaryLimit;
        return false;
      }
      const float boundary_side = dot(trace_result.surface_point.geo_normal, direction) >= 0.0f ? 1.0f : -1.0f;
      current_origin = offset_ray(trace_result.surface_point.vertex.pos, trace_result.surface_point.geo_normal * boundary_side);
      current_medium_index = boundary_medium;
      current_inline_extinction = spectral_response_make(spect, 0.0f);
      current_inline_flags = 0u;
      retry_from_source = false;
      continue;
    }

    GPUUPBPConnectionInterval interval = (GPUUPBPConnectionInterval)0;
    uint terminal_type = kUPBPMediumFailure;
    GPUUPBPResources upbp_resources = upbp_load_resources(wavefront_load_resources());
    bool tracking_valid = false;
    if (current_medium_index != kInvalidIndex) {
      tracking_valid = upbp_track_connection_interval(current_medium_index, current_origin, direction, interval_distance, spect,
        upbp_resources.iteration.maximum_null_events_per_interval, medium_seed, interval, terminal_type);
    } else if ((current_inline_flags & GPUWavefrontSubsurfaceFlags::InlineMedium) != 0u) {
      tracking_valid = upbp_track_connection_homogeneous_extinction(current_inline_extinction, interval_distance, spect, upbp_resources.iteration.maximum_null_events_per_interval,
        medium_seed, interval, terminal_type);
    } else {
      tracking_valid = upbp_track_connection_interval(kInvalidIndex, current_origin, direction, interval_distance, spect, upbp_resources.iteration.maximum_null_events_per_interval,
        medium_seed, interval, terminal_type);
    }
    if (tracking_valid == false) {
      failure = GPUUPBPConnectionTrackingFailure::MediumTracking;
      return false;
    }
    connection.weight = spectral_response_mul(connection.weight, interval.weight);
    connection.log_transport_pdf_forward += interval.log_transport_pdf_forward;
    connection.log_transport_pdf_reverse += interval.log_transport_pdf_reverse;
    if ((terminal_type == kUPBPMediumScatter) || (terminal_type == kUPBPMediumAbsorb)) {
      return true;
    }
    if (terminal_type != kUPBPMediumEscape) {
      failure = GPUUPBPConnectionTrackingFailure::InvalidTerminal;
      return false;
    }
    if (found_hit == false) {
      visible = true;
      return true;
    }
    if (boundary_hit == false) {
      return true;
    }

    boundary_count += 1u;
    if (boundary_count > upbp_resources.iteration.maximum_boundary_count) {
      failure = GPUUPBPConnectionTrackingFailure::BoundaryLimit;
      return false;
    }
    const float boundary_side = dot(trace_result.surface_point.geo_normal, direction) >= 0.0f ? 1.0f : -1.0f;
    current_origin = offset_ray(trace_result.surface_point.vertex.pos, trace_result.surface_point.geo_normal * boundary_side);
    current_medium_index = boundary_medium;
    current_inline_extinction = spectral_response_make(spect, 0.0f);
    current_inline_flags = 0u;
    retry_from_source = false;
  }
}
#endif

bool wavefront_trace_transmittance_to_point_inline_medium(float3 origin, float3 target, SpectralQuery spect, uint medium_index, SpectralResponse inline_extinction,
  uint inline_flags, inout uint seed, out SpectralResponse transmittance) {
  transmittance = spectral_response_make(spect, 1.0f);
  float3 current_origin = origin;
  uint current_medium_index = medium_index;
  SpectralResponse current_inline_extinction = inline_extinction;
  uint current_inline_flags = inline_flags;

  while (true) {
    float3 delta = target - current_origin;
    float distance = length(delta);
    if (distance <= kRayEpsilon) {
      return true;
    }

    RayDesc ray = (RayDesc)0;
    ray.Origin = current_origin;
    ray.Direction = delta / distance;
    ray.TMin = kRayEpsilon;
    const float t_max_epsilon = max(kRayEpsilon, distance * kRayEpsilon);
    ray.TMax = max(ray.TMin, distance - t_max_epsilon);

    TraceSurfaceResult trace_result = (TraceSurfaceResult)0;
    bool boundary_hit = false;
    uint boundary_medium = kInvalidIndex;
    bool found_hit = wavefront_trace_closest_surface_or_boundary(ray, seed, trace_result, boundary_hit, boundary_medium);
    float segment_distance = found_hit ? trace_result.hit_t : ray.TMax;
    SpectralResponse segment_transmittance = spectral_response_make(spect, 1.0f);
    if (current_medium_index != kInvalidIndex) {
      segment_transmittance = medium_segment_transmittance_spectral(current_medium_index, current_origin, ray.Direction, segment_distance, spect, seed);
    } else if ((current_inline_flags & GPUWavefrontSubsurfaceFlags::InlineMedium) != 0u) {
      segment_transmittance = spectral_response_exp(spectral_response_mul(current_inline_extinction, -segment_distance));
    }
    transmittance = spectral_response_mul(transmittance, segment_transmittance);

    if (found_hit == false) {
      return true;
    }

    if (boundary_hit == false) {
      return false;
    }

    current_medium_index = boundary_medium;
    current_inline_flags = 0u;
    current_inline_extinction = spectral_response_make(spect, 0.0f);
    current_origin = trace_result.surface_point.vertex.pos;
  }
}

bool wavefront_trace_closest_surface_or_boundary(RayDesc ray, inout uint seed, out TraceSurfaceResult result, out bool boundary_hit, out uint boundary_medium_index) {
  result = (TraceSurfaceResult)0;
  result.triangle_index = kInvalidIndex;
  result.emitter_index = kInvalidIndex;
  result.hit_t = ray.TMax;
  boundary_hit = false;
  boundary_medium_index = kInvalidIndex;

  bool has_geometry_buffers = (constants.scene.triangles != kInvalidIndex) && (constants.scene.vertex_positions != kInvalidIndex) &&
                              (constants.scene.vertex_normals != kInvalidIndex) && (constants.scene.scene_globals != kInvalidIndex);
  if (has_geometry_buffers == false) {
    return false;
  }

  SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  uint vertex_count = scene_globals_data.vertex_count;
  uint triangle_count = scene_globals_data.triangle_count;
  bool has_material_buffer = constants.scene.materials != kInvalidIndex;
  bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;

  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> ray_query;
  ray_query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(constants.as_index)], RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);

  while (ray_query.Proceed()) {
    if (ray_query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
      continue;
    }

    const uint candidate_instance_index = ray_query.CandidateInstanceID();
    uint candidate_triangle_index = scene_instance_triangle_index(ray_query.CandidatePrimitiveIndex(), candidate_instance_index);
    if (candidate_triangle_index >= triangle_count) {
      continue;
    }

    TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], candidate_triangle_index);
    bool valid_indices = (tri.i.x < vertex_count) && (tri.i.y < vertex_count) && (tri.i.z < vertex_count);
    if (valid_indices == false) {
      continue;
    }

    float2 candidate_bary = ray_query.CandidateTriangleBarycentrics();
    float2 candidate_uv = float2(0.0f, 0.0f);
    if (has_texcoords) {
      candidate_uv = interpolate_uv(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri, candidate_bary);
    }

    MaterialAccess material_access = ETX_ZERO(MaterialAccess);
    if (has_material_buffer) {
      MaterialAccessGPUContext material_context = {constants.scene.materials};
      material_access_try_load(material_context, tri.material_index, material_access);
    }

    bool alpha_rejected = alpha_test_pass(tri.material_index, candidate_uv, seed);
    const float3 candidate_geo_normal = scene_instance_transform_geometric_normal(load_scene_instance(candidate_instance_index), tri.geo_n);
    bool entering_surface = dot(candidate_geo_normal, ray.Direction) < 0.0f;
    HitPolicyDecision hit_policy = hit_policy_evaluate(HitPolicyMode::KeepBoundaryHit, material_access.material_class, alpha_rejected, entering_surface,
      material_access.int_medium_index, material_access.ext_medium_index);
    if (hit_policy.action == HitPolicyAction::Ignore) {
      continue;
    }

    ray_query.CommitNonOpaqueTriangleHit();
  }

  if (ray_query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
    return false;
  }

  result.instance_index = ray_query.CommittedInstanceID();
  result.triangle_index = scene_instance_triangle_index(ray_query.CommittedPrimitiveIndex(), result.instance_index);
  result.hit_t = ray_query.CommittedRayT();
  result.tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], result.triangle_index);
  result.surface_point = wavefront_load_surface_point_compact(result.tri, ray_query.CommittedTriangleBarycentrics(), ray.Direction, result.instance_index);
  result.emitter_index = scene_instance_emitter_index(result.triangle_index, result.instance_index);
  if (try_load_material_full(result.tri.material_index, result.material)) {
    surface_point_apply_material_normal_map(result.surface_point, result.material, ray.Direction);
  }
  result.hit = 1u;

  boundary_hit = result.material.cls == MaterialClass::Boundary;
  if (boundary_hit) {
    bool entering_surface = dot(result.surface_point.geo_normal, ray.Direction) < 0.0f;
    boundary_medium_index = entering_surface ? result.material.int_medium : result.material.ext_medium;
  }

  return true;
}

SurfacePoint wavefront_load_surface_point_compact(TriangleData tri, float2 bary, float3 ray_dir, uint instance_index) {
  SurfacePoint result = (SurfacePoint)0;
  result.barycentrics = barycentrics(bary);

  float3 position_0 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)], tri.i.x);
  float3 position_1 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)], tri.i.y);
  float3 position_2 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)], tri.i.z);

  float3 normal_0 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)], tri.i.x);
  float3 normal_1 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)], tri.i.y);
  float3 normal_2 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)], tri.i.z);

  bool has_surface_frame = (constants.scene.vertex_tangents != kInvalidIndex) && (constants.scene.vertex_bitangents != kInvalidIndex);
  float3 tangent_0 = float3(0.0f, 0.0f, 0.0f);
  float3 tangent_1 = float3(0.0f, 0.0f, 0.0f);
  float3 tangent_2 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_0 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_1 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_2 = float3(0.0f, 0.0f, 0.0f);
  if (has_surface_frame) {
    tangent_0 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_tangents)], tri.i.x);
    tangent_1 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_tangents)], tri.i.y);
    tangent_2 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_tangents)], tri.i.z);
    bitangent_0 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_bitangents)], tri.i.x);
    bitangent_1 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_bitangents)], tri.i.y);
    bitangent_2 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_bitangents)], tri.i.z);
  }

  bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;
  float2 texcoord_0 = float2(0.0f, 0.0f);
  float2 texcoord_1 = float2(0.0f, 0.0f);
  float2 texcoord_2 = float2(0.0f, 0.0f);
  if (has_texcoords) {
    texcoord_0 = load_float2(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri.i.x);
    texcoord_1 = load_float2(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri.i.y);
    texcoord_2 = load_float2(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri.i.z);
  }

  surface_point_shared_interpolate_vertex(position_0, position_1, position_2, normal_0, normal_1, normal_2, tangent_0, tangent_1, tangent_2, bitangent_0, bitangent_1, bitangent_2,
    texcoord_0, texcoord_1, texcoord_2, result.barycentrics, has_surface_frame, has_texcoords, result.vertex);
  const GPUSceneInstanceData instance = load_scene_instance(instance_index);
  result.vertex = scene_instance_transform_vertex(instance, result.vertex);
  result.geo_normal = scene_instance_transform_geometric_normal(instance, tri.geo_n);
  scene_math_shared_finalize_shading_frame(result.vertex.nrm, result.vertex.nrm, result.vertex.tan, result.vertex.btn, result.geo_normal, ray_dir, result.vertex.nrm,
    result.vertex.tan, result.vertex.btn);
  return result;
}

float3 wavefront_trace_surface_shading_position(TraceSurfaceResult surface_hit, float3 outgoing_direction) {
  if (surface_hit.triangle_index == kInvalidIndex) {
    float sign_value = (dot(surface_hit.surface_point.geo_normal, outgoing_direction) >= 0.0f) ? 1.0f : -1.0f;
    return offset_ray(surface_hit.surface_point.vertex.pos, surface_hit.surface_point.geo_normal * sign_value);
  }

  ByteAddressBuffer position_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)];
  ByteAddressBuffer normal_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)];

  float3 p0 = load_float3(position_buffer, surface_hit.tri.i.x);
  float3 p1 = load_float3(position_buffer, surface_hit.tri.i.y);
  float3 p2 = load_float3(position_buffer, surface_hit.tri.i.z);
  float3 n0 = load_float3(normal_buffer, surface_hit.tri.i.x);
  float3 n1 = load_float3(normal_buffer, surface_hit.tri.i.y);
  float3 n2 = load_float3(normal_buffer, surface_hit.tri.i.z);

  const GPUSceneInstanceData instance = load_scene_instance(surface_hit.instance_index);
  const float orientation = (instance.flags & 1u) != 0u ? -1.0f : 1.0f;
  p0 = scene_instance_transform_point(instance, p0);
  p1 = scene_instance_transform_point(instance, p1);
  p2 = scene_instance_transform_point(instance, p2);
  n0 = scene_instance_transform_normal(instance, n0) * orientation;
  n1 = scene_instance_transform_normal(instance, n1) * orientation;
  n2 = scene_instance_transform_normal(instance, n2) * orientation;
  const float3 geo_normal = scene_instance_transform_geometric_normal(instance, surface_hit.tri.geo_n);
  return scene_math_shared_shading_pos(p0, p1, p2, n0, n1, n2, geo_normal, surface_hit.surface_point.barycentrics, outgoing_direction);
}

bool wavefront_trace_surface_path_compact(RayDesc ray, SpectralQuery spect, inout uint medium_index, inout uint seed, out TraceSurfaceResult result) {
  result = (TraceSurfaceResult)0;
  result.medium_index = medium_index;
  result.triangle_index = kInvalidIndex;
  result.emitter_index = kInvalidIndex;
  result.hit_t = ray.TMax;
  result.transmittance = spectral_response_make(spect, 1.0f);

  bool has_geometry_buffers = (constants.scene.triangles != kInvalidIndex) && (constants.scene.vertex_positions != kInvalidIndex) &&
                              (constants.scene.vertex_normals != kInvalidIndex) && (constants.scene.scene_globals != kInvalidIndex);
  if (has_geometry_buffers == false) {
    return false;
  }

  SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  uint vertex_count = scene_globals_data.vertex_count;
  uint triangle_count = scene_globals_data.triangle_count;
  bool has_material_buffer = constants.scene.materials != kInvalidIndex;
  bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;

  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> ray_query;
  ray_query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(constants.as_index)], RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);

  float medium_segment_start_t = ray.TMin;
  uint ray_medium_index = medium_index;
  while (ray_query.Proceed()) {
    if (ray_query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
      continue;
    }

    const uint candidate_instance_index = ray_query.CandidateInstanceID();
    uint candidate_triangle_index = scene_instance_triangle_index(ray_query.CandidatePrimitiveIndex(), candidate_instance_index);
    if (candidate_triangle_index >= triangle_count) {
      continue;
    }

    float candidate_t = ray_query.CandidateTriangleRayT();
    if (candidate_t > medium_segment_start_t) {
      float segment_distance = candidate_t - medium_segment_start_t;
      float3 segment_origin = ray.Origin + ray.Direction * medium_segment_start_t;
      SpectralResponse segment_transmittance = medium_segment_transmittance_spectral(ray_medium_index, segment_origin, ray.Direction, segment_distance, spect, seed);
      result.transmittance = spectral_response_mul(result.transmittance, segment_transmittance);
      medium_segment_start_t = candidate_t;
    }

    TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], candidate_triangle_index);
    bool valid_indices = (tri.i.x < vertex_count) && (tri.i.y < vertex_count) && (tri.i.z < vertex_count);
    if (valid_indices == false) {
      continue;
    }

    float2 candidate_bary = ray_query.CandidateTriangleBarycentrics();
    float2 candidate_uv = float2(0.0f, 0.0f);
    if (has_texcoords) {
      candidate_uv = interpolate_uv(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri, candidate_bary);
    }

    MaterialAccess material_access = ETX_ZERO(MaterialAccess);
    if (has_material_buffer) {
      MaterialAccessGPUContext material_context = {constants.scene.materials};
      material_access_try_load(material_context, tri.material_index, material_access);
    }

    bool alpha_rejected = alpha_test_pass(tri.material_index, candidate_uv, seed);
    const float3 candidate_geo_normal = scene_instance_transform_geometric_normal(load_scene_instance(candidate_instance_index), tri.geo_n);
    bool entering_surface = dot(candidate_geo_normal, ray.Direction) < 0.0f;
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
      ray_query.CommitNonOpaqueTriangleHit();
    }
  }

  float medium_segment_end_t = (ray_query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? ray_query.CommittedRayT() : ray.TMax;
  if (medium_segment_end_t > medium_segment_start_t) {
    float segment_distance = medium_segment_end_t - medium_segment_start_t;
    float3 segment_origin = ray.Origin + ray.Direction * medium_segment_start_t;
    SpectralResponse segment_transmittance = medium_segment_transmittance_spectral(ray_medium_index, segment_origin, ray.Direction, segment_distance, spect, seed);
    result.transmittance = spectral_response_mul(result.transmittance, segment_transmittance);
  }

  medium_index = ray_medium_index;
  result.medium_index = ray_medium_index;

  if (ray_query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
    return false;
  }

  result.instance_index = ray_query.CommittedInstanceID();
  result.triangle_index = scene_instance_triangle_index(ray_query.CommittedPrimitiveIndex(), result.instance_index);
  result.hit_t = ray_query.CommittedRayT();
  result.tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], result.triangle_index);
  result.surface_point = wavefront_load_surface_point_compact(result.tri, ray_query.CommittedTriangleBarycentrics(), ray.Direction, result.instance_index);
  result.emitter_index = scene_instance_emitter_index(result.triangle_index, result.instance_index);
  if (try_load_material_full(result.tri.material_index, result.material)) {
    surface_point_apply_material_normal_map(result.surface_point, result.material, ray.Direction);
  }
  result.hit = 1u;
  return true;
}

bool wavefront_try_sample_medium_segment(uint medium_index, float3 segment_origin, float3 ray_direction, float segment_distance, SpectralQuery spect, SpectralResponse throughput,
  inout uint seed, out MediumSample medium_sample) {
  medium_sample = (MediumSample)0;
  if ((segment_distance <= 0.0f) || (medium_index == kInvalidIndex)) {
    medium_sample.weight = spectral_response_make(spect, 1.0f);
    return false;
  }

  MediumAccess medium_access = (MediumAccess)0;
  if (wavefront_try_load_medium(medium_index, medium_access) == false) {
    medium_sample.weight = spectral_response_make(spect, 1.0f);
    return false;
  }

  medium_sample = gpu_sample_medium(medium_access, spect, throughput, segment_origin, ray_direction, segment_distance, seed);
  return medium_sample_sampled_medium(medium_sample);
}

float wavefront_subsurface_trace_response_component(SpectralResponse value, uint channel) {
  if (spectral_response_is_spectral(value)) {
    return value.value;
  }

  if (channel == 0u) {
    return value.integrated.x;
  }
  if (channel == 1u) {
    return value.integrated.y;
  }
  return value.integrated.z;
}

float wavefront_subsurface_trace_response_sum(SpectralResponse value) {
  if (spectral_response_is_spectral(value)) {
    return value.value;
  }

  return value.integrated.x + value.integrated.y + value.integrated.z;
}

SpectralResponse wavefront_subsurface_trace_safe_mul(SpectralQuery spect, SpectralResponse a, SpectralResponse b) {
  if (spectral_query_is_spectral(spect)) {
    const float value = ((a.value == 0.0f) || (b.value == 0.0f)) ? 0.0f : (a.value * b.value);
    return spectral_response_make(spect, value);
  }

  return spectral_response_make(spect, float3(((a.integrated.x == 0.0f) || (b.integrated.x == 0.0f)) ? 0.0f : (a.integrated.x * b.integrated.x),
                                         ((a.integrated.y == 0.0f) || (b.integrated.y == 0.0f)) ? 0.0f : (a.integrated.y * b.integrated.y),
                                         ((a.integrated.z == 0.0f) || (b.integrated.z == 0.0f)) ? 0.0f : (a.integrated.z * b.integrated.z)));
}

bool wavefront_trace_subsurface_material(RayDesc ray, uint material_index, inout uint seed, out TraceSurfaceResult result) {
  result = (TraceSurfaceResult)0;
  result.triangle_index = kInvalidIndex;
  result.emitter_index = kInvalidIndex;
  result.hit_t = ray.TMax;

  bool has_geometry_buffers = (constants.scene.triangles != kInvalidIndex) && (constants.scene.vertex_positions != kInvalidIndex) &&
                              (constants.scene.vertex_normals != kInvalidIndex) && (constants.scene.scene_globals != kInvalidIndex);
  if (has_geometry_buffers == false) {
    return false;
  }

  SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  uint vertex_count = scene_globals_data.vertex_count;
  uint triangle_count = scene_globals_data.triangle_count;
  bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;

  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> ray_query;
  ray_query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(constants.as_index)], RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);

  while (ray_query.Proceed()) {
    if (ray_query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
      continue;
    }

    const uint candidate_instance_index = ray_query.CandidateInstanceID();
    uint candidate_triangle_index = scene_instance_triangle_index(ray_query.CandidatePrimitiveIndex(), candidate_instance_index);
    if (candidate_triangle_index >= triangle_count) {
      continue;
    }

    TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], candidate_triangle_index);
    if ((material_index != kInvalidIndex) && (tri.material_index != material_index)) {
      continue;
    }

    bool valid_indices = (tri.i.x < vertex_count) && (tri.i.y < vertex_count) && (tri.i.z < vertex_count);
    if (valid_indices == false) {
      continue;
    }

    float2 candidate_bary = ray_query.CandidateTriangleBarycentrics();
    float2 candidate_uv = float2(0.0f, 0.0f);
    if (has_texcoords) {
      candidate_uv = interpolate_uv(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri, candidate_bary);
    }
    if (alpha_test_pass(tri.material_index, candidate_uv, seed)) {
      continue;
    }

    ray_query.CommitNonOpaqueTriangleHit();
  }

  if (ray_query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
    return false;
  }

  result.instance_index = ray_query.CommittedInstanceID();
  result.triangle_index = scene_instance_triangle_index(ray_query.CommittedPrimitiveIndex(), result.instance_index);
  result.hit_t = ray_query.CommittedRayT();
  result.tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], result.triangle_index);
  result.surface_point = wavefront_load_surface_point_compact(result.tri, ray_query.CommittedTriangleBarycentrics(), ray.Direction, result.instance_index);
  result.emitter_index = kInvalidIndex;
  if (try_load_material_full(result.tri.material_index, result.material)) {
    surface_point_apply_material_normal_map(result.surface_point, result.material, ray.Direction);
  }
  result.hit = 1u;
  return true;
}

bool wavefront_trace_closest_surface_or_boundary_with_origin_retry(RayDesc ray, inout uint seed, out TraceSurfaceResult result, out bool boundary_hit,
  out uint boundary_medium_index) {
  const uint initial_seed = seed;
  if (wavefront_trace_closest_surface_or_boundary(ray, seed, result, boundary_hit, boundary_medium_index)) {
    return true;
  }
  if (ray.TMin > 0.0f) {
    return false;
  }

  RayDesc retry_ray = ray;
  retry_ray.Origin -= retry_ray.Direction * kRayEpsilon;
  retry_ray.TMin = kRayEpsilon;
  retry_ray.TMax = ray.TMax < (kMaxFloat - kRayEpsilon) ? (ray.TMax + kRayEpsilon) : kMaxFloat;
  seed = initial_seed;
  if (wavefront_trace_closest_surface_or_boundary(retry_ray, seed, result, boundary_hit, boundary_medium_index) == false) {
    return false;
  }
  result.hit_t -= kRayEpsilon;
  return result.hit_t > 0.0f;
}

bool wavefront_trace_subsurface_material_with_origin_retry(RayDesc ray, uint material_index, inout uint seed, out TraceSurfaceResult result) {
  const uint initial_seed = seed;
  if (wavefront_trace_subsurface_material(ray, material_index, seed, result)) {
    return true;
  }
  if (ray.TMin > 0.0f) {
    return false;
  }

  RayDesc retry_ray = ray;
  retry_ray.Origin -= retry_ray.Direction * kRayEpsilon;
  retry_ray.TMin = kRayEpsilon;
  retry_ray.TMax = ray.TMax < (kMaxFloat - kRayEpsilon) ? (ray.TMax + kRayEpsilon) : kMaxFloat;
  seed = initial_seed;
  if (wavefront_trace_subsurface_material(retry_ray, material_index, seed, result) == false) {
    return false;
  }
  result.hit_t -= kRayEpsilon;
  return result.hit_t > 0.0f;
}

bool wavefront_trace_subsurface_path_state(bool from_camera, uint path_index, RayDesc ray, SpectralQuery spect, SpectralResponse throughput, inout uint seed,
  inout GPUWavefrontSubsurfaceState subsurface_state, out TraceSurfaceResult result) {
  result = (TraceSurfaceResult)0;
  result.medium_index = subsurface_state.medium_index;
  result.triangle_index = kInvalidIndex;
  result.emitter_index = kInvalidIndex;
  result.hit_t = ray.TMax;
  result.transmittance = spectral_response_make(spect, 1.0f);

#if ETX_UPBP
  if (scene_path_mode_is_upbp()) {
    GPUUPBPResources upbp_resources = upbp_load_resources(wavefront_load_resources());
    GPUUPBPPathState upbp_path_state = upbp_load_path_state(upbp_resources.path_state_buffer, upbp_path_state_index(upbp_resources, from_camera, path_index));
    if (((upbp_path_state.flags & GPUUPBPPathStateFlags::Valid) == 0u) ||
        (upbp_begin_transport_segment(upbp_resources, from_camera, path_index, spect, upbp_path_state) == false)) {
      upbp_mark_failed_path(upbp_resources, from_camera, path_index, GPUUPBPPathFailure::BeginSegment);
      return false;
    }

    RayDesc exit_ray = ray;
    exit_ray.TMax = kMaxFloat;
    TraceSurfaceResult surface_hit = (TraceSurfaceResult)0;
    if (wavefront_trace_subsurface_material_with_origin_retry(exit_ray, subsurface_state.material_index, seed, surface_hit) == false) {
      TraceSurfaceResult other_surface_hit = (TraceSurfaceResult)0;
      if (wavefront_trace_subsurface_material_with_origin_retry(exit_ray, kInvalidIndex, seed, other_surface_hit)) {
        if (upbp_mark_failed_path(upbp_resources, from_camera, path_index, GPUUPBPPathFailure::SubsurfaceExitMaterialMismatch)) {
          RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(upbp_resources.counter_buffer);
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail0 * 4u, subsurface_state.material_index);
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail1 * 4u, other_surface_hit.tri.material_index);
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail2 * 4u, upbp_path_state.path_length);
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail3 * 4u, upbp_path_segment_count(upbp_path_state));
        }
        return false;
      }
      upbp_mark_failed_path(upbp_resources, from_camera, path_index, GPUUPBPPathFailure::SubsurfaceExitNotFound);
      return false;
    }
    if ((surface_hit.hit_t <= 0.0f) || (isfinite(surface_hit.hit_t) == false)) {
      if (upbp_mark_failed_path(upbp_resources, from_camera, path_index, GPUUPBPPathFailure::InvalidSubsurfaceExitDistance)) {
        RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(upbp_resources.counter_buffer);
        counters.Store(GPUUPBPCounterIndex::FirstFailureDetail0 * 4u, asuint(surface_hit.hit_t));
        counters.Store(GPUUPBPCounterIndex::FirstFailureDetail1 * 4u, surface_hit.triangle_index);
        counters.Store(GPUUPBPCounterIndex::FirstFailureDetail2 * 4u, surface_hit.instance_index);
      }
      return false;
    }

    const uint medium_domain = from_camera ? kUPBPRandomDomainCameraMediumTracking : kUPBPRandomDomainLightMediumTracking;
    uint medium_seed = upbp_deterministic_seed(upbp_path_state.global_path_index, upbp_path_state.path_length, upbp_path_segment_count(upbp_path_state) - 1u, medium_domain);
    MediumSample medium_sample = (MediumSample)0;
    uint terminal_type = kUPBPMediumFailure;
    bool tracking_valid = false;
    if (subsurface_state.medium_index != kInvalidIndex) {
      tracking_valid = upbp_track_interval(upbp_resources, from_camera, path_index, subsurface_state.medium_index, ray.Origin, ray.Direction, surface_hit.hit_t,
        surface_hit.surface_point.vertex.pos, surface_hit.surface_point.geo_normal, spect, medium_seed, upbp_path_state, medium_sample, terminal_type);
    } else {
      const SpectralResponse absorption = spectral_response_sub(subsurface_state.extinction, subsurface_state.scattering);
      tracking_valid = upbp_track_homogeneous_interval(upbp_resources, from_camera, path_index, subsurface_state.material_index, subsurface_state.scattering, absorption,
        ray.Origin, ray.Direction, surface_hit.hit_t, surface_hit.surface_point.vertex.pos, surface_hit.surface_point.geo_normal, spect, medium_seed, upbp_path_state,
        medium_sample, terminal_type);
    }
    if (tracking_valid == false) {
      upbp_mark_failed_path(upbp_resources, from_camera, path_index, GPUUPBPPathFailure::SubsurfaceTracking);
      return false;
    }

    result.transmittance = medium_sample.weight;
    result.medium_index = subsurface_state.medium_index;
    if ((terminal_type == kUPBPMediumScatter) || (terminal_type == kUPBPMediumAbsorb)) {
      result.hit_t = medium_sample.sampled_medium_t;
      result.surface_point.vertex.pos = medium_sample.pos;
      result.hit = 1u;
      if (terminal_type == kUPBPMediumAbsorb) {
        upbp_mark_terminal_transport_segment(upbp_resources, from_camera, path_index, upbp_path_state);
        return false;
      }
      return true;
    }

    subsurface_state.flags = 0u;
    result = surface_hit;
    result.transmittance = medium_sample.weight;
    result.medium_index = subsurface_state.medium_index;
    result.tri.material_index = subsurface_state.scatter_material_index;
    try_load_material_full(subsurface_state.scatter_material_index, result.material);
    result.emitter_index = kInvalidIndex;
    return true;
  }
#endif

  SpectralResponse pdf = spectral_response_zero(spect);
  float sampled_distance = 0.0f;
  while (sampled_distance < kRayEpsilon) {
    uint channel = medium_sample_shared_sample_spectrum_component(spect, subsurface_state.albedo, throughput, rnd01(seed), pdf);
    float extinction_value = wavefront_subsurface_trace_response_component(subsurface_state.extinction, channel);
    sampled_distance = (extinction_value > 0.0f) ? (-log(1.0f - rnd01(seed)) / extinction_value) : kMaxFloat;
  }

  RayDesc subsurface_ray = ray;
  subsurface_ray.TMin = max(kRayEpsilon, ray.TMin);
  subsurface_ray.TMax = sampled_distance;

  TraceSurfaceResult surface_hit = (TraceSurfaceResult)0;
  bool intersection_found = wavefront_trace_subsurface_material(subsurface_ray, subsurface_state.material_index, seed, surface_hit);
  float segment_distance = intersection_found ? surface_hit.hit_t : sampled_distance;
  SpectralResponse tr = spectral_response_exp(spectral_response_mul(subsurface_state.extinction, -segment_distance));
  SpectralResponse pdf_factor = tr;
  if (intersection_found == false) {
    pdf_factor = wavefront_subsurface_trace_safe_mul(spect, tr, subsurface_state.extinction);
  }
  pdf = spectral_response_mul(pdf, pdf_factor);
  if (spectral_response_is_zero(pdf)) {
    return false;
  }

  SpectralResponse weight = tr;
  if (intersection_found == false) {
    weight = wavefront_subsurface_trace_safe_mul(spect, tr, subsurface_state.scattering);
  }
  SpectralResponse weighted_transmittance = spectral_response_div(weight, max(kEpsilon, wavefront_subsurface_trace_response_sum(pdf)));
  if (intersection_found) {
    subsurface_state.flags = 0u;
    result = surface_hit;
    result.transmittance = weighted_transmittance;
    result.medium_index = subsurface_state.medium_index;
    result.tri.material_index = subsurface_state.scatter_material_index;
    try_load_material_full(subsurface_state.scatter_material_index, result.material);
    result.emitter_index = kInvalidIndex;
    return true;
  }

  result.hit_t = segment_distance;
  result.transmittance = weighted_transmittance;
  result.surface_point.vertex.pos = ray.Origin + ray.Direction * segment_distance;
  result.hit = 1u;
  return true;
}

bool wavefront_trace_path_state(bool from_camera, uint path_index, RayDesc ray, SpectralQuery spect, SpectralResponse throughput, inout uint medium_index, inout uint seed,
  out TraceSurfaceResult result) {
  result = (TraceSurfaceResult)0;
  result.medium_index = medium_index;
  result.triangle_index = kInvalidIndex;
  result.emitter_index = kInvalidIndex;
  result.hit_t = ray.TMax;
  result.transmittance = spectral_response_make(spect, 1.0f);
  float3 current_origin = ray.Origin;
  float traveled_distance = 0.0f;
  uint ray_medium_index = medium_index;
#if ETX_UPBP
  GPUUPBPResources upbp_resources = upbp_load_resources(wavefront_load_resources());
  GPUUPBPPathState upbp_path_state = (GPUUPBPPathState)0;
  uint upbp_medium_seed = 0u;
  const bool record_upbp = scene_path_mode_is_upbp() && (upbp_resources.path_state_buffer != kInvalidIndex);
  if (record_upbp) {
    upbp_path_state = upbp_load_path_state(upbp_resources.path_state_buffer, upbp_path_state_index(upbp_resources, from_camera, path_index));
    if ((from_camera == false) && (upbp_path_state.path_length == 0u) && (upbp_path_state.first_vertex_index != kInvalidIndex)) {
      const GPUUPBPVertex endpoint = upbp_load_vertex(upbp_resources.vertex_buffer, upbp_path_state.first_vertex_index);
      if (((endpoint.flags & GPUUPBPVertexFlags::DistantEndpoint) != 0u) && (upbp_distance_to_scene_sphere_exit(current_origin, ray.Direction) <= 0.0f)) {
        return false;
      }
    }
    if (((upbp_path_state.flags & GPUUPBPPathStateFlags::Valid) == 0u) ||
        (upbp_begin_transport_segment(upbp_resources, from_camera, path_index, spect, upbp_path_state) == false)) {
      upbp_mark_failed_path(upbp_resources, from_camera, path_index, GPUUPBPPathFailure::BeginSegment);
      result.transmittance = spectral_response_make(spect, 0.0f);
      return false;
    }
    const uint medium_domain = from_camera ? kUPBPRandomDomainCameraMediumTracking : kUPBPRandomDomainLightMediumTracking;
    upbp_medium_seed = upbp_deterministic_seed(upbp_path_state.global_path_index, upbp_path_state.path_length, upbp_path_segment_count(upbp_path_state) - 1u, medium_domain);
  }
#endif

  while (true) {
    RayDesc segment_ray = (RayDesc)0;
    segment_ray.Origin = current_origin;
    segment_ray.Direction = ray.Direction;
    segment_ray.TMin = (traveled_distance == 0.0f) ? ray.TMin : kRayEpsilon;
    if (ray.TMax >= kMaxFloat) {
      segment_ray.TMax = ray.TMax;
    } else {
      float remaining_distance = max(0.0f, ray.TMax - traveled_distance);
      segment_ray.TMax = max(segment_ray.TMin, remaining_distance);
    }

    TraceSurfaceResult segment_hit = (TraceSurfaceResult)0;
    bool boundary_hit = false;
    uint boundary_medium = kInvalidIndex;
    bool found_hit = false;
#if ETX_UPBP
    if (record_upbp) {
      found_hit = wavefront_trace_closest_surface_or_boundary_with_origin_retry(segment_ray, seed, segment_hit, boundary_hit, boundary_medium);
    } else
#endif
    {
      found_hit = wavefront_trace_closest_surface_or_boundary(segment_ray, seed, segment_hit, boundary_hit, boundary_medium);
    }

    float segment_distance = found_hit ? segment_hit.hit_t : segment_ray.TMax;
#if ETX_UPBP
    if (record_upbp && (found_hit == false) && (segment_ray.TMax >= (0.5f * kMaxFloat))) {
      segment_distance = upbp_distance_to_scene_sphere_exit(current_origin, ray.Direction);
    }
#endif
    if ((segment_distance <= 0.0f) || (isfinite(segment_distance) == false)) {
#if ETX_UPBP
      if (record_upbp) {
        if (upbp_mark_failed_path(upbp_resources, from_camera, path_index, GPUUPBPPathFailure::InvalidSegmentDistance)) {
          SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
          const float3 sphere_offset = current_origin - globals_data.bounding_sphere_center;
          RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(upbp_resources.counter_buffer);
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail0 * 4u, upbp_path_state.path_length);
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail1 * 4u, asuint(length(sphere_offset)));
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail2 * 4u, asuint(globals_data.bounding_sphere_radius));
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail3 * 4u, asuint(dot(ray.Direction, sphere_offset)));
        }
      }
#endif
      result.transmittance = spectral_response_make(spect, 0.0f);
      return false;
    }
    SpectralResponse segment_throughput = spectral_response_mul(throughput, result.transmittance);
    MediumSample medium_sample = (MediumSample)0;
    bool sampled_medium = false;
#if ETX_UPBP
    uint upbp_terminal_type = kUPBPMediumEscape;
    if (record_upbp) {
      const float3 terminal_normal = found_hit ? segment_hit.surface_point.geo_normal : float3(0.0f, 0.0f, 0.0f);
      if (upbp_track_interval(upbp_resources, from_camera, path_index, ray_medium_index, current_origin, ray.Direction, segment_distance, segment_hit.surface_point.vertex.pos,
            terminal_normal, spect, upbp_medium_seed, upbp_path_state, medium_sample, upbp_terminal_type) == false) {
        upbp_mark_failed_path(upbp_resources, from_camera, path_index, GPUUPBPPathFailure::TrackInterval);
        result.transmittance = spectral_response_make(spect, 0.0f);
        return false;
      }
      sampled_medium = (upbp_terminal_type == kUPBPMediumScatter) || (upbp_terminal_type == kUPBPMediumAbsorb);
    } else
#endif
    {
      sampled_medium = wavefront_try_sample_medium_segment(ray_medium_index, current_origin, ray.Direction, segment_distance, spect, segment_throughput, seed, medium_sample);
    }
    if (sampled_medium) {
      result.transmittance = spectral_response_mul(result.transmittance, medium_sample.weight);
      medium_index = ray_medium_index;
      result.medium_index = ray_medium_index;
      result.hit_t = traveled_distance + medium_sample.sampled_medium_t;
      result.surface_point.vertex.pos = medium_sample.pos;
      result.hit = 1u;
#if ETX_UPBP
      if (record_upbp && (upbp_terminal_type == kUPBPMediumAbsorb)) {
        upbp_mark_terminal_transport_segment(upbp_resources, from_camera, path_index, upbp_path_state);
        return false;
      }
#endif
      return true;
    }

    result.transmittance = spectral_response_mul(result.transmittance, medium_sample.weight);
    SpectralResponse accumulated_transmittance = result.transmittance;
    medium_index = ray_medium_index;
    result.medium_index = ray_medium_index;

    if (found_hit == false) {
#if ETX_UPBP
      if (record_upbp) {
        upbp_mark_terminal_transport_segment(upbp_resources, from_camera, path_index, upbp_path_state);
      }
#endif
      return false;
    }

    if (boundary_hit == false) {
      result = segment_hit;
      result.transmittance = accumulated_transmittance;
      result.hit_t = traveled_distance + segment_hit.hit_t;
      result.medium_index = ray_medium_index;
      medium_index = ray_medium_index;
      return true;
    }

#if ETX_UPBP
    if (record_upbp) {
      upbp_record_transport_boundary(upbp_resources, upbp_path_state);
      GPUUPBPSegment upbp_segment = upbp_load_segment(upbp_resources.segment_buffer, upbp_path_state.current_segment_index);
      if (upbp_path_boundary_count(upbp_path_state) > upbp_resources.iteration.maximum_boundary_count) {
        if (upbp_mark_failed_path(upbp_resources, from_camera, path_index, GPUUPBPPathFailure::BoundaryLimit)) {
          RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(upbp_resources.counter_buffer);
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail0 * 4u, segment_hit.triangle_index);
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail1 * 4u, asuint(segment_hit.hit_t));
        }
        result.transmittance = spectral_response_make(spect, 0.0f);
        return false;
      }
    }
#endif

    // Match CPU bidirectional sampler consumption for boundary intersections.
    // The CPU path allocates per-surface randoms before discovering the boundary and continuing.
    rnd01(seed);
    rnd01(seed);
    rnd01(seed);
    rnd01(seed);
    rnd01(seed);
    rnd01(seed);

    traveled_distance += segment_hit.hit_t;
    if ((ray.TMax < kMaxFloat) && (traveled_distance >= ray.TMax)) {
      return false;
    }

    const float boundary_side = dot(segment_hit.surface_point.geo_normal, ray.Direction) >= 0.0f ? 1.0f : -1.0f;
    current_origin = offset_ray(segment_hit.surface_point.vertex.pos, segment_hit.surface_point.geo_normal * boundary_side);
    ray_medium_index = boundary_medium;
  }
}

void wavefront_trace_path(bool from_camera, uint dispatch_index) {
  uint queue_descriptor = wavefront_queue_current_descriptor(from_camera);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  GPUWavefrontResources resources = wavefront_load_resources();
  uint path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  uint state_descriptor = from_camera ? resources.camera_state_buffer : resources.light_state_buffer;
  uint hit_descriptor = from_camera ? resources.camera_hit_buffer : resources.light_hit_buffer;
  GPUWavefrontPathState state = wavefront_load_path_state(state_descriptor, path_index);
  if (wavefront_path_state_valid(state) == false) {
    return;
  }
  RayDesc ray = (RayDesc)0;
  ray.Origin = state.ray.o;
  ray.Direction = state.ray.d;
  GPUWavefrontSubsurfaceState subsurface_state = (GPUWavefrontSubsurfaceState)0;
  uint subsurface_state_buffer = wavefront_subsurface_state_buffer(resources, from_camera);
  bool subsurface_active = wavefront_subsurface_state_active(resources, from_camera, path_index, subsurface_state);
  ray.TMin = scene_path_mode_is_upbp() ? max(0.0f, state.ray.min_t) : max(kRayEpsilon, state.ray.min_t);
  ray.TMax = max(ray.TMin + kRayEpsilon, state.ray.max_t);

  uint medium_index = state.medium_index;
  uint seed = state.sampler_seed;
  TraceSurfaceResult trace_result = (TraceSurfaceResult)0;
  bool hit_found = false;
  if (subsurface_active) {
    hit_found = wavefront_trace_subsurface_path_state(from_camera, path_index, ray, state.spect, state.throughput, seed, subsurface_state, trace_result);
    wavefront_store_subsurface_state(subsurface_state_buffer, path_index, subsurface_state);
  } else {
    hit_found = wavefront_trace_path_state(from_camera, path_index, ray, state.spect, state.throughput, medium_index, seed, trace_result);
  }
  state.medium_index = medium_index;
  state.sampler_seed = seed;
  wavefront_store_path_state(state_descriptor, path_index, state);

  GPUWavefrontHit hit = (GPUWavefrontHit)0;
  hit.transmittance = trace_result.transmittance;
  hit.medium_index = medium_index;
  hit.flags = GPUWavefrontHitFlags::Valid;
  if (hit_found) {
    hit.vertex = trace_result.surface_point.vertex;
    hit.geo_normal = trace_result.surface_point.geo_normal;
    hit.hit_t = trace_result.hit_t;
    hit.triangle_index = trace_result.triangle_index;
    hit.instance_index = trace_result.instance_index;
    hit.material_index = trace_result.tri.material_index;
    hit.emitter_index = trace_result.emitter_index;
    hit.barycentric = trace_result.surface_point.barycentrics.yz;
    if (trace_result.triangle_index == kInvalidIndex) {
      hit.flags |= GPUWavefrontHitFlags::Medium;
      if (subsurface_active) {
        hit.flags |= GPUWavefrontHitFlags::Subsurface;
      }
    }
  } else {
    hit.instance_index = kInvalidIndex;
    hit.flags |= GPUWavefrontHitFlags::Miss;
  }
  wavefront_store_hit(hit_descriptor, path_index, hit);
}
