#pragma once
#include <interop/subsurface_free_path_shared.hxx>

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
bool wavefront_trace_path_state(bool from_camera, uint path_index, bool restricted_subsurface, RayDesc ray, SpectralQuery spect, SpectralResponse throughput,
  inout uint medium_index, inout uint seed, inout GPUWavefrontSubsurfaceState subsurface_state, out float2 flight_pdf, out TraceSurfaceResult result,
  out SpectralResponse thermal_radiance);
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

bool wavefront_trace_transmittance_to_point_inline_medium(float3 origin, float3 target, SpectralQuery spect, uint medium_index, SpectralResponse inline_extinction,
  SpectralResponse inline_scattering, uint inline_flags, float subsurface_packing, bool source_is_medium, bool target_is_medium, inout uint seed,
  out SpectralResponse transmittance, out float2 flight_pdf);

bool wavefront_trace_transmittance_to_point(float3 origin, float3 target, SpectralQuery spect, uint medium_index, inout uint seed, out SpectralResponse transmittance) {
  float2 flight_pdf = float2(1.0f, 1.0f);
  return wavefront_trace_transmittance_to_point_inline_medium(origin, target, spect, medium_index, spectral_response_zero(spect), spectral_response_zero(spect), 0u, 0.0f, false,
    false, seed, transmittance, flight_pdf);
}

void wavefront_connection_boundary_medium(TraceSurfaceResult boundary, float3 direction, SpectralQuery spect, inout uint medium_index, inout SpectralResponse inline_extinction,
  inout SpectralResponse inline_scattering, inout uint inline_flags, inout float packing) {
  const bool entering = dot(boundary.surface_point.geo_normal, direction) < 0.0f;
  medium_index = entering ? boundary.material.int_medium : boundary.material.ext_medium;
  inline_extinction = spectral_response_zero(spect);
  inline_scattering = spectral_response_zero(spect);
  inline_flags = 0u;
  packing = 0.0f;
  if (entering && material_has_incident_subsurface_boundary(boundary.material)) {
    if (medium_index == kInvalidIndex) {
      SpectralResponse albedo = spectral_response_zero(spect);
      wavefront_subsurface_remap(spect, load_scene_spectrum_or_zero(boundary.material.scattering.spectrum_index, spect),
        load_scene_spectrum_or_zero(boundary.material.subsurface.spectrum_index, spect), albedo, inline_extinction, inline_scattering);
    } else {
      MediumAccess medium = (MediumAccess)0;
      if (wavefront_try_load_medium(medium_index, medium)) {
        inline_scattering = gpu_medium_scattering(medium, spect);
        inline_extinction = spectral_response_add(inline_scattering, gpu_medium_absorption(medium, spect));
      }
    }
    inline_flags = GPUWavefrontSubsurfaceFlags::InlineMedium;
    packing = boundary.material.subsurface_packing;
  }
}

#if ETX_UPBP
bool wavefront_trace_closest_surface_or_boundary_with_origin_retry(RayDesc ray, inout uint seed, out TraceSurfaceResult result, out bool boundary_hit,
  out uint boundary_medium_index);

bool wavefront_upbp_trace_connection_to_point(float3 origin, float3 target, SpectralQuery spect, uint medium_index, SpectralResponse inline_extinction,
  SpectralResponse inline_scattering, uint inline_flags, float subsurface_packing, bool source_is_medium, bool target_is_medium, inout uint intersection_seed,
  inout uint medium_seed, out bool visible, out GPUUPBPConnectionInterval connection, out uint failure) {
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
  SpectralResponse current_inline_scattering = inline_scattering;
  uint current_inline_flags = inline_flags;
  float current_packing = subsurface_packing;
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
    boundary_hit = boundary_hit || (found_hit && material_has_incident_subsurface_boundary(trace_result.material));
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
      if (material_has_incident_subsurface_boundary(trace_result.material)) {
        connection.weight = spectral_response_mul(connection.weight,
          bsdf_resource_subsurface_boundary_color(make_scene_bsdf_resource_gpu_context(), spect, trace_result.material, trace_result.surface_point.vertex.tex));
      }
      current_origin = offset_ray(trace_result.surface_point.vertex.pos, trace_result.surface_point.geo_normal * boundary_side);
      wavefront_connection_boundary_medium(trace_result, direction, spect, current_medium_index, current_inline_extinction, current_inline_scattering, current_inline_flags,
        current_packing);
      retry_from_source = false;
      continue;
    }

    GPUUPBPConnectionInterval interval = (GPUUPBPConnectionInterval)0;
    uint terminal_type = kUPBPMediumFailure;
    GPUUPBPResources upbp_resources = upbp_load_resources(wavefront_load_resources());
    bool tracking_valid = false;
    if (current_packing > 0.0f) {
      const bool source_collision = source_is_medium && (boundary_count == 0u);
      const bool target_collision = target_is_medium && (found_hit == false);
      const float flight_distance = found_hit ? interval_distance : target_distance;
      const float kernel_forward =
        subsurface_transport_sampling_kernel(current_inline_extinction, current_inline_scattering, current_packing, source_collision, target_collision, flight_distance);
      const float kernel_reverse =
        subsurface_transport_sampling_kernel(current_inline_extinction, current_inline_scattering, current_packing, target_collision, source_collision, flight_distance);
      if ((kernel_forward <= 0.0f) || (kernel_reverse <= 0.0f)) {
        return true;
      }
      interval.weight = subsurface_transport_kernel(current_inline_extinction, current_packing, source_collision, target_collision, flight_distance);
      interval.log_transport_pdf_forward = log(kernel_forward);
      interval.log_transport_pdf_reverse = log(kernel_reverse);
      const float2 log_beam_survival = upbp_subsurface_beam_survival(current_inline_extinction, current_inline_scattering, current_packing, source_collision,
        target_collision, flight_distance);
      interval.log_beam_survival_forward = log_beam_survival.x;
      interval.log_beam_survival_reverse = log_beam_survival.y;
      interval.has_exclusion_transport = true;
      terminal_type = kUPBPMediumEscape;
      tracking_valid = true;
    } else if (current_medium_index != kInvalidIndex) {
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
    connection.log_beam_survival_forward += interval.has_exclusion_transport ? interval.log_beam_survival_forward : interval.log_transport_pdf_forward;
    connection.log_beam_survival_reverse += interval.has_exclusion_transport ? interval.log_beam_survival_reverse : interval.log_transport_pdf_reverse;
    connection.has_exclusion_transport = connection.has_exclusion_transport || interval.has_exclusion_transport;
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
    if (material_has_incident_subsurface_boundary(trace_result.material)) {
      connection.weight = spectral_response_mul(connection.weight,
        bsdf_resource_subsurface_boundary_color(make_scene_bsdf_resource_gpu_context(), spect, trace_result.material, trace_result.surface_point.vertex.tex));
    }
    current_origin = offset_ray(trace_result.surface_point.vertex.pos, trace_result.surface_point.geo_normal * boundary_side);
    wavefront_connection_boundary_medium(trace_result, direction, spect, current_medium_index, current_inline_extinction, current_inline_scattering, current_inline_flags,
      current_packing);
    retry_from_source = false;
  }
}
#endif

bool wavefront_trace_transmittance_to_point_inline_medium(float3 origin, float3 target, SpectralQuery spect, uint medium_index, SpectralResponse inline_extinction,
  SpectralResponse inline_scattering, uint inline_flags, float subsurface_packing, bool source_is_medium, bool target_is_medium, inout uint seed,
  out SpectralResponse transmittance, out float2 flight_pdf) {
  transmittance = spectral_response_make(spect, 1.0f);
  flight_pdf = float2(1.0f, 1.0f);
  float3 current_origin = origin;
  uint current_medium_index = medium_index;
  SpectralResponse current_inline_extinction = inline_extinction;
  SpectralResponse current_inline_scattering = inline_scattering;
  uint current_inline_flags = inline_flags;
  float current_packing = subsurface_packing;
  bool current_source_collision = source_is_medium;

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
    boundary_hit = boundary_hit || (found_hit && material_has_incident_subsurface_boundary(trace_result.material));
    float segment_distance = found_hit ? trace_result.hit_t : ray.TMax;
    SpectralResponse segment_transmittance = spectral_response_make(spect, 1.0f);
    if (current_packing > 0.0f) {
      const bool target_collision = target_is_medium && (found_hit == false);
      const float flight_distance = found_hit ? segment_distance : distance;
      segment_transmittance = subsurface_transport_kernel(current_inline_extinction, current_packing, current_source_collision, target_collision, flight_distance);
      const float kernel_forward =
        subsurface_transport_sampling_kernel(current_inline_extinction, current_inline_scattering, current_packing, current_source_collision, target_collision, flight_distance);
      const float kernel_reverse =
        subsurface_transport_sampling_kernel(current_inline_extinction, current_inline_scattering, current_packing, target_collision, current_source_collision, flight_distance);
      const float extinction = subsurface_transport_sampling_extinction(current_inline_extinction);
      flight_pdf *= float2(kernel_forward * (target_collision ? extinction : 1.0f), kernel_reverse * (current_source_collision ? extinction : 1.0f));
    } else if (current_medium_index != kInvalidIndex) {
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

    if (material_has_incident_subsurface_boundary(trace_result.material)) {
      transmittance = spectral_response_mul(transmittance,
        bsdf_resource_subsurface_boundary_color(make_scene_bsdf_resource_gpu_context(), spect, trace_result.material, trace_result.surface_point.vertex.tex));
    }
    wavefront_connection_boundary_medium(trace_result, ray.Direction, spect, current_medium_index, current_inline_extinction, current_inline_scattering, current_inline_flags,
      current_packing);
    current_source_collision = false;
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
  uint closest_boundary_triangle = kInvalidIndex;
  uint closest_boundary_instance = kInvalidIndex;
  float closest_boundary_t = ray.TMax;
  float2 closest_boundary_bary = float2(0.0f, 0.0f);

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

    bool null_boundary = material_access.material_class == MaterialClass::Boundary;
    if (material_access.material_class == MaterialClass::Diffuse) {
      Material candidate_material = (Material)0;
      null_boundary = try_load_material_full(tri.material_index, candidate_material) && material_has_incident_subsurface_boundary(candidate_material);
    }
    if (null_boundary) {
      const float candidate_t = ray_query.CandidateTriangleRayT();
      if ((closest_boundary_triangle == kInvalidIndex) || (candidate_t < closest_boundary_t)) {
        closest_boundary_triangle = candidate_triangle_index;
        closest_boundary_instance = candidate_instance_index;
        closest_boundary_t = candidate_t;
        closest_boundary_bary = candidate_bary;
      }
      continue;
    }
    ray_query.CommitNonOpaqueTriangleHit();
  }

  const bool opaque_hit = ray_query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
  if ((opaque_hit == false) && (closest_boundary_triangle == kInvalidIndex)) {
    return false;
  }

  // An opaque surface at a boundary's distance must remain visible to transport.
  bool use_boundary = closest_boundary_triangle != kInvalidIndex;
  if (opaque_hit) {
    use_boundary = use_boundary && (closest_boundary_t < ray_query.CommittedRayT());
  }
  float2 hit_bary = float2(0.0f, 0.0f);
  if (use_boundary) {
    result.instance_index = closest_boundary_instance;
    result.triangle_index = closest_boundary_triangle;
    result.hit_t = closest_boundary_t;
    hit_bary = closest_boundary_bary;
  } else {
    result.instance_index = ray_query.CommittedInstanceID();
    result.triangle_index = scene_instance_triangle_index(ray_query.CommittedPrimitiveIndex(), result.instance_index);
    result.hit_t = ray_query.CommittedRayT();
    hit_bary = ray_query.CommittedTriangleBarycentrics();
  }
  result.tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], result.triangle_index);
  result.surface_point = wavefront_load_surface_point_compact(result.tri, hit_bary, ray.Direction, result.instance_index);
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

#if ETX_UPBP
ETX_SHARED_NOINLINE bool wavefront_upbp_track_transport_interval(GPUUPBPResources resources, bool from_camera, uint path_index, uint medium_index, bool inside_subsurface,
  bool restricted_subsurface, GPUWavefrontSubsurfaceState body, float3 origin, float3 direction, float maximum_distance, float3 surface_position, float3 surface_normal,
  SpectralQuery spect, inout uint seed, inout GPUUPBPPathState path_state, out MediumSample sample, out uint terminal_type) {
  if (inside_subsurface && (body.packing > 0.0f)) {
    return upbp_track_subsurface_interval(resources, from_camera, path_index, body, origin, direction, maximum_distance, surface_position, surface_normal, seed, path_state, sample,
      terminal_type);
  }
  if (inside_subsurface && (medium_index == kInvalidIndex)) {
    return upbp_track_homogeneous_interval(resources, from_camera, path_index, body.material_index, body.owner_instance_index, body.scattering, spectral_response_sub(body.extinction, body.scattering),
      body.phase_function_g, origin, direction, maximum_distance, surface_position, surface_normal, spect, seed, path_state, sample, terminal_type);
  }
  return upbp_track_interval(resources, from_camera, path_index, medium_index, inside_subsurface && restricted_subsurface, origin, direction, maximum_distance, surface_position,
    surface_normal, spect, seed, path_state, sample, terminal_type);
}
#endif

bool wavefront_sample_subsurface_interval(RayDesc ray, float maximum_distance, SpectralQuery spect, SpectralResponse throughput, inout uint seed,
  inout GPUWavefrontSubsurfaceState subsurface_state, out MediumSample sample, out bool scattered) {
  sample = (MediumSample)0;
  scattered = false;
  subsurface_state.flight_pdf_forward = 1.0f;
  subsurface_state.flight_pdf_reverse = 1.0f;
  const bool correlated_origin = (subsurface_state.flags & GPUWavefrontSubsurfaceFlags::CorrelatedOrigin) != 0u;
  if ((subsurface_state.packing > 0.0f) && (scene_path_mode_is_path_tracing() == false)) {
    const SubsurfaceTransportSample flight = subsurface_transport_sample(subsurface_state.extinction, subsurface_state.scattering, subsurface_state.packing, correlated_origin,
      ray.Origin, ray.Direction, maximum_distance, rnd01(seed));
    if ((flight.pdf_forward <= 0.0f) || (flight.pdf_reverse <= 0.0f)) {
      return false;
    }
    subsurface_state.flight_pdf_forward = flight.pdf_forward;
    subsurface_state.flight_pdf_reverse = flight.pdf_reverse;
    sample = flight.sample;
    scattered = flight.scattered;
    return true;
  }
  SpectralResponse channel_weight = subsurface_state.albedo;
  if ((subsurface_state.packing > 0.0f) && (spectral_query_is_spectral(spect) == false)) {
    channel_weight = spectral_response_make(spect, float3(subsurface_free_path_channel_weight(subsurface_state.extinction.integrated.x, subsurface_state.packing, correlated_origin,
                                                            maximum_distance, subsurface_state.albedo.integrated.x),
                                                     subsurface_free_path_channel_weight(subsurface_state.extinction.integrated.y, subsurface_state.packing, correlated_origin,
                                                       maximum_distance, subsurface_state.albedo.integrated.y),
                                                     subsurface_free_path_channel_weight(subsurface_state.extinction.integrated.z, subsurface_state.packing, correlated_origin,
                                                       maximum_distance, subsurface_state.albedo.integrated.z)));
  }
  SpectralResponse pdf = spectral_response_zero(spect);
  const uint channel = medium_sample_shared_sample_spectrum_component(spect, channel_weight, throughput, rnd01(seed), pdf);
  const float extinction_value = wavefront_subsurface_trace_response_component(subsurface_state.extinction, channel);
  const float sampled_distance = subsurface_free_path_sample(extinction_value, subsurface_state.packing, correlated_origin, rnd01(seed));
  const bool intersection_found = maximum_distance <= sampled_distance;
  float segment_distance = intersection_found ? maximum_distance : sampled_distance;
  SpectralResponse tr = spectral_response_zero(spect);
  SpectralResponse density = spectral_response_zero(spect);
  if (spectral_query_is_spectral(spect)) {
    const SubsurfaceFreePath flight = subsurface_free_path_evaluate(subsurface_state.extinction.value, subsurface_state.packing, correlated_origin, segment_distance);
    tr = spectral_response_make(spect, flight.survival);
    density = spectral_response_make(spect, flight.density);
  } else {
    const SubsurfaceFreePath x = subsurface_free_path_evaluate(subsurface_state.extinction.integrated.x, subsurface_state.packing, correlated_origin, segment_distance);
    const SubsurfaceFreePath y = subsurface_free_path_evaluate(subsurface_state.extinction.integrated.y, subsurface_state.packing, correlated_origin, segment_distance);
    const SubsurfaceFreePath z = subsurface_free_path_evaluate(subsurface_state.extinction.integrated.z, subsurface_state.packing, correlated_origin, segment_distance);
    tr = spectral_response_make(spect, float3(x.survival, y.survival, z.survival));
    density = spectral_response_make(spect, float3(x.density, y.density, z.density));
  }
  SpectralResponse pdf_factor = tr;
  if (intersection_found == false) {
    pdf_factor = density;
  }
  pdf = spectral_response_mul(pdf, pdf_factor);
  if (spectral_response_is_zero(pdf)) {
    return false;
  }

  SpectralResponse weight = tr;
  if (intersection_found == false) {
    weight = wavefront_subsurface_trace_safe_mul(spect, density, subsurface_state.albedo);
  }
  sample.weight = spectral_response_div(weight, wavefront_subsurface_trace_response_sum(pdf));
  sample.pos = ray.Origin + ray.Direction * segment_distance;
  sample.sampled_medium_t = intersection_found ? 0.0f : segment_distance;
  scattered = intersection_found == false;
  return true;
}

bool wavefront_trace_path_state(bool from_camera, uint path_index, bool restricted_subsurface, RayDesc ray, SpectralQuery spect, SpectralResponse throughput,
  inout uint medium_index, inout uint seed, inout GPUWavefrontSubsurfaceState subsurface_state, out float2 flight_pdf, out TraceSurfaceResult result,
  out SpectralResponse thermal_radiance) {
  flight_pdf = float2(1.0f, 1.0f);
  thermal_radiance = spectral_response_zero(spect);
  result = (TraceSurfaceResult)0;
  result.medium_index = medium_index;
  result.triangle_index = kInvalidIndex;
  result.emitter_index = kInvalidIndex;
  result.hit_t = ray.TMax;
  result.transmittance = spectral_response_make(spect, 1.0f);
  float3 current_origin = ray.Origin;
  float traveled_distance = 0.0f;
  uint ray_medium_index = restricted_subsurface ? subsurface_state.medium_index : medium_index;
  if (restricted_subsurface) {
    ray.TMin = 0.0f;
    ray.TMax = kMaxFloat;
  }
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
        ((restricted_subsurface == false) && (upbp_begin_transport_segment(upbp_resources, from_camera, path_index, spect, upbp_path_state) == false))) {
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
    segment_ray.Origin = ray.Origin;
    segment_ray.Direction = ray.Direction;
    segment_ray.TMin = (traveled_distance == 0.0f) ? ray.TMin : asfloat(asuint(traveled_distance) + 1u);
    segment_ray.TMax = ray.TMax;

    TraceSurfaceResult segment_hit = (TraceSurfaceResult)0;
    bool boundary_hit = false;
    uint boundary_medium = kInvalidIndex;
    bool found_hit = false;
    if (restricted_subsurface) {
#if ETX_UPBP
      if (record_upbp) {
        found_hit = wavefront_trace_closest_surface_or_boundary_with_origin_retry(segment_ray, seed, segment_hit, boundary_hit, boundary_medium);
      } else
#endif
      {
        found_hit = wavefront_trace_closest_surface_or_boundary(segment_ray, seed, segment_hit, boundary_hit, boundary_medium);
      }
      if (found_hit == false) {
#if ETX_UPBP
        if (record_upbp) {
          GPUUPBPVertex vertex = upbp_load_vertex(upbp_resources.vertex_buffer, upbp_path_state.last_vertex_index);
          vertex.flags &= ~GPUUPBPVertexFlags::HasDeparture;
          upbp_store_vertex(upbp_resources.vertex_buffer, upbp_path_state.last_vertex_index, vertex);
        }
#endif
        subsurface_state.flags = 0u;
        result.transmittance = spectral_response_zero(spect);
        return false;
      }
    } else {
#if ETX_UPBP
      if (record_upbp) {
        found_hit = wavefront_trace_closest_surface_or_boundary_with_origin_retry(segment_ray, seed, segment_hit, boundary_hit, boundary_medium);
      } else
#endif
      {
        found_hit = wavefront_trace_closest_surface_or_boundary(segment_ray, seed, segment_hit, boundary_hit, boundary_medium);
      }
    }

    const bool incident_boundary = found_hit && material_has_incident_subsurface_boundary(segment_hit.material);
    boundary_hit = boundary_hit || incident_boundary;
    const bool inside_subsurface = (subsurface_state.flags & GPUWavefrontSubsurfaceFlags::Active) != 0u;
    float segment_distance = (found_hit ? segment_hit.hit_t : segment_ray.TMax) - traveled_distance;
    bool unbounded_transport = from_camera;
#if ETX_UPBP
    unbounded_transport = unbounded_transport || (record_upbp && (segment_ray.TMax >= (0.5f * kMaxFloat)));
#endif
    if ((found_hit == false) && unbounded_transport) {
      MediumAccessGPUContext access_context =
        make_medium_access_gpu_context(constants.scene.mediums, constants.scene.images, constants.scene.spectrums, constants.scene.spectral_values);
      MediumAccess medium_access = (MediumAccess)0;
      const bool infinite_homogeneous_medium = medium_access_try_load(access_context, ray_medium_index, medium_access) && (medium_access.medium_class == Medium::Homogeneous) &&
                                               ((spectral_response_maximum(medium_access_load_extinction_spectral(access_context, medium_access, spect)) > 0.0f) ||
                                                 ((medium_access.emission_flags & Medium::EmissionRequiresBoundedRegion) != 0u));
      if (infinite_homogeneous_medium) {
        // Camera far clipping limits geometry, not an unbounded medium's transport.
        segment_distance = kMaxFloat;
      }
#if ETX_UPBP
      else if (record_upbp && (segment_ray.TMax >= (0.5f * kMaxFloat))) {
        segment_distance = upbp_distance_to_scene_sphere_exit(current_origin, ray.Direction);
      }
#endif
    }
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
#if ETX_UPBP
    if (record_upbp && restricted_subsurface) {
      if (upbp_begin_transport_segment(upbp_resources, from_camera, path_index, spect, upbp_path_state) == false) {
        upbp_mark_failed_path(upbp_resources, from_camera, path_index, GPUUPBPPathFailure::BeginSegment);
        result.transmittance = spectral_response_zero(spect);
        return false;
      }
      const uint medium_domain = from_camera ? kUPBPRandomDomainCameraMediumTracking : kUPBPRandomDomainLightMediumTracking;
      upbp_medium_seed = upbp_deterministic_seed(upbp_path_state.global_path_index, upbp_path_state.path_length, upbp_path_segment_count(upbp_path_state) - 1u, medium_domain);
    }
#endif
    SpectralResponse segment_throughput = spectral_response_mul(throughput, result.transmittance);
    if (from_camera && (inside_subsurface == false)) {
      bool source_valid = true;
      const SpectralResponse source = medium_segment_emission_radiance(ray_medium_index, current_origin, ray.Direction, segment_distance, spect, seed, source_valid);
      if (source_valid == false) {
        uint ignored;
        // Camera-queue padding reports source failures through the existing queue readback.
        WAVEFRONT_RW_BUFFER(wavefront_queue_next_descriptor(true)).InterlockedOr(kGPUWavefrontQueuePad1Offset, 1u, ignored);
        result.transmittance = spectral_response_zero(spect);
        return false;
      }
      thermal_radiance = spectral_response_add(thermal_radiance, spectral_response_mul(segment_throughput, source));
    }
    MediumSample medium_sample = (MediumSample)0;
    bool sampled_medium = false;
#if ETX_UPBP
    uint upbp_terminal_type = kUPBPMediumEscape;
    if (record_upbp) {
      const float3 terminal_normal = found_hit ? segment_hit.surface_point.geo_normal : float3(0.0f, 0.0f, 0.0f);
      const bool tracking_valid = wavefront_upbp_track_transport_interval(upbp_resources, from_camera, path_index, ray_medium_index, inside_subsurface, restricted_subsurface,
        subsurface_state, current_origin, ray.Direction, segment_distance, segment_hit.surface_point.vertex.pos, terminal_normal, spect, upbp_medium_seed, upbp_path_state,
        medium_sample, upbp_terminal_type);
      if (tracking_valid == false) {
        if (upbp_mark_failed_path(upbp_resources, from_camera, path_index, GPUUPBPPathFailure::TrackInterval)) {
          RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(upbp_resources.counter_buffer);
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail0 * 4u, upbp_medium_seed);
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail1 * 4u, asuint(segment_distance));
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail2 * 4u, asuint(subsurface_state.extinction.value));
          counters.Store(GPUUPBPCounterIndex::FirstFailureDetail3 * 4u, subsurface_state.flags);
        }
        result.transmittance = spectral_response_make(spect, 0.0f);
        return false;
      }
      sampled_medium = (upbp_terminal_type == kUPBPMediumScatter) || (upbp_terminal_type == kUPBPMediumAbsorb);
    } else
#endif
    {
      if (inside_subsurface) {
        RayDesc bulk_ray = segment_ray;
        bulk_ray.Origin = current_origin;
        if (wavefront_sample_subsurface_interval(bulk_ray, segment_distance, spect, segment_throughput, seed, subsurface_state, medium_sample, sampled_medium) == false) {
          result.transmittance = spectral_response_zero(spect);
          return false;
        }
        flight_pdf *= float2(subsurface_state.flight_pdf_forward, subsurface_state.flight_pdf_reverse);
        subsurface_state.flight_pdf_forward = 1.0f;
        subsurface_state.flight_pdf_reverse = 1.0f;
      } else {
        sampled_medium = wavefront_try_sample_medium_segment(ray_medium_index, current_origin, ray.Direction, segment_distance, spect, segment_throughput, seed, medium_sample);
      }
    }
    if (sampled_medium) {
      if (inside_subsurface) {
        subsurface_state.flags |= GPUWavefrontSubsurfaceFlags::CorrelatedOrigin;
      }
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
      result.hit_t = segment_hit.hit_t;
      result.medium_index = ray_medium_index;
      medium_index = ray_medium_index;
      if (restricted_subsurface && (segment_hit.tri.material_index == subsurface_state.material_index)) {
        subsurface_state.flags &= GPUWavefrontSubsurfaceFlags::InlineMedium;
        result.tri.material_index = subsurface_state.scatter_material_index;
        try_load_material_full(subsurface_state.scatter_material_index, result.material);
        result.emitter_index = kInvalidIndex;
      } else if (inside_subsurface) {
        subsurface_state.flags &= ~GPUWavefrontSubsurfaceFlags::CorrelatedOrigin;
      }
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
    if (incident_boundary == false) {
      rnd01(seed);
      rnd01(seed);
      rnd01(seed);
      rnd01(seed);
      rnd01(seed);
      rnd01(seed);
    }

    traveled_distance = segment_hit.hit_t;
    if ((ray.TMax < kMaxFloat) && (traveled_distance >= ray.TMax)) {
      return false;
    }

    current_origin = ray.Origin + ray.Direction * traveled_distance;
    if (incident_boundary) {
      const SpectralResponse boundary_color =
        bsdf_resource_subsurface_boundary_color(make_scene_bsdf_resource_gpu_context(), spect, segment_hit.material, segment_hit.surface_point.vertex.tex);
      result.transmittance = spectral_response_mul(result.transmittance, boundary_color);
#if ETX_UPBP
      if (record_upbp) {
        GPUUPBPSegment boundary_segment = upbp_load_segment(upbp_resources.segment_buffer, upbp_path_state.current_segment_index);
        boundary_segment.weight = upbp_pack_spectral_response(spectral_response_mul(upbp_unpack_spectral_response(boundary_segment.weight), boundary_color));
        upbp_store_segment(upbp_resources.segment_buffer, upbp_path_state.current_segment_index, boundary_segment);
        GPUUPBPInterval boundary_interval = upbp_load_interval(upbp_resources.interval_buffer, upbp_path_state.current_interval_index);
        boundary_interval.weight = upbp_pack_spectral_response(spectral_response_mul(upbp_unpack_spectral_response(boundary_interval.weight), boundary_color));
        upbp_store_interval(upbp_resources.interval_buffer, upbp_path_state.current_interval_index, boundary_interval);
      }
#endif
      const bool entering = dot(segment_hit.surface_point.geo_normal, ray.Direction) < 0.0f;
      ray_medium_index = entering ? segment_hit.material.int_medium : segment_hit.material.ext_medium;
      subsurface_state.flags = 0u;
      if (entering) {
        subsurface_state.material_index = segment_hit.tri.material_index;
        subsurface_state.owner_instance_index = segment_hit.instance_index;
        subsurface_state.scatter_material_index = segment_hit.tri.material_index;
        subsurface_state.medium_index = ray_medium_index;
        subsurface_state.phase_function_g = segment_hit.material.subsurface_anisotropy;
        subsurface_state.packing = segment_hit.material.subsurface_packing;
        subsurface_state.flags = GPUWavefrontSubsurfaceFlags::Active;
        if (ray_medium_index == kInvalidIndex) {
          wavefront_subsurface_remap(spect, load_scene_spectrum_or_zero(segment_hit.material.scattering.spectrum_index, spect),
            load_scene_spectrum_or_zero(segment_hit.material.subsurface.spectrum_index, spect), subsurface_state.albedo, subsurface_state.extinction, subsurface_state.scattering);
          subsurface_state.flags |= GPUWavefrontSubsurfaceFlags::InlineMedium;
        } else {
          MediumAccess medium = (MediumAccess)0;
          if (wavefront_try_load_medium(ray_medium_index, medium) == false) {
            result.transmittance = spectral_response_zero(spect);
            return false;
          }
          subsurface_state.phase_function_g = medium.phase_function_g;
          subsurface_state.scattering = gpu_medium_scattering(medium, spect);
          subsurface_state.extinction = spectral_response_add(subsurface_state.scattering, gpu_medium_absorption(medium, spect));
          subsurface_state.albedo = medium_sample_shared_calculate_albedo(spect, subsurface_state.scattering, subsurface_state.extinction);
        }
      }
    } else {
      ray_medium_index = boundary_medium;
    }
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
  ray.TMin = max(0.0f, state.ray.min_t);
  ray.TMax = max(ray.TMin + kRayEpsilon, state.ray.max_t);

  uint medium_index = state.medium_index;
  uint seed = state.sampler_seed;
  TraceSurfaceResult trace_result = (TraceSurfaceResult)0;
  SpectralResponse thermal_radiance = spectral_response_zero(state.spect);
  bool hit_found = false;
  float2 flight_pdf = float2(1.0f, 1.0f);
  Material body_material = (Material)0;
  const bool incident_active =
    subsurface_active && try_load_material_full(subsurface_state.material_index, body_material) && material_has_incident_subsurface_boundary(body_material);
  const bool restricted_subsurface = subsurface_active && (incident_active == false);
  hit_found = wavefront_trace_path_state(from_camera, path_index, restricted_subsurface, ray, state.spect, state.throughput, medium_index, seed, subsurface_state, flight_pdf,
    trace_result, thermal_radiance);
  if (subsurface_state_buffer != kInvalidIndex) {
    wavefront_store_subsurface_state(subsurface_state_buffer, path_index, subsurface_state);
  }
  subsurface_active = restricted_subsurface || ((subsurface_state.flags & GPUWavefrontSubsurfaceFlags::Active) != 0u);
  state.flight_pdf = flight_pdf;
  if ((scene_path_mode_is_path_tracing() == false) && (scene_path_mode_is_upbp() == false)) {
    state.forward_pdf = wavefront_safe_div(state.forward_pdf, flight_pdf.x);
    const float ratio = wavefront_safe_div(flight_pdf.y, flight_pdf.x);
    state.reverse_pdf *= ratio;
    state.d_vm *= ratio;
    state.d_surface *= ratio;
  }
  if (from_camera && (state.path_length >= load_scene_options_min_path_length()) && (state.path_length <= load_scene_options_max_path_length()) &&
      (spectral_response_is_zero(thermal_radiance) == false)) {
    wavefront_film_add(state.pixel_index, wavefront_spectral_estimate(thermal_radiance, state.spect));
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
    if (subsurface_active) {
      hit.flags |= GPUWavefrontHitFlags::Subsurface;
    }
    if (trace_result.triangle_index == kInvalidIndex) {
      hit.flags |= GPUWavefrontHitFlags::Medium;
    }
  } else {
    hit.instance_index = kInvalidIndex;
    hit.flags |= GPUWavefrontHitFlags::Miss;
  }
  wavefront_store_hit(hit_descriptor, path_index, hit);
}
