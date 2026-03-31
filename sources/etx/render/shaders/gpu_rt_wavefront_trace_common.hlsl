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
      pdf_dir += emitter_discrete_pdf(emitter_index);
    }
  }

  if (environment_count == 0u) {
    return 0.0f;
  }

  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  float normal_factor = target_is_surface ? abs(dot(target_geo_normal, direction)) : 1.0f;
  return (normal_factor / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius)) * (pdf_dir / float(environment_count));
}

bool wavefront_trace_surface_path_compact(RayDesc ray, SpectralQuery spect, inout uint medium_index, inout uint seed, out TraceSurfaceResult result);
bool wavefront_trace_path_state(RayDesc ray, SpectralQuery spect, SpectralResponse throughput, inout uint medium_index, inout uint seed, out TraceSurfaceResult result);
bool wavefront_trace_closest_surface_or_boundary(RayDesc ray, inout uint seed, out TraceSurfaceResult result, out bool boundary_hit, out uint boundary_medium_index);
SurfacePoint wavefront_load_surface_point_compact(TriangleData tri, float2 bary, float3 ray_dir);
float3 wavefront_trace_surface_shading_position(TraceSurfaceResult surface_hit, float3 outgoing_direction);

bool wavefront_trace_transmittance_to_point(float3 origin, float3 target, SpectralQuery spect, uint medium_index, inout uint seed, out SpectralResponse transmittance) {
  transmittance = spectral_response_make(spect, 1.0f);
  float3 current_origin = origin;
  uint current_medium_index = medium_index;

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
    SpectralResponse segment_transmittance = medium_segment_transmittance_spectral(current_medium_index, current_origin, ray.Direction, segment_distance, spect, seed);
    transmittance = spectral_response_mul(transmittance, segment_transmittance);

    if (found_hit == false) {
      return true;
    }

    if (boundary_hit == false) {
      return false;
    }

    current_medium_index = boundary_medium;
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

    uint candidate_triangle_index = ray_query.CandidatePrimitiveIndex();
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
    bool entering_surface = dot(tri.geo_n, ray.Direction) < 0.0f;
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

  result.triangle_index = ray_query.CommittedPrimitiveIndex();
  result.hit_t = ray_query.CommittedRayT();
  result.tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], result.triangle_index);
  result.surface_point = wavefront_load_surface_point_compact(result.tri, ray_query.CommittedTriangleBarycentrics(), ray.Direction);
  result.emitter_index = result.tri.emitter_index;
  try_load_material_full(result.tri.material_index, result.material);
  result.hit = 1u;

  boundary_hit = result.material.cls == MaterialClass::Boundary;
  if (boundary_hit) {
    bool entering_surface = dot(result.tri.geo_n, ray.Direction) < 0.0f;
    boundary_medium_index = entering_surface ? result.material.int_medium : result.material.ext_medium;
  }

  return true;
}

SurfacePoint wavefront_load_surface_point_compact(TriangleData tri, float2 bary, float3 ray_dir) {
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
  result.geo_normal = surface_point_shared_orient_geo_normal(tri.geo_n, ray_dir);
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

  return scene_math_shared_shading_pos(p0, p1, p2, n0, n1, n2, surface_hit.tri.geo_n, surface_hit.surface_point.barycentrics, outgoing_direction);
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

    uint candidate_triangle_index = ray_query.CandidatePrimitiveIndex();
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

  result.triangle_index = ray_query.CommittedPrimitiveIndex();
  result.hit_t = ray_query.CommittedRayT();
  result.tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], result.triangle_index);
  result.surface_point = wavefront_load_surface_point_compact(result.tri, ray_query.CommittedTriangleBarycentrics(), ray.Direction);
  result.emitter_index = result.tri.emitter_index;
  try_load_material_full(result.tri.material_index, result.material);
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

bool wavefront_trace_path_state(RayDesc ray, SpectralQuery spect, SpectralResponse throughput, inout uint medium_index, inout uint seed, out TraceSurfaceResult result) {
  result = (TraceSurfaceResult)0;
  result.medium_index = medium_index;
  result.triangle_index = kInvalidIndex;
  result.emitter_index = kInvalidIndex;
  result.hit_t = ray.TMax;
  result.transmittance = spectral_response_make(spect, 1.0f);
  float3 current_origin = ray.Origin;
  float traveled_distance = 0.0f;
  uint ray_medium_index = medium_index;

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
    bool found_hit = wavefront_trace_closest_surface_or_boundary(segment_ray, seed, segment_hit, boundary_hit, boundary_medium);

    float segment_distance = found_hit ? segment_hit.hit_t : segment_ray.TMax;
    SpectralResponse segment_throughput = spectral_response_mul(throughput, result.transmittance);
    MediumSample medium_sample = (MediumSample)0;
    if (wavefront_try_sample_medium_segment(ray_medium_index, current_origin, ray.Direction, segment_distance, spect, segment_throughput, seed, medium_sample)) {
      result.transmittance = spectral_response_mul(result.transmittance, medium_sample.weight);
      medium_index = ray_medium_index;
      result.medium_index = ray_medium_index;
      result.hit_t = traveled_distance + medium_sample.sampled_medium_t;
      result.surface_point.vertex.pos = medium_sample.pos;
      result.hit = 1u;
      return true;
    }

    result.transmittance = spectral_response_mul(result.transmittance, medium_sample.weight);
    SpectralResponse accumulated_transmittance = result.transmittance;
    medium_index = ray_medium_index;
    result.medium_index = ray_medium_index;

    if (found_hit == false) {
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

    current_origin = wavefront_trace_surface_shading_position(segment_hit, ray.Direction);
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
  ray.TMin = max(kRayEpsilon, state.ray.min_t);
  ray.TMax = max(ray.TMin + kRayEpsilon, state.ray.max_t);

  uint medium_index = state.medium_index;
  uint seed = state.sampler_seed;
  TraceSurfaceResult trace_result = (TraceSurfaceResult)0;
  bool hit_found = wavefront_trace_path_state(ray, state.spect, state.throughput, medium_index, seed, trace_result);
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
    hit.material_index = trace_result.tri.material_index;
    hit.emitter_index = trace_result.emitter_index;
    hit.barycentric = trace_result.surface_point.barycentrics.yz;
    if (trace_result.triangle_index == kInvalidIndex) {
      hit.flags |= GPUWavefrontHitFlags::Medium;
    }
  } else {
    hit.flags |= GPUWavefrontHitFlags::Miss;
  }
  wavefront_store_hit(hit_descriptor, path_index, hit);
}
