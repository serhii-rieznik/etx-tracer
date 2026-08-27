#pragma once

#include <etx/rt/integrators/upbp_subpath.hxx>

#include <utility>

namespace etx {

struct UPBPScatteringEval {
  SpectralResponse value = {};
  float pdf_forward = 0.0f;
  float pdf_reverse = 0.0f;

  bool valid() const {
    return (pdf_forward > 0.0f) && (pdf_reverse > 0.0f) && (value.is_zero() == false);
  }
};

inline UPBPScatteringEval upbp_evaluate_vertex_scattering(const Scene& scene, const SpectralQuery spect, const UPBPPathVertexRecord& vertex, const float3& outgoing_direction,
  Sampler& sampler) {
  UPBPScatteringEval result = {};
  result.value = SpectralResponse{spect, 0.0f};
  if (vertex.cls == UPBPVertexClass::Medium) {
    result.pdf_forward = phase_function(vertex.intersection.w_i, outgoing_direction, vertex.medium.anisotropy);
    result.pdf_reverse = phase_function(outgoing_direction, vertex.intersection.w_i, vertex.medium.anisotropy);
    result.value = SpectralResponse{spect, result.pdf_forward};
    return result;
  }

  if (vertex.cls != UPBPVertexClass::Surface) {
    return result;
  }

  const Material& material = scene.materials[vertex.intersection.material_index];
  const BSDFData data = {spect, vertex.incident_medium_index, vertex.source, vertex.intersection, vertex.intersection.w_i};
  const BSDFEval eval = bsdf::evaluate(data, outgoing_direction, material, sampler);
  result.value = eval.bsdf;
  result.pdf_forward = eval.pdf;
  result.pdf_reverse = bsdf::reverse_pdf(data, outgoing_direction, material, sampler);
  if ((vertex.source == PathSource::Light) && (result.value.is_zero() == false)) {
    const Triangle& triangle = scene.triangles[vertex.intersection.triangle_index];
    const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, vertex.intersection.instance_index);
    result.value *= fix_shading_normal(geometric_normal, vertex.intersection.nrm, vertex.intersection.w_i, outgoing_direction);
  }
  return result;
}

inline uint32_t upbp_connection_medium(const Scene& scene, const UPBPPathVertexRecord& vertex, const float3& outgoing_direction) {
  if (vertex.cls == UPBPVertexClass::Medium) {
    return vertex.medium.index;
  }
  if (vertex.cls != UPBPVertexClass::Surface) {
    return vertex.outgoing_medium_index;
  }

  const Material& material = scene.materials[vertex.intersection.material_index];
  const Triangle& triangle = scene.triangles[vertex.intersection.triangle_index];
  const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, vertex.intersection.instance_index);
  return (dot(geometric_normal, outgoing_direction) < 0.0f) ? material.int_medium : material.ext_medium;
}

struct UPBPConnectionTransmittanceResult {
  UPBPTransportSegmentRecord segment = {};
  SpectralResponse weight = {};
  Intersection failure_intersection = {};
  UPBPSceneSegmentFailure failure = UPBPSceneSegmentFailure::None;
  MediumTrackingFailure medium_failure = MediumTrackingFailure::None;
  bool visible = false;
};

inline bool upbp_sample_connection_transmittance(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const UPBPPathVertexRecord& source,
  const float3& target_position, Sampler& intersection_sampler, Sampler& medium_sampler, const uint32_t maximum_boundary_count, const uint32_t maximum_null_events_per_interval,
  UPBPConnectionTransmittanceResult& result) {
  result = {};
  result.weight = SpectralResponse{spect, 0.0f};
  float3 direction = target_position - source.position;
  const float distance_squared = dot(direction, direction);
  if (distance_squared <= kRayEpsilon * kRayEpsilon) {
    return true;
  }
  const float distance = sqrtf(distance_squared);
  direction /= distance;

  float3 origin = source.position;
  if (source.cls == UPBPVertexClass::Surface) {
    const Triangle& triangle = scene.triangles[source.intersection.triangle_index];
    origin = shading_pos(scene, triangle, source.intersection.barycentric, direction, source.intersection.instance_index);
  }
  direction = target_position - origin;
  const float offset_distance_squared = dot(direction, direction);
  if (offset_distance_squared <= kRayEpsilon * kRayEpsilon) {
    return true;
  }
  const float offset_distance = sqrtf(offset_distance_squared);
  direction /= offset_distance;
  const float maximum_distance = offset_distance - fmaxf(kRayEpsilon, offset_distance * kRayEpsilon);
  if (maximum_distance <= kRayEpsilon) {
    return true;
  }

  const float minimum_distance = source.cls == UPBPVertexClass::Medium ? 0.0f : kRayEpsilon;
  const Ray ray = {origin, direction, minimum_distance, maximum_distance};
  UPBPSceneSegmentResult scene_segment = {};
  if (upbp_walk_scene_segment(rt, scene, spect, intersection_sampler, medium_sampler, ray, upbp_connection_medium(scene, source, direction), maximum_boundary_count,
        maximum_null_events_per_interval, scene_segment) == false) {
    result.segment = std::move(scene_segment.segment);
    result.failure = scene_segment.failure;
    result.medium_failure = scene_segment.medium_failure;
    result.failure_intersection = scene_segment.intersection;
    return false;
  }

  result.segment = std::move(scene_segment.segment);
  result.visible = scene_segment.terminal == UPBPSceneSegmentTerminal::Miss;
  result.weight = result.visible ? result.segment.weight : SpectralResponse{spect, 0.0f};
  return true;
}

struct UPBPVertexConnectionResult {
  UPBPConnectionTransmittanceResult transmittance = {};
  UPBPScatteringEval light_scattering = {};
  UPBPScatteringEval camera_scattering = {};
  SpectralResponse contribution = {};
  float distance_squared = 0.0f;
  bool applicable = false;
};

inline bool upbp_evaluate_vertex_connection(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const UPBPPathVertexRecord& light_vertex,
  const UPBPPathVertexRecord& camera_vertex, Sampler& scattering_sampler, Sampler& intersection_sampler, Sampler& medium_sampler, const uint32_t maximum_boundary_count,
  const uint32_t maximum_null_events_per_interval, UPBPVertexConnectionResult& result) {
  result = {};
  result.contribution = SpectralResponse{spect, 0.0f};
  if ((light_vertex.connectible == false) || (camera_vertex.connectible == false)) {
    return true;
  }

  float3 direction = camera_vertex.position - light_vertex.position;
  result.distance_squared = dot(direction, direction);
  if (result.distance_squared <= kRayEpsilon * kRayEpsilon) {
    return true;
  }
  direction /= sqrtf(result.distance_squared);
  result.light_scattering = upbp_evaluate_vertex_scattering(scene, spect, light_vertex, direction, scattering_sampler);
  result.camera_scattering = upbp_evaluate_vertex_scattering(scene, spect, camera_vertex, -direction, scattering_sampler);
  if ((result.light_scattering.valid() == false) || (result.camera_scattering.valid() == false)) {
    return true;
  }

  if (upbp_sample_connection_transmittance(rt, scene, spect, light_vertex, camera_vertex.position, intersection_sampler, medium_sampler, maximum_boundary_count,
        maximum_null_events_per_interval, result.transmittance) == false) {
    return false;
  }
  if (result.transmittance.visible == false) {
    return true;
  }

  result.applicable = true;
  result.contribution =
    light_vertex.throughput * camera_vertex.throughput * result.light_scattering.value * result.camera_scattering.value * result.transmittance.weight / result.distance_squared;
  return true;
}

}  // namespace etx
