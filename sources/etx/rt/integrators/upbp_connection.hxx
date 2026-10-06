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

  if ((vertex.cls != UPBPVertexClass::Surface) || (scattering_direction_valid(scene, vertex.intersection, vertex.intersection.w_i, outgoing_direction) == false)) {
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
  if ((dot(geometric_normal, vertex.intersection.w_i) * dot(geometric_normal, outgoing_direction)) < 0.0f) {
    return vertex.incident_medium_index;
  }
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
  const float3& target_position, const bool target_collision, Sampler& intersection_sampler, Sampler& medium_sampler, const uint32_t maximum_boundary_count,
  const uint32_t maximum_null_events_per_interval, UPBPConnectionTransmittanceResult& result) {
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
  MediumInstance current_medium = {.index = upbp_connection_medium(scene, source, direction)};
  if (source.medium.subsurface_material != kInvalidIndex) {
    bool inside_subsurface =
      (source.cls == UPBPVertexClass::Medium) || ((source.cls == UPBPVertexClass::Surface) && (source.intersection.material_index != source.medium.subsurface_material));
    if ((source.cls == UPBPVertexClass::Surface) && (inside_subsurface == false)) {
      const Triangle& triangle = scene.triangles[source.intersection.triangle_index];
      const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, source.intersection.instance_index);
      inside_subsurface = dot(geometric_normal, direction) < 0.0f;
    }
    if (inside_subsurface) {
      current_medium = source.medium;
    }
  }
  result.segment.reset(spect);
  float traveled_distance = 0.0f;
  Ray interval_ray = ray;
  for (;;) {
    Intersection intersection = {};
    const Sampler initial_intersection_sampler = intersection_sampler;
    const bool found_intersection = upbp_trace_with_medium_origin_retry(
      interval_ray,
      [&rt, &scene, &intersection_sampler](const Ray& trace_ray, Intersection& trace_intersection) {
        return rt.trace(scene, trace_ray, trace_intersection, intersection_sampler);
      },
      [&intersection_sampler, &initial_intersection_sampler]() {
        intersection_sampler = initial_intersection_sampler;
      },
      intersection);
    const float interval_distance = (found_intersection ? intersection.t : ray.max_t) - traveled_distance;
    if ((interval_distance <= 0.0f) || (std::isfinite(interval_distance) == false)) {
      result.failure = UPBPSceneSegmentFailure::InvalidIntervalDistance;
      result.failure_intersection = intersection;
      return false;
    }
    const float3 interval_origin = ray.o + direction * traveled_distance;
    UPBPSegmentRecord interval = {};
    if (current_medium.subsurface_packing > 0.0f) {
      const bool source_collision = (source.cls == UPBPVertexClass::Medium) && (traveled_distance == 0.0f);
      const bool interval_target_collision = target_collision && (found_intersection == false);
      const float flight_distance = found_intersection ? interval_distance : (offset_distance - traveled_distance);
      const SpectralResponse scattering = subsurface_medium_scattering(scene, spect, current_medium);
      const float kernel_forward = subsurface_transport_sampling_kernel(current_medium.extinction, scattering, current_medium.subsurface_packing, source_collision,
        interval_target_collision, flight_distance);
      const float kernel_reverse = subsurface_transport_sampling_kernel(current_medium.extinction, scattering, current_medium.subsurface_packing, interval_target_collision,
        source_collision, flight_distance);
      if ((kernel_forward <= 0.0f) || (kernel_reverse <= 0.0f)) {
        return true;
      }
      interval.reset(spect, current_medium.index, interval_origin);
      interval.distance = flight_distance;
      interval.set_subsurface_law(current_medium.extinction, scattering, current_medium.subsurface_packing, source_collision, interval_target_collision);
      interval.end_position = interval_origin + direction * flight_distance;
      interval.weight = subsurface_transport_kernel(current_medium.extinction, current_medium.subsurface_packing, source_collision, interval_target_collision, flight_distance);
      interval.log_transport_pdf_forward = std::log(static_cast<double>(kernel_forward));
      interval.log_transport_pdf_reverse = std::log(static_cast<double>(kernel_reverse));
      interval.log_pdf_forward = interval.log_transport_pdf_forward;
      interval.log_pdf_reverse = interval.log_transport_pdf_reverse;
      interval.terminal_event = MediumTrackingEventType::Escape;
      interval.complete = true;
    } else if (current_medium.index != kInvalidIndex) {
      if (upbp_track_medium_segment(scene.mediums[current_medium.index], spect, medium_sampler, interval_origin, direction, interval_distance, current_medium.index,
            maximum_null_events_per_interval, interval) == false) {
        result.failure = UPBPSceneSegmentFailure::MediumTracking;
        result.medium_failure = interval.failure;
        return false;
      }
    } else if (current_medium.subsurface_material != kInvalidIndex) {
      const MediumTrackingInput input = {.spect = spect, .scattering = SpectralResponse{spect, 0.0f}, .absorption = current_medium.extinction, .density_majorant = 1.0f};
      if (upbp_track_medium_segment(
            input, medium_sampler, interval_origin, direction, interval_distance, kInvalidIndex, maximum_null_events_per_interval,
            [](const float3&) {
              return 1.0f;
            },
            interval) == false) {
        result.failure = UPBPSceneSegmentFailure::MediumTracking;
        result.medium_failure = interval.failure;
        return false;
      }
    } else {
      interval = upbp_vacuum_interval(spect, interval_origin, direction, interval_distance);
    }
    if (result.segment.append(interval) == false) {
      result.failure = UPBPSceneSegmentFailure::MediumTracking;
      result.medium_failure = result.segment.failure;
      return false;
    }
    if (interval.terminal_event != MediumTrackingEventType::Escape) {
      return true;
    }
    if (found_intersection == false) {
      result.visible = true;
      result.weight = result.segment.weight;
      return true;
    }
    const Material& material = scene.materials[intersection.material_index];
    const bool incident_boundary = material_has_incident_subsurface_boundary(material);
    if ((material.cls != MaterialClass::Boundary) && (incident_boundary == false)) {
      return true;
    }
    result.segment.record_boundary();
    if (result.segment.boundary_count > maximum_boundary_count) {
      result.failure = UPBPSceneSegmentFailure::BoundaryLimitExceeded;
      result.failure_intersection = intersection;
      return false;
    }
    const Triangle& triangle = scene.triangles[intersection.triangle_index];
    const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, intersection.instance_index);
    const bool entering = dot(geometric_normal, direction) < 0.0f;
    if (incident_boundary) {
      const SpectralResponse boundary_color = subsurface_boundary_color(scene, spect, material, intersection.tex);
      result.segment.weight *= boundary_color;
      result.segment.intervals.back().weight *= boundary_color;
    }
    current_medium = entering && incident_boundary ? make_subsurface_medium_instance(scene, spect, intersection.material_index)
                                                   : MediumInstance{.index = entering ? material.int_medium : material.ext_medium};
    traveled_distance = intersection.t;
    interval_ray.min_t = std::nextafter(traveled_distance, kMaxFloat);
    if (interval_ray.min_t >= ray.max_t) {
      result.visible = true;
      result.weight = result.segment.weight;
      return true;
    }
  }
}

inline bool upbp_sample_connection_transmittance(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const UPBPPathVertexRecord& source,
  const float3& target_position, Sampler& intersection_sampler, Sampler& medium_sampler, const uint32_t maximum_boundary_count, const uint32_t maximum_null_events_per_interval,
  UPBPConnectionTransmittanceResult& result) {
  return upbp_sample_connection_transmittance(rt, scene, spect, source, target_position, false, intersection_sampler, medium_sampler, maximum_boundary_count,
    maximum_null_events_per_interval, result);
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

  if (upbp_sample_connection_transmittance(rt, scene, spect, light_vertex, camera_vertex.position, camera_vertex.cls == UPBPVertexClass::Medium, intersection_sampler,
        medium_sampler, maximum_boundary_count, maximum_null_events_per_interval, result.transmittance) == false) {
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
