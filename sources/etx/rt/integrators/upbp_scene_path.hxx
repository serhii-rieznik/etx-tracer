#pragma once

#include <etx/rt/integrators/upbp_core.hxx>
#include <etx/rt/rt.hxx>
#include <etx/render/interop/medium_position_shared.hxx>

namespace etx {

enum class UPBPSceneSegmentTerminal : uint8_t {
  Miss,
  Surface,
  Scatter,
  Absorb,
  Failure,
};

enum class UPBPSceneSegmentFailure : uint8_t {
  None,
  InvalidRay,
  InvalidIntervalDistance,
  InvalidBoundaryContinuation,
  MediumTracking,
  BoundaryLimitExceeded,
  SubsurfaceExitNotFound,
  InvalidSubsurfaceExitDistance,
  SubsurfaceExitMaterialMismatch,
};

struct UPBPSceneSegmentResult {
  UPBPTransportSegmentRecord segment = {};
  Intersection intersection = {};
  MediumTrackingEvent medium_event = {};
  uint32_t active_medium_index = kInvalidIndex;
  UPBPSceneSegmentTerminal terminal = UPBPSceneSegmentTerminal::Failure;
  UPBPSceneSegmentFailure failure = UPBPSceneSegmentFailure::None;
  MediumTrackingFailure medium_failure = MediumTrackingFailure::None;

  bool valid() const {
    return (failure == UPBPSceneSegmentFailure::None) && (terminal != UPBPSceneSegmentTerminal::Failure) && segment.valid();
  }
};

inline UPBPSegmentRecord upbp_vacuum_interval(const SpectralQuery spect, const float3& origin, const float3& direction, const float distance) {
  UPBPSegmentRecord result = {};
  result.reset(spect, kInvalidIndex, origin);
  result.distance = distance;
  result.end_position = origin + direction * distance;
  result.terminal_event = MediumTrackingEventType::Escape;
  result.complete = true;
  return result;
}

inline float upbp_distance_to_scene_sphere_exit(const float3& origin, const float3& direction, const float3& center, const float radius) {
  const float3 offset = origin - center;
  const float projected = dot(direction, offset);
  const float discriminant = projected * projected - dot(offset, offset) + radius * radius;
  if (discriminant < 0.0f) {
    return 0.0f;
  }
  return -projected + sqrtf(discriminant);
}

template <typename TraceFunction, typename ResetTraceStateFunction>
bool upbp_trace_with_medium_origin_retry(const Ray& ray, TraceFunction&& trace, ResetTraceStateFunction&& reset_trace_state, Intersection& intersection) {
  if (trace(ray, intersection)) {
    return true;
  }
  if (ray.min_t > 0.0f) {
    return false;
  }

  Ray retry = ray;
  retry.o -= retry.d * kRayEpsilon;
  retry.min_t = kRayEpsilon;
  retry.max_t = ray.max_t < (kMaxFloat - kRayEpsilon) ? ray.max_t + kRayEpsilon : kMaxFloat;
  reset_trace_state();
  if (trace(retry, intersection) == false) {
    return false;
  }
  intersection.t -= kRayEpsilon;
  return intersection.t > 0.0f;
}

inline bool upbp_walk_scene_segment(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, Sampler& intersection_sampler, Sampler& medium_sampler,
  const Ray& input_ray, const uint32_t initial_medium_index, const uint32_t maximum_boundary_count, const uint32_t maximum_null_events_per_interval,
  UPBPSceneSegmentResult& result) {
  result = {};
  result.segment.reset(spect);
  result.active_medium_index = initial_medium_index;

  const float direction_length_squared = dot(input_ray.d, input_ray.d);
  if ((std::isfinite(direction_length_squared) == false) || (fabsf(direction_length_squared - 1.0f) > 1.0e-4f) || (input_ray.max_t <= 0.0f) ||
      (std::isfinite(input_ray.max_t) == false)) {
    result.failure = UPBPSceneSegmentFailure::InvalidRay;
    return false;
  }

  Ray ray = input_ray;
  const bool bounded_segment = input_ray.max_t < 0.5f * kMaxFloat;
  const float3 segment_target = bounded_segment ? input_ray.o + input_ray.d * input_ray.max_t : float3{};
  for (;;) {
    Intersection intersection = {};
    const Sampler initial_intersection_sampler = intersection_sampler;
    const bool found_intersection = upbp_trace_with_medium_origin_retry(
      ray,
      [&rt, &scene, &intersection_sampler](const Ray& trace_ray, Intersection& trace_intersection) {
        return rt.trace(scene, trace_ray, trace_intersection, intersection_sampler);
      },
      [&intersection_sampler, &initial_intersection_sampler]() {
        intersection_sampler = initial_intersection_sampler;
      },
      intersection);
    float interval_distance = found_intersection ? intersection.t : ray.max_t;
    if ((found_intersection == false) && (ray.max_t >= 0.5f * kMaxFloat)) {
      interval_distance = upbp_distance_to_scene_sphere_exit(ray.o, ray.d, scene.bounding_sphere_center, scene.bounding_sphere_radius);
    }
    if ((interval_distance <= 0.0f) || (std::isfinite(interval_distance) == false)) {
      result.failure = UPBPSceneSegmentFailure::InvalidIntervalDistance;
      return false;
    }

    UPBPSegmentRecord interval = {};
    if (result.active_medium_index != kInvalidIndex) {
      if (result.active_medium_index >= scene.mediums.count) {
        result.failure = UPBPSceneSegmentFailure::InvalidRay;
        return false;
      }

      const Medium& medium = scene.mediums[result.active_medium_index];
      if (upbp_track_medium_segment(medium, spect, medium_sampler, ray.o, ray.d, interval_distance, result.active_medium_index, maximum_null_events_per_interval, interval) ==
          false) {
        result.segment.failure = interval.failure;
        result.failure = UPBPSceneSegmentFailure::MediumTracking;
        result.medium_failure = interval.failure;
        return false;
      }
    } else {
      interval = upbp_vacuum_interval(spect, ray.o, ray.d, interval_distance);
    }

    if (found_intersection && (interval.terminal_event == MediumTrackingEventType::Scatter)) {
      const Triangle& triangle = scene.triangles[intersection.triangle_index];
      const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, intersection.instance_index);
      interval.end_position = medium_position_before_surface(interval.end_position, intersection.pos, geometric_normal, ray.d);
    }
    if (result.segment.append(interval) == false) {
      result.failure = UPBPSceneSegmentFailure::MediumTracking;
      result.medium_failure = result.segment.failure;
      return false;
    }

    if (interval.terminal_event == MediumTrackingEventType::Scatter) {
      result.medium_event.type = MediumTrackingEventType::Scatter;
      result.medium_event.position = interval.end_position;
      result.terminal = UPBPSceneSegmentTerminal::Scatter;
      return true;
    }

    if (interval.terminal_event == MediumTrackingEventType::Absorb) {
      result.medium_event.type = MediumTrackingEventType::Absorb;
      result.medium_event.position = interval.end_position;
      result.terminal = UPBPSceneSegmentTerminal::Absorb;
      return true;
    }

    if (found_intersection == false) {
      result.terminal = UPBPSceneSegmentTerminal::Miss;
      return true;
    }

    const Material& material = scene.materials[intersection.material_index];
    if (material.cls != MaterialClass::Boundary) {
      result.intersection = intersection;
      result.terminal = UPBPSceneSegmentTerminal::Surface;
      return true;
    }

    result.segment.record_boundary();
    if (result.segment.boundary_count > maximum_boundary_count) {
      result.intersection = intersection;
      result.failure = UPBPSceneSegmentFailure::BoundaryLimitExceeded;
      return false;
    }

    const Triangle& triangle = scene.triangles[intersection.triangle_index];
    const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, intersection.instance_index);
    result.active_medium_index = (dot(geometric_normal, ray.d) < 0.0f) ? material.int_medium : material.ext_medium;
    const float normal_direction = dot(geometric_normal, ray.d) >= 0.0f ? 1.0f : -1.0f;
    ray.o = offset_ray(intersection.pos, geometric_normal * normal_direction);
    ray.min_t = kRayEpsilon;
    ray.max_t = bounded_segment ? dot(segment_target - ray.o, ray.d) : ray.max_t - intersection.t;
    if (ray.max_t <= kRayEpsilon) {
      if (bounded_segment) {
        result.terminal = UPBPSceneSegmentTerminal::Miss;
        return true;
      }
      result.failure = UPBPSceneSegmentFailure::InvalidBoundaryContinuation;
      return false;
    }
  }
}

}  // namespace etx
