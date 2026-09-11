#pragma once

#include <etx/render/host/film.hxx>
#include <etx/rt/integrators/upbp_scene_path.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

namespace etx {

enum class UPBPSubpathFailure : uint8_t {
  None,
  InvalidInput,
  SegmentTraversal,
  InvalidScatteringSample,
  PathCapacity,
};

struct UPBPSubpathBuildInput {
  SpectralQuery spect = {};
  PathSource source = PathSource::Undefined;
  uint32_t render_seed = 0u;
  uint64_t iteration = 0u;
  uint64_t path_index = 0u;
  uint32_t maximum_physical_vertices = 0u;
  uint32_t maximum_boundary_count = 0u;
  uint32_t maximum_null_events_per_interval = 0u;
};

struct UPBPSubpathBuildResult {
  UPBPPathRecord path = {};
  SpectralResponse throughput = {};
  Ray terminal_ray = {};
  uint32_t active_medium_index = kInvalidIndex;
  UPBPSceneSegmentTerminal terminal = UPBPSceneSegmentTerminal::Failure;
  UPBPSceneSegmentFailure segment_failure = UPBPSceneSegmentFailure::None;
  UPBPSubpathFailure failure = UPBPSubpathFailure::None;

  bool valid() const {
    return (failure == UPBPSubpathFailure::None) && path.valid();
  }
};

struct UPBPCameraSubpathResult {
  UPBPSubpathBuildResult subpath = {};
  Ray camera_ray = {};
  CameraEval camera_eval = {};
};

struct UPBPLightSubpathResult {
  UPBPSubpathBuildResult subpath = {};
  EmitterSample emitter_sample = {};
};

struct UPBPSubsurfaceState {
  MediumTrackingInput tracking = {};
  MediumInstance medium = {};
  uint32_t material_index = kInvalidIndex;
  bool active = false;
};

inline bool upbp_surface_arrival_has_positive_measure(const Scene& scene, const Intersection& intersection) {
  const Triangle& triangle = scene.triangles[intersection.triangle_index];
  const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, intersection.instance_index);
  return upbp_projected_measure_valid(fabs(static_cast<double>(dot(intersection.nrm, intersection.w_i)))) &&
         upbp_projected_measure_valid(fabs(static_cast<double>(dot(geometric_normal, intersection.w_i))));
}

inline bool upbp_surface_arrival_has_recursive_measure(const Scene& scene, const float3& source_position, const Intersection& intersection) {
  const float3 edge = intersection.pos - source_position;
  const float edge_length_squared = dot(edge, edge);
  if ((edge_length_squared <= 0.0f) || (std::isfinite(edge_length_squared) == false)) {
    return false;
  }
  const Triangle& triangle = scene.triangles[intersection.triangle_index];
  const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, intersection.instance_index);
  const float3 edge_direction = edge / sqrtf(edge_length_squared);
  return upbp_projected_measure_valid(fabs(static_cast<double>(dot(geometric_normal, edge_direction))));
}

inline bool upbp_surface_departure_is_valid(const Scene& scene, const Intersection& intersection, const float3& outgoing_direction, const uint32_t sample_properties) {
  const Triangle& triangle = scene.triangles[intersection.triangle_index];
  const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, intersection.instance_index);
  const double incoming_shading_cosine = static_cast<double>(dot(intersection.nrm, intersection.w_i));
  const double outgoing_shading_cosine = static_cast<double>(dot(intersection.nrm, outgoing_direction));
  const double incoming_geometric_cosine = static_cast<double>(dot(geometric_normal, intersection.w_i));
  const double outgoing_geometric_cosine = static_cast<double>(dot(geometric_normal, outgoing_direction));
  if ((upbp_projected_measure_valid(fabs(outgoing_shading_cosine)) == false) || (upbp_projected_measure_valid(fabs(outgoing_geometric_cosine)) == false)) {
    return false;
  }

  const bool transmission = (sample_properties & BSDFSample::Transmission) != 0u;
  const bool crosses_shading_surface = (incoming_shading_cosine * outgoing_shading_cosine) > 0.0;
  const bool crosses_geometric_surface = (incoming_geometric_cosine * outgoing_geometric_cosine) > 0.0;
  return transmission ? (crosses_shading_surface && crosses_geometric_surface) : ((crosses_shading_surface == false) && (crosses_geometric_surface == false));
}

inline bool upbp_make_subsurface_state(const Scene& scene, const SpectralQuery spect, const Material& material, const Intersection& intersection, UPBPSubsurfaceState& result) {
  result = {};
  result.tracking.spect = spect;
  result.tracking.density_majorant = 1.0f;
  result.material_index = intersection.material_index;
  if (material.int_medium != kInvalidIndex) {
    if (material.int_medium >= scene.mediums.count) {
      return false;
    }
    const Medium& medium = scene.mediums[material.int_medium];
    result.tracking.scattering = medium_scattering(medium, spect);
    result.tracking.absorption = medium_absorption(medium, spect);
    result.medium = make_medium_instance(medium, spect, material.int_medium);
  } else {
    const SpectralResponse color = apply_image(spect, material.scattering, intersection.tex);
    const SpectralResponse distances = apply_image(spect, material.subsurface, intersection.tex);
    SpectralResponse albedo{spect};
    SpectralResponse extinction{spect};
    SpectralResponse scattering{spect};
    subsurface::remap(color.integrated, distances.integrated, albedo.integrated, extinction.integrated, scattering.integrated);
    subsurface::remap_channel(color.value, distances.value, albedo.value, extinction.value, scattering.value);
    result.tracking.scattering = scattering;
    result.tracking.absorption = extinction - scattering;
    result.medium = {.extinction = extinction, .anisotropy = 0.0f, .index = kInvalidIndex};
  }
  result.active = medium_tracking_majorant(result.tracking) > 0.0f;
  return result.active;
}

inline bool upbp_walk_subsurface_segment(const Raytracing& rt, const Scene& scene, const UPBPSubsurfaceState& subsurface_state, Sampler& intersection_sampler,
  Sampler& medium_sampler, const Ray& ray, const uint32_t maximum_null_events_per_interval, UPBPSceneSegmentResult& result) {
  result = {};
  result.segment.reset(subsurface_state.tracking.spect);
  result.active_medium_index = subsurface_state.medium.index;
  Intersection exit_intersection = {};
  const Sampler initial_exit_sampler = intersection_sampler;
  const bool exit_found = upbp_trace_with_medium_origin_retry(
    ray,
    [&rt, &scene, &subsurface_state, &intersection_sampler](const Ray& trace_ray, Intersection& trace_intersection) {
      return rt.trace_material(scene, trace_ray, subsurface_state.material_index, trace_intersection, intersection_sampler);
    },
    [&intersection_sampler, &initial_exit_sampler]() {
      intersection_sampler = initial_exit_sampler;
    },
    exit_intersection);
  if (exit_found == false) {
    Intersection other_intersection = {};
    const Sampler initial_surface_sampler = intersection_sampler;
    const bool other_surface_found = upbp_trace_with_medium_origin_retry(
      ray,
      [&rt, &scene, &intersection_sampler](const Ray& trace_ray, Intersection& trace_intersection) {
        return rt.trace(scene, trace_ray, trace_intersection, intersection_sampler);
      },
      [&intersection_sampler, &initial_surface_sampler]() {
        intersection_sampler = initial_surface_sampler;
      },
      other_intersection);
    if (other_surface_found) {
      result.intersection = other_intersection;
      result.failure = UPBPSceneSegmentFailure::SubsurfaceExitMaterialMismatch;
      return false;
    }
    result.failure = UPBPSceneSegmentFailure::SubsurfaceExitNotFound;
    return false;
  }
  if ((exit_intersection.t <= 0.0f) || (std::isfinite(exit_intersection.t) == false)) {
    result.intersection = exit_intersection;
    result.failure = UPBPSceneSegmentFailure::InvalidSubsurfaceExitDistance;
    return false;
  }

  UPBPSegmentRecord interval = {};
  if (upbp_track_medium_segment(
        subsurface_state.tracking, medium_sampler, ray.o, ray.d, exit_intersection.t, kInvalidIndex, maximum_null_events_per_interval,
        [](const float3&) {
          return 1.0f;
        },
        interval) == false) {
    result.segment.failure = interval.failure;
    result.failure = UPBPSceneSegmentFailure::MediumTracking;
    return false;
  }
  if (interval.terminal_event == MediumTrackingEventType::Scatter) {
    const Triangle& triangle = scene.triangles[exit_intersection.triangle_index];
    const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, exit_intersection.instance_index);
    interval.end_position = medium_position_before_surface(interval.end_position, exit_intersection.pos, geometric_normal, ray.d);
  }
  if (result.segment.append(interval) == false) {
    result.failure = UPBPSceneSegmentFailure::MediumTracking;
    return false;
  }
  if (interval.terminal_event == MediumTrackingEventType::Scatter) {
    result.medium_event.type = MediumTrackingEventType::Scatter;
    result.medium_event.position = interval.end_position;
    result.terminal = UPBPSceneSegmentTerminal::Scatter;
  } else if (interval.terminal_event == MediumTrackingEventType::Absorb) {
    result.medium_event.type = MediumTrackingEventType::Absorb;
    result.medium_event.position = interval.end_position;
    result.terminal = UPBPSceneSegmentTerminal::Absorb;
  } else {
    result.intersection = exit_intersection;
    result.intersection.material_index = scene.defaults.subsurface_scatter_material;
    result.terminal = UPBPSceneSegmentTerminal::Surface;
  }
  return true;
}

inline UPBPPathVertexRecord upbp_make_emitter_endpoint(const Scene& scene, const SpectralQuery spect, const EmitterSample& emitter_sample) {
  UPBPPathVertexRecord endpoint = {};
  endpoint.position = emitter_sample.origin;
  endpoint.sampled_direction = emitter_sample.direction;
  endpoint.intersection.triangle_index = emitter_sample.triangle_index;
  endpoint.intersection.barycentric = emitter_sample.barycentric;
  endpoint.intersection.pos = emitter_sample.origin;
  endpoint.intersection.nrm = emitter_sample.normal;
  endpoint.intersection.w_i = emitter_sample.direction;
  endpoint.intersection.emitter_index = emitter_sample.emitter_index;
  endpoint.intersection.instance_index = emitter_sample.instance_index;
  if (emitter_sample.triangle_index != kInvalidIndex) {
    endpoint.intersection.tex = lerp_vertex(scene, scene.triangles[emitter_sample.triangle_index], emitter_sample.barycentric).tex;
  } else {
    endpoint.intersection.tex = emitter_sample.image_uv;
  }
  endpoint.throughput = emitter_sample.value;
  endpoint.scatter_pdf_forward = emitter_sample.pdf_dir;
  endpoint.scatter_pdf_reverse = emitter_sample.pdf_dir_out;
  endpoint.endpoint_pdf_area = emitter_sample.pdf_area;
  endpoint.endpoint_pdf_sample = emitter_sample.pdf_sample;
  endpoint.endpoint_pdf_direction = emitter_sample.pdf_dir;
  endpoint.incident_medium_index = emitter_sample.medium_index;
  endpoint.outgoing_medium_index = emitter_sample.medium_index;
  endpoint.cls = UPBPVertexClass::Emitter;
  endpoint.source = PathSource::Light;
  endpoint.connectible = true;
  endpoint.delta = emitter_sample.is_delta;
  endpoint.distant_endpoint = emitter_sample.is_distant;
  return endpoint;
}

inline void upbp_reset_subpath_build_result(UPBPSubpathBuildResult& result, const uint32_t maximum_physical_vertices, const uint32_t initial_capacity) {
  result.path.reset(maximum_physical_vertices, initial_capacity);
  result.throughput = {};
  result.terminal_ray = {};
  result.active_medium_index = kInvalidIndex;
  result.terminal = UPBPSceneSegmentTerminal::Failure;
  result.segment_failure = UPBPSceneSegmentFailure::None;
  result.failure = UPBPSubpathFailure::None;
}

inline bool upbp_build_subpath(const Raytracing& rt, const Scene& scene, const UPBPSubpathBuildInput& input, Sampler& path_sampler, const Ray& initial_ray,
  const UPBPPathVertexRecord& endpoint, const SpectralResponse& initial_throughput, const uint32_t initial_medium_index, UPBPSubpathBuildResult& result) {
  constexpr uint32_t kCameraPathInitialCapacity = 8u;
  const uint32_t initial_capacity = input.source == PathSource::Camera ? kCameraPathInitialCapacity : 0u;
  upbp_reset_subpath_build_result(result, input.maximum_physical_vertices, initial_capacity);
  result.throughput = initial_throughput;
  result.terminal_ray = initial_ray;
  result.active_medium_index = initial_medium_index;

  if (((input.source != PathSource::Camera) && (input.source != PathSource::Light)) || (input.maximum_physical_vertices < 2u) || (result.path.append_endpoint(endpoint) == false)) {
    result.failure = UPBPSubpathFailure::InvalidInput;
    return false;
  }
  if (initial_throughput.is_zero()) {
    if (input.source != PathSource::Light) {
      result.failure = UPBPSubpathFailure::InvalidInput;
      return false;
    }
    result.terminal = UPBPSceneSegmentTerminal::Absorb;
    return true;
  }

  if ((input.source == PathSource::Light) && endpoint.distant_endpoint &&
      (upbp_distance_to_scene_sphere_exit(initial_ray.o, initial_ray.d, scene.bounding_sphere_center, scene.bounding_sphere_radius) <= 0.0f)) {
    result.terminal = UPBPSceneSegmentTerminal::Miss;
    return true;
  }

  Ray ray = initial_ray;
  float eta = 1.0f;
  UPBPSubsurfaceState subsurface_state = {};
  while (result.path.vertices.size() < input.maximum_physical_vertices) {
    const uint32_t physical_depth = result.path.physical_length();
    const uint32_t segment_index = static_cast<uint32_t>(result.path.segments.size());
    const UPBPRandomDomain medium_tracking_domain = upbp_medium_tracking_random_domain(input.source);
    Sampler medium_sampler{upbp_sampler_seed(input.render_seed, input.iteration, input.path_index, physical_depth, segment_index, medium_tracking_domain)};
    if (result.path.boundary_count > input.maximum_boundary_count) {
      result.failure = UPBPSubpathFailure::SegmentTraversal;
      result.segment_failure = UPBPSceneSegmentFailure::BoundaryLimitExceeded;
      return false;
    }
    const uint32_t remaining_boundary_count = input.maximum_boundary_count - result.path.boundary_count;

    UPBPSceneSegmentResult scene_segment = {};
    const bool inside_subsurface = subsurface_state.active;
    const bool segment_valid =
      inside_subsurface ? upbp_walk_subsurface_segment(rt, scene, subsurface_state, path_sampler, medium_sampler, ray, input.maximum_null_events_per_interval, scene_segment)
                        : upbp_walk_scene_segment(rt, scene, input.spect, path_sampler, medium_sampler, ray, result.active_medium_index, remaining_boundary_count,
                            input.maximum_null_events_per_interval, scene_segment);
    if (segment_valid == false) {
      result.segment_failure = scene_segment.failure;
      result.failure = UPBPSubpathFailure::SegmentTraversal;
      return false;
    }

    const bool exiting_subsurface = inside_subsurface && (scene_segment.terminal == UPBPSceneSegmentTerminal::Surface);
    if (exiting_subsurface) {
      subsurface_state.active = false;
    }

    result.throughput *= scene_segment.segment.weight;
    result.active_medium_index = scene_segment.active_medium_index;
    result.terminal = scene_segment.terminal;
    if ((scene_segment.terminal == UPBPSceneSegmentTerminal::Miss) || (scene_segment.terminal == UPBPSceneSegmentTerminal::Absorb)) {
      if (result.path.append_terminal_segment(scene_segment.segment) == false) {
        result.failure = UPBPSubpathFailure::PathCapacity;
        return false;
      }
      result.terminal_ray = ray;
      return true;
    }

    UPBPPathVertexRecord vertex = {};
    vertex.throughput = result.throughput;
    vertex.source = input.source;
    vertex.eta = eta;
    vertex.incident_medium_index = result.active_medium_index;
    vertex.outgoing_medium_index = result.active_medium_index;

    if (scene_segment.terminal == UPBPSceneSegmentTerminal::Scatter) {
      if ((inside_subsurface == false) && (result.active_medium_index >= scene.mediums.count)) {
        result.failure = UPBPSubpathFailure::InvalidInput;
        return false;
      }
      const MediumInstance medium_instance =
        inside_subsurface ? subsurface_state.medium : make_medium_instance(scene.mediums[result.active_medium_index], input.spect, result.active_medium_index);
      const float3 outgoing_direction = sample_phase_function(ray.d, medium_instance.anisotropy, path_sampler.next_2d());
      const float pdf_forward = phase_function(ray.d, outgoing_direction, medium_instance.anisotropy);
      const float pdf_reverse = phase_function(outgoing_direction, ray.d, medium_instance.anisotropy);
      if ((pdf_forward <= 0.0f) || (pdf_reverse <= 0.0f)) {
        result.failure = UPBPSubpathFailure::InvalidScatteringSample;
        return false;
      }

      vertex.position = scene_segment.medium_event.position;
      vertex.sampled_direction = outgoing_direction;
      vertex.intersection.pos = vertex.position;
      vertex.intersection.w_i = ray.d;
      vertex.medium = medium_instance;
      vertex.log_medium_event_density = scene_segment.segment.log_terminal_event_density;
      vertex.scatter_pdf_forward = pdf_forward;
      vertex.scatter_pdf_reverse = pdf_reverse;
      vertex.cls = UPBPVertexClass::Medium;
      vertex.connectible = inside_subsurface == false;
      vertex.density_connectible = inside_subsurface == false;
      vertex.delta = false;

      ray.o = vertex.position;
      ray.d = outgoing_direction;
      ray.min_t = 0.0f;
      ray.max_t = kMaxFloat;
    } else if (scene_segment.terminal == UPBPSceneSegmentTerminal::Surface) {
      const Intersection& intersection = scene_segment.intersection;
      if ((upbp_surface_arrival_has_positive_measure(scene, intersection) == false) ||
          (upbp_surface_arrival_has_recursive_measure(scene, result.path.vertices.back().position, intersection) == false)) {
        if (result.path.append_terminal_segment(scene_segment.segment) == false) {
          result.failure = UPBPSubpathFailure::PathCapacity;
          return false;
        }
        result.terminal_ray = ray;
        return true;
      }
      auto append_terminal_surface = [&]() {
        vertex.position = intersection.pos;
        vertex.intersection = intersection;
        vertex.cls = UPBPVertexClass::Surface;
        vertex.connectible = true;
        if (result.path.append_physical_vertex(scene_segment.segment, vertex) == false) {
          result.failure = UPBPSubpathFailure::PathCapacity;
          return false;
        }
        result.terminal_ray = ray;
        return true;
      };
      const Material& material = scene.materials[intersection.material_index];
      const BSDFData bsdf_data = {input.spect, result.active_medium_index, input.source, intersection, intersection.w_i};
      BSDFSample sample = bsdf::sample(bsdf_data, material, path_sampler);
      if (sample.valid() == false) {
        return append_terminal_surface();
      }

      const bool subsurface_path = (exiting_subsurface == false) && (material.subsurface_cls != SubsurfaceMaterial::Disabled) &&
                                   ((sample.properties & BSDFSample::Reflection) != 0u) && ((sample.properties & BSDFSample::Diffuse) != 0u);
      if (subsurface_path) {
        if (upbp_make_subsurface_state(scene, input.spect, material, intersection, subsurface_state) == false) {
          result.failure = UPBPSubpathFailure::InvalidScatteringSample;
          return false;
        }
        const bool diffuse_path = material.subsurface_path == SubsurfaceMaterial::DiffusePath;
        sample.w_o = diffuse_path ? sample_cosine_distribution(path_sampler.next_2d(), -intersection.nrm, 1.0f) : intersection.w_i;
        sample.weight = SpectralResponse{input.spect, 1.0f};
        sample.pdf = fabsf(dot(sample.w_o, intersection.nrm)) / kPi;
        sample.eta = 1.0f;
        sample.medium_index = subsurface_state.medium.index;
        sample.properties = BSDFSample::Transmission | BSDFSample::Diffuse | BSDFSample::MediumChanged;
        if (sample.pdf <= 0.0f) {
          result.failure = UPBPSubpathFailure::InvalidScatteringSample;
          return false;
        }
      }

      if (upbp_surface_departure_is_valid(scene, intersection, sample.w_o, sample.properties) == false) {
        return append_terminal_surface();
      }

      vertex.position = intersection.pos;
      vertex.sampled_direction = sample.w_o;
      vertex.intersection = intersection;
      if (subsurface_path) {
        vertex.intersection.material_index = scene.defaults.subsurface_scatter_material;
      }
      vertex.scatter_pdf_forward = sample.pdf;
      const Material& scattering_material = scene.materials[vertex.intersection.material_index];
      vertex.scatter_pdf_reverse = ((sample.properties & BSDFSample::Delta) != 0u) ? sample.pdf : bsdf::reverse_pdf(bsdf_data, sample.w_o, scattering_material, path_sampler);
      vertex.sample_properties = sample.properties;
      vertex.cls = UPBPVertexClass::Surface;
      vertex.connectible = (sample.properties & BSDFSample::Delta) == 0u;
      vertex.delta = (sample.properties & BSDFSample::Delta) != 0u;
      vertex.outgoing_medium_index = ((sample.properties & BSDFSample::MediumChanged) != 0u) ? sample.medium_index : result.active_medium_index;
      if (subsurface_path) {
        vertex.medium = subsurface_state.medium;
      } else if (vertex.outgoing_medium_index != kInvalidIndex) {
        vertex.medium = make_medium_instance(scene.mediums[vertex.outgoing_medium_index], input.spect, vertex.outgoing_medium_index);
      } else {
        vertex.medium.index = kInvalidIndex;
      }

      result.throughput *= sample.weight;
      if (input.source == PathSource::Light) {
        const Triangle& triangle = scene.triangles[intersection.triangle_index];
        const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, intersection.instance_index);
        result.throughput *= fix_shading_normal(geometric_normal, intersection.nrm, intersection.w_i, sample.w_o);
      } else {
        eta *= sample.eta;
      }

      result.active_medium_index = vertex.outgoing_medium_index;
      const Triangle& triangle = scene.triangles[intersection.triangle_index];
      ray.o = shading_pos(scene, triangle, intersection.barycentric, sample.w_o, intersection.instance_index);
      ray.d = sample.w_o;
      ray.min_t = subsurface_path ? 0.0f : kRayEpsilon;
      ray.max_t = kMaxFloat;
    } else {
      result.failure = UPBPSubpathFailure::SegmentTraversal;
      return false;
    }

    const UPBPRandomDomain roulette_domain = upbp_russian_roulette_random_domain(input.source);
    Sampler roulette_sampler{upbp_sampler_seed(input.render_seed, input.iteration, input.path_index, physical_depth, segment_index, roulette_domain)};
    const bool continue_path = random_continue(physical_depth, scene.options.random_path_termination, eta, roulette_sampler, result.throughput);
    vertex.outgoing_throughput = continue_path ? result.throughput : SpectralResponse{input.spect, 0.0f};
    if (result.path.append_physical_vertex(scene_segment.segment, vertex) == false) {
      result.failure = UPBPSubpathFailure::PathCapacity;
      return false;
    }
    result.terminal_ray = ray;
    if (continue_path == false) {
      return true;
    }
  }

  return true;
}

inline bool upbp_build_camera_subpath(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const float2& film_uv, const uint32_t render_seed,
  const uint64_t iteration, const uint64_t path_index, const uint32_t maximum_physical_vertices, const uint32_t maximum_boundary_count,
  const uint32_t maximum_null_events_per_interval, UPBPCameraSubpathResult& result) {
  constexpr uint32_t kCameraPathInitialCapacity = 8u;
  upbp_reset_subpath_build_result(result.subpath, maximum_physical_vertices, kCameraPathInitialCapacity);
  result.camera_ray = {};
  result.camera_eval = {};
  Sampler path_sampler{upbp_sampler_seed(render_seed, iteration, path_index, 0u, 0u, UPBPRandomDomain::CameraPath)};
  result.camera_ray = generate_ray(rt.camera(), film_uv, path_sampler.next_2d());
  result.camera_eval = film_evaluate_out(spect, rt.camera(), result.camera_ray);
  if (result.camera_eval.pdf_dir <= 0.0f) {
    result.subpath.failure = UPBPSubpathFailure::InvalidInput;
    return false;
  }

  UPBPPathVertexRecord endpoint = {};
  endpoint.position = result.camera_ray.o;
  endpoint.sampled_direction = result.camera_ray.d;
  endpoint.intersection.pos = result.camera_ray.o;
  endpoint.intersection.nrm = result.camera_eval.normal;
  endpoint.intersection.w_i = result.camera_ray.d;
  endpoint.throughput = SpectralResponse{spect, 1.0f};
  endpoint.outgoing_throughput = endpoint.throughput;
  endpoint.scatter_pdf_forward = result.camera_eval.pdf_dir;
  endpoint.scatter_pdf_reverse = result.camera_eval.pdf_dir;
  const float lens_area = (rt.camera().lens_radius > kEpsilon) ? (kPi * sqr(rt.camera().lens_radius)) : 1.0f;
  endpoint.endpoint_pdf_area = 1.0f / lens_area;
  endpoint.endpoint_pdf_sample = 1.0f;
  endpoint.endpoint_pdf_direction = result.camera_eval.pdf_dir;
  endpoint.incident_medium_index = rt.camera().medium_index;
  endpoint.outgoing_medium_index = rt.camera().medium_index;
  endpoint.cls = UPBPVertexClass::Camera;
  endpoint.source = PathSource::Camera;
  endpoint.connectible = true;

  const UPBPSubpathBuildInput input = {
    spect,
    PathSource::Camera,
    render_seed,
    iteration,
    path_index,
    maximum_physical_vertices,
    maximum_boundary_count,
    maximum_null_events_per_interval,
  };
  return upbp_build_subpath(rt, scene, input, path_sampler, result.camera_ray, endpoint, endpoint.throughput, rt.camera().medium_index, result.subpath);
}

inline bool upbp_build_light_subpath(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const uint32_t render_seed, const uint64_t iteration,
  const uint64_t path_index, const uint32_t maximum_physical_vertices, const uint32_t maximum_boundary_count, const uint32_t maximum_null_events_per_interval,
  UPBPLightSubpathResult& result) {
  upbp_reset_subpath_build_result(result.subpath, maximum_physical_vertices, 0u);
  result.emitter_sample = {};
  Sampler path_sampler{upbp_sampler_seed(render_seed, iteration, path_index, 0u, 0u, UPBPRandomDomain::LightPath)};
  result.emitter_sample = sample_emission(spect, path_sampler);
  const SpectralResponse& emission = result.emitter_sample.value;
  const bool zero_emission =
    emission.spectral() ? (emission.value == 0.0f) : ((emission.integrated.x == 0.0f) && (emission.integrated.y == 0.0f) && (emission.integrated.z == 0.0f));
  if ((result.emitter_sample.pdf_sample > 0.0f) && (result.emitter_sample.emitter_index < scene.emitter_instances.count) && zero_emission) {
    // A sampled black emitter contributes zero but still counts in the emitted-path population.
    if (result.subpath.path.append_endpoint(upbp_make_emitter_endpoint(scene, spect, result.emitter_sample)) == false) {
      result.subpath.failure = UPBPSubpathFailure::InvalidInput;
      return false;
    }
    result.subpath.terminal = UPBPSceneSegmentTerminal::Absorb;
    return true;
  }
  if ((result.emitter_sample.pdf_area <= 0.0f) || (result.emitter_sample.pdf_dir <= 0.0f) || (result.emitter_sample.pdf_sample <= 0.0f)) {
    result.subpath.failure = UPBPSubpathFailure::InvalidInput;
    return false;
  }

  const float cosine = dot(result.emitter_sample.direction, result.emitter_sample.normal);
  const SpectralResponse initial_throughput =
    result.emitter_sample.value * (cosine / (result.emitter_sample.pdf_dir * result.emitter_sample.pdf_area * result.emitter_sample.pdf_sample));
  UPBPPathVertexRecord endpoint = upbp_make_emitter_endpoint(scene, spect, result.emitter_sample);
  endpoint.outgoing_throughput = initial_throughput;
  Ray ray = {
    result.emitter_sample.origin,
    result.emitter_sample.direction,
    kRayEpsilon,
    kMaxFloat,
  };
  if (result.emitter_sample.triangle_index != kInvalidIndex) {
    const Triangle& triangle = scene.triangles[result.emitter_sample.triangle_index];
    ray.o = shading_pos(scene, triangle, result.emitter_sample.barycentric, result.emitter_sample.direction, result.emitter_sample.instance_index);
  }

  const UPBPSubpathBuildInput input = {
    spect,
    PathSource::Light,
    render_seed,
    iteration,
    path_index,
    maximum_physical_vertices,
    maximum_boundary_count,
    maximum_null_events_per_interval,
  };
  return upbp_build_subpath(rt, scene, input, path_sampler, ray, endpoint, initial_throughput, result.emitter_sample.medium_index, result.subpath);
}

}  // namespace etx
