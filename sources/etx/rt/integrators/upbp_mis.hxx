#pragma once

#include <etx/rt/integrators/upbp_connection.hxx>

namespace etx {

struct UPBPJoinedPathEdge {
  const UPBPTransportSegmentRecord* segment = nullptr;
  bool reversed = false;
};

struct UPBPJoinedPath {
  std::vector<const UPBPPathVertexRecord*> vertices = {};
  std::vector<UPBPJoinedPathEdge> edges = {};
  uint32_t selected_light_vertex_count = 0u;

  bool valid() const {
    if ((vertices.size() < 2u) || (edges.size() + 1u != vertices.size()) || (selected_light_vertex_count + 1u > vertices.size())) {
      return false;
    }
    for (const UPBPJoinedPathEdge& edge : edges) {
      if ((edge.segment == nullptr) || (edge.segment->valid() == false)) {
        return false;
      }
    }
    return true;
  }
};

inline bool upbp_join_subpaths(const UPBPPathRecord& light_path, const uint32_t light_vertex_count, const UPBPPathRecord& camera_path, const uint32_t camera_vertex_count,
  const UPBPTransportSegmentRecord& connection_segment, const bool connection_reversed, UPBPJoinedPath& result) {
  result = {};
  if ((light_path.valid() == false) || (camera_path.valid() == false) || (connection_segment.valid() == false) || (light_vertex_count < 1u) || (camera_vertex_count < 1u) ||
      (light_vertex_count > light_path.vertices.size()) || (camera_vertex_count > camera_path.vertices.size())) {
    return false;
  }

  result.selected_light_vertex_count = light_vertex_count;
  result.vertices.reserve(light_vertex_count + camera_vertex_count);
  result.edges.reserve(light_vertex_count + camera_vertex_count - 1u);
  for (uint32_t index = 0u; index < light_vertex_count; ++index) {
    result.vertices.emplace_back(&light_path.vertices[index]);
  }
  for (uint32_t index = camera_vertex_count; index > 0u; --index) {
    result.vertices.emplace_back(&camera_path.vertices[index - 1u]);
  }

  for (uint32_t index = 0u; index + 1u < light_vertex_count; ++index) {
    result.edges.emplace_back(UPBPJoinedPathEdge{&light_path.segments[index], false});
  }
  result.edges.emplace_back(UPBPJoinedPathEdge{&connection_segment, connection_reversed});
  for (uint32_t index = camera_vertex_count - 1u; index > 0u; --index) {
    result.edges.emplace_back(UPBPJoinedPathEdge{&camera_path.segments[index - 1u], true});
  }
  return result.valid();
}

inline bool upbp_join_direct_hit(const UPBPPathRecord& camera_path, const uint32_t camera_vertex_count, UPBPJoinedPath& result) {
  result = {};
  if ((camera_path.valid() == false) || (camera_vertex_count < 2u) || (camera_vertex_count > camera_path.vertices.size()) ||
      (camera_path.vertices[camera_vertex_count - 1u].intersection.emitter_index == kInvalidIndex)) {
    return false;
  }

  result.vertices.reserve(camera_vertex_count);
  result.edges.reserve(camera_vertex_count - 1u);
  for (uint32_t index = camera_vertex_count; index > 0u; --index) {
    result.vertices.emplace_back(&camera_path.vertices[index - 1u]);
  }
  for (uint32_t index = camera_vertex_count - 1u; index > 0u; --index) {
    result.edges.emplace_back(UPBPJoinedPathEdge{&camera_path.segments[index - 1u], true});
  }
  return result.valid();
}

inline double upbp_log_measure_conversion(const Scene& scene, const UPBPPathVertexRecord& target, const float3& direction, const double distance_squared) {
  if ((distance_squared <= 0.0) || (std::isfinite(distance_squared) == false)) {
    return -std::numeric_limits<double>::infinity();
  }
  if (target.cls == UPBPVertexClass::Medium) {
    return -std::log(distance_squared);
  }
  if ((target.cls == UPBPVertexClass::Surface) || ((target.cls == UPBPVertexClass::Emitter) && (target.intersection.triangle_index != kInvalidIndex))) {
    const Triangle& triangle = scene.triangles[target.intersection.triangle_index];
    const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, target.intersection.instance_index);
    const double cosine = fabs(static_cast<double>(dot(direction, geometric_normal)));
    return cosine > 0.0 ? std::log(cosine) - std::log(distance_squared) : -std::numeric_limits<double>::infinity();
  }
  return 0.0;
}

inline double upbp_direction_pdf(const Scene& scene, const SpectralQuery spect, const UPBPPathVertexRecord& vertex, const PathSource source, const float3& incoming_direction,
  const float3& outgoing_direction, Sampler& sampler) {
  if (vertex.cls == UPBPVertexClass::Medium) {
    const double pdf = phase_function(incoming_direction, outgoing_direction, vertex.medium.anisotropy);
    return pdf > 0.0 ? pdf : 0.0;
  }
  if (vertex.cls != UPBPVertexClass::Surface) {
    return 0.0;
  }

  Intersection intersection = vertex.intersection;
  intersection.w_i = incoming_direction;
  const BSDFData data = {spect, vertex.incident_medium_index, source, intersection, incoming_direction};
  const double pdf = bsdf::pdf(data, outgoing_direction, scene.materials[intersection.material_index], sampler);
  return pdf > 0.0 ? pdf : 0.0;
}

inline double upbp_emitter_direction_pdf(const Scene& scene, const SpectralQuery spect, const UPBPPathVertexRecord& emitter_vertex, const UPBPPathVertexRecord& next_vertex) {
  if (emitter_vertex.intersection.emitter_index == kInvalidIndex) {
    return 0.0;
  }
  const Emitter& emitter = scene.emitter_instances[emitter_vertex.intersection.emitter_index];
  if (emitter.is_distant()) {
    const float3 outward_direction = normalize(emitter_vertex.position - next_vertex.position);
    const double emitter_selection_pdf = emitter_discrete_pdf(emitter);
    if (emitter_selection_pdf <= 0.0) {
      return 0.0;
    }
    return emitter_sample_pdf(emitter, outward_direction) / emitter_selection_pdf;
  }

  EmitterRadianceQuery query = {
    .source_position = next_vertex.position,
    .target_position = emitter_vertex.position,
    .uv = emitter_vertex.intersection.tex,
  };
  float pdf_area = 0.0f;
  float pdf_dir = 0.0f;
  float pdf_dir_out = 0.0f;
  (void)emitter_get_radiance(emitter, spect, query, pdf_area, pdf_dir, pdf_dir_out);
  return pdf_area > 0.0f ? pdf_dir_out / pdf_area : 0.0;
}

inline bool upbp_build_joined_path_probability(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const UPBPJoinedPath& path, Sampler& sampler,
  UPBPPathProbabilityRecord& result) {
  result = {};
  if (path.valid() == false) {
    result.failure = UPBPPathProbabilityFailure::InvalidJoinedPath;
    return false;
  }

  const UPBPPathVertexRecord& emitter_vertex = *path.vertices.front();
  const UPBPPathVertexRecord& camera_vertex = *path.vertices.back();
  if ((camera_vertex.cls != UPBPVertexClass::Camera) || (emitter_vertex.intersection.emitter_index == kInvalidIndex)) {
    result.failure = UPBPPathProbabilityFailure::InvalidEndpointClasses;
    return false;
  }
  const Emitter& emitter = scene.emitter_instances[emitter_vertex.intersection.emitter_index];
  const double emitter_endpoint_density = (emitter_vertex.endpoint_pdf_sample > 0.0f) && (emitter_vertex.endpoint_pdf_area > 0.0f)
                                            ? static_cast<double>(emitter_vertex.endpoint_pdf_sample) * emitter_vertex.endpoint_pdf_area
                                            : static_cast<double>(emitter_discrete_pdf(emitter)) * emitter_pdf_area_local(emitter);
  if (emitter_endpoint_density <= 0.0) {
    result.failure = UPBPPathProbabilityFailure::InvalidEmitterDensity;
    return false;
  }
  result.log_emitter_endpoint_density = std::log(emitter_endpoint_density);
  if (camera_vertex.endpoint_pdf_area <= 0.0f) {
    result.failure = UPBPPathProbabilityFailure::InvalidCameraDensity;
    return false;
  }
  result.log_camera_endpoint_density = std::log(static_cast<double>(camera_vertex.endpoint_pdf_area));
  result.vertices.resize(path.vertices.size());
  result.edges.resize(path.edges.size());
  for (uint32_t index = 0u; index < path.vertices.size(); ++index) {
    result.vertices[index].connectible = (index == 0u) || (index + 1u == path.vertices.size()) || path.vertices[index]->connectible;
  }

  for (uint32_t edge_index = 0u; edge_index < path.edges.size(); ++edge_index) {
    const UPBPPathVertexRecord& from = *path.vertices[edge_index];
    const UPBPPathVertexRecord& to = *path.vertices[edge_index + 1u];
    float3 direction = to.position - from.position;
    const double distance_squared = static_cast<double>(dot(direction, direction));
    if ((distance_squared <= 0.0) || (std::isfinite(distance_squared) == false)) {
      result.failure = UPBPPathProbabilityFailure::DegenerateEdge;
      return false;
    }
    direction /= sqrtf(static_cast<float>(distance_squared));

    double forward_direction_pdf = 0.0;
    if (edge_index == 0u) {
      forward_direction_pdf = from.delta ? from.endpoint_pdf_direction : upbp_emitter_direction_pdf(scene, spect, from, to);
    } else if (from.delta) {
      if (edge_index < path.selected_light_vertex_count - 1u) {
        forward_direction_pdf = from.scatter_pdf_forward;
      } else if (edge_index >= path.selected_light_vertex_count) {
        forward_direction_pdf = from.scatter_pdf_reverse;
      }
    } else {
      const UPBPPathVertexRecord& previous = *path.vertices[edge_index - 1u];
      const float3 incoming_direction = normalize(from.position - previous.position);
      forward_direction_pdf = upbp_direction_pdf(scene, spect, from, PathSource::Light, incoming_direction, direction, sampler);
    }

    double reverse_direction_pdf = 0.0;
    if (edge_index + 2u == path.vertices.size()) {
      const Ray camera_ray = {camera_vertex.position, -direction, kRayEpsilon, kMaxFloat};
      reverse_direction_pdf = film_evaluate_out(spect, rt.camera(), camera_ray).pdf_dir;
    } else if (to.delta) {
      const uint32_t target_index = edge_index + 1u;
      if (target_index < path.selected_light_vertex_count - 1u) {
        reverse_direction_pdf = to.scatter_pdf_reverse;
      } else if (target_index >= path.selected_light_vertex_count) {
        reverse_direction_pdf = to.scatter_pdf_forward;
      }
    } else {
      const UPBPPathVertexRecord& next = *path.vertices[edge_index + 2u];
      const float3 incoming_direction = normalize(to.position - next.position);
      reverse_direction_pdf = upbp_direction_pdf(scene, spect, to, PathSource::Camera, incoming_direction, -direction, sampler);
    }
    const UPBPJoinedPathEdge& joined_edge = path.edges[edge_index];
    const double segment_log_forward = joined_edge.reversed ? joined_edge.segment->log_transport_pdf_reverse : joined_edge.segment->log_transport_pdf_forward;
    const double segment_log_reverse = joined_edge.reversed ? joined_edge.segment->log_transport_pdf_forward : joined_edge.segment->log_transport_pdf_reverse;
    const double target_event_log_density = to.cls == UPBPVertexClass::Medium ? to.log_medium_event_density : 0.0;
    const double source_event_log_density = from.cls == UPBPVertexClass::Medium ? from.log_medium_event_density : 0.0;
    result.edges[edge_index].log_density_forward = forward_direction_pdf > 0.0 ? std::log(forward_direction_pdf) + segment_log_forward + target_event_log_density +
                                                                                   upbp_log_measure_conversion(scene, to, direction, distance_squared)
                                                                               : -std::numeric_limits<double>::infinity();
    result.edges[edge_index].log_density_reverse = reverse_direction_pdf > 0.0 ? std::log(reverse_direction_pdf) + segment_log_reverse + source_event_log_density +
                                                                                   upbp_log_measure_conversion(scene, from, -direction, distance_squared)
                                                                               : -std::numeric_limits<double>::infinity();
  }
  if (result.valid() == false) {
    result.failure = UPBPPathProbabilityFailure::InvalidRecord;
    return false;
  }
  return true;
}

inline void upbp_apply_bpt_strategy_constraints(const Scene& scene, const uint32_t path_vertex_count, std::vector<UPBPBPTStrategyProbability>& strategies) {
  const uint32_t path_length = path_vertex_count > 0u ? path_vertex_count - 1u : 0u;
  const bool valid_path_length = (path_length >= scene.options.min_path_length) && (path_length <= scene.options.max_path_length);
  for (UPBPBPTStrategyProbability& strategy : strategies) {
    const uint32_t camera_vertex_count = path_vertex_count - strategy.light_vertex_count;
    const bool enabled = strategy.light_vertex_count == 0u   ? scene.strategy_enabled(Scene::Strategy::DirectHit)
                         : strategy.light_vertex_count == 1u ? scene.strategy_enabled(Scene::Strategy::ConnectToLight)
                         : camera_vertex_count == 1u         ? scene.strategy_enabled(Scene::Strategy::ConnectToCamera)
                                                             : scene.strategy_enabled(Scene::Strategy::ConnectVertices);
    strategy.applicable = strategy.applicable && enabled && valid_path_length;
  }
}

struct UPBPBPTConnectionContribution {
  UPBPVertexConnectionResult connection = {};
  UPBPJoinedPath joined_path = {};
  UPBPPathProbabilityRecord probability_path = {};
  std::vector<UPBPBPTStrategyProbability> strategies = {};
  SpectralResponse contribution = {};
  double mis_weight = 0.0;
  bool applicable = false;
};

inline bool upbp_evaluate_bpt_connection(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const UPBPPathRecord& light_path, const uint32_t light_vertex_count,
  const UPBPPathRecord& camera_path, const uint32_t camera_vertex_count, Sampler& scattering_sampler, Sampler& intersection_sampler, Sampler& medium_sampler,
  const uint32_t maximum_boundary_count, const uint32_t maximum_null_events_per_interval, const bool evaluate_exhaustive_weight, UPBPBPTConnectionContribution& result) {
  result = {};
  result.contribution = SpectralResponse{spect, 0.0f};
  if ((light_vertex_count < 2u) || (camera_vertex_count < 2u) || (light_vertex_count > light_path.vertices.size()) || (camera_vertex_count > camera_path.vertices.size())) {
    return false;
  }

  const UPBPPathVertexRecord& light_vertex = light_path.vertices[light_vertex_count - 1u];
  const UPBPPathVertexRecord& camera_vertex = camera_path.vertices[camera_vertex_count - 1u];
  if (upbp_evaluate_vertex_connection(rt, scene, spect, light_vertex, camera_vertex, scattering_sampler, intersection_sampler, medium_sampler, maximum_boundary_count,
        maximum_null_events_per_interval, result.connection) == false) {
    return false;
  }
  if (result.connection.applicable == false) {
    return true;
  }
  if (evaluate_exhaustive_weight == false) {
    result.mis_weight = 1.0;
    result.applicable = true;
    result.contribution = result.connection.contribution;
    return true;
  }

  if (upbp_join_subpaths(light_path, light_vertex_count, camera_path, camera_vertex_count, result.connection.transmittance.segment, false, result.joined_path) == false) {
    return false;
  }
  if (upbp_build_joined_path_probability(rt, scene, spect, result.joined_path, scattering_sampler, result.probability_path) == false) {
    return false;
  }
  if (upbp_enumerate_bpt_strategies_recursive(result.probability_path, result.strategies) == false) {
    return false;
  }
  upbp_apply_bpt_strategy_constraints(scene, static_cast<uint32_t>(result.joined_path.vertices.size()), result.strategies);

  result.mis_weight = upbp_bpt_balance_weight(result.strategies, light_vertex_count);
  result.applicable = result.mis_weight > 0.0;
  result.contribution = result.connection.contribution * static_cast<float>(result.mis_weight);
  return true;
}

struct UPBPNEEContribution {
  EmitterSample emitter_sample = {};
  UPBPScatteringEval camera_scattering = {};
  UPBPConnectionTransmittanceResult transmittance = {};
  UPBPPathRecord endpoint_path = {};
  UPBPJoinedPath joined_path = {};
  UPBPPathProbabilityRecord probability_path = {};
  std::vector<UPBPBPTStrategyProbability> strategies = {};
  SpectralResponse contribution = {};
  double mis_weight = 0.0;
  bool applicable = false;
};

inline bool upbp_evaluate_nee(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const UPBPPathRecord& camera_path, const uint32_t camera_vertex_count,
  Sampler& emitter_sampler, Sampler& scattering_sampler, Sampler& intersection_sampler, Sampler& medium_sampler, const uint32_t maximum_boundary_count,
  const uint32_t maximum_null_events_per_interval, const bool evaluate_exhaustive_weight, UPBPNEEContribution& result) {
  result = {};
  result.contribution = SpectralResponse{spect, 0.0f};
  if ((camera_vertex_count < 2u) || (camera_vertex_count > camera_path.vertices.size())) {
    return false;
  }

  const UPBPPathVertexRecord& camera_vertex = camera_path.vertices[camera_vertex_count - 1u];
  if (camera_vertex.connectible == false) {
    return true;
  }
  const bool source_is_surface = camera_vertex.cls == UPBPVertexClass::Surface;
  const EmitterSampleQuery query = {
    .spect = spect,
    .source_type = source_is_surface ? InteractionType::Surface : InteractionType::Medium,
    .source_position = camera_vertex.position,
    .source_normal = source_is_surface ? camera_vertex.intersection.nrm : float3{},
  };
  result.emitter_sample = sample_emitter(scene.light_sampling_method(), query, emitter_sampler);
  if (result.emitter_sample.value.is_zero() || (result.emitter_sample.pdf_dir <= 0.0f) || (result.emitter_sample.pdf_sample <= 0.0f)) {
    return true;
  }

  result.camera_scattering = upbp_evaluate_vertex_scattering(scene, spect, camera_vertex, result.emitter_sample.direction, scattering_sampler);
  if (result.camera_scattering.valid() == false) {
    return true;
  }
  if (upbp_sample_connection_transmittance(rt, scene, spect, camera_vertex, result.emitter_sample.origin, intersection_sampler, medium_sampler, maximum_boundary_count,
        maximum_null_events_per_interval, result.transmittance) == false) {
    return false;
  }
  if (result.transmittance.visible == false) {
    return true;
  }
  if (evaluate_exhaustive_weight == false) {
    result.mis_weight = 1.0;
    result.applicable = true;
    result.contribution = camera_vertex.throughput * result.camera_scattering.value * result.emitter_sample.value * result.transmittance.weight /
                          (static_cast<double>(result.emitter_sample.pdf_dir) * result.emitter_sample.pdf_sample);
    return true;
  }

  result.endpoint_path.reset(1u);
  const UPBPPathVertexRecord endpoint = upbp_make_emitter_endpoint(scene, spect, result.emitter_sample);
  if (result.endpoint_path.append_endpoint(endpoint) == false) {
    return false;
  }
  if (upbp_join_subpaths(result.endpoint_path, 1u, camera_path, camera_vertex_count, result.transmittance.segment, true, result.joined_path) == false) {
    return false;
  }
  if (upbp_build_joined_path_probability(rt, scene, spect, result.joined_path, scattering_sampler, result.probability_path) == false) {
    return false;
  }
  if (upbp_enumerate_bpt_strategies_recursive(result.probability_path, result.strategies) == false) {
    return false;
  }
  upbp_apply_bpt_strategy_constraints(scene, static_cast<uint32_t>(result.joined_path.vertices.size()), result.strategies);

  result.mis_weight = upbp_bpt_balance_weight(result.strategies, 1u);
  result.applicable = result.mis_weight > 0.0;
  result.contribution = camera_vertex.throughput * result.camera_scattering.value * result.emitter_sample.value * result.transmittance.weight *
                        static_cast<float>(result.mis_weight / (static_cast<double>(result.emitter_sample.pdf_dir) * result.emitter_sample.pdf_sample));
  return true;
}

struct UPBPDirectHitContribution {
  UPBPJoinedPath joined_path = {};
  UPBPPathProbabilityRecord probability_path = {};
  std::vector<UPBPBPTStrategyProbability> strategies = {};
  SpectralResponse contribution = {};
  double mis_weight = 0.0;
  bool applicable = false;
};

struct UPBPDirectEnvironmentContribution {
  SpectralResponse contribution = {};
  uint32_t contributing_emitters = 0u;
  bool applicable = false;
};

inline bool upbp_evaluate_direct_environment_hit(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const UPBPSubpathBuildResult& camera_subpath,
  Sampler& sampler, UPBPDirectEnvironmentContribution& result) {
  result = {};
  result.contribution = SpectralResponse{spect, 0.0f};
  if ((camera_subpath.valid() == false) || (camera_subpath.terminal != UPBPSceneSegmentTerminal::Miss) || (camera_subpath.path.has_terminal_segment == false) ||
      camera_subpath.path.terminal_segment.intervals.empty()) {
    return false;
  }

  const uint32_t environment_emitter_count = environment_emitter_shared_count();
  for (uint32_t local_emitter_index = 0u; local_emitter_index < environment_emitter_count; ++local_emitter_index) {
    uint32_t emitter_index = kInvalidIndex;
    if (environment_emitter_shared_try_load_index(local_emitter_index, emitter_index) == false) {
      continue;
    }

    const Emitter& emitter = scene.emitter_instances[emitter_index];
    const EmitterRadianceQuery query = {
      .direction = camera_subpath.terminal_ray.d,
      .directly_visible = camera_subpath.path.physical_length() == 0u,
    };
    float pdf_area = 0.0f;
    float pdf_dir = 0.0f;
    float pdf_dir_out = 0.0f;
    const SpectralResponse radiance = emitter_get_radiance(emitter, spect, query, pdf_area, pdf_dir, pdf_dir_out);
    if (radiance.is_zero() || (pdf_area <= 0.0f) || (pdf_dir <= 0.0f)) {
      continue;
    }

    UPBPPathVertexRecord endpoint = {};
    endpoint.position = camera_subpath.path.terminal_segment.intervals.back().end_position;
    endpoint.intersection.pos = endpoint.position;
    endpoint.intersection.w_i = camera_subpath.terminal_ray.d;
    endpoint.intersection.emitter_index = emitter_index;
    endpoint.throughput = camera_subpath.throughput;
    endpoint.endpoint_pdf_area = pdf_area;
    endpoint.endpoint_pdf_sample = emitter_discrete_pdf(emitter);
    endpoint.endpoint_pdf_direction = pdf_dir;
    endpoint.incident_medium_index = camera_subpath.active_medium_index;
    endpoint.outgoing_medium_index = camera_subpath.active_medium_index;
    endpoint.cls = UPBPVertexClass::Emitter;
    endpoint.source = PathSource::Camera;
    endpoint.connectible = true;
    endpoint.delta = emitter.is_delta();
    endpoint.distant_endpoint = true;

    UPBPPathRecord endpoint_path = camera_subpath.path;
    if (endpoint_path.promote_terminal_segment(endpoint) == false) {
      return false;
    }

    UPBPJoinedPath joined_path = {};
    if (upbp_join_direct_hit(endpoint_path, static_cast<uint32_t>(endpoint_path.vertices.size()), joined_path) == false) {
      return false;
    }
    UPBPPathProbabilityRecord probability_path = {};
    if (upbp_build_joined_path_probability(rt, scene, spect, joined_path, sampler, probability_path) == false) {
      return false;
    }
    std::vector<UPBPBPTStrategyProbability> strategies = {};
    if (upbp_enumerate_bpt_strategies_recursive(probability_path, strategies) == false) {
      return false;
    }
    upbp_apply_bpt_strategy_constraints(scene, static_cast<uint32_t>(joined_path.vertices.size()), strategies);

    const double mis_weight = upbp_bpt_balance_weight(strategies, 0u);
    if (mis_weight <= 0.0) {
      continue;
    }
    result.contribution += endpoint.throughput * radiance * static_cast<float>(mis_weight);
    ++result.contributing_emitters;
  }

  result.applicable = result.contributing_emitters > 0u;
  return true;
}

enum class UPBPLightToCameraFailure : uint8_t {
  None,
  InvalidInput,
  ConnectionTransmittance,
  CameraEndpoint,
  JoinedPath,
  PathProbability,
  StrategyEnumeration,
};

struct UPBPLightToCameraContribution {
  CameraSample camera_sample = {};
  float2 splat_uv = {};
  UPBPScatteringEval light_scattering = {};
  UPBPConnectionTransmittanceResult transmittance = {};
  UPBPPathRecord camera_endpoint_path = {};
  UPBPJoinedPath joined_path = {};
  UPBPPathProbabilityRecord probability_path = {};
  std::vector<UPBPBPTStrategyProbability> strategies = {};
  SpectralResponse contribution = {};
  double mis_weight = 0.0;
  UPBPLightToCameraFailure failure = UPBPLightToCameraFailure::None;
  bool applicable = false;
};

inline bool upbp_evaluate_light_to_camera(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const UPBPPathRecord& light_path, const uint32_t light_vertex_count,
  Sampler& camera_sampler, Sampler& scattering_sampler, Sampler& intersection_sampler, Sampler& medium_sampler, const uint32_t maximum_boundary_count,
  const uint32_t maximum_null_events_per_interval, const bool evaluate_exhaustive_weight, UPBPLightToCameraContribution& result) {
  result = {};
  result.contribution = SpectralResponse{spect, 0.0f};
  if ((light_vertex_count < 2u) || (light_vertex_count > light_path.vertices.size())) {
    result.failure = UPBPLightToCameraFailure::InvalidInput;
    return false;
  }

  const UPBPPathVertexRecord& light_vertex = light_path.vertices[light_vertex_count - 1u];
  if (light_vertex.connectible == false) {
    return true;
  }
  result.camera_sample = sample_film(camera_sampler, rt.camera(), light_vertex.position);
  if (result.camera_sample.valid() == false) {
    return true;
  }
  result.splat_uv = pixel_filter_splat_uv(result.camera_sample.uv, rt.camera().film_size, sample_pixel_filter_offset(scene.pixel_sampler, camera_sampler.next_2d()));
  if (pixel_filter_contains_uv(result.splat_uv) == false) {
    return true;
  }

  result.light_scattering = upbp_evaluate_vertex_scattering(scene, spect, light_vertex, result.camera_sample.direction, scattering_sampler);
  if (result.light_scattering.valid() == false) {
    return true;
  }
  if (upbp_sample_connection_transmittance(rt, scene, spect, light_vertex, result.camera_sample.position, intersection_sampler, medium_sampler, maximum_boundary_count,
        maximum_null_events_per_interval, result.transmittance) == false) {
    result.failure = UPBPLightToCameraFailure::ConnectionTransmittance;
    return false;
  }
  if (result.transmittance.visible == false) {
    return true;
  }
  if (evaluate_exhaustive_weight == false) {
    result.mis_weight = 1.0;
    result.applicable = true;
    result.contribution = light_vertex.throughput * result.light_scattering.value * result.transmittance.weight * result.camera_sample.weight;
    return true;
  }

  UPBPPathVertexRecord camera_endpoint = {};
  camera_endpoint.position = result.camera_sample.position;
  camera_endpoint.intersection.pos = result.camera_sample.position;
  camera_endpoint.intersection.nrm = result.camera_sample.normal;
  camera_endpoint.intersection.w_i = -result.camera_sample.direction;
  camera_endpoint.throughput = SpectralResponse{spect, 1.0f};
  camera_endpoint.endpoint_pdf_area = result.camera_sample.pdf_area;
  camera_endpoint.endpoint_pdf_sample = 1.0f;
  camera_endpoint.endpoint_pdf_direction = result.camera_sample.pdf_dir_out;
  camera_endpoint.incident_medium_index = rt.camera().medium_index;
  camera_endpoint.outgoing_medium_index = rt.camera().medium_index;
  camera_endpoint.cls = UPBPVertexClass::Camera;
  camera_endpoint.source = PathSource::Camera;
  camera_endpoint.connectible = true;

  result.camera_endpoint_path.reset(1u);
  if (result.camera_endpoint_path.append_endpoint(camera_endpoint) == false) {
    result.failure = UPBPLightToCameraFailure::CameraEndpoint;
    return false;
  }
  if (upbp_join_subpaths(light_path, light_vertex_count, result.camera_endpoint_path, 1u, result.transmittance.segment, false, result.joined_path) == false) {
    result.failure = UPBPLightToCameraFailure::JoinedPath;
    return false;
  }
  if (upbp_build_joined_path_probability(rt, scene, spect, result.joined_path, scattering_sampler, result.probability_path) == false) {
    result.failure = UPBPLightToCameraFailure::PathProbability;
    return false;
  }
  if (upbp_enumerate_bpt_strategies_recursive(result.probability_path, result.strategies) == false) {
    result.failure = UPBPLightToCameraFailure::StrategyEnumeration;
    return false;
  }
  upbp_apply_bpt_strategy_constraints(scene, static_cast<uint32_t>(result.joined_path.vertices.size()), result.strategies);

  result.mis_weight = upbp_bpt_balance_weight(result.strategies, light_vertex_count);
  result.applicable = result.mis_weight > 0.0;
  result.contribution =
    light_vertex.throughput * result.light_scattering.value * result.transmittance.weight * (result.camera_sample.weight * static_cast<float>(result.mis_weight));
  return true;
}

inline bool upbp_evaluate_direct_area_hit(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const UPBPPathRecord& camera_path,
  const uint32_t camera_vertex_count, Sampler& sampler, const bool evaluate_exhaustive_weight, UPBPDirectHitContribution& result) {
  result = {};
  result.contribution = SpectralResponse{spect, 0.0f};
  if ((camera_vertex_count < 2u) || (camera_vertex_count > camera_path.vertices.size())) {
    return false;
  }

  const UPBPPathVertexRecord& emitter_vertex = camera_path.vertices[camera_vertex_count - 1u];
  if (emitter_vertex.intersection.emitter_index == kInvalidIndex) {
    return true;
  }
  const Emitter& emitter = scene.emitter_instances[emitter_vertex.intersection.emitter_index];
  if (emitter.is_local() == false) {
    return true;
  }

  const UPBPPathVertexRecord& previous = camera_path.vertices[camera_vertex_count - 2u];
  const EmitterRadianceQuery query = {
    .source_position = previous.position,
    .target_position = emitter_vertex.position,
    .uv = emitter_vertex.intersection.tex,
    .directly_visible = camera_vertex_count <= 2u,
  };
  float pdf_area = 0.0f;
  float pdf_dir = 0.0f;
  float pdf_dir_out = 0.0f;
  const SpectralResponse radiance = emitter_get_radiance(emitter, spect, query, pdf_area, pdf_dir, pdf_dir_out);
  if ((pdf_dir <= 0.0f) || radiance.is_zero()) {
    return true;
  }
  if (evaluate_exhaustive_weight == false) {
    result.mis_weight = 1.0;
    result.applicable = true;
    result.contribution = emitter_vertex.throughput * radiance;
    return true;
  }

  if (upbp_join_direct_hit(camera_path, camera_vertex_count, result.joined_path) == false) {
    return false;
  }
  if (upbp_build_joined_path_probability(rt, scene, spect, result.joined_path, sampler, result.probability_path) == false) {
    return false;
  }
  if (upbp_enumerate_bpt_strategies_recursive(result.probability_path, result.strategies) == false) {
    return false;
  }
  upbp_apply_bpt_strategy_constraints(scene, static_cast<uint32_t>(result.joined_path.vertices.size()), result.strategies);

  result.mis_weight = upbp_bpt_balance_weight(result.strategies, 0u);
  result.applicable = result.mis_weight > 0.0;
  result.contribution = emitter_vertex.throughput * radiance * static_cast<float>(result.mis_weight);
  return true;
}

}  // namespace etx
