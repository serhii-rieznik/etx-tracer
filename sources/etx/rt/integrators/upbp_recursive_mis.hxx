#pragma once

#include <etx/rt/integrators/upbp_density_mis.hxx>

namespace etx {

struct UPBPRecursiveVertexWeights {
  double d_shared = 0.0;
  double d_bpt_base = 0.0;
  double d_pde_base = 0.0;
  double d_surface = 0.0;
  double ray_sample_forward_pdf_inverse = 0.0;
  double ray_sample_reverse_pdf_inverse = 0.0;
  double ray_sample_forward_ratio = 0.0;
  double ray_sample_reverse_ratio = 0.0;
  bool previous_in_medium = false;
  bool previous_delta = false;

  double bpt(const double surface_factor) const {
    return d_bpt_base + surface_factor * d_surface;
  }

  double pde(const double surface_factor) const {
    return d_pde_base + surface_factor * d_surface;
  }
};

enum class UPBPRecursiveWeightFailure : uint8_t {
  None,
  InvalidPath,
  InvalidEndpointDensity,
  InvalidSegmentDensity,
  InvalidMeasureCosine,
  InvalidScatteringDensity,
  NonFiniteArrival,
  NonFiniteDeparture,
};

struct UPBPRecursiveState {
  UPBPRecursiveVertexWeights weights = {};
  double last_sin_theta = 0.0;
  double d_bpt_a = 0.0;
  double d_bpt_b = 0.0;
  double d_surface_b = 0.0;
  double d_pde_b = 0.0;
  UPBPRecursiveWeightFailure failure = UPBPRecursiveWeightFailure::None;
  uint32_t failure_vertex_index = 0u;
};

struct UPBPRecursivePathWeights {
  std::vector<UPBPRecursiveVertexWeights> arrivals = {};
  std::vector<UPBPRecursiveState> departures = {};
  std::vector<bool> has_departure = {};
  UPBPRecursiveWeightFailure failure = UPBPRecursiveWeightFailure::None;
  uint32_t failure_vertex_index = 0u;
};

inline double upbp_segment_sampling_log_density(const UPBPTransportSegmentRecord& segment, const UPBPPathVertexRecord& source, const UPBPPathVertexRecord& target,
  const bool reverse) {
  const double transport_density = reverse ? segment.log_transport_pdf_reverse : segment.log_transport_pdf_forward;
  const UPBPPathVertexRecord& terminal = reverse ? source : target;
  return transport_density + (terminal.cls == UPBPVertexClass::Medium ? terminal.log_medium_event_density : 0.0);
}

inline double upbp_short_beam_ray_factor(const UPBPPathVertexRecord& terminal) {
  if (terminal.cls != UPBPVertexClass::Medium) {
    return 0.0;
  }
  const double density = std::exp(terminal.log_medium_event_density);
  return density > 0.0 ? 1.0 / density : 0.0;
}

inline double upbp_vertex_cosine(const Scene& scene, const UPBPPathVertexRecord& vertex, const float3& direction) {
  if (vertex.cls == UPBPVertexClass::Medium) {
    return 1.0;
  }
  if (vertex.cls != UPBPVertexClass::Surface) {
    return 1.0;
  }
  const Triangle& triangle = scene.triangles[vertex.intersection.triangle_index];
  const float3 geometric_normal = scene_triangle_world_geometric_normal(scene, triangle, vertex.intersection.instance_index);
  return fabs(static_cast<double>(dot(geometric_normal, direction)));
}

inline double upbp_recursive_local_volume_factor(const UPBPDensityMISConfiguration& configuration, const UPBPVertexClass vertex_class, const bool vertex_delta,
  const bool vertex_density_connectible, const UPBPRecursiveVertexWeights& weights, const double next_reverse_pdf_inverse, const double next_reverse_ratio, const double sin_theta,
  const PathSource source) {
  if ((vertex_class != UPBPVertexClass::Medium) || vertex_delta || (vertex_density_connectible == false)) {
    return 0.0;
  }
  const double forward_inverse = source == PathSource::Light ? weights.ray_sample_forward_pdf_inverse : next_reverse_pdf_inverse;
  const double forward_ratio = source == PathSource::Light ? weights.ray_sample_forward_ratio : next_reverse_ratio;
  const double reverse_inverse = source == PathSource::Light ? next_reverse_pdf_inverse : weights.ray_sample_forward_pdf_inverse;
  const double reverse_ratio = source == PathSource::Light ? next_reverse_ratio : weights.ray_sample_forward_ratio;
  const UPBPDensityMISContext context = {
    2u,
    vertex_class,
    configuration.photon_beams_long ? forward_inverse : forward_ratio,
    configuration.camera_beams_long ? reverse_inverse : reverse_ratio,
    sin_theta,
    vertex_delta,
    vertex_density_connectible,
  };

  double result = 0.0;
  constexpr UPBPTechnique techniques[] = {
    UPBPTechnique::PP3D,
    UPBPTechnique::PB2D,
    UPBPTechnique::BP2D,
    UPBPTechnique::BB1D,
  };
  for (const UPBPTechnique technique : techniques) {
    result += upbp_density_strategy_factor(configuration, context, technique);
  }
  return result;
}

inline double upbp_recursive_surface_coefficient(const UPBPDensityMISConfiguration& configuration, const UPBPVertexClass vertex_class, const bool vertex_delta,
  const bool vertex_density_connectible) {
  return ((vertex_class == UPBPVertexClass::Surface) && (vertex_delta == false) && vertex_density_connectible && configuration.enabled(UPBPTechnique::Surface)) ? 1.0 : 0.0;
}

inline double upbp_recursive_local_pde_factor(const UPBPDensityMISConfiguration& configuration, const UPBPVertexClass vertex_class, const bool vertex_delta,
  const bool vertex_density_connectible, const UPBPRecursiveVertexWeights& weights, const double next_reverse_pdf_inverse, const double next_reverse_ratio, const double sin_theta,
  const PathSource source) {
  return upbp_recursive_local_volume_factor(configuration, vertex_class, vertex_delta, vertex_density_connectible, weights, next_reverse_pdf_inverse, next_reverse_ratio, sin_theta,
           source) +
         configuration.factor(UPBPTechnique::Surface) * upbp_recursive_surface_coefficient(configuration, vertex_class, vertex_delta, vertex_density_connectible);
}

inline double upbp_recursive_local_pde_factor(const UPBPDensityMISConfiguration& configuration, const UPBPPathVertexRecord& vertex, const UPBPRecursiveVertexWeights& weights,
  const double next_reverse_pdf_inverse, const double next_reverse_ratio, const double sin_theta, const PathSource source) {
  return upbp_recursive_local_pde_factor(configuration, vertex.cls, vertex.delta, vertex.density_connectible, weights, next_reverse_pdf_inverse, next_reverse_ratio, sin_theta,
    source);
}

struct UPBPRecursiveLocalPDEAffine {
  double surface_coefficient = 0.0;
  double reverse_pdf_inverse_coefficient = 0.0;
  double constant = 0.0;
};

inline UPBPRecursiveLocalPDEAffine upbp_recursive_local_pde_affine(const UPBPDensityMISConfiguration& configuration, const UPBPVertexClass vertex_class, const bool vertex_delta,
  const bool vertex_density_connectible, const UPBPRecursiveVertexWeights& weights, const double next_reverse_ratio, const double sin_theta, const PathSource source) {
  UPBPRecursiveLocalPDEAffine result = {};
  result.constant = upbp_recursive_local_volume_factor(configuration, vertex_class, vertex_delta, vertex_density_connectible, weights, 0.0, next_reverse_ratio, sin_theta, source);
  result.surface_coefficient = upbp_recursive_surface_coefficient(configuration, vertex_class, vertex_delta, vertex_density_connectible);

  UPBPDensityMISContext variable_context = {
    2u,
    vertex_class,
    0.0,
    0.0,
    sin_theta,
    vertex_delta,
    vertex_density_connectible,
  };
  if ((source == PathSource::Light) && configuration.camera_beams_long) {
    variable_context.forward_ray_factor = configuration.photon_beams_long ? weights.ray_sample_forward_pdf_inverse : weights.ray_sample_forward_ratio;
    variable_context.reverse_ray_factor = 1.0;
    result.reverse_pdf_inverse_coefficient =
      upbp_density_strategy_factor(configuration, variable_context, UPBPTechnique::PB2D) + upbp_density_strategy_factor(configuration, variable_context, UPBPTechnique::BB1D);
  } else if ((source == PathSource::Camera) && configuration.photon_beams_long) {
    variable_context.forward_ray_factor = 1.0;
    variable_context.reverse_ray_factor = configuration.camera_beams_long ? weights.ray_sample_forward_pdf_inverse : weights.ray_sample_forward_ratio;
    result.reverse_pdf_inverse_coefficient =
      upbp_density_strategy_factor(configuration, variable_context, UPBPTechnique::BP2D) + upbp_density_strategy_factor(configuration, variable_context, UPBPTechnique::BB1D);
  }
  return result;
}

inline bool upbp_initialize_recursive_state(const UPBPPathRecord& path, const uint64_t light_subpath_count, const uint64_t bpt_sample_count, UPBPRecursiveState& state) {
  state = {};
  if ((path.valid() == false) || path.vertices.empty() || (light_subpath_count == 0u)) {
    state.failure = UPBPRecursiveWeightFailure::InvalidPath;
    return false;
  }

  const UPBPPathVertexRecord& endpoint = path.vertices.front();
  if (endpoint.source == PathSource::Camera) {
    if (endpoint.endpoint_pdf_direction <= 0.0f) {
      state.failure = UPBPRecursiveWeightFailure::InvalidEndpointDensity;
      return false;
    }
    state.weights.d_shared = 1.0 / endpoint.endpoint_pdf_direction;
    return true;
  }

  if ((endpoint.source != PathSource::Light) || (endpoint.endpoint_pdf_area <= 0.0f) || (endpoint.endpoint_pdf_sample <= 0.0f) || (endpoint.endpoint_pdf_direction <= 0.0f)) {
    state.failure = UPBPRecursiveWeightFailure::InvalidEndpointDensity;
    return false;
  }
  const double emission_density = static_cast<double>(endpoint.endpoint_pdf_area) * endpoint.endpoint_pdf_sample * endpoint.endpoint_pdf_direction;
  state.weights.d_shared = endpoint.distant_endpoint ? 1.0 / endpoint.endpoint_pdf_area : 1.0 / endpoint.endpoint_pdf_direction;
  if (endpoint.delta == false) {
    const double cosine = endpoint.distant_endpoint ? 1.0 : fabs(static_cast<double>(dot(endpoint.intersection.nrm, endpoint.sampled_direction)));
    state.weights.d_bpt_base = cosine / emission_density;
  }
  state.weights.d_pde_base = state.weights.d_bpt_base * static_cast<double>(bpt_sample_count);
  const bool valid =
    std::isfinite(state.weights.d_shared) && std::isfinite(state.weights.d_bpt_base) && (std::isfinite(state.weights.d_pde_base) && std::isfinite(state.weights.d_surface));
  state.failure = valid ? UPBPRecursiveWeightFailure::None : UPBPRecursiveWeightFailure::NonFiniteArrival;
  return valid;
}

inline bool upbp_complete_recursive_arrival(const Scene& scene, const UPBPPathRecord& path, const uint32_t vertex_index, const UPBPDensityMISConfiguration& configuration,
  UPBPRecursiveState& state) {
  if ((vertex_index == 0u) || (vertex_index >= path.vertices.size())) {
    state.failure = UPBPRecursiveWeightFailure::InvalidPath;
    state.failure_vertex_index = vertex_index;
    return false;
  }
  const UPBPPathVertexRecord& source = path.vertices[vertex_index - 1u];
  const UPBPPathVertexRecord& target = path.vertices[vertex_index];
  const UPBPTransportSegmentRecord& segment = path.segments[vertex_index - 1u];
  const double log_forward_pdf = upbp_segment_sampling_log_density(segment, source, target, false);
  const double log_reverse_pdf = upbp_segment_sampling_log_density(segment, source, target, true);
  const double forward_pdf = std::exp(log_forward_pdf);
  const double reverse_pdf = std::exp(log_reverse_pdf);
  if ((forward_pdf <= 0.0) || (reverse_pdf <= 0.0) || (std::isfinite(forward_pdf) == false) || (std::isfinite(reverse_pdf) == false)) {
    state.failure = UPBPRecursiveWeightFailure::InvalidSegmentDensity;
    state.failure_vertex_index = vertex_index;
    return false;
  }

  if (vertex_index > 1u) {
    const double next_reverse_inverse = 1.0 / reverse_pdf;
    const double next_reverse_ratio = upbp_short_beam_ray_factor(source);
    const double local_factor = upbp_recursive_local_volume_factor(configuration, source.cls, source.delta, source.density_connectible, state.weights, next_reverse_inverse,
      next_reverse_ratio, state.last_sin_theta, source.source);
    state.weights.d_bpt_base = state.d_bpt_a * local_factor + state.d_bpt_b;
    state.weights.d_pde_base = state.d_bpt_a * local_factor + state.d_pde_b;
    state.weights.d_surface = state.d_bpt_a * upbp_recursive_surface_coefficient(configuration, source.cls, source.delta, source.density_connectible) + state.d_surface_b;
  }

  state.weights.d_shared /= forward_pdf;
  state.weights.d_bpt_base /= forward_pdf;
  state.weights.d_pde_base /= forward_pdf;
  state.weights.d_surface /= forward_pdf;

  const float3 edge_direction = normalize(target.position - source.position);
  const double cosine = upbp_vertex_cosine(scene, target, edge_direction);
  if (upbp_projected_measure_valid(cosine) == false) {
    state.failure = UPBPRecursiveWeightFailure::InvalidMeasureCosine;
    state.failure_vertex_index = vertex_index;
    return false;
  }
  const double distance_squared = static_cast<double>(segment.distance) * segment.distance;
  if ((vertex_index > 1u) || (path.vertices.front().distant_endpoint == false)) {
    state.weights.d_shared *= distance_squared;
  }
  state.weights.d_shared /= cosine;
  state.weights.d_bpt_base /= cosine;
  state.weights.d_pde_base /= cosine;
  state.weights.d_surface /= cosine;
  state.weights.ray_sample_forward_pdf_inverse = 1.0 / forward_pdf;
  state.weights.ray_sample_reverse_pdf_inverse = 1.0 / reverse_pdf;
  state.weights.ray_sample_forward_ratio = upbp_short_beam_ray_factor(target);
  state.weights.ray_sample_reverse_ratio = upbp_short_beam_ray_factor(source);
  const bool valid =
    std::isfinite(state.weights.d_shared) && std::isfinite(state.weights.d_bpt_base) && (std::isfinite(state.weights.d_pde_base) && std::isfinite(state.weights.d_surface));
  state.failure = valid ? UPBPRecursiveWeightFailure::None : UPBPRecursiveWeightFailure::NonFiniteArrival;
  state.failure_vertex_index = valid ? 0u : vertex_index;
  return valid;
}

inline bool upbp_prepare_recursive_departure(const Scene& scene, const UPBPPathRecord& path, const uint32_t vertex_index, const uint64_t bpt_sample_count,
  UPBPRecursiveState& state) {
  if ((vertex_index == 0u) || (vertex_index >= path.vertices.size())) {
    state.failure = UPBPRecursiveWeightFailure::InvalidPath;
    state.failure_vertex_index = vertex_index;
    return false;
  }
  const UPBPPathVertexRecord& vertex = path.vertices[vertex_index];
  const double forward_pdf = vertex.scatter_pdf_forward;
  const double reverse_pdf = vertex.scatter_pdf_reverse;
  if ((forward_pdf <= 0.0) || (std::isfinite(forward_pdf) == false) || (reverse_pdf < 0.0) || (std::isfinite(reverse_pdf) == false)) {
    state.failure = UPBPRecursiveWeightFailure::InvalidScatteringDensity;
    state.failure_vertex_index = vertex_index;
    return false;
  }

  const double cosine = upbp_vertex_cosine(scene, vertex, vertex.sampled_direction);
  if (upbp_projected_measure_valid(cosine) == false) {
    state.failure = UPBPRecursiveWeightFailure::InvalidMeasureCosine;
    state.failure_vertex_index = vertex_index;
    return false;
  }
  const bool bpt_previous = (state.weights.previous_delta == false) && (vertex.delta == false);
  state.d_bpt_a = cosine / forward_pdf;
  if (vertex.delta) {
    state.d_bpt_b = cosine * state.weights.d_bpt_base / state.weights.ray_sample_reverse_pdf_inverse;
    state.d_pde_b = cosine * state.weights.d_pde_base / state.weights.ray_sample_reverse_pdf_inverse;
    state.d_surface_b = cosine * state.weights.d_surface / state.weights.ray_sample_reverse_pdf_inverse;
  } else {
    state.d_bpt_b =
      (cosine / forward_pdf) * (state.weights.d_shared * static_cast<double>(bpt_previous) + reverse_pdf * state.weights.d_bpt_base / state.weights.ray_sample_reverse_pdf_inverse);
    state.d_pde_b = (cosine / forward_pdf) * (state.weights.d_shared * static_cast<double>(bpt_previous) * static_cast<double>(bpt_sample_count) +
                                               reverse_pdf * state.weights.d_pde_base / state.weights.ray_sample_reverse_pdf_inverse);
    state.d_surface_b = (cosine / forward_pdf) * reverse_pdf * state.weights.d_surface / state.weights.ray_sample_reverse_pdf_inverse;
  }
  state.weights.d_shared = 1.0 / forward_pdf;
  state.weights.previous_in_medium = vertex.cls == UPBPVertexClass::Medium;
  state.weights.previous_delta = vertex.delta;

  const float3 incoming_direction = vertex.intersection.w_i;
  const double cosine_directions = static_cast<double>(dot(incoming_direction, vertex.sampled_direction));
  state.last_sin_theta = std::sqrt(fmax(0.0, 1.0 - cosine_directions * cosine_directions));
  const bool valid = std::isfinite(state.d_bpt_a) && std::isfinite(state.d_bpt_b) && std::isfinite(state.d_surface_b) && std::isfinite(state.d_pde_b);
  state.failure = valid ? UPBPRecursiveWeightFailure::None : UPBPRecursiveWeightFailure::NonFiniteDeparture;
  state.failure_vertex_index = valid ? 0u : vertex_index;
  return valid;
}

inline bool upbp_compute_recursive_path_weights(const Scene& scene, const UPBPPathRecord& path, const UPBPDensityMISConfiguration& configuration,
  const uint64_t light_subpath_count, const uint64_t bpt_sample_count, UPBPRecursivePathWeights& result) {
  result.arrivals.clear();
  result.departures.clear();
  result.has_departure.clear();
  result.failure = UPBPRecursiveWeightFailure::None;
  result.failure_vertex_index = 0u;
  UPBPRecursiveState state = {};
  if (upbp_initialize_recursive_state(path, light_subpath_count, bpt_sample_count, state) == false) {
    result.failure = state.failure;
    result.failure_vertex_index = state.failure_vertex_index;
    return false;
  }
  result.arrivals.resize(path.vertices.size());
  result.departures.resize(path.vertices.size());
  result.has_departure.resize(path.vertices.size(), false);
  result.arrivals[0u] = state.weights;
  result.departures[0u] = state;
  result.has_departure[0u] = path.segments.empty() == false;
  for (uint32_t vertex_index = 1u; vertex_index < path.vertices.size(); ++vertex_index) {
    if (upbp_complete_recursive_arrival(scene, path, vertex_index, configuration, state) == false) {
      result.arrivals.clear();
      result.departures.clear();
      result.has_departure.clear();
      result.failure = state.failure;
      result.failure_vertex_index = state.failure_vertex_index;
      return false;
    }
    result.arrivals[vertex_index] = state.weights;
    const bool has_departure = (vertex_index + 1u < path.vertices.size()) || ((vertex_index + 1u == path.vertices.size()) && path.has_terminal_segment);
    if (has_departure) {
      if (upbp_prepare_recursive_departure(scene, path, vertex_index, bpt_sample_count, state) == false) {
        result.arrivals.clear();
        result.departures.clear();
        result.has_departure.clear();
        result.failure = state.failure;
        result.failure_vertex_index = state.failure_vertex_index;
        return false;
      }
      result.departures[vertex_index] = state;
      result.has_departure[vertex_index] = true;
    }
  }
  return true;
}

inline bool upbp_compute_recursive_vertex_weights(const Scene& scene, const UPBPPathRecord& path, const UPBPDensityMISConfiguration& configuration,
  const uint64_t light_subpath_count, const uint64_t bpt_sample_count, std::vector<UPBPRecursiveVertexWeights>& result) {
  UPBPRecursivePathWeights path_weights = {};
  if (upbp_compute_recursive_path_weights(scene, path, configuration, light_subpath_count, bpt_sample_count, path_weights) == false) {
    result.clear();
    return false;
  }
  result = std::move(path_weights.arrivals);
  return true;
}

inline double upbp_medium_real_event_density(const Medium& medium, const SpectralQuery spect, const float3& position) {
  const float density = medium.cls == Medium::Homogeneous ? 1.0f : medium.sample_density_world(position);
  if ((density < 0.0f) || (density > medium_tracking_density_majorant(medium)) || (std::isfinite(density) == false)) {
    return 0.0;
  }
  const double result = static_cast<double>(((medium_scattering(medium, spect) + medium_absorption(medium, spect)) * density).average());
  return (result > 0.0) && std::isfinite(result) ? result : 0.0;
}

inline bool upbp_complete_recursive_partial_medium_arrival(const UPBPPathRecord& path, const uint32_t source_vertex_index, const double log_transport_pdf_forward,
  const double log_transport_pdf_reverse, const float transport_distance, const double query_real_event_density, const UPBPDensityMISConfiguration& configuration,
  UPBPRecursiveState& state) {
  if ((source_vertex_index >= path.vertices.size()) || (transport_distance <= 0.0f) || (query_real_event_density <= 0.0)) {
    return false;
  }
  const UPBPPathVertexRecord& source = path.vertices[source_vertex_index];
  const double forward_pdf = std::exp(log_transport_pdf_forward) * query_real_event_density;
  const double source_event_density = source.cls == UPBPVertexClass::Medium ? std::exp(source.log_medium_event_density) : 1.0;
  const double reverse_pdf = std::exp(log_transport_pdf_reverse) * source_event_density;
  if ((forward_pdf <= 0.0) || (reverse_pdf <= 0.0) || (std::isfinite(forward_pdf) == false) || (std::isfinite(reverse_pdf) == false)) {
    return false;
  }

  if (source_vertex_index > 0u) {
    const double local_factor = upbp_recursive_local_volume_factor(configuration, source.cls, source.delta, source.density_connectible, state.weights, 1.0 / reverse_pdf,
      upbp_short_beam_ray_factor(source), state.last_sin_theta, source.source);
    state.weights.d_bpt_base = state.d_bpt_a * local_factor + state.d_bpt_b;
    state.weights.d_pde_base = state.d_bpt_a * local_factor + state.d_pde_b;
    state.weights.d_surface = state.d_bpt_a * upbp_recursive_surface_coefficient(configuration, source.cls, source.delta, source.density_connectible) + state.d_surface_b;
  }
  state.weights.d_shared /= forward_pdf;
  state.weights.d_bpt_base /= forward_pdf;
  state.weights.d_pde_base /= forward_pdf;
  state.weights.d_surface /= forward_pdf;
  if ((source_vertex_index > 0u) || (path.vertices.front().distant_endpoint == false)) {
    state.weights.d_shared *= static_cast<double>(transport_distance) * transport_distance;
  }
  state.weights.ray_sample_forward_pdf_inverse = 1.0 / forward_pdf;
  state.weights.ray_sample_reverse_pdf_inverse = 1.0 / reverse_pdf;
  state.weights.ray_sample_forward_ratio = 1.0 / query_real_event_density;
  state.weights.ray_sample_reverse_ratio = source.cls == UPBPVertexClass::Medium ? 1.0 / source_event_density : 0.0;
  return std::isfinite(state.weights.d_shared) && std::isfinite(state.weights.d_bpt_base) && (std::isfinite(state.weights.d_pde_base) && std::isfinite(state.weights.d_surface));
}

}  // namespace etx
