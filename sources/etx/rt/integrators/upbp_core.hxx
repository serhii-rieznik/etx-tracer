#pragma once

#include <etx/rt/medium_tracking.hxx>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace etx {

enum class UPBPTechnique : uint32_t {
  BPT = 1u << 0u,
  Surface = 1u << 1u,
  PP3D = 1u << 2u,
  PB2D = 1u << 3u,
  BP2D = 1u << 4u,
  BB1D = 1u << 5u,
};

enum class UPBPKernel : uint32_t {
  TopHat,
  Epanechnikov,
};

constexpr double kUPBPAutomaticSurfaceRadiusScale = 0.0015;
constexpr double kUPBPAutomaticVolumeRadiusScale = 0.001;

enum class UPBPRandomDomain : uint32_t {
  FilmSample = 0x0d86c4fu,
  CameraPath = 0x17b42d1u,
  LightPath = 0x2eb7193u,
  LightPathSelection = 0x35c84a7u,
  CameraMediumTracking = 0x43ca905u,
  LightMediumTracking = 0x4d2c6dfu,
  ConnectionTransmittance = 0x59d03b7u,
  PB2D = 0x6f14ae9u,
  BP2D = 0x748e2cbu,
  BB1D = 0x8ad753du,
  EmitterConnection = 0x91f02a5u,
  ScatteringEvaluation = 0xa70c3d9u,
  IntersectionTraversal = 0xb86e14fu,
  FilmConnection = 0xc49a721u,
  DirectHitEvaluation = 0xd3158cbu,
  CameraRussianRoulette = 0xe2874adu,
  LightRussianRoulette = 0xf1bbcd9u,
};

ETX_SHARED_INLINE UPBPRandomDomain upbp_medium_tracking_random_domain(const PathSource source) {
  ETX_ASSERT((source == PathSource::Camera) || (source == PathSource::Light));
  return source == PathSource::Camera ? UPBPRandomDomain::CameraMediumTracking : UPBPRandomDomain::LightMediumTracking;
}

ETX_SHARED_INLINE UPBPRandomDomain upbp_russian_roulette_random_domain(const PathSource source) {
  ETX_ASSERT((source == PathSource::Camera) || (source == PathSource::Light));
  return source == PathSource::Camera ? UPBPRandomDomain::CameraRussianRoulette : UPBPRandomDomain::LightRussianRoulette;
}

ETX_SHARED_INLINE uint32_t upbp_sampler_seed(const uint32_t render_seed, const uint64_t iteration, const uint64_t path_index, const uint32_t physical_depth,
  const uint32_t segment_index, const UPBPRandomDomain domain) {
  uint32_t result = Sampler::random_seed(render_seed, static_cast<uint32_t>(domain));
  result = Sampler::random_seed(result, static_cast<uint32_t>(iteration));
  result = Sampler::random_seed(result, static_cast<uint32_t>(iteration >> 32u));
  result = Sampler::random_seed(result, static_cast<uint32_t>(path_index));
  result = Sampler::random_seed(result, static_cast<uint32_t>(path_index >> 32u));
  result = Sampler::random_seed(result, physical_depth);
  return Sampler::random_seed(result, segment_index);
}

inline uint32_t upbp_select_light_path_index(const uint32_t render_seed, const uint64_t iteration, const uint64_t camera_path_index, const uint32_t light_path_count) {
  if (light_path_count == 0u) {
    return kInvalidIndex;
  }
  Sampler sampler{upbp_sampler_seed(render_seed, iteration, camera_path_index, 0u, 0u, UPBPRandomDomain::LightPathSelection)};
  return min(static_cast<uint32_t>(sampler.next() * static_cast<float>(light_path_count)), light_path_count - 1u);
}

inline double upbp_population_radius_scale(const uint64_t camera_subpath_count, const uint64_t light_subpath_count, const uint32_t dimension) {
  if ((camera_subpath_count == 0u) || (light_subpath_count == 0u) || (dimension == 0u)) {
    return 0.0;
  }
  return pow(static_cast<double>(camera_subpath_count) / static_cast<double>(light_subpath_count), 1.0 / static_cast<double>(dimension));
}

inline double upbp_automatic_initial_radius(const double scene_radius, const uint64_t camera_subpath_count, const uint64_t light_subpath_count, const uint32_t dimension,
  const double relative_radius_scale) {
  if ((scene_radius <= 0.0) || (relative_radius_scale <= 0.0) || (std::isfinite(scene_radius) == false) || (std::isfinite(relative_radius_scale) == false)) {
    return 0.0;
  }
  return scene_radius * relative_radius_scale * upbp_population_radius_scale(camera_subpath_count, light_subpath_count, dimension);
}

struct UPBPTechniqueProbability {
  UPBPTechnique technique = UPBPTechnique::BPT;
  double log_density = 0.0;
  uint64_t sample_count = 0u;
  bool applicable = false;
};

enum class UPBPVertexClass : uint8_t {
  Camera,
  Emitter,
  Surface,
  Medium,
};

inline bool upbp_projected_measure_valid(const double cosine) {
  return (cosine >= static_cast<double>(kEpsilon)) && std::isfinite(cosine);
}

enum class UPBPPathFailure : uint8_t {
  None,
  InvalidSegment,
  InvalidVertex,
  CapacityExceeded,
};

struct UPBPMediumTrackingEventRecord {
  SpectralResponse weight_before = {};
  double log_transport_pdf_forward_before = 0.0;
  double log_transport_pdf_reverse_before = 0.0;
  float distance_before = 0.0f;
  float end_distance = 0.0f;
  float majorant = 0.0f;
};

struct UPBPSegmentRecord {
  std::vector<UPBPMediumTrackingEventRecord> events = {};
  SpectralResponse weight = {};
  double log_pdf_forward = 0.0;
  double log_pdf_reverse = 0.0;
  double log_transport_pdf_forward = 0.0;
  double log_transport_pdf_reverse = 0.0;
  double log_terminal_event_density = 0.0;
  float distance = 0.0f;
  float3 start_position = {};
  float3 end_position = {};
  uint32_t medium_index = kInvalidIndex;
  uint32_t event_count = 0u;
  uint32_t null_event_count = 0u;
  MediumTrackingEventType terminal_event = MediumTrackingEventType::Escape;
  MediumTrackingFailure failure = MediumTrackingFailure::None;
  bool complete = false;

  void reset(const SpectralQuery spect, const uint32_t active_medium_index, const float3& origin) {
    events.clear();
    weight = SpectralResponse{spect, 1.0f};
    log_pdf_forward = 0.0;
    log_pdf_reverse = 0.0;
    log_transport_pdf_forward = 0.0;
    log_transport_pdf_reverse = 0.0;
    log_terminal_event_density = 0.0;
    distance = 0.0f;
    start_position = origin;
    end_position = origin;
    medium_index = active_medium_index;
    event_count = 0u;
    null_event_count = 0u;
    terminal_event = MediumTrackingEventType::Escape;
    failure = MediumTrackingFailure::None;
    complete = false;
  }

  bool append(const MediumTrackingEvent& event) {
    if (complete) {
      failure = MediumTrackingFailure::InvalidInput;
      return false;
    }

    if (event.valid() == false) {
      failure = event.failure;
      return false;
    }

    if ((event.pdf_forward <= 0.0f) || (event.pdf_reverse <= 0.0f) || (std::isfinite(event.pdf_forward) == false) || (std::isfinite(event.pdf_reverse) == false)) {
      failure = MediumTrackingFailure::InvalidProbability;
      return false;
    }
    const bool terminal_medium_event = (event.type == MediumTrackingEventType::Scatter) || (event.type == MediumTrackingEventType::Absorb);
    if (terminal_medium_event && ((event.majorant_transmittance <= 0.0f) || (event.event_probability <= 0.0f) || (event.majorant <= 0.0f))) {
      failure = MediumTrackingFailure::InvalidProbability;
      return false;
    }

    events.emplace_back(UPBPMediumTrackingEventRecord{
      weight,
      log_transport_pdf_forward,
      log_transport_pdf_reverse,
      distance,
      distance + event.distance,
      event.majorant,
    });
    distance += event.distance;
    end_position = event.position;
    log_pdf_forward += std::log(static_cast<double>(event.pdf_forward));
    log_pdf_reverse += std::log(static_cast<double>(event.pdf_reverse));
    ++event_count;
    if (event.type == MediumTrackingEventType::Null) {
      log_transport_pdf_forward += std::log(static_cast<double>(event.pdf_forward));
      log_transport_pdf_reverse += std::log(static_cast<double>(event.pdf_reverse));
      weight *= event.weight;
      ++null_event_count;
      return true;
    }

    if (terminal_medium_event) {
      const double event_density = static_cast<double>(event.event_probability) * event.majorant;
      log_transport_pdf_forward += std::log(static_cast<double>(event.majorant_transmittance));
      log_transport_pdf_reverse += std::log(static_cast<double>(event.majorant_transmittance));
      log_terminal_event_density = std::log(event_density);
      weight *= event.weight;
    } else {
      log_transport_pdf_forward += std::log(static_cast<double>(event.pdf_forward));
      log_transport_pdf_reverse += std::log(static_cast<double>(event.pdf_reverse));
    }

    terminal_event = event.type;
    complete = true;
    return true;
  }

  double pdf_forward() const {
    return std::exp(log_pdf_forward);
  }

  double pdf_reverse() const {
    return std::exp(log_pdf_reverse);
  }

  bool valid() const {
    return failure == MediumTrackingFailure::None;
  }
};

struct UPBPTransportSegmentRecord {
  std::vector<UPBPSegmentRecord> intervals = {};
  SpectralResponse weight = {};
  double log_pdf_forward = 0.0;
  double log_pdf_reverse = 0.0;
  double log_transport_pdf_forward = 0.0;
  double log_transport_pdf_reverse = 0.0;
  double log_terminal_event_density = 0.0;
  float distance = 0.0f;
  uint32_t boundary_count = 0u;
  MediumTrackingFailure failure = MediumTrackingFailure::None;
  bool has_terminal_event_density = false;

  void reset(const SpectralQuery spect) {
    intervals.clear();
    weight = SpectralResponse{spect, 1.0f};
    log_pdf_forward = 0.0;
    log_pdf_reverse = 0.0;
    log_transport_pdf_forward = 0.0;
    log_transport_pdf_reverse = 0.0;
    log_terminal_event_density = 0.0;
    distance = 0.0f;
    boundary_count = 0u;
    failure = MediumTrackingFailure::None;
    has_terminal_event_density = false;
  }

  bool append(const UPBPSegmentRecord& interval) {
    if ((failure != MediumTrackingFailure::None) || (interval.complete == false) || (interval.valid() == false)) {
      failure = interval.failure == MediumTrackingFailure::None ? MediumTrackingFailure::InvalidInput : interval.failure;
      return false;
    }

    const bool terminal_medium_event = (interval.terminal_event == MediumTrackingEventType::Scatter) || (interval.terminal_event == MediumTrackingEventType::Absorb);
    if (terminal_medium_event && has_terminal_event_density) {
      failure = MediumTrackingFailure::InvalidInput;
      return false;
    }

    intervals.emplace_back(interval);
    weight *= interval.weight;
    log_pdf_forward += interval.log_pdf_forward;
    log_pdf_reverse += interval.log_pdf_reverse;
    log_transport_pdf_forward += interval.log_transport_pdf_forward;
    log_transport_pdf_reverse += interval.log_transport_pdf_reverse;
    if (terminal_medium_event) {
      log_terminal_event_density = interval.log_terminal_event_density;
      has_terminal_event_density = true;
    }
    distance += interval.distance;
    return true;
  }

  void record_boundary() {
    ++boundary_count;
  }

  uint32_t null_event_count() const {
    uint32_t result = 0u;
    for (const UPBPSegmentRecord& interval : intervals) {
      result += interval.null_event_count;
    }
    return result;
  }

  bool valid() const {
    return (failure == MediumTrackingFailure::None) && (intervals.empty() == false);
  }
};

template <typename DensityEvaluator>
bool upbp_track_medium_segment(const MediumTrackingInput& input, Sampler& sampler, const float3& origin, const float3& direction, const float maximum_distance,
  const uint32_t medium_index, const uint32_t maximum_null_events, DensityEvaluator&& evaluate_density, UPBPSegmentRecord& result) {
  result.reset(input.spect, medium_index, origin);
  float3 event_origin = origin;
  float remaining_distance = maximum_distance;

  for (;;) {
    const MediumTrackingEvent event = sample_medium_tracking_weighted_scattering_event(input, sampler, event_origin, direction, remaining_distance, evaluate_density);
    if (result.append(event) == false) {
      return false;
    }

    if (event.type != MediumTrackingEventType::Null) {
      return true;
    }

    if (result.null_event_count > maximum_null_events) {
      result.failure = MediumTrackingFailure::EventLimitExceeded;
      return false;
    }

    event_origin = event.position;
    remaining_distance -= event.distance;
    if ((remaining_distance <= 0.0f) || (std::isfinite(remaining_distance) == false)) {
      result.failure = MediumTrackingFailure::InvalidInput;
      return false;
    }
  }
}

inline bool upbp_track_medium_segment(const Medium& medium, const SpectralQuery spect, Sampler& sampler, const float3& origin, const float3& direction,
  const float maximum_distance, const uint32_t medium_index, const uint32_t maximum_null_events, UPBPSegmentRecord& result) {
  const MediumTrackingInput input = make_medium_tracking_input(medium, spect);
  auto density_evaluator = [&medium](const float3& position) {
    return (medium.cls == Medium::Homogeneous) ? 1.0f : medium.sample_density_world(position);
  };
  return upbp_track_medium_segment(input, sampler, origin, direction, maximum_distance, medium_index, maximum_null_events, density_evaluator, result);
}

struct UPBPPathVertexRecord {
  float3 position = {};
  float3 sampled_direction = {};
  Intersection intersection = {};
  MediumInstance medium = {};
  SpectralResponse throughput = {};
  SpectralResponse outgoing_throughput = {};
  float scatter_pdf_forward = 0.0f;
  float scatter_pdf_reverse = 0.0f;
  float endpoint_pdf_area = 0.0f;
  float endpoint_pdf_sample = 0.0f;
  float endpoint_pdf_direction = 0.0f;
  double log_medium_event_density = 0.0;
  float eta = 1.0f;
  uint32_t incident_medium_index = kInvalidIndex;
  uint32_t outgoing_medium_index = kInvalidIndex;
  uint32_t sample_properties = 0u;
  UPBPVertexClass cls = UPBPVertexClass::Camera;
  PathSource source = PathSource::Undefined;
  bool connectible = false;
  bool density_connectible = true;
  bool delta = false;
  bool distant_endpoint = false;
};

struct UPBPPathRecord {
  std::vector<UPBPPathVertexRecord> vertices = {};
  std::vector<UPBPTransportSegmentRecord> segments = {};
  UPBPTransportSegmentRecord terminal_segment = {};
  UPBPPathFailure failure = UPBPPathFailure::None;
  uint32_t maximum_physical_vertices = 0u;
  uint32_t boundary_count = 0u;
  uint32_t null_event_count = 0u;
  bool has_terminal_segment = false;

  void reset(const uint32_t maximum_vertices, const uint32_t initial_capacity) {
    vertices.clear();
    segments.clear();
    vertices.reserve(min(maximum_vertices, initial_capacity));
    const uint32_t maximum_segments = maximum_vertices > 0u ? maximum_vertices - 1u : 0u;
    segments.reserve(min(maximum_segments, initial_capacity));
    failure = UPBPPathFailure::None;
    maximum_physical_vertices = maximum_vertices;
    boundary_count = 0u;
    null_event_count = 0u;
    has_terminal_segment = false;
  }

  void reset(const uint32_t maximum_vertices) {
    reset(maximum_vertices, 0u);
  }

  bool append_endpoint(const UPBPPathVertexRecord& vertex) {
    if ((failure != UPBPPathFailure::None) || (vertices.empty() == false)) {
      failure = UPBPPathFailure::InvalidVertex;
      return false;
    }

    if (maximum_physical_vertices == 0u) {
      failure = UPBPPathFailure::CapacityExceeded;
      return false;
    }

    vertices.emplace_back(vertex);
    return true;
  }

  bool append_physical_vertex(const UPBPTransportSegmentRecord& segment, const UPBPPathVertexRecord& vertex) {
    if ((failure != UPBPPathFailure::None) || vertices.empty() || (segment.valid() == false)) {
      failure = UPBPPathFailure::InvalidSegment;
      return false;
    }

    if (vertices.size() >= maximum_physical_vertices) {
      failure = UPBPPathFailure::CapacityExceeded;
      return false;
    }

    segments.emplace_back(segment);
    vertices.emplace_back(vertex);
    boundary_count += segment.boundary_count;
    null_event_count += segment.null_event_count();
    return true;
  }

  bool append_terminal_segment(const UPBPTransportSegmentRecord& segment) {
    if ((failure != UPBPPathFailure::None) || vertices.empty() || has_terminal_segment || (segment.valid() == false)) {
      failure = UPBPPathFailure::InvalidSegment;
      return false;
    }

    terminal_segment = segment;
    boundary_count += segment.boundary_count;
    null_event_count += segment.null_event_count();
    has_terminal_segment = true;
    return true;
  }

  bool promote_terminal_segment(const UPBPPathVertexRecord& vertex) {
    if ((failure != UPBPPathFailure::None) || (has_terminal_segment == false) || (terminal_segment.valid() == false)) {
      failure = UPBPPathFailure::InvalidSegment;
      return false;
    }

    if (vertices.size() >= maximum_physical_vertices) {
      failure = UPBPPathFailure::CapacityExceeded;
      return false;
    }

    segments.emplace_back(std::move(terminal_segment));
    vertices.emplace_back(vertex);
    terminal_segment = {};
    has_terminal_segment = false;
    return true;
  }

  void record_boundary() {
    ++boundary_count;
  }

  uint32_t physical_length() const {
    return static_cast<uint32_t>(segments.size());
  }

  bool valid() const {
    return (failure == UPBPPathFailure::None) && (vertices.size() == segments.size() + 1u) && ((has_terminal_segment == false) || terminal_segment.valid());
  }
};

struct UPBPMISAccumulator {
  double maximum_log_term = -std::numeric_limits<double>::infinity();
  double scaled_sum = 0.0;
  double selected_log_term = -std::numeric_limits<double>::infinity();
  uint32_t candidate_count = 0u;
  bool selected_seen = false;

  bool append(const UPBPTechniqueProbability& probability, const bool selected) {
    if ((probability.applicable == false) || (probability.sample_count == 0u) || (std::isfinite(probability.log_density) == false)) {
      return selected == false;
    }

    const double log_term = std::log(static_cast<double>(probability.sample_count)) + probability.log_density;
    if (candidate_count == 0u) {
      maximum_log_term = log_term;
      scaled_sum = 1.0;
    } else if (log_term <= maximum_log_term) {
      scaled_sum += std::exp(log_term - maximum_log_term);
    } else {
      scaled_sum = scaled_sum * std::exp(maximum_log_term - log_term) + 1.0;
      maximum_log_term = log_term;
    }

    ++candidate_count;
    if (selected) {
      if (selected_seen) {
        return false;
      }
      selected_log_term = log_term;
      selected_seen = true;
    }
    return true;
  }

  double weight() const {
    if ((selected_seen == false) || (candidate_count == 0u) || (scaled_sum <= 0.0)) {
      return 0.0;
    }
    return std::exp(selected_log_term - maximum_log_term) / scaled_sum;
  }
};

struct UPBPPathProbabilityVertex {
  bool connectible = true;
};

struct UPBPPathProbabilityEdge {
  double log_density_forward = 0.0;
  double log_density_reverse = 0.0;
};

enum class UPBPPathProbabilityFailure : uint8_t {
  None,
  InvalidJoinedPath,
  InvalidEndpointClasses,
  InvalidEmitterDensity,
  InvalidCameraDensity,
  DegenerateEdge,
  InvalidRecord,
};

struct UPBPPathProbabilityRecord {
  std::vector<UPBPPathProbabilityVertex> vertices = {};
  std::vector<UPBPPathProbabilityEdge> edges = {};
  double log_emitter_endpoint_density = 0.0;
  double log_camera_endpoint_density = 0.0;
  UPBPPathProbabilityFailure failure = UPBPPathProbabilityFailure::None;

  bool valid() const {
    auto valid_log_density = [](const double value) {
      return std::isfinite(value) || (value == -std::numeric_limits<double>::infinity());
    };
    if ((vertices.size() < 2u) || (edges.size() + 1u != vertices.size()) || (valid_log_density(log_emitter_endpoint_density) == false) ||
        (valid_log_density(log_camera_endpoint_density) == false)) {
      return false;
    }

    for (const UPBPPathProbabilityEdge& edge : edges) {
      if ((valid_log_density(edge.log_density_forward) == false) || (valid_log_density(edge.log_density_reverse) == false)) {
        return false;
      }
    }
    return true;
  }
};

struct UPBPBPTStrategyProbability {
  uint32_t light_vertex_count = 0u;
  double log_density = 0.0;
  bool applicable = false;
};

inline bool upbp_bpt_strategy_applicable(const UPBPPathProbabilityRecord& path, const uint32_t light_vertex_count) {
  if ((path.valid() == false) || (light_vertex_count >= path.vertices.size())) {
    return false;
  }

  if (light_vertex_count == 0u) {
    return true;
  }

  return path.vertices[light_vertex_count - 1u].connectible && path.vertices[light_vertex_count].connectible;
}

inline double upbp_bpt_strategy_log_density_exhaustive(const UPBPPathProbabilityRecord& path, const uint32_t light_vertex_count) {
  if ((path.valid() == false) || (light_vertex_count >= path.vertices.size())) {
    return -std::numeric_limits<double>::infinity();
  }

  double result = path.log_camera_endpoint_density;
  if (light_vertex_count == 0u) {
    for (const UPBPPathProbabilityEdge& edge : path.edges) {
      result += edge.log_density_reverse;
    }
    return result;
  }

  result += path.log_emitter_endpoint_density;
  for (uint32_t edge_index = 0u; edge_index + 1u < light_vertex_count; ++edge_index) {
    result += path.edges[edge_index].log_density_forward;
  }
  for (uint32_t edge_index = light_vertex_count; edge_index < path.edges.size(); ++edge_index) {
    result += path.edges[edge_index].log_density_reverse;
  }
  return result;
}

inline bool upbp_enumerate_bpt_strategies_exhaustive(const UPBPPathProbabilityRecord& path, std::vector<UPBPBPTStrategyProbability>& result) {
  result.clear();
  if (path.valid() == false) {
    return false;
  }

  result.reserve(path.vertices.size());
  for (uint32_t light_vertex_count = 0u; light_vertex_count < path.vertices.size(); ++light_vertex_count) {
    result.emplace_back(UPBPBPTStrategyProbability{
      light_vertex_count,
      upbp_bpt_strategy_log_density_exhaustive(path, light_vertex_count),
      upbp_bpt_strategy_applicable(path, light_vertex_count),
    });
  }
  return true;
}

inline bool upbp_enumerate_bpt_strategies_recursive(const UPBPPathProbabilityRecord& path, std::vector<UPBPBPTStrategyProbability>& result) {
  result.clear();
  if (path.valid() == false) {
    return false;
  }

  result.reserve(path.vertices.size());
  double log_density = path.log_camera_endpoint_density;
  for (const UPBPPathProbabilityEdge& edge : path.edges) {
    log_density += edge.log_density_reverse;
  }
  result.emplace_back(UPBPBPTStrategyProbability{0u, log_density, true});

  if (std::isfinite(log_density) && std::isfinite(path.log_emitter_endpoint_density) && std::isfinite(path.edges[0u].log_density_reverse)) {
    log_density += path.log_emitter_endpoint_density - path.edges[0u].log_density_reverse;
  } else {
    log_density = upbp_bpt_strategy_log_density_exhaustive(path, 1u);
  }
  result.emplace_back(UPBPBPTStrategyProbability{1u, log_density, upbp_bpt_strategy_applicable(path, 1u)});
  for (uint32_t light_vertex_count = 2u; light_vertex_count < path.vertices.size(); ++light_vertex_count) {
    const double forward_density = path.edges[light_vertex_count - 2u].log_density_forward;
    const double reverse_density = path.edges[light_vertex_count - 1u].log_density_reverse;
    if (std::isfinite(log_density) && std::isfinite(forward_density) && std::isfinite(reverse_density)) {
      log_density += forward_density - reverse_density;
    } else {
      log_density = upbp_bpt_strategy_log_density_exhaustive(path, light_vertex_count);
    }
    result.emplace_back(UPBPBPTStrategyProbability{
      light_vertex_count,
      log_density,
      upbp_bpt_strategy_applicable(path, light_vertex_count),
    });
  }
  return true;
}

inline double upbp_bpt_balance_weight(const std::vector<UPBPBPTStrategyProbability>& probabilities, const uint32_t selected_light_vertex_count) {
  UPBPMISAccumulator accumulator = {};
  for (const UPBPBPTStrategyProbability& probability : probabilities) {
    const UPBPTechniqueProbability candidate = {
      UPBPTechnique::BPT,
      probability.log_density,
      1u,
      probability.applicable,
    };
    if (accumulator.append(candidate, probability.light_vertex_count == selected_light_vertex_count) == false) {
      return 0.0;
    }
  }
  return accumulator.weight();
}

ETX_SHARED_INLINE uint32_t upbp_kernel_dimension(const UPBPTechnique technique) {
  if (technique == UPBPTechnique::PP3D) {
    return 3u;
  }

  if (technique == UPBPTechnique::BB1D) {
    return 1u;
  }

  if ((technique == UPBPTechnique::Surface) || (technique == UPBPTechnique::PB2D) || (technique == UPBPTechnique::BP2D)) {
    return 2u;
  }

  return 0u;
}

ETX_SHARED_INLINE double upbp_support_measure(const uint32_t dimension, const double radius) {
  if ((radius <= 0.0) || (std::isfinite(radius) == false)) {
    return 0.0;
  }

  if (dimension == 1u) {
    return 2.0 * radius;
  }

  if (dimension == 2u) {
    return static_cast<double>(kPi) * radius * radius;
  }

  if (dimension == 3u) {
    return (4.0 / 3.0) * static_cast<double>(kPi) * radius * radius * radius;
  }

  return 0.0;
}

ETX_SHARED_INLINE double upbp_kernel_value(const UPBPKernel kernel, const uint32_t dimension, const double radius, const double distance_squared) {
  if ((distance_squared < 0.0) || (radius <= 0.0) || (std::isfinite(distance_squared) == false) || (std::isfinite(radius) == false)) {
    return 0.0;
  }

  const double radius_squared = radius * radius;
  if (distance_squared >= radius_squared) {
    return 0.0;
  }

  const double support_measure = upbp_support_measure(dimension, radius);
  if (support_measure <= 0.0) {
    return 0.0;
  }

  if (kernel == UPBPKernel::TopHat) {
    return 1.0 / support_measure;
  }

  const double profile = 1.0 - distance_squared / radius_squared;
  if (dimension == 1u) {
    return 3.0 * profile / (4.0 * radius);
  }

  if (dimension == 2u) {
    return 2.0 * profile / (static_cast<double>(kPi) * radius_squared);
  }

  if (dimension == 3u) {
    return 15.0 * profile / (8.0 * static_cast<double>(kPi) * radius_squared * radius);
  }

  return 0.0;
}

ETX_SHARED_INLINE double upbp_progressive_radius(const double initial_radius, const double alpha, const uint32_t dimension, const uint64_t iteration) {
  if ((initial_radius <= 0.0) || (alpha <= 0.0) || (alpha > 1.0) || (dimension == 0u) || (std::isfinite(initial_radius) == false) || (std::isfinite(alpha) == false)) {
    return 0.0;
  }

  const double exponent = (alpha - 1.0) / static_cast<double>(dimension);
  return initial_radius * std::pow(static_cast<double>(iteration) + 1.0, exponent);
}

ETX_SHARED_INLINE double upbp_progressive_radius_for_sample_fraction(const double initial_radius, const double alpha, const uint32_t dimension, const uint64_t iteration,
  const double sample_fraction) {
  if ((initial_radius <= 0.0) || (alpha <= 0.0) || (alpha > 1.0) || (dimension == 0u) || (sample_fraction <= 0.0) || (sample_fraction > 1.0) ||
      (std::isfinite(initial_radius) == false) || (std::isfinite(alpha) == false) || (std::isfinite(sample_fraction) == false)) {
    return 0.0;
  }

  const double exponent = (alpha - 1.0) / static_cast<double>(dimension);
  const double effective_iteration = static_cast<double>(iteration) * sample_fraction;
  return initial_radius * std::pow(effective_iteration + 1.0, exponent);
}

ETX_SHARED_INLINE double upbp_density_mis_factor(const UPBPTechnique technique, const uint64_t light_subpath_count, const double radius, const double beam_selection_probability) {
  if ((light_subpath_count == 0u) || (radius <= 0.0) || (beam_selection_probability <= 0.0) || (beam_selection_probability > 1.0)) {
    return 0.0;
  }

  const double path_count = static_cast<double>(light_subpath_count);
  if (technique == UPBPTechnique::BB1D) {
    return 0.5 * radius * path_count * beam_selection_probability;
  }

  const uint32_t dimension = upbp_kernel_dimension(technique);
  return upbp_support_measure(dimension, radius) * path_count;
}

struct UPBPDensityCompetitorInput {
  UPBPTechnique technique = UPBPTechnique::BPT;
  UPBPVertexClass vertex_class = UPBPVertexClass::Surface;
  double density_mis_factor = 0.0;
  double forward_ray_factor = 0.0;
  double reverse_ray_factor = 0.0;
  double sin_theta = 0.0;
  bool delta = false;
};

ETX_SHARED_INLINE double upbp_density_competitor_factor(const UPBPDensityCompetitorInput& input) {
  if ((input.density_mis_factor <= 0.0) || (std::isfinite(input.density_mis_factor) == false) || input.delta) {
    return 0.0;
  }

  if (input.technique == UPBPTechnique::Surface) {
    return input.vertex_class == UPBPVertexClass::Surface ? input.density_mis_factor : 0.0;
  }
  if (input.vertex_class != UPBPVertexClass::Medium) {
    return 0.0;
  }

  if (input.technique == UPBPTechnique::PP3D) {
    return input.density_mis_factor;
  }
  if (input.technique == UPBPTechnique::PB2D) {
    return input.reverse_ray_factor > 0.0 ? input.density_mis_factor * input.reverse_ray_factor : 0.0;
  }
  if (input.technique == UPBPTechnique::BP2D) {
    return input.forward_ray_factor > 0.0 ? input.density_mis_factor * input.forward_ray_factor : 0.0;
  }
  if (input.technique == UPBPTechnique::BB1D) {
    return ((input.forward_ray_factor > 0.0) && (input.reverse_ray_factor > 0.0) && (input.sin_theta > 0.0))
             ? input.density_mis_factor * input.sin_theta * input.forward_ray_factor * input.reverse_ray_factor
             : 0.0;
  }
  return 0.0;
}

ETX_SHARED_INLINE double upbp_bb1d_kernel_value(const UPBPKernel kernel, const double radius, const double distance_squared, const double sin_theta) {
  if ((sin_theta <= 0.0) || (std::isfinite(sin_theta) == false)) {
    return 0.0;
  }
  return upbp_kernel_value(kernel, 1u, radius, distance_squared) / sin_theta;
}

ETX_SHARED_INLINE double upbp_density_estimator_scale(const UPBPTechnique technique, const UPBPKernel kernel, const uint64_t light_subpath_count, const double radius,
  const double distance_squared, const double sin_theta, const double beam_selection_probability) {
  if ((light_subpath_count == 0u) || (beam_selection_probability <= 0.0) || (beam_selection_probability > 1.0)) {
    return 0.0;
  }

  const double kernel_value = technique == UPBPTechnique::BB1D ? upbp_bb1d_kernel_value(kernel, radius, distance_squared, sin_theta)
                                                               : upbp_kernel_value(kernel, upbp_kernel_dimension(technique), radius, distance_squared);
  return kernel_value / (static_cast<double>(light_subpath_count) * beam_selection_probability);
}

ETX_SHARED_INLINE double upbp_balance_weight(const UPBPTechniqueProbability* probabilities, const uint32_t probability_count, const uint32_t selected_index) {
  if ((probabilities == nullptr) || (selected_index >= probability_count)) {
    return 0.0;
  }

  const UPBPTechniqueProbability& selected = probabilities[selected_index];
  if ((selected.applicable == false) || (selected.sample_count == 0u) || (std::isfinite(selected.log_density) == false)) {
    return 0.0;
  }

  double maximum_log_term = -std::numeric_limits<double>::infinity();
  for (uint32_t index = 0u; index < probability_count; ++index) {
    const UPBPTechniqueProbability& probability = probabilities[index];
    if ((probability.applicable == false) || (probability.sample_count == 0u) || (std::isfinite(probability.log_density) == false)) {
      continue;
    }

    const double log_term = std::log(static_cast<double>(probability.sample_count)) + probability.log_density;
    maximum_log_term = std::max(maximum_log_term, log_term);
  }

  if (std::isfinite(maximum_log_term) == false) {
    return 0.0;
  }

  double denominator = 0.0;
  for (uint32_t index = 0u; index < probability_count; ++index) {
    const UPBPTechniqueProbability& probability = probabilities[index];
    if ((probability.applicable == false) || (probability.sample_count == 0u) || (std::isfinite(probability.log_density) == false)) {
      continue;
    }

    const double log_term = std::log(static_cast<double>(probability.sample_count)) + probability.log_density;
    denominator += std::exp(log_term - maximum_log_term);
  }

  const double selected_log_term = std::log(static_cast<double>(selected.sample_count)) + selected.log_density;
  return std::exp(selected_log_term - maximum_log_term) / denominator;
}

}  // namespace etx
