#pragma once

#include <etx/rt/integrators/upbp_point_merge.hxx>
#include <etx/rt/integrators/upbp_spatial.hxx>

namespace etx {

struct UPBPBeamContribution {
  SpectralResponse contribution = {};
  double kernel_value = 0.0;
  double mis_weight = 0.0;
  bool applicable = false;
};

inline bool upbp_partial_beam_vertex(const Scene& scene, const SpectralQuery spect, const UPBPPathRecord& path, const UPBPRecursivePathWeights& path_weights,
  const UPBPBeamReference& beam, const float distance, const UPBPDensityMISConfiguration& configuration, UPBPRecursiveVertexWeights& weights, SpectralResponse& throughput) {
  if ((beam.source_vertex_index >= path.vertices.size()) || (beam.source_vertex_index >= path_weights.departures.size()) ||
      (path_weights.has_departure[beam.source_vertex_index] == false)) {
    return false;
  }
  const UPBPSegmentRecord* interval = upbp_beam_interval(path, beam);
  UPBPBeamTransportPrefix transport_prefix = {};
  if ((interval == nullptr) || (interval->medium_index != beam.medium_index) || (upbp_beam_transport_prefix(path, beam, distance, transport_prefix) == false)) {
    return false;
  }
  const float3 position = beam.origin + beam.direction * distance;
  const Medium& medium = scene.mediums[beam.medium_index];
  const double real_event_density = upbp_medium_real_event_density(medium, spect, position);
  if (real_event_density <= 0.0) {
    return false;
  }
  UPBPRecursiveState state = path_weights.departures[beam.source_vertex_index];
  if (upbp_complete_recursive_partial_medium_arrival(path, beam.source_vertex_index, transport_prefix.log_transport_pdf_forward, transport_prefix.log_transport_pdf_reverse,
        transport_prefix.distance, real_event_density, configuration, state) == false) {
    return false;
  }
  weights = state.weights;
  throughput = path.vertices[beam.source_vertex_index].outgoing_throughput * transport_prefix.weight;
  return throughput.is_zero() == false;
}

inline double upbp_medium_phase_sine(const float3& incoming_direction, const float3& outgoing_direction) {
  const double cosine = static_cast<double>(dot(incoming_direction, outgoing_direction));
  return std::sqrt(fmax(0.0, 1.0 - cosine * cosine));
}

inline bool upbp_evaluate_pb2d(const Scene& scene, const SpectralQuery spect, const UPBPPathRecord& light_path, const uint32_t light_vertex_index,
  const UPBPRecursiveVertexWeights& light_weights, const UPBPPathRecord& camera_path, const UPBPRecursivePathWeights& camera_weights, const UPBPBeamReference& camera_beam,
  const UPBPDensityMISConfiguration& configuration, const UPBPKernel kernel, const double radius, const uint64_t light_subpath_count, const uint64_t bpt_sample_count,
  UPBPBeamContribution& result) {
  result = {};
  result.contribution = SpectralResponse{spect, 0.0f};
  if ((light_vertex_index == 0u) || (light_vertex_index >= light_path.vertices.size()) || (light_subpath_count == 0u)) {
    return false;
  }
  const UPBPPathVertexRecord& light_vertex = light_path.vertices[light_vertex_index];
  if ((light_vertex.cls != UPBPVertexClass::Medium) || (light_vertex.medium.index != camera_beam.medium_index) ||
      (spectral_query_compatible(light_vertex.throughput.as_query(), camera_beam.throughput_at_origin.as_query()) == false)) {
    return true;
  }
  UPBPPointBeamIntersection intersection = {};
  if (upbp_intersect_point_beam(light_vertex.position, camera_beam, static_cast<float>(radius), intersection) == false) {
    return true;
  }
  const uint32_t path_length = light_vertex_index + camera_beam.source_vertex_index + 1u;
  if ((path_length < scene.options.min_path_length) || (path_length > scene.options.max_path_length)) {
    return true;
  }

  UPBPRecursiveVertexWeights partial_camera_weights = {};
  SpectralResponse camera_throughput = {};
  if (upbp_partial_beam_vertex(scene, spect, camera_path, camera_weights, camera_beam, intersection.beam_distance, configuration, partial_camera_weights, camera_throughput) ==
      false) {
    return true;
  }
  const float3 query_position = camera_beam.origin + camera_beam.direction * intersection.beam_distance;
  const Medium& medium = scene.mediums[camera_beam.medium_index];
  const SpectralResponse scattering = upbp_medium_scattering_coefficient(medium, spect, query_position);
  const float3 outgoing_direction = -light_vertex.intersection.w_i;
  const double phase = phase_function(camera_beam.direction, outgoing_direction, medium.phase_function_g);
  const double sin_theta = upbp_medium_phase_sine(camera_beam.direction, outgoing_direction);
  result.kernel_value = upbp_kernel_value(kernel, 2u, radius, intersection.distance_squared);
  result.mis_weight = upbp_point_merge_mis_weight({
    UPBPTechnique::PB2D,
    UPBPVertexClass::Medium,
    light_weights,
    partial_camera_weights,
    configuration,
    phase,
    phase,
    sin_theta,
    bpt_sample_count,
  });
  if ((result.kernel_value <= 0.0) || (result.mis_weight <= 0.0) || scattering.is_zero()) {
    return true;
  }
  SpectralResponse light_throughput = {};
  if (upbp_medium_pre_collision_throughput(scene, spect, light_vertex, light_throughput) == false) {
    return false;
  }
  const double estimator_scale = result.kernel_value / static_cast<double>(light_subpath_count);
  result.contribution = light_throughput * camera_throughput * scattering * static_cast<float>(phase * estimator_scale * result.mis_weight);
  result.applicable = true;
  return true;
}

inline bool upbp_evaluate_bp2d(const Scene& scene, const SpectralQuery spect, const UPBPPathRecord& light_path, const UPBPRecursivePathWeights& light_weights,
  const UPBPBeamReference& light_beam, const UPBPPathRecord& camera_path, const uint32_t camera_vertex_index, const UPBPRecursiveVertexWeights& camera_weights,
  const UPBPDensityMISConfiguration& configuration, const UPBPKernel kernel, const double radius, const uint64_t light_subpath_count, const uint64_t bpt_sample_count,
  UPBPBeamContribution& result) {
  result = {};
  result.contribution = SpectralResponse{spect, 0.0f};
  if ((camera_vertex_index == 0u) || (camera_vertex_index >= camera_path.vertices.size()) || (light_subpath_count == 0u)) {
    return false;
  }
  const UPBPSegmentRecord* light_interval = upbp_beam_interval(light_path, light_beam);
  const UPBPPathVertexRecord& camera_vertex = camera_path.vertices[camera_vertex_index];
  if ((light_interval == nullptr) || (camera_vertex.cls != UPBPVertexClass::Medium) || (camera_vertex.medium.index != light_beam.medium_index) ||
      (spectral_query_compatible(camera_vertex.throughput.as_query(), light_beam.throughput_at_origin.as_query()) == false)) {
    return true;
  }
  UPBPPointBeamIntersection intersection = {};
  if (upbp_intersect_point_beam(camera_vertex.position, light_beam, static_cast<float>(radius), intersection) == false) {
    return true;
  }
  const uint32_t path_length = light_beam.source_vertex_index + 1u + camera_vertex_index;
  if ((path_length < scene.options.min_path_length) || (path_length > scene.options.max_path_length)) {
    return true;
  }

  UPBPRecursiveVertexWeights partial_light_weights = {};
  SpectralResponse light_throughput = {};
  if (upbp_partial_beam_vertex(scene, spect, light_path, light_weights, light_beam, intersection.beam_distance, configuration, partial_light_weights, light_throughput) == false) {
    return true;
  }
  const Medium& medium = scene.mediums[light_beam.medium_index];
  const SpectralResponse scattering = upbp_medium_scattering_coefficient(medium, spect, camera_vertex.position);
  const float3 outgoing_direction = -light_beam.direction;
  const double phase = phase_function(camera_vertex.intersection.w_i, outgoing_direction, medium.phase_function_g);
  const double sin_theta = upbp_medium_phase_sine(camera_vertex.intersection.w_i, outgoing_direction);
  result.kernel_value = upbp_kernel_value(kernel, 2u, radius, intersection.distance_squared);
  result.mis_weight = upbp_point_merge_mis_weight({
    UPBPTechnique::BP2D,
    UPBPVertexClass::Medium,
    partial_light_weights,
    camera_weights,
    configuration,
    phase,
    phase,
    sin_theta,
    bpt_sample_count,
  });
  if ((result.kernel_value <= 0.0) || (result.mis_weight <= 0.0) || scattering.is_zero()) {
    return true;
  }
  SpectralResponse camera_throughput = {};
  if (upbp_medium_pre_collision_throughput(scene, spect, camera_vertex, camera_throughput) == false) {
    return false;
  }
  const double estimator_scale = result.kernel_value / static_cast<double>(light_subpath_count);
  result.contribution = light_throughput * camera_throughput * scattering * static_cast<float>(phase * estimator_scale * result.mis_weight);
  result.applicable = true;
  return true;
}

inline bool upbp_evaluate_bb1d(const Scene& scene, const SpectralQuery spect, const UPBPPathRecord& light_path, const UPBPRecursivePathWeights& light_weights,
  const UPBPBeamReference& light_beam, const UPBPPathRecord& camera_path, const UPBPRecursivePathWeights& camera_weights, const UPBPBeamReference& camera_beam,
  const UPBPDensityMISConfiguration& configuration, const UPBPKernel kernel, const double radius, const uint64_t light_subpath_count, const double beam_selection_probability,
  const uint64_t bpt_sample_count, UPBPBeamContribution& result) {
  result = {};
  result.contribution = SpectralResponse{spect, 0.0f};
  if ((light_subpath_count == 0u) || (beam_selection_probability <= 0.0)) {
    return false;
  }
  if (light_beam.medium_index != camera_beam.medium_index) {
    return true;
  }
  const UPBPSegmentRecord* light_interval = upbp_beam_interval(light_path, light_beam);
  if ((light_interval == nullptr) || (spectral_query_compatible(light_beam.throughput_at_origin.as_query(), camera_beam.throughput_at_origin.as_query()) == false)) {
    return true;
  }
  UPBPBeamBeamIntersection intersection = {};
  if (upbp_intersect_beams(light_beam, camera_beam, static_cast<float>(radius), intersection) == false) {
    return true;
  }
  const uint32_t path_length = light_beam.source_vertex_index + camera_beam.source_vertex_index + 2u;
  if ((path_length < scene.options.min_path_length) || (path_length > scene.options.max_path_length)) {
    return true;
  }

  UPBPRecursiveVertexWeights partial_light_weights = {};
  UPBPRecursiveVertexWeights partial_camera_weights = {};
  SpectralResponse light_throughput = {};
  SpectralResponse camera_throughput = {};
  if ((upbp_partial_beam_vertex(scene, spect, light_path, light_weights, light_beam, intersection.first_distance, configuration, partial_light_weights, light_throughput) ==
        false) ||
      (upbp_partial_beam_vertex(scene, spect, camera_path, camera_weights, camera_beam, intersection.second_distance, configuration, partial_camera_weights, camera_throughput) ==
        false)) {
    return true;
  }
  const float3 camera_position = camera_beam.origin + camera_beam.direction * intersection.second_distance;
  const Medium& medium = scene.mediums[camera_beam.medium_index];
  const SpectralResponse scattering = upbp_medium_scattering_coefficient(medium, spect, camera_position);
  const float3 outgoing_direction = -light_beam.direction;
  const double phase = phase_function(camera_beam.direction, outgoing_direction, medium.phase_function_g);
  result.kernel_value = upbp_bb1d_kernel_value(kernel, radius, intersection.distance_squared, intersection.sin_theta);
  result.mis_weight = upbp_point_merge_mis_weight({
    UPBPTechnique::BB1D,
    UPBPVertexClass::Medium,
    partial_light_weights,
    partial_camera_weights,
    configuration,
    phase,
    phase,
    intersection.sin_theta,
    bpt_sample_count,
  });
  if ((result.kernel_value <= 0.0) || (result.mis_weight <= 0.0) || scattering.is_zero()) {
    return true;
  }
  const double estimator_scale = result.kernel_value / (static_cast<double>(light_subpath_count) * beam_selection_probability);
  result.contribution = light_throughput * camera_throughput * scattering * static_cast<float>(phase * estimator_scale * result.mis_weight);
  result.applicable = true;
  return true;
}

}  // namespace etx
