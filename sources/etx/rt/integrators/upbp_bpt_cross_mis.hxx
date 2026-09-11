#pragma once

#include <etx/rt/integrators/upbp_beam_estimators.hxx>
#include <etx/rt/integrators/upbp_mis.hxx>

namespace etx {

inline double upbp_connection_sampling_density(const Scene& scene, const UPBPTransportSegmentRecord& segment, const UPBPPathVertexRecord& source,
  const UPBPPathVertexRecord& target, const double direction_pdf, const bool segment_reversed) {
  if (direction_pdf <= 0.0) {
    return 0.0;
  }
  const float3 direction = normalize(target.position - source.position);
  const double distance_squared = dot(target.position - source.position, target.position - source.position);
  const double transport_log_density = segment_reversed ? segment.log_transport_pdf_reverse : segment.log_transport_pdf_forward;
  const double target_event_log_density = target.cls == UPBPVertexClass::Medium ? target.log_medium_event_density : 0.0;
  const double log_density = std::log(direction_pdf) + transport_log_density + target_event_log_density + upbp_log_measure_conversion(scene, target, direction, distance_squared);
  return std::isfinite(log_density) ? std::exp(log_density) : 0.0;
}

struct UPBPBPTConnectionMISInput {
  const Scene* scene = nullptr;
  const UPBPPathVertexRecord* light_vertex = nullptr;
  const UPBPPathVertexRecord* camera_vertex = nullptr;
  const UPBPTransportSegmentRecord* connection_segment = nullptr;
  UPBPRecursiveVertexWeights light_weights = {};
  UPBPRecursiveVertexWeights camera_weights = {};
  UPBPScatteringEval light_scattering = {};
  UPBPScatteringEval camera_scattering = {};
  UPBPDensityMISConfiguration configuration = {};
};

inline void upbp_bpt_add_vertex_mis_terms(UPBPSurfaceMISWeights& result, const UPBPDensityMISConfiguration& configuration, const UPBPPathVertexRecord& vertex,
  const UPBPRecursiveVertexWeights& weights, const double sampling_density, const double local_volume_factor, const double scattering_pdf_reverse) {
  const double surface_factor = configuration.factor(UPBPTechnique::Surface);
  const double local_surface_coefficient = upbp_recursive_surface_coefficient(configuration, vertex.cls, vertex.delta, vertex.density_connectible);
  result.add_term(false, {sampling_density, local_volume_factor}, {});
  result.add_term(false, {sampling_density, static_cast<double>(weights.previous_delta == false), weights.d_shared}, {});
  result.add_term(false, {sampling_density, scattering_pdf_reverse, weights.d_bpt_base}, {weights.ray_sample_reverse_pdf_inverse});
  result.add_term(true, {sampling_density, surface_factor, local_surface_coefficient}, {});
  result.add_term(true, {sampling_density, scattering_pdf_reverse, surface_factor, weights.d_surface}, {weights.ray_sample_reverse_pdf_inverse});
}

inline UPBPSurfaceMISWeights upbp_bpt_connection_cross_technique_weights(const UPBPBPTConnectionMISInput& input) {
  if ((input.scene == nullptr) || (input.light_vertex == nullptr) || (input.camera_vertex == nullptr) || (input.connection_segment == nullptr) ||
      (input.configuration.enabled(UPBPTechnique::BPT) == false) || (input.light_weights.ray_sample_reverse_pdf_inverse <= 0.0) ||
      (input.camera_weights.ray_sample_reverse_pdf_inverse <= 0.0)) {
    return {};
  }
  const UPBPPathVertexRecord& light_vertex = *input.light_vertex;
  const UPBPPathVertexRecord& camera_vertex = *input.camera_vertex;
  const double light_event_density = light_vertex.cls == UPBPVertexClass::Medium ? std::exp(light_vertex.log_medium_event_density) : 1.0;
  const double camera_event_density = camera_vertex.cls == UPBPVertexClass::Medium ? std::exp(camera_vertex.log_medium_event_density) : 1.0;
  const double camera_to_light_transport = std::exp(input.connection_segment->log_transport_pdf_reverse);
  const double light_to_camera_transport = std::exp(input.connection_segment->log_transport_pdf_forward);
  const float3 light_to_camera_direction = normalize(camera_vertex.position - light_vertex.position);
  const double light_sin_theta = upbp_medium_phase_sine(light_vertex.intersection.w_i, light_to_camera_direction);
  const double camera_sin_theta = upbp_medium_phase_sine(camera_vertex.intersection.w_i, -light_to_camera_direction);
  const double light_local_factor =
    upbp_recursive_local_volume_factor(input.configuration, light_vertex.cls, light_vertex.delta, light_vertex.density_connectible, input.light_weights,
      1.0 / (camera_to_light_transport * light_event_density), light_vertex.cls == UPBPVertexClass::Medium ? 1.0 / light_event_density : 0.0, light_sin_theta, PathSource::Light);
  const double camera_local_factor = upbp_recursive_local_volume_factor(input.configuration, camera_vertex.cls, camera_vertex.delta, camera_vertex.density_connectible,
    input.camera_weights, 1.0 / (light_to_camera_transport * camera_event_density), camera_vertex.cls == UPBPVertexClass::Medium ? 1.0 / camera_event_density : 0.0,
    camera_sin_theta, PathSource::Camera);

  const double camera_to_light_density =
    upbp_connection_sampling_density(*input.scene, *input.connection_segment, camera_vertex, light_vertex, input.camera_scattering.pdf_forward, true);
  const double light_to_camera_density =
    upbp_connection_sampling_density(*input.scene, *input.connection_segment, light_vertex, camera_vertex, input.light_scattering.pdf_forward, false);
  UPBPSurfaceMISWeights result = {1.0};
  upbp_bpt_add_vertex_mis_terms(result, input.configuration, light_vertex, input.light_weights, camera_to_light_density, light_local_factor, input.light_scattering.pdf_reverse);
  upbp_bpt_add_vertex_mis_terms(result, input.configuration, camera_vertex, input.camera_weights, light_to_camera_density, camera_local_factor,
    input.camera_scattering.pdf_reverse);
  return result;
}

inline double upbp_bpt_connection_cross_technique_weight(const UPBPBPTConnectionMISInput& input) {
  return upbp_bpt_connection_cross_technique_weights(input).weight(1.0);
}

inline UPBPSurfaceMISWeights upbp_bpt_direct_hit_cross_technique_weights(const UPBPPathRecord& camera_path, const uint32_t emitter_vertex_index,
  const UPBPRecursiveVertexWeights& camera_weights, const double emitter_selection_pdf, const double direct_area_pdf, const double emission_direction_pdf,
  const UPBPDensityMISConfiguration& configuration) {
  if ((configuration.enabled(UPBPTechnique::BPT) == false) || (emitter_vertex_index == 0u) || (emitter_vertex_index >= camera_path.vertices.size()) ||
      (emitter_selection_pdf <= 0.0) || (direct_area_pdf <= 0.0) || (emission_direction_pdf <= 0.0)) {
    return {};
  }
  if (emitter_vertex_index == 1u) {
    return {1.0};
  }
  const UPBPPathVertexRecord& previous = camera_path.vertices[emitter_vertex_index - 1u];
  const UPBPPathVertexRecord& emitter = camera_path.vertices[emitter_vertex_index];
  const UPBPTransportSegmentRecord& segment = camera_path.segments[emitter_vertex_index - 1u];
  const double reverse_ray_pdf = std::exp(upbp_segment_sampling_log_density(segment, previous, emitter, true));
  UPBPSurfaceMISWeights result = {1.0};
  result.add_term(false, {emitter_selection_pdf, direct_area_pdf, static_cast<double>(camera_weights.previous_delta == false), camera_weights.d_shared}, {});
  result.add_term(false, {emitter_selection_pdf, emission_direction_pdf, reverse_ray_pdf, camera_weights.d_bpt_base}, {});
  result.add_term(true, {emitter_selection_pdf, emission_direction_pdf, reverse_ray_pdf, configuration.factor(UPBPTechnique::Surface), camera_weights.d_surface}, {});
  return result;
}

inline double upbp_bpt_direct_hit_cross_technique_weight(const UPBPPathRecord& camera_path, const uint32_t emitter_vertex_index, const UPBPRecursiveVertexWeights& camera_weights,
  const double emitter_selection_pdf, const double direct_area_pdf, const double emission_direction_pdf, const UPBPDensityMISConfiguration& configuration) {
  return upbp_bpt_direct_hit_cross_technique_weights(camera_path, emitter_vertex_index, camera_weights, emitter_selection_pdf, direct_area_pdf, emission_direction_pdf,
    configuration)
    .weight(1.0);
}

struct UPBPBPTNEEMISInput {
  const Scene* scene = nullptr;
  const UPBPPathVertexRecord* camera_vertex = nullptr;
  const UPBPTransportSegmentRecord* connection_segment = nullptr;
  UPBPRecursiveVertexWeights camera_weights = {};
  UPBPScatteringEval camera_scattering = {};
  UPBPDensityMISConfiguration configuration = {};
  EmitterSample emitter_sample = {};
};

inline UPBPSurfaceMISWeights upbp_bpt_nee_cross_technique_weights(const UPBPBPTNEEMISInput& input) {
  if ((input.scene == nullptr) || (input.camera_vertex == nullptr) || (input.connection_segment == nullptr) || (input.configuration.enabled(UPBPTechnique::BPT) == false) ||
      (input.emitter_sample.pdf_sample <= 0.0f) || (input.emitter_sample.pdf_dir <= 0.0f) || (input.emitter_sample.pdf_dir_out <= 0.0f) ||
      (input.camera_weights.ray_sample_reverse_pdf_inverse <= 0.0)) {
    return {};
  }
  const UPBPPathVertexRecord& camera_vertex = *input.camera_vertex;
  const float3 direction_to_light = normalize(input.emitter_sample.origin - camera_vertex.position);
  const double camera_cosine = upbp_vertex_cosine(*input.scene, camera_vertex, direction_to_light);
  const double light_cosine = input.emitter_sample.is_distant ? 1.0 : fabs(static_cast<double>(dot(input.emitter_sample.normal, -direction_to_light)));
  if ((camera_cosine <= 0.0) || (light_cosine <= 0.0)) {
    return {};
  }

  const double direct_density = static_cast<double>(input.emitter_sample.pdf_sample) * input.emitter_sample.pdf_dir;
  const double w_light = input.emitter_sample.is_delta ? 0.0 : input.camera_scattering.pdf_forward / direct_density;
  const double camera_event_density = camera_vertex.cls == UPBPVertexClass::Medium ? std::exp(camera_vertex.log_medium_event_density) : 1.0;
  const double reverse_ray_pdf = std::exp(input.connection_segment->log_transport_pdf_reverse) * camera_event_density;
  const double sin_theta = upbp_medium_phase_sine(camera_vertex.intersection.w_i, direction_to_light);
  const double local_factor = upbp_recursive_local_volume_factor(input.configuration, camera_vertex.cls, camera_vertex.delta, camera_vertex.density_connectible,
    input.camera_weights, 1.0 / reverse_ray_pdf, camera_vertex.cls == UPBPVertexClass::Medium ? 1.0 / camera_event_density : 0.0, sin_theta, PathSource::Camera);
  const double emission_to_direct_ratio =
    static_cast<double>(input.emitter_sample.pdf_dir_out) * camera_cosine / (static_cast<double>(input.emitter_sample.pdf_dir) * light_cosine);
  UPBPSurfaceMISWeights result = {1.0 + w_light};
  upbp_bpt_add_vertex_mis_terms(result, input.configuration, camera_vertex, input.camera_weights, emission_to_direct_ratio * reverse_ray_pdf, local_factor,
    input.camera_scattering.pdf_reverse);
  return result;
}

inline double upbp_bpt_nee_cross_technique_weight(const UPBPBPTNEEMISInput& input) {
  return upbp_bpt_nee_cross_technique_weights(input).weight(1.0);
}

struct UPBPBPTLightTracingMISInput {
  const Scene* scene = nullptr;
  const UPBPPathVertexRecord* light_vertex = nullptr;
  const UPBPTransportSegmentRecord* connection_segment = nullptr;
  UPBPRecursiveVertexWeights light_weights = {};
  UPBPScatteringEval light_scattering = {};
  UPBPDensityMISConfiguration configuration = {};
  CameraSample camera_sample = {};
  uint64_t light_subpath_count = 0u;
};

inline UPBPSurfaceMISWeights upbp_bpt_light_tracing_cross_technique_weights(const UPBPBPTLightTracingMISInput& input) {
  if ((input.scene == nullptr) || (input.light_vertex == nullptr) || (input.connection_segment == nullptr) || (input.light_subpath_count == 0u) ||
      (input.configuration.enabled(UPBPTechnique::BPT) == false) || (input.camera_sample.pdf_dir_out <= 0.0f) || (input.light_weights.ray_sample_reverse_pdf_inverse <= 0.0)) {
    return {};
  }
  const UPBPPathVertexRecord& light_vertex = *input.light_vertex;
  const float3 direction_to_camera = normalize(input.camera_sample.position - light_vertex.position);
  const double distance_squared = dot(input.camera_sample.position - light_vertex.position, input.camera_sample.position - light_vertex.position);
  const double log_measure = upbp_log_measure_conversion(*input.scene, light_vertex, -direction_to_camera, distance_squared);
  const double camera_area_density = std::isfinite(log_measure) ? static_cast<double>(input.camera_sample.pdf_dir_out) * std::exp(log_measure) : 0.0;
  if (camera_area_density <= 0.0) {
    return {};
  }
  const double light_event_density = light_vertex.cls == UPBPVertexClass::Medium ? std::exp(light_vertex.log_medium_event_density) : 1.0;
  const double reverse_ray_pdf = std::exp(input.connection_segment->log_transport_pdf_reverse) * light_event_density;
  const double sin_theta = upbp_medium_phase_sine(light_vertex.intersection.w_i, direction_to_camera);
  const double local_factor = upbp_recursive_local_volume_factor(input.configuration, light_vertex.cls, light_vertex.delta, light_vertex.density_connectible, input.light_weights,
    1.0 / reverse_ray_pdf, light_vertex.cls == UPBPVertexClass::Medium ? 1.0 / light_event_density : 0.0, sin_theta, PathSource::Light);
  UPBPSurfaceMISWeights result = {1.0};
  upbp_bpt_add_vertex_mis_terms(result, input.configuration, light_vertex, input.light_weights, camera_area_density * reverse_ray_pdf, local_factor,
    input.light_scattering.pdf_reverse);
  return result;
}

inline double upbp_bpt_light_tracing_cross_technique_weight(const UPBPBPTLightTracingMISInput& input) {
  return upbp_bpt_light_tracing_cross_technique_weights(input).weight(1.0);
}

}  // namespace etx
