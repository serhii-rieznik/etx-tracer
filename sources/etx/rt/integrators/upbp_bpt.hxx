#pragma once

#include <etx/rt/integrators/upbp_bpt_cross_mis.hxx>

namespace etx {

struct UPBPLightSplat {
  SpectralResponse value = {};
  float2 film_uv = {};
};

enum class UPBPLightSplatEvaluationFailure : uint8_t {
  None,
  InvalidPath,
  ConnectionEvaluation,
  RecursiveWeights,
};

inline double upbp_light_splat_iteration_scale(const uint64_t camera_subpath_count, const uint64_t light_subpath_count) {
  return light_subpath_count > 0u ? static_cast<double>(camera_subpath_count) / static_cast<double>(light_subpath_count) : 0.0;
}

inline bool upbp_path_length_enabled(const Scene& scene, const uint64_t path_length) {
  return (path_length >= scene.options.min_path_length) && (path_length <= scene.options.max_path_length);
}

inline bool upbp_all_bpt_strategies_enabled(const Scene& scene) {
  return scene.strategy_enabled(Scene::Strategy::DirectHit) && scene.strategy_enabled(Scene::Strategy::ConnectToLight) &&
         scene.strategy_enabled(Scene::Strategy::ConnectToCamera) && scene.strategy_enabled(Scene::Strategy::ConnectVertices);
}

inline bool upbp_requires_exhaustive_bpt_weight(const Scene& scene) {
  return scene.multiple_importance_sampling() && (upbp_all_bpt_strategies_enabled(scene) == false);
}

inline double upbp_selected_bpt_weight(const Scene& scene, const double recursive_weight, const double exhaustive_weight) {
  if (scene.multiple_importance_sampling() == false) {
    return 1.0;
  }
  return upbp_all_bpt_strategies_enabled(scene) ? recursive_weight : exhaustive_weight;
}

inline bool upbp_camera_prefix_is_specular(const UPBPPathRecord& camera_path, const uint32_t endpoint_index) {
  if ((endpoint_index == 0u) || (endpoint_index >= camera_path.vertices.size())) {
    return false;
  }
  for (uint32_t vertex_index = 1u; vertex_index < endpoint_index; ++vertex_index) {
    if (camera_path.vertices[vertex_index].delta == false) {
      return false;
    }
  }
  return true;
}

enum class UPBPBPTEvaluationFailure : uint8_t {
  None,
  InvalidInput,
  DirectHitEvaluation,
  DirectHitWeights,
  EmitterConnectionEvaluation,
  EmitterConnectionWeights,
  VertexConnectionEvaluation,
  VertexConnectionWeights,
  EnvironmentTerminal,
  EnvironmentWeights,
};

struct UPBPBPTCameraEvaluation {
  SpectralResponse value = {};
  uint32_t direct_hit_count = 0u;
  uint32_t emitter_connection_count = 0u;
  uint32_t vertex_connection_count = 0u;
  UPBPBPTEvaluationFailure failure = UPBPBPTEvaluationFailure::None;
  UPBPSceneSegmentFailure segment_failure = UPBPSceneSegmentFailure::None;
  MediumTrackingFailure medium_failure = MediumTrackingFailure::None;
  Intersection failure_intersection = {};
  bool valid = false;
};

inline Sampler upbp_evaluation_sampler(const uint32_t render_seed, const uint64_t iteration, const uint64_t path_index, const uint32_t camera_vertex_count,
  const uint32_t light_vertex_count, const UPBPRandomDomain domain) {
  return Sampler{upbp_sampler_seed(render_seed, iteration, path_index, camera_vertex_count, light_vertex_count, domain)};
}

inline bool upbp_evaluate_light_splats(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const UPBPPathRecord& light_path, const uint32_t render_seed,
  const uint64_t iteration, const uint64_t path_index, const uint32_t maximum_boundary_count, const uint32_t maximum_null_events_per_interval,
  const UPBPRecursivePathWeights& light_weights, const UPBPDensityMISConfiguration& configuration, const uint64_t camera_subpath_count, const uint64_t light_subpath_count,
  std::vector<UPBPLightSplat>& result, UPBPLightSplatEvaluationFailure& failure, UPBPLightToCameraFailure& connection_failure, UPBPSceneSegmentFailure& segment_failure,
  UPBPPathProbabilityFailure& probability_failure, uint32_t& failure_vertex_count) {
  result.clear();
  failure = UPBPLightSplatEvaluationFailure::None;
  connection_failure = UPBPLightToCameraFailure::None;
  segment_failure = UPBPSceneSegmentFailure::None;
  probability_failure = UPBPPathProbabilityFailure::None;
  failure_vertex_count = 0u;
  if (light_path.valid() == false) {
    failure = UPBPLightSplatEvaluationFailure::InvalidPath;
    return false;
  }

  const bool evaluate_exhaustive_weight = upbp_requires_exhaustive_bpt_weight(scene);
  result.reserve(light_path.vertices.size());
  for (uint32_t light_vertex_count = 2u; light_vertex_count <= light_path.vertices.size(); ++light_vertex_count) {
    if (upbp_path_length_enabled(scene, light_vertex_count) == false) {
      continue;
    }
    Sampler camera_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, 1u, light_vertex_count, UPBPRandomDomain::FilmConnection);
    Sampler scattering_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, 1u, light_vertex_count, UPBPRandomDomain::ScatteringEvaluation);
    Sampler intersection_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, 1u, light_vertex_count, UPBPRandomDomain::IntersectionTraversal);
    Sampler medium_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, 1u, light_vertex_count, UPBPRandomDomain::ConnectionTransmittance);
    UPBPLightToCameraContribution contribution = {};
    if (upbp_evaluate_light_to_camera(rt, scene, spect, light_path, light_vertex_count, camera_sampler, scattering_sampler, intersection_sampler, medium_sampler,
          maximum_boundary_count, maximum_null_events_per_interval, evaluate_exhaustive_weight, contribution) == false) {
      failure = UPBPLightSplatEvaluationFailure::ConnectionEvaluation;
      connection_failure = contribution.failure;
      segment_failure = contribution.transmittance.failure;
      probability_failure = contribution.probability_path.failure;
      failure_vertex_count = light_vertex_count;
      return false;
    }
    if (contribution.applicable) {
      const uint32_t light_vertex_index = light_vertex_count - 1u;
      if (light_vertex_index >= light_weights.arrivals.size()) {
        failure = UPBPLightSplatEvaluationFailure::RecursiveWeights;
        failure_vertex_count = light_vertex_count;
        return false;
      }
      const double recursive_weight = upbp_bpt_light_tracing_cross_technique_weight({
        &scene,
        &light_path.vertices[light_vertex_index],
        &contribution.transmittance.segment,
        light_weights.arrivals[light_vertex_index],
        contribution.light_scattering,
        configuration,
        contribution.camera_sample,
        light_subpath_count,
      });
      const double mis_weight = upbp_selected_bpt_weight(scene, recursive_weight, contribution.mis_weight);
      const SpectralResponse value =
        light_path.vertices[light_vertex_index].throughput * contribution.light_scattering.value * contribution.transmittance.weight *
        (contribution.camera_sample.weight * static_cast<float>(mis_weight * upbp_light_splat_iteration_scale(camera_subpath_count, light_subpath_count)));
      if (mis_weight > 0.0) {
        result.emplace_back(UPBPLightSplat{value, contribution.camera_sample.uv});
      }
    }
  }
  return true;
}

inline bool upbp_evaluate_bpt_camera_path(const Raytracing& rt, const Scene& scene, const SpectralQuery spect, const UPBPPathRecord& light_path,
  const UPBPSubpathBuildResult& camera_subpath, const uint32_t render_seed, const uint64_t iteration, const uint64_t path_index, const uint32_t maximum_boundary_count,
  const uint32_t maximum_null_events_per_interval, const UPBPRecursivePathWeights& light_weights, const UPBPRecursivePathWeights& camera_weights,
  const UPBPDensityMISConfiguration& configuration, const uint64_t light_subpath_count, UPBPBPTCameraEvaluation& result) {
  result = {};
  result.value = SpectralResponse{spect, 0.0f};
  if ((light_path.valid() == false) || (camera_subpath.valid() == false)) {
    result.failure = UPBPBPTEvaluationFailure::InvalidInput;
    return false;
  }

  const UPBPPathRecord& camera_path = camera_subpath.path;
  const bool evaluate_exhaustive_weight = upbp_requires_exhaustive_bpt_weight(scene);
  for (uint32_t camera_vertex_count = 2u; camera_vertex_count <= camera_path.vertices.size(); ++camera_vertex_count) {
    if (scene.strategy_enabled(Scene::Strategy::DirectHit) && upbp_path_length_enabled(scene, camera_vertex_count - 1u)) {
      Sampler direct_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, camera_vertex_count, 0u, UPBPRandomDomain::DirectHitEvaluation);
      UPBPDirectHitContribution direct_hit = {};
      if (upbp_evaluate_direct_area_hit(rt, scene, spect, camera_path, camera_vertex_count, direct_sampler, evaluate_exhaustive_weight, direct_hit) == false) {
        result.failure = UPBPBPTEvaluationFailure::DirectHitEvaluation;
        return false;
      }
      if (direct_hit.applicable) {
        const uint32_t emitter_vertex_index = camera_vertex_count - 1u;
        if (emitter_vertex_index >= camera_weights.arrivals.size()) {
          result.failure = UPBPBPTEvaluationFailure::DirectHitWeights;
          return false;
        }
        const UPBPPathVertexRecord& emitter_vertex = camera_path.vertices[emitter_vertex_index];
        const UPBPPathVertexRecord& previous = camera_path.vertices[emitter_vertex_index - 1u];
        const Emitter& emitter = scene.emitter_instances[emitter_vertex.intersection.emitter_index];
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
        const bool bpt_enabled = configuration.enabled(UPBPTechnique::BPT);
        const double recursive_weight = upbp_bpt_direct_hit_cross_technique_weight(camera_path, emitter_vertex_index, camera_weights.arrivals[emitter_vertex_index],
          emitter_discrete_pdf(emitter), pdf_area, pdf_dir_out, bpt_enabled);
        const double mis_weight =
          bpt_enabled ? upbp_selected_bpt_weight(scene, recursive_weight, direct_hit.mis_weight) : (upbp_camera_prefix_is_specular(camera_path, emitter_vertex_index) ? 1.0 : 0.0);
        if (mis_weight > 0.0) {
          result.value += emitter_vertex.throughput * radiance * static_cast<float>(mis_weight);
          ++result.direct_hit_count;
        }
      }
    }

    if (configuration.enabled(UPBPTechnique::BPT) == false) {
      continue;
    }

    if (scene.strategy_enabled(Scene::Strategy::ConnectToLight) && upbp_path_length_enabled(scene, camera_vertex_count)) {
      Sampler emitter_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, camera_vertex_count, 1u, UPBPRandomDomain::EmitterConnection);
      Sampler emitter_scattering_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, camera_vertex_count, 1u, UPBPRandomDomain::ScatteringEvaluation);
      Sampler emitter_intersection_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, camera_vertex_count, 1u, UPBPRandomDomain::IntersectionTraversal);
      Sampler emitter_medium_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, camera_vertex_count, 1u, UPBPRandomDomain::ConnectionTransmittance);
      UPBPNEEContribution emitter_connection = {};
      if (upbp_evaluate_nee(rt, scene, spect, camera_path, camera_vertex_count, emitter_sampler, emitter_scattering_sampler, emitter_intersection_sampler, emitter_medium_sampler,
            maximum_boundary_count, maximum_null_events_per_interval, evaluate_exhaustive_weight, emitter_connection) == false) {
        result.failure = UPBPBPTEvaluationFailure::EmitterConnectionEvaluation;
        result.segment_failure = emitter_connection.transmittance.failure;
        result.medium_failure = emitter_connection.transmittance.medium_failure;
        result.failure_intersection = emitter_connection.transmittance.failure_intersection;
        return false;
      }
      if (emitter_connection.applicable) {
        const uint32_t camera_vertex_index = camera_vertex_count - 1u;
        if (camera_vertex_index >= camera_weights.arrivals.size()) {
          result.failure = UPBPBPTEvaluationFailure::EmitterConnectionWeights;
          return false;
        }
        const UPBPPathVertexRecord& camera_vertex = camera_path.vertices[camera_vertex_index];
        const double recursive_weight = upbp_bpt_nee_cross_technique_weight({
          &scene,
          &camera_vertex,
          &emitter_connection.transmittance.segment,
          camera_weights.arrivals[camera_vertex_index],
          emitter_connection.camera_scattering,
          configuration,
          emitter_connection.emitter_sample,
        });
        const double mis_weight = upbp_selected_bpt_weight(scene, recursive_weight, emitter_connection.mis_weight);
        if (mis_weight > 0.0) {
          result.value += camera_vertex.throughput * emitter_connection.camera_scattering.value * emitter_connection.emitter_sample.value *
                          emitter_connection.transmittance.weight *
                          static_cast<float>(mis_weight / (static_cast<double>(emitter_connection.emitter_sample.pdf_dir) * emitter_connection.emitter_sample.pdf_sample));
          ++result.emitter_connection_count;
        }
      }
    }

    for (uint32_t light_vertex_count = 2u; scene.strategy_enabled(Scene::Strategy::ConnectVertices) && (light_vertex_count <= light_path.vertices.size()); ++light_vertex_count) {
      const uint64_t path_length = static_cast<uint64_t>(light_vertex_count) + static_cast<uint64_t>(camera_vertex_count) - 1u;
      if (upbp_path_length_enabled(scene, path_length) == false) {
        continue;
      }
      Sampler scattering_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, camera_vertex_count, light_vertex_count, UPBPRandomDomain::ScatteringEvaluation);
      Sampler intersection_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, camera_vertex_count, light_vertex_count, UPBPRandomDomain::IntersectionTraversal);
      Sampler medium_sampler = upbp_evaluation_sampler(render_seed, iteration, path_index, camera_vertex_count, light_vertex_count, UPBPRandomDomain::ConnectionTransmittance);
      UPBPBPTConnectionContribution vertex_connection = {};
      if (upbp_evaluate_bpt_connection(rt, scene, spect, light_path, light_vertex_count, camera_path, camera_vertex_count, scattering_sampler, intersection_sampler, medium_sampler,
            maximum_boundary_count, maximum_null_events_per_interval, evaluate_exhaustive_weight, vertex_connection) == false) {
        result.failure = UPBPBPTEvaluationFailure::VertexConnectionEvaluation;
        result.segment_failure = vertex_connection.connection.transmittance.failure;
        result.medium_failure = vertex_connection.connection.transmittance.medium_failure;
        result.failure_intersection = vertex_connection.connection.transmittance.failure_intersection;
        return false;
      }
      if (vertex_connection.applicable) {
        const uint32_t light_vertex_index = light_vertex_count - 1u;
        const uint32_t camera_vertex_index = camera_vertex_count - 1u;
        if ((light_vertex_index >= light_weights.arrivals.size()) || (camera_vertex_index >= camera_weights.arrivals.size())) {
          result.failure = UPBPBPTEvaluationFailure::VertexConnectionWeights;
          return false;
        }
        const double recursive_weight = upbp_bpt_connection_cross_technique_weight({
          &scene,
          &light_path.vertices[light_vertex_index],
          &camera_path.vertices[camera_vertex_index],
          &vertex_connection.connection.transmittance.segment,
          light_weights.arrivals[light_vertex_index],
          camera_weights.arrivals[camera_vertex_index],
          vertex_connection.connection.light_scattering,
          vertex_connection.connection.camera_scattering,
          configuration,
        });
        const double mis_weight = upbp_selected_bpt_weight(scene, recursive_weight, vertex_connection.mis_weight);
        if (mis_weight > 0.0) {
          result.value += vertex_connection.connection.contribution * static_cast<float>(mis_weight);
          ++result.vertex_connection_count;
        }
      }
    }
  }

  if ((camera_subpath.terminal == UPBPSceneSegmentTerminal::Miss) && scene.strategy_enabled(Scene::Strategy::DirectHit)) {
    const uint32_t environment_emitter_count = environment_emitter_shared_count();
    for (uint32_t local_emitter_index = 0u; local_emitter_index < environment_emitter_count; ++local_emitter_index) {
      uint32_t emitter_index = kInvalidIndex;
      if (environment_emitter_shared_try_load_index(local_emitter_index, emitter_index) == false) {
        continue;
      }
      const Emitter& emitter = scene.emitter_instances[emitter_index];
      const EmitterRadianceQuery query = {
        .direction = camera_subpath.terminal_ray.d,
        .directly_visible = camera_path.physical_length() == 0u,
      };
      float pdf_area = 0.0f;
      float pdf_dir = 0.0f;
      float pdf_dir_out = 0.0f;
      const SpectralResponse radiance = emitter_get_radiance(emitter, spect, query, pdf_area, pdf_dir, pdf_dir_out);
      if (radiance.is_zero() || (pdf_area <= 0.0f) || (pdf_dir <= 0.0f) || (pdf_dir_out <= 0.0f)) {
        continue;
      }
      UPBPPathVertexRecord endpoint = {};
      endpoint.position = camera_path.terminal_segment.intervals.back().end_position;
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
      UPBPPathRecord endpoint_path = camera_path;
      if (endpoint_path.promote_terminal_segment(endpoint) == false) {
        result.failure = UPBPBPTEvaluationFailure::EnvironmentTerminal;
        return false;
      }
      UPBPRecursivePathWeights endpoint_weights = {};
      if (upbp_compute_recursive_path_weights(scene, endpoint_path, configuration, light_subpath_count, configuration.enabled(UPBPTechnique::BPT) ? 1u : 0u, endpoint_weights) ==
          false) {
        result.failure = UPBPBPTEvaluationFailure::EnvironmentWeights;
        return false;
      }
      const uint32_t endpoint_index = static_cast<uint32_t>(endpoint_path.vertices.size() - 1u);
      const bool bpt_enabled = configuration.enabled(UPBPTechnique::BPT);
      const double recursive_weight = upbp_bpt_direct_hit_cross_technique_weight(endpoint_path, endpoint_index, endpoint_weights.arrivals[endpoint_index],
        emitter_discrete_pdf(emitter), pdf_area, pdf_dir_out, bpt_enabled);
      double exhaustive_weight = 1.0;
      if (evaluate_exhaustive_weight) {
        UPBPJoinedPath joined_path = {};
        UPBPPathProbabilityRecord probability_path = {};
        std::vector<UPBPBPTStrategyProbability> strategies = {};
        Sampler direct_sampler =
          upbp_evaluation_sampler(render_seed, iteration, path_index, static_cast<uint32_t>(endpoint_path.vertices.size()), 0u, UPBPRandomDomain::DirectHitEvaluation);
        if ((upbp_join_direct_hit(endpoint_path, static_cast<uint32_t>(endpoint_path.vertices.size()), joined_path) == false) ||
            (upbp_build_joined_path_probability(rt, scene, spect, joined_path, direct_sampler, probability_path) == false) ||
            (upbp_enumerate_bpt_strategies_recursive(probability_path, strategies) == false)) {
          result.failure = UPBPBPTEvaluationFailure::EnvironmentWeights;
          return false;
        }
        upbp_apply_bpt_strategy_constraints(scene, static_cast<uint32_t>(joined_path.vertices.size()), strategies);
        exhaustive_weight = upbp_bpt_balance_weight(strategies, 0u);
      }
      const double mis_weight =
        bpt_enabled ? upbp_selected_bpt_weight(scene, recursive_weight, exhaustive_weight) : (upbp_camera_prefix_is_specular(endpoint_path, endpoint_index) ? 1.0 : 0.0);
      if (mis_weight > 0.0) {
        result.value += endpoint.throughput * radiance * static_cast<float>(mis_weight);
        ++result.direct_hit_count;
      }
    }
  }

  result.valid = true;
  return true;
}

}  // namespace etx
