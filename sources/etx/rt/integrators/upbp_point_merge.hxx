#pragma once

#include <etx/rt/integrators/upbp_connection.hxx>
#include <etx/rt/integrators/upbp_recursive_mis.hxx>

namespace etx {

inline SpectralResponse upbp_medium_scattering_coefficient(const Medium& medium, const SpectralQuery spect, const float3& position) {
  const float density = medium.cls == Medium::Homogeneous ? 1.0f : medium.sample_density_world(position);
  if ((density < 0.0f) || (density > medium_tracking_density_majorant(medium)) || (std::isfinite(density) == false)) {
    return SpectralResponse{spect, 0.0f};
  }
  return medium_scattering(medium, spect) * density;
}

inline bool upbp_remove_medium_collision_weight(const SpectralResponse& throughput, const SpectralResponse& scattering, const double real_event_density, SpectralResponse& result) {
  result = SpectralResponse{throughput.as_query(), 0.0f};
  if ((spectral_query_compatible(throughput.as_query(), scattering.as_query()) == false) || (throughput.valid() == false) || (scattering.valid() == false) ||
      (scattering.minimum() < 0.0f) || (real_event_density <= 0.0) || (std::isfinite(real_event_density) == false)) {
    return false;
  }

  const float density = static_cast<float>(real_event_density);
  if (throughput.spectral()) {
    if (scattering.value > 0.0f) {
      result.value = throughput.value * density / scattering.value;
    } else if (throughput.value != 0.0f) {
      return false;
    }
    return result.valid();
  }

  for (uint32_t component = 0u; component < 3u; ++component) {
    const float coefficient = *(&scattering.integrated.x + component);
    const float value = *(&throughput.integrated.x + component);
    if (coefficient > 0.0f) {
      *(&result.integrated.x + component) = value * density / coefficient;
    } else if (value != 0.0f) {
      return false;
    }
  }
  return result.valid();
}

inline bool upbp_medium_pre_collision_throughput(const Scene& scene, const SpectralQuery spect, const UPBPPathVertexRecord& vertex, SpectralResponse& result) {
  if ((vertex.cls != UPBPVertexClass::Medium) || (vertex.medium.index >= scene.mediums.count)) {
    return false;
  }
  const SpectralResponse scattering = upbp_medium_scattering_coefficient(scene.mediums[vertex.medium.index], spect, vertex.position);
  const double real_event_density = std::exp(vertex.log_medium_event_density);
  return (scattering.is_zero() == false) && upbp_remove_medium_collision_weight(vertex.throughput, scattering, real_event_density, result);
}

struct UPBPPointMergeMISInput {
  UPBPTechnique selected_technique = UPBPTechnique::Surface;
  UPBPVertexClass vertex_class = UPBPVertexClass::Surface;
  UPBPRecursiveVertexWeights light = {};
  UPBPRecursiveVertexWeights camera = {};
  UPBPDensityMISConfiguration configuration = {};
  double scattering_pdf_forward = 0.0;
  double scattering_pdf_reverse = 0.0;
  double sin_theta = 0.0;
  uint64_t bpt_sample_count = 0u;
};

inline double upbp_point_merge_mis_weight(const UPBPPointMergeMISInput& input) {
  const double forward_ray_factor = input.configuration.photon_beams_long ? input.light.ray_sample_forward_pdf_inverse : input.light.ray_sample_forward_ratio;
  const double reverse_ray_factor = input.configuration.camera_beams_long ? input.camera.ray_sample_forward_pdf_inverse : input.camera.ray_sample_forward_ratio;
  const UPBPDensityMISContext context = {
    2u,
    input.vertex_class,
    forward_ray_factor,
    reverse_ray_factor,
    input.sin_theta,
    false,
    true,
  };
  const double selected_factor = upbp_density_strategy_factor(input.configuration, context, input.selected_technique);
  if ((selected_factor <= 0.0) || (input.scattering_pdf_forward <= 0.0) || (input.scattering_pdf_reverse <= 0.0) || (input.light.ray_sample_reverse_pdf_inverse <= 0.0) ||
      (input.camera.ray_sample_reverse_pdf_inverse <= 0.0)) {
    return 0.0;
  }

  const double inverse_selected_factor = 1.0 / selected_factor;
  const double light_bpt_applicable = input.light.previous_delta ? 0.0 : 1.0;
  const double camera_bpt_applicable = input.camera.previous_delta ? 0.0 : 1.0;
  const double w_light = inverse_selected_factor * (input.light.d_shared * light_bpt_applicable * static_cast<double>(input.bpt_sample_count) +
                                                     input.scattering_pdf_forward * input.light.d_pde / input.light.ray_sample_reverse_pdf_inverse);
  const double w_camera = inverse_selected_factor * (input.camera.d_shared * camera_bpt_applicable * static_cast<double>(input.bpt_sample_count) +
                                                      input.scattering_pdf_reverse * input.camera.d_pde / input.camera.ray_sample_reverse_pdf_inverse);

  double w_local = 1.0;
  if (input.vertex_class == UPBPVertexClass::Medium) {
    constexpr UPBPTechnique local_techniques[] = {
      UPBPTechnique::PP3D,
      UPBPTechnique::PB2D,
      UPBPTechnique::BP2D,
      UPBPTechnique::BB1D,
    };
    for (const UPBPTechnique technique : local_techniques) {
      if (technique != input.selected_technique) {
        w_local += inverse_selected_factor * upbp_density_strategy_factor(input.configuration, context, technique);
      }
    }
  }

  const double denominator = w_light + w_local + w_camera;
  return (denominator > 0.0) && std::isfinite(denominator) ? 1.0 / denominator : 0.0;
}

inline bool upbp_surface_merge_compatible(const Scene& scene, const UPBPPathVertexRecord& light, const UPBPPathVertexRecord& camera) {
  if ((light.cls != UPBPVertexClass::Surface) || (camera.cls != UPBPVertexClass::Surface) || light.delta || camera.delta || (light.density_connectible == false) ||
      (camera.density_connectible == false)) {
    return false;
  }
  const Triangle& light_triangle = scene.triangles[light.intersection.triangle_index];
  const Triangle& camera_triangle = scene.triangles[camera.intersection.triangle_index];
  const float3 light_normal = scene_triangle_world_geometric_normal(scene, light_triangle, light.intersection.instance_index);
  const float3 camera_normal = scene_triangle_world_geometric_normal(scene, camera_triangle, camera.intersection.instance_index);
  return dot(light_normal, camera_normal) > 0.0f;
}

inline bool upbp_medium_merge_compatible(const UPBPPathVertexRecord& light, const UPBPPathVertexRecord& camera) {
  return (light.cls == UPBPVertexClass::Medium) && (camera.cls == UPBPVertexClass::Medium) && light.density_connectible && camera.density_connectible &&
         (light.medium.index != kInvalidIndex) && (light.medium.index == camera.medium.index) && (light.delta == false) && (camera.delta == false);
}

struct UPBPPointMergeContribution {
  UPBPScatteringEval scattering = {};
  SpectralResponse contribution = {};
  double kernel_value = 0.0;
  double mis_weight = 0.0;
  bool applicable = false;
};

inline bool upbp_evaluate_point_merge(const Scene& scene, const SpectralQuery spect, const UPBPPathRecord& light_path, const uint32_t light_vertex_index,
  const UPBPRecursiveVertexWeights& light_weights, const UPBPPathRecord& camera_path, const uint32_t camera_vertex_index, const UPBPRecursiveVertexWeights& camera_weights,
  const UPBPDensityMISConfiguration& configuration, const UPBPTechnique technique, const UPBPKernel kernel, const double radius, const uint64_t light_subpath_count,
  const uint64_t bpt_sample_count, Sampler& sampler, UPBPPointMergeContribution& result) {
  result = {};
  result.contribution = SpectralResponse{spect, 0.0f};
  if ((light_vertex_index == 0u) || (camera_vertex_index == 0u) || (light_vertex_index >= light_path.vertices.size()) || (camera_vertex_index >= camera_path.vertices.size()) ||
      (light_subpath_count == 0u) || (radius <= 0.0)) {
    return false;
  }
  const UPBPPathVertexRecord& light_vertex = light_path.vertices[light_vertex_index];
  const UPBPPathVertexRecord& camera_vertex = camera_path.vertices[camera_vertex_index];
  const bool surface = technique == UPBPTechnique::Surface;
  if ((surface && (upbp_surface_merge_compatible(scene, light_vertex, camera_vertex) == false)) ||
      ((technique == UPBPTechnique::PP3D) && (upbp_medium_merge_compatible(light_vertex, camera_vertex) == false)) ||
      ((technique != UPBPTechnique::Surface) && (technique != UPBPTechnique::PP3D))) {
    return true;
  }
  if (spectral_query_compatible(light_vertex.throughput.as_query(), camera_vertex.throughput.as_query()) == false) {
    return true;
  }
  const uint32_t path_length = light_vertex_index + camera_vertex_index;
  if ((path_length < scene.options.min_path_length) || (path_length > scene.options.max_path_length)) {
    return true;
  }

  const float3 delta = light_vertex.position - camera_vertex.position;
  const double distance_squared = dot(delta, delta);
  result.kernel_value = upbp_kernel_value(kernel, surface ? 2u : 3u, radius, distance_squared);
  if (result.kernel_value <= 0.0) {
    return true;
  }

  const float3 outgoing_direction = -light_vertex.intersection.w_i;
  result.scattering = upbp_evaluate_vertex_scattering(scene, spect, camera_vertex, outgoing_direction, sampler);
  if (result.scattering.valid() == false) {
    return true;
  }
  const double direction_cosine = static_cast<double>(dot(camera_vertex.intersection.w_i, outgoing_direction));
  const double sin_theta = std::sqrt(fmax(0.0, 1.0 - direction_cosine * direction_cosine));
  result.mis_weight = upbp_point_merge_mis_weight({
    technique,
    camera_vertex.cls,
    light_weights,
    camera_weights,
    configuration,
    result.scattering.pdf_forward,
    result.scattering.pdf_reverse,
    sin_theta,
    bpt_sample_count,
  });
  if (result.mis_weight <= 0.0) {
    return true;
  }

  SpectralResponse light_throughput = light_vertex.throughput;
  SpectralResponse camera_throughput = camera_vertex.throughput;
  if (surface == false) {
    if ((upbp_medium_pre_collision_throughput(scene, spect, light_vertex, light_throughput) == false) ||
        (upbp_medium_pre_collision_throughput(scene, spect, camera_vertex, camera_throughput) == false)) {
      return false;
    }
    const SpectralResponse scattering = upbp_medium_scattering_coefficient(scene.mediums[camera_vertex.medium.index], spect, camera_vertex.position);
    result.scattering.value *= scattering;
  }

  const double estimator_scale = result.kernel_value / static_cast<double>(light_subpath_count);
  result.contribution = light_throughput * camera_throughput * result.scattering.value * static_cast<float>(estimator_scale * result.mis_weight);
  result.applicable = true;
  return true;
}

}  // namespace etx
