#pragma once

#include <etx/rt/integrators/upbp_connection.hxx>
#include <etx/rt/integrators/upbp_recursive_mis.hxx>

#include <initializer_list>

namespace etx {

inline SpectralResponse upbp_medium_scattering_coefficient(const Medium& medium, const SpectralQuery spect, const float3& position) {
  const float density = medium.cls == Medium::Homogeneous ? 1.0f : medium.sample_density_world(position);
  if ((density < 0.0f) || (density > medium_tracking_density_majorant(medium)) || (std::isfinite(density) == false)) {
    return SpectralResponse{spect, 0.0f};
  }
  return medium_scattering(medium, spect) * density;
}

inline bool upbp_remove_medium_collision_weight(const SpectralResponse& throughput, const SpectralResponse& scattering, SpectralResponse& result) {
  result = SpectralResponse{throughput.as_query(), 0.0f};
  if ((spectral_query_compatible(throughput.as_query(), scattering.as_query()) == false) || (throughput.valid() == false) || (scattering.valid() == false) ||
      (scattering.minimum() < 0.0f)) {
    return false;
  }

  // Remove scattering, retaining the inverse event density of the sampled point.
  if (throughput.spectral()) {
    if (scattering.value > 0.0f) {
      result.value = throughput.value / scattering.value;
    } else if (throughput.value != 0.0f) {
      return false;
    }
    return result.valid();
  }

  for (uint32_t component = 0u; component < 3u; ++component) {
    const float coefficient = *(&scattering.integrated.x + component);
    const float value = *(&throughput.integrated.x + component);
    if (coefficient > 0.0f) {
      *(&result.integrated.x + component) = value / coefficient;
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
  return (scattering.is_zero() == false) && upbp_remove_medium_collision_weight(vertex.throughput, scattering, result);
}

struct UPBPPointMergeMISInput {
  struct Weights {
    double d_shared = 0.0;
    double d_pde_base = 0.0;
    double d_surface = 0.0;
    double ray_sample_forward_pdf_inverse = 0.0;
    double ray_sample_reverse_pdf_inverse = 0.0;
    double ray_sample_forward_ratio = 0.0;
    bool previous_delta = false;

    Weights() = default;

    Weights(const UPBPRecursiveVertexWeights& weights)
      : d_shared(weights.d_shared)
      , d_pde_base(weights.d_pde_base)
      , d_surface(weights.d_surface)
      , ray_sample_forward_pdf_inverse(weights.ray_sample_forward_pdf_inverse)
      , ray_sample_reverse_pdf_inverse(weights.ray_sample_reverse_pdf_inverse)
      , ray_sample_forward_ratio(weights.ray_sample_forward_ratio)
      , previous_delta(weights.previous_delta) {
    }
  };

  UPBPTechnique selected_technique = UPBPTechnique::Surface;
  UPBPVertexClass vertex_class = UPBPVertexClass::Surface;
  Weights light = {};
  Weights camera = {};
  UPBPDensityMISConfiguration configuration = {};
  double scattering_pdf_forward = 0.0;
  double scattering_pdf_reverse = 0.0;
  double sin_theta = 0.0;
  uint64_t bpt_sample_count = 0u;
};

inline UPBPPointMergeMISInput::Weights upbp_point_merge_weights(const UPBPRecursiveVertexWeights& weights) {
  return UPBPPointMergeMISInput::Weights{weights};
}

struct UPBPSurfaceMISWeights {
  double constant_denominator = 0.0;
  double surface_denominator = 0.0;
  int denominator_exponent = 0;
  bool surface_selected = false;

  void add_scaled_term(const bool surface, const std::initializer_list<double> numerators, const std::initializer_list<double> denominators) {
    double mantissa = 1.0;
    int exponent = 0;
    for (const double value : numerators) {
      if (value == 0.0) {
        return;
      }
      int value_exponent = 0;
      mantissa *= std::frexp(value, &value_exponent);
      exponent += value_exponent;
    }
    for (const double value : denominators) {
      int value_exponent = 0;
      mantissa /= std::frexp(value, &value_exponent);
      exponent -= value_exponent;
    }
    if ((constant_denominator == 0.0) && (surface_denominator == 0.0)) {
      denominator_exponent = exponent;
    } else if (exponent > denominator_exponent) {
      constant_denominator = std::ldexp(constant_denominator, denominator_exponent - exponent);
      surface_denominator = std::ldexp(surface_denominator, denominator_exponent - exponent);
      denominator_exponent = exponent;
    }
    double& coefficient = surface ? surface_denominator : constant_denominator;
    coefficient += std::ldexp(mantissa, exponent - denominator_exponent);
  }

  void add_term(const bool surface, const std::initializer_list<double> numerators, const std::initializer_list<double> denominators) {
    if (denominator_exponent == 0) {
      double value = 1.0;
      for (const double factor : numerators) {
        if (factor == 0.0) {
          return;
        }
        value *= factor;
        if (std::isnormal(value) == false) {
          add_scaled_term(surface, numerators, denominators);
          return;
        }
      }
      for (const double factor : denominators) {
        value /= factor;
        if (std::isnormal(value) == false) {
          add_scaled_term(surface, numerators, denominators);
          return;
        }
      }
      double& coefficient = surface ? surface_denominator : constant_denominator;
      const double sum = coefficient + value;
      if ((value > 0.0) && std::isfinite(sum)) {
        coefficient = sum;
        return;
      }
    }
    add_scaled_term(surface, numerators, denominators);
  }

  double weight(const double relative_surface_factor) const {
    if (relative_surface_factor <= 0.0) {
      return 0.0;
    }
    const double denominator =
      surface_selected ? constant_denominator / relative_surface_factor + surface_denominator : constant_denominator + relative_surface_factor * surface_denominator;
    if ((denominator <= 0.0) || (std::isfinite(denominator) == false)) {
      return 0.0;
    }
    const double result = 1.0 / denominator;
    return denominator_exponent == 0 ? result : std::ldexp(result, -denominator_exponent);
  }
};

inline UPBPSurfaceMISWeights upbp_point_merge_mis_weights(const UPBPPointMergeMISInput& input) {
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
    return {};
  }

  const double inverse_selected_factor = 1.0 / selected_factor;
  const double light_bpt_applicable = input.light.previous_delta ? 0.0 : 1.0;
  const double camera_bpt_applicable = input.camera.previous_delta ? 0.0 : 1.0;
  const double w_light_constant = inverse_selected_factor * (input.light.d_shared * light_bpt_applicable * static_cast<double>(input.bpt_sample_count) +
                                                              input.scattering_pdf_forward * input.light.d_pde_base / input.light.ray_sample_reverse_pdf_inverse);
  const double w_camera_constant = inverse_selected_factor * (input.camera.d_shared * camera_bpt_applicable * static_cast<double>(input.bpt_sample_count) +
                                                               input.scattering_pdf_reverse * input.camera.d_pde_base / input.camera.ray_sample_reverse_pdf_inverse);
  const double surface_factor = input.configuration.factor(UPBPTechnique::Surface);
  const double w_light_surface = inverse_selected_factor * (input.scattering_pdf_forward * (surface_factor * input.light.d_surface) / input.light.ray_sample_reverse_pdf_inverse);
  const double w_camera_surface =
    inverse_selected_factor * (input.scattering_pdf_reverse * (surface_factor * input.camera.d_surface) / input.camera.ray_sample_reverse_pdf_inverse);

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

  const bool surface_selected = input.selected_technique == UPBPTechnique::Surface;
  UPBPSurfaceMISWeights result = {
    w_light_constant + (surface_selected ? 0.0 : w_local) + w_camera_constant,
    w_light_surface + (surface_selected ? w_local : 0.0) + w_camera_surface,
    0,
    surface_selected,
  };
  if (std::isfinite(result.constant_denominator) && std::isfinite(result.surface_denominator)) {
    return result;
  }

  // Retain coefficients whose normalized magnitude exceeds double range at the reference radius.
  result = {0.0, 0.0, 0, surface_selected};
  result.add_scaled_term(surface_selected, {1.0}, {});
  result.add_scaled_term(false, {input.light.d_shared, light_bpt_applicable, static_cast<double>(input.bpt_sample_count)}, {selected_factor});
  result.add_scaled_term(false, {input.camera.d_shared, camera_bpt_applicable, static_cast<double>(input.bpt_sample_count)}, {selected_factor});
  result.add_scaled_term(false, {input.scattering_pdf_forward, input.light.d_pde_base}, {input.light.ray_sample_reverse_pdf_inverse, selected_factor});
  result.add_scaled_term(false, {input.scattering_pdf_reverse, input.camera.d_pde_base}, {input.camera.ray_sample_reverse_pdf_inverse, selected_factor});
  result.add_scaled_term(true, {input.scattering_pdf_forward, surface_factor, input.light.d_surface}, {input.light.ray_sample_reverse_pdf_inverse, selected_factor});
  result.add_scaled_term(true, {input.scattering_pdf_reverse, surface_factor, input.camera.d_surface}, {input.camera.ray_sample_reverse_pdf_inverse, selected_factor});
  if (input.vertex_class == UPBPVertexClass::Medium) {
    for (const UPBPTechnique technique : {UPBPTechnique::PP3D, UPBPTechnique::PB2D, UPBPTechnique::BP2D, UPBPTechnique::BB1D}) {
      if (technique != input.selected_technique) {
        result.add_scaled_term(false, {upbp_density_strategy_factor(input.configuration, context, technique)}, {selected_factor});
      }
    }
  }
  return result;
}

inline double upbp_point_merge_mis_weight(const UPBPPointMergeMISInput& input) {
  return upbp_point_merge_mis_weights(input).weight(1.0);
}

inline bool upbp_surface_merge_compatible(const Scene& scene, const UPBPPathVertexRecord& light, const UPBPPathVertexRecord& camera, const float3& camera_geometric_normal) {
  if ((light.cls != UPBPVertexClass::Surface) || (camera.cls != UPBPVertexClass::Surface) || light.delta || camera.delta || (light.density_connectible == false) ||
      (camera.density_connectible == false)) {
    return false;
  }
  const Triangle& light_triangle = scene.triangles[light.intersection.triangle_index];
  const float3 light_geometric_normal = scene_triangle_world_geometric_normal(scene, light_triangle, light.intersection.instance_index);
  return dot(light_geometric_normal, camera_geometric_normal) > 0.0f;
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
  const uint64_t bpt_sample_count, const float3& camera_geometric_normal, Sampler& sampler, UPBPPointMergeContribution& result) {
  result = {};
  result.contribution = SpectralResponse{spect, 0.0f};
  if ((light_vertex_index == 0u) || (camera_vertex_index == 0u) || (light_vertex_index >= light_path.vertices.size()) || (camera_vertex_index >= camera_path.vertices.size()) ||
      (light_subpath_count == 0u) || (radius <= 0.0)) {
    return false;
  }
  const UPBPPathVertexRecord& light_vertex = light_path.vertices[light_vertex_index];
  const UPBPPathVertexRecord& camera_vertex = camera_path.vertices[camera_vertex_index];
  const bool surface = technique == UPBPTechnique::Surface;
  if ((surface && (upbp_surface_merge_compatible(scene, light_vertex, camera_vertex, camera_geometric_normal) == false)) ||
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
  if (surface) {
    const Material& material = scene.materials[camera_vertex.intersection.material_index];
    const BSDFData data = {spect, camera_vertex.incident_medium_index, PathSource::Camera, camera_vertex.intersection, camera_vertex.intersection.w_i};
    const BSDFEval evaluation = bsdf::evaluate(data, outgoing_direction, material, sampler);
    // Photon density already contains the incoming surface projection.
    result.scattering.value = evaluation.func;
    result.scattering.pdf_forward = evaluation.pdf;
    result.scattering.pdf_reverse = bsdf::reverse_pdf(data, outgoing_direction, material, sampler);
  } else {
    result.scattering = upbp_evaluate_vertex_scattering(scene, spect, camera_vertex, outgoing_direction, sampler);
  }
  if (result.scattering.valid() == false) {
    return true;
  }
  const double direction_cosine = static_cast<double>(dot(camera_vertex.intersection.w_i, outgoing_direction));
  const double sin_theta = std::sqrt(fmax(0.0, 1.0 - direction_cosine * direction_cosine));
  result.mis_weight = upbp_point_merge_mis_weight({
    technique,
    camera_vertex.cls,
    upbp_point_merge_weights(light_weights),
    upbp_point_merge_weights(camera_weights),
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
