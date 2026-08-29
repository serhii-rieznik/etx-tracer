#pragma once

#include "gpu_rt_wavefront_upbp_nee.hlsl"

bool upbp_vertex_is_medium(GPUUPBPVertex vertex) {
  return (vertex.flags & GPUUPBPVertexFlags::Medium) != 0u;
}

bool upbp_vertex_is_surface(GPUUPBPVertex vertex) {
  return (vertex.flags & GPUUPBPVertexFlags::Surface) != 0u;
}

bool upbp_vertex_is_delta(GPUUPBPVertex vertex) {
  return (vertex.flags & GPUUPBPVertexFlags::Delta) != 0u;
}

bool upbp_vertex_is_density_connectible(GPUUPBPVertex vertex) {
  return (vertex.flags & GPUUPBPVertexFlags::DensityConnectible) != 0u;
}

bool upbp_vertex_from_light(GPUUPBPVertex vertex) {
  return (vertex.flags & GPUUPBPVertexFlags::Camera) == 0u;
}

float upbp_vertex_cosine(GPUUPBPVertex vertex, float3 direction) {
  return upbp_vertex_is_surface(vertex) ? abs(dot(vertex.geo_normal, direction)) : 1.0f;
}

float upbp_medium_phase_sine(GPUUPBPVertex vertex, float3 direction) {
  if (upbp_vertex_is_medium(vertex) == false) {
    return 0.0f;
  }
  const float cosine = dot(vertex.w_i, direction);
  return sqrt(max(0.0f, 1.0f - cosine * cosine));
}

static const float kUPBPLogZero = -3.402823466e+38f;

bool upbp_log_is_zero(float value) {
  return value == kUPBPLogZero;
}

float upbp_log_positive(float value) {
  return value > 0.0f ? log(value) : kUPBPLogZero;
}

float upbp_log_add(float first, float second) {
  if (upbp_log_is_zero(first)) {
    return second;
  }
  if (upbp_log_is_zero(second)) {
    return first;
  }
  const float maximum = max(first, second);
  return maximum + log(1.0f + exp(min(first, second) - maximum));
}

float upbp_log_product(float first, float second) {
  return (upbp_log_is_zero(first) || upbp_log_is_zero(second)) ? kUPBPLogZero : first + second;
}

float upbp_log_multiply_positive(float log_value, float factor) {
  return (upbp_log_is_zero(log_value) || (factor <= 0.0f)) ? kUPBPLogZero : log_value + log(factor);
}

float upbp_log_short_beam_ray_factor(GPUUPBPVertex vertex) {
  return upbp_vertex_is_medium(vertex) ? -vertex.log_medium_event_density : kUPBPLogZero;
}

void upbp_reset_recursive_logs(inout GPUUPBPRecursiveState state) {
  state.weights.log_d_shared = kUPBPLogZero;
  state.weights.log_d_bpt = kUPBPLogZero;
  state.weights.log_d_pde = kUPBPLogZero;
  state.weights.log_ray_sample_forward_pdf_inverse = kUPBPLogZero;
  state.weights.log_ray_sample_reverse_pdf_inverse = kUPBPLogZero;
  state.weights.log_ray_sample_forward_ratio = kUPBPLogZero;
  state.weights.log_ray_sample_reverse_ratio = kUPBPLogZero;
  state.log_d_bpt_a = kUPBPLogZero;
  state.log_d_bpt_b = kUPBPLogZero;
  state.log_d_pde_a = kUPBPLogZero;
  state.log_d_pde_b = kUPBPLogZero;
}

float upbp_log_density_competitor_factor(GPUUPBPIteration iteration, uint technique_flag, GPUUPBPVertex vertex, float log_forward_ray_factor, float log_reverse_ray_factor,
  float sin_theta) {
  if (((iteration.technique_mask & technique_flag) == 0u) || upbp_vertex_is_delta(vertex) || (upbp_vertex_is_density_connectible(vertex) == false)) {
    return kUPBPLogZero;
  }

  float density_factor = 0.0f;
  if (technique_flag == GPUUPBPTechnique::Surface) {
    density_factor = iteration.technique_factors[1u];
    return upbp_vertex_is_surface(vertex) ? upbp_log_positive(density_factor) : kUPBPLogZero;
  }
  if (upbp_vertex_is_medium(vertex) == false) {
    return kUPBPLogZero;
  }
  if (technique_flag == GPUUPBPTechnique::PP3D) {
    return upbp_log_positive(iteration.technique_factors[2u]);
  }
  if (technique_flag == GPUUPBPTechnique::PB2D) {
    density_factor = iteration.technique_factors[3u];
    return upbp_log_product(upbp_log_positive(density_factor), log_reverse_ray_factor);
  }
  if (technique_flag == GPUUPBPTechnique::BP2D) {
    density_factor = iteration.technique_factors[4u];
    return upbp_log_product(upbp_log_positive(density_factor), log_forward_ray_factor);
  }
  if (technique_flag == GPUUPBPTechnique::BB1D) {
    density_factor = iteration.technique_factors[5u];
    if ((upbp_log_is_zero(log_forward_ray_factor) == false) && (upbp_log_is_zero(log_reverse_ray_factor) == false) && (sin_theta > 0.0f)) {
      return upbp_log_positive(density_factor) + log(sin_theta) + log_forward_ray_factor + log_reverse_ray_factor;
    }
    return kUPBPLogZero;
  }
  return kUPBPLogZero;
}

float upbp_log_recursive_local_pde_factor(GPUUPBPIteration iteration, GPUUPBPVertex vertex, GPUUPBPRecursiveWeights weights, float log_next_reverse_pdf_inverse,
  float log_next_reverse_ratio, float sin_theta) {
  const bool from_light = upbp_vertex_from_light(vertex);
  // UPBP uses short photon beams and long camera beams.
  const float log_forward_ray_factor = from_light ? weights.log_ray_sample_forward_ratio : log_next_reverse_pdf_inverse;
  const float log_reverse_ray_factor = from_light ? log_next_reverse_pdf_inverse : weights.log_ray_sample_forward_pdf_inverse;
  float result = kUPBPLogZero;
  result = upbp_log_add(result, upbp_log_density_competitor_factor(iteration, GPUUPBPTechnique::Surface, vertex, log_forward_ray_factor, log_reverse_ray_factor, sin_theta));
  result = upbp_log_add(result, upbp_log_density_competitor_factor(iteration, GPUUPBPTechnique::PP3D, vertex, log_forward_ray_factor, log_reverse_ray_factor, sin_theta));
  result = upbp_log_add(result, upbp_log_density_competitor_factor(iteration, GPUUPBPTechnique::PB2D, vertex, log_forward_ray_factor, log_reverse_ray_factor, sin_theta));
  result = upbp_log_add(result, upbp_log_density_competitor_factor(iteration, GPUUPBPTechnique::BP2D, vertex, log_forward_ray_factor, log_reverse_ray_factor, sin_theta));
  return upbp_log_add(result, upbp_log_density_competitor_factor(iteration, GPUUPBPTechnique::BB1D, vertex, log_forward_ray_factor, log_reverse_ray_factor, sin_theta));
}

bool upbp_recursive_weights_finite(GPUUPBPRecursiveWeights weights) {
  return isfinite(weights.log_d_shared) && isfinite(weights.log_d_bpt) && isfinite(weights.log_d_pde) && isfinite(weights.log_ray_sample_forward_pdf_inverse) &&
         isfinite(weights.log_ray_sample_reverse_pdf_inverse) && isfinite(weights.log_ray_sample_forward_ratio) && isfinite(weights.log_ray_sample_reverse_ratio);
}

bool upbp_initialize_recursive_state(GPUUPBPVertex endpoint, GPUUPBPIteration iteration, out GPUUPBPRecursiveState state) {
  state = (GPUUPBPRecursiveState)0;
  upbp_reset_recursive_logs(state);
  const bool camera = (endpoint.flags & GPUUPBPVertexFlags::Camera) != 0u;
  if (camera) {
    if (endpoint.endpoint_pdf_direction <= 0.0f) {
      state.failure = GPUUPBPRecursiveFailure::InvalidEndpointDensity;
      return false;
    }
    state.weights.log_d_shared = -log(endpoint.endpoint_pdf_direction);
    return true;
  }

  if ((endpoint.endpoint_pdf_area <= 0.0f) || (endpoint.endpoint_pdf_sample <= 0.0f) || (endpoint.endpoint_pdf_direction <= 0.0f)) {
    state.failure = GPUUPBPRecursiveFailure::InvalidEndpointDensity;
    return false;
  }
  const float log_emission_density = log(endpoint.endpoint_pdf_area) + log(endpoint.endpoint_pdf_sample) + log(endpoint.endpoint_pdf_direction);
  state.weights.log_d_shared = ((endpoint.flags & GPUUPBPVertexFlags::DistantEndpoint) != 0u) ? -log(endpoint.endpoint_pdf_area) : -log(endpoint.endpoint_pdf_direction);
  if (upbp_vertex_is_delta(endpoint) == false) {
    const float cosine = ((endpoint.flags & GPUUPBPVertexFlags::DistantEndpoint) != 0u) ? 1.0f : abs(dot(endpoint.normal, endpoint.sampled_direction));
    state.weights.log_d_bpt = upbp_log_positive(cosine) - log_emission_density;
  }
  state.weights.log_d_pde = upbp_log_product(state.weights.log_d_bpt, upbp_log_positive(iteration.bpt_sample_count));
  if (upbp_recursive_weights_finite(state.weights) == false) {
    state.failure = GPUUPBPRecursiveFailure::NonFiniteArrival;
    return false;
  }
  return true;
}

float upbp_segment_sampling_log_density(GPUUPBPSegment segment, GPUUPBPVertex terminal, bool reverse) {
  const float transport_density = reverse ? segment.log_transport_pdf_reverse : segment.log_transport_pdf_forward;
  return transport_density + (upbp_vertex_is_medium(terminal) ? terminal.log_medium_event_density : 0.0f);
}

bool upbp_complete_recursive_arrival(GPUUPBPVertex source, GPUUPBPVertex target, GPUUPBPSegment segment, GPUUPBPIteration iteration, uint vertex_index,
  inout GPUUPBPRecursiveState state) {
  const float log_forward_pdf = upbp_segment_sampling_log_density(segment, target, false);
  const float log_reverse_pdf = upbp_segment_sampling_log_density(segment, source, true);
  if ((isfinite(log_forward_pdf) == false) || (isfinite(log_reverse_pdf) == false)) {
    state.failure = GPUUPBPRecursiveFailure::InvalidSegmentDensity;
    state.failure_vertex_index = vertex_index;
    return false;
  }

  if (vertex_index > 1u) {
    const float log_local_factor =
      upbp_log_recursive_local_pde_factor(iteration, source, state.weights, -log_reverse_pdf, upbp_log_short_beam_ray_factor(source), state.last_sin_theta);
    state.weights.log_d_bpt = upbp_log_add(upbp_log_product(state.log_d_bpt_a, log_local_factor), state.log_d_bpt_b);
    state.weights.log_d_pde = upbp_log_add(upbp_log_product(state.log_d_pde_a, log_local_factor), state.log_d_pde_b);
  }
  state.weights.log_d_shared -= log_forward_pdf;
  state.weights.log_d_bpt = upbp_log_is_zero(state.weights.log_d_bpt) ? kUPBPLogZero : state.weights.log_d_bpt - log_forward_pdf;
  state.weights.log_d_pde = upbp_log_is_zero(state.weights.log_d_pde) ? kUPBPLogZero : state.weights.log_d_pde - log_forward_pdf;

  const float3 edge_direction = normalize(target.position - source.position);
  const float cosine = upbp_vertex_cosine(target, edge_direction);
  if ((cosine < kEpsilon) || (isfinite(cosine) == false)) {
    state.failure = GPUUPBPRecursiveFailure::InvalidMeasureCosine;
    state.failure_vertex_index = vertex_index;
    return false;
  }
  if ((vertex_index > 1u) || ((source.flags & GPUUPBPVertexFlags::DistantEndpoint) == 0u)) {
    const float log_distance = upbp_log_positive(segment.distance);
    state.weights.log_d_shared = upbp_log_product(state.weights.log_d_shared, upbp_log_product(log_distance, log_distance));
  }
  const float log_cosine = log(cosine);
  state.weights.log_d_shared -= log_cosine;
  state.weights.log_d_bpt = upbp_log_is_zero(state.weights.log_d_bpt) ? kUPBPLogZero : state.weights.log_d_bpt - log_cosine;
  state.weights.log_d_pde = upbp_log_is_zero(state.weights.log_d_pde) ? kUPBPLogZero : state.weights.log_d_pde - log_cosine;
  state.weights.log_ray_sample_forward_pdf_inverse = -log_forward_pdf;
  state.weights.log_ray_sample_reverse_pdf_inverse = -log_reverse_pdf;
  state.weights.log_ray_sample_forward_ratio = upbp_log_short_beam_ray_factor(target);
  state.weights.log_ray_sample_reverse_ratio = upbp_log_short_beam_ray_factor(source);
  if (upbp_recursive_weights_finite(state.weights) == false) {
    state.failure = GPUUPBPRecursiveFailure::NonFiniteArrival;
    state.failure_vertex_index = vertex_index;
    return false;
  }
  state.failure = GPUUPBPRecursiveFailure::None;
  state.failure_vertex_index = 0u;
  return true;
}

bool upbp_prepare_recursive_departure(GPUUPBPVertex vertex, GPUUPBPIteration iteration, uint vertex_index, inout GPUUPBPRecursiveState state) {
  const float forward_pdf = vertex.scatter_pdf_forward;
  const float reverse_pdf = vertex.scatter_pdf_reverse;
  if ((forward_pdf <= 0.0f) || (isfinite(forward_pdf) == false) || (reverse_pdf < 0.0f) || (isfinite(reverse_pdf) == false)) {
    state.failure = GPUUPBPRecursiveFailure::InvalidScatteringDensity;
    state.failure_vertex_index = vertex_index;
    return false;
  }
  const float cosine = upbp_vertex_cosine(vertex, vertex.sampled_direction);
  if ((cosine < kEpsilon) || (isfinite(cosine) == false)) {
    state.failure = GPUUPBPRecursiveFailure::InvalidMeasureCosine;
    state.failure_vertex_index = vertex_index;
    return false;
  }
  const bool previous_delta = (state.weights.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u;
  const bool bpt_previous = (previous_delta == false) && (upbp_vertex_is_delta(vertex) == false);
  const float log_forward_pdf = log(forward_pdf);
  const float log_reverse_pdf = upbp_log_positive(reverse_pdf);
  const float log_cosine = log(cosine);
  state.log_d_bpt_a = log_cosine - log_forward_pdf;
  state.log_d_pde_a = state.log_d_bpt_a;
  if (upbp_vertex_is_delta(vertex)) {
    state.log_d_bpt_b = upbp_log_is_zero(state.weights.log_d_bpt) ? kUPBPLogZero : log_cosine + state.weights.log_d_bpt - state.weights.log_ray_sample_reverse_pdf_inverse;
    state.log_d_pde_b = upbp_log_is_zero(state.weights.log_d_pde) ? kUPBPLogZero : log_cosine + state.weights.log_d_pde - state.weights.log_ray_sample_reverse_pdf_inverse;
  } else {
    const float log_bpt_shared = bpt_previous ? state.weights.log_d_shared : kUPBPLogZero;
    const float log_bpt_reverse = (upbp_log_is_zero(log_reverse_pdf) || upbp_log_is_zero(state.weights.log_d_bpt))
                                    ? kUPBPLogZero
                                    : log_reverse_pdf + state.weights.log_d_bpt - state.weights.log_ray_sample_reverse_pdf_inverse;
    state.log_d_bpt_b = upbp_log_product(state.log_d_bpt_a, upbp_log_add(log_bpt_shared, log_bpt_reverse));
    const float log_pde_shared = bpt_previous ? upbp_log_product(state.weights.log_d_shared, upbp_log_positive(iteration.bpt_sample_count)) : kUPBPLogZero;
    const float log_pde_reverse = (upbp_log_is_zero(log_reverse_pdf) || upbp_log_is_zero(state.weights.log_d_pde))
                                    ? kUPBPLogZero
                                    : log_reverse_pdf + state.weights.log_d_pde - state.weights.log_ray_sample_reverse_pdf_inverse;
    state.log_d_pde_b = upbp_log_product(state.log_d_pde_a, upbp_log_add(log_pde_shared, log_pde_reverse));
  }
  state.weights.log_d_shared = -log_forward_pdf;
  state.weights.flags = upbp_vertex_is_medium(vertex) ? GPUUPBPRecursiveWeightFlags::PreviousInMedium : 0u;
  state.weights.flags |= upbp_vertex_is_delta(vertex) ? GPUUPBPRecursiveWeightFlags::PreviousDelta : 0u;
  const float direction_cosine = dot(vertex.w_i, vertex.sampled_direction);
  state.last_sin_theta = sqrt(max(0.0f, 1.0f - direction_cosine * direction_cosine));
  const bool valid = isfinite(state.log_d_bpt_a) && isfinite(state.log_d_bpt_b) && isfinite(state.log_d_pde_a) && isfinite(state.log_d_pde_b);
  state.failure = valid ? GPUUPBPRecursiveFailure::None : GPUUPBPRecursiveFailure::NonFiniteDeparture;
  state.failure_vertex_index = valid ? 0u : vertex_index;
  return valid;
}

float upbp_bpt_nee_cross_technique_weight(GPUUPBPIteration iteration, GPUUPBPVertex camera_vertex, float3 direction_to_light, float w_light, float emission_to_direct_ratio,
  float scattering_pdf_reverse, float connection_log_transport_pdf_reverse) {
  if ((iteration.technique_mask & GPUUPBPTechnique::BPT) == 0u) {
    return 0.0f;
  }
  if ((iteration.flags & GPUUPBPIterationFlags::MultipleImportanceSampling) == 0u) {
    return 1.0f;
  }
  if ((w_light < 0.0f) || (emission_to_direct_ratio <= 0.0f) || (scattering_pdf_reverse < 0.0f)) {
    return 0.0f;
  }
  const float log_w_light = upbp_log_positive(w_light);
  const float log_camera_event_density = upbp_vertex_is_medium(camera_vertex) ? camera_vertex.log_medium_event_density : 0.0f;
  const float log_reverse_ray_pdf = connection_log_transport_pdf_reverse + log_camera_event_density;
  if (isfinite(log_reverse_ray_pdf) == false) {
    return 0.0f;
  }
  const float sin_theta = upbp_medium_phase_sine(camera_vertex, direction_to_light);
  const float log_local_factor = upbp_log_recursive_local_pde_factor(iteration, camera_vertex, camera_vertex.arrival_weights, -log_reverse_ray_pdf,
    upbp_vertex_is_medium(camera_vertex) ? -log_camera_event_density : kUPBPLogZero, sin_theta);
  const bool previous_delta = (camera_vertex.arrival_weights.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u;
  float log_inner = log_local_factor;
  log_inner = upbp_log_add(log_inner, previous_delta ? kUPBPLogZero : camera_vertex.arrival_weights.log_d_shared);
  const float log_reverse_term = (scattering_pdf_reverse > 0.0f) && (upbp_log_is_zero(camera_vertex.arrival_weights.log_d_bpt) == false)
                                   ? log(scattering_pdf_reverse) + camera_vertex.arrival_weights.log_d_bpt - camera_vertex.arrival_weights.log_ray_sample_reverse_pdf_inverse
                                   : kUPBPLogZero;
  log_inner = upbp_log_add(log_inner, log_reverse_term);
  const float log_w_camera = upbp_log_is_zero(log_inner) ? kUPBPLogZero : log(emission_to_direct_ratio) + log_reverse_ray_pdf + log_inner;
  const float log_denominator = upbp_log_add(0.0f, upbp_log_add(log_w_light, log_w_camera));
  return isfinite(log_denominator) ? exp(-log_denominator) : 0.0f;
}

float upbp_bpt_light_tracing_cross_technique_weight(GPUUPBPIteration iteration, GPUUPBPVertex light_vertex, float3 direction_to_camera, float camera_area_density,
  float scattering_pdf_reverse, float connection_log_transport_pdf_reverse) {
  if ((iteration.technique_mask & GPUUPBPTechnique::BPT) == 0u) {
    return 0.0f;
  }
  if ((iteration.flags & GPUUPBPIterationFlags::MultipleImportanceSampling) == 0u) {
    return 1.0f;
  }
  if ((camera_area_density <= 0.0f) || (scattering_pdf_reverse < 0.0f)) {
    return 0.0f;
  }
  const float log_light_event_density = upbp_vertex_is_medium(light_vertex) ? light_vertex.log_medium_event_density : 0.0f;
  const float log_reverse_ray_pdf = connection_log_transport_pdf_reverse + log_light_event_density;
  if (isfinite(log_reverse_ray_pdf) == false) {
    return 0.0f;
  }
  const float sin_theta = upbp_medium_phase_sine(light_vertex, direction_to_camera);
  const float log_local_factor = upbp_log_recursive_local_pde_factor(iteration, light_vertex, light_vertex.arrival_weights, -log_reverse_ray_pdf,
    upbp_vertex_is_medium(light_vertex) ? -log_light_event_density : kUPBPLogZero, sin_theta);
  const bool previous_delta = (light_vertex.arrival_weights.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u;
  float log_inner = log_local_factor;
  log_inner = upbp_log_add(log_inner, previous_delta ? kUPBPLogZero : light_vertex.arrival_weights.log_d_shared);
  const float log_reverse_term = (scattering_pdf_reverse > 0.0f) && (upbp_log_is_zero(light_vertex.arrival_weights.log_d_bpt) == false)
                                   ? log(scattering_pdf_reverse) + light_vertex.arrival_weights.log_d_bpt - light_vertex.arrival_weights.log_ray_sample_reverse_pdf_inverse
                                   : kUPBPLogZero;
  log_inner = upbp_log_add(log_inner, log_reverse_term);
  const float log_w_light = upbp_log_is_zero(log_inner) ? kUPBPLogZero : log(camera_area_density) + log_reverse_ray_pdf + log_inner;
  const float log_denominator = upbp_log_add(0.0f, log_w_light);
  return isfinite(log_denominator) ? exp(-log_denominator) : 0.0f;
}

float upbp_log_connection_sampling_density(GPUUPBPVertex source, GPUUPBPVertex target, float direction_pdf, float transport_log_density) {
  if (direction_pdf <= 0.0f) {
    return kUPBPLogZero;
  }
  const float3 delta = target.position - source.position;
  const float distance_squared = dot(delta, delta);
  if ((distance_squared <= 0.0f) || (isfinite(distance_squared) == false)) {
    return kUPBPLogZero;
  }
  const float3 direction = delta * rsqrt(distance_squared);
  const float measure = upbp_vertex_is_surface(target) ? abs(dot(target.geo_normal, direction)) / distance_squared : rcp(distance_squared);
  if (measure <= 0.0f) {
    return kUPBPLogZero;
  }
  const float event_log_density = upbp_vertex_is_medium(target) ? target.log_medium_event_density : 0.0f;
  const float result = log(direction_pdf) + transport_log_density + event_log_density + log(measure);
  return isfinite(result) ? result : kUPBPLogZero;
}

float upbp_bpt_connection_cross_technique_weight(GPUUPBPIteration iteration, GPUUPBPVertex light_vertex, GPUUPBPVertex camera_vertex, float light_scattering_pdf_forward,
  float light_scattering_pdf_reverse, float camera_scattering_pdf_forward, float camera_scattering_pdf_reverse, float connection_log_transport_pdf_forward,
  float connection_log_transport_pdf_reverse) {
  if ((iteration.technique_mask & GPUUPBPTechnique::BPT) == 0u) {
    return 0.0f;
  }
  if ((iteration.flags & GPUUPBPIterationFlags::MultipleImportanceSampling) == 0u) {
    return 1.0f;
  }
  if ((light_scattering_pdf_forward <= 0.0f) || (light_scattering_pdf_reverse < 0.0f) || (camera_scattering_pdf_forward <= 0.0f) || (camera_scattering_pdf_reverse < 0.0f)) {
    return 0.0f;
  }

  const float log_light_event_density = upbp_vertex_is_medium(light_vertex) ? light_vertex.log_medium_event_density : 0.0f;
  const float log_camera_event_density = upbp_vertex_is_medium(camera_vertex) ? camera_vertex.log_medium_event_density : 0.0f;
  const float log_light_reverse_ray_pdf = connection_log_transport_pdf_reverse + log_light_event_density;
  const float log_camera_reverse_ray_pdf = connection_log_transport_pdf_forward + log_camera_event_density;
  if ((isfinite(log_light_reverse_ray_pdf) == false) || (isfinite(log_camera_reverse_ray_pdf) == false)) {
    return 0.0f;
  }

  const float3 light_to_camera_direction = normalize(camera_vertex.position - light_vertex.position);
  const float log_light_local_factor = upbp_log_recursive_local_pde_factor(iteration, light_vertex, light_vertex.arrival_weights, -log_light_reverse_ray_pdf,
    upbp_vertex_is_medium(light_vertex) ? -log_light_event_density : kUPBPLogZero, upbp_medium_phase_sine(light_vertex, light_to_camera_direction));
  const float log_camera_local_factor = upbp_log_recursive_local_pde_factor(iteration, camera_vertex, camera_vertex.arrival_weights, -log_camera_reverse_ray_pdf,
    upbp_vertex_is_medium(camera_vertex) ? -log_camera_event_density : kUPBPLogZero, upbp_medium_phase_sine(camera_vertex, -light_to_camera_direction));

  const float log_camera_to_light_density = upbp_log_connection_sampling_density(camera_vertex, light_vertex, camera_scattering_pdf_forward, connection_log_transport_pdf_reverse);
  const float log_light_to_camera_density = upbp_log_connection_sampling_density(light_vertex, camera_vertex, light_scattering_pdf_forward, connection_log_transport_pdf_forward);
  const bool light_previous_delta = (light_vertex.arrival_weights.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u;
  const bool camera_previous_delta = (camera_vertex.arrival_weights.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u;
  float log_light_inner = upbp_log_add(log_light_local_factor, light_previous_delta ? kUPBPLogZero : light_vertex.arrival_weights.log_d_shared);
  const float log_light_reverse_term =
    (light_scattering_pdf_reverse > 0.0f) && (upbp_log_is_zero(light_vertex.arrival_weights.log_d_bpt) == false)
      ? log(light_scattering_pdf_reverse) + light_vertex.arrival_weights.log_d_bpt - light_vertex.arrival_weights.log_ray_sample_reverse_pdf_inverse
      : kUPBPLogZero;
  log_light_inner = upbp_log_add(log_light_inner, log_light_reverse_term);
  float log_camera_inner = upbp_log_add(log_camera_local_factor, camera_previous_delta ? kUPBPLogZero : camera_vertex.arrival_weights.log_d_shared);
  const float log_camera_reverse_term =
    (camera_scattering_pdf_reverse > 0.0f) && (upbp_log_is_zero(camera_vertex.arrival_weights.log_d_bpt) == false)
      ? log(camera_scattering_pdf_reverse) + camera_vertex.arrival_weights.log_d_bpt - camera_vertex.arrival_weights.log_ray_sample_reverse_pdf_inverse
      : kUPBPLogZero;
  log_camera_inner = upbp_log_add(log_camera_inner, log_camera_reverse_term);
  const float log_w_light = upbp_log_product(log_camera_to_light_density, log_light_inner);
  const float log_w_camera = upbp_log_product(log_light_to_camera_density, log_camera_inner);
  const float log_denominator = upbp_log_add(0.0f, upbp_log_add(log_w_light, log_w_camera));
  return isfinite(log_denominator) ? exp(-log_denominator) : 0.0f;
}
