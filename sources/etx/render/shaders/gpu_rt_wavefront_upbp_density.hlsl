#include "gpu_rt_wavefront_common.hlsl"

struct UPBPGPUBeamTransportPrefix {
  SpectralResponse weight;
  float log_transport_pdf_forward;
  float log_transport_pdf_reverse;
  float distance;
};

struct UPBPGPUPreparedBeam {
  SpectralResponse source_throughput;
  UPBPGPUBeamTransportPrefix transport_at_origin;
  float log_d_shared;
  float log_d_pde_reverse_coefficient;
  float log_d_pde_constant;
  float source_event_log_density;
  float interval_distance;
  uint first_event_index;
  uint tracking_event_count;
  uint event_buffer;
  uint event_index_offset;
  bool scale_d_shared_by_distance;
  bool previous_delta;
  bool valid;
};

struct UPBPGPUPartialBeamVertex {
  GPUUPBPRecursiveWeights weights;
  SpectralResponse throughput;
  float medium_density;
};

struct UPBPGPUPointBeamIntersection {
  float beam_distance;
  float distance_squared;
};

struct UPBPGPUBeamBeamIntersection {
  float first_distance;
  float second_distance;
  float distance_squared;
  float sin_theta;
  float direction_dot;
};

bool upbp_vertex_medium_properties(GPUUPBPVertex vertex, SpectralQuery spect, out SpectralResponse scattering, out float phase_function_g) {
  scattering = spectral_response_zero(spect);
  phase_function_g = 0.0f;
  if ((vertex.flags & GPUUPBPVertexFlags::InlineMedium) != 0u) {
    scattering = upbp_unpack_spectral_response(vertex.inline_scattering);
    phase_function_g = vertex.inline_phase_function_g;
    return upbp_spectral_non_negative(scattering);
  }
  MediumAccess medium_access = (MediumAccess)0;
  if ((vertex.medium_index == kInvalidIndex) || (wavefront_try_load_medium(vertex.medium_index, medium_access) == false)) {
    return false;
  }
  const MediumAccessGPUContext access_context =
    make_medium_access_gpu_context(constants.scene.mediums, constants.scene.images, constants.scene.spectrums, constants.scene.spectral_values);
  const float density = medium_access.medium_class == Medium::Homogeneous ? 1.0f : medium_access_sample_density(access_context, medium_access, vertex.position);
  if ((density < 0.0f) || (density > 1.0f) || (isfinite(density) == false)) {
    return false;
  }
  scattering = spectral_response_mul(gpu_medium_scattering(medium_access, spect), density);
  phase_function_g = medium_access.phase_function_g;
  return upbp_spectral_non_negative(scattering);
}

bool upbp_interval_medium_properties(GPUUPBPInterval interval, SpectralQuery spect, float3 position, out SpectralResponse scattering, out SpectralResponse extinction,
  out float density, out float phase_function_g) {
  scattering = spectral_response_zero(spect);
  extinction = spectral_response_zero(spect);
  density = 0.0f;
  phase_function_g = 0.0f;
  if ((interval.flags & GPUUPBPIntervalFlags::InlineMedium) != 0u) {
    scattering = upbp_unpack_spectral_response(interval.inline_scattering);
    extinction = spectral_response_add(scattering, upbp_unpack_spectral_response(interval.inline_absorption));
    density = 1.0f;
    return upbp_spectral_non_negative(scattering) && upbp_spectral_non_negative(extinction);
  }
  MediumAccess medium_access = (MediumAccess)0;
  if ((interval.medium_index == kInvalidIndex) || (wavefront_try_load_medium(interval.medium_index, medium_access) == false)) {
    return false;
  }
  const MediumAccessGPUContext access_context =
    make_medium_access_gpu_context(constants.scene.mediums, constants.scene.images, constants.scene.spectrums, constants.scene.spectral_values);
  density = medium_access.medium_class == Medium::Homogeneous ? 1.0f : medium_access_sample_density(access_context, medium_access, position);
  if ((density < 0.0f) || (density > 1.0f) || (isfinite(density) == false)) {
    return false;
  }
  scattering = spectral_response_mul(gpu_medium_scattering(medium_access, spect), density);
  extinction = spectral_response_mul(spectral_response_add(gpu_medium_scattering(medium_access, spect), gpu_medium_absorption(medium_access, spect)), density);
  phase_function_g = medium_access.phase_function_g;
  return upbp_spectral_non_negative(scattering) && upbp_spectral_non_negative(extinction);
}

float upbp_medium_phase(float phase_function_g, float3 incoming_direction, float3 outgoing_direction) {
  MediumAccess medium_access = (MediumAccess)0;
  medium_access.phase_function_g = phase_function_g;
  return gpu_medium_phase_function(medium_access, incoming_direction, outgoing_direction);
}

bool upbp_make_beam_from_interval(GPUUPBPResources resources, uint interval_index, out GPUUPBPBeam beam) {
  beam = (GPUUPBPBeam)0;
  if (interval_index >= resources.camera_interval_capacity) {
    return false;
  }
  const GPUUPBPInterval interval = upbp_load_interval(resources.interval_buffer, interval_index);
  if (((interval.flags & GPUUPBPIntervalFlags::Valid) == 0u) || (interval.medium_index == kInvalidIndex) || (interval.segment_index >= resources.camera_segment_capacity)) {
    return false;
  }
  const GPUUPBPSegment segment = upbp_load_segment(resources.segment_buffer, interval.segment_index);
  if (((segment.flags & GPUUPBPSegmentFlags::Valid) == 0u) || (segment.source_vertex_index >= resources.camera_vertex_capacity)) {
    return false;
  }
  const GPUUPBPVertex source = upbp_load_vertex(resources.vertex_buffer, segment.source_vertex_index);
  const float3 delta = interval.end_position - interval.start_position;
  const float length_squared = dot(delta, delta);
  if (((source.flags & GPUUPBPVertexFlags::Valid) == 0u) || (length_squared <= 0.0f) || (isfinite(length_squared) == false)) {
    return false;
  }
  beam.origin = interval.start_position;
  beam.length = sqrt(length_squared);
  beam.direction = delta / beam.length;
  beam.flags = GPUUPBPBeamFlags::Valid;
  beam.source_vertex_index = segment.source_vertex_index;
  beam.interval_index = interval_index;
  beam.global_path_index = source.global_path_index;
  beam.path_length = source.path_length;
  return true;
}

bool upbp_intersect_point_beam(float3 point_position, GPUUPBPBeam beam, float radius, out UPBPGPUPointBeamIntersection result) {
  result = (UPBPGPUPointBeamIntersection)0;
  if ((beam.length <= 0.0f) || (radius <= 0.0f)) {
    return false;
  }
  result.beam_distance = dot(point_position - beam.origin, beam.direction);
  if ((result.beam_distance < 0.0f) || (result.beam_distance >= beam.length)) {
    return false;
  }
  const float3 delta = point_position - (beam.origin + beam.direction * result.beam_distance);
  result.distance_squared = dot(delta, delta);
  return result.distance_squared < (radius * radius);
}

bool upbp_intersect_beams(GPUUPBPBeam first, GPUUPBPBeam second, float radius, out UPBPGPUBeamBeamIntersection result) {
  result = (UPBPGPUBeamBeamIntersection)0;
  if ((first.length <= 0.0f) || (second.length <= 0.0f) || (radius <= 0.0f)) {
    return false;
  }
  const float3 direction_cross = cross(first.direction, second.direction);
  const float sin_theta_squared = dot(direction_cross, direction_cross);
  if (sin_theta_squared <= (16.0f * 1.192092896e-7f)) {
    return false;
  }
  const float3 origin_delta = first.origin - second.origin;
  const float scaled_distance = dot(origin_delta, direction_cross);
  const float scaled_distance_squared = scaled_distance * scaled_distance;
  if (scaled_distance_squared >= (radius * radius * sin_theta_squared)) {
    return false;
  }
  result.direction_dot = dot(first.direction, second.direction);
  const float first_projection = dot(first.direction, origin_delta);
  const float second_projection = dot(second.direction, origin_delta);
  result.first_distance = (result.direction_dot * second_projection - first_projection) / sin_theta_squared;
  result.second_distance = (second_projection - result.direction_dot * first_projection) / sin_theta_squared;
  if ((result.first_distance < 0.0f) || (result.first_distance >= first.length) || (result.second_distance < 0.0f) || (result.second_distance >= second.length)) {
    return false;
  }
  result.distance_squared = scaled_distance_squared / sin_theta_squared;
  result.sin_theta = sqrt(sin_theta_squared);
  return true;
}

uint upbp_pair_seed(GPUUPBPIteration iteration, GPUUPBPVertex camera_vertex, GPUUPBPVertex light_vertex) {
  uint result = sampler_random_seed(load_scene_options_random_seed(), kUPBPRandomDomainScatteringEvaluation);
  result = sampler_random_seed(result, iteration.sample_index);
  result = sampler_random_seed(result, 0u);
  result = sampler_random_seed(result, light_vertex.global_path_index);
  result = sampler_random_seed(result, camera_vertex.global_path_index);
  result = sampler_random_seed(result, camera_vertex.path_length);
  return sampler_random_seed(result, light_vertex.path_length);
}

float upbp_kernel_value(uint kernel, uint dimension, float radius, float distance_squared) {
  if ((distance_squared < 0.0f) || (radius <= 0.0f) || (distance_squared >= (radius * radius))) {
    return 0.0f;
  }
  const float radius_squared = radius * radius;
  float support_measure = 0.0f;
  if (dimension == 1u) {
    support_measure = 2.0f * radius;
  } else if (dimension == 2u) {
    support_measure = kPi * radius_squared;
  } else if (dimension == 3u) {
    support_measure = (4.0f / 3.0f) * kPi * radius_squared * radius;
  }
  if (support_measure <= 0.0f) {
    return 0.0f;
  }
  if (kernel == 0u) {
    return rcp(support_measure);
  }
  const float profile = 1.0f - distance_squared / radius_squared;
  if (dimension == 1u) {
    return 3.0f * profile / (4.0f * radius);
  }
  if (dimension == 2u) {
    return 2.0f * profile / (kPi * radius_squared);
  }
  return dimension == 3u ? 15.0f * profile / (8.0f * kPi * radius_squared * radius) : 0.0f;
}

bool upbp_density_contribution_finite(GPUUPBPResources resources, uint technique_flag, uint global_path_index, SpectralResponse contribution) {
  if (gpu_valid_spectral_response(contribution)) {
    return true;
  }
  RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(resources.counter_buffer);
  uint ignored = 0u;
  counters.InterlockedAdd(GPUUPBPCounterIndex::FailedConnections * sizeof(uint), 1u, ignored);
  uint previous_failure = 0u;
  counters.InterlockedCompareExchange(GPUUPBPCounterIndex::FirstFailureCode * sizeof(uint), GPUUPBPPathFailure::None, GPUUPBPPathFailure::NonFiniteDensityContribution,
    previous_failure);
  if (previous_failure == GPUUPBPPathFailure::None) {
    counters.Store(GPUUPBPCounterIndex::FirstFailureGlobalPath * sizeof(uint), global_path_index);
    counters.Store(GPUUPBPCounterIndex::FirstFailureDetail0 * sizeof(uint), technique_flag);
    counters.Store(GPUUPBPCounterIndex::FirstFailureDetail1 * sizeof(uint), asuint(contribution.value));
    const uint finite_mask = (isfinite(contribution.value) ? 1u : 0u) | (isfinite(contribution.integrated.x) ? 2u : 0u) | (isfinite(contribution.integrated.y) ? 4u : 0u) |
                             (isfinite(contribution.integrated.z) ? 8u : 0u);
    counters.Store(GPUUPBPCounterIndex::FirstFailureDetail2 * sizeof(uint), finite_mask);
    counters.Store(GPUUPBPCounterIndex::FirstFailureDetail3 * sizeof(uint), resources.iteration.light_batch_offset);
  }
  return false;
}

float upbp_point_merge_mis_weight(GPUUPBPIteration iteration, uint selected_technique, GPUUPBPVertex light_vertex, GPUUPBPVertex camera_vertex, float scattering_pdf_forward,
  float scattering_pdf_reverse, float sin_theta) {
  const GPUUPBPRecursiveWeights light = light_vertex.arrival_weights;
  const GPUUPBPRecursiveWeights camera = camera_vertex.arrival_weights;
  const float log_forward_ray_factor = light.log_ray_sample_forward_ratio;
  const float log_reverse_ray_factor = camera.log_ray_sample_forward_pdf_inverse;
  const float log_selected_factor = upbp_log_density_competitor_factor(iteration, selected_technique, camera_vertex, log_forward_ray_factor, log_reverse_ray_factor, sin_theta);
  if (upbp_log_is_zero(log_selected_factor) || (scattering_pdf_forward <= 0.0f) || (scattering_pdf_reverse <= 0.0f)) {
    return 0.0f;
  }

  const float light_bpt_applicable = (light.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u ? 0.0f : 1.0f;
  const float camera_bpt_applicable = (camera.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u ? 0.0f : 1.0f;
  const float log_light_shared = light_bpt_applicable > 0.0f ? light.log_d_shared + upbp_log_positive(iteration.bpt_sample_count) : kUPBPLogZero;
  const float log_light_pde = upbp_log_is_zero(light.log_d_pde) ? kUPBPLogZero : log(scattering_pdf_forward) + light.log_d_pde - light.log_ray_sample_reverse_pdf_inverse;
  const float log_w_light = upbp_log_add(log_light_shared, log_light_pde) - log_selected_factor;
  const float log_camera_shared = camera_bpt_applicable > 0.0f ? camera.log_d_shared + upbp_log_positive(iteration.bpt_sample_count) : kUPBPLogZero;
  const float log_camera_pde = upbp_log_is_zero(camera.log_d_pde) ? kUPBPLogZero : log(scattering_pdf_reverse) + camera.log_d_pde - camera.log_ray_sample_reverse_pdf_inverse;
  const float log_w_camera = upbp_log_add(log_camera_shared, log_camera_pde) - log_selected_factor;
  float log_w_local = 0.0f;
  if (upbp_vertex_is_medium(camera_vertex)) {
    const uint techniques[4] = {GPUUPBPTechnique::PP3D, GPUUPBPTechnique::PB2D, GPUUPBPTechnique::BP2D, GPUUPBPTechnique::BB1D};
    [unroll] for (uint index = 0u; index < 4u; ++index) {
      if (techniques[index] != selected_technique) {
        const float log_factor = upbp_log_density_competitor_factor(iteration, techniques[index], camera_vertex, log_forward_ray_factor, log_reverse_ray_factor, sin_theta);
        log_w_local = upbp_log_add(log_w_local, upbp_log_is_zero(log_factor) ? kUPBPLogZero : log_factor - log_selected_factor);
      }
    }
  }
  const float log_denominator = upbp_log_add(log_w_light, upbp_log_add(log_w_local, log_w_camera));
  return isfinite(log_denominator) ? exp(-log_denominator) : 0.0f;
}

bool upbp_prepare_beam(GPUUPBPResources resources, GPUUPBPBeam beam, out UPBPGPUPreparedBeam result) {
  result = (UPBPGPUPreparedBeam)0;
  result.log_d_shared = kUPBPLogZero;
  result.log_d_pde_reverse_coefficient = kUPBPLogZero;
  result.log_d_pde_constant = kUPBPLogZero;
  result.source_throughput = spectral_response_zero((SpectralQuery)0);
  result.transport_at_origin.weight = spectral_response_zero((SpectralQuery)0);
  const uint total_vertex_capacity = resources.camera_vertex_capacity + resources.light_vertex_capacity;
  const uint total_segment_capacity = resources.camera_segment_capacity + resources.light_segment_capacity;
  const uint total_interval_capacity = resources.camera_interval_capacity + resources.light_interval_capacity;
  if (((beam.flags & GPUUPBPBeamFlags::Valid) == 0u) || (beam.source_vertex_index >= total_vertex_capacity) || (beam.interval_index >= total_interval_capacity)) {
    return false;
  }
  const GPUUPBPVertex source = upbp_load_vertex(resources.vertex_buffer, beam.source_vertex_index);
  const GPUUPBPInterval beam_interval = upbp_load_interval(resources.interval_buffer, beam.interval_index);
  const bool recompute_tracking = (beam_interval.flags & GPUUPBPIntervalFlags::RecomputeTracking) != 0u;
  if (((source.flags & GPUUPBPVertexFlags::Valid) == 0u) || ((source.flags & GPUUPBPVertexFlags::HasDeparture) == 0u) ||
      ((beam_interval.flags & GPUUPBPIntervalFlags::Valid) == 0u) || (beam_interval.medium_index == kInvalidIndex) || (beam_interval.segment_index >= total_segment_capacity) ||
      ((recompute_tracking == false) && ((beam_interval.event_count == 0u) || (beam_interval.first_event_index == kInvalidIndex)))) {
    return false;
  }
  const GPUUPBPSegment segment = upbp_load_segment(resources.segment_buffer, beam_interval.segment_index);
  if (((segment.flags & GPUUPBPSegmentFlags::Valid) == 0u) || (segment.source_vertex_index != beam.source_vertex_index) || (segment.first_interval_index == kInvalidIndex)) {
    return false;
  }

  result.source_throughput = upbp_unpack_spectral_response(source.outgoing_throughput);
  result.transport_at_origin.weight = spectral_response_make(spectral_response_as_query(result.source_throughput), 1.0f);
  uint interval_index = segment.first_interval_index;
  [loop] for (uint ordinal = 0u; ordinal < segment.interval_count; ++ordinal) {
    if (interval_index >= total_interval_capacity) {
      return false;
    }
    const GPUUPBPInterval interval = upbp_load_interval(resources.interval_buffer, interval_index);
    if (((interval.flags & GPUUPBPIntervalFlags::Valid) == 0u) || (interval.segment_index != beam_interval.segment_index)) {
      return false;
    }
    if (interval_index == beam.interval_index) {
      break;
    }
    if ((interval.flags & (GPUUPBPIntervalFlags::Scatter | GPUUPBPIntervalFlags::Absorb)) != 0u) {
      return false;
    }
    result.transport_at_origin.weight = spectral_response_mul(result.transport_at_origin.weight, upbp_unpack_spectral_response(interval.weight));
    result.transport_at_origin.log_transport_pdf_forward += interval.log_transport_pdf_forward;
    result.transport_at_origin.log_transport_pdf_reverse += interval.log_transport_pdf_reverse;
    result.transport_at_origin.distance += interval.distance;
    interval_index = interval.next_interval_index;
  }
  if (interval_index != beam.interval_index) {
    return false;
  }

  const GPUUPBPRecursiveState departure = source.departure_state;
  result.first_event_index = beam_interval.first_event_index;
  result.tracking_event_count = beam_interval.event_count;
  result.event_buffer = resources.event_buffer;
  result.event_index_offset = 0u;
  result.interval_distance = beam_interval.distance;
  result.log_d_shared = departure.weights.log_d_shared;
  result.previous_delta = (departure.weights.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u;
  result.source_event_log_density = upbp_vertex_is_medium(source) ? source.log_medium_event_density : 0.0f;
  const float log_source_ray_ratio = upbp_vertex_is_medium(source) ? -result.source_event_log_density : kUPBPLogZero;
  result.scale_d_shared_by_distance = (source.path_length > 0u) || ((source.flags & GPUUPBPVertexFlags::DistantEndpoint) == 0u);
  if (source.path_length > 0u) {
    const float log_constant = upbp_log_recursive_local_pde_factor(resources.iteration, source, departure.weights, kUPBPLogZero, log_source_ray_ratio, departure.last_sin_theta);
    float log_coefficient = kUPBPLogZero;
    if ((source.flags & GPUUPBPVertexFlags::Camera) == 0u) {
      const float log_forward_ray_factor = departure.weights.log_ray_sample_forward_ratio;
      log_coefficient =
        upbp_log_add(upbp_log_density_competitor_factor(resources.iteration, GPUUPBPTechnique::PB2D, source, log_forward_ray_factor, 0.0f, departure.last_sin_theta),
          upbp_log_density_competitor_factor(resources.iteration, GPUUPBPTechnique::BB1D, source, log_forward_ray_factor, 0.0f, departure.last_sin_theta));
    }
    result.log_d_pde_reverse_coefficient = upbp_log_product(departure.log_d_pde_a, log_coefficient);
    result.log_d_pde_constant = upbp_log_add(upbp_log_product(departure.log_d_pde_a, log_constant), departure.log_d_pde_b);
  } else {
    result.log_d_pde_constant = departure.weights.log_d_pde;
  }
  result.valid = (beam_interval.medium_index != kInvalidIndex) && isfinite(result.source_event_log_density) && isfinite(result.log_d_shared) &&
                 isfinite(result.log_d_pde_reverse_coefficient) && isfinite(result.log_d_pde_constant) && (result.interval_distance > 0.0f);
  return result.valid;
}

GPUUPBPDensityPoint upbp_pack_density_point(GPUUPBPVertex vertex) {
  GPUUPBPDensityPoint result = (GPUUPBPDensityPoint)0;
  result.throughput = vertex.throughput;
  result.position = vertex.position;
  result.flags = vertex.flags;
  result.w_i = vertex.w_i;
  result.medium_index = vertex.medium_index;
  result.geo_normal = vertex.geo_normal;
  result.path_length = vertex.path_length;
  result.arrival_weights = vertex.arrival_weights;
  result.global_path_index = vertex.global_path_index;
  result.log_medium_event_density = vertex.log_medium_event_density;
  result.inline_phase_function_g = vertex.inline_phase_function_g;
  result.inline_scattering = vertex.inline_scattering;
  result.inline_extinction = vertex.inline_extinction;
  return result;
}

GPUUPBPVertex upbp_unpack_density_point(GPUUPBPDensityPoint record) {
  GPUUPBPVertex result = (GPUUPBPVertex)0;
  result.throughput = record.throughput;
  result.position = record.position;
  result.flags = record.flags;
  result.w_i = record.w_i;
  result.medium_index = record.medium_index;
  result.geo_normal = record.geo_normal;
  result.path_length = record.path_length;
  result.arrival_weights = record.arrival_weights;
  result.global_path_index = record.global_path_index;
  result.log_medium_event_density = record.log_medium_event_density;
  result.inline_phase_function_g = record.inline_phase_function_g;
  result.inline_scattering = record.inline_scattering;
  result.inline_extinction = record.inline_extinction;
  return result;
}

GPUUPBPDensityBeam upbp_pack_density_beam(GPUUPBPResources resources, GPUUPBPBeam beam, GPUUPBPInterval interval, UPBPGPUPreparedBeam prepared) {
  GPUUPBPDensityBeam result = (GPUUPBPDensityBeam)0;
  result.beam = beam;
  result.beam.source_vertex_index = kInvalidIndex;
  result.beam.interval_index = kInvalidIndex;
  result.interval = interval;
  result.interval.segment_index = kInvalidIndex;
  result.interval.next_interval_index = kInvalidIndex;
  result.source_throughput = upbp_pack_spectral_response(prepared.source_throughput);
  result.transport_weight = upbp_pack_spectral_response(prepared.transport_at_origin.weight);
  result.transport_log_pdf_forward = prepared.transport_at_origin.log_transport_pdf_forward;
  result.transport_log_pdf_reverse = prepared.transport_at_origin.log_transport_pdf_reverse;
  result.transport_distance = prepared.transport_at_origin.distance;
  result.log_d_shared = prepared.log_d_shared;
  result.log_d_pde_reverse_coefficient = prepared.log_d_pde_reverse_coefficient;
  result.log_d_pde_constant = prepared.log_d_pde_constant;
  result.source_event_log_density = prepared.source_event_log_density;
  result.interval_distance = prepared.interval_distance;
  result.flags = GPUUPBPDensityBeamFlags::Valid;
  result.event_buffer = (interval.flags & GPUUPBPIntervalFlags::RecomputeTracking) != 0u ? kInvalidIndex : resources.density_output_event_buffer;
  result.event_index_offset = 0u;
  result.flags |= prepared.scale_d_shared_by_distance ? GPUUPBPDensityBeamFlags::ScaleDSharedByDistance : 0u;
  result.flags |= prepared.previous_delta ? GPUUPBPDensityBeamFlags::PreviousDelta : 0u;
  return result;
}

UPBPGPUPreparedBeam upbp_unpack_density_beam(GPUUPBPDensityBeam record) {
  UPBPGPUPreparedBeam result = (UPBPGPUPreparedBeam)0;
  result.source_throughput = upbp_unpack_spectral_response(record.source_throughput);
  result.transport_at_origin.weight = upbp_unpack_spectral_response(record.transport_weight);
  result.transport_at_origin.log_transport_pdf_forward = record.transport_log_pdf_forward;
  result.transport_at_origin.log_transport_pdf_reverse = record.transport_log_pdf_reverse;
  result.transport_at_origin.distance = record.transport_distance;
  result.log_d_shared = record.log_d_shared;
  result.log_d_pde_reverse_coefficient = record.log_d_pde_reverse_coefficient;
  result.log_d_pde_constant = record.log_d_pde_constant;
  result.source_event_log_density = record.source_event_log_density;
  result.interval_distance = record.interval_distance;
  result.first_event_index = record.interval.first_event_index;
  result.tracking_event_count = record.interval.event_count;
  result.event_buffer = record.event_buffer;
  result.event_index_offset = record.event_index_offset;
  result.scale_d_shared_by_distance = (record.flags & GPUUPBPDensityBeamFlags::ScaleDSharedByDistance) != 0u;
  result.previous_delta = (record.flags & GPUUPBPDensityBeamFlags::PreviousDelta) != 0u;
  result.valid = (record.flags & GPUUPBPDensityBeamFlags::Valid) != 0u;
  return result;
}

bool upbp_prepared_interval_prefix(GPUUPBPResources resources, UPBPGPUPreparedBeam prepared, GPUUPBPBeam beam, GPUUPBPInterval interval, float prefix_distance,
  out UPBPGPUBeamTransportPrefix result) {
  result = (UPBPGPUBeamTransportPrefix)0;
  result.weight = spectral_response_zero(spectral_response_as_query(prepared.source_throughput));
  if ((prepared.valid == false) || (prefix_distance <= 0.0f) || (prefix_distance >= prepared.interval_distance)) {
    return false;
  }
  if ((interval.flags & GPUUPBPIntervalFlags::RecomputeTracking) != 0u) {
    GPUUPBPConnectionInterval partial = (GPUUPBPConnectionInterval)0;
    uint terminal_type = kUPBPMediumFailure;
    uint seed = interval.tracking_seed;
    bool tracking_valid = false;
    if ((interval.flags & GPUUPBPIntervalFlags::InlineMedium) != 0u) {
      const SpectralResponse extinction =
        spectral_response_add(upbp_unpack_spectral_response(interval.inline_scattering), upbp_unpack_spectral_response(interval.inline_absorption));
      tracking_valid = upbp_track_connection_homogeneous_extinction(extinction, prefix_distance, spectral_response_as_query(prepared.source_throughput),
        resources.iteration.maximum_null_events_per_interval, seed, partial, terminal_type);
    } else {
      tracking_valid = upbp_track_connection_interval(interval.medium_index, beam.origin, beam.direction, prefix_distance, spectral_response_as_query(prepared.source_throughput),
        resources.iteration.maximum_null_events_per_interval, seed, partial, terminal_type);
    }
    if ((tracking_valid == false) || (terminal_type != kUPBPMediumEscape)) {
      return false;
    }
    result.weight = partial.weight;
    result.log_transport_pdf_forward = partial.log_transport_pdf_forward;
    result.log_transport_pdf_reverse = partial.log_transport_pdf_reverse;
    result.distance = prefix_distance;
    return true;
  }
  uint event_index = prepared.first_event_index;
  if (prepared.event_buffer == kInvalidIndex) {
    return false;
  }
  [loop] for (uint ordinal = 0u; ordinal < prepared.tracking_event_count; ++ordinal) {
    if (event_index == kInvalidIndex) {
      return false;
    }
    const GPUUPBPTrackingEvent event_record = upbp_load_tracking_event(prepared.event_buffer, prepared.event_index_offset + event_index);
    if (event_record.end_distance >= prefix_distance) {
      if ((event_record.majorant <= 0.0f) || (isfinite(event_record.majorant) == false)) {
        return false;
      }
      result.weight = upbp_unpack_spectral_response(event_record.weight_before);
      result.log_transport_pdf_forward = event_record.log_transport_pdf_forward_before;
      result.log_transport_pdf_reverse = event_record.log_transport_pdf_reverse_before;
      const float remaining_distance = prefix_distance - event_record.distance_before;
      const float log_transmittance = -event_record.majorant * remaining_distance;
      result.log_transport_pdf_forward += log_transmittance;
      result.log_transport_pdf_reverse += log_transmittance;
      result.distance = prefix_distance;
      return isfinite(log_transmittance);
    }
    event_index = event_record.next_event_index;
  }
  return false;
}

bool upbp_partial_prepared_beam_vertex_with_interval(GPUUPBPResources resources, UPBPGPUPreparedBeam prepared, GPUUPBPBeam beam, GPUUPBPInterval interval, float distance,
  out UPBPGPUPartialBeamVertex result) {
  result = (UPBPGPUPartialBeamVertex)0;
  result.throughput = spectral_response_zero(spectral_response_as_query(prepared.source_throughput));
  if (prepared.valid == false) {
    return false;
  }
  UPBPGPUBeamTransportPrefix partial_interval = (UPBPGPUBeamTransportPrefix)0;
  if (upbp_prepared_interval_prefix(resources, prepared, beam, interval, distance, partial_interval) == false) {
    return false;
  }
  SpectralResponse scattering = spectral_response_zero(spectral_response_as_query(prepared.source_throughput));
  SpectralResponse extinction = spectral_response_zero(spectral_response_as_query(prepared.source_throughput));
  float phase_function_g = 0.0f;
  if (upbp_interval_medium_properties(interval, spectral_response_as_query(prepared.source_throughput), beam.origin + beam.direction * distance, scattering, extinction,
        result.medium_density, phase_function_g) == false) {
    return false;
  }
  const float real_event_density = upbp_spectral_average(extinction);
  const float log_forward = prepared.transport_at_origin.log_transport_pdf_forward + partial_interval.log_transport_pdf_forward;
  const float log_reverse = prepared.transport_at_origin.log_transport_pdf_reverse + partial_interval.log_transport_pdf_reverse;
  const float log_real_event_density = upbp_log_positive(real_event_density);
  const float log_forward_pdf = log_forward + log_real_event_density;
  const float log_reverse_pdf = log_reverse + prepared.source_event_log_density;
  const float transport_distance = prepared.transport_at_origin.distance + partial_interval.distance;
  if ((real_event_density <= 0.0f) || (transport_distance <= 0.0f) || (isfinite(log_forward_pdf) == false) || (isfinite(log_reverse_pdf) == false)) {
    return false;
  }
  result.weights.log_d_pde =
    upbp_log_add(upbp_log_is_zero(prepared.log_d_pde_reverse_coefficient) ? kUPBPLogZero : prepared.log_d_pde_reverse_coefficient - log_reverse_pdf, prepared.log_d_pde_constant);
  result.weights.log_d_pde = upbp_log_is_zero(result.weights.log_d_pde) ? kUPBPLogZero : result.weights.log_d_pde - log_forward_pdf;
  result.weights.log_d_shared = prepared.log_d_shared - log_forward_pdf;
  if (prepared.scale_d_shared_by_distance) {
    result.weights.log_d_shared += 2.0f * log(transport_distance);
  }
  result.weights.log_d_bpt = kUPBPLogZero;
  result.weights.log_ray_sample_forward_pdf_inverse = -log_forward_pdf;
  result.weights.log_ray_sample_reverse_pdf_inverse = -log_reverse_pdf;
  result.weights.log_ray_sample_forward_ratio = -log_real_event_density;
  result.weights.log_ray_sample_reverse_ratio = kUPBPLogZero;
  result.weights.flags = prepared.previous_delta ? GPUUPBPRecursiveWeightFlags::PreviousDelta : 0u;
  result.throughput = spectral_response_mul(prepared.source_throughput, spectral_response_mul(prepared.transport_at_origin.weight, partial_interval.weight));
  return upbp_recursive_weights_finite(result.weights) && (spectral_response_is_zero(result.throughput) == false);
}

bool upbp_partial_prepared_beam_vertex(GPUUPBPResources resources, UPBPGPUPreparedBeam prepared, GPUUPBPBeam beam, float distance, out UPBPGPUPartialBeamVertex result) {
  if (beam.interval_index >= (resources.camera_interval_capacity + resources.light_interval_capacity)) {
    result = (UPBPGPUPartialBeamVertex)0;
    return false;
  }
  return upbp_partial_prepared_beam_vertex_with_interval(resources, prepared, beam, upbp_load_interval(resources.interval_buffer, beam.interval_index), distance, result);
}

float upbp_point_merge_mis_weight(GPUUPBPIteration iteration, uint selected_technique, GPUUPBPRecursiveWeights light, GPUUPBPRecursiveWeights camera, GPUUPBPVertex context_vertex,
  float scattering_pdf_forward, float scattering_pdf_reverse, float sin_theta) {
  const float log_forward_ray_factor = light.log_ray_sample_forward_ratio;
  const float log_reverse_ray_factor = camera.log_ray_sample_forward_pdf_inverse;
  const float log_selected_factor = upbp_log_density_competitor_factor(iteration, selected_technique, context_vertex, log_forward_ray_factor, log_reverse_ray_factor, sin_theta);
  if (upbp_log_is_zero(log_selected_factor) || (scattering_pdf_forward <= 0.0f) || (scattering_pdf_reverse <= 0.0f)) {
    return 0.0f;
  }
  const bool light_previous_delta = (light.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u;
  const bool camera_previous_delta = (camera.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u;
  const float log_light_shared = light_previous_delta ? kUPBPLogZero : light.log_d_shared + upbp_log_positive(iteration.bpt_sample_count);
  const float log_light_pde = upbp_log_is_zero(light.log_d_pde) ? kUPBPLogZero : log(scattering_pdf_forward) + light.log_d_pde - light.log_ray_sample_reverse_pdf_inverse;
  const float log_w_light = upbp_log_add(log_light_shared, log_light_pde) - log_selected_factor;
  const float log_camera_shared = camera_previous_delta ? kUPBPLogZero : camera.log_d_shared + upbp_log_positive(iteration.bpt_sample_count);
  const float log_camera_pde = upbp_log_is_zero(camera.log_d_pde) ? kUPBPLogZero : log(scattering_pdf_reverse) + camera.log_d_pde - camera.log_ray_sample_reverse_pdf_inverse;
  const float log_w_camera = upbp_log_add(log_camera_shared, log_camera_pde) - log_selected_factor;
  float log_w_local = 0.0f;
  const uint techniques[4] = {GPUUPBPTechnique::PP3D, GPUUPBPTechnique::PB2D, GPUUPBPTechnique::BP2D, GPUUPBPTechnique::BB1D};
  [unroll] for (uint index = 0u; index < 4u; ++index) {
    if (techniques[index] != selected_technique) {
      const float log_factor = upbp_log_density_competitor_factor(iteration, techniques[index], context_vertex, log_forward_ray_factor, log_reverse_ray_factor, sin_theta);
      log_w_local = upbp_log_add(log_w_local, upbp_log_is_zero(log_factor) ? kUPBPLogZero : log_factor - log_selected_factor);
    }
  }
  const float log_denominator = upbp_log_add(log_w_light, upbp_log_add(log_w_local, log_w_camera));
  return isfinite(log_denominator) ? exp(-log_denominator) : 0.0f;
}

#if defined(ETX_UPBP_SURFACE_VARIANT)
Vertex upbp_surface_vertex(GPUUPBPVertex vertex) {
  TriangleData triangle_data = load_triangle(WAVEFRONT_RO_BUFFER(constants.scene.triangles), vertex.triangle_index);
  Vertex result = wavefront_interpolate_vertex(triangle_data, vertex.barycentric);
  if (vertex.instance_index != kInvalidIndex) {
    result = scene_instance_transform_vertex(load_scene_instance(vertex.instance_index), result);
  }
  const float handedness = dot(cross(result.nrm, result.tan), result.btn) >= 0.0f ? 1.0f : -1.0f;
  result.pos = vertex.position;
  result.nrm = vertex.normal;
  result.tex = vertex.texcoord;
  scene_math_shared_build_sampling_frame_with_handedness(result.nrm, result.tan, result.btn, handedness, result.nrm, result.tan, result.btn);
  return result;
}
#endif

bool upbp_medium_pre_collision_throughput_with_scattering(GPUUPBPVertex vertex, SpectralResponse throughput, SpectralResponse scattering, out SpectralResponse result) {
  result = spectral_response_zero(spectral_response_as_query(throughput));
  const float event_density = exp(vertex.log_medium_event_density);
  const float minimum_scattering =
    (scattering.flags & SpectralFlags::Spectral) != 0u ? scattering.value : min(scattering.integrated.x, min(scattering.integrated.y, scattering.integrated.z));
  if ((event_density <= 0.0f) || (minimum_scattering < 0.0f)) {
    return false;
  }
  if ((throughput.flags & SpectralFlags::Spectral) != 0u) {
    if (scattering.value > 0.0f) {
      result.value = throughput.value * event_density / scattering.value;
    } else if (throughput.value != 0.0f) {
      return false;
    }
    return isfinite(result.value);
  }
  if (scattering.integrated.x > 0.0f) {
    result.integrated.x = throughput.integrated.x * event_density / scattering.integrated.x;
  } else if (throughput.integrated.x != 0.0f) {
    return false;
  }
  if (scattering.integrated.y > 0.0f) {
    result.integrated.y = throughput.integrated.y * event_density / scattering.integrated.y;
  } else if (throughput.integrated.y != 0.0f) {
    return false;
  }
  if (scattering.integrated.z > 0.0f) {
    result.integrated.z = throughput.integrated.z * event_density / scattering.integrated.z;
  } else if (throughput.integrated.z != 0.0f) {
    return false;
  }
  return all(isfinite(result.integrated));
}

bool upbp_medium_pre_collision_throughput(GPUUPBPVertex vertex, SpectralResponse throughput, out SpectralResponse result) {
  SpectralResponse scattering = spectral_response_zero(spectral_response_as_query(throughput));
  float phase_function_g = 0.0f;
  if (upbp_vertex_medium_properties(vertex, spectral_response_as_query(throughput), scattering, phase_function_g) == false) {
    result = spectral_response_zero(spectral_response_as_query(throughput));
    return false;
  }
  return upbp_medium_pre_collision_throughput_with_scattering(vertex, throughput, scattering, result);
}

void upbp_evaluate_point_vertex(GPUUPBPResources resources, GPUUPBPVertex camera_vertex, GPUUPBPVertex light_vertex, uint selected_technique, float radius,
  inout SpectralResponse accumulated) {
  if (camera_vertex.path_length == 0u) {
    return;
  }
  if (((light_vertex.flags & GPUUPBPVertexFlags::Valid) == 0u) || (light_vertex.path_length == 0u)) {
    return;
  }
  const bool surface = selected_technique == GPUUPBPTechnique::Surface;
  if (surface) {
    if ((upbp_vertex_is_surface(light_vertex) == false) || (upbp_vertex_is_surface(camera_vertex) == false) || upbp_vertex_is_delta(light_vertex) ||
        upbp_vertex_is_delta(camera_vertex) || (upbp_vertex_is_density_connectible(light_vertex) == false) || (upbp_vertex_is_density_connectible(camera_vertex) == false) ||
        (dot(light_vertex.geo_normal, camera_vertex.geo_normal) <= 0.0f)) {
      return;
    }
  } else if ((selected_technique != GPUUPBPTechnique::PP3D) || (upbp_vertex_is_medium(light_vertex) == false) || (upbp_vertex_is_medium(camera_vertex) == false) ||
             upbp_vertex_is_delta(light_vertex) || upbp_vertex_is_delta(camera_vertex) || (upbp_vertex_is_density_connectible(light_vertex) == false) ||
             (upbp_vertex_is_density_connectible(camera_vertex) == false) || (light_vertex.medium_index == kInvalidIndex) ||
             (light_vertex.medium_index != camera_vertex.medium_index)) {
    return;
  }
  const uint path_length = light_vertex.path_length + camera_vertex.path_length;
  if ((path_length < load_scene_options_min_path_length()) || (path_length > load_scene_options_max_path_length())) {
    return;
  }
  SpectralResponse light_throughput = upbp_unpack_spectral_response(light_vertex.throughput);
  SpectralResponse camera_throughput = upbp_unpack_spectral_response(camera_vertex.throughput);
  if (spectral_query_compatible(spectral_response_as_query(light_throughput), spectral_response_as_query(camera_throughput)) == false) {
    return;
  }
  const float3 delta = light_vertex.position - camera_vertex.position;
  const float kernel_value = upbp_kernel_value(resources.iteration.kernel, surface ? 2u : 3u, radius, dot(delta, delta));
  if (kernel_value <= 0.0f) {
    return;
  }

  const float3 outgoing_direction = -light_vertex.w_i;
  SpectralResponse scattering_value = spectral_response_zero(spectral_response_as_query(camera_throughput));
  float pdf_forward = 0.0f;
  float pdf_reverse = 0.0f;
  uint seed = upbp_pair_seed(resources.iteration, camera_vertex, light_vertex);
  if (surface) {
#if defined(ETX_UPBP_SURFACE_VARIANT)
    Material material = (Material)0;
    if ((camera_vertex.material_index == kInvalidIndex) || (try_load_material_full(camera_vertex.material_index, material) == false) ||
        (upbp_surface_stage_matches_material(material.cls) == false)) {
      return;
    }
    const Vertex vertex = upbp_surface_vertex(camera_vertex);
    BSDFData data = bsdf_data_make(vertex, spectral_response_as_query(camera_throughput), camera_vertex.incident_medium_index, PathSource::Camera, camera_vertex.w_i);
    Sampler sampler = make_bsdf_sampler(seed);
    const BSDFResourceContext context = make_scene_bsdf_resource_gpu_context();
    const BSDFEval evaluation = upbp_surface_stage_bsdf_eval(context, data, outgoing_direction, material, sampler);
    // Photon density already contains the incoming surface projection.
    scattering_value = evaluation.func;
    pdf_forward = evaluation.pdf;
    pdf_reverse = upbp_surface_stage_reverse_pdf(context, data, outgoing_direction, material, sampler);
#else
    return;
#endif
  } else {
    SpectralResponse medium_scattering = spectral_response_zero(spectral_response_as_query(camera_throughput));
    float phase_function_g = 0.0f;
    if (upbp_vertex_medium_properties(camera_vertex, spectral_response_as_query(camera_throughput), medium_scattering, phase_function_g) == false) {
      return;
    }
    pdf_forward = upbp_medium_phase(phase_function_g, camera_vertex.w_i, outgoing_direction);
    pdf_reverse = upbp_medium_phase(phase_function_g, outgoing_direction, camera_vertex.w_i);
    scattering_value = spectral_response_make(spectral_response_as_query(camera_throughput), pdf_forward);
  }
  if ((pdf_forward <= 0.0f) || (pdf_reverse <= 0.0f) || spectral_response_is_zero(scattering_value)) {
    return;
  }
  const float direction_cosine = dot(camera_vertex.w_i, outgoing_direction);
  const float sin_theta = sqrt(max(0.0f, 1.0f - direction_cosine * direction_cosine));
  const float mis_weight = upbp_point_merge_mis_weight(resources.iteration, selected_technique, light_vertex, camera_vertex, pdf_forward, pdf_reverse, sin_theta);
  if (mis_weight <= 0.0f) {
    return;
  }
  if (surface == false) {
    SpectralResponse light_pre_collision = spectral_response_zero(spectral_response_as_query(light_throughput));
    SpectralResponse camera_pre_collision = spectral_response_zero(spectral_response_as_query(camera_throughput));
    if ((upbp_medium_pre_collision_throughput(light_vertex, light_throughput, light_pre_collision) == false) ||
        (upbp_medium_pre_collision_throughput(camera_vertex, camera_throughput, camera_pre_collision) == false)) {
      return;
    }
    light_throughput = light_pre_collision;
    camera_throughput = camera_pre_collision;
    SpectralResponse medium_scattering = spectral_response_zero(spectral_response_as_query(camera_throughput));
    float phase_function_g = 0.0f;
    if (upbp_vertex_medium_properties(camera_vertex, spectral_response_as_query(camera_throughput), medium_scattering, phase_function_g) == false) {
      return;
    }
    scattering_value = spectral_response_mul(scattering_value, medium_scattering);
  }
  const float estimator_scale = kernel_value / float(resources.iteration.global_light_path_count);
  const SpectralResponse contribution =
    spectral_response_mul(spectral_response_mul(light_throughput, camera_throughput), spectral_response_mul(scattering_value, estimator_scale * mis_weight));
  accumulated = spectral_response_add(accumulated, contribution);
}

#if defined(ETX_UPBP_SURFACE_VARIANT)
groupshared GPUUPBPVertex upbp_surface_camera_vertices[64u];
groupshared GPUUPBPVertex upbp_surface_camera_vertex;
groupshared uint upbp_surface_query_valid;
groupshared uint upbp_surface_partition_counts[kGPUUPBPSurfacePartitionCount];
groupshared uint upbp_surface_partition_exhausted[kGPUUPBPSurfacePartitionCount];
groupshared uint upbp_surface_point_indices[64u];
groupshared GPUWavefrontCompactSpectralResponse upbp_surface_lane_contributions[64u];
# define ETX_UPBP_POINT_CALL [noinline]
#else
# define ETX_UPBP_POINT_CALL
#endif

ETX_UPBP_POINT_CALL void upbp_evaluate_point_indices(GPUUPBPResources resources, uint camera_vertex_index, uint camera_group_index, uint light_vertex_buffer,
  uint light_vertex_index, uint selected_technique, float radius, inout SpectralResponse accumulated) {
#if defined(ETX_UPBP_SURFACE_VARIANT)
  const GPUUPBPVertex camera_vertex = upbp_surface_camera_vertices[camera_group_index];
#else
  const GPUUPBPVertex camera_vertex = upbp_load_vertex(resources.vertex_buffer, camera_vertex_index);
#endif
  upbp_evaluate_point_vertex(resources, camera_vertex, upbp_unpack_density_point(upbp_load_density_point(light_vertex_buffer, light_vertex_index)), selected_technique, radius,
    accumulated);
}

#undef ETX_UPBP_POINT_CALL

void upbp_evaluate_point_candidate(GPUUPBPResources resources, GPUUPBPVertex camera_vertex, GPUUPBPPoint point_record, uint selected_technique, float radius,
  inout SpectralResponse accumulated) {
  if (point_record.vertex_index >= (resources.camera_vertex_capacity + resources.light_vertex_capacity)) {
    return;
  }
  upbp_evaluate_point_vertex(resources, camera_vertex, upbp_load_vertex(resources.vertex_buffer, point_record.vertex_index), selected_technique, radius, accumulated);
}

[noinline] void upbp_evaluate_point_merge_dispatch(uint dispatch_index, uint group_thread_index, uint selected_technique) {
  GPUWavefrontResources wavefront_resources = wavefront_load_resources();
  GPUUPBPResources resources = upbp_load_resources(wavefront_resources);
  const uint acceleration_structure = selected_technique == GPUUPBPTechnique::Surface ? resources.point_acceleration_structure : resources.medium_point_acceleration_structure;
  if ((resources.counter_buffer == kInvalidIndex) || (acceleration_structure == kInvalidIndex) || (dispatch_index >= resources.camera_vertex_capacity)) {
    return;
  }
  const uint camera_vertex_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraVertex * 4u), resources.camera_vertex_capacity);
  if (dispatch_index >= camera_vertex_count) {
    return;
  }
  const GPUUPBPVertex camera_vertex = upbp_load_vertex(resources.vertex_buffer, dispatch_index);
  if (((camera_vertex.flags & GPUUPBPVertexFlags::Valid) == 0u) || (camera_vertex.path_length == 0u) || (upbp_vertex_is_density_connectible(camera_vertex) == false)) {
    return;
  }
#if defined(ETX_UPBP_SURFACE_VARIANT)
  Material camera_material = (Material)0;
  if ((camera_vertex.material_index == kInvalidIndex) || (try_load_material_full(camera_vertex.material_index, camera_material) == false) ||
      (upbp_surface_stage_matches_material(camera_material.cls) == false)) {
    return;
  }
  upbp_surface_camera_vertices[group_thread_index] = camera_vertex;
#endif
  const float radius = selected_technique == GPUUPBPTechnique::Surface ? resources.iteration.surface_radius : resources.iteration.pp3d_radius;
  const float acceleration_radius =
    selected_technique == GPUUPBPTechnique::Surface ? resources.iteration.surface_radius : max(resources.iteration.pp3d_radius, resources.iteration.pb2d_radius);
  if ((radius <= 0.0f) || (acceleration_radius <= 0.0f)) {
    return;
  }
  SpectralResponse accumulated = spectral_response_zero(spectral_response_as_query(upbp_unpack_spectral_response(camera_vertex.throughput)));
  RayDesc ray = (RayDesc)0;
  ray.Origin = camera_vertex.position;
  ray.Direction = float3(1.0f, 0.0f, 0.0f);
  ray.TMin = 0.0f;
  ray.TMax = 1.0e-7f;
  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
  query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(acceleration_structure)], RAY_FLAG_FORCE_NON_OPAQUE, 0xff, ray);
  const uint point_buffer = selected_technique == GPUUPBPTechnique::Surface ? resources.density_output_surface_point_buffer : resources.density_output_medium_point_buffer;
  const uint point_count = selected_technique == GPUUPBPTechnique::Surface ? resources.density_output_surface_point_capacity : resources.density_output_medium_point_capacity;
  [loop] while (query.Proceed()) {
    if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
      const uint primitive_index = query.CandidatePrimitiveIndex();
      if ((primitive_index < point_count) && (point_buffer != kInvalidIndex)) {
        upbp_evaluate_point_indices(resources, dispatch_index, group_thread_index, point_buffer, primitive_index, selected_technique, radius, accumulated);
      }
    }
  }
  if (spectral_response_is_zero(accumulated)) {
    return;
  }
  const uint local_path_index = camera_vertex.global_path_index - resources.iteration.camera_batch_offset;
  if (local_path_index >= resources.iteration.camera_batch_count) {
    return;
  }
  const GPUWavefrontPathState camera_state = wavefront_load_path_state(wavefront_resources.camera_state_buffer, local_path_index);
  if (upbp_density_contribution_finite(resources, selected_technique, camera_vertex.global_path_index, accumulated)) {
    wavefront_film_add(camera_state.pixel_index, wavefront_spectral_estimate(accumulated, spectral_response_as_query(accumulated)));
  }
}

#if defined(ETX_UPBP_SURFACE_VARIANT)
uint upbp_surface_partition_offset(uint total_count, uint partition_index) {
  const uint base_count = total_count / kGPUUPBPSurfacePartitionCount;
  return partition_index * base_count + min(partition_index, total_count % kGPUUPBPSurfacePartitionCount);
}

void upbp_evaluate_surface_point_merge_group(uint dispatch_index, uint group_thread_index) {
  GPUWavefrontResources wavefront_resources = wavefront_load_resources();
  GPUUPBPResources resources = upbp_load_resources(wavefront_resources);
  if (group_thread_index == 0u) {
    upbp_surface_query_valid = 0u;
    uint vertex_index = dispatch_index;
    if (constants.work_queue_index == GPUUPBPDensityQueryMode::Compacted) {
      const uint query_family = ETX_UPBP_SURFACE_QUERY_FAMILY;
      const uint query_count = WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load((GPUUPBPCounterIndex::CameraSurfaceVariousQuery + query_family) * sizeof(uint));
      vertex_index = ((dispatch_index < query_count) && (resources.point_buffer != kInvalidIndex))
                       ? WAVEFRONT_RO_BUFFER(resources.point_buffer).Load((query_family * resources.point_capacity + dispatch_index) * sizeof(uint))
                       : kInvalidIndex;
    }
    if ((resources.counter_buffer != kInvalidIndex) && (resources.point_acceleration_structure != kInvalidIndex) &&
        (resources.density_output_surface_point_buffer != kInvalidIndex) && (vertex_index < resources.camera_vertex_capacity)) {
      const uint camera_vertex_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraVertex * sizeof(uint)), resources.camera_vertex_capacity);
      if (vertex_index < camera_vertex_count) {
        const GPUUPBPVertex camera_vertex = upbp_load_vertex(resources.vertex_buffer, vertex_index);
        Material camera_material = (Material)0;
        if (((camera_vertex.flags & GPUUPBPVertexFlags::Valid) != 0u) && (camera_vertex.path_length > 0u) && upbp_vertex_is_density_connectible(camera_vertex) &&
            (camera_vertex.material_index != kInvalidIndex) && try_load_material_full(camera_vertex.material_index, camera_material) &&
            upbp_surface_stage_matches_material(camera_material.cls)) {
          upbp_surface_camera_vertex = camera_vertex;
          upbp_surface_query_valid = 1u;
        }
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (upbp_surface_query_valid == 0u) {
    return;
  }

  SpectralResponse accumulated = spectral_response_zero(spectral_response_as_query(upbp_unpack_spectral_response(upbp_surface_camera_vertex.throughput)));
  RayDesc ray = (RayDesc)0;
  ray.Origin = upbp_surface_camera_vertex.position;
  ray.Direction = float3(1.0f, 0.0f, 0.0f);
  ray.TMin = 0.0f;
  ray.TMax = 1.0e-7f;
  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
  const uint partition_mask = group_thread_index < kGPUUPBPSurfacePartitionCount ? (1u << group_thread_index) : 1u;
  query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(resources.point_acceleration_structure)], RAY_FLAG_FORCE_NON_OPAQUE, partition_mask, ray);
  for (;;) {
    if (group_thread_index < kGPUUPBPSurfacePartitionCount) {
      uint partition_count = 0u;
      uint partition_exhausted = 0u;
      [loop] while (partition_count < 8u) {
        if (query.Proceed() == false) {
          partition_exhausted = 1u;
          break;
        }
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
          const uint primitive_index = upbp_surface_partition_offset(resources.density_output_surface_point_capacity, group_thread_index) + query.CandidatePrimitiveIndex();
          if (primitive_index < resources.density_output_surface_point_capacity) {
            upbp_surface_point_indices[group_thread_index * 8u + partition_count] = primitive_index;
            ++partition_count;
          }
        }
      }
      upbp_surface_partition_counts[group_thread_index] = partition_count;
      upbp_surface_partition_exhausted[group_thread_index] = partition_exhausted;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint partition_index = group_thread_index / 8u;
    const uint partition_item_index = group_thread_index % 8u;
    if (partition_item_index < upbp_surface_partition_counts[partition_index]) {
      const GPUUPBPVertex light_vertex =
        upbp_unpack_density_point(upbp_load_density_point(resources.density_output_surface_point_buffer, upbp_surface_point_indices[group_thread_index]));
      upbp_evaluate_point_vertex(resources, upbp_surface_camera_vertex, light_vertex, GPUUPBPTechnique::Surface, resources.iteration.surface_radius, accumulated);
    }
    GroupMemoryBarrierWithGroupSync();
    uint exhausted_partition_count = 0u;
    [unroll] for (uint partition = 0u; partition < kGPUUPBPSurfacePartitionCount; ++partition) {
      exhausted_partition_count += upbp_surface_partition_exhausted[partition];
    }
    if (exhausted_partition_count == kGPUUPBPSurfacePartitionCount) {
      break;
    }
  }
  upbp_surface_lane_contributions[group_thread_index] = upbp_pack_spectral_response(accumulated);
  GroupMemoryBarrierWithGroupSync();
  if (group_thread_index == 0u) {
    SpectralResponse total = spectral_response_zero(spectral_response_as_query(upbp_unpack_spectral_response(upbp_surface_camera_vertex.throughput)));
    [unroll] for (uint lane_index = 0u; lane_index < 64u; ++lane_index) {
      total = spectral_response_add(total, upbp_unpack_spectral_response(upbp_surface_lane_contributions[lane_index]));
    }
    if (spectral_response_is_zero(total)) {
      return;
    }
    const uint local_path_index = upbp_surface_camera_vertex.global_path_index - resources.iteration.camera_batch_offset;
    if (local_path_index < resources.iteration.camera_batch_count) {
      const GPUWavefrontPathState camera_state = wavefront_load_path_state(wavefront_resources.camera_state_buffer, local_path_index);
      if (upbp_density_contribution_finite(resources, GPUUPBPTechnique::Surface, upbp_surface_camera_vertex.global_path_index, total)) {
        wavefront_film_add(camera_state.pixel_index, wavefront_spectral_estimate(total, spectral_response_as_query(total)));
      }
    }
  }
}
#endif

#if !defined(ETX_UPBP_SURFACE_VARIANT)
void upbp_submit_camera_contribution(GPUWavefrontResources wavefront_resources, GPUUPBPResources resources, uint technique_flag, uint global_path_index,
  SpectralResponse contribution) {
  if (spectral_response_is_zero(contribution) || (global_path_index < resources.iteration.camera_batch_offset)) {
    return;
  }
  const uint local_path_index = global_path_index - resources.iteration.camera_batch_offset;
  if (local_path_index >= resources.iteration.camera_batch_count) {
    return;
  }
  if (upbp_density_contribution_finite(resources, technique_flag, global_path_index, contribution) == false) {
    return;
  }
  const GPUWavefrontPathState camera_state = wavefront_load_path_state(wavefront_resources.camera_state_buffer, local_path_index);
  wavefront_film_add(camera_state.pixel_index, wavefront_spectral_estimate(contribution, spectral_response_as_query(contribution)));
}

void upbp_evaluate_pb2d_vertex(GPUUPBPResources resources, GPUUPBPBeam camera_beam, UPBPGPUPreparedBeam prepared_camera, GPUUPBPVertex light_vertex,
  inout SpectralResponse accumulated) {
  const uint path_length = light_vertex.path_length + camera_beam.path_length + 1u;
  if (((light_vertex.flags & GPUUPBPVertexFlags::Valid) == 0u) || (light_vertex.path_length == 0u) || (upbp_vertex_is_medium(light_vertex) == false) ||
      (light_vertex.medium_index == kInvalidIndex) || (light_vertex.medium_index != upbp_load_interval(resources.interval_buffer, camera_beam.interval_index).medium_index) ||
      (path_length < load_scene_options_min_path_length()) || (path_length > load_scene_options_max_path_length())) {
    return;
  }
  SpectralResponse light_throughput = upbp_unpack_spectral_response(light_vertex.throughput);
  if (spectral_query_compatible(spectral_response_as_query(light_throughput), spectral_response_as_query(prepared_camera.source_throughput)) == false) {
    return;
  }
  UPBPGPUPointBeamIntersection intersection = (UPBPGPUPointBeamIntersection)0;
  if (upbp_intersect_point_beam(light_vertex.position, camera_beam, resources.iteration.pb2d_radius, intersection) == false) {
    return;
  }
  UPBPGPUPartialBeamVertex partial_camera = (UPBPGPUPartialBeamVertex)0;
  if (upbp_partial_prepared_beam_vertex(resources, prepared_camera, camera_beam, intersection.beam_distance, partial_camera) == false) {
    return;
  }
  SpectralResponse light_pre_collision = spectral_response_zero(spectral_response_as_query(light_throughput));
  if (upbp_medium_pre_collision_throughput(light_vertex, light_throughput, light_pre_collision) == false) {
    return;
  }
  SpectralResponse light_scattering = spectral_response_zero(spectral_response_as_query(light_throughput));
  float phase_function_g = 0.0f;
  if (upbp_vertex_medium_properties(light_vertex, spectral_response_as_query(light_throughput), light_scattering, phase_function_g) == false) {
    return;
  }
  const float phase = upbp_medium_phase(phase_function_g, camera_beam.direction, -light_vertex.w_i);
  const float cosine = dot(camera_beam.direction, -light_vertex.w_i);
  const float sin_theta = sqrt(max(0.0f, 1.0f - cosine * cosine));
  const float kernel_value = upbp_kernel_value(resources.iteration.kernel, 2u, resources.iteration.pb2d_radius, intersection.distance_squared);
  const float mis_weight =
    upbp_point_merge_mis_weight(resources.iteration, GPUUPBPTechnique::PB2D, light_vertex.arrival_weights, partial_camera.weights, light_vertex, phase, phase, sin_theta);
  if ((phase <= 0.0f) || (kernel_value <= 0.0f) || (mis_weight <= 0.0f)) {
    return;
  }
  const GPUUPBPInterval camera_interval = upbp_load_interval(resources.interval_buffer, camera_beam.interval_index);
  SpectralResponse scattering = spectral_response_zero(spectral_response_as_query(partial_camera.throughput));
  SpectralResponse extinction = spectral_response_zero(spectral_response_as_query(partial_camera.throughput));
  float density = 0.0f;
  float camera_phase_function_g = 0.0f;
  if (upbp_interval_medium_properties(camera_interval, spectral_response_as_query(partial_camera.throughput),
        camera_beam.origin + camera_beam.direction * intersection.beam_distance, scattering, extinction, density, camera_phase_function_g) == false) {
    return;
  }
  const float scale = phase * kernel_value * mis_weight / float(resources.iteration.global_light_path_count);
  accumulated =
    spectral_response_add(accumulated, spectral_response_mul(spectral_response_mul(light_pre_collision, partial_camera.throughput), spectral_response_mul(scattering, scale)));
}

void upbp_evaluate_pb2d_candidate(GPUUPBPResources resources, GPUUPBPBeam camera_beam, UPBPGPUPreparedBeam prepared_camera, GPUUPBPPoint point_record,
  inout SpectralResponse accumulated) {
  if (point_record.vertex_index >= (resources.camera_vertex_capacity + resources.light_vertex_capacity)) {
    return;
  }
  upbp_evaluate_pb2d_vertex(resources, camera_beam, prepared_camera, upbp_load_vertex(resources.vertex_buffer, point_record.vertex_index), accumulated);
}

void upbp_evaluate_bp2d_prepared_candidate(GPUUPBPResources resources, GPUUPBPVertex camera_vertex, GPUUPBPBeam light_beam, GPUUPBPInterval light_interval,
  UPBPGPUPreparedBeam prepared_light, UPBPGPUPointBeamIntersection intersection, SpectralResponse camera_pre_collision, SpectralResponse camera_scattering,
  float camera_phase_function_g, inout SpectralResponse accumulated) {
  UPBPGPUPartialBeamVertex partial_light = (UPBPGPUPartialBeamVertex)0;
  if (upbp_partial_prepared_beam_vertex_with_interval(resources, prepared_light, light_beam, light_interval, intersection.beam_distance, partial_light) == false) {
    return;
  }
  if (spectral_query_compatible(spectral_response_as_query(camera_pre_collision), spectral_response_as_query(partial_light.throughput)) == false) {
    return;
  }
  const float phase = upbp_medium_phase(camera_phase_function_g, camera_vertex.w_i, -light_beam.direction);
  const float cosine = dot(camera_vertex.w_i, -light_beam.direction);
  const float sin_theta = sqrt(max(0.0f, 1.0f - cosine * cosine));
  const float kernel_value = upbp_kernel_value(resources.iteration.kernel, 2u, resources.iteration.bp2d_radius, intersection.distance_squared);
  const float mis_weight =
    upbp_point_merge_mis_weight(resources.iteration, GPUUPBPTechnique::BP2D, partial_light.weights, camera_vertex.arrival_weights, camera_vertex, phase, phase, sin_theta);
  if ((phase <= 0.0f) || (kernel_value <= 0.0f) || (mis_weight <= 0.0f)) {
    return;
  }
  const float scale = phase * kernel_value * mis_weight / float(resources.iteration.global_light_path_count);
  accumulated = spectral_response_add(accumulated,
    spectral_response_mul(spectral_response_mul(partial_light.throughput, camera_pre_collision), spectral_response_mul(camera_scattering, scale)));
}

void upbp_evaluate_bp2d_density_candidate(GPUUPBPResources resources, GPUUPBPVertex camera_vertex, GPUUPBPDensityBeam record, UPBPGPUPointBeamIntersection intersection,
  SpectralResponse camera_pre_collision, SpectralResponse camera_scattering, float camera_phase_function_g, inout SpectralResponse accumulated) {
  if ((record.flags & GPUUPBPDensityBeamFlags::Valid) == 0u) {
    return;
  }
  upbp_evaluate_bp2d_prepared_candidate(resources, camera_vertex, record.beam, record.interval, upbp_unpack_density_beam(record), intersection, camera_pre_collision,
    camera_scattering, camera_phase_function_g, accumulated);
}

void upbp_evaluate_bb1d_prepared_candidate(GPUUPBPResources resources, GPUUPBPBeam camera_beam, UPBPGPUPreparedBeam prepared_camera, GPUUPBPBeam light_beam,
  GPUUPBPInterval camera_interval, GPUUPBPVertex context_vertex, GPUUPBPInterval light_interval, UPBPGPUPreparedBeam prepared_light, UPBPGPUBeamBeamIntersection intersection,
  inout SpectralResponse accumulated) {
  UPBPGPUPartialBeamVertex partial_light = (UPBPGPUPartialBeamVertex)0;
  UPBPGPUPartialBeamVertex partial_camera = (UPBPGPUPartialBeamVertex)0;
  if ((upbp_partial_prepared_beam_vertex_with_interval(resources, prepared_light, light_beam, light_interval, intersection.first_distance, partial_light) == false) ||
      (upbp_partial_prepared_beam_vertex(resources, prepared_camera, camera_beam, intersection.second_distance, partial_camera) == false) ||
      (spectral_query_compatible(spectral_response_as_query(partial_light.throughput), spectral_response_as_query(partial_camera.throughput)) == false)) {
    return;
  }
  SpectralResponse scattering = spectral_response_zero(spectral_response_as_query(partial_camera.throughput));
  SpectralResponse extinction = spectral_response_zero(spectral_response_as_query(partial_camera.throughput));
  float density = 0.0f;
  float phase_function_g = 0.0f;
  if (upbp_interval_medium_properties(camera_interval, spectral_response_as_query(partial_camera.throughput),
        camera_beam.origin + camera_beam.direction * intersection.second_distance, scattering, extinction, density, phase_function_g) == false) {
    return;
  }
  const float phase = upbp_medium_phase(phase_function_g, camera_beam.direction, -light_beam.direction);
  const float kernel_value = upbp_kernel_value(resources.iteration.kernel, 1u, resources.iteration.bb1d_radius, intersection.distance_squared) / intersection.sin_theta;
  const float mis_weight =
    upbp_point_merge_mis_weight(resources.iteration, GPUUPBPTechnique::BB1D, partial_light.weights, partial_camera.weights, context_vertex, phase, phase, intersection.sin_theta);
  if ((phase <= 0.0f) || (kernel_value <= 0.0f) || (mis_weight <= 0.0f) || (resources.iteration.bb1d_light_path_count == 0u) ||
      (resources.iteration.beam_selection_probability <= 0.0f)) {
    return;
  }
  const float estimator_normalization = rcp(float(resources.iteration.bb1d_light_path_count) * resources.iteration.beam_selection_probability);
  const float scale = phase * kernel_value * estimator_normalization * mis_weight;
  accumulated =
    spectral_response_add(accumulated, spectral_response_mul(spectral_response_mul(partial_light.throughput, partial_camera.throughput), spectral_response_mul(scattering, scale)));
}

void upbp_evaluate_bb1d_density_candidate(GPUUPBPResources resources, GPUUPBPBeam camera_beam, UPBPGPUPreparedBeam prepared_camera, GPUUPBPDensityBeam record,
  GPUUPBPInterval camera_interval, GPUUPBPVertex context_vertex, UPBPGPUBeamBeamIntersection intersection, inout SpectralResponse accumulated) {
  if (((record.flags & GPUUPBPDensityBeamFlags::Valid) == 0u) || ((record.beam.flags & GPUUPBPBeamFlags::SelectedForBB1D) == 0u)) {
    return;
  }
  upbp_evaluate_bb1d_prepared_candidate(resources, camera_beam, prepared_camera, record.beam, camera_interval, context_vertex, record.interval, upbp_unpack_density_beam(record),
    intersection, accumulated);
}

GPUUPBPBeam upbp_beam_from_reference(GPUUPBPBeamReference reference) {
  GPUUPBPBeam result = (GPUUPBPBeam)0;
  result.origin = reference.origin;
  result.length = reference.length;
  result.direction = reference.direction;
  result.path_length = reference.path_length;
  return result;
}

bool upbp_bp2d_density_candidate_intersects(GPUUPBPResources resources, GPUUPBPBeamReference reference, GPUUPBPVertex camera_vertex,
  out UPBPGPUPointBeamIntersection intersection) {
  intersection = (UPBPGPUPointBeamIntersection)0;
  const uint path_length = reference.path_length + 1u + camera_vertex.path_length;
  if ((reference.medium_index != camera_vertex.medium_index) || (path_length < load_scene_options_min_path_length()) || (path_length > load_scene_options_max_path_length())) {
    return false;
  }
  return upbp_intersect_point_beam(camera_vertex.position, upbp_beam_from_reference(reference), resources.iteration.bp2d_radius, intersection);
}

bool upbp_bb1d_density_candidate_intersects(GPUUPBPResources resources, GPUUPBPBeamReference reference, GPUUPBPBeam camera_beam, uint camera_medium_index,
  out UPBPGPUBeamBeamIntersection intersection) {
  intersection = (UPBPGPUBeamBeamIntersection)0;
  const uint path_length = reference.path_length + camera_beam.path_length + 2u;
  if ((camera_medium_index == kInvalidIndex) || (camera_medium_index != reference.medium_index) || (path_length < load_scene_options_min_path_length()) ||
      (path_length > load_scene_options_max_path_length())) {
    return false;
  }
  return upbp_intersect_beams(upbp_beam_from_reference(reference), camera_beam, resources.iteration.bb1d_radius, intersection);
}

GPUUPBPBeamReference upbp_load_bb1d_beam_reference(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPDensityBeamStride;
  GPUUPBPBeamReference result = (GPUUPBPBeamReference)0;
  result.origin = wavefront_load_float3(buffer, base_offset + kGPUUPBPDensityBeamBeamOffset + kGPUUPBPBeamOriginOffset);
  result.length = asfloat(buffer.Load(base_offset + kGPUUPBPDensityBeamBeamOffset + kGPUUPBPBeamLengthOffset));
  result.direction = wavefront_load_float3(buffer, base_offset + kGPUUPBPDensityBeamBeamOffset + kGPUUPBPBeamDirectionOffset);
  result.path_length = buffer.Load(base_offset + kGPUUPBPDensityBeamBeamOffset + kGPUUPBPBeamPathLengthOffset);
  result.medium_index = buffer.Load(base_offset + kGPUUPBPDensityBeamIntervalOffset + kGPUUPBPIntervalMediumIndexOffset);
  return result;
}

SpectralResponse upbp_local_emitter_radiance(uint emitter_index, SpectralQuery spect, float3 source_position, float3 target_position, float2 uv, bool directly_visible,
  out float pdf_area, out float pdf_dir, out float pdf_dir_out) {
  (void)directly_visible;
  pdf_area = 0.0f;
  pdf_dir = 0.0f;
  pdf_dir_out = 0.0f;
  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  GPUEmitterProfileABIData emitter_profile = (GPUEmitterProfileABIData)0;
  if ((try_load_emitter_instance(emitter_index, emitter_instance) == false) || (try_load_emitter_profile(emitter_instance.emitter_profile_index, emitter_profile) == false) ||
      (emitter_instance.emitter_class != EmitterClass::Area)) {
    return spectral_response_zero(spect);
  }
  const TriangleData triangle_data = load_triangle(WAVEFRONT_RO_BUFFER(constants.scene.triangles), emitter_instance.triangle_index);
  const float3 geo_normal = scene_instance_transform_geometric_normal(load_scene_instance(emitter_instance.instance_index), triangle_data.geo_n);
  Material material = (Material)0;
  if (try_load_material_full(triangle_data.material_index, material) == false) {
    return spectral_response_zero(spect);
  }
  const float3 target_delta = target_position - source_position;
  if (dot(geo_normal, target_delta) >= 0.0f) {
    return spectral_response_zero(spect);
  }
  pdf_area = emitter_instance.triangle_area > 0.0f ? rcp(emitter_instance.triangle_area) : 0.0f;
  const float distance_squared = dot(target_delta, target_delta);
  if ((pdf_area <= 0.0f) || (distance_squared <= 0.0f)) {
    return spectral_response_zero(spect);
  }
  const float cosine = max(0.0f, dot(-target_delta, geo_normal)) * rsqrt(distance_squared);
  const float exponent = scene_math_shared_collimation_to_exponent(material.emission_collimation);
  if (cosine > kEpsilon) {
    pdf_dir = pdf_area * distance_squared / cosine;
    pdf_dir_out = pdf_area * scene_math_shared_collimated_direction_pdf(cosine, exponent);
    const float emission_scale = scene_math_shared_collimated_emission_scale(cosine, exponent);
    return spectral_response_mul(evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, uv, spect), emission_scale);
  }
  return spectral_response_zero(spect);
}

float upbp_distant_emitter_sample_pdf(uint emitter_index, float3 in_direction) {
  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  GPUEmitterProfileABIData emitter_profile = (GPUEmitterProfileABIData)0;
  if ((try_load_emitter_instance(emitter_index, emitter_instance) == false) || (try_load_emitter_profile(emitter_instance.emitter_profile_index, emitter_profile) == false)) {
    return 0.0f;
  }
  const float discrete_pdf = emitter_discrete_pdf(emitter_index);
  if (emitter_instance.emitter_class == EmitterClass::Directional) {
    float cosine_threshold = emitter_profile.emitter_angular_size_cosine;
    if (cosine_threshold <= 0.0f) {
      cosine_threshold = 1.0f;
    }
    return emitter_access_accepts_direction(in_direction, emitter_profile.emitter_direction, cosine_threshold) ? discrete_pdf : 0.0f;
  }
  if (emitter_instance.emitter_class != EmitterClass::Environment) {
    return 0.0f;
  }
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  EmitterAccess access = (EmitterAccess)0;
  if (emitter_access_try_load_distant(context, emitter_index, in_direction, access) == false) {
    return 0.0f;
  }
  const float2 uv = emitter_access_environment_uv(context, access, in_direction);
  float image_pdf = 0.0f;
  float4 image_value = float4(0.0f, 0.0f, 0.0f, 0.0f);
  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(constants.scene.images);
  if (image_evaluate_try_rgba(image_context, access.emission_image_index, uv, image_pdf, image_value) == false) {
    return 0.0f;
  }
  const bool is_atmosphere = (access.emitter_profile_meta & EmitterProfileMeta::Atmosphere) != 0u;
  return discrete_pdf * projection_environment_image_pdf_to_solid_angle(image_pdf, uv, projection_environment_mode(is_atmosphere));
}

bool upbp_camera_prefix_is_specular(GPUUPBPResources resources, GPUUPBPVertex endpoint);

SpectralResponse upbp_distant_emitter_radiance(uint emitter_index, SpectralQuery spect, float3 direction, bool directly_visible, out float pdf_area, out float pdf_dir,
  out float pdf_dir_out) {
  pdf_area = 0.0f;
  pdf_dir = 0.0f;
  pdf_dir_out = 0.0f;
  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  GPUEmitterProfileABIData emitter_profile = (GPUEmitterProfileABIData)0;
  if ((try_load_emitter_instance(emitter_index, emitter_instance) == false) || (try_load_emitter_profile(emitter_instance.emitter_profile_index, emitter_profile) == false) ||
      (emitter_instance.emitter_class == EmitterClass::Area)) {
    return spectral_response_zero(spect);
  }

  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  pdf_area = rcp(kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius);
  SpectralResponse result = spectral_response_zero(spect);
  if (emitter_instance.emitter_class == EmitterClass::Directional) {
    const float directional_cosine = dot(direction, emitter_profile.emitter_direction);
    if ((directly_visible == false) || (emitter_profile.emitter_angular_size_cosine >= 1.0f) || (directional_cosine < emitter_profile.emitter_angular_size_cosine)) {
      return result;
    }
    pdf_dir = 1.0f;
    const float sin_half_angle = sqrt(max(0.0f, 1.0f - emitter_profile.emitter_angular_size_cosine * emitter_profile.emitter_angular_size_cosine));
    const float equivalent_disk_size = emitter_profile.emitter_angular_size_cosine > kEpsilon ? 2.0f * sin_half_angle / emitter_profile.emitter_angular_size_cosine : 0.0f;
    const float2 uv = disk_uv(emitter_profile.emitter_direction, direction, equivalent_disk_size, emitter_profile.emitter_angular_size_cosine);
    result = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, uv, spect);
    const SpectralResponse spectrum_value = load_scene_spectrum_or_zero(emitter_profile.emission_spectrum_index, spect);
    const float normalization = kDoublePi * (1.0f - emitter_profile.emitter_angular_size_cosine);
    if (spectral_response_is_zero(spectrum_value) || (normalization <= 0.0f)) {
      return spectral_response_zero(spect);
    }
    result = spectral_response_div(result, spectral_response_mul(spectrum_value, normalization));
  } else {
    pdf_dir = upbp_distant_emitter_sample_pdf(emitter_index, direction) / max(kEpsilon, emitter_discrete_pdf(emitter_index));
    result = evaluate_distant_emission_spectral(emitter_index, direction, spect);
  }
  pdf_dir_out = pdf_area * pdf_dir;
  return result;
}

void upbp_evaluate_environment_direct_hit(uint path_index, GPUWavefrontResources wavefront_resources, GPUUPBPResources resources) {
  if (path_index >= resources.iteration.camera_batch_count) {
    return;
  }
  const GPUUPBPPathState path_state = upbp_load_path_state(resources.path_state_buffer, upbp_path_state_index(resources, true, path_index));
  if (((path_state.flags & (GPUUPBPPathStateFlags::Valid | GPUUPBPPathStateFlags::HasTerminalSegment)) !=
        (GPUUPBPPathStateFlags::Valid | GPUUPBPPathStateFlags::HasTerminalSegment)) ||
      (path_state.last_vertex_index >= resources.camera_vertex_capacity) || (path_state.current_segment_index >= resources.camera_segment_capacity) ||
      (path_state.current_interval_index >= resources.camera_interval_capacity)) {
    return;
  }
  const GPUWavefrontPathState state = wavefront_load_path_state(wavefront_resources.camera_state_buffer, path_index);
  const GPUUPBPVertex previous = upbp_load_vertex(resources.vertex_buffer, path_state.last_vertex_index);
  const GPUUPBPSegment segment = upbp_load_segment(resources.segment_buffer, path_state.current_segment_index);
  const GPUUPBPInterval terminal_interval = upbp_load_interval(resources.interval_buffer, path_state.current_interval_index);
  if (((previous.flags & GPUUPBPVertexFlags::Valid) == 0u) || ((segment.flags & GPUUPBPSegmentFlags::Terminal) == 0u) ||
      ((terminal_interval.flags & GPUUPBPIntervalFlags::Escape) == 0u)) {
    return;
  }
  const uint endpoint_path_length = path_state.path_length + 1u;
  if ((endpoint_path_length < load_scene_options_min_path_length()) || (endpoint_path_length > load_scene_options_max_path_length())) {
    return;
  }

  EmitterAccessGPUContext emitter_context = make_scene_emitter_access_gpu_context();
  uint emitter_instance_count = 0u;
  uint environment_emitter_count = 0u;
  if (emitter_access_try_load_environment_state(emitter_context, emitter_instance_count, environment_emitter_count) == false) {
    return;
  }
  (void)emitter_instance_count;
  for (uint local_emitter_index = 0u; local_emitter_index < environment_emitter_count; ++local_emitter_index) {
    uint emitter_index = kInvalidIndex;
    if (emitter_access_try_load_environment_emitter(emitter_context, local_emitter_index, emitter_index) == false) {
      continue;
    }
    float pdf_area = 0.0f;
    float pdf_dir = 0.0f;
    float pdf_dir_out = 0.0f;
    const SpectralResponse radiance = upbp_distant_emitter_radiance(emitter_index, state.spect, state.ray.d, path_state.path_length == 0u, pdf_area, pdf_dir, pdf_dir_out);
    const float emitter_selection_pdf = emitter_discrete_pdf(emitter_index);
    if (spectral_response_is_zero(radiance) || (pdf_area <= 0.0f) || (pdf_dir <= 0.0f) || (pdf_dir_out <= 0.0f) || (emitter_selection_pdf <= 0.0f)) {
      continue;
    }

    GPUUPBPVertex endpoint = (GPUUPBPVertex)0;
    endpoint.throughput = upbp_pack_spectral_response(state.throughput);
    endpoint.position = terminal_interval.end_position;
    endpoint.medium_index = state.medium_index;
    endpoint.w_i = state.ray.d;
    endpoint.incident_medium_index = state.medium_index;
    endpoint.outgoing_medium_index = state.medium_index;
    endpoint.endpoint_pdf_area = pdf_area;
    endpoint.endpoint_pdf_sample = emitter_selection_pdf;
    endpoint.endpoint_pdf_direction = pdf_dir;
    endpoint.emitter_index = emitter_index;
    endpoint.flags = GPUUPBPVertexFlags::Valid | GPUUPBPVertexFlags::Emitter | GPUUPBPVertexFlags::Connectible | GPUUPBPVertexFlags::DistantEndpoint;
    endpoint.previous_vertex_index = path_state.last_vertex_index;
    endpoint.incoming_segment_index = path_state.current_segment_index;
    endpoint.path_length = endpoint_path_length;
    endpoint.global_path_index = path_state.global_path_index;
    GPUUPBPRecursiveState recursive_state = path_state.recursive_state;
    if (upbp_complete_recursive_arrival(previous, endpoint, segment, resources.iteration, endpoint_path_length, recursive_state) == false) {
      upbp_mark_failed_connection(resources, path_state.global_path_index, 4u, endpoint_path_length + 1u, 1u, GPUUPBPConnectionTrackingFailure::None);
      continue;
    }
    endpoint.arrival_weights = recursive_state.weights;

    float mis_weight = 0.0f;
    if ((resources.iteration.technique_mask & GPUUPBPTechnique::BPT) != 0u) {
      if (((resources.iteration.flags & GPUUPBPIterationFlags::MultipleImportanceSampling) == 0u) || (endpoint_path_length == 1u)) {
        mis_weight = 1.0f;
      } else {
        const float log_reverse_ray_pdf = segment.log_transport_pdf_reverse + (upbp_vertex_is_medium(previous) ? previous.log_medium_event_density : 0.0f);
        const bool previous_delta = (endpoint.arrival_weights.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u;
        const float log_shared = previous_delta ? kUPBPLogZero : log(emitter_selection_pdf) + log(pdf_area) + endpoint.arrival_weights.log_d_shared;
        const float log_bpt = upbp_log_is_zero(endpoint.arrival_weights.log_d_bpt)
                                ? kUPBPLogZero
                                : log(emitter_selection_pdf) + log(pdf_dir_out) + log_reverse_ray_pdf + endpoint.arrival_weights.log_d_bpt;
        const float log_denominator = upbp_log_add(0.0f, upbp_log_add(log_shared, log_bpt));
        mis_weight = isfinite(log_denominator) ? exp(-log_denominator) : 0.0f;
      }
    } else if (upbp_camera_prefix_is_specular(resources, endpoint)) {
      mis_weight = 1.0f;
    }
    if (mis_weight > 0.0f) {
      wavefront_film_add(state.pixel_index, wavefront_spectral_estimate(spectral_response_mul(spectral_response_mul(state.throughput, radiance), mis_weight), state.spect));
    }
  }
}

bool upbp_camera_prefix_is_specular(GPUUPBPResources resources, GPUUPBPVertex endpoint) {
  uint vertex_index = endpoint.previous_vertex_index;
  [loop] while (vertex_index != kInvalidIndex) {
    if (vertex_index >= resources.camera_vertex_capacity) {
      return false;
    }
    const GPUUPBPVertex vertex = upbp_load_vertex(resources.vertex_buffer, vertex_index);
    if (((vertex.flags & GPUUPBPVertexFlags::Valid) == 0u) || ((vertex.path_length > 0u) && (upbp_vertex_is_delta(vertex) == false))) {
      return false;
    }
    if (vertex.path_length == 0u) {
      return true;
    }
    vertex_index = vertex.previous_vertex_index;
  }
  return false;
}

void upbp_evaluate_direct_hit_dispatch(uint dispatch_index) {
  GPUWavefrontResources wavefront_resources = wavefront_load_resources();
  GPUUPBPResources resources = upbp_load_resources(wavefront_resources);
  if (((resources.iteration.flags & GPUUPBPIterationFlags::EvaluateCameraIndependentTerms) == 0u) || (scene_strategy_enabled(kSceneStrategyDirectHit) == false) ||
      (resources.counter_buffer == kInvalidIndex)) {
    return;
  }
  upbp_evaluate_environment_direct_hit(dispatch_index, wavefront_resources, resources);
  if (dispatch_index >= resources.camera_vertex_capacity) {
    return;
  }
  const uint camera_vertex_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraVertex * 4u), resources.camera_vertex_capacity);
  if (dispatch_index >= camera_vertex_count) {
    return;
  }
  const GPUUPBPVertex emitter_vertex = upbp_load_vertex(resources.vertex_buffer, dispatch_index);
  if (((emitter_vertex.flags & (GPUUPBPVertexFlags::Valid | GPUUPBPVertexFlags::Emitter | GPUUPBPVertexFlags::Surface)) !=
        (GPUUPBPVertexFlags::Valid | GPUUPBPVertexFlags::Emitter | GPUUPBPVertexFlags::Surface)) ||
      (emitter_vertex.path_length == 0u) || (emitter_vertex.previous_vertex_index >= resources.camera_vertex_capacity) ||
      (emitter_vertex.incoming_segment_index >= resources.camera_segment_capacity) || (emitter_vertex.path_length < load_scene_options_min_path_length()) ||
      (emitter_vertex.path_length > load_scene_options_max_path_length())) {
    return;
  }
  const GPUUPBPVertex previous = upbp_load_vertex(resources.vertex_buffer, emitter_vertex.previous_vertex_index);
  const GPUUPBPSegment segment = upbp_load_segment(resources.segment_buffer, emitter_vertex.incoming_segment_index);
  if (((previous.flags & GPUUPBPVertexFlags::Valid) == 0u) || ((segment.flags & GPUUPBPSegmentFlags::Valid) == 0u)) {
    return;
  }
  const SpectralResponse throughput = upbp_unpack_spectral_response(emitter_vertex.throughput);
  const SpectralQuery spect = spectral_response_as_query(throughput);
  float pdf_area = 0.0f;
  float pdf_dir = 0.0f;
  float pdf_dir_out = 0.0f;
  const SpectralResponse radiance = upbp_local_emitter_radiance(emitter_vertex.emitter_index, spect, previous.position, emitter_vertex.position, emitter_vertex.texcoord,
    emitter_vertex.path_length <= 1u, pdf_area, pdf_dir, pdf_dir_out);
  if (spectral_response_is_zero(radiance) || (pdf_area <= 0.0f) || (pdf_dir <= 0.0f) || (pdf_dir_out <= 0.0f)) {
    return;
  }
  float mis_weight = 0.0f;
  if ((resources.iteration.technique_mask & GPUUPBPTechnique::BPT) != 0u) {
    if (((resources.iteration.flags & GPUUPBPIterationFlags::MultipleImportanceSampling) == 0u) || (emitter_vertex.path_length == 1u)) {
      mis_weight = 1.0f;
    } else {
      const float log_reverse_ray_pdf = segment.log_transport_pdf_reverse + (upbp_vertex_is_medium(previous) ? previous.log_medium_event_density : 0.0f);
      const float emitter_selection_pdf = emitter_discrete_pdf(emitter_vertex.emitter_index);
      const bool previous_delta = (emitter_vertex.arrival_weights.flags & GPUUPBPRecursiveWeightFlags::PreviousDelta) != 0u;
      const float log_shared = previous_delta ? kUPBPLogZero : log(emitter_selection_pdf) + log(pdf_area) + emitter_vertex.arrival_weights.log_d_shared;
      const float log_bpt = upbp_log_is_zero(emitter_vertex.arrival_weights.log_d_bpt)
                              ? kUPBPLogZero
                              : log(emitter_selection_pdf) + log(pdf_dir_out) + log_reverse_ray_pdf + emitter_vertex.arrival_weights.log_d_bpt;
      const float log_denominator = upbp_log_add(0.0f, upbp_log_add(log_shared, log_bpt));
      mis_weight = isfinite(log_denominator) ? exp(-log_denominator) : 0.0f;
    }
  } else if (upbp_camera_prefix_is_specular(resources, emitter_vertex)) {
    mis_weight = 1.0f;
  }
  if (mis_weight > 0.0f) {
    upbp_submit_camera_contribution(wavefront_resources, resources, GPUUPBPTechnique::BPT, emitter_vertex.global_path_index,
      spectral_response_mul(spectral_response_mul(throughput, radiance), mis_weight));
  }
}

void upbp_store_density_aabb(uint descriptor_index, uint index, float3 minimum, float3 maximum, uint record_index) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  const uint byte_offset = index * kGPUUPBPAABBStride;
  buffer.Store3(byte_offset, asuint(minimum));
  buffer.Store3(byte_offset + 12u, asuint(maximum));
  buffer.Store(byte_offset + 24u, record_index);
}

uint upbp_surface_query_family_from_material(uint material_index) {
  if ((material_index == kInvalidIndex) || (constants.scene.materials == kInvalidIndex)) {
    return kInvalidIndex;
  }
  const uint material_class = WAVEFRONT_RO_BUFFER(constants.scene.materials).Load(material_index * kMaterialStride + kMaterialClassOffset);
  switch (material_class) {
    case MaterialClass::Plastic:
      return GPUUPBPSurfaceQueryFamily::Plastic;
    case MaterialClass::Conductor:
    case MaterialClass::OpenPBR:
      return GPUUPBPSurfaceQueryFamily::Conductor;
    case MaterialClass::Dielectric:
    case MaterialClass::Thinfilm:
      return GPUUPBPSurfaceQueryFamily::Dielectric;
    case MaterialClass::Diffuse:
    case MaterialClass::Translucent:
    case MaterialClass::Mirror:
    case MaterialClass::Boundary:
    case MaterialClass::Velvet:
    case MaterialClass::Void:
    case MaterialClass::DiffractionGrating:
      return GPUUPBPSurfaceQueryFamily::Various;
    default:
      return kInvalidIndex;
  }
}

[numthreads(64, 1, 1)] void wavefront_upbp_density_compact_main(uint3 dtid : SV_DispatchThreadID) {
  GPUUPBPResources resources = upbp_load_resources(wavefront_load_resources());
  if ((resources.counter_buffer == kInvalidIndex) || (dtid.x >= constants.dispatch_item_count)) {
    return;
  }
  const uint input_index = constants.dispatch_item_offset + dtid.x;
  if (constants.work_queue_index == GPUUPBPDensityCompactMode::BPTVertices) {
    const uint vertex_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::LightVertex * sizeof(uint)), resources.light_vertex_capacity);
    if ((input_index < vertex_count) && (resources.bpt_light_vertex_buffer != kInvalidIndex)) {
      upbp_store_bpt_light_vertex(resources.bpt_light_vertex_buffer, input_index, upbp_load_vertex(resources.vertex_buffer, input_index));
    }
    return;
  }
  if (constants.work_queue_index == GPUUPBPDensityCompactMode::BPTPathStates) {
    if ((input_index < resources.iteration.light_batch_count) && (resources.bpt_light_path_state_buffer != kInvalidIndex)) {
      const GPUUPBPPathState path_state = upbp_load_path_state(resources.path_state_buffer, upbp_path_state_index(resources, false, input_index));
      upbp_store_bpt_light_path_state(resources.bpt_light_path_state_buffer, input_index, path_state);
    }
    return;
  }
  if (constants.work_queue_index == GPUUPBPDensityCompactMode::CameraVertices) {
    const uint vertex_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraVertex * sizeof(uint)), resources.camera_vertex_capacity);
    if (input_index >= vertex_count) {
      return;
    }
    const GPUUPBPVertex vertex = upbp_load_vertex(resources.vertex_buffer, input_index);
    if (((vertex.flags & GPUUPBPVertexFlags::Valid) == 0u) || (vertex.path_length == 0u)) {
      return;
    }
    RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(resources.counter_buffer);
    if (((resources.iteration.technique_mask & GPUUPBPTechnique::Surface) != 0u) && (resources.point_buffer != kInvalidIndex) && upbp_vertex_is_surface(vertex) &&
        upbp_vertex_is_density_connectible(vertex)) {
      const uint query_family = upbp_surface_query_family_from_material(vertex.material_index);
      if (query_family < GPUUPBPSurfaceQueryFamily::Count) {
        uint query_index = 0u;
        counters.InterlockedAdd((GPUUPBPCounterIndex::CameraSurfaceVariousQuery + query_family) * sizeof(uint), 1u, query_index);
        if (query_index < resources.point_capacity) {
          WAVEFRONT_RW_BUFFER(resources.point_buffer).Store((query_family * resources.point_capacity + query_index) * sizeof(uint), input_index);
        }
      }
    }
    if (((resources.iteration.technique_mask & GPUUPBPTechnique::BP2D) != 0u) && (resources.beam_buffer != kInvalidIndex) && upbp_vertex_is_medium(vertex) &&
        (vertex.medium_index != kInvalidIndex)) {
      uint query_index = 0u;
      counters.InterlockedAdd(GPUUPBPCounterIndex::CameraMediumVertexQuery * sizeof(uint), 1u, query_index);
      if (query_index < resources.beam_capacity) {
        WAVEFRONT_RW_BUFFER(resources.beam_buffer).Store(query_index * sizeof(uint), input_index);
      }
    }
    return;
  }
  if (constants.work_queue_index == GPUUPBPDensityCompactMode::CameraIntervals) {
    const uint interval_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraInterval * sizeof(uint)), resources.camera_interval_capacity);
    if ((input_index >= interval_count) || (resources.beam_buffer == kInvalidIndex)) {
      return;
    }
    const GPUUPBPInterval interval = upbp_load_interval(resources.interval_buffer, input_index);
    if (((interval.flags & GPUUPBPIntervalFlags::Valid) == 0u) || (interval.medium_index == kInvalidIndex) ||
        ((resources.iteration.technique_mask & (GPUUPBPTechnique::PB2D | GPUUPBPTechnique::BB1D)) == 0u)) {
      return;
    }
    uint query_index = 0u;
    WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedAdd(GPUUPBPCounterIndex::CameraMediumIntervalQuery * sizeof(uint), 1u, query_index);
    const uint query_capacity = resources.beam_capacity * ((kGPUUPBPBeamStride / sizeof(uint)) - 1u);
    if (query_index < query_capacity) {
      WAVEFRONT_RW_BUFFER(resources.beam_buffer).Store((resources.beam_capacity + query_index) * sizeof(uint), input_index);
    }
    return;
  }
  if (constants.work_queue_index == GPUUPBPDensityCompactMode::Points) {
    const uint point_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::Point * sizeof(uint)), resources.point_capacity);
    if (input_index >= point_count) {
      return;
    }
    const GPUUPBPPoint point_record = upbp_load_point(resources.point_buffer, input_index);
    if (point_record.vertex_index >= (resources.camera_vertex_capacity + resources.light_vertex_capacity)) {
      return;
    }
    const GPUUPBPVertex vertex = upbp_load_vertex(resources.vertex_buffer, point_record.vertex_index);
    if ((vertex.flags & GPUUPBPVertexFlags::Valid) == 0u) {
      return;
    }
    const bool surface = upbp_vertex_is_surface(vertex);
    if ((surface == false) && (upbp_vertex_is_medium(vertex) == false)) {
      return;
    }
    const uint output_buffer = surface ? resources.density_output_surface_point_buffer : resources.density_output_medium_point_buffer;
    const uint output_capacity = surface ? resources.density_output_surface_point_capacity : resources.density_output_medium_point_capacity;
    const uint output_aabb_buffer = surface ? resources.point_aabb_buffer : resources.density_output_medium_point_aabb_buffer;
    const uint output_aabb_capacity = surface ? resources.point_aabb_capacity : resources.density_output_medium_point_aabb_capacity;
    const uint output_counter = surface ? GPUUPBPCounterIndex::DensitySurfacePoint : GPUUPBPCounterIndex::DensityMediumPoint;
    if ((output_buffer == kInvalidIndex) || (output_aabb_buffer == kInvalidIndex)) {
      return;
    }
    uint output_index = 0u;
    WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedAdd(output_counter * sizeof(uint), 1u, output_index);
    if ((output_index >= output_capacity) || (output_index >= output_aabb_capacity)) {
      uint ignored = 0u;
      WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedOr(GPUUPBPCounterIndex::OverflowFlags * sizeof(uint), GPUUPBPOverflowFlags::Point, ignored);
      return;
    }
    upbp_store_density_point(output_buffer, output_index, upbp_pack_density_point(vertex));
    const float radius = max(surface ? resources.iteration.surface_radius : max(resources.iteration.pp3d_radius, resources.iteration.pb2d_radius), 1.0e-7f);
    upbp_store_density_aabb(output_aabb_buffer, output_index, vertex.position - radius, vertex.position + radius, output_index);
    return;
  }
  if (constants.work_queue_index != GPUUPBPDensityCompactMode::Beams) {
    return;
  }
  const uint beam_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::Beam * sizeof(uint)), resources.beam_capacity);
  if ((input_index >= beam_count) || (resources.density_output_beam_buffer == kInvalidIndex)) {
    return;
  }
  const GPUUPBPBeam beam = upbp_load_beam(resources.beam_buffer, input_index);
  UPBPGPUPreparedBeam prepared = (UPBPGPUPreparedBeam)0;
  if (upbp_prepare_beam(resources, beam, prepared) == false) {
    return;
  }
  uint output_index = 0u;
  WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedAdd(GPUUPBPCounterIndex::DensityBeam * sizeof(uint), 1u, output_index);
  if (output_index >= resources.density_output_beam_capacity) {
    uint ignored = 0u;
    WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedOr(GPUUPBPCounterIndex::OverflowFlags * sizeof(uint), GPUUPBPOverflowFlags::Beam, ignored);
    return;
  }
  const GPUUPBPInterval interval = upbp_load_interval(resources.interval_buffer, beam.interval_index);
  upbp_store_density_beam(resources.density_output_beam_buffer, output_index, upbp_pack_density_beam(resources, beam, interval, prepared));
  if ((beam.flags & GPUUPBPBeamFlags::SelectedForBB1D) != 0u) {
    uint selected_index = 0u;
    WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedAdd(GPUUPBPCounterIndex::DensitySelectedBeam * sizeof(uint), 1u, selected_index);
  }
}

void upbp_store_beam_instance(GPUUPBPResources resources, uint instance_buffer_index, uint output_index, uint reference_index, uint instance_mask, float3 origin, float3 direction,
  float length, float radius) {
  const float3 helper_axis = abs(direction.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(0.0f, 1.0f, 0.0f);
  const float3 tangent = normalize(cross(helper_axis, direction));
  const float3 bitangent = cross(direction, tangent);
  const uint instance_offset = output_index * kGPUUPBPAccelerationStructureInstanceStride;
  RWByteAddressBuffer instance_buffer = WAVEFRONT_RW_BUFFER(instance_buffer_index);
  instance_buffer.Store4(instance_offset + 0u, asuint(float4(radius * tangent.x, radius * bitangent.x, length * direction.x, origin.x)));
  instance_buffer.Store4(instance_offset + 16u, asuint(float4(radius * tangent.y, radius * bitangent.y, length * direction.y, origin.y)));
  instance_buffer.Store4(instance_offset + 32u, asuint(float4(radius * tangent.z, radius * bitangent.z, length * direction.z, origin.z)));
  instance_buffer.Store(instance_offset + 48u, (reference_index & 0x00ffffffu) | ((instance_mask & 0xffu) << 24u));
  instance_buffer.Store(instance_offset + 52u, 0u);
  instance_buffer.Store(instance_offset + 56u, resources.density_beam_acceleration_structure_reference_low);
  instance_buffer.Store(instance_offset + 60u, resources.density_beam_acceleration_structure_reference_high);
}

[numthreads(64, 1, 1)] void wavefront_upbp_beam_instances_main(uint3 dtid : SV_DispatchThreadID) {
  GPUUPBPResources resources = upbp_load_resources(wavefront_load_resources());
  if ((resources.counter_buffer == kInvalidIndex) || (resources.density_batch_buffer == kInvalidIndex) || (constants.work_queue_index >= resources.density_batch_count) ||
      (dtid.x >= constants.dispatch_item_count)) {
    return;
  }
  const GPUUPBPDensityBatch batch = upbp_load_density_batch(resources.density_batch_buffer, constants.work_queue_index);
  const uint beam_index = constants.dispatch_item_offset + dtid.x;
  if ((beam_index >= batch.beam_count) || (batch.beam_buffer == kInvalidIndex)) {
    return;
  }
  GPUUPBPDensityBeam density_beam = upbp_load_density_beam(batch.beam_buffer, beam_index);
  const bool bp2d_enabled = (resources.iteration.technique_mask & GPUUPBPTechnique::BP2D) != 0u;
  const bool bb1d_enabled = (resources.iteration.technique_mask & GPUUPBPTechnique::BB1D) != 0u;
  const bool use_acceleration_structures = resources.beam_index_mode == GPUUPBPBeamIndexMode::AccelerationStructure;
  const bool selected_for_bb1d = (density_beam.beam.flags & GPUUPBPBeamFlags::SelectedForBB1D) != 0u;
  if ((bp2d_enabled == false) && ((bb1d_enabled == false) || (selected_for_bb1d == false))) {
    return;
  }
  if ((density_beam.interval.flags & GPUUPBPIntervalFlags::RecomputeTracking) == 0u) {
    if (density_beam.event_buffer == kInvalidIndex) {
      uint ignored = 0u;
      WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedOr(GPUUPBPCounterIndex::OverflowFlags * sizeof(uint), GPUUPBPOverflowFlags::BeamInstance, ignored);
      return;
    }
  } else {
    density_beam.event_buffer = kInvalidIndex;
    density_beam.event_index_offset = 0u;
  }
  const uint bp2d_output_index = batch.beam_instance_offset + beam_index;
  const GPUUPBPBeam beam = density_beam.beam;
  const float direction_length = length(beam.direction);
  if (any(isfinite(beam.origin) == false) || any(isfinite(beam.direction) == false) || (isfinite(beam.length) == false) || (beam.length <= 0.0f) ||
      (isfinite(direction_length) == false) || (direction_length <= 0.0f)) {
    uint ignored = 0u;
    WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedOr(GPUUPBPCounterIndex::OverflowFlags * sizeof(uint), GPUUPBPOverflowFlags::BeamInstance, ignored);
    return;
  }
  const float3 direction = beam.direction / direction_length;
  const uint medium_mask = 1u << (density_beam.interval.medium_index & 7u);
  if (bp2d_enabled) {
    if ((resources.density_output_beam_buffer == kInvalidIndex) || (bp2d_output_index >= resources.density_output_beam_instance_capacity) ||
        (bp2d_output_index >= resources.density_output_beam_capacity) || (use_acceleration_structures && (resources.density_output_beam_instance_buffer == kInvalidIndex))) {
      uint ignored = 0u;
      WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedOr(GPUUPBPCounterIndex::OverflowFlags * sizeof(uint), GPUUPBPOverflowFlags::BeamInstance, ignored);
      return;
    }
    if (use_acceleration_structures) {
      upbp_store_beam_instance(resources, resources.density_output_beam_instance_buffer, bp2d_output_index, bp2d_output_index, medium_mask, beam.origin, direction, beam.length,
        max(resources.iteration.bp2d_radius, 1.0e-7f));
    }
    upbp_store_density_beam(resources.density_output_beam_buffer, bp2d_output_index, density_beam);
  }
  const float bb1d_radius = bb1d_enabled ? resources.iteration.bb1d_radius : 0.0f;
  if (bb1d_enabled && selected_for_bb1d) {
    uint bb1d_output_index = 0u;
    WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedAdd(GPUUPBPCounterIndex::DensityBeamInstance * sizeof(uint), 1u, bb1d_output_index);
    if ((resources.bb1d_beam_buffer == kInvalidIndex) || (bb1d_output_index >= resources.density_output_bb1d_beam_instance_capacity) ||
        (use_acceleration_structures && (resources.density_output_bb1d_beam_instance_buffer == kInvalidIndex))) {
      uint ignored = 0u;
      WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedOr(GPUUPBPCounterIndex::OverflowFlags * sizeof(uint), GPUUPBPOverflowFlags::BeamInstance, ignored);
      return;
    }
    if (use_acceleration_structures) {
      const uint bb1d_partition_index = bb1d_output_index % kGPUUPBPBB1DPartitionCount;
      const uint partition_base_count = resources.density_output_bb1d_beam_instance_capacity / kGPUUPBPBB1DPartitionCount;
      const uint partition_remainder = resources.density_output_bb1d_beam_instance_capacity % kGPUUPBPBB1DPartitionCount;
      const uint partition_offset = bb1d_partition_index * partition_base_count + min(bb1d_partition_index, partition_remainder);
      const uint partition_storage_index = partition_offset + bb1d_output_index / kGPUUPBPBB1DPartitionCount;
      upbp_store_beam_instance(resources, resources.density_output_bb1d_beam_instance_buffer, partition_storage_index, bb1d_output_index, medium_mask, beam.origin, direction,
        beam.length, max(bb1d_radius, 1.0e-7f));
    }
    upbp_store_density_beam(resources.bb1d_beam_buffer, bb1d_output_index, density_beam);
  }
  if (bp2d_enabled) {
    if (resources.density_output_beam_reference_buffer == kInvalidIndex) {
      uint ignored = 0u;
      WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedOr(GPUUPBPCounterIndex::OverflowFlags * sizeof(uint), GPUUPBPOverflowFlags::BeamInstance, ignored);
      return;
    }
    RWByteAddressBuffer reference_buffer = WAVEFRONT_RW_BUFFER(resources.density_output_beam_reference_buffer);
    const uint reference_offset = bp2d_output_index * kGPUUPBPBeamReferenceStride;
    wavefront_store_float3(reference_buffer, reference_offset + kGPUUPBPBeamReferenceOriginOffset, beam.origin);
    reference_buffer.Store(reference_offset + kGPUUPBPBeamReferenceLengthOffset, asuint(beam.length));
    wavefront_store_float3(reference_buffer, reference_offset + kGPUUPBPBeamReferenceDirectionOffset, beam.direction);
    reference_buffer.Store(reference_offset + kGPUUPBPBeamReferencePathLengthOffset, beam.path_length);
    reference_buffer.Store(reference_offset + kGPUUPBPBeamReferenceMediumIndexOffset, density_beam.interval.medium_index);
  }
}

  [numthreads(64, 1, 1)] void wavefront_upbp_pp3d_main(uint3 dtid : SV_DispatchThreadID) {
  if (dtid.x < constants.dispatch_item_count) {
    upbp_evaluate_point_merge_dispatch(constants.dispatch_item_offset + dtid.x, dtid.x & 63u, GPUUPBPTechnique::PP3D);
  }
}

[numthreads(64, 1, 1)] void wavefront_upbp_direct_hit_main(uint3 dtid : SV_DispatchThreadID) {
  if (dtid.x < constants.dispatch_item_count) {
    upbp_evaluate_direct_hit_dispatch(constants.dispatch_item_offset + dtid.x);
  }
}

  [numthreads(64, 1, 1)] void wavefront_upbp_pb2d_main(uint3 dtid : SV_DispatchThreadID) {
  GPUWavefrontResources wavefront_resources = wavefront_load_resources();
  GPUUPBPResources resources = upbp_load_resources(wavefront_resources);
  if ((resources.counter_buffer == kInvalidIndex) || (resources.medium_point_acceleration_structure == kInvalidIndex) || (dtid.x >= constants.dispatch_item_count)) {
    return;
  }
  uint interval_index = constants.dispatch_item_offset + dtid.x;
  if (constants.work_queue_index == GPUUPBPDensityQueryMode::Compacted) {
    const uint query_count = WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraMediumIntervalQuery * sizeof(uint));
    interval_index = ((interval_index < query_count) && (resources.beam_buffer != kInvalidIndex))
                       ? WAVEFRONT_RO_BUFFER(resources.beam_buffer).Load((resources.beam_capacity + interval_index) * sizeof(uint))
                       : kInvalidIndex;
  }
  const uint interval_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraInterval * 4u), resources.camera_interval_capacity);
  if (interval_index >= interval_count) {
    return;
  }
  GPUUPBPBeam camera_beam = (GPUUPBPBeam)0;
  UPBPGPUPreparedBeam prepared_camera = (UPBPGPUPreparedBeam)0;
  if ((upbp_make_beam_from_interval(resources, interval_index, camera_beam) == false) || (upbp_prepare_beam(resources, camera_beam, prepared_camera) == false)) {
    return;
  }
  SpectralResponse accumulated = spectral_response_zero(spectral_response_as_query(prepared_camera.source_throughput));
  RayDesc ray = (RayDesc)0;
  ray.Origin = camera_beam.origin;
  ray.Direction = camera_beam.direction;
  ray.TMin = 0.0f;
  ray.TMax = camera_beam.length;
  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
  query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(resources.medium_point_acceleration_structure)], RAY_FLAG_FORCE_NON_OPAQUE, 0xff, ray);
  [loop] while (query.Proceed()) {
    if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
      const uint primitive_index = query.CandidatePrimitiveIndex();
      if ((primitive_index < resources.density_output_medium_point_capacity) && (resources.density_output_medium_point_buffer != kInvalidIndex)) {
        upbp_evaluate_pb2d_vertex(resources, camera_beam, prepared_camera,
          upbp_unpack_density_point(upbp_load_density_point(resources.density_output_medium_point_buffer, primitive_index)), accumulated);
      }
    }
  }
  upbp_submit_camera_contribution(wavefront_resources, resources, GPUUPBPTechnique::PB2D, camera_beam.global_path_index, accumulated);
}

GPUUPBPBeamGridMetadata upbp_load_beam_grid_metadata(GPUUPBPBeamGridResources grid) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(grid.metadata_buffer);
  GPUUPBPBeamGridMetadata result = (GPUUPBPBeamGridMetadata)0;
  result.minimum = wavefront_load_float3(buffer, kGPUUPBPBeamGridMetadataMinimumOffset);
  result.resolution_x = buffer.Load(kGPUUPBPBeamGridMetadataResolutionXOffset);
  result.maximum = wavefront_load_float3(buffer, kGPUUPBPBeamGridMetadataMaximumOffset);
  result.resolution_y = buffer.Load(kGPUUPBPBeamGridMetadataResolutionYOffset);
  result.inverse_cell_size = wavefront_load_float3(buffer, kGPUUPBPBeamGridMetadataInverseCellSizeOffset);
  result.resolution_z = buffer.Load(kGPUUPBPBeamGridMetadataResolutionZOffset);
  result.cell_count = buffer.Load(kGPUUPBPBeamGridMetadataCellCountOffset);
  result.beam_count = buffer.Load(kGPUUPBPBeamGridMetadataBeamCountOffset);
  result.entry_count = buffer.Load(kGPUUPBPBeamGridMetadataEntryCountOffset);
  result.reserved0 = buffer.Load(kGPUUPBPBeamGridMetadataReserved0Offset);
  return result;
}

bool upbp_beam_grid_valid(GPUUPBPBeamGridResources grid) {
  return (grid.metadata_buffer != kInvalidIndex) && (grid.cell_offsets_buffer != kInvalidIndex) && (grid.beam_indices_buffer != kInvalidIndex) && (grid.beam_count > 0u) &&
         (grid.beam_index_count > 0u);
}

uint3 upbp_beam_grid_cell(GPUUPBPBeamGridMetadata metadata, float3 position) {
  const uint3 resolution = uint3(metadata.resolution_x, metadata.resolution_y, metadata.resolution_z);
  const int3 cell = int3(floor((position - metadata.minimum) * metadata.inverse_cell_size));
  return uint3(clamp(cell, int3(0, 0, 0), int3(resolution) - 1));
}

uint upbp_beam_grid_cell_index(GPUUPBPBeamGridMetadata metadata, uint3 cell) {
  return cell.x + metadata.resolution_x * (cell.y + metadata.resolution_y * cell.z);
}

void upbp_store_beam_grid_metadata(uint descriptor_index, GPUUPBPBeamGridMetadata metadata) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  wavefront_store_float3(buffer, kGPUUPBPBeamGridMetadataMinimumOffset, metadata.minimum);
  buffer.Store(kGPUUPBPBeamGridMetadataResolutionXOffset, metadata.resolution_x);
  wavefront_store_float3(buffer, kGPUUPBPBeamGridMetadataMaximumOffset, metadata.maximum);
  buffer.Store(kGPUUPBPBeamGridMetadataResolutionYOffset, metadata.resolution_y);
  wavefront_store_float3(buffer, kGPUUPBPBeamGridMetadataInverseCellSizeOffset, metadata.inverse_cell_size);
  buffer.Store(kGPUUPBPBeamGridMetadataResolutionZOffset, metadata.resolution_z);
  buffer.Store(kGPUUPBPBeamGridMetadataCellCountOffset, metadata.cell_count);
  buffer.Store(kGPUUPBPBeamGridMetadataBeamCountOffset, metadata.beam_count);
  buffer.Store(kGPUUPBPBeamGridMetadataEntryCountOffset, metadata.entry_count);
  buffer.Store(kGPUUPBPBeamGridMetadataReserved0Offset, metadata.reserved0);
}

GPUUPBPBeamGridResources upbp_beam_grid_build_resources(GPUUPBPResources resources, uint grid_type) {
  if (grid_type == GPUUPBPBeamGridType::BP2D) {
    return resources.bp2d_beam_grid;
  }
  return resources.bb1d_beam_grid;
}

GPUUPBPBeamReference upbp_beam_grid_build_reference(GPUUPBPBeamGridResources grid, uint grid_type, uint beam_index) {
  if (grid_type == GPUUPBPBeamGridType::BP2D) {
    return upbp_load_beam_reference(grid.reserved1, beam_index);
  }
  return upbp_load_bb1d_beam_reference(grid.reserved1, beam_index);
}

void upbp_beam_grid_build_fail(GPUUPBPBeamGridResources grid, uint failure) {
  uint ignored = 0u;
  WAVEFRONT_RW_BUFFER(grid.metadata_buffer).InterlockedOr(kGPUUPBPBeamGridMetadataReserved0Offset, failure, ignored);
}

uint3 upbp_beam_grid_build_cell(GPUUPBPBeamGridMetadata metadata, float3 position) {
  const uint3 resolution = uint3(metadata.resolution_x, metadata.resolution_y, metadata.resolution_z);
  precise float3 grid_position = (position - metadata.minimum) * metadata.inverse_cell_size;
  const int3 cell = int3(floor(grid_position));
  return uint3(clamp(cell, int3(0, 0, 0), int3(resolution) - 1));
}

bool upbp_beam_grid_build_process_beam(GPUUPBPBeamGridResources grid, GPUUPBPBeamGridMetadata metadata, uint grid_type, float radius, uint shard_index, uint beam_index,
  bool scatter) {
  const GPUUPBPBeamReference beam = upbp_beam_grid_build_reference(grid, grid_type, beam_index);
  const float direction_length_squared = dot(beam.direction, beam.direction);
  if ((beam.length <= 0.0f) || (isfinite(beam.length) == false) || any(isfinite(beam.origin) == false) || any(isfinite(beam.direction) == false) ||
      (isfinite(direction_length_squared) == false) || (abs(direction_length_squared - 1.0f) > 1.0e-4f)) {
    return false;
  }

  const float3 absolute_direction = abs(beam.direction);
  const float dominant_direction = max(absolute_direction.x, max(absolute_direction.y, absolute_direction.z));
  const float cell_size = 1.0f / metadata.inverse_cell_size.x;
  precise float projected_cell_count = ceil(beam.length * dominant_direction / cell_size);
  if ((isfinite(projected_cell_count) == false) || (projected_cell_count > 4294967040.0f)) {
    return false;
  }
  const uint segment_count = max(1u, (uint)projected_cell_count);
  precise float inverse_segment_count = 1.0f / (float)segment_count;
  const float3 support_extent = float3(radius, radius, radius);
  uint3 previous_minimum_cell = uint3(0u, 0u, 0u);
  uint3 previous_maximum_cell = uint3(0u, 0u, 0u);
  RWByteAddressBuffer shard_offsets = WAVEFRONT_RW_BUFFER(grid.reserved0);
  [loop] for (uint segment_index = 0u; segment_index < segment_count; ++segment_index) {
    precise float first_distance = beam.length * ((float)segment_index * inverse_segment_count);
    precise float second_distance = beam.length * ((float)(segment_index + 1u) * inverse_segment_count);
    precise float3 first = beam.origin + beam.direction * first_distance;
    precise float3 second = beam.origin + beam.direction * second_distance;
    const uint3 minimum_cell = upbp_beam_grid_build_cell(metadata, min(first, second) - support_extent);
    const uint3 maximum_cell = upbp_beam_grid_build_cell(metadata, max(first, second) + support_extent);
    [loop] for (uint z = minimum_cell.z; z <= maximum_cell.z; ++z) {
      [loop] for (uint y = minimum_cell.y; y <= maximum_cell.y; ++y) {
        [loop] for (uint x = minimum_cell.x; x <= maximum_cell.x; ++x) {
          const uint3 cell = uint3(x, y, z);
          const bool seen_in_previous_segment = (segment_index > 0u) && all(cell >= previous_minimum_cell) && all(cell <= previous_maximum_cell);
          if (seen_in_previous_segment) {
            continue;
          }
          const uint cell_index = upbp_beam_grid_cell_index(metadata, cell);
          const uint shard_cell_index = shard_index * metadata.cell_count + cell_index;
          const uint byte_offset = shard_cell_index * sizeof(uint);
          const uint output_index = shard_offsets.Load(byte_offset);
          if (scatter) {
            if (output_index >= grid.beam_index_count) {
              upbp_beam_grid_build_fail(grid, GPUUPBPBeamGridBuildFailure::OutputCapacity);
              return false;
            }
            WAVEFRONT_RW_BUFFER(grid.beam_indices_buffer).Store(output_index * sizeof(uint), beam_index);
          }
          shard_offsets.Store(byte_offset, output_index + 1u);
        }
      }
    }
    previous_minimum_cell = minimum_cell;
    previous_maximum_cell = maximum_cell;
  }
  return true;
}

void upbp_beam_grid_describe(GPUUPBPBeamGridResources grid, uint grid_type, float radius) {
  GPUUPBPBeamGridMetadata metadata = (GPUUPBPBeamGridMetadata)0;
  metadata.minimum = float3(kMaxFloat, kMaxFloat, kMaxFloat);
  metadata.maximum = float3(-kMaxFloat, -kMaxFloat, -kMaxFloat);
  metadata.beam_count = grid.beam_count;
  if ((grid.beam_count == 0u) || (grid.reserved1 == kInvalidIndex) || (radius <= 0.0f) || (isfinite(radius) == false)) {
    metadata.reserved0 = GPUUPBPBeamGridBuildFailure::InvalidInput;
    upbp_store_beam_grid_metadata(grid.metadata_buffer, metadata);
    return;
  }
  [loop] for (uint beam_index = 0u; beam_index < grid.beam_count; ++beam_index) {
    const GPUUPBPBeamReference beam = upbp_beam_grid_build_reference(grid, grid_type, beam_index);
    const float direction_length_squared = dot(beam.direction, beam.direction);
    if ((beam.length <= 0.0f) || (isfinite(beam.length) == false) || any(isfinite(beam.origin) == false) || any(isfinite(beam.direction) == false) ||
        (isfinite(direction_length_squared) == false) || (abs(direction_length_squared - 1.0f) > 1.0e-4f)) {
      metadata.reserved0 = GPUUPBPBeamGridBuildFailure::InvalidInput;
      upbp_store_beam_grid_metadata(grid.metadata_buffer, metadata);
      return;
    }
    const float3 end = beam.origin + beam.direction * beam.length;
    const float3 support_extent = float3(radius, radius, radius);
    metadata.minimum = min(metadata.minimum, min(beam.origin, end) - support_extent);
    metadata.maximum = max(metadata.maximum, max(beam.origin, end) + support_extent);
  }
  const float3 extent = metadata.maximum - metadata.minimum;
  const float maximum_extent = max(extent.x, max(extent.y, extent.z));
  if ((maximum_extent <= 0.0f) || (isfinite(maximum_extent) == false)) {
    metadata.reserved0 = GPUUPBPBeamGridBuildFailure::InvalidInput;
    upbp_store_beam_grid_metadata(grid.metadata_buffer, metadata);
    return;
  }
  const float cell_size_value = maximum_extent / 32.0f;
  const float cell_size = asfloat(asuint(cell_size_value) + 1u);
  if ((cell_size <= 0.0f) || (isfinite(cell_size) == false)) {
    metadata.reserved0 = GPUUPBPBeamGridBuildFailure::InvalidInput;
    upbp_store_beam_grid_metadata(grid.metadata_buffer, metadata);
    return;
  }
  const float inverse_cell_size = 1.0f / cell_size;
  const uint3 resolution = min(uint3(32u, 32u, 32u), max(uint3(1u, 1u, 1u), uint3(ceil(extent / cell_size))));
  metadata.maximum = metadata.minimum + float3(resolution) * cell_size;
  metadata.inverse_cell_size = float3(inverse_cell_size, inverse_cell_size, inverse_cell_size);
  metadata.resolution_x = resolution.x;
  metadata.resolution_y = resolution.y;
  metadata.resolution_z = resolution.z;
  metadata.cell_count = resolution.x * resolution.y * resolution.z;
  upbp_store_beam_grid_metadata(grid.metadata_buffer, metadata);
}

void upbp_beam_grid_process_shard(GPUUPBPBeamGridResources grid, GPUUPBPBeamGridMetadata metadata, uint grid_type, float radius, uint shard_index, bool scatter) {
  if ((shard_index >= grid.reserved2) || (metadata.reserved0 != GPUUPBPBeamGridBuildFailure::None)) {
    return;
  }
  RWByteAddressBuffer shard_offsets = WAVEFRONT_RW_BUFFER(grid.reserved0);
  const uint shard_cell_offset = shard_index * metadata.cell_count;
  if (scatter == false) {
    [loop] for (uint cell_index = 0u; cell_index < metadata.cell_count; ++cell_index) {
      shard_offsets.Store((shard_cell_offset + cell_index) * sizeof(uint), 0u);
    }
  }
  const uint beams_per_shard = grid.beam_count / grid.reserved2;
  const uint remainder = grid.beam_count % grid.reserved2;
  const uint first_beam = shard_index * beams_per_shard + min(shard_index, remainder);
  const uint shard_beam_count = beams_per_shard + (shard_index < remainder ? 1u : 0u);
  [loop] for (uint beam_index = first_beam; beam_index < first_beam + shard_beam_count; ++beam_index) {
    if (upbp_beam_grid_build_process_beam(grid, metadata, grid_type, radius, shard_index, beam_index, scatter) == false) {
      if (scatter == false) {
        upbp_beam_grid_build_fail(grid, GPUUPBPBeamGridBuildFailure::InvalidInput);
      }
      return;
    }
  }
}

void upbp_beam_grid_cell_total(GPUUPBPBeamGridResources grid, GPUUPBPBeamGridMetadata metadata, uint cell_index) {
  if (metadata.reserved0 != GPUUPBPBeamGridBuildFailure::None) {
    return;
  }
  ByteAddressBuffer shard_counts = WAVEFRONT_RO_BUFFER(grid.reserved0);
  uint cell_total = 0u;
  [loop] for (uint shard_index = 0u; shard_index < grid.reserved2; ++shard_index) {
    const uint count = shard_counts.Load((shard_index * metadata.cell_count + cell_index) * sizeof(uint));
    if (count > (0xffffffffu - cell_total)) {
      upbp_beam_grid_build_fail(grid, GPUUPBPBeamGridBuildFailure::EntryCountOverflow);
      return;
    }
    cell_total += count;
  }
  WAVEFRONT_RW_BUFFER(grid.cell_offsets_buffer).Store(cell_index * sizeof(uint), cell_total);
}

void upbp_beam_grid_prefix(GPUUPBPBeamGridResources grid, GPUUPBPBeamGridMetadata metadata) {
  if (metadata.reserved0 != GPUUPBPBeamGridBuildFailure::None) {
    return;
  }
  RWByteAddressBuffer cell_offsets = WAVEFRONT_RW_BUFFER(grid.cell_offsets_buffer);
  uint entry_count = 0u;
  [loop] for (uint cell_index = 0u; cell_index < metadata.cell_count; ++cell_index) {
    const uint count = cell_offsets.Load(cell_index * sizeof(uint));
    cell_offsets.Store(cell_index * sizeof(uint), entry_count);
    if (count > (0xffffffffu - entry_count)) {
      upbp_beam_grid_build_fail(grid, GPUUPBPBeamGridBuildFailure::EntryCountOverflow);
      return;
    }
    entry_count += count;
  }
  cell_offsets.Store(metadata.cell_count * sizeof(uint), entry_count);
  WAVEFRONT_RW_BUFFER(grid.metadata_buffer).Store(kGPUUPBPBeamGridMetadataEntryCountOffset, entry_count);
}

void upbp_beam_grid_shard_offsets(GPUUPBPBeamGridResources grid, GPUUPBPBeamGridMetadata metadata, uint cell_index) {
  if (metadata.reserved0 != GPUUPBPBeamGridBuildFailure::None) {
    return;
  }
  RWByteAddressBuffer shard_offsets = WAVEFRONT_RW_BUFFER(grid.reserved0);
  uint output_offset = WAVEFRONT_RO_BUFFER(grid.cell_offsets_buffer).Load(cell_index * sizeof(uint));
  const uint shard_cell_count = grid.reserved2 * metadata.cell_count;
  [loop] for (uint shard_index = 0u; shard_index < grid.reserved2; ++shard_index) {
    const uint shard_cell_index = shard_index * metadata.cell_count + cell_index;
    const uint byte_offset = shard_cell_index * sizeof(uint);
    const uint count = shard_offsets.Load(byte_offset);
    shard_offsets.Store(byte_offset, output_offset);
    output_offset += count;
    shard_offsets.Store((shard_cell_count + shard_cell_index) * sizeof(uint), output_offset);
  }
}

void upbp_beam_grid_validate(GPUUPBPBeamGridResources grid, GPUUPBPBeamGridMetadata metadata, uint cell_index) {
  const ByteAddressBuffer shard_offsets = WAVEFRONT_RO_BUFFER(grid.reserved0);
  const uint shard_cell_count = grid.reserved2 * metadata.cell_count;
  [loop] for (uint shard_index = 0u; shard_index < grid.reserved2; ++shard_index) {
    const uint shard_cell_index = shard_index * metadata.cell_count + cell_index;
    const uint cursor = shard_offsets.Load(shard_cell_index * sizeof(uint));
    const uint expected_end = shard_offsets.Load((shard_cell_count + shard_cell_index) * sizeof(uint));
    if (cursor != expected_end) {
      upbp_beam_grid_build_fail(grid, GPUUPBPBeamGridBuildFailure::CountScatterMismatch);
      return;
    }
  }
  const ByteAddressBuffer cell_offsets = WAVEFRONT_RO_BUFFER(grid.cell_offsets_buffer);
  const ByteAddressBuffer beam_indices = WAVEFRONT_RO_BUFFER(grid.beam_indices_buffer);
  const uint first = cell_offsets.Load(cell_index * sizeof(uint));
  const uint end = cell_offsets.Load((cell_index + 1u) * sizeof(uint));
  if ((end > metadata.entry_count) || (end < first)) {
    upbp_beam_grid_build_fail(grid, GPUUPBPBeamGridBuildFailure::OutputCapacity);
    return;
  }
  if (first == end) {
    return;
  }
  uint previous = beam_indices.Load(first * sizeof(uint));
  if (previous >= metadata.beam_count) {
    upbp_beam_grid_build_fail(grid, GPUUPBPBeamGridBuildFailure::InvalidIndex);
    return;
  }
  [loop] for (uint offset = first + 1u; offset < end; ++offset) {
    const uint current = beam_indices.Load(offset * sizeof(uint));
    if (current >= metadata.beam_count) {
      upbp_beam_grid_build_fail(grid, GPUUPBPBeamGridBuildFailure::InvalidIndex);
      return;
    }
    if (current == previous) {
      upbp_beam_grid_build_fail(grid, GPUUPBPBeamGridBuildFailure::DuplicateIndex);
      return;
    }
    if (current < previous) {
      upbp_beam_grid_build_fail(grid, GPUUPBPBeamGridBuildFailure::InvalidOrder);
      return;
    }
    previous = current;
  }
}

[numthreads(64, 1, 1)] void wavefront_upbp_beam_grid_build_main(uint3 dtid : SV_DispatchThreadID) {
  const uint grid_type = constants.path_iteration;
  GPUUPBPResources resources = upbp_load_resources(wavefront_load_resources());
  const GPUUPBPBeamGridResources grid = upbp_beam_grid_build_resources(resources, grid_type);
  if ((grid.metadata_buffer == kInvalidIndex) || (grid_type > GPUUPBPBeamGridType::BB1D)) {
    return;
  }
  const float radius = grid_type == GPUUPBPBeamGridType::BP2D ? resources.iteration.bp2d_radius : resources.iteration.bb1d_radius;
  if (constants.work_queue_index == GPUUPBPBeamGridBuildMode::Describe) {
    if (dtid.x == 0u) {
      upbp_beam_grid_describe(grid, grid_type, radius);
    }
    return;
  }
  const GPUUPBPBeamGridMetadata metadata = upbp_load_beam_grid_metadata(grid);
  if ((constants.work_queue_index == GPUUPBPBeamGridBuildMode::Count) || (constants.work_queue_index == GPUUPBPBeamGridBuildMode::Scatter)) {
    if (dtid.x < constants.dispatch_item_count) {
      const bool scatter = constants.work_queue_index == GPUUPBPBeamGridBuildMode::Scatter;
      upbp_beam_grid_process_shard(grid, metadata, grid_type, radius, dtid.x, scatter);
    }
    return;
  }
  if (constants.work_queue_index == GPUUPBPBeamGridBuildMode::CellTotals) {
    if (dtid.x < constants.dispatch_item_count) {
      upbp_beam_grid_cell_total(grid, metadata, dtid.x);
    }
    return;
  }
  if (constants.work_queue_index == GPUUPBPBeamGridBuildMode::Prefix) {
    if (dtid.x == 0u) {
      upbp_beam_grid_prefix(grid, metadata);
    }
    return;
  }
  if (constants.work_queue_index == GPUUPBPBeamGridBuildMode::ShardOffsets) {
    if (dtid.x < constants.dispatch_item_count) {
      upbp_beam_grid_shard_offsets(grid, metadata, dtid.x);
    }
    return;
  }
  if (constants.work_queue_index == GPUUPBPBeamGridBuildMode::Validate) {
    if (dtid.x < constants.dispatch_item_count) {
      upbp_beam_grid_validate(grid, metadata, dtid.x);
    }
    return;
  }
}

bool upbp_beam_grid_ray_range(GPUUPBPBeamGridMetadata metadata, GPUUPBPBeam beam, out float minimum_distance, out float maximum_distance) {
  minimum_distance = 0.0f;
  maximum_distance = beam.length;
  [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
    const float origin = beam.origin[axis];
    const float direction = beam.direction[axis];
    if (direction == 0.0f) {
      if ((origin < metadata.minimum[axis]) || (origin > metadata.maximum[axis])) {
        return false;
      }
      continue;
    }
    const float inverse_direction = 1.0f / direction;
    const float first = (metadata.minimum[axis] - origin) * inverse_direction;
    const float second = (metadata.maximum[axis] - origin) * inverse_direction;
    minimum_distance = max(minimum_distance, min(first, second));
    maximum_distance = min(maximum_distance, max(first, second));
    if (minimum_distance > maximum_distance) {
      return false;
    }
  }
  return minimum_distance < maximum_distance;
}

groupshared GPUUPBPVertex upbp_bp2d_camera_vertex;
groupshared GPUWavefrontCompactSpectralResponse upbp_bp2d_camera_pre_collision;
groupshared GPUWavefrontCompactSpectralResponse upbp_bp2d_camera_scattering;
groupshared float upbp_bp2d_camera_phase_function_g;
groupshared uint upbp_bp2d_query_valid;
groupshared GPUWavefrontCompactSpectralResponse upbp_bp2d_lane_contributions[kGPUUPBPBP2DPartitionCount];

[numthreads(32, 1, 1)] void wavefront_upbp_bp2d_main(uint3 group_id : SV_GroupID, uint group_thread_index : SV_GroupIndex) {
  GPUWavefrontResources wavefront_resources = wavefront_load_resources();
  GPUUPBPResources resources = upbp_load_resources(wavefront_resources);
  if ((resources.counter_buffer == kInvalidIndex) || (resources.beam_reference_buffer == kInvalidIndex) || (resources.density_output_beam_buffer == kInvalidIndex) ||
      (group_id.x >= constants.dispatch_item_count)) {
    return;
  }
  const uint query_index = constants.dispatch_item_offset + group_id.x;
  uint vertex_index = query_index;
  if (constants.work_queue_index == GPUUPBPDensityQueryMode::Compacted) {
    const uint query_count = WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraMediumVertexQuery * sizeof(uint));
    vertex_index =
      ((query_index < query_count) && (resources.beam_buffer != kInvalidIndex)) ? WAVEFRONT_RO_BUFFER(resources.beam_buffer).Load(query_index * sizeof(uint)) : kInvalidIndex;
  }
  if (group_thread_index == 0u) {
    upbp_bp2d_query_valid = 0u;
    const uint vertex_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraVertex * sizeof(uint)), resources.camera_vertex_capacity);
    if (vertex_index < vertex_count) {
      upbp_bp2d_camera_vertex = upbp_load_vertex(resources.vertex_buffer, vertex_index);
      if (((upbp_bp2d_camera_vertex.flags & GPUUPBPVertexFlags::Valid) != 0u) && (upbp_bp2d_camera_vertex.path_length > 0u) && upbp_vertex_is_medium(upbp_bp2d_camera_vertex) &&
          (upbp_bp2d_camera_vertex.medium_index != kInvalidIndex)) {
        const SpectralResponse camera_throughput = upbp_unpack_spectral_response(upbp_bp2d_camera_vertex.throughput);
        SpectralResponse camera_scattering = spectral_response_zero(spectral_response_as_query(camera_throughput));
        SpectralResponse camera_pre_collision = spectral_response_zero(spectral_response_as_query(camera_throughput));
        float camera_phase_function_g = 0.0f;
        if (upbp_vertex_medium_properties(upbp_bp2d_camera_vertex, spectral_response_as_query(camera_throughput), camera_scattering, camera_phase_function_g) &&
            upbp_medium_pre_collision_throughput_with_scattering(upbp_bp2d_camera_vertex, camera_throughput, camera_scattering, camera_pre_collision)) {
          upbp_bp2d_camera_pre_collision = upbp_pack_spectral_response(camera_pre_collision);
          upbp_bp2d_camera_scattering = upbp_pack_spectral_response(camera_scattering);
          upbp_bp2d_camera_phase_function_g = camera_phase_function_g;
          upbp_bp2d_query_valid = 1u;
        }
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (upbp_bp2d_query_valid == 0u) {
    return;
  }
  const GPUUPBPVertex camera_vertex = upbp_bp2d_camera_vertex;
  const SpectralResponse camera_pre_collision = upbp_unpack_spectral_response(upbp_bp2d_camera_pre_collision);
  const SpectralResponse camera_scattering = upbp_unpack_spectral_response(upbp_bp2d_camera_scattering);
  SpectralResponse accumulated = spectral_response_zero(spectral_response_as_query(upbp_unpack_spectral_response(camera_vertex.throughput)));
  if (resources.beam_index_mode == GPUUPBPBeamIndexMode::ComputeGrid) {
    const GPUUPBPBeamGridResources grid = resources.bp2d_beam_grid;
    if (upbp_beam_grid_valid(grid)) {
      const GPUUPBPBeamGridMetadata metadata = upbp_load_beam_grid_metadata(grid);
      if (all(camera_vertex.position >= metadata.minimum) && all(camera_vertex.position <= metadata.maximum)) {
        const uint cell_index = upbp_beam_grid_cell_index(metadata, upbp_beam_grid_cell(metadata, camera_vertex.position));
        ByteAddressBuffer offsets = WAVEFRONT_RO_BUFFER(grid.cell_offsets_buffer);
        ByteAddressBuffer indices = WAVEFRONT_RO_BUFFER(grid.beam_indices_buffer);
        const uint first = offsets.Load(cell_index * sizeof(uint));
        const uint end = min(offsets.Load((cell_index + 1u) * sizeof(uint)), min(grid.beam_index_count, metadata.entry_count));
        for (uint offset = first + group_thread_index; offset < end; offset += 32u) {
          const uint reference_index = indices.Load(offset * sizeof(uint));
          if (reference_index < min(grid.beam_count, resources.density_output_beam_capacity)) {
            const GPUUPBPBeamReference reference = upbp_load_beam_reference(resources.beam_reference_buffer, reference_index);
            UPBPGPUPointBeamIntersection intersection = (UPBPGPUPointBeamIntersection)0;
            if (upbp_bp2d_density_candidate_intersects(resources, reference, camera_vertex, intersection)) {
              upbp_evaluate_bp2d_density_candidate(resources, camera_vertex, upbp_load_density_beam(resources.density_output_beam_buffer, reference_index), intersection,
                camera_pre_collision, camera_scattering, upbp_bp2d_camera_phase_function_g, accumulated);
            }
          }
        }
      }
    }
  } else if (group_thread_index < kGPUUPBPBP2DPartitionCount) {
    const uint acceleration_structure = upbp_load_bp2d_partition_acceleration_structure(wavefront_resources, group_thread_index);
    if (acceleration_structure != kInvalidIndex) {
      RayDesc ray = (RayDesc)0;
      ray.Origin = camera_vertex.position;
      ray.Direction = float3(kRayEpsilon, kRayEpsilon, kRayEpsilon);
      ray.TMin = 0.0f;
      ray.TMax = kRayEpsilon;
      RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
      const uint medium_mask = 1u << (camera_vertex.medium_index & 7u);
      query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(acceleration_structure)], RAY_FLAG_FORCE_NON_OPAQUE, medium_mask, ray);
      [loop] while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
          const uint reference_index = query.CandidateInstanceID();
          if (reference_index < resources.density_output_beam_capacity) {
            const GPUUPBPBeamReference reference = upbp_load_beam_reference(resources.beam_reference_buffer, reference_index);
            UPBPGPUPointBeamIntersection intersection = (UPBPGPUPointBeamIntersection)0;
            if (upbp_bp2d_density_candidate_intersects(resources, reference, camera_vertex, intersection)) {
              upbp_evaluate_bp2d_density_candidate(resources, camera_vertex, upbp_load_density_beam(resources.density_output_beam_buffer, reference_index), intersection,
                camera_pre_collision, camera_scattering, upbp_bp2d_camera_phase_function_g, accumulated);
            }
          }
        }
      }
    }
  }
  upbp_bp2d_lane_contributions[group_thread_index] = upbp_pack_spectral_response(accumulated);
  GroupMemoryBarrierWithGroupSync();
  if (group_thread_index == 0u) {
    SpectralResponse total = spectral_response_zero(spectral_response_as_query(upbp_unpack_spectral_response(camera_vertex.throughput)));
    [unroll] for (uint lane_index = 0u; lane_index < kGPUUPBPBP2DPartitionCount; ++lane_index) {
      total = spectral_response_add(total, upbp_unpack_spectral_response(upbp_bp2d_lane_contributions[lane_index]));
    }
    upbp_submit_camera_contribution(wavefront_resources, resources, GPUUPBPTechnique::BP2D, camera_vertex.global_path_index, total);
  }
}

groupshared GPUUPBPBeam upbp_bb1d_camera_beam;
groupshared UPBPGPUPreparedBeam upbp_bb1d_prepared_camera;
groupshared GPUUPBPInterval upbp_bb1d_camera_interval;
groupshared GPUUPBPVertex upbp_bb1d_context_vertex;
groupshared uint upbp_bb1d_query_valid;
groupshared uint upbp_bb1d_exhausted_partition_count;
groupshared GPUWavefrontCompactSpectralResponse upbp_bb1d_lane_contributions[64u];

[numthreads(64, 1, 1)] void wavefront_upbp_bb1d_main(uint3 group_id : SV_GroupID, uint group_thread_index : SV_GroupIndex) {
  GPUWavefrontResources wavefront_resources = wavefront_load_resources();
  GPUUPBPResources resources = upbp_load_resources(wavefront_resources);
  if (group_id.x >= constants.dispatch_item_count) {
    return;
  }
  const uint query_index = constants.dispatch_item_offset + group_id.x;
  uint interval_index = query_index;
  if (constants.work_queue_index == GPUUPBPDensityQueryMode::Compacted) {
    const uint query_count = WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraMediumIntervalQuery * sizeof(uint));
    interval_index = ((query_index < query_count) && (resources.beam_buffer != kInvalidIndex))
                       ? WAVEFRONT_RO_BUFFER(resources.beam_buffer).Load((resources.beam_capacity + query_index) * sizeof(uint))
                       : kInvalidIndex;
  }
  if (group_thread_index == 0u) {
    upbp_bb1d_query_valid = 0u;
    const bool beam_index_valid =
      (resources.beam_index_mode == GPUUPBPBeamIndexMode::ComputeGrid) ? upbp_beam_grid_valid(resources.bb1d_beam_grid) : (resources.beam_acceleration_structure != kInvalidIndex);
    if ((resources.counter_buffer != kInvalidIndex) && beam_index_valid && (resources.bb1d_beam_buffer != kInvalidIndex)) {
      const uint interval_count = min(WAVEFRONT_RO_BUFFER(resources.counter_buffer).Load(GPUUPBPCounterIndex::CameraInterval * sizeof(uint)), resources.camera_interval_capacity);
      if ((interval_index < interval_count) && upbp_make_beam_from_interval(resources, interval_index, upbp_bb1d_camera_beam) &&
          upbp_prepare_beam(resources, upbp_bb1d_camera_beam, upbp_bb1d_prepared_camera)) {
        upbp_bb1d_camera_interval = upbp_load_interval(resources.interval_buffer, upbp_bb1d_camera_beam.interval_index);
        upbp_bb1d_context_vertex = upbp_load_vertex(resources.vertex_buffer, upbp_bb1d_camera_beam.source_vertex_index);
        upbp_bb1d_context_vertex.flags &= ~(GPUUPBPVertexFlags::Surface | GPUUPBPVertexFlags::Delta);
        upbp_bb1d_context_vertex.flags |= GPUUPBPVertexFlags::Medium | GPUUPBPVertexFlags::DensityConnectible;
        upbp_bb1d_query_valid = 1u;
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (upbp_bb1d_query_valid == 0u) {
    return;
  }
  SpectralResponse accumulated = spectral_response_zero(spectral_response_as_query(upbp_bb1d_prepared_camera.source_throughput));
  if (resources.beam_index_mode == GPUUPBPBeamIndexMode::ComputeGrid) {
    const GPUUPBPBeamGridResources grid = resources.bb1d_beam_grid;
    const GPUUPBPBeamGridMetadata metadata = upbp_load_beam_grid_metadata(grid);
    float minimum_distance = 0.0f;
    float maximum_distance = 0.0f;
    if (upbp_beam_grid_ray_range(metadata, upbp_bb1d_camera_beam, minimum_distance, maximum_distance)) {
      const uint3 resolution = uint3(metadata.resolution_x, metadata.resolution_y, metadata.resolution_z);
      int3 cell = int3(upbp_beam_grid_cell(metadata, upbp_bb1d_camera_beam.origin + upbp_bb1d_camera_beam.direction * minimum_distance));
      int3 step = int3(0, 0, 0);
      float3 next_distance = float3(kMaxFloat, kMaxFloat, kMaxFloat);
      float3 distance_step = float3(kMaxFloat, kMaxFloat, kMaxFloat);
      [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
        const float direction = upbp_bb1d_camera_beam.direction[axis];
        if (direction > 0.0f) {
          step[axis] = 1;
          const float boundary = metadata.minimum[axis] + (float)(cell[axis] + 1) / metadata.inverse_cell_size[axis];
          next_distance[axis] = (boundary - upbp_bb1d_camera_beam.origin[axis]) / direction;
          distance_step[axis] = 1.0f / (metadata.inverse_cell_size[axis] * direction);
        } else if (direction < 0.0f) {
          step[axis] = -1;
          const float boundary = metadata.minimum[axis] + (float)cell[axis] / metadata.inverse_cell_size[axis];
          next_distance[axis] = (boundary - upbp_bb1d_camera_beam.origin[axis]) / direction;
          distance_step[axis] = -1.0f / (metadata.inverse_cell_size[axis] * direction);
        }
      }
      ByteAddressBuffer offsets = WAVEFRONT_RO_BUFFER(grid.cell_offsets_buffer);
      ByteAddressBuffer indices = WAVEFRONT_RO_BUFFER(grid.beam_indices_buffer);
      for (;;) {
        const uint3 current_cell = uint3(cell);
        const uint cell_index = upbp_beam_grid_cell_index(metadata, current_cell);
        const uint first = offsets.Load(cell_index * sizeof(uint));
        const uint end = min(offsets.Load((cell_index + 1u) * sizeof(uint)), min(grid.beam_index_count, metadata.entry_count));
        for (uint offset = first + group_thread_index; offset < end; offset += 64u) {
          const uint beam_index = indices.Load(offset * sizeof(uint));
          if (beam_index < min(grid.beam_count, metadata.beam_count)) {
            const GPUUPBPBeamReference reference = upbp_load_bb1d_beam_reference(resources.bb1d_beam_buffer, beam_index);
            UPBPGPUBeamBeamIntersection intersection = (UPBPGPUBeamBeamIntersection)0;
            if (upbp_bb1d_density_candidate_intersects(resources, reference, upbp_bb1d_camera_beam, upbp_bb1d_camera_interval.medium_index, intersection)) {
              const float3 camera_intersection = upbp_bb1d_camera_beam.origin + upbp_bb1d_camera_beam.direction * intersection.second_distance;
              if (all(upbp_beam_grid_cell(metadata, camera_intersection) == current_cell)) {
                upbp_evaluate_bb1d_density_candidate(resources, upbp_bb1d_camera_beam, upbp_bb1d_prepared_camera, upbp_load_density_beam(resources.bb1d_beam_buffer, beam_index),
                  upbp_bb1d_camera_interval, upbp_bb1d_context_vertex, intersection, accumulated);
              }
            }
          }
        }
        const float next = min(next_distance.x, min(next_distance.y, next_distance.z));
        if (next > maximum_distance) {
          break;
        }
        bool inside = true;
        [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
          if (next_distance[axis] == next) {
            cell[axis] += step[axis];
            next_distance[axis] += distance_step[axis];
            inside = inside && (cell[axis] >= 0) && (cell[axis] < (int)resolution[axis]);
          }
        }
        if (inside == false) {
          break;
        }
      }
    }
  } else {
    RayDesc ray = (RayDesc)0;
    ray.Origin = upbp_bb1d_camera_beam.origin;
    ray.Direction = upbp_bb1d_camera_beam.direction;
    ray.TMin = 0.0f;
    ray.TMax = upbp_bb1d_camera_beam.length;
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    const uint partition_acceleration_structure = upbp_load_bb1d_partition_acceleration_structure(wavefront_resources, group_thread_index);
    const bool partition_valid = partition_acceleration_structure != kInvalidIndex;
    const uint query_acceleration_structure = partition_valid ? partition_acceleration_structure : resources.beam_acceleration_structure;
    const uint medium_mask = 1u << (upbp_bb1d_camera_interval.medium_index & 7u);
    query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(query_acceleration_structure)], RAY_FLAG_FORCE_NON_OPAQUE, medium_mask, ray);
    for (;;) {
      if (group_thread_index == 0u) {
        upbp_bb1d_exhausted_partition_count = 0u;
      }
      GroupMemoryBarrierWithGroupSync();
      bool candidate_found = false;
      uint beam_index = kInvalidIndex;
      UPBPGPUBeamBeamIntersection intersection = (UPBPGPUBeamBeamIntersection)0;
      uint partition_exhausted = partition_valid ? 0u : 1u;
      if (group_thread_index < kGPUUPBPBB1DPartitionCount) {
        [loop] for (;;) {
          if (partition_valid == false) {
            break;
          }
          if (query.Proceed() == false) {
            partition_exhausted = 1u;
            break;
          }
          if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            beam_index = query.CandidateInstanceID();
            const GPUUPBPBeamReference reference = upbp_load_bb1d_beam_reference(resources.bb1d_beam_buffer, beam_index);
            if (upbp_bb1d_density_candidate_intersects(resources, reference, upbp_bb1d_camera_beam, upbp_bb1d_camera_interval.medium_index, intersection)) {
              candidate_found = true;
              break;
            }
          }
        }
      }
      if (partition_exhausted != 0u) {
        uint ignored = 0u;
        InterlockedAdd(upbp_bb1d_exhausted_partition_count, 1u, ignored);
      }
      if (candidate_found) {
        upbp_evaluate_bb1d_density_candidate(resources, upbp_bb1d_camera_beam, upbp_bb1d_prepared_camera, upbp_load_density_beam(resources.bb1d_beam_buffer, beam_index),
          upbp_bb1d_camera_interval, upbp_bb1d_context_vertex, intersection, accumulated);
      }
      GroupMemoryBarrierWithGroupSync();
      if (upbp_bb1d_exhausted_partition_count == kGPUUPBPBB1DPartitionCount) {
        break;
      }
    }
  }
  upbp_bb1d_lane_contributions[group_thread_index] = upbp_pack_spectral_response(accumulated);
  GroupMemoryBarrierWithGroupSync();
  if (group_thread_index == 0u) {
    SpectralResponse total = spectral_response_zero(spectral_response_as_query(upbp_bb1d_prepared_camera.source_throughput));
    [unroll] for (uint lane_index = 0u; lane_index < 64u; ++lane_index) {
      total = spectral_response_add(total, upbp_unpack_spectral_response(upbp_bb1d_lane_contributions[lane_index]));
    }
    upbp_submit_camera_contribution(wavefront_resources, resources, GPUUPBPTechnique::BB1D, upbp_bb1d_camera_beam.global_path_index, total);
  }
}
#endif
