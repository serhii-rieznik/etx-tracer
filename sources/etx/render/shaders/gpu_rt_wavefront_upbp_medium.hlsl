#pragma once

#include <interop/medium_position_shared.hxx>

static const uint kUPBPMediumEscape = 0u;
static const uint kUPBPMediumScatter = 1u;
static const uint kUPBPMediumAbsorb = 2u;
static const uint kUPBPMediumNull = 3u;
static const uint kUPBPMediumFailure = 4u;

uint upbp_path_segment_count(GPUUPBPPathState path_state) {
  return path_state.transport_counts & 0xffffu;
}

uint upbp_path_boundary_count(GPUUPBPPathState path_state) {
  return path_state.transport_counts >> 16u;
}

void upbp_set_path_transport_counts(inout GPUUPBPPathState path_state, uint segment_count, uint boundary_count) {
  path_state.transport_counts = (segment_count & 0xffffu) | ((boundary_count & 0xffffu) << 16u);
}

bool upbp_append_light_beam(GPUUPBPResources resources, GPUUPBPPathState path_state, GPUUPBPInterval interval, uint transport_interval_index) {
  if ((resources.iteration.flags & GPUUPBPIterationFlags::CollectLightDensityRecords) == 0u) {
    return true;
  }
  if (((path_state.flags & GPUUPBPPathStateFlags::Light) == 0u) || (interval.medium_index == kInvalidIndex)) {
    return true;
  }
  const float3 delta = interval.end_position - interval.start_position;
  const float beam_length = length(delta);
  if (beam_length <= 0.0f) {
    return true;
  }
  const bool collect_bp2d = (resources.iteration.technique_mask & GPUUPBPTechnique::BP2D) != 0u;
  const bool selected_bb1d = ((resources.iteration.technique_mask & GPUUPBPTechnique::BB1D) != 0u) && (path_state.global_path_index < resources.iteration.bb1d_light_path_count);
  if ((collect_bp2d == false) && (selected_bb1d == false)) {
    return true;
  }
  uint beam_index = kInvalidIndex;
  if (upbp_append_index(resources, GPUUPBPCounterIndex::Beam, resources.beam_capacity, GPUUPBPOverflowFlags::Beam, beam_index) == false) {
    return false;
  }
  GPUUPBPBeam beam = (GPUUPBPBeam)0;
  beam.origin = interval.start_position;
  beam.length = beam_length;
  beam.direction = delta / beam_length;
  beam.flags = GPUUPBPBeamFlags::Valid | (selected_bb1d ? GPUUPBPBeamFlags::SelectedForBB1D : 0u);
  beam.source_vertex_index = path_state.last_vertex_index;
  beam.interval_index = path_state.current_interval_index;
  beam.global_path_index = path_state.global_path_index;
  beam.path_length = path_state.path_length;
  upbp_store_beam(resources.beam_buffer, beam_index, beam);
  return true;
}

float upbp_spectral_average(SpectralResponse value) {
  return spectral_response_is_spectral(value) ? value.value : ((value.integrated.x + value.integrated.y + value.integrated.z) / 3.0f);
}

float upbp_spectral_component(SpectralResponse value, uint component) {
  if (spectral_response_is_spectral(value)) {
    return value.value;
  }
  return component == 0u ? value.integrated.x : (component == 1u ? value.integrated.y : value.integrated.z);
}

bool upbp_spectral_non_negative(SpectralResponse value) {
  if (spectral_response_is_spectral(value)) {
    return isfinite(value.value) && (value.value >= 0.0f);
  }
  return all(isfinite(value.integrated)) && all(value.integrated >= 0.0f);
}

struct GPUUPBPConnectionInterval {
  SpectralResponse weight;
  float log_transport_pdf_forward;
  float log_transport_pdf_reverse;
};

bool upbp_track_connection_interval(uint medium_index, float3 origin, float3 direction, float maximum_distance, SpectralQuery spect, uint maximum_null_events, inout uint seed,
  out GPUUPBPConnectionInterval result, out uint terminal_type) {
  result = (GPUUPBPConnectionInterval)0;
  result.weight = spectral_response_make(spect, 1.0f);
  terminal_type = kUPBPMediumFailure;
  if (medium_index == kInvalidIndex) {
    terminal_type = kUPBPMediumEscape;
    return true;
  }

  MediumAccess medium_access = (MediumAccess)0;
  if (wavefront_try_load_medium(medium_index, medium_access) == false) {
    return false;
  }
  MediumAccessGPUContext access_context =
    make_medium_access_gpu_context(constants.scene.mediums, constants.scene.images, constants.scene.spectrums, constants.scene.spectral_values);
  const SpectralResponse scattering_base = gpu_medium_scattering(medium_access, spect);
  const SpectralResponse absorption_base = gpu_medium_absorption(medium_access, spect);
  if ((upbp_spectral_non_negative(scattering_base) == false) || (upbp_spectral_non_negative(absorption_base) == false)) {
    return false;
  }
  const SpectralResponse extinction_base = spectral_response_add(scattering_base, absorption_base);
  const float majorant = spectral_response_maximum(extinction_base);
  if ((majorant < 0.0f) || (isfinite(majorant) == false)) {
    return false;
  }
  if (majorant <= 0.0f) {
    terminal_type = kUPBPMediumEscape;
    return true;
  }

  float traveled_distance = 0.0f;
  uint null_event_count = 0u;
  for (;;) {
    const float random_distance = rnd01(seed) + (1.0f / 16777216.0f);
    const float sampled_distance = -log(1.0f - random_distance) / majorant;
    const float remaining_distance = maximum_distance - traveled_distance;
    const float event_distance = min(sampled_distance, remaining_distance);
    const float3 event_position = origin + direction * (traveled_distance + event_distance);
    const float majorant_transmittance = exp(-majorant * event_distance);
    traveled_distance += event_distance;

    if (sampled_distance >= remaining_distance) {
      const float log_transmittance = log(majorant_transmittance);
      result.log_transport_pdf_forward += log_transmittance;
      result.log_transport_pdf_reverse += log_transmittance;
      terminal_type = kUPBPMediumEscape;
      return true;
    }

    const float density = medium_access.medium_class == Medium::Homogeneous ? 1.0f : medium_access_sample_density(access_context, medium_access, event_position);
    if ((density < 0.0f) || (density > 1.0f) || (isfinite(density) == false)) {
      return false;
    }
    const SpectralResponse scattering = spectral_response_mul(scattering_base, density);
    const SpectralResponse absorption = spectral_response_mul(absorption_base, density);
    const SpectralResponse extinction = spectral_response_add(scattering, absorption);
    const SpectralResponse null_coefficient = spectral_response_sub(spectral_response_make(spect, majorant), extinction);
    if (upbp_spectral_non_negative(null_coefficient) == false) {
      return false;
    }
    const uint component_count = spectral_query_is_spectral(spect) ? 1u : 3u;
    const uint component = min(component_count - 1u, uint(rnd01(seed) * float(component_count)));
    const bool real_event = (rnd01(seed) * majorant) < upbp_spectral_component(extinction, component);
    SpectralResponse sampling_coefficient = null_coefficient;
    if (real_event) {
      sampling_coefficient = extinction;
    }
    const float mean_sampling_coefficient = upbp_spectral_average(sampling_coefficient);
    if ((mean_sampling_coefficient <= 0.0f) || (isfinite(mean_sampling_coefficient) == false)) {
      return false;
    }
    if (real_event) {
      const float log_transmittance = log(majorant_transmittance);
      result.log_transport_pdf_forward += log_transmittance;
      result.log_transport_pdf_reverse += log_transmittance;
      terminal_type = upbp_spectral_average(scattering) > 0.0f ? kUPBPMediumScatter : kUPBPMediumAbsorb;
      return true;
    }

    const float event_pdf = majorant_transmittance * mean_sampling_coefficient;
    const SpectralResponse event_weight = spectral_response_div(null_coefficient, mean_sampling_coefficient);
    const float log_event_pdf = log(event_pdf);
    result.log_transport_pdf_forward += log_event_pdf;
    result.log_transport_pdf_reverse += log_event_pdf;
    result.weight = spectral_response_mul(result.weight, event_weight);
    null_event_count += 1u;
    if (null_event_count > maximum_null_events) {
      return false;
    }
  }
}

bool upbp_track_connection_homogeneous_extinction(SpectralResponse extinction, float maximum_distance, SpectralQuery spect, uint maximum_null_events, inout uint seed,
  out GPUUPBPConnectionInterval result, out uint terminal_type) {
  result = (GPUUPBPConnectionInterval)0;
  result.weight = spectral_response_make(spect, 1.0f);
  terminal_type = kUPBPMediumFailure;
  if (upbp_spectral_non_negative(extinction) == false) {
    return false;
  }
  const float majorant = spectral_response_maximum(extinction);
  if ((majorant < 0.0f) || (isfinite(majorant) == false)) {
    return false;
  }
  if (majorant <= 0.0f) {
    terminal_type = kUPBPMediumEscape;
    return true;
  }

  float traveled_distance = 0.0f;
  uint null_event_count = 0u;
  for (;;) {
    const float random_distance = rnd01(seed) + (1.0f / 16777216.0f);
    const float sampled_distance = -log(1.0f - random_distance) / majorant;
    const float remaining_distance = maximum_distance - traveled_distance;
    const float event_distance = min(sampled_distance, remaining_distance);
    const float majorant_transmittance = exp(-majorant * event_distance);
    traveled_distance += event_distance;

    if (sampled_distance >= remaining_distance) {
      const float log_transmittance = log(majorant_transmittance);
      result.log_transport_pdf_forward += log_transmittance;
      result.log_transport_pdf_reverse += log_transmittance;
      terminal_type = kUPBPMediumEscape;
      return true;
    }

    const SpectralResponse null_coefficient = spectral_response_sub(spectral_response_make(spect, majorant), extinction);
    if (upbp_spectral_non_negative(null_coefficient) == false) {
      return false;
    }
    const uint component_count = spectral_query_is_spectral(spect) ? 1u : 3u;
    const uint component = min(component_count - 1u, uint(rnd01(seed) * float(component_count)));
    const bool real_event = (rnd01(seed) * majorant) < upbp_spectral_component(extinction, component);
    SpectralResponse sampling_coefficient = null_coefficient;
    if (real_event) {
      sampling_coefficient = extinction;
    }
    const float mean_sampling_coefficient = upbp_spectral_average(sampling_coefficient);
    if ((mean_sampling_coefficient <= 0.0f) || (isfinite(mean_sampling_coefficient) == false)) {
      return false;
    }
    if (real_event) {
      const float log_transmittance = log(majorant_transmittance);
      result.log_transport_pdf_forward += log_transmittance;
      result.log_transport_pdf_reverse += log_transmittance;
      terminal_type = kUPBPMediumScatter;
      return true;
    }

    const float event_pdf = majorant_transmittance * mean_sampling_coefficient;
    const SpectralResponse event_weight = spectral_response_div(null_coefficient, mean_sampling_coefficient);
    const float log_event_pdf = log(event_pdf);
    result.log_transport_pdf_forward += log_event_pdf;
    result.log_transport_pdf_reverse += log_event_pdf;
    result.weight = spectral_response_mul(result.weight, event_weight);
    null_event_count += 1u;
    if (null_event_count > maximum_null_events) {
      return false;
    }
  }
}

bool upbp_begin_transport_segment(GPUUPBPResources resources, bool from_camera, uint path_index, SpectralQuery spect, inout GPUUPBPPathState path_state) {
  uint segment_index = kInvalidIndex;
  if (upbp_append_partitioned_index(resources, from_camera, GPUUPBPCounterIndex::LightSegment, GPUUPBPCounterIndex::CameraSegment, resources.light_segment_capacity,
        resources.camera_segment_capacity, GPUUPBPOverflowFlags::Segment, segment_index) == false) {
    return false;
  }
  GPUUPBPSegment segment = (GPUUPBPSegment)0;
  segment.weight = upbp_pack_spectral_response(spectral_response_make(spect, 1.0f));
  segment.first_interval_index = kInvalidIndex;
  segment.source_vertex_index = path_state.last_vertex_index;
  segment.target_vertex_index = kInvalidIndex;
  segment.flags = GPUUPBPSegmentFlags::Valid;
  upbp_store_segment(resources.segment_buffer, segment_index, segment);
  if (path_state.last_vertex_index != kInvalidIndex) {
    GPUUPBPVertex source = upbp_load_vertex(resources.vertex_buffer, path_state.last_vertex_index);
    source.flags |= GPUUPBPVertexFlags::HasDeparture;
    upbp_store_vertex(resources.vertex_buffer, path_state.last_vertex_index, source);
  }
  path_state.current_segment_index = segment_index;
  path_state.current_interval_index = kInvalidIndex;
  upbp_set_path_transport_counts(path_state, upbp_path_segment_count(path_state) + 1u, upbp_path_boundary_count(path_state));
  upbp_store_path_state(resources.path_state_buffer, upbp_path_state_index(resources, from_camera, path_index), path_state);
  return true;
}

void upbp_record_transport_boundary(GPUUPBPResources resources, inout GPUUPBPPathState path_state) {
  GPUUPBPSegment segment = upbp_load_segment(resources.segment_buffer, path_state.current_segment_index);
  segment.boundary_count += 1u;
  upbp_store_segment(resources.segment_buffer, path_state.current_segment_index, segment);
  upbp_set_path_transport_counts(path_state, upbp_path_segment_count(path_state), upbp_path_boundary_count(path_state) + 1u);
}

void upbp_mark_terminal_transport_segment(GPUUPBPResources resources, bool from_camera, uint path_index, inout GPUUPBPPathState path_state) {
  GPUUPBPSegment segment = upbp_load_segment(resources.segment_buffer, path_state.current_segment_index);
  segment.flags |= GPUUPBPSegmentFlags::Terminal;
  upbp_store_segment(resources.segment_buffer, path_state.current_segment_index, segment);
  path_state.flags |= GPUUPBPPathStateFlags::HasTerminalSegment;
  upbp_store_path_state(resources.path_state_buffer, upbp_path_state_index(resources, from_camera, path_index), path_state);
}

float upbp_distance_to_scene_sphere_exit(float3 origin, float3 direction) {
  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  const float3 offset = origin - globals_data.bounding_sphere_center;
  const float projected = dot(direction, offset);
  const float discriminant = projected * projected - dot(offset, offset) + globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius;
  return discriminant < 0.0f ? 0.0f : (-projected + sqrt(discriminant));
}

bool upbp_append_tracking_event(GPUUPBPResources resources, bool from_camera, uint interval_index, GPUUPBPTrackingEvent event_record, inout GPUUPBPInterval interval,
  inout uint previous_event_index) {
  if ((from_camera == false) && ((resources.iteration.flags & GPUUPBPIterationFlags::CollectLightDensityRecords) == 0u)) {
    return true;
  }
  const uint required_techniques = from_camera ? (GPUUPBPTechnique::PB2D | GPUUPBPTechnique::BB1D) : (GPUUPBPTechnique::BP2D | GPUUPBPTechnique::BB1D);
  if ((resources.iteration.technique_mask & required_techniques) == 0u) {
    return true;
  }
  if ((interval.flags & GPUUPBPIntervalFlags::RecomputeTracking) != 0u) {
    return true;
  }
  uint event_index = kInvalidIndex;
  if (upbp_try_append_partitioned_index_wave(resources, from_camera, GPUUPBPCounterIndex::LightEvent, GPUUPBPCounterIndex::CameraEvent, resources.light_event_capacity,
        resources.camera_event_capacity, event_index) == false) {
    interval.flags |= GPUUPBPIntervalFlags::RecomputeTracking;
    interval.first_event_index = kInvalidIndex;
    interval.event_count = 0u;
    previous_event_index = kInvalidIndex;
    return true;
  }
  event_record.interval_index = interval_index;
  event_record.next_event_index = kInvalidIndex;
  upbp_store_tracking_event(resources.event_buffer, event_index, event_record);
  if (interval.first_event_index == kInvalidIndex) {
    interval.first_event_index = event_index;
  }
  if (previous_event_index != kInvalidIndex) {
    upbp_store_tracking_event_next_index(resources.event_buffer, previous_event_index, event_index);
  }
  previous_event_index = event_index;
  interval.event_count += 1u;
  return true;
}

bool upbp_finish_interval(GPUUPBPResources resources, bool from_camera, uint path_index, inout GPUUPBPPathState path_state, GPUUPBPInterval interval) {
  upbp_store_interval(resources.interval_buffer, path_state.current_interval_index, interval);
  GPUUPBPSegment segment = upbp_load_segment(resources.segment_buffer, path_state.current_segment_index);
  if (segment.first_interval_index == kInvalidIndex) {
    segment.first_interval_index = path_state.current_interval_index;
  }
  const uint transport_interval_index = segment.interval_count;
  segment.interval_count += 1u;
  segment.weight = upbp_pack_spectral_response(spectral_response_mul(upbp_unpack_spectral_response(segment.weight), upbp_unpack_spectral_response(interval.weight)));
  segment.log_pdf_forward += interval.log_pdf_forward;
  segment.log_pdf_reverse += interval.log_pdf_reverse;
  segment.log_transport_pdf_forward += interval.log_transport_pdf_forward;
  segment.log_transport_pdf_reverse += interval.log_transport_pdf_reverse;
  segment.distance += interval.distance;
  if ((interval.flags & (GPUUPBPIntervalFlags::Scatter | GPUUPBPIntervalFlags::Absorb)) != 0u) {
    segment.log_terminal_event_density = interval.log_terminal_event_density;
    segment.flags |= GPUUPBPSegmentFlags::HasTerminalEventDensity;
  }
  upbp_store_segment(resources.segment_buffer, path_state.current_segment_index, segment);
  if (upbp_append_light_beam(resources, path_state, interval, transport_interval_index) == false) {
    return false;
  }
  upbp_store_path_state(resources.path_state_buffer, upbp_path_state_index(resources, from_camera, path_index), path_state);
  return true;
}

bool upbp_track_interval(GPUUPBPResources resources, bool from_camera, uint path_index, uint medium_index, float3 origin, float3 direction, float maximum_distance,
  float3 surface_position, float3 surface_normal, SpectralQuery spect, inout uint seed, inout GPUUPBPPathState path_state, out MediumSample medium_sample, out uint terminal_type) {
  medium_sample = (MediumSample)0;
  medium_sample.weight = spectral_response_make(spect, 1.0f);
  medium_sample.pos = origin + direction * maximum_distance;
  terminal_type = kUPBPMediumFailure;
  uint interval_index = kInvalidIndex;
  if (upbp_append_partitioned_index(resources, from_camera, GPUUPBPCounterIndex::LightInterval, GPUUPBPCounterIndex::CameraInterval, resources.light_interval_capacity,
        resources.camera_interval_capacity, GPUUPBPOverflowFlags::Interval, interval_index) == false) {
    return false;
  }

  GPUUPBPInterval interval = (GPUUPBPInterval)0;
  interval.weight = upbp_pack_spectral_response(spectral_response_make(spect, 1.0f));
  interval.start_position = origin;
  interval.end_position = origin;
  interval.medium_index = medium_index;
  interval.flags = GPUUPBPIntervalFlags::Valid;
  interval.first_event_index = kInvalidIndex;
  interval.segment_index = path_state.current_segment_index;
  interval.next_interval_index = kInvalidIndex;
  interval.tracking_seed = seed;
  if (path_state.current_interval_index != kInvalidIndex) {
    GPUUPBPInterval previous_interval = upbp_load_interval(resources.interval_buffer, path_state.current_interval_index);
    previous_interval.next_interval_index = interval_index;
    upbp_store_interval(resources.interval_buffer, path_state.current_interval_index, previous_interval);
  }
  path_state.current_interval_index = interval_index;

  if (medium_index == kInvalidIndex) {
    interval.flags |= GPUUPBPIntervalFlags::Vacuum | GPUUPBPIntervalFlags::Escape;
    interval.end_position = medium_sample.pos;
    interval.distance = maximum_distance;
    terminal_type = kUPBPMediumEscape;
    return upbp_finish_interval(resources, from_camera, path_index, path_state, interval);
  }

  MediumAccess medium_access = (MediumAccess)0;
  if (wavefront_try_load_medium(medium_index, medium_access) == false) {
    return false;
  }
  MediumAccessGPUContext access_context =
    make_medium_access_gpu_context(constants.scene.mediums, constants.scene.images, constants.scene.spectrums, constants.scene.spectral_values);
  const SpectralResponse scattering_base = gpu_medium_scattering(medium_access, spect);
  const SpectralResponse absorption_base = gpu_medium_absorption(medium_access, spect);
  if ((upbp_spectral_non_negative(scattering_base) == false) || (upbp_spectral_non_negative(absorption_base) == false)) {
    return false;
  }
  const SpectralResponse extinction_base = spectral_response_add(scattering_base, absorption_base);
  const float majorant = spectral_response_maximum(extinction_base);
  if ((majorant < 0.0f) || (isfinite(majorant) == false)) {
    return false;
  }
  if (majorant <= 0.0f) {
    interval.flags |= GPUUPBPIntervalFlags::Escape;
    interval.end_position = medium_sample.pos;
    interval.distance = maximum_distance;
    terminal_type = kUPBPMediumEscape;
    return upbp_finish_interval(resources, from_camera, path_index, path_state, interval);
  }

  SpectralResponse interval_weight = spectral_response_make(spect, 1.0f);
  float traveled_distance = 0.0f;
  uint previous_event_index = kInvalidIndex;
  uint null_event_count = 0u;
  for (;;) {
    const float random_distance = rnd01(seed) + (1.0f / 16777216.0f);
    const float sampled_distance = -log(1.0f - random_distance) / majorant;
    const float remaining_distance = maximum_distance - traveled_distance;
    const float event_distance = min(sampled_distance, remaining_distance);
    const float3 event_position = origin + direction * (traveled_distance + event_distance);
    const float majorant_transmittance = exp(-majorant * event_distance);

    GPUUPBPTrackingEvent event_record = (GPUUPBPTrackingEvent)0;
    event_record.weight_before = upbp_pack_spectral_response(interval_weight);
    event_record.log_transport_pdf_forward_before = interval.log_transport_pdf_forward;
    event_record.log_transport_pdf_reverse_before = interval.log_transport_pdf_reverse;
    event_record.distance_before = traveled_distance;
    event_record.end_distance = traveled_distance + event_distance;
    event_record.majorant = majorant;
    if (upbp_append_tracking_event(resources, from_camera, interval_index, event_record, interval, previous_event_index) == false) {
      return false;
    }
    traveled_distance += event_distance;
    interval.end_position = event_position;
    interval.log_pdf_forward += log(majorant_transmittance);
    interval.log_pdf_reverse += log(majorant_transmittance);

    if (sampled_distance >= remaining_distance) {
      interval.log_transport_pdf_forward += log(majorant_transmittance);
      interval.log_transport_pdf_reverse += log(majorant_transmittance);
      interval.flags |= GPUUPBPIntervalFlags::Escape;
      interval.distance = traveled_distance;
      interval.weight = upbp_pack_spectral_response(interval_weight);
      medium_sample.weight = interval_weight;
      medium_sample.pos = event_position;
      terminal_type = kUPBPMediumEscape;
      return upbp_finish_interval(resources, from_camera, path_index, path_state, interval);
    }

    const float density = medium_access.medium_class == Medium::Homogeneous ? 1.0f : medium_access_sample_density(access_context, medium_access, event_position);
    if ((density < 0.0f) || (density > 1.0f) || (isfinite(density) == false)) {
      return false;
    }
    const SpectralResponse scattering = spectral_response_mul(scattering_base, density);
    const SpectralResponse absorption = spectral_response_mul(absorption_base, density);
    const SpectralResponse extinction = spectral_response_add(scattering, absorption);
    const SpectralResponse null_coefficient = spectral_response_sub(spectral_response_make(spect, majorant), extinction);
    if (upbp_spectral_non_negative(null_coefficient) == false) {
      return false;
    }
    const uint component_count = spectral_query_is_spectral(spect) ? 1u : 3u;
    const uint component = min(component_count - 1u, uint(rnd01(seed) * float(component_count)));
    const bool real_event = (rnd01(seed) * majorant) < upbp_spectral_component(extinction, component);
    SpectralResponse sampling_coefficient = null_coefficient;
    SpectralResponse coefficient = null_coefficient;
    if (real_event) {
      sampling_coefficient = extinction;
      coefficient = scattering;
    }
    const float mean_sampling_coefficient = upbp_spectral_average(sampling_coefficient);
    if ((mean_sampling_coefficient <= 0.0f) || (isfinite(mean_sampling_coefficient) == false)) {
      return false;
    }
    const float event_pdf = majorant_transmittance * mean_sampling_coefficient;
    interval.log_pdf_forward += log(mean_sampling_coefficient);
    interval.log_pdf_reverse += log(mean_sampling_coefficient);
    const SpectralResponse event_weight = spectral_response_div(coefficient, mean_sampling_coefficient);
    if (real_event) {
      interval.log_transport_pdf_forward += log(majorant_transmittance);
      interval.log_transport_pdf_reverse += log(majorant_transmittance);
      interval.log_terminal_event_density = log(mean_sampling_coefficient);
      interval_weight = spectral_response_mul(interval_weight, event_weight);
      interval.weight = upbp_pack_spectral_response(interval_weight);
      interval.distance = traveled_distance;
      const bool scatter = upbp_spectral_average(scattering) > 0.0f;
      interval.flags |= scatter ? GPUUPBPIntervalFlags::Scatter : GPUUPBPIntervalFlags::Absorb;
      if (scatter) {
        interval.end_position = medium_position_before_surface(event_position, surface_position, surface_normal, direction);
      }
      medium_sample.weight = interval_weight;
      medium_sample.pos = interval.end_position;
      medium_sample.sampled_medium_t = traveled_distance;
      terminal_type = scatter ? kUPBPMediumScatter : kUPBPMediumAbsorb;
      return upbp_finish_interval(resources, from_camera, path_index, path_state, interval);
    }

    interval.log_transport_pdf_forward += log(event_pdf);
    interval.log_transport_pdf_reverse += log(event_pdf);
    interval_weight = spectral_response_mul(interval_weight, event_weight);
    null_event_count += 1u;
    if (null_event_count > resources.iteration.maximum_null_events_per_interval) {
      return false;
    }
  }
}

bool upbp_track_homogeneous_interval(GPUUPBPResources resources, bool from_camera, uint path_index, uint material_index, SpectralResponse scattering, SpectralResponse absorption,
  float3 origin, float3 direction, float maximum_distance, float3 surface_position, float3 surface_normal, SpectralQuery spect, inout uint seed, inout GPUUPBPPathState path_state,
  out MediumSample medium_sample, out uint terminal_type) {
  medium_sample = (MediumSample)0;
  medium_sample.weight = spectral_response_make(spect, 1.0f);
  medium_sample.pos = origin + direction * maximum_distance;
  terminal_type = kUPBPMediumFailure;

  uint interval_index = kInvalidIndex;
  if (upbp_append_partitioned_index(resources, from_camera, GPUUPBPCounterIndex::LightInterval, GPUUPBPCounterIndex::CameraInterval, resources.light_interval_capacity,
        resources.camera_interval_capacity, GPUUPBPOverflowFlags::Interval, interval_index) == false) {
    return false;
  }

  GPUUPBPInterval interval = (GPUUPBPInterval)0;
  interval.weight = upbp_pack_spectral_response(spectral_response_make(spect, 1.0f));
  interval.start_position = origin;
  interval.end_position = origin;
  interval.medium_index = upbp_inline_medium_key(material_index);
  interval.flags = GPUUPBPIntervalFlags::Valid | GPUUPBPIntervalFlags::InlineMedium;
  interval.inline_scattering = upbp_pack_spectral_response(scattering);
  interval.inline_absorption = upbp_pack_spectral_response(absorption);
  interval.first_event_index = kInvalidIndex;
  interval.segment_index = path_state.current_segment_index;
  interval.next_interval_index = kInvalidIndex;
  interval.tracking_seed = seed;
  if (path_state.current_interval_index != kInvalidIndex) {
    GPUUPBPInterval previous_interval = upbp_load_interval(resources.interval_buffer, path_state.current_interval_index);
    previous_interval.next_interval_index = interval_index;
    upbp_store_interval(resources.interval_buffer, path_state.current_interval_index, previous_interval);
  }
  path_state.current_interval_index = interval_index;

  const SpectralResponse extinction = spectral_response_add(scattering, absorption);
  const float majorant = spectral_response_maximum(extinction);
  if ((upbp_spectral_non_negative(scattering) == false) || (upbp_spectral_non_negative(absorption) == false) || (majorant <= 0.0f) || (isfinite(majorant) == false)) {
    return false;
  }

  SpectralResponse interval_weight = spectral_response_make(spect, 1.0f);
  float traveled_distance = 0.0f;
  uint previous_event_index = kInvalidIndex;
  uint null_event_count = 0u;
  for (;;) {
    const float random_distance = rnd01(seed) + (1.0f / 16777216.0f);
    const float sampled_distance = -log(1.0f - random_distance) / majorant;
    const float remaining_distance = maximum_distance - traveled_distance;
    const float event_distance = min(sampled_distance, remaining_distance);
    const float3 event_position = origin + direction * (traveled_distance + event_distance);
    const float majorant_transmittance = exp(-majorant * event_distance);

    GPUUPBPTrackingEvent event_record = (GPUUPBPTrackingEvent)0;
    event_record.weight_before = upbp_pack_spectral_response(interval_weight);
    event_record.log_transport_pdf_forward_before = interval.log_transport_pdf_forward;
    event_record.log_transport_pdf_reverse_before = interval.log_transport_pdf_reverse;
    event_record.distance_before = traveled_distance;
    event_record.end_distance = traveled_distance + event_distance;
    event_record.majorant = majorant;
    if (upbp_append_tracking_event(resources, from_camera, interval_index, event_record, interval, previous_event_index) == false) {
      return false;
    }
    traveled_distance += event_distance;
    interval.end_position = event_position;
    interval.log_pdf_forward += log(majorant_transmittance);
    interval.log_pdf_reverse += log(majorant_transmittance);

    if (sampled_distance >= remaining_distance) {
      interval.log_transport_pdf_forward += log(majorant_transmittance);
      interval.log_transport_pdf_reverse += log(majorant_transmittance);
      interval.flags |= GPUUPBPIntervalFlags::Escape;
      interval.distance = traveled_distance;
      interval.weight = upbp_pack_spectral_response(interval_weight);
      medium_sample.weight = interval_weight;
      medium_sample.pos = event_position;
      terminal_type = kUPBPMediumEscape;
      return upbp_finish_interval(resources, from_camera, path_index, path_state, interval);
    }

    const SpectralResponse null_coefficient = spectral_response_sub(spectral_response_make(spect, majorant), extinction);
    if (upbp_spectral_non_negative(null_coefficient) == false) {
      return false;
    }
    const uint component_count = spectral_query_is_spectral(spect) ? 1u : 3u;
    const uint component = min(component_count - 1u, uint(rnd01(seed) * float(component_count)));
    const bool real_event = (rnd01(seed) * majorant) < upbp_spectral_component(extinction, component);
    SpectralResponse sampling_coefficient = null_coefficient;
    SpectralResponse coefficient = null_coefficient;
    if (real_event) {
      sampling_coefficient = extinction;
      coefficient = scattering;
    }
    const float mean_sampling_coefficient = upbp_spectral_average(sampling_coefficient);
    if ((mean_sampling_coefficient <= 0.0f) || (isfinite(mean_sampling_coefficient) == false)) {
      return false;
    }
    const float event_pdf = majorant_transmittance * mean_sampling_coefficient;
    interval.log_pdf_forward += log(mean_sampling_coefficient);
    interval.log_pdf_reverse += log(mean_sampling_coefficient);
    const SpectralResponse event_weight = spectral_response_div(coefficient, mean_sampling_coefficient);
    if (real_event) {
      interval.log_transport_pdf_forward += log(majorant_transmittance);
      interval.log_transport_pdf_reverse += log(majorant_transmittance);
      interval.log_terminal_event_density = log(mean_sampling_coefficient);
      interval_weight = spectral_response_mul(interval_weight, event_weight);
      interval.weight = upbp_pack_spectral_response(interval_weight);
      interval.distance = traveled_distance;
      const bool scatter = upbp_spectral_average(scattering) > 0.0f;
      interval.flags |= scatter ? GPUUPBPIntervalFlags::Scatter : GPUUPBPIntervalFlags::Absorb;
      if (scatter) {
        interval.end_position = medium_position_before_surface(event_position, surface_position, surface_normal, direction);
      }
      medium_sample.weight = interval_weight;
      medium_sample.pos = interval.end_position;
      medium_sample.sampled_medium_t = traveled_distance;
      terminal_type = scatter ? kUPBPMediumScatter : kUPBPMediumAbsorb;
      return upbp_finish_interval(resources, from_camera, path_index, path_state, interval);
    }

    interval.log_transport_pdf_forward += log(event_pdf);
    interval.log_transport_pdf_reverse += log(event_pdf);
    interval_weight = spectral_response_mul(interval_weight, event_weight);
    null_event_count += 1u;
    if (null_event_count > resources.iteration.maximum_null_events_per_interval) {
      return false;
    }
  }
}
