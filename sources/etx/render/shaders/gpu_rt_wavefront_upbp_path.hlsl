#pragma once

GPUWavefrontPathVertex upbp_make_wavefront_path_vertex(GPUUPBPVertex vertex, bool from_camera, uint pixel_index) {
  GPUWavefrontPathVertex result = (GPUWavefrontPathVertex)0;
  result.throughput = upbp_unpack_spectral_response(vertex.throughput);
  result.position = vertex.position;
  result.triangle_index = vertex.triangle_index;
  result.normal = vertex.normal;
  result.material_index = vertex.material_index;
  result.geo_normal = vertex.geo_normal;
  result.medium_index = vertex.medium_index;
  result.w_i = vertex.w_i;
  result.emitter_index = vertex.emitter_index;
  result.texcoord = vertex.texcoord;
  result.sampled_bsdf_pdf = vertex.scatter_pdf_forward;
  result.path_length = vertex.path_length;
  result.pixel_index = pixel_index;
  result.barycentric = vertex.barycentric.yz;
  result.instance_index = vertex.instance_index;
  result.flags = ((vertex.flags & GPUUPBPVertexFlags::Valid) != 0u) ? GPUWavefrontVertexFlags::Valid : 0u;
  result.flags |= ((vertex.flags & GPUUPBPVertexFlags::Connectible) != 0u) ? GPUWavefrontVertexFlags::Connectible : 0u;
  result.flags |= ((vertex.flags & GPUUPBPVertexFlags::Connectible) != 0u) ? GPUWavefrontVertexFlags::Mis_connectible : 0u;
  result.flags |= ((vertex.flags & GPUUPBPVertexFlags::Delta) != 0u) ? GPUWavefrontVertexFlags::Delta : 0u;
  result.flags |= ((vertex.flags & GPUUPBPVertexFlags::Surface) != 0u) ? GPUWavefrontVertexFlags::Surface : 0u;
  result.flags |= ((vertex.flags & GPUUPBPVertexFlags::Medium) != 0u) ? GPUWavefrontVertexFlags::Medium : 0u;
  result.flags |= ((vertex.flags & GPUUPBPVertexFlags::Emitter) != 0u) ? GPUWavefrontVertexFlags::Emitter : 0u;
  result.flags |= ((vertex.flags & GPUUPBPVertexFlags::Camera) != 0u) ? GPUWavefrontVertexFlags::Camera : 0u;
  result.flags |= ((vertex.flags & GPUUPBPVertexFlags::InlineMedium) != 0u) ? GPUWavefrontVertexFlags::Subsurface : 0u;
  if ((vertex.flags & GPUUPBPVertexFlags::InlineMedium) != 0u) {
    result.inline_medium_extinction = upbp_unpack_spectral_response(vertex.inline_extinction);
    result.inline_medium_flags = GPUWavefrontSubsurfaceFlags::InlineMedium;
  }
  result.flags |= from_camera ? GPUWavefrontVertexFlags::From_camera : GPUWavefrontVertexFlags::From_light;
  return result;
}

bool upbp_append_vertex(GPUUPBPResources resources, bool from_camera, GPUUPBPVertex vertex, out uint vertex_index) {
  vertex.flags = (vertex.flags & ~GPUUPBPVertexFlags::FromLight) | (from_camera ? 0u : GPUUPBPVertexFlags::FromLight);
  if (upbp_append_partitioned_index(resources, from_camera, GPUUPBPCounterIndex::LightVertex, GPUUPBPCounterIndex::CameraVertex, resources.light_vertex_capacity,
        resources.camera_vertex_capacity, GPUUPBPOverflowFlags::Vertex, vertex_index) == false) {
    return false;
  }
  upbp_store_vertex(resources.vertex_buffer, vertex_index, vertex);
  return true;
}

bool upbp_append_light_point(GPUUPBPResources resources, uint vertex_index, GPUUPBPVertex vertex) {
  if ((resources.iteration.flags & GPUUPBPIterationFlags::CollectLightDensityRecords) == 0u) {
    return true;
  }
  const bool surface_point = upbp_vertex_is_surface(vertex) && (upbp_vertex_is_delta(vertex) == false) && ((vertex.flags & GPUUPBPVertexFlags::DensityConnectible) != 0u) &&
                             ((resources.iteration.technique_mask & GPUUPBPTechnique::Surface) != 0u);
  const bool medium_point = upbp_vertex_is_medium(vertex) && (upbp_vertex_is_delta(vertex) == false) && ((vertex.flags & GPUUPBPVertexFlags::DensityConnectible) != 0u) &&
                            (vertex.medium_index != kInvalidIndex) && ((resources.iteration.technique_mask & (GPUUPBPTechnique::PP3D | GPUUPBPTechnique::PB2D)) != 0u);
  if ((surface_point == false) && (medium_point == false)) {
    return true;
  }
  uint point_index = kInvalidIndex;
  if (upbp_append_index(resources, GPUUPBPCounterIndex::Point, resources.point_capacity, GPUUPBPOverflowFlags::Point, point_index) == false) {
    return false;
  }
  GPUUPBPPoint point_record = (GPUUPBPPoint)0;
  point_record.position = vertex.position;
  point_record.vertex_index = vertex_index;
  upbp_store_point(resources.point_buffer, point_index, point_record);
  RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(resources.counter_buffer);
  uint ignored = 0u;
  counters.InterlockedAdd((surface_point ? GPUUPBPCounterIndex::SurfacePoint : GPUUPBPCounterIndex::MediumPoint) * 4u, 1u, ignored);
  return true;
}

bool upbp_append_physical_vertex(GPUUPBPResources resources, bool from_camera, uint path_index, inout GPUUPBPPathState path_state, inout GPUUPBPVertex vertex) {
  if ((path_state.last_vertex_index == kInvalidIndex) || (path_state.current_segment_index == kInvalidIndex)) {
    path_state.recursive_state.failure = GPUUPBPRecursiveFailure::InvalidPath;
    path_state.recursive_state.failure_vertex_index = 1u;
    return false;
  }
  const GPUUPBPVertex source = upbp_load_vertex(resources.vertex_buffer, path_state.last_vertex_index);
  GPUUPBPSegment segment = upbp_load_segment(resources.segment_buffer, path_state.current_segment_index);
  const uint structural_failure = ((source.flags & GPUUPBPVertexFlags::Valid) == 0u ? 1u : 0u) | ((segment.flags & GPUUPBPSegmentFlags::Valid) == 0u ? 2u : 0u) |
                                  (segment.source_vertex_index != path_state.last_vertex_index ? 4u : 0u) | (segment.target_vertex_index != kInvalidIndex ? 8u : 0u);
  if (structural_failure != 0u) {
    path_state.recursive_state.failure = GPUUPBPRecursiveFailure::InvalidPath;
    path_state.recursive_state.failure_vertex_index = 2u | (structural_failure << 8u);
    return false;
  }

  const uint vertex_path_length = path_state.path_length + 1u;
  vertex.flags |= GPUUPBPVertexFlags::Valid;
  vertex.previous_vertex_index = path_state.last_vertex_index;
  vertex.incoming_segment_index = path_state.current_segment_index;
  vertex.path_length = vertex_path_length;
  vertex.global_path_index = path_state.global_path_index;
  if (upbp_complete_recursive_arrival(source, vertex, segment, resources.iteration, vertex_path_length, path_state.recursive_state) == false) {
    return false;
  }
  vertex.arrival_weights = path_state.recursive_state.weights;
  vertex.departure_state = path_state.recursive_state;

  uint vertex_index = kInvalidIndex;
  if (upbp_append_vertex(resources, from_camera, vertex, vertex_index) == false) {
    path_state.recursive_state.failure = GPUUPBPRecursiveFailure::InvalidPath;
    path_state.recursive_state.failure_vertex_index = 3u;
    return false;
  }
  segment.target_vertex_index = vertex_index;
  segment.flags &= ~GPUUPBPSegmentFlags::Terminal;
  upbp_store_segment(resources.segment_buffer, path_state.current_segment_index, segment);
  path_state.last_vertex_index = vertex_index;
  path_state.path_length = vertex_path_length;
  path_state.current_segment_index = kInvalidIndex;
  path_state.current_interval_index = kInvalidIndex;
  upbp_store_path_state(resources.path_state_buffer, upbp_path_state_index(resources, from_camera, path_index), path_state);
  if (from_camera == false) {
    uint previous_maximum = 0u;
    WAVEFRONT_RW_BUFFER(resources.counter_buffer).InterlockedMax(GPUUPBPCounterIndex::MaximumLightPathLength * sizeof(uint), vertex_path_length, previous_maximum);
  }
  return true;
}

bool upbp_terminate_degenerate_arrival(GPUUPBPResources resources, bool from_camera, uint path_index, inout GPUUPBPPathState path_state) {
  if ((path_state.recursive_state.failure != GPUUPBPRecursiveFailure::InvalidMeasureCosine) || (path_state.current_segment_index == kInvalidIndex)) {
    return false;
  }
  GPUUPBPSegment segment = upbp_load_segment(resources.segment_buffer, path_state.current_segment_index);
  if (((segment.flags & GPUUPBPSegmentFlags::Valid) == 0u) || (segment.source_vertex_index != path_state.last_vertex_index) || (segment.target_vertex_index != kInvalidIndex)) {
    return false;
  }
  segment.flags |= GPUUPBPSegmentFlags::Terminal;
  upbp_store_segment(resources.segment_buffer, path_state.current_segment_index, segment);
  path_state.flags |= GPUUPBPPathStateFlags::HasTerminalSegment;
  path_state.current_interval_index = kInvalidIndex;
  upbp_store_path_state(resources.path_state_buffer, upbp_path_state_index(resources, from_camera, path_index), path_state);
  return true;
}

bool upbp_finalize_physical_vertex(GPUUPBPResources resources, bool from_camera, uint path_index, SpectralResponse outgoing_throughput, bool has_departure, out bool terminal) {
  terminal = false;
  GPUUPBPPathState path_state = upbp_load_path_state(resources.path_state_buffer, upbp_path_state_index(resources, from_camera, path_index));
  if (((path_state.flags & GPUUPBPPathStateFlags::Valid) == 0u) || (path_state.last_vertex_index == kInvalidIndex) || (path_state.path_length == 0u)) {
    return false;
  }
  GPUUPBPVertex vertex = upbp_load_vertex(resources.vertex_buffer, path_state.last_vertex_index);
  if (((vertex.flags & GPUUPBPVertexFlags::Valid) == 0u) || (vertex.path_length != path_state.path_length)) {
    return false;
  }
  vertex.outgoing_throughput = upbp_pack_spectral_response(outgoing_throughput);
  if (has_departure) {
    if (upbp_prepare_recursive_departure(vertex, resources.iteration, path_state.path_length, path_state.recursive_state) == false) {
      if (path_state.recursive_state.failure == GPUUPBPRecursiveFailure::InvalidMeasureCosine) {
        vertex.flags &= ~GPUUPBPVertexFlags::HasDeparture;
        upbp_store_vertex(resources.vertex_buffer, path_state.last_vertex_index, vertex);
        upbp_store_path_state(resources.path_state_buffer, upbp_path_state_index(resources, from_camera, path_index), path_state);
        terminal = true;
        return true;
      }
      return false;
    }
    vertex.flags |= GPUUPBPVertexFlags::HasDeparture;
    vertex.departure_state = path_state.recursive_state;
  } else {
    vertex.flags &= ~GPUUPBPVertexFlags::HasDeparture;
  }
  // Merging uses the arrival; survival of the following roulette decision is independent.
  if (((path_state.flags & GPUUPBPPathStateFlags::Light) != 0u) && (upbp_append_light_point(resources, path_state.last_vertex_index, vertex) == false)) {
    path_state.recursive_state.failure = GPUUPBPRecursiveFailure::InvalidPath;
    path_state.recursive_state.failure_vertex_index = 4u;
    return false;
  }
  upbp_store_vertex(resources.vertex_buffer, path_state.last_vertex_index, vertex);
  upbp_store_path_state(resources.path_state_buffer, upbp_path_state_index(resources, from_camera, path_index), path_state);
  return true;
}

bool upbp_mark_failed_path(GPUUPBPResources resources, bool from_camera, uint path_index, uint failure_code) {
  if (resources.counter_buffer == kInvalidIndex) {
    return false;
  }
  RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(resources.counter_buffer);
  uint ignored = 0u;
  const uint counter = from_camera ? GPUUPBPCounterIndex::FailedCameraPaths : GPUUPBPCounterIndex::FailedLightPaths;
  counters.InterlockedAdd(counter * 4u, 1u, ignored);
  uint previous_failure = 0u;
  counters.InterlockedCompareExchange(GPUUPBPCounterIndex::FirstFailureCode * 4u, GPUUPBPPathFailure::None, failure_code, previous_failure);
  if (previous_failure == GPUUPBPPathFailure::None) {
    const uint global_path_index = (from_camera ? resources.iteration.camera_batch_offset : resources.iteration.light_batch_offset) + path_index;
    counters.Store(GPUUPBPCounterIndex::FirstFailureGlobalPath * 4u, global_path_index);
    return true;
  }
  return false;
}

void upbp_mark_failed_connection(GPUUPBPResources resources, uint global_path_index, uint technique_code, uint camera_vertex_count, uint light_vertex_count,
  uint tracking_failure) {
  if (resources.counter_buffer == kInvalidIndex) {
    return;
  }
  RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(resources.counter_buffer);
  uint ignored = 0u;
  counters.InterlockedAdd(GPUUPBPCounterIndex::FailedConnections * 4u, 1u, ignored);
  uint previous_failure = 0u;
  counters.InterlockedCompareExchange(GPUUPBPCounterIndex::FirstFailureCode * 4u, GPUUPBPPathFailure::None, GPUUPBPPathFailure::ConnectionTracking, previous_failure);
  if (previous_failure == GPUUPBPPathFailure::None) {
    counters.Store(GPUUPBPCounterIndex::FirstFailureGlobalPath * 4u, global_path_index);
    counters.Store(GPUUPBPCounterIndex::FirstFailureDetail0 * 4u, technique_code);
    counters.Store(GPUUPBPCounterIndex::FirstFailureDetail1 * 4u, camera_vertex_count);
    counters.Store(GPUUPBPCounterIndex::FirstFailureDetail2 * 4u, light_vertex_count);
    counters.Store(GPUUPBPCounterIndex::FirstFailureDetail3 * 4u, tracking_failure);
  }
}

bool upbp_initialize_camera_path(GPUWavefrontResources wavefront_resources, uint path_index, uint global_path_index, Camera camera, GPUWavefrontPathState wavefront_state) {
  GPUUPBPResources resources = upbp_load_resources(wavefront_resources);
  if ((resources.path_state_buffer == kInvalidIndex) || (path_index >= resources.iteration.camera_batch_count)) {
    return false;
  }

  CameraFilmEvalShared camera_eval = camera_film_shared_evaluate_out(camera, wavefront_state.ray);
  GPUUPBPVertex endpoint = (GPUUPBPVertex)0;
  endpoint.throughput = upbp_pack_spectral_response(spectral_response_make(wavefront_state.spect, 1.0f));
  endpoint.outgoing_throughput = endpoint.throughput;
  endpoint.position = wavefront_state.ray.o;
  endpoint.sampled_direction = wavefront_state.ray.d;
  endpoint.medium_index = camera.medium_index;
  endpoint.w_i = wavefront_state.ray.d;
  endpoint.incident_medium_index = camera.medium_index;
  endpoint.outgoing_medium_index = camera.medium_index;
  endpoint.normal = camera_eval.normal;
  endpoint.geo_normal = camera_eval.normal;
  endpoint.material_index = kInvalidIndex;
  endpoint.triangle_index = kInvalidIndex;
  endpoint.instance_index = kInvalidIndex;
  endpoint.scatter_pdf_forward = camera_eval.pdf_dir;
  endpoint.scatter_pdf_reverse = camera_eval.pdf_dir;
  endpoint.endpoint_pdf_area = (camera.lens_radius > kEpsilon) ? rcp(kPi * camera.lens_radius * camera.lens_radius) : 1.0f;
  endpoint.endpoint_pdf_sample = 1.0f;
  endpoint.endpoint_pdf_direction = camera_eval.pdf_dir;
  endpoint.eta = 1.0f;
  endpoint.flags = GPUUPBPVertexFlags::Valid | GPUUPBPVertexFlags::Camera | GPUUPBPVertexFlags::Connectible | GPUUPBPVertexFlags::DensityConnectible;
  endpoint.previous_vertex_index = kInvalidIndex;
  endpoint.incoming_segment_index = kInvalidIndex;
  endpoint.path_length = 0u;
  endpoint.global_path_index = global_path_index;
  endpoint.emitter_index = kInvalidIndex;
  if (upbp_initialize_recursive_state(endpoint, resources.iteration, endpoint.departure_state) == false) {
    upbp_mark_failed_path(resources, true, path_index, GPUUPBPPathFailure::InitializeCamera);
    return false;
  }
  endpoint.arrival_weights = endpoint.departure_state.weights;

  uint vertex_index = kInvalidIndex;
  if (upbp_append_vertex(resources, true, endpoint, vertex_index) == false) {
    return false;
  }
  GPUUPBPPathState state = (GPUUPBPPathState)0;
  state.recursive_state = endpoint.departure_state;
  state.first_vertex_index = vertex_index;
  state.last_vertex_index = vertex_index;
  state.current_segment_index = kInvalidIndex;
  state.current_interval_index = kInvalidIndex;
  state.transport_counts = 0u;
  state.global_path_index = global_path_index;
  state.path_length = 0u;
  state.flags = GPUUPBPPathStateFlags::Valid;
  upbp_store_path_state(resources.path_state_buffer, upbp_path_state_index(resources, true, path_index), state);
  return true;
}

bool upbp_initialize_light_path(GPUWavefrontResources wavefront_resources, uint path_index, uint global_path_index, WavefrontEmitterSample emitter_sample,
  GPUWavefrontPathState wavefront_state) {
  GPUUPBPResources resources = upbp_load_resources(wavefront_resources);
  if ((resources.path_state_buffer == kInvalidIndex) || (path_index >= resources.iteration.light_batch_count)) {
    return false;
  }

  GPUUPBPVertex endpoint = (GPUUPBPVertex)0;
  endpoint.throughput = upbp_pack_spectral_response(emitter_sample.value);
  endpoint.outgoing_throughput = upbp_pack_spectral_response(wavefront_state.throughput);
  endpoint.position = emitter_sample.origin;
  endpoint.sampled_direction = emitter_sample.direction;
  endpoint.medium_index = emitter_sample.medium_index;
  endpoint.w_i = emitter_sample.direction;
  endpoint.incident_medium_index = emitter_sample.medium_index;
  endpoint.outgoing_medium_index = emitter_sample.medium_index;
  endpoint.normal = emitter_sample.normal;
  endpoint.geo_normal = emitter_sample.normal;
  endpoint.material_index = kInvalidIndex;
  endpoint.texcoord = emitter_sample.image_uv;
  endpoint.triangle_index = emitter_sample.triangle_index;
  endpoint.instance_index = emitter_sample.instance_index;
  endpoint.scatter_pdf_forward = emitter_sample.pdf_dir;
  endpoint.scatter_pdf_reverse = emitter_sample.pdf_dir_out;
  endpoint.endpoint_pdf_area = emitter_sample.pdf_area;
  endpoint.endpoint_pdf_sample = emitter_sample.pdf_sample;
  endpoint.endpoint_pdf_direction = emitter_sample.pdf_dir;
  endpoint.eta = 1.0f;
  endpoint.flags = GPUUPBPVertexFlags::Valid | GPUUPBPVertexFlags::Emitter | GPUUPBPVertexFlags::Connectible | GPUUPBPVertexFlags::DensityConnectible;
  endpoint.flags |= emitter_sample.is_delta != 0u ? GPUUPBPVertexFlags::Delta : 0u;
  endpoint.flags |= emitter_sample.is_distant != 0u ? GPUUPBPVertexFlags::DistantEndpoint : 0u;
  endpoint.flags |= emitter_sample.triangle_index != kInvalidIndex ? GPUUPBPVertexFlags::Surface : 0u;
  endpoint.previous_vertex_index = kInvalidIndex;
  endpoint.incoming_segment_index = kInvalidIndex;
  endpoint.path_length = 0u;
  endpoint.global_path_index = global_path_index;
  endpoint.barycentric = emitter_sample.barycentric;
  endpoint.emitter_index = emitter_sample.emitter_index;
  if (upbp_initialize_recursive_state(endpoint, resources.iteration, endpoint.departure_state) == false) {
    upbp_mark_failed_path(resources, false, path_index, GPUUPBPPathFailure::InitializeLight);
    return false;
  }
  endpoint.arrival_weights = endpoint.departure_state.weights;

  uint vertex_index = kInvalidIndex;
  if (upbp_append_vertex(resources, false, endpoint, vertex_index) == false) {
    return false;
  }
  GPUUPBPPathState state = (GPUUPBPPathState)0;
  state.recursive_state = endpoint.departure_state;
  state.first_vertex_index = vertex_index;
  state.last_vertex_index = vertex_index;
  state.current_segment_index = kInvalidIndex;
  state.current_interval_index = kInvalidIndex;
  state.transport_counts = 0u;
  state.global_path_index = global_path_index;
  state.path_length = 0u;
  state.flags = GPUUPBPPathStateFlags::Valid | GPUUPBPPathStateFlags::Light;
  upbp_store_path_state(resources.path_state_buffer, upbp_path_state_index(resources, false, path_index), state);
  return true;
}
