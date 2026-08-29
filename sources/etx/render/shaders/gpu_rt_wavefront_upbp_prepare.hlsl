#include "gpu_rt_wavefront_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_upbp_clear_main(uint3 dtid : SV_DispatchThreadID) {
  GPUUPBPResources resources = upbp_load_resources(wavefront_load_resources());
  if ((resources.counter_buffer == kInvalidIndex) || (dtid.x >= GPUUPBPCounterIndex::Count)) {
    return;
  }
  if (constants.dispatch_item_offset == GPUUPBPClearMode::LightBatch) {
    const bool light_counter = (dtid.x == GPUUPBPCounterIndex::LightVertex) || (dtid.x == GPUUPBPCounterIndex::LightSegment) || (dtid.x == GPUUPBPCounterIndex::LightInterval) ||
                               (dtid.x == GPUUPBPCounterIndex::LightEvent) || (dtid.x == GPUUPBPCounterIndex::Point) || (dtid.x == GPUUPBPCounterIndex::SurfacePoint) ||
                               (dtid.x == GPUUPBPCounterIndex::MediumPoint) || (dtid.x == GPUUPBPCounterIndex::Beam) || (dtid.x == GPUUPBPCounterIndex::DensityBeamInstance) ||
                               (dtid.x == GPUUPBPCounterIndex::OverflowFlags) || (dtid.x == GPUUPBPCounterIndex::FailedLightPaths) ||
                               (dtid.x == GPUUPBPCounterIndex::MaximumLightPathLength) || (dtid.x == GPUUPBPCounterIndex::BeamAccelerationPrimitive) ||
                               (dtid.x == GPUUPBPCounterIndex::DensitySurfacePoint) || (dtid.x == GPUUPBPCounterIndex::DensityMediumPoint) ||
                               (dtid.x == GPUUPBPCounterIndex::DensityBeam) || (dtid.x == GPUUPBPCounterIndex::DensitySelectedBeam);
    if (light_counter == false) {
      return;
    }
  } else if (constants.dispatch_item_offset == GPUUPBPClearMode::BeamInstances) {
    if (dtid.x != GPUUPBPCounterIndex::DensityBeamInstance) {
      return;
    }
  } else if (constants.dispatch_item_offset == GPUUPBPClearMode::BeamAcceleration) {
    if (dtid.x != GPUUPBPCounterIndex::BeamAccelerationPrimitive) {
      return;
    }
  } else if (constants.dispatch_item_offset == GPUUPBPClearMode::DensityCompact) {
    if ((dtid.x != GPUUPBPCounterIndex::DensitySurfacePoint) && (dtid.x != GPUUPBPCounterIndex::DensityMediumPoint) && (dtid.x != GPUUPBPCounterIndex::DensityBeam) &&
        (dtid.x != GPUUPBPCounterIndex::DensitySelectedBeam)) {
      return;
    }
  } else if (constants.dispatch_item_offset == GPUUPBPClearMode::CameraQueries) {
    const bool surface_query_counter = (dtid.x >= GPUUPBPCounterIndex::CameraSurfaceVariousQuery) && (dtid.x <= GPUUPBPCounterIndex::CameraSurfaceDielectricQuery);
    if ((surface_query_counter == false) && (dtid.x != GPUUPBPCounterIndex::CameraMediumVertexQuery) && (dtid.x != GPUUPBPCounterIndex::CameraMediumIntervalQuery)) {
      return;
    }
  }
  WAVEFRONT_RW_BUFFER(resources.counter_buffer).Store(dtid.x * sizeof(uint), 0u);
}

  [numthreads(1, 1, 1)] void wavefront_upbp_validate_main(uint3 dtid : SV_DispatchThreadID) {
  (void)dtid;
  GPUUPBPResources resources = upbp_load_resources(wavefront_load_resources());
  if (resources.counter_buffer == kInvalidIndex) {
    return;
  }
  RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(resources.counter_buffer);
  uint overflow_flags = counters.Load(GPUUPBPCounterIndex::OverflowFlags * sizeof(uint));
  overflow_flags |= counters.Load(GPUUPBPCounterIndex::LightVertex * sizeof(uint)) > resources.light_vertex_capacity ? GPUUPBPOverflowFlags::Vertex : 0u;
  overflow_flags |= counters.Load(GPUUPBPCounterIndex::CameraVertex * sizeof(uint)) > resources.camera_vertex_capacity ? GPUUPBPOverflowFlags::Vertex : 0u;
  overflow_flags |= counters.Load(GPUUPBPCounterIndex::LightSegment * sizeof(uint)) > resources.light_segment_capacity ? GPUUPBPOverflowFlags::Segment : 0u;
  overflow_flags |= counters.Load(GPUUPBPCounterIndex::CameraSegment * sizeof(uint)) > resources.camera_segment_capacity ? GPUUPBPOverflowFlags::Segment : 0u;
  overflow_flags |= counters.Load(GPUUPBPCounterIndex::LightInterval * sizeof(uint)) > resources.light_interval_capacity ? GPUUPBPOverflowFlags::Interval : 0u;
  overflow_flags |= counters.Load(GPUUPBPCounterIndex::CameraInterval * sizeof(uint)) > resources.camera_interval_capacity ? GPUUPBPOverflowFlags::Interval : 0u;
  overflow_flags |= counters.Load(GPUUPBPCounterIndex::LightEvent * sizeof(uint)) > resources.light_event_capacity ? GPUUPBPOverflowFlags::Event : 0u;
  overflow_flags |= counters.Load(GPUUPBPCounterIndex::CameraEvent * sizeof(uint)) > resources.camera_event_capacity ? GPUUPBPOverflowFlags::Event : 0u;
  overflow_flags |= counters.Load(GPUUPBPCounterIndex::Point * sizeof(uint)) > resources.point_capacity ? GPUUPBPOverflowFlags::Point : 0u;
  overflow_flags |= counters.Load(GPUUPBPCounterIndex::Beam * sizeof(uint)) > resources.beam_capacity ? GPUUPBPOverflowFlags::Beam : 0u;
  if (resources.density_output_bb1d_beam_instance_buffer != kInvalidIndex) {
    overflow_flags |=
      counters.Load(GPUUPBPCounterIndex::DensityBeamInstance * sizeof(uint)) > resources.density_output_bb1d_beam_instance_capacity ? GPUUPBPOverflowFlags::BeamInstance : 0u;
  }
  counters.Store(GPUUPBPCounterIndex::OverflowFlags * sizeof(uint), overflow_flags);
}
