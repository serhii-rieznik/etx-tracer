#pragma once

static const uint kGPUUPBPInlineMediumBit = 0x80000000u;

uint upbp_inline_medium_key(uint material_index) {
  return kGPUUPBPInlineMediumBit | material_index;
}

bool upbp_medium_key_is_inline(uint medium_key) {
  return (medium_key != kInvalidIndex) && ((medium_key & kGPUUPBPInlineMediumBit) != 0u);
}

#include <interop/gpu_upbp_abi.hxx>

GPUWavefrontCompactSpectralResponse upbp_pack_spectral_response(SpectralResponse value) {
  GPUWavefrontCompactSpectralResponse result = (GPUWavefrontCompactSpectralResponse)0;
  if (spectral_response_is_spectral(value)) {
    result.payload = float4(value.value, value.wavelength, 0.0f, asfloat(value.flags));
  } else {
    result.payload = float4(value.integrated, asfloat(value.flags));
  }
  return result;
}

SpectralResponse upbp_unpack_spectral_response(GPUWavefrontCompactSpectralResponse value) {
  SpectralResponse result = (SpectralResponse)0;
  result.flags = asuint(value.payload.w);
  if ((result.flags & SpectralFlags::Spectral) != 0u) {
    result.value = value.payload.x;
    result.wavelength = value.payload.y;
  } else {
    result.integrated = value.payload.xyz;
    result.wavelength = kUndefinedWavelength;
  }
  return result;
}

GPUUPBPRecursiveWeights upbp_load_recursive_weights(ByteAddressBuffer buffer, uint byte_offset) {
  GPUUPBPRecursiveWeights result = (GPUUPBPRecursiveWeights)0;
  result.log_d_shared = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveWeightsDSharedOffset));
  result.log_d_bpt = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveWeightsDBPTOffset));
  result.log_d_pde = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveWeightsDPDEOffset));
  result.log_ray_sample_forward_pdf_inverse = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveWeightsRaySampleForwardPdfInverseOffset));
  result.log_ray_sample_reverse_pdf_inverse = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveWeightsRaySampleReversePdfInverseOffset));
  result.log_ray_sample_forward_ratio = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveWeightsRaySampleForwardRatioOffset));
  result.log_ray_sample_reverse_ratio = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveWeightsRaySampleReverseRatioOffset));
  result.flags = buffer.Load(byte_offset + kGPUUPBPRecursiveWeightsFlagsOffset);
  return result;
}

void upbp_store_recursive_weights(RWByteAddressBuffer buffer, uint byte_offset, GPUUPBPRecursiveWeights value) {
  buffer.Store(byte_offset + kGPUUPBPRecursiveWeightsDSharedOffset, asuint(value.log_d_shared));
  buffer.Store(byte_offset + kGPUUPBPRecursiveWeightsDBPTOffset, asuint(value.log_d_bpt));
  buffer.Store(byte_offset + kGPUUPBPRecursiveWeightsDPDEOffset, asuint(value.log_d_pde));
  buffer.Store(byte_offset + kGPUUPBPRecursiveWeightsRaySampleForwardPdfInverseOffset, asuint(value.log_ray_sample_forward_pdf_inverse));
  buffer.Store(byte_offset + kGPUUPBPRecursiveWeightsRaySampleReversePdfInverseOffset, asuint(value.log_ray_sample_reverse_pdf_inverse));
  buffer.Store(byte_offset + kGPUUPBPRecursiveWeightsRaySampleForwardRatioOffset, asuint(value.log_ray_sample_forward_ratio));
  buffer.Store(byte_offset + kGPUUPBPRecursiveWeightsRaySampleReverseRatioOffset, asuint(value.log_ray_sample_reverse_ratio));
  buffer.Store(byte_offset + kGPUUPBPRecursiveWeightsFlagsOffset, value.flags);
}

GPUUPBPRecursiveState upbp_load_recursive_state(ByteAddressBuffer buffer, uint byte_offset) {
  GPUUPBPRecursiveState result = (GPUUPBPRecursiveState)0;
  result.weights = upbp_load_recursive_weights(buffer, byte_offset + kGPUUPBPRecursiveStateWeightsOffset);
  result.last_sin_theta = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveStateLastSinThetaOffset));
  result.log_d_bpt_a = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveStateDBPTAOffset));
  result.log_d_bpt_b = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveStateDBPTBOffset));
  result.log_d_pde_a = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveStateDPDEAOffset));
  result.log_d_pde_b = asfloat(buffer.Load(byte_offset + kGPUUPBPRecursiveStateDPDEBOffset));
  result.failure = buffer.Load(byte_offset + kGPUUPBPRecursiveStateFailureOffset);
  result.failure_vertex_index = buffer.Load(byte_offset + kGPUUPBPRecursiveStateFailureVertexIndexOffset);
  return result;
}

void upbp_store_recursive_state(RWByteAddressBuffer buffer, uint byte_offset, GPUUPBPRecursiveState value) {
  upbp_store_recursive_weights(buffer, byte_offset + kGPUUPBPRecursiveStateWeightsOffset, value.weights);
  buffer.Store(byte_offset + kGPUUPBPRecursiveStateLastSinThetaOffset, asuint(value.last_sin_theta));
  buffer.Store(byte_offset + kGPUUPBPRecursiveStateDBPTAOffset, asuint(value.log_d_bpt_a));
  buffer.Store(byte_offset + kGPUUPBPRecursiveStateDBPTBOffset, asuint(value.log_d_bpt_b));
  buffer.Store(byte_offset + kGPUUPBPRecursiveStateDPDEAOffset, asuint(value.log_d_pde_a));
  buffer.Store(byte_offset + kGPUUPBPRecursiveStateDPDEBOffset, asuint(value.log_d_pde_b));
  buffer.Store(byte_offset + kGPUUPBPRecursiveStateFailureOffset, value.failure);
  buffer.Store(byte_offset + kGPUUPBPRecursiveStateFailureVertexIndexOffset, value.failure_vertex_index);
}

GPUUPBPIteration upbp_load_iteration(ByteAddressBuffer buffer, uint byte_offset) {
  GPUUPBPIteration result = (GPUUPBPIteration)0;
  result.technique_mask = buffer.Load(byte_offset + kGPUUPBPIterationTechniqueMaskOffset);
  result.kernel = buffer.Load(byte_offset + kGPUUPBPIterationKernelOffset);
  result.flags = buffer.Load(byte_offset + kGPUUPBPIterationFlagsOffset);
  result.sample_index = buffer.Load(byte_offset + kGPUUPBPIterationSampleIndexOffset);
  result.global_camera_path_count = buffer.Load(byte_offset + kGPUUPBPIterationGlobalCameraPathCountOffset);
  result.global_light_path_count = buffer.Load(byte_offset + kGPUUPBPIterationGlobalLightPathCountOffset);
  result.bb1d_light_path_count = buffer.Load(byte_offset + kGPUUPBPIterationBB1DLightPathCountOffset);
  result.light_batch_offset = buffer.Load(byte_offset + kGPUUPBPIterationLightBatchOffsetOffset);
  result.light_batch_count = buffer.Load(byte_offset + kGPUUPBPIterationLightBatchCountOffset);
  result.camera_batch_offset = buffer.Load(byte_offset + kGPUUPBPIterationCameraBatchOffsetOffset);
  result.camera_batch_count = buffer.Load(byte_offset + kGPUUPBPIterationCameraBatchCountOffset);
  result.maximum_null_events_per_interval = buffer.Load(byte_offset + kGPUUPBPIterationMaximumNullEventsPerIntervalOffset);
  result.surface_radius = asfloat(buffer.Load(byte_offset + kGPUUPBPIterationSurfaceRadiusOffset));
  result.pp3d_radius = asfloat(buffer.Load(byte_offset + kGPUUPBPIterationPP3DRadiusOffset));
  result.pb2d_radius = asfloat(buffer.Load(byte_offset + kGPUUPBPIterationPB2DRadiusOffset));
  result.bp2d_radius = asfloat(buffer.Load(byte_offset + kGPUUPBPIterationBP2DRadiusOffset));
  result.bb1d_radius = asfloat(buffer.Load(byte_offset + kGPUUPBPIterationBB1DRadiusOffset));
  result.beam_selection_probability = asfloat(buffer.Load(byte_offset + kGPUUPBPIterationBeamSelectionProbabilityOffset));
  result.bpt_sample_count = asfloat(buffer.Load(byte_offset + kGPUUPBPIterationBPTSampleCountOffset));
  result.maximum_boundary_count = buffer.Load(byte_offset + kGPUUPBPIterationMaximumBoundaryCountOffset);
  [unroll] for (uint index = 0u; index < 6u; ++index) {
    result.technique_factors[index] = asfloat(buffer.Load(byte_offset + kGPUUPBPIterationTechniqueFactorsOffset + index * 4u));
  }
  return result;
}

GPUUPBPResources upbp_load_resources(GPUWavefrontResources wavefront_resources) {
  GPUUPBPResources result = (GPUUPBPResources)0;
  if (wavefront_resources.upbp_resources_buffer == kInvalidIndex) {
    return result;
  }
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(wavefront_resources.upbp_resources_buffer);
  result.iteration = upbp_load_iteration(buffer, kGPUUPBPResourcesIterationOffset);
  result.vertex_buffer = buffer.Load(kGPUUPBPResourcesVertexBufferOffset);
  result.segment_buffer = buffer.Load(kGPUUPBPResourcesSegmentBufferOffset);
  result.interval_buffer = buffer.Load(kGPUUPBPResourcesIntervalBufferOffset);
  result.event_buffer = buffer.Load(kGPUUPBPResourcesEventBufferOffset);
  result.point_buffer = buffer.Load(kGPUUPBPResourcesPointBufferOffset);
  result.beam_buffer = buffer.Load(kGPUUPBPResourcesBeamBufferOffset);
  result.density_output_beam_instance_buffer = buffer.Load(kGPUUPBPResourcesDensityOutputBeamInstanceBufferOffset);
  result.density_output_beam_reference_buffer = buffer.Load(kGPUUPBPResourcesDensityOutputBeamReferenceBufferOffset);
  result.density_output_beam_instance_capacity = buffer.Load(kGPUUPBPResourcesDensityOutputBeamInstanceCapacityOffset);
  result.density_beam_acceleration_structure_reference_low = buffer.Load(kGPUUPBPResourcesDensityBeamAccelerationStructureReferenceLowOffset);
  result.counter_buffer = buffer.Load(kGPUUPBPResourcesCounterBufferOffset);
  result.path_state_buffer = buffer.Load(kGPUUPBPResourcesPathStateBufferOffset);
  result.light_vertex_capacity = buffer.Load(kGPUUPBPResourcesLightVertexCapacityOffset);
  result.camera_vertex_capacity = buffer.Load(kGPUUPBPResourcesCameraVertexCapacityOffset);
  result.light_segment_capacity = buffer.Load(kGPUUPBPResourcesLightSegmentCapacityOffset);
  result.camera_segment_capacity = buffer.Load(kGPUUPBPResourcesCameraSegmentCapacityOffset);
  result.light_interval_capacity = buffer.Load(kGPUUPBPResourcesLightIntervalCapacityOffset);
  result.camera_interval_capacity = buffer.Load(kGPUUPBPResourcesCameraIntervalCapacityOffset);
  result.light_event_capacity = buffer.Load(kGPUUPBPResourcesLightEventCapacityOffset);
  result.camera_event_capacity = buffer.Load(kGPUUPBPResourcesCameraEventCapacityOffset);
  result.point_capacity = buffer.Load(kGPUUPBPResourcesPointCapacityOffset);
  result.beam_capacity = buffer.Load(kGPUUPBPResourcesBeamCapacityOffset);
  result.density_beam_acceleration_structure_reference_high = buffer.Load(kGPUUPBPResourcesDensityBeamAccelerationStructureReferenceHighOffset);
  result.light_path_state_capacity = buffer.Load(kGPUUPBPResourcesLightPathStateCapacityOffset);
  result.camera_path_state_capacity = buffer.Load(kGPUUPBPResourcesCameraPathStateCapacityOffset);
  result.bp2d_grid_buffer = buffer.Load(kGPUUPBPResourcesBP2DGridBufferOffset);
  result.bb1d_beam_buffer = buffer.Load(kGPUUPBPResourcesBB1DBeamBufferOffset);
  result.beam_acceleration_structure = buffer.Load(kGPUUPBPResourcesBeamAccelerationStructureOffset);
  result.density_output_bb1d_beam_instance_buffer = buffer.Load(kGPUUPBPResourcesDensityOutputBB1DBeamInstanceBufferOffset);
  result.density_output_bb1d_beam_instance_capacity = buffer.Load(kGPUUPBPResourcesDensityOutputBB1DBeamInstanceCapacityOffset);
  result.point_acceleration_structure = buffer.Load(kGPUUPBPResourcesPointAccelerationStructureOffset);
  result.point_aabb_buffer = buffer.Load(kGPUUPBPResourcesPointAABBBufferOffset);
  result.point_aabb_capacity = buffer.Load(kGPUUPBPResourcesPointAABBCapacityOffset);
  result.density_batch_buffer = buffer.Load(kGPUUPBPResourcesDensityBatchBufferOffset);
  result.density_batch_count = buffer.Load(kGPUUPBPResourcesDensityBatchCountOffset);
  result.density_output_surface_point_buffer = buffer.Load(kGPUUPBPResourcesDensityOutputSurfacePointBufferOffset);
  result.density_output_surface_point_capacity = buffer.Load(kGPUUPBPResourcesDensityOutputSurfacePointCapacityOffset);
  result.density_output_beam_buffer = buffer.Load(kGPUUPBPResourcesDensityOutputBeamBufferOffset);
  result.density_output_beam_capacity = buffer.Load(kGPUUPBPResourcesDensityOutputBeamCapacityOffset);
  result.density_output_event_buffer = buffer.Load(kGPUUPBPResourcesDensityOutputEventBufferOffset);
  result.density_output_medium_point_buffer = buffer.Load(kGPUUPBPResourcesDensityOutputMediumPointBufferOffset);
  result.density_output_medium_point_capacity = buffer.Load(kGPUUPBPResourcesDensityOutputMediumPointCapacityOffset);
  result.medium_point_acceleration_structure = buffer.Load(kGPUUPBPResourcesMediumPointAccelerationStructureOffset);
  result.density_output_medium_point_aabb_buffer = buffer.Load(kGPUUPBPResourcesDensityOutputMediumPointAABBBufferOffset);
  result.density_output_medium_point_aabb_capacity = buffer.Load(kGPUUPBPResourcesDensityOutputMediumPointAABBCapacityOffset);
  result.beam_reference_buffer = buffer.Load(kGPUUPBPResourcesBeamReferenceBufferOffset);
  result.bpt_light_vertex_buffer = buffer.Load(kGPUUPBPResourcesBPTLightVertexBufferOffset);
  result.bpt_light_path_state_buffer = buffer.Load(kGPUUPBPResourcesBPTLightPathStateBufferOffset);
  [unroll] for (uint partition_index = 1u; partition_index < kGPUUPBPBB1DPartitionCount; ++partition_index) {
    result.bb1d_partition_acceleration_structures[partition_index - 1u] =
      buffer.Load(kGPUUPBPResourcesBB1DPartitionAccelerationStructuresOffset + (partition_index - 1u) * sizeof(uint));
  }
  return result;
}

GPUUPBPBeamGrid upbp_load_bp2d_grid(GPUUPBPResources resources) {
  GPUUPBPBeamGrid result = (GPUUPBPBeamGrid)0;
  if (resources.bp2d_grid_buffer == kInvalidIndex) {
    return result;
  }
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(resources.bp2d_grid_buffer);
  result.minimum_inverse_cell_size = asfloat(buffer.Load4(kGPUUPBPBeamGridMinimumInverseCellSizeOffset));
  result.maximum_cell_size = asfloat(buffer.Load4(kGPUUPBPBeamGridMaximumCellSizeOffset));
  result.resolution_cell_count = buffer.Load4(kGPUUPBPBeamGridResolutionCellCountOffset);
  result.buffers_entry_capacity = buffer.Load4(kGPUUPBPBeamGridBuffersEntryCapacityOffset);
  return result;
}

GPUUPBPVertex upbp_load_vertex(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPVertexStride;
  GPUUPBPVertex result = (GPUUPBPVertex)0;
  result.throughput.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPVertexThroughputOffset);
  result.outgoing_throughput.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPVertexOutgoingThroughputOffset);
  result.position = wavefront_load_float3(buffer, base_offset + kGPUUPBPVertexPositionOffset);
  result.flags = buffer.Load(base_offset + kGPUUPBPVertexFlagsOffset);
  result.sampled_direction = wavefront_load_float3(buffer, base_offset + kGPUUPBPVertexSampledDirectionOffset);
  result.medium_index = buffer.Load(base_offset + kGPUUPBPVertexMediumIndexOffset);
  result.w_i = wavefront_load_float3(buffer, base_offset + kGPUUPBPVertexWiOffset);
  result.incident_medium_index = buffer.Load(base_offset + kGPUUPBPVertexIncidentMediumIndexOffset);
  result.normal = wavefront_load_float3(buffer, base_offset + kGPUUPBPVertexNormalOffset);
  result.outgoing_medium_index = buffer.Load(base_offset + kGPUUPBPVertexOutgoingMediumIndexOffset);
  result.geo_normal = wavefront_load_float3(buffer, base_offset + kGPUUPBPVertexGeoNormalOffset);
  result.material_index = buffer.Load(base_offset + kGPUUPBPVertexMaterialIndexOffset);
  result.texcoord = wavefront_load_float2(buffer, base_offset + kGPUUPBPVertexTexcoordOffset);
  result.triangle_index = buffer.Load(base_offset + kGPUUPBPVertexTriangleIndexOffset);
  result.instance_index = buffer.Load(base_offset + kGPUUPBPVertexInstanceIndexOffset);
  result.scatter_pdf_forward = asfloat(buffer.Load(base_offset + kGPUUPBPVertexScatterPdfForwardOffset));
  result.scatter_pdf_reverse = asfloat(buffer.Load(base_offset + kGPUUPBPVertexScatterPdfReverseOffset));
  result.endpoint_pdf_area = asfloat(buffer.Load(base_offset + kGPUUPBPVertexEndpointPdfAreaOffset));
  result.endpoint_pdf_sample = asfloat(buffer.Load(base_offset + kGPUUPBPVertexEndpointPdfSampleOffset));
  result.endpoint_pdf_direction = asfloat(buffer.Load(base_offset + kGPUUPBPVertexEndpointPdfDirectionOffset));
  result.log_medium_event_density = asfloat(buffer.Load(base_offset + kGPUUPBPVertexLogMediumEventDensityOffset));
  result.eta = asfloat(buffer.Load(base_offset + kGPUUPBPVertexEtaOffset));
  result.sample_properties = buffer.Load(base_offset + kGPUUPBPVertexSamplePropertiesOffset);
  result.arrival_weights = upbp_load_recursive_weights(buffer, base_offset + kGPUUPBPVertexArrivalWeightsOffset);
  result.departure_state = upbp_load_recursive_state(buffer, base_offset + kGPUUPBPVertexDepartureStateOffset);
  result.previous_vertex_index = buffer.Load(base_offset + kGPUUPBPVertexPreviousVertexIndexOffset);
  result.incoming_segment_index = buffer.Load(base_offset + kGPUUPBPVertexIncomingSegmentIndexOffset);
  result.path_length = buffer.Load(base_offset + kGPUUPBPVertexPathLengthOffset);
  result.global_path_index = buffer.Load(base_offset + kGPUUPBPVertexGlobalPathIndexOffset);
  result.barycentric = wavefront_load_float3(buffer, base_offset + kGPUUPBPVertexBarycentricOffset);
  result.emitter_index = buffer.Load(base_offset + kGPUUPBPVertexEmitterIndexOffset);
  result.inline_scattering.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPVertexInlineScatteringOffset);
  result.inline_extinction.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPVertexInlineExtinctionOffset);
  result.inline_phase_function_g = asfloat(buffer.Load(base_offset + kGPUUPBPVertexInlinePhaseFunctionGOffset));
  return result;
}

GPUUPBPVertex upbp_load_bpt_light_vertex(GPUUPBPResources resources, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(resources.bpt_light_vertex_buffer);
  const uint base_offset = index * kGPUUPBPBPTVertexStride;
  GPUUPBPVertex result = (GPUUPBPVertex)0;
  result.throughput.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPBPTVertexThroughputOffset);
  result.position = wavefront_load_float3(buffer, base_offset + kGPUUPBPBPTVertexPositionOffset);
  result.flags = buffer.Load(base_offset + kGPUUPBPBPTVertexFlagsOffset);
  result.w_i = wavefront_load_float3(buffer, base_offset + kGPUUPBPBPTVertexWiOffset);
  result.medium_index = buffer.Load(base_offset + kGPUUPBPBPTVertexMediumIndexOffset);
  result.normal = wavefront_load_float3(buffer, base_offset + kGPUUPBPBPTVertexNormalOffset);
  result.material_index = buffer.Load(base_offset + kGPUUPBPBPTVertexMaterialIndexOffset);
  result.geo_normal = wavefront_load_float3(buffer, base_offset + kGPUUPBPBPTVertexGeoNormalOffset);
  result.log_medium_event_density = asfloat(buffer.Load(base_offset + kGPUUPBPBPTVertexLogMediumEventDensityOffset));
  result.texcoord = wavefront_load_float2(buffer, base_offset + kGPUUPBPBPTVertexTexcoordOffset);
  result.triangle_index = buffer.Load(base_offset + kGPUUPBPBPTVertexTriangleIndexOffset);
  result.instance_index = buffer.Load(base_offset + kGPUUPBPBPTVertexInstanceIndexOffset);
  result.arrival_weights = upbp_load_recursive_weights(buffer, base_offset + kGPUUPBPBPTVertexArrivalWeightsOffset);
  result.previous_vertex_index = buffer.Load(base_offset + kGPUUPBPBPTVertexPreviousVertexIndexOffset);
  result.path_length = buffer.Load(base_offset + kGPUUPBPBPTVertexPathLengthOffset);
  result.global_path_index = buffer.Load(base_offset + kGPUUPBPBPTVertexGlobalPathIndexOffset);
  result.emitter_index = buffer.Load(base_offset + kGPUUPBPBPTVertexEmitterIndexOffset);
  const float2 barycentric = wavefront_load_float2(buffer, base_offset + kGPUUPBPBPTVertexBarycentricOffset);
  result.barycentric = float3(0.0f, barycentric);
  result.scatter_pdf_forward = asfloat(buffer.Load(base_offset + kGPUUPBPBPTVertexScatterPdfForwardOffset));
  result.inline_extinction.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPBPTVertexInlineExtinctionOffset);
  return result;
}

void upbp_store_bpt_light_vertex(uint descriptor_index, uint index, GPUUPBPVertex value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPBPTVertexStride;
  wavefront_store_float4(buffer, base_offset + kGPUUPBPBPTVertexThroughputOffset, value.throughput.payload);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPBPTVertexPositionOffset, value.position);
  buffer.Store(base_offset + kGPUUPBPBPTVertexFlagsOffset, value.flags);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPBPTVertexWiOffset, value.w_i);
  buffer.Store(base_offset + kGPUUPBPBPTVertexMediumIndexOffset, value.medium_index);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPBPTVertexNormalOffset, value.normal);
  buffer.Store(base_offset + kGPUUPBPBPTVertexMaterialIndexOffset, value.material_index);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPBPTVertexGeoNormalOffset, value.geo_normal);
  buffer.Store(base_offset + kGPUUPBPBPTVertexLogMediumEventDensityOffset, asuint(value.log_medium_event_density));
  wavefront_store_float2(buffer, base_offset + kGPUUPBPBPTVertexTexcoordOffset, value.texcoord);
  buffer.Store(base_offset + kGPUUPBPBPTVertexTriangleIndexOffset, value.triangle_index);
  buffer.Store(base_offset + kGPUUPBPBPTVertexInstanceIndexOffset, value.instance_index);
  upbp_store_recursive_weights(buffer, base_offset + kGPUUPBPBPTVertexArrivalWeightsOffset, value.arrival_weights);
  buffer.Store(base_offset + kGPUUPBPBPTVertexPreviousVertexIndexOffset, value.previous_vertex_index);
  buffer.Store(base_offset + kGPUUPBPBPTVertexPathLengthOffset, value.path_length);
  buffer.Store(base_offset + kGPUUPBPBPTVertexGlobalPathIndexOffset, value.global_path_index);
  buffer.Store(base_offset + kGPUUPBPBPTVertexEmitterIndexOffset, value.emitter_index);
  wavefront_store_float2(buffer, base_offset + kGPUUPBPBPTVertexBarycentricOffset, value.barycentric.yz);
  buffer.Store(base_offset + kGPUUPBPBPTVertexScatterPdfForwardOffset, asuint(value.scatter_pdf_forward));
  wavefront_store_float4(buffer, base_offset + kGPUUPBPBPTVertexInlineExtinctionOffset, value.inline_extinction.payload);
}

GPUUPBPPathState upbp_load_bpt_light_path_state(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPBPTPathStateStride;
  GPUUPBPPathState result = (GPUUPBPPathState)0;
  result.last_vertex_index = buffer.Load(base_offset + kGPUUPBPBPTPathStateLastVertexIndexOffset);
  result.global_path_index = buffer.Load(base_offset + kGPUUPBPBPTPathStateGlobalPathIndexOffset);
  result.path_length = buffer.Load(base_offset + kGPUUPBPBPTPathStatePathLengthOffset);
  result.flags = buffer.Load(base_offset + kGPUUPBPBPTPathStateFlagsOffset);
  return result;
}

void upbp_store_bpt_light_path_state(uint descriptor_index, uint index, GPUUPBPPathState value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPBPTPathStateStride;
  buffer.Store(base_offset + kGPUUPBPBPTPathStateLastVertexIndexOffset, value.last_vertex_index);
  buffer.Store(base_offset + kGPUUPBPBPTPathStateGlobalPathIndexOffset, value.global_path_index);
  buffer.Store(base_offset + kGPUUPBPBPTPathStatePathLengthOffset, value.path_length);
  buffer.Store(base_offset + kGPUUPBPBPTPathStateFlagsOffset, value.flags);
}

void upbp_store_vertex(uint descriptor_index, uint index, GPUUPBPVertex value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPVertexStride;
  wavefront_store_float4(buffer, base_offset + kGPUUPBPVertexThroughputOffset, value.throughput.payload);
  wavefront_store_float4(buffer, base_offset + kGPUUPBPVertexOutgoingThroughputOffset, value.outgoing_throughput.payload);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPVertexPositionOffset, value.position);
  buffer.Store(base_offset + kGPUUPBPVertexFlagsOffset, value.flags);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPVertexSampledDirectionOffset, value.sampled_direction);
  buffer.Store(base_offset + kGPUUPBPVertexMediumIndexOffset, value.medium_index);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPVertexWiOffset, value.w_i);
  buffer.Store(base_offset + kGPUUPBPVertexIncidentMediumIndexOffset, value.incident_medium_index);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPVertexNormalOffset, value.normal);
  buffer.Store(base_offset + kGPUUPBPVertexOutgoingMediumIndexOffset, value.outgoing_medium_index);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPVertexGeoNormalOffset, value.geo_normal);
  buffer.Store(base_offset + kGPUUPBPVertexMaterialIndexOffset, value.material_index);
  wavefront_store_float2(buffer, base_offset + kGPUUPBPVertexTexcoordOffset, value.texcoord);
  buffer.Store(base_offset + kGPUUPBPVertexTriangleIndexOffset, value.triangle_index);
  buffer.Store(base_offset + kGPUUPBPVertexInstanceIndexOffset, value.instance_index);
  buffer.Store(base_offset + kGPUUPBPVertexScatterPdfForwardOffset, asuint(value.scatter_pdf_forward));
  buffer.Store(base_offset + kGPUUPBPVertexScatterPdfReverseOffset, asuint(value.scatter_pdf_reverse));
  buffer.Store(base_offset + kGPUUPBPVertexEndpointPdfAreaOffset, asuint(value.endpoint_pdf_area));
  buffer.Store(base_offset + kGPUUPBPVertexEndpointPdfSampleOffset, asuint(value.endpoint_pdf_sample));
  buffer.Store(base_offset + kGPUUPBPVertexEndpointPdfDirectionOffset, asuint(value.endpoint_pdf_direction));
  buffer.Store(base_offset + kGPUUPBPVertexLogMediumEventDensityOffset, asuint(value.log_medium_event_density));
  buffer.Store(base_offset + kGPUUPBPVertexEtaOffset, asuint(value.eta));
  buffer.Store(base_offset + kGPUUPBPVertexSamplePropertiesOffset, value.sample_properties);
  upbp_store_recursive_weights(buffer, base_offset + kGPUUPBPVertexArrivalWeightsOffset, value.arrival_weights);
  upbp_store_recursive_state(buffer, base_offset + kGPUUPBPVertexDepartureStateOffset, value.departure_state);
  buffer.Store(base_offset + kGPUUPBPVertexPreviousVertexIndexOffset, value.previous_vertex_index);
  buffer.Store(base_offset + kGPUUPBPVertexIncomingSegmentIndexOffset, value.incoming_segment_index);
  buffer.Store(base_offset + kGPUUPBPVertexPathLengthOffset, value.path_length);
  buffer.Store(base_offset + kGPUUPBPVertexGlobalPathIndexOffset, value.global_path_index);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPVertexBarycentricOffset, value.barycentric);
  buffer.Store(base_offset + kGPUUPBPVertexEmitterIndexOffset, value.emitter_index);
  wavefront_store_float4(buffer, base_offset + kGPUUPBPVertexInlineScatteringOffset, value.inline_scattering.payload);
  wavefront_store_float4(buffer, base_offset + kGPUUPBPVertexInlineExtinctionOffset, value.inline_extinction.payload);
  buffer.Store(base_offset + kGPUUPBPVertexInlinePhaseFunctionGOffset, asuint(value.inline_phase_function_g));
}

GPUUPBPPathState upbp_load_path_state(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPPathStateStride;
  GPUUPBPPathState result = (GPUUPBPPathState)0;
  result.recursive_state = upbp_load_recursive_state(buffer, base_offset + kGPUUPBPPathStateRecursiveStateOffset);
  result.first_vertex_index = buffer.Load(base_offset + kGPUUPBPPathStateFirstVertexIndexOffset);
  result.last_vertex_index = buffer.Load(base_offset + kGPUUPBPPathStateLastVertexIndexOffset);
  result.current_segment_index = buffer.Load(base_offset + kGPUUPBPPathStateCurrentSegmentIndexOffset);
  result.current_interval_index = buffer.Load(base_offset + kGPUUPBPPathStateCurrentIntervalIndexOffset);
  result.transport_counts = buffer.Load(base_offset + kGPUUPBPPathStateTransportCountsOffset);
  result.global_path_index = buffer.Load(base_offset + kGPUUPBPPathStateGlobalPathIndexOffset);
  result.path_length = buffer.Load(base_offset + kGPUUPBPPathStatePathLengthOffset);
  result.flags = buffer.Load(base_offset + kGPUUPBPPathStateFlagsOffset);
  return result;
}

void upbp_store_path_state(uint descriptor_index, uint index, GPUUPBPPathState value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPPathStateStride;
  upbp_store_recursive_state(buffer, base_offset + kGPUUPBPPathStateRecursiveStateOffset, value.recursive_state);
  buffer.Store(base_offset + kGPUUPBPPathStateFirstVertexIndexOffset, value.first_vertex_index);
  buffer.Store(base_offset + kGPUUPBPPathStateLastVertexIndexOffset, value.last_vertex_index);
  buffer.Store(base_offset + kGPUUPBPPathStateCurrentSegmentIndexOffset, value.current_segment_index);
  buffer.Store(base_offset + kGPUUPBPPathStateCurrentIntervalIndexOffset, value.current_interval_index);
  buffer.Store(base_offset + kGPUUPBPPathStateTransportCountsOffset, value.transport_counts);
  buffer.Store(base_offset + kGPUUPBPPathStateGlobalPathIndexOffset, value.global_path_index);
  buffer.Store(base_offset + kGPUUPBPPathStatePathLengthOffset, value.path_length);
  buffer.Store(base_offset + kGPUUPBPPathStateFlagsOffset, value.flags);
}

GPUUPBPSegment upbp_load_segment(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPSegmentStride;
  GPUUPBPSegment result = (GPUUPBPSegment)0;
  result.weight.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPSegmentWeightOffset);
  result.log_pdf_forward = asfloat(buffer.Load(base_offset + kGPUUPBPSegmentLogPdfForwardOffset));
  result.log_pdf_reverse = asfloat(buffer.Load(base_offset + kGPUUPBPSegmentLogPdfReverseOffset));
  result.log_transport_pdf_forward = asfloat(buffer.Load(base_offset + kGPUUPBPSegmentLogTransportPdfForwardOffset));
  result.log_transport_pdf_reverse = asfloat(buffer.Load(base_offset + kGPUUPBPSegmentLogTransportPdfReverseOffset));
  result.log_terminal_event_density = asfloat(buffer.Load(base_offset + kGPUUPBPSegmentLogTerminalEventDensityOffset));
  result.distance = asfloat(buffer.Load(base_offset + kGPUUPBPSegmentDistanceOffset));
  result.first_interval_index = buffer.Load(base_offset + kGPUUPBPSegmentFirstIntervalIndexOffset);
  result.interval_count = buffer.Load(base_offset + kGPUUPBPSegmentIntervalCountOffset);
  result.source_vertex_index = buffer.Load(base_offset + kGPUUPBPSegmentSourceVertexIndexOffset);
  result.target_vertex_index = buffer.Load(base_offset + kGPUUPBPSegmentTargetVertexIndexOffset);
  result.boundary_count = buffer.Load(base_offset + kGPUUPBPSegmentBoundaryCountOffset);
  result.flags = buffer.Load(base_offset + kGPUUPBPSegmentFlagsOffset);
  return result;
}

void upbp_store_segment(uint descriptor_index, uint index, GPUUPBPSegment value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPSegmentStride;
  wavefront_store_float4(buffer, base_offset + kGPUUPBPSegmentWeightOffset, value.weight.payload);
  buffer.Store(base_offset + kGPUUPBPSegmentLogPdfForwardOffset, asuint(value.log_pdf_forward));
  buffer.Store(base_offset + kGPUUPBPSegmentLogPdfReverseOffset, asuint(value.log_pdf_reverse));
  buffer.Store(base_offset + kGPUUPBPSegmentLogTransportPdfForwardOffset, asuint(value.log_transport_pdf_forward));
  buffer.Store(base_offset + kGPUUPBPSegmentLogTransportPdfReverseOffset, asuint(value.log_transport_pdf_reverse));
  buffer.Store(base_offset + kGPUUPBPSegmentLogTerminalEventDensityOffset, asuint(value.log_terminal_event_density));
  buffer.Store(base_offset + kGPUUPBPSegmentDistanceOffset, asuint(value.distance));
  buffer.Store(base_offset + kGPUUPBPSegmentFirstIntervalIndexOffset, value.first_interval_index);
  buffer.Store(base_offset + kGPUUPBPSegmentIntervalCountOffset, value.interval_count);
  buffer.Store(base_offset + kGPUUPBPSegmentSourceVertexIndexOffset, value.source_vertex_index);
  buffer.Store(base_offset + kGPUUPBPSegmentTargetVertexIndexOffset, value.target_vertex_index);
  buffer.Store(base_offset + kGPUUPBPSegmentBoundaryCountOffset, value.boundary_count);
  buffer.Store(base_offset + kGPUUPBPSegmentFlagsOffset, value.flags);
}

GPUUPBPInterval upbp_load_interval_at_offset(ByteAddressBuffer buffer, uint base_offset) {
  GPUUPBPInterval result = (GPUUPBPInterval)0;
  result.weight.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPIntervalWeightOffset);
  result.start_position = wavefront_load_float3(buffer, base_offset + kGPUUPBPIntervalStartPositionOffset);
  result.medium_index = buffer.Load(base_offset + kGPUUPBPIntervalMediumIndexOffset);
  result.end_position = wavefront_load_float3(buffer, base_offset + kGPUUPBPIntervalEndPositionOffset);
  result.flags = buffer.Load(base_offset + kGPUUPBPIntervalFlagsOffset);
  result.log_pdf_forward = asfloat(buffer.Load(base_offset + kGPUUPBPIntervalLogPdfForwardOffset));
  result.log_pdf_reverse = asfloat(buffer.Load(base_offset + kGPUUPBPIntervalLogPdfReverseOffset));
  result.log_transport_pdf_forward = asfloat(buffer.Load(base_offset + kGPUUPBPIntervalLogTransportPdfForwardOffset));
  result.log_transport_pdf_reverse = asfloat(buffer.Load(base_offset + kGPUUPBPIntervalLogTransportPdfReverseOffset));
  result.log_terminal_event_density = asfloat(buffer.Load(base_offset + kGPUUPBPIntervalLogTerminalEventDensityOffset));
  result.distance = asfloat(buffer.Load(base_offset + kGPUUPBPIntervalDistanceOffset));
  result.first_event_index = buffer.Load(base_offset + kGPUUPBPIntervalFirstEventIndexOffset);
  result.event_count = buffer.Load(base_offset + kGPUUPBPIntervalEventCountOffset);
  result.segment_index = buffer.Load(base_offset + kGPUUPBPIntervalSegmentIndexOffset);
  result.next_interval_index = buffer.Load(base_offset + kGPUUPBPIntervalNextIntervalIndexOffset);
  result.tracking_seed = buffer.Load(base_offset + kGPUUPBPIntervalTrackingSeedOffset);
  result.inline_scattering.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPIntervalInlineScatteringOffset);
  result.inline_absorption.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPIntervalInlineAbsorptionOffset);
  return result;
}

void upbp_store_interval_at_offset(RWByteAddressBuffer buffer, uint base_offset, GPUUPBPInterval value) {
  wavefront_store_float4(buffer, base_offset + kGPUUPBPIntervalWeightOffset, value.weight.payload);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPIntervalStartPositionOffset, value.start_position);
  buffer.Store(base_offset + kGPUUPBPIntervalMediumIndexOffset, value.medium_index);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPIntervalEndPositionOffset, value.end_position);
  buffer.Store(base_offset + kGPUUPBPIntervalFlagsOffset, value.flags);
  buffer.Store(base_offset + kGPUUPBPIntervalLogPdfForwardOffset, asuint(value.log_pdf_forward));
  buffer.Store(base_offset + kGPUUPBPIntervalLogPdfReverseOffset, asuint(value.log_pdf_reverse));
  buffer.Store(base_offset + kGPUUPBPIntervalLogTransportPdfForwardOffset, asuint(value.log_transport_pdf_forward));
  buffer.Store(base_offset + kGPUUPBPIntervalLogTransportPdfReverseOffset, asuint(value.log_transport_pdf_reverse));
  buffer.Store(base_offset + kGPUUPBPIntervalLogTerminalEventDensityOffset, asuint(value.log_terminal_event_density));
  buffer.Store(base_offset + kGPUUPBPIntervalDistanceOffset, asuint(value.distance));
  buffer.Store(base_offset + kGPUUPBPIntervalFirstEventIndexOffset, value.first_event_index);
  buffer.Store(base_offset + kGPUUPBPIntervalEventCountOffset, value.event_count);
  buffer.Store(base_offset + kGPUUPBPIntervalSegmentIndexOffset, value.segment_index);
  buffer.Store(base_offset + kGPUUPBPIntervalNextIntervalIndexOffset, value.next_interval_index);
  buffer.Store(base_offset + kGPUUPBPIntervalTrackingSeedOffset, value.tracking_seed);
  wavefront_store_float4(buffer, base_offset + kGPUUPBPIntervalInlineScatteringOffset, value.inline_scattering.payload);
  wavefront_store_float4(buffer, base_offset + kGPUUPBPIntervalInlineAbsorptionOffset, value.inline_absorption.payload);
}

GPUUPBPInterval upbp_load_interval(uint descriptor_index, uint index) {
  return upbp_load_interval_at_offset(WAVEFRONT_RO_BUFFER(descriptor_index), index * kGPUUPBPIntervalStride);
}

void upbp_store_interval(uint descriptor_index, uint index, GPUUPBPInterval value) {
  upbp_store_interval_at_offset(WAVEFRONT_RW_BUFFER(descriptor_index), index * kGPUUPBPIntervalStride, value);
}

GPUUPBPTrackingEvent upbp_load_tracking_event(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPTrackingEventStride;
  GPUUPBPTrackingEvent result = (GPUUPBPTrackingEvent)0;
  result.weight_before.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPTrackingEventWeightBeforeOffset);
  result.log_transport_pdf_forward_before = asfloat(buffer.Load(base_offset + kGPUUPBPTrackingEventLogTransportPdfForwardBeforeOffset));
  result.log_transport_pdf_reverse_before = asfloat(buffer.Load(base_offset + kGPUUPBPTrackingEventLogTransportPdfReverseBeforeOffset));
  result.distance_before = asfloat(buffer.Load(base_offset + kGPUUPBPTrackingEventDistanceBeforeOffset));
  result.end_distance = asfloat(buffer.Load(base_offset + kGPUUPBPTrackingEventEndDistanceOffset));
  result.majorant = asfloat(buffer.Load(base_offset + kGPUUPBPTrackingEventMajorantOffset));
  result.interval_index = buffer.Load(base_offset + kGPUUPBPTrackingEventIntervalIndexOffset);
  result.next_event_index = buffer.Load(base_offset + kGPUUPBPTrackingEventNextEventIndexOffset);
  return result;
}

void upbp_store_tracking_event(uint descriptor_index, uint index, GPUUPBPTrackingEvent value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPTrackingEventStride;
  wavefront_store_float4(buffer, base_offset + kGPUUPBPTrackingEventWeightBeforeOffset, value.weight_before.payload);
  buffer.Store(base_offset + kGPUUPBPTrackingEventLogTransportPdfForwardBeforeOffset, asuint(value.log_transport_pdf_forward_before));
  buffer.Store(base_offset + kGPUUPBPTrackingEventLogTransportPdfReverseBeforeOffset, asuint(value.log_transport_pdf_reverse_before));
  buffer.Store(base_offset + kGPUUPBPTrackingEventDistanceBeforeOffset, asuint(value.distance_before));
  buffer.Store(base_offset + kGPUUPBPTrackingEventEndDistanceOffset, asuint(value.end_distance));
  buffer.Store(base_offset + kGPUUPBPTrackingEventMajorantOffset, asuint(value.majorant));
  buffer.Store(base_offset + kGPUUPBPTrackingEventIntervalIndexOffset, value.interval_index);
  buffer.Store(base_offset + kGPUUPBPTrackingEventNextEventIndexOffset, value.next_event_index);
}

void upbp_store_tracking_event_next_index(uint descriptor_index, uint index, uint next_event_index) {
  WAVEFRONT_RW_BUFFER(descriptor_index).Store(index * kGPUUPBPTrackingEventStride + kGPUUPBPTrackingEventNextEventIndexOffset, next_event_index);
}

GPUUPBPPoint upbp_load_point(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPPointStride;
  GPUUPBPPoint result = (GPUUPBPPoint)0;
  result.position = wavefront_load_float3(buffer, base_offset + kGPUUPBPPointPositionOffset);
  result.vertex_index = buffer.Load(base_offset + kGPUUPBPPointVertexIndexOffset);
  return result;
}

void upbp_store_point(uint descriptor_index, uint index, GPUUPBPPoint value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPPointStride;
  wavefront_store_float3(buffer, base_offset + kGPUUPBPPointPositionOffset, value.position);
  buffer.Store(base_offset + kGPUUPBPPointVertexIndexOffset, value.vertex_index);
}

GPUUPBPBeam upbp_load_beam_at_offset(ByteAddressBuffer buffer, uint base_offset) {
  GPUUPBPBeam result = (GPUUPBPBeam)0;
  result.origin = wavefront_load_float3(buffer, base_offset + kGPUUPBPBeamOriginOffset);
  result.length = asfloat(buffer.Load(base_offset + kGPUUPBPBeamLengthOffset));
  result.direction = wavefront_load_float3(buffer, base_offset + kGPUUPBPBeamDirectionOffset);
  result.flags = buffer.Load(base_offset + kGPUUPBPBeamFlagsOffset);
  result.source_vertex_index = buffer.Load(base_offset + kGPUUPBPBeamSourceVertexIndexOffset);
  result.interval_index = buffer.Load(base_offset + kGPUUPBPBeamIntervalIndexOffset);
  result.global_path_index = buffer.Load(base_offset + kGPUUPBPBeamGlobalPathIndexOffset);
  result.path_length = buffer.Load(base_offset + kGPUUPBPBeamPathLengthOffset);
  return result;
}

void upbp_store_beam_at_offset(RWByteAddressBuffer buffer, uint base_offset, GPUUPBPBeam value) {
  wavefront_store_float3(buffer, base_offset + kGPUUPBPBeamOriginOffset, value.origin);
  buffer.Store(base_offset + kGPUUPBPBeamLengthOffset, asuint(value.length));
  wavefront_store_float3(buffer, base_offset + kGPUUPBPBeamDirectionOffset, value.direction);
  buffer.Store(base_offset + kGPUUPBPBeamFlagsOffset, value.flags);
  buffer.Store(base_offset + kGPUUPBPBeamSourceVertexIndexOffset, value.source_vertex_index);
  buffer.Store(base_offset + kGPUUPBPBeamIntervalIndexOffset, value.interval_index);
  buffer.Store(base_offset + kGPUUPBPBeamGlobalPathIndexOffset, value.global_path_index);
  buffer.Store(base_offset + kGPUUPBPBeamPathLengthOffset, value.path_length);
}

GPUUPBPBeam upbp_load_beam(uint descriptor_index, uint index) {
  return upbp_load_beam_at_offset(WAVEFRONT_RO_BUFFER(descriptor_index), index * kGPUUPBPBeamStride);
}

void upbp_store_beam(uint descriptor_index, uint index, GPUUPBPBeam value) {
  upbp_store_beam_at_offset(WAVEFRONT_RW_BUFFER(descriptor_index), index * kGPUUPBPBeamStride, value);
}

GPUUPBPDensityPoint upbp_load_density_point(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPDensityPointStride;
  GPUUPBPDensityPoint result = (GPUUPBPDensityPoint)0;
  result.throughput.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPDensityPointThroughputOffset);
  result.position = wavefront_load_float3(buffer, base_offset + kGPUUPBPDensityPointPositionOffset);
  result.flags = buffer.Load(base_offset + kGPUUPBPDensityPointFlagsOffset);
  result.w_i = wavefront_load_float3(buffer, base_offset + kGPUUPBPDensityPointWiOffset);
  result.medium_index = buffer.Load(base_offset + kGPUUPBPDensityPointMediumIndexOffset);
  result.geo_normal = wavefront_load_float3(buffer, base_offset + kGPUUPBPDensityPointGeoNormalOffset);
  result.path_length = buffer.Load(base_offset + kGPUUPBPDensityPointPathLengthOffset);
  result.arrival_weights = upbp_load_recursive_weights(buffer, base_offset + kGPUUPBPDensityPointArrivalWeightsOffset);
  result.global_path_index = buffer.Load(base_offset + kGPUUPBPDensityPointGlobalPathIndexOffset);
  result.log_medium_event_density = asfloat(buffer.Load(base_offset + kGPUUPBPDensityPointLogMediumEventDensityOffset));
  result.inline_phase_function_g = asfloat(buffer.Load(base_offset + kGPUUPBPDensityPointInlinePhaseFunctionGOffset));
  result.inline_scattering.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPDensityPointInlineScatteringOffset);
  result.inline_extinction.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPDensityPointInlineExtinctionOffset);
  return result;
}

void upbp_store_density_point(uint descriptor_index, uint index, GPUUPBPDensityPoint value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPDensityPointStride;
  wavefront_store_float4(buffer, base_offset + kGPUUPBPDensityPointThroughputOffset, value.throughput.payload);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPDensityPointPositionOffset, value.position);
  buffer.Store(base_offset + kGPUUPBPDensityPointFlagsOffset, value.flags);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPDensityPointWiOffset, value.w_i);
  buffer.Store(base_offset + kGPUUPBPDensityPointMediumIndexOffset, value.medium_index);
  wavefront_store_float3(buffer, base_offset + kGPUUPBPDensityPointGeoNormalOffset, value.geo_normal);
  buffer.Store(base_offset + kGPUUPBPDensityPointPathLengthOffset, value.path_length);
  upbp_store_recursive_weights(buffer, base_offset + kGPUUPBPDensityPointArrivalWeightsOffset, value.arrival_weights);
  buffer.Store(base_offset + kGPUUPBPDensityPointGlobalPathIndexOffset, value.global_path_index);
  buffer.Store(base_offset + kGPUUPBPDensityPointLogMediumEventDensityOffset, asuint(value.log_medium_event_density));
  buffer.Store(base_offset + kGPUUPBPDensityPointInlinePhaseFunctionGOffset, asuint(value.inline_phase_function_g));
  wavefront_store_float4(buffer, base_offset + kGPUUPBPDensityPointInlineScatteringOffset, value.inline_scattering.payload);
  wavefront_store_float4(buffer, base_offset + kGPUUPBPDensityPointInlineExtinctionOffset, value.inline_extinction.payload);
}

GPUUPBPDensityBeam upbp_load_density_beam(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPDensityBeamStride;
  GPUUPBPDensityBeam result = (GPUUPBPDensityBeam)0;
  result.beam = upbp_load_beam_at_offset(buffer, base_offset + kGPUUPBPDensityBeamBeamOffset);
  result.interval = upbp_load_interval_at_offset(buffer, base_offset + kGPUUPBPDensityBeamIntervalOffset);
  result.source_throughput.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPDensityBeamSourceThroughputOffset);
  result.transport_weight.payload = wavefront_load_float4(buffer, base_offset + kGPUUPBPDensityBeamTransportWeightOffset);
  result.transport_log_pdf_forward = asfloat(buffer.Load(base_offset + kGPUUPBPDensityBeamTransportLogPdfForwardOffset));
  result.transport_log_pdf_reverse = asfloat(buffer.Load(base_offset + kGPUUPBPDensityBeamTransportLogPdfReverseOffset));
  result.transport_distance = asfloat(buffer.Load(base_offset + kGPUUPBPDensityBeamTransportDistanceOffset));
  result.log_d_shared = asfloat(buffer.Load(base_offset + kGPUUPBPDensityBeamDSharedOffset));
  result.log_d_pde_reverse_coefficient = asfloat(buffer.Load(base_offset + kGPUUPBPDensityBeamDPDEReverseCoefficientOffset));
  result.log_d_pde_constant = asfloat(buffer.Load(base_offset + kGPUUPBPDensityBeamDPDEConstantOffset));
  result.source_event_log_density = asfloat(buffer.Load(base_offset + kGPUUPBPDensityBeamSourceEventLogDensityOffset));
  result.interval_distance = asfloat(buffer.Load(base_offset + kGPUUPBPDensityBeamIntervalDistanceOffset));
  result.flags = buffer.Load(base_offset + kGPUUPBPDensityBeamFlagsOffset);
  result.event_buffer = buffer.Load(base_offset + kGPUUPBPDensityBeamEventBufferOffset);
  return result;
}

void upbp_store_density_beam(uint descriptor_index, uint index, GPUUPBPDensityBeam value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPDensityBeamStride;
  upbp_store_beam_at_offset(buffer, base_offset + kGPUUPBPDensityBeamBeamOffset, value.beam);
  upbp_store_interval_at_offset(buffer, base_offset + kGPUUPBPDensityBeamIntervalOffset, value.interval);
  wavefront_store_float4(buffer, base_offset + kGPUUPBPDensityBeamSourceThroughputOffset, value.source_throughput.payload);
  wavefront_store_float4(buffer, base_offset + kGPUUPBPDensityBeamTransportWeightOffset, value.transport_weight.payload);
  buffer.Store(base_offset + kGPUUPBPDensityBeamTransportLogPdfForwardOffset, asuint(value.transport_log_pdf_forward));
  buffer.Store(base_offset + kGPUUPBPDensityBeamTransportLogPdfReverseOffset, asuint(value.transport_log_pdf_reverse));
  buffer.Store(base_offset + kGPUUPBPDensityBeamTransportDistanceOffset, asuint(value.transport_distance));
  buffer.Store(base_offset + kGPUUPBPDensityBeamDSharedOffset, asuint(value.log_d_shared));
  buffer.Store(base_offset + kGPUUPBPDensityBeamDPDEReverseCoefficientOffset, asuint(value.log_d_pde_reverse_coefficient));
  buffer.Store(base_offset + kGPUUPBPDensityBeamDPDEConstantOffset, asuint(value.log_d_pde_constant));
  buffer.Store(base_offset + kGPUUPBPDensityBeamSourceEventLogDensityOffset, asuint(value.source_event_log_density));
  buffer.Store(base_offset + kGPUUPBPDensityBeamIntervalDistanceOffset, asuint(value.interval_distance));
  buffer.Store(base_offset + kGPUUPBPDensityBeamFlagsOffset, value.flags);
  buffer.Store(base_offset + kGPUUPBPDensityBeamEventBufferOffset, value.event_buffer);
}

GPUUPBPDensityBatch upbp_load_density_batch(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPDensityBatchStride;
  GPUUPBPDensityBatch result = (GPUUPBPDensityBatch)0;
  result.surface_point_buffer = buffer.Load(base_offset + kGPUUPBPDensityBatchSurfacePointBufferOffset);
  result.surface_point_count = buffer.Load(base_offset + kGPUUPBPDensityBatchSurfacePointCountOffset);
  result.medium_point_buffer = buffer.Load(base_offset + kGPUUPBPDensityBatchMediumPointBufferOffset);
  result.medium_point_count = buffer.Load(base_offset + kGPUUPBPDensityBatchMediumPointCountOffset);
  result.beam_buffer = buffer.Load(base_offset + kGPUUPBPDensityBatchBeamBufferOffset);
  result.beam_count = buffer.Load(base_offset + kGPUUPBPDensityBatchBeamCountOffset);
  result.beam_instance_offset = buffer.Load(base_offset + kGPUUPBPDensityBatchBeamInstanceOffsetOffset);
  result.selected_beam_count = buffer.Load(base_offset + kGPUUPBPDensityBatchSelectedBeamCountOffset);
  return result;
}

GPUUPBPBeamReference upbp_load_beam_reference(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  const uint base_offset = index * kGPUUPBPBeamReferenceStride;
  GPUUPBPBeamReference result = (GPUUPBPBeamReference)0;
  result.origin = wavefront_load_float3(buffer, base_offset + kGPUUPBPBeamReferenceOriginOffset);
  result.length = asfloat(buffer.Load(base_offset + kGPUUPBPBeamReferenceLengthOffset));
  result.direction = wavefront_load_float3(buffer, base_offset + kGPUUPBPBeamReferenceDirectionOffset);
  result.path_length = buffer.Load(base_offset + kGPUUPBPBeamReferencePathLengthOffset);
  result.medium_index = buffer.Load(base_offset + kGPUUPBPBeamReferenceMediumIndexOffset);
  return result;
}

bool upbp_append_index(GPUUPBPResources resources, uint counter_index, uint capacity, uint overflow_flag, out uint index) {
  index = kInvalidIndex;
  if (resources.counter_buffer == kInvalidIndex) {
    return false;
  }
  RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(resources.counter_buffer);
  counters.InterlockedAdd(counter_index * 4u, 1u, index);
  if (index < capacity) {
    return true;
  }
  uint ignored = 0u;
  counters.InterlockedOr(GPUUPBPCounterIndex::OverflowFlags * 4u, overflow_flag, ignored);
  index = kInvalidIndex;
  return false;
}

bool upbp_append_partitioned_index(GPUUPBPResources resources, bool from_camera, uint light_counter_index, uint camera_counter_index, uint light_capacity, uint camera_capacity,
  uint overflow_flag, out uint index) {
  const uint counter_index = from_camera ? camera_counter_index : light_counter_index;
  const uint capacity = from_camera ? camera_capacity : light_capacity;
  uint local_index = kInvalidIndex;
  if (upbp_append_index(resources, counter_index, capacity, overflow_flag, local_index) == false) {
    index = kInvalidIndex;
    return false;
  }
  index = local_index;
  return true;
}

bool upbp_try_append_partitioned_index_wave(GPUUPBPResources resources, bool from_camera, uint light_counter_index, uint camera_counter_index, uint light_capacity,
  uint camera_capacity, out uint index) {
  index = kInvalidIndex;
  const uint wave_count = WaveActiveCountBits(true);
  const uint wave_offset = WavePrefixCountBits(true);
  uint wave_base = kInvalidIndex;
  uint reserved_count = 0u;
  if (WaveIsFirstLane() && (resources.counter_buffer != kInvalidIndex)) {
    const uint counter_index = from_camera ? camera_counter_index : light_counter_index;
    const uint capacity = from_camera ? camera_capacity : light_capacity;
    RWByteAddressBuffer counters = WAVEFRONT_RW_BUFFER(resources.counter_buffer);
    uint observed = counters.Load(counter_index * sizeof(uint));
    while (observed < capacity) {
      const uint available_count = capacity - observed;
      const uint requested_count = min(wave_count, available_count);
      uint original = 0u;
      counters.InterlockedCompareExchange(counter_index * sizeof(uint), observed, observed + requested_count, original);
      if (original == observed) {
        wave_base = observed;
        reserved_count = requested_count;
        break;
      }
      observed = original;
    }
  }
  wave_base = WaveReadLaneFirst(wave_base);
  reserved_count = WaveReadLaneFirst(reserved_count);
  if (wave_offset >= reserved_count) {
    return false;
  }
  index = wave_base + wave_offset;
  return true;
}

bool upbp_try_grid_cell(float3 position, float cell_size, out int3 cell) {
  cell = (int3)0;
  const float maximum_coordinate = 8388607.0f;
  if ((cell_size <= 0.0f) || (isfinite(cell_size) == false) || (all(isfinite(position)) == false)) {
    return false;
  }
  const float3 coordinate = floor(position / cell_size);
  if ((all(isfinite(coordinate)) == false) || any(coordinate < -maximum_coordinate) || any(coordinate > maximum_coordinate)) {
    return false;
  }
  cell = int3(coordinate);
  return true;
}

uint upbp_path_state_index(GPUUPBPResources resources, bool from_camera, uint path_index) {
  return path_index;
}
