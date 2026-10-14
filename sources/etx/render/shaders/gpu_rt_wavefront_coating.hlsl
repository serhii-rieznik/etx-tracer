#define ETX_COATING_PHASE_wavefront_coating_clear_main          0
#define ETX_COATING_PHASE_wavefront_coating_direct_main         1
#define ETX_COATING_PHASE_wavefront_coating_sample_begin_main   2
#define ETX_COATING_PHASE_wavefront_coating_sample_delta_main   3
#define ETX_COATING_PHASE_wavefront_coating_sample_resolve_main 4
#define ETX_COATING_PHASE_wavefront_coating_advance_main        5
#define ETX_COATING_PHASE_wavefront_coating_phase_main          6
#define ETX_COATING_PHASE_wavefront_coating_base_evaluate_main  7
#define ETX_COATING_PHASE_wavefront_coating_base_sample_main    8
#define ETX_COATING_PHASE_wavefront_coating_exit_main           9
#define ETX_COATING_PHASE_wavefront_coating_exit_pdf_main       10
#define ETX_COATING_PHASE_wavefront_coating_velvet_main         11
#define ETX_COATING_PHASE_wavefront_coating_finish_main         12
#define ETX_COATING_PHASE_EXPAND(entry)                         ETX_COATING_PHASE_##entry
#define ETX_COATING_PHASE_SELECT(entry)                         ETX_COATING_PHASE_EXPAND(entry)
#define ETX_COATING_PHASE                                       ETX_COATING_PHASE_SELECT(ETX_STAGE_ENTRY)

#include "gpu_rt_wavefront_transport_work_storage.hlsl"
#include "gpu_rt_wavefront_coating_storage.hlsl"

BSDFResourceContext wavefront_coating_context() {
  return make_bsdf_resource_gpu_context(constants.scene.images, constants.scene.spectrums, constants.scene.spectral_values, constants.scene.energy_compensation_interfaces,
    constants.scene.scene_globals);
}

BSDFCoatedConductor wavefront_coating_prepare(RWByteAddressBuffer buffer, uint offset, BSDFResourceContext context, out BSDFData data) {
  data = wavefront_coating_load_data(buffer, offset + kGPUCoatingRequestDataOffset);
  const uint material_index = buffer.Load(offset + kGPUCoatingRequestMaterialOffset);
  const Material material = gpu_abi_load_material_full(WAVEFRONT_RO_BUFFER(constants.scene.materials), material_index);
  return bsdf_coated_conductor_prepare(context, data, material);
}

#if ETX_COATING_PHASE == 0
[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 id : SV_DispatchThreadID) {
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.coating_request_buffer);
  if ((constants.dispatch_item_offset + id.x) == 0u) {
    buffer.Store(kGPUCoatingRequestPendingOffset, 0u);
    const uint path_sets = uint(resources.camera_state_buffer != kInvalidIndex) + uint(resources.light_state_buffer != kInvalidIndex);
    const uint capacity = resources.path_capacity * path_sets;
    buffer.Store(kGPUCoatingRequestCapacityOffset, resources.vcm_coating_buffer != kInvalidIndex ? max(capacity, kGPUVCMCoatingLaneCount) : capacity);
  }
  if (id.x < constants.dispatch_item_count) {
    buffer.Store(wavefront_coating_request_offset(constants.dispatch_item_offset + id.x) + kGPUCoatingRequestActiveOffset, 0u);
  }
}
#elif ETX_COATING_PHASE == 1
[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.coating_request_buffer);
  if (request_index >= buffer.Load(kGPUCoatingRequestCapacityOffset)) {
    return;
  }
  const uint offset = wavefront_coating_request_offset(request_index);
  const uint walk_offset = offset + kGPUCoatingRequestWalkOffset;
  if ((buffer.Load(offset + kGPUCoatingRequestActiveOffset) == 0u) || (buffer.Load(walk_offset + kGPUCoatingWalkOperationOffset) != BSDFCoatingWalk::Direct)) {
    return;
  }
  const BSDFResourceContext context = wavefront_coating_context();
  BSDFData data = wavefront_coating_load_data(buffer, offset + kGPUCoatingRequestDataOffset);
  const uint material_index = buffer.Load(offset + kGPUCoatingRequestMaterialOffset);
  const Material material = gpu_abi_load_material_full(WAVEFRONT_RO_BUFFER(constants.scene.materials), material_index);
  if (bsdf_coating_transmission_base(material_coating_base_class(material))) {
    const BSDFCoatedConductor layer = bsdf_coated_conductor_prepare(context, data, material);
    BSDFCoatingWalk state = (BSDFCoatingWalk)0;
    state.outgoing = asfloat(buffer.Load3(walk_offset + kGPUCoatingWalkOutgoingOffset));
    Sampler sampler = wavefront_coating_load_sampler(buffer, offset + kGPUCoatingRequestSamplerOffset);
    bsdf_sampler_pop_fixed(sampler);
    bsdf_coating_transmission_direct(context, layer, state, sampler);
    wavefront_coating_store_response(buffer, walk_offset + kGPUCoatingWalkResultOffset, state.result);
    buffer.Store(walk_offset + kGPUCoatingWalkOperationOffset, state.operation);
    buffer.Store(offset + kGPUCoatingRequestSamplerOffset, sampler.seed);
    return;
  }
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  data.nrm = float3(0.0f, 0.0f, 1.0f);
  data.tan = float3(1.0f, 0.0f, 0.0f);
  data.btn = float3(0.0f, 1.0f, 0.0f);
  data.w_i = local_frame_to_local(frame, data.w_i);
  const float3 outgoing = asfloat(buffer.Load3(walk_offset + kGPUCoatingWalkOutgoingOffset));
  Sampler sampler = wavefront_coating_load_sampler(buffer, offset + kGPUCoatingRequestSamplerOffset);
  bsdf_sampler_pop_fixed(sampler);
  const SpectralResponse result = bsdf_coating_direct_function(context, data, bsdf_coating_top_interface_optics(material), outgoing, sampler);
  wavefront_coating_store_response(buffer, walk_offset + kGPUCoatingWalkResultOffset, result);
  buffer.Store(walk_offset + kGPUCoatingWalkOperationOffset, uint(BSDFCoatingWalk::SampleBegin));
  buffer.Store(offset + kGPUCoatingRequestSamplerOffset, sampler.seed);
}
#elif ETX_COATING_PHASE == 11
[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
# if ETX_ENABLE_COATED_VELVET_SUBSTRATE
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.coating_request_buffer);
  if (request_index >= buffer.Load(kGPUCoatingRequestCapacityOffset)) {
    return;
  }
  const uint offset = wavefront_coating_request_offset(request_index);
  const uint operation = buffer.Load(offset + kGPUCoatingRequestWalkOffset + kGPUCoatingWalkOperationOffset);
  if ((buffer.Load(offset + kGPUCoatingRequestActiveOffset) == 0u) ||
      ((operation != BSDFCoatingWalk::VelvetBaseEvaluate) && (operation != BSDFCoatingWalk::VelvetBaseSample) && (operation != BSDFCoatingWalk::VelvetProposal))) {
    return;
  }
  const BSDFResourceContext context = wavefront_coating_context();
  BSDFData data;
  const BSDFCoatedConductor layer = wavefront_coating_prepare(buffer, offset, context, data);
  BSDFCoatingWalk state = wavefront_coating_load_walk(buffer, offset + kGPUCoatingRequestWalkOffset);
  Sampler sampler = wavefront_coating_load_sampler(buffer, offset + kGPUCoatingRequestSamplerOffset);
  if (operation == BSDFCoatingWalk::VelvetProposal) {
    const BSDFEval evaluation = bsdf_coating_walk_velvet_evaluation(context, layer, state, sampler);
    if (state.velvet.operation == BSDFVelvetWalk::Complete) {
      const BSDFSample sample = bsdf_velvet_sample_resolve(data.spectrum_sample, state.velvet_outgoing, evaluation);
      const BSDFCoatingProposal proposal = bsdf_coated_conductor_proposal_resolve(layer, sample);
      if (proposal.kind == BSDFCoatingProposal::Complete) {
        wavefront_coating_store_sample(buffer, offset + kGPUCoatingRequestResultOffset, proposal.sample);
        buffer.Store(offset + kGPUCoatingRequestSamplerOffset, sampler.seed);
        buffer.Store(offset + kGPUCoatingRequestActiveOffset, 0u);
        uint unused;
        buffer.InterlockedAdd(kGPUCoatingRequestPendingOffset, 0xffffffffu, unused);
        return;
      }
      buffer.Store3(offset + kGPUCoatingRequestOutgoingOffset, asuint(proposal.outgoing));
      state = bsdf_coating_walk_begin(layer, local_frame_to_local(layer.frame, proposal.outgoing));
    }
  } else {
    bsdf_coating_walk_velvet(context, layer, state, sampler);
  }
  buffer.Store(offset + kGPUCoatingRequestSamplerOffset, sampler.seed);
  wavefront_coating_store_walk(buffer, offset + kGPUCoatingRequestWalkOffset, state);
# endif
}
#elif ETX_COATING_PHASE == 12
[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.coating_request_buffer);
  if (request_index >= buffer.Load(kGPUCoatingRequestCapacityOffset)) {
    return;
  }
  const uint offset = wavefront_coating_request_offset(request_index);
  if ((buffer.Load(offset + kGPUCoatingRequestActiveOffset) == 0u) ||
      (buffer.Load(offset + kGPUCoatingRequestWalkOffset + kGPUCoatingWalkOperationOffset) != BSDFCoatingWalk::Complete)) {
    return;
  }
  const BSDFResourceContext context = wavefront_coating_context();
  BSDFData data;
  const BSDFCoatedConductor layer = wavefront_coating_prepare(buffer, offset, context, data);
  const BSDFCoatingWalk state = wavefront_coating_load_walk(buffer, offset + kGPUCoatingRequestWalkOffset);
  Sampler sampler = wavefront_coating_load_sampler(buffer, offset + kGPUCoatingRequestSamplerOffset);
  const BSDFEval result = bsdf_coated_conductor_resolve_evaluation(context, layer, state.outgoing, state.result, sampler);
  if (buffer.Load(offset + kGPUCoatingRequestKindOffset) == GPUCoatingRequestKind::Sample) {
    const float3 outgoing = asfloat(buffer.Load3(offset + kGPUCoatingRequestOutgoingOffset));
    const uint proposal_properties = buffer.Load(offset + kGPUCoatingRequestResultOffset + 44u) & BSDFSample::WavelengthDependentDirection;
    BSDFSample sample = bsdf_coated_conductor_resolve_sample(data, outgoing, result);
    sample.properties |= proposal_properties;
    wavefront_coating_store_sample(buffer, offset + kGPUCoatingRequestResultOffset, sample);
  } else {
    wavefront_coating_store_evaluation(buffer, offset + kGPUCoatingRequestResultOffset, result);
  }
  buffer.Store(offset + kGPUCoatingRequestSamplerOffset, sampler.seed);
  buffer.Store(offset + kGPUCoatingRequestActiveOffset, 0u);
  uint unused;
  buffer.InterlockedAdd(kGPUCoatingRequestPendingOffset, 0xffffffffu, unused);
}
#else
[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.coating_request_buffer);
  if (request_index >= buffer.Load(kGPUCoatingRequestCapacityOffset)) {
    return;
  }
  const uint offset = wavefront_coating_request_offset(request_index);
  const uint current_operation = buffer.Load(offset + kGPUCoatingRequestWalkOffset + kGPUCoatingWalkOperationOffset);
# if ETX_COATING_PHASE == 2
  const uint operation = BSDFCoatingWalk::SampleBegin;
# elif ETX_COATING_PHASE == 3
  const uint operation = BSDFCoatingWalk::SampleDelta;
# elif ETX_COATING_PHASE == 4
  const uint operation = BSDFCoatingWalk::SampleResolve;
# elif ETX_COATING_PHASE == 5
  const uint operation = BSDFCoatingWalk::Advance;
# elif ETX_COATING_PHASE == 6
  const uint operation = BSDFCoatingWalk::Phase;
# elif ETX_COATING_PHASE == 7
  const uint operation = BSDFCoatingWalk::BaseEvaluate;
# elif ETX_COATING_PHASE == 8
  const uint operation = BSDFCoatingWalk::BaseSample;
# elif ETX_COATING_PHASE == 9
  const uint operation = BSDFCoatingWalk::ExitEvaluate;
# elif ETX_COATING_PHASE == 10
  const uint operation = BSDFCoatingWalk::ExitPdf;
# endif
  const bool entrance_evaluate = (operation == BSDFCoatingWalk::BaseEvaluate) && (current_operation == BSDFCoatingWalk::EntranceEvaluate);
  if ((buffer.Load(offset + kGPUCoatingRequestActiveOffset) == 0u) || ((current_operation != operation) && (entrance_evaluate == false))) {
    return;
  }
  const BSDFResourceContext context = wavefront_coating_context();
  BSDFData data;
  const BSDFCoatedConductor layer = wavefront_coating_prepare(buffer, offset, context, data);
  BSDFCoatingWalk state = wavefront_coating_load_walk(buffer, offset + kGPUCoatingRequestWalkOffset);
  Sampler sampler = wavefront_coating_load_sampler(buffer, offset + kGPUCoatingRequestSamplerOffset);
  bsdf_sampler_pop_fixed(sampler);
# if ETX_COATING_PHASE == 2
  bsdf_coating_walk_sample_begin(context, layer, state, sampler);
# elif ETX_COATING_PHASE == 3
  bsdf_coating_walk_sample_delta(context, layer, state, sampler);
# elif ETX_COATING_PHASE == 4
  bsdf_coating_walk_sample_resolve(context, layer, state);
# elif ETX_COATING_PHASE == 5
  bsdf_coating_walk_advance(layer, state, sampler);
# elif ETX_COATING_PHASE == 6
  bsdf_coating_walk_phase(context, layer, state, sampler);
# elif ETX_COATING_PHASE == 7
  if (entrance_evaluate) {
    bsdf_coating_transmission_entrance_evaluate(context, layer, state, sampler);
  } else {
    BSDFCoatingRenewalPrepared renewal;
    bool renewed = false;
    if (bsdf_coating_renewal_prepare(context, layer, sampler, renewal)) {
      renewed = bsdf_coating_renewal_walk_step(renewal, layer, state, sampler);
    }
    if (renewed == false) {
      bsdf_coating_walk_base_evaluate(context, layer, state, sampler);
    }
  }
# elif ETX_COATING_PHASE == 8
  bsdf_coating_walk_base_sample(context, layer, state, sampler);
# elif ETX_COATING_PHASE == 9
  bsdf_coating_walk_exit_evaluate(context, layer, state, sampler);
# elif ETX_COATING_PHASE == 10
  bsdf_coating_walk_exit_pdf(context, layer, state, sampler);
# endif
  buffer.Store(offset + kGPUCoatingRequestSamplerOffset, sampler.seed);
# if (ETX_COATING_PHASE == 4) || (ETX_COATING_PHASE == 5) || (ETX_COATING_PHASE == 7) || (ETX_COATING_PHASE == 9) || (ETX_COATING_PHASE == 10)
  const uint walk_offset = offset + kGPUCoatingRequestWalkOffset;
  buffer.Store(walk_offset + kGPUCoatingWalkOperationOffset, state.operation);
#  if ETX_COATING_PHASE == 4
  wavefront_coating_store_sample(buffer, walk_offset + kGPUCoatingWalkReverseExitOffset, state.reverse_exit);
  wavefront_coating_store_response(buffer, walk_offset + kGPUCoatingWalkThroughputOffset, state.throughput);
  buffer.Store3(walk_offset + kGPUCoatingWalkDirectionOffset, asuint(state.direction));
  buffer.Store2(walk_offset + kGPUCoatingWalkHeightOffset, asuint(float2(state.height, state.continuation_pdf)));
  buffer.Store2(walk_offset + kGPUCoatingWalkSampleTargetOffset, uint2(state.sample_target, state.exit_from_phase ? 1u : 0u));
  buffer.Store(walk_offset + kGPUCoatingWalkContinuationDeltaOffset, state.continuation_delta ? 1u : 0u);
#  elif ETX_COATING_PHASE == 5
  wavefront_coating_store_response(buffer, walk_offset + kGPUCoatingWalkThroughputOffset, state.throughput);
  buffer.Store(walk_offset + kGPUCoatingWalkHeightOffset, asuint(state.height));
  buffer.Store(walk_offset + kGPUCoatingWalkDepthOffset, state.depth);
  buffer.Store(walk_offset + kGPUCoatingWalkSampleTargetOffset, state.sample_target);
#  elif ETX_COATING_PHASE == 7
  wavefront_coating_store_response(buffer, walk_offset + kGPUCoatingWalkResultOffset, state.result);
  wavefront_coating_store_response(buffer, walk_offset + kGPUCoatingWalkThroughputOffset, state.throughput);
  buffer.Store3(walk_offset + kGPUCoatingWalkDirectionOffset, asuint(state.direction));
  buffer.Store2(walk_offset + kGPUCoatingWalkHeightOffset, asuint(float2(state.height, state.continuation_pdf)));
  buffer.Store(walk_offset + kGPUCoatingWalkDepthOffset, state.depth);
  buffer.Store2(walk_offset + kGPUCoatingWalkSampleTargetOffset, uint2(state.sample_target, state.exit_from_phase ? 1u : 0u));
  buffer.Store3(walk_offset + 256u, uint3(state.interface_sample.prepared.supported ? 1u : 0u, state.interface_sample.prepared.conductor_delta ? 1u : 0u,
                                      state.interface_sample.prepared.dielectric_delta ? 1u : 0u));
  buffer.Store(walk_offset + 376u, state.interface_sample.proposal.wavelength_dependent_direction ? 1u : 0u);
  buffer.Store(walk_offset + kGPUCoatingWalkContinuationDeltaOffset, state.continuation_delta ? 1u : 0u);
#   if ETX_ENABLE_COATED_VELVET_SUBSTRATE
  if (state.operation == BSDFCoatingWalk::VelvetBaseEvaluate) {
    wavefront_coating_store_velvet(buffer, walk_offset + kGPUCoatingWalkVelvetOffset, state.velvet);
    buffer.Store3(walk_offset + kGPUCoatingWalkVelvetOutgoingOffset, asuint(state.velvet_outgoing));
  }
#   endif
#  elif ETX_COATING_PHASE == 9
  wavefront_coating_store_response(buffer, walk_offset + kGPUCoatingWalkExitFunctionOffset, state.exit_func);
#  elif ETX_COATING_PHASE == 10
  wavefront_coating_store_response(buffer, walk_offset + kGPUCoatingWalkResultOffset, state.result);
#  endif
# else
  wavefront_coating_store_walk(buffer, offset + kGPUCoatingRequestWalkOffset, state);
# endif
}
#endif
