#pragma once

#include "gpu_rt_wavefront_base.hlsl"
#include <interop/gpu_coating_abi.hxx>
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>
#include <access/material_access_gpu.hxx>
#include <interop/bsdf_coated_conductor_shared.hxx>

uint wavefront_coating_request_index(GPUWavefrontResources resources, bool from_camera, uint path_index) {
  return path_index + (((from_camera == false) && (resources.camera_state_buffer != kInvalidIndex)) ? resources.path_capacity : 0u);
}

uint wavefront_coating_request_offset(uint index) {
  return kGPUCoatingRequestHeaderSize + index * kGPUCoatingRequestStride;
}

void wavefront_coating_store_response(RWByteAddressBuffer buffer, uint offset, SpectralResponse value) {
  buffer.Store3(offset, asuint(value.integrated));
  buffer.Store3(offset + 12u, uint3(asuint(value.value), asuint(value.wavelength), value.flags));
}

SpectralResponse wavefront_coating_load_response(RWByteAddressBuffer buffer, uint offset) {
  SpectralResponse value = (SpectralResponse)0;
  value.integrated = asfloat(buffer.Load3(offset));
  value.value = asfloat(buffer.Load(offset + 12u));
  value.wavelength = asfloat(buffer.Load(offset + 16u));
  value.flags = buffer.Load(offset + 20u);
  return value;
}

void wavefront_coating_store_sample(RWByteAddressBuffer buffer, uint offset, BSDFSample value) {
  wavefront_coating_store_response(buffer, offset, value.weight);
  buffer.Store3(offset + 24u, asuint(value.w_o));
  buffer.Store4(offset + 36u, uint4(asuint(value.pdf), asuint(value.eta), value.properties, value.medium_index));
}

BSDFSample wavefront_coating_load_sample_value(RWByteAddressBuffer buffer, uint offset) {
  BSDFSample value = (BSDFSample)0;
  value.weight = wavefront_coating_load_response(buffer, offset);
  value.w_o = asfloat(buffer.Load3(offset + 24u));
  value.pdf = asfloat(buffer.Load(offset + 36u));
  value.eta = asfloat(buffer.Load(offset + 40u));
  value.properties = buffer.Load(offset + 44u);
  value.medium_index = buffer.Load(offset + 48u);
  return value;
}

void wavefront_coating_store_evaluation(RWByteAddressBuffer buffer, uint offset, BSDFEval value) {
  wavefront_coating_store_response(buffer, offset, value.func);
  wavefront_coating_store_response(buffer, offset + 24u, value.bsdf);
  buffer.Store4(offset + 48u, uint4(asuint(value.pdf), asuint(value.eta), value.properties, value.medium_index));
}

BSDFEval wavefront_coating_load_evaluation_value(RWByteAddressBuffer buffer, uint offset) {
  BSDFEval value = (BSDFEval)0;
  value.func = wavefront_coating_load_response(buffer, offset);
  value.bsdf = wavefront_coating_load_response(buffer, offset + 24u);
  value.pdf = asfloat(buffer.Load(offset + 48u));
  value.eta = asfloat(buffer.Load(offset + 52u));
  value.properties = buffer.Load(offset + 56u);
  value.medium_index = buffer.Load(offset + 60u);
  return value;
}

void wavefront_coating_store_sampler(RWByteAddressBuffer buffer, uint offset, Sampler sampler) {
  buffer.Store4(offset, uint4(sampler.seed, asuint(sampler.fixed_u), asuint(sampler.fixed_v), asuint(sampler.fixed_w)));
}

Sampler wavefront_coating_load_sampler(RWByteAddressBuffer buffer, uint offset) {
  const uint4 value = buffer.Load4(offset);
  Sampler sampler = (Sampler)0;
  sampler.seed = value.x;
  sampler.fixed_u = asfloat(value.y);
  sampler.fixed_v = asfloat(value.z);
  sampler.fixed_w = asfloat(value.w);
  return sampler;
}

void wavefront_coating_store_data(RWByteAddressBuffer buffer, uint offset, BSDFData data) {
  buffer.Store3(offset, asuint(data.pos));
  buffer.Store3(offset + 12u, asuint(data.nrm));
  buffer.Store3(offset + 24u, asuint(data.tan));
  buffer.Store3(offset + 36u, asuint(data.btn));
  buffer.Store2(offset + 48u, asuint(data.tex));
  buffer.Store3(offset + 56u, asuint(data.w_i));
  buffer.Store4(offset + 68u, uint4(asuint(data.spectrum_sample.wavelength), data.spectrum_sample.flags, data.path_source, data.current_medium));
}

BSDFData wavefront_coating_load_data(RWByteAddressBuffer buffer, uint offset) {
  BSDFData data = (BSDFData)0;
  data.pos = asfloat(buffer.Load3(offset));
  data.nrm = asfloat(buffer.Load3(offset + 12u));
  data.tan = asfloat(buffer.Load3(offset + 24u));
  data.btn = asfloat(buffer.Load3(offset + 36u));
  data.tex = asfloat(buffer.Load2(offset + 48u));
  data.w_i = asfloat(buffer.Load3(offset + 56u));
  data.spectrum_sample.wavelength = asfloat(buffer.Load(offset + 68u));
  data.spectrum_sample.flags = buffer.Load(offset + 72u);
  data.path_source = buffer.Load(offset + 76u);
  data.current_medium = buffer.Load(offset + 80u);
  return data;
}

#if ETX_ENABLE_COATED_VELVET_SUBSTRATE
void wavefront_coating_store_velvet(RWByteAddressBuffer buffer, uint offset, BSDFVelvetWalk state) {
  wavefront_coating_store_response(buffer, offset + 0u, state.value);
  wavefront_coating_store_response(buffer, offset + 24u, state.absorption);
  wavefront_coating_store_response(buffer, offset + 48u, state.throughput);
  wavefront_coating_store_response(buffer, offset + 72u, state.transmission);
  wavefront_coating_store_response(buffer, offset + 96u, state.multiple);
  buffer.Store3(offset + 120u, asuint(state.direction));
  buffer.Store3(offset + 132u, asuint(state.outgoing));
  buffer.Store3(offset + 144u, uint3(asuint(state.exponent), state.depth, state.operation));
}
BSDFVelvetWalk wavefront_coating_load_velvet(RWByteAddressBuffer buffer, uint offset) {
  BSDFVelvetWalk state = (BSDFVelvetWalk)0;
  state.value = wavefront_coating_load_response(buffer, offset + 0u);
  state.absorption = wavefront_coating_load_response(buffer, offset + 24u);
  state.throughput = wavefront_coating_load_response(buffer, offset + 48u);
  state.transmission = wavefront_coating_load_response(buffer, offset + 72u);
  state.multiple = wavefront_coating_load_response(buffer, offset + 96u);
  state.direction = asfloat(buffer.Load3(offset + 120u));
  state.outgoing = asfloat(buffer.Load3(offset + 132u));
  state.exponent = asfloat(buffer.Load(offset + 144u));
  state.depth = buffer.Load(offset + 148u);
  state.operation = buffer.Load(offset + 152u);
  return state;
}
#endif

void wavefront_coating_store_walk(RWByteAddressBuffer buffer, uint base_offset, BSDFCoatingWalk state) {
  buffer.Store2(base_offset + 0u, asuint(state.interface_sample.prepared.roughness));
  buffer.Store(base_offset + 8u, asuint(state.interface_sample.prepared.alpha));
  buffer.Store3(base_offset + 12u, asuint(state.interface_sample.prepared.ext_ior.eta.integrated));
  buffer.Store(base_offset + 24u, asuint(state.interface_sample.prepared.ext_ior.eta.value));
  buffer.Store(base_offset + 28u, asuint(state.interface_sample.prepared.ext_ior.eta.wavelength));
  buffer.Store(base_offset + 32u, state.interface_sample.prepared.ext_ior.eta.flags);
  buffer.Store3(base_offset + 36u, asuint(state.interface_sample.prepared.ext_ior.k.integrated));
  buffer.Store(base_offset + 48u, asuint(state.interface_sample.prepared.ext_ior.k.value));
  buffer.Store(base_offset + 52u, asuint(state.interface_sample.prepared.ext_ior.k.wavelength));
  buffer.Store(base_offset + 56u, state.interface_sample.prepared.ext_ior.k.flags);
  buffer.Store(base_offset + 60u, state.interface_sample.prepared.ext_ior.cls);
  buffer.Store3(base_offset + 64u, asuint(state.interface_sample.prepared.int_ior.eta.integrated));
  buffer.Store(base_offset + 76u, asuint(state.interface_sample.prepared.int_ior.eta.value));
  buffer.Store(base_offset + 80u, asuint(state.interface_sample.prepared.int_ior.eta.wavelength));
  buffer.Store(base_offset + 84u, state.interface_sample.prepared.int_ior.eta.flags);
  buffer.Store3(base_offset + 88u, asuint(state.interface_sample.prepared.int_ior.k.integrated));
  buffer.Store(base_offset + 100u, asuint(state.interface_sample.prepared.int_ior.k.value));
  buffer.Store(base_offset + 104u, asuint(state.interface_sample.prepared.int_ior.k.wavelength));
  buffer.Store(base_offset + 108u, state.interface_sample.prepared.int_ior.k.flags);
  buffer.Store(base_offset + 112u, state.interface_sample.prepared.int_ior.cls);
  buffer.Store3(base_offset + 116u, asuint(state.interface_sample.prepared.thinfilm.ior.eta.integrated));
  buffer.Store(base_offset + 128u, asuint(state.interface_sample.prepared.thinfilm.ior.eta.value));
  buffer.Store(base_offset + 132u, asuint(state.interface_sample.prepared.thinfilm.ior.eta.wavelength));
  buffer.Store(base_offset + 136u, state.interface_sample.prepared.thinfilm.ior.eta.flags);
  buffer.Store3(base_offset + 140u, asuint(state.interface_sample.prepared.thinfilm.ior.k.integrated));
  buffer.Store(base_offset + 152u, asuint(state.interface_sample.prepared.thinfilm.ior.k.value));
  buffer.Store(base_offset + 156u, asuint(state.interface_sample.prepared.thinfilm.ior.k.wavelength));
  buffer.Store(base_offset + 160u, state.interface_sample.prepared.thinfilm.ior.k.flags);
  buffer.Store(base_offset + 164u, state.interface_sample.prepared.thinfilm.ior.cls);
  buffer.Store3(base_offset + 168u, asuint(state.interface_sample.prepared.thinfilm.rgb_wavelengths));
  buffer.Store(base_offset + 180u, asuint(state.interface_sample.prepared.thinfilm.thickness));
  buffer.Store(base_offset + 184u, asuint(state.interface_sample.prepared.thinfilm.weight));
  buffer.Store3(base_offset + 188u, asuint(state.interface_sample.prepared.thinfilm.coating_ior.eta.integrated));
  buffer.Store(base_offset + 200u, asuint(state.interface_sample.prepared.thinfilm.coating_ior.eta.value));
  buffer.Store(base_offset + 204u, asuint(state.interface_sample.prepared.thinfilm.coating_ior.eta.wavelength));
  buffer.Store(base_offset + 208u, state.interface_sample.prepared.thinfilm.coating_ior.eta.flags);
  buffer.Store3(base_offset + 212u, asuint(state.interface_sample.prepared.thinfilm.coating_ior.k.integrated));
  buffer.Store(base_offset + 224u, asuint(state.interface_sample.prepared.thinfilm.coating_ior.k.value));
  buffer.Store(base_offset + 228u, asuint(state.interface_sample.prepared.thinfilm.coating_ior.k.wavelength));
  buffer.Store(base_offset + 232u, state.interface_sample.prepared.thinfilm.coating_ior.k.flags);
  buffer.Store(base_offset + 236u, state.interface_sample.prepared.thinfilm.coating_ior.cls);
  buffer.Store(base_offset + 240u, asuint(state.interface_sample.prepared.thinfilm.coating_thickness_nm));
  buffer.Store(base_offset + 244u, state.interface_sample.prepared.thinfilm.incident_inside);
  buffer.Store2(base_offset + 248u, asuint(state.interface_sample.prepared.interface_lut_value));
  buffer.Store(base_offset + 256u, (state.interface_sample.prepared.supported ? 1u : 0u));
  buffer.Store(base_offset + 260u, (state.interface_sample.prepared.conductor_delta ? 1u : 0u));
  buffer.Store(base_offset + 264u, (state.interface_sample.prepared.dielectric_delta ? 1u : 0u));
  buffer.Store4(base_offset + 268u, asuint(state.interface_sample.incident_pair.outside_albedo));
  buffer.Store4(base_offset + 284u, asuint(state.interface_sample.incident_pair.inside_albedo));
  buffer.Store(base_offset + 300u, asuint(state.interface_sample.incident_pair.outside_probability));
  buffer.Store(base_offset + 304u, asuint(state.interface_sample.incident_pair.inside_probability));
  buffer.Store3(base_offset + 308u, asuint(state.interface_sample.proposal.sample.weight.integrated));
  buffer.Store(base_offset + 320u, asuint(state.interface_sample.proposal.sample.weight.value));
  buffer.Store(base_offset + 324u, asuint(state.interface_sample.proposal.sample.weight.wavelength));
  buffer.Store(base_offset + 328u, state.interface_sample.proposal.sample.weight.flags);
  buffer.Store3(base_offset + 332u, asuint(state.interface_sample.proposal.sample.w_o));
  buffer.Store(base_offset + 344u, asuint(state.interface_sample.proposal.sample.pdf));
  buffer.Store(base_offset + 348u, asuint(state.interface_sample.proposal.sample.eta));
  buffer.Store(base_offset + 352u, state.interface_sample.proposal.sample.properties);
  buffer.Store(base_offset + 356u, state.interface_sample.proposal.sample.medium_index);
  buffer.Store3(base_offset + 360u, asuint(state.interface_sample.proposal.local_w_o));
  buffer.Store(base_offset + 372u, state.interface_sample.proposal.kind);
  buffer.Store(base_offset + 376u, (state.interface_sample.proposal.wavelength_dependent_direction ? 1u : 0u));
  buffer.Store3(base_offset + 380u, asuint(state.interface_sample.w_i_local));
  buffer.Store(base_offset + 392u, state.interface_sample.stage);
  buffer.Store3(base_offset + 396u, asuint(state.reverse_exit.weight.integrated));
  buffer.Store(base_offset + 408u, asuint(state.reverse_exit.weight.value));
  buffer.Store(base_offset + 412u, asuint(state.reverse_exit.weight.wavelength));
  buffer.Store(base_offset + 416u, state.reverse_exit.weight.flags);
  buffer.Store3(base_offset + 420u, asuint(state.reverse_exit.w_o));
  buffer.Store(base_offset + 432u, asuint(state.reverse_exit.pdf));
  buffer.Store(base_offset + 436u, asuint(state.reverse_exit.eta));
  buffer.Store(base_offset + 440u, state.reverse_exit.properties);
  buffer.Store(base_offset + 444u, state.reverse_exit.medium_index);
  buffer.Store3(base_offset + 448u, asuint(state.result.integrated));
  buffer.Store(base_offset + 460u, asuint(state.result.value));
  buffer.Store(base_offset + 464u, asuint(state.result.wavelength));
  buffer.Store(base_offset + 468u, state.result.flags);
  buffer.Store3(base_offset + 472u, asuint(state.throughput.integrated));
  buffer.Store(base_offset + 484u, asuint(state.throughput.value));
  buffer.Store(base_offset + 488u, asuint(state.throughput.wavelength));
  buffer.Store(base_offset + 492u, state.throughput.flags);
  buffer.Store3(base_offset + 496u, asuint(state.exit_func.integrated));
  buffer.Store(base_offset + 508u, asuint(state.exit_func.value));
  buffer.Store(base_offset + 512u, asuint(state.exit_func.wavelength));
  buffer.Store(base_offset + 516u, state.exit_func.flags);
  buffer.Store3(base_offset + 520u, asuint(state.outgoing));
  buffer.Store3(base_offset + 532u, asuint(state.direction));
  buffer.Store(base_offset + 544u, asuint(state.height));
  buffer.Store(base_offset + 548u, asuint(state.continuation_pdf));
  buffer.Store(base_offset + 552u, state.depth);
  buffer.Store(base_offset + 556u, state.operation);
  buffer.Store(base_offset + 560u, state.sample_target);
  buffer.Store(base_offset + 564u, (state.exit_from_phase ? 1u : 0u));
#if ETX_ENABLE_COATED_VELVET_SUBSTRATE
  wavefront_coating_store_velvet(buffer, base_offset + kGPUCoatingWalkVelvetOffset, state.velvet);
  buffer.Store3(base_offset + kGPUCoatingWalkVelvetOutgoingOffset, asuint(state.velvet_outgoing));
#endif
  buffer.Store(base_offset + kGPUCoatingWalkContinuationDeltaOffset, state.continuation_delta ? 1u : 0u);
}
BSDFCoatingWalk wavefront_coating_load_walk(RWByteAddressBuffer buffer, uint base_offset, bool load_interface_sample) {
  BSDFCoatingWalk state = (BSDFCoatingWalk)0;
  if (load_interface_sample) {
    state.interface_sample.prepared.roughness = asfloat(buffer.Load2(base_offset + 0u));
    state.interface_sample.prepared.alpha = asfloat(buffer.Load(base_offset + 8u));
    state.interface_sample.prepared.ext_ior.eta.integrated = asfloat(buffer.Load3(base_offset + 12u));
    state.interface_sample.prepared.ext_ior.eta.value = asfloat(buffer.Load(base_offset + 24u));
    state.interface_sample.prepared.ext_ior.eta.wavelength = asfloat(buffer.Load(base_offset + 28u));
    state.interface_sample.prepared.ext_ior.eta.flags = buffer.Load(base_offset + 32u);
    state.interface_sample.prepared.ext_ior.k.integrated = asfloat(buffer.Load3(base_offset + 36u));
    state.interface_sample.prepared.ext_ior.k.value = asfloat(buffer.Load(base_offset + 48u));
    state.interface_sample.prepared.ext_ior.k.wavelength = asfloat(buffer.Load(base_offset + 52u));
    state.interface_sample.prepared.ext_ior.k.flags = buffer.Load(base_offset + 56u);
    state.interface_sample.prepared.ext_ior.cls = buffer.Load(base_offset + 60u);
    state.interface_sample.prepared.int_ior.eta.integrated = asfloat(buffer.Load3(base_offset + 64u));
    state.interface_sample.prepared.int_ior.eta.value = asfloat(buffer.Load(base_offset + 76u));
    state.interface_sample.prepared.int_ior.eta.wavelength = asfloat(buffer.Load(base_offset + 80u));
    state.interface_sample.prepared.int_ior.eta.flags = buffer.Load(base_offset + 84u);
    state.interface_sample.prepared.int_ior.k.integrated = asfloat(buffer.Load3(base_offset + 88u));
    state.interface_sample.prepared.int_ior.k.value = asfloat(buffer.Load(base_offset + 100u));
    state.interface_sample.prepared.int_ior.k.wavelength = asfloat(buffer.Load(base_offset + 104u));
    state.interface_sample.prepared.int_ior.k.flags = buffer.Load(base_offset + 108u);
    state.interface_sample.prepared.int_ior.cls = buffer.Load(base_offset + 112u);
    state.interface_sample.prepared.thinfilm.ior.eta.integrated = asfloat(buffer.Load3(base_offset + 116u));
    state.interface_sample.prepared.thinfilm.ior.eta.value = asfloat(buffer.Load(base_offset + 128u));
    state.interface_sample.prepared.thinfilm.ior.eta.wavelength = asfloat(buffer.Load(base_offset + 132u));
    state.interface_sample.prepared.thinfilm.ior.eta.flags = buffer.Load(base_offset + 136u);
    state.interface_sample.prepared.thinfilm.ior.k.integrated = asfloat(buffer.Load3(base_offset + 140u));
    state.interface_sample.prepared.thinfilm.ior.k.value = asfloat(buffer.Load(base_offset + 152u));
    state.interface_sample.prepared.thinfilm.ior.k.wavelength = asfloat(buffer.Load(base_offset + 156u));
    state.interface_sample.prepared.thinfilm.ior.k.flags = buffer.Load(base_offset + 160u);
    state.interface_sample.prepared.thinfilm.ior.cls = buffer.Load(base_offset + 164u);
    state.interface_sample.prepared.thinfilm.rgb_wavelengths = asfloat(buffer.Load3(base_offset + 168u));
    state.interface_sample.prepared.thinfilm.thickness = asfloat(buffer.Load(base_offset + 180u));
    state.interface_sample.prepared.thinfilm.weight = asfloat(buffer.Load(base_offset + 184u));
    state.interface_sample.prepared.thinfilm.coating_ior.eta.integrated = asfloat(buffer.Load3(base_offset + 188u));
    state.interface_sample.prepared.thinfilm.coating_ior.eta.value = asfloat(buffer.Load(base_offset + 200u));
    state.interface_sample.prepared.thinfilm.coating_ior.eta.wavelength = asfloat(buffer.Load(base_offset + 204u));
    state.interface_sample.prepared.thinfilm.coating_ior.eta.flags = buffer.Load(base_offset + 208u);
    state.interface_sample.prepared.thinfilm.coating_ior.k.integrated = asfloat(buffer.Load3(base_offset + 212u));
    state.interface_sample.prepared.thinfilm.coating_ior.k.value = asfloat(buffer.Load(base_offset + 224u));
    state.interface_sample.prepared.thinfilm.coating_ior.k.wavelength = asfloat(buffer.Load(base_offset + 228u));
    state.interface_sample.prepared.thinfilm.coating_ior.k.flags = buffer.Load(base_offset + 232u);
    state.interface_sample.prepared.thinfilm.coating_ior.cls = buffer.Load(base_offset + 236u);
    state.interface_sample.prepared.thinfilm.coating_thickness_nm = asfloat(buffer.Load(base_offset + 240u));
    state.interface_sample.prepared.thinfilm.incident_inside = buffer.Load(base_offset + 244u);
    state.interface_sample.prepared.interface_lut_value = asfloat(buffer.Load2(base_offset + 248u));
    state.interface_sample.prepared.supported = (buffer.Load(base_offset + 256u) != 0u);
    state.interface_sample.prepared.conductor_delta = (buffer.Load(base_offset + 260u) != 0u);
    state.interface_sample.prepared.dielectric_delta = (buffer.Load(base_offset + 264u) != 0u);
    state.interface_sample.incident_pair.outside_albedo = asfloat(buffer.Load4(base_offset + 268u));
    state.interface_sample.incident_pair.inside_albedo = asfloat(buffer.Load4(base_offset + 284u));
    state.interface_sample.incident_pair.outside_probability = asfloat(buffer.Load(base_offset + 300u));
    state.interface_sample.incident_pair.inside_probability = asfloat(buffer.Load(base_offset + 304u));
    state.interface_sample.proposal.sample.weight.integrated = asfloat(buffer.Load3(base_offset + 308u));
    state.interface_sample.proposal.sample.weight.value = asfloat(buffer.Load(base_offset + 320u));
    state.interface_sample.proposal.sample.weight.wavelength = asfloat(buffer.Load(base_offset + 324u));
    state.interface_sample.proposal.sample.weight.flags = buffer.Load(base_offset + 328u);
    state.interface_sample.proposal.sample.w_o = asfloat(buffer.Load3(base_offset + 332u));
    state.interface_sample.proposal.sample.pdf = asfloat(buffer.Load(base_offset + 344u));
    state.interface_sample.proposal.sample.eta = asfloat(buffer.Load(base_offset + 348u));
    state.interface_sample.proposal.sample.properties = buffer.Load(base_offset + 352u);
    state.interface_sample.proposal.sample.medium_index = buffer.Load(base_offset + 356u);
    state.interface_sample.proposal.local_w_o = asfloat(buffer.Load3(base_offset + 360u));
    state.interface_sample.proposal.kind = buffer.Load(base_offset + 372u);
    state.interface_sample.proposal.wavelength_dependent_direction = (buffer.Load(base_offset + 376u) != 0u);
    state.interface_sample.w_i_local = asfloat(buffer.Load3(base_offset + 380u));
    state.interface_sample.stage = buffer.Load(base_offset + 392u);
  }
  state.reverse_exit.weight.integrated = asfloat(buffer.Load3(base_offset + 396u));
  state.reverse_exit.weight.value = asfloat(buffer.Load(base_offset + 408u));
  state.reverse_exit.weight.wavelength = asfloat(buffer.Load(base_offset + 412u));
  state.reverse_exit.weight.flags = buffer.Load(base_offset + 416u);
  state.reverse_exit.w_o = asfloat(buffer.Load3(base_offset + 420u));
  state.reverse_exit.pdf = asfloat(buffer.Load(base_offset + 432u));
  state.reverse_exit.eta = asfloat(buffer.Load(base_offset + 436u));
  state.reverse_exit.properties = buffer.Load(base_offset + 440u);
  state.reverse_exit.medium_index = buffer.Load(base_offset + 444u);
  state.result.integrated = asfloat(buffer.Load3(base_offset + 448u));
  state.result.value = asfloat(buffer.Load(base_offset + 460u));
  state.result.wavelength = asfloat(buffer.Load(base_offset + 464u));
  state.result.flags = buffer.Load(base_offset + 468u);
  state.throughput.integrated = asfloat(buffer.Load3(base_offset + 472u));
  state.throughput.value = asfloat(buffer.Load(base_offset + 484u));
  state.throughput.wavelength = asfloat(buffer.Load(base_offset + 488u));
  state.throughput.flags = buffer.Load(base_offset + 492u);
  state.exit_func.integrated = asfloat(buffer.Load3(base_offset + 496u));
  state.exit_func.value = asfloat(buffer.Load(base_offset + 508u));
  state.exit_func.wavelength = asfloat(buffer.Load(base_offset + 512u));
  state.exit_func.flags = buffer.Load(base_offset + 516u);
  state.outgoing = asfloat(buffer.Load3(base_offset + 520u));
  state.direction = asfloat(buffer.Load3(base_offset + 532u));
  state.height = asfloat(buffer.Load(base_offset + 544u));
  state.continuation_pdf = asfloat(buffer.Load(base_offset + 548u));
  state.depth = buffer.Load(base_offset + 552u);
  state.operation = buffer.Load(base_offset + 556u);
  state.sample_target = buffer.Load(base_offset + 560u);
  state.exit_from_phase = (buffer.Load(base_offset + 564u) != 0u);
#if ETX_ENABLE_COATED_VELVET_SUBSTRATE
  state.velvet = wavefront_coating_load_velvet(buffer, base_offset + kGPUCoatingWalkVelvetOffset);
  state.velvet_outgoing = asfloat(buffer.Load3(base_offset + kGPUCoatingWalkVelvetOutgoingOffset));
#endif
  state.continuation_delta = buffer.Load(base_offset + kGPUCoatingWalkContinuationDeltaOffset) != 0u;
  return state;
}

void wavefront_coating_begin_walk(GPUWavefrontResources resources, uint index, uint material_index, BSDFData data, float3 outgoing, BSDFCoatedConductor layer, uint kind,
  Sampler sampler) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.coating_request_buffer);
  const uint offset = wavefront_coating_request_offset(index);
  wavefront_coating_store_sampler(buffer, offset + kGPUCoatingRequestSamplerOffset, sampler);
  buffer.Store(offset + kGPUCoatingRequestMaterialOffset, material_index);
  wavefront_coating_store_data(buffer, offset + kGPUCoatingRequestDataOffset, data);
  buffer.Store3(offset + kGPUCoatingRequestOutgoingOffset, asuint(outgoing));
  buffer.Store(offset + kGPUCoatingRequestKindOffset, kind);
  const float3 local_outgoing = local_frame_to_local(layer.frame, outgoing);
  const bool invalid_direction = bsdf_coating_transmission_enabled(layer) ? ((abs(layer.data.w_i.z) <= kEpsilon) || (abs(local_outgoing.z) <= kEpsilon))
                                                                          : ((layer.data.w_i.z >= -kEpsilon) || (local_outgoing.z <= kEpsilon));
  if (invalid_direction) {
    if (kind == GPUCoatingRequestKind::Sample) {
      wavefront_coating_store_sample(buffer, offset + kGPUCoatingRequestResultOffset, bsdf_sample_zero(data.spectrum_sample));
    } else {
      wavefront_coating_store_evaluation(buffer, offset + kGPUCoatingRequestResultOffset, bsdf_eval_zero(data.spectrum_sample));
    }
    buffer.Store(offset + kGPUCoatingRequestActiveOffset, 0u);
    return;
  }
  wavefront_coating_store_walk(buffer, offset + kGPUCoatingRequestWalkOffset, bsdf_coating_walk_begin(layer, local_outgoing));
  buffer.Store(offset + kGPUCoatingRequestActiveOffset, 1u);
  uint unused;
  buffer.InterlockedAdd(kGPUCoatingRequestPendingOffset, 1u, unused);
}

void wavefront_coating_begin_sample(BSDFResourceContext context, GPUWavefrontResources resources, uint index, uint material_index, BSDFData data, Material material,
  inout Sampler sampler) {
  const BSDFCoatedConductor layer = bsdf_coated_conductor_prepare(context, data, material);
  const BSDFCoatingProposal proposal = bsdf_coated_conductor_propose_begin(context, data, layer, sampler);
  if (proposal.kind == BSDFCoatingProposal::Complete) {
    RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.coating_request_buffer);
    const uint offset = wavefront_coating_request_offset(index);
    wavefront_coating_store_sample(buffer, offset + kGPUCoatingRequestResultOffset, proposal.sample);
    wavefront_coating_store_sampler(buffer, offset + kGPUCoatingRequestSamplerOffset, sampler);
    buffer.Store(offset + kGPUCoatingRequestActiveOffset, 0u);
    return;
  }
  RWByteAddressBuffer result_buffer = WAVEFRONT_RW_BUFFER(resources.coating_request_buffer);
  result_buffer.Store(wavefront_coating_request_offset(index) + kGPUCoatingRequestResultOffset + 44u, proposal.sample.properties & BSDFSample::WavelengthDependentDirection);
#if ETX_ENABLE_COATED_VELVET_SUBSTRATE
  if (proposal.kind == BSDFCoatingProposal::Velvet) {
    RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.coating_request_buffer);
    const uint offset = wavefront_coating_request_offset(index);
    wavefront_coating_store_sampler(buffer, offset + kGPUCoatingRequestSamplerOffset, sampler);
    buffer.Store(offset + kGPUCoatingRequestMaterialOffset, material_index);
    wavefront_coating_store_data(buffer, offset + kGPUCoatingRequestDataOffset, data);
    buffer.Store(offset + kGPUCoatingRequestKindOffset, uint(GPUCoatingRequestKind::Sample));
    BSDFCoatingWalk state = (BSDFCoatingWalk)0;
    const Material substrate_material = bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
    bsdf_coating_walk_begin_velvet(context, layer.data, substrate_material, proposal.outgoing, BSDFCoatingWalk::VelvetProposal, state);
    wavefront_coating_store_walk(buffer, offset + kGPUCoatingRequestWalkOffset, state);
    buffer.Store(offset + kGPUCoatingRequestActiveOffset, 1u);
    uint unused;
    buffer.InterlockedAdd(kGPUCoatingRequestPendingOffset, 1u, unused);
    return;
  }
#endif
  wavefront_coating_begin_walk(resources, index, material_index, data, proposal.outgoing, layer, GPUCoatingRequestKind::Sample, sampler);
}

void wavefront_coating_begin_evaluation(BSDFResourceContext context, GPUWavefrontResources resources, uint index, uint material_index, BSDFData data, float3 outgoing,
  Material material, inout Sampler sampler) {
  const BSDFCoatedConductor layer = bsdf_coated_conductor_prepare(context, data, material);
  wavefront_coating_begin_walk(resources, index, material_index, data, outgoing, layer, GPUCoatingRequestKind::Evaluation, sampler);
}

BSDFSample wavefront_coating_load_sample(GPUWavefrontResources resources, uint index, out Sampler sampler) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.coating_request_buffer);
  const uint offset = wavefront_coating_request_offset(index);
  sampler = wavefront_coating_load_sampler(buffer, offset + kGPUCoatingRequestSamplerOffset);
  return wavefront_coating_load_sample_value(buffer, offset + kGPUCoatingRequestResultOffset);
}

BSDFEval wavefront_coating_load_evaluation(GPUWavefrontResources resources, uint index, out Sampler sampler) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.coating_request_buffer);
  const uint offset = wavefront_coating_request_offset(index);
  sampler = wavefront_coating_load_sampler(buffer, offset + kGPUCoatingRequestSamplerOffset);
  return wavefront_coating_load_evaluation_value(buffer, offset + kGPUCoatingRequestResultOffset);
}
