#pragma once

#include "gpu_rt_wavefront_base.hlsl"
#include <interop/gpu_emission_abi.hxx>
#include <access/bsdf_resource_gpu.hxx>
#include <interop/bsdf_core_shared.hxx>

uint wavefront_emission_request_offset(uint index, uint stride) {
  return kGPUEmissionRequestHeaderSize + index * stride;
}

void wavefront_emission_store_response(RWByteAddressBuffer buffer, uint offset, SpectralResponse value) {
  buffer.Store3(offset, asuint(value.integrated));
  buffer.Store3(offset + 12u, uint3(asuint(value.value), asuint(value.wavelength), value.flags));
}

SpectralResponse wavefront_emission_load_response(RWByteAddressBuffer buffer, uint offset) {
  SpectralResponse result = (SpectralResponse)0;
  result.integrated = asfloat(buffer.Load3(offset));
  result.value = asfloat(buffer.Load(offset + 12u));
  result.wavelength = asfloat(buffer.Load(offset + 16u));
  result.flags = buffer.Load(offset + 20u);
  return result;
}

void wavefront_emission_store_sampler(RWByteAddressBuffer buffer, uint offset, Sampler sampler) {
  buffer.Store4(offset, uint4(sampler.seed, asuint(sampler.fixed_u), asuint(sampler.fixed_v), asuint(sampler.fixed_w)));
}

Sampler wavefront_emission_load_sampler(RWByteAddressBuffer buffer, uint offset) {
  const uint4 value = buffer.Load4(offset);
  Sampler sampler = (Sampler)0;
  sampler.seed = value.x;
  sampler.fixed_u = asfloat(value.y);
  sampler.fixed_v = asfloat(value.z);
  sampler.fixed_w = asfloat(value.w);
  return sampler;
}

void wavefront_emission_begin(GPUWavefrontResources resources, uint slot, uint material_index, SpectralQuery spect, float2 uv, float geometric_cosine, float3 local_outgoing,
  Sampler sampler) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_request_buffer);
  const uint offset = wavefront_emission_request_offset(slot, resources.emission_request_stride);
  wavefront_emission_store_sampler(buffer, offset + kGPUEmissionRequestSamplerOffset, sampler);
  buffer.Store(offset + kGPUEmissionRequestMaterialOffset, material_index);
  buffer.Store2(offset + kGPUEmissionRequestSpectrumOffset, uint2(asuint(spect.wavelength), spect.flags));
  buffer.Store2(offset + kGPUEmissionRequestUVOffset, asuint(uv));
  buffer.Store(offset + kGPUEmissionRequestGeometricCosineOffset, asuint(geometric_cosine));
  buffer.Store3(offset + kGPUEmissionRequestOutgoingOffset, asuint(local_outgoing));
  buffer.Store(offset + kGPUEmissionRequestOperationOffset, uint(GPUEmissionRequestOperation::Classify));
  buffer.Store(offset + kGPUEmissionRequestActiveOffset, 1u);
  uint unused;
  buffer.InterlockedAdd(kGPUEmissionRequestPendingOffset, 1u, unused);
}

void wavefront_emission_complete(GPUWavefrontResources resources, uint slot, out Sampler sampler, out SpectralResponse radiance) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_request_buffer);
  const uint offset = wavefront_emission_request_offset(slot, resources.emission_request_stride);
  sampler = wavefront_emission_load_sampler(buffer, offset + kGPUEmissionRequestSamplerOffset);
  radiance = wavefront_emission_load_response(buffer, offset + kGPUEmissionRequestResultOffset);
}
