#pragma once

#include <interop/gpu_emission_consumer_abi.hxx>
#include "gpu_rt_wavefront_emission_storage.hlsl"

struct WavefrontEmissionInput {
  uint material_index;
  float2 uv;
  float geometric_cosine;
  float3 local_outgoing;
};

WavefrontEmissionInput wavefront_empty_emission_input() {
  WavefrontEmissionInput result = (WavefrontEmissionInput)0;
  result.material_index = kInvalidIndex;
  return result;
}

void wavefront_queue_emission_input(GPUWavefrontResources resources, uint slot, WavefrontEmissionInput input_value, SpectralQuery spect, uint seed) {
  const Sampler sampler = make_bsdf_sampler(seed);
  wavefront_emission_begin(resources, slot, input_value.material_index, spect, input_value.uv, input_value.geometric_cosine, input_value.local_outgoing, sampler);
}

SpectralResponse wavefront_completed_emission(GPUWavefrontResources resources, uint slot, out uint seed) {
  Sampler sampler = (Sampler)0;
  SpectralResponse radiance = (SpectralResponse)0;
  wavefront_emission_complete(resources, slot, sampler, radiance);
  seed = sampler.seed;
  return radiance;
}

struct WavefrontEmissionConsumer {
  uint owner_index;
  uint candidate_count;
  uint candidate_index;
  uint sampler_seed;
  float weight_sum;
  float selected_weight;
  uint output_pixel_index;
  uint2 output_pixel;
  SpectralQuery spect;
  SpectralResponse factor;
  WavefrontEmitterSample candidate;
  WavefrontEmitterSample selected;
};

void wavefront_store_emitter_sample(RWByteAddressBuffer buffer, uint offset, WavefrontEmitterSample sample_value) {
  wavefront_store_spectral_response(buffer, offset, sample_value.value);
  buffer.Store4(offset + 32u, uint4(asuint(sample_value.barycentric), asuint(sample_value.pdf_sample)));
  buffer.Store4(offset + 48u, uint4(asuint(sample_value.origin), asuint(sample_value.pdf_area)));
  buffer.Store4(offset + 64u, uint4(asuint(sample_value.normal), asuint(sample_value.pdf_dir)));
  buffer.Store4(offset + 80u, uint4(asuint(sample_value.direction), asuint(sample_value.pdf_dir_out)));
  buffer.Store4(offset + 96u, uint4(asuint(sample_value.image_uv), sample_value.emitter_index, sample_value.triangle_index));
  buffer.Store4(offset + 112u, uint4(sample_value.instance_index, sample_value.medium_index, sample_value.material_index, sample_value.is_delta));
  buffer.Store2(offset + 128u, uint2(sample_value.is_distant, sample_value.is_sample_only));
}

WavefrontEmitterSample wavefront_load_emitter_sample(ByteAddressBuffer buffer, uint offset) {
  WavefrontEmitterSample result = (WavefrontEmitterSample)0;
  result.value = wavefront_load_spectral_response(buffer, offset);
  const uint4 barycentric_pdf = buffer.Load4(offset + 32u);
  result.barycentric = asfloat(barycentric_pdf.xyz);
  result.pdf_sample = asfloat(barycentric_pdf.w);
  const uint4 origin_pdf = buffer.Load4(offset + 48u);
  result.origin = asfloat(origin_pdf.xyz);
  result.pdf_area = asfloat(origin_pdf.w);
  const uint4 normal_pdf = buffer.Load4(offset + 64u);
  result.normal = asfloat(normal_pdf.xyz);
  result.pdf_dir = asfloat(normal_pdf.w);
  const uint4 direction_pdf = buffer.Load4(offset + 80u);
  result.direction = asfloat(direction_pdf.xyz);
  result.pdf_dir_out = asfloat(direction_pdf.w);
  const uint4 image_indices = buffer.Load4(offset + 96u);
  result.image_uv = asfloat(image_indices.xy);
  result.emitter_index = image_indices.z;
  result.triangle_index = image_indices.w;
  const uint4 material_indices = buffer.Load4(offset + 112u);
  result.instance_index = material_indices.x;
  result.medium_index = material_indices.y;
  result.material_index = material_indices.z;
  result.is_delta = material_indices.w;
  const uint2 distant_sample_only = buffer.Load2(offset + 128u);
  result.is_distant = distant_sample_only.x;
  result.is_sample_only = distant_sample_only.y;
  return result;
}

uint wavefront_emission_consumer_offset(uint slot) {
  return kGPUEmissionConsumerHeaderSize + slot * kGPUEmissionConsumerStride;
}

void wavefront_emission_consumer_suspend(GPUWavefrontResources resources, uint slot, WavefrontEmissionConsumer consumer) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_consumer_buffer);
  const uint offset = wavefront_emission_consumer_offset(slot);
  if (buffer.Load(offset) == 0u) {
    uint previous = 0u;
    buffer.InterlockedAdd(kGPUEmissionConsumerPendingOffset, 1u, previous);
  }
  buffer.Store4(offset, uint4(1u, consumer.owner_index, consumer.candidate_count, consumer.candidate_index));
  buffer.Store4(offset + 16u, uint4(consumer.sampler_seed, asuint(consumer.weight_sum), asuint(consumer.selected_weight), consumer.output_pixel_index));
  buffer.Store4(offset + 32u, uint4(consumer.output_pixel, asuint(consumer.spect.wavelength), consumer.spect.flags));
  wavefront_store_spectral_response(buffer, offset + 48u, consumer.factor);
  wavefront_store_emitter_sample(buffer, offset + 80u, consumer.candidate);
  wavefront_store_emitter_sample(buffer, offset + 216u, consumer.selected);
}

bool wavefront_emission_consumer_load(GPUWavefrontResources resources, uint slot, out WavefrontEmissionConsumer consumer) {
  consumer = (WavefrontEmissionConsumer)0;
  if ((resources.emission_consumer_buffer == kInvalidIndex) || (slot >= resources.path_capacity)) {
    return false;
  }
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(resources.emission_consumer_buffer);
  const uint offset = wavefront_emission_consumer_offset(slot);
  const uint4 header = buffer.Load4(offset);
  if (header.x == 0u) {
    return false;
  }
  consumer.owner_index = header.y;
  consumer.candidate_count = header.z;
  consumer.candidate_index = header.w;
  const uint4 reservoir = buffer.Load4(offset + 16u);
  consumer.sampler_seed = reservoir.x;
  consumer.weight_sum = asfloat(reservoir.y);
  consumer.selected_weight = asfloat(reservoir.z);
  consumer.output_pixel_index = reservoir.w;
  const uint4 pixel_spect = buffer.Load4(offset + 32u);
  consumer.output_pixel = pixel_spect.xy;
  consumer.spect.wavelength = asfloat(pixel_spect.z);
  consumer.spect.flags = pixel_spect.w;
  consumer.factor = wavefront_load_spectral_response(buffer, offset + 48u);
  consumer.candidate = wavefront_load_emitter_sample(buffer, offset + 80u);
  consumer.selected = wavefront_load_emitter_sample(buffer, offset + 216u);
  return true;
}

void wavefront_emission_consumer_complete(GPUWavefrontResources resources, uint slot) {
  if (resources.emission_consumer_buffer == kInvalidIndex) {
    return;
  }
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_consumer_buffer);
  const uint offset = wavefront_emission_consumer_offset(slot);
  if (buffer.Load(offset) != 0u) {
    buffer.Store(offset, 0u);
    uint previous = 0u;
    buffer.InterlockedAdd(kGPUEmissionConsumerPendingOffset, 0xffffffffu, previous);
  }
}
