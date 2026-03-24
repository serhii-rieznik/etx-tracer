#pragma once

#include <interop/gpu_rt_shared.hxx>
#include <interop/gpu_wavefront_abi.hxx>

[[vk::push_constant]] GPURTConstants constants;
#include "gpu_rt_shared.hlsl"

struct WavefrontEmitterSample {
  SpectralResponse value;
  float3 barycentric;
  float pdf_sample;
  float3 origin;
  float pdf_area;
  float3 normal;
  float pdf_dir;
  float3 direction;
  float pdf_dir_out;
  float2 image_uv;
  uint emitter_index;
  uint triangle_index;
  uint medium_index;
  uint is_delta;
  uint is_distant;
};

struct GPUWavefrontPendingContinuationFlags {
  enum : uint {
    Prepared = 1u << 0u,
    Continue = 1u << 1u,
  };
};

#define WAVEFRONT_RW_BUFFER(descriptor_index) bindless_rw_buffers[NonUniformResourceIndex(descriptor_index)]
#define WAVEFRONT_RO_BUFFER(descriptor_index) bindless_buffers[NonUniformResourceIndex(descriptor_index)]

float2 wavefront_load_float2(ByteAddressBuffer buffer, uint byte_offset) {
  return asfloat(buffer.Load2(byte_offset));
}

float3 wavefront_load_float3(ByteAddressBuffer buffer, uint byte_offset) {
  return asfloat(buffer.Load3(byte_offset));
}

float4 wavefront_load_float4(ByteAddressBuffer buffer, uint byte_offset) {
  return asfloat(buffer.Load4(byte_offset));
}

void wavefront_store_float2(RWByteAddressBuffer buffer, uint byte_offset, float2 value) {
  buffer.Store2(byte_offset, asuint(value));
}

void wavefront_store_float3(RWByteAddressBuffer buffer, uint byte_offset, float3 value) {
  buffer.Store3(byte_offset, asuint(value));
}

void wavefront_store_float4(RWByteAddressBuffer buffer, uint byte_offset, float4 value) {
  buffer.Store4(byte_offset, asuint(value));
}

void wavefront_store_spectral_response(RWByteAddressBuffer buffer, uint byte_offset, SpectralResponse value) {
  wavefront_store_float4(buffer, byte_offset + 0u, float4(value.integrated, value.value));
  buffer.Store(byte_offset + 16u, asuint(value.wavelength));
  buffer.Store(byte_offset + 20u, value.flags);
  buffer.Store(byte_offset + 24u, value.pad0);
  buffer.Store(byte_offset + 28u, value.pad1);
}

SpectralResponse wavefront_load_spectral_response(ByteAddressBuffer buffer, uint byte_offset) {
  SpectralResponse result = (SpectralResponse)0;
  float4 packed_0 = wavefront_load_float4(buffer, byte_offset + 0u);
  result.integrated = packed_0.xyz;
  result.value = packed_0.w;
  result.wavelength = asfloat(buffer.Load(byte_offset + 16u));
  result.flags = buffer.Load(byte_offset + 20u);
  result.pad0 = buffer.Load(byte_offset + 24u);
  result.pad1 = buffer.Load(byte_offset + 28u);
  return result;
}

void wavefront_store_ray(RWByteAddressBuffer buffer, uint byte_offset, Ray value) {
  wavefront_store_float3(buffer, byte_offset + 0u, value.o);
  buffer.Store(byte_offset + 12u, asuint(value.min_t));
  wavefront_store_float3(buffer, byte_offset + 16u, value.d);
  buffer.Store(byte_offset + 28u, asuint(value.max_t));
}

Ray wavefront_load_ray(ByteAddressBuffer buffer, uint byte_offset) {
  Ray result = (Ray)0;
  result.o = wavefront_load_float3(buffer, byte_offset + 0u);
  result.min_t = asfloat(buffer.Load(byte_offset + 12u));
  result.d = wavefront_load_float3(buffer, byte_offset + 16u);
  result.max_t = asfloat(buffer.Load(byte_offset + 28u));
  return result;
}

void wavefront_store_vertex(RWByteAddressBuffer buffer, uint byte_offset, Vertex vertex) {
  wavefront_store_float3(buffer, byte_offset + 0u, vertex.pos);
  wavefront_store_float3(buffer, byte_offset + 12u, vertex.nrm);
  wavefront_store_float3(buffer, byte_offset + 24u, vertex.tan);
  wavefront_store_float3(buffer, byte_offset + 36u, vertex.btn);
  wavefront_store_float2(buffer, byte_offset + 48u, vertex.tex);
}

Vertex wavefront_load_vertex(ByteAddressBuffer buffer, uint byte_offset) {
  Vertex result = (Vertex)0;
  result.pos = wavefront_load_float3(buffer, byte_offset + 0u);
  result.nrm = wavefront_load_float3(buffer, byte_offset + 12u);
  result.tan = wavefront_load_float3(buffer, byte_offset + 24u);
  result.btn = wavefront_load_float3(buffer, byte_offset + 36u);
  result.tex = wavefront_load_float2(buffer, byte_offset + 48u);
  return result;
}

void wavefront_store_path_state(uint descriptor_index, uint index, GPUWavefrontPathState state) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontPathStateStride;
  wavefront_store_ray(buffer, base_offset + kGPUWavefrontPathStateRayOffset, state.ray);
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontPathStateThroughputOffset, state.throughput);
  buffer.Store(base_offset + kGPUWavefrontPathStateEtaOffset, asuint(state.eta));
  buffer.Store(base_offset + kGPUWavefrontPathStateEtaScaleOffset, asuint(state.eta_scale));
  buffer.Store(base_offset + kGPUWavefrontPathStateForwardPdfOffset, asuint(state.forward_pdf));
  buffer.Store(base_offset + kGPUWavefrontPathStateReversePdfOffset, asuint(state.reverse_pdf));
  buffer.Store(base_offset + kGPUWavefrontPathStateSampledBsdfPdfOffset, asuint(state.sampled_bsdf_pdf));
  buffer.Store(base_offset + kGPUWavefrontPathStateLastEmitterPdfOffset, asuint(state.last_emitter_pdf));
  buffer.Store(base_offset + kGPUWavefrontPathStateMediumIndexOffset, state.medium_index);
  buffer.Store(base_offset + kGPUWavefrontPathStatePathLengthOffset, state.path_length);
  buffer.Store(base_offset + kGPUWavefrontPathStatePixelIndexOffset, state.pixel_index);
  buffer.Store(base_offset + kGPUWavefrontPathStateFlagsOffset, state.flags);
  buffer.Store(base_offset + kGPUWavefrontPathStatePathSourceOffset, state.path_source);
  buffer.Store(base_offset + kGPUWavefrontPathStateSamplerSeedOffset, state.sampler_seed);
  buffer.Store2(base_offset + kGPUWavefrontPathStatePixelOffset, state.pixel);
  buffer.Store(base_offset + kGPUWavefrontPathStateSpectralQueryOffset + 0u, asuint(state.spect.wavelength));
  buffer.Store(base_offset + kGPUWavefrontPathStateSpectralQueryOffset + 4u, state.spect.flags);
  wavefront_store_float2(buffer, base_offset + kGPUWavefrontPathStateFilmUvOffset, state.film_uv);
  buffer.Store(base_offset + kGPUWavefrontPathStateLastVertexIndexOffset, state.last_vertex_index);
  buffer.Store(base_offset + kGPUWavefrontPathStateReserved0Offset, state.reserved0);
}

GPUWavefrontPathState wavefront_load_path_state(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontPathStateStride;
  GPUWavefrontPathState result = (GPUWavefrontPathState)0;
  result.ray = wavefront_load_ray(buffer, base_offset + kGPUWavefrontPathStateRayOffset);
  result.throughput = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontPathStateThroughputOffset);
  result.eta = asfloat(buffer.Load(base_offset + kGPUWavefrontPathStateEtaOffset));
  result.eta_scale = asfloat(buffer.Load(base_offset + kGPUWavefrontPathStateEtaScaleOffset));
  result.forward_pdf = asfloat(buffer.Load(base_offset + kGPUWavefrontPathStateForwardPdfOffset));
  result.reverse_pdf = asfloat(buffer.Load(base_offset + kGPUWavefrontPathStateReversePdfOffset));
  result.sampled_bsdf_pdf = asfloat(buffer.Load(base_offset + kGPUWavefrontPathStateSampledBsdfPdfOffset));
  result.last_emitter_pdf = asfloat(buffer.Load(base_offset + kGPUWavefrontPathStateLastEmitterPdfOffset));
  result.medium_index = buffer.Load(base_offset + kGPUWavefrontPathStateMediumIndexOffset);
  result.path_length = buffer.Load(base_offset + kGPUWavefrontPathStatePathLengthOffset);
  result.pixel_index = buffer.Load(base_offset + kGPUWavefrontPathStatePixelIndexOffset);
  result.flags = buffer.Load(base_offset + kGPUWavefrontPathStateFlagsOffset);
  result.path_source = buffer.Load(base_offset + kGPUWavefrontPathStatePathSourceOffset);
  result.sampler_seed = buffer.Load(base_offset + kGPUWavefrontPathStateSamplerSeedOffset);
  result.pixel = buffer.Load2(base_offset + kGPUWavefrontPathStatePixelOffset);
  result.spect.wavelength = asfloat(buffer.Load(base_offset + kGPUWavefrontPathStateSpectralQueryOffset + 0u));
  result.spect.flags = buffer.Load(base_offset + kGPUWavefrontPathStateSpectralQueryOffset + 4u);
  result.film_uv = wavefront_load_float2(buffer, base_offset + kGPUWavefrontPathStateFilmUvOffset);
  result.last_vertex_index = buffer.Load(base_offset + kGPUWavefrontPathStateLastVertexIndexOffset);
  result.reserved0 = buffer.Load(base_offset + kGPUWavefrontPathStateReserved0Offset);
  return result;
}

void wavefront_store_hit(uint descriptor_index, uint index, GPUWavefrontHit hit) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontHitStride;
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontHitTransmittanceOffset, hit.transmittance);
  wavefront_store_vertex(buffer, base_offset + kGPUWavefrontHitVertexOffset, hit.vertex);
  wavefront_store_float3(buffer, base_offset + kGPUWavefrontHitGeoNormalOffset, hit.geo_normal);
  buffer.Store(base_offset + kGPUWavefrontHitHitTOffset, asuint(hit.hit_t));
  buffer.Store(base_offset + kGPUWavefrontHitTriangleIndexOffset, hit.triangle_index);
  buffer.Store(base_offset + kGPUWavefrontHitMaterialIndexOffset, hit.material_index);
  buffer.Store(base_offset + kGPUWavefrontHitEmitterIndexOffset, hit.emitter_index);
  buffer.Store(base_offset + kGPUWavefrontHitMediumIndexOffset, hit.medium_index);
  buffer.Store(base_offset + kGPUWavefrontHitFlagsOffset, hit.flags);
  wavefront_store_float2(buffer, base_offset + kGPUWavefrontHitBarycentricOffset, hit.barycentric);
}

GPUWavefrontHit wavefront_load_hit(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontHitStride;
  GPUWavefrontHit result = (GPUWavefrontHit)0;
  result.transmittance = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontHitTransmittanceOffset);
  result.vertex = wavefront_load_vertex(buffer, base_offset + kGPUWavefrontHitVertexOffset);
  result.geo_normal = wavefront_load_float3(buffer, base_offset + kGPUWavefrontHitGeoNormalOffset);
  result.hit_t = asfloat(buffer.Load(base_offset + kGPUWavefrontHitHitTOffset));
  result.triangle_index = buffer.Load(base_offset + kGPUWavefrontHitTriangleIndexOffset);
  result.material_index = buffer.Load(base_offset + kGPUWavefrontHitMaterialIndexOffset);
  result.emitter_index = buffer.Load(base_offset + kGPUWavefrontHitEmitterIndexOffset);
  result.medium_index = buffer.Load(base_offset + kGPUWavefrontHitMediumIndexOffset);
  result.flags = buffer.Load(base_offset + kGPUWavefrontHitFlagsOffset);
  result.barycentric = wavefront_load_float2(buffer, base_offset + kGPUWavefrontHitBarycentricOffset);
  return result;
}

void wavefront_store_path_vertex(uint descriptor_index, uint index, GPUWavefrontPathVertex vertex) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontPathVertexStride;
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontPathVertexThroughputOffset, vertex.throughput);
  wavefront_store_float3(buffer, base_offset + kGPUWavefrontPathVertexPositionOffset, vertex.position);
  buffer.Store(base_offset + kGPUWavefrontPathVertexTriangleIndexOffset, vertex.triangle_index);
  wavefront_store_float3(buffer, base_offset + kGPUWavefrontPathVertexNormalOffset, vertex.normal);
  buffer.Store(base_offset + kGPUWavefrontPathVertexMaterialIndexOffset, vertex.material_index);
  wavefront_store_float3(buffer, base_offset + kGPUWavefrontPathVertexGeoNormalOffset, vertex.geo_normal);
  buffer.Store(base_offset + kGPUWavefrontPathVertexMediumIndexOffset, vertex.medium_index);
  wavefront_store_float3(buffer, base_offset + kGPUWavefrontPathVertexWiOffset, vertex.w_i);
  buffer.Store(base_offset + kGPUWavefrontPathVertexEmitterIndexOffset, vertex.emitter_index);
  wavefront_store_float2(buffer, base_offset + kGPUWavefrontPathVertexTexcoordOffset, vertex.texcoord);
  buffer.Store(base_offset + kGPUWavefrontPathVertexForwardPdfOffset, asuint(vertex.forward_pdf));
  buffer.Store(base_offset + kGPUWavefrontPathVertexReversePdfOffset, asuint(vertex.reverse_pdf));
  buffer.Store(base_offset + kGPUWavefrontPathVertexSampledBsdfPdfOffset, asuint(vertex.sampled_bsdf_pdf));
  buffer.Store(base_offset + kGPUWavefrontPathVertexEtaScaleOffset, asuint(vertex.eta_scale));
  buffer.Store(base_offset + kGPUWavefrontPathVertexPathLengthOffset, vertex.path_length);
  buffer.Store(base_offset + kGPUWavefrontPathVertexPixelIndexOffset, vertex.pixel_index);
  buffer.Store(base_offset + kGPUWavefrontPathVertexFlagsOffset, vertex.flags);
  buffer.Store(base_offset + kGPUWavefrontPathVertexPdfFromPrevOffset, asuint(vertex.pdf_from_prev));
  buffer.Store(base_offset + kGPUWavefrontPathVertexPdfFromNextOffset, asuint(vertex.pdf_from_next));
  buffer.Store(base_offset + kGPUWavefrontPathVertexPdfAccumulatedOffset, asuint(vertex.pdf_accumulated));
  buffer.Store(base_offset + kGPUWavefrontPathVertexPdfHistoryOffset, asuint(vertex.pdf_history));
  buffer.Store(base_offset + kGPUWavefrontPathVertexPdfRatioOffset, asuint(vertex.pdf_ratio));
}

GPUWavefrontPathVertex wavefront_load_path_vertex(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontPathVertexStride;
  GPUWavefrontPathVertex result = (GPUWavefrontPathVertex)0;
  result.throughput = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontPathVertexThroughputOffset);
  result.position = wavefront_load_float3(buffer, base_offset + kGPUWavefrontPathVertexPositionOffset);
  result.triangle_index = buffer.Load(base_offset + kGPUWavefrontPathVertexTriangleIndexOffset);
  result.normal = wavefront_load_float3(buffer, base_offset + kGPUWavefrontPathVertexNormalOffset);
  result.material_index = buffer.Load(base_offset + kGPUWavefrontPathVertexMaterialIndexOffset);
  result.geo_normal = wavefront_load_float3(buffer, base_offset + kGPUWavefrontPathVertexGeoNormalOffset);
  result.medium_index = buffer.Load(base_offset + kGPUWavefrontPathVertexMediumIndexOffset);
  result.w_i = wavefront_load_float3(buffer, base_offset + kGPUWavefrontPathVertexWiOffset);
  result.emitter_index = buffer.Load(base_offset + kGPUWavefrontPathVertexEmitterIndexOffset);
  result.texcoord = wavefront_load_float2(buffer, base_offset + kGPUWavefrontPathVertexTexcoordOffset);
  result.forward_pdf = asfloat(buffer.Load(base_offset + kGPUWavefrontPathVertexForwardPdfOffset));
  result.reverse_pdf = asfloat(buffer.Load(base_offset + kGPUWavefrontPathVertexReversePdfOffset));
  result.sampled_bsdf_pdf = asfloat(buffer.Load(base_offset + kGPUWavefrontPathVertexSampledBsdfPdfOffset));
  result.eta_scale = asfloat(buffer.Load(base_offset + kGPUWavefrontPathVertexEtaScaleOffset));
  result.path_length = buffer.Load(base_offset + kGPUWavefrontPathVertexPathLengthOffset);
  result.pixel_index = buffer.Load(base_offset + kGPUWavefrontPathVertexPixelIndexOffset);
  result.flags = buffer.Load(base_offset + kGPUWavefrontPathVertexFlagsOffset);
  result.pdf_from_prev = asfloat(buffer.Load(base_offset + kGPUWavefrontPathVertexPdfFromPrevOffset));
  result.pdf_from_next = asfloat(buffer.Load(base_offset + kGPUWavefrontPathVertexPdfFromNextOffset));
  result.pdf_accumulated = asfloat(buffer.Load(base_offset + kGPUWavefrontPathVertexPdfAccumulatedOffset));
  result.pdf_history = asfloat(buffer.Load(base_offset + kGPUWavefrontPathVertexPdfHistoryOffset));
  result.pdf_ratio = asfloat(buffer.Load(base_offset + kGPUWavefrontPathVertexPdfRatioOffset));
  return result;
}

void wavefront_store_path_meta(uint descriptor_index, uint index, GPUWavefrontPathMeta meta) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontPathMetaStride;
  buffer.Store(base_offset + kGPUWavefrontPathMetaCameraPathLengthOffset, meta.camera_path_length);
  buffer.Store(base_offset + kGPUWavefrontPathMetaLightPathLengthOffset, meta.light_path_length);
  buffer.Store(base_offset + kGPUWavefrontPathMetaFlagsOffset, meta.flags);
  buffer.Store(base_offset + kGPUWavefrontPathMetaCameraMisHistoryOffset, asuint(meta.camera_mis_history));
  buffer.Store(base_offset + kGPUWavefrontPathMetaLightMisHistoryOffset, asuint(meta.light_mis_history));
  buffer.Store(base_offset + kGPUWavefrontPathMetaFromDeltaOffset, meta.from_delta);
}

GPUWavefrontPathMeta wavefront_load_path_meta(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontPathMetaStride;
  GPUWavefrontPathMeta result = (GPUWavefrontPathMeta)0;
  result.camera_path_length = buffer.Load(base_offset + kGPUWavefrontPathMetaCameraPathLengthOffset);
  result.light_path_length = buffer.Load(base_offset + kGPUWavefrontPathMetaLightPathLengthOffset);
  result.flags = buffer.Load(base_offset + kGPUWavefrontPathMetaFlagsOffset);
  result.camera_mis_history = asfloat(buffer.Load(base_offset + kGPUWavefrontPathMetaCameraMisHistoryOffset));
  result.light_mis_history = asfloat(buffer.Load(base_offset + kGPUWavefrontPathMetaLightMisHistoryOffset));
  result.from_delta = buffer.Load(base_offset + kGPUWavefrontPathMetaFromDeltaOffset);
  return result;
}

void wavefront_store_direct_light_sample(uint descriptor_index, uint index, GPUWavefrontDirectLightSample sample_value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontDirectLightSampleStride;
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontDirectLightSampleValueOffset, sample_value.value);
  wavefront_store_float3(buffer, base_offset + kGPUWavefrontDirectLightSampleOriginOffset, sample_value.origin);
  buffer.Store(base_offset + kGPUWavefrontDirectLightSamplePdfSampleOffset, asuint(sample_value.pdf_sample));
  wavefront_store_float3(buffer, base_offset + kGPUWavefrontDirectLightSampleDirectionOffset, sample_value.direction);
  buffer.Store(base_offset + kGPUWavefrontDirectLightSamplePdfAreaOffset, asuint(sample_value.pdf_area));
  wavefront_store_float3(buffer, base_offset + kGPUWavefrontDirectLightSampleNormalOffset, sample_value.normal);
  buffer.Store(base_offset + kGPUWavefrontDirectLightSamplePdfDirOffset, asuint(sample_value.pdf_dir));
  wavefront_store_float2(buffer, base_offset + kGPUWavefrontDirectLightSampleTexcoordOffset, sample_value.texcoord);
  buffer.Store(base_offset + kGPUWavefrontDirectLightSampleEmitterIndexOffset, sample_value.emitter_index);
  buffer.Store(base_offset + kGPUWavefrontDirectLightSampleTriangleIndexOffset, sample_value.triangle_index);
  buffer.Store(base_offset + kGPUWavefrontDirectLightSampleFlagsOffset, sample_value.flags);
}

GPUWavefrontDirectLightSample wavefront_load_direct_light_sample(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontDirectLightSampleStride;
  GPUWavefrontDirectLightSample sample_value = (GPUWavefrontDirectLightSample)0;
  sample_value.value = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontDirectLightSampleValueOffset);
  sample_value.origin = wavefront_load_float3(buffer, base_offset + kGPUWavefrontDirectLightSampleOriginOffset);
  sample_value.pdf_sample = asfloat(buffer.Load(base_offset + kGPUWavefrontDirectLightSamplePdfSampleOffset));
  sample_value.direction = wavefront_load_float3(buffer, base_offset + kGPUWavefrontDirectLightSampleDirectionOffset);
  sample_value.pdf_area = asfloat(buffer.Load(base_offset + kGPUWavefrontDirectLightSamplePdfAreaOffset));
  sample_value.normal = wavefront_load_float3(buffer, base_offset + kGPUWavefrontDirectLightSampleNormalOffset);
  sample_value.pdf_dir = asfloat(buffer.Load(base_offset + kGPUWavefrontDirectLightSamplePdfDirOffset));
  sample_value.texcoord = wavefront_load_float2(buffer, base_offset + kGPUWavefrontDirectLightSampleTexcoordOffset);
  sample_value.emitter_index = buffer.Load(base_offset + kGPUWavefrontDirectLightSampleEmitterIndexOffset);
  sample_value.triangle_index = buffer.Load(base_offset + kGPUWavefrontDirectLightSampleTriangleIndexOffset);
  sample_value.flags = buffer.Load(base_offset + kGPUWavefrontDirectLightSampleFlagsOffset);
  return sample_value;
}

void wavefront_store_direct_light_task(uint descriptor_index, uint index, GPUWavefrontDirectLightTask task) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontDirectLightTaskStride;
  wavefront_store_ray(buffer, base_offset + kGPUWavefrontDirectLightTaskShadowRayOffset, task.shadow_ray);
  wavefront_store_float3(buffer, base_offset + kGPUWavefrontDirectLightTaskShadowTargetOffset, task.shadow_target);
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontDirectLightTaskContributionOffset, task.contribution);
  buffer.Store(base_offset + kGPUWavefrontDirectLightTaskMisWeightOffset, asuint(task.mis_weight));
  buffer.Store(base_offset + kGPUWavefrontDirectLightTaskPixelIndexOffset, task.pixel_index);
  buffer.Store(base_offset + kGPUWavefrontDirectLightTaskMediumIndexOffset, task.medium_index);
  buffer.Store(base_offset + kGPUWavefrontDirectLightTaskFlagsOffset, task.flags);
  buffer.Store(base_offset + kGPUWavefrontDirectLightTaskPathIndexOffset, task.path_index);
  buffer.Store(base_offset + kGPUWavefrontDirectLightTaskSamplerSeedOffset, task.sampler_seed);
}

GPUWavefrontDirectLightTask wavefront_load_direct_light_task(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontDirectLightTaskStride;
  GPUWavefrontDirectLightTask result = (GPUWavefrontDirectLightTask)0;
  result.shadow_ray = wavefront_load_ray(buffer, base_offset + kGPUWavefrontDirectLightTaskShadowRayOffset);
  result.shadow_target = wavefront_load_float3(buffer, base_offset + kGPUWavefrontDirectLightTaskShadowTargetOffset);
  result.contribution = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontDirectLightTaskContributionOffset);
  result.mis_weight = asfloat(buffer.Load(base_offset + kGPUWavefrontDirectLightTaskMisWeightOffset));
  result.pixel_index = buffer.Load(base_offset + kGPUWavefrontDirectLightTaskPixelIndexOffset);
  result.medium_index = buffer.Load(base_offset + kGPUWavefrontDirectLightTaskMediumIndexOffset);
  result.flags = buffer.Load(base_offset + kGPUWavefrontDirectLightTaskFlagsOffset);
  result.path_index = buffer.Load(base_offset + kGPUWavefrontDirectLightTaskPathIndexOffset);
  result.sampler_seed = buffer.Load(base_offset + kGPUWavefrontDirectLightTaskSamplerSeedOffset);
  return result;
}

void wavefront_store_direct_light_result(uint descriptor_index, uint index, GPUWavefrontDirectLightResult result_value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontDirectLightResultStride;
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontDirectLightResultTransmittanceOffset, result_value.transmittance);
  buffer.Store(base_offset + kGPUWavefrontDirectLightResultVisibleOffset, result_value.visible);
}

GPUWavefrontDirectLightResult wavefront_load_direct_light_result(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontDirectLightResultStride;
  GPUWavefrontDirectLightResult result_value = (GPUWavefrontDirectLightResult)0;
  result_value.transmittance = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontDirectLightResultTransmittanceOffset);
  result_value.visible = buffer.Load(base_offset + kGPUWavefrontDirectLightResultVisibleOffset);
  return result_value;
}

void wavefront_store_connect_light_task(uint descriptor_index, uint index, GPUWavefrontConnectLightTask task) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontConnectLightTaskStride;
  wavefront_store_ray(buffer, base_offset + kGPUWavefrontConnectLightTaskShadowRayOffset, task.shadow_ray);
  wavefront_store_float3(buffer, base_offset + kGPUWavefrontConnectLightTaskShadowTargetOffset, task.shadow_target);
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontConnectLightTaskContributionOffset, task.contribution);
  buffer.Store(base_offset + kGPUWavefrontConnectLightTaskMisWeightOffset, asuint(task.mis_weight));
  buffer.Store(base_offset + kGPUWavefrontConnectLightTaskPixelIndexOffset, task.pixel_index);
  buffer.Store(base_offset + kGPUWavefrontConnectLightTaskMediumIndexOffset, task.medium_index);
  buffer.Store(base_offset + kGPUWavefrontConnectLightTaskFlagsOffset, task.flags);
  buffer.Store(base_offset + kGPUWavefrontConnectLightTaskPathIndexOffset, task.path_index);
  buffer.Store(base_offset + kGPUWavefrontConnectLightTaskSamplerSeedOffset, task.sampler_seed);
}

GPUWavefrontConnectLightTask wavefront_load_connect_light_task(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontConnectLightTaskStride;
  GPUWavefrontConnectLightTask result_value = (GPUWavefrontConnectLightTask)0;
  result_value.shadow_ray = wavefront_load_ray(buffer, base_offset + kGPUWavefrontConnectLightTaskShadowRayOffset);
  result_value.shadow_target = wavefront_load_float3(buffer, base_offset + kGPUWavefrontConnectLightTaskShadowTargetOffset);
  result_value.contribution = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontConnectLightTaskContributionOffset);
  result_value.mis_weight = asfloat(buffer.Load(base_offset + kGPUWavefrontConnectLightTaskMisWeightOffset));
  result_value.pixel_index = buffer.Load(base_offset + kGPUWavefrontConnectLightTaskPixelIndexOffset);
  result_value.medium_index = buffer.Load(base_offset + kGPUWavefrontConnectLightTaskMediumIndexOffset);
  result_value.flags = buffer.Load(base_offset + kGPUWavefrontConnectLightTaskFlagsOffset);
  result_value.path_index = buffer.Load(base_offset + kGPUWavefrontConnectLightTaskPathIndexOffset);
  result_value.sampler_seed = buffer.Load(base_offset + kGPUWavefrontConnectLightTaskSamplerSeedOffset);
  return result_value;
}

void wavefront_store_connect_light_result(uint descriptor_index, uint index, GPUWavefrontConnectLightResult result_value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontConnectLightResultStride;
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontConnectLightResultTransmittanceOffset, result_value.transmittance);
  buffer.Store(base_offset + kGPUWavefrontConnectLightResultVisibleOffset, result_value.visible);
}

GPUWavefrontConnectLightResult wavefront_load_connect_light_result(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontConnectLightResultStride;
  GPUWavefrontConnectLightResult result_value = (GPUWavefrontConnectLightResult)0;
  result_value.transmittance = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontConnectLightResultTransmittanceOffset);
  result_value.visible = buffer.Load(base_offset + kGPUWavefrontConnectLightResultVisibleOffset);
  return result_value;
}

void wavefront_store_connect_camera_task(uint descriptor_index, uint index, GPUWavefrontConnectCameraTask task) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontConnectCameraTaskStride;
  wavefront_store_ray(buffer, base_offset + kGPUWavefrontConnectCameraTaskShadowRayOffset, task.shadow_ray);
  wavefront_store_float3(buffer, base_offset + kGPUWavefrontConnectCameraTaskShadowTargetOffset, task.shadow_target);
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontConnectCameraTaskContributionOffset, task.contribution);
  buffer.Store(base_offset + kGPUWavefrontConnectCameraTaskMisWeightOffset, asuint(task.mis_weight));
  buffer.Store(base_offset + kGPUWavefrontConnectCameraTaskPixelIndexOffset, task.pixel_index);
  buffer.Store(base_offset + kGPUWavefrontConnectCameraTaskMediumIndexOffset, task.medium_index);
  buffer.Store(base_offset + kGPUWavefrontConnectCameraTaskFlagsOffset, task.flags);
  buffer.Store(base_offset + kGPUWavefrontConnectCameraTaskPathIndexOffset, task.path_index);
  buffer.Store(base_offset + kGPUWavefrontConnectCameraTaskSamplerSeedOffset, task.sampler_seed);
}

GPUWavefrontConnectCameraTask wavefront_load_connect_camera_task(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontConnectCameraTaskStride;
  GPUWavefrontConnectCameraTask result_value = (GPUWavefrontConnectCameraTask)0;
  result_value.shadow_ray = wavefront_load_ray(buffer, base_offset + kGPUWavefrontConnectCameraTaskShadowRayOffset);
  result_value.shadow_target = wavefront_load_float3(buffer, base_offset + kGPUWavefrontConnectCameraTaskShadowTargetOffset);
  result_value.contribution = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontConnectCameraTaskContributionOffset);
  result_value.mis_weight = asfloat(buffer.Load(base_offset + kGPUWavefrontConnectCameraTaskMisWeightOffset));
  result_value.pixel_index = buffer.Load(base_offset + kGPUWavefrontConnectCameraTaskPixelIndexOffset);
  result_value.medium_index = buffer.Load(base_offset + kGPUWavefrontConnectCameraTaskMediumIndexOffset);
  result_value.flags = buffer.Load(base_offset + kGPUWavefrontConnectCameraTaskFlagsOffset);
  result_value.path_index = buffer.Load(base_offset + kGPUWavefrontConnectCameraTaskPathIndexOffset);
  result_value.sampler_seed = buffer.Load(base_offset + kGPUWavefrontConnectCameraTaskSamplerSeedOffset);
  return result_value;
}

void wavefront_store_connect_camera_result(uint descriptor_index, uint index, GPUWavefrontConnectCameraResult result_value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontConnectCameraResultStride;
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontConnectCameraResultTransmittanceOffset, result_value.transmittance);
  buffer.Store(base_offset + kGPUWavefrontConnectCameraResultVisibleOffset, result_value.visible);
}

GPUWavefrontConnectCameraResult wavefront_load_connect_camera_result(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontConnectCameraResultStride;
  GPUWavefrontConnectCameraResult result_value = (GPUWavefrontConnectCameraResult)0;
  result_value.transmittance = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontConnectCameraResultTransmittanceOffset);
  result_value.visible = buffer.Load(base_offset + kGPUWavefrontConnectCameraResultVisibleOffset);
  return result_value;
}

GPUWavefrontResources wavefront_load_resources() {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(constants.wavefront_buffer_index);
  GPUWavefrontResources result = (GPUWavefrontResources)0;
  result.camera_state_buffer = buffer.Load(kGPUWavefrontResourcesCameraStateBufferOffset);
  result.light_state_buffer = buffer.Load(kGPUWavefrontResourcesLightStateBufferOffset);
  result.camera_hit_buffer = buffer.Load(kGPUWavefrontResourcesCameraHitBufferOffset);
  result.light_hit_buffer = buffer.Load(kGPUWavefrontResourcesLightHitBufferOffset);
  result.camera_queue_a_buffer = buffer.Load(kGPUWavefrontResourcesCameraQueueABufferOffset);
  result.camera_queue_b_buffer = buffer.Load(kGPUWavefrontResourcesCameraQueueBBufferOffset);
  result.light_queue_a_buffer = buffer.Load(kGPUWavefrontResourcesLightQueueABufferOffset);
  result.light_queue_b_buffer = buffer.Load(kGPUWavefrontResourcesLightQueueBBufferOffset);
  result.camera_vertex_buffer = buffer.Load(kGPUWavefrontResourcesCameraVertexBufferOffset);
  result.light_vertex_buffer = buffer.Load(kGPUWavefrontResourcesLightVertexBufferOffset);
  result.film_buffer = buffer.Load(kGPUWavefrontResourcesFilmBufferOffset);
  result.path_meta_buffer = buffer.Load(kGPUWavefrontResourcesPathMetaBufferOffset);
  result.direct_light_sample_buffer = buffer.Load(kGPUWavefrontResourcesDirectLightSampleBufferOffset);
  result.direct_light_task_buffer = buffer.Load(kGPUWavefrontResourcesDirectLightTaskBufferOffset);
  result.direct_light_result_buffer = buffer.Load(kGPUWavefrontResourcesDirectLightResultBufferOffset);
  result.connect_light_task_buffer = buffer.Load(kGPUWavefrontResourcesConnectLightTaskBufferOffset);
  result.connect_light_result_buffer = buffer.Load(kGPUWavefrontResourcesConnectLightResultBufferOffset);
  result.connect_camera_task_buffer = buffer.Load(kGPUWavefrontResourcesConnectCameraTaskBufferOffset);
  result.connect_camera_result_buffer = buffer.Load(kGPUWavefrontResourcesConnectCameraResultBufferOffset);
  result.path_capacity = buffer.Load(kGPUWavefrontResourcesPathCapacityOffset);
  result.max_path_length = buffer.Load(kGPUWavefrontResourcesMaxPathLengthOffset);
  result.vertex_capacity = buffer.Load(kGPUWavefrontResourcesVertexCapacityOffset);
  result.fixed_max_bounces = buffer.Load(kGPUWavefrontResourcesFixedMaxBouncesOffset);
  return result;
}

uint wavefront_queue_current_descriptor(bool from_camera) {
  GPUWavefrontResources resources = wavefront_load_resources();
  bool even_iteration = (constants.path_iteration & 1u) == 0u;
  if (from_camera) {
    return even_iteration ? resources.camera_queue_a_buffer : resources.camera_queue_b_buffer;
  }
  return even_iteration ? resources.light_queue_a_buffer : resources.light_queue_b_buffer;
}

uint wavefront_queue_next_descriptor(bool from_camera) {
  GPUWavefrontResources resources = wavefront_load_resources();
  bool even_iteration = (constants.path_iteration & 1u) == 0u;
  if (from_camera) {
    return even_iteration ? resources.camera_queue_b_buffer : resources.camera_queue_a_buffer;
  }
  return even_iteration ? resources.light_queue_b_buffer : resources.light_queue_a_buffer;
}

uint wavefront_queue_count(uint descriptor_index) {
  if (descriptor_index == kInvalidIndex) {
    return 0u;
  }
  return WAVEFRONT_RO_BUFFER(descriptor_index).Load(0u);
}

void wavefront_queue_reset(uint descriptor_index) {
  if (descriptor_index == kInvalidIndex) {
    return;
  }
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  buffer.Store(0u, 0u);
  buffer.Store(4u, 0u);
  buffer.Store(8u, 0u);
  buffer.Store(12u, 0u);
}

void wavefront_queue_store(uint descriptor_index, uint slot, uint value) {
  WAVEFRONT_RW_BUFFER(descriptor_index).Store(kGPUWavefrontQueueIndicesOffset + slot * 4u, value);
}

uint wavefront_queue_load(uint descriptor_index, uint slot) {
  return WAVEFRONT_RO_BUFFER(descriptor_index).Load(kGPUWavefrontQueueIndicesOffset + slot * 4u);
}

uint wavefront_queue_append(uint descriptor_index, uint value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint slot = 0u;
  buffer.InterlockedAdd(0u, 1u, slot);
  buffer.Store(kGPUWavefrontQueueIndicesOffset + slot * 4u, value);
  return slot;
}

uint2 wavefront_render_window_origin() {
  return uint2(constants.render_window_origin_x, constants.render_window_origin_y);
}

uint2 wavefront_render_window_size() {
  return uint2(constants.render_window_width, constants.render_window_height);
}

bool wavefront_render_window_contains(uint2 local_pixel) {
  return all(local_pixel < wavefront_render_window_size());
}

uint2 wavefront_output_pixel(uint2 local_pixel) {
  return wavefront_render_window_origin() + local_pixel;
}

uint wavefront_render_window_local_index(uint2 local_pixel) {
  uint2 render_window_size = wavefront_render_window_size();
  return local_pixel.x + local_pixel.y * render_window_size.x;
}

float4 wavefront_film_load(uint pixel_index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(wavefront_load_resources().film_buffer);
  return wavefront_load_float4(buffer, pixel_index * 16u);
}

void wavefront_film_store(uint pixel_index, float4 value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(wavefront_load_resources().film_buffer);
  wavefront_store_float4(buffer, pixel_index * 16u, value);
}

void wavefront_film_atomic_add_f32(RWByteAddressBuffer buffer, uint byte_offset, float value) {
  uint expected = buffer.Load(byte_offset);
  while (true) {
    uint original = 0u;
    float accumulated = asfloat(expected) + value;
    buffer.InterlockedCompareExchange(byte_offset, expected, asuint(accumulated), original);
    if (original == expected) {
      return;
    }
    expected = original;
  }
}

void wavefront_film_add(uint pixel_index, float3 value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(wavefront_load_resources().film_buffer);
  uint base_offset = pixel_index * 16u;
  wavefront_film_atomic_add_f32(buffer, base_offset + 0u, value.x);
  wavefront_film_atomic_add_f32(buffer, base_offset + 4u, value.y);
  wavefront_film_atomic_add_f32(buffer, base_offset + 8u, value.z);
  buffer.Store(base_offset + 12u, asuint(1.0f));
}

float wavefront_spectral_weight(SpectralQuery spect) {
  float spectral_pdf = spectral_query_sampling_pdf(spect);
  return (spectral_pdf > 0.0f) ? (1.0f / spectral_pdf) : 0.0f;
}

bool wavefront_path_state_valid(GPUWavefrontPathState state) {
  return (state.flags & GPUWavefrontPathFlags::Valid) != 0u;
}

bool wavefront_hit_valid(GPUWavefrontHit hit) {
  return (hit.flags & GPUWavefrontHitFlags::Valid) != 0u;
}

bool wavefront_hit_is_miss(GPUWavefrontHit hit) {
  return (hit.flags & GPUWavefrontHitFlags::Miss) != 0u;
}

bool wavefront_path_vertex_valid(GPUWavefrontPathVertex vertex) {
  return (vertex.flags & GPUWavefrontVertexFlags::Valid) != 0u;
}

bool wavefront_path_vertex_connectible(GPUWavefrontPathVertex vertex) {
  return (vertex.flags & GPUWavefrontVertexFlags::Connectible) != 0u;
}

bool wavefront_path_vertex_is_surface(GPUWavefrontPathVertex vertex) {
  return (vertex.flags & GPUWavefrontVertexFlags::Surface) != 0u;
}

bool wavefront_path_vertex_is_infinite_emitter(GPUWavefrontPathVertex vertex) {
  return ((vertex.flags & GPUWavefrontVertexFlags::Emitter) != 0u) && (vertex.triangle_index == kInvalidIndex);
}

Vertex wavefront_interpolate_vertex(TriangleData tri, float3 barycentrics) {
  ByteAddressBuffer position_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)];
  ByteAddressBuffer normal_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)];
  Vertex result = (Vertex)0;
  float3 p0 = load_float3(position_buffer, tri.i.x);
  float3 p1 = load_float3(position_buffer, tri.i.y);
  float3 p2 = load_float3(position_buffer, tri.i.z);
  float3 n0 = load_float3(normal_buffer, tri.i.x);
  float3 n1 = load_float3(normal_buffer, tri.i.y);
  float3 n2 = load_float3(normal_buffer, tri.i.z);

  float3 tangent_0 = float3(0.0f, 0.0f, 0.0f);
  float3 tangent_1 = float3(0.0f, 0.0f, 0.0f);
  float3 tangent_2 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_0 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_1 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_2 = float3(0.0f, 0.0f, 0.0f);
  float2 texcoord_0 = float2(0.0f, 0.0f);
  float2 texcoord_1 = float2(0.0f, 0.0f);
  float2 texcoord_2 = float2(0.0f, 0.0f);
  bool has_surface_frame = (constants.scene.vertex_tangents != kInvalidIndex) && (constants.scene.vertex_bitangents != kInvalidIndex);
  bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;
  if (has_surface_frame) {
    ByteAddressBuffer tangent_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_tangents)];
    ByteAddressBuffer bitangent_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_bitangents)];
    tangent_0 = load_float3(tangent_buffer, tri.i.x);
    tangent_1 = load_float3(tangent_buffer, tri.i.y);
    tangent_2 = load_float3(tangent_buffer, tri.i.z);
    bitangent_0 = load_float3(bitangent_buffer, tri.i.x);
    bitangent_1 = load_float3(bitangent_buffer, tri.i.y);
    bitangent_2 = load_float3(bitangent_buffer, tri.i.z);
  }
  if (has_texcoords) {
    ByteAddressBuffer texcoord_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)];
    texcoord_0 = load_float2(texcoord_buffer, tri.i.x);
    texcoord_1 = load_float2(texcoord_buffer, tri.i.y);
    texcoord_2 = load_float2(texcoord_buffer, tri.i.z);
  }

  surface_point_shared_interpolate_vertex(p0, p1, p2, n0, n1, n2, tangent_0, tangent_1, tangent_2, bitangent_0, bitangent_1, bitangent_2, texcoord_0, texcoord_1, texcoord_2,
    barycentrics, has_surface_frame, has_texcoords, result);
  return result;
}

float3 wavefront_surface_shading_position(GPUWavefrontHit hit, float3 outgoing_direction) {
  if (hit.triangle_index == kInvalidIndex) {
    float sign_value = (dot(hit.geo_normal, outgoing_direction) >= 0.0f) ? 1.0f : -1.0f;
    return offset_ray(hit.vertex.pos, hit.geo_normal * sign_value);
  }

  TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], hit.triangle_index);
  ByteAddressBuffer position_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)];
  ByteAddressBuffer normal_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)];

  float3 p0 = load_float3(position_buffer, tri.i.x);
  float3 p1 = load_float3(position_buffer, tri.i.y);
  float3 p2 = load_float3(position_buffer, tri.i.z);
  float3 n0 = load_float3(normal_buffer, tri.i.x);
  float3 n1 = load_float3(normal_buffer, tri.i.y);
  float3 n2 = load_float3(normal_buffer, tri.i.z);

  float3 bc = barycentrics(hit.barycentric);
  return scene_math_shared_shading_pos(p0, p1, p2, n0, n1, n2, tri.geo_n, bc, outgoing_direction);
}

uint wavefront_vertex_slot(uint path_index, uint path_length) {
  GPUWavefrontResources resources = wavefront_load_resources();
  uint vertex_index = 0u;
  if (resources.fixed_max_bounces <= 2u) {
    vertex_index = (path_length == 0u) ? 0u : (1u + ((path_length - 1u) & 1u));
  } else {
    vertex_index = min(path_length, resources.fixed_max_bounces);
  }
  return path_index * (resources.fixed_max_bounces + 1u) + vertex_index;
}

float wavefront_safe_div(float a, float b) {
  return (b == 0.0f) ? 0.0f : (a / b);
}

float wavefront_convert_solid_angle_pdf_to_area(float pdf_dir, float3 from_position, float3 to_position, bool target_is_surface, float3 target_geo_normal) {
  if (pdf_dir == 0.0f) {
    return 0.0f;
  }

  float3 w_o = to_position - from_position;
  float d_squared = max(dot(w_o, w_o), kRayEpsilon * kRayEpsilon);
  float inv_d_squared = 1.0f / d_squared;
  w_o *= sqrt(inv_d_squared);
  float cos_t = target_is_surface ? abs(dot(w_o, target_geo_normal)) : 1.0f;
  return cos_t * pdf_dir * inv_d_squared;
}

bool wavefront_camera_ndc_to_pixel_index(Camera camera, float2 ndc_uv, out uint pixel_index) {
  pixel_index = 0u;
  float2 uv = ndc_uv * 0.5f + 0.5f;
  uint2 pixel = uint2(uv * float2(camera.film_size));
  if ((pixel.x >= camera.film_size.x) || (pixel.y >= camera.film_size.y)) {
    return false;
  }

  pixel_index = pixel.x + (camera.film_size.y - 1u - pixel.y) * camera.film_size.x;
  return true;
}

void wavefront_write_root_camera_vertex(uint path_index, Camera camera, Ray ray, SpectralQuery spect, uint pixel_index) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.camera_vertex_buffer == kInvalidIndex) {
    return;
  }

  CameraFilmEvalShared eval = camera_film_shared_evaluate_out(camera, ray);
  GPUWavefrontPathVertex vertex = (GPUWavefrontPathVertex)0;
  vertex.throughput = spectral_response_make(spect, 1.0f);
  vertex.position = ray.o;
  vertex.normal = eval.normal;
  vertex.geo_normal = eval.normal;
  vertex.medium_index = camera.medium_index;
  vertex.w_i = ray.d;
  vertex.forward_pdf = eval.pdf_dir;
  vertex.sampled_bsdf_pdf = eval.pdf_dir;
  vertex.path_length = 0u;
  vertex.pixel_index = pixel_index;
  vertex.flags = GPUWavefrontVertexFlags::Valid | GPUWavefrontVertexFlags::Connectible | GPUWavefrontVertexFlags::Mis_connectible | GPUWavefrontVertexFlags::From_camera |
                 GPUWavefrontVertexFlags::Camera;
  vertex.pdf_from_prev = 1.0f;
  float history_value = scene_path_mode_uses_bdpt_fast() ? 1.0f : 0.0f;
  vertex.pdf_history = history_value;
  vertex.pdf_accumulated = history_value;
  wavefront_store_path_vertex(resources.camera_vertex_buffer, wavefront_vertex_slot(path_index, 0u), vertex);
}

void wavefront_write_root_light_vertex(uint path_index, WavefrontEmitterSample emitter_sample, SpectralQuery spect) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.light_vertex_buffer == kInvalidIndex) {
    return;
  }

  GPUWavefrontPathVertex vertex = (GPUWavefrontPathVertex)0;
  vertex.triangle_index = kInvalidIndex;
  vertex.material_index = kInvalidIndex;
  vertex.throughput = emitter_sample.value;
  vertex.position = emitter_sample.origin;
  vertex.normal = emitter_sample.normal;
  vertex.geo_normal = emitter_sample.normal;
  vertex.medium_index = emitter_sample.medium_index;
  vertex.w_i = emitter_sample.direction;
  vertex.emitter_index = emitter_sample.emitter_index;
  vertex.forward_pdf = emitter_sample.pdf_dir;
  vertex.sampled_bsdf_pdf = emitter_sample.pdf_dir;
  vertex.path_length = 0u;
  vertex.pixel_index = path_index;
  vertex.flags = GPUWavefrontVertexFlags::Valid | GPUWavefrontVertexFlags::Connectible | GPUWavefrontVertexFlags::Emitter | GPUWavefrontVertexFlags::From_light;
  if (emitter_sample.is_distant == 0u) {
    vertex.triangle_index = emitter_sample.triangle_index;
    vertex.texcoord = emitter_sample.image_uv;
    vertex.flags |= GPUWavefrontVertexFlags::Surface;

    GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
    if (try_load_emitter_instance(emitter_sample.emitter_index, emitter_instance) && (emitter_instance.triangle_index != kInvalidIndex)) {
      TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], emitter_instance.triangle_index);
      vertex.material_index = tri.material_index;
      vertex.geo_normal = tri.geo_n;
    }
  }
  if (emitter_sample.is_delta == 0u) {
    vertex.flags |= GPUWavefrontVertexFlags::Mis_connectible;
  } else {
    vertex.flags |= GPUWavefrontVertexFlags::Delta;
  }
  vertex.pdf_from_prev = emitter_sample.pdf_area * emitter_sample.pdf_sample;
  float history_value = scene_path_mode_uses_bdpt_fast() ? 1.0f : 0.0f;
  vertex.pdf_history = history_value;
  vertex.pdf_accumulated = history_value;
  wavefront_store_path_vertex(resources.light_vertex_buffer, wavefront_vertex_slot(path_index, 0u), vertex);
}

void wavefront_write_vertex(bool from_camera, uint path_index, GPUWavefrontPathState state, GPUWavefrontHit hit) {
  GPUWavefrontResources resources = wavefront_load_resources();
  uint vertex_slot = wavefront_vertex_slot(path_index, state.path_length);
  if (vertex_slot >= resources.vertex_capacity) {
    return;
  }

  GPUWavefrontPathVertex vertex = (GPUWavefrontPathVertex)0;
  vertex.throughput = state.throughput;
  vertex.position = hit.vertex.pos;
  vertex.triangle_index = hit.triangle_index;
  vertex.normal = hit.vertex.nrm;
  vertex.material_index = hit.material_index;
  vertex.geo_normal = hit.geo_normal;
  vertex.medium_index = state.medium_index;
  vertex.w_i = state.ray.d;
  vertex.emitter_index = hit.emitter_index;
  vertex.texcoord = hit.vertex.tex;
  vertex.forward_pdf = state.forward_pdf;
  vertex.reverse_pdf = state.reverse_pdf;
  vertex.sampled_bsdf_pdf = state.sampled_bsdf_pdf;
  vertex.eta_scale = state.eta_scale;
  vertex.path_length = state.path_length;
  vertex.pixel_index = state.pixel_index;
  vertex.flags = GPUWavefrontVertexFlags::Valid | GPUWavefrontVertexFlags::Surface | (from_camera ? GPUWavefrontVertexFlags::From_camera : GPUWavefrontVertexFlags::From_light);
  if ((state.flags & GPUWavefrontPathFlags::Connectible) != 0u) {
    vertex.flags |= GPUWavefrontVertexFlags::Connectible | GPUWavefrontVertexFlags::Mis_connectible;
  }
  if ((state.flags & GPUWavefrontPathFlags::Delta) != 0u) {
    vertex.flags |= GPUWavefrontVertexFlags::Delta;
  }

  uint descriptor_index = from_camera ? resources.camera_vertex_buffer : resources.light_vertex_buffer;
  wavefront_store_path_vertex(descriptor_index, vertex_slot, vertex);
}

void wavefront_write_path_meta(bool from_camera, uint path_index, GPUWavefrontPathState state) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.path_meta_buffer == kInvalidIndex) {
    return;
  }

  GPUWavefrontPathMeta meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
  if (from_camera) {
    meta.camera_path_length = state.path_length;
    meta.flags |= GPUWavefrontPathMetaFlags::Camera_active;
  } else {
    meta.light_path_length = state.path_length;
    meta.flags |= GPUWavefrontPathMetaFlags::Light_active;
  }
  wavefront_store_path_meta(resources.path_meta_buffer, path_index, meta);
}

void wavefront_enqueue_next_state(bool from_camera, uint path_index, GPUWavefrontPathState state) {
  GPUWavefrontResources resources = wavefront_load_resources();
  uint state_buffer = from_camera ? resources.camera_state_buffer : resources.light_state_buffer;
  wavefront_store_path_state(state_buffer, path_index, state);
  wavefront_queue_append(wavefront_queue_next_descriptor(from_camera), path_index);
}
