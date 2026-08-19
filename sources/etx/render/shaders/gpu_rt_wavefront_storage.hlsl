#include "bindless.hlsl"
#include <interop/gpu_rt_shared.hxx>
#include <interop/gpu_wavefront_abi.hxx>
#include <interop/math_shared.hxx>

[[vk::push_constant]] GPURTConstants constants;
#include "gpu_rt_shared.hlsl"

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
  result.instance_index = buffer.Load(base_offset + kGPUWavefrontHitInstanceIndexOffset);
  return result;
}

bool wavefront_path_vertex_descriptor_is_light(uint descriptor_index) {
  if ((constants.wavefront_buffer_index == kInvalidIndex) || (descriptor_index == kInvalidIndex)) {
    return false;
  }
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(constants.wavefront_buffer_index);
  return descriptor_index == buffer.Load(kGPUWavefrontResourcesLightVertexBufferOffset);
}

GPUWavefrontPathVertex wavefront_load_path_vertex(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  GPUWavefrontPathVertex result = (GPUWavefrontPathVertex)0;
  if (wavefront_path_vertex_descriptor_is_light(descriptor_index)) {
    uint base_offset = index * kGPUWavefrontLightPathVertexStride;
    const uint packed_flags = buffer.Load(base_offset + kGPUWavefrontLightPathVertexFlagsOffset);
    result.throughput = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontLightPathVertexThroughputOffset);
    result.inline_medium_extinction = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontLightPathVertexInlineMediumExtinctionOffset);
    result.position = wavefront_load_float3(buffer, base_offset + kGPUWavefrontLightPathVertexPositionOffset);
    result.triangle_index = buffer.Load(base_offset + kGPUWavefrontLightPathVertexTriangleIndexOffset);
    result.normal = wavefront_load_float3(buffer, base_offset + kGPUWavefrontLightPathVertexNormalOffset);
    result.material_index = buffer.Load(base_offset + kGPUWavefrontLightPathVertexMaterialIndexOffset);
    result.geo_normal = wavefront_load_float3(buffer, base_offset + kGPUWavefrontLightPathVertexGeoNormalOffset);
    result.medium_index = buffer.Load(base_offset + kGPUWavefrontLightPathVertexMediumIndexOffset);
    result.w_i = wavefront_load_float3(buffer, base_offset + kGPUWavefrontLightPathVertexWiOffset);
    result.emitter_index = buffer.Load(base_offset + kGPUWavefrontLightPathVertexEmitterIndexOffset);
    result.texcoord = wavefront_load_float2(buffer, base_offset + kGPUWavefrontLightPathVertexTexcoordOffset);
    result.forward_pdf = asfloat(buffer.Load(base_offset + kGPUWavefrontLightPathVertexForwardPdfOffset));
    result.reverse_pdf = asfloat(buffer.Load(base_offset + kGPUWavefrontLightPathVertexReversePdfOffset));
    result.sampled_bsdf_pdf = asfloat(buffer.Load(base_offset + kGPUWavefrontLightPathVertexSampledBsdfPdfOffset));
    result.eta_scale = 1.0f;
    result.path_length = buffer.Load(base_offset + kGPUWavefrontLightPathVertexPathLengthOffset);
    result.pixel_index = 0u;
    result.flags = packed_flags & kGPUWavefrontLightPathVertexFlagsMask;
    result.pdf_from_prev = asfloat(buffer.Load(base_offset + kGPUWavefrontLightPathVertexPdfFromPrevOffset));
    result.pdf_from_next = asfloat(buffer.Load(base_offset + kGPUWavefrontLightPathVertexPdfFromNextOffset));
    result.pdf_accumulated = asfloat(buffer.Load(base_offset + kGPUWavefrontLightPathVertexPdfAccumulatedOffset));
    result.pdf_history = asfloat(buffer.Load(base_offset + kGPUWavefrontLightPathVertexPdfHistoryOffset));
    result.pdf_ratio = asfloat(buffer.Load(base_offset + kGPUWavefrontLightPathVertexPdfRatioOffset));
    result.barycentric = wavefront_load_float2(buffer, base_offset + kGPUWavefrontLightPathVertexBarycentricOffset);
    result.reserved0 = buffer.Load(base_offset + kGPUWavefrontLightPathVertexPreviousVertexIndexOffset);
    result.instance_index = buffer.Load(base_offset + kGPUWavefrontLightPathVertexInstanceIndexOffset);
    result.inline_medium_flags = packed_flags >> kGPUWavefrontLightPathVertexInlineMediumFlagsShift;
    return result;
  }

  uint base_offset = index * kGPUWavefrontPathVertexStride;
  result.throughput = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontPathVertexThroughputOffset);
  result.inline_medium_extinction = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontPathVertexInlineMediumExtinctionOffset);
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
  result.barycentric = wavefront_load_float2(buffer, base_offset + kGPUWavefrontPathVertexBarycentricOffset);
  result.inline_medium_flags = buffer.Load(base_offset + kGPUWavefrontPathVertexInlineMediumFlagsOffset);
  result.instance_index = buffer.Load(base_offset + kGPUWavefrontPathVertexInstanceIndexOffset);
  return result;
}

void wavefront_store_path_vertex(uint descriptor_index, uint index, GPUWavefrontPathVertex vertex) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  if (wavefront_path_vertex_descriptor_is_light(descriptor_index)) {
    uint base_offset = index * kGPUWavefrontLightPathVertexStride;
    const uint packed_flags = (vertex.flags & kGPUWavefrontLightPathVertexFlagsMask) | (vertex.inline_medium_flags << kGPUWavefrontLightPathVertexInlineMediumFlagsShift);
    wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontLightPathVertexThroughputOffset, vertex.throughput);
    wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontLightPathVertexInlineMediumExtinctionOffset, vertex.inline_medium_extinction);
    wavefront_store_float3(buffer, base_offset + kGPUWavefrontLightPathVertexPositionOffset, vertex.position);
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexTriangleIndexOffset, vertex.triangle_index);
    wavefront_store_float3(buffer, base_offset + kGPUWavefrontLightPathVertexNormalOffset, vertex.normal);
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexMaterialIndexOffset, vertex.material_index);
    wavefront_store_float3(buffer, base_offset + kGPUWavefrontLightPathVertexGeoNormalOffset, vertex.geo_normal);
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexMediumIndexOffset, vertex.medium_index);
    wavefront_store_float3(buffer, base_offset + kGPUWavefrontLightPathVertexWiOffset, vertex.w_i);
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexEmitterIndexOffset, vertex.emitter_index);
    wavefront_store_float2(buffer, base_offset + kGPUWavefrontLightPathVertexTexcoordOffset, vertex.texcoord);
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexForwardPdfOffset, asuint(vertex.forward_pdf));
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexReversePdfOffset, asuint(vertex.reverse_pdf));
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexSampledBsdfPdfOffset, asuint(vertex.sampled_bsdf_pdf));
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexPathLengthOffset, vertex.path_length);
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexFlagsOffset, packed_flags);
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexPdfFromPrevOffset, asuint(vertex.pdf_from_prev));
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexPdfFromNextOffset, asuint(vertex.pdf_from_next));
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexPdfAccumulatedOffset, asuint(vertex.pdf_accumulated));
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexPdfHistoryOffset, asuint(vertex.pdf_history));
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexPdfRatioOffset, asuint(vertex.pdf_ratio));
    wavefront_store_float2(buffer, base_offset + kGPUWavefrontLightPathVertexBarycentricOffset, vertex.barycentric);
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexPreviousVertexIndexOffset, vertex.reserved0);
    buffer.Store(base_offset + kGPUWavefrontLightPathVertexInstanceIndexOffset, vertex.instance_index);
    return;
  }

  uint base_offset = index * kGPUWavefrontPathVertexStride;
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontPathVertexThroughputOffset, vertex.throughput);
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontPathVertexInlineMediumExtinctionOffset, vertex.inline_medium_extinction);
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
  wavefront_store_float2(buffer, base_offset + kGPUWavefrontPathVertexBarycentricOffset, vertex.barycentric);
  buffer.Store(base_offset + kGPUWavefrontPathVertexInlineMediumFlagsOffset, vertex.inline_medium_flags);
  buffer.Store(base_offset + kGPUWavefrontPathVertexInstanceIndexOffset, vertex.instance_index);
}

GPUWavefrontPathMeta wavefront_load_path_meta(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontPathMetaStride;
  GPUWavefrontPathMeta result = (GPUWavefrontPathMeta)0;
  result.camera_path_length = buffer.Load(base_offset + kGPUWavefrontPathMetaCameraPathLengthOffset);
  result.light_path_length = buffer.Load(base_offset + kGPUWavefrontPathMetaLightPathLengthOffset);
  result.flags = buffer.Load(base_offset + kGPUWavefrontPathMetaFlagsOffset);
  result.reserved0 = buffer.Load(base_offset + kGPUWavefrontPathMetaReserved0Offset);
  result.camera_mis_history = asfloat(buffer.Load(base_offset + kGPUWavefrontPathMetaCameraMisHistoryOffset));
  result.light_mis_history = asfloat(buffer.Load(base_offset + kGPUWavefrontPathMetaLightMisHistoryOffset));
  result.from_delta = buffer.Load(base_offset + kGPUWavefrontPathMetaFromDeltaOffset);
  result.reserved1 = buffer.Load(base_offset + kGPUWavefrontPathMetaReserved1Offset);
  return result;
}

void wavefront_store_path_meta(uint descriptor_index, uint index, GPUWavefrontPathMeta meta) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontPathMetaStride;
  buffer.Store(base_offset + kGPUWavefrontPathMetaCameraPathLengthOffset, meta.camera_path_length);
  buffer.Store(base_offset + kGPUWavefrontPathMetaLightPathLengthOffset, meta.light_path_length);
  buffer.Store(base_offset + kGPUWavefrontPathMetaFlagsOffset, meta.flags);
  buffer.Store(base_offset + kGPUWavefrontPathMetaReserved0Offset, meta.reserved0);
  buffer.Store(base_offset + kGPUWavefrontPathMetaCameraMisHistoryOffset, asuint(meta.camera_mis_history));
  buffer.Store(base_offset + kGPUWavefrontPathMetaLightMisHistoryOffset, asuint(meta.light_mis_history));
  buffer.Store(base_offset + kGPUWavefrontPathMetaFromDeltaOffset, meta.from_delta);
  buffer.Store(base_offset + kGPUWavefrontPathMetaReserved1Offset, meta.reserved1);
}

GPUWavefrontSubsurfaceState wavefront_load_subsurface_state(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontSubsurfaceStateStride;
  GPUWavefrontSubsurfaceState result = (GPUWavefrontSubsurfaceState)0;
  result.extinction = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontSubsurfaceStateExtinctionOffset);
  result.scattering = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontSubsurfaceStateScatteringOffset);
  result.albedo = wavefront_load_spectral_response(buffer, base_offset + kGPUWavefrontSubsurfaceStateAlbedoOffset);
  result.material_index = buffer.Load(base_offset + kGPUWavefrontSubsurfaceStateMaterialIndexOffset);
  result.medium_index = buffer.Load(base_offset + kGPUWavefrontSubsurfaceStateMediumIndexOffset);
  result.scatter_material_index = buffer.Load(base_offset + kGPUWavefrontSubsurfaceStateScatterMaterialIndexOffset);
  result.flags = buffer.Load(base_offset + kGPUWavefrontSubsurfaceStateFlagsOffset);
  result.phase_function_g = asfloat(buffer.Load(base_offset + kGPUWavefrontSubsurfaceStatePhaseFunctionGOffset));
  return result;
}

void wavefront_store_subsurface_state(uint descriptor_index, uint index, GPUWavefrontSubsurfaceState state) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontSubsurfaceStateStride;
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontSubsurfaceStateExtinctionOffset, state.extinction);
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontSubsurfaceStateScatteringOffset, state.scattering);
  wavefront_store_spectral_response(buffer, base_offset + kGPUWavefrontSubsurfaceStateAlbedoOffset, state.albedo);
  buffer.Store(base_offset + kGPUWavefrontSubsurfaceStateMaterialIndexOffset, state.material_index);
  buffer.Store(base_offset + kGPUWavefrontSubsurfaceStateMediumIndexOffset, state.medium_index);
  buffer.Store(base_offset + kGPUWavefrontSubsurfaceStateScatterMaterialIndexOffset, state.scatter_material_index);
  buffer.Store(base_offset + kGPUWavefrontSubsurfaceStateFlagsOffset, state.flags);
  buffer.Store(base_offset + kGPUWavefrontSubsurfaceStatePhaseFunctionGOffset, asuint(state.phase_function_g));
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
  result.camera_subsurface_state_buffer = buffer.Load(kGPUWavefrontResourcesCameraSubsurfaceStateBufferOffset);
  result.light_subsurface_state_buffer = buffer.Load(kGPUWavefrontResourcesLightSubsurfaceStateBufferOffset);
  result.path_capacity = buffer.Load(kGPUWavefrontResourcesPathCapacityOffset);
  result.max_path_length = buffer.Load(kGPUWavefrontResourcesMaxPathLengthOffset);
  result.camera_vertex_capacity = buffer.Load(kGPUWavefrontResourcesCameraVertexCapacityOffset);
  result.light_vertex_capacity = buffer.Load(kGPUWavefrontResourcesLightVertexCapacityOffset);
  result.camera_fixed_max_bounces = buffer.Load(kGPUWavefrontResourcesCameraFixedMaxBouncesOffset);
  result.light_fixed_max_bounces = buffer.Load(kGPUWavefrontResourcesLightFixedMaxBouncesOffset);
  result.dispatch_args_buffer = buffer.Load(kGPUWavefrontResourcesDispatchArgsBufferOffset);
  result.material_queue_buffer = buffer.Load(kGPUWavefrontResourcesMaterialQueueBufferOffset);
  result.shadow_queue_buffer = buffer.Load(kGPUWavefrontResourcesShadowQueueBufferOffset);
  result.light_vertex_counter_buffer = buffer.Load(kGPUWavefrontResourcesLightVertexCounterBufferOffset);
  result.fast_light_endpoint_buffer = buffer.Load(kGPUWavefrontResourcesFastLightEndpointBufferOffset);
  return result;
}

uint wavefront_subsurface_state_buffer(GPUWavefrontResources resources, bool from_camera) {
  return from_camera ? resources.camera_subsurface_state_buffer : resources.light_subsurface_state_buffer;
}

uint wavefront_queue_current_descriptor(bool from_camera) {
  GPUWavefrontResources resources = wavefront_load_resources();
  bool even_iteration = (constants.path_iteration & 1u) == 0u;
  if (from_camera) {
    return even_iteration ? resources.camera_queue_a_buffer : resources.camera_queue_b_buffer;
  }
  return even_iteration ? resources.light_queue_a_buffer : resources.light_queue_b_buffer;
}

uint wavefront_queue_count(uint descriptor_index) {
  if (descriptor_index == kInvalidIndex) {
    return 0u;
  }
  return WAVEFRONT_RO_BUFFER(descriptor_index).Load(kGPUWavefrontQueueCountOffset);
}

uint wavefront_queue_load(uint descriptor_index, uint slot) {
  return WAVEFRONT_RO_BUFFER(descriptor_index).Load(kGPUWavefrontQueueIndicesOffset + slot * 4u);
}

#include "gpu_rt_wavefront_work_queue.hlsl"

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

bool wavefront_path_vertex_mis_connectible(GPUWavefrontPathVertex vertex) {
  return (vertex.flags & GPUWavefrontVertexFlags::Mis_connectible) != 0u;
}

bool wavefront_path_vertex_is_surface(GPUWavefrontPathVertex vertex) {
  return (vertex.flags & GPUWavefrontVertexFlags::Surface) != 0u;
}

uint wavefront_camera_fixed_max_bounces(GPUWavefrontResources resources) {
  return resources.camera_fixed_max_bounces;
}

uint wavefront_light_fixed_max_bounces(GPUWavefrontResources resources) {
  return resources.light_fixed_max_bounces;
}

uint wavefront_vertex_slot_from_limit(uint path_index, uint path_length, uint fixed_max_bounces) {
  uint vertex_index = 0u;
  if (fixed_max_bounces == 0u) {
    vertex_index = 0u;
  } else if (fixed_max_bounces == 1u) {
    vertex_index = path_length & 1u;
  } else if (fixed_max_bounces == 2u) {
    vertex_index = (path_length == 0u) ? 0u : (1u + ((path_length - 1u) & 1u));
  } else if (fixed_max_bounces == 3u) {
    vertex_index = (path_length <= 1u) ? path_length : (2u + ((path_length - 2u) & 1u));
  } else {
    vertex_index = min(path_length, fixed_max_bounces);
  }
  return path_index * (fixed_max_bounces + 1u) + vertex_index;
}

uint wavefront_camera_vertex_slot(uint path_index, uint path_length) {
  GPUWavefrontResources resources = wavefront_load_resources();
  return wavefront_vertex_slot_from_limit(path_index, path_length, wavefront_camera_fixed_max_bounces(resources));
}

uint wavefront_light_previous_vertex_index(GPUWavefrontResources resources, uint vertex_index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(resources.light_vertex_buffer);
  return buffer.Load(vertex_index * kGPUWavefrontLightPathVertexStride + kGPUWavefrontLightPathVertexPreviousVertexIndexOffset);
}

uint wavefront_light_vertex_slot(uint path_index, uint path_length) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.light_vertex_counter_buffer != kInvalidIndex) {
    if (path_length == 0u) {
      return path_index;
    }
    GPUWavefrontPathMeta meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
    uint vertex_index = meta.reserved0;
    uint vertex_path_length = meta.light_path_length;
    while ((vertex_index != kInvalidIndex) && (vertex_path_length > path_length)) {
      vertex_index = wavefront_light_previous_vertex_index(resources, vertex_index);
      vertex_path_length -= 1u;
    }
    return vertex_index;
  }
  uint fixed_max_bounces = wavefront_light_fixed_max_bounces(resources);
  if (fixed_max_bounces <= 3u) {
    return wavefront_vertex_slot_from_limit(path_index, path_length, fixed_max_bounces);
  }
  return min(path_length, fixed_max_bounces) * resources.path_capacity + path_index;
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
  const GPUSceneInstanceData instance = load_scene_instance(hit.instance_index);
  const float orientation = (instance.flags & 1u) != 0u ? -1.0f : 1.0f;
  p0 = scene_instance_transform_point(instance, p0);
  p1 = scene_instance_transform_point(instance, p1);
  p2 = scene_instance_transform_point(instance, p2);
  n0 = scene_instance_transform_normal(instance, n0) * orientation;
  n1 = scene_instance_transform_normal(instance, n1) * orientation;
  n2 = scene_instance_transform_normal(instance, n2) * orientation;
  const float3 geo_normal = scene_instance_transform_geometric_normal(instance, tri.geo_n);
  return scene_math_shared_shading_pos(p0, p1, p2, n0, n1, n2, geo_normal, bc, outgoing_direction);
}
