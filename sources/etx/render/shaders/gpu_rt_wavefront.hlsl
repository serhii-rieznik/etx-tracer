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
}

void wavefront_store_path_meta(uint descriptor_index, uint index, GPUWavefrontPathMeta meta) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontPathMetaStride;
  buffer.Store(base_offset + kGPUWavefrontPathMetaCameraPathLengthOffset, meta.camera_path_length);
  buffer.Store(base_offset + kGPUWavefrontPathMetaLightPathLengthOffset, meta.light_path_length);
  buffer.Store(base_offset + kGPUWavefrontPathMetaFlagsOffset, meta.flags);
}

GPUWavefrontPathMeta wavefront_load_path_meta(uint descriptor_index, uint index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(descriptor_index);
  uint base_offset = index * kGPUWavefrontPathMetaStride;
  GPUWavefrontPathMeta result = (GPUWavefrontPathMeta)0;
  result.camera_path_length = buffer.Load(base_offset + kGPUWavefrontPathMetaCameraPathLengthOffset);
  result.light_path_length = buffer.Load(base_offset + kGPUWavefrontPathMetaLightPathLengthOffset);
  result.flags = buffer.Load(base_offset + kGPUWavefrontPathMetaFlagsOffset);
  return result;
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

float4 wavefront_film_load(uint pixel_index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(wavefront_load_resources().film_buffer);
  return wavefront_load_float4(buffer, pixel_index * 16u);
}

void wavefront_film_store(uint pixel_index, float4 value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(wavefront_load_resources().film_buffer);
  wavefront_store_float4(buffer, pixel_index * 16u, value);
}

void wavefront_film_add(uint pixel_index, float3 value) {
  float4 current = wavefront_film_load(pixel_index);
  current.xyz += value;
  current.w = 1.0f;
  wavefront_film_store(pixel_index, current);
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

float wavefront_surface_shading_pdf_environment(float3 direction, bool target_is_surface, float3 target_geo_normal) {
  EmitterAccessGPUContext context = make_scene_emitter_access_gpu_context();
  uint emitter_instance_count = 0u;
  uint environment_count = 0u;
  if (emitter_access_try_load_environment_state(context, emitter_instance_count, environment_count) == false) {
    return 0.0f;
  }
  (void)emitter_instance_count;

  float pdf_dir = 0.0f;
  for (uint i = 0u; i < environment_count; ++i) {
    uint emitter_index = kInvalidIndex;
    if (emitter_access_try_load_environment_emitter(context, i, emitter_index)) {
      pdf_dir += emitter_discrete_pdf(emitter_index);
    }
  }

  if (environment_count == 0u) {
    return 0.0f;
  }

  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  float normal_factor = target_is_surface ? abs(dot(target_geo_normal, direction)) : 1.0f;
  return (normal_factor / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius)) * (pdf_dir / float(environment_count));
}

bool wavefront_sample_emitter_index(uint light_sampling_mode, inout uint seed, out uint emitter_index, out float pdf_sample) {
  emitter_index = kInvalidIndex;
  pdf_sample = 0.0f;
  if ((light_sampling_mode == kSceneLightSamplingFromDistribution) || (light_sampling_mode == kSceneLightSamplingRISFromDistribution)) {
    emitter_index = sample_emitter_distribution(seed, pdf_sample);
    return emitter_index != kInvalidIndex;
  }

  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(scene_globals);
  if (globals_data.emitter_instance_count == 0u) {
    return false;
  }

  emitter_index = min(uint(rnd01(seed) * float(globals_data.emitter_instance_count)), globals_data.emitter_instance_count - 1u);
  pdf_sample = 1.0f / float(globals_data.emitter_instance_count);
  return true;
}

bool wavefront_trace_surface_path_compact(RayDesc ray, SpectralQuery spect, inout uint medium_index, inout uint seed, out TraceSurfaceResult result);

bool wavefront_trace_transmittance_to_point(float3 origin, float3 target, SpectralQuery spect, uint medium_index, inout uint seed, out SpectralResponse transmittance) {
  transmittance = spectral_response_make(spect, 1.0f);
  float3 delta = target - origin;
  float distance = length(delta);
  if (distance <= kRayEpsilon) {
    return true;
  }

  RayDesc ray = (RayDesc)0;
  ray.Origin = origin;
  ray.Direction = delta / distance;
  ray.TMin = kRayEpsilon;
  ray.TMax = max(ray.TMin, distance - kRayEpsilon);

  TraceSurfaceResult trace_result = (TraceSurfaceResult)0;
  uint transmittance_medium = medium_index;
  bool found_surface = wavefront_trace_surface_path_compact(ray, spect, transmittance_medium, seed, trace_result);
  transmittance = trace_result.transmittance;
  return found_surface == false;
}

float3 wavefront_surface_shading_position(GPUWavefrontHit hit, float3 outgoing_direction) {
  float sign_value = (dot(hit.geo_normal, outgoing_direction) >= 0.0f) ? 1.0f : -1.0f;
  return offset_ray(hit.vertex.pos, hit.geo_normal * sign_value);
}

SurfacePoint wavefront_load_surface_point_compact(TriangleData tri, float2 bary, float3 ray_dir) {
  SurfacePoint result = (SurfacePoint)0;
  result.barycentrics = barycentrics(bary);

  float3 position_0 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)], tri.i.x);
  float3 position_1 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)], tri.i.y);
  float3 position_2 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_positions)], tri.i.z);

  float3 normal_0 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)], tri.i.x);
  float3 normal_1 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)], tri.i.y);
  float3 normal_2 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_normals)], tri.i.z);

  bool has_surface_frame = (constants.scene.vertex_tangents != kInvalidIndex) && (constants.scene.vertex_bitangents != kInvalidIndex);
  float3 tangent_0 = float3(0.0f, 0.0f, 0.0f);
  float3 tangent_1 = float3(0.0f, 0.0f, 0.0f);
  float3 tangent_2 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_0 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_1 = float3(0.0f, 0.0f, 0.0f);
  float3 bitangent_2 = float3(0.0f, 0.0f, 0.0f);
  if (has_surface_frame) {
    tangent_0 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_tangents)], tri.i.x);
    tangent_1 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_tangents)], tri.i.y);
    tangent_2 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_tangents)], tri.i.z);
    bitangent_0 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_bitangents)], tri.i.x);
    bitangent_1 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_bitangents)], tri.i.y);
    bitangent_2 = load_float3(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_bitangents)], tri.i.z);
  }

  bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;
  float2 texcoord_0 = float2(0.0f, 0.0f);
  float2 texcoord_1 = float2(0.0f, 0.0f);
  float2 texcoord_2 = float2(0.0f, 0.0f);
  if (has_texcoords) {
    texcoord_0 = load_float2(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri.i.x);
    texcoord_1 = load_float2(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri.i.y);
    texcoord_2 = load_float2(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri.i.z);
  }

  surface_point_shared_interpolate_vertex(position_0, position_1, position_2, normal_0, normal_1, normal_2, tangent_0, tangent_1, tangent_2, bitangent_0, bitangent_1, bitangent_2,
    texcoord_0, texcoord_1, texcoord_2, result.barycentrics, has_surface_frame, has_texcoords, result.vertex);

  result.geo_normal = surface_point_shared_orient_geo_normal(tri.geo_n, ray_dir);
  return result;
}

bool wavefront_trace_surface_path_compact(RayDesc ray, SpectralQuery spect, inout uint medium_index, inout uint seed, out TraceSurfaceResult result) {
  result = (TraceSurfaceResult)0;
  result.medium_index = medium_index;
  result.triangle_index = kInvalidIndex;
  result.emitter_index = kInvalidIndex;
  result.hit_t = ray.TMax;
  result.transmittance = spectral_response_make(spect, 1.0f);

  bool has_geometry_buffers = (constants.scene.triangles != kInvalidIndex) && (constants.scene.vertex_positions != kInvalidIndex) &&
                              (constants.scene.vertex_normals != kInvalidIndex) && (constants.scene.scene_globals != kInvalidIndex);
  if (has_geometry_buffers == false) {
    return false;
  }

  SceneGPUSharedGlobals scene_globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  uint vertex_count = scene_globals_data.vertex_count;
  uint triangle_count = scene_globals_data.triangle_count;
  bool has_material_buffer = constants.scene.materials != kInvalidIndex;
  bool has_texcoords = constants.scene.vertex_texcoords != kInvalidIndex;

  RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> ray_query;
  ray_query.TraceRayInline(bindless_accel_structs[NonUniformResourceIndex(constants.as_index)], RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);

  float medium_segment_start_t = ray.TMin;
  uint ray_medium_index = medium_index;
  while (ray_query.Proceed()) {
    if (ray_query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
      continue;
    }

    uint candidate_triangle_index = ray_query.CandidatePrimitiveIndex();
    if (candidate_triangle_index >= triangle_count) {
      continue;
    }

    float candidate_t = ray_query.CandidateTriangleRayT();
    if (candidate_t > medium_segment_start_t) {
      float segment_distance = candidate_t - medium_segment_start_t;
      float3 segment_origin = ray.Origin + ray.Direction * medium_segment_start_t;
      SpectralResponse segment_transmittance = medium_segment_transmittance_spectral(ray_medium_index, segment_origin, ray.Direction, segment_distance, spect, seed);
      result.transmittance = spectral_response_mul(result.transmittance, segment_transmittance);
      medium_segment_start_t = candidate_t;
    }

    TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], candidate_triangle_index);
    bool valid_indices = (tri.i.x < vertex_count) && (tri.i.y < vertex_count) && (tri.i.z < vertex_count);
    if (valid_indices == false) {
      continue;
    }

    float2 candidate_bary = ray_query.CandidateTriangleBarycentrics();
    float2 candidate_uv = float2(0.0f, 0.0f);
    if (has_texcoords) {
      candidate_uv = interpolate_uv(bindless_buffers[NonUniformResourceIndex(constants.scene.vertex_texcoords)], tri, candidate_bary);
    }

    MaterialAccess material_access = ETX_ZERO(MaterialAccess);
    if (has_material_buffer) {
      MaterialAccessGPUContext material_context = {constants.scene.materials};
      material_access_try_load(material_context, tri.material_index, material_access);
    }

    bool alpha_rejected = alpha_test_pass(tri.material_index, candidate_uv, seed);
    bool entering_surface = dot(tri.geo_n, ray.Direction) < 0.0f;
    HitPolicyDecision hit_policy = hit_policy_evaluate(HitPolicyMode::SkipBoundaryWithMediumTransition, material_access.material_class, alpha_rejected, entering_surface,
      material_access.int_medium_index, material_access.ext_medium_index);
    if (hit_policy.action == HitPolicyAction::Ignore) {
      continue;
    }

    if (hit_policy.action == HitPolicyAction::TransitionMedium) {
      ray_medium_index = hit_policy.medium_index;
      continue;
    }

    if (hit_policy.action == HitPolicyAction::CommitSurface) {
      ray_query.CommitNonOpaqueTriangleHit();
    }
  }

  float medium_segment_end_t = (ray_query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? ray_query.CommittedRayT() : ray.TMax;
  if (medium_segment_end_t > medium_segment_start_t) {
    float segment_distance = medium_segment_end_t - medium_segment_start_t;
    float3 segment_origin = ray.Origin + ray.Direction * medium_segment_start_t;
    SpectralResponse segment_transmittance = medium_segment_transmittance_spectral(ray_medium_index, segment_origin, ray.Direction, segment_distance, spect, seed);
    result.transmittance = spectral_response_mul(result.transmittance, segment_transmittance);
  }

  medium_index = ray_medium_index;
  result.medium_index = ray_medium_index;

  if (ray_query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
    return false;
  }

  result.triangle_index = ray_query.CommittedPrimitiveIndex();
  result.hit_t = ray_query.CommittedRayT();
  result.tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], result.triangle_index);
  result.surface_point = wavefront_load_surface_point_compact(result.tri, ray_query.CommittedTriangleBarycentrics(), ray.Direction);
  result.emitter_index = result.tri.emitter_index;
  try_load_material_full(result.tri.material_index, result.material);
  result.hit = 1u;
  return true;
}

BSDFEval wavefront_evaluate_material_bsdf(BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return bsdf_evaluate(make_scene_bsdf_resource_gpu_context(), data, outgoing_direction, material, sampler);
}

bool wavefront_sample_emitter_to_point(uint light_sampling_mode, SpectralQuery spect, float3 from_point, inout uint seed, out WavefrontEmitterSample sample_value) {
  sample_value = (WavefrontEmitterSample)0;
  uint emitter_index = kInvalidIndex;
  float pdf_sample = 0.0f;
  if (wavefront_sample_emitter_index(light_sampling_mode, seed, emitter_index, pdf_sample) == false) {
    return false;
  }

  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  GPUEmitterProfileABIData emitter_profile = (GPUEmitterProfileABIData)0;
  if ((try_load_emitter_instance(emitter_index, emitter_instance) == false) || (try_load_emitter_profile(emitter_instance.emitter_profile_index, emitter_profile) == false)) {
    return false;
  }

  sample_value.emitter_index = emitter_index;
  sample_value.triangle_index = emitter_instance.triangle_index;
  sample_value.medium_index = kInvalidIndex;
  sample_value.pdf_sample = pdf_sample;

  if (emitter_instance.emitter_class == EmitterClass::Area) {
    TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], emitter_instance.triangle_index);
    sample_value.barycentric = random_barycentric(float2(rnd01(seed), rnd01(seed)));
    Vertex vertex = wavefront_interpolate_vertex(tri, sample_value.barycentric);
    sample_value.origin = vertex.pos;
    sample_value.normal = normalize(vertex.nrm);
    sample_value.direction = normalize(sample_value.origin - from_point);
    sample_value.image_uv = vertex.tex;
    sample_value.pdf_area = (emitter_instance.triangle_area > 0.0f) ? (1.0f / emitter_instance.triangle_area) : 0.0f;
    sample_value.pdf_dir = max(0.0f, dot(sample_value.normal, sample_value.direction)) * kInvPi;
    sample_value.pdf_dir_out = sample_value.pdf_area * sample_value.pdf_dir;
    sample_value.value = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, vertex.tex, spect);
    sample_value.is_delta = 0u;
    sample_value.is_distant = 0u;
    return sample_value.pdf_dir > 0.0f;
  }

  EmitterAccess access = (EmitterAccess)0;
  if (emitter_access_try_load(make_scene_emitter_access_gpu_context(), emitter_index, access) == false) {
    return false;
  }

  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  if (emitter_instance.emitter_class == EmitterClass::Directional) {
    sample_value.direction = normalize(access.emitter_direction);
    sample_value.normal = -sample_value.direction;
    sample_value.origin =
      from_point + sample_value.direction * distance_to_sphere(from_point, sample_value.direction, globals_data.bounding_sphere_center, globals_data.bounding_sphere_radius);
    sample_value.pdf_area = 1.0f / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius);
    sample_value.pdf_dir = 1.0f;
    sample_value.pdf_dir_out = sample_value.pdf_area;
    sample_value.value = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, float2(0.5f, 0.5f), spect);
    sample_value.is_delta = 1u;
    sample_value.is_distant = 1u;
    return true;
  }

  ImageSampleGPUContext image_context = make_image_sample_gpu_context(constants.scene.images);
  float2 sample_rnd = float2(rnd01(seed), rnd01(seed));
  ImageSampleAccess image_sample = image_sample_access_default(sample_rnd);
  if (image_sample_try_sample(image_context, emitter_profile.emission_image_index, sample_rnd, image_sample) == false) {
    return false;
  }

  bool is_atmosphere = (emitter_profile.emitter_profile_meta & EmitterProfileMeta::Atmosphere) != 0u;
  uint projection = projection_environment_mode(is_atmosphere);
  sample_value.image_uv = image_sample.uv;
  float2 image_offset = float2(0.0f, 0.0f);
  float image_u_scale = 1.0f;
  emitter_access_try_load_image_params(make_scene_emitter_access_gpu_context(), emitter_profile.emission_image_index, image_offset, image_u_scale);
  sample_value.direction = uv_to_direction(image_sample.uv, image_offset, image_u_scale, projection);
  sample_value.normal = -sample_value.direction;
  sample_value.origin =
    from_point + sample_value.direction * distance_to_sphere(from_point, sample_value.direction, globals_data.bounding_sphere_center, globals_data.bounding_sphere_radius);
  sample_value.pdf_dir = projection_environment_image_pdf_to_solid_angle(image_sample.pdf, image_sample.uv, projection);
  sample_value.pdf_area = 1.0f / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius);
  sample_value.pdf_dir_out = sample_value.pdf_area * sample_value.pdf_dir;
  sample_value.value = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, image_sample.uv, spect);
  sample_value.is_delta = 0u;
  sample_value.is_distant = 1u;
  return true;
}

bool wavefront_sample_light_emission(SpectralQuery spect, inout uint seed, out WavefrontEmitterSample sample_value) {
  sample_value = (WavefrontEmitterSample)0;
  float pdf_sample = 0.0f;
  uint emitter_index = sample_emitter_distribution(seed, pdf_sample);
  if (emitter_index == kInvalidIndex) {
    return false;
  }

  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  GPUEmitterProfileABIData emitter_profile = (GPUEmitterProfileABIData)0;
  if ((try_load_emitter_instance(emitter_index, emitter_instance) == false) || (try_load_emitter_profile(emitter_instance.emitter_profile_index, emitter_profile) == false)) {
    return false;
  }

  sample_value.emitter_index = emitter_index;
  sample_value.triangle_index = emitter_instance.triangle_index;
  sample_value.medium_index = kInvalidIndex;
  sample_value.pdf_sample = pdf_sample;

  if (emitter_instance.emitter_class == EmitterClass::Area) {
    TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], emitter_instance.triangle_index);
    sample_value.barycentric = random_barycentric(float2(rnd01(seed), rnd01(seed)));
    Vertex vertex = wavefront_interpolate_vertex(tri, sample_value.barycentric);
    sample_value.origin = vertex.pos;
    sample_value.normal = normalize(vertex.nrm);
    sample_value.direction = sample_cosine_distribution(float2(rnd01(seed), rnd01(seed)), sample_value.normal, 1.0f);
    sample_value.image_uv = vertex.tex;
    sample_value.pdf_area = (emitter_instance.triangle_area > 0.0f) ? (1.0f / emitter_instance.triangle_area) : 0.0f;
    sample_value.pdf_dir = max(0.0f, dot(sample_value.normal, sample_value.direction)) * kInvPi;
    sample_value.pdf_dir_out = sample_value.pdf_area * sample_value.pdf_dir;
    sample_value.value = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, vertex.tex, spect);
    sample_value.is_delta = 0u;
    sample_value.is_distant = 0u;
    return sample_value.pdf_dir > 0.0f;
  }

  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)]);
  if (emitter_instance.emitter_class == EmitterClass::Directional) {
    EmitterAccess access = (EmitterAccess)0;
    if (emitter_access_try_load(make_scene_emitter_access_gpu_context(), emitter_index, access) == false) {
      return false;
    }

    float3 direction_to_scene = -normalize(access.emitter_direction);
    OrthonormalBasis basis = orthonormal_basis(direction_to_scene);
    float2 disk_sample = sample_disk(float2(rnd01(seed), rnd01(seed)));
    sample_value.direction = direction_to_scene;
    sample_value.normal = direction_to_scene;
    sample_value.origin = globals_data.bounding_sphere_center + globals_data.bounding_sphere_radius * (disk_sample.x * basis.u + disk_sample.y * basis.v - direction_to_scene);
    sample_value.origin +=
      sample_value.direction * distance_to_sphere(sample_value.origin, sample_value.direction, globals_data.bounding_sphere_center, globals_data.bounding_sphere_radius);
    sample_value.pdf_dir = 1.0f;
    sample_value.pdf_area = 1.0f / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius);
    sample_value.pdf_dir_out = sample_value.pdf_area;
    sample_value.value = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, float2(0.5f, 0.5f), spect);
    sample_value.is_delta = 1u;
    sample_value.is_distant = 1u;
    return true;
  }

  ImageSampleGPUContext image_context = make_image_sample_gpu_context(constants.scene.images);
  float2 sample_rnd = float2(rnd01(seed), rnd01(seed));
  ImageSampleAccess image_sample = image_sample_access_default(sample_rnd);
  if (image_sample_try_sample(image_context, emitter_profile.emission_image_index, sample_rnd, image_sample) == false) {
    return false;
  }

  bool is_atmosphere = (emitter_profile.emitter_profile_meta & EmitterProfileMeta::Atmosphere) != 0u;
  uint projection = projection_environment_mode(is_atmosphere);
  sample_value.image_uv = image_sample.uv;
  float2 image_offset = float2(0.0f, 0.0f);
  float image_u_scale = 1.0f;
  emitter_access_try_load_image_params(make_scene_emitter_access_gpu_context(), emitter_profile.emission_image_index, image_offset, image_u_scale);
  sample_value.direction = -uv_to_direction(image_sample.uv, image_offset, image_u_scale, projection);
  sample_value.normal = sample_value.direction;
  OrthonormalBasis basis = orthonormal_basis(sample_value.direction);
  float2 disk_sample = sample_disk(float2(rnd01(seed), rnd01(seed)));
  sample_value.origin = globals_data.bounding_sphere_center + globals_data.bounding_sphere_radius * (disk_sample.x * basis.u + disk_sample.y * basis.v - sample_value.direction);
  sample_value.origin +=
    sample_value.direction * distance_to_sphere(sample_value.origin, sample_value.direction, globals_data.bounding_sphere_center, globals_data.bounding_sphere_radius);
  sample_value.pdf_dir = projection_environment_image_pdf_to_solid_angle(image_sample.pdf, image_sample.uv, projection);
  sample_value.pdf_area = 1.0f / (kPi * globals_data.bounding_sphere_radius * globals_data.bounding_sphere_radius);
  sample_value.pdf_dir_out = sample_value.pdf_area * sample_value.pdf_dir;
  sample_value.value = evaluate_emission_spectral_source(emitter_profile.emission_spectrum_index, emitter_profile.emission_image_index, image_sample.uv, spect);
  sample_value.is_delta = 0u;
  sample_value.is_distant = 1u;
  return sample_value.pdf_dir > 0.0f;
}

uint wavefront_camera_fixed_max_bounces(GPUWavefrontResources resources) {
  return resources.fixed_max_bounces >> 16u;
}

uint wavefront_light_fixed_max_bounces(GPUWavefrontResources resources) {
  uint result = resources.fixed_max_bounces & 0xFFFFu;
  return (result == 0u) ? wavefront_camera_fixed_max_bounces(resources) : result;
}

uint wavefront_vertex_slot_from_limit(uint path_index, uint path_length, uint fixed_max_bounces) {
  uint vertex_index = 0u;
  if (fixed_max_bounces <= 2u) {
    vertex_index = (path_length == 0u) ? 0u : (1u + ((path_length - 1u) & 1u));
  } else {
    vertex_index = min(path_length, fixed_max_bounces);
  }
  return path_index * (fixed_max_bounces + 1u) + vertex_index;
}

uint wavefront_camera_vertex_slot(uint path_index, uint path_length) {
  GPUWavefrontResources resources = wavefront_load_resources();
  return wavefront_vertex_slot_from_limit(path_index, path_length, wavefront_camera_fixed_max_bounces(resources));
}

uint wavefront_light_vertex_slot(uint path_index, uint path_length) {
  GPUWavefrontResources resources = wavefront_load_resources();
  return wavefront_vertex_slot_from_limit(path_index, path_length, wavefront_light_fixed_max_bounces(resources));
}

uint wavefront_path_vertex_slot(bool from_camera, uint path_index, uint path_length) {
  return from_camera ? wavefront_camera_vertex_slot(path_index, path_length) : wavefront_light_vertex_slot(path_index, path_length);
}

void wavefront_write_vertex(bool from_camera, uint path_index, GPUWavefrontPathState state, GPUWavefrontHit hit) {
  GPUWavefrontResources resources = wavefront_load_resources();
  uint vertex_slot = wavefront_path_vertex_slot(from_camera, path_index, state.path_length - 1u);
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

[numthreads(8, 8, 1)] void wavefront_prepare_main(uint3 dtid : SV_DispatchThreadID) {
  if (constants.camera_buffer_index == kInvalidIndex) {
    return;
  }

  Camera camera = load_camera(bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)]);
  if (any(dtid.xy >= camera.film_size)) {
    return;
  }

  uint pixel_index = dtid.x + dtid.y * camera.film_size.x;
  if (constants.sample_index == 0u) {
    wavefront_film_store(pixel_index, float4(0.0f, 0.0f, 0.0f, 0.0f));
  }
  if ((dtid.x == 0u) && (dtid.y == 0u)) {
    wavefront_queue_reset(wavefront_queue_current_descriptor(true));
    wavefront_queue_reset(wavefront_queue_next_descriptor(true));
    wavefront_queue_reset(wavefront_queue_current_descriptor(false));
    wavefront_queue_reset(wavefront_queue_next_descriptor(false));
  }
}

  [numthreads(8, 8, 1)] void wavefront_init_camera_main(uint3 dtid : SV_DispatchThreadID) {
  if (constants.camera_buffer_index == kInvalidIndex) {
    return;
  }
  Camera camera = load_camera(bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)]);
  if (any(dtid.xy >= camera.film_size)) {
    return;
  }

  uint output_pixel_index = dtid.x + dtid.y * camera.film_size.x;
  uint2 camera_space_pixel = uint2(dtid.x, camera.film_size.y - 1u - dtid.y);
  uint seed_pixel_index = camera_space_pixel.x + camera_space_pixel.y * camera.film_size.x;
  uint seed = scene_random_seed(seed_pixel_index, constants.sample_index);
  SpectralQuery spect = spectral_query_sample();
  if (scene_uses_spectral_mode()) {
    spect = spectral_query_spectral_sample(rnd01(seed));
  }
  float2 film_sample_rnd = float2(rnd01(seed), rnd01(seed));
  float2 uv = camera_sample_film_uv(dtid.xy, camera.film_size, film_sample_rnd);
  float2 lens_rnd = camera_lens_sampling_enabled(camera.lens_radius, camera.focal_distance) ? sample_primary_hybrid_2d(dtid.xy, constants.sample_index, kSamplerStreamSupport, seed)
                                                                                            : float2(0.0f, 0.0f);
  GPUWavefrontPathState state = (GPUWavefrontPathState)0;
  state.ray = camera_generate_primary_ray(camera, uv, lens_rnd);
  state.throughput = spectral_response_make(spect, 1.0f);
  state.eta = 1.0f;
  state.eta_scale = 1.0f;
  state.medium_index = camera.medium_index;
  state.path_length = 1u;
  state.pixel_index = output_pixel_index;
  state.flags = GPUWavefrontPathFlags::Valid | GPUWavefrontPathFlags::Connectible | GPUWavefrontPathFlags::From_camera;
  state.path_source = PathSource::Camera;
  state.sampler_seed = seed;
  state.pixel = camera_space_pixel;
  state.spect = spect;
  state.film_uv = uv;
  state.last_vertex_index = output_pixel_index;
  wavefront_store_path_state(wavefront_load_resources().camera_state_buffer, output_pixel_index, state);
  wavefront_queue_store(wavefront_queue_current_descriptor(true), output_pixel_index, output_pixel_index);
  if ((dtid.x == 0u) && (dtid.y == 0u)) {
    WAVEFRONT_RW_BUFFER(wavefront_queue_current_descriptor(true)).Store(0u, camera.film_size.x * camera.film_size.y);
  }
}

[numthreads(8, 8, 1)] void wavefront_init_light_main(uint3 dtid : SV_DispatchThreadID) {
  if (constants.camera_buffer_index == kInvalidIndex) {
    return;
  }
  Camera camera = load_camera(bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)]);
  if (any(dtid.xy >= camera.film_size)) {
    return;
  }

  uint path_index = dtid.x + dtid.y * camera.film_size.x;
  uint seed = scene_random_seed(path_index, constants.sample_index ^ 0x9e3779b9u);
  SpectralQuery spect = spectral_query_sample();
  if (scene_uses_spectral_mode()) {
    spect = spectral_query_spectral_sample(rnd01(seed));
  }
  WavefrontEmitterSample emitter_sample = (WavefrontEmitterSample)0;
  if (wavefront_sample_light_emission(spect, seed, emitter_sample) == false) {
    return;
  }

  float cosine_term = max(kEpsilon, dot(emitter_sample.direction, emitter_sample.normal));
  GPUWavefrontPathState state = (GPUWavefrontPathState)0;
  state.ray.o = offset_ray(emitter_sample.origin, emitter_sample.normal);
  state.ray.d = emitter_sample.direction;
  state.ray.min_t = kRayEpsilon;
  state.ray.max_t = kMaxFloat;
  state.throughput = spectral_response_mul(emitter_sample.value, cosine_term / max(kEpsilon, emitter_sample.pdf_dir * emitter_sample.pdf_area * emitter_sample.pdf_sample));
  state.eta = 1.0f;
  state.eta_scale = 1.0f;
  state.sampled_bsdf_pdf = emitter_sample.pdf_dir;
  state.medium_index = emitter_sample.medium_index;
  state.path_length = 1u;
  state.pixel_index = path_index;
  state.flags = GPUWavefrontPathFlags::Valid | GPUWavefrontPathFlags::From_light | GPUWavefrontPathFlags::Connectible;
  state.path_source = PathSource::Light;
  state.sampler_seed = seed;
  state.pixel = dtid.xy;
  state.spect = spect;
  state.last_vertex_index = path_index;
  wavefront_store_path_state(wavefront_load_resources().light_state_buffer, path_index, state);
  wavefront_queue_store(wavefront_queue_current_descriptor(false), path_index, path_index);
  if ((dtid.x == 0u) && (dtid.y == 0u)) {
    WAVEFRONT_RW_BUFFER(wavefront_queue_current_descriptor(false)).Store(0u, camera.film_size.x * camera.film_size.y);
  }
}

  [numthreads(1, 1, 1)] void wavefront_reset_queues_main(uint3 dtid : SV_DispatchThreadID) {
  if ((dtid.x != 0u) || (dtid.y != 0u) || (dtid.z != 0u)) {
    return;
  }
  wavefront_queue_reset(wavefront_queue_next_descriptor(true));
  wavefront_queue_reset(wavefront_queue_next_descriptor(false));
}

void wavefront_swap_queues_stage(uint3 dtid) {
  if ((dtid.x != 0u) || (dtid.y != 0u) || (dtid.z != 0u)) {
    return;
  }

  wavefront_queue_reset(wavefront_queue_current_descriptor(true));
  wavefront_queue_reset(wavefront_queue_current_descriptor(false));
}

void wavefront_trace_path(bool from_camera, uint dispatch_index) {
  uint queue_descriptor = wavefront_queue_current_descriptor(from_camera);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  GPUWavefrontResources resources = wavefront_load_resources();
  uint path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  uint state_descriptor = from_camera ? resources.camera_state_buffer : resources.light_state_buffer;
  uint hit_descriptor = from_camera ? resources.camera_hit_buffer : resources.light_hit_buffer;
  GPUWavefrontPathState state = wavefront_load_path_state(state_descriptor, path_index);
  if (wavefront_path_state_valid(state) == false) {
    return;
  }

  RayDesc ray = (RayDesc)0;
  ray.Origin = state.ray.o;
  ray.Direction = state.ray.d;
  ray.TMin = max(kRayEpsilon, state.ray.min_t);
  ray.TMax = max(ray.TMin + kRayEpsilon, state.ray.max_t);

  uint medium_index = state.medium_index;
  uint seed = state.sampler_seed;
  TraceSurfaceResult trace_result = (TraceSurfaceResult)0;
  bool hit_found = wavefront_trace_surface_path_compact(ray, state.spect, medium_index, seed, trace_result);
  state.medium_index = medium_index;
  state.sampler_seed = seed;
  wavefront_store_path_state(state_descriptor, path_index, state);

  GPUWavefrontHit hit = (GPUWavefrontHit)0;
  hit.transmittance = trace_result.transmittance;
  hit.medium_index = medium_index;
  hit.flags = GPUWavefrontHitFlags::Valid;
  if (hit_found) {
    hit.vertex = trace_result.surface_point.vertex;
    hit.geo_normal = trace_result.surface_point.geo_normal;
    hit.hit_t = trace_result.hit_t;
    hit.triangle_index = trace_result.triangle_index;
    hit.material_index = trace_result.tri.material_index;
    hit.emitter_index = trace_result.emitter_index;
  } else {
    hit.flags |= GPUWavefrontHitFlags::Miss;
  }
  wavefront_store_hit(hit_descriptor, path_index, hit);
}

[numthreads(64, 1, 1)] void wavefront_trace_camera_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_trace_path(true, dtid.x);
}

  [numthreads(64, 1, 1)] void wavefront_trace_light_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_trace_path(false, dtid.x);
}

void wavefront_camera_direct_light_prepare_stage(uint dispatch_index) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.direct_light_task_buffer == kInvalidIndex) {
    return;
  }

  GPUWavefrontDirectLightTask empty_task = (GPUWavefrontDirectLightTask)0;
  empty_task.medium_index = kInvalidIndex;
  wavefront_store_direct_light_task(resources.direct_light_task_buffer, dispatch_index, empty_task);

  uint queue_descriptor = wavefront_queue_current_descriptor(true);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  uint path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  GPUWavefrontPathState state = wavefront_load_path_state(resources.camera_state_buffer, path_index);
  GPUWavefrontHit hit = wavefront_load_hit(resources.camera_hit_buffer, path_index);
  if ((wavefront_path_state_valid(state) == false) || (wavefront_hit_valid(hit) == false) || wavefront_hit_is_miss(hit)) {
    return;
  }

  uint connection_length = state.path_length + 1u;
  if ((scene_strategy_enabled(kSceneStrategyConnectToLight) == false) || (connection_length < load_scene_options_min_path_length()) ||
      (connection_length > load_scene_options_max_path_length())) {
    return;
  }

  Material material = (Material)0;
  if ((try_load_material_full(hit.material_index, material) == false) || (gpu_bsdf_sample_supported_class(material.cls) == false)) {
    return;
  }

  uint seed = state.sampler_seed;
  WavefrontEmitterSample emitter_sample = (WavefrontEmitterSample)0;
  if (wavefront_sample_emitter_to_point(load_scene_options_light_sampling(), state.spect, hit.vertex.pos, seed, emitter_sample) == false) {
    state.sampler_seed = seed;
    wavefront_store_path_state(resources.camera_state_buffer, path_index, state);
    return;
  }
  state.sampler_seed = seed;
  wavefront_store_path_state(resources.camera_state_buffer, path_index, state);

  Sampler bsdf_sampler = make_bsdf_sampler(seed);
  BSDFData bsdf_data = make_surface_bsdf_data(hit.vertex, state.spect, state.medium_index, state.ray.d);
  BSDFEval bsdf_eval = wavefront_evaluate_material_bsdf(bsdf_data, emitter_sample.direction, material, bsdf_sampler);
  if ((bsdf_eval_valid(bsdf_eval) == false) || gpu_valid_spectral_response(bsdf_eval.bsdf) == false) {
    return;
  }

  float sampling_pdf = emitter_sample.pdf_dir * emitter_sample.pdf_sample;
  if (sampling_pdf <= 0.0f) {
    return;
  }

  float mis_weight = ((scene_multiple_importance_sampling_enabled() == false) || (emitter_sample.is_delta != 0u)) ? 1.0f : power_heuristic(sampling_pdf, bsdf_eval.pdf);
  SpectralResponse contribution =
    spectral_response_mul(spectral_response_mul(state.throughput, bsdf_eval.bsdf), spectral_response_mul(emitter_sample.value, mis_weight / max(kEpsilon, sampling_pdf)));
  if (gpu_valid_spectral_response(contribution) == false) {
    return;
  }

  float3 shadow_origin = wavefront_surface_shading_position(hit, emitter_sample.direction);
  float3 shadow_delta = emitter_sample.origin - shadow_origin;
  float shadow_distance = length(shadow_delta);
  if (shadow_distance <= kRayEpsilon) {
    return;
  }

  GPUWavefrontDirectLightTask task = (GPUWavefrontDirectLightTask)0;
  task.shadow_ray.o = shadow_origin;
  task.shadow_ray.d = shadow_delta / shadow_distance;
  task.shadow_ray.min_t = kRayEpsilon;
  task.shadow_ray.max_t = shadow_distance;
  task.contribution = contribution;
  task.mis_weight = mis_weight;
  task.pixel_index = state.pixel_index;
  task.medium_index = state.medium_index;
  task.flags = 1u;
  wavefront_store_direct_light_task(resources.direct_light_task_buffer, dispatch_index, task);
}

void wavefront_camera_direct_light_shadow_stage(uint dispatch_index) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.direct_light_task_buffer == kInvalidIndex) || (resources.direct_light_result_buffer == kInvalidIndex)) {
    return;
  }

  GPUWavefrontDirectLightTask task = wavefront_load_direct_light_task(resources.direct_light_task_buffer, dispatch_index);
  GPUWavefrontDirectLightResult result_value = (GPUWavefrontDirectLightResult)0;
  result_value.transmittance = spectral_response_make(spectral_query_sample(), 0.0f);
  if (task.flags == 0u) {
    wavefront_store_direct_light_result(resources.direct_light_result_buffer, dispatch_index, result_value);
    return;
  }

  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = task.contribution.wavelength;
  spect.flags = task.contribution.flags;
  result_value.transmittance = spectral_response_make(spect, 1.0f);

  uint seed = scene_random_seed(task.pixel_index, (constants.sample_index * 33u) + constants.path_iteration + 1u);
  result_value.visible = wavefront_trace_transmittance_to_point(task.shadow_ray.o, task.shadow_target, spect, task.medium_index, seed, result_value.transmittance) ? 1u : 0u;
  wavefront_store_direct_light_result(resources.direct_light_result_buffer, dispatch_index, result_value);
}

void wavefront_camera_direct_light_accumulate_stage(uint dispatch_index) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.direct_light_task_buffer == kInvalidIndex) || (resources.direct_light_result_buffer == kInvalidIndex)) {
    return;
  }

  GPUWavefrontDirectLightTask task = wavefront_load_direct_light_task(resources.direct_light_task_buffer, dispatch_index);
  GPUWavefrontDirectLightResult result_value = wavefront_load_direct_light_result(resources.direct_light_result_buffer, dispatch_index);
  if ((task.flags == 0u) || (result_value.visible == 0u)) {
    return;
  }

  SpectralResponse value = spectral_response_mul(task.contribution, result_value.transmittance);
  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = value.wavelength;
  spect.flags = value.flags;
  wavefront_film_add(task.pixel_index, spectral_response_to_rgb(value) * wavefront_spectral_weight(spect));
}

[numthreads(64, 1, 1)] void wavefront_direct_light_camera_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_camera_direct_light_prepare_stage(dtid.x);
  wavefront_camera_direct_light_shadow_stage(dtid.x);
  wavefront_camera_direct_light_accumulate_stage(dtid.x);
}

void wavefront_surface_classify(bool from_camera, uint dispatch_index) {
  uint queue_descriptor = wavefront_queue_current_descriptor(from_camera);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  GPUWavefrontResources resources = wavefront_load_resources();
  uint path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  uint state_descriptor = from_camera ? resources.camera_state_buffer : resources.light_state_buffer;
  uint hit_descriptor = from_camera ? resources.camera_hit_buffer : resources.light_hit_buffer;
  GPUWavefrontPathState state = wavefront_load_path_state(state_descriptor, path_index);
  GPUWavefrontHit hit = wavefront_load_hit(hit_descriptor, path_index);
  if ((wavefront_path_state_valid(state) == false) || (wavefront_hit_valid(hit) == false)) {
    return;
  }

  state.throughput = spectral_response_mul(state.throughput, hit.transmittance);
  if (wavefront_hit_is_miss(hit)) {
    if (from_camera && scene_strategy_enabled(kSceneStrategyDirectHit) && (state.path_length >= load_scene_options_min_path_length()) &&
        (state.path_length <= load_scene_options_max_path_length())) {
      wavefront_film_add(state.pixel_index,
        spectral_response_to_rgb(spectral_response_mul(state.throughput, gpu_evaluate_distant_emission_spectral_all(state.ray.d, state.spect))) *
          wavefront_spectral_weight(state.spect));
    }
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  if (from_camera && scene_strategy_enabled(kSceneStrategyDirectHit) && (state.path_length >= load_scene_options_min_path_length()) &&
      (state.path_length <= load_scene_options_max_path_length()) && (hit.emitter_index != kInvalidIndex)) {
    wavefront_film_add(state.pixel_index,
      spectral_response_to_rgb(spectral_response_mul(state.throughput, gpu_evaluate_local_emission_spectral(hit.emitter_index, hit.vertex.tex, state.spect))) *
        wavefront_spectral_weight(state.spect));
  }

  wavefront_write_vertex(from_camera, path_index, state, hit);
  wavefront_write_path_meta(from_camera, path_index, state);
  wavefront_store_path_state(state_descriptor, path_index, state);
}

void wavefront_surface_continue(bool from_camera, uint dispatch_index) {
  uint queue_descriptor = wavefront_queue_current_descriptor(from_camera);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  GPUWavefrontResources resources = wavefront_load_resources();
  uint path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  uint state_descriptor = from_camera ? resources.camera_state_buffer : resources.light_state_buffer;
  uint hit_descriptor = from_camera ? resources.camera_hit_buffer : resources.light_hit_buffer;
  GPUWavefrontPathState state = wavefront_load_path_state(state_descriptor, path_index);
  GPUWavefrontHit hit = wavefront_load_hit(hit_descriptor, path_index);
  if ((wavefront_path_state_valid(state) == false) || (wavefront_hit_valid(hit) == false) || wavefront_hit_is_miss(hit)) {
    return;
  }

  if ((state.path_length + 1u) > resources.max_path_length) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  Material material = (Material)0;
  if ((try_load_material_full(hit.material_index, material) == false) || (gpu_bsdf_sample_supported_class(material.cls) == false)) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  Sampler bsdf_sampler = make_bsdf_sampler(state.sampler_seed);
  BSDFData bsdf_data = make_surface_bsdf_data(hit.vertex, state.spect, state.medium_index, state.ray.d);
  if (from_camera == false) {
    bsdf_data.path_source = PathSource::Light;
  }
  BSDFSample bsdf_sample = gpu_sample_material_bsdf(make_scene_bsdf_resource_gpu_context(), bsdf_data, material, bsdf_sampler);
  state.sampler_seed = bsdf_sampler.seed;
  if ((bsdf_sample_valid(bsdf_sample) == false) || (gpu_valid_direction(bsdf_sample.w_o) == false) || (gpu_valid_spectral_response(bsdf_sample.weight) == false)) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  state.throughput = spectral_response_mul(state.throughput, bsdf_sample.weight);
  state.sampled_bsdf_pdf = bsdf_sample.pdf;
  state.eta *= bsdf_sample.eta;
  state.eta_scale *= abs(bsdf_sample.eta);
  state.path_length += 1u;
  uint continuation_path_length = 0u;
  if (state.path_length > 0u) {
    continuation_path_length = state.path_length - 1u;
  }
  if (spectral_response_is_zero(state.throughput) ||
      (gpu_random_continue(continuation_path_length, load_scene_options_random_path_termination(), state.eta, state.sampler_seed, state.throughput) == false)) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }
  state.flags = GPUWavefrontPathFlags::Valid | (from_camera ? GPUWavefrontPathFlags::From_camera : GPUWavefrontPathFlags::From_light);
  if (bsdf_sample_is_delta(bsdf_sample) == false) {
    state.flags |= GPUWavefrontPathFlags::Connectible;
  }
  state.ray.o = wavefront_surface_shading_position(hit, bsdf_sample.w_o);
  state.ray.d = normalize(bsdf_sample.w_o);
  state.ray.min_t = kRayEpsilon;
  state.ray.max_t = kMaxFloat;
  wavefront_enqueue_next_state(from_camera, path_index, state);
}

[numthreads(64, 1, 1)] void wavefront_surface_camera_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_classify(true, dtid.x);
  wavefront_surface_continue(true, dtid.x);
}

  [numthreads(64, 1, 1)] void wavefront_surface_light_main(uint3 dtid : SV_DispatchThreadID) {
  wavefront_surface_classify(false, dtid.x);
  wavefront_surface_continue(false, dtid.x);
}

[numthreads(8, 8, 1)] void wavefront_finalize_main(uint3 dtid : SV_DispatchThreadID) {
  if (constants.camera_buffer_index == kInvalidIndex) {
    return;
  }
  Camera camera = load_camera(bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)]);
  if (any(dtid.xy >= camera.film_size)) {
    return;
  }

  uint pixel_index = dtid.x + dtid.y * camera.film_size.x;
  float4 value = wavefront_film_load(pixel_index);
  float sample_count = float(max(1u, constants.sample_index + 1u));
  bindless_storage_textures[NonUniformResourceIndex(constants.output_image_index)][dtid.xy] = float4(max(value.xyz / sample_count, float3(0.0f, 0.0f, 0.0f)), 1.0f);
}
