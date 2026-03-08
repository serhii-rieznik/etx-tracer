#pragma once

#include "gpu_abi_constants.hxx"

struct GPUABIAccessSharedContext {
  ByteAddressBuffer buffer;
};

ETX_SHARED_INLINE GPUABIAccessSharedContext make_gpu_abi_access_shared_context(ByteAddressBuffer buffer) {
  GPUABIAccessSharedContext context;
  context.buffer = buffer;
  return context;
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_load_u32(ETX_IN(GPUABIAccessSharedContext, context), uint32_t byte_offset) {
  return context.buffer.Load(byte_offset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_load_f32(ETX_IN(GPUABIAccessSharedContext, context), uint32_t byte_offset) {
  return asfloat(context.buffer.Load(byte_offset));
}

ETX_SHARED_INLINE float3 gpu_abi_access_shared_load_f32x3(ETX_IN(GPUABIAccessSharedContext, context), uint32_t byte_offset) {
  return asfloat(context.buffer.Load3(byte_offset));
}

ETX_SHARED_INLINE uint2 gpu_abi_access_shared_load_u32x2(ETX_IN(GPUABIAccessSharedContext, context), uint32_t byte_offset) {
  return context.buffer.Load2(byte_offset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_material_scattering_spectrum_index(ETX_IN(GPUABIAccessSharedContext, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return gpu_abi_access_shared_load_u32(context, base_offset + kMaterialScatteringSpectrumIndexOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_material_scattering_image_index(ETX_IN(GPUABIAccessSharedContext, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return gpu_abi_access_shared_load_u32(context, base_offset + kMaterialScatteringImageIndexOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_material_class(ETX_IN(GPUABIAccessSharedContext, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return gpu_abi_access_shared_load_u32(context, base_offset + kMaterialClassOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_material_int_medium(ETX_IN(GPUABIAccessSharedContext, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return gpu_abi_access_shared_load_u32(context, base_offset + kMaterialIntMediumOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_material_ext_medium(ETX_IN(GPUABIAccessSharedContext, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return gpu_abi_access_shared_load_u32(context, base_offset + kMaterialExtMediumOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_material_opacity(ETX_IN(GPUABIAccessSharedContext, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return gpu_abi_access_shared_load_f32(context, base_offset + kMaterialOpacityOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_class(ETX_IN(GPUABIAccessSharedContext, context), uint32_t emitter_index) {
  uint32_t base_offset = emitter_index * kEmitterStride;
  return gpu_abi_access_shared_load_u32(context, base_offset + kEmitterClassOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_profile_index(ETX_IN(GPUABIAccessSharedContext, context), uint32_t emitter_index) {
  uint32_t base_offset = emitter_index * kEmitterStride;
  return gpu_abi_access_shared_load_u32(context, base_offset + kEmitterProfileOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_emission_spectrum_index(ETX_IN(GPUABIAccessSharedContext, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return gpu_abi_access_shared_load_u32(context, base_offset + kEmitterProfileEmissionSpectrumIndexOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_emission_image_index(ETX_IN(GPUABIAccessSharedContext, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return gpu_abi_access_shared_load_u32(context, base_offset + kEmitterProfileEmissionImageIndexOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_profile_class(ETX_IN(GPUABIAccessSharedContext, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return gpu_abi_access_shared_load_u32(context, base_offset + kEmitterProfileClassOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_profile_meta(ETX_IN(GPUABIAccessSharedContext, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return gpu_abi_access_shared_load_u32(context, base_offset + kEmitterProfileMetaOffset);
}

ETX_SHARED_INLINE float3 gpu_abi_access_shared_emitter_profile_direction(ETX_IN(GPUABIAccessSharedContext, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return gpu_abi_access_shared_load_f32x3(context, base_offset + kEmitterProfileDirectionalDirectionOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_emitter_profile_angular_size_cosine(ETX_IN(GPUABIAccessSharedContext, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return gpu_abi_access_shared_load_f32(context, base_offset + kEmitterProfileDirectionalAngularSizeCosineOffset);
}

ETX_SHARED_INLINE float3 gpu_abi_access_shared_camera_position(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_f32x3(context, kCameraPositionOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_camera_class(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_u32(context, kCameraClassOffset);
}

ETX_SHARED_INLINE float3 gpu_abi_access_shared_camera_direction(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_f32x3(context, kCameraDirectionOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_aspect(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_f32(context, kCameraAspectOffset);
}

ETX_SHARED_INLINE float3 gpu_abi_access_shared_camera_side(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_f32x3(context, kCameraSideOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_tan_half_fov(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_f32(context, kCameraTanHalfFovOffset);
}

ETX_SHARED_INLINE float3 gpu_abi_access_shared_camera_up(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_f32x3(context, kCameraUpOffset);
}

ETX_SHARED_INLINE uint2 gpu_abi_access_shared_camera_film_size(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_u32x2(context, kCameraFilmSizeOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_lens_radius(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_f32(context, kCameraLensRadiusOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_focal_distance(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_f32(context, kCameraFocalDistanceOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_clip_near(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_f32(context, kCameraClipNearOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_clip_far(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_f32(context, kCameraClipFarOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_camera_lens_image(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_u32(context, kCameraLensImageOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_camera_medium_index(ETX_IN(GPUABIAccessSharedContext, context)) {
  return gpu_abi_access_shared_load_u32(context, kCameraMediumIndexOffset);
}
