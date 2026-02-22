#pragma once

#include "gpu_abi_constants.hxx"

#ifndef ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE
# error "ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE must be defined before including gpu_abi_access_shared.hxx"
#endif

#ifndef ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32
# error "ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32 must be defined before including gpu_abi_access_shared.hxx"
#endif

#ifndef ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32
# error "ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32 must be defined before including gpu_abi_access_shared.hxx"
#endif

#ifndef ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32X3
# error "ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32X3 must be defined before including gpu_abi_access_shared.hxx"
#endif

#ifndef ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32X2
# error "ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32X2 must be defined before including gpu_abi_access_shared.hxx"
#endif

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_material_scattering_spectrum_index(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, base_offset + kMaterialScatteringSpectrumIndexOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_material_scattering_image_index(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, base_offset + kMaterialScatteringImageIndexOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_material_class(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, base_offset + kMaterialClassOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_material_int_medium(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, base_offset + kMaterialIntMediumOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_material_ext_medium(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, base_offset + kMaterialExtMediumOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_material_opacity(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32(context, base_offset + kMaterialOpacityOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_class(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t emitter_index) {
  uint32_t base_offset = emitter_index * kEmitterStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, base_offset + kEmitterClassOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_profile_index(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t emitter_index) {
  uint32_t base_offset = emitter_index * kEmitterStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, base_offset + kEmitterProfileOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_emission_spectrum_index(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, base_offset + kEmitterProfileEmissionSpectrumIndexOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_emission_image_index(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, base_offset + kEmitterProfileEmissionImageIndexOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_profile_class(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, base_offset + kEmitterProfileClassOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_emitter_profile_meta(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, base_offset + kEmitterProfileMetaOffset);
}

ETX_SHARED_INLINE float3 gpu_abi_access_shared_emitter_profile_direction(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32X3(context, base_offset + kEmitterProfileDirectionalDirectionOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_emitter_profile_angular_size_cosine(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32(context, base_offset + kEmitterProfileDirectionalAngularSizeCosineOffset);
}

ETX_SHARED_INLINE float3 gpu_abi_access_shared_camera_position(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32X3(context, kCameraPositionOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_camera_class(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, kCameraClassOffset);
}

ETX_SHARED_INLINE float3 gpu_abi_access_shared_camera_direction(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32X3(context, kCameraDirectionOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_aspect(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32(context, kCameraAspectOffset);
}

ETX_SHARED_INLINE float3 gpu_abi_access_shared_camera_side(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32X3(context, kCameraSideOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_tan_half_fov(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32(context, kCameraTanHalfFovOffset);
}

ETX_SHARED_INLINE float3 gpu_abi_access_shared_camera_up(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32X3(context, kCameraUpOffset);
}

ETX_SHARED_INLINE uint2 gpu_abi_access_shared_camera_film_size(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32X2(context, kCameraFilmSizeOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_lens_radius(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32(context, kCameraLensRadiusOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_focal_distance(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32(context, kCameraFocalDistanceOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_clip_near(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32(context, kCameraClipNearOffset);
}

ETX_SHARED_INLINE float gpu_abi_access_shared_camera_clip_far(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_F32(context, kCameraClipFarOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_camera_lens_image(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, kCameraLensImageOffset);
}

ETX_SHARED_INLINE uint32_t gpu_abi_access_shared_camera_medium_index(ETX_IN(ETX_GPU_ABI_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_GPU_ABI_ACCESS_SHARED_LOAD_U32(context, kCameraMediumIndexOffset);
}
