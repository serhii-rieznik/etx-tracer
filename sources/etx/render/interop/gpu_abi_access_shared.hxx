#pragma once

#include "camera.hxx"
#include "gpu_abi_constants.hxx"

struct GPUMaterialABIData {
  uint32_t material_class;
  uint32_t int_medium_index;
  uint32_t ext_medium_index;
  uint32_t scattering_spectrum_index;
  uint32_t scattering_image_index;
  float opacity;
};

struct GPUEmitterInstanceABIData {
  uint32_t emitter_class;
  uint32_t emitter_profile_index;
};

struct GPUEmitterProfileABIData {
  uint32_t emission_spectrum_index;
  uint32_t emission_image_index;
  uint32_t emitter_profile_class;
  uint32_t emitter_profile_meta;
  float3 emitter_direction;
  float emitter_angular_size_cosine;
};

ETX_SHARED_INLINE uint32_t gpu_abi_load_u32(ByteAddressBuffer buffer, uint32_t byte_offset) {
  return buffer.Load(byte_offset);
}

ETX_SHARED_INLINE float gpu_abi_load_f32(ByteAddressBuffer buffer, uint32_t byte_offset) {
  return asfloat(buffer.Load(byte_offset));
}

ETX_SHARED_INLINE float3 gpu_abi_load_f32x3(ByteAddressBuffer buffer, uint32_t byte_offset) {
  return asfloat(buffer.Load3(byte_offset));
}

ETX_SHARED_INLINE uint2 gpu_abi_load_u32x2(ByteAddressBuffer buffer, uint32_t byte_offset) {
  return buffer.Load2(byte_offset);
}

ETX_SHARED_INLINE GPUMaterialABIData gpu_abi_load_material(ByteAddressBuffer buffer, uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  GPUMaterialABIData result;
  result.material_class = gpu_abi_load_u32(buffer, base_offset + kMaterialClassOffset);
  result.int_medium_index = gpu_abi_load_u32(buffer, base_offset + kMaterialIntMediumOffset);
  result.ext_medium_index = gpu_abi_load_u32(buffer, base_offset + kMaterialExtMediumOffset);
  result.scattering_spectrum_index = gpu_abi_load_u32(buffer, base_offset + kMaterialScatteringSpectrumIndexOffset);
  result.scattering_image_index = gpu_abi_load_u32(buffer, base_offset + kMaterialScatteringImageIndexOffset);
  result.opacity = gpu_abi_load_f32(buffer, base_offset + kMaterialOpacityOffset);
  return result;
}

ETX_SHARED_INLINE GPUEmitterInstanceABIData gpu_abi_load_emitter_instance(ByteAddressBuffer buffer, uint32_t emitter_index) {
  uint32_t base_offset = emitter_index * kEmitterStride;
  GPUEmitterInstanceABIData result;
  result.emitter_class = gpu_abi_load_u32(buffer, base_offset + kEmitterClassOffset);
  result.emitter_profile_index = gpu_abi_load_u32(buffer, base_offset + kEmitterProfileOffset);
  return result;
}

ETX_SHARED_INLINE GPUEmitterProfileABIData gpu_abi_load_emitter_profile(ByteAddressBuffer buffer, uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  GPUEmitterProfileABIData result;
  result.emission_spectrum_index = gpu_abi_load_u32(buffer, base_offset + kEmitterProfileEmissionSpectrumIndexOffset);
  result.emission_image_index = gpu_abi_load_u32(buffer, base_offset + kEmitterProfileEmissionImageIndexOffset);
  result.emitter_profile_class = gpu_abi_load_u32(buffer, base_offset + kEmitterProfileClassOffset);
  result.emitter_profile_meta = gpu_abi_load_u32(buffer, base_offset + kEmitterProfileMetaOffset);
  result.emitter_direction = gpu_abi_load_f32x3(buffer, base_offset + kEmitterProfileDirectionalDirectionOffset);
  result.emitter_angular_size_cosine = gpu_abi_load_f32(buffer, base_offset + kEmitterProfileDirectionalAngularSizeCosineOffset);
  return result;
}

ETX_SHARED_INLINE Camera gpu_abi_load_camera(ByteAddressBuffer buffer) {
  ETX_ZERO_INIT(Camera, camera);
  camera.position = gpu_abi_load_f32x3(buffer, kCameraPositionOffset);
  camera.cls = gpu_abi_load_u32(buffer, kCameraClassOffset);
  camera.direction = gpu_abi_load_f32x3(buffer, kCameraDirectionOffset);
  camera.aspect = gpu_abi_load_f32(buffer, kCameraAspectOffset);
  camera.side = gpu_abi_load_f32x3(buffer, kCameraSideOffset);
  camera.tan_half_fov = gpu_abi_load_f32(buffer, kCameraTanHalfFovOffset);
  camera.up = gpu_abi_load_f32x3(buffer, kCameraUpOffset);
  camera.film_size = gpu_abi_load_u32x2(buffer, kCameraFilmSizeOffset);
  camera.lens_radius = gpu_abi_load_f32(buffer, kCameraLensRadiusOffset);
  camera.focal_distance = gpu_abi_load_f32(buffer, kCameraFocalDistanceOffset);
  camera.clip_near = gpu_abi_load_f32(buffer, kCameraClipNearOffset);
  camera.clip_far = gpu_abi_load_f32(buffer, kCameraClipFarOffset);
  camera.lens_image = gpu_abi_load_u32(buffer, kCameraLensImageOffset);
  camera.medium_index = gpu_abi_load_u32(buffer, kCameraMediumIndexOffset);
  return camera;
}
