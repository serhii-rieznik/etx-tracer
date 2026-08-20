#pragma once

#include "camera.hxx"
#include "gpu_abi_constants.hxx"
#include "material.hxx"

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
  uint32_t triangle_index;
  float spectrum_weight;
  float additional_weight;
  float triangle_area;
  uint32_t instance_index;
};

struct GPUEmitterProfileABIData {
  uint32_t emission_spectrum_index;
  uint32_t emission_image_index;
  uint32_t emitter_profile_class;
  uint32_t medium_index;
  uint32_t emitter_profile_meta;
  float3 emitter_direction;
  float emitter_angular_size;
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

ETX_SHARED_INLINE float4 gpu_abi_load_f32x4(ByteAddressBuffer buffer, uint32_t byte_offset) {
  return asfloat(buffer.Load4(byte_offset));
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

ETX_SHARED_INLINE Material gpu_abi_load_material_full(ByteAddressBuffer buffer, uint32_t material_index) {
  uint32_t base_offset = material_index * kMaterialStride;
  Material result = ETX_ZERO(Material);

  result.reflectance.spectrum_index = gpu_abi_load_u32(buffer, base_offset + kMaterialReflectanceSpectrumIndexOffset);
  result.reflectance.image_index = gpu_abi_load_u32(buffer, base_offset + kMaterialReflectanceImageIndexOffset);
  result.scattering.spectrum_index = gpu_abi_load_u32(buffer, base_offset + kMaterialScatteringSpectrumIndexOffset);
  result.scattering.image_index = gpu_abi_load_u32(buffer, base_offset + kMaterialScatteringImageIndexOffset);
  result.emission.spectrum_index = gpu_abi_load_u32(buffer, base_offset + kMaterialEmissionSpectrumIndexOffset);
  result.emission.image_index = gpu_abi_load_u32(buffer, base_offset + kMaterialEmissionImageIndexOffset);
  result.subsurface.spectrum_index = gpu_abi_load_u32(buffer, base_offset + kMaterialSubsurfaceSpectrumIndexOffset);
  result.subsurface.image_index = gpu_abi_load_u32(buffer, base_offset + kMaterialSubsurfaceImageIndexOffset);

  result.roughness.value = asfloat(buffer.Load4(base_offset + kMaterialRoughnessValueOffset));
  result.roughness.image_index = gpu_abi_load_u32(buffer, base_offset + kMaterialRoughnessImageIndexOffset);
  result.roughness.channel = gpu_abi_load_u32(buffer, base_offset + kMaterialRoughnessChannelOffset);

  result.metalness.value = asfloat(buffer.Load4(base_offset + kMaterialMetalnessValueOffset));
  result.metalness.image_index = gpu_abi_load_u32(buffer, base_offset + kMaterialMetalnessImageIndexOffset);
  result.metalness.channel = gpu_abi_load_u32(buffer, base_offset + kMaterialMetalnessChannelOffset);

  result.transmission.value = asfloat(buffer.Load4(base_offset + kMaterialTransmissionValueOffset));
  result.transmission.image_index = gpu_abi_load_u32(buffer, base_offset + kMaterialTransmissionImageIndexOffset);
  result.transmission.channel = gpu_abi_load_u32(buffer, base_offset + kMaterialTransmissionChannelOffset);

  result.thinfilm.ior.cls = gpu_abi_load_u32(buffer, base_offset + kMaterialThinfilmIorClassOffset);
  result.thinfilm.ior.eta_index = gpu_abi_load_u32(buffer, base_offset + kMaterialThinfilmIorEtaIndexOffset);
  result.thinfilm.ior.k_index = gpu_abi_load_u32(buffer, base_offset + kMaterialThinfilmIorKIndexOffset);
  result.thinfilm.thinkness_image = gpu_abi_load_u32(buffer, base_offset + kMaterialThinfilmThicknessImageOffset);
  result.thinfilm.min_thickness = gpu_abi_load_f32(buffer, base_offset + kMaterialThinfilmMinThicknessOffset);
  result.thinfilm.max_thickness = gpu_abi_load_f32(buffer, base_offset + kMaterialThinfilmMaxThicknessOffset);
  result.thinfilm.weight = gpu_abi_load_f32(buffer, base_offset + kMaterialThinfilmWeightOffset);

  result.diffraction_grating.period_nm = gpu_abi_load_f32(buffer, base_offset + kMaterialDiffractionGratingPeriodNmOffset);
  result.diffraction_grating.optical_path_difference_nm = gpu_abi_load_f32(buffer, base_offset + kMaterialDiffractionGratingOpticalPathDifferenceNmOffset);
  result.diffraction_grating.duty_cycle = gpu_abi_load_f32(buffer, base_offset + kMaterialDiffractionGratingDutyCycleOffset);
  result.diffraction_grating.rotation = gpu_abi_load_f32(buffer, base_offset + kMaterialDiffractionGratingRotationOffset);

  result.ext_ior.cls = gpu_abi_load_u32(buffer, base_offset + kMaterialExtIorClassOffset);
  result.ext_ior.eta_index = gpu_abi_load_u32(buffer, base_offset + kMaterialExtIorEtaIndexOffset);
  result.ext_ior.k_index = gpu_abi_load_u32(buffer, base_offset + kMaterialExtIorKIndexOffset);

  result.int_ior.cls = gpu_abi_load_u32(buffer, base_offset + kMaterialIntIorClassOffset);
  result.int_ior.eta_index = gpu_abi_load_u32(buffer, base_offset + kMaterialIntIorEtaIndexOffset);
  result.int_ior.k_index = gpu_abi_load_u32(buffer, base_offset + kMaterialIntIorKIndexOffset);

  result.subsurface_cls = gpu_abi_load_u32(buffer, base_offset + kMaterialSubsurfaceClassOffset);
  result.subsurface_path = gpu_abi_load_u32(buffer, base_offset + kMaterialSubsurfacePathOffset);
  result.cls = gpu_abi_load_u32(buffer, base_offset + kMaterialClassOffset);
  result.int_medium = gpu_abi_load_u32(buffer, base_offset + kMaterialIntMediumOffset);
  result.ext_medium = gpu_abi_load_u32(buffer, base_offset + kMaterialExtMediumOffset);
  result.normal_image_index = gpu_abi_load_u32(buffer, base_offset + kMaterialNormalImageIndexOffset);
  result.two_sided = gpu_abi_load_u32(buffer, base_offset + kMaterialTwoSidedOffset);
  result.normal_scale = gpu_abi_load_f32(buffer, base_offset + kMaterialNormalScaleOffset);
  result.opacity = gpu_abi_load_f32(buffer, base_offset + kMaterialOpacityOffset);
  result.emission_collimation = gpu_abi_load_f32(buffer, base_offset + kMaterialEmissionCollimationOffset);
  result.energy_compensation_interface_index = gpu_abi_load_u32(buffer, base_offset + kMaterialEnergyCompensationInterfaceIndexOffset);
  result.conductor_energy_compensation_interface_index = gpu_abi_load_u32(buffer, base_offset + kMaterialConductorEnergyCompensationInterfaceIndexOffset);
  return result;
}

ETX_SHARED_INLINE GPUEmitterInstanceABIData gpu_abi_load_emitter_instance(ByteAddressBuffer buffer, uint32_t emitter_index) {
  uint32_t base_offset = emitter_index * kEmitterStride;
  GPUEmitterInstanceABIData result;
  result.emitter_class = gpu_abi_load_u32(buffer, base_offset + kEmitterClassOffset);
  result.emitter_profile_index = gpu_abi_load_u32(buffer, base_offset + kEmitterProfileOffset);
  result.triangle_index = gpu_abi_load_u32(buffer, base_offset + kEmitterTriangleIndexOffset);
  result.spectrum_weight = gpu_abi_load_f32(buffer, base_offset + kEmitterSpectrumWeightOffset);
  result.additional_weight = gpu_abi_load_f32(buffer, base_offset + kEmitterAdditionalWeightOffset);
  result.triangle_area = gpu_abi_load_f32(buffer, base_offset + kEmitterTriangleAreaOffset);
  result.instance_index = gpu_abi_load_u32(buffer, base_offset + kEmitterInstanceIndexOffset);
  return result;
}

ETX_SHARED_INLINE GPUEmitterProfileABIData gpu_abi_load_emitter_profile(ByteAddressBuffer buffer, uint32_t emitter_profile_index) {
  uint32_t base_offset = emitter_profile_index * kEmitterProfileStride;
  GPUEmitterProfileABIData result;
  result.emission_spectrum_index = gpu_abi_load_u32(buffer, base_offset + kEmitterProfileEmissionSpectrumIndexOffset);
  result.emission_image_index = gpu_abi_load_u32(buffer, base_offset + kEmitterProfileEmissionImageIndexOffset);
  result.emitter_profile_class = gpu_abi_load_u32(buffer, base_offset + kEmitterProfileClassOffset);
  result.medium_index = gpu_abi_load_u32(buffer, base_offset + kEmitterProfileMediumIndexOffset);
  result.emitter_profile_meta = gpu_abi_load_u32(buffer, base_offset + kEmitterProfileMetaOffset);
  result.emitter_direction = gpu_abi_load_f32x3(buffer, base_offset + kEmitterProfileDirectionalDirectionOffset);
  result.emitter_angular_size = gpu_abi_load_f32(buffer, base_offset + kEmitterProfileDirectionalAngularSizeOffset);
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
  float4 col_0 = gpu_abi_load_f32x4(buffer, kCameraViewProjOffset + 0u);
  float4 col_1 = gpu_abi_load_f32x4(buffer, kCameraViewProjOffset + 16u);
  float4 col_2 = gpu_abi_load_f32x4(buffer, kCameraViewProjOffset + 32u);
  float4 col_3 = gpu_abi_load_f32x4(buffer, kCameraViewProjOffset + 48u);
  camera.view_proj[0][0] = col_0.x;
  camera.view_proj[1][0] = col_0.y;
  camera.view_proj[2][0] = col_0.z;
  camera.view_proj[3][0] = col_0.w;
  camera.view_proj[0][1] = col_1.x;
  camera.view_proj[1][1] = col_1.y;
  camera.view_proj[2][1] = col_1.z;
  camera.view_proj[3][1] = col_1.w;
  camera.view_proj[0][2] = col_2.x;
  camera.view_proj[1][2] = col_2.y;
  camera.view_proj[2][2] = col_2.z;
  camera.view_proj[3][2] = col_2.w;
  camera.view_proj[0][3] = col_3.x;
  camera.view_proj[1][3] = col_3.y;
  camera.view_proj[2][3] = col_3.z;
  camera.view_proj[3][3] = col_3.w;
  camera.area = gpu_abi_load_f32(buffer, kCameraAreaOffset);
  return camera;
}
