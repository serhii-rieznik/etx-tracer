#pragma once

uint wavefront_work_queue_count(uint descriptor_index, uint base_offset) {
  if (descriptor_index == kInvalidIndex) {
    return 0u;
  }
  return WAVEFRONT_RO_BUFFER(descriptor_index).Load(base_offset + kGPUWavefrontQueueCountOffset);
}

uint wavefront_work_queue_load(uint descriptor_index, uint base_offset, uint slot) {
  return WAVEFRONT_RO_BUFFER(descriptor_index).Load(base_offset + kGPUWavefrontQueueIndicesOffset + slot * 4u);
}

void wavefront_work_queue_append_wave(uint descriptor_index, uint base_offset, uint value, bool append_value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  const uint wave_count = WaveActiveCountBits(append_value);
  uint wave_base = 0u;
  if (WaveIsFirstLane()) {
    if (wave_count > 0u) {
      buffer.InterlockedAdd(base_offset + kGPUWavefrontQueueCountOffset, wave_count, wave_base);
    }
  }
  wave_base = WaveReadLaneFirst(wave_base);
  if (append_value) {
    const uint wave_offset = WavePrefixCountBits(append_value);
    buffer.Store(base_offset + kGPUWavefrontQueueIndicesOffset + (wave_base + wave_offset) * 4u, value);
  }
}

void wavefront_work_queue_reset(uint descriptor_index, uint base_offset) {
  WAVEFRONT_RW_BUFFER(descriptor_index).Store(base_offset + kGPUWavefrontQueueCountOffset, 0u);
}

uint wavefront_material_queue_stride(GPUWavefrontResources resources) {
  return kGPUWavefrontQueueHeaderSize + resources.path_capacity * 4u;
}

uint wavefront_material_queue_global_index(bool from_camera, uint material_queue_index) {
  return material_queue_index + (from_camera ? 0u : kGPUWavefrontMaterialQueueCountPerPathType);
}

uint wavefront_material_queue_base_offset(GPUWavefrontResources resources, bool from_camera, uint material_queue_index) {
  return wavefront_material_queue_global_index(from_camera, material_queue_index) * wavefront_material_queue_stride(resources);
}

uint wavefront_material_queue_count(GPUWavefrontResources resources, bool from_camera, uint material_queue_index) {
  return wavefront_work_queue_count(resources.material_queue_buffer, wavefront_material_queue_base_offset(resources, from_camera, material_queue_index));
}

uint wavefront_material_queue_load(GPUWavefrontResources resources, bool from_camera, uint material_queue_index, uint slot) {
  return wavefront_work_queue_load(resources.material_queue_buffer, wavefront_material_queue_base_offset(resources, from_camera, material_queue_index), slot);
}

void wavefront_material_queue_append(GPUWavefrontResources resources, bool from_camera, uint material_queue_index, uint dispatch_index) {
  for (uint queue_index = 0u; queue_index < kGPUWavefrontMaterialQueueCountPerPathType; ++queue_index) {
    wavefront_work_queue_append_wave(resources.material_queue_buffer, wavefront_material_queue_base_offset(resources, from_camera, queue_index), dispatch_index,
      material_queue_index == queue_index);
  }
}

uint wavefront_material_queue_index(uint material_class) {
  switch (material_class) {
    case MaterialClass::Plastic:
      return kGPUWavefrontMaterialQueuePlastic;
    case MaterialClass::Conductor:
      return kGPUWavefrontMaterialQueueConductor;
    case MaterialClass::Dielectric:
      return kGPUWavefrontMaterialQueueDielectric;
    case MaterialClass::Thinfilm:
      return kGPUWavefrontMaterialQueueThinfilm;
    case MaterialClass::Diffuse:
    case MaterialClass::Translucent:
    case MaterialClass::Mirror:
    case MaterialClass::Boundary:
    case MaterialClass::Velvet:
    case MaterialClass::Void:
    case MaterialClass::DiffractionGrating:
      return kGPUWavefrontMaterialQueueVarious;
    default:
      return kInvalidIndex;
  }
}

uint wavefront_material_queue_index_from_material(uint material_index) {
  if ((material_index == kInvalidIndex) || (constants.scene.materials == kInvalidIndex)) {
    return kInvalidIndex;
  }
  ByteAddressBuffer materials = WAVEFRONT_RO_BUFFER(constants.scene.materials);
  const uint material_class = materials.Load(material_index * kMaterialStride + kMaterialClassOffset);
  return wavefront_material_queue_index(material_class);
}

uint wavefront_shadow_queue_base_offset(GPUWavefrontResources resources, uint shadow_queue_index) {
  const uint direct_light_size = kGPUWavefrontQueueHeaderSize + resources.path_capacity * 4u;
  const uint connect_light_size = kGPUWavefrontQueueHeaderSize + resources.path_capacity * kGPUWavefrontConnectDispatchArgsCount * 4u;
  if (shadow_queue_index == kGPUWavefrontShadowQueueDirectLight) {
    return 0u;
  }
  if (shadow_queue_index == kGPUWavefrontShadowQueueConnectLight) {
    return direct_light_size;
  }
  return direct_light_size + connect_light_size;
}

uint wavefront_shadow_queue_count(GPUWavefrontResources resources, uint shadow_queue_index) {
  return wavefront_work_queue_count(resources.shadow_queue_buffer, wavefront_shadow_queue_base_offset(resources, shadow_queue_index));
}

uint wavefront_shadow_queue_load(GPUWavefrontResources resources, uint shadow_queue_index, uint slot) {
  return wavefront_work_queue_load(resources.shadow_queue_buffer, wavefront_shadow_queue_base_offset(resources, shadow_queue_index), slot);
}

void wavefront_shadow_queue_append(GPUWavefrontResources resources, uint shadow_queue_index, uint task_index) {
  wavefront_work_queue_append_wave(resources.shadow_queue_buffer, wavefront_shadow_queue_base_offset(resources, shadow_queue_index), task_index, true);
}

void wavefront_reset_work_queues(GPUWavefrontResources resources) {
  for (uint queue_index = 0u; queue_index < kGPUWavefrontMaterialQueueCount; ++queue_index) {
    wavefront_work_queue_reset(resources.material_queue_buffer, queue_index * wavefront_material_queue_stride(resources));
  }
  for (uint queue_index = 0u; queue_index < kGPUWavefrontShadowQueueCount; ++queue_index) {
    wavefront_work_queue_reset(resources.shadow_queue_buffer, wavefront_shadow_queue_base_offset(resources, queue_index));
  }
}
