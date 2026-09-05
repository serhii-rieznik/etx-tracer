#pragma once

uint wavefront_connect_queue_base(GPUWavefrontResources resources) {
  return resources.path_capacity * (kGPUWavefrontConnectDispatchArgsCount * kGPUWavefrontConnectLightTaskStride + 2u * sizeof(uint));
}

uint wavefront_connect_material_family(GPUWavefrontPathVertex vertex, uint material_class) {
  if (wavefront_path_vertex_is_medium(vertex)) {
    return 0u;
  }
  switch (material_class) {
    case MaterialClass::Plastic:
      return 1u;
    case MaterialClass::Conductor:
      return 2u;
    case MaterialClass::Dielectric:
      return 3u;
    case MaterialClass::Diffuse:
    case MaterialClass::Translucent:
    case MaterialClass::Mirror:
    case MaterialClass::Boundary:
    case MaterialClass::Void:
    case MaterialClass::DiffractionGrating:
    case MaterialClass::Thinfilm:
    case MaterialClass::Velvet:
      return 0u;
    default:
      return kInvalidIndex;
  }
}

bool wavefront_connect_queue_load(GPUWavefrontResources resources, uint queue, uint index, out uint camera_queue_index, out uint batch_index) {
  camera_queue_index = 0u;
  batch_index = 0u;
  const uint base = wavefront_connect_queue_base(resources);
  const uint count = WAVEFRONT_RO_BUFFER(resources.connect_light_task_buffer).Load(base + queue * sizeof(uint));
  if (index >= count) {
    return false;
  }
  const uint start = WAVEFRONT_RO_BUFFER(resources.connect_light_task_buffer).Load(base + (kGPUWavefrontConnectQueueCount + queue) * sizeof(uint));
  const uint pool_offset = (queue / kGPUWavefrontConnectQueueFamilyCount) * resources.path_capacity * kGPUWavefrontConnectDispatchArgsCount;
  const uint task = WAVEFRONT_RO_BUFFER(resources.connect_light_task_buffer).Load(base + kGPUWavefrontConnectQueueHeaderSize + (pool_offset + start + index) * sizeof(uint));
  camera_queue_index = task % resources.path_capacity;
  batch_index = task / resources.path_capacity;
  return true;
}

void wavefront_connect_queue_count(GPUWavefrontResources resources, uint task_index, uint camera_family, uint light_family) {
  // Classification uses the future shadow payload; scatter finishes before resolve overwrites it.
  WAVEFRONT_RW_BUFFER(resources.connect_light_task_buffer)
    .Store2(task_index * kGPUWavefrontConnectLightTaskStride + kGPUWavefrontConnectQueueClassOffset, uint2(camera_family, light_family));
  const uint base = wavefront_connect_queue_base(resources);
  WAVEFRONT_RW_BUFFER(resources.connect_light_task_buffer).InterlockedAdd(base + camera_family * sizeof(uint), 1u);
  WAVEFRONT_RW_BUFFER(resources.connect_light_task_buffer).InterlockedAdd(base + (kGPUWavefrontConnectQueueFamilyCount + light_family) * sizeof(uint), 1u);
}

void wavefront_connect_queue_scatter(GPUWavefrontResources resources, uint task_index) {
  const uint2 families = WAVEFRONT_RO_BUFFER(resources.connect_light_task_buffer).Load2(task_index * kGPUWavefrontConnectLightTaskStride + kGPUWavefrontConnectQueueClassOffset);
  if ((families.x >= kGPUWavefrontConnectQueueFamilyCount) || (families.y >= kGPUWavefrontConnectQueueFamilyCount)) {
    return;
  }
  const uint base = wavefront_connect_queue_base(resources);
  [unroll] for (uint side = 0u; side < 2u; ++side) {
    const uint queue = side * kGPUWavefrontConnectQueueFamilyCount + families[side];
    const uint start = WAVEFRONT_RO_BUFFER(resources.connect_light_task_buffer).Load(base + (kGPUWavefrontConnectQueueCount + queue) * sizeof(uint));
    uint local_index = 0u;
    WAVEFRONT_RW_BUFFER(resources.connect_light_task_buffer).InterlockedAdd(base + (2u * kGPUWavefrontConnectQueueCount + queue) * sizeof(uint), 1u, local_index);
    const uint pool_offset = side * resources.path_capacity * kGPUWavefrontConnectDispatchArgsCount;
    WAVEFRONT_RW_BUFFER(resources.connect_light_task_buffer).Store(base + kGPUWavefrontConnectQueueHeaderSize + (pool_offset + start + local_index) * sizeof(uint), task_index);
  }
}

void wavefront_connect_queue_prepare(GPUWavefrontResources resources, bool clear_counts) {
  const uint base = wavefront_connect_queue_base(resources);
  uint start = 0u;
  for (uint queue = 0u; queue < kGPUWavefrontConnectQueueCount; ++queue) {
    if (clear_counts) {
      WAVEFRONT_RW_BUFFER(resources.connect_light_task_buffer).Store(base + queue * sizeof(uint), 0u);
      WAVEFRONT_RW_BUFFER(resources.connect_light_task_buffer).Store(base + (2u * kGPUWavefrontConnectQueueCount + queue) * sizeof(uint), 0u);
    } else {
      if ((queue % kGPUWavefrontConnectQueueFamilyCount) == 0u) {
        start = 0u;
      }
      const uint count = WAVEFRONT_RO_BUFFER(resources.connect_light_task_buffer).Load(base + queue * sizeof(uint));
      WAVEFRONT_RW_BUFFER(resources.connect_light_task_buffer).Store(base + (kGPUWavefrontConnectQueueCount + queue) * sizeof(uint), start);
      const uint groups = (count + 63u) / 64u;
      WAVEFRONT_RW_BUFFER(resources.dispatch_args_buffer)
        .Store4(kGPUWavefrontConnectQueueDispatchArgsOffset + queue * kGPUWavefrontDispatchArgsStride, uint4(min(groups, 65535u), max(1u, (groups + 65534u) / 65535u), 1u, 0u));
      start += count;
    }
  }
}
