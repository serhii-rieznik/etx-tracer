#pragma once

#include <access/emitter_access_shared.hxx>
#include <access/image_access_gpu.hxx>
#include <access/material_access_gpu.hxx>
#include <interop/gpu_abi_access_shared.hxx>
#include <interop/scene_gpu_access_shared.hxx>

struct EmitterAccessGPUContext {
  uint emitter_instances_descriptor_index;
  uint emitter_profiles_descriptor_index;
  uint spectrums_descriptor_index;
  uint images_descriptor_index;
  uint scene_globals_descriptor_index;
};

EmitterAccessGPUContext make_emitter_access_gpu_context(uint emitter_instances_descriptor_index, uint emitter_profiles_descriptor_index, uint spectrums_descriptor_index,
  uint images_descriptor_index, uint scene_globals_descriptor_index) {
  EmitterAccessGPUContext context;
  context.emitter_instances_descriptor_index = emitter_instances_descriptor_index;
  context.emitter_profiles_descriptor_index = emitter_profiles_descriptor_index;
  context.spectrums_descriptor_index = spectrums_descriptor_index;
  context.images_descriptor_index = images_descriptor_index;
  context.scene_globals_descriptor_index = scene_globals_descriptor_index;
  return context;
}

bool emitter_access_try_load_scene_state(EmitterAccessGPUContext context, out uint emitter_instance_count, out uint emitter_profile_count) {
  emitter_instance_count = 0u;
  emitter_profile_count = 0u;
  if ((context.scene_globals_descriptor_index == kInvalidIndex) || (context.emitter_instances_descriptor_index == kInvalidIndex) ||
      (context.emitter_profiles_descriptor_index == kInvalidIndex)) {
    return false;
  }

  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(context.scene_globals_descriptor_index)];
  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(scene_globals);
  emitter_instance_count = globals_data.emitter_instance_count;
  emitter_profile_count = globals_data.emitter_profile_count;
  return (emitter_instance_count > 0u) && (emitter_profile_count > 0u);
}

bool emitter_access_try_load_environment_state(EmitterAccessGPUContext context, out uint emitter_instance_count, out uint environment_emitter_count) {
  emitter_instance_count = 0u;
  environment_emitter_count = 0u;
  if ((context.scene_globals_descriptor_index == kInvalidIndex) || (context.emitter_instances_descriptor_index == kInvalidIndex)) {
    return false;
  }

  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(context.scene_globals_descriptor_index)];
  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(scene_globals);
  emitter_instance_count = globals_data.emitter_instance_count;
  environment_emitter_count = min(globals_data.environment_emitter_count, SceneLimits::MaxEnvironmentEmitters);
  return (emitter_instance_count > 0u) && (environment_emitter_count > 0u);
}

bool emitter_access_try_load_profile(EmitterAccessGPUContext context, uint emitter_profile_count, inout EmitterAccess access) {
  if (access.emitter_profile_index >= emitter_profile_count) {
    return false;
  }

  ByteAddressBuffer emitter_profile_buffer = bindless_buffers[NonUniformResourceIndex(context.emitter_profiles_descriptor_index)];
  GPUEmitterProfileABIData profile_data = gpu_abi_load_emitter_profile(emitter_profile_buffer, access.emitter_profile_index);
  access.emission_spectrum_index = profile_data.emission_spectrum_index;
  access.emission_image_index = profile_data.emission_image_index;
  access.emitter_profile_class = profile_data.emitter_profile_class;
  access.medium_index = profile_data.medium_index;
  access.emitter_profile_meta = profile_data.emitter_profile_meta;
  access.emitter_direction = profile_data.emitter_direction;
  access.emitter_angular_size = profile_data.emitter_angular_size;
  access.emitter_angular_size_cosine = profile_data.emitter_angular_size_cosine;
  return access.emission_spectrum_index != kInvalidIndex;
}

bool emitter_access_try_load_from_instance(EmitterAccessGPUContext context, uint emitter_class, uint emitter_profile_index, out EmitterAccess access) {
  access = ETX_ZERO(EmitterAccess);

  uint emitter_instance_count = 0u;
  uint emitter_profile_count = 0u;
  if (emitter_access_try_load_scene_state(context, emitter_instance_count, emitter_profile_count) == false) {
    return false;
  }
  (void)emitter_instance_count;

  access.emitter_class = emitter_class;
  access.emitter_profile_index = emitter_profile_index;
  return emitter_access_try_load_profile(context, emitter_profile_count, access);
}

bool emitter_access_try_load(EmitterAccessGPUContext context, uint emitter_index, out EmitterAccess access) {
  access = ETX_ZERO(EmitterAccess);

  uint emitter_instance_count = 0u;
  uint emitter_profile_count = 0u;
  if (emitter_access_try_load_scene_state(context, emitter_instance_count, emitter_profile_count) == false) {
    return false;
  }
  if (emitter_index >= emitter_instance_count) {
    return false;
  }

  ByteAddressBuffer emitter_instance_buffer = bindless_buffers[NonUniformResourceIndex(context.emitter_instances_descriptor_index)];
  GPUEmitterInstanceABIData instance_data = gpu_abi_load_emitter_instance(emitter_instance_buffer, emitter_index);
  access.emitter_class = instance_data.emitter_class;
  access.emitter_profile_index = instance_data.emitter_profile_index;
  return emitter_access_try_load_profile(context, emitter_profile_count, access);
}

bool emitter_access_try_load_local(EmitterAccessGPUContext context, uint emitter_index, out EmitterAccess access) {
  if (emitter_access_try_load(context, emitter_index, access) == false) {
    return false;
  }

  return emitter_access_is_local_class(access.emitter_class);
}

bool emitter_access_try_load_distant(EmitterAccessGPUContext context, uint emitter_index, float3 direction, out EmitterAccess access) {
  if (emitter_access_try_load(context, emitter_index, access) == false) {
    return false;
  }

  return emitter_access_accepts_distant(access.emitter_class, access.emitter_profile_class, direction, access.emitter_direction, access.emitter_angular_size_cosine);
}

bool emitter_access_try_load_image_params(EmitterAccessGPUContext context, uint emission_image_index, out float2 image_offset, out float image_u_scale) {
  image_offset = float2(0.0f, 0.0f);
  image_u_scale = 1.0f;

  ImageAccessGPUContext image_context = {context.images_descriptor_index};
  ETX_ZERO_INIT(ImageAccessGPUDesc, image_access);
  if (image_access_try_load(image_context, emission_image_index, image_access) == false) {
    return false;
  }

  image_offset = image_access.uv_offset.xy;
  image_u_scale = image_access.uv_scale.x;
  return true;
}

float2 emitter_access_environment_uv(EmitterAccessGPUContext context, EmitterAccess access, float3 direction) {
  float2 image_offset = float2(0.0f, 0.0f);
  float image_u_scale = 1.0f;
  emitter_access_try_load_image_params(context, access.emission_image_index, image_offset, image_u_scale);
  return emitter_access_shared_environment_uv(access.emitter_class, access.emitter_profile_meta, image_offset, image_u_scale, access.emitter_direction,
    access.emitter_angular_size, access.emitter_angular_size_cosine, direction);
}

bool emitter_access_can_sample_spectrum(EmitterAccessGPUContext context, uint emission_spectrum_index) {
  return scene_gpu_can_sample_spectrum(context.spectrums_descriptor_index, emission_spectrum_index);
}

bool emitter_access_try_load_environment_emitter(EmitterAccessGPUContext context, uint local_index, out uint emitter_index) {
  emitter_index = kInvalidIndex;

  uint emitter_instance_count = 0u;
  uint environment_emitter_count = 0u;
  if (emitter_access_try_load_environment_state(context, emitter_instance_count, environment_emitter_count) == false) {
    return false;
  }
  if (local_index >= environment_emitter_count) {
    return false;
  }

  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(context.scene_globals_descriptor_index)];
  emitter_index = scene_gpu_environment_emitter(scene_globals, local_index);
  if (emitter_index >= emitter_instance_count) {
    emitter_index = kInvalidIndex;
    return false;
  }

  return true;
}

uint emitter_access_external_medium_index(EmitterAccessGPUContext context, uint emitter_index) {
  uint emitter_instance_count = 0u;
  uint emitter_profile_count = 0u;
  if (emitter_access_try_load_scene_state(context, emitter_instance_count, emitter_profile_count) == false) {
    return kInvalidIndex;
  }
  if (emitter_index >= emitter_instance_count) {
    return kInvalidIndex;
  }

  ByteAddressBuffer emitter_instance_buffer = bindless_buffers[NonUniformResourceIndex(context.emitter_instances_descriptor_index)];
  GPUEmitterInstanceABIData emitter_instance = gpu_abi_load_emitter_instance(emitter_instance_buffer, emitter_index);
  if (emitter_instance.emitter_class == EmitterClass::Area) {
    if (constants.scene.triangles == kInvalidIndex) {
      return kInvalidIndex;
    }

    SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(bindless_buffers[NonUniformResourceIndex(context.scene_globals_descriptor_index)]);
    if (emitter_instance.triangle_index >= globals_data.triangle_count) {
      return kInvalidIndex;
    }

    ByteAddressBuffer triangle_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)];
    uint triangle_offset = emitter_instance.triangle_index * kTriangleStride;
    uint material_index = gpu_abi_load_u32(triangle_buffer, triangle_offset + 12u);
    MaterialAccessGPUContext material_context = {constants.scene.materials};
    MaterialAccess material_access = ETX_ZERO(MaterialAccess);
    if (material_access_try_load(material_context, material_index, material_access) == false) {
      return kInvalidIndex;
    }

    return material_access.ext_medium_index;
  }

  if (emitter_instance.emitter_profile_index >= emitter_profile_count) {
    return kInvalidIndex;
  }

  ByteAddressBuffer emitter_profile_buffer = bindless_buffers[NonUniformResourceIndex(context.emitter_profiles_descriptor_index)];
  GPUEmitterProfileABIData profile_data = gpu_abi_load_emitter_profile(emitter_profile_buffer, emitter_instance.emitter_profile_index);
  return profile_data.medium_index;
}
