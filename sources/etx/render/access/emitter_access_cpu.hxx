#pragma once

#include <etx/render/access/emitter_access_shared.hxx>
#include <etx/render/access/image_access_cpu.hxx>
#include <etx/render/access/material_access_cpu.hxx>
#include <etx/render/access/spectrum_access_cpu.hxx>

namespace etx {

struct Scene;

struct EmitterAccessCPUContext {
  const Scene* scene = nullptr;
};

ETX_SHARED_INLINE EmitterAccessCPUContext make_emitter_access_cpu_context(const Scene& scene) {
  EmitterAccessCPUContext result = {};
  result.scene = &scene;
  return result;
}

ETX_SHARED_INLINE bool emitter_access_try_load_scene_state(ETX_IN(EmitterAccessCPUContext, context), ETX_OUT(uint32_t, emitter_instance_count),
  ETX_OUT(uint32_t, emitter_profile_count)) {
  emitter_instance_count = 0u;
  emitter_profile_count = 0u;
  if (context.scene == nullptr) {
    return false;
  }

  emitter_instance_count = static_cast<uint32_t>(context.scene->emitter_instances.count);
  emitter_profile_count = static_cast<uint32_t>(context.scene->emitter_profiles.count);
  return (emitter_instance_count > 0u) && (emitter_profile_count > 0u);
}

ETX_SHARED_INLINE bool emitter_access_try_load_profile(ETX_IN(EmitterAccessCPUContext, context), uint32_t emitter_profile_count, ETX_INOUT(EmitterAccess, access)) {
  if ((context.scene == nullptr) || (access.emitter_profile_index >= emitter_profile_count)) {
    return false;
  }

  const auto& profile = context.scene->emitter_profiles[access.emitter_profile_index];
  access.emission_spectrum_index = profile.emission.spectrum_index;
  access.emission_image_index = profile.emission.image_index;
  access.emitter_profile_class = static_cast<uint32_t>(profile.cls);
  access.emitter_profile_meta = profile.meta;
  access.emitter_direction = profile.directional.direction;
  access.emitter_angular_size_cosine = profile.directional.angular_size_cosine;
  return access.emission_spectrum_index != kInvalidIndex;
}

ETX_SHARED_INLINE bool emitter_access_try_load_from_instance(ETX_IN(EmitterAccessCPUContext, context), uint32_t emitter_class, uint32_t emitter_profile_index,
  ETX_OUT(EmitterAccess, access)) {
  access = {};

  uint32_t emitter_instance_count = 0u;
  uint32_t emitter_profile_count = 0u;
  if (emitter_access_try_load_scene_state(context, emitter_instance_count, emitter_profile_count) == false) {
    return false;
  }
  (void)emitter_instance_count;

  access.emitter_class = emitter_class;
  access.emitter_profile_index = emitter_profile_index;
  return emitter_access_try_load_profile(context, emitter_profile_count, access);
}

ETX_SHARED_INLINE bool emitter_access_try_load(ETX_IN(EmitterAccessCPUContext, context), uint32_t emitter_index, ETX_OUT(EmitterAccess, access)) {
  access = {};

  uint32_t emitter_instance_count = 0u;
  uint32_t emitter_profile_count = 0u;
  if (emitter_access_try_load_scene_state(context, emitter_instance_count, emitter_profile_count) == false) {
    return false;
  }
  if (emitter_index >= emitter_instance_count) {
    return false;
  }

  const auto& emitter_instance = context.scene->emitter_instances[emitter_index];
  access.emitter_class = static_cast<uint32_t>(emitter_instance.cls);
  access.emitter_profile_index = emitter_instance.profile;
  return emitter_access_try_load_profile(context, emitter_profile_count, access);
}

ETX_SHARED_INLINE bool emitter_access_try_load_local(ETX_IN(EmitterAccessCPUContext, context), uint32_t emitter_index, ETX_OUT(EmitterAccess, access)) {
  if (emitter_access_try_load(context, emitter_index, access) == false) {
    return false;
  }

  return emitter_access_is_local_class(access.emitter_class);
}

ETX_SHARED_INLINE bool emitter_access_try_load_distant(ETX_IN(EmitterAccessCPUContext, context), uint32_t emitter_index, ETX_IN(float3, direction),
  ETX_OUT(EmitterAccess, access)) {
  if (emitter_access_try_load(context, emitter_index, access) == false) {
    return false;
  }

  return emitter_access_accepts_distant(access.emitter_class, access.emitter_profile_class, direction, access.emitter_direction, access.emitter_angular_size_cosine);
}

ETX_SHARED_INLINE bool emitter_access_try_load_from_instance(ETX_IN(EmitterAccessCPUContext, context), ETX_IN(Emitter, emitter_instance), ETX_OUT(EmitterAccess, access)) {
  return emitter_access_try_load_from_instance(context, static_cast<uint32_t>(emitter_instance.cls), emitter_instance.profile, access);
}

ETX_SHARED_INLINE bool emitter_access_try_load_local_from_instance(ETX_IN(EmitterAccessCPUContext, context), ETX_IN(Emitter, emitter_instance), ETX_OUT(EmitterAccess, access)) {
  if (emitter_access_try_load_from_instance(context, emitter_instance, access) == false) {
    return false;
  }

  return emitter_access_is_local_class(access.emitter_class);
}

ETX_SHARED_INLINE bool emitter_access_try_load_distant_from_instance(ETX_IN(EmitterAccessCPUContext, context), ETX_IN(Emitter, emitter_instance), ETX_IN(float3, direction),
  ETX_OUT(EmitterAccess, access)) {
  if (emitter_access_try_load_from_instance(context, emitter_instance, access) == false) {
    return false;
  }

  return emitter_access_accepts_distant(access.emitter_class, access.emitter_profile_class, direction, access.emitter_direction, access.emitter_angular_size_cosine);
}

ETX_SHARED_INLINE bool emitter_access_try_load_image_params(ETX_IN(EmitterAccessCPUContext, context), uint32_t emission_image_index, ETX_OUT(float2, image_offset),
  ETX_OUT(float, image_u_scale)) {
  image_offset = float2(0.0f, 0.0f);
  image_u_scale = 1.0f;
  if ((context.scene == nullptr) || (emission_image_index == kInvalidIndex) || (emission_image_index >= context.scene->images.count)) {
    return false;
  }

  const auto& image = context.scene->images[emission_image_index];
  image_offset = float2{image.offset.x, image.offset.y};
  image_u_scale = image.scale.x;
  return true;
}

ETX_SHARED_INLINE float2 emitter_access_environment_uv(ETX_IN(EmitterAccessCPUContext, context), ETX_IN(EmitterAccess, access), ETX_IN(float3, direction)) {
  float2 image_offset = float2(0.0f, 0.0f);
  float image_u_scale = 1.0f;
  emitter_access_try_load_image_params(context, access.emission_image_index, image_offset, image_u_scale);
  return ::emitter_access_shared_environment_uv(access.emitter_class, access.emitter_profile_meta, image_offset, image_u_scale, access.emitter_direction,
    access.emitter_angular_size_cosine, direction);
}

ETX_SHARED_INLINE bool emitter_access_can_sample_spectrum(ETX_IN(EmitterAccessCPUContext, context), uint32_t emission_spectrum_index) {
  if (context.scene == nullptr) {
    return false;
  }

  SpectrumAccessCPUContext spectrum_context = make_spectrum_access_cpu_context(context.scene->spectrums.a, static_cast<uint32_t>(context.scene->spectrums.count));
  return spectrum_access_can_evaluate(spectrum_context, emission_spectrum_index);
}

ETX_SHARED_INLINE uint32_t emitter_access_external_medium_index(ETX_IN(EmitterAccessCPUContext, context), ETX_IN(Emitter, emitter_instance)) {
  if (context.scene == nullptr) {
    return kInvalidIndex;
  }

  if (static_cast<uint32_t>(emitter_instance.cls) == EmitterClass::Area) {
    if (emitter_instance.triangle_index >= context.scene->triangles.count) {
      return kInvalidIndex;
    }

    uint32_t material_index = context.scene->triangles[emitter_instance.triangle_index].material_index;
    MaterialAccess material_access = {};
    MaterialAccessCPUContext material_context = make_material_access_cpu_context(*context.scene);
    if (material_access_try_load(material_context, material_index, material_access) == false) {
      return kInvalidIndex;
    }

    return material_access.ext_medium_index;
  }

  if (emitter_instance.profile >= context.scene->emitter_profiles.count) {
    return kInvalidIndex;
  }

  return context.scene->emitter_profiles[emitter_instance.profile].medium_index;
}

}  // namespace etx
