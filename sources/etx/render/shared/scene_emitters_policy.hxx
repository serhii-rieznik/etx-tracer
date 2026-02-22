#pragma once

#include <etx/render/interop/scene_resource_shared.hxx>

namespace etx {
struct Scene;
struct ImageSceneAccessCPUDesc;
struct Sampler;
ETX_SHARED_INLINE bool try_load_scene_image_access(const Scene& scene, uint32_t image_index, ImageSceneAccessCPUDesc& image_access);

ETX_SHARED_INLINE uint32_t scene_resource_shared_cpu_descriptor_from_count(uint32_t count) {
  return (count == 0u) ? kInvalidIndex : 0u;
}

ETX_SHARED_INLINE bool scene_resource_shared_cpu_can_sample_spectrum(const Scene& scene, uint32_t spectrum_index) {
  uint32_t spectrums_descriptor_index = scene_resource_shared_cpu_descriptor_from_count(static_cast<uint32_t>(scene.spectrums.count));
  return scene_resource_shared_can_sample_spectrum(spectrums_descriptor_index, spectrum_index);
}

ETX_SHARED_INLINE bool scene_resource_shared_cpu_can_apply_image(const Scene& scene, uint32_t image_index) {
  uint32_t images_descriptor_index = scene_resource_shared_cpu_descriptor_from_count(static_cast<uint32_t>(scene.images.count));
  return scene_resource_shared_can_apply_image(images_descriptor_index, image_index);
}

ETX_SHARED_INLINE bool scene_resource_shared_cpu_has_environment_state(const Scene& scene) {
  return (scene_resource_shared_cpu_descriptor_from_count(static_cast<uint32_t>(scene.emitter_instances.count)) != kInvalidIndex) &&
         (scene_resource_shared_cpu_descriptor_from_count(static_cast<uint32_t>(scene.environment_emitters.count)) != kInvalidIndex);
}
}  // namespace etx

struct EmitterEmissionCPUSharedAccess {
  uint32_t emitter_class;
  uint32_t emitter_profile_index;
  uint32_t emitter_profile_class;
  uint32_t emitter_profile_meta;
  uint32_t emission_spectrum_index;
  uint32_t emission_image_index;
  float3 emitter_direction;
  float emitter_angular_size_cosine;
};

struct EmitterEmissionCPUSharedContext {
  const etx::Scene& scene;
};

ETX_SHARED_INLINE bool emitter_emission_shared_cpu_has_required_scene_buffers(ETX_IN(EmitterEmissionCPUSharedContext, context)) {
  uint32_t emitter_instances_descriptor_index = etx::scene_resource_shared_cpu_descriptor_from_count(static_cast<uint32_t>(context.scene.emitter_instances.count));
  uint32_t emitter_profiles_descriptor_index = etx::scene_resource_shared_cpu_descriptor_from_count(static_cast<uint32_t>(context.scene.emitter_profiles.count));
  uint32_t spectrums_descriptor_index = etx::scene_resource_shared_cpu_descriptor_from_count(static_cast<uint32_t>(context.scene.spectrums.count));
  return (emitter_instances_descriptor_index != kInvalidIndex) && (emitter_profiles_descriptor_index != kInvalidIndex) &&
         (spectrums_descriptor_index != kInvalidIndex);
}

ETX_SHARED_INLINE uint32_t emitter_emission_shared_cpu_load_emitter_instance_count(ETX_IN(EmitterEmissionCPUSharedContext, context)) {
  return static_cast<uint32_t>(context.scene.emitter_instances.count);
}

ETX_SHARED_INLINE uint32_t emitter_emission_shared_cpu_load_emitter_profile_count(ETX_IN(EmitterEmissionCPUSharedContext, context)) {
  return static_cast<uint32_t>(context.scene.emitter_profiles.count);
}

ETX_SHARED_INLINE uint32_t emitter_emission_shared_cpu_load_instance_class(ETX_IN(EmitterEmissionCPUSharedContext, context), uint32_t emitter_index) {
  return static_cast<uint32_t>(context.scene.emitter_instances[emitter_index].cls);
}

ETX_SHARED_INLINE uint32_t emitter_emission_shared_cpu_load_instance_profile_index(ETX_IN(EmitterEmissionCPUSharedContext, context), uint32_t emitter_index) {
  return context.scene.emitter_instances[emitter_index].profile;
}

ETX_SHARED_INLINE uint32_t emitter_emission_shared_cpu_load_profile_emission_spectrum_index(
  ETX_IN(EmitterEmissionCPUSharedContext, context), uint32_t emitter_profile_index) {
  return context.scene.emitter_profiles[emitter_profile_index].emission.spectrum_index;
}

ETX_SHARED_INLINE uint32_t emitter_emission_shared_cpu_load_profile_emission_image_index(
  ETX_IN(EmitterEmissionCPUSharedContext, context), uint32_t emitter_profile_index) {
  return context.scene.emitter_profiles[emitter_profile_index].emission.image_index;
}

ETX_SHARED_INLINE uint32_t emitter_emission_shared_cpu_load_profile_class(ETX_IN(EmitterEmissionCPUSharedContext, context), uint32_t emitter_profile_index) {
  return static_cast<uint32_t>(context.scene.emitter_profiles[emitter_profile_index].cls);
}

ETX_SHARED_INLINE uint32_t emitter_emission_shared_cpu_load_profile_meta(ETX_IN(EmitterEmissionCPUSharedContext, context), uint32_t emitter_profile_index) {
  return context.scene.emitter_profiles[emitter_profile_index].meta;
}

ETX_SHARED_INLINE float3 emitter_emission_shared_cpu_load_profile_direction(ETX_IN(EmitterEmissionCPUSharedContext, context), uint32_t emitter_profile_index) {
  return context.scene.emitter_profiles[emitter_profile_index].directional.direction;
}

ETX_SHARED_INLINE float emitter_emission_shared_cpu_load_profile_angular_size_cosine(
  ETX_IN(EmitterEmissionCPUSharedContext, context), uint32_t emitter_profile_index) {
  return context.scene.emitter_profiles[emitter_profile_index].directional.angular_size_cosine;
}

#define ETX_EMITTER_EMISSION_SHARED_CONTEXT_TYPE EmitterEmissionCPUSharedContext
#define ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE EmitterEmissionCPUSharedAccess
#define ETX_EMITTER_EMISSION_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) emitter_emission_shared_cpu_has_required_scene_buffers(context)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_INSTANCE_COUNT(context) emitter_emission_shared_cpu_load_emitter_instance_count(context)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_PROFILE_COUNT(context) emitter_emission_shared_cpu_load_emitter_profile_count(context)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_CLASS(context, emitter_index) emitter_emission_shared_cpu_load_instance_class(context, emitter_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_PROFILE_INDEX(context, emitter_index) emitter_emission_shared_cpu_load_instance_profile_index(context, emitter_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_SPECTRUM_INDEX(context, emitter_profile_index) \
  emitter_emission_shared_cpu_load_profile_emission_spectrum_index(context, emitter_profile_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_IMAGE_INDEX(context, emitter_profile_index) \
  emitter_emission_shared_cpu_load_profile_emission_image_index(context, emitter_profile_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_CLASS(context, emitter_profile_index) emitter_emission_shared_cpu_load_profile_class(context, emitter_profile_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_META(context, emitter_profile_index) emitter_emission_shared_cpu_load_profile_meta(context, emitter_profile_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_DIRECTION(context, emitter_profile_index) emitter_emission_shared_cpu_load_profile_direction(context, emitter_profile_index)
#define ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_ANGULAR_SIZE_COSINE(context, emitter_profile_index) \
  emitter_emission_shared_cpu_load_profile_angular_size_cosine(context, emitter_profile_index)
#include <etx/render/interop/emitter_emission_shared.hxx>
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_ANGULAR_SIZE_COSINE
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_DIRECTION
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_META
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_CLASS
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_IMAGE_INDEX
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_SPECTRUM_INDEX
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_PROFILE_INDEX
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_CLASS
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_PROFILE_COUNT
#undef ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_INSTANCE_COUNT
#undef ETX_EMITTER_EMISSION_SHARED_HAS_REQUIRED_SCENE_BUFFERS
#undef ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE
#undef ETX_EMITTER_EMISSION_SHARED_CONTEXT_TYPE

struct EmitterMediumCPUSharedContext {
  const etx::Scene& scene;
};

ETX_SHARED_INLINE bool emitter_medium_shared_cpu_has_required_scene_buffers(ETX_IN(EmitterMediumCPUSharedContext, context)) {
  uint32_t emitter_profiles_descriptor_index = etx::scene_resource_shared_cpu_descriptor_from_count(static_cast<uint32_t>(context.scene.emitter_profiles.count));
  uint32_t triangles_descriptor_index = etx::scene_resource_shared_cpu_descriptor_from_count(static_cast<uint32_t>(context.scene.triangles.count));
  uint32_t materials_descriptor_index = etx::scene_resource_shared_cpu_descriptor_from_count(static_cast<uint32_t>(context.scene.materials.count));
  return (emitter_profiles_descriptor_index != kInvalidIndex) && (triangles_descriptor_index != kInvalidIndex) && (materials_descriptor_index != kInvalidIndex);
}

ETX_SHARED_INLINE uint32_t emitter_medium_shared_cpu_load_emitter_profile_count(ETX_IN(EmitterMediumCPUSharedContext, context)) {
  return static_cast<uint32_t>(context.scene.emitter_profiles.count);
}

ETX_SHARED_INLINE uint32_t emitter_medium_shared_cpu_load_triangle_count(ETX_IN(EmitterMediumCPUSharedContext, context)) {
  return static_cast<uint32_t>(context.scene.triangles.count);
}

ETX_SHARED_INLINE uint32_t emitter_medium_shared_cpu_load_material_count(ETX_IN(EmitterMediumCPUSharedContext, context)) {
  return static_cast<uint32_t>(context.scene.materials.count);
}

ETX_SHARED_INLINE uint32_t emitter_medium_shared_cpu_load_profile_medium_index(ETX_IN(EmitterMediumCPUSharedContext, context), uint32_t emitter_profile_index) {
  return context.scene.emitter_profiles[emitter_profile_index].medium_index;
}

ETX_SHARED_INLINE uint32_t emitter_medium_shared_cpu_load_triangle_material_index(ETX_IN(EmitterMediumCPUSharedContext, context), uint32_t emitter_triangle_index) {
  return context.scene.triangles[emitter_triangle_index].material_index;
}

ETX_SHARED_INLINE uint32_t emitter_medium_shared_cpu_load_material_ext_medium_index(ETX_IN(EmitterMediumCPUSharedContext, context), uint32_t material_index) {
  return context.scene.materials[material_index].ext_medium;
}

#define ETX_EMITTER_MEDIUM_SHARED_CONTEXT_TYPE EmitterMediumCPUSharedContext
#define ETX_EMITTER_MEDIUM_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) emitter_medium_shared_cpu_has_required_scene_buffers(context)
#define ETX_EMITTER_MEDIUM_SHARED_LOAD_EMITTER_PROFILE_COUNT(context) emitter_medium_shared_cpu_load_emitter_profile_count(context)
#define ETX_EMITTER_MEDIUM_SHARED_LOAD_TRIANGLE_COUNT(context) emitter_medium_shared_cpu_load_triangle_count(context)
#define ETX_EMITTER_MEDIUM_SHARED_LOAD_MATERIAL_COUNT(context) emitter_medium_shared_cpu_load_material_count(context)
#define ETX_EMITTER_MEDIUM_SHARED_LOAD_PROFILE_MEDIUM_INDEX(context, emitter_profile_index) \
  emitter_medium_shared_cpu_load_profile_medium_index(context, emitter_profile_index)
#define ETX_EMITTER_MEDIUM_SHARED_LOAD_TRIANGLE_MATERIAL_INDEX(context, emitter_triangle_index) \
  emitter_medium_shared_cpu_load_triangle_material_index(context, emitter_triangle_index)
#define ETX_EMITTER_MEDIUM_SHARED_LOAD_MATERIAL_EXT_MEDIUM_INDEX(context, material_index) emitter_medium_shared_cpu_load_material_ext_medium_index(context, material_index)
#include <etx/render/interop/emitter_medium_shared.hxx>
#undef ETX_EMITTER_MEDIUM_SHARED_LOAD_MATERIAL_EXT_MEDIUM_INDEX
#undef ETX_EMITTER_MEDIUM_SHARED_LOAD_TRIANGLE_MATERIAL_INDEX
#undef ETX_EMITTER_MEDIUM_SHARED_LOAD_PROFILE_MEDIUM_INDEX
#undef ETX_EMITTER_MEDIUM_SHARED_LOAD_MATERIAL_COUNT
#undef ETX_EMITTER_MEDIUM_SHARED_LOAD_TRIANGLE_COUNT
#undef ETX_EMITTER_MEDIUM_SHARED_LOAD_EMITTER_PROFILE_COUNT
#undef ETX_EMITTER_MEDIUM_SHARED_HAS_REQUIRED_SCENE_BUFFERS
#undef ETX_EMITTER_MEDIUM_SHARED_CONTEXT_TYPE

struct EnvironmentEmissionUVSharedCPUContext {
  const etx::Scene& scene;
};

ETX_SHARED_INLINE bool environment_emission_uv_shared_cpu_is_environment_class(uint32_t emitter_class) {
  return (emitter_class == EmitterClass::Environment);
}

ETX_SHARED_INLINE bool environment_emission_uv_shared_cpu_is_directional_class(uint32_t emitter_class) {
  return (emitter_class == EmitterClass::Directional);
}

ETX_SHARED_INLINE bool environment_emission_uv_shared_cpu_try_load_image_params(
  ETX_IN(EnvironmentEmissionUVSharedCPUContext, context), uint32_t emission_image_index, ETX_OUT(float2, image_offset), ETX_OUT(float, image_u_scale)) {
  image_offset = float2(0.0f, 0.0f);
  image_u_scale = 1.0f;
  etx::ImageSceneAccessCPUDesc image_access = {};
  if (etx::try_load_scene_image_access(context.scene, emission_image_index, image_access) == false) {
    return false;
  }

  const auto& image = context.scene.images[image_access.image_index];
  image_offset = image.offset;
  image_u_scale = image.scale.x;
  return true;
}

#define ETX_ENVIRONMENT_EMISSION_UV_SHARED_CONTEXT_TYPE EnvironmentEmissionUVSharedCPUContext
#define ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_ENVIRONMENT_CLASS(emitter_class) environment_emission_uv_shared_cpu_is_environment_class(emitter_class)
#define ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_DIRECTIONAL_CLASS(emitter_class) environment_emission_uv_shared_cpu_is_directional_class(emitter_class)
#define ETX_ENVIRONMENT_EMISSION_UV_SHARED_TRY_LOAD_IMAGE_PARAMS(context, emission_image_index, image_offset, image_u_scale) \
  environment_emission_uv_shared_cpu_try_load_image_params(context, emission_image_index, image_offset, image_u_scale)
#include <etx/render/interop/environment_emission_uv_shared.hxx>
#undef ETX_ENVIRONMENT_EMISSION_UV_SHARED_TRY_LOAD_IMAGE_PARAMS
#undef ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_DIRECTIONAL_CLASS
#undef ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_ENVIRONMENT_CLASS
#undef ETX_ENVIRONMENT_EMISSION_UV_SHARED_CONTEXT_TYPE

struct EmissionSourceSharedCPUContext {
  const etx::Scene& scene;
};

ETX_SHARED_INLINE bool emission_source_shared_cpu_can_sample_spectrum(
  ETX_IN(EmissionSourceSharedCPUContext, context), uint32_t emission_spectrum_index) {
  return etx::scene_resource_shared_cpu_can_sample_spectrum(context.scene, emission_spectrum_index);
}

ETX_SHARED_INLINE float3 emission_source_shared_cpu_load_spectrum_integrated(
  ETX_IN(EmissionSourceSharedCPUContext, context), uint32_t emission_spectrum_index) {
  if ((emission_spectrum_index == kInvalidIndex) || (emission_spectrum_index >= context.scene.spectrums.count)) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  return context.scene.spectrums[emission_spectrum_index].integrated_value;
}

ETX_SHARED_INLINE SpectralResponse emission_source_shared_cpu_load_spectrum_spectral(
  ETX_IN(EmissionSourceSharedCPUContext, context), uint32_t emission_spectrum_index, ETX_IN(SpectralQuery, spect)) {
  ::SpectralResponse zero_value = ::spectral_response_zero(spect);
  if ((emission_spectrum_index == kInvalidIndex) || (emission_spectrum_index >= context.scene.spectrums.count)) {
    return zero_value;
  }

  etx::SpectralQuery query = {spect.wavelength, spect.flags};
  const etx::SpectralResponse result = context.scene.spectrums[emission_spectrum_index](query);
  return static_cast<const ::SpectralResponse&>(result);
}

ETX_SHARED_INLINE bool emission_source_shared_cpu_can_apply_image(ETX_IN(EmissionSourceSharedCPUContext, context), uint32_t emission_image_index) {
  if (etx::scene_resource_shared_cpu_can_apply_image(context.scene, emission_image_index) == false) {
    return false;
  }

  etx::ImageSceneAccessCPUDesc image_access = {};
  return etx::try_load_scene_image_access(context.scene, emission_image_index, image_access);
}

ETX_SHARED_INLINE float3 emission_source_shared_cpu_evaluate_image_rgb(
  ETX_IN(EmissionSourceSharedCPUContext, context), uint32_t emission_image_index, ETX_IN(float2, uv)) {
  etx::ImageSceneAccessCPUDesc image_access = {};
  if (etx::try_load_scene_image_access(context.scene, emission_image_index, image_access) == false) {
    return float3(1.0f, 1.0f, 1.0f);
  }

  float4 eval = context.scene.images[image_access.image_index].evaluate(uv, nullptr);
  return float3(eval.x, eval.y, eval.z);
}

#define ETX_EMISSION_SOURCE_SHARED_CONTEXT_TYPE EmissionSourceSharedCPUContext
#define ETX_EMISSION_SOURCE_SHARED_CAN_SAMPLE_SPECTRUM(context, emission_spectrum_index) emission_source_shared_cpu_can_sample_spectrum(context, emission_spectrum_index)
#define ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_INTEGRATED(context, emission_spectrum_index) emission_source_shared_cpu_load_spectrum_integrated(context, emission_spectrum_index)
#define ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_SPECTRAL(context, emission_spectrum_index, spect) \
  emission_source_shared_cpu_load_spectrum_spectral(context, emission_spectrum_index, spect)
#define ETX_EMISSION_SOURCE_SHARED_CAN_APPLY_IMAGE(context, emission_image_index) emission_source_shared_cpu_can_apply_image(context, emission_image_index)
#define ETX_EMISSION_SOURCE_SHARED_EVALUATE_IMAGE_RGB(context, emission_image_index, uv) emission_source_shared_cpu_evaluate_image_rgb(context, emission_image_index, uv)
#include <etx/render/interop/emission_source_shared.hxx>
#undef ETX_EMISSION_SOURCE_SHARED_EVALUATE_IMAGE_RGB
#undef ETX_EMISSION_SOURCE_SHARED_CAN_APPLY_IMAGE
#undef ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_SPECTRAL
#undef ETX_EMISSION_SOURCE_SHARED_LOAD_SPECTRUM_INTEGRATED
#undef ETX_EMISSION_SOURCE_SHARED_CAN_SAMPLE_SPECTRUM
#undef ETX_EMISSION_SOURCE_SHARED_CONTEXT_TYPE

struct EnvironmentEmitterSelectCPUSharedContext {
  const etx::Scene& scene;
  etx::Sampler& sampler;
};

ETX_SHARED_INLINE bool environment_emitter_select_shared_cpu_has_scene_globals(ETX_IN(EnvironmentEmitterSelectCPUSharedContext, context)) {
  return etx::scene_resource_shared_cpu_has_environment_state(context.scene);
}

ETX_SHARED_INLINE uint32_t environment_emitter_select_shared_cpu_load_emitter_instance_count(ETX_IN(EnvironmentEmitterSelectCPUSharedContext, context)) {
  return static_cast<uint32_t>(context.scene.emitter_instances.count);
}

ETX_SHARED_INLINE uint32_t environment_emitter_select_shared_cpu_load_environment_emitter_count(ETX_IN(EnvironmentEmitterSelectCPUSharedContext, context)) {
  return static_cast<uint32_t>(context.scene.environment_emitters.count);
}

ETX_SHARED_INLINE uint32_t environment_emitter_select_shared_cpu_load_environment_emitter(ETX_IN(EnvironmentEmitterSelectCPUSharedContext, context), uint32_t index) {
  if (index >= context.scene.environment_emitters.count) {
    return kInvalidIndex;
  }
  return context.scene.environment_emitters.emitters[index];
}

ETX_SHARED_INLINE uint32_t environment_emitter_select_shared_cpu_max_count(ETX_IN(EnvironmentEmitterSelectCPUSharedContext, context)) {
  (void)context;
  return SceneLimits::MaxEnvironmentEmitters;
}

ETX_SHARED_INLINE float environment_emitter_select_shared_cpu_rnd(ETX_INOUT(EnvironmentEmitterSelectCPUSharedContext, context)) {
  return context.sampler.next();
}

#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_CONTEXT_TYPE EnvironmentEmitterSelectCPUSharedContext
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_HAS_SCENE_GLOBALS(context) environment_emitter_select_shared_cpu_has_scene_globals(context)
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_EMITTER_INSTANCE_COUNT(context) environment_emitter_select_shared_cpu_load_emitter_instance_count(context)
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER_COUNT(context) environment_emitter_select_shared_cpu_load_environment_emitter_count(context)
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER(context, index) environment_emitter_select_shared_cpu_load_environment_emitter(context, index)
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_MAX_COUNT(context) environment_emitter_select_shared_cpu_max_count(context)
#define ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_RND(context) environment_emitter_select_shared_cpu_rnd(context)
#include <etx/render/interop/environment_emitter_select_shared.hxx>
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_RND
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_MAX_COUNT
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER_COUNT
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_EMITTER_INSTANCE_COUNT
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_HAS_SCENE_GLOBALS
#undef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_CONTEXT_TYPE

namespace etx {

ETX_SHARED_INLINE bool try_load_emitter_scene_state_shared(const Scene& scene, uint32_t& emitter_instance_count, uint32_t& emitter_profile_count) {
  EmitterEmissionCPUSharedContext context = {scene};
  return emitter_emission_shared_try_load_scene_state(context, emitter_instance_count, emitter_profile_count);
}

ETX_SHARED_INLINE bool try_load_emitter_instance_count_shared(const Scene& scene, uint32_t& emitter_instance_count) {
  uint32_t emitter_profile_count = 0u;
  return try_load_emitter_scene_state_shared(scene, emitter_instance_count, emitter_profile_count);
}

ETX_SHARED_INLINE bool try_load_emitter_emission_access_shared(const Scene& scene, uint32_t emitter_index, EmitterEmissionCPUSharedAccess& access) {
  EmitterEmissionCPUSharedContext context = {scene};
  return emitter_emission_shared_try_load_access(context, emitter_index, access);
}

ETX_SHARED_INLINE bool try_load_local_emitter_emission_access_shared(const Scene& scene, uint32_t emitter_index, EmitterEmissionCPUSharedAccess& access) {
  EmitterEmissionCPUSharedContext context = {scene};
  return emitter_emission_shared_try_load_local_access(context, emitter_index, access);
}

ETX_SHARED_INLINE bool try_load_distant_emitter_emission_access_shared(
  const Scene& scene, uint32_t emitter_index, ETX_IN(float3, direction), EmitterEmissionCPUSharedAccess& access) {
  EmitterEmissionCPUSharedContext context = {scene};
  return emitter_emission_shared_try_load_distant_access(context, emitter_index, direction, access);
}

ETX_SHARED_INLINE bool try_load_emitter_emission_access_from_instance_shared(
  const Scene& scene, const Emitter& emitter_instance, EmitterEmissionCPUSharedAccess& access) {
  EmitterEmissionCPUSharedContext context = {scene};
  return emitter_emission_shared_try_load_access_from_instance(context, static_cast<uint32_t>(emitter_instance.cls), emitter_instance.profile, access);
}

ETX_SHARED_INLINE bool try_load_local_emitter_emission_access_from_instance_shared(
  const Scene& scene, const Emitter& emitter_instance, EmitterEmissionCPUSharedAccess& access) {
  if (try_load_emitter_emission_access_from_instance_shared(scene, emitter_instance, access) == false) {
    return false;
  }

  return emitter_emission_shared_is_local_class(access.emitter_class);
}

ETX_SHARED_INLINE bool try_load_distant_emitter_emission_access_from_instance_shared(
  const Scene& scene, const Emitter& emitter_instance, ETX_IN(float3, direction), EmitterEmissionCPUSharedAccess& access) {
  if (try_load_emitter_emission_access_from_instance_shared(scene, emitter_instance, access) == false) {
    return false;
  }

  return emitter_emission_shared_accepts_distant_access(
    access.emitter_class, access.emitter_profile_class, direction, access.emitter_direction, access.emitter_angular_size_cosine);
}

ETX_SHARED_INLINE uint32_t emitter_external_medium_index_shared(const Scene& scene, const Emitter& emitter_instance) {
  EmitterMediumCPUSharedContext context = {scene};
  return emitter_medium_shared_external_index(context, static_cast<uint32_t>(emitter_instance.cls), emitter_instance.profile, emitter_instance.triangle_index);
}

ETX_SHARED_INLINE bool try_select_environment_emitter_random(const Scene& scene, Sampler& smp, uint32_t& emitter_index, uint32_t& emitter_count) {
  EnvironmentEmitterSelectCPUSharedContext context = {scene, smp};
  return environment_emitter_select_shared_try_select_random(context, emitter_index, emitter_count);
}

ETX_SHARED_INLINE uint32_t environment_emitter_shared_count(const Scene& scene) {
  if (scene_resource_shared_cpu_has_environment_state(scene) == false) {
    return 0u;
  }

  return min(static_cast<uint32_t>(scene.environment_emitters.count), uint32_t(SceneLimits::MaxEnvironmentEmitters));
}

ETX_SHARED_INLINE bool environment_emitter_shared_try_load_index(const Scene& scene, uint32_t local_index, uint32_t& emitter_index) {
  emitter_index = kInvalidIndex;
  uint32_t emitter_count = environment_emitter_shared_count(scene);
  if (local_index >= emitter_count) {
    return false;
  }

  emitter_index = scene.environment_emitters.emitters[local_index];
  if (emitter_index >= scene.emitter_instances.count) {
    emitter_index = kInvalidIndex;
    return false;
  }

  return true;
}

ETX_SHARED_INLINE SpectralResponse evaluate_emission_spectral_source(
  const Scene& scene, uint32_t emission_spectrum_index, uint32_t emission_image_index, ETX_IN(float2, uv), ETX_IN(SpectralQuery, spect)) {
  EmissionSourceSharedCPUContext context = {scene};
  const ::SpectralResponse shared_response =
    ::emission_source_shared_evaluate_spectral(context, emission_spectrum_index, emission_image_index, uv, static_cast<const ::SpectralQuery&>(spect));
  SpectralQuery response_query{shared_response.wavelength, shared_response.flags};
  return ::spectral_response_is_spectral(shared_response) ? SpectralResponse{response_query, shared_response.value} : SpectralResponse{response_query, shared_response.integrated};
}

ETX_SHARED_INLINE SpectralResponse evaluate_emission_spectrum_spectral(const Scene& scene, uint32_t emission_spectrum_index, ETX_IN(SpectralQuery, spect)) {
  EmissionSourceSharedCPUContext context = {scene};
  if (::emission_source_shared_cpu_can_sample_spectrum(context, emission_spectrum_index) == false) {
    return SpectralResponse{spect, 0.0f};
  }

  return scene.spectrums[emission_spectrum_index](spect);
}

}  // namespace etx
