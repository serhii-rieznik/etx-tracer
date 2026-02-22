#pragma once

#include "interop.hxx"

ETX_SHARED_INLINE bool scene_resource_shared_is_available(uint32_t descriptor_index) {
  return descriptor_index != kInvalidIndex;
}

ETX_SHARED_INLINE bool scene_resource_shared_has_material_spectrum_buffers(uint32_t materials_descriptor_index, uint32_t spectrums_descriptor_index) {
  return scene_resource_shared_is_available(materials_descriptor_index) && scene_resource_shared_is_available(spectrums_descriptor_index);
}

ETX_SHARED_INLINE bool scene_resource_shared_has_medium_spectrum_buffers(uint32_t mediums_descriptor_index, uint32_t spectrums_descriptor_index) {
  return scene_resource_shared_is_available(mediums_descriptor_index) && scene_resource_shared_is_available(spectrums_descriptor_index);
}

ETX_SHARED_INLINE bool scene_resource_shared_has_emitter_buffers(
  uint32_t emitter_instances_descriptor_index, uint32_t emitter_profiles_descriptor_index, uint32_t spectrums_descriptor_index, uint32_t scene_globals_descriptor_index) {
  return scene_resource_shared_is_available(emitter_instances_descriptor_index) && scene_resource_shared_is_available(emitter_profiles_descriptor_index) &&
         scene_resource_shared_is_available(spectrums_descriptor_index) && scene_resource_shared_is_available(scene_globals_descriptor_index);
}

ETX_SHARED_INLINE bool scene_resource_shared_can_sample_spectrum(uint32_t spectrums_descriptor_index, uint32_t spectrum_index) {
  return scene_resource_shared_is_available(spectrums_descriptor_index) && scene_resource_shared_is_available(spectrum_index);
}

ETX_SHARED_INLINE bool scene_resource_shared_can_apply_image(uint32_t images_descriptor_index, uint32_t image_index) {
  return scene_resource_shared_is_available(images_descriptor_index) && scene_resource_shared_is_available(image_index);
}
