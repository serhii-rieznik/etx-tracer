#pragma once

#include "image.hxx"

#ifndef ETX_IMAGE_SCENE_ACCESS_SHARED_CONTEXT_TYPE
# error "ETX_IMAGE_SCENE_ACCESS_SHARED_CONTEXT_TYPE must be defined before including image_scene_access_shared.hxx"
#endif

#ifndef ETX_IMAGE_SCENE_ACCESS_SHARED_DESC_TYPE
# error "ETX_IMAGE_SCENE_ACCESS_SHARED_DESC_TYPE must be defined before including image_scene_access_shared.hxx"
#endif

#ifndef ETX_IMAGE_SCENE_ACCESS_SHARED_HAS_IMAGES
# error "ETX_IMAGE_SCENE_ACCESS_SHARED_HAS_IMAGES must be defined before including image_scene_access_shared.hxx"
#endif

#ifndef ETX_IMAGE_SCENE_ACCESS_SHARED_LOAD_DESC
# error "ETX_IMAGE_SCENE_ACCESS_SHARED_LOAD_DESC must be defined before including image_scene_access_shared.hxx"
#endif

#ifndef ETX_IMAGE_SCENE_ACCESS_SHARED_CHUNK_DESCRIPTOR
# error "ETX_IMAGE_SCENE_ACCESS_SHARED_CHUNK_DESCRIPTOR must be defined before including image_scene_access_shared.hxx"
#endif

ETX_SHARED_INLINE bool image_scene_access_shared_try_load_desc(
  ETX_IN(ETX_IMAGE_SCENE_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_index, ETX_OUT(ETX_IMAGE_SCENE_ACCESS_SHARED_DESC_TYPE, image_access)) {
  if (ETX_IMAGE_SCENE_ACCESS_SHARED_HAS_IMAGES(context) == false) {
    return false;
  }

  return ETX_IMAGE_SCENE_ACCESS_SHARED_LOAD_DESC(context, image_index, image_access);
}

ETX_SHARED_INLINE bool image_scene_access_shared_has_alpha(ETX_IN(ETX_IMAGE_SCENE_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_index) {
  ETX_IMAGE_SCENE_ACCESS_SHARED_DESC_TYPE image_access;
  if (image_scene_access_shared_try_load_desc(context, image_index, image_access) == false) {
    return false;
  }

  return (image_access.options & Image::HasAlphaChannel) != 0u;
}

ETX_SHARED_INLINE bool image_scene_access_shared_try_load_pixel_payload_descriptor(ETX_IN(ETX_IMAGE_SCENE_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_index,
  ETX_OUT(ETX_IMAGE_SCENE_ACCESS_SHARED_DESC_TYPE, image_access), ETX_OUT(uint32_t, payload_descriptor_index)) {
  payload_descriptor_index = kInvalidIndex;
  if (image_scene_access_shared_try_load_desc(context, image_index, image_access) == false) {
    return false;
  }

  if ((image_access.pixel_data_offset == kInvalidIndex) || (image_access.pixel_data_stride == 0u) || (image_access.size.x == 0u) || (image_access.size.y == 0u)) {
    return false;
  }

  payload_descriptor_index = ETX_IMAGE_SCENE_ACCESS_SHARED_CHUNK_DESCRIPTOR(context, image_access.pixel_data_chunk_index);
  return payload_descriptor_index != kInvalidIndex;
}

ETX_SHARED_INLINE bool image_scene_access_shared_try_load_distribution_payload_descriptors(ETX_IN(ETX_IMAGE_SCENE_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_index,
  ETX_OUT(ETX_IMAGE_SCENE_ACCESS_SHARED_DESC_TYPE, image_access), ETX_OUT(uint32_t, x_payload_descriptor_index), ETX_OUT(uint32_t, y_payload_descriptor_index),
  ETX_OUT(uint32_t, y_count)) {
  x_payload_descriptor_index = kInvalidIndex;
  y_payload_descriptor_index = kInvalidIndex;
  y_count = 0u;
  if (image_scene_access_shared_try_load_desc(context, image_index, image_access) == false) {
    return false;
  }

  if ((image_access.fsize.x <= 0.0f) || (image_access.fsize.y <= 0.0f) || (image_access.x_entries_stride == 0u) || (image_access.x_distribution_count == 0u) ||
      (image_access.y_entries_count == 0u) || (image_access.x_distribution_entries_offset == kInvalidIndex) || (image_access.y_distribution_entries_offset == kInvalidIndex) ||
      (image_access.x_distribution_chunk_index == kInvalidIndex) || (image_access.y_distribution_chunk_index == kInvalidIndex)) {
    return false;
  }

  x_payload_descriptor_index = ETX_IMAGE_SCENE_ACCESS_SHARED_CHUNK_DESCRIPTOR(context, image_access.x_distribution_chunk_index);
  y_payload_descriptor_index = ETX_IMAGE_SCENE_ACCESS_SHARED_CHUNK_DESCRIPTOR(context, image_access.y_distribution_chunk_index);
  if ((x_payload_descriptor_index == kInvalidIndex) || (y_payload_descriptor_index == kInvalidIndex)) {
    return false;
  }

  y_count = image_access.y_entries_count - 1u;
  if ((y_count == 0u) || (image_access.x_entries_stride <= 1u)) {
    return false;
  }

  return true;
}
