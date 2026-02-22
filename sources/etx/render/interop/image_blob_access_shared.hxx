#pragma once

#include "gpu_abi_constants.hxx"

#ifndef ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE
# error "ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE must be defined before including image_blob_access_shared.hxx"
#endif

#ifndef ETX_IMAGE_BLOB_ACCESS_SHARED_DESC_TYPE
# error "ETX_IMAGE_BLOB_ACCESS_SHARED_DESC_TYPE must be defined before including image_blob_access_shared.hxx"
#endif

#ifndef ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32
# error "ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32 must be defined before including image_blob_access_shared.hxx"
#endif

#ifndef ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32X2
# error "ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32X2 must be defined before including image_blob_access_shared.hxx"
#endif

#ifndef ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_F32X2
# error "ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_F32X2 must be defined before including image_blob_access_shared.hxx"
#endif

ETX_SHARED_INLINE uint32_t image_blob_access_shared_image_count(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, kImageBlobHeaderImageCountOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_images_offset(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, kImageBlobHeaderImagesOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_data_chunk_count(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, kImageBlobHeaderDataChunkCountOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_data_chunk_indices_offset(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, kImageBlobHeaderDataChunkIndicesOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_offset(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_index) {
  uint32_t images_offset = image_blob_access_shared_images_offset(context);
  return images_offset + image_index * kImageDescStride;
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_format(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescFormatOffset);
}

ETX_SHARED_INLINE uint2 image_blob_access_shared_desc_size(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32X2(context, image_desc_offset + kImageDescISizeOffset);
}

ETX_SHARED_INLINE float2 image_blob_access_shared_desc_fsize(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_F32X2(context, image_desc_offset + kImageDescFSizeOffset);
}

ETX_SHARED_INLINE float2 image_blob_access_shared_desc_uv_offset(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_F32X2(context, image_desc_offset + kImageDescOffsetOffset);
}

ETX_SHARED_INLINE float2 image_blob_access_shared_desc_uv_scale(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_F32X2(context, image_desc_offset + kImageDescScaleOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_options(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescOptionsOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_pixel_data_offset(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescPixelDataOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_x_distribution_entries_offset(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescXDistributionEntriesOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_y_distribution_entries_offset(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescYDistributionEntriesOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_x_entries_stride(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescXEntriesStrideOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_x_distribution_count(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescXDistributionCountOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_y_entries_count(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescYEntriesCountOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_pixel_data_stride(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescPixelDataStrideOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_pixel_data_chunk_index(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescPixelDataChunkIndexOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_x_distribution_chunk_index(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescXDistributionChunkIndexOffset);
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_desc_y_distribution_chunk_index(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset) {
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, image_desc_offset + kImageDescYDistributionChunkIndexOffset);
}

ETX_SHARED_INLINE void image_blob_access_shared_load_desc(
  ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_desc_offset, ETX_OUT(ETX_IMAGE_BLOB_ACCESS_SHARED_DESC_TYPE, desc)) {
  desc.desc_offset = image_desc_offset;
  desc.format = image_blob_access_shared_desc_format(context, image_desc_offset);
  desc.size = image_blob_access_shared_desc_size(context, image_desc_offset);
  desc.fsize = image_blob_access_shared_desc_fsize(context, image_desc_offset);
  desc.uv_offset = image_blob_access_shared_desc_uv_offset(context, image_desc_offset);
  desc.uv_scale = image_blob_access_shared_desc_uv_scale(context, image_desc_offset);
  desc.options = image_blob_access_shared_desc_options(context, image_desc_offset);
  desc.pixel_data_offset = image_blob_access_shared_desc_pixel_data_offset(context, image_desc_offset);
  desc.x_distribution_entries_offset = image_blob_access_shared_desc_x_distribution_entries_offset(context, image_desc_offset);
  desc.y_distribution_entries_offset = image_blob_access_shared_desc_y_distribution_entries_offset(context, image_desc_offset);
  desc.x_entries_stride = image_blob_access_shared_desc_x_entries_stride(context, image_desc_offset);
  desc.x_distribution_count = image_blob_access_shared_desc_x_distribution_count(context, image_desc_offset);
  desc.y_entries_count = image_blob_access_shared_desc_y_entries_count(context, image_desc_offset);
  desc.pixel_data_stride = image_blob_access_shared_desc_pixel_data_stride(context, image_desc_offset);
  desc.pixel_data_chunk_index = image_blob_access_shared_desc_pixel_data_chunk_index(context, image_desc_offset);
  desc.x_distribution_chunk_index = image_blob_access_shared_desc_x_distribution_chunk_index(context, image_desc_offset);
  desc.y_distribution_chunk_index = image_blob_access_shared_desc_y_distribution_chunk_index(context, image_desc_offset);
}

ETX_SHARED_INLINE bool image_blob_access_shared_try_load_desc(
  ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t image_index, ETX_OUT(ETX_IMAGE_BLOB_ACCESS_SHARED_DESC_TYPE, desc)) {
  uint32_t image_count = image_blob_access_shared_image_count(context);
  if (image_index >= image_count) {
    return false;
  }

  uint32_t image_desc_offset = image_blob_access_shared_desc_offset(context, image_index);
  image_blob_access_shared_load_desc(context, image_desc_offset, desc);
  return true;
}

ETX_SHARED_INLINE uint32_t image_blob_access_shared_chunk_descriptor_index(ETX_IN(ETX_IMAGE_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t chunk_index) {
  uint32_t data_chunk_count = image_blob_access_shared_data_chunk_count(context);
  uint32_t data_chunk_indices_offset = image_blob_access_shared_data_chunk_indices_offset(context);
  if ((chunk_index == kInvalidIndex) || (chunk_index >= data_chunk_count) || (data_chunk_indices_offset == kInvalidIndex)) {
    return kInvalidIndex;
  }
  return ETX_IMAGE_BLOB_ACCESS_SHARED_LOAD_U32(context, data_chunk_indices_offset + chunk_index * 4u);
}
