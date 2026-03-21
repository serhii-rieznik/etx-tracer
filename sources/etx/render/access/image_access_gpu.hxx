#pragma once

#include <interop/gpu_abi_constants.hxx>
#include <interop/scene_gpu_access_shared.hxx>
#include <interop/image.hxx>

struct ImageAccessGPUDesc {
  uint format;
  uint2 size;
  float2 fsize;
  float2 uv_offset;
  float2 uv_scale;
  float normalization;
  uint options;
  uint pixel_data_offset;
  uint x_distribution_entries_offset;
  uint y_distribution_entries_offset;
  uint x_entries_stride;
  uint x_distribution_count;
  uint y_entries_count;
  float y_distribution_total_weight;
  uint pixel_data_stride;
  uint pixel_data_chunk_index;
  uint x_distribution_chunk_index;
  uint y_distribution_chunk_index;
};

struct ImageAccessGPUContext {
  uint images_descriptor_index;
};

uint image_access_gpu_blob_load_u32(ByteAddressBuffer buffer, uint byte_offset) {
  return buffer.Load(byte_offset);
}

uint2 image_access_gpu_blob_load_u32x2(ByteAddressBuffer buffer, uint byte_offset) {
  return buffer.Load2(byte_offset);
}

float2 image_access_gpu_blob_load_f32x2(ByteAddressBuffer buffer, uint byte_offset) {
  return asfloat(buffer.Load2(byte_offset));
}

uint image_access_gpu_blob_image_count(ByteAddressBuffer buffer) {
  return image_access_gpu_blob_load_u32(buffer, kImageBlobHeaderImageCountOffset);
}

uint image_access_gpu_blob_images_offset(ByteAddressBuffer buffer) {
  return image_access_gpu_blob_load_u32(buffer, kImageBlobHeaderImagesOffset);
}

uint image_access_gpu_blob_data_chunk_count(ByteAddressBuffer buffer) {
  return image_access_gpu_blob_load_u32(buffer, kImageBlobHeaderDataChunkCountOffset);
}

uint image_access_gpu_blob_data_chunk_indices_offset(ByteAddressBuffer buffer) {
  return image_access_gpu_blob_load_u32(buffer, kImageBlobHeaderDataChunkIndicesOffset);
}

uint image_access_gpu_blob_desc_offset(ByteAddressBuffer buffer, uint image_index) {
  uint images_offset = image_access_gpu_blob_images_offset(buffer);
  return images_offset + image_index * kImageDescStride;
}

void image_access_gpu_blob_load_desc(ByteAddressBuffer buffer, uint image_desc_offset, out ImageAccessGPUDesc desc) {
  desc.format = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescFormatOffset);
  desc.size = image_access_gpu_blob_load_u32x2(buffer, image_desc_offset + kImageDescISizeOffset);
  desc.fsize = image_access_gpu_blob_load_f32x2(buffer, image_desc_offset + kImageDescFSizeOffset);
  desc.uv_offset = image_access_gpu_blob_load_f32x2(buffer, image_desc_offset + kImageDescOffsetOffset);
  desc.uv_scale = image_access_gpu_blob_load_f32x2(buffer, image_desc_offset + kImageDescScaleOffset);
  desc.normalization = asfloat(image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescNormalizationOffset));
  desc.options = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescOptionsOffset);
  desc.pixel_data_offset = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescPixelDataOffset);
  desc.x_distribution_entries_offset = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescXDistributionEntriesOffset);
  desc.y_distribution_entries_offset = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescYDistributionEntriesOffset);
  desc.x_entries_stride = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescXEntriesStrideOffset);
  desc.x_distribution_count = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescXDistributionCountOffset);
  desc.y_entries_count = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescYEntriesCountOffset);
  desc.y_distribution_total_weight = asfloat(image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescYDistributionTotalWeightOffset));
  desc.pixel_data_stride = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescPixelDataStrideOffset);
  desc.pixel_data_chunk_index = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescPixelDataChunkIndexOffset);
  desc.x_distribution_chunk_index = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescXDistributionChunkIndexOffset);
  desc.y_distribution_chunk_index = image_access_gpu_blob_load_u32(buffer, image_desc_offset + kImageDescYDistributionChunkIndexOffset);
}

bool image_access_gpu_blob_try_load_desc(ByteAddressBuffer buffer, uint image_index, out ImageAccessGPUDesc desc) {
  uint image_count = image_access_gpu_blob_image_count(buffer);
  if (image_index >= image_count) {
    return false;
  }

  uint image_desc_offset = image_access_gpu_blob_desc_offset(buffer, image_index);
  image_access_gpu_blob_load_desc(buffer, image_desc_offset, desc);
  return true;
}

uint image_access_gpu_blob_chunk_descriptor(ByteAddressBuffer buffer, uint chunk_index) {
  uint data_chunk_count = image_access_gpu_blob_data_chunk_count(buffer);
  uint data_chunk_indices_offset = image_access_gpu_blob_data_chunk_indices_offset(buffer);
  if ((chunk_index == kInvalidIndex) || (chunk_index >= data_chunk_count) || (data_chunk_indices_offset == kInvalidIndex)) {
    return kInvalidIndex;
  }

  return image_access_gpu_blob_load_u32(buffer, data_chunk_indices_offset + chunk_index * 4u);
}

bool image_access_gpu_has_images(ImageAccessGPUContext context) {
  return scene_gpu_has_descriptor(context.images_descriptor_index);
}

bool image_access_gpu_load_desc(ImageAccessGPUContext context, uint image_index, out ImageAccessGPUDesc image_access) {
  ByteAddressBuffer image_blob = bindless_buffers[NonUniformResourceIndex(context.images_descriptor_index)];
  return image_access_gpu_blob_try_load_desc(image_blob, image_index, image_access);
}

uint image_access_gpu_chunk_descriptor(ImageAccessGPUContext context, uint chunk_index) {
  ByteAddressBuffer image_blob = bindless_buffers[NonUniformResourceIndex(context.images_descriptor_index)];
  return image_access_gpu_blob_chunk_descriptor(image_blob, chunk_index);
}

bool image_access_try_load(ImageAccessGPUContext context, uint image_index, out ImageAccessGPUDesc image_access) {
  if (image_access_gpu_has_images(context) == false) {
    return false;
  }

  return image_access_gpu_load_desc(context, image_index, image_access);
}

bool image_access_has_alpha(ImageAccessGPUContext context, uint image_index) {
  ImageAccessGPUDesc image_access = ETX_ZERO(ImageAccessGPUDesc);
  if (image_access_try_load(context, image_index, image_access) == false) {
    return false;
  }

  return (image_access.options & Image::HasAlphaChannel) != 0u;
}

bool image_access_try_load_pixel_payload(
  ImageAccessGPUContext context, uint image_index, out ImageAccessGPUDesc image_access, out uint payload_descriptor_index) {
  payload_descriptor_index = kInvalidIndex;
  if (image_access_try_load(context, image_index, image_access) == false) {
    return false;
  }

  if ((image_access.pixel_data_offset == kInvalidIndex) || (image_access.pixel_data_stride == 0u) || (image_access.size.x == 0u) || (image_access.size.y == 0u)) {
    return false;
  }

  payload_descriptor_index = image_access_gpu_chunk_descriptor(context, image_access.pixel_data_chunk_index);
  return payload_descriptor_index != kInvalidIndex;
}

bool image_access_try_load_distribution_payloads(
  ImageAccessGPUContext context, uint image_index, out ImageAccessGPUDesc image_access, out uint x_payload_descriptor_index, out uint y_payload_descriptor_index,
  out uint y_count) {
  x_payload_descriptor_index = kInvalidIndex;
  y_payload_descriptor_index = kInvalidIndex;
  y_count = 0u;
  if (image_access_try_load(context, image_index, image_access) == false) {
    return false;
  }

  if ((image_access.fsize.x <= 0.0f) || (image_access.fsize.y <= 0.0f) || (image_access.x_entries_stride == 0u) || (image_access.x_distribution_count == 0u) ||
      (image_access.y_entries_count == 0u) || (image_access.x_distribution_entries_offset == kInvalidIndex) || (image_access.y_distribution_entries_offset == kInvalidIndex) ||
      (image_access.x_distribution_chunk_index == kInvalidIndex) || (image_access.y_distribution_chunk_index == kInvalidIndex)) {
    return false;
  }

  x_payload_descriptor_index = image_access_gpu_chunk_descriptor(context, image_access.x_distribution_chunk_index);
  y_payload_descriptor_index = image_access_gpu_chunk_descriptor(context, image_access.y_distribution_chunk_index);
  if ((x_payload_descriptor_index == kInvalidIndex) || (y_payload_descriptor_index == kInvalidIndex)) {
    return false;
  }

  y_count = image_access.y_entries_count - 1u;
  if ((y_count == 0u) || (image_access.x_entries_stride <= 1u)) {
    return false;
  }

  return true;
}
