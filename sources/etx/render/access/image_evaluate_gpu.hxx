#pragma once

#include <access/image_access_gpu.hxx>

struct ImageEvaluateGPUContext {
  uint images_descriptor_index;
};

ImageEvaluateGPUContext make_image_evaluate_gpu_context(uint images_descriptor_index) {
  ImageEvaluateGPUContext result;
  result.images_descriptor_index = images_descriptor_index;
  return result;
}

float4 image_evaluate_gpu_load_pixel(ByteAddressBuffer payload_buffer, uint format, uint byte_offset) {
  if (format == (uint)Image::Format::RGBA32F) {
    return asfloat(payload_buffer.Load4(byte_offset));
  }

  if (format == (uint)Image::Format::RGBA8) {
    uint packed_rgba = payload_buffer.Load(byte_offset);
    float4 result = float4(float((packed_rgba >> 0u) & 0xFFu), float((packed_rgba >> 8u) & 0xFFu), float((packed_rgba >> 16u) & 0xFFu), float((packed_rgba >> 24u) & 0xFFu));
    return result * (1.0f / 255.0f);
  }

  if (format == (uint)Image::Format::R32F) {
    float value = asfloat(payload_buffer.Load(byte_offset));
    return float4(value, value, value, 1.0f);
  }

  return float4(1.0f, 1.0f, 1.0f, 1.0f);
}

bool image_evaluate_gpu_pixel_offset_valid(ImageAccessGPUDesc image_access, uint byte_offset, uint byte_count) {
  if ((byte_offset < image_access.pixel_data_offset) || (byte_count == 0u)) {
    return false;
  }

  const uint relative_offset = byte_offset - image_access.pixel_data_offset;
  if (relative_offset > image_access.data_size) {
    return false;
  }

  return byte_count <= (image_access.data_size - relative_offset);
}

float4 image_evaluate_gpu_load_pixel_checked(ByteAddressBuffer payload_buffer, ImageAccessGPUDesc image_access, uint byte_offset) {
  if (image_evaluate_gpu_pixel_offset_valid(image_access, byte_offset, image_access.pixel_data_stride) == false) {
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  return image_evaluate_gpu_load_pixel(payload_buffer, image_access.format, byte_offset);
}

float4 image_evaluate_gpu_rgba(ImageEvaluateGPUContext context, uint image_index, float2 uv) {
  ImageAccessGPUContext access_context = {context.images_descriptor_index};
  ImageAccessGPUDesc image_access;
  uint payload_descriptor_index = kInvalidIndex;
  if (image_access_try_load_pixel_payload(access_context, image_index, image_access, payload_descriptor_index) == false) {
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(payload_descriptor_index)];
  ImageFilterSharedAddress sample = image_filter_shared_address(uv, image_access.fsize.xy, image_access.size.xy, image_access.options);

  uint pixel_offset_00 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_01 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);
  uint pixel_offset_10 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_11 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);

  float4 p00 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_00);
  float4 p01 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_01);
  float4 p10 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_10);
  float4 p11 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_11);
  return image_filter_shared_bilinear(p00, p01, p10, p11, sample.dx, sample.dy);
}

float image_evaluate_gpu_row_weight(ImageAccessGPUDesc image_access, float uv_y) {
  if (((image_access.options & Image::UniformSamplingTable) != 0u) || (image_access.size.y == 1u)) {
    return 1.0f;
  }

  return max(0.0f, sin(kPi * saturate(uv_y)));
}

float image_evaluate_gpu_luminance(float3 value) {
  return dot(value, float3(0.212671f, 0.715160f, 0.072169f));
}

bool image_evaluate_gpu_try_rgba(ImageEvaluateGPUContext context, uint image_index, float2 uv, out float image_pdf, out float4 image_value) {
  image_pdf = 0.0f;
  image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);

  ImageAccessGPUContext access_context = {context.images_descriptor_index};
  ImageAccessGPUDesc image_access;
  uint payload_descriptor_index = kInvalidIndex;
  if (image_access_try_load_pixel_payload(access_context, image_index, image_access, payload_descriptor_index) == false) {
    return false;
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(payload_descriptor_index)];
  ImageFilterSharedAddress sample = image_filter_shared_address(uv, image_access.fsize.xy, image_access.size.xy, image_access.options);

  uint pixel_offset_00 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_01 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);
  uint pixel_offset_10 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_11 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);

  float4 p00 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_00);
  float4 p01 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_01);
  float4 p10 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_10);
  float4 p11 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_11);
  image_value = image_filter_shared_bilinear(p00, p01, p10, p11, sample.dx, sample.dy);

  if (image_access.normalization > 0.0f) {
    float3 top_value = (p00 * (1.0f - sample.dx) * (1.0f - sample.dy) + p01 * sample.dx * (1.0f - sample.dy)).xyz;
    float3 bottom_value = (p10 * (1.0f - sample.dx) * sample.dy + p11 * sample.dx * sample.dy).xyz;
    float top_weight = image_evaluate_gpu_row_weight(image_access, uv.y + 0.0f / image_access.fsize.y);
    float bottom_weight = image_evaluate_gpu_row_weight(image_access, uv.y + 1.0f / image_access.fsize.y);
    image_pdf = (image_evaluate_gpu_luminance(top_value) * top_weight + image_evaluate_gpu_luminance(bottom_value) * bottom_weight) / image_access.normalization;
  }

  return true;
}

bool image_evaluate_gpu_try_rgba_no_pdf(ImageEvaluateGPUContext context, uint image_index, float2 uv, out float4 image_value) {
  image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);

  ImageAccessGPUContext access_context = {context.images_descriptor_index};
  ImageAccessGPUDesc image_access;
  uint payload_descriptor_index = kInvalidIndex;
  if (image_access_try_load_pixel_payload(access_context, image_index, image_access, payload_descriptor_index) == false) {
    return false;
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(payload_descriptor_index)];
  ImageFilterSharedAddress sample = image_filter_shared_address(uv, image_access.fsize.xy, image_access.size.xy, image_access.options);

  uint pixel_offset_00 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_01 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);
  uint pixel_offset_10 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_11 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);

  float4 p00 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_00);
  float4 p01 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_01);
  float4 p10 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_10);
  float4 p11 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_11);
  image_value = image_filter_shared_bilinear(p00, p01, p10, p11, sample.dx, sample.dy);
  return true;
}

bool image_evaluate_gpu_try_rgba_layer_no_pdf(ImageEvaluateGPUContext context, uint image_index, float2 uv, uint layer, out float4 image_value) {
  image_value = float4(0.0f, 0.0f, 0.0f, 0.0f);
  ImageAccessGPUContext access_context = {context.images_descriptor_index};
  ImageAccessGPUDesc image_access;
  uint payload_descriptor_index = kInvalidIndex;
  if (image_access_try_load_pixel_payload(access_context, image_index, image_access, payload_descriptor_index) == false) {
    return false;
  }

  if ((image_access.format != (uint)Image::Format::RGBA32F) || (image_access.size.z == 0u)) {
    return false;
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(payload_descriptor_index)];
  ImageFilterSharedAddress sample = image_filter_shared_address(uv, image_access.fsize.xy, image_access.size.xy, image_access.options);
  uint slice = min(layer, image_access.size.z - 1u);
  uint row_stride = image_access.size.x;
  uint slice_offset = slice * image_access.size.x * image_access.size.y;
  uint row_offset_0 = sample.row_0 * row_stride;
  uint row_offset_1 = sample.row_1 * row_stride;

  uint pixel_offset_00 = image_access.pixel_data_offset + ((slice_offset + row_offset_0 + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_01 = image_access.pixel_data_offset + ((slice_offset + row_offset_0 + sample.col_1) * image_access.pixel_data_stride);
  uint pixel_offset_10 = image_access.pixel_data_offset + ((slice_offset + row_offset_1 + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_11 = image_access.pixel_data_offset + ((slice_offset + row_offset_1 + sample.col_1) * image_access.pixel_data_stride);

  float4 p00 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_00);
  float4 p01 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_01);
  float4 p10 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_10);
  float4 p11 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_11);
  image_value = image_filter_shared_bilinear(p00, p01, p10, p11, sample.dx, sample.dy);
  return true;
}

bool image_evaluate_gpu_try_rgba_3d(ImageEvaluateGPUContext context, uint image_index, float3 uvw, out float4 image_value) {
  image_value = float4(0.0f, 0.0f, 0.0f, 0.0f);
  ImageAccessGPUContext access_context = {context.images_descriptor_index};
  ImageAccessGPUDesc image_access;
  uint payload_descriptor_index = kInvalidIndex;
  if (image_access_try_load_pixel_payload(access_context, image_index, image_access, payload_descriptor_index) == false) {
    return false;
  }

  if ((image_access.format != (uint)Image::Format::RGBA32F) || (image_access.size.z == 0u)) {
    return false;
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(payload_descriptor_index)];
  ImageFilterSharedAddress3D sample = image_filter_shared_address_3d(uvw, image_access.fsize, image_access.size, image_access.options);
  uint row_stride = image_access.size.x;
  uint slice_stride = image_access.size.x * image_access.size.y;
  uint slice_offset_0 = sample.slice_0 * slice_stride;
  uint slice_offset_1 = sample.slice_1 * slice_stride;
  uint row_offset_0 = sample.row_0 * row_stride;
  uint row_offset_1 = sample.row_1 * row_stride;

  uint pixel_offset_000 = image_access.pixel_data_offset + ((slice_offset_0 + row_offset_0 + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_001 = image_access.pixel_data_offset + ((slice_offset_0 + row_offset_0 + sample.col_1) * image_access.pixel_data_stride);
  uint pixel_offset_010 = image_access.pixel_data_offset + ((slice_offset_0 + row_offset_1 + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_011 = image_access.pixel_data_offset + ((slice_offset_0 + row_offset_1 + sample.col_1) * image_access.pixel_data_stride);
  uint pixel_offset_100 = image_access.pixel_data_offset + ((slice_offset_1 + row_offset_0 + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_101 = image_access.pixel_data_offset + ((slice_offset_1 + row_offset_0 + sample.col_1) * image_access.pixel_data_stride);
  uint pixel_offset_110 = image_access.pixel_data_offset + ((slice_offset_1 + row_offset_1 + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_111 = image_access.pixel_data_offset + ((slice_offset_1 + row_offset_1 + sample.col_1) * image_access.pixel_data_stride);

  float4 p000 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_000);
  float4 p001 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_001);
  float4 p010 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_010);
  float4 p011 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_011);
  float4 p100 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_100);
  float4 p101 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_101);
  float4 p110 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_110);
  float4 p111 = image_evaluate_gpu_load_pixel_checked(payload_buffer, image_access, pixel_offset_111);
  image_value = image_filter_shared_trilinear(p000, p001, p010, p011, p100, p101, p110, p111, sample.dx, sample.dy, sample.dz);
  return true;
}

float4 image_evaluate_gpu_rgba_3d(ImageEvaluateGPUContext context, uint image_index, float3 uvw) {
  float4 image_value;
  if (image_evaluate_gpu_try_rgba_3d(context, image_index, uvw, image_value) == false) {
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
  }
  return image_value;
}

float image_evaluate_gpu_r32_3d(ImageEvaluateGPUContext context, uint image_index, float3 uvw, float default_value) {
  if (image_index == kInvalidIndex) {
    return default_value;
  }

  float4 image_value;
  if (image_evaluate_gpu_try_rgba_3d(context, image_index, uvw, image_value) == false) {
    return default_value;
  }
  return image_value.x;
}

bool image_evaluate_try_rgba(ImageEvaluateGPUContext context, uint image_index, float2 uv, out float image_pdf, out float4 image_value) {
  return image_evaluate_gpu_try_rgba(context, image_index, uv, image_pdf, image_value);
}

bool image_evaluate_try_rgba_no_pdf(ImageEvaluateGPUContext context, uint image_index, float2 uv, out float4 image_value) {
  return image_evaluate_gpu_try_rgba_no_pdf(context, image_index, uv, image_value);
}

float4 image_evaluate_sample_whole_or_default(ImageEvaluateGPUContext context, uint image_index, float2 uv, float4 default_value) {
  if (image_index == kInvalidIndex) {
    return default_value;
  }

  float4 image_value;
  if (image_evaluate_try_rgba_no_pdf(context, image_index, uv, image_value) == false) {
    return default_value;
  }

  return default_value * image_value;
}

float image_evaluate_sample_channel_or_default(ImageEvaluateGPUContext context, uint image_index, uint channel, float2 uv, float default_value) {
  if ((image_index == kInvalidIndex) || (channel >= 4u)) {
    return default_value;
  }

  float4 image_value;
  if (image_evaluate_try_rgba_no_pdf(context, image_index, uv, image_value) == false) {
    return default_value;
  }

  if (channel == 0u) {
    return image_value.x;
  }
  if (channel == 1u) {
    return image_value.y;
  }
  if (channel == 2u) {
    return image_value.z;
  }

  return image_value.w;
}

bool image_can_apply(ImageAccessGPUContext context, uint image_index) {
  ImageAccessGPUDesc image_access;
  return image_access_try_load(context, image_index, image_access);
}
