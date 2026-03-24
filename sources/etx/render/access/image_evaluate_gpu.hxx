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

  return float4(1.0f, 1.0f, 1.0f, 1.0f);
}

float4 image_evaluate_gpu_rgba(ImageEvaluateGPUContext context, uint image_index, float2 uv) {
  ImageAccessGPUContext access_context = {context.images_descriptor_index};
  ImageAccessGPUDesc image_access;
  uint payload_descriptor_index = kInvalidIndex;
  if (image_access_try_load_pixel_payload(access_context, image_index, image_access, payload_descriptor_index) == false) {
    return float4(1.0f, 1.0f, 1.0f, 1.0f);
  }

  ByteAddressBuffer payload_buffer = bindless_buffers[NonUniformResourceIndex(payload_descriptor_index)];
  ImageFilterSharedAddress sample = image_filter_shared_address(uv, image_access.fsize, image_access.size, image_access.options);

  uint pixel_offset_00 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_01 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);
  uint pixel_offset_10 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_11 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);

  float4 p00 = image_evaluate_gpu_load_pixel(payload_buffer, image_access.format, pixel_offset_00);
  float4 p01 = image_evaluate_gpu_load_pixel(payload_buffer, image_access.format, pixel_offset_01);
  float4 p10 = image_evaluate_gpu_load_pixel(payload_buffer, image_access.format, pixel_offset_10);
  float4 p11 = image_evaluate_gpu_load_pixel(payload_buffer, image_access.format, pixel_offset_11);
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
  ImageFilterSharedAddress sample = image_filter_shared_address(uv, image_access.fsize, image_access.size, image_access.options);

  uint pixel_offset_00 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_01 = image_access.pixel_data_offset + ((sample.row_0 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);
  uint pixel_offset_10 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_0) * image_access.pixel_data_stride);
  uint pixel_offset_11 = image_access.pixel_data_offset + ((sample.row_1 * image_access.size.x + sample.col_1) * image_access.pixel_data_stride);

  float4 p00 = image_evaluate_gpu_load_pixel(payload_buffer, image_access.format, pixel_offset_00);
  float4 p01 = image_evaluate_gpu_load_pixel(payload_buffer, image_access.format, pixel_offset_01);
  float4 p10 = image_evaluate_gpu_load_pixel(payload_buffer, image_access.format, pixel_offset_10);
  float4 p11 = image_evaluate_gpu_load_pixel(payload_buffer, image_access.format, pixel_offset_11);
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

bool image_evaluate_try_rgba(ImageEvaluateGPUContext context, uint image_index, float2 uv, out float image_pdf, out float4 image_value) {
  image_pdf = 0.0f;
  image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  return image_evaluate_gpu_try_rgba(context, image_index, uv, image_pdf, image_value);
}

float4 image_evaluate_sample_whole_or_default(ImageEvaluateGPUContext context, uint image_index, float2 uv, float4 default_value) {
  if (image_index == kInvalidIndex) {
    return default_value;
  }

  float image_pdf = 0.0f;
  float4 image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  if (image_evaluate_try_rgba(context, image_index, uv, image_pdf, image_value) == false) {
    return default_value;
  }

  return default_value * image_value;
}

float image_evaluate_sample_channel_or_default(ImageEvaluateGPUContext context, uint image_index, uint channel, float2 uv, float default_value) {
  if ((image_index == kInvalidIndex) || (channel >= 4u)) {
    return default_value;
  }

  float image_pdf = 0.0f;
  float4 image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  if (image_evaluate_try_rgba(context, image_index, uv, image_pdf, image_value) == false) {
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
