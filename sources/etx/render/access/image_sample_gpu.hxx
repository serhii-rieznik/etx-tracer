#pragma once

#include <access/image_access_gpu.hxx>
#include <access/image_evaluate_gpu.hxx>
#include <access/image_sample_shared.hxx>
#include <interop/distribution.hxx>

struct ImageSampleGPUContext {
  uint images_descriptor_index;
};

ImageSampleGPUContext make_image_sample_gpu_context(uint images_descriptor_index) {
  ImageSampleGPUContext result;
  result.images_descriptor_index = images_descriptor_index;
  return result;
}

DistributionEntry image_sample_gpu_load_distribution_entry(ByteAddressBuffer payload_buffer, uint byte_offset) {
  uint4 raw = payload_buffer.Load4(byte_offset);
  DistributionEntry result;
  result.value = asfloat(raw.x);
  result.pdf = asfloat(raw.y);
  result.cdf = asfloat(raw.z);
  result.reference = raw.w;
  return result;
}

struct ImageSampleGPUDistributionContext {
  ByteAddressBuffer x_payload_buffer;
  ByteAddressBuffer y_payload_buffer;
  uint x_distribution_entries_offset;
  uint y_distribution_entries_offset;
  uint x_entries_stride;
  uint x_distribution_count;
  uint y_count;
  float2 fsize;
};

ImageSampleGPUDistributionContext image_sample_gpu_make_distribution_context(ByteAddressBuffer x_payload_buffer, ByteAddressBuffer y_payload_buffer,
  ImageAccessGPUDesc image_access, uint y_count) {
  ImageSampleGPUDistributionContext context;
  context.x_payload_buffer = x_payload_buffer;
  context.y_payload_buffer = y_payload_buffer;
  context.x_distribution_entries_offset = image_access.x_distribution_entries_offset;
  context.y_distribution_entries_offset = image_access.y_distribution_entries_offset;
  context.x_entries_stride = image_access.x_entries_stride;
  context.x_distribution_count = image_access.x_distribution_count;
  context.y_count = y_count;
  context.fsize = image_access.fsize.xy;
  return context;
}

uint image_sample_gpu_y_count(ImageSampleGPUDistributionContext context) {
  return context.y_count;
}

uint image_sample_gpu_x_count(ImageSampleGPUDistributionContext context, uint y_index) {
  if ((y_index >= context.x_distribution_count) || (context.x_entries_stride == 0u)) {
    return 0u;
  }

  return context.x_entries_stride - 1u;
}

float2 image_sample_gpu_image_fsize(ImageSampleGPUDistributionContext context) {
  return context.fsize;
}

float image_sample_gpu_cdf_y(ImageSampleGPUDistributionContext context, uint y_index) {
  uint y_offset = context.y_distribution_entries_offset + y_index * kDistributionEntryStride;
  DistributionEntry entry = image_sample_gpu_load_distribution_entry(context.y_payload_buffer, y_offset);
  return entry.cdf;
}

float image_sample_gpu_cdf_x(ImageSampleGPUDistributionContext context, uint y_index, uint x_index) {
  uint row_base_offset = context.x_distribution_entries_offset + y_index * context.x_entries_stride * kDistributionEntryStride;
  uint x_offset = row_base_offset + x_index * kDistributionEntryStride;
  DistributionEntry entry = image_sample_gpu_load_distribution_entry(context.x_payload_buffer, x_offset);
  return entry.cdf;
}

uint image_sample_gpu_sample_distribution(ByteAddressBuffer payload_buffer, uint entries_base_offset, uint count, float rnd, out float pdf) {
  if (count == 0u) {
    pdf = 0.0f;
    return kInvalidIndex;
  }

  DistributionSearchRange search = distribution_search_begin(count);
  while (distribution_search_active(search)) {
    uint middle = distribution_search_middle(search);
    float middle_cdf = image_sample_gpu_load_distribution_entry(payload_buffer, entries_base_offset + middle * kDistributionEntryStride).cdf;
    distribution_search_update(search, middle, middle_cdf, rnd);
  }

  pdf = image_sample_gpu_load_distribution_entry(payload_buffer, entries_base_offset + search.begin * kDistributionEntryStride).pdf;
  return search.begin;
}

uint image_sample_gpu_sample_y(inout ImageSampleGPUDistributionContext context, float rnd, out float pdf) {
  return image_sample_gpu_sample_distribution(context.y_payload_buffer, context.y_distribution_entries_offset, context.y_count, rnd, pdf);
}

uint image_sample_gpu_sample_x(inout ImageSampleGPUDistributionContext context, uint y_index, float rnd, out float pdf) {
  uint row_base_offset = context.x_distribution_entries_offset + y_index * context.x_entries_stride * kDistributionEntryStride;
  uint x_count = image_sample_gpu_x_count(context, y_index);
  return image_sample_gpu_sample_distribution(context.x_payload_buffer, row_base_offset, x_count, rnd, pdf);
}

bool image_sample_gpu_distribution_sample(inout ImageSampleGPUDistributionContext context, float2 rnd, out float image_pdf, out uint2 location, out float2 uv) {
  image_pdf = 0.0f;
  location = uint2(0u, 0u);
  uv = rnd;

  uint y_count = image_sample_gpu_y_count(context);
  if (y_count == 0u) {
    return false;
  }

  float y_pdf = 0.0f;
  location.y = image_sample_gpu_sample_y(context, rnd.y, y_pdf);
  if ((location.y == kInvalidIndex) || (location.y >= y_count)) {
    return false;
  }

  uint x_count = image_sample_gpu_x_count(context, location.y);
  if (x_count == 0u) {
    return false;
  }

  float x_pdf = 0.0f;
  location.x = image_sample_gpu_sample_x(context, location.y, rnd.x, x_pdf);
  if ((location.x == kInvalidIndex) || (location.x >= x_count)) {
    return false;
  }

  uint x1_index = min(location.x + 1u, x_count);
  uint y1_index = min(location.y + 1u, y_count);

  float x0_cdf = image_sample_gpu_cdf_x(context, location.y, location.x);
  float x1_cdf = image_sample_gpu_cdf_x(context, location.y, x1_index);
  float y0_cdf = image_sample_gpu_cdf_y(context, location.y);
  float y1_cdf = image_sample_gpu_cdf_y(context, y1_index);

  uv = image_sample_uv_from_distribution(rnd, location, image_sample_gpu_image_fsize(context), x0_cdf, x1_cdf, y0_cdf, y1_cdf);
  image_pdf = x_pdf * y_pdf;
  return true;
}

bool image_sample_try_sample(ImageSampleGPUContext context, uint image_index, float2 rnd, out ImageSampleAccess sample) {
  sample = image_sample_access_default(rnd);

  ImageAccessGPUContext access_context = {context.images_descriptor_index};
  ETX_ZERO_INIT(ImageAccessGPUDesc, image_access);
  uint x_payload_descriptor_index = kInvalidIndex;
  uint y_payload_descriptor_index = kInvalidIndex;
  uint y_count = 0u;
  if (image_access_try_load_distribution_payloads(access_context, image_index, image_access, x_payload_descriptor_index, y_payload_descriptor_index, y_count) == false) {
    return false;
  }

  ByteAddressBuffer x_payload_buffer = bindless_buffers[NonUniformResourceIndex(x_payload_descriptor_index)];
  ByteAddressBuffer y_payload_buffer = bindless_buffers[NonUniformResourceIndex(y_payload_descriptor_index)];
  ImageSampleGPUDistributionContext distribution_context = image_sample_gpu_make_distribution_context(x_payload_buffer, y_payload_buffer, image_access, y_count);

  float distribution_pdf = 0.0f;
  if (image_sample_gpu_distribution_sample(distribution_context, rnd, distribution_pdf, sample.location, sample.uv) == false) {
    return false;
  }
  (void)distribution_pdf;

  ImageEvaluateGPUContext evaluate_context = make_image_evaluate_gpu_context(context.images_descriptor_index);
  return image_evaluate_try_rgba(evaluate_context, image_index, sample.uv, sample.pdf, sample.value);
}
