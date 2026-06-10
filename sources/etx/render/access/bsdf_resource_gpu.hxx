#pragma once

#include <access/image_access_gpu.hxx>
#include <access/image_evaluate_gpu.hxx>
#include <access/spectrum_access_gpu.hxx>
#include <interop/scene_gpu_access_shared.hxx>

struct BSDFResourceContext {
  uint images_descriptor_index;
  uint spectrums_descriptor_index;
};

BSDFResourceContext make_bsdf_resource_gpu_context(uint images_descriptor_index, uint spectrums_descriptor_index) {
  BSDFResourceContext result;
  result.images_descriptor_index = images_descriptor_index;
  result.spectrums_descriptor_index = spectrums_descriptor_index;
  return result;
}

SpectralResponse bsdf_resource_load_spectrum(BSDFResourceContext context, uint spectrum_index, SpectralQuery spect) {
  if ((scene_gpu_has_descriptor(context.spectrums_descriptor_index) == false) || (spectrum_index == kInvalidIndex)) {
    return spectral_response_make(spect, 0.0f);
  }

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(context.spectrums_descriptor_index)];
  SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, context.spectrums_descriptor_index);
  return spectrum_access_evaluate(spectrum_context, spectrum_index, spect);
}

bool bsdf_resource_image_has_alpha(BSDFResourceContext context, uint image_index) {
  if (scene_gpu_has_descriptor(context.images_descriptor_index) == false) {
    return false;
  }

  ImageAccessGPUContext image_context = {context.images_descriptor_index};
  return image_access_has_alpha(image_context, image_index);
}

bool bsdf_resource_image_try_evaluate_rgba(BSDFResourceContext context, uint image_index, float2 uv, out float image_pdf, out float4 image_value) {
  if (scene_gpu_has_descriptor(context.images_descriptor_index) == false) {
    image_pdf = 0.0f;
    image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
    return false;
  }

  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(context.images_descriptor_index);
  return image_evaluate_try_rgba(image_context, image_index, uv, image_pdf, image_value);
}

bool bsdf_resource_image_try_evaluate_rgba_no_pdf(BSDFResourceContext context, uint image_index, float2 uv, out float4 image_value) {
  if (scene_gpu_has_descriptor(context.images_descriptor_index) == false) {
    image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
    return false;
  }

  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(context.images_descriptor_index);
  return image_evaluate_try_rgba_no_pdf(image_context, image_index, uv, image_value);
}

float bsdf_resource_image_sample_channel_or_default(BSDFResourceContext context, uint image_index, uint channel, float2 uv, float default_value) {
  if (scene_gpu_has_descriptor(context.images_descriptor_index) == false) {
    return default_value;
  }

  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(context.images_descriptor_index);
  return image_evaluate_sample_channel_or_default(image_context, image_index, channel, uv, default_value);
}
