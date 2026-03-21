#pragma once

#include <etx/render/access/image_access_cpu.hxx>
#include <etx/render/access/image_evaluate_cpu.hxx>
#include <etx/render/access/spectrum_access_cpu.hxx>

namespace etx {
struct Scene;
}

struct BSDFResourceContext {
  const etx::Scene* scene = nullptr;
};

ETX_SHARED_INLINE BSDFResourceContext make_bsdf_resource_cpu_context(const etx::Scene& scene) {
  BSDFResourceContext result = {};
  result.scene = &scene;
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_resource_load_spectrum(
  ETX_IN(BSDFResourceContext, context), uint32_t spectrum_index, ETX_IN(SpectralQuery, spect)) {
  if ((context.scene == nullptr) || (spectrum_index == kInvalidIndex) || (spectrum_index >= context.scene->spectrums.count)) {
    return spectral_response_make(spect, 0.0f);
  }

  etx::SpectrumAccessCPUContext spectrum_context =
    etx::make_spectrum_access_cpu_context(reinterpret_cast<const ::SpectralDistribution*>(context.scene->spectrums.a), static_cast<uint32_t>(context.scene->spectrums.count));
  return etx::spectrum_access_evaluate(spectrum_context, spectrum_index, spect);
}

ETX_SHARED_INLINE bool bsdf_resource_image_has_alpha(ETX_IN(BSDFResourceContext, context), uint32_t image_index) {
  if (context.scene == nullptr) {
    return false;
  }

  etx::ImageAccessCPUContext image_access = etx::make_image_access_cpu_context(*context.scene);
  return etx::image_access_has_alpha(image_access, image_index);
}

ETX_SHARED_INLINE bool bsdf_resource_image_try_evaluate_rgba(
  ETX_IN(BSDFResourceContext, context), uint32_t image_index, ETX_IN(float2, uv), ETX_OUT(float, image_pdf), ETX_OUT(float4, image_value)) {
  image_pdf = 0.0f;
  image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  if (context.scene == nullptr) {
    return false;
  }

  etx::ImageEvaluateCPUContext image_context = etx::make_image_evaluate_cpu_context(*context.scene);
  return etx::image_evaluate_try_rgba(image_context, image_index, uv, image_pdf, image_value);
}

ETX_SHARED_INLINE float bsdf_resource_image_sample_channel_or_default(
  ETX_IN(BSDFResourceContext, context), uint32_t image_index, uint32_t channel, ETX_IN(float2, uv), float default_value) {
  if (context.scene == nullptr) {
    return default_value;
  }

  etx::ImageEvaluateCPUContext image_context = etx::make_image_evaluate_cpu_context(*context.scene);
  return etx::image_evaluate_sample_channel_or_default(image_context, image_index, channel, uv, default_value);
}
