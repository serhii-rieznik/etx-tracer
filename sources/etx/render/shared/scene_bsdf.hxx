#pragma once

namespace etx {

#define ETX_DECLARE_BSDF(Class)                                                               \
  namespace Class##BSDF {                                                                     \
    ETX_GPU_CODE BSDFSample sample(const BSDFData&, const Material&, const Scene&, Sampler&); \
    ETX_GPU_CODE BSDFEval evaluate(const BSDFData&, const Material&, const Scene&, Sampler&); \
    ETX_GPU_CODE float pdf(const BSDFData&, const Material&, const Scene&, Sampler&);         \
    ETX_GPU_CODE bool is_delta(const Material&, const float2&, const Scene&, Sampler&);       \
  }

ETX_DECLARE_BSDF(Diffuse);
ETX_DECLARE_BSDF(Plastic);
ETX_DECLARE_BSDF(Conductor);
ETX_DECLARE_BSDF(Dielectric);
ETX_DECLARE_BSDF(Thinfilm);
ETX_DECLARE_BSDF(Translucent);
ETX_DECLARE_BSDF(Mirror);
ETX_DECLARE_BSDF(Boundary);
ETX_DECLARE_BSDF(Generic);
ETX_DECLARE_BSDF(Coating);
ETX_DECLARE_BSDF(Velvet);
ETX_DECLARE_BSDF(Subsurface);

#define CASE_IMPL(CLS, FUNC, ...) \
  case Material::Class::CLS:      \
    return CLS##BSDF::FUNC(__VA_ARGS__)

#define CASE_IMPL_SAMPLE(A) CASE_IMPL(A, sample, data, mtl, scene, smp)
#define CASE_IMPL_EVALUATE(A) CASE_IMPL(A, evaluate, data, mtl, scene, smp)
#define CASE_IMPL_PDF(A) CASE_IMPL(A, pdf, data, mtl, scene, smp)
#define CASE_IMPL_IS_DELTA(A) CASE_IMPL(A, is_delta, mtl, tex, scene, smp)

#define ALL_CASES(MACRO)                    \
  switch (mtl.cls) {                        \
    MACRO(Diffuse);                         \
    MACRO(Plastic);                         \
    MACRO(Conductor);                       \
    MACRO(Dielectric);                      \
    MACRO(Thinfilm);                        \
    MACRO(Translucent);                     \
    MACRO(Mirror);                          \
    MACRO(Boundary);                        \
    MACRO(Generic);                         \
    MACRO(Coating);                         \
    MACRO(Velvet);                          \
    MACRO(Subsurface);                      \
    default:                                \
      ETX_FAIL("Unhandled material class"); \
      return {};                            \
  }

namespace bsdf {

[[nodiscard]] ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::sample(data, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_SAMPLE);
}

[[nodiscard]] ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::evaluate(data, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_EVALUATE);
}

[[nodiscard]] ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::pdf(data, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_PDF);
}

[[nodiscard]] ETX_GPU_CODE bool continue_tracing(const Material& mtl, const float2& tex, const Scene& scene, Sampler& smp) {
  if (mtl.diffuse.image_index == kInvalidIndex) {
    return false;
  }

  const auto& img = scene.images[mtl.diffuse.image_index];
  if ((img.options & Image::HasAlphaChannel) == 0)
    return false;

  return img.evaluate(tex).w <= smp.next();
}

[[nodiscard]] ETX_GPU_CODE bool is_delta(const Material& mtl, const float2& tex, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::is_delta(mtl, tex, scene, smp);
#endif
  ALL_CASES(CASE_IMPL_IS_DELTA);
}

#undef CASE_IMPL
}  // namespace bsdf

ETX_GPU_CODE SpectralResponse apply_image(SpectralQuery spect, const SpectralImage& img, const float2& uv, const Scene& scene) {
  SpectralResponse result = img.spectrum(spect);

  if (img.image_index != kInvalidIndex) {
    float4 eval = scene.images[img.image_index].evaluate(uv);
    result *= rgb::query_spd(spect, {eval.x, eval.y, eval.z}, scene.spectrums->rgb_reflection);
    ETX_VALIDATE(result);
  }
  return result;
}

ETX_GPU_CODE Thinfilm::Eval evaluate_thinfilm(SpectralQuery spect, const Thinfilm& film, const float2& uv, const Scene& scene) {
  if (film.max_thickness * film.min_thickness <= 0.0f) {
    return {{}, 0.0f};
  }

  float t = (film.thinkness_image == kInvalidIndex) ? 1.0f : scene.images[film.thinkness_image].evaluate(uv).x;
  float thickness = lerp(film.min_thickness, film.max_thickness, t);
  return {film.ior(spect), thickness};
}

ETX_GPU_CODE SpectralResponse apply_emitter_image(SpectralQuery spect, const SpectralImage& img, const float2& uv, const Scene& scene) {
  auto result = img.spectrum(spect);
  ETX_VALIDATE(result);

  if (img.image_index != kInvalidIndex) {
    float4 eval = scene.images[img.image_index].evaluate(uv);
    ETX_VALIDATE(eval);
    auto scale = rgb::query_spd(spect, {eval.x, eval.y, eval.z}, scene.spectrums->rgb_illuminant);
    ETX_VALIDATE(result);
    result *= scale;
    ETX_VALIDATE(result);
  }
  return result;
}

}  // namespace etx

#include <etx/render/shared/bsdf_external.hxx>
#include <etx/render/shared/bsdf_conductor.hxx>
#include <etx/render/shared/bsdf_dielectric.hxx>
#include <etx/render/shared/bsdf_generic.hxx>
#include <etx/render/shared/bsdf_plastic.hxx>
#include <etx/render/shared/bsdf_various.hxx>
#include <etx/render/shared/bsdf_velvet.hxx>

#include <etx/render/shared/bssrdf_subsurface.hxx>
