#pragma once

namespace etx {

#define ETX_DECLARE_BSDF(Class)                                                                                  \
  namespace Class##BSDF {                                                                                        \
    ETX_GPU_CODE BSDFSample sample(const BSDFData&, const Material&, const Scene&, Sampler&);                    \
    ETX_GPU_CODE BSDFEval evaluate(const BSDFData&, const float3& w_o, const Material&, const Scene&, Sampler&); \
    ETX_GPU_CODE float pdf(const BSDFData&, const float3& w_o, const Material&, const Scene&, Sampler&);         \
    ETX_GPU_CODE bool is_delta(const Material&, const float2&, const Scene&, Sampler&);                          \
    ETX_GPU_CODE SpectralResponse albedo(const BSDFData&, const Material&, const Scene&, Sampler&);              \
  }

ETX_DECLARE_BSDF(Diffuse);
ETX_DECLARE_BSDF(Translucent);
ETX_DECLARE_BSDF(Plastic);
ETX_DECLARE_BSDF(Conductor);
ETX_DECLARE_BSDF(Dielectric);
ETX_DECLARE_BSDF(Thinfilm);
ETX_DECLARE_BSDF(Mirror);
ETX_DECLARE_BSDF(Boundary);
ETX_DECLARE_BSDF(Velvet);
ETX_DECLARE_BSDF(Principled)

#define CASE_IMPL(CLS, FUNC, ...) \
  case Material::Class::CLS:      \
    return CLS##BSDF::FUNC(__VA_ARGS__)

#define CASE_IMPL_SAMPLE(A)   CASE_IMPL(A, sample, data, mtl, scene, smp)
#define CASE_IMPL_EVALUATE(A) CASE_IMPL(A, evaluate, data, w_o, mtl, scene, smp)
#define CASE_IMPL_PDF(A)      CASE_IMPL(A, pdf, data, w_o, mtl, scene, smp)
#define CASE_IMPL_IS_DELTA(A) CASE_IMPL(A, is_delta, mtl, tex, scene, smp)
#define CASE_IMPL_ALBEDO(A)   CASE_IMPL(A, albedo, data, mtl, scene, smp)

#define ALL_CASES(MACRO)                    \
  switch (mtl.cls) {                        \
    MACRO(Diffuse);                         \
    MACRO(Translucent);                     \
    MACRO(Plastic);                         \
    MACRO(Conductor);                       \
    MACRO(Dielectric);                      \
    MACRO(Thinfilm);                        \
    MACRO(Mirror);                          \
    MACRO(Boundary);                        \
    MACRO(Velvet);                          \
    MACRO(Principled);                      \
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

[[nodiscard]] ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::evaluate(data, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_EVALUATE);
}

[[nodiscard]] ETX_GPU_CODE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::pdf(data, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_PDF);
}

[[nodiscard]] ETX_GPU_CODE float reverse_pdf(const BSDFData& in_data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  float3 w_o = -in_data.w_i;
  BSDFData data = in_data;
  data.w_i = -in_w_o;

#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::pdf(data, w_o, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_PDF);
}

[[nodiscard]] ETX_GPU_CODE bool is_delta(const Material& mtl, const float2& tex, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::is_delta(mtl, tex, scene, smp);
#endif
  ALL_CASES(CASE_IMPL_IS_DELTA);
}

[[nodiscard]] ETX_GPU_CODE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::albedo(data, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_ALBEDO);
}

#undef CASE_IMPL
}  // namespace bsdf

ETX_GPU_CODE Thinfilm::Eval evaluate_thinfilm(SpectralQuery spect, const Thinfilm& film, const float2& uv, const Scene& scene, Sampler& smp) {
  if (film.max_thickness * film.min_thickness <= 0.0f) {
    return {{}, 0.0f};
  }

  float t = (film.thinkness_image == kInvalidIndex) ? 1.0f : scene.images[film.thinkness_image].evaluate(uv, nullptr).x;
  float thickness = lerp(film.min_thickness, film.max_thickness, t);

  float3 wavelengths = {spect.wavelength, spect.wavelength, spect.wavelength};
  if (spect.spectral() == false) {
    wavelengths.x = Thinfilm::kRGBWavelengths.x + Thinfilm::kRGBWavelengthsSpan.x * (2.0f * smp.next() - 1.0f);
    wavelengths.y = Thinfilm::kRGBWavelengths.y + Thinfilm::kRGBWavelengthsSpan.y * (2.0f * smp.next() - 1.0f);
    wavelengths.z = Thinfilm::kRGBWavelengths.z + Thinfilm::kRGBWavelengthsSpan.z * (2.0f * smp.next() - 1.0f);
  }

  return {evaluate_refractive_index(scene, film.ior, spect), wavelengths, thickness};
}

ETX_GPU_CODE bool alpha_test_pass(const Material& mat, const Triangle& t, const float3& bc, const Scene& scene, Sampler& smp) {
  if (mat.scattering.image_index == kInvalidIndex)
    return false;

  auto uv = lerp_uv(scene.vertices, t, bc);
  const auto& img = scene.images[mat.scattering.image_index];
  return (img.options & Image::HasAlphaChannel) && (img.evaluate_alpha(uv) <= smp.next());
}

}  // namespace etx

#include <etx/render/shared/bsdf_external.hxx>
#include <etx/render/shared/bsdf_various.hxx>
#include <etx/render/shared/bsdf_plastic.hxx>
#include <etx/render/shared/bsdf_conductor.hxx>
#include <etx/render/shared/bsdf_dielectric.hxx>
#include <etx/render/shared/bsdf_velvet.hxx>
#include <etx/render/shared/bsdf_principled.hxx>
