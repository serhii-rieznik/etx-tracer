#pragma once

namespace etx {

#define ETX_DECLARE_BSDF(Class)                                                                     \
  namespace Class##BSDF {                                                                           \
    ETX_GPU_CODE BSDFSample sample(const BSDFData&, const Scene&, Sampler&);                        \
    ETX_GPU_CODE BSDFEval evaluate(const BSDFData&, const Scene&, Sampler&);                        \
    ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene, Sampler& smp);                 \
    ETX_GPU_CODE bool continue_tracing(const Material&, const float2&, const Scene&, Sampler& smp); \
    ETX_GPU_CODE bool is_delta(const Material&, const float2&, const Scene&, Sampler& smp);         \
  }

ETX_DECLARE_BSDF(Diffuse);
ETX_DECLARE_BSDF(MultiscatteringDiffuse);
ETX_DECLARE_BSDF(Plastic);
ETX_DECLARE_BSDF(Conductor);
ETX_DECLARE_BSDF(MultiscatteringConductor);
ETX_DECLARE_BSDF(Dielectric);
ETX_DECLARE_BSDF(MultiscatteringDielectric);
ETX_DECLARE_BSDF(Thinfilm);
ETX_DECLARE_BSDF(Translucent);
ETX_DECLARE_BSDF(Mirror);
ETX_DECLARE_BSDF(Boundary);
ETX_DECLARE_BSDF(Generic);
ETX_DECLARE_BSDF(Coating);
ETX_DECLARE_BSDF(Mixture);

#define CASE_IMPL(CLS, FUNC, ...) \
  case Material::Class::CLS:      \
    return CLS##BSDF::FUNC(__VA_ARGS__)

#define CASE_IMPL_SAMPLE(A) CASE_IMPL(A, sample, data, scene, smp)
#define CASE_IMPL_EVALUATE(A) CASE_IMPL(A, evaluate, data, scene, smp)
#define CASE_IMPL_PDF(A) CASE_IMPL(A, pdf, data, scene, smp)
#define CASE_IMPL_CONTINUE(A) CASE_IMPL(A, continue_tracing, material, tex, scene, smp)
#define CASE_IMPL_IS_DELTA(A) CASE_IMPL(A, is_delta, material, tex, scene, smp)

#define ALL_CASES(MACRO, CLS)               \
  switch (CLS) {                            \
    MACRO(Diffuse);                         \
    MACRO(MultiscatteringDiffuse);          \
    MACRO(Plastic);                         \
    MACRO(Conductor);                       \
    MACRO(MultiscatteringConductor);        \
    MACRO(Dielectric);                      \
    MACRO(MultiscatteringDielectric);       \
    MACRO(Thinfilm);                        \
    MACRO(Translucent);                     \
    MACRO(Mirror);                          \
    MACRO(Boundary);                        \
    MACRO(Generic);                         \
    MACRO(Coating);                         \
    MACRO(Mixture);                         \
    default:                                \
      ETX_FAIL("Unhandled material class"); \
      return {};                            \
  }

namespace bsdf {
[[nodiscard]] ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Scene& scene, Sampler& smp) {
  ALL_CASES(CASE_IMPL_SAMPLE, data.material.cls);
};
[[nodiscard]] ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene, Sampler& smp) {
  ALL_CASES(CASE_IMPL_EVALUATE, data.material.cls);
}

[[nodiscard]] ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene, Sampler& smp) {
  ALL_CASES(CASE_IMPL_PDF, data.material.cls);
}

[[nodiscard]] ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  ALL_CASES(CASE_IMPL_CONTINUE, material.cls);
}

[[nodiscard]] ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  ALL_CASES(CASE_IMPL_IS_DELTA, material.cls);
}

#undef CASE_IMPL

ETX_GPU_CODE SpectralResponse apply_image(SpectralQuery spect, const SpectralResponse& value, uint32_t image_index, const float2& uv, const Scene& scene) {
  SpectralResponse result = value;

  if (image_index != kInvalidIndex) {
    float4 eval = scene.images[image_index].evaluate(uv);
    result *= rgb::query_spd(spect, {eval.x, eval.y, eval.z}, scene.spectrums->rgb_reflection);
    ETX_VALIDATE(result);
  }
  return result;
}

}  // namespace bsdf
}  // namespace etx

#include <etx/render/shared/bsdf_external.hxx>
#include <etx/render/shared/bsdf_conductor.hxx>
#include <etx/render/shared/bsdf_dielectric.hxx>
#include <etx/render/shared/bsdf_generic.hxx>
#include <etx/render/shared/bsdf_plastic.hxx>
#include <etx/render/shared/bsdf_various.hxx>
