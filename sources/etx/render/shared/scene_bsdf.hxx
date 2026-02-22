#pragma once
namespace etx {

#define ETX_DECLARE_BSDF(Class)                                                                                       \
  namespace Class##BSDF {                                                                                             \
    ETX_SHARED_INLINE BSDFSample sample(const BSDFData&, const Material&, const Scene&, Sampler&);                    \
    ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData&, const float3& w_o, const Material&, const Scene&, Sampler&); \
    ETX_SHARED_INLINE float pdf(const BSDFData&, const float3& w_o, const Material&, const Scene&, Sampler&);         \
    ETX_SHARED_INLINE bool is_delta(const Material&, const float2&, const Scene&, Sampler&);                          \
    ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData&, const Material&, const Scene&, Sampler&);              \
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
ETX_DECLARE_BSDF(Void);

#define CASE_IMPL(CLS, FUNC, ...) \
  case MaterialClass::CLS:        \
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
    MACRO(Void);                            \
    default:                                \
      ETX_FAIL("Unhandled material class"); \
      return {};                            \
  }

namespace bsdf {

[[nodiscard]] ETX_SHARED_INLINE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::sample(data, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_SAMPLE);
}

[[nodiscard]] ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::evaluate(data, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_EVALUATE);
}

[[nodiscard]] ETX_SHARED_INLINE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::pdf(data, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_PDF);
}

[[nodiscard]] ETX_SHARED_INLINE float reverse_pdf(const BSDFData& in_data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  float3 w_o = -in_data.w_i;
  BSDFData data = in_data;
  data.w_i = -in_w_o;

#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::pdf(data, w_o, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_PDF);
}

[[nodiscard]] ETX_SHARED_INLINE bool is_delta(const Material& mtl, const float2& tex, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::is_delta(mtl, tex, scene, smp);
#endif
  ALL_CASES(CASE_IMPL_IS_DELTA);
}

[[nodiscard]] ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::albedo(data, mtl, scene, smp);
#endif

  ALL_CASES(CASE_IMPL_ALBEDO);
}

#undef CASE_IMPL
}  // namespace bsdf

ETX_SHARED_INLINE ThinFilmEval evaluate_thinfilm(SpectralQuery spect, const Thinfilm& film, const float2& uv, const Scene& scene, Sampler& smp) {
  if (film.max_thickness * film.min_thickness <= 0.0f) {
    return {{}, 0.0f};
  }

  float t = (film.thinkness_image == kInvalidIndex) ? 1.0f : scene.images[film.thinkness_image].evaluate(uv, nullptr).x;
  float thickness = lerp(film.min_thickness, film.max_thickness, t);

  float3 wavelengths = {spect.wavelength, spect.wavelength, spect.wavelength};
  if (spect.spectral() == false) {
    wavelengths.x = kRGBWavelengths.x + kRGBWavelengthsSpan.x * (2.0f * smp.next() - 1.0f);
    wavelengths.y = kRGBWavelengths.y + kRGBWavelengthsSpan.y * (2.0f * smp.next() - 1.0f);
    wavelengths.z = kRGBWavelengths.z + kRGBWavelengthsSpan.z * (2.0f * smp.next() - 1.0f);
  }

  return {evaluate_refractive_index(scene, film.ior, spect), wavelengths, thickness};
}

struct AlphaTestSharedCPUContext {
  const Material& material;
  const Triangle& triangle;
  const Scene& scene;
  Sampler& sampler;
  float3 barycentric;
};

ETX_SHARED_INLINE uint32_t alpha_test_shared_cpu_material_class(ETX_IN(AlphaTestSharedCPUContext, context)) {
  return context.material.cls;
}

ETX_SHARED_INLINE float alpha_test_shared_cpu_material_opacity(ETX_IN(AlphaTestSharedCPUContext, context)) {
  return context.material.opacity;
}

ETX_SHARED_INLINE uint32_t alpha_test_shared_cpu_scattering_image_index(ETX_IN(AlphaTestSharedCPUContext, context)) {
  return context.material.scattering.image_index;
}

ETX_SHARED_INLINE bool alpha_test_shared_cpu_image_has_alpha(ETX_IN(AlphaTestSharedCPUContext, context), uint32_t image_index) {
  ImageSceneAccessCPUContext access_context = {context.scene};
  return image_scene_access_shared_has_alpha(access_context, image_index);
}

ETX_SHARED_INLINE float alpha_test_shared_cpu_evaluate_alpha(ETX_IN(AlphaTestSharedCPUContext, context), uint32_t image_index) {
  ImageSceneAccessCPUContext access_context = {context.scene};
  ImageSceneAccessCPUDesc image_access = {};
  if (image_scene_access_shared_try_load_desc(access_context, image_index, image_access) == false) {
    return 1.0f;
  }

  float2 uv = lerp_uv(context.scene, context.triangle, context.barycentric);
  return context.scene.images[image_access.image_index].evaluate_alpha(uv);
}

ETX_SHARED_INLINE float alpha_test_shared_cpu_rnd(ETX_INOUT(AlphaTestSharedCPUContext, context)) {
  return context.sampler.next();
}

#define ETX_ALPHA_TEST_SHARED_CONTEXT_TYPE AlphaTestSharedCPUContext
#define ETX_ALPHA_TEST_SHARED_MATERIAL_CLASS(context) alpha_test_shared_cpu_material_class(context)
#define ETX_ALPHA_TEST_SHARED_MATERIAL_OPACITY(context) alpha_test_shared_cpu_material_opacity(context)
#define ETX_ALPHA_TEST_SHARED_SCATTERING_IMAGE_INDEX(context) alpha_test_shared_cpu_scattering_image_index(context)
#define ETX_ALPHA_TEST_SHARED_IMAGE_HAS_ALPHA(context, image_index) alpha_test_shared_cpu_image_has_alpha(context, image_index)
#define ETX_ALPHA_TEST_SHARED_EVALUATE_ALPHA(context, image_index) alpha_test_shared_cpu_evaluate_alpha(context, image_index)
#define ETX_ALPHA_TEST_SHARED_RND(context) alpha_test_shared_cpu_rnd(context)
#include <etx/render/interop/alpha_test_shared.hxx>
#undef ETX_ALPHA_TEST_SHARED_RND
#undef ETX_ALPHA_TEST_SHARED_EVALUATE_ALPHA
#undef ETX_ALPHA_TEST_SHARED_IMAGE_HAS_ALPHA
#undef ETX_ALPHA_TEST_SHARED_SCATTERING_IMAGE_INDEX
#undef ETX_ALPHA_TEST_SHARED_MATERIAL_OPACITY
#undef ETX_ALPHA_TEST_SHARED_MATERIAL_CLASS
#undef ETX_ALPHA_TEST_SHARED_CONTEXT_TYPE

ETX_SHARED_INLINE bool alpha_test_pass(const Material& mat, const Triangle& t, const float3& bc, const Scene& scene, Sampler& smp) {
  AlphaTestSharedCPUContext context = {mat, t, scene, smp, bc};
  return alpha_test_shared_pass(context);
}

}  // namespace etx

#include <etx/render/shared/bsdf_external.hxx>
#include <etx/render/shared/bsdf_various.hxx>
#include <etx/render/shared/bsdf_plastic.hxx>
#include <etx/render/shared/bsdf_conductor.hxx>
#include <etx/render/shared/bsdf_dielectric.hxx>
#include <etx/render/shared/bsdf_velvet.hxx>
#include <etx/render/shared/bsdf_principled.hxx>
