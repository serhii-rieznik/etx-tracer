#pragma once

#include <etx/render/access/bsdf_resource_cpu.hxx>
#include <etx/render/host/scene_global.hxx>
#include <etx/render/interop/bsdf_dispatch_shared.hxx>

namespace etx {

#define ETX_DECLARE_BSDF(Class)                                                                         \
  namespace Class##BSDF {                                                                               \
    ETX_SHARED_INLINE BSDFSample sample(const BSDFData&, const Material&, Sampler&);                    \
    ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData&, const float3& w_o, const Material&, Sampler&); \
    ETX_SHARED_INLINE float pdf(const BSDFData&, const float3& w_o, const Material&, Sampler&);         \
    ETX_SHARED_INLINE bool is_delta(const Material&, const float2&, Sampler&);                          \
    ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData&, const Material&, Sampler&);              \
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
ETX_DECLARE_BSDF(OpenPBR)
ETX_DECLARE_BSDF(Void);

#define CASE_IMPL(CLS, FUNC, ...) \
  case MaterialClass::CLS:        \
    return CLS##BSDF::FUNC(__VA_ARGS__)

#define CASE_IMPL_SAMPLE(A)   CASE_IMPL(A, sample, data, mtl, smp)
#define CASE_IMPL_EVALUATE(A) CASE_IMPL(A, evaluate, data, w_o, mtl, smp)
#define CASE_IMPL_PDF(A)      CASE_IMPL(A, pdf, data, w_o, mtl, smp)
#define CASE_IMPL_IS_DELTA(A) CASE_IMPL(A, is_delta, mtl, tex, smp)
#define CASE_IMPL_ALBEDO(A)   CASE_IMPL(A, albedo, data, mtl, smp)

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
    MACRO(OpenPBR);                         \
    MACRO(Void);                            \
    default:                                \
      ETX_FAIL("Unhandled material class"); \
      return {};                            \
  }

namespace bsdf {

namespace detail {

ETX_SHARED_INLINE bool interop_supported(const Material& mtl) {
  switch (mtl.cls) {
    case MaterialClass::Diffuse:
    case MaterialClass::Translucent:
    case MaterialClass::Conductor:
    case MaterialClass::Dielectric:
    case MaterialClass::Plastic:
    case MaterialClass::Thinfilm:
    case MaterialClass::Mirror:
    case MaterialClass::Boundary:
    case MaterialClass::Velvet:
    case MaterialClass::Void:
    case MaterialClass::DiffractionGrating: {
      return true;
    }

    default: {
      return false;
    }
  }
}

ETX_SHARED_INLINE ::Sampler make_interop_sampler(const Sampler& smp) {
  ::Sampler result = {};
  result.seed = smp.seed;
  result.fixed_u = smp.fixed_u;
  result.fixed_v = smp.fixed_v;
  result.fixed_w = smp.fixed_w;
  return result;
}

ETX_SHARED_INLINE void copy_interop_sampler_back(const ::Sampler& source, Sampler& target) {
  target.seed = source.seed;
  target.fixed_u = source.fixed_u;
  target.fixed_v = source.fixed_v;
  target.fixed_w = source.fixed_w;
}

ETX_SHARED_INLINE ::BSDFData make_interop_data(const BSDFData& data) {
  ::BSDFData result = {};
  result.pos = data.pos;
  result.nrm = data.nrm;
  result.tan = data.tan;
  result.btn = data.btn;
  result.tex = data.tex;
  result.w_i = data.w_i;
  result.spectrum_sample = static_cast<const ::SpectralQuery&>(data.spectrum_sample);
  result.path_source = static_cast<uint32_t>(data.path_source);
  result.current_medium = data.current_medium;
  return result;
}

ETX_SHARED_INLINE SpectralResponse make_public_response(const ::SpectralResponse& value) {
  SpectralResponse result = {};
  static_cast<::SpectralResponse&>(result) = value;
  return result;
}

ETX_SHARED_INLINE BSDFEval make_public_eval(const ::BSDFEval& value) {
  BSDFEval result = {};
  result.func = make_public_response(value.func);
  result.bsdf = make_public_response(value.bsdf);
  result.pdf = value.pdf;
  result.eta = value.eta;
  result.properties = value.properties;
  result.medium_index = value.medium_index;
  return result;
}

ETX_SHARED_INLINE BSDFSample make_public_sample(const ::BSDFSample& value) {
  BSDFSample result = {};
  result.weight = make_public_response(value.weight);
  result.w_o = value.w_o;
  result.pdf = value.pdf;
  result.eta = value.eta;
  result.properties = value.properties;
  result.medium_index = value.medium_index;
  result.pad = value.pad;
  return result;
}

ETX_SHARED_INLINE BSDFResourceContext make_interop_context() {
  const Scene& scene = scene_global_get();
  BSDFResourceContext result = make_bsdf_resource_cpu_context(scene);
  return result;
}

[[nodiscard]] ETX_SHARED_INLINE BSDFSample sample_interop(const BSDFData& data, const Material& mtl, Sampler& smp) {
  BSDFResourceContext context = make_interop_context();
  ::Sampler interop_sampler = make_interop_sampler(smp);
  ::BSDFSample result = ::bsdf_sample(context, make_interop_data(data), mtl, interop_sampler);
  copy_interop_sampler_back(interop_sampler, smp);
  return make_public_sample(result);
}

[[nodiscard]] ETX_SHARED_INLINE BSDFEval evaluate_interop(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  BSDFResourceContext context = make_interop_context();
  ::Sampler interop_sampler = make_interop_sampler(smp);
  ::BSDFEval result = ::bsdf_evaluate(context, make_interop_data(data), w_o, mtl, interop_sampler);
  copy_interop_sampler_back(interop_sampler, smp);
  return make_public_eval(result);
}

[[nodiscard]] ETX_SHARED_INLINE float pdf_interop(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  BSDFResourceContext context = make_interop_context();
  ::Sampler interop_sampler = make_interop_sampler(smp);
  float result = ::bsdf_pdf(context, make_interop_data(data), w_o, mtl, interop_sampler);
  copy_interop_sampler_back(interop_sampler, smp);
  return result;
}

[[nodiscard]] ETX_SHARED_INLINE float reverse_pdf_interop(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  BSDFResourceContext context = make_interop_context();
  ::Sampler interop_sampler = make_interop_sampler(smp);
  float result = ::bsdf_reverse_pdf(context, make_interop_data(data), w_o, mtl, interop_sampler);
  copy_interop_sampler_back(interop_sampler, smp);
  return result;
}

[[nodiscard]] ETX_SHARED_INLINE bool is_delta_interop(const Material& mtl, const float2& tex, Sampler& smp) {
  BSDFResourceContext context = make_interop_context();
  ::Sampler interop_sampler = make_interop_sampler(smp);
  bool result = ::bsdf_is_delta_with_context(context, mtl, tex, interop_sampler);
  copy_interop_sampler_back(interop_sampler, smp);
  return result;
}

[[nodiscard]] ETX_SHARED_INLINE SpectralResponse albedo_interop(const BSDFData& data, const Material& mtl, Sampler& smp) {
  BSDFResourceContext context = make_interop_context();
  ::Sampler interop_sampler = make_interop_sampler(smp);
  ::SpectralResponse result = ::bsdf_albedo(context, make_interop_data(data), mtl, interop_sampler);
  copy_interop_sampler_back(interop_sampler, smp);
  return make_public_response(result);
}

}  // namespace detail

[[nodiscard]] ETX_SHARED_INLINE BSDFSample sample(const BSDFData& data, const Material& mtl, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::sample(data, mtl, smp);
#endif

  if (detail::interop_supported(mtl)) {
    return detail::sample_interop(data, mtl, smp);
  }

  ALL_CASES(CASE_IMPL_SAMPLE);
}

[[nodiscard]] ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::evaluate(data, mtl, smp);
#endif

  if (detail::interop_supported(mtl)) {
    return detail::evaluate_interop(data, w_o, mtl, smp);
  }

  ALL_CASES(CASE_IMPL_EVALUATE);
}

[[nodiscard]] ETX_SHARED_INLINE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::pdf(data, w_o, mtl, smp);
#endif

  if (detail::interop_supported(mtl)) {
    return detail::pdf_interop(data, w_o, mtl, smp);
  }

  ALL_CASES(CASE_IMPL_PDF);
}

[[nodiscard]] ETX_SHARED_INLINE float reverse_pdf(const BSDFData& in_data, const float3& in_w_o, const Material& mtl, Sampler& smp) {
  if (detail::interop_supported(mtl)) {
    return detail::reverse_pdf_interop(in_data, in_w_o, mtl, smp);
  }

  float3 w_o = -in_data.w_i;
  BSDFData data = in_data;
  data.w_i = -in_w_o;
  if (in_data.path_source == PathSource::Camera) {
    data.path_source = PathSource::Light;
  } else if (in_data.path_source == PathSource::Light) {
    data.path_source = PathSource::Camera;
  }

#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::pdf(data, w_o, mtl, smp);
#endif

  ALL_CASES(CASE_IMPL_PDF);
}

[[nodiscard]] ETX_SHARED_INLINE bool is_delta(const Material& mtl, const float2& tex, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::is_delta(mtl, tex, smp);
#endif

  if (detail::interop_supported(mtl)) {
    return detail::is_delta_interop(mtl, tex, smp);
  }

  ALL_CASES(CASE_IMPL_IS_DELTA);
}

[[nodiscard]] ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData& data, const Material& mtl, Sampler& smp) {
#if defined(ETX_FORCED_BSDF)
  return ETX_FORCED_BSDF::albedo(data, mtl, smp);
#endif

  if (detail::interop_supported(mtl)) {
    return detail::albedo_interop(data, mtl, smp);
  }

  ALL_CASES(CASE_IMPL_ALBEDO);
}

#undef CASE_IMPL
}  // namespace bsdf

ETX_SHARED_INLINE ThinFilmEval evaluate_thinfilm(SpectralQuery spect, const Thinfilm& film, const float2& uv, Sampler& smp) {
  BSDFResourceContext context = make_bsdf_resource_cpu_context(scene_global_get());
  ::Sampler interop_sampler = bsdf::detail::make_interop_sampler(smp);
  ThinFilmEval result = ::bsdf_resource_evaluate_thinfilm(context, static_cast<const ::SpectralQuery&>(spect), film, uv, interop_sampler);
  bsdf::detail::copy_interop_sampler_back(interop_sampler, smp);
  return result;
}

ETX_SHARED_INLINE bool alpha_test_pass(const Material& mat, const float2& uv, Sampler& smp) {
  BSDFResourceContext context = make_bsdf_resource_cpu_context(scene_global_get());
  ::Sampler interop_sampler = bsdf::detail::make_interop_sampler(smp);
  bool result = ::bsdf_alpha_test_pass(context, mat, uv, interop_sampler);
  bsdf::detail::copy_interop_sampler_back(interop_sampler, smp);
  return result;
}

}  // namespace etx

#include <etx/render/shared/bsdf_various.hxx>
#include <etx/render/shared/bsdf_plastic.hxx>
#include <etx/render/shared/bsdf_conductor.hxx>
#include <etx/render/shared/bsdf_dielectric.hxx>
#include <etx/render/shared/bsdf_velvet.hxx>
#include <etx/render/shared/bsdf_openpbr.hxx>
