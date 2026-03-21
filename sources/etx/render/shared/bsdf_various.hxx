namespace etx {

namespace VoidBSDF {

ETX_SHARED_INLINE BSDFSample sample(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return bsdf::detail::sample_interop(data, mtl, smp);
}

ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  return bsdf::detail::evaluate_interop(data, w_o, mtl, smp);
}

ETX_SHARED_INLINE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  return bsdf::detail::pdf_interop(data, w_o, mtl, smp);
}

ETX_SHARED_INLINE bool is_delta(const Material& mtl, const float2& tex, Sampler& smp) {
  return bsdf::detail::is_delta_interop(mtl, tex, smp);
}

ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return bsdf::detail::albedo_interop(data, mtl, smp);
}

}  // namespace VoidBSDF

namespace DiffuseBSDF {

ETX_SHARED_INLINE BSDFEval diffuse_layer(const BSDFData& data, const float3& local_w_i, const float3& local_w_o, const Material& mtl, Sampler& smp) {
  BSDFResourceContext context = bsdf::detail::make_interop_context();
  ::Sampler interop_sampler = bsdf::detail::make_interop_sampler(smp);
  ::BSDFEval result = ::bsdf_diffuse_layer(context, bsdf::detail::make_interop_data(data), local_w_i, local_w_o, mtl, interop_sampler);
  bsdf::detail::copy_interop_sampler_back(interop_sampler, smp);
  return bsdf::detail::make_public_eval(result);
}

ETX_SHARED_INLINE BSDFSample sample(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return bsdf::detail::sample_interop(data, mtl, smp);
}

ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  return bsdf::detail::evaluate_interop(data, w_o, mtl, smp);
}

ETX_SHARED_INLINE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  return bsdf::detail::pdf_interop(data, w_o, mtl, smp);
}

ETX_SHARED_INLINE bool is_delta(const Material& mtl, const float2& tex, Sampler& smp) {
  return bsdf::detail::is_delta_interop(mtl, tex, smp);
}

ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return bsdf::detail::albedo_interop(data, mtl, smp);
}

}  // namespace DiffuseBSDF

namespace TranslucentBSDF {

ETX_SHARED_INLINE BSDFSample sample(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return bsdf::detail::sample_interop(data, mtl, smp);
}

ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  return bsdf::detail::evaluate_interop(data, w_o, mtl, smp);
}

ETX_SHARED_INLINE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  return bsdf::detail::pdf_interop(data, w_o, mtl, smp);
}

ETX_SHARED_INLINE bool is_delta(const Material& mtl, const float2& tex, Sampler& smp) {
  return bsdf::detail::is_delta_interop(mtl, tex, smp);
}

ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return bsdf::detail::albedo_interop(data, mtl, smp);
}

}  // namespace TranslucentBSDF

namespace MirrorBSDF {

ETX_SHARED_INLINE BSDFSample sample(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return bsdf::detail::sample_interop(data, mtl, smp);
}

ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  return bsdf::detail::evaluate_interop(data, w_o, mtl, smp);
}

ETX_SHARED_INLINE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  return bsdf::detail::pdf_interop(data, w_o, mtl, smp);
}

ETX_SHARED_INLINE bool is_delta(const Material& mtl, const float2& tex, Sampler& smp) {
  return bsdf::detail::is_delta_interop(mtl, tex, smp);
}

ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return bsdf::detail::albedo_interop(data, mtl, smp);
}

}  // namespace MirrorBSDF

namespace BoundaryBSDF {

ETX_SHARED_INLINE BSDFSample sample(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return bsdf::detail::sample_interop(data, mtl, smp);
}

ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  return bsdf::detail::evaluate_interop(data, w_o, mtl, smp);
}

ETX_SHARED_INLINE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  return bsdf::detail::pdf_interop(data, w_o, mtl, smp);
}

ETX_SHARED_INLINE bool is_delta(const Material& mtl, const float2& tex, Sampler& smp) {
  return bsdf::detail::is_delta_interop(mtl, tex, smp);
}

ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return bsdf::detail::albedo_interop(data, mtl, smp);
}

}  // namespace BoundaryBSDF

}  // namespace etx
