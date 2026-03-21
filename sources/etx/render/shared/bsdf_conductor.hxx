namespace etx {

namespace ConductorBSDF {

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

}  // namespace ConductorBSDF

}  // namespace etx
