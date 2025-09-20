namespace etx {

namespace PrincipledBSDF {

struct PrincipledMaterial {
  SpectralImage scattering;
  SpectralImage reflectance;
  SampledImage roughness;
  SampledImage metalness;
  SampledImage transmission;
};

#define WOMP_DEBUG_PRINCIPLED_BSDF        0
#define WOMP_DEBUG_PRINCIPLED_BSDF_ENTITY metalness

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& in_mtl, const Scene& scene, Sampler& smp) {
  auto m_local = in_mtl;
  auto metalness = evaluate_metalness(m_local, data.tex, scene);

#if (WOMP_DEBUG_PRINCIPLED_BSDF)
  auto roughness = evaluate_roughness(m_local, data.tex, scene).x;

  auto frame = data.get_normal_frame();
  auto local_w_o = sample_cosine_distribution(smp.next_2d(), 1.0f);
  BSDFSample result = {};
  result.eta = 1.0f;
  result.properties = BSDFSample::Reflection | BSDFSample::Diffuse;
  result.medium_index = in_mtl.ext_medium;
  result.pdf = max(0.0f, local_w_o.z);
  result.weight = {data.spectrum_sample, WOMP_DEBUG_PRINCIPLED_BSDF_ENTITY};
  result.w_o = frame.from_local(local_w_o);
  return result;
#endif

  if (smp.next() < metalness) {
    m_local.int_ior.cls = SpectralDistribution::Class::Conductor;
    m_local.int_ior.eta_index = scene.default_conductor_eta;
    m_local.int_ior.k_index = scene.default_conductor_k;
    m_local.scattering.image_index = kInvalidIndex;
    return ConductorBSDF::sample(data, m_local, scene, smp);
  } else {
    m_local.int_ior.cls = SpectralDistribution::Class::Dielectric;
    m_local.int_ior.eta_index = scene.default_dielectric_eta;
    m_local.int_ior.k_index = kInvalidIndex;
    m_local.reflectance.image_index = kInvalidIndex;
    if (smp.next() < m_local.transmission.value.x) {
      return DielectricBSDF::sample(data, m_local, scene, smp);
    } else {
      return PlasticBSDF::sample(data, m_local, scene, smp);
    }
  }
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& in_mtl, const Scene& scene, Sampler& smp) {
  auto m_local = in_mtl;
  auto metalness = evaluate_metalness(m_local, data.tex, scene);

#if (WOMP_DEBUG_PRINCIPLED_BSDF)
  auto roughness = evaluate_roughness(m_local, data.tex, scene).x;
  float o_dot_n = max(0.0f, dot(w_o, data.nrm) * kInvPi);

  BSDFEval eval = {};
  eval.eta = 1.0f;
  eval.func = {data.spectrum_sample, WOMP_DEBUG_PRINCIPLED_BSDF_ENTITY};
  eval.bsdf = eval.func * o_dot_n;
  eval.pdf = o_dot_n;
  eval.weight = eval.pdf > 0.0f ? eval.bsdf / eval.pdf : SpectralResponse{data.spectrum_sample, 0.0f};
  return eval;
#endif

  if (smp.next() < metalness) {
    m_local.int_ior.cls = SpectralDistribution::Class::Conductor;
    m_local.int_ior.eta_index = scene.default_conductor_eta;
    m_local.int_ior.k_index = scene.default_conductor_k;
    m_local.scattering.image_index = kInvalidIndex;
    return ConductorBSDF::evaluate(data, w_o, m_local, scene, smp);
  } else {
    m_local.int_ior.cls = SpectralDistribution::Class::Dielectric;
    m_local.int_ior.eta_index = scene.default_dielectric_eta;
    m_local.int_ior.k_index = kInvalidIndex;
    m_local.reflectance.image_index = kInvalidIndex;
    if (smp.next() < m_local.transmission.value.x) {
      return DielectricBSDF::evaluate(data, w_o, m_local, scene, smp);
    } else {
      return PlasticBSDF::evaluate(data, w_o, m_local, scene, smp);
    }
  }
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& w_o, const Material& in_mtl, const Scene& scene, Sampler& smp) {
#if (WOMP_DEBUG_PRINCIPLED_BSDF)
  return max(0.0f, dot(data.front_fracing_normal(), w_o) * kInvPi);
#endif

  auto m_local = in_mtl;
  auto metalness = evaluate_metalness(m_local, data.tex, scene);
  if (smp.next() < metalness) {
    m_local.int_ior.cls = SpectralDistribution::Class::Conductor;
    m_local.int_ior.eta_index = scene.default_conductor_eta;
    m_local.int_ior.k_index = scene.default_conductor_k;
    m_local.scattering.image_index = kInvalidIndex;
    return ConductorBSDF::pdf(data, w_o, m_local, scene, smp);
  } else {
    m_local.int_ior.cls = SpectralDistribution::Class::Dielectric;
    m_local.int_ior.eta_index = scene.default_dielectric_eta;
    m_local.int_ior.k_index = kInvalidIndex;
    m_local.reflectance.image_index = kInvalidIndex;
    if (smp.next() < m_local.transmission.value.x) {
      return DielectricBSDF::pdf(data, w_o, m_local, scene, smp);
    } else {
      return PlasticBSDF::pdf(data, w_o, m_local, scene, smp);
    }
  }
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return apply_image(data.spectrum_sample, mtl.scattering, data.tex, scene, nullptr);
}

}  // namespace PrincipledBSDF
}  // namespace etx
