namespace etx {
namespace subsurface {

constexpr uint32_t kMaxIntersections = 8u;

struct ETX_ALIGNED BSSRDFData {
  SpectralQuery spect;
  Vertex p_i;
  float3 w_i;
  Vertex p_o;
  float3 w_o;
};

struct ETX_ALIGNED BSSRDFSample {
  float3 u = {};
  float radius = 0.0f;
  float3 v = {};
  float phi = 0.0f;
  float3 w = {};
  float height = 0.0f;
};

ETX_GPU_CODE float fresnel_moment_1(float eta) {
  float eta2 = eta * eta;
  float eta3 = eta2 * eta;
  float eta4 = eta3 * eta;
  float eta5 = eta4 * eta;
  if (eta < 1.0f)
    return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945f * eta3 + 2.49277f * eta4 - 0.68441f * eta5;

  return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 - 1.27198f * eta4 + 0.12746f * eta5;
}

ETX_GPU_CODE float fresnel_moment_2(float eta) {
  float eta2 = eta * eta;
  float eta3 = eta2 * eta;
  float eta4 = eta3 * eta;
  float eta5 = eta4 * eta;
  if (eta < 1.0f) {
    return 0.27614f - 0.87350f * eta + 1.12077f * eta2 - 0.65095f * eta3 + 0.07883f * eta4 + 0.04860f * eta5;
  }

  float r_eta = 1.0f / eta, r_eta2 = r_eta * r_eta, r_eta3 = r_eta2 * r_eta;
  return -547.033f + 45.3087f * r_eta3 - 218.725f * r_eta2 + 458.843f * r_eta + 404.557f * eta - 189.519f * eta2 + 54.9327f * eta3 - 9.00603f * eta4 + 0.63942f * eta5;
}

ETX_GPU_CODE SpectralResponse radius_function(const BSSRDFData& data, const Material& mtl, const Scene& scene) {
  // TODO : specialize
  return {data.spect.wavelength, 1.0f};
}

ETX_GPU_CODE SpectralResponse spatial_function(const BSSRDFData& data, const Material& mtl, const Scene& scene) {
  return radius_function(data, mtl, scene);
}

ETX_GPU_CODE SpectralResponse directional_function(const BSSRDFData& data, const Material& mtl, const Scene& scene) {
  auto eta_e = mtl.ext_ior(data.spect);
  auto eta_i = mtl.int_ior(data.spect);
  auto thinfilm = evaluate_thinfilm(data.spect, mtl.thinfilm, data.p_i.tex, scene);
  float eta = eta_e.eta.monochromatic() / eta_i.eta.monochromatic();
  float c = 1.0f - 2.0f * fresnel_moment_1(1.0f / eta);
  return (1.0f - fresnel::dielectric(data.spect, data.w_i, data.p_i.nrm, eta_e, eta_i, thinfilm)) / (c * kPi);
}

ETX_GPU_CODE SpectralResponse evaluate(const BSSRDFData& data, const Material& mtl, const Scene& scene) {
  auto eta_e = mtl.ext_ior(data.spect);
  auto eta_i = mtl.int_ior(data.spect);
  auto thinfilm = evaluate_thinfilm(data.spect, mtl.thinfilm, data.p_o.tex, scene);
  auto f = fresnel::dielectric(data.spect, data.w_o, data.p_o.nrm, eta_e, eta_i, thinfilm);
  return (1.0f - f) * directional_function(data, mtl, scene) * spatial_function(data, mtl, scene);
}

ETX_GPU_CODE float sample_s_r(float rnd) {
  if (rnd < 0.25f) {
    rnd = fminf(4.0f * rnd, 1.0f - kEpsilon);
    return logf(1.0f / (1.0f - rnd));
  }

  rnd = fminf((rnd - 0.25f) / 0.75f, 1.0f - kEpsilon);
  return 3.0f * logf(1.0f / (1.0f - rnd));
}

ETX_GPU_CODE SpectralResponse eval_s_r(const SpectralQuery spect, const SpectralDistribution& scattering_distance, float radius) {
  auto sd = scattering_distance(spect);
  ETX_VALIDATE(sd);

  radius = fmaxf(radius, kEpsilon);

  auto term_0 = exp(-radius / sd);
  ETX_VALIDATE(term_0);

  auto term_1 = exp(-radius / (3.0f * sd));
  ETX_VALIDATE(term_1);

  auto div = 8.0f * kPi * radius * sd;
  ETX_VALIDATE(div);

  auto result = (term_0 + term_1) / div;
  ETX_VALIDATE(result);

  return result;
}

ETX_GPU_CODE float pdf_s_r(float scattering_distance, float radius) {
  radius = fmaxf(radius, kEpsilon);
  return 0.25f * expf(-radius / scattering_distance) / (2.0f * kPi * scattering_distance * radius) +
         0.75f * expf(-radius / (3.0f * scattering_distance)) / (6.0f * kPi * scattering_distance * radius);
}

ETX_GPU_CODE float pdf_s_p(const Vertex& i0, const BSSRDFSample& b0, const Vertex& i1, const SpectralDistribution& scattering) {
  float3 d = i1.pos - i0.pos;
  float3 d_local = {dot(b0.u, d), dot(b0.v, d), dot(b0.w, d)};

  const float r_proj[3] = {
    sqrtf(d_local.y * d_local.y + d_local.z * d_local.z),
    sqrtf(d_local.z * d_local.z + d_local.x * d_local.x),
    sqrtf(d_local.x * d_local.x + d_local.y * d_local.y),
  };
  const float s_pdf = 1.0f / float(scattering.count);
  const float n_local[] = {
    0.25f * fabsf(dot(b0.u, i1.nrm)) * s_pdf,
    0.25f * fabsf(dot(b0.v, i1.nrm)) * s_pdf,
    0.5f * fabsf(dot(b0.w, i1.nrm)) * s_pdf,
  };

  float pdf = 0;
  for (uint64_t axis = 0; axis < 3llu; ++axis) {
    for (uint64_t channel = 0; channel < scattering.count; ++channel) {
      float scattering_distance = scattering.entries[channel].power;
      pdf += pdf_s_r(scattering_distance, r_proj[axis]) * n_local[axis];
    }
  }
  return pdf;
}

ETX_GPU_CODE BSSRDFSample sample_spatial(const Vertex& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  BSSRDFSample sample;
  auto ortho = orthonormal_basis(data.nrm);

  float rnd_0 = smp.next();
  if (rnd_0 < 0.5f) {
    sample.u = ortho.u;
    sample.v = ortho.v;
    sample.w = data.nrm;
  } else if (rnd_0 < 0.75f) {
    sample.u = ortho.v;
    sample.v = data.nrm;
    sample.w = ortho.u;
  } else {
    sample.u = data.nrm;
    sample.v = ortho.u;
    sample.w = ortho.v;
  }

  float scattering_distance = mtl.subsurface.scattering.random_entry_power(smp.next());
  float r_max = scattering_distance * sample_s_r(0.9999f);

  sample.radius = scattering_distance * sample_s_r(smp.next());
  sample.phi = kDoublePi * smp.next();
  sample.height = 2.0f * sqrtf(sqr(r_max) - sqr(sample.radius));
  return sample;
}

ETX_GPU_CODE Ray make_ray(const BSSRDFSample& sample, const float3& p0) {
  Ray ray;
  ray.o = p0 + 0.5f * sample.height * sample.w + sample.radius * (cosf(sample.phi) * sample.u + sinf(sample.phi) * sample.v);
  ray.d = -sample.w;
  ray.max_t = sample.height;
  return ray;
}

}  // namespace subsurface
}  // namespace etx
