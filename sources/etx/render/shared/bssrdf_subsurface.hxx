namespace etx {
namespace subsurface {

constexpr uint32_t kMaxIntersections = 8u;

struct Gather {
  Intersection intersections[subsurface::kMaxIntersections] = {};
  SpectralResponse weights[subsurface::kMaxIntersections] = {};
  uint32_t intersection_count = 0;
  uint32_t selected_intersection = kInvalidIndex;
  float selected_sample_weight = 0.0f;
};

ETX_GPU_CODE float sample_s_r(float rnd) {
  if (rnd < 0.25f) {
    rnd = fminf(4.0f * rnd, 1.0f - kEpsilon);
    return logf(1.0f / (1.0f - rnd));
  }

  rnd = fminf((rnd - 0.25f) / 0.75f, 1.0f - kEpsilon);
  return 3.0f * logf(1.0f / (1.0f - rnd));
}

ETX_GPU_CODE SpectralResponse eval_s_r(const SpectralQuery spect, const SubsurfaceMaterial& m, float radius) {
  auto sd = m.scattering_distance(spect) * m.scattering_distance_scale;
  ETX_VALIDATE(sd);

  radius = fmaxf(radius, kEpsilon);

  auto term_0 = exp(-radius / sd);
  ETX_VALIDATE(term_0);

  auto term_1 = exp(-radius / (3.0f * sd));
  ETX_VALIDATE(term_1);

  auto div = 8.0f * kPi * radius * sd;
  div.components = max(div.components, float3{kEpsilon, kEpsilon, kEpsilon});
  ETX_VALIDATE(div);

  auto t = term_0 + term_1;
  ETX_VALIDATE(t);

  auto result = t / div;
  ETX_VALIDATE(result);

  return result;
}

ETX_GPU_CODE float pdf_s_r(float scattering_distance, float radius) {
  radius = fmaxf(radius, kEpsilon);
  scattering_distance = fmaxf(scattering_distance, kEpsilon);

  radius = fmaxf(radius, kEpsilon);
  float t0 = 0.25f * expf(-radius / scattering_distance) / (2.0f * kPi * scattering_distance * radius);
  ETX_VALIDATE(t0);
  float t1 = 0.75f * expf(-radius / (3.0f * scattering_distance)) / (6.0f * kPi * scattering_distance * radius);
  ETX_VALIDATE(t1);
  return t0 + t1;
}

ETX_GPU_CODE float pdf_s_p(const Vertex& i0, const Vertex& i1, const SubsurfaceMaterial& m) {
  const float s_pdf = 1.0f / float(m.scattering_distance.count);

  float3 d = i1.pos - i0.pos;
  float3 d_local = {
    dot(i0.tan, d),
    dot(i0.btn, d),
    dot(i0.nrm, d),
  };
  const float n_local[] = {
    0.25f * fabsf(dot(i0.tan, i1.nrm)) * s_pdf,
    0.25f * fabsf(dot(i0.btn, i1.nrm)) * s_pdf,
    0.50f * fabsf(dot(i0.nrm, i1.nrm)) * s_pdf,
  };

  const float r_proj[3] = {
    sqrtf(sqr(d_local.y) + sqr(d_local.z)),
    sqrtf(sqr(d_local.x) + sqr(d_local.z)),
    sqrtf(sqr(d_local.x) + sqr(d_local.y)),
  };

  float pdf = 0;
  for (uint64_t axis = 0; axis < 3llu; ++axis) {
    for (uint64_t channel = 0; channel < m.scattering_distance.count; ++channel) {
      float scattering_distance = m.scattering_distance_scale * m.scattering_distance.entries[channel].power;
      pdf += pdf_s_r(scattering_distance, r_proj[axis]) * n_local[axis];
    }
  }
  return pdf;
}

ETX_GPU_CODE bool sample(const Vertex& data, const Material& mtl, Sampler& smp, Ray& ray) {
  float scattering_distance = mtl.subsurface.scattering_distance_scale * mtl.subsurface.scattering_distance.random_entry_power(smp.next());
  if (scattering_distance == 0.0f)
    return false;

  float3 u, v, w;
  float rnd_0 = smp.next();
  if (rnd_0 <= 0.5f) {
    u = data.tan;
    v = data.btn;
    w = data.nrm;
  } else if (rnd_0 < 0.75f) {
    u = data.btn;
    v = data.nrm;
    w = data.tan;
  } else {
    u = data.nrm;
    v = data.tan;
    w = data.btn;
  }

  float r_max = scattering_distance * sample_s_r(1.0f);
  float radius = scattering_distance * sample_s_r(smp.next());
  if (radius >= r_max)
    return false;

  float phi = kDoublePi * smp.next();
  float height = 2.0f * sqrtf(sqr(r_max) - sqr(radius));
  if (height <= kRayEpsilon)
    return false;

  ray.o = data.pos + 0.5f * height * w + radius * (cosf(phi) * u + sinf(phi) * v);
  ray.d = -w;
  ray.max_t = height;
  return true;
}

}  // namespace subsurface
}  // namespace etx
