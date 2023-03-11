namespace etx {
namespace subsurface {

constexpr uint32_t kIntersectionDirections = 3u;
constexpr uint32_t kIntersectionsPerDirection = 8u;
constexpr uint32_t kTotalIntersections = kIntersectionDirections * kIntersectionsPerDirection;

struct Gather {
  Intersection intersections[kTotalIntersections] = {};
  SpectralResponse weights[kTotalIntersections] = {};
  uint32_t intersection_count = 0u;
  uint32_t selected_intersection = 0u;
  float selected_sample_weight = 0.0f;
  float total_weight = 0.0f;
};

ETX_GPU_CODE float sample_s_r(float rnd) {
  if (rnd < 0.25f) {
    rnd = fminf(4.0f * rnd, 1.0f - kEpsilon);
    return logf(1.0f / (1.0f - rnd));
  }

  rnd = fminf((rnd - 0.25f) / 0.75f, 1.0f - kEpsilon);
  return 3.0f * logf(1.0f / (1.0f - rnd));
}

ETX_GPU_CODE SpectralResponse evaluate(const SpectralQuery spect, const SubsurfaceMaterial& m, float radius) {
  auto sd = m.scattering_distance(spect) * m.scattering_distance_scale;
  ETX_VALIDATE(sd);

  radius = fmaxf(radius, kEpsilon);

  auto term_0 = exp(-radius / (3.0f * sd));
  ETX_VALIDATE(term_0);

  auto term_1 = term_0 * term_0 * term_0;
  ETX_VALIDATE(term_1);

  auto div = sd * (4.0f * radius * kDoublePi);
  ETX_VALIDATE(div);
  div.components = max(div.components, float3{kEpsilon, kEpsilon, kEpsilon});

  return (term_0 + term_1) / div;
}

struct Sample {
  Ray ray;
  float3 u = {};
  float3 v = {};
  float3 w = {};
  float3 basis_prob = {};
  float sampled_radius = 0.0f;

  bool operator()() const {
    return dot(basis_prob, basis_prob) > 0.0f;
  }
};

ETX_GPU_CODE Sample sample(const Vertex& data, const SubsurfaceMaterial& mtl, const uint32_t direction, Sampler& smp) {
  float scattering_distance = mtl.scattering_distance_scale * mtl.scattering_distance.random_entry_power(smp.next());
  if (scattering_distance == 0.0f)
    return {};

  Sample result = {};
  switch (direction) {
    case 0: {
      result.u = data.tan;
      result.v = data.btn;
      result.w = data.nrm;
      result.basis_prob = {0.25f, 0.25f, 0.5f};
      break;
    }
    case 1: {
      result.u = data.btn;
      result.v = data.nrm;
      result.w = data.tan;
      result.basis_prob = {0.25f, 0.50f, 0.25f};
      break;
    }
    case 2: {
      result.u = data.nrm;
      result.v = data.tan;
      result.w = data.btn;
      result.basis_prob = {0.5f, 0.25f, 0.25f};
      break;
    }
    default:
      ETX_FAIL("Invalid direction");
  }

  constexpr float kMaxRadius = 47.827155457397595950044717258511f;
  float r_max = scattering_distance * kMaxRadius;
  result.sampled_radius = scattering_distance * sample_s_r(smp.next());
  if (result.sampled_radius >= r_max)
    return {};

  float phi = kDoublePi * smp.next();
  float height = sqrtf(sqr(r_max) - sqr(result.sampled_radius));
  if (height <= kRayEpsilon)
    return {};

  result.ray.o = data.pos + height * result.w + result.sampled_radius * (cosf(phi) * result.u + sinf(phi) * result.v);
  result.ray.d = -result.w;
  result.ray.max_t = 2.0f * height;
  return result;
}

ETX_GPU_CODE float geometric_weigth(const float3& nrm, const Sample& smp) {
  float pdf_t = smp.basis_prob.x * fabsf(dot(nrm, smp.u));
  float pdf_b = smp.basis_prob.y * fabsf(dot(nrm, smp.v));
  float pdf_n = smp.basis_prob.z * fabsf(dot(nrm, smp.w));
  return sqr(pdf_n) / (sqr(pdf_t) + sqr(pdf_b) + sqr(pdf_n));
}

}  // namespace subsurface
}  // namespace etx
