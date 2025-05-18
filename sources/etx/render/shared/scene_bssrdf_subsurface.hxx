namespace etx {
namespace subsurface {

constexpr uint32_t kIntersectionDirections = 3u;
constexpr uint32_t kIntersectionsPerDirection = 8u;
constexpr uint32_t kTotalIntersections = kIntersectionDirections * kIntersectionsPerDirection;

struct Gather {
  Intersection intersections[kTotalIntersections];
  SpectralResponse weights[kTotalIntersections];
  uint32_t intersection_count;
  uint32_t selected_intersection;
  float selected_sample_weight;
  float total_weight;
};

ETX_GPU_CODE void remap_channel(float color, const float scattering_distances, float& albedo, float& extinction, float& scattering) {
  constexpr float a = 1.826052378200f;
  constexpr float b = 4.985111943850f + 0.12735595943800f;
  constexpr float c = 1.096861024240f;
  constexpr float d = 0.496310210422f;
  constexpr float e = 4.231902997010f + 0.00310603949088f;
  constexpr float f = 2.406029994080f;
  constexpr float kMinScattering = 1.0f / 1024.0f;

  color = max(color, 0.0f);

  float blend = powf(color, 0.25f);
  albedo = saturate((1.0f - blend) * a * powf(atanf(b * color), c) + blend * d * powf(atanf(e * color), f));
  ETX_VALIDATE(albedo);

  extinction = 1.0f / fmaxf(scattering_distances, kMinScattering);
  ETX_VALIDATE(extinction);

  scattering = extinction * albedo;
  ETX_VALIDATE(scattering);
}

ETX_GPU_CODE void remap(float3 color, const float3& scattering_distances, float3& albedo, float3& extinction, float3& scattering) {
  remap_channel(color.x, scattering_distances.x, albedo.x, extinction.x, scattering.x);
  remap_channel(color.y, scattering_distances.y, albedo.y, extinction.y, scattering.y);
  remap_channel(color.z, scattering_distances.z, albedo.z, extinction.z, scattering.z);
}

ETX_GPU_CODE float sample_s_r(float rnd) {
  if (rnd < 0.25f) {
    rnd = fminf(4.0f * rnd, 1.0f - kEpsilon);
    return logf(1.0f / (1.0f - rnd));
  }

  rnd = fminf((rnd - 0.25f) / 0.75f, 1.0f - kEpsilon);
  return 3.0f * logf(1.0f / (1.0f - rnd));
}

ETX_GPU_CODE SpectralResponse evaluate(const SpectralQuery spect, const Scene& scene, const Intersection& data, const SubsurfaceMaterial& m, float radius) {
  auto sd = apply_image(spect, m, data.tex, scene, nullptr) * m.scale;
  ETX_VALIDATE(sd);

  radius = fmaxf(radius, kEpsilon);

  auto term_0 = exp(-radius / (3.0f * sd));
  ETX_VALIDATE(term_0);

  auto term_1 = term_0 * term_0 * term_0;
  ETX_VALIDATE(term_1);

  auto div = sd * (4.0f * radius * kDoublePi);
  ETX_VALIDATE(div);
  div.integrated = max(div.integrated, float3{kEpsilon, kEpsilon, kEpsilon});
  div.value = max(div.value, kEpsilon);

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

ETX_GPU_CODE Sample sample(SpectralQuery spect, const Scene& scene, const Vertex& data, const SubsurfaceMaterial& mtl, const uint32_t direction, Sampler& smp) {
  SpectralResponse sampled_distance = apply_image(spect, mtl, data.tex, scene, nullptr);
  uint32_t channel = uint32_t(sampled_distance.component_count() * smp.next());
  float scattering_distance = mtl.scale * sampled_distance.component(channel);
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
