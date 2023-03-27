#pragma once

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/scene.hxx>

namespace etx {

struct ETX_ALIGNED PTOptions {
  uint32_t iterations ETX_INIT_WITH(128u);
  uint32_t max_depth ETX_INIT_WITH(65536u);
  uint32_t rr_start ETX_INIT_WITH(6u);
  uint32_t path_per_iteration ETX_INIT_WITH(1u);
  bool nee ETX_INIT_WITH(true);
  bool mis ETX_INIT_WITH(true);
};

struct ETX_ALIGNED PTRayPayload {
  Ray ray = {};
  SpectralResponse throughput = {spectrum::kUndefinedWavelength, 1.0f};
  SpectralResponse accumulated = {spectrum::kUndefinedWavelength, 0.0f};
  uint32_t index = kInvalidIndex;
  uint32_t medium = kInvalidIndex;
  uint32_t path_length = 0u;
  uint32_t iteration = 0u;
  SpectralQuery spect = {};
  float eta = 1.0f;
  float sampled_bsdf_pdf = 0.0f;
  float2 uv = {};
  Sampler smp = {};
  bool mis_weight = true;
};

enum PTRayState : uint8_t {
  IntersectionFound,
  NoIntersection,
  ContinueIteration,
  EndIteration,
  Finished,
};

struct PTRayPayloadSoA {
  ArrayView<Ray> ray = {};
  ArrayView<Intersection> intersection = {};
  ArrayView<PTRayState> ray_state = {};
  ArrayView<SpectralQuery> spect = {};
  ArrayView<uint32_t> iteration = {};
  ArrayView<uint32_t> medium = {};
  ArrayView<SpectralResponse> throughput = {};
  ArrayView<SpectralResponse> accumulated = {};
  ArrayView<Sampler> smp = {};
  ArrayView<uint8_t> mis_weight = {};
  ArrayView<uint32_t> path_length = {};
  ArrayView<float> sampled_bsdf_pdf = {};
  ArrayView<float> eta = {};
  ArrayView<subsurface::Gather> ss_gather = {};
  ArrayView<BSDFSample> bsdf_sample = {};
  ArrayView<uint8_t> subsurface_sampled = {};

  PTRayPayloadSoA() = default;
  PTRayPayloadSoA(const PTRayPayloadSoA&) = delete;
  PTRayPayloadSoA& operator=(const PTRayPayloadSoA&) = delete;
};

ETX_GPU_CODE PTRayPayload make_ray_payload(const Scene& scene, uint2 px, uint2 dim, uint32_t iteration) {
  PTRayPayload payload = {};
  payload.index = px.x + px.y * dim.x;
  payload.iteration = iteration;
  payload.smp.init(payload.index, payload.iteration);
  payload.spect = spectrum::sample(payload.smp.next());
  payload.uv = get_jittered_uv(payload.smp, px, dim);
  payload.ray = generate_ray(payload.smp, scene, payload.uv);
  payload.throughput = {payload.spect.wavelength, 1.0f};
  payload.accumulated = {payload.spect.wavelength, 0.0f};
  payload.medium = scene.camera_medium_index;
  payload.path_length = 1;
  payload.eta = 1.0f;
  payload.sampled_bsdf_pdf = 0.0f;
  payload.mis_weight = true;
  return payload;
}

ETX_GPU_CODE void make_ray_payload(const Scene& scene, uint2 dim, uint32_t iteration, PTRayPayloadSoA& payload, uint32_t index) {
  Sampler smp(index, iteration);
  SpectralQuery spect = spectrum::sample(smp.next());
  float2 uv = get_jittered_uv(smp, {index % dim.x, index / dim.x}, dim);
  payload.spect[index] = spect;
  payload.ray[index] = generate_ray(smp, scene, uv);
  payload.throughput[index] = {spect.wavelength, 1.0f};
  payload.accumulated[index] = {spect.wavelength, 0.0f};
  payload.medium[index] = scene.camera_medium_index;
  payload.smp[index] = smp;
  payload.path_length[index] = 1;
  payload.eta[index] = 1.0f;
  payload.sampled_bsdf_pdf[index] = 0.0f;
  payload.mis_weight[index] = 1u;
  payload.ray_state[index] = PTRayState::ContinueIteration;
  payload.iteration[index] = iteration;
}

ETX_GPU_CODE Medium::Sample try_sampling_medium(const Scene& scene, PTRayPayload& payload, float max_t) {
  if (payload.medium == kInvalidIndex) {
    return {};
  }

  auto medium_sample = scene.mediums[payload.medium].sample(payload.spect, payload.smp, payload.ray.o, payload.ray.d, max_t);
  payload.throughput *= medium_sample.weight;
  ETX_VALIDATE(payload.throughput);
  return medium_sample;
}

ETX_GPU_CODE Medium::Sample try_sampling_medium(const Scene& scene, PTRayPayloadSoA& payload, uint32_t i) {
  if (payload.medium[i] == kInvalidIndex) {
    return {};
  }

  auto medium_sample = scene.mediums[payload.medium[i]].sample(payload.spect[i], payload.smp[i], payload.ray[i].o, payload.ray[i].d, payload.intersection[i].t);
  payload.throughput[i] *= medium_sample.weight;
  ETX_VALIDATE(payload.throughput[i]);
  return medium_sample;
}

ETX_GPU_CODE void handle_sampled_medium(const Scene& scene, const Medium::Sample& medium_sample, uint32_t max_depth, const Raytracing& rt, PTRayPayload& payload) {
  const auto& medium = scene.mediums[payload.medium];
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   * direct light sampling from medium
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  if (payload.path_length + 1 <= max_depth) {
    uint32_t emitter_index = sample_emitter_index(scene, payload.smp);
    auto emitter_sample = sample_emitter(payload.spect, emitter_index, payload.smp, medium_sample.pos, scene);
    if (emitter_sample.pdf_dir > 0) {
      auto tr = rt.trace_transmittance(payload.spect, scene, medium_sample.pos, emitter_sample.origin, payload.medium, payload.smp);
      float phase_function = medium.phase_function(payload.spect, medium_sample.pos, payload.ray.d, emitter_sample.direction);
      auto weight = emitter_sample.is_delta ? 1.0f : power_heuristic(emitter_sample.pdf_dir * emitter_sample.pdf_sample, phase_function);
      payload.accumulated += payload.throughput * emitter_sample.value * tr * (phase_function * weight / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));
      ETX_VALIDATE(payload.accumulated);
    }
  }

  float3 w_o = medium.sample_phase_function(payload.spect, payload.smp, medium_sample.pos, payload.ray.d);
  payload.sampled_bsdf_pdf = medium.phase_function(payload.spect, medium_sample.pos, payload.ray.d, w_o);
  payload.mis_weight = true;
  payload.ray.o = medium_sample.pos;
  payload.ray.d = w_o;
  payload.path_length += 1;
  ETX_CHECK_FINITE(payload.ray.d);
}

ETX_GPU_CODE void handle_sampled_medium(const Scene& scene, const Medium::Sample& medium_sample, uint32_t max_depth, const Raytracing& rt, PTRayPayloadSoA& payload, uint32_t i) {
  const auto& medium = scene.mediums[payload.medium[i]];
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   * direct light sampling from medium
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  if (payload.path_length[i] + 1 <= max_depth) {
    uint32_t emitter_index = sample_emitter_index(scene, payload.smp[i]);
    auto emitter_sample = sample_emitter(payload.spect[i], emitter_index, payload.smp[i], medium_sample.pos, scene);
    if (emitter_sample.pdf_dir > 0) {
      auto tr = rt.trace_transmittance(payload.spect[i], scene, medium_sample.pos, emitter_sample.origin, payload.medium[i], payload.smp[i]);
      float phase_function = medium.phase_function(payload.spect[i], medium_sample.pos, payload.ray[i].d, emitter_sample.direction);
      auto weight = emitter_sample.is_delta ? 1.0f : power_heuristic(emitter_sample.pdf_dir * emitter_sample.pdf_sample, phase_function);
      payload.accumulated[i] += payload.throughput[i] * emitter_sample.value * tr * (phase_function * weight / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));
      ETX_VALIDATE(payload.accumulated[i]);
    }
  }

  float3 w_o = medium.sample_phase_function(payload.spect[i], payload.smp[i], medium_sample.pos, payload.ray[i].d);
  payload.sampled_bsdf_pdf[i] = medium.phase_function(payload.spect[i], medium_sample.pos, payload.ray[i].d, w_o);
  payload.mis_weight[i] = true;
  payload.ray[i].o = medium_sample.pos;
  payload.ray[i].d = w_o;
  payload.path_length[i] += 1;
  ETX_CHECK_FINITE(payload.ray[i].d);
}

ETX_GPU_CODE SpectralResponse evaluate_light(const Scene& scene, const Intersection& intersection, const Raytracing& rt, const Material& mat, const uint32_t medium,
  const SpectralQuery spect, const EmitterSample& emitter_sample, Sampler& smp, bool mis) {
  if (emitter_sample.pdf_dir == 0.0f) {
    return {spect.wavelength, 0.0f};
  }

  BSDFEval bsdf_eval = bsdf::evaluate({spect, medium, PathSource::Camera, intersection, intersection.w_i}, emitter_sample.direction, mat, scene, smp);
  if (bsdf_eval.valid() == false) {
    return {spect.wavelength, 0.0f};
  }

  ETX_VALIDATE(bsdf_eval.bsdf);

  const auto& tri = scene.triangles[intersection.triangle_index];
  auto pos = shading_pos(scene.vertices, tri, intersection.barycentric, emitter_sample.direction);
  auto tr = rt.trace_transmittance(spect, scene, pos, emitter_sample.origin, medium, smp);
  ETX_VALIDATE(tr);

  bool no_weight = (mis == false) || emitter_sample.is_delta;
  auto weight = no_weight ? 1.0f : power_heuristic(emitter_sample.pdf_dir * emitter_sample.pdf_sample, bsdf_eval.pdf);
  ETX_VALIDATE(weight);

  return bsdf_eval.bsdf * emitter_sample.value * tr * (weight / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));
}

ETX_GPU_CODE void handle_direct_emitter(const Scene& scene, const Triangle& tri, const Intersection& intersection, const Raytracing& rt, const bool mis, PTRayPayload& payload) {
  if (intersection.emitter_index == kInvalidIndex)
    return;

  const auto& emitter = scene.emitters[intersection.emitter_index];
  float pdf_emitter_area = 0.0f;
  float pdf_emitter_dir = 0.0f;
  float pdf_emitter_dir_out = 0.0f;
  auto e = emitter_get_radiance(emitter, payload.spect, intersection.tex, payload.ray.o, intersection.pos, pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene,
    (payload.path_length == 0));

  if (pdf_emitter_dir > 0.0f) {
    auto tr = rt.trace_transmittance(payload.spect, scene, payload.ray.o, intersection.pos, payload.medium, payload.smp);
    float pdf_emitter_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
    bool no_weight = (mis == false) || (payload.path_length == 1) || (payload.mis_weight == false);
    auto weight = no_weight ? 1.0f : power_heuristic(payload.sampled_bsdf_pdf, pdf_emitter_discrete * pdf_emitter_dir);
    payload.accumulated += payload.throughput * e * tr * weight;
    ETX_VALIDATE(payload.accumulated);
  }
}

ETX_GPU_CODE void handle_direct_emitter(const Scene& scene, const Triangle& tri, const Raytracing& rt, const bool mis, PTRayPayloadSoA& payload, uint32_t i) {
  const auto& intersection = payload.intersection[i];
  if (intersection.emitter_index == kInvalidIndex)
    return;

  const auto& emitter = scene.emitters[intersection.emitter_index];
  float pdf_emitter_area = 0.0f;
  float pdf_emitter_dir = 0.0f;
  float pdf_emitter_dir_out = 0.0f;
  auto e = emitter_get_radiance(emitter, payload.spect[i], intersection.tex, payload.ray[i].o, intersection.pos, pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene,
    (payload.path_length[i] == 0));

  if (pdf_emitter_dir > 0.0f) {
    auto tr = rt.trace_transmittance(payload.spect[i], scene, payload.ray[i].o, intersection.pos, payload.medium[i], payload.smp[i]);
    float pdf_emitter_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
    bool no_weight = (mis == false) || (payload.path_length[i] == 1) || (payload.mis_weight[i] == false);
    auto weight = no_weight ? 1.0f : power_heuristic(payload.sampled_bsdf_pdf[i], pdf_emitter_discrete * pdf_emitter_dir);
    payload.accumulated[i] += payload.throughput[i] * e * tr * weight;
    ETX_VALIDATE(payload.accumulated[i]);
  }
}

ETX_GPU_CODE bool handle_hit_ray(const Scene& scene, const Intersection& intersection, const PTOptions& options, const Raytracing& rt, PTRayPayload& payload) {
  const auto& tri = scene.triangles[intersection.triangle_index];
  const auto& mat = scene.materials[intersection.material_index];

  if (mat.cls == Material::Class::Boundary) {
    payload.medium = (dot(intersection.nrm, payload.ray.d) < 0.0f) ? mat.int_medium : mat.ext_medium;
    payload.ray.o = shading_pos(scene.vertices, tri, intersection.barycentric, payload.ray.d);
    return true;
  }

  handle_direct_emitter(scene, tri, intersection, rt, options.mis, payload);

  auto bsdf_sample = bsdf::sample({payload.spect, payload.medium, PathSource::Camera, intersection, intersection.w_i}, mat, scene, payload.smp);
  bool subsurface_path = mat.has_subsurface_scattering() && (bsdf_sample.properties & BSDFSample::Diffuse);

  subsurface::Gather ss_gather = {};
  bool subsurface_sampled = subsurface_path && subsurface::gather(payload.spect, scene, intersection, intersection.material_index, rt, payload.smp, ss_gather);

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // direct light sampling
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  if (options.nee && (payload.path_length + 1 <= options.max_depth)) {
    uint32_t emitter_index = sample_emitter_index(scene, payload.smp);
    SpectralResponse direct_light = {payload.spect.wavelength, 0.0f};
    if (subsurface_sampled) {
      for (uint32_t i = 0; i < ss_gather.intersection_count; ++i) {
        auto local_sample = sample_emitter(payload.spect, emitter_index, payload.smp, ss_gather.intersections[i].pos, scene);
        SpectralResponse light_value = evaluate_light(scene, ss_gather.intersections[i], rt, mat, payload.medium, payload.spect, local_sample, payload.smp, options.mis);
        direct_light += ss_gather.weights[i] * light_value;
        ETX_VALIDATE(direct_light);
      }
    } else {
      auto emitter_sample = sample_emitter(payload.spect, emitter_index, payload.smp, intersection.pos, scene);
      direct_light += evaluate_light(scene, intersection, rt, mat, payload.medium, payload.spect, emitter_sample, payload.smp, options.mis);
      ETX_VALIDATE(payload.accumulated);
    }
    payload.accumulated += payload.throughput * direct_light;
  }

  if (bsdf_sample.valid() == false) {
    return false;
  }

  if (subsurface_path && (subsurface_sampled == false)) {
    return false;
  }

  if (subsurface_sampled) {
    const auto& out_intersection = ss_gather.intersections[ss_gather.selected_intersection];
    payload.ray.d = sample_cosine_distribution(payload.smp.next_2d(), out_intersection.nrm, 1.0f);
    payload.throughput *= ss_gather.weights[ss_gather.selected_intersection] * ss_gather.selected_sample_weight;
    payload.sampled_bsdf_pdf = fabsf(dot(payload.ray.d, out_intersection.nrm)) / kPi;
    payload.mis_weight = true;
    payload.ray.o = shading_pos(scene.vertices, scene.triangles[out_intersection.triangle_index], out_intersection.barycentric, payload.ray.d);
  } else {
    payload.medium = (bsdf_sample.properties & BSDFSample::MediumChanged) ? bsdf_sample.medium_index : payload.medium;
    payload.sampled_bsdf_pdf = bsdf_sample.pdf;
    payload.mis_weight = bsdf_sample.is_delta() == false;
    payload.eta *= bsdf_sample.eta;
    payload.ray.d = bsdf_sample.w_o;
    payload.ray.o = shading_pos(scene.vertices, scene.triangles[intersection.triangle_index], intersection.barycentric, payload.ray.d);
  }

  payload.throughput *= bsdf_sample.weight;
  if (payload.throughput.is_zero())
    return false;

  payload.path_length += 1;
  ETX_CHECK_FINITE(payload.ray.d);
  return random_continue(payload.path_length, options.rr_start, payload.eta, payload.smp, payload.throughput);
}

ETX_GPU_CODE void handle_missed_ray(const Scene& scene, PTRayPayload& payload) {
  for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
    const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
    float pdf_emitter_area = 0.0f;
    float pdf_emitter_dir = 0.0f;
    float pdf_emitter_dir_out = 0.0f;
    auto e = emitter_get_radiance(emitter, payload.spect, payload.ray.d, pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene);
    ETX_VALIDATE(e);
    if ((pdf_emitter_dir > 0) && (e.is_zero() == false)) {
      float pdf_emitter_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
      auto weight = ((payload.mis_weight == false) || (payload.path_length == 1)) ? 1.0f : power_heuristic(payload.sampled_bsdf_pdf, pdf_emitter_discrete * pdf_emitter_dir);
      payload.accumulated += payload.throughput * e * weight;
      ETX_VALIDATE(payload.accumulated);
    }
  }
}

ETX_GPU_CODE bool run_path_iteration(const Scene& scene, const PTOptions& options, const Raytracing& rt, PTRayPayload& payload) {
  if (payload.path_length > options.max_depth)
    return false;

  ETX_CHECK_FINITE(payload.ray.d);

  Intersection intersection = {};
  bool found_intersection = rt.trace(scene, payload.ray, intersection, payload.smp);

  Medium::Sample medium_sample = try_sampling_medium(scene, payload, intersection.t);

  if (medium_sample.sampled_medium()) {
    handle_sampled_medium(scene, medium_sample, options.max_depth, rt, payload);
    return random_continue(payload.path_length, options.rr_start, payload.eta, payload.smp, payload.throughput);
  }

  if (found_intersection) {
    return handle_hit_ray(scene, intersection, options, rt, payload);
  }

  handle_missed_ray(scene, payload);
  return false;
}

struct ETX_ALIGNED PTGPUData {
  PTRayPayload* payloads ETX_EMPTY_INIT;
  float4* output ETX_EMPTY_INIT;
  Scene scene ETX_EMPTY_INIT;
  PTOptions options ETX_EMPTY_INIT;
};

}  // namespace etx
