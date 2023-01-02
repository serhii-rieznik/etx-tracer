#pragma once

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/scene.hxx>

namespace etx {

struct ETX_ALIGNED PTOptions {
  uint32_t iterations ETX_INIT_WITH(65536u);
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
  bool sampled_delta_bsdf = false;
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
  payload.sampled_delta_bsdf = false;
  return payload;
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

ETX_GPU_CODE void handle_sampled_medium(const Scene& scene, const Medium::Sample& medium_sample, const PTOptions& options, const Raytracing& rt, PTRayPayload& payload) {
  const auto& medium = scene.mediums[payload.medium];
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   * direct light sampling from medium
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  if (payload.path_length + 1 <= options.max_depth) {
    auto emitter_sample = sample_emitter(payload.spect, payload.smp, medium_sample.pos, scene);
    if (emitter_sample.pdf_dir > 0) {
      auto tr = transmittance(payload.spect, payload.smp, medium_sample.pos, emitter_sample.origin, payload.medium, scene, rt);
      float phase_function = medium.phase_function(payload.spect, medium_sample.pos, payload.ray.d, emitter_sample.direction);
      auto weight = emitter_sample.is_delta ? 1.0f : power_heuristic(emitter_sample.pdf_dir * emitter_sample.pdf_sample, phase_function);
      payload.accumulated += payload.throughput * emitter_sample.value * tr * (phase_function * weight / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));
      ETX_VALIDATE(payload.accumulated);
    }
  }

  float3 w_o = medium.sample_phase_function(payload.spect, payload.smp, medium_sample.pos, payload.ray.d);
  payload.sampled_bsdf_pdf = medium.phase_function(payload.spect, medium_sample.pos, payload.ray.d, w_o);
  payload.sampled_delta_bsdf = false;
  payload.ray.o = medium_sample.pos;
  payload.ray.d = w_o;
  payload.path_length += 1;
  ETX_CHECK_FINITE(payload.ray.d);
}

ETX_GPU_CODE SpectralResponse sample_light(const Scene& scene, const Intersection& intersection, const Raytracing& rt, const uint32_t medium, const SpectralQuery spect,
  Sampler& smp, bool mis) {
  auto emitter_sample = sample_emitter(spect, smp, intersection.pos, scene);
  if (emitter_sample.pdf_dir <= 0) {
    return {spect.wavelength, 0.0f};
  }

  const auto& tri = scene.triangles[intersection.triangle_index];
  const auto& mat = scene.materials[tri.material_index];

  BSDFEval bsdf_eval = bsdf::evaluate({spect, medium, PathSource::Camera, intersection, intersection.w_i, emitter_sample.direction}, mat, scene, smp);
  if (bsdf_eval.valid() == false) {
    return {spect.wavelength, 0.0f};
  }

  ETX_VALIDATE(bsdf_eval.bsdf);

  auto pos = shading_pos(scene.vertices, tri, intersection.barycentric, emitter_sample.direction);
  auto tr = transmittance(spect, smp, pos, emitter_sample.origin, medium, scene, rt);
  ETX_VALIDATE(tr);

  bool no_weight = (mis == false) || emitter_sample.is_delta;
  auto weight = no_weight ? 1.0f : power_heuristic(emitter_sample.pdf_dir * emitter_sample.pdf_sample, bsdf_eval.pdf);
  ETX_VALIDATE(weight);

  ETX_VALIDATE(emitter_sample.value);
  return bsdf_eval.bsdf * emitter_sample.value * tr * (weight / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));
}

ETX_GPU_CODE bool update_payload_with_bsdf_sample(const Scene& scene, const Intersection& intersection, PTRayPayload& payload, const BSDFSample& bsdf_sample) {
  if (bsdf_sample.valid() == false) {
    return false;
  }

  ETX_VALIDATE(payload.throughput);
  payload.throughput *= bsdf_sample.weight;
  ETX_VALIDATE(payload.throughput);
  if (payload.throughput.is_zero())
    return false;

  payload.medium = (bsdf_sample.properties & BSDFSample::MediumChanged) ? bsdf_sample.medium_index : payload.medium;
  payload.sampled_bsdf_pdf = bsdf_sample.pdf;
  payload.sampled_delta_bsdf = bsdf_sample.is_delta();
  payload.eta *= bsdf_sample.eta;
  payload.ray.d = bsdf_sample.w_o;
  payload.ray.o = shading_pos(scene.vertices, scene.triangles[intersection.triangle_index], intersection.barycentric, bsdf_sample.w_o);

  return true;
}

ETX_GPU_CODE bool handle_subsurface(const Scene& scene, const Intersection& in_intersection, const PTOptions& options, const Raytracing& rt, const uint32_t material_index,
  PTRayPayload& payload) {
  const auto& mtl = scene.materials[material_index];
  auto ss_sample = subsurface::sample_spatial(in_intersection, mtl, scene, payload.smp);
  auto ss_ray = subsurface::make_ray(ss_sample, in_intersection.pos);

  IntersectionBase intersections[subsurface::kMaxIntersections] = {};
  uint32_t intersection_count = rt.continuous_trace(scene, ss_ray, {intersections, subsurface::kMaxIntersections, material_index}, payload.smp);
  if (intersection_count == 0) {
    return false;
  }

  uint32_t selected_intersection = uint32_t(payload.smp.next() * float(intersection_count));
  auto out_intersection = rt.make_intersection(scene, ss_ray.d, intersections[selected_intersection]);
  const auto& out_tri = scene.triangles[out_intersection.triangle_index];

  float actual_distance = length(out_intersection.pos - in_intersection.pos);

  auto eval = subsurface::eval_s_r(payload.spect, mtl.subsurface.scattering, actual_distance);
  ETX_VALIDATE(eval);

  auto pdf = subsurface::pdf_s_p(in_intersection, ss_sample, out_intersection, mtl.subsurface.scattering) / float(intersection_count);
  ETX_VALIDATE(pdf);

  payload.throughput *= eval / pdf;
  ETX_VALIDATE(payload.throughput);

  auto light_value = sample_light(scene, out_intersection, rt, payload.medium, payload.spect, payload.smp, options.mis);
  payload.accumulated += payload.throughput * light_value;
  ETX_VALIDATE(payload.accumulated);

  auto bsdf_sample = bsdf::sample({payload.spect, payload.medium, PathSource::Camera, out_intersection, out_intersection.w_i, {}}, mtl, scene, payload.smp);
  if (bsdf_sample.valid() == false) {
    return false;
  }

  return update_payload_with_bsdf_sample(scene, out_intersection, payload, bsdf_sample);
}

ETX_GPU_CODE bool handle_hit_ray(const Scene& scene, const Intersection& intersection, const PTOptions& options, const Raytracing& rt, PTRayPayload& payload) {
  const auto& tri = scene.triangles[intersection.triangle_index];
  const auto& mat = scene.materials[tri.material_index];

  if (mat.cls == Material::Class::Boundary) {
    payload.medium = (dot(intersection.nrm, payload.ray.d) < 0.0f) ? mat.int_medium : mat.ext_medium;
    payload.ray.o = shading_pos(scene.vertices, tri, intersection.barycentric, payload.ray.d);
    return true;
  }

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // direct light sampling
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  if (options.nee && (payload.path_length + 1 <= options.max_depth)) {
    payload.accumulated += payload.throughput * sample_light(scene, intersection, rt, payload.medium, payload.spect, payload.smp, options.mis);
    ETX_VALIDATE(payload.accumulated);
  }
  //*
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // directly visible emitters
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  if (tri.emitter_index != kInvalidIndex) {
    const auto& emitter = scene.emitters[tri.emitter_index];
    float pdf_emitter_area = 0.0f;
    float pdf_emitter_dir = 0.0f;
    float pdf_emitter_dir_out = 0.0f;
    auto e = emitter_get_radiance(emitter, payload.spect, intersection.tex, payload.ray.o, intersection.pos, pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene,
      (payload.path_length == 0));
    if (pdf_emitter_dir > 0.0f) {
      auto tr = transmittance(payload.spect, payload.smp, payload.ray.o, intersection.pos, payload.medium, scene, rt);
      float pdf_emitter_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
      bool no_weight = (options.mis == false) || (payload.path_length == 1) || payload.sampled_delta_bsdf;
      auto weight = no_weight ? 1.0f : power_heuristic(payload.sampled_bsdf_pdf, pdf_emitter_discrete * pdf_emitter_dir);
      payload.accumulated += payload.throughput * e * tr * weight;
      ETX_VALIDATE(payload.accumulated);
    }
  }

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // bsdf sampling
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  if (mat.subsurface.scattering.is_zero()) {
    auto bsdf_sample = bsdf::sample({payload.spect, payload.medium, PathSource::Camera, intersection, intersection.w_i, {}}, mat, scene, payload.smp);
    if (update_payload_with_bsdf_sample(scene, intersection, payload, bsdf_sample) == false) {
      return false;
    }
  } else {
    if (handle_subsurface(scene, intersection, options, rt, tri.material_index, payload) == false) {
      return false;
    }
  }

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
      auto weight = ((payload.path_length == 1) || payload.sampled_delta_bsdf) ? 1.0f : power_heuristic(payload.sampled_bsdf_pdf, pdf_emitter_discrete * pdf_emitter_dir);
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

  Medium::Sample medium_sample = try_sampling_medium(scene, payload, found_intersection ? intersection.t : kMaxFloat);

  if (medium_sample.sampled_medium()) {
    handle_sampled_medium(scene, medium_sample, options, rt, payload);
    return true;
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
