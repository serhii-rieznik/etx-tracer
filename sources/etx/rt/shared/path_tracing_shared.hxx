#pragma once

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/scene.hxx>

namespace etx {

struct ETX_ALIGNED PTOptions {
  uint32_t path_per_iteration ETX_INIT_WITH(1u);
  bool nee ETX_INIT_WITH(true);
  bool mis ETX_INIT_WITH(true);
};

struct ETX_ALIGNED PTRayPayload {
  Ray ray = {};
  SpectralResponse throughput = {};
  SpectralResponse accumulated = {};
  SpectralResponse view_albedo = {};
  float3 view_normal = {};
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

namespace subsurface {

ETX_GPU_CODE bool gather_rw(SpectralQuery spect, const Scene& scene, const Intersection& in_intersection, const Raytracing& rt, Sampler& smp, Gather& result) {
  constexpr uint32_t kMaxIterations = 1024u;

  const auto& mat = scene.materials[in_intersection.material_index];
  if (mat.int_medium == kInvalidIndex)
    return false;

  const Medium& medium = scene.mediums[mat.int_medium];

  float anisotropy = medium.phase_function_g;
  SpectralResponse scattering = medium.s_scattering(spect);
  SpectralResponse absorption = medium.s_absorption(spect);

  SpectralResponse extinction = scattering + absorption;
  SpectralResponse albedo = Medium::calculate_albedo(spect, scattering, extinction);

  Ray ray = {};
  ray.d = sample_cosine_distribution(smp.next_2d(), -in_intersection.nrm, 1.0f);
  ray.min_t = kRayEpsilon;
  ray.o = shading_pos(scene.vertices, scene.triangles[in_intersection.triangle_index], in_intersection.barycentric, ray.d);
  ray.max_t = kMaxFloat;

  SpectralResponse throughput = {spect, 1.0f};
  for (uint32_t i = 0; i < kMaxIterations; ++i) {
    SpectralResponse pdf = {};
    uint32_t channel = medium::sample_spectrum_component(spect, albedo, throughput, smp, pdf);
    float scattering_distance = extinction.component(channel);

    ray.max_t = scattering_distance > 0.0f ? (-logf(1.0f - smp.next()) / scattering_distance) : kMaxFloat;
    ETX_VALIDATE(ray.max_t);

    Intersection local_i;
    bool intersection_found = rt.trace_material(scene, ray, in_intersection.material_index, local_i, smp);
    if (intersection_found) {
      ray.max_t = local_i.t;
    }

    SpectralResponse tr = exp(-ray.max_t * extinction);
    ETX_VALIDATE(tr);

    pdf *= intersection_found ? tr : tr * extinction;
    ETX_VALIDATE(pdf);

    if (pdf.is_zero())
      return false;

    SpectralResponse weight = intersection_found ? tr : tr * scattering;
    ETX_VALIDATE(weight);

    throughput *= weight / pdf.sum();
    ETX_VALIDATE(throughput);

    if (throughput.maximum() <= kEpsilon)
      return false;

    if (intersection_found) {
      bool w_i_in = dot(local_i.w_i, local_i.nrm) > 0.0f;

      result.intersections[0] = local_i;
      result.intersections[0].w_i *= w_i_in ? -1.0f : +1.0f;
      result.weights[0] = throughput * apply_image(spect, mat.transmittance, local_i.tex, scene, rgb::SpectrumClass::Reflection, nullptr);
      result.intersection_count = 1u;
      result.selected_intersection = 0;
      result.selected_sample_weight = 1.0f;
      result.total_weight = 1.0f;
      return true;
    }

    ray.o = ray.o + ray.d * ray.max_t;
    ray.d = medium::sample_phase_function(ray.d, anisotropy, smp);
  }
  return false;
}

ETX_GPU_CODE bool gather_cb(SpectralQuery spect, const Scene& scene, const Intersection& in_intersection, const Raytracing& rt, Sampler& smp, Gather& result) {
  const auto& mtl = scene.materials[in_intersection.material_index].subsurface;

  Sample ss_samples[kIntersectionDirections] = {
    sample(spect, in_intersection, mtl, 0u, smp),
    sample(spect, in_intersection, mtl, 1u, smp),
    sample(spect, in_intersection, mtl, 2u, smp),
  };

  IntersectionBase intersections[kTotalIntersections] = {};
  ContinousTraceOptions ct = {intersections, kIntersectionsPerDirection, in_intersection.material_index};
  uint32_t intersections_0 = rt.continuous_trace(scene, ss_samples[0].ray, ct, smp);
  ct.intersection_buffer += intersections_0;
  uint32_t intersections_1 = rt.continuous_trace(scene, ss_samples[1].ray, ct, smp);
  ct.intersection_buffer += intersections_1;
  uint32_t intersections_2 = rt.continuous_trace(scene, ss_samples[2].ray, ct, smp);

  uint32_t intersection_count = intersections_0 + intersections_1 + intersections_2;
  ETX_CRITICAL(intersection_count <= kTotalIntersections);
  if (intersection_count == 0) {
    return false;
  }

  result = {};
  for (uint32_t i = 0; i < intersection_count; ++i) {
    const Sample& ss_sample = (i < intersections_0) ? ss_samples[0] : (i < intersections_0 + intersections_1 ? ss_samples[1] : ss_samples[2]);

    auto out_intersection = make_intersection(scene, ss_sample.ray.d, intersections[i]);

    float gw = geometric_weigth(out_intersection.nrm, ss_sample);
    float pdf = evaluate(spect, mtl, ss_sample.sampled_radius).average();
    ETX_VALIDATE(pdf);
    if (pdf <= 0.0f)
      continue;

    auto eval = evaluate(spect, mtl, length(out_intersection.pos - in_intersection.pos));
    ETX_VALIDATE(eval);

    auto weight = eval / pdf * gw;
    ETX_VALIDATE(weight);

    if (weight.is_zero())
      continue;

    result.total_weight += weight.average();
    result.intersections[result.intersection_count] = out_intersection;
    result.weights[result.intersection_count] = weight;
    result.intersection_count += 1u;
  }

  if (result.total_weight > 0.0f) {
    float rnd = smp.next() * result.total_weight;
    float partial_sum = 0.0f;
    float sample_weight = 0.0f;
    for (uint32_t i = 0; i < result.intersection_count; ++i) {
      sample_weight = result.weights[i].average();
      float next_sum = partial_sum + sample_weight;
      if (rnd < next_sum) {
        result.selected_intersection = i;
        result.selected_sample_weight = result.total_weight / sample_weight;
        break;
      }
      partial_sum = next_sum;
    }
    ETX_ASSERT(result.selected_intersection != kInvalidIndex);
  }

  return result.intersection_count > 0;
}

template <class RT>
ETX_GPU_CODE bool gather(SpectralQuery spect, const Scene& scene, const Intersection& in_intersection, const RT& rt, Sampler& smp, Gather& result) {
  const auto& mtl = scene.materials[in_intersection.material_index].subsurface;
  ETX_FUNCTION_SCOPE();

  switch (mtl.cls) {
    case SubsurfaceMaterial::Class::ChristensenBurley:
      return gather_cb(spect, scene, in_intersection, rt, smp, result);
    default:
      return gather_rw(spect, scene, in_intersection, rt, smp, result);
  }
}

}  // namespace subsurface

ETX_GPU_CODE PTRayPayload make_ray_payload(const Scene& scene, uint2 px, uint2 dim, uint32_t iteration, bool spectral) {
  ETX_FUNCTION_SCOPE();

  PTRayPayload payload = {};
  payload.index = px.x + px.y * dim.x;
  payload.iteration = iteration;
  payload.smp.init(payload.index, payload.iteration);
  payload.spect = spectral ? SpectralQuery::spectral_sample(payload.smp.next()) : SpectralQuery::sample();
  payload.uv = get_jittered_uv(payload.smp, px, dim);
  payload.ray = generate_ray(payload.smp, scene, payload.uv);
  payload.throughput = {payload.spect, 1.0f};
  payload.accumulated = {payload.spect, 0.0f};
  payload.medium = scene.camera_medium_index;
  payload.path_length = 1;
  payload.eta = 1.0f;
  payload.sampled_bsdf_pdf = 0.0f;
  payload.mis_weight = true;
  return payload;
}

ETX_GPU_CODE Medium::Sample try_sampling_medium(const Scene& scene, PTRayPayload& payload, float max_t) {
  ETX_FUNCTION_SCOPE();

  if (payload.medium == kInvalidIndex) {
    return {};
  }

  auto medium_sample = scene.mediums[payload.medium].sample(payload.spect, payload.throughput, payload.smp, payload.ray.o, payload.ray.d, max_t);
  payload.throughput *= medium_sample.weight;
  ETX_VALIDATE(payload.throughput);
  return medium_sample;
}

ETX_GPU_CODE void handle_sampled_medium(const Scene& scene, const Medium::Sample& medium_sample, const Raytracing& rt, PTRayPayload& payload) {
  ETX_FUNCTION_SCOPE();

  const auto& medium = scene.mediums[payload.medium];
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   * direct light sampling from medium
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  if (payload.path_length + 1 <= rt.scene().max_path_length) {
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

  float3 w_o = medium.sample_phase_function(payload.spect, payload.smp, payload.ray.d);
  payload.sampled_bsdf_pdf = medium.phase_function(payload.spect, medium_sample.pos, payload.ray.d, w_o);
  payload.mis_weight = true;
  payload.ray.o = medium_sample.pos;
  payload.ray.d = w_o;
  payload.path_length += 1;
  ETX_CHECK_FINITE(payload.ray.d);
}

ETX_GPU_CODE SpectralResponse evaluate_light(const Scene& scene, const Intersection& intersection, const Raytracing& rt, const Material& mat, const uint32_t medium,
  const SpectralQuery spect, const EmitterSample& emitter_sample, Sampler& smp, bool mis) {
  ETX_FUNCTION_SCOPE();

  if (emitter_sample.pdf_dir == 0.0f) {
    return {spect, 0.0f};
  }

  BSDFEval bsdf_eval = bsdf::evaluate({spect, medium, PathSource::Camera, intersection, intersection.w_i}, emitter_sample.direction, mat, scene, smp);
  if (bsdf_eval.valid() == false) {
    return {spect, 0.0f};
  }

  ETX_VALIDATE(bsdf_eval.bsdf);

  const auto& tri = scene.triangles[intersection.triangle_index];
  auto pos = shading_pos(scene.vertices, tri, intersection.barycentric, emitter_sample.direction);
  auto tr = rt.trace_transmittance(spect, scene, pos, emitter_sample.origin, medium, smp);
  ETX_VALIDATE(tr);

  bool no_weight = (mis == false) || emitter_sample.is_delta;
  auto weight = no_weight ? 1.0f : power_heuristic(emitter_sample.pdf_dir * emitter_sample.pdf_sample, bsdf_eval.pdf);
  ETX_VALIDATE(weight);

  float wscale = weight / (emitter_sample.pdf_dir * emitter_sample.pdf_sample);
  ETX_VALIDATE(wscale);
  return bsdf_eval.bsdf * emitter_sample.value * tr * wscale;
}

ETX_GPU_CODE void handle_direct_emitter(const Scene& scene, const Triangle& tri, const Intersection& intersection, const Raytracing& rt, const bool mis, PTRayPayload& payload) {
  ETX_FUNCTION_SCOPE();

  if (intersection.emitter_index == kInvalidIndex)
    return;

  const auto& emitter = scene.emitters[intersection.emitter_index];
  float pdf_emitter_area = 0.0f;
  float pdf_emitter_dir = 0.0f;
  float pdf_emitter_dir_out = 0.0f;

  EmitterRadianceQuery q = {
    .source_position = payload.ray.o,
    .target_position = intersection.pos,
    .uv = intersection.tex,
    .directly_visible = payload.path_length == 1,
  };

  auto e = emitter_get_radiance(emitter, payload.spect, q, pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene);

  if (pdf_emitter_dir > 0.0f) {
    auto tr = rt.trace_transmittance(payload.spect, scene, payload.ray.o, intersection.pos, payload.medium, payload.smp);
    float pdf_emitter_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
    bool no_weight = (mis == false) || q.directly_visible || (payload.mis_weight == false);
    auto weight = no_weight ? 1.0f : power_heuristic(payload.sampled_bsdf_pdf, pdf_emitter_discrete * pdf_emitter_dir);
    payload.accumulated += payload.throughput * e * tr * weight;
    ETX_VALIDATE(payload.accumulated);
  }
}

ETX_GPU_CODE bool handle_hit_ray(const Scene& scene, const Intersection& intersection, const PTOptions& options, const Raytracing& rt, PTRayPayload& payload) {
  ETX_FUNCTION_SCOPE();

  const auto& tri = scene.triangles[intersection.triangle_index];
  const auto& mat = scene.materials[intersection.material_index];

  if (mat.cls == Material::Class::Boundary) {
    payload.medium = (dot(intersection.nrm, payload.ray.d) < 0.0f) ? mat.int_medium : mat.ext_medium;
    payload.ray.o = shading_pos(scene.vertices, tri, intersection.barycentric, payload.ray.d);
    return true;
  }

  handle_direct_emitter(scene, tri, intersection, rt, options.mis, payload);

  BSDFData bsdf_data = {payload.spect, payload.medium, PathSource::Camera, intersection, intersection.w_i};

  if (payload.path_length == 1) {
    payload.view_normal = intersection.nrm;
    payload.view_albedo = bsdf::albedo(bsdf_data, mat, scene, payload.smp);
  }

  auto bsdf_sample = bsdf::sample(bsdf_data, mat, scene, payload.smp);
  bool subsurface_path = (bsdf_sample.properties & BSDFSample::Diffuse) && (mat.subsurface.cls != SubsurfaceMaterial::Class::Disabled);

  // uint8_t ss_gather_data[sizeof(subsurface::Gather)];
  subsurface::Gather ss_gather;
  bool subsurface_sampled = subsurface_path && subsurface::gather(payload.spect, scene, intersection, rt, payload.smp, ss_gather);

  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // direct light sampling
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  if (options.nee && (payload.path_length + 1 <= rt.scene().max_path_length)) {
    uint32_t emitter_index = sample_emitter_index(scene, payload.smp);
    SpectralResponse direct_light = {payload.spect, 0.0f};
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
      ETX_VALIDATE(direct_light);
    }
    payload.accumulated += payload.throughput * direct_light;
    ETX_VALIDATE(payload.accumulated);
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
  return random_continue(payload.path_length, scene.random_path_termination, payload.eta, payload.smp, payload.throughput);
}

ETX_GPU_CODE void handle_missed_ray(const Scene& scene, PTRayPayload& payload) {
  ETX_FUNCTION_SCOPE();

  for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
    const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
    float pdf_emitter_area = 0.0f;
    float pdf_emitter_dir = 0.0f;
    float pdf_emitter_dir_out = 0.0f;
    EmitterRadianceQuery q = {
      .direction = payload.ray.d,
      .directly_visible = payload.path_length == 1,
    };
    auto e = emitter_get_radiance(emitter, payload.spect, q, pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene);
    ETX_VALIDATE(e);
    if ((pdf_emitter_dir > 0) && (e.is_zero() == false)) {
      float pdf_emitter_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
      auto weight = ((payload.mis_weight == false) || q.directly_visible) ? 1.0f : power_heuristic(payload.sampled_bsdf_pdf, pdf_emitter_discrete * pdf_emitter_dir);
      payload.accumulated += payload.throughput * e * weight;
      ETX_VALIDATE(payload.accumulated);
    }
  }
}

ETX_GPU_CODE bool run_path_iteration(const Scene& scene, const PTOptions& options, const Raytracing& rt, PTRayPayload& payload) {
  if (payload.path_length > rt.scene().max_path_length)
    return false;

  ETX_FUNCTION_SCOPE();
  ETX_CHECK_FINITE(payload.ray.d);

  Intersection intersection;
  bool found_intersection = rt.trace(scene, payload.ray, intersection, payload.smp);

  Medium::Sample medium_sample = try_sampling_medium(scene, payload, intersection.t);

  if (medium_sample.sampled_medium()) {
    handle_sampled_medium(scene, medium_sample, rt, payload);
    return random_continue(payload.path_length, scene.random_path_termination, payload.eta, payload.smp, payload.throughput);
  }

  if (found_intersection) {
    return handle_hit_ray(scene, intersection, options, rt, payload);
  }

  handle_missed_ray(scene, payload);
  return false;
}

}  // namespace etx
