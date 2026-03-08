#pragma once
#include <etx/render/interop/projection.hxx>
#include <etx/render/access/emitter_access_cpu.hxx>

namespace etx {

ETX_SHARED_INLINE bool scene_has_environment_emitter_state(const Scene& scene) {
  return (scene.emitter_instances.count > 0u) && (scene.environment_emitters.count > 0u);
}

ETX_SHARED_INLINE bool try_load_emitter_scene_state_shared(const Scene& scene, uint32_t& emitter_instance_count, uint32_t& emitter_profile_count) {
  EmitterAccessCPUContext context = make_emitter_access_cpu_context(scene);
  return emitter_access_try_load_scene_state(context, emitter_instance_count, emitter_profile_count);
}

ETX_SHARED_INLINE bool try_load_emitter_instance_count_shared(const Scene& scene, uint32_t& emitter_instance_count) {
  uint32_t emitter_profile_count = 0u;
  return try_load_emitter_scene_state_shared(scene, emitter_instance_count, emitter_profile_count);
}

ETX_SHARED_INLINE uint32_t environment_emitter_shared_count(const Scene& scene) {
  if (scene_has_environment_emitter_state(scene) == false) {
    return 0u;
  }

  return min(static_cast<uint32_t>(scene.environment_emitters.count), uint32_t(SceneLimits::MaxEnvironmentEmitters));
}

ETX_SHARED_INLINE bool environment_emitter_shared_try_load_index(const Scene& scene, uint32_t local_index, uint32_t& emitter_index) {
  emitter_index = kInvalidIndex;
  uint32_t emitter_count = environment_emitter_shared_count(scene);
  if (local_index >= emitter_count) {
    return false;
  }

  emitter_index = scene.environment_emitters.emitters[local_index];
  if (emitter_index >= scene.emitter_instances.count) {
    emitter_index = kInvalidIndex;
    return false;
  }

  return true;
}

ETX_SHARED_INLINE bool try_select_environment_emitter_random(const Scene& scene, Sampler& smp, uint32_t& emitter_index, uint32_t& emitter_count) {
  emitter_index = kInvalidIndex;
  emitter_count = environment_emitter_shared_count(scene);
  if (emitter_count == 0u) {
    return false;
  }

  uint32_t selected = uint32_t(smp.next() * float(emitter_count));
  if (selected >= emitter_count) {
    selected = emitter_count - 1u;
  }

  return environment_emitter_shared_try_load_index(scene, selected, emitter_index);
}

ETX_SHARED_INLINE float emitter_pdf_area_local(const Emitter& em, const Scene& scene) {
  ETX_ASSERT(em.is_local());
  return 1.0f / em.triangle_area;
}

ETX_SHARED_INLINE uint32_t emitter_external_medium_index(const Scene& scene, const Emitter& em_inst) {
  EmitterAccessCPUContext access_context = make_emitter_access_cpu_context(scene);
  return emitter_access_external_medium_index(access_context, em_inst);
}

ETX_SHARED_INLINE SpectralResponse emitter_evaluate_out_local(const Emitter& em_inst, const SpectralQuery spect, const float2& uv, const float3& emitter_normal, const float3& direction,
  float& pdf_area, float& pdf_dir, float& pdf_dir_out, const Scene& scene) {
  EmitterAccessCPUContext access_context = make_emitter_access_cpu_context(scene);
  EmitterAccess emission_access = {};
  if (emitter_access_try_load_local_from_instance(access_context, em_inst, emission_access) == false) {
    return {spect, 0.0f};
  }

  ETX_ASSERT(em_inst.is_local());

  // Get collimation from material
  float collimation = 0.0f;
  if (em_inst.triangle_index != kInvalidIndex) {
    const auto& tri = scene.triangles[em_inst.triangle_index];
    const auto& material = scene.materials[tri.material_index];
    collimation = material.emission_collimation;
  }

  float cos_t = max(0.0f, dot(emitter_normal, direction));
  pdf_dir = powf(cos_t, collimation_to_exponent(collimation)) * kInvPi;

  if (pdf_dir <= 0.0f) {
    return {spect, 0.0f};
  }

  pdf_area = emitter_pdf_area_local(em_inst, scene);
  ETX_ASSERT(pdf_area > 0.0f);

  pdf_dir_out = pdf_dir * pdf_area;
  ETX_ASSERT(pdf_dir_out > 0.0f);

  return emitter_access_evaluate_spectral_source(access_context, emission_access.emission_spectrum_index, emission_access.emission_image_index, uv, spect);
}

ETX_SHARED_INLINE SpectralResponse emitter_get_radiance(const Emitter& em_inst, const SpectralQuery spect, const EmitterRadianceQuery& query, float& pdf_area, float& pdf_dir,
  float& pdf_dir_out, const Scene& scene) {
  EmitterAccessCPUContext access_context = make_emitter_access_cpu_context(scene);
  EmitterAccess emission_access = {};
  pdf_dir = 0.0f;
  pdf_area = 0.0f;
  pdf_dir_out = 0.0f;
  if (emitter_access_try_load_from_instance(access_context, em_inst, emission_access) == false) {
    return {spect, 0.0f};
  }

  switch (emission_access.emitter_class) {
    case EmitterClass::Directional: {
      bool accepts_direction = emitter_access_accepts_direction(query.direction, emission_access.emitter_direction, emission_access.emitter_angular_size_cosine);
      if ((query.directly_visible == false) || (emission_access.emitter_angular_size_cosine >= 1.0f) || (accepts_direction == false)) {
        return {spect, 0.0f};
      }

      pdf_dir = 1.0f;
      pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      pdf_dir_out = pdf_dir * pdf_area;
      float equivalent_disk_size = 0.0f;
      if (emission_access.emitter_angular_size_cosine > kEpsilon) {
        float sin_half_angle =
          sqrt(max(0.0f, 1.0f - (emission_access.emitter_angular_size_cosine * emission_access.emitter_angular_size_cosine)));
        equivalent_disk_size = 2.0f * (sin_half_angle / emission_access.emitter_angular_size_cosine);
      }
      float2 uv = disk_uv(emission_access.emitter_direction, query.direction, equivalent_disk_size, emission_access.emitter_angular_size_cosine);
      SpectralResponse directional_spectrum = emitter_access_load_spectrum_spectral(access_context, emission_access.emission_spectrum_index, spect);
      SpectralResponse direct_scale = 1.0f / (directional_spectrum * kDoublePi * (1.0f - emission_access.emitter_angular_size_cosine));
      return emitter_access_evaluate_spectral_source(access_context, emission_access.emission_spectrum_index, emission_access.emission_image_index, uv, spect) * direct_scale;
    }

    case EmitterClass::Environment: {
      bool is_atmosphere = emitter_access_is_atmosphere(emission_access.emitter_profile_meta);
      uint32_t projection_mode = projection_environment_mode(is_atmosphere);
      float2 uv = emitter_access_environment_uv(access_context, emission_access, query.direction);

      auto image_pdf = 0.0f;
      SpectralImage emission_image = {};
      emission_image.spectrum_index = emission_access.emission_spectrum_index;
      emission_image.image_index = emission_access.emission_image_index;
      auto eval = apply_image(spect, emission_image, uv, scene, &image_pdf);
      pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      ETX_VALIDATE(pdf_area);
      pdf_dir = projection_environment_image_pdf_to_solid_angle(image_pdf, uv, projection_mode);
      ETX_VALIDATE(pdf_dir);
      pdf_dir_out = pdf_area * pdf_dir;
      ETX_VALIDATE(pdf_dir_out);
      return eval;
    }

    case EmitterClass::Area: {
      const auto& tri = scene.triangles[em_inst.triangle_index];
      const Material& material = scene.materials[tri.material_index];

      if (dot(tri.geo_n, query.target_position - query.source_position) >= 0.0f) {
        return {spect, 0.0f};
      }
      pdf_area = emitter_pdf_area_local(em_inst, scene);

      float3 dp = query.source_position - query.target_position;
      float distance_squared = dot(dp, dp);
      if (distance_squared > 0.0f) {
        float cos_t = fabsf(dot(dp, tri.geo_n)) / sqrtf(distance_squared);
        float exponent = collimation_to_exponent(material.emission_collimation);
        float cos_tx = query.directly_visible ? cos_t : powf(cos_t, exponent);
        if (cos_tx > kEpsilon) {
          pdf_dir = pdf_area * distance_squared / cos_tx;
          pdf_dir_out = pdf_area * cos_tx * kInvPi;
        }
      }

      return emitter_access_evaluate_spectral_source(access_context, emission_access.emission_spectrum_index, emission_access.emission_image_index, query.uv, spect);
    }

    default: {
      ETX_FAIL("Unknown emitter class");
      return {spect, 0.0f};
    }
  }
}

ETX_SHARED_INLINE SpectralResponse emitter_evaluate_out_dist(const Emitter& em_inst, const SpectralQuery spect, const float3& in_direction, float& pdf_area, float& pdf_dir,
  const Scene& scene) {
  ETX_ASSERT(em_inst.is_distant());

  EmitterAccessCPUContext access_context = make_emitter_access_cpu_context(scene);
  EmitterAccess emission_access = {};
  pdf_area = 0.0f;
  pdf_dir = 0.0f;
  if (emitter_access_try_load_distant_from_instance(access_context, em_inst, in_direction, emission_access) == false) {
    return {spect, 0.0f};
  }
  pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);

  switch (emission_access.emitter_class) {
    case EmitterClass::Directional: {
      pdf_dir = 1.0f;
      float2 uv = emitter_access_environment_uv(access_context, emission_access, in_direction);
      return emitter_access_evaluate_spectral_source(access_context, emission_access.emission_spectrum_index, emission_access.emission_image_index, uv, spect);
    }

    case EmitterClass::Environment: {
      bool is_atmosphere = emitter_access_is_atmosphere(emission_access.emitter_profile_meta);
      uint32_t projection_mode = projection_environment_mode(is_atmosphere);
      float2 uv = emitter_access_environment_uv(access_context, emission_access, in_direction);

      auto image_pdf = 0.0f;
      SpectralImage emission_image = {};
      emission_image.spectrum_index = emission_access.emission_spectrum_index;
      emission_image.image_index = emission_access.emission_image_index;
      auto eval = apply_image(spect, emission_image, uv, scene, &image_pdf);
      pdf_dir = projection_environment_image_pdf_to_solid_angle(image_pdf, uv, projection_mode);
      ETX_VALIDATE(pdf_dir);
      return eval;
    }

    default:
      ETX_FAIL("Unknown emitter class");
      return {spect, 0.0f};
  }
}

ETX_SHARED_INLINE EmitterSample emitter_sample_in(const Emitter& em_inst, const SpectralQuery spect, const float3& from_point, const Scene& scene, const float2& smp) {
  EmitterSample result = {};
  EmitterAccessCPUContext access_context = make_emitter_access_cpu_context(scene);
  EmitterAccess emission_access = {};
  if (emitter_access_try_load_from_instance(access_context, em_inst, emission_access) == false) {
    result.medium_index = emitter_external_medium_index(scene, em_inst);
    return result;
  }

  switch (emission_access.emitter_class) {
    case EmitterClass::Area: {
      const auto& tri = scene.triangles[em_inst.triangle_index];
      result.barycentric = random_barycentric(smp);
      result.origin = lerp_pos(scene, tri, result.barycentric);
      result.normal = lerp_normal(scene, tri, result.barycentric);
      result.direction = normalize(result.origin - from_point);

      EmitterRadianceQuery q = {
        .source_position = from_point,
        .target_position = result.origin,
        .uv = lerp_uv(scene, tri, result.barycentric),
      };

      result.value = emitter_get_radiance(em_inst, spect, q, result.pdf_area, result.pdf_dir, result.pdf_dir_out, scene);
      break;
    }

    case EmitterClass::Directional: {
      float equivalent_disk_size = 0.0f;
      if (emission_access.emitter_angular_size_cosine > kEpsilon) {
        float sin_half_angle =
          sqrt(max(0.0f, 1.0f - (emission_access.emitter_angular_size_cosine * emission_access.emitter_angular_size_cosine)));
        equivalent_disk_size = 2.0f * (sin_half_angle / emission_access.emitter_angular_size_cosine);
      }

      float2 disk_sample = {};
      if (equivalent_disk_size > 0.0f) {
        auto basis = orthonormal_basis(emission_access.emitter_direction);
        disk_sample = sample_disk(smp);
        result.direction = normalize(
          emission_access.emitter_direction + basis.u * disk_sample.x * (0.5f * equivalent_disk_size) + basis.v * disk_sample.y * (0.5f * equivalent_disk_size));
      } else {
        result.direction = emission_access.emitter_direction;
      }
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir = 1.0f;
      result.pdf_dir_out = result.pdf_dir * result.pdf_area;
      result.origin = from_point + result.direction * distance_to_sphere(from_point, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.normal = emission_access.emitter_direction * (-1.0f);
      result.value =
        emitter_access_evaluate_spectral_source(access_context, emission_access.emission_spectrum_index, emission_access.emission_image_index, disk_sample * 0.5f + 0.5f, spect);
      break;
    }

    case EmitterClass::Environment: {
      if ((emission_access.emission_image_index == kInvalidIndex) || (emission_access.emission_image_index >= scene.images.count)) {
        break;
      }

      const auto& img = scene.images[emission_access.emission_image_index];
      bool is_atmosphere = emitter_access_is_atmosphere(emission_access.emitter_profile_meta);
      uint32_t projection_mode = projection_environment_mode(is_atmosphere);
      float pdf_image = 0.0f;
      uint2 image_location = {};
      float4 image_value = {};
      float2 uv = img.sample(smp, pdf_image, image_location, image_value);

      result.image_uv = uv;
      result.direction = uv_to_direction(result.image_uv, img.offset, img.scale.x, projection_mode);
      result.normal = -result.direction;
      result.origin = from_point + result.direction * distance_to_sphere(from_point, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.pdf_dir = projection_environment_image_pdf_to_solid_angle(pdf_image, uv, projection_mode);
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir_out = result.pdf_area * result.pdf_dir;
      result.value = apply_rgb(spect, emitter_access_load_spectrum_spectral(access_context, emission_access.emission_spectrum_index, spect), image_value, scene);
      break;
    }

    default: {
      ETX_FAIL("Unknown emitter class");
    }
  }

  result.medium_index = emitter_external_medium_index(scene, em_inst);
  return result;
}

ETX_SHARED_INLINE float emitter_discrete_pdf(const Emitter& emitter, const Distribution& dist) {
  if (dist.total_weight == 0.0f) {
    return 0.0f;
  }
  return (emitter.spectrum_weight * emitter.additional_weight) / dist.total_weight;
}

ETX_SHARED_INLINE float emitter_sample_pdf(const Emitter& em_inst, ETX_IN(float3, in_direction), const Scene& scene) {
  EmitterAccessCPUContext access_context = make_emitter_access_cpu_context(scene);
  EmitterAccess emission_access = {};
  if (emitter_access_try_load_from_instance(access_context, em_inst, emission_access) == false) {
    return 0.0f;
  }

  float pdf_discrete = emitter_discrete_pdf(em_inst, scene.emitters_distribution);

  switch (emission_access.emitter_class) {
    case EmitterClass::Area: {
      return pdf_discrete * emitter_pdf_area_local(em_inst, scene);
    }

    case EmitterClass::Directional: {
      bool accepts_direction = emitter_access_accepts_direction(in_direction, emission_access.emitter_direction, emission_access.emitter_angular_size_cosine);
      return accepts_direction ? pdf_discrete : 0.0f;
    }

    case EmitterClass::Environment: {
      bool is_atmosphere = emitter_access_is_atmosphere(emission_access.emitter_profile_meta);
      uint32_t projection_mode = projection_environment_mode(is_atmosphere);
      float2 uv = emitter_access_environment_uv(access_context, emission_access, in_direction);

      float image_pdf = 0.0f;
      if ((emission_access.emission_image_index == kInvalidIndex) || (emission_access.emission_image_index >= scene.images.count)) {
        return 0.0f;
      }

      const auto& img = scene.images[emission_access.emission_image_index];
      img.evaluate(uv, &image_pdf);
      return pdf_discrete * projection_environment_image_pdf_to_solid_angle(image_pdf, uv, projection_mode);
    }

    default: {
      ETX_FAIL("Unknown emitter class");
      return 0.0f;
    }
  }
}

ETX_SHARED_INLINE float2 emitter_environment_pdf(ETX_IN(float3, in_direction), bool target_is_surface, uint32_t target_triangle_index, const Scene& scene) {
  uint32_t environment_emitter_count = environment_emitter_shared_count(scene);
  if (environment_emitter_count == 0u) {
    return {};
  }

  float pdf_dir = 0.0f;
  for (uint32_t ie = 0; ie < environment_emitter_count; ++ie) {
    uint32_t emitter_index = kInvalidIndex;
    if (environment_emitter_shared_try_load_index(scene, ie, emitter_index) == false) {
      continue;
    }
    const auto& emitter_instance = scene.emitter_instances[emitter_index];
    pdf_dir += emitter_sample_pdf(emitter_instance, in_direction, scene);
  }

  float w_o_dot_n = 1.0f;
  if (target_is_surface) {
    if (target_triangle_index >= scene.triangles.count) {
      return {};
    }
    w_o_dot_n = fabsf(dot(scene.triangles[target_triangle_index].geo_n, in_direction));
  }

  float pdf_area = w_o_dot_n / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
  pdf_dir = pdf_dir / float(environment_emitter_count);
  return {pdf_area, pdf_dir};
}

ETX_SHARED_INLINE float emitter_ris_candidate_weight(const EmitterSample& emitter_sample, const EmitterSampleQuery& query) {
  float radiance_weight = emitter_sample.value.luminance();
  if (radiance_weight <= 0.0f) {
    return 0.0f;
  }

  float source_alignment = 1.0f;
  float3 to_emitter = emitter_sample.origin - query.source_position;
  float len_sq = dot(to_emitter, to_emitter);

  if (query.source_type == InteractionType::Surface) {
    source_alignment = fabsf(dot(query.source_normal, to_emitter) / sqrtf(len_sq));
  }

  if (emitter_sample.is_distant) {
    return radiance_weight * source_alignment;
  }

  float emitter_orientation = dot(emitter_sample.normal, -to_emitter);
  if ((emitter_orientation <= 0.0f) || (len_sq <= kEpsilon)) {
    return 0.0f;
  }

  float distance_weight = 1.0f / fmaxf(1.0f, len_sq);
  return radiance_weight * distance_weight * (emitter_orientation / sqrtf(len_sq)) * source_alignment;
}

ETX_SHARED_INLINE bool scene_has_only_environment_emitters(const Scene& scene) {
  uint32_t emitter_count = 0u;
  if (try_load_emitter_instance_count_shared(scene, emitter_count) == false) {
    return false;
  }
  uint32_t environment_emitter_count = min(static_cast<uint32_t>(scene.environment_emitters.count), uint32_t(SceneLimits::MaxEnvironmentEmitters));
  return (emitter_count > 0u) && (environment_emitter_count == emitter_count);
}

ETX_SHARED_INLINE EmitterSample sample_emitter(const Scene& scene, const EmitterSampleQuery& query, Sampler& smp) {
  if (scene.emitters_distribution.values.count == 0) {
    return {};
  }

  uint32_t emitter_instance_count = 0u;
  if (try_load_emitter_instance_count_shared(scene, emitter_instance_count) == false) {
    return {};
  }

  auto sampling_method = scene.light_sampling_method();

  if ((sampling_method == Scene::LightSampling::RIS_Uniform) || (sampling_method == Scene::LightSampling::RIS_FromDistribution)) {
    uint32_t candidate_count = min(4u * emitter_instance_count, 16u);

    float weight_sum = 0.0f;
    float selected_weight = 0.0f;
    EmitterSample selected_sample = {};

    for (uint32_t i = 0; i < candidate_count; ++i) {
      float pdf_sample = 0.0f;
      uint32_t emitter_index = kInvalidIndex;

      if (sampling_method == Scene::LightSampling::RIS_FromDistribution) {
        uint32_t dist_index = scene.emitters_distribution.sample(smp.next(), pdf_sample);
        ETX_ASSERT(dist_index < scene.emitters_distribution.values.count);
        emitter_index = scene.emitters_distribution.values[dist_index].reference;
      } else {
        if (scene_has_only_environment_emitters(scene)) {
          uint32_t emitter_count = 0u;
          if (try_select_environment_emitter_random(scene, smp, emitter_index, emitter_count) == false) {
            continue;
          }
          pdf_sample = 1.0f / float(emitter_count);
        } else {
          emitter_index = uint32_t(smp.next() * float(emitter_instance_count));
          pdf_sample = 1.0f / float(emitter_instance_count);
        }
      }

      EmitterAccessCPUContext access_context = make_emitter_access_cpu_context(scene);
      EmitterAccess emission_access = {};
      if (emitter_access_try_load(access_context, emitter_index, emission_access) == false) {
        continue;
      }

      const auto& emitter = scene.emitter_instances[emitter_index];
      EmitterSample sample = emitter_sample_in(emitter, query.spect, query.source_position, scene, smp.next_2d());
      sample.pdf_sample = pdf_sample;
      sample.emitter_index = emitter_index;
      sample.triangle_index = emitter.triangle_index;
      sample.is_delta = emitter.is_delta();
      sample.is_distant = emitter.is_distant();

      float candidate_weight = emitter_ris_candidate_weight(sample, query);
      float weight = candidate_weight / pdf_sample;

      weight_sum += weight;
      if (smp.next() * weight_sum < weight) {
        selected_sample = sample;
        selected_weight = weight;
      }
    }

    if (selected_weight <= 0.0f)
      return {};

    selected_sample.value *= weight_sum / (float(candidate_count) * selected_weight);
    return selected_sample;
  }

  float pdf_sample = 0.0f;
  uint32_t emitter_index = kInvalidIndex;

  if (sampling_method == Scene::LightSampling::FromDistribution) {
    if (scene.emitters_distribution.values.count == 0) {
      return {};
    }
    uint32_t dist_index = scene.emitters_distribution.sample(smp.next(), pdf_sample);
    ETX_ASSERT(dist_index < scene.emitters_distribution.values.count);
    emitter_index = scene.emitters_distribution.values[dist_index].reference;
  } else {
    if (scene_has_only_environment_emitters(scene)) {
      uint32_t emitter_count = 0u;
      if (try_select_environment_emitter_random(scene, smp, emitter_index, emitter_count) == false) {
        return {};
      }
      pdf_sample = 1.0f / float(emitter_count);
    } else {
      emitter_index = uint32_t(smp.next() * float(emitter_instance_count));
      pdf_sample = 1.0f / float(emitter_instance_count);
    }
  }

  EmitterAccessCPUContext access_context = make_emitter_access_cpu_context(scene);
  EmitterAccess emission_access = {};
  if (emitter_access_try_load(access_context, emitter_index, emission_access) == false) {
    return {};
  }

  const auto& emitter = scene.emitter_instances[emitter_index];
  EmitterSample sample = emitter_sample_in(emitter, query.spect, query.source_position, scene, smp.next_2d());
  sample.pdf_sample = pdf_sample;
  sample.emitter_index = emitter_index;
  sample.triangle_index = emitter.triangle_index;
  sample.is_delta = emitter.is_delta();
  sample.is_distant = emitter.is_distant();
  return sample;
}

ETX_SHARED_INLINE const EmitterSample sample_emission(const Scene& scene, SpectralQuery spect, Sampler& smp) {
  if (scene.emitters_distribution.values.count == 0) {
    return {};
  }

  uint32_t emitter_instance_count = 0u;
  if (try_load_emitter_instance_count_shared(scene, emitter_instance_count) == false) {
    return {};
  }
  EmitterSample result = {};
  uint32_t dist_index = scene.emitters_distribution.sample(smp.next(), result.pdf_sample);
  if ((dist_index == kInvalidIndex) || (dist_index >= scene.emitters_distribution.values.count)) {
    return {};
  }
  result.emitter_index = scene.emitters_distribution.values[dist_index].reference;
  if (result.emitter_index >= emitter_instance_count) {
    return {};
  }

  EmitterAccessCPUContext access_context = make_emitter_access_cpu_context(scene);
  EmitterAccess emission_access = {};
  if (emitter_access_try_load(access_context, result.emitter_index, emission_access) == false) {
    return {};
  }

  const auto& em_inst = scene.emitter_instances[result.emitter_index];
  switch (em_inst.cls) {
    case EmitterProfile::Class::Area: {
      const auto& tri = scene.triangles[em_inst.triangle_index];
      const Material& material = scene.materials[tri.material_index];

      result.triangle_index = em_inst.triangle_index;
      result.barycentric = random_barycentric(smp.next_2d());

      auto vertex = lerp_vertex(scene, tri, result.barycentric);
      result.origin = vertex.pos;
      result.normal = vertex.nrm;
      result.direction = sample_cosine_distribution(smp.next_2d(), result.normal, vertex.tan, vertex.btn, collimation_to_exponent(material.emission_collimation));
      result.value = emitter_evaluate_out_local(em_inst, spect, vertex.tex, result.normal, result.direction, result.pdf_area, result.pdf_dir, result.pdf_dir_out, scene);
      break;
    }

    case EmitterProfile::Class::Directional: {
      auto direction_to_scene = emission_access.emitter_direction * (-1.0f);
      float equivalent_disk_size = 0.0f;
      if (emission_access.emitter_angular_size_cosine > kEpsilon) {
        float sin_half_angle =
          sqrt(max(0.0f, 1.0f - (emission_access.emitter_angular_size_cosine * emission_access.emitter_angular_size_cosine)));
        equivalent_disk_size = 2.0f * (sin_half_angle / emission_access.emitter_angular_size_cosine);
      }
      auto basis = orthonormal_basis(direction_to_scene);
      auto pos_sample = sample_disk(smp.next_2d());
      auto dir_sample = sample_disk(smp.next_2d());
      result.direction = normalize(
        direction_to_scene + basis.u * dir_sample.x * (0.5f * equivalent_disk_size) + basis.v * dir_sample.y * (0.5f * equivalent_disk_size));
      result.triangle_index = kInvalidIndex;
      result.pdf_dir = 1.0f;
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir_out = result.pdf_dir * result.pdf_area;
      result.normal = direction_to_scene;
      result.origin = scene.bounding_sphere_center + scene.bounding_sphere_radius * (pos_sample.x * basis.u + pos_sample.y * basis.v - direction_to_scene);
      result.origin += result.direction * distance_to_sphere(result.origin, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.value =
        emitter_access_evaluate_spectral_source(access_context, emission_access.emission_spectrum_index, emission_access.emission_image_index, dir_sample * 0.5f + 0.5f, spect);
      break;
    }

    case EmitterProfile::Class::Environment: {
      if ((emission_access.emission_image_index == kInvalidIndex) || (emission_access.emission_image_index >= scene.images.count)) {
        return {};
      }

      const auto& img = scene.images[emission_access.emission_image_index];
      bool is_atmosphere = emitter_access_is_atmosphere(emission_access.emitter_profile_meta);
      uint32_t projection_mode = projection_environment_mode(is_atmosphere);
      float pdf_image = 0.0f;
      uint2 image_location = {};
      float4 image_value = {};
      float2 uv = img.sample(smp.next_2d(), pdf_image, image_location, image_value);
      if (pdf_image == 0.0f) {
        return {};
      }

      auto d = -uv_to_direction(uv, img.offset, img.scale.x, projection_mode);
      auto basis = orthonormal_basis(d);
      auto disk_sample = sample_disk(smp.next_2d());

      result.triangle_index = kInvalidIndex;
      result.direction = d;
      result.normal = result.direction;
      result.origin = scene.bounding_sphere_center + scene.bounding_sphere_radius * (disk_sample.x * basis.u + disk_sample.y * basis.v - result.direction);
      result.origin += result.direction * distance_to_sphere(result.origin, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.value = apply_rgb(spect, emitter_access_load_spectrum_spectral(access_context, emission_access.emission_spectrum_index, spect), image_value, scene);
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir = projection_environment_image_pdf_to_solid_angle(pdf_image, uv, projection_mode);
      result.pdf_dir_out = result.pdf_area * result.pdf_dir;
      ETX_VALIDATE(result.pdf_area);
      ETX_VALIDATE(result.pdf_dir);
      ETX_VALIDATE(result.pdf_dir_out);
      ETX_VALIDATE(result.value);
      break;
    }

    default: {
      ETX_FAIL("Unknown emitter class");
    }
  }
  result.triangle_index = em_inst.triangle_index;
  result.medium_index = emitter_external_medium_index(scene, em_inst);
  result.is_delta = em_inst.is_delta();
  result.is_distant = em_inst.is_distant();
  return result;
}

}  // namespace etx
