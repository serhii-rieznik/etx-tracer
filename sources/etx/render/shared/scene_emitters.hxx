#pragma once

namespace etx {

ETX_GPU_CODE float emitter_pdf_area_local(const Emitter& em, const Scene& scene) {
  ETX_ASSERT(em.is_local());
  return 1.0f / em.triangle_area;
}

ETX_GPU_CODE uint32_t emitter_external_medium_index(const Scene& scene, const Emitter& em_inst) {
  if ((em_inst.cls != EmitterProfile::Class::Area) || (em_inst.triangle_index >= scene.triangles.count)) {
    return kInvalidIndex;
  }
  const auto& tri = scene.triangles[em_inst.triangle_index];
  if (tri.material_index >= scene.materials.count) {
    return kInvalidIndex;
  }
  return scene.materials[tri.material_index].ext_medium;
}

ETX_GPU_CODE SpectralResponse emitter_evaluate_out_local(const Emitter& em_inst, const SpectralQuery spect, const float2& uv, const float3& emitter_normal,
  const float3& adirection, float& pdf_area, float& pdf_dir, float& pdf_dir_out, const Scene& scene) {
  const auto& em = scene.emitter_profiles[em_inst.profile];
  ETX_ASSERT(em_inst.is_local());

  pdf_dir = max(0.0f, dot(emitter_normal, adirection)) * kInvPi;
  if (pdf_dir <= 0.0f) {
    return {spect, 0.0f};
  }

  pdf_area = emitter_pdf_area_local(em_inst, scene);
  ETX_ASSERT(pdf_area > 0.0f);

  pdf_dir_out = pdf_dir * pdf_area;
  ETX_ASSERT(pdf_dir_out > 0.0f);

  return apply_image(spect, em.emission, uv, scene, nullptr);
}

ETX_GPU_CODE SpectralResponse emitter_get_radiance(const Emitter& em_inst, const SpectralQuery spect, const EmitterRadianceQuery& query, float& pdf_area, float& pdf_dir,
  float& pdf_dir_out, const Scene& scene) {
  const auto& em = scene.emitter_profiles[em_inst.profile];
  pdf_dir = 0.0f;
  pdf_area = 0.0f;
  pdf_dir_out = 0.0f;

  switch (em_inst.cls) {
    case EmitterProfile::Class::Directional: {
      if ((query.directly_visible == false) || (em.angular_size <= 0.0f) || (dot(query.direction, em.direction) < em.angular_size_cosine)) {
        return {spect, 0.0f};
      }

      pdf_dir = 1.0f;
      pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      pdf_dir_out = pdf_dir * pdf_area;
      float2 uv = disk_uv(em.direction, query.direction, em.equivalent_disk_size, em.angular_size_cosine);
      SpectralResponse direct_scale = 1.0f / (scene.spectrums[em.emission.spectrum_index](spect) * kDoublePi * (1.0f - em.angular_size_cosine));
      return apply_image(spect, em.emission, uv, scene, nullptr) * direct_scale;
    }

    case EmitterProfile::Class::Environment: {
      const auto& img = scene.images[em.emission.image_index];
      float2 uv = direction_to_uv(query.direction, img.offset, img.scale.x);
      auto sin_t = fmaxf(kEpsilon, sinf(uv.y * kPi));
      auto image_pdf = 0.0f;
      auto eval = apply_image(spect, em.emission, uv, scene, &image_pdf);
      pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      ETX_VALIDATE(pdf_area);
      pdf_dir = image_pdf / (2.0f * kPi * kPi * sin_t);
      ETX_VALIDATE(pdf_dir);
      pdf_dir_out = pdf_area * pdf_dir;
      ETX_VALIDATE(pdf_dir_out);
      return eval;
    }

    case EmitterProfile::Class::Area: {
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

      return apply_image(spect, em.emission, query.uv, scene, nullptr);
    }

    default: {
      ETX_FAIL("Unknown emitter class");
      return {spect, 0.0f};
    }
  }
}

ETX_GPU_CODE SpectralResponse emitter_evaluate_out_dist(const Emitter& em_inst, const SpectralQuery spect, const float3& in_direction, float& pdf_area, float& pdf_dir,
  const Scene& scene) {
  const auto& em = scene.emitter_profiles[em_inst.profile];
  ETX_ASSERT(em_inst.is_distant());

  pdf_dir = 0.0f;
  pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);

  switch (em_inst.cls) {
    case EmitterProfile::Class::Directional: {
      pdf_dir = 1.0f;
      float2 uv = disk_uv(em.direction, in_direction, em.equivalent_disk_size, em.angular_size_cosine);
      return apply_image(spect, em.emission, uv, scene, nullptr);
    }

    case EmitterProfile::Class::Environment: {
      const auto& img = scene.images[em.emission.image_index];
      float2 uv = direction_to_uv(in_direction, img.offset, 1.0f);
      auto sin_t = fmaxf(kEpsilon, sinf(uv.y * kPi));
      auto image_pdf = 0.0f;
      auto eval = apply_image(spect, em.emission, uv, scene, &image_pdf);
      pdf_dir = image_pdf / (2.0f * kPi * kPi * sin_t);
      ETX_VALIDATE(pdf_dir);
      return eval;
    }

    default:
      ETX_FAIL("Unknown emitter class");
      return {spect, 0.0f};
  }
}

ETX_GPU_CODE EmitterSample emitter_sample_in(const Emitter& em_inst, const SpectralQuery spect, const float3& from_point, const Scene& scene, const float2& smp) {
  const auto& em = scene.emitter_profiles[em_inst.profile];
  EmitterSample result;
  switch (em_inst.cls) {
    case EmitterProfile::Class::Area: {
      const auto& tri = scene.triangles[em_inst.triangle_index];
      result.barycentric = random_barycentric(smp);
      result.origin = lerp_pos(scene.vertices, tri, result.barycentric);
      result.normal = lerp_normal(scene.vertices, tri, result.barycentric);
      result.direction = normalize(result.origin - from_point);

      EmitterRadianceQuery q = {
        .source_position = from_point,
        .target_position = result.origin,
        .uv = lerp_uv(scene.vertices, tri, result.barycentric),
      };

      result.value = emitter_get_radiance(em_inst, spect, q, result.pdf_area, result.pdf_dir, result.pdf_dir_out, scene);
      break;
    }

    case EmitterProfile::Class::Directional: {
      float2 disk_sample = {};
      if (em.angular_size > 0.0f) {
        auto basis = orthonormal_basis(em.direction);
        disk_sample = sample_disk(smp);
        result.direction = normalize(em.direction + basis.u * disk_sample.x * (0.5f * em.equivalent_disk_size) + basis.v * disk_sample.y * (0.5f * em.equivalent_disk_size));
      } else {
        result.direction = em.direction;
      }
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir = 1.0f;
      result.pdf_dir_out = result.pdf_dir * result.pdf_area;
      result.origin = from_point + result.direction * distance_to_sphere(from_point, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.normal = em.direction * (-1.0f);
      result.value = apply_image(spect, em.emission, disk_sample * 0.5f + 0.5f, scene, nullptr);
      break;
    }

    case EmitterProfile::Class::Environment: {
      const auto& img = scene.images[em.emission.image_index];
      float pdf_image = 0.0f;
      uint2 image_location = {};
      float4 image_value = {};
      float2 uv = img.sample(smp, pdf_image, image_location, image_value);
      float sin_t = fmaxf(kEpsilon, sinf(uv.y * kPi));
      result.image_uv = uv;
      result.direction = uv_to_direction(result.image_uv, img.offset, img.scale.x);
      result.normal = -result.direction;
      result.origin = from_point + result.direction * distance_to_sphere(from_point, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.pdf_dir = pdf_image / (2.0f * kPi * kPi * sin_t);
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir_out = result.pdf_area * result.pdf_dir;
      result.value = apply_rgb(spect, scene.spectrums[em.emission.spectrum_index](spect), image_value, scene);
      break;
    }

    default: {
      ETX_FAIL("Unknown emitter class");
    }
  }

  result.medium_index = emitter_external_medium_index(scene, em_inst);
  return result;
}

ETX_GPU_CODE float emitter_discrete_pdf(const Emitter& emitter, const Distribution& dist) {
  return (emitter.spectrum_weight * emitter.additional_weight) / dist.total_weight;
}

ETX_GPU_CODE uint32_t sample_emitter_index(const Scene& scene, float rnd) {
  float pdf_sample = 0.0f;
  uint32_t emitter_index = static_cast<uint32_t>(scene.emitters_distribution.sample(rnd, pdf_sample));
  ETX_ASSERT(emitter_index < scene.emitters_distribution.values.count);
  return emitter_index;
}

ETX_GPU_CODE EmitterSample sample_emitter(SpectralQuery spect, uint32_t emitter_index, const float2& smp, const float3& from_point, const Scene& scene) {
  const auto& emitter = scene.emitter_instances[emitter_index];
  EmitterSample sample = emitter_sample_in(emitter, spect, from_point, scene, smp);
  sample.pdf_sample = emitter_discrete_pdf(emitter, scene.emitters_distribution);
  sample.emitter_index = emitter_index;
  sample.triangle_index = emitter.triangle_index;
  sample.is_delta = emitter.is_delta();
  return sample;
}

ETX_GPU_CODE const EmitterSample sample_emission(const Scene& scene, SpectralQuery spect, Sampler& smp) {
  EmitterSample result = {};
  result.emitter_index = scene.emitters_distribution.sample(smp.next(), result.pdf_sample);
  ETX_ASSERT(result.emitter_index < scene.emitter_instances.count);

  const auto& em_inst = scene.emitter_instances[result.emitter_index];
  const auto& em = scene.emitter_profiles[em_inst.profile];
  switch (em_inst.cls) {
    case EmitterProfile::Class::Area: {
      const auto& tri = scene.triangles[em_inst.triangle_index];
      const Material& material = scene.materials[tri.material_index];

      result.triangle_index = em_inst.triangle_index;
      result.barycentric = random_barycentric(smp.next_2d());

      auto vertex = lerp_vertex(scene.vertices, tri, result.barycentric);
      result.origin = vertex.pos;
      result.normal = vertex.nrm;
      result.direction = sample_cosine_distribution(smp.next_2d(), result.normal, vertex.tan, vertex.btn, collimation_to_exponent(material.emission_collimation));
      result.value = emitter_evaluate_out_local(em_inst, spect, vertex.tex, result.normal, result.direction, result.pdf_area, result.pdf_dir, result.pdf_dir_out, scene);
      break;
    }

    case EmitterProfile::Class::Directional: {
      auto direction_to_scene = em.direction * (-1.0f);
      auto basis = orthonormal_basis(direction_to_scene);
      auto pos_sample = sample_disk(smp.next_2d());
      auto dir_sample = sample_disk(smp.next_2d());
      result.direction = normalize(direction_to_scene + basis.u * dir_sample.x * (0.5f * em.equivalent_disk_size) + basis.v * dir_sample.y * (0.5f * em.equivalent_disk_size));
      result.triangle_index = kInvalidIndex;
      result.pdf_dir = 1.0f;
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir_out = result.pdf_dir * result.pdf_area;
      result.normal = direction_to_scene;
      result.origin = scene.bounding_sphere_center + scene.bounding_sphere_radius * (pos_sample.x * basis.u + pos_sample.y * basis.v - direction_to_scene);
      result.origin += result.direction * distance_to_sphere(result.origin, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.value = apply_image(spect, em.emission, dir_sample * 0.5f + 0.5f, scene, nullptr);
      break;
    }

    case EmitterProfile::Class::Environment: {
      const auto& img = scene.images[em.emission.image_index];
      float pdf_image = 0.0f;
      uint2 image_location = {};
      float4 image_value = {};
      float2 uv = img.sample(smp.next_2d(), pdf_image, image_location, image_value);
      if (pdf_image == 0.0f) {
        return {};
      }

      auto sin_t = fmaxf(kEpsilon, sinf(uv.y * kPi));
      auto d = -uv_to_direction(uv, img.offset, img.scale.x);
      auto basis = orthonormal_basis(d);
      auto disk_sample = sample_disk(smp.next_2d());

      result.triangle_index = kInvalidIndex;
      result.direction = d;
      result.normal = result.direction;
      result.origin = scene.bounding_sphere_center + scene.bounding_sphere_radius * (disk_sample.x * basis.u + disk_sample.y * basis.v - result.direction);
      result.origin += result.direction * distance_to_sphere(result.origin, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.value = apply_rgb(spect, scene.spectrums[em.emission.spectrum_index](spect), image_value, scene);
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir = pdf_image / (2.0f * kPi * kPi * sin_t);
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
