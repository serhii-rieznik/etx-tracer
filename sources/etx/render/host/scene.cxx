#pragma once

#include <etx/render/host/scene_global.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/access/image_sample_cpu.hxx>

namespace etx {

namespace {

float emitter_discrete_pdf_from_distribution(const Emitter& emitter, const Distribution& dist) {
  if (dist.total_weight == 0.0f) {
    return 0.0f;
  }

  return (emitter.spectrum_weight * emitter.additional_weight) / dist.total_weight;
}

}  // namespace

float4 sample_whole_image(const SampledImage& img, const float2& uv) {
  if (img.image_index == kInvalidIndex) {
    return img.value;
  }

  const auto& scene = scene_global_get();
  ETX_ASSERT(img.image_index < static_cast<uint32_t>(scene.images.count));
  float4 eval = scene.images[img.image_index].evaluate(uv, nullptr);
  return img.value * eval;
}

float evaluate_image_channel(uint32_t image_index, uint32_t channel, const float2& uv, float default_value) {
  if ((image_index == kInvalidIndex) || (channel >= 4u)) {
    return default_value;
  }

  const auto& scene = scene_global_get();
  if (image_index >= static_cast<uint32_t>(scene.images.count)) {
    return default_value;
  }

  float4 eval = scene.images[image_index].evaluate(uv, nullptr);
  const float* data = reinterpret_cast<const float*>(&eval);
  return data[channel];
}

bool image_has_alpha_channel(uint32_t image_index) {
  if (image_index == kInvalidIndex) {
    return false;
  }

  const auto& scene = scene_global_get();
  if (image_index >= static_cast<uint32_t>(scene.images.count)) {
    return false;
  }

  return (scene.images[image_index].options & Image::HasAlphaChannel) != 0u;
}

float2 sample_image_uv(uint32_t image_index, const float2& rnd) {
  if (image_index == kInvalidIndex) {
    return rnd;
  }

  const auto& scene = scene_global_get();
  ImageSampleCPUContext image_context = make_image_sample_cpu_context(scene);
  ImageSampleAccess image_sample = image_sample_access_default(rnd);
  if (image_sample_try_sample(image_context, image_index, rnd, image_sample) == false) {
    return rnd;
  }

  return image_sample.uv;
}

float2 sample_image_uv(uint32_t image_index, const float2& rnd, float& pdf, uint2& location, float4& value) {
  pdf = 0.0f;
  location = {};
  value = {};

  if (image_index == kInvalidIndex) {
    return rnd;
  }

  const auto& scene = scene_global_get();
  ImageSampleCPUContext image_context = make_image_sample_cpu_context(scene);
  ImageSampleAccess image_sample = image_sample_access_default(rnd);
  if (image_sample_try_sample(image_context, image_index, rnd, image_sample) == false) {
    return rnd;
  }

  pdf = image_sample.pdf;
  location = image_sample.location;
  value = image_sample.value;
  return image_sample.uv;
}

float evaluate_image(const SampledImage& img, const float2& uv, const float default_value) {
  return evaluate_image_channel(img.image_index, img.channel, uv, default_value);
}

RefractiveIndexSample evaluate_refractive_index(const RefractiveIndex& ri, const SpectralQuery q) {
  RefractiveIndexSample result = {};
  result.cls = ri.cls;
  result.eta = (ri.eta_index == kInvalidIndex) ? SpectralResponse(q, 1.0f) : medium_load_spectrum_or_zero(ri.eta_index, q);
  result.k = (ri.k_index == kInvalidIndex) ? SpectralResponse(q, 0.0f) : medium_load_spectrum_or_zero(ri.k_index, q);
  return result;
}

uint32_t default_dielectric_eta_index() {
  return scene_global_get().defaults.dielectric_eta;
}

uint32_t default_conductor_eta_index() {
  return scene_global_get().defaults.conductor_eta;
}

uint32_t default_conductor_k_index() {
  return scene_global_get().defaults.conductor_k;
}

SpectralResponse medium_load_spectrum_or_zero(uint32_t spectrum_index, const SpectralQuery spect) {
  if (spectrum_index == kInvalidIndex) {
    return {spect, 0.0f};
  }

  const auto& scene = scene_global_get();
  if (spectrum_index >= static_cast<uint32_t>(scene.spectrums.count)) {
    return {spect, 0.0f};
  }

  return scene.spectrums[spectrum_index](spect);
}

bool try_load_emitter_instance_count_shared(uint32_t& emitter_instance_count) {
  const auto& scene = scene_global_get();
  emitter_instance_count = static_cast<uint32_t>(scene.emitter_instances.count);
  return (emitter_instance_count > 0u) && (scene.emitter_profiles.count > 0u);
}

uint32_t environment_emitter_shared_count() {
  const auto& scene = scene_global_get();
  if ((scene.emitter_instances.count == 0u) || (scene.environment_emitters.count == 0u)) {
    return 0u;
  }

  return min(static_cast<uint32_t>(scene.environment_emitters.count), uint32_t(SceneLimits::MaxEnvironmentEmitters));
}

bool environment_emitter_shared_try_load_index(uint32_t local_index, uint32_t& emitter_index) {
  const auto& scene = scene_global_get();
  emitter_index = kInvalidIndex;

  uint32_t emitter_count = environment_emitter_shared_count();
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

bool try_select_environment_emitter_random(Sampler& smp, uint32_t& emitter_index, uint32_t& emitter_count) {
  emitter_index = kInvalidIndex;
  emitter_count = environment_emitter_shared_count();
  if (emitter_count == 0u) {
    return false;
  }

  uint32_t selected = uint32_t(smp.next() * float(emitter_count));
  if (selected >= emitter_count) {
    selected = emitter_count - 1u;
  }

  return environment_emitter_shared_try_load_index(selected, emitter_index);
}

uint32_t emitter_external_medium_index(const Emitter& em_inst) {
  const auto& scene = scene_global_get();

  if (em_inst.cls == EmitterProfile::Class::Area) {
    if (em_inst.triangle_index >= scene.triangles.count) {
      return kInvalidIndex;
    }

    const auto& tri = scene.triangles[em_inst.triangle_index];
    if (tri.material_index >= scene.materials.count) {
      return kInvalidIndex;
    }

    return scene.materials[tri.material_index].ext_medium;
  }

  if (em_inst.profile >= scene.emitter_profiles.count) {
    return kInvalidIndex;
  }

  return scene.emitter_profiles[em_inst.profile].medium_index;
}

SpectralResponse emitter_evaluate_out_local(const Emitter& em_inst, const SpectralQuery spect, const float2& uv, const float3& emitter_normal, const float3& direction,
  float& pdf_area, float& pdf_dir, float& pdf_dir_out) {
  const auto& scene = scene_global_get();
  const auto& em = scene.emitter_profiles[em_inst.profile];
  ETX_ASSERT(em_inst.is_local());

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

  pdf_area = emitter_pdf_area_local(em_inst);
  ETX_ASSERT(pdf_area > 0.0f);

  pdf_dir_out = pdf_dir * pdf_area;
  ETX_ASSERT(pdf_dir_out > 0.0f);

  return apply_image(spect, em.emission, uv);
}

SpectralResponse emitter_evaluate_out_dist(const Emitter& em_inst, const SpectralQuery spect, const float3& in_direction, float& pdf_area, float& pdf_dir) {
  const auto& scene = scene_global_get();
  const auto& em = scene.emitter_profiles[em_inst.profile];
  ETX_ASSERT(em_inst.is_distant());

  pdf_dir = 0.0f;
  pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);

  switch (em_inst.cls) {
    case EmitterProfile::Class::Directional: {
      pdf_dir = 1.0f;
      float2 uv = disk_uv(em.directional.direction, in_direction, em.directional.equivalent_disk_size, em.directional.angular_size_cosine);
      return apply_image(spect, em.emission, uv);
    }

    case EmitterProfile::Class::Environment: {
      const auto& img = scene.images[em.emission.image_index];
      bool is_atmosphere = (em.meta & EmitterProfile::Meta::Atmosphere) != 0u;
      uint32_t projection = projection_environment_mode(is_atmosphere);
      float2 uv = direction_to_uv(in_direction, img.offset, 1.0f, projection);

      float image_pdf = 0.0f;
      SpectralResponse eval = apply_image(spect, em.emission, uv, image_pdf);
      pdf_dir = projection_environment_image_pdf_to_solid_angle(image_pdf, uv, projection);
      ETX_VALIDATE(pdf_dir);
      return eval;
    }

    default: {
      ETX_FAIL("Unknown emitter class");
      return {spect, 0.0f};
    }
  }
}

SpectralResponse emitter_get_radiance(const Emitter& em_inst, const SpectralQuery spect, const EmitterRadianceQuery& query, float& pdf_area, float& pdf_dir,
  float& pdf_dir_out) {
  const auto& scene = scene_global_get();
  const auto& em = scene.emitter_profiles[em_inst.profile];
  pdf_dir = 0.0f;
  pdf_area = 0.0f;
  pdf_dir_out = 0.0f;

  switch (em_inst.cls) {
    case EmitterProfile::Class::Directional: {
      if ((query.directly_visible == false) || (em.directional.angular_size <= 0.0f) || (dot(query.direction, em.directional.direction) < em.directional.angular_size_cosine)) {
        return {spect, 0.0f};
      }

      pdf_dir = 1.0f;
      pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      pdf_dir_out = pdf_dir * pdf_area;
      float2 uv = disk_uv(em.directional.direction, query.direction, em.directional.equivalent_disk_size, em.directional.angular_size_cosine);
      SpectralResponse direct_scale = 1.0f / (scene.spectrums[em.emission.spectrum_index](spect) * kDoublePi * (1.0f - em.directional.angular_size_cosine));
      return apply_image(spect, em.emission, uv) * direct_scale;
    }

    case EmitterProfile::Class::Environment: {
      const auto& img = scene.images[em.emission.image_index];
      bool is_atmosphere = (em.meta & EmitterProfile::Meta::Atmosphere) != 0u;
      uint32_t projection = projection_environment_mode(is_atmosphere);
      float2 uv = direction_to_uv(query.direction, img.offset, img.scale.x, projection);

      float image_pdf = 0.0f;
      SpectralResponse eval = apply_image(spect, em.emission, uv, image_pdf);
      pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      ETX_VALIDATE(pdf_area);
      pdf_dir = projection_environment_image_pdf_to_solid_angle(image_pdf, uv, projection);
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
      pdf_area = emitter_pdf_area_local(em_inst);

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

      return apply_image(spect, em.emission, query.uv);
    }

    default: {
      ETX_FAIL("Unknown emitter class");
      return {spect, 0.0f};
    }
  }
}

float emitter_discrete_pdf(const Emitter& emitter) {
  const auto& scene = scene_global_get();
  return emitter_discrete_pdf_from_distribution(emitter, scene.emitters_distribution);
}

float emitter_sample_pdf(const Emitter& em_inst, ETX_IN(float3, in_direction)) {
  const auto& scene = scene_global_get();
  if (em_inst.profile >= scene.emitter_profiles.count) {
    return 0.0f;
  }

  const auto& em = scene.emitter_profiles[em_inst.profile];
  float pdf_discrete = emitter_discrete_pdf(em_inst);

  switch (em_inst.cls) {
    case EmitterProfile::Class::Area: {
      return pdf_discrete * emitter_pdf_area_local(em_inst);
    }

    case EmitterProfile::Class::Directional: {
      float cosine_threshold = (em.directional.angular_size > 0.0f) ? em.directional.angular_size_cosine : 1.0f;
      return direction_matches(in_direction, em.directional.direction, cosine_threshold) ? pdf_discrete : 0.0f;
    }

    case EmitterProfile::Class::Environment: {
      if ((em.emission.image_index == kInvalidIndex) || (em.emission.image_index >= scene.images.count)) {
        return 0.0f;
      }

      const auto& img = scene.images[em.emission.image_index];
      bool is_atmosphere = (em.meta & EmitterProfile::Meta::Atmosphere) != 0u;
      uint32_t projection = projection_environment_mode(is_atmosphere);
      float2 uv = direction_to_uv(in_direction, img.offset, img.scale.x, projection);

      float image_pdf = 0.0f;
      img.evaluate(uv, &image_pdf);
      return pdf_discrete * projection_environment_image_pdf_to_solid_angle(image_pdf, uv, projection);
    }

    default: {
      ETX_FAIL("Unknown emitter class");
      return 0.0f;
    }
  }
}

float2 emitter_environment_pdf(ETX_IN(float3, in_direction), bool target_is_surface, uint32_t target_triangle_index) {
  const auto& scene = scene_global_get();
  uint32_t environment_emitter_count = environment_emitter_shared_count();
  if (environment_emitter_count == 0u) {
    return {};
  }

  float pdf_dir = 0.0f;
  for (uint32_t ie = 0; ie < environment_emitter_count; ++ie) {
    uint32_t emitter_index = kInvalidIndex;
    if (environment_emitter_shared_try_load_index(ie, emitter_index) == false) {
      continue;
    }

    pdf_dir += emitter_sample_pdf(scene.emitter_instances[emitter_index], in_direction);
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

bool emitter_distribution_has_values() {
  const auto& scene = scene_global_get();
  return scene.emitters_distribution.values.count != 0u;
}

uint32_t sample_emitter_distribution(Sampler& smp, float& pdf_sample) {
  const auto& scene = scene_global_get();
  pdf_sample = 0.0f;
  if (scene.emitters_distribution.values.count == 0u) {
    return kInvalidIndex;
  }

  uint32_t dist_index = scene.emitters_distribution.sample(smp.next(), pdf_sample);
  if ((dist_index == kInvalidIndex) || (dist_index >= scene.emitters_distribution.values.count)) {
    return kInvalidIndex;
  }

  return scene.emitters_distribution.values[dist_index].reference;
}

bool try_load_emitter_instance(uint32_t emitter_index, Emitter& emitter) {
  const auto& scene = scene_global_get();
  emitter = {};
  if (emitter_index >= scene.emitter_instances.count) {
    return false;
  }

  emitter = scene.emitter_instances[emitter_index];
  return true;
}

EmitterSample emitter_sample_in(const Emitter& em_inst, const SpectralQuery spect, const float3& from_point, const float2& smp) {
  const auto& scene = scene_global_get();
  const auto& em = scene.emitter_profiles[em_inst.profile];
  EmitterSample result = {};
  switch (em_inst.cls) {
    case EmitterProfile::Class::Area: {
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

      result.value = emitter_get_radiance(em_inst, spect, q, result.pdf_area, result.pdf_dir, result.pdf_dir_out);
      break;
    }

    case EmitterProfile::Class::Directional: {
      float2 disk_sample = {};
      if (em.directional.angular_size > 0.0f) {
        auto basis = orthonormal_basis(em.directional.direction);
        disk_sample = sample_disk(smp);
        result.direction = normalize(em.directional.direction + basis.u * disk_sample.x * (0.5f * em.directional.equivalent_disk_size) +
                                     basis.v * disk_sample.y * (0.5f * em.directional.equivalent_disk_size));
      } else {
        result.direction = em.directional.direction;
      }
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir = 1.0f;
      result.pdf_dir_out = result.pdf_dir * result.pdf_area;
      result.origin = from_point + result.direction * distance_to_sphere(from_point, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.normal = em.directional.direction * (-1.0f);
      result.value = apply_image(spect, em.emission, disk_sample * 0.5f + 0.5f);
      break;
    }

    case EmitterProfile::Class::Environment: {
      const auto& img = scene.images[em.emission.image_index];
      bool is_atmosphere = (em.meta & EmitterProfile::Meta::Atmosphere) != 0u;
      uint32_t projection = projection_environment_mode(is_atmosphere);
      float pdf_image = 0.0f;
      uint2 image_location = {};
      float4 image_value = {};
      float2 uv = img.sample(smp, pdf_image, image_location, image_value);

      result.image_uv = uv;
      result.direction = uv_to_direction(result.image_uv, img.offset, img.scale.x, projection);
      result.normal = -result.direction;
      result.origin = from_point + result.direction * distance_to_sphere(from_point, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.pdf_dir = projection_environment_image_pdf_to_solid_angle(pdf_image, uv, projection);
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir_out = result.pdf_area * result.pdf_dir;
      result.value = apply_rgb(spect, scene.spectrums[em.emission.spectrum_index](spect), image_value);
      break;
    }

    default: {
      ETX_FAIL("Unknown emitter class");
    }
  }

  result.medium_index = emitter_external_medium_index(em_inst);
  return result;
}

EmitterSample sample_emission_from_emitter(const Emitter& em_inst, const SpectralQuery spect, Sampler& smp) {
  const auto& scene = scene_global_get();
  const auto& em = scene.emitter_profiles[em_inst.profile];
  EmitterSample result = {};

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
      result.value = emitter_evaluate_out_local(em_inst, spect, vertex.tex, result.normal, result.direction, result.pdf_area, result.pdf_dir, result.pdf_dir_out);
      break;
    }

    case EmitterProfile::Class::Directional: {
      auto direction_to_scene = em.directional.direction * (-1.0f);
      float equivalent_disk_size = 0.0f;
      if (em.directional.angular_size_cosine > kEpsilon) {
        float sin_half_angle = sqrt(max(0.0f, 1.0f - (em.directional.angular_size_cosine * em.directional.angular_size_cosine)));
        equivalent_disk_size = 2.0f * (sin_half_angle / em.directional.angular_size_cosine);
      }

      auto basis = orthonormal_basis(direction_to_scene);
      auto pos_sample = sample_disk(smp.next_2d());
      auto dir_sample = sample_disk(smp.next_2d());
      result.direction = normalize(direction_to_scene + basis.u * dir_sample.x * (0.5f * equivalent_disk_size) + basis.v * dir_sample.y * (0.5f * equivalent_disk_size));
      result.triangle_index = kInvalidIndex;
      result.pdf_dir = 1.0f;
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir_out = result.pdf_dir * result.pdf_area;
      result.normal = direction_to_scene;
      result.origin = scene.bounding_sphere_center + scene.bounding_sphere_radius * (pos_sample.x * basis.u + pos_sample.y * basis.v - direction_to_scene);
      result.origin += result.direction * distance_to_sphere(result.origin, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.value = apply_image(spect, em.emission, dir_sample * 0.5f + 0.5f);
      break;
    }

    case EmitterProfile::Class::Environment: {
      if ((em.emission.image_index == kInvalidIndex) || (em.emission.image_index >= scene.images.count)) {
        return {};
      }

      const auto& img = scene.images[em.emission.image_index];
      bool is_atmosphere = (em.meta & EmitterProfile::Meta::Atmosphere) != 0u;
      uint32_t projection = projection_environment_mode(is_atmosphere);
      float pdf_image = 0.0f;
      uint2 image_location = {};
      float4 image_value = {};
      float2 uv = img.sample(smp.next_2d(), pdf_image, image_location, image_value);
      if (pdf_image == 0.0f) {
        return {};
      }

      auto d = -uv_to_direction(uv, img.offset, img.scale.x, projection);
      auto basis = orthonormal_basis(d);
      auto disk_sample = sample_disk(smp.next_2d());

      result.triangle_index = kInvalidIndex;
      result.direction = d;
      result.normal = result.direction;
      result.origin = scene.bounding_sphere_center + scene.bounding_sphere_radius * (disk_sample.x * basis.u + disk_sample.y * basis.v - result.direction);
      result.origin += result.direction * distance_to_sphere(result.origin, result.direction, scene.bounding_sphere_center, scene.bounding_sphere_radius);
      result.value = apply_rgb(spect, scene.spectrums[em.emission.spectrum_index](spect), image_value);
      result.pdf_area = 1.0f / (kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius);
      result.pdf_dir = projection_environment_image_pdf_to_solid_angle(pdf_image, uv, projection);
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
  result.medium_index = emitter_external_medium_index(em_inst);
  result.is_delta = em_inst.is_delta();
  result.is_distant = em_inst.is_distant();
  return result;
}

SpectralResponse apply_image(SpectralQuery spect, const SpectralImage& img, const float2& uv, float& image_pdf) {
  image_pdf = 0.0f;

  const auto& scene = scene_global_get();
  ETX_ASSERT(img.spectrum_index < static_cast<uint32_t>(scene.spectrums.count));
  SpectralResponse result = scene.spectrums[img.spectrum_index](spect);
  ETX_VALIDATE(result);
  if (img.image_index == kInvalidIndex) {
    return result;
  }
  float4 eval = scene.images[img.image_index].evaluate(uv, &image_pdf);
  ETX_VALIDATE(eval);
  return apply_rgb(spect, result, eval);
}

SpectralResponse apply_image(SpectralQuery spect, const SpectralImage& img, const float2& uv) {
  const auto& scene = scene_global_get();
  ETX_ASSERT(img.spectrum_index < static_cast<uint32_t>(scene.spectrums.count));
  SpectralResponse result = scene.spectrums[img.spectrum_index](spect);
  ETX_VALIDATE(result);
  if (img.image_index == kInvalidIndex) {
    return result;
  }
  float4 eval = scene.images[img.image_index].evaluate(uv, nullptr);
  ETX_VALIDATE(eval);
  return apply_rgb(spect, result, eval);
}

}  // namespace etx
