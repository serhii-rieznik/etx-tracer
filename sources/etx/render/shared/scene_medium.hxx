#pragma once

namespace etx {

namespace {

ETX_SHARED_INLINE constexpr float medium_gamma(int n) {
  constexpr auto e = kEpsilon * 0.5f;
  return (n * e) / (1.0f - n * e);
}

ETX_SHARED_INLINE bool medium_bounds(const Medium& medium, const float3& in_pos, const float3& in_dir, float max_t, float& t_min, float& t_max) {
  constexpr float g3 = 1.0f + 2.0f * medium_gamma(3);

  float pos[3] = {in_pos.x, in_pos.y, in_pos.z};
  float dir[3] = {in_dir.x, in_dir.y, in_dir.z};

  t_min = 0.0f;
  t_max = max_t;
  for (int i = 0; i < 3; ++i) {
    float t_near = (0.0f - pos[i]) / dir[i];
    float t_far = (1.0f - pos[i]) / dir[i];

    if (t_near > t_far) {
      float t = t_far;
      t_far = t_near;
      t_near = t;
    }

    t_far *= g3;

    t_min = t_near > t_min ? t_near : t_min;
    ETX_CHECK_FINITE(t_min);
    t_max = t_far < t_max ? t_far : t_max;
    ETX_CHECK_FINITE(t_max);
    if (t_min > t_max)
      return false;
  }

  return true;
}

ETX_SHARED_INLINE bool medium_intersects_bounds(const Medium& medium, const float3& in_pos, const float3& in_direction, const float in_max_t, float3& medium_pos, float3& medium_dir,
  float& t_min, float& t_max, float3& world_dir_normalized, float3& bbox_size) {
  if (in_max_t >= kMaxFloat) {
    return false;
  }

  medium_pos = medium.bounds.to_local(in_pos);
  ETX_CHECK_FINITE(medium_pos);

  float3 end_pos = in_pos + in_direction * in_max_t;
  ETX_CHECK_FINITE(end_pos);
  float3 medium_end_pos = medium.bounds.to_local(end_pos);
  ETX_CHECK_FINITE(medium_end_pos);

  medium_dir = medium_end_pos - medium_pos;
  float d_len = dot(medium_dir, medium_dir);
  constexpr float kTreshold = kRayEpsilon * kRayEpsilon;
  if (d_len <= kTreshold) {
    return false;
  }

  medium_dir *= 1.0f / sqrtf(d_len);
  ETX_CHECK_FINITE(medium_dir);

  world_dir_normalized = normalize(in_direction);
  bbox_size = medium.bounds.p_max - medium.bounds.p_min;

  float segment = length(medium_end_pos - medium_pos);
  ETX_CHECK_FINITE(segment);
  return medium_bounds(medium, medium_pos, medium_dir, segment, t_min, t_max);
}

}  // namespace

ETX_SHARED_INLINE uint32_t sample_spectrum_component(const SpectralQuery spect, const SpectralResponse& albedo, const SpectralResponse& throughput, const float rnd,
  SpectralResponse& pdf) {
  if (spect.spectral()) {
    pdf = {spect, 1.0f};
    return 0;
  }

  SpectralResponse at = albedo * throughput;

  if (at.is_zero()) {
    pdf = {spect, 1.0f / at.component_count()};
    return uint32_t(at.component_count() * rnd);
  }

  pdf = at / at.sum();
  return 2u - uint32_t(rnd < pdf.integrated.x + pdf.integrated.y) - uint32_t(rnd < pdf.integrated.x);
}

ETX_SHARED_INLINE SpectralResponse calculate_albedo(const SpectralQuery spect, const SpectralResponse& scattering, const SpectralResponse& extinction) {
  SpectralResponse albedo = {spect, extinction.value > 0.0f ? (scattering.value / extinction.value) : 0.0f};
  albedo.integrated.x = extinction.integrated.x > 0.0f ? (scattering.integrated.x / extinction.integrated.x) : 0.0f;
  albedo.integrated.y = extinction.integrated.y > 0.0f ? (scattering.integrated.y / extinction.integrated.y) : 0.0f;
  albedo.integrated.z = extinction.integrated.z > 0.0f ? (scattering.integrated.z / extinction.integrated.z) : 0.0f;
  return albedo;
}

ETX_SHARED_INLINE float phase_function(const float3& w_i, const float3& w_o, const float g) {
  float cos_t = dot(w_i, w_o);
  float d = 1.0f + g * g - 2.0f * g * cos_t;
  return (1.0f / (4.0f * kPi)) * (1.0f - g * g) / (d * sqrtf(d));
}

ETX_SHARED_INLINE float3 sample_phase_function(const float3& w_i, const float g, const float2& smp_rnd) {
  float cos_theta = 0.0f;
  if (fabsf(g) < 1e-3f) {
    cos_theta = 1.0f - 2.0f * smp_rnd.x;
  } else {
    float sqr_term = (1.0f - g * g) / (1.0f + g * (2.0f * smp_rnd.x - 1.0f));
    cos_theta = (1.0f + g * g - sqr_term * sqr_term) / (2.0f * g);
  }

  float sin_theta = sqrtf(max(0.0f, 1.0f - cos_theta * cos_theta));
  float phi = kDoublePi * smp_rnd.y;

  auto basis = orthonormal_basis(w_i);
  return (basis.u * cosf(phi) + basis.v * sinf(phi)) * sin_theta - w_i * cos_theta;
}

ETX_SHARED_INLINE SpectralResponse medium_absorption(const Scene& scene, const Medium& medium, const SpectralQuery spect) {
  if ((medium.absorption_index == kInvalidIndex) || (medium.absorption_index >= scene.spectrums.count)) {
    return {spect, 0.0f};
  }
  return scene.spectrums[medium.absorption_index](spect);
}

ETX_SHARED_INLINE SpectralResponse medium_scattering(const Scene& scene, const Medium& medium, const SpectralQuery spect) {
  if ((medium.scattering_index == kInvalidIndex) || (medium.scattering_index >= scene.spectrums.count)) {
    return {spect, 0.0f};
  }
  return scene.spectrums[medium.scattering_index](spect);
}

ETX_SHARED_INLINE SpectralResponse medium_extinction(const Scene& scene, const Medium& medium, const SpectralQuery spect) {
  return medium_absorption(scene, medium, spect) + medium_scattering(scene, medium, spect);
}

ETX_SHARED_INLINE Medium::Instance make_medium_instance(const Scene& scene, const Medium& medium, const SpectralQuery spect, uint32_t index) {
  Medium::Instance result = {};
  result.extinction = medium_extinction(scene, medium, spect);
  result.anisotropy = medium.phase_function_g;
  result.index = index;
  return result;
}

ETX_SHARED_INLINE SpectralResponse medium_transmittance(const Medium::Instance& instance, float distance) {
  return spectrum_exp(instance.extinction * (-distance));
}

ETX_SHARED_INLINE SpectralResponse medium_transmittance(const Scene& scene, const Medium& medium, const SpectralQuery spect, Sampler& smp, const float3& pos, const float3& direction,
  float distance) {
  switch (medium.cls) {
    case Medium::Class::Homogeneous:
      return spectrum_exp(medium_extinction(scene, medium, spect) * (-distance));

    case Medium::Class::Heterogeneous: {
      SpectralResponse base_extinction = medium_extinction(scene, medium, spect);
      float max_sigma = base_extinction.maximum();
      if (max_sigma <= 0.0f) {
        return {spect, 1.0f};
      }

      float3 medium_pos = pos;
      float3 medium_dir = direction;
      float t_min = 0.0f;
      float t_max = 0.0f;
      float3 world_dir_normalized = {};
      float3 bbox_size = {};
      if (medium_intersects_bounds(medium, pos, direction, distance, medium_pos, medium_dir, t_min, t_max, world_dir_normalized, bbox_size) == false) {
        return {spect, 1.0f};
      }
      const float rr_threshold = 0.1f;
      SpectralResponse transmittance = {spect, 1.0f};

      float t_world = 0.0f;
      while (true) {
        t_world += -logf(1.0f - smp.next()) / max_sigma;
        float3 world_pos_at_t = pos + world_dir_normalized * t_world;
        float3 local_pos = medium.bounds.to_local(world_pos_at_t);
        float t_local_along_dir = dot(local_pos - medium_pos, medium_dir);
        if (t_local_along_dir >= (t_max - t_min)) {
          break;
        }

        float density_value = medium.grid.sample(local_pos, medium.bounds);
        SpectralResponse extinction_at_point = base_extinction * density_value;
        SpectralResponse weight = SpectralResponse{spect, 1.0f} - extinction_at_point / max_sigma;
        transmittance *= spectrum_max(0.0f, weight);
        ETX_VALIDATE(transmittance);

        float transmittance_max = transmittance.maximum();
        if (transmittance_max < rr_threshold) {
          float p = clamp(transmittance_max, 0.01f, 0.95f);
          if (smp.next() > p) {
            return {spect, 0.0f};
          }
          transmittance *= 1.0f / p;
          ETX_VALIDATE(transmittance);
        }
      }
      return transmittance;
    }

    default:
      ETX_FAIL_FMT("Invalid medium: %u\n", uint32_t(medium.cls));
      return {};
  }
}

ETX_SHARED_INLINE Medium::Sample sample_medium(const Scene& scene, const Medium& medium, const SpectralQuery spect, const SpectralResponse& throughput, Sampler& smp, const float3& pos,
  const float3& w_i, float max_t) {
  ETX_CRITICAL(max_t > 0.0f);

  SpectralResponse scattering_value = medium_scattering(scene, medium, spect);
  ETX_VALIDATE(scattering_value);
  SpectralResponse absorption_value = medium_absorption(scene, medium, spect);
  ETX_VALIDATE(absorption_value);
  SpectralResponse extinction_value = scattering_value + absorption_value;
  ETX_VALIDATE(extinction_value);
  SpectralResponse albedo = calculate_albedo(spect, scattering_value, extinction_value);
  ETX_VALIDATE(albedo);

  switch (medium.cls) {
    case Medium::Class::Homogeneous: {
      float t = 0.0f;
      SpectralResponse pdf = {};
      while (t < kRayEpsilon) {
        uint32_t channel = sample_spectrum_component(spect, albedo, throughput, smp.next(), pdf);
        float sample_t = extinction_value.component(channel);
        t = (sample_t > 0.0f) ? -logf(1.0f - smp.next()) / sample_t : max_t;
        ETX_VALIDATE(t);
      }

      t = min(t, max_t);
      ETX_VALIDATE(t);

      bool sampled_medium = t < max_t;

      SpectralResponse tr = spectrum_exp(-t * extinction_value);
      pdf *= sampled_medium ? tr * extinction_value : tr;

      if (pdf.is_zero())
        return {{spect, 0.0f}};

      Medium::Sample result = {};
      result.pos = pos + w_i * t;
      result.sampled_medium_t = sampled_medium ? t : 0.0f;
      result.weight = (sampled_medium ? tr * scattering_value : tr) / pdf.sum();
      ETX_VALIDATE(result.weight);
      return result;
    }

    case Medium::Class::Heterogeneous: {
      float max_sigma = extinction_value.maximum();
      if ((max_sigma <= 0.0f) || (medium.grid.has_data() == false)) {
        Medium::Sample result = {};
        result.weight = {spect, 1.0f};
        result.pos = pos + w_i * max_t;
        result.sampled_medium_t = 0.0f;
        return result;
      }

      float3 medium_pos = {};
      float3 medium_dir = {};
      float t_min = 0.0f;
      float t_max = 0.0f;
      float3 world_dir_normalized = {};
      float3 bbox_size = {};
      if (medium_intersects_bounds(medium, pos, w_i, max_t, medium_pos, medium_dir, t_min, t_max, world_dir_normalized, bbox_size) == false) {
        Medium::Sample result = {};
        result.weight = {spect, 1.0f};
        result.pos = pos + w_i * max_t;
        result.sampled_medium_t = 0.0f;
        return result;
      }

      SpectralResponse pdf = {};
      uint32_t channel = sample_spectrum_component(spect, albedo, throughput, smp.next(), pdf);

      SpectralResponse transmittance = {spect, 1.0f};
      const float rr_threshold = 0.1f;
      float t_world = 0.0f;
      float segment_length = t_max - t_min;

      while (true) {
        t_world += -logf(1.0f - smp.next()) / max_sigma;
        float3 world_pos_at_t = pos + world_dir_normalized * t_world;
        float3 local_pos = medium.bounds.to_local(world_pos_at_t);
        float t_local_along_dir = dot(local_pos - medium_pos, medium_dir);
        if (t_local_along_dir >= segment_length) {
          pdf *= transmittance;
          Medium::Sample result = {};
          result.pos = pos + world_dir_normalized * min(t_world, max_t);
          result.sampled_medium_t = 0.0f;
          result.weight = pdf.is_zero() ? SpectralResponse{spect, 0.0f} : transmittance / pdf.sum();
          ETX_VALIDATE(result.weight);
          return result;
        }

        float density_value = medium.grid.sample(local_pos, medium.bounds);
        SpectralResponse extinction_at_point = extinction_value * density_value;
        float sigma_t_channel = extinction_at_point.component(channel);

        if ((sigma_t_channel > 0.0f) && (smp.next() < sigma_t_channel / max_sigma)) {
          SpectralResponse scattering_at_point = scattering_value * density_value;
          pdf *= transmittance * extinction_at_point;
          if (pdf.is_zero()) {
            return {{spect, 0.0f}};
          }

          Medium::Sample result = {};
          result.pos = world_pos_at_t;
          result.sampled_medium_t = t_world;
          result.weight = (transmittance * scattering_at_point) / pdf.sum();
          ETX_VALIDATE(result.weight);
          return result;
        }

        SpectralResponse weight = SpectralResponse{spect, 1.0f} - extinction_at_point / max_sigma;
        transmittance *= spectrum_max(0.0f, weight);
        ETX_VALIDATE(transmittance);

        float transmittance_max = transmittance.maximum();
        if (transmittance_max < rr_threshold) {
          float p = fminf(fmaxf(transmittance_max, 0.01f), 0.95f);
          if (smp.next() > p) {
            return {{spect, 0.0f}};
          }
          transmittance *= 1.0f / p;
          ETX_VALIDATE(transmittance);
        }
      }
    }

    default:
      ETX_FAIL_FMT("Invalid medium: %u\n", uint32_t(medium.cls));
      return {};
  }
}

ETX_SHARED_INLINE float medium_phase_function(const Medium& medium, const float3& w_i, const float3& w_o) {
  return phase_function(w_i, w_o, medium.phase_function_g);
}

ETX_SHARED_INLINE float3 medium_sample_phase_function(const Medium& medium, const float2& smp_rnd, const float3& w_i) {
  return sample_phase_function(w_i, medium.phase_function_g, smp_rnd);
}

}  // namespace etx
