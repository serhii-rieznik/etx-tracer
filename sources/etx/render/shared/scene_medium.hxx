#pragma once

namespace etx {

namespace {

ETX_GPU_CODE constexpr float medium_gamma(int n) {
  constexpr auto e = kEpsilon * 0.5f;
  return (n * e) / (1.0f - n * e);
}

ETX_GPU_CODE bool medium_bounds(const Medium& medium, const float3& in_pos, const float3& in_dir, float max_t, float& t_min, float& t_max) {
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
    t_max = t_far < t_max ? t_far : t_max;

    if (t_min > t_max)
      return false;
  }

  return true;
}

ETX_GPU_CODE bool medium_intersects_bounds(const Medium& medium, const float3& in_pos, const float3& in_direction, float in_max_t, float3& medium_pos, float3& medium_dir,
  float& t_min, float& t_max) {
  if (in_max_t >= kMaxFloat) {
    return false;
  }

  float3 end_pos = in_pos + in_direction * in_max_t;
  float3 medium_end_pos = medium.bounds.to_local(end_pos);

  medium_pos = medium.bounds.to_local(in_pos);
  medium_dir = normalize(medium_end_pos - medium_pos);
  float segment = length(medium_end_pos - medium_pos);

  return medium_bounds(medium, medium_pos, medium_dir, segment, t_min, t_max);
}

ETX_GPU_CODE float medium_sample_density_internal(const Medium& medium, const float3& coord) {
  ETX_ASSERT(medium.cls == Medium::Class::Heterogeneous);

  if ((coord.x < 0.0f) || (coord.y < 0.0f) || (coord.z < 0.0f) || (coord.x >= 1.0f) || (coord.y >= 1.0f) || (coord.z >= 1.0f)) {
    return 0.0f;
  }

  float px = clamp(coord.x * float(medium.dimensions.x) - 0.5f, 0.0f, float(medium.dimensions.x) - 1.0f);
  float py = clamp(coord.y * float(medium.dimensions.y) - 0.5f, 0.0f, float(medium.dimensions.y) - 1.0f);
  float pz = clamp(coord.z * float(medium.dimensions.z) - 0.5f, 0.0f, float(medium.dimensions.z) - 1.0f);

  uint32_t ix = min(medium.dimensions.x - 1u, static_cast<uint32_t>(px));
  uint32_t nx = min(medium.dimensions.x - 1u, ix + 1u);

  uint32_t iy = min(medium.dimensions.y - 1u, static_cast<uint32_t>(py));
  uint32_t ny = min(medium.dimensions.y - 1u, iy + 1u);

  uint32_t iz = min(medium.dimensions.z - 1u, static_cast<uint32_t>(pz));
  uint32_t nz = min(medium.dimensions.z - 1u, iz + 1u);

  auto density = medium.density;
  float d000 = density[ix + iy * medium.dimensions.x + iz * medium.dimensions.x * medium.dimensions.y];
  float d001 = density[nx + iy * medium.dimensions.x + iz * medium.dimensions.x * medium.dimensions.y];
  float d010 = density[ix + ny * medium.dimensions.x + iz * medium.dimensions.x * medium.dimensions.y];
  float d011 = density[nx + ny * medium.dimensions.x + iz * medium.dimensions.x * medium.dimensions.y];
  float d100 = density[ix + iy * medium.dimensions.x + nz * medium.dimensions.x * medium.dimensions.y];
  float d101 = density[nx + iy * medium.dimensions.x + nz * medium.dimensions.x * medium.dimensions.y];
  float d110 = density[ix + ny * medium.dimensions.x + nz * medium.dimensions.x * medium.dimensions.y];
  float d111 = density[nx + ny * medium.dimensions.x + nz * medium.dimensions.x * medium.dimensions.y];

  float dx = px - floorf(px);
  float dy = py - floorf(py);
  float dz = pz - floorf(pz);

  float d_bottom = lerp(lerp(d000, d001, dx), lerp(d010, d011, dx), dy);
  float d_top = lerp(lerp(d100, d101, dx), lerp(d110, d111, dx), dy);
  return lerp(d_bottom, d_top, dz);
}

}  // namespace

ETX_GPU_CODE uint32_t sample_spectrum_component(const SpectralQuery spect, const SpectralResponse& albedo, const SpectralResponse& throughput, const float rnd,
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

ETX_GPU_CODE SpectralResponse calculate_albedo(const SpectralQuery spect, const SpectralResponse& scattering, const SpectralResponse& extinction) {
  SpectralResponse albedo = {spect, extinction.value > 0.0f ? (scattering.value / extinction.value) : 0.0f};
  albedo.integrated.x = extinction.integrated.x > 0.0f ? (scattering.integrated.x / extinction.integrated.x) : 0.0f;
  albedo.integrated.y = extinction.integrated.y > 0.0f ? (scattering.integrated.y / extinction.integrated.y) : 0.0f;
  albedo.integrated.z = extinction.integrated.z > 0.0f ? (scattering.integrated.z / extinction.integrated.z) : 0.0f;
  return albedo;
}

ETX_GPU_CODE float phase_function(const float3& w_i, const float3& w_o, const float g) {
  float cos_t = dot(w_i, w_o);
  float d = 1.0f + g * g - 2.0f * g * cos_t;
  return (1.0f / (4.0f * kPi)) * (1.0f - g * g) / (d * sqrtf(d));
}

ETX_GPU_CODE float3 sample_phase_function(const float3& w_i, const float g, const float2& smp_rnd) {
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

ETX_GPU_CODE SpectralResponse medium_absorption(const Scene& scene, const Medium& medium, const SpectralQuery spect) {
  if ((medium.absorption_index == kInvalidIndex) || (medium.absorption_index >= scene.spectrums.count)) {
    return {spect, 0.0f};
  }
  return scene.spectrums[medium.absorption_index](spect);
}

ETX_GPU_CODE SpectralResponse medium_scattering(const Scene& scene, const Medium& medium, const SpectralQuery spect) {
  if ((medium.scattering_index == kInvalidIndex) || (medium.scattering_index >= scene.spectrums.count)) {
    return {spect, 0.0f};
  }
  return scene.spectrums[medium.scattering_index](spect);
}

ETX_GPU_CODE SpectralResponse medium_extinction(const Scene& scene, const Medium& medium, const SpectralQuery spect) {
  return medium_absorption(scene, medium, spect) + medium_scattering(scene, medium, spect);
}

ETX_GPU_CODE float3 medium_absorption_rgb(const Scene& scene, const Medium& medium) {
  if ((medium.absorption_index == kInvalidIndex) || (medium.absorption_index >= scene.spectrums.count)) {
    return {};
  }
  return scene.spectrums[medium.absorption_index].integrated();
}

ETX_GPU_CODE float3 medium_scattering_rgb(const Scene& scene, const Medium& medium) {
  if ((medium.scattering_index == kInvalidIndex) || (medium.scattering_index >= scene.spectrums.count)) {
    return {};
  }
  return scene.spectrums[medium.scattering_index].integrated();
}

ETX_GPU_CODE Medium::Instance make_medium_instance(const Scene& scene, const Medium& medium, const SpectralQuery spect, uint32_t index) {
  Medium::Instance result = {};
  result.extinction = medium_extinction(scene, medium, spect);
  result.anisotropy = medium.phase_function_g;
  result.index = index;
  return result;
}

ETX_GPU_CODE SpectralResponse medium_transmittance(const Medium::Instance& instance, float distance) {
  return exp(instance.extinction * (-distance));
}

ETX_GPU_CODE SpectralResponse medium_transmittance(const Scene& scene, const Medium& medium, const SpectralQuery spect, Sampler& smp, const float3& pos, const float3& direction,
  float distance) {
  switch (medium.cls) {
    case Medium::Class::Homogeneous:
      return exp(medium_extinction(scene, medium, spect) * (-distance));

    case Medium::Class::Heterogeneous: {
      if (medium.max_sigma <= 0.0f) {
        return {spect, 1.0f};
      }

      float3 medium_pos = pos;
      float3 medium_dir = direction;
      float t_min = 0.0f;
      float t_max = 0.0f;
      if (medium_intersects_bounds(medium, pos, direction, distance, medium_pos, medium_dir, t_min, t_max) == false) {
        return {spect, 1.0f};
      }

      const float rr_threshold = 0.1f;
      float transmittance = 1.0f;

      float t = t_min;
      while (true) {
        t -= logf(1.0f - smp.next()) / medium.max_sigma;
        if (t >= t_max) {
          break;
        }

        float density_value = medium_sample_density_internal(medium, medium_pos + medium_dir * t);
        transmittance *= max(0.0f, 1.0f - density_value);

        if (transmittance < rr_threshold) {
          float q = max(0.05f, 1.0f - transmittance);
          if (smp.next() < q) {
            return {spect, 0.0f};
          }
          transmittance /= (1.0f - q);
        }
      }

      return {spect, transmittance};
    }

    default:
      ETX_FAIL_FMT("Invalid medium: %u\n", uint32_t(medium.cls));
      return {};
  }
}

ETX_GPU_CODE Medium::Sample sample_medium(const Scene& scene, const Medium& medium, const SpectralQuery spect, const SpectralResponse& throughput, Sampler& smp, const float3& pos,
  const float3& w_i, float max_t) {
  ETX_CRITICAL(max_t > 0.0f);

  switch (medium.cls) {
    case Medium::Class::Homogeneous: {
      SpectralResponse scattering_value = medium_scattering(scene, medium, spect);
      ETX_VALIDATE(scattering_value);
      SpectralResponse absorption_value = medium_absorption(scene, medium, spect);
      ETX_VALIDATE(absorption_value);
      SpectralResponse extinction_value = scattering_value + absorption_value;
      ETX_VALIDATE(extinction_value);
      SpectralResponse albedo = calculate_albedo(spect, scattering_value, extinction_value);
      ETX_VALIDATE(albedo);

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

      SpectralResponse tr = exp(-t * extinction_value);
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
      Medium::Sample result = {};
      if (medium.max_sigma <= 0.0f) {
        return result;
      }

      float3 medium_pos = pos;
      float3 medium_dir = w_i;
      float t_min = 0.0f;
      float t_max = 0.0f;
      if (medium_intersects_bounds(medium, pos, w_i, max_t, medium_pos, medium_dir, t_min, t_max) == false) {
        return result;
      }

      SpectralResponse scattering_value = medium_scattering(scene, medium, spect);
      SpectralResponse extinction_value = medium_extinction(scene, medium, spect);
      SpectralResponse albedo = calculate_albedo(spect, scattering_value, extinction_value);

      float t = t_min;
      float previous_t = t_min;
      SpectralResponse accumulated_transmittance = {spect, 1.0f};

      while (true) {
        t -= logf(1.0f - smp.next()) / medium.max_sigma;
        if (t >= t_max) {
          break;
        }

        float distance = max(0.0f, t - previous_t);
        accumulated_transmittance *= exp(-extinction_value * distance);
        ETX_VALIDATE(accumulated_transmittance);
        previous_t = t;

        float density_value = medium_sample_density_internal(medium, medium_pos + medium_dir * t);
        if (density_value * medium.max_sigma == 0.0f) {
          continue;
        }

        SpectralResponse pdf = {};
        uint32_t channel = sample_spectrum_component(spect, albedo, scattering_value, smp.next(), pdf);
        float sigma_t = extinction_value.component(channel);

        float random = smp.next();
        if ((sigma_t > 0.0f) && (random < density_value)) {
          float pdf_sum = pdf.sum();
          if (pdf_sum > 0.0f) {
            result.weight = (scattering_value * accumulated_transmittance) / pdf_sum;
          } else {
            result.weight = scattering_value * accumulated_transmittance;
          }
          ETX_VALIDATE(result.weight);
          result.pos = medium_pos + medium_dir * t;
          result.sampled_medium_t = t - t_min;
          return result;
        }
      }

      float remaining_distance = max(0.0f, t_max - previous_t);
      accumulated_transmittance *= exp(-extinction_value * remaining_distance);
      ETX_VALIDATE(accumulated_transmittance);
      result.weight = accumulated_transmittance;
      return result;
    }

    default:
      ETX_FAIL_FMT("Invalid medium: %u\n", uint32_t(medium.cls));
      return {};
  }
}

ETX_GPU_CODE float medium_phase_function(const Medium& medium, const float3& w_i, const float3& w_o) {
  return phase_function(w_i, w_o, medium.phase_function_g);
}

ETX_GPU_CODE float3 medium_sample_phase_function(const Medium& medium, const float2& smp_rnd, const float3& w_i) {
  return sample_phase_function(w_i, medium.phase_function_g, smp_rnd);
}

}  // namespace etx
