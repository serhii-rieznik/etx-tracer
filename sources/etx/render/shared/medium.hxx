#pragma once

#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/sampler.hxx>

namespace etx {

struct alignas(16) Medium {
  enum class Class : uint32_t {
    Vacuum,
    Homogeneous,
    Heterogeneous,
  };

  struct alignas(16) Sample {
    SpectralResponse weight = {};
    float3 pos = {};
    float t = {};

    ETX_GPU_CODE bool sampled_medium() const {
      return t > 0.0f;
    }

    ETX_GPU_CODE bool valid() const {
      return weight.valid();
    }
  };

  SpectralDistribution s_absorption = {};
  SpectralDistribution s_outscattering = {};
  ArrayView<float> density = {};
  BoundingBox bounds = {};
  Class cls = Class::Vacuum;
  float phase_function_g = 0.0f;
  float max_sigma = 0.0f;
  uint3 dimensions = {};
  uint32_t pad = {};

  ETX_GPU_CODE SpectralResponse transmittance(const SpectralQuery spect, Sampler& smp, const float3& pos, const float3& direction, float distance) const {
    switch (cls) {
      case Class::Vacuum:
        return {spect.wavelength, 1.0f};

      case Class::Homogeneous:
        return transmittance_homogeneous(spect, smp, pos, direction, distance);

      case Class::Heterogeneous:
        return transmittance_heterogeneous(spect, smp, pos, direction, distance);

      default:
        ETX_FAIL_FMT("Invalid medium: %u\n", uint32_t(cls));
        return {};
    }
  }

  ETX_GPU_CODE SpectralResponse transmittance_homogeneous(const SpectralQuery spect, Sampler& smp, const float3& pos, const float3& direction, float distance) const {
    return exp((s_outscattering(spect) + s_absorption(spect)) * (-distance));
  }

  ETX_GPU_CODE SpectralResponse transmittance_heterogeneous(const SpectralQuery spect, Sampler& smp, const float3& p, const float3& d, float max_t) const {
    float3 pos = p;
    float3 dir = d;
    float t_min = 0.0f;
    float t_max = 0.0f;
    if (intersects_medium_bounds(p, d, max_t, pos, dir, t_min, t_max) == false) {
      return {spect.wavelength, 1.0f};
    }

    const float rr_threshold = 0.1f;

    float tr = 1.0f;

    float t = t_min;
    while (true) {
      t -= logf(1.0f - smp.next()) / max_sigma;
      if (t >= t_max)
        break;

      float density = sample_density(pos + dir * t);
      tr *= max(0.0f, 1.0f - density);

      if (tr < rr_threshold) {
        float q = max(0.05f, 1.0f - tr);

        if (smp.next() < q)
          return {spect.wavelength, 0.0f};

        tr /= 1.0f - q;
      }
    }

    return {spect.wavelength, tr};
  }

  ETX_GPU_CODE Sample sample(const SpectralQuery spect, Sampler& smp, const float3& pos, const float3& w_i, float max_t) const {
    switch (cls) {
      case Class::Vacuum:
        return {{spect.wavelength, 1.0f}};

      case Class::Homogeneous:
        return sample_homogeneous(spect, smp, pos, w_i, max_t);

      case Class::Heterogeneous:
        return sample_heterogeneous(spect, smp, pos, w_i, max_t);

      default:
        ETX_FAIL_FMT("Invalid medium: %u\n", uint32_t(cls));
        return {};
    }
  }

  ETX_GPU_CODE Sample sample_homogeneous(const SpectralQuery spect, Sampler& smp, const float3& pos, const float3& w_i, float max_t) const {
    float rnd = smp.next();
    auto s_t = s_outscattering.random_entry_power(rnd) + s_absorption.random_entry_power(rnd);
    float t = -logf(1.0f - smp.next()) / s_t;
    t = min(t, max_t);

    bool sampled_medium = t < max_t;

    Sample result = {};
    result.pos = pos + w_i * t;
    result.t = sampled_medium ? t : 0.0f;

    SpectralResponse tr = transmittance_homogeneous(spect, smp, pos, w_i, t);
    SpectralResponse density_spectrum = sampled_medium ? (tr * (s_outscattering(spect) + s_absorption(spect))) : (tr);

    float density = density_spectrum.average();
    float pdf = density;
    if (pdf == 0.0f) {
      ETX_ASSERT(tr.is_zero());
      pdf = 1.0f;
    }

    result.weight = tr / pdf;
    ETX_VALIDATE(result.weight);

    if (result.sampled_medium()) {
      result.weight *= s_outscattering(spect);
      ETX_VALIDATE(result.weight);
    }

    return result;
  }

  ETX_GPU_CODE Sample sample_heterogeneous(const SpectralQuery spect, Sampler& smp, const float3& in_pos, const float3& in_dir, float in_max_t) const {
    float3 pos = in_pos;
    float3 dir = in_dir;
    float t_min = 0.0f;
    float t_max = 0.0f;
    if (intersects_medium_bounds(in_pos, in_dir, in_max_t, pos, dir, t_min, t_max) == false) {
      return {{spect.wavelength, 1.0f}};
    }

    float t = t_min;
    while (true) {
      t -= logf(1.0f - smp.next()) / max_sigma;
      if (t >= t_max)
        break;

      float3 local_pos = pos + dir * t;
      float density = sample_density(local_pos);
      if (smp.next() <= density) {
        Sample result;
        result.weight = s_outscattering(spect) / (s_outscattering(spect) + s_absorption(spect));
        result.pos = bounds.from_bounding_box(local_pos);
        result.t = t;
        return result;
      }
    }

    return {{spect.wavelength, 1.0f}};
  }

  ETX_GPU_CODE float phase_function(const SpectralQuery spect, const float3& pos, const float3& w_i, const float3& w_o) const {
    if (cls == Class::Vacuum) {
      return 1.0f;
    }

    float cos_t = dot(w_i, w_o);
    float d = 1.0f + phase_function_g * phase_function_g + 2.0f * phase_function_g * cos_t;
    return (1.0f / (4.0f * kPi)) * (1.0f - phase_function_g * phase_function_g) / (d * sqrtf(d));
  }

  ETX_GPU_CODE float3 sample_phase_function(const SpectralQuery spect, Sampler& smp, const float3& pos, const float3& w_i) const {
    if ((cls == Class::Vacuum) || s_outscattering.is_zero()) {
      return w_i;
    }

    float xi0 = smp.next();
    float xi1 = smp.next();

    float cos_theta = 0.0f;
    if (fabsf(phase_function_g) < 1e-3) {
      cos_theta = 1.0f - 2.0f * xi0;
    } else {
      float sqr_term = (1.0f - phase_function_g * phase_function_g) / (1.0f + phase_function_g - 2.0f * phase_function_g * xi0);
      cos_theta = -(1.0f + phase_function_g * phase_function_g - sqr_term * sqr_term) / (2.0f * phase_function_g);
    }

    float sin_theta = sqrtf(max(0.0f, 1.0f - cos_theta * cos_theta));

    auto basis = orthonormal_basis(w_i);
    return (basis.u * cosf(xi1 * kDoublePi) + basis.v * sinf(xi1 * kDoublePi)) * sin_theta + w_i * cos_theta;
  }

  ETX_GPU_CODE float sample_density(const float3& coord) const {
    ETX_ASSERT(cls == Class::Heterogeneous);

    if ((coord.x < 0.0f) || (coord.y < 0.0f) || (coord.z < 0.0f) || (coord.x >= 1.0f) || (coord.y >= 1.0f) || (coord.z >= 1.0f)) {
      return 0.0f;
    }

    float px = clamp(coord.x * float(dimensions.x) - 0.5f, 0.0f, float(dimensions.x) - 1.0f);
    float py = clamp(coord.y * float(dimensions.y) - 0.5f, 0.0f, float(dimensions.y) - 1.0f);
    float pz = clamp(coord.z * float(dimensions.z) - 0.5f, 0.0f, float(dimensions.z) - 1.0f);

    uint64_t ix = min(dimensions.x - 1llu, static_cast<uint64_t>(px));
    uint64_t nx = min(dimensions.x - 1llu, ix + 1);

    uint64_t iy = min(dimensions.y - 1llu, static_cast<uint64_t>(py));
    uint64_t ny = min(dimensions.y - 1llu, iy + 1);

    uint64_t iz = min(dimensions.z - 1llu, static_cast<uint64_t>(pz));
    uint64_t nz = min(dimensions.z - 1llu, iz + 1);

    float d000 = density[ix + iy * dimensions.x + iz * dimensions.x * dimensions.y];
    float d001 = density[nx + iy * dimensions.x + iz * dimensions.x * dimensions.y];
    float d010 = density[ix + ny * dimensions.x + iz * dimensions.x * dimensions.y];
    float d011 = density[nx + ny * dimensions.x + iz * dimensions.x * dimensions.y];
    float d100 = density[ix + iy * dimensions.x + nz * dimensions.x * dimensions.y];
    float d101 = density[nx + iy * dimensions.x + nz * dimensions.x * dimensions.y];
    float d110 = density[ix + ny * dimensions.x + nz * dimensions.x * dimensions.y];
    float d111 = density[nx + ny * dimensions.x + nz * dimensions.x * dimensions.y];

    float dx = px - floorf(px);
    float dy = py - floorf(py);
    float dz = pz - floorf(pz);

    float d_bottom = lerp(lerp(d000, d001, dx), lerp(d010, d011, dx), dy);
    float d_top = lerp(lerp(d100, d101, dx), lerp(d110, d111, dx), dy);
    return lerp(d_bottom, d_top, dz);
  }

 private:
  ETX_GPU_CODE constexpr static float gamma(int n) {
    constexpr auto e = kEpsilon * 0.5f;
    return (n * e) / (1.0f - n * e);
  }

  ETX_GPU_CODE bool medium_bounds(const float3& in_pos, const float3& in_dir, const float max_t, float& t_min, float& t_max) const {
    constexpr float g3 = 1.0f + 2.0f * gamma(3);

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

  ETX_GPU_CODE bool intersects_medium_bounds(const float3& in_pos, const float3& in_direction, float in_max_t, float3& medium_pos, float3& medium_dir, float& t_min,
    float& t_max) const {
    if (in_max_t >= kMaxFloat) {
      return false;
    }

    float3 end_pos = in_pos + in_direction * in_max_t;
    float3 medium_end_pos = bounds.to_bounding_box(end_pos);

    medium_pos = bounds.to_bounding_box(in_pos);
    medium_dir = normalize(medium_end_pos - medium_pos);
    in_max_t = length(medium_end_pos - medium_pos);

    return medium_bounds(medium_pos, medium_dir, in_max_t, t_min, t_max);
  }
};

}  // namespace etx