#pragma once

#include <etx/render/shared/medium.hxx>

namespace etx {

struct MediumTransmittanceSharedContext {
  const Medium* medium = nullptr;
  BoundingBox bounds = {};
  Sampler* sampler = nullptr;
};

ETX_SHARED_INLINE float medium_transmittance_shared_rnd(ETX_INOUT(MediumTransmittanceSharedContext, context)) {
  ETX_ASSERT(context.sampler != nullptr);
  return context.sampler->next();
}

ETX_SHARED_INLINE float medium_transmittance_shared_density(ETX_INOUT(MediumTransmittanceSharedContext, context), ETX_IN(float3, local_pos)) {
  ETX_ASSERT(context.medium != nullptr);
  return context.medium->sample_density(local_pos, context.bounds);
}

ETX_SHARED_INLINE SpectralResponse medium_transmittance_shared_to_spectral_response(ETX_IN(::SpectralResponse, response)) {
  SpectralQuery query = {response.wavelength, response.flags};
  if (::spectral_response_is_spectral(response)) {
    return {query, response.value};
  }
  return {query, response.integrated};
}

#define ETX_MEDIUM_SHARED_CONTEXT_TYPE MediumTransmittanceSharedContext
#define ETX_MEDIUM_SHARED_RND(context) medium_transmittance_shared_rnd(context)
#define ETX_MEDIUM_SHARED_DENSITY(context, local_pos) medium_transmittance_shared_density(context, local_pos)
#include <etx/render/interop/medium_transmittance_shared.hxx>
#undef ETX_MEDIUM_SHARED_DENSITY
#undef ETX_MEDIUM_SHARED_RND
#undef ETX_MEDIUM_SHARED_CONTEXT_TYPE

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

ETX_SHARED_INLINE MediumInstance make_medium_instance(const Scene& scene, const Medium& medium, const SpectralQuery spect, uint32_t index) {
  MediumInstance result = {};
  result.extinction = medium_extinction(scene, medium, spect);
  result.anisotropy = medium.phase_function_g;
  result.index = index;
  return result;
}

ETX_SHARED_INLINE SpectralResponse medium_transmittance(const Scene& scene, const Medium& medium, const SpectralQuery spect, Sampler& smp, const float3& pos,
  const float3& direction, float distance) {
  switch (medium.cls) {
    case Medium::Homogeneous: {
      const ::SpectralResponse extinction = static_cast<const ::SpectralResponse&>(medium_extinction(scene, medium, spect));
      const ::SpectralResponse transmittance = medium_shared_transmittance_homogeneous_spectral(extinction, distance);
      return medium_transmittance_shared_to_spectral_response(transmittance);
    }

    case Medium::Heterogeneous: {
      SpectralResponse base_extinction = medium_extinction(scene, medium, spect);
      float max_sigma = base_extinction.maximum();
      if (max_sigma <= 0.0f) {
        return {spect, 1.0f};
      }

      MediumTransmittanceSharedContext context = {};
      context.medium = &medium;
      context.bounds = medium.bounds;
      context.sampler = &smp;
      const ::SpectralResponse transmittance = medium_shared_transmittance_heterogeneous_spectral(static_cast<const ::SpectralResponse&>(base_extinction), pos, direction, distance,
        medium.bounds.p_min, medium.bounds.p_max, context, static_cast<const ::SpectralQuery&>(spect));
      return medium_transmittance_shared_to_spectral_response(transmittance);
    }

    default:
      ETX_FAIL_FMT("Invalid medium: %u\n", uint32_t(medium.cls));
      return {};
  }
}

ETX_SHARED_INLINE MediumSample sample_medium(const Scene& scene, const Medium& medium, const SpectralQuery spect, const SpectralResponse& throughput, Sampler& smp,
  const float3& pos, const float3& w_i, float max_t) {
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
    case Medium::Homogeneous: {
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
        return {spectral_response_make(spect, 0.0f)};

      MediumSample result = {};
      result.pos = pos + w_i * t;
      result.sampled_medium_t = sampled_medium ? t : 0.0f;
      result.weight = (sampled_medium ? tr * scattering_value : tr) / pdf.sum();
      ETX_VALIDATE(result.weight);
      return result;
    }

    case Medium::Heterogeneous: {
      float max_sigma = extinction_value.maximum();
      if ((max_sigma <= 0.0f) || (medium.has_grid_data() == false)) {
        MediumSample result = {};
        result.weight = spectral_response_make(spect, 0.0f), result.pos = pos + w_i * max_t;
        result.sampled_medium_t = 0.0f;
        return result;
      }

      const BoundingBox bounds_box = medium.bounds;
      MediumSharedIntersection medium_intersection = {};
      if (medium_shared_intersects_bounds(bounds_box.p_min, bounds_box.p_max, pos, w_i, max_t, medium_intersection) == false) {
        MediumSample result = {
          .weight = spectral_response_make(spect, 1.0f),
          .pos = pos + w_i * max_t,
          .sampled_medium_t = 0.0f,
        };
        return result;
      }

      SpectralResponse pdf = {};
      uint32_t channel = sample_spectrum_component(spect, albedo, throughput, smp.next(), pdf);

      SpectralResponse transmittance = {spect, 1.0f};
      const float rr_threshold = 0.1f;
      float t_world = 0.0f;
      float segment_length = medium_intersection.t_max - medium_intersection.t_min;

      while (true) {
        t_world += -logf(1.0f - smp.next()) / max_sigma;
        float3 world_pos_at_t = pos + medium_intersection.world_dir_normalized * t_world;
        float3 local_pos = medium_shared_bounds_to_local(world_pos_at_t, bounds_box.p_min, bounds_box.p_max);
        float t_local_along_dir = dot(local_pos - medium_intersection.medium_pos, medium_intersection.medium_dir);
        if (t_local_along_dir >= segment_length) {
          pdf *= transmittance;
          MediumSample result = {};
          result.pos = pos + medium_intersection.world_dir_normalized * min(t_world, max_t);
          result.sampled_medium_t = 0.0f;
          result.weight = pdf.is_zero() ? SpectralResponse{spect, 0.0f} : transmittance / pdf.sum();
          ETX_VALIDATE(result.weight);
          return result;
        }

        float density_value = medium.sample_density(local_pos, bounds_box);
        SpectralResponse extinction_at_point = extinction_value * density_value;
        float sigma_t_channel = extinction_at_point.component(channel);

        if ((sigma_t_channel > 0.0f) && (smp.next() < sigma_t_channel / max_sigma)) {
          SpectralResponse scattering_at_point = scattering_value * density_value;
          pdf *= transmittance * extinction_at_point;
          if (pdf.is_zero()) {
            return {spectral_response_make(spect, 0.0f)};
          }

          MediumSample result = {};
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
            return {spectral_response_make(spect, 0.0f)};
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
