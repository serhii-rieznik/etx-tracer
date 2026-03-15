#pragma once

#include <etx/render/interop/medium_phase_shared.hxx>
#include <etx/render/shared/medium.hxx>

namespace etx {

struct MediumSharedContext {
  const Medium* medium = nullptr;
  Sampler* sampler = nullptr;
  uint32_t medium_class = Medium::Homogeneous;
  uint32_t has_grid_data = 0u;
  float3 bounds_min = {};
  float3 bounds_max = {};
};

ETX_SHARED_INLINE MediumSharedContext make_medium_shared_context(const Medium& medium, Sampler& sampler) {
  MediumSharedContext result = {&medium, &sampler, static_cast<uint32_t>(medium.cls), medium.has_grid_data() ? 1u : 0u, medium.bounds.p_min, medium.bounds.p_max};
  return result;
}

ETX_SHARED_INLINE float medium_shared_rnd(ETX_INOUT(MediumSharedContext, context)) {
  return context.sampler->next();
}

ETX_SHARED_INLINE float medium_shared_density(ETX_INOUT(MediumSharedContext, context), ETX_IN(float3, local_pos)) {
  return context.medium->sample_density(local_pos, context.medium->bounds);
}

ETX_SHARED_INLINE SpectralResponse medium_transmittance_shared_to_spectral_response(ETX_IN(::SpectralResponse, response)) {
  SpectralQuery query = {response.wavelength, response.flags};
  if (::spectral_response_is_spectral(response)) {
    return {query, response.value};
  }

  return {query, response.integrated};
}

#include <etx/render/interop/medium_transmittance_shared.hxx>
#include <etx/render/interop/medium_sample_shared.hxx>

ETX_SHARED_INLINE uint32_t sample_spectrum_component(const SpectralQuery spect, const SpectralResponse& albedo, const SpectralResponse& throughput, const float rnd,
  SpectralResponse& pdf) {
  ::SpectralResponse shared_pdf = {};
  uint32_t result = medium_sample_shared_sample_spectrum_component(
    static_cast<const ::SpectralQuery&>(spect), static_cast<const ::SpectralResponse&>(albedo), static_cast<const ::SpectralResponse&>(throughput), rnd, shared_pdf);
  pdf = medium_transmittance_shared_to_spectral_response(shared_pdf);
  return result;
}

ETX_SHARED_INLINE SpectralResponse calculate_albedo(const SpectralQuery spect, const SpectralResponse& scattering, const SpectralResponse& extinction) {
  const ::SpectralResponse albedo = medium_sample_shared_calculate_albedo(
    static_cast<const ::SpectralQuery&>(spect), static_cast<const ::SpectralResponse&>(scattering), static_cast<const ::SpectralResponse&>(extinction));
  return medium_transmittance_shared_to_spectral_response(albedo);
}

ETX_SHARED_INLINE float phase_function(const float3& w_i, const float3& w_o, const float g) {
  return medium_phase_shared_henyey_greenstein(w_i, w_o, g);
}

ETX_SHARED_INLINE float3 sample_phase_function(const float3& w_i, const float g, const float2& smp_rnd) {
  return medium_phase_shared_sample_henyey_greenstein(w_i, g, smp_rnd);
}

ETX_SHARED_INLINE bool medium_supports_tracking(const Medium& medium) {
  return (medium.cls == Medium::Homogeneous) || (medium.cls == Medium::Heterogeneous);
}

ETX_SHARED_INLINE SpectralResponse medium_absorption(const Medium& medium, const SpectralQuery spect) {
  return medium_load_spectrum_or_zero(medium.absorption_index, spect);
}

ETX_SHARED_INLINE SpectralResponse medium_scattering(const Medium& medium, const SpectralQuery spect) {
  return medium_load_spectrum_or_zero(medium.scattering_index, spect);
}

ETX_SHARED_INLINE SpectralResponse medium_extinction(const Medium& medium, const SpectralQuery spect) {
  return medium_absorption(medium, spect) + medium_scattering(medium, spect);
}

ETX_SHARED_INLINE MediumInstance make_medium_instance(const Medium& medium, const SpectralQuery spect, uint32_t index) {
  MediumInstance result = {};
  result.extinction = medium_extinction(medium, spect);
  result.anisotropy = medium.phase_function_g;
  result.index = index;
  return result;
}

ETX_SHARED_INLINE SpectralResponse medium_transmittance(const Medium& medium, const SpectralQuery spect, Sampler& smp, const float3& pos, const float3& direction,
  float distance) {
  SpectralResponse one = {spect, 1.0f};
  if (distance <= 0.0f) {
    return one;
  }

  if (medium_supports_tracking(medium) == false) {
    return one;
  }

  SpectralResponse extinction = medium_extinction(medium, spect);
  if (medium.cls == Medium::Homogeneous) {
    const ::SpectralResponse transmittance = medium_shared_transmittance_homogeneous_spectral(static_cast<const ::SpectralResponse&>(extinction), distance);
    return medium_transmittance_shared_to_spectral_response(transmittance);
  }

  if (medium.has_grid_data() == false) {
    return one;
  }

  MediumSharedContext context = make_medium_shared_context(medium, smp);
  const ::SpectralResponse transmittance = medium_shared_transmittance_heterogeneous_spectral(static_cast<const ::SpectralResponse&>(extinction), pos, direction, distance,
    medium.bounds.p_min, medium.bounds.p_max, context, static_cast<const ::SpectralQuery&>(spect));
  return medium_transmittance_shared_to_spectral_response(transmittance);
}

ETX_SHARED_INLINE MediumSample sample_medium(const Medium& medium, const SpectralQuery spect, const SpectralResponse& throughput, Sampler& smp, const float3& pos,
  const float3& w_i, float max_t) {
  ETX_CRITICAL(max_t > 0.0f);

  const SpectralResponse scattering_value = medium_scattering(medium, spect);
  ETX_VALIDATE(scattering_value);
  const SpectralResponse absorption_value = medium_absorption(medium, spect);
  ETX_VALIDATE(absorption_value);

  MediumSharedContext context = make_medium_shared_context(medium, smp);
  return medium_sample_shared_sample(
    context, static_cast<const ::SpectralQuery&>(spect), static_cast<const ::SpectralResponse&>(throughput), static_cast<const ::SpectralResponse&>(scattering_value),
    static_cast<const ::SpectralResponse&>(absorption_value), pos, w_i, max_t);
}

ETX_SHARED_INLINE float medium_phase_function(const Medium& medium, const float3& w_i, const float3& w_o) {
  return phase_function(w_i, w_o, medium.phase_function_g);
}

ETX_SHARED_INLINE float3 medium_sample_phase_function(const Medium& medium, const float2& smp_rnd, const float3& w_i) {
  return sample_phase_function(w_i, medium.phase_function_g, smp_rnd);
}

}  // namespace etx
