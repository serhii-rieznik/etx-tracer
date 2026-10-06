#pragma once

#include "medium.hxx"
#include "subsurface_free_path_shared.hxx"

ETX_SHARED_INLINE bool subsurface_density_domain_matches(uint32_t first_medium_key, uint32_t first_owner_instance, uint32_t second_medium_key, uint32_t second_owner_instance) {
  return (first_medium_key != kInvalidIndex) && (first_medium_key == second_medium_key) && (first_owner_instance == second_owner_instance);
}

// Factoring the collision coefficient out of the flight gives a symmetric edge
// kernel: Xu between boundaries, Xc at one collision, and pc / sigma at two.
ETX_SHARED_INLINE float subsurface_transport_kernel_component(float extinction, float packing, bool source_collision, bool target_collision, float distance) {
  if (source_collision && target_collision) {
    return extinction > 0.0f ? subsurface_free_path_evaluate(extinction, packing, true, distance).density / extinction : 0.0f;
  }
  return subsurface_free_path_evaluate(extinction, packing, source_collision || target_collision, distance).survival;
}

ETX_SHARED_INLINE SpectralResponse subsurface_transport_kernel(ETX_IN(SpectralResponse, extinction), float packing, bool source_collision, bool target_collision, float distance) {
  const SpectralQuery spect = spectral_response_as_query(extinction);
  if (spectral_query_is_spectral(spect)) {
    return spectral_response_make(spect, subsurface_transport_kernel_component(extinction.value, packing, source_collision, target_collision, distance));
  }
  return spectral_response_make(spect, float3(subsurface_transport_kernel_component(extinction.integrated.x, packing, source_collision, target_collision, distance),
                                         subsurface_transport_kernel_component(extinction.integrated.y, packing, source_collision, target_collision, distance),
                                         subsurface_transport_kernel_component(extinction.integrated.z, packing, source_collision, target_collision, distance)));
}

struct SubsurfaceTransportSample {
  MediumSample sample;
  float pdf_forward;
  float pdf_reverse;
  float transport_pdf_forward;
  float transport_pdf_reverse;
  float event_density;
  bool scattered;
};

struct SubsurfaceRGBProposal {
  float extinction;
  float vacuum_boundary_weight;
};

ETX_SHARED_INLINE SubsurfaceRGBProposal subsurface_transport_rgb_proposal(ETX_IN(SpectralResponse, extinction)) {
  float count = 0.0f;
  float inverse_extinction_sum = 0.0f;
  for (uint32_t channel = 0u; channel < 3u; ++channel) {
    const float coefficient = channel == 0u ? extinction.integrated.x : (channel == 1u ? extinction.integrated.y : extinction.integrated.z);
    if (coefficient > 0.0f) {
      count += 1.0f;
      inverse_extinction_sum += 1.0f / coefficient;
    }
  }
  SubsurfaceRGBProposal result;
  result.extinction = count > 0.0f ? (count / inverse_extinction_sum) * (count / 3.0f) : 0.0f;
  result.vacuum_boundary_weight = count > 0.0f ? inverse_extinction_sum / count : 1.0f;
  return result;
}

ETX_SHARED_INLINE float subsurface_transport_sampling_extinction(ETX_IN(SpectralResponse, extinction)) {
  return spectral_response_is_spectral(extinction) ? extinction.value : subsurface_transport_rgb_proposal(extinction).extinction;
}

ETX_SHARED_INLINE float subsurface_transport_collision_albedo_sum(ETX_IN(SpectralResponse, extinction), ETX_IN(SpectralResponse, scattering)) {
  float sum = 0.0f;
  for (uint32_t channel = 0u; channel < 3u; ++channel) {
    const float coefficient = channel == 0u ? extinction.integrated.x : (channel == 1u ? extinction.integrated.y : extinction.integrated.z);
    const float sigma_s = channel == 0u ? scattering.integrated.x : (channel == 1u ? scattering.integrated.y : scattering.integrated.z);
    if (coefficient > 0.0f) {
      sum += sigma_s / coefficient;
    }
  }
  return sum;
}

ETX_SHARED_INLINE float subsurface_transport_channel_weight(float coefficient, float sigma_s, float packing, bool source_collision, float albedo_sum,
  float vacuum_boundary_weight) {
  if (source_collision == false) {
    return coefficient > 0.0f ? (1.0f / coefficient) : vacuum_boundary_weight;
  }
  if (coefficient <= 0.0f) {
    return 0.0f;
  }
  return ((packing > 0.0f) && (albedo_sum > 0.0f)) ? (sigma_s / coefficient) : 1.0f;
}

ETX_SHARED_INLINE SubsurfaceFreePath subsurface_transport_sampling_free_path(ETX_IN(SpectralResponse, extinction), ETX_IN(SpectralResponse, scattering), float packing,
  bool source_collision, float distance) {
  if (spectral_response_is_spectral(extinction)) {
    return subsurface_free_path_evaluate(extinction.value, packing, source_collision, distance);
  }
  const SubsurfaceRGBProposal proposal = subsurface_transport_rgb_proposal(extinction);
  const float albedo_sum = subsurface_transport_collision_albedo_sum(extinction, scattering);
  SubsurfaceFreePath result;
  result.survival = 0.0f;
  result.density = 0.0f;
  float normalization = 0.0f;
  for (uint32_t channel = 0u; channel < 3u; ++channel) {
    const float coefficient = channel == 0u ? extinction.integrated.x : (channel == 1u ? extinction.integrated.y : extinction.integrated.z);
    if ((coefficient > 0.0f) || (source_collision == false)) {
      const float sigma_s = channel == 0u ? scattering.integrated.x : (channel == 1u ? scattering.integrated.y : scattering.integrated.z);
      const float weight = subsurface_transport_channel_weight(coefficient, sigma_s, packing, source_collision, albedo_sum, proposal.vacuum_boundary_weight);
      const SubsurfaceFreePath flight = subsurface_free_path_evaluate(coefficient, packing, source_collision, distance);
      result.survival += weight * flight.survival;
      result.density += weight * flight.density;
      normalization += weight;
    }
  }
  if (normalization > 0.0f) {
    result.survival /= normalization;
    result.density /= normalization;
  } else {
    result.survival = 1.0f;
  }
  return result;
}

// Keep proposal evaluation out of the GPU trace stages' inlined control flow.
ETX_SHARED_NOINLINE float subsurface_transport_sampling_kernel(ETX_IN(SpectralResponse, extinction), ETX_IN(SpectralResponse, scattering), float packing, bool source_collision,
  bool target_collision, float distance) {
  const SubsurfaceFreePath flight = subsurface_transport_sampling_free_path(extinction, scattering, packing, source_collision, distance);
  if (target_collision == false) {
    return flight.survival;
  }
  const float sampling_extinction = subsurface_transport_sampling_extinction(extinction);
  if (sampling_extinction > 0.0f) {
    return flight.density / sampling_extinction;
  }
  return (spectral_response_is_spectral(extinction) && source_collision) ? 0.0f : 1.0f;
}

struct SubsurfaceBeamTransport {
  SpectralResponse weight;
  float transport_pdf_forward;
  float transport_pdf_reverse;
  float survival_forward;
  float survival_reverse;
  float event_density;
  bool supported;
};

ETX_SHARED_INLINE SubsurfaceBeamTransport subsurface_transport_beam(ETX_IN(SpectralResponse, extinction), ETX_IN(SpectralResponse, scattering), float packing,
  bool source_collision, float distance) {
  SubsurfaceBeamTransport result;
  result.weight = spectral_response_make(spectral_response_as_query(extinction), 0.0f);
  result.survival_forward = subsurface_transport_sampling_free_path(extinction, scattering, packing, source_collision, distance).survival;
  result.survival_reverse = subsurface_transport_sampling_free_path(extinction, scattering, packing, true, distance).survival;
  result.transport_pdf_forward = subsurface_transport_sampling_kernel(extinction, scattering, packing, source_collision, true, distance);
  result.transport_pdf_reverse = subsurface_transport_sampling_kernel(extinction, scattering, packing, true, source_collision, distance);
  result.event_density = subsurface_transport_sampling_extinction(extinction);
  result.supported = (result.survival_forward > 0.0f) && (result.transport_pdf_forward > 0.0f) && (result.transport_pdf_reverse > 0.0f) && (result.event_density > 0.0f);
  if (result.supported) {
    result.weight = spectral_response_div(subsurface_transport_kernel(extinction, packing, source_collision, true, distance), result.survival_forward);
    result.supported = spectral_response_is_zero(result.weight) == false;
  }
  return result;
}

ETX_SHARED_INLINE SubsurfaceTransportSample subsurface_transport_sample(ETX_IN(SpectralResponse, extinction), ETX_IN(SpectralResponse, scattering), float packing,
  bool source_collision, ETX_IN(float3, origin), ETX_IN(float3, direction), float maximum_distance, float random) {
  SubsurfaceTransportSample result;
  const SpectralQuery spect = spectral_response_as_query(extinction);
  const float sampling_extinction = subsurface_transport_sampling_extinction(extinction);
  float component_extinction = sampling_extinction;
  if ((spectral_query_is_spectral(spect) == false) && (sampling_extinction > 0.0f)) {
    // Albedo-weighted exclusion collision channels bound each collision weight by the sum of albedos.
    // Boundary channels retain absorbing and transparent escape support; reverse PDFs are evaluated separately.
    const SubsurfaceRGBProposal proposal = subsurface_transport_rgb_proposal(extinction);
    const float albedo_sum = subsurface_transport_collision_albedo_sum(extinction, scattering);
    // Transparent channels have an escape atom; use the finite channels' mean boundary weight.
    float normalization = 0.0f;
    for (uint32_t channel = 0u; channel < 3u; ++channel) {
      const float coefficient = channel == 0u ? extinction.integrated.x : (channel == 1u ? extinction.integrated.y : extinction.integrated.z);
      const float sigma_s = channel == 0u ? scattering.integrated.x : (channel == 1u ? scattering.integrated.y : scattering.integrated.z);
      normalization += subsurface_transport_channel_weight(coefficient, sigma_s, packing, source_collision, albedo_sum, proposal.vacuum_boundary_weight);
    }
    float selector = random * normalization;
    for (uint32_t channel = 0u; channel < 3u; ++channel) {
      const float coefficient = channel == 0u ? extinction.integrated.x : (channel == 1u ? extinction.integrated.y : extinction.integrated.z);
      const float sigma_s = channel == 0u ? scattering.integrated.x : (channel == 1u ? scattering.integrated.y : scattering.integrated.z);
      const float weight = subsurface_transport_channel_weight(coefficient, sigma_s, packing, source_collision, albedo_sum, proposal.vacuum_boundary_weight);
      if ((weight > 0.0f) && (selector < weight)) {
        component_extinction = coefficient;
        random = selector / weight;
        break;
      }
      selector -= weight;
    }
  }
  const float sampled_distance = subsurface_free_path_sample(component_extinction, packing, source_collision, random);
  result.scattered = sampled_distance < maximum_distance;
  const float distance = result.scattered ? sampled_distance : maximum_distance;
  result.transport_pdf_forward = subsurface_transport_sampling_kernel(extinction, scattering, packing, source_collision, result.scattered, distance);
  result.transport_pdf_reverse = ((packing > 0.0f) && (spectral_query_is_spectral(spect) == false) && (source_collision != result.scattered))
                                   ? subsurface_transport_sampling_kernel(extinction, scattering, packing, result.scattered, source_collision, distance)
                                   : result.transport_pdf_forward;
  result.event_density = result.scattered ? sampling_extinction : 1.0f;
  result.pdf_forward = result.transport_pdf_forward * result.event_density;
  result.pdf_reverse = result.transport_pdf_reverse * (source_collision ? sampling_extinction : 1.0f);
  SpectralResponse weight = subsurface_transport_kernel(extinction, packing, source_collision, result.scattered, distance);
  if (result.scattered) {
    weight = spectral_response_mul(weight, scattering);
  }
  result.sample.weight = spectral_response_make(spect, 0.0f);
  if (result.pdf_forward > 0.0f) {
    result.sample.weight = spectral_response_div(weight, result.pdf_forward);
  }
  result.sample.pos = origin + direction * distance;
  result.sample.sampled_medium_t = result.scattered ? distance : 0.0f;
  return result;
}
