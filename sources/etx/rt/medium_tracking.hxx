#pragma once

#include <etx/render/interop/interop.hxx>
#include <etx/render/shared/base.hxx>
#include <etx/render/shared/scene.hxx>

#include <cmath>

namespace etx {

enum class MediumTrackingEventType : uint8_t {
  Escape,
  Scatter,
  Absorb,
  Null,
};

enum class MediumTrackingMode : uint8_t {
  Analog,
  WeightedScattering,
};

enum class MediumTrackingFailure : uint8_t {
  None,
  InvalidInput,
  InvalidCoefficients,
  InvalidDensity,
  MajorantViolation,
  InvalidProbability,
  EventLimitExceeded,
};

struct MediumTrackingInput {
  SpectralQuery spect = {};
  SpectralResponse scattering = {};
  SpectralResponse absorption = {};
  float density_majorant = 1.0f;
};

struct MediumTrackingEvent {
  MediumTrackingEventType type = MediumTrackingEventType::Escape;
  MediumTrackingFailure failure = MediumTrackingFailure::None;
  SpectralResponse weight = {};
  SpectralResponse coefficient = {};
  float3 position = {};
  float distance = 0.0f;
  float density = 0.0f;
  float majorant = 0.0f;
  float majorant_transmittance = 1.0f;
  float event_probability = 1.0f;
  float pdf_forward = 1.0f;
  float pdf_reverse = 1.0f;
  uint32_t selected_component = 0u;

  bool valid() const {
    return failure == MediumTrackingFailure::None;
  }
};

ETX_SHARED_INLINE bool medium_tracking_response_non_negative(const SpectralResponse& response) {
  return response.valid() && (response.minimum() >= 0.0f);
}

ETX_SHARED_INLINE float medium_tracking_majorant(const MediumTrackingInput& input) {
  return (input.scattering + input.absorption).maximum() * input.density_majorant;
}

ETX_SHARED_INLINE float medium_tracking_density_majorant(const Medium&) {
  return 1.0f;
}

ETX_SHARED_INLINE MediumTrackingInput make_medium_tracking_input(const Medium& medium, const SpectralQuery spect) {
  MediumTrackingInput result = {};
  result.spect = spect;
  result.scattering = medium_scattering(medium, spect);
  result.absorption = medium_absorption(medium, spect);
  result.density_majorant = medium_tracking_density_majorant(medium);
  return result;
}

ETX_SHARED_INLINE MediumTrackingEvent medium_tracking_failure_event(const MediumTrackingInput& input, const MediumTrackingFailure failure) {
  MediumTrackingEvent result = {};
  result.failure = failure;
  result.weight = SpectralResponse{input.spect, 0.0f};
  result.coefficient = SpectralResponse{input.spect, 0.0f};
  result.pdf_forward = 0.0f;
  result.pdf_reverse = 0.0f;
  return result;
}

template <typename DensityEvaluator>
ETX_SHARED_INLINE MediumTrackingEvent sample_medium_tracking_event_impl(const MediumTrackingInput& input, Sampler& sampler, const float3& origin, const float3& direction,
  const float max_distance, const MediumTrackingMode mode, DensityEvaluator&& evaluate_density) {
  const float direction_length_squared = dot(direction, direction);
  if ((std::isfinite(max_distance) == false) || (max_distance <= 0.0f) || (input.density_majorant < 0.0f) || (std::isfinite(input.density_majorant) == false) ||
      (std::isfinite(direction_length_squared) == false) || (fabsf(direction_length_squared - 1.0f) > 1.0e-4f)) {
    return medium_tracking_failure_event(input, MediumTrackingFailure::InvalidInput);
  }

  if ((medium_tracking_response_non_negative(input.scattering) == false) || (medium_tracking_response_non_negative(input.absorption) == false)) {
    return medium_tracking_failure_event(input, MediumTrackingFailure::InvalidCoefficients);
  }

  MediumTrackingEvent result = {};
  result.weight = SpectralResponse{input.spect, 1.0f};
  result.coefficient = SpectralResponse{input.spect, 0.0f};
  result.majorant = medium_tracking_majorant(input);

  if ((std::isfinite(result.majorant) == false) || (result.majorant < 0.0f)) {
    return medium_tracking_failure_event(input, MediumTrackingFailure::InvalidCoefficients);
  }

  if (result.majorant <= 0.0f) {
    result.position = origin + direction * max_distance;
    result.distance = max_distance;
    return result;
  }

  constexpr float kRandomHalfStep = 1.0f / 16777216.0f;
  const float distance_random = sampler.next() + kRandomHalfStep;
  const float sampled_distance = -logf(1.0f - distance_random) / result.majorant;
  result.distance = min(sampled_distance, max_distance);
  result.position = origin + direction * result.distance;
  result.majorant_transmittance = expf(-result.majorant * result.distance);

  if (sampled_distance >= max_distance) {
    result.pdf_forward = result.majorant_transmittance;
    result.pdf_reverse = result.majorant_transmittance;
    return result;
  }

  result.density = evaluate_density(result.position);
  if (std::isfinite(result.density) == false) {
    return medium_tracking_failure_event(input, MediumTrackingFailure::InvalidDensity);
  }

  if ((result.density < 0.0f) || (result.density > input.density_majorant)) {
    return medium_tracking_failure_event(input, MediumTrackingFailure::MajorantViolation);
  }

  const SpectralResponse scattering = input.scattering * result.density;
  const SpectralResponse absorption = input.absorption * result.density;
  const SpectralResponse extinction = scattering + absorption;
  const SpectralResponse null_coefficient = SpectralResponse{input.spect, result.majorant} - extinction;
  if (medium_tracking_response_non_negative(null_coefficient) == false) {
    return medium_tracking_failure_event(input, MediumTrackingFailure::MajorantViolation);
  }

  const uint32_t component_count = static_cast<uint32_t>(input.scattering.component_count());
  result.selected_component = min(component_count - 1u, static_cast<uint32_t>(sampler.next() * static_cast<float>(component_count)));

  SpectralResponse sampling_coefficient{input.spect, 0.0f};
  const float event_value = sampler.next() * result.majorant;
  if (mode == MediumTrackingMode::WeightedScattering) {
    if (event_value < extinction.component(result.selected_component)) {
      sampling_coefficient = extinction;
      result.coefficient = scattering;
      result.type = scattering.average() > 0.0f ? MediumTrackingEventType::Scatter : MediumTrackingEventType::Absorb;
    } else {
      sampling_coefficient = null_coefficient;
      result.coefficient = null_coefficient;
      result.type = MediumTrackingEventType::Null;
    }
  } else {
    const float scattering_component = scattering.component(result.selected_component);
    const float absorption_component = absorption.component(result.selected_component);
    if (event_value < scattering_component) {
      result.type = MediumTrackingEventType::Scatter;
      result.coefficient = scattering;
    } else if (event_value < (scattering_component + absorption_component)) {
      result.type = MediumTrackingEventType::Absorb;
      result.coefficient = absorption;
    } else {
      result.type = MediumTrackingEventType::Null;
      result.coefficient = null_coefficient;
    }
    sampling_coefficient = result.coefficient;
  }

  const float mean_sampling_coefficient = sampling_coefficient.average();
  if ((std::isfinite(mean_sampling_coefficient) == false) || (mean_sampling_coefficient <= 0.0f)) {
    return medium_tracking_failure_event(input, MediumTrackingFailure::InvalidProbability);
  }

  result.event_probability = mean_sampling_coefficient / result.majorant;
  result.pdf_forward = result.majorant_transmittance * mean_sampling_coefficient;
  result.pdf_reverse = result.pdf_forward;
  result.weight = result.coefficient / mean_sampling_coefficient;
  return result;
}

template <typename DensityEvaluator>
ETX_SHARED_INLINE MediumTrackingEvent sample_medium_tracking_event(const MediumTrackingInput& input, Sampler& sampler, const float3& origin, const float3& direction,
  const float max_distance, DensityEvaluator&& evaluate_density) {
  return sample_medium_tracking_event_impl(input, sampler, origin, direction, max_distance, MediumTrackingMode::Analog, evaluate_density);
}

template <typename DensityEvaluator>
ETX_SHARED_INLINE MediumTrackingEvent sample_medium_tracking_weighted_scattering_event(const MediumTrackingInput& input, Sampler& sampler, const float3& origin,
  const float3& direction, const float max_distance, DensityEvaluator&& evaluate_density) {
  return sample_medium_tracking_event_impl(input, sampler, origin, direction, max_distance, MediumTrackingMode::WeightedScattering, evaluate_density);
}

ETX_SHARED_INLINE MediumTrackingEvent sample_medium_tracking_event(const Medium& medium, const SpectralQuery spect, Sampler& sampler, const float3& origin, const float3& direction,
  const float max_distance) {
  const MediumTrackingInput input = make_medium_tracking_input(medium, spect);
  auto density_evaluator = [&medium](const float3& position) {
    return (medium.cls == Medium::Homogeneous) ? 1.0f : medium.sample_density_world(position);
  };
  return sample_medium_tracking_event(input, sampler, origin, direction, max_distance, density_evaluator);
}

}  // namespace etx
