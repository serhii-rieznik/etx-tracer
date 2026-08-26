#include <etx/rt/medium_tracking.hxx>

#include <cmath>
#include <cstdio>
#include <array>

namespace {

constexpr uint32_t kEventSamples = 1u << 19u;
constexpr uint32_t kTransmittanceSamples = 1u << 18u;

bool close_value(const float actual, const float expected, const float tolerance, const char* label) {
  if (fabsf(actual - expected) <= tolerance) {
    return true;
  }

  std::printf("%s failed: actual %.9f expected %.9f tolerance %.9f\n", label, actual, expected, tolerance);
  return false;
}

bool close_response(const etx::SpectralResponse& actual, const float3& expected, const float tolerance, const char* label) {
  bool valid = true;
  valid = close_value(actual.integrated.x, expected.x, tolerance, label) && valid;
  valid = close_value(actual.integrated.y, expected.y, tolerance, label) && valid;
  valid = close_value(actual.integrated.z, expected.z, tolerance, label) && valid;
  return valid;
}

etx::MediumTrackingInput integrated_input(const float3& scattering, const float3& absorption) {
  const etx::SpectralQuery spect = {550.0f, 0u};
  etx::MediumTrackingInput result = {};
  result.spect = spect;
  result.scattering = etx::SpectralResponse{spect, scattering};
  result.absorption = etx::SpectralResponse{spect, absorption};
  result.density_majorant = 1.0f;
  return result;
}

bool validate_zero_extinction() {
  const etx::MediumTrackingInput input = integrated_input(float3{}, float3{});
  etx::Sampler sampler{11u};
  const auto event = etx::sample_medium_tracking_event(input, sampler, float3{}, float3{1.0f, 0.0f, 0.0f}, 3.0f, [](const float3&) {
    return 1.0f;
  });

  bool valid = event.valid();
  valid = (event.type == etx::MediumTrackingEventType::Escape) && valid;
  valid = close_value(event.distance, 3.0f, 0.0f, "zero extinction distance") && valid;
  valid = close_response(event.weight, float3{1.0f, 1.0f, 1.0f}, 0.0f, "zero extinction weight") && valid;
  std::printf("zero extinction %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_density_majorants() {
  std::array<float, 8u> density_values = {-2.0f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 2.0f, 5.0f};
  etx::DensityGrid density_grid = {};
  density_grid.density = etx::ArrayView<float>{density_values.data(), density_values.size()};

  etx::MediumGrid texture_grid = {};
  texture_grid.type = MediumGridType::Texture3D;
  texture_grid.dimensions = uint3{2u, 2u, 2u};

  BoundingBox bounds = {};
  bounds.p_min = float3{-1.0f, -1.0f, -1.0f};
  bounds.p_max = float3{1.0f, 1.0f, 1.0f};
  etx::Sampler sampler{0x12d7u};
  bool valid = true;
  for (uint32_t sample_index = 0u; sample_index < 65536u; ++sample_index) {
    const float3 local_coord = {sampler.next(), sampler.next(), sampler.next()};
    const float density = density_grid.sample(local_coord, bounds, texture_grid);
    valid = std::isfinite(density) && (density >= 0.0f) && (density <= 1.0f) && valid;
  }

  etx::MediumGrid noise_grid = {};
  noise_grid.type = MediumGridType::NoiseFunction;
  noise_grid.noise_octaves = 8u;
  noise_grid.noise_scale = 3.0f;
  noise_grid.noise_lacunarity = 2.1f;
  noise_grid.noise_persistence = 0.75f;
  noise_grid.noise_power = 0.7f;
  noise_grid.noise_sharpness = 3.0f;
  noise_grid.noise_enable_border_fade = 1u;
  noise_grid.noise_border_fade_distance = 0.2f;

  for (uint32_t noise_type = 0u; noise_type < static_cast<uint32_t>(etx::NoiseFunction::Count); ++noise_type) {
    noise_grid.noise_type = noise_type;
    for (uint32_t sample_index = 0u; sample_index < 4096u; ++sample_index) {
      const float3 local_coord = {sampler.next(), sampler.next(), sampler.next()};
      const float density = density_grid.sample(local_coord, bounds, noise_grid);
      valid = std::isfinite(density) && (density >= 0.0f) && (density <= 1.0f) && valid;
    }
  }

  etx::Medium medium = {};
  valid = close_value(etx::medium_tracking_density_majorant(medium), 1.0f, 0.0f, "medium density majorant") && valid;
  std::printf("density majorants %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_event_distribution() {
  const float3 scattering = {0.15f, 0.30f, 0.45f};
  const float3 absorption = {0.05f, 0.10f, 0.15f};
  const etx::MediumTrackingInput input = integrated_input(scattering, absorption);
  const float majorant = etx::medium_tracking_majorant(input);

  uint64_t scatter_count = 0u;
  uint64_t absorb_count = 0u;
  uint64_t null_count = 0u;
  etx::SpectralResponse scatter_weight_sum{input.spect, 0.0f};
  etx::SpectralResponse null_weight_sum{input.spect, 0.0f};
  etx::Sampler sampler{0x7a31u};

  bool valid = true;
  for (uint32_t sample_index = 0u; sample_index < kEventSamples; ++sample_index) {
    const auto event = etx::sample_medium_tracking_event(input, sampler, float3{}, float3{1.0f, 0.0f, 0.0f}, 100.0f, [](const float3&) {
      return 1.0f;
    });
    if (event.valid() == false) {
      std::printf("event distribution produced failure %u\n", static_cast<uint32_t>(event.failure));
      return false;
    }

    if (event.type == etx::MediumTrackingEventType::Scatter) {
      ++scatter_count;
      scatter_weight_sum += event.weight;
    } else if (event.type == etx::MediumTrackingEventType::Absorb) {
      ++absorb_count;
    } else if (event.type == etx::MediumTrackingEventType::Null) {
      ++null_count;
      null_weight_sum += event.weight;
    } else {
      std::printf("event distribution unexpectedly escaped\n");
      return false;
    }

    valid = (event.pdf_forward > 0.0f) && (event.pdf_forward == event.pdf_reverse) && valid;
  }

  const float inverse_samples = 1.0f / static_cast<float>(kEventSamples);
  const float scatter_probability = static_cast<float>(scatter_count) * inverse_samples;
  const float absorb_probability = static_cast<float>(absorb_count) * inverse_samples;
  const float null_probability = static_cast<float>(null_count) * inverse_samples;
  const float expected_scatter_probability = (scattering.x + scattering.y + scattering.z) / (3.0f * majorant);
  const float expected_absorb_probability = (absorption.x + absorption.y + absorption.z) / (3.0f * majorant);
  const float expected_null_probability = 1.0f - expected_scatter_probability - expected_absorb_probability;

  valid = close_value(scatter_probability, expected_scatter_probability, 0.0025f, "scatter probability") && valid;
  valid = close_value(absorb_probability, expected_absorb_probability, 0.0025f, "absorb probability") && valid;
  valid = close_value(null_probability, expected_null_probability, 0.0025f, "null probability") && valid;

  scatter_weight_sum *= inverse_samples;
  null_weight_sum *= inverse_samples;
  const float3 expected_scatter_weight = scattering / majorant;
  const float3 extinction = scattering + absorption;
  const float3 expected_null_weight = (float3{majorant, majorant, majorant} - extinction) / majorant;
  valid = close_response(scatter_weight_sum, expected_scatter_weight, 0.0035f, "scatter weighted expectation") && valid;
  valid = close_response(null_weight_sum, expected_null_weight, 0.0035f, "null weighted expectation") && valid;

  std::printf("event distribution %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_weighted_scattering_event_distribution() {
  const float3 scattering = {0.15f, 0.30f, 0.45f};
  const float3 absorption = {0.05f, 0.10f, 0.15f};
  const float3 extinction = scattering + absorption;
  const etx::MediumTrackingInput input = integrated_input(scattering, absorption);
  const float majorant = etx::medium_tracking_majorant(input);
  const float mean_extinction = (extinction.x + extinction.y + extinction.z) / 3.0f;

  uint64_t scatter_count = 0u;
  uint64_t absorb_count = 0u;
  uint64_t null_count = 0u;
  etx::SpectralResponse scatter_weight_sum{input.spect, 0.0f};
  etx::SpectralResponse null_weight_sum{input.spect, 0.0f};
  etx::Sampler sampler{0x83d7u};

  bool valid = true;
  for (uint32_t sample_index = 0u; sample_index < kEventSamples; ++sample_index) {
    const auto event = etx::sample_medium_tracking_weighted_scattering_event(input, sampler, float3{}, float3{1.0f, 0.0f, 0.0f}, 100.0f, [](const float3&) {
      return 1.0f;
    });
    if (event.valid() == false) {
      std::printf("weighted event distribution produced failure %u\n", static_cast<uint32_t>(event.failure));
      return false;
    }

    if (event.type == etx::MediumTrackingEventType::Scatter) {
      ++scatter_count;
      scatter_weight_sum += event.weight;
      valid = close_value(event.event_probability * event.majorant, mean_extinction, 1.0e-6f, "weighted real-event density") && valid;
    } else if (event.type == etx::MediumTrackingEventType::Absorb) {
      ++absorb_count;
    } else if (event.type == etx::MediumTrackingEventType::Null) {
      ++null_count;
      null_weight_sum += event.weight;
    } else {
      std::printf("weighted event distribution unexpectedly escaped\n");
      return false;
    }
  }

  const float inverse_samples = 1.0f / static_cast<float>(kEventSamples);
  const float scatter_probability = static_cast<float>(scatter_count) * inverse_samples;
  const float null_probability = static_cast<float>(null_count) * inverse_samples;
  valid = close_value(scatter_probability, mean_extinction / majorant, 0.0025f, "weighted real-event probability") && valid;
  valid = close_value(null_probability, 1.0f - mean_extinction / majorant, 0.0025f, "weighted null probability") && valid;
  valid = (absorb_count == 0u) && valid;

  scatter_weight_sum *= inverse_samples;
  null_weight_sum *= inverse_samples;
  const float3 expected_scatter_weight = scattering / majorant;
  const float3 expected_null_weight = (float3{majorant, majorant, majorant} - extinction) / majorant;
  valid = close_response(scatter_weight_sum, expected_scatter_weight, 0.0035f, "weighted scatter expectation") && valid;
  valid = close_response(null_weight_sum, expected_null_weight, 0.0035f, "weighted null expectation") && valid;

  const etx::MediumTrackingInput absorption_only = integrated_input(float3{}, absorption);
  for (uint32_t seed = 1u; seed < 4096u; ++seed) {
    etx::Sampler absorption_sampler{seed};
    const auto event = etx::sample_medium_tracking_weighted_scattering_event(absorption_only, absorption_sampler, float3{}, float3{1.0f, 0.0f, 0.0f}, 100.0f, [](const float3&) {
      return 1.0f;
    });
    if (event.type == etx::MediumTrackingEventType::Null) {
      continue;
    }
    valid = (event.type == etx::MediumTrackingEventType::Absorb) && event.weight.is_zero() && valid;
    valid = close_value(event.event_probability * event.majorant, (absorption.x + absorption.y + absorption.z) / 3.0f, 1.0e-6f, "weighted absorption-event density") && valid;
    break;
  }

  std::printf("weighted scattering event distribution %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_transmittance() {
  const float3 scattering = {0.15f, 0.30f, 0.45f};
  const float3 absorption = {0.05f, 0.10f, 0.15f};
  const float3 extinction = scattering + absorption;
  const etx::MediumTrackingInput input = integrated_input(scattering, absorption);
  constexpr float segment_length = 2.0f;

  etx::SpectralResponse estimate_sum{input.spect, 0.0f};
  etx::Sampler sampler{0x91c5u};
  bool valid = true;

  for (uint32_t sample_index = 0u; sample_index < kTransmittanceSamples; ++sample_index) {
    etx::SpectralResponse throughput{input.spect, 1.0f};
    float3 origin = {};
    float remaining_distance = segment_length;
    bool escaped = false;

    for (uint32_t event_index = 0u; event_index < 4096u; ++event_index) {
      const auto event = etx::sample_medium_tracking_event(input, sampler, origin, float3{1.0f, 0.0f, 0.0f}, remaining_distance, [](const float3&) {
        return 1.0f;
      });
      if (event.valid() == false) {
        std::printf("transmittance produced failure %u\n", static_cast<uint32_t>(event.failure));
        return false;
      }

      if (event.type == etx::MediumTrackingEventType::Escape) {
        escaped = true;
        break;
      }

      if (event.type != etx::MediumTrackingEventType::Null) {
        break;
      }

      throughput *= event.weight;
      origin = event.position;
      remaining_distance -= event.distance;
    }

    if (escaped) {
      estimate_sum += throughput;
    }
  }

  estimate_sum *= 1.0f / static_cast<float>(kTransmittanceSamples);
  const float3 expected = {expf(-extinction.x * segment_length), expf(-extinction.y * segment_length), expf(-extinction.z * segment_length)};
  valid = close_response(estimate_sum, expected, 0.0045f, "transmittance") && valid;
  std::printf("transmittance %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_single_scattering_integral() {
  const float3 scattering = {0.15f, 0.30f, 0.45f};
  const float3 absorption = {0.05f, 0.10f, 0.15f};
  const float3 extinction = scattering + absorption;
  const etx::MediumTrackingInput input = integrated_input(scattering, absorption);
  constexpr float segment_length = 2.0f;

  etx::SpectralResponse estimate_sum{input.spect, 0.0f};
  etx::Sampler sampler{0x31aeu};
  for (uint32_t sample_index = 0u; sample_index < kTransmittanceSamples; ++sample_index) {
    etx::SpectralResponse throughput{input.spect, 1.0f};
    float3 origin = {};
    float remaining_distance = segment_length;
    for (uint32_t event_index = 0u; event_index < 4096u; ++event_index) {
      const auto event = etx::sample_medium_tracking_event(input, sampler, origin, float3{1.0f, 0.0f, 0.0f}, remaining_distance, [](const float3&) {
        return 1.0f;
      });
      if (event.valid() == false) {
        std::printf("single-scattering integral produced failure %u\n", static_cast<uint32_t>(event.failure));
        return false;
      }
      if (event.type == etx::MediumTrackingEventType::Scatter) {
        estimate_sum += throughput * event.weight;
        break;
      }
      if (event.type != etx::MediumTrackingEventType::Null) {
        break;
      }
      throughput *= event.weight;
      origin = event.position;
      remaining_distance -= event.distance;
    }
  }

  estimate_sum *= 1.0f / static_cast<float>(kTransmittanceSamples);
  const float3 expected = scattering / extinction * (float3{1.0f, 1.0f, 1.0f} - exp(-extinction * segment_length));
  const bool valid = close_response(estimate_sum, expected, 0.0045f, "single-scattering integral");
  std::printf("single-scattering integral %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_weighted_single_scattering_integral() {
  const float3 scattering = {0.15f, 0.30f, 0.45f};
  const float3 absorption = {0.05f, 0.10f, 0.15f};
  const float3 extinction = scattering + absorption;
  const etx::MediumTrackingInput input = integrated_input(scattering, absorption);
  constexpr float segment_length = 2.0f;

  etx::SpectralResponse estimate_sum{input.spect, 0.0f};
  etx::Sampler sampler{0x417du};
  for (uint32_t sample_index = 0u; sample_index < kTransmittanceSamples; ++sample_index) {
    etx::SpectralResponse throughput{input.spect, 1.0f};
    float3 origin = {};
    float remaining_distance = segment_length;
    for (uint32_t event_index = 0u; event_index < 4096u; ++event_index) {
      const auto event = etx::sample_medium_tracking_weighted_scattering_event(input, sampler, origin, float3{1.0f, 0.0f, 0.0f}, remaining_distance, [](const float3&) {
        return 1.0f;
      });
      if (event.valid() == false) {
        std::printf("weighted single-scattering integral produced failure %u\n", static_cast<uint32_t>(event.failure));
        return false;
      }
      if (event.type == etx::MediumTrackingEventType::Scatter) {
        estimate_sum += throughput * event.weight;
        break;
      }
      if (event.type != etx::MediumTrackingEventType::Null) {
        break;
      }
      throughput *= event.weight;
      origin = event.position;
      remaining_distance -= event.distance;
    }
  }

  estimate_sum *= 1.0f / static_cast<float>(kTransmittanceSamples);
  const float3 expected = scattering / extinction * (float3{1.0f, 1.0f, 1.0f} - exp(-extinction * segment_length));
  const bool valid = close_response(estimate_sum, expected, 0.0045f, "weighted single-scattering integral");
  std::printf("weighted single-scattering integral %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_spectral_tracking() {
  const etx::SpectralQuery spect = etx::SpectralQuery::spectral_sample(0.5f);
  etx::MediumTrackingInput input = {};
  input.spect = spect;
  input.scattering = etx::SpectralResponse{spect, 0.4f};
  input.absorption = etx::SpectralResponse{spect, 0.1f};
  input.density_majorant = 1.0f;

  etx::Sampler sampler{0x2e19u};
  bool valid = true;
  for (uint32_t sample_index = 0u; sample_index < 65536u; ++sample_index) {
    const auto event = etx::sample_medium_tracking_event(input, sampler, float3{}, float3{1.0f, 0.0f, 0.0f}, 100.0f, [](const float3&) {
      return 1.0f;
    });
    valid = event.valid() && valid;
    valid = (event.type != etx::MediumTrackingEventType::Null) && valid;
    valid = (event.selected_component == 0u) && valid;
  }

  std::printf("spectral tracking %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_failures() {
  const etx::MediumTrackingInput input = integrated_input(float3{0.2f, 0.2f, 0.2f}, float3{});
  etx::Sampler sampler{77u};

  const auto invalid_distance = etx::sample_medium_tracking_event(input, sampler, float3{}, float3{1.0f, 0.0f, 0.0f}, 0.0f, [](const float3&) {
    return 1.0f;
  });
  const auto invalid_direction = etx::sample_medium_tracking_event(input, sampler, float3{}, float3{2.0f, 0.0f, 0.0f}, 1.0f, [](const float3&) {
    return 1.0f;
  });
  const auto majorant_violation = etx::sample_medium_tracking_event(input, sampler, float3{}, float3{1.0f, 0.0f, 0.0f}, 100.0f, [](const float3&) {
    return 1.1f;
  });

  bool valid = (invalid_distance.failure == etx::MediumTrackingFailure::InvalidInput);
  valid = (invalid_direction.failure == etx::MediumTrackingFailure::InvalidInput) && valid;
  valid = (majorant_violation.failure == etx::MediumTrackingFailure::MajorantViolation) && valid;
  std::printf("failure handling %s\n", valid ? "valid" : "failed");
  return valid;
}

}  // namespace

int main() {
  bool valid = true;
  valid = validate_zero_extinction() && valid;
  valid = validate_density_majorants() && valid;
  valid = validate_event_distribution() && valid;
  valid = validate_weighted_scattering_event_distribution() && valid;
  valid = validate_transmittance() && valid;
  valid = validate_single_scattering_integral() && valid;
  valid = validate_weighted_single_scattering_integral() && valid;
  valid = validate_spectral_tracking() && valid;
  valid = validate_failures() && valid;
  return valid ? 0 : 1;
}
