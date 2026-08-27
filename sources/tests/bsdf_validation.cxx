#include <algorithm>

#include <etx/core/environment.hxx>
#include <etx/render/host/bsdf_energy_compensation_lut.hxx>
#include <etx/render/host/gpu_asset_descriptor.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_global.hxx>
#include <etx/render/host/scene_serialization.hxx>
#include <etx/render/interop/gpu_scene_shared.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scene_bsdf.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr uint32_t kIntegrationSamples = 32768u;
constexpr uint32_t kBsdfSamples = 2048u;

enum SpectrumSlot : uint32_t {
  SpectrumWhite,
  SpectrumBlack,
  SpectrumHalf,
  SpectrumColored,
  SpectrumAirEta,
  SpectrumDielectricEta,
  SpectrumSapphireEta,
  SpectrumNamedPlasticEta,
  SpectrumNamedPlasticK,
  SpectrumNamedWaterEta,
  SpectrumNamedWaterK,
  SpectrumConductorEta,
  SpectrumConductorK,
  SpectrumMirrorEta,
  SpectrumMirrorK,
  SpectrumSpectralWhite,
  SpectrumCount,
};

etx::SpectralDistribution make_spectrum(const float3& value) {
  etx::SpectralDistribution result = {};
  result.integrated_value = value;
  result.spectral_entry_count = 0u;
  return result;
}

etx::SpectralDistribution make_loaded_ior_constant(const float value) {
  etx::SpectralDistribution result = etx::SpectralDistribution::constant(value);
  result.integrated_value = etx::rgb_to_xyz(result.integrated_value);
  return result;
}

float3 sample_uniform_sphere(etx::Sampler& sampler) {
  const float2 sample = sampler.next_2d();
  const float z = 1.0f - 2.0f * sample.x;
  const float radius = sqrtf(max(0.0f, 1.0f - z * z));
  const float phi = kDoublePi * sample.y;
  return float3{radius * cosf(phi), radius * sinf(phi), z};
}

bool finite_response(const etx::SpectralResponse& value) {
  if (value.spectral()) {
    return std::isfinite(value.value);
  }

  return (std::isfinite(value.integrated.x)) && (std::isfinite(value.integrated.y)) && (std::isfinite(value.integrated.z));
}

bool non_negative_response(const etx::SpectralResponse& value) {
  if (value.spectral()) {
    return value.value >= -kEpsilon;
  }

  return (value.integrated.x >= -kEpsilon) && (value.integrated.y >= -kEpsilon) && (value.integrated.z >= -kEpsilon);
}

bool validate_sample(const etx::BSDFSample& sample) {
  if (std::isfinite(sample.pdf) == false) {
    return false;
  }

  if (sample.pdf < 0.0f) {
    return false;
  }

  if (finite_response(sample.weight) == false) {
    return false;
  }

  return non_negative_response(sample.weight);
}

bool close_value(const float a, const float b, const float tolerance) {
  return fabsf(a - b) <= tolerance;
}

bool validate_spectral_sample_invariants() {
  const etx::SpectralQuery query = etx::SpectralQuery::spectral_sample(0.371f);
  const etx::SpectralResponse response{::spectral_response_make(query, 1.0f)};
  const float3 expected_rgb = ::spectral_response_to_rgb(response) / ::spectral_query_sampling_pdf(query);

  const float3 actual_rgb = response.to_rgb_estimate();
  bool valid = query.spectral();
  valid = close_value(actual_rgb.x, expected_rgb.x, 1.0e-5f) && close_value(actual_rgb.y, expected_rgb.y, 1.0e-5f) && close_value(actual_rgb.z, expected_rgb.z, 1.0e-5f) && valid;
  valid = (response.component_count() == 1.0f) && close_value(response.sum(), 1.0f, 1.0e-6f) && close_value(response.maximum(), 1.0f, 1.0e-6f) && valid;
  valid = close_value(response.integrated.x, 0.0f, 1.0e-6f) && close_value(response.integrated.y, 0.0f, 1.0e-6f) && close_value(response.integrated.z, 0.0f, 1.0e-6f) && valid;
  valid = ::spectral_query_compatible(query, ::spectral_response_as_query(response)) && valid;

  ::RefractiveIndexSample equal_ext_ior = {};
  equal_ext_ior.eta = ::spectral_response_make(query, 1.0f);
  ::RefractiveIndexSample equal_int_ior = equal_ext_ior;
  valid = ::bsdf_dielectric_equal_eta(equal_ext_ior, equal_int_ior) && valid;
  equal_int_ior.eta.value = 1.5f;
  valid = (::bsdf_dielectric_equal_eta(equal_ext_ior, equal_int_ior) == false) && valid;

  std::printf("spectral sample invariants %s\n", valid ? "valid" : "failed");
  return valid;
}

bool validate_spectral_sampling_distribution() {
  constexpr uint32_t kProgressiveBlockSize = 32u;
  constexpr uint32_t kEarlySampleCount = 12u;
  constexpr uint32_t kRandomSeed = 0u;

  std::vector<double> sensor_importance(WavelengthCount - 1u);
  double total_sensor_importance = 0.0;
  for (uint32_t i = 0u; i < (WavelengthCount - 1u); ++i) {
    const float3 xyz = 0.5f * (::spectral_xyz(i) + ::spectral_xyz(i + 1u));
    const float3 rgb = ::spectral_xyz_to_rgb(xyz);
    sensor_importance[i] = std::sqrt(double(dot(rgb, rgb)));
    total_sensor_importance += sensor_importance[i];
  }

  double pdf_integral = 0.0;
  float3 estimated_unit_radiance_xyz = {};
  float3 reference_unit_radiance_xyz = {};
  bool valid = true;
  valid = close_value(kWavelengthSamplingCDF[0], 0.0f, 0.0f) && close_value(kWavelengthSamplingCDF[WavelengthCount - 1u], 1.0f, 0.0f) && valid;
  for (uint32_t i = 0u; i < (WavelengthCount - 1u); ++i) {
    const float interval_probability = kWavelengthSamplingCDF[i + 1u] - kWavelengthSamplingCDF[i];
    const double expected_probability =
      (1.0 - double(kWavelengthSamplingUniformMixture)) * sensor_importance[i] / total_sensor_importance + double(kWavelengthSamplingUniformMixture) / double(WavelengthCount - 1u);
    valid = std::isfinite(interval_probability) && (interval_probability > 0.0f) && valid;
    valid = (std::abs(double(interval_probability) - expected_probability) <= 1.0e-6) && valid;
    valid = close_value(::spectral_query_wavelength_pdf(kShortestWavelength + float(i) + 0.5f), interval_probability, 1.0e-7f) && valid;
    pdf_integral += double(interval_probability);

    const ::SpectralQuery query = ::spectral_query_spectral_sample(0.5f * (kWavelengthSamplingCDF[i] + kWavelengthSamplingCDF[i + 1u]));
    const ::SpectralResponse response = ::spectral_response_make(query, 1.0f);
    estimated_unit_radiance_xyz += interval_probability * ::spectral_response_to_xyz_estimate(response);
    reference_unit_radiance_xyz += 0.5f * kInvCIEYIntegral * (::spectral_xyz(i) + ::spectral_xyz(i + 1u));
  }
  valid = (std::abs(pdf_integral - 1.0) <= 1.0e-6) && valid;
  valid = close_value(estimated_unit_radiance_xyz.x, reference_unit_radiance_xyz.x, 1.0e-5f) &&
          close_value(estimated_unit_radiance_xyz.y, reference_unit_radiance_xyz.y, 1.0e-5f) &&
          close_value(estimated_unit_radiance_xyz.z, reference_unit_radiance_xyz.z, 1.0e-5f) && valid;
  valid = (::spectral_query_wavelength_pdf(kShortestWavelength - 1.0f) == 0.0f) && (::spectral_query_wavelength_pdf(kLongestWavelength + 1.0f) == 0.0f) && valid;

  const uint32_t scramble = ::sampler_random_seed(0u, kRandomSeed);
  for (uint32_t block_size = 4u; block_size <= kProgressiveBlockSize; block_size *= 2u) {
    bool occupied[kProgressiveBlockSize] = {};
    for (uint32_t i = 0u; i < block_size; ++i) {
      const float sample = ::sampler_scrambled_radical_inverse_base2(i, scramble);
      const uint32_t bin = min(uint32_t(sample * float(block_size)), block_size - 1u);
      valid = (occupied[bin] == false) && valid;
      occupied[bin] = true;
    }
    for (uint32_t i = 0u; i < block_size; ++i) {
      valid = occupied[i] && valid;
    }
  }

  std::vector<float> early_samples;
  early_samples.reserve(kEarlySampleCount);
  float minimum_weighted_sensor_response = std::numeric_limits<float>::max();
  float maximum_weighted_sensor_response = 0.0f;
  for (uint32_t i = 0u; i < kEarlySampleCount; ++i) {
    const float sample = ::sampler_scrambled_radical_inverse_base2(i, scramble);
    const etx::SpectralQuery query = etx::SpectralQuery::progressive_sample(i, kRandomSeed);
    const ::SpectralQuery expected_query = ::spectral_query_spectral_sample(sample);
    valid = close_value(query.wavelength, expected_query.wavelength, 1.0e-6f) && valid;

    const uint32_t wavelength_index = min(uint32_t(floorf(query.wavelength) - kShortestWavelength), WavelengthCount - 2u);
    const float wavelength_fraction = query.wavelength - floorf(query.wavelength);
    const float3 xyz = lerp(::spectral_xyz(wavelength_index), ::spectral_xyz(wavelength_index + 1u), wavelength_fraction);
    const float3 rgb = ::spectral_xyz_to_rgb(xyz);
    const float weighted_sensor_response = sqrtf(dot(rgb, rgb)) / query.sampling_pdf();
    minimum_weighted_sensor_response = min(minimum_weighted_sensor_response, weighted_sensor_response);
    maximum_weighted_sensor_response = max(maximum_weighted_sensor_response, weighted_sensor_response);

    const uint32_t interval = min(uint32_t(query.wavelength - kShortestWavelength), WavelengthCount - 2u);
    const float reconstructed_sample =
      kWavelengthSamplingCDF[interval] + (query.wavelength - (kShortestWavelength + float(interval))) * (kWavelengthSamplingCDF[interval + 1u] - kWavelengthSamplingCDF[interval]);
    valid = close_value(reconstructed_sample, sample, 2.0e-6f) && valid;
    early_samples.emplace_back(sample);
  }
  valid = (maximum_weighted_sensor_response <= (1.1f * minimum_weighted_sensor_response)) && valid;
  std::sort(early_samples.begin(), early_samples.end());
  float previous_sample = 0.0f;
  for (const float sample : early_samples) {
    valid = ((sample - previous_sample) <= (1.0f / 8.0f + 1.0e-6f)) && valid;
    previous_sample = sample;
  }
  valid = ((1.0f - previous_sample) <= (1.0f / 8.0f + 1.0e-6f)) && valid;

  std::printf("spectral sampling distribution %s (pdf integral %.8f)\n", valid ? "valid" : "failed", pdf_integral);
  return valid;
}

::RefractiveIndexSample make_spectral_ior(const etx::SpectralQuery& query, const float eta, const float k = 0.0f, const uint32_t cls = etx::SpectralDistribution::Dielectric) {
  ::RefractiveIndexSample result = {};
  result.cls = cls;
  result.eta = spectral_response_make(query, eta);
  result.k = spectral_response_make(query, k);
  return result;
}

bool validate_thinfilm_optical_invariants() {
  constexpr float wavelength = 550.0f;
  const complex air = bsdf_complex_make(1.0f, 0.0f);
  const complex film = bsdf_complex_make(1.5f, 0.0f);
  const complex glass = bsdf_complex_make(2.25f, 0.0f);
  bool valid = true;

  const float direct = bsdf_fresnel_generic(1.0f, air, glass);
  const float zero_thickness = bsdf_fresnel_thinfilm(wavelength, 1.0f, air, film, glass, 0.0f);
  if (close_value(direct, zero_thickness, 2.0e-6f) == false) {
    std::printf("thinfilm zero-thickness limit failed direct %.9f film %.9f\n", direct, zero_thickness);
    valid = false;
  }

  const float quarter_wave_thickness = wavelength / (4.0f * 1.5f);
  const float quarter_wave = bsdf_fresnel_thinfilm(wavelength, 1.0f, air, film, glass, quarter_wave_thickness);
  if ((std::isfinite(quarter_wave) == false) || (quarter_wave > 2.0e-6f)) {
    std::printf("thinfilm ideal quarter-wave antireflection failed %.9f\n", quarter_wave);
    valid = false;
  }

  const float phase_period = wavelength / (2.0f * 1.5f);
  const float phase_a = bsdf_fresnel_thinfilm(wavelength, 1.0f, air, film, glass, 137.0f);
  const float phase_b = bsdf_fresnel_thinfilm(wavelength, 1.0f, air, film, glass, 137.0f + phase_period);
  if (close_value(phase_a, phase_b, 2.0e-6f) == false) {
    std::printf("thinfilm phase periodicity failed %.9f %.9f\n", phase_a, phase_b);
    valid = false;
  }

  const float grazing = bsdf_fresnel_thinfilm(wavelength, 0.0f, air, film, glass, 400.0f);
  if (close_value(grazing, 1.0f, 1.0e-7f) == false) {
    std::printf("thinfilm grazing limit failed %.9f\n", grazing);
    valid = false;
  }

  for (float tested_wavelength = 390.0f; tested_wavelength <= 830.0f; tested_wavelength += 20.0f) {
    for (uint32_t angle_index = 0u; angle_index <= 20u; ++angle_index) {
      const float cosine = static_cast<float>(angle_index) / 20.0f;
      for (float thickness = 0.0f; thickness <= 2000.0f; thickness += 100.0f) {
        const float reflectance = bsdf_fresnel_thinfilm(tested_wavelength, cosine, air, film, glass, thickness);
        const float transmittance = 1.0f - reflectance;
        if ((std::isfinite(reflectance) == false) || (reflectance < -2.0e-6f) || (reflectance > 1.0f + 2.0e-6f) ||
            (close_value(reflectance + transmittance, 1.0f, 2.0e-6f) == false)) {
          std::printf("thinfilm lossless energy bound failed wavelength %.1f cosine %.3f thickness %.1f R %.9f T %.9f\n", tested_wavelength, cosine, thickness, reflectance,
            transmittance);
          valid = false;
        }
      }
    }
  }

  const etx::SpectralQuery query{wavelength, SpectralFlags::Spectral};
  const ::RefractiveIndexSample ext_ior = make_spectral_ior(query, 1.0f);
  const ::RefractiveIndexSample int_ior = make_spectral_ior(query, 2.25f);
  ::ThinfilmEval evaluated_film = {};
  evaluated_film.ior = make_spectral_ior(query, 1.5f);
  evaluated_film.rgb_wavelengths = kRGBWavelengths;
  evaluated_film.thickness = quarter_wave_thickness;

  evaluated_film.weight = 0.0f;
  const float uncoated = bsdf_fresnel_calculate(query, 1.0f, ext_ior, int_ior, evaluated_film).value;
  evaluated_film.weight = 1.0f;
  const float coated = bsdf_fresnel_calculate(query, 1.0f, ext_ior, int_ior, evaluated_film).value;
  evaluated_film.weight = 0.25f;
  const float partial = bsdf_fresnel_calculate(query, 1.0f, ext_ior, int_ior, evaluated_film).value;
  const float expected_partial = uncoated + 0.25f * (coated - uncoated);
  if ((close_value(uncoated, direct, 2.0e-6f) == false) || (close_value(coated, quarter_wave, 2.0e-6f) == false) || (close_value(partial, expected_partial, 2.0e-6f) == false)) {
    std::printf("thinfilm coverage blend failed bare %.9f coated %.9f partial %.9f expected %.9f\n", uncoated, coated, partial, expected_partial);
    valid = false;
  }

  if (valid) {
    std::printf("thinfilm optical invariants valid\n");
  }
  return valid;
}

bool validate_diffraction_grating_contract(const etx::Scene& scene) {
  (void)scene;
  bool valid = true;
  auto require = [&](const bool condition, const char* label) {
    if (condition == false) {
      std::printf("diffraction grating failed: %s\n", label);
      valid = false;
    }
  };

  constexpr float wavelength_nm = 550.0f;
  constexpr float period_nm = 1600.0f;
  constexpr float optical_path_difference_nm = 0.5f * wavelength_nm;
  constexpr float duty_cycle = 0.5f;
  const float3 local_w_i = float3{0.0f, 0.0f, 1.0f};
  const BSDFDiffractionOrderRange range = bsdf_diffraction_grating_order_range(local_w_i, wavelength_nm, period_nm);
  require((range.minimum == -2) && (range.maximum == 2), "normal-incidence propagating order range");

  const float phase_difference = kPi;
  const float zero_order = bsdf_diffraction_grating_binary_phase_fourier_power(0, duty_cycle, phase_difference);
  const float first_order = bsdf_diffraction_grating_binary_phase_fourier_power(1, duty_cycle, phase_difference);
  const float second_order = bsdf_diffraction_grating_binary_phase_fourier_power(2, duty_cycle, phase_difference);
  require(zero_order < 1.0e-7f, "half-wave binary phase profile cancels the analytic zero order");
  require(close_value(first_order, 4.0f / (kPi * kPi), 5.0e-7f), "half-wave binary phase first-order coefficient");
  require(second_order < 1.0e-12f, "half-duty binary phase profile suppresses even sidebands");

  double parseval_sum = 0.0;
  for (int order = -10000; order <= 10000; ++order) {
    parseval_sum += static_cast<double>(bsdf_diffraction_grating_binary_phase_fourier_power(order, duty_cycle, phase_difference));
  }
  // The omitted 1 / m^2 tail beyond order 10000 is less than 4.1e-5.
  require(std::abs(parseval_sum - 1.0) <= 4.1e-5, "lossless binary phase Fourier power satisfies Parseval within the analytic tail bound");

  etx::Material material = {};
  material.cls = MaterialClass::DiffractionGrating;
  material.reflectance.spectrum_index = SpectrumSpectralWhite;
  material.diffraction_grating.period_nm = period_nm;
  material.diffraction_grating.optical_path_difference_nm = optical_path_difference_nm;
  material.diffraction_grating.duty_cycle = duty_cycle;
  material.diffraction_grating.rotation = 0.0f;

  const float sideband_efficiency = bsdf_diffraction_grating_propagating_sideband_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  const float runtime_zero_order = bsdf_diffraction_grating_zero_order_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  const float propagating_efficiency = bsdf_diffraction_grating_propagating_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  require(std::isfinite(sideband_efficiency) && (sideband_efficiency >= 0.0f) && (sideband_efficiency <= 1.0f), "propagating phase sidebands are finite and bounded");
  require(close_value(runtime_zero_order, zero_order, 2.0e-6f), "runtime zero order equals the analytic binary-phase coefficient");
  require(close_value(propagating_efficiency, runtime_zero_order + sideband_efficiency, 2.0e-6f), "propagating power is the sum of analytic order coefficients");
  require((propagating_efficiency > 0.0f) && (propagating_efficiency < 1.0f), "evanescent Fourier orders are not reassigned to a radiative lobe");

  material.diffraction_grating.optical_path_difference_nm = 0.0f;
  const float flat_sidebands = bsdf_diffraction_grating_propagating_sideband_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  const float flat_zero = bsdf_diffraction_grating_zero_order_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  require(flat_sidebands <= 1.0e-12f, "zero optical path difference has no diffracted sidebands");
  require(close_value(flat_zero, 1.0f, 1.0e-7f), "zero optical path difference is a lossless mirror");
  material.diffraction_grating.optical_path_difference_nm = optical_path_difference_nm;

  float3 blue_first = {};
  float3 red_first = {};
  float3 evanescent = {};
  require(bsdf_diffraction_grating_order_direction(local_w_i, 450.0f, period_nm, 1, blue_first), "blue first order propagates");
  require(bsdf_diffraction_grating_order_direction(local_w_i, 650.0f, period_nm, 1, red_first), "red first order propagates");
  require(red_first.x > blue_first.x, "longer wavelengths diffract farther");
  require(bsdf_diffraction_grating_order_direction(local_w_i, wavelength_nm, period_nm, 3, evanescent) == false, "evanescent order is rejected");

  const float3 oblique_w_i = normalize(float3{0.35f, -0.2f, 1.0f});
  const BSDFDiffractionOrderRange oblique_range = bsdf_diffraction_grating_order_range(oblique_w_i, wavelength_nm, period_nm);
  require(oblique_range.maximum >= oblique_range.minimum, "oblique incidence has propagating orders");
  const float oblique_efficiency = bsdf_diffraction_grating_propagating_efficiency(oblique_range, oblique_w_i, wavelength_nm, period_nm, material);
  require(std::isfinite(oblique_efficiency) && (oblique_efficiency >= 0.0f) && (oblique_efficiency <= 1.0f), "oblique propagating phase-mask power is passive");
  for (int order = oblique_range.minimum; order <= oblique_range.maximum; ++order) {
    float3 oblique_w_o = {};
    require(bsdf_diffraction_grating_order_direction(oblique_w_i, wavelength_nm, period_nm, order, oblique_w_o), "oblique order range contains only propagating directions");
    require(close_value(oblique_w_o.x + oblique_w_i.x, float(order) * wavelength_nm / period_nm, 2.0e-6f), "oblique order satisfies the grating equation");
    require(close_value(oblique_w_o.y, -oblique_w_i.y, 2.0e-6f), "oblique order preserves the groove-parallel wavevector");
    require(close_value(dot(oblique_w_o, oblique_w_o), 1.0f, 2.0e-6f), "oblique order direction is normalized");
    int reverse_order = 0;
    require(bsdf_diffraction_grating_match_order(oblique_w_o, oblique_w_i, wavelength_nm, period_nm, reverse_order), "oblique order maps back under path reversal");
    require(reverse_order == order, "oblique reciprocal path uses the same grating order");
  }

  const float validation_wavelengths[] = {kShortestWavelength, 550.0f, kLongestWavelength};
  const float validation_periods[] = {kDiffractionGratingMinimumPeriodNm, 390.0f, 1600.0f, kDiffractionGratingMaximumPeriodNm};
  const float validation_duties[] = {0.0f, 0.1f, 0.5f, 0.9f, 1.0f};
  const float validation_optical_path_differences[] = {0.0f, 50.0f, 275.0f, 500.0f, kDiffractionGratingMaximumOpticalPathDifferenceNm};
  const float3 validation_incident_directions[] = {
    float3{0.0f, 0.0f, 1.0f},
    normalize(float3{0.8f, 0.1f, 1.0f}),
    normalize(float3{-0.95f, 0.2f, 0.25f}),
  };
  for (const float tested_wavelength : validation_wavelengths) {
    for (const float tested_period : validation_periods) {
      for (const float3 tested_w_i : validation_incident_directions) {
        const BSDFDiffractionOrderRange tested_range = bsdf_diffraction_grating_order_range(tested_w_i, tested_wavelength, tested_period);
        require((tested_range.maximum - tested_range.minimum) <= 64, "supported period range keeps exact order enumeration bounded");
        for (const float tested_duty : validation_duties) {
          for (const float tested_optical_path_difference : validation_optical_path_differences) {
            material.diffraction_grating.period_nm = tested_period;
            material.diffraction_grating.duty_cycle = tested_duty;
            material.diffraction_grating.optical_path_difference_nm = tested_optical_path_difference;
            const float tested_total = bsdf_diffraction_grating_propagating_efficiency(tested_range, tested_w_i, tested_wavelength, tested_period, material);
            require(std::isfinite(tested_total) && (tested_total >= 0.0f) && (tested_total <= 1.0f + 3.0e-6f),
              "phase-mask propagating energy remains finite and passive across the parameter domain");

            float enumerated_total = 0.0f;
            for (int order = tested_range.minimum; order <= tested_range.maximum; ++order) {
              float3 tested_w_o = {};
              if (bsdf_diffraction_grating_order_direction(tested_w_i, tested_wavelength, tested_period, order, tested_w_o)) {
                const float efficiency = bsdf_diffraction_grating_order_efficiency(tested_range, tested_w_i, tested_w_o, tested_wavelength, tested_period, order, material);
                require(std::isfinite(efficiency) && (efficiency >= 0.0f) && (efficiency <= 1.0f), "every phase-grating order efficiency is finite and bounded");
                enumerated_total += efficiency;

                const BSDFDiffractionOrderRange reverse_range = bsdf_diffraction_grating_order_range(tested_w_o, tested_wavelength, tested_period);
                const float reverse_efficiency =
                  bsdf_diffraction_grating_order_efficiency(reverse_range, tested_w_o, tested_w_i, tested_wavelength, tested_period, order, material);
                require(close_value(reverse_efficiency, efficiency, 4.0e-6f), "phase-grating order efficiency is reciprocal");
              }
            }
            require(close_value(enumerated_total, tested_total, 4.0e-6f), "reported propagating power equals the independently enumerated analytic orders");
          }
        }
        for (int order = tested_range.minimum; order <= tested_range.maximum; ++order) {
          float3 tested_w_o = {};
          require(bsdf_diffraction_grating_order_direction(tested_w_i, tested_wavelength, tested_period, order, tested_w_o), "enumerated edge-case order is propagating");
          int reverse_order = 0;
          require(bsdf_diffraction_grating_match_order(tested_w_o, tested_w_i, tested_wavelength, tested_period, reverse_order), "enumerated edge-case order is reciprocal");
          require(reverse_order == order, "edge-case reciprocal path preserves the order index");
        }
      }
    }
  }
  const BSDFDiffractionOrderRange short_wavelength_range = bsdf_diffraction_grating_order_range(local_w_i, kShortestWavelength - 1.0f, period_nm);
  const BSDFDiffractionOrderRange long_wavelength_range = bsdf_diffraction_grating_order_range(local_w_i, kLongestWavelength + 1.0f, period_nm);
  require(short_wavelength_range.maximum < short_wavelength_range.minimum, "wavelengths below the renderer spectral domain are rejected");
  require(long_wavelength_range.maximum < long_wavelength_range.minimum, "wavelengths above the renderer spectral domain are rejected");
  const float invalid_float = std::numeric_limits<float>::quiet_NaN();
  const BSDFDiffractionOrderRange invalid_direction_range = bsdf_diffraction_grating_order_range(float3{invalid_float, 0.0f, 1.0f}, wavelength_nm, period_nm);
  require(invalid_direction_range.maximum < invalid_direction_range.minimum, "non-finite incident directions are rejected before integer order conversion");
  float3 invalid_order_direction = {};
  require(bsdf_diffraction_grating_order_direction(local_w_i, wavelength_nm, invalid_float, 0, invalid_order_direction) == false,
    "non-finite periods are rejected by direction construction");
  int invalid_matched_order = 0;
  require(bsdf_diffraction_grating_match_order(local_w_i, float3{invalid_float, 0.0f, 1.0f}, wavelength_nm, period_nm, invalid_matched_order) == false,
    "non-finite outgoing directions are rejected before integer order conversion");

  material.diffraction_grating.period_nm = period_nm;
  material.diffraction_grating.optical_path_difference_nm = optical_path_difference_nm;
  material.diffraction_grating.duty_cycle = duty_cycle;
  material.diffraction_grating.rotation = 0.0f;

  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  etx::BSDFData data = {etx::SpectralQuery{wavelength_nm, SpectralFlags::Spectral}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};

  constexpr uint32_t sample_count = 65536u;
  uint32_t order_counts[5] = {};
  double mean_weight = 0.0;
  for (uint32_t sample_index = 0u; sample_index < sample_count; ++sample_index) {
    etx::Sampler sampler(0x51f15e5u, sample_index + 1u);
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    require(validate_sample(sample), "sample is finite and non-negative");
    require((sample.properties & BSDFSample::Delta) != 0u, "sample is marked delta");
    require((sample.properties & BSDFSample::Reflection) != 0u, "sample is marked reflection");
    require((sample.properties & BSDFSample::WavelengthDependentDirection) != 0u, "sample is marked wavelength-dependent");
    require(close_value(sample.weight.monochromatic(), propagating_efficiency, 2.0e-6f), "white phase-mask sample carries the physically modeled propagating power");

    int matched_order = 0;
    require(bsdf_diffraction_grating_match_order(local_w_i, sample.w_o, wavelength_nm, period_nm, matched_order), "sample lies on an exact grating order");
    require((matched_order >= -2) && (matched_order <= 2), "sampled order is propagating");
    order_counts[matched_order + 2] += 1u;

    etx::Sampler eval_sampler(17u, sample_index + 3u);
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, sample.w_o, material, eval_sampler);
    etx::Sampler pdf_sampler(23u, sample_index + 5u);
    const float pdf = etx::bsdf::pdf(data, sample.w_o, material, pdf_sampler);
    etx::Sampler reverse_sampler(29u, sample_index + 7u);
    const float reverse_pdf = etx::bsdf::reverse_pdf(data, sample.w_o, material, reverse_sampler);
    float3 matched_w_o = {};
    require(bsdf_diffraction_grating_order_direction(local_w_i, wavelength_nm, period_nm, matched_order, matched_w_o),
      "sampled phase-grating order direction can be reconstructed");
    const float expected_efficiency = bsdf_diffraction_grating_order_efficiency(range, local_w_i, matched_w_o, wavelength_nm, period_nm, matched_order, material);
    require(close_value(eval.bsdf.monochromatic(), expected_efficiency, 2.0e-6f), "evaluate returns the selected phase-order power");
    const float expected_pdf = expected_efficiency / propagating_efficiency;
    require(close_value(sample.pdf, expected_pdf, 2.0e-6f), "sample PDF matches normalized propagating phase-order power");
    require(close_value(pdf, sample.pdf, 2.0e-6f), "sample and queried PDFs agree");
    const BSDFDiffractionOrderRange reverse_range = bsdf_diffraction_grating_order_range(matched_w_o, wavelength_nm, period_nm);
    const float reverse_total = bsdf_diffraction_grating_propagating_efficiency(reverse_range, matched_w_o, wavelength_nm, period_nm, material);
    const float reverse_efficiency = bsdf_diffraction_grating_order_efficiency(reverse_range, matched_w_o, local_w_i, wavelength_nm, period_nm, matched_order, material);
    require(close_value(reverse_pdf, reverse_efficiency / reverse_total, 3.0e-6f), "reverse PDF uses the reverse path's propagating-order normalization");
    etx::BSDFData reverse_data = data;
    reverse_data.w_i = -sample.w_o;
    etx::Sampler reverse_eval_sampler(31u, sample_index + 9u);
    const etx::BSDFEval reverse_eval = etx::bsdf::evaluate(reverse_data, -data.w_i, material, reverse_eval_sampler);
    require(close_value(reverse_eval.bsdf.monochromatic(), eval.bsdf.monochromatic(), 2.0e-6f), "reciprocal BSDF power agrees");
    mean_weight += sample.weight.monochromatic();
  }
  mean_weight /= static_cast<double>(sample_count);
  require(close_value(static_cast<float>(mean_weight), propagating_efficiency, 2.0e-6f), "white-furnace estimator returns the analytic propagating reflected power");

  for (int order = -2; order <= 2; ++order) {
    float3 order_w_o = {};
    require(bsdf_diffraction_grating_order_direction(local_w_i, wavelength_nm, period_nm, order, order_w_o), "histogram phase-grating order direction is valid");
    const float expected_probability = bsdf_diffraction_grating_order_efficiency(range, local_w_i, order_w_o, wavelength_nm, period_nm, order, material) / propagating_efficiency;
    const float observed_probability = static_cast<float>(order_counts[order + 2]) / static_cast<float>(sample_count);
    require(fabsf(observed_probability - expected_probability) <= 0.006f, "sampled order histogram matches analytic probabilities");
  }

  ::BSDFData interop_data = etx::bsdf::detail::make_interop_data(data);
  float3 local_first = {};
  require(bsdf_diffraction_grating_order_direction(local_w_i, wavelength_nm, period_nm, 1, local_first), "rotation reference order propagates");
  const LocalFrame unrotated_frame = bsdf_diffraction_grating_frame(interop_data, material);
  const float3 unrotated_world = local_frame_from_local(unrotated_frame, local_first);
  material.diffraction_grating.rotation = 0.5f * kPi;
  const LocalFrame rotated_frame = bsdf_diffraction_grating_frame(interop_data, material);
  const float3 rotated_world = local_frame_from_local(rotated_frame, local_first);
  require((fabsf(unrotated_world.x) > 0.3f) && (fabsf(unrotated_world.y) < 1.0e-5f), "zero rotation disperses along tangent");
  require((fabsf(rotated_world.y) > 0.3f) && (fabsf(rotated_world.x) < 1.0e-5f), "ninety-degree rotation rotates dispersion axis");

  etx::BSDFData rgb_data = data;
  rgb_data.spectrum_sample = etx::SpectralQuery{};
  const ::BSDFData rgb_interop_data = etx::bsdf::detail::make_interop_data(rgb_data);
  const LocalFrame rgb_frame = bsdf_diffraction_grating_frame(rgb_interop_data, material);
  float3 rgb_efficiency = {};
  constexpr uint32_t wavelength_integration_samples = 4096u;
  for (uint32_t channel = 0u; channel < 3u; ++channel) {
    float channel_efficiency = 0.0f;
    for (uint32_t wavelength_index = 0u; wavelength_index < wavelength_integration_samples; ++wavelength_index) {
      const float wavelength_sample = (static_cast<float>(wavelength_index) + 0.5f) / static_cast<float>(wavelength_integration_samples);
      const float sampled_wavelength = bsdf_diffraction_grating_rgb_sample_wavelength(channel, wavelength_sample);
      const BSDFDiffractionOrderRange sampled_range = bsdf_diffraction_grating_order_range(local_w_i, sampled_wavelength, period_nm);
      channel_efficiency += bsdf_diffraction_grating_propagating_efficiency(sampled_range, local_w_i, sampled_wavelength, period_nm, material);
    }
    rgb_efficiency = bsdf_diffraction_grating_rgb_set_component(rgb_efficiency, channel, channel_efficiency / static_cast<float>(wavelength_integration_samples));
  }
  double rgb_mean_weight_x = 0.0;
  double rgb_mean_weight_y = 0.0;
  double rgb_mean_weight_z = 0.0;
  bool sampled_between_representative_wavelengths = false;
  for (uint32_t sample_index = 0u; sample_index < sample_count; ++sample_index) {
    etx::Sampler rgb_sampler(0x8d41c3bu, sample_index + 1u);
    const etx::BSDFSample rgb_sample = etx::bsdf::sample(rgb_data, material, rgb_sampler);
    require(validate_sample(rgb_sample), "RGB diffraction sample is finite and non-negative");
    require(rgb_sample.valid(), "RGB diffraction sample is valid");
    require((rgb_sample.properties & BSDFSample::Delta) != 0u, "RGB diffraction sample is marked delta");
    require((rgb_sample.properties & BSDFSample::Reflection) != 0u, "RGB diffraction sample is marked reflection");
    require((rgb_sample.properties & BSDFSample::WavelengthDependentDirection) != 0u, "RGB diffraction sample is marked wavelength-dependent");

    uint32_t selected_channel = 0u;
    uint32_t positive_channel_count = 0u;
    for (uint32_t channel = 0u; channel < 3u; ++channel) {
      if (bsdf_diffraction_grating_rgb_component(rgb_sample.weight.integrated, channel) > kEpsilon) {
        selected_channel = channel;
        positive_channel_count += 1u;
      }
    }
    require(positive_channel_count == 1u, "RGB diffraction sample transports one stochastically selected channel");

    bool matched_rgb_order = false;
    const float3 local_rgb_w_o = local_frame_to_local(rgb_frame, rgb_sample.w_o);
    const float grating_shift = (local_rgb_w_o.x + local_w_i.x) * period_nm;
    if (fabsf(grating_shift) <= 1.0e-4f) {
      matched_rgb_order = true;
    } else {
      const float wavelength_center = bsdf_diffraction_grating_rgb_wavelength(selected_channel);
      const float wavelength_span = bsdf_diffraction_grating_rgb_wavelength_span(selected_channel);
      for (int order = -64; order <= 64; ++order) {
        if (order == 0) {
          continue;
        }
        const float sampled_wavelength = grating_shift / static_cast<float>(order);
        if ((sampled_wavelength < (wavelength_center - wavelength_span - 1.0e-3f)) || (sampled_wavelength > (wavelength_center + wavelength_span + 1.0e-3f))) {
          continue;
        }
        int matched_order = 0;
        if (bsdf_diffraction_grating_match_order(local_w_i, local_rgb_w_o, sampled_wavelength, period_nm, matched_order) && (matched_order == order)) {
          matched_rgb_order = true;
          sampled_between_representative_wavelengths = sampled_between_representative_wavelengths || (fabsf(sampled_wavelength - wavelength_center) > 0.1f);
          break;
        }
      }
    }
    require(matched_rgb_order, "RGB diffraction sample follows a wavelength within the selected channel span");
    rgb_mean_weight_x += rgb_sample.weight.integrated.x;
    rgb_mean_weight_y += rgb_sample.weight.integrated.y;
    rgb_mean_weight_z += rgb_sample.weight.integrated.z;
  }
  require(sampled_between_representative_wavelengths, "RGB diffraction samples continuous wavelengths instead of only three representatives");
  const double inverse_rgb_sample_count = 1.0 / static_cast<double>(sample_count);
  require(close_value(static_cast<float>(rgb_mean_weight_x * inverse_rgb_sample_count), rgb_efficiency.x, 0.01f) &&
            close_value(static_cast<float>(rgb_mean_weight_y * inverse_rgb_sample_count), rgb_efficiency.y, 0.01f) &&
            close_value(static_cast<float>(rgb_mean_weight_z * inverse_rgb_sample_count), rgb_efficiency.z, 0.01f),
    "RGB diffraction estimator returns the three-channel propagating power");

  material.diffraction_grating.period_nm = kDiffractionGratingMaximumPeriodNm + 1.0f;
  etx::Sampler invalid_period_sampler(47u, 53u);
  const etx::BSDFSample invalid_period_sample = etx::bsdf::sample(data, material, invalid_period_sampler);
  require(invalid_period_sample.weight.maximum() == 0.0f, "out-of-contract periods fail closed instead of being silently clamped at runtime");
  material.diffraction_grating.period_nm = invalid_float;
  etx::Sampler nonfinite_period_sampler(59u, 61u);
  const etx::BSDFSample nonfinite_period_sample = etx::bsdf::sample(data, material, nonfinite_period_sampler);
  require(nonfinite_period_sample.weight.maximum() == 0.0f, "non-finite material parameters fail closed");

  material.diffraction_grating.period_nm = period_nm;
  material.diffraction_grating.optical_path_difference_nm = kDiffractionGratingMaximumOpticalPathDifferenceNm + 1.0f;
  etx::Sampler invalid_depth_sampler(67u, 71u);
  const etx::BSDFSample invalid_depth_sample = etx::bsdf::sample(data, material, invalid_depth_sampler);
  require(invalid_depth_sample.weight.maximum() == 0.0f, "out-of-contract optical path differences fail closed");
  material.diffraction_grating.optical_path_difference_nm = invalid_float;
  etx::Sampler nonfinite_depth_sampler(73u, 79u);
  const etx::BSDFSample nonfinite_depth_sample = etx::bsdf::sample(data, material, nonfinite_depth_sampler);
  require(nonfinite_depth_sample.weight.maximum() == 0.0f, "non-finite optical path differences fail closed");

  if (valid) {
    std::printf("diffraction grating contract valid: phase mask period %.1f nm optical path difference %.1f nm duty %.3f propagating power %.9f sampled %u paths\n", period_nm,
      optical_path_difference_nm, duty_cycle, propagating_efficiency, sample_count);
  }
  return valid;
}

bool validate_diffraction_grating_serialization() {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.mediums.init(16u);
  scene_data.defaults.missing_material = scene_data.add_material("__missing");

  etx::MaterialDefinition canonical = {};
  canonical.name = "diffraction-canonical";
  canonical.properties["material"] = "class diffraction_grating";
  canonical.properties["diffraction_grating"] = "period_nm 1700 optical_path_difference_nm 321 duty_cycle 0.4 rotation_degrees 25";

  etx::MaterialDefinition legacy = {};
  legacy.name = "diffraction-legacy";
  legacy.properties["material"] = "class diffraction_grating";
  legacy.properties["diffraction_grating"] = "period_nm 1800 groove_depth_nm 123 duty_cycle 0.3 rotation_degrees -40";

  etx::IORDatabase ior_database = {};
  etx::SceneSerialization serialization;
  serialization.parse_material_definitions("", {canonical, legacy}, scene_data, ior_database, scheduler);

  const auto canonical_index = scene_data.material_mapping.find(canonical.name);
  const auto legacy_index = scene_data.material_mapping.find(legacy.name);
  if ((canonical_index == scene_data.material_mapping.end()) || (legacy_index == scene_data.material_mapping.end())) {
    std::printf("diffraction grating serialization failed: material mapping missing\n");
    return false;
  }

  const etx::Material& canonical_material = scene_data.materials[canonical_index->second];
  const etx::Material& legacy_material = scene_data.materials[legacy_index->second];
  const bool valid =
    (canonical_material.cls == MaterialClass::DiffractionGrating) && close_value(canonical_material.diffraction_grating.period_nm, 1700.0f, 1.0e-6f) &&
    close_value(canonical_material.diffraction_grating.optical_path_difference_nm, 321.0f, 1.0e-6f) &&
    close_value(canonical_material.diffraction_grating.duty_cycle, 0.4f, 1.0e-6f) && close_value(canonical_material.diffraction_grating.rotation, 25.0f * kPi / 180.0f, 1.0e-6f) &&
    (legacy_material.cls == MaterialClass::DiffractionGrating) && close_value(legacy_material.diffraction_grating.period_nm, 1800.0f, 1.0e-6f) &&
    close_value(legacy_material.diffraction_grating.optical_path_difference_nm, 246.0f, 1.0e-6f) && close_value(legacy_material.diffraction_grating.duty_cycle, 0.3f, 1.0e-6f) &&
    close_value(legacy_material.diffraction_grating.rotation, -40.0f * kPi / 180.0f, 1.0e-6f);
  if (valid == false) {
    std::printf("diffraction grating serialization failed: canonical or legacy fields changed\n");
  }
  return valid;
}

void set_test_float4_channel(float4& value, uint32_t channel, float scalar) {
  if (channel == 0u) {
    value.x = scalar;
  } else if (channel == 1u) {
    value.y = scalar;
  } else if (channel == 2u) {
    value.z = scalar;
  } else {
    value.w = scalar;
  }
}

bool validate_image_3d_sampling() {
  etx::BufferPool buffer_pool;
  std::vector<etx::Image> images;
  etx::ImagePool image_pool(images, buffer_pool);
  image_pool.init(4u);

  const float4 rgba_pixels[] = {
    float4{0.0f, 0.0f, 0.0f, 1.0f},
    float4{1.0f, 0.0f, 0.0f, 1.0f},
    float4{2.0f, 0.0f, 0.0f, 1.0f},
    float4{3.0f, 0.0f, 0.0f, 1.0f},
    float4{4.0f, 0.0f, 0.0f, 1.0f},
    float4{5.0f, 0.0f, 0.0f, 1.0f},
    float4{6.0f, 0.0f, 0.0f, 1.0f},
    float4{7.0f, 0.0f, 0.0f, 1.0f},
  };
  float r32_pixels[] = {
    0.0f,
    1.0f,
    2.0f,
    3.0f,
    4.0f,
    5.0f,
    6.0f,
    7.0f,
  };

  const uint32_t rgba_index = image_pool.add_from_data_3d(rgba_pixels, uint3{2u, 2u, 2u}, 0u, {}, float3{1.0f, 1.0f, 1.0f});
  const uint32_t r32_index = image_pool.add_from_data_3d_r32(r32_pixels, uint3{2u, 2u, 2u}, 0u, {}, float3{1.0f, 1.0f, 1.0f});
  const etx::Image& rgba_image = image_pool.get(rgba_index);
  const etx::Image& r32_image = image_pool.get(r32_index);

  bool valid = true;
  const float4 rgba_center = rgba_image.evaluate_rgba32f_fast_3d(float3{0.25f, 0.25f, 0.25f});
  const float r32_center = r32_image.evaluate_r32f_fast_3d(float3{0.25f, 0.25f, 0.25f});
  if ((close_value(rgba_center.x, 3.5f, 1.0e-5f) == false) || (close_value(r32_center, 3.5f, 1.0e-5f) == false)) {
    std::printf("3D image trilinear sampling failed rgba %.6f r32 %.6f\n", rgba_center.x, r32_center);
    valid = false;
  }

  const float4 rgba_2d = rgba_image.evaluate_rgba32f_fast(float2{0.25f, 0.25f});
  if (close_value(rgba_2d.x, 1.5f, 1.0e-5f) == false) {
    std::printf("2D wrapper sampling changed %.6f\n", rgba_2d.x);
    valid = false;
  }

  const uint32_t repeat_index = image_pool.add_from_data_3d_r32(r32_pixels, uint3{2u, 2u, 2u}, etx::Image::RepeatW, {}, float3{1.0f, 1.0f, 1.0f});
  const etx::Image& repeat_image = image_pool.get(repeat_index);
  const float repeated = repeat_image.evaluate_r32f_fast_3d(float3{0.0f, 0.0f, 1.25f});
  if (close_value(repeated, 2.0f, 1.0e-5f) == false) {
    std::printf("3D image RepeatW sampling failed %.6f\n", repeated);
    valid = false;
  }

  const float3 medium_samples[] = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.25f, 0.25f, 0.25f},
    float3{0.5f, 0.5f, 0.5f},
    float3{0.75f, 0.75f, 0.75f},
    float3{0.99f, 0.1f, 0.6f},
    float3{-0.01f, 0.5f, 0.5f},
    float3{1.0f, 0.5f, 0.5f},
  };
  etx::MediumTextureSampleContext medium_context = {etx::ArrayView<float>{r32_pixels, 8u}};
  for (const float3& local_coord : medium_samples) {
    const float old_value = etx::medium_texture_sample_shared_3d(medium_context, local_coord, uint3{2u, 2u, 2u});
    float3 uvw = {};
    float image_value = 0.0f;
    if (::medium_density_shared_texture_uvw(local_coord, uint3{2u, 2u, 2u}, uvw)) {
      image_value = r32_image.evaluate_r32f_fast_3d(uvw);
    }
    if (close_value(old_value, image_value, 1.0e-5f) == false) {
      std::printf("Medium density image parity failed old %.6f image %.6f\n", old_value, image_value);
      valid = false;
    }
  }

  if (valid) {
    std::printf("3D image sampling valid\n");
  }

  image_pool.cleanup();
  return valid;
}

bool validate_spectral_energy_compensation_lut_sampling() {
  etx::BufferPool buffer_pool;
  std::vector<etx::Image> images;
  etx::ImagePool image_pool(images, buffer_pool);
  image_pool.init(4u);

  const uint32_t thinfilm_slice_count = 7u;
  const uint32_t wavelength_group_count = kBSDFEnergyCompensationSpectralWavelengthGroupCount;
  const uint32_t depth = thinfilm_slice_count * wavelength_group_count;
  std::vector<float4> pixels(depth, float4{0.0f, 0.0f, 0.0f, 0.0f});
  for (uint32_t thinfilm_slice = 0u; thinfilm_slice < thinfilm_slice_count; ++thinfilm_slice) {
    for (uint32_t group = 0u; group < wavelength_group_count; ++group) {
      const uint32_t layer = thinfilm_slice * wavelength_group_count + group;
      for (uint32_t channel = 0u; channel < kBSDFEnergyCompensationSpectralWavelengthGroupSize; ++channel) {
        const uint32_t wavelength_index = group * kBSDFEnergyCompensationSpectralWavelengthGroupSize + channel;
        set_test_float4_channel(pixels[layer], channel, static_cast<float>(wavelength_index + thinfilm_slice * 1000u));
      }
    }
  }

  const uint32_t image_index = image_pool.add_from_data_3d(pixels.data(), uint3{1u, 1u, depth}, etx::Image::SkipSRGBConversion, {}, float3{1.0f, 1.0f, 1.0f});

  etx::Scene::EnergyCompensationInterface interface_data = {};
  interface_data.cache_mode = kBSDFEnergyCompensationCacheModeSpectralScalar;
  interface_data.spectral_wavelength_count = kBSDFEnergyCompensationSpectralWavelengthCount;
  interface_data.thinfilm_slice_count = thinfilm_slice_count;
  interface_data.spectral_shortest_wavelength = kShortestWavelength;
  interface_data.spectral_longest_wavelength = kLongestWavelength;

  etx::Scene scene = {};
  scene.images = etx::ArrayView<etx::Image>{images.data(), images.size()};
  scene.energy_compensation_interfaces = etx::ArrayView<etx::Scene::EnergyCompensationInterface>{&interface_data, 1u};
  const BSDFResourceContext context = make_bsdf_resource_cpu_context(scene);

  bool valid = true;
  etx::SpectralQuery spect = etx::SpectralQuery::spectral_sample(0.5f);
  spect.wavelength = kShortestWavelength + (kLongestWavelength - kShortestWavelength) * (5.0f / 127.0f);
  const float exact_value = bsdf_energy_compensated_sample_spectral_scalar_image(context, image_index, spect, float2{0.0f, 0.0f}, 1u, 1u, 0.0f, interface_data.cache_mode,
    interface_data.spectral_wavelength_count, interface_data.thinfilm_slice_count, interface_data.spectral_shortest_wavelength, interface_data.spectral_longest_wavelength);
  if (close_value(exact_value, 5.0f, 1.0e-5f) == false) {
    std::printf("Spectral energy-compensation LUT exact wavelength failed %.6f\n", exact_value);
    valid = false;
  }

  spect.wavelength = kShortestWavelength + (kLongestWavelength - kShortestWavelength) * (5.5f / 127.0f);
  const float wavelength_interp_value =
    bsdf_energy_compensated_sample_spectral_scalar_image(context, image_index, spect, float2{0.0f, 0.0f}, 1u, 1u, 0.0f, interface_data.cache_mode,
      interface_data.spectral_wavelength_count, interface_data.thinfilm_slice_count, interface_data.spectral_shortest_wavelength, interface_data.spectral_longest_wavelength);
  if (close_value(wavelength_interp_value, 5.5f, 1.0e-5f) == false) {
    std::printf("Spectral energy-compensation LUT wavelength interpolation failed %.6f\n", wavelength_interp_value);
    valid = false;
  }

  const float thinfilm_interp_value = bsdf_energy_compensated_sample_spectral_scalar_image(context, image_index, spect, float2{0.0f, 0.0f}, 1u, 1u, 0.5f, interface_data.cache_mode,
    interface_data.spectral_wavelength_count, interface_data.thinfilm_slice_count, interface_data.spectral_shortest_wavelength, interface_data.spectral_longest_wavelength);
  if (close_value(thinfilm_interp_value, 3005.5f, 1.0e-5f) == false) {
    std::printf("Spectral energy-compensation LUT thinfilm interpolation failed %.6f\n", thinfilm_interp_value);
    valid = false;
  }

  spect.wavelength = kShortestWavelength + (kLongestWavelength - kShortestWavelength) * (100.0f / 127.0f);
  const float exact_layer_value =
    bsdf_energy_compensated_sample_spectral_scalar_image(context, image_index, spect, float2{0.0f, 0.0f}, 1u, 1u, 1.0f / 6.0f, interface_data.cache_mode,
      interface_data.spectral_wavelength_count, interface_data.thinfilm_slice_count, interface_data.spectral_shortest_wavelength, interface_data.spectral_longest_wavelength);
  if (close_value(exact_layer_value, 1100.0f, 1.0e-6f) == false) {
    std::printf("Spectral energy-compensation LUT exact packed layer failed %.6f\n", exact_layer_value);
    valid = false;
  }

  etx::Material material = {};
  material.energy_compensation_interface_index = 0u;
  const etx::SpectralQuery sample = etx::SpectralQuery::spectral_sample(0.0f);
  const ::SpectralResponse sample_value = bsdf_energy_compensated_lut_response(context, sample, material, image_index, float2{0.0f, 0.0f}, 1u, 1u, 0.0f);
  if ((close_value(sample_value.value, 0.0f, 1.0e-5f) == false) || (close_value(sample_value.integrated.x, 0.0f, 1.0e-5f) == false) ||
      (close_value(sample_value.integrated.y, 0.0f, 1.0e-5f) == false) || (close_value(sample_value.integrated.z, 0.0f, 1.0e-5f) == false)) {
    std::printf("Spectral energy-compensation LUT sampling failed %.6f %.6f %.6f %.6f\n", sample_value.value, sample_value.integrated.x, sample_value.integrated.y,
      sample_value.integrated.z);
    valid = false;
  }

  return valid;
}

etx::RHIBackend select_default_backend() {
#if ETX_PLATFORM_APPLE
  return etx::RHIBackend::Metal;
#else
  return etx::RHIBackend::Vulkan;
#endif
}

bool validate_energy_compensation_gpu_shader_compile() {
  auto& compiler = etx::ShaderCompiler::instance();
  if (compiler.initialize() != etx::RHIResult::Success) {
    std::printf("Energy-compensation GPU shader compiler initialization failed\n");
    return false;
  }

  const std::vector<etx::ShaderCompiler::ShaderEntryPoint> entry_points = {{"main", etx::RHIShaderStage::Compute}};
  const auto compilation = compiler.compile("shaders/bsdf_energy_compensation.hlsl", entry_points, {}, select_default_backend());
  bool valid = true;
  if ((compilation.result != etx::RHIResult::Success) || compilation.binaries.empty()) {
    std::printf("Energy-compensation GPU shader compilation failed: %s\n", compilation.error_message.c_str());
    return false;
  }

  const etx::RHIShaderBinary& binary = compilation.binaries[0];
  if (((binary.local_size_x != 8u) || (binary.local_size_y != 8u)) || (binary.local_size_z != 1u)) {
    std::printf("Energy-compensation GPU shader local size changed: %u %u %u\n", binary.local_size_x, binary.local_size_y, binary.local_size_z);
    valid = false;
  }

  if (binary.spirv_size == 0u) {
    std::printf("Energy-compensation GPU shader produced empty binary\n");
    valid = false;
  }

  if (valid) {
    std::printf("Energy-compensation GPU shader compile valid\n");
  }
  return valid;
}

bool validate_diffraction_grating_gpu_shader_compile() {
  auto& compiler = etx::ShaderCompiler::instance();
  if (compiler.initialize() != etx::RHIResult::Success) {
    std::printf("Diffraction-grating GPU shader compiler initialization failed\n");
    return false;
  }

  const std::unordered_map<std::string, std::string> validation_defines = {
    {"ETX_BSDF_RUNTIME_VALIDATION_MODE", "3"},
    {"ETX_BSDF_RUNTIME_VALIDATION_OPERATION", "0"},
    {"ETX_DXC_OPT_LEVEL", "0"},
    {"ETX_DXC_SPIRV_OPT_CONFIG", "--compact-ids"},
  };
  const auto compilation = compiler.compile("shaders/bsdf_runtime_validation.hlsl", {{"main", etx::RHIShaderStage::Compute}}, validation_defines, select_default_backend());
  if ((compilation.result != etx::RHIResult::Success) || compilation.binaries.empty() || (compilation.binaries[0].spirv_size == 0u)) {
    std::printf("Diffraction-grating GPU shader compilation failed: %s\n", compilation.error_message.c_str());
    return false;
  }

  struct ProductionShaderVariant {
    const char* source_file;
    const char* entry_point;
  };
  const ProductionShaderVariant production_variants[] = {
    {"shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl", "wavefront_camera_direct_light_prepare_diffuse_main"},
    {"shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl", "wavefront_camera_connect_light_prepare_diffuse_main"},
    {"shaders/gpu_rt_wavefront_connect_light_resolve_variant.hlsl", "wavefront_camera_connect_light_resolve_diffuse_main"},
    {"shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl", "wavefront_camera_continue_prepare_diffuse_main"},
    {"shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl", "wavefront_light_continue_prepare_diffuse_main"},
    {"shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl", "wavefront_light_connect_camera_prepare_diffuse_main"},
  };
  const char* spectral_modes[] = {"1", "2"};
  for (const char* spectral_mode : spectral_modes) {
    for (const ProductionShaderVariant& variant : production_variants) {
      const std::unordered_map<std::string, std::string> production_defines = {
        {"ETX_BSDF_KIND", "1"},
        {"ETX_STAGE_ENTRY", variant.entry_point},
        {"ETX_WAVEFRONT_PATH_TRACING_ONLY", "0"},
        {"ETX_SPECTRAL_MODE", spectral_mode},
        {"ETX_DXC_OPT_LEVEL", "0"},
        {"ETX_DXC_SPIRV_OPT_CONFIG", "--compact-ids"},
      };
      const auto production_compilation =
        compiler.compile(variant.source_file, {{variant.entry_point, etx::RHIShaderStage::Compute}}, production_defines, select_default_backend());
      if ((production_compilation.result != etx::RHIResult::Success) || production_compilation.binaries.empty() || (production_compilation.binaries[0].spirv_size == 0u)) {
        std::printf("Diffraction-grating production GPU shader compilation failed for %s in spectral mode %s: %s\n", variant.entry_point, spectral_mode,
          production_compilation.error_message.c_str());
        return false;
      }
    }
  }

  std::printf("diffraction grating validation and production GPU shader variants compile valid\n");
  return true;
}

bool validate_energy_compensation_gpu_lut_parity() {
  etx::RHIInitInfo init_info = {
    .backend = select_default_backend(),
    .enable_validation = ETX_DEBUG,
    .headless = true,
  };
  etx::RHIContext rhi = etx::RHIContext::create(init_info);
  if (rhi.valid() == false) {
    std::printf("Energy-compensation GPU LUT parity failed to create RHI context\n");
    return false;
  }

  rhi.initialize_headless();
  etx::TaskScheduler scheduler = {};
  const bool valid = etx::validate_energy_compensation_gpu_lut_parity(rhi, scheduler);
  rhi.wait_idle();
  rhi = {};

  if (valid) {
    std::printf("Energy-compensation GPU LUT parity valid\n");
  }
  return valid;
}

float integrate_pdf(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler sampler(seed, seed ^ 0x9e3779b9u);
  float sum = 0.0f;

  for (uint32_t i = 0u; i < kIntegrationSamples; ++i) {
    const float3 w_o = sample_uniform_sphere(sampler);
    etx::Sampler pdf_sampler(seed + i, seed ^ i);
    const float pdf = etx::bsdf::pdf(data, w_o, material, pdf_sampler);
    if (std::isfinite(pdf) == false) {
      return -1.0f;
    }
    if (pdf < 0.0f) {
      return -1.0f;
    }
    sum += pdf;
  }

  return (sum / static_cast<float>(kIntegrationSamples)) * 4.0f * kPi;
}

float integrate_reverse_pdf(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler sampler(seed, seed ^ 0x63d8a1cbu);
  float sum = 0.0f;

  for (uint32_t i = 0u; i < kIntegrationSamples; ++i) {
    const float3 w_o = sample_uniform_sphere(sampler);
    etx::Sampler pdf_sampler(seed + i, seed ^ i);
    const float pdf = etx::bsdf::reverse_pdf(data, w_o, material, pdf_sampler);
    if (std::isfinite(pdf) == false) {
      return -1.0f;
    }
    if (pdf < 0.0f) {
      return -1.0f;
    }
    sum += pdf;
  }

  return (sum / static_cast<float>(kIntegrationSamples)) * 4.0f * kPi;
}

bool validate_sampling(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 17u + 3u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if (validate_sample(sample) == false) {
      return false;
    }
  }

  return true;
}

bool validate_sample_pdf_match(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 19u + 5u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if (validate_sample(sample) == false) {
      return false;
    }

    if (sample.valid() == false) {
      continue;
    }

    etx::Sampler pdf_sampler(seed + i + 100000u, seed ^ (i * 23u + 13u));
    const float pdf = etx::bsdf::pdf(data, sample.w_o, material, pdf_sampler);
    if (std::isfinite(pdf) == false) {
      return false;
    }

    const float tolerance = max(1.0e-3f, 0.05f * max(sample.pdf, pdf));
    if (fabsf(sample.pdf - pdf) > tolerance) {
      return false;
    }
  }

  return true;
}

bool validate_reverse_pdf_match(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler sampler(seed, seed ^ 0x8ed4f2a7u);

  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    const float3 w_o = sample_uniform_sphere(sampler);
    etx::Sampler reverse_pdf_sampler(seed + i + 120000u, seed ^ (i * 29u + 17u));
    const float reverse_pdf = etx::bsdf::reverse_pdf(data, w_o, material, reverse_pdf_sampler);
    if (std::isfinite(reverse_pdf) == false) {
      return false;
    }

    etx::BSDFData reverse_data = data;
    reverse_data.w_i = -w_o;
    if (data.path_source == etx::PathSource::Camera) {
      reverse_data.path_source = etx::PathSource::Light;
    } else if (data.path_source == etx::PathSource::Light) {
      reverse_data.path_source = etx::PathSource::Camera;
    }
    etx::Sampler expected_pdf_sampler(seed + i + 140000u, seed ^ (i * 31u + 19u));
    const float expected_pdf = etx::bsdf::pdf(reverse_data, -data.w_i, material, expected_pdf_sampler);
    if (std::isfinite(expected_pdf) == false) {
      return false;
    }

    const float tolerance = max(1.0e-3f, 0.05f * max(reverse_pdf, expected_pdf));
    if (fabsf(reverse_pdf - expected_pdf) > tolerance) {
      return false;
    }
  }

  return true;
}

bool validate_equal_ior_dielectric_direction(const char* label, const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;

  etx::Sampler sampler(seed, seed ^ 0x9234a7bu);
  const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
  const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
  const bool transmission = (sample.properties & etx::BSDFSample::Transmission) != 0u;
  const bool medium_changed = (sample.properties & etx::BSDFSample::MediumChanged) != 0u;
  const bool entering = dot(data.nrm, data.w_i) < 0.0f;
  const uint32_t expected_medium = entering ? material_with_media.int_medium : material_with_media.ext_medium;
  const float3 direction_error = sample.w_o - data.w_i;

  if ((sample.valid() == false) || (sample.is_delta() == false) || reflection || (transmission == false) || (medium_changed == false) || (sample.medium_index != expected_medium) ||
      (fabsf(sample.pdf - 1.0f) > kEpsilon) || (fabsf(sample.eta - 1.0f) > kEpsilon) || (sample.weight.valid() == false) ||
      (fabsf(sample.weight.monochromatic() - 1.0f) > kEpsilon) || (dot(direction_error, direction_error) > 1.0e-8f)) {
    std::printf("%s equal-IOR sample failed pdf %.6f eta %.6f medium %u expected %u weight %.6f\n", label, sample.pdf, sample.eta, sample.medium_index, expected_medium,
      sample.weight.monochromatic());
    return false;
  }

  etx::Sampler eval_sampler(seed + 1u, seed ^ 0x34bac21u);
  const etx::BSDFEval eval = etx::bsdf::evaluate(data, data.w_i, material_with_media, eval_sampler);
  etx::Sampler pdf_sampler(seed + 2u, seed ^ 0x7129ea1u);
  const float pdf = etx::bsdf::pdf(data, data.w_i, material_with_media, pdf_sampler);
  if ((eval.bsdf.maximum() > kEpsilon) || (pdf > kEpsilon)) {
    std::printf("%s equal-IOR delta eval/pdf failed bsdf %.6f pdf %.6f\n", label, eval.bsdf.maximum(), pdf);
    return false;
  }

  return true;
}

struct SampleWeightStats {
  float3 average = {};
  float average_inverse_pdf = 0.0f;
  float p95_weight = 0.0f;
  float p99_weight = 0.0f;
  float max_weight = 0.0f;
  uint32_t valid_count = 0u;
  uint32_t nonzero_count = 0u;
  uint32_t invalid_count = 0u;
  uint32_t zero_pdf_count = 0u;
};

SampleWeightStats sample_weight_stats_rgb(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  SampleWeightStats result = {};
  float weights[kBsdfSamples] = {};

  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 41u + 23u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if (validate_sample(sample) == false) {
      result.average = float3{-1.0f, -1.0f, -1.0f};
      result.invalid_count += 1u;
      return result;
    }

    weights[i] = 0.0f;
    if (sample.valid() == false) {
      result.zero_pdf_count += 1u;
      continue;
    }

    result.average += sample.weight.to_rgb();
    if (sample.pdf > kEpsilon) {
      result.average_inverse_pdf += 1.0f / sample.pdf;
    }
    const float weight = sample.weight.maximum();
    weights[result.valid_count] = weight;
    result.max_weight = max(result.max_weight, weight);
    result.valid_count += 1u;
    if (sample.weight.maximum() > kEpsilon) {
      result.nonzero_count += 1u;
    }
  }

  std::sort(weights, weights + kBsdfSamples);
  const uint32_t p95_index = min(kBsdfSamples - 1u, static_cast<uint32_t>(0.95f * static_cast<float>(kBsdfSamples - 1u)));
  const uint32_t p99_index = min(kBsdfSamples - 1u, static_cast<uint32_t>(0.99f * static_cast<float>(kBsdfSamples - 1u)));
  result.p95_weight = weights[p95_index];
  result.p99_weight = weights[p99_index];

  if (result.valid_count > 0u) {
    result.average *= (1.0f / static_cast<float>(kBsdfSamples));
    result.average_inverse_pdf *= (1.0f / static_cast<float>(result.valid_count));
  }
  return result;
}

float integrate_bsdf_energy(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler sampler(seed, seed ^ 0x517cc1b7u);
  float sum = 0.0f;

  for (uint32_t i = 0u; i < kIntegrationSamples; ++i) {
    const float3 w_o = sample_uniform_sphere(sampler);
    etx::Sampler eval_sampler(seed + i, seed ^ (i * 13u + 11u));
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, w_o, material, eval_sampler);
    if (finite_response(eval.bsdf) == false) {
      return -1.0f;
    }
    if (non_negative_response(eval.bsdf) == false) {
      return -1.0f;
    }
    sum += eval.bsdf.monochromatic();
  }

  return (sum / static_cast<float>(kIntegrationSamples)) * 4.0f * kPi;
}

etx::Material make_conductor(const float roughness) {
  etx::Material result = {};
  result.cls = MaterialClass::Conductor;
  result.reflectance.spectrum_index = SpectrumWhite;
  result.roughness.value = float4{roughness, roughness, 0.0f, 0.0f};
  result.ext_ior.cls = etx::SpectralDistribution::Dielectric;
  result.int_ior.cls = etx::SpectralDistribution::Conductor;
  result.int_ior.eta_index = SpectrumConductorEta;
  result.int_ior.k_index = SpectrumConductorK;
  return result;
}

etx::Material make_mirror_conductor(const float roughness) {
  etx::Material result = make_conductor(roughness);
  result.int_ior.eta_index = SpectrumMirrorEta;
  result.int_ior.k_index = SpectrumMirrorK;
  return result;
}

etx::Material make_dielectric(const float roughness) {
  etx::Material result = {};
  result.cls = MaterialClass::Dielectric;
  result.reflectance.spectrum_index = SpectrumWhite;
  result.scattering.spectrum_index = SpectrumColored;
  result.roughness.value = float4{roughness, roughness, 0.0f, 0.0f};
  result.ext_ior.cls = etx::SpectralDistribution::Dielectric;
  result.int_ior.cls = etx::SpectralDistribution::Dielectric;
  result.int_ior.eta_index = SpectrumDielectricEta;
  return result;
}

etx::Material make_white_dielectric(const float roughness) {
  etx::Material result = make_dielectric(roughness);
  result.scattering.spectrum_index = SpectrumWhite;
  return result;
}

etx::Material make_white_sapphire_dielectric(const float roughness) {
  etx::Material result = make_white_dielectric(roughness);
  result.int_ior.eta_index = SpectrumSapphireEta;
  return result;
}

etx::Material make_white_equal_ior_dielectric(const float roughness) {
  etx::Material result = make_white_dielectric(roughness);
  result.ext_ior.eta_index = SpectrumAirEta;
  result.int_ior.eta_index = SpectrumAirEta;
  return result;
}

etx::Material make_named_plastic_water_dielectric(const float roughness) {
  etx::Material result = make_white_dielectric(roughness);
  result.ext_ior.eta_index = SpectrumNamedPlasticEta;
  result.ext_ior.k_index = SpectrumNamedPlasticK;
  result.int_ior.eta_index = SpectrumNamedWaterEta;
  result.int_ior.k_index = SpectrumNamedWaterK;
  return result;
}

etx::Material make_diffuse(const float roughness) {
  etx::Material result = {};
  result.cls = MaterialClass::Diffuse;
  result.scattering.spectrum_index = SpectrumWhite;
  result.roughness.value = float4{roughness, roughness, 0.0f, 0.0f};
  return result;
}

etx::Material make_velvet(const float roughness) {
  etx::Material result = {};
  result.cls = MaterialClass::Velvet;
  result.reflectance.spectrum_index = SpectrumWhite;
  result.scattering.spectrum_index = SpectrumWhite;
  result.roughness.value = float4{roughness, roughness, 0.0f, 0.0f};
  return result;
}

etx::Material make_balanced_translucent(const float roughness) {
  etx::Material result = {};
  result.cls = MaterialClass::Translucent;
  result.reflectance.spectrum_index = SpectrumHalf;
  result.scattering.spectrum_index = SpectrumHalf;
  result.roughness.value = float4{roughness, roughness, 0.0f, 0.0f};
  return result;
}

etx::Material make_plastic(const float roughness) {
  etx::Material result = {};
  result.cls = MaterialClass::Plastic;
  result.reflectance.spectrum_index = SpectrumWhite;
  result.scattering.spectrum_index = SpectrumWhite;
  result.roughness.value = float4{roughness, roughness, 0.0f, 0.0f};
  result.ext_ior.cls = etx::SpectralDistribution::Dielectric;
  result.ext_ior.eta_index = SpectrumAirEta;
  result.int_ior.cls = etx::SpectralDistribution::Dielectric;
  result.int_ior.eta_index = SpectrumDielectricEta;
  return result;
}

etx::Material make_black_substrate_plastic(const float roughness) {
  etx::Material result = make_plastic(roughness);
  result.scattering.spectrum_index = SpectrumBlack;
  return result;
}

etx::Material make_colored_substrate_black_coating_plastic(const float roughness) {
  etx::Material result = make_plastic(roughness);
  result.reflectance.spectrum_index = SpectrumBlack;
  result.scattering.spectrum_index = SpectrumColored;
  return result;
}

void set_basic_thinfilm(etx::Material& material, const float min_thickness, const float max_thickness) {
  material.thinfilm.min_thickness = min_thickness;
  material.thinfilm.max_thickness = max_thickness;
  material.thinfilm.ior.cls = etx::SpectralDistribution::Dielectric;
  material.thinfilm.ior.eta_index = SpectrumDielectricEta;
  material.thinfilm.ior.k_index = SpectrumBlack;
  material.thinfilm.weight = 1.0f;
}

etx::Material make_standalone_thinfilm(const float min_thickness, const float max_thickness) {
  etx::Material result = {};
  result.cls = MaterialClass::Thinfilm;
  result.reflectance.spectrum_index = SpectrumWhite;
  result.scattering.spectrum_index = SpectrumWhite;
  result.ext_ior.cls = etx::SpectralDistribution::Dielectric;
  result.ext_ior.eta_index = SpectrumAirEta;
  result.int_ior.cls = etx::SpectralDistribution::Dielectric;
  result.int_ior.eta_index = SpectrumAirEta;
  set_basic_thinfilm(result, min_thickness, max_thickness);
  return result;
}

etx::Material make_thinfilm_delta_dielectric() {
  etx::Material result = make_white_dielectric(0.0f);
  set_basic_thinfilm(result, 500.0f, 500.0f);
  return result;
}

etx::Material make_thinfilm_delta_conductor() {
  etx::Material result = make_mirror_conductor(0.0f);
  set_basic_thinfilm(result, 500.0f, 500.0f);
  return result;
}

etx::Material make_thinfilm_delta_plastic() {
  etx::Material result = make_plastic(0.0f);
  set_basic_thinfilm(result, 500.0f, 500.0f);
  return result;
}

etx::Material make_thinfilm_rough_conductor(const float roughness) {
  etx::Material result = make_mirror_conductor(roughness);
  set_basic_thinfilm(result, 500.0f, 500.0f);
  return result;
}

etx::Material make_thinfilm_rough_dielectric(const float roughness) {
  etx::Material result = make_white_sapphire_dielectric(roughness);
  set_basic_thinfilm(result, 500.0f, 500.0f);
  return result;
}

etx::Material make_thinfilm_rough_plastic(const float roughness) {
  etx::Material result = make_plastic(roughness);
  set_basic_thinfilm(result, 500.0f, 500.0f);
  return result;
}

bool validate_energy_compensated_material(const char* label, const etx::BSDFData& data, const etx::Material& material, const float roughness, const uint32_t seed) {
  const float pdf_integral = integrate_pdf(data, material, seed);
  const float reverse_pdf_integral = integrate_reverse_pdf(data, material, seed + 50000u);
  const bool narrow_lobe = roughness <= 0.1f;

  if ((std::isfinite(pdf_integral) == false) || (pdf_integral < 0.0f) || (std::isfinite(reverse_pdf_integral) == false) || (reverse_pdf_integral < 0.0f)) {
    std::printf("%s roughness %.3f pdf %.6f reverse %.6f\n", label, roughness, pdf_integral, reverse_pdf_integral);
    return false;
  }

  if ((narrow_lobe == false) && (fabsf(pdf_integral - 1.0f) > 0.12f)) {
    std::printf("%s roughness %.3f pdf %.6f reverse %.6f\n", label, roughness, pdf_integral, reverse_pdf_integral);
    return false;
  }

  if (validate_sampling(data, material, seed + 10000u) == false) {
    std::printf("%s roughness %.3f produced invalid sample\n", label, roughness);
    return false;
  }

  if (validate_sample_pdf_match(data, material, seed + 15000u) == false) {
    std::printf("%s roughness %.3f sample pdf mismatch\n", label, roughness);
    return false;
  }

  if (validate_reverse_pdf_match(data, material, seed + 17000u) == false) {
    std::printf("%s roughness %.3f reverse pdf mismatch\n", label, roughness);
    return false;
  }

  const SampleWeightStats stats = sample_weight_stats_rgb(data, material, seed + 20000u);
  if ((stats.invalid_count > 0u) || (stats.valid_count == 0u) || (std::isfinite(stats.average.x) == false) || (std::isfinite(stats.max_weight) == false)) {
    std::printf("%s roughness %.3f invalid %u valid %u max_weight %.6f\n", label, roughness, stats.invalid_count, stats.valid_count, stats.max_weight);
    return false;
  }

  const float bsdf_energy = integrate_bsdf_energy(data, material, seed + 30000u);
  if ((std::isfinite(bsdf_energy) == false) || (bsdf_energy < 0.0f)) {
    std::printf("%s roughness %.3f bsdf energy %.6f\n", label, roughness, bsdf_energy);
    return false;
  }

  std::printf("%s roughness %.3f pdf %.6f reverse %.6f avg %.6f %.6f %.6f p95 %.6f p99 %.6f max %.6f valid %u zero_pdf %u bsdf_energy %.6f\n", label, roughness, pdf_integral,
    reverse_pdf_integral, stats.average.x, stats.average.y, stats.average.z, stats.p95_weight, stats.p99_weight, stats.max_weight, stats.valid_count, stats.zero_pdf_count,
    bsdf_energy);
  return true;
}

bool validate_energy_compensated_white_furnace_direction(const char* label, const float3& w_i_world, const etx::Material& material, const float roughness, const uint32_t seed) {
  const Vertex contract_vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, contract_vertex, w_i_world};
  const SampleWeightStats stats = sample_weight_stats_rgb(data, material, seed);
  const float3 average_rgb = stats.average;
  const float max_rgb_error = max(fabsf(average_rgb.x - 1.0f), max(fabsf(average_rgb.y - 1.0f), fabsf(average_rgb.z - 1.0f)));

  if ((stats.invalid_count > 0u) || (stats.valid_count == 0u) || (std::isfinite(average_rgb.x) == false) || (std::isfinite(average_rgb.y) == false) ||
      (std::isfinite(average_rgb.z) == false) || (max_rgb_error > 0.12f)) {
    std::printf("%s furnace wi %.3f %.3f %.3f roughness %.3f rgb %.6f %.6f %.6f p95 %.6f p99 %.6f max %.6f valid %u zero_pdf %u invalid %u\n", label, w_i_world.x, w_i_world.y,
      w_i_world.z, roughness, average_rgb.x, average_rgb.y, average_rgb.z, stats.p95_weight, stats.p99_weight, stats.max_weight, stats.valid_count, stats.zero_pdf_count,
      stats.invalid_count);
    return false;
  }

  std::printf("%s furnace wi %.3f %.3f %.3f roughness %.3f rgb %.6f %.6f %.6f p95 %.6f p99 %.6f max %.6f valid %u zero_pdf %u invalid %u\n", label, w_i_world.x, w_i_world.y,
    w_i_world.z, roughness, average_rgb.x, average_rgb.y, average_rgb.z, stats.p95_weight, stats.p99_weight, stats.max_weight, stats.valid_count, stats.zero_pdf_count,
    stats.invalid_count);
  return true;
}

bool validate_eon_diffuse_material(const etx::BSDFData& data, const float roughness, const uint32_t seed) {
  const etx::Material material = make_diffuse(roughness);
  bool diagnostic_valid = validate_energy_compensated_material("eon diffuse", data, material, roughness, seed);
  diagnostic_valid = validate_energy_compensated_white_furnace_direction("eon diffuse", float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 100u) && diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction("eon diffuse", normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness, seed + 200u) && diagnostic_valid;
  return diagnostic_valid;
}

bool validate_velvet_material(const etx::BSDFData& data, const float roughness, const uint32_t seed) {
  const etx::Material material = make_velvet(roughness);
  bool diagnostic_valid = validate_energy_compensated_material("velvet", data, material, roughness, seed);
  diagnostic_valid = validate_energy_compensated_white_furnace_direction("velvet", float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 100u) && diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction("velvet", normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness, seed + 200u) && diagnostic_valid;
  return diagnostic_valid;
}

bool validate_balanced_translucent_material(const etx::BSDFData& data, const float roughness, const uint32_t seed) {
  const etx::Material material = make_balanced_translucent(roughness);
  bool diagnostic_valid = validate_energy_compensated_material("balanced translucent", data, material, roughness, seed);
  diagnostic_valid = validate_energy_compensated_white_furnace_direction("balanced translucent", float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 100u) && diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction("balanced translucent", normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness, seed + 200u) && diagnostic_valid;
  return diagnostic_valid;
}

bool validate_plastic_sample_contract(const char* label, const etx::BSDFData& data, const etx::Material& material, const float roughness, const uint32_t seed) {
  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;

  bool saw_coating = false;
  bool saw_substrate = false;
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 47u + 31u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false)) {
      std::printf("%s roughness %.3f invalid plastic sample\n", label, roughness);
      return false;
    }

    const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
    const bool diffuse = (sample.properties & etx::BSDFSample::Diffuse) != 0u;
    const bool transmission = (sample.properties & etx::BSDFSample::Transmission) != 0u;
    const bool medium_changed = (sample.properties & etx::BSDFSample::MediumChanged) != 0u;
    if ((((reflection == false) || transmission) || medium_changed) || ((sample.medium_index != data.current_medium) || (fabsf(sample.eta - 1.0f) > 1.0e-4f))) {
      std::printf("%s roughness %.3f invalid plastic metadata eta %.6f medium %u\n", label, roughness, sample.eta, sample.medium_index);
      return false;
    }

    if (diffuse) {
      saw_substrate = true;
    } else {
      saw_coating = true;
    }

    etx::Sampler eval_sampler(seed + i + 10000u, seed ^ (i * 53u + 37u));
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, sample.w_o, material_with_media, eval_sampler);
    if ((eval.valid() == false) || (finite_response(eval.bsdf) == false)) {
      std::printf("%s roughness %.3f invalid plastic eval at sampled direction\n", label, roughness);
      return false;
    }

    etx::Sampler pdf_sampler(seed + i + 20000u, seed ^ (i * 59u + 41u));
    const float pdf = etx::bsdf::pdf(data, sample.w_o, material_with_media, pdf_sampler);
    if ((std::isfinite(pdf) == false) || (pdf <= kEpsilon)) {
      std::printf("%s roughness %.3f invalid plastic pdf %.6f\n", label, roughness, pdf);
      return false;
    }

    const float pdf_tolerance = max(1.0e-3f, 0.05f * max(sample.pdf, pdf));
    if (fabsf(sample.pdf - pdf) > pdf_tolerance) {
      std::printf("%s roughness %.3f plastic sample/pdf mismatch sample %.6f pdf %.6f\n", label, roughness, sample.pdf, pdf);
      return false;
    }

    const etx::SpectralResponse expected_weight = eval.bsdf / pdf;
    const float weight_error = fabsf(sample.weight.monochromatic() - expected_weight.monochromatic());
    const float weight_tolerance = max(1.0e-3f, 0.05f * max(sample.weight.monochromatic(), expected_weight.monochromatic()));
    if (weight_error > weight_tolerance) {
      std::printf("%s roughness %.3f plastic weight mismatch sample %.6f expected %.6f\n", label, roughness, sample.weight.monochromatic(), expected_weight.monochromatic());
      return false;
    }
  }

  if ((saw_coating == false) || (saw_substrate == false)) {
    std::printf("%s roughness %.3f missing plastic branch coating %u substrate %u\n", label, roughness, saw_coating ? 1u : 0u, saw_substrate ? 1u : 0u);
    return false;
  }

  std::printf("%s roughness %.3f plastic sample contract valid\n", label, roughness);
  return true;
}

bool validate_plastic_inner_matches_outer(const char* label, const etx::BSDFData& outside_data, const etx::BSDFData& inside_data, const etx::Material& material,
  const float roughness, const uint32_t seed) {
  const float3 outside_w_o = normalize(float3{0.25f, 0.1f, 0.963068f});
  const float3 inside_w_o = -outside_w_o;
  etx::Sampler outside_eval_sampler(seed, seed ^ 0x75e1a9cu);
  const etx::BSDFEval outside_eval = etx::bsdf::evaluate(outside_data, outside_w_o, material, outside_eval_sampler);
  etx::Sampler inside_eval_sampler(seed + 1u, seed ^ 0x9830cf1u);
  const etx::BSDFEval inside_eval = etx::bsdf::evaluate(inside_data, inside_w_o, material, inside_eval_sampler);
  if ((outside_eval.valid() == false) || (inside_eval.valid() == false)) {
    std::printf("%s roughness %.3f invalid inner/outer plastic eval\n", label, roughness);
    return false;
  }

  const float bsdf_error = fabsf(outside_eval.bsdf.monochromatic() - inside_eval.bsdf.monochromatic());
  const float bsdf_tolerance = max(1.0e-4f, 5.0e-3f * max(outside_eval.bsdf.monochromatic(), inside_eval.bsdf.monochromatic()));
  if (bsdf_error > bsdf_tolerance) {
    std::printf("%s roughness %.3f inner/outer bsdf mismatch outside %.6f inside %.6f\n", label, roughness, outside_eval.bsdf.monochromatic(), inside_eval.bsdf.monochromatic());
    return false;
  }

  etx::Sampler outside_pdf_sampler(seed + 2u, seed ^ 0x16bca45u);
  const float outside_pdf = etx::bsdf::pdf(outside_data, outside_w_o, material, outside_pdf_sampler);
  etx::Sampler inside_pdf_sampler(seed + 3u, seed ^ 0x85cf034u);
  const float inside_pdf = etx::bsdf::pdf(inside_data, inside_w_o, material, inside_pdf_sampler);
  const float pdf_tolerance = max(1.0e-4f, 5.0e-3f * max(outside_pdf, inside_pdf));
  if (fabsf(outside_pdf - inside_pdf) > pdf_tolerance) {
    std::printf("%s roughness %.3f inner/outer pdf mismatch outside %.6f inside %.6f\n", label, roughness, outside_pdf, inside_pdf);
    return false;
  }

  if (validate_plastic_sample_contract(label, inside_data, material, roughness, seed + 1000u) == false) {
    return false;
  }

  std::printf("%s roughness %.3f inner side matches outer side\n", label, roughness);
  return true;
}

bool validate_plastic_black_substrate_matches_dielectric_reflection(const char* label, const etx::BSDFData& data, const etx::Material& plastic, const etx::Material& dielectric,
  const float roughness, const uint32_t seed) {
  const float3 outgoing_direction = normalize(float3{0.0f, 0.0f, 1.0f});
  etx::Sampler plastic_eval_sampler(seed, seed ^ 0x7a53d21u);
  const etx::BSDFEval plastic_eval = etx::bsdf::evaluate(data, outgoing_direction, plastic, plastic_eval_sampler);
  etx::Sampler dielectric_eval_sampler(seed + 1u, seed ^ 0x3e19f5bu);
  const etx::BSDFEval dielectric_eval = etx::bsdf::evaluate(data, outgoing_direction, dielectric, dielectric_eval_sampler);
  if ((plastic_eval.valid() == false) || (dielectric_eval.valid() == false)) {
    std::printf("%s roughness %.3f invalid coating comparison eval\n", label, roughness);
    return false;
  }

  const float bsdf_error = fabsf(plastic_eval.bsdf.monochromatic() - dielectric_eval.bsdf.monochromatic());
  const float bsdf_tolerance = max(1.0e-4f, 5.0e-3f * max(plastic_eval.bsdf.monochromatic(), dielectric_eval.bsdf.monochromatic()));
  if (bsdf_error > bsdf_tolerance) {
    std::printf("%s roughness %.3f coating bsdf %.6f dielectric %.6f\n", label, roughness, plastic_eval.bsdf.monochromatic(), dielectric_eval.bsdf.monochromatic());
    return false;
  }

  std::printf("%s roughness %.3f coating response matches dielectric reflection\n", label, roughness);
  return true;
}

bool validate_plastic_low_roughness_substrate_exit(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count) {
  const float roughness = 0.0f;
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(make_colored_substrate_black_coating_plastic(roughness));

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("plastic low roughness substrate exit failed to bind LUT\n");
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const etx::Material material = scene_data.materials[0];
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};
  const float external_mu_values[] = {0.8f, 0.5f, 0.25f, 0.05f};
  bool diagnostic_valid = true;
  for (uint32_t i = 0u; i < 4u; ++i) {
    const float mu = external_mu_values[i];
    const float sin_theta = sqrtf(max(0.0f, 1.0f - mu * mu));
    const float3 outgoing_direction = normalize(float3{sin_theta, 0.0f, mu});
    etx::Sampler eval_sampler(43000u + i, 0x9e3779b9u ^ i);
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, outgoing_direction, material, eval_sampler);
    const float3 rgb = eval.bsdf.to_rgb();
    const bool substrate_visible = (eval.valid() && (std::isfinite(rgb.x)) && (rgb.x > 1.0e-3f));
    if (substrate_visible == false) {
      std::printf("plastic low roughness substrate exit mu %.3f failed rgb %.6f %.6f %.6f pdf %.6f\n", mu, rgb.x, rgb.y, rgb.z, eval.pdf);
      diagnostic_valid = false;
    } else {
      std::printf("plastic low roughness substrate exit mu %.3f rgb %.6f %.6f %.6f pdf %.6f\n", mu, rgb.x, rgb.y, rgb.z, eval.pdf);
    }
  }

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

bool validate_standalone_thinfilm_contract(const char* label, const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;
  const uint32_t expected_transmission_medium = data.current_medium;

  bool saw_reflection = false;
  bool saw_transmission = false;
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 67u + 43u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false) || (sample.is_delta() == false)) {
      std::printf("%s invalid standalone thinfilm sample\n", label);
      return false;
    }

    const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
    const bool transmission = (sample.properties & etx::BSDFSample::Transmission) != 0u;
    const bool medium_changed = (sample.properties & etx::BSDFSample::MediumChanged) != 0u;
    if (reflection) {
      saw_reflection = true;
      if ((transmission || medium_changed) || ((sample.medium_index != data.current_medium) || (fabsf(sample.eta - 1.0f) > kEpsilon))) {
        std::printf("%s invalid standalone thinfilm reflection metadata\n", label);
        return false;
      }
    } else if (transmission) {
      saw_transmission = true;
      if ((medium_changed || (sample.medium_index != expected_transmission_medium)) || (fabsf(sample.eta - 1.0f) > kEpsilon)) {
        std::printf("%s invalid standalone thinfilm transmission metadata medium %u expected %u\n", label, sample.medium_index, expected_transmission_medium);
        return false;
      }
    } else {
      std::printf("%s standalone thinfilm sample missing branch flags\n", label);
      return false;
    }

    etx::Sampler eval_sampler(seed + i + 10000u, seed ^ (i * 71u + 47u));
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, sample.w_o, material_with_media, eval_sampler);
    etx::Sampler pdf_sampler(seed + i + 20000u, seed ^ (i * 73u + 53u));
    const float pdf = etx::bsdf::pdf(data, sample.w_o, material_with_media, pdf_sampler);
    if ((eval.valid() == false) || (std::isfinite(pdf) == false) || (pdf <= kEpsilon)) {
      std::printf("%s invalid standalone thinfilm eval/pdf %.6f\n", label, pdf);
      return false;
    }

    const float pdf_tolerance = max(1.0e-4f, 5.0e-3f * max(sample.pdf, pdf));
    if (fabsf(sample.pdf - pdf) > pdf_tolerance) {
      std::printf("%s standalone thinfilm pdf mismatch sample %.6f pdf %.6f\n", label, sample.pdf, pdf);
      return false;
    }

    const etx::SpectralResponse expected_weight = eval.bsdf / pdf;
    const float weight_error = fabsf(sample.weight.monochromatic() - expected_weight.monochromatic());
    const float weight_tolerance = max(1.0e-4f, 5.0e-3f * max(sample.weight.monochromatic(), expected_weight.monochromatic()));
    if (weight_error > weight_tolerance) {
      std::printf("%s standalone thinfilm weight mismatch sample %.6f expected %.6f\n", label, sample.weight.monochromatic(), expected_weight.monochromatic());
      return false;
    }
  }

  if ((saw_reflection == false) || (saw_transmission == false)) {
    std::printf("%s standalone thinfilm missing branch reflection %u transmission %u\n", label, saw_reflection ? 1u : 0u, saw_transmission ? 1u : 0u);
    return false;
  }

  std::printf("%s standalone thinfilm contract valid\n", label);
  return true;
}

bool validate_standalone_thinfilm_sheet_symmetry(const char* label, const etx::BSDFData& base_data, const etx::Material& material, const uint32_t seed) {
  etx::Material material_with_boundary_ior = material;
  material_with_boundary_ior.ext_medium = 7u;
  material_with_boundary_ior.int_medium = 11u;
  material_with_boundary_ior.int_ior.cls = etx::SpectralDistribution::Dielectric;
  material_with_boundary_ior.int_ior.eta_index = SpectrumDielectricEta;
  material_with_boundary_ior.int_ior.k_index = SpectrumBlack;

  const float mu_values[] = {1.0f, 0.5f, 0.2f, 0.05f};
  for (uint32_t i = 0u; i < 4u; ++i) {
    const float mu = mu_values[i];
    const float sin_theta = sqrtf(max(0.0f, 1.0f - mu * mu));
    etx::BSDFData outside_data = base_data;
    outside_data.w_i = normalize(float3{sin_theta, 0.0f, -mu});
    etx::BSDFData inside_data = base_data;
    inside_data.w_i = normalize(float3{sin_theta, 0.0f, mu});

    const float3 outside_reflection = normalize(reflect(outside_data.w_i, outside_data.nrm));
    const float3 inside_reflection = normalize(reflect(inside_data.w_i, -inside_data.nrm));

    etx::Sampler outside_sampler(seed + i, seed ^ (i * 101u + 73u));
    const etx::BSDFEval outside_eval = etx::bsdf::evaluate(outside_data, outside_reflection, material_with_boundary_ior, outside_sampler);
    etx::Sampler inside_sampler(seed + i + 100u, seed ^ (i * 103u + 79u));
    const etx::BSDFEval inside_eval = etx::bsdf::evaluate(inside_data, inside_reflection, material_with_boundary_ior, inside_sampler);
    if ((outside_eval.valid() == false) || (inside_eval.valid() == false)) {
      std::printf("%s invalid symmetric thinfilm reflection mu %.3f\n", label, mu);
      return false;
    }

    const float outside_value = outside_eval.bsdf.monochromatic();
    const float inside_value = inside_eval.bsdf.monochromatic();
    const float value_tolerance = max(1.0e-4f, 1.0e-3f * max(outside_value, inside_value));
    if (fabsf(outside_value - inside_value) > value_tolerance) {
      std::printf("%s asymmetric thinfilm reflection mu %.3f outside %.6f inside %.6f\n", label, mu, outside_value, inside_value);
      return false;
    }

    etx::Sampler outside_transmission_sampler(seed + i + 200u, seed ^ (i * 107u + 83u));
    const etx::BSDFEval outside_transmission = etx::bsdf::evaluate(outside_data, outside_data.w_i, material_with_boundary_ior, outside_transmission_sampler);
    etx::Sampler inside_transmission_sampler(seed + i + 300u, seed ^ (i * 109u + 89u));
    const etx::BSDFEval inside_transmission = etx::bsdf::evaluate(inside_data, inside_data.w_i, material_with_boundary_ior, inside_transmission_sampler);
    if ((outside_transmission.valid() == false) || (inside_transmission.valid() == false)) {
      std::printf("%s invalid symmetric thinfilm transmission mu %.3f\n", label, mu);
      return false;
    }

    const bool outside_medium_changed = (outside_transmission.properties & etx::BSDFSample::MediumChanged) != 0u;
    const bool inside_medium_changed = (inside_transmission.properties & etx::BSDFSample::MediumChanged) != 0u;
    if (((outside_medium_changed || inside_medium_changed) || (outside_transmission.medium_index != outside_data.current_medium)) ||
        (inside_transmission.medium_index != inside_data.current_medium)) {
      std::printf("%s standalone thinfilm transmission changed medium at mu %.3f\n", label, mu);
      return false;
    }

    const float outside_transmission_pdf = etx::bsdf::pdf(outside_data, outside_data.w_i, material_with_boundary_ior, outside_transmission_sampler);
    const float inside_transmission_pdf = etx::bsdf::pdf(inside_data, inside_data.w_i, material_with_boundary_ior, inside_transmission_sampler);
    if ((outside_transmission_pdf <= kEpsilon) || (inside_transmission_pdf <= kEpsilon)) {
      std::printf("%s standalone thinfilm artificial total reflection mu %.3f outside %.6f inside %.6f\n", label, mu, outside_transmission_pdf, inside_transmission_pdf);
      return false;
    }
  }

  std::printf("%s standalone thinfilm sheet symmetry valid\n", label);
  return true;
}

bool validate_delta_thinfilm_coating_sample(const char* label, const etx::BSDFData& data, const etx::Material& material, const uint32_t seed, bool require_transmission) {
  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;
  const bool entering = dot(data.nrm, data.w_i) < 0.0f;
  const uint32_t expected_transmission_medium = entering ? material_with_media.int_medium : material_with_media.ext_medium;

  bool saw_reflection = false;
  bool saw_transmission = false;
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 79u + 59u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false) || (sample.is_delta() == false)) {
      std::printf("%s invalid delta thinfilm coating sample\n", label);
      return false;
    }

    const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
    const bool transmission = (sample.properties & etx::BSDFSample::Transmission) != 0u;
    const bool medium_changed = (sample.properties & etx::BSDFSample::MediumChanged) != 0u;
    if (reflection) {
      saw_reflection = true;
      if ((transmission || medium_changed) || (sample.medium_index != data.current_medium)) {
        std::printf("%s invalid delta thinfilm reflection metadata\n", label);
        return false;
      }
    } else if (transmission) {
      saw_transmission = true;
      if (((medium_changed == false) || (sample.medium_index != expected_transmission_medium)) || (sample.eta <= 0.0f)) {
        std::printf("%s invalid delta thinfilm transmission metadata eta %.6f medium %u expected %u\n", label, sample.eta, sample.medium_index, expected_transmission_medium);
        return false;
      }
    } else {
      std::printf("%s delta thinfilm coating sample missing branch flags\n", label);
      return false;
    }
  }

  if ((saw_reflection == false) || (require_transmission && (saw_transmission == false))) {
    std::printf("%s delta thinfilm coating missing branch reflection %u transmission %u\n", label, saw_reflection ? 1u : 0u, saw_transmission ? 1u : 0u);
    return false;
  }

  std::printf("%s delta thinfilm coating sample valid\n", label);
  return true;
}

bool validate_delta_plastic_thinfilm_contract(const char* label, const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  bool saw_coating = false;
  bool saw_substrate = false;
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 83u + 61u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false)) {
      std::printf("%s invalid delta plastic thinfilm sample\n", label);
      return false;
    }

    const bool diffuse = (sample.properties & etx::BSDFSample::Diffuse) != 0u;
    const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
    const bool transmission = (sample.properties & etx::BSDFSample::Transmission) != 0u;
    const bool medium_changed = (sample.properties & etx::BSDFSample::MediumChanged) != 0u;
    if (((reflection == false) || transmission) || medium_changed) {
      std::printf("%s invalid delta plastic thinfilm metadata\n", label);
      return false;
    }
    if (diffuse) {
      saw_substrate = true;
    } else {
      saw_coating = true;
    }

    etx::Sampler eval_sampler(seed + i + 10000u, seed ^ (i * 89u + 67u));
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, sample.w_o, material, eval_sampler);
    etx::Sampler pdf_sampler(seed + i + 20000u, seed ^ (i * 97u + 71u));
    const float pdf = etx::bsdf::pdf(data, sample.w_o, material, pdf_sampler);
    if ((eval.valid() == false) || (std::isfinite(pdf) == false) || (pdf <= kEpsilon)) {
      std::printf("%s invalid delta plastic thinfilm eval/pdf %.6f\n", label, pdf);
      return false;
    }

    const float pdf_tolerance = max(1.0e-4f, 5.0e-3f * max(sample.pdf, pdf));
    if (fabsf(sample.pdf - pdf) > pdf_tolerance) {
      std::printf("%s delta plastic thinfilm pdf mismatch sample %.6f pdf %.6f\n", label, sample.pdf, pdf);
      return false;
    }
  }

  if ((saw_coating == false) || (saw_substrate == false)) {
    std::printf("%s delta plastic thinfilm missing branch coating %u substrate %u\n", label, saw_coating ? 1u : 0u, saw_substrate ? 1u : 0u);
    return false;
  }

  std::printf("%s delta plastic thinfilm contract valid\n", label);
  return true;
}

bool validate_exact_plastic_interface(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count, const float roughness,
  const uint32_t seed) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(make_plastic(roughness));
  scene_data.materials.emplace_back(make_black_substrate_plastic(roughness));
  scene_data.materials.emplace_back(make_white_dielectric(roughness));

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("plastic exact interface roughness %.3f failed to bind LUT\n", roughness);
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const etx::Material material = scene_data.materials[0];
  const etx::Material black_substrate_material = scene_data.materials[1];
  const etx::Material dielectric_material = scene_data.materials[2];
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData camera_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};
  const etx::BSDFData light_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Light, vertex, float3{0.0f, 0.0f, -1.0f}};
  const etx::BSDFData inside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, 1.0f}};

  bool diagnostic_valid = validate_energy_compensated_material("plastic coated diffuse", camera_data, material, roughness, seed);
  diagnostic_valid = validate_energy_compensated_material("plastic coated diffuse light", light_data, material, roughness, seed + 500u) && diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction("plastic coated diffuse", float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 1000u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_white_furnace_direction("plastic coated diffuse", normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness, seed + 1100u) &&
                     diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction("plastic coated diffuse grazing", normalize(float3{0.9848077f, 0.0f, -0.1736482f}), material, roughness, seed + 1150u) &&
    diagnostic_valid;
  diagnostic_valid = validate_plastic_sample_contract("plastic coated diffuse", camera_data, material, roughness, seed + 1250u) && diagnostic_valid;
  diagnostic_valid = validate_plastic_inner_matches_outer("plastic coated diffuse", camera_data, inside_data, material, roughness, seed + 1300u) && diagnostic_valid;
  diagnostic_valid =
    validate_plastic_black_substrate_matches_dielectric_reflection("plastic coated diffuse", camera_data, black_substrate_material, dielectric_material, roughness, seed + 1350u) &&
    diagnostic_valid;
  diagnostic_valid = validate_plastic_sample_contract("plastic black substrate", camera_data, black_substrate_material, roughness, seed + 1375u) && diagnostic_valid;

  const float bsdf_energy = integrate_bsdf_energy(camera_data, material, seed + 1200u);
  if ((std::isfinite(bsdf_energy) == false) || (bsdf_energy < 0.0f) || (bsdf_energy > 1.05f)) {
    std::printf("plastic coated diffuse roughness %.3f overcompensated bsdf energy %.6f\n", roughness, bsdf_energy);
    diagnostic_valid = false;
  } else {
    std::printf("plastic coated diffuse roughness %.3f bsdf energy %.6f\n", roughness, bsdf_energy);
  }

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

bool validate_thinfilm_plastic_interface(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count, const float roughness,
  const uint32_t seed) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(make_thinfilm_rough_plastic(roughness));

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("plastic thinfilm coated diffuse roughness %.3f failed to bind LUT\n", roughness);
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const etx::Material material = scene_data.materials[0];
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};

  bool diagnostic_valid = validate_energy_compensated_material("plastic thinfilm coated diffuse", data, material, roughness, seed);
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction("plastic thinfilm coated diffuse", float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 100u) && diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction("plastic thinfilm coated diffuse", normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness, seed + 200u) &&
    diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_white_furnace_direction("plastic thinfilm coated diffuse grazing", normalize(float3{0.9848077f, 0.0f, -0.1736482f}), material,
                       roughness, seed + 300u) &&
                     diagnostic_valid;
  diagnostic_valid = validate_plastic_sample_contract("plastic thinfilm coated diffuse", data, material, roughness, seed + 400u) && diagnostic_valid;

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

bool validate_thinfilm_energy_compensation_cache_key(const etx::SpectralDistribution* spectra, const uint32_t spectrum_count) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  etx::Material material_500 = make_thinfilm_rough_conductor(0.5f);
  etx::Material material_650 = material_500;
  set_basic_thinfilm(material_650, 650.0f, 650.0f);
  etx::Material material_half_weight = material_500;
  material_half_weight.thinfilm.weight = 0.5f;
  scene_data.materials.emplace_back(material_500);
  scene_data.materials.emplace_back(material_650);
  scene_data.materials.emplace_back(material_half_weight);

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("thinfilm EC cache-key validation failed to bind LUTs\n");
    return false;
  }

  const uint32_t interface_500 = scene_data.materials[0].energy_compensation_interface_index;
  const uint32_t interface_650 = scene_data.materials[1].energy_compensation_interface_index;
  const uint32_t interface_half_weight = scene_data.materials[2].energy_compensation_interface_index;
  if ((interface_500 == kInvalidIndex) || (interface_650 == kInvalidIndex) || (interface_half_weight == kInvalidIndex) || (interface_500 == interface_650) ||
      (interface_500 == interface_half_weight) || (interface_650 == interface_half_weight)) {
    std::printf("thinfilm EC cache-key validation failed interfaces %u %u %u\n", interface_500, interface_650, interface_half_weight);
    return false;
  }

  std::printf("thinfilm EC cache-key validation interfaces %u %u %u\n", interface_500, interface_650, interface_half_weight);
  return true;
}

bool validate_variable_thinfilm_texture_lut(const etx::SpectralDistribution* spectra, const uint32_t spectrum_count) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(32u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);

  const float4 thickness_pixels[] = {
    float4{0.0f, 0.0f, 0.0f, 1.0f},
    float4{0.5f, 0.5f, 0.5f, 1.0f},
    float4{1.0f, 1.0f, 1.0f, 1.0f},
  };
  const uint32_t thickness_image = scene_data.images.add_from_data(thickness_pixels, uint2{3u, 1u}, etx::Image::SkipSRGBConversion, {}, float2{1.0f, 1.0f});

  etx::Material variable_material = make_thinfilm_rough_conductor(0.5f);
  variable_material.thinfilm.min_thickness = 400.0f;
  variable_material.thinfilm.max_thickness = 700.0f;
  variable_material.thinfilm.thinkness_image = thickness_image;

  etx::Material constant_material = make_thinfilm_rough_conductor(0.5f);
  set_basic_thinfilm(constant_material, 550.0f, 550.0f);

  scene_data.materials.emplace_back(variable_material);
  scene_data.materials.emplace_back(constant_material);

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("variable thinfilm texture LUT validation failed to bind interfaces\n");
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  const BSDFResourceContext context = make_bsdf_resource_cpu_context(exact_scene);
  etx::Material absorbing_film_material = variable_material;
  absorbing_film_material.thinfilm.ior.k_index = SpectrumHalf;
  Sampler absorbing_film_sampler(56999u, 0x34a1u);
  const ThinfilmEval lossless_film = bsdf_resource_evaluate_thinfilm(context, etx::SpectralQuery{}, absorbing_film_material.thinfilm, float2{0.0f, 0.0f}, absorbing_film_sampler);
  if (spectral_response_is_zero(lossless_film.ior.k) == false) {
    std::printf("thinfilm runtime contract did not force extinction to zero\n");
    return false;
  }
  const etx::Material& bound_variable_material = scene_data.materials[0];
  const etx::Material& bound_constant_material = scene_data.materials[1];
  const uint32_t variable_interface_index = bound_variable_material.energy_compensation_interface_index;
  const uint32_t constant_interface_index = bound_constant_material.energy_compensation_interface_index;
  if ((variable_interface_index == kInvalidIndex) || (constant_interface_index == kInvalidIndex)) {
    std::printf("variable thinfilm texture LUT validation missing interfaces %u %u\n", variable_interface_index, constant_interface_index);
    return false;
  }

  const etx::Scene::EnergyCompensationInterface& variable_interface = scene_data.energy_compensation_interfaces[variable_interface_index];
  const etx::Scene::EnergyCompensationInterface& constant_interface = scene_data.energy_compensation_interfaces[constant_interface_index];
  const etx::Image& variable_directional_lut = scene_data.images.get(variable_interface.directional_lut);
  const etx::Image& variable_average_lut = scene_data.images.get(variable_interface.average_lut);
  const etx::Image& constant_directional_lut = scene_data.images.get(constant_interface.directional_lut);
  if ((variable_directional_lut.isize.z <= 1u) || (variable_average_lut.isize.z <= 1u)) {
    std::printf("variable thinfilm texture LUT validation expected 3D LUTs directional %u average %u\n", variable_directional_lut.isize.z, variable_average_lut.isize.z);
    return false;
  }
  if (constant_directional_lut.isize.z != 1u) {
    std::printf("constant thinfilm LUT validation expected depth 1, got %u\n", constant_directional_lut.isize.z);
    return false;
  }

  const float2 uvs[] = {
    float2{0.0f, 0.0f},
    float2{1.0f / 3.0f, 0.0f},
    float2{2.0f / 3.0f, 0.0f},
  };
  const float expected_thickness[] = {400.0f, 550.0f, 700.0f};
  const float expected_lut_value[] = {0.0f, 0.5f, 1.0f};
  for (uint32_t i = 0u; i < 3u; ++i) {
    Sampler sampler(57000u + i, 0x1234u + i);
    const ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, etx::SpectralQuery{}, bound_variable_material.thinfilm, uvs[i], sampler);
    const float lut_value = bsdf_energy_compensated_thinfilm_lut_value(bound_variable_material, thinfilm);
    if ((close_value(thinfilm.thickness, expected_thickness[i], 1.0e-4f) == false) || (close_value(lut_value, expected_lut_value[i], 1.0e-5f) == false)) {
      std::printf("variable thinfilm texture mapping failed index %u thickness %.6f lut %.6f\n", i, thinfilm.thickness, lut_value);
      return false;
    }
  }

  Sampler constant_sampler(58000u, 0x5678u);
  const ThinfilmEval constant_thinfilm = bsdf_resource_evaluate_thinfilm(context, etx::SpectralQuery{}, bound_constant_material.thinfilm, float2{0.0f, 0.0f}, constant_sampler);
  const float constant_lut_value = bsdf_energy_compensated_thinfilm_lut_value(bound_constant_material, constant_thinfilm);
  if ((close_value(constant_thinfilm.thickness, 550.0f, 1.0e-4f) == false) || (close_value(constant_lut_value, 0.0f, 1.0e-5f) == false)) {
    std::printf("constant thinfilm mapping failed thickness %.6f lut %.6f\n", constant_thinfilm.thickness, constant_lut_value);
    return false;
  }

  constexpr uint32_t kSyntheticLutSize = kBSDFEnergyCompensationConductorLutSize;
  constexpr uint32_t kSyntheticLutLayerPixels = kSyntheticLutSize * kSyntheticLutSize;
  constexpr uint32_t kSyntheticLutPixels = 2u * kSyntheticLutLayerPixels;
  float4 synthetic_3d_lut[kSyntheticLutPixels] = {};
  float4 synthetic_1d_lut[kSyntheticLutLayerPixels] = {};
  for (uint32_t i = 0u; i < kSyntheticLutLayerPixels; ++i) {
    synthetic_3d_lut[i] = float4{0.1f, 0.2f, 0.3f, 1.0f};
    synthetic_3d_lut[i + kSyntheticLutLayerPixels] = float4{0.7f, 0.6f, 0.5f, 1.0f};
    synthetic_1d_lut[i] = float4{0.4f, 0.5f, 0.6f, 1.0f};
  }

  const uint32_t synthetic_3d_lut_index =
    scene_data.images.add_from_data_3d(synthetic_3d_lut, uint3{kSyntheticLutSize, kSyntheticLutSize, 2u}, etx::Image::SkipSRGBConversion, {}, float3{1.0f, 1.0f, 1.0f});
  const uint32_t synthetic_1d_lut_index =
    scene_data.images.add_from_data_3d(synthetic_1d_lut, uint3{kSyntheticLutSize, kSyntheticLutSize, 1u}, etx::Image::SkipSRGBConversion, {}, float3{1.0f, 1.0f, 1.0f});

  etx::Scene::EnergyCompensationInterface synthetic_3d_interface = {};
  synthetic_3d_interface.cls = MaterialClass::Conductor;
  synthetic_3d_interface.directional_lut = synthetic_3d_lut_index;
  const uint32_t synthetic_3d_interface_index = scene_data.add_energy_compensation_interface(synthetic_3d_interface);

  etx::Scene::EnergyCompensationInterface synthetic_1d_interface = {};
  synthetic_1d_interface.cls = MaterialClass::Conductor;
  synthetic_1d_interface.directional_lut = synthetic_1d_lut_index;
  const uint32_t synthetic_1d_interface_index = scene_data.add_energy_compensation_interface(synthetic_1d_interface);

  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::Material synthetic_3d_material = bound_variable_material;
  synthetic_3d_material.energy_compensation_interface_index = synthetic_3d_interface_index;
  const SpectralResponse synthetic_low = bsdf_energy_compensated_conductor_directional_albedo(context, etx::SpectralQuery{}, synthetic_3d_material, 0.5f, 0.5f, 0.0f);
  const SpectralResponse synthetic_high = bsdf_energy_compensated_conductor_directional_albedo(context, etx::SpectralQuery{}, synthetic_3d_material, 0.5f, 0.5f, 1.0f);
  if ((close_value(synthetic_low.integrated.x, 0.1f, 1.0e-5f) == false) || (close_value(synthetic_low.integrated.y, 0.2f, 1.0e-5f) == false) ||
      (close_value(synthetic_low.integrated.z, 0.3f, 1.0e-5f) == false) || (close_value(synthetic_high.integrated.x, 0.7f, 1.0e-5f) == false) ||
      (close_value(synthetic_high.integrated.y, 0.6f, 1.0e-5f) == false) || (close_value(synthetic_high.integrated.z, 0.5f, 1.0e-5f) == false)) {
    std::printf("synthetic variable thinfilm LUT sampling failed low %.6f %.6f %.6f high %.6f %.6f %.6f\n", synthetic_low.integrated.x, synthetic_low.integrated.y,
      synthetic_low.integrated.z, synthetic_high.integrated.x, synthetic_high.integrated.y, synthetic_high.integrated.z);
    return false;
  }

  etx::Material synthetic_1d_material = bound_constant_material;
  synthetic_1d_material.energy_compensation_interface_index = synthetic_1d_interface_index;
  const SpectralResponse synthetic_constant_low = bsdf_energy_compensated_conductor_directional_albedo(context, etx::SpectralQuery{}, synthetic_1d_material, 0.5f, 0.5f, 0.0f);
  const SpectralResponse synthetic_constant_high = bsdf_energy_compensated_conductor_directional_albedo(context, etx::SpectralQuery{}, synthetic_1d_material, 0.5f, 0.5f, 1.0f);
  const float synthetic_constant_delta = length(synthetic_constant_low.integrated - synthetic_constant_high.integrated);
  if ((synthetic_constant_delta > 1.0e-6f) || (close_value(synthetic_constant_low.integrated.x, 0.4f, 1.0e-5f) == false) ||
      (close_value(synthetic_constant_low.integrated.y, 0.5f, 1.0e-5f) == false) || (close_value(synthetic_constant_low.integrated.z, 0.6f, 1.0e-5f) == false)) {
    std::printf("synthetic constant thinfilm LUT sampling failed low %.6f %.6f %.6f high %.6f %.6f %.6f\n", synthetic_constant_low.integrated.x,
      synthetic_constant_low.integrated.y, synthetic_constant_low.integrated.z, synthetic_constant_high.integrated.x, synthetic_constant_high.integrated.y,
      synthetic_constant_high.integrated.z);
    return false;
  }

  std::printf("variable thinfilm texture LUT validation valid depth %u synthetic delta %.6f\n", variable_directional_lut.isize.z,
    length(synthetic_low.integrated - synthetic_high.integrated));
  return true;
}

bool validate_energy_compensated_dielectric_transport_contract(const char* label, const etx::BSDFData& data, const etx::Material& material, const float roughness,
  const uint32_t seed) {
  if (validate_sampling(data, material, seed) == false) {
    std::printf("%s roughness %.3f produced invalid sample\n", label, roughness);
    return false;
  }

  if (validate_sample_pdf_match(data, material, seed + 1000u) == false) {
    std::printf("%s roughness %.3f sample pdf mismatch\n", label, roughness);
    return false;
  }

  if (validate_reverse_pdf_match(data, material, seed + 2000u) == false) {
    std::printf("%s roughness %.3f reverse pdf mismatch\n", label, roughness);
    return false;
  }

  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;
  const bool outside = (dot(data.nrm, data.w_i) < 0.0f);
  const uint32_t expected_transmission_medium = outside ? material_with_media.int_medium : material_with_media.ext_medium;
  bool saw_reflection = false;
  bool saw_transmission = false;
  bool saw_wavelength_dependent_transmission = false;
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i + 3000u, seed ^ (i * 37u + 23u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false)) {
      std::printf("%s roughness %.3f invalid media sample\n", label, roughness);
      return false;
    }

    const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
    const bool transmission = ((sample.properties & etx::BSDFSample::Transmission) != 0u);
    const bool medium_changed = ((sample.properties & etx::BSDFSample::MediumChanged) != 0u);
    const bool wavelength_dependent_direction = ((sample.properties & etx::BSDFSample::WavelengthDependentDirection) != 0u);
    if (reflection) {
      saw_reflection = true;
      if ((transmission) || medium_changed || wavelength_dependent_direction || (sample.medium_index != data.current_medium) || (fabsf(sample.eta - 1.0f) > 1.0e-4f)) {
        std::printf("%s roughness %.3f invalid reflection metadata eta %.6f medium %u\n", label, roughness, sample.eta, sample.medium_index);
        return false;
      }
    } else if (transmission) {
      saw_transmission = true;
      saw_wavelength_dependent_transmission = wavelength_dependent_direction || saw_wavelength_dependent_transmission;
      if ((medium_changed == false) || (sample.medium_index != expected_transmission_medium) || (sample.eta <= 0.0f)) {
        std::printf("%s roughness %.3f invalid transmission metadata eta %.6f medium %u expected %u\n", label, roughness, sample.eta, sample.medium_index,
          expected_transmission_medium);
        return false;
      }
    } else {
      std::printf("%s roughness %.3f missing reflection/transmission flag\n", label, roughness);
      return false;
    }
  }

  if ((saw_reflection == false) || (saw_transmission == false) || (saw_wavelength_dependent_transmission == false)) {
    std::printf("%s roughness %.3f missing sampled branch reflection %u transmission %u wavelength-dependent %u\n", label, roughness, saw_reflection ? 1u : 0u,
      saw_transmission ? 1u : 0u, saw_wavelength_dependent_transmission ? 1u : 0u);
    return false;
  }

  std::printf("%s roughness %.3f transport contract valid\n", label, roughness);
  return true;
}

bool validate_delta_dielectric_transmission_sample(const char* label, const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;

  const bool outside = (dot(data.nrm, data.w_i) < 0.0f);
  const uint32_t expected_medium = outside ? material_with_media.int_medium : material_with_media.ext_medium;

  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 43u + 29u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false)) {
      std::printf("%s invalid delta sample\n", label);
      return false;
    }

    const bool transmission = ((sample.properties & etx::BSDFSample::Transmission) != 0u);
    if (transmission == false) {
      continue;
    }

    const bool medium_changed = ((sample.properties & etx::BSDFSample::MediumChanged) != 0u);
    const bool wavelength_dependent_direction = ((sample.properties & etx::BSDFSample::WavelengthDependentDirection) != 0u);
    if ((medium_changed == false) || (wavelength_dependent_direction == false) || (sample.medium_index != expected_medium) || (sample.eta <= 0.0f)) {
      std::printf("%s invalid transmission metadata eta %.6f medium %u expected %u\n", label, sample.eta, sample.medium_index, expected_medium);
      return false;
    }

    const float expected_weight = data.path_source == etx::PathSource::Light ? 1.0f : sample.eta * sample.eta;
    const float weight = sample.weight.monochromatic();
    const float tolerance = max(1.0e-4f, 5.0e-4f * expected_weight);
    if (fabsf(weight - expected_weight) > tolerance) {
      std::printf("%s transmission weight %.6f expected %.6f eta %.6f\n", label, weight, expected_weight, sample.eta);
      return false;
    }

    std::printf("%s delta transmission weight %.6f eta %.6f valid\n", label, weight, sample.eta);
    return true;
  }

  std::printf("%s did not sample transmission branch\n", label);
  return false;
}

bool validate_named_plastic_water_exact_interface(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(make_named_plastic_water_dielectric(0.25f));

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("energy compensated named plastic-water exact interface diagnostics failed to bind LUT\n");
    return true;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const etx::Material material = scene_data.materials[0];
  const float roughness = 0.25f;
  bool diagnostic_valid = true;
  const float outside_mu_values[] = {1.0f, 0.5f, 0.1736482f, 0.05f, 0.01f};
  for (uint32_t i = 0u; i < 5u; ++i) {
    const float mu = outside_mu_values[i];
    const float sin_theta = sqrtf(max(0.0f, 1.0f - mu * mu));
    char outside_label[128] = {};
    snprintf(outside_label, sizeof(outside_label), "energy compensated named plastic-water exact outside mu %.3f", mu);
    diagnostic_valid =
      validate_energy_compensated_white_furnace_direction(outside_label, normalize(float3{sin_theta, 0.0f, -mu}), material, roughness, 34000u + i) && diagnostic_valid;

    char inside_label[128] = {};
    snprintf(inside_label, sizeof(inside_label), "energy compensated named plastic-water exact inside mu %.3f", mu);
    diagnostic_valid =
      validate_energy_compensated_white_furnace_direction(inside_label, normalize(float3{sin_theta, 0.0f, mu}), material, roughness, 34100u + i) && diagnostic_valid;
  }

  const Vertex contract_vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData camera_outside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, contract_vertex, normalize(float3{0.5f, 0.0f, -0.8660254f})};
  const etx::BSDFData light_outside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Light, contract_vertex, normalize(float3{0.5f, 0.0f, -0.8660254f})};
  const etx::BSDFData camera_inside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, contract_vertex, normalize(float3{0.5f, 0.0f, 0.8660254f})};
  const etx::BSDFData light_inside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Light, contract_vertex, normalize(float3{0.5f, 0.0f, 0.8660254f})};
  diagnostic_valid =
    validate_energy_compensated_dielectric_transport_contract("energy compensated named plastic-water exact camera outside", camera_outside_data, material, roughness, 34200u) &&
    diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_dielectric_transport_contract("energy compensated named plastic-water exact light outside", light_outside_data, material, roughness, 34300u) &&
    diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_dielectric_transport_contract("energy compensated named plastic-water exact camera inside", camera_inside_data, material, roughness, 34400u) &&
    diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_dielectric_transport_contract("energy compensated named plastic-water exact light inside", light_inside_data, material, roughness, 34500u) &&
    diagnostic_valid;

  if (diagnostic_valid == false) {
    std::printf("energy compensated named plastic-water exact interface diagnostics detected non-unit furnace energy\n");
  }

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return true;
}

bool validate_exact_energy_compensated_dielectric_interface(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count, const char* label,
  const etx::Material& source_material, const float roughness, const uint32_t seed) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(source_material);

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("%s exact dielectric interface failed to bind LUT\n", label);
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const etx::Material material = scene_data.materials[0];
  bool diagnostic_valid = true;
  diagnostic_valid = validate_energy_compensated_white_furnace_direction(label, float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 100u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_white_furnace_direction(label, float3{0.0f, 0.0f, 1.0f}, material, roughness, seed + 200u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_white_furnace_direction(label, normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness, seed + 300u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_white_furnace_direction(label, normalize(float3{0.8660254f, 0.0f, 0.5f}), material, roughness, seed + 400u) && diagnostic_valid;

  const Vertex contract_vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData camera_outside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, contract_vertex, normalize(float3{0.5f, 0.0f, -0.8660254f})};
  const etx::BSDFData light_outside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Light, contract_vertex, normalize(float3{0.5f, 0.0f, -0.8660254f})};
  const etx::BSDFData camera_inside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, contract_vertex, normalize(float3{0.5f, 0.0f, 0.8660254f})};
  const etx::BSDFData light_inside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Light, contract_vertex, normalize(float3{0.5f, 0.0f, 0.8660254f})};
  const float camera_outside_energy = integrate_bsdf_energy(camera_outside_data, material, seed + 900u);
  const float light_outside_energy = integrate_bsdf_energy(light_outside_data, material, seed + 1000u);
  const float camera_inside_energy = integrate_bsdf_energy(camera_inside_data, material, seed + 1100u);
  const float light_inside_energy = integrate_bsdf_energy(light_inside_data, material, seed + 1200u);
  std::printf("%s roughness %.3f eval energy camera outside %.6f light outside %.6f camera inside %.6f light inside %.6f\n", label, roughness, camera_outside_energy,
    light_outside_energy, camera_inside_energy, light_inside_energy);
  diagnostic_valid = validate_energy_compensated_dielectric_transport_contract(label, camera_outside_data, material, roughness, seed + 500u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_dielectric_transport_contract(label, light_outside_data, material, roughness, seed + 600u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_dielectric_transport_contract(label, camera_inside_data, material, roughness, seed + 700u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_dielectric_transport_contract(label, light_inside_data, material, roughness, seed + 800u) && diagnostic_valid;

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

bool validate_exact_energy_compensated_conductor_interface(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count, const char* label,
  const etx::Material& source_material, const float roughness, const uint32_t seed) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(source_material);

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("%s exact interface failed to bind LUT\n", label);
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const etx::Material material = scene_data.materials[0];
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};
  const BSDFResourceContext context = etx::bsdf::detail::make_interop_context();
  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  Sampler thinfilm_sampler(seed + 17u, seed ^ 0x4d31a2bu);
  const ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, thinfilm_sampler);
  const SpectralResponse e_i_response = bsdf_energy_compensated_conductor_directional_albedo(context, data.spectrum_sample, material, 1.0f, alpha);
  const SpectralResponse e_average_response = bsdf_energy_compensated_conductor_average_albedo(context, data.spectrum_sample, material, alpha);
  const float e_i_scalar = bsdf_energy_compensated_conductor_geometric_directional_albedo(context, material, 1.0f, alpha);
  const float e_average_scalar = bsdf_energy_compensated_conductor_geometric_average_albedo(context, material, alpha);
  const float visible_probability = bsdf_energy_compensated_conductor_visible_probability(context, material, 1.0f, alpha);
  const SpectralResponse f_ms = bsdf_energy_compensated_conductor_fms(data.spectrum_sample, ext_ior, int_ior, thinfilm, e_average_scalar);
  std::printf("%s exact interface roughness %.3f e_i %.6f e_avg %.6f geom_i %.6f geom_avg %.6f visible %.6f f_ms %.6f\n", label, roughness,
    spectral_response_monochromatic(e_i_response), spectral_response_monochromatic(e_average_response), e_i_scalar, e_average_scalar, visible_probability,
    spectral_response_monochromatic(f_ms));

  bool diagnostic_valid = validate_energy_compensated_material(label, data, material, roughness, seed);
  diagnostic_valid = validate_energy_compensated_white_furnace_direction(label, float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 100u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_white_furnace_direction(label, normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness, seed + 200u) && diagnostic_valid;

  const float bsdf_energy = integrate_bsdf_energy(data, material, seed + 300u);
  if ((std::isfinite(bsdf_energy) == false) || (bsdf_energy < 0.0f) || (bsdf_energy > 1.05f)) {
    std::printf("%s exact interface roughness %.3f overcompensated bsdf energy %.6f\n", label, roughness, bsdf_energy);
    diagnostic_valid = false;
  } else {
    std::printf("%s exact interface roughness %.3f bsdf energy %.6f\n", label, roughness, bsdf_energy);
  }

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

void set_openpbr_validation_defaults(etx::SceneData& scene_data) {
  scene_data.defaults.white_spectrum = SpectrumWhite;
  scene_data.defaults.dielectric_eta = SpectrumDielectricEta;
  scene_data.defaults.conductor_eta = SpectrumMirrorEta;
  scene_data.defaults.conductor_k = SpectrumMirrorK;
}

void set_openpbr_validation_defaults(etx::Scene& scene) {
  scene.defaults.white_spectrum = SpectrumWhite;
  scene.defaults.dielectric_eta = SpectrumDielectricEta;
  scene.defaults.conductor_eta = SpectrumMirrorEta;
  scene.defaults.conductor_k = SpectrumMirrorK;
}

bool validate_openpbr_white_furnace(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count, const float roughness,
  const uint32_t seed) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  set_openpbr_validation_defaults(scene_data);
  scene_data.materials.emplace_back(make_plastic(roughness));
  scene_data.materials[0].cls = MaterialClass::OpenPBR;
  scene_data.materials[0].metalness.value = {0.0f, 0.0f, 0.0f, 0.0f};
  scene_data.materials[0].transmission.value = {0.0f, 0.0f, 0.0f, 0.0f};

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("openpbr white furnace failed to bind LUT\n");
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene openpbr_scene = {};
  openpbr_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  openpbr_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  openpbr_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  set_openpbr_validation_defaults(openpbr_scene);
  openpbr_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&openpbr_scene, &openpbr_scene);

  const etx::Material material = scene_data.materials[0];
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};
  bool diagnostic_valid = validate_energy_compensated_material("openpbr coated base", data, material, roughness, seed);
  diagnostic_valid = validate_energy_compensated_white_furnace_direction("openpbr coated base", float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 100u) && diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction("openpbr coated base", normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness, seed + 200u) && diagnostic_valid;

  etx::scene_global_clear(&openpbr_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

etx::Material make_openpbr(const float roughness, const float metalness, const float transmission, const uint32_t base_spectrum, const bool thinfilm) {
  etx::Material result = make_plastic(roughness);
  result.cls = MaterialClass::OpenPBR;
  result.scattering.spectrum_index = base_spectrum;
  result.reflectance.spectrum_index = SpectrumWhite;
  result.metalness.value = {metalness, metalness, metalness, metalness};
  result.transmission.value = {transmission, transmission, transmission, transmission};
  if (thinfilm) {
    set_basic_thinfilm(result, 500.0f, 500.0f);
  }
  return result;
}

struct BSDFRuntimeValidationCase {
  uint32_t material_index = 0u;
  uint32_t seed = 0u;
  float fixed_u = 0.0f;
  float fixed_v = 0.0f;
  float fixed_w = 0.0f;
  uint32_t pad0 = 0u;
  uint32_t pad1 = 0u;
  uint32_t pad2 = 0u;
};

struct BSDFRuntimeValidationExpected {
  etx::BSDFSample sample = {};
  etx::BSDFEval eval = {};
  float pdf = 0.0f;
  float reverse_pdf = 0.0f;
  uint32_t is_delta = 0u;
  etx::SpectralResponse albedo = {};
  uint32_t sample_seed = 0u;
  uint32_t eval_seed = 0u;
  uint32_t pdf_seed = 0u;
  uint32_t reverse_pdf_seed = 0u;
  uint32_t delta_seed = 0u;
  uint32_t albedo_seed = 0u;
};

constexpr uint32_t kBSDFRuntimeValidationOutputStride = 148u;

uint64_t validation_align_up_u64(const uint64_t value, const uint64_t alignment) {
  const uint64_t a = (alignment == 0u) ? 1u : alignment;
  return ((value + a - 1u) / a) * a;
}

uint32_t validation_append_aligned_bytes(std::vector<uint8_t>& blob, const void* data, const uint64_t byte_size, const uint64_t alignment) {
  if ((data == nullptr) || (byte_size == 0u)) {
    return kInvalidIndex;
  }

  const uint64_t aligned_offset = validation_align_up_u64(static_cast<uint64_t>(blob.size()), alignment);
  if (aligned_offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return kInvalidIndex;
  }
  if (aligned_offset > static_cast<uint64_t>(blob.size())) {
    blob.resize(static_cast<size_t>(aligned_offset), 0u);
  }

  if (aligned_offset > (std::numeric_limits<uint64_t>::max() - byte_size)) {
    return kInvalidIndex;
  }

  const uint64_t end_offset = aligned_offset + byte_size;
  if (end_offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return kInvalidIndex;
  }

  const size_t old_size = blob.size();
  blob.resize(static_cast<size_t>(end_offset), 0u);
  std::memcpy(blob.data() + old_size, data, static_cast<size_t>(byte_size));
  return static_cast<uint32_t>(aligned_offset);
}

template <typename T>
uint32_t validation_append_aligned_array(std::vector<uint8_t>& blob, const T* data, const size_t count, const uint64_t alignment) {
  if (count > (std::numeric_limits<uint64_t>::max() / sizeof(T))) {
    return kInvalidIndex;
  }
  return validation_append_aligned_bytes(blob, data, static_cast<uint64_t>(count) * sizeof(T), alignment);
}

struct ValidationChunkedPayloadLocation {
  uint32_t chunk_index = kInvalidIndex;
  uint32_t offset = kInvalidIndex;
};

struct ValidationChunkedBlobPayloadBuilder {
  uint64_t chunk_size = 512ull * 1024ull * 1024ull;
  std::vector<uint8_t> payload_blob = {};
  std::vector<etx::RHIChunkedBufferRange> chunk_ranges = {};
  std::vector<uint64_t> chunk_capacities = {};

  bool append(const void* data, const uint64_t byte_size, const uint64_t alignment, ValidationChunkedPayloadLocation& location) {
    location = {};
    if ((data == nullptr) || (byte_size == 0u)) {
      return true;
    }

    uint64_t required_chunk_capacity = chunk_size;
    if (required_chunk_capacity < byte_size) {
      required_chunk_capacity = byte_size;
    }
    if (required_chunk_capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      return false;
    }

    uint32_t target_chunk_index = kInvalidIndex;
    uint64_t aligned_offset = 0u;
    uint64_t chunk_start_offset = 0u;

    if (chunk_ranges.empty() == false) {
      const uint32_t last_chunk_index = static_cast<uint32_t>(chunk_ranges.size() - 1u);
      const uint64_t last_chunk_capacity = chunk_capacities[last_chunk_index];
      const auto& last_chunk_range = chunk_ranges[last_chunk_index];
      const uint64_t candidate_offset = validation_align_up_u64(last_chunk_range.size, alignment);
      if ((candidate_offset <= last_chunk_capacity) && ((last_chunk_capacity - candidate_offset) >= byte_size)) {
        target_chunk_index = last_chunk_index;
        aligned_offset = candidate_offset;
        chunk_start_offset = last_chunk_range.offset;
      }
    }

    if (target_chunk_index == kInvalidIndex) {
      chunk_start_offset = static_cast<uint64_t>(payload_blob.size());
      chunk_ranges.push_back({.offset = chunk_start_offset, .size = 0u});
      chunk_capacities.push_back(required_chunk_capacity);
      target_chunk_index = static_cast<uint32_t>(chunk_ranges.size() - 1u);
      aligned_offset = 0u;
    }

    if (aligned_offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      return false;
    }
    if (aligned_offset > (std::numeric_limits<uint64_t>::max() - byte_size)) {
      return false;
    }

    const uint64_t end_offset = aligned_offset + byte_size;
    if (end_offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      return false;
    }

    const uint64_t chunk_capacity = chunk_capacities[target_chunk_index];
    if (end_offset > chunk_capacity) {
      return false;
    }

    const uint64_t global_dst_offset = chunk_start_offset + aligned_offset;
    const uint64_t global_end_offset = chunk_start_offset + end_offset;
    if (global_end_offset > static_cast<uint64_t>(payload_blob.size())) {
      payload_blob.resize(static_cast<size_t>(global_end_offset), 0u);
    }
    std::memcpy(payload_blob.data() + static_cast<size_t>(global_dst_offset), data, static_cast<size_t>(byte_size));
    chunk_ranges[target_chunk_index].size = end_offset;

    location.chunk_index = target_chunk_index;
    location.offset = static_cast<uint32_t>(aligned_offset);
    return true;
  }
};

etx::RHIChunkedBufferUploadData validation_build_packed_images_blob(const etx::SceneData& scene_data) {
  etx::RHIChunkedBufferUploadData result = {};
  GPUImageBlobHeader header = {};
  const uint64_t image_count_u64 = scene_data.images.array_size();
  if (image_count_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    result.success = false;
    return result;
  }

  header.image_count = static_cast<uint32_t>(image_count_u64);
  result.metadata = std::vector<uint8_t>(sizeof(GPUImageBlobHeader), 0u);
  std::vector<::Image> packed_images(header.image_count);
  ValidationChunkedBlobPayloadBuilder payload_builder = {};

  const auto* images = scene_data.images.as_array();
  for (uint32_t i = 0u; i < header.image_count; ++i) {
    const auto& src = images[i];
    auto& dst = packed_images[i];
    etx::PackedPayloadLocation pixel_payload = {};
    etx::PackedPayloadLocation x_distribution_payload = {};
    etx::PackedPayloadLocation y_distribution_payload = {};

    if (src.data.valid()) {
      const void* ptr = scene_data.buffer_pool.map(src.data);
      ValidationChunkedPayloadLocation payload_location = {};
      if (payload_builder.append(ptr, src.data.byte_size, 16u, payload_location) == false) {
        result.success = false;
        return result;
      }
      pixel_payload.offset = payload_location.offset;
      pixel_payload.chunk_index = payload_location.chunk_index;
    }

    if (src.x_distributions_storage.valid()) {
      const void* ptr = scene_data.buffer_pool.map(src.x_distributions_storage);
      ValidationChunkedPayloadLocation payload_location = {};
      if (payload_builder.append(ptr, src.x_distributions_storage.byte_size, alignof(etx::Distribution::Entry), payload_location) == false) {
        result.success = false;
        return result;
      }
      x_distribution_payload.offset = payload_location.offset;
      x_distribution_payload.chunk_index = payload_location.chunk_index;
    }

    if (src.y_distribution_storage.valid()) {
      const void* ptr = scene_data.buffer_pool.map(src.y_distribution_storage);
      ValidationChunkedPayloadLocation payload_location = {};
      if (payload_builder.append(ptr, src.y_distribution_storage.byte_size, alignof(etx::Distribution::Entry), payload_location) == false) {
        result.success = false;
        return result;
      }
      y_distribution_payload.offset = payload_location.offset;
      y_distribution_payload.chunk_index = payload_location.chunk_index;
    }

    dst = etx::make_gpu_image_descriptor(src, pixel_payload, x_distribution_payload, y_distribution_payload);
  }

  header.images_offset = validation_append_aligned_array(result.metadata, packed_images.data(), packed_images.size(), alignof(::Image));
  if ((packed_images.empty() == false) && (header.images_offset == kInvalidIndex)) {
    result.success = false;
    return result;
  }

  if (payload_builder.chunk_ranges.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    result.success = false;
    return result;
  }

  header.data_chunk_count = static_cast<uint32_t>(payload_builder.chunk_ranges.size());
  if (header.data_chunk_count > 0u) {
    std::vector<uint32_t> placeholder_chunk_indices(header.data_chunk_count, kInvalidIndex);
    header.data_chunk_indices_offset = validation_append_aligned_array(result.metadata, placeholder_chunk_indices.data(), placeholder_chunk_indices.size(), alignof(uint32_t));
    if (header.data_chunk_indices_offset == kInvalidIndex) {
      result.success = false;
      return result;
    }
  } else {
    header.data_chunk_indices_offset = kInvalidIndex;
  }

  result.chunk_indices_offset = header.data_chunk_indices_offset;
  result.payload_data = std::move(payload_builder.payload_blob);
  result.payload_chunk_ranges = std::move(payload_builder.chunk_ranges);
  std::memcpy(result.metadata.data(), &header, sizeof(header));
  return result;
}

bool validation_create_storage_buffer(etx::RHIContext& rhi, const void* data, const uint64_t size, const etx::RHIBufferUsage usage, const bool host_visible,
  etx::RHIBuffer& out_buffer, const char* label) {
  if (size == 0u) {
    std::printf("%s buffer size is zero\n", label);
    return false;
  }

  etx::RHIBufferDesc desc = {};
  desc.size = size;
  desc.usage = etx::RHIBufferUsage::Storage | usage;
  desc.host_visible = host_visible;
  auto result = rhi.device().create_buffer(desc);
  if ((result.result != etx::RHIResult::Success) || (result.handle.valid() == false)) {
    std::printf("Failed to create %s buffer (%u)\n", label, static_cast<uint32_t>(result.result));
    return false;
  }

  if (data != nullptr) {
    const etx::RHIResult update_result = rhi.device().update_buffer(result.handle, data, size);
    if (update_result != etx::RHIResult::Success) {
      std::printf("Failed to upload %s buffer (%u)\n", label, static_cast<uint32_t>(update_result));
      rhi.device().destroy_buffer(result.handle);
      return false;
    }
  }

  out_buffer = result.handle;
  return true;
}

bool validation_create_readback_buffer(etx::RHIContext& rhi, const uint64_t size, etx::RHIBuffer& out_buffer, const char* label) {
  if (size == 0u) {
    std::printf("%s readback buffer size is zero\n", label);
    return false;
  }

  etx::RHIBufferDesc desc = {};
  desc.size = size;
  desc.usage = etx::RHIBufferUsage::TransferDst;
  desc.host_visible = true;
  auto result = rhi.device().create_buffer(desc);
  if ((result.result != etx::RHIResult::Success) || (result.handle.valid() == false)) {
    std::printf("Failed to create %s readback buffer (%u)\n", label, static_cast<uint32_t>(result.result));
    return false;
  }

  out_buffer = result.handle;
  return true;
}

void validation_destroy_buffer(etx::RHIContext& rhi, etx::RHIBuffer& buffer) {
  if (buffer.valid()) {
    rhi.device().destroy_buffer(buffer);
    buffer = {};
  }
}

float validation_read_f32(const std::vector<uint8_t>& data, const uint32_t case_index, const uint32_t offset) {
  float result = 0.0f;
  const size_t byte_offset = static_cast<size_t>(case_index) * kBSDFRuntimeValidationOutputStride + offset;
  std::memcpy(&result, data.data() + byte_offset, sizeof(result));
  return result;
}

uint32_t validation_read_u32(const std::vector<uint8_t>& data, const uint32_t case_index, const uint32_t offset) {
  uint32_t result = 0u;
  const size_t byte_offset = static_cast<size_t>(case_index) * kBSDFRuntimeValidationOutputStride + offset;
  std::memcpy(&result, data.data() + byte_offset, sizeof(result));
  return result;
}

float3 validation_read_f32x3(const std::vector<uint8_t>& data, const uint32_t case_index, const uint32_t offset) {
  return float3{validation_read_f32(data, case_index, offset + 0u), validation_read_f32(data, case_index, offset + 4u), validation_read_f32(data, case_index, offset + 8u)};
}

float validation_max_abs_diff(const float3& a, const float3& b) {
  return max(fabsf(a.x - b.x), max(fabsf(a.y - b.y), fabsf(a.z - b.z)));
}

bool validation_close_float(const char* label, const char* field, const float cpu, const float gpu, const float tolerance) {
  if ((std::isfinite(cpu) == false) || (std::isfinite(gpu) == false) || (fabsf(cpu - gpu) > tolerance)) {
    std::printf("%s %s CPU %.9f GPU %.9f tolerance %.9f\n", label, field, cpu, gpu, tolerance);
    return false;
  }
  return true;
}

bool validation_close_float3(const char* label, const char* field, const float3& cpu, const float3& gpu, const float tolerance) {
  const float error = validation_max_abs_diff(cpu, gpu);
  if ((std::isfinite(cpu.x) == false) || (std::isfinite(cpu.y) == false) || (std::isfinite(cpu.z) == false) || (std::isfinite(gpu.x) == false) || (std::isfinite(gpu.y) == false) ||
      (std::isfinite(gpu.z) == false) || (error > tolerance)) {
    std::printf("%s %s CPU %.9f %.9f %.9f GPU %.9f %.9f %.9f tolerance %.9f\n", label, field, cpu.x, cpu.y, cpu.z, gpu.x, gpu.y, gpu.z, tolerance);
    return false;
  }
  return true;
}

bool validation_equal_u32(const char* label, const char* field, const uint32_t cpu, const uint32_t gpu) {
  if (cpu != gpu) {
    std::printf("%s %s CPU %u GPU %u\n", label, field, cpu, gpu);
    return false;
  }
  return true;
}

BSDFRuntimeValidationExpected validation_expected(const etx::BSDFData& data, const etx::Material& material, const BSDFRuntimeValidationCase& test_case) {
  float3 outgoing_direction = normalize(float3{0.35f, 0.0f, 0.9367497f});
  if (material.cls == MaterialClass::DiffractionGrating) {
    const bool direction_valid =
      bsdf_diffraction_grating_order_direction(float3{0.0f, 0.0f, 1.0f}, data.spectrum_sample.wavelength, material.diffraction_grating.period_nm, 1, outgoing_direction);
    ETX_ASSERT(direction_valid);
  }
  BSDFRuntimeValidationExpected result = {};

  etx::Sampler sample_sampler(test_case.seed);
  sample_sampler.push_fixed(test_case.fixed_u, test_case.fixed_v, test_case.fixed_w);
  result.sample = etx::bsdf::sample(data, material, sample_sampler);
  result.sample_seed = sample_sampler.seed;

  etx::Sampler eval_sampler(test_case.seed + 1u);
  eval_sampler.push_fixed(test_case.fixed_u, test_case.fixed_v, test_case.fixed_w);
  result.eval = etx::bsdf::evaluate(data, outgoing_direction, material, eval_sampler);
  result.eval_seed = eval_sampler.seed;

  etx::Sampler pdf_sampler(test_case.seed + 2u);
  pdf_sampler.push_fixed(test_case.fixed_u, test_case.fixed_v, test_case.fixed_w);
  result.pdf = etx::bsdf::pdf(data, outgoing_direction, material, pdf_sampler);
  result.pdf_seed = pdf_sampler.seed;

  etx::Sampler reverse_pdf_sampler(test_case.seed + 3u);
  reverse_pdf_sampler.push_fixed(test_case.fixed_u, test_case.fixed_v, test_case.fixed_w);
  result.reverse_pdf = etx::bsdf::reverse_pdf(data, outgoing_direction, material, reverse_pdf_sampler);
  result.reverse_pdf_seed = reverse_pdf_sampler.seed;

  etx::Sampler delta_sampler(test_case.seed + 4u);
  delta_sampler.push_fixed(test_case.fixed_u, test_case.fixed_v, test_case.fixed_w);
  result.is_delta = etx::bsdf::is_delta(material, data.tex, delta_sampler) ? 1u : 0u;
  result.delta_seed = delta_sampler.seed;

  etx::Sampler albedo_sampler(test_case.seed + 5u);
  albedo_sampler.push_fixed(test_case.fixed_u, test_case.fixed_v, test_case.fixed_w);
  result.albedo = etx::bsdf::albedo(data, material, albedo_sampler);
  result.albedo_seed = albedo_sampler.seed;
  return result;
}

bool validation_check_runtime_case(const char* label, const etx::BSDFData& data, const etx::Material& material, const BSDFRuntimeValidationCase& test_case,
  const std::vector<uint8_t>& output, const uint32_t output_case_index, const uint32_t validation_operation) {
  const BSDFRuntimeValidationExpected expected = validation_expected(data, material, test_case);
  constexpr float value_tolerance = 2.5e-4f;
  constexpr float pdf_tolerance = 2.5e-4f;
  const bool check_sample = (validation_operation == 0u) || (validation_operation == 1u);
  const bool check_eval = (validation_operation == 0u) || (validation_operation == 2u);
  const bool check_pdf = (validation_operation == 0u) || (validation_operation == 3u);
  const bool check_albedo = (validation_operation == 0u) || (validation_operation == 4u);
  bool case_valid = true;

  if (check_sample) {
    case_valid =
      validation_close_float3(label, "sample.weight", expected.sample.weight.to_rgb(), validation_read_f32x3(output, output_case_index, 0u), value_tolerance) && case_valid;
    case_valid =
      validation_close_float(label, "sample.weight.value", expected.sample.weight.value, validation_read_f32(output, output_case_index, 12u), value_tolerance) && case_valid;
    case_valid = validation_close_float3(label, "sample.w_o", expected.sample.w_o, validation_read_f32x3(output, output_case_index, 16u), value_tolerance) && case_valid;
    case_valid = validation_close_float(label, "sample.pdf", expected.sample.pdf, validation_read_f32(output, output_case_index, 28u), pdf_tolerance) && case_valid;
    case_valid = validation_close_float(label, "sample.eta", expected.sample.eta, validation_read_f32(output, output_case_index, 32u), value_tolerance) && case_valid;
    case_valid = validation_equal_u32(label, "sample.properties", expected.sample.properties, validation_read_u32(output, output_case_index, 36u)) && case_valid;
    case_valid = validation_equal_u32(label, "sample.medium_index", expected.sample.medium_index, validation_read_u32(output, output_case_index, 40u)) && case_valid;
    case_valid = validation_equal_u32(label, "sample.seed", expected.sample_seed, validation_read_u32(output, output_case_index, 44u)) && case_valid;
  }

  if (check_eval) {
    case_valid = validation_close_float3(label, "eval.func", expected.eval.func.to_rgb(), validation_read_f32x3(output, output_case_index, 48u), value_tolerance) && case_valid;
    case_valid = validation_close_float(label, "eval.func.value", expected.eval.func.value, validation_read_f32(output, output_case_index, 60u), value_tolerance) && case_valid;
    case_valid = validation_close_float3(label, "eval.bsdf", expected.eval.bsdf.to_rgb(), validation_read_f32x3(output, output_case_index, 64u), value_tolerance) && case_valid;
    case_valid = validation_close_float(label, "eval.bsdf.value", expected.eval.bsdf.value, validation_read_f32(output, output_case_index, 76u), value_tolerance) && case_valid;
    case_valid = validation_close_float(label, "eval.pdf", expected.eval.pdf, validation_read_f32(output, output_case_index, 80u), pdf_tolerance) && case_valid;
    case_valid = validation_close_float(label, "eval.eta", expected.eval.eta, validation_read_f32(output, output_case_index, 84u), value_tolerance) && case_valid;
    case_valid = validation_equal_u32(label, "eval.properties", expected.eval.properties, validation_read_u32(output, output_case_index, 88u)) && case_valid;
    case_valid = validation_equal_u32(label, "eval.medium_index", expected.eval.medium_index, validation_read_u32(output, output_case_index, 92u)) && case_valid;
    case_valid = validation_equal_u32(label, "eval.seed", expected.eval_seed, validation_read_u32(output, output_case_index, 96u)) && case_valid;
  }

  if (check_pdf) {
    case_valid = validation_close_float(label, "pdf", expected.pdf, validation_read_f32(output, output_case_index, 100u), pdf_tolerance) && case_valid;
    case_valid = validation_equal_u32(label, "pdf.seed", expected.pdf_seed, validation_read_u32(output, output_case_index, 104u)) && case_valid;
    case_valid = validation_close_float(label, "reverse_pdf", expected.reverse_pdf, validation_read_f32(output, output_case_index, 108u), pdf_tolerance) && case_valid;
    case_valid = validation_equal_u32(label, "reverse_pdf.seed", expected.reverse_pdf_seed, validation_read_u32(output, output_case_index, 112u)) && case_valid;
  }

  if (check_albedo) {
    case_valid = validation_equal_u32(label, "is_delta", expected.is_delta, validation_read_u32(output, output_case_index, 116u)) && case_valid;
    case_valid = validation_equal_u32(label, "delta.seed", expected.delta_seed, validation_read_u32(output, output_case_index, 120u)) && case_valid;
    case_valid = validation_close_float3(label, "albedo", expected.albedo.to_rgb(), validation_read_f32x3(output, output_case_index, 128u), value_tolerance) && case_valid;
    case_valid = validation_close_float(label, "albedo.value", expected.albedo.value, validation_read_f32(output, output_case_index, 140u), value_tolerance) && case_valid;
    case_valid = validation_equal_u32(label, "albedo.seed", expected.albedo_seed, validation_read_u32(output, output_case_index, 144u)) && case_valid;
  }

  return case_valid;
}

bool validate_bsdf_runtime_numeric_harness(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count,
  const bool diffraction_only = false) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(128u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  set_openpbr_validation_defaults(scene_data);

  scene_data.materials.emplace_back(make_mirror_conductor(0.5f));
  scene_data.materials.emplace_back(make_white_sapphire_dielectric(0.5f));
  scene_data.materials.emplace_back(make_plastic(0.5f));
  scene_data.materials.emplace_back(make_openpbr(0.5f, 1.0f, 0.0f, SpectrumWhite, false));
  scene_data.materials.emplace_back(make_openpbr(0.5f, 0.0f, 1.0f, SpectrumWhite, false));
  etx::Material diffraction = {};
  diffraction.cls = MaterialClass::DiffractionGrating;
  diffraction.reflectance.spectrum_index = SpectrumHalf;
  diffraction.diffraction_grating.period_nm = 1600.0f;
  diffraction.diffraction_grating.optical_path_difference_nm = 275.0f;
  diffraction.diffraction_grating.duty_cycle = 0.5f;
  diffraction.diffraction_grating.rotation = 0.0f;
  scene_data.materials.emplace_back(diffraction);

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("BSDF runtime numeric harness failed to bind LUTs\n");
    return false;
  }
  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  set_openpbr_validation_defaults(exact_scene);
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};

  const char* labels[] = {"rough conductor", "rough dielectric", "rough plastic", "openpbr conductor", "openpbr dielectric", "diffraction grating"};
  const BSDFRuntimeValidationCase cases[] = {
    {0u, 41001u, 0.31f, 0.63f, 0.17f, 0u, 0u, 0u},
    {1u, 42001u, 0.41f, 0.23f, 0.72f, 0u, 0u, 0u},
    {2u, 43001u, 0.19f, 0.77f, 0.44f, 0u, 0u, 0u},
    {3u, 44001u, 0.55f, 0.37f, 0.28f, 0u, 0u, 0u},
    {4u, 45001u, 0.27f, 0.52f, 0.66f, 0u, 0u, 0u},
    {5u, 46001u, 0.43f, 0.29f, 0.61f, 0u, 0u, 0u},
  };
  etx::RHIInitInfo init_info = {
    .backend = select_default_backend(),
    .enable_validation = ETX_DEBUG,
    .headless = true,
  };
  etx::RHIContext rhi = etx::RHIContext::create(init_info);
  if (rhi.valid() == false) {
    std::printf("BSDF runtime numeric harness failed to create RHI context\n");
    etx::scene_global_clear(&exact_scene);
    etx::scene_global_publish(&original_scene, &original_scene);
    return false;
  }
  rhi.initialize_headless();

  etx::RHIBuffer material_buffer = {};
  etx::RHIBuffer spectrum_buffer = {};
  etx::RHIBuffer interface_buffer = {};
  etx::RHIBuffer globals_buffer = {};
  etx::RHIChunkedBufferState image_blob_state = {};
  bool valid = true;

  do {
    const etx::RHIChunkedBufferUploadData image_blob = validation_build_packed_images_blob(scene_data);
    if (rhi.device().upload_or_update_chunked_buffer(image_blob, etx::RHIBufferUsage::Storage | etx::RHIBufferUsage::TransferDst, image_blob_state, "bsdf_validation_images") ==
        false) {
      std::printf("BSDF runtime numeric harness failed to upload images\n");
      valid = false;
      break;
    }

    GPUSceneGlobals globals = {};
    globals.default_black_spectrum = scene_data.defaults.black_spectrum;
    globals.default_white_spectrum = scene_data.defaults.white_spectrum;
    globals.default_rayleigh_spectrum = scene_data.defaults.rayleigh_spectrum;
    globals.default_mie_spectrum = scene_data.defaults.mie_spectrum;
    globals.default_ozone_spectrum = scene_data.defaults.ozone_spectrum;
    globals.default_subsurface_scatter_material = scene_data.defaults.subsurface_scatter_material;
    globals.default_subsurface_exit_material = scene_data.defaults.subsurface_exit_material;
    globals.default_missing_material = scene_data.defaults.missing_material;
    globals.default_dielectric_eta = scene_data.defaults.dielectric_eta;
    globals.default_conductor_eta = scene_data.defaults.conductor_eta;
    globals.default_conductor_k = scene_data.defaults.conductor_k;
    globals.pixel_filter_image_index = scene_data.pixel_filter.image_index;
    globals.pixel_filter_radius = scene_data.pixel_filter.radius;

    valid = validation_create_storage_buffer(rhi, scene_data.materials.data(), scene_data.materials.size() * sizeof(etx::Material), etx::RHIBufferUsage::TransferDst, true,
              material_buffer, "bsdf validation materials") &&
            valid;
    valid = validation_create_storage_buffer(rhi, scene_data.spectrum_values.data(), scene_data.spectrum_values.size() * sizeof(etx::SpectralDistribution),
              etx::RHIBufferUsage::TransferDst, true, spectrum_buffer, "bsdf validation spectra") &&
            valid;
    valid = validation_create_storage_buffer(rhi, scene_data.energy_compensation_interfaces.data(),
              scene_data.energy_compensation_interfaces.size() * sizeof(etx::Scene::EnergyCompensationInterface), etx::RHIBufferUsage::TransferDst, true, interface_buffer,
              "bsdf validation interfaces") &&
            valid;
    valid = validation_create_storage_buffer(rhi, &globals, sizeof(globals), etx::RHIBufferUsage::TransferDst, true, globals_buffer, "bsdf validation globals") && valid;
    if (valid == false) {
      break;
    }

    struct PushConstants {
      uint32_t case_buffer_index;
      uint32_t output_buffer_index;
      uint32_t materials_descriptor_index;
      uint32_t images_descriptor_index;
      uint32_t spectrums_descriptor_index;
      uint32_t energy_compensation_interfaces_descriptor_index;
      uint32_t scene_globals_descriptor_index;
      uint32_t case_count;
    };

    enum : uint32_t {
      ValidationKindPlastic = 1u,
      ValidationKindConductor = 2u,
      ValidationKindDielectric = 3u,
      ValidationKindDiffraction = 4u,
    };

    auto validate_batch = [&](const char* batch_label, const uint32_t first_case, const uint32_t batch_case_count, const uint32_t validation_mode,
                            const uint32_t validation_operation, const uint32_t validation_kind) -> bool {
      std::vector<BSDFRuntimeValidationCase> batch_cases(cases + first_case, cases + first_case + batch_case_count);
      etx::RHIBuffer case_buffer = {};
      etx::RHIBuffer output_buffer = {};
      etx::RHIBuffer readback_buffer = {};
      etx::RHIPipeline pipeline = {};
      etx::RHICommandBuffer cmd = {};
      bool batch_valid = true;

      do {
        batch_valid = validation_create_storage_buffer(rhi, batch_cases.data(), batch_cases.size() * sizeof(BSDFRuntimeValidationCase), etx::RHIBufferUsage::TransferDst, true,
                        case_buffer, "bsdf validation cases") &&
                      batch_valid;
        batch_valid = validation_create_storage_buffer(rhi, nullptr, batch_case_count * kBSDFRuntimeValidationOutputStride, etx::RHIBufferUsage::TransferSrc, false, output_buffer,
                        "bsdf validation output") &&
                      batch_valid;
        batch_valid = validation_create_readback_buffer(rhi, batch_case_count * kBSDFRuntimeValidationOutputStride, readback_buffer, "bsdf validation") && batch_valid;
        if (batch_valid == false) {
          break;
        }

        auto& compiler = etx::ShaderCompiler::instance();
        std::unordered_map<std::string, std::string> defines = {
          {"ETX_BSDF_RUNTIME_VALIDATION_MODE", std::to_string(validation_mode)},
          {"ETX_BSDF_RUNTIME_VALIDATION_OPERATION", std::to_string(validation_operation)},
          {"ETX_BSDF_RUNTIME_VALIDATION_KIND", std::to_string(validation_kind)},
        };
        if (validation_mode == 3u) {
          defines["ETX_DXC_OPT_LEVEL"] = "0";
          defines["ETX_DXC_SPIRV_OPT_CONFIG"] = "--compact-ids";
        }
        const auto compilation = compiler.compile("shaders/bsdf_runtime_validation.hlsl", {{"main", etx::RHIShaderStage::Compute}}, defines, rhi.backend());
        if ((compilation.result != etx::RHIResult::Success) || compilation.binaries.empty()) {
          std::printf("BSDF runtime numeric %s operation %u shader compilation failed: %s\n", batch_label, validation_operation, compilation.error_message.c_str());
          batch_valid = false;
          break;
        }

        const etx::RHIComputePipelineDesc pipeline_desc = rhi.device().make_compute_pipeline_desc(compilation.binaries[0]);
        auto pipeline_result = rhi.device().create_compute_pipeline(pipeline_desc);
        if ((pipeline_result.result != etx::RHIResult::Success) || (pipeline_result.handle.valid() == false)) {
          std::printf("BSDF runtime numeric %s operation %u pipeline creation failed (%u)\n", batch_label, validation_operation, static_cast<uint32_t>(pipeline_result.result));
          batch_valid = false;
          break;
        }
        pipeline = pipeline_result.handle;

        const PushConstants pc = {
          etx::get_bindless_descriptor_index(case_buffer),
          etx::get_bindless_descriptor_index(output_buffer),
          etx::get_bindless_descriptor_index(material_buffer),
          image_blob_state.metadata_descriptor_index,
          etx::get_bindless_descriptor_index(spectrum_buffer),
          etx::get_bindless_descriptor_index(interface_buffer),
          etx::get_bindless_descriptor_index(globals_buffer),
          batch_case_count,
        };

        cmd = rhi.get_command_buffer();
        if (cmd.valid() == false) {
          std::printf("BSDF runtime numeric %s harness failed to acquire command buffer\n", batch_label);
          batch_valid = false;
          break;
        }

        rhi.command_buffer_begin(cmd);
        rhi.cmd_set_pipeline(cmd, pipeline);
        rhi.cmd_push_constants(cmd, &pc, sizeof(pc), 0);
        etx::RHIDispatchDesc dispatch = {};
        dispatch.group_count_x = 1u;
        dispatch.group_count_y = 1u;
        dispatch.group_count_z = 1u;
        rhi.cmd_dispatch(cmd, dispatch);
        rhi.cmd_buffer_barrier(cmd, output_buffer, etx::RHIResourceState::General, etx::RHIResourceState::TransferSrc);
        rhi.cmd_copy_buffer(cmd, output_buffer, readback_buffer, batch_case_count * kBSDFRuntimeValidationOutputStride);
        rhi.command_buffer_end(cmd);
        rhi.submit_command_buffer({cmd});
        const etx::RHIResult wait_result = rhi.wait_idle();
        if (wait_result != etx::RHIResult::Success) {
          std::printf("BSDF runtime numeric %s operation %u dispatch failed to wait (%u)\n", batch_label, validation_operation, static_cast<uint32_t>(wait_result));
          batch_valid = false;
          break;
        }

        std::vector<uint8_t> output(batch_case_count * kBSDFRuntimeValidationOutputStride);
        const etx::RHIResult read_result = rhi.device().read_buffer(readback_buffer, output.data(), output.size());
        if (read_result != etx::RHIResult::Success) {
          std::printf("BSDF runtime numeric %s operation %u readback failed (%u)\n", batch_label, validation_operation, static_cast<uint32_t>(read_result));
          batch_valid = false;
          break;
        }

        for (uint32_t i = 0u; i < batch_case_count; ++i) {
          const uint32_t case_index = first_case + i;
          const auto& test_case = cases[case_index];
          const etx::Material& material = scene_data.materials[test_case.material_index];
          etx::BSDFData expected_data = data;
          if (validation_mode == 3u) {
            expected_data.spectrum_sample = etx::SpectralQuery{550.0f, SpectralFlags::Spectral};
          }
          batch_valid = validation_check_runtime_case(labels[case_index], expected_data, material, test_case, output, i, validation_operation) && batch_valid;
        }
      } while (false);

      if (cmd.valid()) {
        rhi.destroy_command_buffer(cmd);
      }
      if (pipeline.valid()) {
        rhi.device().destroy_pipeline(pipeline);
      }
      validation_destroy_buffer(rhi, case_buffer);
      validation_destroy_buffer(rhi, output_buffer);
      validation_destroy_buffer(rhi, readback_buffer);
      return batch_valid;
    };

    if (diffraction_only == false) {
      for (uint32_t operation = 1u; operation <= 4u; ++operation) {
        valid = validate_batch("energy conductor", 0u, 1u, 0u, operation, ValidationKindConductor) && valid;
        valid = validate_batch("energy dielectric", 1u, 1u, 0u, operation, ValidationKindDielectric) && valid;
        valid = validate_batch("plastic", 2u, 1u, 1u, operation, ValidationKindPlastic) && valid;
      }
    }
    for (uint32_t operation = 1u; operation <= 4u; ++operation) {
      valid = validate_batch("diffraction", 5u, 1u, 3u, operation, ValidationKindDiffraction) && valid;
    }
  } while (false);

  validation_destroy_buffer(rhi, material_buffer);
  validation_destroy_buffer(rhi, spectrum_buffer);
  validation_destroy_buffer(rhi, interface_buffer);
  validation_destroy_buffer(rhi, globals_buffer);
  rhi.device().destroy_chunked_buffer(image_blob_state);
  rhi.wait_idle();
  rhi = {};

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);

  if (valid) {
    std::printf("BSDF runtime numeric CPU/GPU harness valid\n");
  }
  return valid;
}

bool validate_openpbr_case(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count, const char* label,
  const etx::Material& source_material, const uint32_t seed, const bool require_integrated_energy) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  set_openpbr_validation_defaults(scene_data);
  scene_data.materials.emplace_back(source_material);

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("%s failed to bind OpenPBR LUTs\n", label);
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene openpbr_scene = {};
  openpbr_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  openpbr_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  openpbr_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  set_openpbr_validation_defaults(openpbr_scene);
  openpbr_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&openpbr_scene, &openpbr_scene);

  const etx::Material material = scene_data.materials[0];
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};

  bool diagnostic_valid = true;
  float maximum_weight = 0.0f;
  uint32_t valid_sample_count = 0u;
  etx::Sampler sampler(seed);
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if (validate_sample(sample) == false) {
      diagnostic_valid = false;
      continue;
    }
    if (sample.valid()) {
      valid_sample_count += 1u;
      maximum_weight = max(maximum_weight, sample.weight.maximum());
    }
  }

  if ((valid_sample_count == 0u) || (maximum_weight <= kEpsilon)) {
    std::printf("%s produced no visible OpenPBR samples\n", label);
    diagnostic_valid = false;
  }

  if (require_integrated_energy) {
    const float bsdf_energy = integrate_bsdf_energy(data, material, seed + 30000u);
    if (((std::isfinite(bsdf_energy) == false) || (bsdf_energy < 0.0f)) || (bsdf_energy > 1.2f)) {
      std::printf("%s invalid OpenPBR integrated energy %.6f\n", label, bsdf_energy);
      diagnostic_valid = false;
    } else {
      std::printf("%s OpenPBR integrated energy %.6f valid samples %u max weight %.6f\n", label, bsdf_energy, valid_sample_count, maximum_weight);
    }
  } else {
    std::printf("%s OpenPBR valid samples %u max weight %.6f\n", label, valid_sample_count, maximum_weight);
  }

  etx::scene_global_clear(&openpbr_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

float spectral_response_max_abs_difference(const etx::SpectralResponse& a, const etx::SpectralResponse& b) {
  if ((a.spectral()) || (b.spectral())) {
    return fabsf(a.monochromatic() - b.monochromatic());
  }

  const float3 d = abs(a.integrated - b.integrated);
  return max(d.x, max(d.y, d.z));
}

bool validate_openpbr_thinfilm_delegate(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count, const char* label,
  const etx::Material& source_material, const uint32_t expected_class, const uint32_t seed) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  set_openpbr_validation_defaults(scene_data);
  scene_data.materials.emplace_back(source_material);

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("%s failed to bind OpenPBR thinfilm delegate LUTs\n", label);
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene openpbr_scene = {};
  openpbr_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  openpbr_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  openpbr_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  set_openpbr_validation_defaults(openpbr_scene);
  openpbr_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&openpbr_scene, &openpbr_scene);

  const etx::Material material = scene_data.materials[0];
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};
  const float3 outgoing_direction = normalize(float3{0.35f, 0.0f, 0.9367497f});
  const etx::OpenPBRBSDF::OpenPBRComponents components = etx::OpenPBRBSDF::make_components(data, material);

  etx::Material delegate = {};
  float delegate_weight = 0.0f;
  if (expected_class == MaterialClass::Conductor) {
    delegate = components.conductor;
    delegate_weight = components.conductor_weight;
  } else if (expected_class == MaterialClass::Dielectric) {
    delegate = components.dielectric;
    delegate_weight = components.dielectric_weight;
  } else if (expected_class == MaterialClass::Plastic) {
    delegate = components.plastic;
    delegate_weight = components.plastic_weight;
  } else {
    std::printf("%s invalid expected delegate class %u\n", label, expected_class);
    etx::scene_global_clear(&openpbr_scene);
    etx::scene_global_publish(&original_scene, &original_scene);
    return false;
  }

  bool diagnostic_valid = true;
  if ((fabsf(delegate_weight - 1.0f) > 1.0e-6f) || (delegate.cls != expected_class)) {
    std::printf("%s invalid delegate weight %.6f class %u expected %u\n", label, delegate_weight, delegate.cls, expected_class);
    diagnostic_valid = false;
  }

  if ((delegate.thinfilm.min_thickness <= 0.0f) || (delegate.thinfilm.max_thickness <= 0.0f) || (delegate.thinfilm.ior.cls == etx::SpectralDistribution::Invalid)) {
    std::printf("%s did not preserve thinfilm on delegate\n", label);
    diagnostic_valid = false;
  }

  etx::Sampler open_eval_sampler(seed, seed ^ 0x72163c1u);
  const etx::BSDFEval openpbr_eval = etx::bsdf::evaluate(data, outgoing_direction, material, open_eval_sampler);
  etx::Sampler delegate_eval_sampler(seed + 1u, seed ^ 0x18cc04du);
  const etx::BSDFEval delegate_eval = etx::bsdf::evaluate(data, outgoing_direction, delegate, delegate_eval_sampler);
  const float bsdf_error = spectral_response_max_abs_difference(openpbr_eval.bsdf, delegate_eval.bsdf);
  const float bsdf_scale = max(openpbr_eval.bsdf.maximum(), delegate_eval.bsdf.maximum());
  const float bsdf_tolerance = max(1.0e-5f, 1.0e-4f * bsdf_scale);
  if ((openpbr_eval.valid() == false) || (delegate_eval.valid() == false) || (bsdf_error > bsdf_tolerance)) {
    std::printf("%s delegate bsdf mismatch open %.6f delegate %.6f error %.6f\n", label, openpbr_eval.bsdf.monochromatic(), delegate_eval.bsdf.monochromatic(), bsdf_error);
    diagnostic_valid = false;
  }

  etx::Sampler open_pdf_sampler(seed + 2u, seed ^ 0x6adc8d5u);
  const float openpbr_pdf = etx::bsdf::pdf(data, outgoing_direction, material, open_pdf_sampler);
  etx::Sampler delegate_pdf_sampler(seed + 3u, seed ^ 0x5cfcb0bu);
  const float delegate_pdf = etx::bsdf::pdf(data, outgoing_direction, delegate, delegate_pdf_sampler);
  const float pdf_tolerance = max(1.0e-5f, 1.0e-4f * max(openpbr_pdf, delegate_pdf));
  if ((std::isfinite(openpbr_pdf) == false) || (std::isfinite(delegate_pdf) == false) || (fabsf(openpbr_pdf - delegate_pdf) > pdf_tolerance)) {
    std::printf("%s delegate pdf mismatch open %.6f delegate %.6f\n", label, openpbr_pdf, delegate_pdf);
    diagnostic_valid = false;
  }

  if (diagnostic_valid) {
    std::printf("%s OpenPBR thinfilm delegate valid\n", label);
  }

  etx::scene_global_clear(&openpbr_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

bool validate_openpbr_parameter_sweeps(etx::Scene& scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count) {
  bool valid = true;
  valid = validate_openpbr_case(scene, spectra, spectrum_count, "openpbr roughness low", make_openpbr(0.05f, 0.0f, 0.0f, SpectrumWhite, false), 40000u, true) && valid;
  valid = validate_openpbr_case(scene, spectra, spectrum_count, "openpbr roughness high", make_openpbr(1.0f, 0.0f, 0.0f, SpectrumWhite, false), 41000u, true) && valid;
  valid = validate_openpbr_case(scene, spectra, spectrum_count, "openpbr transmission zero", make_openpbr(0.5f, 0.0f, 0.0f, SpectrumWhite, false), 42000u, true) && valid;
  valid = validate_openpbr_case(scene, spectra, spectrum_count, "openpbr transmission one", make_openpbr(0.5f, 0.0f, 1.0f, SpectrumColored, false), 43000u, true) && valid;
  valid = validate_openpbr_case(scene, spectra, spectrum_count, "openpbr transmission delta", make_openpbr(0.0f, 0.0f, 1.0f, SpectrumColored, false), 43500u, false) && valid;
  valid = validate_openpbr_case(scene, spectra, spectrum_count, "openpbr metalness zero", make_openpbr(0.5f, 0.0f, 0.0f, SpectrumWhite, false), 44000u, true) && valid;
  valid = validate_openpbr_case(scene, spectra, spectrum_count, "openpbr metalness one", make_openpbr(0.5f, 1.0f, 0.0f, SpectrumColored, false), 45000u, true) && valid;
  valid = validate_openpbr_case(scene, spectra, spectrum_count, "openpbr base color colored", make_openpbr(0.5f, 0.0f, 0.0f, SpectrumColored, false), 46000u, true) && valid;
  valid = validate_openpbr_case(scene, spectra, spectrum_count, "openpbr thinfilm disabled", make_openpbr(0.0f, 0.0f, 0.0f, SpectrumWhite, false), 47000u, false) && valid;
  valid = validate_openpbr_case(scene, spectra, spectrum_count, "openpbr thinfilm enabled", make_openpbr(0.0f, 0.0f, 0.0f, SpectrumWhite, true), 48000u, false) && valid;
  valid = validate_openpbr_case(scene, spectra, spectrum_count, "openpbr thinfilm rough", make_openpbr(0.5f, 0.0f, 0.0f, SpectrumWhite, true), 49000u, true) && valid;
  valid = validate_openpbr_thinfilm_delegate(scene, spectra, spectrum_count, "openpbr thinfilm plastic delegate", make_openpbr(0.5f, 0.0f, 0.0f, SpectrumWhite, true),
            MaterialClass::Plastic, 50000u) &&
          valid;
  valid = validate_openpbr_thinfilm_delegate(scene, spectra, spectrum_count, "openpbr thinfilm dielectric delegate", make_openpbr(0.5f, 0.0f, 1.0f, SpectrumWhite, true),
            MaterialClass::Dielectric, 51000u) &&
          valid;
  valid = validate_openpbr_thinfilm_delegate(scene, spectra, spectrum_count, "openpbr thinfilm conductor delegate", make_openpbr(0.5f, 1.0f, 0.0f, SpectrumWhite, true),
            MaterialClass::Conductor, 52000u) &&
          valid;
  return valid;
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  etx::env().setup("bin/bsdf_validation.exe");
  if (validate_spectral_sample_invariants() == false) {
    return 1;
  }
  if (validate_spectral_sampling_distribution() == false) {
    return 1;
  }
  bool runtime_only = false;
  bool thinfilm_optics_only = false;
  bool thinfilm_validation_only = false;
  bool thinfilm_furnace_only = false;
  bool energy_compensation_parity_only = false;
  bool diffraction_validation_only = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--bsdf-runtime-only") == 0) {
      runtime_only = true;
    } else if (std::strcmp(argv[i], "--thinfilm-optics-only") == 0) {
      thinfilm_optics_only = true;
    } else if (std::strcmp(argv[i], "--thinfilm-validation-only") == 0) {
      thinfilm_validation_only = true;
    } else if (std::strcmp(argv[i], "--thinfilm-furnace-only") == 0) {
      thinfilm_furnace_only = true;
    } else if (std::strcmp(argv[i], "--energy-compensation-parity-only") == 0) {
      energy_compensation_parity_only = true;
    } else if (std::strcmp(argv[i], "--diffraction-validation-only") == 0) {
      diffraction_validation_only = true;
    }
  }

  if (thinfilm_optics_only) {
    return validate_thinfilm_optical_invariants() ? 0 : 1;
  }

  if (energy_compensation_parity_only) {
    const bool valid = validate_spectral_energy_compensation_lut_sampling() && validate_energy_compensation_gpu_shader_compile() && validate_energy_compensation_gpu_lut_parity();
    return valid ? 0 : 1;
  }

  etx::SpectralDistribution spectra[SpectrumCount] = {};
  spectra[SpectrumWhite] = make_spectrum(float3{1.0f, 1.0f, 1.0f});
  spectra[SpectrumBlack] = make_spectrum(float3{0.0f, 0.0f, 0.0f});
  spectra[SpectrumHalf] = make_spectrum(float3{0.5f, 0.5f, 0.5f});
  spectra[SpectrumColored] = make_spectrum(float3{0.8f, 0.35f, 0.15f});
  spectra[SpectrumAirEta] = make_spectrum(float3{1.0f, 1.0f, 1.0f});
  spectra[SpectrumDielectricEta] = make_spectrum(float3{1.5f, 1.5f, 1.5f});
  spectra[SpectrumSapphireEta] = make_spectrum(float3{1.77f, 1.77f, 1.77f});
  std::string named_ior_title = {};
  etx::SpectralDistribution::load_refractive_index(etx::env().file_in_data("spectrum/dielectric/plastic.spd"), spectra[SpectrumNamedPlasticEta], spectra[SpectrumNamedPlasticK],
    named_ior_title);
  etx::SpectralDistribution::load_refractive_index(etx::env().file_in_data("spectrum/dielectric/water.spd"), spectra[SpectrumNamedWaterEta], spectra[SpectrumNamedWaterK],
    named_ior_title);
  spectra[SpectrumConductorEta] = make_spectrum(float3{0.25f, 0.45f, 1.05f});
  spectra[SpectrumConductorK] = make_spectrum(float3{3.4f, 2.4f, 1.9f});
  spectra[SpectrumMirrorEta] = make_loaded_ior_constant(0.0f);
  spectra[SpectrumMirrorK] = make_loaded_ior_constant(1000000.0f);
  spectra[SpectrumSpectralWhite] = etx::SpectralDistribution::constant(1.0f);

  etx::Scene scene = {};
  scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{spectra, SpectrumCount};
  scene.defaults.white_spectrum = SpectrumWhite;
  scene.defaults.dielectric_eta = SpectrumDielectricEta;
  scene.defaults.conductor_eta = SpectrumMirrorEta;
  scene.defaults.conductor_k = SpectrumMirrorK;

  etx::scene_global_init();
  etx::scene_global_publish(&scene, &scene);
  if (diffraction_validation_only) {
    const bool diffraction_valid = validate_diffraction_grating_serialization() && validate_diffraction_grating_gpu_shader_compile() &&
                                   validate_diffraction_grating_contract(scene) && validate_bsdf_runtime_numeric_harness(scene, spectra, SpectrumCount, true);
    etx::scene_global_clear(&scene);
    etx::scene_global_deinit();
    return diffraction_valid ? 0 : 1;
  }
  if (runtime_only) {
    const bool runtime_valid = validate_bsdf_runtime_numeric_harness(scene, spectra, SpectrumCount);
    etx::scene_global_clear(&scene);
    etx::scene_global_deinit();
    return runtime_valid ? 0 : 1;
  }

  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};
  etx::BSDFData inside_data = data;
  inside_data.w_i = float3{0.0f, 0.0f, 1.0f};

  if (thinfilm_validation_only || thinfilm_furnace_only) {
    bool thinfilm_valid = validate_thinfilm_optical_invariants();
    if (thinfilm_validation_only) {
      thinfilm_valid = validate_energy_compensation_gpu_shader_compile() && thinfilm_valid;
      thinfilm_valid = validate_energy_compensation_gpu_lut_parity() && thinfilm_valid;
    }

    const etx::Material standalone = make_standalone_thinfilm(0.0f, 500.0f);
    thinfilm_valid = validate_standalone_thinfilm_contract("standalone thinfilm outside", data, standalone, 22000u) && thinfilm_valid;
    thinfilm_valid = validate_standalone_thinfilm_contract("standalone thinfilm inside", inside_data, standalone, 22100u) && thinfilm_valid;
    thinfilm_valid = validate_standalone_thinfilm_sheet_symmetry("standalone thinfilm boundary ior ignored", data, standalone, 22150u) && thinfilm_valid;
    thinfilm_valid = validate_delta_thinfilm_coating_sample("delta dielectric thinfilm outside", data, make_thinfilm_delta_dielectric(), 22200u, true) && thinfilm_valid;
    thinfilm_valid = validate_delta_thinfilm_coating_sample("delta conductor thinfilm outside", data, make_thinfilm_delta_conductor(), 22400u, false) && thinfilm_valid;
    thinfilm_valid = validate_delta_plastic_thinfilm_contract("delta plastic thinfilm outside", data, make_thinfilm_delta_plastic(), 22500u) && thinfilm_valid;

    const float plastic_roughness_values[] = {0.25f, 0.5f, 1.0f};
    for (uint32_t i = 0u; i < 3u; ++i) {
      thinfilm_valid = validate_thinfilm_plastic_interface(scene, spectra, SpectrumCount, plastic_roughness_values[i], 30500u + i * 1000u) && thinfilm_valid;
    }
    thinfilm_valid = validate_exact_energy_compensated_conductor_interface(scene, spectra, SpectrumCount, "mirror conductor thinfilm exact interface",
                       make_thinfilm_rough_conductor(0.5f), 0.5f, 33600u) &&
                     thinfilm_valid;
    thinfilm_valid = validate_exact_energy_compensated_dielectric_interface(scene, spectra, SpectrumCount, "sapphire dielectric thinfilm exact interface",
                       make_thinfilm_rough_dielectric(0.5f), 0.5f, 38600u) &&
                     thinfilm_valid;
    thinfilm_valid = validate_thinfilm_energy_compensation_cache_key(spectra, SpectrumCount) && thinfilm_valid;
    thinfilm_valid = validate_variable_thinfilm_texture_lut(spectra, SpectrumCount) && thinfilm_valid;
    thinfilm_valid =
      validate_openpbr_case(scene, spectra, SpectrumCount, "openpbr thinfilm rough", make_openpbr(0.5f, 0.0f, 0.0f, SpectrumWhite, true), 49000u, true) && thinfilm_valid;

    etx::scene_global_clear(&scene);
    etx::scene_global_deinit();
    return thinfilm_valid ? 0 : 1;
  }

  bool valid = true;
  valid = validate_thinfilm_optical_invariants() && valid;
  valid = validate_image_3d_sampling() && valid;
  valid = validate_spectral_energy_compensation_lut_sampling() && valid;
  valid = validate_energy_compensation_gpu_shader_compile() && valid;
  valid = validate_energy_compensation_gpu_lut_parity() && valid;
  valid = validate_bsdf_runtime_numeric_harness(scene, spectra, SpectrumCount) && valid;
  const etx::Material standalone_thinfilm = make_standalone_thinfilm(0.0f, 500.0f);
  valid = validate_standalone_thinfilm_contract("standalone thinfilm outside", data, standalone_thinfilm, 22000u) && valid;
  valid = validate_standalone_thinfilm_contract("standalone thinfilm inside", inside_data, standalone_thinfilm, 22100u) && valid;
  valid = validate_standalone_thinfilm_sheet_symmetry("standalone thinfilm boundary ior ignored", data, standalone_thinfilm, 22150u) && valid;
  valid = validate_delta_thinfilm_coating_sample("delta dielectric thinfilm outside", data, make_thinfilm_delta_dielectric(), 22200u, true) && valid;
  valid = validate_delta_thinfilm_coating_sample("delta dielectric thinfilm inside", inside_data, make_thinfilm_delta_dielectric(), 22300u, true) && valid;
  valid = validate_delta_thinfilm_coating_sample("delta conductor thinfilm outside", data, make_thinfilm_delta_conductor(), 22400u, false) && valid;
  valid = validate_delta_plastic_thinfilm_contract("delta plastic thinfilm outside", data, make_thinfilm_delta_plastic(), 22500u) && valid;
  valid = validate_delta_plastic_thinfilm_contract("delta plastic thinfilm inside", inside_data, make_thinfilm_delta_plastic(), 22600u) && valid;
  valid = validate_named_plastic_water_exact_interface(scene, spectra, SpectrumCount) && valid;
  valid = validate_plastic_low_roughness_substrate_exit(scene, spectra, SpectrumCount) && valid;

  const float diffuse_roughness_values[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
  for (uint32_t i = 0u; i < 5u; ++i) {
    const float roughness = diffuse_roughness_values[i];
    valid = validate_eon_diffuse_material(data, roughness, 24000u + i * 1000u) && valid;
  }
  const float velvet_roughness_values[] = {0.25f, 0.5f, 0.75f, 1.0f};
  for (uint32_t i = 0u; i < 4u; ++i) {
    const float roughness = velvet_roughness_values[i];
    valid = validate_velvet_material(data, roughness, 25000u + i * 1000u) && valid;
  }
  const float translucent_roughness_values[] = {0.0f, 0.5f, 1.0f};
  for (uint32_t i = 0u; i < 3u; ++i) {
    const float roughness = translucent_roughness_values[i];
    valid = validate_balanced_translucent_material(data, roughness, 24500u + i * 1000u) && valid;
  }
  const float plastic_roughness_values[] = {0.25f, 0.5f, 0.75f, 1.0f};
  for (uint32_t i = 0u; i < 4u; ++i) {
    const float plastic_roughness = plastic_roughness_values[i];
    valid = validate_exact_plastic_interface(scene, spectra, SpectrumCount, plastic_roughness, 26000u + i * 1000u) && valid;
  }
  const float thinfilm_plastic_roughness_values[] = {0.25f, 0.5f, 1.0f};
  for (uint32_t i = 0u; i < 3u; ++i) {
    const float plastic_roughness = thinfilm_plastic_roughness_values[i];
    valid = validate_thinfilm_plastic_interface(scene, spectra, SpectrumCount, plastic_roughness, 30500u + i * 1000u) && valid;
  }
  const etx::Material equal_ior_dielectric = make_white_equal_ior_dielectric(1.0f);
  valid = validate_equal_ior_dielectric_direction("dielectric outside", data, equal_ior_dielectric, 25500u) && valid;
  valid = validate_equal_ior_dielectric_direction("dielectric inside", inside_data, equal_ior_dielectric, 25510u) && valid;

  const etx::Material delta_sapphire_dielectric = make_white_sapphire_dielectric(0.0f);
  valid = (validate_delta_dielectric_transmission_sample("sapphire delta dielectric outside", data, delta_sapphire_dielectric, 25520u) && valid);
  valid = (validate_delta_dielectric_transmission_sample("sapphire delta dielectric inside", inside_data, delta_sapphire_dielectric, 25530u) && valid);
  etx::BSDFData light_data = data;
  light_data.path_source = etx::PathSource::Light;
  etx::BSDFData light_inside_data = inside_data;
  light_inside_data.path_source = etx::PathSource::Light;
  valid = (validate_delta_dielectric_transmission_sample("sapphire delta dielectric light outside", light_data, delta_sapphire_dielectric, 25540u) && valid);
  valid = (validate_delta_dielectric_transmission_sample("sapphire delta dielectric light inside", light_inside_data, delta_sapphire_dielectric, 25550u) && valid);
  const float exact_conductor_roughness = 0.5f;
  valid = validate_exact_energy_compensated_conductor_interface(scene, spectra, SpectrumCount, "mirror conductor exact interface", make_mirror_conductor(exact_conductor_roughness),
            exact_conductor_roughness, 33400u) &&
          valid;
  valid = validate_exact_energy_compensated_conductor_interface(scene, spectra, SpectrumCount, "mirror conductor thinfilm exact interface",
            make_thinfilm_rough_conductor(exact_conductor_roughness), exact_conductor_roughness, 33600u) &&
          valid;

  const float exact_dielectric_roughness_values[] = {0.25f, 0.5f, 0.75f, 1.0f};
  for (uint32_t i = 0u; i < 4u; ++i) {
    const float rough_sapphire_roughness = exact_dielectric_roughness_values[i];
    valid = validate_exact_energy_compensated_dielectric_interface(scene, spectra, SpectrumCount, "sapphire dielectric exact interface",
              make_white_sapphire_dielectric(rough_sapphire_roughness), rough_sapphire_roughness, 34600u + i * 1000u) &&
            valid;
  }
  valid = validate_exact_energy_compensated_dielectric_interface(scene, spectra, SpectrumCount, "sapphire dielectric thinfilm exact interface",
            make_thinfilm_rough_dielectric(0.5f), 0.5f, 38600u) &&
          valid;
  valid = validate_thinfilm_energy_compensation_cache_key(spectra, SpectrumCount) && valid;
  valid = validate_variable_thinfilm_texture_lut(spectra, SpectrumCount) && valid;
  valid = validate_openpbr_white_furnace(scene, spectra, SpectrumCount, 0.5f, 39000u) && valid;
  valid = validate_openpbr_parameter_sweeps(scene, spectra, SpectrumCount) && valid;

  etx::scene_global_clear(&scene);
  etx::scene_global_deinit();

  return valid ? 0 : 1;
}
