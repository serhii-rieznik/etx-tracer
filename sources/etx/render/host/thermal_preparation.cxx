#include <etx/core/log.hxx>
#include <etx/render/host/thermal_preparation.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/access/bsdf_resource_cpu.hxx>
#include <etx/render/interop/thermal_surface_shared.hxx>

namespace etx {
namespace {

bool thermal_interface_tint_is_unit(const SceneData& data, uint32_t index) {
  if (index >= data.spectrum_values.size()) {
    return false;
  }
  const SpectralDistribution& spectrum = data.spectrum_values[index];
  const auto found = data.spectrum_sources.find(index);
  if ((found != data.spectrum_sources.end()) && found->second.matches(spectrum)) {
    const SpectrumSource& source = found->second;
    if ((source.mode == SpectrumSource::Mode::Color) && (source.kind == SpectrumSource::Kind::Reflectance) && (source.color.x == 1.0f) && (source.color.y == 1.0f) &&
        (source.color.z == 1.0f) && (source.strength == 1.0f)) {
      return true;
    }
  }
  for (uint32_t i = 0u; i < WavelengthCount; ++i) {
    if (spectrum({kShortestWavelength + float(i), SpectralFlags::Spectral}).value != 1.0f) {
      return false;
    }
  }
  return true;
}

bool prepare_medium_emission(SceneData& data) {
  for (uint32_t index = 0u; index < data.mediums_vector.size(); ++index) {
    Medium& medium = data.mediums_vector[index];
    medium.emission_flags = 0u;
    if (medium.emission_index == kInvalidIndex) {
      continue;
    }
    if (medium.emission_index >= data.spectrum_values.size()) {
      log::error("Medium %u has an invalid emission spectrum", index);
      return false;
    }
    const SpectralDistribution& emission = data.spectrum_values[medium.emission_index];
    const float3 integrated = emission.integrated();
    if ((valid_value(integrated) == false) || (integrated.x < 0.0f) || (integrated.y < 0.0f) || (integrated.z < 0.0f)) {
      log::error("Medium %u requires finite, nonnegative emission", index);
      return false;
    }
    bool emits = false;
    for (uint32_t i = 0u; i < WavelengthCount; ++i) {
      const float value = emission({kShortestWavelength + float(i), SpectralFlags::Spectral}).value;
      if ((std::isfinite(value) == false) || (value < 0.0f)) {
        log::error("Medium %u requires finite, nonnegative emission", index);
        return false;
      }
      emits |= value > 0.0f;
    }
    if (emits == false) {
      continue;
    }
    medium.emission_flags = Medium::EmissionEnabled;
    if ((medium.absorption_index >= data.spectrum_values.size()) || (medium.scattering_index >= data.spectrum_values.size()) || (std::isfinite(medium.phase_function_g) == false) ||
        (std::abs(medium.phase_function_g) >= 1.0f)) {
      log::error("Emissive medium %u has invalid optical coefficients or anisotropy", index);
      return false;
    }
    for (uint32_t i = 0u; i < WavelengthCount; ++i) {
      const SpectralQuery spect = {kShortestWavelength + float(i), SpectralFlags::Spectral};
      const float absorption = data.spectrum_values[medium.absorption_index](spect).value;
      const float scattering = data.spectrum_values[medium.scattering_index](spect).value;
      if ((std::isfinite(absorption + scattering) == false) || (absorption < 0.0f) || (scattering < 0.0f)) {
        log::error("Emissive medium %u requires finite, nonnegative optical coefficients", index);
        return false;
      }
      if ((emission(spect).value > 0.0f) && (absorption == 0.0f)) {
        medium.emission_flags |= Medium::EmissionRequiresBoundedRegion;
      }
    }
    for (const Material& material : data.materials) {
      const bool mapped_coated_medium = (material.cls == MaterialClass::Plastic) && (material.subsurface_packing == 0.0f);
      if ((material.int_medium == index) && (material.subsurface_cls != SubsurfaceMaterial::Disabled) && (mapped_coated_medium == false)) {
        log::error("Emissive medium %u is unsupported by the subsurface diffusion model", index);
        return false;
      }
    }
  }
  return true;
}

bool prepare_thermal_medium(SceneData& data, Material& material, uint32_t material_index) {
  if (material.int_medium == kInvalidIndex) {
    return true;
  }
  if (material.int_medium >= data.mediums_vector.size()) {
    log::error("Thermal material %u has an invalid internal medium", material_index);
    return false;
  }
  SceneData::ThermalMediumResources& resources = data.thermal_medium_resources[material_index];
  uint32_t absorption_index = data.mediums_vector[material.int_medium].absorption_index;
  const auto author = data.spectrum_sources.find(material.int_ior.k_index);
  const bool bulk_profile = (material.cls == MaterialClass::Dielectric) && (author != data.spectrum_sources.end()) && (author->second.temperature_profile != nullptr) &&
                            (author->second.temperature_profile->cls == SpectralDistribution::Dielectric);
  if (bulk_profile) {
    const SpectrumSource& source = author->second;
    SpectralDistribution eta, extinction;
    if (source.temperature_profile->evaluate(source.temperature_profile_enabled ? material.temperature_kelvin : 0.0f, source.temperature_profile_hold_endpoints, eta, extinction) ==
        false) {
      return false;
    }
    float2 samples[WavelengthCount];
    for (uint32_t i = 0u; i < WavelengthCount; ++i) {
      const float wavelength = kShortestWavelength + float(i);
      samples[i] = {wavelength, extinction.spectral_entries[i].power * source.strength};
      if (std::isfinite(samples[i].y) == false) {
        log::error("Material %u optical absorption exceeds the supported numeric range", material_index);
        return false;
      }
    }
    const SpectralDistribution absorption = SpectralDistribution::from_samples(samples, WavelengthCount);
    if (resources.absorption_spectrum_index == kInvalidIndex) {
      resources.absorption_spectrum_index = data.add_spectrum(absorption);
    } else {
      data.spectrum_values[resources.absorption_spectrum_index] = absorption;
    }
    absorption_index = resources.absorption_spectrum_index;
  }
  for (uint32_t i = 0u; i < data.thermal_medium_states.size(); ++i) {
    const SceneData::ThermalMediumState& state = data.thermal_medium_states[i];
    if ((state.base_medium_index == material.int_medium) && (state.temperature_kelvin == material.temperature_kelvin) &&
        (state.absorption_spectrum_index == (bulk_profile ? absorption_index : kInvalidIndex))) {
      material.thermal_int_medium_index = uint32_t(data.mediums_vector.size()) + i;
      return true;
    }
  }
  const Medium& medium = data.mediums_vector[material.int_medium];
  if ((std::isfinite(medium.phase_function_g) == false) || (std::abs(medium.phase_function_g) >= 1.0f)) {
    log::error("Thermal material %u requires finite medium anisotropy strictly between -1 and 1", material_index);
    return false;
  }
  if (((medium.cls != Medium::Homogeneous) && (medium.cls != Medium::Heterogeneous)) || (medium.absorption_index >= data.spectrum_values.size()) ||
      (medium.scattering_index >= data.spectrum_values.size())) {
    log::error("Thermal material %u has an unsupported medium configuration", material_index);
    return false;
  }
  const SpectralDistribution& absorption = data.spectrum_values[absorption_index];
  const SpectralDistribution& scattering = data.spectrum_values[medium.scattering_index];
  uint64_t hash = etx_hash64(&material.temperature_kelvin, sizeof(material.temperature_kelvin));
  hash = etx_hash64_continue(&material.int_medium, sizeof(material.int_medium), hash);
  hash = etx_hash64_continue(&absorption, sizeof(absorption), hash);
  hash = etx_hash64_continue(&scattering, sizeof(scattering), hash);
  if ((resources.source_spectrum_index == kInvalidIndex) || (resources.input_hash != hash)) {
    float2 samples[WavelengthCount] = {};
    for (uint32_t i = 0u; i < WavelengthCount; ++i) {
      const float wavelength = kShortestWavelength + float(i);
      const SpectralQuery spect = {wavelength, SpectralFlags::Spectral};
      const float sigma_a = absorption(spect).value;
      const float sigma_s = scattering(spect).value;
      const float sigma_t = sigma_a + sigma_s;
      if ((std::isfinite(sigma_t) == false) || (sigma_a < 0.0f) || (sigma_s < 0.0f)) {
        log::error("Thermal material %u requires finite, nonnegative medium coefficients", material_index);
        return false;
      }
      // Camera transmission transports reduced radiance: physical j/n^2 = sigma_a * B_vacuum.
      // Uniform temperature and coefficient ratios give S = j/sigma_t and Le = S * (1 - T).
      const float source = (sigma_t > 0.0f) ? (sigma_a / sigma_t) * thermal_blackbody_radiance_nm(wavelength, material.temperature_kelvin) : 0.0f;
      if (std::isfinite(source) == false) {
        log::error("Thermal material %u has medium radiance outside the supported numeric range", material_index);
        return false;
      }
      samples[i] = {wavelength, source};
    }
    const SpectralDistribution source = SpectralDistribution::from_samples(samples, WavelengthCount);
    if (resources.source_spectrum_index == kInvalidIndex) {
      resources.source_spectrum_index = data.add_spectrum(source);
    } else {
      data.spectrum_values[resources.source_spectrum_index] = source;
    }
    resources.input_hash = hash;
  }
  uint32_t emission_flags = medium.emission_flags;
  if (bulk_profile && ((emission_flags & Medium::EmissionEnabled) != 0u)) {
    emission_flags &= ~Medium::EmissionRequiresBoundedRegion;
    const SpectralDistribution& emission = data.spectrum_values[medium.emission_index];
    const SpectralDistribution& effective_absorption = data.spectrum_values[absorption_index];
    for (uint32_t i = 0u; i < WavelengthCount; ++i) {
      const SpectralQuery spect = {kShortestWavelength + float(i), SpectralFlags::Spectral};
      if ((emission(spect).value > 0.0f) && (effective_absorption(spect).value == 0.0f)) {
        emission_flags |= Medium::EmissionRequiresBoundedRegion;
      }
    }
  }
  material.thermal_int_medium_index = uint32_t(data.mediums_vector.size() + data.thermal_medium_states.size());
  data.thermal_medium_states.push_back(
    {material.int_medium, resources.source_spectrum_index, material.temperature_kelvin, bulk_profile ? absorption_index : kInvalidIndex, emission_flags});
  return true;
}

uint64_t thermal_surface_scattering_input_hash(const SceneData& data, const Material& material) {
  uint64_t hash = etx_hash64(&material.cls, sizeof(material.cls));
  hash = etx_hash64_continue(&material.roughness.value, sizeof(material.roughness.value), hash);
  hash = etx_hash64_continue(&material.ext_ior.cls, sizeof(material.ext_ior.cls), hash);
  hash = etx_hash64_continue(&material.int_ior.cls, sizeof(material.int_ior.cls), hash);
  const uint32_t spectrum_indices[] = {material.scattering.spectrum_index, material.reflectance.spectrum_index, material.ext_ior.eta_index, material.ext_ior.k_index,
    material.int_ior.eta_index, material.int_ior.k_index};
  for (const uint32_t index : spectrum_indices) {
    hash = etx_hash64_continue(&index, sizeof(index), hash);
    if (index < data.spectrum_values.size()) {
      const SpectralDistribution& spectrum = data.spectrum_values[index];
      hash = etx_hash64_continue(&spectrum, sizeof(spectrum), hash);
    }
  }
  return hash;
}

struct ThermalQuadratureNode {
  double position;
  double weight;
};

std::vector<ThermalQuadratureNode> thermal_quadrature_rule(uint32_t count) {
  constexpr double pi = 3.14159265358979323846;
  std::vector<ThermalQuadratureNode> result(count);
  for (uint32_t i = 0u; i < ((count + 1u) / 2u); ++i) {
    double x = std::cos(pi * (double(i) + 0.75) / (double(count) + 0.5));
    double derivative = 0.0;
    for (;;) {
      double p0 = 1.0;
      double p1 = x;
      for (uint32_t order = 2u; order <= count; ++order) {
        const double p = ((2.0 * double(order) - 1.0) * x * p1 - (double(order) - 1.0) * p0) / double(order);
        p0 = p1;
        p1 = p;
      }
      derivative = double(count) * (x * p1 - p0) / (x * x - 1.0);
      const double delta = p1 / derivative;
      x -= delta;
      if (std::abs(delta) < 1.0e-15) {
        break;
      }
    }
    const double weight = 1.0 / ((1.0 - x * x) * derivative * derivative);
    result[i] = {(1.0 - x) * 0.5, weight};
    result[count - i - 1u] = {(1.0 + x) * 0.5, weight};
  }
  return result;
}

bool prepare_thermal_surface_reflection(SceneData& data, Material& material, SceneData::ThermalSurfaceResources& resources, uint64_t input_hash) {
  if (((material.cls != MaterialClass::Conductor) && (material.cls != MaterialClass::Plastic)) ||
      ((material.cls == MaterialClass::Conductor) && (max(material.roughness.value.x, material.roughness.value.y) <= kDeltaAlphaTreshold))) {
    return true;
  }
  if ((resources.conductor_image_index != kInvalidIndex) && (resources.scattering_input_hash == input_hash)) {
    material.thermal_conductor_image_index = resources.conductor_image_index;
    material.thermal_conductor_average_albedo = resources.conductor_average_albedo;
    return true;
  }
  constexpr double pi = 3.14159265358979323846;
  constexpr uint32_t level_count = 6u;
  static const std::array<std::vector<ThermalQuadratureNode>, level_count> rules = []() {
    std::array<std::vector<ThermalQuadratureNode>, level_count> result;
    for (uint32_t level = 0u; level < level_count; ++level) {
      result[level] = thermal_quadrature_rule(16u << level);
    }
    return result;
  }();
  Scene scene = {};
  scene.spectrums = {data.spectrum_values.data(), data.spectrum_values.size()};
  const BSDFResourceContext context = make_bsdf_resource_cpu_context(scene);
  std::array<::RefractiveIndexSample, WavelengthCount> external_iors;
  std::array<::RefractiveIndexSample, WavelengthCount> internal_iors;
  bool uniform_iors = true;
  for (uint32_t wavelength_index = 0u; wavelength_index < WavelengthCount; ++wavelength_index) {
    const ::SpectralQuery spect = {kShortestWavelength + float(wavelength_index), SpectralFlags::Spectral};
    external_iors[wavelength_index] = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, spect);
    internal_iors[wavelength_index] = bsdf_resource_evaluate_refractive_index(context, material.int_ior, spect);
    const ::RefractiveIndexSample& int_ior = internal_iors[wavelength_index];
    if ((material.cls == MaterialClass::Plastic) && ((std::isfinite(int_ior.eta.value) == false) || (int_ior.eta.value <= 0.0f) || (int_ior.k.value != 0.0f))) {
      log::error("Thermal Plastic requires a lossless dielectric coating with real positive IOR; choose a dielectric IOR preset");
      return false;
    }
    uniform_iors = uniform_iors && (external_iors[wavelength_index].eta.value == external_iors[0].eta.value) &&
                   (external_iors[wavelength_index].k.value == external_iors[0].k.value) && (int_ior.eta.value == internal_iors[0].eta.value) &&
                   (int_ior.k.value == internal_iors[0].k.value);
  }
  const uint32_t ior_sample_count = uniform_iors ? 1u : WavelengthCount;
  const double alpha = bsdf_energy_compensated_scalar_roughness_from_value({material.roughness.value.x, material.roughness.value.y});
  const ::ThinfilmEval no_film = {};
  std::vector<float4> pixels(size_t(kThermalConductorDirectionCount) * WavelengthCount);
  std::array<bool, kThermalConductorDirectionCount> failed = {};
  data.scheduler.execute(kThermalConductorDirectionCount, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t direction_index = begin; direction_index < end; ++direction_index) {
      const double axis = double(direction_index) / double(kThermalConductorDirectionCount - 1u);
      const double mu = axis * axis;
      const double sin_i = std::sqrt(1.0 - mu * mu);
      const double g_i = std::sqrt(mu * mu + alpha * alpha * sin_i * sin_i);
      std::array<double, WavelengthCount> previous = {};
      std::array<double, WavelengthCount> integral = {};
      double previous_geometry = 0.0;
      double geometric_integral = 0.0;
      double previous_visibility = 0.0;
      double visible_probability = 0.0;
      bool converged = false;
      for (uint32_t level = 0u; level < level_count; ++level) {
        integral.fill(0.0);
        geometric_integral = 0.0;
        visible_probability = 0.0;
        for (const ThermalQuadratureNode& azimuth : rules[level]) {
          for (uint32_t half = 0u; half < 2u; ++half) {
            const double phi_axis = azimuth.position * azimuth.position;
            const double phi = 0.5 * pi * ((half == 0u) ? (1.0 - phi_axis) : (1.0 + phi_axis));
            const double a = sin_i * std::cos(phi);
            double q_max = a > 0.0 ? 1.0 : 0.0;
            double q_max_complement = 1.0 - q_max;
            if (mu > 0.0) {
              const double root = std::sqrt(a * a + mu * mu);
              const double t_max = (a >= 0.0) ? (a + root) / mu : mu / (root - a);
              const double denominator = alpha * alpha + t_max * t_max;
              q_max = t_max * t_max / denominator;
              q_max_complement = alpha * alpha / denominator;
            }
            if (q_max == 0.0) {
              continue;
            }
            for (const ThermalQuadratureNode& polar : rules[level]) {
              // q = tan(theta_m)^2 / (alpha^2 + tan(theta_m)^2) cancels the GGX NDF Jacobian.
              // Squaring the distance to pi/2 resolves the grazing tail; q_max bounds the reflected hemisphere.
              const double axis_complement = 1.0 - polar.position;
              const double theta_complement = 0.5 * pi * axis_complement * axis_complement;
              const double sine = std::cos(theta_complement);
              const double cosine = std::sin(theta_complement);
              const double q = q_max * sine * sine;
              const double t = alpha * std::sqrt(q / (q_max_complement + q_max * cosine * cosine));
              const double w_o_z = 2.0 * (mu + a * t) / (1.0 + t * t) - mu;
              const double g_weight = 2.0 * (mu + a * t) * w_o_z / (g_i * w_o_z + mu * std::sqrt(w_o_z * w_o_z + alpha * alpha * (1.0 - w_o_z * w_o_z)));
              const double measure_weight = azimuth.weight * azimuth.position * polar.weight * q_max * 2.0 * pi * axis_complement * sine * cosine;
              const double weight = measure_weight * g_weight;
              geometric_integral += weight;
              visible_probability += measure_weight * 2.0 * (mu + a * t) / (mu + g_i);
              const float fresnel_cosine = float((mu + a * t) / std::sqrt(1.0 + t * t));
              for (uint32_t wavelength_index = 0u; wavelength_index < ior_sample_count; ++wavelength_index) {
                const ::SpectralQuery spect = {kShortestWavelength + float(wavelength_index), SpectralFlags::Spectral};
                integral[wavelength_index] +=
                  weight * bsdf_fresnel_calculate(spect, fresnel_cosine, external_iors[wavelength_index], internal_iors[wavelength_index], no_film).value;
              }
            }
          }
        }
        double maximum_error = std::max(std::abs(geometric_integral - previous_geometry), std::abs(visible_probability - previous_visibility));
        for (uint32_t wavelength_index = 0u; wavelength_index < ior_sample_count; ++wavelength_index) {
          if (std::isfinite(integral[wavelength_index]) == false) {
            log::error("Thermal conductor has non-finite directional scattering");
            failed[direction_index] = true;
            return;
          }
          maximum_error = std::max(maximum_error, std::abs(integral[wavelength_index] - previous[wavelength_index]));
        }
        if ((level > 0u) && (maximum_error <= 1.0e-7)) {
          converged = true;
          break;
        }
        previous = integral;
        previous_geometry = geometric_integral;
        previous_visibility = visible_probability;
      }
      if (converged == false) {
        log::error("Thermal conductor directional scattering did not converge at cosine %.9g", mu);
        failed[direction_index] = true;
        return;
      }
      for (uint32_t wavelength_index = 0u; wavelength_index < WavelengthCount; ++wavelength_index) {
        pixels[size_t(wavelength_index) * kThermalConductorDirectionCount + direction_index] = {float(integral[uniform_iors ? 0u : wavelength_index]), float(geometric_integral),
          float(visible_probability), 0.0f};
      }
    }
  });
  if (std::any_of(failed.begin(), failed.end(), [](bool value) {
        return value;
      })) {
    return false;
  }
  if (resources.conductor_image_index == kInvalidIndex) {
    resources.conductor_image_index = data.images.add_from_data(pixels.data(), {kThermalConductorDirectionCount, WavelengthCount}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f});
  } else if (data.buffer_pool.write(data.images_vector[resources.conductor_image_index].data, pixels.data(), pixels.size() * sizeof(float4), 0u) == false) {
    log::error("Unable to update thermal conductor scattering table");
    return false;
  }
  if (resources.conductor_image_index == kInvalidIndex) {
    log::error("Unable to allocate thermal conductor scattering table");
    return false;
  }
  resources.scattering_input_hash = input_hash;
  material.thermal_conductor_image_index = resources.conductor_image_index;
  scene.images = {data.images_vector.data(), data.images_vector.size()};
  const BSDFResourceContext prepared_context = make_bsdf_resource_cpu_context(scene);
  const ::SpectralQuery geometry_spect = {kShortestWavelength, SpectralFlags::Spectral};
  const auto average_rule = thermal_quadrature_rule(4u);
  // Cubic interpolation in sqrt(mu), multiplied by 4*sqrt(mu)^3, is degree six on each interval.
  double directional_average = 0.0;
  constexpr double step = 1.0 / double(kThermalConductorDirectionCount - 1u);
  for (uint32_t i = 0u; (i + 1u) < kThermalConductorDirectionCount; ++i) {
    for (const ThermalQuadratureNode& node : average_rule) {
      const double axis = (double(i) + node.position) * step;
      const double geometric_albedo = bsdf_energy_compensated_conductor_prepared_albedo(prepared_context, geometry_spect, material, float(axis * axis)).y;
      directional_average += node.weight * step * 4.0 * axis * axis * axis * geometric_albedo;
    }
  }
  resources.conductor_average_albedo = float(directional_average);
  material.thermal_conductor_average_albedo = resources.conductor_average_albedo;
  return true;
}

}  // namespace

bool thermal_medium_binding_valid(const SceneData& data, uint32_t index) {
  if ((index != kInvalidIndex) && (data.transport_medium_index(index) == kInvalidIndex)) {
    log::error("Medium %u has interiors at different temperatures or optical coefficients; external and camera bindings require a distinct authored medium", index);
    return false;
  }
  return true;
}

bool prepare_thermal_materials(SceneData& data, uint32_t camera_medium_index) {
  data.thermal_medium_states.clear();
  if (prepare_medium_emission(data) == false) {
    return false;
  }
  for (uint32_t index = 0u; index < data.materials.size(); ++index) {
    Material& material = data.materials[index];
    material.thermal_rgb_image_index = kInvalidIndex;
    material.thermal_emission_weight = 0.0f;
    material.thermal_int_medium_index = kInvalidIndex;
    material.thermal_conductor_image_index = kInvalidIndex;
    if ((std::isfinite(material.temperature_kelvin) == false) || (material.temperature_kelvin < 0.0f)) {
      log::error("Material %u has an invalid thermal temperature", index);
      return false;
    }
    if (material.temperature_kelvin == 0.0f) {
      const auto source = data.spectrum_sources.find(material.int_ior.k_index);
      if ((material.cls == MaterialClass::Dielectric) && (source != data.spectrum_sources.end()) && (source->second.temperature_profile != nullptr) &&
          (source->second.temperature_profile->cls == SpectralDistribution::Dielectric) && (prepare_thermal_medium(data, material, index) == false)) {
        return false;
      }
      continue;
    }
    if ((material.cls == MaterialClass::Dielectric) || (material.cls == MaterialClass::Boundary)) {
      if ((material.subsurface_cls != SubsurfaceMaterial::Disabled) || (material.opacity != 1.0f) || bsdf_resource_thinfilm_enabled(material.thinfilm) ||
          (material.normal_image_index != kInvalidIndex) ||
          (material.reflectance.image_index != kInvalidIndex) || (material.scattering.image_index != kInvalidIndex)) {
        log::error("Thermal material %u requires a lossless dielectric interface or medium boundary", index);
        return false;
      }
      if (material.cls == MaterialClass::Dielectric) {
        if ((thermal_interface_tint_is_unit(data, material.reflectance.spectrum_index) == false) ||
            (thermal_interface_tint_is_unit(data, material.scattering.spectrum_index) == false)) {
          log::error("Thermal material %u requires unit interface reflection/transmission tint; place absorption in its internal medium", index);
          return false;
        }
        if (data.thermal_unit_spectrum_index == kInvalidIndex) {
          data.thermal_unit_spectrum_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
        }
      }
      Scene scene = {};
      scene.spectrums = {data.spectrum_values.data(), data.spectrum_values.size()};
      const BSDFResourceContext context = make_bsdf_resource_cpu_context(scene);
      for (uint32_t i = 0u; i < WavelengthCount; ++i) {
        const ::SpectralQuery spect = {kShortestWavelength + float(i), SpectralFlags::Spectral};
        const ::RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, spect);
        const ::RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, spect);
        if ((std::isfinite(ext_ior.eta.value) == false) || (ext_ior.eta.value <= 0.0f) || (ext_ior.k.value != 0.0f) ||
            ((material.cls == MaterialClass::Dielectric) && ((std::isfinite(int_ior.eta.value) == false) || (int_ior.eta.value <= 0.0f) || (int_ior.k.value != 0.0f)))) {
          log::error("Thermal material %u requires real positive IORs", index);
          return false;
        }
      }
      if (prepare_thermal_medium(data, material, index) == false) {
        return false;
      }
      continue;
    }
    if (((material.cls != MaterialClass::Diffuse) && (material.cls != MaterialClass::Conductor) && (material.cls != MaterialClass::Plastic)) ||
        (material.subsurface_cls != SubsurfaceMaterial::Disabled) || (material.opacity != 1.0f) || (material.normal_image_index != kInvalidIndex) ||
        (material.roughness.image_index != kInvalidIndex) || (material.scattering.image_index != kInvalidIndex) || (material.reflectance.image_index != kInvalidIndex) ||
        bsdf_resource_thinfilm_enabled(material.thinfilm) || (bsdf_energy_compensated_roughness_isotropic({material.roughness.value.x, material.roughness.value.y}) == false)) {
      log::error("Material %u has no supported thermal surface model for its scattering configuration", index);
      return false;
    }
    if (((material.cls == MaterialClass::Diffuse) || (material.cls == MaterialClass::Plastic)) && (material.scattering.spectrum_index < data.spectrum_values.size())) {
      const SpectralDistribution& albedo = data.spectrum_values[material.scattering.spectrum_index];
      for (uint32_t wavelength_index = 0u; wavelength_index < WavelengthCount; ++wavelength_index) {
        const float value = albedo({kShortestWavelength + float(wavelength_index), SpectralFlags::Spectral}).value;
        if ((std::isfinite(value) == false) || (value < 0.0f) || (value > 1.0f)) {
          log::error("Thermal material %u requires finite diffuse-layer spectral albedo in [0, 1]", index);
          return false;
        }
      }
    }
    const uint64_t scattering_input_hash = thermal_surface_scattering_input_hash(data, material);
    const uint64_t input_hash = etx_hash64_continue(&material.temperature_kelvin, sizeof(material.temperature_kelvin), scattering_input_hash);
    auto resource_it = data.thermal_surface_resources.find(index);
    if ((resource_it != data.thermal_surface_resources.end()) && (resource_it->second.input_hash == input_hash)) {
      const SceneData::ThermalSurfaceResources& resources = resource_it->second;
      material.thermal_rgb_image_index = resources.rgb_image_index;
      material.thermal_conductor_image_index = resources.conductor_image_index;
      material.thermal_conductor_average_albedo = resources.conductor_average_albedo;
      material.thermal_emission_weight = resources.emission_weight;
      continue;
    }

    SceneData::ThermalSurfaceResources& resources = data.thermal_surface_resources[index];
    if (prepare_thermal_surface_reflection(data, material, resources, scattering_input_hash) == false) {
      return false;
    }

    Scene scene = {};
    scene.spectrums = {data.spectrum_values.data(), data.spectrum_values.size()};
    scene.images = {data.images_vector.data(), data.images_vector.size()};
    scene.energy_compensation_interfaces = {data.energy_compensation_interfaces.data(), data.energy_compensation_interfaces.size()};
    const BSDFResourceContext context = make_bsdf_resource_cpu_context(scene);
    float4 pixels[kThermalSurfaceDirectionCount] = {};
    for (uint32_t direction_index = 0u; direction_index < kThermalSurfaceDirectionCount; ++direction_index) {
      const float mu = float(direction_index) / float(kThermalSurfaceDirectionCount - 1u);
      float3 xyz = {};
      for (uint32_t wavelength_index = 0u; wavelength_index < WavelengthCount; ++wavelength_index) {
        ::SpectralQuery spect = {};
        spect.flags = SpectralFlags::Spectral;
        spect.wavelength = kShortestWavelength + float(wavelength_index);
        const float reflectance = thermal_surface_directional_reflectance(context, spect, material, mu);
        const ::RefractiveIndexSample external_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, spect);
        if ((std::isfinite(external_ior.eta.value) == false) || (external_ior.eta.value <= 0.0f) || (external_ior.k.value != 0.0f)) {
          log::error("Thermal material %u requires a positive, real external refractive index", index);
          return false;
        }
        const float blackbody = thermal_surface_equilibrium_radiance(context, spect, material);
        if ((std::isfinite(reflectance) == false) || (reflectance < 0.0f) || (reflectance > 1.0005f) || (std::isfinite(blackbody) == false)) {
          log::error("Material %u has invalid thermal radiance or non-passive scattering (reflectance %.9g, cosine %.9g, wavelength %.1f nm)", index, reflectance, mu,
            spect.wavelength);
          return false;
        }
        const float radiance = blackbody * max(0.0f, 1.0f - reflectance);
        const float endpoint_weight = ((wavelength_index == 0u) || ((wavelength_index + 1u) == WavelengthCount)) ? 0.5f : 1.0f;
        xyz += spectral_response_to_xyz(spectral_response_make(spect, radiance)) * endpoint_weight;
      }
      const float3 rgb = max(spectral_xyz_to_rgb(xyz), float3{0.0f, 0.0f, 0.0f});
      if ((valid_value(rgb) == false) || (std::isfinite(luminance(rgb)) == false)) {
        log::error("Material %u has thermal radiance outside the supported numeric range", index);
        return false;
      }
      pixels[direction_index] = {rgb.x, rgb.y, rgb.z, 0.0f};
    }

    if (resources.rgb_image_index == kInvalidIndex) {
      resources.rgb_image_index = data.images.add_from_data(pixels, {kThermalSurfaceDirectionCount, 1u}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f});
    } else if (data.buffer_pool.write(data.images_vector[resources.rgb_image_index].data, pixels, sizeof(pixels), 0u) == false) {
      log::error("Unable to update thermal emission table for material %u", index);
      return false;
    }
    if (resources.rgb_image_index == kInvalidIndex) {
      log::error("Unable to allocate thermal emission table for material %u", index);
      return false;
    }
    resources.input_hash = input_hash;
    resources.conductor_average_albedo = material.thermal_conductor_average_albedo;
    float maximum_weight = 0.0f;
    for (const float4& pixel : pixels) {
      maximum_weight = max(maximum_weight, luminance(float3{pixel.x, pixel.y, pixel.z}));
    }
    resources.emission_weight = maximum_weight;
    material.thermal_rgb_image_index = resources.rgb_image_index;
    material.thermal_emission_weight = resources.emission_weight;
  }
  for (const Material& material : data.materials) {
    if (thermal_medium_binding_valid(data, material.ext_medium) == false) {
      return false;
    }
  }
  for (const EmitterProfile& profile : data.emitter_profiles) {
    if (thermal_medium_binding_valid(data, profile.medium_index) == false) {
      return false;
    }
  }
  for (const SceneData::CameraInfo& camera : data.cameras) {
    if (thermal_medium_binding_valid(data, camera.cam.medium_index) == false) {
      return false;
    }
  }
  return thermal_medium_binding_valid(data, camera_medium_index);
}

}  // namespace etx
