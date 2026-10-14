#include "app.hxx"

namespace etx {
bool RTApplication::on_spectrum_applied(const std::vector<SpectrumEdit>& edits) {
  SceneData& data = scene.data();
  if (edits.empty())
    return false;
  data.diagnostic = {.phase = "Spectrum edit"};
  _material_edit_diagnostic = {};
  _dismissed_diagnostic_message.clear();
  const auto reject_spectrum = [&](const SpectrumTarget& target, const char* field, const char* format, auto... args) {
    if (target.material_index != kInvalidIndex) {
      data.fail_material(target.material_index, field, format, args...);
    } else if (target.medium_index != kInvalidIndex) {
      data.fail_medium(target.medium_index, field, format, args...);
    } else {
      if (target.emitter_index != kInvalidIndex)
        data.diagnostic.resource_name = "Emitter #" + std::to_string(target.emitter_index);
      data.diagnostic.fail(SceneDiagnosticKind::Validation, field, format, args...);
    }
    _material_edit_diagnostic = data.diagnostic;
    return false;
  };
  std::vector<SpectralDistribution> spectra;
  spectra.reserve(edits.size());
  for (const auto& edit : edits) {
    if (edit.targets.empty())
      return reject_spectrum(SpectrumTarget{}, "Target", "The spectrum has no bound property. Reopen the property's spectrum editor.");
    const auto spectrum = edit.source.output();
    if (spectrum.empty() || (spectrum.valid() == false) || (valid_value(spectrum.integrated()) == false)) {
      const SpectrumTarget& target = edit.targets.front();
      return reject_spectrum(target, "Spectrum", "Source '%s' contains empty, non-finite or unsupported spectral values. The edit was not applied.", edit.source.title.c_str());
    }
    for (const auto& target : edit.targets) {
      const uint32_t* slot = target.spectrum_slot(data);
      if ((slot == nullptr) || (*slot != target.spectrum_index) || (*slot >= data.spectrum_values.size())) {
        return reject_spectrum(target, "Spectrum binding", "Spectrum #%u no longer belongs to the selected property. Reopen the property's spectrum editor.",
          target.spectrum_index);
      }
    }
    spectra.push_back(spectrum);
  }
  const auto proposed_spectrum = [&](uint32_t index) -> const SpectralDistribution* {
    for (size_t remaining = edits.size(); remaining > 0u; --remaining) {
      const size_t edit_index = remaining - 1u;
      for (const SpectrumTarget& target : edits[edit_index].targets) {
        if (target.spectrum_index == index)
          return &spectra[edit_index];
      }
    }
    return index < data.spectrum_values.size() ? &data.spectrum_values[index] : nullptr;
  };
  const auto proposed_source = [&](uint32_t index) -> const SpectrumSource* {
    for (size_t remaining = edits.size(); remaining > 0u; --remaining) {
      const SpectrumEdit& edit = edits[remaining - 1u];
      for (const SpectrumTarget& target : edit.targets) {
        if (target.spectrum_index == index)
          return &edit.source;
      }
    }
    const auto source = data.spectrum_sources.find(index);
    return source != data.spectrum_sources.end() ? &source->second : nullptr;
  };
  const auto proposed_ior_class = [&](const MaterialDescription& material, const RefractiveIndex& ior) -> SpectralDistribution::Class {
    const SpectrumSource* eta_source = proposed_source(ior.eta_index);
    const SpectrumSource* k_source = proposed_source(ior.k_index);
    if ((&ior == &material.int_ior) && (eta_source != nullptr) && (k_source != nullptr) && (eta_source->temperature_profile != nullptr) &&
        (k_source->temperature_profile != nullptr)) {
      return eta_source->temperature_profile->cls;
    }
    const SpectralDistribution* k = proposed_spectrum(ior.k_index);
    if ((k != nullptr) && (k->is_zero() == false)) {
      return SpectralDistribution::Conductor;
    }
    if (k_source != nullptr) {
      if (k_source->classification == "conductor")
        return SpectralDistribution::Conductor;
      if (k_source->classification == "dielectric")
        return SpectralDistribution::Dielectric;
    }
    const bool conductor_substrate = ((material.cls == MaterialClass::Conductor) || (material.cls == MaterialClass::CoatedConductor)) && (&ior == &material.int_ior);
    if (conductor_substrate) {
      return SpectralDistribution::Conductor;
    }
    const bool conductor_coating = (material.layers.explicit_layers != 0u) && (&ior == &material.coating.ior) && (ior.cls == SpectralDistribution::Conductor);
    return conductor_coating ? SpectralDistribution::Conductor : SpectralDistribution::Dielectric;
  };
  const bool cpu_running = cpu_renderer.is_running();
  if (cpu_running)
    cpu_renderer.stop();
  bool materials = _material_render_resource_preparation_active;
  bool emitters = false;
  for (const SpectrumEdit& edit : edits) {
    for (const SpectrumTarget& target : edit.targets) {
      materials |= target.material_index != kInvalidIndex;
      emitters |= target.emitter_index != kInvalidIndex;
    }
  }
  for (size_t edit_index = 0u; edit_index < edits.size(); ++edit_index) {
    const SpectrumEdit& edit = edits[edit_index];
    const SpectralDistribution& spectrum = spectra[edit_index];
    for (const auto& target : edit.targets) {
      const uint32_t index = *target.spectrum_slot(data);
      if (materials) {
        data.remember_material_spectrum_edit(index);
      }
      data.spectrum_values[index] = spectrum;
      data.spectrum_sources[index] = edit.source;
    }
  }
  for (const auto& edit : edits) {
    for (const auto& target : edit.targets) {
      if (target.material_index < data.material_descriptions.size()) {
        MaterialDescription& material = data.material_descriptions[target.material_index];
        RefractiveIndex* ior = nullptr;
        switch (target.channel) {
          case SpectrumTarget::Channel::InsideEta:
          case SpectrumTarget::Channel::InsideK:
            ior = &material.int_ior;
            break;
          case SpectrumTarget::Channel::OutsideEta:
          case SpectrumTarget::Channel::OutsideK:
            ior = &material.ext_ior;
            break;
          case SpectrumTarget::Channel::ThinfilmEta:
          case SpectrumTarget::Channel::ThinfilmK:
            ior = &material.thinfilm.ior;
            break;
          case SpectrumTarget::Channel::CoatingEta:
          case SpectrumTarget::Channel::CoatingK:
            data.author_material_layers(target.material_index);
            ior = &material.coating.ior;
            break;
          case SpectrumTarget::Channel::CoatingAbsorption:
          case SpectrumTarget::Channel::CoatingScattering:
            data.author_material_layers(target.material_index);
            break;
          default:
            break;
        }
        if (ior != nullptr) {
          ior->cls = proposed_ior_class(material, *ior);
          if ((ior == &material.int_ior) && (material.layers.explicit_layers != 0u) && (material.layers.bsdf == MaterialClass::Undefined) &&
              ((material.cls == MaterialClass::Conductor) || (material.cls == MaterialClass::Dielectric))) {
            material.cls = ior->cls == SpectralDistribution::Conductor ? MaterialClass::Conductor : MaterialClass::Dielectric;
          }
        }
      }
    }
  }
  mark_scene_dirty();
  scene.update_medium_bounds();
  if (emitters)
    rebuild_all_atmosphere_emitters();
  if (materials) {
    _restart_cpu_after_material_resource_preparation |= cpu_running;
    on_material_changed(kInvalidIndex);
    return (_material_render_resource_preparation_failed == false) && (_material_edit_diagnostic.rejected() == false);
  } else {
    notify_scene_might_have_changed();
    if (cpu_running)
      cpu_renderer.restart();
  }
  return true;
}
}  // namespace etx
