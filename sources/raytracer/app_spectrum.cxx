#include "app.hxx"

namespace etx {
bool RTApplication::on_spectrum_applied(const std::vector<SpectrumEdit>& edits) {
  SceneData& data = scene.data();
  if (edits.empty())
    return false;
  for (const auto& edit : edits) {
    if (edit.targets.empty())
      return false;
    const auto spectrum = edit.source.output();
    if (spectrum.empty() || (spectrum.valid() == false) || (valid_value(spectrum.integrated()) == false))
      return false;
    for (const auto& target : edit.targets) {
      const uint32_t* slot = target.spectrum_slot(data);
      if ((slot == nullptr) || (*slot != target.spectrum_index) || (*slot >= data.spectrum_values.size()))
        return false;
    }
  }
  const bool cpu_running = cpu_renderer.is_running();
  if (cpu_running)
    cpu_renderer.stop();
  bool materials = false;
  bool emitters = false;
  for (const auto& edit : edits) {
    const auto spectrum = edit.source.output();
    for (const auto& target : edit.targets) {
      const uint32_t index = *target.spectrum_slot(data);
      data.spectrum_values[index] = spectrum;
      data.spectrum_sources[index] = edit.source;
      materials |= target.material_index != kInvalidIndex;
      emitters |= target.emitter_index != kInvalidIndex;
    }
  }
  for (const auto& edit : edits) {
    for (const auto& target : edit.targets) {
      if (target.material_index < data.materials.size()) {
        Material& material = data.materials[target.material_index];
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
          default:
            break;
        }
        if (ior != nullptr) {
          const bool extinction = (ior->k_index < data.spectrum_values.size()) && (data.spectrum_values[ior->k_index].is_zero() == false);
          ior->cls = extinction ? SpectralDistribution::Conductor : SpectralDistribution::Dielectric;
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
  } else {
    notify_scene_might_have_changed();
    if (cpu_running)
      cpu_renderer.restart();
  }
  return true;
}
}  // namespace etx
