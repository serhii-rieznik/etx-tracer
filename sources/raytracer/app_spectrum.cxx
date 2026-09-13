#include "app.hxx"

namespace etx {

SpectrumTarget RTApplication::on_spectrum_applied(const SpectrumTarget& target, const SpectralDistribution& spectrum) {
  SceneData& data = scene.data();
  if ((target.material_index >= data.materials.size()) || spectrum.empty() || (spectrum.valid() == false) || (valid_value(spectrum.integrated()) == false)) {
    return {};
  }
  Material& material = data.materials[target.material_index];
  RefractiveIndex* ior = nullptr;
  uint32_t* const index = target.spectrum_slot(material);
  if (index == nullptr) {
    return {};
  }
  bool assigning_k = false;
  switch (target.channel) {
    case SpectrumTarget::Channel::ThinfilmEta:
    case SpectrumTarget::Channel::ThinfilmK:
      ior = &material.thinfilm.ior;
      assigning_k = target.channel == SpectrumTarget::Channel::ThinfilmK;
      break;
    case SpectrumTarget::Channel::InsideEta:
    case SpectrumTarget::Channel::InsideK:
      ior = &material.int_ior;
      assigning_k = target.channel == SpectrumTarget::Channel::InsideK;
      break;
    case SpectrumTarget::Channel::OutsideEta:
    case SpectrumTarget::Channel::OutsideK:
      ior = &material.ext_ior;
      assigning_k = target.channel == SpectrumTarget::Channel::OutsideK;
      break;
    default:
      break;
  }
  SpectralDistribution assigned = spectrum;
  if (ior != nullptr) {
    // Match load_refractive_index's representation for RGB IOR sampling.
    assigned.integrated_value = rgb_to_xyz(spectrum.integrated());
    if (valid_value(assigned.integrated()) == false) {
      return {};
    }
  }
  if ((target.spectrum_index != kInvalidIndex) && (*index != target.spectrum_index)) {
    return {};
  }
  if (cpu_renderer.is_running()) {
    cpu_renderer.stop();
    _restart_cpu_after_material_resource_preparation = true;
  }
  if (*index >= data.spectrum_values.size()) {
    *index = data.add_spectrum(assigned);
  } else {
    data.spectrum_values[*index] = assigned;
  }
  if ((ior != nullptr) && (assigning_k || (ior->cls == SpectralDistribution::Invalid))) {
    const bool has_extinction = (ior->k_index < data.spectrum_values.size()) && (data.spectrum_values[ior->k_index].is_zero() == false);
    ior->cls = has_extinction ? SpectralDistribution::Conductor : SpectralDistribution::Dielectric;
  }
  SpectrumTarget result = target;
  result.spectrum_index = *index;
  on_material_changed(result.material_index);
  return result;
}

}  // namespace etx