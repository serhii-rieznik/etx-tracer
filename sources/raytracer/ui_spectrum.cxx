#include "ui.hxx"
#include <imgui.h>
#include <cstring>

namespace etx {

void UI::edit_spectrum_button(SceneRepresentation& scene_rep, const char* label, SpectrumTarget::Channel channel, uint32_t spectrum_index) {
  const auto& spectra = scene_rep.data().spectrum_values;
  const bool available =
    (_editing_material_indices != nullptr) && (_editing_material_indices->size() == 1u) && (spectrum_index < spectra.size()) && (spectra[spectrum_index].empty() == false);
  ImGui::BeginDisabled(available == false);
  const bool edit = ImGui::SmallButton(label);
  ImGui::EndDisabled();
  if (edit == false) {
    return;
  }
  const uint32_t material_index = _editing_material_indices->front();
  _spectrum_target = {material_index, spectrum_index, channel};
  read_spectrum_source(scene_rep.data());
  _spectrum_live_preview = true;
  _spectrum_tab_requested = true;
  _ui_setup |= UIMemoryDiagnostics;
}

void UI::read_spectrum_source(const SceneData& scene_data) {
  const SpectralDistribution& spectrum = scene_data.spectrum_values[_spectrum_target.spectrum_index];
  _spectrum_curve.modified = false;
  _spectrum_curve.source_update_requested = false;
  _spectrum_curve.error.clear();
  auto& document = _spectrum_curve.document;
  document.classification = "reflectance";
  document.points.clear();
  document.points.reserve(spectrum.spectral_entry_count);
  for (uint32_t i = 0; i < spectrum.spectral_entry_count; ++i) {
    const auto& entry = spectrum.spectral_entries[i];
    document.points.push_back({entry.wavelength, entry.power});
  }
  const char* material_name = _material_mapping.name_for(_spectrum_target.material_index);
  document.title = material_name != nullptr ? material_name : "Material spectrum";
  switch (_spectrum_target.channel) {
    case SpectrumTarget::Channel::Emission:
      document.classification = "illuminant";
      break;
    case SpectrumTarget::Channel::InsideEta:
    case SpectrumTarget::Channel::OutsideEta:
    case SpectrumTarget::Channel::ThinfilmEta:
      document.classification = "dielectric";
      break;
    case SpectrumTarget::Channel::InsideK:
    case SpectrumTarget::Channel::OutsideK:
    case SpectrumTarget::Channel::ThinfilmK:
    case SpectrumTarget::Channel::Subsurface:
      document.classification.clear();
      break;
    default:
      break;
  }
  _spectrum_curve.select_point(0);
  _spectrum_curve.fit();
  _spectrum_preview_pending = false;
  _spectrum_source = spectrum;
}

void UI::build_spectrum_editor(SceneRepresentation& scene_rep) {
  SceneData& scene_data = scene_rep.data();
  if ((_spectrum_target.spectrum_index != kInvalidIndex) && (_spectrum_target.material_index < scene_data.materials.size())) {
    const uint32_t* const slot = _spectrum_target.spectrum_slot(scene_data.materials[_spectrum_target.material_index]);
    if ((slot == nullptr) || (*slot >= scene_data.spectrum_values.size()) || scene_data.spectrum_values[*slot].empty()) {
      _spectrum_target.spectrum_index = kInvalidIndex;
      _spectrum_preview_pending = false;
      _spectrum_curve.source_update_requested = false;
      _spectrum_curve.error = "The source spectrum is no longer available.";
    } else {
      const SpectralDistribution& source = scene_data.spectrum_values[*slot];
      if ((*slot != _spectrum_target.spectrum_index) || (source.spectral_entry_count != _spectrum_source.spectral_entry_count) ||
          (source.integrated_value.x != _spectrum_source.integrated_value.x) || (source.integrated_value.y != _spectrum_source.integrated_value.y) ||
          (source.integrated_value.z != _spectrum_source.integrated_value.z) ||
          (std::memcmp(source.spectral_entries, _spectrum_source.spectral_entries, source.spectral_entry_count * sizeof(SpectralDistribution::Entry)) != 0)) {
        _spectrum_target.spectrum_index = *slot;
        read_spectrum_source(scene_data);
      }
    }
  }
  const float spacing = ImGui::GetStyle().ItemSpacing.x;
  const float preview_width = ImGui::GetFontSize() * 23.0f;
  const bool side_by_side = ImGui::GetContentRegionAvail().x >= (preview_width + ImGui::GetFontSize() * 55.0f);
  const float curve_width = side_by_side ? ImGui::GetContentRegionAvail().x - preview_width - spacing : 0.0f;
  const float curve_height =
    side_by_side ? 0.0f : std::max(ImGui::GetFontSize() * 16.0f, ImGui::GetWindowHeight() - ImGui::GetFrameHeightWithSpacing() - 2.0f * ImGui::GetStyle().WindowPadding.y);
  ImGui::BeginChild("##spectrum_edit", ImVec2(curve_width, curve_height));
  const bool changed = _spectrum_curve.build();
  if (changed && (_spectrum_target.spectrum_index != kInvalidIndex)) {
    _spectrum_preview_pending = true;
  }
  ImGui::EndChild();
  if (side_by_side) {
    ImGui::SameLine();
  }
  ImGui::BeginChild("##spectrum_preview", ImVec2(0.0f, side_by_side ? 0.0f : ImGui::GetFontSize() * 23.0f));
  if (_spectrum_target.material_index >= scene_data.materials.size()) {
    _spectrum_target.material_index = kInvalidIndex;
    _spectrum_target.spectrum_index = kInvalidIndex;
    _spectrum_preview_pending = false;
    _spectrum_curve.source_update_requested = false;
  }
  ImGui::SeparatorText("Material assignment");
  ImGui::TextUnformatted("Material");
  const char* material_name = _material_mapping.name_for(_spectrum_target.material_index);
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::BeginCombo("##spectrum_material", material_name != nullptr ? material_name : "Select material")) {
    for (uint64_t i = 0; i < _material_mapping.size(); ++i) {
      const uint32_t index = _material_mapping.at(static_cast<int32_t>(i));
      ImGui::PushID(static_cast<int>(index));
      if (ImGui::Selectable(_material_mapping.name(static_cast<int32_t>(i)), index == _spectrum_target.material_index)) {
        _spectrum_target.material_index = index;
        _spectrum_target.spectrum_index = kInvalidIndex;
        _spectrum_preview_pending = false;
        _spectrum_curve.source_update_requested = false;
      }
      ImGui::PopID();
    }
    ImGui::EndCombo();
  }
  ImGui::TextUnformatted("Assignment target");
  constexpr const char* target_names[] = {"Select target", "Scattering (base / diffuse)", "Reflectance (specular)", "Emission", "IOR eta (inside)", "IOR k (inside)",
    "IOR eta (outside)", "IOR k (outside)", "Subsurface distance", "Thin-film eta", "Thin-film k"};
  int channel = static_cast<int>(_spectrum_target.channel);
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::Combo("##spectrum_assignment_target", &channel, target_names, static_cast<int>(std::size(target_names)))) {
    _spectrum_target.channel = static_cast<SpectrumTarget::Channel>(channel);
    _spectrum_target.spectrum_index = kInvalidIndex;
    _spectrum_preview_pending = false;
    _spectrum_curve.source_update_requested = false;
  }
  if (_spectrum_target.spectrum_index == kInvalidIndex) {
    _spectrum_curve.source_update_requested = false;
  }
  ImGui::Checkbox("Live preview", &_spectrum_live_preview);
  const bool can_apply = (_spectrum_target.material_index < scene_data.materials.size()) && (_spectrum_target.channel != SpectrumTarget::Channel::None) &&
                         static_cast<bool>(callbacks.spectrum_applied) && (preparation_active() == false);
  ImGui::BeginDisabled(can_apply == false);
  const bool apply = ImGui::Button("Assign", ImVec2(-FLT_MIN, 0.0f));
  ImGui::EndDisabled();
  const bool ior_target = (_spectrum_target.channel == SpectrumTarget::Channel::InsideEta) || (_spectrum_target.channel == SpectrumTarget::Channel::InsideK) ||
                          (_spectrum_target.channel == SpectrumTarget::Channel::OutsideEta) || (_spectrum_target.channel == SpectrumTarget::Channel::OutsideK) ||
                          (_spectrum_target.channel == SpectrumTarget::Channel::ThinfilmEta) || (_spectrum_target.channel == SpectrumTarget::Channel::ThinfilmK);
  const bool preview_ready = _spectrum_live_preview && _spectrum_preview_pending && ((ior_target == false) || (ImGui::IsMouseDown(ImGuiMouseButton_Left) == false));
  if (can_apply && (apply || _spectrum_curve.source_update_requested || preview_ready)) {
    if (_spectrum_curve.document.validate(_spectrum_curve.error)) {
      const SpectrumTarget result = callbacks.spectrum_applied(_spectrum_target, _spectrum_curve.document.distribution());
      if (result.spectrum_index != kInvalidIndex) {
        _spectrum_target = result;
        _spectrum_source = scene_data.spectrum_values[result.spectrum_index];
      } else {
        _spectrum_target.spectrum_index = kInvalidIndex;
        _spectrum_curve.error = "The target changed or assignment failed. Select the target and assign again.";
      }
    }
    _spectrum_preview_pending = false;
    _spectrum_curve.source_update_requested = false;
  }
  if (_spectrum_target.spectrum_index != kInvalidIndex) {
    ImGui::TextDisabled(
      _spectrum_live_preview ? "Live preview active" : (_spectrum_preview_pending ? "Preview paused; click Assign to update" : "Source up to date; live preview paused"));
  }
  ImGui::TextWrapped("Objects sharing this material update together.");
  ImGui::EndChild();
}

}  // namespace etx
