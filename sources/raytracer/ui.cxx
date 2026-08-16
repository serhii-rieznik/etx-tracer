#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>

#include <etx/render/host/film.hxx>
#include <etx/render/host/openpbr_material_loader.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/host/scene_serialization.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/render/shared/math.hxx>
#include <etx/render/shared/scene_medium.hxx>

#include "ui.hxx"
#include <etx/engine/camera_controller.hxx>

#include <imgui.h>
#include <imgui_internal.h>

#include <etx/rhi/rhi_types.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/rhi_imgui.hxx>

#if (ETX_PLATFORM_APPLE)
# include <unistd.h>
#endif

#include <map>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdarg>
#include <cctype>
#include <cstdio>
#include <filesystem>

namespace etx {

namespace {

const ImVec4 kOptionStringColor(1.0f, 0.5f, 0.25f, 1.0f);

const ImVec4 kIorPickerColors[] = {
  ImVec4(0.3333f, 0.3333f, 0.3333f, 1.0f),
  ImVec4(1.0f, 1.0f, 1.0f, 1.0f),
  ImVec4(0.5f, 0.75f, 1.0f, 1.0f),
  ImVec4(1.0f, 0.75f, 0.5f, 1.0f),
  ImVec4(1.0f, 0.75f, 1.0f, 1.0f),
};

const ImVec4 kToolbarLaunchColor(0.11f, 0.44f, 0.11f, 1.0f);
const ImVec4 kToolbarFinishColor(0.44f, 0.33f, 0.11f, 1.0f);
const ImVec4 kToolbarTerminateColor(0.44f, 0.11f, 0.11f, 1.0f);
const ImVec4 kToolbarRestartColor(0.11f, 0.22f, 0.44f, 1.0f);
const ImVec4 kToolbarDisabledColor(0.25f, 0.25f, 0.25f, 1.0f);

const ImVec4 kHistoryButtonBaseColor(0.2f, 0.6f, 0.2f, 1.0f);
const ImVec4 kHistoryButtonHoverColor(0.25f, 0.75f, 0.25f, 1.0f);
const ImVec4 kHistoryButtonActiveColor(0.15f, 0.5f, 0.15f, 1.0f);

const ImVec4 kDeleteButtonColor(0.44f, 0.11f, 0.11f, 1.0f);
const ImVec4 kDeleteButtonHoverColor(0.54f, 0.21f, 0.21f, 1.0f);
const ImVec4 kDeleteButtonActiveColor(0.64f, 0.31f, 0.31f, 1.0f);

const ImVec4 kErrorTextColor(1.0f, 0.0f, 0.0f, 1.0f);

const ImVec4 kSceneTextColor(0.30f, 0.45f, 0.90f, 1.0f);
const ImVec4 kCameraTextColor(0.90f, 0.80f, 0.35f, 1.0f);

const ImVec4 kMaterialHeaderPrimaryColor(0.96f, 0.79f, 0.45f, 1.0f);
const ImVec4 kMaterialHeaderSpecializedColor(0.54f, 0.80f, 0.98f, 1.0f);
const ImVec4 kMaterialHeaderInterfacesColor(0.88f, 0.68f, 0.97f, 1.0f);

enum MaterialBatchChangedField : uint32_t {
  MaterialBatchChangedClass = 1u << 0u,
  MaterialBatchChangedIntIOR = 1u << 1u,
  MaterialBatchChangedExtIOR = 1u << 2u,
  MaterialBatchChangedThinfilmIOR = 1u << 3u,
};

inline void decrease_exposure(ViewParameters& o) {
  o.exposure = fmaxf(1.0f / 1024.0f, 0.5f * o.exposure);
}

inline void increase_exposure(ViewParameters& o) {
  o.exposure = fmaxf(1.0f / 1024.0f, 2.0f * o.exposure);
}

template <class T>
inline auto hash_mapping(const T& m) -> uint64_t {
  uint64_t h = 0;
  for (const auto& kv : m) {
    h = etx_hash64_continue(kv.first.c_str(), h);
    h = etx_hash64_continue(reinterpret_cast<const uint8_t*>(&kv.second), sizeof(kv.second), h);
  }
  return h;
};

std::string duration_string(double seconds) {
  if (seconds < 0.0) {
    return "-";
  }

  const uint64_t total_seconds = static_cast<uint64_t>(seconds + 0.5);
  const uint64_t hours = total_seconds / 3600ull;
  const uint64_t minutes = (total_seconds / 60ull) % 60ull;
  const uint64_t remaining_seconds = total_seconds % 60ull;

  char buffer[64] = {};
  if (hours > 0ull) {
    snprintf(buffer, sizeof(buffer), "%lluh %02llum %02llus", static_cast<unsigned long long>(hours), static_cast<unsigned long long>(minutes),
      static_cast<unsigned long long>(remaining_seconds));
  } else if (minutes > 0ull) {
    snprintf(buffer, sizeof(buffer), "%llum %02llus", static_cast<unsigned long long>(minutes), static_cast<unsigned long long>(remaining_seconds));
  } else {
    snprintf(buffer, sizeof(buffer), "%llus", static_cast<unsigned long long>(remaining_seconds));
  }
  return std::string(buffer);
}

const char* material_class_display_name(const Material::Class cls) {
  switch (cls) {
    case MaterialClass::Diffuse:
      return "Diffuse";
    case MaterialClass::Translucent:
      return "Translucent";
    case MaterialClass::Plastic:
      return "Plastic";
    case MaterialClass::Conductor:
      return "Conductor";
    case MaterialClass::Dielectric:
      return "Dielectric";
    case MaterialClass::Thinfilm:
      return "Thinfilm";
    case MaterialClass::Mirror:
      return "Mirror";
    case MaterialClass::Boundary:
      return "Boundary";
    case MaterialClass::Velvet:
      return "Velvet";
    case MaterialClass::OpenPBR:
      return "OpenPBR";
    case MaterialClass::Void:
      return "Void";
    case MaterialClass::DiffractionGrating:
      return "Diffraction Grating";
    default:
      return "Undefined";
  }
}

const char* spectral_distribution_class_display_name(SpectralDistribution::Class cls) {
  switch (cls) {
    case SpectralDistribution::Conductor:
      return "Conductor";
    case SpectralDistribution::Dielectric:
      return "Dielectric";
    case SpectralDistribution::Illuminant:
      return "Illuminant";
    case SpectralDistribution::Reflectance:
      return "Reflectance";
    default:
      return "Invalid";
  }
}

float3 spectral_distribution_display_rgb(const SpectralDistribution& distribution) {
  if (distribution.spectral_entry_count == 0u) {
    return max(distribution.integrated(), float3{});
  }

  return max(xyz_to_rgb(distribution.integrate_to_xyz()), float3{});
}

void draw_ior_tooltip(const char* label, const char* title, SpectralDistribution::Class cls, const float3& eta_rgb, const float3& k_rgb) {
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal) == false) {
    return;
  }

  const float tooltip_width = ImGui::GetFontSize() * 34.0f;
  ImGui::SetNextWindowSize(ImVec2(tooltip_width, 0.0f), ImGuiCond_Always);
  ImGui::BeginTooltip();
  ImGui::Text("%s", label);
  ImGui::Separator();
  ImGui::Text("Name: %s", title);
  ImGui::Text("Class: %s", spectral_distribution_class_display_name(cls));
  ImGui::Text("Eta RGB: %.6f  %.6f  %.6f", eta_rgb.x, eta_rgb.y, eta_rgb.z);
  ImGui::Text("K RGB:   %.6f  %.6f  %.6f", k_rgb.x, k_rgb.y, k_rgb.z);
  ImGui::EndTooltip();
}

}  // namespace

void UI::full_width_item() {
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
}

bool UI::labeled_control(const char* label, std::function<bool()>&& control_func) {
  ImGui::Text("%s:", label);
  full_width_item();
  return control_func();
}

bool UI::validated_float_control(const char* label, float& value, float min_val, float max_val, const char* format) {
  float original_value = value;
  bool changed = labeled_control(label, [&]() {
    return ImGui::DragFloat(("##" + std::string(label)).c_str(), &value, 0.1f, min_val, max_val, format);
  });
  if (changed) {
    value = std::clamp(value, min_val, max_val);
    if (!std::isfinite(value)) {
      value = original_value;
      changed = false;
    }
  }
  return changed;
}

bool UI::validated_int_control(const char* label, int32_t& value, int32_t min_val, int32_t max_val) {
  int32_t original_value = value;
  bool changed = labeled_control(label, [&]() {
    return ImGui::InputInt(("##" + std::string(label)).c_str(), &value);
  });
  if (changed) {
    value = std::clamp(value, min_val, max_val);
  }
  return changed;
}

const char* UI::format_string(const char* format, ...) {
  static char buffer[1024];
  va_list args;
  va_start(args, format);
  vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  return buffer;
}

void UI::MappingRepresentation::build(const std::unordered_map<std::string, uint32_t>& in_mapping) {
  std::vector<std::pair<std::string, uint32_t>> unfold;
  unfold.reserve(in_mapping.size());
  for (const auto& m : in_mapping) {
    if ((m.first.starts_with("etx::") == false) && (m.first.starts_with("et::") == false)) {
      unfold.emplace_back(m.first, m.second);
    }
  }
  std::sort(unfold.begin(), unfold.end(), [](const auto& a, const auto& b) {
    std::string a_lower = a.first;
    std::string b_lower = b.first;
    std::transform(a_lower.begin(), a_lower.end(), a_lower.begin(), ::tolower);
    std::transform(b_lower.begin(), b_lower.end(), b_lower.begin(), ::tolower);
    return a_lower < b_lower;
  });

  entries.clear();
  entries.reserve(unfold.size());
  reverse.clear();
  reverse.reserve(unfold.size());

  size_t total_length = 0;
  for (const auto& entry : unfold) {
    total_length += entry.first.size() + 1u;
  }
  data.resize(total_length);

  char* ptr = data.data();
  size_t offset = 0;
  for (auto& m : unfold) {
    const char* name_ptr = ptr + offset;
    uint32_t entry_index = static_cast<uint32_t>(entries.size());
    entries.push_back(Entry{m.second, name_ptr});
    reverse.emplace(m.second, entry_index);
    size_t len = m.first.size();
    std::memcpy(ptr + offset, m.first.c_str(), len + 1u);
    offset += len + 1u;
  }
}

void UI::update_name_buffer(SelectionKind kind, int32_t index, const char* current_name) {
  if ((_name_edit_selection.kind != kind) || (_name_edit_selection.index != index)) {
    _name_edit_selection = {kind, index};
    std::snprintf(_name_edit_buffer, sizeof(_name_edit_buffer), "%s", (current_name != nullptr) ? current_name : "");
  }
}

void UI::validate_selections(SceneRepresentation& scene_rep) {
  if (_selected_material_positions.empty() == false) {
    _selected_material_positions.erase(std::remove_if(_selected_material_positions.begin(), _selected_material_positions.end(),
                                         [&](int32_t index) {
                                           return (index < 0) || (static_cast<uint64_t>(index) >= _material_mapping.size());
                                         }),
      _selected_material_positions.end());
  }

  switch (_selection.kind) {
    case SelectionKind::Material:
      if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= _material_mapping.size())) {
        set_selection(SelectionKind::Rendering, 0, false);
      } else if (_selected_material_positions.empty()) {
        _selected_material_positions.push_back(_selection.index);
        _material_selection_anchor = _selection.index;
      } else if (material_list_position_selected(_selection.index) == false) {
        _selection.index = _selected_material_positions.back();
      }
      break;
    case SelectionKind::Medium:
      if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= _medium_mapping.size())) {
        set_selection(SelectionKind::Rendering, 0, false);
      }
      break;
    case SelectionKind::Mesh:
      if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= _mesh_mapping.size())) {
        set_selection(SelectionKind::Rendering, 0, false);
      }
      break;
    case SelectionKind::Emitter:
      if ((_selection.index < 0) || (static_cast<uint32_t>(_selection.index) >= scene_rep.data().emitter_profiles.size())) {
        set_selection(SelectionKind::Rendering, 0, false);
      }
      break;
    default:
      break;
  }
}

void UI::set_selection(SelectionKind kind, int32_t index, bool track_history) {
  SelectionState next{kind, index};
  if ((_selection.kind == next.kind) && (_selection.index == next.index)) {
    if ((kind == SelectionKind::Material) && (_updating_material_multi_selection == false)) {
      _selected_material_positions.clear();
      if (index >= 0) {
        _selected_material_positions.push_back(index);
      }
      _material_selection_anchor = index;
    }
    return;
  }

  _selection = next;

  if (kind == SelectionKind::Material) {
    if (_updating_material_multi_selection == false) {
      _selected_material_positions.clear();
      if (index >= 0) {
        _selected_material_positions.push_back(index);
      }
      _material_selection_anchor = index;
    }
  } else {
    _selected_material_positions.clear();
    _material_selection_anchor = -1;
  }

  if (track_history == false) {
    return;
  }

  if (_selection_history_cursor >= 0) {
    size_t next_cursor = static_cast<size_t>(_selection_history_cursor + 1);
    if (next_cursor < _selection_history.size()) {
      _selection_history.erase(_selection_history.begin() + next_cursor, _selection_history.end());
    }
  } else {
    _selection_history.clear();
  }

  _selection_history.emplace_back(next);
  _selection_history_cursor = static_cast<int32_t>(_selection_history.size()) - 1;
}

bool UI::can_navigate_back() const {
  return (_selection_history_cursor > 0) && (_selection_history.size() > 1);
}

bool UI::can_navigate_forward() const {
  return (_selection_history_cursor >= 0) && (static_cast<size_t>(_selection_history_cursor + 1) < _selection_history.size());
}

uint32_t UI::selected_material_count() const {
  return static_cast<uint32_t>(_selected_material_positions.size());
}

bool UI::material_list_position_selected(int32_t index) const {
  return std::find(_selected_material_positions.begin(), _selected_material_positions.end(), index) != _selected_material_positions.end();
}

void UI::set_single_material_selection(int32_t index, bool track_history) {
  _selected_material_positions.clear();
  if (index >= 0) {
    _selected_material_positions.push_back(index);
  }
  _material_selection_anchor = index;
  _updating_material_multi_selection = true;
  set_selection(SelectionKind::Material, index, track_history);
  _updating_material_multi_selection = false;
}

void UI::toggle_material_selection(int32_t index) {
  if (index < 0) {
    return;
  }

  if (_selection.kind != SelectionKind::Material) {
    set_single_material_selection(index, true);
    return;
  }

  auto found = std::find(_selected_material_positions.begin(), _selected_material_positions.end(), index);
  if (found == _selected_material_positions.end()) {
    _selected_material_positions.push_back(index);
    _material_selection_anchor = index;
  } else if (_selected_material_positions.size() > 1u) {
    _selected_material_positions.erase(found);
  }

  const int32_t active_index = material_list_position_selected(index) ? index : _selected_material_positions.back();
  _updating_material_multi_selection = true;
  set_selection(SelectionKind::Material, active_index, true);
  _updating_material_multi_selection = false;
}

void UI::set_material_selection_range(int32_t index) {
  if (index < 0) {
    return;
  }

  if ((_selection.kind != SelectionKind::Material) || (_material_selection_anchor < 0)) {
    set_single_material_selection(index, true);
    return;
  }

  const int32_t range_begin = std::min(_material_selection_anchor, index);
  const int32_t range_end = std::max(_material_selection_anchor, index);
  _selected_material_positions.clear();
  for (int32_t i = range_begin; i <= range_end; ++i) {
    if (static_cast<uint64_t>(i) < _material_mapping.size()) {
      _selected_material_positions.push_back(i);
    }
  }

  _updating_material_multi_selection = true;
  set_selection(SelectionKind::Material, index, true);
  _updating_material_multi_selection = false;
}

std::vector<uint32_t> UI::selected_material_indices(SceneRepresentation& scene_rep) const {
  std::vector<uint32_t> result;
  result.reserve(_selected_material_positions.size());
  for (const int32_t position : _selected_material_positions) {
    if ((position >= 0) && (static_cast<uint64_t>(position) < _material_mapping.size())) {
      const uint32_t material_index = _material_mapping.at(position);
      if (material_index < scene_rep.data().materials.size()) {
        result.push_back(material_index);
      }
    }
  }
  return result;
}

void UI::apply_material_changes(SceneRepresentation& scene_rep, const std::vector<uint32_t>& material_indices, const Material& before, const Material& after) const {
  auto bytes_equal = [](const auto& a, const auto& b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
  };

  auto apply_field = [&](const auto& before_value, const auto& after_value, auto setter) {
    if (bytes_equal(before_value, after_value)) {
      return false;
    }

    for (const uint32_t material_index : material_indices) {
      if (material_index < scene_rep.data().materials.size()) {
        setter(scene_rep.data().materials[material_index], after_value);
      }
    }
    return true;
  };

  apply_field(before.reflectance, after.reflectance, [](Material& material, const SpectralImage& value) {
    material.reflectance = value;
  });
  apply_field(before.scattering, after.scattering, [](Material& material, const SpectralImage& value) {
    material.scattering = value;
  });
  apply_field(before.emission, after.emission, [](Material& material, const SpectralImage& value) {
    material.emission = value;
  });
  apply_field(before.subsurface, after.subsurface, [](Material& material, const SpectralImage& value) {
    material.subsurface = value;
  });
  apply_field(before.roughness, after.roughness, [](Material& material, const SampledImage& value) {
    material.roughness = value;
  });
  apply_field(before.metalness, after.metalness, [](Material& material, const SampledImage& value) {
    material.metalness = value;
  });
  apply_field(before.transmission, after.transmission, [](Material& material, const SampledImage& value) {
    material.transmission = value;
  });
  apply_field(before.thinfilm.ior.cls, after.thinfilm.ior.cls, [](Material& material, const uint32_t value) {
    material.thinfilm.ior.cls = value;
  });
  apply_field(before.thinfilm.thinkness_image, after.thinfilm.thinkness_image, [](Material& material, const uint32_t value) {
    material.thinfilm.thinkness_image = value;
  });
  apply_field(before.thinfilm.min_thickness, after.thinfilm.min_thickness, [](Material& material, const float value) {
    material.thinfilm.min_thickness = value;
  });
  apply_field(before.thinfilm.max_thickness, after.thinfilm.max_thickness, [](Material& material, const float value) {
    material.thinfilm.max_thickness = value;
  });
  apply_field(before.thinfilm.weight, after.thinfilm.weight, [](Material& material, const float value) {
    material.thinfilm.weight = value;
  });
  apply_field(before.diffraction_grating.period_nm, after.diffraction_grating.period_nm, [](Material& material, const float value) {
    material.diffraction_grating.period_nm = value;
  });
  apply_field(before.diffraction_grating.optical_path_difference_nm, after.diffraction_grating.optical_path_difference_nm, [](Material& material, const float value) {
    material.diffraction_grating.optical_path_difference_nm = value;
  });
  apply_field(before.diffraction_grating.duty_cycle, after.diffraction_grating.duty_cycle, [](Material& material, const float value) {
    material.diffraction_grating.duty_cycle = value;
  });
  apply_field(before.diffraction_grating.rotation, after.diffraction_grating.rotation, [](Material& material, const float value) {
    material.diffraction_grating.rotation = value;
  });
  apply_field(before.ext_ior.cls, after.ext_ior.cls, [](Material& material, const uint32_t value) {
    material.ext_ior.cls = value;
  });
  apply_field(before.int_ior.cls, after.int_ior.cls, [](Material& material, const uint32_t value) {
    material.int_ior.cls = value;
  });
  apply_field(before.subsurface_cls, after.subsurface_cls, [](Material& material, const uint32_t value) {
    material.subsurface_cls = value;
  });
  apply_field(before.subsurface_path, after.subsurface_path, [](Material& material, const uint32_t value) {
    material.subsurface_path = value;
  });
  apply_field(before.cls, after.cls, [](Material& material, const uint32_t value) {
    material.cls = value;
  });
  apply_field(before.int_medium, after.int_medium, [](Material& material, const uint32_t value) {
    material.int_medium = value;
  });
  apply_field(before.ext_medium, after.ext_medium, [](Material& material, const uint32_t value) {
    material.ext_medium = value;
  });
  apply_field(before.normal_image_index, after.normal_image_index, [](Material& material, const uint32_t value) {
    material.normal_image_index = value;
  });
  apply_field(before.two_sided, after.two_sided, [](Material& material, const uint32_t value) {
    material.two_sided = value;
  });
  apply_field(before.normal_scale, after.normal_scale, [](Material& material, const float value) {
    material.normal_scale = value;
  });
  apply_field(before.opacity, after.opacity, [](Material& material, const float value) {
    material.opacity = value;
  });
  apply_field(before.emission_collimation, after.emission_collimation, [](Material& material, const float value) {
    material.emission_collimation = value;
  });
  apply_field(before.energy_compensation_interface_index, after.energy_compensation_interface_index, [](Material& material, const uint32_t value) {
    material.energy_compensation_interface_index = value;
  });
}

void UI::navigate_history(int32_t step) {
  if ((_selection_history.empty()) || (step == 0)) {
    return;
  }

  int32_t target = std::clamp(_selection_history_cursor + step, 0, static_cast<int32_t>(_selection_history.size()) - 1);
  if (target == _selection_history_cursor) {
    return;
  }

  _selection_history_cursor = target;
  const auto& state = _selection_history[_selection_history_cursor];
  set_selection(state.kind, state.index, false);
}

bool UI::build_options(Options& options) {
  bool global_changed = false;
  for (auto& option : options.options) {
    bool changed = false;
    switch (option.cls) {
      case Option::Class::String: {
        auto& data = option.as<Option::Class::String>();
        ImGui::TextColored(kOptionStringColor, "%s", data.value.c_str());
        break;
      };
      case Option::Class::Boolean: {
        auto& data = option.as<Option::Class::Boolean>();
        changed = ImGui::Checkbox(option.description.c_str(), &data.value);
        break;
      }
      case Option::Class::Float: {
        auto& data = option.as<Option::Class::Float>();
        const std::string& id_source = option.id.empty() ? option.description : option.id;
        std::string label = "##" + id_source;
        std::string fmt = option.description.empty() ? "%.3f" : option.description + " %.3f";
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        if (data.bounds.maximum > data.bounds.minimum) {
          changed = ImGui::DragFloat(label.c_str(), &data.value, 0.1f, data.bounds.minimum, data.bounds.maximum, fmt.c_str(), ImGuiSliderFlags_AlwaysClamp);
        } else {
          changed = ImGui::InputFloat(label.c_str(), &data.value, 0.0f, 0.0f, fmt.c_str());
        }
        break;
      }
      case Option::Class::Integral: {
        auto& data = option.as<Option::Class::Integral>();
        if (option.meta & Option::Meta::EnumValue) {
          ETX_CRITICAL(data.bounds.maximum > data.bounds.minimum);
          ETX_CRITICAL(option.name_getter);
          int32_t original = data.value;
          if (ImGui::TreeNodeEx(option.description.c_str(), ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed)) {
            for (int32_t i = data.bounds.minimum; i <= data.bounds.maximum; ++i) {
              if (ImGui::RadioButton(option.name_getter(i).c_str(), &data.value, i)) {
                data.value = i;
              }
            }
            ImGui::TreePop();
          }
          changed = data.value != original;
        } else {
          const std::string& id_source = option.id.empty() ? option.description : option.id;
          std::string label = "##" + id_source;
          std::string fmt = option.description.empty() ? "%d" : option.description + " %d";
          ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
          if (data.bounds.maximum > data.bounds.minimum) {
            changed = ImGui::DragInt(label.c_str(), &data.value, 0.1f, data.bounds.minimum, data.bounds.maximum, fmt.c_str(), ImGuiSliderFlags_AlwaysClamp);
          } else {
            changed = ImGui::InputInt(label.c_str(), &data.value, 0, 0);
          }
        }
        break;
      }
      case Option::Class::Float3: {
        auto& data = option.as<Option::Class::Float3>();
        ImGui::Text("%s", option.description.c_str());
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        const char* buffer_name = format_string("##%s", option.description.c_str());
        changed = ImGui::DragFloat3(buffer_name, &data.value.x, 0.1f, data.bounds.minimum.x, data.bounds.maximum.x, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        break;
      }
      default:
        ETX_FAIL("Invalid option");
    }
    global_changed = global_changed || changed;
  }

  return global_changed;
}

bool UI::angle_editor(const char* label, float2& angles, float min_azimuth, float max_azimuth, float min_elevation, float max_elevation, float pole_threshold) {
  if ((min_azimuth >= max_azimuth) || (min_elevation >= max_elevation) || (pole_threshold <= 0.0f) || (pole_threshold >= 90.0f)) {
    ImGui::TextColored(kErrorTextColor, "Invalid angle editor parameters");
    return false;
  }

  bool changed = false;

  ImGui::Text("%s", label);
  ImGui::PushID(label);

  float azimuth_deg = angles.x * 180.0f / kPi;
  float elevation_deg = angles.y * 180.0f / kPi;

  if (!std::isfinite(azimuth_deg) || !std::isfinite(elevation_deg)) {
    ImGui::TextColored(kErrorTextColor, "Invalid angle values detected");
    ImGui::PopID();
    return false;
  }

  float clamped_min_elevation = max(min_elevation, -pole_threshold);
  float clamped_max_elevation = min(max_elevation, pole_threshold);
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  if (ImGui::SliderFloat("##elevation", &elevation_deg, clamped_min_elevation, clamped_max_elevation, "Elevation: %.1f°")) {
    angles.y = std::clamp(elevation_deg * kPi / 180.0f, clamped_min_elevation * kPi / 180.0f, clamped_max_elevation * kPi / 180.0f);
    if (!std::isfinite(angles.y))
      angles.y = 0.0f;
    changed = true;
  }

  bool near_pole = (std::abs(elevation_deg) >= pole_threshold);

  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  if (near_pole) {
    ImGui::BeginDisabled();
  }
  if (ImGui::SliderFloat("##azimuth", &azimuth_deg, min_azimuth, max_azimuth, "Azimuth: %.1f°")) {
    angles.x = std::clamp(azimuth_deg * kPi / 180.0f, min_azimuth * kPi / 180.0f, max_azimuth * kPi / 180.0f);
    if (!std::isfinite(angles.x))
      angles.x = 0.0f;
    changed = true;
  }
  if (near_pole) {
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
      ImGui::SetTooltip("Azimuth control disabled near poles to prevent gimbal lock");
    }
  }

  ImGui::PopID();
  return changed;
}

bool UI::ior_picker(SceneRepresentation& scene, const char* name, RefractiveIndex& ior, const FrameData& data) {
  return ior_picker(scene, name, ior, data, false, false);
}

bool UI::ior_picker(SceneRepresentation& scene, const char* name, RefractiveIndex& ior, const FrameData& data, bool mixed, bool dielectric_only) {
  bool changed = false;
  bool load_from_file = false;

  int matched_index = -1;
  static const SpectralDistribution null_spectrum = SpectralDistribution::constant(0.0f);
  const SpectralDistribution* tooltip_eta = nullptr;
  const SpectralDistribution* tooltip_k = &null_spectrum;
  if (ior.cls != SpectralDistribution::Invalid) {
    const SpectralDistribution& current_eta = scene.data().spectrum_values[ior.eta_index];
    const SpectralDistribution& current_k = (ior.k_index != kInvalidIndex) ? scene.data().spectrum_values[ior.k_index] : null_spectrum;
    tooltip_eta = &current_eta;
    tooltip_k = &current_k;
    matched_index = data.ior_database.find_matching_index(current_eta, current_k, ior.cls);
  }

  const char* preview_text = name;
  const char* tooltip_title = name;
  float3 eta_rgb = {};
  float3 k_rgb = {};
  SpectralDistribution::Class tooltip_class = ior.cls;
  if (tooltip_eta != nullptr) {
    eta_rgb = spectral_distribution_display_rgb(*tooltip_eta);
    k_rgb = spectral_distribution_display_rgb(*tooltip_k);
  }
  if (mixed) {
    preview_text = "mixed";
    tooltip_title = "mixed";
    tooltip_class = SpectralDistribution::Invalid;
  } else if ((matched_index >= 0) && (matched_index < static_cast<int>(data.ior_database.definitions.size()))) {
    const IORDefinition& matched_definition = data.ior_database.definitions[static_cast<size_t>(matched_index)];
    preview_text = matched_definition.title.c_str();
    tooltip_title = matched_definition.title.c_str();
    eta_rgb = spectral_distribution_display_rgb(matched_definition.eta);
    k_rgb = spectral_distribution_display_rgb(matched_definition.k);
    tooltip_class = matched_definition.cls;
  }
  std::string button_label = std::string(preview_text) + "##ior_" + name;
  const char* popup_id = format_string("ior_popup##%s", name);

  if (ImGui::Button(button_label.c_str(), ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
    ImGui::OpenPopup(popup_id);
  }
  draw_ior_tooltip(name, tooltip_title, tooltip_class, eta_rgb, k_rgb);

  ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 32.0f, 0.0f), ImGuiCond_Always);
  if (ImGui::BeginPopup(popup_id)) {
    if (data.ior_database.definitions.empty() == false) {
      struct ColumnInfo {
        SpectralDistribution::Class cls;
        const char* title;
      };

      std::vector<ColumnInfo> columns;
      const auto& conductors = data.ior_database.class_entries(SpectralDistribution::Conductor);
      if ((dielectric_only == false) && (conductors.empty() == false)) {
        columns.push_back({SpectralDistribution::Conductor, "Conductors"});
      }
      const auto& dielectrics = data.ior_database.class_entries(SpectralDistribution::Dielectric);
      if (dielectrics.empty() == false) {
        columns.push_back({SpectralDistribution::Dielectric, "Dielectrics"});
      }

      if (columns.empty()) {
        ImGui::TextDisabled("No predefined IORs available");
        ImGui::Separator();
      } else {
        ImGui::Columns(static_cast<int>(columns.size()), nullptr, true);

        auto draw_column = [&](SpectralDistribution::Class cls, const char* title) {
          ImGui::PushStyleColor(ImGuiCol_Text, kIorPickerColors[uint32_t(cls)]);
          ImGui::Text("%s", title);
          ImGui::PopStyleColor();
          const auto& entries = data.ior_database.class_entries(cls);
          for (size_t idx : entries) {
            if (idx >= data.ior_database.definitions.size())
              continue;
            const IORDefinition& def = data.ior_database.definitions[idx];
            int def_index = static_cast<int>(idx);
            bool is_current = (matched_index == def_index);
            if (ImGui::Selectable(def.title.c_str(), is_current)) {
              matched_index = def_index;
              ior.cls = def.cls;
              ETX_CRITICAL(ior.eta_index != kInvalidIndex);
              scene.data().spectrum_values[ior.eta_index] = def.eta;
              ETX_CRITICAL(ior.k_index != kInvalidIndex);
              scene.data().spectrum_values[ior.k_index] = def.k;
              changed = true;
              ImGui::CloseCurrentPopup();
            }
            if (is_current) {
              ImGui::SetItemDefaultFocus();
            }
          }
        };

        for (size_t c = 0; c < columns.size(); ++c) {
          draw_column(columns[c].cls, columns[c].title);
          if (c + 1 < columns.size()) {
            ImGui::NextColumn();
          }
        }

        ImGui::Columns(1);
        ImGui::Separator();
      }
    } else {
      ImGui::TextDisabled("No predefined IORs available");
      ImGui::Separator();
    }
    if (ImGui::Selectable("Load from file...", false)) {
      load_from_file = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  if (load_from_file) {
    auto filename = open_file("spd");
    std::string title = {};
    SpectralDistribution t_eta = {};
    SpectralDistribution t_k = {};
    auto cls = SpectralDistribution::load_refractive_index(filename.c_str(), t_eta, t_k, title);
    if ((cls != SpectralDistribution::Invalid) && ((dielectric_only == false) || (cls == SpectralDistribution::Dielectric))) {
      ior.cls = cls;
      ETX_CRITICAL(ior.eta_index != kInvalidIndex);
      scene.data().spectrum_values[ior.eta_index] = t_eta;
      ETX_CRITICAL(ior.k_index != kInvalidIndex);
      scene.data().spectrum_values[ior.k_index] = t_k;
      changed = true;
    }
  }

  return changed;
}

bool UI::emission_picker(SceneRepresentation& scene, const char* label, const char* id_suffix, uint32_t& spectrum_index, const FrameData& data) {
  if (scene.data().spectrum_values.empty())
    return false;

  bool changed = false;
  bool load_from_file = false;

  const char* base_label = (label != nullptr) ? label : "Emission";
  const char* unique_id = (id_suffix != nullptr) ? id_suffix : base_label;

  std::string color_name = std::string(base_label) + "_Color_" + unique_id;
  char editor_key_buf[32] = {};
  snprintf(editor_key_buf, sizeof(editor_key_buf), "%p", scene.data().spectrum_values.data() + spectrum_index);
  std::string editor_key = std::string(editor_key_buf);
  auto [state_it, inserted] = _spectrum_editors.emplace(editor_key, SpectrumEditorState{});
  SpectrumEditorState& editor_state = state_it->second;

  static const char* kModeLabels[] = {
    "Color",
    "Temperature",
    "Preset",
  };

  ImGui::Text("%s", base_label);
  SpectrumEditorState::Mode previous_mode = editor_state.mode;
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  if (ImGui::BeginCombo("##emission_mode", kModeLabels[static_cast<uint32_t>(editor_state.mode)])) {
    for (uint32_t i = 0; i < 3; ++i) {
      bool selected = (editor_state.mode == static_cast<SpectrumEditorState::Mode>(i));
      if (ImGui::Selectable(kModeLabels[i], selected)) {
        editor_state.mode = static_cast<SpectrumEditorState::Mode>(i);
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  if (editor_state.mode == SpectrumEditorState::Mode::Temperature) {
    if (editor_state.mode != previous_mode) {
      SpectralDistribution temp_spd = SpectralDistribution::from_normalized_black_body(editor_state.temperature, editor_state.scale);
      scene.data().spectrum_values[spectrum_index] = temp_spd;
      changed = true;
    }
    const char* temperature_label = format_string("##emission_temp_%s", unique_id);
    float temperature = editor_state.temperature;
    if (ImGui::InputFloat(temperature_label, &temperature, 100.0f, 1000.0f, "%.0f")) {
      temperature = std::clamp(temperature, 1000.0f, 40000.0f);
      editor_state.temperature = temperature;
      SpectralDistribution temp_spd = SpectralDistribution::from_normalized_black_body(temperature, editor_state.scale);
      scene.data().spectrum_values[spectrum_index] = temp_spd;
      editor_state.color = {};
      changed = true;
    }
  }
  changed |= spectrum_picker(scene, color_name.c_str(), spectrum_index, true, true, editor_state.mode == SpectrumEditorState::Mode::Color, true);

  int matched_index = -1;
  if ((spectrum_index < scene.data().spectrum_values.size())) {
    static const SpectralDistribution null_spectrum = SpectralDistribution::constant(0.0f);
    matched_index = data.ior_database.find_matching_index(scene.data().spectrum_values[spectrum_index], null_spectrum, SpectralDistribution::Illuminant);
  }

  const char* preview_text = base_label;
  std::string preview_storage;
  if ((matched_index >= 0) && (matched_index < static_cast<int>(data.ior_database.definitions.size()))) {
    preview_storage = data.ior_database.definitions[static_cast<size_t>(matched_index)].title;
    preview_text = preview_storage.c_str();
  } else {
    preview_text = "Select Preset";
  }

  const char* popup_id = format_string("emission_popup##%s", unique_id);

  if (editor_state.mode == SpectrumEditorState::Mode::Preset) {
    std::string button_label = std::string(preview_text) + "##emission_" + unique_id;
    if (ImGui::Button(button_label.c_str(), ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
      ImGui::OpenPopup(popup_id);
    }
  }

  ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 28.0f, 0.0f), ImGuiCond_Always);
  if (ImGui::BeginPopup(popup_id)) {
    if (ImGui::Selectable("None", false)) {
      scene.data().spectrum_values[spectrum_index] = SpectralDistribution::constant(0.0f);
      matched_index = -1;
      changed = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::Separator();

    bool has_presets = false;
    if (data.ior_database.definitions.empty() == false) {
      const auto& entries = data.ior_database.class_entries(SpectralDistribution::Illuminant);
      has_presets = entries.empty() == false;

      if (has_presets) {
        size_t column_count = std::min<size_t>(2, std::max<size_t>(1, entries.size()));
        size_t per_column = (entries.size() + column_count - 1u) / column_count;

        std::string table_id = std::string("emission_presets_##") + unique_id;
        if (ImGui::BeginTable(table_id.c_str(), static_cast<int>(column_count), ImGuiTableFlags_SizingStretchSame)) {
          for (size_t col = 0; col < column_count; ++col) {
            ImGui::TableNextColumn();
            size_t start = col * per_column;
            size_t end = min(start + per_column, entries.size());
            for (size_t idx = start; idx < end; ++idx) {
              size_t def_index = entries[idx];
              if (def_index >= data.ior_database.definitions.size())
                continue;
              const IORDefinition& def = data.ior_database.definitions[def_index];
              bool is_current = (matched_index == static_cast<int>(def_index));
              if (ImGui::Selectable(def.title.c_str(), is_current)) {
                scene.data().spectrum_values[spectrum_index] = def.eta;
                scene.data().spectrum_values[spectrum_index].scale(editor_state.scale);
                matched_index = static_cast<int>(def_index);
                changed = true;
                ImGui::CloseCurrentPopup();
              }
              if (is_current) {
                ImGui::SetItemDefaultFocus();
              }
            }
          }
          ImGui::EndTable();
        }
      }
    }

    if (has_presets == false) {
      ImGui::TextDisabled("No emission presets available");
    }

    if (ImGui::Selectable("Load from file...", false)) {
      load_from_file = true;
      ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
  }

  if (load_from_file) {
    auto filename = open_file("spd");
    if (filename.empty() == false) {
      std::string title = {};
      SpectralDistribution loaded = {};
      auto cls = SpectralDistribution::load_from_file(filename.c_str(), loaded, nullptr, false, title);
      if (cls != SpectralDistribution::Invalid) {
        scene.data().spectrum_values[spectrum_index] = loaded;
        scene.data().spectrum_values[spectrum_index].scale(editor_state.scale);
        matched_index = -1;
        changed = true;
      } else {
        log::warning("Failed to load emission spectrum from `%s`", filename.c_str());
      }
    }
  }

  return changed;
}

bool UI::medium_dropdown(const char* label, uint32_t& medium) {
  if (_medium_mapping.empty()) {
    return false;
  }

  bool changed = false;
  const char* current = _medium_mapping.name_for(medium);
  if (current == nullptr) {
    current = "None";
  }

  if (ImGui::BeginCombo(label, current)) {
    bool is_none = (medium == kInvalidIndex);
    if (ImGui::Selectable("None", is_none)) {
      medium = kInvalidIndex;
      changed = true;
    }
    ImGui::Separator();
    for (uint64_t i = 0, e = _medium_mapping.size(); i < e; ++i) {
      const auto& entry = _medium_mapping.entry(static_cast<int32_t>(i));
      bool is_selected = (medium == entry.index);
      if (ImGui::Selectable(entry.name, is_selected)) {
        medium = entry.index;
        changed = true;
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  return changed;
}

bool UI::image_picker(SceneRepresentation& scene_rep, const char* label, uint32_t& image_index, uint32_t image_options) {
  auto image_label = [&](uint32_t index) -> std::string {
    if (index >= scene_rep.data().images_vector.size()) {
      return "None";
    }

    const Image& image = scene_rep.data().images_vector[index];
    std::string path = scene_rep.data().images.path(index);
    std::string name = {};
    if ((path.empty() == false) && (path.starts_with("##") == false)) {
      name = std::filesystem::path(path).filename().string();
    }
    if (name.empty()) {
      name = "Image " + std::to_string(index);
    }

    if ((image.isize.x > 0u) && (image.isize.y > 0u) && (image.isize.z > 0u)) {
      name += format_string(" (%u x %u x %u)", image.isize.x, image.isize.y, image.isize.z);
    }
    return name;
  };

  bool changed = false;
  bool load_from_file = false;
  const std::string current = (image_index == kInvalidIndex) ? std::string("None") : image_label(image_index);
  char buffer[256] = {};
  snprintf(buffer, sizeof(buffer), "##picker_%s", label);
  if (ImGui::BeginCombo(buffer, current.c_str())) {
    const bool is_none = image_index == kInvalidIndex;
    if (ImGui::Selectable("None", is_none)) {
      image_index = kInvalidIndex;
      changed = true;
    }
    if (is_none) {
      ImGui::SetItemDefaultFocus();
    }

    ImGui::Separator();
    for (uint32_t i = 0u; i < scene_rep.data().images_vector.size(); ++i) {
      const Image& image = scene_rep.data().images_vector[i];
      if (image.isize.z > 1u) {
        continue;
      }

      const std::string entry_label = image_label(i);
      const bool is_selected = image_index == i;
      if (ImGui::Selectable(entry_label.c_str(), is_selected)) {
        image_index = i;
        changed = true;
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }

    ImGui::Separator();
    if (ImGui::Selectable("Load Texture...", false)) {
      load_from_file = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndCombo();
  }

  if (load_from_file) {
    const std::string selected_file = open_file("exr,png,hdr,pfm,jpg,bmp,tga");
    if (selected_file.empty() == false) {
      image_index = scene_rep.data().add_image(selected_file.c_str(), image_options, {}, {1.0f, 1.0f});
      changed = true;
    }
  }

  return changed;
}

bool UI::sampled_image_picker(SceneRepresentation& scene_rep, const char* label, SampledImage& image, uint32_t image_options) {
  bool changed = image_picker(scene_rep, label, image.image_index, image_options);
  if (image.image_index == kInvalidIndex) {
    if (image.channel != kInvalidIndex) {
      image.channel = kInvalidIndex;
      changed = true;
    }
    return changed;
  }

  if (image.channel >= 4u) {
    image.channel = 0u;
    changed = true;
  }

  const char* channel_names[] = {"R", "G", "B", "A"};
  int32_t channel_index = static_cast<int32_t>(image.channel);
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  const std::string channel_label = std::string("Channel##") + label;
  if (ImGui::Combo(channel_label.c_str(), &channel_index, channel_names, 4)) {
    image.channel = static_cast<uint32_t>(channel_index);
    changed = true;
  }

  return changed;
}

bool UI::spectrum_picker(SceneRepresentation& scene, const char* widget_id, uint32_t spd_index, bool linear, bool scale, bool show_color, bool show_scale) {
  if (scene.data().spectrum_values.empty()) {
    return false;
  }
  if (spd_index >= scene.data().spectrum_values.size()) {
    return false;
  }
  SpectralDistribution& spd = scene.data().spectrum_values[spd_index];
  ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
  bool result = spectrum_picker(widget_id, spd, linear, scale, show_color, show_scale);
  ImGui::PopItemWidth();
  return result;
}

bool UI::spectrum_picker(const char* widget_id, SpectralDistribution& spd, bool linear, bool scale, bool show_color, bool show_scale) {
  scale = scale && linear;

  char unique_key_buf[32] = {};
  snprintf(unique_key_buf, sizeof(unique_key_buf), "%p", (void*)&spd);
  std::string editor_key = std::string(unique_key_buf);
  auto [state_it, state_inserted] = _spectrum_editors.emplace(editor_key, SpectrumEditorState{});
  SpectrumEditorState& editor_state = state_it->second;

  float3 linear_rgb = spd.integrated();
  float default_scale = 1.0f;
  float max_component = max(max(linear_rgb.x, linear_rgb.y), linear_rgb.z);
  if (scale) {
    constexpr float kScaleUnityThreshold = 1.0001f;
    if (max_component > kScaleUnityThreshold) {
      default_scale = max_component;
      if (default_scale > 0.0f) {
        linear_rgb /= default_scale;
      }
    }
  }

  float3 default_display = linear ? linear_rgb : linear_to_gamma(linear_rgb);
  if (state_inserted) {
    editor_state.color = default_display;
    editor_state.scale = default_scale;
  }

  const char* name_buffer = format_string("##color_%s", widget_id);

  const char* scale_label = nullptr;
  if (scale && show_scale) {
    scale_label = format_string("##scale_%s", widget_id);
  }

  auto sync_from_spd = [&](bool update_color, bool update_scale) {
    if ((update_color == false) && (update_scale == false))
      return;

    float3 refreshed_linear = spd.integrated();
    float refreshed_scale = editor_state.scale;
    float3 refreshed_display = editor_state.color;

    constexpr float kScaleUnityThreshold = 1.0001f;
    if (scale && (update_color || update_scale)) {
      float refreshed_max = max(max(refreshed_linear.x, refreshed_linear.y), refreshed_linear.z);
      if (refreshed_max > kScaleUnityThreshold) {
        refreshed_scale = refreshed_max;
        if (refreshed_scale > 0.0f) {
          refreshed_linear /= refreshed_scale;
        }
      } else {
        refreshed_scale = 1.0f;
      }
    }

    if (update_color) {
      refreshed_display = linear ? refreshed_linear : linear_to_gamma(refreshed_linear);
      if (scale) {
        refreshed_display.x = std::clamp(refreshed_display.x, 0.0f, 1.0f);
        refreshed_display.y = std::clamp(refreshed_display.y, 0.0f, 1.0f);
        refreshed_display.z = std::clamp(refreshed_display.z, 0.0f, 1.0f);
      }
      editor_state.color = refreshed_display;
    }

    if (update_scale) {
      editor_state.scale = refreshed_scale;
    }
  };

  auto rebuild_spectrum = [&](float applied_scale, bool from_color) {
    float3 value = editor_state.color;
    if ((linear == false) && from_color) {
      value = gamma_to_linear(value);
    }
    applied_scale = from_color ? max(applied_scale, 1.0f) : max(applied_scale, 0.0f);

    if (from_color) {
      value *= applied_scale;
      spd = SpectralDistribution::rgb_luminance(value);
      editor_state.scale = applied_scale;
    } else if (scale) {
      if (editor_state.mode == SpectrumEditorState::Mode::Temperature) {
        spd = SpectralDistribution::from_normalized_black_body(editor_state.temperature, applied_scale);
      } else {
        spd.scale(applied_scale / max(editor_state.scale, 1.0e-6f));
      }
      editor_state.scale = applied_scale;
    }
  };

  ImGuiColorEditFlags color_flags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_InputRGB;
  if (linear && (scale == false)) {
    color_flags |= ImGuiColorEditFlags_HDR;
  }

  bool color_active = false;
  bool color_edited = false;
  bool color_deactivated_after_edit = false;
  if (show_color) {
    ImGui::ColorEdit3(name_buffer, &editor_state.color.x, color_flags);
    color_active = ImGui::IsItemActive();
    color_edited = ImGui::IsItemEdited();
    color_deactivated_after_edit = ImGui::IsItemDeactivatedAfterEdit();

    if (scale) {
      editor_state.color.x = std::clamp(editor_state.color.x, 0.0f, 1.0f);
      editor_state.color.y = std::clamp(editor_state.color.y, 0.0f, 1.0f);
      editor_state.color.z = std::clamp(editor_state.color.z, 0.0f, 1.0f);
    }
  }

  bool changed = false;
  if (show_color && color_deactivated_after_edit) {
    editor_state.scale = 1.0f;
    rebuild_spectrum(editor_state.scale, true);
    changed = true;
  }

  bool scale_changed = false;
  bool scale_active = false;
  bool scale_deactivated_after_edit = false;
  if (scale && show_scale) {
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    float drag_speed = max(0.01f, max(editor_state.scale, 1.0f) * 0.01f);
    float scale_value = editor_state.scale;
    scale_changed = ImGui::DragFloat(scale_label, &scale_value, drag_speed, show_color ? 1.0f : 0.01f, 1000.0f, "Scale: %.2f", ImGuiSliderFlags_NoRoundToFormat);
    scale_active = ImGui::IsItemActive();
    scale_deactivated_after_edit = ImGui::IsItemDeactivatedAfterEdit();
    if (scale_changed) {
      if (show_color) {
        scale_value = max(scale_value, 1.0f);
      } else {
        scale_value = max(scale_value, 0.0f);
      }
      rebuild_spectrum(scale_value, show_color);
      editor_state.scale = scale_value;
      changed = true;
    }
  }

  bool skip_sync = (show_color && (color_active || color_edited || color_deactivated_after_edit)) ||  //
                   (show_scale && (scale_active || scale_changed || scale_deactivated_after_edit));
  if (skip_sync == false) {
    sync_from_spd(show_color, show_color);
  }

  return changed;
}

constexpr uint32_t kWindowFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize;

void UI::build(SceneRepresentation& scene_rep, const FrameData& data) {
  ETX_PROFILER_SCOPE();

  if (_selection.kind == SelectionKind::None) {
    set_selection(SelectionKind::Rendering, 0);
  }

  BuildContext ctx = {};
  ctx.wpadding = {ImGui::GetStyle().WindowPadding.x, ImGui::GetStyle().WindowPadding.y};
  ctx.fpadding = {ImGui::GetStyle().FramePadding.x, ImGui::GetStyle().FramePadding.y};
  ctx.text_size = ImGui::GetFontSize();
  ctx.button_size = 32.0f;
  ctx.input_size = 64.0f;
  ctx.has_integrator = (_current_integrator != nullptr);

  _frame_count++;
  _last_fps_update_time += data.dt;
  if (_last_fps_update_time >= 0.5) {
    _current_fps = static_cast<float>(_frame_count) / static_cast<float>(_last_fps_update_time);
    _frame_count = 0;
    _last_fps_update_time = 0.0;
  }

  ctx.emitter_primary_instance.clear();

  ctx.with_window = [&](uint32_t flag, const char* title, std::function<void()>&& body) {
    if ((_ui_setup & flag) == 0)
      return;

    float char_width = ImGui::CalcTextSize("W").x;
    float target_width = char_width * 30.0f + ImGui::GetStyle().WindowPadding.x * 2.0f;
    ImGui::SetNextWindowSize(ImVec2(target_width, 0.0f), ImGuiCond_Always);
    if (ImGui::Begin(title, nullptr, kWindowFlags | ImGuiWindowFlags_NoResize)) {
      body();
    }
    ImGui::End();
  };

  uint64_t mmh = hash_mapping(scene_rep.material_mapping());
  if (mmh != _material_mapping_hash) {
    _material_mapping.build(scene_rep.material_mapping());
    _material_mapping_hash = mmh;
  }

  uint64_t medh = hash_mapping(scene_rep.medium_mapping());
  if (medh != _medium_mapping_hash) {
    _medium_mapping.build(scene_rep.medium_mapping());
    _medium_mapping_hash = medh;
  }

  uint64_t meshh = hash_mapping(scene_rep.mesh_mapping());
  if (meshh != _mesh_mapping_hash) {
    _mesh_mapping.build(scene_rep.mesh_mapping());
    _mesh_mapping_hash = meshh;
  }

  uint64_t camh = hash_mapping(scene_rep.camera_mapping());
  if (camh != _camera_mapping_hash) {
    _camera_mapping.build(scene_rep.camera_mapping());
    _camera_mapping_hash = camh;
  }

  auto apply_pending_selection = [&](const MappingRepresentation& map, SelectionKind kind) {
    if ((_pending_selection.has == false) || (_pending_selection.kind != kind)) {
      return;
    }
    auto it = map.reverse.find(_pending_selection.index);
    if (it != map.reverse.end()) {
      set_selection(kind, static_cast<int32_t>(it->second), false);
    }
  };

  apply_pending_selection(_material_mapping, SelectionKind::Material);
  apply_pending_selection(_medium_mapping, SelectionKind::Medium);
  apply_pending_selection(_mesh_mapping, SelectionKind::Mesh);
  apply_pending_selection(_camera_mapping, SelectionKind::Camera);
  _pending_selection = {};

  validate_selections(scene_rep);

  if (_embedded_menu_enabled) {
    build_main_menu_bar(data.recent_files);
  }
  build_toolbar(ctx);
  build_scene_objects_window(scene_rep, ctx);
  build_properties_window(scene_rep, scene_rep.camera(), ctx, data);

  if (ctx.has_integrator && (_current_integrator->status().debug_info_count > 0) && (_current_integrator->status().debug_info != nullptr)) {
    if (ImGui::Begin("Debug Info", nullptr, kWindowFlags)) {
      auto debug_info = _current_integrator->status().debug_info;
      for (uint64_t i = 0, e = _current_integrator->status().debug_info_count; i < e; ++i) {
        const char* value_buffer = format_string("%.3f", debug_info[i].value);
        ImGui::LabelText(debug_info[i].title, "%s", value_buffer);
      }
      ImGui::End();
    }
  }
}

bool UI::handle_event(const sapp_event* e) {
  if (e->type != SAPP_EVENTTYPE_KEY_DOWN) {
    return false;
  }

  auto modifiers = e->modifiers;
  bool has_alt = modifiers & SAPP_MODIFIER_ALT;
  bool has_shift = modifiers & SAPP_MODIFIER_SHIFT;

  if ((modifiers & SAPP_MODIFIER_CTRL) || (modifiers & SAPP_MODIFIER_SUPER)) {
    switch (e->key_code) {
      case SAPP_KEYCODE_Q: {
        execute_menu_command(MenuCommand::Quit);
        break;
      }
      case SAPP_KEYCODE_O: {
        execute_menu_command(MenuCommand::OpenScene);
        break;
      }
      case SAPP_KEYCODE_I: {
        execute_menu_command(MenuCommand::OpenReferenceImage);
        break;
      }
      case SAPP_KEYCODE_R: {
        if (has_shift) {
          execute_menu_command(MenuCommand::UseImageAsReference);
        } else {
          execute_menu_command(MenuCommand::ReloadScene);
        }
        break;
      }
      case SAPP_KEYCODE_G: {
        execute_menu_command(MenuCommand::ReloadGeometry);
        break;
      }
      case SAPP_KEYCODE_S: {
        if (has_alt && (has_shift == false)) {
          // TODO : use
        } else if (has_shift && (has_alt == false)) {
          execute_menu_command(MenuCommand::SaveImageLDR);
        } else if ((has_shift == false) && (has_alt == false)) {
          execute_menu_command(MenuCommand::SaveImageRGB);
        }
        break;
      }
      default:
        break;
    }
  }

  switch (e->key_code) {
    case SAPP_KEYCODE_F1:
    case SAPP_KEYCODE_F2:
    case SAPP_KEYCODE_F3: {
      uint32_t flag = 1u << (e->key_code - SAPP_KEYCODE_F1);
      _ui_setup = (_ui_setup & flag) ? (_ui_setup & (~flag)) : (_ui_setup | flag);
      break;
    };

    case SAPP_KEYCODE_1:
    case SAPP_KEYCODE_2:
    case SAPP_KEYCODE_3:
    case SAPP_KEYCODE_4:
    case SAPP_KEYCODE_5:
    case SAPP_KEYCODE_6: {
      _view_options.view_image = static_cast<uint32_t>(e->key_code - SAPP_KEYCODE_1);
      break;
    }
    case SAPP_KEYCODE_KP_DIVIDE: {
      execute_menu_command(MenuCommand::DecreaseExposure);
      break;
    }
    case SAPP_KEYCODE_KP_MULTIPLY: {
      execute_menu_command(MenuCommand::IncreaseExposure);
      break;
    }
    default:
      break;
  }

  return false;
}

void UI::execute_menu_command(MenuCommand command, uint32_t argument, const std::string& value) {
  switch (command) {
    case MenuCommand::Quit:
      quit();
      break;
    case MenuCommand::SelectCPURenderer:
      if (callbacks.renderer_selected) {
        callbacks.renderer_selected(RendererMode::CPURaytracing);
      }
      break;
    case MenuCommand::SelectRasterRenderer:
      if (callbacks.renderer_selected) {
        callbacks.renderer_selected(RendererMode::Rasterization);
      }
      break;
    case MenuCommand::SelectGPURenderer:
      if (_gpu_renderer_available && callbacks.renderer_selected) {
        callbacks.renderer_selected(RendererMode::GPURaytracing);
      }
      break;
    case MenuCommand::OpenScene:
      select_scene_file();
      break;
    case MenuCommand::ReloadScene:
      reload_scene();
      break;
    case MenuCommand::ReloadGeometry:
      reload_geometry();
      break;
    case MenuCommand::OpenRecentScene:
      if (!value.empty() && callbacks.scene_file_selected) {
        callbacks.scene_file_selected(value);
      }
      break;
    case MenuCommand::ClearRecentScenes:
      if (callbacks.clear_recent_files) {
        callbacks.clear_recent_files();
      }
      break;
    case MenuCommand::SaveScene:
      save_scene_file();
      break;
    case MenuCommand::SaveSceneAs:
      save_scene_file_as();
      break;
    case MenuCommand::SelectIntegrator:
      if ((argument < _integrators.count) && (_integrators[argument] != nullptr) && _integrators[argument]->enabled()) {
        if (callbacks.integrator_selected) {
          callbacks.integrator_selected(_integrators[argument]->type());
        }
        set_current_integrator(_integrators[argument]);
      }
      break;
    case MenuCommand::OpenReferenceImage:
      load_image();
      break;
    case MenuCommand::SaveImageRGB:
      save_image(SaveImageMode::RGB);
      break;
    case MenuCommand::SaveImageLDR:
      save_image(SaveImageMode::TonemappedLDR);
      break;
    case MenuCommand::UseImageAsReference:
      if (callbacks.use_image_as_reference) {
        callbacks.use_image_as_reference();
      }
      break;
    case MenuCommand::ViewWholeScene:
    case MenuCommand::ViewPositiveX:
    case MenuCommand::ViewNegativeX:
    case MenuCommand::ViewPositiveY:
    case MenuCommand::ViewNegativeY:
    case MenuCommand::ViewPositiveZ:
    case MenuCommand::ViewNegativeZ:
      if (callbacks.view_scene) {
        const uint32_t direction = static_cast<uint32_t>(command) - static_cast<uint32_t>(MenuCommand::ViewWholeScene);
        callbacks.view_scene(direction);
      }
      break;
    case MenuCommand::IncreaseExposure:
      increase_exposure(_view_options);
      break;
    case MenuCommand::DecreaseExposure:
      decrease_exposure(_view_options);
      break;
    case MenuCommand::ToggleSceneObjects:
      _ui_setup ^= UIObjects;
      break;
    case MenuCommand::ToggleProperties:
      _ui_setup ^= UIProperties;
      break;
  }
}

ViewParameters UI::view_options() const {
  return _view_options;
}

ViewParameters& UI::mutable_view_options() {
  return _view_options;
}

void UI::set_current_integrator(Integrator* i) {
  _current_integrator = i;
  if (i != nullptr) {
    set_selection(SelectionKind::Rendering, 0, false);
  }
}

void UI::quit() {
  sapp_quit();
}

void UI::select_scene_file() const {
  auto selected_file = open_file("json,obj,gltf,glb");
  if ((selected_file.empty() == false) && callbacks.scene_file_selected) {
    callbacks.scene_file_selected(selected_file);
  }
}

void UI::save_scene_file() const {
  if (callbacks.save_scene_file_selected) {
    callbacks.save_scene_file_selected({});
  }
}

void UI::save_scene_file_as() const {
  if (callbacks.save_scene_file_as_selected) {
    callbacks.save_scene_file_as_selected();
  }
}

void UI::save_image(SaveImageMode mode) const {
  auto selected_file = save_file(mode == SaveImageMode::TonemappedLDR ? "png" : "exr");
  if ((selected_file.empty() == false) && callbacks.save_image_selected) {
    callbacks.save_image_selected(selected_file, mode);
  }
}

void UI::load_image() const {
  auto selected_file = open_file("exr,png,hdr,pfm,jpg,bmp,tga");
  if ((selected_file.empty() == false) && callbacks.reference_image_selected) {
    callbacks.reference_image_selected(selected_file);
  }
}

bool UI::build_material(SceneRepresentation& scene_rep, Material& material, const FrameData& data) {
  auto material_values_mixed = [&](auto getter) -> bool {
    if ((_editing_material_indices == nullptr) || (_editing_material_indices->size() <= 1u)) {
      return false;
    }

    const uint32_t first_index = _editing_material_indices->front();
    if (first_index >= scene_rep.data().materials.size()) {
      return false;
    }

    const auto first_value = getter(scene_rep.data().materials[first_index]);
    for (const uint32_t material_index : *_editing_material_indices) {
      if (material_index >= scene_rep.data().materials.size()) {
        continue;
      }

      const auto value = getter(scene_rep.data().materials[material_index]);
      if (std::memcmp(&first_value, &value, sizeof(first_value)) != 0) {
        return true;
      }
    }

    return false;
  };

  static const SpectralDistribution zero_spectrum = SpectralDistribution::constant(0.0f);
  auto spectrum_or_fallback = [&](uint32_t spectrum_index, const SpectralDistribution* fallback) -> const SpectralDistribution* {
    if (spectrum_index < scene_rep.data().spectrum_values.size()) {
      return &scene_rep.data().spectrum_values[spectrum_index];
    }

    return fallback;
  };
  auto spectra_equal = [](const SpectralDistribution* a, const SpectralDistribution* b) -> bool {
    if ((a == nullptr) || (b == nullptr)) {
      return a == b;
    }

    return std::memcmp(a, b, sizeof(SpectralDistribution)) == 0;
  };
  auto refractive_indices_equal = [&](const RefractiveIndex& a, const RefractiveIndex& b) -> bool {
    if (a.cls != b.cls) {
      return false;
    }

    const SpectralDistribution* a_eta = spectrum_or_fallback(a.eta_index, nullptr);
    const SpectralDistribution* b_eta = spectrum_or_fallback(b.eta_index, nullptr);
    if (spectra_equal(a_eta, b_eta) == false) {
      return false;
    }

    const SpectralDistribution* a_k = spectrum_or_fallback(a.k_index, &zero_spectrum);
    const SpectralDistribution* b_k = spectrum_or_fallback(b.k_index, &zero_spectrum);
    return spectra_equal(a_k, b_k);
  };
  auto material_ior_values_mixed = [&](auto getter) -> bool {
    if ((_editing_material_indices == nullptr) || (_editing_material_indices->size() <= 1u)) {
      return false;
    }

    const uint32_t first_index = _editing_material_indices->front();
    if (first_index >= scene_rep.data().materials.size()) {
      return false;
    }

    const RefractiveIndex first_value = getter(scene_rep.data().materials[first_index]);
    for (const uint32_t material_index : *_editing_material_indices) {
      if (material_index >= scene_rep.data().materials.size()) {
        continue;
      }

      const RefractiveIndex value = getter(scene_rep.data().materials[material_index]);
      if (refractive_indices_equal(first_value, value) == false) {
        return true;
      }
    }

    return false;
  };

  auto mixed_control = [&](bool mixed, auto&& control) {
    if (mixed) {
      ImGui::PushItemFlag(ImGuiItemFlags_MixedValue, true);
    }

    const bool result = control();
    if (mixed) {
      ImGui::PopItemFlag();
    }
    return result;
  };

  const bool class_mixed = material_values_mixed([](const Material& value) {
    return value.cls;
  });
  bool changed = mixed_control(class_mixed, [&]() {
    return build_material_class_selector(material, class_mixed);
  });
  if (changed) {
    _material_batch_changed_fields |= MaterialBatchChangedClass;
  }
  if (class_mixed) {
    ImGui::TextDisabled("Mixed material classes");
  }

  if ((material.cls == MaterialClass::OpenPBR) && ((_editing_material_indices == nullptr) || (_editing_material_indices->size() <= 1u))) {
    if (ImGui::Button("Load OpenPBR MaterialX...", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
      const std::string selected_file = open_file("mtlx");
      if (selected_file.empty() == false) {
        changed = load_openpbr_material_file(selected_file, scene_rep.data(), material) || changed;
      }
    }
  }

  const auto uses_surface_spectra = [&]() -> bool {
    return (material.cls != MaterialClass::Boundary) && (material.cls != MaterialClass::Void);
  };

  const auto uses_roughness = [&]() -> bool {
    switch (material.cls) {
      case MaterialClass::Diffuse:
      case MaterialClass::Plastic:
      case MaterialClass::Conductor:
      case MaterialClass::Dielectric:
      case MaterialClass::Velvet:
      case MaterialClass::OpenPBR:
        return true;
      default:
        return false;
    }
  };

  const auto uses_interface_ior = [&]() -> bool {
    switch (material.cls) {
      case MaterialClass::Plastic:
      case MaterialClass::Conductor:
      case MaterialClass::Dielectric:
      case MaterialClass::Thinfilm:
      case MaterialClass::OpenPBR:
        return true;
      default:
        return false;
    }
  };

  const auto uses_subsurface = [&]() -> bool {
    return (material.cls == MaterialClass::Diffuse) || (material.cls == MaterialClass::Plastic) || (material.cls == MaterialClass::OpenPBR);
  };

  const ImVec4 base_bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
  auto clamp01 = [](float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
  };
  ImVec4 sec_col[6] = {
    {clamp01(base_bg.x + 0.02f), clamp01(base_bg.y + 0.02f), clamp01(base_bg.z + 0.02f), base_bg.w},
    {clamp01(base_bg.x + 0.015f), clamp01(base_bg.y + 0.010f), clamp01(base_bg.z + 0.000f), base_bg.w},
    {clamp01(base_bg.x + 0.000f), clamp01(base_bg.y + 0.015f), clamp01(base_bg.z + 0.010f), base_bg.w},
    {clamp01(base_bg.x + 0.015f), clamp01(base_bg.y + 0.000f), clamp01(base_bg.z + 0.015f), base_bg.w},
    {clamp01(base_bg.x + 0.010f), clamp01(base_bg.y + 0.010f), clamp01(base_bg.z + 0.000f), base_bg.w},
    {clamp01(base_bg.x + 0.000f), clamp01(base_bg.y + 0.010f), clamp01(base_bg.z + 0.010f), base_bg.w},
  };

  auto brighten = [&](const ImVec4& c, float d) {
    return ImVec4{clamp01(c.x + d), clamp01(c.y + d), clamp01(c.z + d), c.w};
  };
  auto with_section = [&](int color_index, const char* title, auto&& body, bool force_open) {
    if (force_open) {
      ImGui::SetNextItemOpen(true, ImGuiCond_Always);
    }
    ImGui::PushStyleColor(ImGuiCol_Header, sec_col[color_index]);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brighten(sec_col[color_index], 0.04f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, brighten(sec_col[color_index], 0.08f));
    const bool open = ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_Framed);
    ImGui::PopStyleColor(3);
    if (open) {
      body();
    }
  };

  if (uses_surface_spectra()) {
    with_section(
      0, "Surface",
      [&]() {
        ImGui::Text("Reflectance Spectrum");
        changed |= mixed_control(material_values_mixed([](const Material& value) {
          return value.reflectance.spectrum_index;
        }),
          [&]() {
            return spectrum_picker(scene_rep, "Reflectance", material.reflectance.spectrum_index, false, false);
          });
        changed |= mixed_control(material_values_mixed([](const Material& value) {
          return value.reflectance.image_index;
        }),
          [&]() {
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            return image_picker(scene_rep, "Reflectance Texture##reflectance_texture", material.reflectance.image_index, Image::RepeatU | Image::RepeatV);
          });
        if (material.cls != MaterialClass::DiffractionGrating) {
          ImGui::Spacing();
          ImGui::Text("Scattering Spectrum");
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return value.scattering.spectrum_index;
          }),
            [&]() {
              return spectrum_picker(scene_rep, "Scattering", material.scattering.spectrum_index, false, false);
            });
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return value.scattering.image_index;
          }),
            [&]() {
              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
              return image_picker(scene_rep, "Scattering Texture##scattering_texture", material.scattering.image_index, Image::RepeatU | Image::RepeatV);
            });
        }
        ImGui::Spacing();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        const bool opacity_mixed = material_values_mixed([](const Material& value) {
          return value.opacity;
        });
        changed |= mixed_control(opacity_mixed, [&]() {
          return ImGui::SliderFloat("##opacity", &material.opacity, 0.0f, 1.0f, opacity_mixed ? "Opacity mixed" : "Opacity %.3f",
            ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
        });
        if (uses_roughness()) {
          ImGui::Spacing();
          float& rough_u = material.roughness.value.x;
          float& rough_v = material.roughness.value.y;

          char material_key_buf[32] = {};
          snprintf(material_key_buf, sizeof(material_key_buf), "%p", (void*)&material);
          std::string material_key(material_key_buf);

          auto aniso_insert = _material_anisotropy.emplace(material_key, std::fabs(rough_u - rough_v) > 1.0e-4f);
          auto aniso_entry = aniso_insert.first;
          bool anisotropic = aniso_entry->second;
          const bool force_isotropic_roughness = material.cls == MaterialClass::Diffuse;

          if (force_isotropic_roughness) {
            anisotropic = false;
            aniso_entry->second = false;
            if (rough_v != rough_u) {
              rough_v = rough_u;
              changed = true;
            }
          } else {
            if (ImGui::Checkbox("Anisotropic##rough_aniso", &anisotropic)) {
              aniso_entry->second = anisotropic;
              if (anisotropic == false) {
                rough_v = rough_u;
                changed = true;
              }
            }
            ImGui::Spacing();
          }

          ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
          const bool rough_u_mixed = material_values_mixed([](const Material& value) {
            return value.roughness.value.x;
          });
          const bool rough_u_changed = mixed_control(rough_u_mixed, [&]() {
            return ImGui::SliderFloat("##rough_u", &rough_u, 0.0f, 1.0f, rough_u_mixed ? "Roughness mixed" : (anisotropic ? "Roughness U %.3f" : "Roughness %.3f"),
              ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
          });
          if (rough_u_changed) {
            if (anisotropic == false) {
              rough_v = rough_u;
              aniso_entry->second = false;
            }
            changed = true;
          }

          if (anisotropic) {
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            const bool rough_v_mixed = material_values_mixed([](const Material& value) {
              return value.roughness.value.y;
            });
            const bool rough_v_changed = mixed_control(rough_v_mixed, [&]() {
              return ImGui::SliderFloat("##rough_v", &rough_v, 0.0f, 1.0f, rough_v_mixed ? "Roughness V mixed" : "Roughness V %.3f",
                ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
            });
            if (rough_v_changed) {
              changed = true;
            }
          }

          ImGui::Spacing();
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return uint2{value.roughness.image_index, value.roughness.channel};
          }),
            [&]() {
              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
              return sampled_image_picker(scene_rep, "Roughness Texture##roughness_texture", material.roughness, Image::RepeatU | Image::RepeatV | Image::SkipSRGBConversion);
            });
        }
        if (material.cls == MaterialClass::OpenPBR) {
          ImGui::Spacing();
          float metal = material.metalness.value.x;
          ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
          const bool metalness_mixed = material_values_mixed([](const Material& value) {
            return value.metalness.value.x;
          });
          const bool metalness_changed = mixed_control(metalness_mixed, [&]() {
            return ImGui::SliderFloat("##metalness", &metal, 0.0f, 1.0f, metalness_mixed ? "Metalness mixed" : "Metalness %.3f",
              ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
          });
          if (metalness_changed) {
            material.metalness.value = {metal, metal, metal, metal};
            changed = true;
          }
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return uint2{value.metalness.image_index, value.metalness.channel};
          }),
            [&]() {
              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
              return sampled_image_picker(scene_rep, "Metalness Texture##metalness_texture", material.metalness, Image::RepeatU | Image::RepeatV | Image::SkipSRGBConversion);
            });

          float trans = material.transmission.value.x;
          ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
          const bool transmission_mixed = material_values_mixed([](const Material& value) {
            return value.transmission.value.x;
          });
          const bool transmission_changed = mixed_control(transmission_mixed, [&]() {
            return ImGui::SliderFloat("##transmission", &trans, 0.0f, 1.0f, transmission_mixed ? "Transmission mixed" : "Transmission %.3f",
              ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
          });
          if (transmission_changed) {
            material.transmission.value = {trans, trans, trans, trans};
            changed = true;
          }
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return uint2{value.transmission.image_index, value.transmission.channel};
          }),
            [&]() {
              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
              return sampled_image_picker(scene_rep, "Transmission Texture##transmission_texture", material.transmission,
                Image::RepeatU | Image::RepeatV | Image::SkipSRGBConversion);
            });
        }

        ImGui::Spacing();
        ImGui::Text("Normal Map");
        changed |= mixed_control(material_values_mixed([](const Material& value) {
          return value.normal_image_index;
        }),
          [&]() {
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            return image_picker(scene_rep, "Normal Texture##normal_texture", material.normal_image_index, Image::RepeatU | Image::RepeatV | Image::SkipSRGBConversion);
          });
        if (material.normal_image_index != kInvalidIndex) {
          ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return value.normal_scale;
          }),
            [&]() {
              return ImGui::SliderFloat("##normal_scale", &material.normal_scale, 0.0f, 4.0f, "Normal Scale %.3f", ImGuiSliderFlags_AlwaysClamp);
            });
        }

        if (uses_interface_ior()) {
          if ((uses_roughness()) || (material.cls == MaterialClass::OpenPBR)) {
            ImGui::Spacing();
          }
          const ImVec2 old_cell_padding = ImGui::GetStyle().CellPadding;
          ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(1.0f, old_cell_padding.y));
          if (ImGui::BeginTable("ior_inout", 2, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            const bool int_ior_mixed = material_ior_values_mixed([](const Material& value) {
              return value.int_ior;
            });
            const bool int_ior_changed = mixed_control(int_ior_mixed, [&]() {
              return ior_picker(scene_rep, "Inside", material.int_ior, data, int_ior_mixed);
            });
            if (int_ior_changed) {
              _material_batch_changed_fields |= MaterialBatchChangedIntIOR;
              changed = true;
            }
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            const bool ext_ior_mixed = material_ior_values_mixed([](const Material& value) {
              return value.ext_ior;
            });
            const bool ext_ior_changed = mixed_control(ext_ior_mixed, [&]() {
              return ior_picker(scene_rep, "Outside", material.ext_ior, data, ext_ior_mixed);
            });
            if (ext_ior_changed) {
              _material_batch_changed_fields |= MaterialBatchChangedExtIOR;
              changed = true;
            }
            ImGui::EndTable();
          }
          ImGui::PopStyleVar();
        }
      },
      false);
  }

  if (material.cls == MaterialClass::DiffractionGrating) {
    with_section(
      1, "Diffraction Grating",
      [&]() {
        const float previous_period_nm = material.diffraction_grating.period_nm;
        const bool period_changed = mixed_control(material_values_mixed([](const Material& value) {
          return value.diffraction_grating.period_nm;
        }),
          [&]() {
            ImGui::Text("Period (nm)");
            full_width_item();
            return ImGui::InputFloat("##diffraction_period", &material.diffraction_grating.period_nm, 0.0f, 0.0f, "%.3f");
          });
        if (period_changed) {
          material.diffraction_grating.period_nm = std::isfinite(material.diffraction_grating.period_nm)
                                                     ? clamp(material.diffraction_grating.period_nm, kDiffractionGratingMinimumPeriodNm, kDiffractionGratingMaximumPeriodNm)
                                                     : previous_period_nm;
          changed = true;
        }

        ImGui::Spacing();
        const float previous_optical_path_difference_nm = material.diffraction_grating.optical_path_difference_nm;
        const bool optical_path_difference_changed = mixed_control(material_values_mixed([](const Material& value) {
          return value.diffraction_grating.optical_path_difference_nm;
        }),
          [&]() {
            ImGui::Text("Optical Path Difference (nm)");
            full_width_item();
            return ImGui::InputFloat("##diffraction_optical_path_difference", &material.diffraction_grating.optical_path_difference_nm, 0.0f, 0.0f, "%.3f");
          });
        if (optical_path_difference_changed) {
          material.diffraction_grating.optical_path_difference_nm =
            std::isfinite(material.diffraction_grating.optical_path_difference_nm)
              ? clamp(material.diffraction_grating.optical_path_difference_nm, kDiffractionGratingMinimumOpticalPathDifferenceNm, kDiffractionGratingMaximumOpticalPathDifferenceNm)
              : previous_optical_path_difference_nm;
          changed = true;
        }

        ImGui::Spacing();
        changed |= mixed_control(material_values_mixed([](const Material& value) {
          return value.diffraction_grating.duty_cycle;
        }),
          [&]() {
            ImGui::Text("Duty Cycle");
            full_width_item();
            return ImGui::SliderFloat("##diffraction_duty", &material.diffraction_grating.duty_cycle, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
          });

        ImGui::Spacing();
        const float previous_rotation = material.diffraction_grating.rotation;
        float rotation_degrees = previous_rotation * 180.0f / kPi;
        const bool rotation_changed = mixed_control(material_values_mixed([](const Material& value) {
          return value.diffraction_grating.rotation;
        }),
          [&]() {
            ImGui::Text("Rotation (degrees)");
            full_width_item();
            return ImGui::InputFloat("##diffraction_rotation", &rotation_degrees, 0.0f, 0.0f, "%.3f");
          });
        if (rotation_changed) {
          material.diffraction_grating.rotation = std::isfinite(rotation_degrees) ? fmodf(rotation_degrees * kPi / 180.0f, kDoublePi) : previous_rotation;
          changed = true;
        }
      },
      true);
  }

  if (material.cls != MaterialClass::DiffractionGrating) {
    with_section(
      2, "Thin Film",
      [&]() {
        ImGui::Text("IoR");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        const bool thinfilm_ior_mixed = material_ior_values_mixed([](const Material& value) {
          return value.thinfilm.ior;
        });
        const bool thinfilm_ior_changed = mixed_control(thinfilm_ior_mixed, [&]() {
          return ior_picker(scene_rep, "Thinfilm IoR", material.thinfilm.ior, data, thinfilm_ior_mixed, true);
        });
        if (thinfilm_ior_changed) {
          _material_batch_changed_fields |= MaterialBatchChangedThinfilmIOR;
          changed = true;
        }

        ImGui::Spacing();
        ImGui::Text("Weight");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        changed |= mixed_control(material_values_mixed([](const Material& value) {
          return value.thinfilm.weight;
        }),
          [&]() {
            return ImGui::SliderFloat("##thinfilm_weight", &material.thinfilm.weight, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
          });

        ImGui::Spacing();
        ImGui::Text("Thickness (nm)");
        const float avail = ImGui::GetContentRegionAvail().x;
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float dash_width = ImGui::CalcTextSize(" - ").x;
        const float field_width = max((avail - dash_width - spacing * 2.0f) * 0.5f, 0.0f);
        ImGui::SetNextItemWidth(field_width);
        changed |= mixed_control(material_values_mixed([](const Material& value) {
          return value.thinfilm.min_thickness;
        }),
          [&]() {
            return ImGui::InputFloat("##tftmin", &material.thinfilm.min_thickness);
          });
        ImGui::SameLine();
        ImGui::Text(" - ");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(field_width);
        changed |= mixed_control(material_values_mixed([](const Material& value) {
          return value.thinfilm.max_thickness;
        }),
          [&]() {
            return ImGui::InputFloat("##tftmax", &material.thinfilm.max_thickness);
          });
        ImGui::Spacing();
        changed |= mixed_control(material_values_mixed([](const Material& value) {
          return value.thinfilm.thinkness_image;
        }),
          [&]() {
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            return image_picker(scene_rep, "Thickness Texture##thinfilm_thickness_texture", material.thinfilm.thinkness_image,
              Image::RepeatU | Image::RepeatV | Image::SkipSRGBConversion);
          });
      },
      material.cls == MaterialClass::Thinfilm);
  }

  if ((uses_subsurface()) || (_medium_mapping.empty() == false)) {
    with_section(
      3, "Volume & Media",
      [&]() {
        if (uses_subsurface()) {
          ImGui::TextDisabled("Subsurface Scattering");
          ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return value.subsurface_cls;
          }),
            [&]() {
              return ImGui::Combo("##sssclass", reinterpret_cast<int*>(&material.subsurface_cls), "Disabled\0Random Walk\0");
            });
          ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return value.subsurface_path;
          }),
            [&]() {
              return ImGui::Combo("##ssspath", reinterpret_cast<int*>(&material.subsurface_path), "Diffuse Transmittance\0Refraction\0");
            });
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return value.subsurface.spectrum_index;
          }),
            [&]() {
              return spectrum_picker(scene_rep, "Subsurface Distance", material.subsurface.spectrum_index, true, true);
            });
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return value.subsurface.image_index;
          }),
            [&]() {
              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
              return image_picker(scene_rep, "Subsurface Texture##subsurface_texture", material.subsurface.image_index, Image::RepeatU | Image::RepeatV);
            });

          ImGui::Spacing();
        }

        if (_medium_mapping.empty()) {
          ImGui::TextDisabled("No mediums available");
        } else {
          const float avail = ImGui::GetContentRegionAvail().x;
          const float spacing = ImGui::GetStyle().ItemSpacing.x;
          float combo_width = (avail - spacing) * 0.5f;
          combo_width = max(combo_width, 0.0f);

          ImGui::TextDisabled("Internal / External Medium");
          ImGui::SetNextItemWidth(combo_width);
          const bool int_medium_changed = mixed_control(material_values_mixed([](const Material& value) {
            return value.int_medium;
          }),
            [&]() {
              return medium_dropdown("##internal_medium", material.int_medium);
            });
          if (int_medium_changed) {
            changed = true;
          }
          ImGui::SameLine(0.0f, spacing);
          ImGui::SetNextItemWidth(combo_width);
          const bool ext_medium_changed = mixed_control(material_values_mixed([](const Material& value) {
            return value.ext_medium;
          }),
            [&]() {
              return medium_dropdown("##external_medium", material.ext_medium);
            });
          if (ext_medium_changed) {
            changed = true;
          }
        }
      },
      false);
  }

  with_section(
    4, "Emission",
    [&]() {
      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      float collimation = material.emission_collimation;
      const bool collimation_mixed = material_values_mixed([](const Material& value) {
        return value.emission_collimation;
      });
      const bool collimation_changed = mixed_control(collimation_mixed, [&]() {
        return ImGui::SliderFloat("##material_emission_collimation", &collimation, 0.0f, 1.0f, collimation_mixed ? "Collimation mixed" : "Collimation %.2f",
          ImGuiSliderFlags_AlwaysClamp);
      });
      if (collimation_changed) {
        material.emission_collimation = std::clamp(collimation, 0.0f, 1.0f);
        changed = true;
      }

      std::string preset_id = "material_emission_" + std::to_string(material.emission.spectrum_index);
      changed |= mixed_control(material_values_mixed([](const Material& value) {
        return value.emission.spectrum_index;
      }),
        [&]() {
          return emission_picker(scene_rep, "Emission", preset_id.c_str(), material.emission.spectrum_index, data);
        });
      changed |= mixed_control(material_values_mixed([](const Material& value) {
        return value.emission.image_index;
      }),
        [&]() {
          ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
          return image_picker(scene_rep, "Emission Texture##emission_texture", material.emission.image_index, Image::RepeatU | Image::RepeatV);
        });
    },
    _auto_open_emission_section);

  _auto_open_emission_section = false;

  return changed;
}

bool UI::build_material(SceneRepresentation& scene_rep, Material& material, const FrameData& data, const std::vector<uint32_t>& material_indices) {
  const std::vector<uint32_t>* previous_indices = _editing_material_indices;
  _editing_material_indices = &material_indices;
  const bool changed = build_material(scene_rep, material, data);
  _editing_material_indices = previous_indices;
  return changed;
}

bool UI::build_medium(SceneRepresentation& scene_rep, Medium& m) {
  if (scene_rep.data().spectrum_values.empty()) {
    return false;
  }

  bool changed = false;

  ImGui::Text("Medium Type");
  const char* medium_type_names[] = {"Homogeneous", "Heterogeneous (Noise)"};
  int32_t medium_type_idx = (m.cls == Medium::Heterogeneous) ? 1 : 0;
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  if (ImGui::Combo("##medium_type", &medium_type_idx, medium_type_names, 2)) {
    if (medium_type_idx == 0) {
      m.cls = Medium::Homogeneous;
      changed = true;
    } else {
      m.cls = Medium::Heterogeneous;
      m.set_grid_type(DensityGrid::Type::NoiseFunction);
      changed = true;
    }
  }

  ImGui::Text("Absorption");
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  if (spectrum_picker(scene_rep, "Absorption##medium_absorption", m.absorption_index, true, true)) {
    changed = true;
  }

  ImGui::Text("Scattering");
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  if (spectrum_picker(scene_rep, "Scattering##medium_scattering", m.scattering_index, true, true)) {
    changed = true;
  }

  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  if (ImGui::SliderFloat("##medium_phase_g", &m.phase_function_g, -0.999f, 0.999f, "Anisotropy %.3f", ImGuiSliderFlags_AlwaysClamp)) {
    changed = true;
  }

  bool explicit_connections = (m.enable_explicit_connections != 0u);
  if (ImGui::Checkbox("Explicit connections##medium_explicit_connections", &explicit_connections)) {
    m.enable_explicit_connections = explicit_connections ? 1u : 0u;
    changed = true;
  }

  if (m.cls == Medium::Heterogeneous) {
    if (m.grid_type_enum() == DensityGrid::Type::Texture3D) {
      ImGui::Text("Density grid (3D Texture)");
      ImGui::Text("%u x %u x %u", m.grid.dimensions.x, m.grid.dimensions.y, m.grid.dimensions.z);
      ImGui::Text("Bounds");
      ImGui::Text("min (%.3f, %.3f, %.3f)  max (%.3f, %.3f, %.3f)", m.bounds.p_min.x, m.bounds.p_min.y, m.bounds.p_min.z, m.bounds.p_max.x, m.bounds.p_max.y, m.bounds.p_max.z);
    } else if (m.grid_type_enum() == DensityGrid::Type::NoiseFunction) {
      ImGui::Text("Density grid (Noise Function)");
      ImGui::Text("Noise Type");
      const char* noise_names[] = {"Perlin", "Worley", "Billow", "Voronoi", "Lattice", "Uniform"};
      int32_t noise_idx = static_cast<int32_t>(m.noise_type_enum());
      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      if (ImGui::Combo("##medium_noise_type", &noise_idx, noise_names, static_cast<int32_t>(NoiseFunction::Count))) {
        m.set_noise_type(static_cast<NoiseFunction>(noise_idx));
        changed = true;
      }
      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      if (ImGui::SliderFloat("##medium_noise_scale", &m.grid.noise_scale, 0.01f, 100.0f, "Scale: %.2f", ImGuiSliderFlags_AlwaysClamp)) {
        changed = true;
      }
      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      if (ImGui::SliderInt("##medium_noise_octaves", reinterpret_cast<int32_t*>(&m.grid.noise_octaves), 1, 16, "Octaves: %d", ImGuiSliderFlags_AlwaysClamp)) {
        changed = true;
      }
      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      if (ImGui::SliderFloat("##medium_noise_lacunarity", &m.grid.noise_lacunarity, 1.0f, 4.0f, "Lacunarity: %.2f", ImGuiSliderFlags_AlwaysClamp)) {
        changed = true;
      }
      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      if (ImGui::SliderFloat("##medium_noise_persistence", &m.grid.noise_persistence, 0.01f, 1.0f, "Persistence: %.2f", ImGuiSliderFlags_AlwaysClamp)) {
        changed = true;
      }
      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      if (ImGui::InputInt("##medium_noise_seed", reinterpret_cast<int32_t*>(&m.grid.noise_seed))) {
        changed = true;
      }
      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      if (ImGui::SliderFloat("##medium_noise_power", &m.grid.noise_power, 0.1f, 10.0f, "Power: %.2f", ImGuiSliderFlags_AlwaysClamp)) {
        changed = true;
      }
      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      if (ImGui::SliderFloat("##medium_noise_sharpness", &m.grid.noise_sharpness, 0.1f, 10.0f, "Sharpness: %.2f", ImGuiSliderFlags_AlwaysClamp)) {
        changed = true;
      }
      ImGui::Text("Offset");
      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      if (ImGui::DragFloat3("##medium_noise_offset", &m.grid.noise_offset.x, 0.1f, -100.0f, 100.0f, "%.2f")) {
        changed = true;
      }
      bool border_fade_enabled = m.grid.noise_enable_border_fade != 0u;
      if (ImGui::Checkbox("Border Fade", &border_fade_enabled)) {
        m.grid.noise_enable_border_fade = border_fade_enabled ? 1u : 0u;
        changed = true;
      }
      if (border_fade_enabled) {
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        if (ImGui::SliderFloat("##medium_noise_border_fade_distance", &m.grid.noise_border_fade_distance, 0.01f, 0.5f, "Fade Distance: %.3f", ImGuiSliderFlags_AlwaysClamp)) {
          changed = true;
        }
      }
    }
  }

  return changed;
}

void UI::reset_selection() {
  _selection = {};
  _selection_history.clear();
  _selection_history_cursor = -1;
  _spectrum_editors.clear();
  _material_anisotropy.clear();
  _selected_material_positions.clear();
  _material_selection_anchor = -1;
}

void UI::reload_geometry() {
  reset_selection();

  if (callbacks.reload_geometry_selected) {
    callbacks.reload_geometry_selected();
  }
}

void UI::reload_scene() {
  reset_selection();

  if (callbacks.reload_scene_selected) {
    callbacks.reload_scene_selected();
  }
}

void UI::build_main_menu_bar(const std::vector<std::string>& recent_files) {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("etx-tracer")) {
      if (ImGui::MenuItem("CPU Raytracer", nullptr, _current_renderer_mode == RendererMode::CPURaytracing)) {
        execute_menu_command(MenuCommand::SelectCPURenderer);
      }
      if (ImGui::MenuItem("Rasterizer", nullptr, _current_renderer_mode == RendererMode::Rasterization)) {
        execute_menu_command(MenuCommand::SelectRasterRenderer);
      }
      const char* gpu_renderer_label = _gpu_renderer_available ? "GPU Raytracer" : "GPU Raytracer (Unavailable)";
      if (ImGui::MenuItem(gpu_renderer_label, nullptr, _current_renderer_mode == RendererMode::GPURaytracing, _gpu_renderer_available)) {
        execute_menu_command(MenuCommand::SelectGPURenderer);
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Exit", "Ctrl+Q", false, true)) {
        execute_menu_command(MenuCommand::Quit);
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Scene", true)) {
      if (ImGui::MenuItem("Open...", "Ctrl+O", false, true)) {
        execute_menu_command(MenuCommand::OpenScene);
      }
      if (ImGui::MenuItem("Reload Scene", "Ctrl+R", false, true)) {
        execute_menu_command(MenuCommand::ReloadScene);
      }
      if (ImGui::MenuItem("Reload Geometry and Materials", "Ctrl+G", false, true)) {
        execute_menu_command(MenuCommand::ReloadGeometry);
      }

      if (recent_files.empty() == false) {
        ImGui::Separator();
        bool clear_recent_requested = false;
        if (ImGui::BeginMenu("Recent Files")) {
          for (uint64_t i = recent_files.size(); i > 0; --i) {
            const std::string& entry = recent_files[i - 1u];
            std::string display_name = std::filesystem::path(entry).filename().string();
            if (display_name.empty()) {
              display_name = entry;
            }
            std::string label = display_name + "##recent_" + std::to_string(i - 1u);
            if (ImGui::MenuItem(label.c_str(), nullptr, nullptr)) {
              execute_menu_command(MenuCommand::OpenRecentScene, 0u, entry);
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
              ImGui::SetTooltip("%s", entry.c_str());
            }
          }
          ImGui::Separator();
          if (ImGui::MenuItem("Clear Recent Files")) {
            clear_recent_requested = true;
          }
          ImGui::EndMenu();
        }
        if (clear_recent_requested) {
          execute_menu_command(MenuCommand::ClearRecentScenes);
        }
      }

      ImGui::Separator();
      if (ImGui::MenuItem("Save", nullptr, false, true)) {
        execute_menu_command(MenuCommand::SaveScene);
      }
      if (ImGui::MenuItem("Save as...", nullptr, false, true)) {
        execute_menu_command(MenuCommand::SaveSceneAs);
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Integrator", true)) {
      for (uint64_t i = 0; i < _integrators.count; ++i) {
        if (ImGui::MenuItem(_integrators[i]->name(), nullptr, _current_integrator == _integrators[i], _integrators[i]->enabled())) {
          execute_menu_command(MenuCommand::SelectIntegrator, static_cast<uint32_t>(i));
        }
      }

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Image", true)) {
      if (ImGui::MenuItem("Open Reference Image...", "Ctrl+I", false, true)) {
        execute_menu_command(MenuCommand::OpenReferenceImage);
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Save Current Image (RGB)...", "Ctrl+S", false, true)) {
        execute_menu_command(MenuCommand::SaveImageRGB);
      }
      if (ImGui::MenuItem("Save Current Image (LDR)...", "Shift+Ctrl+S", false, true)) {
        execute_menu_command(MenuCommand::SaveImageLDR);
      }
      if (ImGui::MenuItem("Use as Reference", "Ctrl+Shift+R", false, true)) {
        execute_menu_command(MenuCommand::UseImageAsReference);
      }
      ImGui::EndMenu();
    }

    if (callbacks.view_scene && ImGui::BeginMenu("View", true)) {
      if (ImGui::MenuItem("View whole scene", nullptr, false, true)) {
        execute_menu_command(MenuCommand::ViewWholeScene);
      }
      if (ImGui::BeginMenu("View scene")) {
        if (ImGui::MenuItem("From +X", nullptr, false, true)) {
          execute_menu_command(MenuCommand::ViewPositiveX);
        }
        if (ImGui::MenuItem("From -X", nullptr, false, true)) {
          execute_menu_command(MenuCommand::ViewNegativeX);
        }
        if (ImGui::MenuItem("From +Y", nullptr, false, true)) {
          execute_menu_command(MenuCommand::ViewPositiveY);
        }
        if (ImGui::MenuItem("From -Y", nullptr, false, true)) {
          execute_menu_command(MenuCommand::ViewNegativeY);
        }
        if (ImGui::MenuItem("From +Z", nullptr, false, true)) {
          execute_menu_command(MenuCommand::ViewPositiveZ);
        }
        if (ImGui::MenuItem("From -Z", nullptr, false, true)) {
          execute_menu_command(MenuCommand::ViewNegativeZ);
        }
        ImGui::EndMenu();
      }

      ImGui::Separator();

      if (ImGui::MenuItem("Increase Exposure", "*", false, true)) {
        execute_menu_command(MenuCommand::IncreaseExposure);
      }
      if (ImGui::MenuItem("Decrease Exposure", "/", false, true)) {
        execute_menu_command(MenuCommand::DecreaseExposure);
      }

      ImGui::Separator();

      auto ui_toggle = [this](const char* label, uint32_t flag, MenuCommand command) {
        uint32_t k = 0;
        for (; (k < 8) && (flag != (1u << k)); ++k) {
        }
        const char* buffer = format_string("F%u", k + 1u);
        bool ui_integrator = (_ui_setup & flag) == flag;
        if (ImGui::MenuItem(label, buffer, ui_integrator, true)) {
          execute_menu_command(command);
        }
      };
      ui_toggle("Scene Objects", UIObjects, MenuCommand::ToggleSceneObjects);
      ui_toggle("Properties", UIProperties, MenuCommand::ToggleProperties);
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }
}

void UI::build_toolbar(const BuildContext& ctx) {
  if (ImGui::BeginViewportSideBar("##toolbar", ImGui::GetMainViewport(), ImGuiDir_Up, ctx.button_size + 2.0f * ctx.wpadding.y, ImGuiWindowFlags_NoDecoration)) {
    const bool cpu_mode = _current_renderer_mode == RendererMode::CPURaytracing;
    if (cpu_mode) {
      bool can_run = ctx.has_integrator && _current_integrator->can_run();
      Integrator::State state = can_run ? _current_integrator->state() : Integrator::State::Stopped;

      bool state_available[4] = {
        can_run && (state == Integrator::State::Stopped),
        can_run && (state == Integrator::State::Running),
        can_run && (state != Integrator::State::Stopped),
        can_run && (state == Integrator::State::Running),
      };

      std::string labels[4] = {
        (state == Integrator::State::Running) ? "> Running <" : "  Launch  ",
        (state == Integrator::State::WaitingForCompletion) ? "> Finishing <" : "  Finish  ",
        " Terminate ",
        (state == Integrator::State::Running) ? " Restart " : "  Restart  ",
      };

      ImGui::SameLine(0.0f, ctx.wpadding.x);
      ImGui::PushStyleColor(ImGuiCol_Button, state_available[0] ? kToolbarLaunchColor : kToolbarDisabledColor);
      if (state_available[0] == false) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Button(labels[0].c_str(), {0.0f, ctx.button_size}) && (state_available[0] == true)) {
        callbacks.run_selected();
      }
      if (state_available[0] == false) {
        ImGui::EndDisabled();
      }

      ImGui::SameLine(0.0f, ctx.wpadding.x);
      ImGui::PushStyleColor(ImGuiCol_Button, state_available[1] ? kToolbarFinishColor : kToolbarDisabledColor);
      if (state_available[1] == false) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Button(labels[1].c_str(), {0.0f, ctx.button_size}) && (state_available[1] == true)) {
        callbacks.stop_selected(true);
      }
      if (state_available[1] == false) {
        ImGui::EndDisabled();
      }

      ImGui::SameLine(0.0f, ctx.wpadding.x);
      ImGui::PushStyleColor(ImGuiCol_Button, state_available[2] ? kToolbarTerminateColor : kToolbarDisabledColor);
      if (state_available[2] == false) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Button(labels[2].c_str(), {0.0f, ctx.button_size}) && (state_available[2] == true)) {
        callbacks.stop_selected(false);
      }
      if (state_available[2] == false) {
        ImGui::EndDisabled();
      }

      ImGui::SameLine(0.0f, ctx.wpadding.x);
      ImGui::PushStyleColor(ImGuiCol_Button, state_available[3] ? kToolbarRestartColor : kToolbarDisabledColor);
      if (state_available[3] == false) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Button(labels[3].c_str(), {0.0f, ctx.button_size}) && (state_available[3] == true)) {
        callbacks.restart_selected();
      }
      if (state_available[3] == false) {
        ImGui::EndDisabled();
      }
      ImGui::PopStyleColor(4);

      ImGui::SameLine(0.0f, ctx.wpadding.x);
      ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);

      ImGui::SameLine(0.0f, ctx.wpadding.x);
      if (state_available[0] == false) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Button("  Denoise (preview)  ", {0.0f, ctx.button_size})) {
        callbacks.denoise_selected();
      }
      if (state_available[0] == false) {
        ImGui::EndDisabled();
      }
    } else {
      const bool gpu_mode = _current_renderer_mode == RendererMode::GPURaytracing;
      const RendererPreparationStatus status = _current_renderer_status;
      const char* label = "  Ready  ";
      ImVec4 status_color = kToolbarLaunchColor;
      if (status.state == RendererPreparationState::Preparing) {
        label = " Preparing... ";
        status_color = kToolbarFinishColor;
      } else if (status.state == RendererPreparationState::Failed) {
        label = "  Failed  ";
        status_color = kToolbarTerminateColor;
      }

      ImGui::SameLine(0.0f, ctx.wpadding.x);
      ImGui::PushStyleColor(ImGuiCol_Button, status_color);
      ImGui::BeginDisabled();
      ImGui::Button(label, {0.0f, ctx.button_size});
      ImGui::EndDisabled();
      ImGui::PopStyleColor();

      if (gpu_mode) {
        ImGui::SameLine(0.0f, ctx.wpadding.x);
        if (ImGui::Button((status.state == RendererPreparationState::Failed) ? "  Retry  " : "  Reload Shaders  ", {0.0f, ctx.button_size})) {
          if (callbacks.reload_shaders_selected) {
            callbacks.reload_shaders_selected();
          }
        }

        ImGui::SameLine(0.0f, ctx.wpadding.x);
        if (status.state != RendererPreparationState::Preparing) {
          ImGui::BeginDisabled();
        }
        if (ImGui::Button("  Cancel  ", {0.0f, ctx.button_size})) {
          if (callbacks.cancel_renderer_preparation_selected) {
            callbacks.cancel_renderer_preparation_selected();
          }
        }
        if (status.state != RendererPreparationState::Preparing) {
          ImGui::EndDisabled();
        }
      }
    }

    ImGui::SameLine(0.0f, ctx.wpadding.x);
    ImGui::GetStyle().FramePadding.y = (ctx.button_size - ctx.text_size) / 2.0f;
    ImGui::PushItemWidth(ctx.input_size);
    {
      ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
      ImGui::SameLine(0.0f, ctx.wpadding.x);
      ImGui::DragFloat("Exposure", &_view_options.exposure, 1.0f / 256.0f, 1.0f / 1024.0f, 1024.0f, "%.4f", ImGuiSliderFlags_NoRoundToFormat);
      ImGui::SameLine(0.0f, ctx.wpadding.x);
    }
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine(0.0f, ctx.wpadding.x);
    ImGui::PopItemWidth();

    ImGui::PushItemWidth(2.25f * ctx.input_size);
    if (ImGui::BeginCombo("##view_layer", Film::layer_name(_view_options.view_layer))) {
      for (uint32_t i = 0; i < ViewLayer::Count; ++i) {
        bool selected = i == _view_options.view_layer;
        if (ImGui::Selectable(Film::layer_name(i), &selected)) {
          if (selected) {
            _view_options.view_layer = i;
          }
        }
      }
      ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    ImGui::SameLine(0.0f, ctx.wpadding.x);
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine(0.0f, ctx.wpadding.x);
    ImGui::PushItemWidth(2.25f * ctx.input_size);
    if (ImGui::BeginCombo("##view_img", output_view_to_string(uint32_t(_view_options.view_image)).c_str())) {
      for (uint32_t i = 0; i < uint32_t(OutputView::Count); ++i) {
        bool selected = i == uint32_t(_view_options.view_image);
        if (ImGui::Selectable(output_view_to_string(i).c_str(), &selected)) {
          _view_options.view_image = i;
        }
      }
      ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    ImGui::SameLine(0.0f, ctx.wpadding.x);
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine(0.0f, ctx.wpadding.x);
    ImGui::PushItemWidth(2.25f * ctx.input_size);
    if (ImGui::BeginCombo("##view_opt", view_option_to_string(uint32_t(_view_options.view_option)).c_str())) {
      for (uint32_t i = 0; i < uint32_t(ViewOptions::Count); ++i) {
        bool selected = i == uint32_t(_view_options.view_option);
        if (ImGui::Selectable(view_option_to_string(i).c_str(), &selected)) {
          _view_options.view_option = i;
        }
      }
      ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    ImGui::GetStyle().FramePadding.y = ctx.fpadding.y;
    ImGui::End();
  }

  if (ImGui::BeginViewportSideBar("##status", ImGui::GetMainViewport(), ImGuiDir_Down, ctx.text_size + 2.0f * ctx.wpadding.y, ImGuiWindowFlags_NoDecoration)) {
    if (_current_renderer_mode == RendererMode::CPURaytracing) {
      constexpr const char* status_str[] = {
        "Stopped",
        "Running",
        "Completing",
      };

      auto status = _current_integrator ? _current_integrator->status() : Integrator::Status{};
      auto state = _current_integrator ? _current_integrator->state() : Integrator::State::Stopped;

      double average_time = status.completed_iterations > 0 ? status.total_time / status.completed_iterations : 0.0;

      const char* buffer = format_string("%-4d | %s | %.3fms last, %.3fms avg, %.3fs total | %.1f FPS", status.completed_iterations, status_str[uint32_t(state)],
        status.last_iteration_time * 1000.0, average_time * 1000.0f, status.total_time, _current_fps);

      ImGui::Text("%s", buffer);
    } else {
      const RendererPreparationStatus status = _current_renderer_status;
      const char* state_str = "Ready";
      if (status.state == RendererPreparationState::Preparing) {
        state_str = "Preparing";
      } else if (status.state == RendererPreparationState::Failed) {
        state_str = "Failed";
      }

      std::string progress = {};
      if (status.total_steps > 0u) {
        progress = format_string("%u/%u", status.completed_steps, status.total_steps);
      } else {
        progress = "-";
      }

      const RendererRuntimeStats stats = _current_renderer_stats;
      const bool gpu_stats_available = (_current_renderer_mode == RendererMode::GPURaytracing) && stats.valid;
      const char* buffer = nullptr;
      if (gpu_stats_available) {
        const std::string sample_progress =
          (stats.target_samples > 0u) ? (std::to_string(stats.completed_samples) + "/" + std::to_string(stats.target_samples)) : std::to_string(stats.completed_samples);
        const std::string elapsed = duration_string(stats.elapsed_seconds);
        const std::string remaining = duration_string(stats.estimated_remaining_seconds);
        buffer = format_string("%s | %s | %s | sample %s | elapsed %s | remaining %s | %.1f FPS",
          (_current_renderer_mode == RendererMode::GPURaytracing) ? "GPU Raytracing" : "Rasterization", state_str, status.phase.empty() ? "-" : status.phase.c_str(),
          sample_progress.c_str(), elapsed.c_str(), remaining.c_str(), _current_fps);
      } else {
        buffer = format_string("%s | %s | %s | %s | %.1f FPS", (_current_renderer_mode == RendererMode::GPURaytracing) ? "GPU Raytracing" : "Rasterization", state_str,
          status.phase.empty() ? "-" : status.phase.c_str(), progress.c_str(), _current_fps);
      }
      ImGui::Text("%s", buffer);
      if (status.message.empty() == false) {
        ImGui::SameLine(0.0f, ctx.wpadding.x);
        ImGui::TextUnformatted(status.message.c_str());
      }
    }
    ImGui::End();
  }
}

void UI::build_scene_objects_window(SceneRepresentation& scene_rep, const BuildContext& ctx) {
  const float kDefaultListHeight = 5.0f * ImGui::GetTextLineHeightWithSpacing();

  ctx.with_window(UIObjects, "Scene Objects", [&]() {
    auto draw_history_button = [&](const char* label, bool enabled, int32_t step) {
      ImGui::PushStyleColor(ImGuiCol_Button, kHistoryButtonBaseColor);
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kHistoryButtonHoverColor);
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, kHistoryButtonActiveColor);
      if (enabled == false)
        ImGui::BeginDisabled();
      if (ImGui::Button(label)) {
        navigate_history(step);
      }
      if (enabled == false)
        ImGui::EndDisabled();
      ImGui::PopStyleColor(3);
    };

    ImGui::AlignTextToFramePadding();
    ImGui::Text("Navigate");
    ImGui::SameLine();
    draw_history_button(" < ##selection_history_back", can_navigate_back(), -1);
    ImGui::SameLine();
    draw_history_button(" > ##selection_history_forward", can_navigate_forward(), 1);

    ImGui::SameLine();
    const char* add_scene_object_popup_id = "##add_scene_object_popup";
    if (ImGui::Button(" + ##add_scene_object")) {
      ImGui::OpenPopup(add_scene_object_popup_id);
    }
    ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 20.0f, 0.0f), ImGuiCond_Always);
    if (ImGui::BeginPopup(add_scene_object_popup_id)) {
      if (ImGui::Selectable("Add Material")) {
        if (callbacks.material_added) {
          callbacks.material_added();
          _material_mapping.build(scene_rep.material_mapping());
          _material_mapping_hash = hash_mapping(scene_rep.material_mapping());
        }
        ImGui::CloseCurrentPopup();
      }
      if (ImGui::Selectable("Add Medium")) {
        if (callbacks.medium_added) {
          callbacks.medium_added();
          _medium_mapping.build(scene_rep.medium_mapping());
          _medium_mapping_hash = hash_mapping(scene_rep.medium_mapping());
        }
        ImGui::CloseCurrentPopup();
      }
      ImGui::Separator();
      if (ImGui::Selectable("Add Environment Emitter")) {
        if (callbacks.emitter_added) {
          callbacks.emitter_added(0);
        }
        ImGui::CloseCurrentPopup();
      }
      if (ImGui::Selectable("Add Directional Emitter")) {
        if (callbacks.emitter_added) {
          callbacks.emitter_added(1);
        }
        ImGui::CloseCurrentPopup();
      }
      if (ImGui::Selectable("Add Atmosphere Emitter")) {
        if (callbacks.emitter_added) {
          callbacks.emitter_added(2);
        }
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }

    ImGui::Spacing();
    ImGui::Separator();

    ImGui::PushStyleColor(ImGuiCol_Text, kSceneTextColor);
    bool rendering_selected = (_selection.kind == SelectionKind::Rendering);
    if (ImGui::Selectable("Rendering", rendering_selected)) {
      set_selection(SelectionKind::Rendering, 0);
    }
    ImGui::PopStyleColor();

    ImGui::Separator();

    ImGui::Text("Cameras (%zu)", static_cast<size_t>(_camera_mapping.size()));
    if (_camera_mapping.empty()) {
      ImGui::TextDisabled("None");
    } else if (ImGui::BeginListBox("##cameras_list", ImVec2(-FLT_MIN, kDefaultListHeight))) {
      for (uint64_t i = 0; i < _camera_mapping.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        bool camera_selected = (_selection.kind == SelectionKind::Camera) && (_selection.index == static_cast<int32_t>(i));
        const auto& entry = _camera_mapping.entry(static_cast<int32_t>(i));
        uint32_t camera_index = entry.index;

        bool is_active = (camera_index < scene_rep.data().cameras.size()) && scene_rep.data().cameras[camera_index].active;
        if (is_active) {
          ImGui::PushStyleColor(ImGuiCol_Text, kCameraTextColor);
          std::string display_name = std::string("[") + entry.name + "]";
          if (ImGui::Selectable(display_name.c_str(), camera_selected)) {
            set_selection(SelectionKind::Camera, static_cast<int32_t>(i));
          }
          ImGui::PopStyleColor();
        } else {
          if (ImGui::Selectable(entry.name, camera_selected)) {
            set_selection(SelectionKind::Camera, static_cast<int32_t>(i));
          }
        }

        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0) && callbacks.camera_activated) {
          callbacks.camera_activated(camera_index);
        }
        ImGui::PopID();
      }
      ImGui::EndListBox();
    }

    ImGui::Separator();

    ImGui::Text("Materials (%zu)", static_cast<size_t>(_material_mapping.size()));
    if (_material_mapping.empty()) {
      ImGui::TextDisabled("None");
    } else if (ImGui::BeginListBox("##materials_list", ImVec2(-FLT_MIN, kDefaultListHeight))) {
      for (uint64_t i = 0; i < _material_mapping.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        const int32_t list_index = static_cast<int32_t>(i);
        bool material_selected = (_selection.kind == SelectionKind::Material) && material_list_position_selected(list_index);
        const auto& entry = _material_mapping.entry(static_cast<int32_t>(i));
        if (ImGui::Selectable(entry.name, material_selected)) {
          const ImGuiIO& io = ImGui::GetIO();
          if (io.KeyShift) {
            set_material_selection_range(list_index);
          } else if ((io.KeyCtrl) || (io.KeySuper)) {
            toggle_material_selection(list_index);
          } else {
            set_single_material_selection(list_index, true);
          }
        }
        ImGui::PopID();
      }
      ImGui::EndListBox();
    }

    ImGui::Separator();

    ImGui::Text("Meshes (%zu)", static_cast<size_t>(_mesh_mapping.size()));
    if (_mesh_mapping.empty()) {
      ImGui::TextDisabled("None");
    } else if (ImGui::BeginListBox("##meshes_list", ImVec2(-FLT_MIN, kDefaultListHeight))) {
      for (uint64_t i = 0; i < _mesh_mapping.size(); ++i) {
        ImGui::PushID(static_cast<int>(i + 512));  // Use different ID range
        bool mesh_selected = (_selection.kind == SelectionKind::Mesh) && (_selection.index == static_cast<int32_t>(i));
        const auto& entry = _mesh_mapping.entry(static_cast<int32_t>(i));
        if (ImGui::Selectable(entry.name, mesh_selected)) {
          set_selection(SelectionKind::Mesh, static_cast<int32_t>(i));
        }
        ImGui::PopID();
      }
      ImGui::EndListBox();
    }

    ImGui::Separator();

    ImGui::AlignTextToFramePadding();
    ImGui::Text("Mediums (%zu)", static_cast<size_t>(_medium_mapping.size()));
    ImGui::Spacing();
    if (_medium_mapping.empty()) {
      ImGui::TextDisabled("None");
    } else if (ImGui::BeginListBox("##mediums_list", ImVec2(-FLT_MIN, kDefaultListHeight))) {
      for (uint64_t i = 0; i < _medium_mapping.size(); ++i) {
        ImGui::PushID(static_cast<int>(i + 1024));
        bool medium_selected = (_selection.kind == SelectionKind::Medium) && (_selection.index == static_cast<int32_t>(i));
        const auto& entry = _medium_mapping.entry(static_cast<int32_t>(i));
        if (ImGui::Selectable(entry.name, medium_selected)) {
          set_selection(SelectionKind::Medium, static_cast<int32_t>(i));
        }
        ImGui::PopID();
      }
      ImGui::EndListBox();
    }

    ImGui::Separator();
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Emitters (%zu)", scene_rep.data().emitter_profiles.size());
    ImGui::Spacing();
    if (scene_rep.data().emitter_profiles.empty()) {
      ImGui::TextDisabled("None");
    } else if (ImGui::BeginListBox("##emitters_list", ImVec2(-FLT_MIN, kDefaultListHeight))) {
      for (uint32_t emitter_index = 0; emitter_index < scene_rep.data().emitter_profiles.size(); ++emitter_index) {
        const auto& emitter = scene_rep.data().emitter_profiles[emitter_index];
        const char* label = nullptr;
        switch (emitter.cls) {
          case EmitterProfile::Class::Area: {
            const char* material_name = nullptr;
            // Find a triangle that uses this emitter profile to get material name
            for (size_t tri_idx = 0; tri_idx < scene_rep.data().triangles.size(); ++tri_idx) {
              const Triangle& tri = scene_rep.data().triangles[tri_idx];
              if (tri.emitter_index == emitter_index && tri.material_index < scene_rep.data().materials.size()) {
                material_name = _material_mapping.name_for(tri.material_index);
                if (material_name == nullptr) {
                  label = format_string("%u: area (material %u)", emitter_index, tri.material_index);
                } else {
                  label = format_string("%u: area (%s)", emitter_index, material_name);
                }
                break;
              }
            }
            if (material_name == nullptr) {
              label = format_string("%u: area", emitter_index);
            }
            break;
          }
          case EmitterProfile::Class::Directional:
            label = format_string("%u: directional", emitter_index);
            break;
          case EmitterProfile::Class::Environment:
            if ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0) {
              label = format_string("%u: atmosphere", emitter_index);
            } else {
              label = format_string("%u: environment", emitter_index);
            }
            break;
          default:
            label = format_string("%u", emitter_index);
            break;
        }
        ImGui::PushID(static_cast<int>(emitter_index + 2048));
        bool emitter_selected = (_selection.kind == SelectionKind::Emitter) && (_selection.index == static_cast<int32_t>(emitter_index));
        if (ImGui::Selectable(label, emitter_selected)) {
          set_selection(SelectionKind::Emitter, static_cast<int32_t>(emitter_index));
        }
        ImGui::PopID();
      }
      ImGui::EndListBox();
    }
  });
}

void UI::build_properties_window(SceneRepresentation& scene_rep, Camera& camera, const BuildContext& ctx, const FrameData& data) {
  if ((_ui_setup & UIProperties) == 0)
    return;

  std::string properties_title = "Properties";
  auto title_suffix = [&](const char* type, const char* name) {
    properties_title = type;
    if (name != nullptr && (name[0] != '\0')) {
      properties_title += std::string(": ") + name;
    }
  };

  switch (_selection.kind) {
    case SelectionKind::Material:
      if (selected_material_count() > 1u) {
        properties_title = format_string("Materials: %u selected", selected_material_count());
      } else if ((_selection.index >= 0) && (static_cast<uint64_t>(_selection.index) < _material_mapping.size())) {
        title_suffix("Material", _material_mapping.name(_selection.index));
      }
      break;
    case SelectionKind::Medium:
      if ((_selection.index >= 0) && (static_cast<uint64_t>(_selection.index) < _medium_mapping.size())) {
        uint32_t medium_index = _medium_mapping.at(_selection.index);
        if (medium_index < scene_rep.data().mediums_vector.size()) {
          const Medium& medium = scene_rep.data().mediums_vector[medium_index];
          constexpr const char* kMediumClassNames[] = {"Homogeneous Medium", "Density Grid Medium"};
          int32_t class_index = static_cast<int32_t>(medium.cls);
          class_index = clamp(class_index, 0, static_cast<int32_t>(sizeof(kMediumClassNames) / sizeof(kMediumClassNames[0])) - 1);
          title_suffix(kMediumClassNames[class_index], nullptr);
        }
      }
      break;
    case SelectionKind::Emitter: {
      if ((_selection.index >= 0) && (static_cast<uint32_t>(_selection.index) < scene_rep.data().emitter_profiles.size())) {
        uint32_t emitter_index = static_cast<uint32_t>(_selection.index);
        const auto& emitter = scene_rep.data().emitter_profiles[emitter_index];
        const char* emitter_type = nullptr;
        switch (emitter.cls) {
          case EmitterProfile::Class::Directional:
            emitter_type = ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0) ? "Sun" : "Directional";
            break;
          case EmitterProfile::Class::Environment:
            emitter_type = ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0) ? "Sky" : "Environment";
            break;
          case EmitterProfile::Class::Area:
            emitter_type = "Area";
            break;
          default:
            emitter_type = "Emitter";
            break;
        }
        title_suffix(emitter_type, nullptr);
      } else {
        title_suffix("Emitter", nullptr);
      }
      break;
    }
    case SelectionKind::Mesh:
      if ((_selection.index >= 0) && (static_cast<uint64_t>(_selection.index) < _mesh_mapping.size())) {
        title_suffix("Mesh", _mesh_mapping.name(_selection.index));
      }
      break;
    case SelectionKind::Camera:
      if (_selection.index >= 0 && _selection.index < int32_t(_camera_mapping.size())) {
        title_suffix("Camera", _camera_mapping.name(_selection.index));
      } else {
        title_suffix("Camera", nullptr);
      }
      break;
    case SelectionKind::Rendering:
      title_suffix("Rendering", nullptr);
      break;
    default:
      break;
  }

  std::string properties_window_name = properties_title + "###properties";
  ctx.with_window(UIProperties, properties_window_name.c_str(), [&]() {
    switch (_selection.kind) {
      case SelectionKind::Material: {
        build_material_selection_properties(scene_rep, ctx, data);
        break;
      }
      case SelectionKind::Medium: {
        build_medium_selection_properties(scene_rep, ctx, data);
        break;
      }
      case SelectionKind::Mesh: {
        build_mesh_selection_properties(scene_rep, ctx);
        break;
      }
      case SelectionKind::Emitter: {
        build_emitter_selection_properties(scene_rep, ctx, data);
        break;
      }
      case SelectionKind::Camera: {
        if (_selection.index >= 0 && _selection.index < int32_t(_camera_mapping.size())) {
          uint32_t camera_index = _camera_mapping.at(_selection.index);
          if (camera_index < scene_rep.data().cameras.size()) {
            Camera* selected_camera = &scene_rep.data().cameras[camera_index].cam;
            if (scene_rep.data().cameras[camera_index].active) {
              selected_camera = &scene_rep.mutable_camera();
            }
            build_camera_selection_properties(scene_rep, *selected_camera, camera_index, ctx, data);
          }
        }
        break;
      }
      case SelectionKind::Rendering: {
        build_rendering_properties(scene_rep, ctx, data);
        break;
      }
      default:
        ImGui::Text("No Object Selected");
        break;
    }
  });
}

bool UI::build_material_class_selector(Material& material) {
  return build_material_class_selector(material, false);
}

bool UI::build_material_class_selector(Material& material, bool mixed) {
  bool changed = false;

  const char* material_name = mixed ? "mixed" : material_class_display_name(material.cls);
  ImVec2 button_size = ImVec2(ImGui::GetContentRegionAvail().x, 0.0f);
  const char* button_label = format_string("%s##material_class", material_name);
  if (ImGui::Button(button_label, button_size)) {
    ImGui::OpenPopup("material_class_popup");
  }

  ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 56.0f, 0.0f), ImGuiCond_Always);
  if (ImGui::BeginPopup("material_class_popup")) {
    const ImVec4 header_colors[] = {
      kMaterialHeaderPrimaryColor,
      kMaterialHeaderSpecializedColor,
      kMaterialHeaderInterfacesColor,
    };

    ImGui::Columns(3, "material_class_columns", true);

    auto draw_material_column = [&](uint32_t column_index, const char* title, std::initializer_list<Material::Class> entries) {
      ImGui::PushStyleColor(ImGuiCol_Text, header_colors[column_index % (sizeof(header_colors) / sizeof(header_colors[0]))]);
      ImGui::Text("%s", title);
      ImGui::PopStyleColor();
      for (auto cls : entries) {
        const char* material_name = material_class_display_name(cls);
        const char* selectable_label = format_string("%s##cls_%u", material_name, static_cast<uint32_t>(cls));
        const bool is_selected = (material.cls == cls);
        if (ImGui::Selectable(selectable_label, is_selected)) {
          if (material.cls != cls) {
            material.cls = cls;
            changed = true;
          }
          ImGui::CloseCurrentPopup();
        }
      }
    };

    uint32_t column_index = 0u;
    draw_material_column(column_index++, "Primary", {MaterialClass::Diffuse, MaterialClass::Plastic, MaterialClass::Conductor, MaterialClass::Dielectric});
    ImGui::NextColumn();
    draw_material_column(column_index++, "Specialized",
      {MaterialClass::OpenPBR, MaterialClass::Translucent, MaterialClass::Thinfilm, MaterialClass::DiffractionGrating, MaterialClass::Velvet, MaterialClass::Mirror});
    ImGui::NextColumn();
    draw_material_column(column_index++, "Interfaces", {MaterialClass::Boundary, MaterialClass::Void});

    ImGui::Columns(1);
    ImGui::EndPopup();
  }

  return changed;
}

void UI::build_material_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  if (_material_mapping.empty()) {
    ImGui::Text("No materials available");
    return;
  }
  if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= _material_mapping.size())) {
    ImGui::Text("Invalid material selection");
    return;
  }

  std::vector<uint32_t> material_indices = selected_material_indices(scene_rep);
  if (material_indices.empty()) {
    ImGui::Text("Invalid material selection");
    return;
  }

  uint32_t material_index = material_indices.front();
  Material& material = scene_rep.data().materials[material_index];
  const char* material_name = _material_mapping.name(_selection.index);

  if (material_indices.size() == 1u) {
    update_name_buffer(SelectionKind::Material, _selection.index, material_name);

    ImGui::AlignTextToFramePadding();
    ImGui::Text("Name");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    const bool name_edit_active = ImGui::InputText("##material_name", _name_edit_buffer, sizeof(_name_edit_buffer), ImGuiInputTextFlags_AutoSelectAll);
    const bool name_commit = (ImGui::IsItemDeactivatedAfterEdit() || (name_edit_active && ImGui::IsKeyPressed(ImGuiKey_Enter)));
    if (name_commit) {
      if (callbacks.material_renamed) {
        callbacks.material_renamed(material_index, std::string(_name_edit_buffer));
        _pending_selection = {SelectionKind::Material, material_index, true};
      }
    }
    bool changed = build_material(scene_rep, material, data);
    if (changed) {
      if (callbacks.material_changed) {
        callbacks.material_changed(material_index);
      }
    }
    return;
  }

  ImGui::Text("%u materials selected", static_cast<uint32_t>(material_indices.size()));
  ImGui::Separator();

  Material edit_material = material;
  const Material before = edit_material;
  auto spectrum_value = [&](uint32_t spectrum_index) {
    SpectralDistribution result = {};
    if (spectrum_index < scene_rep.data().spectrum_values.size()) {
      result = scene_rep.data().spectrum_values[spectrum_index];
    }
    return result;
  };

  const SpectralDistribution before_reflectance = spectrum_value(before.reflectance.spectrum_index);
  const SpectralDistribution before_scattering = spectrum_value(before.scattering.spectrum_index);
  const SpectralDistribution before_emission = spectrum_value(before.emission.spectrum_index);
  const SpectralDistribution before_subsurface = spectrum_value(before.subsurface.spectrum_index);
  const SpectralDistribution before_int_eta = spectrum_value(before.int_ior.eta_index);
  const SpectralDistribution before_int_k = spectrum_value(before.int_ior.k_index);
  const SpectralDistribution before_ext_eta = spectrum_value(before.ext_ior.eta_index);
  const SpectralDistribution before_ext_k = spectrum_value(before.ext_ior.k_index);
  const SpectralDistribution before_thinfilm_eta = spectrum_value(before.thinfilm.ior.eta_index);
  const SpectralDistribution before_thinfilm_k = spectrum_value(before.thinfilm.ior.k_index);

  _material_batch_changed_fields = 0u;
  const bool changed = build_material(scene_rep, edit_material, data, material_indices);
  const uint32_t changed_fields = _material_batch_changed_fields;
  _material_batch_changed_fields = 0u;
  if (changed == false) {
    return;
  }

  auto apply_spectrum_change = [&](const SpectralDistribution& before_spectrum, uint32_t source_spectrum_index, bool force_apply, auto target_spectrum_index) {
    if (source_spectrum_index >= scene_rep.data().spectrum_values.size()) {
      return;
    }

    const SpectralDistribution& after_spectrum = scene_rep.data().spectrum_values[source_spectrum_index];
    if ((force_apply == false) && (std::memcmp(&before_spectrum, &after_spectrum, sizeof(SpectralDistribution)) == 0)) {
      return;
    }

    for (const uint32_t selected_material_index : material_indices) {
      if (selected_material_index >= scene_rep.data().materials.size()) {
        continue;
      }

      const uint32_t target_index = target_spectrum_index(scene_rep.data().materials[selected_material_index]);
      if (target_index < scene_rep.data().spectrum_values.size()) {
        scene_rep.data().spectrum_values[target_index] = after_spectrum;
      }
    }
  };

  apply_spectrum_change(before_reflectance, before.reflectance.spectrum_index, false, [](const Material& value) {
    return value.reflectance.spectrum_index;
  });
  apply_spectrum_change(before_scattering, before.scattering.spectrum_index, false, [](const Material& value) {
    return value.scattering.spectrum_index;
  });
  apply_spectrum_change(before_emission, before.emission.spectrum_index, false, [](const Material& value) {
    return value.emission.spectrum_index;
  });
  apply_spectrum_change(before_subsurface, before.subsurface.spectrum_index, false, [](const Material& value) {
    return value.subsurface.spectrum_index;
  });
  const bool int_ior_changed = (changed_fields & MaterialBatchChangedIntIOR) != 0u;
  const bool ext_ior_changed = (changed_fields & MaterialBatchChangedExtIOR) != 0u;
  const bool thinfilm_ior_changed = (changed_fields & MaterialBatchChangedThinfilmIOR) != 0u;
  apply_spectrum_change(before_int_eta, before.int_ior.eta_index, int_ior_changed, [](const Material& value) {
    return value.int_ior.eta_index;
  });
  apply_spectrum_change(before_int_k, before.int_ior.k_index, int_ior_changed, [](const Material& value) {
    return value.int_ior.k_index;
  });
  apply_spectrum_change(before_ext_eta, before.ext_ior.eta_index, ext_ior_changed, [](const Material& value) {
    return value.ext_ior.eta_index;
  });
  apply_spectrum_change(before_ext_k, before.ext_ior.k_index, ext_ior_changed, [](const Material& value) {
    return value.ext_ior.k_index;
  });
  apply_spectrum_change(before_thinfilm_eta, before.thinfilm.ior.eta_index, thinfilm_ior_changed, [](const Material& value) {
    return value.thinfilm.ior.eta_index;
  });
  apply_spectrum_change(before_thinfilm_k, before.thinfilm.ior.k_index, thinfilm_ior_changed, [](const Material& value) {
    return value.thinfilm.ior.k_index;
  });

  if ((changed_fields & MaterialBatchChangedClass) != 0u) {
    for (const uint32_t selected_material_index : material_indices) {
      if (selected_material_index < scene_rep.data().materials.size()) {
        scene_rep.data().materials[selected_material_index].cls = edit_material.cls;
      }
    }
  }

  if (int_ior_changed) {
    for (const uint32_t selected_material_index : material_indices) {
      if (selected_material_index < scene_rep.data().materials.size()) {
        scene_rep.data().materials[selected_material_index].int_ior.cls = edit_material.int_ior.cls;
      }
    }
  }

  if (ext_ior_changed) {
    for (const uint32_t selected_material_index : material_indices) {
      if (selected_material_index < scene_rep.data().materials.size()) {
        scene_rep.data().materials[selected_material_index].ext_ior.cls = edit_material.ext_ior.cls;
      }
    }
  }

  if (thinfilm_ior_changed) {
    for (const uint32_t selected_material_index : material_indices) {
      if (selected_material_index < scene_rep.data().materials.size()) {
        scene_rep.data().materials[selected_material_index].thinfilm.ior.cls = edit_material.thinfilm.ior.cls;
      }
    }
  }

  apply_material_changes(scene_rep, material_indices, before, edit_material);
  if (callbacks.material_changed) {
    for (const uint32_t changed_material_index : material_indices) {
      callbacks.material_changed(changed_material_index);
    }
  }
}

void UI::build_medium_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  if (_medium_mapping.empty()) {
    ImGui::Text("No mediums available");
    return;
  }
  if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= _medium_mapping.size())) {
    ImGui::Text("Invalid medium selection");
    return;
  }
  uint32_t medium_index = _medium_mapping.at(_selection.index);
  Medium& medium = scene_rep.data().mediums.get(medium_index);
  const char* medium_name = _medium_mapping.name(_selection.index);
  update_name_buffer(SelectionKind::Medium, _selection.index, medium_name);
  ImGui::AlignTextToFramePadding();
  ImGui::Text("Name");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  bool name_edit_active = ImGui::InputText("##medium_name", _name_edit_buffer, sizeof(_name_edit_buffer), ImGuiInputTextFlags_AutoSelectAll);
  bool name_commit = ImGui::IsItemDeactivatedAfterEdit() || (name_edit_active && ImGui::IsKeyPressed(ImGuiKey_Enter));
  if (name_commit && callbacks.medium_renamed) {
    callbacks.medium_renamed(medium_index, std::string(_name_edit_buffer));
    _pending_selection = {SelectionKind::Medium, medium_index, true};
  }
  bool changed = build_medium(scene_rep, medium);
  if (changed) {
    if (callbacks.medium_changed) {
      callbacks.medium_changed(medium_index);
    }
  }
}

void UI::build_emitter_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  if ((_selection.index < 0) || (static_cast<uint32_t>(_selection.index) >= scene_rep.data().emitter_profiles.size())) {
    ImGui::Text("Invalid emitter selection");
    return;
  }
  uint32_t emitter_index = static_cast<uint32_t>(_selection.index);
  auto& emitter = scene_rep.data().emitter_profiles[emitter_index];
  bool changed = false;
  bool material_changed = false;
  uint32_t material_index = kInvalidIndex;

  if (emitter.cls == EmitterProfile::Class::Area) {
    // Find a triangle that uses this emitter profile to get the material
    for (size_t tri_idx = 0; tri_idx < scene_rep.data().triangles.size(); ++tri_idx) {
      const Triangle& tri = scene_rep.data().triangles[tri_idx];
      if (tri.emitter_index == emitter_index && tri.material_index < scene_rep.data().materials.size()) {
        material_index = tri.material_index;
        break;
      }
    }
  }

  if ((emitter.cls == EmitterProfile::Class::Area) && (material_index >= scene_rep.data().materials.size())) {
    ImGui::TextDisabled("Area emitter has no material reference");
    return;
  }

  bool common_changed = false;
  if (emitter.cls == EmitterProfile::Class::Area) {
    auto& material = scene_rep.data().materials[material_index];
    std::string area_preset_id = "area_material_emission_" + std::to_string(material_index);
    common_changed = emission_picker(scene_rep, "Emission", area_preset_id.c_str(), material.emission.spectrum_index, data);
    if (common_changed && (material_index < scene_rep.data().materials.size())) {
      float3 integrated = material.emission.spectrum_index < scene_rep.data().spectrum_values.size()  //
                            ? scene_rep.data().spectrum_values[material.emission.spectrum_index].integrated()
                            : float3{0.0f};
      if ((integrated.x <= 0.0f) && (integrated.y <= 0.0f) && (integrated.z <= 0.0f)) {
        for (uint64_t i = 0; i < _material_mapping.size(); ++i) {
          if (_material_mapping.at(static_cast<int32_t>(i)) == material_index) {
            set_selection(SelectionKind::Material, static_cast<int32_t>(i), false);
            _auto_open_emission_section = true;
            break;
          }
        }
      }
    }
  } else {
    std::string emitter_preset_id = "emitter_emission_" + std::to_string(emitter_index);
    common_changed = emission_picker(scene_rep, "Emission", emitter_preset_id.c_str(), emitter.emission.spectrum_index, data);
  }

  if (_medium_mapping.empty() == false) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Text("External Medium");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    if (emitter.cls == EmitterProfile::Class::Area && material_index < scene_rep.data().materials.size()) {
      auto& material = scene_rep.data().materials[material_index];
      if (medium_dropdown("##area_external_medium", material.ext_medium)) {
        common_changed = true;
      }
    } else {
      const char* medium_id = format_string("##emitter_medium_%u", emitter_index);
      if (medium_dropdown(medium_id, emitter.medium_index)) {
        common_changed = true;
      }
    }
  } else {
    ImGui::TextDisabled("No mediums available");
  }

  if (common_changed) {
    if (emitter.cls == EmitterProfile::Class::Area && material_index < scene_rep.data().materials.size()) {
      material_changed = true;
    }
    changed = true;
  }

  if (emitter.cls == EmitterProfile::Class::Area) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    if (material_index < scene_rep.data().materials.size()) {
      auto& material = scene_rep.data().materials[material_index];

      const char* mat_name = _material_mapping.name_for(material_index);
      if (mat_name != nullptr) {
        ImGui::Text("Material: %s", mat_name);
      } else {
        ImGui::Text("Material index: %u", material_index);
      }

      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      float collimation = material.emission_collimation;
      if (ImGui::SliderFloat("##area_collimation", &collimation, 0.0f, 1.0f, "Collimation %.2f", ImGuiSliderFlags_AlwaysClamp)) {
        material.emission_collimation = std::clamp(collimation, 0.0f, 1.0f);
        material_changed = true;
        changed = true;
      }
    }
  } else if (emitter.cls == EmitterProfile::Class::Directional) {
    ImGui::Text("Angular Size");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    float angular_size_deg = emitter.directional.angular_size * 180.0f / kPi;
    if (ImGui::DragFloat("##angularsize", &angular_size_deg, 0.1f, 0.0f, 90.0f, "%.2f°", ImGuiSliderFlags_NoRoundToFormat)) {
      emitter.directional.angular_size = angular_size_deg * kPi / 180.0f;
      emitter.directional.angular_size_cosine = cosf(emitter.directional.angular_size / 2.0f);
      emitter.directional.equivalent_disk_size = 2.0f * std::tan(emitter.directional.angular_size / 2.0f);
      changed = true;
    }

    auto spherical = to_spherical(emitter.directional.direction);
    float2 angles = {spherical.phi, spherical.theta};

    if (angle_editor("Light Direction", angles, -180.0f, 180.0f, -89.99f, 89.99f, 89.99f)) {
      emitter.directional.direction = from_spherical(angles.x, angles.y);
      changed = true;
    }

    bool has_atmosphere = false;
    for (uint32_t i = 0; i < scene_rep.data().emitter_profiles.size(); ++i) {
      auto& candidate = scene_rep.data().emitter_profiles[i];
      if (candidate.cls == EmitterProfile::Class::Environment && (candidate.meta & EmitterProfile::Meta::Atmosphere) != 0) {
        has_atmosphere = true;
        break;
      }
    }

    if (has_atmosphere) {
      bool is_sun = (emitter.reference_emitter_index != kInvalidIndex) && (emitter.reference_emitter_index < scene_rep.data().emitter_profiles.size()) &&
                    (scene_rep.data().emitter_profiles[emitter.reference_emitter_index].cls == EmitterProfile::Class::Environment) &&
                    ((scene_rep.data().emitter_profiles[emitter.reference_emitter_index].meta & EmitterProfile::Meta::Atmosphere) != 0);

      ImGui::Spacing();
      ImGui::Separator();
      ImGui::Spacing();

      if (ImGui::Checkbox("Use as Sun", &is_sun)) {
        if (is_sun) {
          uint32_t atmosphere_index = kInvalidIndex;
          for (uint32_t i = 0; i < scene_rep.data().emitter_profiles.size(); ++i) {
            auto& candidate = scene_rep.data().emitter_profiles[i];
            if (candidate.cls == EmitterProfile::Class::Environment && (candidate.meta & EmitterProfile::Meta::Atmosphere) != 0) {
              atmosphere_index = i;
              break;
            }
          }

          if (atmosphere_index != kInvalidIndex) {
            emitter.reference_emitter_index = atmosphere_index;
          } else {
            is_sun = false;
          }
        } else {
          emitter.reference_emitter_index = kInvalidIndex;
        }
        changed = true;
      }
    }
  }

  if ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0) {
    uint32_t sky_emitter_index = (emitter.cls == EmitterProfile::Class::Environment) ? emitter_index : emitter.reference_emitter_index;

    if (sky_emitter_index < scene_rep.data().emitter_profiles.size()) {
      auto& sky_emitter = scene_rep.data().emitter_profiles[sky_emitter_index];

      ImGui::Spacing();
      ImGui::Separator();
      ImGui::Spacing();

      if (ImGui::CollapsingHeader("Atmosphere Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::DragFloat("##altitude", &sky_emitter.atmosphere.scattering.altitude, 10.0f, 100.0f, 100000.0f, "Altitude: %.0f m")) {
          changed = true;
        }
        if (ImGui::SliderFloat("##anisotropy", &sky_emitter.atmosphere.scattering.anisotropy, -0.999f, 0.999f, "Anisotropy: %.3f", ImGuiSliderFlags_None)) {
          changed = true;
        }
        if (ImGui::DragFloat("##rayleigh", &sky_emitter.atmosphere.scattering.rayleigh_scale, 0.0f, 0.0f, 10.0f, "Rayleigh: %.3f")) {
          changed = true;
        }
        if (ImGui::DragFloat("##mie", &sky_emitter.atmosphere.scattering.mie_scale, 0.0f, 0.0f, 10.0f, "Mie: %.3f")) {
          changed = true;
        }
        if (ImGui::DragFloat("##ozone", &sky_emitter.atmosphere.scattering.ozone_scale, 0.0f, 0.0f, 10.0f, "Ozone: %.3f")) {
          changed = true;
        }
        bool primary_scattering = sky_emitter.atmosphere.scattering.primary_scattering != 0u;
        if (ImGui::Checkbox("Primary Scattering", &primary_scattering)) {
          sky_emitter.atmosphere.scattering.primary_scattering = primary_scattering ? 1u : 0u;
          changed = true;
        }
        bool secondary_scattering = sky_emitter.atmosphere.scattering.secondary_scattering != 0u;
        if (ImGui::Checkbox("Secondary Scattering", &secondary_scattering)) {
          sky_emitter.atmosphere.scattering.secondary_scattering = secondary_scattering ? 1u : 0u;
          changed = true;
        }
      }
    }
  }

  if (emitter.cls != EmitterProfile::Class::Area) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Button, kDeleteButtonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kDeleteButtonHoverColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, kDeleteButtonActiveColor);
    if (ImGui::Button("Delete Emitter", ImVec2(-1.0f, 0.0f))) {
      if ((callbacks.emitter_deleted) && callbacks.emitter_deleted(emitter_index)) {
        set_selection(SelectionKind::Rendering, 0, false);
      }
    }
    ImGui::PopStyleColor(3);
  }
  if (material_changed && callbacks.material_changed && (material_index < scene_rep.data().materials.size())) {
    callbacks.material_changed(material_index);
  }
  if (changed && callbacks.emitter_changed) {
    callbacks.emitter_changed(emitter_index);
  }
}

void UI::build_mesh_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx) {
  if (_mesh_mapping.empty()) {
    ImGui::Text("No meshes available");
    return;
  }
  if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= _mesh_mapping.size())) {
    ImGui::Text("Invalid mesh selection");
    return;
  }

  uint32_t mesh_index = _mesh_mapping.at(_selection.index);
  const Mesh& mesh = scene_rep.data().meshes[mesh_index];

  const char* mesh_name = _mesh_mapping.name(_selection.index);
  update_name_buffer(SelectionKind::Mesh, _selection.index, mesh_name);
  ImGui::AlignTextToFramePadding();
  ImGui::Text("Name");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  bool name_edit_active = ImGui::InputText("##mesh_name", _name_edit_buffer, sizeof(_name_edit_buffer), ImGuiInputTextFlags_AutoSelectAll);
  bool name_commit = ImGui::IsItemDeactivatedAfterEdit() || (name_edit_active && ImGui::IsKeyPressed(ImGuiKey_Enter));
  if (name_commit && callbacks.mesh_renamed) {
    callbacks.mesh_renamed(mesh_index, std::string(_name_edit_buffer));
    _pending_selection = {SelectionKind::Mesh, mesh_index, true};
  }

  ImGui::Text("Triangles: %u", mesh.triangle_count);

  uint32_t current_material = kInvalidIndex;
  if (mesh.triangle_count > 0) {
    uint32_t first_triangle_index = mesh.triangle_offset;
    if (first_triangle_index < scene_rep.data().triangles.size()) {
      current_material = scene_rep.data().triangles[first_triangle_index].material_index;
    }
  }
  std::vector<const char*> material_names;
  for (uint64_t i = 0; i < _material_mapping.size(); ++i) {
    const auto& entry = _material_mapping.entry(static_cast<int32_t>(i));
    material_names.push_back(entry.name);
  }

  int selected_material = -1;
  for (int i = 0; i < static_cast<int>(material_names.size()); ++i) {
    if (_material_mapping.at(i) == current_material) {
      selected_material = i;
      break;
    }
  }

  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  if (ImGui::Combo("##mesh_material", &selected_material, material_names.data(), static_cast<int>(material_names.size()))) {
    uint32_t new_material_index = _material_mapping.at(selected_material);
    if (callbacks.mesh_material_changed) {
      callbacks.mesh_material_changed(mesh_index, new_material_index);
    }
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Text("Edit Links");
  ImGui::Spacing();

  bool has_valid_material = (current_material != kInvalidIndex);
  if (has_valid_material == false) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("Edit Material", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
    for (int i = 0; i < static_cast<int>(_material_mapping.size()); ++i) {
      if (_material_mapping.at(i) == current_material) {
        set_selection(SelectionKind::Material, i);
        break;
      }
    }
  }
  if (has_valid_material == false) {
    ImGui::EndDisabled();
  }

  uint32_t emitter_profile_index = kInvalidIndex;
  if (mesh.triangle_count > 0) {
    uint32_t start_triangle = mesh.triangle_offset;
    uint32_t end_triangle = start_triangle + mesh.triangle_count;
    for (uint32_t tri_idx = start_triangle; tri_idx < end_triangle; ++tri_idx) {
      const Triangle& tri = scene_rep.data().triangles[tri_idx];
      if (tri.emitter_index != kInvalidIndex && tri.emitter_index < scene_rep.data().emitter_profiles.size()) {
        const auto& profile = scene_rep.data().emitter_profiles[tri.emitter_index];
        if (profile.cls == EmitterProfile::Class::Area) {
          emitter_profile_index = tri.emitter_index;
          break;
        }
      }
    }
  }

  bool has_emitter = (emitter_profile_index != kInvalidIndex);
  if (has_emitter == false) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button("Edit Emitter", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
    set_selection(SelectionKind::Emitter, static_cast<int32_t>(emitter_profile_index));
  }
  if (has_emitter == false) {
    ImGui::EndDisabled();
  }
}

void UI::build_camera_selection_properties(SceneRepresentation& scene_rep, Camera& camera, uint32_t camera_index, const BuildContext& ctx, const FrameData& data) {
  // bool film_changed = false;
  bool camera_changed = false;
  bool camera_is_active = false;
  if (camera_index < scene_rep.data().cameras.size()) {
    camera_is_active = scene_rep.data().cameras[camera_index].active;
  }

  uint2 viewport = camera.film_size;
  float3 pos = camera.position;
  float focal_len = get_camera_focal_length(camera);
  int32_t pixel_size = std::countr_zero(data.film.pixel_size());

  if (ImGui::CollapsingHeader("Lens & Focus", ImGuiTreeNodeFlags_Framed)) {
    static int control_mode = 0;  // 0 = Focal Length, 1 = Field of View
    const char* control_modes[] = {"Focal Length", "Field of View"};

    if (labeled_control("Control Mode", [&]() {
          return ImGui::Combo("##control_mode", &control_mode, control_modes, IM_ARRAYSIZE(control_modes));
        })) {
    }

    if (control_mode == 0) {
      if (labeled_control("Focal Length", [&]() {
            return ImGui::DragFloat("##focal_length", &focal_len, 0.1f, 1.0f, 5000.0f, "%.1fmm");
          })) {
        camera_changed = true;
      }
    } else {
      float current_fov_deg = focal_length_to_fov(focal_len) * 180.0f / kPi;
      static float fov_input = current_fov_deg;  // Static to maintain value between frames
      fov_input = current_fov_deg;               // Sync with current camera FOV

      if (labeled_control("FOV (Horizontal)", [&]() {
            return ImGui::InputFloat("##fov_input", &fov_input, 0.1f, 1.0f, "%.1f°");
          })) {
        focal_len = fov_to_focal_length(fov_input * kPi / 180.0f);
        camera_changed = true;
      }
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 0));
      ImGui::AlignTextToFramePadding();
      ImGui::Text("Convert FOV");
      ImGui::SameLine();
      if (ImGui::Button("H -> V")) {
        float horizontal_fov_rad = fov_input * kPi / 180.0f;
        float vertical_fov_rad = horizontal_fov_to_vertical_fov(horizontal_fov_rad);
        fov_input = vertical_fov_rad * 180.0f / kPi;
        focal_len = fov_to_focal_length(horizontal_fov_rad);
        camera_changed = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("V -> H")) {
        float vertical_fov_rad = fov_input * kPi / 180.0f;
        float horizontal_fov_rad = vertical_fov_to_horizontal_fov(vertical_fov_rad);
        fov_input = horizontal_fov_rad * 180.0f / kPi;
        focal_len = fov_to_focal_length(horizontal_fov_rad);
        camera_changed = true;
      }
      ImGui::PopStyleVar();
    }

    if (labeled_control("Focus Distance", [&]() {
          return ImGui::DragFloat("##focus_distance", &camera.focal_distance, 0.1f, 0.0f, 65536.0f, "%.3f");
        })) {
      camera_changed = true;
    }

    if (labeled_control("Lens Radius", [&]() {
          return ImGui::DragFloat("##lens_radius", &camera.lens_radius, 0.01f, 0.0f, 2.0f, "%.3f");
        })) {
      camera_changed = true;
    }
  }

  if (ImGui::CollapsingHeader("Position & Orientation", ImGuiTreeNodeFlags_Framed)) {
    ImGui::Text("Camera Position:");
    full_width_item();
    if (ImGui::InputFloat3("##campos", &pos.x, "%.3f")) {
      pos.x = std::clamp(pos.x, -CameraController::kMaxCameraDistance, CameraController::kMaxCameraDistance);
      pos.y = std::clamp(pos.y, -CameraController::kMaxCameraDistance, CameraController::kMaxCameraDistance);
      pos.z = std::clamp(pos.z, -CameraController::kMaxCameraDistance, CameraController::kMaxCameraDistance);
      camera_changed = true;
    }

    auto spherical = to_spherical(camera.direction);
    float2 angles = {spherical.phi, spherical.theta};

    if (angle_editor("Camera Direction", angles, -180.0f, 180.0f, -89.99f, 89.99f, 89.99f)) {
      camera.direction = from_spherical(angles.x, angles.y);
      camera_changed = true;
    }

    float clip_values[2] = {camera.clip_near, camera.clip_far};
    if (labeled_control("Clip Planes (near/far)", [&]() {
          return ImGui::DragFloat2("##clipplanes", clip_values, 0.01f, 0.0f, 5000.0f, "%.3f");
        })) {
      camera.clip_near = max(0.0f, clip_values[0]);
      camera.clip_far = max(camera.clip_near + 0.001f, clip_values[1]);
      camera_changed = true;
    }
  }

  float pixel_filter_radius = scene_rep.data().pixel_filter.radius;
  if (ImGui::CollapsingHeader("Output", ImGuiTreeNodeFlags_Framed)) {
    ImGui::Text("Output Image Size:");
    full_width_item();
    if (ImGui::InputInt2("##outimgsize", reinterpret_cast<int32_t*>(&viewport.x))) {
      // film_changed = true;
      camera_changed = true;
    }

    if (labeled_control("Pixel Filter Radius", [&]() {
          return ImGui::DragFloat("##pixelfiler", &pixel_filter_radius, 0.05f, 0.0f, 32.0f, "%.3fpx");
        })) {
      camera_changed = true;
    }

    ImGui::Text("Pixel Size:");
    full_width_item();
    if (ImGui::Combo("##pixelsize", &pixel_size, "Default\0Scaled 2x\0Scaled 4x\0Scaled 8x\0Scaled 16x\0")) {
      camera_changed = true;
    }
  }

  if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_Framed)) {
    if (_medium_mapping.empty() == false) {
      if (labeled_control("External Medium", [&]() {
            return medium_dropdown("##camera_external_medium", camera.medium_index);
          })) {
        camera_changed = true;
      }
    } else {
      ImGui::TextDisabled("No mediums available. Add mediums in the scene to enable external medium selection.");
    }
  }

  if (camera_changed) {
    viewport.x = clamp(viewport.x, 1u, 1024u * 16u);
    viewport.y = clamp(viewport.y, 1u, 1024u * 16u);
    camera.film_size = {uint32_t(viewport.x), uint32_t(viewport.y)};
    camera.lens_radius = fmaxf(camera.lens_radius, 0.0f);
    camera.focal_distance = fmaxf(camera.focal_distance, 0.0f);
    camera.clip_near = max(camera.clip_near, 0.0f);
    camera.clip_far = max(camera.clip_near + 0.001f, camera.clip_far);
    scene_rep.data().pixel_filter.radius = clamp(pixel_filter_radius, 0.0f, 32.0f);

    auto fov = focal_length_to_fov(focal_len) * 180.0f / kPi;
    build_camera(camera, pos, camera.direction, camera.up, camera.film_size, fov);

    if (camera_is_active) {
      scene_rep.data().cameras[camera_index].cam = camera;
      if (callbacks.camera_changed) {
        callbacks.camera_changed(viewport, 1u << pixel_size);
      }
    } else if (callbacks.scene_settings_changed) {
      callbacks.scene_settings_changed();
    }
  }
}

void UI::build_scene_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  bool scene_settings_changed = false;

  if (validated_int_control("Samples Per Pixel", reinterpret_cast<int32_t&>(scene_rep.data().options.samples), 1, 1000000)) {
    scene_settings_changed = true;
  }

  int32_t min_path = static_cast<int32_t>(scene_rep.data().options.min_path_length);
  int32_t max_path = static_cast<int32_t>(scene_rep.data().options.max_path_length);

  bool min_changed = validated_int_control("Min Path Length", min_path, 0, 65536);
  bool max_changed = validated_int_control("Max Path Length", max_path, 0, 65536);

  if (min_changed || max_changed) {
    scene_rep.data().options.min_path_length = static_cast<uint32_t>(min(min_path, max_path));
    scene_rep.data().options.max_path_length = static_cast<uint32_t>(max(min_path, max_path));
    scene_settings_changed = true;
  }

  if (validated_int_control("Random Termination", reinterpret_cast<int32_t&>(scene_rep.data().options.random_path_termination), 0, 65536)) {
    scene_settings_changed = true;
  }

  if (labeled_control("Radiance Clamp", [&]() {
        return ImGui::InputFloat("##radiance_clamp", &scene_rep.data().options.radiance_clamp, 0.1f, 1.f, "%0.2f");
      })) {
    scene_rep.data().options.radiance_clamp = max(scene_rep.data().options.radiance_clamp, 0.0f);
    scene_settings_changed = true;
  }

  const bool spectral_changed = ImGui::Checkbox("Spectral rendering", scene_rep.data().options.properties + Scene::Properties::Spectral);
  scene_settings_changed = (scene_settings_changed || spectral_changed);
  const bool blue_noise_changed = ImGui::Checkbox("Blue Noise", scene_rep.data().options.properties + Scene::Properties::BlueNoise);
  scene_settings_changed = (scene_settings_changed || blue_noise_changed);

  ImGui::Separator();
  if (labeled_control("Noise Threshold (experimental)", [&]() {
        return ImGui::InputFloat("##noise_thresh", &scene_rep.data().options.noise_threshold, 0.0001f, 0.01f, "%0.5f");
      })) {
    scene_rep.data().options.noise_threshold = std::clamp(scene_rep.data().options.noise_threshold, 0.0f, 1.0f);
    scene_settings_changed = true;
  }
  if (scene_rep.data().options.noise_threshold > 0.0f) {
    ImGui::Text("Active pixels: %.2f%%", double(data.film.active_pixel_count()) / double(data.film.current_pixel_count()) * 100.0);
  }

  if (scene_settings_changed) {
    scene_rep.data().options.max_path_length = min(scene_rep.data().options.max_path_length, 65536u);
    if (callbacks.scene_settings_changed) {
      callbacks.scene_settings_changed();
    }
  }
}

void UI::build_integrator_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx) {
  if (ctx.has_integrator == false) {
    ImGui::Text("No integrator available");
    return;
  }

  ImGui::Text("Integrator Type:");
  full_width_item();

  const bool gpu_renderer_mode = _current_renderer_mode == RendererMode::GPURaytracing;
  const auto gpu_integrator_supported = [](Integrator::Type type) {
    return (type == Integrator::Type::PathTracing) || (type == Integrator::Type::Bidirectional);
  };

  if (ImGui::BeginCombo("##integrator_type", _current_integrator->name())) {
    for (uint64_t i = 0; i < _integrators.count; ++i) {
      const bool is_selected = (_integrators[i] == _current_integrator);
      const bool supported_by_gpu = gpu_integrator_supported(_integrators[i]->type());
      const bool selectable_enabled = (gpu_renderer_mode == false) || supported_by_gpu;
      if (selectable_enabled == false) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Selectable(_integrators[i]->name(), is_selected) && selectable_enabled) {
        if (callbacks.integrator_selected) {
          callbacks.integrator_selected(_integrators[i]->type());
          set_current_integrator(_integrators[i]);
        }
      }
      if (selectable_enabled == false) {
        ImGui::EndDisabled();
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  if (gpu_renderer_mode && (gpu_integrator_supported(_current_integrator->type()) == false)) {
    ImGui::TextColored(kErrorTextColor, "Selected integrator is not supported by GPU raytracing");
  }

  bool options_changed = false;

  if (_current_integrator->options().options.empty() == false) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    options_changed = build_options(_current_integrator->options());
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  if (ImGui::CollapsingHeader("Rendering Strategies", ImGuiTreeNodeFlags_None)) {
    const uint32_t supported = _current_integrator->supported_strategies();
    bool strategies_changed = false;

    auto draw_strategy_checkbox = [&](const char* label, uint32_t flag) {
      const bool supported_flag = (supported & flag) == flag;
      const bool scene_value = (scene_rep.data().options.strategy_flags & flag) != 0u;
      bool enabled = supported_flag ? scene_value : false;

      if (supported_flag == false) {
        ImGui::BeginDisabled();
      }
      const bool changed = ImGui::Checkbox(label, &enabled);
      if (supported_flag == false) {
        ImGui::EndDisabled();
      }
      if (changed && supported_flag) {
        strategies_changed = true;
        scene_rep.data().options.strategy_flags = (scene_rep.data().options.strategy_flags & (~flag)) | (enabled ? flag : 0u);
      }
    };

    draw_strategy_checkbox("Direct Illumination", Scene::Strategy::DirectHit);
    draw_strategy_checkbox("Light Sampling", Scene::Strategy::ConnectToLight);
    draw_strategy_checkbox("Camera Connections", Scene::Strategy::ConnectToCamera);
    draw_strategy_checkbox("Bidirectional Connections", Scene::Strategy::ConnectVertices);
    draw_strategy_checkbox("Photon Merging", Scene::Strategy::MergeVertices);

    if (strategies_changed && callbacks.scene_settings_changed) {
      callbacks.scene_settings_changed();
    }
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  const bool mis_changed = ImGui::Checkbox("Multiple Importance Sampling", scene_rep.data().options.properties + Scene::Properties::MultipleImportanceSampling);
  if (mis_changed && callbacks.scene_settings_changed) {
    callbacks.scene_settings_changed();
  }

  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  int current_light_sampling = static_cast<int>(scene_rep.data().options.light_sampling);
  const char* light_sampling_options[] = {"Uniform", "From Distribution", "RIS Uniform", "RIS From Distribution"};
  const bool light_sampling_changed = ImGui::Combo("##light_sampling", &current_light_sampling, light_sampling_options, IM_ARRAYSIZE(light_sampling_options));
  if (light_sampling_changed) {
    scene_rep.data().options.light_sampling = static_cast<Scene::LightSampling>(current_light_sampling);
    if (callbacks.scene_settings_changed) {
      callbacks.scene_settings_changed();
    }
  }

  if (options_changed && callbacks.options_changed) {
    callbacks.options_changed();
  }
}

void UI::build_rendering_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  ImGui::Text("Renderer Mode:");
  full_width_item();
  const char* renderer_modes[] = {"CPU Raytracing", "Rasterization", _gpu_renderer_available ? "GPU Raytracing" : "GPU Raytracing (Unavailable)"};
  int current_mode = static_cast<int>(_current_renderer_mode);
  if (ImGui::Combo("##renderer_mode", &current_mode, renderer_modes, IM_ARRAYSIZE(renderer_modes))) {
    RendererMode selected_mode = static_cast<RendererMode>(current_mode);
    if ((selected_mode == RendererMode::GPURaytracing) && (_gpu_renderer_available == false)) {
      selected_mode = RendererMode::CPURaytracing;
    }
    _current_renderer_mode = selected_mode;
    if (callbacks.renderer_selected) {
      callbacks.renderer_selected(_current_renderer_mode);
    }
  }

  if (_current_renderer_mode == RendererMode::GPURaytracing) {
    int32_t wavefront_steps = static_cast<int32_t>(_gpu_wavefront_steps_per_frame);
    if (validated_int_control("GPU Wavefront Steps / Frame", wavefront_steps, 1, 1024)) {
      _gpu_wavefront_steps_per_frame = static_cast<uint32_t>(wavefront_steps);
      if (callbacks.gpu_wavefront_steps_per_frame_changed) {
        callbacks.gpu_wavefront_steps_per_frame_changed(_gpu_wavefront_steps_per_frame);
      }
    }
  }

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  build_scene_selection_properties(scene_rep, ctx, data);

  if ((_current_renderer_mode == RendererMode::CPURaytracing) || (_current_renderer_mode == RendererMode::GPURaytracing)) {
    build_integrator_selection_properties(scene_rep, ctx);
  }
}

}  // namespace etx
