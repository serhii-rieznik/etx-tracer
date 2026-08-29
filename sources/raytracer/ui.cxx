#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/core/platform.hxx>

#include <etx/render/host/film.hxx>
#include <etx/render/host/openpbr_material_loader.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/host/scene_serialization.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/render/shared/math.hxx>
#include <etx/render/shared/scene_medium.hxx>

#include "ui.hxx"
#include "platform_ui.hxx"
#include <imgui.h>
#include <imgui_internal.h>
#include <ImGuizmo.h>

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

const ImVec4 kIorPickerColors[] = {
  ImVec4(0.3333f, 0.3333f, 0.3333f, 1.0f),
  ImVec4(1.0f, 1.0f, 1.0f, 1.0f),
  ImVec4(0.5f, 0.75f, 1.0f, 1.0f),
  ImVec4(1.0f, 0.75f, 0.5f, 1.0f),
  ImVec4(1.0f, 0.75f, 1.0f, 1.0f),
};

struct SemanticButtonColors {
  ImVec4 normal = {};
  ImVec4 hovered = {};
  ImVec4 active = {};
};

const SemanticButtonColors kDarkLaunchButtonColors = {
  .normal = {0.11f, 0.44f, 0.11f, 1.0f},
  .hovered = {0.16f, 0.52f, 0.16f, 1.0f},
  .active = {0.09f, 0.36f, 0.09f, 1.0f},
};
const SemanticButtonColors kDarkTerminateButtonColors = {
  .normal = {0.44f, 0.11f, 0.11f, 1.0f},
  .hovered = {0.52f, 0.16f, 0.16f, 1.0f},
  .active = {0.36f, 0.09f, 0.09f, 1.0f},
};
const SemanticButtonColors kLightLaunchButtonColors = {
  .normal = {0.72f, 0.89f, 0.76f, 1.0f},
  .hovered = {0.64f, 0.85f, 0.69f, 1.0f},
  .active = {0.56f, 0.80f, 0.62f, 1.0f},
};
const SemanticButtonColors kLightTerminateButtonColors = {
  .normal = {0.95f, 0.75f, 0.75f, 1.0f},
  .hovered = {0.92f, 0.67f, 0.67f, 1.0f},
  .active = {0.89f, 0.58f, 0.58f, 1.0f},
};

const SemanticButtonColors& launch_button_colors(RHIImGuiTheme theme) {
  return theme == RHIImGuiTheme::Light ? kLightLaunchButtonColors : kDarkLaunchButtonColors;
}

const SemanticButtonColors& terminate_button_colors(RHIImGuiTheme theme) {
  return theme == RHIImGuiTheme::Light ? kLightTerminateButtonColors : kDarkTerminateButtonColors;
}

void push_semantic_button_colors(const SemanticButtonColors& colors) {
  ImGui::PushStyleColor(ImGuiCol_Button, colors.normal);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors.hovered);
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, colors.active);
}

struct ViewportZoomOption {
  const char* label = nullptr;
  float scale = 0.0f;
};

constexpr ViewportZoomOption kViewportZoomOptions[] = {
  {"Fit", 0.0f},
  {"25%", 0.25f},
  {"50%", 0.50f},
  {"75%", 0.75f},
  {"100%", 1.00f},
  {"150%", 1.50f},
  {"200%", 2.00f},
  {"400%", 4.00f},
};

const ImVec4 kErrorTextColor(1.0f, 0.0f, 0.0f, 1.0f);

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

const char* renderer_mode_status_name(RendererMode mode) {
  switch (mode) {
    case RendererMode::Rasterization:
      return "Rasterization";
    case RendererMode::GPURaytracing:
      return "GPU Raytracing";
    default:
      return "CPU Raytracing";
  }
}

const char* renderer_status_state_name(RendererStatusState state) {
  switch (state) {
    case RendererStatusState::Idle:
      return "Idle";
    case RendererStatusState::Preparing:
      return "Preparing";
    case RendererStatusState::Running:
      return "Running";
    case RendererStatusState::Finishing:
      return "Finishing";
    case RendererStatusState::Completed:
      return "Completed";
    case RendererStatusState::Failed:
      return "Failed";
    default:
      return "Unavailable";
  }
}

const char* renderer_progress_unit_name(RendererProgressKind kind) {
  switch (kind) {
    case RendererProgressKind::Samples:
      return "spp";
    case RendererProgressKind::Steps:
      return "steps";
    default:
      return "";
  }
}

const char* renderer_path_phase_name(RendererPathPhase phase) {
  switch (phase) {
    case RendererPathPhase::Light:
      return "Light paths";
    case RendererPathPhase::Camera:
      return "Camera paths";
    default:
      return "Paths";
  }
}

const char* renderer_path_phase_short_name(RendererPathPhase phase) {
  return phase == RendererPathPhase::Light ? "Light" : "Camera";
}

std::string compact_duration_string(double seconds) {
  if (seconds < 0.0) {
    return "-";
  }
  char buffer[32] = {};
  if (seconds < 1.0) {
    snprintf(buffer, sizeof(buffer), "%.0f ms", seconds * 1000.0);
    return buffer;
  }
  if (seconds < 10.0) {
    snprintf(buffer, sizeof(buffer), "%.1f s", seconds);
    return buffer;
  }
  return duration_string(seconds);
}

void draw_workspace_splitter(ImGuiMouseCursor cursor) {
  const bool active = ImGui::IsItemActive();
  const bool hovered = ImGui::IsItemHovered();
  const ImVec2 minimum = ImGui::GetItemRectMin();
  const ImVec2 maximum = ImGui::GetItemRectMax();
  const ImU32 color = ImGui::GetColorU32(active ? ImGuiCol_SeparatorActive : (hovered ? ImGuiCol_SeparatorHovered : ImGuiCol_Separator));
  const float thickness = active ? 2.0f : 1.0f;
  if (cursor == ImGuiMouseCursor_ResizeEW) {
    const float x = 0.5f * (minimum.x + maximum.x);
    ImGui::GetWindowDrawList()->AddLine(ImVec2(x, minimum.y), ImVec2(x, maximum.y), color, thickness);
  } else {
    const float y = 0.5f * (minimum.y + maximum.y);
    ImGui::GetWindowDrawList()->AddLine(ImVec2(minimum.x, y), ImVec2(maximum.x, y), color, thickness);
  }
  if (hovered || active) {
    ImGui::SetMouseCursor(cursor);
  }
}

std::string memory_size_string(uint64_t bytes) {
  constexpr double kKiB = 1024.0;
  constexpr double kMiB = 1024.0 * kKiB;
  constexpr double kGiB = 1024.0 * kMiB;
  char buffer[64] = {};
  if (bytes >= static_cast<uint64_t>(kGiB)) {
    snprintf(buffer, sizeof(buffer), "%.2f GiB", static_cast<double>(bytes) / kGiB);
  } else if (bytes >= static_cast<uint64_t>(kMiB)) {
    snprintf(buffer, sizeof(buffer), "%.2f MiB", static_cast<double>(bytes) / kMiB);
  } else if (bytes >= static_cast<uint64_t>(kKiB)) {
    snprintf(buffer, sizeof(buffer), "%.2f KiB", static_cast<double>(bytes) / kKiB);
  } else {
    snprintf(buffer, sizeof(buffer), "%llu B", static_cast<unsigned long long>(bytes));
  }
  return std::string(buffer);
}

bool contains_case_insensitive(const char* text, const char* filter) {
  if ((filter == nullptr) || (filter[0] == '\0')) {
    return true;
  }
  if (text == nullptr) {
    return false;
  }

  const std::string text_value(text);
  const std::string filter_value(filter);
  return std::search(text_value.begin(), text_value.end(), filter_value.begin(), filter_value.end(), [](char a, char b) {
    return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
  }) != text_value.end();
}

const char* renderer_memory_location_name(RendererMemoryLocation location) {
  switch (location) {
    case RendererMemoryLocation::CPU:
      return "CPU";
    case RendererMemoryLocation::GPUDevice:
      return "GPU device";
    case RendererMemoryLocation::GPUHostVisible:
      return "GPU host-visible";
  }
  return "Unknown";
}

struct MemorySummaryRow {
  std::string label;
  uint64_t bytes = 0u;
};

void draw_item_tooltip(const char* text, ImGuiHoveredFlags flags) {
  if ((text == nullptr) || (text[0] == '\0') || (ImGui::IsItemHovered(flags) == false)) {
    return;
  }

  ImGui::BeginTooltip();
  ImGui::PushTextWrapPos(ImGui::GetFontSize() * 34.0f);
  ImGui::TextUnformatted(text);
  ImGui::PopTextWrapPos();
  ImGui::EndTooltip();
}

void draw_memory_summary_table(const char* id, std::vector<MemorySummaryRow> rows) {
  constexpr ImGuiTableFlags flags =
    ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Sortable | ImGuiTableFlags_Resizable;
  if (ImGui::BeginTable(id, 2, flags) == false) {
    return;
  }

  ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthStretch, 0.0f, 0u);
  ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending, 130.0f, 1u);
  ImGui::TableHeadersRow();

  const ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs();
  if ((sort_specs != nullptr) && (sort_specs->SpecsCount > 0)) {
    const ImGuiTableColumnSortSpecs& spec = sort_specs->Specs[0];
    std::stable_sort(rows.begin(), rows.end(), [&spec](const MemorySummaryRow& lhs, const MemorySummaryRow& rhs) {
      int comparison = 0;
      if (spec.ColumnUserID == 0u) {
        comparison = lhs.label.compare(rhs.label);
      } else if (lhs.bytes != rhs.bytes) {
        comparison = (lhs.bytes < rhs.bytes) ? -1 : 1;
      }
      return (spec.SortDirection == ImGuiSortDirection_Ascending) ? (comparison < 0) : (comparison > 0);
    });
  }

  for (const MemorySummaryRow& row : rows) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(row.label.c_str());
    draw_item_tooltip(row.label.c_str(), ImGuiHoveredFlags_DelayNormal);
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(memory_size_string(row.bytes).c_str());
  }
  ImGui::EndTable();
}

struct MemoryAllocationRow {
  std::string category;
  std::string allocation;
  std::string location;
  uint32_t count = 0u;
  uint64_t bytes = 0u;
  bool count_available = true;
};

void sort_memory_allocation_rows(std::vector<MemoryAllocationRow>& rows, const ImGuiTableSortSpecs* sort_specs) {
  if ((sort_specs == nullptr) || (sort_specs->SpecsCount == 0)) {
    return;
  }

  const ImGuiTableColumnSortSpecs& spec = sort_specs->Specs[0];
  std::stable_sort(rows.begin(), rows.end(), [&spec](const MemoryAllocationRow& lhs, const MemoryAllocationRow& rhs) {
    int comparison = 0;
    switch (spec.ColumnUserID) {
      case 0u:
        comparison = lhs.category.compare(rhs.category);
        break;
      case 1u:
        comparison = lhs.allocation.compare(rhs.allocation);
        break;
      case 2u:
        comparison = lhs.location.compare(rhs.location);
        break;
      case 3u:
        if (lhs.count != rhs.count) {
          comparison = (lhs.count < rhs.count) ? -1 : 1;
        }
        break;
      case 4u:
        if (lhs.bytes != rhs.bytes) {
          comparison = (lhs.bytes < rhs.bytes) ? -1 : 1;
        }
        break;
    }
    return (spec.SortDirection == ImGuiSortDirection_Ascending) ? (comparison < 0) : (comparison > 0);
  });
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

bool UI::gpu_integrator_supported(Integrator::Type type) {
  return (type == Integrator::Type::PathTracing) || (type == Integrator::Type::Bidirectional) || (type == Integrator::Type::VCM) || (type == Integrator::Type::UPBP);
}

std::string UI::render_configuration_label(RendererMode renderer, Integrator* integrator) {
  if (renderer == RendererMode::Rasterization) {
    return "Raster Preview";
  }
  if (integrator == nullptr) {
    return "No Integrator";
  }
  if (renderer == RendererMode::CPURaytracing) {
    return integrator->name();
  }

  switch (integrator->type()) {
    case Integrator::Type::PathTracing:
      return "Path Tracing (GPU)";
    case Integrator::Type::Bidirectional:
      return "Bidirectional (GPU)";
    case Integrator::Type::VCM:
      return "VCM (GPU)";
    case Integrator::Type::UPBP:
      return "UPBP (GPU)";
    default:
      return std::string(integrator->name()) + " (GPU)";
  }
}

uint32_t UI::render_configuration_argument(RendererMode renderer, uint32_t integrator_index) {
  constexpr uint32_t index_mask = 0x00ffffffu;
  const uint32_t encoded_index = (integrator_index == kInvalidIndex) ? 0u : (integrator_index + 1u);
  return (static_cast<uint32_t>(renderer) << 24u) | (encoded_index & index_mask);
}

bool UI::render_configuration_selected(uint32_t argument) const {
  constexpr uint32_t index_mask = 0x00ffffffu;
  const RendererMode renderer = static_cast<RendererMode>(argument >> 24u);
  const uint32_t encoded_index = argument & index_mask;
  if (renderer == RendererMode::Rasterization) {
    return _current_renderer_mode == RendererMode::Rasterization;
  }
  if ((encoded_index == 0u) || ((encoded_index - 1u) >= _integrators.count)) {
    return false;
  }
  return (_current_renderer_mode == renderer) && (_current_integrator == _integrators[encoded_index - 1u]);
}

void UI::full_width_item() {
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
}

bool UI::labeled_control(const char* label, std::function<bool()>&& control_func) {
  ImGui::TextUnformatted(label);
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
    case SelectionKind::Node:
      if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= scene_rep.data().hierarchy.nodes.size())) {
        set_selection(SelectionKind::None, -1, false);
      }
      break;
    case SelectionKind::Material:
      if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= _material_mapping.size())) {
        set_selection(SelectionKind::None, -1, false);
      } else if (_selected_material_positions.empty()) {
        _selected_material_positions.push_back(_selection.index);
        _material_selection_anchor = _selection.index;
      } else if (material_list_position_selected(_selection.index) == false) {
        _selection.index = _selected_material_positions.back();
      }
      break;
    case SelectionKind::Medium:
      if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= _medium_mapping.size())) {
        set_selection(SelectionKind::None, -1, false);
      }
      break;
    case SelectionKind::Emitter:
      if ((_selection.index < 0) || (static_cast<uint32_t>(_selection.index) >= scene_rep.data().emitter_profiles.size())) {
        set_selection(SelectionKind::None, -1, false);
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

void UI::queue_material_change(uint32_t material_index) {
  if (material_index == kInvalidIndex) {
    return;
  }

  if (_medium_interaction_active) {
    finish_medium_interaction();
  }

  if (_material_interaction_active == false) {
    _material_interaction_active = true;
    if (callbacks.material_interaction_started) {
      callbacks.material_interaction_started();
    }
  }

  if (std::find(_material_interaction_indices.begin(), _material_interaction_indices.end(), material_index) == _material_interaction_indices.end()) {
    _material_interaction_indices.push_back(material_index);
  }
}

void UI::finish_material_interaction() {
  if (_material_interaction_active == false) {
    return;
  }

  _material_interaction_active = false;
  if (callbacks.material_interaction_finished) {
    callbacks.material_interaction_finished(_material_interaction_indices);
  } else if (callbacks.material_changed) {
    for (const uint32_t material_index : _material_interaction_indices) {
      callbacks.material_changed(material_index);
    }
  }
  _material_interaction_indices.clear();
}

void UI::queue_medium_change(uint32_t medium_index) {
  if (medium_index == kInvalidIndex) {
    return;
  }

  if (_material_interaction_active) {
    finish_material_interaction();
  }

  if (_medium_interaction_active == false) {
    _medium_interaction_active = true;
    if (callbacks.medium_interaction_started) {
      callbacks.medium_interaction_started();
    }
  }

  if (std::find(_medium_interaction_indices.begin(), _medium_interaction_indices.end(), medium_index) == _medium_interaction_indices.end()) {
    _medium_interaction_indices.push_back(medium_index);
  }
}

void UI::finish_medium_interaction() {
  if (_medium_interaction_active == false) {
    return;
  }

  _medium_interaction_active = false;
  if (callbacks.medium_interaction_finished) {
    callbacks.medium_interaction_finished(_medium_interaction_indices);
  } else if (callbacks.medium_changed) {
    for (const uint32_t medium_index : _medium_interaction_indices) {
      callbacks.medium_changed(medium_index);
    }
  }
  _medium_interaction_indices.clear();
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
        if (option.description.empty() == false) {
          ImGui::TextUnformatted(option.description.c_str());
        }
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("%s", data.value.c_str());
        ImGui::PopStyleColor();
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
        if (option.description.empty() == false) {
          ImGui::TextUnformatted(option.description.c_str());
        }
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
        if (data.bounds.maximum > data.bounds.minimum) {
          changed = ImGui::DragFloat(label.c_str(), &data.value, 0.1f, data.bounds.minimum, data.bounds.maximum, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        } else {
          changed = ImGui::InputFloat(label.c_str(), &data.value, 0.0f, 0.0f, "%.3f");
        }
        break;
      }
      case Option::Class::Integral: {
        auto& data = option.as<Option::Class::Integral>();
        if (option.meta & Option::Meta::EnumValue) {
          ETX_CRITICAL(data.bounds.maximum > data.bounds.minimum);
          ETX_CRITICAL(option.name_getter);
          int32_t original = data.value;
          const std::string& id_source = option.id.empty() ? option.description : option.id;
          const bool value_valid = (data.value >= data.bounds.minimum) && (data.value <= data.bounds.maximum);
          const std::string preview = value_valid ? option.name_getter(data.value) : std::string("Invalid");
          if (option.description.empty() == false) {
            ImGui::TextUnformatted(option.description.c_str());
          }
          full_width_item();
          if (ImGui::BeginCombo(("##" + id_source).c_str(), preview.c_str())) {
            for (int32_t value = data.bounds.minimum; value <= data.bounds.maximum; ++value) {
              const bool selected = data.value == value;
              if (ImGui::Selectable(option.name_getter(value).c_str(), selected)) {
                data.value = value;
              }
              if (selected) {
                ImGui::SetItemDefaultFocus();
              }
            }
            ImGui::EndCombo();
          }
          changed = data.value != original;
        } else {
          const std::string& id_source = option.id.empty() ? option.description : option.id;
          std::string label = "##" + id_source;
          if (option.description.empty() == false) {
            ImGui::TextUnformatted(option.description.c_str());
          }
          ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
          if (data.bounds.maximum > data.bounds.minimum) {
            changed = ImGui::DragInt(label.c_str(), &data.value, 0.1f, data.bounds.minimum, data.bounds.maximum, "%d", ImGuiSliderFlags_AlwaysClamp);
          } else {
            changed = ImGui::InputInt(label.c_str(), &data.value, 0, 0);
          }
        }
        break;
      }
      case Option::Class::Float3: {
        auto& data = option.as<Option::Class::Float3>();
        ImGui::TextUnformatted(option.description.c_str());
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
  ImGui::TextUnformatted("Elevation");
  full_width_item();
  if (ImGui::SliderFloat("##elevation", &elevation_deg, clamped_min_elevation, clamped_max_elevation, "%.1f°")) {
    angles.y = std::clamp(elevation_deg * kPi / 180.0f, clamped_min_elevation * kPi / 180.0f, clamped_max_elevation * kPi / 180.0f);
    if (!std::isfinite(angles.y))
      angles.y = 0.0f;
    changed = true;
  }

  bool near_pole = (std::abs(elevation_deg) >= pole_threshold);

  ImGui::TextUnformatted("Azimuth");
  full_width_item();
  if (near_pole) {
    ImGui::BeginDisabled();
  }
  if (ImGui::SliderFloat("##azimuth", &azimuth_deg, min_azimuth, max_azimuth, "%.1f°")) {
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

  ImGui::TextUnformatted(base_label);
  SpectrumEditorState::Mode previous_mode = editor_state.mode;
  full_width_item();
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
    full_width_item();
    if (ImGui::InputFloat(temperature_label, &temperature, 100.0f, 1000.0f, "%.0f K")) {
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
  full_width_item();
  const std::string channel_label = std::string("##channel_") + label;
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
    ImGui::TextUnformatted("Scale");
    full_width_item();
    float drag_speed = max(0.01f, max(editor_state.scale, 1.0f) * 0.01f);
    float scale_value = editor_state.scale;
    scale_changed = ImGui::DragFloat(scale_label, &scale_value, drag_speed, show_color ? 1.0f : 0.01f, 1000.0f, "%.2f", ImGuiSliderFlags_NoRoundToFormat);
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

void UI::build(SceneRepresentation& scene_rep, const FrameData& data) {
  ETX_PROFILER_SCOPE();
  ImGuizmo::BeginFrame();
  _node_transform_editor_interaction_rendered_this_frame = false;
  _material_editor_rendered_this_frame = false;
  _medium_editor_rendered_this_frame = false;

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

  ctx.with_window = [](uint32_t flag, const char* title, std::function<void()>&& body) {
    (void)flag;
    (void)title;
    body();
  };

  uint64_t mmh = hash_mapping(scene_rep.material_mapping());
  if (mmh != _material_mapping_hash) {
    if (_selection.kind == SelectionKind::Material) {
      set_selection(SelectionKind::None, -1, false);
    }
    clear_selection_history();
    _material_mapping.build(scene_rep.material_mapping());
    _material_mapping_hash = mmh;
  }

  uint64_t medh = hash_mapping(scene_rep.medium_mapping());
  if (medh != _medium_mapping_hash) {
    if (_selection.kind == SelectionKind::Medium) {
      set_selection(SelectionKind::None, -1, false);
    }
    clear_selection_history();
    _medium_mapping.build(scene_rep.medium_mapping());
    _medium_mapping_hash = medh;
  }

  uint64_t meshh = hash_mapping(scene_rep.mesh_mapping());
  if (meshh != _mesh_mapping_hash) {
    _mesh_mapping.build(scene_rep.mesh_mapping());
    _mesh_mapping_hash = meshh;
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
  _pending_selection = {};

  validate_selections(scene_rep);
  if (_node_transform_editor_interaction_active && ((_selection.kind != SelectionKind::Node) || (_selection.index != _node_transform_editor_interaction_node_index))) {
    finish_node_transform_editor_interaction();
  }

  if (_embedded_menu_enabled) {
    build_main_menu_bar(data.recent_files);
  }
  if (_embedded_toolbar_enabled) {
    build_toolbar(ctx);
  }
  build_status_bar(ctx);
  build_workspace(scene_rep, ctx, data);
  build_transform_gizmo(scene_rep, data);
  if (_node_transform_editor_interaction_active && (_node_transform_editor_interaction_rendered_this_frame == false)) {
    finish_node_transform_editor_interaction();
  }
  if (_material_interaction_active && ((_material_editor_rendered_this_frame == false) || (ImGui::IsAnyItemActive() == false))) {
    finish_material_interaction();
  }
  if (_medium_interaction_active && ((_medium_editor_rendered_this_frame == false) || (ImGui::IsAnyItemActive() == false))) {
    finish_medium_interaction();
  }
  build_unsaved_changes_modal();
  build_renderer_preparation_modal();
}

void UI::build_unsaved_changes_modal() {
  constexpr const char* popup_id = "Unsaved scene changes##unsaved_changes";
  if (_unsaved_changes_modal_requested) {
    ImGui::OpenPopup(popup_id);
    _unsaved_changes_modal_requested = false;
    _unsaved_save_failed = false;
  }

  bool continue_action = false;
  bool discard_changes = false;
  if (ImGui::BeginPopupModal(popup_id, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted("The scene contains unsaved changes.");
    ImGui::TextDisabled("Save before continuing to keep your edits.");
    if (_unsaved_save_failed) {
      ImGui::TextColored(kErrorTextColor, "The scene could not be saved.");
    }
    ImGui::Spacing();
    if (ImGui::Button("Cancel", ImVec2(120.0f, 0.0f))) {
      _unsaved_save_failed = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    push_semantic_button_colors(terminate_button_colors(_theme));
    if (ImGui::Button("Don't Save", ImVec2(120.0f, 0.0f))) {
      continue_action = true;
      discard_changes = true;
      _unsaved_save_failed = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::PopStyleColor(3);
    ImGui::SameLine();
    if (ImGui::Button("Save", ImVec2(120.0f, 0.0f))) {
      if (save_scene_file()) {
        continue_action = true;
        _unsaved_save_failed = false;
        ImGui::CloseCurrentPopup();
      } else {
        _unsaved_save_failed = true;
      }
    }
    ImGui::EndPopup();
  }

  if (continue_action) {
    const MenuCommand command = _pending_menu_command;
    const std::string value = _pending_menu_value;
    _pending_menu_value.clear();
    _scene_dirty = false;
    if (discard_changes && callbacks.scene_discarded) {
      callbacks.scene_discarded();
    }
    if (command != MenuCommand::Quit) {
      _skip_unsaved_check_once = true;
    }
    execute_menu_command(command, 0u, value);
  }
}

bool UI::handle_event(const sapp_event* e) {
  const auto pointer_inside_viewport = [&]() {
    if (_viewport_geometry.valid == false) {
      return false;
    }
    const float scale_x = std::max(1.0e-6f, _viewport_geometry.framebuffer_scale.x);
    const float scale_y = std::max(1.0e-6f, _viewport_geometry.framebuffer_scale.y);
    const float pointer_x = e->mouse_x / scale_x;
    const float pointer_y = e->mouse_y / scale_y;
    const float image_left = std::max(_viewport_geometry.logical_position.x, _viewport_geometry.image_position.x);
    const float image_top = std::max(_viewport_geometry.logical_position.y, _viewport_geometry.image_position.y);
    const float image_right =
      std::min(_viewport_geometry.logical_position.x + _viewport_geometry.logical_size.x, _viewport_geometry.image_position.x + _viewport_geometry.image_size.x);
    const float image_bottom =
      std::min(_viewport_geometry.logical_position.y + _viewport_geometry.logical_size.y, _viewport_geometry.image_position.y + _viewport_geometry.image_size.y);
    return (pointer_x >= image_left) && (pointer_y >= image_top) && (pointer_x < image_right) && (pointer_y < image_bottom);
  };

  if (e->type == SAPP_EVENTTYPE_MOUSE_DOWN) {
    if (_gizmo_captures_mouse) {
      _viewport_pointer_active = false;
      return true;
    }
    _viewport_pointer_active = pointer_inside_viewport();
    return _viewport_pointer_active == false;
  }
  if (e->type == SAPP_EVENTTYPE_MOUSE_UP) {
    const bool route_to_viewport = _viewport_pointer_active;
    _viewport_pointer_active = false;
    return route_to_viewport == false;
  }
  if (e->type == SAPP_EVENTTYPE_MOUSE_MOVE) {
    if (_gizmo_captures_mouse) {
      return true;
    }
    return (_viewport_pointer_active || pointer_inside_viewport()) == false;
  }
  if (e->type == SAPP_EVENTTYPE_MOUSE_SCROLL) {
    return pointer_inside_viewport() == false;
  }
  if (e->type != SAPP_EVENTTYPE_KEY_DOWN) {
    return false;
  }

  auto modifiers = e->modifiers;
  bool has_shift = modifiers & SAPP_MODIFIER_SHIFT;

  bool handled = false;
  if ((modifiers & SAPP_MODIFIER_CTRL) || (modifiers & SAPP_MODIFIER_SUPER)) {
    switch (e->key_code) {
      case SAPP_KEYCODE_Q: {
        execute_menu_command(MenuCommand::Quit);
        handled = true;
        break;
      }
      case SAPP_KEYCODE_O: {
        execute_menu_command(MenuCommand::OpenScene);
        handled = true;
        break;
      }
      case SAPP_KEYCODE_I: {
        execute_menu_command(MenuCommand::OpenReferenceImage);
        handled = true;
        break;
      }
      case SAPP_KEYCODE_R: {
        if (has_shift) {
          execute_menu_command(MenuCommand::UseImageAsReference);
        } else {
          execute_menu_command(MenuCommand::ReloadScene);
        }
        handled = true;
        break;
      }
      case SAPP_KEYCODE_G: {
        execute_menu_command(MenuCommand::ReloadGeometry);
        handled = true;
        break;
      }
      case SAPP_KEYCODE_S: {
        if (has_shift) {
          execute_menu_command(MenuCommand::SaveSceneAs);
        } else {
          execute_menu_command(MenuCommand::SaveScene);
        }
        handled = true;
        break;
      }
      case SAPP_KEYCODE_E: {
        execute_menu_command(has_shift ? MenuCommand::SaveImageLDR : MenuCommand::SaveImageRGB);
        handled = true;
        break;
      }
      default:
        break;
    }
  }

  if (handled) {
    return true;
  }

  switch (e->key_code) {
    case SAPP_KEYCODE_F1:
    case SAPP_KEYCODE_F2:
    case SAPP_KEYCODE_F3: {
      uint32_t flag = 1u << (e->key_code - SAPP_KEYCODE_F1);
      _ui_setup = (_ui_setup & flag) ? (_ui_setup & (~flag)) : (_ui_setup | flag);
      return true;
    };

    case SAPP_KEYCODE_1:
    case SAPP_KEYCODE_2:
    case SAPP_KEYCODE_3:
    case SAPP_KEYCODE_4:
    case SAPP_KEYCODE_5: {
      _view_options.view_image = static_cast<uint32_t>(e->key_code - SAPP_KEYCODE_1);
      if (callbacks.output_view_changed)
        callbacks.output_view_changed(_view_options.view_image);
      return true;
    }
    case SAPP_KEYCODE_KP_DIVIDE: {
      execute_menu_command(MenuCommand::DecreaseExposure);
      return true;
    }
    case SAPP_KEYCODE_KP_MULTIPLY: {
      execute_menu_command(MenuCommand::IncreaseExposure);
      return true;
    }
    default:
      break;
  }

  return false;
}

void UI::execute_menu_command(MenuCommand command, uint32_t argument, const std::string& value) {
  const auto requires_unsaved_confirmation = [&](MenuCommand pending_command) {
    const bool destructive_command = (pending_command == MenuCommand::Quit) || (pending_command == MenuCommand::OpenScene) || (pending_command == MenuCommand::OpenRecentScene) ||
                                     (pending_command == MenuCommand::ReloadScene) || (pending_command == MenuCommand::ReloadGeometry);
    if ((destructive_command == false) || (_scene_dirty == false)) {
      return false;
    }
    if (_skip_unsaved_check_once) {
      _skip_unsaved_check_once = false;
      return false;
    }
    _pending_menu_command = pending_command;
    _pending_menu_value = value;
    _unsaved_changes_modal_requested = true;
    return true;
  };

  if (requires_unsaved_confirmation(command)) {
    return;
  }

  switch (command) {
    case MenuCommand::Quit:
      quit();
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
    case MenuCommand::SelectIntegrator: {
      constexpr uint32_t index_mask = 0x00ffffffu;
      const RendererMode renderer = static_cast<RendererMode>(argument >> 24u);
      const uint32_t encoded_index = argument & index_mask;
      if (renderer == RendererMode::Rasterization) {
        select_render_configuration(renderer, nullptr);
      } else if ((encoded_index > 0u) && ((encoded_index - 1u) < _integrators.count)) {
        select_render_configuration(renderer, _integrators[encoded_index - 1u]);
      }
      break;
    }
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
    case MenuCommand::RunRenderer:
      if (_current_renderer_controls.can_run && callbacks.run_selected) {
        callbacks.run_selected();
      }
      break;
    case MenuCommand::FinishRenderer:
      if (_current_renderer_controls.can_finish && callbacks.stop_selected) {
        callbacks.stop_selected(true);
      }
      break;
    case MenuCommand::StopRenderer:
      if (_current_renderer_controls.can_stop && callbacks.stop_selected) {
        callbacks.stop_selected(false);
      }
      break;
    case MenuCommand::RestartRenderer:
      if (_current_renderer_controls.can_restart && callbacks.restart_selected) {
        callbacks.restart_selected();
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
      if (callbacks.exposure_changed)
        callbacks.exposure_changed(_view_options.exposure);
      break;
    case MenuCommand::DecreaseExposure:
      decrease_exposure(_view_options);
      if (callbacks.exposure_changed)
        callbacks.exposure_changed(_view_options.exposure);
      break;
    case MenuCommand::ToggleSceneObjects:
      _ui_setup ^= UIObjects;
      break;
    case MenuCommand::ToggleProperties:
      _ui_setup ^= UIProperties;
      break;
    case MenuCommand::ToggleMemoryDiagnostics:
      _ui_setup ^= UIMemoryDiagnostics;
      break;
    case MenuCommand::ResetLayout:
      _reset_layout_requested = true;
      break;
  }
}

void UI::set_current_integrator(Integrator* i) {
  _current_integrator = i;
}

void UI::quit() {
  if (callbacks.quit_selected) {
    callbacks.quit_selected();
  } else {
    sapp_quit();
  }
}

void UI::select_scene_file() const {
  auto selected_file = open_file("json,obj,gltf,glb");
  if ((selected_file.empty() == false) && callbacks.scene_file_selected) {
    callbacks.scene_file_selected(selected_file);
  }
}

bool UI::save_scene_file() const {
  if (callbacks.save_scene_file_selected) {
    return callbacks.save_scene_file_selected({});
  }
  return false;
}

bool UI::save_scene_file_as() const {
  const std::string selected_file = save_file("json");
  if ((selected_file.empty() == false) && callbacks.save_scene_file_selected) {
    return callbacks.save_scene_file_selected(selected_file);
  }
  return false;
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
  _material_editor_rendered_this_frame = true;
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
  auto with_section = [&](int color_index, const char* title, auto&& body, bool default_open) {
    if (default_open) {
      ImGui::SetNextItemOpen(true, ImGuiCond_Once);
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
            ImGui::TextUnformatted("Texture");
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
              ImGui::TextUnformatted("Texture");
              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
              return image_picker(scene_rep, "Scattering Texture##scattering_texture", material.scattering.image_index, Image::RepeatU | Image::RepeatV);
            });
        }
        ImGui::Spacing();
        const bool opacity_mixed = material_values_mixed([](const Material& value) {
          return value.opacity;
        });
        ImGui::TextUnformatted("Opacity");
        full_width_item();
        changed |= mixed_control(opacity_mixed, [&]() {
          return ImGui::SliderFloat("##opacity", &material.opacity, 0.0f, 1.0f, opacity_mixed ? "mixed" : "%.3f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
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

          const bool rough_u_mixed = material_values_mixed([](const Material& value) {
            return value.roughness.value.x;
          });
          ImGui::TextUnformatted(anisotropic ? "Roughness U" : "Roughness");
          full_width_item();
          const bool rough_u_changed = mixed_control(rough_u_mixed, [&]() {
            return ImGui::SliderFloat("##rough_u", &rough_u, 0.0f, 1.0f, rough_u_mixed ? "mixed" : "%.3f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
          });
          if (rough_u_changed) {
            if (anisotropic == false) {
              rough_v = rough_u;
              aniso_entry->second = false;
            }
            changed = true;
          }

          if (anisotropic) {
            const bool rough_v_mixed = material_values_mixed([](const Material& value) {
              return value.roughness.value.y;
            });
            ImGui::TextUnformatted("Roughness V");
            full_width_item();
            const bool rough_v_changed = mixed_control(rough_v_mixed, [&]() {
              return ImGui::SliderFloat("##rough_v", &rough_v, 0.0f, 1.0f, rough_v_mixed ? "mixed" : "%.3f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
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
              ImGui::TextUnformatted("Texture");
              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
              return sampled_image_picker(scene_rep, "Roughness Texture##roughness_texture", material.roughness, Image::RepeatU | Image::RepeatV | Image::SkipSRGBConversion);
            });
        }
        if (material.cls == MaterialClass::OpenPBR) {
          ImGui::Spacing();
          float metal = material.metalness.value.x;
          const bool metalness_mixed = material_values_mixed([](const Material& value) {
            return value.metalness.value.x;
          });
          ImGui::TextUnformatted("Metalness");
          full_width_item();
          const bool metalness_changed = mixed_control(metalness_mixed, [&]() {
            return ImGui::SliderFloat("##metalness", &metal, 0.0f, 1.0f, metalness_mixed ? "mixed" : "%.3f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
          });
          if (metalness_changed) {
            material.metalness.value = {metal, metal, metal, metal};
            changed = true;
          }
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return uint2{value.metalness.image_index, value.metalness.channel};
          }),
            [&]() {
              ImGui::TextUnformatted("Texture");
              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
              return sampled_image_picker(scene_rep, "Metalness Texture##metalness_texture", material.metalness, Image::RepeatU | Image::RepeatV | Image::SkipSRGBConversion);
            });

          float trans = material.transmission.value.x;
          const bool transmission_mixed = material_values_mixed([](const Material& value) {
            return value.transmission.value.x;
          });
          ImGui::TextUnformatted("Transmission");
          full_width_item();
          const bool transmission_changed = mixed_control(transmission_mixed, [&]() {
            return ImGui::SliderFloat("##transmission", &trans, 0.0f, 1.0f, transmission_mixed ? "mixed" : "%.3f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
          });
          if (transmission_changed) {
            material.transmission.value = {trans, trans, trans, trans};
            changed = true;
          }
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return uint2{value.transmission.image_index, value.transmission.channel};
          }),
            [&]() {
              ImGui::TextUnformatted("Texture");
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
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return value.normal_scale;
          }),
            [&]() {
              ImGui::TextUnformatted("Scale");
              full_width_item();
              return ImGui::SliderFloat("##normal_scale", &material.normal_scale, 0.0f, 4.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
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
      true);
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
            ImGui::TextUnformatted("Texture");
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
          ImGui::TextUnformatted("Subsurface");
          ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return value.subsurface_cls;
          }),
            [&]() {
              return ImGui::Combo("##sssclass", reinterpret_cast<int*>(&material.subsurface_cls), "Disabled\0Random Walk\0");
            });
          ImGui::TextUnformatted("Path");
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
              ImGui::TextUnformatted("Distance");
              return spectrum_picker(scene_rep, "Subsurface Distance", material.subsurface.spectrum_index, true, true);
            });
          changed |= mixed_control(material_values_mixed([](const Material& value) {
            return value.subsurface.image_index;
          }),
            [&]() {
              ImGui::TextUnformatted("Texture");
              ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
              return image_picker(scene_rep, "Subsurface Texture##subsurface_texture", material.subsurface.image_index, Image::RepeatU | Image::RepeatV);
            });

          ImGui::Spacing();
        }

        if (_medium_mapping.empty() == false) {
          const float avail = ImGui::GetContentRegionAvail().x;
          const float spacing = ImGui::GetStyle().ItemSpacing.x;
          float combo_width = (avail - spacing) * 0.5f;
          combo_width = max(combo_width, 0.0f);

          ImGui::Text("Inside / Outside");
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
      float collimation = material.emission_collimation;
      const bool collimation_mixed = material_values_mixed([](const Material& value) {
        return value.emission_collimation;
      });
      ImGui::TextUnformatted("Collimation");
      full_width_item();
      const bool collimation_changed = mixed_control(collimation_mixed, [&]() {
        return ImGui::SliderFloat("##material_emission_collimation", &collimation, 0.0f, 1.0f, collimation_mixed ? "mixed" : "%.2f", ImGuiSliderFlags_AlwaysClamp);
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

bool UI::build_medium(Medium& m, SpectralDistribution* absorption, SpectralDistribution* scattering) {
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
  if ((absorption != nullptr) && spectrum_picker("Absorption##medium_absorption", *absorption, true, true)) {
    changed = true;
  }

  ImGui::Text("Scattering");
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  if ((scattering != nullptr) && spectrum_picker("Scattering##medium_scattering", *scattering, true, true)) {
    changed = true;
  }

  ImGui::TextUnformatted("Anisotropy");
  full_width_item();
  if (ImGui::SliderFloat("##medium_phase_g", &m.phase_function_g, -0.999f, 0.999f, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
    changed = true;
  }

  bool explicit_connections = (m.enable_explicit_connections != 0u);
  if (ImGui::Checkbox("Explicit connections##medium_explicit_connections", &explicit_connections)) {
    m.enable_explicit_connections = explicit_connections ? 1u : 0u;
    changed = true;
  }

  if (m.cls == Medium::Heterogeneous) {
    if (m.grid_type_enum() == DensityGrid::Type::Texture3D) {
      ImGui::TextUnformatted("Density Texture");
      ImGui::Text("%u x %u x %u", m.grid.dimensions.x, m.grid.dimensions.y, m.grid.dimensions.z);
      ImGui::Text("Bounds");
      ImGui::Text("Min: %.3f %.3f %.3f", m.bounds.p_min.x, m.bounds.p_min.y, m.bounds.p_min.z);
      ImGui::Text("Max: %.3f %.3f %.3f", m.bounds.p_max.x, m.bounds.p_max.y, m.bounds.p_max.z);
    } else if (m.grid_type_enum() == DensityGrid::Type::NoiseFunction) {
      ImGui::TextUnformatted("Density Noise");
      ImGui::Text("Noise Type");
      const char* noise_names[] = {"Perlin", "Worley", "Billow", "Voronoi", "Lattice", "Uniform"};
      int32_t noise_idx = static_cast<int32_t>(m.noise_type_enum());
      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
      if (ImGui::Combo("##medium_noise_type", &noise_idx, noise_names, static_cast<int32_t>(NoiseFunction::Count))) {
        m.set_noise_type(static_cast<NoiseFunction>(noise_idx));
        changed = true;
      }
      ImGui::TextUnformatted("Scale");
      full_width_item();
      if (ImGui::SliderFloat("##medium_noise_scale", &m.grid.noise_scale, 0.01f, 100.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
        changed = true;
      }
      ImGui::TextUnformatted("Octaves");
      full_width_item();
      if (ImGui::SliderInt("##medium_noise_octaves", reinterpret_cast<int32_t*>(&m.grid.noise_octaves), 1, 16, "%d", ImGuiSliderFlags_AlwaysClamp)) {
        changed = true;
      }
      ImGui::TextUnformatted("Lacunarity");
      full_width_item();
      if (ImGui::SliderFloat("##medium_noise_lacunarity", &m.grid.noise_lacunarity, 1.0f, 4.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
        changed = true;
      }
      ImGui::TextUnformatted("Persistence");
      full_width_item();
      if (ImGui::SliderFloat("##medium_noise_persistence", &m.grid.noise_persistence, 0.01f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
        changed = true;
      }
      ImGui::TextUnformatted("Seed");
      full_width_item();
      if (ImGui::InputInt("##medium_noise_seed", reinterpret_cast<int32_t*>(&m.grid.noise_seed))) {
        changed = true;
      }
      ImGui::TextUnformatted("Power");
      full_width_item();
      if (ImGui::SliderFloat("##medium_noise_power", &m.grid.noise_power, 0.1f, 10.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
        changed = true;
      }
      ImGui::TextUnformatted("Sharpness");
      full_width_item();
      if (ImGui::SliderFloat("##medium_noise_sharpness", &m.grid.noise_sharpness, 0.1f, 10.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
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
        ImGui::TextUnformatted("Fade Distance");
        full_width_item();
        if (ImGui::SliderFloat("##medium_noise_border_fade_distance", &m.grid.noise_border_fade_distance, 0.01f, 0.5f, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
          changed = true;
        }
      }
    }
  }

  return changed;
}

void UI::clear_selection_history() {
  _selection_history.clear();
  _selection_history_cursor = -1;
}

void UI::reset_selection() {
  _selection = {};
  _pending_selection = {};
  _name_edit_selection = {};
  _name_edit_buffer[0] = 0;
  clear_selection_history();
  _spectrum_editors.clear();
  _material_anisotropy.clear();
  _selected_material_positions.clear();
  _material_selection_anchor = -1;
}

void UI::reset_scene_state() {
  reset_selection();
  _node_transform_editor = {};
  _node_geometry_edit_result_node = -1;
  _node_geometry_edit_result = NodeGeometryEditResult::Success;
  _node_transform_editor_interaction_active = false;
  _node_transform_editor_interaction_rendered_this_frame = false;
  _node_transform_editor_interaction_node_index = -1;
  _material_interaction_active = false;
  _material_editor_rendered_this_frame = false;
  _material_interaction_indices.clear();
  _editing_material_indices = nullptr;
  _material_batch_changed_fields = 0u;
  _auto_open_emission_section = false;
  _scene_tree_open_subtree_ends.clear();
  _viewport_geometry = {};
  _viewport_pointer_active = false;
  _gizmo_captures_mouse = false;
  _gizmo_was_using = false;
  _resource_filter[0] = 0;
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
#if defined(ETX_PLATFORM_WINDOWS)
  constexpr float title_bar_height = 40.0f;
  const ImGuiStyle& style = ImGui::GetStyle();
  const float title_bar_padding = std::max(style.FramePadding.y, 0.5f * (title_bar_height - ImGui::GetFontSize()));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(style.FramePadding.x, title_bar_padding));
  const bool menu_bar_open = ImGui::BeginMainMenuBar();
  ImGui::PopStyleVar();
#else
  const bool menu_bar_open = ImGui::BeginMainMenuBar();
#endif

  if (menu_bar_open) {
#if defined(ETX_PLATFORM_WINDOWS)
    ImGui::PushStyleVar(ImGuiStyleVar_MenuItemRounding, 6.0f);
#endif

    if (ImGui::BeginMenu("ETX Tracer")) {
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
      if (ImGui::MenuItem("Save", "Ctrl+S", false, true)) {
        execute_menu_command(MenuCommand::SaveScene);
      }
      if (ImGui::MenuItem("Save as...", "Ctrl+Shift+S", false, true)) {
        execute_menu_command(MenuCommand::SaveSceneAs);
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Integrator", true)) {
      const uint32_t raster_argument = render_configuration_argument(RendererMode::Rasterization, kInvalidIndex);
      if (ImGui::MenuItem("Raster Preview", nullptr, render_configuration_selected(raster_argument))) {
        execute_menu_command(MenuCommand::SelectIntegrator, raster_argument);
      }

      ImGui::SeparatorText("CPU");
      for (uint64_t i = 0u; i < _integrators.count; ++i) {
        Integrator* const integrator = _integrators[i];
        if ((integrator == nullptr) || (integrator->enabled() == false)) {
          continue;
        }
        const uint32_t configuration_argument = render_configuration_argument(RendererMode::CPURaytracing, static_cast<uint32_t>(i));
        const std::string label = render_configuration_label(RendererMode::CPURaytracing, integrator);
        if (ImGui::MenuItem(label.c_str(), nullptr, render_configuration_selected(configuration_argument))) {
          execute_menu_command(MenuCommand::SelectIntegrator, configuration_argument);
        }
      }

      if (_gpu_renderer_available) {
        ImGui::SeparatorText("GPU");
        for (uint64_t i = 0u; i < _integrators.count; ++i) {
          Integrator* const integrator = _integrators[i];
          if ((integrator == nullptr) || (integrator->enabled() == false) || (gpu_integrator_supported(integrator->type()) == false)) {
            continue;
          }
          const uint32_t configuration_argument = render_configuration_argument(RendererMode::GPURaytracing, static_cast<uint32_t>(i));
          const std::string label = render_configuration_label(RendererMode::GPURaytracing, integrator);
          if (ImGui::MenuItem(label.c_str(), nullptr, render_configuration_selected(configuration_argument))) {
            execute_menu_command(MenuCommand::SelectIntegrator, configuration_argument);
          }
        }
      }

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Image", true)) {
      if (ImGui::MenuItem("Open Reference Image...", "Ctrl+I", false, true)) {
        execute_menu_command(MenuCommand::OpenReferenceImage);
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Save Current Image (RGB)...", "Ctrl+E", false, true)) {
        execute_menu_command(MenuCommand::SaveImageRGB);
      }
      if (ImGui::MenuItem("Save Current Image (LDR)...", "Ctrl+Shift+E", false, true)) {
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
      ui_toggle("Scene Explorer", UIObjects, MenuCommand::ToggleSceneObjects);
      ui_toggle("Inspector", UIProperties, MenuCommand::ToggleProperties);
      ui_toggle("Diagnostics", UIMemoryDiagnostics, MenuCommand::ToggleMemoryDiagnostics);
      ImGui::Separator();
      if (ImGui::MenuItem("Reset Layout")) {
        execute_menu_command(MenuCommand::ResetLayout);
      }
      ImGui::EndMenu();
    }

#if defined(ETX_PLATFORM_WINDOWS)
    const ImVec2 menu_bar_position = ImGui::GetWindowPos();
    const float menu_width = ImGui::GetCursorScreenPos().x - menu_bar_position.x + ImGui::GetStyle().ItemSpacing.x;
    ImGui::PopStyleVar();
    const float controls_left = build_title_bar_controls();
    platform_ui().set_title_bar_layout(menu_width, controls_left, ImGui::GetWindowHeight(), ImGui::GetIO().DisplayFramebufferScale.x);
#endif

    ImGui::EndMainMenuBar();
  }
}

#if defined(ETX_PLATFORM_WINDOWS)
float UI::build_title_bar_controls() {
  constexpr float control_width = 46.0f;
  constexpr float control_rounding = 6.0f;
  constexpr float glyph_half_size = 5.0f;
  constexpr float glyph_thickness = 1.25f;
  constexpr uint32_t control_count = 3u;

  const ImVec2 window_position = ImGui::GetWindowPos();
  const ImVec2 window_size = ImGui::GetWindowSize();
  const float controls_left = std::max(0.0f, window_size.x - control_width * static_cast<float>(control_count));
  const ImVec2 control_size(control_width, window_size.y);
  const char* const control_ids[control_count] = {
    "##title-bar-minimize",
    "##title-bar-maximize",
    "##title-bar-close",
  };
  const PlatformTitleBarCommand control_commands[control_count] = {
    PlatformTitleBarCommand::Minimize,
    PlatformTitleBarCommand::ToggleMaximize,
    PlatformTitleBarCommand::Close,
  };

  ImGui::SetCursorScreenPos(ImVec2(window_position.x + controls_left, window_position.y));
  for (uint32_t control_index = 0u; control_index < control_count; ++control_index) {
    const bool pressed = ImGui::InvisibleButton(control_ids[control_index], control_size);
    const bool hovered = ImGui::IsItemHovered();
    const bool held = ImGui::IsItemActive();
    const ImVec2 control_min = ImGui::GetItemRectMin();
    const ImVec2 control_max = ImGui::GetItemRectMax();
    const ImVec2 center(0.5f * (control_min.x + control_max.x), 0.5f * (control_min.y + control_max.y));
    ImDrawList* const draw_list = ImGui::GetForegroundDrawList(ImGui::GetMainViewport());

    ImU32 background_color = ImGui::GetColorU32(ImGuiCol_MenuBarBg);
    if (hovered || held) {
      background_color = control_index == 2u ? IM_COL32(196, 43, 28, held ? 255 : 230) : ImGui::GetColorU32(held ? ImGuiCol_HeaderActive : ImGuiCol_HeaderHovered);
      draw_list->AddRectFilled(control_min, control_max, background_color, control_rounding);
    }

    const ImU32 glyph_color = ImGui::GetColorU32(ImGuiCol_Text);
    if (control_index == 0u) {
      draw_list->AddLine(ImVec2(center.x - glyph_half_size, center.y + 3.0f), ImVec2(center.x + glyph_half_size, center.y + 3.0f), glyph_color, glyph_thickness);
    } else if (control_index == 1u) {
      if (platform_ui().window_maximized()) {
        draw_list->AddRect(ImVec2(center.x - 3.0f, center.y - glyph_half_size), ImVec2(center.x + glyph_half_size, center.y + 3.0f), glyph_color, 0.0f, 0, glyph_thickness);
        draw_list->AddRectFilled(ImVec2(center.x - glyph_half_size - 1.0f, center.y - 3.0f), ImVec2(center.x + 3.0f, center.y + glyph_half_size + 1.0f), background_color);
        draw_list->AddRect(ImVec2(center.x - glyph_half_size, center.y - 3.0f), ImVec2(center.x + 3.0f, center.y + glyph_half_size), glyph_color, 0.0f, 0, glyph_thickness);
      } else {
        draw_list->AddRect(ImVec2(center.x - glyph_half_size, center.y - glyph_half_size), ImVec2(center.x + glyph_half_size, center.y + glyph_half_size), glyph_color, 0.0f, 0,
          glyph_thickness);
      }
    } else {
      draw_list->AddLine(ImVec2(center.x - glyph_half_size, center.y - glyph_half_size), ImVec2(center.x + glyph_half_size, center.y + glyph_half_size), glyph_color,
        glyph_thickness);
      draw_list->AddLine(ImVec2(center.x + glyph_half_size, center.y - glyph_half_size), ImVec2(center.x - glyph_half_size, center.y + glyph_half_size), glyph_color,
        glyph_thickness);
    }

    if (pressed) {
      platform_ui().execute_title_bar_command(control_commands[control_index]);
    }

    if ((control_index + 1u) < control_count) {
      ImGui::SameLine(0.0f, 0.0f);
    }
  }

  return controls_left;
}
#endif

void UI::select_render_configuration(RendererMode renderer, Integrator* integrator) {
  if ((renderer != RendererMode::Rasterization) && (integrator == nullptr)) {
    return;
  }
  if ((renderer == RendererMode::GPURaytracing) && ((_gpu_renderer_available == false) || (gpu_integrator_supported(integrator->type()) == false))) {
    return;
  }
  if ((integrator != nullptr) && (integrator->enabled() == false)) {
    return;
  }

  if (callbacks.render_configuration_selected) {
    callbacks.render_configuration_selected(renderer, integrator != nullptr ? integrator->type() : Integrator::Type::Invalid);
  }
  _current_renderer_mode = renderer;
  if (integrator != nullptr) {
    set_current_integrator(integrator);
  }
}

void UI::build_render_configuration_selector(const char* id) {
  const std::string current_label = render_configuration_label(_current_renderer_mode, _current_integrator);
  if (ImGui::BeginCombo(id, current_label.c_str())) {
    const bool raster_selected = _current_renderer_mode == RendererMode::Rasterization;
    if (ImGui::Selectable("Raster Preview", raster_selected)) {
      select_render_configuration(RendererMode::Rasterization, nullptr);
    }

    ImGui::SeparatorText("CPU");
    for (uint64_t i = 0u; i < _integrators.count; ++i) {
      Integrator* const integrator = _integrators[i];
      if ((integrator == nullptr) || (integrator->enabled() == false)) {
        continue;
      }
      const bool selected = (_current_renderer_mode == RendererMode::CPURaytracing) && (_current_integrator == integrator);
      const std::string label = render_configuration_label(RendererMode::CPURaytracing, integrator);
      if (ImGui::Selectable(label.c_str(), selected)) {
        select_render_configuration(RendererMode::CPURaytracing, integrator);
      }
    }

    if (_gpu_renderer_available) {
      ImGui::SeparatorText("GPU");
      for (uint64_t i = 0u; i < _integrators.count; ++i) {
        Integrator* const integrator = _integrators[i];
        if ((integrator == nullptr) || (integrator->enabled() == false) || (gpu_integrator_supported(integrator->type()) == false)) {
          continue;
        }
        const bool selected = (_current_renderer_mode == RendererMode::GPURaytracing) && (_current_integrator == integrator);
        const std::string label = render_configuration_label(RendererMode::GPURaytracing, integrator);
        if (ImGui::Selectable(label.c_str(), selected)) {
          select_render_configuration(RendererMode::GPURaytracing, integrator);
        }
      }
    }
    ImGui::EndCombo();
  }
}

void UI::build_toolbar(const BuildContext& ctx) {
  if (ImGui::BeginViewportSideBar("##toolbar", ImGui::GetMainViewport(), ImGuiDir_Up, ctx.button_size + 2.0f * ctx.wpadding.y, ImGuiWindowFlags_NoDecoration)) {
    const bool cpu_mode = _current_renderer_mode == RendererMode::CPURaytracing;
    const bool gpu_mode = _current_renderer_mode == RendererMode::GPURaytracing;
    const bool compact_toolbar = ImGui::GetContentRegionAvail().x < 1080.0f;
    ImGui::GetStyle().FramePadding.y = (ctx.button_size - ctx.text_size) / 2.0f;
    ImGui::SetCursorPosX(ctx.wpadding.x);
    if (compact_toolbar == false) {
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted("Integrator");
      ImGui::SameLine();
    }
    ImGui::SetNextItemWidth(compact_toolbar ? 190.0f : 230.0f);
    build_render_configuration_selector("##toolbar_integrator");

    const RendererControlState& controls = _current_renderer_controls;
    auto toolbar_action = [&](const char* label, bool available, const SemanticButtonColors* colors, const char* tooltip, const std::function<void()>& action) {
      ImGui::SameLine(0.0f, ctx.wpadding.x);
      if (colors != nullptr) {
        push_semantic_button_colors(*colors);
      }
      if (available == false) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Button(label, ImVec2(0.0f, ctx.button_size)) && available && action) {
        action();
      }
      if (available == false) {
        ImGui::EndDisabled();
      }
      if (colors != nullptr) {
        ImGui::PopStyleColor(3);
      }
      draw_item_tooltip(tooltip, ImGuiHoveredFlags_DelayNormal);
    };

    if (cpu_mode || gpu_mode) {
      const SemanticButtonColors& launch_colors = launch_button_colors(_theme);
      const SemanticButtonColors& terminate_colors = terminate_button_colors(_theme);
      toolbar_action("Start", controls.can_run, &launch_colors, "Start a new render.", callbacks.run_selected);
      toolbar_action(compact_toolbar ? "Finish" : "Finish sample", controls.can_finish, nullptr, "Stop after the in-progress sample completes.", [this]() {
        if (callbacks.stop_selected) {
          callbacks.stop_selected(true);
        }
      });
      toolbar_action(compact_toolbar ? "Stop" : "Stop now", controls.can_stop, &terminate_colors, "Stop immediately and keep the latest completed output.", [this]() {
        if (callbacks.stop_selected) {
          callbacks.stop_selected(false);
        }
      });
      toolbar_action("Restart", controls.can_restart, nullptr, "Clear the current result and restart rendering.", callbacks.restart_selected);
    }

    ImGui::SameLine(0.0f, ctx.wpadding.x);
    if (ImGui::Button("Display", ImVec2(0.0f, ctx.button_size))) {
      ImGui::OpenPopup("##display_controls");
    }
    ImGui::SetNextWindowSize(ImVec2(360.0f, 0.0f), ImGuiCond_Always);
    if (ImGui::BeginPopup("##display_controls")) {
      ImGui::TextUnformatted("View layer");
      full_width_item();
      if (ImGui::BeginCombo("##toolbar_view_layer", Film::layer_name(_view_options.view_layer))) {
        for (uint32_t i = 0u; i < ViewLayer::Count; ++i) {
          const bool selected = i == _view_options.view_layer;
          if (ImGui::Selectable(Film::layer_name(i), selected)) {
            _view_options.view_layer = i;
            if (callbacks.view_layer_changed) {
              callbacks.view_layer_changed(i);
            }
          }
        }
        ImGui::EndCombo();
      }
      ImGui::TextUnformatted("Output");
      full_width_item();
      if (ImGui::BeginCombo("##toolbar_output", output_view_to_string(_view_options.view_image).c_str())) {
        for (uint32_t i = 0u; i < static_cast<uint32_t>(OutputView::Count); ++i) {
          const bool selected = i == _view_options.view_image;
          if (ImGui::Selectable(output_view_to_string(i).c_str(), selected)) {
            _view_options.view_image = i;
            if (callbacks.output_view_changed) {
              callbacks.output_view_changed(i);
            }
          }
        }
        ImGui::EndCombo();
      }
      ImGui::TextUnformatted("Display transform");
      full_width_item();
      if (ImGui::BeginCombo("##toolbar_display_transform", view_option_to_string(_view_options.view_option).c_str())) {
        for (uint32_t i = 0u; i < static_cast<uint32_t>(ViewOptions::Count); ++i) {
          const bool selected = i == _view_options.view_option;
          if (ImGui::Selectable(view_option_to_string(i).c_str(), selected)) {
            _view_options.view_option = i;
            if (callbacks.display_transform_changed) {
              callbacks.display_transform_changed(i);
            }
          }
        }
        ImGui::EndCombo();
      }
      if (cpu_mode) {
        const bool can_denoise = controls.can_run && (_current_renderer_status.progress_kind == RendererProgressKind::Samples) && (_current_renderer_status.completed_units > 0u);
        if (can_denoise == false) {
          ImGui::BeginDisabled();
        }
        if (ImGui::Button("Denoise result", ImVec2(-FLT_MIN, 0.0f)) && callbacks.denoise_selected) {
          callbacks.denoise_selected();
        }
        if (can_denoise == false) {
          ImGui::EndDisabled();
        }
      }
      ImGui::EndPopup();
    }

    ImGui::SameLine(0.0f, ctx.wpadding.x);
    if (compact_toolbar == false) {
      ImGui::AlignTextToFramePadding();
      ImGui::TextUnformatted("Zoom");
      ImGui::SameLine();
    }
    ImGui::SetNextItemWidth(compact_toolbar ? 95.0f : 125.0f);
    if (ImGui::BeginCombo("##viewport_zoom", kViewportZoomOptions[_viewport_zoom_option].label)) {
      for (uint32_t i = 0u; i < static_cast<uint32_t>(IM_ARRAYSIZE(kViewportZoomOptions)); ++i) {
        const bool selected = i == _viewport_zoom_option;
        if (ImGui::Selectable(kViewportZoomOptions[i].label, selected)) {
          _viewport_zoom_option = i;
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }

    ImGui::GetStyle().FramePadding.y = ctx.fpadding.y;
    ImGui::End();
  }
}

void UI::build_status_bar(const BuildContext& ctx) {
  if (ImGui::BeginViewportSideBar("##status", ImGui::GetMainViewport(), ImGuiDir_Down, ctx.text_size + 2.0f * ctx.wpadding.y, ImGuiWindowFlags_NoDecoration)) {
    const RendererStatus& status = _current_renderer_status;
    std::string core_status = format_string("%s  |  %s", renderer_mode_status_name(status.mode), renderer_status_state_name(status.state));
    if (_scene_dirty) {
      core_status += "  |  Modified";
    }
    if (status.progress_kind != RendererProgressKind::None) {
      const char* unit = renderer_progress_unit_name(status.progress_kind);
      if (status.total_units > 0u) {
        core_status += format_string("  |  %u / %u %s", status.completed_units, status.total_units, unit);
      } else {
        core_status += format_string("  |  %u %s", status.completed_units, unit);
      }
    }
    const bool path_progress_visible = ((status.state == RendererStatusState::Running) || (status.state == RendererStatusState::Finishing)) &&
                                       (status.path_phase != RendererPathPhase::None) && (status.total_path_count > 0u);
    double path_progress = 0.0;
    if (path_progress_visible) {
      path_progress = std::min(1.0, static_cast<double>(status.completed_path_count) / static_cast<double>(status.total_path_count));
      core_status += format_string("  |  %s %.1f%%", renderer_path_phase_short_name(status.path_phase), 100.0 * path_progress);
    }

    std::string timing_status = {};
    if (status.elapsed_available) {
      timing_status += "  |  Elapsed " + compact_duration_string(status.elapsed_seconds);
    }
    if (status.remaining_available) {
      timing_status += "  |  ETA " + compact_duration_string(status.remaining_seconds);
    }

    const double ui_frame_ms = (_current_fps > 0.0f) ? (1000.0 / static_cast<double>(_current_fps)) : 0.0;
    std::string performance_status = (_current_fps > 0.0f) ? format_string("UI %.1f FPS  |  %.1f ms", _current_fps, ui_frame_ms) : std::string("UI -- FPS");
    const float available_width = ImGui::GetContentRegionAvail().x;
    float performance_width = ImGui::CalcTextSize(performance_status.c_str()).x;
    float reserved_width = performance_width + 2.0f * ImGui::GetStyle().ItemSpacing.x;
    if ((ImGui::CalcTextSize(core_status.c_str()).x + reserved_width) > available_width) {
      performance_status = (_current_fps > 0.0f) ? format_string("%.1f FPS", _current_fps) : std::string("-- FPS");
      performance_width = ImGui::CalcTextSize(performance_status.c_str()).x;
      reserved_width = performance_width + 2.0f * ImGui::GetStyle().ItemSpacing.x;
    }
    std::string primary_status = core_status + timing_status;
    if ((ImGui::CalcTextSize(primary_status.c_str()).x + reserved_width) > available_width) {
      primary_status = core_status;
    }
    const bool performance_fits = (ImGui::CalcTextSize(primary_status.c_str()).x + reserved_width) <= available_width;

    std::string detailed_status = {};
    if (path_progress_visible) {
      detailed_status = format_string("%s: %llu / %llu (%.1f%%)", renderer_path_phase_name(status.path_phase), static_cast<unsigned long long>(status.completed_path_count),
        static_cast<unsigned long long>(status.total_path_count), 100.0 * path_progress);
    }
    if (status.elapsed_available) {
      if (detailed_status.empty() == false) {
        detailed_status += "\n";
      }
      detailed_status += "Elapsed: " + duration_string(status.elapsed_seconds);
    }
    if (status.remaining_available) {
      if (detailed_status.empty() == false) {
        detailed_status += "\n";
      }
      detailed_status += "Remaining: " + duration_string(status.remaining_seconds);
    }
    if (((status.state == RendererStatusState::Preparing) || (status.state == RendererStatusState::Failed)) && (_current_renderer_preparation.message.empty() == false)) {
      if (detailed_status.empty() == false) {
        detailed_status += "\n";
      }
      detailed_status += _current_renderer_preparation.message;
    }
    if (_current_fps > 0.0f) {
      if (detailed_status.empty() == false) {
        detailed_status += "\n";
      }
      detailed_status += format_string("UI frame: %.1f ms (%.1f FPS)", ui_frame_ms, _current_fps);
    }

    ImGui::TextUnformatted(primary_status.c_str());
    if (detailed_status.empty() == false) {
      draw_item_tooltip(detailed_status.c_str(), ImGuiHoveredFlags_DelayNormal);
    }
    const float performance_x = ImGui::GetCursorPosX() + available_width - performance_width;
    if (performance_fits) {
      ImGui::SameLine(performance_x);
      ImGui::TextDisabled("%s", performance_status.c_str());
      draw_item_tooltip("UI frame rate and frame time", ImGuiHoveredFlags_DelayNormal);
    }
    ImGui::End();
  }
}

void UI::request_quit_confirmation() {
  execute_menu_command(MenuCommand::Quit);
}

void UI::build_workspace(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  ImGuiViewport* const main_viewport = ImGui::GetMainViewport();
  if (main_viewport == nullptr) {
    _viewport_geometry = {};
    return;
  }

  if (_reset_layout_requested) {
    _explorer_width = 300.0f;
    _inspector_width = 400.0f;
    _diagnostics_height = 260.0f;
    _ui_setup = UIDefaults;
    _reset_layout_requested = false;
  }

  const bool explorer_visible = (_ui_setup & UIObjects) != 0u;
  const bool inspector_visible = (_ui_setup & UIProperties) != 0u;
  const bool diagnostics_visible = (_ui_setup & UIMemoryDiagnostics) != 0u;
  constexpr float splitter_size = 8.0f;
  constexpr float minimum_viewport_width = 320.0f;
  constexpr float minimum_panel_width = 220.0f;
  constexpr float minimum_inspector_width = 300.0f;
  constexpr float minimum_main_height = 240.0f;

  const ImVec2 workspace_position = main_viewport->WorkPos;
  const ImVec2 workspace_size = main_viewport->WorkSize;
  ImGui::SetNextWindowPos(workspace_position, ImGuiCond_Always);
  ImGui::SetNextWindowSize(workspace_size, ImGuiCond_Always);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  constexpr ImGuiWindowFlags workspace_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                               ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMouseInputs;
  if (ImGui::Begin("##editor_workspace", nullptr, workspace_flags) == false) {
    ImGui::End();
    ImGui::PopStyleVar();
    _viewport_geometry = {};
    return;
  }

  const ImVec2 content_origin = ImGui::GetCursorScreenPos();
  const ImVec2 content_size = ImGui::GetContentRegionAvail();
  const float horizontal_splitter_count = static_cast<float>((explorer_visible ? 1u : 0u) + (inspector_visible ? 1u : 0u));
  const float available_panel_width = std::max(0.0f, content_size.x - minimum_viewport_width - horizontal_splitter_count * splitter_size);
  float explorer_width = explorer_visible ? std::min(std::max(_explorer_width, minimum_panel_width), available_panel_width) : 0.0f;
  float inspector_width = inspector_visible ? std::min(std::max(_inspector_width, minimum_inspector_width), available_panel_width) : 0.0f;
  const float requested_panel_width = explorer_width + inspector_width;
  if ((requested_panel_width > available_panel_width) && (requested_panel_width > 0.0f)) {
    const float panel_scale = available_panel_width / requested_panel_width;
    explorer_width *= panel_scale;
    inspector_width *= panel_scale;
  }

  float main_height = content_size.y;
  if (diagnostics_visible) {
    _diagnostics_height = std::clamp(_diagnostics_height, 150.0f, std::max(150.0f, content_size.y - minimum_main_height - splitter_size));
    main_height = std::max(minimum_main_height, content_size.y - _diagnostics_height - splitter_size);
  }

  float cursor_x = content_origin.x;
  if (explorer_visible) {
    ImGui::SetCursorScreenPos(ImVec2(cursor_x, content_origin.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ctx.wpadding.x, ctx.wpadding.y));
    ImGui::BeginChild("##workspace_explorer", ImVec2(explorer_width, main_height), ImGuiChildFlags_Borders);
    if (data.scene_loaded == false) {
      ImGui::BeginDisabled();
    }
    build_scene_explorer(scene_rep, ctx);
    if (data.scene_loaded == false) {
      ImGui::EndDisabled();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    cursor_x += explorer_width;

    ImGui::SetCursorScreenPos(ImVec2(cursor_x, content_origin.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::BeginChild("##explorer_splitter_region", ImVec2(splitter_size, main_height), ImGuiChildFlags_None,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground);
    ImGui::InvisibleButton("##explorer_splitter", ImGui::GetContentRegionAvail());
    if (ImGui::IsItemActive()) {
      _explorer_width = std::max(minimum_panel_width, _explorer_width + ImGui::GetIO().MouseDelta.x);
    }
    draw_workspace_splitter(ImGuiMouseCursor_ResizeEW);
    ImGui::EndChild();
    ImGui::PopStyleVar();
    cursor_x += splitter_size;
  }

  const float inspector_and_splitter_width = inspector_visible ? (inspector_width + splitter_size) : 0.0f;
  const float viewport_width = std::max(1.0f, content_origin.x + content_size.x - cursor_x - inspector_and_splitter_width);
  ImGui::SetCursorScreenPos(ImVec2(cursor_x, content_origin.y));
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGuiWindowFlags viewport_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground;
  if (data.scene_loaded) {
    viewport_flags |= ImGuiWindowFlags_NoInputs;
  }
  ImGui::BeginChild("##workspace_viewport", ImVec2(viewport_width, main_height), ImGuiChildFlags_None, viewport_flags);
  const ImVec2 viewport_position = ImGui::GetWindowPos();
  const ImVec2 viewport_size = ImGui::GetWindowSize();
  const ImVec2 framebuffer_scale = ImGui::GetIO().DisplayFramebufferScale;
  ImVec2 displayed_image_size = viewport_size;
  if ((data.output_size.x > 0u) && (data.output_size.y > 0u) && (viewport_size.x > 0.0f) && (viewport_size.y > 0.0f)) {
    const float output_aspect = static_cast<float>(data.output_size.x) / static_cast<float>(data.output_size.y);
    const float zoom_scale = kViewportZoomOptions[_viewport_zoom_option].scale;
    if (zoom_scale > 0.0f) {
      displayed_image_size = ImVec2(static_cast<float>(data.output_size.x) * zoom_scale / std::max(1.0e-6f, framebuffer_scale.x),
        static_cast<float>(data.output_size.y) * zoom_scale / std::max(1.0e-6f, framebuffer_scale.y));
    } else {
      const float viewport_aspect = viewport_size.x / viewport_size.y;
      if (output_aspect > viewport_aspect) {
        displayed_image_size.y = viewport_size.x / output_aspect;
      } else {
        displayed_image_size.x = viewport_size.y * output_aspect;
      }
    }
  }
  const ImVec2 displayed_image_position(viewport_position.x + 0.5f * (viewport_size.x - displayed_image_size.x),
    viewport_position.y + 0.5f * (viewport_size.y - displayed_image_size.y));
  _viewport_geometry = {
    .logical_position = {viewport_position.x, viewport_position.y},
    .logical_size = {viewport_size.x, viewport_size.y},
    .image_position = {displayed_image_position.x, displayed_image_position.y},
    .image_size = {displayed_image_size.x, displayed_image_size.y},
    .framebuffer_scale = {framebuffer_scale.x, framebuffer_scale.y},
    .valid = data.scene_loaded && (viewport_size.x > 0.0f) && (viewport_size.y > 0.0f) && (displayed_image_size.x > 0.0f) && (displayed_image_size.y > 0.0f),
  };
  if (data.scene_loaded == false) {
    const float content_width = std::max(1.0f, std::min(280.0f, viewport_size.x - 32.0f));
    const float content_height = data.recent_files.empty() ? 90.0f : 190.0f;
    ImGui::SetCursorPos(ImVec2(std::max(16.0f, 0.5f * (viewport_size.x - content_width)), std::max(16.0f, 0.5f * (viewport_size.y - content_height))));
    ImGui::BeginGroup();
    const char* empty_title = "Open a scene to begin";
    const float title_x = ImGui::GetCursorPosX() + 0.5f * (content_width - ImGui::CalcTextSize(empty_title).x);
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), title_x));
    ImGui::TextUnformatted(empty_title);
    ImGui::Spacing();
    if (ImGui::Button("Open Scene…", ImVec2(content_width, 0.0f))) {
      execute_menu_command(MenuCommand::OpenScene);
    }
    if (data.recent_files.empty() == false) {
      ImGui::Spacing();
      ImGui::TextDisabled("Recent Scenes");
      const size_t recent_count = std::min<size_t>(3u, data.recent_files.size());
      for (size_t recent_index = 0u; recent_index < recent_count; ++recent_index) {
        const std::string& path = data.recent_files[data.recent_files.size() - recent_index - 1u];
        const std::string display_name = std::filesystem::path(path).filename().string();
        ImGui::PushID(static_cast<int>(recent_index));
        if (ImGui::Button(display_name.c_str(), ImVec2(content_width, 0.0f))) {
          execute_menu_command(MenuCommand::OpenRecentScene, 0u, path);
        }
        draw_item_tooltip(path.c_str(), ImGuiHoveredFlags_DelayNormal);
        ImGui::PopID();
      }
    }
    ImGui::EndGroup();
  }
  ImGui::EndChild();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
  cursor_x += viewport_width;

  if (inspector_visible) {
    ImGui::SetCursorScreenPos(ImVec2(cursor_x, content_origin.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::BeginChild("##inspector_splitter_region", ImVec2(splitter_size, main_height), ImGuiChildFlags_None,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground);
    ImGui::InvisibleButton("##inspector_splitter", ImGui::GetContentRegionAvail());
    if (ImGui::IsItemActive()) {
      _inspector_width = std::max(minimum_inspector_width, _inspector_width - ImGui::GetIO().MouseDelta.x);
    }
    draw_workspace_splitter(ImGuiMouseCursor_ResizeEW);
    ImGui::EndChild();
    ImGui::PopStyleVar();
    cursor_x += splitter_size;

    ImGui::SetCursorScreenPos(ImVec2(cursor_x, content_origin.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ctx.wpadding.x, ctx.wpadding.y));
    ImGui::BeginChild("##workspace_inspector", ImVec2(inspector_width, main_height), ImGuiChildFlags_Borders);
    if (data.scene_loaded == false) {
      ImGui::BeginDisabled();
    }
    build_inspector(scene_rep, ctx, data);
    if (data.scene_loaded == false) {
      ImGui::EndDisabled();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
  }

  if (diagnostics_visible) {
    const float splitter_y = content_origin.y + main_height;
    ImGui::SetCursorScreenPos(ImVec2(content_origin.x, splitter_y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::BeginChild("##diagnostics_splitter_region", ImVec2(content_size.x, splitter_size), ImGuiChildFlags_None,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoBackground);
    ImGui::InvisibleButton("##diagnostics_splitter", ImGui::GetContentRegionAvail());
    if (ImGui::IsItemActive()) {
      _diagnostics_height = std::max(150.0f, _diagnostics_height - ImGui::GetIO().MouseDelta.y);
    }
    draw_workspace_splitter(ImGuiMouseCursor_ResizeNS);
    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::SetCursorScreenPos(ImVec2(content_origin.x, splitter_y + splitter_size));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(ctx.wpadding.x, ctx.wpadding.y));
    ImGui::BeginChild("##workspace_diagnostics", ImVec2(content_size.x, content_size.y - main_height - splitter_size), ImGuiChildFlags_Borders);
    build_diagnostics(scene_rep, ctx, data);
    ImGui::EndChild();
    ImGui::PopStyleVar();
  }

  ImGui::End();
  ImGui::PopStyleVar();
}

void UI::build_scene_explorer(SceneRepresentation& scene_rep, const BuildContext& ctx) {
  if (ImGui::BeginTabBar("##scene_explorer_tabs")) {
    if (ImGui::BeginTabItem("Scene")) {
      build_scene_tree_window(scene_rep, ctx);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Resources")) {
      build_scene_objects_window(scene_rep, ctx);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

void UI::build_inspector(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  if (ImGui::BeginTabBar("##inspector_tabs")) {
    if (ImGui::BeginTabItem("Inspector")) {
      build_properties_window(scene_rep, ctx, data);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Render")) {
      build_rendering_properties(scene_rep, ctx, data);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

void UI::build_diagnostics(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  (void)ctx;
  if (ImGui::BeginTabBar("##diagnostics_tabs")) {
    if (ImGui::BeginTabItem("Performance")) {
      build_debug_info_content();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Memory")) {
      build_memory_diagnostics_content(scene_rep, data.film);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

void UI::build_debug_info_content() {
  const bool has_integrator_debug_info =
    (_current_integrator != nullptr) && (_current_integrator->status().debug_info_count > 0u) && (_current_integrator->status().debug_info != nullptr);
  const bool show_gpu_kernel_timings = (_current_renderer_mode == RendererMode::GPURaytracing) && _gpu_kernel_timing_stats.enabled;
  if ((has_integrator_debug_info == false) && (show_gpu_kernel_timings == false)) {
    ImGui::TextDisabled("Enable GPU kernel profiling or run an integrator that exposes debug metrics.");
    return;
  }

  if (has_integrator_debug_info) {
    const auto debug_info = _current_integrator->status().debug_info;
    float metric_width = 160.0f;
    for (uint64_t i = 0, e = _current_integrator->status().debug_info_count; i < e; ++i) {
      metric_width = std::max(metric_width, ImGui::CalcTextSize(debug_info[i].title).x);
    }
    metric_width = std::min(metric_width, 480.0f);
    const ImGuiTableFlags table_flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoHostExtendX | ImGuiTableFlags_BordersInnerV;
    if (ImGui::BeginTable("##integrator_debug_info", 2, table_flags)) {
      ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, metric_width);
      ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 88.0f);
      for (uint64_t i = 0, e = _current_integrator->status().debug_info_count; i < e; ++i) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(debug_info[i].title);
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", debug_info[i].value);
      }
      ImGui::EndTable();
    }
  }

  if (show_gpu_kernel_timings) {
    if (has_integrator_debug_info) {
      ImGui::Separator();
    }
    ImGui::Text("Kernel execution: %.3f ms", _gpu_kernel_timing_stats.total_ms);
    draw_item_tooltip("Sum of timestamped compute dispatch durations. CPU scheduling, command-buffer gaps, barriers, transfers, and UI time are excluded.",
      ImGuiHoveredFlags_DelayNormal);
    if (_gpu_kernel_timing_stats.supported == false) {
      ImGui::TextDisabled("GPU timestamps are unavailable on the active backend.");
      return;
    }
    uint64_t captured_dispatch_count = 0u;
    for (const RendererKernelTiming& timing : _gpu_kernel_timing_stats.kernels) {
      captured_dispatch_count += timing.dispatch_count;
    }
    if (_gpu_kernel_timing_stats.capture_elapsed_ms > 0.0) {
      const double timestamped_percentage = (_gpu_kernel_timing_stats.total_ms * 100.0) / _gpu_kernel_timing_stats.capture_elapsed_ms;
      ImGui::SameLine();
      ImGui::TextDisabled("Profile wall: %.0f ms | kernel coverage: %.1f%%", _gpu_kernel_timing_stats.capture_elapsed_ms, timestamped_percentage);
      if (_gpu_kernel_timing_stats.captured_sample_count > 0u) {
        const double completed_samples = static_cast<double>(_gpu_kernel_timing_stats.captured_sample_count);
        ImGui::TextDisabled("Average / captured sample: elapsed %.1f ms | kernels %.1f ms | dispatches %.1f", _gpu_kernel_timing_stats.capture_elapsed_ms / completed_samples,
          _gpu_kernel_timing_stats.total_ms / completed_samples, static_cast<double>(captured_dispatch_count) / completed_samples);
      }
    }
    ImGui::TextDisabled("Captured samples: %llu | dispatches: %llu | dropped: %llu", static_cast<unsigned long long>(_gpu_kernel_timing_stats.captured_sample_count),
      static_cast<unsigned long long>(captured_dispatch_count), static_cast<unsigned long long>(_gpu_kernel_timing_stats.dropped_dispatch_count));
    if (_gpu_kernel_timing_stats.kernels.empty()) {
      ImGui::TextDisabled("Run the GPU renderer to collect kernel timings.");
      return;
    }
    if (ImGui::BeginTable("gpu_kernel_timings", 5,
          ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable)) {
      ImGui::TableSetupColumn("Kernel");
      ImGui::TableSetupColumn("Calls");
      ImGui::TableSetupColumn("Total ms");
      ImGui::TableSetupColumn("Avg us");
      ImGui::TableSetupColumn("%");
      ImGui::TableHeadersRow();
      for (const RendererKernelTiming& timing : _gpu_kernel_timing_stats.kernels) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(timing.name.c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%llu", static_cast<unsigned long long>(timing.dispatch_count));
        ImGui::TableNextColumn();
        ImGui::Text("%.3f", timing.total_ms);
        ImGui::TableNextColumn();
        ImGui::Text("%.2f", timing.average_ms * 1000.0);
        ImGui::TableNextColumn();
        ImGui::Text("%.1f", timing.percentage);
      }
      ImGui::EndTable();
    }
  }
}

void UI::build_renderer_preparation_modal() {
  constexpr const char* popup_id = "Activity##renderer_preparation";
  const bool preparing = _current_renderer_preparation.state == RendererPreparationState::Preparing;
  const bool popup_open = ImGui::IsPopupOpen(popup_id);
  if (preparing && (popup_open == false)) {
    ImGui::OpenPopup(popup_id);
  }
  if ((preparing == false) && (popup_open == false)) {
    return;
  }

  const ImVec2 display_size = ImGui::GetIO().DisplaySize;
  const ImVec2 modal_size = {
    std::min(760.0f, display_size.x - 48.0f),
    std::min(620.0f, display_size.y - 48.0f),
  };
  ImGui::SetNextWindowPos(ImVec2(display_size.x * 0.5f, display_size.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(modal_size, ImGuiCond_Always);

  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;
  if (ImGui::BeginPopupModal(popup_id, nullptr, flags)) {
    if (preparing == false) {
      ImGui::CloseCurrentPopup();
      ImGui::EndPopup();
      return;
    }

    const RendererPreparationStatus& status = _current_renderer_preparation;
    ImGui::TextUnformatted(status.phase.empty() ? "Preparing" : status.phase.c_str());
    if (status.message.empty() == false) {
      ImGui::TextWrapped("%s", status.message.c_str());
    }

    const float progress = (status.total_steps > 0u) ? (static_cast<float>(status.completed_steps) / static_cast<float>(status.total_steps)) : 0.0f;
    const std::string progress_label =
      (status.total_steps > 0u) ? (std::to_string(status.completed_steps) + " / " + std::to_string(status.total_steps)) : std::string("Starting...");
    ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), progress_label.c_str());

    if (status.remaining_available) {
      ImGui::Text("Elapsed: %s | Remaining: %s", duration_string(status.elapsed_seconds).c_str(), duration_string(status.remaining_seconds).c_str());
    } else if (status.worker_count > 0u) {
      ImGui::Text("Elapsed: %s | Workers: %u", duration_string(status.elapsed_seconds).c_str(), status.worker_count);
    } else {
      ImGui::Text("Elapsed: %s", duration_string(status.elapsed_seconds).c_str());
    }

    if (status.steps.empty() == false) {
      ImGui::SeparatorText("Shader pipelines");
      constexpr ImGuiTableFlags table_flags =
        ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Resizable;
      if (ImGui::BeginTable("##pipeline_creation_progress", 5, table_flags, ImVec2(0.0f, -52.0f))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Pipeline", ImGuiTableColumnFlags_WidthStretch, 0.30f);
        ImGui::TableSetupColumn("Variant", ImGuiTableColumnFlags_WidthStretch, 0.31f);
        ImGui::TableSetupColumn("SPIR-V", ImGuiTableColumnFlags_WidthFixed, 72.0f);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 92.0f);
        ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 82.0f);
        ImGui::TableHeadersRow();
        for (const auto& step : status.steps) {
          const char* step_state = "DXC queue";
          if (step.state == RendererPreparationStepState::CompilingSpirV) {
            step_state = "DXC";
          } else if (step.state == RendererPreparationStepState::QueuedForDriver) {
            step_state = "Driver queue";
          } else if (step.state == RendererPreparationStepState::CheckingCache) {
            step_state = "Cache";
          } else if (step.state == RendererPreparationStepState::DriverCompiling) {
            step_state = "Driver";
          } else if (step.state == RendererPreparationStepState::Complete) {
            step_state = step.cache_hit ? "Cached" : "Complete";
          } else if (step.state == RendererPreparationStepState::Failed) {
            step_state = "Failed";
          }

          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          ImGui::TextUnformatted(step.name.c_str());
          draw_item_tooltip(step.name.c_str(), ImGuiHoveredFlags_DelayNormal);
          ImGui::TableSetColumnIndex(1);
          ImGui::TextUnformatted(step.detail.c_str());
          draw_item_tooltip(step.detail.c_str(), ImGuiHoveredFlags_DelayNormal);
          ImGui::TableSetColumnIndex(2);
          if (step.spirv_size_bytes == 0u) {
            ImGui::TextUnformatted("-");
          } else if (step.spirv_size_bytes >= 1024u) {
            ImGui::Text("%.1f KiB", static_cast<double>(step.spirv_size_bytes) / 1024.0);
          } else {
            ImGui::Text("%llu B", static_cast<unsigned long long>(step.spirv_size_bytes));
          }
          ImGui::TableSetColumnIndex(3);
          ImGui::TextUnformatted(step_state);
          ImGui::TableSetColumnIndex(4);
          if ((step.state == RendererPreparationStepState::Complete) || (step.state == RendererPreparationStepState::Failed)) {
            ImGui::Text("%.1f ms", step.elapsed_ms);
          } else {
            ImGui::TextUnformatted("-");
          }
        }
        ImGui::EndTable();
      }
    } else {
      ImGui::Dummy(ImVec2(0.0f, ImGui::GetContentRegionAvail().y - 42.0f));
    }

    if (status.cancelable) {
      const float button_width = 180.0f;
      ImGui::SetCursorPosX((ImGui::GetWindowWidth() - button_width) * 0.5f);
      if (ImGui::Button("Cancel and use CPU", ImVec2(button_width, 0.0f)) && callbacks.cancel_renderer_preparation_selected) {
        callbacks.cancel_renderer_preparation_selected();
      }
    }
    ImGui::EndPopup();
  }
}

void UI::build_memory_diagnostics_content(SceneRepresentation& scene_rep, const Film& film) {
  const float column_spacing = ImGui::GetStyle().ItemSpacing.x;
  const float column_width = std::max(1.0f, (ImGui::GetContentRegionAvail().x - column_spacing) * 0.5f);
  ImGui::BeginChild("##cpu_memory_column", ImVec2(column_width, 0.0f));
  if (ImGui::CollapsingHeader("CPU memory", ImGuiTreeNodeFlags_DefaultOpen)) {
    draw_memory_summary_table("cpu_memory_summary", {
                                                      {"Process working set", _rhi_memory_stats.cpu_used_bytes},
                                                      {"Peak working set", _rhi_memory_stats.cpu_peak_used_bytes},
                                                      {"Process private commit", _rhi_memory_stats.cpu_private_bytes},
                                                    });

    const SceneData& scene_data = scene_rep.data();
    const uint64_t geometry_bytes = scene_data.vertices.pos.capacity() * sizeof(float3) + scene_data.vertices.nrm.capacity() * sizeof(float3) +
                                    scene_data.vertices.tan.capacity() * sizeof(float3) + scene_data.vertices.btn.capacity() * sizeof(float3) +
                                    scene_data.vertices.tex.capacity() * sizeof(float2) + scene_data.triangles.capacity() * sizeof(Triangle) +
                                    scene_data.meshes.capacity() * sizeof(Mesh);
    const uint64_t shading_bytes = scene_data.materials.capacity() * sizeof(Material) + scene_data.spectrum_values.capacity() * sizeof(SpectralDistribution) +
                                   scene_data.emitter_profiles.capacity() * sizeof(EmitterProfile) +
                                   scene_data.energy_compensation_interfaces.capacity() * sizeof(Scene::EnergyCompensationInterface);
    const uint64_t descriptor_bytes =
      scene_data.images_vector.capacity() * sizeof(Image) + scene_data.mediums_vector.capacity() * sizeof(Medium) + scene_data.cameras.capacity() * sizeof(SceneData::CameraInfo);
    const BufferPool::Stats pool_stats = scene_data.buffer_pool.stats();
    const Film::MemoryStats film_stats = film.memory_stats();
    const uint64_t film_bytes = film_stats.accumulation_bytes + film_stats.adaptive_bytes + film_stats.normals_bytes + film_stats.albedo_bytes + film_stats.denoised_bytes +
                                film_stats.output_bytes + film_stats.internal_bytes;
    const uint64_t tracked_cpu_bytes = geometry_bytes + shading_bytes + descriptor_bytes + pool_stats.capacity_bytes + film_bytes;
    const uint64_t untracked_working_set = (_rhi_memory_stats.cpu_used_bytes > tracked_cpu_bytes) ? (_rhi_memory_stats.cpu_used_bytes - tracked_cpu_bytes) : 0u;

    ImGui::SeparatorText("Tracked CPU allocations");
    draw_memory_summary_table("cpu_memory_breakdown", {
                                                        {"Scene geometry containers", geometry_bytes},
                                                        {"Scene shading containers", shading_bytes},
                                                        {"Scene image, medium, and camera descriptors", descriptor_bytes},
                                                        {"Scene payload used", pool_stats.used_bytes},
                                                        {"Scene payload reserved but unused", pool_stats.capacity_bytes - pool_stats.used_bytes},
                                                        {"Film accumulation", film_stats.accumulation_bytes},
                                                        {"Film adaptive accumulation", film_stats.adaptive_bytes},
                                                        {"Film normals", film_stats.normals_bytes},
                                                        {"Film albedo", film_stats.albedo_bytes},
                                                        {"Film denoised layer", film_stats.denoised_bytes},
                                                        {"Film output", film_stats.output_bytes},
                                                        {"Film internal state", film_stats.internal_bytes},
                                                        {"Other process working set", untracked_working_set},
                                                      });
  }
  ImGui::EndChild();

  ImGui::SameLine(0.0f, column_spacing);
  ImGui::BeginChild("##gpu_memory_column", ImVec2(0.0f, 0.0f));
  if (ImGui::CollapsingHeader("GPU memory", ImGuiTreeNodeFlags_DefaultOpen)) {
    draw_memory_summary_table("gpu_memory_summary",
      {
        {"RHI tracked allocations", _rhi_memory_stats.gpu_allocated_bytes},
        {"Buffers (" + std::to_string(_rhi_memory_stats.gpu_buffer_allocation_count) + ")", _rhi_memory_stats.gpu_buffer_allocated_bytes},
        {"Textures (" + std::to_string(_rhi_memory_stats.gpu_texture_allocation_count) + ")", _rhi_memory_stats.gpu_texture_allocated_bytes},
        {"Acceleration structures (" + std::to_string(_rhi_memory_stats.gpu_acceleration_structure_allocation_count) + ")",
          _rhi_memory_stats.gpu_acceleration_structure_allocated_bytes},
        {"Host-visible allocations (subset)", _rhi_memory_stats.gpu_host_visible_allocated_bytes},
      });

    uint64_t renderer_gpu_bytes = 0u;
    for (const RendererMemoryEntry& entry : _renderer_memory_stats.entries) {
      if (entry.location != RendererMemoryLocation::CPU) {
        renderer_gpu_bytes += entry.bytes;
      }
    }
    const uint64_t other_rhi_bytes = (_rhi_memory_stats.gpu_allocated_bytes > renderer_gpu_bytes) ? (_rhi_memory_stats.gpu_allocated_bytes - renderer_gpu_bytes) : 0u;

    std::vector<MemoryAllocationRow> allocation_rows;
    allocation_rows.reserve(_renderer_memory_stats.entries.size() + ((other_rhi_bytes > 0u) ? 1u : 0u));
    for (const RendererMemoryEntry& entry : _renderer_memory_stats.entries) {
      allocation_rows.push_back({entry.category, entry.name, renderer_memory_location_name(entry.location), entry.allocation_count, entry.bytes, true});
    }
    if (other_rhi_bytes > 0u) {
      allocation_rows.push_back({"RHI", "Other allocations and alignment overhead", "GPU", 0u, other_rhi_bytes, false});
    }

    ImGui::SeparatorText("Active renderer allocations");
    constexpr ImGuiTableFlags allocation_table_flags = ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp |
                                                       ImGuiTableFlags_ScrollY | ImGuiTableFlags_Sortable | ImGuiTableFlags_Resizable;
    if (ImGui::BeginTable("renderer_memory_breakdown", 5, allocation_table_flags, ImVec2(0.0f, 300.0f))) {
      ImGui::TableSetupScrollFreeze(0, 1);
      ImGui::TableSetupColumn("Category", ImGuiTableColumnFlags_WidthFixed, 90.0f, 0u);
      ImGui::TableSetupColumn("Allocation", ImGuiTableColumnFlags_WidthStretch, 0.0f, 1u);
      ImGui::TableSetupColumn("Location", ImGuiTableColumnFlags_WidthFixed, 112.0f, 2u);
      ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending, 54.0f, 3u);
      ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending, 112.0f, 4u);
      ImGui::TableHeadersRow();
      sort_memory_allocation_rows(allocation_rows, ImGui::TableGetSortSpecs());
      for (const MemoryAllocationRow& row : allocation_rows) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(row.category.c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(row.allocation.c_str());
        draw_item_tooltip(row.allocation.c_str(), ImGuiHoveredFlags_DelayNormal);
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(row.location.c_str());
        ImGui::TableNextColumn();
        if (row.count_available) {
          ImGui::Text("%u", row.count);
        } else {
          ImGui::TextUnformatted("-");
        }
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(memory_size_string(row.bytes).c_str());
      }
      ImGui::EndTable();
    }

    if (_renderer_memory_stats.light_vertex_capacity > 0u) {
      ImGui::SeparatorText("Compact light history");
      const float history_fraction =
        std::clamp(static_cast<float>(_renderer_memory_stats.light_vertex_count) / static_cast<float>(_renderer_memory_stats.light_vertex_capacity), 0.0f, 1.0f);
      const std::string history_label =
        std::to_string(_renderer_memory_stats.light_vertex_count) + " / " + std::to_string(_renderer_memory_stats.light_vertex_capacity) + " vertices";
      ImGui::ProgressBar(history_fraction, ImVec2(-1.0f, 0.0f), history_label.c_str());
      ImGui::Text("Paths: %u | Tile: %u / %u", _renderer_memory_stats.wavefront_path_capacity,
        std::min(_renderer_memory_stats.tile_index + 1u, std::max(1u, _renderer_memory_stats.tile_count)), std::max(1u, _renderer_memory_stats.tile_count));
    }
  }
  ImGui::EndChild();
}

void UI::build_scene_objects_window(SceneRepresentation& scene_rep, const BuildContext& ctx) {
  ctx.with_window(UIObjects, "Scene Objects", [&]() {
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##resource_filter", "Search resources", _resource_filter, sizeof(_resource_filter));
    ImGui::Spacing();
    if (ImGui::BeginTabBar("##resource_categories")) {
      const std::string materials_label = format_string("Materials (%zu)###resource_materials", static_cast<size_t>(_material_mapping.size()));
      if (ImGui::BeginTabItem(materials_label.c_str())) {
        if (ImGui::Button("Add Material", ImVec2(-FLT_MIN, 0.0f)) && callbacks.material_added) {
          const uint32_t material_index = callbacks.material_added();
          _material_mapping.build(scene_rep.material_mapping());
          _material_mapping_hash = hash_mapping(scene_rep.material_mapping());
          reset_selection();
          const auto material_position = _material_mapping.reverse.find(material_index);
          if (material_position != _material_mapping.reverse.end()) {
            set_single_material_selection(static_cast<int32_t>(material_position->second), true);
          }
        }
        ImGui::Spacing();
        if (ImGui::BeginListBox("##materials_list", ImVec2(-FLT_MIN, -FLT_MIN))) {
          uint32_t visible_count = 0u;
          for (uint64_t i = 0u; i < _material_mapping.size(); ++i) {
            const int32_t list_index = static_cast<int32_t>(i);
            const auto& entry = _material_mapping.entry(list_index);
            if (contains_case_insensitive(entry.name, _resource_filter) == false) {
              continue;
            }
            ++visible_count;
            ImGui::PushID(list_index);
            const bool material_selected = (_selection.kind == SelectionKind::Material) && material_list_position_selected(list_index);
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
          if (visible_count == 0u) {
            ImGui::TextDisabled("No matching materials");
          }
          ImGui::EndListBox();
        }
        ImGui::EndTabItem();
      }

      const std::string mediums_label = format_string("Mediums (%zu)###resource_mediums", static_cast<size_t>(_medium_mapping.size()));
      if (ImGui::BeginTabItem(mediums_label.c_str())) {
        if (ImGui::Button("Add Medium", ImVec2(-FLT_MIN, 0.0f)) && callbacks.medium_added) {
          const uint32_t medium_index = callbacks.medium_added();
          _medium_mapping.build(scene_rep.medium_mapping());
          _medium_mapping_hash = hash_mapping(scene_rep.medium_mapping());
          reset_selection();
          const auto medium_position = _medium_mapping.reverse.find(medium_index);
          if (medium_position != _medium_mapping.reverse.end()) {
            set_selection(SelectionKind::Medium, static_cast<int32_t>(medium_position->second), true);
          }
        }
        ImGui::Spacing();
        if (ImGui::BeginListBox("##mediums_list", ImVec2(-FLT_MIN, -FLT_MIN))) {
          uint32_t visible_count = 0u;
          for (uint64_t i = 0u; i < _medium_mapping.size(); ++i) {
            const int32_t list_index = static_cast<int32_t>(i);
            const auto& entry = _medium_mapping.entry(list_index);
            if (contains_case_insensitive(entry.name, _resource_filter) == false) {
              continue;
            }
            ++visible_count;
            ImGui::PushID(static_cast<int>(i + 1024u));
            const bool medium_selected = (_selection.kind == SelectionKind::Medium) && (_selection.index == list_index);
            if (ImGui::Selectable(entry.name, medium_selected)) {
              set_selection(SelectionKind::Medium, list_index);
            }
            ImGui::PopID();
          }
          if (visible_count == 0u) {
            ImGui::TextDisabled("No matching mediums");
          }
          ImGui::EndListBox();
        }
        ImGui::EndTabItem();
      }

      const std::string emitters_label = format_string("Lights (%zu)###resource_lights", scene_rep.data().emitter_profiles.size());
      if (ImGui::BeginTabItem(emitters_label.c_str())) {
        if (ImGui::Button("Add Light", ImVec2(-FLT_MIN, 0.0f))) {
          ImGui::OpenPopup("##add_light_popup");
        }
        if (ImGui::BeginPopup("##add_light_popup")) {
          constexpr const char* light_names[] = {"Environment", "Directional", "Atmosphere"};
          for (uint32_t light_type = 0u; light_type < IM_ARRAYSIZE(light_names); ++light_type) {
            if (ImGui::Selectable(light_names[light_type])) {
              if (callbacks.emitter_added) {
                callbacks.emitter_added(light_type);
              }
              ImGui::CloseCurrentPopup();
            }
          }
          ImGui::EndPopup();
        }
        ImGui::Spacing();
        if (ImGui::BeginListBox("##emitters_list", ImVec2(-FLT_MIN, -FLT_MIN))) {
          uint32_t visible_count = 0u;
          for (uint32_t emitter_index = 0u; emitter_index < scene_rep.data().emitter_profiles.size(); ++emitter_index) {
            const auto& emitter = scene_rep.data().emitter_profiles[emitter_index];
            std::string label;
            switch (emitter.cls) {
              case EmitterProfile::Class::Area:
                label = format_string("Area light %u", emitter_index + 1u);
                break;
              case EmitterProfile::Class::Directional:
                label = ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0u) ? format_string("Sun %u", emitter_index + 1u)
                                                                                  : format_string("Directional light %u", emitter_index + 1u);
                break;
              case EmitterProfile::Class::Environment:
                label =
                  ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0u) ? format_string("Sky %u", emitter_index + 1u) : format_string("Environment %u", emitter_index + 1u);
                break;
              default:
                label = format_string("Light %u", emitter_index + 1u);
                break;
            }
            if (contains_case_insensitive(label.c_str(), _resource_filter) == false) {
              continue;
            }
            ++visible_count;
            ImGui::PushID(static_cast<int>(emitter_index + 4096u));
            const bool emitter_selected = (_selection.kind == SelectionKind::Emitter) && (_selection.index == static_cast<int32_t>(emitter_index));
            if (ImGui::Selectable(label.c_str(), emitter_selected)) {
              set_selection(SelectionKind::Emitter, static_cast<int32_t>(emitter_index));
            }
            ImGui::PopID();
          }
          if (visible_count == 0u) {
            ImGui::TextDisabled("No matching lights");
          }
          ImGui::EndListBox();
        }
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
  });
}

void UI::build_scene_tree_window(SceneRepresentation& scene_rep, const BuildContext& ctx) {
  ctx.with_window(UIObjects, "Scene Tree", [&]() {
    SceneHierarchy& hierarchy = scene_rep.data().hierarchy;
    if (hierarchy.nodes.empty() || hierarchy.evaluation_order.empty()) {
      ImGui::TextDisabled("No scene nodes");
      return;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 0.75f * ImGui::GetFontSize());
    if (ImGui::BeginChild("##scene_tree", ImVec2(-FLT_MIN, -FLT_MIN), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar)) {
      const ImGuiTreeNodeFlags scene_flags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
      const bool scene_open = ImGui::TreeNodeEx("##scene_root", scene_flags, "Scene (%zu)", hierarchy.nodes.size());
      if (scene_open) {
        _scene_tree_open_subtree_ends.clear();
        auto& open_subtree_ends = _scene_tree_open_subtree_ends;
        uint32_t order_position = 0u;
        while (order_position < hierarchy.evaluation_order.size()) {
          while (open_subtree_ends.empty() == false && open_subtree_ends.back() <= order_position) {
            ImGui::TreePop();
            open_subtree_ends.pop_back();
          }

          const uint32_t node_index = hierarchy.evaluation_order[order_position];
          if (node_index >= hierarchy.nodes.size()) {
            ++order_position;
            continue;
          }
          const SceneNode& node = hierarchy.nodes[node_index];

          uint32_t attached_camera_index = kInvalidIndex;
          bool contains_active_camera = false;
          bool has_camera_attachment = false;
          bool has_emitter_attachment = false;
          const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
          if ((attachment_end >= node.attachment_offset) && (attachment_end <= hierarchy.attachments.size())) {
            for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
              const SceneAttachment& attachment = hierarchy.attachments[attachment_index];
              if (attachment.type == SceneAttachment::Type::Emitter) {
                has_emitter_attachment = true;
              }
              if (attachment.type == SceneAttachment::Type::Camera) {
                has_camera_attachment = true;
              }
              if ((attachment.type != SceneAttachment::Type::Camera) || (attachment.resource_index >= scene_rep.data().cameras.size())) {
                continue;
              }
              if (attached_camera_index == kInvalidIndex) {
                attached_camera_index = attachment.resource_index;
              }
              contains_active_camera = contains_active_camera || scene_rep.data().cameras[attachment.resource_index].active;
            }
          }

          const uint32_t subtree_end = (node_index < hierarchy.subtree_end_position.size())
                                         ? std::min<uint32_t>(hierarchy.subtree_end_position[node_index], static_cast<uint32_t>(hierarchy.evaluation_order.size()))
                                         : (order_position + 1u);
          const bool leaf = subtree_end <= (order_position + 1u);
          ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
          if (node.parent_index == kInvalidIndex) {
            flags |= ImGuiTreeNodeFlags_DefaultOpen;
          }
          if ((_selection.kind == SelectionKind::Node) && (_selection.index == static_cast<int32_t>(node_index))) {
            flags |= ImGuiTreeNodeFlags_Selected;
          }
          if (leaf) {
            flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
          }

          const bool enabled = (node_index < hierarchy.effective_enabled.size()) && (hierarchy.effective_enabled[node_index] != 0u);
          if (enabled == false) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
          } else if (contains_active_camera) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_CheckMark));
          }
          const char* node_name = (node_index < hierarchy.node_names.size() && hierarchy.node_names[node_index].empty() == false) ? hierarchy.node_names[node_index].c_str()
                                                                                                                                  : format_string("Node %u", node_index);
          std::string node_label;
          if (has_camera_attachment && has_emitter_attachment) {
            node_label = "Camera + Light · ";
          } else if (has_camera_attachment) {
            node_label = "Camera · ";
          } else if (has_emitter_attachment) {
            node_label = "Light · ";
          }
          node_label += node_name;
          const bool open = ImGui::TreeNodeEx(reinterpret_cast<void*>(static_cast<uintptr_t>(node_index) + 1u), flags, "%s", node_label.c_str());
          if (ImGui::IsItemClicked()) {
            set_selection(SelectionKind::Node, static_cast<int32_t>(node_index));
          }
          if (enabled && ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0) && (attached_camera_index != kInvalidIndex) && callbacks.camera_activated) {
            callbacks.camera_activated(attached_camera_index);
          }
          if ((enabled == false) || contains_active_camera) {
            ImGui::PopStyleColor();
          }

          if (leaf == false && open) {
            open_subtree_ends.push_back(subtree_end);
            ++order_position;
          } else if (leaf == false) {
            order_position = std::max(order_position + 1u, subtree_end);
          } else {
            ++order_position;
          }
        }

        while (open_subtree_ends.empty() == false) {
          ImGui::TreePop();
          open_subtree_ends.pop_back();
        }
        ImGui::TreePop();
      }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
  });
}

void UI::build_properties_window(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  if ((_ui_setup & UIProperties) == 0)
    return;

  std::string properties_title = "Nothing selected";

  switch (_selection.kind) {
    case SelectionKind::Node:
      if ((_selection.index >= 0) && (static_cast<uint64_t>(_selection.index) < scene_rep.data().hierarchy.node_names.size())) {
        const std::string& node_name = scene_rep.data().hierarchy.node_names[_selection.index];
        properties_title = node_name.empty() ? format_string("Node %u", static_cast<uint32_t>(_selection.index)) : node_name;
      }
      break;
    case SelectionKind::Material:
      if (selected_material_count() > 1u) {
        properties_title = format_string("%u Materials", selected_material_count());
      } else if ((_selection.index >= 0) && (static_cast<uint64_t>(_selection.index) < _material_mapping.size())) {
        properties_title = _material_mapping.name(_selection.index);
      }
      break;
    case SelectionKind::Medium:
      if ((_selection.index >= 0) && (static_cast<uint64_t>(_selection.index) < _medium_mapping.size())) {
        properties_title = _medium_mapping.name(_selection.index);
      }
      break;
    case SelectionKind::Emitter: {
      if ((_selection.index >= 0) && (static_cast<uint32_t>(_selection.index) < scene_rep.data().emitter_profiles.size())) {
        uint32_t emitter_index = static_cast<uint32_t>(_selection.index);
        const auto& emitter = scene_rep.data().emitter_profiles[emitter_index];
        const char* emitter_type = nullptr;
        switch (emitter.cls) {
          case EmitterProfile::Class::Directional:
            emitter_type = ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0u) ? "Sun" : "Directional Light";
            break;
          case EmitterProfile::Class::Environment:
            emitter_type = ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0u) ? "Sky" : "Environment";
            break;
          case EmitterProfile::Class::Area:
            emitter_type = "Area Light";
            break;
          default:
            emitter_type = "Emitter";
            break;
        }
        properties_title = emitter_type;
      } else {
        properties_title = "Emitter";
      }
      break;
    }
    case SelectionKind::Rendering:
      properties_title = "Rendering";
      break;
    default:
      break;
  }

  std::string properties_window_name = properties_title + "###properties";
  ctx.with_window(UIProperties, properties_window_name.c_str(), [&]() {
    const auto history_button = [&](const char* label, bool enabled, int32_t step, const char* tooltip) {
      if (enabled == false) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Button(label)) {
        navigate_history(step);
      }
      if (enabled == false) {
        ImGui::EndDisabled();
      }
      draw_item_tooltip(tooltip, ImGuiHoveredFlags_DelayNormal);
    };
    history_button("Back##selection_history", can_navigate_back(), -1, "Return to the previous selection.");
    ImGui::SameLine();
    history_button("Forward##selection_history", can_navigate_forward(), 1, "Advance to the next selection.");
    ImGui::Spacing();
    ImGui::SeparatorText(properties_title.c_str());
    switch (_selection.kind) {
      case SelectionKind::Node: {
        build_node_selection_properties(scene_rep, ctx, data);
        break;
      }
      case SelectionKind::Material: {
        build_material_selection_properties(scene_rep, ctx, data);
        break;
      }
      case SelectionKind::Medium: {
        build_medium_selection_properties(scene_rep, ctx, data);
        break;
      }
      case SelectionKind::Emitter: {
        build_emitter_selection_properties(scene_rep, ctx, data);
        break;
      }
      case SelectionKind::Rendering: {
        ImGui::TextDisabled("Rendering controls are available in the Render tab.");
        break;
      }
      default:
        ImGui::TextDisabled("Select a scene node or resource to inspect it.");
        break;
    }
  });
}

void UI::build_node_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  SceneHierarchy& hierarchy = scene_rep.data().hierarchy;
  if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= hierarchy.nodes.size())) {
    ImGui::Text("Invalid node selection");
    return;
  }

  const uint32_t node_index = static_cast<uint32_t>(_selection.index);
  if (_node_transform_editor_interaction_active && (_node_transform_editor_interaction_node_index != static_cast<int32_t>(node_index))) {
    finish_node_transform_editor_interaction();
  }
  if (_node_transform_editor_interaction_active) {
    _node_transform_editor_interaction_rendered_this_frame = true;
  }
  SceneNode& node = hierarchy.nodes[node_index];
  const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
  const bool attachment_range_valid = (attachment_end >= node.attachment_offset) && (attachment_end <= hierarchy.attachments.size());
  if (_node_geometry_edit_result_node != static_cast<int32_t>(node_index)) {
    _node_geometry_edit_result_node = static_cast<int32_t>(node_index);
    _node_geometry_edit_result = NodeGeometryEditResult::Success;
  }
  auto subtree_contains_active_camera = [&]() {
    if ((node_index >= hierarchy.order_position.size()) || (node_index >= hierarchy.subtree_end_position.size())) {
      return false;
    }
    const uint32_t subtree_begin = hierarchy.order_position[node_index];
    const uint32_t subtree_end = std::min<uint32_t>(hierarchy.subtree_end_position[node_index], static_cast<uint32_t>(hierarchy.evaluation_order.size()));
    if (subtree_begin >= subtree_end) {
      return false;
    }
    for (uint32_t order_position = subtree_begin; order_position < subtree_end; ++order_position) {
      const uint32_t descendant_index = hierarchy.evaluation_order[order_position];
      if (descendant_index >= hierarchy.nodes.size()) {
        continue;
      }
      const SceneNode& descendant = hierarchy.nodes[descendant_index];
      const uint32_t attachment_end = descendant.attachment_offset + descendant.attachment_count;
      if ((attachment_end < descendant.attachment_offset) || (attachment_end > hierarchy.attachments.size())) {
        continue;
      }
      for (uint32_t attachment_index = descendant.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
        const SceneAttachment& attachment = hierarchy.attachments[attachment_index];
        if ((attachment.type == SceneAttachment::Type::Camera) && (attachment.resource_index < scene_rep.data().cameras.size()) &&
            scene_rep.data().cameras[attachment.resource_index].active) {
          return true;
        }
      }
    }
    return false;
  };
  auto resolve_node_change = [&](bool transforms_only) {
    if (scene_rep.data().resolve_hierarchy() == false) {
      return false;
    }
    scene_rep.update_active_camera();
    if (transforms_only && callbacks.scene_transforms_changed) {
      callbacks.scene_transforms_changed();
    } else {
      scene_rep.update_medium_bounds();
      if (callbacks.scene_settings_changed) {
        callbacks.scene_settings_changed();
      }
    }
    return true;
  };
  const char* selected_node_name = (node_index < hierarchy.node_names.size()) ? hierarchy.node_names[node_index].c_str() : "";
  update_name_buffer(SelectionKind::Node, _selection.index, selected_node_name);
  ImGui::TextUnformatted("Name");
  full_width_item();
  const bool name_edit_active = ImGui::InputText("##node_name", _name_edit_buffer, sizeof(_name_edit_buffer), ImGuiInputTextFlags_AutoSelectAll);
  if (ImGui::IsItemDeactivatedAfterEdit() || (name_edit_active && ImGui::IsKeyPressed(ImGuiKey_Enter))) {
    if (hierarchy.node_names.size() < hierarchy.nodes.size()) {
      hierarchy.node_names.resize(hierarchy.nodes.size());
    }
    hierarchy.node_names[node_index] = _name_edit_buffer;
    if (callbacks.scene_modified) {
      callbacks.scene_modified();
    }
  }

  bool enabled = (node.flags & SceneNode::Enabled) != 0u;
  const bool protects_active_camera = enabled && subtree_contains_active_camera();
  if (protects_active_camera) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Checkbox("Enabled", &enabled)) {
    if (hierarchy.set_enabled(node_index, enabled)) {
      if (resolve_node_change(true) == false) {
        hierarchy.set_enabled(node_index, enabled == false);
        if (scene_rep.data().resolve_hierarchy() == false) {
          log::error("Failed to restore scene hierarchy after rejecting an invalid node visibility edit");
        }
      }
    }
  }
  if (protects_active_camera) {
    ImGui::EndDisabled();
    draw_item_tooltip("Activate another camera before disabling this node.", ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_AllowWhenDisabled);
  }

  const char* parent_name = "None";
  if ((node.parent_index != kInvalidIndex) && (node.parent_index < hierarchy.node_names.size())) {
    parent_name = hierarchy.node_names[node.parent_index].c_str();
  }
  ImGui::TextUnformatted("Parent");
  full_width_item();
  const bool parent_combo_open = ImGui::BeginCombo("##node_parent", parent_name);
  draw_item_tooltip("Reparenting preserves the world transform.", ImGuiHoveredFlags_DelayNormal);
  if (parent_combo_open) {
    bool parent_changed = false;
    if (ImGui::Selectable("None", node.parent_index == kInvalidIndex)) {
      if ((node.parent_index != kInvalidIndex) && hierarchy.reparent_preserve_world(node_index, kInvalidIndex)) {
        parent_changed = resolve_node_change(false);
      }
    }
    const bool traversal_valid = (node_index < hierarchy.order_position.size()) && (node_index < hierarchy.subtree_end_position.size());
    const uint32_t subtree_begin = traversal_valid ? hierarchy.order_position[node_index] : kInvalidIndex;
    const uint32_t subtree_end = traversal_valid ? hierarchy.subtree_end_position[node_index] : kInvalidIndex;
    for (uint32_t candidate_index = 0u; (candidate_index < hierarchy.nodes.size()) && (parent_changed == false); ++candidate_index) {
      ImGui::PushID(static_cast<int>(candidate_index));
      const uint32_t candidate_position = candidate_index < hierarchy.order_position.size() ? hierarchy.order_position[candidate_index] : kInvalidIndex;
      const bool creates_cycle = (candidate_index == node_index) ||
                                 (traversal_valid && (candidate_position != kInvalidIndex) && (candidate_position >= subtree_begin) && (candidate_position < subtree_end));
      AffineTransform inverse_parent = {};
      double parent_determinant = 0.0;
      const bool singular_parent =
        (candidate_index >= hierarchy.world_transforms.size()) || (invert_affine(hierarchy.world_transforms[candidate_index], inverse_parent, parent_determinant) == false);
      const bool invalid_parent = creates_cycle || singular_parent;
      if (invalid_parent) {
        ImGui::BeginDisabled();
      }
      const char* candidate_name = candidate_index < hierarchy.node_names.size() ? hierarchy.node_names[candidate_index].c_str() : "Unnamed node";
      if (ImGui::Selectable(candidate_name, node.parent_index == candidate_index) && (invalid_parent == false)) {
        if ((node.parent_index != candidate_index) && hierarchy.reparent_preserve_world(node_index, candidate_index)) {
          parent_changed = resolve_node_change(false);
        }
      }
      if (invalid_parent) {
        ImGui::EndDisabled();
      }
      ImGui::PopID();
    }
    ImGui::EndCombo();
  }

  ImGui::Spacing();
  ImGui::TextUnformatted("Local Transform");
  const float gizmo_spacing = ImGui::GetStyle().ItemSpacing.x;
  const float gizmo_button_width = (ImGui::GetContentRegionAvail().x - 3.0f * gizmo_spacing) / 4.0f;
  auto gizmo_button = [&](const char* label, bool selected) {
    if (selected) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    const bool pressed = ImGui::Button(label, ImVec2(gizmo_button_width, 0.0f));
    if (selected) {
      ImGui::PopStyleColor();
    }
    return pressed;
  };
  if (gizmo_button("Move##gizmo_move", _gizmo_operation == GizmoOperation::Translate)) {
    _gizmo_operation = GizmoOperation::Translate;
  }
  ImGui::SameLine();
  if (gizmo_button("Rotate##gizmo_rotate", _gizmo_operation == GizmoOperation::Rotate)) {
    _gizmo_operation = GizmoOperation::Rotate;
  }
  ImGui::SameLine();
  if (gizmo_button("Scale##gizmo_scale", _gizmo_operation == GizmoOperation::Scale)) {
    _gizmo_operation = GizmoOperation::Scale;
  }
  ImGui::SameLine();
  const bool local_gizmo = _gizmo_mode == GizmoMode::Local;
  const bool scale_gizmo = _gizmo_operation == GizmoOperation::Scale;
  const char* gizmo_mode_label = scale_gizmo ? "Local##gizmo_mode" : (local_gizmo ? "Local##gizmo_mode" : "World##gizmo_mode");
  if (scale_gizmo) {
    ImGui::BeginDisabled();
  }
  if (ImGui::Button(gizmo_mode_label, ImVec2(gizmo_button_width, 0.0f)) && (scale_gizmo == false)) {
    _gizmo_mode = local_gizmo ? GizmoMode::World : GizmoMode::Local;
  }
  if (scale_gizmo) {
    ImGui::EndDisabled();
  }
  auto refresh_transform_editor = [&]() {
    _node_transform_editor.node_index = static_cast<int32_t>(node_index);
    _node_transform_editor.source_transform = node.local_transform;
    _node_transform_editor.trs = {};
    _node_transform_editor.decomposable = affine_to_trs(node.local_transform, _node_transform_editor.trs);
    _node_transform_editor.rotation_degrees = _node_transform_editor.trs.rotation_radians * (180.0f / kPi);
  };
  if ((_node_transform_editor.node_index != static_cast<int32_t>(node_index)) ||
      (std::memcmp(&_node_transform_editor.source_transform, &node.local_transform, sizeof(AffineTransform)) != 0)) {
    refresh_transform_editor();
  }

  bool transform_changed = false;
  bool transform_interaction_started = false;
  bool transform_interaction_finished = false;
  auto collect_transform_interaction = [&]() {
    transform_interaction_started = transform_interaction_started || ImGui::IsItemActivated();
    transform_interaction_finished = transform_interaction_finished || ImGui::IsItemDeactivated();
  };
  ImGui::TextUnformatted("Position");
  full_width_item();
  transform_changed = ImGui::DragFloat3("##node_position", &_node_transform_editor.trs.translation.x, 0.01f, 0.0f, 0.0f, "%.3f") || transform_changed;
  collect_transform_interaction();
  if (_node_transform_editor.decomposable) {
    ImGui::TextUnformatted("Rotation");
    full_width_item();
    transform_changed = ImGui::DragFloat3("##node_rotation", &_node_transform_editor.rotation_degrees.x, 0.25f, 0.0f, 0.0f, "%.2f°") || transform_changed;
    collect_transform_interaction();
    ImGui::TextUnformatted("Scale");
    full_width_item();
    transform_changed = ImGui::DragFloat3("##node_scale", &_node_transform_editor.trs.scale.x, 0.01f, 0.0f, 0.0f, "%.3f") || transform_changed;
    collect_transform_interaction();
    _node_transform_editor.trs.rotation_radians = _node_transform_editor.rotation_degrees * (kPi / 180.0f);
  } else {
    ImGui::TextWrapped("Transform contains shear or zero scale. Reset it to edit rotation and scale.");
    if (ImGui::Button("Reset to TRS")) {
      _node_transform_editor.trs.rotation_radians = {};
      _node_transform_editor.rotation_degrees = {};
      _node_transform_editor.trs.scale = {1.0f, 1.0f, 1.0f};
      _node_transform_editor.decomposable = true;
      transform_changed = true;
    }
  }
  if (transform_interaction_started && (_node_transform_editor_interaction_active == false)) {
    _node_transform_editor_interaction_active = true;
    _node_transform_editor_interaction_rendered_this_frame = true;
    _node_transform_editor_interaction_node_index = static_cast<int32_t>(node_index);
    if (callbacks.scene_transform_interaction_started) {
      callbacks.scene_transform_interaction_started();
    }
  }
  if (transform_changed) {
    AffineTransform transform = _node_transform_editor.decomposable ? affine_from_trs(_node_transform_editor.trs) : _node_transform_editor.source_transform;
    if (_node_transform_editor.decomposable == false) {
      transform.rows[0].w = _node_transform_editor.trs.translation.x;
      transform.rows[1].w = _node_transform_editor.trs.translation.y;
      transform.rows[2].w = _node_transform_editor.trs.translation.z;
    }
    if (hierarchy.set_local_transform(node_index, transform)) {
      _node_transform_editor.source_transform = transform;
      resolve_node_change(true);
    } else {
      refresh_transform_editor();
    }
  }
  if (transform_interaction_finished && _node_transform_editor_interaction_active) {
    finish_node_transform_editor_interaction();
  }

  const NodeGeometryEditResult bake_status = scene_rep.validate_node_geometry_edit(node_index, NodeGeometryOperation::BakeLocalTransform);
  const NodeGeometryEditResult center_status = scene_rep.validate_node_geometry_edit(node_index, NodeGeometryOperation::CenterPivot);
  const float geometry_action_spacing = ImGui::GetStyle().ItemSpacing.x;
  const float geometry_action_width = (ImGui::GetContentRegionAvail().x - geometry_action_spacing) * 0.5f;
  auto geometry_action = [&](const char* label, NodeGeometryOperation operation, NodeGeometryEditResult status, const char* description) {
    const bool available = status == NodeGeometryEditResult::Success;
    if (available == false) {
      ImGui::BeginDisabled();
    }
    const bool pressed = ImGui::Button(label, ImVec2(geometry_action_width, 0.0f));
    if (available == false) {
      ImGui::EndDisabled();
      draw_item_tooltip(node_geometry_edit_result_message(status), ImGuiHoveredFlags_DelayNormal | ImGuiHoveredFlags_AllowWhenDisabled);
    } else {
      draw_item_tooltip(description, ImGuiHoveredFlags_DelayNormal);
    }
    if (pressed == false) {
      return;
    }

    _node_geometry_edit_result = scene_rep.edit_node_geometry(node_index, operation);
    if (_node_geometry_edit_result == NodeGeometryEditResult::Success) {
      scene_rep.update_medium_bounds();
      scene_rep.update_active_camera();
      refresh_transform_editor();
      if (callbacks.scene_settings_changed) {
        callbacks.scene_settings_changed();
      }
    } else {
      log::error("Node geometry edit failed: %s", node_geometry_edit_result_message(_node_geometry_edit_result));
    }
  };
  geometry_action("Bake Transform", NodeGeometryOperation::BakeLocalTransform, bake_status, "Apply the local transform to private mesh geometry and reset it to identity.");
  ImGui::SameLine();
  geometry_action("Center Pivot", NodeGeometryOperation::CenterPivot, center_status, "Move the pivot to the area-weighted center of the attached triangles.");
  if (_node_geometry_edit_result != NodeGeometryEditResult::Success) {
    ImGui::TextColored(kErrorTextColor, "%s", node_geometry_edit_result_message(_node_geometry_edit_result));
  }

  AffineTransform inverse_transform = {};
  double determinant = 0.0;
  if (invert_affine(node.local_transform, inverse_transform, determinant) == false) {
    ImGui::TextColored(kErrorTextColor, "Singular transform; attachments disabled.");
  }

  ImGui::Spacing();
  ImGui::SeparatorText("Attachments");
  if (node.attachment_count == 0u) {
    ImGui::TextDisabled("None");
  }
  static constexpr const char* kAttachmentNames[] = {"Mesh", "Camera", "Emitter", "Medium"};
  if (attachment_range_valid == false) {
    ImGui::TextColored(kErrorTextColor, "Invalid attachment range");
    return;
  }
  for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
    const SceneAttachment& attachment = hierarchy.attachments[attachment_index];
    const uint32_t type_index = static_cast<uint32_t>(attachment.type);
    const char* type_name = type_index < std::size(kAttachmentNames) ? kAttachmentNames[type_index] : "Unknown";
    ImGui::PushID(static_cast<int>(attachment_index));
    ImGui::SeparatorText(type_name);
    switch (attachment.type) {
      case SceneAttachment::Type::Mesh: {
        if (attachment.resource_index < scene_rep.data().meshes.size()) {
          ImGui::Text("Triangles: %u", scene_rep.data().meshes[attachment.resource_index].triangle_count);
        } else {
          ImGui::TextColored(kErrorTextColor, "Invalid mesh");
        }
        break;
      }
      case SceneAttachment::Type::Camera:
        if (attachment.resource_index < scene_rep.data().cameras.size()) {
          Camera& camera = scene_rep.data().cameras[attachment.resource_index].cam;
          const bool attachment_enabled = (node_index < hierarchy.effective_enabled.size()) && (hierarchy.effective_enabled[node_index] != 0u);
          build_camera_selection_properties(scene_rep, camera, attachment.resource_index, attachment_enabled, data);
        } else {
          ImGui::TextColored(kErrorTextColor, "Invalid camera");
        }
        break;
      case SceneAttachment::Type::Emitter:
        build_emitter_resource_properties(scene_rep, attachment.resource_index, data, false);
        break;
      case SceneAttachment::Type::Medium:
        build_medium_resource_properties(scene_rep, attachment.resource_index);
        break;
      default:
        ImGui::TextColored(kErrorTextColor, "Unsupported attachment");
        break;
    }
    ImGui::PopID();
  }

  build_node_appearance_properties(scene_rep, node, attachment_end, data);
}

void UI::finish_node_transform_editor_interaction() {
  if (_node_transform_editor_interaction_active == false) {
    return;
  }

  _node_transform_editor_interaction_active = false;
  _node_transform_editor_interaction_rendered_this_frame = false;
  _node_transform_editor_interaction_node_index = -1;
  if (callbacks.scene_transform_interaction_finished) {
    callbacks.scene_transform_interaction_finished();
  }
}

void UI::build_transform_gizmo(SceneRepresentation& scene_rep, const FrameData& data) {
  _gizmo_captures_mouse = false;
  auto notify_transform_change = [&]() {
    if (callbacks.scene_transforms_changed) {
      callbacks.scene_transforms_changed();
    } else {
      scene_rep.update_medium_bounds();
      if (callbacks.scene_settings_changed) {
        callbacks.scene_settings_changed();
      }
    }
  };
  auto finish_interaction = [&]() {
    if (_gizmo_was_using && callbacks.scene_transform_interaction_finished) {
      callbacks.scene_transform_interaction_finished();
    }
    _gizmo_was_using = false;
  };
  if ((_selection.kind != SelectionKind::Node) || (_selection.index < 0)) {
    finish_interaction();
    return;
  }

  SceneHierarchy& hierarchy = scene_rep.data().hierarchy;
  const uint32_t node_index = static_cast<uint32_t>(_selection.index);
  if ((node_index >= hierarchy.nodes.size()) || (node_index >= hierarchy.world_transforms.size())) {
    finish_interaction();
    return;
  }

  const Camera& camera = scene_rep.camera();
  if (camera.cls != Camera::Class::Perspective) {
    finish_interaction();
    return;
  }

  uint2 output_size = data.output_size;
  if ((output_size.x == 0u) || (output_size.y == 0u)) {
    output_size = camera.film_size;
  }
  if ((output_size.x == 0u) || (output_size.y == 0u)) {
    finish_interaction();
    return;
  }

  AffineTransform inverse_parent = {};
  const SceneNode& node = hierarchy.nodes[node_index];
  if (node.parent_index != kInvalidIndex) {
    double determinant = 0.0;
    if ((node.parent_index >= hierarchy.world_transforms.size()) || (invert_affine(hierarchy.world_transforms[node.parent_index], inverse_parent, determinant) == false)) {
      finish_interaction();
      return;
    }
  }

  if (_viewport_geometry.valid == false) {
    finish_interaction();
    return;
  }
  const ImVec2 rect_position(_viewport_geometry.logical_position.x, _viewport_geometry.logical_position.y);
  const ImVec2 rect_size(_viewport_geometry.logical_size.x, _viewport_geometry.logical_size.y);

  const uint2 projection_size = ((camera.film_size.x > 0u) && (camera.film_size.y > 0u)) ? camera.film_size : output_size;
  float4x4 view = look_at(camera.position, camera_target(camera), camera.up);
  float4x4 projection = perspective(get_camera_fov(camera) * kPi / 180.0f, projection_size.x, projection_size.y, camera.clip_near, camera.clip_far);
  projection.col[0].x *= _viewport_geometry.image_size.x / std::max(1.0f, _viewport_geometry.logical_size.x);
  projection.col[1].y *= _viewport_geometry.image_size.y / std::max(1.0f, _viewport_geometry.logical_size.y);
  // The resolved node translation is its pivot. Attachments must not shift the
  // gizmo to a mesh bounding-box center or another resource-specific origin.
  float4x4 world_matrix = matrix_from_affine(hierarchy.world_transforms[node_index]);

  ImGuizmo::SetRect(rect_position.x, rect_position.y, rect_size.x, rect_size.y);
  ImGuizmo::SetOrthographic(false);
  ImGuizmo::SetGizmoSizeClipSpace(0.18f);
  ImGuizmo::Style& gizmo_style = ImGuizmo::GetStyle();
  gizmo_style.TranslationLineThickness = 1.5f;
  gizmo_style.TranslationLineArrowSize = 5.0f;
  gizmo_style.RotationLineThickness = 1.5f;
  gizmo_style.RotationOuterLineThickness = 2.0f;
  gizmo_style.ScaleLineThickness = 1.5f;
  gizmo_style.ScaleLineCircleSize = 5.0f;
  gizmo_style.HatchedAxisLineThickness = 3.0f;
  gizmo_style.CenterCircleSize = 3.0f;
  ImGuizmo::PushID(static_cast<int>(node_index));

  ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
  switch (_gizmo_operation) {
    case GizmoOperation::Rotate:
      operation = ImGuizmo::ROTATE;
      break;
    case GizmoOperation::Scale:
      operation = ImGuizmo::SCALE;
      break;
    case GizmoOperation::Translate:
      break;
  }
  const ImGuizmo::MODE mode = (_gizmo_mode == GizmoMode::Local) ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
  const bool transform_changed = ImGuizmo::Manipulate(&view.col[0].x, &projection.col[0].x, operation, mode, &world_matrix.col[0].x);
  const bool gizmo_using = ImGuizmo::IsUsing();
  const bool interaction_started = (_gizmo_was_using == false) && gizmo_using;
  const bool interaction_finished = _gizmo_was_using && (gizmo_using == false);
  _gizmo_captures_mouse = ImGuizmo::IsOver() || gizmo_using;
  _gizmo_was_using = gizmo_using;

  ImGuizmo::PopID();

  if (interaction_started && callbacks.scene_transform_interaction_started) {
    callbacks.scene_transform_interaction_started();
  }

  if (transform_changed) {
    const AffineTransform previous_local_transform = node.local_transform;
    const AffineTransform edited_world_transform = affine_from_matrix(world_matrix);
    const AffineTransform edited_local_transform = (node.parent_index == kInvalidIndex) ? edited_world_transform : multiply_affine(inverse_parent, edited_world_transform);
    if (hierarchy.set_local_transform(node_index, edited_local_transform) && scene_rep.data().resolve_hierarchy()) {
      scene_rep.update_active_camera();
      notify_transform_change();
    } else {
      hierarchy.set_local_transform(node_index, previous_local_transform);
      if (scene_rep.data().resolve_hierarchy() == false) {
        log::error("Failed to restore scene hierarchy after rejecting an invalid gizmo transform");
      }
    }
  }

  if (interaction_finished) {
    if (callbacks.scene_transform_interaction_finished) {
      callbacks.scene_transform_interaction_finished();
    }
  }
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

    ImGui::TextUnformatted("Name");
    full_width_item();
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
      queue_material_change(material_index);
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
  for (const uint32_t changed_material_index : material_indices) {
    queue_material_change(changed_material_index);
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
  const char* medium_name = _medium_mapping.name(_selection.index);
  update_name_buffer(SelectionKind::Medium, _selection.index, medium_name);
  ImGui::TextUnformatted("Name");
  full_width_item();
  bool name_edit_active = ImGui::InputText("##medium_name", _name_edit_buffer, sizeof(_name_edit_buffer), ImGuiInputTextFlags_AutoSelectAll);
  bool name_commit = ImGui::IsItemDeactivatedAfterEdit() || (name_edit_active && ImGui::IsKeyPressed(ImGuiKey_Enter));
  if (name_commit && callbacks.medium_renamed) {
    callbacks.medium_renamed(medium_index, std::string(_name_edit_buffer));
    _pending_selection = {SelectionKind::Medium, medium_index, true};
  }
  build_medium_resource_properties(scene_rep, medium_index);
}

void UI::build_medium_resource_properties(SceneRepresentation& scene_rep, uint32_t medium_index) {
  if (medium_index >= scene_rep.data().mediums_vector.size()) {
    ImGui::TextColored(kErrorTextColor, "Invalid medium");
    return;
  }
  _medium_editor_rendered_this_frame = true;
  Medium& medium = scene_rep.data().mediums.get(medium_index);
  Medium edited_medium = medium;
  SpectralDistribution* absorption = nullptr;
  SpectralDistribution edited_absorption = {};
  if (medium.absorption_index < scene_rep.data().spectrum_values.size()) {
    edited_absorption = scene_rep.data().spectrum_values[medium.absorption_index];
    absorption = &edited_absorption;
  }
  SpectralDistribution* scattering = nullptr;
  SpectralDistribution edited_scattering = {};
  if (medium.scattering_index < scene_rep.data().spectrum_values.size()) {
    edited_scattering = scene_rep.data().spectrum_values[medium.scattering_index];
    scattering = &edited_scattering;
  }
  const bool changed = build_medium(edited_medium, absorption, scattering);
  if (changed) {
    queue_medium_change(medium_index);
    medium = edited_medium;
    if (absorption != nullptr) {
      scene_rep.data().spectrum_values[medium.absorption_index] = edited_absorption;
    }
    if (scattering != nullptr) {
      scene_rep.data().spectrum_values[medium.scattering_index] = edited_scattering;
    }
  }
}

void UI::build_emitter_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  if ((_selection.index < 0) || (static_cast<uint32_t>(_selection.index) >= scene_rep.data().emitter_profiles.size())) {
    ImGui::Text("Invalid emitter selection");
    return;
  }
  uint32_t emitter_index = static_cast<uint32_t>(_selection.index);
  build_emitter_resource_properties(scene_rep, emitter_index, data, true);
}

void UI::build_emitter_resource_properties(SceneRepresentation& scene_rep, uint32_t emitter_index, const FrameData& data, bool standalone_actions) {
  if (emitter_index >= scene_rep.data().emitter_profiles.size()) {
    ImGui::TextColored(kErrorTextColor, "Invalid emitter");
    return;
  }
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
    ImGui::TextColored(kErrorTextColor, "Material: None");
    draw_item_tooltip("This area emitter has no material reference.", ImGuiHoveredFlags_DelayNormal);
    return;
  }
  if (emitter.cls == EmitterProfile::Class::Area) {
    _material_editor_rendered_this_frame = true;
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
      if (standalone_actions && (integrated.x <= 0.0f) && (integrated.y <= 0.0f) && (integrated.z <= 0.0f)) {
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

      ImGui::TextUnformatted("Collimation");
      full_width_item();
      float collimation = material.emission_collimation;
      if (ImGui::SliderFloat("##area_collimation", &collimation, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
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
        ImGui::TextUnformatted("Altitude");
        full_width_item();
        if (ImGui::DragFloat("##altitude", &sky_emitter.atmosphere.scattering.altitude, 10.0f, 100.0f, 100000.0f, "%.0f m")) {
          changed = true;
        }
        ImGui::TextUnformatted("Anisotropy");
        full_width_item();
        if (ImGui::SliderFloat("##anisotropy", &sky_emitter.atmosphere.scattering.anisotropy, -0.999f, 0.999f, "%.3f", ImGuiSliderFlags_None)) {
          changed = true;
        }
        ImGui::TextUnformatted("Rayleigh");
        full_width_item();
        if (ImGui::DragFloat("##rayleigh", &sky_emitter.atmosphere.scattering.rayleigh_scale, 0.0f, 0.0f, 10.0f, "%.3f")) {
          changed = true;
        }
        ImGui::TextUnformatted("Mie");
        full_width_item();
        if (ImGui::DragFloat("##mie", &sky_emitter.atmosphere.scattering.mie_scale, 0.0f, 0.0f, 10.0f, "%.3f")) {
          changed = true;
        }
        ImGui::TextUnformatted("Ozone");
        full_width_item();
        if (ImGui::DragFloat("##ozone", &sky_emitter.atmosphere.scattering.ozone_scale, 0.0f, 0.0f, 10.0f, "%.3f")) {
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

  if (standalone_actions && (emitter.cls != EmitterProfile::Class::Area)) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    push_semantic_button_colors(terminate_button_colors(_theme));
    bool emitter_deleted = false;
    if (ImGui::Button("Delete Emitter", ImVec2(-1.0f, 0.0f))) {
      emitter_deleted = callbacks.emitter_deleted && callbacks.emitter_deleted(emitter_index);
    }
    ImGui::PopStyleColor(3);
    if (emitter_deleted) {
      set_selection(SelectionKind::None, -1, false);
      return;
    }
  }
  if (material_changed && (material_index < scene_rep.data().materials.size())) {
    queue_material_change(material_index);
  }
  if (changed && (material_changed == false) && callbacks.emitter_changed) {
    callbacks.emitter_changed(emitter_index);
  }
}

uint32_t UI::build_mesh_material_assignment(SceneRepresentation& scene_rep, uint32_t mesh_index) {
  if (mesh_index >= scene_rep.data().meshes.size()) {
    return kInvalidIndex;
  }

  const Mesh& mesh = scene_rep.data().meshes[mesh_index];
  uint32_t current_material = kInvalidIndex;
  if (mesh.triangle_count > 0u) {
    const uint32_t first_triangle_index = mesh.triangle_offset;
    if (first_triangle_index < scene_rep.data().triangles.size()) {
      current_material = scene_rep.data().triangles[first_triangle_index].material_index;
    }
  }

  ImGui::TextUnformatted("Assigned Material");
  if (_material_mapping.empty()) {
    ImGui::TextDisabled("No materials available");
    return current_material;
  }

  std::vector<const char*> material_names = {};
  material_names.reserve(_material_mapping.size());
  int selected_material = -1;
  for (uint64_t material_position = 0u; material_position < _material_mapping.size(); ++material_position) {
    const auto& entry = _material_mapping.entry(static_cast<int32_t>(material_position));
    material_names.push_back(entry.name);
    if (_material_mapping.at(static_cast<int32_t>(material_position)) == current_material) {
      selected_material = static_cast<int>(material_position);
    }
  }

  full_width_item();
  if (ImGui::Combo("##mesh_material", &selected_material, material_names.data(), static_cast<int>(material_names.size())) && (selected_material >= 0)) {
    const uint32_t new_material_index = _material_mapping.at(selected_material);
    if (callbacks.mesh_material_changed) {
      callbacks.mesh_material_changed(mesh_index, new_material_index);
    }
    current_material = new_material_index;
  }
  return current_material;
}

uint32_t UI::material_mesh_usage_count(const SceneRepresentation& scene_rep, uint32_t material_index) const {
  uint32_t usage_count = 0u;
  const auto& scene = scene_rep.data();
  std::vector<uint8_t> counted_meshes(scene.meshes.size(), 0u);
  for (const SceneNode& node : scene.hierarchy.nodes) {
    const uint64_t attachment_begin = node.attachment_offset;
    const uint64_t attachment_end = std::min<uint64_t>(attachment_begin + node.attachment_count, scene.hierarchy.attachments.size());
    for (uint64_t attachment_index = attachment_begin; attachment_index < attachment_end; ++attachment_index) {
      const SceneAttachment& attachment = scene.hierarchy.attachments[attachment_index];
      if ((attachment.type != SceneAttachment::Type::Mesh) || (attachment.resource_index >= scene.meshes.size()) || (counted_meshes[attachment.resource_index] != 0u)) {
        continue;
      }

      counted_meshes[attachment.resource_index] = 1u;
      const Mesh& mesh = scene.meshes[attachment.resource_index];
      const uint64_t triangle_begin = mesh.triangle_offset;
      const uint64_t triangle_end = std::min<uint64_t>(triangle_begin + mesh.triangle_count, scene.triangles.size());
      for (uint64_t triangle_index = triangle_begin; triangle_index < triangle_end; ++triangle_index) {
        if (scene.triangles[triangle_index].material_index == material_index) {
          ++usage_count;
          break;
        }
      }
    }
  }
  return usage_count;
}

void UI::build_node_appearance_properties(SceneRepresentation& scene_rep, const SceneNode& node, uint32_t attachment_end, const FrameData& data) {
  const SceneHierarchy& hierarchy = scene_rep.data().hierarchy;
  std::vector<uint32_t> mesh_indices = {};
  mesh_indices.reserve(node.attachment_count);
  for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
    const SceneAttachment& attachment = hierarchy.attachments[attachment_index];
    if (attachment.type == SceneAttachment::Type::Mesh) {
      mesh_indices.push_back(attachment.resource_index);
    }
  }
  if (mesh_indices.empty()) {
    return;
  }

  ImGui::Spacing();
  ImGui::SeparatorText(mesh_indices.size() == 1u ? "Appearance" : "Appearances");
  ImGui::PushID("node_appearance");
  for (size_t mesh_position = 0u; mesh_position < mesh_indices.size(); ++mesh_position) {
    const uint32_t mesh_index = mesh_indices[mesh_position];
    ImGui::PushID(static_cast<int>(mesh_position));
    bool appearance_open = true;
    if (mesh_indices.size() > 1u) {
      const char* mesh_name = _mesh_mapping.name_for(mesh_index);
      ImGui::SetNextItemOpen(mesh_position == 0u, ImGuiCond_Once);
      appearance_open = ImGui::CollapsingHeader(mesh_name != nullptr ? mesh_name : "Mesh", ImGuiTreeNodeFlags_Framed);
    }
    if (appearance_open) {
      uint32_t material_index = build_mesh_material_assignment(scene_rep, mesh_index);
      if (material_index < scene_rep.data().materials.size()) {
        const uint32_t usage_count = material_mesh_usage_count(scene_rep, material_index);
        if (usage_count > 1u) {
          ImGui::TextDisabled("Shared by %u meshes", usage_count);
          if (ImGui::Button("Make Unique", ImVec2(-FLT_MIN, 0.0f)) && callbacks.mesh_material_made_unique) {
            const uint32_t unique_material_index = callbacks.mesh_material_made_unique(mesh_index, material_index);
            if (unique_material_index < scene_rep.data().materials.size()) {
              material_index = unique_material_index;
              _material_mapping.build(scene_rep.material_mapping());
              _material_mapping_hash = hash_mapping(scene_rep.material_mapping());
              clear_selection_history();
            }
          }
        }
        ImGui::Spacing();
        ImGui::PushID("inline_material_editor");
        Material& material = scene_rep.data().materials[material_index];
        if (build_material(scene_rep, material, data)) {
          queue_material_change(material_index);
        }
        ImGui::PopID();
      } else {
        ImGui::TextDisabled("No material assigned");
      }
    }
    ImGui::PopID();
  }
  ImGui::PopID();
}

void UI::build_camera_selection_properties(SceneRepresentation& scene_rep, Camera& camera, uint32_t camera_index, bool attachment_enabled, const FrameData& data) {
  bool camera_changed = false;
  bool camera_is_active = false;
  if (camera_index < scene_rep.data().cameras.size()) {
    camera_is_active = scene_rep.data().cameras[camera_index].active;
  }

  bool active_control = camera_is_active;
  if (camera_is_active || (attachment_enabled == false)) {
    ImGui::BeginDisabled();
  }
  const bool activate_camera = ImGui::Checkbox("Active", &active_control) && active_control;
  if (camera_is_active || (attachment_enabled == false)) {
    ImGui::EndDisabled();
  }
  if (activate_camera && callbacks.camera_activated) {
    callbacks.camera_activated(camera_index);
    return;
  }

  uint2 viewport = camera.film_size;
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

      if (labeled_control("Horizontal FOV", [&]() {
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

    float clip_values[2] = {camera.clip_near, camera.clip_far};
    if (labeled_control("Clip Planes", [&]() {
          return ImGui::DragFloat2("##clipplanes", clip_values, 0.01f, 0.0f, 5000.0f, "%.3f");
        })) {
      camera.clip_near = max(0.0f, clip_values[0]);
      camera.clip_far = max(camera.clip_near + 0.001f, clip_values[1]);
      camera_changed = true;
    }
  }

  float pixel_filter_radius = scene_rep.data().pixel_filter.radius;
  if (ImGui::CollapsingHeader("Output", ImGuiTreeNodeFlags_Framed)) {
    ImGui::TextUnformatted("Image Size");
    full_width_item();
    if (ImGui::InputInt2("##outimgsize", reinterpret_cast<int32_t*>(&viewport.x))) {
      camera_changed = true;
    }

    if (labeled_control("Pixel Filter Radius", [&]() {
          return ImGui::DragFloat("##pixelfiler", &pixel_filter_radius, 0.05f, 0.0f, 32.0f, "%.3fpx");
        })) {
      camera_changed = true;
    }

    ImGui::TextUnformatted("Pixel Size");
    full_width_item();
    if (ImGui::Combo("##pixelsize", &pixel_size, "Default\0Scaled 2x\0Scaled 4x\0Scaled 8x\0Scaled 16x\0")) {
      camera_changed = true;
    }
  }

  if (_medium_mapping.empty() == false) {
    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_Framed)) {
      if (labeled_control("External Medium", [&]() {
            return medium_dropdown("##camera_external_medium", camera.medium_index);
          })) {
        camera_changed = true;
      }
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
    build_camera(camera, camera.position, camera.direction, camera.up, camera.film_size, fov);

    if (camera_is_active) {
      scene_rep.update_active_camera();
      if (callbacks.camera_changed) {
        callbacks.camera_changed(viewport, 1u << pixel_size);
      }
    } else if (callbacks.scene_settings_changed) {
      callbacks.scene_settings_changed();
    }
  }
}

void UI::build_scene_selection_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  (void)ctx;
  bool scene_settings_changed = false;

  if (ImGui::CollapsingHeader("Sampling", ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_DefaultOpen)) {
    if (validated_int_control("Samples Per Pixel", reinterpret_cast<int32_t&>(scene_rep.data().options.samples), 1, 1000000)) {
      scene_settings_changed = true;
    }

    const bool noise_threshold_changed = labeled_control("Noise Threshold", [&]() {
      return ImGui::InputFloat("##noise_thresh", &scene_rep.data().options.noise_threshold, 0.0001f, 0.01f, "%0.5f");
    });
    draw_item_tooltip("Experimental adaptive-sampling threshold.", ImGuiHoveredFlags_DelayNormal);
    if (noise_threshold_changed) {
      scene_rep.data().options.noise_threshold = std::clamp(scene_rep.data().options.noise_threshold, 0.0f, 1.0f);
      scene_settings_changed = true;
    }
    if (scene_rep.data().options.noise_threshold > 0.0f) {
      const uint32_t current_pixel_count = data.film.current_pixel_count();
      const double active_pixel_percentage = (current_pixel_count > 0u) ? (double(data.film.active_pixel_count()) / double(current_pixel_count) * 100.0) : 0.0;
      ImGui::Text("Active pixels: %.2f%%", active_pixel_percentage);
    }
  }

  ImGui::Spacing();
  if (ImGui::CollapsingHeader("Path Transport", ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_DefaultOpen)) {
    int32_t min_path = static_cast<int32_t>(scene_rep.data().options.min_path_length);
    int32_t max_path = static_cast<int32_t>(scene_rep.data().options.max_path_length);
    const bool min_changed = validated_int_control("Min Path Length", min_path, 0, static_cast<int32_t>(kMaximumPathLength));
    const bool max_changed = validated_int_control("Max Path Length", max_path, 0, static_cast<int32_t>(kMaximumPathLength));
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
    const bool spectral_changed = ImGui::Checkbox("Spectral Rendering", scene_rep.data().options.properties + Scene::Properties::Spectral);
    scene_settings_changed = (scene_settings_changed || spectral_changed);
  }

  if (scene_settings_changed) {
    scene_rep.data().options.max_path_length = min(scene_rep.data().options.max_path_length, kMaximumPathLength);
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

  bool options_changed = false;

  if ((_current_integrator->options().options.empty() == false) && ImGui::CollapsingHeader("Integrator Settings", ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_DefaultOpen)) {
    options_changed = build_options(_current_integrator->options());
  }

  ImGui::Spacing();
  if (ImGui::CollapsingHeader("Strategies", ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_DefaultOpen)) {
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
    draw_strategy_checkbox("Connect to Lights", Scene::Strategy::ConnectToLight);
    draw_strategy_checkbox("Camera Connections", Scene::Strategy::ConnectToCamera);
    draw_strategy_checkbox("Bidirectional Connections", Scene::Strategy::ConnectVertices);
    draw_strategy_checkbox("Photon Merging", Scene::Strategy::MergeVertices);

    if (strategies_changed && callbacks.scene_settings_changed) {
      callbacks.scene_settings_changed();
    }
    const bool mis_changed = ImGui::Checkbox("Multiple Importance Sampling", scene_rep.data().options.properties + Scene::Properties::MultipleImportanceSampling);
    if (mis_changed && callbacks.scene_settings_changed) {
      callbacks.scene_settings_changed();
    }

    int current_light_sampling = static_cast<int>(scene_rep.data().options.light_sampling);
    const char* light_sampling_options[] = {"Uniform", "From Distribution", "RIS Uniform", "RIS From Distribution"};
    ImGui::TextUnformatted("Light Selection");
    full_width_item();
    const bool light_sampling_changed = ImGui::Combo("##light_sampling", &current_light_sampling, light_sampling_options, IM_ARRAYSIZE(light_sampling_options));
    if (light_sampling_changed) {
      scene_rep.data().options.light_sampling = static_cast<Scene::LightSampling>(current_light_sampling);
      if (callbacks.scene_settings_changed) {
        callbacks.scene_settings_changed();
      }
    }
  }

  if (options_changed && callbacks.options_changed) {
    callbacks.options_changed();
  }
}

void UI::build_rendering_properties(SceneRepresentation& scene_rep, const BuildContext& ctx, const FrameData& data) {
  if (_embedded_toolbar_enabled == false) {
    ImGui::TextUnformatted("Integrator");
    full_width_item();
    build_render_configuration_selector("##render_configuration");
    ImGui::Spacing();
  }

  if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_DefaultOpen)) {
    if (labeled_control("Exposure",
          [&]() {
            return ImGui::DragFloat("##exposure", &_view_options.exposure, 1.0f / 256.0f, 1.0f / 1024.0f, 1024.0f, "%.4f", ImGuiSliderFlags_NoRoundToFormat);
          }) &&
        callbacks.exposure_changed) {
      callbacks.exposure_changed(_view_options.exposure);
    }

    if (_embedded_toolbar_enabled == false) {
      ImGui::TextUnformatted("View Layer");
      full_width_item();
      if (ImGui::BeginCombo("##view_layer", Film::layer_name(_view_options.view_layer))) {
        for (uint32_t i = 0; i < ViewLayer::Count; ++i) {
          const bool selected = i == _view_options.view_layer;
          if (ImGui::Selectable(Film::layer_name(i), selected)) {
            _view_options.view_layer = i;
            if (callbacks.view_layer_changed)
              callbacks.view_layer_changed(i);
          }
        }
        ImGui::EndCombo();
      }

      ImGui::TextUnformatted("Output Image");
      full_width_item();
      if (ImGui::BeginCombo("##view_image", output_view_to_string(uint32_t(_view_options.view_image)).c_str())) {
        for (uint32_t i = 0; i < uint32_t(OutputView::Count); ++i) {
          const bool selected = i == uint32_t(_view_options.view_image);
          if (ImGui::Selectable(output_view_to_string(i).c_str(), selected)) {
            _view_options.view_image = i;
            if (callbacks.output_view_changed)
              callbacks.output_view_changed(i);
          }
        }
        ImGui::EndCombo();
      }

      ImGui::TextUnformatted("Display Transform");
      full_width_item();
      if (ImGui::BeginCombo("##view_option", view_option_to_string(uint32_t(_view_options.view_option)).c_str())) {
        for (uint32_t i = 0; i < uint32_t(ViewOptions::Count); ++i) {
          const bool selected = i == uint32_t(_view_options.view_option);
          if (ImGui::Selectable(view_option_to_string(i).c_str(), selected)) {
            _view_options.view_option = i;
            if (callbacks.display_transform_changed)
              callbacks.display_transform_changed(i);
          }
        }
        ImGui::EndCombo();
      }

      if (_current_renderer_mode == RendererMode::CPURaytracing) {
        const bool can_denoise =
          _current_renderer_controls.can_run && (_current_renderer_status.progress_kind == RendererProgressKind::Samples) && (_current_renderer_status.completed_units > 0u);
        ImGui::Spacing();
        if (can_denoise == false) {
          ImGui::BeginDisabled();
        }
        if (ImGui::Button("Denoise Image", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f)) && callbacks.denoise_selected) {
          callbacks.denoise_selected();
        }
        if (can_denoise == false) {
          ImGui::EndDisabled();
        }
      }
    }
  }

  ImGui::Spacing();

  const bool raytracing_mode = (_current_renderer_mode == RendererMode::CPURaytracing) || (_current_renderer_mode == RendererMode::GPURaytracing);
  if (raytracing_mode) {
    build_scene_selection_properties(scene_rep, ctx, data);
    build_integrator_selection_properties(scene_rep, ctx);
  }

  if (_current_renderer_mode == RendererMode::GPURaytracing) {
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("GPU Diagnostics", ImGuiTreeNodeFlags_Framed)) {
      ImGui::Text("Wavefront: %u step%s", _gpu_wavefront_steps_per_frame, (_gpu_wavefront_steps_per_frame == 1u) ? "" : "s");
      draw_item_tooltip(_gpu_wavefront_automatic ? "Automatically tuned wavefront steps per UI frame." : "Wavefront steps per UI frame.", ImGuiHoveredFlags_DelayNormal);
      if (_gpu_wavefront_automatic && (_gpu_wavefront_last_batch_ms > 0.0)) {
        ImGui::Text("GPU batch: %.1f ms", _gpu_wavefront_last_batch_ms);
        draw_item_tooltip("Time spent executing the most recent wavefront batch.", ImGuiHoveredFlags_DelayNormal);
      } else if (_gpu_wavefront_automatic) {
        ImGui::TextUnformatted("GPU batch: measuring");
        draw_item_tooltip("Waiting for the first valid wavefront timing measurement.", ImGuiHoveredFlags_DelayNormal);
      }
      ImGui::Text("Path depth: camera %u | light %u | limit %u", _renderer_memory_stats.max_observed_camera_path_length, _renderer_memory_stats.max_observed_light_path_length,
        _renderer_memory_stats.max_path_length);
      draw_item_tooltip("Maximum camera and light path depth processed since render accumulation was reset.", ImGuiHoveredFlags_DelayNormal);
      bool kernel_timing_enabled = _gpu_kernel_timing_stats.enabled;
      if (ImGui::Checkbox("Profile kernels", &kernel_timing_enabled)) {
        _gpu_kernel_timing_stats.enabled = kernel_timing_enabled;
        if (callbacks.gpu_kernel_timing_enabled_changed) {
          callbacks.gpu_kernel_timing_enabled_changed(kernel_timing_enabled);
        }
      }
      draw_item_tooltip("Collects per-kernel GPU timestamps and adds query overhead.", ImGuiHoveredFlags_DelayNormal);
    }
  }
}

}  // namespace etx
