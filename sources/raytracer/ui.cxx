#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>

#include <etx/render/host/film.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/render/shared/ior_database.hxx>

#include "ui.hxx"

#include <sokol_app.h>
#include <sokol_gfx.h>

#include <imgui.h>
#include <imgui_internal.h>
#include <util/sokol_imgui.h>

#if (ETX_PLATFORM_APPLE)
# include <unistd.h>
#endif

#include <map>
#include <vector>
#include <algorithm>
#include <cmath>

namespace etx {

namespace {

inline void decrease_exposure(ViewOptions& o) {
  o.exposure = fmaxf(1.0f / 1024.0f, 0.5f * o.exposure);
}

inline void increase_exposure(ViewOptions& o) {
  o.exposure = fmaxf(1.0f / 1024.0f, 2.0f * o.exposure);
}

}  // namespace

void UI::MappingRepresentation::build(const std::unordered_map<std::string, uint32_t>& in_mapping) {
  size_t max_len = 32u;
  std::vector<std::pair<std::string, uint32_t>> unfold;
  unfold.reserve(in_mapping.size());
  for (const auto& m : in_mapping) {
    if ((m.first.starts_with("etx::") == false) && (m.first.starts_with("et::") == false)) {
      unfold.emplace_back(m.first, m.second);
      max_len = std::max(max_len, m.first.size() + 1u);
    }
  }
  std::sort(unfold.begin(), unfold.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  indices.clear();
  indices.reserve(unfold.size());
  names.clear();
  names.reserve(unfold.size());
  data.resize(unfold.size() * max_len);
  std::fill(data.begin(), data.end(), 0);
  int32_t pp = 0;
  char* ptr = data.data();
  for (auto& m : unfold) {
    indices.emplace_back(m.second);
    names.emplace_back(ptr + pp);
    pp += snprintf(ptr + pp, max_len, "%s", m.first.c_str()) + 1;
  }
}

void UI::initialize(Film* film, const IORDatabase* db) {
  _film = film;
  _ior_database = db;

  simgui_desc_t imggui_desc = {};
  imggui_desc.depth_format = SG_PIXELFORMAT_NONE;
  imggui_desc.no_default_font = true;
  simgui_setup(imggui_desc);
  {
    auto font_config = ImFontConfig();
    font_config.OversampleH = 4;
    font_config.OversampleV = 4;

    auto& io = ImGui::GetIO();
    unsigned char* font_pixels = nullptr;
    int font_width = 0;
    int font_height = 0;
    int bytes_per_pixel = 0;

    char font_file[1024] = {};
    env().file_in_data("fonts/roboto.ttf", font_file, sizeof(font_file));
    auto font = io.Fonts->AddFontFromFileTTF(font_file, 15.0f * sapp_dpi_scale(), &font_config, nullptr);
    font->Scale = 1.0f / sapp_dpi_scale();
    io.Fonts->GetTexDataAsRGBA32(&font_pixels, &font_width, &font_height, &bytes_per_pixel);

    sg_image_desc img_desc = {};
    img_desc.width = font_width;
    img_desc.height = font_height;
    img_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
    img_desc.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
    img_desc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
    img_desc.min_filter = SG_FILTER_LINEAR;
    img_desc.mag_filter = SG_FILTER_LINEAR;
    img_desc.data.subimage[0][0].ptr = font_pixels;
    img_desc.data.subimage[0][0].size = (size_t)(font_width * font_height) * sizeof(uint32_t);
    img_desc.label = "sokol-imgui-font";
    _font_image = sg_make_image(&img_desc).id;
    io.Fonts->TexID = (ImTextureID)(uintptr_t)_font_image;
  }
  ImGui::LoadIniSettingsFromDisk(env().file_in_data("ui.ini"));
}

void UI::cleanup() {
  ImGui::SaveIniSettingsToDisk(env().file_in_data("ui.ini"));
}

void UI::set_selection(SelectionKind kind, int32_t index, bool track_history) {
  SelectionState next{kind, index};
  if ((_selection.kind == next.kind) && (_selection.index == next.index))
    return;

  _selection = next;
  // _spectrum_editors.clear();

  if (!track_history)
    return;

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

void UI::navigate_history(int32_t step) {
  if ((_selection_history.empty()) || (step == 0))
    return;

  int32_t target = std::clamp(_selection_history_cursor + step, 0, static_cast<int32_t>(_selection_history.size()) - 1);
  if (target == _selection_history_cursor)
    return;

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
        ImGui::TextColored({1.0f, 0.5f, 0.25f, 1.0f}, "%s", data.value.c_str());
        break;
      };
      case Option::Class::Boolean: {
        auto& data = option.as<Option::Class::Boolean>();
        changed = ImGui::Checkbox(option.description.c_str(), &data.value);
        break;
      }
      case Option::Class::Float: {
        auto& data = option.as<Option::Class::Float>();
        if (data.bounds.maximum > data.bounds.minimum) {
          changed = ImGui::DragFloat(option.description.c_str(), &data.value, 0.1f, data.bounds.minimum, data.bounds.maximum, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        } else {
          changed = ImGui::InputFloat(option.description.c_str(), &data.value);
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
          if (data.bounds.maximum > data.bounds.minimum) {
            changed = ImGui::DragInt(option.description.c_str(), &data.value, 0.1f, data.bounds.minimum, data.bounds.maximum, "%d", ImGuiSliderFlags_AlwaysClamp);
          } else {
            changed = ImGui::DragInt(option.description.c_str(), &data.value);
          }
        }
        break;
      }
      case Option::Class::Float3: {
        auto& data = option.as<Option::Class::Float3>();
        ImGui::SetNextItemWidth(4.0f * ImGui::GetFontSize());
        ImGui::Text("%s", option.description.c_str());
        char buffer_name[128] = {};
        snprintf(buffer_name, sizeof(buffer_name), "##%s", option.description.c_str());
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

bool UI::ior_picker(Scene& scene, const char* name, RefractiveIndex& ior) {
  bool changed = false;
  bool load_from_file = false;

  const ImVec4 colors[5] = {
    {0.3333f, 0.3333f, 0.3333f, 1.0f},
    {1.0f, 1.0f, 1.0f, 1.0f},
    {0.5f, 0.75f, 1.0f, 1.0f},
    {1.0f, 0.75f, 0.5f, 1.0f},
    {1.0f, 0.75f, 1.0f, 1.0f},
  };

  const IORDatabase* database = _ior_database;
  int matched_index = -1;
  static const SpectralDistribution null_spectrum = SpectralDistribution::null();
  if ((database != nullptr) && (ior.cls != SpectralDistribution::Class::Invalid)) {
    const SpectralDistribution& current_eta = scene.spectrums[ior.eta_index];
    const SpectralDistribution& current_k = (ior.k_index != kInvalidIndex) ? scene.spectrums[ior.k_index] : null_spectrum;
    matched_index = database->find_matching_index(current_eta, current_k, ior.cls);
  }

  const char* preview_text = name;
  if ((database != nullptr) && (matched_index >= 0) && (matched_index < static_cast<int>(database->definitions.size()))) {
    preview_text = database->definitions[static_cast<size_t>(matched_index)].title.c_str();
  }
  std::string button_label = std::string(preview_text) + "##ior_" + name;
  if (ImGui::Button(button_label.c_str(), ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
    char popup_id[64] = {};
    snprintf(popup_id, sizeof(popup_id), "ior_popup_##%s", name);
    ImGui::OpenPopup(popup_id);
  }

  char popup_id[64] = {};
  snprintf(popup_id, sizeof(popup_id), "ior_popup_##%s", name);
  ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 32.0f, 0.0f), ImGuiCond_Always);
  if (ImGui::BeginPopup(popup_id)) {
    if ((database != nullptr) && (database->definitions.empty() == false)) {
      struct ColumnInfo {
        SpectralDistribution::Class cls;
        const char* title;
      };

      std::vector<ColumnInfo> columns;
      const auto& conductors = database->class_entries(SpectralDistribution::Class::Conductor);
      if (conductors.empty() == false) {
        columns.push_back({SpectralDistribution::Class::Conductor, "Conductors"});
      }
      const auto& dielectrics = database->class_entries(SpectralDistribution::Class::Dielectric);
      if (dielectrics.empty() == false) {
        columns.push_back({SpectralDistribution::Class::Dielectric, "Dielectrics"});
      }

      if (columns.empty()) {
        ImGui::TextDisabled("No predefined IORs available");
        ImGui::Separator();
      } else {
        ImGui::Columns(static_cast<int>(columns.size()), nullptr, true);

        auto draw_column = [&](SpectralDistribution::Class cls, const char* title) {
          ImGui::PushStyleColor(ImGuiCol_Text, colors[uint32_t(cls)]);
          ImGui::Text("%s", title);
          ImGui::PopStyleColor();
          const auto& entries = database->class_entries(cls);
          for (size_t idx : entries) {
            if (idx >= database->definitions.size())
              continue;
            const IORDefinition& def = database->definitions[idx];
            int def_index = static_cast<int>(idx);
            bool is_current = (matched_index == def_index);
            if (ImGui::Selectable(def.title.c_str(), is_current)) {
              matched_index = def_index;
              ior.cls = def.cls;
              ETX_CRITICAL(ior.eta_index != kInvalidIndex);
              scene.spectrums[ior.eta_index] = def.eta;
              ETX_CRITICAL(ior.k_index != kInvalidIndex);
              scene.spectrums[ior.k_index] = def.k;
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
    SpectralDistribution t_eta = {};
    SpectralDistribution t_k = {};
    auto cls = RefractiveIndex::load_from_file(filename.c_str(), t_eta, t_k);
    if (cls != SpectralDistribution::Class::Invalid) {
      ior.cls = cls;
      ETX_CRITICAL(ior.eta_index != kInvalidIndex);
      scene.spectrums[ior.eta_index] = t_eta;
      ETX_CRITICAL(ior.k_index != kInvalidIndex);
      scene.spectrums[ior.k_index] = t_k;
      changed = true;
    }
  }

  return changed;
}

bool UI::emission_picker(Scene& scene, const char* label, const char* id_suffix, uint32_t& spectrum_index) {
  if (scene.spectrums.count == 0)
    return false;

  auto ensure_index = [&](uint32_t& index) {
    if ((index == kInvalidIndex) || (index >= scene.spectrums.count)) {
      uint32_t fallback = scene.black_spectrum;
      if ((fallback == kInvalidIndex) || (fallback >= scene.spectrums.count)) {
        fallback = 0u;
      }
      index = fallback;
    }
  };

  ensure_index(spectrum_index);

  bool changed = false;
  bool load_from_file = false;

  const char* base_label = (label != nullptr) ? label : "Emission";
  const char* unique_id = (id_suffix != nullptr) ? id_suffix : base_label;

  std::string color_name = std::string(base_label) + "_Color_" + unique_id;
  char editor_key_buf[32] = {};
  snprintf(editor_key_buf, sizeof(editor_key_buf), "%p", (void*)&scene.spectrums[spectrum_index]);
  std::string editor_key = std::string(editor_key_buf);
  auto [state_it, inserted] = _spectrum_editors.emplace(editor_key, SpectrumEditorState{});
  SpectrumEditorState& editor_state = state_it->second;

  static const char* kModeLabels[] = {
    "Color",
    "Temperature",
    "Preset",
  };

  ImGui::Text("Emission Input");
  ImGui::SetNextItemWidth(-FLT_MIN);
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

  ImGui::SetNextItemWidth(-FLT_MIN);
  if (editor_state.mode == SpectrumEditorState::Mode::Temperature) {
    char temperature_label[128] = {};
    snprintf(temperature_label, sizeof(temperature_label), "##emission_temp_%s", unique_id);
    float temperature = editor_state.temperature;
    if (ImGui::InputFloat(temperature_label, &temperature, 100.0f, 1000.0f, "%.0f")) {
      temperature = std::clamp(temperature, 1000.0f, 40000.0f);
      editor_state.temperature = temperature;
      SpectralDistribution temp_spd = SpectralDistribution::from_normalized_black_body(temperature, editor_state.scale);
      scene.spectrums[spectrum_index] = temp_spd;
      editor_state.color = {};
      changed = true;
    }
  }
  changed |= spectrum_picker(scene, color_name.c_str(), spectrum_index, true, true, editor_state.mode == SpectrumEditorState::Mode::Color, true);

  const IORDatabase* database = _ior_database;
  int matched_index = -1;
  if ((database != nullptr) && (spectrum_index < scene.spectrums.count)) {
    static const SpectralDistribution null_spectrum = SpectralDistribution::null();
    matched_index = database->find_matching_index(scene.spectrums[spectrum_index], null_spectrum, SpectralDistribution::Class::Illuminant);
  }

  const char* preview_text = base_label;
  std::string preview_storage;
  if ((database != nullptr) && (matched_index >= 0) && (matched_index < static_cast<int>(database->definitions.size()))) {
    preview_storage = database->definitions[static_cast<size_t>(matched_index)].title;
    preview_text = preview_storage.c_str();
  }

  if (editor_state.mode == SpectrumEditorState::Mode::Preset) {
    std::string button_label = std::string(preview_text) + "##emission_" + unique_id;
    if (ImGui::Button(button_label.c_str(), ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
      char popup_id[64] = {};
      snprintf(popup_id, sizeof(popup_id), "emission_popup_##%s", unique_id);
      ImGui::OpenPopup(popup_id);
    }
  }

  char popup_id[64] = {};
  snprintf(popup_id, sizeof(popup_id), "emission_popup_##%s", unique_id);
  ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 28.0f, 0.0f), ImGuiCond_Always);
  if (ImGui::BeginPopup(popup_id)) {
    if (ImGui::Selectable("None", false)) {
      scene.spectrums[spectrum_index] = SpectralDistribution::null();
      matched_index = -1;
      changed = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::Separator();

    bool has_presets = false;
    if ((database != nullptr) && (database->definitions.empty() == false)) {
      const auto& entries = database->class_entries(SpectralDistribution::Class::Illuminant);
      has_presets = entries.empty() == false;

      if (has_presets) {
        size_t column_count = std::min<size_t>(2, std::max<size_t>(1, entries.size()));
        size_t per_column = (entries.size() + column_count - 1u) / column_count;

        std::string table_id = std::string("emission_presets_##") + unique_id;
        if (ImGui::BeginTable(table_id.c_str(), static_cast<int>(column_count), ImGuiTableFlags_SizingStretchSame)) {
          for (size_t col = 0; col < column_count; ++col) {
            ImGui::TableNextColumn();
            size_t start = col * per_column;
            size_t end = std::min(start + per_column, entries.size());
            for (size_t idx = start; idx < end; ++idx) {
              size_t def_index = entries[idx];
              if (def_index >= database->definitions.size())
                continue;
              const IORDefinition& def = database->definitions[def_index];
              bool is_current = (matched_index == static_cast<int>(def_index));
              if (ImGui::Selectable(def.title.c_str(), is_current)) {
                scene.spectrums[spectrum_index] = def.eta;
                scene.spectrums[spectrum_index].scale(editor_state.scale);
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
      SpectralDistribution loaded = {};
      auto cls = SpectralDistribution::load_from_file(filename.c_str(), loaded, nullptr, false);
      if (cls != SpectralDistribution::Class::Invalid) {
        scene.spectrums[spectrum_index] = loaded;
        scene.spectrums[spectrum_index].scale(editor_state.scale);
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
  const char* current = "None";
  for (uint64_t i = 0, e = _medium_mapping.size(); i < e; ++i) {
    if (_medium_mapping.indices[i] == medium) {
      current = _medium_mapping.names[i];
      break;
    }
  }

  if (ImGui::BeginCombo(label, current)) {
    bool is_none = (medium == kInvalidIndex);
    if (ImGui::Selectable("None", is_none)) {
      medium = kInvalidIndex;
      changed = true;
    }
    ImGui::Separator();
    for (uint64_t i = 0, e = _medium_mapping.size(); i < e; ++i) {
      bool is_selected = (medium == _medium_mapping.indices[i]);
      if (ImGui::Selectable(_medium_mapping.names[i], is_selected)) {
        medium = _medium_mapping.indices[i];
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

bool UI::spectrum_picker(Scene& scene, const char* widget_id, uint32_t spd_index, bool linear, bool scale, bool show_color, bool show_scale) {
  if (scene.spectrums.count == 0) {
    return false;
  }
  if (spd_index >= scene.spectrums.count) {
    return false;
  }
  SpectralDistribution& spd = scene.spectrums[spd_index];
  return spectrum_picker(widget_id, spd, linear, scale, show_color, show_scale);
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
  float max_component = std::max(std::max(linear_rgb.x, linear_rgb.y), linear_rgb.z);
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

  char name_buffer[128] = {};
  snprintf(name_buffer, sizeof(name_buffer), "##color_%s", widget_id);

  char scale_label[128] = {};
  if (scale && show_scale) {
    snprintf(scale_label, sizeof(scale_label), "##scale_%s", widget_id);
  }

  auto sync_from_spd = [&](bool update_color, bool update_scale) {
    if ((update_color == false) && (update_scale == false))
      return;

    float3 refreshed_linear = spd.integrated();
    float refreshed_scale = editor_state.scale;
    float3 refreshed_display = editor_state.color;

    constexpr float kScaleUnityThreshold = 1.0001f;
    if (scale && (update_color || update_scale)) {
      float refreshed_max = std::max(std::max(refreshed_linear.x, refreshed_linear.y), refreshed_linear.z);
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
    applied_scale = from_color ? std::max(applied_scale, 1.0f) : std::max(applied_scale, 0.0f);

    if (from_color) {
      value *= applied_scale;
      spd = SpectralDistribution::rgb_reflectance(value);
      editor_state.scale = applied_scale;
    } else if (scale) {
      spd.scale(applied_scale / std::max(editor_state.scale, 1.0e-6f));
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
    ImGui::SetNextItemWidth(-FLT_MIN);
    float drag_speed = std::max(0.01f, std::max(editor_state.scale, 1.0f) * 0.01f);
    float scale_value = editor_state.scale;
    scale_changed = ImGui::DragFloat(scale_label, &scale_value, drag_speed, show_color ? 1.0f : 0.01f, 1000.0f, "Scale: %.2f", ImGuiSliderFlags_NoRoundToFormat);
    scale_active = ImGui::IsItemActive();
    scale_deactivated_after_edit = ImGui::IsItemDeactivatedAfterEdit();
    if (scale_changed) {
      if (show_color) {
        scale_value = std::max(scale_value, 1.0f);
      } else {
        scale_value = std::max(scale_value, 0.0f);
      }
      rebuild_spectrum(scale_value, show_color);
      editor_state.scale = scale_value;
      changed = true;
    }
  }

  bool skip_sync =
    (show_color && (color_active || color_edited || color_deactivated_after_edit)) || (show_scale && (scale_active || scale_changed || scale_deactivated_after_edit));
  if (skip_sync == false) {
    sync_from_spd(show_color, show_color);
  }

  return changed;
}

constexpr uint32_t kWindowFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize;

void UI::build(double dt, const std::vector<std::string>& recent_files, Scene& scene, Camera& camera, const SceneRepresentation::MaterialMapping& materials,
  const SceneRepresentation::MediumMapping& mediums) {
  ETX_PROFILER_SCOPE();

  bool has_integrator = (_current_integrator != nullptr);
  bool has_scene = true;

  _panel_width = 0.0f;

  if (_selection.kind == SelectionKind::None) {
    set_selection(SelectionKind::Scene, 0);
  }

  std::vector<int32_t> emitter_primary_instance;
  if (scene.emitter_profiles.count > 0) {
    emitter_primary_instance.assign(scene.emitter_profiles.count, -1);
    for (uint32_t instance_index = 0; instance_index < scene.emitter_instances.count; ++instance_index) {
      uint32_t profile = scene.emitter_instances[instance_index].profile;
      if ((profile < emitter_primary_instance.size()) && (emitter_primary_instance[profile] == -1)) {
        emitter_primary_instance[profile] = static_cast<int32_t>(instance_index);
      }
    }
  }

  auto material_name_from_index = [&](uint32_t material_index) -> const char* {
    for (uint64_t i = 0; i < _material_mapping.size(); ++i) {
      if (_material_mapping.indices[i] == material_index) {
        return _material_mapping.names[i];
      }
    }
    return nullptr;
  };

  auto with_window = [&](uint32_t flag, const char* title, auto&& body, float min_chars = 30.0f) {
    if ((_ui_setup & flag) == 0)
      return;

    float char_width = ImGui::CalcTextSize("W").x;
    float fallback_width = char_width * min_chars + ImGui::GetStyle().WindowPadding.x * 2.0f;
    float target_width = (_panel_width > 0.0f) ? std::max(_panel_width, fallback_width) : fallback_width;

    ImGui::SetNextWindowSizeConstraints(ImVec2(target_width, 0.0f), ImVec2(FLT_MAX, FLT_MAX));

    if (ImGui::Begin(title, nullptr, kWindowFlags)) {
      body();
      float current_width = ImGui::GetWindowSize().x;
      _panel_width = std::max(_panel_width, current_width);
    }

    ImGui::End();
  };

  simgui_new_frame(simgui_frame_desc_t{sapp_width(), sapp_height(), dt, sapp_dpi_scale()});

  auto hash_mapping = [](const auto& m) -> uint64_t {
    uint64_t h = kFnv1a64Begin;
    for (const auto& kv : m) {
      h = fnv1a64(kv.first.c_str(), h);
      h = fnv1a64(reinterpret_cast<const uint8_t*>(&kv.second), sizeof(kv.second), h);
    }
    return h;
  };

  uint64_t mmh = hash_mapping(materials);
  if (mmh != _material_mapping_hash) {
    _material_mapping.build(materials);
    _material_mapping_hash = mmh;
  }

  uint64_t medh = hash_mapping(mediums);
  if (medh != _medium_mapping_hash) {
    _medium_mapping.build(mediums);
    _medium_mapping_hash = medh;
  }

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("etx-tracer")) {
      if (ImGui::MenuItem("Exit", "Ctrl+Q", false, true)) {
        quit();
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Scene", true)) {
      if (ImGui::MenuItem("Open...", "Ctrl+O", false, true)) {
        select_scene_file();
      }
      if (ImGui::MenuItem("Reload Scene", "Ctrl+R", false, true)) {
        reload_scene();
      }
      if (ImGui::MenuItem("Reload Geometry and Materials", "Ctrl+G", false, true)) {
        reload_geometry();
      }

      if (recent_files.empty() == false) {
        ImGui::Separator();
        if (ImGui::BeginMenu("Recent Files")) {
          for (uint64_t i = recent_files.size(); i > 0; --i) {
            if (ImGui::MenuItem(recent_files[i - 1u].c_str(), nullptr, nullptr)) {
              if (callbacks.scene_file_selected) {
                callbacks.scene_file_selected(recent_files[i - 1u]);
              }
            }
          }
          ImGui::EndMenu();
        }
      }

      ImGui::Separator();
      if (ImGui::MenuItem("Save", nullptr, false, true)) {
        save_scene_file();
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Integrator", true)) {
      for (uint64_t i = 0; i < _integrators.count; ++i) {
        if (ImGui::MenuItem(_integrators[i]->name(), nullptr, _current_integrator == _integrators[i], _integrators[i]->enabled())) {
          if (callbacks.integrator_selected) {
            callbacks.integrator_selected(_integrators[i]);
            set_current_integrator(_integrators[i]);
          }
        }
      }

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Image", true)) {
      if (ImGui::MenuItem("Open Reference Image...", "Ctrl+I", false, true)) {
        load_image();
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Save Current Image (RGB)...", "Ctrl+S", false, true)) {
        save_image(SaveImageMode::RGB);
      }
      if (ImGui::MenuItem("Save Current Image (LDR)...", "Shift+Ctrl+S", false, true)) {
        save_image(SaveImageMode::TonemappedLDR);
      }
      if (ImGui::MenuItem("Use as Reference", "Ctrl+Shift+R", false, true)) {
        if (callbacks.use_image_as_reference) {
          callbacks.use_image_as_reference();
        }
      }
      ImGui::EndMenu();
    }

    if (callbacks.view_scene && ImGui::BeginMenu("View", true)) {
      if (ImGui::MenuItem("View whole scene", nullptr, false, true)) {
        callbacks.view_scene(0);
      }
      if (ImGui::BeginMenu("View scene")) {
        if (ImGui::MenuItem("From +X", nullptr, false, true)) {
          callbacks.view_scene(1);
        }
        if (ImGui::MenuItem("From -X", nullptr, false, true)) {
          callbacks.view_scene(2);
        }
        if (ImGui::MenuItem("From +Y", nullptr, false, true)) {
          callbacks.view_scene(3);
        }
        if (ImGui::MenuItem("From -Y", nullptr, false, true)) {
          callbacks.view_scene(4);
        }
        if (ImGui::MenuItem("From +Z", nullptr, false, true)) {
          callbacks.view_scene(5);
        }
        if (ImGui::MenuItem("From -Z", nullptr, false, true)) {
          callbacks.view_scene(6);
        }
        ImGui::EndMenu();
      }

      ImGui::Separator();

      if (ImGui::MenuItem("Increase Exposure", "*", false, true)) {
        increase_exposure(_view_options);
      }
      if (ImGui::MenuItem("Decrease Exposure", "/", false, true)) {
        decrease_exposure(_view_options);
      }

      ImGui::Separator();

      auto ui_toggle = [this](const char* label, uint32_t flag) {
        uint32_t k = 0;
        for (; (k < 8) && (flag != (1u << k)); ++k) {
        }
        char buffer[8] = {};
        snprintf(buffer, sizeof(buffer), "F%u", k + 1u);
        bool ui_integrator = (_ui_setup & flag) == flag;
        if (ImGui::MenuItem(label, buffer, ui_integrator, true)) {
          _ui_setup = ui_integrator ? (_ui_setup & (~flag)) : (_ui_setup | flag);
        }
      };
      ui_toggle("Integrator Options", UIIntegrator);
      ui_toggle("Scene Objects", UIObjects);
      ui_toggle("Properties", UIProperties);
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }

  // TODO : lock
  bool scene_editable = has_integrator && has_scene && (_current_integrator->state() != Integrator::State::WaitingForCompletion);

  ImVec2 wpadding = ImGui::GetStyle().WindowPadding;
  ImVec2 fpadding = ImGui::GetStyle().FramePadding;
  float text_size = ImGui::GetFontSize();
  float button_size = 32.0f;
  float input_size = 64.0f;
  if (ImGui::BeginViewportSideBar("##toolbar", ImGui::GetMainViewport(), ImGuiDir_Up, button_size + 2.0f * wpadding.y, ImGuiWindowFlags_NoDecoration)) {
    bool can_run = has_integrator && _current_integrator->can_run();
    Integrator::State state = can_run ? _current_integrator->state() : Integrator::State::Stopped;

    bool state_available[4] = {
      can_run && (state == Integrator::State::Stopped),
      can_run && (state == Integrator::State::Running),
      can_run && (state != Integrator::State::Stopped),
      can_run && (state == Integrator::State::Running),
    };

    ImVec4 colors[4] = {
      state_available[0] ? ImVec4{0.11f, 0.44f, 0.11f, 1.0f} : ImVec4{0.25f, 0.25f, 0.25f, 1.0f},
      state_available[1] ? ImVec4{0.44f, 0.33f, 0.11f, 1.0f} : ImVec4{0.25f, 0.25f, 0.25f, 1.0f},
      state_available[2] ? ImVec4{0.44f, 0.11f, 0.11f, 1.0f} : ImVec4{0.25f, 0.25f, 0.25f, 1.0f},
      state_available[3] ? ImVec4{0.11f, 0.22f, 0.44f, 1.0f} : ImVec4{0.25f, 0.25f, 0.25f, 1.0f},
    };

    std::string labels[4] = {
      (state == Integrator::State::Running) ? "> Running <" : "  Launch  ",
      (state == Integrator::State::WaitingForCompletion) ? "> Finishing <" : "  Finish  ",
      " Terminate ",
      (state == Integrator::State::Running) ? " Restart " : "  Restart  ",
    };

    ImGui::SameLine(0.0f, wpadding.x);
    ImGui::PushStyleColor(ImGuiCol_Button, colors[0]);
    if (ImGui::Button(labels[0].c_str(), {0.0f, button_size})) {
      callbacks.run_selected();
    }

    ImGui::SameLine(0.0f, wpadding.x);
    ImGui::PushStyleColor(ImGuiCol_Button, colors[1]);
    if (ImGui::Button(labels[1].c_str(), {0.0f, button_size})) {
      callbacks.stop_selected(true);
    }

    ImGui::SameLine(0.0f, wpadding.x);
    ImGui::PushStyleColor(ImGuiCol_Button, colors[2]);
    if (ImGui::Button(labels[2].c_str(), {0.0f, button_size})) {
      callbacks.stop_selected(false);
    }

    ImGui::SameLine(0.0f, wpadding.x);
    ImGui::PushStyleColor(ImGuiCol_Button, colors[3]);
    if (ImGui::Button(labels[3].c_str(), {0.0f, button_size})) {
      callbacks.restart_selected();
    }
    ImGui::PopStyleColor(4);

    ImGui::SameLine(0.0f, wpadding.x);
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);

    ImGui::SameLine(0.0f, wpadding.x);
    if (state_available[0] == false) {
      ImGui::BeginDisabled();
    }
    if (ImGui::Button("  Denoise (preview)  ", {0.0f, button_size})) {
      callbacks.denoise_selected();
    }
    if (state_available[0] == false) {
      ImGui::EndDisabled();
    }

    ImGui::SameLine(0.0f, wpadding.x);

    ImGui::GetStyle().FramePadding.y = (button_size - text_size) / 2.0f;

    ImGui::PushItemWidth(input_size);
    {
      ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
      ImGui::SameLine(0.0f, wpadding.x);
      ImGui::DragFloat("Exposure", &_view_options.exposure, 1.0f / 256.0f, 1.0f / 1024.0f, 1024.0f, "%.4f", ImGuiSliderFlags_NoRoundToFormat);
      ImGui::SameLine(0.0f, wpadding.x);
    }
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine(0.0f, wpadding.x);
    ImGui::PopItemWidth();

    ImGui::PushItemWidth(2.0f * input_size);
    if (ImGui::BeginCombo("Layer", Film::layer_name(_view_options.layer))) {
      for (uint32_t i = 0; i < Film::LayerCount; ++i) {
        bool selected = i == _view_options.layer;
        if (ImGui::Selectable(Film::layer_name(i), &selected)) {
          if (selected) {
            _view_options.layer = i;
          }
        }
      }
      ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    ImGui::SameLine(0.0f, wpadding.x);
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine(0.0f, wpadding.x);
    ImGui::PushItemWidth(2.5f * input_size);
    if (ImGui::BeginCombo("##view_opt", output_view_to_string(uint32_t(_view_options.view)).c_str())) {
      for (uint32_t i = 0; i < uint32_t(OutputView::Count); ++i) {
        bool selected = i == uint32_t(_view_options.view);
        if (ImGui::Selectable(output_view_to_string(i).c_str(), &selected)) {
          _view_options.view = static_cast<OutputView>(i);
        }
      }
      ImGui::EndCombo();
    }
    ImGui::PopItemWidth();
    ImGui::GetStyle().FramePadding.y = fpadding.y;
    ImGui::End();
  }

  if (ImGui::BeginViewportSideBar("##status", ImGui::GetMainViewport(), ImGuiDir_Down, text_size + 2.0f * wpadding.y, ImGuiWindowFlags_NoDecoration)) {
    constexpr const char* status_str[] = {
      "Stopped",
      "Running",
      "Completing",
    };

    auto status = _current_integrator ? _current_integrator->status() : Integrator::Status{};
    auto state = _current_integrator ? _current_integrator->state() : Integrator::State::Stopped;

    double average_time = status.completed_iterations > 0 ? status.total_time / status.completed_iterations : 0.0;

    char buffer[2048] = {};
    snprintf(buffer, sizeof(buffer), "%-4d | %s | %.3fms last, %.3fms avg, %.3fs total",  //
      status.completed_iterations, status_str[uint32_t(state)],                           //
      status.last_iteration_time * 1000.0, average_time * 1000.0f, status.total_time);

    ImGui::Text("%s", buffer);
    ImGui::End();
  }

  with_window(UIIntegrator, has_integrator ? _current_integrator->name() : "Integrator options", [&]() {
    if (has_integrator) {
      bool options_changed = build_options(_current_integrator->options());
      if (options_changed && callbacks.options_changed) {
        callbacks.options_changed();
      }
    } else {
      ImGui::Text("No integrator selected");
    }
  });

  with_window(UIObjects, "Scene Objects", [&]() {
    if (!has_scene) {
      ImGui::Text("No scene loaded");
      return;
    }

    if (!scene_editable) {
      ImGui::TextDisabled("Rendering in progress; editing disabled.");
    }

    auto draw_history_button = [&](const char* label, bool enabled, int32_t step) {
      const ImVec4 base_color = ImVec4(0.2f, 0.6f, 0.2f, 1.0f);
      const ImVec4 hover_color = ImVec4(0.25f, 0.75f, 0.25f, 1.0f);
      const ImVec4 active_color = ImVec4(0.15f, 0.5f, 0.15f, 1.0f);
      ImGui::PushStyleColor(ImGuiCol_Button, base_color);
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover_color);
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, active_color);
      if (enabled == false)
        ImGui::BeginDisabled();
      if (ImGui::Button(label)) {
        navigate_history(step);
      }
      if (enabled == false)
        ImGui::EndDisabled();
      ImGui::PopStyleColor(3);
    };

    draw_history_button("Back##selection_history_back", can_navigate_back(), -1);
    ImGui::SameLine();
    draw_history_button("Forward##selection_history_forward", can_navigate_forward(), 1);

    ImGui::Spacing();
    ImGui::Separator();

    const ImVec4 scene_color = ImVec4(0.30f, 0.45f, 0.90f, 1.0f);
    const ImVec4 camera_color = ImVec4(0.90f, 0.80f, 0.35f, 1.0f);

    ImGui::PushStyleColor(ImGuiCol_Text, scene_color);
    bool scene_selected = (_selection.kind == SelectionKind::Scene);
    if (ImGui::Selectable("Scene", scene_selected)) {
      set_selection(SelectionKind::Scene, 0);
    }
    ImGui::PopStyleColor();

    ImGui::PushStyleColor(ImGuiCol_Text, camera_color);
    bool camera_selected = (_selection.kind == SelectionKind::Camera);
    if (ImGui::Selectable("Camera", camera_selected)) {
      set_selection(SelectionKind::Camera, 0);
    }
    ImGui::PopStyleColor();

    ImGui::Separator();

    ImGui::Text("Materials");
    if (_material_mapping.empty()) {
      ImGui::TextDisabled("None");
    } else if (ImGui::BeginListBox("##materials_list", ImVec2(-FLT_MIN, 6.0f * ImGui::GetTextLineHeightWithSpacing()))) {
      for (uint64_t i = 0; i < _material_mapping.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        bool material_selected = (_selection.kind == SelectionKind::Material) && (_selection.index == static_cast<int32_t>(i));
        if (ImGui::Selectable(_material_mapping.names[i], material_selected)) {
          set_selection(SelectionKind::Material, static_cast<int32_t>(i));
        }
        ImGui::PopID();
      }
      ImGui::EndListBox();
    }

    ImGui::Separator();

    ImGui::Text("Mediums");
    if (!scene_editable)
      ImGui::BeginDisabled();
    if (ImGui::Button("Add Medium", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
      if (callbacks.medium_added) {
        callbacks.medium_added();
        _medium_mapping.build(mediums);
        _medium_mapping_hash = hash_mapping(mediums);
      }
    }
    if (!scene_editable)
      ImGui::EndDisabled();
    ImGui::Spacing();
    if (_medium_mapping.empty()) {
      ImGui::TextDisabled("None");
    } else if (ImGui::BeginListBox("##mediums_list", ImVec2(-FLT_MIN, 4.0f * ImGui::GetTextLineHeightWithSpacing()))) {
      for (uint64_t i = 0; i < _medium_mapping.size(); ++i) {
        ImGui::PushID(static_cast<int>(i + 1024));
        bool medium_selected = (_selection.kind == SelectionKind::Medium) && (_selection.index == static_cast<int32_t>(i));
        if (ImGui::Selectable(_medium_mapping.names[i], medium_selected)) {
          set_selection(SelectionKind::Medium, static_cast<int32_t>(i));
        }
        ImGui::PopID();
      }
      ImGui::EndListBox();
    }

    ImGui::Separator();

    ImGui::Text("Emitters");
    if (scene.emitter_profiles.count == 0) {
      ImGui::TextDisabled("None");
    } else if (ImGui::BeginListBox("##emitters_list", ImVec2(-FLT_MIN, 4.0f * ImGui::GetTextLineHeightWithSpacing()))) {
      for (uint32_t emitter_index = 0; emitter_index < scene.emitter_profiles.count; ++emitter_index) {
        const auto& emitter = scene.emitter_profiles[emitter_index];
        char label[64] = {};
        switch (emitter.cls) {
          case EmitterProfile::Class::Area: {
            const char* material_name = nullptr;
            if (emitter_index < emitter_primary_instance.size()) {
              int32_t instance_index = emitter_primary_instance[emitter_index];
              if ((instance_index >= 0) && (static_cast<uint32_t>(instance_index) < scene.emitter_instances.count)) {
                const auto& instance = scene.emitter_instances[instance_index];
                if (instance.triangle_index < scene.triangles.count) {
                  uint32_t material_index = scene.triangles[instance.triangle_index].material_index;
                  material_name = material_name_from_index(material_index);
                  if (material_name == nullptr) {
                    snprintf(label, sizeof(label), "%05u : area (material %u)", emitter_index, material_index);
                    break;
                  }
                }
              }
            }
            if (material_name != nullptr) {
              snprintf(label, sizeof(label), "%05u : area (%s)", emitter_index, material_name);
            } else {
              snprintf(label, sizeof(label), "%05u : area", emitter_index);
            }
            break;
          }
          case EmitterProfile::Class::Directional:
            snprintf(label, sizeof(label), "%05u : directional", emitter_index);
            break;
          case EmitterProfile::Class::Environment:
            snprintf(label, sizeof(label), "%05u : environment", emitter_index);
            break;
          default:
            snprintf(label, sizeof(label), "%05u", emitter_index);
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

  if ((_ui_setup & UIProperties) == UIProperties) {
    std::string properties_title = "Properties";
    auto title_suffix = [&](const char* type, const char* name) {
      properties_title = type;
      if (name != nullptr && (name[0] != '\0')) {
        properties_title += std::string(": ") + name;
      }
    };

    if (has_scene) {
      switch (_selection.kind) {
        case SelectionKind::Material:
          if ((_selection.index >= 0) && (static_cast<uint64_t>(_selection.index) < _material_mapping.size())) {
            title_suffix("Material", _material_mapping.names[_selection.index]);
          }
          break;
        case SelectionKind::Medium:
          if ((_selection.index >= 0) && (static_cast<uint64_t>(_selection.index) < _medium_mapping.size())) {
            title_suffix("Medium", _medium_mapping.names[_selection.index]);
          }
          break;
        case SelectionKind::Emitter: {
          if ((_selection.index >= 0) && has_scene && (static_cast<uint32_t>(_selection.index) < scene.emitter_profiles.count)) {
            const auto& emitter = scene.emitter_profiles[_selection.index];
            char label[64] = {};
            const char* type = "Emitter";
            switch (emitter.cls) {
              case EmitterProfile::Class::Directional:
                snprintf(label, sizeof(label), "%05u (Directional)", static_cast<uint32_t>(_selection.index));
                break;
              case EmitterProfile::Class::Environment:
                snprintf(label, sizeof(label), "%05u (Environment)", static_cast<uint32_t>(_selection.index));
                break;
              default:
                snprintf(label, sizeof(label), "%05u", static_cast<uint32_t>(_selection.index));
                break;
            }
            title_suffix(type, label);
          }
          break;
        }
        case SelectionKind::Camera:
          title_suffix("Camera", nullptr);
          break;
        case SelectionKind::Scene:
          title_suffix("Scene", nullptr);
          break;
        default:
          break;
      }
    }

    std::string properties_window_name = properties_title + "###properties";
    with_window(UIProperties, properties_window_name.c_str(), [&]() {
      if (!has_scene) {
        ImGui::Text("No scene loaded");
        return;
      }

      if (!scene_editable) {
        ImGui::TextDisabled("Rendering in progress; editing disabled.");
      }

      switch (_selection.kind) {
        case SelectionKind::Material: {
          if (_material_mapping.empty()) {
            ImGui::Text("No materials available");
            break;
          }
          if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= _material_mapping.size())) {
            ImGui::Text("Invalid material selection");
            break;
          }
          uint32_t material_index = _material_mapping.at(_selection.index);
          Material& material = scene.materials[material_index];
          if (!scene_editable)
            ImGui::BeginDisabled();
          bool changed = build_material(scene, material);
          if (!scene_editable)
            ImGui::EndDisabled();
          if (scene_editable && changed && callbacks.material_changed) {
            callbacks.material_changed(material_index);
          }
          break;
        }

        case SelectionKind::Medium: {
          if (_medium_mapping.empty()) {
            ImGui::Text("No mediums available");
            break;
          }
          if ((_selection.index < 0) || (static_cast<uint64_t>(_selection.index) >= _medium_mapping.size())) {
            ImGui::Text("Invalid medium selection");
            break;
          }
          uint32_t medium_index = _medium_mapping.at(_selection.index);
          Medium& medium = scene.mediums[medium_index];
          if (!scene_editable)
            ImGui::BeginDisabled();
          bool changed = build_medium(scene, medium);
          if (!scene_editable)
            ImGui::EndDisabled();
          if (scene_editable && changed && callbacks.medium_changed) {
            callbacks.medium_changed(medium_index);
          }
          break;
        }

        case SelectionKind::Emitter: {
          if ((_selection.index < 0) || (!has_scene) || (static_cast<uint32_t>(_selection.index) >= scene.emitter_profiles.count)) {
            ImGui::Text("Invalid emitter selection");
            break;
          }
          uint32_t emitter_index = static_cast<uint32_t>(_selection.index);
          auto& emitter = scene.emitter_profiles[emitter_index];
          if (!scene_editable)
            ImGui::BeginDisabled();
          bool changed = false;
          bool material_changed = false;
          uint32_t material_index = kInvalidIndex;
          if (emitter.cls == EmitterProfile::Class::Area) {
            for (uint32_t instance_index = 0; instance_index < scene.emitter_instances.count; ++instance_index) {
              const auto& instance = scene.emitter_instances[instance_index];
              if (instance.profile != emitter_index) {
                continue;
              }
              if (instance.triangle_index < scene.triangles.count) {
                material_index = scene.triangles[instance.triangle_index].material_index;
              }
              break;
            }
            if (material_index < scene.materials.count) {
              auto& material = scene.materials[material_index];
              bool area_material_changed = false;
              if (_medium_mapping.empty() == false) {
                const char* mat_name = material_name_from_index(material_index);
                if (mat_name != nullptr) {
                  ImGui::Text("Material: %s", mat_name);
                } else {
                  ImGui::Text("Material index: %u", material_index);
                }
                ImGui::Text("External Medium");
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (medium_dropdown("##area_external_medium", material.ext_medium)) {
                  area_material_changed = true;
                }
              } else {
                ImGui::TextDisabled("No mediums available");
              }
              ImGui::Separator();

              ImGui::Text("Collimation");
              ImGui::SetNextItemWidth(-FLT_MIN);
              float collimation = material.emission_collimation;
              if (ImGui::SliderFloat("##area_collimation", &collimation, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
                material.emission_collimation = std::clamp(collimation, 0.0f, 1.0f);
                area_material_changed = true;
              }
              ImGui::Separator();

              std::string area_preset_id = "area_material_emission_" + std::to_string(material_index);
              area_material_changed |= emission_picker(scene, "Emission", area_preset_id.c_str(), material.emission.spectrum_index);
              if (area_material_changed) {
                material_changed = true;
                changed = true;
              }
            }
          }

          if (emitter.cls != EmitterProfile::Class::Area) {
            std::string emitter_preset_id = "emitter_emission_" + std::to_string(emitter_index);
            changed |= emission_picker(scene, "Emission", emitter_preset_id.c_str(), emitter.emission.spectrum_index);
            if (emitter.cls == EmitterProfile::Class::Directional) {
              ImGui::Text("Angular Size");
              if (ImGui::DragFloat("##angularsize", &emitter.angular_size, 0.01f, 0.0f, kHalfPi, "%.3f", ImGuiSliderFlags_NoRoundToFormat)) {
                emitter.angular_size_cosine = cosf(emitter.angular_size / 2.0f);
                emitter.equivalent_disk_size = 2.0f * std::tan(emitter.angular_size / 2.0f);
                changed = true;
              }

              bool direction_changed = false;
              auto s = to_spherical(emitter.direction);
              ImGui::Text("Rotation:");
              direction_changed = direction_changed || ImGui::DragFloat("##rotation", &s.phi, kPi / 360.0f, -kPi, kPi);
              ImGui::Text("Elevation:");
              direction_changed = direction_changed || ImGui::DragFloat("##elevation", &s.theta, kPi / 180.0f, -kHalfPi, kHalfPi);
              if (direction_changed) {
                emitter.direction = from_spherical(s);
                changed = true;
              }
            }
          }
          if (!scene_editable)
            ImGui::EndDisabled();
          if (scene_editable && material_changed && callbacks.material_changed && (material_index < scene.materials.count)) {
            callbacks.material_changed(material_index);
          }
          if (scene_editable && changed && callbacks.emitter_changed) {
            callbacks.emitter_changed(emitter_index);
          }
          break;
        }

        case SelectionKind::Camera: {
          if (_film == nullptr) {
            ImGui::Text("No camera available");
            break;
          }
          bool film_changed = false;
          bool camera_changed = false;

          int2 viewport = {int32_t(camera.film_size.x), int32_t(camera.film_size.y)};
          float3 pos = camera.position;
          float3 target = camera.position + camera.direction;
          float focal_len = get_camera_focal_length(camera);
          int32_t pixel_size = std::countr_zero(_film->pixel_size());

          if (!scene_editable)
            ImGui::BeginDisabled();
          ImGui::Text("Output Image Size:");
          if (ImGui::InputInt2("##outimgsize", &viewport.x)) {
            film_changed = true;
            camera_changed = true;
          }

          ImGui::Text("Lens Radius");
          camera_changed = camera_changed || ImGui::DragFloat("##lens", &camera.lens_radius, 0.01f, 0.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);
          ImGui::Text("Focal Distance");
          camera_changed = camera_changed || ImGui::DragFloat("##focaldist", &camera.focal_distance, 0.1f, 0.0f, 65536.0f, "%.3f", ImGuiSliderFlags_None);
          ImGui::Text("Focal Length");
          camera_changed = camera_changed || ImGui::DragFloat("##focal", &focal_len, 0.1f, 1.0f, 5000.0f, "%.1fmm", ImGuiSliderFlags_None);
          ImGui::Text("Clip Planes (near/far)");
          float clip_values[2] = {camera.clip_near, camera.clip_far};
          if (ImGui::DragFloat2("##clipplanes", clip_values, 0.01f, 0.0f, 5000.0f, "%.3f")) {
            camera.clip_near = std::max(0.0f, clip_values[0]);
            camera.clip_far = std::max(camera.clip_near + 0.001f, clip_values[1]);
            camera_changed = true;
          }

          ImGui::Text("Pixel Filter Radius");
          camera_changed = camera_changed || ImGui::DragFloat("##pixelfiler", &scene.pixel_sampler.radius, 0.05f, 0.0f, 32.0f, "%.3fpx", ImGuiSliderFlags_None);

          ImGui::Text("Pixel Size");
          camera_changed = camera_changed || ImGui::Combo("##pixelsize", &pixel_size, "Default\0Scaled 2x\0Scaled 4x\0Scaled 8x\0Scaled 16x\0");
          if (!scene_editable)
            ImGui::EndDisabled();

          if (scene_editable && camera_changed && callbacks.camera_changed) {
            _film->set_pixel_size(1u << pixel_size);

            viewport.x = clamp(viewport.x, 1, 1024 * 16);
            viewport.y = clamp(viewport.y, 1, 1024 * 16);
            camera.film_size = {uint32_t(viewport.x), uint32_t(viewport.y)};
            camera.lens_radius = fmaxf(camera.lens_radius, 0.0f);
            camera.focal_distance = fmaxf(camera.focal_distance, 0.0f);
            camera.clip_near = std::max(camera.clip_near, 0.0f);
            camera.clip_far = std::max(camera.clip_far, camera.clip_near + 0.001f);
            scene.pixel_sampler.radius = clamp(scene.pixel_sampler.radius, 0.0f, 32.0f);

            auto fov = focal_length_to_fov(focal_len) * 180.0f / kPi;
            build_camera(camera, pos, target, kWorldUp, camera.film_size, fov);

            callbacks.camera_changed(film_changed);
          }
          break;
        }

        case SelectionKind::Scene: {
          if (!scene_editable)
            ImGui::BeginDisabled();
          bool scene_settings_changed = false;

          ImGui::Text("Samples Per Pixel (Iterations):");
          scene_settings_changed = scene_settings_changed || ImGui::InputInt("##samples", reinterpret_cast<int*>(&scene.samples));

          ImGui::Text("Path Length (min/max):");
          int32_t min_value = int32_t(scene.min_path_length);
          int32_t max_value = int32_t(scene.max_path_length);
          scene_settings_changed = scene_settings_changed || ImGui::InputInt("##minpathlength", &min_value, 1, 10);
          scene_settings_changed = scene_settings_changed || ImGui::InputInt("##maxpathlength", &max_value, 1, 10);
          scene.min_path_length = clamp(min_value, 0, 65536);
          scene.max_path_length = clamp(max_value, 0, 65536);

          ImGui::Text("Min Path Length:");
          scene_settings_changed = scene_settings_changed || ImGui::InputInt("##bounces", reinterpret_cast<int*>(&scene.random_path_termination));
          ImGui::Text("Noise Threshold:");
          scene_settings_changed = scene_settings_changed || ImGui::InputFloat("##noiseth", &scene.noise_threshold, 0.0001f, 0.01f, "%0.5f");
          ImGui::Text("Radiance Clamp:");
          scene_settings_changed = scene_settings_changed || ImGui::InputFloat("##radclmp", &scene.radiance_clamp, 0.1f, 1.f, "%0.2f");
          ImGui::Text("Active pixels: %0.2f%%", double(_film->active_pixel_count()) / double(_film->pixel_count()) * 100.0);

          bool is_spectral = scene.spectral();
          scene_settings_changed = scene_settings_changed || ImGui::Checkbox("Spectral rendering", &is_spectral);
          scene.flags = (scene.flags & (~Scene::Spectral)) | (is_spectral ? Scene::Spectral : 0u);
          if (!scene_editable)
            ImGui::EndDisabled();

          if (scene_editable && scene_settings_changed) {
            scene.max_path_length = std::min(scene.max_path_length, 65536u);
            if (callbacks.scene_settings_changed) {
              callbacks.scene_settings_changed();
            }
          }
          break;
        }

        default:
          ImGui::Text("No Object Selected");
          break;
      }
    });
  }

  if (has_integrator && (_current_integrator->status().debug_info_count > 0) && (_current_integrator->status().debug_info != nullptr)) {
    if (ImGui::Begin("Debug Info", nullptr, kWindowFlags)) {
      auto debug_info = _current_integrator->status().debug_info;
      for (uint64_t i = 0, e = _current_integrator->status().debug_info_count; i < e; ++i) {
        char value_buffer[64] = {};
        snprintf(value_buffer, sizeof(value_buffer), "%.3f", debug_info[i].value);
        ImGui::LabelText(debug_info[i].title, "%s", value_buffer);
      }
      ImGui::End();
    }
  }

  simgui_render();
}

bool UI::handle_event(const sapp_event* e) {
  if (simgui_handle_event(e)) {
    return true;
  }

  if (e->type != SAPP_EVENTTYPE_KEY_DOWN) {
    return false;
  }

  auto modifiers = e->modifiers;
  bool has_alt = modifiers & SAPP_MODIFIER_ALT;
  bool has_shift = modifiers & SAPP_MODIFIER_SHIFT;

  if ((modifiers & SAPP_MODIFIER_CTRL) || (modifiers & SAPP_MODIFIER_SUPER)) {
    switch (e->key_code) {
      case SAPP_KEYCODE_Q: {
        quit();
        break;
      }
      case SAPP_KEYCODE_O: {
        select_scene_file();
        break;
      }
      case SAPP_KEYCODE_I: {
        load_image();
        break;
      }
      case SAPP_KEYCODE_R: {
        if (has_shift) {
          if (callbacks.use_image_as_reference) {
            callbacks.use_image_as_reference();
          }
        } else {
          reload_scene();
        }
        break;
      }
      case SAPP_KEYCODE_G: {
        reload_geometry();
        break;
      }
      case SAPP_KEYCODE_S: {
        if (has_alt && (has_shift == false)) {
          // TODO : use
        } else if (has_shift && (has_alt == false)) {
          save_image(SaveImageMode::TonemappedLDR);
        } else if ((has_shift == false) && (has_alt == false)) {
          save_image(SaveImageMode::RGB);
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
      _view_options.view = static_cast<OutputView>(e->key_code - SAPP_KEYCODE_1);
      break;
    }
    case SAPP_KEYCODE_KP_DIVIDE: {
      decrease_exposure(_view_options);
      break;
    }
    case SAPP_KEYCODE_KP_MULTIPLY: {
      increase_exposure(_view_options);
      break;
    }
    default:
      break;
  }

  return false;
}

ViewOptions UI::view_options() const {
  return _view_options;
}

ViewOptions& UI::mutable_view_options() {
  return _view_options;
}

void UI::set_current_integrator(Integrator* i) {
  _current_integrator = i;
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

bool UI::build_material(Scene& scene, Material& material) {
  bool changed = false;

  char buffer[64] = {};
  snprintf(buffer, sizeof(buffer), "%s", material_class_to_string(material.cls));
  buffer[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(buffer[0])));
  ImVec2 button_size = ImVec2(ImGui::GetContentRegionAvail().x, 0.0f);
  char button_label[128] = {};
  snprintf(button_label, sizeof(button_label), "%s##material_class", buffer);
  if (ImGui::Button(button_label, button_size)) {
    ImGui::OpenPopup("material_class_popup");
  }

  ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 28.0f, 0.0f), ImGuiCond_Always);
  if (ImGui::BeginPopup("material_class_popup")) {
    const ImVec4 header_colors[] = {
      {0.96f, 0.79f, 0.45f, 1.0f},
      {0.54f, 0.80f, 0.98f, 1.0f},
      {0.88f, 0.68f, 0.97f, 1.0f},
    };

    ImGui::Columns(3, "material_class_columns", true);

    auto draw_material_column = [&](uint32_t column_index, const char* title, std::initializer_list<Material::Class> entries) {
      ImGui::PushStyleColor(ImGuiCol_Text, header_colors[column_index % (sizeof(header_colors) / sizeof(header_colors[0]))]);
      ImGui::Text("%s", title);
      ImGui::PopStyleColor();
      for (auto cls : entries) {
        snprintf(buffer, sizeof(buffer), "%s", material_class_to_string(cls));
        buffer[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(buffer[0])));
        char selectable_label[128] = {};
        snprintf(selectable_label, sizeof(selectable_label), "%s##cls_%u", buffer, static_cast<uint32_t>(cls));
        bool is_selected = (material.cls == cls);
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
    draw_material_column(column_index++, "Primary", {Material::Class::Diffuse, Material::Class::Plastic, Material::Class::Conductor, Material::Class::Dielectric});
    ImGui::NextColumn();
    draw_material_column(column_index++, "Specialized",
      {Material::Class::Principled, Material::Class::Translucent, Material::Class::Thinfilm, Material::Class::Velvet, Material::Class::Mirror});
    ImGui::NextColumn();
    draw_material_column(column_index++, "Interfaces", {Material::Class::Boundary, Material::Class::Void});

    ImGui::Columns(1);
    ImGui::EndPopup();
  }

  if (material.has_diffuse()) {
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Diffuse Type:");
    ImGui::SameLine();
    int dv = static_cast<int>(material.diffuse_variation);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::Combo("##diff_var", &dv, "Lambert\0Microfacet\0vMF\0")) {
      dv = clamp(dv, 0, 2);
      if (material.diffuse_variation != static_cast<uint32_t>(dv)) {
        material.diffuse_variation = static_cast<uint32_t>(dv);
        changed = true;
      }
    }
  }

  // section background colors (slight tints from window bg)
  ImVec4 base_bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
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

  // Roughness section
  auto brighten = [&](const ImVec4& c, float d) {
    return ImVec4{clamp01(c.x + d), clamp01(c.y + d), clamp01(c.z + d), c.w};
  };
  ImGui::PushStyleColor(ImGuiCol_Header, sec_col[0]);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brighten(sec_col[0], 0.04f));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, brighten(sec_col[0], 0.08f));
  bool open_rough = ImGui::CollapsingHeader("Roughness", ImGuiTreeNodeFlags_DefaultOpen);
  ImGui::PopStyleColor(3);
  if (open_rough) {
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.75f);
    changed |= ImGui::SliderFloat("##r_u", &material.roughness.value.x, 0.0f, 1.0f, "Roughness U %.3f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.75f);
    changed |= ImGui::SliderFloat("##r_v", &material.roughness.value.y, 0.0f, 1.0f, "Roughness V %.3f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.25f);
    if (ImGui::Button("Sync")) {
      material.roughness.value.y = material.roughness.value.x;
      changed = true;
    }
  }

  // Principled-specific parameters
  if (material.cls == Material::Class::Principled) {
    ImGui::PushStyleColor(ImGuiCol_Header, sec_col[1]);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brighten(sec_col[1], 0.04f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, brighten(sec_col[1], 0.08f));
    bool open_principled = ImGui::CollapsingHeader("Metalness / Transmission", ImGuiTreeNodeFlags_DefaultOpen);
    ImGui::PopStyleColor(3);
    if (open_principled) {
      float metal = material.metalness.value.x;
      float trans = material.transmission.value.x;
      ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.75f);
      if (ImGui::SliderFloat("##metalness", &metal, 0.0f, 1.0f, "Metalness %.3f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat)) {
        material.metalness.value = {metal, metal, metal, metal};
        changed = true;
      }
      ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.75f);
      if (ImGui::SliderFloat("##transmission", &trans, 0.0f, 1.0f, "Transmission %.3f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat)) {
        material.transmission.value = {trans, trans, trans, trans};
        changed = true;
      }
    }
  }

  ImGui::PushStyleColor(ImGuiCol_Header, sec_col[1]);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brighten(sec_col[1], 0.04f));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, brighten(sec_col[1], 0.08f));
  bool open_ior = ImGui::CollapsingHeader("Index Of Refraction (in/out)", ImGuiTreeNodeFlags_DefaultOpen);
  ImGui::PopStyleColor(3);
  if (open_ior) {
    ImVec2 old_cell_padding = ImGui::GetStyle().CellPadding;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(1.0f, old_cell_padding.y));
    if (ImGui::BeginTable("ior_inout", 2, ImGuiTableFlags_SizingStretchSame)) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::PushItemWidth(-FLT_MIN);
      changed |= ior_picker(scene, "Inside", material.int_ior);
      ImGui::PopItemWidth();
      ImGui::TableSetColumnIndex(1);
      ImGui::PushItemWidth(-FLT_MIN);
      changed |= ior_picker(scene, "Outside", material.ext_ior);
      ImGui::PopItemWidth();
      ImGui::EndTable();
    }
    ImGui::PopStyleVar();
  }

  ImGui::PushStyleColor(ImGuiCol_Header, sec_col[2]);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brighten(sec_col[2], 0.04f));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, brighten(sec_col[2], 0.08f));
  bool open_refl = ImGui::CollapsingHeader("Reflectance / Scattering", ImGuiTreeNodeFlags_DefaultOpen);
  ImGui::PopStyleColor(3);
  if (open_refl) {
    changed |= spectrum_picker(scene, "Reflectance", material.reflectance.spectrum_index, false, false);
    changed |= spectrum_picker(scene, "Scattering", material.scattering.spectrum_index, false, false);
  }

  ImGui::PushStyleColor(ImGuiCol_Header, sec_col[4]);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brighten(sec_col[4], 0.04f));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, brighten(sec_col[4], 0.08f));
  bool open_emission = ImGui::CollapsingHeader("Emission", ImGuiTreeNodeFlags_DefaultOpen);
  ImGui::PopStyleColor(3);
  if (open_emission) {
    ImGui::Text("Collimation");
    ImGui::SetNextItemWidth(-FLT_MIN);
    float collimation = material.emission_collimation;
    if (ImGui::SliderFloat("##material_emission_collimation", &collimation, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp)) {
      material.emission_collimation = std::clamp(collimation, 0.0f, 1.0f);
      changed = true;
    }

    std::string preset_id = "material_emission_" + std::to_string(material.emission.spectrum_index);
    changed |= emission_picker(scene, "Emission", preset_id.c_str(), material.emission.spectrum_index);
  }

  // Opacity
  ImGui::PushStyleColor(ImGuiCol_Header, sec_col[0]);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brighten(sec_col[0], 0.04f));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, brighten(sec_col[0], 0.08f));
  bool open_opacity = ImGui::CollapsingHeader("Opacity", ImGuiTreeNodeFlags_DefaultOpen);
  ImGui::PopStyleColor(3);
  if (open_opacity) {
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() * 0.75f);
    changed |= ImGui::SliderFloat("##opacity", &material.opacity, 0.0f, 1.0f, "Opacity %.3f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
  }

  ImGui::PushStyleColor(ImGuiCol_Header, sec_col[3]);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brighten(sec_col[3], 0.04f));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, brighten(sec_col[3], 0.08f));
  bool open_sss = ImGui::CollapsingHeader("Subsurface Scattering", ImGuiTreeNodeFlags_DefaultOpen);
  ImGui::PopStyleColor(3);
  if (open_sss) {
    changed |= ImGui::Combo("##sssclass", reinterpret_cast<int*>(&material.subsurface.cls), "Disabled\0Random Walk\0Christensen-Burley\0");
    changed |= ImGui::Combo("##ssspath", reinterpret_cast<int*>(&material.subsurface.path), "Diffuse Transmittance\0Refraction\0");
    changed |= spectrum_picker(scene, "Subsurface Distance", material.subsurface.spectrum_index, true, true);
  }

  ImGui::PushStyleColor(ImGuiCol_Header, sec_col[4]);
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brighten(sec_col[4], 0.04f));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, brighten(sec_col[4], 0.08f));
  bool open_thin = ImGui::CollapsingHeader("Thinfilm", ImGuiTreeNodeFlags_DefaultOpen);
  ImGui::PopStyleColor(3);
  if (open_thin) {
    ImGui::AlignTextToFramePadding();
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() / 2.0f);
    ImGui::Text("%s", "IoR:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() / 2.0f);
    changed |= ior_picker(scene, "Thinfilm IoR", material.thinfilm.ior);
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() / 3.0f);
    changed |= ImGui::InputFloat("##tftmin", &material.thinfilm.min_thickness);
    ImGui::SameLine();
    ImGui::Text(" - ");
    ImGui::SetNextItemWidth(ImGui::GetWindowWidth() / 3.0f);
    ImGui::SameLine();
    changed |= ImGui::InputFloat("##tftmax", &material.thinfilm.max_thickness);
    ImGui::SameLine();
    ImGui::Text("nm");
  }

  if (_medium_mapping.empty() == false) {
    ImGui::PushStyleColor(ImGuiCol_Header, sec_col[5]);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, brighten(sec_col[5], 0.04f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, brighten(sec_col[5], 0.08f));
    bool open_med = ImGui::CollapsingHeader("Medium (internal/external)", ImGuiTreeNodeFlags_DefaultOpen);
    ImGui::PopStyleColor(3);
    if (open_med) {
      ImVec2 old_cell_padding = ImGui::GetStyle().CellPadding;
      ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(1.0f, old_cell_padding.y));
      if (ImGui::BeginTable("medium_inout", 2, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= medium_dropdown("##internal_medium", material.int_medium);
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-FLT_MIN);
        changed |= medium_dropdown("##external_medium", material.ext_medium);
        ImGui::EndTable();
      }
      ImGui::PopStyleVar();
    }
  }

  return changed;
}

bool UI::build_medium(Scene& scene, Medium& m) {
  bool changed = false;

  if (scene.spectrums.count == 0) {
    return changed;
  }

  auto ensure_index = [&](uint32_t& index, uint32_t fallback) {
    if ((index == kInvalidIndex) || (index >= scene.spectrums.count)) {
      index = fallback;
    }
  };

  ensure_index(m.absorption_index, scene.black_spectrum);
  ensure_index(m.scattering_index, scene.black_spectrum);

  constexpr const char* kMediumClassNames[] = {"Homogeneous Medium", "Density Grid Medium"};
  const int32_t class_count = static_cast<int32_t>(sizeof(kMediumClassNames) / sizeof(kMediumClassNames[0]));
  int32_t class_index = static_cast<int32_t>(m.cls);
  class_index = clamp(class_index, 0, class_count - 1);

  bool has_density_grid = m.density.count != 0;
  bool recompute_extinction = false;
  float updated_max_sigma = m.max_sigma;

  ImGui::Text("%s", kMediumClassNames[class_index]);

  ImGui::Text("Absorption");
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (spectrum_picker(scene, "Absorption", m.absorption_index, true, true)) {
    changed = true;
    recompute_extinction = true;
  }

  ImGui::Text("Scattering");
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (spectrum_picker(scene, "Scattering", m.scattering_index, true, true)) {
    changed = true;
    recompute_extinction = true;
  }

  if (recompute_extinction) {
    updated_max_sigma = 0.0f;
    if (m.absorption_index < scene.spectrums.count) {
      updated_max_sigma += scene.spectrums[m.absorption_index].maximum_spectral_power();
    }
    if (m.scattering_index < scene.spectrums.count) {
      updated_max_sigma += scene.spectrums[m.scattering_index].maximum_spectral_power();
    }
  }

  ImGui::Text("Anisotropy");
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::SliderFloat("##medium_phase_g", &m.phase_function_g, -0.999f, 0.999f, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
    changed = true;
  }

  bool explicit_connections = (m.enable_explicit_connections != 0u);
  if (ImGui::Checkbox("Explicit connections##medium_explicit_connections", &explicit_connections)) {
    m.enable_explicit_connections = explicit_connections ? 1u : 0u;
    changed = true;
  }

  if (has_density_grid) {
    ImGui::Text("Density grid");
    ImGui::Text("%u x %u x %u", m.dimensions.x, m.dimensions.y, m.dimensions.z);
    ImGui::Text("Bounds");
    ImGui::Text("min (%.3f, %.3f, %.3f)  max (%.3f, %.3f, %.3f)", m.bounds.p_min.x, m.bounds.p_min.y, m.bounds.p_min.z, m.bounds.p_max.x, m.bounds.p_max.y, m.bounds.p_max.z);
  }

  if (changed) {
    m.max_sigma = updated_max_sigma;
  }

  return changed;
}

void UI::reset_selection() {
  _selection = {};
  _selection_history.clear();
  _selection_history_cursor = -1;
  _spectrum_editors.clear();
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

}  // namespace etx
