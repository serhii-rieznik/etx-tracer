#include "ui.hxx"
#include <etx/core/platform.hxx>
#include <imgui.h>
#include <etx/render/shared/ior_database.hxx>
#include <algorithm>
#include <cstring>
#include <cstdio>

namespace etx {
namespace {
bool same_spectrum(const SpectralDistribution& a, const SpectralDistribution& b) {
  return (a.spectral_entry_count == b.spectral_entry_count) && (std::memcmp(&a.integrated_value, &b.integrated_value, sizeof(a.integrated_value)) == 0) &&
         (std::memcmp(a.spectral_entries, b.spectral_entries, a.spectral_entry_count * sizeof(SpectralDistribution::Entry)) == 0);
}
SpectrumDocument source_document(const SpectrumSource& source) {
  SpectrumDocument document;
  document.title = source.title;
  document.path = source.path;
  document.classification = source.classification;
  document.points = source.points;
  if (document.points.empty()) {
    for (uint32_t i = 0u; i < source.base.spectral_entry_count; ++i) {
      document.points.push_back({source.base.spectral_entries[i].wavelength, source.base.spectral_entries[i].power});
    }
  }
  return document;
}
void set_document(SpectrumSource& source, const SpectrumDocument& document) {
  source.mode = SpectrumSource::Mode::Spectrum;
  source.points = document.points;
  source.title = document.title;
  source.path = document.path;
  source.classification = document.classification;
  source.base = document.distribution();
  if (source.kind == SpectrumSource::Kind::IOR) {
    source.base.integrated_value = rgb_to_xyz(source.base.integrated());
  }
}
}  // namespace

bool UI::material_spectrum_control(SceneRepresentation& scene, const char* label, SpectrumTarget::Channel channel, SpectrumSource::Kind kind) {
  std::vector<SpectrumTarget> targets;
  if (_editing_material_indices != nullptr) {
    for (uint32_t index : *_editing_material_indices) {
      SpectrumTarget target{.material_index = index, .channel = channel};
      if (const uint32_t* slot = target.spectrum_slot(scene.data()); slot != nullptr) {
        target.spectrum_index = *slot;
        targets.push_back(target);
      }
    }
  }
  return spectrum_control(scene, label, targets, kind, false);
}

void UI::edit_spectrum_button(SceneRepresentation& scene, const char* label, SpectrumTarget::Channel channel, uint32_t spectrum_index) {
  (void)spectrum_index;
  if (ImGui::SmallButton(label)) {
    _spectrum_targets.clear();
    if (_editing_material_indices != nullptr) {
      for (uint32_t index : *_editing_material_indices) {
        SpectrumTarget target{.material_index = index, .channel = channel};
        if (const uint32_t* slot = target.spectrum_slot(scene.data()); slot != nullptr) {
          target.spectrum_index = *slot;
          _spectrum_targets.push_back(target);
        }
      }
    }
    _spectrum_kind = SpectrumSource::Kind::IOR;
    _spectrum_label = label;
    if (const auto separator = _spectrum_label.find("##"); separator != std::string::npos)
      _spectrum_label.resize(separator);
    _spectrum_tab_requested = true;
    _ui_setup |= UIMemoryDiagnostics;
  }
}

bool UI::spectrum_control(SceneRepresentation& scene, const char* label, const std::vector<SpectrumTarget>& targets, SpectrumSource::Kind kind, bool expanded) {
  SceneData& data = scene.data();
  if (targets.empty()) {
    ImGui::TextDisabled("Select a spectrum to edit.");
    return false;
  }
  const uint32_t* slot = targets.front().spectrum_slot(data);
  if ((slot == nullptr) || (*slot != targets.front().spectrum_index) || (*slot >= data.spectrum_values.size())) {
    ImGui::TextDisabled("This spectrum is no longer available. Select it again.");
    return false;
  }
  const uint32_t index = *slot;
  const SpectralDistribution& actual = data.spectrum_values[index];
  auto& state = _spectrum_controls[index];
  if ((state.initialized == false) || ((state.pending == false) && (same_spectrum(state.observed, actual) == false))) {
    state = {};
    state.initialized = true;
    state.observed = actual;
    if (const auto found = data.spectrum_sources.find(index); (found != data.spectrum_sources.end()) && found->second.matches(actual)) {
      state.source = found->second;
    } else {
      state.source.base = actual;
      float3 color = max(actual.integrated(), float3{0.0f});
      const float peak = max(color.x, max(color.y, color.z));
      state.source.color = peak > 1.0f ? color / peak : color;
    }
    state.source.kind = kind;
    if (actual.empty())
      state.source.base = SpectralDistribution::constant(0.0f);
    if ((kind == SpectrumSource::Kind::IOR) && (state.source.mode == SpectrumSource::Mode::Spectrum) && (actual.spectral_entry_count > 0u)) {
      const float value = actual.spectral_entries[actual.spectral_entry_count / 2u].power;
      state.source.color = {value, value, value};
    }
    state.custom = state.source;
    state.custom.mode = SpectrumSource::Mode::Spectrum;
    state.curve.document = source_document(state.custom);
    state.curve.select_point(0);
    state.curve.fit();
  }
  ImGui::PushID(label);
  ImGui::PushID(static_cast<int>(index));
  auto& source = state.source;
  bool changed = false;
  bool generated_changed = false;
  state.interacting = false;
  const bool numeric = kind == SpectrumSource::Kind::IOR;
  const bool temperature_available = (kind == SpectrumSource::Kind::Reflectance) || (kind == SpectrumSource::Kind::Emission);
  bool mixed = false;
  for (const auto& target : targets) {
    const uint32_t* other = target.spectrum_slot(data);
    if ((other == nullptr) || (*other >= data.spectrum_values.size()) || (same_spectrum(actual, data.spectrum_values[*other]) == false))
      mixed = true;
  }
  const auto select_mode = [&](SpectrumSource::Mode mode) {
    if (source.mode == mode)
      return;
    const float strength = source.strength;
    const float3 color = source.color;
    const float temperature = source.temperature;
    if (source.mode == SpectrumSource::Mode::Spectrum)
      state.custom = source;
    if (mode == SpectrumSource::Mode::Spectrum) {
      source = state.custom;
      state.curve.document = source_document(source);
      state.curve.select_point(0);
      state.curve.fit();
    }
    source.color = color;
    source.temperature = temperature;
    source.strength = strength;
    source.mode = mode;
    generated_changed = mode != SpectrumSource::Mode::Spectrum;
    changed = true;
  };
  const auto load_source = [&](const std::string& path) {
    if (path.empty())
      return;
    SpectrumDocument loaded;
    if (loaded.load(path, state.error)) {
      set_document(source, loaded);
      state.curve.document = loaded;
      state.curve.select_point(0);
      state.curve.fit();
      state.curve.modified = false;
      changed = true;
    }
  };
  const float available = ImGui::GetContentRegionAvail().x;
  const float font = ImGui::GetFontSize();
  const float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
  const float frame_height = ImGui::GetFrameHeight();
  const float compact_font_size = ImGui::GetStyle().FontSizeBase * 0.75f;
  const float compact_padding = ImGui::GetStyle().FramePadding.x * 1.25f;
  char strength_text[64];
  std::snprintf(strength_text, sizeof(strength_text), "x %.4f", source.strength);
  ImGui::PushFont(nullptr, compact_font_size);
  const float strength_width = std::min(available * 0.35f, std::ceil(ImGui::CalcTextSize(strength_text).x + compact_padding * 2.0f));
  const float more_width = expanded ? 0.0f : std::ceil(ImGui::CalcTextSize("...").x + compact_padding * 2.0f);
  ImGui::PopFont();
  const float mode_width = std::min(available * 0.20f, font * 5.0f);
  const float value_width = std::max(1.0f, available - mode_width - strength_width - more_width - spacing * (expanded ? 2.0f : 3.0f));
  const bool warning = (kind == SpectrumSource::Kind::Reflectance) && (source.output().maximum_spectral_power() > 1.0f);
  if ((state.error.empty() == false) || warning)
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.7f, 0.3f, 1.0f));
  ImGui::TextUnformatted(label);
  if ((state.error.empty() == false) || warning)
    ImGui::PopStyleColor();
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::TextUnformatted(label);
    if (mixed)
      ImGui::TextUnformatted("Mixed values. Edits apply to all selected materials.");
    if (warning)
      ImGui::TextUnformatted("Some reflectance values exceed 1.");
    if (state.error.empty() == false)
      ImGui::TextWrapped("%s", state.error.c_str());
    ImGui::EndTooltip();
  }
  ImGui::SetNextItemWidth(mode_width);
  const char* mode_name = mixed                                              ? "Mixed"
                          : source.mode == SpectrumSource::Mode::Spectrum    ? "SPD"
                          : source.mode == SpectrumSource::Mode::Temperature ? "Temp"
                          : numeric                                          ? "Value"
                                                                             : "Color";
  if (ImGui::BeginCombo("##mode", mode_name)) {
    if (ImGui::Selectable(numeric ? "Value" : "Color", source.mode == SpectrumSource::Mode::Color))
      select_mode(SpectrumSource::Mode::Color);
    if (temperature_available && ImGui::Selectable("Temperature", source.mode == SpectrumSource::Mode::Temperature))
      select_mode(SpectrumSource::Mode::Temperature);
    if (ImGui::Selectable("Spectrum", source.mode == SpectrumSource::Mode::Spectrum))
      select_mode(SpectrumSource::Mode::Spectrum);
    ImGui::EndCombo();
  }
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Spectrum source mode");
  ImGui::SameLine(0.0f, spacing);
  ImGui::SetNextItemWidth(value_width);
  if (source.mode == SpectrumSource::Mode::Color) {
    if (numeric) {
      float value = source.color.x;
      if (ImGui::DragFloat("##value", &value, 0.01f, 0.0f, 0.0f, "%.6g") && std::isfinite(value) && (value >= 0.0f)) {
        source.color = {value, value, value};
        generated_changed = true;
      }
      state.interacting |= ImGui::IsItemActive();
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Wavelength-independent IOR value");
    } else {
      float3 display = kind == SpectrumSource::Kind::Coefficient ? source.color : linear_to_gamma(source.color);
      if (ImGui::ColorButton("##color", ImVec4(display.x, display.y, display.z, 1.0f), ImGuiColorEditFlags_NoTooltip, ImVec2(value_width, ImGui::GetFrameHeight())))
        ImGui::OpenPopup("color");
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip(kind == SpectrumSource::Kind::Coefficient ? "Edit linear RGB values" : "Edit sRGB color");
      if (ImGui::BeginPopup("color")) {
        ImGui::SetNextItemWidth(font * 10.0f);
        if (ImGui::ColorPicker3("##picker", &display.x,
              ImGuiColorEditFlags_Float | ImGuiColorEditFlags_InputRGB | (kind == SpectrumSource::Kind::Coefficient ? ImGuiColorEditFlags_HDR : 0))) {
          if (std::isfinite(display.x) && std::isfinite(display.y) && std::isfinite(display.z)) {
            display = max(display, float3{0.0f});
            source.color = kind == SpectrumSource::Kind::Coefficient ? display : gamma_to_linear(display);
            generated_changed = true;
          }
        }
        state.interacting |= ImGui::IsItemActive();
        ImGui::EndPopup();
      }
    }
  } else if (source.mode == SpectrumSource::Mode::Temperature) {
    float temperature = source.temperature;
    if (ImGui::DragFloat("##temperature", &temperature, 10.0f, 1000.0f, 40000.0f, "%.0f K", ImGuiSliderFlags_AlwaysClamp) && std::isfinite(temperature)) {
      source.temperature = temperature;
      generated_changed = true;
    }
    state.interacting |= ImGui::IsItemActive();
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip(kind == SpectrumSource::Kind::Emission ? "Blackbody emission temperature" : "Temperature color for reflectance");
  } else {
    if (ImGui::Button((source.title + "##source").c_str(), ImVec2(value_width, 0.0f)))
      ImGui::OpenPopup("source");
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("%s\n%s", source.title.c_str(), source.path.empty() ? "Choose a file or preset" : source.path.c_str());
    ImGui::SetNextWindowSizeConstraints(ImVec2(font * 20.0f, 0.0f), ImVec2(font * 32.0f, font * 18.0f));
    if (ImGui::BeginPopup("source")) {
      if (ImGui::Selectable("Load spectrum file..."))
        load_source(open_file("spd", nullptr));
      ImGui::Separator();
      ImGui::SetNextItemWidth(-FLT_MIN);
      ImGui::InputTextWithHint("##filter", "Search presets...", state.preset_filter, sizeof(state.preset_filter));
      const ImGuiTextFilter filter(state.preset_filter);
      bool any = false;
      if ((_spectrum_database != nullptr) && (numeric == false)) {
        const auto classification = kind == SpectrumSource::Kind::Emission ? SpectralDistribution::Illuminant : SpectralDistribution::Reflectance;
        for (size_t entry : _spectrum_database->class_entries(classification)) {
          const auto& preset = _spectrum_database->definitions[entry];
          if (filter.PassFilter(preset.title.c_str()) == false)
            continue;
          any = true;
          if (ImGui::Selectable(preset.title.c_str()))
            load_source(preset.filename);
        }
      }
      if (any == false)
        ImGui::TextDisabled(numeric ? "Use the IOR picker for paired presets." : "No matching presets.");
      ImGui::EndPopup();
    }
  }
  ImGui::SameLine(0.0f, spacing);
  ImGui::SetNextItemWidth(strength_width);
  const ImVec2 frame_padding = ImGui::GetStyle().FramePadding;
  ImGui::PushFont(nullptr, compact_font_size);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(compact_padding, frame_padding.y + (font - ImGui::GetFontSize()) * 0.5f));
  float strength = source.strength;
  if (ImGui::DragFloat("##strength", &strength, std::max(0.001f, source.strength * 0.01f), 0.0f, FLT_MAX, "x %.4f",
        ImGuiSliderFlags_NoRoundToFormat | ImGuiSliderFlags_AlwaysClamp) &&
      std::isfinite(strength) && (strength >= 0.0f)) {
    source.strength = strength;
    changed = true;
  }
  state.interacting |= ImGui::IsItemActive();
  ImGui::PopStyleVar();
  ImGui::PopFont();
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Strength multiplier. Zero preserves the source. Double-click or Ctrl-click to type a value.");
  if (expanded == false) {
    ImGui::SameLine(0.0f, spacing);
    ImGui::PushFont(nullptr, compact_font_size);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(compact_padding, frame_padding.y));
    if (ImGui::Button("...", ImVec2(more_width, frame_height))) {
      _spectrum_targets = targets;
      _spectrum_kind = kind;
      _spectrum_label = label;
      _spectrum_tab_requested = true;
      _ui_setup |= UIMemoryDiagnostics;
    }
    ImGui::PopStyleVar();
    ImGui::PopFont();
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Open spectrum editor");
  }
  if (generated_changed) {
    source.generate();
    source.classification = kind == SpectrumSource::Kind::Emission ? "illuminant" : kind == SpectrumSource::Kind::Reflectance ? "reflectance" : "";
    source.points.clear();
    source.path.clear();
    source.title = source.mode == SpectrumSource::Mode::Temperature ? "Temperature" : numeric ? "Constant value" : "RGB color";
    changed = true;
  }
  if (source.output().empty()) {
    state.error = "These values exceed the supported numeric range.";
    changed = false;
  }
  if (expanded) {
    ImGui::SeparatorText("Source curve (before strength)");
    const auto& points = state.curve.document.points;
    bool same_points = false;
    if (source.points.empty()) {
      same_points = points.size() == source.base.spectral_entry_count;
      for (size_t i = 0; same_points && (i < points.size()); ++i) {
        same_points = (points[i].x == source.base.spectral_entries[i].wavelength) && (points[i].y == source.base.spectral_entries[i].power);
      }
    } else {
      same_points = (source.points.size() == points.size()) && std::equal(source.points.begin(), source.points.end(), points.begin(), [](const float2& a, const float2& b) {
        return (a.x == b.x) && (a.y == b.y);
      });
    }
    if ((same_points == false) || (state.curve.document.title != source.title) || (state.curve.document.path != source.path) ||
        (state.curve.document.classification != source.classification)) {
      state.curve.document = source_document(source);
      state.curve.select_point(0);
      state.curve.fit();
    }
    if (state.curve.build()) {
      if (source.mode != SpectrumSource::Mode::Spectrum)
        state.curve.document.title = "Custom spectrum";
      set_document(source, state.curve.document);
      changed = true;
    }
    state.interacting |= ImGui::IsMouseDown(ImGuiMouseButton_Left);
  }
  if (changed) {
    if (source.mode == SpectrumSource::Mode::Spectrum)
      state.custom = source;
    state.targets = targets;
    state.pending = true;
    state.error.clear();
  }
  ImGui::PopID();
  ImGui::PopID();
  // Application updates run after the UI has released its references into scene arrays.
  return false;
}

void UI::flush_spectrum_changes(SceneRepresentation& scene) {
  if (preparation_active())
    return;
  std::vector<SpectrumEdit> edits;
  std::vector<uint32_t> indices;
  for (auto& [index, state] : _spectrum_controls) {
    if (state.pending == false)
      continue;
    if ((state.source.kind == SpectrumSource::Kind::IOR) && (state.interacting || ImGui::IsMouseDown(ImGuiMouseButton_Left))) {
      state.interacting = false;
      continue;
    }
    edits.push_back({state.targets, state.source});
    indices.push_back(index);
  }
  if (edits.empty())
    return;
  const bool applied = callbacks.spectrum_applied && callbacks.spectrum_applied(edits);
  for (uint32_t index : indices) {
    auto& state = _spectrum_controls.at(index);
    if (applied) {
      if (index < scene.data().spectrum_values.size())
        state.observed = scene.data().spectrum_values[index];
      state.error.clear();
    } else {
      state.error = "The spectrum could not be applied. Check its values or select the target again.";
    }
    state.pending = false;
  }
}

void UI::build_spectrum_editor(SceneRepresentation& scene) {
  if (_spectrum_targets.empty()) {
    ImGui::TextWrapped("Click ... on a spectrum control to open its editor here.");
    return;
  }
  spectrum_control(scene, _spectrum_label.c_str(), _spectrum_targets, _spectrum_kind, true);
}
}  // namespace etx
