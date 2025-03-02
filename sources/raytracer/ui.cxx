#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>

#include <etx/render/host/film.hxx>

#include "ui.hxx"

#include <sokol_app.h>
#include <sokol_gfx.h>

#include <imgui.h>
#include <imgui_internal.h>
#include <util/sokol_imgui.h>

#if (ETX_PLATFORM_APPLE)
# include <unistd.h>
#endif

#include <algorithm>
#include <filesystem>

namespace etx {

namespace {

inline void decrease_exposure(ViewOptions& o) {
  o.exposure = fmaxf(1.0f / 1024.0f, 0.5f * o.exposure);
}

inline void increase_exposure(ViewOptions& o) {
  o.exposure = fmaxf(1.0f / 1024.0f, 2.0f * o.exposure);
}

}  // namespace

void UI::MappingRepresentation::build(const std::unordered_map<std::string, uint32_t>& mapping) {
  constexpr uint64_t kMaterialNameMaxSize = 1024llu;

  indices.clear();
  indices.reserve(mapping.size());
  names.clear();
  names.reserve(mapping.size());
  data.resize(mapping.size() * kMaterialNameMaxSize);
  std::fill(data.begin(), data.end(), 0);
  char* ptr = data.data();
  int32_t pp = 0;
  for (auto& m : mapping) {
    indices.emplace_back(m.second);
    names.emplace_back(ptr + pp);
    pp += snprintf(ptr + pp, kMaterialNameMaxSize, "%s", m.first.c_str()) + 1;
  }
}

void UI::initialize() {
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

  _ior_files.clear();

  std::string path = env().file_in_data("spectrum/");
  for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
    auto filename = entry.path().filename();
    const auto& ext = entry.path().extension();
    if (ext != L".spd")
      continue;

    auto cls = RefractiveIndex::load_from_file(entry.path().string().c_str()).cls;
    if (cls == SpectralDistribution::Class::Invalid)
      continue;

    auto& e = _ior_files.emplace_back();
    e.cls = cls;
    e.filename = entry.path().string();
    e.title = filename.string();
    e.title.resize(e.title.size() - 4u);
    e.title[0] = std::toupper(e.title[0]);
    for (uint32_t i = 0; i < e.title.size(); ++i) {
      if (e.title[i] == '_') {
        e.title[i] = ' ';
        if (i + 1llu < e.title.size()) {
          e.title[i + 1llu] = std::toupper(e.title[i + 1llu]);
        }
      }
    }
  }

  std::sort(_ior_files.begin(), _ior_files.end(), [](const IORFile& a, const IORFile& b) {
    return a.cls < b.cls;
  });
}

void UI::cleanup() {
  ImGui::SaveIniSettingsToDisk(env().file_in_data("ui.ini"));
}

bool UI::build_options(Options& options) {
  bool changed = false;

  for (auto& option : options.values) {
    switch (option.cls) {
      case OptionalValue::Class::InfoString: {
        ImGui::TextColored({1.0f, 0.5f, 0.25f, 1.0f}, "%s", option.name.c_str());
        break;
      };

      case OptionalValue::Class::Boolean: {
        bool value = option.to_bool();
        if (ImGui::Checkbox(option.name.c_str(), &value)) {
          option.set(value);
          changed = true;
        }
        break;
      }

      case OptionalValue::Class::Float: {
        float value = option.to_float();
        ImGui::SetNextItemWidth(4.0f * ImGui::GetFontSize());
        if (ImGui::DragFloat(option.name.c_str(), &value, 0.1f, option.min_value.flt, option.max_value.flt, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
          option.set(value);
          changed = true;
        }
        break;
      }

      case OptionalValue::Class::Float3: {
        float3 value = option.to_float3();
        ImGui::SetNextItemWidth(4.0f * ImGui::GetFontSize());
        ImGui::Text("%s", option.name.c_str());
        char buffer_name[128] = {};
        snprintf(buffer_name, sizeof(buffer_name), "##%s", option.name.c_str());
        if (ImGui::DragFloat3(buffer_name, &value.x, 0.1f, option.min_value.f3.x, option.max_value.f3.x, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
          option.set(value);
          changed = true;
        }
        break;
      }

      case OptionalValue::Class::Integer: {
        int value = option.to_integer();
        ImGui::SetNextItemWidth(4.0f * ImGui::GetFontSize());
        if (ImGui::DragInt(option.name.c_str(), &value, 1.0f, option.min_value.integer, option.max_value.integer, "%u", ImGuiSliderFlags_AlwaysClamp)) {
          option.set(uint32_t(value));
          changed = true;
        }
        break;
      }

      case OptionalValue::Class::Enum: {
        int value = option.to_integer();
        ImGui::SetNextItemWidth(4.0f * ImGui::GetFontSize());
        if (ImGui::TreeNodeEx(option.name.c_str(), ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed)) {
          for (uint32_t i = 0; i <= option.max_value.integer; ++i) {
            if (ImGui::RadioButton(option.name_func(i).c_str(), &value, i)) {
              value = i;
            }
          }
          if (value != option.to_integer()) {
            option.set(uint32_t(value));
            changed = true;
          }
          ImGui::TreePop();
        }
        break;
      }

      default:
        ETX_FAIL("Invalid option");
    }
  }
  return changed;
}

bool UI::ior_picker(const char* name, RefractiveIndex& ior) {
  bool changed = false;
  bool selected = false;
  bool load_from_file = false;

  const char* names[5] = {
    "Invalid",
    "Reflectance",
    "Conductors",
    "Dielectrics",
    "Illuminants",
  };
  const ImVec4 colors[5] = {
    {0.3333f, 0.3333f, 0.3333f, 1.0f},
    {1.0f, 1.0f, 1.0f, 1.0f},
    {0.5f, 0.75f, 1.0f, 1.0f},
    {1.0f, 0.75f, 0.5f, 1.0f},
    {1.0f, 0.75f, 1.0f, 1.0f},
  };

  char buffer[64] = {};
  snprintf(buffer, sizeof(buffer), "##%s", name);
  if (ImGui::BeginCombo(buffer, name)) {
    SpectralDistribution::Class cls = SpectralDistribution::Class::Reflectance;
    if (_ior_files.empty() == false) {
      cls = _ior_files[0].cls;
    }

    ImGui::PushStyleColor(ImGuiCol_Text, colors[0]);
    ImGui::Text("%s", names[cls]);
    ImGui::PopStyleColor();

    for (const auto& i : _ior_files) {
      if (i.cls != cls) {
        cls = i.cls;
        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Text, colors[0]);
        ImGui::Text("%s", names[cls]);
        ImGui::PopStyleColor();
      }

      ImGui::PushStyleColor(ImGuiCol_Text, colors[cls]);
      if (ImGui::Selectable(i.title.c_str(), &selected)) {
        ior = RefractiveIndex::load_from_file(i.filename.c_str());
        changed = true;
      }
      ImGui::PopStyleColor();
    }
    ImGui::Separator();
    if (ImGui::Selectable("Load from file...", &selected)) {
      changed = true;
      load_from_file = true;
    }
    ImGui::EndCombo();
  }

  if (load_from_file) {
    auto filename = open_file("spd");
    RefractiveIndex tmp_ior = RefractiveIndex::load_from_file(filename.c_str());
    if (tmp_ior.cls != SpectralDistribution::Class::Invalid) {
      ior = tmp_ior;
      changed = true;
    }
  }

  return changed;
}

bool UI::spectrum_picker(const char* name, SpectralDistribution& spd, bool linear) {
  float3 rgb = {};

  if (_editor_values.count(name) == 0) {
    rgb = spd.integrated();
    if (linear == false) {
      rgb = linear_to_gamma(rgb);
    }
    _editor_values.emplace(name, rgb);
  } else {
    rgb = _editor_values.at(name);
  }

  bool changed = false;
  char name_buffer[64] = {};
  snprintf(name_buffer, sizeof(name_buffer), "##%s", name);
  if (ImGui::ColorEdit3(name_buffer, &rgb.x, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_InputRGB | (linear ? ImGuiColorEditFlags_HDR : 0))) {
    _editor_values[name] = rgb;
  }

  snprintf(name_buffer, sizeof(name_buffer), "Set %s", name);
  if (ImGui::Button(name_buffer)) {
    if (linear == false) {
      rgb = gamma_to_linear(rgb);
    }
    spd = SpectralDistribution::rgb_reflectance(rgb);
    changed = true;
  }

  return changed;
}

void UI::build(double dt) {
  ETX_FUNCTION_SCOPE();

  constexpr uint32_t kWindowFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize;
  bool has_integrator = (_current_integrator != nullptr);
  bool has_scene = (_current_scene != nullptr);

  simgui_new_frame(simgui_frame_desc_t{sapp_width(), sapp_height(), dt, sapp_dpi_scale()});

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
      ImGui::Separator();
      if (ImGui::MenuItem("Save...", nullptr, false, true)) {
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
        increase_exposure(_view_options);
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
      ui_toggle("Interator options", UIIntegrator);
      ui_toggle("Materials", UIMaterial);
      ui_toggle("Mediums", UIMedium);
      ui_toggle("Emitters", UIEmitters);
      ui_toggle("Camera", UICamera);
      ui_toggle("Scene", UIScene);
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
      ImGui::PopItemWidth();
    }

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

  if ((_ui_setup & UIIntegrator) && ImGui::Begin("Integrator options", nullptr, kWindowFlags)) {
    if (has_integrator) {
      ImGui::Text("%s", _current_integrator->name());
      _integrator_options = _current_integrator->options();
      if (build_options(_integrator_options) && callbacks.options_changed) {
        callbacks.options_changed();
      }
    } else {
      ImGui::Text("No integrator selected");
    }
    ImGui::End();
  }

  if ((_ui_setup & UIMaterial) && ImGui::Begin("Materials", nullptr, kWindowFlags)) {
    ImGui::Text("Materials");
    int32_t previous_selected = _selected_material;
    ImGui::ListBox("##materials", &_selected_material, _material_mapping.names.data(), static_cast<int32_t>(_material_mapping.size()), 6);
    if (scene_editable && (_selected_material >= 0) && (_selected_material < _material_mapping.size())) {
      if (previous_selected != _selected_material) {
        _editor_values.clear();
      }
      uint32_t material_index = _material_mapping.at(_selected_material);
      Material& material = _current_scene->materials[material_index];
      bool changed = build_material(material);
      if (changed && callbacks.material_changed) {
        callbacks.material_changed(material_index);
      }
    }
    ImGui::End();
  }

  if ((_ui_setup & UIMedium) && ImGui::Begin("Mediums", nullptr, kWindowFlags)) {
    ImGui::Text("Mediums");
    int32_t previous_selected = _selected_medium;
    ImGui::ListBox("##mediums", &_selected_medium, _medium_mapping.names.data(), static_cast<int32_t>(_medium_mapping.size()), 6);
    if (scene_editable && (_selected_medium >= 0) && (_selected_medium < _medium_mapping.size())) {
      if (previous_selected != _selected_medium) {
        _editor_values.clear();
      }
      uint32_t medium_index = _medium_mapping.at(_selected_medium);
      Medium& m = _current_scene->mediums[medium_index];
      bool changed = build_medium(m);
      if (changed && callbacks.material_changed) {
        callbacks.medium_changed(medium_index);
      }
    }
    ImGui::End();
  }

  if ((_ui_setup & UIEmitters) && ImGui::Begin("Emitters", nullptr, kWindowFlags)) {
    ImGui::Text("Emitters");
    if (ImGui::BeginListBox("##emitters", {})) {
      for (uint32_t index = 0; has_scene && (index < _current_scene->emitters.count); ++index) {
        auto& emitter = _current_scene->emitters[index];
        if (emitter.cls == Emitter::Class::Area)
          continue;

        char buffer[32] = {};
        switch (emitter.cls) {
          case Emitter::Class::Directional: {
            snprintf(buffer, sizeof(buffer), "%05u : directional", index);
            break;
          }
          case Emitter::Class::Environment: {
            snprintf(buffer, sizeof(buffer), "%05u : environment", index);
            break;
          }
          default:
            break;
        }
        if (ImGui::Selectable(buffer, _selected_emitter == index, ImGuiSelectableFlags_None, {})) {
          _selected_emitter = index;
        }
      }
      ImGui::EndListBox();
    }

    if (scene_editable && (_selected_emitter >= 0) && (_selected_emitter < _current_scene->emitters.count)) {
      auto& emitter = _current_scene->emitters[_selected_emitter];

      bool changed = spectrum_picker("Emission", emitter.emission.spectrum, true);

      if (emitter.cls == Emitter::Class::Directional) {
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

      if (changed && callbacks.emitter_changed) {
        callbacks.emitter_changed(_selected_emitter);
      }
    }

    ImGui::End();
  }

  if ((_ui_setup & UICamera) && ImGui::Begin("Camera", nullptr, kWindowFlags)) {
    if (scene_editable) {
      auto& camera = _current_scene->camera;
      bool camera_changed = false;
      bool film_changed = false;
      float3 pos = camera.position;
      float3 target = camera.position + camera.direction;
      float focal_len = get_camera_focal_length(camera);
      ImGui::Text("Lens Radius");
      camera_changed = camera_changed || ImGui::DragFloat("##lens", &_current_scene->lens.radius, 0.01f, 0.0f, 2.0, "%.3f", ImGuiSliderFlags_None);
      ImGui::Text("Focal Distance");
      camera_changed = camera_changed || ImGui::DragFloat("##focaldist", &_current_scene->lens.focal_distance, 0.1f, 0.0f, 65536.0f, "%.3f", ImGuiSliderFlags_None);
      ImGui::Text("Focal Length");
      camera_changed = camera_changed || ImGui::DragFloat("##focal", &focal_len, 0.1f, 1.0f, 90.0f, "%.3fmm", ImGuiSliderFlags_None);
      ImGui::Text("Pixel Filter Radius");
      camera_changed = camera_changed || ImGui::DragFloat("##pixelfiler", &_current_scene->pixel_sampler.radius, 0.05f, 0.0f, 32.0f, "%.3fpx", ImGuiSliderFlags_None);

      if (camera_changed && callbacks.camera_changed) {
        _current_scene->lens.radius = fmaxf(_current_scene->lens.radius, 0.0f);
        _current_scene->lens.focal_distance = fmaxf(_current_scene->lens.focal_distance, 0.0f);
        _current_scene->pixel_sampler.radius = clamp(_current_scene->pixel_sampler.radius, 0.0f, 32.0f);
        build_camera(camera, pos, target, float3{0.0f, 1.0f, 0.0f}, camera.image_size, focal_length_to_fov(focal_len) * 180.0f / kPi);
        callbacks.camera_changed();
      }
    } else {
      ImGui::Text("No options available");
    }
    ImGui::End();
  }

  if ((_ui_setup & UIScene) && ImGui::Begin("Scene", nullptr, kWindowFlags)) {
    if (scene_editable) {
      bool scene_settings_changed = false;

      ImGui::Text("Max samples per pixel / iterations:");
      scene_settings_changed = scene_settings_changed || ImGui::InputInt("##samples", reinterpret_cast<int*>(&_current_scene->samples));
      ImGui::Text("Maximum path length:");
      scene_settings_changed = scene_settings_changed || ImGui::InputInt("##maxpathlLength", reinterpret_cast<int*>(&_current_scene->max_path_length));
      ImGui::Text("Path length w/o random termination:");
      scene_settings_changed = scene_settings_changed || ImGui::InputInt("##bounces", reinterpret_cast<int*>(&_current_scene->random_path_termination));
      ImGui::Text("Noise Threshold:");
      scene_settings_changed = scene_settings_changed || ImGui::InputFloat("##noiseth", &_current_scene->noise_threshold, 0.0001f, 0.01f, "%0.5f");
      ImGui::Text("Radiance Clamp:");
      scene_settings_changed = scene_settings_changed || ImGui::InputFloat("##radclmp", &_current_scene->radiance_clamp, 0.1f, 1.f, "%0.2f");
      ImGui::Text("Active pixels: %0.2f%%", double(_film->active_pixel_count()) / double(_film->total_pixel_count()) * 100.0);
      scene_settings_changed = scene_settings_changed || ImGui::Checkbox("Spectral rendering", &_current_scene->spectral);

      if (scene_settings_changed) {
        callbacks.scene_settings_changed();
      }
    } else {
      ImGui::Text("No options available");
    }

    ImGui::End();
  }

  if (has_integrator && (_current_integrator->status().debug_info_count > 0)) {
    if (ImGui::Begin("Debug Info", nullptr, kWindowFlags)) {
      auto debug_info = _current_integrator->status().debug_info;
      for (uint64_t i = 0, e = _current_integrator->status().debug_info_count; i < e; ++i) {
        char buffer[8] = {};
        snprintf(buffer, sizeof(buffer), "%.3f     .", debug_info[i].value);
        ImGui::LabelText(buffer, "%s", debug_info[i].title);
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
    case SAPP_KEYCODE_F3:
    case SAPP_KEYCODE_F4:
    case SAPP_KEYCODE_F5:
    case SAPP_KEYCODE_F6: {
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
  auto selected_file = save_file("json");
  if ((selected_file.empty() == false) && callbacks.save_scene_file_selected) {
    callbacks.save_scene_file_selected(selected_file);
  }
}

void UI::save_image(SaveImageMode mode) const {
  auto selected_file = save_file(mode == SaveImageMode::TonemappedLDR ? "png" : "exr");
  if ((selected_file.empty() == false) && callbacks.save_image_selected) {
    callbacks.save_image_selected(selected_file, mode);
  }
}

void UI::load_image() const {
  auto selected_file = open_file("exr;png;hdr;pfm;jpg;bmp;tga");
  if ((selected_file.empty() == false) && callbacks.reference_image_selected) {
    callbacks.reference_image_selected(selected_file);
  }
}

void UI::set_film(Film* film) {
  _film = film;
}

void UI::set_scene(Scene* scene, const SceneRepresentation::MaterialMapping& materials, const SceneRepresentation::MediumMapping& mediums) {
  _current_scene = scene;

  _selected_material = -1;
  _material_mapping.build(materials);

  _selected_medium = -1;
  _medium_mapping.build(mediums);
}

bool UI::build_material(Material& material) {
  static uint32_t buffer_i = 0;

  bool changed = false;

  char buffer[64] = {};
  snprintf(buffer + buffer_i, sizeof(buffer) - buffer_i, "%s", material_class_to_string(material.cls));
  buffer[0] = std::toupper(buffer[0]);
  if (ImGui::BeginCombo("##type", buffer)) {
    for (uint32_t i = uint32_t(Material::Class::Diffuse); i < uint32_t(Material::Class::Count); ++i) {
      snprintf(buffer + buffer_i, sizeof(buffer) - buffer_i, "%s", material_class_to_string(Material::Class(i)));
      buffer[0] = std::toupper(buffer[0]);
      bool selected = material.cls == Material::Class(i);
      if (ImGui::Selectable(buffer, &selected)) {
        material.cls = static_cast<Material::Class>(i);
        changed = true;
      }
    }
    ImGui::EndCombo();
  }

  if (material.has_diffuse()) {
    ImGui::Text("Diffuse variation:");
    changed |= ImGui::InputInt("##var", reinterpret_cast<int32_t*>(&material.diffuse_variation));
  }
  ImGui::Separator();

  changed |= ImGui::SliderFloat("##r_u", &material.roughness.value.x, 0.0f, 1.0f, "Roughness U %.2f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
  changed |= ImGui::SliderFloat("##r_v", &material.roughness.value.y, 0.0f, 1.0f, "Roughness V %.2f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
  if (ImGui::Button("Sync Roughness")) {
    material.roughness.value.y = material.roughness.value.x;
  }
  ImGui::Separator();

  changed |= ior_picker("Index Of Refraction", material.int_ior);
  changed |= ior_picker("Index Of Refraction (outside)", material.ext_ior);
  ImGui::Separator();
  changed |= spectrum_picker("Reflectance", material.reflectance.spectrum, false);
  changed |= spectrum_picker("Transmittance", material.transmittance.spectrum, false);
  ImGui::Separator();

  ImGui::Text("%s", "Subsurface Scattering");
  changed |= ImGui::Combo("##sssclass", reinterpret_cast<int*>(&material.subsurface.cls), "Disabled\0Random Walk\0Christensen-Burley\0");
  changed |= ImGui::Combo("##ssspath", reinterpret_cast<int*>(&material.subsurface.path), "Diffuse Transmittance\0Refraction\0");
  changed |= spectrum_picker("Subsurface Distance", material.subsurface.scattering_distance, true);
  ImGui::Text("%s", "Subsurface Distance Scale");
  changed |= ImGui::InputFloat("##sssdist", &material.subsurface.scale);
  ImGui::Separator();

  ImGui::Text("%s", "Thinfilm");
  ImGui::Text("%s", "Thickness Range (nm)");
  ImGui::SetNextItemWidth(ImGui::GetWindowWidth() / 3.0f);
  changed |= ImGui::InputFloat("##tftmin", &material.thinfilm.min_thickness);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(ImGui::GetWindowWidth() / 3.0f);
  changed |= ImGui::InputFloat("##tftmax", &material.thinfilm.max_thickness);
  changed |= ior_picker("Thinfilm IoR", material.thinfilm.ior);
  ImGui::Separator();

  auto medium_editor = [](const char* name, uint32_t& medium, uint64_t medium_count) -> bool {
    bool has_medium = medium != kInvalidIndex;
    bool changed = ImGui::Checkbox(name, &has_medium);
    if (has_medium) {
      int32_t medium_index = static_cast<int32_t>(medium);
      if (medium_index == -1)
        medium_index = 0;
      changed |= ImGui::SliderInt("##medium_index", &medium_index, 0, int32_t(medium_count - 1u), "Index: %d", ImGuiSliderFlags_AlwaysClamp);
      medium = uint32_t(medium_index);
    } else {
      medium = kInvalidIndex;
    }
    return changed;
  };

  if (_medium_mapping.empty() == false) {
    changed |= medium_editor("Internal medium", material.int_medium, _current_scene->mediums.count);
    changed |= medium_editor("Extenral medium", material.ext_medium, _current_scene->mediums.count);
  }

  return changed;
}

bool UI::build_medium(Medium& m) {
  bool changed = false;
  changed |= spectrum_picker("Absorption", m.s_absorption, true);
  changed |= spectrum_picker("Scattering", m.s_scattering, true);
  changed |= ImGui::SliderFloat("##g", &m.phase_function_g, -0.999f, 0.999f, "Asymmetry %.2f", ImGuiSliderFlags_AlwaysClamp);
  changed |= ImGui::Checkbox("Connections to light / camera", &m.enable_explicit_connections);
  return changed;
}

void UI::reset_selection() {
  _selected_emitter = -1;
  _selected_material = -1;
  _selected_medium = -1;
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
