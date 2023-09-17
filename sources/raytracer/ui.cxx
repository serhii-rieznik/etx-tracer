#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/render/host/image_pool.hxx>

#include "ui.hxx"

#include <sokol_app.h>
#include <sokol_gfx.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <imgui.h>
#include <imgui_internal.h>
#include <util/sokol_imgui.h>

#if (ETX_PLATFORM_APPLE)
#include <unistd.h>
#endif

#include <filesystem>

namespace etx {

inline void decrease_exposure(ViewOptions& o) {
  o.exposure = fmaxf(1.0f / 1024.0f, 0.5f * o.exposure);
}

inline void increase_exposure(ViewOptions& o) {
  o.exposure = fmaxf(1.0f / 1024.0f, 2.0f * o.exposure);
}

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

  RefractiveIndex tmp;
  std::string path = env().file_in_data("spectrum/");
  for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
    auto filename = entry.path().filename();
    const auto& ext = entry.path().extension();
    if (ext != L".spd")
      continue;

    log::info("Preloading spectral distribution from %s", entry.path().string().c_str());

    auto cls = SpectralDistribution::load_from_file(entry.path().string().c_str(), tmp.eta, &tmp.k, shared_spectrums());
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
        if (ImGui::DragFloat(option.name.c_str(), &value, 0.001f, option.min_value.flt, option.max_value.flt, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
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

bool UI::ior_picker(const char* name, RefractiveIndex& ior, const Pointer<Spectrums> spectrums) {
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
    ImGui::Text(names[cls]);
    ImGui::PopStyleColor();

    for (const auto& i : _ior_files) {
      if (i.cls != cls) {
        cls = i.cls;
        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Text, colors[0]);
        ImGui::Text(names[cls]);
        ImGui::PopStyleColor();
      }

      ImGui::PushStyleColor(ImGuiCol_Text, colors[cls]);
      if (ImGui::Selectable(i.title.c_str(), &selected)) {
        SpectralDistribution::load_from_file(i.filename.c_str(), ior.eta, &ior.k, spectrums);
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
    RefractiveIndex tmp_ior = {};
    auto cls = SpectralDistribution::load_from_file(filename.c_str(), tmp_ior.eta, &tmp_ior.k, spectrums);
    if (cls != SpectralDistribution::Class::Invalid) {
      ior = tmp_ior;
      changed = true;
    }
  }

  return changed;
}

bool UI::spectrum_picker(const char* name, SpectralDistribution& spd, const Pointer<Spectrums> spectrums, bool linear) {
  float3 rgb = spectrum::xyz_to_rgb(spd.to_xyz());
  if (linear == false) {
    rgb = linear_to_gamma(rgb);
  }

  uint32_t flags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_InputRGB;

  if (linear) {
    flags = flags | ImGuiColorEditFlags_HDR;
  }

  ImGui::Text("%s", name);

  bool changed = false;
  char name_buffer[64] = {};
  snprintf(name_buffer, sizeof(name_buffer), "##%s", name);
  if (ImGui::ColorEdit3(name_buffer, &rgb.x, flags)) {
    if (linear == false) {
      rgb = gamma_to_linear(rgb);
    }
    rgb = max(rgb, float3{});
    spd = rgb::make_reflectance_spd(rgb, spectrums);
    changed = true;
  }

  char buffer[128] = {};
  snprintf(buffer, sizeof(buffer), "Clear %s", name);
  if (linear && ImGui::Button(buffer)) {
    spd = SpectralDistribution::from_constant(0.0f);
    changed = true;
  }

  return changed;
}

void UI::build(double dt, const char* status) {
  constexpr uint32_t kWindowFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize;
  bool has_integrator = (_current_integrator != nullptr);
  bool has_scene = (_current_scene != nullptr);

  simgui_new_frame(simgui_frame_desc_t{sapp_width(), sapp_height(), dt, sapp_dpi_scale()});

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("Scene", true)) {
      if (ImGui::MenuItem("Open...", "Ctrl+O", false, true)) {
        select_scene_file();
      }
      if (ImGui::MenuItem("Reload Scene", "Ctrl+R", false, true)) {
        if (callbacks.reload_scene_selected) {
          callbacks.reload_scene_selected();
        }
      }
      if (ImGui::MenuItem("Reload Geometry and Materials", "Ctrl+G", false, true)) {
        if (callbacks.reload_geometry_selected) {
          callbacks.reload_geometry_selected();
        }
      }
      if (ImGui::MenuItem("Reload Materials", "Ctrl+M", false, false)) {
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Save...", nullptr, false, true)) {
        save_scene_file();
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Integrator", true)) {
      for (uint64_t i = 0; i < _integrators.count; ++i) {
        if (ImGui::MenuItem(_integrators[i]->name(), nullptr, false, _integrators[i]->enabled())) {
          if (callbacks.integrator_selected) {
            callbacks.integrator_selected(_integrators[i]);
          }
        }
      }

      if (has_integrator) {
        ImGui::Separator();
        if (ImGui::MenuItem("Reload Integrator State", "Ctrl+A", false, true)) {
          if (callbacks.reload_integrator) {
            callbacks.reload_integrator();
          }
        }
      }

      ImGui::Separator();
      if (ImGui::MenuItem("Exit", "Ctrl+Q", false, true)) {
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
      if (ImGui::MenuItem("Save Current Image (XYZ)...", "Alt+Ctrl+S", false, true)) {
        save_image(SaveImageMode::XYZ);
      }
      if (ImGui::MenuItem("Use as Reference", "Ctrl+Shift+R", false, true)) {
        if (callbacks.use_image_as_reference) {
          callbacks.use_image_as_reference();
        }
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View", true)) {
      char shortcut[2] = "X";
      for (uint32_t i = 0; i < uint32_t(OutputView::Count); ++i) {
        shortcut[0] = char('1' + i);
        if (ImGui::MenuItem(output_view_to_string(i).c_str(), shortcut, uint32_t(_view_options.view) == i, true)) {
          _view_options.view = static_cast<OutputView>(i);
        }
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
      ui_toggle("Materials and mediums", UIMaterial);
      ui_toggle("Emitters", UIEmitters);
      ui_toggle("Camera", UICamera);
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }

  bool scene_editable = has_integrator && has_scene &&  //
                        ((_current_integrator->state() == Integrator::State::Preview) || (_current_integrator->state() == Integrator::State::Stopped));

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
      can_run && ((state == Integrator::State::Stopped) || (state == Integrator::State::Preview)),
      can_run && (state == Integrator::State::Running),
      can_run && ((state != Integrator::State::Stopped)),
    };

    ImVec4 colors[4] = {
      state_available[0] ? ImVec4{0.1f, 0.1f, 0.33f, 1.0f} : ImVec4{0.25f, 0.25f, 0.25f, 1.0f},
      state_available[1] ? ImVec4{0.1f, 0.33f, 0.1f, 1.0f} : ImVec4{0.25f, 0.25f, 0.25f, 1.0f},
      state_available[2] ? ImVec4{0.33f, 0.22f, 0.11f, 1.0f} : ImVec4{0.25f, 0.25f, 0.25f, 1.0f},
      state_available[3] ? ImVec4{0.33f, 0.1f, 0.1f, 1.0f} : ImVec4{0.25f, 0.25f, 0.25f, 1.0f},
    };

    std::string labels[4] = {
      (state == Integrator::State::Preview) ? "> Preview <" : "  Preview  ",
      (state == Integrator::State::Running) ? "> Launch <" : "  Launch  ",
      (state == Integrator::State::WaitingForCompletion) ? "> Finish <" : "  Finish  ",
      " Terminate ",
    };

    ImGui::PushStyleColor(ImGuiCol_Button, colors[0]);
    if (ImGui::Button(labels[0].c_str(), {0.0f, button_size})) {
      callbacks.preview_selected();
    }

    ImGui::SameLine(0.0f, wpadding.x);
    ImGui::PushStyleColor(ImGuiCol_Button, colors[1]);
    if (ImGui::Button(labels[1].c_str(), {0.0f, button_size})) {
      callbacks.run_selected();
    }

    ImGui::SameLine(0.0f, wpadding.x);
    ImGui::PushStyleColor(ImGuiCol_Button, colors[2]);
    if (ImGui::Button(labels[2].c_str(), {0.0f, button_size})) {
      callbacks.stop_selected(true);
    }

    ImGui::SameLine(0.0f, wpadding.x);
    ImGui::PushStyleColor(ImGuiCol_Button, colors[3]);
    if (ImGui::Button(labels[3].c_str(), {0.0f, button_size})) {
      callbacks.stop_selected(false);
    }

    ImGui::PopStyleColor(4);
    ImGui::SameLine(0.0f, wpadding.x);

    ImGui::GetStyle().FramePadding.y = (button_size - text_size) / 2.0f;

    if (_current_scene != nullptr) {
      bool scene_settings_changed = false;
      ImGui::PushItemWidth(2.0f * input_size);
      {
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine(0.0f, wpadding.x);
        scene_settings_changed = scene_settings_changed || ImGui::InputInt("Samples", reinterpret_cast<int*>(&_current_scene->samples));
        ImGui::SameLine(0.0f, wpadding.x);
      }
      {
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine(0.0f, wpadding.x);
        scene_settings_changed = scene_settings_changed || ImGui::InputInt("Max Path Length", reinterpret_cast<int*>(&_current_scene->max_path_length));
        ImGui::SameLine(0.0f, wpadding.x);
      }
      ImGui::PopItemWidth();
      if (scene_settings_changed && callbacks.scene_settings_changed) {
        callbacks.scene_settings_changed();
      }
    }

    ImGui::PushItemWidth(input_size);
    {
      ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
      ImGui::SameLine(0.0f, wpadding.x);
      ImGui::DragFloat("Exposure", &_view_options.exposure, 1.0f / 256.0f, 1.0f / 1024.0f, 1024.0f, "%.4f", ImGuiSliderFlags_NoRoundToFormat);
      ImGui::SameLine(0.0f, wpadding.x);
    }
    ImGui::GetStyle().FramePadding.y = fpadding.y;
    ImGui::PopItemWidth();

    ImGui::End();
  }

  if (ImGui::BeginViewportSideBar("##status", ImGui::GetMainViewport(), ImGuiDir_Down, text_size + 2.0f * wpadding.y, ImGuiWindowFlags_NoDecoration)) {
    char status_buffer[2048] = {};
    uint32_t cpu_load = static_cast<uint32_t>(get_cpu_load() * 100.0f);
    snprintf(status_buffer, sizeof(status_buffer), "%-3u cpu | %.2fms | %.2ffps | %s", cpu_load, 1000.0 * dt, 1.0f / dt, status ? status : "");
    ImGui::Text("%s", status_buffer);
    ImGui::End();
  }

  if ((_ui_setup & UIIntegrator) && ImGui::Begin("Integrator options", nullptr, kWindowFlags)) {
    if (has_integrator) {
      ImGui::Text("%s", _current_integrator->name());
      if (build_options(_integrator_options) && callbacks.options_changed) {
        callbacks.options_changed();
      }
    } else {
      ImGui::Text("No integrator selected");
    }
    ImGui::End();
  }

  if ((_ui_setup & UIMaterial) && ImGui::Begin("Materials and mediums", nullptr, kWindowFlags)) {
    ImGui::Text("Materials");
    ImGui::ListBox("##materials", &_selected_material, _material_mapping.names.data(), static_cast<int32_t>(_material_mapping.size()), 6);
    if (scene_editable && (_selected_material >= 0) && (_selected_material < _material_mapping.size())) {
      uint32_t material_index = _material_mapping.at(_selected_material);
      Material& material = _current_scene->materials[material_index];
      bool changed = build_material(material);
      if (changed && callbacks.material_changed) {
        callbacks.material_changed(material_index);
      }
    }

    ImGui::Separator();

    ImGui::Text("Mediums");
    ImGui::ListBox("##mediums", &_selected_medium, _medium_mapping.names.data(), static_cast<int32_t>(_medium_mapping.size()), 6);
    if (scene_editable && (_selected_medium >= 0) && (_selected_medium < _medium_mapping.size())) {
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

      bool changed = spectrum_picker("Emission", emitter.emission.spectrum, _current_scene->spectrums, true);

      if (emitter.cls == Emitter::Class::Directional) {
        ImGui::Text("Angular Size");
        if (ImGui::DragFloat("##angularsize", &emitter.angular_size, 0.01f, 0.0f, kHalfPi, "%.3f", ImGuiSliderFlags_NoRoundToFormat)) {
          emitter.angular_size_cosine = cosf(emitter.angular_size / 2.0f);
          emitter.equivalent_disk_size = 2.0f * std::tan(emitter.angular_size / 2.0f);
          changed = true;
        }

        ImGui::Text("Direction");
        if (ImGui::DragFloat3("##direction", &emitter.direction.x, 0.1f, -256.0f, 256.0f, "%.02f", ImGuiSliderFlags_NoRoundToFormat)) {
          emitter.direction = normalize(emitter.direction);
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
      bool changed = false;
      float3 pos = camera.position;
      float3 target = camera.position + camera.direction;
      float focal_len = get_camera_focal_length(camera);
      ImGui::Text("Lens size");
      changed = changed || ImGui::DragFloat("##lens", &camera.lens_radius, 0.01f, 0.0f, 2.0, "%.3f", ImGuiSliderFlags_None);
      ImGui::Text("Focal distance");
      changed = changed || ImGui::DragFloat("##focaldist", &camera.focal_distance, 0.1f, 0.0f, 65536.0f, "%.3f", ImGuiSliderFlags_None);
      ImGui::Text("Focal length");
      changed = changed || ImGui::DragFloat("##focal", &focal_len, 0.1f, 1.0f, 90.0f, "%.3fmm", ImGuiSliderFlags_None);

      if (changed && callbacks.camera_changed) {
        camera.lens_radius = fmaxf(camera.lens_radius, 0.0f);
        camera.focal_distance = fmaxf(camera.focal_distance, 0.0f);
        update_camera(camera, pos, target, float3{0.0f, 1.0f, 0.0f}, camera.image_size, focal_length_to_fov(focal_len) * 180.0f / kPi);
        callbacks.camera_changed();
      }
    } else {
      ImGui::Text("No options available");
    }
    ImGui::End();
  }

  if (has_integrator && (_current_integrator->debug_info_count() > 0)) {
    if (ImGui::Begin("Debug Info", nullptr, kWindowFlags)) {
      auto debug_info = _current_integrator->debug_info();
      for (uint64_t i = 0, e = _current_integrator->debug_info_count(); i < e; ++i) {
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
        } else if (callbacks.reload_scene_selected) {
          callbacks.reload_scene_selected();
        }
        break;
      }
      case SAPP_KEYCODE_G: {
        if (callbacks.reload_geometry_selected)
          callbacks.reload_geometry_selected();
        break;
      }
      case SAPP_KEYCODE_A: {
        if (callbacks.reload_integrator)
          callbacks.reload_integrator();
        break;
      }
      case SAPP_KEYCODE_S: {
        if (has_alt && (has_shift == false))
          save_image(SaveImageMode::XYZ);
        else if (has_shift && (has_alt == false)) {
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
    case SAPP_KEYCODE_F5: {
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

void UI::set_current_integrator(Integrator* i) {
  _current_integrator = i;
  _integrator_options = _current_integrator ? _current_integrator->options() : Options{};
}

void UI::quit() {
  sapp_quit();
}

void UI::select_scene_file() {
  auto selected_file = open_file("json;obj");  // TODO : add gltf;pbrt
  if ((selected_file.empty() == false) && callbacks.scene_file_selected) {
    callbacks.scene_file_selected(selected_file);
  }
}

void UI::save_scene_file() {
  auto selected_file = save_file("json");
  if ((selected_file.empty() == false) && callbacks.save_scene_file_selected) {
    callbacks.save_scene_file_selected(selected_file);
  }
}

void UI::save_image(SaveImageMode mode) {
  auto selected_file = save_file(mode == SaveImageMode::TonemappedLDR ? "png" : "exr");
  if ((selected_file.empty() == false) && callbacks.save_image_selected) {
    callbacks.save_image_selected(selected_file, mode);
  }
}

void UI::load_image() {
  auto selected_file = open_file("exr;png;hdr;pfm;jpg;bmp;tga");
  if ((selected_file.empty() == false) && callbacks.reference_image_selected) {
    callbacks.reference_image_selected(selected_file);
  }
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

  changed |= ImGui::SliderFloat("##r_u", &material.roughness.x, 0.0f, 1.0f, "Roughness U %.2f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
  changed |= ImGui::SliderFloat("##r_v", &material.roughness.y, 0.0f, 1.0f, "Roughness V %.2f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoRoundToFormat);
  changed |= ior_picker("Index Of Refraction", material.int_ior, _current_scene->spectrums);
  changed |= spectrum_picker("Diffuse", material.diffuse.spectrum, _current_scene->spectrums, false);
  changed |= spectrum_picker("Specular", material.specular.spectrum, _current_scene->spectrums, false);
  changed |= spectrum_picker("Transmittance", material.transmittance.spectrum, _current_scene->spectrums, false);
  ImGui::Separator();

  ImGui::Text("%s", "Subsurface Scattering");
  changed |= ImGui::Combo("##sssclass", reinterpret_cast<int*>(&material.subsurface.cls), "Disabled\0Random Walk\0Christensen-Burley");
  if (material.subsurface.cls == SubsurfaceMaterial::Class::RandomWalk) {
    ImGui::Text("(controlled by the internal meidum");
  } else if (material.subsurface.cls == SubsurfaceMaterial::Class::ChristensenBurley) {
    changed |= spectrum_picker("Subsurface Distance", material.subsurface.scattering_distance, _current_scene->spectrums, true);
    ImGui::Text("%s", "Subsurface Distance Scale");
    changed |= ImGui::InputFloat("##sssdist", &material.subsurface.scale);
  }
  ImGui::Separator();

  ImGui::Text("%s", "Thinfilm");
  ImGui::Text("%s", "Thickness Range (nm)");
  ImGui::SetNextItemWidth(ImGui::GetWindowWidth() / 3.0f);
  changed |= ImGui::InputFloat("##tftmin", &material.thinfilm.min_thickness);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(ImGui::GetWindowWidth() / 3.0f);
  changed |= ImGui::InputFloat("##tftmax", &material.thinfilm.max_thickness);
  changed |= ior_picker("Thinn film IoR", material.thinfilm.ior, _current_scene->spectrums);
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
  changed |= spectrum_picker("Absorption", m.s_absorption, _current_scene->spectrums, true);
  changed |= spectrum_picker("Outscattering", m.s_outscattering, _current_scene->spectrums, true);
  changed |= ImGui::SliderFloat("##g", &m.phase_function_g, -0.999f, 0.999f, "Asymmetry %.2f", ImGuiSliderFlags_AlwaysClamp);
  return changed;
}

}  // namespace etx
