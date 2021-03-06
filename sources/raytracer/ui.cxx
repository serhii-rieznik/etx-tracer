#include <etx/core/core.hxx>
#include <etx/render/host/image_pool.hxx>

#include "ui.hxx"

#include <sokol_app.h>
#include <sokol_gfx.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>
#include <sokol_imgui.h>

namespace etx {

void UI::initialize() {
  simgui_desc_t imggui_desc = {};
  imggui_desc.depth_format = SG_PIXELFORMAT_NONE;
  simgui_setup(imggui_desc);

  _view_options = Options{{
    {OutputView::Result, OutputView::Count, output_view_to_string, "out_view", "View Image"},
    {1.0f / 1000.0f, 1.0f, 1000.0f, "exp", "Exposure"},
  }};
}

bool UI::build_options(Options& options) {
  bool changed = false;

  for (auto& option : options.values) {
    switch (option.cls) {
      case OptionalValue::Class::InfoString: {
        igTextColored({1.0f, 0.5f, 0.25f, 1.0f}, option.name.c_str());
        break;
      };

      case OptionalValue::Class::Boolean: {
        bool value = option.to_bool();
        if (igCheckbox(option.name.c_str(), &value)) {
          option.set(value);
          changed = true;
        }
        break;
      }

      case OptionalValue::Class::Float: {
        float value = option.to_float();
        igSetNextItemWidth(4.0f * igGetFontSize());
        if (igDragFloat(option.name.c_str(), &value, 0.001f, option.min_value.flt, option.max_value.flt, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
          option.set(value);
          changed = true;
        }
        break;
      }

      case OptionalValue::Class::Integer: {
        int value = option.to_integer();
        igSetNextItemWidth(4.0f * igGetFontSize());
        if (igDragInt(option.name.c_str(), &value, 1.0f, option.min_value.integer, option.max_value.integer, "%u", ImGuiSliderFlags_AlwaysClamp)) {
          option.set(uint32_t(value));
          changed = true;
        }
        break;
      }

      case OptionalValue::Class::Enum: {
        int value = option.to_integer();
        igSetNextItemWidth(4.0f * igGetFontSize());
        if (igTreeNodeEx_Str(option.name.c_str(), ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed)) {
          for (uint32_t i = 0; i <= option.max_value.integer; ++i) {
            if (igRadioButton_IntPtr(option.name_func(i).c_str(), &value, i)) {
              value = i;
            }
          }
          if (value != option.to_integer()) {
            option.set(uint32_t(value));
            changed = true;
          }
          igTreePop();
        }
        break;
      }

      default:
        ETX_FAIL("Invalid option");
    }
  }
  return changed;
}

void UI::build(double dt, const char* status) {
  simgui_new_frame(simgui_frame_desc_t{sapp_width(), sapp_height(), dt, sapp_dpi_scale()});

  float offset_size = igGetFontSize();
  igSetNextWindowPos({sapp_widthf() / sapp_dpi_scale() - offset_size, 2.0f * offset_size}, ImGuiCond_Always, {1.0f, 0.0f});
  igBegin("View Options", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);
  build_options(_view_options);

  char status_buffer[2048] = {};
  float dy = igGetStyle()->FramePadding.y;
  uint32_t cpu_load = static_cast<uint32_t>(TimeMeasure::get_cpu_load() * 100.0f);
  snprintf(status_buffer, sizeof(status_buffer), "% 3u cpu | %.2fms | %.2ffps | %s", cpu_load, 1000.0 * dt, 1.0f / dt, status ? status : "");
  igBeginViewportSideBar("Sidebar", igGetMainViewport(), ImGuiDir_Down, dy + 2.0f * offset_size, ImGuiWindowFlags_NoDecoration);
  igText(status_buffer);
  igEnd();

  if (igBeginMainMenuBar()) {
    if (igBeginMenu("Integrator", true)) {
      for (uint64_t i = 0; i < _integrators.count; ++i) {
        if (igMenuItemEx(_integrators[i]->name(), nullptr, nullptr, false, _integrators[i]->enabled())) {
          if (callbacks.integrator_selected) {
            callbacks.integrator_selected(_integrators[i]);
          }
        }
      }

      if (_current_integrator != nullptr) {
        igSeparator();
        if (igMenuItemEx("Reload Integrator State", nullptr, "Ctrl+A", false, true)) {
          if (callbacks.reload_integrator) {
            callbacks.reload_integrator();
          }
        }
      }

      igSeparator();
      if (igMenuItemEx("Exit", nullptr, "Ctrl+Q", false, true)) {
      }
      igEndMenu();
    }

    if (igBeginMenu("Scene", true)) {
      if (igMenuItemEx("Open...", nullptr, "Ctrl+O", false, true)) {
        select_scene_file();
      }
      if (igMenuItemEx("Reload Scene", nullptr, "Ctrl+R", false, true)) {
        if (callbacks.reload_scene_selected) {
          callbacks.reload_scene_selected();
        }
      }
      if (igMenuItemEx("Reload Geometry and Materials", nullptr, "Ctrl+G", false, true)) {
        if (callbacks.reload_geometry_selected) {
          callbacks.reload_geometry_selected();
        }
      }
      if (igMenuItemEx("Reload Materials", nullptr, "Ctrl+M", false, false)) {
      }
      igSeparator();
      if (igMenuItemEx("Save...", nullptr, nullptr, false, false)) {
      }
      igEndMenu();
    }

    if (igBeginMenu("Image", true)) {
      if (igMenuItemEx("Open Reference Image...", nullptr, "Ctrl+I", false, true)) {
        load_image();
      }
      igSeparator();
      if (igMenuItemEx("Save Current Image (RGB)...", nullptr, "Ctrl+S", false, true)) {
        save_image(SaveImageMode::RGB);
      }
      if (igMenuItemEx("Save Current Image (LDR)...", nullptr, "Shift+Ctrl+S", false, true)) {
        save_image(SaveImageMode::TonemappedLDR);
      }
      if (igMenuItemEx("Save Current Image (XYZ)...", nullptr, "Alt+Ctrl+S", false, true)) {
        save_image(SaveImageMode::XYZ);
      }
      igEndMenu();
    }
    igEndMainMenuBar();
  }
  igEnd();

  if ((_current_integrator != nullptr) && (_integrator_options.values.empty() == false)) {
    igSetNextWindowPos({offset_size, 2.0f * offset_size}, ImGuiCond_Always, {0.0f, 0.0f});
    igBegin(_integrator_name, nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
    igText("Integrator options");

    if (build_options(_integrator_options) && callbacks.options_changed) {
      callbacks.options_changed();
    }

    if (_current_integrator->can_run()) {
      igSeparator();
      igNewLine();

      auto state = _current_integrator->state();
      bool has_complete = (state == Integrator::State::Running);
      bool has_stop = (state != Integrator::State::Stopped);
      bool has_preview = (state == Integrator::State::Stopped);
      bool has_run = (state == Integrator::State::Stopped) || (state == Integrator::State::Preview);

      igPushStyleColor_Vec4(ImGuiCol_Button, {0.33f, 0.22f, 0.11f, 1.0f});
      if (has_complete && igButton("[ Complete iteration and stop ]", {}) && callbacks.stop_selected) {
        callbacks.stop_selected(true);
      }

      igPushStyleColor_Vec4(ImGuiCol_Button, {0.33f, 0.1f, 0.1f, 1.0f});
      if (has_stop && igButton("[ Break Immediately ]", {}) && callbacks.stop_selected) {
        callbacks.stop_selected(false);
      }

      igPushStyleColor_Vec4(ImGuiCol_Button, {0.1f, 0.1f, 0.33f, 1.0f});
      if (has_preview && igButton("[ Preview ]", {}) && callbacks.preview_selected) {
        callbacks.preview_selected();
      }

      igPushStyleColor_Vec4(ImGuiCol_Button, {0.1f, 0.33f, 0.1f, 1.0f});
      if (has_run && igButton("[ Launch ]", {}) && callbacks.run_selected) {
        callbacks.run_selected();
      }

      igPopStyleColor(4);
      igNewLine();
    }
    igEnd();
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

  if (modifiers & SAPP_MODIFIER_CTRL) {
    switch (e->key_code) {
      case SAPP_KEYCODE_O: {
        select_scene_file();
        break;
      }
      case SAPP_KEYCODE_I: {
        load_image();
        break;
      }
      case SAPP_KEYCODE_R: {
        if (callbacks.reload_scene_selected)
          callbacks.reload_scene_selected();
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
    case SAPP_KEYCODE_1: {
      _view_options.set_enum("out_view", OutputView::Result);
      break;
    }
    case SAPP_KEYCODE_2: {
      _view_options.set_enum("out_view", OutputView::CameraImage);
      break;
    }
    case SAPP_KEYCODE_3: {
      _view_options.set_enum("out_view", OutputView::LightImage);
      break;
    }
    case SAPP_KEYCODE_4: {
      _view_options.set_enum("out_view", OutputView::ReferenceImage);
      break;
    }
    case SAPP_KEYCODE_5: {
      _view_options.set_enum("out_view", OutputView::RelativeDifference);
      break;
    }
    case SAPP_KEYCODE_6: {
      _view_options.set_enum("out_view", OutputView::AbsoluteDifference);
      break;
    }
    case SAPP_KEYCODE_KP_DIVIDE: {
      float e = _view_options.get("exp", 1.0f).to_float();
      _view_options.set("exp", 0.5f * e);
      break;
    }
    case SAPP_KEYCODE_KP_MULTIPLY: {
      float e = _view_options.get("exp", 1.0f).to_float();
      _view_options.set("exp", 2.0f * e);
      break;
    }
    default:
      break;
  }

  return false;
}

ViewOptions UI::view_options() const {
  return {
    _view_options.get("out_view", uint32_t(OutputView::Result)).to_enum<OutputView>(),
    ViewOptions::ToneMapping | ViewOptions::sRGB,
    _view_options.get("exp", 1.0f).to_float(),
  };
}

void UI::set_current_integrator(Integrator* i) {
  _current_integrator = i;
  _integrator_name = _current_integrator ? _current_integrator->name() : "";
  _integrator_options = _current_integrator ? _current_integrator->options() : Options{};
}

void UI::select_scene_file() {
  auto selected_file = open_file({"Supported formats", "*.json;*.obj"});  // TODO : add *.gltf;*.pbrt
  if ((selected_file.empty() == false) && callbacks.scene_file_selected) {
    callbacks.scene_file_selected(selected_file);
  }
}

void UI::save_image(SaveImageMode mode) {
  auto selected_file = save_file({
    mode == SaveImageMode::TonemappedLDR ? "PNG images" : "EXR images",
    mode == SaveImageMode::TonemappedLDR ? "*.png" : "*.exr",
  });
  if ((selected_file.empty() == false) && callbacks.save_image_selected) {
    callbacks.save_image_selected(selected_file, mode);
  }
}

void UI::load_image() {
  auto selected_file = open_file({"Supported images", "*.exr;*.png;*.hdr;*.pfm;*.jpg;*.bmp;*.tga"});
  if ((selected_file.empty() == false) && callbacks.reference_image_selected) {
    callbacks.reference_image_selected(selected_file);
  }
}

}  // namespace etx
