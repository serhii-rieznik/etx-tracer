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
    {0.001f, 1.0f, +10.0f, "exp", "Exposure"},
  }};
}

void UI::build_options(Options& options) {
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
        }
        break;
      }

      case OptionalValue::Class::Float: {
        float value = option.to_float();
        igSetNextItemWidth(4.0f * igGetFontSize());
        if (igDragFloat(option.name.c_str(), &value, 0.001f, option.min_value.flt, option.max_value.flt, "%.3f", ImGuiSliderFlags_AlwaysClamp)) {
          option.set(value);
        }
        break;
      }

      case OptionalValue::Class::Integer: {
        int value = option.to_integer();
        igSetNextItemWidth(4.0f * igGetFontSize());
        if (igDragInt(option.name.c_str(), &value, 1.0f, option.min_value.integer, option.max_value.integer, "%u", ImGuiSliderFlags_AlwaysClamp)) {
          option.set(uint32_t(value));
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
          }
          igTreePop();
        }
        break;
      }

      default:
        ETX_FAIL("Invalid option");
    }
  }
}

void UI::build(double dt, const char* status) {
  simgui_new_frame(simgui_frame_desc_t{sapp_width(), sapp_height(), dt, sapp_dpi_scale()});

  igSetNextWindowPos({sapp_widthf() - igGetFontSize(), 2.0f * igGetFontSize()}, ImGuiCond_Always, {1.0f, 0.0f});
  igBegin("View", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);
  igText("View options");
  build_options(_view_options);

  char status_buffer[2048] = {};
  float dy = igGetStyle()->FramePadding.y;
  snprintf(status_buffer, sizeof(status_buffer), "%.2fms | %.2ffps | %s", 1000.0 * dt, 1.0f / dt, status ? status : "");
  igBeginViewportSideBar("Sidebar", igGetMainViewport(), ImGuiDir_Down, dy + 2.0f * igGetFontSize(), ImGuiWindowFlags_NoDecoration);
  igText(status_buffer);
  igEnd();

  if (igBeginMainMenuBar()) {
    if (igBeginMenu("Raytracer", true)) {
      for (uint64_t i = 0; i < _integrators.count; ++i) {
        if (igMenuItemEx(_integrators[i]->name(), nullptr, nullptr, false, true)) {
          if (callbacks.integrator_selected) {
            callbacks.integrator_selected(_integrators[i]);
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
        auto selected_file = open_file({"Supported formats", "*.json;*.obj;*.gltf;*.pbrt"});
        if ((selected_file.empty() == false) && callbacks.scene_file_selected) {
          callbacks.scene_file_selected(selected_file);
        }
      }
      if (igMenuItemEx("Reload ", nullptr, "Ctrl+R", false, true)) {
      }
      if (igMenuItemEx("Reload Materials", nullptr, "Ctrl+M", false, true)) {
      }
      igSeparator();
      if (igMenuItemEx("Save...", nullptr, "Ctrl+S", false, true)) {
      }
      igEndMenu();
    }

    if (igBeginMenu("Reference image", true)) {
      if (igMenuItemEx("Open...", nullptr, nullptr, false, true)) {
        auto selected_file = open_file({"Supported images", "*.exr;*.png;*.hdr;*.pfm;*.jpg;*.bmp;*.tga"});
        if ((selected_file.empty() == false) && callbacks.reference_image_selected) {
          callbacks.reference_image_selected(selected_file);
        }
      }
      igEndMenu();
    }
    igEndMainMenuBar();
  }
  igEnd();

  if (_integrator_options.values.empty() == false) {
    igSetNextWindowPos({igGetFontSize(), 2.0f * igGetFontSize()}, ImGuiCond_Always, {0.0f, 0.0f});
    igBegin(_integrator_name, nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
    igText("Integrator options");
    build_options(_integrator_options);
    igEnd();
  }
}

void UI::render() {
  simgui_render();
}

void UI::cleanup() {
}

bool UI::handle_event(const sapp_event* e) {
  return simgui_handle_event(e);
}

ViewOptions UI::view_options() const {
  return {
    _view_options.get("out_view", uint32_t(OutputView::Result)).to_enum<OutputView>(),
    ViewOptions::ToneMapping | ViewOptions::sRGB,
    _view_options.get("exp", 1.0f).to_float(),
  };
}

void UI::set_current_integrator(Integrator* i) {
  _integrator_name = i->name();
  _integrator_options = i->options();
}

}  // namespace etx
