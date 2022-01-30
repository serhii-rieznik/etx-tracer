#include <etx/core/core.hxx>
#include <etx/render/host/image_pool.hxx>

#include "ui.hxx"

#include <sokol_app.h>
#include <sokol_gfx.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>
#include <sokol_imgui.h>

namespace etx {

std::string optional_value_to_string(uint32_t i) {
  switch (OptionalValue::Class(i)) {
    case OptionalValue::Class::Undefined:
      return "OptionalValue::Class::Undefined";
    case OptionalValue::Class::Integer:
      return "OptionalValue::Class::Integer";
    case OptionalValue::Class::Boolean:
      return "OptionalValue::Class::Boolean";
    case OptionalValue::Class::InfoString:
      return "OptionalValue::Class::InfoString";
    case OptionalValue::Class::Float:
      return "OptionalValue::Class::Float";
    case OptionalValue::Class::Enum:
      return "OptionalValue::Class::Enum";
    default:
      return "???";
  }
}

void UI::initialize() {
  simgui_desc_t imggui_desc = {};
  imggui_desc.depth_format = SG_PIXELFORMAT_NONE;
  simgui_setup(imggui_desc);

  _integrator_options = Options{{
    {"info", "VCM (GPU)"},
    {1u, 4096u, 65536u, "spp", "Samples per Pixel / Iterations"},
    {1u, 8u, 4096u, "len", "Max Path Length"},
    {0.0f, 0.0f, 1.0f, "r0", "Initial Radius"},
    {0.0f, 0.0f, 1.0f, "r1", "Radius"},
    {true, "mis", "MIS"},
    {OptionalValue::Class::Boolean, OptionalValue::Class::Count, optional_value_to_string, "val", "Enum Test"},
  }};
}

void UI::build(double dt) {
  simgui_new_frame(simgui_frame_desc_t{sapp_width(), sapp_height(), dt, sapp_dpi_scale()});

  igSetNextWindowPos({igGetFontSize(), 2.0f * igGetFontSize()}, ImGuiCond_Always, {0.0f, 0.0f});
  igBegin("Properties", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);

  for (auto& option : _integrator_options.values) {
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
        if (igTreeNode_Str(option.name.c_str())) {
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

  if (igBeginMainMenuBar()) {
    if (igBeginMenu("Raytracer", true)) {
      if (igMenuItemEx("Path Tracing CPU", nullptr, nullptr, false, true)) {
      }
      if (igMenuItemEx("Path Tracing GPU", nullptr, nullptr, false, true)) {
      }
      if (igMenuItemEx("VCM CPU", nullptr, nullptr, false, true)) {
      }
      if (igMenuItemEx("VCM GPU", nullptr, nullptr, false, true)) {
      }
      igSeparator();
      if (igMenuItemEx("Exit", nullptr, "Ctrl+Q", false, true)) {
      }
      igEndMenu();
    }

    if (igBeginMenu("Scene", true)) {
      if (igMenuItemEx("Open...", nullptr, "Ctrl+O", false, true)) {
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
}

void UI::render() {
  simgui_render();
}

void UI::cleanup() {
}

bool UI::handle_event(const sapp_event* e) {
  return simgui_handle_event(e);
}

void UI::set_options(const Options& options) {
  _integrator_options = options;
}

}  // namespace etx
