#include <etx/core/core.hxx>
#include <etx/log/log.hxx>

#include <sokol_app.h>
#include <sokol_gfx.h>

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS
#include <cimgui.h>
#include <sokol_imgui.h>

namespace etx {

struct RTApplication {
  void init() {
    sg_desc context = {};
    context.context.d3d11.device = sapp_d3d11_get_device();
    context.context.d3d11.device_context = sapp_d3d11_get_device_context();
    context.context.d3d11.depth_stencil_view_cb = sapp_d3d11_get_depth_stencil_view;
    context.context.d3d11.render_target_view_cb = sapp_d3d11_get_render_target_view;
    context.context.depth_format = SG_PIXELFORMAT_NONE;
    sg_setup(context);

    simgui_desc_t imggui_desc = {};
    imggui_desc.depth_format = context.context.depth_format;
    simgui_setup(imggui_desc);
  }

  void frame() {
    sg_pass_action pass_action = {};
    pass_action.colors[0].action = SG_ACTION_CLEAR;
    pass_action.colors[0].value = {0.05f, 0.07f, 0.1f, 1.0f};
    sg_apply_viewport(0, 0, sapp_width(), sapp_height(), true);

    simgui_new_frame(simgui_frame_desc_t{sapp_width(), sapp_height(), time_measure.lap(), sapp_dpi_scale()});

    igPushItemWidth(-12.0f * igGetFontSize());
    if (igBeginMainMenuBar()) {
      if (igBeginMenu("Scene", true)) {
        if (igMenuItemEx("Open...", nullptr, nullptr, false, true)) {
          // open scene
        }
        igEndMenu();
      }
      if (igBeginMenu("Reference image", true)) {
        if (igMenuItemEx("Open...", nullptr, nullptr, false, true)) {
          // load reference image
        }
        igEndMenu();
      }
      igEndMainMenuBar();
    }

    sg_begin_default_pass(&pass_action, sapp_width(), sapp_height());
    simgui_render();
    sg_end_pass();
    sg_commit();
  }

  void cleanup() {
    simgui_shutdown();
    sg_shutdown();
  }

  void process_event(const sapp_event* e) {
    simgui_handle_event(e);
  }

  TimeMeasure time_measure;
};

extern "C" int main(int argc, char* argv[]) {
  auto s_init = [](void* data) {
    reinterpret_cast<RTApplication*>(data)->init();
  };
  auto s_frame = [](void* data) {
    reinterpret_cast<RTApplication*>(data)->frame();
  };
  auto s_cleanup = [](void* data) {
    reinterpret_cast<RTApplication*>(data)->cleanup();
  };
  auto s_event = [](const sapp_event* e, void* data) {
    reinterpret_cast<RTApplication*>(data)->process_event(e);
  };

  RTApplication app;
  sapp_desc desc = {};
  {
    desc.init_userdata_cb = s_init;
    desc.frame_userdata_cb = s_frame;
    desc.cleanup_userdata_cb = s_cleanup;
    desc.event_userdata_cb = s_event;
    desc.width = 1280;
    desc.height = 720;
    desc.window_title = "etx-tracer";
    desc.high_dpi = true;
    desc.win32_console_utf8 = true;
    desc.win32_console_create = true;
    desc.user_data = &app;
  };
  sapp_run(desc);
  return 0;
}

}  // namespace etx
