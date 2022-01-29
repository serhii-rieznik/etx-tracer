#include "render.hxx"

#include <sokol_app.h>
#include <sokol_gfx.h>

namespace etx {

void RenderContext::init() {
  sg_desc context = {};
  context.context.d3d11.device = sapp_d3d11_get_device();
  context.context.d3d11.device_context = sapp_d3d11_get_device_context();
  context.context.d3d11.depth_stencil_view_cb = sapp_d3d11_get_depth_stencil_view;
  context.context.d3d11.render_target_view_cb = sapp_d3d11_get_render_target_view;
  context.context.depth_format = SG_PIXELFORMAT_NONE;
  sg_setup(context);
}

void RenderContext::cleanup() {
  sg_shutdown();
}

void RenderContext::start_frame() {
  sg_pass_action pass_action = {};
  pass_action.colors[0].action = SG_ACTION_CLEAR;
  pass_action.colors[0].value = {0.05f, 0.07f, 0.1f, 1.0f};
  sg_apply_viewport(0, 0, sapp_width(), sapp_height(), true);
  sg_begin_default_pass(&pass_action, sapp_width(), sapp_height());
}

void RenderContext::end_frame() {
  sg_end_pass();
  sg_commit();
}

}  // namespace etx
