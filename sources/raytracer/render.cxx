#include "render.hxx"

#include <sokol_app.h>
#include <sokol_gfx.h>

namespace etx {

extern const char* vs_source;
extern const char* fs_source;

struct RenderContextPrivate {
  sg_shader output_shader = {};
  sg_pipeline output_pipeline = {};
};

ETX_PIMPL_IMPLEMENT_ALL(RenderContext, Private);

void RenderContext::init() {
  sg_desc context = {};
  context.context.d3d11.device = sapp_d3d11_get_device();
  context.context.d3d11.device_context = sapp_d3d11_get_device_context();
  context.context.d3d11.depth_stencil_view_cb = sapp_d3d11_get_depth_stencil_view;
  context.context.d3d11.render_target_view_cb = sapp_d3d11_get_render_target_view;
  context.context.depth_format = SG_PIXELFORMAT_NONE;
  sg_setup(context);

  sg_shader_desc shader_desc = {};
  shader_desc.vs.source = vs_source;
  shader_desc.attrs[0].sem_name = "POSITION";
  shader_desc.vs.entry = "vertex_main";

  shader_desc.fs.source = fs_source;
  shader_desc.fs.entry = "fragment_main";
  _private->output_shader = sg_make_shader(shader_desc);

  sg_pipeline_desc pipeline_desc = {};
  pipeline_desc.shader = _private->output_shader;
  _private->output_pipeline = sg_make_pipeline(pipeline_desc);
}

void RenderContext::cleanup() {
  sg_destroy_pipeline(_private->output_pipeline);
  sg_destroy_shader(_private->output_shader);
  sg_shutdown();
}

void RenderContext::start_frame() {
  sg_pass_action pass_action = {};
  pass_action.colors[0].action = SG_ACTION_CLEAR;
  pass_action.colors[0].value = {0.05f, 0.07f, 0.1f, 1.0f};
  sg_apply_viewport(0, 0, sapp_width(), sapp_height(), true);
  sg_begin_default_pass(&pass_action, sapp_width(), sapp_height());

  sg_bindings bindings = {};
  sg_apply_pipeline(_private->output_pipeline);
  sg_apply_bindings(bindings);
  sg_draw(0, 3, 1);
}

void RenderContext::end_frame() {
  sg_end_pass();
  sg_commit();
}

const char* vs_source = R"(

struct VSOutput {
  float4 pos : SV_Position;
};

VSOutput vertex_main(uint vertexIndex : SV_VertexID) {
  float2 pos = float2((vertexIndex << 1) & 2, vertexIndex & 2) * 2.0 - 1.0;

  VSOutput output = (VSOutput)0;
  output.pos = float4(pos * 0.5f, 0.0f, 1.0f);
  return output;
}

)";

const char* fs_source = R"(

float4 fragment_main() : SV_Target0 {
  return float4(1.0f, 1.0f, 0.5f, 1.0f);
}

)";

}  // namespace etx
