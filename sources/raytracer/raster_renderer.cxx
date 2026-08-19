#include "raster_renderer.hxx"
#include <etx/rhi/shader/shader_compiler.hxx>

namespace etx {

RasterizationRenderer::RasterizationRenderer(TaskScheduler& s)
  : Renderer(s) {
}

RasterizationRenderer::~RasterizationRenderer() {
}

void RasterizationRenderer::init(RHIContext& ctx, SceneRepresentation& scene) {
  Renderer::init(ctx, scene);
  _initialized = true;
}

void RasterizationRenderer::render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& data) {
  Renderer::update_camera(scene, data.dt);
  Renderer::render(ctx, scene, data);
}

void RasterizationRenderer::cleanup(RHIContext& ctx) {
  (void)ctx;
  /*
  if (_pipeline.valid()) {
    auto device = render_context.get_device();
    if (device) {
      device->destroy_pipeline(_pipeline);
    }
    _pipeline = {};
  }
  */
}

RendererStatus RasterizationRenderer::status() const {
  return {
    .mode = RendererMode::Rasterization,
    .state = _initialized ? RendererStatusState::Running : RendererStatusState::Unavailable,
  };
}

void RasterizationRenderer::on_scene_changed(SceneRepresentation& scene) {
  Renderer::on_scene_changed(scene);
}

void RasterizationRenderer::create_pipeline() {
  /*
  auto device = render_context.get_device();
  auto& compiler = ShaderCompiler::instance();

  auto vs = compiler->load_and_compile_shader_from_file("shaders/raster.hlsl", "vertex_main", RHIShaderStage::Vertex);
  auto fs =
  compiler->load_and_compile_shader_from_file("shaders/raster.hlsl", "fragment_main", RHIShaderStage::Fragment);
 RHIGraphicsPipelineDesc desc = {}; desc.vertex_shader.spirv_data =
  vs.spirv_data.data(); desc.vertex_shader.spirv_size = vs.spirv_data.size(); desc.vertex_shader.stage = RHIShaderStage::Vertex; desc.vertex_shader.entry_point = "vertex_main";

  desc.fragment_shader.spirv_data = fs.spirv_data.data();
  desc.fragment_shader.spirv_size = fs.spirv_data.size();
  desc.fragment_shader.stage = RHIShaderStage::Fragment;
  desc.fragment_shader.entry_point = "fragment_main";

  desc.vertex_attribute_count = 2;
  desc.vertex_attributes[0] = {.location = 0, .binding = 0, .format = RHIVertexFormat::Float3, .offset = 0};
  desc.vertex_attributes[1] = {.location = 1, .binding = 1, .format = RHIVertexFormat::Float3, .offset = 0};

  desc.vertex_binding_count = 2;
  desc.vertex_bindings[0] = {.binding = 0, .stride = sizeof(float3), .input_rate = RHIVertexInputRate::Vertex};
  desc.vertex_bindings[1] = {.binding = 1, .stride = sizeof(float3), .input_rate = RHIVertexInputRate::Vertex};

  desc.color_attachment_count = 1;
  desc.color_formats[0] = render_context.get_swapchain_format();
  desc.depth_format = render_context.get_depth_format();

  _pipeline = device->create_graphics_pipeline(desc).handle;
  */
}

}  // namespace etx
