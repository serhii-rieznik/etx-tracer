#include "gpu_renderer.hxx"
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/render/host/scene_representation.hxx>

namespace etx {

GPURaytracingRenderer::GPURaytracingRenderer(TaskScheduler& s)
  : Renderer(s) {
}

GPURaytracingRenderer::~GPURaytracingRenderer() {
}

void GPURaytracingRenderer::init(RHIContext* ctx, SceneRepresentation& scene) {
  Renderer::init(ctx, scene);

  /*
  auto device = render_context.get_device();
  auto compiler = ShaderCompiler::get_global_instance();
  auto cs = compiler->load_and_compile_shader_from_file("shaders/gpu_rt.hlsl", "compute_main", RHIShaderStage::Compute);

  RHIComputePipelineDesc desc = {};
  desc.compute_shader.spirv_data = cs.spirv_data.data();
  desc.compute_shader.spirv_size = cs.spirv_data.size();
  desc.compute_shader.stage = RHIShaderStage::Compute;
  desc.compute_shader.entry_point = "compute_main";

  _pipeline = device->create_compute_pipeline(desc).handle;
  _initialized = true;
  */
}

void GPURaytracingRenderer::frame(RHIContext* ctx, SceneRepresentation& scene, const FrameData& frame_data) {
  Renderer::frame(ctx, scene, frame_data);
  /*
  if (!_initialized || _pipeline.valid() == false)
    return;

  auto cmd = render_context.current_command_buffer();
  if (cmd == nullptr)
    return;

  struct GPUConstants {
    uint32_t as_index;
    uint32_t output_image_index;
    uint32_t pad[2];
  } constants = {
    .as_index = get_bindless_descriptor_index(_tlas),
    .output_image_index = get_bindless_descriptor_index(render_context.get_output_texture()),
  };

  cmd->set_pipeline(_pipeline);
  cmd->push_constants(&constants, sizeof(constants));

  uint2 dim = render_context.get_output_dimensions();
  cmd->dispatch({.group_count_x = (dim.x + 15) / 16, .group_count_y = (dim.y + 15) / 16});
  */
}

void GPURaytracingRenderer::cleanup(RHIContext* ctx) {
  /*
  if (_pipeline.valid()) {
    render_context.get_device()->destroy_pipeline(_pipeline);
    _pipeline = {};
  }
  _initialized = false;
  */
}

void GPURaytracingRenderer::process_event(const sapp_event* e) {
  Renderer::process_event(e);
}

void GPURaytracingRenderer::on_camera_changed(SceneRepresentation& scene) {
}

void GPURaytracingRenderer::on_scene_changed(SceneRepresentation& scene) {
}

void GPURaytracingRenderer::build_acceleration_structures(RenderContext& render_context, SceneRepresentation& scene) {
}

}  // namespace etx
