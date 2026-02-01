#include "cpu_renderer.hxx"

#include <etx/rhi/shader/shader_compiler.hxx>

namespace etx {

CPURaytracingRenderer::CPURaytracingRenderer(Raytracing& rt, SceneRepresentation& scene)
  : Renderer(rt.scheduler())
  , _raytracing(rt)
  , _integrator_thread(scene, _raytracing)
  , image_pool(images, images_storage) {
}

CPURaytracingRenderer::~CPURaytracingRenderer() {
}

void CPURaytracingRenderer::init(RHIContext* ctx, SceneRepresentation& scene) {
  constexpr float4 kBlack = {};

  context = ctx;
  Renderer::init(ctx, scene);

  image_pool.init(1024u);
  def_image_handle = image_pool.add_from_data(&kBlack, {1u, 1u}, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
  image_pool.load_images(scheduler);

  ShaderCompiler* compiler = ShaderCompiler::get_global_instance();
  auto vs = compiler->load_and_compile_shader_from_file("shaders/render.hlsl", "vertex_main", RHIShaderStage::Vertex);
  auto fs = compiler->load_and_compile_shader_from_file("shaders/render.hlsl", "fragment_main", RHIShaderStage::Fragment);
  RHIGraphicsPipelineDesc pipeline_desc = {
    .vertex_shader =
      {
        .spirv_data = vs.spirv_data.data(),
        .spirv_size = vs.spirv_data.size(),
        .stage = RHIShaderStage::Vertex,
        .entry_point = "vertex_main",
      },
    .fragment_shader =
      {
        .spirv_data = fs.spirv_data.data(),
        .spirv_size = fs.spirv_data.size(),
        .stage = RHIShaderStage::Fragment,
        .entry_point = "fragment_main",
      },
    .color_attachment_count = 1,
    .color_formats = {ctx->get_swapchain_format()},
  };
  rhi_pipeline = ctx->get_device()->create_graphics_pipeline(pipeline_desc).handle;
  apply_reference_image(ctx, def_image_handle);
}

void CPURaytracingRenderer::prepare_frame(RHIContext* ctx, SceneRepresentation& scene, const FrameData& data) {
  Renderer::prepare_frame(ctx, scene, data);
  _integrator_thread.update();
  const auto frame_data = _raytracing.film().layer(data.view_parameters.view_layer, _raytracing.scene());
  update_image(frame_data);
}

void CPURaytracingRenderer::frame(RHIContext* ctx, SceneRepresentation& scene, const FrameData& data) {
  const RHIViewport viewport = {.width = float(sapp_width()), .height = float(sapp_height())};
  const RHIRect scissor = {.width = uint32_t(sapp_width()), .height = uint32_t(sapp_height())};
  const RenderParameters render_params = {
    .view = data.view_parameters,
    .dimensions =
      {
        sapp_widthf(),
        sapp_heightf(),
        float(_raytracing.film().base_dimensions().x),
        float(_raytracing.film().base_dimensions().y),
      },
    .sample_count = scene.data().options.samples,
    .sample_image_index = get_bindless_descriptor_index(rhi_output_texture),
    .reference_image_index = get_bindless_descriptor_index(rhi_reference_texture),
  };
  data.cmd->set_viewport(viewport);
  data.cmd->set_scissor(scissor);
  data.cmd->set_pipeline(rhi_pipeline);
  data.cmd->push_constants(&render_params, sizeof(render_params));
  data.cmd->draw({.vertex_count = 3});
}

void CPURaytracingRenderer::cleanup(RHIContext* ctx) {
  _integrator_thread.stop(Integrator::Stop::Immediate);
  _camera_controller.reset();

  ctx->get_device()->destroy_texture(rhi_output_texture);
  ctx->get_device()->destroy_texture(rhi_reference_texture);
  ctx->get_device()->destroy_pipeline(rhi_pipeline);

  image_pool.remove(ref_image_handle);
  image_pool.remove(def_image_handle);
  image_pool.cleanup();
}

bool CPURaytracingRenderer::is_running() const {
  return _integrator_thread.running();
}

void CPURaytracingRenderer::start() {
  _raytracing.film().clear(Film::ClearEverything);
  _integrator_thread.run();
}

void CPURaytracingRenderer::stop() {
  _integrator_thread.stop(Integrator::Stop::Immediate);
}

void CPURaytracingRenderer::restart() {
  _integrator_thread.restart();
}

void CPURaytracingRenderer::on_camera_changed(SceneRepresentation& scene) {
  _raytracing.film().set_pixel_size(8u);
  _integrator_thread.restart();
}

void CPURaytracingRenderer::on_camera_become_steady(SceneRepresentation& scene) {
  _raytracing.film().set_pixel_size(1u);
  _integrator_thread.restart();
}

void CPURaytracingRenderer::on_scene_changed(SceneRepresentation& scene) {
  _integrator_thread.reset_scene_hashes();
  _raytracing.film().clear(Film::ClearEverything);
  if (_integrator_thread.running()) {
    _integrator_thread.restart();
  }
}

Integrator* CPURaytracingRenderer::current_integrator() const {
  return _integrator_thread.integrator();
}

void CPURaytracingRenderer::set_integrator(Integrator* i) {
  _integrator_thread.set_integrator(i);
}

Integrator** CPURaytracingRenderer::integrator_list() {
  return _integrator_array;
}

uint64_t CPURaytracingRenderer::integrator_count() const {
  return std::size(_integrator_array);
}

void CPURaytracingRenderer::apply_reference_image(RHIContext* ctx, uint32_t handle) {
  const auto& img = image_pool.get(handle);
  if (rhi_reference_texture) {
    ctx->get_device()->destroy_texture(rhi_reference_texture);
    rhi_reference_texture = {};
  }
  RHITextureDesc desc = {
    .width = img.isize.x,
    .height = img.isize.y,
    .format = (img.format == Image::Format::RGBA32F) ? RHITextureFormat::R32G32B32A32_FLOAT : RHITextureFormat::R8G8B8A8_UNORM,
    .usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst,
  };
  rhi_reference_texture = ctx->get_device()->create_texture(desc).handle;
  const void* data_ptr = (img.format == Image::Format::RGBA32F) ? (const void*)img.pixels.f32.a : (const void*)img.pixels.u8.a;
  ctx->get_device()->update_texture(rhi_reference_texture, data_ptr, 0, 0);
}

void CPURaytracingRenderer::set_reference_image(const char* file_name) {
  image_pool.remove(ref_image_handle);
  ref_image_handle = image_pool.add_from_file(file_name, 0, {}, {1.0f, 1.0f});
  image_pool.load_images(scheduler);
  apply_reference_image(context, ref_image_handle);
}

void CPURaytracingRenderer::set_reference_image(const float4 data[], const uint2 dimensions) {
  image_pool.remove(ref_image_handle);
  ref_image_handle = image_pool.add_from_data(data, dimensions, 0u, {}, {1.0f, 1.0f});
  image_pool.load_images(scheduler);
  apply_reference_image(context, ref_image_handle);
}

void CPURaytracingRenderer::update_image(const float4* camera) {
  ETX_PROFILER_SCOPE();

  std::vector<float4> black_image;

  const void* data_ptr = camera;
  if (data_ptr == nullptr) {
    black_image.resize(_raytracing.film().total_pixel_count(), {});
    data_ptr = black_image.data();
  }

  context->get_device()->update_texture(rhi_output_texture, data_ptr, 0, 0);
}

void CPURaytracingRenderer::set_output_dimensions(const uint2& dim) {
  if (output_dimensions == dim) {
    return;
  }

  stop();

  output_dimensions = {std::max(1u, dim.x), std::max(1u, dim.y)};

  if (rhi_output_texture) {
    context->get_device()->destroy_texture(rhi_output_texture);
    rhi_output_texture = {};
  }

  RHITextureDesc desc = {
    .width = output_dimensions.x,
    .height = output_dimensions.y,
    .format = RHITextureFormat::R32G32B32A32_FLOAT,
    .usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst | RHITextureUsage::Storage,
  };
  rhi_output_texture = context->get_device()->create_texture(desc).handle;
}

}  // namespace etx
