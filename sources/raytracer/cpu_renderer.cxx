#include "cpu_renderer.hxx"

#include <etx/rhi/shader/shader_compiler.hxx>

namespace etx {

CPURaytracingRenderer::CPURaytracingRenderer(Raytracing& rt, SceneRepresentation& scene)
  : Renderer(rt.scheduler())
  , _raytracing(rt)
  , _integrator_thread(scene, _raytracing) {
}

CPURaytracingRenderer::~CPURaytracingRenderer() {
}

void CPURaytracingRenderer::init(RHIContext& ctx, SceneRepresentation& scene) {
  Renderer::init(ctx, scene);
}

void CPURaytracingRenderer::render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& frame_data) {
  Renderer::update_camera(scene, frame_data.dt);
  if (consume_scene_update_request()) {
    _integrator_thread.request_scene_check();
  }
  _integrator_thread.update();

  const auto film_layer_data = _raytracing.film().layer(frame_data.view_parameters.view_layer, _raytracing.scene());
  update_image(ctx, film_layer_data);
}

void CPURaytracingRenderer::cleanup(RHIContext& ctx) {
  _integrator_thread.stop(Integrator::Stop::Immediate);
  _camera_controller.reset();

  ctx.device().destroy_texture(_output_texture);
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
  Renderer::on_scene_changed(scene);
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

void CPURaytracingRenderer::update_image(RHIContext& ctx, const float4* camera) {
  ETX_PROFILER_SCOPE();

  std::vector<float4> black_image;

  const void* data_ptr = camera;
  if (data_ptr == nullptr) {
    black_image.resize(_raytracing.film().total_pixel_count(), {});
    data_ptr = black_image.data();
  }

  if (_output_texture.valid()) {
    ctx.device().update_texture(_output_texture, data_ptr, 0, 0);
  }
}

void CPURaytracingRenderer::set_output_dimensions(RHIContext& ctx, const uint2& dim) {
  if (_output_dimensions == dim) {
    return;
  }

  stop();

  _output_dimensions = {max(1u, dim.x), max(1u, dim.y)};

  if (_output_texture.valid()) {
    ctx.device().destroy_texture(_output_texture);
    _output_texture = {};
  }

  RHITextureDesc desc = {
    .width = _output_dimensions.x,
    .height = _output_dimensions.y,
    .format = RHITextureFormat::R32G32B32A32_FLOAT,
    .usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst | RHITextureUsage::Storage,
  };
  _output_texture = ctx.device().create_texture(desc).handle;
}

}  // namespace etx
