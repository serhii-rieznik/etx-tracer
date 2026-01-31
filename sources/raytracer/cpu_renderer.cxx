#include "cpu_renderer.hxx"

namespace etx {

CPURaytracingRenderer::CPURaytracingRenderer(TaskScheduler& s, SceneRepresentation& scene)
  : Renderer(s)
  , _raytracing()
  , _integrator_thread(scene, _raytracing) {
}

CPURaytracingRenderer::~CPURaytracingRenderer() {
}

void CPURaytracingRenderer::init(RenderContext& render_context, SceneRepresentation& scene) {
  Renderer::init(render_context, scene);
}

void CPURaytracingRenderer::frame(RenderContext& render_context, SceneRepresentation& scene, float dt) {
  Renderer::frame(render_context, scene, dt);
  _integrator_thread.update();

  const auto frame_data = _raytracing.film().layer(uint32_t(render_context.get_view_layer()), _raytracing.scene());
  render_context.update_image(frame_data);
}

void CPURaytracingRenderer::cleanup(RenderContext& render_context) {
  _integrator_thread.stop(Integrator::Stop::Immediate);
  _camera_controller.reset();
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

void CPURaytracingRenderer::on_camera_changed(SceneRepresentation& scene, bool path_changed) {
  if (path_changed) {
    _raytracing.film().set_pixel_size(8u);
  } else {
    _raytracing.film().set_pixel_size(1u);
  }
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

}  // namespace etx
