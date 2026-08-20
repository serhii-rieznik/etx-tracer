#include "cpu_renderer.hxx"

#include <etx/core/log.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include <algorithm>

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
  const Integrator* integrator = current_integrator();
  if (_render_timing_active && (integrator != nullptr) && (integrator->state() == Integrator::State::Stopped)) {
    stop_render_timing();
  }

  const auto film_layer_data = _raytracing.film().layer(frame_data.view_parameters.view_layer, _raytracing.scene().options.radiance_clamp);
  update_image(ctx, film_layer_data);
}

void CPURaytracingRenderer::cleanup(RHIContext& ctx) {
  _integrator_thread.stop(Integrator::Stop::Immediate);
  stop_render_timing();
  _camera_controller.reset();

  ctx.device().destroy_texture(_output_texture);
}

bool CPURaytracingRenderer::is_running() const {
  return _integrator_thread.running();
}

RendererStatus CPURaytracingRenderer::status() const {
  RendererStatus result = {
    .mode = RendererMode::CPURaytracing,
  };
  const Integrator* integrator = current_integrator();
  if ((integrator == nullptr) || (integrator->can_run() == false)) {
    return result;
  }

  const Integrator::Status& status = _integrator_thread.status();
  result.progress_kind = RendererProgressKind::Samples;
  result.completed_units = status.completed_iterations;
  result.total_units = std::max(1u, _raytracing.scene().options.samples);

  switch (integrator->state()) {
    case Integrator::State::Running:
      result.state = RendererStatusState::Running;
      break;
    case Integrator::State::WaitingForCompletion:
      result.state = RendererStatusState::Finishing;
      break;
    default:
      result.state = (result.completed_units >= result.total_units) ? RendererStatusState::Completed : RendererStatusState::Idle;
      break;
  }

  result.elapsed_seconds = _last_render_elapsed_seconds;
  if (_render_timing_active) {
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - _render_started_at).count();
  }
  result.elapsed_available = _render_timing_active || (result.elapsed_seconds > 0.0);
  if ((result.elapsed_seconds > 0.0) && (result.completed_units > 0u) && (result.completed_units < result.total_units)) {
    const double seconds_per_sample = result.elapsed_seconds / static_cast<double>(result.completed_units);
    result.remaining_seconds = seconds_per_sample * static_cast<double>(result.total_units - result.completed_units);
    result.remaining_available = true;
  } else if (result.completed_units >= result.total_units) {
    result.remaining_seconds = 0.0;
    result.remaining_available = true;
  }
  return result;
}

RendererControlState CPURaytracingRenderer::control_state() const {
  const Integrator* integrator = current_integrator();
  if ((integrator == nullptr) || (integrator->can_run() == false)) {
    return {};
  }

  RendererControlState result = {};
  switch (integrator->state()) {
    case Integrator::State::Running:
      result.can_finish = true;
      result.can_stop = true;
      result.can_restart = true;
      break;
    case Integrator::State::WaitingForCompletion:
      result.can_stop = true;
      break;
    default:
      result.can_run = true;
      break;
  }
  return result;
}

void CPURaytracingRenderer::start() {
  _raytracing.film().clear(Film::ClearEverything);
  start_render_timing();
  _integrator_thread.run();
}

void CPURaytracingRenderer::stop() {
  _integrator_thread.stop(Integrator::Stop::Immediate);
  stop_render_timing();
}

void CPURaytracingRenderer::finish() {
  _integrator_thread.stop(Integrator::Stop::WaitForCompletion);
}

void CPURaytracingRenderer::restart() {
  start_render_timing();
  _integrator_thread.restart();
}

void CPURaytracingRenderer::on_camera_changed(SceneRepresentation& scene) {
  _raytracing.film().set_pixel_size(8u);
  start_render_timing();
  _integrator_thread.restart();
}

void CPURaytracingRenderer::on_camera_become_steady(SceneRepresentation& scene) {
  _raytracing.film().set_pixel_size(1u);
  start_render_timing();
  _integrator_thread.restart();
}

void CPURaytracingRenderer::on_scene_changed(SceneRepresentation& scene) {
  Renderer::on_scene_changed(scene);
  start_render_timing();
}

Integrator* CPURaytracingRenderer::current_integrator() const {
  return _integrator_thread.integrator();
}

void CPURaytracingRenderer::set_integrator(Integrator* i) {
  _integrator_thread.set_integrator(i);
  reset_render_timing();
}

Integrator** CPURaytracingRenderer::integrator_list() {
  return _integrator_array;
}

uint64_t CPURaytracingRenderer::integrator_count() const {
  return std::size(_integrator_array);
}

void CPURaytracingRenderer::start_render_timing() {
  _render_started_at = std::chrono::steady_clock::now();
  _last_render_elapsed_seconds = 0.0;
  _render_timing_active = true;
}

void CPURaytracingRenderer::stop_render_timing() {
  if (_render_timing_active) {
    _last_render_elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - _render_started_at).count();
    _render_timing_active = false;
  }
}

void CPURaytracingRenderer::reset_render_timing() {
  _render_started_at = {};
  _last_render_elapsed_seconds = 0.0;
  _render_timing_active = false;
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
  const uint2 output_dimensions = {max(1u, dim.x), max(1u, dim.y)};
  const bool output_state_matches_context = ctx.valid() ? _output_texture.valid() : (_output_texture.valid() == false);
  if ((_output_dimensions == output_dimensions) && output_state_matches_context) {
    return;
  }

  stop();

  if (_output_texture.valid()) {
    if (ctx.valid() == false) {
      log::error("Cannot release the CPU renderer output texture without a valid RHI context");
      return;
    }
    ctx.device().destroy_texture(_output_texture);
    _output_texture = {};
  }

  _output_dimensions = output_dimensions;
  if (ctx.valid() == false) {
    return;
  }

  RHITextureDesc desc = {
    .width = _output_dimensions.x,
    .height = _output_dimensions.y,
    .format = RHITextureFormat::R32G32B32A32_FLOAT,
    .usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst,
  };
  _output_texture = ctx.device().create_texture(desc).handle;
}

}  // namespace etx
