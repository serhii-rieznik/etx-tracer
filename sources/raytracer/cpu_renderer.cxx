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
  const uint64_t previous_scene_revision = _integrator_thread.scene_revision();
  const SceneUpdateScope scene_update_scope = consume_scene_update_request();
  if (scene_update_scope != SceneUpdateScope::None) {
    _integrator_thread.request_scene_check(scene_update_scope);
  }
  if (_output_dimensions != _raytracing.film().base_dimensions()) {
    _integrator_thread.commit_scene_changes();
  }

  const bool preview_iteration_completed = _preview_active ? _integrator_thread.update_integrator() : false;
  if (_preview_active == false) {
    _integrator_thread.update();
  }
  if (_integrator_thread.scene_revision() != previous_scene_revision) {
    _last_uploaded_completed_iterations = 0u;
    start_render_timing();
  }
  const Integrator::Status& status = _integrator_thread.status();
  const uint32_t view_layer = frame_data.view_parameters.view_layer;
  const bool completed_iteration_available = _preview_active ? preview_iteration_completed : (status.completed_iterations > _last_uploaded_completed_iterations);
  const bool view_layer_changed = (status.completed_iterations > 0u) && (view_layer != _last_uploaded_view_layer);
  if (completed_iteration_available || view_layer_changed) {
    const float4* film_layer_data = _raytracing.film().layer(view_layer, _raytracing.scene().options.radiance_clamp);
    if (update_image(ctx, frame_data.cmd, film_layer_data)) {
      _last_uploaded_completed_iterations = status.completed_iterations;
      _last_uploaded_view_layer = view_layer;
    } else if (_runtime_failure_reason.empty() == false) {
      stop();
      return;
    }
  }

  if (_preview_active) {
    bool preview_resolution_changed = false;
    if (preview_iteration_completed) {
      preview_resolution_changed = _preview_resolution.update(status.last_iteration_time, true);
      if (preview_resolution_changed) {
        _raytracing.film().set_pixel_size(_preview_resolution.pixel_size());
      }
    }

    const bool scene_changes_pending = _integrator_thread.scene_changes_pending();
    if (scene_changes_pending && (preview_iteration_completed || (_integrator_thread.running() == false))) {
      _integrator_thread.commit_scene_changes();
    } else if (preview_resolution_changed) {
      restart_render_at_pixel_size(_preview_resolution.pixel_size());
    }
  }

  const Integrator* integrator = current_integrator();
  if (_render_timing_active && (integrator != nullptr) && (integrator->state() == Integrator::State::Stopped)) {
    stop_render_timing();
  }
}

void CPURaytracingRenderer::cleanup(RHIContext& ctx) {
  _integrator_thread.stop(Integrator::Stop::Immediate);
  stop_render_timing();
  _camera_controller.reset();

  const RHIResult wait_result = ctx.wait_idle();
  if (wait_result != RHIResult::Success) {
    log::warning("CPU RT: wait_idle failed during cleanup (%u)", static_cast<uint32_t>(wait_result));
  }

  ctx.device().destroy_texture(_output_texture);
  for (uint32_t i = 0u; i < kRHIMaxFrames; ++i) {
    ctx.device().destroy_buffer(_output_staging_buffers[i]);
    _output_staging_buffers[i] = {};
    _output_staging_buffer_sizes[i] = 0u;
  }
  _output_texture = {};
  _output_texture_state = RHIResourceState::Undefined;
  _last_uploaded_completed_iterations = 0u;
  _last_uploaded_view_layer = kInvalidIndex;
  _display_output_valid = false;
  _runtime_failure_reason.clear();
  reset_preview_state();
}

bool CPURaytracingRenderer::is_running() const {
  return _integrator_thread.running();
}

RendererStatus CPURaytracingRenderer::status() const {
  RendererStatus result = {
    .mode = RendererMode::CPURaytracing,
  };
  result.output_stale = display_texture().valid() && (_last_uploaded_completed_iterations == 0u);
  const Integrator* integrator = current_integrator();
  if ((integrator == nullptr) || (integrator->can_run() == false)) {
    if (_runtime_failure_reason.empty() == false) {
      result.state = RendererStatusState::Failed;
      result.message = _runtime_failure_reason;
    }
    return result;
  }

  const Integrator::Status& status = _integrator_thread.status();
  result.progress_kind = RendererProgressKind::Samples;
  result.completed_units = status.completed_iterations;
  result.total_units = std::max(1u, _raytracing.scene().options.samples);
  const Integrator::PathProgress path_progress = integrator->path_progress();
  switch (path_progress.phase) {
    case Integrator::PathProgress::Phase::Light:
      result.path_phase = RendererPathPhase::Light;
      break;
    case Integrator::PathProgress::Phase::Camera:
      result.path_phase = RendererPathPhase::Camera;
      break;
    default:
      break;
  }
  result.completed_path_count = path_progress.completed_path_count;
  result.total_path_count = path_progress.total_path_count;

  if (_runtime_failure_reason.empty() == false) {
    result.state = RendererStatusState::Failed;
    result.message = _runtime_failure_reason;
    result.completed_units = _last_uploaded_completed_iterations;
  } else if (integrator->failed()) {
    result.state = RendererStatusState::Failed;
    result.message = integrator->failure_reason();
  } else {
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
  }

  result.elapsed_seconds = _last_render_elapsed_seconds;
  if (_render_timing_active) {
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - _render_started_at).count();
  }
  result.elapsed_available = _render_timing_active || (result.elapsed_seconds > 0.0);
  if (result.state != RendererStatusState::Failed) {
    if ((result.elapsed_seconds > 0.0) && (result.completed_units > 0u) && (result.completed_units < result.total_units)) {
      const double seconds_per_sample = result.elapsed_seconds / static_cast<double>(result.completed_units);
      result.remaining_seconds = seconds_per_sample * static_cast<double>(result.total_units - result.completed_units);
      result.remaining_available = true;
    } else if (result.completed_units >= result.total_units) {
      result.remaining_seconds = 0.0;
      result.remaining_available = true;
    }
  }
  return result;
}

RendererControlState CPURaytracingRenderer::control_state() const {
  const Integrator* integrator = current_integrator();
  if ((integrator == nullptr) || (integrator->can_run() == false) || (_runtime_failure_reason.empty() == false)) {
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
  if (_runtime_failure_reason.empty() == false) {
    return;
  }

  reset_preview_state();
  _raytracing.film().set_pixel_size(1u);
  _raytracing.film().clear(Film::ClearEverything);
  _last_uploaded_completed_iterations = 0u;
  _display_output_valid = false;
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
  _last_uploaded_completed_iterations = 0u;
  _display_output_valid = false;
  start_render_timing();
  _integrator_thread.restart();
}

void CPURaytracingRenderer::on_camera_changed(SceneRepresentation& scene) {
  (void)scene;
  _preview_camera_active = true;
  if (update_preview_active_state()) {
    _raytracing.film().set_pixel_size(_preview_resolution.pixel_size());
    _integrator_thread.stop(Integrator::Stop::Immediate);
  }
  _last_uploaded_completed_iterations = 0u;
  start_render_timing();
}

void CPURaytracingRenderer::on_camera_become_steady(SceneRepresentation& scene) {
  (void)scene;
  _preview_camera_active = false;
  if (update_preview_active_state() && (_preview_active == false)) {
    restart_render_at_pixel_size(1u);
  }
}

void CPURaytracingRenderer::on_scene_changed(SceneRepresentation& scene) {
  Renderer::on_scene_changed(scene);
  _last_uploaded_completed_iterations = 0u;
  start_render_timing();
}

void CPURaytracingRenderer::on_scene_transforms_changed(SceneRepresentation& scene) {
  Renderer::on_scene_transforms_changed(scene);
  if (_preview_active && (_integrator_thread.running() == false)) {
    _integrator_thread.request_scene_check(SceneUpdateScope::Transforms);
    _integrator_thread.commit_scene_changes();
  }
  _last_uploaded_completed_iterations = 0u;
  start_render_timing();
}

void CPURaytracingRenderer::on_scene_transform_interaction_started(SceneRepresentation& scene) {
  (void)scene;
  _preview_transform_active = true;
  if (update_preview_active_state()) {
    _raytracing.film().set_pixel_size(_preview_resolution.pixel_size());
    _integrator_thread.stop(Integrator::Stop::Immediate);
    _last_uploaded_completed_iterations = 0u;
    start_render_timing();
  }
}

void CPURaytracingRenderer::on_scene_transform_interaction_finished(SceneRepresentation& scene) {
  (void)scene;
  _preview_transform_active = false;
  if (update_preview_active_state() && (_preview_active == false)) {
    _integrator_thread.request_scene_check(SceneUpdateScope::Transforms);
    restart_render_at_pixel_size(1u);
  }
}

void CPURaytracingRenderer::restart_render_at_pixel_size(uint32_t pixel_size) {
  _integrator_thread.stop(Integrator::Stop::Immediate);
  _raytracing.film().set_pixel_size(pixel_size);
  _last_uploaded_completed_iterations = 0u;
  start_render_timing();
  _integrator_thread.run();
  _integrator_thread.update();
}

Integrator* CPURaytracingRenderer::current_integrator() const {
  return _integrator_thread.integrator();
}

void CPURaytracingRenderer::set_integrator(Integrator* i) {
  _integrator_thread.set_integrator(i);
  _last_uploaded_completed_iterations = 0u;
  _display_output_valid = false;
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

bool CPURaytracingRenderer::update_image(RHIContext& ctx, RHICommandBuffer cmd, const float4* camera) {
  ETX_PROFILER_SCOPE();

  if (_output_texture.valid() == false) {
    _runtime_failure_reason = "CPU renderer output texture is unavailable";
    log::error("%s", _runtime_failure_reason.c_str());
    return false;
  }
  if (cmd.valid() == false) {
    _runtime_failure_reason = "CPU renderer received an invalid output command buffer";
    log::error("%s", _runtime_failure_reason.c_str());
    return false;
  }

  if (_output_dimensions != _raytracing.film().base_dimensions()) {
    _runtime_failure_reason = "CPU renderer output dimensions do not match the film dimensions";
    log::error("%s", _runtime_failure_reason.c_str());
    return false;
  }
  const uint64_t output_pixel_count = static_cast<uint64_t>(_output_dimensions.x) * static_cast<uint64_t>(_output_dimensions.y);

  std::vector<float4> black_image;

  const void* data_ptr = camera;
  if (data_ptr == nullptr) {
    black_image.resize(_raytracing.film().total_pixel_count(), {});
    data_ptr = black_image.data();
  }

  const uint64_t upload_size = output_pixel_count * sizeof(float4);
  const uint32_t frame_index = ctx.get_current_frame_index();
  RHIBindlessHandle& staging_buffer = _output_staging_buffers[frame_index];
  uint64_t& staging_buffer_size = _output_staging_buffer_sizes[frame_index];
  if ((staging_buffer.valid() == false) || (staging_buffer_size != upload_size)) {
    if (staging_buffer.valid()) {
      ctx.device().destroy_buffer(staging_buffer);
      staging_buffer = {};
      staging_buffer_size = 0u;
    }

    const RHIBufferDesc desc = {
      .size = upload_size,
      .usage = RHIBufferUsage::TransferSrc,
      .host_visible = true,
    };
    const RHICreateBindlessResult create_result = ctx.device().create_buffer(desc);
    if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
      if (create_result.handle.valid()) {
        ctx.device().destroy_buffer(create_result.handle);
      }
      _runtime_failure_reason = "CPU renderer failed to create its output staging buffer (" + std::to_string(static_cast<uint32_t>(create_result.result)) + ")";
      log::error("%s", _runtime_failure_reason.c_str());
      return false;
    }
    staging_buffer = create_result.handle;
    staging_buffer_size = upload_size;
  }

  const RHIResult update_result = ctx.device().update_buffer(staging_buffer, data_ptr, upload_size);
  if (update_result != RHIResult::Success) {
    _runtime_failure_reason = "CPU renderer failed to update its output staging buffer (" + std::to_string(static_cast<uint32_t>(update_result)) + ")";
    log::error("%s", _runtime_failure_reason.c_str());
    return false;
  }

  ctx.cmd_texture_barrier(cmd, _output_texture, _output_texture_state, RHIResourceState::TransferDst);
  ctx.cmd_copy_buffer_to_texture(cmd, staging_buffer, _output_texture, _output_dimensions.x, _output_dimensions.y);
  ctx.cmd_texture_barrier(cmd, _output_texture, RHIResourceState::TransferDst, RHIResourceState::ShaderReadOnly);
  _output_texture_state = RHIResourceState::ShaderReadOnly;
  _display_output_valid = true;
  return true;
}

void CPURaytracingRenderer::set_output_dimensions(RHIContext& ctx, const uint2& dim) {
  const uint2 output_dimensions = {max(1u, dim.x), max(1u, dim.y)};
  const bool output_state_matches_context = ctx.valid() ? _output_texture.valid() : (_output_texture.valid() == false);
  if ((_output_dimensions == output_dimensions) && output_state_matches_context) {
    return;
  }

  stop();

  if (ctx.valid() == false) {
    if (_output_texture.valid()) {
      log::error("Cannot replace the CPU renderer output texture without a valid RHI context");
      return;
    }
    _output_dimensions = output_dimensions;
    return;
  }

  RHITextureDesc desc = {
    .width = output_dimensions.x,
    .height = output_dimensions.y,
    .format = RHITextureFormat::R32G32B32A32_FLOAT,
    .usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst,
  };
  const RHICreateBindlessResult create_result = ctx.device().create_texture(desc);
  if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
    if (create_result.handle.valid()) {
      ctx.device().destroy_texture(create_result.handle);
    }
    _runtime_failure_reason = "CPU renderer failed to create its output texture (" + std::to_string(static_cast<uint32_t>(create_result.result)) + ")";
    log::error("%s", _runtime_failure_reason.c_str());
    return;
  }
  const RHIResult replacement_wait_result = ctx.wait_idle();
  if (replacement_wait_result != RHIResult::Success) {
    ctx.device().destroy_texture(create_result.handle);
    _runtime_failure_reason = "CPU renderer failed to synchronize before replacing its output texture (" + std::to_string(static_cast<uint32_t>(replacement_wait_result)) + ")";
    log::error("%s", _runtime_failure_reason.c_str());
    return;
  }

  const RHITexture previous_output_texture = _output_texture;
  _output_texture = create_result.handle;
  _output_dimensions = output_dimensions;
  _output_texture_state = RHIResourceState::Undefined;
  _last_uploaded_completed_iterations = 0u;
  _last_uploaded_view_layer = kInvalidIndex;
  _display_output_valid = false;
  _runtime_failure_reason.clear();
  if (previous_output_texture.valid()) {
    const RHIResult destroy_result = ctx.device().destroy_texture(previous_output_texture);
    if (destroy_result != RHIResult::Success) {
      log::warning("Failed to destroy the previous CPU renderer output texture (%u)", static_cast<uint32_t>(destroy_result));
    }
  }
}

}  // namespace etx
