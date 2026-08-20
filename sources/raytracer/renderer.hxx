#pragma once
#include <etx/core/core.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/rhi/rhi_types.hxx>

#include "render_context.hxx"
#include <etx/engine/camera_controller.hxx>

#include <memory>
#include <string>
#include <vector>

struct sapp_event;

namespace etx {

struct TaskScheduler;

enum class RendererMode {
  CPURaytracing,
  Rasterization,
  GPURaytracing,
};

enum class RendererPreparationState : uint32_t {
  Ready,
  Preparing,
  Failed,
};

enum class RendererPreparationStepState : uint32_t {
  QueuedForShaderCompilation,
  CompilingSpirV,
  QueuedForDriver,
  CheckingCache,
  DriverCompiling,
  Complete,
  Failed,
};

struct RendererPreparationStepStatus {
  std::string name = {};
  std::string detail = {};
  RendererPreparationStepState state = RendererPreparationStepState::QueuedForShaderCompilation;
  uint64_t spirv_size_bytes = 0u;
  double elapsed_ms = 0.0;
  bool cache_hit = false;
};

struct RendererPreparationStatus {
  RendererPreparationState state = RendererPreparationState::Ready;
  std::string phase = "Ready";
  std::string message = {};
  uint32_t completed_steps = 0u;
  uint32_t total_steps = 0u;
  uint32_t worker_count = 0u;
  double elapsed_seconds = 0.0;
  bool cancelable = false;
  std::vector<RendererPreparationStepStatus> steps = {};
};

enum class RendererStatusState : uint32_t {
  Unavailable,
  Idle,
  Preparing,
  Running,
  Finishing,
  Completed,
  Failed,
};

enum class RendererProgressKind : uint32_t {
  None,
  Samples,
  Steps,
};

struct RendererStatus {
  RendererMode mode = RendererMode::CPURaytracing;
  RendererStatusState state = RendererStatusState::Unavailable;
  RendererProgressKind progress_kind = RendererProgressKind::None;
  uint32_t completed_units = 0u;
  uint32_t total_units = 0u;
  double elapsed_seconds = 0.0;
  double remaining_seconds = 0.0;
  bool elapsed_available = false;
  bool remaining_available = false;
};

enum class RendererMemoryLocation : uint32_t {
  CPU,
  GPUDevice,
  GPUHostVisible,
};

struct RendererMemoryEntry {
  std::string category = {};
  std::string name = {};
  RendererMemoryLocation location = RendererMemoryLocation::CPU;
  uint64_t bytes = 0u;
  uint32_t allocation_count = 0u;
};

struct RendererMemoryStats {
  std::vector<RendererMemoryEntry> entries = {};
  uint32_t wavefront_path_capacity = 0u;
  uint32_t light_vertex_capacity = 0u;
  uint32_t light_vertex_count = 0u;
  uint32_t tile_index = 0u;
  uint32_t tile_count = 0u;
  uint32_t max_path_length = 0u;
};

struct RendererKernelTiming {
  std::string name = {};
  uint64_t dispatch_count = 0u;
  double total_ms = 0.0;
  double average_ms = 0.0;
  double percentage = 0.0;
};

struct RendererKernelTimingStats {
  std::vector<RendererKernelTiming> kernels = {};
  uint64_t dropped_dispatch_count = 0u;
  double total_ms = 0.0;
  bool supported = false;
  bool enabled = false;
};

struct RendererControlState {
  bool can_run = false;
  bool can_finish = false;
  bool can_stop = false;
  bool can_restart = false;
};

struct PreviewResolutionController {
  PreviewResolutionController(uint32_t initial_pixel_size, uint32_t maximum_pixel_size)
    : _pixel_size(initial_pixel_size > 0u ? initial_pixel_size : 1u)
    , _initial_pixel_size(_pixel_size)
    , _maximum_pixel_size(maximum_pixel_size > _pixel_size ? maximum_pixel_size : _pixel_size) {
  }

  void begin() {
    _pixel_size = _initial_pixel_size;
    _active = true;
    reset_observations();
  }

  void end() {
    _active = false;
    reset_observations();
  }

  bool active() const {
    return _active;
  }

  uint32_t pixel_size() const {
    return _pixel_size;
  }

  bool update(double elapsed_seconds, bool output_completed) {
    if ((_active == false) || (elapsed_seconds <= 0.0)) {
      return false;
    }

    if (output_completed == false) {
      _seconds_without_output += elapsed_seconds;
      if (_seconds_without_output < kMaximumOutputLatencySeconds) {
        return false;
      }

      reset_observations();
      return increase_pixel_size();
    }

    _seconds_without_output = 0.0;
    if (elapsed_seconds > kSlowOutputSeconds) {
      _slow_output_count += 1u;
      _fast_output_count = 0u;
      if (_slow_output_count < kSlowOutputThreshold) {
        return false;
      }

      reset_observations();
      return increase_pixel_size();
    }

    if (elapsed_seconds < kFastOutputSeconds) {
      _fast_output_count += 1u;
      _slow_output_count = 0u;
      if (_fast_output_count < kFastOutputThreshold) {
        return false;
      }

      reset_observations();
      return decrease_pixel_size();
    }

    _slow_output_count = 0u;
    _fast_output_count = 0u;
    return false;
  }

 private:
  bool increase_pixel_size() {
    if (_pixel_size >= _maximum_pixel_size) {
      return false;
    }
    _pixel_size = (_pixel_size > (_maximum_pixel_size / 2u)) ? _maximum_pixel_size : (_pixel_size * 2u);
    return true;
  }

  bool decrease_pixel_size() {
    if (_pixel_size <= 1u) {
      return false;
    }
    _pixel_size = (_pixel_size + 1u) / 2u;
    return true;
  }

  void reset_observations() {
    _seconds_without_output = 0.0;
    _slow_output_count = 0u;
    _fast_output_count = 0u;
  }

 private:
  static constexpr double kTargetOutputSeconds = 1.0 / 30.0;
  static constexpr double kSlowOutputSeconds = kTargetOutputSeconds * 1.25;
  static constexpr double kFastOutputSeconds = kTargetOutputSeconds * 0.60;
  static constexpr double kMaximumOutputLatencySeconds = kTargetOutputSeconds * 2.0;
  static constexpr uint32_t kSlowOutputThreshold = 2u;
  static constexpr uint32_t kFastOutputThreshold = 6u;

  double _seconds_without_output = 0.0;
  uint32_t _pixel_size = 1u;
  uint32_t _initial_pixel_size = 1u;
  uint32_t _maximum_pixel_size = 1u;
  uint32_t _slow_output_count = 0u;
  uint32_t _fast_output_count = 0u;
  bool _active = false;
};

struct Renderer {
  struct FrameData {
    ViewParameters view_parameters = {};
    RHICommandBuffer cmd = {};
    float dt = 0.0f;
  };

  Renderer(TaskScheduler& s)
    : scheduler(s) {
  }

  virtual ~Renderer() = default;

  virtual void init(RHIContext& ctx, SceneRepresentation& scene) {
    (void)ctx;
    ETX_CRITICAL(_camera_controller == nullptr);
    _camera_controller.reset(new CameraController(scene.mutable_camera()));
    _camera_controller->enable_inertia = false;
    reset_preview_state();
  }

  virtual void update_camera(SceneRepresentation& scene, float dt) {
    ETX_CRITICAL(_camera_controller);

    const bool camera_updated = _camera_controller->update(dt);
    const bool camera_input_active = _camera_controller->camera_navigation_input_active();
    if (camera_updated) {
      if (scene.store_active_camera()) {
        scene.update_medium_bounds();
      }
      request_scene_transform_update();
      _camera_interaction_active = true;
      on_camera_changed(scene);
      return;
    }

    if (_camera_interaction_active && (camera_input_active == false)) {
      _camera_interaction_active = false;
      on_camera_become_steady(scene);
    }
  }

  virtual void render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& data) {
    (void)ctx;
    (void)scene;
    (void)data;
  }

  virtual RHITexture output_texture() const {
    return _output_texture;
  }
  virtual RHITexture display_texture() const {
    return output_texture();
  }
  virtual uint2 output_size() const {
    return _output_dimensions;
  }

  virtual void cleanup(RHIContext& ctx) {
    (void)ctx;
  }

  virtual void process_event(const sapp_event* e) {
    ETX_CRITICAL(_camera_controller);
    _camera_controller->handle_event(e);
  }

  virtual void on_camera_changed(SceneRepresentation& scene) {
  }

  virtual void on_camera_become_steady(SceneRepresentation& scene) {
  }

  virtual void on_scene_changed(SceneRepresentation& scene) {
    (void)scene;
    request_scene_update();
  }

  virtual void on_scene_transforms_changed(SceneRepresentation& scene) {
    (void)scene;
    request_scene_transform_update();
  }

  virtual void on_scene_transform_interaction_started(SceneRepresentation& scene) {
    (void)scene;
  }

  virtual void on_scene_transform_interaction_finished(SceneRepresentation& scene) {
    (void)scene;
  }

  void request_scene_update() {
    _scene_update_scope = SceneUpdateScope::Full;
  }

  void request_scene_transform_update() {
    if (_scene_update_scope == SceneUpdateScope::None) {
      _scene_update_scope = SceneUpdateScope::Transforms;
    }
  }

  SceneUpdateScope consume_scene_update_request() {
    const SceneUpdateScope result = _scene_update_scope;
    _scene_update_scope = SceneUpdateScope::None;
    return result;
  }

  virtual const char* name() const = 0;
  virtual RendererMode mode() const = 0;
  virtual RendererPreparationStatus preparation_status() const {
    return {};
  }

  virtual RendererStatus status() const {
    return {.mode = mode()};
  }

  virtual RendererMemoryStats memory_stats() const {
    return {};
  }

  virtual RendererControlState control_state() const {
    return {};
  }

  virtual bool is_running() const {
    return false;
  }

  virtual void start() {
  }

  virtual void stop() {
  }

  virtual void finish() {
  }

  virtual void restart() {
  }

  virtual void cancel_preparation() {
  }

  CameraController* camera_controller() {
    return _camera_controller.get();
  }

 protected:
  bool update_preview_active_state() {
    const bool preview_active = _preview_camera_active || _preview_transform_active;
    if (preview_active == _preview_active) {
      return false;
    }

    _preview_active = preview_active;
    if (_preview_active) {
      _preview_resolution.begin();
    } else {
      _preview_resolution.end();
    }
    return true;
  }

  void reset_preview_state() {
    _preview_camera_active = false;
    _preview_transform_active = false;
    _preview_active = false;
    _camera_interaction_active = false;
    _preview_resolution.end();
  }

  TaskScheduler& scheduler;
  std::unique_ptr<CameraController> _camera_controller = nullptr;
  uint2 _output_dimensions = {};
  RHITexture _output_texture = {};
  PreviewResolutionController _preview_resolution = {4u, 4u};
  bool _preview_camera_active = false;
  bool _preview_transform_active = false;
  bool _preview_active = false;
  bool _camera_interaction_active = false;
  SceneUpdateScope _scene_update_scope = SceneUpdateScope::Full;
};

}  // namespace etx
