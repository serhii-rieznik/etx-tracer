#pragma once
#include <etx/core/core.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/rhi/rhi_types.hxx>

#include "render_context.hxx"
#include <etx/engine/camera_controller.hxx>

#include <functional>
#include <memory>
#include <string>
#include <utility>
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
  double remaining_seconds = 0.0;
  bool remaining_available = false;
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

enum class RendererPathPhase : uint32_t {
  None,
  Light,
  Camera,
};

enum class RendererUPBPPhase : uint32_t {
  None,
  LightPaths,
  LightCompaction,
  DensityIndex,
  CameraPaths,
  CameraEvaluation,
  Finalize,
};

struct RendererUPBPStatus {
  RendererUPBPPhase phase = RendererUPBPPhase::None;
  uint32_t current_light_batch = 0u;
  uint32_t total_light_batches = 0u;
  uint32_t current_camera_batch = 0u;
  uint32_t total_camera_batches = 0u;
  uint32_t active_path_count = 0u;
  uint32_t resident_path_count = 0u;
  uint32_t global_path_count = 0u;
  uint32_t density_batch_count = 0u;
  uint64_t surface_point_count = 0u;
  uint64_t medium_point_count = 0u;
  uint64_t tracking_event_count = 0u;
  uint64_t tracking_event_bytes = 0u;
  uint64_t bp2d_beam_count = 0u;
  uint64_t bb1d_beam_count = 0u;
  uint32_t bp2d_partition_count = 0u;
  uint32_t bb1d_partition_count = 0u;
  uint64_t gpu_memory_used_bytes = 0u;
  uint64_t gpu_memory_budget_bytes = 0u;
  bool density_cache_ready = false;

  bool active() const {
    return phase != RendererUPBPPhase::None;
  }
};

struct RendererStatus {
  RendererMode mode = RendererMode::CPURaytracing;
  RendererStatusState state = RendererStatusState::Unavailable;
  bool output_stale = false;
  bool preview_active = false;
  std::string message = {};
  RendererProgressKind progress_kind = RendererProgressKind::None;
  uint32_t completed_units = 0u;
  uint32_t total_units = 0u;
  RendererPathPhase path_phase = RendererPathPhase::None;
  uint64_t completed_path_count = 0u;
  uint64_t total_path_count = 0u;
  double elapsed_seconds = 0.0;
  double remaining_seconds = 0.0;
  bool elapsed_available = false;
  bool remaining_available = false;
  RendererUPBPStatus upbp = {};
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
  uint32_t max_observed_camera_path_length = 0u;
  uint32_t max_observed_light_path_length = 0u;
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
  uint64_t captured_sample_count = 0u;
  double total_ms = 0.0;
  double capture_elapsed_ms = 0.0;
  bool supported = false;
  bool enabled = false;
};

struct RendererControlState {
  bool can_run = false;
  bool can_finish = false;
  bool can_stop = false;
  bool can_restart = false;
};

constexpr uint32_t kInteractionPreviewPixelSize = 4u;

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
  }

  struct CameraUpdateResult {
    bool changed = false;
    bool mouse_input_active = false;
    bool keyboard_input_active = false;
    bool scene_resources_changed = false;
  };

  CameraUpdateResult update_camera(SceneRepresentation& scene, float dt) {
    ETX_CRITICAL(_camera_controller);

    CameraUpdateResult result = {
      .changed = _camera_controller->update(dt),
      .mouse_input_active = _camera_controller->mouse_navigation_input_active(),
      .keyboard_input_active = _camera_controller->keyboard_navigation_input_active(),
    };
    if (result.changed) {
      result.scene_resources_changed = scene.store_active_camera();
      if (result.scene_resources_changed) {
        scene.update_medium_bounds();
      }
      if (_camera_modified_callback) {
        _camera_modified_callback();
      }
    }
    return result;
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

  virtual void on_scene_changed(SceneRepresentation& scene) {
    (void)scene;
    request_scene_update();
  }

  virtual void on_camera_changed(SceneRepresentation& scene) {
    (void)scene;
    request_camera_update();
  }

  virtual void on_scene_transforms_changed(SceneRepresentation& scene) {
    (void)scene;
    request_scene_transform_update();
  }

  void request_scene_update() {
    _scene_update_scope = SceneUpdateScope::Full;
  }

  void request_camera_update() {
    if (_scene_update_scope == SceneUpdateScope::None) {
      _scene_update_scope = SceneUpdateScope::Camera;
    }
  }

  void request_scene_transform_update() {
    if (static_cast<uint32_t>(_scene_update_scope) < static_cast<uint32_t>(SceneUpdateScope::Transforms)) {
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

  virtual void stop_rendering() {
    stop();
  }

  virtual void finish() {
  }

  virtual void restart() {
  }

  virtual void discard_render_output() {
  }

  virtual void cancel_preparation() {
  }

  CameraController* camera_controller() {
    return _camera_controller.get();
  }

  void set_camera_modified_callback(std::function<void()> callback) {
    _camera_modified_callback = std::move(callback);
  }

  void set_output_pixel_size(uint32_t pixel_size) {
    _output_pixel_size = clamp(pixel_size, 1u, 1024u);
  }

  uint32_t output_pixel_size() const {
    return _output_pixel_size;
  }

  void set_preview_pixel_size(uint32_t pixel_size) {
    const uint32_t previous_pixel_size = _preview_pixel_size;
    _preview_pixel_size = pixel_size == 0u ? 0u : clamp(pixel_size, 1u, 1024u);
    if ((previous_pixel_size == 0u) != (_preview_pixel_size == 0u)) {
      on_preview_mode_changed(_preview_pixel_size > 0u);
    }
  }

  uint32_t preview_pixel_size() const {
    return _preview_pixel_size;
  }

  uint32_t render_pixel_size() const {
    return _preview_pixel_size > 0u ? _preview_pixel_size : _output_pixel_size;
  }

  uint2 scaled_output_dimensions(const uint2& dimensions) const {
    return {
      (dimensions.x + _output_pixel_size - 1u) / _output_pixel_size,
      (dimensions.y + _output_pixel_size - 1u) / _output_pixel_size,
    };
  }

  uint2 scaled_render_dimensions(const uint2& dimensions) const {
    const uint32_t pixel_size = render_pixel_size();
    return {
      (dimensions.x + pixel_size - 1u) / pixel_size,
      (dimensions.y + pixel_size - 1u) / pixel_size,
    };
  }

 protected:
  virtual void on_preview_mode_changed(bool active) {
    (void)active;
  }

  TaskScheduler& scheduler;
  std::unique_ptr<CameraController> _camera_controller = nullptr;
  std::function<void()> _camera_modified_callback = {};
  uint2 _output_dimensions = {};
  RHITexture _output_texture = {};
  uint32_t _output_pixel_size = 1u;
  uint32_t _preview_pixel_size = 0u;
  SceneUpdateScope _scene_update_scope = SceneUpdateScope::Full;
};

}  // namespace etx
