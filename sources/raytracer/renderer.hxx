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

  virtual void update_camera(SceneRepresentation& scene, float dt) {
    ETX_CRITICAL(_camera_controller);

    bool camera_updated = _camera_controller->update(dt);
    if (camera_updated) {
      scene.store_active_camera();
      on_camera_changed(scene);
    } else if (camera_updated != last_camera_update_state) {
      on_camera_become_steady(scene);
    }
    last_camera_update_state = camera_updated;
  }

  virtual void render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& data) {
    (void)ctx;
    (void)scene;
    (void)data;
  }

  virtual RHITexture output_texture() const {
    return _output_texture;
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

  void request_scene_update() {
    _scene_update_requested = true;
  }

  bool consume_scene_update_request() {
    const bool result = _scene_update_requested;
    _scene_update_requested = false;
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
  TaskScheduler& scheduler;
  std::unique_ptr<CameraController> _camera_controller = nullptr;
  uint2 _output_dimensions = {};
  RHITexture _output_texture = {};
  bool last_camera_update_state = false;
  bool _scene_update_requested = true;
};

}  // namespace etx
