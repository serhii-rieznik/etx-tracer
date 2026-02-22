#pragma once
#include <etx/core/core.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/rhi/rhi_types.hxx>

#include "render_context.hxx"
#include <etx/engine/camera_controller.hxx>

#include <memory>

struct sapp_event;

namespace etx {

struct TaskScheduler;

enum class RendererMode {
  CPURaytracing,
  Rasterization,
  GPURaytracing,
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
  }

  virtual void update_camera(SceneRepresentation& scene, float dt) {
    ETX_CRITICAL(_camera_controller);

    bool camera_updated = _camera_controller->update(dt);
    if (camera_updated) {
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

  virtual bool is_running() const {
    return false;
  }

  virtual void start() {
  }

  virtual void stop() {
  }

  virtual void restart() {
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
