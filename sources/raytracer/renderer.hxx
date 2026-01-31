#pragma once
#include <etx/core/core.hxx>
#include <etx/render/host/scene_representation.hxx>
#include "render.hxx"
#include "camera_controller.hxx"
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
  Renderer(TaskScheduler& s)
    : scheduler(s) {
  }

  virtual ~Renderer() = default;

  virtual void init(RenderContext& render_context, SceneRepresentation& scene) {
    if (_camera_controller == nullptr) {
      _camera_controller.reset(new CameraController(scene.mutable_camera()));
    }
  }
  virtual void frame(RenderContext& render_context, SceneRepresentation& scene, float dt) {
    if (_camera_controller && _camera_controller->update(dt)) {
      on_camera_changed(scene, false);
    }
  }
  virtual void cleanup(RenderContext& render_context) = 0;
  virtual void process_event(const sapp_event* e) {
    if (_camera_controller) {
      _camera_controller->handle_event(e);
    }
  }

  virtual void on_camera_changed(SceneRepresentation& scene, bool path_changed) = 0;
  virtual void on_scene_changed(SceneRepresentation& scene) = 0;

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
};

}  // namespace etx
