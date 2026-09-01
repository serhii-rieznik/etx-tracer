#include "integrator.hxx"

#include <etx/render/host/tasks.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/rt/rt.hxx>
namespace etx {

struct IntegratorThreadImpl {
  SceneRepresentation& scene_representation;
  Raytracing& raytracing;
  SceneHashes current_scene_hashes = {};
  uint64_t current_camera_hash = 0;
  uint64_t scene_revision = 0u;
  Integrator* integrator = nullptr;
  Integrator::State latest_state = Integrator::State::Stopped;
  Integrator::Status latest_status = {};
  std::atomic<SceneUpdateScope> scene_update_scope = {SceneUpdateScope::Full};
  std::atomic_bool suppress_scene_commit_run = false;

  IntegratorThreadImpl(SceneRepresentation& scene_rep, Raytracing& rt)
    : scene_representation(scene_rep)
    , raytracing(rt) {
  }

  ~IntegratorThreadImpl() = default;

  void reset_scene_hashes() {
    current_scene_hashes = {};
    current_camera_hash = 0;
    scene_update_scope.store(SceneUpdateScope::Full);
  }

  void request_scene_check(SceneUpdateScope requested_scope) {
    SceneUpdateScope current_scope = scene_update_scope.load();
    while ((static_cast<uint32_t>(current_scope) < static_cast<uint32_t>(requested_scope)) && (scene_update_scope.compare_exchange_weak(current_scope, requested_scope) == false)) {
    }
  }

  void check_and_commit_scene_changes() {
    const SceneUpdateScope pending_scope = scene_update_scope.exchange(SceneUpdateScope::None);
    if (scene_representation.energy_compensation_interface_preparation_status().state == EnergyCompensationPreparationState::Preparing) {
      if (pending_scope != SceneUpdateScope::None) {
        request_scene_check(pending_scope);
      }
      return;
    }

    const bool camera_only_update = pending_scope == SceneUpdateScope::Camera;
    const bool transform_only_update = pending_scope == SceneUpdateScope::Transforms;
    const bool scoped_transform_update = camera_only_update || transform_only_update;
    SceneHashes new_hashes = current_scene_hashes;
    UpdateFlags changes = {};
    if (scoped_transform_update) {
      new_hashes.transforms_hash = scene_representation.data().compute_transforms_hash();
      if (transform_only_update) {
        new_hashes.instance_transforms_hash = scene_representation.data().compute_instance_transforms_hash();
        changes = new_hashes.compare(current_scene_hashes);
      }
    } else {
      new_hashes = scene_representation.data().compute_hashes();
      changes = new_hashes.compare(current_scene_hashes);
    }
    const bool full_update_requested = pending_scope == SceneUpdateScope::Full;
    const auto& preliminary_camera = scene_representation.camera();
    const uint64_t preliminary_camera_hash = xxh64(&preliminary_camera, sizeof(preliminary_camera));
    const bool update_requested = (pending_scope != SceneUpdateScope::None) || changes.any() || (current_camera_hash != preliminary_camera_hash);
    bool resume_after_commit = false;
    if (update_requested && (integrator != nullptr) && (latest_state != Integrator::State::Stopped)) {
      resume_after_commit = latest_state == Integrator::State::Running;
      integrator->stop(Integrator::Stop::Immediate);
      latest_state = integrator->state();
    }

    if (scoped_transform_update == false) {
      bool dependencies_updated = false;
      if (scene_representation.synchronize_render_dependencies(changes, full_update_requested, dependencies_updated) == false) {
        log::error("Failed to synchronize derived scene state before CPU render commit");
        request_scene_check(SceneUpdateScope::Full);
        return;
      }

      if (dependencies_updated) {
        new_hashes = scene_representation.data().compute_hashes();
        changes = new_hashes.compare(current_scene_hashes);
      }
      bool refresh_hashes = false;
      if (full_update_requested || changes[UpdateFlags::Images]) {
        scene_representation.data().images.load_images(raytracing.scheduler());
        refresh_hashes = true;
      }
      if (full_update_requested || changes[UpdateFlags::AnyMaterials]) {
        if (scene_representation.ensure_energy_compensation_interfaces() == false) {
          log::error("Failed to ensure BSDF energy-compensation interfaces before CPU render commit");
          request_scene_check(SceneUpdateScope::Full);
          return;
        }
        refresh_hashes = true;
      }

      if (refresh_hashes) {
        new_hashes = scene_representation.data().compute_hashes();
        changes = new_hashes.compare(current_scene_hashes);
      }
    }
    current_scene_hashes = new_hashes;

    const bool suppress_run = (pending_scope != SceneUpdateScope::None) && suppress_scene_commit_run.exchange(false);
    const auto& camera = scene_representation.camera();
    const uint64_t new_camera_hash = xxh64(&camera, sizeof(camera));

    if (changes.any() || (current_camera_hash != new_camera_hash)) {
      raytracing.commit(scene_representation.data(), scene_representation.camera(), changes);
      current_camera_hash = new_camera_hash;
      scene_revision += 1u;

      if ((integrator != nullptr) && resume_after_commit && (suppress_run == false)) {
        integrator->run();
        latest_state = integrator->state();
      }
    } else if (resume_after_commit && (suppress_run == false)) {
      integrator->run();
      latest_state = integrator->state();
    }
  }

  void refresh_integrator_state() {
    if (integrator == nullptr) {
      latest_state = Integrator::State::Stopped;
      latest_status = {};
      return;
    }
    latest_state = integrator->state();
    latest_status = integrator->status();
  }
};

IntegratorThread::IntegratorThread(SceneRepresentation& scene_rep, Raytracing& raytracing) {
  ETX_PIMPL_INIT(IntegratorThread, scene_rep, raytracing);
}

IntegratorThread ::~IntegratorThread() {
  terminate();
  ETX_PIMPL_CLEANUP(IntegratorThread);
}

void IntegratorThread::start(Integrator* i) {
  _private->integrator = i;
  _private->refresh_integrator_state();
}

void IntegratorThread::terminate() {
  stop(Integrator::Stop::Immediate);
}

Integrator* IntegratorThread::integrator() const {
  return _private->integrator;
}

void IntegratorThread::set_integrator(Integrator* i) {
  stop(Integrator::Stop::Immediate);
  _private->integrator = i;
  _private->refresh_integrator_state();
}

bool IntegratorThread::running() const {
  return (_private->integrator != nullptr) && (_private->latest_state != Integrator::State::Stopped);
}

const Integrator::Status& IntegratorThread::status() const {
  return _private->latest_status;
}

void IntegratorThread::run() {
  if (_private->integrator == nullptr) {
    return;
  }
  _private->integrator->run();
  _private->refresh_integrator_state();
}

void IntegratorThread::stop(Integrator::Stop st) {
  if (_private->integrator == nullptr) {
    return;
  }
  _private->integrator->stop(st);
  _private->refresh_integrator_state();
}

void IntegratorThread::restart() {
  stop(Integrator::Stop::Immediate);
  run();
}

void IntegratorThread::reset_scene_hashes() {
  _private->reset_scene_hashes();
}

void IntegratorThread::request_scene_check(SceneUpdateScope scope) {
  _private->request_scene_check(scope);
}

void IntegratorThread::suppress_next_scene_commit_run() {
  _private->suppress_scene_commit_run.store(true);
}

bool IntegratorThread::scene_changes_pending() const {
  return _private->scene_update_scope.load() != SceneUpdateScope::None;
}

uint64_t IntegratorThread::scene_revision() const {
  return _private->scene_revision;
}

bool IntegratorThread::update_integrator() {
  if (_private->integrator == nullptr) {
    return false;
  }

  const uint32_t completed_iterations = _private->latest_status.completed_iterations;
  _private->integrator->update();
  _private->refresh_integrator_state();
  return _private->latest_status.completed_iterations > completed_iterations;
}

void IntegratorThread::commit_scene_changes() {
  _private->check_and_commit_scene_changes();
  if (_private->integrator != nullptr) {
    _private->latest_state = _private->integrator->state();
    _private->latest_status = _private->integrator->status();
  }
}

void IntegratorThread::update() {
  ETX_PROFILER_SCOPE();

  commit_scene_changes();
  update_integrator();
}

}  // namespace etx
