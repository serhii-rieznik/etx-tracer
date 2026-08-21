#include "integrator.hxx"

#include <etx/render/host/tasks.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/rt/rt.hxx>
namespace etx {

struct ITMessage {
  enum class Cls : uint32_t {
    Run,
    Stop,
  } cls;
  Integrator::Stop stop_option = Integrator::Stop::Immediate;
};

struct IntegratorThreadImpl {
  IntegratorThread* i = nullptr;
  std::atomic<bool> running = {};
  std::vector<ITMessage> messages;
  std::mutex lock;

  SceneRepresentation& scene_representation;
  Raytracing& raytracing;
  SceneHashes current_scene_hashes = {};
  uint64_t current_camera_hash = 0;
  Integrator* integrator = nullptr;
  Integrator::State latest_state = Integrator::State::Stopped;
  Integrator::Status latest_status = {};
  std::atomic<SceneUpdateScope> scene_update_scope = {SceneUpdateScope::Full};

  IntegratorThreadImpl(SceneRepresentation& scene_rep, Raytracing& rt)
    : scene_representation(scene_rep)
    , raytracing(rt)
    , running(true) {
    // External control mode - no background thread
  }

  ~IntegratorThreadImpl() {
    running = false;
  }

  void post_message(const ITMessage& msg) {
    std::unique_lock l(lock);
    messages.push_back(msg);
  }

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
    SceneHashes new_hashes = current_scene_hashes;
    UpdateFlags changes = {};

    if (pending_scope != SceneUpdateScope::None) {
      if (pending_scope == SceneUpdateScope::Full) {
        if (scene_representation.ensure_energy_compensation_interfaces() == false) {
          log::error("Failed to ensure BSDF energy-compensation interfaces before CPU render commit");
          request_scene_check(pending_scope);
          return;
        }
        scene_representation.data().images.load_images(raytracing.scheduler());
      }
      if (scene_representation.data().resolve_hierarchy() == false) {
        log::error("Failed to resolve scene hierarchy before CPU render commit");
        request_scene_check(pending_scope);
        return;
      }
      if (pending_scope == SceneUpdateScope::Full) {
        new_hashes = scene_representation.data().compute_hashes();
      } else {
        new_hashes.transforms_hash = scene_representation.data().compute_transforms_hash();
      }
      changes = new_hashes.compare(current_scene_hashes);
      current_scene_hashes = new_hashes;
    }

    const auto& camera = scene_representation.camera();
    const uint64_t new_camera_hash = xxh64(&camera, sizeof(camera));

    if (changes.any() || (current_camera_hash != new_camera_hash)) {
      if ((integrator != nullptr) && (latest_state == Integrator::State::Running)) {
        integrator->stop(Integrator::Stop::Immediate);
        latest_state = integrator->state();
      }

      raytracing.commit(scene_representation.data(), scene_representation.camera(), changes);
      current_camera_hash = new_camera_hash;

      if (integrator != nullptr) {
        integrator->run();
        latest_state = integrator->state();
      }
    }
  }

  void post_messages(const std::initializer_list<ITMessage>& msgs) {
    std::unique_lock l(lock);
    for (const auto& msg : msgs) {
      messages.push_back(msg);
    }
  }

  bool fetch_message(ITMessage& msg) {
    std::unique_lock l(lock);
    if (messages.empty())
      return false;

    msg = messages.front();
    messages.erase(messages.begin());
    return true;
  }

  bool has_messages() {
    std::unique_lock l(lock);
    return messages.empty() == false;
  }

  void process_messages() {
    ITMessage msg = {};
    while (fetch_message(msg)) {
      switch (msg.cls) {
        case ITMessage::Cls::Run: {
          if (integrator) {
            integrator->run();
          }
          break;
        }
        case ITMessage::Cls::Stop: {
          if (integrator) {
            integrator->stop(msg.stop_option);
          }
          break;
        }
        default:
          break;
      }
    }
  }
};

IntegratorThread::IntegratorThread(SceneRepresentation& scene_rep, Raytracing& raytracing) {
  ETX_PIMPL_INIT(IntegratorThread, scene_rep, raytracing);
}

IntegratorThread ::~IntegratorThread() {
  _private->running = false;
  ETX_PIMPL_CLEANUP(IntegratorThread);
}

void IntegratorThread::start(Integrator* i) {
  _private->integrator = i;
}

void IntegratorThread::terminate() {
}

Integrator* IntegratorThread::integrator() const {
  return _private->integrator;
}

void IntegratorThread::set_integrator(Integrator* i) {
  stop(Integrator::Stop::Immediate);
  _private->integrator = i;
}

bool IntegratorThread::running() const {
  return (_private->integrator != nullptr) && (_private->latest_state == Integrator::State::Running);
}

const Integrator::Status& IntegratorThread::status() const {
  return _private->latest_status;
}

void IntegratorThread::run() {
  _private->post_message({.cls = ITMessage::Cls::Run});
}

void IntegratorThread::stop(Integrator::Stop st) {
  _private->post_message({.cls = ITMessage::Cls::Stop, .stop_option = st});

  if (st == Integrator::Stop::Immediate) {
    do {
      update_integrator();
    } while (_private->latest_state != Integrator::State::Stopped);
  }
}

void IntegratorThread::restart() {
  _private->post_messages({
    {.cls = ITMessage::Cls::Stop, .stop_option = Integrator::Stop::Immediate},
    {.cls = ITMessage::Cls::Run},
  });
}

void IntegratorThread::reset_scene_hashes() {
  _private->reset_scene_hashes();
}

void IntegratorThread::request_scene_check(SceneUpdateScope scope) {
  _private->request_scene_check(scope);
}

bool IntegratorThread::scene_changes_pending() const {
  return _private->scene_update_scope.load() != SceneUpdateScope::None;
}

bool IntegratorThread::update_integrator() {
  if (_private->integrator == nullptr) {
    _private->process_messages();
    return false;
  }

  const uint32_t completed_iterations = _private->latest_status.completed_iterations;
  _private->process_messages();
  _private->integrator->update();
  _private->latest_state = _private->integrator->state();
  _private->latest_status = _private->integrator->status();
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
