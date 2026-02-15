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
  std::atomic<bool> scene_check_requested = {true};

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
    scene_check_requested.store(true);
  }

  void request_scene_check() {
    scene_check_requested.store(true);
  }

  void check_and_commit_scene_changes() {
    const bool has_pending_scene_check = scene_check_requested.exchange(false);
    SceneHashes new_hashes = current_scene_hashes;
    UpdateFlags changes = {};

    if (has_pending_scene_check) {
      scene_representation.data().images.load_images(raytracing.scheduler());
      new_hashes = scene_representation.data().compute_hashes();
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
    while (_private->latest_state != Integrator::State::Stopped) {
      update();  // External control mode - call update directly
    }
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

void IntegratorThread::request_scene_check() {
  _private->request_scene_check();
}

void IntegratorThread::update() {
  ETX_PROFILER_SCOPE();

  if (_private->integrator == nullptr) {
    _private->process_messages();
    return;
  }

  _private->check_and_commit_scene_changes();
  _private->process_messages();

  _private->integrator->update();
  _private->latest_state = _private->integrator->state();
  _private->latest_status = _private->integrator->status();
}

}  // namespace etx
