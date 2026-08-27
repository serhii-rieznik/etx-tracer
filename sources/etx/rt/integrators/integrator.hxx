#pragma once

#include <etx/core/profiler.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/engine/options.hxx>
#include <etx/rt/rt.hxx>
#include <etx/render/shared/math.hxx>
namespace etx {

enum class SceneUpdateScope : uint32_t;

struct Integrator {
  struct PathProgress {
    enum class Phase : uint32_t {
      None,
      Light,
      Camera,
    } phase = Phase::None;

    uint64_t completed_path_count = 0u;
    uint64_t total_path_count = 0u;
  };

  enum class Type : uint32_t {
    Debug = 0,
    PathTracing = 1,
    Bidirectional = 2,
    VCM = 3,
    UPBP = 4,

    Count,
    Invalid = kInvalidIndex,
  };

  enum class State : uint32_t {
    Stopped,
    Running,
    WaitingForCompletion,
  };

  enum class Stop : uint32_t {
    Immediate,
    WaitForCompletion,
  };

  struct Status {
    struct DebugInfo {
      const char* title = "";
      float value = 0.0f;
    };

    double last_iteration_time = 0.0;
    double total_time = 0.0;
    uint32_t completed_iterations = 0;
    uint32_t current_iteration = 0;

    DebugInfo* debug_info = nullptr;
    uint32_t debug_info_count = 0;
  };

  Integrator(Raytracing& r)
    : rt(r) {
    integrator_options.set_string("desc", "No options available", "general-options");
  }

  virtual ~Integrator() = default;

  virtual const char* name() {
    return "Basic Integrator";
  }

  virtual bool enabled() const {
    return true;
  }

  virtual const char* status_str() const {
    return "Basic Integrator (not able to render anything)";
  }

  virtual PathProgress path_progress() const {
    return {};
  }

  virtual void run() {
  }

  virtual void update() {
  }

  virtual void stop(Stop) {
  }

  virtual void update_options() {
  }

  virtual void sync_from_options(const Options& options) {
  }

  virtual const Status& status() const = 0;

  virtual Type type() const {
    return Type::Invalid;
  }

  virtual uint32_t supported_strategies() const {
    return Scene::Strategy::Default;
  }

 public:
  Options& options() {
    return integrator_options;
  }

  bool can_run() const {
    return rt.scene().committed();
  }

  State state() const {
    return current_state.load();
  }

 protected:
  Raytracing& rt;
  Options integrator_options = {};
  std::atomic<State> current_state = {State::Stopped};
  uint32_t pad = 0;
};

struct TaskScheduler;
struct SceneRepresentation;
struct IntegratorThreadImpl;
struct IntegratorThread {
  IntegratorThread(SceneRepresentation& scene_rep, Raytracing& raytracing);
  ~IntegratorThread();

  void start(Integrator*);
  void terminate();

  void update();
  bool update_integrator();
  void commit_scene_changes();

  Integrator* integrator() const;
  void set_integrator(Integrator*);

  bool running() const;
  const Integrator::Status& status() const;

  void run();
  void stop(Integrator::Stop);
  void restart();

  void reset_scene_hashes();
  void request_scene_check(SceneUpdateScope scope);
  void suppress_next_scene_commit_run();
  bool scene_changes_pending() const;

 private:
  ETX_DECLARE_PIMPL(IntegratorThread, 1024);
};

inline const char* integrator_type_to_id(Integrator::Type type) {
  switch (type) {
    case Integrator::Type::Debug:
      return "debug";
    case Integrator::Type::PathTracing:
      return "pt";
    case Integrator::Type::Bidirectional:
      return "bdpt";
    case Integrator::Type::VCM:
      return "vcm";
    case Integrator::Type::UPBP:
      return "upbp";
    default:
      return nullptr;
  }
}

inline Integrator::Type integrator_id_to_type(const char* id) {
  if (id == nullptr)
    return Integrator::Type::PathTracing;
  if (strcmp(id, "debug") == 0)
    return Integrator::Type::Debug;
  if (strcmp(id, "pt") == 0)
    return Integrator::Type::PathTracing;
  if (strcmp(id, "bdpt") == 0)
    return Integrator::Type::Bidirectional;
  if (strcmp(id, "vcm") == 0)
    return Integrator::Type::VCM;
  if (strcmp(id, "upbp") == 0)
    return Integrator::Type::UPBP;
  return Integrator::Type::Invalid;
}

inline Integrator::Type integrator_to_type(Integrator* integrator) {
  return (integrator != nullptr) ? integrator->type() : Integrator::Type::Invalid;
}

inline Integrator* integrator_type_to_instance(Integrator::Type type, Integrator* array[], size_t count) {
  if (type == Integrator::Type::Invalid || type >= Integrator::Type::Count)
    return nullptr;

  uint32_t index = static_cast<uint32_t>(type);
  if (index < count)
    return array[index];

  return nullptr;
}

}  // namespace etx
