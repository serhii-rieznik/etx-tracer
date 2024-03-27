#include <etx/core/core.hxx>

#include <etx/rt/integrators/vcm_cpu.hxx>
#include <etx/rt/integrators/vcm_spatial_grid.hxx>

#include <etx/render/host/film.hxx>
#include <etx/render/shared/scene_camera.hxx>

#include <etx/rt/shared/vcm_shared.hxx>

#include <mutex>

namespace etx {

struct CPUVCMImpl : public Task {
  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    if (mode == Mode::LightPath) {
      gather_light_vertices(begin, end, thread_id);
    } else {
      gather_camera_vertices(begin, end, thread_id);
    }
  }

  void completed() override {
    if (mode == Mode::LightPath) {
      complete_light_vertices();
    } else {
      complete_camera_vertices();
    }
  }

  enum Mode : uint32_t {
    LightPath,
    CameraPath,
  } mode = Mode::LightPath;

  Raytracing& rt;
  std::atomic<Integrator::State>* state = {};
  Integrator::Status status = {};
  TimeMeasure iteration_time = {};
  Task::Handle task_handle = {};
  std::atomic<bool> have_light_image = {};
  std::atomic<bool> have_camera_image = {};

  VCMOptions vcm_options = {};
  VCMIteration vcm_iteration = {};
  VCMSpatialGrid _current_grid = {};

  std::mutex _light_vertices_lock;
  std::vector<VCMLightPath> _light_paths;
  std::vector<VCMLightVertex> _light_vertices;

  CPUVCMImpl(Raytracing& r, std::atomic<Integrator::State>* st)
    : rt(r)
    , state(st)
    , vcm_options(VCMOptions::default_values()) {
  }

  ~CPUVCMImpl() {
    rt.scheduler().wait(task_handle);
    task_handle = {};
  }

  bool running() const {
    return state->load() != Integrator::State::Stopped;
  }

  void start(const Options& opt) {
    rt.scheduler().wait(task_handle);
    task_handle = {};

    status = {};

    rt.film().clear();
    have_camera_image = true;
    have_light_image = true;

    vcm_options.load(opt);
    vcm_iteration.iteration = 0;
    start_next_iteration();
  }

  void start_next_iteration() {
    iteration_time = {};

    float used_radius = vcm_options.initial_radius;
    if (used_radius == 0.0f) {
      used_radius = 5.0f * rt.scene().bounding_sphere_radius * min(1.0f / float(rt.film().dimensions().x), 1.0f / float(rt.film().dimensions().y));
    }

    float radius_scale = 1.0f / (1.0f + float(vcm_iteration.iteration) / float(vcm_options.radius_decay));
    vcm_iteration.current_radius = used_radius * radius_scale;

    float eta_vcm = kPi * sqr(vcm_iteration.current_radius) * float(rt.film().count());
    vcm_iteration.vc_weight = 1.0f / eta_vcm;
    vcm_iteration.vm_weight = vcm_options.enable_merging() ? eta_vcm : 0.0f;
    vcm_iteration.vm_normalization = 1.0f / eta_vcm;

    status.current_iteration = vcm_iteration.iteration;

    _light_paths.clear();
    _light_vertices.clear();
    rt.film().clear({Film::LightIteration});

    mode = Mode::LightPath;

    if (task_handle.data == Task::InvalidHandle) {
      task_handle = rt.scheduler().schedule(rt.film().count(), this);
    } else {
      rt.scheduler().restart(task_handle, rt.film().count());
    }
  }

  void gather_light_vertices(uint32_t range_begin, uint32_t range_end, uint32_t thread_id) {
    const Scene& scene = rt.scene();
    auto& film = rt.film();

    std::vector<VCMLightVertex> local_vertices;
    local_vertices.reserve(4llu * (range_end - range_begin));

    std::vector<VCMLightPath> local_paths;
    local_paths.reserve(range_end - range_begin);

    for (uint64_t i = range_begin; running() && (i < range_end); ++i) {
      VCMPathState state = vcm_generate_emitter_state(static_cast<uint32_t>(i), scene, vcm_iteration);

      uint32_t path_begin = static_cast<uint32_t>(local_vertices.size());
      while (running()) {
        auto step_result = vcm_light_step(scene, vcm_iteration, vcm_options, static_cast<uint32_t>(i), state, rt);

        if (step_result.add_vertex) {
          local_vertices.emplace_back(step_result.vertex_to_add);
        }

        for (uint32_t i = 0; i < step_result.splat_count; ++i) {
          const float3& val = step_result.values_to_splat[i];
          if (dot(val, val) > kEpsilon) {
            film.atomic_add(Film::LightImage, {val.x, val.y, val.z, 1.0f}, step_result.splat_uvs[i]);
          }
        }

        if (step_result.continue_tracing == false) {
          break;
        }
      }

      auto& lp = local_paths.emplace_back();
      lp.index = path_begin;
      lp.count = static_cast<uint32_t>(local_vertices.size() - path_begin);
      lp.spect = state.spect;
    }

    {
      std::scoped_lock lock(_light_vertices_lock);
      for (auto& path : local_paths) {
        path.index += static_cast<uint32_t>(_light_vertices.size());
      }
      _light_paths.insert(_light_paths.end(), local_paths.begin(), local_paths.end());
      _light_vertices.insert(_light_vertices.end(), local_vertices.begin(), local_vertices.end());
    }
  }

  void gather_camera_vertices(uint32_t range_begin, uint32_t range_end, uint32_t thread_id) {
    auto light_vertices = make_array_view<VCMLightVertex>(_light_vertices.data(), _light_vertices.size());
    auto light_paths = make_array_view<VCMLightPath>(_light_paths.data(), _light_paths.size());
    const auto& scene = rt.scene();
    auto& film = rt.film();

    for (uint32_t pi = range_begin; running() && (pi < range_end); ++pi) {
      uint32_t x = pi % film.dimensions().x;
      uint32_t y = pi / film.dimensions().x;

      const auto& light_path = _light_paths[pi];

      VCMPathState state = vcm_generate_camera_state({x, y}, scene, vcm_iteration, light_path.spect);
      while (running()) {
        bool continue_tracing = vcm_camera_step(scene, vcm_iteration, vcm_options, light_paths, light_vertices, state, rt, _current_grid.data);

        if (continue_tracing == false) {
          break;
        }
      }

      state.merged *= vcm_iteration.vm_normalization;
      state.merged += (state.gathered / state.spect.sampling_pdf()).to_xyz();

      float t = float(vcm_iteration.iteration) / float(vcm_iteration.iteration + 1);
      film.accumulate(Film::Camera, {state.merged.x, state.merged.y, state.merged.z, 1.0f}, state.uv, t);

      if (pi % 256 == 0) {
        have_camera_image = true;
      }
    }

    have_camera_image = true;
  }

  void complete_light_vertices() {
    rt.film().commit_light_iteration(vcm_iteration.iteration);
    have_light_image = true;

    if (*state == Integrator::State::Stopped) {
      return;
    }

    if (vcm_options.merge_vertices()) {
      _current_grid.construct(rt.scene(), _light_vertices.data(), _light_vertices.size(), vcm_iteration.current_radius, rt.scheduler());
    }

    mode = Mode::CameraPath;
    rt.scheduler().restart(task_handle, rt.film().count());
  }

  void complete_camera_vertices() {
    status.completed_iterations += 1u;
    status.last_iteration_time = iteration_time.measure();
    status.total_time += status.last_iteration_time;

    have_camera_image = true;

    if ((*state == Integrator::State::WaitingForCompletion) || (*state == Integrator::State::Stopped) || (vcm_iteration.iteration + 1 >= rt.scene().samples)) {
      *state = Integrator::State::Stopped;
      return;
    }

    vcm_iteration.iteration += 1;
    start_next_iteration();
  }
};

CPUVCM::CPUVCM(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUVCM, rt, &current_state);
}

CPUVCM::~CPUVCM() {
  stop(Stop::Immediate);
  ETX_PIMPL_CLEANUP(CPUVCM);
}

Options CPUVCM::options() const {
  Options result = {};
  _private->vcm_options.store(result);
  return result;
}

void CPUVCM::preview(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Preview;
    _private->start(opt);
  }
}

void CPUVCM::run(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->start(opt);
  }
}

void CPUVCM::update() {
}

void CPUVCM::stop(Stop st) {
  if (current_state == State::Stopped) {
    return;
  }

  current_state = (st == Stop::Immediate) ? State::Stopped : State::WaitingForCompletion;

  if (current_state == State::Stopped) {
    rt.scheduler().wait(_private->task_handle);
    _private->task_handle = {};
  }
}

void CPUVCM::update_options(const Options& opt) {
  if (current_state == State::Preview) {
    preview(opt);
  }
}

bool CPUVCM::have_updated_camera_image() const {
  bool result = _private->have_camera_image;
  _private->have_camera_image = false;
  return result;
}

bool CPUVCM::have_updated_light_image() const {
  bool result = _private->have_light_image;
  _private->have_light_image = false;
  return result;
}

Integrator::Status CPUVCM::status() const {
  return _private->status;
}

}  // namespace etx
