#include <etx/core/core.hxx>

#include <etx/rt/integrators/vcm_cpu.hxx>
#include <etx/rt/integrators/vcm_spatial_grid.hxx>

#include <etx/render/host/film.hxx>
#include <etx/render/shared/scene_camera.hxx>

#include <etx/rt/shared/vcm_shared.hxx>

#include <mutex>

namespace etx {

struct CPUVCMImpl {
  struct GatherLightVerticesTask : public Task {
    CPUVCMImpl* impl = nullptr;
    GatherLightVerticesTask(CPUVCMImpl* i)
      : impl(i) {
    }
    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      impl->gather_light_vertices(begin, end, thread_id);
    }
  } gather_light_job = {this};

  struct GatherCameraVerticesTask : public Task {
    CPUVCMImpl* impl = nullptr;
    GatherCameraVerticesTask(CPUVCMImpl* i)
      : impl(i) {
    }
    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      impl->gather_camera_vertices(begin, end, thread_id);
    }
  } gather_camera_job = {this};

  Raytracing& rt;
  std::atomic<Integrator::State>* state = {};
  char status[2048] = {};
  Film camera_image;
  Film light_image;
  Film iteration_light_image;
  Task::Handle current_task = {};

  bool light_image_updated = false;
  bool camera_image_updated = false;

  struct {
    std::atomic<uint32_t> l;
    std::atomic<uint32_t> c;
    TimeMeasure light_gather_time = {};
    TimeMeasure camera_gather_time = {};
    TimeMeasure iteration_time = {};
    TimeMeasure total_time = {};
    double l_time = {};
    double c_time = {};
    double g_time = {};
    double m_time = {};
    double last_iteration_time = {};
  } stats;

  VCMState vcm_state = VCMState::Stopped;
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

  bool running() const {
    return state->load() != Integrator::State::Stopped;
  }

  void build_stats() {
    static const char* str_state[] = {
      "Stopped",
      "Preview",
      "Running",
      "WaitingForCompletion",
    };
    static const char* str_vcm_state[] = {
      "Stopped",
      "Gathering Light Vertices",
      "Gathering Camera Vertices",
    };

    double l_c = 100.0 * double(stats.l.load()) / double(camera_image.count());
    double c_c = 100.0 * double(stats.c.load()) / double(camera_image.count());

    if (vcm_iteration.iteration == 0) {
      snprintf(status, sizeof(status), "0 | %s / %s : L: %.2f, C: %.2f", str_state[uint32_t(state->load())], str_vcm_state[uint32_t(vcm_state)], l_c, c_c);
    } else {
      snprintf(status, sizeof(status), "%u | %s / %s : L: %.2f, C: %.2f, last iteration time: %.2fs (L: %.2fs, C: %.2fs, G: %.2fs, M: %.2f)", vcm_iteration.iteration,  //
        str_state[uint32_t(state->load())], str_vcm_state[uint32_t(vcm_state)], l_c, c_c, stats.last_iteration_time, stats.l_time, stats.c_time, stats.g_time, stats.m_time);
    }
  }

  void start(const Options& opt) {
    camera_image.clear();
    light_image.clear();
    iteration_light_image.clear();
    light_image_updated = true;
    camera_image_updated = true;
    vcm_options.load(opt);
    stats.total_time = {};
    vcm_iteration.iteration = 0;
    vcm_state = VCMState::Stopped;
    start_next_iteration();
  }

  void start_next_iteration() {
    ETX_ASSERT((vcm_state == VCMState::Stopped) || (vcm_state == VCMState::GatheringCameraVertices));
    rt.scheduler().wait(current_task);

    stats.c_time = stats.camera_gather_time.measure();
    stats.last_iteration_time = stats.iteration_time.measure();
    stats.iteration_time = {};
    stats.light_gather_time = {};
    stats.l = 0;
    stats.c = 0;

    float used_radius = vcm_options.initial_radius;
    if (used_radius == 0.0f) {
      used_radius = 5.0f * rt.scene().bounding_sphere_radius * min(1.0f / float(camera_image.dimensions().x), 1.0f / float(camera_image.dimensions().y));
    }

    vcm_state = VCMState::GatheringLightVertices;

    float radius_scale = 1.0f / (1.0f + float(vcm_iteration.iteration) / float(vcm_options.radius_decay));
    vcm_iteration.current_radius = used_radius * radius_scale;

    float eta_vcm = kPi * sqr(vcm_iteration.current_radius) * float(camera_image.count());
    vcm_iteration.vc_weight = 1.0f / eta_vcm;
    vcm_iteration.vm_weight = vcm_options.enable_merging() ? eta_vcm : 0.0f;
    vcm_iteration.vm_normalization = 1.0f / eta_vcm;

    _light_paths.clear();
    _light_vertices.clear();
    current_task = rt.scheduler().schedule(camera_image.count(), &gather_light_job);
  }

  void continue_iteration() {
    ETX_ASSERT(vcm_state == VCMState::GatheringLightVertices);
    rt.scheduler().wait(current_task);

    stats.l_time = stats.light_gather_time.measure();
    stats.g_time = 0.0f;

    if (vcm_options.merge_vertices()) {
      TimeMeasure grid_time = {};
      _current_grid.construct(rt.scene(), _light_vertices.data(), _light_vertices.size(), vcm_iteration.current_radius, rt.scheduler());
      stats.g_time = grid_time.measure();
    }

    stats.camera_gather_time = {};

    vcm_state = VCMState::GatheringCameraVertices;
    current_task = rt.scheduler().schedule(camera_image.count(), &gather_camera_job);
  }

  void gather_light_vertices(uint32_t range_begin, uint32_t range_end, uint32_t thread_id) {
    const Scene& scene = rt.scene();

    std::vector<VCMLightVertex> local_vertices;
    local_vertices.reserve(4llu * (range_end - range_begin));

    std::vector<VCMLightPath> local_paths;
    local_paths.reserve(range_end - range_begin);

    for (uint64_t i = range_begin; running() && (i < range_end); ++i) {
      stats.l++;

      VCMPathState state = vcm_generate_emitter_state(static_cast<uint32_t>(i), scene, vcm_iteration);

      uint32_t path_begin = static_cast<uint32_t>(local_vertices.size());
      while (running()) {
        auto step_result = vcm_light_step(scene, vcm_iteration, vcm_options, static_cast<uint32_t>(i), state, rt);

        if (step_result.add_vertex) {
          local_vertices.emplace_back(step_result.vertex_to_add);
        }

        for (uint32_t i = 0; i < step_result.splat_count; ++i) {
          const float3& val = step_result.values_to_splat[i];
          iteration_light_image.atomic_add({val.x, val.y, val.z, 1.0f}, step_result.splat_uvs[i], thread_id);
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

    for (uint32_t pi = range_begin; running() && (pi < range_end); ++pi) {
      uint32_t x = pi % camera_image.dimensions().x;
      uint32_t y = pi / camera_image.dimensions().x;

      const auto& light_path = _light_paths[pi];

      stats.c++;
      VCMPathState state = vcm_generate_camera_state({x, y}, scene, vcm_iteration, light_path.spect);
      while (running() && vcm_camera_step(scene, vcm_iteration, vcm_options, light_paths, light_vertices, state, rt, _current_grid.data)) {
      }

      state.merged *= vcm_iteration.vm_normalization;
      state.merged += (state.gathered / state.spect.sampling_pdf()).to_xyz();

      float t = float(vcm_iteration.iteration) / float(vcm_iteration.iteration + 1);
      camera_image.accumulate({state.merged.x, state.merged.y, state.merged.z, 1.0f}, state.uv, t);
    }
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

void CPUVCM::set_output_size(const uint2& dim) {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  _private->camera_image.resize(dim, 1);
  _private->light_image.resize(dim, 1);
  _private->iteration_light_image.resize(dim, rt.scheduler().max_thread_count());
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
  _private->build_stats();
  _private->camera_image_updated = _private->vcm_state == VCMState::GatheringCameraVertices;

  if ((current_state == State::Stopped) || (rt.scheduler().completed(_private->current_task) == false)) {
    return;
  }

  if (_private->vcm_state == VCMState::GatheringLightVertices) {
    TimeMeasure tm = {};
    _private->light_image_updated = true;
    _private->iteration_light_image.flush_to(_private->light_image, float(_private->vcm_iteration.iteration) / float(_private->vcm_iteration.iteration + 1));
    _private->stats.m_time = tm.measure();
    _private->continue_iteration();
  } else if (current_state == State::WaitingForCompletion) {
    current_state = Integrator::State::Stopped;
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};
  } else if (_private->vcm_iteration.iteration + 1 < rt.scene().samples) {
    _private->vcm_iteration.iteration += 1;
    _private->start_next_iteration();
  } else {
    current_state = Integrator::State::Stopped;
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};
  }
}

void CPUVCM::stop(Stop st) {
  if (current_state == State::Stopped) {
    return;
  }

  if (st == Stop::Immediate) {
    current_state = State::Stopped;
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};
  } else {
    current_state = State::WaitingForCompletion;
    snprintf(_private->status, sizeof(_private->status), "[%u] Waiting for completion", _private->vcm_iteration.iteration);
  }
}

void CPUVCM::update_options(const Options& opt) {
  if (current_state == State::Preview) {
    preview(opt);
  }
}

bool CPUVCM::have_updated_camera_image() const {
  return _private->camera_image_updated;
}

bool CPUVCM::have_updated_light_image() const {
  return _private->light_image_updated;
}

const float4* CPUVCM::get_camera_image(bool) {
  _private->camera_image_updated = false;
  return _private->camera_image.data();
}

const float4* CPUVCM::get_light_image(bool) {
  _private->light_image_updated = false;
  return _private->light_image.data();
}

const char* CPUVCM::status() const {
  return _private->status;
}

}  // namespace etx
