#include <etx/core/core.hxx>
#include <etx/rt/integrators/vcm_cpu.hxx>

#include <etx/render/host/film.hxx>
#include <etx/render/shared/scene_camera.hxx>

#include <etx/rt/shared/vcm_shared.hxx>

#include <mutex>

namespace etx {

void VCMSpatialGrid::construct(const Scene& scene, const VCMLightVertex* samples, uint64_t sample_count, float radius) {
  data = {};
  if (sample_count == 0) {
    return;
  }

  _indices.resize(sample_count);

  TimeMeasure time_measure = {};

  data.radius_squared = radius * radius;
  data.cell_size = 2.0f * radius;
  data.bounding_box = {{kMaxFloat, kMaxFloat, kMaxFloat}, {-kMaxFloat, -kMaxFloat, -kMaxFloat}};
  for (uint64_t i = 0; i < sample_count; ++i) {
    const auto& p = samples[i];
    data.bounding_box.p_min = min(data.bounding_box.p_min, p.position(scene));
    data.bounding_box.p_max = max(data.bounding_box.p_max, p.position(scene));
  }

  uint32_t hash_table_size = static_cast<uint32_t>(next_power_of_two(sample_count));
  data.hash_table_mask = hash_table_size - 1u;

  _cell_ends.resize(hash_table_size);
  memset(_cell_ends.data(), 0, sizeof(uint32_t) * hash_table_size);

  for (uint64_t i = 0; i < sample_count; ++i) {
    _cell_ends[data.position_to_index(samples[i].position(scene))] += 1;
  }

  uint32_t sum = 0;
  for (auto& cell_end : _cell_ends) {
    uint32_t t = cell_end;
    cell_end = sum;
    sum += t;
  }

  for (uint32_t i = 0, e = static_cast<uint32_t>(sample_count); i < e; ++i) {
    uint32_t index = data.position_to_index(samples[i].position(scene));
    auto target_cell = _cell_ends[index]++;
    _indices[target_cell] = i;
  }

  data.indices = make_array_view<uint32_t>(_indices.data(), _indices.size());
  data.cell_ends = make_array_view<uint32_t>(_cell_ends.data(), _cell_ends.size());
}

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
  uint32_t opt_max_iterations = 0x7fffffff;
  uint32_t opt_radius_decay = 256;
  float opt_radius = 0.0f;

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

  VCMOptions _vcm_options = {};
  VCMIteration vcm_iteration = {};
  VCMSpatialGrid _current_grid = {};

  std::mutex _light_vertices_lock;
  std::vector<VCMLightPath> _light_paths;
  std::vector<VCMLightVertex> _light_vertices;

  CPUVCMImpl(Raytracing& r, std::atomic<Integrator::State>* st)
    : rt(r)
    , state(st) {
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

    opt_max_iterations = opt.get("spp", opt_max_iterations).to_integer();
    opt_radius = opt.get("vcm_r", opt_radius).to_float();
    opt_radius_decay = opt.get("vcm_r_decay", opt_radius_decay).to_integer();

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

    float used_radius = opt_radius;
    if (used_radius == 0.0f) {
      used_radius = 5.0f * rt.scene().bounding_sphere_radius * min(1.0f / float(camera_image.dimensions().x), 1.0f / float(camera_image.dimensions().y));
    }

    vcm_state = VCMState::GatheringLightVertices;

    float radius_scale = 1.0f / (1.0f + float(vcm_iteration.iteration) / float(opt_radius_decay));
    vcm_iteration.current_radius = used_radius * radius_scale;

    float eta_vcm = kPi * sqr(vcm_iteration.current_radius) * float(camera_image.count());
    vcm_iteration.vm_weight = _vcm_options.merge_vertices() ? eta_vcm : 0.0f;
    vcm_iteration.vc_weight = 1.0f / eta_vcm;
    vcm_iteration.vm_normalization = 1.0f / eta_vcm;

    _light_paths.clear();
    _light_vertices.clear();
    current_task = rt.scheduler().schedule(&gather_light_job, camera_image.count());
  }

  void continue_iteration() {
    ETX_ASSERT(vcm_state == VCMState::GatheringLightVertices);
    rt.scheduler().wait(current_task);

    stats.l_time = stats.light_gather_time.measure();

    TimeMeasure grid_time = {};
    if (_vcm_options.merge_vertices()) {
      _current_grid.construct(rt.scene(), _light_vertices.data(), _light_vertices.size(), vcm_iteration.current_radius);
    }
    stats.g_time = grid_time.measure();
    stats.camera_gather_time = {};

    vcm_state = VCMState::GatheringCameraVertices;
    current_task = rt.scheduler().schedule(&gather_camera_job, camera_image.count());
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
      while (running() && (state.path_length + 1 <= kVCMMaxDepth)) {
        Intersection intersection;
        bool found_intersection = rt.trace(state.ray, intersection, state.sampler);

        Medium::Sample medium_sample = vcm_try_sampling_medium(scene, state, found_intersection ? intersection.t : kMaxFloat);

        if (medium_sample.sampled_medium()) {
          vcm_handle_sampled_medium(scene, medium_sample, state);
          continue;
        } else if (found_intersection == false) {
          break;
        }

        state.path_distance += intersection.t;
        state.path_length += 1;
        const auto& tri = scene.triangles[intersection.triangle_index];
        const auto& mat = scene.materials[tri.material_index];

        if (vcm_handle_boundary_bsdf(scene, mat, intersection, PathSource::Light, state)) {
          continue;
        }

        vcm_update_light_vcm(intersection, state);

        if (bsdf::is_delta(mat, intersection.tex, rt.scene(), state.sampler) == false) {
          local_vertices.emplace_back(state, intersection.pos, intersection.barycentric, intersection.triangle_index, static_cast<uint32_t>(i));

          if (_vcm_options.connect_to_camera() && (state.path_length + 1 <= kVCMMaxDepth)) {
            float2 uv = {};
            auto value = vcm_connect_to_camera(rt, scene, intersection, mat, tri, vcm_iteration, state, uv);
            if (dot(value, value) > 0.0f) {
              iteration_light_image.atomic_add({value.x, value.y, value.z, 1.0f}, uv, thread_id);
            }
          }
        }

        if (vcm_next_ray(rt.scene(), PathSource::Light, intersection, kVCMRRStart, state, vcm_iteration) == false) {
          break;
        }
      }

      local_paths.emplace_back(path_begin, static_cast<uint32_t>(local_vertices.size() - path_begin), state.spect);
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
    ArrayView<VCMLightVertex> light_vertices = make_array_view<VCMLightVertex>(_light_vertices.data(), _light_vertices.size());
    const auto& scene = rt.scene();

    for (uint32_t pi = range_begin; running() && (pi < range_end); ++pi) {
      uint32_t x = pi % camera_image.dimensions().x;
      uint32_t y = pi / camera_image.dimensions().x;

      const auto& light_path = _light_paths[pi];

      stats.c++;
      VCMPathState state = vcm_generate_camera_state({x, y}, scene, vcm_iteration, light_path.spect);
      while (running() && (state.path_length <= kVCMMaxDepth)) {
        Intersection intersection;
        bool found_intersection = rt.trace(state.ray, intersection, state.sampler);

        Medium::Sample medium_sample = vcm_try_sampling_medium(scene, state, found_intersection ? intersection.t : kMaxFloat);

        if (medium_sample.sampled_medium()) {
          vcm_handle_sampled_medium(scene, medium_sample, state);
        } else if (found_intersection) {
          state.path_distance += intersection.t;

          const auto& tri = scene.triangles[intersection.triangle_index];
          const auto& mat = scene.materials[tri.material_index];

          if (vcm_handle_boundary_bsdf(scene, mat, intersection, PathSource::Camera, state)) {
            continue;
          }

          vcm_update_camera_vcm(intersection, state);
          vcm_handle_direct_hit(scene, tri, intersection, state);

          if (bsdf::is_delta(mat, intersection.tex, scene, state.sampler) == false) {
            vcm_connect_to_light(scene, tri, mat, intersection, vcm_iteration, rt, state);
            vcm_connect_to_light_path(scene, tri, mat, intersection, vcm_iteration, light_path, light_vertices, rt, state);
          }

          if (_vcm_options.merge_vertices() && (state.path_length + 1 <= kVCMMaxDepth)) {
            state.merged += _current_grid.data.gather(scene, state, _light_vertices.data(), intersection, kVCMMaxDepth, vcm_iteration.vc_weight);
          }

          if (vcm_next_ray(scene, PathSource::Camera, intersection, kVCMRRStart, state, vcm_iteration) == false) {
            break;
          }

          state.path_length += 1;

        } else {
          vcm_cam_handle_miss(scene, intersection, state);
          break;
        }
      }

      state.merged *= vcm_iteration.vm_normalization;
      state.merged += (state.gathered / spectrum::sample_pdf()).to_xyz();

      camera_image.accumulate({state.merged.x, state.merged.y, state.merged.z, 1.0f}, state.uv, float(vcm_iteration.iteration) / float(vcm_iteration.iteration + 1));
    }
  }
};

CPUVCM::CPUVCM(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUVCM, rt, &current_state);
}

CPUVCM::~CPUVCM() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  ETX_PIMPL_CLEANUP(CPUVCM);
}

Options CPUVCM::options() const {
  Options result = {};
  result.add(1u, _private->opt_max_iterations, 0xffffu, "spp", "Max Iterations");
  result.add(0.0f, _private->opt_radius, 10.0f, "vcm_r", "Initial Radius");
  result.add(1u, _private->opt_radius_decay, 65536u, "vcm_r_decay", "Radius Decay");
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
    rt.scheduler().wait(_private->current_task);
    current_state = Integrator::State::Stopped;
    _private->current_task = {};
  } else if (_private->vcm_iteration.iteration + 1 < _private->opt_max_iterations) {
    _private->vcm_iteration.iteration += 1;
    _private->start_next_iteration();
  } else {
    current_state = Integrator::State::Stopped;
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

void CPUVCM::update_options(const Options&) {
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
