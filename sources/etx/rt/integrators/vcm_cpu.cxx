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
  struct LightPath {
    uint64_t begin = 0;
    uint64_t length = 0;
    SpectralQuery spect = {};
  };

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
  uint32_t iteration = {};
  uint32_t opt_max_iterations = 0x7fffffff;
  uint32_t opt_max_depth = 0x7fffffff;
  uint32_t opt_rr_start = 0x5;
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
  VCMIteration _it = {};
  VCMSpatialGrid _current_grid = {};

  std::mutex _light_vertices_lock;
  std::vector<LightPath> _light_paths;
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

    if (iteration == 0) {
      snprintf(status, sizeof(status), "0 | %s / %s : L: %.2f, C: %.2f", str_state[uint32_t(state->load())], str_vcm_state[uint32_t(vcm_state)], l_c, c_c);
    } else {
      snprintf(status, sizeof(status), "%u | %s / %s : L: %.2f, C: %.2f, last iteration time: %.2fs (L: %.2fs, C: %.2fs, G: %.2fs, M: %.2f)", iteration,  //
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
    opt_max_depth = opt.get("pathlen", opt_max_depth).to_integer();
    opt_rr_start = opt.get("rrstart", opt_rr_start).to_integer();
    opt_radius = opt.get("vcm_r", opt_radius).to_float();
    opt_radius_decay = opt.get("vcm_r_decay", opt_radius_decay).to_integer();

    stats.total_time = {};
    iteration = 0;
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

    float radius_scale = 1.0f / (1.0f + float(iteration) / float(opt_radius_decay));
    _it.current_radius = used_radius * radius_scale;

    float eta_vcm = kPi * sqr(_it.current_radius) * float(camera_image.count());
    _it.vm_weight = _vcm_options.merge_vertices() ? eta_vcm : 0.0f;
    _it.vc_weight = 1.0f / eta_vcm;
    _it.vm_normalization = 1.0f / eta_vcm;

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
      _current_grid.construct(rt.scene(), _light_vertices.data(), _light_vertices.size(), _it.current_radius);
    }
    stats.g_time = grid_time.measure();
    stats.camera_gather_time = {};

    vcm_state = VCMState::GatheringCameraVertices;
    current_task = rt.scheduler().schedule(&gather_camera_job, camera_image.count());
  }

  void gather_light_vertices(uint32_t range_begin, uint32_t range_end, uint32_t thread_id) {
    RNDSampler smp;

    std::vector<VCMLightVertex> local_vertices;
    local_vertices.reserve(4llu * (range_end - range_begin));

    std::vector<LightPath> local_paths;
    local_paths.reserve(range_end - range_begin);

    for (uint64_t i = range_begin; running() && (i < range_end); ++i) {
      stats.l++;
      auto spect = spectrum::sample(smp.next());

      auto emitter_sample = sample_emission(rt.scene(), spect, smp);
      float cos_t = dot(emitter_sample.direction, emitter_sample.normal);

      VCMPathState state;
      state.throughput = emitter_sample.value * (cos_t / (emitter_sample.pdf_dir * emitter_sample.pdf_area * emitter_sample.pdf_sample));
      state.ray = {emitter_sample.origin, emitter_sample.direction};
      state.d_vcm = emitter_sample.pdf_area / emitter_sample.pdf_dir_out;
      state.d_vc = emitter_sample.is_delta ? 0.0f : (cos_t / (emitter_sample.pdf_dir_out * emitter_sample.pdf_sample));
      state.d_vm = state.d_vc * _it.vc_weight;
      ETX_VALIDATE(state.d_vcm);
      ETX_VALIDATE(state.d_vc);
      ETX_VALIDATE(state.d_vm);

      uint32_t medium_index = emitter_sample.medium_index;
      float state_eta = 1.0f;

      if (emitter_sample.triangle_index != kInvalidIndex) {
        state.ray.o = shading_pos(rt.scene().vertices, rt.scene().triangles[emitter_sample.triangle_index], emitter_sample.barycentric, state.ray.d);
      }

      uint64_t path_begin = local_vertices.size();
      float path_distance = 0.0f;
      uint32_t path_length = 0;
      while (running() && (path_length + 1 <= opt_max_depth)) {
        Intersection intersection;
        bool found_intersection = rt.trace(state.ray, intersection, smp);

        Medium::Sample medium_sample = {};
        if (medium_index != kInvalidIndex) {
          medium_sample = rt.scene().mediums[medium_index].sample(spect, smp, state.ray.o, state.ray.d, found_intersection ? intersection.t : kMaxFloat);
          state.throughput *= medium_sample.weight;
          ETX_VALIDATE(state.throughput);
        }

        if (medium_sample.sampled_medium()) {
          path_distance += medium_sample.t;
          path_length += 1;
          const auto& medium = rt.scene().mediums[medium_index];
          state.ray.o = medium_sample.pos;
          state.ray.d = medium.sample_phase_function(spect, smp, medium_sample.pos, state.ray.d);
        } else if (found_intersection) {
          path_distance += intersection.t;
          path_length += 1;
          const auto& tri = rt.scene().triangles[intersection.triangle_index];
          const auto& mat = rt.scene().materials[tri.material_index];

          if (mat.cls == Material::Class::Boundary) {
            auto bsdf_sample = bsdf::sample({spect, medium_index, PathSource::Light, intersection, intersection.w_i, {}}, mat, rt.scene(), smp);
            if (bsdf_sample.properties & BSDFSample::MediumChanged) {
              medium_index = bsdf_sample.medium_index;
            }
            state.ray.o = intersection.pos;
            state.ray.d = bsdf_sample.w_o;
            continue;
          }

          {
            if ((path_length > 1) || (emitter_sample.is_delta == false)) {
              state.d_vcm *= sqr(path_distance);
            }

            float cos_to_prev = fabsf(dot(intersection.nrm, -state.ray.d));
            state.d_vcm /= cos_to_prev;
            state.d_vc /= cos_to_prev;
            state.d_vm /= cos_to_prev;
            path_distance = 0.0f;
          }

          if (bsdf::is_delta(mat, intersection.tex, rt.scene(), smp) == false) {
            local_vertices.emplace_back(state, intersection.pos, intersection.barycentric, intersection.triangle_index, path_length, static_cast<uint32_t>(i));

            if (_vcm_options.connect_to_camera() && (path_length + 1 <= opt_max_depth)) {
              auto camera_sample = sample_film(smp, rt.scene(), intersection.pos);
              if (camera_sample.pdf_dir > 0.0f) {
                auto direction = camera_sample.position - intersection.pos;
                auto w_o = normalize(direction);
                auto data = BSDFData{spect, medium_index, PathSource::Light, intersection, state.ray.d, w_o};
                auto eval = bsdf::evaluate(data, mat, rt.scene(), smp);
                if (eval.valid()) {
                  float3 p0 = shading_pos(rt.scene().vertices, tri, intersection.barycentric, w_o);
                  auto tr = transmittance(spect, smp, p0, camera_sample.position, medium_index, rt.scene(), rt);
                  if (tr.is_zero() == false) {
                    float reverse_pdf = bsdf::pdf(data.swap_directions(), mat, rt.scene(), smp);
                    float camera_pdf = camera_sample.pdf_dir_out * fabsf(dot(intersection.nrm, w_o)) / dot(direction, direction);
                    ETX_VALIDATE(camera_pdf);

                    float w_light = camera_pdf * (_it.vm_weight + state.d_vcm + state.d_vc * reverse_pdf);
                    ETX_VALIDATE(w_light);

                    float weight = _vcm_options.enable_mis() ? (1.0f / (1.0f + w_light)) : 1.0f;
                    ETX_VALIDATE(weight);

                    eval.bsdf *= fix_shading_normal(tri.geo_n, data.nrm, data.w_i, data.w_o);
                    auto result = (tr * eval.bsdf * state.throughput * camera_sample.weight) * weight;

                    auto value = (result / spectrum::sample_pdf()).to_xyz();
                    iteration_light_image.atomic_add({value.x, value.y, value.z, 1.0f}, camera_sample.uv, thread_id);
                  }
                }
              }
            }
          }

          if (vcm_next_ray(rt.scene(), spect, PathSource::Light, intersection, path_length, opt_rr_start, smp, state, medium_index, state_eta, _it) == false) {
            break;
          }

        } else {
          break;
        }
      }

      local_paths.emplace_back(path_begin, local_vertices.size() - path_begin, spect);
    }

    {
      std::scoped_lock lock(_light_vertices_lock);
      for (auto& path : local_paths) {
        path.begin += _light_vertices.size();
      }
      _light_paths.insert(_light_paths.end(), local_paths.begin(), local_paths.end());
      _light_vertices.insert(_light_vertices.end(), local_vertices.begin(), local_vertices.end());
    }
  }

  void gather_camera_vertices(uint32_t range_begin, uint32_t range_end, uint32_t thread_id) {
    RNDSampler smp;

    for (uint32_t pi = range_begin; running() && (pi < range_end); ++pi) {
      stats.c++;
      uint32_t x = pi % camera_image.dimensions().x;
      uint32_t y = pi / camera_image.dimensions().x;

      const auto& light_path = _light_paths[pi];
      auto spect = light_path.spect;

      VCMPathState state;
      float2 uv = get_jittered_uv(smp, {x, y}, camera_image.dimensions());
      state.ray = generate_ray(smp, rt.scene(), uv);

      auto film_eval = film_evaluate_out(spect, rt.scene().camera, state.ray);

      state.throughput = {spect.wavelength, 1.0f};
      state.d_vcm = 1.0f / film_eval.pdf_dir;
      state.d_vc = 0.0f;
      state.d_vm = 0.0f;

      uint32_t medium_index = rt.scene().camera_medium_index;
      float state_eta = 1.0f;
      float d = 0.0f;
      SpectralResponse gathered = {spect.wavelength, 0.0f};
      float3 merged = {};

      float path_distance = 0.0f;
      uint32_t path_length = 1;
      while (running() && (path_length <= opt_max_depth)) {
        Intersection intersection;
        bool found_intersection = rt.trace(state.ray, intersection, smp);

        Medium::Sample medium_sample = {};
        if (medium_index != kInvalidIndex) {
          medium_sample = rt.scene().mediums[medium_index].sample(spect, smp, state.ray.o, state.ray.d, found_intersection ? intersection.t : kMaxFloat);
          state.throughput *= medium_sample.weight;
          ETX_VALIDATE(state.throughput);
        }

        if (medium_sample.sampled_medium()) {
          path_distance += medium_sample.t;
          const auto& medium = rt.scene().mediums[medium_index];
          state.ray.o = medium_sample.pos;
          state.ray.d = medium.sample_phase_function(spect, smp, medium_sample.pos, state.ray.d);
          path_length += 1;
        } else if (found_intersection) {
          path_distance += intersection.t;

          const auto& tri = rt.scene().triangles[intersection.triangle_index];
          const auto& mat = rt.scene().materials[tri.material_index];

          if (mat.cls == Material::Class::Boundary) {
            auto bsdf_sample = bsdf::sample({spect, medium_index, PathSource::Camera, intersection, intersection.w_i, {}}, mat, rt.scene(), smp);
            if (bsdf_sample.properties & BSDFSample::MediumChanged) {
              medium_index = bsdf_sample.medium_index;
            }
            state.ray.o = intersection.pos;
            state.ray.d = bsdf_sample.w_o;
            continue;
          }

          {
            float cos_to_prev = fabsf(dot(intersection.nrm, -state.ray.d));
            state.d_vcm *= sqr(path_distance) / cos_to_prev;
            state.d_vc /= cos_to_prev;
            state.d_vm /= cos_to_prev;
            path_distance = 0.0f;
          }

          if (_vcm_options.direct_hit() && (tri.emitter_index != kInvalidIndex)) {
            const auto& emitter = rt.scene().emitters[tri.emitter_index];
            gathered += vcm_get_radiance(rt.scene(), spect, emitter, intersection, state, path_length, _vcm_options.enable_mis());
          }

          bool do_connection = _vcm_options.connect_to_light() && (bsdf::is_delta(mat, intersection.tex, rt.scene(), smp) == false);
          if (do_connection && ((path_length + 1) <= opt_max_depth)) {
            auto emitter_sample = sample_emitter(spect, smp, intersection.pos, rt.scene());

            if (emitter_sample.pdf_dir > 0) {
              BSDFData connection_data = {spect, medium_index, PathSource::Camera, intersection, state.ray.d, emitter_sample.direction};
              BSDFEval connection_eval = bsdf::evaluate(connection_data, mat, rt.scene(), smp);
              if (connection_eval.valid()) {
                float3 p0 = shading_pos(rt.scene().vertices, tri, intersection.barycentric, normalize(emitter_sample.origin - intersection.pos));
                auto tr = transmittance(spect, smp, p0, emitter_sample.origin, medium_index, rt.scene(), rt);
                if (tr.is_zero() == false) {
                  float l_dot_n = fabsf(dot(emitter_sample.direction, intersection.nrm));
                  float l_dot_e = fabsf(dot(emitter_sample.direction, emitter_sample.normal));
                  float reverse_pdf = bsdf::pdf(connection_data.swap_directions(), mat, rt.scene(), smp);

                  float w_light = emitter_sample.is_delta ? 0.0f : (connection_eval.pdf / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));

                  float w_camera = (emitter_sample.pdf_dir_out * l_dot_n) / (emitter_sample.pdf_dir * l_dot_e) *  //
                                   (_it.vm_weight + state.d_vcm + state.d_vc * reverse_pdf);                      //

                  float weight = _vcm_options.enable_mis() ? (1.0f / (1.0f + w_light + w_camera)) : 1.0f;

                  gathered += tr * state.throughput * connection_eval.bsdf * emitter_sample.value * (weight / (emitter_sample.pdf_dir * emitter_sample.pdf_sample));
                  ETX_VALIDATE(gathered);
                }
              }
            }
          }

          do_connection = _vcm_options.connect_vertices() && (bsdf::is_delta(mat, intersection.tex, rt.scene(), smp) == false);
          for (uint64_t i = 0; do_connection && (i < light_path.length) && (path_length + i + 2 <= opt_max_depth); ++i) {
            float3 target_position = {};
            SpectralResponse value = {};
            if (vcm_connect_to_light_vertex(rt.scene(), spect, state, intersection, _light_vertices[light_path.begin + i], _it.vm_weight, medium_index, target_position, value,
                  smp)) {
              float3 p0 = shading_pos(rt.scene().vertices, tri, intersection.barycentric, normalize(target_position - intersection.pos));
              auto tr = transmittance(spect, smp, p0, target_position, medium_index, rt.scene(), rt);
              if (tr.is_zero() == false) {
                gathered += tr * value;
                ETX_VALIDATE(gathered);
              }
            }
          }

          if (_vcm_options.merge_vertices() && (path_length + 1 <= opt_max_depth)) {
            merged += _current_grid.data.gather(rt.scene(), spect, state, _light_vertices.data(), intersection, medium_index, path_length, opt_max_depth, _it.vc_weight, smp);
          }

          if (vcm_next_ray(rt.scene(), spect, PathSource::Camera, intersection, path_length, opt_rr_start, smp, state, medium_index, state_eta, _it) == false) {
            break;
          }

          path_length += 1;
        } else {
          bool gather_light = _vcm_options.direct_hit() || (path_length == 1);
          for (uint32_t ie = 0; gather_light && (ie < rt.scene().environment_emitters.count); ++ie) {
            const auto& emitter = rt.scene().emitters[rt.scene().environment_emitters.emitters[ie]];
            gathered += vcm_get_radiance(rt.scene(), spect, emitter, intersection, state, path_length, _vcm_options.enable_mis());
          }
          break;
        }
      }

      merged *= _it.vm_normalization;
      merged += (gathered / spectrum::sample_pdf()).to_xyz();

      camera_image.accumulate({merged.x, merged.y, merged.z, 1.0f}, uv, float(iteration) / float(iteration + 1));
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
  result.add(1u, _private->opt_max_depth, 65536u, "pathlen", "Maximal Path Length");
  result.add(1u, _private->opt_rr_start, 65536u, "rrstart", "Start Russian Roulette at");
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
    _private->iteration_light_image.flush_to(_private->light_image, float(_private->iteration) / float(_private->iteration + 1));
    _private->stats.m_time = tm.measure();
    _private->continue_iteration();
  } else if (current_state == State::WaitingForCompletion) {
    rt.scheduler().wait(_private->current_task);
    current_state = Integrator::State::Stopped;
    _private->current_task = {};
  } else if (_private->iteration + 1 < _private->opt_max_iterations) {
    _private->iteration += 1;
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
    snprintf(_private->status, sizeof(_private->status), "[%u] Waiting for completion", _private->iteration);
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
