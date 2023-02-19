#include <etx/core/core.hxx>
#include <etx/log/log.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/bsdf.hxx>
#include <etx/render/host/rnd_sampler.hxx>
#include <etx/render/host/film.hxx>

#include <etx/rt/integrators/path_tracing_2.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

namespace etx {

struct PTRayPayloadSoAStorage {
#define DATA(Cls, name)                                     \
  constexpr static uint64_t k_##name##_size = sizeof(Cls);  \
  std::vector<Cls> name##_data;                             \
  void init_##name(uint32_t count, PTRayPayloadSoA& data) { \
    name##_data.resize(count);                              \
    data.name = ArrayView<Cls>(name##_data.data(), count);  \
    total_size += k_##name##_size;                          \
  }
  DATA(Ray, ray);
  DATA(Intersection, intersection);
  DATA(PTRayState, ray_state);
  DATA(SpectralQuery, spect);
  DATA(uint32_t, iteration);
  DATA(uint32_t, medium);
  DATA(SpectralResponse, throughput);
  DATA(SpectralResponse, accumulated);
  DATA(Sampler, smp);
  DATA(uint8_t, mis_weight);
  DATA(uint32_t, path_length);
  DATA(float, sampled_bsdf_pdf);
  DATA(float, eta);
  DATA(subsurface::Gather, ss_gather);
  DATA(BSDFSample, bsdf_sample);
  DATA(uint8_t, subsurface_sampled);
#undef DATA

  uint64_t total_size = 0;

  void init(uint32_t count, PTRayPayloadSoA& data) {
    total_size = 0llu;
    init_ray(count, data);
    init_intersection(count, data);
    init_ray_state(count, data);
    init_spect(count, data);
    init_iteration(count, data);
    init_medium(count, data);
    init_throughput(count, data);
    init_accumulated(count, data);
    init_smp(count, data);
    init_mis_weight(count, data);
    init_path_length(count, data);
    init_sampled_bsdf_pdf(count, data);
    init_eta(count, data);
    init_ss_gather(count, data);
    init_bsdf_sample(count, data);
    init_subsurface_sampled(count, data);
  }
};

struct CPUPathTracing2Impl {
  char status[2048] = {};
  Film camera_image;

  uint32_t count = 0;
  uint32_t iteration = 0;

  PTOptions options = {};

  uint2 dimensions = {};
  TimeMeasure total_time = {};

  PTRayPayloadSoA payload;
  PTRayPayloadSoAStorage storage;

  double durations[7] = {};
  uint32_t wasted[sizeof(durations) / sizeof(durations[0])] = {};
  Integrator::DebugInfo debug_infos[2 * sizeof(durations) / sizeof(durations[0]) + 1llu] = {};
  std::atomic<uint32_t> wasted_invocations = {};

  void start(Raytracing& rt, const Options& opt) {
    options.iterations = opt.get("spp", options.iterations).to_integer();
    options.max_depth = opt.get("pathlen", options.max_depth).to_integer();
    options.rr_start = opt.get("rrstart", options.rr_start).to_integer();
    options.nee = opt.get("nee", options.nee).to_bool();
    options.mis = opt.get("mis", options.mis).to_bool();

    iteration = 0;
    dimensions = camera_image.dimensions();
    count = dimensions.x * dimensions.y;
    storage.init(count, payload);
    double size_current = double(storage.total_size * count);
    double size_full_hd = double(storage.total_size * 1920llu * 1080llu);
    double size_4k = size_full_hd * 4.0;
    constexpr double kMegabyte = 1024.0 * 1024.0;
    constexpr double kGigabyte = kMegabyte * 1024.0;

    log::info(
      "Memory usage:\n"
      "  - Payload : %12llu |\n"
      "  - Current : %12llu | %8.2f Mb | %8.2f Gb\n"
      "  - FullHD  : %12llu | %8.2f Mb | %8.2f Gb\n"
      "  - 4K      : %12llu | %8.2f Mb | %8.2f Gb",
      storage.total_size,                                                          //
      uint64_t(size_current), size_current / kMegabyte, size_current / kGigabyte,  //
      uint64_t(size_full_hd), size_full_hd / kMegabyte, size_full_hd / kGigabyte,  //
      uint64_t(size_4k), size_4k / kMegabyte, size_4k / kGigabyte                  //
    );

    total_time = {};

    task_trace.init(rt, payload, wasted_invocations, options);
    task_trace.camera_image = &camera_image;
    task_trace.dimensions = dimensions;
    task_trace.completed_rays = 0u;

    task_medium.init(rt, payload, wasted_invocations, options);
    task_direct.init(rt, payload, wasted_invocations, options);
    task_intersected_subsurface.init(rt, payload, wasted_invocations, options);
    task_intersected_gather_lights.init(rt, payload, wasted_invocations, options);
    task_intersected_walk.init(rt, payload, wasted_invocations, options);
    task_missed.init(rt, payload, wasted_invocations, options);

    task_raygen.init(rt, payload, wasted_invocations, options);
    task_raygen.dimensions = dimensions;
    rt.scheduler().execute(count, &task_raygen);
  }

  struct PTTask : public Task {
    PTRayPayloadSoA* payload_ptr = nullptr;
    Raytracing* rt_ptr = nullptr;
    PTOptions options = {};
    std::atomic<uint32_t>* wasted = nullptr;

    void init(Raytracing& rt, PTRayPayloadSoA& payload, std::atomic<uint32_t>& w, const PTOptions& opt) {
      rt_ptr = &rt;
      payload_ptr = &payload;
      wasted = &w;
      options = opt;
    }
  };

  struct TraceRaygen : PTTask {
    Film* camera_image = nullptr;
    std::atomic<uint32_t> completed_rays = {};
    uint2 dimensions = {};

    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      auto& rt = *rt_ptr;
      auto& payload = *payload_ptr;
      const auto& scene = rt.scene();
      for (uint32_t i = begin; i < end; ++i) {
        make_ray_payload(scene, dimensions, 0u, payload, i);
      }
    }
  } task_raygen;

  struct TraceTask : PTTask {
    Film* camera_image = nullptr;
    std::atomic<uint32_t> completed_rays = {};
    uint2 dimensions = {};

    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      auto& rt = *rt_ptr;
      auto& payload = *payload_ptr;
      const auto& scene = rt.scene();
      for (uint32_t i = begin; i < end; ++i) {
        if (payload.ray_state[i] == PTRayState::Finished) {
          // wasted->fetch_add(1u);
        } else if (payload.ray_state[i] == PTRayState::EndIteration) {
          float it = float(payload.iteration[i]) / float(payload.iteration[i] + 1u);
          auto xyz = (payload.accumulated[i] / spectrum::sample_pdf()).to_xyz();
          camera_image->accumulate({xyz.x, xyz.y, xyz.z, 1.0f}, i % dimensions.x, i / dimensions.x, it);
          make_ray_payload(scene, dimensions, payload.iteration[i] + 1u, payload, i);
          if (payload.iteration[i] == options.iterations) {
            payload.ray_state[i] = PTRayState::Finished;
            completed_rays += 1u;
          }
        } else if (payload.ray_state[i] == PTRayState::ContinueIteration) {
          payload.intersection[i].t = kMaxFloat;
          bool found = rt.trace(scene, payload.ray[i], payload.intersection[i], payload.smp[i]);
          payload.ray_state[i] = found ? PTRayState::IntersectionFound : PTRayState::NoIntersection;
        } else {
          ETX_FAIL("Invalid ray state");
        }
      }
    }
  } task_trace;

  struct HandleMediumTask : PTTask {
    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      auto& rt = *rt_ptr;
      auto& payload = *payload_ptr;
      const auto& scene = rt.scene();
      for (uint32_t i = begin; i < end; ++i) {
        if ((payload.ray_state[i] != PTRayState::IntersectionFound) && (payload.ray_state[i] != PTRayState::NoIntersection)) {
          // wasted->fetch_add(1u);
          continue;
        }

        const auto& scene = rt.scene();
        Medium::Sample medium_sample = try_sampling_medium(scene, payload, i);
        if (medium_sample.sampled_medium()) {
          handle_sampled_medium(scene, medium_sample, options.max_depth, rt, payload, i);
          bool should_continue = random_continue(payload.path_length[i], options.rr_start, payload.eta[i], payload.smp[i], payload.throughput[i]);
          payload.ray_state[i] = should_continue ? PTRayState::ContinueIteration : PTRayState::EndIteration;
        }
      }
    }
  } task_medium;

  struct TaskDirect : PTTask {
    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      auto& rt = *rt_ptr;
      auto& payload = *payload_ptr;
      const auto& scene = rt.scene();
      for (uint32_t i = begin; i < end; ++i) {
        if (payload.ray_state[i] != PTRayState::IntersectionFound) {
          // wasted->fetch_add(1u);
          continue;
        }

        const auto& intersection = payload.intersection[i];
        const auto& tri = scene.triangles[intersection.triangle_index];
        const auto& mat = scene.materials[tri.material_index];

        if (mat.cls == Material::Class::Boundary) {
          payload.medium[i] = (dot(intersection.nrm, payload.ray[i].d) < 0.0f) ? mat.int_medium : mat.ext_medium;
          payload.ray[i].o = shading_pos(scene.vertices, tri, intersection.barycentric, payload.ray[i].d);
          payload.ray_state[i] = PTRayState::ContinueIteration;
          continue;
        }

        handle_direct_emitter(scene, tri, rt, options.mis, payload, i);
      }
    }
  } task_direct;

  struct TaskIntersectedSubsurface : PTTask {
    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      auto& rt = *rt_ptr;
      auto& payload = *payload_ptr;
      const auto& scene = rt.scene();
      for (uint32_t i = begin; i < end; ++i) {
        if (payload.ray_state[i] != PTRayState::IntersectionFound) {
          // wasted->fetch_add(1u);
          continue;
        }

        auto& smp = payload.smp[i];
        const auto& intersection = payload.intersection[i];
        const auto& tri = scene.triangles[intersection.triangle_index];
        const auto& mat = scene.materials[tri.material_index];
        payload.bsdf_sample[i] = bsdf::sample({payload.spect[i], payload.medium[i], PathSource::Camera, intersection, intersection.w_i}, mat, scene, smp);

        bool subsurface_path = mat.has_subsurface_scattering() && (payload.bsdf_sample[i].properties & BSDFSample::Diffuse);
        payload.subsurface_sampled[i] = subsurface_path && subsurface::gather(payload.spect[i], scene, intersection, tri.material_index, rt, smp, payload.ss_gather[i]);
      }
    }
  } task_intersected_subsurface;

  struct TaskIntersectedLights : PTTask {
    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      auto& rt = *rt_ptr;
      auto& payload = *payload_ptr;
      const auto& scene = rt.scene();
      for (uint32_t i = begin; i < end; ++i) {
        if (payload.ray_state[i] != PTRayState::IntersectionFound) {
          // wasted->fetch_add(1u);
          continue;
        }

        auto& smp = payload.smp[i];
        const auto& ss_gather = payload.ss_gather[i];
        const auto& bsdf_sample = payload.bsdf_sample[i];
        const auto& intersection = payload.intersection[i];
        const auto& tri = scene.triangles[intersection.triangle_index];
        const auto& mat = scene.materials[tri.material_index];

        if (options.nee && (payload.path_length[i] + 1 <= options.max_depth)) {
          uint32_t emitter_index = sample_emitter_index(scene, smp);
          SpectralResponse direct_light = {payload.spect[i].wavelength, 0.0f};
          if (payload.subsurface_sampled[i]) {
            for (uint32_t is = 0; is < ss_gather.intersection_count; ++is) {
              auto local_sample = sample_emitter(payload.spect[i], emitter_index, smp, ss_gather.intersections[is].pos, scene);
              SpectralResponse light_value = evaluate_light(scene, ss_gather.intersections[is], rt, mat, payload.medium[i], payload.spect[i], local_sample, smp, options.mis);
              direct_light += ss_gather.weights[is] * light_value;
              ETX_VALIDATE(direct_light);
            }
          } else {
            auto emitter_sample = sample_emitter(payload.spect[i], emitter_index, smp, intersection.pos, scene);
            direct_light += evaluate_light(scene, intersection, rt, mat, payload.medium[i], payload.spect[i], emitter_sample, smp, options.mis);
            ETX_VALIDATE(payload.accumulated[i]);
          }
          payload.accumulated[i] += payload.throughput[i] * direct_light;
        }

        if (payload.bsdf_sample[i].valid() == false) {
          payload.ray_state[i] = PTRayState::EndIteration;
          continue;
        }

        bool subsurface_path = mat.has_subsurface_scattering() && (payload.bsdf_sample[i].properties & BSDFSample::Diffuse);
        if (subsurface_path && (payload.subsurface_sampled[i] == 0)) {
          payload.ray_state[i] = PTRayState::EndIteration;
          continue;
        }

        if (payload.subsurface_sampled[i]) {
          payload.intersection[i] = ss_gather.intersections[ss_gather.selected_intersection];
          payload.throughput[i] *= ss_gather.weights[ss_gather.selected_intersection] * ss_gather.selected_sample_weight;
        }
      }
    }
  } task_intersected_gather_lights;

  struct TaskIntersectedWalk : PTTask {
    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      auto& rt = *rt_ptr;
      auto& payload = *payload_ptr;
      const auto& scene = rt.scene();
      for (uint32_t i = begin; i < end; ++i) {
        if (payload.ray_state[i] != PTRayState::IntersectionFound) {
          // wasted->fetch_add(1u);
          continue;
        }

        auto& smp = payload.smp[i];
        const auto& bsdf_sample = payload.bsdf_sample[i];
        const auto& intersection = payload.intersection[i];
        const auto& tri = scene.triangles[intersection.triangle_index];
        const auto& mat = scene.materials[tri.material_index];

        if (payload.subsurface_sampled[i]) {
          payload.ray[i].d = sample_cosine_distribution(smp.next_2d(), intersection.nrm, 1.0f);
          payload.sampled_bsdf_pdf[i] = fabsf(dot(payload.ray[i].d, intersection.nrm)) / kPi;
          payload.mis_weight[i] = true;
          payload.ray[i].o = shading_pos(scene.vertices, scene.triangles[intersection.triangle_index], intersection.barycentric, payload.ray[i].d);
        } else {
          payload.medium[i] = (payload.bsdf_sample[i].properties & BSDFSample::MediumChanged) ? payload.bsdf_sample[i].medium_index : payload.medium[i];
          payload.sampled_bsdf_pdf[i] = payload.bsdf_sample[i].pdf;
          payload.mis_weight[i] = payload.bsdf_sample[i].is_delta() == false;
          payload.eta[i] *= payload.bsdf_sample[i].eta;
          payload.ray[i].d = payload.bsdf_sample[i].w_o;
          payload.ray[i].o = shading_pos(scene.vertices, scene.triangles[intersection.triangle_index], intersection.barycentric, payload.ray[i].d);
        }

        payload.throughput[i] *= payload.bsdf_sample[i].weight;
        if (payload.throughput[i].is_zero()) {
          payload.ray_state[i] = PTRayState::EndIteration;
          continue;
        }

        payload.path_length[i] += 1;
        ETX_CHECK_FINITE(payload.ray[i].d);
        bool should_continue = random_continue(payload.path_length[i], options.rr_start, payload.eta[i], smp, payload.throughput[i]);
        payload.ray_state[i] = should_continue ? PTRayState::ContinueIteration : PTRayState::EndIteration;
      }
    }
  } task_intersected_walk;

  struct TaskMissed : PTTask {
    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      auto& rt = *rt_ptr;
      auto& payload = *payload_ptr;
      const auto& scene = rt.scene();
      for (uint32_t i = begin; i < end; ++i) {
        if (payload.ray_state[i] != PTRayState::NoIntersection) {
          // wasted->fetch_add(1u);
          continue;
        }

        for (uint32_t ie = 0; ie < scene.environment_emitters.count; ++ie) {
          const auto& emitter = scene.emitters[scene.environment_emitters.emitters[ie]];
          float pdf_emitter_area = 0.0f;
          float pdf_emitter_dir = 0.0f;
          float pdf_emitter_dir_out = 0.0f;
          auto e = emitter_get_radiance(emitter, payload.spect[i], payload.ray[i].d, pdf_emitter_area, pdf_emitter_dir, pdf_emitter_dir_out, scene);
          ETX_VALIDATE(e);
          if ((pdf_emitter_dir > 0) && (e.is_zero() == false)) {
            float pdf_emitter_discrete = emitter_discrete_pdf(emitter, scene.emitters_distribution);
            auto weight = ((payload.mis_weight[i] == false) || (payload.path_length[i] == 1))  //
                            ? 1.0f                                                             //
                            : power_heuristic(payload.sampled_bsdf_pdf[i], pdf_emitter_discrete * pdf_emitter_dir);
            payload.accumulated[i] += payload.throughput[i] * e * weight;
            ETX_VALIDATE(payload.accumulated[i]);
          }
        }
        payload.ray_state[i] = PTRayState::EndIteration;
      }
    }
  } task_missed;

  void frame(Raytracing& rt) {
    TimeMeasure tm = {};

    wasted_invocations = 0;
    rt.scheduler().execute(count, &task_trace);
    durations[0] = tm.lap();
    wasted[0] = wasted_invocations;

    wasted_invocations = 0;
    rt.scheduler().execute(count, &task_medium);
    durations[1] = tm.lap();
    wasted[1] = wasted_invocations;

    wasted_invocations = 0;
    rt.scheduler().execute(count, &task_direct);
    durations[2] = tm.lap();
    wasted[2] = wasted_invocations;

    wasted_invocations = 0;
    rt.scheduler().execute(count, &task_intersected_subsurface);
    durations[3] = tm.lap();
    wasted[3] = wasted_invocations;

    wasted_invocations = 0;
    rt.scheduler().execute(count, &task_intersected_gather_lights);
    durations[4] = tm.lap();
    wasted[4] = wasted_invocations;

    wasted_invocations = 0;
    rt.scheduler().execute(count, &task_intersected_walk);
    durations[5] = tm.lap();
    wasted[5] = wasted_invocations;

    wasted_invocations = 0;
    rt.scheduler().execute(count, &task_missed);
    durations[6] = tm.lap();
    wasted[6] = wasted_invocations;
  }
};

CPUPathTracing2::CPUPathTracing2(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUPathTracing2);
}

CPUPathTracing2::~CPUPathTracing2() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }

  ETX_PIMPL_CLEANUP(CPUPathTracing2);
}

void CPUPathTracing2::set_output_size(const uint2& dim) {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  _private->camera_image.resize(dim, 1);
}

const float4* CPUPathTracing2::get_camera_image(bool force_update) {
  return _private->camera_image.data();
}

const float4* CPUPathTracing2::get_light_image(bool force_update) {
  return nullptr;
}

const char* CPUPathTracing2::status() const {
  return _private->status;
}

void CPUPathTracing2::preview(const Options& opt) {
  stop(Stop::Immediate);
  if (rt.has_scene() == false)
    return;

  current_state = State::Preview;
  _private->start(rt, opt);
}

void CPUPathTracing2::run(const Options& opt) {
  stop(Stop::Immediate);
  if (rt.has_scene() == false)
    return;

  current_state = State::Running;
  _private->start(rt, opt);
}

void CPUPathTracing2::update() {
  constexpr uint32_t frames_per_update = 1u;
  const uint32_t debug_info_size = sizeof(CPUPathTracing2Impl::durations) / sizeof(CPUPathTracing2Impl::durations[0]);

  double durations[debug_info_size] = {};
  uint32_t wasted[debug_info_size] = {};
  TimeMeasure iteration_timer = {};
  for (uint32_t i = 0; (i < frames_per_update) && ((current_state == State::Preview) || (current_state == State::Running)); ++i) {
    _private->frame(rt);

    for (uint32_t j = 0; j < debug_info_size; ++j) {
      durations[j] += _private->durations[j];
      wasted[j] += _private->wasted[j];
    }

    if (_private->task_trace.completed_rays == _private->count) {
      snprintf(_private->status, sizeof(_private->status), "Completed in %.3f", _private->total_time.measure());
      current_state = State::Stopped;
    }
  }

  if ((current_state == State::Preview) || (current_state == State::Running)) {
    double total_rays = double(frames_per_update * _private->count);
    double iteration_time = iteration_timer.measure() * 1000.0f;
    float completeness = float(_private->task_trace.completed_rays) / float(_private->count) * 100.0f;
    snprintf(_private->status, sizeof(_private->status), "Completed %.2f, iteration time: %.3f ms", completeness, iteration_time);

    uint32_t k = 0;
    _private->debug_infos[k++] = {"Tracing: ", float(durations[0] * 1000.0)};
    _private->debug_infos[k++] = {" + wasted: ", float(100.0 * double(wasted[0]) / total_rays)};

    _private->debug_infos[k++] = {"Medium handling: ", float(durations[1] * 1000.0)};
    _private->debug_infos[k++] = {" + wasted: ", float(100.0 * double(wasted[1]) / total_rays)};

    _private->debug_infos[k++] = {"Direct emitters: ", float(durations[2] * 1000.0)};
    _private->debug_infos[k++] = {" + wasted: ", float(100.0 * double(wasted[2]) / total_rays)};

    _private->debug_infos[k++] = {"Hit ray (subsurface): ", float(durations[3] * 1000.0)};
    _private->debug_infos[k++] = {" + wasted: ", float(100.0 * double(wasted[3]) / total_rays)};

    _private->debug_infos[k++] = {"Hit ray (lights): ", float(durations[4] * 1000.0)};
    _private->debug_infos[k++] = {" + wasted: ", float(100.0 * double(wasted[4]) / total_rays)};

    _private->debug_infos[k++] = {"Hit ray (walk): ", float(durations[5] * 1000.0)};
    _private->debug_infos[k++] = {" + wasted: ", float(100.0 * double(wasted[5]) / total_rays)};

    _private->debug_infos[k++] = {"Missed ray handling: ", float(durations[6] * 1000.0)};
    _private->debug_infos[k++] = {" + wasted: ", float(100.0 * double(wasted[6]) / total_rays)};

    _private->debug_infos[k++] = {"Total time: ", float(iteration_time)};
  }
}

void CPUPathTracing2::stop(Stop st) {
  if (current_state == State::Stopped) {
    return;
  }

  current_state = State::Stopped;
}

Options CPUPathTracing2::options() const {
  Options result = {};
  result.add(1u, _private->options.iterations, 0xffffu, "spp", "Samples per Pixel");
  result.add(1u, _private->options.max_depth, 65536u, "pathlen", "Maximal Path Length");
  result.add(1u, _private->options.rr_start, 65536u, "rrstart", "Start Random Path Termination at");
  result.add(_private->options.nee, "nee", "Next Event Estimation");
  result.add(_private->options.mis, "mis", "Multiple Importance Sampling");
  return result;
}

void CPUPathTracing2::update_options(const Options& opt) {
  if (current_state == State::Preview) {
    preview(opt);
  }
}

uint64_t CPUPathTracing2::debug_info_count() const {
  return std::size(_private->debug_infos);
}

Integrator::DebugInfo* CPUPathTracing2::debug_info() const {
  return _private->debug_infos;
}

}  // namespace etx
