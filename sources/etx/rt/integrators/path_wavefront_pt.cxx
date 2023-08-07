#include <etx/rt/integrators/path_wavefront_pt.hxx>

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

struct CPUWavefrontPTImpl {
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
  bool count_wasted = false;

  std::atomic<uint32_t> completed_rays = {};

  std::vector<uint32_t> hit_rays;
  std::atomic<uint32_t> hit_ray_count = {};

  std::vector<uint32_t> miss_rays;
  std::atomic<uint32_t> miss_ray_count = {};

  std::vector<uint32_t> active_rays;
  std::atomic<uint32_t> active_ray_count = {};

  void start(Raytracing& rt, const Options& opt) {
    options.rr_start = opt.get("rrstart", options.rr_start).to_integer();
    options.nee = opt.get("nee", options.nee).to_bool();
    options.mis = opt.get("mis", options.mis).to_bool();
    count_wasted = opt.get("wasted", count_wasted).to_bool();

    iteration = 0;
    dimensions = camera_image.dimensions();
    count = dimensions.x * dimensions.y;
    storage.init(count, payload);

    hit_rays.resize(count);
    miss_rays.resize(count);
    active_rays.resize(count);

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
    completed_rays = 0u;
    rt.scheduler().execute(count, [this, &rt](uint32_t begin, uint32_t end, uint32_t thread_id) {
      raygen_execute_range(rt, begin, end, thread_id);
    });
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

  void raygen_execute_range(Raytracing& rt, uint32_t begin, uint32_t end, uint32_t thread_id) {
    const auto& scene = rt.scene();
    for (uint32_t i = begin; i < end; ++i) {
      make_ray_payload(scene, dimensions, 0u, payload, i);
    }
  }

  void trace_execute_range(Raytracing& rt, uint32_t begin, uint32_t end, uint32_t thread_id) {
    const auto& scene = rt.scene();
    for (uint32_t i = begin; i < end; ++i) {
      switch (payload.ray_state[i]) {
        case PTRayState::EndIteration: {
          float it = float(payload.iteration[i]) / float(payload.iteration[i] + 1u);
          auto xyz = (payload.accumulated[i] / spectrum::sample_pdf()).to_xyz();
          camera_image.accumulate({xyz.x, xyz.y, xyz.z, 1.0f}, i % dimensions.x, i / dimensions.x, it);
          make_ray_payload(scene, dimensions, payload.iteration[i] + 1u, payload, i);
          if (payload.iteration[i] == rt.scene().samples) {
            payload.ray_state[i] = PTRayState::Finished;
            completed_rays += 1u;
          }
          break;
        }
        case PTRayState::ContinueIteration: {
          active_rays[active_ray_count++] = i;
          payload.intersection[i].t = kMaxFloat;
          bool intersection_found = rt.trace(scene, payload.ray[i], payload.intersection[i], payload.smp[i]);
          payload.ray_state[i] = intersection_found ? PTRayState::IntersectionFound : PTRayState::NoIntersection;
          break;
        }
        case PTRayState::Finished: {
          if (count_wasted) {
            ++wasted_invocations;
          }
          break;
        }
        default:
          ETX_FAIL("Invalid ray state");
          break;
      }
    }
  }

  void medium_execute_range(Raytracing& rt, uint32_t begin, uint32_t end, uint32_t thread_id) {
    for (uint32_t ri = begin; ri < end; ++ri) {
      uint32_t i = active_rays[ri];

      auto& ray_state = payload.ray_state[i];
      if ((ray_state != PTRayState::IntersectionFound) && (ray_state != PTRayState::NoIntersection)) {
        if (count_wasted) {
          ++wasted_invocations;
        }
        continue;
      }

      const auto& scene = rt.scene();
      Medium::Sample medium_sample = try_sampling_medium(scene, payload, i);
      if (medium_sample.sampled_medium()) {
        handle_sampled_medium(scene, medium_sample, rt, payload, i);
        bool should_continue = random_continue(payload.path_length[i], options.rr_start, payload.eta[i], payload.smp[i], payload.throughput[i]);
        ray_state = should_continue ? PTRayState::ContinueIteration : PTRayState::EndIteration;
      } else if (ray_state == PTRayState::IntersectionFound) {
        hit_rays[hit_ray_count++] = i;
      } else if (ray_state == PTRayState::NoIntersection) {
        miss_rays[miss_ray_count++] = i;
      } else {
        ETX_FAIL("Invalid ray state");
      }
    }
  }

  void direct_execute_range(Raytracing& rt, uint32_t begin, uint32_t end, uint32_t thread_id) {
    const auto& scene = rt.scene();
    for (uint32_t ri = begin; ri < end; ++ri) {
      uint32_t i = hit_rays[ri];

      if (payload.ray_state[i] != PTRayState::IntersectionFound) {
        if (count_wasted) {
          ++wasted_invocations;
        }
        continue;
      }

      const auto& intersection = payload.intersection[i];
      const auto& tri = scene.triangles[intersection.triangle_index];
      const auto& mat = scene.materials[intersection.material_index];

      if (mat.cls == Material::Class::Boundary) {
        payload.medium[i] = (dot(intersection.nrm, payload.ray[i].d) < 0.0f) ? mat.int_medium : mat.ext_medium;
        payload.ray[i].o = shading_pos(scene.vertices, tri, intersection.barycentric, payload.ray[i].d);
        payload.ray_state[i] = PTRayState::ContinueIteration;
        continue;
      }

      handle_direct_emitter(scene, tri, rt, options.mis, payload, i);
    }
  }

  void subsurface_execute_range(Raytracing& rt, uint32_t begin, uint32_t end, uint32_t thread_id) {
    const auto& scene = rt.scene();
    for (uint32_t ri = begin; ri < end; ++ri) {
      uint32_t i = hit_rays[ri];

      if (payload.ray_state[i] != PTRayState::IntersectionFound) {
        if (count_wasted) {
          ++wasted_invocations;
        }
        continue;
      }

      auto& smp = payload.smp[i];
      const auto& intersection = payload.intersection[i];
      const auto& mat = scene.materials[intersection.material_index];
      payload.bsdf_sample[i] = bsdf::sample({payload.spect[i], payload.medium[i], PathSource::Camera, intersection, intersection.w_i}, mat, scene, smp);

      bool subsurface_path = mat.has_subsurface_scattering() && (payload.bsdf_sample[i].properties & BSDFSample::Diffuse);
      payload.subsurface_sampled[i] = subsurface_path && subsurface::gather(payload.spect[i], scene, intersection, intersection.material_index, rt, smp, payload.ss_gather[i]);
    }
  }

  void lights_execute_range(Raytracing& rt, uint32_t begin, uint32_t end, uint32_t thread_id) {
    const auto& scene = rt.scene();
    for (uint32_t ri = begin; ri < end; ++ri) {
      uint32_t i = hit_rays[ri];

      if (payload.ray_state[i] != PTRayState::IntersectionFound) {
        if (count_wasted) {
          ++wasted_invocations;
        }
        continue;
      }

      auto& smp = payload.smp[i];
      const auto& ss_gather = payload.ss_gather[i];
      const auto& bsdf_sample = payload.bsdf_sample[i];
      const auto& intersection = payload.intersection[i];
      const auto& mat = scene.materials[intersection.material_index];

      if (options.nee && (payload.path_length[i] + 1 <= rt.scene().max_path_length)) {
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

      if (bsdf_sample.valid() == false) {
        payload.ray_state[i] = PTRayState::EndIteration;
        continue;
      }

      bool subsurface_path = mat.has_subsurface_scattering() && (bsdf_sample.properties & BSDFSample::Diffuse);
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

  void walk_execute_range(Raytracing& rt, uint32_t begin, uint32_t end, uint32_t thread_id) {
    const auto& scene = rt.scene();
    for (uint32_t ri = begin; ri < end; ++ri) {
      uint32_t i = hit_rays[ri];

      if (payload.ray_state[i] != PTRayState::IntersectionFound) {
        if (count_wasted) {
          ++wasted_invocations;
        }
        continue;
      }

      auto& smp = payload.smp[i];
      const auto& bsdf_sample = payload.bsdf_sample[i];
      const auto& intersection = payload.intersection[i];

      if (payload.subsurface_sampled[i]) {
        payload.ray[i].d = sample_cosine_distribution(smp.next_2d(), intersection.nrm, 1.0f);
        payload.sampled_bsdf_pdf[i] = fabsf(dot(payload.ray[i].d, intersection.nrm)) / kPi;
        payload.mis_weight[i] = true;
        payload.ray[i].o = shading_pos(scene.vertices, scene.triangles[intersection.triangle_index], intersection.barycentric, payload.ray[i].d);
      } else {
        payload.medium[i] = (bsdf_sample.properties & BSDFSample::MediumChanged) ? bsdf_sample.medium_index : payload.medium[i];
        payload.sampled_bsdf_pdf[i] = bsdf_sample.pdf;
        payload.mis_weight[i] = bsdf_sample.is_delta() == false;
        payload.eta[i] *= bsdf_sample.eta;
        payload.ray[i].d = bsdf_sample.w_o;
        payload.ray[i].o = shading_pos(scene.vertices, scene.triangles[intersection.triangle_index], intersection.barycentric, payload.ray[i].d);
      }

      payload.throughput[i] *= bsdf_sample.weight;
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

  void miss_execute_range(Raytracing& rt, uint32_t begin, uint32_t end, uint32_t thread_id) {
    const auto& scene = rt.scene();
    for (uint32_t ri = begin; ri < end; ++ri) {
      uint32_t i = miss_rays[ri];

      if (payload.ray_state[i] != PTRayState::NoIntersection) {
        if (count_wasted) {
          ++wasted_invocations;
        }
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

  void frame(Raytracing& rt) {
    TimeMeasure tm = {};

    {
      active_ray_count = 0;
      wasted_invocations = 0;
      rt.scheduler().execute(count, [this, &rt](uint32_t begin, uint32_t end, uint32_t thread_id) {
        trace_execute_range(rt, begin, end, thread_id);
      });
      durations[0] = tm.lap();
      wasted[0] = wasted_invocations;

      hit_ray_count = 0;
      miss_ray_count = 0;
      wasted_invocations = 0;
      rt.scheduler().execute(active_ray_count, [this, &rt](uint32_t begin, uint32_t end, uint32_t thread_id) {
        medium_execute_range(rt, begin, end, thread_id);
      });
      durations[1] = tm.lap();
      wasted[1] = wasted_invocations;
    }

    {
      wasted_invocations = 0;
      rt.scheduler().execute(hit_ray_count, [this, &rt](uint32_t begin, uint32_t end, uint32_t thread_id) {
        direct_execute_range(rt, begin, end, thread_id);
      });
      durations[2] = tm.lap();
      wasted[2] = wasted_invocations;

      wasted_invocations = 0;
      rt.scheduler().execute(hit_ray_count, [this, &rt](uint32_t begin, uint32_t end, uint32_t thread_id) {
        subsurface_execute_range(rt, begin, end, thread_id);
      });
      durations[3] = tm.lap();
      wasted[3] = wasted_invocations;

      wasted_invocations = 0;
      rt.scheduler().execute(hit_ray_count, [this, &rt](uint32_t begin, uint32_t end, uint32_t thread_id) {
        lights_execute_range(rt, begin, end, thread_id);
      });
      durations[4] = tm.lap();
      wasted[4] = wasted_invocations;

      wasted_invocations = 0;
      rt.scheduler().execute(hit_ray_count, [this, &rt](uint32_t begin, uint32_t end, uint32_t thread_id) {
        walk_execute_range(rt, begin, end, thread_id);
      });
      durations[5] = tm.lap();
      wasted[5] = wasted_invocations;
    }

    {
      wasted_invocations = 0;
      rt.scheduler().execute(miss_ray_count, [this, &rt](uint32_t begin, uint32_t end, uint32_t thread_id) {
        miss_execute_range(rt, begin, end, thread_id);
      });
      durations[6] = tm.lap();
      wasted[6] = wasted_invocations;
    }
  }
};

CPUWavefrontPT::CPUWavefrontPT(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUWavefrontPT);
}

CPUWavefrontPT::~CPUWavefrontPT() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }

  ETX_PIMPL_CLEANUP(CPUWavefrontPT);
}

void CPUWavefrontPT::set_output_size(const uint2& dim) {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  _private->camera_image.resize(dim, 1);
}

const float4* CPUWavefrontPT::get_camera_image(bool force_update) {
  return _private->camera_image.data();
}

const float4* CPUWavefrontPT::get_light_image(bool force_update) {
  return nullptr;
}

const char* CPUWavefrontPT::status() const {
  return _private->status;
}

void CPUWavefrontPT::preview(const Options& opt) {
  stop(Stop::Immediate);
  if (rt.has_scene() == false)
    return;

  current_state = State::Preview;
  _private->start(rt, opt);
}

void CPUWavefrontPT::run(const Options& opt) {
  stop(Stop::Immediate);
  if (rt.has_scene() == false)
    return;

  current_state = State::Running;
  _private->start(rt, opt);
}

void CPUWavefrontPT::update() {
  constexpr uint32_t frames_per_update = 1u;
  const uint32_t debug_info_size = sizeof(CPUWavefrontPTImpl::durations) / sizeof(CPUWavefrontPTImpl::durations[0]);

  double durations[debug_info_size] = {};
  uint32_t wasted[debug_info_size] = {};

  TimeMeasure iteration_timer = {};
  for (uint32_t i = 0; (i < frames_per_update) && ((current_state == State::Preview) || (current_state == State::Running)); ++i) {
    _private->frame(rt);

    for (uint32_t j = 0; j < debug_info_size; ++j) {
      durations[j] += _private->durations[j];
      wasted[j] += _private->wasted[j];
    }

    if (_private->completed_rays == _private->count) {
      snprintf(_private->status, sizeof(_private->status), "Completed in %.3f", _private->total_time.measure());
      current_state = State::Stopped;
    }
  }

  if ((current_state == State::Preview) || (current_state == State::Running)) {
    double total_rays = double(frames_per_update * _private->count);
    double iteration_time = iteration_timer.measure() * 1000.0f;
    float completeness = float(_private->completed_rays) / float(_private->count) * 100.0f;
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

void CPUWavefrontPT::stop(Stop st) {
  if (current_state == State::Stopped) {
    return;
  }

  current_state = State::Stopped;
}

Options CPUWavefrontPT::options() const {
  Options result = {};
  result.add(1u, _private->options.rr_start, 65536u, "rrstart", "Start Random Path Termination at");
  result.add(_private->options.nee, "nee", "Next Event Estimation");
  result.add(_private->options.mis, "mis", "Multiple Importance Sampling");
  result.add(_private->count_wasted, "wasted", "Count Wasted Rays");
  return result;
}

void CPUWavefrontPT::update_options(const Options& opt) {
  if (current_state == State::Preview) {
    preview(opt);
  }
}

uint64_t CPUWavefrontPT::debug_info_count() const {
  return std::size(_private->debug_infos);
}

Integrator::DebugInfo* CPUWavefrontPT::debug_info() const {
  return _private->debug_infos;
}

}  // namespace etx
