#include <etx/core/core.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/bsdf.hxx>
#include <etx/render/host/rnd_sampler.hxx>
#include <etx/render/host/film.hxx>

#include <etx/rt/integrators/direct.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

namespace etx {

namespace restir {

struct Reservoir {
  EmitterSample sample = {};
  uint32_t sample_count = 0;
  float total_weight = 0.0f;
  float target_pdf = 0.0f;

  void add_samples(uint32_t count, float pdf, float weight, const EmitterSample& emitter_sample, Sampler& smp) {
    total_weight += weight;
    sample_count += count;

    if (smp.next() < weight / total_weight) {
      sample = emitter_sample;
      target_pdf = pdf;
    }
  }

  float weight() const {
    if (target_pdf == 0.0f)
      return 0.0f;

    float weight = (1.0f / target_pdf) * (1.0f / float(sample_count)) * total_weight;
    ETX_VALIDATE(weight);
    return weight;
  }
};

float sample_weight(const SpectralQuery& spect, Sampler& smp, const Intersection& intersection, const Scene& scene, const EmitterSample& emitter_sample, bool use_bsdf) {
  if (emitter_sample.pdf_dir <= 0.0f) {
    return 0.0f;
  }

  const auto& mat = scene.materials[intersection.material_index];
  float bsdf_pdf = bsdf::pdf({spect, kInvalidIndex, PathSource::Camera, intersection, intersection.w_i}, emitter_sample.direction, mat, scene, smp);
  float l_dot_n = fmaxf(0.0f, dot(emitter_sample.direction, intersection.nrm));
  if (((use_bsdf == false) && (l_dot_n > 0.0f)) || (bsdf_pdf > 0.0f)) {
    return (use_bsdf ? bsdf_pdf : l_dot_n) * emitter_sample.value.to_xyz().y / emitter_sample.pdf_dir;
  }

  return 0.0f;
}

Reservoir generate_samples(const SpectralQuery& spect, Sampler& smp, const Intersection& intersection, const Scene& scene, bool use_bsdf) {
  constexpr uint32_t kSampleCount = 32u;

  Reservoir result = {};
  for (uint32_t i = 0; i < kSampleCount; ++i) {
    uint32_t emitter_index = uint32_t(smp.next() * float(scene.emitters.count));
    auto emitter_sample = sample_emitter(spect, emitter_index, smp, intersection.pos, scene);
    float weight = sample_weight(spect, smp, intersection, scene, emitter_sample, use_bsdf);
    result.add_samples(1u, weight, weight * float(scene.emitters.count), emitter_sample, smp);
  }
  return result;
}

restir::Reservoir resample_reservoir(const restir::Reservoir& r0, const restir::Reservoir& r1,  //
  const SpectralQuery& spect, Sampler& smp, const Intersection& intersection, const Scene& scene, bool use_bsdf) {
  float r0_target_pdf = r0.target_pdf;
  float r0_weight = r0.weight() * float(r0.sample_count);

  float r1_target_pdf = sample_weight(spect, smp, intersection, scene, r1.sample, use_bsdf);
  float r1_weight = r1.weight() * float(r1.sample_count);

  restir::Reservoir result = {};
  result.add_samples(r0.sample_count, r0_target_pdf, r0_target_pdf * r0_weight, r0.sample, smp);
  result.add_samples(r1.sample_count, r1_target_pdf, r1_target_pdf * r1_weight, r1.sample, smp);
  return result;
}

}  // namespace restir

struct ETX_ALIGNED DirectOptions {
  uint32_t iterations = 1u;
  bool ris_sampling = false;
  bool ris_bsdf = false;
  bool ris_spatial = false;
};

struct ETX_ALIGNED DirectPayload {
  Ray ray = {};
  SpectralResponse accumulated = {spectrum::kUndefinedWavelength, 0.0f};
  uint32_t medium = kInvalidIndex;
  uint32_t iteration = 0u;
  SpectralQuery spect = {};
  float2 uv = {};
  Sampler smp = {};

  static DirectPayload make(const Scene& scene, uint2 px, uint2 dim, uint32_t iteration) {
    uint32_t index = px.x + px.y * dim.x;
    DirectPayload payload = {};
    payload.iteration = iteration;
    payload.smp.init(index, payload.iteration);
    payload.spect = spectrum::sample(payload.smp.next());
    payload.uv = get_jittered_uv(payload.smp, px, dim);
    payload.ray = generate_ray(payload.smp, scene, payload.uv);
    payload.accumulated = {payload.spect.wavelength, 0.0f};
    payload.medium = scene.camera_medium_index;
    return payload;
  }
};

struct CPUDirectLightingImpl {
  Raytracing& rt;
  Film camera_image;
  uint2 current_dimensions = {};
  char status[2048] = {};

  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Task::Handle current_task = {};
  uint32_t iteration = 0u;
  uint32_t preview_frames = 3;

  DirectOptions options = {};

  std::atomic<Integrator::State>* state = nullptr;
  std::vector<Intersection> intersections = {};
  std::vector<restir::Reservoir> reservoirs = {};

  enum class Operation : uint32_t {
    None,
    GenReservoirs,
    FinalGather,
  } operation = Operation::None;

  CPUDirectLightingImpl(Raytracing& a_rt, std::atomic<Integrator::State>* st)
    : rt(a_rt)
    , state(st) {
  }

  void start(const Options& opt) {
    options.iterations = opt.get("spp", options.iterations).to_integer();
    options.ris_sampling = opt.get("ris", options.ris_sampling).to_bool();
    options.ris_bsdf = opt.get("ris-bsdf", options.ris_bsdf).to_bool();
    options.ris_spatial = opt.get("ris-spat", options.ris_spatial).to_bool();

    iteration = 0;
    snprintf(status, sizeof(status), "[%u] %s ...", iteration, (state->load() == Integrator::State::Running ? "Running" : "Preview"));

    current_dimensions = camera_image.dimensions();
    reservoirs.resize(current_dimensions.x * current_dimensions.y);
    intersections.resize(current_dimensions.x * current_dimensions.y);

    total_time = {};
    start_generating_reservoirs();
  }

  void start_generating_reservoirs() {
    iteration_time = {};
    operation = Operation::GenReservoirs;
    current_task = rt.scheduler().schedule(current_dimensions.x * current_dimensions.y, &gen_reservoirs_task);
  }

  void start_final_gather() {
    operation = Operation::FinalGather;
    current_task = rt.scheduler().schedule(current_dimensions.x * current_dimensions.y, &final_gather_task);
  }

  struct FinalGatherTask : Task {
    CPUDirectLightingImpl* self = nullptr;

    FinalGatherTask(CPUDirectLightingImpl* a)
      : self(a) {
    }

    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      self->final_gather(begin, end, thread_id);
    }
  } final_gather_task = {this};

  struct GenReservoirsTask : Task {
    CPUDirectLightingImpl* self = nullptr;
    GenReservoirsTask(CPUDirectLightingImpl* a)
      : self(a) {
    }
    void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
      self->gen_reservoirs(begin, end, thread_id);
    }
  } gen_reservoirs_task = {this};

  void gen_reservoirs(uint32_t begin, uint32_t end, uint32_t thread_id) {
    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint32_t x = i % current_dimensions.x;
      uint32_t y = i / current_dimensions.x;

      const auto& scene = rt.scene();
      auto payload = DirectPayload::make(scene, {x, y}, current_dimensions, iteration);

      restir::Reservoir reservoir = {};
      Intersection intersection = {};
      if (rt.trace(scene, payload.ray, intersection, payload.smp)) {
        if (options.ris_sampling) {
          reservoir = restir::generate_samples(payload.spect, payload.smp, intersection, scene, options.ris_bsdf);
        }
      }
      reservoirs[i] = reservoir;
      intersections[i] = intersection;
    }
  }

  void spatial_resample(restir::Reservoir& r0, uint32_t x, uint32_t y,  //
    const SpectralQuery& spect, Sampler& smp, const Intersection& intersection, const Scene& scene, bool use_bsdf) {
    constexpr uint32_t kSamples = 32u;
    constexpr float kRadius = 8.0f;
    for (uint32_t i = 0; i < kSamples; ++i) {
      float r = sqrtf(smp.next()) * kRadius;
      float a = kDoublePi * smp.next();

      uint2 offset = {
        uint32_t(float(x) + r * cosf(a)),
        uint32_t(float(y) + r * sinf(a)),
      };

      if ((offset.x == x) || (offset.x >= current_dimensions.x) || (offset.y == y) || (offset.y >= current_dimensions.y))
        continue;

      uint32_t px = offset.x + offset.y * current_dimensions.x;
      const auto& r1 = reservoirs[px];
      r0 = restir::resample_reservoir(r0, r1, spect, smp, intersection, scene, use_bsdf);
    }
  }

  void final_gather(uint32_t begin, uint32_t end, uint32_t thread_id) {
    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint32_t x = i % current_dimensions.x;
      uint32_t y = i / current_dimensions.x;

      const auto& scene = rt.scene();
      auto payload = DirectPayload::make(scene, {x, y}, current_dimensions, iteration);

      const auto& intersection = intersections[i];
      if (intersection.t != kMaxFloat) {
        const auto& tri = scene.triangles[intersection.triangle_index];
        const auto& mat = scene.materials[intersection.material_index];

        EmitterSample emitter_sample = {};

        if (options.ris_sampling) {
          restir::Reservoir reservoir = reservoirs[i];
          if (options.ris_spatial) {
            spatial_resample(reservoir, x, y, payload.spect, payload.smp, intersection, scene, options.ris_bsdf);
          }
          emitter_sample = reservoir.sample;
          emitter_sample.pdf_sample = 1.0f / reservoir.weight();
        } else {
          uint32_t emitter_index = sample_emitter_index(scene, payload.smp);
          emitter_sample = sample_emitter(payload.spect, emitter_index, payload.smp, intersection.pos, scene);
        }

        if (emitter_sample.pdf_dir > 0) {
          BSDFEval bsdf_eval = bsdf::evaluate({payload.spect, payload.medium, PathSource::Camera, intersection, payload.ray.d}, emitter_sample.direction, mat, scene, payload.smp);
          if (bsdf_eval.valid()) {
            auto pos = shading_pos(scene.vertices, tri, intersection.barycentric, emitter_sample.direction);
            auto tr = rt.trace_transmittance(payload.spect, scene, pos, emitter_sample.origin, payload.medium, payload.smp);
            payload.accumulated += bsdf_eval.bsdf * emitter_sample.value * tr / (emitter_sample.pdf_dir * emitter_sample.pdf_sample);
            ETX_VALIDATE(payload.accumulated);
          }
        }

        if (intersection.emitter_index != kInvalidIndex) {
          const auto& emitter = scene.emitters[intersection.emitter_index];
          float pdf_a = 0.0f;
          float pdf_do = 0.0f;
          float pdf_emitter_dir = 0.0f;
          auto e = emitter_get_radiance(emitter, payload.spect, intersection.tex, payload.ray.o, intersection.pos, pdf_a, pdf_emitter_dir, pdf_do, scene, true);
          if (pdf_emitter_dir > 0.0f) {
            auto tr = rt.trace_transmittance(payload.spect, scene, payload.ray.o, intersection.pos, payload.medium, payload.smp);
            payload.accumulated += e * tr;
            ETX_VALIDATE(payload.accumulated);
          }
        }
      }

      auto xyz = (payload.accumulated / spectrum::sample_pdf()).to_xyz();
      ETX_VALIDATE(xyz);
      camera_image.accumulate({xyz.x, xyz.y, xyz.z, 1.0f}, payload.uv, float(iteration) / float(iteration + 1));
    }
  }
};

CPUDirectLighting::CPUDirectLighting(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUDirectLighting, rt, &current_state);
}

CPUDirectLighting::~CPUDirectLighting() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }

  ETX_PIMPL_CLEANUP(CPUDirectLighting);
}

void CPUDirectLighting::set_output_size(const uint2& dim) {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  _private->camera_image.resize(dim, 1);
}

const float4* CPUDirectLighting::get_camera_image(bool force_update) {
  return _private->camera_image.data();
}

const float4* CPUDirectLighting::get_light_image(bool force_update) {
  return nullptr;
}

const char* CPUDirectLighting::status() const {
  return _private->status;
}

void CPUDirectLighting::preview(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Preview;
    _private->start(opt);
  }
}

void CPUDirectLighting::run(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->start(opt);
  }
}

void CPUDirectLighting::update() {
  if (current_state == State::Stopped)
    return;

  if (rt.scheduler().completed(_private->current_task) == false)
    return;

  rt.scheduler().wait(_private->current_task);
  _private->current_task = {};

  switch (_private->operation) {
    case CPUDirectLightingImpl::Operation::GenReservoirs: {
      _private->start_final_gather();
      break;
    }

    case CPUDirectLightingImpl::Operation::FinalGather: {
      if ((current_state == State::WaitingForCompletion) || (_private->iteration + 1 >= _private->options.iterations)) {
        if (current_state == State::Preview) {
          _private->operation = CPUDirectLightingImpl::Operation::None;
          snprintf(_private->status, sizeof(_private->status), "[%u] Preview completed", _private->iteration);
          current_state = Integrator::State::Preview;
        } else {
          _private->operation = CPUDirectLightingImpl::Operation::None;
          snprintf(_private->status, sizeof(_private->status), "[%u] Completed in %.2f seconds", _private->iteration, _private->total_time.measure());
          current_state = Integrator::State::Stopped;
        }
      } else {
        snprintf(_private->status, sizeof(_private->status), "[%u] Running... (%.3fms per iteration)", _private->iteration, _private->iteration_time.measure_ms());
        _private->iteration_time = {};
        _private->iteration += 1;
        _private->start_generating_reservoirs();
      }
      break;
    }

    default:
      break;
  }
}

void CPUDirectLighting::stop(Stop st) {
  if (current_state == State::Stopped) {
    return;
  }

  if (st == Stop::WaitForCompletion) {
    current_state = State::WaitingForCompletion;
    snprintf(_private->status, sizeof(_private->status), "[%u] Waiting for completion", _private->iteration);
  } else {
    current_state = State::Stopped;
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};
    snprintf(_private->status, sizeof(_private->status), "[%u] Stopped", _private->iteration);
  }
}

Options CPUDirectLighting::options() const {
  Options result = {};
  result.add(1u, _private->options.iterations, 0xffffu, "spp", "Samples per Pixel");
  result.add(_private->options.ris_sampling, "ris", "RIS sampling");
  result.add(_private->options.ris_bsdf, "ris-bsdf", "BSDF in RIS sampling");
  result.add(_private->options.ris_spatial, "ris-spat", "Spatial Resampling");
  return result;
}

void CPUDirectLighting::update_options(const Options& opt) {
  if (current_state == State::Preview) {
    preview(opt);
  }
}

}  // namespace etx
