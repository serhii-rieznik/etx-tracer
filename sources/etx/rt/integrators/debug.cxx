#include <etx/core/core.hxx>
#include <etx/log/log.hxx>

#include <etx/render/host/rnd_sampler.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/shared/base.hxx>

#include <etx/rt/integrators/debug.hxx>

namespace etx {

struct CPUDebugIntegratorImpl : public Task {
  char status[2048] = {};
  Raytracing& rt;
  std::atomic<Integrator::State>* state = nullptr;
  std::vector<RNDSampler> samplers;
  Film camera_image;
  uint2 current_dimensions = {};
  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Task::Handle current_task = {};
  uint32_t iteration = 0u;
  uint32_t max_iterations = 32u;
  uint32_t current_scale = 1u;
  uint32_t preview_frames = 3u;
  CPUDebugIntegrator::Mode mode = CPUDebugIntegrator::Mode::Geometry;

  CPUDebugIntegratorImpl(Raytracing& a_rt, std::atomic<Integrator::State>* st)
    : rt(a_rt)
    , samplers(rt.scheduler().max_thread_count())
    , state(st) {
  }

  void start(const Options& opt) {
    mode = opt.get("mode", uint32_t(mode)).to_enum<CPUDebugIntegrator::Mode>();

    iteration = 0;
    snprintf(status, sizeof(status), "[%u] %s ...", iteration, (state->load() == Integrator::State::Running ? "Running" : "Preview"));

    current_scale = (state->load() == Integrator::State::Running) ? 1u : max(1u, uint32_t(exp2(preview_frames)));
    current_dimensions = camera_image.dimensions() / current_scale;

    total_time = {};
    iteration_time = {};
    current_task = rt.scheduler().schedule(this, current_dimensions.x * current_dimensions.y);
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    auto& smp = samplers[thread_id];
    auto mode = state->load();
    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      uint32_t x = i % current_dimensions.x;
      uint32_t y = i / current_dimensions.x;
      float2 uv = get_jittered_uv(smp, {x, y}, current_dimensions);
      float4 xyz = {preview_pixel(smp, uv), 1.0f};

      if (state->load() == Integrator::State::Running) {
        camera_image.accumulate(xyz, uv, float(iteration) / (float(iteration + 1)));
      } else {
        float t = iteration < preview_frames ? 0.0f : float(iteration - preview_frames) / float(iteration - preview_frames + 1);
        for (uint32_t ay = 0; ay < current_scale; ++ay) {
          for (uint32_t ax = 0; ax < current_scale; ++ax) {
            uint32_t rx = x * current_scale + ax;
            uint32_t ry = y * current_scale + ay;
            camera_image.accumulate(xyz, rx, ry, t);
          }
        }
      }
    }
  }

  float3 preview_pixel(RNDSampler& smp, const float2& uv) {
    const auto& scene = rt.scene();
    auto ray = generate_ray(smp, scene, uv);
    auto spect = spectrum::sample(smp.next());

    float3 xyz = {0.1f, 0.1f, 0.1f};

    Intersection intersection;
    if (rt.trace(ray, intersection, smp)) {
      bool entering_material = dot(ray.d, intersection.nrm) < 0.0f;

      switch (mode) {
        case CPUDebugIntegrator::Mode::Barycentrics: {
          xyz = spectrum::rgb_to_xyz(intersection.barycentric);
          break;
        }
        case CPUDebugIntegrator::Mode::Normals: {
          xyz = spectrum::rgb_to_xyz(intersection.nrm * 0.5f + 0.5f);
          break;
        }
        case CPUDebugIntegrator::Mode::Tangents: {
          xyz = spectrum::rgb_to_xyz(intersection.tan * 0.5f + 0.5f);
          break;
        }
        case CPUDebugIntegrator::Mode::Bitangents: {
          xyz = spectrum::rgb_to_xyz(intersection.btn * 0.5f + 0.5f);
          break;
        }
        case CPUDebugIntegrator::Mode::TexCoords: {
          xyz = spectrum::rgb_to_xyz({intersection.tex, 0.0f});
          break;
        }
        case CPUDebugIntegrator::Mode::FaceOrientation: {
          float d = fabsf(dot(intersection.nrm, ray.d));
          xyz = spectrum::rgb_to_xyz(d * (entering_material ? float3{0.2f, 0.2f, 1.0f} : float3{1.0f, 0.2f, 0.2f}));
          break;
        };
        case CPUDebugIntegrator::Mode::DiffuseColors: {
          const auto& tri = scene.triangles[intersection.triangle_index];
          const auto& mat = scene.materials[tri.material_index];
          xyz = mat.diffuse(spect).to_xyz();
          break;
        };
        case CPUDebugIntegrator::Mode::Fresnel: {
          const auto& tri = scene.triangles[intersection.triangle_index];
          const auto& mat = scene.materials[tri.material_index];
          SpectralResponse fr = {};
          switch (mat.cls) {
            case Material::Class::Conductor: {
              fr = fresnel::conductor(spect, ray.d, intersection.nrm, mat.ext_ior(spect), mat.int_ior(spect));
              break;
            }
            case Material::Class::Thinfilm: {
              float thickness = spectrum::kLongestWavelength;
              if (mat.thinfilm.image_index != kInvalidIndex) {
                const auto& img = scene.images[mat.thinfilm.image_index];
                auto t = img.evaluate(intersection.tex);
                thickness = lerp(mat.thinfilm.min_thickness, mat.thinfilm.max_thickness, t.x);
              }
              auto eta_ext = mat.ext_ior(spect).eta.monochromatic();
              auto eta_int = mat.int_ior(spect).eta.monochromatic();
              fr = fresnel::dielectric_thinfilm(spect, ray.d, intersection.nrm, eta_ext, eta_int, eta_ext, thickness);
              break;
            }
            default: {
              auto eta_i = (entering_material ? mat.ext_ior : mat.int_ior)(spect).eta.monochromatic();
              auto eta_o = (entering_material ? mat.int_ior : mat.ext_ior)(spect).eta.monochromatic();
              fr = fresnel::dielectric(spect, ray.d, intersection.nrm, eta_i, eta_o);
            }
          }
          xyz = fr.to_xyz();
          break;
        };
        default: {
          float d = fabsf(dot(intersection.nrm, ray.d));
          xyz = spectrum::rgb_to_xyz({d, d, d});
        }
      }
    }
    return xyz;
  }
};

CPUDebugIntegrator::CPUDebugIntegrator(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUDebugIntegrator, rt, &current_state);
}

CPUDebugIntegrator::~CPUDebugIntegrator() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  ETX_PIMPL_CLEANUP(CPUDebugIntegrator);
}

void CPUDebugIntegrator::set_output_size(const uint2& dim) {
  _private->camera_image.resize(dim, 1);
}

const float4* CPUDebugIntegrator::get_camera_image(bool) {
  return _private->camera_image.data();
}

const float4* CPUDebugIntegrator::get_light_image(bool) {
  return nullptr;
}

const char* CPUDebugIntegrator::status() const {
  return _private->status;
}

void CPUDebugIntegrator::preview(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Preview;
    _private->start(opt);
  }
}

void CPUDebugIntegrator::run(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->start(opt);
  }
}

void CPUDebugIntegrator::update() {
  bool should_stop = (current_state != State::Stopped) || (current_state == State::WaitingForCompletion);

  if (should_stop && rt.scheduler().completed(_private->current_task)) {
    if ((current_state == State::WaitingForCompletion) || (_private->iteration >= _private->max_iterations)) {
      rt.scheduler().wait(_private->current_task);
      _private->current_task = {};
      if (current_state == State::Preview) {
        snprintf(_private->status, sizeof(_private->status), "[%u] Preview completed", _private->iteration);
        current_state = Integrator::State::Preview;
      } else {
        snprintf(_private->status, sizeof(_private->status), "[%u] Completed in %.2f seconds", _private->iteration, _private->total_time.measure());
        current_state = Integrator::State::Stopped;
      }
    } else {
      snprintf(_private->status, sizeof(_private->status), "[%u] %s... (%.3fms per iteration)", _private->iteration,
        (current_state == Integrator::State::Running ? "Running" : "Preview"), _private->iteration_time.measure_ms());
      _private->iteration_time = {};
      _private->iteration += 1;

      _private->current_scale = (current_state == Integrator::State::Running) ? 1u : max(1u, uint32_t(exp2(_private->preview_frames - _private->iteration)));
      _private->current_dimensions = _private->camera_image.dimensions() / _private->current_scale;

      rt.scheduler().restart(_private->current_task, _private->current_dimensions.x * _private->current_dimensions.y);
    }
  }
}

void CPUDebugIntegrator::stop(Stop st) {
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

Options CPUDebugIntegrator::options() const {
  Options result = {};
  result.set(_private->mode, Mode::Count, &CPUDebugIntegrator::mode_to_string, "mode", "Visualize");
  return result;
}

std::string CPUDebugIntegrator::mode_to_string(uint32_t i) {
  switch (Mode(i)) {
    case Mode::Geometry:
      return "Geometry";
    case Mode::Barycentrics:
      return "Barycentrics";
    case Mode::Normals:
      return "Normals";
    case Mode::Tangents:
      return "Tangents";
    case Mode::Bitangents:
      return "Bitangents";
    case Mode::TexCoords:
      return "Texture Coordinates";
    case Mode::FaceOrientation:
      return "Face Orientation";
    case Mode::DiffuseColors:
      return "Diffuse Colors";
    case Mode::Fresnel:
      return "Fresnel Coefficients";
    default:
      return "???";
  }
}

void CPUDebugIntegrator::update_options(const Options& opt) {
  if (current_state == State::Preview) {
    preview(opt);
  }
}

}  // namespace etx
