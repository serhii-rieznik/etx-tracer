#include <etx/core/core.hxx>
#include <etx/log/log.hxx>

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/bsdf.hxx>
#include <etx/render/host/rnd_sampler.hxx>
#include <etx/render/host/film.hxx>

#include <etx/rt/integrators/atmosphere.hxx>

namespace etx {

constexpr float kMaxSteps = 256.0f;

inline float scattering_rayleigh(float l) {
  l /= 100.0f;
  float l2 = l * l;
  return 1.169939f / (l2 * l2 * 100.0f);
}

inline float scattering_mie(float l) {
  constexpr float scale = 0.3954608f * (kPi * kPi * kPi);
  return scale / (l * l);
}

inline float ozone_absorbtion(float l) {
  const float na = 6.022140857f /* e+23f cancelled with base */;
  const float concentration = 41.58f * 0.000001f;
  float x = l;
  float x2 = x * x;
  float x3 = x2 * x;
  float x4 = x2 * x2;
  float x5 = x4 * x;
  float x6 = x4 * x2;
  float base = max(0.0f, -1.109902e-15f * x6 + 3.950001e-12f * x5 - 5.784719e-09f * x4 + 4.460262e-06f * x3 - 1.909367e-03f * x2 + 4.303677e-01f * x - 3.992226e+01f);
  return base * na * concentration;
}

inline float ozone_vertical_profile(float h) {
  float x = h / 1000.0f;
  float x2 = x * x;
  float x3 = x2 * x;
  float x4 = x2 * x2;
  float x5 = x4 * x;
  float x6 = x3 * x3;
  float x7 = x6 * x;
  float f = 3.759384E-08f * x6 - 1.067250E-05f * x5 + 1.080311E-03f * x4 - 4.851181E-02f * x3 + 9.185432E-01f * x2 - 4.886021E+00f * x + 7.900478E+00f;
  const float n = 30.8491249f;
  return max(0.0f, f / n);
}

inline float3 density(float height_above_ground) {
  constexpr float density_h_r = 7994.0f;
  constexpr float density_h_m = 1200.0f;
  height_above_ground = max(0.0f, height_above_ground);
  return {expf(-height_above_ground / density_h_r), expf(-height_above_ground / density_h_m), ozone_vertical_profile(height_above_ground)};
}

inline constexpr float3 density_k() {
  constexpr float density_h_r = 7994.0f;
  constexpr float density_h_m = 1200.0f;
  return {-1.0f / density_h_r, -1.0f / density_h_m, 0.0f};
}

inline float3 optical_length(float3 p, const float3& target, Sampler& smp, float opt_step_scale) {
  constexpr auto d_k = density_k();
  const float delta_density = 0.025f;

  float3 result = {};

  float3 dp = (target - p);
  float total_distance = length(dp);
  dp /= total_distance;
  float min_step = total_distance / kMaxSteps;
  float t = 0.0f;
  while (t < total_distance) {
    float3 current_density = density(length(p) - kPlanetRadius);
    float r_step = -logf(delta_density / current_density.x + 1.0f) / d_k.x;
    float m_step = -logf(delta_density / current_density.y + 1.0f) / d_k.y;
    float dt = min(total_distance - t, min(r_step, m_step) * (1.0f + opt_step_scale * smp.next()));

    p += dp * dt;
    t += dt;

    result += dt * density(length(p) - kPlanetRadius);
  }
  return result;
}

inline float phase_rayleigh(float l_dot_v) {
  constexpr float p = 0.035f;
  constexpr float depolarisation = (6.0f + 3.0f * p) / (6.0f - 7.0f * p);
  return (3.0f / 4.0f) * (1.0f + l_dot_v * l_dot_v) * (1.0f / (4.0f * kPi));
}

inline float phase_mie(float l_dot_v, float g) {
  return (3.0f / 2.0f) * ((1.0f - g * g) * (1.0f + l_dot_v * l_dot_v)) / ((2.0f + g * g) * powf(1.0f + g * g - 2.0f * g * l_dot_v, 1.5f)) * (1.0f / (4.0f * kPi));
}

struct CPUAtmosphereImpl : public Task {
  Raytracing& rt;
  Film camera_image;
  std::vector<RNDSampler> samplers;
  char status[2048] = {};

  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Task::Handle current_task = {};
  uint32_t iteration = 0u;
  uint32_t opt_max_iterations = 4u;
  float opt_phase_function_g = 0.75f;
  float opt_step_scale = 10.0f;
  float opt_rayleigh = 1.0f;
  float opt_mie = 1.0f;
  float opt_ozone = 1.0f;
  float total_time_value = 0.0f;
  bool opt_render_sun = false;

  std::atomic<uint32_t> pixels_processed = {};
  std::atomic<Integrator::State>* state = nullptr;

  SpectralDistribution rayleigh = {};
  SpectralDistribution mie = {};
  SpectralDistribution ozone = {};

  CPUAtmosphereImpl(Raytracing& a_rt, std::atomic<Integrator::State>* st)
    : rt(a_rt)
    , samplers(rt.scheduler().max_thread_count())
    , state(st) {
  }

  void init_spectrums() {
    if (rayleigh.count > 0) {
      return;
    }

    constexpr float kSunTemperature = 5778.0f;
    float sun_norm_w = spectrum::black_body_radiation_maximum_wavelength(kSunTemperature);
    float sun_norm = 16.0f / spectrum::black_body_radiation(sun_norm_w, kSunTemperature);

    for (uint32_t w = spectrum::ShortestWavelength; w <= spectrum::LongestWavelength; ++w) {
      uint32_t i = w - spectrum::ShortestWavelength;
      rayleigh.entries[i] = {float(w), scattering_rayleigh(float(w))};
      mie.entries[i] = {float(w), scattering_mie(float(w))};
      ozone.entries[i] = {float(w), ozone_absorbtion(float(w))};
    }
    rayleigh.count = spectrum::WavelengthCount;
    mie.count = spectrum::WavelengthCount;
    ozone.count = spectrum::WavelengthCount;

    if constexpr (spectrum::kSpectralRendering == false) {
      auto xyz = rayleigh.integrate_to_xyz();
      rayleigh = rgb::make_reflectance_spd(spectrum::xyz_to_rgb(xyz), rt.scene().spectrums);
    }
    if constexpr (spectrum::kSpectralRendering == false) {
      auto xyz = mie.integrate_to_xyz();
      mie = rgb::make_reflectance_spd(spectrum::xyz_to_rgb(xyz), rt.scene().spectrums);
    }
    if constexpr (spectrum::kSpectralRendering == false) {
      auto xyz = ozone.integrate_to_xyz();
      ozone = rgb::make_reflectance_spd(spectrum::xyz_to_rgb(xyz), rt.scene().spectrums);
    }
  }

  bool running() const {
    return (state->load() == Integrator::State::Preview) || (state->load() == Integrator::State::Running);
  }

  void start(const Options& opt) {
    init_spectrums();

    opt_max_iterations = opt.get("spp", opt_max_iterations).to_integer();
    opt_phase_function_g = opt.get("g", opt_phase_function_g).to_float();
    opt_step_scale = opt.get("step", opt_step_scale).to_float();
    opt_render_sun = opt.get("sun", opt_render_sun).to_bool();
    opt_rayleigh = opt.get("r", opt_rayleigh).to_float();
    opt_mie = opt.get("m", opt_mie).to_float();
    opt_ozone = opt.get("o", opt_ozone).to_float();

    iteration = 0;
    snprintf(status, sizeof(status), "[%u] %s ...", iteration, (state->load() == Integrator::State::Running ? "Running" : "Preview"));

    total_time_value = 0.0f;
    total_time = {};
    iteration_time = {};
    pixels_processed = 0;
    current_task = rt.scheduler().schedule(this, camera_image.dimensions().x);
  }

  void execute_range(uint32_t x_begin, uint32_t x_end, uint32_t thread_id) override {
    auto& smp = samplers[thread_id];
    for (uint32_t x = x_begin; (state->load() != Integrator::State::Stopped) && (x < x_end); ++x) {
      for (uint32_t y = 0; y < camera_image.dimensions().y; ++y) {
        float2 uv = get_jittered_uv(smp, {x, y}, camera_image.dimensions());
        float3 xyz = trace_pixel(smp, uv);
        camera_image.accumulate({xyz.x, xyz.y, xyz.z, 1.0f}, uv, float(iteration) / float(iteration + 1));
        ++pixels_processed;
      }
    }
  }

  float3 trace_pixel(RNDSampler& smp, const float2& uv) {
    constexpr float kSphereSize = kPlanetRadius + kAtmosphereRadius;

    auto& scene = rt.scene();

    auto ray = generate_ray(smp, scene, uv);
    float to_space = distance_to_sphere(ray.o, ray.d, kSphereSize);

    float to_planet = distance_to_sphere(ray.o, ray.d, kPlanetRadius);
    if (to_planet > 0.0f) {
      to_space = to_planet;
    }

    const uint32_t entry_count = spectrum::kSpectralRendering ? spectrum::WavelengthCount : 3;

    SpectralDistribution result = {{}, entry_count};
    for (uint32_t i = 0; i < entry_count; ++i) {
      result.entries[i].wavelength = float(spectrum::ShortestWavelength + i);
    }

    float min_step = to_space / kMaxSteps;

    float3 position = ray.o;
    float3 total_optical_path = {};
    float t = 0.0f;

    constexpr auto d_k = density_k();
    const float delta_density = 0.025f;

    while (running() && (t < to_space)) {
      float3 current_density = density(length(position) - kPlanetRadius);
      float r_step = -logf(delta_density / current_density.x + 1.0f) / d_k.x;
      float m_step = -logf(delta_density / current_density.y + 1.0f) / d_k.y;
      float dt = min(to_space - t, min(r_step, m_step) * (1.0f + opt_step_scale * smp.next()));
      position += ray.d * dt;
      t += dt;

      float3 d = dt * density(length(position) - kPlanetRadius);

      for (uint32_t i = 0; i < scene.environment_emitters.count; ++i) {
        auto local_em = sample_emitter({0.5f * (spectrum::kShortestWavelength + spectrum::kLongestWavelength)}, i, smp, ray.o, scene);
        auto cos_t = dot(local_em.direction, ray.d);
        auto phase_r = phase_rayleigh(cos_t);
        auto phase_m = phase_mie(cos_t, opt_phase_function_g);
        auto distance_to_space = distance_to_sphere(position, local_em.direction, kSphereSize);
        auto optical_path = optical_length(position, position + distance_to_space * local_em.direction, smp, opt_step_scale);
        const auto& em = scene.emitters[local_em.emitter_index];
        float3 current_optical_path = total_optical_path + optical_path;

        if constexpr (spectrum::kSpectralRendering) {
          for (uint32_t s = 0; s < entry_count; ++s) {
            float e = em.emission.spectrum.query({float(spectrum::ShortestWavelength + s)}).components.x;
            auto value = expf(-current_optical_path.x * rayleigh.entries[s].power * opt_rayleigh -  //
                              current_optical_path.y * mie.entries[s].power * opt_mie -             //
                              current_optical_path.z * ozone.entries[s].power * opt_ozone);
            result.entries[s].power += e * value * (d.x * phase_r * rayleigh.entries[s].power * opt_rayleigh + d.y * phase_m * mie.entries[s].power * opt_mie);
            ETX_VALIDATE(result.entries[s].power);
          }
        } else {
          auto e = em.emission.spectrum.query({-1.0f});
          float accum[3] = {};
          for (uint32_t s = 0; s < entry_count; ++s) {
            auto value = expf(-current_optical_path.x * rayleigh.entries[s].power * opt_rayleigh  //
                              - current_optical_path.y * mie.entries[s].power * opt_mie           //
                              - current_optical_path.z * ozone.entries[s].power * opt_ozone);
            accum[s] = value * (d.x * phase_r * rayleigh.entries[s].power * opt_rayleigh + d.y * phase_m * mie.entries[s].power * opt_mie);
            ETX_VALIDATE(result.entries[s].power);
          }
          result.entries[0].power += e.components.x * accum[0];
          result.entries[1].power += e.components.y * accum[1];
          result.entries[2].power += e.components.z * accum[2];
        }

        total_optical_path += d;
      }
    }

    for (uint32_t i = 0; opt_render_sun && (to_planet <= 0.0f) && (i < scene.environment_emitters.count); ++i) {
      auto e_index = scene.environment_emitters.emitters[i];
      if (scene.emitters[e_index].angular_size > 0.0f) {
        float pdfs[3] = {};
        float scale = 1.0f / (kDoublePi * (1.0f - cosf(0.5f * scene.emitters[e_index].angular_size)));
        if constexpr (spectrum::kSpectralRendering) {
          for (uint32_t s = 0; s < entry_count; ++s) {
            auto em = emitter_get_radiance(scene.emitters[e_index], {float(spectrum::ShortestWavelength + s)}, ray.d, pdfs[0], pdfs[1], pdfs[2], scene);
            result.entries[s].power +=
              scale * em.components.x * expf(-total_optical_path.x * rayleigh.entries[s].power * opt_rayleigh - total_optical_path.y * mie.entries[s].power * opt_mie);
          }
        } else {
          auto em = emitter_get_radiance(scene.emitters[e_index], {-1.0f}, ray.d, pdfs[0], pdfs[1], pdfs[2], scene);
          float accum[3] = {};
          for (uint32_t s = 0; s < entry_count; ++s) {
            accum[s] = scale * expf(-total_optical_path.x * rayleigh.entries[s].power * opt_rayleigh - total_optical_path.y * mie.entries[s].power * opt_mie);
          }
          result.entries[0].power += accum[0] * em.components.x;
          result.entries[1].power += accum[1] * em.components.y;
          result.entries[2].power += accum[2] * em.components.z;
        }
      }
    }
    return result.to_xyz();
  }
};

CPUAtmosphere::CPUAtmosphere(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUAtmosphere, rt, &current_state);
}

CPUAtmosphere::~CPUAtmosphere() {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }

  ETX_PIMPL_CLEANUP(CPUAtmosphere);
}

void CPUAtmosphere::set_output_size(const uint2& dim) {
  if (current_state != State::Stopped) {
    stop(Stop::Immediate);
  }
  _private->camera_image.resize(dim, 1);
}

const float4* CPUAtmosphere::get_camera_image(bool force_update) {
  return _private->camera_image.data();
}

const float4* CPUAtmosphere::get_light_image(bool force_update) {
  return nullptr;
}

const char* CPUAtmosphere::status() const {
  return _private->status;
}

void CPUAtmosphere::preview(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Preview;
    _private->start(opt);
  }
}

void CPUAtmosphere::run(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->start(opt);
  }
}

void CPUAtmosphere::update() {
  bool should_stop = (current_state != State::Stopped) || (current_state == State::WaitingForCompletion);

  if (should_stop && rt.scheduler().completed(_private->current_task)) {
    if ((current_state == State::WaitingForCompletion) || (_private->iteration + 1 >= _private->opt_max_iterations)) {
      rt.scheduler().wait(_private->current_task);
      _private->current_task = {};
      if (current_state == State::Preview) {
        if (_private->total_time_value == 0.0f) {
          _private->total_time_value = float(_private->total_time.measure());
          snprintf(_private->status, sizeof(_private->status), "[%u] Preview completed in %.2f seconds", _private->iteration, _private->total_time_value);
        }
        current_state = Integrator::State::Preview;
      } else {
        snprintf(_private->status, sizeof(_private->status), "[%u] Completed in %.2f seconds", _private->iteration, _private->total_time.measure());
        current_state = Integrator::State::Stopped;
      }
    } else {
      snprintf(_private->status, sizeof(_private->status), "[%u] %s... (%.3fms per iteration)", _private->iteration,
        (current_state == Integrator::State::Running ? "Running" : "Preview"), _private->iteration_time.measure_ms());
      _private->iteration_time = {};
      _private->pixels_processed = 0;
      _private->iteration += 1;
      rt.scheduler().restart(_private->current_task, _private->camera_image.dimensions().x);
    }
  } else {
    float t = 100.0f * float(_private->pixels_processed.load()) / float(_private->camera_image.dimensions().x * _private->camera_image.dimensions().y);
    snprintf(_private->status, sizeof(_private->status), "[%u] %s... %.2f, %.3fms", _private->iteration, (current_state == Integrator::State::Running ? "Running" : "Preview"), t,
      _private->iteration_time.measure_ms());
  }
}

void CPUAtmosphere::stop(Stop st) {
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

Options CPUAtmosphere::options() const {
  Options result = {};
  result.add(1u, _private->opt_max_iterations, 0xffffu, "spp", "Samples per Pixel");
  result.add(-1.0f, _private->opt_phase_function_g, 1.0f, "g", "Asymmetry Factor");
  result.add(0.0f, _private->opt_step_scale, 100.0f, "step", "Step Scale");
  result.add(_private->opt_render_sun, "sun", "Render Sun");
  result.add(0.0f, _private->opt_rayleigh, 100.0f, "r", "Rayleigh Scattering");
  result.add(0.0f, _private->opt_mie, 100.0f, "m", "Mie Scattering");
  result.add(0.0f, _private->opt_ozone, 100.0f, "o", "Ozone");
  return result;
}

void CPUAtmosphere::update_options(const Options& opt) {
  if (current_state == State::Preview) {
    preview(opt);
  }
}

}  // namespace etx
