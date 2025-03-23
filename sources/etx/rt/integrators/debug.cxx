#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>

#include <etx/render/host/rnd_sampler.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/shared/base.hxx>

#include <etx/rt/integrators/debug.hxx>

namespace etx {

struct CPUDebugIntegratorImpl : public Task {
  Integrator::Status status = {};

  Raytracing& rt;
  std::atomic<Integrator::State>* state = nullptr;
  TimeMeasure total_time = {};
  TimeMeasure iteration_time = {};
  Task::Handle current_task = {};
  uint32_t current_scale = 1u;
  CPUDebugIntegrator::Mode mode = CPUDebugIntegrator::Mode::Random;
  RefractiveIndex spd_base;
  RefractiveIndex spd_air;
  Thinfilm thinfilm;
  float voxel_data[8] = {-0.1f, -0.1f, -0.1f, -0.1f, +0.1f, +0.1f, +0.1f, +0.1f};
  float th_factor = 1.0f;
  float th_min = 0.0f;
  float th_max = 1.0f;
  float3 thinfilm_rgb = Thinfilm::kRGBWavelengths;
  float3 thinfilm_span = Thinfilm::kRGBWavelengthsSpan;
  uint32_t random_iteration = ~0u;
  uint32_t random_dimension = ~0u;

  bool thinfilm_spectral = true;

  CPUDebugIntegratorImpl(Raytracing& a_rt, std::atomic<Integrator::State>* st)
    : rt(a_rt)
    , state(st) {
    spd_air = RefractiveIndex::load_from_file(env().file_in_data("spectrum/air.spd"));
    spd_base = RefractiveIndex::load_from_file(env().file_in_data("spectrum/water.spd"));
    thinfilm.ior = RefractiveIndex::load_from_file(env().file_in_data("spectrum/glycerol.spd"));
    thinfilm.min_thickness = 100.0f;
    thinfilm.max_thickness = 850.0f;
  }

  void start(const Options& opt) {
    mode = opt.get("mode", uint32_t(mode)).to_enum<CPUDebugIntegrator::Mode>();
    th_factor = opt.get("t-factor", th_factor).to_float();
    th_min = opt.get("t-min", th_min).to_float();
    th_max = opt.get("t-max", th_max).to_float();

    thinfilm.max_thickness = opt.get("test_th", thinfilm.max_thickness).to_float();
    thinfilm_rgb = opt.get("thinfilm_rgb", thinfilm_rgb).to_float3();
    thinfilm_span = opt.get("thinfilm_span", thinfilm_span).to_float3();
    thinfilm_spectral = opt.get("thinfilm_spectral", thinfilm_spectral).to_bool();

    random_iteration = opt.get("rnd_smp", random_iteration).to_integer();
    random_dimension = opt.get("rnd_dim", random_dimension).to_integer();

    voxel_data[0] = opt.get("v000", voxel_data[0]).to_float();
    voxel_data[1] = opt.get("v001", voxel_data[1]).to_float();
    voxel_data[2] = opt.get("v100", voxel_data[2]).to_float();
    voxel_data[3] = opt.get("v101", voxel_data[3]).to_float();
    voxel_data[4] = opt.get("v010", voxel_data[4]).to_float();
    voxel_data[5] = opt.get("v011", voxel_data[5]).to_float();
    voxel_data[6] = opt.get("v110", voxel_data[6]).to_float();
    voxel_data[7] = opt.get("v111", voxel_data[7]).to_float();

    status = {};
    total_time = {};
    iteration_time = {};

    rt.film().clear({Film::Internal});
    current_task = rt.scheduler().schedule(rt.film().pixel_count(), this);
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) override {
    const auto& film = rt.film();
    RNDSampler smp = {};
    for (uint32_t i = begin; (state->load() != Integrator::State::Stopped) && (i < end); ++i) {
      smp.init(i, status.current_iteration);
      uint2 pixel = {};
      if (film.active_pixel(i, pixel)) {
        float2 uv = film.sample(rt.scene(), status.current_iteration == 0u ? PixelFilter::empty() : rt.scene().pixel_sampler, pixel, smp.next_2d());
        float3 xyz = preview_pixel(smp, uv, pixel, i);
        rt.film().accumulate(pixel, {{xyz, Film::CameraImage}});
      }
    }
  }

  bool intersect_plane(const float3& in_o, const float3& in_d, const float3& plane_n, const float3& plane_p, float& t) {
    float denom = dot(in_d, plane_n);
    if (denom >= -kEpsilon) {
      return false;
    }

    t = dot(plane_p - in_o, plane_n) / denom;
    return t > 0.0f;
  }

  bool intersect_bbox(const Ray& r, const float3& box_min, float& out_min, float& out_max) {
    float3 tbot = (box_min - r.o) / r.d;
    float3 ttop = (box_min - r.o + 2.0f) / r.d;
    float3 tmin = min(ttop, tbot);
    float3 tmax = max(ttop, tbot);
    out_min = fmaxf(fmaxf(0.0f, tmin.x), fmaxf(tmin.y, tmin.z));
    out_max = fminf(fminf(tmax.x, tmax.x), fminf(tmax.y, tmax.z));
    return (out_max > out_min) && (out_max >= 0.0f);
  }

  bool intersect_sphere(const float3& in_o, const float3& in_d, const float3& c, float r, float& result_t) {
    float3 e = in_o - c;
    float b = dot(in_d, e);
    float d = (b * b) - dot(e, e) + (r * r);
    if (d < 0.0f) {
      return false;
    }
    d = sqrtf(d);
    float a0 = -b - d;
    float a1 = -b + d;
    result_t = (a0 < 0.0f) ? ((a1 < 0.0f) ? 0.0f : a1) : a0;
    return true;
  }

  float cube_root(float t) {
    float s = powf(fabsf(t), 1.0f / 3.0f);
    return t >= 0.0f ? +s : -s;
  }

  uint32_t solve_linear(float a, float b, float roots[]) {
    if (a * a < kEpsilon)
      return 0;

    // at + b = 0
    roots[0] = -b / a;
    return 1;
  }

  uint32_t solve_quadratic(float a, float b, float c, float roots[]) {
    if (a * a <= kEpsilon) {
      return solve_linear(b, c, roots);
    }

    // at^2 + bt + c = 0
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) {
      return 0;
    }

    float sqrt_d = sqrtf(discriminant);
    roots[0] = (-b - sqrt_d) / (2.0f * a);
    roots[1] = (-b + sqrt_d) / (2.0f * a);
    return 2u;
  }

  uint32_t solve_cubic(float a, float b, float c, float d, float roots[]) {
    if (a * a < kEpsilon) {
      return solve_quadratic(b, c, d, roots);
    }

    // t^3 + pt + q = 0
    float p = (3.0f * a * c - b * b) / (3.0f * a * a);
    float q = (2.0f * b * b * b - 9.0f * a * b * c + 27.0f * a * a * d) / (27.0f * a * a * a);

    if (fabsf(p) <= kEpsilon) {
      roots[0] = cube_root(-q);
      return 1u;
    }

    if (fabsf(q) <= kEpsilon) {
      roots[0u] = 0.0f;
      if (p < 0.0f) {
        roots[1u] = sqrtf(-p);
        roots[2u] = -sqrtf(-p);
        return 3u;
      }
      return 1u;
    }

    float e = b / (3.0f * a);
    float Q = q * q / 4.0f + p * p * p / 27.0f;
    if (fabsf(Q) <= kEpsilon) {
      // discriminant = 0 -> two roots
      roots[0] = -1.5f * q / p - e;
      roots[1] = 3.0f * q / p - e;
      return 2u;
    }

    if (Q > 0.0f) {
      // discriminant > 0 -> only one real root
      float sqrt_Q = sqrtf(Q);
      roots[0] = cube_root(-q / 2.0f + sqrt_Q) + cube_root(-q / 2.0f - sqrt_Q) - e;
      return 1u;
    }

    // discriminant < 0 -> three roots
    float u = 2.0f * sqrtf(-p / 3.0f);
    float cos_phi = clamp(3.0f * q / p / u, -1.0f, 1.0f);
    float t = acosf(cos_phi) / 3.0f;
    float k = 2.0f * kPi / 3.0f;
    roots[0] = u * cosf(t) - e;
    roots[1] = u * cosf(t - k) - e;
    roots[2] = u * cosf(t - 2.0f * k) - e;
    return 3u;
  }

  bool intersect_voxel(const float3& in_o, const float3& in_d, const float voxel[8], float& result_t) {
    float s000 = voxel[0];
    float s001 = voxel[1];
    float s010 = voxel[2];
    float s011 = voxel[3];
    float s100 = voxel[4];
    float s101 = voxel[5];
    float s110 = voxel[6];
    float s111 = voxel[7];
    float ka = s101 - s001;
    float k0 = s000;
    float k1 = s100 - s000;
    float k2 = s010 - s000;
    float k3 = s110 - s010 - k1;
    float k4 = k0 - s001;
    float k5 = k1 - ka;
    float k6 = k2 - (s011 - s001);
    float k7 = k3 - (s111 - s011 - ka);
    float ox = in_o.x;
    float oy = in_o.y;
    float oz = in_o.z;
    float dx = in_d.x;
    float dy = in_d.y;
    float dz = in_d.z;
    float m0 = ox * oy;
    float m1 = dx * dy;
    float m2 = ox * dy + oy * dx;
    float m3 = k5 * oz - k1;
    float m4 = k6 * oz - k2;
    float m5 = k7 * oz - k3;
    float a = k7 * m1 * dz;
    float b = m1 * m5 + dz * (k5 * dx + k6 * dy + k7 * m2);
    float c = dx * m3 + dy * m4 + m2 * m5 + dz * (k4 + k5 * ox + k6 * oy + k7 * m0);
    float d = (k4 * oz - k0) + ox * m3 + oy * m4 + m0 * m5;
    float roots[3] = {kMaxFloat, kMaxFloat, kMaxFloat};
    uint32_t root_count = solve_cubic(a, b, c, d, roots);
    float r0 = roots[0] < 0.0f ? kMaxFloat : roots[0];
    float r1 = roots[1] < 0.0f ? kMaxFloat : roots[1];
    float r2 = roots[2] < 0.0f ? kMaxFloat : roots[2];
    result_t = fminf(fminf(r0, r1), r2);
    return (root_count > 0) && (result_t > 0.0f) && (result_t != kMaxFloat);
  }

  float3 surface_normal(const float3& p, float voxel[8]) {
    float s000 = voxel[0];
    float s001 = voxel[1];
    float s010 = voxel[2];
    float s011 = voxel[3];
    float s100 = voxel[4];
    float s101 = voxel[5];
    float s110 = voxel[6];
    float s111 = voxel[7];

    float3 result = {};
    {
      float y0 = lerp(s100 - s000, s110 - s010, p.y);
      float y1 = lerp(s101 - s001, s111 - s011, p.y);
      result.x = lerp(y0, y1, p.z);
    }
    {
      float x0 = lerp(s010 - s000, s110 - s100, p.x);
      float x1 = lerp(s011 - s001, s111 - s101, p.x);
      result.y = lerp(x0, x1, p.z);
    }
    {
      float x0 = lerp(s001 - s000, s101 - s100, p.x);
      float x1 = lerp(s011 - s010, s111 - s110, p.x);
      result.z = lerp(x0, x1, p.y);
    }
    return normalize(result);
  }

  float3 preview_pixel(Sampler& smp, const float2& uv, const uint2& xy, const uint32_t pixel_index) const {
    const auto& scene = rt.scene();
    const auto& camera = rt.camera();
    auto& film = rt.film();
    auto spect = scene.spectral ? SpectralQuery::spectral_sample(smp.next()) : SpectralQuery::sample();

    float3 output = {};
    const float t = uv.x * 0.5f + 0.5f;
    const float s = uv.y * 0.5f + 0.5f;

    Intersection intersection;

    if (mode == CPUDebugIntegrator::Mode::Random) {
      if ((random_iteration == ~0u) && (random_dimension == ~0u)) {
        float t = smp.next();
        output = {t, t, t};
      } else {
        uint32_t it = random_iteration == ~0u ? status.current_iteration : random_iteration;
        uint32_t dim = random_dimension == ~0u ? 0 : min(16u, random_dimension);

        RNDSampler local_smp = {};
        local_smp.init(pixel_index, it);

        float2 t = {};
        uint32_t k = 0;
        do {
          t = local_smp.next_2d();
          ++k;
        } while (k < dim);
        output = {t.x, t.y, 0.0f};
      }
    } else if (mode == CPUDebugIntegrator::Mode::Thinfilm) {
      float cos_theta = 1.0f - t;
      float thickness = s * 2500.0f;

      float3 local_wl = {
        .x = thinfilm_rgb.x + thinfilm_span.x * (smp.next() * 2.0f - 1.0f),
        .y = thinfilm_rgb.y + thinfilm_span.y * (smp.next() * 2.0f - 1.0f),
        .z = thinfilm_rgb.z + thinfilm_span.z * (smp.next() * 2.0f - 1.0f),
      };

      SpectralQuery q_s = thinfilm_spectral ? SpectralQuery::spectral_sample(smp.next()) : SpectralQuery::sample();
      auto thinfilm_eval_s = evaluate_thinfilm(q_s, thinfilm, {}, scene, smp);
      thinfilm_eval_s.thickness = thickness;
      thinfilm_eval_s.rgb_wavelengths = local_wl;
      auto f_s = fresnel::calculate(q_s, cos_theta, spd_air(q_s), spd_base(q_s), thinfilm_eval_s);
      float3 xyz_s = (thinfilm_spectral ? SpectralDistribution::kRGBLuminanceScale : float3{1.0f, 1.0f, 1.0f}) * f_s.to_rgb() / q_s.sampling_pdf();
      output = xyz_s;
    } else if (mode == CPUDebugIntegrator::Mode::Spectrums) {
      constexpr float kBandCount = 9.0f;
      uint32_t band = static_cast<uint32_t>(clamp(kBandCount * (1.0f - s), 0.0f, kBandCount - 1.0f));

      static SpectralDistribution spds[uint32_t(kBandCount)];

      static bool init_spectrums = ([&](const Scene& scene) -> bool {
        spds[0] = SpectralDistribution::from_normalized_black_body(2700.0f, 1.0f);
        spds[1] = SpectralDistribution::from_normalized_black_body(4000.0f, 1.0f);
        spds[2] = SpectralDistribution::from_normalized_black_body(6500.0f, 1.0f);
        spds[3] = SpectralDistribution::from_normalized_black_body(12000.0f, 1.0f);
        spds[4] = SpectralDistribution::from_normalized_black_body(20000.0f, 1.0f);

        SpectralDistribution::load_from_file(env().file_in_data("spectrum/d65.spd"), spds[5], nullptr, false);

        spds[6] = SpectralDistribution::constant(0.5f);
        spds[7] = SpectralDistribution::rgb_reflectance({0.5, 0.5f, 0.5f});
        spds[8] = SpectralDistribution::rgb_luminance({0.5, 0.5f, 0.5f});

        for (uint32_t i = 0; i < 9; ++i) {
          spds[i].scale(1.0f / spds[i].luminance());
        }

        return true;
      })(scene);

      float3 value_rgb = {};
      float3 value_spectrum = {};

      if (t >= 0.5f) {
        SpectralQuery s_rgb = SpectralQuery::sample();
        value_rgb = (spds[band](s_rgb) / s_rgb.sampling_pdf()).to_rgb();
      } else {
        constexpr uint64_t kSampleCount = 1;  // spectrum::WavelengthCount;
        for (uint64_t i = 0; i < kSampleCount; ++i) {
          SpectralQuery s_s = SpectralQuery::spectral_sample(smp.next());
          auto r = spds[band](s_s);
          value_spectrum += r.to_rgb() / s_s.sampling_pdf();
        }
        value_spectrum *= 1.0f / float(kSampleCount);
      }

      output = (t < 0.5f ? value_spectrum : value_rgb);
    } else if (mode == CPUDebugIntegrator::Mode::IOR) {
      constexpr float kBandCount = 20.0f;
      uint32_t band = static_cast<uint32_t>(clamp(kBandCount * (1.0f - s), 0.0f, kBandCount - 1.0f));

      static RefractiveIndex spds[uint32_t(kBandCount) / 2u];

      static bool init_spectrums = ([&](const Scene& scene) -> bool {
        uint32_t i = 0;
        spds[i++] = RefractiveIndex::load_from_file(env().file_in_data("spectrum/water.spd"));
        spds[i++] = RefractiveIndex::load_from_file(env().file_in_data("spectrum/plastic.spd"));
        spds[i++] = RefractiveIndex::load_from_file(env().file_in_data("spectrum/sapphire.spd"));
        spds[i++] = RefractiveIndex::load_from_file(env().file_in_data("spectrum/diamond.spd"));
        spds[i++] = RefractiveIndex::load_from_file(env().file_in_data("spectrum/gold.spd"));
        spds[i++] = RefractiveIndex::load_from_file(env().file_in_data("spectrum/osmium.spd"));
        spds[i++] = RefractiveIndex::load_from_file(env().file_in_data("spectrum/copper.spd"));
        spds[i++] = RefractiveIndex::load_from_file(env().file_in_data("spectrum/chrome.spd"));
        spds[i++] = RefractiveIndex::load_from_file(env().file_in_data("spectrum/nickel.spd"));
        spds[i++] = RefractiveIndex::load_from_file(env().file_in_data("spectrum/silver.spd"));
        return true;
      })(scene);

      float cos_t = cosf(t * kHalfPi);

      SpectralQuery s_s = SpectralQuery::spectral_sample(smp.next());
      auto f_s = fresnel::calculate(s_s, cos_t, spds[0](s_s), spds[band / 2u](s_s), {});
      float3 value_spectrum = f_s.to_rgb() / s_s.sampling_pdf();

      SpectralQuery s_rgb = SpectralQuery::sample();
      auto f_rgb = fresnel::calculate(s_rgb, cos_t, spds[0](s_rgb), spds[band / 2u](s_rgb), {});
      float3 value_rgb = f_rgb.to_rgb() / s_rgb.sampling_pdf();

      output = (band % 2 == 0 ? value_spectrum : value_rgb);
    } else {
      auto ray = generate_ray(scene, camera, uv, smp.next_2d());
      if (rt.trace(scene, ray, intersection, smp)) {
        bool entering_material = dot(ray.d, intersection.nrm) < 0.0f;

        switch (mode) {
          case CPUDebugIntegrator::Mode::Barycentrics: {
            output = intersection.barycentric;
            break;
          }
          case CPUDebugIntegrator::Mode::Normals: {
            output = saturate(intersection.nrm * 0.5f + 0.5f);
            break;
          }
          case CPUDebugIntegrator::Mode::Tangents: {
            output = saturate(intersection.tan * 0.5f + 0.5f);
            break;
          }
          case CPUDebugIntegrator::Mode::Bitangents: {
            output = saturate(intersection.btn * 0.5f + 0.5f);
            break;
          }
          case CPUDebugIntegrator::Mode::TexCoords: {
            output = {intersection.tex.x, intersection.tex.y, 0.0f};
            break;
          }
          case CPUDebugIntegrator::Mode::FaceOrientation: {
            float d = fabsf(dot(intersection.nrm, ray.d));
            output = d * (entering_material ? float3{0.2f, 0.2f, 1.0f} : float3{1.0f, 0.2f, 0.2f});
            break;
          };
          case CPUDebugIntegrator::Mode::TransmittanceColor: {
            const auto& mat = scene.materials[intersection.material_index];
            output = apply_image(spect, mat.transmittance, intersection.tex, rt.scene(), nullptr).to_rgb();
            break;
          };
          case CPUDebugIntegrator::Mode::ReflectanceColor: {
            const auto& mat = scene.materials[intersection.material_index];
            output = apply_image(spect, mat.reflectance, intersection.tex, rt.scene(), nullptr).to_rgb();
            break;
          };
          case CPUDebugIntegrator::Mode::Fresnel: {
            const auto& mat = scene.materials[intersection.material_index];
            auto thinfilm = evaluate_thinfilm(spect, mat.thinfilm, intersection.tex, scene, smp);
            auto eta_i = (entering_material ? mat.ext_ior : mat.int_ior)(spect);
            auto eta_o = (entering_material ? mat.int_ior : mat.ext_ior)(spect);
            SpectralResponse fr = fresnel::calculate(spect, dot(ray.d, intersection.nrm), eta_i, eta_o, thinfilm);
            output = fr.to_rgb();
            break;
          };

          case CPUDebugIntegrator::Mode::Thickness: {
            auto remap_color = [this](float t) -> float3 {
              t = saturate((t - th_min) / (th_max - th_min));
              float3 result = {
                fmaxf(0.0f, cosf(t * kHalfPi)),
                fmaxf(0.0f, sinf(t * kPi)),
                fmaxf(0.0f, sinf(t * kHalfPi)),
              };
              ETX_VALIDATE(result);
              result = gamma_to_linear(result);
              ETX_VALIDATE(result);
              return result;
            };

            constexpr uint32_t kSampleCount = 64u;
            float distances[kSampleCount] = {};
            float average = 0.0f;
            uint32_t distance_count = 0;
            const float3 p0 = offset_ray(intersection.pos, -intersection.nrm);
            for (uint32_t i = 0; i < kSampleCount; ++i) {
              float3 d = -sample_cosine_distribution(smp.next_2d(), intersection.nrm, th_factor);
              Intersection e;
              Ray tr = {p0, d};
              tr.max_t = 2.0f * th_max;
              bool intersection_found = rt.trace(scene, tr, e, smp);
              average += intersection_found ? e.t : tr.max_t;
              distances[distance_count++] = intersection_found ? e.t : tr.max_t;
            }

            if (distance_count == 0) {
              output = {1.0f, 0.0f, 1.0f};
              break;
            }

            if (distance_count == 1) {
              output = remap_color(distances[0]);
              break;
            }

            average /= float(distance_count);

            float std_dev = 0.0f;
            for (uint32_t i = 0; i < distance_count; ++i) {
              std_dev += sqr(distances[i] - average);
            }
            std_dev = sqrtf(std_dev / distance_count);

            float thickness = 0.0f;
            float total_count = 0.0f;
            for (uint32_t i = 0; i < distance_count; ++i) {
              float dev = fabsf(distances[i] - average) / (std_dev + kEpsilon);
              if (dev <= 1.0f) {
                thickness += distances[i];
                total_count += 1.0f;
              }
            }

            float out_value = thickness / (total_count + kEpsilon);
            output = remap_color(out_value);
            break;
          }

          default: {
            float d = fabsf(dot(intersection.nrm, ray.d));
            output = {d, d, d};
          }
        }
      }
    }

    return output;
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

const Integrator::Status& CPUDebugIntegrator::status() const {
  return _private->status;
}

void CPUDebugIntegrator::run(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = State::Running;
    _private->start(opt);
  }
}

void CPUDebugIntegrator::update() {
  if ((current_state == State::Stopped) || (rt.scheduler().completed(_private->current_task) == false)) {
    return;
  }

  if (current_state == State::WaitingForCompletion) {
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};
    current_state = Integrator::State::Stopped;
  } else {
    _private->iteration_time = {};
    _private->status.completed_iterations = _private->status.current_iteration + 1;
    _private->status.current_iteration += 1;
    rt.scheduler().restart(_private->current_task);
  }
}

void CPUDebugIntegrator::stop(Stop st) {
  if (current_state == State::Stopped) {
    return;
  }

  if (st == Stop::WaitForCompletion) {
    current_state = State::WaitingForCompletion;
  } else {
    current_state = State::Stopped;
    rt.scheduler().wait(_private->current_task);
    _private->current_task = {};
  }
}

Options CPUDebugIntegrator::options() const {
  Options result = {};
  result.add(_private->mode, Mode::Count, &CPUDebugIntegrator::mode_to_string, "mode", "Visualize");

  if (_private->mode == Mode::Thickness) {
    result.add(0.0f, _private->th_factor, 1024.0f, "t-factor", "Thickness cone factor");
    result.add(0.0f, _private->th_min, 1024.0f, "t-min", "Min Thickness");
    result.add(0.0f, _private->th_max, 1024.0f, "t-max", "Max Thickness");
  } else if (_private->mode == Mode::Thinfilm) {
    result.add(_private->thinfilm_spectral, "thinfilm_spectral", "Spectral");
    float3 kMin = {spectrum::kShortestWavelength, spectrum::kShortestWavelength, spectrum::kShortestWavelength};
    float3 kMax = {spectrum::kLongestWavelength, spectrum::kLongestWavelength, spectrum::kLongestWavelength};
    float3 kMinSpan = {0.5f, 0.5f, 0.5f};
    float3 kMaxSpan = {spectrum::kWavelengthCount, spectrum::kWavelengthCount, spectrum::kWavelengthCount};
    result.add(kMin, _private->thinfilm_rgb, kMax, "thinfilm_rgb", "Wavelengths");
    result.add(kMinSpan, _private->thinfilm_span, kMaxSpan, "thinfilm_span", "Wavelengths Span");
    // result.add(0.0f, _private->thinfilm.max_thickness, 10000.0f, "test_th", "Thickness");
  }

  if (_private->mode == Mode::Random) {
    result.add(_private->random_iteration, "rnd_smp", "Sample");
    result.add(_private->random_dimension, "rnd_dim", "Dimension");
  }

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
    case Mode::TransmittanceColor:
      return "Transmittance Colors";
    case Mode::ReflectanceColor:
      return "Reflectance Colors";
    case Mode::Fresnel:
      return "Fresnel Coefficients";
    case Mode::Thickness:
      return "Thickness";
    case Mode::Thinfilm:
      return "Thinfilm";
    case Mode::Spectrums:
      return "Spectrums";
    case Mode::IOR:
      return "IOR";
    case Mode::Random:
      return "Random Sampler";
    default:
      return "???";
  }
}

void CPUDebugIntegrator::update_options(const Options& opt) {
  if (current_state == State::Running) {
    run(opt);
  }
}

}  // namespace etx
