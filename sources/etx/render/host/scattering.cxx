#include <etx/core/log.hxx>
#include <etx/render/shared/scattering.hxx>

#include <chrono>
#include <stb_image_write.hxx>

namespace etx {

constexpr uint32_t kExtinctionImageWidth = 1024u;
constexpr uint32_t kExtinctionImageHeight = 1024u;
constexpr uint32_t kExtinctionImageSize = kExtinctionImageWidth * kExtinctionImageHeight;

constexpr const float kPlanetRadius = 6371e+3f;
constexpr const float kAtmosphereRadius = 120e+3f;
constexpr const float kOuterSphereSize = kPlanetRadius + kAtmosphereRadius;
constexpr const float kDeltaDensity = 0.01f;
constexpr const float kRayleighDensityScale = 7994.0f;
constexpr const float kMieDensityScale = 1200.0f;

namespace scattering {

float rayleigh(float l) {
  l /= 100.0f;
  float l2 = l * l;
  float l4 = l2 * l2;
  return 1.169939f / (l4 * 100.0f);
}

float mie(float l) {
  constexpr float scale = 0.3954608f * (kPi * kPi * kPi);
  return scale / (l * l);
}

float ozone_absorbtion(float l) {
  const float na = 6.022140857f /* e+23f cancelled with base */;
  const float concentration = 41.58e-6f;
  float x = l;
  float x2 = x * x;
  float x3 = x2 * x;
  float x4 = x2 * x2;
  float x5 = x4 * x;
  float x6 = x4 * x2;
  float base = -1.109902e-15f * x6 + 3.950001e-12f * x5 - 5.784719e-09f * x4 + 4.460262e-06f * x3 - 1.909367e-03f * x2 + 4.303677e-01f * x - 3.992226e+01f;
  return (base > 0.0f) ? base * na * concentration : 0.0f;
}

float3 density(float height_above_surface) {
  float h = fmaxf(0.0f, height_above_surface);
  float x = h / 1000.0f;
  float x2 = x * x;
  float x3 = x2 * x;
  float x4 = x2 * x2;
  float x5 = x4 * x;
  float x6 = x3 * x3;
  float f = 3.759384e-08f * x6 - 1.067250e-05f * x5 + 1.080311e-03f * x4 - 4.851181e-02f * x3 + 9.185432e-01f * x2 - 4.886021e+00f * x + 7.900478e+00f;
  constexpr float kOzoneScale = 1.0f / 30.8491249f;
  return {
    expf(-h / kRayleighDensityScale),
    expf(-h / kMieDensityScale),
    fmaxf(0.0f, f * kOzoneScale),
  };
}

float2 precomputed_params_to_uv(const float2& params) {
  float u = sqr(params.x * 0.5f + 0.5f);
  float v = sqrtf(saturate(params.y / kAtmosphereRadius));
  return {u, v};
}

float2 uv_to_precomputed_params(const float2& uv) {
  float h = sqr(uv.y) * kAtmosphereRadius;
  float n_dot_l = sqrtf(uv.x) * 2.0f - 1.0f;
  return {n_dot_l, h};
}

float phase_rayleigh(float l_dot_v) {
  return (3.0f / 4.0f) * (1.0f + l_dot_v * l_dot_v) * (1.0f / kDoublePi);
}

float phase_mie(float l_dot_v, float g) {
  return (3.0f / 2.0f) * ((1.0f - g * g) * (1.0f + l_dot_v * l_dot_v)) / ((2.0f + g * g) * powf(1.0f + g * g - 2.0f * g * l_dot_v, 1.5f)) * (1.0f / kDoublePi);
}

float3 sample_optical_length(const float3& pos, const float3& light_direction, const Image& img) {
  float height = length(pos);
  float n_dot_l = dot(pos / height, light_direction);
  float2 uv = precomputed_params_to_uv({n_dot_l, height - kPlanetRadius});
  float4 e = img.evaluate(uv);
  ETX_VALIDATE(e);
  return {e.x, e.y, e.z};
}

float calculate_step_size(float current_distance, float total_distance, const float3 origin, const float3& direction, const float3& d0) {
  float3 grad = density(length(origin + direction * (1.0f + current_distance)) - kPlanetRadius) - d0;
  float l0 = logf((1.0f + grad.x) / kDeltaDensity) * kRayleighDensityScale;
  float l1 = logf((1.0f + grad.y) / kDeltaDensity) * kMieDensityScale;
  float calculated = sqrtf(kDeltaDensity * (l0 * l0 + l1 * l1));
  return fminf(total_distance - current_distance, calculated);
}

float3 optical_length(const float3& origin, const float3& direction, float total_distance) {
  float3 result = {};
  float height_above_surface = length(origin) - kPlanetRadius;
  float3 d = density(height_above_surface);
  float3 p = origin;
  float t = 0.0f;

  uint32_t steps = 0;
  while (t < total_distance) {
    float dt = calculate_step_size(t, total_distance, origin, direction, d);
    float3 p = origin + direction * (t + 0.5f * dt);
    t += dt;
    height_above_surface = length(p) - kPlanetRadius;
    d = density(height_above_surface);
    result += dt * d;
  }

  return result;
}

void radiance_spectrum_at_direction(Pointer<Spectrums> spectrums, const Image& extinction, const float3& view_direction, const float3& light_direction,
  const Parameters& parameters, SpectralDistribution& result) {
  const float3 origin = {0.0f, kPlanetRadius + parameters.altitude, 0.0f};
  const float l_dot_v = dot(light_direction, view_direction);
  const float phase_r = phase_rayleigh(l_dot_v);
  const float phase_m = phase_mie(l_dot_v, parameters.anisotropy);
  float height_above_surface = length(origin) - kPlanetRadius;

  const float3 density_scale = {
    parameters.rayleigh_scale,
    parameters.mie_scale,
    parameters.ozone_scale,
  };

  float3 view_optical_path = {};
  float3 current_density = density(height_above_surface);

  for (uint32_t i = 0; i < result.count; ++i) {
    result.entries[i].power = 0;
  }

  float t = 0.0f;
  float to_space = distance_to_sphere(origin, view_direction, {}, kOuterSphereSize);
  float to_planet = distance_to_sphere(origin, view_direction, {}, kPlanetRadius);
  if (to_planet > 0.0f) {
    to_space = to_planet;
  }

  while (t < to_space) {
    float dt = calculate_step_size(t, to_space, origin, view_direction, current_density);
    float3 p = origin + view_direction * (t + 0.5f * dt);
    height_above_surface = length(p) - kPlanetRadius;
    t += dt;

    if (height_above_surface < -kRayleighDensityScale)
      break;

    current_density = density(height_above_surface);
    view_optical_path += dt * density_scale * current_density;

    float3 light_optical_path = density_scale * sample_optical_length(p, light_direction, extinction);
    float3 total_optical_path = view_optical_path + light_optical_path;

    for (uint32_t i = 0; i < result.count; ++i) {
      float r = spectrums->rayleigh.entries[i].power;
      float m = spectrums->mie.entries[i].power;
      float o = spectrums->ozone.entries[i].power;
      float tr = r * total_optical_path.x + m * total_optical_path.y + o * total_optical_path.z;
      float value = expf(-tr) * dt * (phase_r * r * density_scale.x * current_density.x + phase_m * m * density_scale.y * current_density.y);
      result.entries[i].power += value;
    }
  }
}

void extinction_spectrum_at_direction(Pointer<Spectrums> spectrums, const float3& view_direction, const float3& next_direction, const Parameters& parameters,
  SpectralDistribution& result) {
  const float3 origin = {0.0f, kPlanetRadius + parameters.altitude, 0.0f};
  float to_space = distance_to_sphere(origin, view_direction, {}, kOuterSphereSize);

  if (distance_to_sphere(origin, next_direction, {}, kPlanetRadius) > 0.0f) {
    result = spectrums->black;
    return;
  }

  const float3 density_scale = {
    parameters.rayleigh_scale,
    parameters.mie_scale,
    parameters.ozone_scale,
  };

  float3 view_optical_path = {};
  float height_above_surface = length(origin) - kPlanetRadius;
  float3 current_density = density(height_above_surface);

  float t = 0.0f;
  while (t < to_space) {
    float dt = calculate_step_size(t, to_space, origin, view_direction, current_density);
    float3 p = origin + view_direction * (t + 0.5f * dt);
    height_above_surface = length(p) - kPlanetRadius;
    t += dt;
    current_density = density(height_above_surface);
    view_optical_path += dt * density_scale * current_density;
  }

  for (uint32_t i = 0; i < result.count; ++i) {
    float r = spectrums->rayleigh.entries[i].power;
    float m = spectrums->mie.entries[i].power;
    float o = spectrums->ozone.entries[i].power;
    float tr = r * view_optical_path.x + m * view_optical_path.y + o * view_optical_path.z;
    result.entries[i].power = expf(-tr);
  }
}

void init(TaskScheduler& scheduler, Pointer<Spectrums> spectrums, Image& extinction) {
  constexpr uint32_t kSpectrumStepSize = 5u;

  log::info("Precomputing atmosphere spectrums and extinction image %u x %u...", kExtinctionImageWidth, kExtinctionImageHeight);

  auto t0 = std::chrono::steady_clock::now();
  uint32_t count = 0;
  float3 accum = {};
  for (uint32_t w = spectrum::ShortestWavelength; w <= spectrum::LongestWavelength; ++w) {
    uint32_t i = w - spectrum::ShortestWavelength;

    accum.x += scattering::rayleigh(float(w));
    accum.y += scattering::mie(float(w));
    accum.z += scattering::ozone_absorbtion(float(w));

    if (i % kSpectrumStepSize == 0) {
      spectrums->rayleigh.entries[count] = {float(w), accum.x / static_cast<float>(kSpectrumStepSize)};
      spectrums->mie.entries[count] = {float(w), accum.y / static_cast<float>(kSpectrumStepSize)};
      spectrums->ozone.entries[count] = {float(w), accum.z / static_cast<float>(kSpectrumStepSize)};
      spectrums->black.entries[count] = {float(w), 0.0f};
      accum = {};
      ++count;
    }
  }

  spectrums->rayleigh.count = count;
  spectrums->mie.count = count;
  spectrums->ozone.count = count;
  spectrums->black.count = count;

  extinction = {};
}

void generate_sky_image(const Parameters& parameters, const uint2& dimensions, const float3& light_direction, Image& extinction, float4* buffer, TaskScheduler& scheduler) {
  if (extinction.data_size == 0) {
    auto t1 = std::chrono::steady_clock::now();
    log::info("Precomputing extinction image...");

    extinction = {};
    extinction.format = Image::Format::RGBA32F;
    extinction.pixels.f32 = make_array_view<float4>(calloc(kExtinctionImageSize, sizeof(float4)), kExtinctionImageSize);
    extinction.isize = {kExtinctionImageWidth, kExtinctionImageHeight};
    extinction.fsize = {float(kExtinctionImageWidth), float(kExtinctionImageHeight)};
    extinction.data_size = static_cast<uint32_t>(sizeof(float4) * kExtinctionImageSize);
    float4* image = extinction.pixels.f32.a;
    scheduler.execute(kExtinctionImageSize, [image](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        uint32_t x = i % kExtinctionImageWidth;
        uint32_t y = i / kExtinctionImageWidth;
        float2 uv = {float(x) / float(kExtinctionImageWidth), float(y) / float(kExtinctionImageHeight)};
        float2 params = scattering::uv_to_precomputed_params(uv);
        float3 direction = {sqrtf(1.0f - params.x * params.x), params.x, 0.0f};
        float3 origin = {0.0f, kPlanetRadius + params.y, 0.0f};
        float total_distance = distance_to_sphere(origin, direction, {}, kOuterSphereSize);
        float3 value = scattering::optical_length(origin, direction, total_distance);
        ETX_VALIDATE(value);
        image[x + kExtinctionImageWidth * y] = {value.x, value.y, value.z, 0.0f};
      }
    });
    auto t2 = std::chrono::steady_clock::now();
    log::info("Precomputed extinction image: %.3f ms", (t2 - t1).count() / 1.0e+6);
    stbi_write_hdr(env().file_in_data("optical-len.hdr"), kExtinctionImageWidth, kExtinctionImageHeight, 4, &image->x);
  }

  log::info("Generating sky image %u x %u...", dimensions.x, dimensions.y);

  auto t0 = std::chrono::steady_clock::now();
  std::atomic<float> ax = {};
  std::atomic<float> ay = {};
  std::atomic<float> az = {};
  std::atomic<float> aw = {};
  scheduler.execute(dimensions.x * dimensions.y,
    [&parameters, &dimensions, light_direction, &extinction, buffer, &ax, &ay, &az, &aw](uint32_t begin, uint32_t end, uint32_t thread_id) {
      float3 avg = {};
      float w = 0.0f;
      SpectralDistribution radiance = spectrum::shared()->black;
      for (uint32_t i = begin; i < end; ++i) {
        uint32_t x = i % dimensions.x;
        uint32_t y = i / dimensions.x;
        float u = float(x + 0.5f) / float(dimensions.x) * 2.0f - 1.0f;
        float v = float(y + 0.5f) / float(dimensions.y) * 2.0f - 1.0f;
        float3 direction = from_spherical(u * kPi, v * kHalfPi);
        radiance_spectrum_at_direction(spectrum::shared(), extinction, direction, light_direction, parameters, radiance);
        float3 xyz = radiance.integrate_to_xyz();
        float3 rgb = max({}, spectrum::xyz_to_rgb(xyz));
        if (v > 0.0f) {
          // Poor man multiple scattering
          // Gather average color of the upper hemisphere
          // Weighted in the way that top pixels contribute more
          // Not physically correct, but looks nice
          float weight = sinf(v * kHalfPi);
          w += weight;
          avg += rgb * weight;
        }
        buffer[x + dimensions.x * (dimensions.y - y - 1u)] = {rgb.x, rgb.y, rgb.z, 1.0f};
      }
      ax += avg.x;
      ay += avg.y;
      az += avg.z;
      aw += w;
    });

  float3 average_color = float3{ax.load(), ay.load(), az.load()} / aw.load();

  scheduler.execute(dimensions.x * dimensions.y, [buffer, average_color](uint32_t begin, uint32_t end, uint32_t thread_id) {
    for (uint32_t i = begin; i < end; ++i) {
      buffer[i].x += kDoublePi * average_color.x * buffer[i].x + average_color.x;
      buffer[i].y += kDoublePi * average_color.y * buffer[i].y + average_color.y;
      buffer[i].z += kDoublePi * average_color.z * buffer[i].z + average_color.z;
    }
  });

  auto t1 = std::chrono::steady_clock::now();
  auto duration = (t1 - t0).count() / 1.0e+6;
  log::info("Sky image generated: %.3f ms (%.3f ms/pixel)", duration, duration / double(dimensions.x * dimensions.y));
  stbi_write_hdr(env().file_in_data("sky.hdr"), dimensions.x, dimensions.y, 4, &buffer->x);
}

void generate_sun_image(const Parameters& parameters, const uint2& dimensions, const float3& light_direction, const float angular_size, float4* buffer, TaskScheduler& scheduler) {
  auto t0 = std::chrono::steady_clock::now();

  log::info("Generating Sun image %u x %u...", dimensions.x, dimensions.y);

  auto basis = orthonormal_basis(light_direction);
  float tan_half_fov = tanf(0.5f * angular_size);
  float solid_angle = kDoublePi * (1.0f - cosf(0.5f * angular_size));

  scheduler.execute(dimensions.x * dimensions.y,
    [&parameters, &dimensions, &basis, light_direction, solid_angle, tan_half_fov, buffer](uint32_t begin, uint32_t end, uint32_t thread_id) {
      SpectralDistribution radiance = spectrum::shared()->black;
      for (uint32_t i = begin; i < end; ++i) {
        uint32_t x = i % dimensions.x;
        uint32_t y = i / dimensions.x;
        float u = float(x + 0.5f) / float(dimensions.x) * 2.0f - 1.0f;
        float v0 = float(y + 0.5f) / float(dimensions.y) * 2.0f - 1.0f;
        float v1 = float(y + 1.5f) / float(dimensions.y) * 2.0f - 1.0f;
        float3 d0 = normalize(tan_half_fov * (u * basis.u + v0 * basis.v) + light_direction);
        float3 d1 = normalize(tan_half_fov * (u * basis.u + v1 * basis.v) + light_direction);
        extinction_spectrum_at_direction(spectrum::shared(), d0, d1, parameters, radiance);
        float darkening = (1.0f - 0.6f * (1.0f - fmaxf(0.0f, 1.0f - (u * u + v0 * v0))));
        float3 xyz = darkening * radiance.integrate_to_xyz();
        float3 rgb = max({}, spectrum::xyz_to_rgb(xyz));
        buffer[x + dimensions.x * y] = {rgb.x, rgb.y, rgb.z, 1.0f};
      }
    });
  auto t1 = std::chrono::steady_clock::now();
  auto duration = (t1 - t0).count() / 1.0e+6;
  log::info("Sun image generated: %.3f ms (%.3f ms/pixel)", duration, duration / double(dimensions.x * dimensions.y));
  stbi_write_hdr(env().file_in_data("sun.hdr"), dimensions.x, dimensions.y, 4, &buffer->x);
}

}  // namespace scattering
}  // namespace etx
