#include <etx/core/log.hxx>
#include <etx/core/environment.hxx>
#include <etx/render/shared/scattering.hxx>
#include <stb_image_write.hxx>

namespace etx {

constexpr const float kPlanetRadius = 6371e+3f;
constexpr const float kAtmosphereRadius = 120e+3f;
constexpr const float kOuterSphereSize = kPlanetRadius + kAtmosphereRadius;
constexpr const float kDeltaDensity = 0.01f;
constexpr const float kRayleighDensityScale = 7994.0f;
constexpr const float kMieDensityScale = 1200.0f;

namespace scattering {

namespace {

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

float ozone_absorption(float l) {
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

struct DensityPair {
  float rayleigh;
  float mie;
};

struct DensityAndDerivative {
  float3 density;
  float3 derivative;
};

struct TransmittanceTable {
  static constexpr uint32_t kDepthSamples = 8192u;
  static constexpr float kMaxOpticalDepth = 16.0f;
  static constexpr float kDepthStep = kMaxOpticalDepth / float(kDepthSamples - 1u);

  float data[kDepthSamples];

  TransmittanceTable() {
    for (uint32_t d = 0; d < kDepthSamples; ++d) {
      float depth = float(d) * kDepthStep;
      data[d] = expf(-depth);
    }
  }

  ETX_GPU_CODE float lookup(float optical_depth) const {
    if (optical_depth <= 0.0f) {
      return 1.0f;
    }
    if (optical_depth >= kMaxOpticalDepth) {
      return expf(-optical_depth);
    }

    float depth_index = optical_depth / kDepthStep;
    uint32_t index0 = static_cast<uint32_t>(depth_index);
    uint32_t index1 = min(index0 + 1u, kDepthSamples - 1u);
    float t = depth_index - float(index0);
    float v0 = data[index0];
    float v1 = data[index1];
    return v0 * (1.0f - t) + v1 * t;
  }
} g_transmittance_table;

struct DensityTable {
  static constexpr uint32_t kHeightSamples = 8192u;
  static constexpr float kMaxHeight = kAtmosphereRadius * 3.0f / 4.0f;
  static constexpr float kHeightStep = kMaxHeight / float(kHeightSamples - 1u);

  DensityPair data[kHeightSamples];

  DensityTable() {
    for (uint32_t h = 0; h < kHeightSamples; ++h) {
      float height = float(h) * kHeightStep;
      data[h].rayleigh = expf(-height / kRayleighDensityScale);
      data[h].mie = expf(-height / kMieDensityScale);
    }
  }

  ETX_GPU_CODE DensityPair lookup(float height) const {
    if (height <= 0.0f) {
      return {1.0f, 1.0f};
    }
    if (height >= kMaxHeight) {
      return {expf(-height / kRayleighDensityScale), expf(-height / kMieDensityScale)};
    }

    float height_index = height / kHeightStep;
    uint32_t index0 = static_cast<uint32_t>(height_index);
    uint32_t index1 = min(index0 + 1u, kHeightSamples - 1u);
    float t = height_index - float(index0);
    DensityPair v0 = data[index0];
    DensityPair v1 = data[index1];
    return {
      v0.rayleigh * (1.0f - t) + v1.rayleigh * t,
      v0.mie * (1.0f - t) + v1.mie * t,
    };
  }
} g_density_table;

DensityAndDerivative density_and_derivative(float height_above_surface) {
  float h = fmaxf(0.0f, height_above_surface);
  float x = h / 1000.0f;
  float x2 = x * x;
  float x3 = x2 * x;
  float x4 = x2 * x2;
  float x5 = x4 * x;
  float x6 = x3 * x3;

  DensityPair densities = g_density_table.lookup(h);
  float rayleigh_density = densities.rayleigh;
  float mie_density = densities.mie;
  float rayleigh_derivative = -rayleigh_density / kRayleighDensityScale;
  float mie_derivative = -mie_density / kMieDensityScale;

  float f = 3.759384e-08f * x6 - 1.067250e-05f * x5 + 1.080311e-03f * x4 - 4.851181e-02f * x3 + 9.185432e-01f * x2 - 4.886021e+00f * x + 7.900478e+00f;
  float df = 6.0f * 3.759384e-08f * x5 - 5.0f * 1.067250e-05f * x4 + 4.0f * 1.080311e-03f * x3 - 3.0f * 4.851181e-02f * x2 + 2.0f * 9.185432e-01f * x - 4.886021e+00f;

  constexpr float kOzoneScale = 1.0f / 30.8491249f;
  constexpr float dx_dh = 1.0f / 1000.0f;
  float ozone_density = fmaxf(0.0f, f * kOzoneScale);
  float ozone_derivative = fmaxf(0.0f, df) * dx_dh * kOzoneScale;

  return {
    {rayleigh_density, mie_density, ozone_density},
    {rayleigh_derivative, mie_derivative, ozone_derivative},
  };
}

float3 density(float height_above_surface) {
  return density_and_derivative(height_above_surface).density;
}

float3 density_derivative(float height_above_surface) {
  return density_and_derivative(height_above_surface).derivative;
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
  float temp = 1.0f + g * g - 2.0f * g * l_dot_v;
  return (3.0f / 2.0f) * ((1.0f - g * g) * (1.0f + l_dot_v * l_dot_v)) / ((2.0f + g * g) * temp * sqrtf(temp)) * (1.0f / kDoublePi);
}

float3 sample_optical_length(const float3& pos, const float3& light_direction, const OpticalDepthData& extinction) {
  float height = length(pos);
  float n_dot_l = dot(pos / height, light_direction);
  float2 uv = precomputed_params_to_uv({n_dot_l, height - kPlanetRadius});
  float4 e = extinction.evaluate(uv);
  ETX_VALIDATE(e);
  return {e.x, e.y, e.z};
}

float calculate_step_size(float current_distance, float total_distance, const float3 origin, const float3& direction) {
  float3 position = origin + direction * current_distance;
  float height = length(position) - kPlanetRadius;
  float3 directional_derivative = density_derivative(height);

  float3 normalized_position = position / length(position);
  directional_derivative = directional_derivative * dot(normalized_position, direction);

  float l0 = logf(fmaxf(kRayEpsilon, (1.0f + directional_derivative.x) / kDeltaDensity)) * kRayleighDensityScale;
  float l1 = logf(fmaxf(kRayEpsilon, (1.0f + directional_derivative.y) / kDeltaDensity)) * kMieDensityScale;
  float calculated = sqrtf(kDeltaDensity * (l0 * l0 + l1 * l1));
  return fminf(total_distance - current_distance, fmaxf(kRayEpsilon, calculated));
}

float3 optical_length(const float3& origin, const float3& direction, float total_distance) {
  float3 result = {};
  float height_above_surface = length(origin) - kPlanetRadius;
  float t = 0.0f;

  constexpr uint32_t kMaxSteps = 1u << 14u;
  uint32_t steps = 0;
  while ((t < total_distance) && (steps < kMaxSteps)) {
    float dt = calculate_step_size(t, total_distance, origin, direction);
    float3 p = origin + direction * (t + 0.5f * dt);
    t += dt;
    height_above_surface = length(p) - kPlanetRadius;
    result += dt * density(height_above_surface);
    ++steps;
  }

  return result;
}

void radiance_spectrum_at_direction(const ScatteringSpectrums& spectrums, const OpticalDepthData& extinction, const float3& view_direction,
  const std::vector<LightSource>& light_sources, const Parameters& parameters, SpectralDistribution& result) {
  const float3 origin = {0.0f, kPlanetRadius + parameters.altitude, 0.0f};
  float height_above_surface = length(origin) - kPlanetRadius;

  const float3 density_scale = {
    parameters.rayleigh_scale,
    parameters.mie_scale,
    parameters.ozone_scale,
  };

  float3 view_optical_path = {};
  float3 current_density = density(height_above_surface);

  result.spectral_entry_count = spectrum::WavelengthCount;
  for (uint32_t i = 0; i < result.spectral_entry_count; ++i) {
    result.spectral_entries[i].power = 0;
  }

  float t = 0.0f;
  float to_space = distance_to_sphere(origin, view_direction, {}, kOuterSphereSize);
  float to_planet = distance_to_sphere(origin, view_direction, {}, kPlanetRadius);
  if (to_planet > 0.0f) {
    to_space = to_planet;
  }

  while (t < to_space) {
    float dt = calculate_step_size(t, to_space, origin, view_direction);
    float3 p = origin + view_direction * (t + 0.5f * dt);
    height_above_surface = length(p) - kPlanetRadius;
    t += dt;

    if (height_above_surface < -kRayleighDensityScale)
      break;

    current_density = density(height_above_surface);
    view_optical_path += dt * density_scale * current_density;

    for (const auto& light_source : light_sources) {
      float3 light_optical_path = density_scale * sample_optical_length(p, light_source.direction, extinction);
      float3 total_optical_path = view_optical_path + light_optical_path;

      // For extended light sources, phase function should be averaged over solid angle
      // For now, use point source approximation (valid for small angular sizes)
      const float l_dot_v = dot(light_source.direction, view_direction);
      const float phase_r = phase_rayleigh(l_dot_v);
      const float phase_m = phase_mie(l_dot_v, parameters.anisotropy);

      for (uint32_t i = 0; i < result.spectral_entry_count; ++i) {
        float r = spectrums.rayleigh.spectral_entries[i].power;
        float m = spectrums.mie.spectral_entries[i].power;
        float o = spectrums.ozone.spectral_entries[i].power;

        float transmittance_r = g_transmittance_table.lookup(r * total_optical_path.x);
        float transmittance_m = g_transmittance_table.lookup(m * total_optical_path.y);
        float transmittance_o = g_transmittance_table.lookup(o * total_optical_path.z);
        float total_transmittance = transmittance_r * transmittance_m * transmittance_o;

        // Calculate scattering coefficient contribution
        float scattering_coeff = phase_r * r * density_scale.x * current_density.x + phase_m * m * density_scale.y * current_density.y;

        // Multiply by light source emission spectrum at this wavelength
        float light_emission = light_source.emission_spectrum.spectral_entries[i].power * light_source.intensity_scale;

        // Note: Angular size effects are handled at the emitter level, not in scattering calculation
        // For distant light sources, we use the point source approximation
        float value = total_transmittance * dt * scattering_coeff * light_emission;
        result.spectral_entries[i].power += value;
      }
    }
  }
}

void extinction_spectrum_at_direction(const ScatteringSpectrums& spectrums, const float3& view_direction, const float3& next_direction, const Parameters& parameters,
  SpectralDistribution& result) {
  const float3 origin = {0.0f, kPlanetRadius + parameters.altitude, 0.0f};
  float to_space = distance_to_sphere(origin, view_direction, {}, kOuterSphereSize);

  if (distance_to_sphere(origin, next_direction, {}, kPlanetRadius) > 0.0f) {
    result = spectrums.black;
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
    float dt = calculate_step_size(t, to_space, origin, view_direction);
    float3 p = origin + view_direction * (t + 0.5f * dt);
    height_above_surface = length(p) - kPlanetRadius;
    t += dt;
    current_density = density(height_above_surface);
    view_optical_path += dt * density_scale * current_density;
  }

  for (uint32_t i = 0; i < result.spectral_entry_count; ++i) {
    float r = spectrums.rayleigh.spectral_entries[i].power;
    float m = spectrums.mie.spectral_entries[i].power;
    float o = spectrums.ozone.spectral_entries[i].power;

    float transmittance_r = g_transmittance_table.lookup(r * view_optical_path.x);
    float transmittance_m = g_transmittance_table.lookup(m * view_optical_path.y);
    float transmittance_o = g_transmittance_table.lookup(o * view_optical_path.z);
    result.spectral_entries[i].power = transmittance_r * transmittance_m * transmittance_o;
  }
}

}  // namespace

void init(TaskScheduler& scheduler, ScatteringSpectrums& spectrums, OpticalDepthData& extinction) {
  constexpr uint32_t kSpectrumStepSize = 5u;

  log::info("Precomputing atmosphere spectrums and extinction image %u x %u...", OpticalDepthData::kWidth, OpticalDepthData::kHeight);

  std::vector<float2> r_samples;
  r_samples.reserve(spectrum::WavelengthCount / kSpectrumStepSize + 1);

  std::vector<float2> m_samples;
  m_samples.reserve(spectrum::WavelengthCount / kSpectrumStepSize + 1);

  std::vector<float2> o_samples;
  o_samples.reserve(spectrum::WavelengthCount / kSpectrumStepSize + 1);

  std::vector<float2> b_samples;
  b_samples.reserve(spectrum::WavelengthCount / kSpectrumStepSize + 1);

  auto t0 = std::chrono::steady_clock::now();
  uint32_t count = 0;
  float3 accum = {};
  for (uint32_t w = spectrum::ShortestWavelength; w <= spectrum::LongestWavelength; ++w) {
    uint32_t i = w - spectrum::ShortestWavelength;

    accum.x += scattering::rayleigh(float(w));
    accum.y += scattering::mie(float(w));
    accum.z += scattering::ozone_absorption(float(w));

    if (i % kSpectrumStepSize == 0) {
      r_samples.emplace_back(float2{float(w), accum.x / static_cast<float>(kSpectrumStepSize)});
      m_samples.emplace_back(float2{float(w), accum.y / static_cast<float>(kSpectrumStepSize)});
      o_samples.emplace_back(float2{float(w), accum.z / static_cast<float>(kSpectrumStepSize)});
      b_samples.emplace_back(float2{float(w), 0.0f});
      accum = {};
      ++count;
    }
  }

  using SPD = SpectralDistribution;
  spectrums.rayleigh = SPD::from_samples(r_samples.data(), r_samples.size());
  spectrums.mie = SPD::from_samples(m_samples.data(), m_samples.size());
  spectrums.ozone = SPD::from_samples(o_samples.data(), o_samples.size());
  spectrums.black = SPD::from_samples(b_samples.data(), b_samples.size());

  log::info("Precomputing extinction data...");
  auto t1 = std::chrono::steady_clock::now();
  scheduler.execute(OpticalDepthData::kSize, [&extinction](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t i = begin; i < end; ++i) {
      uint32_t x = i % OpticalDepthData::kWidth;
      uint32_t y = i / OpticalDepthData::kWidth;
      float2 uv = {float(x) / float(OpticalDepthData::kWidth), float(y) / float(OpticalDepthData::kHeight)};
      float2 params = scattering::uv_to_precomputed_params(uv);
      float3 direction = {sqrtf(1.0f - params.x * params.x), params.x, 0.0f};
      float3 origin = {0.0f, kPlanetRadius + params.y, 0.0f};
      float total_distance = distance_to_sphere(origin, direction, {}, kOuterSphereSize);
      float3 value = scattering::optical_length(origin, direction, total_distance);
      ETX_VALIDATE(value);
      extinction.data[x + OpticalDepthData::kWidth * y] = {value.x, value.y, value.z, 0.0f};
    }
  });
  auto t2 = std::chrono::steady_clock::now();
  log::info("Precomputed extinction data: %.3f ms", (t2 - t1).count() / 1.0e+6);
}

void generate_sky_image(const Parameters& parameters, const uint2& dimensions, const std::vector<LightSource>& light_sources, const OpticalDepthData& extinction, float4* buffer,
  const ScatteringSpectrums& spectrums, TaskScheduler& scheduler) {
  log::info("Generating sky image %u x %u...", dimensions.x, dimensions.y);

  auto t0 = std::chrono::steady_clock::now();

  std::atomic<float> ax = {};
  std::atomic<float> ay = {};
  std::atomic<float> az = {};
  std::atomic<float> aw = {};
  scheduler.execute(dimensions.x * dimensions.y, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    float3 avg = {};
    float w = 0.0f;
    SpectralDistribution radiance = spectrums.black;
    for (uint32_t i = begin; i < end; ++i) {
      uint32_t x = i % dimensions.x;
      uint32_t y = i / dimensions.x;
      float u = float(x + 0.5f) / float(dimensions.x);
      float v = float(y + 0.5f) / float(dimensions.y);

      float theta = (1.0f - v) * kPi - kHalfPi;
      float phi = u * kDoublePi - kPi;  // 0 to 1 -> -π to π
      float3 direction;
      if (ETX_USE_EQUAL_AREA_PROJECTION) {
        float v_mapped = v * 2.0f - 1.0f;
        theta = asinf(fmaxf(-1.0f, fminf(1.0f, -v_mapped)));
      }
      direction = from_spherical(phi, theta);
      radiance_spectrum_at_direction(spectrums, extinction, direction, light_sources, parameters, radiance);
      float3 xyz = radiance.integrate_to_xyz();
      float3 rgb = max({}, spectrum::xyz_to_rgb(xyz));
      // Poor man multiple scattering
      // Gather average color of the upper hemisphere
      // Weighted in the way that top pixels contribute more
      // Not physically correct, but looks nice
      float weight = 0.0f;
      if (ETX_USE_EQUAL_AREA_PROJECTION) {
        if (theta > 0.0f) {
          weight = sinf(theta);
        }
      } else {
        float sin_theta = sinf(theta);
        if (sin_theta > 0.0f) {
          weight = sin_theta;
        }
      }
      if (weight > 0.0f) {
        w += weight;
        avg += rgb * weight;
      }
      buffer[x + dimensions.x * y] = {rgb.x, rgb.y, rgb.z, 1.0f};
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
  char path[2048] = {};
  env().file_in_tmp("sky.hdr", path, sizeof(path));
  stbi_write_hdr(path, dimensions.x, dimensions.y, 4, &buffer->x);
}

void generate_sun_image(const Parameters& parameters, const uint2& dimensions, const float3& light_direction, const float angular_size, float4* buffer,
  const ScatteringSpectrums& spectrums, TaskScheduler& scheduler) {
  auto t0 = std::chrono::steady_clock::now();

  log::info("Generating Sun image %u x %u...", dimensions.x, dimensions.y);

  auto basis = orthonormal_basis(light_direction);
  float tan_half_fov = tanf(0.5f * angular_size);
  float solid_angle = kDoublePi * (1.0f - cosf(0.5f * angular_size));

  scheduler.execute(dimensions.x * dimensions.y,
    [&parameters, &dimensions, &basis, light_direction, solid_angle, tan_half_fov, buffer, &spectrums](uint32_t begin, uint32_t end, uint32_t thread_id) {
      SpectralDistribution radiance = spectrums.black;
      for (uint32_t i = begin; i < end; ++i) {
        uint32_t x = i % dimensions.x;
        uint32_t y = i / dimensions.x;
        float u = float(x + 0.5f) / float(dimensions.x) * 2.0f - 1.0f;
        float v0 = float(y + 0.5f) / float(dimensions.y) * 2.0f - 1.0f;
        float v1 = float(y + 1.5f) / float(dimensions.y) * 2.0f - 1.0f;
        float3 d0 = normalize(tan_half_fov * (u * basis.u + v0 * basis.v) + light_direction);
        float3 d1 = normalize(tan_half_fov * (u * basis.u + v1 * basis.v) + light_direction);
        extinction_spectrum_at_direction(spectrums, d0, d1, parameters, radiance);
        float darkening = (1.0f - 0.6f * (1.0f - fmaxf(0.0f, 1.0f - (u * u + v0 * v0))));
        float3 xyz = darkening * radiance.integrate_to_xyz();
        float3 rgb = max({}, spectrum::xyz_to_rgb(xyz));
        buffer[x + dimensions.x * y] = {rgb.x, rgb.y, rgb.z, 1.0f};
      }
    });
  auto t1 = std::chrono::steady_clock::now();
  auto duration = (t1 - t0).count() / 1.0e+6;
  log::info("Sun image generated: %.3f ms (%.3f ms/pixel)", duration, duration / double(dimensions.x * dimensions.y));
  char path[2048] = {};
  env().file_in_tmp("sun.hdr", path, sizeof(path));
  stbi_write_hdr(path, dimensions.x, dimensions.y, 4, &buffer->x);
}

}  // namespace scattering
}  // namespace etx
