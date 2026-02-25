#include <etx/core/log.hxx>
#include <etx/render/shared/scattering.hxx>
#include <etx/render/interop/atmosphere_scattering_shared.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

namespace etx {

constexpr const float kPlanetRadius = kScatteringPlanetRadius;
constexpr const float kAtmosphereRadius = kScatteringAtmosphereRadius;
constexpr const float kOuterSphereSize = kScatteringOuterSphereRadius;
constexpr const float kDeltaDensity = kScatteringDeltaDensity;
constexpr const float kRayleighDensityScale = kScatteringRayleighDensityScale;
constexpr const float kMieDensityScale = kScatteringMieDensityScale;

namespace scattering {

namespace {

void gpu_reset_context_handles(GpuContext& context) {
  context.optical_depth_pipeline = {};
  context.sky_pipeline = {};
  context.sky_finalize_pipeline = {};
  context.sun_pipeline = {};
  context.optical_depth_texture = {};
  context.optical_depth_texture_state = RHIResourceState::Undefined;
  context.sky_light_input_buffer = {};
  context.sky_spectrum_input_buffer = {};
  context.sky_input_buffer_capacity = 0u;
  context.initialized = false;
}

struct GpuSkyInputBuffers {
  AtmosphereSkyGpuParameters parameters = {};
  RHIBuffer light_buffer = {};
  RHIBuffer spectrum_buffer = {};
  uint32_t light_count = 0u;
};

void gpu_cleanup_sky_input_buffers(RHIContext& rhi, GpuSkyInputBuffers& buffers) {
  auto& device = rhi.device();

  if (buffers.light_buffer.valid()) {
    device.destroy_buffer(buffers.light_buffer);
    buffers.light_buffer = {};
  }

  if (buffers.spectrum_buffer.valid()) {
    device.destroy_buffer(buffers.spectrum_buffer);
    buffers.spectrum_buffer = {};
  }

  buffers.parameters = {};
  buffers.light_count = 0u;
}

bool gpu_create_and_upload_storage_buffer(RHIContext& rhi, const void* data, uint64_t size, RHIBuffer& out_buffer, const char* debug_name) {
  if ((data == nullptr) || (size == 0u)) {
    log::error("Invalid GPU buffer upload request for atmosphere sky input (%s)", (debug_name != nullptr) ? debug_name : "unknown");
    return false;
  }

  RHIBufferDesc desc = {};
  desc.size = size;
  desc.usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
  desc.host_visible = true;

  auto create_result = rhi.device().create_buffer(desc);
  if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
    log::error("Failed to create atmosphere sky GPU buffer (%s): %u", (debug_name != nullptr) ? debug_name : "unknown", static_cast<uint32_t>(create_result.result));
    return false;
  }

  const RHIResult update_result = rhi.device().update_buffer(create_result.handle, data, size);
  if (update_result != RHIResult::Success) {
    log::error("Failed to upload atmosphere sky GPU buffer (%s): %u", (debug_name != nullptr) ? debug_name : "unknown", static_cast<uint32_t>(update_result));
    rhi.device().destroy_buffer(create_result.handle);
    return false;
  }

  out_buffer = create_result.handle;
  return true;
}

bool gpu_prepare_sky_input_buffers(RHIContext& rhi, const Parameters& parameters, const std::vector<LightSource>& light_sources, GpuSkyInputBuffers& out_buffers) {
  gpu_cleanup_sky_input_buffers(rhi, out_buffers);

  out_buffers.parameters =
    scattering_make_gpu_parameters(parameters.altitude, parameters.anisotropy, parameters.rayleigh_scale, parameters.mie_scale, parameters.ozone_scale);

  if (light_sources.empty()) {
    return true;
  }

  std::vector<AtmosphereSkyGpuLight> gpu_lights;
  gpu_lights.reserve(light_sources.size());

  std::vector<SpectralDistribution> gpu_spectra;
  gpu_spectra.reserve(light_sources.size());

  const uint32_t gpu_light_count = static_cast<uint32_t>(light_sources.size());
  for (uint32_t i = 0u; i < gpu_light_count; ++i) {
    const LightSource& light = light_sources[i];

    AtmosphereSkyGpuLight gpu_light = {};
    gpu_light.direction = light.direction;
    gpu_light.angular_size = light.angular_size;
    gpu_light.intensity_scale = light.intensity_scale;
    gpu_light.emission_spectrum_index = i;

    gpu_lights.push_back(gpu_light);
    gpu_spectra.push_back(light.emission_spectrum);
  }

  const uint64_t lights_size = uint64_t(gpu_lights.size()) * sizeof(AtmosphereSkyGpuLight);
  const uint64_t spectra_size = uint64_t(gpu_spectra.size()) * sizeof(SpectralDistribution);

  if (gpu_create_and_upload_storage_buffer(rhi, gpu_lights.data(), lights_size, out_buffers.light_buffer, "atmosphere_sky_lights") == false) {
    gpu_cleanup_sky_input_buffers(rhi, out_buffers);
    return false;
  }

  if (gpu_create_and_upload_storage_buffer(rhi, gpu_spectra.data(), spectra_size, out_buffers.spectrum_buffer, "atmosphere_sky_spectra") == false) {
    gpu_cleanup_sky_input_buffers(rhi, out_buffers);
    return false;
  }

  out_buffers.light_count = gpu_light_count;
  return true;
}

bool gpu_ensure_sky_input_cache_capacity(RHIContext& rhi, GpuContext& context, uint32_t light_count) {
  if (light_count == 0u) {
    if (context.sky_light_input_buffer.valid()) {
      rhi.device().destroy_buffer(context.sky_light_input_buffer);
      context.sky_light_input_buffer = {};
    }
    if (context.sky_spectrum_input_buffer.valid()) {
      rhi.device().destroy_buffer(context.sky_spectrum_input_buffer);
      context.sky_spectrum_input_buffer = {};
    }
    context.sky_input_buffer_capacity = 0u;
    return true;
  }

  if ((context.sky_input_buffer_capacity >= light_count) && context.sky_light_input_buffer.valid() && context.sky_spectrum_input_buffer.valid()) {
    return true;
  }

  if (context.sky_light_input_buffer.valid()) {
    rhi.device().destroy_buffer(context.sky_light_input_buffer);
    context.sky_light_input_buffer = {};
  }
  if (context.sky_spectrum_input_buffer.valid()) {
    rhi.device().destroy_buffer(context.sky_spectrum_input_buffer);
    context.sky_spectrum_input_buffer = {};
  }
  context.sky_input_buffer_capacity = 0u;

  RHIBufferDesc lights_desc = {};
  lights_desc.size = uint64_t(light_count) * sizeof(AtmosphereSkyGpuLight);
  lights_desc.usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
  lights_desc.host_visible = true;
  auto lights_result = rhi.device().create_buffer(lights_desc);
  if ((lights_result.result != RHIResult::Success) || (lights_result.handle.valid() == false)) {
    log::error("Failed to create cached atmosphere sky light buffer (%u)", static_cast<uint32_t>(lights_result.result));
    return false;
  }

  RHIBufferDesc spectra_desc = {};
  spectra_desc.size = uint64_t(light_count) * sizeof(SpectralDistribution);
  spectra_desc.usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
  spectra_desc.host_visible = true;
  auto spectra_result = rhi.device().create_buffer(spectra_desc);
  if ((spectra_result.result != RHIResult::Success) || (spectra_result.handle.valid() == false)) {
    log::error("Failed to create cached atmosphere sky spectrum buffer (%u)", static_cast<uint32_t>(spectra_result.result));
    rhi.device().destroy_buffer(lights_result.handle);
    return false;
  }

  context.sky_light_input_buffer = lights_result.handle;
  context.sky_spectrum_input_buffer = spectra_result.handle;
  context.sky_input_buffer_capacity = light_count;
  return true;
}

bool gpu_upload_sky_input_cache(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const std::vector<LightSource>& light_sources,
  AtmosphereSkyGpuParameters& out_parameters, uint32_t& out_light_count) {
  out_parameters = scattering_make_gpu_parameters(parameters.altitude, parameters.anisotropy, parameters.rayleigh_scale, parameters.mie_scale, parameters.ozone_scale);
  out_light_count = static_cast<uint32_t>(light_sources.size());

  if (gpu_ensure_sky_input_cache_capacity(rhi, context, out_light_count) == false) {
    return false;
  }
  if (out_light_count == 0u) {
    return true;
  }

  std::vector<AtmosphereSkyGpuLight> gpu_lights;
  gpu_lights.reserve(light_sources.size());
  std::vector<SpectralDistribution> gpu_spectra;
  gpu_spectra.reserve(light_sources.size());

  for (uint32_t i = 0u; i < out_light_count; ++i) {
    const LightSource& light = light_sources[i];
    AtmosphereSkyGpuLight gpu_light = {};
    gpu_light.direction = light.direction;
    gpu_light.angular_size = light.angular_size;
    gpu_light.intensity_scale = light.intensity_scale;
    gpu_light.emission_spectrum_index = i;
    gpu_lights.push_back(gpu_light);
    gpu_spectra.push_back(light.emission_spectrum);
  }

  const RHIResult lights_update_result =
    rhi.device().update_buffer(context.sky_light_input_buffer, gpu_lights.data(), uint64_t(gpu_lights.size()) * sizeof(AtmosphereSkyGpuLight));
  if (lights_update_result != RHIResult::Success) {
    log::error("Failed to upload cached atmosphere sky light buffer (%u)", static_cast<uint32_t>(lights_update_result));
    return false;
  }

  const RHIResult spectra_update_result =
    rhi.device().update_buffer(context.sky_spectrum_input_buffer, gpu_spectra.data(), uint64_t(gpu_spectra.size()) * sizeof(SpectralDistribution));
  if (spectra_update_result != RHIResult::Success) {
    log::error("Failed to upload cached atmosphere sky spectrum buffer (%u)", static_cast<uint32_t>(spectra_update_result));
    return false;
  }

  return true;
}

bool gpu_create_float4_storage_texture(RHIContext& rhi, const uint2& dimensions, RHITextureUsage usage, RHITexture& out_texture) {
  if ((dimensions.x == 0u) || (dimensions.y == 0u)) {
    log::error("Invalid atmosphere sky texture dimensions: %u x %u", dimensions.x, dimensions.y);
    return false;
  }

  RHITextureDesc desc = {};
  desc.width = dimensions.x;
  desc.height = dimensions.y;
  desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
  desc.usage = usage;

  auto result = rhi.device().create_texture(desc);
  if ((result.result != RHIResult::Success) || (result.handle.valid() == false)) {
    log::error("Failed to create atmosphere sky helper texture (%u)", static_cast<uint32_t>(result.result));
    return false;
  }

  out_texture = result.handle;
  return true;
}

bool gpu_download_float4_texture(RHIContext& rhi, RHITexture texture, const uint2& dimensions, std::vector<float4>& out_pixels, RHIResourceState& texture_state) {
  if (texture.valid() == false) {
    log::error("Invalid texture for atmosphere sky readback");
    return false;
  }
  if ((dimensions.x == 0u) || (dimensions.y == 0u)) {
    log::error("Invalid atmosphere sky readback dimensions: %u x %u", dimensions.x, dimensions.y);
    return false;
  }

  RHIBufferDesc desc = {};
  desc.size = uint64_t(dimensions.x) * uint64_t(dimensions.y) * sizeof(float4);
  desc.usage = RHIBufferUsage::TransferDst;
  desc.host_visible = true;

  auto buffer_result = rhi.device().create_buffer(desc);
  if ((buffer_result.result != RHIResult::Success) || (buffer_result.handle.valid() == false)) {
    log::error("Failed to create atmosphere sky readback buffer (%u)", static_cast<uint32_t>(buffer_result.result));
    return false;
  }

  RHIBuffer readback_buffer = buffer_result.handle;
  RHICommandBuffer cmd = {};
  bool success = false;

  do {
    cmd = rhi.get_command_buffer();
    rhi.command_buffer_begin(cmd);
    if (texture_state != RHIResourceState::TransferSrc) {
      rhi.cmd_texture_barrier(cmd, texture, texture_state, RHIResourceState::TransferSrc);
      texture_state = RHIResourceState::TransferSrc;
    }
    rhi.cmd_copy_texture_to_buffer(cmd, texture, readback_buffer, dimensions.x, dimensions.y);
    rhi.command_buffer_end(cmd);
    rhi.submit_command_buffer({cmd});

    const RHIResult wait_result = rhi.wait_idle();
    if (wait_result != RHIResult::Success) {
      log::error("Failed to wait for atmosphere sky readback (%u)", static_cast<uint32_t>(wait_result));
      break;
    }

    out_pixels.resize(uint64_t(dimensions.x) * uint64_t(dimensions.y));
    const RHIResult read_result = rhi.device().read_buffer(readback_buffer, out_pixels.data(), desc.size);
    if (read_result != RHIResult::Success) {
      log::error("Failed to read atmosphere sky readback buffer (%u)", static_cast<uint32_t>(read_result));
      out_pixels.clear();
      break;
    }

    success = true;
  } while (false);

  if (cmd.valid()) {
    rhi.destroy_command_buffer(cmd);
  }
  rhi.device().destroy_buffer(readback_buffer);
  return success;
}

float3 gpu_sky_average_color_from_partial_sums(const std::vector<float4>& partial_sums) {
  float3 sum_rgb = {};
  float sum_w = 0.0f;

  for (const float4& value : partial_sums) {
    sum_rgb += float3(value.x, value.y, value.z);
    sum_w += value.w;
  }

  if (sum_w <= 0.0f) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  return sum_rgb / sum_w;
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

  ETX_SHARED_INLINE float lookup(float optical_depth) const {
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

  ETX_SHARED_INLINE DensityPair lookup(float height) const {
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

void clear_spectral_powers(SpectralDistribution& value) {
  if (value.spectral_entry_count == 0u) {
    value = SpectralDistribution::constant(0.0f);
  }

  value.integrated_value = {};
  for (uint32_t i = 0; i < value.spectral_entry_count; ++i) {
    value.spectral_entries[i].power = 0.0f;
  }
}

float3 sample_optical_length(const float3& pos, const float3& light_direction, const OpticalDepthData& extinction) {
  float2 uv = scattering_optical_depth_precomputed_uv_from_position(pos, light_direction);
  float4 e = extinction.evaluate(uv);
  ETX_VALIDATE(e);
  return {e.x, e.y, e.z};
}

float calculate_step_size(float current_distance, float total_distance, const float3 origin, const float3& direction) {
  float3 position = origin + direction * current_distance;
  float position_len = length(position);
  float height = position_len - kPlanetRadius;
  float3 directional_derivative = density_derivative(height);

  float directional_factor = 0.0f;
  if ((position_len > kRayEpsilon)) {
    float3 normalized_position = position / position_len;
    directional_factor = dot(normalized_position, direction);
  }
  directional_derivative = directional_derivative * directional_factor;

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

void radiance_spectrum_at_direction(const OpticalDepthData& extinction, const float3& view_direction, const std::vector<LightSource>& light_sources, const Parameters& parameters,
  SpectralDistribution& result) {
  const float3 origin = {0.0f, kPlanetRadius + parameters.altitude, 0.0f};
  float height_above_surface = length(origin) - kPlanetRadius;

  const float3 density_scale = {
    parameters.rayleigh_scale,
    parameters.mie_scale,
    parameters.ozone_scale,
  };

  float3 view_optical_path = {};
  float3 current_density = density(height_above_surface);

  clear_spectral_powers(result);

  float t = 0.0f;
  float to_space = scattering_distance_to_atmosphere_or_planet(origin, view_direction);

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
      const float phase_r = scattering_phase_rayleigh(l_dot_v);
      const float phase_m = scattering_phase_mie(l_dot_v, parameters.anisotropy);

      for (uint32_t i = 0; i < result.spectral_entry_count; ++i) {
        float wavelength = result.spectral_entries[i].wavelength;
        float r = scattering_rayleigh(wavelength);
        float m = scattering_mie(wavelength);
        float o = scattering_ozone_absorption(wavelength);

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

void extinction_spectrum_at_direction(const float3& view_direction, const float3& next_direction, const Parameters& parameters, SpectralDistribution& result) {
  const float3 origin = {0.0f, kPlanetRadius + parameters.altitude, 0.0f};
  float to_space = distance_to_sphere(origin, view_direction, {}, kOuterSphereSize);

  if (result.spectral_entry_count == 0u) {
    result = SpectralDistribution::constant(0.0f);
  }

  if (distance_to_sphere(origin, next_direction, {}, kPlanetRadius) > 0.0f) {
    clear_spectral_powers(result);
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
    float wavelength = result.spectral_entries[i].wavelength;
    float r = scattering_rayleigh(wavelength);
    float m = scattering_mie(wavelength);
    float o = scattering_ozone_absorption(wavelength);

    float transmittance_r = g_transmittance_table.lookup(r * view_optical_path.x);
    float transmittance_m = g_transmittance_table.lookup(m * view_optical_path.y);
    float transmittance_o = g_transmittance_table.lookup(o * view_optical_path.z);
    result.spectral_entries[i].power = transmittance_r * transmittance_m * transmittance_o;
  }
}

}  // namespace

SpectralDistribution rayleigh_spectrum() {
  constexpr uint32_t kSpectrumStepSize = 5u;
  std::vector<float2> r_samples;
  r_samples.reserve(WavelengthCount / kSpectrumStepSize + 1);
  float3 accum = {};
  for (uint32_t w = ShortestWavelength; w <= LongestWavelength; ++w) {
    uint32_t i = w - ShortestWavelength;

    accum.x += scattering_rayleigh(float(w));

    if (i % kSpectrumStepSize == 0) {
      r_samples.emplace_back(float2{float(w), accum.x / static_cast<float>(kSpectrumStepSize)});
      accum = {};
    }
  }

  return SpectralDistribution::from_samples(r_samples.data(), r_samples.size());
}

SpectralDistribution mie_spectrum() {
  constexpr uint32_t kSpectrumStepSize = 5u;
  std::vector<float2> m_samples;
  m_samples.reserve(WavelengthCount / kSpectrumStepSize + 1);

  float accum = 0.0f;
  for (uint32_t w = ShortestWavelength; w <= LongestWavelength; ++w) {
    uint32_t i = w - ShortestWavelength;

    accum += scattering_mie(float(w));

    if (i % kSpectrumStepSize == 0) {
      m_samples.emplace_back(float2{float(w), accum / static_cast<float>(kSpectrumStepSize)});
      accum = 0.0f;
    }
  }

  return SpectralDistribution::from_samples(m_samples.data(), m_samples.size());
}

SpectralDistribution ozone_spectrum() {
  constexpr uint32_t kSpectrumStepSize = 5u;
  std::vector<float2> o_samples;
  o_samples.reserve(WavelengthCount / kSpectrumStepSize + 1);

  float accum = 0.0f;
  for (uint32_t w = ShortestWavelength; w <= LongestWavelength; ++w) {
    uint32_t i = w - ShortestWavelength;

    accum += scattering_ozone_absorption(float(w));

    if (i % kSpectrumStepSize == 0) {
      o_samples.emplace_back(float2{float(w), accum / static_cast<float>(kSpectrumStepSize)});
      accum = 0.0f;
    }
  }

  return SpectralDistribution::from_samples(o_samples.data(), o_samples.size());
}

bool generate_sky_image(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const std::vector<LightSource>& light_sources, float4* buffer) {
  if (buffer == nullptr) {
    log::error("Invalid output buffer for atmosphere sky image generation");
    return false;
  }

  RHITexture sky_texture = {};
  RHIResourceState sky_texture_state = RHIResourceState::Undefined;
  if (gpu_create_sky_texture(rhi, context, parameters, dimensions, light_sources, sky_texture, sky_texture_state) == false) {
    return false;
  }

  std::vector<float4> gpu_pixels;
  if (gpu_download_float4_texture(rhi, sky_texture, dimensions, gpu_pixels, sky_texture_state) == false) {
    rhi.device().destroy_texture(sky_texture);
    return false;
  }

  rhi.device().destroy_texture(sky_texture);

  const uint64_t pixel_count = uint64_t(dimensions.x) * uint64_t(dimensions.y);
  if (gpu_pixels.size() != pixel_count) {
    log::error("Unexpected atmosphere sky image pixel count: %llu (expected %llu)", static_cast<unsigned long long>(gpu_pixels.size()),
      static_cast<unsigned long long>(pixel_count));
    return false;
  }

  for (uint64_t i = 0u; i < pixel_count; ++i) {
    buffer[i] = gpu_pixels[i];
  }

  return true;
}

bool generate_sun_image(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const float3& light_direction, const float angular_size,
  float4* buffer) {
  if (buffer == nullptr) {
    log::error("Invalid output buffer for atmosphere sun image generation");
    return false;
  }

  std::vector<float4> gpu_pixels;
  if (gpu_generate_sun_image(rhi, context, parameters, dimensions, light_direction, angular_size, gpu_pixels) == false) {
    return false;
  }

  const uint64_t pixel_count = uint64_t(dimensions.x) * uint64_t(dimensions.y);
  if (gpu_pixels.size() != pixel_count) {
    log::error("Unexpected atmosphere sun image pixel count: %llu (expected %llu)", static_cast<unsigned long long>(gpu_pixels.size()),
      static_cast<unsigned long long>(pixel_count));
    return false;
  }

  for (uint64_t i = 0u; i < pixel_count; ++i) {
    buffer[i] = gpu_pixels[i];
  }

  return true;
}

bool gpu_init(RHIContext& rhi, GpuContext& context) {
  if (context.initialized) {
    log::error("Atmosphere GPU context is already initialized");
    return false;
  }

  auto& device = rhi.device();
  auto& compiler = ShaderCompiler::instance();

  ShaderCompiler::MultiShaderCompilationResult compilation = {};
  compilation = compiler.compile("shaders/atmosphere_optical_depth.hlsl", {{"optical_depth_main", RHIShaderStage::Compute}});
  if (compilation.result != RHIResult::Success) {
    log::error("Failed to compile atmosphere optical depth shader: %s", compilation.error_message.c_str());
    return false;
  }
  if (compilation.binaries.empty()) {
    log::error("Atmosphere optical depth shader compilation returned no binaries");
    return false;
  }

  RHIComputePipelineDesc pipeline_desc = device.make_compute_pipeline_desc(compilation.binaries[0]);
  auto pipeline_result = device.create_compute_pipeline(pipeline_desc);
  if ((pipeline_result.result != RHIResult::Success) || (pipeline_result.handle.valid() == false)) {
    log::error("Failed to create atmosphere optical depth pipeline (%u)", static_cast<uint32_t>(pipeline_result.result));
    return false;
  }
  context.optical_depth_pipeline = pipeline_result.handle;

  compilation = compiler.compile("shaders/atmosphere_sky.hlsl", {{"sky_raw_main", RHIShaderStage::Compute}, {"sky_finalize_main", RHIShaderStage::Compute}});
  if (compilation.result != RHIResult::Success) {
    log::error("Failed to compile atmosphere sky shaders: %s", compilation.error_message.c_str());
    gpu_cleanup(rhi, context);
    return false;
  }
  if (compilation.binaries.size() < 2u) {
    log::error("Atmosphere sky shader compilation returned insufficient binaries");
    gpu_cleanup(rhi, context);
    return false;
  }

  RHIComputePipelineDesc sky_raw_desc = device.make_compute_pipeline_desc(compilation.binaries[0]);
  auto sky_raw_result = device.create_compute_pipeline(sky_raw_desc);
  if ((sky_raw_result.result != RHIResult::Success) || (sky_raw_result.handle.valid() == false)) {
    log::error("Failed to create atmosphere sky raw pipeline (%u)", static_cast<uint32_t>(sky_raw_result.result));
    gpu_cleanup(rhi, context);
    return false;
  }
  context.sky_pipeline = sky_raw_result.handle;

  RHIComputePipelineDesc sky_finalize_desc = device.make_compute_pipeline_desc(compilation.binaries[1]);
  auto sky_finalize_result = device.create_compute_pipeline(sky_finalize_desc);
  if ((sky_finalize_result.result != RHIResult::Success) || (sky_finalize_result.handle.valid() == false)) {
    log::error("Failed to create atmosphere sky finalize pipeline (%u)", static_cast<uint32_t>(sky_finalize_result.result));
    gpu_cleanup(rhi, context);
    return false;
  }
  context.sky_finalize_pipeline = sky_finalize_result.handle;

  compilation = compiler.compile("shaders/atmosphere_sun.hlsl", {{"sun_main", RHIShaderStage::Compute}});
  if (compilation.result != RHIResult::Success) {
    log::error("Failed to compile atmosphere sun shader: %s", compilation.error_message.c_str());
    gpu_cleanup(rhi, context);
    return false;
  }
  if (compilation.binaries.empty()) {
    log::error("Atmosphere sun shader compilation returned no binaries");
    gpu_cleanup(rhi, context);
    return false;
  }

  RHIComputePipelineDesc sun_desc = device.make_compute_pipeline_desc(compilation.binaries[0]);
  auto sun_result = device.create_compute_pipeline(sun_desc);
  if ((sun_result.result != RHIResult::Success) || (sun_result.handle.valid() == false)) {
    log::error("Failed to create atmosphere sun pipeline (%u)", static_cast<uint32_t>(sun_result.result));
    gpu_cleanup(rhi, context);
    return false;
  }
  context.sun_pipeline = sun_result.handle;

  RHITextureDesc optical_depth_desc = {};
  optical_depth_desc.width = OpticalDepthData::kWidth;
  optical_depth_desc.height = OpticalDepthData::kHeight;
  optical_depth_desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
  optical_depth_desc.usage = RHITextureUsage::Storage | RHITextureUsage::Sampled | RHITextureUsage::TransferSrc;

  auto optical_depth_result = device.create_texture(optical_depth_desc);
  if ((optical_depth_result.result != RHIResult::Success) || (optical_depth_result.handle.valid() == false)) {
    log::error("Failed to create atmosphere optical depth texture (%u)", static_cast<uint32_t>(optical_depth_result.result));
    gpu_cleanup(rhi, context);
    return false;
  }

  context.optical_depth_texture = optical_depth_result.handle;
  context.optical_depth_texture_state = RHIResourceState::Undefined;
  context.initialized = true;
  return true;
}
void gpu_cleanup(RHIContext& rhi, GpuContext& context) {
  auto& device = rhi.device();

  if (context.optical_depth_pipeline.valid()) {
    device.destroy_pipeline(context.optical_depth_pipeline);
  }
  if (context.sky_pipeline.valid()) {
    device.destroy_pipeline(context.sky_pipeline);
  }
  if (context.sky_finalize_pipeline.valid()) {
    device.destroy_pipeline(context.sky_finalize_pipeline);
  }
  if (context.sun_pipeline.valid()) {
    device.destroy_pipeline(context.sun_pipeline);
  }

  if (context.optical_depth_texture.valid()) {
    device.destroy_texture(context.optical_depth_texture);
  }
  if (context.sky_light_input_buffer.valid()) {
    device.destroy_buffer(context.sky_light_input_buffer);
  }
  if (context.sky_spectrum_input_buffer.valid()) {
    device.destroy_buffer(context.sky_spectrum_input_buffer);
  }
  gpu_reset_context_handles(context);
}

bool gpu_precompute_optical_depth(RHIContext& rhi, GpuContext& context) {
  auto t0 = std::chrono::steady_clock::now();

  if (context.initialized == false) {
    log::error("Atmosphere GPU context is not initialized");
    return false;
  }
  if (context.optical_depth_pipeline.valid() == false) {
    log::error("Atmosphere optical depth pipeline is not initialized");
    return false;
  }
  if (context.optical_depth_texture.valid() == false) {
    log::error("Atmosphere optical depth texture is not initialized");
    return false;
  }

  struct OpticalDepthPushConstants {
    uint32_t output_texture_index = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t pad0 = 0u;
  };

  OpticalDepthPushConstants pc = {};
  pc.output_texture_index = get_bindless_descriptor_index(context.optical_depth_texture);
  pc.width = OpticalDepthData::kWidth;
  pc.height = OpticalDepthData::kHeight;

  auto cmd = rhi.get_command_buffer();
  rhi.command_buffer_begin(cmd);
  if (context.optical_depth_texture_state != RHIResourceState::General) {
    rhi.cmd_texture_barrier(cmd, context.optical_depth_texture, context.optical_depth_texture_state, RHIResourceState::General);
    context.optical_depth_texture_state = RHIResourceState::General;
  }
  rhi.cmd_set_pipeline(cmd, context.optical_depth_pipeline);
  rhi.cmd_push_constants(cmd, &pc, sizeof(pc));
  rhi.cmd_dispatch(cmd, {(OpticalDepthData::kWidth + 7u) / 8u, (OpticalDepthData::kHeight + 7u) / 8u, 1u});
  rhi.command_buffer_end(cmd);
  rhi.submit_command_buffer({cmd});

  const RHIResult wait_result = rhi.wait_idle();
  if (wait_result != RHIResult::Success) {
    log::error("Failed to wait for atmosphere optical depth dispatch (%u)", static_cast<uint32_t>(wait_result));
    rhi.command_buffer_reset(cmd);
    rhi.destroy_command_buffer(cmd);
    return false;
  }

  rhi.command_buffer_reset(cmd);
  rhi.destroy_command_buffer(cmd);

  auto t1 = std::chrono::steady_clock::now();
  log::info("Atmosphere GPU optical depth generated: %.3f ms", double((t1 - t0).count()) / 1.0e+6);
  return true;
}

bool gpu_record_generate_sky_raw(RHIContext& rhi, RHICommandBuffer cmd, GpuContext& context, const Parameters& parameters, const uint2& dimensions,
  const std::vector<LightSource>& light_sources, RHITexture output_texture, RHIResourceState& output_texture_state) {
  if (context.initialized == false) {
    log::error("Atmosphere GPU context is not initialized");
    return false;
  }
  if (context.sky_pipeline.valid() == false) {
    log::error("Atmosphere sky raw pipeline is not initialized");
    return false;
  }
  if (context.optical_depth_texture.valid() == false) {
    log::error("Atmosphere optical depth texture is not initialized");
    return false;
  }
  if (output_texture.valid() == false) {
    log::error("Atmosphere sky output texture is invalid");
    return false;
  }
  if ((dimensions.x == 0u) || (dimensions.y == 0u)) {
    log::error("Invalid atmosphere sky dimensions: %u x %u", dimensions.x, dimensions.y);
    return false;
  }
  if (cmd.valid() == false) {
    log::error("Invalid command buffer for atmosphere sky generation");
    return false;
  }

  AtmosphereSkyGpuParameters gpu_parameters = {};
  uint32_t gpu_light_count = 0u;
  if (gpu_upload_sky_input_cache(rhi, context, parameters, light_sources, gpu_parameters, gpu_light_count) == false) {
    return false;
  }

  AtmosphereSkyPushConstants pc = {};
  pc.output_texture_index = get_bindless_descriptor_index(output_texture);
  pc.optical_depth_texture_index = get_bindless_descriptor_index(context.optical_depth_texture);
  pc.optical_depth_sampler_index = rhi.get_sampler_index(RHISamplerType::LinearClamp);
  pc.aux_texture_index = kInvalidIndex;
  pc.lights_buffer_index = context.sky_light_input_buffer.valid() ? get_bindless_descriptor_index(context.sky_light_input_buffer) : kInvalidIndex;
  pc.spectra_buffer_index = context.sky_spectrum_input_buffer.valid() ? get_bindless_descriptor_index(context.sky_spectrum_input_buffer) : kInvalidIndex;
  pc.width = dimensions.x;
  pc.height = dimensions.y;
  pc.light_count = gpu_light_count;
  pc.atmosphere_altitude = gpu_parameters.altitude;
  pc.atmosphere_anisotropy = gpu_parameters.anisotropy;
  pc.atmosphere_rayleigh_scale = gpu_parameters.rayleigh_scale;
  pc.atmosphere_mie_scale = gpu_parameters.mie_scale;
  pc.atmosphere_ozone_scale = gpu_parameters.ozone_scale;

  if (context.optical_depth_texture_state != RHIResourceState::ShaderReadOnly) {
    rhi.cmd_texture_barrier(cmd, context.optical_depth_texture, context.optical_depth_texture_state, RHIResourceState::ShaderReadOnly);
    context.optical_depth_texture_state = RHIResourceState::ShaderReadOnly;
  }
  if (output_texture_state != RHIResourceState::General) {
    rhi.cmd_texture_barrier(cmd, output_texture, output_texture_state, RHIResourceState::General);
    output_texture_state = RHIResourceState::General;
  }

  rhi.cmd_set_pipeline(cmd, context.sky_pipeline);
  rhi.cmd_push_constants(cmd, &pc, sizeof(pc));
  rhi.cmd_dispatch(cmd, {(dimensions.x + 7u) / 8u, (dimensions.y + 7u) / 8u, 1u});
  return true;
}

bool gpu_generate_sky(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const std::vector<LightSource>& light_sources,
  RHITexture output_texture, RHIResourceState& output_texture_state) {
  if (context.initialized == false) {
    log::error("Atmosphere GPU context is not initialized");
    return false;
  }
  if (context.sky_pipeline.valid() == false) {
    log::error("Atmosphere sky raw pipeline is not initialized");
    return false;
  }
  if (context.sky_finalize_pipeline.valid() == false) {
    log::error("Atmosphere sky finalize pipeline is not initialized");
    return false;
  }
  if (context.optical_depth_texture.valid() == false) {
    log::error("Atmosphere optical depth texture is not initialized");
    return false;
  }
  if (output_texture.valid() == false) {
    log::error("Atmosphere sky output texture is invalid");
    return false;
  }
  if ((dimensions.x == 0u) || (dimensions.y == 0u)) {
    log::error("Invalid atmosphere sky dimensions: %u x %u", dimensions.x, dimensions.y);
    return false;
  }

  GpuSkyInputBuffers sky_inputs = {};
  if (gpu_prepare_sky_input_buffers(rhi, parameters, light_sources, sky_inputs) == false) {
    return false;
  }

  const uint2 group_dimensions = {(dimensions.x + 7u) / 8u, (dimensions.y + 7u) / 8u};

  RHITexture partial_sums_texture = {};
  if (gpu_create_float4_storage_texture(rhi, group_dimensions, RHITextureUsage::Storage | RHITextureUsage::TransferSrc, partial_sums_texture) == false) {
    gpu_cleanup_sky_input_buffers(rhi, sky_inputs);
    return false;
  }

  RHIResourceState partial_sums_texture_state = RHIResourceState::Undefined;
  bool success = false;
  RHICommandBuffer raw_cmd = {};
  RHICommandBuffer finalize_cmd = {};

  do {
    AtmosphereSkyPushConstants pc = {};
    pc.output_texture_index = get_bindless_descriptor_index(output_texture);
    pc.optical_depth_texture_index = get_bindless_descriptor_index(context.optical_depth_texture);
    pc.optical_depth_sampler_index = rhi.get_sampler_index(RHISamplerType::LinearClamp);
    pc.aux_texture_index = get_bindless_descriptor_index(partial_sums_texture);
    pc.lights_buffer_index = sky_inputs.light_buffer.valid() ? get_bindless_descriptor_index(sky_inputs.light_buffer) : kInvalidIndex;
    pc.spectra_buffer_index = sky_inputs.spectrum_buffer.valid() ? get_bindless_descriptor_index(sky_inputs.spectrum_buffer) : kInvalidIndex;
    pc.width = dimensions.x;
    pc.height = dimensions.y;
    pc.light_count = sky_inputs.light_count;
    pc.atmosphere_altitude = sky_inputs.parameters.altitude;
    pc.atmosphere_anisotropy = sky_inputs.parameters.anisotropy;
    pc.atmosphere_rayleigh_scale = sky_inputs.parameters.rayleigh_scale;
    pc.atmosphere_mie_scale = sky_inputs.parameters.mie_scale;
    pc.atmosphere_ozone_scale = sky_inputs.parameters.ozone_scale;

    raw_cmd = rhi.get_command_buffer();
    rhi.command_buffer_begin(raw_cmd);
    if (context.optical_depth_texture_state != RHIResourceState::ShaderReadOnly) {
      rhi.cmd_texture_barrier(raw_cmd, context.optical_depth_texture, context.optical_depth_texture_state, RHIResourceState::ShaderReadOnly);
      context.optical_depth_texture_state = RHIResourceState::ShaderReadOnly;
    }
    if (output_texture_state != RHIResourceState::General) {
      rhi.cmd_texture_barrier(raw_cmd, output_texture, output_texture_state, RHIResourceState::General);
      output_texture_state = RHIResourceState::General;
    }
    if (partial_sums_texture_state != RHIResourceState::General) {
      rhi.cmd_texture_barrier(raw_cmd, partial_sums_texture, partial_sums_texture_state, RHIResourceState::General);
      partial_sums_texture_state = RHIResourceState::General;
    }
    rhi.cmd_set_pipeline(raw_cmd, context.sky_pipeline);
    rhi.cmd_push_constants(raw_cmd, &pc, sizeof(pc));
    rhi.cmd_dispatch(raw_cmd, {group_dimensions.x, group_dimensions.y, 1u});
    rhi.command_buffer_end(raw_cmd);
    rhi.submit_command_buffer({raw_cmd});

    RHIResult wait_result = rhi.wait_idle();
    if (wait_result != RHIResult::Success) {
      log::error("Failed to wait for atmosphere sky raw dispatch (%u)", static_cast<uint32_t>(wait_result));
      break;
    }

    std::vector<float4> partial_sums;
    if (gpu_download_float4_texture(rhi, partial_sums_texture, group_dimensions, partial_sums, partial_sums_texture_state) == false) {
      break;
    }

    float3 average_color = gpu_sky_average_color_from_partial_sums(partial_sums);

    pc.aux_texture_index = kInvalidIndex;
    pc.average_color_x = average_color.x;
    pc.average_color_y = average_color.y;
    pc.average_color_z = average_color.z;

    finalize_cmd = rhi.get_command_buffer();
    rhi.command_buffer_begin(finalize_cmd);
    if (output_texture_state != RHIResourceState::General) {
      rhi.cmd_texture_barrier(finalize_cmd, output_texture, output_texture_state, RHIResourceState::General);
      output_texture_state = RHIResourceState::General;
    }
    rhi.cmd_set_pipeline(finalize_cmd, context.sky_finalize_pipeline);
    rhi.cmd_push_constants(finalize_cmd, &pc, sizeof(pc));
    rhi.cmd_dispatch(finalize_cmd, {group_dimensions.x, group_dimensions.y, 1u});
    rhi.command_buffer_end(finalize_cmd);
    rhi.submit_command_buffer({finalize_cmd});

    wait_result = rhi.wait_idle();
    if (wait_result != RHIResult::Success) {
      log::error("Failed to wait for atmosphere sky finalize dispatch (%u)", static_cast<uint32_t>(wait_result));
      break;
    }

    success = true;
  } while (false);

  if (raw_cmd.valid()) {
    rhi.destroy_command_buffer(raw_cmd);
  }
  if (finalize_cmd.valid()) {
    rhi.destroy_command_buffer(finalize_cmd);
  }
  if (partial_sums_texture.valid()) {
    rhi.device().destroy_texture(partial_sums_texture);
  }
  gpu_cleanup_sky_input_buffers(rhi, sky_inputs);

  return success;
}

bool gpu_record_generate_sun(RHIContext& rhi, RHICommandBuffer cmd, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const float3& light_direction,
  float angular_size, RHITexture output_texture, RHIResourceState& output_texture_state) {
  if (context.initialized == false) {
    log::error("Atmosphere GPU context is not initialized");
    return false;
  }
  if (context.sun_pipeline.valid() == false) {
    log::error("Atmosphere sun pipeline is not initialized");
    return false;
  }
  if (output_texture.valid() == false) {
    log::error("Atmosphere sun output texture is invalid");
    return false;
  }
  if ((dimensions.x == 0u) || (dimensions.y == 0u)) {
    log::error("Invalid atmosphere sun dimensions: %u x %u", dimensions.x, dimensions.y);
    return false;
  }
  if (angular_size <= 0.0f) {
    log::error("Invalid atmosphere sun angular size: %f", angular_size);
    return false;
  }
  if (cmd.valid() == false) {
    log::error("Invalid command buffer for atmosphere sun generation");
    return false;
  }

  AtmosphereSunPushConstants pc = {};
  pc.output_texture_index = get_bindless_descriptor_index(output_texture);
  pc.width = dimensions.x;
  pc.height = dimensions.y;
  pc.light_direction = normalize(light_direction);
  pc.angular_size = angular_size;
  pc.atmosphere_altitude = parameters.altitude;
  pc.atmosphere_rayleigh_scale = parameters.rayleigh_scale;
  pc.atmosphere_mie_scale = parameters.mie_scale;
  pc.atmosphere_ozone_scale = parameters.ozone_scale;

  if (output_texture_state != RHIResourceState::General) {
    rhi.cmd_texture_barrier(cmd, output_texture, output_texture_state, RHIResourceState::General);
    output_texture_state = RHIResourceState::General;
  }

  rhi.cmd_set_pipeline(cmd, context.sun_pipeline);
  rhi.cmd_push_constants(cmd, &pc, sizeof(pc));
  rhi.cmd_dispatch(cmd, {(dimensions.x + 7u) / 8u, (dimensions.y + 7u) / 8u, 1u});
  return true;
}

bool gpu_generate_sun(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const float3& light_direction, float angular_size,
  RHITexture output_texture, RHIResourceState& output_texture_state) {
  RHICommandBuffer cmd = {};
  bool success = false;

  do {
    cmd = rhi.get_command_buffer();
    rhi.command_buffer_begin(cmd);
    if (gpu_record_generate_sun(rhi, cmd, context, parameters, dimensions, light_direction, angular_size, output_texture, output_texture_state) == false) {
      break;
    }
    rhi.command_buffer_end(cmd);
    rhi.submit_command_buffer({cmd});

    const RHIResult wait_result = rhi.wait_idle();
    if (wait_result != RHIResult::Success) {
      log::error("Failed to wait for atmosphere sun dispatch (%u)", static_cast<uint32_t>(wait_result));
      break;
    }

    success = true;
  } while (false);

  if (cmd.valid()) {
    rhi.destroy_command_buffer(cmd);
  }

  return success;
}

bool gpu_create_sun_texture(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const float3& light_direction, float angular_size,
  RHITexture& out_texture, RHIResourceState& out_texture_state) {
  auto t0 = std::chrono::steady_clock::now();
  out_texture = {};
  out_texture_state = RHIResourceState::Undefined;
  if ((dimensions.x == 0u) || (dimensions.y == 0u)) {
    log::error("Invalid atmosphere sun image dimensions: %u x %u", dimensions.x, dimensions.y);
    return false;
  }

  RHITextureDesc desc = {};
  desc.width = dimensions.x;
  desc.height = dimensions.y;
  desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
  desc.usage = RHITextureUsage::Storage | RHITextureUsage::TransferSrc | RHITextureUsage::Sampled;

  auto texture_result = rhi.device().create_texture(desc);
  if ((texture_result.result != RHIResult::Success) || (texture_result.handle.valid() == false)) {
    log::error("Failed to create atmosphere sun output texture for GPU image generation (%u)", static_cast<uint32_t>(texture_result.result));
    return false;
  }

  RHITexture output_texture = texture_result.handle;
  RHIResourceState output_texture_state = RHIResourceState::Undefined;
  bool success = false;

  do {
    if (gpu_generate_sun(rhi, context, parameters, dimensions, light_direction, angular_size, output_texture, output_texture_state) == false) {
      break;
    }
    success = true;
  } while (false);

  if (success) {
    out_texture = output_texture;
    out_texture_state = output_texture_state;
    auto t1 = std::chrono::steady_clock::now();
    log::info("Atmosphere GPU sun image generated: %.3f ms", double((t1 - t0).count()) / 1.0e+6);
  } else {
    rhi.device().destroy_texture(output_texture);
  }
  return success;
}

bool gpu_create_sky_texture(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const std::vector<LightSource>& light_sources,
  RHITexture& out_texture, RHIResourceState& out_texture_state) {
  auto t0 = std::chrono::steady_clock::now();
  out_texture = {};
  out_texture_state = RHIResourceState::Undefined;
  if ((dimensions.x == 0u) || (dimensions.y == 0u)) {
    log::error("Invalid atmosphere sky image dimensions: %u x %u", dimensions.x, dimensions.y);
    return false;
  }

  RHITextureDesc desc = {};
  desc.width = dimensions.x;
  desc.height = dimensions.y;
  desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
  desc.usage = RHITextureUsage::Storage | RHITextureUsage::TransferSrc | RHITextureUsage::Sampled;

  auto texture_result = rhi.device().create_texture(desc);
  if ((texture_result.result != RHIResult::Success) || (texture_result.handle.valid() == false)) {
    log::error("Failed to create atmosphere sky output texture for GPU image generation (%u)", static_cast<uint32_t>(texture_result.result));
    return false;
  }

  RHITexture output_texture = texture_result.handle;
  RHIResourceState output_texture_state = RHIResourceState::Undefined;
  bool success = false;

  do {
    if (gpu_generate_sky(rhi, context, parameters, dimensions, light_sources, output_texture, output_texture_state) == false) {
      break;
    }
    success = true;
  } while (false);

  if (success) {
    out_texture = output_texture;
    out_texture_state = output_texture_state;
    auto t1 = std::chrono::steady_clock::now();
    log::info("Atmosphere GPU sky image generated: %.3f ms", double((t1 - t0).count()) / 1.0e+6);
  } else {
    rhi.device().destroy_texture(output_texture);
  }
  return success;
}

bool gpu_generate_sun_image(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const float3& light_direction, float angular_size,
  std::vector<float4>& out_pixels) {
  auto t0 = std::chrono::steady_clock::now();
  out_pixels.clear();
  if ((dimensions.x == 0u) || (dimensions.y == 0u)) {
    log::error("Invalid atmosphere sun image dimensions: %u x %u", dimensions.x, dimensions.y);
    return false;
  }

  RHITexture output_texture = {};
  RHIResourceState output_texture_state = RHIResourceState::Undefined;
  bool success = false;

  do {
    if (gpu_create_sun_texture(rhi, context, parameters, dimensions, light_direction, angular_size, output_texture, output_texture_state) == false) {
      break;
    }
    if (gpu_download_float4_texture(rhi, output_texture, dimensions, out_pixels, output_texture_state) == false) {
      out_pixels.clear();
      break;
    }
    success = true;
  } while (false);

  if (output_texture.valid()) {
    rhi.device().destroy_texture(output_texture);
  }
  if (success) {
    auto t1 = std::chrono::steady_clock::now();
    log::info("Atmosphere GPU sun image generated: %.3f ms", double((t1 - t0).count()) / 1.0e+6);
  }
  return success;
}

}  // namespace scattering
}  // namespace etx
