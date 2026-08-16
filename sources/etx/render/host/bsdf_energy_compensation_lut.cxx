#include <etx/render/host/bsdf_energy_compensation_lut.hxx>

#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/exr.hxx>
#include <etx/render/host/image_loaders.hxx>
#include <etx/render/interop/bsdf_energy_compensated_shared.hxx>
#include <etx/render/interop/bsdf_external_shared.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include <cmath>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <unordered_map>

namespace etx {

namespace {

constexpr uint32_t kEnergyCompensationGeneratorVersion = 27u;
constexpr uint32_t kEnergyCompensationConductorLutSize = kBSDFEnergyCompensationConductorLutSize;
constexpr uint32_t kEnergyCompensationDielectricLutSize = kBSDFEnergyCompensationDielectricLutSize;
constexpr uint32_t kEnergyCompensationConductorSampleCount = 2048u;
constexpr uint32_t kEnergyCompensationSampleCount = 2048u;
constexpr uint32_t kEnergyCompensationDielectricMultiScatterSampleCount = 2048u;
constexpr uint32_t kEnergyCompensationDielectricBranchCount = kBSDFEnergyCompensationDielectricBranchCount;
constexpr uint32_t kEnergyCompensationDielectricAverageWidth = kBSDFEnergyCompensationDielectricAverageWidth;
constexpr uint32_t kEnergyCompensationSpectralWavelengthCount = kBSDFEnergyCompensationSpectralWavelengthCount;
constexpr uint32_t kEnergyCompensationSpectralWavelengthGroupSize = kBSDFEnergyCompensationSpectralWavelengthGroupSize;
constexpr uint32_t kEnergyCompensationSpectralWavelengthGroupCount = kBSDFEnergyCompensationSpectralWavelengthGroupCount;
constexpr uint32_t kEnergyCompensationGpuPassConductorDirectional = kBSDFEnergyCompensationGpuPassConductorDirectional;
constexpr uint32_t kEnergyCompensationGpuPassConductorAverage = kBSDFEnergyCompensationGpuPassConductorAverage;
constexpr uint32_t kEnergyCompensationGpuPassDielectricDirectional = kBSDFEnergyCompensationGpuPassDielectricDirectional;
constexpr uint32_t kEnergyCompensationGpuPassDielectricAverage = kBSDFEnergyCompensationGpuPassDielectricAverage;

struct SpectralDirectionalAlbedoResult {
  float3 albedo = {};
  float visible_probability = 0.0f;
  float geometric_albedo = 0.0f;
};

struct DielectricDirectionalAlbedoResult {
  float3 branch_albedo[2] = {};
  float branch_visible_probability[2] = {};
  float visible_probability = 0.0f;
};

struct GeneratedInterfacePaths {
  std::filesystem::path directional;
  std::filesystem::path average;
  std::filesystem::path geometric;
  std::filesystem::path geometric_average;
  std::filesystem::path conductor_fms;
  std::filesystem::path probability;
};

struct EnergyCompensationGpuParams {
  uint32_t output_directional_index = kInvalidIndex;
  uint32_t output_average_index = kInvalidIndex;
  uint32_t output_geometric_index = kInvalidIndex;
  uint32_t output_geometric_average_index = kInvalidIndex;
  uint32_t output_conductor_fms_index = kInvalidIndex;
  uint32_t output_probability_index = kInvalidIndex;
  uint32_t output_total_index = kInvalidIndex;
  uint32_t cache_mode = kBSDFEnergyCompensationCacheModeIntegratedRGB;
  uint32_t wavelength_group_index = 0u;
  uint32_t thinfilm_slice_index = 0u;
  uint32_t thinfilm_slice_count = 1u;
  uint32_t pass_kind = 0u;
  uint32_t sample_count = kEnergyCompensationSampleCount;
  uint32_t multisample_count = kEnergyCompensationDielectricMultiScatterSampleCount;
  uint32_t pad0 = 0u;
  uint32_t pad1 = 0u;
  float4 ext_eta = {};
  float4 ext_k = {};
  float4 int_eta = {};
  float4 int_k = {};
  float4 film_eta = {};
  float4 film_k = {};
  float4 wavelengths = {};
  float thinfilm_thickness = 0.0f;
  uint32_t film_cls = SpectralDistribution::Invalid;
  float thinfilm_weight = 0.0f;
  uint32_t pad3 = 0u;
};

static_assert(sizeof(EnergyCompensationGpuParams) == 192u);
static_assert(offsetof(EnergyCompensationGpuParams, output_directional_index) == 0u);
static_assert(offsetof(EnergyCompensationGpuParams, cache_mode) == 28u);
static_assert(offsetof(EnergyCompensationGpuParams, ext_eta) == 64u);
static_assert(offsetof(EnergyCompensationGpuParams, int_eta) == 96u);
static_assert(offsetof(EnergyCompensationGpuParams, film_eta) == 128u);
static_assert(offsetof(EnergyCompensationGpuParams, wavelengths) == 160u);
static_assert(offsetof(EnergyCompensationGpuParams, thinfilm_thickness) == 176u);
static_assert(offsetof(EnergyCompensationGpuParams, film_cls) == 180u);
static_assert(offsetof(EnergyCompensationGpuParams, thinfilm_weight) == 184u);

struct EnergyCompensationGpuPushConstants {
  uint32_t params_buffer_index = kInvalidIndex;
};

struct EnergyCompensationGpuBuffers {
  RHIBuffer params_directional = {};
  RHIBuffer params_average = {};
  RHIBuffer directional = {};
  RHIBuffer average = {};
  RHIBuffer geometric = {};
  RHIBuffer geometric_average = {};
  RHIBuffer conductor_fms = {};
  RHIBuffer probability = {};
  RHIBuffer total = {};
};

struct EnergyCompensationGpuPipeline {
  RHIPipeline pipeline = {};
  bool initialization_failed = false;
};

float lut_parameter(uint32_t index, uint32_t size) {
  return static_cast<float>(index) / static_cast<float>(size - 1u);
}

float mu_parameter(uint32_t index, uint32_t size) {
  if (index == 0u) {
    return 0.5f / static_cast<float>(size - 1u);
  }
  return lut_parameter(index, size);
}

float alpha_parameter(uint32_t index, uint32_t size) {
  const float axis = lut_parameter(index, size);
  return kBSDFNormalDistributionMinAlpha + (1.0f - kBSDFNormalDistributionMinAlpha) * axis * axis;
}

float radical_inverse_vdc(uint32_t bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
  bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
  bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
  return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

float2 hammersley(uint32_t index, uint32_t count) {
  return float2{(static_cast<float>(index) + 0.5f) / static_cast<float>(count), radical_inverse_vdc(index)};
}

float quasi_random(uint32_t index, uint32_t dimension) {
  const uint32_t seed = index + 1u + dimension * 0x9e3779b9u;
  return min(1.0f - kEpsilon, max(kEpsilon, radical_inverse_vdc(seed)));
}

float2 quasi_random_2d(uint32_t index, uint32_t dimension) {
  return float2{quasi_random(index, dimension), quasi_random(index, dimension + 1u)};
}

SpectralQuery spectral_cache_query(uint32_t wavelength_index) {
  const float t = static_cast<float>(min(wavelength_index, kEnergyCompensationSpectralWavelengthCount - 1u)) /
                  static_cast<float>(kEnergyCompensationSpectralWavelengthCount - 1u);
  SpectralQuery result = {};
  result.wavelength = kShortestWavelength + (kLongestWavelength - kShortestWavelength) * t;
  result.flags = SpectralFlags::Spectral;
  return result;
}

void set_float4_channel(float4& value, uint32_t channel, float scalar) {
  if (channel == 0u) {
    value.x = scalar;
  } else if (channel == 1u) {
    value.y = scalar;
  } else if (channel == 2u) {
    value.z = scalar;
  } else {
    value.w = scalar;
  }
}

float3 incident_direction_from_mu(float mu) {
  return float3{sqrt(max(0.0f, 1.0f - mu * mu)), 0.0f, mu};
}

uint32_t dielectric_side(bool outside) {
  return outside ? 0u : 1u;
}

uint32_t dielectric_branch_index(uint32_t incident_side, uint32_t outgoing_side) {
  return incident_side * 2u + outgoing_side;
}

ThinfilmEval empty_thinfilm() {
  ThinfilmEval result = {};
  result.ior.cls = SpectralDistribution::Invalid;
  result.rgb_wavelengths = kRGBWavelengths;
  result.thickness = 0.0f;
  result.weight = 0.0f;
  return result;
}

SpectralResponse spectrum_or_default(const SceneData& data, uint32_t index, const SpectralQuery& spect, float default_value) {
  if (index < data.spectrum_values.size()) {
    return data.spectrum_values[index].query(spect);
  }
  return SpectralResponse(spect, default_value);
}

RefractiveIndexSample sample_refractive_index(const SceneData& data, const RefractiveIndex& refractive_index, const SpectralQuery& spect) {
  RefractiveIndexSample result = {};
  result.cls = refractive_index.cls;
  result.eta = spectrum_or_default(data, refractive_index.eta_index, spect, 1.0f);
  result.k = spectrum_or_default(data, refractive_index.k_index, spect, 0.0f);
  return result;
}

ThinfilmEval sample_constant_thinfilm(const SceneData& data, const Thinfilm& thinfilm, const SpectralQuery& spect) {
  if (bsdf_resource_thinfilm_enabled(thinfilm) == false) {
    return empty_thinfilm();
  }

  ThinfilmEval result = {};
  result.ior = sample_refractive_index(data, thinfilm.ior, spect);
  result.ior.k = SpectralResponse(spect, 0.0f);
  result.rgb_wavelengths = kRGBWavelengths;
  result.thickness = max(0.0f, thinfilm.min_thickness);
  result.weight = clamp(thinfilm.weight, 0.0f, 1.0f);
  return result;
}

uint32_t thinfilm_lut_slice_count(const Thinfilm& thinfilm) {
  if (bsdf_resource_thinfilm_enabled(thinfilm) == false) {
    return 1u;
  }

  const float thickness_range = fabsf(thinfilm.max_thickness - thinfilm.min_thickness);
  if (thickness_range <= kEpsilon) {
    return 1u;
  }

  const uint32_t estimated_count = static_cast<uint32_t>(ceilf(thickness_range / 50.0f)) + 1u;
  return clamp(estimated_count, 4u, 32u);
}

ThinfilmEval sample_thinfilm_slice(const SceneData& data, const Thinfilm& thinfilm, const SpectralQuery& spect, uint32_t slice_index, uint32_t slice_count) {
  ThinfilmEval result = sample_constant_thinfilm(data, thinfilm, spect);
  if ((bsdf_resource_thinfilm_enabled(thinfilm) == false) || (slice_count <= 1u)) {
    return result;
  }

  const float t = static_cast<float>(slice_index) / static_cast<float>(slice_count - 1u);
  const float minimum_thickness = max(0.0f, thinfilm.min_thickness);
  const float maximum_thickness = max(0.0f, thinfilm.max_thickness);
  result.thickness = minimum_thickness + (maximum_thickness - minimum_thickness) * t;
  return result;
}

float4 spectral_response_to_gpu_float4(const ::SpectralResponse& value) {
  return float4{value.integrated.x, value.integrated.y, value.integrated.z, spectral_response_monochromatic(value)};
}

void set_gpu_float4_channel(float4& value, uint32_t channel, float scalar) {
  if (channel == 0u) {
    value.x = scalar;
  } else if (channel == 1u) {
    value.y = scalar;
  } else if (channel == 2u) {
    value.z = scalar;
  } else {
    value.w = scalar;
  }
}

bool gpu_create_storage_buffer(RHIContext& rhi, const void* data, uint64_t size, RHIBufferUsage usage, bool host_visible, RHIBuffer& out_buffer, const char* debug_name) {
  if (size == 0u) {
    log::error("Invalid BSDF energy-compensation GPU buffer size for %s", debug_name);
    return false;
  }

  RHIBufferDesc desc = {};
  desc.size = size;
  desc.usage = RHIBufferUsage::Storage | usage;
  desc.host_visible = host_visible;
  auto result = rhi.device().create_buffer(desc);
  if ((result.result != RHIResult::Success) || (result.handle.valid() == false)) {
    log::error("Failed to create BSDF energy-compensation GPU buffer %s (%u)", debug_name, static_cast<uint32_t>(result.result));
    return false;
  }

  if (data != nullptr) {
    const RHIResult update_result = rhi.device().update_buffer(result.handle, data, size);
    if (update_result != RHIResult::Success) {
      log::error("Failed to upload BSDF energy-compensation GPU buffer %s (%u)", debug_name, static_cast<uint32_t>(update_result));
      rhi.device().destroy_buffer(result.handle);
      return false;
    }
  }

  out_buffer = result.handle;
  return true;
}

bool gpu_create_params_buffer(RHIContext& rhi, const EnergyCompensationGpuParams& params, RHIBuffer& out_buffer, const char* debug_name) {
  return gpu_create_storage_buffer(rhi, &params, sizeof(params), RHIBufferUsage::TransferDst, true, out_buffer, debug_name);
}

bool gpu_create_float4_output_buffer(RHIContext& rhi, uint64_t element_count, RHIBuffer& out_buffer, const char* debug_name) {
  return gpu_create_storage_buffer(rhi, nullptr, element_count * sizeof(float4), RHIBufferUsage::TransferSrc, false, out_buffer, debug_name);
}

void gpu_destroy_buffer_if_valid(RHIContext& rhi, RHIBuffer& buffer) {
  if (buffer.valid()) {
    rhi.device().destroy_buffer(buffer);
    buffer = {};
  }
}

void gpu_destroy_energy_compensation_buffers(RHIContext& rhi, EnergyCompensationGpuBuffers& buffers) {
  gpu_destroy_buffer_if_valid(rhi, buffers.params_directional);
  gpu_destroy_buffer_if_valid(rhi, buffers.params_average);
  gpu_destroy_buffer_if_valid(rhi, buffers.directional);
  gpu_destroy_buffer_if_valid(rhi, buffers.average);
  gpu_destroy_buffer_if_valid(rhi, buffers.geometric);
  gpu_destroy_buffer_if_valid(rhi, buffers.geometric_average);
  gpu_destroy_buffer_if_valid(rhi, buffers.conductor_fms);
  gpu_destroy_buffer_if_valid(rhi, buffers.probability);
  gpu_destroy_buffer_if_valid(rhi, buffers.total);
}

bool gpu_create_float4_readback_buffer(RHIContext& rhi, uint64_t element_count, RHIBuffer& out_buffer, const char* debug_name) {
  const uint64_t size = element_count * sizeof(float4);
  RHIBufferDesc desc = {};
  desc.size = size;
  desc.usage = RHIBufferUsage::TransferDst;
  desc.host_visible = true;

  auto result = rhi.device().create_buffer(desc);
  if ((result.result != RHIResult::Success) || (result.handle.valid() == false)) {
    log::error("Failed to create BSDF energy-compensation GPU readback buffer %s (%u)", debug_name, static_cast<uint32_t>(result.result));
    return false;
  }

  out_buffer = result.handle;
  return true;
}

bool gpu_read_float4_output_buffer(RHIContext& rhi, RHIBuffer buffer, uint64_t element_count, std::vector<float4>& out_pixels, const char* debug_name) {
  if (buffer.valid() == false) {
    log::error("Invalid BSDF energy-compensation GPU output buffer %s", debug_name);
    return false;
  }

  const uint64_t size = element_count * sizeof(float4);
  RHIBuffer readback_buffer = {};
  RHICommandBuffer cmd = {};
  bool success = false;

  do {
    if (gpu_create_float4_readback_buffer(rhi, element_count, readback_buffer, debug_name) == false) {
      break;
    }

    cmd = rhi.get_command_buffer();
    if (cmd.valid() == false) {
      log::error("Failed to acquire command buffer for BSDF energy-compensation GPU readback %s", debug_name);
      break;
    }

    rhi.command_buffer_begin(cmd);
    rhi.cmd_buffer_barrier(cmd, buffer, RHIResourceState::General, RHIResourceState::TransferSrc);
    rhi.cmd_copy_buffer(cmd, buffer, readback_buffer, size);
    rhi.command_buffer_end(cmd);
    rhi.submit_command_buffer({cmd});

    const RHIResult wait_result = rhi.wait_idle();
    if (wait_result != RHIResult::Success) {
      log::error("Failed to wait for BSDF energy-compensation GPU readback %s (%u)", debug_name, static_cast<uint32_t>(wait_result));
      break;
    }

    out_pixels.resize(static_cast<size_t>(element_count));
    const RHIResult read_result = rhi.device().read_buffer(readback_buffer, out_pixels.data(), size);
    if (read_result != RHIResult::Success) {
      log::error("Failed to read BSDF energy-compensation GPU buffer %s (%u)", debug_name, static_cast<uint32_t>(read_result));
      out_pixels.clear();
      break;
    }

    success = true;
  } while (false);

  if (cmd.valid()) {
    rhi.destroy_command_buffer(cmd);
  }
  gpu_destroy_buffer_if_valid(rhi, readback_buffer);
  return success;
}

bool gpu_init_energy_compensation_pipeline(RHIContext& rhi, EnergyCompensationGpuPipeline& out_pipeline) {
  if (out_pipeline.pipeline.valid()) {
    return true;
  }
  if (out_pipeline.initialization_failed) {
    return false;
  }

  auto compilation = ShaderCompiler::instance().compile("shaders/bsdf_energy_compensation.hlsl", {{"main", RHIShaderStage::Compute}}, {}, rhi.backend());
  if (compilation.result != RHIResult::Success) {
    log::error("Failed to compile BSDF energy-compensation GPU shader: %s", compilation.error_message.c_str());
    out_pipeline.initialization_failed = true;
    return false;
  }
  if (compilation.binaries.empty()) {
    log::error("BSDF energy-compensation GPU shader compilation returned no binaries");
    out_pipeline.initialization_failed = true;
    return false;
  }

  RHIComputePipelineDesc desc = rhi.device().make_compute_pipeline_desc(compilation.binaries[0]);
  auto pipeline_result = rhi.device().create_compute_pipeline(desc);
  if ((pipeline_result.result != RHIResult::Success) || (pipeline_result.handle.valid() == false)) {
    log::error("Failed to create BSDF energy-compensation GPU pipeline (%u)", static_cast<uint32_t>(pipeline_result.result));
    out_pipeline.initialization_failed = true;
    return false;
  }

  out_pipeline.pipeline = pipeline_result.handle;
  return true;
}

bool gpu_dispatch_energy_compensation(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const EnergyCompensationGpuBuffers& buffers, uint32_t width,
  uint32_t height, bool run_average_pass) {
  if (gpu_init_energy_compensation_pipeline(rhi, pipeline) == false) {
    return false;
  }

  RHICommandBuffer cmd = rhi.get_command_buffer();
  if (cmd.valid() == false) {
    log::error("Failed to acquire command buffer for BSDF energy-compensation GPU generation");
    return false;
  }

  rhi.command_buffer_begin(cmd);
  rhi.cmd_set_pipeline(cmd, pipeline.pipeline);

  EnergyCompensationGpuPushConstants pc = {};
  pc.params_buffer_index = get_bindless_descriptor_index(buffers.params_directional);
  rhi.cmd_push_constants(cmd, &pc, sizeof(pc), 0);
  RHIDispatchDesc dispatch = {};
  dispatch.group_count_x = (width + 7u) / 8u;
  dispatch.group_count_y = (height + 7u) / 8u;
  dispatch.group_count_z = 1u;
  rhi.cmd_dispatch(cmd, dispatch);

  if (run_average_pass) {
    rhi.cmd_buffer_barrier(cmd, buffers.directional, RHIResourceState::General, RHIResourceState::General);
    if (buffers.geometric.valid()) {
      rhi.cmd_buffer_barrier(cmd, buffers.geometric, RHIResourceState::General, RHIResourceState::General);
    }
    if (buffers.total.valid()) {
      rhi.cmd_buffer_barrier(cmd, buffers.total, RHIResourceState::General, RHIResourceState::General);
    }

    pc.params_buffer_index = get_bindless_descriptor_index(buffers.params_average);
    rhi.cmd_push_constants(cmd, &pc, sizeof(pc), 0);
    RHIDispatchDesc average_dispatch = {};
    average_dispatch.group_count_x = 8u;
    average_dispatch.group_count_y = 1u;
    average_dispatch.group_count_z = 1u;
    rhi.cmd_dispatch(cmd, average_dispatch);
  }

  rhi.command_buffer_end(cmd);
  rhi.submit_command_buffer({cmd});
  const RHIResult wait_result = rhi.wait_idle();
  rhi.destroy_command_buffer(cmd);
  if (wait_result != RHIResult::Success) {
    log::error("Failed to wait for BSDF energy-compensation GPU generation (%u)", static_cast<uint32_t>(wait_result));
    return false;
  }

  return true;
}

void fill_gpu_refractive_index_integrated(const SceneData& data, const RefractiveIndex& refractive_index, float4& eta, float4& k) {
  const SpectralQuery spect = {};
  const RefractiveIndexSample value = sample_refractive_index(data, refractive_index, spect);
  eta = spectral_response_to_gpu_float4(value.eta);
  k = spectral_response_to_gpu_float4(value.k);
}

void fill_gpu_refractive_index_spectral(const SceneData& data, const RefractiveIndex& refractive_index, uint32_t wavelength_group_index, float4& eta, float4& k,
  float4& wavelengths) {
  eta = {};
  k = {};
  wavelengths = {};
  for (uint32_t channel = 0u; channel < kEnergyCompensationSpectralWavelengthGroupSize; ++channel) {
    const uint32_t wavelength_index = wavelength_group_index * kEnergyCompensationSpectralWavelengthGroupSize + channel;
    const SpectralQuery spect = spectral_cache_query(wavelength_index);
    const RefractiveIndexSample value = sample_refractive_index(data, refractive_index, spect);
    set_gpu_float4_channel(eta, channel, spectral_response_monochromatic(value.eta));
    set_gpu_float4_channel(k, channel, spectral_response_monochromatic(value.k));
    set_gpu_float4_channel(wavelengths, channel, spect.wavelength);
  }
}

EnergyCompensationGpuParams make_gpu_params(const SceneData& data, const Material& material, uint32_t cache_mode, uint32_t pass_kind, uint32_t thinfilm_slice_index,
  uint32_t thinfilm_slice_count, uint32_t wavelength_group_index) {
  EnergyCompensationGpuParams result = {};
  result.cache_mode = cache_mode;
  result.wavelength_group_index = wavelength_group_index;
  result.thinfilm_slice_index = thinfilm_slice_index;
  result.thinfilm_slice_count = thinfilm_slice_count;
  result.pass_kind = pass_kind;
  result.sample_count = kEnergyCompensationSampleCount;
  result.multisample_count = kEnergyCompensationDielectricMultiScatterSampleCount;

  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    fill_gpu_refractive_index_spectral(data, material.ext_ior, wavelength_group_index, result.ext_eta, result.ext_k, result.wavelengths);
    float4 unused_wavelengths = {};
    fill_gpu_refractive_index_spectral(data, material.int_ior, wavelength_group_index, result.int_eta, result.int_k, unused_wavelengths);
    const SpectralQuery spect = spectral_cache_query(wavelength_group_index * kEnergyCompensationSpectralWavelengthGroupSize);
    const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, thinfilm_slice_index, thinfilm_slice_count);
    result.thinfilm_thickness = thinfilm.thickness;
    result.film_cls = thinfilm.ior.cls;
    result.thinfilm_weight = thinfilm.weight;
    fill_gpu_refractive_index_spectral(data, material.thinfilm.ior, wavelength_group_index, result.film_eta, result.film_k, unused_wavelengths);
    result.film_k = float4{};
  } else {
    fill_gpu_refractive_index_integrated(data, material.ext_ior, result.ext_eta, result.ext_k);
    fill_gpu_refractive_index_integrated(data, material.int_ior, result.int_eta, result.int_k);
    const SpectralQuery spect = {};
    const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, thinfilm_slice_index, thinfilm_slice_count);
    result.thinfilm_thickness = thinfilm.thickness;
    result.film_cls = thinfilm.ior.cls;
    result.thinfilm_weight = thinfilm.weight;
    result.film_eta = spectral_response_to_gpu_float4(thinfilm.ior.eta);
    result.film_k = spectral_response_to_gpu_float4(thinfilm.ior.k);
  }

  return result;
}

uint64_t hash_spectrum(const SceneData& data, uint32_t index, uint64_t seed) {
  if (index >= data.spectrum_values.size()) {
    return etx_hash64_continue(&index, sizeof(index), seed);
  }
  return etx_hash64_continue(data.spectrum_values.data() + index, sizeof(SpectralDistribution), seed);
}

uint64_t hash_refractive_index(const SceneData& data, const RefractiveIndex& refractive_index, uint64_t seed) {
  uint64_t result = etx_hash64_continue(&refractive_index.cls, sizeof(refractive_index.cls), seed);
  result = hash_spectrum(data, refractive_index.eta_index, result);
  result = hash_spectrum(data, refractive_index.k_index, result);
  return result;
}

uint64_t hash_thinfilm(const SceneData& data, const Thinfilm& thinfilm, uint64_t seed) {
  const bool enabled = bsdf_resource_thinfilm_enabled(thinfilm);
  uint64_t result = etx_hash64_continue(&enabled, sizeof(enabled), seed);
  if (enabled == false) {
    return result;
  }

  result = hash_refractive_index(data, thinfilm.ior, result);
  const float weight = clamp(thinfilm.weight, 0.0f, 1.0f);
  result = etx_hash64_continue(&weight, sizeof(weight), result);
  result = etx_hash64_continue(&thinfilm.min_thickness, sizeof(thinfilm.min_thickness), result);
  result = etx_hash64_continue(&thinfilm.max_thickness, sizeof(thinfilm.max_thickness), result);
  result = etx_hash64_continue(&thinfilm.thinkness_image, sizeof(thinfilm.thinkness_image), result);
  const uint32_t slice_count = thinfilm_lut_slice_count(thinfilm);
  result = etx_hash64_continue(&slice_count, sizeof(slice_count), result);
  return result;
}

uint64_t hash_material_interface(const SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode) {
  uint64_t result = 0u;
  result = etx_hash64_continue(&kEnergyCompensationGeneratorVersion, sizeof(kEnergyCompensationGeneratorVersion), result);
  result = etx_hash64_continue(&material_class, sizeof(material_class), result);
  result = etx_hash64_continue(&cache_mode, sizeof(cache_mode), result);
  result = etx_hash64_continue(&kEnergyCompensationConductorLutSize, sizeof(kEnergyCompensationConductorLutSize), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricLutSize, sizeof(kEnergyCompensationDielectricLutSize), result);
  result = etx_hash64_continue(&kEnergyCompensationConductorSampleCount, sizeof(kEnergyCompensationConductorSampleCount), result);
  result = etx_hash64_continue(&kEnergyCompensationSampleCount, sizeof(kEnergyCompensationSampleCount), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricMultiScatterSampleCount, sizeof(kEnergyCompensationDielectricMultiScatterSampleCount), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricBranchCount, sizeof(kEnergyCompensationDielectricBranchCount), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricAverageWidth, sizeof(kEnergyCompensationDielectricAverageWidth), result);
  const uint32_t thinfilm_slice_count = thinfilm_lut_slice_count(material.thinfilm);
  result = etx_hash64_continue(&thinfilm_slice_count, sizeof(thinfilm_slice_count), result);
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    result = etx_hash64_continue(&kEnergyCompensationSpectralWavelengthCount, sizeof(kEnergyCompensationSpectralWavelengthCount), result);
    result = etx_hash64_continue(&kEnergyCompensationSpectralWavelengthGroupSize, sizeof(kEnergyCompensationSpectralWavelengthGroupSize), result);
    result = etx_hash64_continue(&kShortestWavelength, sizeof(kShortestWavelength), result);
    result = etx_hash64_continue(&kLongestWavelength, sizeof(kLongestWavelength), result);
  }
  result = hash_refractive_index(data, material.ext_ior, result);
  result = hash_refractive_index(data, material.int_ior, result);
  result = hash_thinfilm(data, material.thinfilm, result);
  return result;
}

std::filesystem::path cache_directory() {
  return std::filesystem::path(env().file_in_cache("bsdf/energy_compensation"));
}

std::filesystem::path parity_cache_directory() {
  return cache_directory() / "parity";
}

std::string hash_string(uint64_t hash) {
  char buffer[32] = {};
  snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(hash));
  return std::string(buffer);
}

GeneratedInterfacePaths interface_paths(uint32_t material_class, uint64_t hash) {
  const std::string prefix = (material_class == MaterialClass::Conductor) ? "ec_conductor_" : "ec_dielectric_";
  const std::string key = prefix + hash_string(hash);
  const std::filesystem::path directory = cache_directory();
  return {
    directory / (key + ".exr"),
    directory / (key + "_average.exr"),
    directory / (key + "_geometric.exr"),
    directory / (key + "_geometric_average.exr"),
    directory / (key + "_conductor_fms.exr"),
    directory / (key + "_probability.exr"),
  };
}

GeneratedInterfacePaths parity_interface_paths(const char* key) {
  const std::filesystem::path directory = parity_cache_directory();
  return {
    directory / (std::string(key) + ".exr"),
    directory / (std::string(key) + "_average.exr"),
    directory / (std::string(key) + "_geometric.exr"),
    directory / (std::string(key) + "_geometric_average.exr"),
    directory / (std::string(key) + "_conductor_fms.exr"),
    directory / (std::string(key) + "_probability.exr"),
  };
}

std::filesystem::path slice_path(const std::filesystem::path& path, uint32_t slice_index) {
  if (slice_index == 0u) {
    return path;
  }

  const std::string stem = path.stem().generic_string() + "_slice_" + std::to_string(slice_index);
  return path.parent_path() / (stem + path.extension().generic_string());
}

GeneratedInterfacePaths interface_slice_paths(const GeneratedInterfacePaths& paths, uint32_t slice_index) {
  return {
    slice_path(paths.directional, slice_index),
    slice_path(paths.average, slice_index),
    slice_path(paths.geometric, slice_index),
    slice_path(paths.geometric_average, slice_index),
    slice_path(paths.conductor_fms, slice_index),
    slice_path(paths.probability, slice_index),
  };
}

bool save_exr_rgba(const std::filesystem::path& path, const std::vector<float4>& pixels, uint32_t width, uint32_t height) {
  const uint64_t expected_pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  if (pixels.size() != expected_pixel_count) {
    log::error("Invalid energy-compensation LUT dimensions for %s", path.generic_string().c_str());
    return false;
  }

  const std::filesystem::path parent_path = path.parent_path();
  if (parent_path.empty() == false) {
    std::error_code ec = {};
    std::filesystem::create_directories(parent_path, ec);
    if (ec.value() != 0) {
      log::error("Failed to create energy-compensation LUT cache directory %s", parent_path.generic_string().c_str());
      return false;
    }
  }

  const std::string file_name = path.generic_string();
  std::string error;
  if (!save_exr_image(file_name.c_str(), pixels.data(), {width, height}, &error)) {
    log::error("Failed to save energy-compensation LUT %s: %s", file_name.c_str(), error.c_str());
    return false;
  }

  return true;
}

bool load_rgba32f_lut(const std::filesystem::path& path, uint32_t width, uint32_t height, std::vector<float4>& out_pixels) {
  std::vector<uint8_t> source_data;
  uint2 dimensions = {};
  const Image::Format format = load_data(path.generic_string().c_str(), source_data, dimensions);
  if ((format != Image::Format::RGBA32F) || (dimensions.x != width) || (dimensions.y != height)) {
    log::error("Failed to load energy-compensation LUT %s", path.generic_string().c_str());
    return false;
  }

  const uint64_t pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  if (source_data.size() != pixel_count * sizeof(float4)) {
    log::error("Invalid energy-compensation LUT size %s", path.generic_string().c_str());
    return false;
  }

  out_pixels.resize(static_cast<size_t>(pixel_count));
  memcpy(out_pixels.data(), source_data.data(), source_data.size());
  return true;
}

bool load_rgba32f_lut_slices(const GeneratedInterfacePaths& paths, uint32_t width, uint32_t height, uint32_t slice_count, std::vector<float4>& out_pixels,
  const std::filesystem::path GeneratedInterfacePaths::*member) {
  const uint64_t slice_pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  out_pixels.resize(static_cast<size_t>(slice_pixel_count * slice_count));
  for (uint32_t slice_index = 0u; slice_index < slice_count; ++slice_index) {
    const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, slice_index);
    std::vector<float4> slice_pixels;
    if (load_rgba32f_lut(slice_paths_value.*member, width, height, slice_pixels) == false) {
      return false;
    }
    std::copy(slice_pixels.begin(), slice_pixels.end(), out_pixels.begin() + static_cast<size_t>(slice_pixel_count * slice_index));
  }
  return true;
}

SpectralDirectionalAlbedoResult integrate_conductor_directional(const SpectralQuery& spect, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior,
  const ThinfilmEval& thinfilm, float mu_i, float alpha) {
  SpectralDirectionalAlbedoResult result = {};
  if (mu_i <= kEpsilon) {
    return result;
  }

  const float2 alpha2 = float2(alpha, alpha);
  const float3 w_i = incident_direction_from_mu(mu_i);
  const float lambda_i = bsdf_external_ray_info_make(w_i, alpha2).Lambda;
  const float g1_i = 1.0f / (1.0f + lambda_i);
  const auto texture = spectral_response_make(spect, 1.0f);

  for (uint32_t sample_index = 0u; sample_index < kEnergyCompensationConductorSampleCount; ++sample_index) {
    const float3 m = bsdf_energy_compensated_sample_vndf_local(w_i, alpha, hammersley(sample_index, kEnergyCompensationConductorSampleCount));
    const float i_dot_m = dot(w_i, m);
    if ((m.z <= kEpsilon) || (i_dot_m <= kEpsilon)) {
      continue;
    }

    const float3 w_o = -w_i + 2.0f * m * i_dot_m;
    if (w_o.z > 0.0f) {
      const float vndf_pdf = bsdf_energy_compensated_vndf_pdf(w_i, m, alpha);
      const float raw_specular_pdf = vndf_pdf / max(kEpsilon, 4.0f * dot(w_o, m));
      const BSDFEnergyCompensatedLobe lobe = bsdf_energy_compensated_conductor_base_lobe(spect, w_i, w_o, alpha, ext_ior, int_ior, thinfilm, texture);
      if ((raw_specular_pdf > kEpsilon) && (lobe.pdf > kEpsilon)) {
        result.albedo += lobe.bsdf.integrated / raw_specular_pdf;
        const float lambda_o = bsdf_external_ray_info_make(w_o, alpha2).Lambda;
        const float d = bsdf_external_d_ggx(m, alpha2);
        const float g2 = 1.0f / (1.0f + lambda_i + lambda_o);
        result.geometric_albedo += (d * g2 / (4.0f * w_i.z)) / raw_specular_pdf;
      }
      if (g1_i > kEpsilon) {
        result.visible_probability += 1.0f;
      }
    }
  }

  const float inv_sample_count = 1.0f / static_cast<float>(kEnergyCompensationConductorSampleCount);
  result.albedo = saturate(result.albedo * inv_sample_count);
  result.geometric_albedo = saturate(result.geometric_albedo * inv_sample_count);
  result.visible_probability = saturate(result.visible_probability * inv_sample_count);
  return result;
}

DielectricDirectionalAlbedoResult integrate_dielectric_directional(const SpectralQuery& spect, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior,
  const ThinfilmEval& thinfilm, bool incident_outside, float mu_i, float alpha) {
  DielectricDirectionalAlbedoResult result = {};
  if (mu_i <= kEpsilon) {
    return result;
  }

  const float3 w_i = incident_direction_from_mu(mu_i);
  const float eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));
  const auto texture = spectral_response_make(spect, 1.0f);
  const uint32_t incident_side = dielectric_side(incident_outside);
  const uint32_t opposite_side = 1u - incident_side;

  const bool no_thinfilm = (thinfilm.weight <= 0.0f) || (thinfilm.thickness <= 0.0f) || spectral_response_is_zero(thinfilm.ior.eta);
  if ((no_thinfilm) && (abs(eta - 1.0f) <= (16.0f * kEpsilon))) {
    result.branch_albedo[opposite_side] = float3(1.0f, 1.0f, 1.0f);
    result.branch_visible_probability[opposite_side] = 1.0f;
    result.visible_probability = 1.0f;
    return result;
  }

  for (uint32_t sample_index = 0u; sample_index < kEnergyCompensationSampleCount; ++sample_index) {
    const float3 m = bsdf_energy_compensated_sample_vndf_local(w_i, alpha, hammersley(sample_index, kEnergyCompensationSampleCount));
    const float i_dot_m = dot(w_i, m);
    if (i_dot_m <= kEpsilon) {
      continue;
    }

    const auto fresnel = bsdf_fresnel_calculate(spect, i_dot_m, ext_ior, int_ior, thinfilm);
    const float fresnel_probability = spectral_response_monochromatic(fresnel);
    const float cos_theta_t2 = 1.0f - (1.0f - i_dot_m * i_dot_m) / (eta * eta);

    const float3 w_o_r = -w_i + 2.0f * m * i_dot_m;
    if (w_o_r.z > 0.0f) {
      const auto lobe = bsdf_energy_compensated_dielectric_base_lobe(spect, w_i, w_o_r, alpha, ext_ior, int_ior, thinfilm, texture);
      if ((fresnel_probability > kEpsilon) && (lobe.pdf > kEpsilon)) {
        result.branch_albedo[incident_side] += lobe.bsdf.integrated * (fresnel_probability / lobe.pdf);
        result.branch_visible_probability[incident_side] += fresnel_probability;
      }
    }

    if ((fresnel_probability < 1.0f) && (cos_theta_t2 > 0.0f)) {
      float3 w_o_t = normalize(bsdf_external_refract(w_i, m, eta));
      w_o_t.z = -abs(w_o_t.z);
      if (w_o_t.z < 0.0f) {
        const auto lobe = bsdf_energy_compensated_dielectric_base_lobe(spect, w_i, w_o_t, alpha, ext_ior, int_ior, thinfilm, texture);
        const float transmission_probability = 1.0f - fresnel_probability;
        if ((transmission_probability > kEpsilon) && (lobe.pdf > kEpsilon)) {
          result.branch_albedo[opposite_side] += lobe.bsdf.integrated * (transmission_probability / lobe.pdf);
          result.branch_visible_probability[opposite_side] += transmission_probability;
        }
      }
    }
  }

  const float inv_sample_count = 1.0f / static_cast<float>(kEnergyCompensationSampleCount);
  result.branch_albedo[0] = saturate(result.branch_albedo[0] * inv_sample_count);
  result.branch_albedo[1] = saturate(result.branch_albedo[1] * inv_sample_count);
  result.branch_visible_probability[0] = saturate(result.branch_visible_probability[0] * inv_sample_count);
  result.branch_visible_probability[1] = saturate(result.branch_visible_probability[1] * inv_sample_count);
  result.visible_probability = saturate(result.branch_visible_probability[0] + result.branch_visible_probability[1]);
  return result;
}

DielectricDirectionalAlbedoResult integrate_dielectric_total_directional(const SpectralQuery& spect, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior,
  const ThinfilmEval& thinfilm, bool incident_outside, float mu_i, float alpha) {
  DielectricDirectionalAlbedoResult result = {};
  if (mu_i <= kEpsilon) {
    return result;
  }

  const float2 alpha2 = float2(alpha, alpha);
  const float3 w_i = incident_direction_from_mu(mu_i);
  const uint32_t incident_side = dielectric_side(incident_outside);
  const uint32_t opposite_side = 1u - incident_side;
  const float eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));

  const bool no_thinfilm = (thinfilm.weight <= 0.0f) || (thinfilm.thickness <= 0.0f) || spectral_response_is_zero(thinfilm.ior.eta);
  if ((no_thinfilm) && (abs(eta - 1.0f) <= (16.0f * kEpsilon))) {
    result.branch_albedo[opposite_side] = float3(1.0f, 1.0f, 1.0f);
    result.visible_probability = 1.0f;
    return result;
  }

  for (uint32_t sample_index = 0u; sample_index < kEnergyCompensationDielectricMultiScatterSampleCount; ++sample_index) {
    BSDFExternalRayInfo ray = bsdf_external_ray_info_make(-w_i, alpha2);
    ray = bsdf_external_ray_info_update_height(ray, 1.0f);
    bool ray_outside = true;
    bool valid = true;
    uint32_t scattering_order = 0u;
    uint32_t dimension = 0u;

    while (valid) {
      const float sampled_height = bsdf_external_sample_height(ray, quasi_random(sample_index, dimension));
      dimension += 1u;
      if (sampled_height == kMaxFloat) {
        break;
      }

      ray = bsdf_external_ray_info_update_height(ray, sampled_height);

      const float2 rnd_slope = quasi_random_2d(sample_index, dimension);
      dimension += 2u;
      const float rnd_reflection = quasi_random(sample_index, dimension);
      dimension += 1u;
      const RefractiveIndexSample& phase_ext_ior = ray_outside ? ext_ior : int_ior;
      const RefractiveIndexSample& phase_int_ior = ray_outside ? int_ior : ext_ior;
      const BSDFExternalDielectricSample sample =
        bsdf_external_sample_phase_function_dielectric(spect, rnd_slope, rnd_reflection, -ray.w, alpha2, phase_ext_ior, phase_int_ior, thinfilm);

      if (sample.reflection) {
        ray = bsdf_external_ray_info_update_direction(ray, sample.w_o, alpha2);
        ray = bsdf_external_ray_info_update_height(ray, ray.h);
      } else {
        ray_outside = (ray_outside == false);
        ray = bsdf_external_ray_info_update_direction(ray, -sample.w_o, alpha2);
        ray = bsdf_external_ray_info_update_height(ray, -ray.h);
      }

      scattering_order += 1u;
      if ((scattering_order > kBSDFExternalScatteringOrderMax) || (ray.h != ray.h) || (ray.w.x != ray.w.x)) {
        valid = false;
      }
    }

    if (valid == false) {
      continue;
    }

    const float3 local_w_o = ray_outside ? ray.w : -ray.w;
    const uint32_t outgoing_side = (local_w_o.z > 0.0f) ? incident_side : opposite_side;
    result.branch_albedo[outgoing_side] += float3(1.0f, 1.0f, 1.0f);
    result.visible_probability += 1.0f;
  }

  const float inv_sample_count = 1.0f / static_cast<float>(kEnergyCompensationDielectricMultiScatterSampleCount);
  result.branch_albedo[0] = saturate(result.branch_albedo[0] * inv_sample_count);
  result.branch_albedo[1] = saturate(result.branch_albedo[1] * inv_sample_count);
  result.visible_probability = saturate(result.visible_probability * inv_sample_count);
  return result;
}

float3 integrate_average(const std::vector<SpectralDirectionalAlbedoResult>& directional, uint32_t alpha_index, uint32_t lut_size, uint32_t side_offset) {
  float3 total = {};
  const float h = 1.0f / static_cast<float>(lut_size - 1u);
  for (uint32_t mu_index = 0u; mu_index < (lut_size - 1u); ++mu_index) {
    const float mu = static_cast<float>(mu_index) * h;
    const float weight_0 = h * mu + h * h / 3.0f;
    const float weight_1 = h * mu + 2.0f * h * h / 3.0f;
    const float3 value_0 = directional[side_offset + alpha_index * lut_size + mu_index].albedo;
    const float3 value_1 = directional[side_offset + alpha_index * lut_size + mu_index + 1u].albedo;
    total += value_0 * weight_0 + value_1 * weight_1;
  }
  return saturate(total);
}

float3 integrate_dielectric_branch_average(const std::vector<DielectricDirectionalAlbedoResult>& directional, uint32_t alpha_index, uint32_t incident_side,
  uint32_t outgoing_side) {
  float3 total = {};
  const float h = 1.0f / static_cast<float>(kEnergyCompensationDielectricLutSize - 1u);
  const uint32_t side_entry_count = kEnergyCompensationDielectricLutSize * kEnergyCompensationDielectricLutSize;
  const uint32_t side_offset = incident_side * side_entry_count;
  for (uint32_t mu_index = 0u; mu_index < (kEnergyCompensationDielectricLutSize - 1u); ++mu_index) {
    const float mu = static_cast<float>(mu_index) * h;
    const float weight_0 = h * mu + h * h / 3.0f;
    const float weight_1 = h * mu + 2.0f * h * h / 3.0f;
    const float3 value_0 = directional[side_offset + alpha_index * kEnergyCompensationDielectricLutSize + mu_index].branch_albedo[outgoing_side];
    const float3 value_1 = directional[side_offset + alpha_index * kEnergyCompensationDielectricLutSize + mu_index + 1u].branch_albedo[outgoing_side];
    total += value_0 * weight_0 + value_1 * weight_1;
  }
  return saturate(total);
}

void clamp_dielectric_residual_row(float3 residual_average[kEnergyCompensationDielectricBranchCount], const float3 side_residual[2]) {
  for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
    const uint32_t branch_0 = dielectric_branch_index(incident_side, 0u);
    const uint32_t branch_1 = dielectric_branch_index(incident_side, 1u);
    const float3 residual_sum = residual_average[branch_0] + residual_average[branch_1];
    const float3 scale = min(float3(1.0f, 1.0f, 1.0f), side_residual[incident_side] / max(float3(kEpsilon, kEpsilon, kEpsilon), residual_sum));
    residual_average[branch_0] *= scale;
    residual_average[branch_1] *= scale;
  }
}

float integrate_geometric_average(const std::vector<SpectralDirectionalAlbedoResult>& directional, uint32_t alpha_index, uint32_t lut_size) {
  float total = 0.0f;
  const float h = 1.0f / static_cast<float>(lut_size - 1u);
  for (uint32_t mu_index = 0u; mu_index < (lut_size - 1u); ++mu_index) {
    const float mu = static_cast<float>(mu_index) * h;
    const float weight_0 = h * mu + h * h / 3.0f;
    const float weight_1 = h * mu + 2.0f * h * h / 3.0f;
    const float value_0 = directional[alpha_index * lut_size + mu_index].geometric_albedo;
    const float value_1 = directional[alpha_index * lut_size + mu_index + 1u].geometric_albedo;
    total += value_0 * weight_0 + value_1 * weight_1;
  }
  return saturate(total);
}

bool generate_conductor_interface(const GeneratedInterfacePaths& paths, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior, const ThinfilmEval& thinfilm,
  TaskScheduler& scheduler) {
  const SpectralQuery spect = {};
  const uint32_t entry_count = kEnergyCompensationConductorLutSize * kEnergyCompensationConductorLutSize;
  std::vector<SpectralDirectionalAlbedoResult> directional(entry_count);
  scheduler.execute(entry_count, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t alpha_index = index / kEnergyCompensationConductorLutSize;
      const uint32_t mu_index = index - alpha_index * kEnergyCompensationConductorLutSize;
      directional[index] = integrate_conductor_directional(spect, ext_ior, int_ior, thinfilm, mu_parameter(mu_index, kEnergyCompensationConductorLutSize),
        alpha_parameter(alpha_index, kEnergyCompensationConductorLutSize));
    }
  });

  std::vector<float4> image(entry_count, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> geometric_image(entry_count, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> geometric_average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> conductor_fms(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationConductorLutSize; ++alpha_index) {
    const float3 average_value = integrate_average(directional, alpha_index, kEnergyCompensationConductorLutSize, 0u);
    average[alpha_index] = float4(average_value.x, average_value.y, average_value.z, 1.0f);
    for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationConductorLutSize; ++mu_index) {
      const uint32_t index = alpha_index * kEnergyCompensationConductorLutSize + mu_index;
      const SpectralDirectionalAlbedoResult& value = directional[index];
      image[index] = float4(value.albedo.x, value.albedo.y, value.albedo.z, value.visible_probability);
      geometric_image[index] = float4(value.geometric_albedo, value.visible_probability, 0.0f, 1.0f);
    }
    const float geometric_average_value = integrate_geometric_average(directional, alpha_index, kEnergyCompensationConductorLutSize);
    geometric_average[alpha_index] = float4(geometric_average_value, 0.0f, 0.0f, 1.0f);
    const ::SpectralResponse fms = bsdf_energy_compensated_conductor_fms(spect, ext_ior, int_ior, thinfilm, geometric_average_value);
    conductor_fms[alpha_index] = float4(fms.integrated.x, fms.integrated.y, fms.integrated.z, 1.0f);
  }

  return save_exr_rgba(paths.directional, image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationConductorLutSize, 1u) &&
         save_exr_rgba(paths.geometric, geometric_image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
         save_exr_rgba(paths.geometric_average, geometric_average, kEnergyCompensationConductorLutSize, 1u) &&
         save_exr_rgba(paths.conductor_fms, conductor_fms, kEnergyCompensationConductorLutSize, 1u);
}

bool generate_conductor_geometric_interface(const GeneratedInterfacePaths& paths, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior,
  const ThinfilmEval& thinfilm, TaskScheduler& scheduler) {
  const SpectralQuery spect = {};
  const uint32_t entry_count = kEnergyCompensationConductorLutSize * kEnergyCompensationConductorLutSize;
  std::vector<SpectralDirectionalAlbedoResult> directional(entry_count);
  scheduler.execute(entry_count, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t alpha_index = index / kEnergyCompensationConductorLutSize;
      const uint32_t mu_index = index - alpha_index * kEnergyCompensationConductorLutSize;
      directional[index] = integrate_conductor_directional(spect, ext_ior, int_ior, thinfilm, mu_parameter(mu_index, kEnergyCompensationConductorLutSize),
        alpha_parameter(alpha_index, kEnergyCompensationConductorLutSize));
    }
  });

  std::vector<float4> geometric_image(entry_count, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> geometric_average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationConductorLutSize; ++alpha_index) {
    for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationConductorLutSize; ++mu_index) {
      const uint32_t index = alpha_index * kEnergyCompensationConductorLutSize + mu_index;
      const SpectralDirectionalAlbedoResult& value = directional[index];
      geometric_image[index] = float4(value.geometric_albedo, value.visible_probability, 0.0f, 1.0f);
    }
    const float geometric_average_value = integrate_geometric_average(directional, alpha_index, kEnergyCompensationConductorLutSize);
    geometric_average[alpha_index] = float4(geometric_average_value, 0.0f, 0.0f, 1.0f);
  }

  return save_exr_rgba(paths.geometric, geometric_image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
         save_exr_rgba(paths.geometric_average, geometric_average, kEnergyCompensationConductorLutSize, 1u);
}

bool generate_conductor_interface_spectral(const SceneData& data, const Material& material, const GeneratedInterfacePaths& paths, uint32_t thinfilm_slice_index,
  uint32_t thinfilm_slice_count, uint32_t wavelength_group_index, TaskScheduler& scheduler) {
  const uint32_t entry_count = kEnergyCompensationConductorLutSize * kEnergyCompensationConductorLutSize;
  std::vector<float4> image(entry_count, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> conductor_fms(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});

  for (uint32_t channel = 0u; channel < kEnergyCompensationSpectralWavelengthGroupSize; ++channel) {
    const uint32_t wavelength_index = wavelength_group_index * kEnergyCompensationSpectralWavelengthGroupSize + channel;
    const SpectralQuery spect = spectral_cache_query(wavelength_index);
    const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
    const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
    const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, thinfilm_slice_index, thinfilm_slice_count);
    std::vector<SpectralDirectionalAlbedoResult> directional(entry_count);
    scheduler.execute(entry_count, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
      (void)thread_id;
      for (uint32_t index = begin; index < end; ++index) {
        const uint32_t alpha_index = index / kEnergyCompensationConductorLutSize;
        const uint32_t mu_index = index - alpha_index * kEnergyCompensationConductorLutSize;
        directional[index] = integrate_conductor_directional(spect, ext_ior, int_ior, thinfilm, mu_parameter(mu_index, kEnergyCompensationConductorLutSize),
          alpha_parameter(alpha_index, kEnergyCompensationConductorLutSize));
      }
    });

    for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationConductorLutSize; ++alpha_index) {
      const float3 average_value = integrate_average(directional, alpha_index, kEnergyCompensationConductorLutSize, 0u);
      set_float4_channel(average[alpha_index], channel, average_value.x);
      for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationConductorLutSize; ++mu_index) {
        const uint32_t index = alpha_index * kEnergyCompensationConductorLutSize + mu_index;
        set_float4_channel(image[index], channel, directional[index].albedo.x);
      }

      const float geometric_average_value = integrate_geometric_average(directional, alpha_index, kEnergyCompensationConductorLutSize);
      const ::SpectralResponse fms = bsdf_energy_compensated_conductor_fms(spect, ext_ior, int_ior, thinfilm, geometric_average_value);
      set_float4_channel(conductor_fms[alpha_index], channel, spectral_response_monochromatic(fms));
    }
  }

  return save_exr_rgba(paths.directional, image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationConductorLutSize, 1u) &&
         save_exr_rgba(paths.conductor_fms, conductor_fms, kEnergyCompensationConductorLutSize, 1u);
}

bool generate_dielectric_interface(const GeneratedInterfacePaths& paths, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior, const ThinfilmEval& thinfilm,
  TaskScheduler& scheduler) {
  const SpectralQuery spect = {};
  const uint32_t side_entry_count = kEnergyCompensationDielectricLutSize * kEnergyCompensationDielectricLutSize;
  std::vector<DielectricDirectionalAlbedoResult> directional(side_entry_count * 2u);
  std::vector<DielectricDirectionalAlbedoResult> total_directional(side_entry_count * 2u);
  scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t side = index / side_entry_count;
      const uint32_t side_index = index - side * side_entry_count;
      const uint32_t alpha_index = side_index / kEnergyCompensationDielectricLutSize;
      const uint32_t mu_index = side_index - alpha_index * kEnergyCompensationDielectricLutSize;
      const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
      const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
      directional[index] = integrate_dielectric_directional(spect, source_ior, target_ior, thinfilm, side == 0u, mu_parameter(mu_index, kEnergyCompensationDielectricLutSize),
        alpha_parameter(alpha_index, kEnergyCompensationDielectricLutSize));
    }
  });
  scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t side = index / side_entry_count;
      const uint32_t side_index = index - side * side_entry_count;
      const uint32_t alpha_index = side_index / kEnergyCompensationDielectricLutSize;
      const uint32_t mu_index = side_index - alpha_index * kEnergyCompensationDielectricLutSize;
      const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
      const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
      total_directional[index] = integrate_dielectric_total_directional(spect, source_ior, target_ior, thinfilm, side == 0u,
        mu_parameter(mu_index, kEnergyCompensationDielectricLutSize), alpha_parameter(alpha_index, kEnergyCompensationDielectricLutSize));
    }
  });

  const uint32_t width = kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize;
  std::vector<float4> image(width * kEnergyCompensationDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationDielectricAverageWidth * kEnergyCompensationDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationDielectricLutSize; ++alpha_index) {
    float3 single_average[kEnergyCompensationDielectricBranchCount] = {};
    float3 total_average[kEnergyCompensationDielectricBranchCount] = {};
    float3 residual_average[kEnergyCompensationDielectricBranchCount] = {};
    float3 residual_coefficient[kEnergyCompensationDielectricBranchCount] = {};

    for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
      for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
        const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
        single_average[branch] = integrate_dielectric_branch_average(directional, alpha_index, incident_side, outgoing_side);
        total_average[branch] = integrate_dielectric_branch_average(total_directional, alpha_index, incident_side, outgoing_side);
      }
    }

    float3 side_residual[2] = {};
    for (uint32_t side = 0u; side < 2u; ++side) {
      const uint32_t branch_0 = dielectric_branch_index(side, 0u);
      const uint32_t branch_1 = dielectric_branch_index(side, 1u);
      const float3 row_single = saturate(single_average[branch_0] + single_average[branch_1]);
      side_residual[side] = max(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f) - row_single);
    }

    for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
      for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
        const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
        residual_average[branch] = max(float3(0.0f, 0.0f, 0.0f), total_average[branch] - single_average[branch]);
      }
    }

    clamp_dielectric_residual_row(residual_average, side_residual);

    for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
      for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
        const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
        const float3 denominator = max(float3(kEpsilon, kEpsilon, kEpsilon), side_residual[incident_side] * side_residual[outgoing_side]);
        residual_coefficient[branch] = residual_average[branch] / denominator;
      }
    }

#if ETX_DEBUG
    const float validation_tolerance = 4.0f / sqrt(static_cast<float>(kEnergyCompensationSampleCount));
    for (uint32_t side = 0u; side < 2u; ++side) {
      const uint32_t branch_0 = dielectric_branch_index(side, 0u);
      const uint32_t branch_1 = dielectric_branch_index(side, 1u);
      const float3 single_row = single_average[branch_0] + single_average[branch_1];
      const float3 model_total_row = total_average[branch_0] + total_average[branch_1];
      const float3 total_row = single_row + residual_average[branch_0] + residual_average[branch_1];
      if ((isfinite(single_row.x) == false) || (isfinite(single_row.y) == false) || (isfinite(single_row.z) == false) || (isfinite(model_total_row.x) == false) ||
          (isfinite(model_total_row.y) == false) || (isfinite(model_total_row.z) == false) || (isfinite(total_row.x) == false) || (isfinite(total_row.y) == false) ||
          (isfinite(total_row.z) == false) || (single_row.x > (1.0f + validation_tolerance)) ||
          (single_row.y > (1.0f + validation_tolerance)) || (single_row.z > (1.0f + validation_tolerance)) || (total_row.x > (1.0f + validation_tolerance)) ||
          (total_row.y > (1.0f + validation_tolerance)) || (total_row.z > (1.0f + validation_tolerance)) ||
          (model_total_row.x > (1.0f + validation_tolerance)) || (model_total_row.y > (1.0f + validation_tolerance)) ||
          (model_total_row.z > (1.0f + validation_tolerance))) {
        log::error(
          "Invalid dielectric energy-compensation row alpha %u side %u single %.6f %.6f %.6f model %.6f %.6f %.6f total %.6f %.6f %.6f tolerance %.6f", alpha_index, side,
          single_row.x, single_row.y, single_row.z, model_total_row.x, model_total_row.y, model_total_row.z, total_row.x, total_row.y, total_row.z, validation_tolerance);
        return false;
      }
    }
    for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
      const float3 coefficient = residual_coefficient[branch];
      if ((isfinite(coefficient.x) == false) || (isfinite(coefficient.y) == false) || (isfinite(coefficient.z) == false) || (coefficient.x < 0.0f) || (coefficient.y < 0.0f) ||
          (coefficient.z < 0.0f)) {
        log::error("Invalid dielectric energy-compensation coefficient alpha %u branch %u coefficient %.6f %.6f %.6f", alpha_index, branch, coefficient.x, coefficient.y,
          coefficient.z);
        return false;
      }
    }
#endif

    for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
      const float3 average_value = saturate(single_average[branch]);
      average[alpha_index * kEnergyCompensationDielectricAverageWidth + branch] = float4(average_value.x, average_value.y, average_value.z, 1.0f);
      const float3 coefficient = max(float3(0.0f, 0.0f, 0.0f), residual_coefficient[branch]);
      average[alpha_index * kEnergyCompensationDielectricAverageWidth + 4u + branch] = float4(coefficient.x, coefficient.y, coefficient.z, 1.0f);
    }

    for (uint32_t side = 0u; side < 2u; ++side) {
      const uint32_t side_offset = side * side_entry_count;
      for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationDielectricLutSize; ++mu_index) {
        const uint32_t source_index = side_offset + alpha_index * kEnergyCompensationDielectricLutSize + mu_index;
        const DielectricDirectionalAlbedoResult& value = directional[source_index];
        for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
          const uint32_t branch = dielectric_branch_index(side, outgoing_side);
          const uint32_t x = branch * kEnergyCompensationDielectricLutSize + mu_index;
          const float3 branch_value = value.branch_albedo[outgoing_side];
          image[alpha_index * width + x] = float4(branch_value.x, branch_value.y, branch_value.z, value.branch_visible_probability[outgoing_side]);
        }
      }
    }
  }

  return save_exr_rgba(paths.directional, image, width, kEnergyCompensationDielectricLutSize) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationDielectricAverageWidth, kEnergyCompensationDielectricLutSize);
}

bool generate_dielectric_interface_spectral(const SceneData& data, const Material& material, const GeneratedInterfacePaths& paths, uint32_t thinfilm_slice_index,
  uint32_t thinfilm_slice_count, uint32_t wavelength_group_index, TaskScheduler& scheduler) {
  const uint32_t side_entry_count = kEnergyCompensationDielectricLutSize * kEnergyCompensationDielectricLutSize;
  const uint32_t width = kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize;
  std::vector<float4> image(width * kEnergyCompensationDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> probability(width * kEnergyCompensationDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationDielectricAverageWidth * kEnergyCompensationDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});

  for (uint32_t channel = 0u; channel < kEnergyCompensationSpectralWavelengthGroupSize; ++channel) {
    const uint32_t wavelength_index = wavelength_group_index * kEnergyCompensationSpectralWavelengthGroupSize + channel;
    const SpectralQuery spect = spectral_cache_query(wavelength_index);
    const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
    const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
    const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, thinfilm_slice_index, thinfilm_slice_count);
    std::vector<DielectricDirectionalAlbedoResult> directional(side_entry_count * 2u);
    std::vector<DielectricDirectionalAlbedoResult> total_directional(side_entry_count * 2u);
    scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
      (void)thread_id;
      for (uint32_t index = begin; index < end; ++index) {
        const uint32_t side = index / side_entry_count;
        const uint32_t side_index = index - side * side_entry_count;
        const uint32_t alpha_index = side_index / kEnergyCompensationDielectricLutSize;
        const uint32_t mu_index = side_index - alpha_index * kEnergyCompensationDielectricLutSize;
        const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
        const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
        directional[index] = integrate_dielectric_directional(spect, source_ior, target_ior, thinfilm, side == 0u,
          mu_parameter(mu_index, kEnergyCompensationDielectricLutSize), alpha_parameter(alpha_index, kEnergyCompensationDielectricLutSize));
      }
    });
    scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
      (void)thread_id;
      for (uint32_t index = begin; index < end; ++index) {
        const uint32_t side = index / side_entry_count;
        const uint32_t side_index = index - side * side_entry_count;
        const uint32_t alpha_index = side_index / kEnergyCompensationDielectricLutSize;
        const uint32_t mu_index = side_index - alpha_index * kEnergyCompensationDielectricLutSize;
        const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
        const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
        total_directional[index] = integrate_dielectric_total_directional(spect, source_ior, target_ior, thinfilm, side == 0u,
          mu_parameter(mu_index, kEnergyCompensationDielectricLutSize), alpha_parameter(alpha_index, kEnergyCompensationDielectricLutSize));
      }
    });

    for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationDielectricLutSize; ++alpha_index) {
      float3 single_average[kEnergyCompensationDielectricBranchCount] = {};
      float3 total_average[kEnergyCompensationDielectricBranchCount] = {};
      float3 residual_average[kEnergyCompensationDielectricBranchCount] = {};
      float3 residual_coefficient[kEnergyCompensationDielectricBranchCount] = {};

      for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
        for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
          const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
          single_average[branch] = integrate_dielectric_branch_average(directional, alpha_index, incident_side, outgoing_side);
          total_average[branch] = integrate_dielectric_branch_average(total_directional, alpha_index, incident_side, outgoing_side);
        }
      }

      float3 side_residual[2] = {};
      for (uint32_t side = 0u; side < 2u; ++side) {
        const uint32_t branch_0 = dielectric_branch_index(side, 0u);
        const uint32_t branch_1 = dielectric_branch_index(side, 1u);
        const float3 row_single = saturate(single_average[branch_0] + single_average[branch_1]);
        side_residual[side] = max(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f) - row_single);
      }

      for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
        for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
          const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
          residual_average[branch] = max(float3(0.0f, 0.0f, 0.0f), total_average[branch] - single_average[branch]);
        }
      }

      clamp_dielectric_residual_row(residual_average, side_residual);

      for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
        for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
          const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
          const float3 denominator = max(float3(kEpsilon, kEpsilon, kEpsilon), side_residual[incident_side] * side_residual[outgoing_side]);
          residual_coefficient[branch] = residual_average[branch] / denominator;
        }
      }

      for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
        set_float4_channel(average[alpha_index * kEnergyCompensationDielectricAverageWidth + branch], channel, saturate(single_average[branch].x));
        set_float4_channel(average[alpha_index * kEnergyCompensationDielectricAverageWidth + 4u + branch], channel, max(0.0f, residual_coefficient[branch].x));
      }

      for (uint32_t side = 0u; side < 2u; ++side) {
        const uint32_t side_offset = side * side_entry_count;
        for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationDielectricLutSize; ++mu_index) {
          const uint32_t source_index = side_offset + alpha_index * kEnergyCompensationDielectricLutSize + mu_index;
          const DielectricDirectionalAlbedoResult& value = directional[source_index];
          for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
            const uint32_t branch = dielectric_branch_index(side, outgoing_side);
            const uint32_t x = branch * kEnergyCompensationDielectricLutSize + mu_index;
            const uint32_t image_index = alpha_index * width + x;
            set_float4_channel(image[image_index], channel, saturate(value.branch_albedo[outgoing_side].x));
            set_float4_channel(probability[image_index], channel, saturate(value.branch_visible_probability[outgoing_side]));
          }
        }
      }
    }
  }

  return save_exr_rgba(paths.directional, image, width, kEnergyCompensationDielectricLutSize) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationDielectricAverageWidth, kEnergyCompensationDielectricLutSize) &&
         save_exr_rgba(paths.probability, probability, width, kEnergyCompensationDielectricLutSize);
}

bool generate_conductor_interface_gpu(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material, const GeneratedInterfacePaths& paths,
  const GeneratedInterfacePaths& geometric_paths, uint32_t cache_mode, uint32_t thinfilm_slice_index, uint32_t thinfilm_slice_count, uint32_t wavelength_group_index,
  bool save_spectral_geometric) {
  const uint64_t entry_count = uint64_t(kEnergyCompensationConductorLutSize) * uint64_t(kEnergyCompensationConductorLutSize);
  EnergyCompensationGpuBuffers buffers = {};
  bool result = false;

  do {
    if ((gpu_create_float4_output_buffer(rhi, entry_count, buffers.directional, "ec_conductor_directional") == false) ||
        (gpu_create_float4_output_buffer(rhi, kEnergyCompensationConductorLutSize, buffers.average, "ec_conductor_average") == false) ||
        (gpu_create_float4_output_buffer(rhi, entry_count, buffers.geometric, "ec_conductor_geometric") == false) ||
        (gpu_create_float4_output_buffer(rhi, kEnergyCompensationConductorLutSize, buffers.geometric_average, "ec_conductor_geometric_average") == false) ||
        (gpu_create_float4_output_buffer(rhi, kEnergyCompensationConductorLutSize, buffers.conductor_fms, "ec_conductor_fms") == false)) {
      break;
    }

    EnergyCompensationGpuParams directional_params = make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassConductorDirectional, thinfilm_slice_index,
      thinfilm_slice_count, wavelength_group_index);
    directional_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
    directional_params.output_geometric_index = get_bindless_descriptor_index(buffers.geometric);

    EnergyCompensationGpuParams average_params = make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassConductorAverage, thinfilm_slice_index,
      thinfilm_slice_count, wavelength_group_index);
    average_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
    average_params.output_average_index = get_bindless_descriptor_index(buffers.average);
    average_params.output_geometric_index = get_bindless_descriptor_index(buffers.geometric);
    average_params.output_geometric_average_index = get_bindless_descriptor_index(buffers.geometric_average);
    average_params.output_conductor_fms_index = get_bindless_descriptor_index(buffers.conductor_fms);

    if ((gpu_create_params_buffer(rhi, directional_params, buffers.params_directional, "ec_conductor_directional_params") == false) ||
        (gpu_create_params_buffer(rhi, average_params, buffers.params_average, "ec_conductor_average_params") == false)) {
      break;
    }

    if (gpu_dispatch_energy_compensation(rhi, pipeline, buffers, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, true) == false) {
      break;
    }

    std::vector<float4> directional_pixels;
    std::vector<float4> average_pixels;
    std::vector<float4> conductor_fms_pixels;
    if ((gpu_read_float4_output_buffer(rhi, buffers.directional, entry_count, directional_pixels, "ec_conductor_directional") == false) ||
        (gpu_read_float4_output_buffer(rhi, buffers.average, kEnergyCompensationConductorLutSize, average_pixels, "ec_conductor_average") == false) ||
        (gpu_read_float4_output_buffer(rhi, buffers.conductor_fms, kEnergyCompensationConductorLutSize, conductor_fms_pixels, "ec_conductor_fms") == false)) {
      break;
    }

    if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
      result = save_exr_rgba(paths.directional, directional_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
               save_exr_rgba(paths.average, average_pixels, kEnergyCompensationConductorLutSize, 1u) &&
               save_exr_rgba(paths.conductor_fms, conductor_fms_pixels, kEnergyCompensationConductorLutSize, 1u);
      if (save_spectral_geometric) {
        std::vector<float4> geometric_pixels;
        std::vector<float4> geometric_average_pixels;
        if ((gpu_read_float4_output_buffer(rhi, buffers.geometric, entry_count, geometric_pixels, "ec_conductor_geometric") == false) ||
            (gpu_read_float4_output_buffer(rhi, buffers.geometric_average, kEnergyCompensationConductorLutSize, geometric_average_pixels, "ec_conductor_geometric_average") == false)) {
          break;
        }
        result = (save_exr_rgba(geometric_paths.geometric, geometric_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
                   save_exr_rgba(geometric_paths.geometric_average, geometric_average_pixels, kEnergyCompensationConductorLutSize, 1u)) &&
                 result;
      }
    } else {
      std::vector<float4> geometric_pixels;
      std::vector<float4> geometric_average_pixels;
      if ((gpu_read_float4_output_buffer(rhi, buffers.geometric, entry_count, geometric_pixels, "ec_conductor_geometric") == false) ||
          (gpu_read_float4_output_buffer(rhi, buffers.geometric_average, kEnergyCompensationConductorLutSize, geometric_average_pixels, "ec_conductor_geometric_average") == false)) {
        break;
      }
      result = save_exr_rgba(paths.directional, directional_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
               save_exr_rgba(paths.average, average_pixels, kEnergyCompensationConductorLutSize, 1u) &&
               save_exr_rgba(paths.geometric, geometric_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
               save_exr_rgba(paths.geometric_average, geometric_average_pixels, kEnergyCompensationConductorLutSize, 1u) &&
               save_exr_rgba(paths.conductor_fms, conductor_fms_pixels, kEnergyCompensationConductorLutSize, 1u);
    }
  } while (false);

  gpu_destroy_energy_compensation_buffers(rhi, buffers);
  return result;
}

bool generate_dielectric_interface_gpu(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material, const GeneratedInterfacePaths& paths,
  uint32_t cache_mode, uint32_t thinfilm_slice_index, uint32_t thinfilm_slice_count, uint32_t wavelength_group_index) {
  const uint32_t width = kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize;
  const uint64_t directional_count = uint64_t(width) * uint64_t(kEnergyCompensationDielectricLutSize);
  const uint64_t average_count = uint64_t(kEnergyCompensationDielectricAverageWidth) * uint64_t(kEnergyCompensationDielectricLutSize);
  EnergyCompensationGpuBuffers buffers = {};
  bool result = false;

  do {
    if ((gpu_create_float4_output_buffer(rhi, directional_count, buffers.directional, "ec_dielectric_directional") == false) ||
        (gpu_create_float4_output_buffer(rhi, directional_count, buffers.total, "ec_dielectric_total") == false) ||
        (gpu_create_float4_output_buffer(rhi, directional_count, buffers.probability, "ec_dielectric_probability") == false) ||
        (gpu_create_float4_output_buffer(rhi, average_count, buffers.average, "ec_dielectric_average") == false)) {
      break;
    }

    EnergyCompensationGpuParams directional_params = make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassDielectricDirectional, thinfilm_slice_index,
      thinfilm_slice_count, wavelength_group_index);
    directional_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
    directional_params.output_total_index = get_bindless_descriptor_index(buffers.total);
    directional_params.output_probability_index = get_bindless_descriptor_index(buffers.probability);

    EnergyCompensationGpuParams average_params =
      make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassDielectricAverage, thinfilm_slice_index, thinfilm_slice_count, wavelength_group_index);
    average_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
    average_params.output_total_index = get_bindless_descriptor_index(buffers.total);
    average_params.output_average_index = get_bindless_descriptor_index(buffers.average);

    if ((gpu_create_params_buffer(rhi, directional_params, buffers.params_directional, "ec_dielectric_directional_params") == false) ||
        (gpu_create_params_buffer(rhi, average_params, buffers.params_average, "ec_dielectric_average_params") == false)) {
      break;
    }

    if (gpu_dispatch_energy_compensation(rhi, pipeline, buffers, kEnergyCompensationDielectricLutSize, 2u * kEnergyCompensationDielectricLutSize, true) == false) {
      break;
    }

    std::vector<float4> directional_pixels;
    std::vector<float4> average_pixels;
    std::vector<float4> probability_pixels;
    if ((gpu_read_float4_output_buffer(rhi, buffers.directional, directional_count, directional_pixels, "ec_dielectric_directional") == false) ||
        (gpu_read_float4_output_buffer(rhi, buffers.average, average_count, average_pixels, "ec_dielectric_average") == false)) {
      break;
    }

    result = save_exr_rgba(paths.directional, directional_pixels, width, kEnergyCompensationDielectricLutSize) &&
             save_exr_rgba(paths.average, average_pixels, kEnergyCompensationDielectricAverageWidth, kEnergyCompensationDielectricLutSize);
    if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
      if (gpu_read_float4_output_buffer(rhi, buffers.probability, directional_count, probability_pixels, "ec_dielectric_probability") == false) {
        break;
      }
      result = save_exr_rgba(paths.probability, probability_pixels, width, kEnergyCompensationDielectricLutSize) && result;
    }
  } while (false);

  gpu_destroy_energy_compensation_buffers(rhi, buffers);
  return result;
}

bool ensure_cache_file(const SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode, const GeneratedInterfacePaths& paths,
  TaskScheduler& scheduler) {
  const bool conductor = material_class == MaterialClass::Conductor;
  const SpectralQuery spect = {};
  const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
  const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
  const uint32_t slice_count = thinfilm_lut_slice_count(material.thinfilm);
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    bool spectral_result = true;
    for (uint32_t thinfilm_slice_index = 0u; thinfilm_slice_index < slice_count; ++thinfilm_slice_index) {
      if (conductor) {
        const GeneratedInterfacePaths geometric_paths = interface_slice_paths(paths, thinfilm_slice_index);
        const bool geometric_exists = std::filesystem::exists(geometric_paths.geometric);
        const bool geometric_average_exists = std::filesystem::exists(geometric_paths.geometric_average);
        if ((geometric_exists && geometric_average_exists) == false) {
          const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, thinfilm_slice_index, slice_count);
          const bool generated_geometric = generate_conductor_geometric_interface(geometric_paths, ext_ior, int_ior, thinfilm, scheduler);
          spectral_result = generated_geometric && spectral_result;
        }
      }

      for (uint32_t wavelength_group = 0u; wavelength_group < kEnergyCompensationSpectralWavelengthGroupCount; ++wavelength_group) {
        const uint32_t packed_slice_index = thinfilm_slice_index * kEnergyCompensationSpectralWavelengthGroupCount + wavelength_group;
        const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, packed_slice_index);
        const bool directional_exists = std::filesystem::exists(slice_paths_value.directional);
        const bool average_exists = std::filesystem::exists(slice_paths_value.average);
        const bool conductor_fms_exists = conductor ? std::filesystem::exists(slice_paths_value.conductor_fms) : true;
        const bool probability_exists = conductor ? true : std::filesystem::exists(slice_paths_value.probability);
        if (((directional_exists && average_exists) && conductor_fms_exists) && probability_exists) {
          continue;
        }

        const auto time_begin = std::chrono::steady_clock::now();
        const bool generated = conductor ? generate_conductor_interface_spectral(data, material, slice_paths_value, thinfilm_slice_index, slice_count, wavelength_group, scheduler)
                                         : generate_dielectric_interface_spectral(data, material, slice_paths_value, thinfilm_slice_index, slice_count, wavelength_group, scheduler);
        const auto time_end = std::chrono::steady_clock::now();
        const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
        if (generated) {
          log::info("Generated spectral energy-compensation LUT cache %s in %.3f seconds", slice_paths_value.directional.generic_string().c_str(), elapsed_seconds);
        }
        spectral_result = generated && spectral_result;
      }
    }
    return spectral_result;
  }

  bool result = true;
  for (uint32_t slice_index = 0u; slice_index < slice_count; ++slice_index) {
    const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, slice_index);
    const bool directional_exists = std::filesystem::exists(slice_paths_value.directional);
    const bool average_exists = std::filesystem::exists(slice_paths_value.average);
    const bool geometric_exists = conductor ? std::filesystem::exists(slice_paths_value.geometric) : true;
    const bool geometric_average_exists = conductor ? std::filesystem::exists(slice_paths_value.geometric_average) : true;
    const bool conductor_fms_exists = conductor ? std::filesystem::exists(slice_paths_value.conductor_fms) : true;
    if ((((directional_exists && average_exists) && geometric_exists) && geometric_average_exists) && conductor_fms_exists) {
      continue;
    }

    const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, slice_index, slice_count);
    const auto time_begin = std::chrono::steady_clock::now();
    const bool generated =
      conductor ? generate_conductor_interface(slice_paths_value, ext_ior, int_ior, thinfilm, scheduler) : generate_dielectric_interface(slice_paths_value, ext_ior, int_ior, thinfilm, scheduler);
    const auto time_end = std::chrono::steady_clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
    if (generated) {
      log::info("Generated energy-compensation LUT cache %s in %.3f seconds", slice_paths_value.directional.generic_string().c_str(), elapsed_seconds);
    }
    result = generated && result;
  }
  return result;
}

bool ensure_cache_file_gpu(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode,
  const GeneratedInterfacePaths& paths) {
  const bool conductor = material_class == MaterialClass::Conductor;
  const uint32_t slice_count = thinfilm_lut_slice_count(material.thinfilm);
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    bool spectral_result = true;
    for (uint32_t thinfilm_slice_index = 0u; thinfilm_slice_index < slice_count; ++thinfilm_slice_index) {
      bool conductor_geometric_generated = false;
      for (uint32_t wavelength_group = 0u; wavelength_group < kEnergyCompensationSpectralWavelengthGroupCount; ++wavelength_group) {
        const uint32_t packed_slice_index = thinfilm_slice_index * kEnergyCompensationSpectralWavelengthGroupCount + wavelength_group;
        const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, packed_slice_index);
        const bool directional_exists = std::filesystem::exists(slice_paths_value.directional);
        const bool average_exists = std::filesystem::exists(slice_paths_value.average);
        const bool conductor_fms_exists = conductor ? std::filesystem::exists(slice_paths_value.conductor_fms) : true;
        const bool probability_exists = conductor ? true : std::filesystem::exists(slice_paths_value.probability);
        const GeneratedInterfacePaths geometric_paths = interface_slice_paths(paths, thinfilm_slice_index);
        const bool geometric_exists = conductor ? std::filesystem::exists(geometric_paths.geometric) : true;
        const bool geometric_average_exists = conductor ? std::filesystem::exists(geometric_paths.geometric_average) : true;
        if (((((directional_exists && average_exists) && conductor_fms_exists) && probability_exists) && geometric_exists) && geometric_average_exists) {
          continue;
        }

        const auto time_begin = std::chrono::steady_clock::now();
        const bool save_spectral_geometric = conductor && (conductor_geometric_generated == false) &&
                                             ((std::filesystem::exists(geometric_paths.geometric) && std::filesystem::exists(geometric_paths.geometric_average)) == false);
        const bool generated = conductor ? generate_conductor_interface_gpu(rhi, pipeline, data, material, slice_paths_value, geometric_paths, cache_mode, thinfilm_slice_index,
                                            slice_count, wavelength_group, save_spectral_geometric)
                                         : generate_dielectric_interface_gpu(rhi, pipeline, data, material, slice_paths_value, cache_mode, thinfilm_slice_index, slice_count,
                                             wavelength_group);
        if (save_spectral_geometric) {
          conductor_geometric_generated = generated || conductor_geometric_generated;
        }
        const auto time_end = std::chrono::steady_clock::now();
        const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
        if (generated) {
          log::info("Generated GPU spectral energy-compensation LUT cache %s in %.3f seconds", slice_paths_value.directional.generic_string().c_str(), elapsed_seconds);
        }
        spectral_result = generated && spectral_result;
      }
    }
    return spectral_result;
  }

  bool result = true;
  for (uint32_t slice_index = 0u; slice_index < slice_count; ++slice_index) {
    const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, slice_index);
    const bool directional_exists = std::filesystem::exists(slice_paths_value.directional);
    const bool average_exists = std::filesystem::exists(slice_paths_value.average);
    const bool geometric_exists = conductor ? std::filesystem::exists(slice_paths_value.geometric) : true;
    const bool geometric_average_exists = conductor ? std::filesystem::exists(slice_paths_value.geometric_average) : true;
    const bool conductor_fms_exists = conductor ? std::filesystem::exists(slice_paths_value.conductor_fms) : true;
    if ((((directional_exists && average_exists) && geometric_exists) && geometric_average_exists) && conductor_fms_exists) {
      continue;
    }

    const auto time_begin = std::chrono::steady_clock::now();
    const bool generated = conductor ? generate_conductor_interface_gpu(rhi, pipeline, data, material, slice_paths_value, slice_paths_value, cache_mode, slice_index, slice_count, 0u, false)
                                     : generate_dielectric_interface_gpu(rhi, pipeline, data, material, slice_paths_value, cache_mode, slice_index, slice_count, 0u);
    const auto time_end = std::chrono::steady_clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
    if (generated) {
      log::info("Generated GPU energy-compensation LUT cache %s in %.3f seconds", slice_paths_value.directional.generic_string().c_str(), elapsed_seconds);
    }
    result = generated && result;
  }
  return result;
}

bool bind_energy_compensation_interface(SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode,
  std::unordered_map<uint64_t, uint32_t>& interface_cache, TaskScheduler& scheduler, RHIContext* rhi, EnergyCompensationGpuPipeline* gpu_pipeline, uint32_t& out_interface_index) {
  const bool conductor = material_class == MaterialClass::Conductor;
  const uint64_t hash = hash_material_interface(data, material, material_class, cache_mode);
  const auto found = interface_cache.find(hash);
  if (found != interface_cache.end()) {
    out_interface_index = found->second;
    return true;
  }

  const GeneratedInterfacePaths paths = interface_paths(material_class, hash);
  bool cache_ready = false;
  if ((rhi != nullptr) && (gpu_pipeline != nullptr)) {
    cache_ready = ensure_cache_file_gpu(*rhi, *gpu_pipeline, data, material, material_class, cache_mode, paths);
  } else {
    cache_ready = ensure_cache_file(data, material, material_class, cache_mode, paths, scheduler);
  }
  if (cache_ready == false) {
    return false;
  }

  Scene::EnergyCompensationInterface interface_data = {};
  interface_data.cls = material_class;
  interface_data.cache_mode = cache_mode;
  const uint32_t slice_count = thinfilm_lut_slice_count(material.thinfilm);
  interface_data.thinfilm_slice_count = slice_count;
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    interface_data.spectral_wavelength_count = kEnergyCompensationSpectralWavelengthCount;
    interface_data.spectral_shortest_wavelength = kShortestWavelength;
    interface_data.spectral_longest_wavelength = kLongestWavelength;
  }
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    std::vector<float4> directional_pixels;
    std::vector<float4> average_pixels;
    std::vector<float4> geometric_pixels;
    std::vector<float4> geometric_average_pixels;
    std::vector<float4> conductor_fms_pixels;
    std::vector<float4> probability_pixels;
    const uint32_t directional_width = conductor ? kEnergyCompensationConductorLutSize : (kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize);
    const uint32_t directional_height = conductor ? kEnergyCompensationConductorLutSize : kEnergyCompensationDielectricLutSize;
    const uint32_t average_width = conductor ? kEnergyCompensationConductorLutSize : kEnergyCompensationDielectricAverageWidth;
    const uint32_t average_height = conductor ? 1u : kEnergyCompensationDielectricLutSize;
    const uint32_t packed_slice_count = slice_count * kEnergyCompensationSpectralWavelengthGroupCount;
    if ((load_rgba32f_lut_slices(paths, directional_width, directional_height, packed_slice_count, directional_pixels, &GeneratedInterfacePaths::directional) == false) ||
        (load_rgba32f_lut_slices(paths, average_width, average_height, packed_slice_count, average_pixels, &GeneratedInterfacePaths::average) == false)) {
      return false;
    }

    interface_data.directional_lut = data.images.add_from_data_3d(directional_pixels.data(), uint3{directional_width, directional_height, packed_slice_count},
      Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    interface_data.average_lut =
      data.images.add_from_data_3d(average_pixels.data(), uint3{average_width, average_height, packed_slice_count}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});

    if (conductor) {
      if ((load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, slice_count, geometric_pixels,
             &GeneratedInterfacePaths::geometric) == false) ||
          (load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, 1u, slice_count, geometric_average_pixels, &GeneratedInterfacePaths::geometric_average) == false) ||
          (load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, 1u, packed_slice_count, conductor_fms_pixels, &GeneratedInterfacePaths::conductor_fms) == false)) {
        return false;
      }
      interface_data.geometric_lut = data.images.add_from_data_3d(geometric_pixels.data(), uint3{kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      interface_data.geometric_average_lut = data.images.add_from_data_3d(geometric_average_pixels.data(), uint3{kEnergyCompensationConductorLutSize, 1u, slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      interface_data.conductor_fms_lut = data.images.add_from_data_3d(conductor_fms_pixels.data(), uint3{kEnergyCompensationConductorLutSize, 1u, packed_slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    } else {
      if (load_rgba32f_lut_slices(paths, directional_width, directional_height, packed_slice_count, probability_pixels, &GeneratedInterfacePaths::probability) == false) {
        return false;
      }
      interface_data.probability_lut = data.images.add_from_data_3d(probability_pixels.data(), uint3{directional_width, directional_height, packed_slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    }
  } else if (slice_count == 1u) {
    interface_data.directional_lut = data.add_image(paths.directional.generic_string().c_str(), Image::SkipSRGBConversion);
    interface_data.average_lut = data.add_image(paths.average.generic_string().c_str(), Image::SkipSRGBConversion);
    interface_data.geometric_lut = conductor ? data.add_image(paths.geometric.generic_string().c_str(), Image::SkipSRGBConversion) : kInvalidIndex;
    interface_data.geometric_average_lut = conductor ? data.add_image(paths.geometric_average.generic_string().c_str(), Image::SkipSRGBConversion) : kInvalidIndex;
    interface_data.conductor_fms_lut = conductor ? data.add_image(paths.conductor_fms.generic_string().c_str(), Image::SkipSRGBConversion) : kInvalidIndex;
  } else {
    std::vector<float4> directional_pixels;
    std::vector<float4> average_pixels;
    std::vector<float4> geometric_pixels;
    std::vector<float4> geometric_average_pixels;
    std::vector<float4> conductor_fms_pixels;
    const uint32_t directional_width = conductor ? kEnergyCompensationConductorLutSize : (kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize);
    const uint32_t directional_height = conductor ? kEnergyCompensationConductorLutSize : kEnergyCompensationDielectricLutSize;
    const uint32_t average_width = conductor ? kEnergyCompensationConductorLutSize : kEnergyCompensationDielectricAverageWidth;
    const uint32_t average_height = conductor ? 1u : kEnergyCompensationDielectricLutSize;
    if ((load_rgba32f_lut_slices(paths, directional_width, directional_height, slice_count, directional_pixels, &GeneratedInterfacePaths::directional) == false) ||
        (load_rgba32f_lut_slices(paths, average_width, average_height, slice_count, average_pixels, &GeneratedInterfacePaths::average) == false)) {
      return false;
    }

    interface_data.directional_lut =
      data.images.add_from_data_3d(directional_pixels.data(), uint3{directional_width, directional_height, slice_count}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    interface_data.average_lut =
      data.images.add_from_data_3d(average_pixels.data(), uint3{average_width, average_height, slice_count}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});

    if (conductor) {
      if ((load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, slice_count, geometric_pixels,
             &GeneratedInterfacePaths::geometric) == false) ||
          (load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, 1u, slice_count, geometric_average_pixels, &GeneratedInterfacePaths::geometric_average) == false) ||
          (load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, 1u, slice_count, conductor_fms_pixels, &GeneratedInterfacePaths::conductor_fms) == false)) {
        return false;
      }
      interface_data.geometric_lut = data.images.add_from_data_3d(geometric_pixels.data(), uint3{kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      interface_data.geometric_average_lut = data.images.add_from_data_3d(geometric_average_pixels.data(), uint3{kEnergyCompensationConductorLutSize, 1u, slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      interface_data.conductor_fms_lut = data.images.add_from_data_3d(conductor_fms_pixels.data(), uint3{kEnergyCompensationConductorLutSize, 1u, slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    } else {
      interface_data.geometric_lut = kInvalidIndex;
      interface_data.geometric_average_lut = kInvalidIndex;
      interface_data.conductor_fms_lut = kInvalidIndex;
    }
  }
  const uint32_t interface_index = data.add_energy_compensation_interface(interface_data);
  interface_cache[hash] = interface_index;
  out_interface_index = interface_index;
  log::info("Bound material energy-compensation interface %u to %s", interface_index, paths.directional.generic_string().c_str());
  return true;
}

}  // namespace

bool ensure_energy_compensation_interfaces_impl(SceneData& data, TaskScheduler& scheduler, RHIContext* rhi) {
  std::unordered_map<uint64_t, uint32_t> interface_cache;
  EnergyCompensationGpuPipeline gpu_pipeline = {};
  bool result = true;
  const uint32_t cache_mode = data.options.properties[Scene::Properties::Spectral] ? kBSDFEnergyCompensationCacheModeSpectralScalar : kBSDFEnergyCompensationCacheModeIntegratedRGB;

  data.energy_compensation_interfaces.clear();
  for (Material& material : data.materials) {
    material.energy_compensation_interface_index = kInvalidIndex;
    material.conductor_energy_compensation_interface_index = kInvalidIndex;
  }

  for (Material& material : data.materials) {
    if (material.cls == MaterialClass::OpenPBR) {
      Material dielectric_material = material;
      dielectric_material.cls = MaterialClass::Dielectric;
      const bool dielectric_bound = bind_energy_compensation_interface(data, dielectric_material, MaterialClass::Dielectric, cache_mode, interface_cache, scheduler, rhi, &gpu_pipeline,
        material.energy_compensation_interface_index);

      Material conductor_material = material;
      conductor_material.cls = MaterialClass::Conductor;
      conductor_material.int_ior.cls = SpectralDistribution::Conductor;
      if (data.defaults.conductor_eta == kInvalidIndex) {
        data.defaults.conductor_eta = data.add_spectrum(SpectralDistribution::constant(0.0f));
      }
      if (data.defaults.conductor_k == kInvalidIndex) {
        data.defaults.conductor_k = data.add_spectrum(SpectralDistribution::constant(1000000.0f));
      }
      conductor_material.int_ior.eta_index = data.defaults.conductor_eta;
      conductor_material.int_ior.k_index = data.defaults.conductor_k;
      const bool conductor_bound = bind_energy_compensation_interface(data, conductor_material, MaterialClass::Conductor, cache_mode, interface_cache, scheduler, rhi, &gpu_pipeline,
        material.conductor_energy_compensation_interface_index);
      result = (dielectric_bound && conductor_bound) && result;
      continue;
    }

    if (((material.cls != MaterialClass::Conductor) && (material.cls != MaterialClass::Dielectric)) && (material.cls != MaterialClass::Plastic)) {
      continue;
    }

    const uint32_t material_class = (material.cls == MaterialClass::Plastic) ? MaterialClass::Dielectric : material.cls;
    result = bind_energy_compensation_interface(data, material, material_class, cache_mode, interface_cache, scheduler, rhi, &gpu_pipeline, material.energy_compensation_interface_index) && result;
  }

  if ((rhi != nullptr) && gpu_pipeline.pipeline.valid()) {
    rhi->device().destroy_pipeline(gpu_pipeline.pipeline);
  }

  return result;
}

namespace {

float parity_channel(const float4& value, uint32_t channel) {
  if (channel == 0u) {
    return value.x;
  }
  if (channel == 1u) {
    return value.y;
  }
  if (channel == 2u) {
    return value.z;
  }
  return value.w;
}

template <typename Function>
bool measure_energy_compensation_parity_generation(const char* label, const char* backend, Function function) {
  const auto start = std::chrono::steady_clock::now();
  const bool result = function();
  const auto finish = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(finish - start).count();
  log::info("Generated %s energy-compensation parity LUTs on %s in %.3f seconds", label, backend, elapsed);
  return result;
}

bool compare_energy_compensation_lut(const char* label, const std::filesystem::path& cpu_path, const std::filesystem::path& gpu_path, uint32_t width, uint32_t height,
  float base_tolerance, float relative_tolerance) {
  std::vector<float4> cpu_pixels;
  std::vector<float4> gpu_pixels;
  if ((load_rgba32f_lut(cpu_path, width, height, cpu_pixels) == false) || (load_rgba32f_lut(gpu_path, width, height, gpu_pixels) == false)) {
    log::error("Failed to load energy-compensation parity LUT %s", label);
    return false;
  }

  float max_error = 0.0f;
  uint32_t max_pixel = 0u;
  uint32_t max_channel = 0u;
  for (uint32_t pixel = 0u; pixel < cpu_pixels.size(); ++pixel) {
    for (uint32_t channel = 0u; channel < 4u; ++channel) {
      const float cpu_value = parity_channel(cpu_pixels[pixel], channel);
      const float gpu_value = parity_channel(gpu_pixels[pixel], channel);
      if ((std::isfinite(cpu_value) == false) || (std::isfinite(gpu_value) == false)) {
        log::error("Non-finite energy-compensation parity value %s pixel %u channel %u cpu %.9f gpu %.9f", label, pixel, channel, cpu_value, gpu_value);
        return false;
      }

      const float error = fabsf(cpu_value - gpu_value);
      const float cpu_abs = fabsf(cpu_value);
      const float gpu_abs = fabsf(gpu_value);
      const float scale = (cpu_abs > gpu_abs) ? cpu_abs : gpu_abs;
      const float tolerance = base_tolerance + relative_tolerance * scale;
      if (error > tolerance) {
        log::error("Energy-compensation GPU parity failed %s pixel %u channel %u cpu %.9f gpu %.9f error %.9f tolerance %.9f", label, pixel, channel,
          cpu_value, gpu_value, error, tolerance);
        return false;
      }

      if (error > max_error) {
        max_error = error;
        max_pixel = pixel;
        max_channel = channel;
      }
    }
  }

  log::info("Energy-compensation GPU parity valid %s max error %.9f at pixel %u channel %u", label, max_error, max_pixel, max_channel);
  return true;
}

bool compare_parity_values(const char* label, uint32_t pixel, uint32_t channel, float cpu_value, float gpu_value, float base_tolerance, float relative_tolerance,
  float& max_error, uint32_t& max_pixel, uint32_t& max_channel) {
  if ((std::isfinite(cpu_value) == false) || (std::isfinite(gpu_value) == false)) {
    log::error("Non-finite energy-compensation parity value %s pixel %u channel %u cpu %.9f gpu %.9f", label, pixel, channel, cpu_value, gpu_value);
    return false;
  }

  const float error = fabsf(cpu_value - gpu_value);
  const float cpu_abs = fabsf(cpu_value);
  const float gpu_abs = fabsf(gpu_value);
  const float scale = (cpu_abs > gpu_abs) ? cpu_abs : gpu_abs;
  const float tolerance = base_tolerance + relative_tolerance * scale;
  if (error > tolerance) {
    log::error("Energy-compensation GPU parity failed %s pixel %u channel %u cpu %.9f gpu %.9f error %.9f tolerance %.9f", label, pixel, channel, cpu_value, gpu_value,
      error, tolerance);
    return false;
  }

  if (error > max_error) {
    max_error = error;
    max_pixel = pixel;
    max_channel = channel;
  }
  return true;
}

bool compare_dielectric_average_lut(const char* label, const std::filesystem::path& cpu_path, const std::filesystem::path& gpu_path, float base_tolerance,
  float relative_tolerance) {
  constexpr uint32_t width = kEnergyCompensationDielectricAverageWidth;
  constexpr uint32_t height = kEnergyCompensationDielectricLutSize;
  std::vector<float4> cpu_pixels;
  std::vector<float4> gpu_pixels;
  if ((load_rgba32f_lut(cpu_path, width, height, cpu_pixels) == false) || (load_rgba32f_lut(gpu_path, width, height, gpu_pixels) == false)) {
    log::error("Failed to load energy-compensation parity LUT %s", label);
    return false;
  }

  float max_error = 0.0f;
  uint32_t max_pixel = 0u;
  uint32_t max_channel = 0u;
  for (uint32_t alpha_index = 0u; alpha_index < height; ++alpha_index) {
    const uint32_t row_offset = alpha_index * width;
    for (uint32_t channel = 0u; channel < 4u; ++channel) {
      float cpu_single[kEnergyCompensationDielectricBranchCount] = {};
      float gpu_single[kEnergyCompensationDielectricBranchCount] = {};
      for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
        const uint32_t pixel = row_offset + branch;
        cpu_single[branch] = parity_channel(cpu_pixels[pixel], channel);
        gpu_single[branch] = parity_channel(gpu_pixels[pixel], channel);
        if (compare_parity_values(label, pixel, channel, cpu_single[branch], gpu_single[branch], base_tolerance, relative_tolerance, max_error, max_pixel, max_channel) ==
            false) {
          return false;
        }
      }

      float cpu_side_residual[2] = {};
      float gpu_side_residual[2] = {};
      for (uint32_t side = 0u; side < 2u; ++side) {
        const uint32_t branch_0 = dielectric_branch_index(side, 0u);
        const uint32_t branch_1 = dielectric_branch_index(side, 1u);
        cpu_side_residual[side] = max(0.0f, 1.0f - saturate(cpu_single[branch_0] + cpu_single[branch_1]));
        gpu_side_residual[side] = max(0.0f, 1.0f - saturate(gpu_single[branch_0] + gpu_single[branch_1]));
      }

      for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
        const uint32_t pixel = row_offset + kEnergyCompensationDielectricBranchCount + branch;
        const uint32_t incident_side = branch / 2u;
        const uint32_t outgoing_side = branch - incident_side * 2u;
        const float cpu_coefficient = parity_channel(cpu_pixels[pixel], channel);
        const float gpu_coefficient = parity_channel(gpu_pixels[pixel], channel);
        const float cpu_value = cpu_coefficient * cpu_side_residual[incident_side] * cpu_side_residual[outgoing_side];
        const float gpu_value = gpu_coefficient * gpu_side_residual[incident_side] * gpu_side_residual[outgoing_side];
        if (compare_parity_values(label, pixel, channel, cpu_value, gpu_value, base_tolerance, relative_tolerance, max_error, max_pixel, max_channel) == false) {
          return false;
        }
      }
    }
  }

  log::info("Energy-compensation GPU parity valid %s max error %.9f at pixel %u channel %u", label, max_error, max_pixel, max_channel);
  return true;
}

Material make_parity_conductor(uint32_t white, uint32_t air_eta, uint32_t zero, uint32_t conductor_eta, uint32_t conductor_k) {
  Material result = {};
  result.cls = MaterialClass::Conductor;
  result.reflectance.spectrum_index = white;
  result.roughness.value = float4{0.5f, 0.5f, 0.0f, 0.0f};
  result.ext_ior.cls = SpectralDistribution::Dielectric;
  result.ext_ior.eta_index = air_eta;
  result.ext_ior.k_index = zero;
  result.int_ior.cls = SpectralDistribution::Conductor;
  result.int_ior.eta_index = conductor_eta;
  result.int_ior.k_index = conductor_k;
  return result;
}

Material make_parity_dielectric(uint32_t white, uint32_t air_eta, uint32_t zero, uint32_t dielectric_eta, uint32_t dielectric_k) {
  Material result = {};
  result.cls = MaterialClass::Dielectric;
  result.reflectance.spectrum_index = white;
  result.scattering.spectrum_index = white;
  result.roughness.value = float4{0.5f, 0.5f, 0.0f, 0.0f};
  result.ext_ior.cls = SpectralDistribution::Dielectric;
  result.ext_ior.eta_index = air_eta;
  result.ext_ior.k_index = zero;
  result.int_ior.cls = SpectralDistribution::Dielectric;
  result.int_ior.eta_index = dielectric_eta;
  result.int_ior.k_index = dielectric_k;
  return result;
}

bool validate_energy_compensation_integrated_conductor_parity(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material,
  TaskScheduler& scheduler) {
  const GeneratedInterfacePaths cpu_paths = parity_interface_paths("cpu_conductor_rgb");
  const GeneratedInterfacePaths gpu_paths = parity_interface_paths("gpu_conductor_rgb");
  const SpectralQuery spect = {};
  const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
  const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
  const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, 0u, 1u);

  const bool cpu_generated = measure_energy_compensation_parity_generation("conductor rgb", "CPU", [&]() {
    return generate_conductor_interface(cpu_paths, ext_ior, int_ior, thinfilm, scheduler);
  });
  const bool gpu_generated = measure_energy_compensation_parity_generation("conductor rgb", "GPU", [&]() {
    return generate_conductor_interface_gpu(rhi, pipeline, data, material, gpu_paths, gpu_paths, kBSDFEnergyCompensationCacheModeIntegratedRGB, 0u, 1u, 0u, false);
  });

  if ((cpu_generated == false) || (gpu_generated == false)) {
    log::error("Failed to generate integrated conductor energy-compensation parity LUTs");
    return false;
  }

  bool valid = true;
  valid = compare_energy_compensation_lut("conductor rgb directional", cpu_paths.directional, gpu_paths.directional, kEnergyCompensationConductorLutSize,
            kEnergyCompensationConductorLutSize, 4.0e-4f, 3.0e-3f) &&
          valid;
  valid = compare_energy_compensation_lut("conductor rgb average", cpu_paths.average, gpu_paths.average, kEnergyCompensationConductorLutSize, 1u, 4.0e-4f, 3.0e-3f) && valid;
  valid = compare_energy_compensation_lut("conductor rgb geometric", cpu_paths.geometric, gpu_paths.geometric, kEnergyCompensationConductorLutSize,
            kEnergyCompensationConductorLutSize, 4.0e-4f, 3.0e-3f) &&
          valid;
  valid = compare_energy_compensation_lut("conductor rgb geometric average", cpu_paths.geometric_average, gpu_paths.geometric_average, kEnergyCompensationConductorLutSize, 1u,
            4.0e-4f, 3.0e-3f) &&
          valid;
  valid = compare_energy_compensation_lut("conductor rgb fms", cpu_paths.conductor_fms, gpu_paths.conductor_fms, kEnergyCompensationConductorLutSize, 1u, 4.0e-4f, 3.0e-3f) &&
          valid;
  return valid;
}

bool validate_energy_compensation_integrated_dielectric_parity(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material,
  const char* key, const char* label, TaskScheduler& scheduler) {
  const std::string cpu_key = std::string("cpu_") + key;
  const std::string gpu_key = std::string("gpu_") + key;
  const GeneratedInterfacePaths cpu_paths = parity_interface_paths(cpu_key.c_str());
  const GeneratedInterfacePaths gpu_paths = parity_interface_paths(gpu_key.c_str());
  const SpectralQuery spect = {};
  const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
  const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
  const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, 0u, 1u);

  const bool cpu_generated = measure_energy_compensation_parity_generation(label, "CPU", [&]() {
    return generate_dielectric_interface(cpu_paths, ext_ior, int_ior, thinfilm, scheduler);
  });
  const bool gpu_generated = measure_energy_compensation_parity_generation(label, "GPU", [&]() {
    return generate_dielectric_interface_gpu(rhi, pipeline, data, material, gpu_paths, kBSDFEnergyCompensationCacheModeIntegratedRGB, 0u, 1u, 0u);
  });

  if ((cpu_generated == false) || (gpu_generated == false)) {
    log::error("Failed to generate integrated dielectric energy-compensation parity LUTs");
    return false;
  }

  const std::string directional_label = std::string(label) + " directional";
  const std::string average_label = std::string(label) + " average";
  bool valid = true;
  valid = compare_energy_compensation_lut(directional_label.c_str(), cpu_paths.directional, gpu_paths.directional,
            kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize, kEnergyCompensationDielectricLutSize, 8.0e-4f, 6.0e-3f) &&
          valid;
  valid = compare_dielectric_average_lut(average_label.c_str(), cpu_paths.average, gpu_paths.average, 8.0e-4f, 6.0e-3f) && valid;
  return valid;
}

bool validate_energy_compensation_spectral_conductor_parity(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material,
  TaskScheduler& scheduler) {
  const GeneratedInterfacePaths cpu_paths = parity_interface_paths("cpu_conductor_spectral");
  const GeneratedInterfacePaths gpu_paths = parity_interface_paths("gpu_conductor_spectral");
  const SpectralQuery spect = {};
  const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
  const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
  const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, 0u, 1u);

  const bool cpu_generated = measure_energy_compensation_parity_generation("conductor spectral", "CPU", [&]() {
    const bool geometric_generated = generate_conductor_geometric_interface(cpu_paths, ext_ior, int_ior, thinfilm, scheduler);
    const bool spectral_generated = generate_conductor_interface_spectral(data, material, cpu_paths, 0u, 1u, 0u, scheduler);
    return (geometric_generated && spectral_generated);
  });
  const bool gpu_generated = measure_energy_compensation_parity_generation("conductor spectral", "GPU", [&]() {
    return generate_conductor_interface_gpu(rhi, pipeline, data, material, gpu_paths, gpu_paths, kBSDFEnergyCompensationCacheModeSpectralScalar, 0u, 1u, 0u, true);
  });

  if ((cpu_generated == false) || (gpu_generated == false)) {
    log::error("Failed to generate spectral conductor energy-compensation parity LUTs");
    return false;
  }

  bool valid = true;
  valid = compare_energy_compensation_lut("conductor spectral directional", cpu_paths.directional, gpu_paths.directional, kEnergyCompensationConductorLutSize,
            kEnergyCompensationConductorLutSize, 4.0e-4f, 3.0e-3f) &&
          valid;
  valid = compare_energy_compensation_lut("conductor spectral average", cpu_paths.average, gpu_paths.average, kEnergyCompensationConductorLutSize, 1u, 4.0e-4f, 3.0e-3f) &&
          valid;
  valid = compare_energy_compensation_lut("conductor spectral fms", cpu_paths.conductor_fms, gpu_paths.conductor_fms, kEnergyCompensationConductorLutSize, 1u, 4.0e-4f,
            3.0e-3f) &&
          valid;
  valid = compare_energy_compensation_lut("conductor spectral geometric", cpu_paths.geometric, gpu_paths.geometric, kEnergyCompensationConductorLutSize,
            kEnergyCompensationConductorLutSize, 4.0e-4f, 3.0e-3f) &&
          valid;
  valid = compare_energy_compensation_lut("conductor spectral geometric average", cpu_paths.geometric_average, gpu_paths.geometric_average,
            kEnergyCompensationConductorLutSize, 1u, 4.0e-4f, 3.0e-3f) &&
          valid;
  return valid;
}

bool validate_energy_compensation_spectral_dielectric_parity(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material,
  const char* key, const char* label, TaskScheduler& scheduler) {
  const std::string cpu_key = std::string("cpu_") + key;
  const std::string gpu_key = std::string("gpu_") + key;
  const GeneratedInterfacePaths cpu_paths = parity_interface_paths(cpu_key.c_str());
  const GeneratedInterfacePaths gpu_paths = parity_interface_paths(gpu_key.c_str());
  const bool cpu_generated = measure_energy_compensation_parity_generation(label, "CPU", [&]() {
    return generate_dielectric_interface_spectral(data, material, cpu_paths, 0u, 1u, 0u, scheduler);
  });
  const bool gpu_generated = measure_energy_compensation_parity_generation(label, "GPU", [&]() {
    return generate_dielectric_interface_gpu(rhi, pipeline, data, material, gpu_paths, kBSDFEnergyCompensationCacheModeSpectralScalar, 0u, 1u, 0u);
  });

  if ((cpu_generated == false) || (gpu_generated == false)) {
    log::error("Failed to generate spectral dielectric energy-compensation parity LUTs");
    return false;
  }

  const std::string directional_label = std::string(label) + " directional";
  const std::string average_label = std::string(label) + " average";
  const std::string probability_label = std::string(label) + " probability";
  bool valid = true;
  valid = compare_energy_compensation_lut(directional_label.c_str(), cpu_paths.directional, gpu_paths.directional,
            kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize, kEnergyCompensationDielectricLutSize, 8.0e-4f, 6.0e-3f) &&
          valid;
  valid = compare_dielectric_average_lut(average_label.c_str(), cpu_paths.average, gpu_paths.average, 8.0e-4f, 6.0e-3f) && valid;
  valid = compare_energy_compensation_lut(probability_label.c_str(), cpu_paths.probability, gpu_paths.probability,
            kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize, kEnergyCompensationDielectricLutSize, 8.0e-4f, 6.0e-3f) &&
          valid;
  return valid;
}

bool validate_energy_compensation_gpu_lut_parity_impl(RHIContext& rhi, TaskScheduler& scheduler) {
  std::error_code filesystem_error = {};
  std::filesystem::remove_all(parity_cache_directory(), filesystem_error);
  if (filesystem_error.value() != 0) {
    log::error("Failed to clear energy-compensation parity cache %s", parity_cache_directory().generic_string().c_str());
    return false;
  }

  SceneData data(scheduler);
  data.images.init(16u);
  const uint32_t white = data.add_spectrum(SpectralDistribution::constant(1.0f));
  const uint32_t zero = data.add_spectrum(SpectralDistribution::constant(0.0f));
  const uint32_t air_eta = data.add_spectrum(SpectralDistribution::constant(1.0f));
  SpectralDistribution dielectric_eta_spd = {};
  SpectralDistribution dielectric_k_spd = {};
  SpectralDistribution conductor_eta_spd = {};
  SpectralDistribution conductor_k_spd = {};
  std::string spectrum_title = {};
  SpectralDistribution::load_refractive_index(env().file_in_data("spectrum/dielectric/sapphire.spd"), dielectric_eta_spd, dielectric_k_spd, spectrum_title);
  SpectralDistribution::load_refractive_index(env().file_in_data("spectrum/conductor/copper.spd"), conductor_eta_spd, conductor_k_spd, spectrum_title);
  const uint32_t dielectric_eta = data.add_spectrum(dielectric_eta_spd);
  const uint32_t dielectric_k = data.add_spectrum(dielectric_k_spd);
  const uint32_t conductor_eta = data.add_spectrum(conductor_eta_spd);
  const uint32_t conductor_k = data.add_spectrum(conductor_k_spd);
  const Material conductor = make_parity_conductor(white, air_eta, zero, conductor_eta, conductor_k);
  const Material dielectric = make_parity_dielectric(white, air_eta, zero, dielectric_eta, dielectric_k);
  EnergyCompensationGpuPipeline pipeline = {};

  // Keep strict element-wise parity focused on backend-stable, non-coherent
  // interfaces. Coherent-film phase uses backend-native transcendental
  // functions, so its correctness is covered by optical invariants and CPU/GPU
  // furnace tests rather than an artificial bit-level equality requirement.
  bool valid = true;
  valid = validate_energy_compensation_integrated_conductor_parity(rhi, pipeline, data, conductor, scheduler) && valid;
  valid = validate_energy_compensation_integrated_dielectric_parity(rhi, pipeline, data, dielectric, "dielectric_rgb", "dielectric rgb", scheduler) && valid;
  valid = validate_energy_compensation_spectral_conductor_parity(rhi, pipeline, data, conductor, scheduler) && valid;
  valid = validate_energy_compensation_spectral_dielectric_parity(rhi, pipeline, data, dielectric, "dielectric_spectral", "dielectric spectral", scheduler) && valid;

  if (pipeline.pipeline.valid()) {
    rhi.device().destroy_pipeline(pipeline.pipeline);
  }
  data.images.cleanup();
  return valid;
}

}  // namespace

bool ensure_energy_compensation_interfaces(SceneData& data, TaskScheduler& scheduler) {
  return ensure_energy_compensation_interfaces_impl(data, scheduler, nullptr);
}

bool ensure_energy_compensation_interfaces(SceneData& data, TaskScheduler& scheduler, RHIContext& rhi) {
  return ensure_energy_compensation_interfaces_impl(data, scheduler, &rhi);
}

bool validate_energy_compensation_gpu_lut_parity(RHIContext& rhi, TaskScheduler& scheduler) {
  return validate_energy_compensation_gpu_lut_parity_impl(rhi, scheduler);
}

}  // namespace etx
