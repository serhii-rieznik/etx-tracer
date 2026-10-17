#include <etx/render/host/bsdf_energy_compensation_lut.hxx>

#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/exr.hxx>
#include <etx/render/host/image_loaders.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/interop/bsdf_energy_compensated_shared.hxx>
#include <etx/render/interop/bsdf_external_shared.hxx>
#include <etx/render/interop/bsdf_velvet_constants_shared.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include <cmath>
#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace etx {

namespace {

constexpr uint32_t kEnergyCompensationGeneratorVersion = 42u;
constexpr uint32_t kEnergyCompensationDielectricGeneratorVersion = 44u;
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
  float3 terminal_source = {};
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
  std::filesystem::path terminal_source;
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
  uint32_t spectral_channel = 0u;
  uint32_t terminal_source_enabled = 0u;
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
  float4 coating_eta = {};
  float4 coating_k = {};
  float coating_thickness_nm = 0.0f;
  uint32_t coating_cls = SpectralDistribution::Invalid;
  BSDFEnergyCompensationDielectricLut dielectric_lut = {};
};

static_assert(sizeof(EnergyCompensationGpuParams) == 248u);
static_assert(offsetof(EnergyCompensationGpuParams, output_directional_index) == 0u);
static_assert(offsetof(EnergyCompensationGpuParams, cache_mode) == 28u);
static_assert(offsetof(EnergyCompensationGpuParams, spectral_channel) == 56u);
static_assert(offsetof(EnergyCompensationGpuParams, terminal_source_enabled) == 60u);
static_assert(offsetof(EnergyCompensationGpuParams, ext_eta) == 64u);
static_assert(offsetof(EnergyCompensationGpuParams, int_eta) == 96u);
static_assert(offsetof(EnergyCompensationGpuParams, film_eta) == 128u);
static_assert(offsetof(EnergyCompensationGpuParams, wavelengths) == 160u);
static_assert(offsetof(EnergyCompensationGpuParams, thinfilm_thickness) == 176u);
static_assert(offsetof(EnergyCompensationGpuParams, film_cls) == 180u);
static_assert(offsetof(EnergyCompensationGpuParams, thinfilm_weight) == 184u);
static_assert(offsetof(EnergyCompensationGpuParams, coating_eta) == 192u);
static_assert(offsetof(EnergyCompensationGpuParams, coating_k) == 208u);
static_assert(offsetof(EnergyCompensationGpuParams, coating_thickness_nm) == 224u);
static_assert(offsetof(EnergyCompensationGpuParams, coating_cls) == 228u);
static_assert(offsetof(EnergyCompensationGpuParams, dielectric_lut) == 232u);

struct EnergyCompensationGpuPushConstants {
  uint32_t params_buffer_index = kInvalidIndex;
};

struct EnergyCompensationGpuBuffers {
  RHIBuffer params_directional[kEnergyCompensationSpectralWavelengthGroupSize] = {};
  uint32_t params_directional_count = 0u;
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

enum class EnergyCompensationAsyncStepKind : uint32_t {
  Conductor,
  Dielectric,
};

enum class CacheGenerationStepResult : uint32_t {
  Complete,
  Pending,
  Generated,
  Failed,
};

struct EnergyCompensationAsyncStep {
  EnergyCompensationGpuBuffers buffers = {};
  RHICommandBuffer command_buffer = {};
  GeneratedInterfacePaths paths = {};
  GeneratedInterfacePaths geometric_paths = {};
  TaskScheduler* scheduler = nullptr;
  Task::Handle save_task = {};
  std::vector<float4> directional_pixels = {};
  std::vector<float4> average_pixels = {};
  std::vector<float4> geometric_pixels = {};
  std::vector<float4> geometric_average_pixels = {};
  std::vector<float4> conductor_fms_pixels = {};
  std::vector<float4> probability_pixels = {};
  std::vector<float4> terminal_source_pixels = {};
  uint32_t cache_mode = kBSDFEnergyCompensationCacheModeIntegratedRGB;
  EnergyCompensationAsyncStepKind kind = EnergyCompensationAsyncStepKind::Dielectric;
  bool save_spectral_geometric = false;
  bool gpu_complete = false;
  bool save_result = false;
  BSDFEnergyCompensationDielectricLut dielectric_lut = {};
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

SpectralQuery spectral_cache_query(uint32_t wavelength_index) {
  const float t = static_cast<float>(min(wavelength_index, kEnergyCompensationSpectralWavelengthCount - 1u)) / static_cast<float>(kEnergyCompensationSpectralWavelengthCount - 1u);
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
  static_cast<::RefractiveIndexSample&>(result) = refractive_index_sample_transport(result);
  return result;
}

ThinfilmEval sample_constant_thinfilm(const SceneData& data, const Thinfilm& thinfilm, const SpectralQuery& spect) {
  if (bsdf_resource_thinfilm_enabled(thinfilm) == false) {
    return empty_thinfilm();
  }

  ThinfilmEval result = {};
  result.ior = sample_refractive_index(data, thinfilm.ior, spect);
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

  const float estimated_count = ceilf(thickness_range / 50.0f) + 1.0f;
  return static_cast<uint32_t>(clamp(estimated_count, 4.0f, 32.0f));
}

uint32_t coherent_coating_lut_slice_count(const CoherentCoating& coating) {
  if (coating.max_thickness_nm <= 0.0f) {
    return 1u;
  }
  const float range = fabsf(coating.max_thickness_nm - coating.min_thickness_nm);
  if (range <= kEpsilon) {
    return 1u;
  }
  const float estimated_count = ceilf(range / 50.0f) + 1.0f;
  return static_cast<uint32_t>(clamp(estimated_count, 4.0f, 32.0f));
}

uint32_t interface_lut_slice_count(const Material& material) {
  return thinfilm_lut_slice_count(material.thinfilm) * coherent_coating_lut_slice_count(material.coherent_coating);
}

bool dielectric_interface_is_untextured_delta(const Material& material) {
  const float alpha_x = material.roughness.value.x;
  const float alpha_y = material.roughness.value.y;
  return (material.roughness.image_index == kInvalidIndex) && std::isfinite(alpha_x) && std::isfinite(alpha_y) && (alpha_x >= 0.0f) && (alpha_y >= 0.0f) &&
         (max(alpha_x, alpha_y) <= kDeltaAlphaTreshold);
}

bool conductor_terminal_source_required(const Material& material) {
  // A valid coherent carrier also requests the zero-thickness source-reference basis.
  return (material.coherent_coating.ior.cls == SpectralDistribution::Dielectric) || (material.coherent_coating.ior.cls == SpectralDistribution::Conductor);
}

BSDFEnergyCompensationDielectricLut dielectric_lut_layout(const Material& material) {
  BSDFEnergyCompensationDielectricLut result = {};
  const float2 roughness = {material.roughness.value.x, material.roughness.value.y};
  if ((material.cls != MaterialClass::Dielectric) || bsdf_energy_compensated_roughness_isotropic(roughness)) {
    return result;
  }
  result.azimuth_count = kBSDFEnergyCompensationDielectricAzimuthCount;
  if (material.roughness.image_index == kInvalidIndex) {
    result.roughness = roughness;
    result.roughness_count = 1u;
  } else {
    // A roughness texture scales both authored axes, so their ratio is fixed.
    result.roughness = roughness / (0.5f * (roughness.x + roughness.y));
  }
  return result;
}

float2 dielectric_roughness_parameter(const BSDFEnergyCompensationDielectricLut& layout, uint32_t index) {
  return layout.roughness_count == 1u ? layout.roughness : layout.roughness * alpha_parameter(index, layout.roughness_count);
}

bool interface_lut_dimensions_valid(const Material& material, uint32_t material_class, uint32_t cache_mode, SceneDiagnostic& diagnostic) {
  const auto layout = dielectric_lut_layout(material);
  const uint64_t slices = thinfilm_lut_slice_count(material.thinfilm);
  const uint64_t groups = cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar ? kEnergyCompensationSpectralWavelengthGroupCount : 1u;
  const uint64_t branches = material_class == MaterialClass::Conductor ? 1u : kEnergyCompensationDielectricBranchCount;
  const uint64_t height = ((material_class == MaterialClass::Conductor) && conductor_terminal_source_required(material))
                            ? kBSDFEnergyCompensationConductorTerminalLutHeight
                            : (material_class == MaterialClass::Conductor ? kEnergyCompensationConductorLutSize : layout.roughness_count);
  const uint64_t bytes =
    branches * kEnergyCompensationDielectricLutSize * (material_class == MaterialClass::Dielectric ? layout.azimuth_count : 1u) * height * slices * groups * sizeof(float4);
  if (bytes > std::numeric_limits<uint32_t>::max()) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Material interface LUT requires %llu bytes per image, exceeding the 32-bit image payload limit",
      static_cast<unsigned long long>(bytes));
    return false;
  }
  return true;
}

InterfaceOptics sample_interface_slice(const SceneData& data, const Material& material, const SpectralQuery& spect, uint32_t slice_index, uint32_t slice_count) {
  const Thinfilm& thinfilm = material.thinfilm;
  InterfaceOptics result = sample_constant_thinfilm(data, thinfilm, spect);
  // Zero-thickness source-reference interfaces retain the shared reflection/source quadrature.
  result.coating_ior.cls = material.coherent_coating.ior.cls;
  const uint32_t film_count = thinfilm_lut_slice_count(thinfilm);
  if (bsdf_resource_thinfilm_enabled(thinfilm) && (film_count > 1u) && (slice_count > 1u)) {
    const float t = static_cast<float>(slice_index % film_count) / static_cast<float>(film_count - 1u);
    const float minimum_thickness = max(0.0f, thinfilm.min_thickness);
    const float maximum_thickness = max(0.0f, thinfilm.max_thickness);
    result.thickness = minimum_thickness + (maximum_thickness - minimum_thickness) * t;
  }
  if (material.coherent_coating.max_thickness_nm > 0.0f) {
    const CoherentCoating& coating = material.coherent_coating;
    const uint32_t coating_count = coherent_coating_lut_slice_count(coating);
    const float t = ((coating_count > 1u) && (slice_count > 1u)) ? static_cast<float>(slice_index / film_count) / static_cast<float>(coating_count - 1u) : 0.0f;
    result.coating_ior = sample_refractive_index(data, coating.ior, spect);
    result.coating_thickness_nm = coating.min_thickness_nm + (coating.max_thickness_nm - coating.min_thickness_nm) * t;
  }
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

bool gpu_create_storage_buffer(RHIContext& rhi, const void* data, uint64_t size, RHIBufferUsage usage, bool host_visible, RHIBuffer& out_buffer, const char* debug_name,
  SceneDiagnostic& diagnostic) {
  if (size == 0u) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Invalid BSDF energy-compensation GPU buffer size for %s", debug_name);
    return false;
  }

  RHIBufferDesc desc = {};
  desc.size = size;
  desc.usage = RHIBufferUsage::Storage | usage;
  desc.host_visible = host_visible;
  auto result = rhi.device().create_buffer(desc);
  if ((result.result != RHIResult::Success) || (result.handle.valid() == false)) {
    diagnostic.set_result(result.result);
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to create BSDF energy-compensation GPU buffer %s (%s)", debug_name,
      rhi_result_name(result.result));
    return false;
  }

  if (data != nullptr) {
    const RHIResult update_result = rhi.device().update_buffer(result.handle, data, size);
    if (update_result != RHIResult::Success) {
      diagnostic.set_result(update_result);
      diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to upload BSDF energy-compensation GPU buffer %s (%s)", debug_name,
        rhi_result_name(update_result));
      rhi.device().destroy_buffer(result.handle);
      return false;
    }
  }

  out_buffer = result.handle;
  return true;
}

bool gpu_create_params_buffer(RHIContext& rhi, const EnergyCompensationGpuParams& params, RHIBuffer& out_buffer, const char* debug_name, SceneDiagnostic& diagnostic) {
  return gpu_create_storage_buffer(rhi, &params, sizeof(params), RHIBufferUsage::TransferDst, true, out_buffer, debug_name, diagnostic);
}

bool gpu_create_float4_output_buffer(RHIContext& rhi, uint64_t element_count, RHIBuffer& out_buffer, const char* debug_name, SceneDiagnostic& diagnostic) {
  return gpu_create_storage_buffer(rhi, nullptr, element_count * sizeof(float4), RHIBufferUsage::TransferSrc, false, out_buffer, debug_name, diagnostic);
}

bool gpu_create_float4_async_output_buffer(RHIContext& rhi, uint64_t element_count, RHIBuffer& out_buffer, const char* debug_name, SceneDiagnostic& diagnostic) {
  return gpu_create_storage_buffer(rhi, nullptr, element_count * sizeof(float4), static_cast<RHIBufferUsage>(0u), true, out_buffer, debug_name, diagnostic);
}

void gpu_destroy_buffer_if_valid(RHIContext& rhi, RHIBuffer& buffer) {
  if (buffer.valid()) {
    rhi.device().destroy_buffer(buffer);
    buffer = {};
  }
}

void gpu_destroy_energy_compensation_buffers(RHIContext& rhi, EnergyCompensationGpuBuffers& buffers) {
  for (RHIBuffer& params_buffer : buffers.params_directional) {
    gpu_destroy_buffer_if_valid(rhi, params_buffer);
  }
  gpu_destroy_buffer_if_valid(rhi, buffers.params_average);
  gpu_destroy_buffer_if_valid(rhi, buffers.directional);
  gpu_destroy_buffer_if_valid(rhi, buffers.average);
  gpu_destroy_buffer_if_valid(rhi, buffers.geometric);
  gpu_destroy_buffer_if_valid(rhi, buffers.geometric_average);
  gpu_destroy_buffer_if_valid(rhi, buffers.conductor_fms);
  gpu_destroy_buffer_if_valid(rhi, buffers.probability);
  gpu_destroy_buffer_if_valid(rhi, buffers.total);
}

bool gpu_create_float4_readback_buffer(RHIContext& rhi, uint64_t element_count, RHIBuffer& out_buffer, const char* debug_name, SceneDiagnostic& diagnostic) {
  const uint64_t size = element_count * sizeof(float4);
  RHIBufferDesc desc = {};
  desc.size = size;
  desc.usage = RHIBufferUsage::TransferDst;
  desc.host_visible = true;

  auto result = rhi.device().create_buffer(desc);
  if ((result.result != RHIResult::Success) || (result.handle.valid() == false)) {
    diagnostic.set_result(result.result);
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to create BSDF energy-compensation GPU readback buffer %s (%s)", debug_name,
      rhi_result_name(result.result));
    return false;
  }

  out_buffer = result.handle;
  return true;
}

bool gpu_read_float4_output_buffer(RHIContext& rhi, RHIBuffer buffer, uint64_t element_count, std::vector<float4>& out_pixels, const char* debug_name,
  SceneDiagnostic& diagnostic) {
  if (buffer.valid() == false) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Invalid BSDF energy-compensation GPU output buffer %s", debug_name);
    return false;
  }

  const uint64_t size = element_count * sizeof(float4);
  RHIBuffer readback_buffer = {};
  RHICommandBuffer cmd = {};
  bool success = false;

  do {
    if (gpu_create_float4_readback_buffer(rhi, element_count, readback_buffer, debug_name, diagnostic) == false) {
      break;
    }

    cmd = rhi.get_command_buffer();
    if (cmd.valid() == false) {
      diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to acquire command buffer for BSDF energy-compensation GPU readback %s", debug_name);
      break;
    }

    rhi.command_buffer_begin(cmd);
    rhi.cmd_buffer_barrier(cmd, buffer, RHIResourceState::General, RHIResourceState::TransferSrc);
    rhi.cmd_copy_buffer(cmd, buffer, readback_buffer, size);
    rhi.command_buffer_end(cmd);
    const RHIResult submit_result = rhi.submit_command_buffer({cmd});
    if (submit_result != RHIResult::Success) {
      diagnostic.set_result(submit_result);
      diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to submit BSDF energy-compensation GPU readback %s (%s)", debug_name,
        rhi_result_name(submit_result));
      break;
    }

    const RHIResult wait_result = rhi.wait_for_command_buffer(cmd);
    if (wait_result != RHIResult::Success) {
      diagnostic.set_result(wait_result);
      diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to wait for BSDF energy-compensation GPU readback %s (%s)", debug_name,
        rhi_result_name(wait_result));
      break;
    }

    out_pixels.resize(static_cast<size_t>(element_count));
    const RHIResult read_result = rhi.device().read_buffer(readback_buffer, out_pixels.data(), size);
    if (read_result != RHIResult::Success) {
      diagnostic.set_result(read_result);
      diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to read BSDF energy-compensation GPU buffer %s (%s)", debug_name,
        rhi_result_name(read_result));
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

bool gpu_init_energy_compensation_pipeline(RHIContext& rhi, EnergyCompensationGpuPipeline& out_pipeline, SceneDiagnostic& diagnostic) {
  if (out_pipeline.pipeline.valid()) {
    return true;
  }
  if (out_pipeline.initialization_failed) {
    return false;
  }

  auto compilation = ShaderCompiler::instance().compile("shaders/bsdf_energy_compensation.hlsl", {{"main", RHIShaderStage::Compute}}, {}, rhi.backend());
  if (compilation.result != RHIResult::Success) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to compile BSDF energy-compensation GPU shader: %s", compilation.error_message.c_str());
    out_pipeline.initialization_failed = true;
    return false;
  }
  if (compilation.binaries.empty()) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "BSDF energy-compensation GPU shader compilation returned no binaries");
    out_pipeline.initialization_failed = true;
    return false;
  }

  RHIComputePipelineDesc desc = rhi.device().make_compute_pipeline_desc(compilation.binaries[0]);
  auto pipeline_result = rhi.device().create_compute_pipeline(desc);
  if ((pipeline_result.result != RHIResult::Success) || (pipeline_result.handle.valid() == false)) {
    diagnostic.set_result(pipeline_result.result);
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to create BSDF energy-compensation GPU pipeline (%s)",
      rhi_result_name(pipeline_result.result));
    out_pipeline.initialization_failed = true;
    return false;
  }

  out_pipeline.pipeline = pipeline_result.handle;
  return true;
}

bool gpu_submit_energy_compensation(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const EnergyCompensationGpuBuffers& buffers, uint32_t width, uint32_t height,
  bool run_average_pass, bool asynchronous, RHICommandBuffer& out_command_buffer, SceneDiagnostic& diagnostic) {
  if (gpu_init_energy_compensation_pipeline(rhi, pipeline, diagnostic) == false) {
    return false;
  }

  RHICommandBuffer cmd = asynchronous ? rhi.get_async_command_buffer() : rhi.get_command_buffer();
  if (cmd.valid() == false) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to acquire command buffer for BSDF energy-compensation GPU generation");
    return false;
  }

  rhi.command_buffer_begin(cmd);
  auto dispatch_pass = [&](RHIBuffer params_buffer, uint32_t dispatch_width, uint32_t dispatch_height, bool synchronize_outputs) {
    if (synchronize_outputs) {
      rhi.cmd_buffer_barrier(cmd, buffers.directional, RHIResourceState::General, RHIResourceState::General);
      if (buffers.geometric.valid()) {
        rhi.cmd_buffer_barrier(cmd, buffers.geometric, RHIResourceState::General, RHIResourceState::General);
      }
      if (buffers.total.valid()) {
        rhi.cmd_buffer_barrier(cmd, buffers.total, RHIResourceState::General, RHIResourceState::General);
      }
      if (buffers.probability.valid()) {
        rhi.cmd_buffer_barrier(cmd, buffers.probability, RHIResourceState::General, RHIResourceState::General);
      }
    }
    rhi.cmd_set_pipeline(cmd, pipeline.pipeline);

    EnergyCompensationGpuPushConstants pc = {};
    pc.params_buffer_index = get_bindless_descriptor_index(params_buffer);
    rhi.cmd_push_constants(cmd, &pc, sizeof(pc), 0);
    RHIDispatchDesc dispatch = {};
    dispatch.group_count_x = (dispatch_width + 7u) / 8u;
    dispatch.group_count_y = (dispatch_height + 7u) / 8u;
    dispatch.group_count_z = 1u;
    rhi.cmd_dispatch(cmd, dispatch);
  };

  for (uint32_t pass_index = 0u; pass_index < buffers.params_directional_count; ++pass_index) {
    dispatch_pass(buffers.params_directional[pass_index], width, height, pass_index > 0u);
  }
  if (run_average_pass) {
    dispatch_pass(buffers.params_average, width, 1u, true);
  }

  rhi.command_buffer_end(cmd);
  const RHIResult submit_result = rhi.submit_command_buffer({cmd});
  if (submit_result != RHIResult::Success) {
    diagnostic.set_result(submit_result);
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to submit BSDF energy-compensation GPU generation (%s)", rhi_result_name(submit_result));
    rhi.destroy_command_buffer(cmd);
    return false;
  }
  out_command_buffer = cmd;
  return true;
}

bool gpu_dispatch_energy_compensation(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const EnergyCompensationGpuBuffers& buffers, uint32_t width, uint32_t height,
  bool run_average_pass, SceneDiagnostic& diagnostic) {
  RHICommandBuffer cmd = {};
  if (gpu_submit_energy_compensation(rhi, pipeline, buffers, width, height, run_average_pass, false, cmd, diagnostic) == false) {
    return false;
  }

  const RHIResult wait_result = rhi.wait_for_command_buffer(cmd);
  rhi.destroy_command_buffer(cmd);
  if (wait_result != RHIResult::Success) {
    diagnostic.set_result(wait_result);
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to wait for BSDF energy-compensation GPU generation (%s)", rhi_result_name(wait_result));
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
  result.terminal_source_enabled =
    ((pass_kind == kEnergyCompensationGpuPassConductorDirectional) || (pass_kind == kEnergyCompensationGpuPassConductorAverage)) && conductor_terminal_source_required(material);

  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    fill_gpu_refractive_index_spectral(data, material.ext_ior, wavelength_group_index, result.ext_eta, result.ext_k, result.wavelengths);
    float4 unused_wavelengths = {};
    fill_gpu_refractive_index_spectral(data, material.int_ior, wavelength_group_index, result.int_eta, result.int_k, unused_wavelengths);
    const SpectralQuery spect = spectral_cache_query(wavelength_group_index * kEnergyCompensationSpectralWavelengthGroupSize);
    const ThinfilmEval thinfilm = sample_interface_slice(data, material, spect, thinfilm_slice_index, thinfilm_slice_count);
    result.thinfilm_thickness = thinfilm.thickness;
    result.film_cls = thinfilm.ior.cls;
    result.thinfilm_weight = thinfilm.weight;
    fill_gpu_refractive_index_spectral(data, material.thinfilm.ior, wavelength_group_index, result.film_eta, result.film_k, unused_wavelengths);
    result.coating_thickness_nm = thinfilm.coating_thickness_nm;
    result.coating_cls = thinfilm.coating_ior.cls;
    fill_gpu_refractive_index_spectral(data, material.coherent_coating.ior, wavelength_group_index, result.coating_eta, result.coating_k, unused_wavelengths);
  } else {
    fill_gpu_refractive_index_integrated(data, material.ext_ior, result.ext_eta, result.ext_k);
    fill_gpu_refractive_index_integrated(data, material.int_ior, result.int_eta, result.int_k);
    const SpectralQuery spect = {};
    const ThinfilmEval thinfilm = sample_interface_slice(data, material, spect, thinfilm_slice_index, thinfilm_slice_count);
    result.thinfilm_thickness = thinfilm.thickness;
    result.film_cls = thinfilm.ior.cls;
    result.thinfilm_weight = thinfilm.weight;
    result.film_eta = spectral_response_to_gpu_float4(thinfilm.ior.eta);
    result.film_k = spectral_response_to_gpu_float4(thinfilm.ior.k);
    result.coating_thickness_nm = thinfilm.coating_thickness_nm;
    result.coating_cls = thinfilm.coating_ior.cls;
    result.coating_eta = spectral_response_to_gpu_float4(thinfilm.coating_ior.eta);
    result.coating_k = spectral_response_to_gpu_float4(thinfilm.coating_ior.k);
  }

  result.dielectric_lut = dielectric_lut_layout(material);
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
  constexpr uint32_t film_model_version = 5u;
  result = etx_hash64_continue(&film_model_version, sizeof(film_model_version), result);
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
  const uint32_t generator_version = (material_class == MaterialClass::Dielectric ? kEnergyCompensationDielectricGeneratorVersion : kEnergyCompensationGeneratorVersion) +
                                     uint32_t(cache_mode == kBSDFEnergyCompensationCacheModeIntegratedRGB);
  result = etx_hash64_continue(&generator_version, sizeof(generator_version), result);
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
  if ((material_class == MaterialClass::Conductor) && conductor_terminal_source_required(material)) {
    constexpr uint32_t terminal_source_model_version = 5u;
    result = etx_hash64_continue(&terminal_source_model_version, sizeof(terminal_source_model_version), result);
  }
  const bool coherent_enabled = material.coherent_coating.max_thickness_nm > 0.0f;
  if (coherent_enabled) {
    constexpr uint32_t coherent_model_version = 3u;
    result = etx_hash64_continue(&coherent_model_version, sizeof(coherent_model_version), result);
    result = hash_refractive_index(data, material.coherent_coating.ior, result);
    result = etx_hash64_continue(&material.coherent_coating.min_thickness_nm, sizeof(float), result);
    result = etx_hash64_continue(&material.coherent_coating.max_thickness_nm, sizeof(float), result);
    const uint32_t coating_count = coherent_coating_lut_slice_count(material.coherent_coating);
    result = etx_hash64_continue(&coating_count, sizeof(coating_count), result);
  }
  if (material_class == MaterialClass::Dielectric) {
    const auto layout = dielectric_lut_layout(material);
    if (layout.azimuth_count > 1u) {
      constexpr uint32_t anisotropic_model_version = 2u;
      result = etx_hash64_continue(&anisotropic_model_version, sizeof(anisotropic_model_version), result);
      result = etx_hash64_continue(&layout, sizeof(layout), result);
    }
  }
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
    directory / (key + "_terminal_source.exr"),
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
    directory / (std::string(key) + "_terminal_source.exr"),
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
    slice_path(paths.terminal_source, slice_index),
  };
}

bool save_exr_rgba(const std::filesystem::path& path, const std::vector<float4>& pixels, uint32_t width, uint32_t height, SceneDiagnostic& diagnostic) {
  const uint64_t expected_pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  if (pixels.size() != expected_pixel_count) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Invalid energy-compensation LUT dimensions for %s", path.generic_string().c_str());
    return false;
  }

  const std::filesystem::path parent_path = path.parent_path();
  if (parent_path.empty() == false) {
    std::error_code ec = {};
    std::filesystem::create_directories(parent_path, ec);
    if (ec.value() != 0) {
      diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to create energy-compensation LUT cache directory %s",
        parent_path.generic_string().c_str());
      return false;
    }
  }

  const std::string file_name = path.generic_string();
  std::string error;
  if (!save_exr_image(file_name.c_str(), pixels.data(), {width, height}, &error)) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to save energy-compensation LUT %s: %s", file_name.c_str(), error.c_str());
    return false;
  }

  return true;
}

bool load_rgba32f_lut(const std::filesystem::path& path, uint32_t width, uint32_t height, std::vector<float4>& out_pixels, SceneDiagnostic& diagnostic) {
  std::vector<uint8_t> source_data;
  uint2 dimensions = {};
  const Image::Format format = load_data(path.generic_string().c_str(), source_data, dimensions);
  if ((format != Image::Format::RGBA32F) || (dimensions.x != width) || (dimensions.y != height)) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to load energy-compensation LUT %s", path.generic_string().c_str());
    return false;
  }

  const uint64_t pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  if (source_data.size() != pixel_count * sizeof(float4)) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Invalid energy-compensation LUT size %s", path.generic_string().c_str());
    return false;
  }

  out_pixels.resize(static_cast<size_t>(pixel_count));
  memcpy(out_pixels.data(), source_data.data(), source_data.size());
  return true;
}

bool load_rgba32f_lut_slices(const GeneratedInterfacePaths& paths, uint32_t width, uint32_t height, uint32_t slice_count, std::vector<float4>& out_pixels,
  const std::filesystem::path GeneratedInterfacePaths::* member, uint32_t first_slice, SceneDiagnostic& diagnostic) {
  const uint64_t slice_pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  out_pixels.resize(static_cast<size_t>(slice_pixel_count * slice_count));
  for (uint32_t slice_index = 0u; slice_index < slice_count; ++slice_index) {
    const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, first_slice + slice_index);
    std::vector<float4> slice_pixels;
    if (load_rgba32f_lut(slice_paths_value.*member, width, height, slice_pixels, diagnostic) == false) {
      return false;
    }
    std::copy(slice_pixels.begin(), slice_pixels.end(), out_pixels.begin() + static_cast<size_t>(slice_pixel_count * slice_index));
  }
  return true;
}

SpectralDirectionalAlbedoResult integrate_conductor_directional(const SpectralQuery& spect, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior,
  const ThinfilmEval& thinfilm, bool terminal_source, float mu_i, float alpha) {
  SpectralDirectionalAlbedoResult result = {};
  if (mu_i <= kEpsilon) {
    return result;
  }

  const float2 alpha2 = float2(alpha, alpha);
  const float3 w_i = incident_direction_from_mu(mu_i);
  const float lambda_i = bsdf_external_ray_info_make(w_i, alpha2).Lambda;
  const float g1_i = 1.0f / (1.0f + lambda_i);

  for (uint32_t sample_index = 0u; sample_index < kEnergyCompensationConductorSampleCount; ++sample_index) {
    const float3 m = bsdf_energy_compensated_sample_vndf_local(w_i, alpha, hammersley(sample_index, kEnergyCompensationConductorSampleCount));
    const float i_dot_m = dot(w_i, m);
    if ((m.z <= kEpsilon) || (i_dot_m <= kEpsilon)) {
      continue;
    }

    const float3 w_o = -w_i + 2.0f * m * i_dot_m;
    if (w_o.z > 0.0f) {
      const float lambda_o = bsdf_external_ray_info_make(w_o, alpha2).Lambda;
      const float geometric_weight = (1.0f + lambda_i) / (1.0f + lambda_i + lambda_o);
      const auto fresnel = bsdf_fresnel_calculate(spect, i_dot_m, ext_ior, int_ior, thinfilm);
      const float3 albedo = spectral_response_is_spectral(fresnel) ? float3(fresnel.value, 0.0f, 0.0f) : fresnel.integrated;
      result.albedo += albedo * geometric_weight;
      if (terminal_source) {
        const auto terminal = bsdf_terminal_source_normal_sample(spect, ext_ior, int_ior, thinfilm, w_i, m, alpha, geometric_weight);
        result.terminal_source += spectral_response_is_spectral(terminal) ? float3(terminal.value, 0.0f, 0.0f) : terminal.integrated;
      }
      result.geometric_albedo += geometric_weight;
      if (g1_i > kEpsilon) {
        result.visible_probability += 1.0f;
      }
    }
  }

  if (terminal_source) {
    for (uint32_t sample_index = 0u; sample_index < kEnergyCompensationConductorSampleCount; ++sample_index) {
      const auto terminal = bsdf_terminal_source_guided_sample(spect, ext_ior, int_ior, thinfilm, w_i, alpha, hammersley(sample_index, kEnergyCompensationConductorSampleCount));
      result.terminal_source += spectral_response_is_spectral(terminal) ? float3(terminal.value, 0.0f, 0.0f) : terminal.integrated;
    }
  }

  const float inv_sample_count = 1.0f / static_cast<float>(kEnergyCompensationConductorSampleCount);
  result.albedo = saturate(result.albedo * inv_sample_count);
  result.terminal_source *= inv_sample_count;
  result.geometric_albedo = saturate(result.geometric_albedo * inv_sample_count);
  result.visible_probability = saturate(result.visible_probability * inv_sample_count);
  return result;
}

DielectricDirectionalAlbedoResult integrate_dielectric_directional(const SpectralQuery& spect, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior,
  const ThinfilmEval& thinfilm, bool incident_outside, const float3& w_i, const float2& alpha) {
  DielectricDirectionalAlbedoResult result = {};
  if (w_i.z <= kEpsilon) {
    return result;
  }

  const uint32_t incident_side = dielectric_side(incident_outside);
  const uint32_t opposite_side = 1u - incident_side;

  const bool no_thinfilm = bsdf_resource_interface_has_layers(thinfilm) == false;
  if (no_thinfilm && bsdf_fresnel_equal_indices(ext_ior, int_ior)) {
    result.branch_albedo[opposite_side] = float3(1.0f, 1.0f, 1.0f);
    result.branch_visible_probability[opposite_side] = 1.0f;
    result.visible_probability = 1.0f;
    return result;
  }

  const InterfaceOptics oriented = bsdf_fresnel_orient_coherent_interface(thinfilm, incident_outside);
  const float lambda_i = bsdf_external_ray_info_make(w_i, alpha).Lambda;
  for (uint32_t sample_index = 0u; sample_index < kEnergyCompensationSampleCount; ++sample_index) {
    const float3 m = bsdf_energy_compensated_sample_vndf_local(w_i, alpha, hammersley(sample_index, kEnergyCompensationSampleCount));
    const float i_dot_m = dot(w_i, m);
    if (i_dot_m <= kEpsilon) {
      continue;
    }

    const auto fresnel = bsdf_fresnel_calculate(spect, i_dot_m, ext_ior, int_ior, oriented);
    const auto transmission = bsdf_fresnel_transmission_calculate(spect, i_dot_m, ext_ior, int_ior, oriented);
    const float fresnel_probability = spectral_response_monochromatic(fresnel);
    const float cos_theta_t = bsdf_external_transmitted_cosine(i_dot_m, ext_ior, int_ior);

    const float3 w_o_r = -w_i + 2.0f * m * i_dot_m;
    if (w_o_r.z > kEpsilon) {
      result.branch_visible_probability[incident_side] += fresnel_probability;
    }
    if (w_o_r.z > kEpsilon) {
      const float lambda_o = bsdf_external_ray_info_make(w_o_r, alpha).Lambda;
      const float weight = (1.0f + lambda_i) / (1.0f + lambda_i + lambda_o);
      const float3 albedo = spectral_response_is_spectral(fresnel) ? float3(fresnel.value, 0.0f, 0.0f) : fresnel.integrated;
      result.branch_albedo[incident_side] += albedo * weight;
    }

    if ((spectral_response_is_zero(transmission) == false) && (cos_theta_t > 0.0f)) {
      const float3 w_o_t = normalize(bsdf_external_refract(w_i, m, ext_ior, int_ior));
      if (w_o_t.z < -kEpsilon) {
        result.branch_visible_probability[opposite_side] += spectral_response_monochromatic(transmission);
      }
      if (w_o_t.z < -kEpsilon) {
        const float lambda_o = bsdf_external_ray_info_make(-w_o_t, alpha).Lambda;
        const float weight = (1.0f + lambda_i) * bsdf_external_beta(1.0f + lambda_i, 1.0f + lambda_o);
        const float3 albedo = spectral_response_is_spectral(transmission) ? float3(transmission.value, 0.0f, 0.0f) : transmission.integrated;
        result.branch_albedo[opposite_side] += albedo * weight;
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

bool dielectric_lut_values_valid(const float4& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) && std::isfinite(value.w) && (value.x >= 0.0f) && (value.y >= 0.0f) && (value.z >= 0.0f) &&
         (value.w >= 0.0f);
}

bool dielectric_lut_pixels_valid(const std::vector<float4>& pixels, SceneDiagnostic& diagnostic) {
  for (const float4& value : pixels) {
    if (dielectric_lut_values_valid(value) == false) {
      diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Dielectric interface generation produced a numerical failure");
      return false;
    }
  }
  return true;
}

bool dielectric_multiscatter_results_valid(const std::vector<DielectricDirectionalAlbedoResult>& directional, SceneDiagnostic& diagnostic) {
  for (const DielectricDirectionalAlbedoResult& value : directional) {
    for (uint32_t side = 0u; side < 2u; ++side) {
      const float3& branch = value.branch_albedo[side];
      if (dielectric_lut_values_valid(float4(branch.x, branch.y, branch.z, value.visible_probability)) == false) {
        diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Dielectric interface generation produced a numerical failure");
        return false;
      }
    }
  }
  return true;
}

bool dielectric_lut_ray_finite(const BSDFExternalRayInfo& ray) {
  return std::isfinite(ray.h) && std::isfinite(ray.w.x) && std::isfinite(ray.w.y) && std::isfinite(ray.w.z) && std::isfinite(ray.Lambda) && std::isfinite(ray.C1) &&
         std::isfinite(ray.G1);
}

DielectricDirectionalAlbedoResult invalid_dielectric_directional_result() {
  DielectricDirectionalAlbedoResult result = {};
  result.branch_albedo[0] = float3(-1.0f, -1.0f, -1.0f);
  result.branch_albedo[1] = result.branch_albedo[0];
  result.visible_probability = -1.0f;
  return result;
}

DielectricDirectionalAlbedoResult integrate_dielectric_multiscatter_directional(const SpectralQuery& spect, const RefractiveIndexSample& ext_ior,
  const RefractiveIndexSample& int_ior, const ThinfilmEval& thinfilm, bool incident_outside, const float3& w_i, const float2& alpha) {
  DielectricDirectionalAlbedoResult result = {};
  if (w_i.z <= kEpsilon) {
    return result;
  }

  const uint32_t incident_side = dielectric_side(incident_outside);
  const uint32_t opposite_side = 1u - incident_side;

  const bool no_thinfilm = bsdf_resource_interface_has_layers(thinfilm) == false;
  if (no_thinfilm && bsdf_fresnel_equal_indices(ext_ior, int_ior)) {
    return result;
  }

  for (uint32_t sample_index = 0u; sample_index < kEnergyCompensationDielectricMultiScatterSampleCount; ++sample_index) {
    BSDFExternalRayInfo ray = bsdf_external_ray_info_make(-w_i, alpha);
    ray = bsdf_external_ray_info_update_height(ray, 1.0f);
    bool ray_outside = true;
    bool valid = true;
    uint32_t scattering_order = 0u;
    uint32_t random_state = sampler_random_seed(sample_index, 0u);
    SpectralResponse throughput = spectral_response_make(spect, 1.0f);

    while (valid) {
      if ((dielectric_lut_ray_finite(ray) == false) || (bsdf_external_response_finite(throughput) == false)) {
        return invalid_dielectric_directional_result();
      }
      const float sampled_height = bsdf_external_sample_height(ray, min(1.0f - kEpsilon, max(kEpsilon, sampler_next_random(random_state))));
      if (sampled_height == kMaxFloat) {
        break;
      }

      ray = bsdf_external_ray_info_update_height(ray, sampled_height);

      const float rnd_slope_x = min(1.0f - kEpsilon, max(kEpsilon, sampler_next_random(random_state)));
      const float rnd_slope_y = min(1.0f - kEpsilon, max(kEpsilon, sampler_next_random(random_state)));
      const float2 rnd_slope = float2{rnd_slope_x, rnd_slope_y};
      const float rnd_reflection = sampler_next_random(random_state);
      const RefractiveIndexSample& phase_ext_ior = ray_outside ? ext_ior : int_ior;
      const RefractiveIndexSample& phase_int_ior = ray_outside ? int_ior : ext_ior;
      const BSDFExternalDielectricSample sample = bsdf_external_sample_phase_function_dielectric(spect, rnd_slope, rnd_reflection, -ray.w, alpha, phase_ext_ior, phase_int_ior,
        bsdf_fresnel_orient_coherent_interface(thinfilm, incident_outside == ray_outside));
      if (sample.numerical_failure) {
        return invalid_dielectric_directional_result();
      }
      if (dot(sample.w_o, sample.w_o) == 0.0f) {
        valid = false;
        break;
      }
      throughput = spectral_response_mul(throughput, sample.weight);
      if (bsdf_external_response_finite(throughput) == false) {
        return invalid_dielectric_directional_result();
      }

      if (sample.reflection) {
        ray = bsdf_external_ray_info_update_direction(ray, sample.w_o, alpha);
        ray = bsdf_external_ray_info_update_height(ray, ray.h);
      } else {
        ray_outside = (ray_outside == false);
        ray = bsdf_external_ray_info_update_direction(ray, -sample.w_o, alpha);
        ray = bsdf_external_ray_info_update_height(ray, -ray.h);
      }

      scattering_order += 1u;
      if (dielectric_lut_ray_finite(ray) == false) {
        return invalid_dielectric_directional_result();
      }
      if (scattering_order > 1u) {
        const float continuation_probability = bsdf_external_continuation_probability(throughput);
        if (continuation_probability == 0.0f) {
          valid = false;
          break;
        }
        if (continuation_probability < 1.0f) {
          if (sampler_next_random(random_state) >= continuation_probability) {
            valid = false;
            break;
          }
          throughput = spectral_response_mul(throughput, 1.0f / continuation_probability);
        }
      }
    }

    if ((valid == false) || (scattering_order <= 1u)) {
      continue;
    }

    const float3 local_w_o = ray_outside ? ray.w : -ray.w;
    const uint32_t outgoing_side = (local_w_o.z > 0.0f) ? incident_side : opposite_side;
    result.branch_albedo[outgoing_side] += spectral_response_is_spectral(throughput) ? float3(throughput.value, 0.0f, 0.0f) : throughput.integrated;
    result.visible_probability += spectral_response_monochromatic(throughput);
    const float3& branch = result.branch_albedo[outgoing_side];
    if (dielectric_lut_values_valid(float4(branch.x, branch.y, branch.z, result.visible_probability)) == false) {
      return invalid_dielectric_directional_result();
    }
  }

  const float inv_sample_count = 1.0f / static_cast<float>(kEnergyCompensationDielectricMultiScatterSampleCount);
  result.branch_albedo[0] *= inv_sample_count;
  result.branch_albedo[1] *= inv_sample_count;
  result.visible_probability *= inv_sample_count;
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

float3 integrate_dielectric_branch_average(const std::vector<DielectricDirectionalAlbedoResult>& directional, uint32_t alpha_index, uint32_t incident_side, uint32_t outgoing_side,
  const BSDFEnergyCompensationDielectricLut& layout) {
  float3 total = {};
  const uint32_t direction_count = kEnergyCompensationDielectricLutSize * layout.azimuth_count;
  const uint32_t side_offset = incident_side * direction_count * layout.roughness_count;
  for (uint32_t azimuth = 0u; azimuth < layout.azimuth_count; ++azimuth) {
    float3 azimuth_total = {};
    const uint32_t offset = side_offset + alpha_index * direction_count + azimuth * kEnergyCompensationDielectricLutSize;
    for (uint32_t mu_index = 0u; mu_index < (kEnergyCompensationDielectricLutSize - 1u); ++mu_index) {
      const float2 weights = bsdf_energy_compensated_dielectric_average_weights(mu_index, kEnergyCompensationDielectricLutSize);
      const float3 value_0 = directional[offset + mu_index].branch_albedo[outgoing_side];
      const float3 value_1 = directional[offset + mu_index + 1u].branch_albedo[outgoing_side];
      azimuth_total += value_0 * weights.x + value_1 * weights.y;
    }
    total += azimuth_total * bsdf_energy_compensated_dielectric_azimuth_weight(azimuth, layout.azimuth_count);
  }
  return total;
}

void complete_dielectric_residual_rows(float3 residual_average[kEnergyCompensationDielectricBranchCount], const float3 side_residual[2], float eta, bool absorbing) {
  const float3 epsilon = {kEpsilon, kEpsilon, kEpsilon};
  const float eta_squared = eta * eta;
  if ((std::isfinite(eta_squared) == false) || (eta_squared <= 0.0f)) {
    for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
      const float failure = std::numeric_limits<float>::quiet_NaN();
      residual_average[branch] = float3(failure, failure, failure);
    }
    return;
  }
  if (absorbing) {
    // Keep the walk's absorption. Only reconcile reciprocal transmission and enforce passive budgets.
    const float3 reflection_0 = min(residual_average[0], side_residual[0]);
    const float3 reflection_1 = min(residual_average[3], side_residual[1]);
    const float3 transmission =
      min(0.5f * (residual_average[1] + eta_squared * residual_average[2]), min(side_residual[0] - reflection_0, eta_squared * (side_residual[1] - reflection_1)));
    residual_average[0] = reflection_0;
    residual_average[1] = transmission;
    residual_average[2] = transmission / eta_squared;
    residual_average[3] = reflection_1;
    return;
  }
  const float3 outside_transmission = residual_average[1] * side_residual[0] / max(epsilon, residual_average[0] + residual_average[1]);
  const float3 inside_transmission = residual_average[2] * side_residual[1] / max(epsilon, residual_average[2] + residual_average[3]);
  // Close both lossless rows while retaining reciprocal transmission across the interface.
  const float3 transmission = min(0.5f * (outside_transmission + eta_squared * inside_transmission), min(side_residual[0], eta_squared * side_residual[1]));
  residual_average[0] = side_residual[0] - transmission;
  residual_average[1] = transmission;
  residual_average[2] = transmission / eta_squared;
  residual_average[3] = side_residual[1] - residual_average[2];
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
  bool terminal_source, TaskScheduler& scheduler, SceneDiagnostic& diagnostic) {
  const SpectralQuery spect = {};
  const uint32_t entry_count = kEnergyCompensationConductorLutSize * kEnergyCompensationConductorLutSize;
  std::vector<SpectralDirectionalAlbedoResult> directional(entry_count);
  scheduler.execute(entry_count, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t alpha_index = index / kEnergyCompensationConductorLutSize;
      const uint32_t mu_index = index - alpha_index * kEnergyCompensationConductorLutSize;
      directional[index] = integrate_conductor_directional(spect, ext_ior, int_ior, thinfilm, terminal_source, mu_parameter(mu_index, kEnergyCompensationConductorLutSize),
        alpha_parameter(alpha_index, kEnergyCompensationConductorLutSize));
    }
  });

  std::vector<float4> image(entry_count, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> geometric_image(entry_count, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> geometric_average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> conductor_fms(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> terminal_pixels;
  if (terminal_source) {
    terminal_pixels.resize(kEnergyCompensationConductorLutSize * kBSDFEnergyCompensationConductorTerminalLutHeight);
  }
  for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationConductorLutSize; ++alpha_index) {
    const float3 average_value = integrate_average(directional, alpha_index, kEnergyCompensationConductorLutSize, 0u);
    average[alpha_index] = float4(average_value.x, average_value.y, average_value.z, 1.0f);
    for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationConductorLutSize; ++mu_index) {
      const uint32_t index = alpha_index * kEnergyCompensationConductorLutSize + mu_index;
      const SpectralDirectionalAlbedoResult& value = directional[index];
      image[index] = float4(value.albedo.x, value.albedo.y, value.albedo.z, value.visible_probability);
      geometric_image[index] = float4(value.geometric_albedo, value.visible_probability, 0.0f, 1.0f);
      if (terminal_source) {
        terminal_pixels[index] = float4(value.terminal_source.x, value.terminal_source.y, value.terminal_source.z, 0.0f);
      }
    }
    const float geometric_average_value = integrate_geometric_average(directional, alpha_index, kEnergyCompensationConductorLutSize);
    geometric_average[alpha_index] = float4(geometric_average_value, 0.0f, 0.0f, 1.0f);
    const ::SpectralResponse fms = bsdf_energy_compensated_conductor_fms(spect, ext_ior, int_ior, thinfilm, geometric_average_value);
    conductor_fms[alpha_index] = float4(fms.integrated.x, fms.integrated.y, fms.integrated.z, 1.0f);
    if (terminal_source) {
      const auto terminal_ms = bsdf_energy_compensated_conductor_terminal_ms(spect, ext_ior, int_ior, thinfilm, geometric_average_value);
      terminal_pixels[entry_count + alpha_index] = float4(terminal_ms.integrated.x, terminal_ms.integrated.y, terminal_ms.integrated.z, 0.0f);
    }
  }

  return save_exr_rgba(paths.directional, image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, diagnostic) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationConductorLutSize, 1u, diagnostic) &&
         save_exr_rgba(paths.geometric, geometric_image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, diagnostic) &&
         save_exr_rgba(paths.geometric_average, geometric_average, kEnergyCompensationConductorLutSize, 1u, diagnostic) &&
         save_exr_rgba(paths.conductor_fms, conductor_fms, kEnergyCompensationConductorLutSize, 1u, diagnostic) &&
         ((terminal_source == false) ||
           save_exr_rgba(paths.terminal_source, terminal_pixels, kEnergyCompensationConductorLutSize, kBSDFEnergyCompensationConductorTerminalLutHeight, diagnostic));
}

bool generate_conductor_geometric_interface(const GeneratedInterfacePaths& paths, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior,
  const ThinfilmEval& thinfilm, TaskScheduler& scheduler, SceneDiagnostic& diagnostic) {
  const SpectralQuery spect = {};
  const uint32_t entry_count = kEnergyCompensationConductorLutSize * kEnergyCompensationConductorLutSize;
  std::vector<SpectralDirectionalAlbedoResult> directional(entry_count);
  scheduler.execute(entry_count, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t alpha_index = index / kEnergyCompensationConductorLutSize;
      const uint32_t mu_index = index - alpha_index * kEnergyCompensationConductorLutSize;
      directional[index] = integrate_conductor_directional(spect, ext_ior, int_ior, thinfilm, false, mu_parameter(mu_index, kEnergyCompensationConductorLutSize),
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

  return save_exr_rgba(paths.geometric, geometric_image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, diagnostic) &&
         save_exr_rgba(paths.geometric_average, geometric_average, kEnergyCompensationConductorLutSize, 1u, diagnostic);
}

bool generate_conductor_interface_spectral(const SceneData& data, const Material& material, const GeneratedInterfacePaths& paths, uint32_t thinfilm_slice_index,
  uint32_t thinfilm_slice_count, uint32_t wavelength_group_index, TaskScheduler& scheduler) {
  const uint32_t entry_count = kEnergyCompensationConductorLutSize * kEnergyCompensationConductorLutSize;
  std::vector<float4> image(entry_count, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> conductor_fms(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  const bool terminal_source = conductor_terminal_source_required(material);
  std::vector<float4> terminal_pixels;
  if (terminal_source) {
    terminal_pixels.resize(kEnergyCompensationConductorLutSize * kBSDFEnergyCompensationConductorTerminalLutHeight);
  }

  for (uint32_t channel = 0u; channel < kEnergyCompensationSpectralWavelengthGroupSize; ++channel) {
    const uint32_t wavelength_index = wavelength_group_index * kEnergyCompensationSpectralWavelengthGroupSize + channel;
    const SpectralQuery spect = spectral_cache_query(wavelength_index);
    const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
    const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
    const ThinfilmEval thinfilm = sample_interface_slice(data, material, spect, thinfilm_slice_index, thinfilm_slice_count);
    std::vector<SpectralDirectionalAlbedoResult> directional(entry_count);
    scheduler.execute(entry_count, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
      (void)thread_id;
      for (uint32_t index = begin; index < end; ++index) {
        const uint32_t alpha_index = index / kEnergyCompensationConductorLutSize;
        const uint32_t mu_index = index - alpha_index * kEnergyCompensationConductorLutSize;
        directional[index] = integrate_conductor_directional(spect, ext_ior, int_ior, thinfilm, terminal_source, mu_parameter(mu_index, kEnergyCompensationConductorLutSize),
          alpha_parameter(alpha_index, kEnergyCompensationConductorLutSize));
      }
    });

    for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationConductorLutSize; ++alpha_index) {
      const float3 average_value = integrate_average(directional, alpha_index, kEnergyCompensationConductorLutSize, 0u);
      set_float4_channel(average[alpha_index], channel, average_value.x);
      for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationConductorLutSize; ++mu_index) {
        const uint32_t index = alpha_index * kEnergyCompensationConductorLutSize + mu_index;
        set_float4_channel(image[index], channel, directional[index].albedo.x);
        if (terminal_source) {
          set_float4_channel(terminal_pixels[index], channel, directional[index].terminal_source.x);
        }
      }

      const float geometric_average_value = integrate_geometric_average(directional, alpha_index, kEnergyCompensationConductorLutSize);
      const ::SpectralResponse fms = bsdf_energy_compensated_conductor_fms(spect, ext_ior, int_ior, thinfilm, geometric_average_value);
      set_float4_channel(conductor_fms[alpha_index], channel, spectral_response_monochromatic(fms));
      if (terminal_source) {
        const auto terminal_ms = bsdf_energy_compensated_conductor_terminal_ms(spect, ext_ior, int_ior, thinfilm, geometric_average_value);
        set_float4_channel(terminal_pixels[entry_count + alpha_index], channel, terminal_ms.value);
      }
    }
  }

  return save_exr_rgba(paths.directional, image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, data.diagnostic) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationConductorLutSize, 1u, data.diagnostic) &&
         save_exr_rgba(paths.conductor_fms, conductor_fms, kEnergyCompensationConductorLutSize, 1u, data.diagnostic) &&
         ((terminal_source == false) ||
           save_exr_rgba(paths.terminal_source, terminal_pixels, kEnergyCompensationConductorLutSize, kBSDFEnergyCompensationConductorTerminalLutHeight, data.diagnostic));
}

bool generate_dielectric_interface(const GeneratedInterfacePaths& paths, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior, const ThinfilmEval& thinfilm,
  const BSDFEnergyCompensationDielectricLut& layout, TaskScheduler& scheduler, SceneDiagnostic& diagnostic) {
  const SpectralQuery spect = {};
  const uint32_t direction_count = kEnergyCompensationDielectricLutSize * layout.azimuth_count;
  const uint32_t side_entry_count = direction_count * layout.roughness_count;
  std::vector<DielectricDirectionalAlbedoResult> directional(side_entry_count * 2u);
  std::vector<DielectricDirectionalAlbedoResult> multiscatter_directional(side_entry_count * 2u);
  scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t side = index / side_entry_count;
      const uint32_t side_index = index - side * side_entry_count;
      const uint32_t alpha_index = side_index / direction_count;
      const uint32_t mu_index = side_index - alpha_index * direction_count;
      const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
      const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
      directional[index] = integrate_dielectric_directional(spect, source_ior, target_ior, thinfilm, side == 0u,
        bsdf_energy_compensated_dielectric_lut_direction(
          bsdf_energy_compensated_dielectric_mu_parameter(mu_index % kEnergyCompensationDielectricLutSize, kEnergyCompensationDielectricLutSize),
          mu_index / kEnergyCompensationDielectricLutSize, layout.azimuth_count),
        dielectric_roughness_parameter(layout, alpha_index));
    }
  });
  scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t side = index / side_entry_count;
      const uint32_t side_index = index - side * side_entry_count;
      const uint32_t alpha_index = side_index / direction_count;
      const uint32_t mu_index = side_index - alpha_index * direction_count;
      const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
      const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
      multiscatter_directional[index] = integrate_dielectric_multiscatter_directional(spect, source_ior, target_ior, thinfilm, side == 0u,
        bsdf_energy_compensated_dielectric_lut_direction(
          bsdf_energy_compensated_dielectric_mu_parameter(mu_index % kEnergyCompensationDielectricLutSize, kEnergyCompensationDielectricLutSize),
          mu_index / kEnergyCompensationDielectricLutSize, layout.azimuth_count),
        dielectric_roughness_parameter(layout, alpha_index));
    }
  });

  if (dielectric_multiscatter_results_valid(multiscatter_directional, diagnostic) == false) {
    return false;
  }
  const uint32_t width = kEnergyCompensationDielectricBranchCount * direction_count;
  std::vector<float4> image(width * layout.roughness_count, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationDielectricAverageWidth * layout.roughness_count, float4{0.0f, 0.0f, 0.0f, 1.0f});
  for (uint32_t alpha_index = 0u; alpha_index < layout.roughness_count; ++alpha_index) {
    float3 single_average[kEnergyCompensationDielectricBranchCount] = {};
    float3 multiscatter_average[kEnergyCompensationDielectricBranchCount] = {};
    float3 residual_average[kEnergyCompensationDielectricBranchCount] = {};

    for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
      for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
        const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
        single_average[branch] = integrate_dielectric_branch_average(directional, alpha_index, incident_side, outgoing_side, layout);
        multiscatter_average[branch] = integrate_dielectric_branch_average(multiscatter_directional, alpha_index, incident_side, outgoing_side, layout);
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
        residual_average[branch] = multiscatter_average[branch];
      }
    }

    complete_dielectric_residual_rows(residual_average, side_residual, bsdf_dielectric_directional_eta(ext_ior, int_ior), bsdf_fresnel_film_absorbs(thinfilm));
    for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
      const float3& residual = residual_average[branch];
      if ((std::isfinite(residual.x) == false) || (std::isfinite(residual.y) == false) || (std::isfinite(residual.z) == false)) {
        diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Dielectric interface reconstruction produced a numerical failure");
        return false;
      }
    }

#if ETX_DEBUG
    const float validation_tolerance = 4.0f / sqrt(static_cast<float>(kEnergyCompensationSampleCount));
    for (uint32_t side = 0u; side < 2u; ++side) {
      const uint32_t branch_0 = dielectric_branch_index(side, 0u);
      const uint32_t branch_1 = dielectric_branch_index(side, 1u);
      const float3 single_row = single_average[branch_0] + single_average[branch_1];
      const float3 model_total_row = multiscatter_average[branch_0] + multiscatter_average[branch_1];
      const float3 total_row = single_row + residual_average[branch_0] + residual_average[branch_1];
      if ((isfinite(single_row.x) == false) || (isfinite(single_row.y) == false) || (isfinite(single_row.z) == false) || (isfinite(model_total_row.x) == false) ||
          (isfinite(model_total_row.y) == false) || (isfinite(model_total_row.z) == false) || (isfinite(total_row.x) == false) || (isfinite(total_row.y) == false) ||
          (isfinite(total_row.z) == false) || (single_row.x > (1.0f + validation_tolerance)) || (single_row.y > (1.0f + validation_tolerance)) ||
          (single_row.z > (1.0f + validation_tolerance)) || (total_row.x > (1.0f + validation_tolerance)) || (total_row.y > (1.0f + validation_tolerance)) ||
          (total_row.z > (1.0f + validation_tolerance)) || (model_total_row.x > (1.0f + validation_tolerance)) || (model_total_row.y > (1.0f + validation_tolerance)) ||
          (model_total_row.z > (1.0f + validation_tolerance))) {
        diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables",
          "Invalid dielectric energy-compensation row alpha %u side %u single %.6f %.6f %.6f model %.6f %.6f %.6f total %.6f %.6f %.6f tolerance %.6f", alpha_index, side,
          single_row.x, single_row.y, single_row.z, model_total_row.x, model_total_row.y, model_total_row.z, total_row.x, total_row.y, total_row.z, validation_tolerance);
        return false;
      }
    }
    for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
      const float3 residual = residual_average[branch];
      if ((isfinite(residual.x) == false) || (isfinite(residual.y) == false) || (isfinite(residual.z) == false) || (residual.x < 0.0f) || (residual.y < 0.0f) ||
          (residual.z < 0.0f)) {
        diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Invalid dielectric energy-compensation residual alpha %u branch %u residual %.6f %.6f %.6f",
          alpha_index, branch, residual.x, residual.y, residual.z);
        return false;
      }
    }
#endif

    for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
      const float3 average_value = saturate(single_average[branch]);
      average[alpha_index * kEnergyCompensationDielectricAverageWidth + branch] = float4(average_value.x, average_value.y, average_value.z, 1.0f);
      const float3 residual = max(float3(0.0f, 0.0f, 0.0f), residual_average[branch]);
      average[alpha_index * kEnergyCompensationDielectricAverageWidth + 4u + branch] = float4(residual.x, residual.y, residual.z, 1.0f);
    }

    for (uint32_t side = 0u; side < 2u; ++side) {
      const uint32_t side_offset = side * side_entry_count;
      for (uint32_t mu_index = 0u; mu_index < direction_count; ++mu_index) {
        const uint32_t source_index = side_offset + alpha_index * direction_count + mu_index;
        const DielectricDirectionalAlbedoResult& value = directional[source_index];
        for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
          const uint32_t branch = dielectric_branch_index(side, outgoing_side);
          const uint32_t x = (mu_index / kEnergyCompensationDielectricLutSize * kEnergyCompensationDielectricBranchCount + branch) * kEnergyCompensationDielectricLutSize +
                             mu_index % kEnergyCompensationDielectricLutSize;
          const float3 branch_value = value.branch_albedo[outgoing_side];
          image[alpha_index * width + x] = float4(branch_value.x, branch_value.y, branch_value.z, value.branch_visible_probability[outgoing_side]);
        }
      }
    }
  }

  if ((dielectric_lut_pixels_valid(image, diagnostic) == false) || (dielectric_lut_pixels_valid(average, diagnostic) == false)) {
    return false;
  }
  return save_exr_rgba(paths.directional, image, width / layout.azimuth_count, layout.roughness_count * layout.azimuth_count, diagnostic) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationDielectricAverageWidth, layout.roughness_count, diagnostic);
}

bool generate_dielectric_interface_spectral(const SceneData& data, const Material& material, const GeneratedInterfacePaths& paths, uint32_t thinfilm_slice_index,
  uint32_t thinfilm_slice_count, uint32_t wavelength_group_index, TaskScheduler& scheduler) {
  const auto layout = dielectric_lut_layout(material);
  const uint32_t direction_count = kEnergyCompensationDielectricLutSize * layout.azimuth_count;
  const uint32_t side_entry_count = direction_count * layout.roughness_count;
  const uint32_t width = kEnergyCompensationDielectricBranchCount * direction_count;
  std::vector<float4> image(width * layout.roughness_count, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> probability(width * layout.roughness_count, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationDielectricAverageWidth * layout.roughness_count, float4{0.0f, 0.0f, 0.0f, 0.0f});

  for (uint32_t channel = 0u; channel < kEnergyCompensationSpectralWavelengthGroupSize; ++channel) {
    const uint32_t wavelength_index = wavelength_group_index * kEnergyCompensationSpectralWavelengthGroupSize + channel;
    const SpectralQuery spect = spectral_cache_query(wavelength_index);
    const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
    const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
    const ThinfilmEval thinfilm = sample_interface_slice(data, material, spect, thinfilm_slice_index, thinfilm_slice_count);
    std::vector<DielectricDirectionalAlbedoResult> directional(side_entry_count * 2u);
    std::vector<DielectricDirectionalAlbedoResult> multiscatter_directional(side_entry_count * 2u);
    scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
      (void)thread_id;
      for (uint32_t index = begin; index < end; ++index) {
        const uint32_t side = index / side_entry_count;
        const uint32_t side_index = index - side * side_entry_count;
        const uint32_t alpha_index = side_index / direction_count;
        const uint32_t mu_index = side_index - alpha_index * direction_count;
        const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
        const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
        directional[index] = integrate_dielectric_directional(spect, source_ior, target_ior, thinfilm, side == 0u,
          bsdf_energy_compensated_dielectric_lut_direction(
            bsdf_energy_compensated_dielectric_mu_parameter(mu_index % kEnergyCompensationDielectricLutSize, kEnergyCompensationDielectricLutSize),
            mu_index / kEnergyCompensationDielectricLutSize, layout.azimuth_count),
          dielectric_roughness_parameter(layout, alpha_index));
      }
    });
    scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
      (void)thread_id;
      for (uint32_t index = begin; index < end; ++index) {
        const uint32_t side = index / side_entry_count;
        const uint32_t side_index = index - side * side_entry_count;
        const uint32_t alpha_index = side_index / direction_count;
        const uint32_t mu_index = side_index - alpha_index * direction_count;
        const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
        const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
        multiscatter_directional[index] = integrate_dielectric_multiscatter_directional(spect, source_ior, target_ior, thinfilm, side == 0u,
          bsdf_energy_compensated_dielectric_lut_direction(
            bsdf_energy_compensated_dielectric_mu_parameter(mu_index % kEnergyCompensationDielectricLutSize, kEnergyCompensationDielectricLutSize),
            mu_index / kEnergyCompensationDielectricLutSize, layout.azimuth_count),
          dielectric_roughness_parameter(layout, alpha_index));
      }
    });

    if (dielectric_multiscatter_results_valid(multiscatter_directional, data.diagnostic) == false) {
      return false;
    }
    for (uint32_t alpha_index = 0u; alpha_index < layout.roughness_count; ++alpha_index) {
      float3 single_average[kEnergyCompensationDielectricBranchCount] = {};
      float3 multiscatter_average[kEnergyCompensationDielectricBranchCount] = {};
      float3 residual_average[kEnergyCompensationDielectricBranchCount] = {};

      for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
        for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
          const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
          single_average[branch] = integrate_dielectric_branch_average(directional, alpha_index, incident_side, outgoing_side, layout);
          multiscatter_average[branch] = integrate_dielectric_branch_average(multiscatter_directional, alpha_index, incident_side, outgoing_side, layout);
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
          residual_average[branch] = multiscatter_average[branch];
        }
      }

      complete_dielectric_residual_rows(residual_average, side_residual, bsdf_dielectric_directional_eta(ext_ior, int_ior), bsdf_fresnel_film_absorbs(thinfilm));
      for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
        const float3& residual = residual_average[branch];
        if ((std::isfinite(residual.x) == false) || (std::isfinite(residual.y) == false) || (std::isfinite(residual.z) == false)) {
          data.diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Dielectric interface reconstruction produced a numerical failure");
          return false;
        }
      }

      for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
        set_float4_channel(average[alpha_index * kEnergyCompensationDielectricAverageWidth + branch], channel, saturate(single_average[branch].x));
        set_float4_channel(average[alpha_index * kEnergyCompensationDielectricAverageWidth + 4u + branch], channel, max(0.0f, residual_average[branch].x));
      }

      for (uint32_t side = 0u; side < 2u; ++side) {
        const uint32_t side_offset = side * side_entry_count;
        for (uint32_t mu_index = 0u; mu_index < direction_count; ++mu_index) {
          const uint32_t source_index = side_offset + alpha_index * direction_count + mu_index;
          const DielectricDirectionalAlbedoResult& value = directional[source_index];
          for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
            const uint32_t branch = dielectric_branch_index(side, outgoing_side);
            const uint32_t x = (mu_index / kEnergyCompensationDielectricLutSize * kEnergyCompensationDielectricBranchCount + branch) * kEnergyCompensationDielectricLutSize +
                               mu_index % kEnergyCompensationDielectricLutSize;
            const uint32_t image_index = alpha_index * width + x;
            set_float4_channel(image[image_index], channel, saturate(value.branch_albedo[outgoing_side].x));
            set_float4_channel(probability[image_index], channel, saturate(value.branch_visible_probability[outgoing_side]));
          }
        }
      }
    }
  }

  if ((dielectric_lut_pixels_valid(image, data.diagnostic) == false) || (dielectric_lut_pixels_valid(average, data.diagnostic) == false) ||
      (dielectric_lut_pixels_valid(probability, data.diagnostic) == false)) {
    return false;
  }
  return save_exr_rgba(paths.directional, image, width / layout.azimuth_count, layout.roughness_count * layout.azimuth_count, data.diagnostic) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationDielectricAverageWidth, layout.roughness_count, data.diagnostic) &&
         save_exr_rgba(paths.probability, probability, width / layout.azimuth_count, layout.roughness_count * layout.azimuth_count, data.diagnostic);
}

bool generate_conductor_interface_gpu(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material,
  const GeneratedInterfacePaths& paths, const GeneratedInterfacePaths& geometric_paths, uint32_t cache_mode, uint32_t thinfilm_slice_index, uint32_t thinfilm_slice_count,
  uint32_t wavelength_group_index, bool save_spectral_geometric) {
  const uint64_t entry_count = uint64_t(kEnergyCompensationConductorLutSize) * uint64_t(kEnergyCompensationConductorLutSize);
  const bool terminal_source = conductor_terminal_source_required(material);
  EnergyCompensationGpuBuffers buffers = {};
  bool result = false;

  do {
    if ((gpu_create_float4_output_buffer(rhi, entry_count, buffers.directional, "ec_conductor_directional", data.diagnostic) == false) ||
        (gpu_create_float4_output_buffer(rhi, kEnergyCompensationConductorLutSize, buffers.average, "ec_conductor_average", data.diagnostic) == false) ||
        (gpu_create_float4_output_buffer(rhi, entry_count, buffers.geometric, "ec_conductor_geometric", data.diagnostic) == false) ||
        (gpu_create_float4_output_buffer(rhi, kEnergyCompensationConductorLutSize, buffers.geometric_average, "ec_conductor_geometric_average", data.diagnostic) == false) ||
        (gpu_create_float4_output_buffer(rhi, kEnergyCompensationConductorLutSize, buffers.conductor_fms, "ec_conductor_fms", data.diagnostic) == false)) {
      break;
    }

    if (terminal_source &&
        (gpu_create_float4_output_buffer(rhi, entry_count + kEnergyCompensationConductorLutSize, buffers.total, "ec_conductor_terminal_source", data.diagnostic) == false)) {
      break;
    }
    EnergyCompensationGpuParams directional_params =
      make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassConductorDirectional, thinfilm_slice_index, thinfilm_slice_count, wavelength_group_index);
    directional_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
    directional_params.output_geometric_index = get_bindless_descriptor_index(buffers.geometric);
    if (terminal_source) {
      directional_params.output_total_index = get_bindless_descriptor_index(buffers.total);
    }

    EnergyCompensationGpuParams average_params =
      make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassConductorAverage, thinfilm_slice_index, thinfilm_slice_count, wavelength_group_index);
    average_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
    average_params.output_average_index = get_bindless_descriptor_index(buffers.average);
    average_params.output_geometric_index = get_bindless_descriptor_index(buffers.geometric);
    average_params.output_geometric_average_index = get_bindless_descriptor_index(buffers.geometric_average);
    average_params.output_conductor_fms_index = get_bindless_descriptor_index(buffers.conductor_fms);
    average_params.output_total_index = directional_params.output_total_index;

    buffers.params_directional_count = (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) ? kEnergyCompensationSpectralWavelengthGroupSize : 1u;
    bool directional_params_created = true;
    for (uint32_t channel = 0u; channel < buffers.params_directional_count; ++channel) {
      directional_params.spectral_channel = channel;
      const std::string debug_name = "ec_conductor_directional_params_" + std::to_string(channel);
      directional_params_created =
        gpu_create_params_buffer(rhi, directional_params, buffers.params_directional[channel], debug_name.c_str(), data.diagnostic) && directional_params_created;
    }
    if ((directional_params_created == false) || (gpu_create_params_buffer(rhi, average_params, buffers.params_average, "ec_conductor_average_params", data.diagnostic) == false)) {
      break;
    }

    if (gpu_dispatch_energy_compensation(rhi, pipeline, buffers, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, true, data.diagnostic) == false) {
      break;
    }

    std::vector<float4> directional_pixels;
    std::vector<float4> average_pixels;
    std::vector<float4> conductor_fms_pixels;
    std::vector<float4> terminal_pixels;
    if (terminal_source && (gpu_read_float4_output_buffer(rhi, buffers.total, entry_count + kEnergyCompensationConductorLutSize, terminal_pixels, "ec_conductor_terminal_source",
                              data.diagnostic) == false)) {
      break;
    }
    if ((gpu_read_float4_output_buffer(rhi, buffers.directional, entry_count, directional_pixels, "ec_conductor_directional", data.diagnostic) == false) ||
        (gpu_read_float4_output_buffer(rhi, buffers.average, kEnergyCompensationConductorLutSize, average_pixels, "ec_conductor_average", data.diagnostic) == false) ||
        (gpu_read_float4_output_buffer(rhi, buffers.conductor_fms, kEnergyCompensationConductorLutSize, conductor_fms_pixels, "ec_conductor_fms", data.diagnostic) == false)) {
      break;
    }

    if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
      result = save_exr_rgba(paths.directional, directional_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, data.diagnostic) &&
               save_exr_rgba(paths.average, average_pixels, kEnergyCompensationConductorLutSize, 1u, data.diagnostic) &&
               save_exr_rgba(paths.conductor_fms, conductor_fms_pixels, kEnergyCompensationConductorLutSize, 1u, data.diagnostic);
      if (save_spectral_geometric) {
        std::vector<float4> geometric_pixels;
        std::vector<float4> geometric_average_pixels;
        if ((gpu_read_float4_output_buffer(rhi, buffers.geometric, entry_count, geometric_pixels, "ec_conductor_geometric", data.diagnostic) == false) ||
            (gpu_read_float4_output_buffer(rhi, buffers.geometric_average, kEnergyCompensationConductorLutSize, geometric_average_pixels, "ec_conductor_geometric_average",
               data.diagnostic) == false)) {
          break;
        }
        result = (save_exr_rgba(geometric_paths.geometric, geometric_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, data.diagnostic) &&
                   save_exr_rgba(geometric_paths.geometric_average, geometric_average_pixels, kEnergyCompensationConductorLutSize, 1u, data.diagnostic)) &&
                 result;
      }
    } else {
      std::vector<float4> geometric_pixels;
      std::vector<float4> geometric_average_pixels;
      if ((gpu_read_float4_output_buffer(rhi, buffers.geometric, entry_count, geometric_pixels, "ec_conductor_geometric", data.diagnostic) == false) ||
          (gpu_read_float4_output_buffer(rhi, buffers.geometric_average, kEnergyCompensationConductorLutSize, geometric_average_pixels, "ec_conductor_geometric_average",
             data.diagnostic) == false)) {
        break;
      }
      result = save_exr_rgba(paths.directional, directional_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, data.diagnostic) &&
               save_exr_rgba(paths.average, average_pixels, kEnergyCompensationConductorLutSize, 1u, data.diagnostic) &&
               save_exr_rgba(paths.geometric, geometric_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, data.diagnostic) &&
               save_exr_rgba(paths.geometric_average, geometric_average_pixels, kEnergyCompensationConductorLutSize, 1u, data.diagnostic) &&
               save_exr_rgba(paths.conductor_fms, conductor_fms_pixels, kEnergyCompensationConductorLutSize, 1u, data.diagnostic);
    }
    if (terminal_source) {
      result =
        save_exr_rgba(paths.terminal_source, terminal_pixels, kEnergyCompensationConductorLutSize, kBSDFEnergyCompensationConductorTerminalLutHeight, data.diagnostic) && result;
    }
  } while (false);

  gpu_destroy_energy_compensation_buffers(rhi, buffers);
  return result;
}

bool generate_dielectric_interface_gpu(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material,
  const GeneratedInterfacePaths& paths, uint32_t cache_mode, uint32_t thinfilm_slice_index, uint32_t thinfilm_slice_count, uint32_t wavelength_group_index) {
  const auto layout = dielectric_lut_layout(material);
  const uint32_t width = kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize * layout.azimuth_count;
  const uint64_t directional_count = uint64_t(width) * uint64_t(layout.roughness_count);
  const uint64_t average_count = uint64_t(kEnergyCompensationDielectricAverageWidth) * uint64_t(layout.roughness_count);
  EnergyCompensationGpuBuffers buffers = {};
  bool result = false;

  do {
    if ((gpu_create_float4_output_buffer(rhi, directional_count, buffers.directional, "ec_dielectric_directional", data.diagnostic) == false) ||
        (gpu_create_float4_output_buffer(rhi, directional_count, buffers.total, "ec_dielectric_total", data.diagnostic) == false) ||
        (gpu_create_float4_output_buffer(rhi, directional_count, buffers.probability, "ec_dielectric_probability", data.diagnostic) == false) ||
        (gpu_create_float4_output_buffer(rhi, average_count, buffers.average, "ec_dielectric_average", data.diagnostic) == false)) {
      break;
    }

    EnergyCompensationGpuParams directional_params =
      make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassDielectricDirectional, thinfilm_slice_index, thinfilm_slice_count, wavelength_group_index);
    directional_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
    directional_params.output_total_index = get_bindless_descriptor_index(buffers.total);
    directional_params.output_probability_index = get_bindless_descriptor_index(buffers.probability);

    EnergyCompensationGpuParams average_params =
      make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassDielectricAverage, thinfilm_slice_index, thinfilm_slice_count, wavelength_group_index);
    average_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
    average_params.output_total_index = get_bindless_descriptor_index(buffers.total);
    average_params.output_average_index = get_bindless_descriptor_index(buffers.average);

    buffers.params_directional_count = (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) ? kEnergyCompensationSpectralWavelengthGroupSize : 1u;
    bool directional_params_created = true;
    for (uint32_t channel = 0u; channel < buffers.params_directional_count; ++channel) {
      directional_params.spectral_channel = channel;
      const std::string debug_name = "ec_dielectric_directional_params_" + std::to_string(channel);
      directional_params_created =
        gpu_create_params_buffer(rhi, directional_params, buffers.params_directional[channel], debug_name.c_str(), data.diagnostic) && directional_params_created;
    }
    if ((directional_params_created == false) ||
        (gpu_create_params_buffer(rhi, average_params, buffers.params_average, "ec_dielectric_average_params", data.diagnostic) == false)) {
      break;
    }

    if (gpu_dispatch_energy_compensation(rhi, pipeline, buffers, kEnergyCompensationDielectricLutSize * layout.azimuth_count, 2u * layout.roughness_count, true, data.diagnostic) ==
        false) {
      break;
    }

    std::vector<float4> directional_pixels;
    std::vector<float4> average_pixels;
    std::vector<float4> probability_pixels;
    std::vector<float4> raw_total_pixels;
    if ((gpu_read_float4_output_buffer(rhi, buffers.directional, directional_count, directional_pixels, "ec_dielectric_directional", data.diagnostic) == false) ||
        (gpu_read_float4_output_buffer(rhi, buffers.average, average_count, average_pixels, "ec_dielectric_average", data.diagnostic) == false)) {
      break;
    }

    if ((gpu_read_float4_output_buffer(rhi, buffers.total, directional_count, raw_total_pixels, "ec_dielectric_raw_total", data.diagnostic) == false) ||
        (dielectric_lut_pixels_valid(raw_total_pixels, data.diagnostic) == false) || (dielectric_lut_pixels_valid(directional_pixels, data.diagnostic) == false) ||
        (dielectric_lut_pixels_valid(average_pixels, data.diagnostic) == false)) {
      break;
    }
    if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
      if ((gpu_read_float4_output_buffer(rhi, buffers.probability, directional_count, probability_pixels, "ec_dielectric_probability", data.diagnostic) == false) ||
          (dielectric_lut_pixels_valid(probability_pixels, data.diagnostic) == false)) {
        break;
      }
    }
    result = save_exr_rgba(paths.directional, directional_pixels, width / layout.azimuth_count, layout.roughness_count * layout.azimuth_count, data.diagnostic) &&
             save_exr_rgba(paths.average, average_pixels, kEnergyCompensationDielectricAverageWidth, layout.roughness_count, data.diagnostic);
    if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
      result = save_exr_rgba(paths.probability, probability_pixels, width / layout.azimuth_count, layout.roughness_count * layout.azimuth_count, data.diagnostic) && result;
    }
  } while (false);

  gpu_destroy_energy_compensation_buffers(rhi, buffers);
  return result;
}

bool gpu_read_host_float4_output_buffer(RHIContext& rhi, RHIBuffer buffer, uint64_t element_count, std::vector<float4>& out_pixels, const char* debug_name,
  SceneDiagnostic& diagnostic) {
  if (buffer.valid() == false) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Invalid asynchronous BSDF energy-compensation GPU output buffer %s", debug_name);
    return false;
  }

  out_pixels.resize(static_cast<size_t>(element_count));
  const RHIResult read_result = rhi.device().read_buffer(buffer, out_pixels.data(), element_count * sizeof(float4));
  if (read_result == RHIResult::Success) {
    return true;
  }

  diagnostic.set_result(read_result);
  diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to read asynchronous BSDF energy-compensation GPU buffer %s (%s)", debug_name,
    rhi_result_name(read_result));
  out_pixels.clear();
  return false;
}

void cleanup_energy_compensation_async_step(RHIContext& rhi, EnergyCompensationAsyncStep* step) {
  if (step == nullptr) {
    return;
  }

  if (step->command_buffer.valid() && rhi.valid()) {
    if (step->gpu_complete == false) {
      rhi.wait_for_command_buffer(step->command_buffer);
    }
    rhi.destroy_command_buffer(step->command_buffer);
  }
  if ((step->scheduler != nullptr) && (step->save_task.data != Task::InvalidHandle)) {
    step->scheduler->wait_and_release(step->save_task);
  }
  if (rhi.valid()) {
    gpu_destroy_energy_compensation_buffers(rhi, step->buffers);
  }
  delete step;
}

bool begin_energy_compensation_async_command(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, EnergyCompensationAsyncStep& step, uint32_t width, uint32_t height,
  SceneDiagnostic& diagnostic) {
  if (gpu_submit_energy_compensation(rhi, pipeline, step.buffers, width, height, true, true, step.command_buffer, diagnostic) == false) {
    return false;
  }

  const RHIResult query_result = rhi.query_command_buffer(step.command_buffer);
  if (query_result == RHIResult::Success) {
    step.gpu_complete = true;
    return true;
  }
  if (query_result == RHIResult::NotReady) {
    return true;
  }

  diagnostic.set_result(query_result);
  diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to query asynchronous BSDF energy-compensation GPU generation (%s)",
    rhi_result_name(query_result));
  return false;
}

EnergyCompensationAsyncStep* begin_conductor_interface_gpu_async(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, TaskScheduler& scheduler, const SceneData& data,
  const Material& material, const GeneratedInterfacePaths& paths, const GeneratedInterfacePaths& geometric_paths, uint32_t cache_mode, uint32_t thinfilm_slice_index,
  uint32_t thinfilm_slice_count, uint32_t wavelength_group_index, bool save_spectral_geometric) {
  const uint64_t entry_count = uint64_t(kEnergyCompensationConductorLutSize) * uint64_t(kEnergyCompensationConductorLutSize);
  auto step = std::make_unique<EnergyCompensationAsyncStep>();
  step->paths = paths;
  step->geometric_paths = geometric_paths;
  step->scheduler = &scheduler;
  step->cache_mode = cache_mode;
  step->kind = EnergyCompensationAsyncStepKind::Conductor;
  step->save_spectral_geometric = save_spectral_geometric;

  EnergyCompensationGpuBuffers& buffers = step->buffers;
  if ((gpu_create_float4_async_output_buffer(rhi, entry_count, buffers.directional, "ec_async_conductor_directional", data.diagnostic) == false) ||
      (gpu_create_float4_async_output_buffer(rhi, kEnergyCompensationConductorLutSize, buffers.average, "ec_async_conductor_average", data.diagnostic) == false) ||
      (gpu_create_float4_async_output_buffer(rhi, entry_count, buffers.geometric, "ec_async_conductor_geometric", data.diagnostic) == false) ||
      (gpu_create_float4_async_output_buffer(rhi, kEnergyCompensationConductorLutSize, buffers.geometric_average, "ec_async_conductor_geometric_average", data.diagnostic) ==
        false) ||
      (gpu_create_float4_async_output_buffer(rhi, kEnergyCompensationConductorLutSize, buffers.conductor_fms, "ec_async_conductor_fms", data.diagnostic) == false)) {
    cleanup_energy_compensation_async_step(rhi, step.release());
    return nullptr;
  }

  const bool terminal_source = conductor_terminal_source_required(material);
  if (terminal_source && (gpu_create_float4_async_output_buffer(rhi, entry_count + kEnergyCompensationConductorLutSize, buffers.total, "ec_async_conductor_terminal_source",
                            data.diagnostic) == false)) {
    cleanup_energy_compensation_async_step(rhi, step.release());
    return nullptr;
  }
  EnergyCompensationGpuParams directional_params =
    make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassConductorDirectional, thinfilm_slice_index, thinfilm_slice_count, wavelength_group_index);
  directional_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
  directional_params.output_geometric_index = get_bindless_descriptor_index(buffers.geometric);
  if (terminal_source) {
    directional_params.output_total_index = get_bindless_descriptor_index(buffers.total);
  }

  EnergyCompensationGpuParams average_params =
    make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassConductorAverage, thinfilm_slice_index, thinfilm_slice_count, wavelength_group_index);
  average_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
  average_params.output_average_index = get_bindless_descriptor_index(buffers.average);
  average_params.output_geometric_index = get_bindless_descriptor_index(buffers.geometric);
  average_params.output_geometric_average_index = get_bindless_descriptor_index(buffers.geometric_average);
  average_params.output_conductor_fms_index = get_bindless_descriptor_index(buffers.conductor_fms);
  average_params.output_total_index = directional_params.output_total_index;

  buffers.params_directional_count = (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) ? kEnergyCompensationSpectralWavelengthGroupSize : 1u;
  bool params_created = true;
  for (uint32_t channel = 0u; channel < buffers.params_directional_count; ++channel) {
    directional_params.spectral_channel = channel;
    const std::string debug_name = "ec_async_conductor_directional_params_" + std::to_string(channel);
    params_created = gpu_create_params_buffer(rhi, directional_params, buffers.params_directional[channel], debug_name.c_str(), data.diagnostic) && params_created;
  }
  params_created = gpu_create_params_buffer(rhi, average_params, buffers.params_average, "ec_async_conductor_average_params", data.diagnostic) && params_created;
  if ((params_created == false) ||
      (begin_energy_compensation_async_command(rhi, pipeline, *step, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, data.diagnostic) == false)) {
    cleanup_energy_compensation_async_step(rhi, step.release());
    return nullptr;
  }
  return step.release();
}

EnergyCompensationAsyncStep* begin_dielectric_interface_gpu_async(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, TaskScheduler& scheduler, const SceneData& data,
  const Material& material, const GeneratedInterfacePaths& paths, uint32_t cache_mode, uint32_t thinfilm_slice_index, uint32_t thinfilm_slice_count,
  uint32_t wavelength_group_index) {
  const auto layout = dielectric_lut_layout(material);
  const uint32_t width = kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize * layout.azimuth_count;
  const uint64_t directional_count = uint64_t(width) * uint64_t(layout.roughness_count);
  const uint64_t average_count = uint64_t(kEnergyCompensationDielectricAverageWidth) * uint64_t(layout.roughness_count);
  auto step = std::make_unique<EnergyCompensationAsyncStep>();
  step->paths = paths;
  step->scheduler = &scheduler;
  step->cache_mode = cache_mode;
  step->kind = EnergyCompensationAsyncStepKind::Dielectric;
  step->dielectric_lut = layout;

  EnergyCompensationGpuBuffers& buffers = step->buffers;
  if ((gpu_create_float4_async_output_buffer(rhi, directional_count, buffers.directional, "ec_async_dielectric_directional", data.diagnostic) == false) ||
      (gpu_create_float4_async_output_buffer(rhi, directional_count, buffers.total, "ec_async_dielectric_total", data.diagnostic) == false) ||
      (gpu_create_float4_async_output_buffer(rhi, directional_count, buffers.probability, "ec_async_dielectric_probability", data.diagnostic) == false) ||
      (gpu_create_float4_async_output_buffer(rhi, average_count, buffers.average, "ec_async_dielectric_average", data.diagnostic) == false)) {
    cleanup_energy_compensation_async_step(rhi, step.release());
    return nullptr;
  }

  EnergyCompensationGpuParams directional_params =
    make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassDielectricDirectional, thinfilm_slice_index, thinfilm_slice_count, wavelength_group_index);
  directional_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
  directional_params.output_total_index = get_bindless_descriptor_index(buffers.total);
  directional_params.output_probability_index = get_bindless_descriptor_index(buffers.probability);

  EnergyCompensationGpuParams average_params =
    make_gpu_params(data, material, cache_mode, kEnergyCompensationGpuPassDielectricAverage, thinfilm_slice_index, thinfilm_slice_count, wavelength_group_index);
  average_params.output_directional_index = get_bindless_descriptor_index(buffers.directional);
  average_params.output_total_index = get_bindless_descriptor_index(buffers.total);
  average_params.output_average_index = get_bindless_descriptor_index(buffers.average);

  buffers.params_directional_count = (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) ? kEnergyCompensationSpectralWavelengthGroupSize : 1u;
  bool params_created = true;
  for (uint32_t channel = 0u; channel < buffers.params_directional_count; ++channel) {
    directional_params.spectral_channel = channel;
    const std::string debug_name = "ec_async_dielectric_directional_params_" + std::to_string(channel);
    params_created = gpu_create_params_buffer(rhi, directional_params, buffers.params_directional[channel], debug_name.c_str(), data.diagnostic) && params_created;
  }
  params_created = gpu_create_params_buffer(rhi, average_params, buffers.params_average, "ec_async_dielectric_average_params", data.diagnostic) && params_created;
  if ((params_created == false) || (begin_energy_compensation_async_command(rhi, pipeline, *step, kEnergyCompensationDielectricLutSize * layout.azimuth_count,
                                      2u * layout.roughness_count, data.diagnostic) == false)) {
    cleanup_energy_compensation_async_step(rhi, step.release());
    return nullptr;
  }
  return step.release();
}

CacheGenerationStepResult poll_energy_compensation_async_step(RHIContext& rhi, EnergyCompensationAsyncStep*& step, SceneDiagnostic& diagnostic) {
  if (step == nullptr) {
    return CacheGenerationStepResult::Failed;
  }

  if (step->save_task.data != Task::InvalidHandle) {
    if (step->scheduler->completed(step->save_task) == false) {
      return CacheGenerationStepResult::Pending;
    }

    step->scheduler->release(step->save_task);
    const bool save_result = step->save_result;
    cleanup_energy_compensation_async_step(rhi, step);
    step = nullptr;
    return save_result ? CacheGenerationStepResult::Generated : CacheGenerationStepResult::Failed;
  }

  if (step->gpu_complete == false) {
    const RHIResult query_result = rhi.query_command_buffer(step->command_buffer);
    if (query_result == RHIResult::NotReady) {
      return CacheGenerationStepResult::Pending;
    }
    if (query_result != RHIResult::Success) {
      diagnostic.set_result(query_result);
      diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Asynchronous BSDF energy-compensation GPU generation failed (%s)",
        rhi_result_name(query_result));
      step->gpu_complete = true;
      cleanup_energy_compensation_async_step(rhi, step);
      step = nullptr;
      return CacheGenerationStepResult::Failed;
    }
    step->gpu_complete = true;
  }

  rhi.destroy_command_buffer(step->command_buffer);
  step->command_buffer = {};

  bool read_result = false;
  if (step->kind == EnergyCompensationAsyncStepKind::Conductor) {
    const uint64_t entry_count = uint64_t(kEnergyCompensationConductorLutSize) * uint64_t(kEnergyCompensationConductorLutSize);
    read_result =
      gpu_read_host_float4_output_buffer(rhi, step->buffers.directional, entry_count, step->directional_pixels, "ec_async_conductor_directional", diagnostic) &&
      gpu_read_host_float4_output_buffer(rhi, step->buffers.average, kEnergyCompensationConductorLutSize, step->average_pixels, "ec_async_conductor_average", diagnostic) &&
      gpu_read_host_float4_output_buffer(rhi, step->buffers.conductor_fms, kEnergyCompensationConductorLutSize, step->conductor_fms_pixels, "ec_async_conductor_fms", diagnostic);
    if (step->buffers.total.valid()) {
      read_result = gpu_read_host_float4_output_buffer(rhi, step->buffers.total, entry_count + kEnergyCompensationConductorLutSize, step->terminal_source_pixels,
                      "ec_async_conductor_terminal_source", diagnostic) &&
                    read_result;
    }
    if ((step->cache_mode != kBSDFEnergyCompensationCacheModeSpectralScalar) || step->save_spectral_geometric) {
      read_result = gpu_read_host_float4_output_buffer(rhi, step->buffers.geometric, entry_count, step->geometric_pixels, "ec_async_conductor_geometric", diagnostic) &&
                    gpu_read_host_float4_output_buffer(rhi, step->buffers.geometric_average, kEnergyCompensationConductorLutSize, step->geometric_average_pixels,
                      "ec_async_conductor_geometric_average", diagnostic) &&
                    read_result;
    }
  } else {
    const auto& layout = step->dielectric_lut;
    const uint32_t width = kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize * layout.azimuth_count;
    const uint64_t directional_count = uint64_t(width) * uint64_t(layout.roughness_count);
    const uint64_t average_count = uint64_t(kEnergyCompensationDielectricAverageWidth) * uint64_t(layout.roughness_count);
    std::vector<float4> raw_total_pixels;
    read_result = gpu_read_host_float4_output_buffer(rhi, step->buffers.total, directional_count, raw_total_pixels, "ec_async_dielectric_raw_total", diagnostic) &&
                  dielectric_lut_pixels_valid(raw_total_pixels, diagnostic) &&
                  gpu_read_host_float4_output_buffer(rhi, step->buffers.directional, directional_count, step->directional_pixels, "ec_async_dielectric_directional", diagnostic) &&
                  dielectric_lut_pixels_valid(step->directional_pixels, diagnostic) &&
                  gpu_read_host_float4_output_buffer(rhi, step->buffers.average, average_count, step->average_pixels, "ec_async_dielectric_average", diagnostic) &&
                  dielectric_lut_pixels_valid(step->average_pixels, diagnostic);
    if (step->cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
      read_result =
        gpu_read_host_float4_output_buffer(rhi, step->buffers.probability, directional_count, step->probability_pixels, "ec_async_dielectric_probability", diagnostic) &&
        dielectric_lut_pixels_valid(step->probability_pixels, diagnostic) && read_result;
    }
  }

  gpu_destroy_energy_compensation_buffers(rhi, step->buffers);
  if (read_result == false) {
    cleanup_energy_compensation_async_step(rhi, step);
    step = nullptr;
    return CacheGenerationStepResult::Failed;
  }

  EnergyCompensationAsyncStep* save_step = step;
  step->save_task = step->scheduler->schedule(1u, [save_step, &diagnostic](uint32_t, uint32_t, uint32_t) {
    if (save_step->kind == EnergyCompensationAsyncStepKind::Conductor) {
      if ((save_step->terminal_source_pixels.empty() == false) &&
          (save_exr_rgba(save_step->paths.terminal_source, save_step->terminal_source_pixels, kEnergyCompensationConductorLutSize,
             kBSDFEnergyCompensationConductorTerminalLutHeight, diagnostic) == false)) {
        save_step->save_result = false;
        return;
      }
      if (save_step->cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
        save_step->save_result =
          save_exr_rgba(save_step->paths.directional, save_step->directional_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, diagnostic) &&
          save_exr_rgba(save_step->paths.average, save_step->average_pixels, kEnergyCompensationConductorLutSize, 1u, diagnostic) &&
          save_exr_rgba(save_step->paths.conductor_fms, save_step->conductor_fms_pixels, kEnergyCompensationConductorLutSize, 1u, diagnostic);
        if (save_step->save_spectral_geometric) {
          save_step->save_result =
            (save_exr_rgba(save_step->geometric_paths.geometric, save_step->geometric_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize,
               diagnostic) &&
              save_exr_rgba(save_step->geometric_paths.geometric_average, save_step->geometric_average_pixels, kEnergyCompensationConductorLutSize, 1u, diagnostic)) &&
            save_step->save_result;
        }
        return;
      }

      save_step->save_result =
        save_exr_rgba(save_step->paths.directional, save_step->directional_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, diagnostic) &&
        save_exr_rgba(save_step->paths.average, save_step->average_pixels, kEnergyCompensationConductorLutSize, 1u, diagnostic) &&
        save_exr_rgba(save_step->paths.geometric, save_step->geometric_pixels, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, diagnostic) &&
        save_exr_rgba(save_step->paths.geometric_average, save_step->geometric_average_pixels, kEnergyCompensationConductorLutSize, 1u, diagnostic) &&
        save_exr_rgba(save_step->paths.conductor_fms, save_step->conductor_fms_pixels, kEnergyCompensationConductorLutSize, 1u, diagnostic);
      return;
    }

    const auto& layout = save_step->dielectric_lut;
    const uint32_t width = kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize * layout.azimuth_count;
    save_step->save_result =
      save_exr_rgba(save_step->paths.directional, save_step->directional_pixels, width / layout.azimuth_count, layout.roughness_count * layout.azimuth_count, diagnostic) &&
      save_exr_rgba(save_step->paths.average, save_step->average_pixels, kEnergyCompensationDielectricAverageWidth, layout.roughness_count, diagnostic);
    if (save_step->cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
      save_step->save_result =
        save_exr_rgba(save_step->paths.probability, save_step->probability_pixels, width / layout.azimuth_count, layout.roughness_count * layout.azimuth_count, diagnostic) &&
        save_step->save_result;
    }
  });
  return CacheGenerationStepResult::Pending;
}

bool ensure_cache_file(const SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode, const GeneratedInterfacePaths& paths,
  TaskScheduler& scheduler) {
  if (interface_lut_dimensions_valid(material, material_class, cache_mode, data.diagnostic) == false) {
    return false;
  }
  const bool conductor = material_class == MaterialClass::Conductor;
  const SpectralQuery spect = {};
  const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
  const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
  const uint32_t slice_count = interface_lut_slice_count(material);
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    bool spectral_result = true;
    for (uint32_t thinfilm_slice_index = 0u; thinfilm_slice_index < slice_count; ++thinfilm_slice_index) {
      if (conductor) {
        const GeneratedInterfacePaths geometric_paths = interface_slice_paths(paths, thinfilm_slice_index);
        const bool geometric_exists = std::filesystem::exists(geometric_paths.geometric);
        const bool geometric_average_exists = std::filesystem::exists(geometric_paths.geometric_average);
        if ((geometric_exists && geometric_average_exists) == false) {
          const ThinfilmEval thinfilm = sample_interface_slice(data, material, spect, thinfilm_slice_index, slice_count);
          const bool generated_geometric = generate_conductor_geometric_interface(geometric_paths, ext_ior, int_ior, thinfilm, scheduler, data.diagnostic);
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
        const bool terminal_exists = ((conductor && conductor_terminal_source_required(material)) == false) || std::filesystem::exists(slice_paths_value.terminal_source);
        if ((((directional_exists && average_exists) && conductor_fms_exists) && probability_exists) && terminal_exists) {
          continue;
        }

        const auto time_begin = std::chrono::steady_clock::now();
        const bool generated = conductor
                                 ? generate_conductor_interface_spectral(data, material, slice_paths_value, thinfilm_slice_index, slice_count, wavelength_group, scheduler)
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
    const bool terminal_exists = ((conductor && conductor_terminal_source_required(material)) == false) || std::filesystem::exists(slice_paths_value.terminal_source);
    if (((((directional_exists && average_exists) && geometric_exists) && geometric_average_exists) && conductor_fms_exists) && terminal_exists) {
      continue;
    }

    const ThinfilmEval thinfilm = sample_interface_slice(data, material, spect, slice_index, slice_count);
    const auto time_begin = std::chrono::steady_clock::now();
    const bool generated = conductor
                             ? generate_conductor_interface(slice_paths_value, ext_ior, int_ior, thinfilm, conductor_terminal_source_required(material), scheduler, data.diagnostic)
                             : generate_dielectric_interface(slice_paths_value, ext_ior, int_ior, thinfilm, dielectric_lut_layout(material), scheduler, data.diagnostic);
    const auto time_end = std::chrono::steady_clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
    if (generated) {
      log::info("Generated energy-compensation LUT cache %s in %.3f seconds", slice_paths_value.directional.generic_string().c_str(), elapsed_seconds);
    }
    result = generated && result;
  }
  return result;
}

bool spectral_cache_outputs_ready(const GeneratedInterfacePaths& paths, bool conductor, bool terminal_source) {
  const bool directional_exists = std::filesystem::exists(paths.directional);
  const bool average_exists = std::filesystem::exists(paths.average);
  const bool conductor_fms_exists = conductor ? std::filesystem::exists(paths.conductor_fms) : true;
  const bool probability_exists = conductor ? true : std::filesystem::exists(paths.probability);
  const bool terminal_exists = (terminal_source == false) || std::filesystem::exists(paths.terminal_source);
  return (((directional_exists && average_exists) && conductor_fms_exists) && probability_exists) && terminal_exists;
}

bool spectral_cache_slice_ready(const GeneratedInterfacePaths& paths, const GeneratedInterfacePaths& geometric_paths, bool conductor, bool terminal_source) {
  const bool geometric_exists = conductor ? std::filesystem::exists(geometric_paths.geometric) : true;
  const bool geometric_average_exists = conductor ? std::filesystem::exists(geometric_paths.geometric_average) : true;
  return (spectral_cache_outputs_ready(paths, conductor, terminal_source) && geometric_exists) && geometric_average_exists;
}

bool integrated_cache_slice_ready(const GeneratedInterfacePaths& paths, bool conductor, bool terminal_source) {
  const bool directional_exists = std::filesystem::exists(paths.directional);
  const bool average_exists = std::filesystem::exists(paths.average);
  const bool geometric_exists = conductor ? std::filesystem::exists(paths.geometric) : true;
  const bool geometric_average_exists = conductor ? std::filesystem::exists(paths.geometric_average) : true;
  const bool conductor_fms_exists = conductor ? std::filesystem::exists(paths.conductor_fms) : true;
  const bool terminal_exists = (terminal_source == false) || std::filesystem::exists(paths.terminal_source);
  return ((((directional_exists && average_exists) && geometric_exists) && geometric_average_exists) && conductor_fms_exists) && terminal_exists;
}

uint32_t pending_cache_file_step_count(const Material& material, uint32_t material_class, uint32_t cache_mode, const GeneratedInterfacePaths& paths) {
  const bool conductor = material_class == MaterialClass::Conductor;
  const uint32_t slice_count = interface_lut_slice_count(material);
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    uint32_t result = 0u;
    for (uint32_t thinfilm_slice_index = 0u; thinfilm_slice_index < slice_count; ++thinfilm_slice_index) {
      const GeneratedInterfacePaths geometric_paths = interface_slice_paths(paths, thinfilm_slice_index);
      bool geometric_missing = conductor && ((std::filesystem::exists(geometric_paths.geometric) && std::filesystem::exists(geometric_paths.geometric_average)) == false);
      for (uint32_t wavelength_group = 0u; wavelength_group < kEnergyCompensationSpectralWavelengthGroupCount; ++wavelength_group) {
        const uint32_t packed_slice_index = thinfilm_slice_index * kEnergyCompensationSpectralWavelengthGroupCount + wavelength_group;
        const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, packed_slice_index);
        if ((spectral_cache_outputs_ready(slice_paths_value, conductor, conductor && conductor_terminal_source_required(material)) == false) || geometric_missing) {
          result += 1u;
          geometric_missing = false;
        }
      }
    }
    return result;
  }

  uint32_t result = 0u;
  for (uint32_t slice_index = 0u; slice_index < slice_count; ++slice_index) {
    if (integrated_cache_slice_ready(interface_slice_paths(paths, slice_index), conductor, conductor && conductor_terminal_source_required(material)) == false) {
      result += 1u;
    }
  }
  return result;
}

CacheGenerationStepResult ensure_cache_file_gpu_step(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material,
  uint32_t material_class, uint32_t cache_mode, const GeneratedInterfacePaths& paths) {
  if (interface_lut_dimensions_valid(material, material_class, cache_mode, data.diagnostic) == false) {
    return CacheGenerationStepResult::Failed;
  }
  const bool conductor = material_class == MaterialClass::Conductor;
  const uint32_t slice_count = interface_lut_slice_count(material);
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    for (uint32_t thinfilm_slice_index = 0u; thinfilm_slice_index < slice_count; ++thinfilm_slice_index) {
      for (uint32_t wavelength_group = 0u; wavelength_group < kEnergyCompensationSpectralWavelengthGroupCount; ++wavelength_group) {
        const uint32_t packed_slice_index = thinfilm_slice_index * kEnergyCompensationSpectralWavelengthGroupCount + wavelength_group;
        const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, packed_slice_index);
        const GeneratedInterfacePaths geometric_paths = interface_slice_paths(paths, thinfilm_slice_index);
        if (spectral_cache_slice_ready(slice_paths_value, geometric_paths, conductor, conductor && conductor_terminal_source_required(material))) {
          continue;
        }

        const auto time_begin = std::chrono::steady_clock::now();
        const bool save_spectral_geometric =
          conductor && ((std::filesystem::exists(geometric_paths.geometric) && std::filesystem::exists(geometric_paths.geometric_average)) == false);
        const bool generated =
          conductor ? generate_conductor_interface_gpu(rhi, pipeline, data, material, slice_paths_value, geometric_paths, cache_mode, thinfilm_slice_index, slice_count,
                        wavelength_group, save_spectral_geometric)
                    : generate_dielectric_interface_gpu(rhi, pipeline, data, material, slice_paths_value, cache_mode, thinfilm_slice_index, slice_count, wavelength_group);
        const auto time_end = std::chrono::steady_clock::now();
        const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
        if (generated) {
          log::info("Generated GPU spectral energy-compensation LUT cache %s in %.3f seconds", slice_paths_value.directional.generic_string().c_str(), elapsed_seconds);
        }
        return generated ? CacheGenerationStepResult::Generated : CacheGenerationStepResult::Failed;
      }
    }
    return CacheGenerationStepResult::Complete;
  }

  for (uint32_t slice_index = 0u; slice_index < slice_count; ++slice_index) {
    const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, slice_index);
    if (integrated_cache_slice_ready(slice_paths_value, conductor, conductor && conductor_terminal_source_required(material))) {
      continue;
    }

    const auto time_begin = std::chrono::steady_clock::now();
    const bool generated =
      conductor ? generate_conductor_interface_gpu(rhi, pipeline, data, material, slice_paths_value, slice_paths_value, cache_mode, slice_index, slice_count, 0u, false)
                : generate_dielectric_interface_gpu(rhi, pipeline, data, material, slice_paths_value, cache_mode, slice_index, slice_count, 0u);
    const auto time_end = std::chrono::steady_clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
    if (generated) {
      log::info("Generated GPU energy-compensation LUT cache %s in %.3f seconds", slice_paths_value.directional.generic_string().c_str(), elapsed_seconds);
    }
    return generated ? CacheGenerationStepResult::Generated : CacheGenerationStepResult::Failed;
  }
  return CacheGenerationStepResult::Complete;
}

CacheGenerationStepResult ensure_cache_file_gpu_async_step(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, TaskScheduler& scheduler, const SceneData& data,
  const Material& material, uint32_t material_class, uint32_t cache_mode, const GeneratedInterfacePaths& paths, EnergyCompensationGenerationContext& context) {
  if (context.pending_step != nullptr) {
    auto* pending_step = static_cast<EnergyCompensationAsyncStep*>(context.pending_step);
    const CacheGenerationStepResult result = poll_energy_compensation_async_step(rhi, pending_step, data.diagnostic);
    context.pending_step = pending_step;
    return result;
  }

  if (interface_lut_dimensions_valid(material, material_class, cache_mode, data.diagnostic) == false) {
    return CacheGenerationStepResult::Failed;
  }
  const bool conductor = material_class == MaterialClass::Conductor;
  const uint32_t slice_count = interface_lut_slice_count(material);
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    for (uint32_t thinfilm_slice_index = 0u; thinfilm_slice_index < slice_count; ++thinfilm_slice_index) {
      for (uint32_t wavelength_group = 0u; wavelength_group < kEnergyCompensationSpectralWavelengthGroupCount; ++wavelength_group) {
        const uint32_t packed_slice_index = thinfilm_slice_index * kEnergyCompensationSpectralWavelengthGroupCount + wavelength_group;
        const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, packed_slice_index);
        const GeneratedInterfacePaths geometric_paths = interface_slice_paths(paths, thinfilm_slice_index);
        if (spectral_cache_slice_ready(slice_paths_value, geometric_paths, conductor, conductor && conductor_terminal_source_required(material))) {
          continue;
        }

        const bool save_spectral_geometric =
          conductor && ((std::filesystem::exists(geometric_paths.geometric) && std::filesystem::exists(geometric_paths.geometric_average)) == false);
        EnergyCompensationAsyncStep* pending_step = conductor ? begin_conductor_interface_gpu_async(rhi, pipeline, scheduler, data, material, slice_paths_value, geometric_paths,
                                                                  cache_mode, thinfilm_slice_index, slice_count, wavelength_group, save_spectral_geometric)
                                                              : begin_dielectric_interface_gpu_async(rhi, pipeline, scheduler, data, material, slice_paths_value, cache_mode,
                                                                  thinfilm_slice_index, slice_count, wavelength_group);
        if (pending_step == nullptr) {
          return CacheGenerationStepResult::Failed;
        }
        context.pending_step = pending_step;
        return CacheGenerationStepResult::Pending;
      }
    }
    return CacheGenerationStepResult::Complete;
  }

  for (uint32_t slice_index = 0u; slice_index < slice_count; ++slice_index) {
    const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, slice_index);
    if (integrated_cache_slice_ready(slice_paths_value, conductor, conductor && conductor_terminal_source_required(material))) {
      continue;
    }

    EnergyCompensationAsyncStep* pending_step =
      conductor
        ? begin_conductor_interface_gpu_async(rhi, pipeline, scheduler, data, material, slice_paths_value, slice_paths_value, cache_mode, slice_index, slice_count, 0u, false)
        : begin_dielectric_interface_gpu_async(rhi, pipeline, scheduler, data, material, slice_paths_value, cache_mode, slice_index, slice_count, 0u);
    if (pending_step == nullptr) {
      return CacheGenerationStepResult::Failed;
    }
    context.pending_step = pending_step;
    return CacheGenerationStepResult::Pending;
  }
  return CacheGenerationStepResult::Complete;
}

bool ensure_cache_file_gpu(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode,
  const GeneratedInterfacePaths& paths) {
  while (true) {
    const CacheGenerationStepResult result = ensure_cache_file_gpu_step(rhi, pipeline, data, material, material_class, cache_mode, paths);
    if (result == CacheGenerationStepResult::Complete) {
      return true;
    }
    if (result == CacheGenerationStepResult::Failed) {
      return false;
    }
  }
}

void release_energy_compensation_interface_images(SceneData& data, const Scene::EnergyCompensationInterface& interface_data) {
  for (const uint32_t index : {interface_data.directional_lut, interface_data.average_lut, interface_data.geometric_lut, interface_data.geometric_average_lut,
         interface_data.conductor_fms_lut, interface_data.probability_lut, interface_data.terminal_source_lut, interface_data.rgb_fresnel_lut}) {
    data.energy_compensation_image_hashes.erase(index);
    data.images.remove(index);
  }
}

void release_energy_compensation_interface_block(SceneData& data, const std::vector<Scene::EnergyCompensationInterface>& interfaces, uint32_t index) {
  // A root record is followed by one record per additional coating knot.
  const uint32_t count = interfaces[index].coating_slice_count;
  for (uint32_t coating_index = 0u; coating_index < count; ++coating_index) {
    release_energy_compensation_interface_images(data, interfaces[index + coating_index]);
  }
}

struct VelvetQuadratureSample {
  double coordinate = 0.0;
  double weight = 0.0;
};

std::array<VelvetQuadratureSample, 128u> velvet_quadrature() {
  std::array<VelvetQuadratureSample, 128u> result = {};
  constexpr double pi = 3.14159265358979323846;
  for (uint32_t index = 0u; index < (result.size() / 2u); ++index) {
    double root = std::cos(pi * (double(index) + 0.75) / (double(result.size()) + 0.5));
    double derivative = 0.0;
    double correction = 0.0;
    do {
      double previous = 1.0;
      double polynomial = root;
      for (uint32_t order = 2u; order <= result.size(); ++order) {
        const double next = ((2.0 * double(order) - 1.0) * root * polynomial - (double(order) - 1.0) * previous) / double(order);
        previous = polynomial;
        polynomial = next;
      }
      derivative = double(result.size()) * (root * polynomial - previous) / (root * root - 1.0);
      correction = polynomial / derivative;
      root -= correction;
    } while (std::abs(correction) > 1.0e-15);
    const double weight = 1.0 / ((1.0 - root * root) * derivative * derivative);
    result[index] = {(1.0 - root) * 0.5, weight};
    result[result.size() - 1u - index] = {(1.0 + root) * 0.5, weight};
  }
  return result;
}

double velvet_projected_area(double alpha, double mu, const std::array<VelvetQuadratureSample, 128u>& quadrature) {
  constexpr double pi = 3.14159265358979323846;
  const double sine = std::sqrt(std::max(0.0, 1.0 - mu * mu));
  const double scale = std::sqrt(alpha);
  const double end = sine / (sine + scale);
  double integral = 0.0;
  for (const VelvetQuadratureSample& sample : quadrature) {
    const double t = sample.coordinate * end;
    const double z = scale * t / (1.0 - t);
    const double jacobian = scale * end / ((1.0 - t) * (1.0 - t));
    const double a = sine * std::sqrt(std::max(0.0, 1.0 - z * z));
    const double b = mu * z;
    const double ratio = a > 0.0 ? std::min(1.0, b / a) : 0.0;
    const double density = (2.0 + 1.0 / alpha) * std::pow(std::max(0.0, 1.0 - z * z), 0.5 / alpha) / (2.0 * pi);
    integral += density * (a * std::sqrt(std::max(0.0, 1.0 - ratio * ratio)) - b * std::acos(ratio)) * jacobian * sample.weight;
  }
  return mu + 2.0 * integral;
}

bool generate_velvet_lookup(const std::filesystem::path& path, std::vector<float4>& pixels, TaskScheduler& scheduler, SceneDiagnostic& diagnostic) {
  constexpr double pi = 3.14159265358979323846;
  const std::array<VelvetQuadratureSample, 128u> quadrature = velvet_quadrature();
  std::array<double, 128u> azimuth_cosine = {};
  std::array<double, 128u> azimuth_weight = {};
  for (uint32_t index = 0u; index < quadrature.size(); ++index) {
    azimuth_cosine[index] = std::cos(pi * quadrature[index].coordinate);
    azimuth_weight[index] = pi * quadrature[index].weight;
  }
  const double alpha_span = std::log(1.0 / double(kEpsilon));
  pixels.resize(size_t(kVelvetPhysicalLutSize) * kVelvetPhysicalLutSize);
  scheduler.execute(kVelvetPhysicalLutSize, [&](uint32_t begin, uint32_t end, uint32_t) {
    for (uint32_t row = begin; row < end; ++row) {
      const double alpha = double(kEpsilon) * std::exp(alpha_span * double(row) / double(kVelvetPhysicalLutSize - 1u));
      const double root_alpha = std::sqrt(alpha);
      const double mu_span = std::log1p(1.0 / root_alpha);
      const double density_scale = (2.0 + 1.0 / alpha) / (2.0 * pi);
      const double density_power = 0.5 / alpha;
      std::array<double, 128u> outgoing_mu = {};
      std::array<double, 128u> outgoing_sine = {};
      std::array<double, 128u> outgoing_area = {};
      std::array<double, 128u> outgoing_weight = {};
      for (uint32_t index = 0u; index < quadrature.size(); ++index) {
        outgoing_mu[index] = root_alpha * std::expm1(mu_span * quadrature[index].coordinate);
        outgoing_sine[index] = std::sqrt(std::max(0.0, 1.0 - outgoing_mu[index] * outgoing_mu[index]));
        outgoing_area[index] = velvet_projected_area(alpha, outgoing_mu[index], quadrature);
        outgoing_weight[index] = quadrature[index].weight * (outgoing_mu[index] + root_alpha) * mu_span;
      }
      for (uint32_t column = 0u; column < kVelvetPhysicalLutSize; ++column) {
        const double axis = double(column) / double(kVelvetPhysicalLutSize - 1u);
        const double mu = std::min(1.0, root_alpha * std::expm1(mu_span * axis * axis));
        const double sine_i = std::sqrt(std::max(0.0, 1.0 - mu * mu));
        const double area_i = velvet_projected_area(alpha, mu, quadrature);
        double energy = 0.0;
        for (uint32_t index = 0u; (column != 0u) && (index < quadrature.size()); ++index) {
          const double mo = outgoing_mu[index];
          const double sine_o = outgoing_sine[index];
          const double denominator = 4.0 * (area_i * mo + outgoing_area[index] * mu - mu * mo);
          for (uint32_t azimuth = 0u; azimuth < quadrature.size(); ++azimuth) {
            const double dot_value = mu * mo + sine_i * sine_o * azimuth_cosine[azimuth];
            const double half_z_squared = (mu + mo) * (mu + mo) / (2.0 + 2.0 * dot_value);
            const double density = density_scale * std::pow(std::max(0.0, 1.0 - half_z_squared), density_power);
            energy += 2.0 * density * mo * outgoing_weight[index] * azimuth_weight[azimuth] / denominator;
          }
        }
        // Projected-area normalization gives unit directional albedo at the exact grazing endpoint.
        const float albedo = column == 0u ? 1.0f : float(energy);
        const float area = float(velvet_projected_area(alpha, std::sin(axis * pi * 0.5), quadrature) * root_alpha);
        pixels[row * kVelvetPhysicalLutSize + column] = {albedo, area, 0.0f, 0.0f};
      }
    }
  });
  for (const float4& pixel : pixels) {
    if ((std::isfinite(pixel.x) == false) || (pixel.x < 0.0f) || (pixel.x > 1.0f) || (std::isfinite(pixel.y) == false) || (pixel.y <= 0.0f)) {
      diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Velvet projected-area lookup generation failed");
      return false;
    }
  }
  return save_exr_rgba(path, pixels, kVelvetPhysicalLutSize, kVelvetPhysicalLutSize, diagnostic);
}

bool velvet_tint_passive(const SceneData& data, const SpectralImage& tint) {
  if (tint.spectrum_index >= data.spectrum_values.size()) {
    return false;
  }
  const SpectralDistribution& spectrum = data.spectrum_values[tint.spectrum_index];
  const float3 rgb = spectrum.integrated();
  const bool rgb_mode = data.options.properties[Scene::Properties::Spectral] == false;
  if (rgb_mode && ((valid_value(rgb) == false) || (min(min(rgb.x, rgb.y), rgb.z) < 0.0f))) {
    return false;
  }
  const Image* image = nullptr;
  uint64_t pixel_count = 0u;
  float3 maximum = {1.0f, 1.0f, 1.0f};
  if (tint.image_index != kInvalidIndex) {
    if (tint.image_index >= data.images_vector.size()) {
      return false;
    }
    image = &data.images_vector[tint.image_index];
    pixel_count = uint64_t(image->isize.x) * image->isize.y * image->isize.z;
    if ((image->format == Image::Format::Undefined) || (pixel_count == 0u) || (pixel_count > std::numeric_limits<uint32_t>::max())) {
      return false;
    }
    maximum = {};
    for (uint32_t index = 0u; index < pixel_count; ++index) {
      const float4 texel = image->pixel(index);
      const float3 value = {texel.x, texel.y, texel.z};
      if ((valid_value(value) == false) || (min(min(value.x, value.y), value.z) < 0.0f) || (rgb_mode && (max(max(rgb.x * value.x, rgb.y * value.y), rgb.z * value.z) > 1.0f))) {
        return false;
      }
      maximum = max(maximum, value);
    }
  } else if (rgb_mode && (max(max(rgb.x, rgb.y), rgb.z) > 1.0f)) {
    return false;
  }
  if (rgb_mode) {
    return true;
  }
  for (uint32_t wavelength = 0u; wavelength < WavelengthCount; ++wavelength) {
    const SpectralQuery query = {kShortestWavelength + float(wavelength), SpectralFlags::Spectral};
    const float value = spectrum(query).value;
    if ((std::isfinite(value) == false) || (value < 0.0f)) {
      return false;
    }
    const float upper = image != nullptr ? value * spectral_rgb_evaluate_wavelength(query.wavelength, maximum) : value;
    if (upper <= 1.0f) {
      continue;
    }
    if (image == nullptr) {
      return false;
    }
    // Component maxima can belong to different texels; verify the effective spectrum before rejecting.
    for (uint32_t index = 0u; index < pixel_count; ++index) {
      const float4 texel = image->pixel(index);
      const float effective = value * spectral_rgb_evaluate_wavelength(query.wavelength, {texel.x, texel.y, texel.z});
      if ((std::isfinite(effective) == false) || (effective > 1.0f)) {
        return false;
      }
    }
  }
  return true;
}

bool bind_velvet_interface(SceneData& data, const std::vector<Scene::EnergyCompensationInterface>& previous_interfaces, std::unordered_map<uint64_t, uint32_t>& interface_cache,
  TaskScheduler& scheduler, uint32_t& out_interface_index) {
  constexpr uint64_t hash = 0x76656c7665740001ull;
  const auto found = interface_cache.find(hash);
  if (found != interface_cache.end()) {
    out_interface_index = found->second;
    return true;
  }
  const auto resident = data.energy_compensation_interface_cache.find(hash);
  if (resident != data.energy_compensation_interface_cache.end()) {
    out_interface_index = data.add_energy_compensation_interface(previous_interfaces[resident->second]);
    interface_cache[hash] = out_interface_index;
    return true;
  }
  const std::filesystem::path path = cache_directory() / "velvet_projected_area_v1.exr";
  std::vector<float4> pixels;
  if (std::filesystem::exists(path)) {
    if (load_rgba32f_lut(path, kVelvetPhysicalLutSize, kVelvetPhysicalLutSize, pixels, data.diagnostic) == false) {
      return false;
    }
  } else if (generate_velvet_lookup(path, pixels, scheduler, data.diagnostic) == false) {
    return false;
  }
  Scene::EnergyCompensationInterface interface_data = {};
  interface_data.cls = MaterialClass::Velvet;
  interface_data.directional_lut =
    data.images.add_from_data_3d(pixels.data(), uint3{kVelvetPhysicalLutSize, kVelvetPhysicalLutSize, 1u}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
  if (interface_data.directional_lut == kInvalidIndex) {
    return false;
  }
  out_interface_index = data.add_energy_compensation_interface(interface_data);
  interface_cache[hash] = out_interface_index;
  return true;
}

std::array<float3, kEnergyCompensationSpectralWavelengthCount> conductor_rgb_integration_weights() {
  std::array<float3, kEnergyCompensationSpectralWavelengthCount> weights = {};
  const auto white = SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f});
  float3 normalization = {};
  for (uint32_t index = 0u; index < WavelengthCount; ++index) {
    const float wavelength = kShortestWavelength + float(index);
    const float axis = float(index) * float(kEnergyCompensationSpectralWavelengthCount - 1u) / float(WavelengthCount - 1u);
    const uint32_t first = min(uint32_t(axis), kEnergyCompensationSpectralWavelengthCount - 2u);
    const float fraction = axis - float(first);
    const float endpoint_weight = ((index == 0u) || (index == (WavelengthCount - 1u))) ? 0.5f : 1.0f;
    const float3 value = spectral_xyz_to_rgb(spectral_xyz(index)) * white.query(SpectralQuery(wavelength, SpectralFlags::Spectral)).value * endpoint_weight;
    weights[first] += value * (1.0f - fraction);
    weights[first + 1u] += value * fraction;
    normalization += value;
  }
  for (float3& weight : weights) {
    weight /= normalization;
  }
  return weights;
}

void integrate_conductor_rgb_lut(std::vector<float4>& pixels, uint32_t entry_count, uint32_t slice_count,
  const std::array<float3, kEnergyCompensationSpectralWavelengthCount>& weights, bool bounded) {
  if (pixels.empty()) {
    return;
  }
  std::vector<float4> integrated(size_t(entry_count) * slice_count);
  for (uint32_t slice = 0u; slice < slice_count; ++slice) {
    for (uint32_t entry = 0u; entry < entry_count; ++entry) {
      float3 value = {};
      for (uint32_t wavelength = 0u; wavelength < kEnergyCompensationSpectralWavelengthCount; ++wavelength) {
        const uint32_t group = slice * kEnergyCompensationSpectralWavelengthGroupCount + wavelength / kEnergyCompensationSpectralWavelengthGroupSize;
        value +=
          weights[wavelength] * bsdf_energy_compensated_float4_channel(pixels[size_t(group) * entry_count + entry], wavelength % kEnergyCompensationSpectralWavelengthGroupSize);
      }
      // RGB can only represent passive reflectances inside its color gamut.
      value = bounded ? saturate(value) : max(value, float3{});
      integrated[size_t(slice) * entry_count + entry] = {value.x, value.y, value.z, 1.0f};
    }
  }
  pixels = std::move(integrated);
}

std::vector<float4> conductor_rgb_fresnel(const SceneData& data, const Material& material, uint32_t coating_index, uint32_t slice_count,
  const std::array<float3, kEnergyCompensationSpectralWavelengthCount>& weights) {
  std::vector<float4> pixels(size_t(kBSDFConductorRGBFresnelSize) * slice_count);
  for (uint32_t slice = 0u; slice < slice_count; ++slice) {
    for (uint32_t wavelength = 0u; wavelength < kEnergyCompensationSpectralWavelengthCount; ++wavelength) {
      const SpectralQuery spect = spectral_cache_query(wavelength);
      const auto exterior = sample_refractive_index(data, material.ext_ior, spect);
      const auto interior = sample_refractive_index(data, material.int_ior, spect);
      const auto film = sample_interface_slice(data, material, spect, coating_index * slice_count + slice, interface_lut_slice_count(material));
      for (uint32_t cosine = 0u; cosine < kBSDFConductorRGBFresnelSize; ++cosine) {
        const float mu = float(cosine) / float(kBSDFConductorRGBFresnelSize - 1u);
        const float response = bsdf_fresnel_calculate(spect, mu, exterior, interior, film).value;
        const float3 value = weights[wavelength] * response;
        pixels[size_t(slice) * kBSDFConductorRGBFresnelSize + cosine] += float4{value.x, value.y, value.z, 0.0f};
      }
    }
  }
  for (float4& pixel : pixels) {
    pixel = {saturate(pixel.x), saturate(pixel.y), saturate(pixel.z), 1.0f};
  }
  return pixels;
}

void pack_anisotropic_dielectric_spectral_lut(std::vector<float4>& directional, std::vector<float4>& probability, uint32_t slice_count,
  const BSDFEnergyCompensationDielectricLut& layout) {
  // Keep the disk cache wavelength-packed; runtime texels contain both exit energies and probabilities for one incident side.
  std::vector<float4> outside(directional.size());
  std::vector<float4> inside(directional.size());
  const uint32_t width = kEnergyCompensationDielectricLutSize;
  const uint32_t rows = layout.roughness_count * layout.azimuth_count;
  for (uint32_t slice = 0u; slice < slice_count; ++slice) {
    for (uint32_t wavelength = 0u; wavelength < kEnergyCompensationSpectralWavelengthCount; ++wavelength) {
      const uint32_t group = slice * kEnergyCompensationSpectralWavelengthGroupCount + wavelength / kEnergyCompensationSpectralWavelengthGroupSize;
      const uint32_t channel = wavelength % kEnergyCompensationSpectralWavelengthGroupSize;
      const size_t source_slice = size_t(group) * rows * 4u * width;
      const size_t target_slice = size_t(slice * kEnergyCompensationSpectralWavelengthCount + wavelength) * rows * width;
      for (uint32_t row = 0u; row < rows; ++row) {
        for (uint32_t mu = 0u; mu < width; ++mu) {
          const size_t source = source_slice + size_t(row) * 4u * width + mu;
          const size_t target = target_slice + size_t(row) * width + mu;
          outside[target] = {bsdf_energy_compensated_float4_channel(directional[source], channel), bsdf_energy_compensated_float4_channel(directional[source + width], channel),
            bsdf_energy_compensated_float4_channel(probability[source], channel), bsdf_energy_compensated_float4_channel(probability[source + width], channel)};
          inside[target] = {bsdf_energy_compensated_float4_channel(directional[source + 2u * width], channel),
            bsdf_energy_compensated_float4_channel(directional[source + 3u * width], channel), bsdf_energy_compensated_float4_channel(probability[source + 2u * width], channel),
            bsdf_energy_compensated_float4_channel(probability[source + 3u * width], channel)};
        }
      }
    }
  }
  directional.swap(outside);
  probability.swap(inside);
}

bool load_energy_compensation_interface_images(SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode, uint32_t target_cache_mode,
  const GeneratedInterfacePaths& paths, std::vector<Scene::EnergyCompensationInterface>& prepared_interfaces) {
  const bool conductor = material_class == MaterialClass::Conductor;
  const auto layout = dielectric_lut_layout(material);
  const uint32_t coating_count = coherent_coating_lut_slice_count(material.coherent_coating);
  prepared_interfaces.reserve(coating_count);
  for (uint32_t coating_index = 0u; coating_index < coating_count; ++coating_index) {
    Scene::EnergyCompensationInterface interface_data = {};
    interface_data.cls = material_class;
    interface_data.cache_mode = cache_mode;
    interface_data.dielectric_azimuth_count = layout.azimuth_count;
    interface_data.dielectric_roughness_count = layout.roughness_count;
    const uint32_t slice_count = thinfilm_lut_slice_count(material.thinfilm);
    interface_data.thinfilm_slice_count = thinfilm_lut_slice_count(material.thinfilm);
    interface_data.coating_slice_count = coating_index == 0u ? coating_count : 1u;
    const bool spectral = cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar;
    if (spectral) {
      interface_data.spectral_wavelength_count = kEnergyCompensationSpectralWavelengthCount;
      interface_data.spectral_shortest_wavelength = kShortestWavelength;
      interface_data.spectral_longest_wavelength = kLongestWavelength;
    }

    std::vector<float4> directional_pixels;
    std::vector<float4> average_pixels;
    std::vector<float4> geometric_pixels;
    std::vector<float4> geometric_average_pixels;
    std::vector<float4> conductor_fms_pixels;
    std::vector<float4> probability_pixels;
    std::vector<float4> terminal_pixels;
    const uint32_t directional_width = conductor ? kEnergyCompensationConductorLutSize : (kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize);
    const uint32_t directional_height = conductor ? kEnergyCompensationConductorLutSize : (layout.roughness_count * layout.azimuth_count);
    const uint32_t average_width = conductor ? kEnergyCompensationConductorLutSize : kEnergyCompensationDielectricAverageWidth;
    const uint32_t average_height = conductor ? 1u : layout.roughness_count;
    uint32_t packed_slice_count = spectral ? (slice_count * kEnergyCompensationSpectralWavelengthGroupCount) : slice_count;
    bool loaded = load_rgba32f_lut_slices(paths, directional_width, directional_height, packed_slice_count, directional_pixels, &GeneratedInterfacePaths::directional,
                    coating_index * packed_slice_count, data.diagnostic) &&
                  load_rgba32f_lut_slices(paths, average_width, average_height, packed_slice_count, average_pixels, &GeneratedInterfacePaths::average,
                    coating_index * packed_slice_count, data.diagnostic);
    if (conductor) {
      loaded = loaded && (load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, slice_count, geometric_pixels,
                            &GeneratedInterfacePaths::geometric, coating_index * slice_count, data.diagnostic) &&
                           load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, 1u, slice_count, geometric_average_pixels,
                             &GeneratedInterfacePaths::geometric_average, coating_index * slice_count, data.diagnostic) &&
                           load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, 1u, packed_slice_count, conductor_fms_pixels,
                             &GeneratedInterfacePaths::conductor_fms, coating_index * packed_slice_count, data.diagnostic));
      if (conductor_terminal_source_required(material)) {
        loaded = loaded && load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, kBSDFEnergyCompensationConductorTerminalLutHeight, packed_slice_count,
                             terminal_pixels, &GeneratedInterfacePaths::terminal_source, coating_index * packed_slice_count, data.diagnostic);
      }
    } else if (spectral) {
      loaded = loaded && load_rgba32f_lut_slices(paths, directional_width, directional_height, packed_slice_count, probability_pixels, &GeneratedInterfacePaths::probability,
                           coating_index * packed_slice_count, data.diagnostic);
    }
    if (loaded == false) {
      for (const auto& prepared : prepared_interfaces) {
        release_energy_compensation_interface_images(data, prepared);
      }
      prepared_interfaces.clear();
      return false;
    }

    std::vector<float4> fresnel_pixels;
    if (conductor && spectral && (target_cache_mode == kBSDFEnergyCompensationCacheModeIntegratedRGB)) {
      const auto weights = conductor_rgb_integration_weights();
      integrate_conductor_rgb_lut(directional_pixels, directional_width * directional_height, slice_count, weights, true);
      integrate_conductor_rgb_lut(average_pixels, average_width * average_height, slice_count, weights, true);
      integrate_conductor_rgb_lut(conductor_fms_pixels, kEnergyCompensationConductorLutSize, slice_count, weights, true);
      integrate_conductor_rgb_lut(terminal_pixels, kEnergyCompensationConductorLutSize * kBSDFEnergyCompensationConductorTerminalLutHeight, slice_count, weights, false);
      fresnel_pixels = conductor_rgb_fresnel(data, material, coating_index, slice_count, weights);
      packed_slice_count = slice_count;
      interface_data.cache_mode = kBSDFEnergyCompensationCacheModeIntegratedRGB;
      interface_data.spectral_wavelength_count = 0u;
      interface_data.spectral_shortest_wavelength = 0.0f;
      interface_data.spectral_longest_wavelength = 0.0f;
    }

    const bool packed_dielectric = (conductor == false) && spectral && (layout.azimuth_count > 1u);
    const uint3 directional_size = packed_dielectric ? uint3{kEnergyCompensationDielectricLutSize, directional_height, slice_count * kEnergyCompensationSpectralWavelengthCount}
                                                     : uint3{directional_width, directional_height, packed_slice_count};
    if (packed_dielectric) {
      pack_anisotropic_dielectric_spectral_lut(directional_pixels, probability_pixels, slice_count, layout);
    }

    // Load one coating plane before allocating its images; earlier planes are released on failure.
    interface_data.directional_lut = data.images.add_from_data_3d(directional_pixels.data(), directional_size, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    interface_data.average_lut =
      data.images.add_from_data_3d(average_pixels.data(), uint3{average_width, average_height, packed_slice_count}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    if (conductor) {
      if (fresnel_pixels.empty() == false) {
        interface_data.rgb_fresnel_lut =
          data.images.add_from_data_3d(fresnel_pixels.data(), uint3{kBSDFConductorRGBFresnelSize, 1u, slice_count}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      }
      interface_data.geometric_lut = data.images.add_from_data_3d(geometric_pixels.data(),
        uint3{kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, slice_count}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      interface_data.geometric_average_lut = data.images.add_from_data_3d(geometric_average_pixels.data(), uint3{kEnergyCompensationConductorLutSize, 1u, slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      interface_data.conductor_fms_lut = data.images.add_from_data_3d(conductor_fms_pixels.data(), uint3{kEnergyCompensationConductorLutSize, 1u, packed_slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      if (conductor_terminal_source_required(material)) {
        interface_data.terminal_source_lut = data.images.add_from_data_3d(terminal_pixels.data(),
          uint3{kEnergyCompensationConductorLutSize, kBSDFEnergyCompensationConductorTerminalLutHeight, packed_slice_count}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      }
    } else if (spectral) {
      interface_data.probability_lut = data.images.add_from_data_3d(probability_pixels.data(), directional_size, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    }
    for (const uint32_t index : {interface_data.directional_lut, interface_data.average_lut, interface_data.geometric_lut, interface_data.geometric_average_lut,
           interface_data.conductor_fms_lut, interface_data.probability_lut, interface_data.terminal_source_lut, interface_data.rgb_fresnel_lut}) {
      if (index != kInvalidIndex) {
        const auto& image = data.images.get(index);
        data.energy_compensation_image_hashes[index] = xxh64(image.pixels.f32.a, image.data_size);
      }
    }
    prepared_interfaces.push_back(interface_data);
  }
  return true;
}

uint32_t interface_source_cache_mode(const SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode) {
  if ((material_class != MaterialClass::Conductor) || (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar)) {
    return cache_mode;
  }
  bool sampled = false;
  const auto spectral_ior = [&](const RefractiveIndex& ior) {
    for (const uint32_t index : {ior.eta_index, ior.k_index}) {
      if (index >= data.spectrum_values.size()) {
        continue;
      }
      const auto& spectrum = data.spectrum_values[index];
      const float3 rgb = spectrum.integrated_value;
      if ((spectrum.spectral_entry_count == 0u) && ((rgb.x != rgb.y) || (rgb.y != rgb.z))) {
        // Preserve explicitly RGB optical constants, which have no unique spectral interpretation.
        return false;
      }
      sampled = sampled || (spectrum.spectral_entry_count > 0u);
    }
    return true;
  };
  const bool film = bsdf_resource_thinfilm_enabled(material.thinfilm);
  const bool coating = material.coherent_coating.max_thickness_nm > 0.0f;
  if ((spectral_ior(material.ext_ior) == false) || (spectral_ior(material.int_ior) == false) || (film && (spectral_ior(material.thinfilm.ior) == false)) ||
      (coating && (spectral_ior(material.coherent_coating.ior) == false))) {
    return cache_mode;
  }
  return (sampled || film || coating) ? kBSDFEnergyCompensationCacheModeSpectralScalar : cache_mode;
}

bool bind_energy_compensation_interface(SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode,
  const std::vector<Scene::EnergyCompensationInterface>& previous_interfaces, std::unordered_map<uint64_t, uint32_t>& interface_cache, TaskScheduler& scheduler, RHIContext* rhi,
  EnergyCompensationGpuPipeline* gpu_pipeline, uint32_t& out_interface_index) {
  if (interface_lut_dimensions_valid(material, material_class, cache_mode, data.diagnostic) == false) {
    return false;
  }
  const uint64_t hash = hash_material_interface(data, material, material_class, cache_mode);
  const auto found = interface_cache.find(hash);
  if (found != interface_cache.end()) {
    out_interface_index = found->second;
    return true;
  }

  const auto resident = data.energy_compensation_interface_cache.find(hash);
  if (resident != data.energy_compensation_interface_cache.end()) {
    const uint32_t previous_index = resident->second;
    out_interface_index = static_cast<uint32_t>(data.energy_compensation_interfaces.size());
    for (uint32_t coating_index = 0u; coating_index < previous_interfaces[previous_index].coating_slice_count; ++coating_index) {
      data.add_energy_compensation_interface(previous_interfaces[previous_index + coating_index]);
    }
    interface_cache[hash] = out_interface_index;
    return true;
  }

  const uint32_t source_mode = interface_source_cache_mode(data, material, material_class, cache_mode);
  const GeneratedInterfacePaths paths = interface_paths(material_class, hash_material_interface(data, material, material_class, source_mode));
  bool cache_ready = false;
  if ((rhi != nullptr) && (gpu_pipeline != nullptr)) {
    cache_ready = ensure_cache_file_gpu(*rhi, *gpu_pipeline, data, material, material_class, source_mode, paths);
  } else {
    cache_ready = ensure_cache_file(data, material, material_class, source_mode, paths, scheduler);
  }
  if (cache_ready == false) {
    return false;
  }

  std::vector<Scene::EnergyCompensationInterface> prepared_interfaces;
  if (load_energy_compensation_interface_images(data, material, material_class, source_mode, cache_mode, paths, prepared_interfaces) == false) {
    return false;
  }
  out_interface_index = static_cast<uint32_t>(data.energy_compensation_interfaces.size());
  for (const auto& prepared : prepared_interfaces) {
    data.add_energy_compensation_interface(prepared);
  }
  interface_cache[hash] = out_interface_index;
  log::info("Bound material energy-compensation interface %u to %s", out_interface_index, paths.directional.generic_string().c_str());
  return true;
}

Material coherent_conductor_black_endpoint(const Material& material) {
  Material result = material;
  result.int_ior = material.ext_ior;
  result.int_ior.cls = SpectralDistribution::Conductor;
  return result;
}

Material coherent_conductor_bare_endpoint(const Material& material) {
  Material result = material;
  result.thinfilm = {};
  result.coherent_coating.min_thickness_nm = 0.0f;
  result.coherent_coating.max_thickness_nm = 0.0f;
  result.coherent_coating.thickness_image = kInvalidIndex;
  return result;
}

}  // namespace

bool ensure_energy_compensation_interfaces_impl(SceneData& data, TaskScheduler& scheduler, RHIContext* rhi) {
  std::unordered_map<uint64_t, uint32_t> interface_cache;
  EnergyCompensationGpuPipeline gpu_pipeline = {};
  bool result = true;
  const uint32_t cache_mode = data.options.properties[Scene::Properties::Spectral] ? kBSDFEnergyCompensationCacheModeSpectralScalar : kBSDFEnergyCompensationCacheModeIntegratedRGB;
  const std::vector<Scene::EnergyCompensationInterface> previous_interfaces = data.energy_compensation_interfaces;
  std::vector<uint4> previous_material_interfaces;
  previous_material_interfaces.reserve(data.materials.size());
  for (const Material& material : data.materials) {
    previous_material_interfaces.push_back({material.energy_compensation_interface_index, material.conductor_energy_compensation_interface_index,
      material.thermal_energy_compensation_interface_index, material.coating.energy_compensation_interface_index});
  }

  data.energy_compensation_interfaces.clear();
  for (Material& material : data.materials) {
    material.energy_compensation_interface_index = kInvalidIndex;
    material.conductor_energy_compensation_interface_index = kInvalidIndex;
    material.thermal_energy_compensation_interface_index = kInvalidIndex;
    material.coating.energy_compensation_interface_index = kInvalidIndex;
  }

  for (Material& material : data.materials) {
    if (material.cls == MaterialClass::Velvet) {
      if ((velvet_tint_passive(data, material.reflectance) == false) || (velvet_tint_passive(data, material.scattering) == false)) {
        data.diagnostic.fail(SceneDiagnosticKind::Validation, "Energy-compensation tables",
          "Velvet requires finite effective sheen and base reflectance in [0, 1] in the current render mode");
        result = false;
        continue;
      }
      result = bind_velvet_interface(data, previous_interfaces, interface_cache, scheduler, material.energy_compensation_interface_index) && result;
      continue;
    }
    if (material.cls == MaterialClass::CoatedConductor) {
      if (data.thermal_unit_spectrum_index == kInvalidIndex) {
        data.thermal_unit_spectrum_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
      }
      material.coating.unit_spectrum_index = data.thermal_unit_spectrum_index;
      Material interface_material = material;
      interface_material.cls = MaterialClass::Dielectric;
      interface_material.int_ior = material.coating.ior;
      interface_material.roughness = material.coating.roughness;
      const bool interface_bound = dielectric_interface_is_untextured_delta(interface_material) ||
                                   bind_energy_compensation_interface(data, interface_material, MaterialClass::Dielectric, cache_mode, previous_interfaces, interface_cache,
                                     scheduler, rhi, &gpu_pipeline, material.coating.energy_compensation_interface_index);
      const Material substrate_material = material_coating_substrate(material);
      bool substrate_bound = true;
      if (substrate_material.cls == MaterialClass::Conductor) {
        substrate_bound = bind_energy_compensation_interface(data, substrate_material, MaterialClass::Conductor, cache_mode, previous_interfaces, interface_cache, scheduler, rhi,
          &gpu_pipeline, material.energy_compensation_interface_index);
      } else if (substrate_material.cls == MaterialClass::Dielectric) {
        substrate_bound = (material.coherent_coating.ior.cls != SpectralDistribution::Conductor) || velvet_tint_passive(data, substrate_material.scattering);
        if (substrate_bound == false) {
          data.diagnostic.fail(SceneDiagnosticKind::Validation, "Energy-compensation tables",
            "Coated dielectric requires finite effective transmission tint in [0, 1] in the current render mode");
        } else {
          substrate_bound = dielectric_interface_is_untextured_delta(substrate_material) ||
                            bind_energy_compensation_interface(data, substrate_material, MaterialClass::Dielectric, cache_mode, previous_interfaces, interface_cache, scheduler,
                              rhi, &gpu_pipeline, material.energy_compensation_interface_index);
        }
      } else if (substrate_material.cls == MaterialClass::Velvet) {
        substrate_bound = velvet_tint_passive(data, substrate_material.reflectance) && velvet_tint_passive(data, substrate_material.scattering);
        if (substrate_bound == false) {
          data.diagnostic.fail(SceneDiagnosticKind::Validation, "Energy-compensation tables",
            "Coated Velvet requires finite effective sheen and base reflectance in [0, 1] in the current render mode");
        } else {
          substrate_bound = bind_velvet_interface(data, previous_interfaces, interface_cache, scheduler, material.energy_compensation_interface_index);
        }
      }
      result = (interface_bound && substrate_bound) && result;
      continue;
    }
    if (material.cls == MaterialClass::OpenPBR) {
      Material dielectric_material = material;
      dielectric_material.cls = MaterialClass::Dielectric;
      const bool dielectric_bound = bind_energy_compensation_interface(data, dielectric_material, MaterialClass::Dielectric, cache_mode, previous_interfaces, interface_cache,
        scheduler, rhi, &gpu_pipeline, material.energy_compensation_interface_index);

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
      const bool conductor_bound = bind_energy_compensation_interface(data, conductor_material, MaterialClass::Conductor, cache_mode, previous_interfaces, interface_cache,
        scheduler, rhi, &gpu_pipeline, material.conductor_energy_compensation_interface_index);
      result = (dielectric_bound && conductor_bound) && result;
      continue;
    }

    if (bsdf_conductor_has_finite_coating(material)) {
      if (velvet_tint_passive(data, material.reflectance) == false) {
        data.diagnostic.fail(SceneDiagnosticKind::Validation, "Energy-compensation tables",
          "Finite-coated conductor requires finite effective substrate reflectance in [0, 1] in the current render mode");
        result = false;
        continue;
      }
      const Material black_endpoint = coherent_conductor_black_endpoint(material);
      const Material bare_endpoint = coherent_conductor_bare_endpoint(material);
      const bool metal_bound = bind_energy_compensation_interface(data, material, MaterialClass::Conductor, cache_mode, previous_interfaces, interface_cache, scheduler, rhi,
        &gpu_pipeline, material.energy_compensation_interface_index);
      const bool black_bound = bind_energy_compensation_interface(data, black_endpoint, MaterialClass::Conductor, cache_mode, previous_interfaces, interface_cache, scheduler, rhi,
        &gpu_pipeline, material.coating.energy_compensation_interface_index);
      const bool bare_bound = bind_energy_compensation_interface(data, bare_endpoint, MaterialClass::Conductor, cache_mode, previous_interfaces, interface_cache, scheduler, rhi,
        &gpu_pipeline, material.conductor_energy_compensation_interface_index);
      result = (metal_bound && black_bound && bare_bound) && result;
      continue;
    }

    if (((material.cls != MaterialClass::Conductor) && (material.cls != MaterialClass::Dielectric)) && (material.cls != MaterialClass::Plastic)) {
      continue;
    }
    if ((material.cls == MaterialClass::Dielectric) && dielectric_interface_is_untextured_delta(material)) {
      continue;
    }

    const uint32_t material_class = (material.cls == MaterialClass::Plastic) ? MaterialClass::Dielectric : material.cls;
    result = bind_energy_compensation_interface(data, material, material_class, cache_mode, previous_interfaces, interface_cache, scheduler, rhi, &gpu_pipeline,
               material.energy_compensation_interface_index) &&
             result;
    if (((material.cls == MaterialClass::Conductor) || (material.cls == MaterialClass::Plastic)) && (material.temperature_kelvin > 0.0f)) {
      result = bind_energy_compensation_interface(data, material, material_class, kBSDFEnergyCompensationCacheModeSpectralScalar, previous_interfaces, interface_cache, scheduler,
                 rhi, &gpu_pipeline, material.thermal_energy_compensation_interface_index) &&
               result;
    }
  }

  if ((rhi != nullptr) && gpu_pipeline.pipeline.valid()) {
    rhi->device().destroy_pipeline(gpu_pipeline.pipeline);
  }

  if (result == false) {
    for (const auto& [hash, index] : interface_cache) {
      if (data.energy_compensation_interface_cache.contains(hash) == false) {
        release_energy_compensation_interface_block(data, data.energy_compensation_interfaces, index);
      }
    }
    data.energy_compensation_interfaces = previous_interfaces;
    for (uint32_t material_index = 0u; material_index < data.materials.size(); ++material_index) {
      data.materials[material_index].energy_compensation_interface_index = previous_material_interfaces[material_index].x;
      data.materials[material_index].conductor_energy_compensation_interface_index = previous_material_interfaces[material_index].y;
      data.materials[material_index].thermal_energy_compensation_interface_index = previous_material_interfaces[material_index].z;
      data.materials[material_index].coating.energy_compensation_interface_index = previous_material_interfaces[material_index].w;
    }
  } else {
    for (const auto& [hash, index] : data.energy_compensation_interface_cache) {
      if (interface_cache.contains(hash) == false) {
        release_energy_compensation_interface_block(data, previous_interfaces, index);
      }
    }
    data.energy_compensation_interface_cache = std::move(interface_cache);
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
  float base_tolerance, float relative_tolerance, SceneDiagnostic& diagnostic) {
  std::vector<float4> cpu_pixels;
  std::vector<float4> gpu_pixels;
  if ((load_rgba32f_lut(cpu_path, width, height, cpu_pixels, diagnostic) == false) || (load_rgba32f_lut(gpu_path, width, height, gpu_pixels, diagnostic) == false)) {
    diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to load energy-compensation parity LUT %s", label);
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
        diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Non-finite energy-compensation parity value %s pixel %u channel %u cpu %.9f gpu %.9f", label,
          pixel, channel, cpu_value, gpu_value);
        return false;
      }

      const float error = fabsf(cpu_value - gpu_value);
      const float cpu_abs = fabsf(cpu_value);
      const float gpu_abs = fabsf(gpu_value);
      const float scale = (cpu_abs > gpu_abs) ? cpu_abs : gpu_abs;
      const float tolerance = base_tolerance + relative_tolerance * scale;
      if (error > tolerance) {
        diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables",
          "Energy-compensation GPU parity failed %s pixel %u channel %u cpu %.9f gpu %.9f error %.9f tolerance %.9f", label, pixel, channel, cpu_value, gpu_value, error,
          tolerance);
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
  const ThinfilmEval thinfilm = sample_interface_slice(data, material, spect, 0u, 1u);

  const bool cpu_generated = measure_energy_compensation_parity_generation("conductor rgb", "CPU", [&]() {
    return generate_conductor_interface(cpu_paths, ext_ior, int_ior, thinfilm, false, scheduler, data.diagnostic);
  });
  const bool gpu_generated = measure_energy_compensation_parity_generation("conductor rgb", "GPU", [&]() {
    return generate_conductor_interface_gpu(rhi, pipeline, data, material, gpu_paths, gpu_paths, kBSDFEnergyCompensationCacheModeIntegratedRGB, 0u, 1u, 0u, false);
  });

  if ((cpu_generated == false) || (gpu_generated == false)) {
    data.diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to generate integrated conductor energy-compensation parity LUTs");
    return false;
  }

  bool valid = true;
  valid = compare_energy_compensation_lut("conductor rgb directional", cpu_paths.directional, gpu_paths.directional, kEnergyCompensationConductorLutSize,
            kEnergyCompensationConductorLutSize, 4.0e-4f, 3.0e-3f, data.diagnostic) &&
          valid;
  valid =
    compare_energy_compensation_lut("conductor rgb average", cpu_paths.average, gpu_paths.average, kEnergyCompensationConductorLutSize, 1u, 4.0e-4f, 3.0e-3f, data.diagnostic) &&
    valid;
  valid = compare_energy_compensation_lut("conductor rgb geometric", cpu_paths.geometric, gpu_paths.geometric, kEnergyCompensationConductorLutSize,
            kEnergyCompensationConductorLutSize, 4.0e-4f, 3.0e-3f, data.diagnostic) &&
          valid;
  valid = compare_energy_compensation_lut("conductor rgb geometric average", cpu_paths.geometric_average, gpu_paths.geometric_average, kEnergyCompensationConductorLutSize, 1u,
            4.0e-4f, 3.0e-3f, data.diagnostic) &&
          valid;
  valid = compare_energy_compensation_lut("conductor rgb fms", cpu_paths.conductor_fms, gpu_paths.conductor_fms, kEnergyCompensationConductorLutSize, 1u, 4.0e-4f, 3.0e-3f,
            data.diagnostic) &&
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
  const ThinfilmEval thinfilm = sample_interface_slice(data, material, spect, 0u, 1u);

  const bool cpu_generated = measure_energy_compensation_parity_generation(label, "CPU", [&]() {
    return generate_dielectric_interface(cpu_paths, ext_ior, int_ior, thinfilm, dielectric_lut_layout(material), scheduler, data.diagnostic);
  });
  const bool gpu_generated = measure_energy_compensation_parity_generation(label, "GPU", [&]() {
    return generate_dielectric_interface_gpu(rhi, pipeline, data, material, gpu_paths, kBSDFEnergyCompensationCacheModeIntegratedRGB, 0u, 1u, 0u);
  });

  if ((cpu_generated == false) || (gpu_generated == false)) {
    data.diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to generate integrated dielectric energy-compensation parity LUTs");
    return false;
  }

  const std::string directional_label = std::string(label) + " directional";
  const std::string average_label = std::string(label) + " average";
  bool valid = true;
  valid = compare_energy_compensation_lut(directional_label.c_str(), cpu_paths.directional, gpu_paths.directional,
            kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize, kEnergyCompensationDielectricLutSize, 8.0e-4f, 6.0e-3f, data.diagnostic) &&
          valid;
  valid = compare_energy_compensation_lut(average_label.c_str(), cpu_paths.average, gpu_paths.average, kEnergyCompensationDielectricAverageWidth,
            kEnergyCompensationDielectricLutSize, 8.0e-4f, 6.0e-3f, data.diagnostic) &&
          valid;
  return valid;
}

bool validate_energy_compensation_spectral_conductor_parity(RHIContext& rhi, EnergyCompensationGpuPipeline& pipeline, const SceneData& data, const Material& material,
  TaskScheduler& scheduler) {
  const GeneratedInterfacePaths cpu_paths = parity_interface_paths("cpu_conductor_spectral");
  const GeneratedInterfacePaths gpu_paths = parity_interface_paths("gpu_conductor_spectral");
  const SpectralQuery spect = {};
  const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
  const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
  const ThinfilmEval thinfilm = sample_interface_slice(data, material, spect, 0u, 1u);

  const bool cpu_generated = measure_energy_compensation_parity_generation("conductor spectral", "CPU", [&]() {
    const bool geometric_generated = generate_conductor_geometric_interface(cpu_paths, ext_ior, int_ior, thinfilm, scheduler, data.diagnostic);
    const bool spectral_generated = generate_conductor_interface_spectral(data, material, cpu_paths, 0u, 1u, 0u, scheduler);
    return (geometric_generated && spectral_generated);
  });
  const bool gpu_generated = measure_energy_compensation_parity_generation("conductor spectral", "GPU", [&]() {
    return generate_conductor_interface_gpu(rhi, pipeline, data, material, gpu_paths, gpu_paths, kBSDFEnergyCompensationCacheModeSpectralScalar, 0u, 1u, 0u, true);
  });

  if ((cpu_generated == false) || (gpu_generated == false)) {
    data.diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to generate spectral conductor energy-compensation parity LUTs");
    return false;
  }

  bool valid = true;
  valid = compare_energy_compensation_lut("conductor spectral directional", cpu_paths.directional, gpu_paths.directional, kEnergyCompensationConductorLutSize,
            kEnergyCompensationConductorLutSize, 4.0e-4f, 3.0e-3f, data.diagnostic) &&
          valid;
  valid = compare_energy_compensation_lut("conductor spectral average", cpu_paths.average, gpu_paths.average, kEnergyCompensationConductorLutSize, 1u, 4.0e-4f, 3.0e-3f,
            data.diagnostic) &&
          valid;
  valid = compare_energy_compensation_lut("conductor spectral fms", cpu_paths.conductor_fms, gpu_paths.conductor_fms, kEnergyCompensationConductorLutSize, 1u, 4.0e-4f, 3.0e-3f,
            data.diagnostic) &&
          valid;
  valid = compare_energy_compensation_lut("conductor spectral geometric", cpu_paths.geometric, gpu_paths.geometric, kEnergyCompensationConductorLutSize,
            kEnergyCompensationConductorLutSize, 4.0e-4f, 3.0e-3f, data.diagnostic) &&
          valid;
  valid = compare_energy_compensation_lut("conductor spectral geometric average", cpu_paths.geometric_average, gpu_paths.geometric_average, kEnergyCompensationConductorLutSize, 1u,
            4.0e-4f, 3.0e-3f, data.diagnostic) &&
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
    data.diagnostic.fail(SceneDiagnosticKind::Resource, "Energy-compensation tables", "Failed to generate spectral dielectric energy-compensation parity LUTs");
    return false;
  }

  const std::string directional_label = std::string(label) + " directional";
  const std::string average_label = std::string(label) + " average";
  const std::string probability_label = std::string(label) + " probability";
  bool valid = true;
  valid = compare_energy_compensation_lut(directional_label.c_str(), cpu_paths.directional, gpu_paths.directional,
            kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize, kEnergyCompensationDielectricLutSize, 8.0e-4f, 6.0e-3f, data.diagnostic) &&
          valid;
  valid = compare_energy_compensation_lut(average_label.c_str(), cpu_paths.average, gpu_paths.average, kEnergyCompensationDielectricAverageWidth,
            kEnergyCompensationDielectricLutSize, 8.0e-4f, 6.0e-3f, data.diagnostic) &&
          valid;
  valid = compare_energy_compensation_lut(probability_label.c_str(), cpu_paths.probability, gpu_paths.probability,
            kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize, kEnergyCompensationDielectricLutSize, 8.0e-4f, 6.0e-3f, data.diagnostic) &&
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
  return ensure_energy_compensation_interfaces_impl(data, scheduler, rhi.valid() ? &rhi : nullptr);
}

EnergyCompensationGenerationResult generate_energy_compensation_interfaces_step(SceneData& data, TaskScheduler& scheduler, RHIContext& rhi,
  EnergyCompensationGenerationContext& context) {
  if (rhi.valid() == false) {
    return EnergyCompensationGenerationResult::Failed;
  }

  if (data.defaults.conductor_eta == kInvalidIndex) {
    data.defaults.conductor_eta = data.add_spectrum(SpectralDistribution::constant(0.0f));
  }
  if (data.defaults.conductor_k == kInvalidIndex) {
    data.defaults.conductor_k = data.add_spectrum(SpectralDistribution::constant(1000000.0f));
  }

  const uint32_t cache_mode = data.options.properties[Scene::Properties::Spectral] ? kBSDFEnergyCompensationCacheModeSpectralScalar : kBSDFEnergyCompensationCacheModeIntegratedRGB;
  auto visit_interfaces = [&](auto&& callback) {
    for (const Material& material : data.materials) {
      if (material.cls == MaterialClass::CoatedConductor) {
        Material interface_material = material;
        interface_material.cls = MaterialClass::Dielectric;
        interface_material.int_ior = material.coating.ior;
        interface_material.roughness = material.coating.roughness;
        const Material substrate_material = material_coating_substrate(material);
        if ((material.coherent_coating.ior.cls == SpectralDistribution::Conductor) && (substrate_material.cls == MaterialClass::Dielectric) &&
            (velvet_tint_passive(data, substrate_material.scattering) == false)) {
          data.diagnostic.fail(SceneDiagnosticKind::Validation, "Energy-compensation tables",
            "Coated dielectric requires finite effective transmission tint in [0, 1] in the current render mode");
          return false;
        }
        if ((dielectric_interface_is_untextured_delta(interface_material) == false) && (callback(interface_material, MaterialClass::Dielectric, cache_mode) == false)) {
          return false;
        }
        if ((substrate_material.cls == MaterialClass::Conductor) ||
            ((substrate_material.cls == MaterialClass::Dielectric) && (dielectric_interface_is_untextured_delta(substrate_material) == false))) {
          if (callback(substrate_material, substrate_material.cls, cache_mode) == false) {
            return false;
          }
        }
        continue;
      }
      if (material.cls == MaterialClass::OpenPBR) {
        Material dielectric_material = material;
        dielectric_material.cls = MaterialClass::Dielectric;
        if (callback(dielectric_material, MaterialClass::Dielectric, cache_mode) == false) {
          return false;
        }

        Material conductor_material = material;
        conductor_material.cls = MaterialClass::Conductor;
        conductor_material.int_ior.cls = SpectralDistribution::Conductor;
        conductor_material.int_ior.eta_index = data.defaults.conductor_eta;
        conductor_material.int_ior.k_index = data.defaults.conductor_k;
        if (callback(conductor_material, MaterialClass::Conductor, cache_mode) == false) {
          return false;
        }
        continue;
      }

      if (bsdf_conductor_has_finite_coating(material)) {
        const Material black_endpoint = coherent_conductor_black_endpoint(material);
        const Material bare_endpoint = coherent_conductor_bare_endpoint(material);
        if ((callback(material, MaterialClass::Conductor, cache_mode) == false) || (callback(black_endpoint, MaterialClass::Conductor, cache_mode) == false) ||
            (callback(bare_endpoint, MaterialClass::Conductor, cache_mode) == false)) {
          return false;
        }
        continue;
      }

      if (((material.cls != MaterialClass::Conductor) && (material.cls != MaterialClass::Dielectric)) && (material.cls != MaterialClass::Plastic)) {
        continue;
      }
      if ((material.cls == MaterialClass::Dielectric) && dielectric_interface_is_untextured_delta(material)) {
        continue;
      }

      const uint32_t material_class = (material.cls == MaterialClass::Plastic) ? MaterialClass::Dielectric : material.cls;
      if (callback(material, material_class, cache_mode) == false) {
        return false;
      }
      if (((material.cls == MaterialClass::Conductor) || (material.cls == MaterialClass::Plastic)) && (material.temperature_kelvin > 0.0f) &&
          (cache_mode != kBSDFEnergyCompensationCacheModeSpectralScalar)) {
        if (callback(material, material_class, kBSDFEnergyCompensationCacheModeSpectralScalar) == false) {
          return false;
        }
      }
    }
    return true;
  };

  if (context.initialized == false) {
    std::unordered_set<uint64_t> counted_interfaces;
    uint32_t total_steps = 0u;
    const bool dimensions_valid = visit_interfaces([&](const Material& material, uint32_t material_class, uint32_t interface_cache_mode) {
      interface_cache_mode = interface_source_cache_mode(data, material, material_class, interface_cache_mode);
      if (interface_lut_dimensions_valid(material, material_class, interface_cache_mode, data.diagnostic) == false) {
        return false;
      }
      const uint64_t hash = hash_material_interface(data, material, material_class, interface_cache_mode);
      if (counted_interfaces.insert(hash).second) {
        total_steps += pending_cache_file_step_count(material, material_class, interface_cache_mode, interface_paths(material_class, hash));
      }
      return true;
    });
    if (dimensions_valid == false) {
      return EnergyCompensationGenerationResult::Failed;
    }
    context.total_steps = total_steps;
    context.completed_steps = 0u;
    context.initialized = true;
  }

  EnergyCompensationGpuPipeline pipeline = {
    .pipeline = context.pipeline,
    .initialization_failed = context.pipeline_initialization_failed,
  };
  std::unordered_set<uint64_t> generated_interfaces;
  CacheGenerationStepResult generation_result = CacheGenerationStepResult::Complete;
  visit_interfaces([&](const Material& material, uint32_t material_class, uint32_t interface_cache_mode) {
    interface_cache_mode = interface_source_cache_mode(data, material, material_class, interface_cache_mode);
    const uint64_t hash = hash_material_interface(data, material, material_class, interface_cache_mode);
    if (generated_interfaces.insert(hash).second == false) {
      return true;
    }

    generation_result =
      ensure_cache_file_gpu_async_step(rhi, pipeline, scheduler, data, material, material_class, interface_cache_mode, interface_paths(material_class, hash), context);
    return generation_result == CacheGenerationStepResult::Complete;
  });
  context.pipeline = pipeline.pipeline;
  context.pipeline_initialization_failed = pipeline.initialization_failed;

  if (generation_result == CacheGenerationStepResult::Failed) {
    return EnergyCompensationGenerationResult::Failed;
  }
  if (generation_result == CacheGenerationStepResult::Pending) {
    return EnergyCompensationGenerationResult::Pending;
  }
  if (generation_result == CacheGenerationStepResult::Generated) {
    context.completed_steps = std::min(context.completed_steps + 1u, context.total_steps);
    return EnergyCompensationGenerationResult::Pending;
  }

  const bool interfaces_ready = ensure_energy_compensation_interfaces_impl(data, scheduler, &rhi);
  cleanup_energy_compensation_generation(rhi, context);
  return interfaces_ready ? EnergyCompensationGenerationResult::Complete : EnergyCompensationGenerationResult::Failed;
}

void cleanup_energy_compensation_generation(RHIContext& rhi, EnergyCompensationGenerationContext& context) {
  if (context.pending_step != nullptr) {
    cleanup_energy_compensation_async_step(rhi, static_cast<EnergyCompensationAsyncStep*>(context.pending_step));
  }
  if (context.pipeline.valid() && rhi.valid()) {
    rhi.device().destroy_pipeline(context.pipeline);
  }
  context = {};
}

bool validate_energy_compensation_gpu_lut_parity(RHIContext& rhi, TaskScheduler& scheduler) {
  return validate_energy_compensation_gpu_lut_parity_impl(rhi, scheduler);
}

}  // namespace etx
