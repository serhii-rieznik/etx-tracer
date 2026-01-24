#pragma once

#include <etx/core/handle.hxx>

#include <stdint.h>
#include <string>
#include <unordered_map>

namespace etx {

enum class RHIResult : uint32_t {
  Success = 0,
  OutOfMemory,
  InvalidHandle,
  DeviceLost,
  UnsupportedFeature,
  ValidationError,
  InvalidArgument,
  NotImplemented,
};

struct RHIResultInfo {
  RHIResult result = RHIResult::Success;
  const char* message = nullptr;
  uint32_t line = 0;
  const char* file = nullptr;
  const char* function = nullptr;
};

template <typename T>
struct RHICreateResult {
  RHIResult result = RHIResult::Success;
  T handle = {};
};

using RHIBindlessHandle = uint64_t;

using RHIBuffer = RHIBindlessHandle;
using RHITexture = RHIBindlessHandle;
using RHISampler = RHIBindlessHandle;
using RHIShader = Handle;
using RHIPipeline = Handle;

using RHICreateBindlessResult = RHICreateResult<RHIBindlessHandle>;
using RHICreateShaderResult = RHICreateResult<RHIShader>;
using RHICreatePipelineResult = RHICreateResult<RHIPipeline>;

enum class RHIBackend : uint32_t {
  Vulkan = 0,
  Metal = 1,
};

enum class RHIShaderStage : uint32_t {
  Vertex = 0,
  Fragment = 1,
  Compute = 2,
};

enum class RHIIndexType : uint32_t {
  UInt16 = 0,
  UInt32 = 1,
};

enum class RHIBufferUsage : uint32_t {
  Vertex = 1u << 0u,
  Index = 1u << 1u,
  Uniform = 1u << 2u,
  Storage = 1u << 3u,
  TransferSrc = 1u << 4u,
  TransferDst = 1u << 5u,
  AccelerationStructureBuild = 1u << 6u,
  AccelerationStructureStorage = 1u << 7u,
  ShaderBindingTable = 1u << 8u,
};

inline RHIBufferUsage operator|(RHIBufferUsage a, RHIBufferUsage b) {
  return static_cast<RHIBufferUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

enum class RHITextureFormat : uint32_t {
  Undefined = 0,

  R8_UNORM = 1,
  R8G8_UNORM = 2,
  R8G8B8_UNORM = 3,
  R8G8B8A8_UNORM = 4,

  R32_FLOAT = 5,
  R32G32_FLOAT = 6,
  R32G32B32_FLOAT = 7,
  R32G32B32A32_FLOAT = 8,

  R8G8B8A8_SRGB = 9,
  B8G8R8A8_SRGB = 13,

  D32_FLOAT = 10,
  D24_UNORM_S8_UINT = 11,
  D32_FLOAT_S8_UINT = 12,
};

enum class RHITextureUsage : uint32_t {
  Sampled = 1u << 0u,
  Storage = 1u << 1u,
  ColorAttachment = 1u << 2u,
  DepthStencilAttachment = 1u << 3u,
  TransferSrc = 1u << 4u,
  TransferDst = 1u << 5u,
};

inline RHITextureUsage operator|(RHITextureUsage a, RHITextureUsage b) {
  return static_cast<RHITextureUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

enum class RHISamplerFilter : uint32_t {
  Nearest = 0,
  Linear = 1,
};

enum class RHISamplerAddressMode : uint32_t {
  Repeat = 0,
  MirroredRepeat = 1,
  ClampToEdge = 2,
  ClampToBorder = 3,
  MirrorClampToEdge = 4,
};

enum class RHISamplerMipmapMode : uint32_t {
  Nearest = 0,
  Linear = 1,
};

struct RHISamplerDesc {
  RHISamplerFilter min_filter = RHISamplerFilter::Linear;
  RHISamplerFilter mag_filter = RHISamplerFilter::Linear;
  RHISamplerMipmapMode mipmap_mode = RHISamplerMipmapMode::Linear;
  RHISamplerAddressMode address_mode_u = RHISamplerAddressMode::Repeat;
  RHISamplerAddressMode address_mode_v = RHISamplerAddressMode::Repeat;
  RHISamplerAddressMode address_mode_w = RHISamplerAddressMode::Repeat;
  float max_anisotropy = 1.0f;
};

enum class RHISamplerType : uint32_t {
  LinearRepeat = 0,
  LinearClamp = 1,
  NearestRepeat = 2,
  NearestClamp = 3,

  Count,
};

enum class RHIPrimitiveTopology : uint32_t {
  TriangleList = 0,
};

enum class RHIVertexFormat : uint32_t {
  Float2 = 0,
  Float3 = 1,
  Float4 = 2,
};

enum class RHIVertexInputRate : uint32_t {
  Vertex = 0,
  Instance = 1,
};

enum class RHICompareOp : uint32_t {
  Never = 0,
  Less = 1,
  Equal = 2,
  LessOrEqual = 3,
  Greater = 4,
  NotEqual = 5,
  GreaterOrEqual = 6,
  Always = 7,
};

struct RHIRasterizationState {
  bool depth_clamp_enable = false;
  bool rasterizer_discard_enable = false;
  float depth_bias_constant_factor = 0.0f;
  float depth_bias_clamp = 0.0f;
  float depth_bias_slope_factor = 0.0f;
  float line_width = 1.0f;
};

struct RHIDepthStencilState {
  bool depth_test_enable = true;
  bool depth_write_enable = true;
  RHICompareOp depth_compare_op = RHICompareOp::Less;
  bool depth_bounds_test_enable = false;
  float min_depth_bounds = 0.0f;
  float max_depth_bounds = 1.0f;
};

enum class RHIBlendFactor : uint32_t {
  Zero = 0,
  One = 1,
  SrcColor = 2,
  OneMinusSrcColor = 3,
  DstColor = 4,
  OneMinusDstColor = 5,
  SrcAlpha = 6,
  OneMinusSrcAlpha = 7,
  DstAlpha = 8,
  OneMinusDstAlpha = 9,
  ConstantColor = 10,
  OneMinusConstantColor = 11,
  ConstantAlpha = 12,
  OneMinusConstantAlpha = 13,
  SrcAlphaSaturate = 14,
};

enum class RHIBlendOp : uint32_t {
  Add = 0,
  Subtract = 1,
  ReverseSubtract = 2,
  Min = 3,
  Max = 4,
};

struct RHIBlendState {
  bool blend_enable = false;
  RHIBlendFactor src_color_blend_factor = RHIBlendFactor::One;
  RHIBlendFactor dst_color_blend_factor = RHIBlendFactor::Zero;
  RHIBlendOp color_blend_op = RHIBlendOp::Add;
  RHIBlendFactor src_alpha_blend_factor = RHIBlendFactor::One;
  RHIBlendFactor dst_alpha_blend_factor = RHIBlendFactor::Zero;
  RHIBlendOp alpha_blend_op = RHIBlendOp::Add;
};

enum class RHIResourceState : uint32_t {
  Undefined = 0,
  General = 1,
  ColorAttachment = 2,
  DepthStencilAttachment = 3,
  ShaderReadOnly = 4,
  TransferSrc = 5,
  TransferDst = 6,
  Present = 7,
  AccelerationStructure = 8,
};

enum class RHIResourceType : uint32_t {
  Buffer = 0,
  Texture = 1,
  Sampler = 2,
  AccelerationStructure = 3,
};

inline constexpr uint32_t kRHIBindlessDescriptorIndexBits = 32;
inline constexpr uint32_t kRHIBindlessGenerationBits = 30;
inline constexpr uint32_t kRHIBindlessResourceTypeBits = 2;

inline constexpr uint32_t kRHIBindlessDescriptorIndexMask = 0xFFFFFFFFu;
inline constexpr uint32_t kRHIBindlessGenerationMask = 0x3FFFFFFFu;
inline constexpr uint32_t kRHIBindlessResourceTypeMask = 0x3u;

inline RHIBindlessHandle make_bindless_handle(RHIResourceType type, uint32_t generation, uint32_t descriptor_index) {
  uint64_t result = 0;
  result |= static_cast<uint64_t>(descriptor_index) & kRHIBindlessDescriptorIndexMask;
  result |= (static_cast<uint64_t>(generation) & kRHIBindlessGenerationMask) << kRHIBindlessDescriptorIndexBits;
  result |= (static_cast<uint64_t>(type) & kRHIBindlessResourceTypeMask) << (kRHIBindlessDescriptorIndexBits + kRHIBindlessGenerationBits);
  return result;
}

inline RHIResourceType get_bindless_resource_type(RHIBindlessHandle handle) {
  return static_cast<RHIResourceType>((handle >> (kRHIBindlessGenerationBits + kRHIBindlessDescriptorIndexBits)) & kRHIBindlessResourceTypeMask);
}

inline uint32_t get_bindless_generation(RHIBindlessHandle handle) {
  return (handle >> kRHIBindlessDescriptorIndexBits) & kRHIBindlessGenerationMask;
}

inline uint32_t get_bindless_descriptor_index(RHIBindlessHandle handle) {
  return handle & kRHIBindlessDescriptorIndexMask;
}

struct RHIViewport {
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float min_depth = 0.0f;
  float max_depth = 1.0f;
};

struct RHIRect {
  int32_t x = 0;
  int32_t y = 0;
  uint32_t width = 0;
  uint32_t height = 0;
};

struct RHIDispatchDesc {
  uint32_t group_count_x = 1;
  uint32_t group_count_y = 1;
  uint32_t group_count_z = 1;
};

struct RHIDrawDesc {
  uint32_t vertex_count;
  uint32_t instance_count;
  uint32_t first_vertex;
  uint32_t first_instance;
};

struct RHIIndexedDrawDesc {
  uint32_t index_count = 0;
  uint32_t instance_count = 1;
  uint32_t first_index = 0;
  uint32_t vertex_offset = 0;
  uint32_t first_instance = 0;
  RHIIndexType index_type = RHIIndexType::UInt16;
};

struct RHIBufferDesc {
  uint64_t size = 0;
  RHIBufferUsage usage = RHIBufferUsage::Vertex;
  bool host_visible = false;
};

struct RHITextureDesc {
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t depth = 1;
  uint32_t mip_levels = 1;
  uint32_t array_layers = 1;
  RHITextureFormat format = RHITextureFormat::R8G8B8A8_UNORM;
  RHITextureUsage usage = RHITextureUsage::Sampled;
  bool host_visible = false;
};

struct RHIShaderDesc {
  const void* spirv_data = nullptr;
  uint64_t spirv_size = 0;
  RHIShaderStage stage = RHIShaderStage::Vertex;
};

struct RHIShaderVariantDesc {
  std::string hlsl_source;
  std::string entry_point = "main";
  RHIShaderStage stage = RHIShaderStage::Vertex;
  std::string source_name = "shader.hlsl";
  std::unordered_map<std::string, std::string> defines;
};

struct RHIVertexAttribute {
  uint32_t location = 0;
  uint32_t binding = 0;
  RHIVertexFormat format = RHIVertexFormat::Float3;
  uint32_t offset = 0;
};

struct RHIVertexBinding {
  uint32_t binding = 0;
  uint32_t stride = 0;
  RHIVertexInputRate input_rate = RHIVertexInputRate::Vertex;
};

struct RHIGraphicsPipelineDesc {
  RHIShaderDesc vertex_shader = {};
  std::string vertex_entry_point = "main";
  RHIShaderDesc fragment_shader = {};
  std::string fragment_entry_point = "main";
  RHIRasterizationState rasterization = {};
  RHIDepthStencilState depth_stencil = {};
  RHIBlendState blend = {};

  uint32_t vertex_attribute_count = 0;
  RHIVertexAttribute vertex_attributes[16] = {};
  uint32_t vertex_binding_count = 0;
  RHIVertexBinding vertex_bindings[8] = {};

  RHIPrimitiveTopology primitive_topology = RHIPrimitiveTopology::TriangleList;

  uint32_t color_attachment_count = 1;
  RHITextureFormat color_formats[8] = {};
  RHITextureFormat depth_format = RHITextureFormat::Undefined;
};

struct RHIComputePipelineDesc {
  RHIShaderDesc compute_shader = {};
  std::string entry_point = "main";

  uint32_t local_size_x = 1;
  uint32_t local_size_y = 1;
  uint32_t local_size_z = 1;
};

}  // namespace etx
