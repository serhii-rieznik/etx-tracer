#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <array>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <etx/rhi/metal/mt_rhi.hxx>
#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>

#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <sys/wait.h>

namespace etx {

namespace {

constexpr uint32_t kMetalBindlessBufferBinding = 0u;
constexpr uint32_t kMetalBindlessTextureBinding = 1u;
constexpr uint32_t kMetalBindlessSamplerBinding = 2u;
constexpr uint32_t kMetalBindlessStorageTextureBinding = 3u;
constexpr uint32_t kMetalBindlessAccelerationStructureBinding = 4u;
constexpr uint32_t kMetalBindlessRWBufferBinding = 5u;
constexpr uint32_t kMetalBindlessBindingCount = 6u;
constexpr uint32_t kInvalidMetalBufferIndex = std::numeric_limits<uint32_t>::max();
static_assert(kMetalBindlessBindingCount == kRHIMetalBindlessBindingCount);
static_assert(kInvalidMetalBufferIndex == kRHIInvalidMetalBufferIndex);
constexpr size_t kMetalMaxColorAttachments = 8u;
constexpr uint32_t kMetalShaderCacheVersion = 2u;
constexpr uint32_t kMetalTimestampQueryCount = 2048u;
constexpr uint32_t kRHIAccelerationStructureInstanceFlagDisableTriangleCulling = 1u << 0u;
constexpr uint32_t kRHIAccelerationStructureInstanceFlagFrontFacingCCW = 1u << 1u;
constexpr uint32_t kRHIAccelerationStructureInstanceFlagForceOpaque = 1u << 2u;
constexpr uint32_t kRHIAccelerationStructureInstanceFlagForceNonOpaque = 1u << 3u;

template <typename T>
bool has_flag(T value, T flag) {
  using U = std::underlying_type_t<T>;
  return (static_cast<U>(value) & static_cast<U>(flag)) != 0;
}

MTLPixelFormat to_metal_format(RHITextureFormat format) {
  switch (format) {
    case RHITextureFormat::R8_UNORM:
      return MTLPixelFormatR8Unorm;
    case RHITextureFormat::R8G8_UNORM:
      return MTLPixelFormatRG8Unorm;
    case RHITextureFormat::R8G8B8A8_UNORM:
      return MTLPixelFormatRGBA8Unorm;
    case RHITextureFormat::B8G8R8A8_UNORM:
      return MTLPixelFormatBGRA8Unorm;
    case RHITextureFormat::R32_FLOAT:
      return MTLPixelFormatR32Float;
    case RHITextureFormat::R32G32_FLOAT:
      return MTLPixelFormatRG32Float;
    case RHITextureFormat::R32G32B32A32_FLOAT:
      return MTLPixelFormatRGBA32Float;
    case RHITextureFormat::R16G16B16A16_FLOAT:
      return MTLPixelFormatRGBA16Float;
    case RHITextureFormat::R8G8B8A8_SRGB:
      return MTLPixelFormatRGBA8Unorm_sRGB;
    case RHITextureFormat::B8G8R8A8_SRGB:
      return MTLPixelFormatBGRA8Unorm_sRGB;
    case RHITextureFormat::D32_FLOAT:
      return MTLPixelFormatDepth32Float;
    default:
      return MTLPixelFormatInvalid;
  }
}

uint32_t format_bytes_per_pixel(RHITextureFormat format) {
  switch (format) {
    case RHITextureFormat::R8_UNORM:
      return 1u;
    case RHITextureFormat::R8G8_UNORM:
      return 2u;
    case RHITextureFormat::R8G8B8A8_UNORM:
    case RHITextureFormat::B8G8R8A8_UNORM:
    case RHITextureFormat::R32_FLOAT:
    case RHITextureFormat::R8G8B8A8_SRGB:
    case RHITextureFormat::B8G8R8A8_SRGB:
    case RHITextureFormat::D32_FLOAT:
      return 4u;
    case RHITextureFormat::R32G32_FLOAT:
      return 8u;
    case RHITextureFormat::R16G16B16A16_FLOAT:
      return 8u;
    case RHITextureFormat::R32G32B32A32_FLOAT:
      return 16u;
    default:
      return 0u;
  }
}

MTLTextureUsage to_metal_texture_usage(RHITextureUsage usage) {
  MTLTextureUsage result = MTLTextureUsageUnknown;
  if (has_flag(usage, RHITextureUsage::Sampled)) {
    result |= MTLTextureUsageShaderRead;
  }
  if (has_flag(usage, RHITextureUsage::Storage)) {
    result |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  }
  if (has_flag(usage, RHITextureUsage::ColorAttachment) || has_flag(usage, RHITextureUsage::DepthAttachment)) {
    result |= MTLTextureUsageRenderTarget;
  }
  return result;
}

MTLSamplerMinMagFilter to_metal_filter(RHISamplerFilter filter) {
  return (filter == RHISamplerFilter::Linear) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
}

MTLSamplerMipFilter to_metal_mip_filter(RHISamplerMipmapMode mode) {
  return (mode == RHISamplerMipmapMode::Linear) ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
}

MTLSamplerAddressMode to_metal_address_mode(RHISamplerAddressMode mode) {
  switch (mode) {
    case RHISamplerAddressMode::Repeat:
      return MTLSamplerAddressModeRepeat;
    case RHISamplerAddressMode::MirroredRepeat:
      return MTLSamplerAddressModeMirrorRepeat;
    case RHISamplerAddressMode::ClampToEdge:
      return MTLSamplerAddressModeClampToEdge;
    case RHISamplerAddressMode::ClampToBorder:
      return MTLSamplerAddressModeClampToBorderColor;
    case RHISamplerAddressMode::MirrorClampToEdge:
      return MTLSamplerAddressModeMirrorClampToEdge;
    default:
      return MTLSamplerAddressModeClampToEdge;
  }
}

MTLCompareFunction to_metal_compare(RHICompareOp op) {
  switch (op) {
    case RHICompareOp::Never:
      return MTLCompareFunctionNever;
    case RHICompareOp::Less:
      return MTLCompareFunctionLess;
    case RHICompareOp::Equal:
      return MTLCompareFunctionEqual;
    case RHICompareOp::LessOrEqual:
      return MTLCompareFunctionLessEqual;
    case RHICompareOp::Greater:
      return MTLCompareFunctionGreater;
    case RHICompareOp::NotEqual:
      return MTLCompareFunctionNotEqual;
    case RHICompareOp::GreaterOrEqual:
      return MTLCompareFunctionGreaterEqual;
    case RHICompareOp::Always:
      return MTLCompareFunctionAlways;
    default:
      return MTLCompareFunctionAlways;
  }
}

MTLBlendFactor to_metal_blend_factor(RHIBlendFactor factor) {
  switch (factor) {
    case RHIBlendFactor::Zero:
      return MTLBlendFactorZero;
    case RHIBlendFactor::One:
      return MTLBlendFactorOne;
    case RHIBlendFactor::SrcColor:
      return MTLBlendFactorSourceColor;
    case RHIBlendFactor::OneMinusSrcColor:
      return MTLBlendFactorOneMinusSourceColor;
    case RHIBlendFactor::DstColor:
      return MTLBlendFactorDestinationColor;
    case RHIBlendFactor::OneMinusDstColor:
      return MTLBlendFactorOneMinusDestinationColor;
    case RHIBlendFactor::SrcAlpha:
      return MTLBlendFactorSourceAlpha;
    case RHIBlendFactor::OneMinusSrcAlpha:
      return MTLBlendFactorOneMinusSourceAlpha;
    case RHIBlendFactor::DstAlpha:
      return MTLBlendFactorDestinationAlpha;
    case RHIBlendFactor::OneMinusDstAlpha:
      return MTLBlendFactorOneMinusDestinationAlpha;
    case RHIBlendFactor::ConstantColor:
      return MTLBlendFactorBlendColor;
    case RHIBlendFactor::OneMinusConstantColor:
      return MTLBlendFactorOneMinusBlendColor;
    case RHIBlendFactor::ConstantAlpha:
      return MTLBlendFactorBlendAlpha;
    case RHIBlendFactor::OneMinusConstantAlpha:
      return MTLBlendFactorOneMinusBlendAlpha;
    case RHIBlendFactor::SrcAlphaSaturate:
      return MTLBlendFactorSourceAlphaSaturated;
    default:
      return MTLBlendFactorOne;
  }
}

MTLBlendOperation to_metal_blend_op(RHIBlendOp op) {
  switch (op) {
    case RHIBlendOp::Add:
      return MTLBlendOperationAdd;
    case RHIBlendOp::Subtract:
      return MTLBlendOperationSubtract;
    case RHIBlendOp::ReverseSubtract:
      return MTLBlendOperationReverseSubtract;
    case RHIBlendOp::Min:
      return MTLBlendOperationMin;
    case RHIBlendOp::Max:
      return MTLBlendOperationMax;
    default:
      return MTLBlendOperationAdd;
  }
}

MTLPrimitiveType to_metal_primitive(RHIPrimitiveTopology topology) {
  return (topology == RHIPrimitiveTopology::LineList) ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;
}

MTLIndexType to_metal_index_type(RHIIndexType type) {
  return (type == RHIIndexType::UInt32) ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
}

MTLVertexFormat to_metal_vertex_format(RHIVertexFormat format) {
  switch (format) {
    case RHIVertexFormat::Float2:
      return MTLVertexFormatFloat2;
    case RHIVertexFormat::Float3:
      return MTLVertexFormatFloat3;
    case RHIVertexFormat::Float4:
      return MTLVertexFormatFloat4;
    default:
      return MTLVertexFormatInvalid;
  }
}

MTLAttributeFormat to_metal_acceleration_structure_vertex_format(RHIVertexFormat format) {
  switch (format) {
    case RHIVertexFormat::Float2:
      return MTLAttributeFormatFloat2;
    case RHIVertexFormat::Float3:
      return MTLAttributeFormatFloat3;
    case RHIVertexFormat::Float4:
      return MTLAttributeFormatFloat4;
    default:
      return MTLAttributeFormatInvalid;
  }
}

MTLAccelerationStructureInstanceOptions to_metal_instance_options(uint32_t flags) {
  MTLAccelerationStructureInstanceOptions result = MTLAccelerationStructureInstanceOptionNone;
  if ((flags & kRHIAccelerationStructureInstanceFlagDisableTriangleCulling) != 0u) {
    result |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
  }
  if ((flags & kRHIAccelerationStructureInstanceFlagFrontFacingCCW) != 0u) {
    result |= MTLAccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise;
  }
  if ((flags & kRHIAccelerationStructureInstanceFlagForceOpaque) != 0u) {
    result |= MTLAccelerationStructureInstanceOptionOpaque;
  }
  if ((flags & kRHIAccelerationStructureInstanceFlagForceNonOpaque) != 0u) {
    result |= MTLAccelerationStructureInstanceOptionNonOpaque;
  }
  return result;
}

MTLPackedFloat4x3 to_metal_transform(const float transform[12]) {
  return MTLPackedFloat4x3(MTLPackedFloat3Make(transform[0], transform[4], transform[8]), MTLPackedFloat3Make(transform[1], transform[5], transform[9]), MTLPackedFloat3Make(transform[2], transform[6], transform[10]),
    MTLPackedFloat3Make(transform[3], transform[7], transform[11]));
}

NSString* make_nsstring(const void* data, size_t size) {
  if ((data == nullptr) || (size == 0u)) {
    return nil;
  }
  return [[[NSString alloc] initWithBytes:data length:size encoding:NSUTF8StringEncoding] autorelease];
}

std::string sanitize_path_component(std::string value) {
  for (char& c : value) {
    if (std::isalnum(static_cast<unsigned char>(c)) == 0) {
      c = '_';
    }
  }

  while ((value.empty() == false) && (value.back() == '_')) {
    value.pop_back();
  }

  return value.empty() ? std::string("unknown") : value;
}

std::string metal_device_archive_suffix(id<MTLDevice> device) {
  if (device == nil) {
    return "unknown";
  }

  std::string device_name = "unknown";
  if (device.name != nil) {
    const char* utf8_name = [device.name UTF8String];
    if ((utf8_name != nullptr) && (utf8_name[0] != 0)) {
      device_name = sanitize_path_component(utf8_name);
    }
  }

  char registry_id[32] = {};
  if (@available(macOS 10.13, *)) {
    std::snprintf(registry_id, sizeof(registry_id), "%016llx", static_cast<unsigned long long>(device.registryID));
  } else {
    std::snprintf(registry_id, sizeof(registry_id), "legacy");
  }

  return device_name + "_" + registry_id;
}

std::filesystem::path metal_pipeline_archive_root_directory() {
  std::filesystem::path root(env().cache_folder());
  root /= "metal";
  root /= "pipeline_archives";
  root /= ("v" + std::to_string(kMetalShaderCacheVersion));
  return root;
}

static std::string format_hash_hex(uint64_t value) {
  char buffer[17] = {};
  std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(value));
  return buffer;
}

std::filesystem::path metal_pipeline_archive_directory(id<MTLDevice> device) {
  return metal_pipeline_archive_root_directory() / metal_device_archive_suffix(device);
}

std::filesystem::path metal_pipeline_archive_path(id<MTLDevice> device, uint64_t cache_key, uint64_t content_hash) {
  return metal_pipeline_archive_directory(device) / (format_hash_hex(cache_key) + "_" + format_hash_hex(content_hash) + ".bin");
}

bool ensure_directory_exists(const std::filesystem::path& directory) {
  if (directory.empty()) {
    return false;
  }

  std::error_code ec;
  std::filesystem::create_directories(directory, ec);
  return ec.value() == 0;
}

const char* safe_nsstring(NSString* string, const char* fallback = "<null>") {
  return (string != nil) ? [string UTF8String] : fallback;
}

const char* metal_command_buffer_status_name(MTLCommandBufferStatus status) {
  switch (status) {
    case MTLCommandBufferStatusNotEnqueued:
      return "NotEnqueued";
    case MTLCommandBufferStatusEnqueued:
      return "Enqueued";
    case MTLCommandBufferStatusCommitted:
      return "Committed";
    case MTLCommandBufferStatusScheduled:
      return "Scheduled";
    case MTLCommandBufferStatusCompleted:
      return "Completed";
    case MTLCommandBufferStatusError:
      return "Error";
    default:
      return "Unknown";
  }
}

const char* metal_command_buffer_error_name(NSInteger code) {
  switch (static_cast<MTLCommandBufferError>(code)) {
    case MTLCommandBufferErrorNone:
      return "None";
    case MTLCommandBufferErrorInternal:
      return "Internal";
    case MTLCommandBufferErrorTimeout:
      return "Timeout";
    case MTLCommandBufferErrorPageFault:
      return "PageFault";
    case MTLCommandBufferErrorAccessRevoked:
      return "AccessRevoked";
    case MTLCommandBufferErrorNotPermitted:
      return "NotPermitted";
    case MTLCommandBufferErrorOutOfMemory:
      return "OutOfMemory";
    case MTLCommandBufferErrorInvalidResource:
      return "InvalidResource";
    case MTLCommandBufferErrorMemoryless:
      return "Memoryless";
    case MTLCommandBufferErrorDeviceRemoved:
      return "DeviceRemoved";
    case MTLCommandBufferErrorStackOverflow:
      return "StackOverflow";
    default:
      return "Unknown";
  }
}

const char* metal_command_encoder_error_state_name(MTLCommandEncoderErrorState state) {
  switch (state) {
    case MTLCommandEncoderErrorStateUnknown:
      return "Unknown";
    case MTLCommandEncoderErrorStateCompleted:
      return "Completed";
    case MTLCommandEncoderErrorStateAffected:
      return "Affected";
    case MTLCommandEncoderErrorStatePending:
      return "Pending";
    case MTLCommandEncoderErrorStateFaulted:
      return "Faulted";
    default:
      return "Unknown";
  }
}

void log_metal_command_buffer_error_details(id<MTLCommandBuffer> command_buffer) {
  if (command_buffer == nil) {
    return;
  }

  NSError* error = command_buffer.error;
  log::error("Metal command buffer failed: label='%s', status=%s", safe_nsstring(command_buffer.label, "<unnamed>"), metal_command_buffer_status_name(command_buffer.status));
  if (error == nil) {
    return;
  }

  log::error("Metal command buffer error: domain=%s, code=%ld (%s), description=%s", safe_nsstring(error.domain), static_cast<long>(error.code), metal_command_buffer_error_name(error.code), safe_nsstring(error.localizedDescription));
  if (error.localizedFailureReason != nil) {
    log::error("Metal command buffer failure reason: %s", safe_nsstring(error.localizedFailureReason));
  }
  if (error.localizedRecoverySuggestion != nil) {
    log::error("Metal command buffer recovery suggestion: %s", safe_nsstring(error.localizedRecoverySuggestion));
  }

  if (@available(macOS 11.0, *)) {
    NSArray<id<MTLCommandBufferEncoderInfo>>* encoder_infos = error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
    for (id<MTLCommandBufferEncoderInfo> encoder_info in encoder_infos) {
      log::error("Metal encoder status: label='%s', state=%s", safe_nsstring(encoder_info.label, "<unnamed>"), metal_command_encoder_error_state_name(encoder_info.errorState));
      for (NSString* signpost in encoder_info.debugSignposts) {
        log::error("Metal encoder signpost: %s", safe_nsstring(signpost));
      }
    }
  }
}

id<MTLCommandBuffer> create_diagnostic_command_buffer(id<MTLCommandQueue> queue, NSString* label) {
  if (queue == nil) {
    return nil;
  }

  id<MTLCommandBuffer> command_buffer = nil;
  if (@available(macOS 11.0, *)) {
    MTLCommandBufferDescriptor* descriptor = [[[MTLCommandBufferDescriptor alloc] init] autorelease];
    descriptor.retainedReferences = YES;
    descriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    command_buffer = [queue commandBufferWithDescriptor:descriptor];
  } else {
    command_buffer = [queue commandBuffer];
  }

  if ((command_buffer != nil) && (label != nil)) {
    command_buffer.label = label;
  }

  if (command_buffer != nil) {
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_command_buffer) {
      if (completed_command_buffer.status == MTLCommandBufferStatusError) {
        log_metal_command_buffer_error_details(completed_command_buffer);
      }
    }];
  }

  return command_buffer;
}

static bool device_reports_raytracing(id<MTLDevice> device) {
  if (device == nil) {
    return false;
  }
  if (@available(macOS 11.0, *)) {
    if ([device respondsToSelector:@selector(supportsRaytracing)]) {
      return device.supportsRaytracing;
    }
  }
  return false;
}

bool texture_prefers_private_storage(const RHITextureDesc& desc) {
  if (desc.host_visible == false) {
    return true;
  }

  if (desc.sample_count > 1u) {
    return true;
  }

  if (has_flag(desc.usage, RHITextureUsage::ColorAttachment) || has_flag(desc.usage, RHITextureUsage::DepthAttachment)) {
    return true;
  }

  return has_flag(desc.usage, RHITextureUsage::Storage) && (has_flag(desc.usage, RHITextureUsage::TransferDst) == false);
}

bool metal_texture_state_is_sampled_read_only(RHIResourceState state) {
  return state == RHIResourceState::ShaderReadOnly;
}

bool metal_texture_state_requires_compute_encoder_break(RHIResourceState old_state, RHIResourceState new_state) {
  if (old_state == new_state) {
    return false;
  }

  const bool old_is_sampled = metal_texture_state_is_sampled_read_only(old_state);
  const bool new_is_sampled = metal_texture_state_is_sampled_read_only(new_state);
  return old_is_sampled != new_is_sampled;
}

}  // namespace

struct MTBindlessManager::Impl {
  struct ResourceEntry {
    uint32_t generation = 0;
    bool valid = false;
    void* resource = nullptr;
  };

  uint32_t max_buffers = kDefaultMaxBuffers;
  uint32_t max_textures = kDefaultMaxTextures;
  uint32_t max_samplers = kDefaultMaxSamplers;
  uint32_t max_acceleration_structures = kDefaultMaxAccelerationStructures;

  uint32_t buffer_count = 0;
  uint32_t texture_count = 0;
  uint32_t sampler_count = 0;
  uint32_t acceleration_structure_count = 0;

  uint64_t revision = 1;

  std::vector<ResourceEntry> buffer_entries = std::vector<ResourceEntry>(max_buffers);
  std::vector<ResourceEntry> texture_entries = std::vector<ResourceEntry>(max_textures);
  std::vector<ResourceEntry> sampler_entries = std::vector<ResourceEntry>(max_samplers);
  std::vector<ResourceEntry> acceleration_structure_entries = std::vector<ResourceEntry>(max_acceleration_structures);
};

struct MTBufferData {
  id<MTLBuffer> buffer = nil;
  RHIBufferDesc desc = {};
  RHIResourceState state = RHIResourceState::Undefined;
};

struct MTTextureData {
  id<MTLTexture> texture = nil;
  RHITextureDesc desc = {};
  RHIResourceState state = RHIResourceState::Undefined;
  bool is_swapchain_texture = false;
};

struct MTAccelerationStructureData {
  id<MTLAccelerationStructure> acceleration_structure = nil;
  id<MTLBuffer> instance_descriptor_buffer = nil;
  RHIAccelerationStructureDesc desc = {};
  uint64_t allocated_size = 0;
  uint64_t instance_descriptor_buffer_size = 0;
  uint64_t build_scratch_size = 0;
};

static bool texture_is_bindless_2d_compatible(const MTTextureData& data, bool require_storage_usage) {
  if (data.texture == nil) {
    return false;
  }

  if (data.texture.textureType != MTLTextureType2D) {
    return false;
  }

  if ((data.desc.sample_count > 1u) || (data.desc.format == RHITextureFormat::D32_FLOAT)) {
    return false;
  }

  if (require_storage_usage) {
    return has_flag(data.desc.usage, RHITextureUsage::Storage);
  }

  return has_flag(data.desc.usage, RHITextureUsage::Sampled);
}

struct MTSamplerData {
  id<MTLSamplerState> sampler = nil;
  RHISamplerDesc desc = {};
};

struct MTSemaphoreData {
  uint64_t value = 0;
};

struct MTPipelineStageData {
  id<MTLFunction> function = nil;
  std::array<id<MTLArgumentEncoder>, kMetalBindlessBindingCount> bindless_argument_encoders = {};
  std::array<id<MTLBuffer>, kMetalBindlessBindingCount> bindless_argument_buffers = {};
  std::array<uint32_t, kMetalBindlessBindingCount> bindless_buffer_indices = {kInvalidMetalBufferIndex, kInvalidMetalBufferIndex, kInvalidMetalBufferIndex, kInvalidMetalBufferIndex, kInvalidMetalBufferIndex, kInvalidMetalBufferIndex};
  std::array<uint64_t, kMetalBindlessBindingCount> encoded_revisions = {};
  std::array<bool, kMetalBindlessBindingCount> uses_bindless_binding = {};
  std::array<MTLBindingAccess, kMetalBindlessBindingCount> bindless_binding_access = {MTLBindingAccessReadOnly, MTLBindingAccessReadOnly, MTLBindingAccessReadOnly, MTLBindingAccessReadWrite, MTLBindingAccessReadOnly, MTLBindingAccessReadWrite};
  bool uses_push_constants = false;
  uint32_t push_constants_buffer_index = kInvalidMetalBufferIndex;
};

struct MTPipelineData {
  bool is_compute = false;
  MTLPrimitiveType primitive = MTLPrimitiveTypeTriangle;
  MTLTriangleFillMode fill_mode = MTLTriangleFillModeFill;
  id<MTLRenderPipelineState> render_pipeline = nil;
  id<MTLComputePipelineState> compute_pipeline = nil;
  id<MTLDepthStencilState> depth_state = nil;
  std::string debug_name = {};
  RHIComputePipelineDesc compute_desc = {};
  RHIGraphicsPipelineDesc graphics_desc = {};
  MTPipelineStageData vertex_stage = {};
  MTPipelineStageData fragment_stage = {};
  MTPipelineStageData compute_stage = {};
};

struct MTLibraryCacheEntry {
  uint64_t content_hash = 0u;
  id<MTLLibrary> library = nil;
};

class MTDevice::Impl {
 public:
  id<MTLDevice> metal_device = nil;
  id<MTLCommandQueue> command_queue = nil;
  MTBindlessManager* bindless_manager = nullptr;
  std::unordered_map<uint64_t, MTLibraryCacheEntry> library_cache = {};

  std::unordered_map<RHIBindlessHandle, MTBufferData> buffers = {};
  std::unordered_map<RHIBindlessHandle, MTTextureData> textures = {};
  std::unordered_map<RHIBindlessHandle, MTSamplerData> samplers = {};
  std::unordered_map<RHIBindlessHandle, MTAccelerationStructureData> acceleration_structures = {};
  std::unordered_map<RHIPipeline, MTPipelineData> pipelines = {};
  std::unordered_map<RHISemaphore, MTSemaphoreData> semaphores = {};

  uint64_t gpu_allocated_bytes = 0;
  uint32_t next_pipeline_index = 1;
  uint32_t next_semaphore_index = 1;
};

class MTCommandBuffer::Impl {
 public:
  void* owner = nullptr;
  id<MTLCommandBuffer> command_buffer = nil;
  id<MTLRenderCommandEncoder> render_encoder = nil;
  id<MTLComputeCommandEncoder> compute_encoder = nil;
  id<MTLBlitCommandEncoder> blit_encoder = nil;
  id<MTLCounterSampleBuffer> timestamp_sample_buffer = nil;
  bool timestamp_sample_buffer_allocation_failed = false;
  uint32_t timestamp_scope_begin_query = ~0u;
  uint32_t timestamp_scope_end_query = ~0u;
  bool timestamp_scope_active = false;
  bool timestamp_scope_uses_stage_sampling = false;

  RHIPipeline current_pipeline = {};
  std::array<uint8_t, 4096> push_constants = {};
  uint32_t push_constants_size = 0;
  MTLViewport viewport = {};
  MTLScissorRect scissor = {};
  bool viewport_valid = false;
  bool scissor_valid = false;
  std::array<RHITexture, kMetalMaxColorAttachments> current_color_attachments = {};
  std::array<RHIResourceState, kMetalMaxColorAttachments> current_color_final_states = {};
  uint32_t current_color_attachment_count = 0u;
  RHITexture current_depth_attachment = {};
  RHIResourceState current_depth_final_state = RHIResourceState::Undefined;
};

struct MTInflightSubmission {
  RHICommandBuffer handle = {};
  id<MTLCommandBuffer> command_buffer = nil;
  id<CAMetalDrawable> drawable = nil;
  bool completion_polled = false;
};

class MTContext::Impl {
 public:
  MTCommandBuffer* find_command_buffer(RHICommandBuffer handle) {
    auto it = command_buffers.find(handle);
    return (it != command_buffers.end()) ? it->second.get() : nullptr;
  }

  const MTCommandBuffer* find_command_buffer(RHICommandBuffer handle) const {
    return const_cast<Impl*>(this)->find_command_buffer(handle);
  }

  MTDevice device = {};
  MTBindlessManager bindless_manager = {};
  std::unordered_map<RHICommandBuffer, std::unique_ptr<MTCommandBuffer>> command_buffers = {};
  std::unordered_map<RHICommandBuffer, id<MTLCounterSampleBuffer>> submitted_timestamp_sample_buffers = {};

  id<MTLDevice> metal_device = nil;
  id<MTLCommandQueue> command_queue = nil;
  id<MTLCounterSet> timestamp_counter_set = nil;
  CAMetalLayer* metal_layer = nil;
  id<CAMetalDrawable> current_drawable = nil;

  RHISemaphore image_acquired = {};
  RHISemaphore render_complete = {};
  RHIBindlessHandle swapchain_texture = {};
  std::array<uint32_t, static_cast<size_t>(RHISamplerType::Count)> predefined_sampler_indices = {kRHIBindlessDescriptorIndexMask, kRHIBindlessDescriptorIndexMask, kRHIBindlessDescriptorIndexMask, kRHIBindlessDescriptorIndexMask,
    kRHIBindlessDescriptorIndexMask};
  std::vector<MTInflightSubmission> inflight_command_buffers = {};
  std::unordered_map<RHICommandBuffer, RHIResult> polled_completion_results = {};
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t current_frame = 0;
  uint32_t next_command_buffer_index = 1u;
  uint64_t next_command_buffer_serial = 1u;
  bool headless = false;
  bool supports_ray_tracing = false;
  bool supports_timestamps = false;
  bool supports_timestamp_dispatch_sampling = false;
};

static void reset_command_buffer_state(MTCommandBuffer::Impl* impl) {
  if (impl == nullptr) {
    return;
  }

  impl->current_pipeline = {};
  impl->push_constants_size = 0u;
  impl->viewport_valid = false;
  impl->scissor_valid = false;
  impl->current_color_attachments = {};
  impl->current_color_final_states = {};
  impl->current_color_attachment_count = 0u;
  impl->current_depth_attachment = {};
  impl->current_depth_final_state = RHIResourceState::Undefined;
  impl->timestamp_sample_buffer_allocation_failed = false;
  impl->timestamp_scope_begin_query = ~0u;
  impl->timestamp_scope_end_query = ~0u;
  impl->timestamp_scope_active = false;
  impl->timestamp_scope_uses_stage_sampling = false;
}

static id<MTLComputeCommandEncoder> ensure_compute_encoder(MTCommandBuffer::Impl* impl) {
  if ((impl == nullptr) || (impl->command_buffer == nil)) {
    return nil;
  }

  [impl->render_encoder endEncoding];
  impl->render_encoder = nil;
  [impl->blit_encoder endEncoding];
  impl->blit_encoder = nil;
  if (impl->compute_encoder == nil) {
    if (impl->timestamp_scope_active && impl->timestamp_scope_uses_stage_sampling) {
      MTLComputePassDescriptor* descriptor = [[MTLComputePassDescriptor alloc] init];
      MTLComputePassSampleBufferAttachmentDescriptor* attachment = descriptor.sampleBufferAttachments[0];
      attachment.sampleBuffer = impl->timestamp_sample_buffer;
      attachment.startOfEncoderSampleIndex = impl->timestamp_scope_begin_query;
      attachment.endOfEncoderSampleIndex = impl->timestamp_scope_end_query;
      impl->compute_encoder = [impl->command_buffer computeCommandEncoderWithDescriptor:descriptor];
      [descriptor release];
    } else {
      impl->compute_encoder = [impl->command_buffer computeCommandEncoder];
    }
  }
  return impl->compute_encoder;
}

static bool ensure_timestamp_sample_buffer(MTCommandBuffer::Impl* command_buffer, id<MTLDevice> device, id<MTLCounterSet> counter_set) {
  if ((command_buffer == nullptr) || (device == nil) || (counter_set == nil)) {
    return false;
  }
  if (command_buffer->timestamp_sample_buffer_allocation_failed) {
    return false;
  }
  if (command_buffer->timestamp_sample_buffer != nil) {
    return true;
  }

  MTLCounterSampleBufferDescriptor* descriptor = [[MTLCounterSampleBufferDescriptor alloc] init];
  descriptor.counterSet = counter_set;
  descriptor.label = @"ETX timestamp queries";
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.sampleCount = kMetalTimestampQueryCount;
  NSError* error = nil;
  command_buffer->timestamp_sample_buffer = [device newCounterSampleBufferWithDescriptor:descriptor error:&error];
  [descriptor release];
  if (command_buffer->timestamp_sample_buffer == nil) {
    command_buffer->timestamp_sample_buffer_allocation_failed = true;
    const char* error_message = (error != nil) ? [[error localizedDescription] UTF8String] : nullptr;
    if (error_message == nullptr) {
      error_message = "unknown error";
    }
    log::error("Metal RHI: failed to create timestamp sample buffer: %s", error_message);
    return false;
  }
  return true;
}

static MTLPrimitiveAccelerationStructureDescriptor* create_metal_blas_descriptor(const RHIAccelerationStructureGeometry* geometries, uint32_t geometry_count, MTDevice::Impl* device, std::string* out_error = nullptr) {
  if ((device == nullptr) || (geometries == nullptr) || (geometry_count == 0u)) {
    if (out_error != nullptr) {
      *out_error = "Invalid BLAS geometry description.";
    }
    return nil;
  }

  NSMutableArray<MTLAccelerationStructureGeometryDescriptor*>* geometry_descriptors = [NSMutableArray arrayWithCapacity:geometry_count];
  for (uint32_t i = 0; i < geometry_count; ++i) {
    const auto& src_geo = geometries[i];
    const auto vertex_it = device->buffers.find(src_geo.triangles.vertex_buffer);
    if (vertex_it == device->buffers.end() || (vertex_it->second.buffer == nil)) {
      if (out_error != nullptr) {
        *out_error = "BLAS vertex buffer handle is invalid.";
      }
      return nil;
    }

    const MTLAttributeFormat vertex_format = to_metal_acceleration_structure_vertex_format(src_geo.triangles.vertex_format);
    if (vertex_format == MTLAttributeFormatInvalid) {
      if (out_error != nullptr) {
        *out_error = "BLAS vertex format is unsupported by Metal.";
      }
      return nil;
    }

    MTLAccelerationStructureTriangleGeometryDescriptor* triangle_descriptor = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    triangle_descriptor.vertexBuffer = vertex_it->second.buffer;
    triangle_descriptor.vertexBufferOffset = 0u;
    triangle_descriptor.vertexStride = src_geo.triangles.vertex_stride;
    triangle_descriptor.vertexFormat = vertex_format;
    triangle_descriptor.opaque = src_geo.is_opaque ? YES : NO;

    if (src_geo.triangles.index_buffer.valid()) {
      const auto index_it = device->buffers.find(src_geo.triangles.index_buffer);
      if (index_it == device->buffers.end() || (index_it->second.buffer == nil)) {
        if (out_error != nullptr) {
          *out_error = "BLAS index buffer handle is invalid.";
        }
        return nil;
      }
      triangle_descriptor.indexBuffer = index_it->second.buffer;
      triangle_descriptor.indexBufferOffset = static_cast<NSUInteger>(src_geo.triangles.index_buffer_offset);
      triangle_descriptor.indexType = to_metal_index_type(src_geo.triangles.index_type);
      triangle_descriptor.triangleCount = static_cast<NSUInteger>(src_geo.triangles.index_count / 3u);
    } else {
      triangle_descriptor.indexBuffer = nil;
      triangle_descriptor.triangleCount = static_cast<NSUInteger>(src_geo.triangles.vertex_count / 3u);
    }

    [geometry_descriptors addObject:triangle_descriptor];
  }

  MTLPrimitiveAccelerationStructureDescriptor* descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
  descriptor.usage = MTLAccelerationStructureUsageNone;
  descriptor.geometryDescriptors = geometry_descriptors;
  return descriptor;
}

static MTLInstanceAccelerationStructureDescriptor* create_metal_tlas_sizing_descriptor(uint32_t instance_count, bool allow_update) {
  MTLInstanceAccelerationStructureDescriptor* descriptor = [MTLInstanceAccelerationStructureDescriptor descriptor];
  descriptor.usage = allow_update ? MTLAccelerationStructureUsageRefit : MTLAccelerationStructureUsageNone;
  descriptor.instanceCount = instance_count;
  descriptor.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;
  return descriptor;
}

static std::vector<MTBindlessManager::Impl::ResourceEntry>& bindless_entries_for_type(MTBindlessManager::Impl* impl, RHIResourceType type) {
  switch (type) {
    case RHIResourceType::Buffer:
      return impl->buffer_entries;
    case RHIResourceType::Texture:
      return impl->texture_entries;
    case RHIResourceType::Sampler:
      return impl->sampler_entries;
    case RHIResourceType::AccelerationStructure:
      return impl->acceleration_structure_entries;
    default:
      ETX_ASSERT(false);
      return impl->buffer_entries;
  }
}

static uint32_t& bindless_count_for_type(MTBindlessManager::Impl* impl, RHIResourceType type) {
  switch (type) {
    case RHIResourceType::Buffer:
      return impl->buffer_count;
    case RHIResourceType::Texture:
      return impl->texture_count;
    case RHIResourceType::Sampler:
      return impl->sampler_count;
    case RHIResourceType::AccelerationStructure:
      return impl->acceleration_structure_count;
    default:
      ETX_ASSERT(false);
      return impl->buffer_count;
  }
}

static void update_bindless_revision(MTBindlessManager::Impl* impl) {
  impl->revision = std::max<uint64_t>(impl->revision + 1u, 1u);
}

static void reap_completed_command_buffers(std::vector<MTInflightSubmission>& inflight) {
  size_t write_index = 0u;
  for (MTInflightSubmission& submission : inflight) {
    id<MTLCommandBuffer> command_buffer = submission.command_buffer;
    if (command_buffer == nil) {
      if (submission.drawable != nil) {
        [submission.drawable release];
      }
      continue;
    }

    const MTLCommandBufferStatus status = command_buffer.status;
    if (((status == MTLCommandBufferStatusCompleted) || (status == MTLCommandBufferStatusError)) && (submission.completion_polled == false)) {
      [command_buffer release];
      if (submission.drawable != nil) {
        [submission.drawable release];
      }
      continue;
    }

    inflight[write_index++] = submission;
  }
  inflight.resize(write_index);
}

static bool wait_for_inflight_command_buffers(std::vector<MTInflightSubmission>& inflight, std::unordered_map<RHICommandBuffer, RHIResult>* polled_results) {
  bool success = true;
  for (MTInflightSubmission& submission : inflight) {
    id<MTLCommandBuffer> command_buffer = submission.command_buffer;
    if (command_buffer == nil) {
      if (submission.drawable != nil) {
        [submission.drawable release];
      }
      continue;
    }

    [command_buffer waitUntilCompleted];
    const bool command_succeeded = command_buffer.status != MTLCommandBufferStatusError;
    success = command_succeeded && success;
    if ((polled_results != nullptr) && submission.completion_polled) {
      (*polled_results)[submission.handle] = command_succeeded ? RHIResult::Success : RHIResult::ValidationError;
    }
    [command_buffer release];
    if (submission.drawable != nil) {
      [submission.drawable release];
    }
  }
  inflight.clear();
  return success;
}

static void prune_cache_slot_versions(const std::filesystem::path& directory, const std::string& slot_prefix, const std::filesystem::path& current_path) {
  std::error_code ec = {};
  if (std::filesystem::exists(directory, ec) == false) {
    return;
  }

  for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
    if (ec.value() != 0) {
      return;
    }
    const std::string file_name = entry.path().filename().string();
    if (entry.is_regular_file() && (entry.path() != current_path) && (file_name.rfind(slot_prefix, 0u) == 0u)) {
      std::filesystem::remove(entry.path(), ec);
      ec.clear();
    }
  }
}

static void prune_obsolete_metal_cache_versions(const std::filesystem::path& current_root) {
  const std::filesystem::path parent = current_root.parent_path();
  std::error_code ec = {};
  if (std::filesystem::exists(parent, ec) == false) {
    return;
  }

  for (const auto& entry : std::filesystem::directory_iterator(parent, ec)) {
    if (ec.value() != 0) {
      return;
    }
    const std::string directory_name = entry.path().filename().string();
    if (entry.is_directory() && (entry.path() != current_root) && (directory_name.rfind("v", 0u) == 0u)) {
      std::filesystem::remove_all(entry.path(), ec);
      ec.clear();
    }
  }
}

static std::filesystem::path metal_library_cache_root_directory() {
  std::filesystem::path root(env().cache_folder());
  root /= "metal";
  root /= "libraries";
  root /= ("v" + std::to_string(kMetalShaderCacheVersion));
  return root;
}

static std::filesystem::path metal_library_cache_path(uint64_t cache_key, uint64_t content_hash) {
  return metal_library_cache_root_directory() / (format_hash_hex(cache_key) + "_" + format_hash_hex(content_hash) + ".metallib");
}

static std::string shell_quote(const std::string& value) {
  std::string result = "'";
  for (char c : value) {
    if (c == '\'') {
      result += "'\\''";
    } else {
      result.push_back(c);
    }
  }
  result.push_back('\'');
  return result;
}

static bool write_binary_file(const std::filesystem::path& path, const void* data, size_t size) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (stream.is_open() == false) {
    return false;
  }
  if ((data != nullptr) && (size > 0u)) {
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
  }
  return stream.good();
}

static bool run_shell_command_capture_output(const std::string& command, int& out_exit_code, std::string& out_output) {
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    out_exit_code = -1;
    out_output = "Failed to launch shell command.";
    return false;
  }

  out_output.clear();
  std::array<char, 4096> buffer = {};
  while (true) {
    const size_t bytes_read = fread(buffer.data(), 1u, buffer.size(), pipe);
    if (bytes_read > 0u) {
      out_output.append(buffer.data(), bytes_read);
    }
    if (bytes_read < buffer.size()) {
      if (feof(pipe)) {
        break;
      }
      if (ferror(pipe)) {
        break;
      }
    }
  }

  const int status = pclose(pipe);
#if defined(WIFEXITED) && defined(WEXITSTATUS)
  out_exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : status;
#else
  out_exit_code = status;
#endif
  return out_exit_code == 0;
}

static bool compile_metal_source_to_metallib(const std::string& source_text, uint64_t source_hash, const std::filesystem::path& metallib_path, std::string& out_error) {
  if (ensure_directory_exists(metallib_path.parent_path()) == false) {
    out_error = "Failed to create metallib cache directory.";
    return false;
  }

  const std::filesystem::path temp_root = std::filesystem::path(env().tmp_folder()) / "metal_library_cache";
  if (ensure_directory_exists(temp_root) == false) {
    out_error = "Failed to create temporary Metal compiler directory.";
    return false;
  }

  static std::atomic<uint64_t> next_temp_file_id = 0u;
  const uint64_t process_id = static_cast<uint64_t>([[NSProcessInfo processInfo] processIdentifier]);
  const std::string hash_name = format_hash_hex(source_hash) + "_" + std::to_string(process_id) + "_" + std::to_string(next_temp_file_id.fetch_add(1u));
  const std::filesystem::path source_path = temp_root / (hash_name + ".metal");
  const std::filesystem::path air_path = temp_root / (hash_name + ".air");
  const std::filesystem::path temp_metallib_path = temp_root / (hash_name + ".metallib");
  std::error_code ec = {};
  std::filesystem::remove(source_path, ec);
  std::filesystem::remove(air_path, ec);
  std::filesystem::remove(temp_metallib_path, ec);

  if (write_binary_file(source_path, source_text.data(), source_text.size()) == false) {
    out_error = "Failed to write temporary Metal source file.";
    return false;
  }

  int exit_code = 0;
  std::string command_output = {};
  const std::string metal_command = "xcrun metal -c " + shell_quote(source_path.string()) + " -o " + shell_quote(air_path.string()) + " 2>&1";
  if (run_shell_command_capture_output(metal_command, exit_code, command_output) == false) {
    out_error = "metal compiler failed (" + std::to_string(exit_code) + "): " + command_output;
    std::filesystem::remove(source_path, ec);
    std::filesystem::remove(air_path, ec);
    return false;
  }

  const std::string metallib_command = "xcrun metallib " + shell_quote(air_path.string()) + " -o " + shell_quote(temp_metallib_path.string()) + " 2>&1";
  if (run_shell_command_capture_output(metallib_command, exit_code, command_output) == false) {
    out_error = "metallib failed (" + std::to_string(exit_code) + "): " + command_output;
    std::filesystem::remove(source_path, ec);
    std::filesystem::remove(air_path, ec);
    std::filesystem::remove(temp_metallib_path, ec);
    return false;
  }

  std::filesystem::remove(source_path, ec);
  std::filesystem::remove(air_path, ec);
  std::filesystem::rename(temp_metallib_path, metallib_path, ec);
  if (ec.value() != 0) {
    std::filesystem::remove(temp_metallib_path, ec);
    out_error = "Failed to publish compiled metallib cache file.";
    return false;
  }
  return true;
}

static id<MTLLibrary> load_or_create_cached_metal_library(MTDevice::Impl* device, const std::string& source_text, uint64_t shader_cache_key, uint64_t shader_content_hash, std::string& out_error) {
  if ((device == nullptr) || (device->metal_device == nil) || source_text.empty()) {
    out_error = "Metal shader source is empty.";
    return nil;
  }

  const uint64_t source_hash = etx_hash64(source_text.data(), source_text.size());
  const uint64_t cache_key = (shader_cache_key != 0u) ? shader_cache_key : source_hash;
  const uint64_t content_hash = (shader_content_hash != 0u) ? shader_content_hash : source_hash;
  const uint64_t memory_key = etx_hash64_continue(&content_hash, sizeof(content_hash), cache_key);
  if (auto it = device->library_cache.find(cache_key); (it != device->library_cache.end()) && (it->second.content_hash == content_hash)) {
    return it->second.library;
  }

  const auto cached_metallib_path = metal_library_cache_path(cache_key, content_hash);
  auto retain_current_library = [&](id<MTLLibrary> library) -> id<MTLLibrary> {
    auto it = device->library_cache.find(cache_key);
    if (it != device->library_cache.end()) {
      [it->second.library release];
      it->second = {content_hash, library};
    } else {
      device->library_cache.emplace(cache_key, MTLibraryCacheEntry{content_hash, library});
    }
    return library;
  };
  auto load_metallib = [&](const std::filesystem::path& path) -> id<MTLLibrary> {
    NSString* library_path = [NSString stringWithUTF8String:path.string().c_str()];
    NSError* library_error = nil;
    id<MTLLibrary> library = [device->metal_device newLibraryWithURL:[NSURL fileURLWithPath:library_path] error:&library_error];
    if (library == nil) {
      out_error = library_error ? std::string([[library_error localizedDescription] UTF8String]) : "Failed to load metallib.";
    }
    return library;
  };

  std::error_code ec = {};
  if (std::filesystem::exists(cached_metallib_path, ec) && (ec.value() == 0)) {
    if (id<MTLLibrary> library = load_metallib(cached_metallib_path)) {
      return retain_current_library(library);
    }
    std::filesystem::remove(cached_metallib_path, ec);
  }

  std::string compile_error = {};
  if (compile_metal_source_to_metallib(source_text, memory_key, cached_metallib_path, compile_error)) {
    if (id<MTLLibrary> library = load_metallib(cached_metallib_path)) {
      return retain_current_library(library);
    }
    std::filesystem::remove(cached_metallib_path, ec);
  } else if (compile_error.empty() == false) {
    log::warning("Metal RHI: failed to build cached metallib, falling back to source compilation: %s", compile_error.c_str());
  }

  NSError* library_error = nil;
  NSString* source = [NSString stringWithUTF8String:source_text.c_str()];
  id<MTLLibrary> fallback_library = [device->metal_device newLibraryWithSource:source options:nil error:&library_error];
  if (fallback_library == nil) {
    out_error = library_error ? std::string([[library_error localizedDescription] UTF8String]) : "Failed to create MTLLibrary.";
    return nil;
  }

  retain_current_library(fallback_library);
  out_error.clear();
  return fallback_library;
}

static void prune_metal_library_cache_slot(const RHIShaderDesc& shader_desc) {
  const uint64_t content_hash = (shader_desc.content_hash != 0u) ? shader_desc.content_hash : etx_hash64(shader_desc.spirv_data, static_cast<size_t>(shader_desc.spirv_size));
  const uint64_t cache_key = (shader_desc.cache_key != 0u) ? shader_desc.cache_key : content_hash;
  const std::filesystem::path path = metal_library_cache_path(cache_key, content_hash);
  std::error_code ec = {};
  if (std::filesystem::exists(path, ec) == false) {
    return;
  }
  prune_cache_slot_versions(path.parent_path(), format_hash_hex(cache_key) + "_", path);
}

static bool bindless_handle_matches(const MTBindlessManager::Impl::ResourceEntry& entry, RHIBindlessHandle handle) {
  return entry.valid && (entry.generation == get_bindless_generation(handle));
}

static MTLBindingAccess to_metal_binding_access(RHIMetalBindingAccess access) {
  switch (access) {
    case RHIMetalBindingAccess::ReadOnly:
      return MTLBindingAccessReadOnly;
    case RHIMetalBindingAccess::WriteOnly:
      return MTLBindingAccessWriteOnly;
    case RHIMetalBindingAccess::ReadWrite:
      return MTLBindingAccessReadWrite;
  }
  return MTLBindingAccessReadOnly;
}

static NSUInteger bindless_argument_array_length(const MTBindlessManager::Impl* bindless, uint32_t binding_index) {
  if (bindless == nullptr) {
    return 0u;
  }

  switch (binding_index) {
    case kMetalBindlessBufferBinding:
    case kMetalBindlessRWBufferBinding:
      return std::max<NSUInteger>(bindless->max_buffers, 1u);
    case kMetalBindlessTextureBinding:
    case kMetalBindlessStorageTextureBinding:
      return std::max<NSUInteger>(bindless->max_textures, 1u);
    case kMetalBindlessSamplerBinding:
      return std::max<NSUInteger>(bindless->max_samplers, 1u);
    case kMetalBindlessAccelerationStructureBinding:
      return std::max<NSUInteger>(bindless->max_acceleration_structures, 1u);
    default:
      return 0u;
  }
}

static id<MTLArgumentEncoder> create_bindless_argument_encoder(id<MTLDevice> device, const MTBindlessManager::Impl* bindless, uint32_t binding_index, MTLBindingAccess binding_access) {
  if ((device == nil) || (bindless == nullptr)) {
    return nil;
  }

  MTLArgumentDescriptor* descriptor = [MTLArgumentDescriptor argumentDescriptor];
  descriptor.index = 0u;
  descriptor.arrayLength = bindless_argument_array_length(bindless, binding_index);
  descriptor.access = binding_access;

  switch (binding_index) {
    case kMetalBindlessBufferBinding:
      descriptor.dataType = MTLDataTypePointer;
      descriptor.access = MTLBindingAccessReadOnly;
      break;
    case kMetalBindlessTextureBinding:
      descriptor.dataType = MTLDataTypeTexture;
      descriptor.access = MTLBindingAccessReadOnly;
      descriptor.textureType = MTLTextureType2D;
      break;
    case kMetalBindlessSamplerBinding:
      descriptor.dataType = MTLDataTypeSampler;
      descriptor.access = MTLBindingAccessReadOnly;
      break;
    case kMetalBindlessStorageTextureBinding:
      descriptor.dataType = MTLDataTypeTexture;
      descriptor.access = binding_access;
      descriptor.textureType = MTLTextureType2D;
      break;
    case kMetalBindlessAccelerationStructureBinding:
      descriptor.dataType = MTLDataTypeInstanceAccelerationStructure;
      descriptor.access = MTLBindingAccessReadOnly;
      break;
    case kMetalBindlessRWBufferBinding:
      descriptor.dataType = MTLDataTypePointer;
      descriptor.access = MTLBindingAccessReadWrite;
      break;
    default:
      return nil;
  }

  return [device newArgumentEncoderWithArguments:@[ descriptor ]];
}

static void encode_stage_bindless_resources(MTPipelineStageData& stage, MTBindlessManager::Impl* bindless, MTDevice::Impl* device) {
  if ((bindless == nullptr) || (device == nullptr)) {
    return;
  }

  auto encode_buffer_entries = [&](uint32_t binding_index) {
    id<MTLArgumentEncoder> encoder = stage.bindless_argument_encoders[binding_index];
    id<MTLBuffer> argument_buffer = stage.bindless_argument_buffers[binding_index];
    if ((encoder == nil) || (argument_buffer == nil) || (stage.encoded_revisions[binding_index] == bindless->revision)) {
      return;
    }

    [encoder setArgumentBuffer:argument_buffer offset:0];
    for (uint32_t i = 0, e = static_cast<uint32_t>(bindless->buffer_entries.size()); i < e; ++i) {
      id<MTLBuffer> buffer = nil;
      if (bindless->buffer_entries[i].valid) {
        RHIBindlessHandle handle = make_bindless_handle(RHIResourceType::Buffer, bindless->buffer_entries[i].generation, i);
        auto buffer_it = device->buffers.find(handle);
        if (buffer_it != device->buffers.end()) {
          const RHIBufferUsage usage = buffer_it->second.desc.usage;
          const bool allow_binding = (binding_index == kMetalBindlessRWBufferBinding) ? has_flag(usage, RHIBufferUsage::Storage) : true;
          if (allow_binding) {
            buffer = buffer_it->second.buffer;
          }
        }
      }
      [encoder setBuffer:buffer offset:0 atIndex:i];
    }
    stage.encoded_revisions[binding_index] = bindless->revision;
  };

  auto encode_texture_entries = [&](uint32_t binding_index) {
    id<MTLArgumentEncoder> encoder = stage.bindless_argument_encoders[binding_index];
    id<MTLBuffer> argument_buffer = stage.bindless_argument_buffers[binding_index];
    if ((encoder == nil) || (argument_buffer == nil) || (stage.encoded_revisions[binding_index] == bindless->revision)) {
      return;
    }

    [encoder setArgumentBuffer:argument_buffer offset:0];
    for (uint32_t i = 0, e = static_cast<uint32_t>(bindless->texture_entries.size()); i < e; ++i) {
      id<MTLTexture> texture = nil;
      if (bindless->texture_entries[i].valid) {
        RHIBindlessHandle handle = make_bindless_handle(RHIResourceType::Texture, bindless->texture_entries[i].generation, i);
        auto texture_it = device->textures.find(handle);
        if (texture_it != device->textures.end()) {
          const bool allow_binding = (binding_index == kMetalBindlessTextureBinding) ? texture_is_bindless_2d_compatible(texture_it->second, false) : texture_is_bindless_2d_compatible(texture_it->second, true);
          if (allow_binding) {
            texture = texture_it->second.texture;
          }
        }
      }
      [encoder setTexture:texture atIndex:i];
    }
    stage.encoded_revisions[binding_index] = bindless->revision;
  };

  if (stage.uses_bindless_binding[kMetalBindlessBufferBinding]) {
    encode_buffer_entries(kMetalBindlessBufferBinding);
  }
  if (stage.uses_bindless_binding[kMetalBindlessRWBufferBinding]) {
    encode_buffer_entries(kMetalBindlessRWBufferBinding);
  }
  if (stage.uses_bindless_binding[kMetalBindlessTextureBinding]) {
    encode_texture_entries(kMetalBindlessTextureBinding);
  }
  if (stage.uses_bindless_binding[kMetalBindlessStorageTextureBinding]) {
    encode_texture_entries(kMetalBindlessStorageTextureBinding);
  }

  if (stage.uses_bindless_binding[kMetalBindlessSamplerBinding]) {
    id<MTLArgumentEncoder> encoder = stage.bindless_argument_encoders[kMetalBindlessSamplerBinding];
    id<MTLBuffer> argument_buffer = stage.bindless_argument_buffers[kMetalBindlessSamplerBinding];
    if ((encoder != nil) && (argument_buffer != nil) && (stage.encoded_revisions[kMetalBindlessSamplerBinding] != bindless->revision)) {
      [encoder setArgumentBuffer:argument_buffer offset:0];
      for (uint32_t i = 0, e = static_cast<uint32_t>(bindless->sampler_entries.size()); i < e; ++i) {
        id<MTLSamplerState> sampler = nil;
        if (bindless->sampler_entries[i].valid) {
          sampler = static_cast<id<MTLSamplerState>>(bindless->sampler_entries[i].resource);
        }
        [encoder setSamplerState:sampler atIndex:i];
      }
      stage.encoded_revisions[kMetalBindlessSamplerBinding] = bindless->revision;
    }
  }

  if (stage.uses_bindless_binding[kMetalBindlessAccelerationStructureBinding]) {
    id<MTLArgumentEncoder> encoder = stage.bindless_argument_encoders[kMetalBindlessAccelerationStructureBinding];
    id<MTLBuffer> argument_buffer = stage.bindless_argument_buffers[kMetalBindlessAccelerationStructureBinding];
    if ((encoder != nil) && (argument_buffer != nil) && (stage.encoded_revisions[kMetalBindlessAccelerationStructureBinding] != bindless->revision)) {
      [encoder setArgumentBuffer:argument_buffer offset:0];
      if (@available(macOS 11.0, *)) {
        for (uint32_t i = 0, e = static_cast<uint32_t>(bindless->acceleration_structure_entries.size()); i < e; ++i) {
          id<MTLAccelerationStructure> as = nil;
          if (bindless->acceleration_structure_entries[i].valid) {
            as = static_cast<id<MTLAccelerationStructure>>(bindless->acceleration_structure_entries[i].resource);
          }
          [encoder setAccelerationStructure:as atIndex:i];
        }
      }
      stage.encoded_revisions[kMetalBindlessAccelerationStructureBinding] = bindless->revision;
    }
  }
}

static void declare_compute_stage_bindless_resources(id<MTLComputeCommandEncoder> encoder, const MTPipelineStageData& stage, MTBindlessManager::Impl* bindless, MTDevice::Impl* device) {
  if ((encoder == nil) || (bindless == nullptr) || (device == nullptr)) {
    return;
  }

  auto declare_buffer_entries = [&](uint32_t binding_index, MTLResourceUsage usage_mask) {
    if (stage.uses_bindless_binding[binding_index] == false) {
      return;
    }
    for (uint32_t i = 0, e = static_cast<uint32_t>(bindless->buffer_entries.size()); i < e; ++i) {
      const auto& entry = bindless->buffer_entries[i];
      if (entry.valid == false) {
        continue;
      }
      RHIBindlessHandle handle = make_bindless_handle(RHIResourceType::Buffer, entry.generation, i);
      auto buffer_it = device->buffers.find(handle);
      if (buffer_it == device->buffers.end()) {
        continue;
      }
      const RHIBufferUsage usage = buffer_it->second.desc.usage;
      const bool allow_binding = (binding_index == kMetalBindlessRWBufferBinding) ? has_flag(usage, RHIBufferUsage::Storage) : true;
      if (allow_binding) {
        [encoder useResource:buffer_it->second.buffer usage:usage_mask];
      }
    }
  };

  auto declare_texture_entries = [&](uint32_t binding_index, MTLResourceUsage usage_mask) {
    if (stage.uses_bindless_binding[binding_index] == false) {
      return;
    }
    for (uint32_t i = 0, e = static_cast<uint32_t>(bindless->texture_entries.size()); i < e; ++i) {
      const auto& entry = bindless->texture_entries[i];
      if (entry.valid == false) {
        continue;
      }
      RHIBindlessHandle handle = make_bindless_handle(RHIResourceType::Texture, entry.generation, i);
      auto texture_it = device->textures.find(handle);
      if (texture_it == device->textures.end()) {
        continue;
      }
      const bool allow_binding = (binding_index == kMetalBindlessTextureBinding) ? texture_is_bindless_2d_compatible(texture_it->second, false) : texture_is_bindless_2d_compatible(texture_it->second, true);
      if (allow_binding) {
        [encoder useResource:texture_it->second.texture usage:usage_mask];
      }
    }
  };

  constexpr MTLResourceUsage kSampledTextureUsage = MTLResourceUsageRead | MTLResourceUsageSample;
  const MTLBindingAccess storage_texture_access = stage.bindless_binding_access[kMetalBindlessStorageTextureBinding];
  const MTLResourceUsage storage_texture_usage = (storage_texture_access == MTLBindingAccessWriteOnly) ? MTLResourceUsageWrite : (MTLResourceUsageRead | MTLResourceUsageWrite);

  declare_buffer_entries(kMetalBindlessBufferBinding, MTLResourceUsageRead);
  declare_buffer_entries(kMetalBindlessRWBufferBinding, MTLResourceUsageRead | MTLResourceUsageWrite);
  declare_texture_entries(kMetalBindlessTextureBinding, kSampledTextureUsage);
  declare_texture_entries(kMetalBindlessStorageTextureBinding, storage_texture_usage);
  if (stage.uses_bindless_binding[kMetalBindlessAccelerationStructureBinding]) {
    for (uint32_t i = 0, e = static_cast<uint32_t>(bindless->acceleration_structure_entries.size()); i < e; ++i) {
      const auto& entry = bindless->acceleration_structure_entries[i];
      if (entry.valid == false) {
        continue;
      }
      RHIBindlessHandle handle = make_bindless_handle(RHIResourceType::AccelerationStructure, entry.generation, i);
      auto as_it = device->acceleration_structures.find(handle);
      if ((as_it != device->acceleration_structures.end()) && (as_it->second.acceleration_structure != nil)) {
        [encoder useResource:(id<MTLResource>)as_it->second.acceleration_structure usage:MTLResourceUsageRead];
      }
    }
  }
}

static void declare_render_stage_bindless_resources(id<MTLRenderCommandEncoder> encoder, const MTPipelineStageData& stage, MTBindlessManager::Impl* bindless, MTDevice::Impl* device, MTLRenderStages stages) {
  if ((encoder == nil) || (bindless == nullptr) || (device == nullptr)) {
    return;
  }

  auto declare_resource = [&](id<MTLResource> resource, MTLResourceUsage usage_mask) {
    if (resource == nil) {
      return;
    }
    if (@available(macOS 10.15, *)) {
      [encoder useResource:resource usage:usage_mask stages:stages];
    } else {
      [encoder useResource:resource usage:usage_mask];
    }
  };

  auto declare_buffer_entries = [&](uint32_t binding_index, MTLResourceUsage usage_mask) {
    if (stage.uses_bindless_binding[binding_index] == false) {
      return;
    }
    for (uint32_t i = 0, e = static_cast<uint32_t>(bindless->buffer_entries.size()); i < e; ++i) {
      const auto& entry = bindless->buffer_entries[i];
      if (entry.valid == false) {
        continue;
      }
      RHIBindlessHandle handle = make_bindless_handle(RHIResourceType::Buffer, entry.generation, i);
      auto buffer_it = device->buffers.find(handle);
      if (buffer_it == device->buffers.end()) {
        continue;
      }
      const RHIBufferUsage usage = buffer_it->second.desc.usage;
      const bool allow_binding = (binding_index == kMetalBindlessRWBufferBinding) ? has_flag(usage, RHIBufferUsage::Storage) : true;
      if (allow_binding) {
        declare_resource(buffer_it->second.buffer, usage_mask);
      }
    }
  };

  auto declare_texture_entries = [&](uint32_t binding_index, MTLResourceUsage usage_mask) {
    if (stage.uses_bindless_binding[binding_index] == false) {
      return;
    }
    for (uint32_t i = 0, e = static_cast<uint32_t>(bindless->texture_entries.size()); i < e; ++i) {
      const auto& entry = bindless->texture_entries[i];
      if (entry.valid == false) {
        continue;
      }
      RHIBindlessHandle handle = make_bindless_handle(RHIResourceType::Texture, entry.generation, i);
      auto texture_it = device->textures.find(handle);
      if (texture_it == device->textures.end()) {
        continue;
      }
      const bool allow_binding = (binding_index == kMetalBindlessTextureBinding) ? texture_is_bindless_2d_compatible(texture_it->second, false) : texture_is_bindless_2d_compatible(texture_it->second, true);
      if (allow_binding) {
        declare_resource(texture_it->second.texture, usage_mask);
      }
    }
  };

  constexpr MTLResourceUsage kSampledTextureUsage = MTLResourceUsageRead | MTLResourceUsageSample;

  declare_buffer_entries(kMetalBindlessBufferBinding, MTLResourceUsageRead);
  declare_buffer_entries(kMetalBindlessRWBufferBinding, MTLResourceUsageRead | MTLResourceUsageWrite);
  declare_texture_entries(kMetalBindlessTextureBinding, kSampledTextureUsage);
  declare_texture_entries(kMetalBindlessStorageTextureBinding, MTLResourceUsageRead | MTLResourceUsageWrite);
  if (stage.uses_bindless_binding[kMetalBindlessAccelerationStructureBinding]) {
    for (uint32_t i = 0, e = static_cast<uint32_t>(bindless->acceleration_structure_entries.size()); i < e; ++i) {
      const auto& entry = bindless->acceleration_structure_entries[i];
      if (entry.valid == false) {
        continue;
      }
      RHIBindlessHandle handle = make_bindless_handle(RHIResourceType::AccelerationStructure, entry.generation, i);
      auto as_it = device->acceleration_structures.find(handle);
      if ((as_it != device->acceleration_structures.end()) && (as_it->second.acceleration_structure != nil)) {
        declare_resource((id<MTLResource>)as_it->second.acceleration_structure, MTLResourceUsageRead);
      }
    }
  }
}

static bool create_stage_resources(MTDevice::Impl* device, const MTBindlessManager::Impl* bindless, const RHIShaderDesc& shader_desc, MTPipelineStageData& out_stage, std::string& out_error) {
  if ((device == nullptr) || (device->metal_device == nil)) {
    out_error = "Metal device is unavailable.";
    return false;
  }

  if (shader_desc.metal_metadata.valid == false) {
    out_error = "Metal shader binding metadata is missing.";
    return false;
  }

  id<MTLLibrary> library = nil;
  if (shader_desc.format == RHIShaderBinaryFormat::MetalLibrary) {
    if ((shader_desc.spirv_data == nullptr) || (shader_desc.spirv_size == 0u)) {
      out_error = "Packaged Metal shader library is empty.";
      return false;
    }
    const uint64_t content_hash = (shader_desc.content_hash != 0u) ? shader_desc.content_hash : etx_hash64(shader_desc.spirv_data, static_cast<size_t>(shader_desc.spirv_size));
    const uint64_t cache_key = (shader_desc.cache_key != 0u) ? shader_desc.cache_key : content_hash;
    if (const auto iterator = device->library_cache.find(cache_key); (iterator != device->library_cache.end()) && (iterator->second.content_hash == content_hash)) {
      library = iterator->second.library;
    } else {
      dispatch_data_t library_data = dispatch_data_create(shader_desc.spirv_data, static_cast<size_t>(shader_desc.spirv_size), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
      NSError* library_error = nil;
      library = [device->metal_device newLibraryWithData:library_data error:&library_error];
      dispatch_release(library_data);
      if (library == nil) {
        out_error = library_error ? std::string([[library_error localizedDescription] UTF8String]) : "Failed to load packaged Metal shader library.";
        return false;
      }
      if (iterator != device->library_cache.end()) {
        [iterator->second.library release];
        iterator->second = {content_hash, library};
      } else {
        device->library_cache.emplace(cache_key, MTLibraryCacheEntry{content_hash, library});
      }
    }
  } else if (shader_desc.format == RHIShaderBinaryFormat::MetalSource) {
    NSString* source_text = make_nsstring(shader_desc.spirv_data, static_cast<size_t>(shader_desc.spirv_size));
    if (source_text == nil) {
      out_error = "Metal shader source is empty.";
      return false;
    }
    if (([source_text rangeOfString:@"ETX_METAL_UNSUPPORTED_OVERLAPPING_BINDLESS"].location != NSNotFound) || ([source_text rangeOfString:@"Overlapping binding:"].location != NSNotFound)) {
      out_error =
        "Metal shader translation produced overlapping bindless descriptor layouts. "
        "The current Metal RHI cannot safely encode mixed bindless resource classes yet.";
      return false;
    }
    const char* utf8_source = [source_text UTF8String];
    const std::string source_string = (utf8_source != nullptr) ? utf8_source : std::string();
    library = load_or_create_cached_metal_library(device, source_string, shader_desc.cache_key, shader_desc.content_hash, out_error);
  } else {
    out_error = "Metal shader binary format is unsupported.";
    return false;
  }
  if (library == nil) {
    return false;
  }

  NSString* function_name = [NSString stringWithUTF8String:shader_desc.entry_point.c_str()];
  id<MTLFunction> function = [library newFunctionWithName:function_name];
  if (function == nil) {
    out_error = "Failed to locate Metal function entry point: " + shader_desc.entry_point;
    return false;
  }

  out_stage.function = function;
  for (uint32_t binding_index = 0; binding_index < kMetalBindlessBindingCount; ++binding_index) {
    const uint32_t metal_buffer_index = shader_desc.metal_metadata.bindless_buffer_indices[binding_index];
    if (metal_buffer_index == kInvalidMetalBufferIndex) {
      continue;
    }

    out_stage.uses_bindless_binding[binding_index] = true;
    out_stage.bindless_buffer_indices[binding_index] = metal_buffer_index;
    out_stage.bindless_binding_access[binding_index] = to_metal_binding_access(shader_desc.metal_metadata.bindless_binding_access[binding_index]);

    id<MTLArgumentEncoder> encoder = create_bindless_argument_encoder(device->metal_device, bindless, binding_index, out_stage.bindless_binding_access[binding_index]);
    if (encoder == nil) {
      out_error = "Failed to create Metal argument encoder for bindless binding " + std::to_string(binding_index) + " at buffer slot " + std::to_string(metal_buffer_index) + ".";
      return false;
    }

    out_stage.bindless_argument_encoders[binding_index] = encoder;
    const NSUInteger encoded_length = std::max<NSUInteger>(encoder.encodedLength, 1u);
    out_stage.bindless_argument_buffers[binding_index] = [device->metal_device newBufferWithLength:encoded_length options:MTLResourceStorageModeShared];
    if (out_stage.bindless_argument_buffers[binding_index] == nil) {
      out_error = "Failed to allocate Metal bindless argument buffer for binding " + std::to_string(binding_index) + " at buffer slot " + std::to_string(metal_buffer_index) + ".";
      return false;
    }
  }

  out_stage.push_constants_buffer_index = shader_desc.metal_metadata.push_constants_buffer_index;
  out_stage.uses_push_constants = out_stage.push_constants_buffer_index != kInvalidMetalBufferIndex;

  return true;
}

static void release_stage_resources(MTPipelineStageData& stage) {
  for (id<MTLBuffer> argument_buffer : stage.bindless_argument_buffers) {
    [argument_buffer release];
  }
  for (id<MTLArgumentEncoder> argument_encoder : stage.bindless_argument_encoders) {
    [argument_encoder release];
  }
  [stage.function release];
  stage = {};
}

void create_metal_context(RHIContext& context, const RHIInitInfo& info) {
  (void)info;
  static_assert(sizeof(MTContext) <= RHIContext::kBackendStorageSize, "MTContext does not fit into RHIContext backend storage");
  static_assert(alignof(MTContext) <= RHIContext::kBackendStorageAlignment, "MTContext alignment exceeds RHIContext backend storage alignment");
  auto* mt_context = new (context._backend_storage) MTContext();
  if (mt_context->valid() == false) {
    mt_context->~MTContext();
    return;
  }
  context.initialize_backend(RHIBackend::Metal, mt_context, mt_context->get_device(), mt_context->get_bindless_manager());
}

MTContext::MTContext()
  : _impl(new Impl()) {
  _impl->metal_device = MTLCreateSystemDefaultDevice();
  if (_impl->metal_device == nil) {
    log::error("Metal RHI: failed to create default MTLDevice");
    return;
  }
  _impl->device._impl->metal_device = _impl->metal_device;

  _impl->command_queue = [_impl->metal_device newCommandQueue];
  if (_impl->command_queue == nil) {
    log::error("Metal RHI: failed to create MTLCommandQueue");
    return;
  }

  _impl->supports_ray_tracing = device_reports_raytracing(_impl->metal_device);
  if (@available(macOS 11.0, *)) {
    _impl->supports_timestamp_dispatch_sampling = [_impl->metal_device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
    const bool supports_timestamp_stage_sampling = [_impl->metal_device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
    if (_impl->supports_timestamp_dispatch_sampling || supports_timestamp_stage_sampling) {
      for (id<MTLCounterSet> counter_set in _impl->metal_device.counterSets) {
        if ([counter_set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
          _impl->timestamp_counter_set = [counter_set retain];
          _impl->supports_timestamps = true;
          break;
        }
      }
    }
  }
  _impl->device._impl->command_queue = _impl->command_queue;
  _impl->device._impl->bindless_manager = &_impl->bindless_manager;
  prune_obsolete_metal_cache_versions(metal_library_cache_root_directory());
  prune_obsolete_metal_cache_versions(metal_pipeline_archive_root_directory());

  _impl->image_acquired = _impl->device.create_semaphore().handle;
  _impl->render_complete = _impl->device.create_semaphore().handle;

  const std::array<RHISamplerDesc, static_cast<size_t>(RHISamplerType::Count)> default_sampler_descs = {
    RHISamplerDesc{.min_filter = RHISamplerFilter::Linear,
      .mag_filter = RHISamplerFilter::Linear,
      .mipmap_mode = RHISamplerMipmapMode::Linear,
      .address_mode_u = RHISamplerAddressMode::Repeat,
      .address_mode_v = RHISamplerAddressMode::Repeat,
      .address_mode_w = RHISamplerAddressMode::Repeat},
    RHISamplerDesc{.min_filter = RHISamplerFilter::Linear,
      .mag_filter = RHISamplerFilter::Linear,
      .mipmap_mode = RHISamplerMipmapMode::Linear,
      .address_mode_u = RHISamplerAddressMode::ClampToEdge,
      .address_mode_v = RHISamplerAddressMode::ClampToEdge,
      .address_mode_w = RHISamplerAddressMode::ClampToEdge},
    RHISamplerDesc{.min_filter = RHISamplerFilter::Nearest,
      .mag_filter = RHISamplerFilter::Nearest,
      .mipmap_mode = RHISamplerMipmapMode::Nearest,
      .address_mode_u = RHISamplerAddressMode::Repeat,
      .address_mode_v = RHISamplerAddressMode::Repeat,
      .address_mode_w = RHISamplerAddressMode::Repeat},
    RHISamplerDesc{.min_filter = RHISamplerFilter::Nearest,
      .mag_filter = RHISamplerFilter::Nearest,
      .mipmap_mode = RHISamplerMipmapMode::Nearest,
      .address_mode_u = RHISamplerAddressMode::ClampToEdge,
      .address_mode_v = RHISamplerAddressMode::ClampToEdge,
      .address_mode_w = RHISamplerAddressMode::ClampToEdge},
    RHISamplerDesc{.min_filter = RHISamplerFilter::Linear,
      .mag_filter = RHISamplerFilter::Linear,
      .mipmap_mode = RHISamplerMipmapMode::Linear,
      .address_mode_u = RHISamplerAddressMode::Repeat,
      .address_mode_v = RHISamplerAddressMode::ClampToEdge,
      .address_mode_w = RHISamplerAddressMode::ClampToEdge},
  };
  for (size_t i = 0; i < default_sampler_descs.size(); ++i) {
    const RHISamplerDesc& sampler_desc = default_sampler_descs[i];
    const auto result = _impl->device.create_sampler(sampler_desc);
    if (result.result != RHIResult::Success) {
      log::warning("Metal RHI: failed to create default sampler");
      continue;
    }
    _impl->predefined_sampler_indices[i] = get_bindless_descriptor_index(result.handle);
  }
}

MTContext::MTContext(MTContext&& other) noexcept
  : _impl(other._impl) {
  other._impl = nullptr;
}

MTContext::~MTContext() {
  if (_impl != nullptr) {
    wait_for_inflight_command_buffers(_impl->inflight_command_buffers, nullptr);
    for (auto& [handle, command_buffer] : _impl->command_buffers) {
      (void)handle;
      if (command_buffer) {
        command_buffer->reset();
      }
    }
    _impl->command_buffers.clear();
    for (auto& [handle, sample_buffer] : _impl->submitted_timestamp_sample_buffers) {
      (void)handle;
      [sample_buffer release];
    }
    _impl->submitted_timestamp_sample_buffers.clear();
    for (auto& [hash, entry] : _impl->device._impl->library_cache) {
      (void)hash;
      [entry.library release];
    }
    _impl->device._impl->library_cache.clear();
    [_impl->current_drawable release];
    _impl->current_drawable = nil;
    [_impl->metal_layer release];
    _impl->metal_layer = nil;
    [_impl->command_queue release];
    _impl->command_queue = nil;
    [_impl->timestamp_counter_set release];
    _impl->timestamp_counter_set = nil;
  }
  delete _impl;
}

bool MTContext::valid() const {
  return (_impl != nullptr) && (_impl->metal_device != nil) && (_impl->command_queue != nil);
}

MTDevice* MTContext::get_device() {
  return &_impl->device;
}

MTBindlessManager* MTContext::get_bindless_manager() {
  return &_impl->bindless_manager;
}

void MTContext::initialize_for_headless() {
  _impl->headless = true;
}

bool MTContext::has_swapchain() const {
  return _impl->metal_layer != nil;
}

void MTContext::create_swapchain(const void* native_window, uint32_t width, uint32_t height) {
  if (_impl->metal_layer != nil) {
    destroy_swapchain();
  }

  NSWindow* window = (__bridge NSWindow*)native_window;
  if (window == nil) {
    log::error("Metal RHI: create_swapchain received null NSWindow");
    return;
  }

  NSView* view = [window contentView];
  if (view == nil) {
    log::error("Metal RHI: failed to get content view for swapchain");
    return;
  }

  [view setWantsLayer:YES];
  CAMetalLayer* layer = nil;
  if ([[view layer] isKindOfClass:[CAMetalLayer class]]) {
    layer = (CAMetalLayer*)[view layer];
  } else {
    layer = [CAMetalLayer layer];
    [view setLayer:layer];
  }

  layer.device = _impl->metal_device;
  layer.pixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
  layer.framebufferOnly = NO;
  layer.drawableSize = CGSizeMake(width, height);
  layer.contentsScale = window.backingScaleFactor > 0.0 ? window.backingScaleFactor : 1.0;

  _impl->metal_layer = [layer retain];
  _impl->width = width;
  _impl->height = height;
}

void MTContext::destroy_swapchain() {
  wait_for_inflight_command_buffers(_impl->inflight_command_buffers, &_impl->polled_completion_results);
  if (_impl->swapchain_texture.valid()) {
    _impl->bindless_manager.unregister_texture(_impl->swapchain_texture);
    auto it = _impl->device._impl->textures.find(_impl->swapchain_texture);
    if (it != _impl->device._impl->textures.end()) {
      _impl->device._impl->textures.erase(it);
    }
    _impl->swapchain_texture = {};
  }
  [_impl->current_drawable release];
  _impl->current_drawable = nil;
  [_impl->metal_layer release];
  _impl->metal_layer = nil;
}

void MTContext::resize_swapchain(uint32_t width, uint32_t height) {
  _impl->width = width;
  _impl->height = height;
  if (_impl->metal_layer != nil) {
    _impl->metal_layer.drawableSize = CGSizeMake(width, height);
  }
}

RHITexture MTContext::get_current_swapchain_texture() {
  if (_impl->metal_layer == nil) {
    return {};
  }

  if (_impl->current_drawable == nil) {
    _impl->current_drawable = [[_impl->metal_layer nextDrawable] retain];
  }
  if (_impl->current_drawable == nil) {
    return {};
  }

  id<MTLTexture> drawable_texture = _impl->current_drawable.texture;
  if (_impl->swapchain_texture.valid() == false) {
    RHIBindlessHandle handle = {};
    const auto reg_result = _impl->bindless_manager.register_texture((__bridge void*)drawable_texture, RHIResourceType::Texture, handle,
      static_cast<uint32_t>(RHITextureUsage::ColorAttachment) | static_cast<uint32_t>(RHITextureUsage::TransferSrc) | static_cast<uint32_t>(RHITextureUsage::TransferDst));
    if (reg_result != RHIResult::Success) {
      log::error("Metal RHI: failed to register swapchain texture");
      return {};
    }
    _impl->swapchain_texture = handle;
    _impl->device._impl->textures[handle] = {
      .texture = drawable_texture,
      .desc = {.width = _impl->width, .height = _impl->height, .format = get_swapchain_format(), .usage = RHITextureUsage::ColorAttachment},
      .state = RHIResourceState::Present,
      .is_swapchain_texture = true,
    };
  } else {
    auto& entry = _impl->bindless_manager._impl->texture_entries[get_bindless_descriptor_index(_impl->swapchain_texture)];
    entry.resource = (__bridge void*)drawable_texture;
    auto tex_it = _impl->device._impl->textures.find(_impl->swapchain_texture);
    if (tex_it != _impl->device._impl->textures.end()) {
      tex_it->second.texture = drawable_texture;
      tex_it->second.desc.width = _impl->width;
      tex_it->second.desc.height = _impl->height;
    }
  }

  return _impl->swapchain_texture;
}

RHITextureFormat MTContext::get_swapchain_format() const {
  return RHITextureFormat::B8G8R8A8_SRGB;
}

RHIExtent2D MTContext::get_swapchain_extent_rhi() const {
  return {.width = _impl->width, .height = _impl->height};
}

void MTContext::present() {
  [_impl->current_drawable release];
  _impl->current_drawable = nil;
}

RHIResult MTContext::wait_idle() {
  return wait_for_inflight_command_buffers(_impl->inflight_command_buffers, &_impl->polled_completion_results) ? RHIResult::Success : RHIResult::DeviceLost;
}

void MTContext::begin_frame() {
  reap_completed_command_buffers(_impl->inflight_command_buffers);
  while (_impl->inflight_command_buffers.size() >= kRHIMaxFrames) {
    auto oldest_it = std::find_if(_impl->inflight_command_buffers.begin(), _impl->inflight_command_buffers.end(), [](const MTInflightSubmission& submission) {
      return submission.completion_polled == false;
    });
    if (oldest_it == _impl->inflight_command_buffers.end()) {
      break;
    }
    MTInflightSubmission oldest = *oldest_it;
    id<MTLCommandBuffer> oldest_command_buffer = oldest.command_buffer;
    id<CAMetalDrawable> oldest_drawable = oldest.drawable;
    if (oldest_command_buffer != nil) {
      [oldest_command_buffer waitUntilCompleted];
      [oldest_command_buffer release];
    }
    if (oldest_drawable != nil) {
      [oldest_drawable release];
    }
    _impl->inflight_command_buffers.erase(oldest_it);
    reap_completed_command_buffers(_impl->inflight_command_buffers);
  }
  _impl->current_frame = (_impl->current_frame + 1u) % kRHIMaxFrames;
}

void MTContext::end_frame() {
}

uint32_t MTContext::get_current_frame_index() const {
  return _impl->current_frame;
}

uint32_t MTContext::get_sampler_index(RHISamplerType type) const {
  const size_t index = static_cast<size_t>(type);
  if ((_impl != nullptr) && (index < _impl->predefined_sampler_indices.size())) {
    const uint32_t descriptor_index = _impl->predefined_sampler_indices[index];
    if (descriptor_index != kRHIBindlessDescriptorIndexMask) {
      return descriptor_index;
    }
  }
  return static_cast<uint32_t>(type);
}

RHICapabilities MTContext::capabilities() const {
  return {
    .supports_swapchain = has_swapchain() || _impl->headless,
    .supports_bindless = true,
    .supports_timestamps = supports_timestamps(),
    .supports_ray_tracing = _impl->supports_ray_tracing,
  };
}

RHICommandBuffer MTContext::get_command_buffer() {
  const uint32_t command_buffer_index = _impl->next_command_buffer_index++;
  const RHICommandBuffer handle = Handle::construct(0u, command_buffer_index, 1u);
  auto command_buffer = std::make_unique<MTCommandBuffer>();
  command_buffer->_impl->owner = _impl;
  _impl->command_buffers.emplace(handle, std::move(command_buffer));
  return handle;
}

RHICommandBuffer MTContext::get_async_command_buffer() {
  return get_command_buffer();
}

void MTContext::destroy_command_buffer(RHICommandBuffer cmd) {
  _impl->polled_completion_results.erase(cmd);
  auto timestamp_it = _impl->submitted_timestamp_sample_buffers.find(cmd);
  if (timestamp_it != _impl->submitted_timestamp_sample_buffers.end()) {
    [timestamp_it->second release];
    _impl->submitted_timestamp_sample_buffers.erase(timestamp_it);
  }
  auto it = _impl->command_buffers.find(cmd);
  if (it == _impl->command_buffers.end()) {
    return;
  }

  if (it->second) {
    it->second->reset();
  }
  _impl->command_buffers.erase(it);
}

RHIResult MTContext::wait_for_command_buffer(RHICommandBuffer cmd) {
  (void)cmd;
  return wait_idle();
}

RHIResult MTContext::query_command_buffer(RHICommandBuffer cmd) {
  auto completed_it = _impl->polled_completion_results.find(cmd);
  if (completed_it != _impl->polled_completion_results.end()) {
    const RHIResult result = completed_it->second;
    _impl->polled_completion_results.erase(completed_it);
    return result;
  }

  for (size_t submission_index = 0u; submission_index < _impl->inflight_command_buffers.size(); ++submission_index) {
    MTInflightSubmission& submission = _impl->inflight_command_buffers[submission_index];
    if (submission.handle != cmd) {
      continue;
    }

    submission.completion_polled = true;
    const MTLCommandBufferStatus status = submission.command_buffer.status;
    if ((status != MTLCommandBufferStatusCompleted) && (status != MTLCommandBufferStatusError)) {
      return RHIResult::NotReady;
    }

    const RHIResult result = (status == MTLCommandBufferStatusCompleted) ? RHIResult::Success : RHIResult::ValidationError;
    [submission.command_buffer release];
    if (submission.drawable != nil) {
      [submission.drawable release];
    }
    _impl->inflight_command_buffers.erase(_impl->inflight_command_buffers.begin() + submission_index);
    return result;
  }
  return RHIResult::InvalidHandle;
}

void MTContext::submit_command_buffer(const RHISubmitInfo& info) {
  (void)info.wait_semaphores;
  (void)info.signal_semaphores;
  MTCommandBuffer* command_buffer = _impl->find_command_buffer(info.command_buffer);
  if (command_buffer == nullptr) {
    return;
  }

  id<MTLCommandBuffer> submitted_command_buffer = command_buffer->_impl->command_buffer;
  if (submitted_command_buffer == nil) {
    return;
  }

  MTInflightSubmission inflight_submission = {
    .handle = info.command_buffer,
    .command_buffer = submitted_command_buffer,
    .drawable = nil,
  };
  if (_impl->current_drawable != nil) {
    [submitted_command_buffer presentDrawable:_impl->current_drawable];
    inflight_submission.drawable = _impl->current_drawable;
    _impl->current_drawable = nil;
  }
  [submitted_command_buffer commit];
  [submitted_command_buffer retain];
  if (command_buffer->_impl->timestamp_sample_buffer != nil) {
    auto existing_it = _impl->submitted_timestamp_sample_buffers.find(info.command_buffer);
    if (existing_it != _impl->submitted_timestamp_sample_buffers.end()) {
      [existing_it->second release];
      existing_it->second = command_buffer->_impl->timestamp_sample_buffer;
    } else {
      _impl->submitted_timestamp_sample_buffers.emplace(info.command_buffer, command_buffer->_impl->timestamp_sample_buffer);
    }
    command_buffer->_impl->timestamp_sample_buffer = nil;
  }
  reap_completed_command_buffers(_impl->inflight_command_buffers);
  _impl->inflight_command_buffers.push_back(inflight_submission);
  command_buffer->detach_submitted();
  _impl->command_buffers.erase(info.command_buffer);
}

void MTContext::program_command_buffer(RHICommandBuffer cmd, std::function<void(void)> func) {
  if (_impl->find_command_buffer(cmd) && func) {
    func();
  }
}

void MTContext::command_buffer_begin(RHICommandBuffer cmd) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->begin();
  }
}

void MTContext::command_buffer_end(RHICommandBuffer cmd) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->end();
  }
}

void MTContext::command_buffer_reset(RHICommandBuffer cmd) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->reset();
  }
}

void MTContext::cmd_compute_barrier(RHICommandBuffer cmd) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->compute_barrier();
  }
}

void MTContext::cmd_buffer_barrier(RHICommandBuffer cmd, RHIBindlessHandle buffer, RHIResourceState old_state, RHIResourceState new_state) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->buffer_barrier(buffer, old_state, new_state);
  }
}

void MTContext::cmd_texture_barrier(RHICommandBuffer cmd, RHIBindlessHandle texture, RHIResourceState old_state, RHIResourceState new_state) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->texture_barrier(texture, old_state, new_state);
  }
}

void MTContext::cmd_begin_render_pass(RHICommandBuffer cmd, uint32_t color_attachment_count, RHIBindlessHandle* color_attachments, const float* clear_colors, RHIBindlessHandle depth_attachment, const RHIResourceState* color_final_states,
  RHIResourceState depth_final_state) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->begin_render_pass(color_attachment_count, color_attachments, clear_colors, depth_attachment, color_final_states, depth_final_state);
  }
}

void MTContext::cmd_end_render_pass(RHICommandBuffer cmd) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->end_render_pass();
  }
}

void MTContext::cmd_set_viewport(RHICommandBuffer cmd, const RHIViewport& viewport) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->set_viewport(viewport);
  }
}

void MTContext::cmd_set_scissor(RHICommandBuffer cmd, const RHIRect& scissor) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->set_scissor(scissor);
  }
}

void MTContext::cmd_set_pipeline(RHICommandBuffer cmd, RHIPipeline pipeline) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->set_pipeline(pipeline);
  }
}

void MTContext::cmd_push_constants(RHICommandBuffer cmd, const void* data, uint32_t size, uint32_t offset) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->push_constants(data, size, offset);
  }
}

void MTContext::cmd_draw(RHICommandBuffer cmd, const RHIDrawDesc& desc) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->draw(desc);
  }
}

void MTContext::cmd_draw_indexed(RHICommandBuffer cmd, const RHIIndexedDrawDesc& desc, RHIBindlessHandle index_buffer) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->draw_indexed(desc, index_buffer);
  }
}

void MTContext::cmd_dispatch(RHICommandBuffer cmd, const RHIDispatchDesc& desc) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->dispatch(desc);
  }
}

void MTContext::cmd_dispatch_indirect(RHICommandBuffer cmd, RHIBindlessHandle argument_buffer, uint64_t argument_buffer_offset) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->dispatch_indirect(argument_buffer, argument_buffer_offset);
  }
}

void MTContext::cmd_reset_timestamps(RHICommandBuffer cmd, uint32_t first_query, uint32_t query_count) {
  if (supports_timestamps() == false) {
    return;
  }
  if ((query_count == 0u) || (first_query >= kMetalTimestampQueryCount) || (query_count > (kMetalTimestampQueryCount - first_query))) {
    log::error("Metal RHI: invalid timestamp query range: first=%u count=%u (max=%u)", first_query, query_count, kMetalTimestampQueryCount);
    return;
  }

  MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd);
  if ((command_buffer == nullptr) || (command_buffer->_impl->command_buffer == nil)) {
    log::error("Metal RHI: cannot reset timestamps for an invalid command buffer");
    return;
  }
  ensure_timestamp_sample_buffer(command_buffer->_impl, _impl->metal_device, _impl->timestamp_counter_set);
}

void MTContext::cmd_write_timestamp(RHICommandBuffer cmd, uint32_t query_index, RHITimestampStage stage) {
  (void)stage;
  if (supports_timestamps() == false) {
    return;
  }
  if (query_index >= kMetalTimestampQueryCount) {
    log::error("Metal RHI: invalid timestamp query index %u (max=%u)", query_index, kMetalTimestampQueryCount);
    return;
  }

  MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd);
  if ((command_buffer == nullptr) || (ensure_timestamp_sample_buffer(command_buffer->_impl, _impl->metal_device, _impl->timestamp_counter_set) == false)) {
    return;
  }
  if (_impl->supports_timestamp_dispatch_sampling == false) {
    return;
  }

  id<MTLComputeCommandEncoder> encoder = ensure_compute_encoder(command_buffer->_impl);
  if (encoder != nil) {
    [encoder sampleCountersInBuffer:command_buffer->_impl->timestamp_sample_buffer atSampleIndex:query_index withBarrier:YES];
  }
}

void MTContext::cmd_begin_timestamp_scope(RHICommandBuffer cmd, uint32_t begin_query_index, uint32_t end_query_index, RHITimestampStage stage) {
  (void)stage;
  if ((begin_query_index >= kMetalTimestampQueryCount) || (end_query_index >= kMetalTimestampQueryCount)) {
    log::error("Metal RHI: invalid timestamp scope: begin=%u end=%u (max=%u)", begin_query_index, end_query_index, kMetalTimestampQueryCount);
    return;
  }

  MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd);
  if ((command_buffer == nullptr) || (ensure_timestamp_sample_buffer(command_buffer->_impl, _impl->metal_device, _impl->timestamp_counter_set) == false)) {
    return;
  }
  if (command_buffer->_impl->timestamp_scope_active) {
    log::error("Metal RHI: nested timestamp scopes are unsupported");
    return;
  }

  command_buffer->_impl->timestamp_scope_begin_query = begin_query_index;
  command_buffer->_impl->timestamp_scope_end_query = end_query_index;
  command_buffer->_impl->timestamp_scope_active = true;
  command_buffer->_impl->timestamp_scope_uses_stage_sampling = _impl->supports_timestamp_dispatch_sampling == false;
  if (_impl->supports_timestamp_dispatch_sampling) {
    id<MTLComputeCommandEncoder> encoder = ensure_compute_encoder(command_buffer->_impl);
    if (encoder != nil) {
      [encoder sampleCountersInBuffer:command_buffer->_impl->timestamp_sample_buffer atSampleIndex:begin_query_index withBarrier:YES];
    }
    return;
  }

  [command_buffer->_impl->render_encoder endEncoding];
  command_buffer->_impl->render_encoder = nil;
  [command_buffer->_impl->compute_encoder endEncoding];
  command_buffer->_impl->compute_encoder = nil;
  [command_buffer->_impl->blit_encoder endEncoding];
  command_buffer->_impl->blit_encoder = nil;
}

void MTContext::cmd_end_timestamp_scope(RHICommandBuffer cmd, uint32_t end_query_index, RHITimestampStage stage) {
  (void)stage;
  MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd);
  if ((command_buffer == nullptr) || (command_buffer->_impl->timestamp_scope_active == false)) {
    return;
  }
  if (command_buffer->_impl->timestamp_scope_end_query != end_query_index) {
    log::error("Metal RHI: timestamp scope ended with query %u, expected %u", end_query_index, command_buffer->_impl->timestamp_scope_end_query);
  }

  if (_impl->supports_timestamp_dispatch_sampling) {
    id<MTLComputeCommandEncoder> encoder = ensure_compute_encoder(command_buffer->_impl);
    if (encoder != nil) {
      [encoder sampleCountersInBuffer:command_buffer->_impl->timestamp_sample_buffer atSampleIndex:end_query_index withBarrier:YES];
    }
  } else {
    [command_buffer->_impl->compute_encoder endEncoding];
    command_buffer->_impl->compute_encoder = nil;
  }
  command_buffer->_impl->timestamp_scope_begin_query = ~0u;
  command_buffer->_impl->timestamp_scope_end_query = ~0u;
  command_buffer->_impl->timestamp_scope_active = false;
  command_buffer->_impl->timestamp_scope_uses_stage_sampling = false;
}

void MTContext::cmd_build_acceleration_structure(RHICommandBuffer cmd, const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->build_acceleration_structure(desc, scratch_buffer, scratch_offset);
  }
}

void MTContext::cmd_copy_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint64_t size, uint64_t src_offset, uint64_t dst_offset) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->copy_buffer(src, dst, size, src_offset, dst_offset);
  }
}

void MTContext::cmd_copy_buffer_to_texture(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->copy_buffer_to_texture(src, dst, width, height, mip_level);
  }
}

void MTContext::cmd_copy_texture_to_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->copy_texture_to_buffer(src, dst, width, height, mip_level);
  }
}

void MTContext::cmd_resolve_texture(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height) {
  (void)width;
  (void)height;
  MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd);
  if (command_buffer == nullptr) {
    return;
  }
  auto src_it = _impl->device._impl->textures.find(src);
  auto dst_it = _impl->device._impl->textures.find(dst);
  if ((src_it == _impl->device._impl->textures.end()) || (dst_it == _impl->device._impl->textures.end())) {
    return;
  }

  [command_buffer->_impl->render_encoder endEncoding];
  command_buffer->_impl->render_encoder = nil;
  [command_buffer->_impl->compute_encoder endEncoding];
  command_buffer->_impl->compute_encoder = nil;
  [command_buffer->_impl->blit_encoder endEncoding];
  command_buffer->_impl->blit_encoder = nil;
  if (command_buffer->_impl->command_buffer == nil) {
    return;
  }

  MTLRenderPassDescriptor* pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];
  pass_desc.colorAttachments[0].texture = src_it->second.texture;
  pass_desc.colorAttachments[0].resolveTexture = dst_it->second.texture;
  pass_desc.colorAttachments[0].loadAction = MTLLoadActionLoad;
  pass_desc.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
  id<MTLRenderCommandEncoder> encoder = [command_buffer->_impl->command_buffer renderCommandEncoderWithDescriptor:pass_desc];
  encoder.label = @"ETX resolve texture";
  [encoder endEncoding];
}

void MTContext::cmd_generate_mipmaps(RHICommandBuffer cmd, RHIBindlessHandle texture) {
  MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd);
  if (command_buffer == nullptr) {
    return;
  }
  auto it = _impl->device._impl->textures.find(texture);
  if (it == _impl->device._impl->textures.end()) {
    return;
  }
  if (command_buffer->_impl->blit_encoder == nil) {
    [command_buffer->_impl->render_encoder endEncoding];
    command_buffer->_impl->render_encoder = nil;
    [command_buffer->_impl->compute_encoder endEncoding];
    command_buffer->_impl->compute_encoder = nil;
    command_buffer->_impl->blit_encoder = [command_buffer->_impl->command_buffer blitCommandEncoder];
    command_buffer->_impl->blit_encoder.label = @"ETX generate mipmaps";
  }
  [command_buffer->_impl->blit_encoder generateMipmapsForTexture:it->second.texture];
}

void MTContext::cmd_set_debug_name(RHICommandBuffer cmd, const char* name) {
  if (MTCommandBuffer* command_buffer = _impl->find_command_buffer(cmd)) {
    command_buffer->set_debug_name(name);
  }
}

bool MTContext::supports_timestamps() const {
  return (_impl != nullptr) && _impl->supports_timestamps;
}

uint32_t MTContext::timestamp_query_capacity() const {
  return supports_timestamps() ? kMetalTimestampQueryCount : 0u;
}

double MTContext::timestamp_period_ns() const {
  return supports_timestamps() ? 1.0 : 0.0;
}

RHIResult MTContext::read_timestamps(RHICommandBuffer cmd, uint32_t first_query, uint32_t query_count, uint64_t* out_values) {
  if (out_values == nullptr) {
    return RHIResult::InvalidArgument;
  }
  if ((query_count == 0u) || (first_query >= kMetalTimestampQueryCount) || (query_count > (kMetalTimestampQueryCount - first_query))) {
    return RHIResult::InvalidArgument;
  }
  if (supports_timestamps() == false) {
    return RHIResult::UnsupportedFeature;
  }

  const auto sample_buffer_it = _impl->submitted_timestamp_sample_buffers.find(cmd);
  if (sample_buffer_it == _impl->submitted_timestamp_sample_buffers.end()) {
    return RHIResult::InvalidHandle;
  }
  NSData* resolved_data = [sample_buffer_it->second resolveCounterRange:NSMakeRange(first_query, query_count)];
  const size_t required_size = static_cast<size_t>(query_count) * sizeof(MTLCounterResultTimestamp);
  if ((resolved_data == nil) || (resolved_data.length < required_size)) {
    return RHIResult::NotReady;
  }

  const auto* timestamp_results = static_cast<const MTLCounterResultTimestamp*>(resolved_data.bytes);
  for (uint32_t query_offset = 0u; query_offset < query_count; ++query_offset) {
    const uint64_t timestamp = timestamp_results[query_offset].timestamp;
    if (timestamp == MTLCounterErrorValue) {
      return RHIResult::ValidationError;
    }
    out_values[query_offset] = timestamp;
  }
  return RHIResult::Success;
}

RHISemaphore MTContext::get_image_acquired_semaphore() {
  return _impl->image_acquired;
}

RHISemaphore MTContext::get_render_complete_semaphore() {
  return _impl->render_complete;
}

MTDevice::MTDevice()
  : _impl(new Impl()) {
}

MTDevice::~MTDevice() {
  for (auto& [handle, buffer] : _impl->buffers) {
    [buffer.buffer release];
  }
  for (auto& [handle, texture] : _impl->textures) {
    if (texture.is_swapchain_texture == false) {
      [texture.texture release];
    }
  }
  for (auto& [handle, sampler] : _impl->samplers) {
    [sampler.sampler release];
  }
  for (auto& [handle, acceleration_structure] : _impl->acceleration_structures) {
    [acceleration_structure.instance_descriptor_buffer release];
    [acceleration_structure.acceleration_structure release];
  }
  for (auto& [handle, pipeline] : _impl->pipelines) {
    release_stage_resources(pipeline.vertex_stage);
    release_stage_resources(pipeline.fragment_stage);
    release_stage_resources(pipeline.compute_stage);
    [pipeline.depth_state release];
    [pipeline.render_pipeline release];
    [pipeline.compute_pipeline release];
  }
  [_impl->metal_device release];
  delete _impl;
}

RHICreateResult<RHISemaphore> MTDevice::create_semaphore() {
  const RHISemaphore handle = Handle::construct(0u, _impl->next_semaphore_index++, 1u);
  _impl->semaphores[handle] = {};
  return {RHIResult::Success, handle};
}

RHIResult MTDevice::destroy_semaphore(RHISemaphore semaphore) {
  if (semaphore.invalid()) {
    return RHIResult::Success;
  }
  _impl->semaphores.erase(semaphore);
  return RHIResult::Success;
}

RHICreateBindlessResult MTDevice::create_buffer(const RHIBufferDesc& desc) {
  if (_impl->metal_device == nil || (_impl->bindless_manager == nullptr)) {
    return {RHIResult::InvalidArgument, {}};
  }

  const NSUInteger length = static_cast<NSUInteger>(std::max<uint64_t>(desc.size, 4u));
  id<MTLBuffer> buffer = [_impl->metal_device newBufferWithLength:length options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return {RHIResult::OutOfMemory, {}};
  }

  RHIBindlessHandle handle = {};
  const RHIResult reg_result = _impl->bindless_manager->register_buffer((__bridge void*)buffer, RHIResourceType::Buffer, handle);
  if (reg_result != RHIResult::Success) {
    [buffer release];
    return {reg_result, {}};
  }

  _impl->buffers.emplace(handle, MTBufferData{.buffer = buffer, .desc = desc, .state = RHIResourceState::Undefined});
  _impl->gpu_allocated_bytes += length;
  return {RHIResult::Success, handle};
}

RHICreateBindlessResult MTDevice::create_texture(const RHITextureDesc& desc) {
  if (_impl->metal_device == nil || (_impl->bindless_manager == nullptr)) {
    return {RHIResult::InvalidArgument, {}};
  }

  const MTLPixelFormat pixel_format = to_metal_format(desc.format);
  if (pixel_format == MTLPixelFormatInvalid) {
    return {RHIResult::InvalidArgument, {}};
  }

  MTLTextureDescriptor* descriptor = nil;
  if (desc.sample_count > 1u) {
    descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format width:desc.width height:desc.height mipmapped:NO];
    descriptor.textureType = MTLTextureType2DMultisample;
    descriptor.sampleCount = desc.sample_count;
  } else {
    descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format width:desc.width height:desc.height mipmapped:(desc.mip_levels > 1u)];
  }
  descriptor.arrayLength = desc.array_layers;
  descriptor.mipmapLevelCount = std::max<uint32_t>(desc.mip_levels, 1u);
  descriptor.usage = to_metal_texture_usage(desc.usage);
  descriptor.storageMode = texture_prefers_private_storage(desc) ? MTLStorageModePrivate : MTLStorageModeShared;

  id<MTLTexture> texture = [_impl->metal_device newTextureWithDescriptor:descriptor];
  if (texture == nil) {
    return {RHIResult::OutOfMemory, {}};
  }

  RHIBindlessHandle handle = {};
  const RHIResult reg_result = _impl->bindless_manager->register_texture((__bridge void*)texture, RHIResourceType::Texture, handle, static_cast<uint32_t>(desc.usage), (__bridge void*)texture);
  if (reg_result != RHIResult::Success) {
    [texture release];
    return {reg_result, {}};
  }

  _impl->textures.emplace(handle, MTTextureData{.texture = texture, .desc = desc, .state = RHIResourceState::Undefined});
  return {RHIResult::Success, handle};
}

RHICreateBindlessResult MTDevice::create_sampler(const RHISamplerDesc& desc) {
  if (_impl->metal_device == nil || (_impl->bindless_manager == nullptr)) {
    return {RHIResult::InvalidArgument, {}};
  }

  MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
  descriptor.minFilter = to_metal_filter(desc.min_filter);
  descriptor.magFilter = to_metal_filter(desc.mag_filter);
  descriptor.mipFilter = to_metal_mip_filter(desc.mipmap_mode);
  descriptor.sAddressMode = to_metal_address_mode(desc.address_mode_u);
  descriptor.tAddressMode = to_metal_address_mode(desc.address_mode_v);
  descriptor.rAddressMode = to_metal_address_mode(desc.address_mode_w);
  descriptor.maxAnisotropy = std::max<float>(1.0f, desc.max_anisotropy);
  if (@available(macOS 11.0, *)) {
    descriptor.supportArgumentBuffers = YES;
  }

  id<MTLSamplerState> sampler = [_impl->metal_device newSamplerStateWithDescriptor:descriptor];
  [descriptor release];
  if (sampler == nil) {
    return {RHIResult::OutOfMemory, {}};
  }

  RHIBindlessHandle handle = {};
  const RHIResult reg_result = _impl->bindless_manager->register_sampler((__bridge void*)sampler, RHIResourceType::Sampler, handle);
  if (reg_result != RHIResult::Success) {
    [sampler release];
    return {reg_result, {}};
  }

  _impl->samplers.emplace(handle, MTSamplerData{.sampler = sampler, .desc = desc});
  return {RHIResult::Success, handle};
}

static id<MTLBinaryArchive> create_metal_binary_archive(id<MTLDevice> device, const std::filesystem::path* path);
static bool serialize_metal_binary_archive(id<MTLBinaryArchive> archive, const std::filesystem::path& path);

static uint64_t metal_graphics_pipeline_hash(const RHIGraphicsPipelineDesc& desc, bool content_hash) {
  uint64_t hash = etx_hash64("graphics", 8u);
  auto append = [&](const auto& value) {
    hash = etx_hash64_continue(&value, sizeof(value), hash);
  };
  const uint64_t vertex_content_hash = (desc.vertex_shader.content_hash != 0u) ? desc.vertex_shader.content_hash : etx_hash64(desc.vertex_shader.spirv_data, static_cast<size_t>(desc.vertex_shader.spirv_size));
  const uint64_t fragment_content_hash = (desc.fragment_shader.content_hash != 0u) ? desc.fragment_shader.content_hash : etx_hash64(desc.fragment_shader.spirv_data, static_cast<size_t>(desc.fragment_shader.spirv_size));
  const uint64_t vertex_shader_hash = content_hash ? vertex_content_hash : ((desc.vertex_shader.cache_key != 0u) ? desc.vertex_shader.cache_key : vertex_content_hash);
  const uint64_t fragment_shader_hash = content_hash ? fragment_content_hash : ((desc.fragment_shader.cache_key != 0u) ? desc.fragment_shader.cache_key : fragment_content_hash);
  append(vertex_shader_hash);
  append(fragment_shader_hash);
  hash = etx_hash64_continue(desc.vertex_shader.entry_point.data(), desc.vertex_shader.entry_point.size(), hash);
  hash = etx_hash64_continue(desc.fragment_shader.entry_point.data(), desc.fragment_shader.entry_point.size(), hash);
  append(desc.sample_count);
  append(desc.depth_format);
  append(desc.color_attachment_count);
  for (uint32_t i = 0u; i < desc.color_attachment_count; ++i) {
    append(desc.color_formats[i]);
  }
  append(desc.blend.src_color_blend_factor);
  append(desc.blend.dst_color_blend_factor);
  append(desc.blend.color_blend_op);
  append(desc.blend.src_alpha_blend_factor);
  append(desc.blend.dst_alpha_blend_factor);
  append(desc.blend.alpha_blend_op);
  append(desc.blend.blend_enable);
  append(desc.vertex_attribute_count);
  for (uint32_t i = 0u; i < desc.vertex_attribute_count; ++i) {
    append(desc.vertex_attributes[i].location);
    append(desc.vertex_attributes[i].binding);
    append(desc.vertex_attributes[i].format);
    append(desc.vertex_attributes[i].offset);
  }
  append(desc.vertex_binding_count);
  for (uint32_t i = 0u; i < desc.vertex_binding_count; ++i) {
    append(desc.vertex_bindings[i].binding);
    append(desc.vertex_bindings[i].stride);
    append(desc.vertex_bindings[i].input_rate);
  }
  return hash;
}

RHICreatePipelineResult MTDevice::create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) {
  if (_impl->metal_device == nil) {
    return {RHIResult::InvalidArgument, {}};
  }
  const bool vertex_format_supported = (desc.vertex_shader.format == RHIShaderBinaryFormat::MetalSource) || (desc.vertex_shader.format == RHIShaderBinaryFormat::MetalLibrary);
  const bool fragment_format_supported = (desc.fragment_shader.format == RHIShaderBinaryFormat::MetalSource) || (desc.fragment_shader.format == RHIShaderBinaryFormat::MetalLibrary);
  if ((desc.vertex_shader.backend != RHIBackend::Metal) || (desc.fragment_shader.backend != RHIBackend::Metal) || (vertex_format_supported == false) || (fragment_format_supported == false)) {
    return {RHIResult::InvalidArgument, {}};
  }

  std::string error_message = {};
  MTPipelineData pipeline = {};
  pipeline.is_compute = false;
  pipeline.primitive = to_metal_primitive(desc.primitive_topology);
  pipeline.fill_mode = desc.rasterization.wireframe_enable ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
  pipeline.debug_name = desc.vertex_shader.entry_point + " + " + desc.fragment_shader.entry_point;
  pipeline.graphics_desc = desc;

  if (!create_stage_resources(_impl, _impl->bindless_manager->_impl, desc.vertex_shader, pipeline.vertex_stage, error_message) ||
      !create_stage_resources(_impl, _impl->bindless_manager->_impl, desc.fragment_shader, pipeline.fragment_stage, error_message)) {
    log::error("Metal RHI: graphics shader stage creation failed: %s", error_message.c_str());
    release_stage_resources(pipeline.vertex_stage);
    release_stage_resources(pipeline.fragment_stage);
    return {RHIResult::ValidationError, {}};
  }

  MTLRenderPipelineDescriptor* pipeline_desc = [[MTLRenderPipelineDescriptor alloc] init];
  pipeline_desc.vertexFunction = pipeline.vertex_stage.function;
  pipeline_desc.fragmentFunction = pipeline.fragment_stage.function;
  pipeline_desc.sampleCount = std::max<uint32_t>(desc.sample_count, 1u);
  pipeline_desc.depthAttachmentPixelFormat = to_metal_format(desc.depth_format);

  for (uint32_t i = 0; i < desc.color_attachment_count; ++i) {
    auto* color_attachment = pipeline_desc.colorAttachments[i];
    color_attachment.pixelFormat = to_metal_format(desc.color_formats[i]);
    color_attachment.blendingEnabled = desc.blend.blend_enable;
    color_attachment.sourceRGBBlendFactor = to_metal_blend_factor(desc.blend.src_color_blend_factor);
    color_attachment.destinationRGBBlendFactor = to_metal_blend_factor(desc.blend.dst_color_blend_factor);
    color_attachment.rgbBlendOperation = to_metal_blend_op(desc.blend.color_blend_op);
    color_attachment.sourceAlphaBlendFactor = to_metal_blend_factor(desc.blend.src_alpha_blend_factor);
    color_attachment.destinationAlphaBlendFactor = to_metal_blend_factor(desc.blend.dst_alpha_blend_factor);
    color_attachment.alphaBlendOperation = to_metal_blend_op(desc.blend.alpha_blend_op);
    color_attachment.writeMask = MTLColorWriteMaskAll;
  }

  if (desc.vertex_attribute_count > 0u) {
    MTLVertexDescriptor* vertex_desc = [MTLVertexDescriptor vertexDescriptor];
    for (uint32_t i = 0; i < desc.vertex_attribute_count; ++i) {
      const auto& attribute = desc.vertex_attributes[i];
      vertex_desc.attributes[attribute.location].format = to_metal_vertex_format(attribute.format);
      vertex_desc.attributes[attribute.location].offset = attribute.offset;
      vertex_desc.attributes[attribute.location].bufferIndex = attribute.binding;
    }
    for (uint32_t i = 0; i < desc.vertex_binding_count; ++i) {
      const auto& binding = desc.vertex_bindings[i];
      vertex_desc.layouts[binding.binding].stride = binding.stride;
      vertex_desc.layouts[binding.binding].stepRate = 1;
      vertex_desc.layouts[binding.binding].stepFunction = (binding.input_rate == RHIVertexInputRate::Instance) ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
    }
    pipeline_desc.vertexDescriptor = vertex_desc;
  }

  NSError* pipeline_error = nil;
  if (@available(macOS 11.0, *)) {
    const uint64_t cache_key = metal_graphics_pipeline_hash(desc, false);
    const uint64_t content_hash = metal_graphics_pipeline_hash(desc, true);
    const std::filesystem::path archive_path = metal_pipeline_archive_path(_impl->metal_device, cache_key, content_hash);
    const std::string slot_prefix = format_hash_hex(cache_key) + "_";
    std::error_code ec = {};
    const bool archive_exists = std::filesystem::exists(archive_path, ec) && (ec.value() == 0);
    if (archive_exists) {
      id<MTLBinaryArchive> archive = create_metal_binary_archive(_impl->metal_device, &archive_path);
      if (archive != nil) {
        pipeline_desc.binaryArchives = @[ archive ];
        pipeline.render_pipeline = [_impl->metal_device newRenderPipelineStateWithDescriptor:pipeline_desc options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&pipeline_error];
        pipeline_desc.binaryArchives = nil;
        [archive release];
      }
    }

    if (pipeline.render_pipeline == nil) {
      id<MTLBinaryArchive> archive = create_metal_binary_archive(_impl->metal_device, nullptr);
      if (archive != nil) {
        pipeline_desc.binaryArchives = @[ archive ];
        NSError* archive_error = nil;
        const BOOL added = [archive addRenderPipelineFunctionsWithDescriptor:pipeline_desc error:&archive_error];
        pipeline_error = nil;
        pipeline.render_pipeline = [_impl->metal_device newRenderPipelineStateWithDescriptor:pipeline_desc options:MTLPipelineOptionNone reflection:nil error:&pipeline_error];
        if ((added == YES) && (pipeline.render_pipeline != nil) && serialize_metal_binary_archive(archive, archive_path)) {
          prune_cache_slot_versions(archive_path.parent_path(), slot_prefix, archive_path);
        } else if ((added == NO) && (archive_error != nil)) {
          log::warning("Metal RHI: failed to populate per-shader render pipeline archive: %s", safe_nsstring([archive_error localizedDescription], "unknown error"));
        }
        pipeline_desc.binaryArchives = nil;
        [archive release];
      }
    }
  }

  if (pipeline.render_pipeline == nil) {
    pipeline_error = nil;
    pipeline.render_pipeline = [_impl->metal_device newRenderPipelineStateWithDescriptor:pipeline_desc error:&pipeline_error];
  }
  [pipeline_desc release];
  if (pipeline.render_pipeline == nil) {
    log::error("Metal RHI: render pipeline creation failed: %s", pipeline_error ? [[pipeline_error localizedDescription] UTF8String] : "unknown error");
    release_stage_resources(pipeline.vertex_stage);
    release_stage_resources(pipeline.fragment_stage);
    return {RHIResult::ValidationError, {}};
  }
  prune_metal_library_cache_slot(desc.vertex_shader);
  prune_metal_library_cache_slot(desc.fragment_shader);

  MTLDepthStencilDescriptor* depth_desc = [[MTLDepthStencilDescriptor alloc] init];
  depth_desc.depthCompareFunction = to_metal_compare(desc.depth_state.depth_compare_op);
  depth_desc.depthWriteEnabled = desc.depth_state.depth_write_enable ? YES : NO;
  pipeline.depth_state = [_impl->metal_device newDepthStencilStateWithDescriptor:depth_desc];
  [depth_desc release];

  const RHIPipeline handle = Handle::construct(0u, _impl->next_pipeline_index++, 1u);
  _impl->pipelines.emplace(handle, std::move(pipeline));
  return {RHIResult::Success, handle};
}

struct MTComputePipelineCreationResult {
  RHICreatePipelineResult result = {};
  bool cache_hit = false;
};

static id<MTLBinaryArchive> create_metal_binary_archive(id<MTLDevice> device, const std::filesystem::path* path) {
  if (device == nil) {
    return nil;
  }

  MTLBinaryArchiveDescriptor* descriptor = [[MTLBinaryArchiveDescriptor alloc] init];
  if (path != nullptr) {
    NSString* path_text = [NSString stringWithUTF8String:path->string().c_str()];
    descriptor.url = [NSURL fileURLWithPath:path_text];
  }
  NSError* error = nil;
  id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:descriptor error:&error];
  [descriptor release];
  if ((archive == nil) && (error != nil)) {
    log::warning("Metal RHI: failed to open per-shader pipeline archive: %s", safe_nsstring([error localizedDescription], "unknown error"));
  }
  return archive;
}

static bool serialize_metal_binary_archive(id<MTLBinaryArchive> archive, const std::filesystem::path& path) {
  if ((archive == nil) || (ensure_directory_exists(path.parent_path()) == false)) {
    return false;
  }

  static std::atomic<uint64_t> next_temp_file_id = 0u;
  std::filesystem::path temp_path = path;
  temp_path += ".tmp." + std::to_string(static_cast<uint64_t>([[NSProcessInfo processInfo] processIdentifier])) + "." + std::to_string(next_temp_file_id.fetch_add(1u));
  std::error_code ec = {};
  std::filesystem::remove(temp_path, ec);
  NSString* temp_path_text = [NSString stringWithUTF8String:temp_path.string().c_str()];
  NSError* error = nil;
  if ([archive serializeToURL:[NSURL fileURLWithPath:temp_path_text] error:&error] == NO) {
    std::filesystem::remove(temp_path, ec);
    log::warning("Metal RHI: failed to serialize per-shader pipeline archive: %s", safe_nsstring([error localizedDescription], "unknown error"));
    return false;
  }

  std::filesystem::rename(temp_path, path, ec);
  if (ec.value() != 0) {
    std::filesystem::remove(temp_path, ec);
    return false;
  }
  return true;
}

static MTComputePipelineCreationResult create_metal_compute_pipeline(MTDevice::Impl* device, const RHIComputePipelineDesc& desc) {
  if ((device == nullptr) || (device->metal_device == nil)) {
    return {{RHIResult::InvalidArgument, {}}, false};
  }
  const bool format_supported = (desc.compute_shader.format == RHIShaderBinaryFormat::MetalSource) || (desc.compute_shader.format == RHIShaderBinaryFormat::MetalLibrary);
  if ((desc.compute_shader.backend != RHIBackend::Metal) || (format_supported == false)) {
    return {{RHIResult::InvalidArgument, {}}, false};
  }

  std::string error_message = {};
  MTPipelineData pipeline = {};
  pipeline.is_compute = true;
  pipeline.debug_name = desc.compute_shader.entry_point;
  pipeline.compute_desc = desc;
  if (create_stage_resources(device, device->bindless_manager->_impl, desc.compute_shader, pipeline.compute_stage, error_message) == false) {
    log::error("Metal RHI: compute shader stage creation failed: %s", error_message.c_str());
    return {{RHIResult::ValidationError, {}}, false};
  }

  MTLComputePipelineDescriptor* pipeline_desc = [[MTLComputePipelineDescriptor alloc] init];
  pipeline_desc.computeFunction = pipeline.compute_stage.function;
  NSError* pipeline_error = nil;
  bool cache_hit = false;
  if (@available(macOS 11.0, *)) {
    const uint64_t content_hash = (desc.compute_shader.content_hash != 0u) ? desc.compute_shader.content_hash : etx_hash64(desc.compute_shader.spirv_data, static_cast<size_t>(desc.compute_shader.spirv_size));
    const uint64_t cache_key = (desc.compute_shader.cache_key != 0u) ? desc.compute_shader.cache_key : content_hash;
    const std::filesystem::path archive_path = metal_pipeline_archive_path(device->metal_device, cache_key, content_hash);
    const std::string slot_prefix = format_hash_hex(cache_key) + "_";
    std::error_code ec = {};
    const bool archive_exists = std::filesystem::exists(archive_path, ec) && (ec.value() == 0);
    if (archive_exists) {
      id<MTLBinaryArchive> archive = create_metal_binary_archive(device->metal_device, &archive_path);
      if (archive != nil) {
        pipeline_desc.binaryArchives = @[ archive ];
        pipeline.compute_pipeline = [device->metal_device newComputePipelineStateWithDescriptor:pipeline_desc options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&pipeline_error];
        cache_hit = pipeline.compute_pipeline != nil;
        pipeline_desc.binaryArchives = nil;
        [archive release];
      }
    }

    if (pipeline.compute_pipeline == nil) {
      id<MTLBinaryArchive> archive = create_metal_binary_archive(device->metal_device, nullptr);
      if (archive != nil) {
        pipeline_desc.binaryArchives = @[ archive ];
        NSError* archive_error = nil;
        const BOOL added = [archive addComputePipelineFunctionsWithDescriptor:pipeline_desc error:&archive_error];
        pipeline_error = nil;
        pipeline.compute_pipeline = [device->metal_device newComputePipelineStateWithDescriptor:pipeline_desc options:MTLPipelineOptionNone reflection:nil error:&pipeline_error];
        if ((added == YES) && (pipeline.compute_pipeline != nil) && serialize_metal_binary_archive(archive, archive_path)) {
          prune_cache_slot_versions(archive_path.parent_path(), slot_prefix, archive_path);
        } else if ((added == NO) && (archive_error != nil)) {
          log::warning("Metal RHI: failed to populate per-shader pipeline archive: %s", safe_nsstring([archive_error localizedDescription], "unknown error"));
        }
        pipeline_desc.binaryArchives = nil;
        [archive release];
      }
    }
  }

  if (pipeline.compute_pipeline == nil) {
    pipeline_error = nil;
    pipeline.compute_pipeline = [device->metal_device newComputePipelineStateWithDescriptor:pipeline_desc options:MTLPipelineOptionNone reflection:nil error:&pipeline_error];
  }
  [pipeline_desc release];
  if (pipeline.compute_pipeline == nil) {
    log::error("Metal RHI: compute pipeline creation failed: %s", pipeline_error ? [[pipeline_error localizedDescription] UTF8String] : "unknown error");
    release_stage_resources(pipeline.compute_stage);
    return {{RHIResult::ValidationError, {}}, false};
  }
  prune_metal_library_cache_slot(desc.compute_shader);

  const RHIPipeline handle = Handle::construct(0u, device->next_pipeline_index++, 1u);
  device->pipelines.emplace(handle, std::move(pipeline));
  return {{RHIResult::Success, handle}, cache_hit};
}

RHICreatePipelineResult MTDevice::create_compute_pipeline(const RHIComputePipelineDesc& desc) {
  return create_metal_compute_pipeline(_impl, desc).result;
}

std::vector<RHICreatePipelineBatchEntry> MTDevice::create_compute_pipelines(const std::vector<RHIComputePipelineDesc>& descs, uint32_t max_concurrency, const RHIPipelineBatchProgressCallback& progress_callback) {
  (void)max_concurrency;
  std::vector<RHICreatePipelineBatchEntry> results(descs.size());
  for (size_t i = 0u; i < descs.size(); ++i) {
    results[i].state = RHIPipelineBatchProgressState::DriverCompiling;
    if (progress_callback) {
      progress_callback(static_cast<uint32_t>(i), results[i]);
    }
    const auto begin = std::chrono::steady_clock::now();
    const MTComputePipelineCreationResult result = create_metal_compute_pipeline(_impl, descs[i]);
    const auto end = std::chrono::steady_clock::now();
    results[i] = {
      .result = result.result.result,
      .handle = result.result.handle,
      .elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count(),
      .cache_hit = result.cache_hit,
      .state = RHIPipelineBatchProgressState::Complete,
    };
    if (progress_callback) {
      progress_callback(static_cast<uint32_t>(i), results[i]);
    }
  }
  return results;
}

void MTDevice::persist_pipeline_cache() {
}

RHIResult MTDevice::destroy_buffer(RHIBuffer buffer) {
  if (!buffer.valid()) {
    return RHIResult::Success;
  }
  auto it = _impl->buffers.find(buffer);
  if (it == _impl->buffers.end()) {
    return RHIResult::Success;
  }
  _impl->bindless_manager->unregister_buffer(buffer);
  _impl->gpu_allocated_bytes -= static_cast<uint64_t>(it->second.buffer.length);
  [it->second.buffer release];
  _impl->buffers.erase(it);
  return RHIResult::Success;
}

RHIResult MTDevice::destroy_texture(RHITexture texture) {
  if (!texture.valid()) {
    return RHIResult::Success;
  }
  auto it = _impl->textures.find(texture);
  if (it == _impl->textures.end()) {
    return RHIResult::Success;
  }
  _impl->bindless_manager->unregister_texture(texture);
  if (!it->second.is_swapchain_texture) {
    [it->second.texture release];
  }
  _impl->textures.erase(it);
  return RHIResult::Success;
}

RHIResult MTDevice::destroy_sampler(RHISampler sampler) {
  if (!sampler.valid()) {
    return RHIResult::Success;
  }
  auto it = _impl->samplers.find(sampler);
  if (it == _impl->samplers.end()) {
    return RHIResult::Success;
  }
  _impl->bindless_manager->unregister_sampler(sampler);
  [it->second.sampler release];
  _impl->samplers.erase(it);
  return RHIResult::Success;
}

RHIResult MTDevice::destroy_pipeline(RHIPipeline pipeline) {
  if (!pipeline.valid()) {
    return RHIResult::Success;
  }
  auto it = _impl->pipelines.find(pipeline);
  if (it == _impl->pipelines.end()) {
    return RHIResult::Success;
  }
  release_stage_resources(it->second.vertex_stage);
  release_stage_resources(it->second.fragment_stage);
  release_stage_resources(it->second.compute_stage);
  [it->second.depth_state release];
  [it->second.render_pipeline release];
  [it->second.compute_pipeline release];
  _impl->pipelines.erase(it);
  return RHIResult::Success;
}

RHIResult MTDevice::update_buffer(RHIBuffer buffer, const void* data, uint64_t size, uint64_t offset) {
  auto it = _impl->buffers.find(buffer);
  if (it == _impl->buffers.end()) {
    return RHIResult::InvalidHandle;
  }
  if ((data == nullptr) || ((offset + size) > it->second.desc.size)) {
    return RHIResult::InvalidArgument;
  }
  std::memcpy(static_cast<uint8_t*>(it->second.buffer.contents) + offset, data, static_cast<size_t>(size));
  [it->second.buffer didModifyRange:NSMakeRange(static_cast<NSUInteger>(offset), static_cast<NSUInteger>(size))];
  return RHIResult::Success;
}

RHIResult MTDevice::read_buffer(RHIBuffer buffer, void* data, uint64_t size, uint64_t offset) {
  auto it = _impl->buffers.find(buffer);
  if (it == _impl->buffers.end()) {
    return RHIResult::InvalidHandle;
  }
  if ((data == nullptr) || ((offset + size) > it->second.desc.size)) {
    return RHIResult::InvalidArgument;
  }
  std::memcpy(data, static_cast<const uint8_t*>(it->second.buffer.contents) + offset, static_cast<size_t>(size));
  return RHIResult::Success;
}

RHIResult MTDevice::update_texture(RHITexture texture, const void* data, uint32_t mip_level, uint32_t array_layer) {
  auto it = _impl->textures.find(texture);
  if (it == _impl->textures.end()) {
    return RHIResult::InvalidHandle;
  }
  if ((data == nullptr) || (it->second.texture == nil) || (it->second.desc.sample_count > 1u)) {
    return RHIResult::InvalidArgument;
  }

  const uint32_t mip_width = std::max<uint32_t>(1u, it->second.desc.width >> mip_level);
  const uint32_t mip_height = std::max<uint32_t>(1u, it->second.desc.height >> mip_level);
  const uint32_t bytes_per_pixel = format_bytes_per_pixel(it->second.desc.format);
  if (bytes_per_pixel == 0u) {
    return RHIResult::InvalidArgument;
  }

  const MTLRegion region = MTLRegionMake2D(0, 0, mip_width, mip_height);
  const NSUInteger bytes_per_row = static_cast<NSUInteger>(bytes_per_pixel * mip_width);
  if (it->second.texture.storageMode == MTLStorageModePrivate) {
    if (_impl->command_queue == nil) {
      return RHIResult::InvalidArgument;
    }

    const NSUInteger upload_size = bytes_per_row * static_cast<NSUInteger>(mip_height);
    id<MTLBuffer> staging_buffer = [_impl->metal_device newBufferWithLength:upload_size options:MTLResourceStorageModeShared];
    if (staging_buffer == nil) {
      return RHIResult::OutOfMemory;
    }

    std::memcpy(staging_buffer.contents, data, static_cast<size_t>(upload_size));
    [staging_buffer didModifyRange:NSMakeRange(0u, upload_size)];

    id<MTLCommandBuffer> upload_command_buffer = create_diagnostic_command_buffer(_impl->command_queue, @"ETX texture upload");
    if (upload_command_buffer == nil) {
      [staging_buffer release];
      return RHIResult::OutOfMemory;
    }

    id<MTLBlitCommandEncoder> encoder = [upload_command_buffer blitCommandEncoder];
    encoder.label = @"ETX texture upload blit";
    [encoder copyFromBuffer:staging_buffer
               sourceOffset:0
          sourceBytesPerRow:bytes_per_row
        sourceBytesPerImage:bytes_per_row * static_cast<NSUInteger>(mip_height)
                 sourceSize:MTLSizeMake(mip_width, mip_height, 1u)
                  toTexture:it->second.texture
           destinationSlice:array_layer
           destinationLevel:mip_level
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [encoder endEncoding];
    [upload_command_buffer commit];
    [upload_command_buffer waitUntilCompleted];
    if (upload_command_buffer.status == MTLCommandBufferStatusError) {
      log_metal_command_buffer_error_details(upload_command_buffer);
      [staging_buffer release];
      return RHIResult::DeviceLost;
    }
    [staging_buffer release];
    return RHIResult::Success;
  }

  [it->second.texture replaceRegion:region mipmapLevel:mip_level slice:array_layer withBytes:data bytesPerRow:bytes_per_row bytesPerImage:0];
  return RHIResult::Success;
}

MTBindlessManager::MTBindlessManager()
  : _impl(new Impl()) {
}

MTBindlessManager::~MTBindlessManager() {
  delete _impl;
}

void MTBindlessManager::set_max_buffers(uint32_t count) {
  _impl->max_buffers = count;
  _impl->buffer_entries.resize(count);
  update_bindless_revision(_impl);
}

void MTBindlessManager::set_max_textures(uint32_t count) {
  _impl->max_textures = count;
  _impl->texture_entries.resize(count);
  update_bindless_revision(_impl);
}

void MTBindlessManager::set_max_samplers(uint32_t count) {
  _impl->max_samplers = count;
  _impl->sampler_entries.resize(count);
  update_bindless_revision(_impl);
}

void MTBindlessManager::set_max_acceleration_structures(uint32_t count) {
  _impl->max_acceleration_structures = count;
  _impl->acceleration_structure_entries.resize(count);
  update_bindless_revision(_impl);
}

RHIResult MTBindlessManager::register_buffer(void* buffer, RHIResourceType type, RHIBindlessHandle& out_handle) {
  auto& entries = bindless_entries_for_type(_impl, type);
  // Reserve descriptor slot 0 as an invalid/null bindless index to match shader-side expectations.
  for (uint32_t i = 1, e = static_cast<uint32_t>(entries.size()); i < e; ++i) {
    if (entries[i].valid == false) {
      entries[i].generation = (entries[i].generation + 1u) & kRHIBindlessGenerationMask;
      entries[i].valid = true;
      entries[i].resource = buffer;
      out_handle = make_bindless_handle(type, entries[i].generation, i);
      bindless_count_for_type(_impl, type) += 1u;
      update_bindless_revision(_impl);
      return RHIResult::Success;
    }
  }
  return RHIResult::OutOfMemory;
}

RHIResult MTBindlessManager::register_texture(void* image_view, RHIResourceType type, RHIBindlessHandle& out_handle, uint32_t usage_flags, void* image) {
  (void)usage_flags;
  return register_buffer(image_view ? image_view : image, type, out_handle);
}

RHIResult MTBindlessManager::register_sampler(void* sampler, RHIResourceType type, RHIBindlessHandle& out_handle) {
  return register_buffer(sampler, type, out_handle);
}

RHIResult MTBindlessManager::register_acceleration_structure(const void* data, uint64_t size, RHIBindlessHandle& out_handle) {
  (void)size;
  return register_buffer(const_cast<void*>(data), RHIResourceType::AccelerationStructure, out_handle);
}

RHIResult MTBindlessManager::unregister_buffer(RHIBindlessHandle handle) {
  return unregister_texture(handle);
}

RHIResult MTBindlessManager::unregister_texture(RHIBindlessHandle handle) {
  const RHIResourceType type = get_bindless_resource_type(handle);
  auto& entries = bindless_entries_for_type(_impl, type);
  const uint32_t descriptor_index = get_bindless_descriptor_index(handle);
  if (descriptor_index >= entries.size()) {
    return RHIResult::InvalidHandle;
  }
  auto& entry = entries[descriptor_index];
  if (!bindless_handle_matches(entry, handle)) {
    return RHIResult::InvalidHandle;
  }
  entry.valid = false;
  entry.resource = nullptr;
  bindless_count_for_type(_impl, type) = std::max<uint32_t>(0u, bindless_count_for_type(_impl, type) - 1u);
  update_bindless_revision(_impl);
  return RHIResult::Success;
}

RHIResult MTBindlessManager::unregister_sampler(RHIBindlessHandle handle) {
  return unregister_texture(handle);
}

RHIResult MTBindlessManager::unregister_acceleration_structure(RHIBindlessHandle handle) {
  return unregister_texture(handle);
}

bool MTBindlessManager::is_valid_handle(RHIBindlessHandle handle) const {
  const RHIResourceType type = get_bindless_resource_type(handle);
  const auto& entries = bindless_entries_for_type(_impl, type);
  const uint32_t descriptor_index = get_bindless_descriptor_index(handle);
  return (descriptor_index < entries.size()) && bindless_handle_matches(entries[descriptor_index], handle);
}

RHIResourceType MTBindlessManager::get_resource_type(RHIBindlessHandle handle) const {
  return get_bindless_resource_type(handle);
}

uint32_t MTBindlessManager::get_max_buffers() const {
  return _impl->max_buffers;
}

uint32_t MTBindlessManager::get_max_textures() const {
  return _impl->max_textures;
}

uint32_t MTBindlessManager::get_max_samplers() const {
  return _impl->max_samplers;
}

uint32_t MTBindlessManager::get_max_acceleration_structures() const {
  return _impl->max_acceleration_structures;
}

uint32_t MTBindlessManager::get_buffer_count() const {
  return _impl->buffer_count;
}

uint32_t MTBindlessManager::get_texture_count() const {
  return _impl->texture_count;
}

uint32_t MTBindlessManager::get_sampler_count() const {
  return _impl->sampler_count;
}

uint32_t MTBindlessManager::get_acceleration_structure_count() const {
  return _impl->acceleration_structure_count;
}

MTCommandBuffer::MTCommandBuffer()
  : _impl(new Impl()) {
}

MTCommandBuffer::~MTCommandBuffer() {
  reset();
  delete _impl;
}

void MTCommandBuffer::begin() {
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  if ((owner == nullptr) || (owner->command_queue == nil)) {
    return;
  }
  reset();
  NSString* command_buffer_label = [NSString stringWithFormat:@"ETX command buffer #%llu", static_cast<unsigned long long>(owner->next_command_buffer_serial++)];
  _impl->command_buffer = create_diagnostic_command_buffer(owner->command_queue, command_buffer_label);
  [_impl->command_buffer retain];
}

void MTCommandBuffer::end() {
  [_impl->render_encoder endEncoding];
  _impl->render_encoder = nil;
  [_impl->compute_encoder endEncoding];
  _impl->compute_encoder = nil;
  [_impl->blit_encoder endEncoding];
  _impl->blit_encoder = nil;
}

void MTCommandBuffer::reset() {
  end();
  [_impl->command_buffer release];
  _impl->command_buffer = nil;
  [_impl->timestamp_sample_buffer release];
  _impl->timestamp_sample_buffer = nil;
  reset_command_buffer_state(_impl);
}

void MTCommandBuffer::detach_submitted() {
  end();
  _impl->command_buffer = nil;
  reset_command_buffer_state(_impl);
}

void MTCommandBuffer::compute_barrier() {
  if (_impl->compute_encoder != nil) {
    // An encoder boundary is a full compute dependency boundary and keeps long heterogeneous dispatch streams schedulable.
    [_impl->compute_encoder endEncoding];
    _impl->compute_encoder = nil;
  }
}

void MTCommandBuffer::buffer_barrier(RHIBuffer buffer, RHIResourceState old_state, RHIResourceState new_state) {
  (void)old_state;
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  auto it = owner->device._impl->buffers.find(buffer);
  if (it != owner->device._impl->buffers.end()) {
    it->second.state = new_state;
    if (@available(macOS 10.14, *)) {
      if (_impl->compute_encoder != nil) {
        id<MTLResource> resource = it->second.buffer;
        [_impl->compute_encoder memoryBarrierWithResources:&resource count:1u];
      }
    }
  }
}

void MTCommandBuffer::texture_barrier(RHITexture texture, RHIResourceState old_state, RHIResourceState new_state) {
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  auto it = owner->device._impl->textures.find(texture);
  if (it != owner->device._impl->textures.end()) {
    it->second.state = new_state;
    if (@available(macOS 10.14, *)) {
      if (_impl->compute_encoder != nil) {
        id<MTLResource> resource = it->second.texture;
        [_impl->compute_encoder memoryBarrierWithResources:&resource count:1u];
        if (metal_texture_state_requires_compute_encoder_break(old_state, new_state)) {
          [_impl->compute_encoder endEncoding];
          _impl->compute_encoder = nil;
        }
      }
    }
  }
}

void MTCommandBuffer::begin_render_pass(uint32_t color_attachment_count, RHITexture* color_attachments, const float* clear_colors, RHITexture depth_attachment, const RHIResourceState* color_final_states, RHIResourceState depth_final_state) {
  [_impl->compute_encoder endEncoding];
  _impl->compute_encoder = nil;
  [_impl->blit_encoder endEncoding];
  _impl->blit_encoder = nil;
  [_impl->render_encoder endEncoding];
  _impl->render_encoder = nil;
  if (_impl->command_buffer == nil) {
    return;
  }
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  _impl->current_color_attachments = {};
  _impl->current_color_final_states = {};
  _impl->current_color_attachment_count = std::min<uint32_t>(color_attachment_count, static_cast<uint32_t>(kMetalMaxColorAttachments));
  _impl->current_depth_attachment = depth_attachment;
  _impl->current_depth_final_state = depth_final_state;

  MTLRenderPassDescriptor* pass_desc = [MTLRenderPassDescriptor renderPassDescriptor];
  for (uint32_t i = 0; i < _impl->current_color_attachment_count; ++i) {
    auto it = owner->device._impl->textures.find(color_attachments[i]);
    if (it == owner->device._impl->textures.end()) {
      continue;
    }
    _impl->current_color_attachments[i] = color_attachments[i];
    _impl->current_color_final_states[i] = (color_final_states != nullptr) ? color_final_states[i] : RHIResourceState::ColorAttachment;
    pass_desc.colorAttachments[i].texture = it->second.texture;
    pass_desc.colorAttachments[i].loadAction = clear_colors ? MTLLoadActionClear : MTLLoadActionLoad;
    pass_desc.colorAttachments[i].storeAction = MTLStoreActionStore;
    if (clear_colors != nullptr) {
      pass_desc.colorAttachments[i].clearColor = MTLClearColorMake(clear_colors[i * 4 + 0], clear_colors[i * 4 + 1], clear_colors[i * 4 + 2], clear_colors[i * 4 + 3]);
    }
  }

  if (depth_attachment.valid()) {
    auto depth_it = owner->device._impl->textures.find(depth_attachment);
    if (depth_it != owner->device._impl->textures.end()) {
      pass_desc.depthAttachment.texture = depth_it->second.texture;
      pass_desc.depthAttachment.loadAction = MTLLoadActionClear;
      pass_desc.depthAttachment.storeAction = MTLStoreActionStore;
      pass_desc.depthAttachment.clearDepth = 1.0;
    }
  }

  _impl->render_encoder = [_impl->command_buffer renderCommandEncoderWithDescriptor:pass_desc];
  _impl->render_encoder.label = @"ETX render pass";
}

void MTCommandBuffer::end_render_pass() {
  [_impl->render_encoder endEncoding];
  _impl->render_encoder = nil;

  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  if (owner != nullptr) {
    for (uint32_t i = 0; i < _impl->current_color_attachment_count; ++i) {
      if (_impl->current_color_attachments[i].valid() == false) {
        continue;
      }
      auto texture_it = owner->device._impl->textures.find(_impl->current_color_attachments[i]);
      if (texture_it != owner->device._impl->textures.end()) {
        texture_it->second.state = _impl->current_color_final_states[i];
      }
    }
    if (_impl->current_depth_attachment.valid()) {
      auto depth_it = owner->device._impl->textures.find(_impl->current_depth_attachment);
      if (depth_it != owner->device._impl->textures.end()) {
        depth_it->second.state = _impl->current_depth_final_state;
      }
    }
  }

  _impl->current_color_attachments = {};
  _impl->current_color_final_states = {};
  _impl->current_color_attachment_count = 0u;
  _impl->current_depth_attachment = {};
  _impl->current_depth_final_state = RHIResourceState::Undefined;
}

void MTCommandBuffer::set_viewport(const RHIViewport& viewport) {
  double x = viewport.x;
  double y = viewport.y;
  double width = viewport.width;
  double height = viewport.height;
  if (width < 0.0) {
    x += width;
    width = -width;
  }
  if (height < 0.0) {
    y += height;
    height = -height;
  }

  _impl->viewport = {x, y, width, height, viewport.min_depth, viewport.max_depth};
  _impl->viewport_valid = true;
  if (_impl->render_encoder != nil) {
    [_impl->render_encoder setViewport:_impl->viewport];
  }

  RHIRect auto_scissor = {};
  auto_scissor.x = static_cast<int32_t>(x);
  auto_scissor.y = static_cast<int32_t>(y);
  auto_scissor.width = static_cast<uint32_t>(std::max(width, 0.0));
  auto_scissor.height = static_cast<uint32_t>(std::max(height, 0.0));
  set_scissor(auto_scissor);
}

void MTCommandBuffer::set_scissor(const RHIRect& scissor) {
  _impl->scissor = {static_cast<NSUInteger>(std::max(scissor.x, 0)), static_cast<NSUInteger>(std::max(scissor.y, 0)), scissor.width, scissor.height};
  _impl->scissor_valid = true;
  if (_impl->render_encoder != nil) {
    [_impl->render_encoder setScissorRect:_impl->scissor];
  }
}

void MTCommandBuffer::set_pipeline(RHIPipeline pipeline) {
  _impl->current_pipeline = pipeline;
}

void MTCommandBuffer::push_constants(const void* data, uint32_t size, uint32_t offset) {
  if ((data == nullptr) || (size == 0u) || ((offset + size) > _impl->push_constants.size())) {
    return;
  }
  std::memcpy(_impl->push_constants.data() + offset, data, size);
  _impl->push_constants_size = std::max<uint32_t>(_impl->push_constants_size, offset + size);
}

void MTCommandBuffer::draw(const RHIDrawDesc& desc) {
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  auto it = owner->device._impl->pipelines.find(_impl->current_pipeline);
  if ((it == owner->device._impl->pipelines.end()) || (it->second.is_compute) || (_impl->render_encoder == nil)) {
    return;
  }

  auto& pipeline = it->second;
  encode_stage_bindless_resources(pipeline.vertex_stage, owner->bindless_manager._impl, owner->device._impl);
  encode_stage_bindless_resources(pipeline.fragment_stage, owner->bindless_manager._impl, owner->device._impl);
  declare_render_stage_bindless_resources(_impl->render_encoder, pipeline.vertex_stage, owner->bindless_manager._impl, owner->device._impl, MTLRenderStageVertex);
  declare_render_stage_bindless_resources(_impl->render_encoder, pipeline.fragment_stage, owner->bindless_manager._impl, owner->device._impl, MTLRenderStageFragment);
  if (pipeline.debug_name.empty() == false) {
    _impl->render_encoder.label = [NSString stringWithFormat:@"ETX render pass: %s", pipeline.debug_name.c_str()];
  }
  [_impl->render_encoder setRenderPipelineState:pipeline.render_pipeline];
  if (pipeline.depth_state != nil) {
    [_impl->render_encoder setDepthStencilState:pipeline.depth_state];
  }
  [_impl->render_encoder setTriangleFillMode:pipeline.fill_mode];
  if (_impl->viewport_valid) {
    [_impl->render_encoder setViewport:_impl->viewport];
  }
  if (_impl->scissor_valid) {
    [_impl->render_encoder setScissorRect:_impl->scissor];
  }
  for (uint32_t binding_index = 0; binding_index < kMetalBindlessBindingCount; ++binding_index) {
    if (pipeline.vertex_stage.bindless_argument_buffers[binding_index] != nil) {
      [_impl->render_encoder setVertexBuffer:pipeline.vertex_stage.bindless_argument_buffers[binding_index] offset:0 atIndex:pipeline.vertex_stage.bindless_buffer_indices[binding_index]];
    }
    if (pipeline.fragment_stage.bindless_argument_buffers[binding_index] != nil) {
      [_impl->render_encoder setFragmentBuffer:pipeline.fragment_stage.bindless_argument_buffers[binding_index] offset:0 atIndex:pipeline.fragment_stage.bindless_buffer_indices[binding_index]];
    }
  }
  if ((_impl->push_constants_size > 0u) && pipeline.vertex_stage.uses_push_constants) {
    [_impl->render_encoder setVertexBytes:_impl->push_constants.data() length:_impl->push_constants_size atIndex:pipeline.vertex_stage.push_constants_buffer_index];
  }
  if ((_impl->push_constants_size > 0u) && pipeline.fragment_stage.uses_push_constants) {
    [_impl->render_encoder setFragmentBytes:_impl->push_constants.data() length:_impl->push_constants_size atIndex:pipeline.fragment_stage.push_constants_buffer_index];
  }
  [_impl->render_encoder drawPrimitives:pipeline.primitive vertexStart:desc.first_vertex vertexCount:desc.vertex_count instanceCount:desc.instance_count baseInstance:desc.first_instance];
}

void MTCommandBuffer::draw_indexed(const RHIIndexedDrawDesc& desc, RHIBuffer index_buffer) {
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  auto pipeline_it = owner->device._impl->pipelines.find(_impl->current_pipeline);
  auto index_it = owner->device._impl->buffers.find(index_buffer);
  if ((pipeline_it == owner->device._impl->pipelines.end()) || (index_it == owner->device._impl->buffers.end()) || (pipeline_it->second.is_compute) || (_impl->render_encoder == nil)) {
    return;
  }

  auto& pipeline = pipeline_it->second;
  encode_stage_bindless_resources(pipeline.vertex_stage, owner->bindless_manager._impl, owner->device._impl);
  encode_stage_bindless_resources(pipeline.fragment_stage, owner->bindless_manager._impl, owner->device._impl);
  declare_render_stage_bindless_resources(_impl->render_encoder, pipeline.vertex_stage, owner->bindless_manager._impl, owner->device._impl, MTLRenderStageVertex);
  declare_render_stage_bindless_resources(_impl->render_encoder, pipeline.fragment_stage, owner->bindless_manager._impl, owner->device._impl, MTLRenderStageFragment);
  if (pipeline.debug_name.empty() == false) {
    _impl->render_encoder.label = [NSString stringWithFormat:@"ETX render pass: %s", pipeline.debug_name.c_str()];
  }
  [_impl->render_encoder setRenderPipelineState:pipeline.render_pipeline];
  if (pipeline.depth_state != nil) {
    [_impl->render_encoder setDepthStencilState:pipeline.depth_state];
  }
  [_impl->render_encoder setTriangleFillMode:pipeline.fill_mode];
  if (_impl->viewport_valid) {
    [_impl->render_encoder setViewport:_impl->viewport];
  }
  if (_impl->scissor_valid) {
    [_impl->render_encoder setScissorRect:_impl->scissor];
  }
  for (uint32_t binding_index = 0; binding_index < kMetalBindlessBindingCount; ++binding_index) {
    if (pipeline.vertex_stage.bindless_argument_buffers[binding_index] != nil) {
      [_impl->render_encoder setVertexBuffer:pipeline.vertex_stage.bindless_argument_buffers[binding_index] offset:0 atIndex:pipeline.vertex_stage.bindless_buffer_indices[binding_index]];
    }
    if (pipeline.fragment_stage.bindless_argument_buffers[binding_index] != nil) {
      [_impl->render_encoder setFragmentBuffer:pipeline.fragment_stage.bindless_argument_buffers[binding_index] offset:0 atIndex:pipeline.fragment_stage.bindless_buffer_indices[binding_index]];
    }
  }
  if ((_impl->push_constants_size > 0u) && pipeline.vertex_stage.uses_push_constants) {
    [_impl->render_encoder setVertexBytes:_impl->push_constants.data() length:_impl->push_constants_size atIndex:pipeline.vertex_stage.push_constants_buffer_index];
  }
  if ((_impl->push_constants_size > 0u) && pipeline.fragment_stage.uses_push_constants) {
    [_impl->render_encoder setFragmentBytes:_impl->push_constants.data() length:_impl->push_constants_size atIndex:pipeline.fragment_stage.push_constants_buffer_index];
  }
  [_impl->render_encoder drawIndexedPrimitives:pipeline.primitive
                                    indexCount:desc.index_count
                                     indexType:to_metal_index_type(desc.index_type)
                                   indexBuffer:index_it->second.buffer
                             indexBufferOffset:static_cast<NSUInteger>(desc.first_index * ((desc.index_type == RHIIndexType::UInt32) ? 4u : 2u))
                                 instanceCount:desc.instance_count
                                    baseVertex:static_cast<NSInteger>(desc.vertex_offset)
                                  baseInstance:desc.first_instance];
}

void MTCommandBuffer::dispatch(const RHIDispatchDesc& desc) {
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  auto it = owner->device._impl->pipelines.find(_impl->current_pipeline);
  if ((it == owner->device._impl->pipelines.end()) || (it->second.is_compute == false) || (_impl->command_buffer == nil)) {
    return;
  }

  if (ensure_compute_encoder(_impl) == nil) {
    return;
  }

  auto& pipeline = it->second;
  if (pipeline.debug_name.empty() == false) {
    _impl->compute_encoder.label = [NSString stringWithFormat:@"ETX compute dispatch: %s", pipeline.debug_name.c_str()];
  } else {
    _impl->compute_encoder.label = @"ETX compute dispatch";
  }
  encode_stage_bindless_resources(pipeline.compute_stage, owner->bindless_manager._impl, owner->device._impl);
  declare_compute_stage_bindless_resources(_impl->compute_encoder, pipeline.compute_stage, owner->bindless_manager._impl, owner->device._impl);
  [_impl->compute_encoder setComputePipelineState:pipeline.compute_pipeline];
  for (uint32_t binding_index = 0; binding_index < kMetalBindlessBindingCount; ++binding_index) {
    if (pipeline.compute_stage.bindless_argument_buffers[binding_index] != nil) {
      [_impl->compute_encoder setBuffer:pipeline.compute_stage.bindless_argument_buffers[binding_index] offset:0 atIndex:pipeline.compute_stage.bindless_buffer_indices[binding_index]];
    }
  }
  if ((_impl->push_constants_size > 0u) && pipeline.compute_stage.uses_push_constants) {
    [_impl->compute_encoder setBytes:_impl->push_constants.data() length:_impl->push_constants_size atIndex:pipeline.compute_stage.push_constants_buffer_index];
  }

  const auto& shader = pipeline.compute_desc.compute_shader;
  const MTLSize threads_per_group = MTLSizeMake(std::max<uint32_t>(shader.local_size_x, 1u), std::max<uint32_t>(shader.local_size_y, 1u), std::max<uint32_t>(shader.local_size_z, 1u));
  const MTLSize threadgroups = MTLSizeMake(desc.group_count_x, desc.group_count_y, desc.group_count_z);
  [_impl->compute_encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_group];
}

void MTCommandBuffer::dispatch_indirect(RHIBindlessHandle argument_buffer, uint64_t argument_buffer_offset) {
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  auto pipeline_it = owner->device._impl->pipelines.find(_impl->current_pipeline);
  auto buffer_it = owner->device._impl->buffers.find(argument_buffer);
  if ((pipeline_it == owner->device._impl->pipelines.end()) || (pipeline_it->second.is_compute == false) || (buffer_it == owner->device._impl->buffers.end()) || (buffer_it->second.buffer == nil) || (_impl->command_buffer == nil)) {
    return;
  }

  if (ensure_compute_encoder(_impl) == nil) {
    return;
  }

  auto& pipeline = pipeline_it->second;
  if (pipeline.debug_name.empty() == false) {
    _impl->compute_encoder.label = [NSString stringWithFormat:@"ETX indirect compute dispatch: %s", pipeline.debug_name.c_str()];
  } else {
    _impl->compute_encoder.label = @"ETX indirect compute dispatch";
  }
  encode_stage_bindless_resources(pipeline.compute_stage, owner->bindless_manager._impl, owner->device._impl);
  declare_compute_stage_bindless_resources(_impl->compute_encoder, pipeline.compute_stage, owner->bindless_manager._impl, owner->device._impl);
  [_impl->compute_encoder setComputePipelineState:pipeline.compute_pipeline];
  for (uint32_t binding_index = 0; binding_index < kMetalBindlessBindingCount; ++binding_index) {
    if (pipeline.compute_stage.bindless_argument_buffers[binding_index] != nil) {
      [_impl->compute_encoder setBuffer:pipeline.compute_stage.bindless_argument_buffers[binding_index] offset:0 atIndex:pipeline.compute_stage.bindless_buffer_indices[binding_index]];
    }
  }
  if ((_impl->push_constants_size > 0u) && pipeline.compute_stage.uses_push_constants) {
    [_impl->compute_encoder setBytes:_impl->push_constants.data() length:_impl->push_constants_size atIndex:pipeline.compute_stage.push_constants_buffer_index];
  }

  const auto& shader = pipeline.compute_desc.compute_shader;
  const MTLSize threads_per_group = MTLSizeMake(std::max<uint32_t>(shader.local_size_x, 1u), std::max<uint32_t>(shader.local_size_y, 1u), std::max<uint32_t>(shader.local_size_z, 1u));
  [_impl->compute_encoder dispatchThreadgroupsWithIndirectBuffer:buffer_it->second.buffer indirectBufferOffset:static_cast<NSUInteger>(argument_buffer_offset) threadsPerThreadgroup:threads_per_group];
}

void MTCommandBuffer::copy_buffer(RHIBuffer src, RHIBuffer dst, uint64_t size, uint64_t src_offset, uint64_t dst_offset) {
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  auto src_it = owner->device._impl->buffers.find(src);
  auto dst_it = owner->device._impl->buffers.find(dst);
  if ((src_it == owner->device._impl->buffers.end()) || (dst_it == owner->device._impl->buffers.end()) || (_impl->command_buffer == nil)) {
    return;
  }
  [_impl->render_encoder endEncoding];
  _impl->render_encoder = nil;
  [_impl->compute_encoder endEncoding];
  _impl->compute_encoder = nil;
  if (_impl->blit_encoder == nil) {
    _impl->blit_encoder = [_impl->command_buffer blitCommandEncoder];
    _impl->blit_encoder.label = @"ETX copy buffer";
  }
  [_impl->blit_encoder copyFromBuffer:src_it->second.buffer sourceOffset:src_offset toBuffer:dst_it->second.buffer destinationOffset:dst_offset size:size];
}

void MTCommandBuffer::copy_buffer_to_texture(RHIBuffer src, RHITexture dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  auto src_it = owner->device._impl->buffers.find(src);
  auto dst_it = owner->device._impl->textures.find(dst);
  if ((src_it == owner->device._impl->buffers.end()) || (dst_it == owner->device._impl->textures.end()) || (_impl->command_buffer == nil)) {
    return;
  }
  const NSUInteger bytes_per_row = static_cast<NSUInteger>(width * format_bytes_per_pixel(dst_it->second.desc.format));
  [_impl->render_encoder endEncoding];
  _impl->render_encoder = nil;
  [_impl->compute_encoder endEncoding];
  _impl->compute_encoder = nil;
  if (_impl->blit_encoder == nil) {
    _impl->blit_encoder = [_impl->command_buffer blitCommandEncoder];
    _impl->blit_encoder.label = @"ETX copy buffer to texture";
  }
  [_impl->blit_encoder copyFromBuffer:src_it->second.buffer
                         sourceOffset:0
                    sourceBytesPerRow:bytes_per_row
                  sourceBytesPerImage:bytes_per_row * height
                           sourceSize:MTLSizeMake(width, height, 1)
                            toTexture:dst_it->second.texture
                     destinationSlice:0
                     destinationLevel:mip_level
                    destinationOrigin:MTLOriginMake(0, 0, 0)];
}

void MTCommandBuffer::copy_texture_to_buffer(RHITexture src, RHIBuffer dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  auto src_it = owner->device._impl->textures.find(src);
  auto dst_it = owner->device._impl->buffers.find(dst);
  if ((src_it == owner->device._impl->textures.end()) || (dst_it == owner->device._impl->buffers.end()) || (_impl->command_buffer == nil)) {
    return;
  }
  const NSUInteger bytes_per_row = static_cast<NSUInteger>(width * format_bytes_per_pixel(src_it->second.desc.format));
  [_impl->render_encoder endEncoding];
  _impl->render_encoder = nil;
  [_impl->compute_encoder endEncoding];
  _impl->compute_encoder = nil;
  if (_impl->blit_encoder == nil) {
    _impl->blit_encoder = [_impl->command_buffer blitCommandEncoder];
    _impl->blit_encoder.label = @"ETX copy texture to buffer";
  }
  [_impl->blit_encoder copyFromTexture:src_it->second.texture
                           sourceSlice:0
                           sourceLevel:mip_level
                          sourceOrigin:MTLOriginMake(0, 0, 0)
                            sourceSize:MTLSizeMake(width, height, 1)
                              toBuffer:dst_it->second.buffer
                     destinationOffset:0
                destinationBytesPerRow:bytes_per_row
              destinationBytesPerImage:bytes_per_row * height];
}

void MTCommandBuffer::set_debug_name(const char* name) {
  if ((_impl->command_buffer != nil) && (name != nullptr)) {
    _impl->command_buffer.label = [NSString stringWithUTF8String:name];
  }
}

RHIResult MTDevice::reload_graphics_pipeline(RHIPipeline pipeline, const RHIGraphicsPipelineDesc& new_desc) {
  const RHIResult destroy_result = destroy_pipeline(pipeline);
  if (destroy_result != RHIResult::Success) {
    return destroy_result;
  }
  return create_graphics_pipeline(new_desc).result;
}

RHIResult MTDevice::reload_compute_pipeline(RHIPipeline pipeline, const RHIComputePipelineDesc& new_desc) {
  const RHIResult destroy_result = destroy_pipeline(pipeline);
  if (destroy_result != RHIResult::Success) {
    return destroy_result;
  }
  return create_compute_pipeline(new_desc).result;
}

RHIMemoryStats MTDevice::get_memory_statistics() const {
  RHIMemoryStats stats = {};
  stats.gpu_allocated_bytes = _impl->gpu_allocated_bytes;
  if (_impl->metal_device != nil) {
    stats.gpu_device_local_allocated_bytes = _impl->gpu_allocated_bytes;
    stats.gpu_device_local_budget_bytes = [_impl->metal_device recommendedMaxWorkingSetSize];
  }
  return stats;
}

RHICreateBindlessResult MTDevice::create_acceleration_structure(const RHIAccelerationStructureDesc& desc) {
  if ((_impl->metal_device == nil) || (_impl->bindless_manager == nullptr)) {
    return {RHIResult::InvalidArgument, {}};
  }
  if (device_reports_raytracing(_impl->metal_device) == false) {
    return {RHIResult::UnsupportedFeature, {}};
  }

  MTLAccelerationStructureDescriptor* descriptor = nil;
  std::string error_message = {};
  if (desc.type == RHIAccelerationStructureType::BottomLevel) {
    descriptor = create_metal_blas_descriptor(desc.geometries, desc.geometry_count, _impl, &error_message);
  } else {
    if (desc.instance_count == 0u) {
      return {RHIResult::InvalidArgument, {}};
    }
    descriptor = create_metal_tlas_sizing_descriptor(desc.instance_count, desc.allow_update);
  }

  if (descriptor == nil) {
    if (error_message.empty() == false) {
      log::error("Metal RHI: failed to create acceleration-structure descriptor: %s", error_message.c_str());
    }
    return {RHIResult::InvalidArgument, {}};
  }

  const MTLAccelerationStructureSizes size_info = [_impl->metal_device accelerationStructureSizesWithDescriptor:descriptor];
  if ((size_info.accelerationStructureSize == 0u) || (size_info.buildScratchBufferSize == 0u)) {
    log::error("Metal RHI: acceleration-structure size query returned zero-sized allocation");
    return {RHIResult::ValidationError, {}};
  }

  id<MTLAccelerationStructure> acceleration_structure = [_impl->metal_device newAccelerationStructureWithSize:size_info.accelerationStructureSize];
  if (acceleration_structure == nil) {
    return {RHIResult::OutOfMemory, {}};
  }

  const uint64_t instance_descriptor_buffer_size = (desc.type == RHIAccelerationStructureType::TopLevel) ? static_cast<uint64_t>(desc.instance_count) * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor) : 0u;
  id<MTLBuffer> instance_descriptor_buffer = nil;
  if (instance_descriptor_buffer_size > 0u) {
    instance_descriptor_buffer = [_impl->metal_device newBufferWithLength:static_cast<NSUInteger>(instance_descriptor_buffer_size) options:MTLResourceStorageModeShared];
    if (instance_descriptor_buffer == nil) {
      [acceleration_structure release];
      return {RHIResult::OutOfMemory, {}};
    }
  }

  RHIBindlessHandle as_handle = {};
  const RHIResult register_result = _impl->bindless_manager->register_acceleration_structure((__bridge const void*)acceleration_structure, size_info.accelerationStructureSize, as_handle);
  if (register_result != RHIResult::Success) {
    [instance_descriptor_buffer release];
    [acceleration_structure release];
    return {register_result, {}};
  }

  const uint64_t build_scratch_size = desc.allow_update ? std::max(static_cast<uint64_t>(size_info.buildScratchBufferSize), static_cast<uint64_t>(size_info.refitScratchBufferSize)) : static_cast<uint64_t>(size_info.buildScratchBufferSize);
  _impl->acceleration_structures.emplace(as_handle, MTAccelerationStructureData{
                                                      .acceleration_structure = acceleration_structure,
                                                      .instance_descriptor_buffer = instance_descriptor_buffer,
                                                      .desc = desc,
                                                      .allocated_size = static_cast<uint64_t>(size_info.accelerationStructureSize),
                                                      .instance_descriptor_buffer_size = instance_descriptor_buffer_size,
                                                      .build_scratch_size = build_scratch_size,
                                                    });
  _impl->gpu_allocated_bytes += static_cast<uint64_t>(size_info.accelerationStructureSize) + instance_descriptor_buffer_size;
  return {RHIResult::Success, as_handle};
}

RHIResult MTDevice::destroy_acceleration_structure(RHIBindlessHandle as_handle) {
  if (!as_handle.valid()) {
    return RHIResult::Success;
  }
  auto it = _impl->acceleration_structures.find(as_handle);
  if (it == _impl->acceleration_structures.end()) {
    return RHIResult::Success;
  }
  _impl->bindless_manager->unregister_acceleration_structure(as_handle);
  _impl->gpu_allocated_bytes -= it->second.allocated_size + it->second.instance_descriptor_buffer_size;
  [it->second.instance_descriptor_buffer release];
  [it->second.acceleration_structure release];
  _impl->acceleration_structures.erase(it);
  return RHIResult::Success;
}

uint64_t MTDevice::get_acceleration_structure_device_address(RHIBindlessHandle as_handle) {
  return (_impl->acceleration_structures.find(as_handle) != _impl->acceleration_structures.end()) ? as_handle.value : 0u;
}

uint64_t MTDevice::get_acceleration_structure_build_scratch_size(RHIBindlessHandle as_handle) {
  const auto it = _impl->acceleration_structures.find(as_handle);
  return (it != _impl->acceleration_structures.end()) ? it->second.build_scratch_size : 0u;
}

void MTCommandBuffer::build_acceleration_structure(const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset) {
  auto* owner = static_cast<MTContext::Impl*>(_impl->owner);
  if ((owner == nullptr) || (_impl->command_buffer == nil)) {
    return;
  }

  auto as_it = owner->device._impl->acceleration_structures.find(desc.as_handle);
  auto scratch_it = owner->device._impl->buffers.find(scratch_buffer);
  if ((as_it == owner->device._impl->acceleration_structures.end()) || (scratch_it == owner->device._impl->buffers.end())) {
    log::error("Metal RHI: acceleration-structure build received invalid destination or scratch buffer handle");
    return;
  }

  [_impl->render_encoder endEncoding];
  _impl->render_encoder = nil;
  [_impl->compute_encoder endEncoding];
  _impl->compute_encoder = nil;
  [_impl->blit_encoder endEncoding];
  _impl->blit_encoder = nil;

  MTLAccelerationStructureDescriptor* descriptor = nil;
  std::string error_message = {};
  if (desc.type == RHIAccelerationStructureType::BottomLevel) {
    descriptor = create_metal_blas_descriptor(desc.geometries, desc.geometry_count, owner->device._impl, &error_message);
  } else {
    auto instance_buffer_it = owner->device._impl->buffers.find(desc.instance_buffer);
    if ((instance_buffer_it == owner->device._impl->buffers.end()) || (instance_buffer_it->second.buffer == nil) || (desc.instance_count == 0u)) {
      log::error("Metal RHI: TLAS build received an invalid instance buffer");
      return;
    }

    const NSUInteger source_stride = sizeof(RHIAccelerationStructureInstance);
    const NSUInteger source_size = static_cast<NSUInteger>(desc.instance_count) * source_stride;
    if (instance_buffer_it->second.buffer.length < source_size) {
      log::error("Metal RHI: TLAS instance source buffer is too small");
      return;
    }

    const NSUInteger descriptor_stride = sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
    const NSUInteger required_size = static_cast<NSUInteger>(desc.instance_count) * descriptor_stride;
    if ((as_it->second.instance_descriptor_buffer == nil) || (as_it->second.instance_descriptor_buffer.length < required_size)) {
      log::error("Metal RHI: TLAS translated instance descriptor buffer is too small");
      return;
    }

    auto* src_instances = static_cast<const RHIAccelerationStructureInstance*>(instance_buffer_it->second.buffer.contents);
    auto* dst_instances = static_cast<MTLAccelerationStructureUserIDInstanceDescriptor*>(as_it->second.instance_descriptor_buffer.contents);
    NSMutableArray<id<MTLAccelerationStructure>>* instanced_acceleration_structures = [NSMutableArray arrayWithCapacity:desc.instance_count];

    for (uint32_t i = 0; i < desc.instance_count; ++i) {
      const auto& src_instance = src_instances[i];
      RHIBindlessHandle referenced_handle = {.value = src_instance.acceleration_structure_reference};
      auto referenced_it = owner->device._impl->acceleration_structures.find(referenced_handle);
      if ((referenced_it == owner->device._impl->acceleration_structures.end()) || (referenced_it->second.acceleration_structure == nil)) {
        log::error("Metal RHI: TLAS instance %u references an invalid BLAS handle", i);
        return;
      }

      dst_instances[i].transformationMatrix = to_metal_transform(src_instance.transform);
      dst_instances[i].options = to_metal_instance_options(src_instance.flags);
      dst_instances[i].mask = src_instance.mask;
      dst_instances[i].intersectionFunctionTableOffset = src_instance.instance_shader_binding_table_record_offset;
      dst_instances[i].accelerationStructureIndex = i;
      dst_instances[i].userID = src_instance.instance_custom_index;
      [instanced_acceleration_structures addObject:referenced_it->second.acceleration_structure];
    }
    [as_it->second.instance_descriptor_buffer didModifyRange:NSMakeRange(0u, required_size)];

    MTLInstanceAccelerationStructureDescriptor* tlas_descriptor = [MTLInstanceAccelerationStructureDescriptor descriptor];
    tlas_descriptor.usage = desc.allow_update ? MTLAccelerationStructureUsageRefit : MTLAccelerationStructureUsageNone;
    tlas_descriptor.instanceDescriptorBuffer = as_it->second.instance_descriptor_buffer;
    tlas_descriptor.instanceDescriptorBufferOffset = 0u;
    tlas_descriptor.instanceDescriptorStride = descriptor_stride;
    tlas_descriptor.instanceCount = desc.instance_count;
    tlas_descriptor.instancedAccelerationStructures = instanced_acceleration_structures;
    tlas_descriptor.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;
    descriptor = tlas_descriptor;
  }

  if (descriptor == nil) {
    if (error_message.empty() == false) {
      log::error("Metal RHI: failed to create acceleration-structure build descriptor: %s", error_message.c_str());
    }
    return;
  }

  id<MTLAccelerationStructureCommandEncoder> as_encoder = [_impl->command_buffer accelerationStructureCommandEncoder];
  if (as_encoder == nil) {
    log::error("Metal RHI: failed to create acceleration-structure command encoder");
    return;
  }

  if (desc.update) {
    as_encoder.label = @"ETX refit acceleration structure";
    [as_encoder refitAccelerationStructure:as_it->second.acceleration_structure descriptor:descriptor destination:as_it->second.acceleration_structure scratchBuffer:scratch_it->second.buffer scratchBufferOffset:static_cast<NSUInteger>(scratch_offset)];
  } else {
    as_encoder.label = @"ETX build acceleration structure";
    [as_encoder buildAccelerationStructure:as_it->second.acceleration_structure descriptor:descriptor scratchBuffer:scratch_it->second.buffer scratchBufferOffset:static_cast<NSUInteger>(scratch_offset)];
  }
  [as_encoder endEncoding];
}

}  // namespace etx
