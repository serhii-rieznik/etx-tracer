#include <etx/rhi/vulkan/vk_rhi.hxx>
#include <etx/rhi/vulkan/vk_utils.hxx>
#include <etx/rhi/rhi_types.hxx>

#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>

#if ETX_PLATFORM_WINDOWS
# define VK_USE_PLATFORM_WIN32_KHR
# include <windows.h>
# include <Psapi.h>
#endif

#include <vulkan/vulkan.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
namespace etx {

namespace {

constexpr const char* kVK_KHR_portability_subset_extension_name = "VK_KHR_portability_subset";
constexpr uint64_t kVulkanPipelineCacheMaxBytes = 64ull * 1024ull * 1024ull;

std::filesystem::path vulkan_pipeline_cache_directory() {
  std::filesystem::path root(env().cache_folder());
  root /= "vulkan";
  return root;
}

std::filesystem::path vulkan_pipeline_cache_file_path(const VkPhysicalDeviceProperties& properties) {
  char file_name[128] = {};
  std::snprintf(file_name, sizeof(file_name), "pipeline_cache_%08x_%08x_%08x.bin", properties.vendorID, properties.deviceID, properties.driverVersion);
  return vulkan_pipeline_cache_directory() / file_name;
}

bool read_binary_file(const std::filesystem::path& path, std::vector<uint8_t>& out_data) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (stream.is_open() == false) {
    return false;
  }

  const std::streamoff size = stream.tellg();
  if (size <= 0) {
    return false;
  }
  if (static_cast<uint64_t>(size) > kVulkanPipelineCacheMaxBytes) {
    return false;
  }

  out_data.resize(static_cast<size_t>(size));
  stream.seekg(0, std::ios::beg);
  stream.read(reinterpret_cast<char*>(out_data.data()), static_cast<std::streamsize>(size));
  return stream.good();
}

bool write_binary_file_atomic(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec.value() != 0) {
    return false;
  }

  std::filesystem::path temp_path = path;
  temp_path += ".tmp";

  {
    std::ofstream stream(temp_path, std::ios::binary | std::ios::trunc);
    if (stream.is_open() == false) {
      return false;
    }

    if (data.empty() == false) {
      stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    if (stream.good() == false) {
      stream.close();
      std::filesystem::remove(temp_path, ec);
      return false;
    }
  }

#if ETX_PLATFORM_WINDOWS
  const std::wstring temp_path_text = temp_path.wstring();
  const std::wstring path_text = path.wstring();
  if (MoveFileExW(temp_path_text.c_str(), path_text.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
    std::filesystem::remove(temp_path, ec);
    return false;
  }
#else
  ec.clear();
  std::filesystem::rename(temp_path, path, ec);
  if (ec.value() != 0) {
    std::filesystem::remove(temp_path, ec);
    return false;
  }
#endif
  return true;
}

}

static VkSampleCountFlagBits convert_sample_count_to_vk(uint32_t sample_count) {
  switch (sample_count) {
    case 1u:
      return VK_SAMPLE_COUNT_1_BIT;
    case 2u:
      return VK_SAMPLE_COUNT_2_BIT;
    case 4u:
      return VK_SAMPLE_COUNT_4_BIT;
    case 8u:
      return VK_SAMPLE_COUNT_8_BIT;
    default:
      log::warning("Vulkan: unsupported sample count %u, falling back to 1x", sample_count);
      return VK_SAMPLE_COUNT_1_BIT;
  }
}

struct VKStagingBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped_ptr = nullptr;
  uint64_t capacity = 0;
  uint64_t frame_offsets[kRHIMaxFrames] = {0};  // Per-frame offsets for proper synchronization
  uint64_t alignment = 0;
  uint64_t per_frame_capacity = 0;  // Capacity allocated per frame

  void initialize(VkDevice device, VkPhysicalDevice physical_device, uint64_t size) {
    capacity = size;
    per_frame_capacity = capacity / kRHIMaxFrames;

    for (uint32_t i = 0; i < kRHIMaxFrames; ++i) {
      frame_offsets[i] = 0;
    }

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    alignment = properties.limits.minMemoryMapAlignment;

    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = capacity;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (etx_vk_call(vkCreateBuffer(device, &buffer_info, nullptr, &buffer)) != VK_SUCCESS) {
      return;
    }

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = mem_requirements.size;

    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    uint32_t memory_type_index = UINT32_MAX;
    VkMemoryPropertyFlags properties_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
      if ((mem_requirements.memoryTypeBits & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties_flags) == properties_flags) {
        memory_type_index = i;
        break;
      }
    }

    if (memory_type_index != UINT32_MAX) {
      alloc_info.memoryTypeIndex = memory_type_index;
      if (etx_vk_call(vkAllocateMemory(device, &alloc_info, nullptr, &memory)) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return;
      }
      if (etx_vk_call(vkBindBufferMemory(device, buffer, memory, 0)) != VK_SUCCESS) {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return;
      }
      if (etx_vk_call(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped_ptr)) != VK_SUCCESS) {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return;
      }
    }
  }

  void destroy(VkDevice device) {
    if (mapped_ptr != nullptr) {
      vkUnmapMemory(device, memory);
      mapped_ptr = nullptr;
    }
    if (memory != VK_NULL_HANDLE) {
      vkFreeMemory(device, memory, nullptr);
      memory = VK_NULL_HANDLE;
    }
    if (buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, buffer, nullptr);
      buffer = VK_NULL_HANDLE;
    }
  }

  // Frame-aware allocation to prevent data corruption
  bool allocate(uint32_t frame_index, uint64_t size, uint64_t& out_offset, void*& out_ptr) {
    if (frame_index >= kRHIMaxFrames) {
      log::error("Invalid frame index %u for staging buffer allocation", frame_index);
      return false;
    }

    // Calculate base offset for this frame's region
    uint64_t frame_base = frame_index * per_frame_capacity;
    uint64_t frame_end = frame_base + per_frame_capacity;

    // Get current offset within this frame's region
    uint64_t current_offset = frame_base + frame_offsets[frame_index];
    uint64_t aligned_offset = (current_offset + alignment - 1) & ~(alignment - 1);

    // Check if allocation fits in this frame's region
    if (aligned_offset + size > frame_end) {
      // Can't fit in this frame's region - caller should use transient staging buffer
      return false;
    }

    out_offset = aligned_offset;
    out_ptr = static_cast<uint8_t*>(mapped_ptr) + aligned_offset;

    // Update frame offset (relative to frame base)
    frame_offsets[frame_index] = (aligned_offset - frame_base) + size;

    return true;
  }

  // Reset a specific frame's region (called at frame start after fence wait)
  void reset_frame(uint32_t frame_index) {
    if (frame_index < kRHIMaxFrames) {
      frame_offsets[frame_index] = 0;
    }
  }
};

struct VKDevice::Impl {
  std::atomic<uint64_t> gpu_allocated_bytes = {0};
  bool memory_budget_supported = false;
  bool fill_mode_non_solid_supported = false;
  bool headless = false;
  bool buffer_device_address_supported = false;
  bool dynamic_rendering_supported = false;
  bool ray_tracing_supported = false;

  Impl(const RHIInitInfo& info);
  ~Impl();

  // Resource pool structures
  struct CommandBufferResource {
    VkCommandBuffer buffer = {};
    uint32_t pool_index = 0u;
    bool used = false;
  };

  struct FenceResource {
    VkFence fence = VK_NULL_HANDLE;
    bool used = false;
  };

  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;

  uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
  uint32_t compute_queue_family = VK_QUEUE_FAMILY_IGNORED;
  uint32_t transfer_queue_family = VK_QUEUE_FAMILY_IGNORED;

  VkQueue graphics_queue = VK_NULL_HANDLE;
  VkQueue compute_queue = VK_NULL_HANDLE;
  VkQueue transfer_queue = VK_NULL_HANDLE;

  VkPhysicalDeviceProperties properties = {};
  VkPhysicalDeviceMemoryProperties memory_properties = {};
  uint64_t min_uniform_buffer_offset_alignment = 0;
  uint64_t min_storage_buffer_offset_alignment = 0;

  bool bindless_supported = false;
  std::vector<const char*> enabled_extensions = {};
  std::vector<VkExtensionProperties> available_device_extensions = {};
  std::vector<VkExtensionProperties> available_instance_extensions = {};

  VKBindlessManager* bindless_manager = nullptr;

  std::unordered_map<RHIBindlessHandle, VkBuffer> buffer_map;

  // POD data structures (no pointers)
  VKResourcePool<VKBufferData, RHIBindlessHandle> buffers;
  VKResourcePool<VKTextureData, RHIBindlessHandle> textures;
  VKResourcePool<VKSamplerData, RHIBindlessHandle> samplers;
  VKResourcePool<VKPipelineData, RHIPipeline> compute_pipelines;
  VKResourcePool<VKPipelineData, RHIPipeline> graphics_pipelines;
  VKResourcePool<VKAccelerationStructureData, RHIBindlessHandle> acceleration_structures;
  VKResourcePool<VkSemaphore, RHISemaphore> semaphores;

  // Separate tracking for mapped buffers (since void* can't be in POD)
  std::unordered_map<RHIBindlessHandle, void*> mapped_buffer_ptrs;

  // Track acceleration structure buffers for destruction
  std::unordered_map<RHIBindlessHandle, RHIBindlessHandle> as_to_buffer_map;

  std::vector<VkCommandPool> command_pools;
  std::vector<CommandBufferResource> command_buffer_pool;

  // Reusable fence pool for staging operations
  std::vector<FenceResource> fence_pool;

  VKStagingBuffer staging_buffer = {};
  VkPipelineLayout bindless_layout = {};
  VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
  std::filesystem::path pipeline_cache_path = {};
  uint32_t max_push_constants_size = 128;

  // Frame tracking for staging buffer synchronization
  uint32_t current_frame_index = 0;

  // AS extension function pointers
  PFN_vkGetAccelerationStructureBuildSizesKHR impl_vkGetAccelerationStructureBuildSizesKHR = nullptr;
  PFN_vkCreateAccelerationStructureKHR impl_vkCreateAccelerationStructureKHR = nullptr;
  PFN_vkDestroyAccelerationStructureKHR impl_vkDestroyAccelerationStructureKHR = nullptr;
  PFN_vkGetAccelerationStructureDeviceAddressKHR impl_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
  PFN_vkCmdBuildAccelerationStructuresKHR impl_vkCmdBuildAccelerationStructuresKHR = nullptr;
  PFN_vkGetBufferDeviceAddress impl_vkGetBufferDeviceAddress = nullptr;

  RHIResult load_acceleration_structure_functions() {
    impl_vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
    if (impl_vkGetAccelerationStructureBuildSizesKHR == nullptr) {
      return RHIResult::UnsupportedFeature;
    }

    impl_vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
    if (impl_vkDestroyAccelerationStructureKHR == nullptr) {
      return RHIResult::UnsupportedFeature;
    }

    impl_vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
    if (impl_vkCreateAccelerationStructureKHR == nullptr) {
      return RHIResult::UnsupportedFeature;
    }

    impl_vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
    if (impl_vkGetAccelerationStructureDeviceAddressKHR == nullptr) {
      return RHIResult::UnsupportedFeature;
    }

    impl_vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
    if (impl_vkCmdBuildAccelerationStructuresKHR == nullptr) {
      return RHIResult::UnsupportedFeature;
    }

    impl_vkGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddress");
    if (impl_vkGetBufferDeviceAddress == nullptr) {
      return RHIResult::UnsupportedFeature;
    }

    return RHIResult::Success;
  }

  struct DeferredResource {
    enum class Type { Buffer, Texture, Sampler, AccelerationStructure, Pipeline, Shader } type;
    struct {
      VkBuffer buffer = VK_NULL_HANDLE;
      VkDeviceMemory memory = VK_NULL_HANDLE;
      VkImageView image_view = VK_NULL_HANDLE;
      VkImage image = VK_NULL_HANDLE;
      VkSampler sampler = VK_NULL_HANDLE;
      VkAccelerationStructureKHR as = VK_NULL_HANDLE;
      VkPipeline pipeline = VK_NULL_HANDLE;
      VkShaderModule shader = VK_NULL_HANDLE;
    } vk;
  };
  std::vector<DeferredResource> deferred_resources[kRHIMaxFrames];

  bool initialize_instance(const RHIInitInfo&);
  bool initialize_physical_device();
  bool initialize_device();
  void initialize_pipeline_cache();
  void save_pipeline_cache();
  void destroy_bindless_pipeline_layout();

  uint32_t find_queue_family(VkQueueFlags required_flags, VkQueueFlags avoid_flags = 0);
  bool check_instance_extension_support(const std::vector<const char*>& extensions);
  bool check_extension_support(const std::vector<const char*>& extensions);
  bool check_bindless_support();

  uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
  RHIResult allocate_memory(VkMemoryRequirements mem_requirements, VkMemoryPropertyFlags properties, VkMemoryAllocateFlags flags, VkDeviceMemory& out_memory);
  RHIResult create_vulkan_buffer(const RHIBufferDesc& desc, VkBuffer& out_buffer, VkDeviceMemory& out_memory);
  RHIResult create_vulkan_texture(const RHITextureDesc& desc, VkImage& out_image, VkDeviceMemory& out_memory);
  RHIResult create_vulkan_image_view(const RHITextureDesc& desc, VkImage image, VkImageView& out_view);
  RHIResult create_vulkan_sampler(const RHISamplerDesc& desc, VkSampler& out_sampler);
  RHIResult create_vulkan_graphics_pipeline(const RHIGraphicsPipelineDesc& desc, VkPipelineLayout layout, VkPipeline& out_pipeline);
  RHIResult create_vulkan_compute_pipeline(const RHIComputePipelineDesc& desc, VkPipelineLayout layout, VkPipeline& out_pipeline);
  RHIResult execute_single_time_commands(std::function<void(VkCommandBuffer)> recorder);

  RHICreateResult<VkPipelineLayout> get_bindless_pipeline_layout();

  uint32_t allocate_buffer_index();
  void free_buffer_index(uint32_t index);
  uint32_t allocate_texture_index();
  void free_texture_index(uint32_t index);
  uint32_t allocate_sampler_index();
  void free_sampler_index(uint32_t index);
  uint32_t allocate_acceleration_structure_index();
  void free_acceleration_structure_index(uint32_t index);
  uint32_t allocate_shader_slot();
  void free_shader_slot(uint32_t index);

  VkCommandBuffer acquire_command_buffer(uint32_t pool_index);
  void release_command_buffer(VkCommandBuffer command_buffer);
  VkFence acquire_fence();
  void release_fence(VkFence fence);

  bool initialize_pools();
  void cleanup_pools();

  // Frame-aware staging buffer management
  void set_current_frame_index(uint32_t index) {
    current_frame_index = index;
  }

  void reset_staging_buffer_for_frame(uint32_t frame_index) {
    staging_buffer.reset_frame(frame_index);
  }

  void queue_deferred_destruction(const VKBufferData& data) {
    DeferredResource res;
    res.type = DeferredResource::Type::Buffer;
    res.vk.buffer = data.buffer;
    res.vk.memory = data.memory;
    deferred_resources[current_frame_index].push_back(res);
  }

  void queue_deferred_destruction(const VKTextureData& data) {
    DeferredResource res;
    res.type = DeferredResource::Type::Texture;
    res.vk.image_view = data.image_view;
    res.vk.image = data.image;
    res.vk.memory = data.memory;
    deferred_resources[current_frame_index].push_back(res);
  }

  void queue_deferred_destruction(const VKSamplerData& data) {
    DeferredResource res;
    res.type = DeferredResource::Type::Sampler;
    res.vk.sampler = data.sampler;
    deferred_resources[current_frame_index].push_back(res);
  }

  void queue_deferred_destruction(const VKAccelerationStructureData& data) {
    DeferredResource res;
    res.type = DeferredResource::Type::AccelerationStructure;
    res.vk.as = data.acceleration_structure;
    deferred_resources[current_frame_index].push_back(res);
  }

  void queue_deferred_destruction(const VKPipelineData& data, bool is_compute) {
    DeferredResource res;
    res.type = DeferredResource::Type::Pipeline;
    res.vk.pipeline = data.pipeline;
    deferred_resources[current_frame_index].push_back(res);
  }

  void queue_deferred_destruction(VkShaderModule shader) {
    DeferredResource res;
    res.type = DeferredResource::Type::Shader;
    res.vk.shader = shader;
    deferred_resources[current_frame_index].push_back(res);
  }

  void process_deferred_destruction(uint32_t frame_index) {
    if (frame_index >= kRHIMaxFrames)
      return;

    auto& resources = deferred_resources[frame_index];
    for (auto& res : resources) {
      switch (res.type) {
        case DeferredResource::Type::Buffer: {
          if (res.vk.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, res.vk.buffer, nullptr);
          }
          if (res.vk.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, res.vk.memory, nullptr);
          }
          break;
        }
        case DeferredResource::Type::Texture: {
          if (res.vk.image_view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, res.vk.image_view, nullptr);
          }
          if (res.vk.image != VK_NULL_HANDLE) {
            vkDestroyImage(device, res.vk.image, nullptr);
          }
          if (res.vk.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, res.vk.memory, nullptr);
          }
          break;
        }
        case DeferredResource::Type::Sampler: {
          if (res.vk.sampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, res.vk.sampler, nullptr);
          }
          break;
        }
        case DeferredResource::Type::AccelerationStructure: {
          if (res.vk.as != VK_NULL_HANDLE) {
            impl_vkDestroyAccelerationStructureKHR(device, res.vk.as, nullptr);
          }
          break;
        }
        case DeferredResource::Type::Pipeline: {
          if (res.vk.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, res.vk.pipeline, nullptr);
          }
          break;
        }
        case DeferredResource::Type::Shader: {
          if (res.vk.shader != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, res.vk.shader, nullptr);
          }
          break;
        }
      }
    }
    resources.clear();
  }
};

VKDevice::Impl::Impl(const RHIInitInfo& info) {
  headless = info.headless;

  if (initialize_instance(info) == false) {
    log::error("Failed to initialize Vulkan instance");
    return;
  }

  if (initialize_physical_device() == false) {
    log::error("Failed to initialize physical device");
    return;
  }

  if (initialize_device() == false) {
    log::error("Failed to initialize device");
    return;
  }
  initialize_pipeline_cache();

  if (ray_tracing_supported) {
    if (load_acceleration_structure_functions() != RHIResult::Success) {
      log::warning("Failed to load Vulkan ray tracing functions; disabling GPU ray tracing support");
      ray_tracing_supported = false;
    }
  }

  staging_buffer.initialize(device, physical_device, 64 * 1024 * 1024);  // 64 MB staging buffer
}

VKDevice::Impl::~Impl() {
  bindless_manager = nullptr;

  cleanup_pools();

  for (uint32_t i = 0; i < kRHIMaxFrames; ++i) {
    process_deferred_destruction(i);
  }

  staging_buffer.destroy(device);

  destroy_bindless_pipeline_layout();

  for (auto& command_pool : command_pools) {
    if ((command_pool != VK_NULL_HANDLE) && (device != VK_NULL_HANDLE)) {
      vkDestroyCommandPool(device, command_pool, nullptr);
      command_pool = VK_NULL_HANDLE;
    }
  }

  save_pipeline_cache();
  if (pipeline_cache != VK_NULL_HANDLE) {
    vkDestroyPipelineCache(device, pipeline_cache, nullptr);
    pipeline_cache = VK_NULL_HANDLE;
  }

  if (device != VK_NULL_HANDLE) {
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
  }

  if (instance != VK_NULL_HANDLE) {
    vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
  }
}

bool VKDevice::Impl::initialize_instance(const RHIInitInfo& init_info) {
  uint32_t extension_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
  available_instance_extensions.resize(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available_instance_extensions.data());

  VkApplicationInfo app_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app_info.pApplicationName = "etx-tracer";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "etx-rhi";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_3;

  std::vector<const char*> extensions = {
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
  };
#if ETX_PLATFORM_APPLE
  extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif
  if (init_info.headless == false) {
    extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#if ETX_PLATFORM_WINDOWS
    extensions.push_back("VK_KHR_win32_surface");
#elif ETX_PLATFORM_APPLE
    extensions.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
#endif
  }

  if (check_instance_extension_support(extensions) == false) {
    log::error("Required instance extensions not supported");
    return false;
  }

  VkInstanceCreateInfo create_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();
#if ETX_PLATFORM_APPLE
  create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

  std::vector<const char*> layers = {};
  if (init_info.enable_validation) {
    layers.emplace_back("VK_LAYER_KHRONOS_validation");
  }

  create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
  create_info.ppEnabledLayerNames = layers.data();

  if (etx_vk_call(vkCreateInstance(&create_info, nullptr, &instance)) != VK_SUCCESS) {
    return false;
  }

  return true;
}

bool VKDevice::Impl::initialize_physical_device() {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

  if (device_count == 0) {
    log::error("No Vulkan-compatible physical devices found");
    return false;
  }

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

  VkPhysicalDevice selected_device = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties device_properties = {};
  for (const auto& device : devices) {
    vkGetPhysicalDeviceProperties(device, &device_properties);

    if (device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      selected_device = device;
      break;
    }

    if (selected_device == VK_NULL_HANDLE) {
      selected_device = device;
    }
  }

  if (selected_device == VK_NULL_HANDLE) {
    log::error("No suitable physical device found");
    return false;
  }

  physical_device = selected_device;
  vkGetPhysicalDeviceProperties(physical_device, &properties);
  vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

  min_uniform_buffer_offset_alignment = properties.limits.minUniformBufferOffsetAlignment;
  min_storage_buffer_offset_alignment = properties.limits.minStorageBufferOffsetAlignment;
  max_push_constants_size = properties.limits.maxPushConstantsSize;

  return true;
}

bool VKDevice::Impl::initialize_device() {
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

  for (uint32_t i = 0; i < queue_family_count; ++i) {
    if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      graphics_queue_family = i;
      break;
    }
  }

  if (graphics_queue_family == VK_QUEUE_FAMILY_IGNORED) {
    log::error("No graphics queue family found");
    return false;
  }

  for (uint32_t i = 0; i < queue_family_count; ++i) {
    if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      compute_queue_family = i;
      break;
    }
  }

  if (compute_queue_family == VK_QUEUE_FAMILY_IGNORED) {
    compute_queue_family = graphics_queue_family;
  }

  for (uint32_t i = 0; i < queue_family_count; ++i) {
    if ((queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(queue_families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
      transfer_queue_family = i;
      break;
    }
  }

  if (transfer_queue_family == VK_QUEUE_FAMILY_IGNORED) {
    transfer_queue_family = compute_queue_family;
  }

  std::set<uint32_t> unique_families = {graphics_queue_family, compute_queue_family, transfer_queue_family};

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  float queue_priority = 1.0f;

  for (uint32_t family : unique_families) {
    VkDeviceQueueCreateInfo queue_info = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_info);
  }

  uint32_t extension_count = 0;
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
  available_device_extensions.resize(extension_count);
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, available_device_extensions.data());

  auto has_extension = [&](const char* name) {
    for (const auto& ext : available_device_extensions) {
      if (strcmp(ext.extensionName, name) == 0) {
        return true;
      }
    }
    return false;
  };

  std::vector<const char*> device_extensions = {
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
    VK_KHR_MAINTENANCE_3_EXTENSION_NAME,
  };
  if (headless == false) {
    device_extensions.insert(device_extensions.begin(), VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  const bool memory_budget_extension_available = has_extension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
  const bool dynamic_rendering_extension_available = has_extension(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
  const bool buffer_device_address_extension_available = has_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  const bool acceleration_structure_extension_available = has_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
  const bool deferred_host_operations_extension_available = has_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  const bool portability_subset_extension_available = has_extension(kVK_KHR_portability_subset_extension_name);
  const bool ray_query_extension_available = has_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME);

  if (memory_budget_extension_available) {
    device_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    memory_budget_supported = true;
  }
  if (dynamic_rendering_extension_available) {
    device_extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
  }
  if (portability_subset_extension_available) {
    device_extensions.push_back(kVK_KHR_portability_subset_extension_name);
  }

  if (!check_extension_support(device_extensions)) {
    log::error("Required device extensions not supported");
    return false;
  }

  VkPhysicalDeviceFeatures device_features = {};
  VkPhysicalDeviceFeatures physical_device_features = {};
  vkGetPhysicalDeviceFeatures(physical_device, &physical_device_features);
  device_features.samplerAnisotropy = VK_TRUE;
  device_features.shaderInt64 = VK_TRUE;
  fill_mode_non_solid_supported = (physical_device_features.fillModeNonSolid == VK_TRUE);
  device_features.fillModeNonSolid = fill_mode_non_solid_supported ? VK_TRUE : VK_FALSE;

  VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
  VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
  VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures demote_to_helper_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES};

  VkPhysicalDeviceFeatures2 supported_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  supported_features.pNext = &descriptor_indexing_features;
  descriptor_indexing_features.pNext = &dynamic_rendering_features;
  dynamic_rendering_features.pNext = &demote_to_helper_features;
  demote_to_helper_features.pNext = &ray_query_features;
  ray_query_features.pNext = &acceleration_structure_features;
  acceleration_structure_features.pNext = &buffer_device_address_features;
  vkGetPhysicalDeviceFeatures2(physical_device, &supported_features);

  const bool descriptor_indexing_supported =
    (descriptor_indexing_features.runtimeDescriptorArray == VK_TRUE) && (descriptor_indexing_features.descriptorBindingPartiallyBound == VK_TRUE) &&
    (descriptor_indexing_features.descriptorBindingVariableDescriptorCount == VK_TRUE) && (descriptor_indexing_features.descriptorBindingStorageBufferUpdateAfterBind == VK_TRUE) &&
    (descriptor_indexing_features.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE) &&
    (descriptor_indexing_features.descriptorBindingStorageImageUpdateAfterBind == VK_TRUE);
  if (descriptor_indexing_supported == false) {
    log::error("Required descriptor indexing features are not supported");
    return false;
  }

  dynamic_rendering_supported = (dynamic_rendering_features.dynamicRendering == VK_TRUE);
  if (dynamic_rendering_supported == false) {
    log::error("Required dynamic rendering feature is not supported");
    return false;
  }

  const bool shader_demote_supported = (demote_to_helper_features.shaderDemoteToHelperInvocation == VK_TRUE);
  buffer_device_address_supported = buffer_device_address_extension_available && (buffer_device_address_features.bufferDeviceAddress == VK_TRUE);
  ray_tracing_supported = buffer_device_address_supported && acceleration_structure_extension_available && deferred_host_operations_extension_available &&
                          ray_query_extension_available && (acceleration_structure_features.accelerationStructure == VK_TRUE) && (ray_query_features.rayQuery == VK_TRUE);

  if (buffer_device_address_supported) {
    device_extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  }
  if (ray_tracing_supported) {
    device_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  }

  if (!check_extension_support(device_extensions)) {
    log::error("Required Vulkan device extensions are not supported after optional feature selection");
    return false;
  }

  buffer_device_address_features.bufferDeviceAddress = buffer_device_address_supported ? VK_TRUE : VK_FALSE;
  acceleration_structure_features.accelerationStructure = ray_tracing_supported ? VK_TRUE : VK_FALSE;
  acceleration_structure_features.descriptorBindingAccelerationStructureUpdateAfterBind = ray_tracing_supported ? VK_TRUE : VK_FALSE;
  ray_query_features.rayQuery = ray_tracing_supported ? VK_TRUE : VK_FALSE;
  descriptor_indexing_features.runtimeDescriptorArray = VK_TRUE;
  descriptor_indexing_features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
  descriptor_indexing_features.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
  descriptor_indexing_features.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
  descriptor_indexing_features.descriptorBindingPartiallyBound = VK_TRUE;
  descriptor_indexing_features.descriptorBindingVariableDescriptorCount = VK_TRUE;
  dynamic_rendering_features.dynamicRendering = VK_TRUE;
  demote_to_helper_features.shaderDemoteToHelperInvocation = shader_demote_supported ? VK_TRUE : VK_FALSE;

  VkDeviceCreateInfo device_create_info = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  void* feature_chain = nullptr;
  if (buffer_device_address_supported) {
    buffer_device_address_features.pNext = feature_chain;
    feature_chain = &buffer_device_address_features;
  }
  if (ray_tracing_supported) {
    acceleration_structure_features.pNext = feature_chain;
    feature_chain = &acceleration_structure_features;
    ray_query_features.pNext = feature_chain;
    feature_chain = &ray_query_features;
  }
  descriptor_indexing_features.pNext = feature_chain;
  feature_chain = &descriptor_indexing_features;
  dynamic_rendering_features.pNext = feature_chain;
  feature_chain = &dynamic_rendering_features;
  if (shader_demote_supported) {
    demote_to_helper_features.pNext = feature_chain;
    feature_chain = &demote_to_helper_features;
  }
  device_create_info.pNext = feature_chain;
  device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
  device_create_info.pQueueCreateInfos = queue_create_infos.data();
  device_create_info.pEnabledFeatures = &device_features;
  device_create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
  device_create_info.ppEnabledExtensionNames = device_extensions.data();

  if (etx_vk_call(vkCreateDevice(physical_device, &device_create_info, nullptr, &device)) != VK_SUCCESS) {
    return false;
  }

  vkGetDeviceQueue(device, graphics_queue_family, 0, &graphics_queue);
  vkGetDeviceQueue(device, compute_queue_family, 0, &compute_queue);
  vkGetDeviceQueue(device, transfer_queue_family, 0, &transfer_queue);

  VkCommandPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.queueFamilyIndex = graphics_queue_family;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  command_pools.resize(kRHIMaxFrames + 1u);
  for (uint32_t i = 0; i <= kRHIMaxFrames; ++i) {
    if (etx_vk_call(vkCreateCommandPool(device, &pool_info, nullptr, command_pools.data() + i)) != VK_SUCCESS) {
      return false;
    }
  }

  if (!initialize_pools()) {
    log::error("Failed to initialize command buffer and fence pools");
    return false;
  }

  bindless_supported = check_bindless_support();
  enabled_extensions = device_extensions;
  return true;
}

void VKDevice::Impl::initialize_pipeline_cache() {
  pipeline_cache_path = vulkan_pipeline_cache_file_path(properties);

  std::vector<uint8_t> cache_data = {};
  std::error_code ec;
  if ((std::filesystem::exists(pipeline_cache_path, ec) && (ec.value() == 0)) && (read_binary_file(pipeline_cache_path, cache_data) == false)) {
    std::filesystem::remove(pipeline_cache_path, ec);
    cache_data.clear();
  }

  VkPipelineCacheCreateInfo cache_info = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  if (cache_data.empty() == false) {
    cache_info.initialDataSize = cache_data.size();
    cache_info.pInitialData = cache_data.data();
  }

  VkResult result = vkCreatePipelineCache(device, &cache_info, nullptr, &pipeline_cache);
  if ((result != VK_SUCCESS) && (cache_data.empty() == false)) {
    std::filesystem::remove(pipeline_cache_path, ec);
    cache_info.initialDataSize = 0u;
    cache_info.pInitialData = nullptr;
    cache_data.clear();
    result = vkCreatePipelineCache(device, &cache_info, nullptr, &pipeline_cache);
  }

  if (result != VK_SUCCESS) {
    pipeline_cache = VK_NULL_HANDLE;
    log::warning("Vulkan: failed to create pipeline cache (%s)", vk_error_to_string(result));
  }
}

void VKDevice::Impl::save_pipeline_cache() {
  if ((device == VK_NULL_HANDLE) || (pipeline_cache == VK_NULL_HANDLE) || (pipeline_cache_path.empty())) {
    return;
  }

  size_t cache_size = 0u;
  VkResult result = vkGetPipelineCacheData(device, pipeline_cache, &cache_size, nullptr);
  if ((result != VK_SUCCESS) || (cache_size == 0u)) {
    return;
  }
  if (static_cast<uint64_t>(cache_size) > kVulkanPipelineCacheMaxBytes) {
    log::warning("Vulkan: skipping oversized pipeline cache write %.2fMB", static_cast<double>(cache_size) / (1024.0 * 1024.0));
    return;
  }

  std::vector<uint8_t> cache_data(cache_size);
  result = vkGetPipelineCacheData(device, pipeline_cache, &cache_size, cache_data.data());
  if (result != VK_SUCCESS) {
    return;
  }

  cache_data.resize(cache_size);
  if (write_binary_file_atomic(pipeline_cache_path, cache_data) == false) {
    log::warning("Vulkan: failed to write pipeline cache %s", pipeline_cache_path.generic_string().c_str());
  }
}

void VKDevice::Impl::destroy_bindless_pipeline_layout() {
  if ((device != VK_NULL_HANDLE) && (bindless_layout != VK_NULL_HANDLE)) {
    vkDestroyPipelineLayout(device, bindless_layout, nullptr);
    bindless_layout = VK_NULL_HANDLE;
  }
}

bool VKDevice::Impl::check_instance_extension_support(const std::vector<const char*>& extensions) {
  for (const char* required : extensions) {
    bool found = false;
    for (const auto& available : available_instance_extensions) {
      if (strcmp(required, available.extensionName) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      log::error("Required instance extension not supported: %s", required);
      return false;
    }
  }

  return true;
}

bool VKDevice::Impl::check_extension_support(const std::vector<const char*>& extensions) {
  uint32_t extension_count;
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);

  std::vector<VkExtensionProperties> available_extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, available_extensions.data());

  for (const char* required : extensions) {
    bool found = false;
    for (const auto& available : available_extensions) {
      if (strcmp(required, available.extensionName) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      log::error("Required device extension not supported: %s", required);
      return false;
    }
  }

  return true;
}

bool VKDevice::Impl::check_bindless_support() {
  VkPhysicalDeviceDescriptorIndexingProperties descriptor_indexing_props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};

  VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &descriptor_indexing_props;

  vkGetPhysicalDeviceProperties2(physical_device, &props2);

  return descriptor_indexing_props.maxUpdateAfterBindDescriptorsInAllPools > 0 && descriptor_indexing_props.maxPerStageDescriptorUpdateAfterBindSamplers > 0;
}

uint32_t VKDevice::Impl::find_queue_family(VkQueueFlags required_flags, VkQueueFlags avoid_flags) {
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

  for (uint32_t i = 0; i < queue_family_count; ++i) {
    if ((queue_families[i].queueFlags & required_flags) == required_flags && (queue_families[i].queueFlags & avoid_flags) == 0) {
      return i;
    }
  }

  return VK_QUEUE_FAMILY_IGNORED;
}

uint32_t VKDevice::Impl::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  log::error("Failed to find suitable memory type");
  return UINT32_MAX;
}

RHIResult VKDevice::Impl::allocate_memory(VkMemoryRequirements requirements, VkMemoryPropertyFlags properties, VkMemoryAllocateFlags flags, VkDeviceMemory& out_memory) {
  VkMemoryAllocateFlagsInfo flags_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags_info.flags = flags;

  VkMemoryAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  alloc_info.pNext = (flags != 0) ? &flags_info : nullptr;
  alloc_info.allocationSize = requirements.size;
  alloc_info.memoryTypeIndex = find_vulkan_memory_type(physical_device, requirements.memoryTypeBits, properties);

  if (alloc_info.memoryTypeIndex == UINT32_MAX) {
    log::error("Failed to find suitable memory type");
    return RHIResult::OutOfMemory;
  }

  if (etx_vk_call(vkAllocateMemory(device, &alloc_info, nullptr, &out_memory)) != VK_SUCCESS) {
    return RHIResult::OutOfMemory;
  }

  gpu_allocated_bytes += requirements.size;
  return RHIResult::Success;
}

RHIResult VKDevice::Impl::create_vulkan_buffer(const RHIBufferDesc& desc, VkBuffer& out_buffer, VkDeviceMemory& out_memory) {
  VkBufferUsageFlags vk_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  using BufferUsage = std::underlying_type<RHIBufferUsage>::type;
  BufferUsage usage = static_cast<BufferUsage>(desc.usage);

  const bool requires_ray_tracing_usage = (usage & static_cast<BufferUsage>(RHIBufferUsage::AccelerationStructureBuild)) ||
                                          (usage & static_cast<BufferUsage>(RHIBufferUsage::AccelerationStructureStorage)) ||
                                          (usage & static_cast<BufferUsage>(RHIBufferUsage::ShaderBindingTable));
  if (requires_ray_tracing_usage && (ray_tracing_supported == false)) {
    log::error("Requested Vulkan ray tracing buffer usage on a device without ray tracing support");
    return RHIResult::UnsupportedFeature;
  }
  if ((usage & static_cast<BufferUsage>(RHIBufferUsage::ShaderDeviceAddress)) && (buffer_device_address_supported == false)) {
    log::error("Requested Vulkan shader device address on a device without buffer device address support");
    return RHIResult::UnsupportedFeature;
  }

  if (usage & static_cast<BufferUsage>(RHIBufferUsage::Vertex)) {
    vk_usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  if (usage & static_cast<BufferUsage>(RHIBufferUsage::Index)) {
    vk_usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }
  if (usage & static_cast<BufferUsage>(RHIBufferUsage::Uniform)) {
    vk_usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  }
  if (usage & static_cast<BufferUsage>(RHIBufferUsage::TransferSrc)) {
    vk_usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if (usage & static_cast<BufferUsage>(RHIBufferUsage::TransferDst)) {
    vk_usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
  if (usage & static_cast<BufferUsage>(RHIBufferUsage::AccelerationStructureBuild)) {
    vk_usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }
  if (usage & static_cast<BufferUsage>(RHIBufferUsage::AccelerationStructureStorage)) {
    vk_usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
  }
  if (usage & static_cast<BufferUsage>(RHIBufferUsage::ShaderBindingTable)) {
    vk_usage |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
  }
  if (usage & static_cast<BufferUsage>(RHIBufferUsage::ShaderDeviceAddress)) {
    vk_usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  buffer_info.size = desc.size;
  buffer_info.usage = vk_usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (etx_vk_call(vkCreateBuffer(device, &buffer_info, nullptr, &out_buffer)) != VK_SUCCESS) {
    return RHIResult::OutOfMemory;
  }

  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(device, out_buffer, &mem_requirements);

  VkMemoryPropertyFlags mem_props = desc.host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  VkMemoryAllocateFlags mem_flags = 0;
  if (vk_usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    mem_flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  }

  RHIResult alloc_result = allocate_memory(mem_requirements, mem_props, mem_flags, out_memory);
  if (alloc_result != RHIResult::Success) {
    log::error("Failed to allocate memory for buffer");
    vkDestroyBuffer(device, out_buffer, nullptr);
    out_buffer = VK_NULL_HANDLE;
    return RHIResult::OutOfMemory;
  }

  if (etx_vk_call(vkBindBufferMemory(device, out_buffer, out_memory, 0)) != VK_SUCCESS) {
    vkFreeMemory(device, out_memory, nullptr);
    vkDestroyBuffer(device, out_buffer, nullptr);
    out_buffer = VK_NULL_HANDLE;
    out_memory = VK_NULL_HANDLE;
    return RHIResult::ValidationError;
  }

  return RHIResult::Success;
}

static VkCompareOp convert_compare_op(RHICompareOp op) {
  switch (op) {
    case RHICompareOp::Never:
      return VK_COMPARE_OP_NEVER;
    case RHICompareOp::Less:
      return VK_COMPARE_OP_LESS;
    case RHICompareOp::Equal:
      return VK_COMPARE_OP_EQUAL;
    case RHICompareOp::LessOrEqual:
      return VK_COMPARE_OP_LESS_OR_EQUAL;
    case RHICompareOp::Greater:
      return VK_COMPARE_OP_GREATER;
    case RHICompareOp::NotEqual:
      return VK_COMPARE_OP_NOT_EQUAL;
    case RHICompareOp::GreaterOrEqual:
      return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case RHICompareOp::Always:
      return VK_COMPARE_OP_ALWAYS;
    default:
      return VK_COMPARE_OP_LESS;
  }
}

static VkBlendFactor convert_blend_factor(RHIBlendFactor factor) {
  switch (factor) {
    case RHIBlendFactor::Zero:
      return VK_BLEND_FACTOR_ZERO;
    case RHIBlendFactor::One:
      return VK_BLEND_FACTOR_ONE;
    case RHIBlendFactor::SrcColor:
      return VK_BLEND_FACTOR_SRC_COLOR;
    case RHIBlendFactor::OneMinusSrcColor:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case RHIBlendFactor::DstColor:
      return VK_BLEND_FACTOR_DST_COLOR;
    case RHIBlendFactor::OneMinusDstColor:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case RHIBlendFactor::SrcAlpha:
      return VK_BLEND_FACTOR_SRC_ALPHA;
    case RHIBlendFactor::OneMinusSrcAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case RHIBlendFactor::DstAlpha:
      return VK_BLEND_FACTOR_DST_ALPHA;
    case RHIBlendFactor::OneMinusDstAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case RHIBlendFactor::ConstantColor:
      return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case RHIBlendFactor::OneMinusConstantColor:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case RHIBlendFactor::ConstantAlpha:
      return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case RHIBlendFactor::OneMinusConstantAlpha:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    case RHIBlendFactor::SrcAlphaSaturate:
      return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    default:
      return VK_BLEND_FACTOR_ONE;
  }
}

static VkBlendOp convert_blend_op(RHIBlendOp op) {
  switch (op) {
    case RHIBlendOp::Add:
      return VK_BLEND_OP_ADD;
    case RHIBlendOp::Subtract:
      return VK_BLEND_OP_SUBTRACT;
    case RHIBlendOp::ReverseSubtract:
      return VK_BLEND_OP_REVERSE_SUBTRACT;
    case RHIBlendOp::Min:
      return VK_BLEND_OP_MIN;
    case RHIBlendOp::Max:
      return VK_BLEND_OP_MAX;
    default:
      return VK_BLEND_OP_ADD;
  }
}

RHICreateResult<VkPipelineLayout> VKDevice::Impl::get_bindless_pipeline_layout() {
  if (bindless_layout)
    return {RHIResult::Success, bindless_layout};

  VkPipelineLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.setLayoutCount = 1;

  VkDescriptorSetLayout ds_set_layout = static_cast<VKBindlessManager*>(bindless_manager)->get_descriptor_set_layout();
  layout_info.pSetLayouts = &ds_set_layout;

  VkPushConstantRange push_constants = {};
  push_constants.stageFlags = VK_SHADER_STAGE_ALL;
  push_constants.offset = 0;
  push_constants.size = max_push_constants_size;

  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push_constants;

  if (etx_vk_call(vkCreatePipelineLayout(device, &layout_info, nullptr, &bindless_layout)) != VK_SUCCESS) {
    return {RHIResult::ValidationError, {}};
  }

  return {RHIResult::Success, bindless_layout};
}

RHIResult VKDevice::Impl::create_vulkan_graphics_pipeline(const RHIGraphicsPipelineDesc& desc, VkPipelineLayout layout, VkPipeline& out_pipeline) {
  if (desc.vertex_shader.spirv_size == 0)
    return RHIResult::ValidationError;

  if (desc.fragment_shader.spirv_size == 0)
    return RHIResult::ValidationError;

  // Create vertex shader module from SPIR-V
  VkShaderModuleCreateInfo vert_info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  vert_info.codeSize = desc.vertex_shader.spirv_size;
  vert_info.pCode = reinterpret_cast<const uint32_t*>(desc.vertex_shader.spirv_data);
  VkShaderModule vert_module = {};
  if (etx_vk_call(vkCreateShaderModule(device, &vert_info, nullptr, &vert_module)) != VK_SUCCESS) {
    return RHIResult::ValidationError;
  }

  // Create fragment shader module from SPIR-V
  VkShaderModuleCreateInfo frag_info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  frag_info.codeSize = desc.fragment_shader.spirv_size;
  frag_info.pCode = reinterpret_cast<const uint32_t*>(desc.fragment_shader.spirv_data);
  VkShaderModule frag_module = {};
  if (etx_vk_call(vkCreateShaderModule(device, &frag_info, nullptr, &frag_module)) != VK_SUCCESS) {
    vkDestroyShaderModule(device, vert_module, nullptr);
    return RHIResult::ValidationError;
  }

  VkPipelineShaderStageCreateInfo shader_stages[] = {
    {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vert_module,
      .pName = desc.vertex_shader.entry_point.c_str(),
    },
    {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = frag_module,
      .pName = desc.fragment_shader.entry_point.c_str(),
    },
  };

  std::vector<VkVertexInputBindingDescription> vertex_bindings;
  for (uint32_t i = 0; i < desc.vertex_binding_count; ++i) {
    const auto& binding = desc.vertex_bindings[i];
    VkVertexInputBindingDescription vk_binding = {};
    vk_binding.binding = binding.binding;
    vk_binding.stride = binding.stride;
    vk_binding.inputRate = binding.input_rate == RHIVertexInputRate::Vertex ? VK_VERTEX_INPUT_RATE_VERTEX : VK_VERTEX_INPUT_RATE_INSTANCE;
    vertex_bindings.push_back(vk_binding);
  }

  std::vector<VkVertexInputAttributeDescription> vertex_attributes;
  for (uint32_t i = 0; i < desc.vertex_attribute_count; ++i) {
    const auto& attr = desc.vertex_attributes[i];
    VkVertexInputAttributeDescription vk_attr = {};
    vk_attr.location = attr.location;
    vk_attr.binding = attr.binding;
    switch (attr.format) {
      case RHIVertexFormat::Float2:
        vk_attr.format = VK_FORMAT_R32G32_SFLOAT;
        break;
      case RHIVertexFormat::Float3:
        vk_attr.format = VK_FORMAT_R32G32B32_SFLOAT;
        break;
      case RHIVertexFormat::Float4:
        vk_attr.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        break;
      default:
        vk_attr.format = VK_FORMAT_R32G32_SFLOAT;
        break;
    }
    vk_attr.offset = attr.offset;
    vertex_attributes.push_back(vk_attr);
  }

  VkPipelineVertexInputStateCreateInfo vertex_input = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vertex_input.vertexBindingDescriptionCount = static_cast<uint32_t>(vertex_bindings.size());
  vertex_input.pVertexBindingDescriptions = vertex_bindings.data();
  vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_attributes.size());
  vertex_input.pVertexAttributeDescriptions = vertex_attributes.data();

  VkPipelineInputAssemblyStateCreateInfo input_assembly = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  if (desc.primitive_topology == RHIPrimitiveTopology::LineList) {
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  } else {
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  }
  input_assembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewport_state = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterizer = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rasterizer.depthClampEnable = desc.rasterization.depth_clamp_enable ? VK_TRUE : VK_FALSE;
  rasterizer.rasterizerDiscardEnable = desc.rasterization.rasterizer_discard_enable ? VK_TRUE : VK_FALSE;
  if (desc.rasterization.wireframe_enable && (fill_mode_non_solid_supported == false)) {
    log::warning("Vulkan: Wireframe pipeline requested, but fillModeNonSolid is not supported. Falling back to filled triangles.");
  }
  bool use_wireframe = desc.rasterization.wireframe_enable && fill_mode_non_solid_supported;
  rasterizer.polygonMode = use_wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = desc.rasterization.line_width;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineDepthStencilStateCreateInfo depth_state = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  depth_state.depthTestEnable = desc.depth_state.depth_test_enable ? VK_TRUE : VK_FALSE;
  depth_state.depthWriteEnable = desc.depth_state.depth_write_enable ? VK_TRUE : VK_FALSE;
  depth_state.depthCompareOp = convert_compare_op(desc.depth_state.depth_compare_op);
  depth_state.depthBoundsTestEnable = desc.depth_state.depth_bounds_test_enable ? VK_TRUE : VK_FALSE;
  depth_state.minDepthBounds = desc.depth_state.min_depth_bounds;
  depth_state.maxDepthBounds = desc.depth_state.max_depth_bounds;
  depth_state.stencilTestEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = convert_sample_count_to_vk((desc.sample_count > 0u) ? desc.sample_count : 1u);

  VkPipelineColorBlendAttachmentState color_blend_attachment = {};
  color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment.blendEnable = desc.blend.blend_enable ? VK_TRUE : VK_FALSE;
  if (desc.blend.blend_enable) {
    color_blend_attachment.srcColorBlendFactor = convert_blend_factor(desc.blend.src_color_blend_factor);
    color_blend_attachment.dstColorBlendFactor = convert_blend_factor(desc.blend.dst_color_blend_factor);
    color_blend_attachment.colorBlendOp = convert_blend_op(desc.blend.color_blend_op);
    color_blend_attachment.srcAlphaBlendFactor = convert_blend_factor(desc.blend.src_alpha_blend_factor);
    color_blend_attachment.dstAlphaBlendFactor = convert_blend_factor(desc.blend.dst_alpha_blend_factor);
    color_blend_attachment.alphaBlendOp = convert_blend_op(desc.blend.alpha_blend_op);
  }

  VkPipelineColorBlendStateCreateInfo color_blending = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &color_blend_attachment;

  VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dynamic_state_info = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dynamic_state_info.dynamicStateCount = 2;
  dynamic_state_info.pDynamicStates = dynamic_states;

  std::vector<VkFormat> color_formats;
  color_formats.reserve(desc.color_attachment_count);
  for (uint32_t i = 0; i < desc.color_attachment_count; ++i) {
    color_formats.push_back(convert_rhi_format_to_vk(desc.color_formats[i]));
  }

  // Handle undefined formats that might be passed in
  for (auto& fmt : color_formats) {
    if (fmt == VK_FORMAT_UNDEFINED) {
      fmt = VK_FORMAT_B8G8R8A8_SRGB;
    }
  }

  VkFormat depth_format = convert_rhi_format_to_vk(desc.depth_format);

  VkPipelineRenderingCreateInfo rendering_info = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rendering_info.colorAttachmentCount = static_cast<uint32_t>(color_formats.size());
  rendering_info.pColorAttachmentFormats = color_formats.data();
  rendering_info.depthAttachmentFormat = depth_format;
  rendering_info.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

  VkGraphicsPipelineCreateInfo pipeline_info = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pipeline_info.pNext = &rendering_info;
  pipeline_info.stageCount = 2;
  pipeline_info.pStages = shader_stages;
  pipeline_info.pVertexInputState = &vertex_input;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pDepthStencilState = &depth_state;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.pDynamicState = &dynamic_state_info;
  pipeline_info.layout = layout;

  if (etx_vk_call(vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &out_pipeline)) != VK_SUCCESS) {
    vkDestroyShaderModule(device, frag_module, nullptr);
    vkDestroyShaderModule(device, vert_module, nullptr);
    return RHIResult::ValidationError;
  }
  vkDestroyShaderModule(device, frag_module, nullptr);
  vkDestroyShaderModule(device, vert_module, nullptr);

  return RHIResult::Success;
}

RHIResult VKDevice::Impl::create_vulkan_compute_pipeline(const RHIComputePipelineDesc& desc, VkPipelineLayout layout, VkPipeline& out_pipeline) {
  VkShaderModuleCreateInfo comp_info = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = desc.compute_shader.spirv_size,
    .pCode = reinterpret_cast<const uint32_t*>(desc.compute_shader.spirv_data),
  };
  VkShaderModule comp_module = {};
  if (etx_vk_call(vkCreateShaderModule(device, &comp_info, nullptr, &comp_module)) != VK_SUCCESS) {
    return RHIResult::ValidationError;
  }

  VkPipelineShaderStageCreateInfo shader_stage = {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
    .module = comp_module,
    .pName = desc.compute_shader.entry_point.c_str(),
  };

  VkComputePipelineCreateInfo pipeline_info = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = shader_stage,
    .layout = layout,
    .basePipelineIndex = -1,
  };

  const bool success = etx_vk_call(vkCreateComputePipelines(device, pipeline_cache, 1, &pipeline_info, nullptr, &out_pipeline)) == VK_SUCCESS;
  vkDestroyShaderModule(device, comp_module, nullptr);

  return success ? RHIResult::Success : RHIResult::ValidationError;
}

static VkFilter convert_sampler_filter(RHISamplerFilter filter) {
  switch (filter) {
    case RHISamplerFilter::Nearest:
      return VK_FILTER_NEAREST;
    case RHISamplerFilter::Linear:
      return VK_FILTER_LINEAR;
    default:
      return VK_FILTER_LINEAR;
  }
}

static VkSamplerMipmapMode convert_sampler_mipmap_mode(RHISamplerMipmapMode mode) {
  switch (mode) {
    case RHISamplerMipmapMode::Nearest:
      return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    case RHISamplerMipmapMode::Linear:
      return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    default:
      return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }
}

static VkSamplerAddressMode convert_sampler_address_mode(RHISamplerAddressMode mode) {
  switch (mode) {
    case RHISamplerAddressMode::Repeat:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case RHISamplerAddressMode::MirroredRepeat:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case RHISamplerAddressMode::ClampToEdge:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case RHISamplerAddressMode::ClampToBorder:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case RHISamplerAddressMode::MirrorClampToEdge:
      return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    default:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
}

RHIResult VKDevice::Impl::create_vulkan_sampler(const RHISamplerDesc& desc, VkSampler& out_sampler) {
  VkSamplerCreateInfo sampler_info = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sampler_info.magFilter = convert_sampler_filter(desc.mag_filter);
  sampler_info.minFilter = convert_sampler_filter(desc.min_filter);
  sampler_info.mipmapMode = convert_sampler_mipmap_mode(desc.mipmap_mode);
  sampler_info.addressModeU = convert_sampler_address_mode(desc.address_mode_u);
  sampler_info.addressModeV = convert_sampler_address_mode(desc.address_mode_v);
  sampler_info.addressModeW = convert_sampler_address_mode(desc.address_mode_w);
  sampler_info.mipLodBias = 0.0f;
  sampler_info.anisotropyEnable = (desc.max_anisotropy > 1.0f) ? VK_TRUE : VK_FALSE;
  sampler_info.maxAnisotropy = max(1.0f, desc.max_anisotropy);
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.minLod = 0.0f;
  sampler_info.maxLod = VK_LOD_CLAMP_NONE;
  sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;

  if (etx_vk_call(vkCreateSampler(device, &sampler_info, nullptr, &out_sampler)) != VK_SUCCESS) {
    return RHIResult::ValidationError;
  }

  return RHIResult::Success;
}

RHIResult VKDevice::Impl::execute_single_time_commands(std::function<void(VkCommandBuffer)> recorder) {
  VkCommandBufferAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc_info.commandPool = command_pools[kRHIMaxFrames];
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1u;

  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  if (etx_vk_call(vkAllocateCommandBuffers(device, &alloc_info, &command_buffer)) != VK_SUCCESS) {
    log::error("Failed to allocate one-time command buffer");
    return RHIResult::OutOfMemory;
  }

  VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  if (etx_vk_call(vkCreateFence(device, &fence_info, nullptr, &fence)) != VK_SUCCESS) {
    log::error("Failed to create one-time command fence");
    vkFreeCommandBuffers(device, command_pools[kRHIMaxFrames], 1u, &command_buffer);
    return RHIResult::OutOfMemory;
  }

  auto cleanup_one_time_resources = [&]() {
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, command_pools[kRHIMaxFrames], 1u, &command_buffer);
  };

  VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  if (etx_vk_call(vkBeginCommandBuffer(command_buffer, &begin_info)) != VK_SUCCESS) {
    cleanup_one_time_resources();
    return RHIResult::ValidationError;
  }

  recorder(command_buffer);

  if (etx_vk_call(vkEndCommandBuffer(command_buffer)) != VK_SUCCESS) {
    cleanup_one_time_resources();
    return RHIResult::ValidationError;
  }

  VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  if (etx_vk_call(vkQueueSubmit(graphics_queue, 1, &submit_info, fence)) != VK_SUCCESS) {
    cleanup_one_time_resources();
    return RHIResult::ValidationError;
  }

  if (etx_vk_call(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX)) != VK_SUCCESS) {
    cleanup_one_time_resources();
    return RHIResult::ValidationError;
  }

  cleanup_one_time_resources();

  return RHIResult::Success;
}

RHIResult VKDevice::Impl::create_vulkan_texture(const RHITextureDesc& desc, VkImage& out_image, VkDeviceMemory& out_memory) {
  VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  image_info.imageType = (desc.depth > 1) ? VK_IMAGE_TYPE_3D : (desc.height > 1) ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_1D;
  image_info.format = convert_rhi_format_to_vk(desc.format);
  image_info.extent.width = desc.width;
  image_info.extent.height = desc.height;
  image_info.extent.depth = desc.depth;
  image_info.mipLevels = desc.mip_levels;
  image_info.arrayLayers = desc.array_layers;
  image_info.samples = convert_sample_count_to_vk((desc.sample_count > 0u) ? desc.sample_count : 1u);
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;

  using TextureUsage = std::underlying_type<RHITextureUsage>::type;
  TextureUsage usage_flags = static_cast<TextureUsage>(desc.usage);
  VkImageUsageFlags vk_usage = 0;
  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::Sampled))
    vk_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::Storage))
    vk_usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::ColorAttachment))
    vk_usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::DepthAttachment))
    vk_usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::TransferSrc))
    vk_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::TransferDst))
    vk_usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  image_info.usage = vk_usage;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  if (etx_vk_call(vkCreateImage(device, &image_info, nullptr, &out_image)) != VK_SUCCESS) {
    return RHIResult::OutOfMemory;
  }

  VkMemoryRequirements mem_requirements;
  vkGetImageMemoryRequirements(device, out_image, &mem_requirements);

  VkMemoryPropertyFlags mem_props = desc.host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  RHIResult alloc_result = allocate_memory(mem_requirements, mem_props, 0u, out_memory);
  if (alloc_result != RHIResult::Success) {
    log::error("Failed to allocate memory for texture");
    vkDestroyImage(device, out_image, nullptr);
    out_image = VK_NULL_HANDLE;
    return RHIResult::OutOfMemory;
  }

  if (etx_vk_call(vkBindImageMemory(device, out_image, out_memory, 0)) != VK_SUCCESS) {
    vkFreeMemory(device, out_memory, nullptr);
    vkDestroyImage(device, out_image, nullptr);
    out_image = VK_NULL_HANDLE;
    out_memory = VK_NULL_HANDLE;
    return RHIResult::ValidationError;
  }

  return RHIResult::Success;
}

RHIResult VKDevice::Impl::create_vulkan_image_view(const RHITextureDesc& desc, VkImage image, VkImageView& out_view) {
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = image;
  view_info.viewType = (desc.depth > 1) ? VK_IMAGE_VIEW_TYPE_3D : (desc.height > 1) ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_1D;
  view_info.format = convert_rhi_format_to_vk(desc.format);

  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = desc.mip_levels;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = desc.array_layers;

  if (desc.format == RHITextureFormat::D32_FLOAT) {
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  }

  if (etx_vk_call(vkCreateImageView(device, &view_info, nullptr, &out_view)) != VK_SUCCESS) {
    return RHIResult::ValidationError;
  }

  return RHIResult::Success;
}

void VKDevice::Impl::free_sampler_index(uint32_t index) {
  samplers.free_index(index);
}

uint32_t VKDevice::Impl::allocate_acceleration_structure_index() {
  return acceleration_structures.allocate_index();
}

void VKDevice::Impl::free_acceleration_structure_index(uint32_t index) {
  acceleration_structures.free_index(index);
}

RHICreateBindlessResult VKDevice::create_acceleration_structure(const RHIAccelerationStructureDesc& desc) {
  if (_impl->ray_tracing_supported == false) {
    return {RHIResult::UnsupportedFeature, {}};
  }

  VkAccelerationStructureBuildGeometryInfoKHR build_info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  build_info.type = (desc.type == RHIAccelerationStructureType::BottomLevel) ? VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR : VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

  VkAccelerationStructureBuildSizesInfoKHR size_info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};

  std::vector<VkAccelerationStructureGeometryKHR> vk_geometries;
  if (desc.type == RHIAccelerationStructureType::BottomLevel) {
    if (desc.geometry_count == 0 || desc.geometries == nullptr) {
      return {RHIResult::InvalidArgument, {}};
    }

    vk_geometries.resize(desc.geometry_count);
    std::vector<uint32_t> max_primitive_counts(desc.geometry_count);

    for (uint32_t i = 0; i < desc.geometry_count; ++i) {
      const auto& src_geo = desc.geometries[i];
      auto& vk_geo = vk_geometries[i];
      vk_geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      vk_geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      vk_geo.flags = src_geo.is_opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
      vk_geo.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
      vk_geo.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;  // Simplified, should ideally match src_geo.triangles.vertex_format
      vk_geo.geometry.triangles.vertexStride = src_geo.triangles.vertex_stride;
      vk_geo.geometry.triangles.maxVertex = src_geo.triangles.vertex_count;
      vk_geo.geometry.triangles.indexType = (src_geo.triangles.index_type == RHIIndexType::UInt32) ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;

      max_primitive_counts[i] = src_geo.triangles.index_count / 3;
    }

    build_info.geometryCount = desc.geometry_count;
    build_info.pGeometries = vk_geometries.data();
    _impl->impl_vkGetAccelerationStructureBuildSizesKHR(_impl->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, max_primitive_counts.data(), &size_info);
  } else {
    VkAccelerationStructureGeometryKHR instances_geo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    instances_geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    instances_geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    instances_geo.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances_geo.geometry.instances.arrayOfPointers = VK_FALSE;

    build_info.geometryCount = 1;
    build_info.pGeometries = &instances_geo;
    _impl->impl_vkGetAccelerationStructureBuildSizesKHR(_impl->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &desc.instance_count, &size_info);
  }

  RHIBufferDesc buffer_desc = {};
  buffer_desc.size = size_info.accelerationStructureSize;
  buffer_desc.usage = RHIBufferUsage::AccelerationStructureStorage | RHIBufferUsage::ShaderDeviceAddress;
  auto buffer_res = create_buffer(buffer_desc);
  if (buffer_res.result != RHIResult::Success) {
    return {buffer_res.result, {}};
  }

  VkAccelerationStructureCreateInfoKHR create_info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  create_info.buffer = _impl->buffers.get_data(_impl->buffers.get_index(buffer_res.handle)).buffer;
  create_info.size = size_info.accelerationStructureSize;
  create_info.type = build_info.type;

  VkAccelerationStructureKHR vk_as = VK_NULL_HANDLE;
  if (etx_vk_call(_impl->impl_vkCreateAccelerationStructureKHR(_impl->device, &create_info, nullptr, &vk_as)) != VK_SUCCESS) {
    destroy_buffer(buffer_res.handle);
    return {RHIResult::ValidationError, {}};
  }

  RHIBindlessHandle as_handle = {};
  auto bindless = static_cast<VKBindlessManager*>(_impl->bindless_manager);
  RHIResult reg_result = bindless->register_acceleration_structure_vk(vk_as, desc.type, as_handle);
  if (reg_result != RHIResult::Success) {
    _impl->impl_vkDestroyAccelerationStructureKHR(_impl->device, vk_as, nullptr);
    destroy_buffer(buffer_res.handle);
    return {reg_result, {}};
  }

  uint32_t index = _impl->acceleration_structures.allocate_index();
  auto& as_data = _impl->acceleration_structures.get_data(index);
  as_data.acceleration_structure = vk_as;
  as_data.buffer = buffer_res.handle;
  as_data.desc = desc;
  as_data.build_scratch_size = size_info.buildScratchSize;
  _impl->acceleration_structures.set_handle_to_index(as_handle, index);
  _impl->as_to_buffer_map[as_handle] = buffer_res.handle;

  return {RHIResult::Success, as_handle};
}

VkCommandBuffer VKDevice::Impl::acquire_command_buffer(uint32_t pool_index) {
  for (auto& resource : command_buffer_pool) {
    if ((resource.pool_index == pool_index) && (resource.used == false)) {
      resource.used = true;
      return resource.buffer;
    }
  }

  VkCommandBufferAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc_info.commandPool = command_pools[pool_index];
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer new_buffer = {};
  if (etx_vk_call(vkAllocateCommandBuffers(device, &alloc_info, &new_buffer)) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }

  CommandBufferResource& resource = command_buffer_pool.emplace_back();
  resource.buffer = new_buffer;
  resource.pool_index = pool_index;
  resource.used = true;
  return new_buffer;
}

void VKDevice::Impl::release_command_buffer(VkCommandBuffer command_buffer) {
  for (auto& resource : command_buffer_pool) {
    if (resource.buffer == command_buffer) {
      resource.used = false;
      return;
    }
  }
  log::error("Attempted to release unknown command buffer");
}

VkFence VKDevice::Impl::acquire_fence() {
  // First try to find an available fence
  for (auto& resource : fence_pool) {
    if (resource.used == false) {
      resource.used = true;
      return resource.fence;
    }
  }

  // No available fence found, create a new one
  VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};

  VkFence new_fence;
  if (etx_vk_call(vkCreateFence(device, &fence_info, nullptr, &new_fence)) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }

  // Add to pool
  FenceResource resource;
  resource.fence = new_fence;
  resource.used = true;
  fence_pool.push_back(resource);
  return new_fence;
}

void VKDevice::Impl::release_fence(VkFence fence) {
  for (auto& resource : fence_pool) {
    if (resource.fence == fence) {
      resource.used = false;
      return;
    }
  }
  log::error("Attempted to release unknown fence");
}

bool VKDevice::Impl::initialize_pools() {
  // Pools now grow dynamically as needed, no pre-allocation required
  return true;
}

void VKDevice::Impl::cleanup_pools() {
  // Destroy command buffers
  for (const auto& resource : command_buffer_pool) {
    if (resource.buffer != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(device, command_pools[resource.pool_index], 1u, &resource.buffer);
    }
  }
  command_buffer_pool.clear();

  // Destroy fences
  for (const auto& resource : fence_pool) {
    if (resource.fence != VK_NULL_HANDLE) {
      vkDestroyFence(device, resource.fence, nullptr);
    }
  }
  fence_pool.clear();

  // Destroy acceleration structures
  acceleration_structures.for_each([this](VKAccelerationStructureData& data) {
    if (data.acceleration_structure != VK_NULL_HANDLE) {
      if (impl_vkDestroyAccelerationStructureKHR) {
        impl_vkDestroyAccelerationStructureKHR(device, data.acceleration_structure, nullptr);
      }
      data.acceleration_structure = VK_NULL_HANDLE;
    }
  });
  acceleration_structures.clear();

  // Destroy pipelines
  auto destroy_pipeline = [this](VKPipelineData& data) {
    if (data.pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(device, data.pipeline, nullptr);
      data.pipeline = VK_NULL_HANDLE;
    }
  };
  compute_pipelines.for_each(destroy_pipeline);
  compute_pipelines.clear();
  graphics_pipelines.for_each(destroy_pipeline);
  graphics_pipelines.clear();

  // Destroy samplers
  samplers.for_each([this](VKSamplerData& data) {
    if (data.sampler != VK_NULL_HANDLE) {
      vkDestroySampler(device, data.sampler, nullptr);
      data.sampler = VK_NULL_HANDLE;
    }
  });
  samplers.clear();

  // Destroy textures
  textures.for_each([this](VKTextureData& data) {
    if (data.image_view != VK_NULL_HANDLE) {
      vkDestroyImageView(device, data.image_view, nullptr);
      data.image_view = VK_NULL_HANDLE;
    }
    if (data.image != VK_NULL_HANDLE) {
      vkDestroyImage(device, data.image, nullptr);
      data.image = VK_NULL_HANDLE;
    }
    if (data.memory != VK_NULL_HANDLE) {
      vkFreeMemory(device, data.memory, nullptr);
      data.memory = VK_NULL_HANDLE;
    }
  });
  textures.clear();

  // Destroy buffers
  buffers.for_each([this](VKBufferData& data) {
    if (data.buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, data.buffer, nullptr);
      data.buffer = VK_NULL_HANDLE;
    }
    if (data.memory != VK_NULL_HANDLE) {
      vkFreeMemory(device, data.memory, nullptr);
      data.memory = VK_NULL_HANDLE;
    }
  });
  buffers.clear();
}

const VKPipelineData* VKDevice::get_compute_pipeline_data(RHIPipeline handle) const {
  return _impl->compute_pipelines.get_data_ptr(handle);
}

const VKPipelineData* VKDevice::get_graphics_pipeline_data(RHIPipeline handle) const {
  return _impl->graphics_pipelines.get_data_ptr(handle);
}

VkBuffer VKDevice::get_vk_buffer_from_bindless(RHIBindlessHandle handle) const {
  if (_impl->bindless_manager && !_impl->bindless_manager->is_valid_handle(handle)) {
    log::error("Invalid bindless handle: %llu", handle);
    return VK_NULL_HANDLE;
  }

  if (_impl->bindless_manager && _impl->bindless_manager->get_resource_type(handle) != RHIResourceType::Buffer) {
    log::error("Handle %llu is not a buffer type", handle);
    return VK_NULL_HANDLE;
  }

  uint32_t index = _impl->buffers.get_index(handle);
  if (index != UINT32_MAX) {
    return _impl->buffers.get_data(index).buffer;
  }

  if (_impl->bindless_manager) {
    VkBuffer buffer = static_cast<VKBindlessManager*>(_impl->bindless_manager)->get_vk_buffer(handle);
    if (buffer != VK_NULL_HANDLE) {
      return buffer;
    }
  }

  log::error("Buffer handle %llu not found in buffer map or bindless manager", handle);
  return VK_NULL_HANDLE;
}

VkImage VKDevice::get_vk_image_from_bindless(RHIBindlessHandle handle) const {
  if (_impl->bindless_manager && !_impl->bindless_manager->is_valid_handle(handle)) {
    log::error("Invalid bindless handle: %llu", handle);
    return VK_NULL_HANDLE;
  }

  if (_impl->bindless_manager && _impl->bindless_manager->get_resource_type(handle) != RHIResourceType::Texture) {
    log::error("Handle %llu is not a texture type", handle);
    return VK_NULL_HANDLE;
  }

  uint32_t index = _impl->textures.get_index(handle);
  if (index != UINT32_MAX) {
    return _impl->textures.get_data(index).image;
  }

  if (_impl->bindless_manager) {
    VkImage image = static_cast<VKBindlessManager*>(_impl->bindless_manager)->get_vk_image(handle);
    if (image != VK_NULL_HANDLE) {
      return image;
    }
  }

  log::error("Texture handle %llu not found in texture map or bindless manager", handle);
  return VK_NULL_HANDLE;
}

VKDevice::VKDevice(const RHIInitInfo& info)
  : _impl(new Impl(info)) {
}

void VKDevice::destroy_all_resources() {
  if (_impl == nullptr) {
    return;
  }

  uint32_t graphics_pipeline_count = static_cast<uint32_t>(_impl->graphics_pipelines.size());
  uint32_t compute_pipeline_count = static_cast<uint32_t>(_impl->compute_pipelines.size());
  uint32_t texture_count = static_cast<uint32_t>(_impl->textures.size());
  uint32_t buffer_count = static_cast<uint32_t>(_impl->buffers.size());
  uint32_t sampler_count = static_cast<uint32_t>(_impl->samplers.size());

  std::vector<RHIPipeline> graphics_pipeline_handles = _impl->graphics_pipelines.get_all_keys();
  std::vector<RHIPipeline> compute_pipeline_handles = _impl->compute_pipelines.get_all_keys();
  std::vector<RHIBindlessHandle> texture_handles = _impl->textures.get_all_keys();
  std::vector<RHIBindlessHandle> buffer_handles = _impl->buffers.get_all_keys();
  std::vector<RHIBindlessHandle> sampler_handles = _impl->samplers.get_all_keys();

  for (RHIPipeline handle : graphics_pipeline_handles) {
    RHIResult result = destroy_pipeline(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy graphics pipeline %llu: %d", handle.value, static_cast<int>(result));
    }
  }
  _impl->graphics_pipelines.clear();

  for (RHIPipeline handle : compute_pipeline_handles) {
    RHIResult result = destroy_pipeline(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy compute pipeline %llu: %d", handle.value, static_cast<int>(result));
    }
  }
  _impl->compute_pipelines.clear();

  for (RHIBindlessHandle handle : texture_handles) {
    RHIResult result = destroy_texture(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy texture %llu: %d", handle, static_cast<int>(result));
    }
  }
  _impl->textures.clear();

  for (RHIBindlessHandle handle : buffer_handles) {
    RHIResult result = destroy_buffer(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy buffer %llu: %d", handle, static_cast<int>(result));
    }
  }
  _impl->buffers.clear();

  for (RHIBindlessHandle handle : sampler_handles) {
    RHIResult result = destroy_sampler(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy sampler %llu: %d", handle, static_cast<int>(result));
    }
  }
  _impl->samplers.clear();

  std::vector<RHISemaphore> semaphore_handles = _impl->semaphores.get_all_keys();
  for (RHISemaphore handle : semaphore_handles) {
    RHIResult result = destroy_semaphore(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy semaphore %llu: %d", handle.value, static_cast<int>(result));
    }
  }
  _impl->semaphores.clear();

  for (uint32_t i = 0; i < kRHIMaxFrames; ++i) {
    _impl->process_deferred_destruction(i);
  }
}

void VKDevice::destroy_bindless_pipeline_layout() {
  if (_impl == nullptr) {
    return;
  }
  _impl->destroy_bindless_pipeline_layout();
}

VKDevice::~VKDevice() {
  delete _impl;
}

RHICreateBindlessResult VKDevice::create_buffer(const RHIBufferDesc& desc) {
  if (_impl->device == VK_NULL_HANDLE) {
    log::error("Vulkan device not initialized");
    return {RHIResult::InvalidArgument, {}};
  }

  if (_impl->bindless_manager == nullptr) {
    log::error("Bindless manager not available for buffer creation");
    return {RHIResult::InvalidArgument, {}};
  }

  using BufferUsage = std::underlying_type<RHIBufferUsage>::type;
  const BufferUsage usage = static_cast<BufferUsage>(desc.usage);
  if ((usage & static_cast<BufferUsage>(RHIBufferUsage::Storage)) && (desc.size > _impl->properties.limits.maxStorageBufferRange)) {
    log::error("Requested Vulkan storage buffer exceeds device maxStorageBufferRange (size=%llu limit=%u)", desc.size, _impl->properties.limits.maxStorageBufferRange);
    return {RHIResult::InvalidArgument, {}};
  }

  VkBuffer vk_handle = VK_NULL_HANDLE;
  VkDeviceMemory vk_memory = VK_NULL_HANDLE;
  RHIResult create_result = _impl->create_vulkan_buffer(desc, vk_handle, vk_memory);
  if (create_result != RHIResult::Success) {
    return {create_result, {}};
  }

  VkMemoryRequirements mem_req = {};
  vkGetBufferMemoryRequirements(_impl->device, vk_handle, &mem_req);

  uint32_t index = _impl->buffers.allocate_index();
  auto& buffer_data = _impl->buffers.get_data(index);
  buffer_data.buffer = vk_handle;
  buffer_data.memory = vk_memory;
  buffer_data.desc = desc;
  buffer_data.allocated_size = mem_req.size;

  // Register with bindless manager
  RHIBindlessHandle handle = {};
  RHIResult reg_result = _impl->bindless_manager->register_buffer(buffer_data.buffer, RHIResourceType::Buffer, handle);
  if (reg_result != RHIResult::Success) {
    log::error("Failed to register buffer with bindless manager");

    if (vk_memory != VK_NULL_HANDLE) {
      vkFreeMemory(_impl->device, vk_memory, nullptr);
      vk_memory = VK_NULL_HANDLE;
      if (mem_req.size > 0u) {
        _impl->gpu_allocated_bytes -= mem_req.size;
      }
    }
    if (vk_handle != VK_NULL_HANDLE) {
      vkDestroyBuffer(_impl->device, vk_handle, nullptr);
      vk_handle = VK_NULL_HANDLE;
    }

    _impl->buffers.free_index(index);
    return {reg_result, {}};
  }

  // Store the mapping
  _impl->buffers.set_handle_to_index(handle, index);

  return {RHIResult::Success, handle};
}

RHICreateBindlessResult VKDevice::create_texture(const RHITextureDesc& desc) {
  if (_impl->device == VK_NULL_HANDLE) {
    log::error("Vulkan device not initialized");
    return {RHIResult::InvalidArgument, {}};
  }

  if (_impl->bindless_manager == nullptr) {
    log::error("Bindless manager not available for texture creation");
    return {RHIResult::InvalidArgument, {}};
  }

  VkImage vk_image = VK_NULL_HANDLE;
  VkDeviceMemory vk_memory = VK_NULL_HANDLE;
  RHIResult create_result = _impl->create_vulkan_texture(desc, vk_image, vk_memory);
  if (create_result != RHIResult::Success) {
    return {create_result, {}};
  }

  VkMemoryRequirements tex_mem_req = {};
  vkGetImageMemoryRequirements(_impl->device, vk_image, &tex_mem_req);

  VkImageView vk_view = VK_NULL_HANDLE;
  RHIResult view_result = _impl->create_vulkan_image_view(desc, vk_image, vk_view);
  if (view_result != RHIResult::Success) {
    if ((vk_memory != VK_NULL_HANDLE) && (tex_mem_req.size != 0u)) {
      _impl->gpu_allocated_bytes -= tex_mem_req.size;
    }
    vkFreeMemory(_impl->device, vk_memory, nullptr);
    vkDestroyImage(_impl->device, vk_image, nullptr);
    return {view_result, {}};
  }

  uint32_t index = _impl->textures.allocate_index();
  auto& texture_data = _impl->textures.get_data(index);
  texture_data.image = vk_image;
  texture_data.image_view = vk_view;
  texture_data.memory = vk_memory;
  texture_data.desc = desc;
  texture_data.current_state = RHIResourceState::Undefined;
  texture_data.allocated_size = tex_mem_req.size;

  // Register with bindless manager
  RHIBindlessHandle handle;
  RHIResult reg_result =
    _impl->bindless_manager->register_texture(texture_data.image_view, RHIResourceType::Texture, handle, static_cast<uint32_t>(desc.usage), texture_data.image);
  if (reg_result != RHIResult::Success) {
    log::error("Failed to register texture with bindless manager");
    _impl->textures.free_index(index);
    vkDestroyImageView(_impl->device, vk_view, nullptr);
    if ((vk_memory != VK_NULL_HANDLE) && (tex_mem_req.size != 0u)) {
      _impl->gpu_allocated_bytes -= tex_mem_req.size;
    }
    vkFreeMemory(_impl->device, vk_memory, nullptr);
    vkDestroyImage(_impl->device, vk_image, nullptr);
    return {reg_result, {}};
  }

  // Store the mapping
  _impl->textures.set_handle_to_index(handle, index);

  return {RHIResult::Success, handle};
}

RHICreateBindlessResult VKDevice::create_sampler(const RHISamplerDesc& desc) {
  if (_impl->device == VK_NULL_HANDLE) {
    log::error("Vulkan device not initialized");
    return {RHIResult::InvalidArgument, {}};
  }

  if (_impl->bindless_manager == nullptr) {
    log::error("Bindless manager not available for sampler creation");
    return {RHIResult::InvalidArgument, {}};
  }

  VkSampler vk_sampler = VK_NULL_HANDLE;
  RHIResult create_result = _impl->create_vulkan_sampler(desc, vk_sampler);
  if (create_result != RHIResult::Success) {
    return {create_result, {}};
  }

  // Initialize POD data
  uint32_t index = _impl->samplers.allocate_index();
  auto& sampler_data = _impl->samplers.get_data(index);
  sampler_data.sampler = vk_sampler;
  sampler_data.desc = desc;

  // Register with bindless manager
  RHIBindlessHandle handle;
  RHIResult reg_result = _impl->bindless_manager->register_sampler(sampler_data.sampler, RHIResourceType::Sampler, handle);
  if (reg_result != RHIResult::Success) {
    log::error("Failed to register sampler with bindless manager");
    _impl->samplers.free_index(index);
    vkDestroySampler(_impl->device, vk_sampler, nullptr);
    return {reg_result, {}};
  }

  // Store the mapping
  _impl->samplers.set_handle_to_index(handle, index);

  return {RHIResult::Success, handle};
}

RHICreatePipelineResult VKDevice::create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) {
  if (_impl->device == VK_NULL_HANDLE) {
    log::error("Vulkan device not initialized");
    return {RHIResult::InvalidArgument, {}};
  }

  if (_impl->bindless_manager == nullptr) {
    log::error("Bindless manager not available for graphics pipeline creation");
    return {RHIResult::InvalidArgument, {}};
  }

  auto vk_layout = _impl->get_bindless_pipeline_layout();
  if (vk_layout.result != RHIResult::Success) {
    return {vk_layout.result, {}};
  }

  VkPipeline vk_pipeline = VK_NULL_HANDLE;
  RHIResult pipeline_result = _impl->create_vulkan_graphics_pipeline(desc, vk_layout.handle, vk_pipeline);
  if (pipeline_result != RHIResult::Success) {
    return {pipeline_result, {}};
  }

  uint32_t index = _impl->graphics_pipelines.allocate_index();
  uint32_t generation = _impl->graphics_pipelines.get_generation(index);
  RHIPipeline pipeline_handle = Handle::construct(0, index, generation);

  // Initialize POD data
  auto& pipeline_data = _impl->graphics_pipelines.get_data(index);
  pipeline_data.pipeline = vk_pipeline;
  pipeline_data.handle = pipeline_handle;

  // Store the mapping
  _impl->graphics_pipelines.set_handle_to_index(pipeline_handle, index);

  return {RHIResult::Success, pipeline_handle};
}

RHICreatePipelineResult VKDevice::create_compute_pipeline(const RHIComputePipelineDesc& desc) {
  if (_impl->device == VK_NULL_HANDLE) {
    log::error("Vulkan device not initialized");
    return {RHIResult::InvalidArgument, {}};
  }

  if (_impl->bindless_manager == nullptr) {
    log::error("Bindless manager not available for compute pipeline creation");
    return {RHIResult::InvalidArgument, {}};
  }

  auto vk_layout = _impl->get_bindless_pipeline_layout();
  if (vk_layout.result != RHIResult::Success) {
    return {vk_layout.result, {}};
  }

  VkPipeline vk_pipeline = VK_NULL_HANDLE;
  RHIResult pipeline_result = _impl->create_vulkan_compute_pipeline(desc, vk_layout.handle, vk_pipeline);
  if (pipeline_result != RHIResult::Success) {
    return {pipeline_result, {}};
  }

  uint32_t index = _impl->compute_pipelines.allocate_index();
  uint32_t generation = _impl->compute_pipelines.get_generation(index);
  RHIPipeline pipeline_handle = Handle::construct(0, index, generation);

  // Initialize POD data
  auto& pipeline_data = _impl->compute_pipelines.get_data(index);
  pipeline_data.pipeline = vk_pipeline;
  pipeline_data.handle = pipeline_handle;

  // Store the mapping
  _impl->compute_pipelines.set_handle_to_index(pipeline_handle, index);

  return {RHIResult::Success, pipeline_handle};
}

RHIResult VKDevice::destroy_buffer(RHIBindlessHandle buffer_handle) {
  if (buffer_handle.valid() == false)
    return RHIResult::Success;

  uint32_t index = _impl->buffers.get_index(buffer_handle);
  if (index == UINT32_MAX) {
    return RHIResult::Success;
  }

  RHIResult result = _impl->bindless_manager->unregister_buffer(buffer_handle);
  if (result != RHIResult::Success) {
    log::error("Failed to unregister buffer from bindless manager");
    return result;
  }

  // Remove from mapped buffer tracking if present
  _impl->mapped_buffer_ptrs.erase(buffer_handle);
  _impl->buffers.remove_handle(buffer_handle);

  const auto& buffer_data = _impl->buffers.get_data(index);
  uint64_t size_to_subtract = buffer_data.allocated_size;

  _impl->buffers.free_index(index, [this, size_to_subtract](const VKBufferData& data) {
    if (size_to_subtract != 0) {
      _impl->gpu_allocated_bytes -= size_to_subtract;
    }
    _impl->queue_deferred_destruction(data);
  });

  return RHIResult::Success;
}

RHIResult VKDevice::destroy_texture(RHIBindlessHandle texture_handle) {
  if (texture_handle.valid() == false)
    return RHIResult::Success;

  uint32_t index = _impl->textures.get_index(texture_handle);
  if (index == UINT32_MAX) {
    return RHIResult::Success;
  }

  if (_impl->bindless_manager == nullptr) {
    return RHIResult::InvalidArgument;
  }

  RHIResult result = _impl->bindless_manager->unregister_texture(texture_handle);
  if (result != RHIResult::Success) {
    log::error("[VKDevice::destroy_texture] Failed to unregister texture from bindless manager");
    return result;
  }

  _impl->textures.remove_handle(texture_handle);

  const auto& tex_data = _impl->textures.get_data(index);
  uint64_t tex_size_to_subtract = tex_data.allocated_size;

  _impl->textures.free_index(index, [this, tex_size_to_subtract](const VKTextureData& data) {
    if (tex_size_to_subtract != 0) {
      _impl->gpu_allocated_bytes -= tex_size_to_subtract;
    }
    _impl->queue_deferred_destruction(data);
  });

  return RHIResult::Success;
}

RHIResult VKDevice::destroy_sampler(RHIBindlessHandle sampler_handle) {
  if (sampler_handle.valid() == false)
    return RHIResult::Success;

  uint32_t index = _impl->samplers.get_index(sampler_handle);
  if (index == UINT32_MAX) {
    return RHIResult::Success;
  }

  if (_impl->bindless_manager == nullptr) {
    return RHIResult::InvalidArgument;
  }

  RHIResult result = _impl->bindless_manager->unregister_sampler(sampler_handle);
  if (result != RHIResult::Success) {
    log::error("Failed to unregister sampler from bindless manager");
    return result;
  }

  _impl->samplers.remove_handle(sampler_handle);

  // Cleanup Vulkan resources
  _impl->samplers.free_index(index, [this](const VKSamplerData& data) {
    _impl->queue_deferred_destruction(data);
  });

  return RHIResult::Success;
}

RHIResult VKDevice::destroy_pipeline(RHIPipeline pipeline_handle) {
  if ((pipeline_handle.valid() == false) || (_impl == nullptr))
    return RHIResult::Success;

  uint32_t compute_index = _impl->compute_pipelines.get_index(pipeline_handle);
  if (compute_index != UINT32_MAX) {
    _impl->compute_pipelines.remove_handle(pipeline_handle);
    _impl->compute_pipelines.free_index(compute_index, [this](const VKPipelineData& data) {
      _impl->queue_deferred_destruction(data, true);
    });
    return RHIResult::Success;
  }

  uint32_t graphics_index = _impl->graphics_pipelines.get_index(pipeline_handle);
  if (graphics_index != UINT32_MAX) {
    _impl->graphics_pipelines.remove_handle(pipeline_handle);
    _impl->graphics_pipelines.free_index(graphics_index, [this](const VKPipelineData& data) {
      _impl->queue_deferred_destruction(data, false);
    });
    return RHIResult::Success;
  }

  return RHIResult::Success;
}

RHIResult VKDevice::update_buffer(RHIBindlessHandle buffer_handle, const void* data, uint64_t size, uint64_t offset) {
  uint32_t index = _impl->buffers.get_index(buffer_handle);
  if (index == UINT32_MAX) {
    log::error("Buffer handle not found: %llu", buffer_handle);
    return RHIResult::InvalidHandle;
  }

  const auto& buffer_data = _impl->buffers.get_data(index);
  const uint64_t buffer_size = buffer_data.desc.size;

  if (size == 0u) {
    return RHIResult::Success;
  }
  if ((data == nullptr) && (size > 0u)) {
    log::error("Buffer update received null data pointer for non-zero size (%llu)", size);
    return RHIResult::InvalidArgument;
  }
  if (offset > buffer_size) {
    log::error("Buffer update offset out of range (offset=%llu, buffer_size=%llu)", offset, buffer_size);
    return RHIResult::InvalidArgument;
  }
  if (size > (buffer_size - offset)) {
    log::error("Buffer update range out of bounds (offset=%llu, size=%llu, buffer_size=%llu)", offset, size, buffer_size);
    return RHIResult::InvalidArgument;
  }

  if (buffer_data.desc.host_visible) {
    // Map buffer if not already mapped
    void* mapped_ptr = nullptr;
    auto mapped_it = _impl->mapped_buffer_ptrs.find(buffer_handle);
    if (mapped_it == _impl->mapped_buffer_ptrs.end()) {
      if (etx_vk_call(vkMapMemory(_impl->device, buffer_data.memory, 0, VK_WHOLE_SIZE, 0, &mapped_ptr)) != VK_SUCCESS) {
        return RHIResult::ValidationError;
      }
      _impl->mapped_buffer_ptrs[buffer_handle] = mapped_ptr;
    } else {
      mapped_ptr = mapped_it->second;
    }

    // Copy data to mapped memory
    memcpy(static_cast<uint8_t*>(mapped_ptr) + offset, data, size);
    return RHIResult::Success;
  }

  // Try to use persistent staging buffer with frame-aware allocation
  uint64_t staging_offset = 0;
  void* staging_ptr = nullptr;
  bool use_persistent_staging = _impl->staging_buffer.allocate(_impl->current_frame_index, size, staging_offset, staging_ptr);

  VkBuffer staging_handle = VK_NULL_HANDLE;
  VkDeviceMemory staging_memory = VK_NULL_HANDLE;

  if (use_persistent_staging) {
    staging_handle = _impl->staging_buffer.buffer;
    memcpy(staging_ptr, data, size);
  } else {
    // Fallback to transient staging buffer if persistent one is full
    const bool update_fits_frame_staging = (size <= _impl->staging_buffer.per_frame_capacity);
    if (update_fits_frame_staging) {
      log::warning("Persistent staging buffer full, falling back to transient buffer for update of size %llu", size);
    }
    RHIBufferDesc staging_desc = {};
    staging_desc.size = size;
    staging_desc.usage = RHIBufferUsage::TransferSrc;
    staging_desc.host_visible = true;

    RHIResult create_result = _impl->create_vulkan_buffer(staging_desc, staging_handle, staging_memory);
    if (create_result != RHIResult::Success) {
      log::error("Failed to create transient staging buffer");
      return create_result;
    }

    void* mapped_ptr = nullptr;
    if (etx_vk_call(vkMapMemory(_impl->device, staging_memory, 0, VK_WHOLE_SIZE, 0, &mapped_ptr)) != VK_SUCCESS) {
      vkFreeMemory(_impl->device, staging_memory, nullptr);
      vkDestroyBuffer(_impl->device, staging_handle, nullptr);
      return RHIResult::ValidationError;
    }
    memcpy(mapped_ptr, data, size);
    vkUnmapMemory(_impl->device, staging_memory);
  }

  RHIResult submit_result = _impl->execute_single_time_commands([&](VkCommandBuffer command_buffer) {
    VkBufferCopy copy_region = {};
    copy_region.srcOffset = use_persistent_staging ? staging_offset : 0;
    copy_region.dstOffset = offset;
    copy_region.size = size;
    vkCmdCopyBuffer(command_buffer, staging_handle, buffer_data.buffer, 1, &copy_region);
  });

  if (!use_persistent_staging) {
    vkFreeMemory(_impl->device, staging_memory, nullptr);
    vkDestroyBuffer(_impl->device, staging_handle, nullptr);
  }

  return submit_result;
}

RHIResult VKDevice::read_buffer(RHIBindlessHandle buffer_handle, void* data, uint64_t size, uint64_t offset) {
  uint32_t index = _impl->buffers.get_index(buffer_handle);
  if (index == UINT32_MAX) {
    log::error("Buffer handle not found: %llu", buffer_handle);
    return RHIResult::InvalidHandle;
  }

  const auto& buffer_data = _impl->buffers.get_data(index);
  const uint64_t buffer_size = buffer_data.desc.size;

  if (size == 0u) {
    return RHIResult::Success;
  }
  if ((data == nullptr) && (size > 0u)) {
    log::error("Buffer read received null destination pointer for non-zero size (%llu)", size);
    return RHIResult::InvalidArgument;
  }
  if (offset > buffer_size) {
    log::error("Buffer read offset out of range (offset=%llu, buffer_size=%llu)", offset, buffer_size);
    return RHIResult::InvalidArgument;
  }
  if (size > (buffer_size - offset)) {
    log::error("Buffer read range out of bounds (offset=%llu, size=%llu, buffer_size=%llu)", offset, size, buffer_size);
    return RHIResult::InvalidArgument;
  }
  if (buffer_data.desc.host_visible == false) {
    log::error("Buffer read requires a host-visible buffer");
    return RHIResult::UnsupportedFeature;
  }

  void* mapped_ptr = nullptr;
  auto mapped_it = _impl->mapped_buffer_ptrs.find(buffer_handle);
  if (mapped_it == _impl->mapped_buffer_ptrs.end()) {
    if (etx_vk_call(vkMapMemory(_impl->device, buffer_data.memory, 0, VK_WHOLE_SIZE, 0, &mapped_ptr)) != VK_SUCCESS) {
      return RHIResult::ValidationError;
    }
    _impl->mapped_buffer_ptrs[buffer_handle] = mapped_ptr;
  } else {
    mapped_ptr = mapped_it->second;
  }

  memcpy(data, static_cast<const uint8_t*>(mapped_ptr) + offset, size);
  return RHIResult::Success;
}

RHIResult VKDevice::update_texture(RHIBindlessHandle texture_handle, const void* data, uint32_t mip_level, uint32_t array_layer) {
  uint32_t index = _impl->textures.get_index(texture_handle);
  if (index == UINT32_MAX) {
    log::error("Texture handle not found: %llu", texture_handle);
    return RHIResult::InvalidHandle;
  }

  auto& texture_data = _impl->textures.get_data(index);
  const RHITextureDesc& desc = texture_data.desc;

  if (mip_level >= desc.mip_levels) {
    log::error("Mip level %u exceeds texture mip levels %u", mip_level, desc.mip_levels);
    return RHIResult::InvalidArgument;
  }

  if (array_layer >= desc.array_layers) {
    log::error("Array layer %u exceeds texture array layers %u", array_layer, desc.array_layers);
    return RHIResult::InvalidArgument;
  }

  using TextureUsage = std::underlying_type<RHITextureUsage>::type;
  const TextureUsage usage_flags = static_cast<TextureUsage>(desc.usage);
  if ((usage_flags & static_cast<TextureUsage>(RHITextureUsage::TransferDst)) == 0u) {
    log::error("Texture update requires TransferDst usage");
    return RHIResult::InvalidArgument;
  }

  if (data == nullptr) {
    log::error("Texture update received null data pointer");
    return RHIResult::InvalidArgument;
  }

  uint32_t mip_width = max(desc.width >> mip_level, 1u);
  uint32_t mip_height = max(desc.height >> mip_level, 1u);
  uint32_t mip_depth = max(desc.depth >> mip_level, 1u);

  uint64_t bytes_per_pixel = convert_rhi_format_to_bytes_per_pixel(desc.format);
  if (bytes_per_pixel == 0u) {
    log::error("Texture update received unsupported texture format (%u)", static_cast<uint32_t>(desc.format));
    return RHIResult::InvalidArgument;
  }

  const uint64_t mip_width_u64 = static_cast<uint64_t>(mip_width);
  const uint64_t mip_height_u64 = static_cast<uint64_t>(mip_height);
  const uint64_t mip_depth_u64 = static_cast<uint64_t>(mip_depth);

  if ((mip_height_u64 > 0u) && (mip_width_u64 > (UINT64_MAX / mip_height_u64))) {
    log::error("Texture update size overflow (width=%llu, height=%llu)", mip_width_u64, mip_height_u64);
    return RHIResult::InvalidArgument;
  }

  const uint64_t slice_texel_count = mip_width_u64 * mip_height_u64;
  if ((mip_depth_u64 > 0u) && (slice_texel_count > (UINT64_MAX / mip_depth_u64))) {
    log::error("Texture update size overflow (slice_texels=%llu, depth=%llu)", slice_texel_count, mip_depth_u64);
    return RHIResult::InvalidArgument;
  }

  const uint64_t texel_count = slice_texel_count * mip_depth_u64;
  if ((bytes_per_pixel > 0u) && (texel_count > (UINT64_MAX / bytes_per_pixel))) {
    log::error("Texture update byte size overflow (texels=%llu, bytes_per_pixel=%llu)", texel_count, bytes_per_pixel);
    return RHIResult::InvalidArgument;
  }

  const uint64_t data_size = texel_count * bytes_per_pixel;
  if (data_size == 0u) {
    log::error("Texture update computed zero byte size");
    return RHIResult::InvalidArgument;
  }

  // Try to use persistent staging buffer with frame-aware allocation
  uint64_t staging_offset = 0;
  void* staging_ptr = nullptr;
  bool use_persistent_staging = _impl->staging_buffer.allocate(_impl->current_frame_index, data_size, staging_offset, staging_ptr);

  VkBuffer staging_handle = VK_NULL_HANDLE;
  VkDeviceMemory staging_memory = VK_NULL_HANDLE;

  if (use_persistent_staging) {
    staging_handle = _impl->staging_buffer.buffer;
    memcpy(staging_ptr, data, data_size);
  } else {
    // Fallback to transient staging
    const bool texture_update_fits_frame_staging = (data_size <= _impl->staging_buffer.per_frame_capacity);
    if (texture_update_fits_frame_staging) {
      log::warning("Persistent staging buffer full, falling back to transient buffer for texture update");
    }
    RHIBufferDesc staging_desc = {};
    staging_desc.size = data_size;
    staging_desc.usage = RHIBufferUsage::TransferSrc;
    staging_desc.host_visible = true;

    RHIResult create_result = _impl->create_vulkan_buffer(staging_desc, staging_handle, staging_memory);
    if (create_result != RHIResult::Success) {
      log::error("Failed to create transient staging buffer for texture");
      return create_result;
    }

    void* mapped_ptr = nullptr;
    if (etx_vk_call(vkMapMemory(_impl->device, staging_memory, 0, VK_WHOLE_SIZE, 0, &mapped_ptr)) != VK_SUCCESS) {
      vkFreeMemory(_impl->device, staging_memory, nullptr);
      vkDestroyBuffer(_impl->device, staging_handle, nullptr);
      return RHIResult::ValidationError;
    }
    memcpy(mapped_ptr, data, data_size);
    vkUnmapMemory(_impl->device, staging_memory);
  }

  RHIResult submit_result = _impl->execute_single_time_commands([&](VkCommandBuffer command_buffer) {
    VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = texture_data.image;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // Safe to discard previous contents
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = mip_level;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = array_layer;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Layout tracking removed - no race conditions
    VkBufferImageCopy copy_region = {};
    copy_region.bufferOffset = use_persistent_staging ? staging_offset : 0;
    copy_region.bufferRowLength = 0;
    copy_region.bufferImageHeight = 0;
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.mipLevel = mip_level;
    copy_region.imageSubresource.baseArrayLayer = array_layer;
    copy_region.imageSubresource.layerCount = 1;
    copy_region.imageOffset = {0, 0, 0};
    copy_region.imageExtent = {mip_width, mip_height, mip_depth};

    vkCmdCopyBufferToImage(command_buffer, staging_handle, texture_data.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  });

  if (!use_persistent_staging) {
    vkFreeMemory(_impl->device, staging_memory, nullptr);
    vkDestroyBuffer(_impl->device, staging_handle, nullptr);
  }

  if (submit_result == RHIResult::Success) {
    texture_data.current_state = RHIResourceState::ShaderReadOnly;
  }

  return submit_result;
}

VkPhysicalDevice VKDevice::get_vk_physical_device() const {
  return _impl->physical_device;
}

void VKDevice::set_bindless_manager(VKBindlessManager* manager) {
  _impl->bindless_manager = manager;
}

RHIResult VKDevice::reload_graphics_pipeline(RHIPipeline pipeline, const RHIGraphicsPipelineDesc& new_desc) {
  RHIResult destroy_result = destroy_pipeline(pipeline);
  if (destroy_result != RHIResult::Success) {
    log::error("Failed to destroy old graphics pipeline for reloading");
    return destroy_result;
  }

  auto create_result = create_graphics_pipeline(new_desc);
  if (create_result.result != RHIResult::Success) {
    log::error("Failed to create new graphics pipeline for reloading");
    return create_result.result;
  }

  if (create_result.handle.value != pipeline.value) {
    log::warning("Pipeline handle changed during reload: %llu -> %llu", pipeline.value, create_result.handle.value);
  }

  return RHIResult::Success;
}

RHIResult VKDevice::reload_compute_pipeline(RHIPipeline pipeline, const RHIComputePipelineDesc& new_desc) {
  RHIResult destroy_result = destroy_pipeline(pipeline);
  if (destroy_result != RHIResult::Success) {
    log::error("Failed to destroy old compute pipeline for reloading");
    return destroy_result;
  }

  auto create_result = create_compute_pipeline(new_desc);
  if (create_result.result != RHIResult::Success) {
    log::error("Failed to create new compute pipeline for reloading");
    return create_result.result;
  }

  if (create_result.handle.value != pipeline.value) {
    log::warning("Pipeline handle changed during reload: %llu -> %llu", pipeline.value, create_result.handle.value);
  }

  return RHIResult::Success;
}

RHIMemoryStats VKDevice::get_memory_statistics() const {
  RHIMemoryStats stats = {};

#if ETX_PLATFORM_WINDOWS
  PROCESS_MEMORY_COUNTERS_EX pmc = {};
  if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
    stats.cpu_used_bytes = pmc.WorkingSetSize;
  }
#endif

  stats.gpu_allocated_bytes = _impl->gpu_allocated_bytes;

  if (_impl->memory_budget_supported) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mem_props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    mem_props2.pNext = &budget_props;
    vkGetPhysicalDeviceMemoryProperties2(_impl->physical_device, &mem_props2);
    for (uint32_t i = 0; i < mem_props2.memoryProperties.memoryHeapCount; ++i) {
      stats.gpu_driver_allocated_bytes += budget_props.heapUsage[i];
      stats.gpu_driver_budget_bytes += budget_props.heapBudget[i];
      if ((mem_props2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0u) {
        stats.gpu_device_local_allocated_bytes += budget_props.heapUsage[i];
        stats.gpu_device_local_budget_bytes += budget_props.heapBudget[i];
      }
    }
  } else {
    for (uint32_t i = 0; i < _impl->memory_properties.memoryHeapCount; ++i) {
      if ((_impl->memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0u) {
        stats.gpu_device_local_budget_bytes += _impl->memory_properties.memoryHeaps[i].size;
      }
    }
  }

  return stats;
}

VkPipelineLayout VKDevice::get_bindless_pipeline_layout() {
  return _impl->get_bindless_pipeline_layout().handle;
}

uint32_t VKDevice::get_max_push_constants_size() const {
  return _impl->max_push_constants_size;
}

uint64_t VKDevice::get_buffer_device_address(RHIBindlessHandle buffer) const {
  uint32_t index = _impl->buffers.get_index(buffer);
  if (index == UINT32_MAX) {
    return 0;
  }
  if (_impl->impl_vkGetBufferDeviceAddress == nullptr) {
    log::error("vkGetBufferDeviceAddress function is not loaded");
    return 0;
  }

  const auto& buffer_data = _impl->buffers.get_data(index);
  VkBufferDeviceAddressInfo address_info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  address_info.buffer = buffer_data.buffer;

  return _impl->impl_vkGetBufferDeviceAddress(_impl->device, &address_info);
}

uint64_t VKDevice::get_acceleration_structure_device_address(RHIBindlessHandle as_handle) {
  if (_impl->ray_tracing_supported == false) {
    return 0u;
  }

  uint32_t index = _impl->acceleration_structures.get_index(as_handle);
  if (index == UINT32_MAX) {
    return 0;
  }

  const auto& as_data = _impl->acceleration_structures.get_data(index);
  VkAccelerationStructureDeviceAddressInfoKHR address_info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  address_info.accelerationStructure = as_data.acceleration_structure;

  return _impl->impl_vkGetAccelerationStructureDeviceAddressKHR(_impl->device, &address_info);
}

uint64_t VKDevice::get_acceleration_structure_build_scratch_size(RHIBindlessHandle as_handle) {
  if (_impl->ray_tracing_supported == false) {
    return 0u;
  }

  uint32_t index = _impl->acceleration_structures.get_index(as_handle);
  if (index == UINT32_MAX) {
    return 0u;
  }

  const auto& as_data = _impl->acceleration_structures.get_data(index);
  return as_data.build_scratch_size;
}

RHIResult VKDevice::destroy_acceleration_structure(RHIBindlessHandle as_handle) {
  if (as_handle.valid() == false) {
    return RHIResult::Success;
  }

  uint32_t index = _impl->acceleration_structures.get_index(as_handle);
  if (index == UINT32_MAX) {
    return RHIResult::Success;
  }

  _impl->bindless_manager->unregister_acceleration_structure(as_handle);
  _impl->acceleration_structures.remove_handle(as_handle);

  // Collect buffer handle BEFORE destroying AS to avoid use-after-free
  // This prevents potential issues if destroy_buffer modifies as_to_buffer_map
  RHIBindlessHandle buffer_to_destroy = {};
  auto it = _impl->as_to_buffer_map.find(as_handle);
  if (it != _impl->as_to_buffer_map.end()) {
    buffer_to_destroy = it->second;
    _impl->as_to_buffer_map.erase(it);
  }

  // Destroy acceleration structure (no map access in callback)
  _impl->acceleration_structures.free_index(index, [this](const VKAccelerationStructureData& data) {
    _impl->queue_deferred_destruction(data);
  });

  // Now safely destroy the buffer after AS is destroyed
  if (buffer_to_destroy.valid()) {
    destroy_buffer(buffer_to_destroy);
  }

  return RHIResult::Success;
}

VkSemaphore VKDevice::get_vk_semaphore(RHISemaphore handle) const {
  if (handle.invalid())
    return VK_NULL_HANDLE;
  const VkSemaphore* sem = _impl->semaphores.get_data_ptr(handle);
  return sem ? *sem : VK_NULL_HANDLE;
}

RHICreateResult<RHISemaphore> VKDevice::create_semaphore() {
  VkSemaphoreCreateInfo create_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkSemaphore vk_sem = VK_NULL_HANDLE;
  if (vkCreateSemaphore(_impl->device, &create_info, nullptr, &vk_sem) != VK_SUCCESS) {
    return {RHIResult::InvalidHandle};
  }

  uint32_t index = _impl->semaphores.allocate_index();
  _impl->semaphores.get_data(index) = vk_sem;
  Handle h = Handle::construct(0, index, _impl->semaphores.get_generation(index));
  _impl->semaphores.set_handle_to_index(h, index);
  return {RHIResult::Success, h};
}

RHIResult VKDevice::destroy_semaphore(RHISemaphore semaphore) {
  if (semaphore.invalid())
    return RHIResult::Success;

  VkSemaphore* vk_sem = _impl->semaphores.get_data_ptr(semaphore);
  if (vk_sem == nullptr)
    return RHIResult::Success;

  if (*vk_sem != VK_NULL_HANDLE) {
    vkDestroySemaphore(_impl->device, *vk_sem, nullptr);
  }

  _impl->semaphores.free_index(_impl->semaphores.get_index(semaphore));
  _impl->semaphores.remove_handle(semaphore);
  return RHIResult::Success;
}

VkDevice VKDevice::get_vk_device() const {
  return _impl->device;
}

VkInstance VKDevice::get_vk_instance() const {
  return _impl->instance;
}

VkQueue VKDevice::get_graphics_queue() const {
  return _impl->graphics_queue;
}

VkCommandPool VKDevice::get_vk_command_pool(uint32_t index) const {
  return _impl->command_pools[index];
}

bool VKDevice::supports_timestamps() const {
#if ETX_PLATFORM_APPLE
  // MoltenVK may advertise timestamp capability, but query reset/write/readback has
  // proven unstable during interactive playback. Keep timestamps disabled on Apple
  // for the Phase 1 portability path.
  return false;
#else
  return (timestamp_valid_bits() > 0u);
#endif
}

bool VKDevice::supports_bindless() const {
  return _impl->bindless_supported;
}

bool VKDevice::supports_ray_tracing() const {
  return _impl->ray_tracing_supported;
}

bool VKDevice::supports_timestamp_stage(RHITimestampStage stage) const {
  if (supports_timestamps() == false) {
    return false;
  }
  if (_impl == nullptr) {
    return false;
  }

  switch (stage) {
    case RHITimestampStage::TopOfPipe:
    case RHITimestampStage::BottomOfPipe:
      return true;
    case RHITimestampStage::ComputeShader:
    case RHITimestampStage::AllCommands:
      return (_impl->properties.limits.timestampComputeAndGraphics != 0u);
    default:
      return false;
  }
}

uint32_t VKDevice::timestamp_valid_bits() const {
  if ((_impl == nullptr) || (_impl->physical_device == VK_NULL_HANDLE)) {
    return 0u;
  }

  uint32_t queue_family_count = 0u;
  vkGetPhysicalDeviceQueueFamilyProperties(_impl->physical_device, &queue_family_count, nullptr);
  if ((queue_family_count == 0u) || (_impl->graphics_queue_family >= queue_family_count)) {
    return 0u;
  }

  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(_impl->physical_device, &queue_family_count, queue_families.data());
  return queue_families[_impl->graphics_queue_family].timestampValidBits;
}

double VKDevice::timestamp_period_ns() const {
  if (_impl == nullptr) {
    return 0.0;
  }
  return static_cast<double>(_impl->properties.limits.timestampPeriod);
}

void VKDevice::set_current_frame_index(uint32_t index) {
  _impl->set_current_frame_index(index);
}

void VKDevice::reset_staging_buffer_for_frame(uint32_t frame_index) {
  _impl->reset_staging_buffer_for_frame(frame_index);
}

void VKDevice::process_deferred_destruction(uint32_t frame_index) {
  _impl->process_deferred_destruction(frame_index);
}

VKTextureData* VKDevice::get_texture_data(RHIBindlessHandle handle) const {
  uint32_t index = _impl->textures.get_index(handle);
  return (index != UINT32_MAX) ? &_impl->textures.get_data(index) : nullptr;
}

const VKAccelerationStructureData* VKDevice::get_acceleration_structure_data(RHIBindlessHandle handle) const {
  uint32_t index = _impl->acceleration_structures.get_index(handle);
  return (index != UINT32_MAX) ? &_impl->acceleration_structures.get_data(index) : nullptr;
}

PFN_vkCmdBuildAccelerationStructuresKHR VKDevice::get_vkCmdBuildAccelerationStructuresKHR() const {
  return _impl->impl_vkCmdBuildAccelerationStructuresKHR;
}

}  // namespace etx
