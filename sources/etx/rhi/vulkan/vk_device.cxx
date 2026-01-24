#include <etx/rhi/vulkan/vk_device.hxx>
#include <etx/rhi/vulkan/vk_rhi.hxx>
#include <etx/rhi/vulkan/vk_buffer.hxx>
#include <etx/rhi/vulkan/vk_texture.hxx>
#include <etx/rhi/vulkan/vk_sampler.hxx>
#include <etx/rhi/vulkan/vk_pipeline.hxx>
#include <etx/rhi/vulkan/vk_shader.hxx>

#include <etx/core/log.hxx>

#ifdef _WIN32
# define VK_USE_PLATFORM_WIN32_KHR
# include <windows.h>
#endif
#include <vulkan/vulkan.h>

#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>

namespace etx {

class VKDevice::Impl {
 public:
  Impl();
  ~Impl();

  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;

  uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
  uint32_t compute_queue_family = VK_QUEUE_FAMILY_IGNORED;
  uint32_t transfer_queue_family = VK_QUEUE_FAMILY_IGNORED;

  VkQueue graphics_queue = VK_NULL_HANDLE;
  VkQueue compute_queue = VK_NULL_HANDLE;
  VkQueue transfer_queue = VK_NULL_HANDLE;

  VkCommandPool command_pool = VK_NULL_HANDLE;

  VkPhysicalDeviceProperties properties = {};
  VkPhysicalDeviceMemoryProperties memory_properties = {};
  uint64_t min_uniform_buffer_offset_alignment = 0;
  uint64_t min_storage_buffer_offset_alignment = 0;

  bool bindless_supported = false;
  std::vector<const char*> enabled_extensions;

  RHIBindlessManager* bindless_manager = nullptr;

  ShaderCompiler* shader_compiler = nullptr;

  std::vector<VKBuffer*> buffers;
  std::vector<uint32_t> free_buffer_indices;
  std::unordered_map<RHIBindlessHandle, VKBuffer*> buffer_handle_map;

  std::vector<VKTexture*> textures;
  std::vector<uint32_t> free_texture_indices;
  std::unordered_map<RHIBindlessHandle, VKTexture*> texture_handle_map;

  std::vector<VKSampler*> samplers;
  std::vector<uint32_t> free_sampler_indices;
  std::unordered_map<RHIBindlessHandle, VKSampler*> sampler_handle_map;

  std::vector<VkShaderModule> shaders;
  std::vector<uint32_t> free_shader_indices;
  std::unordered_map<RHIShader, VkShaderModule> shader_handle_map;

  std::vector<VKComputePipeline*> compute_pipelines;
  std::vector<uint32_t> free_compute_pipeline_indices;
  std::unordered_map<RHIPipeline, VKComputePipeline*> compute_pipeline_handle_map;

  std::vector<VKGraphicsPipeline*> graphics_pipelines;
  std::vector<uint32_t> free_graphics_pipeline_indices;
  std::unordered_map<RHIPipeline, VKGraphicsPipeline*> graphics_pipeline_handle_map;

  std::unordered_map<RHIBindlessHandle, VkBuffer> buffer_map;

  bool initialize_instance();
  bool initialize_physical_device();
  bool initialize_device();

  uint32_t find_queue_family(VkQueueFlags required_flags, VkQueueFlags avoid_flags = 0);
  bool check_instance_extension_support(const std::vector<const char*>& extensions);
  bool check_extension_support(const std::vector<const char*>& extensions);
  bool check_bindless_support();

  uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
  RHIResult allocate_memory(VkMemoryRequirements mem_requirements, VkMemoryPropertyFlags properties, VkDeviceMemory& out_memory);

  uint32_t allocate_buffer_slot();
  void free_buffer_slot(uint32_t index);
  uint32_t allocate_texture_slot();
  void free_texture_slot(uint32_t index);
  uint32_t allocate_sampler_slot();
  void free_sampler_slot(uint32_t index);

  uint32_t allocate_shader_slot();
  void free_shader_slot(uint32_t index);

  uint32_t allocate_compute_pipeline_slot();
  void free_compute_pipeline_slot(uint32_t index);

  uint32_t allocate_graphics_pipeline_slot();
  void free_graphics_pipeline_slot(uint32_t index);
};

VKDevice::Impl::Impl() {
  if (!initialize_instance()) {
    log::error("Failed to initialize Vulkan instance");
    return;
  }

  if (!initialize_physical_device()) {
    log::error("Failed to initialize physical device");
    return;
  }

  if (!initialize_device()) {
    log::error("Failed to initialize device");
    return;
  }
}

VKDevice::Impl::~Impl() {
  bindless_manager = nullptr;

  if (command_pool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device, command_pool, nullptr);
    command_pool = VK_NULL_HANDLE;
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

bool VKDevice::Impl::initialize_instance() {
  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "etx-tracer";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "etx-rhi";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;

  std::vector<const char*> extensions = {
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
#ifdef _WIN32
    "VK_KHR_win32_surface",
#endif
  };

  if (!check_instance_extension_support(extensions)) {
    log::error("Required instance extensions not supported");
    return false;
  }

  VkInstanceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

  std::vector<const char*> layers;
#ifdef _DEBUG
  layers.push_back("VK_LAYER_KHRONOS_validation");
  create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
  create_info.ppEnabledLayerNames = layers.data();
#endif

  VkResult result = vkCreateInstance(&create_info, nullptr, &instance);
  if (result != VK_SUCCESS) {
    log::error("Failed to create Vulkan instance: %d", static_cast<int>(result));
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
  VkPhysicalDeviceProperties device_properties;

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
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_info);
  }

  std::vector<const char*> device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
    VK_KHR_MAINTENANCE_3_EXTENSION_NAME,
  };

  if (!check_extension_support(device_extensions)) {
    log::error("Required device extensions not supported");
    return false;
  }

  VkPhysicalDeviceFeatures device_features = {};
  device_features.samplerAnisotropy = VK_TRUE;

  VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing_features = {};
  descriptor_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
  descriptor_indexing_features.runtimeDescriptorArray = VK_TRUE;
  descriptor_indexing_features.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
  descriptor_indexing_features.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
  descriptor_indexing_features.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
  descriptor_indexing_features.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
  descriptor_indexing_features.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
  descriptor_indexing_features.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
  descriptor_indexing_features.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
  descriptor_indexing_features.descriptorBindingPartiallyBound = VK_TRUE;
  descriptor_indexing_features.descriptorBindingVariableDescriptorCount = VK_TRUE;

  VkDeviceCreateInfo device_create_info = {};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.pNext = &descriptor_indexing_features;
  device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
  device_create_info.pQueueCreateInfos = queue_create_infos.data();
  device_create_info.pEnabledFeatures = &device_features;
  device_create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
  device_create_info.ppEnabledExtensionNames = device_extensions.data();

  VkResult result = vkCreateDevice(physical_device, &device_create_info, nullptr, &device);
  if (result != VK_SUCCESS) {
    log::error("Failed to create Vulkan device: %d", static_cast<int>(result));
    return false;
  }

  vkGetDeviceQueue(device, graphics_queue_family, 0, &graphics_queue);
  vkGetDeviceQueue(device, compute_queue_family, 0, &compute_queue);
  vkGetDeviceQueue(device, transfer_queue_family, 0, &transfer_queue);

  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = graphics_queue_family;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  result = vkCreateCommandPool(device, &pool_info, nullptr, &command_pool);
  if (result != VK_SUCCESS) {
    log::error("Failed to create command pool: %d", static_cast<int>(result));
    return false;
  }

  bindless_supported = check_bindless_support();

  enabled_extensions = device_extensions;

  return true;
}

bool VKDevice::Impl::check_instance_extension_support(const std::vector<const char*>& extensions) {
  uint32_t extension_count;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);

  std::vector<VkExtensionProperties> available_extensions(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available_extensions.data());

  for (const char* required : extensions) {
    bool found = false;
    for (const auto& available : available_extensions) {
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
  VkPhysicalDeviceDescriptorIndexingProperties descriptor_indexing_props = {};
  descriptor_indexing_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;

  VkPhysicalDeviceProperties2 props2 = {};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
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

RHIResult VKDevice::Impl::allocate_memory(VkMemoryRequirements mem_requirements, VkMemoryPropertyFlags properties, VkDeviceMemory& out_memory) {
  uint32_t memory_type_index = find_memory_type(mem_requirements.memoryTypeBits, properties);
  if (memory_type_index == UINT32_MAX) {
    return RHIResult::OutOfMemory;
  }

  VkMemoryAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex = memory_type_index;

  VkResult result = vkAllocateMemory(device, &alloc_info, nullptr, &out_memory);
  if (result != VK_SUCCESS) {
    log::error("Failed to allocate device memory: %d", static_cast<int>(result));
    return RHIResult::OutOfMemory;
  }

  return RHIResult::Success;
}

uint32_t VKDevice::Impl::allocate_buffer_slot() {
  uint32_t index;

  if (!free_buffer_indices.empty()) {
    index = free_buffer_indices.back();
    free_buffer_indices.pop_back();
  } else {
    index = static_cast<uint32_t>(buffers.size());
    buffers.push_back(nullptr);
  }

  return index;
}

void VKDevice::Impl::free_buffer_slot(uint32_t index) {
  if (index < buffers.size()) {
    buffers[index] = nullptr;
    free_buffer_indices.push_back(index);
  }
}

uint32_t VKDevice::Impl::allocate_texture_slot() {
  uint32_t index;

  if (!free_texture_indices.empty()) {
    index = free_texture_indices.back();
    free_texture_indices.pop_back();
  } else {
    index = static_cast<uint32_t>(textures.size());
    textures.push_back(nullptr);
  }

  return index;
}

void VKDevice::Impl::free_texture_slot(uint32_t index) {
  if (index < textures.size()) {
    textures[index] = nullptr;
    free_texture_indices.push_back(index);
  }
}

uint32_t VKDevice::Impl::allocate_sampler_slot() {
  uint32_t index;

  if (!free_sampler_indices.empty()) {
    index = free_sampler_indices.back();
    free_sampler_indices.pop_back();
  } else {
    index = static_cast<uint32_t>(samplers.size());
    samplers.push_back(nullptr);
  }

  return index;
}

void VKDevice::Impl::free_sampler_slot(uint32_t index) {
  if (index < samplers.size()) {
    samplers[index] = nullptr;
    free_sampler_indices.push_back(index);
  }
}

uint32_t VKDevice::Impl::allocate_shader_slot() {
  uint32_t index;

  if (!free_shader_indices.empty()) {
    index = free_shader_indices.back();
    free_shader_indices.pop_back();
  } else {
    index = static_cast<uint32_t>(shaders.size());
    shaders.push_back(VK_NULL_HANDLE);
  }

  return index;
}

void VKDevice::Impl::free_shader_slot(uint32_t handle_index) {
  uint32_t array_index = handle_index;

  if (array_index < shaders.size()) {
    if (shaders[array_index] != VK_NULL_HANDLE) {
      vkDestroyShaderModule(device, shaders[array_index], nullptr);
      shaders[array_index] = VK_NULL_HANDLE;
    }
    free_shader_indices.push_back(array_index);
  }
}

uint32_t VKDevice::Impl::allocate_compute_pipeline_slot() {
  uint32_t index;

  if (!free_compute_pipeline_indices.empty()) {
    index = free_compute_pipeline_indices.back();
    free_compute_pipeline_indices.pop_back();
  } else {
    index = static_cast<uint32_t>(compute_pipelines.size());
    compute_pipelines.push_back(nullptr);
  }

  return index;
}

void VKDevice::Impl::free_compute_pipeline_slot(uint32_t index) {
  if (index < compute_pipelines.size()) {
    compute_pipelines[index] = nullptr;
    free_compute_pipeline_indices.push_back(index);
  }
}

VKComputePipeline* VKDevice::get_compute_pipeline(RHIPipeline handle) const {
  auto it = _impl->compute_pipeline_handle_map.find(handle);
  return (it != _impl->compute_pipeline_handle_map.end()) ? it->second : nullptr;
}

VKGraphicsPipeline* VKDevice::get_graphics_pipeline(RHIPipeline handle) const {
  auto it = _impl->graphics_pipeline_handle_map.find(handle);
  return (it != _impl->graphics_pipeline_handle_map.end()) ? it->second : nullptr;
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

  auto it = _impl->buffer_handle_map.find(handle);
  if (it != _impl->buffer_handle_map.end()) {
    return it->second->get_vk_buffer();
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

  auto it = _impl->texture_handle_map.find(handle);
  if (it != _impl->texture_handle_map.end()) {
    return it->second->get_vk_image();
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

void VKDevice::set_shader_compiler(ShaderCompiler* compiler) {
  _impl->shader_compiler = compiler;
}

ShaderCompiler* VKDevice::get_shader_compiler() const {
  return _impl->shader_compiler;
}

uint32_t VKDevice::Impl::allocate_graphics_pipeline_slot() {
  uint32_t index;

  if (!free_graphics_pipeline_indices.empty()) {
    index = free_graphics_pipeline_indices.back();
    free_graphics_pipeline_indices.pop_back();
  } else {
    index = static_cast<uint32_t>(graphics_pipelines.size());
    graphics_pipelines.push_back(nullptr);
  }

  return index;
}

void VKDevice::Impl::free_graphics_pipeline_slot(uint32_t index) {
  if (index < graphics_pipelines.size()) {
    graphics_pipelines[index] = nullptr;
    free_graphics_pipeline_indices.push_back(index);
  }
}

VKDevice::VKDevice()
  : _impl(new Impl()) {
}

void VKDevice::destroy_all_resources() {
  if (_impl == nullptr) {
    return;
  }

  uint32_t graphics_pipeline_count = static_cast<uint32_t>(_impl->graphics_pipeline_handle_map.size());
  uint32_t compute_pipeline_count = static_cast<uint32_t>(_impl->compute_pipeline_handle_map.size());
  uint32_t shader_count = static_cast<uint32_t>(_impl->shader_handle_map.size());
  uint32_t texture_count = static_cast<uint32_t>(_impl->texture_handle_map.size());
  uint32_t buffer_count = static_cast<uint32_t>(_impl->buffer_handle_map.size());
  uint32_t sampler_count = static_cast<uint32_t>(_impl->sampler_handle_map.size());

  std::vector<RHIPipeline> graphics_pipeline_handles;
  std::vector<RHIPipeline> compute_pipeline_handles;
  std::vector<RHIShader> shader_handles;
  std::vector<RHIBindlessHandle> texture_handles;
  std::vector<RHIBindlessHandle> buffer_handles;
  std::vector<RHIBindlessHandle> sampler_handles;

  for (const auto& pair : _impl->graphics_pipeline_handle_map) {
    graphics_pipeline_handles.push_back(pair.first);
  }
  for (const auto& pair : _impl->compute_pipeline_handle_map) {
    compute_pipeline_handles.push_back(pair.first);
  }
  for (const auto& pair : _impl->shader_handle_map) {
    shader_handles.push_back(pair.first);
  }
  for (const auto& pair : _impl->texture_handle_map) {
    texture_handles.push_back(pair.first);
  }
  for (const auto& pair : _impl->buffer_handle_map) {
    buffer_handles.push_back(pair.first);
  }
  for (const auto& pair : _impl->sampler_handle_map) {
    sampler_handles.push_back(pair.first);
  }

  for (RHIPipeline handle : graphics_pipeline_handles) {
    RHIResult result = destroy_pipeline(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy graphics pipeline %llu: %d", handle.value, static_cast<int>(result));
    }
  }
  _impl->graphics_pipeline_handle_map.clear();

  for (RHIPipeline handle : compute_pipeline_handles) {
    RHIResult result = destroy_pipeline(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy compute pipeline %llu: %d", handle.value, static_cast<int>(result));
    }
  }
  _impl->compute_pipeline_handle_map.clear();

  for (RHIShader handle : shader_handles) {
    RHIResult result = destroy_shader(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy shader %llu: %d", handle.value, static_cast<int>(result));
    }
  }
  _impl->shader_handle_map.clear();

  for (RHIBindlessHandle handle : texture_handles) {
    RHIResult result = destroy_texture(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy texture %llu: %d", handle, static_cast<int>(result));
    }
  }
  _impl->texture_handle_map.clear();

  for (RHIBindlessHandle handle : buffer_handles) {
    RHIResult result = destroy_buffer(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy buffer %llu: %d", handle, static_cast<int>(result));
    }
  }
  _impl->buffer_handle_map.clear();

  for (RHIBindlessHandle handle : sampler_handles) {
    RHIResult result = destroy_sampler(handle);
    if (result != RHIResult::Success) {
      log::warning("Failed to destroy sampler %llu: %d", handle, static_cast<int>(result));
    }
  }
  _impl->sampler_handle_map.clear();
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

  VKBuffer* buffer = new VKBuffer(_impl->device, _impl->physical_device, desc);
  if (buffer->get_vk_buffer() == VK_NULL_HANDLE) {
    log::error("Failed to create Vulkan buffer");
    delete buffer;
    return {RHIResult::OutOfMemory, {}};
  }

  RHIBindlessHandle handle = 0;
  RHIResult reg_result = _impl->bindless_manager->register_buffer(buffer->get_vk_buffer(), RHIResourceType::Buffer, handle);
  if (reg_result != RHIResult::Success) {
    log::error("Failed to register buffer with bindless manager");
    delete buffer;
    return {reg_result, {}};
  }

  buffer->set_bindless_handle(handle);
  uint32_t slot = _impl->allocate_buffer_slot();
  buffer->set_slot_index(slot);
  _impl->buffers[slot] = buffer;
  _impl->buffer_handle_map[handle] = buffer;

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

  VKTexture* texture = new VKTexture(_impl->device, _impl->physical_device, desc);
  if (texture->get_vk_image() == VK_NULL_HANDLE) {
    log::error("Failed to create Vulkan texture");
    delete texture;
    return {RHIResult::OutOfMemory, {}};
  }

  RHIBindlessHandle handle;
  RHIResult reg_result =
    _impl->bindless_manager->register_texture(texture->get_vk_image_view(), RHIResourceType::Texture, handle, static_cast<uint32_t>(desc.usage), texture->get_vk_image());
  if (reg_result != RHIResult::Success) {
    log::error("Failed to register texture with bindless manager");
    delete texture;
    return {reg_result, {}};
  }

  texture->set_bindless_handle(handle);
  uint32_t slot = _impl->allocate_texture_slot();
  texture->set_slot_index(slot);
  _impl->textures[slot] = texture;
  _impl->texture_handle_map[handle] = texture;

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

  VKSampler* sampler = new VKSampler(_impl->device, desc);
  VkSampler vk_sampler = sampler->get_vk_sampler();
  if (vk_sampler == VK_NULL_HANDLE) {
    log::error("Failed to create Vulkan sampler - null handle returned");
    delete sampler;
    return {RHIResult::OutOfMemory, {}};
  }

  RHIBindlessHandle handle;
  RHIResult reg_result = _impl->bindless_manager->register_sampler(sampler->get_vk_sampler(), RHIResourceType::Sampler, handle);
  if (reg_result != RHIResult::Success) {
    log::error("Failed to register sampler with bindless manager");
    delete sampler;
    return {reg_result, {}};
  }

  sampler->set_bindless_handle(handle);
  uint32_t slot = _impl->allocate_sampler_slot();
  sampler->set_slot_index(slot);
  _impl->samplers[slot] = sampler;
  _impl->sampler_handle_map[handle] = sampler;

  return {RHIResult::Success, handle};
}

RHICreateShaderResult VKDevice::create_shader(const RHIShaderDesc& desc) {
  if (_impl->device == VK_NULL_HANDLE) {
    log::error("Vulkan device not initialized");
    return {RHIResult::InvalidArgument, {}};
  }

  if (desc.spirv_data == nullptr || desc.spirv_size == 0) {
    log::error("Invalid SPIR-V data provided to create_shader");
    return {RHIResult::InvalidArgument, {}};
  }

  if (desc.spirv_size % 4 != 0) {
    log::error("SPIR-V data size not aligned to 4 bytes: %llu", desc.spirv_size);
    return {RHIResult::ValidationError, {}};
  }

  VkShaderModuleCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = desc.spirv_size;
  create_info.pCode = reinterpret_cast<const uint32_t*>(desc.spirv_data);

  VkShaderModule shader_module;
  VkResult result = vkCreateShaderModule(_impl->device, &create_info, nullptr, &shader_module);
  if (result != VK_SUCCESS) {
    log::error("Failed to create Vulkan shader module: %d", static_cast<int>(result));
    return {RHIResult::ValidationError, {}};
  }

  uint32_t slot = _impl->allocate_shader_slot();

  RHIShader shader_handle = {};
  shader_handle.value = slot;

  _impl->shaders[slot] = shader_module;
  _impl->shader_handle_map[shader_handle] = shader_module;

  return {RHIResult::Success, shader_handle};
}

RHICreateShaderResult VKDevice::create_shader_variant(const RHIShaderVariantDesc& desc) {
  if (_impl->shader_compiler == nullptr) {
    log::error("Shader compiler not available for variant creation");
    return {RHIResult::InvalidArgument, {}};
  }

  ShaderCompilationResult result = _impl->shader_compiler->get_or_compile_shader_variant(desc.hlsl_source, desc.entry_point, desc.stage, desc.source_name, desc.defines);

  if (result.result != RHIResult::Success) {
    log::error("Shader variant compilation failed: %s", result.error_message.c_str());
    return {result.result, {}};
  }

  if (result.spirv_data.size() % 4 != 0) {
    log::error("SPIR-V data size not aligned to 4 bytes: %zu", result.spirv_data.size());
    return {RHIResult::ValidationError, {}};
  }

  VkShaderModuleCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = result.spirv_data.size();
  create_info.pCode = reinterpret_cast<const uint32_t*>(result.spirv_data.data());

  VkShaderModule shader_module;
  VkResult vk_result = vkCreateShaderModule(_impl->device, &create_info, nullptr, &shader_module);
  if (vk_result != VK_SUCCESS) {
    log::error("Failed to create Vulkan shader module: %d", static_cast<int>(vk_result));
    return {RHIResult::ValidationError, {}};
  }

  uint32_t slot = _impl->allocate_shader_slot();

  RHIShader shader_handle = {};
  shader_handle.value = slot;

  _impl->shaders[slot] = shader_module;
  _impl->shader_handle_map[shader_handle] = shader_module;

  return {RHIResult::Success, shader_handle};
}

RHICreateShaderResult VKDevice::create_shader_from_file(const std::string& file_path, const std::string& entry_point, RHIShaderStage stage,
  const std::unordered_map<std::string, std::string>& defines) {
  if (!_impl->shader_compiler) {
    log::error("Shader compiler not available for file loading");
    return {RHIResult::InvalidArgument, {}};
  }

  ShaderCompilationResult result = _impl->shader_compiler->load_and_compile_shader_from_file(file_path, entry_point, stage, defines);

  if (result.result != RHIResult::Success) {
    log::error("Shader file compilation failed: %s", result.error_message.c_str());
    return {result.result, {}};
  }

  if (result.spirv_data.size() % 4 != 0) {
    log::error("SPIR-V data size not aligned to 4 bytes: %zu", result.spirv_data.size());
    return {RHIResult::ValidationError, {}};
  }

  VkShaderModuleCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = result.spirv_data.size();
  create_info.pCode = reinterpret_cast<const uint32_t*>(result.spirv_data.data());

  VkShaderModule shader_module;
  VkResult vk_result = vkCreateShaderModule(_impl->device, &create_info, nullptr, &shader_module);
  if (vk_result != VK_SUCCESS) {
    log::error("Failed to create Vulkan shader module from file: %d", static_cast<int>(vk_result));
    return {RHIResult::ValidationError, {}};
  }

  uint32_t slot = _impl->allocate_shader_slot();

  RHIShader shader_handle = {};
  shader_handle.value = slot;

  _impl->shaders[slot] = shader_module;
  _impl->shader_handle_map[shader_handle] = shader_module;

  std::filesystem::path file_path_obj(file_path);
  return {RHIResult::Success, shader_handle};
}

RHICreatePipelineResult VKDevice::create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) {
  if (_impl->bindless_manager == nullptr) {
    return {RHIResult::InvalidArgument, {}};
  }

  uint32_t slot = _impl->allocate_graphics_pipeline_slot();

  RHIPipeline pipeline_handle = {};
  pipeline_handle.value = slot;

  VKBindlessManager* vk_bindless = static_cast<VKBindlessManager*>(_impl->bindless_manager);
  VKGraphicsPipeline* pipeline = new VKGraphicsPipeline(_impl->device, vk_bindless->get_descriptor_set_layout());
  if (!pipeline->create_graphics_pipeline(desc)) {
    delete pipeline;
    _impl->free_graphics_pipeline_slot(slot);
    log::error("Failed to create graphics pipeline");
    return {RHIResult::ValidationError, {}};
  }

  pipeline->set_handle(pipeline_handle);

  _impl->graphics_pipelines[slot] = pipeline;
  _impl->graphics_pipeline_handle_map[pipeline_handle] = pipeline;

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

  VkDescriptorSetLayout bindless_layout = static_cast<VKBindlessManager*>(_impl->bindless_manager)->get_descriptor_set_layout();
  if (bindless_layout == VK_NULL_HANDLE) {
    log::error("Bindless descriptor set layout not available");
    return {RHIResult::InvalidArgument, {}};
  }

  VKComputePipeline* pipeline = new VKComputePipeline(_impl->device, bindless_layout);
  if (!pipeline->create_compute_pipeline(desc)) {
    log::error("Failed to create Vulkan compute pipeline");
    delete pipeline;
    return {RHIResult::OutOfMemory, {}};
  }

  uint32_t slot = _impl->allocate_compute_pipeline_slot();

  RHIPipeline pipeline_handle = {};
  pipeline_handle.value = slot;
  pipeline->set_handle(pipeline_handle);

  _impl->compute_pipelines[slot] = pipeline;
  _impl->compute_pipeline_handle_map[pipeline_handle] = pipeline;

  return {RHIResult::Success, pipeline_handle};
}

RHIResult VKDevice::destroy_buffer(RHIBindlessHandle buffer_handle) {
  auto it = _impl->buffer_handle_map.find(buffer_handle);
  if (it == _impl->buffer_handle_map.end()) {
    log::error("Buffer handle not found: %llu", buffer_handle);
    return RHIResult::InvalidHandle;
  }

  VKBuffer* buffer = it->second;

  RHIResult result = _impl->bindless_manager->unregister_buffer(buffer_handle);
  if (result != RHIResult::Success) {
    log::error("Failed to unregister buffer from bindless manager");
    return result;
  }

  uint32_t slot_index = buffer->get_slot_index();
  _impl->buffer_handle_map.erase(it);
  delete buffer;

  _impl->free_buffer_slot(slot_index);

  return RHIResult::Success;
}

RHIResult VKDevice::destroy_texture(RHIBindlessHandle texture_handle) {
  if (_impl->bindless_manager == nullptr) {
    return RHIResult::InvalidArgument;
  }

  auto it = _impl->texture_handle_map.find(texture_handle);
  if (it == _impl->texture_handle_map.end()) {
    log::error("Texture handle not found: %llu", texture_handle);
    return RHIResult::InvalidHandle;
  }

  VKTexture* texture = it->second;

  RHIResult result = _impl->bindless_manager->unregister_texture(texture_handle);
  if (result != RHIResult::Success) {
    log::error("Failed to unregister texture from bindless manager");
    return result;
  }

  _impl->texture_handle_map.erase(it);
  _impl->free_texture_slot(texture->get_slot_index());
  delete texture;

  return RHIResult::Success;
}

RHIResult VKDevice::destroy_sampler(RHIBindlessHandle sampler_handle) {
  if (_impl->bindless_manager == nullptr) {
    return RHIResult::InvalidArgument;
  }

  auto it = _impl->sampler_handle_map.find(sampler_handle);
  if (it == _impl->sampler_handle_map.end()) {
    log::error("Sampler handle not found: %llu", sampler_handle);
    return RHIResult::InvalidHandle;
  }

  VKSampler* sampler = it->second;

  RHIResult result = _impl->bindless_manager->unregister_sampler(sampler_handle);
  if (result != RHIResult::Success) {
    log::error("Failed to unregister sampler from bindless manager");
    return result;
  }

  _impl->sampler_handle_map.erase(it);
  _impl->free_sampler_slot(sampler->get_slot_index());
  delete sampler;

  return RHIResult::Success;
}

RHIResult VKDevice::destroy_shader(RHIShader shader) {
  auto it = _impl->shader_handle_map.find(shader);
  if (it == _impl->shader_handle_map.end()) {
    log::error("Shader handle %llu not found for destruction", shader);
    return RHIResult::InvalidHandle;
  }

  VkShaderModule vk_shader = it->second;

  _impl->free_shader_slot(static_cast<uint32_t>(shader.value));
  _impl->shader_handle_map.erase(it);

  return RHIResult::Success;
}

RHIResult VKDevice::destroy_pipeline(RHIPipeline pipeline_handle) {
  auto compute_it = _impl->compute_pipeline_handle_map.find(pipeline_handle);
  if (compute_it != _impl->compute_pipeline_handle_map.end()) {
    VKComputePipeline* compute_pipeline = compute_it->second;

    _impl->compute_pipeline_handle_map.erase(compute_it);
    uint32_t slot = static_cast<uint32_t>(pipeline_handle.value);
    if (slot < _impl->compute_pipelines.size()) {
      delete _impl->compute_pipelines[slot];
      _impl->compute_pipelines[slot] = nullptr;
      _impl->free_compute_pipeline_slot(slot);
    }

    return RHIResult::Success;
  }

  auto graphics_it = _impl->graphics_pipeline_handle_map.find(pipeline_handle);
  if (graphics_it != _impl->graphics_pipeline_handle_map.end()) {
    VKGraphicsPipeline* graphics_pipeline = graphics_it->second;

    _impl->graphics_pipeline_handle_map.erase(graphics_it);
    uint32_t slot = static_cast<uint32_t>(pipeline_handle.value);
    if (slot < _impl->graphics_pipelines.size()) {
      delete _impl->graphics_pipelines[slot];
      _impl->graphics_pipelines[slot] = nullptr;
      _impl->free_graphics_pipeline_slot(slot);
    }

    return RHIResult::Success;
  }

  log::error("Pipeline handle not found: %llu", pipeline_handle.value);
  return RHIResult::InvalidHandle;
}

RHIResult VKDevice::update_buffer(RHIBindlessHandle buffer_handle, const void* data, uint64_t size, uint64_t offset) {
  auto it = _impl->buffer_handle_map.find(buffer_handle);
  if (it == _impl->buffer_handle_map.end()) {
    log::error("Buffer handle not found: %llu", buffer_handle);
    return RHIResult::InvalidHandle;
  }

  VKBuffer* buffer = it->second;

  if (buffer->get_desc().host_visible) {
    return buffer->update_data(data, size, offset);
  }

  RHIBufferDesc staging_desc = {};
  staging_desc.size = size;
  staging_desc.usage = RHIBufferUsage::TransferSrc;
  staging_desc.host_visible = true;

  VKBuffer staging_buffer(_impl->device, _impl->physical_device, staging_desc);
  if (staging_buffer.get_vk_buffer() == VK_NULL_HANDLE) {
    log::error("Failed to create staging buffer for device-local buffer update");
    return RHIResult::OutOfMemory;
  }

  RHIResult staging_result = staging_buffer.update_data(data, size, 0);
  if (staging_result != RHIResult::Success) {
    log::error("Failed to update staging buffer data");
    return staging_result;
  }

  VkCommandBufferAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = _impl->command_pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer;
  VkResult result = vkAllocateCommandBuffers(_impl->device, &alloc_info, &command_buffer);
  if (result != VK_SUCCESS) {
    log::error("Failed to allocate command buffer for staging copy: %d", static_cast<int>(result));
    return RHIResult::ValidationError;
  }

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  result = vkBeginCommandBuffer(command_buffer, &begin_info);
  if (result != VK_SUCCESS) {
    log::error("Failed to begin command buffer for staging copy: %d", static_cast<int>(result));
    vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);
    return RHIResult::ValidationError;
  }

  VkBufferCopy copy_region = {};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = offset;
  copy_region.size = size;

  vkCmdCopyBuffer(command_buffer, staging_buffer.get_vk_buffer(), buffer->get_vk_buffer(), 1, &copy_region);

  result = vkEndCommandBuffer(command_buffer);
  if (result != VK_SUCCESS) {
    log::error("Failed to end command buffer for staging copy: %d", static_cast<int>(result));
    vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);
    return RHIResult::ValidationError;
  }

  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

  VkFence fence;
  result = vkCreateFence(_impl->device, &fence_info, nullptr, &fence);
  if (result != VK_SUCCESS) {
    log::error("Failed to create fence for staging copy: %d", static_cast<int>(result));
    vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);
    return RHIResult::ValidationError;
  }

  result = vkQueueSubmit(_impl->graphics_queue, 1, &submit_info, fence);
  if (result != VK_SUCCESS) {
    log::error("Failed to submit staging copy command: %d", static_cast<int>(result));
    vkDestroyFence(_impl->device, fence, nullptr);
    vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);
    return RHIResult::ValidationError;
  }

  result = vkWaitForFences(_impl->device, 1, &fence, VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS) {
    log::error("Failed to wait for staging copy completion: %d", static_cast<int>(result));
    vkDestroyFence(_impl->device, fence, nullptr);
    vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);
    return RHIResult::ValidationError;
  }

  vkDestroyFence(_impl->device, fence, nullptr);
  vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);

  return RHIResult::Success;
}

RHIResult VKDevice::update_texture(RHIBindlessHandle texture_handle, const void* data, uint32_t mip_level, uint32_t array_layer) {
  auto it = _impl->texture_handle_map.find(texture_handle);
  if (it == _impl->texture_handle_map.end()) {
    log::error("Texture handle not found: %llu", texture_handle);
    return RHIResult::InvalidHandle;
  }

  VKTexture* texture = it->second;
  const RHITextureDesc& desc = texture->get_desc();

  if (mip_level >= desc.mip_levels) {
    log::error("Mip level %u exceeds texture mip levels %u", mip_level, desc.mip_levels);
    return RHIResult::InvalidArgument;
  }

  if (array_layer >= desc.array_layers) {
    log::error("Array layer %u exceeds texture array layers %u", array_layer, desc.array_layers);
    return RHIResult::InvalidArgument;
  }

  uint32_t mip_width = std::max(desc.width >> mip_level, 1u);
  uint32_t mip_height = std::max(desc.height >> mip_level, 1u);
  uint32_t mip_depth = std::max(desc.depth >> mip_level, 1u);

  uint64_t bytes_per_pixel = 4;
  if (desc.format == RHITextureFormat::R8_UNORM)
    bytes_per_pixel = 1;
  else if (desc.format == RHITextureFormat::R8G8_UNORM)
    bytes_per_pixel = 2;
  else if (desc.format == RHITextureFormat::R8G8B8_UNORM)
    bytes_per_pixel = 3;
  else if (desc.format == RHITextureFormat::R8G8B8A8_UNORM)
    bytes_per_pixel = 4;
  else if (desc.format == RHITextureFormat::R32_FLOAT)
    bytes_per_pixel = 4;
  else if (desc.format == RHITextureFormat::R32G32_FLOAT)
    bytes_per_pixel = 8;
  else if (desc.format == RHITextureFormat::R32G32B32_FLOAT)
    bytes_per_pixel = 12;
  else if (desc.format == RHITextureFormat::R32G32B32A32_FLOAT)
    bytes_per_pixel = 16;

  uint64_t data_size = mip_width * mip_height * mip_depth * bytes_per_pixel;

  RHIBufferDesc staging_desc = {};
  staging_desc.size = data_size;
  staging_desc.usage = RHIBufferUsage::TransferSrc;
  staging_desc.host_visible = true;

  VKBuffer staging_buffer(_impl->device, _impl->physical_device, staging_desc);
  if (staging_buffer.get_vk_buffer() == VK_NULL_HANDLE) {
    log::error("Failed to create staging buffer for texture update");
    return RHIResult::OutOfMemory;
  }

  RHIResult staging_result = staging_buffer.update_data(data, data_size, 0);
  if (staging_result != RHIResult::Success) {
    log::error("Failed to update staging buffer data for texture");
    return staging_result;
  }

  VkCommandBufferAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = _impl->command_pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer;
  VkResult result = vkAllocateCommandBuffers(_impl->device, &alloc_info, &command_buffer);
  if (result != VK_SUCCESS) {
    log::error("Failed to allocate command buffer for texture upload: %d", static_cast<int>(result));
    return RHIResult::ValidationError;
  }

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  result = vkBeginCommandBuffer(command_buffer, &begin_info);
  if (result != VK_SUCCESS) {
    log::error("Failed to begin command buffer for texture upload: %d", static_cast<int>(result));
    vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);
    return RHIResult::ValidationError;
  }

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = texture->get_vk_image();
  barrier.oldLayout = texture->get_current_layout();
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = mip_level;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = array_layer;
  barrier.subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

  texture->set_current_layout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  VkBufferImageCopy copy_region = {};
  copy_region.bufferOffset = 0;
  copy_region.bufferRowLength = 0;
  copy_region.bufferImageHeight = 0;
  copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.imageSubresource.mipLevel = mip_level;
  copy_region.imageSubresource.baseArrayLayer = array_layer;
  copy_region.imageSubresource.layerCount = 1;
  copy_region.imageOffset = {0, 0, 0};
  copy_region.imageExtent = {mip_width, mip_height, mip_depth};

  vkCmdCopyBufferToImage(command_buffer, staging_buffer.get_vk_buffer(), texture->get_vk_image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

  texture->set_current_layout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  result = vkEndCommandBuffer(command_buffer);
  if (result != VK_SUCCESS) {
    log::error("Failed to end command buffer for texture upload: %d", static_cast<int>(result));
    vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);
    return RHIResult::ValidationError;
  }

  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

  VkFence fence;
  result = vkCreateFence(_impl->device, &fence_info, nullptr, &fence);
  if (result != VK_SUCCESS) {
    log::error("Failed to create fence for texture upload: %d", static_cast<int>(result));
    vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);
    return RHIResult::ValidationError;
  }

  result = vkQueueSubmit(_impl->graphics_queue, 1, &submit_info, fence);
  if (result != VK_SUCCESS) {
    log::error("Failed to submit texture upload command: %d", static_cast<int>(result));
    vkDestroyFence(_impl->device, fence, nullptr);
    vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);
    return RHIResult::ValidationError;
  }

  result = vkWaitForFences(_impl->device, 1, &fence, VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS) {
    log::error("Failed to wait for texture upload completion: %d", static_cast<int>(result));
    vkDestroyFence(_impl->device, fence, nullptr);
    vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);
    return RHIResult::ValidationError;
  }

  vkDestroyFence(_impl->device, fence, nullptr);
  vkFreeCommandBuffers(_impl->device, _impl->command_pool, 1, &command_buffer);

  return RHIResult::Success;
}

bool VKDevice::supports_bindless() const {
  return _impl->bindless_supported;
}

uint64_t VKDevice::get_min_uniform_buffer_offset_alignment() const {
  return _impl->min_uniform_buffer_offset_alignment;
}

uint64_t VKDevice::get_min_storage_buffer_offset_alignment() const {
  return _impl->min_storage_buffer_offset_alignment;
}

VkDevice VKDevice::get_vk_device() const {
  return _impl->device;
}

VkPhysicalDevice VKDevice::get_vk_physical_device() const {
  return _impl->physical_device;
}

void VKDevice::set_bindless_manager(RHIBindlessManager* manager) {
  _impl->bindless_manager = manager;
}

RHIResult VKDevice::reload_shader(RHIShader shader, const RHIShaderDesc& new_desc) {
  if (_impl->device == VK_NULL_HANDLE) {
    return RHIResult::InvalidArgument;
  }

  auto shader_it = _impl->shader_handle_map.find(shader);
  if (shader_it == _impl->shader_handle_map.end()) {
    log::error("Shader handle %llu not found for reloading", shader.value);
    return RHIResult::InvalidHandle;
  }

  if ((new_desc.spirv_data == nullptr) || (new_desc.spirv_size == 0)) {
    log::error("Invalid SPIR-V data provided to reload_shader");
    return RHIResult::InvalidArgument;
  }

  if ((new_desc.spirv_size % 4) != 0) {
    log::error("SPIR-V data size not aligned to 4 bytes: %llu", new_desc.spirv_size);
    return RHIResult::ValidationError;
  }

  uint32_t array_index = static_cast<uint32_t>(shader.value);
  if (array_index >= _impl->shaders.size()) {
    log::error("Shader handle %llu out of range", shader.value);
    return RHIResult::InvalidHandle;
  }

  VkShaderModuleCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = new_desc.spirv_size;
  create_info.pCode = reinterpret_cast<const uint32_t*>(new_desc.spirv_data);

  VkShaderModule new_module = VK_NULL_HANDLE;
  VkResult result = vkCreateShaderModule(_impl->device, &create_info, nullptr, &new_module);
  if (result != VK_SUCCESS) {
    log::error("Failed to create Vulkan shader module: %d", static_cast<int>(result));
    return RHIResult::ValidationError;
  }

  VkShaderModule old_module = _impl->shaders[array_index];
  _impl->shaders[array_index] = new_module;
  _impl->shader_handle_map[shader] = new_module;

  if (old_module != VK_NULL_HANDLE) {
    vkDestroyShaderModule(_impl->device, old_module, nullptr);
  }

  return RHIResult::Success;
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

}  // namespace etx
