#include <etx/rhi/vulkan/vk_rhi.hxx>

#include <etx/core/log.hxx>

namespace etx {

struct VKBindlessManager::Impl {
  Impl(VkDevice device, VkPhysicalDevice physical_device, uint32_t max_buffers, uint32_t max_textures, uint32_t max_samplers, uint32_t max_acceleration_structures);
  ~Impl();

  VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

  VkDevice device = VK_NULL_HANDLE;

  uint32_t max_buffers = kDefaultMaxBuffers;
  uint32_t max_textures = kDefaultMaxTextures;
  uint32_t max_samplers = kDefaultMaxSamplers;
  uint32_t max_acceleration_structures = kDefaultMaxAccelerationStructures;

  uint32_t buffer_count = 0;
  uint32_t texture_count = 0;
  uint32_t sampler_count = 0;
  uint32_t acceleration_structure_count = 0;

  VkPhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  struct ResourceEntry {
    uint32_t generation = 0;
    uint32_t descriptor_index = 0;
    RHIResourceType type = RHIResourceType::Buffer;
    bool valid = false;

    union {
      VkBuffer buffer;
      VkImage image;
      VkSampler sampler;
      VkAccelerationStructureKHR acceleration_structure;
    } vulkan_handle = {};
  };

  std::unordered_map<RHIBindlessHandle, ResourceEntry> handle_to_resource;
  std::vector<ResourceEntry> buffer_entries;
  std::vector<ResourceEntry> texture_entries;
  std::vector<ResourceEntry> sampler_entries;
  std::vector<ResourceEntry> acceleration_structure_entries;

  bool create_descriptor_set_layout();
  bool create_descriptor_pool();
  bool allocate_descriptor_set();
  bool initialize_bindless_arrays();

  RHIResult register_resource(RHIResourceType type, uint32_t& out_descriptor_index, RHIBindlessHandle& out_handle);
  RHIResult unregister_resource(RHIBindlessHandle handle);

  void update_descriptor_array(VkDescriptorType descriptor_type, uint32_t binding, uint32_t descriptor_index, VkDescriptorBufferInfo* buffer_info = nullptr,
    VkDescriptorImageInfo* image_info = nullptr, VkWriteDescriptorSetAccelerationStructureKHR* accel_info = nullptr);
};

VKBindlessManager::Impl::Impl(VkDevice vk_device, VkPhysicalDevice physical_device, uint32_t max_buf, uint32_t max_tex, uint32_t max_samp, uint32_t max_accel)
  : device(vk_device)
  , max_buffers(max_buf)
  , max_textures(max_tex)
  , max_samplers(max_samp)
  , max_acceleration_structures(max_accel) {
  if (device == VK_NULL_HANDLE) {
    log::error("Cannot initialize bindless manager with null device handle");
    return;
  }

  if (max_samplers == 0) {
    log::error("Cannot create bindless manager with 0 samplers - this will cause validation errors!");
    return;
  }

  buffer_entries.resize(max_buffers);
  texture_entries.resize(max_textures);
  sampler_entries.resize(max_samplers);
  acceleration_structure_entries.resize(max_acceleration_structures);

  if (!create_descriptor_set_layout()) {
    log::error("Failed to create bindless descriptor set layout");
    return;
  }

  if (!create_descriptor_pool()) {
    log::error("Failed to create bindless descriptor pool");
    return;
  }

  if (!allocate_descriptor_set()) {
    log::error("Failed to allocate bindless descriptor set");
    return;
  }
}

VKBindlessManager::Impl::~Impl() {
  if (descriptor_set != VK_NULL_HANDLE) {
    vkFreeDescriptorSets(device, descriptor_pool, 1, &descriptor_set);
    descriptor_set = VK_NULL_HANDLE;
  }

  if (descriptor_pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    descriptor_pool = VK_NULL_HANDLE;
  }

  if (descriptor_set_layout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    descriptor_set_layout = VK_NULL_HANDLE;
  }
}

bool VKBindlessManager::Impl::create_descriptor_set_layout() {
  std::vector<VkDescriptorSetLayoutBinding> bindings;

  VkDescriptorSetLayoutBinding buffer_binding = {};
  buffer_binding.binding = 0;
  buffer_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  buffer_binding.descriptorCount = max_buffers;
  buffer_binding.stageFlags = VK_SHADER_STAGE_ALL;
  bindings.push_back(buffer_binding);

  VkDescriptorSetLayoutBinding texture_binding = {};
  texture_binding.binding = 1;
  texture_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  texture_binding.descriptorCount = max_textures;
  texture_binding.stageFlags = VK_SHADER_STAGE_ALL;
  bindings.push_back(texture_binding);

  if (max_samplers > 0) {
    VkDescriptorSetLayoutBinding sampler_binding = {};
    sampler_binding.binding = 2;
    sampler_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    sampler_binding.descriptorCount = max_samplers;
    sampler_binding.stageFlags = VK_SHADER_STAGE_ALL;
    bindings.push_back(sampler_binding);
  }

  VkDescriptorSetLayoutBinding storage_texture_binding = {};
  storage_texture_binding.binding = 3;
  storage_texture_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  storage_texture_binding.descriptorCount = max_textures;
  storage_texture_binding.stageFlags = VK_SHADER_STAGE_ALL;
  bindings.push_back(storage_texture_binding);

  if (max_acceleration_structures > 0) {
    VkDescriptorSetLayoutBinding as_binding = {};
    as_binding.binding = 4;
    as_binding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    as_binding.descriptorCount = max_acceleration_structures;
    as_binding.stageFlags = VK_SHADER_STAGE_ALL;
    bindings.push_back(as_binding);
  }

  VkDescriptorSetLayoutBinding rw_buffer_binding = {};
  rw_buffer_binding.binding = 5;
  rw_buffer_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  rw_buffer_binding.descriptorCount = max_buffers;
  rw_buffer_binding.stageFlags = VK_SHADER_STAGE_ALL;
  bindings.push_back(rw_buffer_binding);

  VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
  VkDescriptorBindingFlags binding_flag_value = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
  std::vector<VkDescriptorBindingFlags> binding_flags_array(bindings.size(), binding_flag_value);

  binding_flags.bindingCount = static_cast<uint32_t>(binding_flags_array.size());
  binding_flags.pBindingFlags = binding_flags_array.data();

  if (bindings.size() > 2 && bindings[2].descriptorCount != max_samplers) {
    log::warning("Fixing sampler binding descriptor count from %u to %u", bindings[2].descriptorCount, max_samplers);
    bindings[2].descriptorCount = max_samplers;
  }

  VkDescriptorSetLayoutCreateInfo layout_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layout_info.pNext = &binding_flags;
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();
  layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT;

  if (etx_vk_call(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout)) != VK_SUCCESS) {
    return false;
  }

  return true;
}

bool VKBindlessManager::Impl::create_descriptor_pool() {
  std::vector<VkDescriptorPoolSize> pool_sizes;

  VkDescriptorPoolSize buffer_pool_size = {};
  buffer_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  buffer_pool_size.descriptorCount = max_buffers * 2u;
  pool_sizes.push_back(buffer_pool_size);

  VkDescriptorPoolSize texture_pool_size = {};
  texture_pool_size.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  texture_pool_size.descriptorCount = max_textures;
  pool_sizes.push_back(texture_pool_size);

  VkDescriptorPoolSize storage_texture_pool_size = {};
  storage_texture_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  storage_texture_pool_size.descriptorCount = max_textures;
  pool_sizes.push_back(storage_texture_pool_size);

  if (max_samplers > 0) {
    VkDescriptorPoolSize sampler_pool_size = {};
    sampler_pool_size.type = VK_DESCRIPTOR_TYPE_SAMPLER;
    sampler_pool_size.descriptorCount = max_samplers;
    pool_sizes.push_back(sampler_pool_size);
  }

  if (max_acceleration_structures > 0) {
    VkDescriptorPoolSize as_pool_size = {};
    as_pool_size.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    as_pool_size.descriptorCount = max_acceleration_structures;
    pool_sizes.push_back(as_pool_size);
  }

  VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
  pool_info.pPoolSizes = pool_sizes.data();
  pool_info.maxSets = 1;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

  if (etx_vk_call(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool)) != VK_SUCCESS) {
    return false;
  }

  return true;
}

bool VKBindlessManager::Impl::allocate_descriptor_set() {
  VkDescriptorSetAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  alloc_info.descriptorPool = descriptor_pool;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &descriptor_set_layout;

  if (etx_vk_call(vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set)) != VK_SUCCESS) {
    return false;
  }

  return true;
}

RHIResult VKBindlessManager::Impl::register_resource(RHIResourceType type, uint32_t& out_descriptor_index, RHIBindlessHandle& out_handle) {
  std::vector<ResourceEntry>* resource_array = nullptr;
  uint32_t max_count = 0;
  uint32_t* count_ptr = nullptr;

  switch (type) {
    case RHIResourceType::Buffer:
      resource_array = &buffer_entries;
      max_count = max_buffers;
      count_ptr = &buffer_count;
      break;
    case RHIResourceType::Texture:
      resource_array = &texture_entries;
      max_count = max_textures;
      count_ptr = &texture_count;
      break;
    case RHIResourceType::Sampler:
      resource_array = &sampler_entries;
      max_count = max_samplers;
      count_ptr = &sampler_count;
      break;
    case RHIResourceType::AccelerationStructure:
      resource_array = &acceleration_structure_entries;
      max_count = max_acceleration_structures;
      count_ptr = &acceleration_structure_count;
      break;

    default:
      return RHIResult::InvalidArgument;
  }

  uint32_t descriptor_index = 0;
  bool found_free_slot = false;
  // Reserve descriptor slot 0 as an invalid/null bindless index to match shader-side expectations.
  for (uint32_t i = 1; i < max_count; ++i) {
    if (!resource_array->at(i).valid) {
      descriptor_index = i;
      found_free_slot = true;
      break;
    }
  }

  if (!found_free_slot) {
    log::error("No free slots available for resource type %u", static_cast<uint32_t>(type));
    return RHIResult::OutOfMemory;
  }

  ResourceEntry& entry = resource_array->at(descriptor_index);
  entry.generation = (entry.generation + 1) & kRHIBindlessGenerationMask;
  entry.descriptor_index = descriptor_index;
  entry.type = type;
  entry.valid = true;

  (*count_ptr)++;

  out_descriptor_index = descriptor_index;
  out_handle = make_bindless_handle(type, entry.generation, descriptor_index);

  handle_to_resource[out_handle] = entry;

  return RHIResult::Success;
}

RHIResult VKBindlessManager::Impl::unregister_resource(RHIBindlessHandle handle) {
  auto it = handle_to_resource.find(handle);
  if (it == handle_to_resource.end()) {
    return RHIResult::InvalidHandle;
  }

  ResourceEntry& entry = it->second;
  if (!entry.valid) {
    return RHIResult::InvalidHandle;
  }

  entry.valid = false;

  std::vector<ResourceEntry>* resource_array = nullptr;
  switch (entry.type) {
    case RHIResourceType::Buffer:
      resource_array = &buffer_entries;
      break;
    case RHIResourceType::Texture:
      resource_array = &texture_entries;
      break;
    case RHIResourceType::Sampler:
      resource_array = &sampler_entries;
      break;
    case RHIResourceType::AccelerationStructure:
      resource_array = &acceleration_structure_entries;
      break;
    default:
      resource_array = nullptr;
      break;
  }

  if ((resource_array != nullptr) && (entry.descriptor_index < resource_array->size())) {
    resource_array->at(entry.descriptor_index).valid = false;
  }

  switch (entry.type) {
    case RHIResourceType::Buffer:
      buffer_count--;
      break;
    case RHIResourceType::Texture:
      texture_count--;
      break;
    case RHIResourceType::Sampler:
      sampler_count--;
      break;
    case RHIResourceType::AccelerationStructure:
      acceleration_structure_count--;
      break;
  }

  handle_to_resource.erase(it);

  return RHIResult::Success;
}

void VKBindlessManager::Impl::update_descriptor_array(VkDescriptorType descriptor_type, uint32_t binding, uint32_t descriptor_index, VkDescriptorBufferInfo* buffer_info,
  VkDescriptorImageInfo* image_info, VkWriteDescriptorSetAccelerationStructureKHR* accel_info) {
  uint32_t max_descriptors = 0;
  switch (binding) {
    case 0:
      max_descriptors = max_buffers;
      break;
    case 1:
      max_descriptors = max_textures;
      break;
    case 2:
      max_descriptors = max_samplers;
      break;
    case 3:
      max_descriptors = max_textures;
      break;
    case 4:
      max_descriptors = max_acceleration_structures;
      break;
    case 5:
      max_descriptors = max_buffers;
      break;
    default:
      log::error("Invalid binding %u in update_descriptor_array", binding);
      return;
  }

  if (descriptor_index >= max_descriptors) {
    log::error("Descriptor index %u out of bounds for binding %u (max: %u)", descriptor_index, binding, max_descriptors);
    return;
  }

  VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = descriptor_set;
  write.dstBinding = binding;
  write.dstArrayElement = descriptor_index;
  write.descriptorCount = 1;
  write.descriptorType = descriptor_type;

  if (buffer_info) {
    write.pBufferInfo = buffer_info;
  } else if (image_info) {
    write.pImageInfo = image_info;
  } else if (accel_info) {
    write.pNext = accel_info;
  }

  vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

VKBindlessManager::VKBindlessManager()
  : _impl(nullptr) {
}

void VKBindlessManager::initialize(VkDevice device, VkPhysicalDevice physical_device) {
  if (_impl != nullptr) {
    log::warning("Bindless manager already initialized - reinitializing with new capacities");
    delete _impl;
  }

  if (device == VK_NULL_HANDLE) {
    log::error("Cannot initialize bindless manager: invalid device handle");
    return;
  }

  if (physical_device == VK_NULL_HANDLE) {
    log::error("Cannot initialize bindless manager: invalid physical device handle");
    return;
  }

  VkPhysicalDeviceDescriptorIndexingProperties descriptor_indexing_props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};

  VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &descriptor_indexing_props;

  vkGetPhysicalDeviceProperties2(physical_device, &props2);

  bool bindless_supported = (descriptor_indexing_props.maxUpdateAfterBindDescriptorsInAllPools > 0) && (descriptor_indexing_props.maxPerStageDescriptorUpdateAfterBindSamplers > 0);

  if (bindless_supported == false) {
    log::error("Device does not support bindless descriptors - cannot initialize bindless manager");
    log::error("Required: maxUpdateAfterBindDescriptorsInAllPools > 0 and maxPerStageDescriptorUpdateAfterBindSamplers > 0");
    log::error("Actual values: maxUpdateAfterBindDescriptorsInAllPools=%u, maxPerStageDescriptorUpdateAfterBindSamplers=%u",
      descriptor_indexing_props.maxUpdateAfterBindDescriptorsInAllPools, descriptor_indexing_props.maxPerStageDescriptorUpdateAfterBindSamplers);
    return;
  }

  _impl = new Impl(device, physical_device, _stored_max_buffers, _stored_max_textures, _stored_max_samplers, _stored_max_acceleration_structures);

  vkGetPhysicalDeviceProperties2(physical_device, &props2);
  _impl->acceleration_structure_properties.pNext = nullptr;
  _impl->acceleration_structure_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
  props2.pNext = &_impl->acceleration_structure_properties;
  vkGetPhysicalDeviceProperties2(physical_device, &props2);
}

VKBindlessManager::~VKBindlessManager() {
  delete _impl;
}

void VKBindlessManager::set_max_buffers(uint32_t count) {
  _stored_max_buffers = count;
  if (_impl) {
    log::warning("Changed max buffers to %u, but bindless manager already initialized - layout unchanged", count);
  } else {
  }
}

void VKBindlessManager::set_max_textures(uint32_t count) {
  _stored_max_textures = count;
  if (_impl) {
    log::warning("Changed max textures to %u, but bindless manager already initialized - layout unchanged", count);
  } else {
  }
}

void VKBindlessManager::set_max_samplers(uint32_t count) {
  _stored_max_samplers = count;
  if (_impl) {
    log::warning("Changed max samplers to %u, but bindless manager already initialized - layout unchanged", count);
  } else {
  }
}

void VKBindlessManager::set_max_acceleration_structures(uint32_t count) {
  _stored_max_acceleration_structures = count;
  if (_impl) {
    log::warning("Changed max acceleration structures to %u, but bindless manager already initialized - layout unchanged", count);
  }
}

RHIResult VKBindlessManager::register_buffer(void* vk_buffer, RHIResourceType type, RHIBindlessHandle& out_handle) {
  if (!_impl) {
    return RHIResult::NotImplemented;
  }

  uint32_t descriptor_index = 0;
  RHIResult result = _impl->register_resource(RHIResourceType::Buffer, descriptor_index, out_handle);
  if (result != RHIResult::Success) {
    return result;
  }

  auto& entry = _impl->buffer_entries[descriptor_index];
  entry.vulkan_handle.buffer = static_cast<VkBuffer>(vk_buffer);

  _impl->handle_to_resource[out_handle] = entry;

  VkDescriptorBufferInfo buffer_info = {};
  buffer_info.buffer = static_cast<VkBuffer>(vk_buffer);
  buffer_info.offset = 0;
  buffer_info.range = VK_WHOLE_SIZE;

  _impl->update_descriptor_array(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, descriptor_index, &buffer_info);
  _impl->update_descriptor_array(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5, descriptor_index, &buffer_info);

  return RHIResult::Success;
}

RHIResult VKBindlessManager::unregister_buffer(RHIBindlessHandle handle) {
  if (!_impl) {
    return RHIResult::NotImplemented;
  }

  return _impl->unregister_resource(handle);
}

RHIResult VKBindlessManager::register_texture(void* vk_image_view, RHIResourceType type, RHIBindlessHandle& out_handle, uint32_t usage_flags, void* vk_image) {
  if (!_impl) {
    return RHIResult::NotImplemented;
  }

  uint32_t descriptor_index = 0;
  RHIResult result = _impl->register_resource(RHIResourceType::Texture, descriptor_index, out_handle);
  if (result != RHIResult::Success) {
    return result;
  }

  auto& entry = _impl->texture_entries[descriptor_index];
  entry.vulkan_handle.image = static_cast<VkImage>(vk_image);

  _impl->handle_to_resource[out_handle] = entry;

  using TextureUsage = std::underlying_type<RHITextureUsage>::type;
  TextureUsage usage_mask = static_cast<TextureUsage>(usage_flags);

  if (usage_mask & static_cast<TextureUsage>(RHITextureUsage::Sampled)) {
    VkDescriptorImageInfo sampled_info = {};
    sampled_info.imageView = static_cast<VkImageView>(vk_image_view);
    sampled_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    sampled_info.sampler = VK_NULL_HANDLE;

    _impl->update_descriptor_array(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, descriptor_index, nullptr, &sampled_info);
  }

  if (usage_mask & static_cast<TextureUsage>(RHITextureUsage::Storage)) {
    VkDescriptorImageInfo storage_info = {};
    storage_info.imageView = static_cast<VkImageView>(vk_image_view);
    storage_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    storage_info.sampler = VK_NULL_HANDLE;

    _impl->update_descriptor_array(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3, descriptor_index, nullptr, &storage_info);
  }

  return RHIResult::Success;
}

RHIResult VKBindlessManager::register_sampler(void* vk_sampler, RHIResourceType type, RHIBindlessHandle& out_handle) {
  if (!_impl) {
    return RHIResult::NotImplemented;
  }

  uint32_t descriptor_index = 0;
  RHIResult result = _impl->register_resource(RHIResourceType::Sampler, descriptor_index, out_handle);
  if (result != RHIResult::Success) {
    return result;
  }

  auto& entry = _impl->sampler_entries[descriptor_index];
  entry.vulkan_handle.sampler = static_cast<VkSampler>(vk_sampler);

  _impl->handle_to_resource[out_handle] = entry;

  VkDescriptorImageInfo sampler_info = {};
  sampler_info.sampler = static_cast<VkSampler>(vk_sampler);
  sampler_info.imageView = VK_NULL_HANDLE;
  sampler_info.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  _impl->update_descriptor_array(VK_DESCRIPTOR_TYPE_SAMPLER, 2, descriptor_index, nullptr, &sampler_info);
  return RHIResult::Success;
}

RHIResult VKBindlessManager::unregister_texture(RHIBindlessHandle handle) {
  if (!_impl) {
    return RHIResult::NotImplemented;
  }

  return _impl->unregister_resource(handle);
}

RHIResult VKBindlessManager::unregister_sampler(RHIBindlessHandle handle) {
  if (!_impl) {
    return RHIResult::NotImplemented;
  }

  return _impl->unregister_resource(handle);
}

RHIResult VKBindlessManager::register_acceleration_structure(const void* data, uint64_t size, RHIBindlessHandle& out_handle) {
  return RHIResult::NotImplemented;
}

RHIResult VKBindlessManager::register_acceleration_structure_vk(VkAccelerationStructureKHR vk_as, RHIAccelerationStructureType type, RHIBindlessHandle& out_handle) {
  if (!_impl) {
    return RHIResult::NotImplemented;
  }

  uint32_t descriptor_index = 0;
  RHIResult result = _impl->register_resource(RHIResourceType::AccelerationStructure, descriptor_index, out_handle);
  if (result != RHIResult::Success) {
    return result;
  }

  auto& entry = _impl->acceleration_structure_entries[descriptor_index];
  entry.vulkan_handle.acceleration_structure = vk_as;

  _impl->handle_to_resource[out_handle] = entry;

  if (type == RHIAccelerationStructureType::TopLevel) {
    VkWriteDescriptorSetAccelerationStructureKHR as_info = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    as_info.accelerationStructureCount = 1;
    as_info.pAccelerationStructures = &vk_as;

    _impl->update_descriptor_array(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4, descriptor_index, nullptr, nullptr, &as_info);
  }

  return RHIResult::Success;
}

RHIResult VKBindlessManager::unregister_acceleration_structure(RHIBindlessHandle handle) {
  if (!_impl) {
    return RHIResult::NotImplemented;
  }

  return _impl->unregister_resource(handle);
}

bool VKBindlessManager::is_valid_handle(RHIBindlessHandle handle) const {
  if (_impl == nullptr) {
    return false;
  }

  auto it = _impl->handle_to_resource.find(handle);
  if (it == _impl->handle_to_resource.end()) {
    return false;
  }

  const auto& entry = it->second;
  if (entry.valid == false) {
    return false;
  }

  uint32_t handle_generation = get_bindless_generation(handle);
  return entry.generation == handle_generation;
}

RHIResourceType VKBindlessManager::get_resource_type(RHIBindlessHandle handle) const {
  if (!_impl) {
    return RHIResourceType::Buffer;
  }

  auto it = _impl->handle_to_resource.find(handle);
  if (it == _impl->handle_to_resource.end()) {
    return RHIResourceType::Buffer;
  }

  return it->second.type;
}

uint32_t VKBindlessManager::get_max_buffers() const {
  return _impl ? _impl->max_buffers : kDefaultMaxBuffers;
}

uint32_t VKBindlessManager::get_max_textures() const {
  return _impl ? _impl->max_textures : kDefaultMaxTextures;
}

uint32_t VKBindlessManager::get_max_samplers() const {
  return _impl ? _impl->max_samplers : kDefaultMaxSamplers;
}

uint32_t VKBindlessManager::get_max_acceleration_structures() const {
  return _impl ? _impl->max_acceleration_structures : _stored_max_acceleration_structures;
}

uint32_t VKBindlessManager::get_buffer_count() const {
  return _impl ? _impl->buffer_count : 0;
}

uint32_t VKBindlessManager::get_texture_count() const {
  return _impl ? _impl->texture_count : 0;
}

uint32_t VKBindlessManager::get_sampler_count() const {
  return _impl ? _impl->sampler_count : 0;
}

uint32_t VKBindlessManager::get_acceleration_structure_count() const {
  return _impl ? _impl->acceleration_structure_count : 0;
}

VkDescriptorSetLayout VKBindlessManager::get_descriptor_set_layout() const {
  return _impl ? _impl->descriptor_set_layout : VK_NULL_HANDLE;
}

VkDescriptorSet VKBindlessManager::get_descriptor_set() const {
  return _impl ? _impl->descriptor_set : VK_NULL_HANDLE;
}

VkImage VKBindlessManager::get_vk_image(RHIBindlessHandle handle) const {
  if (!_impl) {
    return VK_NULL_HANDLE;
  }

  auto it = _impl->handle_to_resource.find(handle);
  if (it == _impl->handle_to_resource.end() || !it->second.valid || it->second.type != RHIResourceType::Texture) {
    return VK_NULL_HANDLE;
  }

  return it->second.vulkan_handle.image;
}

VkBuffer VKBindlessManager::get_vk_buffer(RHIBindlessHandle handle) const {
  if (!_impl) {
    return VK_NULL_HANDLE;
  }

  auto it = _impl->handle_to_resource.find(handle);
  if (it == _impl->handle_to_resource.end() || !it->second.valid || it->second.type != RHIResourceType::Buffer) {
    return VK_NULL_HANDLE;
  }

  return it->second.vulkan_handle.buffer;
}

VkSampler VKBindlessManager::get_vk_sampler(RHIBindlessHandle handle) const {
  if (!_impl) {
    return VK_NULL_HANDLE;
  }

  auto it = _impl->handle_to_resource.find(handle);
  if (it == _impl->handle_to_resource.end() || !it->second.valid || it->second.type != RHIResourceType::Sampler) {
    return VK_NULL_HANDLE;
  }

  return it->second.vulkan_handle.sampler;
}

VkAccelerationStructureKHR VKBindlessManager::get_vk_acceleration_structure(RHIBindlessHandle handle) const {
  if (!_impl) {
    return VK_NULL_HANDLE;
  }

  auto it = _impl->handle_to_resource.find(handle);
  if (it == _impl->handle_to_resource.end() || !it->second.valid || it->second.type != RHIResourceType::AccelerationStructure) {
    return VK_NULL_HANDLE;
  }

  return it->second.vulkan_handle.acceleration_structure;
}

}  // namespace etx
