#include <etx/rhi/vulkan/vk_buffer.hxx>
#include <etx/rhi/vulkan/vk_utils.hxx>

#include <etx/core/log.hxx>

#include <cstring>
#include <algorithm>

namespace etx {

VKBuffer::VKBuffer(VkDevice device, VkPhysicalDevice physical_device, const RHIBufferDesc& desc)
  : _device(device)
  , _physical_device(physical_device)
  , _desc(desc) {
  VkBufferUsageFlags vk_usage = 0;

  using BufferUsage = std::underlying_type<RHIBufferUsage>::type;
  BufferUsage usage = static_cast<BufferUsage>(desc.usage);

  vk_usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

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

  VkBufferCreateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = desc.size;
  buffer_info.usage = vk_usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateBuffer(device, &buffer_info, nullptr, &_buffer);
  if (result != VK_SUCCESS) {
    log::error("Failed to create Vulkan buffer: %d", static_cast<int>(result));
    return;
  }

  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(device, _buffer, &mem_requirements);

  RHIResult alloc_result =
    allocate_memory(mem_requirements, desc.host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (alloc_result != RHIResult::Success) {
    log::error("Failed to allocate memory for buffer");
    vkDestroyBuffer(device, _buffer, nullptr);
    _buffer = VK_NULL_HANDLE;
    return;
  }

  result = vkBindBufferMemory(device, _buffer, _memory, 0);
  if (result != VK_SUCCESS) {
    log::error("Failed to bind buffer memory: %d", static_cast<int>(result));
    vkFreeMemory(device, _memory, nullptr);
    vkDestroyBuffer(device, _buffer, nullptr);
    _buffer = VK_NULL_HANDLE;
    _memory = VK_NULL_HANDLE;
    return;
  }
}

VKBuffer::~VKBuffer() {
  if (_mapped_ptr != nullptr) {
    vkUnmapMemory(_device, _memory);
    _mapped_ptr = nullptr;
  }

  if (_buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(_device, _buffer, nullptr);
    _buffer = VK_NULL_HANDLE;
  }

  if (_memory != VK_NULL_HANDLE) {
    vkFreeMemory(_device, _memory, nullptr);
    _memory = VK_NULL_HANDLE;
  }
}

RHIResult VKBuffer::update_data(const void* data, uint64_t size, uint64_t offset) {
  if (_buffer == VK_NULL_HANDLE) {
    return RHIResult::InvalidHandle;
  }

  if (offset + size > _desc.size) {
    log::error("Buffer update out of bounds: offset=%llu, size=%llu, buffer_size=%llu", offset, size, _desc.size);
    return RHIResult::InvalidArgument;
  }

  if (_desc.host_visible) {
    void* mapped_data = nullptr;
    VkResult result = vkMapMemory(_device, _memory, offset, size, 0, &mapped_data);
    if (result != VK_SUCCESS) {
      log::error("Failed to map buffer memory for update: %d", static_cast<int>(result));
      return RHIResult::ValidationError;
    }

    memcpy(mapped_data, data, size);
    vkUnmapMemory(_device, _memory);

    return RHIResult::Success;
  } else {
    log::warning("VKBuffer::update_data should not be called directly for device-local buffers - use VKDevice::update_buffer");
    return RHIResult::NotImplemented;
  }
}

RHIResult VKBuffer::map_memory(void** mapped_data) {
  if (_buffer == VK_NULL_HANDLE) {
    return RHIResult::InvalidHandle;
  }

  if (!_desc.host_visible) {
    log::error("Cannot map device-local buffer memory");
    return RHIResult::InvalidArgument;
  }

  if (_mapped_ptr != nullptr) {
    log::warning("Buffer memory already mapped");
    *mapped_data = _mapped_ptr;
    return RHIResult::Success;
  }

  VkResult result = vkMapMemory(_device, _memory, 0, _desc.size, 0, &_mapped_ptr);
  if (result != VK_SUCCESS) {
    log::error("Failed to map buffer memory: %d", static_cast<int>(result));
    return RHIResult::ValidationError;
  }

  *mapped_data = _mapped_ptr;
  return RHIResult::Success;
}

void VKBuffer::unmap_memory() {
  if (_mapped_ptr != nullptr) {
    vkUnmapMemory(_device, _memory);
    _mapped_ptr = nullptr;
  }
}

uint32_t VKBuffer::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
  uint32_t result = find_vulkan_memory_type(_physical_device, type_filter, properties);
  if (result == UINT32_MAX) {
    log::error("Failed to find suitable memory type");
  }
  return result;
}

RHIResult VKBuffer::allocate_memory(VkMemoryRequirements mem_requirements, VkMemoryPropertyFlags properties) {
  uint32_t memory_type_index = find_memory_type(mem_requirements.memoryTypeBits, properties);
  if (memory_type_index == UINT32_MAX) {
    return RHIResult::OutOfMemory;
  }

  VkMemoryAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex = memory_type_index;

  VkResult result = vkAllocateMemory(_device, &alloc_info, nullptr, &_memory);
  if (result != VK_SUCCESS) {
    log::error("Failed to allocate device memory: %d", static_cast<int>(result));
    return RHIResult::OutOfMemory;
  }

  return RHIResult::Success;
}

}  // namespace etx
