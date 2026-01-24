#pragma once

#include <etx/rhi/rhi_types.hxx>

#include <vulkan/vulkan.h>

#include <vector>

namespace etx {

class VKBuffer {
 public:
  VKBuffer(VkDevice device, VkPhysicalDevice physical_device, const RHIBufferDesc& desc);
  ~VKBuffer();

  const RHIBufferDesc& get_desc() const {
    return _desc;
  }
  VkBuffer get_vk_buffer() const {
    return _buffer;
  }
  VkDeviceMemory get_vk_memory() const {
    return _memory;
  }
  uint64_t get_size() const {
    return _desc.size;
  }

  RHIResult update_data(const void* data, uint64_t size, uint64_t offset = 0);
  RHIResult map_memory(void** mapped_data);
  void unmap_memory();

  RHIBindlessHandle get_bindless_handle() const {
    return _bindless_handle;
  }
  void set_bindless_handle(RHIBindlessHandle handle) {
    _bindless_handle = handle;
  }

  uint32_t get_slot_index() const {
    return _slot_index;
  }
  void set_slot_index(uint32_t index) {
    _slot_index = index;
  }

 private:
  VkDevice _device = VK_NULL_HANDLE;
  VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
  RHIBufferDesc _desc = {};

  VkBuffer _buffer = VK_NULL_HANDLE;
  VkDeviceMemory _memory = VK_NULL_HANDLE;
  void* _mapped_ptr = nullptr;

  RHIBindlessHandle _bindless_handle = 0;

  uint32_t _slot_index = 0;

  uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
  RHIResult allocate_memory(VkMemoryRequirements mem_requirements, VkMemoryPropertyFlags properties);
};

}  // namespace etx
