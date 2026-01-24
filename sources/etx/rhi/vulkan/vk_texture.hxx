#pragma once

#include <etx/rhi/rhi_types.hxx>

#include <vulkan/vulkan.h>

namespace etx {

class VKTexture {
 public:
  VKTexture(VkDevice device, VkPhysicalDevice physical_device, const RHITextureDesc& desc);
  ~VKTexture();

  const RHITextureDesc& get_desc() const {
    return _desc;
  }
  VkImage get_vk_image() const {
    return _image;
  }
  VkImageView get_vk_image_view() const {
    return _image_view;
  }
  VkDeviceMemory get_vk_memory() const {
    return _memory;
  }

  VkImageLayout get_current_layout() const {
    return _current_layout;
  }
  void set_current_layout(VkImageLayout layout) {
    _current_layout = layout;
  }

  uint32_t get_width() const {
    return _desc.width;
  }
  uint32_t get_height() const {
    return _desc.height;
  }
  uint32_t get_depth() const {
    return _desc.depth;
  }
  RHITextureFormat get_format() const {
    return _desc.format;
  }

  RHIResult update_data(const void* data, uint32_t mip_level = 0, uint32_t array_layer = 0);

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
  RHITextureDesc _desc = {};

  VkImage _image = VK_NULL_HANDLE;
  VkImageView _image_view = VK_NULL_HANDLE;
  VkDeviceMemory _memory = VK_NULL_HANDLE;

  VkImageLayout _current_layout = VK_IMAGE_LAYOUT_UNDEFINED;

  RHIBindlessHandle _bindless_handle = 0;

  uint32_t _slot_index = 0;

  VkFormat convert_format(RHITextureFormat format);
  VkImageUsageFlags convert_usage(RHITextureUsage usage);
  uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
  RHIResult allocate_memory(VkMemoryRequirements mem_requirements, VkMemoryPropertyFlags properties);
  RHIResult create_image_view();
};

}  // namespace etx
