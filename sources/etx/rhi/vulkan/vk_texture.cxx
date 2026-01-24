#include <etx/rhi/vulkan/vk_texture.hxx>
#include <etx/rhi/vulkan/vk_utils.hxx>

#include <etx/core/log.hxx>

#include <cstring>
#include <algorithm>

namespace etx {

VKTexture::VKTexture(VkDevice device, VkPhysicalDevice physical_device, const RHITextureDesc& desc)
  : _device(device)
  , _physical_device(physical_device)
  , _desc(desc) {
  VkImageCreateInfo image_info = {};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = (desc.depth > 1) ? VK_IMAGE_TYPE_3D : (desc.height > 1) ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_1D;
  image_info.format = convert_format(desc.format);
  image_info.extent.width = desc.width;
  image_info.extent.height = desc.height;
  image_info.extent.depth = desc.depth;
  image_info.mipLevels = desc.mip_levels;
  image_info.arrayLayers = desc.array_layers;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.usage = convert_usage(desc.usage);
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkResult result = vkCreateImage(device, &image_info, nullptr, &_image);
  if (result != VK_SUCCESS) {
    log::error("Failed to create Vulkan image: %d", static_cast<int>(result));
    return;
  }

  VkMemoryRequirements mem_requirements;
  vkGetImageMemoryRequirements(device, _image, &mem_requirements);

  RHIResult alloc_result =
    allocate_memory(mem_requirements, desc.host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (alloc_result != RHIResult::Success) {
    log::error("Failed to allocate memory for texture");
    vkDestroyImage(device, _image, nullptr);
    _image = VK_NULL_HANDLE;
    return;
  }

  result = vkBindImageMemory(device, _image, _memory, 0);
  if (result != VK_SUCCESS) {
    log::error("Failed to bind image memory: %d", static_cast<int>(result));
    vkFreeMemory(device, _memory, nullptr);
    vkDestroyImage(device, _image, nullptr);
    _image = VK_NULL_HANDLE;
    _memory = VK_NULL_HANDLE;
    return;
  }

  RHIResult view_result = create_image_view();
  if (view_result != RHIResult::Success) {
    log::error("Failed to create image view");
    vkFreeMemory(device, _memory, nullptr);
    vkDestroyImage(device, _image, nullptr);
    _image = VK_NULL_HANDLE;
    _memory = VK_NULL_HANDLE;
    return;
  }
}

VKTexture::~VKTexture() {
  if (_image_view != VK_NULL_HANDLE) {
    vkDestroyImageView(_device, _image_view, nullptr);
    _image_view = VK_NULL_HANDLE;
  }

  if (_image != VK_NULL_HANDLE) {
    vkDestroyImage(_device, _image, nullptr);
    _image = VK_NULL_HANDLE;
  }

  if (_memory != VK_NULL_HANDLE) {
    vkFreeMemory(_device, _memory, nullptr);
    _memory = VK_NULL_HANDLE;
  }
}

RHIResult VKTexture::update_data(const void* data, uint32_t mip_level, uint32_t array_layer) {
  log::warning("VKTexture::update_data should not be called directly - use VKDevice::update_texture");
  return RHIResult::NotImplemented;
}

VkFormat VKTexture::convert_format(RHITextureFormat format) {
  return convert_rhi_format_to_vk(format);
}

VkImageUsageFlags VKTexture::convert_usage(RHITextureUsage usage) {
  using TextureUsage = std::underlying_type<RHITextureUsage>::type;
  TextureUsage usage_flags = static_cast<TextureUsage>(usage);

  VkImageUsageFlags flags = 0;

  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::Sampled)) {
    flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }
  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::Storage)) {
    flags |= VK_IMAGE_USAGE_STORAGE_BIT;
  }
  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::ColorAttachment)) {
    flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  }
  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::DepthStencilAttachment)) {
    flags |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  }
  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::TransferSrc)) {
    flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  }
  if (usage_flags & static_cast<TextureUsage>(RHITextureUsage::TransferDst)) {
    flags |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  }

  return flags;
}

uint32_t VKTexture::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
  uint32_t result = find_vulkan_memory_type(_physical_device, type_filter, properties);
  if (result == UINT32_MAX) {
    log::error("Failed to find suitable memory type for texture");
  }
  return result;
}

RHIResult VKTexture::allocate_memory(VkMemoryRequirements mem_requirements, VkMemoryPropertyFlags properties) {
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
    log::error("Failed to allocate device memory for texture: %d", static_cast<int>(result));
    return RHIResult::OutOfMemory;
  }

  return RHIResult::Success;
}

RHIResult VKTexture::create_image_view() {
  VkImageViewCreateInfo view_info = {};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = _image;
  view_info.viewType = (_desc.depth > 1) ? VK_IMAGE_VIEW_TYPE_3D : (_desc.height > 1) ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_1D;
  view_info.format = convert_format(_desc.format);

  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = _desc.mip_levels;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = _desc.array_layers;

  if (_desc.format == RHITextureFormat::D32_FLOAT || _desc.format == RHITextureFormat::D24_UNORM_S8_UINT || _desc.format == RHITextureFormat::D32_FLOAT_S8_UINT) {
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (_desc.format == RHITextureFormat::D24_UNORM_S8_UINT || _desc.format == RHITextureFormat::D32_FLOAT_S8_UINT) {
      view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  }

  VkResult result = vkCreateImageView(_device, &view_info, nullptr, &_image_view);
  if (result != VK_SUCCESS) {
    log::error("Failed to create image view: %d", static_cast<int>(result));
    return RHIResult::ValidationError;
  }

  return RHIResult::Success;
}

}  // namespace etx
