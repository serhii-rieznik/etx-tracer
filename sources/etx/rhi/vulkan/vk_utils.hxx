#pragma once

#include <etx/rhi/rhi_types.hxx>

#include <vulkan/vulkan.h>

namespace etx {

static VkFormat convert_rhi_format_to_vk(RHITextureFormat format) {
  switch (format) {
    case RHITextureFormat::R8_UNORM:
      return VK_FORMAT_R8_UNORM;
    case RHITextureFormat::R8G8_UNORM:
      return VK_FORMAT_R8G8_UNORM;
    case RHITextureFormat::R8G8B8_UNORM:
      return VK_FORMAT_R8G8B8_UNORM;
    case RHITextureFormat::R8G8B8A8_UNORM:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case RHITextureFormat::R32_FLOAT:
      return VK_FORMAT_R32_SFLOAT;
    case RHITextureFormat::R32G32_FLOAT:
      return VK_FORMAT_R32G32_SFLOAT;
    case RHITextureFormat::R32G32B32_FLOAT:
      return VK_FORMAT_R32G32B32_SFLOAT;
    case RHITextureFormat::R32G32B32A32_FLOAT:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    case RHITextureFormat::R8G8B8A8_SRGB:
      return VK_FORMAT_R8G8B8A8_SRGB;
    case RHITextureFormat::B8G8R8A8_SRGB:
      return VK_FORMAT_B8G8R8A8_SRGB;
    case RHITextureFormat::D32_FLOAT:
      return VK_FORMAT_D32_SFLOAT;
    case RHITextureFormat::D24_UNORM_S8_UINT:
      return VK_FORMAT_D24_UNORM_S8_UINT;
    case RHITextureFormat::D32_FLOAT_S8_UINT:
      return VK_FORMAT_D32_SFLOAT_S8_UINT;
    default:
      return VK_FORMAT_UNDEFINED;
  }
}

static uint32_t find_vulkan_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  return UINT32_MAX;
}

static RHITextureFormat vk_format_to_rhi(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8_UNORM:
      return RHITextureFormat::R8_UNORM;
    case VK_FORMAT_R8G8_UNORM:
      return RHITextureFormat::R8G8_UNORM;
    case VK_FORMAT_R8G8B8_UNORM:
      return RHITextureFormat::R8G8B8_UNORM;
    case VK_FORMAT_R8G8B8A8_UNORM:
      return RHITextureFormat::R8G8B8A8_UNORM;
    case VK_FORMAT_R32_SFLOAT:
      return RHITextureFormat::R32_FLOAT;
    case VK_FORMAT_R32G32_SFLOAT:
      return RHITextureFormat::R32G32_FLOAT;
    case VK_FORMAT_R32G32B32_SFLOAT:
      return RHITextureFormat::R32G32B32_FLOAT;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return RHITextureFormat::R32G32B32A32_FLOAT;
    case VK_FORMAT_R8G8B8A8_SRGB:
      return RHITextureFormat::R8G8B8A8_SRGB;
    case VK_FORMAT_B8G8R8A8_SRGB:
      return RHITextureFormat::B8G8R8A8_SRGB;
    case VK_FORMAT_D32_SFLOAT:
      return RHITextureFormat::D32_FLOAT;
    case VK_FORMAT_D24_UNORM_S8_UINT:
      return RHITextureFormat::D24_UNORM_S8_UINT;
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return RHITextureFormat::D32_FLOAT_S8_UINT;
    default:
      return RHITextureFormat::Undefined;
  }
}

}  // namespace etx
