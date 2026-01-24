#include <etx/rhi/vulkan/vk_sampler.hxx>

#include <etx/core/log.hxx>

#include <algorithm>

namespace etx {

VKSampler::VKSampler(VkDevice device, const RHISamplerDesc& desc)
  : _device(device)
  , _desc(desc) {
  VkSamplerCreateInfo sampler_info = {};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = convert_filter(desc.mag_filter);
  sampler_info.minFilter = convert_filter(desc.min_filter);
  sampler_info.mipmapMode = convert_mipmap_mode(desc.mipmap_mode);
  sampler_info.addressModeU = convert_address_mode(desc.address_mode_u);
  sampler_info.addressModeV = convert_address_mode(desc.address_mode_v);
  sampler_info.addressModeW = convert_address_mode(desc.address_mode_w);
  sampler_info.mipLodBias = 0.0f;
  sampler_info.anisotropyEnable = (desc.max_anisotropy > 1.0f) ? VK_TRUE : VK_FALSE;
  sampler_info.maxAnisotropy = std::max(1.0f, desc.max_anisotropy);
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.minLod = 0.0f;
  sampler_info.maxLod = VK_LOD_CLAMP_NONE;
  sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;

  VkResult result = vkCreateSampler(device, &sampler_info, nullptr, &_sampler);
  if (result != VK_SUCCESS) {
    log::error("Failed to create Vulkan sampler: %d", static_cast<int>(result));
    _sampler = VK_NULL_HANDLE;
    return;
  }

  if (_sampler == VK_NULL_HANDLE) {
    log::error("Vulkan sampler creation returned null handle");
    return;
  }
}

VKSampler::~VKSampler() {
  if (_sampler != VK_NULL_HANDLE) {
    vkDestroySampler(_device, _sampler, nullptr);
    _sampler = VK_NULL_HANDLE;
  }
}

VkFilter VKSampler::convert_filter(RHISamplerFilter filter) {
  switch (filter) {
    case RHISamplerFilter::Nearest:
      return VK_FILTER_NEAREST;
    case RHISamplerFilter::Linear:
      return VK_FILTER_LINEAR;
    default:
      return VK_FILTER_LINEAR;
  }
}

VkSamplerMipmapMode VKSampler::convert_mipmap_mode(RHISamplerMipmapMode mode) {
  switch (mode) {
    case RHISamplerMipmapMode::Nearest:
      return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    case RHISamplerMipmapMode::Linear:
      return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    default:
      return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }
}

VkSamplerAddressMode VKSampler::convert_address_mode(RHISamplerAddressMode mode) {
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

}  // namespace etx
