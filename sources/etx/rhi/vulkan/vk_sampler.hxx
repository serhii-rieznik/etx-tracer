#pragma once

#include <etx/rhi/rhi_types.hxx>

#include <vulkan/vulkan.h>

namespace etx {

class VKSampler {
 public:
  VKSampler(VkDevice device, const RHISamplerDesc& desc);
  ~VKSampler();

  const RHISamplerDesc& get_desc() const {
    return _desc;
  }
  VkSampler get_vk_sampler() const {
    return _sampler;
  }

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
  RHISamplerDesc _desc = {};

  VkSampler _sampler = VK_NULL_HANDLE;

  RHIBindlessHandle _bindless_handle = 0;

  uint32_t _slot_index = 0;

  VkFilter convert_filter(RHISamplerFilter filter);
  VkSamplerMipmapMode convert_mipmap_mode(RHISamplerMipmapMode mode);
  VkSamplerAddressMode convert_address_mode(RHISamplerAddressMode mode);
};

}  // namespace etx
