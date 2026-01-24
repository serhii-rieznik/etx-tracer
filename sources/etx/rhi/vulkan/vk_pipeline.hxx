#pragma once

#include <etx/core/handle.hxx>
#include <etx/rhi/rhi_types.hxx>

#include <vulkan/vulkan.h>

namespace etx {

inline constexpr uint32_t kVKMaxPushConstantsSize = 128u;

class VKPipeline {
 public:
  VKPipeline(VkDevice device, VkDescriptorSetLayout bindless_layout);
  virtual ~VKPipeline();

  virtual bool create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) = 0;
  virtual bool create_compute_pipeline(const RHIComputePipelineDesc& desc) = 0;

  VkPipeline get_vk_pipeline() const {
    return _pipeline;
  }

  VkPipelineLayout get_vk_pipeline_layout() const {
    return _pipeline_layout;
  }

  bool is_valid() const {
    return _pipeline != VK_NULL_HANDLE && _pipeline_layout != VK_NULL_HANDLE;
  }

  void set_handle(RHIPipeline handle) {
    _handle = handle;
  }

  RHIPipeline get_handle() const {
    return _handle;
  }

 protected:
  VkDevice _device = VK_NULL_HANDLE;
  VkDescriptorSetLayout _bindless_layout = VK_NULL_HANDLE;
  VkPipeline _pipeline = VK_NULL_HANDLE;
  VkPipelineLayout _pipeline_layout = VK_NULL_HANDLE;
  RHIPipeline _handle = {};

  bool create_bindless_pipeline_layout();
};

class VKComputePipeline : public VKPipeline {
 public:
  VKComputePipeline(VkDevice device, VkDescriptorSetLayout bindless_layout);
  ~VKComputePipeline() override = default;

  bool create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) override {
    return false;
  }

  bool create_compute_pipeline(const RHIComputePipelineDesc& desc) override;
};

class VKGraphicsPipeline : public VKPipeline {
 public:
  VKGraphicsPipeline(VkDevice device, VkDescriptorSetLayout bindless_layout);
  ~VKGraphicsPipeline() override = default;

  bool create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) override;
  bool create_compute_pipeline(const RHIComputePipelineDesc& desc) override {
    return false;
  }
};

}  // namespace etx
