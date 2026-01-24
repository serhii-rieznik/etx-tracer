#pragma once

#ifdef _WIN32
# define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <vulkan/vulkan.h>

#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

namespace etx {

class VKComputePipeline;
class VKGraphicsPipeline;

struct VKContext : RHIContext {
  VKContext();
  ~VKContext() override;

  RHIDevice* get_device() override;
  RHIBindlessManager* get_bindless_manager() override;

  void initialize_for_headless();

  void create_swapchain(const void* native_window, uint32_t width, uint32_t height) override;
  void destroy_swapchain() override;
  void resize_swapchain(uint32_t width, uint32_t height) override;
  RHITexture get_current_swapchain_texture() override;
  RHITextureFormat get_swapchain_format() const override;
  void present() override;
  void present_with_frame_index(uint32_t frame_index) override;

  void begin_frame() override;
  void end_frame() override;
  uint32_t get_current_frame_index() const override;
  uint32_t get_sampler_index(RHISamplerType type) const override;

  RHICommandBuffer* get_command_buffer() override;
  void submit_command_buffer(RHICommandBuffer* command_buffer) override;

  VkDevice get_vk_device() const;
  VkCommandPool get_vk_command_pool() const;
  VkFence get_current_frame_fence() const;
  VkQueue get_graphics_queue() const;

 public:
  class Impl;
  Impl* _impl = nullptr;
};

struct VKDevice : RHIDevice {
  VKDevice();
  ~VKDevice() override;

 public:
  class Impl;

  VkDevice get_vk_device() const;
  VkPhysicalDevice get_vk_physical_device() const;
  void set_bindless_manager(RHIBindlessManager* manager);

  void destroy_all_resources();

 private:
  friend class VKContext;

  RHICreateBindlessResult create_buffer(const RHIBufferDesc& desc) override;
  RHICreateBindlessResult create_texture(const RHITextureDesc& desc) override;
  RHICreateBindlessResult create_sampler(const RHISamplerDesc& desc) override;
  RHICreateShaderResult create_shader(const RHIShaderDesc& desc) override;
  RHICreateShaderResult create_shader_variant(const RHIShaderVariantDesc& desc) override;
  RHICreateShaderResult create_shader_from_file(const std::string& file_path, const std::string& entry_point, RHIShaderStage stage,
    const std::unordered_map<std::string, std::string>& defines = {}) override;
  RHICreatePipelineResult create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) override;
  RHICreatePipelineResult create_compute_pipeline(const RHIComputePipelineDesc& desc) override;

  RHIResult destroy_buffer(RHIBindlessHandle buffer) override;
  RHIResult destroy_texture(RHIBindlessHandle texture) override;
  RHIResult destroy_sampler(RHIBindlessHandle sampler) override;
  RHIResult destroy_shader(RHIShader shader) override;
  RHIResult destroy_pipeline(RHIPipeline pipeline) override;

  RHIResult update_buffer(RHIBindlessHandle buffer, const void* data, uint64_t size, uint64_t offset = 0) override;
  RHIResult update_texture(RHIBindlessHandle texture, const void* data, uint32_t mip_level = 0, uint32_t array_layer = 0) override;

  RHIResult reload_shader(RHIShader shader, const RHIShaderDesc& new_desc) override;
  RHIResult reload_graphics_pipeline(RHIPipeline pipeline, const RHIGraphicsPipelineDesc& new_desc) override;
  RHIResult reload_compute_pipeline(RHIPipeline pipeline, const RHIComputePipelineDesc& new_desc) override;

  bool supports_bindless() const override;
  uint64_t get_min_uniform_buffer_offset_alignment() const override;
  uint64_t get_min_storage_buffer_offset_alignment() const override;

 public:
  VKComputePipeline* get_compute_pipeline(RHIPipeline handle) const;
  VKGraphicsPipeline* get_graphics_pipeline(RHIPipeline handle) const;
  VkBuffer get_vk_buffer_from_bindless(RHIBindlessHandle handle) const;
  VkImage get_vk_image_from_bindless(RHIBindlessHandle handle) const;

  void set_shader_compiler(ShaderCompiler* compiler);
  ShaderCompiler* get_shader_compiler() const;
  class Impl;
  Impl* _impl = nullptr;
};

struct VKBindlessManager : RHIBindlessManager {
  VKBindlessManager();
  ~VKBindlessManager() override;

  void initialize(VkDevice device, VkPhysicalDevice physical_device);
  bool is_initialized() const {
    return _impl != nullptr;
  }

  void set_max_buffers(uint32_t count) override;
  void set_max_textures(uint32_t count) override;
  void set_max_samplers(uint32_t count) override;
  void set_max_acceleration_structures(uint32_t count) override;

  RHIResult register_buffer(void* vk_buffer, RHIResourceType type, RHIBindlessHandle& out_handle) override;
  RHIResult register_texture(void* vk_image_view, RHIResourceType type, RHIBindlessHandle& out_handle, uint32_t usage_flags, void* vk_image = nullptr) override;
  RHIResult register_sampler(void* vk_sampler, RHIResourceType type, RHIBindlessHandle& out_handle) override;

  RHIResult unregister_buffer(RHIBindlessHandle handle) override;
  RHIResult unregister_texture(RHIBindlessHandle handle) override;
  RHIResult unregister_sampler(RHIBindlessHandle handle) override;

  RHIResult register_acceleration_structure(const void* data, uint64_t size, RHIBindlessHandle& out_handle) override;
  RHIResult unregister_acceleration_structure(RHIBindlessHandle handle) override;

  VkImage get_vk_image(RHIBindlessHandle handle) const;
  VkBuffer get_vk_buffer(RHIBindlessHandle handle) const;
  VkSampler get_vk_sampler(RHIBindlessHandle handle) const;

  bool is_valid_handle(RHIBindlessHandle handle) const override;
  RHIResourceType get_resource_type(RHIBindlessHandle handle) const override;

  uint32_t get_max_buffers() const override;
  uint32_t get_max_textures() const override;
  uint32_t get_max_samplers() const override;
  uint32_t get_max_acceleration_structures() const override;

  uint32_t get_buffer_count() const override;
  uint32_t get_texture_count() const override;
  uint32_t get_sampler_count() const override;
  uint32_t get_acceleration_structure_count() const override;

  VkDescriptorSetLayout get_descriptor_set_layout() const;
  VkDescriptorSet get_descriptor_set() const;

 private:
  uint32_t _stored_max_buffers = kDefaultMaxBuffers;
  uint32_t _stored_max_textures = kDefaultMaxTextures;
  uint32_t _stored_max_samplers = kDefaultMaxSamplers;
  uint32_t _stored_max_acceleration_structures = kDefaultMaxAccelerationStructures;

  class Impl;
  Impl* _impl = nullptr;
};

class VKBindlessManager::Impl {
 public:
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

  struct ResourceEntry {
    uint32_t generation = 0;
    uint32_t descriptor_index = 0;
    RHIResourceType type = RHIResourceType::Buffer;
    bool valid = false;

    union {
      VkBuffer buffer;
      VkImage image;
      VkSampler sampler;
      void* acceleration_structure;
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

struct VKCommandBuffer : RHICommandBuffer {
  VKCommandBuffer();
  ~VKCommandBuffer() override;

  void initialize(VKContext* ctx);
  void destroy_resources();
  bool is_initialized() const;
  VkCommandBuffer get_vk_command_buffer() const;
  bool is_recording() const;

  void begin() override;
  void end() override;
  void reset() override;

  void reset_internal_state();

 private:
  friend class VKContext;

  void buffer_barrier(RHIBindlessHandle buffer, RHIResourceState old_state, RHIResourceState new_state) override;
  void texture_barrier(RHIBindlessHandle texture, RHIResourceState old_state, RHIResourceState new_state) override;

  void set_buffer_state(RHIBindlessHandle buffer, RHIResourceState state) override;
  void set_texture_state(RHIBindlessHandle texture, RHIResourceState state) override;
  RHIResourceState get_buffer_state(RHIBindlessHandle buffer) const override;
  RHIResourceState get_texture_state(RHIBindlessHandle texture) const override;

  void begin_render_pass(uint32_t color_attachment_count, RHIBindlessHandle* color_attachments, const float* clear_colors = nullptr,
    RHIBindlessHandle depth_attachment = {}) override;
  void end_render_pass() override;

  void set_viewport(const RHIViewport& viewport) override;
  void set_scissor(const RHIRect& scissor) override;
  void set_pipeline(RHIPipeline pipeline) override;

  void push_constants(const void* data, uint32_t size, uint32_t offset = 0) override;

  void draw(const RHIDrawDesc& desc) override;
  void draw_indexed(const RHIIndexedDrawDesc& desc, RHIBindlessHandle index_buffer) override;

  void dispatch(const RHIDispatchDesc& desc) override;

  void copy_buffer(RHIBindlessHandle src, RHIBindlessHandle dst, uint64_t size, uint64_t src_offset = 0, uint64_t dst_offset = 0) override;
  void copy_buffer_to_texture(RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0) override;
  void copy_texture_to_buffer(RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0) override;

  void set_debug_name(const char* name) override;

 private:
  void ensure_texture_layout(RHIBindlessHandle texture, VkImageLayout required_layout);
  void set_scissor_from_viewport(const RHIViewport& viewport);
  class Impl;
  Impl* _impl = nullptr;
};

}  // namespace etx
