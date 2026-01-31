#pragma once

#include <etx/rhi/rhi.hxx>

namespace etx {

struct MTContext : RHIContext {
  MTContext();
  ~MTContext() override;

  RHIDevice* get_device() override;
  RHIBindlessManager* get_bindless_manager() override;

  void create_swapchain(const void* native_window, uint32_t width, uint32_t height) override;
  void destroy_swapchain() override;
  void resize_swapchain(uint32_t width, uint32_t height) override;
  RHITexture get_current_swapchain_texture() override;
  RHITextureFormat get_swapchain_format() const override;

  void begin_frame() override;
  void present() override;

  uint32_t get_current_frame_index() const override;
  uint32_t get_sampler_index(RHISamplerType type) const override;

  RHICommandBuffer* get_command_buffer() override;
  void submit_command_buffer(RHICommandBuffer* command_buffer) override;

 private:
  class Impl;
  Impl* _impl = nullptr;
};

struct MTDevice : RHIDevice {
  MTDevice();
  ~MTDevice() override;

  RHICreateBindlessResult create_buffer(const RHIBufferDesc& desc) override;
  RHICreateBindlessResult create_texture(const RHITextureDesc& desc) override;
  RHICreateBindlessResult create_sampler(const RHISamplerDesc& desc) override;
  RHICreateShaderResult create_shader(const RHIShaderDesc& desc) override;
  RHICreateShaderResult create_shader_variant(const RHIShaderVariantDesc& desc) override;
  RHICreateShaderResult create_shader_from_file(const std::string& file_path, const std::string& entry_point, RHIShaderStage stage,
    const std::unordered_map<std::string, std::string>& defines = {}) override;
  RHICreatePipelineResult create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) override;
  RHICreatePipelineResult create_compute_pipeline(const RHIComputePipelineDesc& desc) override;

  RHIResult destroy_buffer(RHIBuffer buffer) override;
  RHIResult destroy_texture(RHITexture texture) override;
  RHIResult destroy_sampler(RHISampler sampler) override;
  RHIResult destroy_shader(RHIShader shader) override;
  RHIResult destroy_pipeline(RHIPipeline pipeline) override;

  RHIResult update_buffer(RHIBuffer buffer, const void* data, uint64_t size, uint64_t offset = 0) override;
  RHIResult update_texture(RHITexture texture, const void* data, uint32_t mip_level = 0, uint32_t array_layer = 0) override;

  RHIResult reload_shader(RHIShader shader, const RHIShaderDesc& new_desc) override;
  RHIResult reload_graphics_pipeline(RHIPipeline pipeline, const RHIGraphicsPipelineDesc& new_desc) override;
  RHIResult reload_compute_pipeline(RHIPipeline pipeline, const RHIComputePipelineDesc& new_desc) override;

  RHICreateBindlessResult create_acceleration_structure(const RHIAccelerationStructureDesc& desc) override;
  RHIResult destroy_acceleration_structure(RHIBindlessHandle as_handle) override;
  uint64_t get_acceleration_structure_device_address(RHIBindlessHandle as_handle) override;

  bool supports_bindless() const override;
  uint64_t get_min_uniform_buffer_offset_alignment() const override;
  uint64_t get_min_storage_buffer_offset_alignment() const override;

  RHIMemoryStats get_memory_statistics() const override;

 private:
  class Impl;
  Impl* _impl = nullptr;
};

struct MTBindlessManager : RHIBindlessManager {
  MTBindlessManager();
  ~MTBindlessManager() override;

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

 private:
  class Impl;
  Impl* _impl = nullptr;
};

struct MTCommandBuffer : RHICommandBuffer {
  MTCommandBuffer();
  ~MTCommandBuffer() override;

  void begin() override;
  void end() override;
  void reset() override;

  void buffer_barrier(RHIBuffer buffer, RHIResourceState old_state, RHIResourceState new_state) override;
  void texture_barrier(RHITexture texture, RHIResourceState old_state, RHIResourceState new_state) override;

  void begin_render_pass(uint32_t color_attachment_count, RHITexture* color_attachments, const float* clear_colors = nullptr, RHITexture depth_attachment = {}) override;
  void end_render_pass() override;

  void set_viewport(const RHIViewport& viewport) override;
  void set_scissor(const RHIRect& scissor) override;
  void set_pipeline(RHIPipeline pipeline) override;

  void push_constants(const void* data, uint32_t size, uint32_t offset = 0) override;

  void draw(const RHIDrawDesc& desc) override;
  void draw_indexed(const RHIIndexedDrawDesc& desc, RHIBuffer index_buffer) override;

  void dispatch(const RHIDispatchDesc& desc) override;

  void build_acceleration_structure(const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset = 0) override;

  void copy_buffer(RHIBuffer src, RHIBuffer dst, uint64_t size, uint64_t src_offset = 0, uint64_t dst_offset = 0) override;
  void copy_buffer_to_texture(RHIBuffer src, RHITexture dst, uint32_t width, uint32_t height, uint32_t mip_level = 0) override;
  void copy_texture_to_buffer(RHITexture src, RHIBuffer dst, uint32_t width, uint32_t height, uint32_t mip_level = 0) override;

  void set_debug_name(const char* name) override;

 private:
  class Impl;
  Impl* _impl = nullptr;
};

}  // namespace etx
