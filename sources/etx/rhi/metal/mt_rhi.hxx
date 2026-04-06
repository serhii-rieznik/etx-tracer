#pragma once

#ifndef ETX_RHI_INTERNAL
# error "mt_rhi.hxx is an internal etx-rhi implementation header."
#endif

#include <etx/rhi/rhi.hxx>

namespace etx {

struct MTDevice;
struct MTBindlessManager;

struct MTContext {
  MTContext();
  ~MTContext();
  MTContext(const MTContext&) = delete;
  MTContext& operator=(const MTContext&) = delete;
  MTContext(MTContext&&) noexcept;
  MTContext& operator=(MTContext&&) noexcept = delete;

  MTDevice* get_device();
  MTBindlessManager* get_bindless_manager();

  void initialize_for_headless();
  bool has_swapchain() const;

  void create_swapchain(const void* native_window, uint32_t width, uint32_t height);
  void destroy_swapchain();
  void resize_swapchain(uint32_t width, uint32_t height);
  RHITexture get_current_swapchain_texture();
  RHITextureFormat get_swapchain_format() const;
  RHIExtent2D get_swapchain_extent_rhi() const;

  void begin_frame();
  void end_frame();
  void present();
  RHIResult wait_idle();

  RHISemaphore get_image_acquired_semaphore();
  RHISemaphore get_render_complete_semaphore();

  uint32_t get_current_frame_index() const;
  uint32_t get_sampler_index(RHISamplerType type) const;
  RHICapabilities capabilities() const;

  RHICommandBuffer get_command_buffer();
  void destroy_command_buffer(RHICommandBuffer cmd);
  void submit_command_buffer(const RHISubmitInfo& info);

  void program_command_buffer(RHICommandBuffer cmd, std::function<void(void)> func);

  void command_buffer_begin(RHICommandBuffer cmd);
  void command_buffer_end(RHICommandBuffer cmd);
  void command_buffer_reset(RHICommandBuffer cmd);

  void cmd_buffer_barrier(RHICommandBuffer cmd, RHIBindlessHandle buffer, RHIResourceState old_state, RHIResourceState new_state);
  void cmd_texture_barrier(RHICommandBuffer cmd, RHIBindlessHandle texture, RHIResourceState old_state, RHIResourceState new_state);

  void cmd_begin_render_pass(RHICommandBuffer cmd, uint32_t color_attachment_count, RHIBindlessHandle* color_attachments, const float* clear_colors = nullptr,
    RHIBindlessHandle depth_attachment = {}, const RHIResourceState* color_final_states = nullptr, RHIResourceState depth_final_state = RHIResourceState::Undefined);
  void cmd_end_render_pass(RHICommandBuffer cmd);

  void cmd_set_viewport(RHICommandBuffer cmd, const RHIViewport& viewport);
  void cmd_set_scissor(RHICommandBuffer cmd, const RHIRect& scissor);
  void cmd_set_pipeline(RHICommandBuffer cmd, RHIPipeline pipeline);

  void cmd_push_constants(RHICommandBuffer cmd, const void* data, uint32_t size, uint32_t offset = 0);

  void cmd_draw(RHICommandBuffer cmd, const RHIDrawDesc& desc);
  void cmd_draw_indexed(RHICommandBuffer cmd, const RHIIndexedDrawDesc& desc, RHIBindlessHandle index_buffer);

  void cmd_dispatch(RHICommandBuffer cmd, const RHIDispatchDesc& desc);
  void cmd_reset_timestamps(RHICommandBuffer cmd, uint32_t first_query, uint32_t query_count);
  void cmd_write_timestamp(RHICommandBuffer cmd, uint32_t query_index, RHITimestampStage stage);

  void cmd_build_acceleration_structure(RHICommandBuffer cmd, const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset = 0);

  void cmd_copy_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint64_t size, uint64_t src_offset = 0, uint64_t dst_offset = 0);
  void cmd_copy_buffer_to_texture(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0);
  void cmd_copy_texture_to_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0);
  void cmd_resolve_texture(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height);
  void cmd_generate_mipmaps(RHICommandBuffer cmd, RHIBindlessHandle texture);

  void cmd_set_debug_name(RHICommandBuffer cmd, const char* name);
  bool supports_timestamps() const;
  double timestamp_period_ns() const;
  RHIResult read_timestamps(RHICommandBuffer cmd, uint32_t first_query, uint32_t query_count, uint64_t* out_values);

 private:
  class Impl;
  Impl* _impl = nullptr;

  friend struct MTContext;
  friend struct MTCommandBuffer;
};

struct MTDevice {
  MTDevice();
  ~MTDevice();

  RHICreateResult<RHISemaphore> create_semaphore();
  RHIResult destroy_semaphore(RHISemaphore semaphore);

  RHICreateBindlessResult create_buffer(const RHIBufferDesc& desc);
  RHICreateBindlessResult create_texture(const RHITextureDesc& desc);
  RHICreateBindlessResult create_sampler(const RHISamplerDesc& desc);
  RHICreatePipelineResult create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc);
  RHICreatePipelineResult create_compute_pipeline(const RHIComputePipelineDesc& desc);

  RHIResult destroy_buffer(RHIBuffer buffer);
  RHIResult destroy_texture(RHITexture texture);
  RHIResult destroy_sampler(RHISampler sampler);
  RHIResult destroy_pipeline(RHIPipeline pipeline);

  RHIResult update_buffer(RHIBuffer buffer, const void* data, uint64_t size, uint64_t offset = 0);
  RHIResult read_buffer(RHIBuffer buffer, void* data, uint64_t size, uint64_t offset = 0);
  RHIResult update_texture(RHITexture texture, const void* data, uint32_t mip_level = 0, uint32_t array_layer = 0);

  RHIResult reload_graphics_pipeline(RHIPipeline pipeline, const RHIGraphicsPipelineDesc& new_desc);
  RHIResult reload_compute_pipeline(RHIPipeline pipeline, const RHIComputePipelineDesc& new_desc);

  RHICreateBindlessResult create_acceleration_structure(const RHIAccelerationStructureDesc& desc);
  RHIResult destroy_acceleration_structure(RHIBindlessHandle as_handle);
  uint64_t get_acceleration_structure_device_address(RHIBindlessHandle as_handle);
  uint64_t get_acceleration_structure_build_scratch_size(RHIBindlessHandle as_handle);

  RHIMemoryStats get_memory_statistics() const;

 public:
  class Impl;
  Impl* _impl = nullptr;

  friend struct MTContext;
  friend struct MTDevice;
  friend struct MTCommandBuffer;
};

struct MTBindlessManager {
  MTBindlessManager();
  ~MTBindlessManager();

  void set_max_buffers(uint32_t count);
  void set_max_textures(uint32_t count);
  void set_max_samplers(uint32_t count);
  void set_max_acceleration_structures(uint32_t count);

  RHIResult register_buffer(void* vk_buffer, RHIResourceType type, RHIBindlessHandle& out_handle);
  RHIResult register_texture(void* vk_image_view, RHIResourceType type, RHIBindlessHandle& out_handle, uint32_t usage_flags, void* vk_image = nullptr);
  RHIResult register_sampler(void* vk_sampler, RHIResourceType type, RHIBindlessHandle& out_handle);

  RHIResult unregister_buffer(RHIBindlessHandle handle);
  RHIResult unregister_texture(RHIBindlessHandle handle);
  RHIResult unregister_sampler(RHIBindlessHandle handle);

  RHIResult register_acceleration_structure(const void* data, uint64_t size, RHIBindlessHandle& out_handle);
  RHIResult unregister_acceleration_structure(RHIBindlessHandle handle);

  bool is_valid_handle(RHIBindlessHandle handle) const;
  RHIResourceType get_resource_type(RHIBindlessHandle handle) const;

  uint32_t get_max_buffers() const;
  uint32_t get_max_textures() const;
  uint32_t get_max_samplers() const;
  uint32_t get_max_acceleration_structures() const;

  uint32_t get_buffer_count() const;
  uint32_t get_texture_count() const;
  uint32_t get_sampler_count() const;
  uint32_t get_acceleration_structure_count() const;

 public:
  class Impl;
  Impl* _impl = nullptr;
};

struct MTCommandBuffer {
  MTCommandBuffer();
  ~MTCommandBuffer();

  void begin();
  void end();
  void reset();
  void detach_submitted();

  void buffer_barrier(RHIBuffer buffer, RHIResourceState old_state, RHIResourceState new_state);
  void texture_barrier(RHITexture texture, RHIResourceState old_state, RHIResourceState new_state);

  void begin_render_pass(uint32_t color_attachment_count, RHITexture* color_attachments, const float* clear_colors = nullptr, RHITexture depth_attachment = {},
    const RHIResourceState* color_final_states = nullptr, RHIResourceState depth_final_state = RHIResourceState::Undefined);
  void end_render_pass();

  void set_viewport(const RHIViewport& viewport);
  void set_scissor(const RHIRect& scissor);
  void set_pipeline(RHIPipeline pipeline);

  void push_constants(const void* data, uint32_t size, uint32_t offset = 0);

  void draw(const RHIDrawDesc& desc);
  void draw_indexed(const RHIIndexedDrawDesc& desc, RHIBuffer index_buffer);

  void dispatch(const RHIDispatchDesc& desc);

  void build_acceleration_structure(const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset = 0);

  void copy_buffer(RHIBuffer src, RHIBuffer dst, uint64_t size, uint64_t src_offset = 0, uint64_t dst_offset = 0);
  void copy_buffer_to_texture(RHIBuffer src, RHITexture dst, uint32_t width, uint32_t height, uint32_t mip_level = 0);
  void copy_texture_to_buffer(RHITexture src, RHIBuffer dst, uint32_t width, uint32_t height, uint32_t mip_level = 0);

  void set_debug_name(const char* name);

 public:
  class Impl;
  Impl* _impl = nullptr;
};

}  // namespace etx
