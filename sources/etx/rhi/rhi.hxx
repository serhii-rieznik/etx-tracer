#pragma once

#include <etx/core/handle.hxx>
#include <etx/rhi/rhi_types.hxx>
#include <etx/rhi/rhi_bindless.hxx>

#include <vector>
#include <span>

namespace etx {

class RHIContext;
class RHIDevice;
class RHICommandBuffer;

struct RHIInitInfo {
  RHIBackend backend = RHIBackend::Vulkan;
  bool enable_validation = false;
  bool enable_debug_names = false;
  uint32_t max_frames_in_flight = 2;
};

RHIContext* create_rhi_context(const RHIInitInfo& info);
void destroy_rhi_context(RHIContext* context);

struct RHIContext {
  virtual ~RHIContext() = default;

  virtual RHIDevice* get_device() = 0;
  virtual RHIBindlessManager* get_bindless_manager() = 0;

  virtual void create_swapchain(const void* native_window, uint32_t width, uint32_t height) = 0;
  virtual void destroy_swapchain() = 0;
  virtual void resize_swapchain(uint32_t width, uint32_t height) = 0;
  virtual RHITexture get_current_swapchain_texture() = 0;
  virtual RHITextureFormat get_swapchain_format() const = 0;
  virtual void present() = 0;
  virtual void present_with_frame_index(uint32_t frame_index) = 0;

  virtual void begin_frame() = 0;
  virtual void end_frame() = 0;
  virtual uint32_t get_current_frame_index() const = 0;

  virtual uint32_t get_sampler_index(RHISamplerType type) const = 0;

  virtual RHICommandBuffer* get_command_buffer() = 0;
  virtual void submit_command_buffer(RHICommandBuffer* command_buffer) = 0;
};

struct RHIDevice {
  virtual ~RHIDevice() = default;

  virtual RHICreateBindlessResult create_buffer(const RHIBufferDesc& desc) = 0;
  virtual RHICreateBindlessResult create_texture(const RHITextureDesc& desc) = 0;
  virtual RHICreateBindlessResult create_sampler(const RHISamplerDesc& desc) = 0;
  virtual RHICreateShaderResult create_shader(const RHIShaderDesc& desc) = 0;
  virtual RHICreateShaderResult create_shader_variant(const RHIShaderVariantDesc& desc) = 0;
  virtual RHICreateShaderResult create_shader_from_file(const std::string& file_path, const std::string& entry_point, RHIShaderStage stage,
    const std::unordered_map<std::string, std::string>& defines = {}) = 0;
  virtual RHICreatePipelineResult create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) = 0;
  virtual RHICreatePipelineResult create_compute_pipeline(const RHIComputePipelineDesc& desc) = 0;

  virtual RHIResult destroy_buffer(RHIBindlessHandle buffer_handle) = 0;
  virtual RHIResult destroy_texture(RHIBindlessHandle texture_handle) = 0;
  virtual RHIResult destroy_sampler(RHIBindlessHandle sampler_handle) = 0;
  virtual RHIResult destroy_shader(RHIShader shader) = 0;
  virtual RHIResult destroy_pipeline(RHIPipeline pipeline) = 0;

  virtual RHIResult update_buffer(RHIBindlessHandle buffer, const void* data, uint64_t size, uint64_t offset = 0) = 0;
  virtual RHIResult update_texture(RHIBindlessHandle texture, const void* data, uint32_t mip_level = 0, uint32_t array_layer = 0) = 0;

  virtual RHIResult reload_shader(RHIShader shader, const RHIShaderDesc& new_desc) = 0;
  virtual RHIResult reload_graphics_pipeline(RHIPipeline pipeline, const RHIGraphicsPipelineDesc& new_desc) = 0;
  virtual RHIResult reload_compute_pipeline(RHIPipeline pipeline, const RHIComputePipelineDesc& new_desc) = 0;

  virtual bool supports_bindless() const = 0;
  virtual uint64_t get_min_uniform_buffer_offset_alignment() const = 0;
  virtual uint64_t get_min_storage_buffer_offset_alignment() const = 0;
};

struct RHICommandBuffer {
  virtual ~RHICommandBuffer() = default;

  virtual void begin() = 0;
  virtual void end() = 0;
  virtual void reset() = 0;

  virtual void buffer_barrier(RHIBindlessHandle buffer, RHIResourceState old_state, RHIResourceState new_state) = 0;
  virtual void texture_barrier(RHIBindlessHandle texture, RHIResourceState old_state, RHIResourceState new_state) = 0;

  virtual void set_buffer_state(RHIBindlessHandle buffer, RHIResourceState state) = 0;
  virtual void set_texture_state(RHIBindlessHandle texture, RHIResourceState state) = 0;
  virtual RHIResourceState get_buffer_state(RHIBindlessHandle buffer) const = 0;
  virtual RHIResourceState get_texture_state(RHIBindlessHandle texture) const = 0;

  virtual void begin_render_pass(uint32_t color_attachment_count, RHIBindlessHandle* color_attachments, const float* clear_colors = nullptr,
    RHIBindlessHandle depth_attachment = {}) = 0;
  virtual void end_render_pass() = 0;

  virtual void set_viewport(const RHIViewport& viewport) = 0;
  virtual void set_scissor(const RHIRect& scissor) = 0;
  virtual void set_pipeline(RHIPipeline pipeline) = 0;

  virtual void push_constants(const void* data, uint32_t size, uint32_t offset = 0) = 0;

  virtual void draw(const RHIDrawDesc& desc) = 0;
  virtual void draw_indexed(const RHIIndexedDrawDesc& desc, RHIBindlessHandle index_buffer) = 0;

  virtual void dispatch(const RHIDispatchDesc& desc) = 0;

  virtual void copy_buffer(RHIBindlessHandle src, RHIBindlessHandle dst, uint64_t size, uint64_t src_offset = 0, uint64_t dst_offset = 0) = 0;
  virtual void copy_buffer_to_texture(RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0) = 0;
  virtual void copy_texture_to_buffer(RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0) = 0;

  virtual void set_debug_name(const char* name) = 0;
};

}  // namespace etx
