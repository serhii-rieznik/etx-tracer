#pragma once

#include <etx/core/handle.hxx>
#include <etx/rhi/rhi_types.hxx>
#include <etx/rhi/rhi_bindless.hxx>

#include <vector>
#include <span>

namespace etx {

class RHIContext;
class RHIDevice;

struct RHISubmitInfo {
  RHICommandBuffer command_buffer = {};
  std::vector<RHISemaphore> wait_semaphores;
  std::vector<RHISemaphore> signal_semaphores;
};

struct RHIInitInfo {
  RHIBackend backend = RHIBackend::Vulkan;
  bool enable_validation = false;
};

struct RHIMemoryStats {
  uint64_t cpu_used_bytes = 0;
  uint64_t gpu_allocated_bytes = 0;
  uint64_t gpu_driver_allocated_bytes = 0;
  uint64_t gpu_driver_budget_bytes = 0;
};

class RHIDevice {
 public:
  virtual ~RHIDevice() = default;

  virtual RHICreateResult<RHISemaphore> create_semaphore() = 0;
  virtual RHIResult destroy_semaphore(RHISemaphore semaphore) = 0;

  virtual RHICreateBindlessResult create_buffer(const RHIBufferDesc& desc) = 0;
  virtual RHIResult update_buffer(RHIBindlessHandle buffer, const void* data, uint64_t size, uint64_t offset = 0) = 0;
  virtual RHIResult destroy_buffer(RHIBindlessHandle buffer) = 0;

  virtual RHICreateBindlessResult create_texture(const RHITextureDesc& desc) = 0;
  virtual RHIResult update_texture(RHIBindlessHandle texture, const void* data, uint32_t mip_level = 0, uint32_t array_layer = 0) = 0;
  virtual RHIResult destroy_texture(RHIBindlessHandle texture) = 0;

  virtual RHICreateBindlessResult create_sampler(const RHISamplerDesc& desc) = 0;
  virtual RHIResult destroy_sampler(RHIBindlessHandle sampler) = 0;

  virtual RHICreateBindlessResult create_acceleration_structure(const RHIAccelerationStructureDesc& desc) = 0;
  virtual RHIResult destroy_acceleration_structure(RHIBindlessHandle as_handle) = 0;
  virtual uint64_t get_acceleration_structure_device_address(RHIBindlessHandle as_handle) = 0;

  virtual RHICreateShaderResult create_shader(const RHIShaderDesc& desc) = 0;
  virtual RHICreateShaderResult create_shader_variant(const RHIShaderVariantDesc& desc) = 0;
  virtual RHICreateShaderResult create_shader_from_file(const std::string& file_path, const std::string& entry_point, RHIShaderStage stage,
    const std::unordered_map<std::string, std::string>& defines = {}) = 0;
  virtual RHIResult reload_shader(RHIShader shader, const RHIShaderDesc& new_desc) = 0;
  virtual RHIResult destroy_shader(RHIShader shader) = 0;

  virtual RHICreatePipelineResult create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) = 0;
  virtual RHICreatePipelineResult create_compute_pipeline(const RHIComputePipelineDesc& desc) = 0;
  virtual RHIResult reload_graphics_pipeline(RHIPipeline pipeline, const RHIGraphicsPipelineDesc& new_desc) = 0;
  virtual RHIResult reload_compute_pipeline(RHIPipeline pipeline, const RHIComputePipelineDesc& new_desc) = 0;
  virtual RHIResult destroy_pipeline(RHIPipeline pipeline) = 0;

  virtual bool supports_bindless() const = 0;
  virtual uint64_t get_min_uniform_buffer_offset_alignment() const = 0;
  virtual uint64_t get_min_storage_buffer_offset_alignment() const = 0;

  virtual RHIMemoryStats get_memory_statistics() const = 0;
};

struct RHIContext {
  static RHIContext* create(const RHIInitInfo& info);
  static void release(RHIContext*);

  virtual ~RHIContext() = default;

  virtual RHIDevice* get_device() = 0;
  virtual RHIBindlessManager* get_bindless_manager() = 0;

  virtual void create_swapchain(const void* native_window, uint32_t width, uint32_t height) = 0;
  virtual void destroy_swapchain() = 0;
  virtual void resize_swapchain(uint32_t width, uint32_t height) = 0;
  virtual RHITexture get_current_swapchain_texture() = 0;
  virtual RHITextureFormat get_swapchain_format() const = 0;

  virtual void begin_frame() = 0;
  virtual void present() = 0;

  virtual RHISemaphore get_image_acquired_semaphore() = 0;
  virtual RHISemaphore get_render_complete_semaphore() = 0;

  virtual uint32_t get_current_frame_index() const = 0;

  virtual uint32_t get_sampler_index(RHISamplerType type) const = 0;

  virtual RHICommandBuffer get_command_buffer() {
    return {};
  }
  virtual void destroy_command_buffer(RHICommandBuffer cmd) {
  }
  virtual void submit_command_buffer(const RHISubmitInfo& info) {
  }

  virtual void program_command_buffer(RHICommandBuffer cmd, std::function<void(void)> func) {
  }

  virtual void command_buffer_begin(RHICommandBuffer cmd) {
  }
  virtual void command_buffer_end(RHICommandBuffer cmd) {
  }
  virtual void command_buffer_reset(RHICommandBuffer cmd) {
  }

  virtual void cmd_buffer_barrier(RHICommandBuffer cmd, RHIBindlessHandle buffer, RHIResourceState old_state, RHIResourceState new_state) {
  }
  virtual void cmd_texture_barrier(RHICommandBuffer cmd, RHIBindlessHandle texture, RHIResourceState old_state, RHIResourceState new_state) {
  }

  virtual void cmd_begin_render_pass(RHICommandBuffer cmd, uint32_t color_attachment_count, RHIBindlessHandle* color_attachments, const float* clear_colors = nullptr,
    RHIBindlessHandle depth_attachment = {}, const RHIResourceState* color_final_states = nullptr, RHIResourceState depth_final_state = RHIResourceState::Undefined) {
  }
  virtual void cmd_end_render_pass(RHICommandBuffer cmd) {
  }

  virtual void cmd_set_viewport(RHICommandBuffer cmd, const RHIViewport& viewport) {
  }
  virtual void cmd_set_scissor(RHICommandBuffer cmd, const RHIRect& scissor) {
  }
  virtual void cmd_set_pipeline(RHICommandBuffer cmd, RHIPipeline pipeline) {
  }

  virtual void cmd_push_constants(RHICommandBuffer cmd, const void* data, uint32_t size, uint32_t offset = 0) {
  }

  virtual void cmd_draw(RHICommandBuffer cmd, const RHIDrawDesc& desc) {
  }
  virtual void cmd_draw_indexed(RHICommandBuffer cmd, const RHIIndexedDrawDesc& desc, RHIBindlessHandle index_buffer) {
  }

  virtual void cmd_dispatch(RHICommandBuffer cmd, const RHIDispatchDesc& desc) {
  }

  virtual void cmd_build_acceleration_structure(RHICommandBuffer cmd, const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer,
    uint64_t scratch_offset = 0) {
  }

  virtual void cmd_copy_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint64_t size, uint64_t src_offset = 0, uint64_t dst_offset = 0) {
  }
  virtual void cmd_copy_buffer_to_texture(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0) {
  }
  virtual void cmd_copy_texture_to_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0) {
  }

  virtual void cmd_set_debug_name(RHICommandBuffer cmd, const char* name) {
  }
};

}  // namespace etx
