#pragma once

#include <etx/core/handle.hxx>
#include <etx/rhi/rhi_types.hxx>
#include <etx/rhi/rhi_bindless.hxx>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace etx {

struct RHIContext;
struct RHIDevice;

struct RHISubmitInfo {
  RHICommandBuffer command_buffer = {};
  std::vector<RHISemaphore> wait_semaphores;
  std::vector<RHISemaphore> signal_semaphores;
};

struct RHIInitInfo {
  RHIBackend backend = RHIBackend::Vulkan;
  bool enable_validation = false;
  bool headless = false;
};

struct RHIMemoryStats {
  uint64_t cpu_used_bytes = 0;
  uint64_t gpu_allocated_bytes = 0;
  uint64_t gpu_driver_allocated_bytes = 0;
  uint64_t gpu_driver_budget_bytes = 0;
};

struct RHICapabilities {
  bool supports_swapchain = false;
  bool supports_bindless = false;
  bool supports_timestamps = false;
  bool supports_ray_tracing = false;
};

struct RHIChunkedBufferRange {
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct RHIChunkedBufferUploadData {
  std::vector<uint8_t> metadata = {};
  std::vector<uint8_t> payload_data = {};
  std::vector<RHIChunkedBufferRange> payload_chunk_ranges = {};
  uint32_t chunk_indices_offset = ~0u;
  bool success = true;
};

struct RHIChunkedBufferState {
  std::vector<RHIBindlessHandle> chunk_buffers = {};
  std::vector<uint64_t> chunk_buffer_sizes = {};
  RHIBindlessHandle metadata_buffer = {};
  uint64_t metadata_buffer_size = 0;
  uint32_t metadata_descriptor_index = ~0u;
};

struct RHIDevice {
  RHIDevice() = default;
  explicit RHIDevice(void* impl, RHIBackend backend = RHIBackend::Vulkan)
    : _impl(impl)
    , _backend(backend) {
  }

  bool valid() const {
    return _impl != nullptr;
  }

  RHIBackend backend() const {
    return _backend;
  }

  RHICreateResult<RHISemaphore> create_semaphore();
  RHIResult destroy_semaphore(RHISemaphore semaphore);

  RHICreateBindlessResult create_buffer(const RHIBufferDesc& desc);
  RHIResult update_buffer(RHIBindlessHandle buffer, const void* data, uint64_t size, uint64_t offset = 0);
  RHIResult read_buffer(RHIBindlessHandle buffer, void* data, uint64_t size, uint64_t offset = 0);
  RHIResult destroy_buffer(RHIBindlessHandle buffer);
  bool upload_or_update_chunked_buffer(const RHIChunkedBufferUploadData& data, RHIBufferUsage usage, RHIChunkedBufferState& state, const char* buffer_name = nullptr);
  void destroy_chunked_buffer(RHIChunkedBufferState& state);

  RHICreateBindlessResult create_texture(const RHITextureDesc& desc);
  RHIResult update_texture(RHIBindlessHandle texture, const void* data, uint32_t mip_level = 0, uint32_t array_layer = 0);
  RHIResult destroy_texture(RHIBindlessHandle texture);

  RHICreateBindlessResult create_sampler(const RHISamplerDesc& desc);
  RHIResult destroy_sampler(RHIBindlessHandle sampler);

  RHICreateBindlessResult create_acceleration_structure(const RHIAccelerationStructureDesc& desc);
  RHIResult destroy_acceleration_structure(RHIBindlessHandle as_handle);
  uint64_t get_acceleration_structure_device_address(RHIBindlessHandle as_handle);
  uint64_t get_acceleration_structure_build_scratch_size(RHIBindlessHandle as_handle);

  RHICreatePipelineResult create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc);
  RHICreatePipelineResult create_compute_pipeline(const RHIComputePipelineDesc& desc);
  RHIResult reload_graphics_pipeline(RHIPipeline pipeline, const RHIGraphicsPipelineDesc& new_desc);
  RHIResult reload_compute_pipeline(RHIPipeline pipeline, const RHIComputePipelineDesc& new_desc);
  RHIResult destroy_pipeline(RHIPipeline pipeline);

  RHIMemoryStats get_memory_statistics() const;

  RHICreatePipelineResult create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc, const RHIShaderBinary& vertex_shader, const RHIShaderBinary& fragment_shader) {
    RHIGraphicsPipelineDesc final_desc = desc;
    final_desc.vertex_shader.spirv_data = vertex_shader.spirv_data;
    final_desc.vertex_shader.spirv_size = vertex_shader.spirv_size;
    final_desc.vertex_shader.stage = vertex_shader.stage;
    final_desc.vertex_shader.backend = vertex_shader.backend;
    final_desc.vertex_shader.format = vertex_shader.format;
    final_desc.vertex_shader.entry_point = vertex_shader.entry_point;
    final_desc.vertex_shader.local_size_x = vertex_shader.local_size_x;
    final_desc.vertex_shader.local_size_y = vertex_shader.local_size_y;
    final_desc.vertex_shader.local_size_z = vertex_shader.local_size_z;
    final_desc.fragment_shader.spirv_data = fragment_shader.spirv_data;
    final_desc.fragment_shader.spirv_size = fragment_shader.spirv_size;
    final_desc.fragment_shader.stage = fragment_shader.stage;
    final_desc.fragment_shader.backend = fragment_shader.backend;
    final_desc.fragment_shader.format = fragment_shader.format;
    final_desc.fragment_shader.entry_point = fragment_shader.entry_point;
    final_desc.fragment_shader.local_size_x = fragment_shader.local_size_x;
    final_desc.fragment_shader.local_size_y = fragment_shader.local_size_y;
    final_desc.fragment_shader.local_size_z = fragment_shader.local_size_z;
    return create_graphics_pipeline(final_desc);
  }

  RHIComputePipelineDesc make_compute_pipeline_desc(const RHIShaderBinary& compute_shader) {
    RHIComputePipelineDesc desc = {
      .compute_shader = {.spirv_data = compute_shader.spirv_data,
        .spirv_size = compute_shader.spirv_size,
        .stage = compute_shader.stage,
        .backend = compute_shader.backend,
        .format = compute_shader.format,
        .entry_point = compute_shader.entry_point,
        .local_size_x = compute_shader.local_size_x,
        .local_size_y = compute_shader.local_size_y,
        .local_size_z = compute_shader.local_size_z},
    };
    return desc;
  }

 private:
  void* _impl = nullptr;
  RHIBackend _backend = RHIBackend::Vulkan;
  friend struct RHIContext;
};

struct RHIContext {
  static RHIContext create(const RHIInitInfo& info);

  RHIContext() = default;
  ~RHIContext();

  RHIContext(const RHIContext&) = delete;
  RHIContext& operator=(const RHIContext&) = delete;

  RHIContext(RHIContext&& other) noexcept;
  RHIContext& operator=(RHIContext&& other) noexcept;

  bool valid() const {
    return _impl != nullptr;
  }

  RHIBackend backend() const {
    return _backend;
  }

  RHIDevice& device() {
    return _device;
  }
  const RHIDevice& device() const {
    return _device;
  }

  RHIBindlessManager& bindless() {
    return _bindless;
  }
  const RHIBindlessManager& bindless() const {
    return _bindless;
  }

  void initialize_headless();
  bool has_swapchain() const;

  void create_swapchain(const void* native_window, uint32_t width, uint32_t height);
  void destroy_swapchain();
  void resize_swapchain(uint32_t width, uint32_t height);
  RHITexture get_current_swapchain_texture();
  RHITextureFormat get_swapchain_format() const;
  RHIExtent2D get_swapchain_extent() const;

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
  void submit_frame_command_buffer(RHICommandBuffer cmd);

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
  void cmd_write_timestamp(RHICommandBuffer cmd, uint32_t query_index, RHITimestampStage stage = RHITimestampStage::AllCommands);

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
  static constexpr size_t kBackendStorageSize = 64;
  static constexpr size_t kBackendStorageAlignment = alignof(std::max_align_t);

  void initialize_backend(RHIBackend backend, void* context_impl, void* device_impl, void* bindless_impl) {
    _backend = backend;
    _impl = context_impl;
    _device = RHIDevice(device_impl, backend);
    _bindless = RHIBindlessManager(bindless_impl, backend);
  }

  void destroy_backend();
  void move_from(RHIContext&& other);

  alignas(kBackendStorageAlignment) unsigned char _backend_storage[kBackendStorageSize] = {};
  RHIDevice _device = {};
  RHIBindlessManager _bindless = {};
  RHIBackend _backend = RHIBackend::Vulkan;

  friend void create_vulkan_context(RHIContext& context, const RHIInitInfo& info);
  friend void create_metal_context(RHIContext& context, const RHIInitInfo& info);

 private:
  void* _impl = nullptr;
};

}  // namespace etx
