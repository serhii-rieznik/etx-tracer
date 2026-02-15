#pragma once

#ifndef ETX_RHI_INTERNAL
# error "vk_rhi.hxx is an internal etx-rhi implementation header."
#endif

#ifdef ETX_PLATFORM_WINDOWS
# define VK_USE_PLATFORM_WIN32_KHR
# include <vulkan/vulkan.h>
# include <vulkan/vulkan_win32.h>
#else
# error Unsupported platform
#endif

#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include <etx/core/log.hxx>
const char* vk_error_to_string(VkResult);

#define etx_vk_call(expr)                                                                                \
  ([&](const char* _f, unsigned _l) -> VkResult {                                                        \
    auto _r = expr;                                                                                      \
    if (_r != VK_SUCCESS) {                                                                              \
      log::error("[Vulkan] Call %s at [%s:%u] resulted with %s", #expr, _f, _l, vk_error_to_string(_r)); \
    }                                                                                                    \
    return _r;                                                                                           \
  }(__FILE__, __LINE__))

namespace etx {

struct VKComputePipeline;
struct VKGraphicsPipeline;
struct VKDevice;
struct VKBindlessManager;

struct VKBufferData {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  RHIBufferDesc desc = {};
  uint64_t allocated_size = 0;
};

struct VKTextureData {
  VkImage image = VK_NULL_HANDLE;
  VkImageView image_view = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  RHITextureDesc desc = {};
  RHIResourceState current_state = RHIResourceState::Undefined;
  uint64_t allocated_size = 0;
};

struct VKSamplerData {
  VkSampler sampler = VK_NULL_HANDLE;
  RHISamplerDesc desc = {};
};

struct VKPipelineData {
  VkPipeline pipeline = VK_NULL_HANDLE;
  RHIPipeline handle = {};
};

struct VKAccelerationStructureData {
  VkAccelerationStructureKHR acceleration_structure = VK_NULL_HANDLE;
  RHIBindlessHandle buffer = {};
  RHIAccelerationStructureDesc desc = {};
  uint64_t build_scratch_size = 0;
};

template <typename T, typename Key>
struct VKResourcePool {
  uint32_t allocate_index() {
    uint32_t index;
    if (free_indices.empty() == false) {
      index = free_indices.back();
      free_indices.pop_back();
      generations[index] = ((generations[index] + 1) & 0x0FFFFFFF);  // Increment generation on reuse
    } else {
      index = static_cast<uint32_t>(data.size());
      data.emplace_back();
      generations.push_back(0);
    }
    return index;
  }

  void free_index(uint32_t index, std::function<void(T&)> cleanup = nullptr) {
    if (index < data.size()) {
      if (cleanup != nullptr) {
        cleanup(data[index]);
      }
      data[index] = T{};  // Reset to default
      free_indices.push_back(index);
    }
  }

  T& get_data(uint32_t index) {
    return data[index];
  }

  const T& get_data(uint32_t index) const {
    return data[index];
  }

  uint32_t get_generation(uint32_t index) const {
    return (index < generations.size()) ? generations[index] : 0;
  }

  uint32_t get_index(const Key& key) const {
    auto it = handle_to_index_map.find(key);
    return (it != handle_to_index_map.end()) ? it->second : UINT32_MAX;
  }

  T* get_data_ptr(const Key& key) {
    uint32_t index = get_index(key);
    return (index != UINT32_MAX) ? &data[index] : nullptr;
  }

  const T* get_data_ptr(const Key& key) const {
    uint32_t index = get_index(key);
    return (index != UINT32_MAX) ? &data[index] : nullptr;
  }

  void set_handle_to_index(const Key& key, uint32_t index) {
    handle_to_index_map[key] = index;
  }

  void remove_handle(const Key& key) {
    handle_to_index_map.erase(key);
  }

  std::vector<Key> get_all_keys() const {
    std::vector<Key> keys;
    keys.reserve(handle_to_index_map.size());
    for (const auto& pair : handle_to_index_map) {
      keys.push_back(pair.first);
    }
    return keys;
  }

  void clear() {
    data.clear();
    generations.clear();
    free_indices.clear();
    handle_to_index_map.clear();
  }

  size_t size() const {
    return handle_to_index_map.size();
  }

  template <typename Func>
  void for_each(Func func) {
    for (auto& item : data) {
      func(item);
    }
  }

 private:
  std::vector<T> data;
  std::vector<uint32_t> generations;
  std::vector<uint32_t> free_indices;
  std::unordered_map<Key, uint32_t> handle_to_index_map;
};

struct VKContext {
  VKContext(const RHIInitInfo&);
  ~VKContext();
  VKContext(const VKContext&) = delete;
  VKContext& operator=(const VKContext&) = delete;
  VKContext(VKContext&&) noexcept;
  VKContext& operator=(VKContext&&) noexcept = delete;

  VKDevice* get_device();
  VKBindlessManager* get_bindless_manager();

  void initialize_for_headless();

  void create_swapchain(const void* native_window, uint32_t width, uint32_t height);
  void destroy_swapchain();
  void resize_swapchain(uint32_t width, uint32_t height);
  RHITexture get_current_swapchain_texture();
  RHITextureFormat get_swapchain_format() const;
  void present();
  RHIResult wait_idle();

  void begin_frame();
  RHISemaphore get_image_acquired_semaphore();
  RHISemaphore get_render_complete_semaphore();
  uint32_t get_current_frame_index() const;
  uint32_t get_sampler_index(RHISamplerType type) const;

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

  void cmd_build_acceleration_structure(RHICommandBuffer cmd, const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset = 0);

  void cmd_copy_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint64_t size, uint64_t src_offset = 0, uint64_t dst_offset = 0);
  void cmd_copy_buffer_to_texture(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0);
  void cmd_copy_texture_to_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0);

  void cmd_set_debug_name(RHICommandBuffer cmd, const char* name);

  VkDevice get_vk_device() const;
  VkCommandPool get_vk_command_pool(uint32_t index) const;
  VkFence get_current_frame_fence() const;
  VkQueue get_graphics_queue() const;

  bool is_swapchain_texture(RHIBindlessHandle handle) const;
  VkImageView get_swapchain_image_view(RHIBindlessHandle handle) const;
  VkExtent2D get_swapchain_extent() const;

 private:
  struct Impl;
  Impl* _impl = nullptr;
};

struct VKDevice {
  VKDevice(const RHIInitInfo&);
  ~VKDevice();

  VkDevice get_vk_device() const;
  VkPhysicalDevice get_vk_physical_device() const;
  void set_bindless_manager(VKBindlessManager* manager);
  void destroy_all_resources();

  uint64_t get_buffer_device_address(RHIBindlessHandle buffer) const;

  RHICreateResult<RHISemaphore> create_semaphore();
  RHIResult destroy_semaphore(RHISemaphore semaphore);

  RHICreateBindlessResult create_buffer(const RHIBufferDesc& desc);
  RHIResult update_buffer(RHIBindlessHandle buffer, const void* data, uint64_t size, uint64_t offset = 0);
  RHIResult destroy_buffer(RHIBindlessHandle buffer);

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

  const VKPipelineData* get_compute_pipeline_data(RHIPipeline handle) const;
  const VKPipelineData* get_graphics_pipeline_data(RHIPipeline handle) const;
  VKTextureData* get_texture_data(RHIBindlessHandle handle) const;
  const VKAccelerationStructureData* get_acceleration_structure_data(RHIBindlessHandle handle) const;
  PFN_vkCmdBuildAccelerationStructuresKHR get_vkCmdBuildAccelerationStructuresKHR() const;

  VkBuffer get_vk_buffer_from_bindless(RHIBindlessHandle handle) const;
  VkImage get_vk_image_from_bindless(RHIBindlessHandle handle) const;
  VkSemaphore get_vk_semaphore(RHISemaphore handle) const;
  VkPipelineLayout get_bindless_pipeline_layout();
  uint32_t get_max_push_constants_size() const;

  VkInstance get_vk_instance() const;
  VkQueue get_graphics_queue() const;
  VkCommandPool get_vk_command_pool(uint32_t index) const;

  void set_current_frame_index(uint32_t index);
  void reset_staging_buffer_for_frame(uint32_t frame_index);
  void process_deferred_destruction(uint32_t frame_index);

 private:
  class Impl;
  Impl* _impl = nullptr;
};

struct VKBindlessManager {
  VKBindlessManager();
  ~VKBindlessManager();

  void initialize(VkDevice device, VkPhysicalDevice physical_device);
  bool is_initialized() const {
    return _impl != nullptr;
  }

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
  RHIResult register_acceleration_structure_vk(VkAccelerationStructureKHR vk_as, RHIAccelerationStructureType type, RHIBindlessHandle& out_handle);
  RHIResult unregister_acceleration_structure(RHIBindlessHandle handle);

  VkImage get_vk_image(RHIBindlessHandle handle) const;
  VkBuffer get_vk_buffer(RHIBindlessHandle handle) const;
  VkSampler get_vk_sampler(RHIBindlessHandle handle) const;
  VkAccelerationStructureKHR get_vk_acceleration_structure(RHIBindlessHandle handle) const;

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

struct VKCommandBuffer {
  VKCommandBuffer();
  ~VKCommandBuffer();

  VKCommandBuffer(VKCommandBuffer&&) noexcept;
  VKCommandBuffer& operator=(VKCommandBuffer&&) noexcept;

  VKCommandBuffer(const VKCommandBuffer&) = delete;
  VKCommandBuffer& operator=(const VKCommandBuffer&) = delete;

  void initialize(VKContext* ctx, uint32_t pool_index);
  void destroy_resources();
  bool is_initialized() const;
  VkCommandBuffer get_vk_command_buffer() const;
  bool is_recording() const;
  bool is_submitted() const;

  void begin();
  void end();
  void reset();

  void reset_internal_state();
  void set_submitted(bool value);

 private:
  friend class VKContext;

  void buffer_barrier(RHIBindlessHandle buffer, RHIResourceState old_state, RHIResourceState new_state);
  void texture_barrier(RHIBindlessHandle texture, RHIResourceState old_state, RHIResourceState new_state);

  void begin_render_pass(uint32_t color_attachment_count, RHIBindlessHandle* color_attachments, const float* clear_colors = nullptr, RHIBindlessHandle depth_attachment = {},
    const RHIResourceState* color_final_states = nullptr, RHIResourceState depth_final_state = RHIResourceState::Undefined);
  void end_render_pass();

  void set_viewport(const RHIViewport& viewport);
  void set_scissor(const RHIRect& scissor);
  void set_pipeline(RHIPipeline pipeline);

  void push_constants(const void* data, uint32_t size, uint32_t offset = 0);

  void draw(const RHIDrawDesc& desc);
  void draw_indexed(const RHIIndexedDrawDesc& desc, RHIBindlessHandle index_buffer);

  void dispatch(const RHIDispatchDesc& desc);

  void build_acceleration_structure(const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset = 0);

  void copy_buffer(RHIBindlessHandle src, RHIBindlessHandle dst, uint64_t size, uint64_t src_offset = 0, uint64_t dst_offset = 0);
  void copy_buffer_to_texture(RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0);
  void copy_texture_to_buffer(RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level = 0);

  void set_debug_name(const char* name);

  void ensure_texture_layout(RHIBindlessHandle texture, VkImageLayout required_layout);
  void set_scissor_from_viewport(const RHIViewport& viewport);

 private:
  VKContext* context = nullptr;
  VKDevice* device = nullptr;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  bool _in_render_pass = false;
  bool _is_recording = false;
  bool _submitted = false;
  bool _rendering_to_swapchain = false;
  uint32_t _render_pass_depth = 0;

  std::vector<RHITexture> current_color_attachments;
  std::vector<RHIResourceState> current_color_final_states;
  RHITexture current_depth_attachment = {};
  RHIResourceState current_depth_final_state = RHIResourceState::Undefined;

  RHIPipeline current_pipeline = {};
  VkPipelineBindPoint current_bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
  VkPipelineLayout current_pipeline_layout = VK_NULL_HANDLE;
};

}  // namespace etx
