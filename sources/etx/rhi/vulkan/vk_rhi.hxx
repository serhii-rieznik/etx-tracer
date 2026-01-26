#pragma once

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
#include <functional>

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

static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2u;

class VKComputePipeline;
class VKGraphicsPipeline;

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
  VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  uint64_t allocated_size = 0;
};

struct VKSamplerData {
  VkSampler sampler = VK_NULL_HANDLE;
  RHISamplerDesc desc = {};
};

struct VKPipelineData {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  RHIPipeline handle = {};
};

template <typename T, typename Key>
struct VKResourcePool {
  uint32_t allocate_index() {
    uint32_t index;
    if (!free_indices.empty()) {
      index = free_indices.back();
      free_indices.pop_back();
    } else {
      index = static_cast<uint32_t>(data.size());
      data.emplace_back();
    }
    return index;
  }

  void free_index(uint32_t index, std::function<void(T&)> cleanup = nullptr) {
    if (index < data.size()) {
      if (cleanup) {
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
    free_indices.clear();
    handle_to_index_map.clear();
  }

  size_t size() const {
    return handle_to_index_map.size();
  }

 private:
  std::vector<T> data;
  std::vector<uint32_t> free_indices;
  std::unordered_map<Key, uint32_t> handle_to_index_map;
};

struct VKContext : RHIContext {
  VKContext(const RHIInitInfo&);
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

  void begin_frame() override;
  uint32_t get_current_frame_index() const override;
  uint32_t get_sampler_index(RHISamplerType type) const override;

  RHICommandBuffer* get_command_buffer() override;
  void submit_command_buffer(RHICommandBuffer* command_buffer) override;

  VkDevice get_vk_device() const;
  VkCommandPool get_vk_command_pool(uint32_t index) const;
  VkFence get_current_frame_fence() const;
  VkQueue get_graphics_queue() const;

 private:
  friend struct VKCommandBuffer;
  struct Impl;
  Impl* _impl = nullptr;
};

struct VKDevice : RHIDevice {
  VKDevice(const RHIInitInfo&);
  ~VKDevice() override;

  VkDevice get_vk_device() const;
  VkPhysicalDevice get_vk_physical_device() const;
  void set_bindless_manager(RHIBindlessManager* manager);
  void destroy_all_resources();

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

  RHIMemoryStats get_memory_statistics() const override;

  const VKPipelineData* get_compute_pipeline_data(RHIPipeline handle) const;
  const VKPipelineData* get_graphics_pipeline_data(RHIPipeline handle) const;
  VkBuffer get_vk_buffer_from_bindless(RHIBindlessHandle handle) const;
  VkImage get_vk_image_from_bindless(RHIBindlessHandle handle) const;

  void set_shader_compiler(ShaderCompiler* compiler);
  ShaderCompiler* get_shader_compiler() const;

 private:
  friend class VKContext;
  friend class VKCommandBuffer;
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

struct VKBindlessManager::Impl {
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

  void initialize(VKContext* ctx, uint32_t pool_index);
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

  void ensure_texture_layout(RHIBindlessHandle texture, VkImageLayout required_layout);
  void set_scissor_from_viewport(const RHIViewport& viewport);

 private:
  class Impl;
  Impl* _impl = nullptr;
};

}  // namespace etx
