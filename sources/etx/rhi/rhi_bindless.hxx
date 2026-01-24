#pragma once

#include <etx/rhi/rhi_types.hxx>

namespace etx {

inline constexpr uint32_t kDefaultMaxBuffers = 1024u;
inline constexpr uint32_t kDefaultMaxTextures = 4096u;
inline constexpr uint32_t kDefaultMaxSamplers = 256u;
inline constexpr uint32_t kDefaultMaxAccelerationStructures = 512u;

struct RHIBindlessManager {
  virtual ~RHIBindlessManager() = default;

  virtual void set_max_buffers(uint32_t count) = 0;
  virtual void set_max_textures(uint32_t count) = 0;
  virtual void set_max_samplers(uint32_t count) = 0;
  virtual void set_max_acceleration_structures(uint32_t count) = 0;

  virtual RHIResult register_buffer(void* vk_buffer, RHIResourceType type, RHIBindlessHandle& out_handle) = 0;
  virtual RHIResult register_texture(void* vk_image_view, RHIResourceType type, RHIBindlessHandle& out_handle, uint32_t usage_flags, void* vk_image = nullptr) = 0;
  virtual RHIResult register_sampler(void* vk_sampler, RHIResourceType type, RHIBindlessHandle& out_handle) = 0;

  virtual RHIResult unregister_buffer(RHIBindlessHandle handle) = 0;
  virtual RHIResult unregister_texture(RHIBindlessHandle handle) = 0;
  virtual RHIResult unregister_sampler(RHIBindlessHandle handle) = 0;

  virtual RHIResult register_acceleration_structure(const void* data, uint64_t size, RHIBindlessHandle& out_handle) = 0;
  virtual RHIResult unregister_acceleration_structure(RHIBindlessHandle handle) = 0;

  virtual bool is_valid_handle(RHIBindlessHandle handle) const = 0;
  virtual RHIResourceType get_resource_type(RHIBindlessHandle handle) const = 0;

  virtual uint32_t get_max_buffers() const = 0;
  virtual uint32_t get_max_textures() const = 0;
  virtual uint32_t get_max_samplers() const = 0;
  virtual uint32_t get_max_acceleration_structures() const = 0;

  virtual uint32_t get_buffer_count() const = 0;
  virtual uint32_t get_texture_count() const = 0;
  virtual uint32_t get_sampler_count() const = 0;
  virtual uint32_t get_acceleration_structure_count() const = 0;
};

}  // namespace etx
