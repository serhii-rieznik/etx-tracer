#pragma once

#include <etx/rhi/rhi_types.hxx>

namespace etx {

inline constexpr uint32_t kDefaultMaxBuffers = 1024u;
inline constexpr uint32_t kDefaultMaxTextures = 4096u;
inline constexpr uint32_t kDefaultMaxSamplers = 256u;
inline constexpr uint32_t kDefaultMaxAccelerationStructures = 512u;

struct RHIBindlessManager {
  RHIBindlessManager() = default;
  explicit RHIBindlessManager(void* impl)
    : _impl(impl) {
  }

  bool valid() const {
    return _impl != nullptr;
  }

  void set_max_buffers(uint32_t count);
  void set_max_textures(uint32_t count);
  void set_max_samplers(uint32_t count);
  void set_max_acceleration_structures(uint32_t count);

  RHIResult register_buffer(void* buffer, RHIResourceType type, RHIBindlessHandle& out_handle);
  RHIResult register_texture(void* image_view, RHIResourceType type, RHIBindlessHandle& out_handle, uint32_t usage_flags, void* image = nullptr);
  RHIResult register_sampler(void* sampler, RHIResourceType type, RHIBindlessHandle& out_handle);

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

 private:
  void* _impl = nullptr;

  friend struct RHIContext;
};

}  // namespace etx
