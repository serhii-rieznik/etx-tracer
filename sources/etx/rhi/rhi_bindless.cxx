#include <etx/rhi/rhi_bindless.hxx>

#if defined(ETX_PLATFORM_WINDOWS)
# include <etx/rhi/vulkan/vk_rhi.hxx>
namespace etx {
using BackendBindlessManager = VKBindlessManager;
}  // namespace etx
#elif defined(ETX_PLATFORM_APPLE)
# include <etx/rhi/metal/mt_rhi.hxx>
namespace etx {
using BackendBindlessManager = MTBindlessManager;
}  // namespace etx
#else
# error Unsupported platform for RHI bindless dispatch
#endif

namespace etx {

static BackendBindlessManager* backend_bindless(void* impl) {
  return static_cast<BackendBindlessManager*>(impl);
}

static const BackendBindlessManager* backend_bindless(const void* impl) {
  return static_cast<const BackendBindlessManager*>(impl);
}

void RHIBindlessManager::set_max_buffers(uint32_t count) {
  backend_bindless(_impl)->set_max_buffers(count);
}

void RHIBindlessManager::set_max_textures(uint32_t count) {
  backend_bindless(_impl)->set_max_textures(count);
}

void RHIBindlessManager::set_max_samplers(uint32_t count) {
  backend_bindless(_impl)->set_max_samplers(count);
}

void RHIBindlessManager::set_max_acceleration_structures(uint32_t count) {
  backend_bindless(_impl)->set_max_acceleration_structures(count);
}

RHIResult RHIBindlessManager::register_buffer(void* buffer, RHIResourceType type, RHIBindlessHandle& out_handle) {
  return backend_bindless(_impl)->register_buffer(buffer, type, out_handle);
}

RHIResult RHIBindlessManager::register_texture(void* image_view, RHIResourceType type, RHIBindlessHandle& out_handle, uint32_t usage_flags, void* image) {
  return backend_bindless(_impl)->register_texture(image_view, type, out_handle, usage_flags, image);
}

RHIResult RHIBindlessManager::register_sampler(void* sampler, RHIResourceType type, RHIBindlessHandle& out_handle) {
  return backend_bindless(_impl)->register_sampler(sampler, type, out_handle);
}

RHIResult RHIBindlessManager::unregister_buffer(RHIBindlessHandle handle) {
  return backend_bindless(_impl)->unregister_buffer(handle);
}

RHIResult RHIBindlessManager::unregister_texture(RHIBindlessHandle handle) {
  return backend_bindless(_impl)->unregister_texture(handle);
}

RHIResult RHIBindlessManager::unregister_sampler(RHIBindlessHandle handle) {
  return backend_bindless(_impl)->unregister_sampler(handle);
}

RHIResult RHIBindlessManager::register_acceleration_structure(const void* data, uint64_t size, RHIBindlessHandle& out_handle) {
  return backend_bindless(_impl)->register_acceleration_structure(data, size, out_handle);
}

RHIResult RHIBindlessManager::unregister_acceleration_structure(RHIBindlessHandle handle) {
  return backend_bindless(_impl)->unregister_acceleration_structure(handle);
}

bool RHIBindlessManager::is_valid_handle(RHIBindlessHandle handle) const {
  return backend_bindless(_impl)->is_valid_handle(handle);
}

RHIResourceType RHIBindlessManager::get_resource_type(RHIBindlessHandle handle) const {
  return backend_bindless(_impl)->get_resource_type(handle);
}

uint32_t RHIBindlessManager::get_max_buffers() const {
  return backend_bindless(_impl)->get_max_buffers();
}

uint32_t RHIBindlessManager::get_max_textures() const {
  return backend_bindless(_impl)->get_max_textures();
}

uint32_t RHIBindlessManager::get_max_samplers() const {
  return backend_bindless(_impl)->get_max_samplers();
}

uint32_t RHIBindlessManager::get_max_acceleration_structures() const {
  return backend_bindless(_impl)->get_max_acceleration_structures();
}

uint32_t RHIBindlessManager::get_buffer_count() const {
  return backend_bindless(_impl)->get_buffer_count();
}

uint32_t RHIBindlessManager::get_texture_count() const {
  return backend_bindless(_impl)->get_texture_count();
}

uint32_t RHIBindlessManager::get_sampler_count() const {
  return backend_bindless(_impl)->get_sampler_count();
}

uint32_t RHIBindlessManager::get_acceleration_structure_count() const {
  return backend_bindless(_impl)->get_acceleration_structure_count();
}

}  // namespace etx
