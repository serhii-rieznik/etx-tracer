#include <etx/rhi/rhi_bindless.hxx>

#if defined(ETX_PLATFORM_WINDOWS)
# include <etx/rhi/vulkan/vk_rhi.hxx>
#elif defined(ETX_PLATFORM_APPLE)
# include <etx/rhi/metal/mt_rhi.hxx>
#else
# error Unsupported platform for RHI bindless dispatch
#endif

namespace etx {

template <typename Fn>
static decltype(auto) dispatch_bindless(RHIBackend backend, void* impl, Fn&& fn) {
#if defined(ETX_PLATFORM_WINDOWS)
  (void)backend;
  return fn(static_cast<VKBindlessManager*>(impl));
#elif defined(ETX_PLATFORM_APPLE)
  (void)backend;
  return fn(static_cast<MTBindlessManager*>(impl));
#endif
}

template <typename Fn>
static decltype(auto) dispatch_bindless(RHIBackend backend, const void* impl, Fn&& fn) {
#if defined(ETX_PLATFORM_WINDOWS)
  (void)backend;
  return fn(static_cast<const VKBindlessManager*>(impl));
#elif defined(ETX_PLATFORM_APPLE)
  (void)backend;
  return fn(static_cast<const MTBindlessManager*>(impl));
#endif
}

void RHIBindlessManager::set_max_buffers(uint32_t count) {
  dispatch_bindless(_backend, _impl, [&](auto* bindless) { bindless->set_max_buffers(count); });
}

void RHIBindlessManager::set_max_textures(uint32_t count) {
  dispatch_bindless(_backend, _impl, [&](auto* bindless) { bindless->set_max_textures(count); });
}

void RHIBindlessManager::set_max_samplers(uint32_t count) {
  dispatch_bindless(_backend, _impl, [&](auto* bindless) { bindless->set_max_samplers(count); });
}

void RHIBindlessManager::set_max_acceleration_structures(uint32_t count) {
  dispatch_bindless(_backend, _impl, [&](auto* bindless) { bindless->set_max_acceleration_structures(count); });
}

RHIResult RHIBindlessManager::register_buffer(void* buffer, RHIResourceType type, RHIBindlessHandle& out_handle) {
  return dispatch_bindless(_backend, _impl, [&](auto* bindless) { return bindless->register_buffer(buffer, type, out_handle); });
}

RHIResult RHIBindlessManager::register_texture(void* image_view, RHIResourceType type, RHIBindlessHandle& out_handle, uint32_t usage_flags, void* image) {
  return dispatch_bindless(_backend, _impl, [&](auto* bindless) { return bindless->register_texture(image_view, type, out_handle, usage_flags, image); });
}

RHIResult RHIBindlessManager::register_sampler(void* sampler, RHIResourceType type, RHIBindlessHandle& out_handle) {
  return dispatch_bindless(_backend, _impl, [&](auto* bindless) { return bindless->register_sampler(sampler, type, out_handle); });
}

RHIResult RHIBindlessManager::unregister_buffer(RHIBindlessHandle handle) {
  return dispatch_bindless(_backend, _impl, [&](auto* bindless) { return bindless->unregister_buffer(handle); });
}

RHIResult RHIBindlessManager::unregister_texture(RHIBindlessHandle handle) {
  return dispatch_bindless(_backend, _impl, [&](auto* bindless) { return bindless->unregister_texture(handle); });
}

RHIResult RHIBindlessManager::unregister_sampler(RHIBindlessHandle handle) {
  return dispatch_bindless(_backend, _impl, [&](auto* bindless) { return bindless->unregister_sampler(handle); });
}

RHIResult RHIBindlessManager::register_acceleration_structure(const void* data, uint64_t size, RHIBindlessHandle& out_handle) {
  return dispatch_bindless(_backend, _impl, [&](auto* bindless) { return bindless->register_acceleration_structure(data, size, out_handle); });
}

RHIResult RHIBindlessManager::unregister_acceleration_structure(RHIBindlessHandle handle) {
  return dispatch_bindless(_backend, _impl, [&](auto* bindless) { return bindless->unregister_acceleration_structure(handle); });
}

bool RHIBindlessManager::is_valid_handle(RHIBindlessHandle handle) const {
  return dispatch_bindless(_backend, _impl, [&](const auto* bindless) { return bindless->is_valid_handle(handle); });
}

RHIResourceType RHIBindlessManager::get_resource_type(RHIBindlessHandle handle) const {
  return dispatch_bindless(_backend, _impl, [&](const auto* bindless) { return bindless->get_resource_type(handle); });
}

uint32_t RHIBindlessManager::get_max_buffers() const {
  return dispatch_bindless(_backend, _impl, [](const auto* bindless) { return bindless->get_max_buffers(); });
}

uint32_t RHIBindlessManager::get_max_textures() const {
  return dispatch_bindless(_backend, _impl, [](const auto* bindless) { return bindless->get_max_textures(); });
}

uint32_t RHIBindlessManager::get_max_samplers() const {
  return dispatch_bindless(_backend, _impl, [](const auto* bindless) { return bindless->get_max_samplers(); });
}

uint32_t RHIBindlessManager::get_max_acceleration_structures() const {
  return dispatch_bindless(_backend, _impl, [](const auto* bindless) { return bindless->get_max_acceleration_structures(); });
}

uint32_t RHIBindlessManager::get_buffer_count() const {
  return dispatch_bindless(_backend, _impl, [](const auto* bindless) { return bindless->get_buffer_count(); });
}

uint32_t RHIBindlessManager::get_texture_count() const {
  return dispatch_bindless(_backend, _impl, [](const auto* bindless) { return bindless->get_texture_count(); });
}

uint32_t RHIBindlessManager::get_sampler_count() const {
  return dispatch_bindless(_backend, _impl, [](const auto* bindless) { return bindless->get_sampler_count(); });
}

uint32_t RHIBindlessManager::get_acceleration_structure_count() const {
  return dispatch_bindless(_backend, _impl, [](const auto* bindless) { return bindless->get_acceleration_structure_count(); });
}

}  // namespace etx
