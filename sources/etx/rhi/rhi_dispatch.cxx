#include <etx/rhi/rhi.hxx>
#include <utility>
#include <new>

#if defined(ETX_PLATFORM_WINDOWS)
# include <etx/rhi/vulkan/vk_rhi.hxx>
namespace etx {
using BackendContext = VKContext;
using BackendDevice = VKDevice;
}  // namespace etx
#elif defined(ETX_PLATFORM_APPLE)
# include <etx/rhi/metal/mt_rhi.hxx>
namespace etx {
using BackendContext = MTContext;
using BackendDevice = MTDevice;
}  // namespace etx
#else
# error Unsupported platform for RHI dispatch
#endif

namespace etx {

static BackendContext* backend_context(void* impl) {
  return static_cast<BackendContext*>(impl);
}

static const BackendContext* backend_context(const void* impl) {
  return static_cast<const BackendContext*>(impl);
}

static BackendDevice* backend_device(void* impl) {
  return static_cast<BackendDevice*>(impl);
}

static const BackendDevice* backend_device(const void* impl) {
  return static_cast<const BackendDevice*>(impl);
}

RHICreateResult<RHISemaphore> RHIDevice::create_semaphore() {
  return backend_device(_impl)->create_semaphore();
}

RHIResult RHIDevice::destroy_semaphore(RHISemaphore semaphore) {
  return backend_device(_impl)->destroy_semaphore(semaphore);
}

RHICreateBindlessResult RHIDevice::create_buffer(const RHIBufferDesc& desc) {
  return backend_device(_impl)->create_buffer(desc);
}

RHIResult RHIDevice::update_buffer(RHIBindlessHandle buffer, const void* data, uint64_t size, uint64_t offset) {
  return backend_device(_impl)->update_buffer(buffer, data, size, offset);
}

RHIResult RHIDevice::destroy_buffer(RHIBindlessHandle buffer) {
  return backend_device(_impl)->destroy_buffer(buffer);
}

RHICreateBindlessResult RHIDevice::create_texture(const RHITextureDesc& desc) {
  return backend_device(_impl)->create_texture(desc);
}

RHIResult RHIDevice::update_texture(RHIBindlessHandle texture, const void* data, uint32_t mip_level, uint32_t array_layer) {
  return backend_device(_impl)->update_texture(texture, data, mip_level, array_layer);
}

RHIResult RHIDevice::destroy_texture(RHIBindlessHandle texture) {
  return backend_device(_impl)->destroy_texture(texture);
}

RHICreateBindlessResult RHIDevice::create_sampler(const RHISamplerDesc& desc) {
  return backend_device(_impl)->create_sampler(desc);
}

RHIResult RHIDevice::destroy_sampler(RHIBindlessHandle sampler) {
  return backend_device(_impl)->destroy_sampler(sampler);
}

RHICreateBindlessResult RHIDevice::create_acceleration_structure(const RHIAccelerationStructureDesc& desc) {
  return backend_device(_impl)->create_acceleration_structure(desc);
}

RHIResult RHIDevice::destroy_acceleration_structure(RHIBindlessHandle as_handle) {
  return backend_device(_impl)->destroy_acceleration_structure(as_handle);
}

uint64_t RHIDevice::get_acceleration_structure_device_address(RHIBindlessHandle as_handle) {
  return backend_device(_impl)->get_acceleration_structure_device_address(as_handle);
}

RHICreatePipelineResult RHIDevice::create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) {
  return backend_device(_impl)->create_graphics_pipeline(desc);
}

RHICreatePipelineResult RHIDevice::create_compute_pipeline(const RHIComputePipelineDesc& desc) {
  return backend_device(_impl)->create_compute_pipeline(desc);
}

RHIResult RHIDevice::reload_graphics_pipeline(RHIPipeline pipeline, const RHIGraphicsPipelineDesc& new_desc) {
  return backend_device(_impl)->reload_graphics_pipeline(pipeline, new_desc);
}

RHIResult RHIDevice::reload_compute_pipeline(RHIPipeline pipeline, const RHIComputePipelineDesc& new_desc) {
  return backend_device(_impl)->reload_compute_pipeline(pipeline, new_desc);
}

RHIResult RHIDevice::destroy_pipeline(RHIPipeline pipeline) {
  return backend_device(_impl)->destroy_pipeline(pipeline);
}

RHIMemoryStats RHIDevice::get_memory_statistics() const {
  return backend_device(_impl)->get_memory_statistics();
}

void RHIContext::destroy_backend() {
  backend_context(_impl)->~BackendContext();
  _impl = nullptr;
  _device._impl = nullptr;
  _bindless._impl = nullptr;
}

void RHIContext::move_from(RHIContext&& other) {
  if (other._impl == nullptr) {
    _impl = nullptr;
    _device._impl = nullptr;
    _bindless._impl = nullptr;
    return;
  }

  auto* moved_context = new (_backend_storage) BackendContext(std::move(*backend_context(other._impl)));
  _impl = moved_context;
  _device = RHIDevice(moved_context->get_device());
  _bindless = RHIBindlessManager(moved_context->get_bindless_manager());

  backend_context(other._impl)->~BackendContext();

  other._impl = nullptr;
  other._device._impl = nullptr;
  other._bindless._impl = nullptr;
}

void RHIContext::create_swapchain(const void* native_window, uint32_t width, uint32_t height) {
  backend_context(_impl)->create_swapchain(native_window, width, height);
}

void RHIContext::destroy_swapchain() {
  backend_context(_impl)->destroy_swapchain();
}

void RHIContext::resize_swapchain(uint32_t width, uint32_t height) {
  backend_context(_impl)->resize_swapchain(width, height);
}

RHITexture RHIContext::get_current_swapchain_texture() {
  return backend_context(_impl)->get_current_swapchain_texture();
}

RHITextureFormat RHIContext::get_swapchain_format() const {
  return backend_context(_impl)->get_swapchain_format();
}

void RHIContext::begin_frame() {
  backend_context(_impl)->begin_frame();
}

void RHIContext::present() {
  backend_context(_impl)->present();
}

RHIResult RHIContext::wait_idle() {
  return backend_context(_impl)->wait_idle();
}

RHISemaphore RHIContext::get_image_acquired_semaphore() {
  return backend_context(_impl)->get_image_acquired_semaphore();
}

RHISemaphore RHIContext::get_render_complete_semaphore() {
  return backend_context(_impl)->get_render_complete_semaphore();
}

uint32_t RHIContext::get_current_frame_index() const {
  return backend_context(_impl)->get_current_frame_index();
}

uint32_t RHIContext::get_sampler_index(RHISamplerType type) const {
  return backend_context(_impl)->get_sampler_index(type);
}

RHICommandBuffer RHIContext::get_command_buffer() {
  return backend_context(_impl)->get_command_buffer();
}

void RHIContext::destroy_command_buffer(RHICommandBuffer cmd) {
  backend_context(_impl)->destroy_command_buffer(cmd);
}

void RHIContext::submit_command_buffer(const RHISubmitInfo& info) {
  backend_context(_impl)->submit_command_buffer(info);
}

void RHIContext::program_command_buffer(RHICommandBuffer cmd, std::function<void(void)> func) {
  backend_context(_impl)->program_command_buffer(cmd, std::move(func));
}

void RHIContext::command_buffer_begin(RHICommandBuffer cmd) {
  backend_context(_impl)->command_buffer_begin(cmd);
}

void RHIContext::command_buffer_end(RHICommandBuffer cmd) {
  backend_context(_impl)->command_buffer_end(cmd);
}

void RHIContext::command_buffer_reset(RHICommandBuffer cmd) {
  backend_context(_impl)->command_buffer_reset(cmd);
}

void RHIContext::cmd_buffer_barrier(RHICommandBuffer cmd, RHIBindlessHandle buffer, RHIResourceState old_state, RHIResourceState new_state) {
  backend_context(_impl)->cmd_buffer_barrier(cmd, buffer, old_state, new_state);
}

void RHIContext::cmd_texture_barrier(RHICommandBuffer cmd, RHIBindlessHandle texture, RHIResourceState old_state, RHIResourceState new_state) {
  backend_context(_impl)->cmd_texture_barrier(cmd, texture, old_state, new_state);
}

void RHIContext::cmd_begin_render_pass(RHICommandBuffer cmd, uint32_t color_attachment_count, RHIBindlessHandle* color_attachments, const float* clear_colors,
  RHIBindlessHandle depth_attachment, const RHIResourceState* color_final_states, RHIResourceState depth_final_state) {
  backend_context(_impl)->cmd_begin_render_pass(cmd, color_attachment_count, color_attachments, clear_colors, depth_attachment, color_final_states, depth_final_state);
}

void RHIContext::cmd_end_render_pass(RHICommandBuffer cmd) {
  backend_context(_impl)->cmd_end_render_pass(cmd);
}

void RHIContext::cmd_set_viewport(RHICommandBuffer cmd, const RHIViewport& viewport) {
  backend_context(_impl)->cmd_set_viewport(cmd, viewport);
}

void RHIContext::cmd_set_scissor(RHICommandBuffer cmd, const RHIRect& scissor) {
  backend_context(_impl)->cmd_set_scissor(cmd, scissor);
}

void RHIContext::cmd_set_pipeline(RHICommandBuffer cmd, RHIPipeline pipeline) {
  backend_context(_impl)->cmd_set_pipeline(cmd, pipeline);
}

void RHIContext::cmd_push_constants(RHICommandBuffer cmd, const void* data, uint32_t size, uint32_t offset) {
  backend_context(_impl)->cmd_push_constants(cmd, data, size, offset);
}

void RHIContext::cmd_draw(RHICommandBuffer cmd, const RHIDrawDesc& desc) {
  backend_context(_impl)->cmd_draw(cmd, desc);
}

void RHIContext::cmd_draw_indexed(RHICommandBuffer cmd, const RHIIndexedDrawDesc& desc, RHIBindlessHandle index_buffer) {
  backend_context(_impl)->cmd_draw_indexed(cmd, desc, index_buffer);
}

void RHIContext::cmd_dispatch(RHICommandBuffer cmd, const RHIDispatchDesc& desc) {
  backend_context(_impl)->cmd_dispatch(cmd, desc);
}

void RHIContext::cmd_build_acceleration_structure(RHICommandBuffer cmd, const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset) {
  backend_context(_impl)->cmd_build_acceleration_structure(cmd, desc, scratch_buffer, scratch_offset);
}

void RHIContext::cmd_copy_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint64_t size, uint64_t src_offset, uint64_t dst_offset) {
  backend_context(_impl)->cmd_copy_buffer(cmd, src, dst, size, src_offset, dst_offset);
}

void RHIContext::cmd_copy_buffer_to_texture(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  backend_context(_impl)->cmd_copy_buffer_to_texture(cmd, src, dst, width, height, mip_level);
}

void RHIContext::cmd_copy_texture_to_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  backend_context(_impl)->cmd_copy_texture_to_buffer(cmd, src, dst, width, height, mip_level);
}

void RHIContext::cmd_set_debug_name(RHICommandBuffer cmd, const char* name) {
  backend_context(_impl)->cmd_set_debug_name(cmd, name);
}

}  // namespace etx
