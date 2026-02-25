#include <etx/rhi/metal/mt_rhi.hxx>

#include <etx/core/log.hxx>
#include <new>

namespace etx {

void create_metal_context(RHIContext& context, const RHIInitInfo& info) {
  (void)info;
  static_assert(sizeof(MTContext) <= RHIContext::kBackendStorageSize, "MTContext does not fit into RHIContext backend storage");
  static_assert(alignof(MTContext) <= RHIContext::kBackendStorageAlignment, "MTContext alignment exceeds RHIContext backend storage alignment");
  auto* mt_context = new (context._backend_storage) MTContext();
  context.initialize_backend(mt_context, mt_context->get_device(), mt_context->get_bindless_manager());
}

class MTContext::Impl {
 public:
  MTDevice device;
  MTBindlessManager bindless_manager;
  MTCommandBuffer command_buffer;

  const void* native_window = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t current_frame = 0;
};

MTContext::MTContext()
  : _impl(new Impl()) {
}

MTContext::MTContext(MTContext&& other) noexcept
  : _impl(other._impl) {
  other._impl = nullptr;
}

MTContext::~MTContext() {
  delete _impl;
}

MTDevice* MTContext::get_device() {
  return &_impl->device;
}

MTBindlessManager* MTContext::get_bindless_manager() {
  return &_impl->bindless_manager;
}

void MTContext::create_swapchain(const void* native_window, uint32_t width, uint32_t height) {
  _impl->native_window = native_window;
  _impl->width = width;
  _impl->height = height;
  log::warning("Metal RHI: create_swapchain not implemented");
}

void MTContext::destroy_swapchain() {
  log::warning("Metal RHI: destroy_swapchain not implemented");
}

void MTContext::resize_swapchain(uint32_t width, uint32_t height) {
  _impl->width = width;
  _impl->height = height;
  log::warning("Metal RHI: resize_swapchain not implemented");
}

RHITexture MTContext::get_current_swapchain_texture() {
  log::warning("Metal RHI: get_current_swapchain_texture not implemented");
  return {};
}

RHITextureFormat MTContext::get_swapchain_format() const {
  return RHITextureFormat::B8G8R8A8_SRGB;
}

RHIExtent2D MTContext::get_swapchain_extent_rhi() const {
  RHIExtent2D result = {};
  result.width = _impl->width;
  result.height = _impl->height;
  return result;
}

void MTContext::present() {
  log::warning("Metal RHI: present not implemented");
}

RHIResult MTContext::wait_idle() {
  return RHIResult::Success;
}

void MTContext::begin_frame() {
  _impl->current_frame = (_impl->current_frame + 1) % 2;
}

uint32_t MTContext::get_current_frame_index() const {
  return _impl->current_frame;
}

uint32_t MTContext::get_sampler_index(RHISamplerType type) const {
  return static_cast<uint32_t>(type);
}

RHICommandBuffer MTContext::get_command_buffer() {
  return {1u};
}

void MTContext::destroy_command_buffer(RHICommandBuffer cmd) {
  (void)cmd;
}

void MTContext::submit_command_buffer(const RHISubmitInfo& info) {
  (void)info;
  log::warning("Metal RHI: submit_command_buffer not implemented");
}

void MTContext::program_command_buffer(RHICommandBuffer cmd, std::function<void(void)> func) {
  (void)cmd;
  if (func) {
    func();
  }
}

void MTContext::command_buffer_begin(RHICommandBuffer cmd) {
  (void)cmd;
  _impl->command_buffer.begin();
}

void MTContext::command_buffer_end(RHICommandBuffer cmd) {
  (void)cmd;
  _impl->command_buffer.end();
}

void MTContext::command_buffer_reset(RHICommandBuffer cmd) {
  (void)cmd;
  _impl->command_buffer.reset();
}

void MTContext::cmd_buffer_barrier(RHICommandBuffer cmd, RHIBindlessHandle buffer, RHIResourceState old_state, RHIResourceState new_state) {
  (void)cmd;
  _impl->command_buffer.buffer_barrier(buffer, old_state, new_state);
}

void MTContext::cmd_texture_barrier(RHICommandBuffer cmd, RHIBindlessHandle texture, RHIResourceState old_state, RHIResourceState new_state) {
  (void)cmd;
  _impl->command_buffer.texture_barrier(texture, old_state, new_state);
}

void MTContext::cmd_begin_render_pass(RHICommandBuffer cmd, uint32_t color_attachment_count, RHIBindlessHandle* color_attachments, const float* clear_colors,
  RHIBindlessHandle depth_attachment, const RHIResourceState* color_final_states, RHIResourceState depth_final_state) {
  (void)cmd;
  (void)color_final_states;
  (void)depth_final_state;
  _impl->command_buffer.begin_render_pass(color_attachment_count, color_attachments, clear_colors, depth_attachment);
}

void MTContext::cmd_end_render_pass(RHICommandBuffer cmd) {
  (void)cmd;
  _impl->command_buffer.end_render_pass();
}

void MTContext::cmd_set_viewport(RHICommandBuffer cmd, const RHIViewport& viewport) {
  (void)cmd;
  _impl->command_buffer.set_viewport(viewport);
}

void MTContext::cmd_set_scissor(RHICommandBuffer cmd, const RHIRect& scissor) {
  (void)cmd;
  _impl->command_buffer.set_scissor(scissor);
}

void MTContext::cmd_set_pipeline(RHICommandBuffer cmd, RHIPipeline pipeline) {
  (void)cmd;
  _impl->command_buffer.set_pipeline(pipeline);
}

void MTContext::cmd_push_constants(RHICommandBuffer cmd, const void* data, uint32_t size, uint32_t offset) {
  (void)cmd;
  _impl->command_buffer.push_constants(data, size, offset);
}

void MTContext::cmd_draw(RHICommandBuffer cmd, const RHIDrawDesc& desc) {
  (void)cmd;
  _impl->command_buffer.draw(desc);
}

void MTContext::cmd_draw_indexed(RHICommandBuffer cmd, const RHIIndexedDrawDesc& desc, RHIBindlessHandle index_buffer) {
  (void)cmd;
  _impl->command_buffer.draw_indexed(desc, index_buffer);
}

void MTContext::cmd_dispatch(RHICommandBuffer cmd, const RHIDispatchDesc& desc) {
  (void)cmd;
  _impl->command_buffer.dispatch(desc);
}

void MTContext::cmd_reset_timestamps(RHICommandBuffer cmd, uint32_t first_query, uint32_t query_count) {
  (void)cmd;
  (void)first_query;
  (void)query_count;
  log::warning("Metal RHI: cmd_reset_timestamps not implemented");
}

void MTContext::cmd_write_timestamp(RHICommandBuffer cmd, uint32_t query_index, RHITimestampStage stage) {
  (void)cmd;
  (void)query_index;
  (void)stage;
  log::warning("Metal RHI: cmd_write_timestamp not implemented");
}

void MTContext::cmd_build_acceleration_structure(RHICommandBuffer cmd, const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset) {
  (void)cmd;
  _impl->command_buffer.build_acceleration_structure(desc, scratch_buffer, scratch_offset);
}

void MTContext::cmd_copy_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint64_t size, uint64_t src_offset, uint64_t dst_offset) {
  (void)cmd;
  _impl->command_buffer.copy_buffer(src, dst, size, src_offset, dst_offset);
}

void MTContext::cmd_copy_buffer_to_texture(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  (void)cmd;
  _impl->command_buffer.copy_buffer_to_texture(src, dst, width, height, mip_level);
}

void MTContext::cmd_copy_texture_to_buffer(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  (void)cmd;
  _impl->command_buffer.copy_texture_to_buffer(src, dst, width, height, mip_level);
}

void MTContext::cmd_resolve_texture(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height) {
  (void)cmd;
  (void)src;
  (void)dst;
  (void)width;
  (void)height;
  log::warning("Metal RHI: cmd_resolve_texture not implemented");
}

void MTContext::cmd_generate_mipmaps(RHICommandBuffer cmd, RHIBindlessHandle texture) {
  (void)cmd;
  (void)texture;
  log::warning("Metal RHI: cmd_generate_mipmaps not implemented");
}

void MTContext::cmd_set_debug_name(RHICommandBuffer cmd, const char* name) {
  (void)cmd;
  _impl->command_buffer.set_debug_name(name);
}

bool MTContext::supports_timestamps() const {
  return false;
}

double MTContext::timestamp_period_ns() const {
  return 0.0;
}

RHIResult MTContext::read_timestamps(RHICommandBuffer cmd, uint32_t first_query, uint32_t query_count, uint64_t* out_values) {
  (void)cmd;
  (void)first_query;
  (void)query_count;
  (void)out_values;
  return RHIResult::NotImplemented;
}

RHISemaphore MTContext::get_image_acquired_semaphore() {
  log::warning("Metal RHI: get_image_acquired_semaphore not implemented");
  return {};
}

RHISemaphore MTContext::get_render_complete_semaphore() {
  log::warning("Metal RHI: get_render_complete_semaphore not implemented");
  return {};
}

class MTDevice::Impl {
 public:
};

MTDevice::MTDevice()
  : _impl(new Impl()) {
}

MTDevice::~MTDevice() {
  delete _impl;
}

RHICreateResult<RHISemaphore> MTDevice::create_semaphore() {
  log::warning("Metal RHI: create_semaphore not implemented");
  return {RHIResult::NotImplemented, {}};
}

RHIResult MTDevice::destroy_semaphore(RHISemaphore semaphore) {
  if (semaphore.invalid()) {
    return RHIResult::Success;
  }

  log::warning("Metal RHI: destroy_semaphore not implemented");
  return RHIResult::NotImplemented;
}

RHICreateBindlessResult MTDevice::create_buffer(const RHIBufferDesc& desc) {
  log::warning("Metal RHI: create_buffer not implemented");
  return {RHIResult::NotImplemented, {}};
}

RHICreateBindlessResult MTDevice::create_texture(const RHITextureDesc& desc) {
  log::warning("Metal RHI: create_texture not implemented");
  return {RHIResult::NotImplemented, {}};
}

RHICreateBindlessResult MTDevice::create_sampler(const RHISamplerDesc& desc) {
  log::warning("Metal RHI: create_sampler not implemented");
  return {RHIResult::NotImplemented, {}};
}

RHICreatePipelineResult MTDevice::create_graphics_pipeline(const RHIGraphicsPipelineDesc& desc) {
  log::warning("Metal RHI: create_graphics_pipeline not implemented");
  return {RHIResult::NotImplemented, {}};
}

RHICreatePipelineResult MTDevice::create_compute_pipeline(const RHIComputePipelineDesc& desc) {
  log::warning("Metal RHI: create_compute_pipeline not implemented");
  return {RHIResult::NotImplemented, {}};
}

RHIResult MTDevice::destroy_buffer(RHIBuffer buffer) {
  if (buffer.valid() == false) {
    return RHIResult::Success;
  }

  log::warning("Metal RHI: destroy_buffer not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::destroy_texture(RHITexture texture) {
  if (texture.valid() == false) {
    return RHIResult::Success;
  }

  log::warning("Metal RHI: destroy_texture not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::destroy_sampler(RHISampler sampler) {
  if (sampler.valid() == false) {
    return RHIResult::Success;
  }

  log::warning("Metal RHI: destroy_sampler not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::destroy_pipeline(RHIPipeline pipeline) {
  if (pipeline.valid() == false) {
    return RHIResult::Success;
  }

  log::warning("Metal RHI: destroy_pipeline not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::update_buffer(RHIBuffer buffer, const void* data, uint64_t size, uint64_t offset) {
  log::warning("Metal RHI: update_buffer not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::read_buffer(RHIBuffer buffer, void* data, uint64_t size, uint64_t offset) {
  log::warning("Metal RHI: read_buffer not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::update_texture(RHITexture texture, const void* data, uint32_t mip_level, uint32_t array_layer) {
  log::warning("Metal RHI: update_texture not implemented");
  return RHIResult::NotImplemented;
}

class MTBindlessManager::Impl {
 public:
};

MTBindlessManager::MTBindlessManager()
  : _impl(new Impl()) {
}

MTBindlessManager::~MTBindlessManager() {
  delete _impl;
}

void MTBindlessManager::set_max_buffers(uint32_t count) {
  log::warning("Metal RHI: set_max_buffers not implemented");
}

void MTBindlessManager::set_max_textures(uint32_t count) {
  log::warning("Metal RHI: set_max_textures not implemented");
}

void MTBindlessManager::set_max_samplers(uint32_t count) {
  log::warning("Metal RHI: set_max_samplers not implemented");
}

void MTBindlessManager::set_max_acceleration_structures(uint32_t count) {
  log::warning("Metal RHI: set_max_acceleration_structures not implemented");
}

RHIResult MTBindlessManager::register_buffer(void* vk_buffer, RHIResourceType type, RHIBindlessHandle& out_handle) {
  log::warning("Metal RHI: register_buffer not implemented");
  out_handle = {};
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::unregister_buffer(RHIBindlessHandle handle) {
  log::warning("Metal RHI: unregister_buffer not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::register_texture(void* vk_image_view, RHIResourceType type, RHIBindlessHandle& out_handle, uint32_t usage_flags, void* vk_image) {
  log::warning("Metal RHI: register_texture not implemented");
  out_handle = {};
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::unregister_texture(RHIBindlessHandle handle) {
  log::warning("Metal RHI: unregister_texture not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::register_sampler(void* vk_sampler, RHIResourceType type, RHIBindlessHandle& out_handle) {
  log::warning("Metal RHI: register_sampler not implemented");
  out_handle = {};
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::unregister_sampler(RHIBindlessHandle handle) {
  log::warning("Metal RHI: unregister_sampler not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::register_acceleration_structure(const void* data, uint64_t size, RHIBindlessHandle& out_handle) {
  log::warning("Metal RHI: register_acceleration_structure not implemented");
  out_handle = {};
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::unregister_acceleration_structure(RHIBindlessHandle handle) {
  log::warning("Metal RHI: unregister_acceleration_structure not implemented");
  return RHIResult::NotImplemented;
}

bool MTBindlessManager::is_valid_handle(RHIBindlessHandle handle) const {
  return false;
}

RHIResourceType MTBindlessManager::get_resource_type(RHIBindlessHandle handle) const {
  return RHIResourceType::Buffer;
}

uint32_t MTBindlessManager::get_max_buffers() const {
  return 1000000;
}

uint32_t MTBindlessManager::get_max_textures() const {
  return 1000000;
}

uint32_t MTBindlessManager::get_max_samplers() const {
  return 1000;
}

uint32_t MTBindlessManager::get_max_acceleration_structures() const {
  return 10000;
}

uint32_t MTBindlessManager::get_buffer_count() const {
  return 0;
}

uint32_t MTBindlessManager::get_texture_count() const {
  return 0;
}

uint32_t MTBindlessManager::get_sampler_count() const {
  return 0;
}

uint32_t MTBindlessManager::get_acceleration_structure_count() const {
  return 0;
}

class MTCommandBuffer::Impl {
 public:
};

MTCommandBuffer::MTCommandBuffer()
  : _impl(new Impl()) {
}

MTCommandBuffer::~MTCommandBuffer() {
  delete _impl;
}

void MTCommandBuffer::begin() {
  log::warning("Metal RHI: command buffer begin not implemented");
}

void MTCommandBuffer::end() {
  log::warning("Metal RHI: command buffer end not implemented");
}

void MTCommandBuffer::reset() {
  log::warning("Metal RHI: command buffer reset not implemented");
}

void MTCommandBuffer::buffer_barrier(RHIBuffer buffer, RHIResourceState old_state, RHIResourceState new_state) {
  log::warning("Metal RHI: buffer_barrier not implemented");
}

void MTCommandBuffer::texture_barrier(RHITexture texture, RHIResourceState old_state, RHIResourceState new_state) {
  log::warning("Metal RHI: texture_barrier not implemented");
}

void MTCommandBuffer::begin_render_pass(uint32_t color_attachment_count, RHITexture* color_attachments, const float* clear_colors, RHITexture depth_attachment) {
  log::warning("Metal RHI: begin_render_pass not implemented");
}

void MTCommandBuffer::end_render_pass() {
  log::warning("Metal RHI: end_render_pass not implemented");
}

void MTCommandBuffer::set_viewport(const RHIViewport& viewport) {
  log::warning("Metal RHI: set_viewport not implemented");

  RHIRect scissor = {};
  scissor.x = static_cast<int32_t>(viewport.x);
  scissor.y = static_cast<int32_t>(viewport.y);
  scissor.width = static_cast<uint32_t>(viewport.width);
  scissor.height = static_cast<uint32_t>(viewport.height);
  set_scissor(scissor);
}

void MTCommandBuffer::set_scissor(const RHIRect& scissor) {
  log::warning("Metal RHI: set_scissor not implemented");
}

void MTCommandBuffer::set_pipeline(RHIPipeline pipeline) {
  log::warning("Metal RHI: set_pipeline not implemented");
}

void MTCommandBuffer::push_constants(const void* data, uint32_t size, uint32_t offset) {
  log::warning("Metal RHI: push_constants not implemented");
}

void MTCommandBuffer::draw(const RHIDrawDesc& desc) {
  log::warning("Metal RHI: draw not implemented");
}

void MTCommandBuffer::draw_indexed(const RHIIndexedDrawDesc& desc, RHIBuffer index_buffer) {
  log::warning("Metal RHI: draw_indexed not implemented");
}

void MTCommandBuffer::dispatch(const RHIDispatchDesc& desc) {
  log::warning("Metal RHI: dispatch not implemented");
}

void MTCommandBuffer::copy_buffer(RHIBuffer src, RHIBuffer dst, uint64_t size, uint64_t src_offset, uint64_t dst_offset) {
  log::warning("Metal RHI: copy_buffer not implemented");
}

void MTCommandBuffer::copy_buffer_to_texture(RHIBuffer src, RHITexture dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  log::warning("Metal RHI: copy_buffer_to_texture not implemented");
}

void MTCommandBuffer::copy_texture_to_buffer(RHITexture src, RHIBuffer dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  log::warning("Metal RHI: copy_texture_to_buffer not implemented");
}

void MTCommandBuffer::set_debug_name(const char* name) {
  log::warning("Metal RHI: set_debug_name not implemented");
}

RHIResult MTDevice::reload_graphics_pipeline(RHIPipeline pipeline, const RHIGraphicsPipelineDesc& new_desc) {
  log::warning("Metal RHI: reload_graphics_pipeline not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::reload_compute_pipeline(RHIPipeline pipeline, const RHIComputePipelineDesc& new_desc) {
  log::warning("Metal RHI: reload_compute_pipeline not implemented");
  return RHIResult::NotImplemented;
}

RHIMemoryStats MTDevice::get_memory_statistics() const {
  return {};
}

RHICreateBindlessResult MTDevice::create_acceleration_structure(const RHIAccelerationStructureDesc& desc) {
  log::warning("Metal RHI: create_acceleration_structure not implemented");
  return {RHIResult::NotImplemented, {}};
}

RHIResult MTDevice::destroy_acceleration_structure(RHIBindlessHandle as_handle) {
  if (as_handle.valid() == false) {
    return RHIResult::Success;
  }

  log::warning("Metal RHI: destroy_acceleration_structure not implemented");
  return RHIResult::NotImplemented;
}

uint64_t MTDevice::get_acceleration_structure_device_address(RHIBindlessHandle as_handle) {
  log::warning("Metal RHI: get_acceleration_structure_device_address not implemented");
  return 0;
}

uint64_t MTDevice::get_acceleration_structure_build_scratch_size(RHIBindlessHandle as_handle) {
  (void)as_handle;
  log::warning("Metal RHI: get_acceleration_structure_build_scratch_size not implemented");
  return 0;
}

void MTCommandBuffer::build_acceleration_structure(const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset) {
  log::warning("Metal RHI: build_acceleration_structure not implemented");
}

}  // namespace etx
