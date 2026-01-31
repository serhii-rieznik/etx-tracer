#include <etx/rhi/metal/mt_rhi.hxx>

#include <etx/core/log.hxx>

namespace etx {

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

MTContext::~MTContext() {
  delete _impl;
}

RHIDevice* MTContext::get_device() {
  return &_impl->device;
}

RHIBindlessManager* MTContext::get_bindless_manager() {
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

void MTContext::present() {
  log::warning("Metal RHI: present not implemented");
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

RHICommandBuffer* MTContext::get_command_buffer() {
  return &_impl->command_buffer;
}

void MTContext::submit_command_buffer(RHICommandBuffer* command_buffer) {
  log::warning("Metal RHI: submit_command_buffer not implemented");
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

RHICreateShaderResult MTDevice::create_shader(const RHIShaderDesc& desc) {
  log::warning("Metal RHI: create_shader not implemented");
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
  log::warning("Metal RHI: destroy_buffer not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::destroy_texture(RHITexture texture) {
  log::warning("Metal RHI: destroy_texture not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::destroy_sampler(RHISampler sampler) {
  log::warning("Metal RHI: destroy_sampler not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::destroy_shader(RHIShader shader) {
  log::warning("Metal RHI: destroy_shader not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::destroy_pipeline(RHIPipeline pipeline) {
  log::warning("Metal RHI: destroy_pipeline not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::update_buffer(RHIBuffer buffer, const void* data, uint64_t size, uint64_t offset) {
  log::warning("Metal RHI: update_buffer not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::update_texture(RHITexture texture, const void* data, uint32_t mip_level, uint32_t array_layer) {
  log::warning("Metal RHI: update_texture not implemented");
  return RHIResult::NotImplemented;
}

bool MTDevice::supports_bindless() const {
  return true;
}

uint64_t MTDevice::get_min_uniform_buffer_offset_alignment() const {
  return 256;
}

uint64_t MTDevice::get_min_storage_buffer_offset_alignment() const {
  return 16;
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
  out_handle = 0;
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::unregister_buffer(RHIBindlessHandle handle) {
  log::warning("Metal RHI: unregister_buffer not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::register_texture(void* vk_image_view, RHIResourceType type, RHIBindlessHandle& out_handle, uint32_t usage_flags, void* vk_image) {
  log::warning("Metal RHI: register_texture not implemented");
  out_handle = 0;
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::unregister_texture(RHIBindlessHandle handle) {
  log::warning("Metal RHI: unregister_texture not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::register_sampler(void* vk_sampler, RHIResourceType type, RHIBindlessHandle& out_handle) {
  log::warning("Metal RHI: register_sampler not implemented");
  out_handle = 0;
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::unregister_sampler(RHIBindlessHandle handle) {
  log::warning("Metal RHI: unregister_sampler not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTBindlessManager::register_acceleration_structure(const void* data, uint64_t size, RHIBindlessHandle& out_handle) {
  log::warning("Metal RHI: register_acceleration_structure not implemented");
  out_handle = 0;
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

RHIResult MTDevice::reload_shader(RHIShader shader, const RHIShaderDesc& new_desc) {
  log::warning("Metal RHI: reload_shader not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::reload_graphics_pipeline(RHIPipeline pipeline, const RHIGraphicsPipelineDesc& new_desc) {
  log::warning("Metal RHI: reload_graphics_pipeline not implemented");
  return RHIResult::NotImplemented;
}

RHIResult MTDevice::reload_compute_pipeline(RHIPipeline pipeline, const RHIComputePipelineDesc& new_desc) {
  log::warning("Metal RHI: reload_compute_pipeline not implemented");
  return RHIResult::NotImplemented;
}

RHICreateShaderResult MTDevice::create_shader_variant(const RHIShaderVariantDesc& desc) {
  log::warning("Metal RHI: create_shader_variant not implemented");
  return {RHIResult::NotImplemented, {}};
}

RHICreateShaderResult MTDevice::create_shader_from_file(const std::string& file_path, const std::string& entry_point, RHIShaderStage stage,
  const std::unordered_map<std::string, std::string>& defines) {
  log::warning("Metal RHI: create_shader_from_file not implemented");
  return {RHIResult::NotImplemented, {}};
}

RHIMemoryStats MTDevice::get_memory_statistics() const {
  return {};
}

RHICreateBindlessResult MTDevice::create_acceleration_structure(const RHIAccelerationStructureDesc& desc) {
  log::warning("Metal RHI: create_acceleration_structure not implemented");
  return {RHIResult::NotImplemented, 0};
}

RHIResult MTDevice::destroy_acceleration_structure(RHIBindlessHandle as_handle) {
  log::warning("Metal RHI: destroy_acceleration_structure not implemented");
  return RHIResult::NotImplemented;
}

uint64_t MTDevice::get_acceleration_structure_device_address(RHIBindlessHandle as_handle) {
  log::warning("Metal RHI: get_acceleration_structure_device_address not implemented");
  return 0;
}

void MTCommandBuffer::build_acceleration_structure(const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset) {
  log::warning("Metal RHI: build_acceleration_structure not implemented");
}

}  // namespace etx
