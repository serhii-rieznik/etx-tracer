#include <etx/rhi/rhi.hxx>
#include <etx/core/log.hxx>
#include <utility>
#include <new>
#include <cstring>
#include <limits>

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

namespace {
constexpr uint32_t kInvalidDescriptorIndex = ~0u;

void destroy_linear_chunked_buffer(RHIDevice& device, RHIBindlessHandle& buffer, uint64_t& buffer_size, uint32_t& descriptor_index) {
  device.destroy_buffer(buffer);
  buffer = {};
  buffer_size = 0u;
  descriptor_index = kInvalidDescriptorIndex;
}

bool upload_or_update_linear_chunked_buffer(RHIDevice& device, const void* data, uint64_t size, RHIBufferUsage usage, RHIBindlessHandle& buffer, uint64_t& buffer_size,
  uint32_t& descriptor_index, const char* buffer_name) {
  if ((data == nullptr) || (size == 0u)) {
    destroy_linear_chunked_buffer(device, buffer, buffer_size, descriptor_index);
    return true;
  }

  const uint64_t required_size = size;
  if ((buffer.valid()) && (buffer_size == required_size)) {
    const RHIResult update_result = device.update_buffer(buffer, data, required_size);
    if (update_result != RHIResult::Success) {
      log::error("RHI: failed to update '%s' buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(update_result));
      return false;
    }
    descriptor_index = get_bindless_descriptor_index(buffer);
    return true;
  }

  RHIBufferDesc desc = {};
  desc.size = required_size;
  desc.usage = usage;

  const RHICreateBindlessResult create_result = device.create_buffer(desc);
  if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
    log::error("RHI: failed to create '%s' buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(create_result.result));
    descriptor_index = buffer.valid() ? get_bindless_descriptor_index(buffer) : kInvalidDescriptorIndex;
    return false;
  }

  const RHIResult upload_result = device.update_buffer(create_result.handle, data, required_size);
  if (upload_result != RHIResult::Success) {
    log::error("RHI: failed to upload '%s' buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(upload_result));
    device.destroy_buffer(create_result.handle);
    descriptor_index = buffer.valid() ? get_bindless_descriptor_index(buffer) : kInvalidDescriptorIndex;
    return false;
  }

  if (buffer.valid()) {
    const RHIResult destroy_result = device.destroy_buffer(buffer);
    if (destroy_result != RHIResult::Success) {
      log::error("RHI: failed to destroy previous '%s' buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(destroy_result));
      device.destroy_buffer(create_result.handle);
      descriptor_index = get_bindless_descriptor_index(buffer);
      return false;
    }
  }

  buffer = create_result.handle;
  buffer_size = required_size;
  descriptor_index = get_bindless_descriptor_index(buffer);
  return true;
}
}  // namespace

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
  if (semaphore.valid() == false) {
    return RHIResult::Success;
  }
  if (_impl == nullptr) {
    return RHIResult::Success;
  }
  return backend_device(_impl)->destroy_semaphore(semaphore);
}

RHICreateBindlessResult RHIDevice::create_buffer(const RHIBufferDesc& desc) {
  return backend_device(_impl)->create_buffer(desc);
}

RHIResult RHIDevice::update_buffer(RHIBindlessHandle buffer, const void* data, uint64_t size, uint64_t offset) {
  return backend_device(_impl)->update_buffer(buffer, data, size, offset);
}

RHIResult RHIDevice::read_buffer(RHIBindlessHandle buffer, void* data, uint64_t size, uint64_t offset) {
  return backend_device(_impl)->read_buffer(buffer, data, size, offset);
}

RHIResult RHIDevice::destroy_buffer(RHIBindlessHandle buffer) {
  if (buffer.valid() == false) {
    return RHIResult::Success;
  }
  if (_impl == nullptr) {
    return RHIResult::Success;
  }
  return backend_device(_impl)->destroy_buffer(buffer);
}

bool RHIDevice::upload_or_update_chunked_buffer(const RHIChunkedBufferUploadData& data, RHIBufferUsage usage, RHIChunkedBufferState& state, const char* buffer_name) {
  if (data.success == false) {
    return false;
  }

  const size_t required_chunk_count = data.payload_chunk_ranges.size();
  for (size_t i = 0u; i < required_chunk_count; ++i) {
    const auto& range = data.payload_chunk_ranges[i];
    if (range.offset > static_cast<uint64_t>(data.payload_data.size())) {
      log::error("RHI: invalid chunk range offset for '%s' (%llu)", (buffer_name != nullptr) ? buffer_name : "unknown", range.offset);
      return false;
    }
    if (range.size > (static_cast<uint64_t>(data.payload_data.size()) - range.offset)) {
      log::error("RHI: invalid chunk range size for '%s' (offset=%llu, size=%llu, payload=%llu)", (buffer_name != nullptr) ? buffer_name : "unknown", range.offset, range.size,
        static_cast<uint64_t>(data.payload_data.size()));
      return false;
    }
  }

  std::vector<RHIBindlessHandle> new_chunk_buffers(required_chunk_count);
  std::vector<uint64_t> new_chunk_buffer_sizes(required_chunk_count, 0u);
  RHIBindlessHandle new_metadata_buffer = {};
  uint64_t new_metadata_buffer_size = 0u;
  uint32_t new_metadata_descriptor_index = kInvalidDescriptorIndex;

  auto cleanup_new_state = [&]() {
    for (auto chunk_buffer : new_chunk_buffers) {
      destroy_buffer(chunk_buffer);
    }
    destroy_linear_chunked_buffer(*this, new_metadata_buffer, new_metadata_buffer_size, new_metadata_descriptor_index);
  };

  for (size_t i = 0u; i < required_chunk_count; ++i) {
    const auto& range = data.payload_chunk_ranges[i];
    const void* chunk_data = nullptr;
    if (range.size > 0u) {
      chunk_data = data.payload_data.data() + static_cast<size_t>(range.offset);
    }

    uint32_t chunk_descriptor_index = kInvalidDescriptorIndex;
    const bool chunk_upload_success =
      upload_or_update_linear_chunked_buffer(*this, chunk_data, range.size, usage, new_chunk_buffers[i], new_chunk_buffer_sizes[i], chunk_descriptor_index, buffer_name);
    if (chunk_upload_success == false) {
      cleanup_new_state();
      return false;
    }
  }

  auto metadata = data.metadata;
  if ((required_chunk_count > 0u) && (data.chunk_indices_offset != kInvalidDescriptorIndex)) {
    const uint64_t table_offset = static_cast<uint64_t>(data.chunk_indices_offset);
    const uint64_t table_size = static_cast<uint64_t>(required_chunk_count) * sizeof(uint32_t);
    if (table_offset > (std::numeric_limits<uint64_t>::max() - table_size)) {
      log::error("RHI: chunk indices table for '%s' overflows metadata range", (buffer_name != nullptr) ? buffer_name : "unknown");
      cleanup_new_state();
      return false;
    }
    const uint64_t table_end = table_offset + table_size;
    if (table_end > static_cast<uint64_t>(metadata.size())) {
      log::error("RHI: chunk indices table for '%s' is out of metadata range", (buffer_name != nullptr) ? buffer_name : "unknown");
      cleanup_new_state();
      return false;
    }

    for (size_t i = 0u; i < required_chunk_count; ++i) {
      const uint32_t chunk_descriptor_index = get_bindless_descriptor_index(new_chunk_buffers[i]);
      const uint64_t descriptor_offset = table_offset + (static_cast<uint64_t>(i) * sizeof(uint32_t));
      std::memcpy(metadata.data() + static_cast<size_t>(descriptor_offset), &chunk_descriptor_index, sizeof(uint32_t));
    }
  }

  const void* metadata_ptr = (metadata.empty() == false) ? metadata.data() : nullptr;
  const bool metadata_upload_success = upload_or_update_linear_chunked_buffer(*this, metadata_ptr, static_cast<uint64_t>(metadata.size()), usage, new_metadata_buffer,
    new_metadata_buffer_size, new_metadata_descriptor_index, buffer_name);
  if (metadata_upload_success == false) {
    cleanup_new_state();
    return false;
  }

  for (auto chunk_buffer : state.chunk_buffers) {
    destroy_buffer(chunk_buffer);
  }
  destroy_linear_chunked_buffer(*this, state.metadata_buffer, state.metadata_buffer_size, state.metadata_descriptor_index);

  state.chunk_buffers = std::move(new_chunk_buffers);
  state.chunk_buffer_sizes = std::move(new_chunk_buffer_sizes);
  state.metadata_buffer = new_metadata_buffer;
  state.metadata_buffer_size = new_metadata_buffer_size;
  state.metadata_descriptor_index = new_metadata_descriptor_index;

  return true;
}

void RHIDevice::destroy_chunked_buffer(RHIChunkedBufferState& state) {
  for (auto chunk_buffer : state.chunk_buffers) {
    destroy_buffer(chunk_buffer);
  }
  state.chunk_buffers.clear();
  state.chunk_buffer_sizes.clear();
  destroy_linear_chunked_buffer(*this, state.metadata_buffer, state.metadata_buffer_size, state.metadata_descriptor_index);
}

RHICreateBindlessResult RHIDevice::create_texture(const RHITextureDesc& desc) {
  return backend_device(_impl)->create_texture(desc);
}

RHIResult RHIDevice::update_texture(RHIBindlessHandle texture, const void* data, uint32_t mip_level, uint32_t array_layer) {
  return backend_device(_impl)->update_texture(texture, data, mip_level, array_layer);
}

RHIResult RHIDevice::destroy_texture(RHIBindlessHandle texture) {
  if (texture.valid() == false) {
    return RHIResult::Success;
  }
  if (_impl == nullptr) {
    return RHIResult::Success;
  }
  return backend_device(_impl)->destroy_texture(texture);
}

RHICreateBindlessResult RHIDevice::create_sampler(const RHISamplerDesc& desc) {
  return backend_device(_impl)->create_sampler(desc);
}

RHIResult RHIDevice::destroy_sampler(RHIBindlessHandle sampler) {
  if (sampler.valid() == false) {
    return RHIResult::Success;
  }
  if (_impl == nullptr) {
    return RHIResult::Success;
  }
  return backend_device(_impl)->destroy_sampler(sampler);
}

RHICreateBindlessResult RHIDevice::create_acceleration_structure(const RHIAccelerationStructureDesc& desc) {
  return backend_device(_impl)->create_acceleration_structure(desc);
}

RHIResult RHIDevice::destroy_acceleration_structure(RHIBindlessHandle as_handle) {
  if (as_handle.valid() == false) {
    return RHIResult::Success;
  }
  if (_impl == nullptr) {
    return RHIResult::Success;
  }
  return backend_device(_impl)->destroy_acceleration_structure(as_handle);
}

uint64_t RHIDevice::get_acceleration_structure_device_address(RHIBindlessHandle as_handle) {
  return backend_device(_impl)->get_acceleration_structure_device_address(as_handle);
}

uint64_t RHIDevice::get_acceleration_structure_build_scratch_size(RHIBindlessHandle as_handle) {
  return backend_device(_impl)->get_acceleration_structure_build_scratch_size(as_handle);
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
  if (pipeline.valid() == false) {
    return RHIResult::Success;
  }
  if (_impl == nullptr) {
    return RHIResult::Success;
  }
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

RHIExtent2D RHIContext::get_swapchain_extent() const {
  return backend_context(_impl)->get_swapchain_extent_rhi();
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

void RHIContext::submit_frame_command_buffer(RHICommandBuffer cmd) {
  RHISubmitInfo submit_info = {};
  submit_info.command_buffer = cmd;
  submit_info.wait_semaphores.push_back(get_image_acquired_semaphore());
  submit_info.signal_semaphores.push_back(get_render_complete_semaphore());
  submit_command_buffer(submit_info);
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

void RHIContext::cmd_reset_timestamps(RHICommandBuffer cmd, uint32_t first_query, uint32_t query_count) {
  backend_context(_impl)->cmd_reset_timestamps(cmd, first_query, query_count);
}

void RHIContext::cmd_write_timestamp(RHICommandBuffer cmd, uint32_t query_index, RHITimestampStage stage) {
  backend_context(_impl)->cmd_write_timestamp(cmd, query_index, stage);
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

void RHIContext::cmd_resolve_texture(RHICommandBuffer cmd, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height) {
  backend_context(_impl)->cmd_resolve_texture(cmd, src, dst, width, height);
}

void RHIContext::cmd_generate_mipmaps(RHICommandBuffer cmd, RHIBindlessHandle texture) {
  backend_context(_impl)->cmd_generate_mipmaps(cmd, texture);
}

void RHIContext::cmd_set_debug_name(RHICommandBuffer cmd, const char* name) {
  backend_context(_impl)->cmd_set_debug_name(cmd, name);
}

bool RHIContext::supports_timestamps() const {
  return backend_context(_impl)->supports_timestamps();
}

double RHIContext::timestamp_period_ns() const {
  return backend_context(_impl)->timestamp_period_ns();
}

RHIResult RHIContext::read_timestamps(RHICommandBuffer cmd, uint32_t first_query, uint32_t query_count, uint64_t* out_values) {
  return backend_context(_impl)->read_timestamps(cmd, first_query, query_count, out_values);
}

}  // namespace etx
