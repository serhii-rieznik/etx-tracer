#include <etx/rhi/vulkan/vk_rhi.hxx>
#include <etx/rhi/vulkan/vk_utils.hxx>

#include <etx/core/log.hxx>
#include <etx/core/core.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <new>
const char* vk_error_to_string(VkResult err) {
  switch (err) {
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_NOT_READY:
      return "VK_NOT_READY";
    case VK_TIMEOUT:
      return "VK_TIMEOUT";
    case VK_EVENT_SET:
      return "VK_EVENT_SET";
    case VK_EVENT_RESET:
      return "VK_EVENT_RESET";
    case VK_INCOMPLETE:
      return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
      return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
      return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:
      return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL:
      return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_UNKNOWN:
      return "VK_ERROR_UNKNOWN";
    case VK_ERROR_OUT_OF_POOL_MEMORY:
      return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE:
      return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_FRAGMENTATION:
      return "VK_ERROR_FRAGMENTATION";
    case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
      return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
    case VK_PIPELINE_COMPILE_REQUIRED:
      return "VK_PIPELINE_COMPILE_REQUIRED";
    case VK_ERROR_SURFACE_LOST_KHR:
      return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
      return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR:
      return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR:
      return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
      return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
    case VK_ERROR_VALIDATION_FAILED_EXT:
      return "VK_ERROR_VALIDATION_FAILED_EXT";
    case VK_ERROR_INVALID_SHADER_NV:
      return "VK_ERROR_INVALID_SHADER_NV";
    case VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR:
      return "VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR";
    case VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR:
      return "VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR";
    case VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR:
      return "VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR";
    case VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR:
      return "VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR";
    case VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR:
      return "VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR";
    case VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR:
      return "VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR";
    case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:
      return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
    case VK_ERROR_NOT_PERMITTED_KHR:
      return "VK_ERROR_NOT_PERMITTED_KHR";
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
      return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
    case VK_THREAD_IDLE_KHR:
      return "VK_THREAD_IDLE_KHR";
    case VK_THREAD_DONE_KHR:
      return "VK_THREAD_DONE_KHR";
    case VK_OPERATION_DEFERRED_KHR:
      return "VK_OPERATION_DEFERRED_KHR";
    case VK_OPERATION_NOT_DEFERRED_KHR:
      return "VK_OPERATION_NOT_DEFERRED_KHR";
    case VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR:
      return "VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR";
    case VK_ERROR_COMPRESSION_EXHAUSTED_EXT:
      return "VK_ERROR_COMPRESSION_EXHAUSTED_EXT";
    case VK_INCOMPATIBLE_SHADER_BINARY_EXT:
      return "VK_INCOMPATIBLE_SHADER_BINARY_EXT";
    default:
      return "Unknown VkResult value";
  }
}

namespace etx {

void create_vulkan_context(RHIContext& context, const RHIInitInfo& info) {
  static_assert(sizeof(VKContext) <= RHIContext::kBackendStorageSize, "VKContext does not fit into RHIContext backend storage");
  static_assert(alignof(VKContext) <= RHIContext::kBackendStorageAlignment, "VKContext alignment exceeds RHIContext backend storage alignment");
  auto* vk_context = new (context._backend_storage) VKContext(info);
  context.initialize_backend(vk_context, vk_context->get_device(), vk_context->get_bindless_manager());
}

static VkImageLayout rhi_state_to_vk_layout(RHIResourceState state, bool is_depth) {
  switch (state) {
    case RHIResourceState::ShaderReadOnly:
      return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case RHIResourceState::TransferSrc:
      return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case RHIResourceState::TransferDst:
      return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case RHIResourceState::General:
      return VK_IMAGE_LAYOUT_GENERAL;
    case RHIResourceState::ColorAttachment:
      return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case RHIResourceState::DepthStencilAttachment:
      return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    case RHIResourceState::Present:
      return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    default:
      return is_depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  }
}

constexpr size_t kMaxColorAttachments = 8;

struct VKContext::Impl {
  RHIInitInfo init_info = {};
  VKDevice device;
  VKBindlessManager bindless_manager = {};
  VKResourcePool<VKCommandBuffer, RHICommandBuffer> command_buffer_pool;
  uint32_t predefined_sampler_indices[4] = {};

  Impl(const RHIInitInfo& inf)
    : init_info(inf)
    , device(inf) {
    initialize_bindless_manager();
  }

  void initialize_bindless_manager() {
    if ((device.get_vk_device() != VK_NULL_HANDLE) && (bindless_manager.is_initialized() == false)) {
      bindless_manager.initialize(device.get_vk_device(), device.get_vk_physical_device());
      device.set_bindless_manager(&bindless_manager);

      initialize_predefined_samplers();
    }
  }

  void initialize_predefined_samplers() {
    {
      RHISamplerDesc desc = {};
      desc.min_filter = RHISamplerFilter::Linear;
      desc.mag_filter = RHISamplerFilter::Linear;
      desc.mipmap_mode = RHISamplerMipmapMode::Linear;
      desc.address_mode_u = RHISamplerAddressMode::Repeat;
      desc.address_mode_v = RHISamplerAddressMode::Repeat;
      desc.address_mode_w = RHISamplerAddressMode::Repeat;

      auto result = device.create_sampler(desc);
      if (result.result == RHIResult::Success) {
        predefined_sampler_indices[static_cast<size_t>(RHISamplerType::LinearRepeat)] = get_bindless_descriptor_index(result.handle);
      }
    }

    {
      RHISamplerDesc desc = {};
      desc.min_filter = RHISamplerFilter::Linear;
      desc.mag_filter = RHISamplerFilter::Linear;
      desc.mipmap_mode = RHISamplerMipmapMode::Linear;
      desc.address_mode_u = RHISamplerAddressMode::ClampToEdge;
      desc.address_mode_v = RHISamplerAddressMode::ClampToEdge;
      desc.address_mode_w = RHISamplerAddressMode::ClampToEdge;

      auto result = device.create_sampler(desc);
      if (result.result == RHIResult::Success) {
        predefined_sampler_indices[static_cast<size_t>(RHISamplerType::LinearClamp)] = (get_bindless_descriptor_index(result.handle));
      }
    }

    {
      RHISamplerDesc desc = {};
      desc.min_filter = RHISamplerFilter::Nearest;
      desc.mag_filter = RHISamplerFilter::Nearest;
      desc.mipmap_mode = RHISamplerMipmapMode::Nearest;
      desc.address_mode_u = RHISamplerAddressMode::Repeat;
      desc.address_mode_v = RHISamplerAddressMode::Repeat;
      desc.address_mode_w = RHISamplerAddressMode::Repeat;

      auto result = device.create_sampler(desc);
      if (result.result == RHIResult::Success) {
        predefined_sampler_indices[static_cast<size_t>(RHISamplerType::NearestRepeat)] = get_bindless_descriptor_index(result.handle);
      }
    }

    {
      RHISamplerDesc desc = {};
      desc.min_filter = RHISamplerFilter::Nearest;
      desc.mag_filter = RHISamplerFilter::Nearest;
      desc.mipmap_mode = RHISamplerMipmapMode::Nearest;
      desc.address_mode_u = RHISamplerAddressMode::ClampToEdge;
      desc.address_mode_v = RHISamplerAddressMode::ClampToEdge;
      desc.address_mode_w = RHISamplerAddressMode::ClampToEdge;

      auto result = device.create_sampler(desc);
      if (result.result == RHIResult::Success) {
        predefined_sampler_indices[static_cast<size_t>(RHISamplerType::NearestClamp)] = (get_bindless_descriptor_index(result.handle));
      }
    }
  }

  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  std::vector<VkImage> swapchain_images;
  std::vector<VkImageView> swapchain_image_views;
  std::vector<RHITexture> swapchain_textures;
  VkFormat swapchain_format = VK_FORMAT_UNDEFINED;
  VkExtent2D swapchain_extent = {0, 0};
  uint32_t current_swapchain_image = 0;

  std::vector<RHISemaphore> image_available_semaphores;
  std::vector<RHISemaphore> render_finished_semaphores;
  std::vector<VkFence> in_flight_fences;
  std::vector<VkFence> temporary_fences;
  uint32_t current_frame = 0;

  const void* native_window = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;

  bool create_surface();
  bool create_swapchain(uint32_t width, uint32_t height);
  void destroy_swapchain();
  void create_sync_objects();
  void destroy_sync_objects();

  bool register_swapchain_textures_with_bindless();
  void unregister_swapchain_textures_from_bindless();
  VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats);
  VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes);
  VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height);
};

VKContext::VKContext(const RHIInitInfo& info)
  : _impl(new Impl(info)) {
}

VKContext::VKContext(VKContext&& other) noexcept
  : _impl(other._impl) {
  other._impl = nullptr;
}

VKContext::~VKContext() {
  if (_impl == nullptr) {
    return;
  }

  if (_impl->device.get_vk_device() != VK_NULL_HANDLE) {
    if (_impl->in_flight_fences.empty() == false) {
      etx_vk_call(vkWaitForFences(_impl->device.get_vk_device(), static_cast<uint32_t>(_impl->in_flight_fences.size()), _impl->in_flight_fences.data(), VK_TRUE, UINT64_MAX));
    }

    if (_impl->temporary_fences.empty() == false) {
      etx_vk_call(vkWaitForFences(_impl->device.get_vk_device(), static_cast<uint32_t>(_impl->temporary_fences.size()), _impl->temporary_fences.data(), VK_TRUE, UINT64_MAX));

      for (VkFence fence : _impl->temporary_fences) {
        if (fence != VK_NULL_HANDLE) {
          vkDestroyFence(_impl->device.get_vk_device(), fence, nullptr);
        }
      }
      _impl->temporary_fences.clear();
    }

    etx_vk_call(vkDeviceWaitIdle(_impl->device.get_vk_device()));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  _impl->command_buffer_pool.clear();
  _impl->destroy_swapchain();
  _impl->device.destroy_all_resources();

  if (_impl->surface != VK_NULL_HANDLE && _impl->device.get_vk_instance() != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(_impl->device.get_vk_instance(), _impl->surface, nullptr);
  }

  delete _impl;
}

VKDevice* VKContext::get_device() {
  return &_impl->device;
}

VKBindlessManager* VKContext::get_bindless_manager() {
  return &_impl->bindless_manager;
}

void VKContext::initialize_for_headless() {
  _impl->initialize_bindless_manager();
}

void VKContext::create_swapchain(const void* native_window, uint32_t width, uint32_t height) {
  _impl->native_window = native_window;
  _impl->width = width;
  _impl->height = height;

  _impl->initialize_bindless_manager();

  if (_impl->create_surface() == false) {
    log::error("Failed to create Vulkan surface");
    return;
  }

  if (_impl->create_swapchain(width, height) == false) {
    log::error("Failed to create Vulkan swapchain");
    return;
  }

  _impl->create_sync_objects();
}

void VKContext::destroy_swapchain() {
  _impl->destroy_swapchain();
}

void VKContext::resize_swapchain(uint32_t width, uint32_t height) {
  if ((_impl->width == width) && (_impl->height == height)) {
    return;
  }

  vkDeviceWaitIdle(_impl->device.get_vk_device());

  _impl->destroy_swapchain();

  _impl->width = width;
  _impl->height = height;

  if (_impl->surface != VK_NULL_HANDLE) {
    if (!_impl->create_swapchain(width, height)) {
      log::error("Failed to recreate Vulkan swapchain");
      return;
    }

    _impl->create_sync_objects();
  }
}

RHITexture VKContext::get_current_swapchain_texture() {
  if (_impl->current_swapchain_image < _impl->swapchain_textures.size()) {
    RHITexture handle = _impl->swapchain_textures[_impl->current_swapchain_image];
    if (_impl->bindless_manager.is_valid_handle(handle)) {
      return handle;
    }
    log::error("Swapchain texture handle %llu is invalid or stale", handle.value);
  }

  return {};
}

RHITextureFormat VKContext::get_swapchain_format() const {
  return vk_format_to_rhi(_impl->swapchain_format);
}

void VKContext::present() {
  VkSemaphore wait_semaphore = _impl->device.get_vk_semaphore(_impl->render_finished_semaphores[_impl->current_frame]);
  VkPresentInfoKHR present_info = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &wait_semaphore;
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &_impl->swapchain;
  present_info.pImageIndices = &_impl->current_swapchain_image;

  VkResult result = vkQueuePresentKHR(_impl->device.get_graphics_queue(), &present_info);

  if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
    if (!_impl->in_flight_fences.empty()) {
      etx_vk_call(vkWaitForFences(_impl->device.get_vk_device(), static_cast<uint32_t>(_impl->in_flight_fences.size()), _impl->in_flight_fences.data(), VK_TRUE, UINT64_MAX));
    }

    _impl->destroy_swapchain();
    if (!_impl->create_swapchain(_impl->width, _impl->height)) {
      log::error("Failed to recreate swapchain after out-of-date condition");
      return;
    }
    _impl->create_sync_objects();
  }

  _impl->current_frame = (_impl->current_frame + 1) % kRHIMaxFrames;
}

RHIResult VKContext::wait_idle() {
  if (_impl->device.get_vk_device() == VK_NULL_HANDLE) {
    return RHIResult::InvalidHandle;
  }

  return etx_vk_call(vkDeviceWaitIdle(_impl->device.get_vk_device())) == VK_SUCCESS ? RHIResult::Success : RHIResult::ValidationError;
}

void VKContext::begin_frame() {
  if (_impl->swapchain == VK_NULL_HANDLE) {
    return;
  }

  if (etx_vk_call(vkWaitForFences(_impl->device.get_vk_device(), 1, &_impl->in_flight_fences[_impl->current_frame], VK_TRUE, UINT64_MAX)) != VK_SUCCESS) {
    return;
  }

  // Set current frame index for staging buffer allocations
  _impl->device.set_current_frame_index(_impl->current_frame);
  // After fence wait, it's safe to reset this frame's staging buffer region
  // GPU has finished reading from it in previous cycle
  _impl->device.reset_staging_buffer_for_frame(_impl->current_frame);

  etx_vk_call(vkResetCommandPool(_impl->device.get_vk_device(), _impl->device.get_vk_command_pool(_impl->current_frame), VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));

  _impl->command_buffer_pool.clear();

  VkResult result = etx_vk_call(vkAcquireNextImageKHR(_impl->device.get_vk_device(), _impl->swapchain, UINT64_MAX,
    _impl->device.get_vk_semaphore(_impl->image_available_semaphores[_impl->current_frame]), VK_NULL_HANDLE, &_impl->current_swapchain_image));

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    if (!_impl->in_flight_fences.empty()) {
      etx_vk_call(vkWaitForFences(_impl->device.get_vk_device(), static_cast<uint32_t>(_impl->in_flight_fences.size()), _impl->in_flight_fences.data(), VK_TRUE, UINT64_MAX));
    }

    _impl->destroy_swapchain();

    VkSurfaceCapabilitiesKHR capabilities;
    if (etx_vk_call(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_impl->device.get_vk_physical_device(), _impl->surface, &capabilities)) != VK_SUCCESS) {
      return;
    }

    VkExtent2D new_extent = _impl->choose_swap_extent(capabilities, _impl->width, _impl->height);

    _impl->width = new_extent.width;
    _impl->height = new_extent.height;

    if (_impl->create_swapchain(new_extent.width, new_extent.height)) {
      _impl->create_sync_objects();
    } else {
      log::error("Failed to recreate swapchain during acquire");
      return;
    }
  } else if (result != VK_SUCCESS) {
    return;
  }

  etx_vk_call(vkResetFences(_impl->device.get_vk_device(), 1, &_impl->in_flight_fences[_impl->current_frame]));
}

RHISemaphore VKContext::get_image_acquired_semaphore() {
  return _impl->image_available_semaphores[_impl->current_frame];
}

RHISemaphore VKContext::get_render_complete_semaphore() {
  return _impl->render_finished_semaphores[_impl->current_frame];
}

uint32_t VKContext::get_current_frame_index() const {
  return _impl->current_frame;
}

uint32_t VKContext::get_sampler_index(RHISamplerType type) const {
  size_t index = static_cast<size_t>(type);
  if (index < 4 && index < static_cast<size_t>(RHISamplerType::Count)) {
    return _impl->predefined_sampler_indices[index];
  }
  return 0;
}

RHICommandBuffer VKContext::get_command_buffer() {
  uint32_t index = _impl->command_buffer_pool.allocate_index();
  auto& cmd = _impl->command_buffer_pool.get_data(index);
  cmd.initialize(this, _impl->current_frame);

  Handle h = Handle::construct(0, index, _impl->command_buffer_pool.get_generation(index));
  _impl->command_buffer_pool.set_handle_to_index(h, index);
  return h;
}

void VKContext::destroy_command_buffer(RHICommandBuffer cmd) {
  uint32_t index = _impl->command_buffer_pool.get_index(cmd);
  if (index != UINT32_MAX) {
    _impl->command_buffer_pool.free_index(index);
    _impl->command_buffer_pool.remove_handle(cmd);
  }
}

void VKContext::submit_command_buffer(const RHISubmitInfo& info) {
  VKCommandBuffer* vk_cmd_buf = _impl->command_buffer_pool.get_data_ptr(info.command_buffer);
  if (vk_cmd_buf == nullptr) {
    log::error("Submit failed: Invalid command buffer handle %llu", info.command_buffer.value);
    return;
  }

  if (vk_cmd_buf->is_recording()) {
    log::error("Attempting to submit command buffer that is still recording - this is invalid");
    return;
  }

  if (vk_cmd_buf->is_submitted()) {
    log::error("Attempting to submit a command buffer that has already been submitted");
    return;
  }

  std::vector<VkSemaphore> wait_semaphores;
  std::vector<VkPipelineStageFlags> wait_stages;
  std::vector<VkSemaphore> signal_semaphores;

  for (auto s : info.wait_semaphores) {
    if (s.valid()) {
      VkSemaphore vk_sem = _impl->device.get_vk_semaphore(s);
      if (vk_sem != VK_NULL_HANDLE) {
        wait_semaphores.push_back(vk_sem);
        wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
      }
    }
  }

  VkFence submit_fence = VK_NULL_HANDLE;

  for (auto s : info.signal_semaphores) {
    if (s.valid()) {
      VkSemaphore vk_sem = _impl->device.get_vk_semaphore(s);
      if (vk_sem != VK_NULL_HANDLE) {
        signal_semaphores.push_back(vk_sem);
        // Hack: if we are signaling a semaphore that matches our internal render_finished semaphore,
        // we assume this is the frame end and we should attach the frame fence.
        if (s == _impl->render_finished_semaphores[_impl->current_frame]) {
          submit_fence = _impl->in_flight_fences[_impl->current_frame];
          etx_vk_call(vkResetFences(_impl->device.get_vk_device(), 1, &submit_fence));
        }
      }
    }
  }

  VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit_info.waitSemaphoreCount = static_cast<uint32_t>(wait_semaphores.size());
  submit_info.pWaitSemaphores = wait_semaphores.data();
  submit_info.pWaitDstStageMask = wait_stages.data();
  submit_info.commandBufferCount = 1;

  VkCommandBuffer vk_command_buffer = vk_cmd_buf->get_vk_command_buffer();
  if (vk_command_buffer == VK_NULL_HANDLE) {
    log::error("Invalid VkCommandBuffer");
    return;
  }
  submit_info.pCommandBuffers = &vk_command_buffer;
  submit_info.signalSemaphoreCount = static_cast<uint32_t>(signal_semaphores.size());
  submit_info.pSignalSemaphores = signal_semaphores.data();

  if (etx_vk_call(vkQueueSubmit(_impl->device.get_graphics_queue(), 1, &submit_info, submit_fence)) != VK_SUCCESS) {
    log::error("Failed to submit command buffer");
    return;
  }

  vk_cmd_buf->set_submitted(true);
}

void VKContext::program_command_buffer(RHICommandBuffer cmd_handle, std::function<void(void)> func) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd) {
    cmd->begin();
    func();
    cmd->end();
  }
}

void VKContext::command_buffer_begin(RHICommandBuffer cmd_handle) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->begin();
}

void VKContext::command_buffer_end(RHICommandBuffer cmd_handle) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->end();
}

void VKContext::command_buffer_reset(RHICommandBuffer cmd_handle) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->reset();
}

void VKContext::cmd_buffer_barrier(RHICommandBuffer cmd_handle, RHIBindlessHandle buffer, RHIResourceState old_state, RHIResourceState new_state) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->buffer_barrier(buffer, old_state, new_state);
}

void VKContext::cmd_texture_barrier(RHICommandBuffer cmd_handle, RHIBindlessHandle texture, RHIResourceState old_state, RHIResourceState new_state) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->texture_barrier(texture, old_state, new_state);
}

void VKContext::cmd_begin_render_pass(RHICommandBuffer cmd_handle, uint32_t color_attachment_count, RHIBindlessHandle* color_attachments, const float* clear_colors,
  RHIBindlessHandle depth_attachment, const RHIResourceState* color_final_states, RHIResourceState depth_final_state) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->begin_render_pass(color_attachment_count, color_attachments, clear_colors, depth_attachment, color_final_states, depth_final_state);
}

void VKContext::cmd_end_render_pass(RHICommandBuffer cmd_handle) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->end_render_pass();
}

void VKContext::cmd_set_viewport(RHICommandBuffer cmd_handle, const RHIViewport& viewport) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->set_viewport(viewport);
}

void VKContext::cmd_set_scissor(RHICommandBuffer cmd_handle, const RHIRect& scissor) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->set_scissor(scissor);
}

void VKContext::cmd_set_pipeline(RHICommandBuffer cmd_handle, RHIPipeline pipeline) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->set_pipeline(pipeline);
}

void VKContext::cmd_push_constants(RHICommandBuffer cmd_handle, const void* data, uint32_t size, uint32_t offset) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->push_constants(data, size, offset);
}

void VKContext::cmd_draw(RHICommandBuffer cmd_handle, const RHIDrawDesc& desc) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->draw(desc);
}

void VKContext::cmd_draw_indexed(RHICommandBuffer cmd_handle, const RHIIndexedDrawDesc& desc, RHIBindlessHandle index_buffer) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->draw_indexed(desc, index_buffer);
}

void VKContext::cmd_dispatch(RHICommandBuffer cmd_handle, const RHIDispatchDesc& desc) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->dispatch(desc);
}

void VKContext::cmd_build_acceleration_structure(RHICommandBuffer cmd_handle, const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer,
  uint64_t scratch_offset) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->build_acceleration_structure(desc, scratch_buffer, scratch_offset);
}

void VKContext::cmd_copy_buffer(RHICommandBuffer cmd_handle, RHIBindlessHandle src, RHIBindlessHandle dst, uint64_t size, uint64_t src_offset, uint64_t dst_offset) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->copy_buffer(src, dst, size, src_offset, dst_offset);
}

void VKContext::cmd_copy_buffer_to_texture(RHICommandBuffer cmd_handle, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->copy_buffer_to_texture(src, dst, width, height, mip_level);
}

void VKContext::cmd_copy_texture_to_buffer(RHICommandBuffer cmd_handle, RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->copy_texture_to_buffer(src, dst, width, height, mip_level);
}

void VKContext::cmd_set_debug_name(RHICommandBuffer cmd_handle, const char* name) {
  VKCommandBuffer* cmd = _impl->command_buffer_pool.get_data_ptr(cmd_handle);
  if (cmd)
    cmd->set_debug_name(name);
}

VkDevice VKContext::get_vk_device() const {
  return _impl->device.get_vk_device();
}

VkCommandPool VKContext::get_vk_command_pool(uint32_t index) const {
  return _impl->device.get_vk_command_pool(index);
}

VkFence VKContext::get_current_frame_fence() const {
  return _impl->in_flight_fences[_impl->current_frame];
}

VkQueue VKContext::get_graphics_queue() const {
  return _impl->device.get_graphics_queue();
}

VKCommandBuffer::VKCommandBuffer() {
}

void VKCommandBuffer::initialize(VKContext* ctx, uint32_t pool_index) {
  context = ctx;
  device = ctx->get_device();
  VkCommandBufferAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc_info.commandPool = ctx->get_vk_command_pool(pool_index);
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  etx_vk_call(vkAllocateCommandBuffers(ctx->get_vk_device(), &alloc_info, &command_buffer));
}

VKCommandBuffer::VKCommandBuffer(VKCommandBuffer&& other) noexcept {
  *this = std::move(other);
}

VKCommandBuffer& VKCommandBuffer::operator=(VKCommandBuffer&& other) noexcept {
  if (this != &other) {
    destroy_resources();
    context = other.context;
    device = other.device;
    command_buffer = other.command_buffer;
    _in_render_pass = other._in_render_pass;
    _is_recording = other._is_recording;
    _submitted = other._submitted;
    _render_pass_depth = other._render_pass_depth;

    current_color_attachments = std::move(other.current_color_attachments);
    current_color_final_states = std::move(other.current_color_final_states);
    current_depth_attachment = other.current_depth_attachment;
    current_depth_final_state = other.current_depth_final_state;
    current_pipeline = other.current_pipeline;
    current_bind_point = other.current_bind_point;
    current_pipeline_layout = other.current_pipeline_layout;

    other.command_buffer = VK_NULL_HANDLE;
    other._is_recording = false;
  }
  return *this;
}

void VKCommandBuffer::destroy_resources() {
}

bool VKCommandBuffer::is_initialized() const {
  return command_buffer != VK_NULL_HANDLE;
}

VkCommandBuffer VKCommandBuffer::get_vk_command_buffer() const {
  return command_buffer;
}

bool VKCommandBuffer::is_recording() const {
  return _is_recording;
}

bool VKCommandBuffer::is_submitted() const {
  return _submitted;
}

VKCommandBuffer::~VKCommandBuffer() {
  destroy_resources();
}

void VKCommandBuffer::reset() {
  _is_recording = false;
  _submitted = false;
  _render_pass_depth = 0;
  _in_render_pass = false;
  current_pipeline = {};
  current_bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
  current_pipeline_layout = VK_NULL_HANDLE;
}

void VKCommandBuffer::begin() {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Command buffer not initialized");
    return;
  }

  if (_is_recording) {
    log::error("Cannot begin command buffer: command buffer is already recording");
    return;
  }

  if (_submitted) {
    log::warning("Beginning a command buffer that was previously submitted - resetting submission state");
    _submitted = false;
  }

  if (etx_vk_call(vkResetCommandBuffer(command_buffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)) != VK_SUCCESS) {
    return;
  }

  VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  if (etx_vk_call(vkBeginCommandBuffer(command_buffer, &begin_info)) != VK_SUCCESS) {
    return;
  }

  _is_recording = true;
  _submitted = false;
  _in_render_pass = false;
  _render_pass_depth = 0;
  _in_render_pass = false;
  _render_pass_depth = 0;
}

void VKCommandBuffer::end() {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Command buffer not initialized");
    return;
  }

  if (!_is_recording) {
    log::error("Cannot end command buffer: command buffer is not recording");
    return;
  }

  if (_in_render_pass) {
    log::warning("Ending command buffer while still in render pass - auto-ending render pass");
    vkCmdEndRendering(command_buffer);
    _in_render_pass = false;
  }

  if (etx_vk_call(vkEndCommandBuffer(command_buffer)) == VK_SUCCESS) {
    _is_recording = false;
  }
}

void VKCommandBuffer::reset_internal_state() {
  current_pipeline = {};
  current_pipeline_layout = VK_NULL_HANDLE;
  _render_pass_depth = 0;
  _in_render_pass = false;
  _render_pass_depth = 0;
  _in_render_pass = false;
  _is_recording = false;
  _submitted = false;
}

void VKCommandBuffer::set_submitted(bool value) {
  _submitted = value;
}

void VKCommandBuffer::buffer_barrier(RHIBindlessHandle buffer, RHIResourceState old_state, RHIResourceState new_state) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot set buffer barrier: command buffer not initialized");
    return;
  }

  if (_is_recording == false) {
    log::error("Cannot set buffer barrier: command buffer is not recording");
    return;
  }

  VkBuffer vk_buffer = device->get_vk_buffer_from_bindless(buffer);
  if (vk_buffer == VK_NULL_HANDLE) {
    log::error("Invalid buffer handle for barrier");
    return;
  }

  VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = vk_buffer;
  barrier.offset = 0;
  barrier.size = VK_WHOLE_SIZE;

  VkAccessFlags src_access = 0, dst_access = 0;
  VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

  switch (old_state) {
    case RHIResourceState::General:
      src_access |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      break;
    case RHIResourceState::TransferSrc:
      src_access |= VK_ACCESS_TRANSFER_READ_BIT;
      src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      break;
    case RHIResourceState::TransferDst:
      src_access |= VK_ACCESS_TRANSFER_WRITE_BIT;
      src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      break;
    case RHIResourceState::ShaderReadOnly:
      src_access |= VK_ACCESS_SHADER_READ_BIT;
      src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      break;
    default:
      break;
  }

  switch (new_state) {
    case RHIResourceState::General:
      dst_access |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      break;
    case RHIResourceState::TransferSrc:
      dst_access |= VK_ACCESS_TRANSFER_READ_BIT;
      dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      break;
    case RHIResourceState::TransferDst:
      dst_access |= VK_ACCESS_TRANSFER_WRITE_BIT;
      dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      break;
    case RHIResourceState::ShaderReadOnly:
      dst_access |= VK_ACCESS_SHADER_READ_BIT;
      dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      break;
    default:
      break;
  }

  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;

  vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}
void VKCommandBuffer::ensure_texture_layout(RHIBindlessHandle texture, VkImageLayout required_layout) {
  if (context->is_swapchain_texture(texture)) {
    VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = required_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = context->get_bindless_manager()->get_vk_image(texture);
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    switch (required_layout) {
      case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        break;
      case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = 0;
        src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        break;
      case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        break;
      case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        break;
      default:
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = 0;
        break;
    }

    vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    return;
  }

  const VKTextureData* texture_data_ptr = device->get_texture_data(texture);
  if (texture_data_ptr == nullptr) {
    return;
  }
  const VKTextureData& texture_data = *texture_data_ptr;
  RHIResourceState current_state = texture_data.current_state;
  RHIResourceState target_state = RHIResourceState::Undefined;

  switch (required_layout) {
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      target_state = RHIResourceState::ShaderReadOnly;
      break;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      target_state = RHIResourceState::ColorAttachment;
      break;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      target_state = RHIResourceState::TransferSrc;
      break;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      target_state = RHIResourceState::TransferDst;
      break;
    case VK_IMAGE_LAYOUT_GENERAL:
      target_state = RHIResourceState::General;
      break;
    default:
      target_state = RHIResourceState::Undefined;
      break;
  }

  if (target_state != RHIResourceState::Undefined && current_state != target_state) {
    texture_barrier(texture, current_state, target_state);
  }
}

void VKCommandBuffer::texture_barrier(RHIBindlessHandle texture, RHIResourceState old_state, RHIResourceState new_state) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot set texture barrier: command buffer not initialized");
    return;
  }

  if (_is_recording == false) {
    log::error("Cannot set texture barrier: command buffer is not recording");
    return;
  }

  VkImage vk_image = device->get_vk_image_from_bindless(texture);
  if (vk_image == VK_NULL_HANDLE) {
    log::error("Invalid texture handle for barrier operation");
    return;
  }

  VKTextureData* texture_data_ptr = device->get_texture_data(texture);
  if (texture_data_ptr == nullptr) {
    log::error("Texture handle not found in texture map");
    return;
  }
  VKTextureData& texture_data = *texture_data_ptr;

  VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = vk_image;

  VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkAccessFlags src_access = 0;
  VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

  switch (old_state) {
    case RHIResourceState::ColorAttachment:
      src_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      src_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      break;
    case RHIResourceState::ShaderReadOnly:
      src_access = VK_ACCESS_SHADER_READ_BIT;
      src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      break;
    case RHIResourceState::TransferSrc:
      src_access = VK_ACCESS_TRANSFER_READ_BIT;
      src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      break;
    case RHIResourceState::TransferDst:
      src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
      src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      break;
    case RHIResourceState::General:
      src_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      break;
    default:
      break;
  }

  VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkAccessFlags dst_access = 0;
  VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

  switch (new_state) {
    case RHIResourceState::Present:
      new_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      dst_access = 0;
      dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
      break;
    case RHIResourceState::ShaderReadOnly:
      new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      dst_access = VK_ACCESS_SHADER_READ_BIT;
      dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      break;
    case RHIResourceState::ColorAttachment:
      new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      dst_access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      dst_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      break;
    case RHIResourceState::TransferSrc:
      new_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      dst_access = VK_ACCESS_TRANSFER_READ_BIT;
      dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      break;
    case RHIResourceState::TransferDst:
      new_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      dst_access = VK_ACCESS_TRANSFER_WRITE_BIT;
      dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      break;
    case RHIResourceState::General:
      new_layout = VK_IMAGE_LAYOUT_GENERAL;
      dst_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      break;
    default:
      new_layout = VK_IMAGE_LAYOUT_GENERAL;
      break;
  }

  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  if (texture_data.desc.format == RHITextureFormat::D32_FLOAT || texture_data.desc.format == RHITextureFormat::D24_UNORM_S8_UINT ||
      texture_data.desc.format == RHITextureFormat::D32_FLOAT_S8_UINT) {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (texture_data.desc.format == RHITextureFormat::D24_UNORM_S8_UINT || texture_data.desc.format == RHITextureFormat::D32_FLOAT_S8_UINT) {
      barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  }

  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;

  vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  texture_data.current_state = new_state;
}

void VKCommandBuffer::begin_render_pass(uint32_t color_attachment_count, RHIBindlessHandle* color_attachments, const float* clear_colors, RHIBindlessHandle depth_attachment,
  const RHIResourceState* color_final_states, RHIResourceState depth_final_state) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot begin render pass: Vulkan command buffer handle is invalid");
    return;
  }

  if (!_is_recording) {
    log::error("Cannot begin render pass: command buffer is not recording");
    return;
  }

  if (((color_attachment_count == 0) || (color_attachments == nullptr)) && (depth_attachment.valid() == false)) {
    log::error("No attachments provided to begin_render_pass");
    return;
  }

  if (_render_pass_depth > 0) {
    log::error("Cannot begin render pass: already in render pass (depth: %u). Call end_render_pass() first", _render_pass_depth);
    return;
  }

  constexpr uint32_t MAX_ATTACHMENTS = 8;
  if (color_attachment_count > MAX_ATTACHMENTS) {
    log::error("Too many color attachments: %u (max %u)", color_attachment_count, MAX_ATTACHMENTS);
    return;
  }

  VkRenderingAttachmentInfo color_attachments_info[MAX_ATTACHMENTS] = {};
  VkRenderingAttachmentInfo depth_attachment_info = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};

  VkRect2D render_area = {};
  bool render_area_set = false;
  _rendering_to_swapchain = false;

  for (uint32_t i = 0; i < color_attachment_count; ++i) {
    bool is_swapchain_texture = false;
    VkImageView image_view = VK_NULL_HANDLE;

    if (context->is_swapchain_texture(color_attachments[i])) {
      is_swapchain_texture = true;
      image_view = context->get_swapchain_image_view(color_attachments[i]);
      if (!render_area_set) {
        render_area.extent = context->get_swapchain_extent();
        render_area_set = true;
      }
      _rendering_to_swapchain = true;
    }

    if (!is_swapchain_texture) {
      const VKTextureData* texture_data_ptr = device->get_texture_data(color_attachments[i]);
      if (texture_data_ptr == nullptr) {
        log::error("Color attachment %u not found in texture map", i);
        return;
      }
      const VKTextureData& texture_data = *texture_data_ptr;
      image_view = texture_data.image_view;
      if (!render_area_set) {
        render_area.extent = {texture_data.desc.width, texture_data.desc.height};
        render_area_set = true;
      }
    }

    VkAttachmentLoadOp load_op = VK_ATTACHMENT_LOAD_OP_LOAD;
    VkClearValue clear_value = {};
    if (clear_colors) {
      load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
      clear_value.color = {clear_colors[i * 4], clear_colors[i * 4 + 1], clear_colors[i * 4 + 2], clear_colors[i * 4 + 3]};
    }

    color_attachments_info[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachments_info[i].imageView = image_view;
    color_attachments_info[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachments_info[i].loadOp = load_op;
    color_attachments_info[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachments_info[i].clearValue = clear_value;

    ensure_texture_layout(color_attachments[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  }

  bool has_depth = depth_attachment.valid();
  if (has_depth) {
    const VKTextureData* texture_data_ptr = device->get_texture_data(depth_attachment);
    if (texture_data_ptr == nullptr) {
      log::error("Depth attachment not found in texture map");
      return;
    }
    const VKTextureData& texture_data = *texture_data_ptr;

    VkClearValue clear_value = {};
    clear_value.depthStencil = {1.0f, 0};

    depth_attachment_info.imageView = texture_data.image_view;
    depth_attachment_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment_info.clearValue = clear_value;

    ensure_texture_layout(depth_attachment, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  }

  VkRenderingInfo rendering_info = {VK_STRUCTURE_TYPE_RENDERING_INFO};
  rendering_info.renderArea = render_area;
  rendering_info.layerCount = 1;
  rendering_info.colorAttachmentCount = color_attachment_count;
  rendering_info.pColorAttachments = color_attachments_info;
  rendering_info.pDepthAttachment = has_depth ? &depth_attachment_info : nullptr;
  rendering_info.pStencilAttachment = nullptr;  // Stencil not supported/used in this path yet

  vkCmdBeginRendering(command_buffer, &rendering_info);

  _render_pass_depth++;
  _in_render_pass = (_render_pass_depth > 0);

  current_color_attachments.assign(color_attachments, color_attachments + color_attachment_count);
  current_color_final_states.clear();
  current_color_final_states.reserve(color_attachment_count);
  for (uint32_t i = 0; i < color_attachment_count; ++i) {
    if (color_final_states != nullptr) {
      current_color_final_states.push_back(color_final_states[i]);
    } else {
      current_color_final_states.push_back(RHIResourceState::ColorAttachment);
    }
  }
  current_depth_attachment = depth_attachment;
  current_depth_final_state = has_depth ? depth_final_state : RHIResourceState::Undefined;
}

void VKCommandBuffer::end_render_pass() {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot end render pass: command buffer not initialized");
    return;
  }

  if (!_is_recording) {
    log::error("Cannot end render pass: command buffer is not recording");
    return;
  }

  if (_render_pass_depth == 0) {
    log::error("Cannot end render pass: not currently in a render pass");
    return;
  }

  vkCmdEndRendering(command_buffer);

  if (device != nullptr) {
    for (size_t i = 0; i < current_color_attachments.size(); ++i) {
      RHITexture tex = current_color_attachments[i];
      if (context->is_swapchain_texture(tex)) {
        continue;
      }

      VKTextureData* texture_data_ptr = device->get_texture_data(tex);
      if (texture_data_ptr != nullptr) {
        texture_data_ptr->current_state = current_color_final_states[i];
      }
    }

    if (current_depth_attachment.valid()) {
      VKTextureData* texture_data_ptr = device->get_texture_data(current_depth_attachment);
      if (texture_data_ptr != nullptr) {
        texture_data_ptr->current_state = current_depth_final_state;
      }
    }
  }

  if ((context != nullptr) && _rendering_to_swapchain) {
    RHITexture current_texture = context->get_current_swapchain_texture();
    if (current_texture.valid()) {
      ensure_texture_layout(current_texture, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    }
  }
  _render_pass_depth--;
  _rendering_to_swapchain = false;
  _in_render_pass = (_render_pass_depth > 0);
  current_color_attachments.clear();
  current_color_final_states.clear();
  current_depth_attachment = {};
  current_depth_final_state = RHIResourceState::Undefined;
}

void VKCommandBuffer::set_viewport(const RHIViewport& viewport) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot set viewport: command buffer not initialized");
    return;
  }

  if (!_is_recording) {
    log::error("Cannot setviewport: command buffer is not recording");
    return;
  }

  VkViewport vk_viewport = {};
  vk_viewport.x = viewport.x;
  vk_viewport.y = viewport.y;
  vk_viewport.width = viewport.width;
  vk_viewport.height = viewport.height;
  vk_viewport.minDepth = viewport.min_depth;
  vk_viewport.maxDepth = viewport.max_depth;

  vkCmdSetViewport(command_buffer, 0, 1, &vk_viewport);

  set_scissor_from_viewport(viewport);
}

void VKCommandBuffer::set_scissor(const RHIRect& scissor) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot set scissor: command buffer not initialized");
    return;
  }

  if (!_is_recording) {
    log::error("Cannot set scissor: command buffer is not recording");
    return;
  }

  VkRect2D vk_scissor = {};
  vk_scissor.offset.x = scissor.x;
  vk_scissor.offset.y = scissor.y;
  vk_scissor.extent.width = scissor.width;
  vk_scissor.extent.height = scissor.height;

  vkCmdSetScissor(command_buffer, 0, 1, &vk_scissor);
}
void VKCommandBuffer::set_scissor_from_viewport(const RHIViewport& viewport) {
  RHIRect scissor = {};
  scissor.x = static_cast<int32_t>(viewport.x);
  scissor.y = static_cast<int32_t>(viewport.y);
  scissor.width = static_cast<uint32_t>(viewport.width);
  scissor.height = static_cast<uint32_t>(viewport.height);
  set_scissor(scissor);
}

void VKCommandBuffer::set_pipeline(RHIPipeline pipeline) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot set pipeline: command buffer not initialized");
    return;
  }

  if (!_is_recording) {
    log::error("Cannot set pipeline: command buffer is not recording");
    return;
  }

  current_pipeline = pipeline;

  const VKPipelineData* graphics_pipeline_data = device->get_graphics_pipeline_data(pipeline);
  if (graphics_pipeline_data != nullptr) {
    current_pipeline_layout = device->get_bindless_pipeline_layout();
    current_bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    vkCmdBindPipeline(command_buffer, current_bind_point, graphics_pipeline_data->pipeline);
    VkDescriptorSet bindless_set = context->get_bindless_manager()->get_descriptor_set();
    if (bindless_set != VK_NULL_HANDLE) {
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, current_pipeline_layout, 0, 1, &bindless_set, 0, nullptr);
    }
    return;
  }

  const VKPipelineData* compute_pipeline_data = device->get_compute_pipeline_data(pipeline);
  if (compute_pipeline_data != nullptr) {
    current_pipeline_layout = device->get_bindless_pipeline_layout();
    current_bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
    vkCmdBindPipeline(command_buffer, current_bind_point, compute_pipeline_data->pipeline);
    return;
  }

  log::error("Pipeline not found: %llu", pipeline.value);
}

void VKCommandBuffer::push_constants(const void* data, uint32_t size, uint32_t offset) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot push constants: command buffer not initialized");
    return;
  }

  if (!_is_recording) {
    log::error("Cannot push constants: command buffer is not recording");
    return;
  }

  if (current_pipeline_layout == VK_NULL_HANDLE) {
    log::error("Cannot push constants: no pipeline bound");
    return;
  }

  uint64_t end_offset = static_cast<uint64_t>(offset) + static_cast<uint64_t>(size);
  if (end_offset > kVKMaxPushConstantsSize) {
    log::error("Push constants size exceeds layout range: size=%u offset=%u", size, offset);
    return;
  }

  vkCmdPushConstants(command_buffer, current_pipeline_layout, VkShaderStageFlags(VK_SHADER_STAGE_ALL), offset, size, data);
}

void VKCommandBuffer::draw_indexed(const RHIIndexedDrawDesc& desc, RHIBindlessHandle index_buffer) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot draw indexed: command buffer not initialized");
    return;
  }

  if (current_pipeline.invalid()) {
    log::error("No pipeline set for draw_indexed");
    return;
  }

  const VKPipelineData* pipeline_data = device->get_graphics_pipeline_data(current_pipeline);
  if (pipeline_data == nullptr) {
    log::error("Failed to find graphics pipeline: %llu", current_pipeline.value);
    return;
  }

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_data->pipeline);

  VkBuffer vk_index_buffer = device->get_vk_buffer_from_bindless(index_buffer);
  if (vk_index_buffer == VK_NULL_HANDLE) {
    log::error("Invalid index buffer handle");
    return;
  }

  VkIndexType vk_index_type = (desc.index_type == RHIIndexType::UInt16) ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
  vkCmdBindIndexBuffer(command_buffer, vk_index_buffer, 0, vk_index_type);

  vkCmdDrawIndexed(command_buffer, desc.index_count, desc.instance_count, desc.first_index, desc.vertex_offset, desc.first_instance);
}

void VKCommandBuffer::draw(const RHIDrawDesc& desc) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot draw: command buffer not initialized");
    return;
  }

  if (current_pipeline.invalid()) {
    log::error("No pipeline set for draw");
    return;
  }

  const VKPipelineData* pipeline_data = device->get_graphics_pipeline_data(current_pipeline);
  if (pipeline_data == nullptr) {
    log::error("Failed to find graphics pipeline: %llu", current_pipeline.value);
    return;
  }

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_data->pipeline);

  vkCmdDraw(command_buffer, desc.vertex_count, desc.instance_count, desc.first_vertex, desc.first_instance);
}

void VKCommandBuffer::dispatch(const RHIDispatchDesc& desc) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot dispatch compute: command buffer not initialized");
    return;
  }

  if (!_is_recording) {
    log::error("Cannot dispatch compute: command buffer is not recording");
    return;
  }

  if (current_pipeline.invalid()) {
    log::error("No pipeline set for compute dispatch");
    return;
  }

  const VKPipelineData* compute_pipeline_data = device->get_compute_pipeline_data(current_pipeline);
  if (compute_pipeline_data == nullptr) {
    log::error("Failed to find compute pipeline: %llu", current_pipeline.value);
    return;
  }

  VkPipeline vk_pipeline = compute_pipeline_data->pipeline;
  if (vk_pipeline == VK_NULL_HANDLE) {
    log::error("Invalid Vulkan pipeline or pipeline layout");
    return;
  }

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline);

  VkDescriptorSet bindless_set = context->get_bindless_manager()->get_descriptor_set();
  if (bindless_set != VK_NULL_HANDLE) {
    auto vk_pipeline_layout = device->get_bindless_pipeline_layout();
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, vk_pipeline_layout, 0, 1, &bindless_set, 0, nullptr);
  }

  vkCmdDispatch(command_buffer, desc.group_count_x, desc.group_count_y, desc.group_count_z);
}

void VKCommandBuffer::copy_buffer(RHIBindlessHandle src, RHIBindlessHandle dst, uint64_t size, uint64_t src_offset, uint64_t dst_offset) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot copy buffer: command buffer not initialized");
    return;
  }

  VkBuffer vk_src_buffer = device->get_vk_buffer_from_bindless(src);
  VkBuffer vk_dst_buffer = device->get_vk_buffer_from_bindless(dst);

  if (vk_src_buffer == VK_NULL_HANDLE || vk_dst_buffer == VK_NULL_HANDLE) {
    log::error("Invalid buffer handle(s) for copy operation");
    return;
  }

  VkBufferCopy copy_region = {};
  copy_region.srcOffset = src_offset;
  copy_region.dstOffset = dst_offset;
  copy_region.size = size;

  vkCmdCopyBuffer(command_buffer, vk_src_buffer, vk_dst_buffer, 1, &copy_region);
}

void VKCommandBuffer::copy_buffer_to_texture(RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot copy buffer to texture: command buffer not initialized");
    return;
  }

  VkBuffer vk_src_buffer = device->get_vk_buffer_from_bindless(src);
  if (vk_src_buffer == VK_NULL_HANDLE) {
    log::error("Invalid source buffer handle for copy operation");
    return;
  }

  VkImage vk_dst_image = device->get_vk_image_from_bindless(dst);
  if (vk_dst_image == VK_NULL_HANDLE) {
    log::error("Invalid destination texture handle for copy operation");
    return;
  }

  ensure_texture_layout(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  VkBufferImageCopy copy_region = {};
  copy_region.bufferOffset = 0;
  copy_region.bufferRowLength = 0;
  copy_region.bufferImageHeight = 0;
  copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.imageSubresource.mipLevel = mip_level;
  copy_region.imageSubresource.baseArrayLayer = 0;
  copy_region.imageSubresource.layerCount = 1;
  copy_region.imageOffset = {0, 0, 0};
  copy_region.imageExtent = {width, height, 1};

  vkCmdCopyBufferToImage(command_buffer, vk_src_buffer, vk_dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);
}

void VKCommandBuffer::copy_texture_to_buffer(RHIBindlessHandle src, RHIBindlessHandle dst, uint32_t width, uint32_t height, uint32_t mip_level) {
  if (command_buffer == VK_NULL_HANDLE) {
    log::error("Cannot copy texture to buffer: command buffer not initialized");
    return;
  }

  VkImage vk_src_image = device->get_vk_image_from_bindless(src);
  if (vk_src_image == VK_NULL_HANDLE) {
    log::error("Invalid source texture handle for copy operation");
    return;
  }

  VkBuffer vk_dst_buffer = device->get_vk_buffer_from_bindless(dst);
  if (vk_dst_buffer == VK_NULL_HANDLE) {
    log::error("Invalid destination buffer handle for copy operation");
    return;
  }

  ensure_texture_layout(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

  VkBufferImageCopy copy_region = {};
  copy_region.bufferOffset = 0;
  copy_region.bufferRowLength = 0;
  copy_region.bufferImageHeight = 0;
  copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.imageSubresource.mipLevel = mip_level;
  copy_region.imageSubresource.baseArrayLayer = 0;
  copy_region.imageSubresource.layerCount = 1;
  copy_region.imageOffset = {0, 0, 0};
  copy_region.imageExtent = {width, height, 1};

  vkCmdCopyImageToBuffer(command_buffer, vk_src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk_dst_buffer, 1, &copy_region);
}

void VKCommandBuffer::set_debug_name(const char* name) {
  if (command_buffer == VK_NULL_HANDLE) {
    return;
  }

  VkDevice vk_device = context->get_vk_device();
  if (auto vkSetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(vk_device, "vkSetDebugUtilsObjectNameEXT")) {
    VkDebugUtilsObjectNameInfoEXT name_info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    name_info.objectType = VK_OBJECT_TYPE_COMMAND_BUFFER;
    name_info.objectHandle = (uint64_t)command_buffer;
    name_info.pObjectName = name;

    etx_vk_call(vkSetDebugUtilsObjectNameEXT(vk_device, &name_info));
  }
}
bool VKContext::Impl::create_surface() {
  if (native_window == nullptr) {
    log::error("Native window is null");
    return false;
  }

#ifdef _WIN32
  VkWin32SurfaceCreateInfoKHR surface_info = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
  surface_info.hinstance = GetModuleHandle(nullptr);
  surface_info.hwnd = HWND(native_window);

  auto vkCreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(device.get_vk_instance(), "vkCreateWin32SurfaceKHR");
  if (!vkCreateWin32SurfaceKHR) {
    log::error("Failed to get vkCreateWin32SurfaceKHR function pointer");
    return false;
  }

  if (etx_vk_call(vkCreateWin32SurfaceKHR(device.get_vk_instance(), &surface_info, nullptr, &surface)) != VK_SUCCESS) {
    return false;
  }
#else

  log::error("Platform not supported for surface creation");
  return false;
#endif

  return true;
}

bool VKContext::Impl::create_swapchain(uint32_t width, uint32_t height) {
  VkPhysicalDevice physical_device = device.get_vk_physical_device();
  if (physical_device == VK_NULL_HANDLE) {
    log::error("Device not initialized");
    return false;
  }

  VkSurfaceCapabilitiesKHR capabilities;
  if (etx_vk_call(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities)) != VK_SUCCESS) {
    return false;
  }

  uint32_t format_count;
  if (etx_vk_call(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr)) != VK_SUCCESS) {
    return false;
  }
  std::vector<VkSurfaceFormatKHR> formats(format_count);
  if (etx_vk_call(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, formats.data())) != VK_SUCCESS) {
    return false;
  }

  uint32_t present_mode_count;
  if (etx_vk_call(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, nullptr)) != VK_SUCCESS) {
    return false;
  }
  std::vector<VkPresentModeKHR> present_modes(present_mode_count);
  if (etx_vk_call(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, present_modes.data())) != VK_SUCCESS) {
    return false;
  }

  VkSurfaceFormatKHR surface_format = choose_swap_surface_format(formats);
  VkPresentModeKHR present_mode = choose_swap_present_mode(present_modes);
  VkExtent2D extent = choose_swap_extent(capabilities, width, height);

  uint32_t image_count = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
    image_count = capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR swapchain_info = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  swapchain_info.surface = surface;
  swapchain_info.minImageCount = image_count;
  swapchain_info.imageFormat = surface_format.format;
  swapchain_info.imageColorSpace = surface_format.colorSpace;
  swapchain_info.imageExtent = extent;
  swapchain_info.imageArrayLayers = 1;
  swapchain_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  swapchain_info.queueFamilyIndexCount = 0;
  swapchain_info.pQueueFamilyIndices = nullptr;
  swapchain_info.preTransform = capabilities.currentTransform;
  swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchain_info.presentMode = present_mode;
  swapchain_info.clipped = VK_TRUE;
  swapchain_info.oldSwapchain = VK_NULL_HANDLE;

  if (etx_vk_call(vkCreateSwapchainKHR(device.get_vk_device(), &swapchain_info, nullptr, &swapchain)) != VK_SUCCESS) {
    return false;
  }

  swapchain_format = surface_format.format;

  etx_vk_call(vkGetSwapchainImagesKHR(device.get_vk_device(), swapchain, &image_count, nullptr));
  swapchain_images.resize(image_count);
  etx_vk_call(vkGetSwapchainImagesKHR(device.get_vk_device(), swapchain, &image_count, swapchain_images.data()));

  swapchain_image_views.resize(image_count);
  for (uint32_t i = 0; i < image_count; i++) {
    VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = swapchain_images[i];
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = surface_format.format;
    view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    if (etx_vk_call(vkCreateImageView(device.get_vk_device(), &view_info, nullptr, &swapchain_image_views[i])) != VK_SUCCESS) {
      return false;
    }
  }

  if (!register_swapchain_textures_with_bindless()) {
    log::error("Failed to register swapchain textures with bindless manager");
    return false;
  }

  swapchain_format = surface_format.format;
  swapchain_extent = extent;

  swapchain_format = surface_format.format;
  swapchain_extent = extent;

  return true;
}

void VKContext::Impl::create_sync_objects() {
  VkSemaphoreCreateInfo semaphore_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

  VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  image_available_semaphores.resize(kRHIMaxFrames);
  render_finished_semaphores.resize(kRHIMaxFrames);
  in_flight_fences.resize(kRHIMaxFrames);

  for (size_t i = 0; i < kRHIMaxFrames; i++) {
    image_available_semaphores[i] = device.create_semaphore().handle;
    render_finished_semaphores[i] = device.create_semaphore().handle;
    etx_vk_call(vkCreateFence(device.get_vk_device(), &fence_info, nullptr, &in_flight_fences[i]));
  }
}

void VKContext::Impl::destroy_sync_objects() {
  uint32_t semaphore_count = static_cast<uint32_t>(image_available_semaphores.size() + render_finished_semaphores.size());
  uint32_t fence_count = static_cast<uint32_t>(in_flight_fences.size());

  for (auto semaphore : image_available_semaphores) {
    if (semaphore.valid()) {
      device.destroy_semaphore(semaphore);
    }
  }
  for (auto semaphore : render_finished_semaphores) {
    if (semaphore.valid()) {
      device.destroy_semaphore(semaphore);
    }
  }
  for (auto fence : in_flight_fences) {
    if (fence != VK_NULL_HANDLE) {
      vkDestroyFence(device.get_vk_device(), fence, nullptr);
    }
  }

  image_available_semaphores.clear();
  render_finished_semaphores.clear();
  in_flight_fences.clear();
}

void VKContext::Impl::destroy_swapchain() {
  if (device.get_vk_device() == VK_NULL_HANDLE) {
    return;
  }

  uint32_t swapchain_texture_count = static_cast<uint32_t>(swapchain_textures.size());
  uint32_t swapchain_image_view_count = static_cast<uint32_t>(swapchain_image_views.size());

  unregister_swapchain_textures_from_bindless();

  for (auto image_view : swapchain_image_views) {
    if (image_view != VK_NULL_HANDLE) {
      vkDestroyImageView(device.get_vk_device(), image_view, nullptr);
    }
  }

  destroy_sync_objects();

  if (swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device.get_vk_device(), swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
  }

  swapchain_images.clear();
  swapchain_image_views.clear();
  swapchain_textures.clear();
  swapchain_format = VK_FORMAT_UNDEFINED;
  swapchain_extent = {0, 0};
  current_swapchain_image = 0;

  current_swapchain_image = 0;
}

VkSurfaceFormatKHR VKContext::Impl::choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats) {
  /*
  for (const auto& format : available_formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }
  */
  return available_formats[0];
}

VkPresentModeKHR VKContext::Impl::choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes) {
  for (const auto& present_mode : available_present_modes) {
    if (present_mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
      return present_mode;
    if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return present_mode;
    }
  }

  return VK_PRESENT_MODE_FIFO_KHR;
}

bool VKContext::Impl::register_swapchain_textures_with_bindless() {
  if (swapchain_images.empty() || swapchain_image_views.empty()) {
    log::error("Cannot register swapchain textures: swapchain images/views not created");
    return false;
  }

  if (swapchain_images.size() != swapchain_image_views.size()) {
    log::error("Swapchain images and image views count mismatch: %zu vs %zu", swapchain_images.size(), swapchain_image_views.size());
    return false;
  }

  if (swapchain_textures.size() != swapchain_images.size()) {
    swapchain_textures.resize(swapchain_images.size());
  }

  for (size_t i = 0; i < swapchain_images.size(); ++i) {
    RHITextureDesc desc = {};
    desc.width = swapchain_extent.width;
    desc.height = swapchain_extent.height;
    desc.format = vk_format_to_rhi(swapchain_format);
    desc.usage = RHITextureUsage::ColorAttachment;

    RHIBindlessHandle texture_handle = {};
    RHIResult rhi_result =
      bindless_manager.register_texture(swapchain_image_views[i], RHIResourceType::Texture, texture_handle, static_cast<uint32_t>(desc.usage), swapchain_images[i]);
    if (rhi_result != RHIResult::Success) {
      log::error("Failed to register swapchain texture %zu with bindless manager: %d", i, static_cast<int>(rhi_result));
      return false;
    }

    if (!bindless_manager.is_valid_handle(texture_handle)) {
      log::error("Registered swapchain texture %zu has invalid handle %llu", i, texture_handle.value);
      return false;
    }

    swapchain_textures[i] = texture_handle;
  }

  return true;
}

void VKContext::Impl::unregister_swapchain_textures_from_bindless() {
  if (swapchain_textures.empty()) {
    return;
  }

  for (size_t i = 0; i < swapchain_textures.size(); ++i) {
    auto texture = swapchain_textures[i];
    if (texture.valid()) {
      RHIResult unregister_result = bindless_manager.unregister_texture(texture);
      if (unregister_result != RHIResult::Success) {
        log::warning("Failed to unregister swapchain texture %zu (handle %llu): %d", i, texture.value, static_cast<int>(unregister_result));
      } else {
      }
    }
  }
}

VkExtent2D VKContext::Impl::choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    return capabilities.currentExtent;
  } else {
    VkExtent2D actual_extent = {width, height};

    actual_extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
    actual_extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual_extent.height));

    return actual_extent;
  }
}

bool VKContext::is_swapchain_texture(RHIBindlessHandle handle) const {
  for (const auto& texture : _impl->swapchain_textures) {
    if (texture == handle) {
      return true;
    }
  }
  return false;
}

VkImageView VKContext::get_swapchain_image_view(RHIBindlessHandle handle) const {
  for (size_t i = 0; i < _impl->swapchain_textures.size(); ++i) {
    if (_impl->swapchain_textures[i] == handle) {
      return _impl->swapchain_image_views[i];
    }
  }
  return VK_NULL_HANDLE;
}

VkExtent2D VKContext::get_swapchain_extent() const {
  return _impl->swapchain_extent;
}

void VKCommandBuffer::build_acceleration_structure(const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset) {
  const VKAccelerationStructureData* as_data_ptr = device->get_acceleration_structure_data(desc.as_handle);
  if (as_data_ptr == nullptr) {
    return;
  }

  const VKAccelerationStructureData& as_data = *as_data_ptr;

  VkAccelerationStructureBuildGeometryInfoKHR build_info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  build_info.type = (desc.type == RHIAccelerationStructureType::BottomLevel) ? VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR : VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  build_info.dstAccelerationStructure = as_data.acceleration_structure;
  build_info.scratchData.deviceAddress = device->get_buffer_device_address(scratch_buffer) + scratch_offset;

  std::vector<VkAccelerationStructureGeometryKHR> geometries;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
  std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> p_ranges;

  if (desc.type == RHIAccelerationStructureType::BottomLevel) {
    geometries.resize(desc.geometry_count);
    ranges.resize(desc.geometry_count);
    p_ranges.resize(desc.geometry_count);

    for (uint32_t i = 0; i < desc.geometry_count; ++i) {
      const auto& src_geo = desc.geometries[i];
      auto& vk_geo = geometries[i];
      vk_geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
      vk_geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      vk_geo.flags = src_geo.is_opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
      vk_geo.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;

      switch (src_geo.triangles.vertex_format) {
        case RHIVertexFormat::Float2:
          vk_geo.geometry.triangles.vertexFormat = VK_FORMAT_R32G32_SFLOAT;
          break;
        case RHIVertexFormat::Float3:
          vk_geo.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
          break;
        case RHIVertexFormat::Float4:
          vk_geo.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
          break;
        default:
          vk_geo.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
          break;
      }

      vk_geo.geometry.triangles.vertexData.deviceAddress = device->get_buffer_device_address(src_geo.triangles.vertex_buffer);
      vk_geo.geometry.triangles.vertexStride = src_geo.triangles.vertex_stride;
      vk_geo.geometry.triangles.maxVertex = src_geo.triangles.vertex_count;
      vk_geo.geometry.triangles.indexType = (src_geo.triangles.index_type == RHIIndexType::UInt32) ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
      vk_geo.geometry.triangles.indexData.deviceAddress = device->get_buffer_device_address(src_geo.triangles.index_buffer);

      ranges[i].primitiveCount = src_geo.triangles.index_count / 3;
      ranges[i].primitiveOffset = 0;
      ranges[i].firstVertex = 0;
      ranges[i].transformOffset = 0;
      p_ranges[i] = &ranges[i];
    }
  } else {
    geometries.resize(1);
    auto& vk_geo = geometries[0];
    vk_geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    vk_geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    vk_geo.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    vk_geo.geometry.instances.arrayOfPointers = VK_FALSE;
    vk_geo.geometry.instances.data.deviceAddress = device->get_buffer_device_address(desc.instance_buffer);

    ranges.resize(1);
    ranges[0].primitiveCount = desc.instance_count;
    ranges[0].primitiveOffset = 0;
    ranges[0].firstVertex = 0;
    ranges[0].transformOffset = 0;
    p_ranges.resize(1);
    p_ranges[0] = &ranges[0];
  }

  build_info.geometryCount = static_cast<uint32_t>(geometries.size());
  build_info.pGeometries = geometries.data();

  auto vkCmdBuildAccelerationStructuresKHR = device->get_vkCmdBuildAccelerationStructuresKHR();
  vkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &build_info, p_ranges.data());

  // Ensure AS build writes are visible to subsequent AS builds or ray queries.
  VkMemoryBarrier as_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  as_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  as_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &as_barrier, 0, nullptr, 0, nullptr);
}

}  // namespace etx
