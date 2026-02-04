#include <etx/rhi/vulkan/vk_rhi.hxx>
#include <etx/rhi/vulkan/vk_utils.hxx>

#include <etx/core/log.hxx>
#include <etx/core/core.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include <chrono>
#include <thread>

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

struct RenderPassKey {
  VkFormat color_format;
  bool has_depth;
  VkFormat depth_format;
  uint32_t color_attachment_count;
  bool is_swapchain;
  VkImageLayout color_final_layout;
  VkImageLayout depth_final_layout;

  bool operator==(const RenderPassKey& other) const {
    return color_format == other.color_format && has_depth == other.has_depth && depth_format == other.depth_format && color_attachment_count == other.color_attachment_count &&
           is_swapchain == other.is_swapchain && color_final_layout == other.color_final_layout && depth_final_layout == other.depth_final_layout;
  }
};

struct RenderPassKeyHash {
  size_t operator()(const RenderPassKey& key) const {
    uint64_t h = 0;
    h = etx_hash64_continue(&key.color_format, sizeof(key.color_format), h);
    h = etx_hash64_continue(&key.has_depth, sizeof(key.has_depth), h);
    h = etx_hash64_continue(&key.depth_format, sizeof(key.depth_format), h);
    h = etx_hash64_continue(&key.color_attachment_count, sizeof(key.color_attachment_count), h);
    h = etx_hash64_continue(&key.is_swapchain, sizeof(key.is_swapchain), h);
    h = etx_hash64_continue(&key.color_final_layout, sizeof(key.color_final_layout), h);
    h = etx_hash64_continue(&key.depth_final_layout, sizeof(key.depth_final_layout), h);
    return static_cast<size_t>(h);
  }
};

struct FramebufferKey {
  VkRenderPass render_pass;
  uint32_t image_index;
  uint32_t width;   // Included in key to handle window resize - different sizes get different framebuffers
  uint32_t height;  // Included in key to handle window resize - different sizes get different framebuffers

  bool operator==(const FramebufferKey& other) const {
    return render_pass == other.render_pass && image_index == other.image_index && width == other.width && height == other.height;
  }
};

struct FramebufferKeyHash {
  size_t operator()(const FramebufferKey& key) const {
    uint64_t h = 0;
    h = etx_hash64_continue(&key.render_pass, sizeof(key.render_pass), h);
    h = etx_hash64_continue(&key.image_index, sizeof(key.image_index), h);
    h = etx_hash64_continue(&key.width, sizeof(key.width), h);
    h = etx_hash64_continue(&key.height, sizeof(key.height), h);
    return static_cast<size_t>(h);
  }
};

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
    if (device._impl && device._impl->device != VK_NULL_HANDLE && !bindless_manager.is_initialized()) {
      bindless_manager.initialize(device._impl->device, device._impl->physical_device);
      device._impl->bindless_manager = &bindless_manager;

      initialize_predefined_samplers();
    }

    ShaderCompiler* global_compiler = ShaderCompiler::get_global_instance();
    if (global_compiler) {
      device.set_shader_compiler(global_compiler);
    } else {
      log::error("Global shader compiler not available - ensure ShaderCompiler::initialize_global() was called");
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
        predefined_sampler_indices[static_cast<size_t>(RHISamplerType::LinearClamp)] = get_bindless_descriptor_index(result.handle);
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
        predefined_sampler_indices[static_cast<size_t>(RHISamplerType::NearestClamp)] = get_bindless_descriptor_index(result.handle);
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

  VkRenderPass swapchain_render_pass = VK_NULL_HANDLE;

  std::unordered_map<RenderPassKey, VkRenderPass, RenderPassKeyHash> permanent_render_pass_cache;

  std::vector<VkFramebuffer> swapchain_framebuffers;
  std::unordered_map<FramebufferKey, VkFramebuffer, FramebufferKeyHash> permanent_framebuffer_cache;

  uint32_t current_framebuffer_width = 0;
  uint32_t current_framebuffer_height = 0;
  VkRenderPass current_framebuffer_render_pass = VK_NULL_HANDLE;

  struct DeferredDestructionObjects {
    std::vector<VkRenderPass> render_passes;
    std::vector<VkFramebuffer> framebuffers;
  };
  std::vector<DeferredDestructionObjects> deferred_destruction_per_frame;

  bool create_surface();
  bool create_swapchain(uint32_t width, uint32_t height);
  void destroy_swapchain();
  void create_sync_objects();
  void destroy_sync_objects();
  bool create_render_pass_cache();
  void destroy_render_pass_cache();
  VkRenderPass get_or_create_permanent_render_pass(const RenderPassKey& key);
  void destroy_permanent_render_pass_cache();
  bool create_framebuffer_cache();
  void destroy_framebuffer_cache();
  VkFramebuffer get_or_create_permanent_framebuffer(VkRenderPass render_pass, uint32_t image_index, uint32_t width, uint32_t height, const VkImageView* attachment_views,
    uint32_t attachment_count);
  void destroy_permanent_framebuffer_cache();
  void initialize_deferred_destruction();
  void destroy_deferred_objects();
  void process_deferred_destruction_for_frame(uint32_t frame_index);
  void queue_deferred_destruction(VkRenderPass render_pass, VkFramebuffer framebuffer);
  bool register_swapchain_textures_with_bindless();
  void unregister_swapchain_textures_from_bindless();
  VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats);
  VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes);
  VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height);
  VkFramebuffer get_or_create_swapchain_framebuffer(VkRenderPass render_pass, uint32_t image_index, uint32_t width, uint32_t height);
};

VKContext::VKContext(const RHIInitInfo& info)
  : _impl(new Impl(info)) {
}

VKContext::~VKContext() {
  if (_impl->device._impl != nullptr && _impl->device._impl->device != VK_NULL_HANDLE) {
    if (!_impl->in_flight_fences.empty()) {
      etx_vk_call(vkWaitForFences(_impl->device._impl->device, static_cast<uint32_t>(_impl->in_flight_fences.size()), _impl->in_flight_fences.data(), VK_TRUE, UINT64_MAX));
    }

    if (!_impl->temporary_fences.empty()) {
      etx_vk_call(vkWaitForFences(_impl->device._impl->device, static_cast<uint32_t>(_impl->temporary_fences.size()), _impl->temporary_fences.data(), VK_TRUE, UINT64_MAX));

      for (VkFence fence : _impl->temporary_fences) {
        if (fence != VK_NULL_HANDLE) {
          vkDestroyFence(_impl->device._impl->device, fence, nullptr);
        }
      }
      _impl->temporary_fences.clear();
    }

    etx_vk_call(vkDeviceWaitIdle(_impl->device._impl->device));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  _impl->destroy_deferred_objects();

  _impl->command_buffer_pool.clear();

  _impl->device.destroy_all_resources();

  _impl->destroy_framebuffer_cache();
  _impl->destroy_permanent_framebuffer_cache();
  _impl->destroy_permanent_render_pass_cache();

  _impl->destroy_swapchain();
  _impl->destroy_render_pass_cache();
  _impl->destroy_render_pass_cache();

  if (_impl->surface != VK_NULL_HANDLE && _impl->device._impl != nullptr && _impl->device._impl->instance != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(_impl->device._impl->instance, _impl->surface, nullptr);
  }

  delete _impl;
}

RHIDevice* VKContext::get_device() {
  return &_impl->device;
}

RHIBindlessManager* VKContext::get_bindless_manager() {
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

  if (!_impl->create_surface()) {
    log::error("Failed to create Vulkan surface");
    return;
  }

  if (!_impl->create_swapchain(width, height)) {
    log::error("Failed to create Vulkan swapchain");
    return;
  }

  _impl->create_sync_objects();
}

void VKContext::destroy_swapchain() {
  _impl->destroy_swapchain();
}

void VKContext::resize_swapchain(uint32_t width, uint32_t height) {
  if (_impl->width == width && _impl->height == height) {
    return;
  }

  vkDeviceWaitIdle(_impl->device._impl->device);

  for (uint32_t frame_index = 0; frame_index < kRHIMaxFrames; ++frame_index) {
    _impl->process_deferred_destruction_for_frame(frame_index);
  }

  _impl->destroy_framebuffer_cache();
  _impl->destroy_permanent_framebuffer_cache();
  _impl->destroy_permanent_render_pass_cache();
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

  if (!_impl->create_render_pass_cache()) {
    log::error("Failed to recreate render pass cache after swapchain resize");
    return;
  }

  if (!_impl->create_framebuffer_cache()) {
    log::error("Failed to recreate framebuffer cache after swapchain resize");
    return;
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

  VkResult result = vkQueuePresentKHR(_impl->device._impl->graphics_queue, &present_info);

  if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
    if (!_impl->in_flight_fences.empty()) {
      etx_vk_call(vkWaitForFences(_impl->device._impl->device, static_cast<uint32_t>(_impl->in_flight_fences.size()), _impl->in_flight_fences.data(), VK_TRUE, UINT64_MAX));
    }
    for (uint32_t frame_index = 0; frame_index < kRHIMaxFrames; ++frame_index) {
      _impl->process_deferred_destruction_for_frame(frame_index);
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

void VKContext::begin_frame() {
  if (_impl->swapchain == VK_NULL_HANDLE) {
    return;
  }

  if (etx_vk_call(vkWaitForFences(_impl->device._impl->device, 1, &_impl->in_flight_fences[_impl->current_frame], VK_TRUE, UINT64_MAX)) != VK_SUCCESS) {
    return;
  }

  // Set current frame index for staging buffer allocations
  _impl->device._impl->set_current_frame_index(_impl->current_frame);
  // After fence wait, it's safe to reset this frame's staging buffer region
  // GPU has finished reading from it in previous cycle
  _impl->device._impl->reset_staging_buffer_for_frame(_impl->current_frame);

  _impl->process_deferred_destruction_for_frame(_impl->current_frame);
  etx_vk_call(vkResetCommandPool(_impl->device.get_vk_device(), _impl->device._impl->command_pools[_impl->current_frame], VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));

  _impl->command_buffer_pool.clear();

  VkResult result = etx_vk_call(vkAcquireNextImageKHR(_impl->device._impl->device, _impl->swapchain, UINT64_MAX,
    _impl->device.get_vk_semaphore(_impl->image_available_semaphores[_impl->current_frame]), VK_NULL_HANDLE, &_impl->current_swapchain_image));

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    if (!_impl->in_flight_fences.empty()) {
      etx_vk_call(vkWaitForFences(_impl->device._impl->device, static_cast<uint32_t>(_impl->in_flight_fences.size()), _impl->in_flight_fences.data(), VK_TRUE, UINT64_MAX));
    }

    for (uint32_t frame_index = 0; frame_index < kRHIMaxFrames; ++frame_index) {
      _impl->process_deferred_destruction_for_frame(frame_index);
    }

    _impl->destroy_framebuffer_cache();
    _impl->destroy_swapchain();

    VkSurfaceCapabilitiesKHR capabilities;
    if (etx_vk_call(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_impl->device._impl->physical_device, _impl->surface, &capabilities)) != VK_SUCCESS) {
      return;
    }

    VkExtent2D new_extent = _impl->choose_swap_extent(capabilities, _impl->width, _impl->height);

    if (new_extent.width != _impl->width || new_extent.height != _impl->height) {
      _impl->destroy_framebuffer_cache();
    }

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

  if (!_impl->swapchain_framebuffers.empty() &&
      (_impl->current_framebuffer_width != _impl->swapchain_extent.width || _impl->current_framebuffer_height != _impl->swapchain_extent.height)) {
    _impl->destroy_framebuffer_cache();
  }

  etx_vk_call(vkResetFences(_impl->device._impl->device, 1, &_impl->in_flight_fences[_impl->current_frame]));
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
  if (!vk_cmd_buf) {
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
          etx_vk_call(vkResetFences(_impl->device._impl->device, 1, &submit_fence));
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

  if (etx_vk_call(vkQueueSubmit(_impl->device._impl->graphics_queue, 1, &submit_info, submit_fence)) != VK_SUCCESS) {
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
  return _impl->device._impl->device;
}

VkCommandPool VKContext::get_vk_command_pool(uint32_t index) const {
  return _impl->device._impl->command_pools[index];
}

VkFence VKContext::get_current_frame_fence() const {
  return _impl->in_flight_fences[_impl->current_frame];
}

VkQueue VKContext::get_graphics_queue() const {
  return _impl->device._impl->graphics_queue;
}

VkRenderPass VKCommandBuffer::create_render_pass_for_attachments(const std::vector<VkFormat>& attachment_formats, bool has_depth) {
  // Fixed-size arrays (max 9 attachments as per Vulkan limits)
  VkAttachmentDescription attachments[9] = {};
  VkAttachmentReference color_refs[8] = {};  // Max 8 color attachments
  VkAttachmentReference depth_ref = {};

  uint32_t attachment_count = 0;
  uint32_t color_ref_count = 0;

  for (size_t i = 0; i < attachment_formats.size() && attachment_count < 9; ++i) {
    VkFormat format = attachment_formats[i];
    VkAttachmentDescription attachment = {};
    attachment.format = format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    bool is_depth = (format == VK_FORMAT_D32_SFLOAT) ||         //
                    (format == VK_FORMAT_D24_UNORM_S8_UINT) ||  //
                    (format == VK_FORMAT_D32_SFLOAT_S8_UINT);

    if (is_depth && has_depth && depth_ref.attachment == VK_ATTACHMENT_UNUSED) {
      attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      depth_ref.attachment = attachment_count;
      depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    } else if (color_ref_count < 8) {
      attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      VkAttachmentReference color_ref = {};
      color_ref.attachment = attachment_count;
      color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      color_refs[color_ref_count++] = color_ref;
    }

    attachments[attachment_count++] = attachment;
  }

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = color_ref_count;
  subpass.pColorAttachments = color_refs;
  if (has_depth) {
    subpass.pDepthStencilAttachment = &depth_ref;
  }

  VkSubpassDependency dependencies[2] = {};

  dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[0].dstSubpass = 0;
  dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[0].srcAccessMask = 0;
  dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  dependencies[1].srcSubpass = 0;
  dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  dependencies[1].dstAccessMask = 0;

  VkRenderPassCreateInfo render_pass_info = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  render_pass_info.attachmentCount = attachment_count;
  render_pass_info.pAttachments = attachments;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 2;
  render_pass_info.pDependencies = dependencies;

  VkRenderPass render_pass;
  if (etx_vk_call(vkCreateRenderPass(context->get_vk_device(), &render_pass_info, nullptr, &render_pass)) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }

  return render_pass;
}

VkFramebuffer VKCommandBuffer::create_framebuffer_for_attachments(VkRenderPass render_pass, const VkImageView* attachment_views, uint32_t attachment_count, uint32_t width,
  uint32_t height) {
  VkFramebufferCreateInfo framebuffer_info = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  framebuffer_info.renderPass = render_pass;
  framebuffer_info.attachmentCount = attachment_count;
  framebuffer_info.pAttachments = attachment_views;
  framebuffer_info.width = width;
  framebuffer_info.height = height;
  framebuffer_info.layers = 1;

  VkFramebuffer framebuffer;
  if (etx_vk_call(vkCreateFramebuffer(context->get_vk_device(), &framebuffer_info, nullptr, &framebuffer)) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }

  return framebuffer;
}

VKCommandBuffer::VKCommandBuffer() {
}

void VKCommandBuffer::initialize(VKContext* ctx, uint32_t pool_index) {
  context = ctx;
  device = static_cast<VKDevice*>(ctx->get_device());
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
    render_pass_depth = other.render_pass_depth;
    current_render_pass = other.current_render_pass;
    current_framebuffer = other.current_framebuffer;
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
  if (current_framebuffer != VK_NULL_HANDLE) {
    const auto& cached_framebuffers = context->_impl->swapchain_framebuffers;
    bool is_cached_framebuffer = false;
    for (VkFramebuffer cached_fb : cached_framebuffers) {
      if (current_framebuffer == cached_fb) {
        is_cached_framebuffer = true;
        break;
      }
    }

    if (!is_cached_framebuffer) {
      vkDestroyFramebuffer(context->get_vk_device(), current_framebuffer, nullptr);
    }
    current_framebuffer = VK_NULL_HANDLE;
  }

  if (current_render_pass != VK_NULL_HANDLE && current_render_pass != context->_impl->swapchain_render_pass) {
    vkDestroyRenderPass(context->get_vk_device(), current_render_pass, nullptr);
    current_render_pass = VK_NULL_HANDLE;
  }
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
  render_pass_depth = 0;
  _in_render_pass = false;
  current_render_pass = VK_NULL_HANDLE;
  current_framebuffer = VK_NULL_HANDLE;
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
  render_pass_depth = 0;
  current_render_pass = VK_NULL_HANDLE;
  current_framebuffer = VK_NULL_HANDLE;
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
    vkCmdEndRenderPass(command_buffer);
    _in_render_pass = false;
  }

  if (etx_vk_call(vkEndCommandBuffer(command_buffer)) == VK_SUCCESS) {
    _is_recording = false;
  }
}

void VKCommandBuffer::reset_internal_state() {
  current_pipeline = {};
  current_pipeline_layout = VK_NULL_HANDLE;
  render_pass_depth = 0;
  _in_render_pass = false;
  current_render_pass = VK_NULL_HANDLE;
  current_framebuffer = VK_NULL_HANDLE;
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
  bool is_swapchain_texture = false;
  for (uint32_t j = 0; j < context->_impl->swapchain_textures.size(); ++j) {
    if (context->_impl->swapchain_textures[j] == texture) {
      is_swapchain_texture = true;
      break;
    }
  }

  if (is_swapchain_texture) {
    VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = required_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = static_cast<VKBindlessManager*>(device->_impl->bindless_manager)->get_vk_image(texture);
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

  uint32_t texture_index = device->_impl->textures.get_index(texture);
  if (texture_index == UINT32_MAX) {
    return;
  }
  const VKTextureData& texture_data = device->_impl->textures.get_data(texture_index);
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

  uint32_t texture_index = device->_impl->textures.get_index(texture);
  if (texture_index == UINT32_MAX) {
    log::error("Texture handle not found in texture map");
    return;
  }
  VKTextureData& texture_data = device->_impl->textures.get_data(texture_index);

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

  if (color_attachment_count == 0 || color_attachments == nullptr) {
    log::error("No color attachments provided to begin_render_pass");
    return;
  }

  if (render_pass_depth > 0) {
    log::error("Cannot begin render pass: already in render pass (depth: %u). Call end_render_pass() first", render_pass_depth);
    return;
  }

  bool should_destroy_render_pass = (current_render_pass != VK_NULL_HANDLE && current_render_pass != context->_impl->swapchain_render_pass);

  if (should_destroy_render_pass) {
    bool is_permanent_render_pass = false;
    for (const auto& pair : context->_impl->permanent_render_pass_cache) {
      if (pair.second == current_render_pass) {
        is_permanent_render_pass = true;
        break;
      }
    }

    if (!is_permanent_render_pass) {
      context->_impl->queue_deferred_destruction(current_render_pass, current_framebuffer);
      current_render_pass = VK_NULL_HANDLE;
      current_framebuffer = VK_NULL_HANDLE;
    } else {
      current_render_pass = VK_NULL_HANDLE;
    }
  }

  if (current_framebuffer != VK_NULL_HANDLE && !should_destroy_render_pass) {
    const auto& cached_framebuffers = context->_impl->swapchain_framebuffers;
    bool is_cached_framebuffer = false;
    for (VkFramebuffer cached_fb : cached_framebuffers) {
      if (current_framebuffer == cached_fb) {
        is_cached_framebuffer = true;
        break;
      }
    }

    if (!is_cached_framebuffer) {
      context->_impl->queue_deferred_destruction(VK_NULL_HANDLE, current_framebuffer);
    }
    current_framebuffer = VK_NULL_HANDLE;
  }

  constexpr uint32_t MAX_ATTACHMENTS = 9;
  VkFormat attachment_formats[MAX_ATTACHMENTS] = {};
  VkImageView attachment_views[MAX_ATTACHMENTS] = {};
  VkClearValue clear_values[MAX_ATTACHMENTS] = {};
  uint32_t total_attachments = 0;

  if (color_attachment_count > MAX_ATTACHMENTS - 1) {  // Reserve 1 for potential depth
    log::error("Too many color attachments: %u (max %u)", color_attachment_count, MAX_ATTACHMENTS - 1);
    return;
  }

  for (uint32_t i = 0; i < color_attachment_count; ++i) {
    VkImage vk_image = device->get_vk_image_from_bindless(color_attachments[i]);
    if (vk_image == VK_NULL_HANDLE) {
      log::error("Invalid color attachment %u", i);
      return;
    }

    bool is_swapchain_texture = false;
    VkFormat swapchain_format = VK_FORMAT_UNDEFINED;
    VkImageView swapchain_image_view = VK_NULL_HANDLE;

    for (uint32_t j = 0; j < context->_impl->swapchain_textures.size(); ++j) {
      if (context->_impl->swapchain_textures[j] == color_attachments[i]) {
        is_swapchain_texture = true;
        swapchain_format = context->_impl->swapchain_format;
        swapchain_image_view = context->_impl->swapchain_image_views[j];
        break;
      }
    }

    VkFormat attachment_format;
    VkImageView attachment_view;

    if (is_swapchain_texture) {
      attachment_format = swapchain_format;
      attachment_view = swapchain_image_view;
    } else {
      uint32_t texture_index = device->_impl->textures.get_index(color_attachments[i]);
      if (texture_index == UINT32_MAX) {
        log::error("Color attachment %u not found in texture map", i);
        return;
      }
      const VKTextureData& texture_data = device->_impl->textures.get_data(texture_index);
      attachment_format = convert_rhi_format_to_vk(texture_data.desc.format);
      attachment_view = texture_data.image_view;
    }

    attachment_formats[total_attachments] = attachment_format;
    attachment_views[total_attachments] = attachment_view;

    VkClearValue clear_value = {};
    if (clear_colors) {
      clear_value.color = {clear_colors[i * 4], clear_colors[i * 4 + 1], clear_colors[i * 4 + 2], clear_colors[i * 4 + 3]};
    } else {
      clear_value.color = {0.0f, 0.0f, 0.0f, 1.0f};
    }
    clear_values[total_attachments] = clear_value;
    total_attachments++;

    ensure_texture_layout(color_attachments[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  }

  bool has_depth = depth_attachment.valid();
  if (has_depth) {
    VkImage vk_image = device->get_vk_image_from_bindless(depth_attachment);
    if (vk_image == VK_NULL_HANDLE) {
      log::error("Invalid depth attachment");
      return;
    }

    uint32_t texture_index = device->_impl->textures.get_index(depth_attachment);
    if (texture_index == UINT32_MAX) {
      log::error("Depth attachment not found in texture map");
      return;
    }
    const VKTextureData& texture_data = device->_impl->textures.get_data(texture_index);

    attachment_formats[total_attachments] = convert_rhi_format_to_vk(texture_data.desc.format);
    attachment_views[total_attachments] = texture_data.image_view;

    VkClearValue clear_value = {};
    clear_value.depthStencil = {1.0f, 0};
    clear_values[total_attachments] = clear_value;
    total_attachments++;

    ensure_texture_layout(depth_attachment, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  }

  VkImageLayout requested_color_final_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  if (color_final_states != nullptr) {
    requested_color_final_layout = rhi_state_to_vk_layout(color_final_states[0], false);
    for (uint32_t i = 1; i < color_attachment_count; ++i) {
      VkImageLayout layout_i = rhi_state_to_vk_layout(color_final_states[i], false);
      if (layout_i != requested_color_final_layout) {
        log::error("Mismatched color final layouts in begin_render_pass - all color attachments must share the same final layout");
        return;
      }
    }
  }

  VkImageLayout requested_depth_final_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  if (has_depth) {
    requested_depth_final_layout = rhi_state_to_vk_layout(depth_final_state, true);
  }

  bool use_cached_render_pass = false;

  bool is_swapchain_pass = false;
  if (color_attachment_count == 1 && !has_depth) {
    for (uint32_t j = 0; j < context->_impl->swapchain_textures.size(); ++j) {
      if (context->_impl->swapchain_textures[j] == color_attachments[0]) {
        is_swapchain_pass = true;
        break;
      }
    }
  }

  if (is_swapchain_pass) {
    use_cached_render_pass = true;
  } else {
    VkFormat color_fmt = attachment_formats[0];
    VkFormat depth_fmt = has_depth ? attachment_formats[color_attachment_count] : VK_FORMAT_UNDEFINED;
    RenderPassKey key = {color_fmt, has_depth, depth_fmt, color_attachment_count, false, requested_color_final_layout, requested_depth_final_layout};
    VkRenderPass permanent_rp = context->_impl->get_or_create_permanent_render_pass(key);
    if (permanent_rp != VK_NULL_HANDLE) {
      current_render_pass = permanent_rp;
      use_cached_render_pass = true;
    }
  }

  VkRenderPass render_pass_to_use = VK_NULL_HANDLE;

  if (use_cached_render_pass) {
    if (is_swapchain_pass) {
      render_pass_to_use = context->_impl->swapchain_render_pass;
    } else {
      render_pass_to_use = current_render_pass;
    }
  } else {
    // Dynamic/Temporary render pass
    std::vector<VkFormat> formats;
    for (uint32_t i = 0; i < total_attachments; ++i) {
      formats.push_back(attachment_formats[i]);
    }
    render_pass_to_use = create_render_pass_for_attachments(formats, has_depth);
  }

  if (render_pass_to_use == VK_NULL_HANDLE) {
    log::error("Failed to obtain render pass");
    return;
  }

  VkFramebuffer framebuffer = VK_NULL_HANDLE;

  if (is_swapchain_pass) {
    uint32_t swapchain_image_index = context->_impl->current_swapchain_image;
    VkExtent2D ext = context->_impl->swapchain_extent;
    framebuffer = context->_impl->get_or_create_swapchain_framebuffer(render_pass_to_use, swapchain_image_index, ext.width, ext.height);
  } else if (use_cached_render_pass) {
    // If using a permanent/cached render pass, try to use/create a permanent framebuffer
    // But we need dimensions. Assume 0th attachment defines dimensions.
    VkImageView attachment_view_0 = attachment_views[0];
    uint32_t width = 0;
    uint32_t height = 0;

    // We can get it from the texture handle (we have color_attachments[0]).
    RHITexture tex0 = color_attachments[0];
    uint32_t t_idx = device->_impl->textures.get_index(tex0);
    if (t_idx != UINT32_MAX) {
      const VKTextureData& t_data = device->_impl->textures.get_data(t_idx);
      width = t_data.desc.width;
      height = t_data.desc.height;
    } else {
      log::error("Invalid texture handle for framebuffer creation: %llu", tex0.value);
      return;
    }

    framebuffer = context->_impl->get_or_create_permanent_framebuffer(render_pass_to_use, 0, width, height, attachment_views, total_attachments);
  }

  if (framebuffer == VK_NULL_HANDLE) {
    // Create temporary framebuffer
    RHITexture tex0 = color_attachments[0];
    uint32_t t_idx = device->_impl->textures.get_index(tex0);
    uint32_t width = 0;
    uint32_t height = 0;

    if (t_idx != UINT32_MAX) {
      const VKTextureData& t_data = device->_impl->textures.get_data(t_idx);
      width = t_data.desc.width;
      height = t_data.desc.height;
    } else {
      log::error("Invalid texture handle for temporary framebuffer creation: %llu", tex0.value);
      return;
    }

    framebuffer = create_framebuffer_for_attachments(render_pass_to_use, attachment_views, total_attachments, width, height);
  }

  if (framebuffer == VK_NULL_HANDLE) {
    log::error("Failed to create framebuffer");
    if (!use_cached_render_pass) {
      vkDestroyRenderPass(context->get_vk_device(), render_pass_to_use, nullptr);
    }
    return;
  }

  VkRenderPassBeginInfo begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  begin_info.renderPass = render_pass_to_use;
  begin_info.framebuffer = framebuffer;
  begin_info.renderArea.offset = {0, 0};

  if (is_swapchain_pass) {
    begin_info.renderArea.extent = context->_impl->swapchain_extent;
  } else {
    RHITexture tex0 = color_attachments[0];
    uint32_t t_idx = device->_impl->textures.get_index(tex0);
    if (t_idx != UINT32_MAX) {
      const VKTextureData& t_data = device->_impl->textures.get_data(t_idx);
      begin_info.renderArea.extent = {t_data.desc.width, t_data.desc.height};
    } else {
      log::error("Invalid texture handle for render area extent: %llu", tex0.value);
      return;
    }
  }
  begin_info.clearValueCount = total_attachments;
  begin_info.pClearValues = clear_values;

  vkCmdBeginRenderPass(command_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

  current_render_pass = render_pass_to_use;
  current_framebuffer = framebuffer;
  render_pass_depth++;
  _in_render_pass = (render_pass_depth > 0);

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

  if (render_pass_depth == 0) {
    log::error("Cannot end render pass: not currently in a render pass");
    return;
  }

  vkCmdEndRenderPass(command_buffer);

  if (device != nullptr) {
    for (size_t i = 0; i < current_color_attachments.size(); ++i) {
      RHITexture tex = current_color_attachments[i];
      bool is_swapchain_texture = false;
      for (uint32_t j = 0; j < context->_impl->swapchain_textures.size(); ++j) {
        if (context->_impl->swapchain_textures[j] == tex) {
          is_swapchain_texture = true;
          break;
        }
      }
      if (is_swapchain_texture) {
        continue;
      }
      uint32_t tex_index = device->_impl->textures.get_index(tex);
      if (tex_index != UINT32_MAX) {
        auto& tex_data = device->_impl->textures.get_data(tex_index);
        tex_data.current_state = current_color_final_states[i];
      }
    }

    if (current_depth_attachment.valid()) {
      uint32_t tex_index = device->_impl->textures.get_index(current_depth_attachment);
      if (tex_index != UINT32_MAX) {
        auto& tex_data = device->_impl->textures.get_data(tex_index);
        tex_data.current_state = current_depth_final_state;
      }
    }
  }

  if (context != nullptr) {
    RHITexture current_texture = context->get_current_swapchain_texture();
    if (current_texture.valid()) {
      ensure_texture_layout(current_texture, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    }
  }

  // Queue framebuffer for deferred destruction if it's not cached
  if (current_framebuffer != VK_NULL_HANDLE && context != nullptr) {
    const auto& cached_framebuffers = context->_impl->swapchain_framebuffers;
    bool is_cached_framebuffer = false;
    for (VkFramebuffer cached_fb : cached_framebuffers) {
      if (current_framebuffer == cached_fb) {
        is_cached_framebuffer = true;
        break;
      }
    }

    // Check if it's in permanent framebuffer cache
    if (!is_cached_framebuffer) {
      bool is_permanent_framebuffer = false;
      for (const auto& pair : context->_impl->permanent_framebuffer_cache) {
        if (pair.second == current_framebuffer) {
          is_permanent_framebuffer = true;
          break;
        }
      }

      if (!is_permanent_framebuffer) {
        context->_impl->queue_deferred_destruction(VK_NULL_HANDLE, current_framebuffer);
      }
    }
    current_framebuffer = VK_NULL_HANDLE;
  }

  render_pass_depth--;
  _in_render_pass = (render_pass_depth > 0);
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
    VkDescriptorSet bindless_set = static_cast<VKBindlessManager*>(context->get_bindless_manager())->get_descriptor_set();
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

  VkDescriptorSet bindless_set = static_cast<VKBindlessManager*>(context->get_bindless_manager())->get_descriptor_set();
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

  auto vkCreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(device._impl->instance, "vkCreateWin32SurfaceKHR");
  if (!vkCreateWin32SurfaceKHR) {
    log::error("Failed to get vkCreateWin32SurfaceKHR function pointer");
    return false;
  }

  if (etx_vk_call(vkCreateWin32SurfaceKHR(device._impl->instance, &surface_info, nullptr, &surface)) != VK_SUCCESS) {
    return false;
  }
#else

  log::error("Platform not supported for surface creation");
  return false;
#endif

  return true;
}

bool VKContext::Impl::create_swapchain(uint32_t width, uint32_t height) {
  VkPhysicalDevice physical_device = device._impl->physical_device;
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

  if (etx_vk_call(vkCreateSwapchainKHR(device._impl->device, &swapchain_info, nullptr, &swapchain)) != VK_SUCCESS) {
    return false;
  }

  swapchain_format = surface_format.format;

  etx_vk_call(vkGetSwapchainImagesKHR(device._impl->device, swapchain, &image_count, nullptr));
  swapchain_images.resize(image_count);
  etx_vk_call(vkGetSwapchainImagesKHR(device._impl->device, swapchain, &image_count, swapchain_images.data()));

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

    if (etx_vk_call(vkCreateImageView(device._impl->device, &view_info, nullptr, &swapchain_image_views[i])) != VK_SUCCESS) {
      return false;
    }
  }

  if (!register_swapchain_textures_with_bindless()) {
    log::error("Failed to register swapchain textures with bindless manager");
    return false;
  }

  swapchain_format = surface_format.format;
  swapchain_extent = extent;

  if (!create_render_pass_cache()) {
    log::error("Failed to create render pass cache");
    return false;
  }

  if (!create_framebuffer_cache()) {
    log::error("Failed to create framebuffer cache");
    return false;
  }

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
    etx_vk_call(vkCreateFence(device._impl->device, &fence_info, nullptr, &in_flight_fences[i]));
  }

  initialize_deferred_destruction();
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
      vkDestroyFence(device._impl->device, fence, nullptr);
    }
  }

  image_available_semaphores.clear();
  render_finished_semaphores.clear();
  in_flight_fences.clear();
}

void VKContext::Impl::destroy_swapchain() {
  // Clear permanent framebuffer cache when swapchain is destroyed since image views will be invalid
  destroy_permanent_framebuffer_cache();
  if (device._impl->device == VK_NULL_HANDLE) {
    return;
  }

  uint32_t swapchain_texture_count = static_cast<uint32_t>(swapchain_textures.size());
  uint32_t swapchain_image_view_count = static_cast<uint32_t>(swapchain_image_views.size());

  unregister_swapchain_textures_from_bindless();

  for (auto image_view : swapchain_image_views) {
    if (image_view != VK_NULL_HANDLE) {
      vkDestroyImageView(device._impl->device, image_view, nullptr);
    }
  }

  destroy_sync_objects();

  if (swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device._impl->device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
  }

  swapchain_images.clear();
  swapchain_image_views.clear();
  swapchain_textures.clear();
  swapchain_format = VK_FORMAT_UNDEFINED;
  swapchain_extent = {0, 0};
  current_swapchain_image = 0;

  current_framebuffer_width = 0;
  current_framebuffer_height = 0;
  current_framebuffer_render_pass = VK_NULL_HANDLE;
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

bool VKContext::Impl::create_render_pass_cache() {
  if (swapchain_render_pass != VK_NULL_HANDLE) {
    return true;
  }

  if (device._impl->device == VK_NULL_HANDLE) {
    log::error("Cannot create render pass cache: device not initialized");
    return false;
  }

  VkAttachmentDescription color_attachment = {};
  color_attachment.format = swapchain_format;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference color_attachment_ref = {};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;

  VkSubpassDependency dependencies[2] = {};
  dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[0].dstSubpass = 0;
  dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[0].srcAccessMask = 0;
  dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  dependencies[1].srcSubpass = 0;
  dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  dependencies[1].dstAccessMask = 0;

  VkRenderPassCreateInfo render_pass_info = {};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &color_attachment;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 2;
  render_pass_info.pDependencies = dependencies;

  if (etx_vk_call(vkCreateRenderPass(device._impl->device, &render_pass_info, nullptr, &swapchain_render_pass)) != VK_SUCCESS) {
    return false;
  }

  return true;
}

void VKContext::Impl::initialize_deferred_destruction() {
  deferred_destruction_per_frame.resize(kRHIMaxFrames);
}

void VKContext::Impl::process_deferred_destruction_for_frame(uint32_t frame_index) {
  if (frame_index >= deferred_destruction_per_frame.size()) {
    log::error("Invalid frame index %u for deferred destruction processing", frame_index);
    return;
  }

  auto& frame_objects = deferred_destruction_per_frame[frame_index];

  for (VkRenderPass rp : frame_objects.render_passes) {
    if (rp != VK_NULL_HANDLE && rp != swapchain_render_pass && device._impl != nullptr && device._impl->device != VK_NULL_HANDLE) {
      vkDestroyRenderPass(device._impl->device, rp, nullptr);
    }
  }

  for (VkFramebuffer fb : frame_objects.framebuffers) {
    if (fb != VK_NULL_HANDLE && device._impl != nullptr && device._impl->device != VK_NULL_HANDLE) {
      vkDestroyFramebuffer(device._impl->device, fb, nullptr);
    }
  }

  device._impl->process_deferred_destruction(frame_index);

  frame_objects.render_passes.clear();
  frame_objects.framebuffers.clear();
}

void VKContext::Impl::destroy_deferred_objects() {
  uint32_t total_render_passes = 0;
  uint32_t total_framebuffers = 0;

  for (const auto& frame_objects : deferred_destruction_per_frame) {
    total_render_passes += static_cast<uint32_t>(frame_objects.render_passes.size());
    total_framebuffers += static_cast<uint32_t>(frame_objects.framebuffers.size());
  }

  for (uint32_t frame_index = 0; frame_index < deferred_destruction_per_frame.size(); ++frame_index) {
    process_deferred_destruction_for_frame(frame_index);
  }

  deferred_destruction_per_frame.clear();
}

void VKContext::Impl::queue_deferred_destruction(VkRenderPass render_pass, VkFramebuffer framebuffer) {
  if (current_frame >= deferred_destruction_per_frame.size()) {
    log::error("Invalid current_frame %u for deferred destruction", current_frame);
    return;
  }

  auto& frame_objects = deferred_destruction_per_frame[current_frame];
  if (render_pass != VK_NULL_HANDLE) {
    frame_objects.render_passes.push_back(render_pass);
  }
  if (framebuffer != VK_NULL_HANDLE) {
    frame_objects.framebuffers.push_back(framebuffer);
  }
}

void VKContext::Impl::destroy_render_pass_cache() {
  if (swapchain_render_pass != VK_NULL_HANDLE && device._impl != nullptr && device._impl->device != VK_NULL_HANDLE) {
    vkDestroyRenderPass(device._impl->device, swapchain_render_pass, nullptr);
    swapchain_render_pass = VK_NULL_HANDLE;
  } else {
  }
}

VkFramebuffer VKContext::Impl::get_or_create_swapchain_framebuffer(VkRenderPass render_pass, uint32_t image_index, uint32_t width, uint32_t height) {
  if (image_index < swapchain_framebuffers.size() && swapchain_framebuffers[image_index] != VK_NULL_HANDLE && current_framebuffer_width == width &&
      current_framebuffer_height == height && current_framebuffer_render_pass == render_pass) {
    if (width == swapchain_extent.width && height == swapchain_extent.height) {
      return swapchain_framebuffers[image_index];
    }
  }

  if (image_index >= swapchain_image_views.size()) {
    log::error("Invalid swapchain image index %u", image_index);
    return VK_NULL_HANDLE;
  }

  if (image_index < swapchain_framebuffers.size() && swapchain_framebuffers[image_index] != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(device._impl->device, swapchain_framebuffers[image_index], nullptr);
    swapchain_framebuffers[image_index] = VK_NULL_HANDLE;
  }

  if (swapchain_framebuffers.size() <= image_index) {
    swapchain_framebuffers.resize(image_index + 1, VK_NULL_HANDLE);
  }

  VkImageView attachments[] = {swapchain_image_views[image_index]};

  VkFramebufferCreateInfo framebuffer_info = {};
  framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_info.renderPass = render_pass;
  framebuffer_info.attachmentCount = 1;
  framebuffer_info.pAttachments = attachments;
  framebuffer_info.width = width;
  framebuffer_info.height = height;
  framebuffer_info.layers = 1;

  if (etx_vk_call(vkCreateFramebuffer(device._impl->device, &framebuffer_info, nullptr, &swapchain_framebuffers[image_index])) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }

  current_framebuffer_width = width;
  current_framebuffer_height = height;
  current_framebuffer_render_pass = render_pass;

  return swapchain_framebuffers[image_index];
}

bool VKContext::Impl::create_framebuffer_cache() {
  if (!swapchain_framebuffers.empty()) {
    if (current_framebuffer_width == swapchain_extent.width && current_framebuffer_height == swapchain_extent.height && current_framebuffer_render_pass == swapchain_render_pass) {
      return true;
    }

    destroy_framebuffer_cache();
  }

  if (swapchain_render_pass == VK_NULL_HANDLE || swapchain_image_views.empty()) {
    log::error("Cannot create framebuffer cache: render pass or image views not available");
    return false;
  }

  swapchain_framebuffers.resize(swapchain_image_views.size());

  for (size_t i = 0; i < swapchain_image_views.size(); ++i) {
    VkImageView attachments[] = {swapchain_image_views[i]};

    VkFramebufferCreateInfo framebuffer_info = {};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = swapchain_render_pass;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = attachments;
    framebuffer_info.width = swapchain_extent.width;
    framebuffer_info.height = swapchain_extent.height;
    framebuffer_info.layers = 1;

    if (etx_vk_call(vkCreateFramebuffer(device._impl->device, &framebuffer_info, nullptr, &swapchain_framebuffers[i])) != VK_SUCCESS) {
      for (size_t j = 0; j < i; ++j) {
        if (swapchain_framebuffers[j] != VK_NULL_HANDLE) {
          vkDestroyFramebuffer(device._impl->device, swapchain_framebuffers[j], nullptr);
        }
      }
      swapchain_framebuffers.clear();
      return false;
    }
  }

  current_framebuffer_width = swapchain_extent.width;
  current_framebuffer_height = swapchain_extent.height;
  current_framebuffer_render_pass = swapchain_render_pass;

  return true;
}

void VKContext::Impl::destroy_framebuffer_cache() {
  if (!swapchain_framebuffers.empty() && device._impl != nullptr && device._impl->device != VK_NULL_HANDLE) {
    for (VkFramebuffer fb : swapchain_framebuffers) {
      if (fb != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(device._impl->device, fb, nullptr);
      }
    }
    swapchain_framebuffers.clear();
    current_framebuffer_width = 0;
    current_framebuffer_height = 0;
    current_framebuffer_render_pass = VK_NULL_HANDLE;
  } else {
  }
}

VkRenderPass VKContext::Impl::get_or_create_permanent_render_pass(const RenderPassKey& key) {
  auto it = permanent_render_pass_cache.find(key);
  if (it != permanent_render_pass_cache.end()) {
    return it->second;
  }

  VkRenderPass render_pass = VK_NULL_HANDLE;

  // Fixed-size arrays for render pass creation (max 2 attachments: color + depth)
  VkAttachmentDescription attachments[2] = {};
  VkAttachmentReference color_refs[1] = {};
  VkAttachmentReference depth_ref = {};

  uint32_t attachment_count = 0;
  uint32_t color_ref_count = 0;

  VkAttachmentDescription color_attachment = {};
  color_attachment.format = key.color_format;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = key.is_swapchain ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : key.color_final_layout;
  attachments[attachment_count++] = color_attachment;

  VkAttachmentReference color_ref = {};
  color_ref.attachment = 0;
  color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color_refs[color_ref_count++] = color_ref;

  if (key.has_depth) {
    VkAttachmentDescription depth_attachment = {};
    depth_attachment.format = key.depth_format;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = key.depth_final_layout;
    attachments[attachment_count++] = depth_attachment;

    depth_ref.attachment = 1;
    depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  }

  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = color_ref_count;
  subpass.pColorAttachments = color_refs;
  if (key.has_depth) {
    subpass.pDepthStencilAttachment = &depth_ref;
  }

  VkSubpassDependency dependencies[2] = {};
  dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[0].dstSubpass = 0;
  dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  if (key.has_depth) {
    dependencies[0].srcStageMask |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  }
  dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  if (key.has_depth) {
    dependencies[0].dstStageMask |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  }
  dependencies[0].srcAccessMask = 0;
  dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  if (key.has_depth) {
    dependencies[0].dstAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  }

  dependencies[1].srcSubpass = 0;
  dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  if (key.has_depth) {
    dependencies[1].srcStageMask |= VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  }
  dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  if (key.has_depth) {
    dependencies[1].srcAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  }
  dependencies[1].dstAccessMask = 0;

  VkRenderPassCreateInfo render_pass_info = {};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = attachment_count;
  render_pass_info.pAttachments = attachments;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;
  render_pass_info.dependencyCount = 2;
  render_pass_info.pDependencies = dependencies;

  if (etx_vk_call(vkCreateRenderPass(device._impl->device, &render_pass_info, nullptr, &render_pass)) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }

  permanent_render_pass_cache[key] = render_pass;

  return render_pass;
}

void VKContext::Impl::destroy_permanent_render_pass_cache() {
  if (!permanent_render_pass_cache.empty() && device._impl != nullptr && device._impl->device != VK_NULL_HANDLE) {
    std::vector<VkRenderPass> render_passes_to_destroy;
    for (const auto& pair : permanent_render_pass_cache) {
      VkRenderPass rp = pair.second;
      if (rp != VK_NULL_HANDLE && rp != swapchain_render_pass) {
        render_passes_to_destroy.push_back(rp);
      }
    }

    permanent_render_pass_cache.clear();

    for (VkRenderPass rp : render_passes_to_destroy) {
      vkDestroyRenderPass(device._impl->device, rp, nullptr);
    }

  } else {
  }
}

VkFramebuffer VKContext::Impl::get_or_create_permanent_framebuffer(VkRenderPass render_pass, uint32_t image_index, uint32_t width, uint32_t height,
  const VkImageView* attachment_views, uint32_t attachment_count) {
  FramebufferKey key = {render_pass, image_index, width, height};
  auto it = permanent_framebuffer_cache.find(key);
  if (it != permanent_framebuffer_cache.end()) {
    return it->second;
  }

  VkFramebufferCreateInfo framebuffer_info = {};
  framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_info.renderPass = render_pass;
  framebuffer_info.attachmentCount = attachment_count;
  framebuffer_info.pAttachments = attachment_views;
  framebuffer_info.width = width;
  framebuffer_info.height = height;
  framebuffer_info.layers = 1;

  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  if (etx_vk_call(vkCreateFramebuffer(device._impl->device, &framebuffer_info, nullptr, &framebuffer)) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }

  permanent_framebuffer_cache[key] = framebuffer;
  return framebuffer;
}

void VKContext::Impl::destroy_permanent_framebuffer_cache() {
  if (!permanent_framebuffer_cache.empty() && device._impl != nullptr && device._impl->device != VK_NULL_HANDLE) {
    for (const auto& pair : permanent_framebuffer_cache) {
      if (pair.second != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(device._impl->device, pair.second, nullptr);
      }
    }
    permanent_framebuffer_cache.clear();
  }
}

void VKCommandBuffer::build_acceleration_structure(const RHIAccelerationStructureBuildDesc& desc, RHIBindlessHandle scratch_buffer, uint64_t scratch_offset) {
  uint32_t as_index = device->_impl->acceleration_structures.get_index(desc.as_handle);
  if (as_index == UINT32_MAX) {
    return;
  }

  VKAccelerationStructureData& as_data = device->_impl->acceleration_structures.get_data(as_index);

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

  device->_impl->impl_vkCmdBuildAccelerationStructuresKHR(command_buffer, 1, &build_info, p_ranges.data());

  // Ensure AS build writes are visible to subsequent AS builds or ray queries.
  VkMemoryBarrier as_barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  as_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  as_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &as_barrier, 0, nullptr, 0, nullptr);
}

}  // namespace etx
