#include <etx/core/profiler.hxx>
#include <etx/core/environment.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/shared/base.hxx>

#include <etx/rhi/rhi.hxx>
#include <etx/rhi/rhi_imgui.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/rhi/vulkan/vk_rhi.hxx>

#include <sokol_app.h>
#include <vulkan/vulkan.h>

#include "render_context.hxx"
#include "renderer.hxx"

#include <vector>
#include <algorithm>

namespace etx {

struct RenderContextImpl {
  RenderContextImpl(TaskScheduler& s)
    : scheduler(s) {
  }

  TaskScheduler& scheduler;
  RHIContext* rhi_context = nullptr;
  RHIImGui rhi_imgui = {};
  RHICommandBuffer* rhi_cmd = nullptr;
  RHIDevice* rhi_device = nullptr;
};

ETX_IMPLEMENT_PIMPL(RenderContext);

RenderContext::RenderContext(TaskScheduler& s) {
  ETX_PIMPL_INIT(RenderContext, s);
}

RenderContext::~RenderContext() {
  ETX_PIMPL_CLEANUP(RenderContext);
}

RHIContext* RenderContext::get_context() {
  return _private->rhi_context;
}

RHIDevice* RenderContext::get_device() {
  return _private->rhi_device;
}

RHITextureFormat RenderContext::get_swapchain_format() {
  return _private->rhi_context != nullptr ? _private->rhi_context->get_swapchain_format() : RHITextureFormat::Undefined;
}

RHITextureFormat RenderContext::get_depth_format() {
  return RHITextureFormat::D32_FLOAT;
}

void RenderContext::init() {
  RHIInitInfo info = {
    .backend = RHIBackend::Vulkan,
    .enable_validation = ETX_DEBUG,
  };
  const void* native_window = nullptr;
#if ETX_PLATFORM_WINDOWS
  native_window = sapp_win32_get_hwnd();
#elif defined(__APPLE__)
  native_window = sapp_macos_get_window();
#endif
  if (native_window == nullptr)
    return;

  _private->rhi_context = RHIContext::create(info);
  _private->rhi_context->create_swapchain(native_window, static_cast<uint32_t>(sapp_width()), static_cast<uint32_t>(sapp_height()));
  _private->rhi_device = _private->rhi_context->get_device();

  static RHIImGuiDesc imgui_desc = {
    .color_format = _private->rhi_context->get_swapchain_format(),
    .ini_filename = env().file_in_data("ui.ini"),
  };
  _private->rhi_imgui.setup(_private->rhi_context, imgui_desc);
}

void RenderContext::cleanup() {
  if (_private->rhi_context != nullptr) {
    auto vk_context = static_cast<VKContext*>(_private->rhi_context);
    if (vk_context != nullptr) {
      VkDevice vk_device = vk_context->get_vk_device();
      if (vk_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(vk_device);
      }
    }
  }

  _private->rhi_imgui.shutdown();

  if (_private->rhi_context) {
    RHIContext::release(_private->rhi_context);
    _private->rhi_context = nullptr;
    _private->rhi_device = nullptr;
    _private->rhi_cmd = nullptr;
  }
}

RHICommandBuffer* RenderContext::current_command_buffer() {
  return _private->rhi_cmd;
}

void RenderContext::start_frame(Renderer* renderer, SceneRepresentation& scene, const FrameData& frame_data) {
  ETX_PROFILER_SCOPE();

  etx::RHIImGuiFrameDesc imgui_frame_desc = {
    .width = uint32_t(sapp_width()),
    .height = uint32_t(sapp_height()),
    .delta_time = frame_data.dt,
    .dpi_scale = sapp_dpi_scale(),
  };
  _private->rhi_imgui.new_frame(imgui_frame_desc);

  Renderer::FrameData render_frame_data = {
    .view_parameters = frame_data.view_parameters,
    .dt = frame_data.dt,
  };
  renderer->prepare_frame(_private->rhi_context, scene, render_frame_data);

  _private->rhi_context->begin_frame();
  render_frame_data.swapchain_image = _private->rhi_context->get_current_swapchain_texture();
  render_frame_data.cmd = _private->rhi_context->get_command_buffer();

  _private->rhi_cmd = render_frame_data.cmd;
  _private->rhi_cmd->begin();
  _private->rhi_cmd->begin_render_pass(1, &render_frame_data.swapchain_image);

  renderer->frame(_private->rhi_context, scene, render_frame_data);
}

void RenderContext::end_frame() {
  ETX_PROFILER_SCOPE();

  _private->rhi_imgui.render(_private->rhi_cmd);
  _private->rhi_cmd->end_render_pass();
  _private->rhi_cmd->end();
  _private->rhi_context->submit_command_buffer(_private->rhi_cmd);
  _private->rhi_context->present();
  _private->rhi_cmd = nullptr;
}

RHIImGui& RenderContext::rhi_ui() {
  return _private->rhi_imgui;
}

}  // namespace etx
