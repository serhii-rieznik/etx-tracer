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
    : scheduler(s)
    , image_pool(images, images_storage) {
  }

  TaskScheduler& scheduler;
  RHIContext* rhi_context = nullptr;
  RHIImGui rhi_imgui = {};
  RHICommandBuffer rhi_cmd = {};
  RHIDevice* rhi_device = nullptr;
  RHIPipeline presentation_pipeline = {};
  Renderer* active_renderer = nullptr;
  RenderContext::FrameData frame_data = {};

  RHITexture reference_texture = {};
  std::vector<Image> images;
  std::vector<ImageStorage> images_storage;
  ImagePool image_pool;
  uint32_t reference_image_handle = kInvalidIndex;

  void apply_reference_image(RHIContext* ctx, uint32_t handle) {
    const auto& img = image_pool.get(handle);
    if (reference_texture.valid()) {
      ctx->get_device()->destroy_texture(reference_texture);
      reference_texture = {};
    }
    RHITextureDesc desc = {
      .width = img.isize.x,
      .height = img.isize.y,
      .format = (img.format == Image::Format::RGBA32F) ? RHITextureFormat::R32G32B32A32_FLOAT : RHITextureFormat::R8G8B8A8_UNORM,
      .usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst,
    };
    reference_texture = ctx->get_device()->create_texture(desc).handle;
    const void* data_ptr = (img.format == Image::Format::RGBA32F) ? (const void*)img.pixels.f32.a : (const void*)img.pixels.u8.a;
    ctx->get_device()->update_texture(reference_texture, data_ptr, 0, 0);
  }
};

struct RenderParameters {
  ViewParameters view;
  float4 dimensions;
  uint32_t sample_count;
  uint32_t sample_image_index;
  uint32_t reference_image_index;
  uint32_t pad;
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
  if ((_private->rhi_context == nullptr) || (_private->rhi_context->get_device() == nullptr)) {
    return;
  }

  _private->rhi_context->create_swapchain(native_window, static_cast<uint32_t>(sapp_width()), static_cast<uint32_t>(sapp_height()));
  _private->rhi_device = _private->rhi_context->get_device();

  static RHIImGuiDesc imgui_desc = {
    .color_format = _private->rhi_context->get_swapchain_format(),
    .ini_filename = env().file_in_data("ui.ini"),
  };
  _private->rhi_imgui.setup(_private->rhi_context, imgui_desc);

  ShaderCompiler* compiler = ShaderCompiler::get_global_instance();
  auto vs = compiler->load_and_compile_shader_from_file("shaders/render.hlsl", "vertex_main", RHIShaderStage::Vertex);
  auto fs = compiler->load_and_compile_shader_from_file("shaders/render.hlsl", "fragment_main", RHIShaderStage::Fragment);
  RHIGraphicsPipelineDesc pipeline_desc = {
    .vertex_shader =
      {
        .spirv_data = vs.spirv_data.data(),
        .spirv_size = vs.spirv_data.size(),
        .stage = RHIShaderStage::Vertex,
        .entry_point = "vertex_main",
      },
    .fragment_shader =
      {
        .spirv_data = fs.spirv_data.data(),
        .spirv_size = fs.spirv_data.size(),
        .stage = RHIShaderStage::Fragment,
        .entry_point = "fragment_main",
      },
    .color_attachment_count = 1,
    .color_formats = {_private->rhi_context->get_swapchain_format()},
  };
  _private->presentation_pipeline = _private->rhi_device->create_graphics_pipeline(pipeline_desc).handle;
}

void RenderContext::cleanup() {
  if (_private->rhi_context == nullptr)
    return;

  auto vk_context = static_cast<VKContext*>(_private->rhi_context);
  if (vk_context != nullptr) {
    VkDevice vk_device = vk_context->get_vk_device();
    if (vk_device != VK_NULL_HANDLE) {
      vkDeviceWaitIdle(vk_device);
    }
  }

  _private->rhi_imgui.shutdown();

  if (_private->rhi_device) {
    _private->rhi_device->destroy_pipeline(_private->presentation_pipeline);
    if (_private->reference_texture.valid()) {
      _private->rhi_device->destroy_texture(_private->reference_texture);
    }
  }
  _private->image_pool.remove(_private->reference_image_handle);
  _private->image_pool.cleanup();

  RHIContext::release(_private->rhi_context);
  _private->rhi_context = nullptr;
  _private->rhi_device = nullptr;
  _private->rhi_cmd = {};
}

void RenderContext::set_reference_image(const char* file_name) {
  _private->image_pool.remove(_private->reference_image_handle);
  if (_private->rhi_context == nullptr)
    return;

  _private->reference_image_handle = _private->image_pool.add_from_file(file_name, 0, {}, {1.0f, 1.0f});
  _private->image_pool.load_images(_private->scheduler);
  _private->apply_reference_image(_private->rhi_context, _private->reference_image_handle);
}

void RenderContext::set_reference_image(const float4 data[], const uint2 dimensions) {
  _private->image_pool.remove(_private->reference_image_handle);
  if (_private->rhi_context == nullptr)
    return;

  _private->reference_image_handle = _private->image_pool.add_from_data(data, dimensions, 0u, {}, {1.0f, 1.0f});
  _private->image_pool.load_images(_private->scheduler);
  _private->apply_reference_image(_private->rhi_context, _private->reference_image_handle);
}

void RenderContext::start_frame(Renderer* renderer, SceneRepresentation& scene, const FrameData& frame_data) {
  if (_private->rhi_context == nullptr)
    return;

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

  _private->rhi_context->begin_frame();
  render_frame_data.cmd = _private->rhi_context->get_command_buffer();
  _private->rhi_cmd = render_frame_data.cmd;
  _private->rhi_context->command_buffer_begin(_private->rhi_cmd);
  renderer->render(_private->rhi_context, scene, render_frame_data);
  _private->active_renderer = renderer;
  _private->frame_data = frame_data;
}

void RenderContext::end_frame() {
  ETX_PROFILER_SCOPE();

  if (_private->rhi_context == nullptr)
    return;

  RHITexture swapchain_image = _private->rhi_context->get_current_swapchain_texture();
  _private->rhi_context->cmd_begin_render_pass(_private->rhi_cmd, 1, &swapchain_image);

  RHITexture output = _private->active_renderer ? _private->active_renderer->output_texture() : RHITexture{};

  if (output.valid()) {
    const RHIViewport viewport = {.width = float(sapp_width()), .height = float(sapp_height())};
    const RHIRect scissor = {.width = uint32_t(sapp_width()), .height = uint32_t(sapp_height())};
    _private->rhi_context->cmd_set_viewport(_private->rhi_cmd, viewport);
    _private->rhi_context->cmd_set_scissor(_private->rhi_cmd, scissor);

    uint2 output_size = _private->active_renderer->output_size();
    RenderParameters params = {
      .view = _private->frame_data.view_parameters,
      .dimensions = {float(sapp_width()), float(sapp_height()), float(output_size.x), float(output_size.y)},
      .sample_count = _private->frame_data.sample_count,
      .sample_image_index = get_bindless_descriptor_index(output),
      .reference_image_index = _private->reference_texture.valid() ? get_bindless_descriptor_index(_private->reference_texture) : ~0u,
    };

    _private->rhi_context->cmd_set_pipeline(_private->rhi_cmd, _private->presentation_pipeline);
    _private->rhi_context->cmd_push_constants(_private->rhi_cmd, &params, sizeof(params));
    _private->rhi_context->cmd_draw(_private->rhi_cmd, {.vertex_count = 3});
  }

  _private->rhi_imgui.render(_private->rhi_cmd);
  _private->rhi_context->cmd_end_render_pass(_private->rhi_cmd);
  _private->rhi_context->command_buffer_end(_private->rhi_cmd);
  auto acquired_sem = _private->rhi_context->get_image_acquired_semaphore();
  auto render_complete_sem = _private->rhi_context->get_render_complete_semaphore();

  RHISubmitInfo submit_info = {};
  submit_info.command_buffer = _private->rhi_cmd;
  submit_info.wait_semaphores.push_back(acquired_sem);
  submit_info.signal_semaphores.push_back(render_complete_sem);

  _private->rhi_context->submit_command_buffer(submit_info);
  _private->rhi_context->present();
  _private->rhi_cmd = {};
}

RHIImGui& RenderContext::rhi_ui() {
  return _private->rhi_imgui;
}

}  // namespace etx
