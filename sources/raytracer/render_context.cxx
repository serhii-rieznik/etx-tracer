#include <etx/render/interop/interop.hxx>

#include <etx/core/profiler.hxx>
#include <etx/core/environment.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/buffer_pool.hxx>
#include <etx/render/host/image_pool.hxx>

#include <etx/rhi/rhi.hxx>
#include <etx/rhi/rhi_imgui.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include <sokol_app.h>

#include "render_context.hxx"
#include "renderer.hxx"

#include <vector>
#include <algorithm>

namespace etx {

struct RenderContextImpl {
  RenderContextImpl(TaskScheduler& s)
    : scheduler(s)
    , image_pool(images, buffer_pool) {
  }

  TaskScheduler& scheduler;
  RHIContext rhi_context = {};
  RHIImGui rhi_imgui = {};
  RHICommandBuffer rhi_cmd = {};
  RHIPipeline presentation_pipeline = {};
  Renderer* active_renderer = nullptr;
  RenderContext::FrameData frame_data = {};

  RHITexture reference_texture = {};
  std::vector<Image> images;
  BufferPool buffer_pool;
  ImagePool image_pool;
  uint32_t reference_image_handle = kInvalidIndex;

  void apply_reference_image(RHIContext& ctx, uint32_t handle) {
    ETX_PROFILER_SCOPE();

    const auto& img = image_pool.get(handle);
    if (reference_texture.valid()) {
      ETX_PROFILER_NAMED_SCOPE("render_context_destroy_reference_texture");
      ctx.device().destroy_texture(reference_texture);
      reference_texture = {};
    }
    RHITextureDesc desc = {
      .width = img.isize.x,
      .height = img.isize.y,
      .format = (img.format == Image::Format::RGBA32F) ? RHITextureFormat::R32G32B32A32_FLOAT : RHITextureFormat::R8G8B8A8_UNORM,
      .usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst,
    };
    ETX_PROFILER_NAMED_SCOPE("render_context_create_and_upload_reference_texture");
    reference_texture = ctx.device().create_texture(desc).handle;
    const void* data_ptr = (img.format == Image::Format::RGBA32F) ? (const void*)img.pixels.f32.a : (const void*)img.pixels.u8.a;
    ctx.device().update_texture(reference_texture, data_ptr, 0, 0);
  }
};

ETX_IMPLEMENT_PIMPL(RenderContext);

RenderContext::RenderContext(TaskScheduler& s) {
  ETX_PIMPL_INIT(RenderContext, s);
}

RenderContext::~RenderContext() {
  ETX_PIMPL_CLEANUP(RenderContext);
}

RHIContext& RenderContext::get_context() {
  ETX_ASSERT(_private->rhi_context.valid());
  return _private->rhi_context;
}

RHIDevice& RenderContext::get_device() {
  ETX_ASSERT(_private->rhi_context.valid());
  return _private->rhi_context.device();
}

RHITextureFormat RenderContext::get_swapchain_format() {
  ETX_ASSERT(_private->rhi_context.valid());
  return _private->rhi_context.get_swapchain_format();
}

RHITextureFormat RenderContext::get_depth_format() {
  return RHITextureFormat::D32_FLOAT;
}

void RenderContext::init() {
  ETX_PROFILER_SCOPE();

  RHIBackend backend = RHIBackend::Vulkan;
#if defined(ETX_PLATFORM_APPLE)
  backend = RHIBackend::Metal;
#endif

  RHIInitInfo info = {
    .backend = backend,
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

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_create_context");
    _private->rhi_context = RHIContext::create(info);
  }
  if (_private->rhi_context.valid() == false) {
    return;
  }

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_create_swapchain");
    _private->rhi_context.create_swapchain(native_window, static_cast<uint32_t>(sapp_width()), static_cast<uint32_t>(sapp_height()));
  }

  static RHIImGuiDesc imgui_desc = {
    .color_format = _private->rhi_context.get_swapchain_format(),
    .ini_filename = env().file_in_data("ui.ini"),
  };
  {
    ETX_PROFILER_NAMED_SCOPE("render_context_setup_imgui");
    _private->rhi_imgui.setup(_private->rhi_context, imgui_desc);
  }

  auto& compiler = ShaderCompiler::instance();

  ShaderCompiler::MultiShaderCompilationResult result = {};
  {
    ETX_PROFILER_NAMED_SCOPE("render_context_compile_presentation_shader");
    result = compiler.compile("shaders/render.hlsl", {{"vertex_main", RHIShaderStage::Vertex}, {"fragment_main", RHIShaderStage::Fragment}});
  }

  if (result.result != RHIResult::Success) {
    log::error("Failed to compile render shader: %s", result.error_message.c_str());
    return;
  }

  RHIGraphicsPipelineDesc pipeline_desc = {
    .color_attachment_count = 1,
    .color_formats = {_private->rhi_context.get_swapchain_format()},
  };

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_create_presentation_pipeline");
    _private->presentation_pipeline = _private->rhi_context.device().create_graphics_pipeline(pipeline_desc, result.binaries[0], result.binaries[1]).handle;
  }
}

void RenderContext::cleanup() {
  ETX_PROFILER_SCOPE();

  if (_private->rhi_context.valid() == false)
    return;

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_wait_idle");
    _private->rhi_context.wait_idle();
  }

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_shutdown_imgui");
    _private->rhi_imgui.shutdown();
  }

  auto& device = _private->rhi_context.device();
  {
    ETX_PROFILER_NAMED_SCOPE("render_context_destroy_resources");
    device.destroy_pipeline(_private->presentation_pipeline);
    if (_private->reference_texture.valid()) {
      device.destroy_texture(_private->reference_texture);
    }
    _private->image_pool.remove(_private->reference_image_handle);
    _private->image_pool.cleanup();
  }

  _private->rhi_context = {};
  _private->rhi_cmd = {};
}

void RenderContext::set_reference_image(const char* file_name) {
  ETX_PROFILER_SCOPE();

  _private->image_pool.remove(_private->reference_image_handle);
  if (_private->rhi_context.valid() == false)
    return;

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_add_reference_image_file");
    _private->reference_image_handle = _private->image_pool.add_from_file(file_name, 0, {}, {1.0f, 1.0f});
  }
  {
    ETX_PROFILER_NAMED_SCOPE("render_context_load_reference_image_file");
    _private->image_pool.load_images(_private->scheduler);
  }
  {
    ETX_PROFILER_NAMED_SCOPE("render_context_apply_reference_image_file");
    _private->apply_reference_image(_private->rhi_context, _private->reference_image_handle);
  }
}

void RenderContext::set_reference_image(const float4 data[], const uint2 dimensions) {
  ETX_PROFILER_SCOPE();

  _private->image_pool.remove(_private->reference_image_handle);
  if (_private->rhi_context.valid() == false)
    return;

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_add_reference_image_data");
    _private->reference_image_handle = _private->image_pool.add_from_data(data, dimensions, 0u, {}, {1.0f, 1.0f});
  }
  {
    ETX_PROFILER_NAMED_SCOPE("render_context_load_reference_image_data");
    _private->image_pool.load_images(_private->scheduler);
  }
  {
    ETX_PROFILER_NAMED_SCOPE("render_context_apply_reference_image_data");
    _private->apply_reference_image(_private->rhi_context, _private->reference_image_handle);
  }
}

void RenderContext::start_frame(Renderer* renderer, SceneRepresentation& scene, const FrameData& frame_data) {
  ETX_PROFILER_SCOPE();

  if (_private->rhi_context.valid() == false)
    return;

  etx::RHIImGuiFrameDesc imgui_frame_desc = {
    .width = uint32_t(sapp_width()),
    .height = uint32_t(sapp_height()),
    .delta_time = frame_data.dt,
    .dpi_scale = sapp_dpi_scale(),
  };
  {
    ETX_PROFILER_NAMED_SCOPE("render_context_imgui_new_frame");
    _private->rhi_imgui.new_frame(imgui_frame_desc);
  }

  Renderer::FrameData render_frame_data = {
    .view_parameters = frame_data.view_parameters,
    .dt = frame_data.dt,
  };

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_renderer_render");
    _private->rhi_context.begin_frame();
    render_frame_data.cmd = _private->rhi_context.get_command_buffer();
    _private->rhi_cmd = render_frame_data.cmd;
    _private->rhi_context.command_buffer_begin(_private->rhi_cmd);
    renderer->render(_private->rhi_context, scene, render_frame_data);
  }
  _private->active_renderer = renderer;
  _private->frame_data = frame_data;
}

void RenderContext::end_frame() {
  ETX_PROFILER_SCOPE();

  if (_private->rhi_context.valid() == false)
    return;

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_begin_present_pass");
    float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    RHITexture swapchain_image = _private->rhi_context.get_current_swapchain_texture();
    _private->rhi_context.cmd_begin_render_pass(_private->rhi_cmd, 1, &swapchain_image, clear_color);
  }

  RHITexture output = _private->active_renderer ? _private->active_renderer->output_texture() : RHITexture{};

  if (output.valid()) {
    ETX_PROFILER_NAMED_SCOPE("render_context_draw_present_quad");
    const RHIViewport viewport = {.width = float(sapp_width()), .height = float(sapp_height())};
    const RHIRect scissor = {.width = uint32_t(sapp_width()), .height = uint32_t(sapp_height())};
    _private->rhi_context.cmd_set_viewport(_private->rhi_cmd, viewport);
    _private->rhi_context.cmd_set_scissor(_private->rhi_cmd, scissor);

    uint2 output_size = _private->active_renderer->output_size();
    RenderParameters params = {
      .view = _private->frame_data.view_parameters,
      .dimensions = {float(sapp_width()), float(sapp_height()), float(output_size.x), float(output_size.y)},
      .sample_count = _private->frame_data.sample_count,
      .sample_image_index = get_bindless_descriptor_index(output),
      .reference_image_index = _private->reference_texture.valid() ? get_bindless_descriptor_index(_private->reference_texture) : ~0u,
    };

    _private->rhi_context.cmd_set_pipeline(_private->rhi_cmd, _private->presentation_pipeline);
    _private->rhi_context.cmd_push_constants(_private->rhi_cmd, &params, sizeof(params));
    _private->rhi_context.cmd_draw(_private->rhi_cmd, {.vertex_count = 3});
  }

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_imgui_render");
    _private->rhi_imgui.render(_private->rhi_cmd);
  }
  {
    ETX_PROFILER_NAMED_SCOPE("render_context_submit_and_present");
    _private->rhi_context.cmd_end_render_pass(_private->rhi_cmd);
    _private->rhi_context.command_buffer_end(_private->rhi_cmd);
    auto acquired_sem = _private->rhi_context.get_image_acquired_semaphore();
    auto render_complete_sem = _private->rhi_context.get_render_complete_semaphore();

    RHISubmitInfo submit_info = {};
    submit_info.command_buffer = _private->rhi_cmd;
    submit_info.wait_semaphores.push_back(acquired_sem);
    submit_info.signal_semaphores.push_back(render_complete_sem);

    _private->rhi_context.submit_command_buffer(submit_info);
    _private->rhi_context.present();
  }
  _private->rhi_cmd = {};
}

RHIImGui& RenderContext::rhi_ui() {
  return _private->rhi_imgui;
}

}  // namespace etx
