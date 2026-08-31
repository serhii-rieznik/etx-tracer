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
#include <cstring>
#include <limits>
#include <stb_image_write.hxx>
namespace etx {
namespace {

RHIBackend select_default_backend() {
#if ETX_PLATFORM_APPLE
  return RHIBackend::Metal;
#else
  return RHIBackend::Vulkan;
#endif
}

const char* backend_name(RHIBackend backend) {
  switch (backend) {
    case RHIBackend::Vulkan:
      return "Vulkan";
    case RHIBackend::Metal:
      return "Metal";
    default:
      return "Unknown";
  }
}

bool read_output_texture(RHIContext& rhi, RHITexture texture, const RHIExtent2D& extent, RHITextureFormat format, std::vector<uint8_t>& output) {
  output.clear();
  if ((texture.valid() == false) || (extent.width == 0u) || (extent.height == 0u)) {
    return false;
  }
  if ((format != RHITextureFormat::B8G8R8A8_UNORM) && (format != RHITextureFormat::R8G8B8A8_UNORM)) {
    log::error("Output capture does not support texture format %u", static_cast<uint32_t>(format));
    return false;
  }

  const uint64_t pixel_count = static_cast<uint64_t>(extent.width) * static_cast<uint64_t>(extent.height);
  if (pixel_count > (std::numeric_limits<uint64_t>::max() / 4u)) {
    return false;
  }
  const uint64_t buffer_size = pixel_count * 4u;
  if (buffer_size > std::numeric_limits<size_t>::max()) {
    return false;
  }
  const RHIBufferDesc readback_desc = {
    .size = buffer_size,
    .usage = RHIBufferUsage::TransferDst,
    .host_visible = true,
  };
  const RHICreateBindlessResult buffer_result = rhi.device().create_buffer(readback_desc);
  if ((buffer_result.result != RHIResult::Success) || (buffer_result.handle.valid() == false)) {
    return false;
  }

  rhi.begin_frame();
  const RHICommandBuffer cmd = rhi.get_command_buffer();
  if (cmd.valid() == false) {
    rhi.end_frame();
    rhi.device().destroy_buffer(buffer_result.handle);
    return false;
  }

  rhi.command_buffer_begin(cmd);
  rhi.cmd_texture_barrier(cmd, texture, RHIResourceState::ColorAttachment, RHIResourceState::TransferSrc);
  rhi.cmd_copy_texture_to_buffer(cmd, texture, buffer_result.handle, extent.width, extent.height);
  rhi.cmd_texture_barrier(cmd, texture, RHIResourceState::TransferSrc, RHIResourceState::ColorAttachment);
  rhi.command_buffer_end(cmd);
  rhi.submit_command_buffer({cmd});
  rhi.end_frame();

  const RHIResult wait_result = rhi.wait_idle();
  if (wait_result != RHIResult::Success) {
    rhi.destroy_command_buffer(cmd);
    rhi.device().destroy_buffer(buffer_result.handle);
    return false;
  }
  rhi.destroy_command_buffer(cmd);

  output.resize(static_cast<size_t>(buffer_size));
  const RHIResult read_result = rhi.device().read_buffer(buffer_result.handle, output.data(), buffer_size);
  rhi.device().destroy_buffer(buffer_result.handle);
  if (read_result != RHIResult::Success) {
    output.clear();
    return false;
  }

  if (format == RHITextureFormat::B8G8R8A8_UNORM) {
    for (size_t pixel = 0u; pixel < output.size(); pixel += 4u) {
      std::swap(output[pixel + 0u], output[pixel + 2u]);
    }
  }
  return true;
}

void append_png_data(void* context, void* data, int size) {
  if ((context == nullptr) || (data == nullptr) || (size <= 0)) {
    return;
  }
  auto& output = *reinterpret_cast<std::vector<uint8_t>*>(context);
  const size_t offset = output.size();
  output.resize(offset + static_cast<size_t>(size));
  std::memcpy(output.data() + offset, data, static_cast<size_t>(size));
}

}  // namespace

struct RenderContextImpl {
  RenderContextImpl(TaskScheduler& s)
    : scheduler(s)
    , image_pool(images, buffer_pool) {
  }

  TaskScheduler& scheduler;
  RHIContext rhi_context = {};
  RuntimeOutput runtime_output = {};
  RHIImGui rhi_imgui = {};
  RHIImGuiTheme ui_theme = RHIImGuiTheme::Dark;
  RenderContextConfig config = {};
  bool initialized = false;
  bool imgui_enabled = false;
  RHICommandBuffer rhi_cmd = {};
  RHIPipeline presentation_pipeline = {};
  Renderer* active_renderer = nullptr;
  RenderContext::FrameData frame_data = {};
  RenderContext::PresentationViewport presentation_viewport = {};

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

bool RenderContext::valid() const {
  return _private->initialized && _private->rhi_context.valid();
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
  return _private->runtime_output.output_format();
}

RHITextureFormat RenderContext::get_depth_format() {
  return RHITextureFormat::D32_FLOAT;
}

RuntimeMode RenderContext::runtime_mode() const {
  return _private->runtime_output.mode();
}

void RenderContext::set_ui_theme(RHIImGuiTheme theme) {
  _private->ui_theme = theme;
  if (_private->imgui_enabled) {
    _private->rhi_imgui.set_theme(theme);
  }
}

bool RenderContext::imgui_enabled() const {
  return _private->imgui_enabled;
}

bool RenderContext::capture_output_png(std::vector<uint8_t>& png_data, uint32_t& width, uint32_t& height) {
  png_data.clear();
  width = 0u;
  height = 0u;
  if (!valid() || (_private->runtime_output.mode() == RuntimeMode::Desktop)) {
    return false;
  }

  const RHIExtent2D extent = _private->runtime_output.extent();
  if ((extent.width > static_cast<uint32_t>(std::numeric_limits<int>::max())) ||
      (extent.height > static_cast<uint32_t>(std::numeric_limits<int>::max())) ||
      (extent.width > static_cast<uint32_t>(std::numeric_limits<int>::max() / 4))) {
    return false;
  }
  std::vector<uint8_t> pixels = {};
  if (read_output_texture(_private->rhi_context, _private->runtime_output.offscreen_texture(), extent, _private->runtime_output.output_format(), pixels) == false) {
    return false;
  }
  if (stbi_write_png_to_func(append_png_data, &png_data, static_cast<int>(extent.width), static_cast<int>(extent.height), 4, pixels.data(),
        static_cast<int>(extent.width * 4u)) == 0) {
    png_data.clear();
    return false;
  }
  width = extent.width;
  height = extent.height;
  return true;
}

void RenderContext::init(const RenderContextConfig& config) {
  ETX_PROFILER_SCOPE();

  _private->initialized = false;
  _private->config = config;
  RHIBackend backend = select_default_backend();

  RHIInitInfo info = {
    .backend = backend,
    .enable_validation = ETX_DEBUG,
    .headless = config.mode != RuntimeMode::Desktop,
  };
  const void* native_window = config.native_window;
  if ((config.mode == RuntimeMode::Desktop) && (native_window == nullptr)) {
#if ETX_PLATFORM_WINDOWS
    native_window = sapp_win32_get_hwnd();
#elif defined(__APPLE__)
    native_window = sapp_macos_get_window();
#endif
  }
  if ((config.mode == RuntimeMode::Desktop) && (native_window == nullptr))
    return;

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_create_context");
    _private->rhi_context = RHIContext::create(info);
  }
  if (_private->rhi_context.valid() == false) {
    return;
  }

  RuntimeOutputConfig runtime_output_config = {
    .mode = config.mode,
    .width = config.mode == RuntimeMode::Desktop ? static_cast<uint32_t>(sapp_width()) : config.width,
    .height = config.mode == RuntimeMode::Desktop ? static_cast<uint32_t>(sapp_height()) : config.height,
    .native_window = native_window,
  };
  if (_private->runtime_output.init(_private->rhi_context, runtime_output_config) == false) {
    return;
  }

  const RHICapabilities capabilities = _private->rhi_context.capabilities();
  log::info("RenderContext RHI backend: %s (swapchain=%u, bindless=%u, timestamps=%u, ray_tracing=%u, ray_traversal_class=%u)", backend_name(backend),
    static_cast<uint32_t>(capabilities.supports_swapchain), static_cast<uint32_t>(capabilities.supports_bindless), static_cast<uint32_t>(capabilities.supports_timestamps),
    static_cast<uint32_t>(capabilities.supports_ray_tracing), static_cast<uint32_t>(capabilities.ray_traversal_class));

  _private->imgui_enabled = config.enable_imgui && _private->runtime_output.imgui_supported();
  if (_private->imgui_enabled) {
    const RHIImGuiDesc imgui_desc = {
      .color_format = _private->runtime_output.output_format(),
      .ini_filename = env().file_in_user_data("ui.ini"),
    };
    {
      ETX_PROFILER_NAMED_SCOPE("render_context_setup_imgui");
      const RHIResult setup_result = _private->rhi_imgui.setup(_private->rhi_context, imgui_desc);
      if (setup_result != RHIResult::Success) {
        log::error("Failed to initialize ImGui rendering (%u)", static_cast<uint32_t>(setup_result));
        return;
      }
    }
  }

  auto& compiler = ShaderCompiler::instance();

  ShaderCompiler::MultiShaderCompilationResult result = {};
  {
    ETX_PROFILER_NAMED_SCOPE("render_context_compile_presentation_shader");
    std::unordered_map<std::string, std::string> defines = {};
    if (rhi_texture_format_is_srgb(_private->runtime_output.output_format())) {
      defines.emplace("ETX_PRESENT_SRGB_TARGET", "1");
    }
    result = compiler.compile("shaders/render.hlsl", {{"vertex_main", RHIShaderStage::Vertex}, {"fragment_main", RHIShaderStage::Fragment}}, defines, backend);
  }

  if (result.result != RHIResult::Success) {
    log::error("Failed to compile render shader: %s", result.error_message.c_str());
    return;
  }
  if (result.binaries.size() != 2u) {
    log::error("Expected 2 presentation shader binaries, got %zu", result.binaries.size());
    return;
  }

  RHIGraphicsPipelineDesc pipeline_desc = {
    .color_attachment_count = 1,
    .color_formats = {_private->runtime_output.output_format()},
  };

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_create_presentation_pipeline");
    const RHICreatePipelineResult pipeline_result =
      _private->rhi_context.device().create_graphics_pipeline(pipeline_desc, result.binaries[0], result.binaries[1]);
    if ((pipeline_result.result != RHIResult::Success) || !pipeline_result.handle.valid()) {
      log::error("Failed to create presentation pipeline (%u)", static_cast<uint32_t>(pipeline_result.result));
      return;
    }
    _private->presentation_pipeline = pipeline_result.handle;
  }
  _private->initialized = true;
}

void RenderContext::cleanup() {
  cleanup(false);
}

void RenderContext::cleanup(bool device_already_idle) {
  ETX_PROFILER_SCOPE();

  if (_private->rhi_context.valid() == false)
    return;

  _private->initialized = false;

  if (device_already_idle == false) {
    ETX_PROFILER_NAMED_SCOPE("render_context_wait_idle");
    _private->rhi_context.wait_idle();
  }

  if (_private->imgui_enabled) {
    ETX_PROFILER_NAMED_SCOPE("render_context_shutdown_imgui");
    _private->rhi_imgui.shutdown();
    _private->imgui_enabled = false;
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

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_runtime_output_shutdown");
    _private->runtime_output.shutdown(_private->rhi_context);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_destroy_rhi_context");
    _private->rhi_context = {};
    _private->rhi_cmd = {};
    _private->presentation_pipeline = {};
    _private->active_renderer = nullptr;
  }
}

void RenderContext::set_reference_image(const char* file_name) {
  ETX_PROFILER_SCOPE();

  _private->image_pool.remove(_private->reference_image_handle);
  if (!valid())
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
  if (!valid())
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

  if (!valid())
    return;

  if (_private->config.mode == RuntimeMode::Desktop) {
    const uint32_t framebuffer_width = static_cast<uint32_t>(sapp_width());
    const uint32_t framebuffer_height = static_cast<uint32_t>(sapp_height());
    const RHIExtent2D output_extent = _private->runtime_output.extent();
    if ((framebuffer_width != output_extent.width) || (framebuffer_height != output_extent.height)) {
      _private->runtime_output.resize(_private->rhi_context, framebuffer_width, framebuffer_height);
    }
  } else if ((_private->config.mode == RuntimeMode::Headless) && (renderer != nullptr)) {
    const uint2 renderer_size = renderer->output_size();
    const RHIExtent2D output_extent = _private->runtime_output.extent();
    if ((renderer_size.x > 0u) && (renderer_size.y > 0u) &&
        ((renderer_size.x != output_extent.width) || (renderer_size.y != output_extent.height))) {
      _private->runtime_output.resize(_private->rhi_context, renderer_size.x, renderer_size.y);
    }
  }

  if (_private->imgui_enabled) {
    const RHIExtent2D extent = _private->runtime_output.extent();
    etx::RHIImGuiFrameDesc imgui_frame_desc = {
      .width = extent.width,
      .height = extent.height,
      .delta_time = frame_data.dt,
      .dpi_scale = _private->config.mode == RuntimeMode::Desktop ? sapp_dpi_scale() : _private->config.dpi_scale,
    };
    ETX_PROFILER_NAMED_SCOPE("render_context_imgui_new_frame");
    _private->rhi_imgui.new_frame(imgui_frame_desc);
  }

  Renderer::FrameData render_frame_data = {
    .view_parameters = frame_data.view_parameters,
    .dt = frame_data.dt,
  };

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_renderer_render");
    _private->runtime_output.begin_frame(_private->rhi_context);
    render_frame_data.cmd = _private->rhi_context.get_command_buffer();
    _private->rhi_cmd = render_frame_data.cmd;
    _private->rhi_context.command_buffer_begin(_private->rhi_cmd);
    if (renderer != nullptr) {
      renderer->render(_private->rhi_context, scene, render_frame_data);
    }
  }
  _private->active_renderer = renderer;
  _private->frame_data = frame_data;
}

void RenderContext::set_presentation_viewport(const PresentationViewport& viewport) {
  _private->presentation_viewport = viewport;
}

void RenderContext::end_frame() {
  ETX_PROFILER_SCOPE();

  if (!valid())
    return;

  const RuntimeOutputTarget output_target = _private->runtime_output.acquire_target(_private->rhi_context);
  if ((output_target.valid() == false) || (output_target.width == 0u) || (output_target.height == 0u)) {
    _private->rhi_context.command_buffer_end(_private->rhi_cmd);
    _private->runtime_output.submit_frame(_private->rhi_context, _private->rhi_cmd);
    _private->rhi_cmd = {};
    return;
  }

  {
    ETX_PROFILER_NAMED_SCOPE("render_context_begin_present_pass");
    constexpr float light_canvas = 0.68f;
    constexpr float dark_canvas = 0.03f;
    const float canvas = _private->ui_theme == RHIImGuiTheme::Light ? light_canvas : dark_canvas;
    float clear_color[4] = {canvas, canvas, canvas, 1.0f};
    RHITexture output_texture = output_target.texture;
    _private->rhi_context.cmd_begin_render_pass(_private->rhi_cmd, 1, &output_texture, clear_color);
  }

  RHITexture output = _private->active_renderer ? _private->active_renderer->display_texture() : RHITexture{};

  if (output.valid()) {
    ETX_PROFILER_NAMED_SCOPE("render_context_draw_present_quad");
    PresentationViewport presentation_viewport = _private->presentation_viewport;
    if (presentation_viewport.valid == false) {
      presentation_viewport = {
        .width = output_target.width,
        .height = output_target.height,
        .valid = true,
      };
    }

    int32_t clamped_x = std::clamp(presentation_viewport.x, 0, static_cast<int32_t>(output_target.width));
    int32_t clamped_y = std::clamp(presentation_viewport.y, 0, static_cast<int32_t>(output_target.height));
    uint32_t viewport_width = std::min(presentation_viewport.width, output_target.width - static_cast<uint32_t>(clamped_x));
    uint32_t viewport_height = std::min(presentation_viewport.height, output_target.height - static_cast<uint32_t>(clamped_y));
    if ((viewport_width == 0u) || (viewport_height == 0u)) {
      clamped_x = 0;
      clamped_y = 0;
      viewport_width = output_target.width;
      viewport_height = output_target.height;
    }
    const RHIViewport viewport = {
      .x = static_cast<float>(clamped_x),
      .y = static_cast<float>(clamped_y),
      .width = static_cast<float>(viewport_width),
      .height = static_cast<float>(viewport_height),
    };
    const RHIRect scissor = {
      .x = clamped_x,
      .y = clamped_y,
      .width = viewport_width,
      .height = viewport_height,
    };
    _private->rhi_context.cmd_set_viewport(_private->rhi_cmd, viewport);
    _private->rhi_context.cmd_set_scissor(_private->rhi_cmd, scissor);

    uint2 display_size = {presentation_viewport.display_width, presentation_viewport.display_height};
    if ((display_size.x == 0u) || (display_size.y == 0u)) {
      const uint2 output_size = _private->active_renderer->output_size();
      const float output_aspect = static_cast<float>(output_size.x) / static_cast<float>(std::max(1u, output_size.y));
      const float viewport_aspect = static_cast<float>(viewport_width) / static_cast<float>(std::max(1u, viewport_height));
      display_size = {viewport_width, viewport_height};
      if (output_aspect > viewport_aspect) {
        display_size.y = std::max(1u, static_cast<uint32_t>(static_cast<float>(viewport_width) / output_aspect));
      } else {
        display_size.x = std::max(1u, static_cast<uint32_t>(static_cast<float>(viewport_height) * output_aspect));
      }
    }
    RenderParameters params = {
      .view = _private->frame_data.view_parameters,
      .dimensions = {static_cast<float>(viewport_width), static_cast<float>(viewport_height), static_cast<float>(display_size.x), static_cast<float>(display_size.y)},
      .viewport = {static_cast<float>(clamped_x), static_cast<float>(clamped_y), static_cast<float>(viewport_width), static_cast<float>(viewport_height)},
      .sample_count = _private->frame_data.sample_count,
      .sample_image_index = get_bindless_descriptor_index(output),
      .reference_image_index = _private->reference_texture.valid() ? get_bindless_descriptor_index(_private->reference_texture) : ~0u,
    };

    _private->rhi_context.cmd_set_pipeline(_private->rhi_cmd, _private->presentation_pipeline);
    _private->rhi_context.cmd_push_constants(_private->rhi_cmd, &params, sizeof(params));
    _private->rhi_context.cmd_draw(_private->rhi_cmd, {.vertex_count = 3});
  }

  if (_private->imgui_enabled) {
    ETX_PROFILER_NAMED_SCOPE("render_context_imgui_render");
    _private->rhi_imgui.render(_private->rhi_cmd);
  }
  {
    ETX_PROFILER_NAMED_SCOPE("render_context_submit_and_present");
    _private->rhi_context.cmd_end_render_pass(_private->rhi_cmd);
    _private->rhi_context.command_buffer_end(_private->rhi_cmd);
    _private->runtime_output.submit_frame(_private->rhi_context, _private->rhi_cmd);
  }
  _private->rhi_cmd = {};
}

RHIImGui& RenderContext::rhi_ui() {
  return _private->rhi_imgui;
}

}  // namespace etx
