#include <etx/core/profiler.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/shared/base.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/rhi/vulkan/vk_rhi.hxx>
#include <sokol_app.h>
#include <vulkan/vulkan.h>

#include "render.hxx"

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
  RHICommandBuffer* rhi_cmd = nullptr;
  RHIDevice* rhi_device = nullptr;
  RHIPipeline rhi_pipeline = {};
  RHITexture rhi_output_texture = {};
  RHITexture rhi_reference_texture = {};
  bool rhi_pipeline_ready = false;

  RenderParameters render_params = {};
  uint32_t def_image_handle = kInvalidIndex;
  uint32_t ref_image_handle = kInvalidIndex;
  uint2 output_dimensions = {};

  std::vector<Image> images;
  std::vector<ImageStorage> images_storage;
  ImagePool image_pool;

  std::vector<float4> black_image;
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
  constexpr float4 kBlack = {};
  _private->image_pool.init(1024u);
  _private->def_image_handle = _private->image_pool.add_from_data(&kBlack, {1u, 1u}, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
  _private->image_pool.load_images(_private->scheduler);

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

  ShaderCompiler* compiler = ShaderCompiler::get_global_instance();
  auto vs = compiler->load_and_compile_shader_from_file("shaders/display.hlsl", "vertex_main", RHIShaderStage::Vertex);
  auto fs = compiler->load_and_compile_shader_from_file("shaders/display.hlsl", "fragment_main", RHIShaderStage::Fragment);
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
  _private->rhi_pipeline = _private->rhi_device->create_graphics_pipeline(pipeline_desc).handle;
  _private->rhi_pipeline_ready = true;

  apply_reference_image(_private->def_image_handle);
}

void RenderContext::cleanup(std::function<void()> clean_resources) {
  if (_private->rhi_context != nullptr) {
    auto vk_context = static_cast<VKContext*>(_private->rhi_context);
    if (vk_context != nullptr) {
      VkDevice vk_device = vk_context->get_vk_device();
      if (vk_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(vk_device);
      }
    }
  }

  clean_resources();

  if (_private->rhi_device) {
    if (_private->rhi_output_texture)
      _private->rhi_device->destroy_texture(_private->rhi_output_texture);
    if (_private->rhi_reference_texture)
      _private->rhi_device->destroy_texture(_private->rhi_reference_texture);
    if (_private->rhi_pipeline.valid())
      _private->rhi_device->destroy_pipeline(_private->rhi_pipeline);
  }

  if (_private->rhi_context) {
    RHIContext::release(_private->rhi_context);
    _private->rhi_context = nullptr;
    _private->rhi_device = nullptr;
    _private->rhi_cmd = nullptr;
  }
  _private->rhi_pipeline_ready = false;
  _private->image_pool.remove(_private->ref_image_handle);
  _private->image_pool.remove(_private->def_image_handle);
  _private->image_pool.cleanup();
}

const ViewParameters& RenderContext::view_parameters() const {
  return _private->render_params.view;
}

void RenderContext::begin_frame() {
  ETX_PROFILER_SCOPE();
  if (_private->rhi_context) {
    _private->rhi_context->begin_frame();
    _private->rhi_cmd = _private->rhi_context->get_command_buffer();
    _private->rhi_cmd->begin();
  }
}

RHICommandBuffer* RenderContext::current_command_buffer() {
  return _private->rhi_cmd;
}

void RenderContext::start_frame(uint32_t sample_count, const ViewParameters& view_options) {
  ETX_PROFILER_SCOPE();

  if (_private->rhi_cmd == nullptr || _private->rhi_pipeline_ready == false)
    return;

  RHIViewport viewport = {
    .width = float(sapp_width()),
    .height = float(sapp_height()),
  };
  RHIRect scissor = {
    .width = uint32_t(sapp_width()),
    .height = uint32_t(sapp_height()),
  };

  _private->render_params = {
    .view = view_options,
    .dimensions =
      {
        sapp_widthf(),
        sapp_heightf(),
        float(_private->output_dimensions.x),
        float(_private->output_dimensions.y),
      },
    .sample_count = sample_count,
    .sample_image_index = get_bindless_descriptor_index(_private->rhi_output_texture),
    .reference_image_index = get_bindless_descriptor_index(_private->rhi_reference_texture),
  };

  auto swapchain_texture = _private->rhi_context->get_current_swapchain_texture();
  _private->rhi_cmd->begin_render_pass(1, &swapchain_texture);
  _private->rhi_cmd->set_viewport(viewport);
  _private->rhi_cmd->set_scissor(scissor);
  _private->rhi_cmd->set_pipeline(_private->rhi_pipeline);
  _private->rhi_cmd->push_constants(&_private->render_params, sizeof(_private->render_params));
  _private->rhi_cmd->draw({.vertex_count = 3});
}

void RenderContext::end_frame() {
  ETX_PROFILER_SCOPE();
  if (_private->rhi_context && _private->rhi_cmd) {
    _private->rhi_cmd->end_render_pass();
    _private->rhi_cmd->end();
    _private->rhi_context->submit_command_buffer(_private->rhi_cmd);
    _private->rhi_context->present();
    _private->rhi_cmd = nullptr;
  }
}

void RenderContext::apply_reference_image(uint32_t handle) {
  const auto& img = _private->image_pool.get(handle);
  if (_private->rhi_reference_texture) {
    _private->rhi_device->destroy_texture(_private->rhi_reference_texture);
    _private->rhi_reference_texture = {};
  }
  RHITextureDesc desc = {};
  desc.width = img.isize.x;
  desc.height = img.isize.y;
  desc.format = (img.format == Image::Format::RGBA32F) ? RHITextureFormat::R32G32B32A32_FLOAT : RHITextureFormat::R8G8B8A8_UNORM;
  desc.usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst;
  _private->rhi_reference_texture = _private->rhi_device->create_texture(desc).handle;
  const void* data_ptr = (img.format == Image::Format::RGBA32F) ? (const void*)img.pixels.f32.a : (const void*)img.pixels.u8.a;
  _private->rhi_device->update_texture(_private->rhi_reference_texture, data_ptr, 0, 0);
}

void RenderContext::set_reference_image(const char* file_name) {
  _private->image_pool.remove(_private->ref_image_handle);
  _private->ref_image_handle = _private->image_pool.add_from_file(file_name, 0, {}, {1.0f, 1.0f});
  _private->image_pool.load_images(_private->scheduler);
  apply_reference_image(_private->ref_image_handle);
}

void RenderContext::set_reference_image(const float4 data[], const uint2 dimensions) {
  _private->image_pool.remove(_private->ref_image_handle);
  _private->ref_image_handle = _private->image_pool.add_from_data(data, dimensions, 0u, {}, {1.0f, 1.0f});
  _private->image_pool.load_images(_private->scheduler);
  apply_reference_image(_private->ref_image_handle);
}

void RenderContext::set_output_dimensions(const uint2& dim) {
  if (_private->output_dimensions == dim) {
    return;
  }
  _private->output_dimensions = {std::max(1u, dim.x), std::max(1u, dim.y)};
  if (_private->rhi_output_texture) {
    _private->rhi_device->destroy_texture(_private->rhi_output_texture);
    _private->rhi_output_texture = {};
  }
  RHITextureDesc desc = {
    .width = _private->output_dimensions.x,
    .height = _private->output_dimensions.y,
    .format = RHITextureFormat::R32G32B32A32_FLOAT,
    .usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst | RHITextureUsage::Storage,
  };
  _private->rhi_output_texture = _private->rhi_device->create_texture(desc).handle;
  _private->black_image.resize(_private->output_dimensions.x * _private->output_dimensions.y);
  std::fill(_private->black_image.begin(), _private->black_image.end(), float4{});
}

uint2 RenderContext::get_output_dimensions() const {
  return _private->output_dimensions;
}

RHITexture RenderContext::get_output_texture() const {
  return _private->rhi_output_texture;
}

void RenderContext::update_image(const float4* camera) {
  ETX_PROFILER_SCOPE();
  if (_private->rhi_output_texture) {
    const void* data_ptr = camera ? camera : _private->black_image.data();
    _private->rhi_device->update_texture(_private->rhi_output_texture, data_ptr, 0, 0);
  }
}

}  // namespace etx
