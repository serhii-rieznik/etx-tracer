#include "rhi_test_app.hxx"

#include <imgui.h>
#include <etx/core/log.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/rhi_types.hxx>
#include <etx/rhi/vulkan/vk_rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/render/shared/base.hxx>

#ifdef _WIN32
# define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

#include <sokol_app.h>

#include <vector>
#include <type_traits>
#include <fstream>
#include <cmath>
#include <random>

namespace etx {

RHITestApplication::RHITestApplication() {
  log::info("RHI Test Application created");
}

RHITestApplication::~RHITestApplication() {
  log::info("RHI Test Application destroyed");
}

void RHITestApplication::init() {
  RHIInitInfo init_info = {
    .backend = RHIBackend::Vulkan,
    .enable_validation = true,
    .enable_debug_names = true,
    .max_frames_in_flight = 2,
  };

  rhi_context = create_rhi_context(init_info);
  if (rhi_context == nullptr) {
    return;
  }

  // In windowed mode, create a swapchain for the window
  if (!headless_mode && rhi_context != nullptr) {
    const void* native_window = nullptr;
#if defined(_WIN32)
    native_window = sapp_win32_get_hwnd();
    log::info("Windowed mode: Got native window handle: %p", native_window);
#elif defined(__APPLE__)
    native_window = sapp_macos_get_window();
#endif

    if (native_window != nullptr) {
      rhi_context->create_swapchain(native_window, static_cast<uint32_t>(sapp_width()), static_cast<uint32_t>(sapp_height()));
    }
  }

  if (create_rendering_resources() == false) {
    log::error("Failed to create rendering resources");
    return;
  }

  RHIImGuiDesc desc = {};
  desc.color_format = rhi_context->get_swapchain_format();
  imgui.setup(rhi_context, desc);

  initialized = true;
}

void RHITestApplication::run_headless_test() {
  for (int frame_count = 0; frame_count < 3; ++frame_count) {
    frame();
  }
}

void RHITestApplication::frame() {
  if (initialized == false)
    return;

  if (headless_mode) {
    time += 0.016667f;
  } else {
    time += sapp_frame_duration();
  }

  if (headless_mode == false) {
    RHIImGuiFrameDesc imgui_frame_desc = {
      .width = static_cast<uint32_t>(sapp_width()),
      .height = static_cast<uint32_t>(sapp_height()),
      .delta_time = sapp_frame_duration(),
      .dpi_scale = sapp_dpi_scale(),
    };
    imgui.new_frame(imgui_frame_desc);

    ImGui::Begin("RHI Test Controls");
    ImGui::SliderFloat("Noise Speed", &ui_params.noise_speed, 0.0f, 5.0f);
    ImGui::SliderFloat("Noise Scale", &ui_params.noise_scale, 0.1f, 10.0f);
    ImGui::ColorEdit3("Quad Color", ui_params.color);
    ImGui::Checkbox("Show ImGui Demo", &ui_params.show_demo_window);
    ImGui::End();

    if (ui_params.show_demo_window) {
      ImGui::ShowDemoWindow(&ui_params.show_demo_window);
    }
  }

  rhi_context->begin_frame();
  RHITexture swapchain_texture = rhi_context->get_current_swapchain_texture();
  command_buffer = rhi_context->get_command_buffer();
  if (command_buffer && swapchain_texture) {
    struct ComputePushConstants {
      uint32_t texture_index;
      uint32_t vertex_buffer_index;  // Not used in compute
      uint32_t viewport_width;       // Not used in compute
      uint32_t viewport_height;      // Not used in compute
      uint32_t texture_width;
      uint32_t texture_height;
      uint32_t noise_seed;
      float noise_scale;
    } compute_pc = {
      .texture_index = get_bindless_descriptor_index(noise_texture),
      .texture_width = static_cast<uint32_t>(sapp_width()),
      .texture_height = static_cast<uint32_t>(sapp_height()),
      .noise_seed = static_cast<uint32_t>(time * 1000.0f * ui_params.noise_speed),
      .noise_scale = ui_params.noise_scale,
    };

    struct PushConstants {
      uint32_t texture_index;
      uint32_t vertex_buffer_index;
      uint32_t viewport_width;
      uint32_t viewport_height;
      uint32_t texture_width;
      uint32_t texture_height;
      float color[3];
    } pc = {
      .texture_index = get_bindless_descriptor_index(noise_texture),
      .vertex_buffer_index = vertex_buffer_index,
      .viewport_width = static_cast<uint32_t>(sapp_width()),
      .viewport_height = static_cast<uint32_t>(sapp_height()),
      .texture_width = static_cast<uint32_t>(sapp_width()),
      .texture_height = static_cast<uint32_t>(sapp_height()),
      .color = {ui_params.color[0], ui_params.color[1], ui_params.color[2]},
    };

    RHIViewport viewport = {
      .width = static_cast<float>(sapp_width()),
      .height = static_cast<float>(sapp_height()),
    };

    command_buffer->begin();
    command_buffer->set_pipeline(compute_pipeline);
    command_buffer->push_constants(&compute_pc, sizeof(ComputePushConstants));
    command_buffer->dispatch({32u, 32u, 1u});
    // TODO: In a complete implementation, we'd add a barrier here to ensure compute writes are visible to graphics pipeline

    float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    command_buffer->begin_render_pass(1, &swapchain_texture, clear_color);
    command_buffer->set_pipeline(graphics_pipeline);
    command_buffer->set_viewport(viewport);
    command_buffer->push_constants(&pc, sizeof(PushConstants));
    command_buffer->draw_indexed({.index_count = 24}, index_buffer);

    if (headless_mode == false) {
      imgui.render(command_buffer);
    }

    command_buffer->end_render_pass();
    command_buffer->end();
    rhi_context->submit_command_buffer(command_buffer);
  }

  uint32_t frame_index = rhi_context->get_current_frame_index();
  rhi_context->end_frame();
  if (swapchain_texture != 0) {
    rhi_context->present_with_frame_index(frame_index);
  }
}  // namespace etx

void RHITestApplication::cleanup() {
  log::info("Cleaning up minimal RHI Test Application...");

  // Wait for all GPU work to complete before destroying resources
  // For Vulkan backend, wait for device idle to ensure all command buffers complete
  if (rhi_context != nullptr) {
    // Cast to Vulkan context to access device wait idle
    // Note: This assumes Vulkan backend - Metal would need different handling
    auto vk_context = static_cast<VKContext*>(rhi_context);
    if (vk_context != nullptr) {
      VkDevice vk_device = vk_context->get_vk_device();
      if (vk_device != VK_NULL_HANDLE) {
        VkResult wait_result = vkDeviceWaitIdle(vk_device);
        if (wait_result != VK_SUCCESS) {
          log::warning("Failed to wait for device idle during cleanup: %d", static_cast<int>(wait_result));
        } else {
          log::info("Device wait idle completed");
        }
      }
    }
  }

  imgui.shutdown();

  // Destroy rendering resources
  if (rhi_context != nullptr && rhi_context->get_device() != nullptr) {
    auto device = rhi_context->get_device();

    if (graphics_pipeline.valid()) {
      device->destroy_pipeline(graphics_pipeline);
      graphics_pipeline = {};
    }

    if (compute_pipeline.valid()) {
      device->destroy_pipeline(compute_pipeline);
      compute_pipeline = {};
    }

    if (vertex_shader.valid()) {
      device->destroy_shader(vertex_shader);
      vertex_shader = {};
    }

    if (fragment_shader.valid()) {
      device->destroy_shader(fragment_shader);
      fragment_shader = {};
    }

    if (compute_shader.valid()) {
      device->destroy_shader(compute_shader);
      compute_shader = {};
    }

    auto bindless_manager = rhi_context->get_bindless_manager();

    // Destroy vertex buffer
    if (bindless_manager->is_valid_handle(vertex_buffer)) {
      device->destroy_buffer(vertex_buffer);
      vertex_buffer = {};
    }

    // Destroy index buffer
    if (bindless_manager->is_valid_handle(index_buffer)) {
      device->destroy_buffer(index_buffer);
      index_buffer = {};
    }

    if (bindless_manager->is_valid_handle(noise_texture)) {
      device->destroy_texture(noise_texture);
      noise_texture = {};
    }

    // Note: Pre-defined samplers are not destroyed here
  }

  if (rhi_context != nullptr) {
    destroy_rhi_context(rhi_context);
    rhi_context = nullptr;
  }

  log::info("Minimal RHI Test Application cleanup complete");
}

bool RHITestApplication::create_rendering_resources() {
  log::info("Creating rendering resources (bindless approach)...");

  auto device = rhi_context->get_device();
  auto bindless_manager = rhi_context->get_bindless_manager();
  if (device == nullptr) {
    log::error("No device available");
    return false;
  }

  // Note: Using procedural vertex generation in shaders, no vertex buffer needed
  // But we still create an index buffer for indexed drawing

  // Create vertex buffer for 4 quads (24 vertices: 4 quads × 6 vertices each)
  // Store vertices as Vertex structures directly (UVs calculated in fragment shader)
  struct Vertex {
    float2 position;
    uint32_t quad_index;
  };

  std::vector<Vertex> vertices;
  vertices.reserve(24);
  uint16_t indices[24];  // 4 quads × 6 indices each

  // Quad positions in each quarter of screen, taking 1/8 of screen area each
  // Centers positioned at +/-0.5 from screen center
  float2 quad_centers[4] = {
    float2(-0.5f, 0.5f),   // Top-left quarter
    float2(0.5f, 0.5f),    // Top-right quarter
    float2(-0.5f, -0.5f),  // Bottom-left quarter
    float2(0.5f, -0.5f)    // Bottom-right quarter
  };

  uint32_t vertex_offset = 0;
  uint32_t index_offset = 0;

  for (uint32_t quad = 0; quad < 4; ++quad) {
    float2 center = quad_centers[quad];

    // Local quad vertices (relative to center, each quad extends ~0.35 units for 1/8 screen area)
    float2 local_positions[4] = {
      float2(-0.5f, -0.5f),  // bottom-left
      float2(0.5f, -0.5f),   // bottom-right
      float2(0.5f, 0.5f),    // top-right
      float2(-0.5f, 0.5f)    // top-left
    };

    float2 local_uvs[4] = {float2(0.0f, 0.0f), float2(1.0f, 0.0f), float2(1.0f, 1.0f), float2(0.0f, 1.0f)};

    // Create vertices for this quad
    for (uint32_t i = 0; i < 4; ++i) {
      Vertex vertex = {
        .position = float2(center.x + local_positions[i].x, center.y + local_positions[i].y),
        .quad_index = quad,
      };
      vertices.push_back(vertex);
    }

    // Create indices for this quad (2 triangles)
    uint16_t base_vertex = vertex_offset;
    indices[index_offset + 0] = base_vertex + 0;  // bottom-left
    indices[index_offset + 1] = base_vertex + 1;  // bottom-right
    indices[index_offset + 2] = base_vertex + 2;  // top-right
    indices[index_offset + 3] = base_vertex + 2;  // top-right
    indices[index_offset + 4] = base_vertex + 3;  // top-left
    indices[index_offset + 5] = base_vertex + 0;  // bottom-left

    vertex_offset += 4;
    index_offset += 6;
  }

  // Create vertex buffer
  RHIBufferDesc vertex_buffer_desc = {};
  vertex_buffer_desc.size = vertices.size() * sizeof(Vertex);
  vertex_buffer_desc.usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;

  auto vertex_result = device->create_buffer(vertex_buffer_desc);
  if (vertex_result.result != RHIResult::Success) {
    log::error("Failed to create vertex buffer");
    return false;
  }
  vertex_buffer = vertex_result.handle;

  // Upload vertex data
  if (device->update_buffer(vertex_buffer, vertices.data(), vertices.size() * sizeof(Vertex)) != RHIResult::Success) {
    log::error("Failed to upload vertex data");
    return false;
  }

  // Create index buffer
  RHIBufferDesc index_buffer_desc = {
    .size = sizeof(indices),  // 24 indices × 2 bytes each = 48 bytes
    .usage = RHIBufferUsage::Index | RHIBufferUsage::Storage | RHIBufferUsage::TransferDst,
  };

  auto index_result = device->create_buffer(index_buffer_desc);
  if (index_result.result != RHIResult::Success) {
    log::error("Failed to create index buffer");
    return false;
  }
  index_buffer = index_result.handle;

  // Upload index data
  if (device->update_buffer(index_buffer, indices, sizeof(indices)) != RHIResult::Success) {
    log::error("Failed to upload index data");
    return false;
  }

  // Get bindless indices for buffers (automatically registered during creation)
  vertex_buffer_index = get_bindless_descriptor_index(vertex_buffer);
  index_buffer_index = get_bindless_descriptor_index(index_buffer);

  // Validate buffers are properly registered
  bool vertex_valid = bindless_manager->is_valid_handle(vertex_buffer);
  bool index_valid = bindless_manager->is_valid_handle(index_buffer);

  log::info("Vertex buffer: handle=%llu, index=%u, valid=%s, size=%zu bytes (%zu vertices)", vertex_buffer, vertex_buffer_index, vertex_valid ? "yes" : "no",
    vertices.size() * sizeof(Vertex), vertices.size());
  log::info("Index buffer: handle=%llu, index=%u, valid=%s, size=%zu bytes (%zu indices)", index_buffer, index_buffer_index, index_valid ? "yes" : "no", sizeof(indices),
    sizeof(indices) / sizeof(uint16_t));

  if (!vertex_valid || !index_valid) {
    log::error("Buffer registration failed!");
    return false;
  }

  log::info("Vertex and index buffers created and registered for 4 quads");

  // Compile vertex shader to SPIR-V
  std::vector<uint8_t> vertex_spirv;
  if (!compile_shader_to_spirv("shaders/quad.hlsl", "vs_main", RHIShaderStage::Vertex, vertex_spirv)) {
    log::error("Failed to compile vertex shader");
    return false;
  }

  RHIShaderDesc vertex_shader_desc = {};
  vertex_shader_desc.spirv_data = vertex_spirv.data();
  vertex_shader_desc.spirv_size = vertex_spirv.size();
  vertex_shader_desc.stage = RHIShaderStage::Vertex;

  auto vertex_shader_result = device->create_shader(vertex_shader_desc);
  if (vertex_shader_result.result != RHIResult::Success) {
    log::error("Failed to create vertex shader");
    return false;
  }
  vertex_shader = vertex_shader_result.handle;
  log::info("Vertex shader created");

  // Compile fragment shader to SPIR-V (from same file as vertex shader)
  std::vector<uint8_t> fragment_spirv;
  if (!compile_shader_to_spirv("shaders/quad.hlsl", "fs_main", RHIShaderStage::Fragment, fragment_spirv)) {
    log::error("Failed to compile fragment shader");
    return false;
  }

  RHIShaderDesc fragment_shader_desc = {};
  fragment_shader_desc.spirv_data = fragment_spirv.data();
  fragment_shader_desc.spirv_size = fragment_spirv.size();
  fragment_shader_desc.stage = RHIShaderStage::Fragment;

  auto fragment_shader_result = device->create_shader(fragment_shader_desc);
  if (fragment_shader_result.result != RHIResult::Success) {
    log::error("Failed to create fragment shader");
    return false;
  }
  fragment_shader = fragment_shader_result.handle;
  log::info("Fragment shader created");

  // Create graphics pipeline
  // Create compute pipeline for noise generation
  std::vector<uint8_t> compute_spirv;
  if (!compile_shader_to_spirv("shaders/quad.hlsl", "cs_main", RHIShaderStage::Compute, compute_spirv)) {
    log::error("Failed to compile compute shader");
    return false;
  }

  RHIShaderDesc compute_shader_desc = {};
  compute_shader_desc.spirv_data = compute_spirv.data();
  compute_shader_desc.spirv_size = compute_spirv.size();
  compute_shader_desc.stage = RHIShaderStage::Compute;

  RHIComputePipelineDesc compute_pipeline_desc = {};
  compute_pipeline_desc.compute_shader = compute_shader_desc;
  compute_pipeline_desc.entry_point = "cs_main";

  auto compute_pipeline_result = device->create_compute_pipeline(compute_pipeline_desc);
  if (compute_pipeline_result.result != RHIResult::Success) {
    log::error("Failed to create compute pipeline");
    return false;
  }
  compute_pipeline = compute_pipeline_result.handle;
  log::info("Compute pipeline created");

  RHIGraphicsPipelineDesc pipeline_desc = {};

  // Vertex shader
  pipeline_desc.vertex_shader = vertex_shader_desc;
  pipeline_desc.vertex_entry_point = "vs_main";

  // Fragment shader
  pipeline_desc.fragment_shader = fragment_shader_desc;
  pipeline_desc.fragment_entry_point = "fs_main";

  // Vertex input - using procedural vertex generation, no vertex buffers
  pipeline_desc.vertex_attribute_count = 0;
  pipeline_desc.vertex_binding_count = 0;

  // Primitive topology
  pipeline_desc.primitive_topology = RHIPrimitiveTopology::TriangleList;

  // Color attachment format (will be set to swapchain format)
  pipeline_desc.color_attachment_count = 1;
  pipeline_desc.color_formats[0] = rhi_context->get_swapchain_format();

  auto pipeline_result = device->create_graphics_pipeline(pipeline_desc);
  if (pipeline_result.result != RHIResult::Success) {
    log::error("Failed to create graphics pipeline");
    return false;
  }
  graphics_pipeline = pipeline_result.handle;
  log::info("Graphics pipeline created");

  // Create noise texture
  if (!create_noise_texture()) {
    log::error("Failed to create noise texture");
    return false;
  }

  log::info("All rendering resources created successfully");
  return true;
}

bool RHITestApplication::create_noise_texture() {
  auto device = rhi_context->get_device();
  if (device == nullptr) {
    log::error("No device available for texture creation");
    return false;
  }

  const uint32_t texture_width = static_cast<uint32_t>(sapp_width());
  const uint32_t texture_height = static_cast<uint32_t>(sapp_height());

  // Create texture for compute shader output
  RHITextureDesc texture_desc = {};
  texture_desc.width = texture_width;
  texture_desc.height = texture_height;
  texture_desc.depth = 1;
  texture_desc.mip_levels = 1;
  texture_desc.array_layers = 1;
  texture_desc.format = RHITextureFormat::R8G8B8A8_UNORM;
  texture_desc.usage = RHITextureUsage::Sampled | RHITextureUsage::Storage;

  auto texture_result = device->create_texture(texture_desc);
  if (texture_result.result != RHIResult::Success) {
    log::error("Failed to create noise texture");
    return false;
  }
  noise_texture = texture_result.handle;

  log::info("Noise texture handle: %llu", noise_texture);
  log::info("Noise texture created for compute shader output");
  return true;
}

bool RHITestApplication::compile_shader_to_spirv(const std::string& file_path, const std::string& entry_point, RHIShaderStage stage, std::vector<uint8_t>& out_spirv) {
  // Read shader source
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    log::error("Failed to open shader file: %s", file_path.c_str());
    return false;
  }

  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string source(file_size, '\0');
  if (!file.read(source.data(), file_size)) {
    log::error("Failed to read shader file: %s", file_path.c_str());
    return false;
  }
  file.close();

  // Get shader compiler from device
  auto device = rhi_context->get_device();
  auto compiler = static_cast<VKDevice*>(device)->get_shader_compiler();
  if (compiler == nullptr) {
    log::error("No shader compiler available");
    return false;
  }

  // Compile to SPIR-V
  ShaderCompilationResult result = compiler->compile_hlsl_to_spirv(source, entry_point, stage, file_path);
  if (result.result != RHIResult::Success) {
    log::error("Failed to compile shader %s: %s", file_path.c_str(), result.error_message.c_str());
    return false;
  }

  out_spirv = std::move(result.spirv_data);
  log::info("Shader compiled successfully: %s (%zu bytes SPIR-V)", file_path.c_str(), out_spirv.size());
  return true;
}

void RHITestApplication::process_event(const sapp_event* event) {
  if (imgui.handle_event(event)) {
    return;
  }

  if (event->type == SAPP_EVENTTYPE_KEY_DOWN) {
    if (event->key_code == SAPP_KEYCODE_ESCAPE) {
      log::info("Escape key pressed - exiting windowed test");
      sapp_request_quit();
    }
  } else if (event->type == SAPP_EVENTTYPE_RESIZED) {
    // Handle window resize (including DPI changes)
    float dpi_scale = sapp_dpi_scale();
    log::info("Window resized - physical size: %dx%d, logical size: %dx%d, DPI scale: %.2f", sapp_width(), sapp_height(),
      static_cast<int>(static_cast<float>(sapp_width()) / dpi_scale), static_cast<int>(static_cast<float>(sapp_height()) / dpi_scale), dpi_scale);

    if (rhi_context != nullptr) {
      rhi_context->resize_swapchain(static_cast<uint32_t>(sapp_width()), static_cast<uint32_t>(sapp_height()));
    }

    if (sapp_high_dpi()) {
      log::info("High DPI resize - logical size: %dx%d, framebuffer size: %dx%d", sapp_width(), sapp_height(), static_cast<int>(static_cast<float>(sapp_width())),
        static_cast<int>(static_cast<float>(sapp_height())));
    }
  }
}

}  // namespace etx