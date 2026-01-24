#include <etx/core/log.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/vulkan/vk_rhi.hxx>
#include <etx/rhi/rhi_bindless.hxx>
#include <etx/rhi/vulkan/vk_device.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

using namespace etx;

int main(int argc, char* argv[]) {
  printf("=== RHI STANDALONE TEST START ===\n");

  // Create RHI context
  RHIInitInfo init_info = {};
  init_info.backend = RHIBackend::Vulkan;
  init_info.enable_validation = false;
  init_info.enable_debug_names = true;
  init_info.max_frames_in_flight = 2;

  // Initialize global shader compiler (must be done before any RHI context creation)
  printf("Initializing global shader compiler...\n");
  if (ShaderCompiler::initialize_global() != RHIResult::Success) {
    printf("Failed to initialize global shader compiler\n");
    return 1;
  }
  printf("Global shader compiler initialized successfully\n");

  printf("Creating RHI context...\n");
  RHIContext* rhi_context = create_rhi_context(init_info);
  if (rhi_context == nullptr) {
    printf("Failed to create RHI context\n");
    return 1;
  }
  printf("RHI context created successfully\n");

  RHIDevice* rhi_device = rhi_context->get_device();
  RHIBindlessManager* bindless_manager = rhi_context->get_bindless_manager();
  printf("Got device and bindless manager pointers\n");

  // Set bindless capacities
  bindless_manager->set_max_buffers(kDefaultMaxBuffers);
  bindless_manager->set_max_textures(kDefaultMaxTextures);
  bindless_manager->set_max_samplers(kDefaultMaxSamplers);
  bindless_manager->set_max_acceleration_structures(kDefaultMaxAccelerationStructures);
  printf("Bindless manager capacities set\n");

  // Manually initialize bindless manager for headless testing
  // (normally done in create_swapchain)
  auto vk_bindless_manager = static_cast<VKBindlessManager*>(bindless_manager);
  auto vk_device = static_cast<VKDevice*>(rhi_device);
  if (!vk_bindless_manager->is_initialized()) {
    vk_bindless_manager->initialize(vk_device->get_vk_device(), vk_device->get_vk_physical_device());
    vk_device->set_bindless_manager(vk_bindless_manager);
    printf("Bindless manager initialized manually for headless testing\n");
  }

  // Test bindless handle encoding/decoding
  printf("Testing bindless handle encoding/decoding...\n");
  RHIBindlessHandle test_handle = make_bindless_handle(RHIResourceType::Texture, 42, 1337);
  RHIResourceType decoded_type = get_bindless_resource_type(test_handle);
  uint32_t decoded_generation = get_bindless_generation(test_handle);
  uint32_t decoded_index = get_bindless_descriptor_index(test_handle);

  printf("Original: type=%u, generation=%u, index=%u\n", static_cast<uint32_t>(RHIResourceType::Texture), 42, 1337);
  printf("Decoded:  type=%u, generation=%u, index=%u\n", static_cast<uint32_t>(decoded_type), decoded_generation, decoded_index);

  if (decoded_type == RHIResourceType::Texture && decoded_generation == 42 && decoded_index == 1337) {
    printf("Bindless handle encoding/decoding test PASSED\n");
  } else {
    printf("Bindless handle encoding/decoding test FAILED\n");
  }

  // Test resource creation
  printf("Testing resource creation with bindless registration...\n");

  // Handle variables for cleanup
  RHIBindlessHandle buffer_handle = 0;
  RHIBindlessHandle texture_handle = 0;
  RHIBindlessHandle sampler_handle = 0;

  // Test buffer creation
  RHIBufferDesc buffer_desc = {};
  buffer_desc.size = 1024;
  buffer_desc.usage = RHIBufferUsage::Storage;
  buffer_desc.host_visible = true;

  RHICreateBindlessResult buffer_result = rhi_device->create_buffer(buffer_desc);
  if (buffer_result.result == RHIResult::Success) {
    printf("Buffer creation PASSED - handle: %llu\n", buffer_result.handle);
    buffer_handle = buffer_result.handle;
  } else {
    printf("Buffer creation FAILED - result: %u\n", static_cast<uint32_t>(buffer_result.result));
  }

  // Test texture creation
  RHITextureDesc texture_desc = {};
  texture_desc.width = 256;
  texture_desc.height = 256;
  texture_desc.format = RHITextureFormat::R8G8B8A8_UNORM;
  texture_desc.usage = RHITextureUsage::Sampled;
  texture_desc.host_visible = false;

  RHICreateBindlessResult texture_result = rhi_device->create_texture(texture_desc);
  if (texture_result.result == RHIResult::Success) {
    printf("Texture creation PASSED - handle: %llu\n", texture_result.handle);
    texture_handle = texture_result.handle;
  } else {
    printf("Texture creation FAILED - result: %u\n", static_cast<uint32_t>(texture_result.result));
  }

  // Test sampler creation
  RHISamplerDesc sampler_desc = {};
  sampler_desc.min_filter = RHISamplerFilter::Linear;
  sampler_desc.mag_filter = RHISamplerFilter::Linear;
  sampler_desc.mipmap_mode = RHISamplerMipmapMode::Linear;

  RHICreateBindlessResult sampler_result = rhi_device->create_sampler(sampler_desc);
  if (sampler_result.result == RHIResult::Success) {
    printf("Sampler creation PASSED - handle: %llu\n", sampler_result.handle);
    sampler_handle = sampler_result.handle;
  } else {
    printf("Sampler creation FAILED - result: %u\n", static_cast<uint32_t>(sampler_result.result));
  }

  // Note: update_texture requires staging buffer operations (Phase 7)

  // Test shader compilation
  printf("Testing shader compilation...\n");
  {
    ShaderCompiler shader_compiler;
    if (shader_compiler.initialize() == RHIResult::Success) {
      printf("Shader compiler initialized successfully\n");

      // Test basic compute shader compilation
      std::string test_hlsl = R"(
        #define VULKAN 1
        #define SPIRV 1
        #define BINDLESS 1

        [[vk::binding(0, 0)]] StructuredBuffer<float4> inputBuffer;
        [[vk::binding(1, 0)]] RWStructuredBuffer<float4> outputBuffer;

        [numthreads(64, 1, 1)]
        void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
          uint index = dispatchThreadID.x;
          outputBuffer[index] = inputBuffer[index] * 2.0f;
        }
      )";

      auto result = shader_compiler.compile_hlsl_to_spirv(test_hlsl, "main", RHIShaderStage::Compute, "test_compute.hlsl");

      if (result.result == RHIResult::Success) {
        printf("Shader compilation PASSED - generated %zu bytes of SPIR-V\n", result.spirv_data.size());
      } else {
        printf("Shader compilation FAILED - %s\n", result.error_message.c_str());
        if (result.error_line > 0) {
          printf("Error at line %u, column %u\n", result.error_line, result.error_column);
        }
      }
    } else {
      printf("Failed to initialize shader compiler\n");
    }
  }

  // Test bindless validation
  printf("Testing bindless handle validation...\n");
  if (bindless_manager->is_valid_handle(buffer_handle)) {
    printf("Buffer handle validation PASSED\n");
  } else {
    printf("Buffer handle validation FAILED\n");
  }

  if (bindless_manager->is_valid_handle(texture_handle)) {
    printf("Texture handle validation PASSED\n");
  } else {
    printf("Texture handle validation FAILED\n");
  }

  if (bindless_manager->is_valid_handle(sampler_handle)) {
    printf("Sampler handle validation PASSED\n");
  } else {
    printf("Sampler handle validation FAILED\n");
  }

  // Cleanup
  if (buffer_handle != 0) {
    rhi_device->destroy_buffer(buffer_handle);
    printf("Destroyed buffer\n");
  }
  if (texture_handle != 0) {
    rhi_device->destroy_texture(texture_handle);
    printf("Destroyed texture\n");
  }
  if (sampler_handle != 0) {
    rhi_device->destroy_sampler(sampler_handle);
    printf("Destroyed sampler\n");
  }

  destroy_rhi_context(rhi_context);
  printf("RHI context destroyed\n");

  // Shutdown global shader compiler
  ShaderCompiler::shutdown_global();
  printf("Global shader compiler shut down\n");

  printf("=== RHI STANDALONE TEST COMPLETE ===\n");
  return 0;
}