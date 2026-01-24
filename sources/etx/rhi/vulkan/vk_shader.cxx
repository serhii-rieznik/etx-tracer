#include <etx/rhi/vulkan/vk_shader.hxx>

#include <etx/core/log.hxx>

namespace etx {

VKShader::VKShader(VkDevice device, const RHIShaderDesc& desc)
  : _device(device)
  , _stage(desc.stage) {
  if (desc.spirv_data != nullptr && desc.spirv_size > 0) {
    if (!create_from_spirv(desc.spirv_data, desc.spirv_size)) {
      log::error("Failed to create shader from SPIR-V data");
    }
  } else {
    log::error("VKShader: No SPIR-V data provided");
  }
}

VKShader::VKShader(VkDevice device, const std::string& hlsl_source, const std::string& entry_point, RHIShaderStage stage, ShaderCompiler* compiler,
  const std::unordered_map<std::string, std::string>& defines)
  : _device(device)
  , _stage(stage) {
  if (!create_from_hlsl(hlsl_source, entry_point, stage, compiler, defines)) {
    log::error("Failed to create shader from HLSL source");
  }
}

VKShader::~VKShader() {
  if (_shader_module != VK_NULL_HANDLE && _device != VK_NULL_HANDLE) {
    vkDestroyShaderModule(_device, _shader_module, nullptr);
    _shader_module = VK_NULL_HANDLE;
  }
}

bool VKShader::create_from_spirv(const void* spirv_data, uint64_t spirv_size) {
  VkShaderModuleCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = spirv_size;
  create_info.pCode = reinterpret_cast<const uint32_t*>(spirv_data);

  VkResult result = vkCreateShaderModule(_device, &create_info, nullptr, &_shader_module);
  if (result != VK_SUCCESS) {
    _last_error = "Failed to create Vulkan shader module: " + std::to_string(static_cast<int>(result));
    log::error("%s", _last_error.c_str());
    return false;
  }

  return true;
}

bool VKShader::create_from_hlsl(const std::string& hlsl_source, const std::string& entry_point, RHIShaderStage stage, ShaderCompiler* compiler,
  const std::unordered_map<std::string, std::string>& defines) {
  if (!compiler || !compiler->is_initialized()) {
    _last_error = "Shader compiler not available or not initialized";
    log::error("%s", _last_error.c_str());
    return false;
  }

  auto result = compiler->compile_hlsl_to_spirv(hlsl_source, entry_point, stage, "shader.hlsl", defines);

  if (result.result != RHIResult::Success) {
    _last_error = "HLSL compilation failed: " + result.error_message;
    if (!result.warning_message.empty()) {
      _last_error += "\nWarnings: " + result.warning_message;
    }
    log::error("%s", _last_error.c_str());
    return false;
  }

  if (!create_from_spirv(result.spirv_data.data(), result.spirv_data.size())) {
    _last_error = "Failed to create shader module from compiled SPIR-V";
    return false;
  }

  return true;
}

}  // namespace etx
