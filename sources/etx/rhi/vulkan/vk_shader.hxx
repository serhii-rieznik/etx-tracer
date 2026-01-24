#pragma once

#include <etx/core/handle.hxx>
#include <etx/rhi/rhi_types.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include <vulkan/vulkan.h>

#include <string>
#include <unordered_map>

namespace etx {

class VKShader {
 public:
  VKShader(VkDevice device, const RHIShaderDesc& desc);
  VKShader(VkDevice device, const std::string& hlsl_source, const std::string& entry_point, RHIShaderStage stage, ShaderCompiler* compiler = nullptr,
    const std::unordered_map<std::string, std::string>& defines = {});
  ~VKShader();

  VkShaderModule get_vk_shader_module() const {
    return _shader_module;
  }

  RHIShaderStage get_stage() const {
    return _stage;
  }

  bool is_valid() const {
    return _shader_module != VK_NULL_HANDLE;
  }

  const std::string& get_last_error() const {
    return _last_error;
  }

 private:
  bool create_from_spirv(const void* spirv_data, uint64_t spirv_size);
  bool create_from_hlsl(const std::string& hlsl_source, const std::string& entry_point, RHIShaderStage stage, ShaderCompiler* compiler,
    const std::unordered_map<std::string, std::string>& defines);

  VkDevice _device = VK_NULL_HANDLE;
  VkShaderModule _shader_module = VK_NULL_HANDLE;
  RHIShaderStage _stage = RHIShaderStage::Vertex;
  std::string _last_error;
};

}  // namespace etx
