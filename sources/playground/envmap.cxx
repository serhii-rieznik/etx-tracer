#include "envmap.hxx"

#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/render/host/image_loaders.hxx>

#include <vector>

namespace etx {

bool EnvMap::setup(RHIContext& rhi, RHITextureFormat color_format, RHITextureFormat depth_format) {
  std::string hdr_path = env().file_in_data("playground/hdr/envmap.hdr");
  std::vector<uint8_t> hdr_data;
  uint2 hdr_dims = {};
  Image::Format fmt = load_data(hdr_path.c_str(), hdr_data, hdr_dims);
  if (fmt == Image::Format::Undefined) {
    log::error("Failed to load envmap: %s", hdr_path.c_str());
    return false;
  }

  RHITextureDesc tex_desc = {};
  tex_desc.width = hdr_dims.x;
  tex_desc.height = hdr_dims.y;
  tex_desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
  tex_desc.usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst;
  auto tex_result = rhi.device().create_texture(tex_desc);
  if (tex_result.result != RHIResult::Success) {
    log::error("Failed to create envmap texture");
    return false;
  }
  _texture = tex_result.handle;
  rhi.device().update_texture(_texture, hdr_data.data());

  std::string shader_source = env().file_in_data("playground/shaders/envmap.hlsl");
  ShaderCompiler::ShaderEntryPoint vs_ep = {"VSMain", RHIShaderStage::Vertex};
  ShaderCompiler::ShaderEntryPoint ps_ep = {"PSMain", RHIShaderStage::Fragment};
  auto compilation = ShaderCompiler::instance().compile(shader_source, {vs_ep, ps_ep});
  if (compilation.result != RHIResult::Success) {
    log::error("Failed to compile envmap shader:\n%s", compilation.error_message.c_str());
    rhi.device().destroy_texture(_texture);
    _texture = {};
    return false;
  }

  RHIGraphicsPipelineDesc p_desc = {};
  p_desc.vertex_shader.stage = RHIShaderStage::Vertex;
  p_desc.vertex_shader.entry_point = "VSMain";
  p_desc.vertex_shader.spirv_data = compilation.binaries[0].spirv_data;
  p_desc.vertex_shader.spirv_size = compilation.binaries[0].spirv_size;
  p_desc.fragment_shader.stage = RHIShaderStage::Fragment;
  p_desc.fragment_shader.entry_point = "PSMain";
  p_desc.fragment_shader.spirv_data = compilation.binaries[1].spirv_data;
  p_desc.fragment_shader.spirv_size = compilation.binaries[1].spirv_size;
  p_desc.depth_state.depth_test_enable = false;
  p_desc.depth_state.depth_write_enable = false;
  p_desc.primitive_topology = RHIPrimitiveTopology::TriangleList;
  p_desc.color_attachment_count = 1;
  p_desc.color_formats[0] = color_format;
  p_desc.depth_format = depth_format;
  auto p_result = rhi.device().create_graphics_pipeline(p_desc);
  if (p_result.result != RHIResult::Success) {
    log::error("Failed to create envmap pipeline");
    rhi.device().destroy_texture(_texture);
    _texture = {};
    return false;
  }
  _pipeline = p_result.handle;
  return true;
}

void EnvMap::draw(RHIContext& rhi, RHICommandBuffer cmd, const float4x4& inv_view_proj) {
  rhi.cmd_set_pipeline(cmd, _pipeline);

  struct PushConstants {
    uint32_t envmap_index;
    uint32_t sampler_index;
    uint32_t padding[2];
    float4x4 inv_view_proj;
  };
  PushConstants pc = {};
  pc.envmap_index = get_bindless_descriptor_index(_texture);
  pc.sampler_index = rhi.get_sampler_index(RHISamplerType::LinearClamp);
  pc.inv_view_proj = inv_view_proj;
  rhi.cmd_push_constants(cmd, &pc, sizeof(PushConstants), 0);
  rhi.cmd_draw(cmd, {.vertex_count = 3, .instance_count = 1});
}

void EnvMap::cleanup(RHIContext& rhi) {
  if (_pipeline.valid()) {
    rhi.device().destroy_pipeline(_pipeline);
    _pipeline = {};
  }
  if (_texture.valid()) {
    rhi.device().destroy_texture(_texture);
    _texture = {};
  }
}

}  // namespace etx
