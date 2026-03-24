#include "envmap.hxx"

#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/render/host/image_loaders.hxx>

#include <vector>

namespace etx {

namespace {

bool envmap_create_pipeline(RHIContext& rhi, RHITextureFormat color_format, RHITextureFormat depth_format, uint32_t sample_count, RHIPipeline& out_pipeline) {
  std::string shader_source = env().file_in_data("playground/shaders/envmap.hlsl");
  ShaderCompiler::ShaderEntryPoint vs_ep = {"VSMain", RHIShaderStage::Vertex};
  ShaderCompiler::ShaderEntryPoint ps_ep = {"PSMain", RHIShaderStage::Fragment};
  auto compilation = ShaderCompiler::instance().compile(shader_source, {vs_ep, ps_ep});
  if (compilation.result != RHIResult::Success) {
    log::error("Failed to compile envmap shader:\n%s", compilation.error_message.c_str());
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
  p_desc.sample_count = sample_count;
  auto p_result = rhi.device().create_graphics_pipeline(p_desc);
  if (p_result.result != RHIResult::Success) {
    log::error("Failed to create envmap pipeline");
    return false;
  }

  out_pipeline = p_result.handle;
  return true;
}

}  // namespace

bool EnvMap::setup(RHIContext& rhi, RHITextureFormat color_format, RHITextureFormat depth_format, uint32_t sample_count) {
  std::string hdr_path = env().file_in_data("playground/hdr/envmap.hdr");
  std::vector<uint8_t> hdr_data;
  uint2 hdr_dims = {};
  Image::Format fmt = load_data(hdr_path.c_str(), hdr_data, hdr_dims);
  if (fmt == Image::Format::Undefined) {
    log::error("Failed to load envmap: %s", hdr_path.c_str());
    return false;
  }

  return setup_from_pixels(rhi, color_format, depth_format, reinterpret_cast<const float4*>(hdr_data.data()), hdr_dims, sample_count);
}

bool EnvMap::setup_from_pixels(RHIContext& rhi, RHITextureFormat color_format, RHITextureFormat depth_format, const float4* pixels, const uint2& dimensions,
  uint32_t sample_count) {
  if (pixels == nullptr) {
    log::error("Failed to create envmap texture: invalid pixels");
    return false;
  }
  if ((dimensions.x == 0u) || (dimensions.y == 0u)) {
    log::error("Failed to create envmap texture: invalid dimensions %u x %u", dimensions.x, dimensions.y);
    return false;
  }

  cleanup(rhi);

  RHITextureDesc tex_desc = {};
  tex_desc.width = dimensions.x;
  tex_desc.height = dimensions.y;
  tex_desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
  tex_desc.usage = RHITextureUsage::Sampled | RHITextureUsage::TransferDst;
  auto tex_result = rhi.device().create_texture(tex_desc);
  if (tex_result.result != RHIResult::Success) {
    log::error("Failed to create envmap texture");
    return false;
  }
  _texture = tex_result.handle;
  _owns_texture = true;
  _equal_area_mapping = false;
  _texture_state = RHIResourceState::TransferDst;
  if (rhi.device().update_texture(_texture, pixels) != RHIResult::Success) {
    log::error("Failed to upload envmap texture");
    rhi.device().destroy_texture(_texture);
    _texture = {};
    return false;
  }

  if (envmap_create_pipeline(rhi, color_format, depth_format, sample_count, _pipeline) == false) {
    rhi.device().destroy_texture(_texture);
    _texture = {};
    return false;
  }

  return true;
}

bool EnvMap::setup_with_texture(RHIContext& rhi, RHITextureFormat color_format, RHITextureFormat depth_format, RHITexture texture, RHIResourceState texture_state,
  bool equal_area_mapping, uint32_t sample_count) {
  if (texture.valid() == false) {
    log::error("Failed to bind envmap texture: invalid texture");
    return false;
  }

  cleanup(rhi);

  _texture = texture;
  _texture_state = texture_state;
  _owns_texture = false;
  _equal_area_mapping = equal_area_mapping;

  if (envmap_create_pipeline(rhi, color_format, depth_format, sample_count, _pipeline) == false) {
    _texture = {};
    _texture_state = RHIResourceState::Undefined;
    _owns_texture = false;
    _equal_area_mapping = false;
    return false;
  }

  return true;
}

void EnvMap::prepare_texture_for_sampling(RHIContext& rhi, RHICommandBuffer cmd) {
  if ((_texture.valid() == false) || (cmd.valid() == false)) {
    return;
  }
  if (_texture_state != RHIResourceState::ShaderReadOnly) {
    rhi.cmd_texture_barrier(cmd, _texture, _texture_state, RHIResourceState::ShaderReadOnly);
    _texture_state = RHIResourceState::ShaderReadOnly;
  }
}

void EnvMap::draw(RHIContext& rhi, RHICommandBuffer cmd, const float4x4& inv_view_proj) {
  rhi.cmd_set_pipeline(cmd, _pipeline);

  struct PushConstants {
    uint32_t envmap_index;
    uint32_t sampler_index;
    uint32_t mapping_mode;
    uint32_t padding;
    float4x4 inv_view_proj;
  };
  PushConstants pc = {};
  pc.envmap_index = get_bindless_descriptor_index(_texture);
  pc.sampler_index = rhi.get_sampler_index(RHISamplerType::LinearClamp);
  pc.mapping_mode = _equal_area_mapping ? 1u : 0u;
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
    if (_owns_texture) {
      rhi.device().destroy_texture(_texture);
    }
    _texture = {};
    _texture_state = RHIResourceState::Undefined;
    _owns_texture = false;
    _equal_area_mapping = false;
  }
}

}  // namespace etx
