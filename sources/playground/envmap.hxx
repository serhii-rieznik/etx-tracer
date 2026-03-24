#pragma once

#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/render/interop/interop.hxx>
#include <etx/render/shared/base.hxx>

namespace etx {

struct EnvMap {
  bool setup(RHIContext& rhi, RHITextureFormat color_format, RHITextureFormat depth_format, uint32_t sample_count = 1u);
  bool setup_from_pixels(RHIContext& rhi, RHITextureFormat color_format, RHITextureFormat depth_format, const float4* pixels, const uint2& dimensions, uint32_t sample_count = 1u);
  bool setup_with_texture(RHIContext& rhi, RHITextureFormat color_format, RHITextureFormat depth_format, RHITexture texture, RHIResourceState texture_state,
    bool equal_area_mapping, uint32_t sample_count = 1u);
  void prepare_texture_for_sampling(RHIContext& rhi, RHICommandBuffer cmd);
  void set_texture_state(RHIResourceState texture_state) {
    _texture_state = texture_state;
  }
  RHIResourceState texture_state() const {
    return _texture_state;
  }
  bool uses_equal_area_mapping() const {
    return _equal_area_mapping;
  }
  void draw(RHIContext& rhi, RHICommandBuffer cmd, const float4x4& inv_view_proj);
  void cleanup(RHIContext& rhi);

  bool valid() const {
    return _texture.valid() && _pipeline.valid();
  }

  RHITexture texture() const {
    return _texture;
  }

 private:
  RHITexture _texture;
  RHIResourceState _texture_state = RHIResourceState::Undefined;
  RHIPipeline _pipeline;
  bool _owns_texture = false;
  bool _equal_area_mapping = false;
};

}  // namespace etx
