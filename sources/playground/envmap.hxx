#pragma once

#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/render/interop/interop.hxx>
#include <etx/render/shared/base.hxx>

namespace etx {

struct EnvMap {
  bool setup(RHIContext& rhi);
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
  RHIPipeline _pipeline;
};

}  // namespace etx
