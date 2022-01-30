#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/base.hxx>

#include "options.hxx"

namespace etx {

struct RenderContext {
  RenderContext();
  ~RenderContext();

  void init();
  void cleanup();

  void start_frame();
  void end_frame();

  void set_output_dimensions(const uint2&);
  void update_output_images(const float4* camera, const float4* ligth);
  void set_view_options(const ViewOptions&);
  void set_reference_image(const char*);

 private:
  void apply_reference_image(Handle);

 private:
  ETX_DECLARE_PIMPL(RenderContext, 256);
};

}  // namespace etx
