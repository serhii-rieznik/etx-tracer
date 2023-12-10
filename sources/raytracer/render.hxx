#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/base.hxx>

#include "options.hxx"

namespace etx {

struct RenderContext {
  RenderContext(TaskScheduler& s);
  ~RenderContext();

  void init();
  void cleanup();

  void start_frame(uint32_t sample_count);
  void end_frame();

  void set_output_dimensions(const uint2&);
  void update_camera_image(const float4* camera);
  void update_light_image(const float4* ligth);
  void set_view_options(const ViewOptions&);
  void set_reference_image(const char*);
  void set_reference_image(const float4 data[], const uint2 dimensions);

  ETX_DECLARE_PIMPL(RenderContext, 512);

 private:
  void apply_reference_image(uint32_t);
};

}  // namespace etx
