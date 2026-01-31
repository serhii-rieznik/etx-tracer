#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/base.hxx>
#include <etx/rhi/rhi_types.hxx>
#include <etx/rhi/rhi.hxx>

#include "options.hxx"

#include <functional>

namespace etx {

struct TaskScheduler;

struct RenderContext {
  RenderContext() {
  }  // Dummy for migration
  RenderContext(TaskScheduler& s);
  ~RenderContext();

  void init();
  void cleanup(std::function<void()> clean_resources);

  void begin_frame();
  void start_frame(uint32_t sample_count, const ViewParameters&);
  void end_frame();

  RHICommandBuffer* current_command_buffer();

  void set_output_dimensions(const uint2&);
  uint2 get_output_dimensions() const;
  RHITexture get_output_texture() const;
  const ViewParameters& view_parameters() const;
  uint32_t get_view_layer() const {
    return view_parameters().view_layer;
  }

  void update_image(const float4* camera);
  void set_reference_image(const char*);
  void set_reference_image(const float4 data[], const uint2 dimensions);

  RHIContext* get_context();
  RHIDevice* get_device();
  RHITextureFormat get_swapchain_format();
  RHITextureFormat get_depth_format();

  ETX_DECLARE_PIMPL(RenderContext, 1024);

 private:
  void apply_reference_image(uint32_t);
};

}  // namespace etx
