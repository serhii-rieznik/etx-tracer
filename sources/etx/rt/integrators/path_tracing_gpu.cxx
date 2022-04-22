#include <etx/core/environment.hxx>
#include <etx/log/log.hxx>

#include <etx/rt/integrators/path_tracing_gpu.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

#include <etx/gpu/gpu.hxx>

namespace etx {

struct ETX_ALIGNED GPUPathTracingImpl {
  PathTracingGPUData gpu_data = {};

  GPUBuffer camera_image = {};
  GPUPipeline main_pipeline = {};
  std::vector<float4> local_camera_image = {};
  uint2 output_size = {};
  device_pointer_t gpu_launch_params = {};
  Options last_options = {};
};

GPUPathTracing::GPUPathTracing(Raytracing& r)
  : Integrator(r) {
  ETX_PIMPL_INIT(GPUPathTracing);
}

GPUPathTracing::~GPUPathTracing() {
  rt.gpu()->destroy_buffer(_private->camera_image);
  rt.gpu()->destroy_pipeline(_private->main_pipeline);
  ETX_PIMPL_CLEANUP(GPUPathTracing);
}

Options GPUPathTracing::options() const {
  return {};
}

void GPUPathTracing::set_output_size(const uint2& size) {
  rt.gpu()->destroy_buffer(_private->camera_image);
  _private->output_size = size;
  _private->camera_image = rt.gpu()->create_buffer({size.x * size.y * sizeof(float4)});
  _private->local_camera_image.resize(1llu * size.x * size.y);
}

void GPUPathTracing::preview(const Options& opt) {
  _private->last_options = opt;

  if (_private->main_pipeline.handle == kInvalidHandle) {
    _private->main_pipeline = rt.gpu()->create_pipeline_from_file(env().file_in_data("optix/pt/test.json"), true);
  }

  if (_private->main_pipeline.handle == kInvalidHandle) {
    log::error("[%s] failed to launch preview", name());
    return;
  }

  _private->gpu_data.output = reinterpret_cast<float4*>(rt.gpu()->get_buffer_device_pointer(_private->camera_image));
  _private->gpu_data.acceleration_structure = rt.gpu()->get_acceleration_structure_device_pointer(rt.gpu_acceleration_structure());
  _private->gpu_data.scene = rt.gpu_scene();

  _private->gpu_launch_params = rt.gpu()->upload_to_shared_buffer(_private->gpu_launch_params, &_private->gpu_data, sizeof(_private->gpu_data));

  rt.gpu()->launch(_private->main_pipeline, _private->output_size.x, _private->output_size.y, _private->gpu_launch_params, sizeof(_private->gpu_data));
}

void GPUPathTracing::run(const Options& opt) {
  _private->last_options = opt;
}

void GPUPathTracing::update() {
}

void GPUPathTracing::stop(Stop) {
}

void GPUPathTracing::update_options(const Options&) {
}

const float4* GPUPathTracing::get_camera_image(bool /* force */) {
  if (_private->camera_image.handle == kInvalidHandle)
    return nullptr;

  rt.gpu()->copy_from_buffer(_private->camera_image, _private->local_camera_image.data(), 0llu, _private->local_camera_image.size() * sizeof(float4));
  return _private->local_camera_image.data();
}

void GPUPathTracing::reload() {
  stop(Stop::Immediate);

  rt.gpu()->destroy_pipeline(_private->main_pipeline);
  _private->main_pipeline = {};

  preview(_private->last_options);
}

}  // namespace etx