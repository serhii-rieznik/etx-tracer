#include <etx/rt/integrators/path_tracing_gpu.hxx>

#include <etx/gpu/gpu.hxx>

namespace etx {

struct GPUPathTracingImpl {
  GPUBuffer camera_image = {};
  GPUPipeline main_pipeline = {};
  std::vector<float4> local_camera_image = {};
};

GPUPathTracing::GPUPathTracing(Raytracing& r)
  : Integrator(r) {
  ETX_PIMPL_INIT(GPUPathTracing);
}

GPUPathTracing::~GPUPathTracing() {
  rt.gpu()->destroy_buffer(_private->camera_image);
  ETX_PIMPL_CLEANUP(GPUPathTracing);
}

Options GPUPathTracing::options() const {
  return {};
}

void GPUPathTracing::set_output_size(const uint2& size) {
  rt.gpu()->destroy_buffer(_private->camera_image);
  _private->camera_image = rt.gpu()->create_buffer({size.x * size.y * sizeof(float4)});
  _private->local_camera_image.resize(1llu * size.x * size.y);
}

void GPUPathTracing::preview(const Options&) {
}

void GPUPathTracing::run(const Options&) {
}

void GPUPathTracing::update() {
}

void GPUPathTracing::stop(Stop) {
}

void GPUPathTracing::update_options(const Options&) {
}

const float4* GPUPathTracing::get_camera_image(bool /* force */) {
  rt.gpu()->copy_from_buffer(_private->camera_image, _private->local_camera_image.data(), 0llu, _private->local_camera_image.size() * sizeof(float4));
  return _private->local_camera_image.data();
}

}  // namespace etx