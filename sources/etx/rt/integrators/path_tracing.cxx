#include <etx/rt/integrators/path_tracing.hxx>

namespace etx {

struct CPUPathTracingImpl {
  std::vector<float4> camera_image;
  uint2 dimensions = {};
  const char* status = nullptr;
};

ETX_PIMPL_IMPLEMENT_ALL(CPUPathTracing, Impl);

void CPUPathTracing::set_output_size(const uint2& dim) {
  _private->dimensions = dim;
  _private->camera_image.resize(dim.x * dim.y);
}

float4* CPUPathTracing::get_updated_camera_image() {
  for (auto& v : _private->camera_image) {
    v.x = float(rand()) / RAND_MAX;
    v.y = float(rand()) / RAND_MAX;
    v.z = float(rand()) / RAND_MAX;
  }
  return _private->camera_image.data();
}

float4* CPUPathTracing::get_updated_light_image() {
  return nullptr;
}

const char* CPUPathTracing::status() const {
  return _private->status;
}

}  // namespace etx