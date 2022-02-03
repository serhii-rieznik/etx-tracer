#include <etx/render/host/rnd_sampler.hxx>
#include <etx/rt/integrators/path_tracing.hxx>

namespace etx {

struct CPUPathTracingImpl {
  std::vector<float4> camera_image;
  uint2 dimensions = {};
  const char* status = nullptr;
};

CPUPathTracing::CPUPathTracing(CPUPathTracing&& other) noexcept
  : Integrator(other.rt) {
  if (_private) {
    _private->~CPUPathTracingImpl();
  }
  memcpy(_private_storage, other._private_storage, sizeof(_private_storage));
  _private = reinterpret_cast<struct CPUPathTracingImpl*>(_private_storage);
  memset(other._private_storage, 0, sizeof(_private_storage));
  other._private = nullptr;
}

CPUPathTracing& CPUPathTracing::operator=(CPUPathTracing&& other) noexcept {
  if (_private) {
    _private->~CPUPathTracingImpl();
  }
  memcpy(_private_storage, other._private_storage, sizeof(_private_storage));
  _private = reinterpret_cast<struct CPUPathTracingImpl*>(_private_storage);
  memset(other._private_storage, 0, sizeof(_private_storage));
  other._private = nullptr;
  return *this;
}

CPUPathTracing::CPUPathTracing(const struct Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUPathTracing);
}

CPUPathTracing::~CPUPathTracing() {
  ETX_PIMPL_CLEANUP(CPUPathTracing);
}

void CPUPathTracing::set_output_size(const uint2& dim) {
  _private->dimensions = dim;
  _private->camera_image.resize(dim.x * dim.y);
}

float4* CPUPathTracing::get_updated_camera_image() {
  return _private->camera_image.data();
}

float4* CPUPathTracing::get_updated_light_image() {
  return nullptr;
}

const char* CPUPathTracing::status() const {
  return _private->status;
}

void CPUPathTracing::preview() {
  RNDSampler smp;
  const auto& scene = rt.scene();
  uint2 size = {scene.camera.image_size.x, scene.camera.image_size.y};
  for (uint32_t y = 0; y < size.y; ++y) {
    for (uint32_t x = 0; x < size.x; ++x) {
      uint32_t i = x + (size.y - 1 - y) * size.x;

      float2 uv = get_jittered_uv(smp, {x, y}, size);
      auto ray = generate_ray(smp, rt.scene(), uv);
      Intersection intersection;
      if (rt.trace(ray, intersection, smp)) {
        _private->camera_image[i] = {spectrum::rgb_to_xyz(intersection.barycentric), 0.0f};
      } else {
        _private->camera_image[i] = {0.1f, 0.1f, 0.1f, 0.0f};
      }
    }
  }
}

}  // namespace etx
