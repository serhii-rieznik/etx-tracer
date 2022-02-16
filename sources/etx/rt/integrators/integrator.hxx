#pragma once

#include <etx/core/options.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/rt/rt.hxx>

#include <atomic>

namespace etx {

struct Integrator {
  enum class State : uint32_t {
    Stopped,
    Preview,
    Running,
    WaitingForCompletion,
  };

  enum class Stop : uint32_t {
    Immediate,
    WaitForCompletion,
  };

  Integrator(Raytracing& r)
    : rt(r) {
  }

  virtual ~Integrator() = default;

  virtual const char* name() {
    return "Basic Integrator";
  }

  virtual const char* status() const {
    return "Basic Integrator (not able to render anything)";
  }

  virtual Options options() const {
    Options result = {};
    result.set("desc", "No options available");
    return result;
  }

  virtual void set_output_size(const uint2&) {
  }

  virtual void preview(const Options&) {
  }

  virtual void run(const Options&) {
  }

  virtual void update() {
  }

  virtual void stop(Stop) {
  }

  virtual void update_options(const Options&) {
  }

  virtual bool have_updated_camera_image() const {
    return true;
  }

  virtual const float4* get_camera_image(bool /* force update */) {
    return nullptr;
  }

  virtual bool have_updated_light_image() const {
    return true;
  }

  virtual const float4* get_light_image(bool /* force update */) {
    return nullptr;
  }

 public:
  bool can_run() const {
    return rt.has_scene();
  }
  State state() const {
    return current_state.load();
  }

 protected:
  Raytracing& rt;
  std::atomic<State> current_state = {State::Stopped};
};

inline SpectralResponse transmittance(SpectralQuery spect, Sampler& smp, const float3& p0, const float3& p1, uint32_t medium_index, const Raytracing& rt) {
  float3 w_o = p1 - p0;
  float max_t = length(w_o);
  w_o /= max_t;
  max_t -= kRayEpsilon;

  float3 origin = p0;

  SpectralResponse result = {spect.wavelength, 1.0f};

  const auto& scene = rt.scene();
  for (;;) {
    Intersection intersection;
    if (rt.trace({origin, w_o, kRayEpsilon, max_t}, intersection, smp) == false) {
      if (medium_index != kInvalidIndex) {
        result *= scene.mediums[medium_index].transmittance(spect, smp, origin, w_o, max_t);
      }
      break;
    }

    const auto& tri = scene.triangles[intersection.triangle_index];
    const auto& mat = scene.materials[tri.material_index];
    if (mat.cls != Material::Class::Boundary) {
      result = {spect.wavelength, 0.0f};
      break;
    }

    if (medium_index != kInvalidIndex) {
      result *= scene.mediums[medium_index].transmittance(spect, smp, origin, w_o, intersection.t);
    }

    medium_index = (dot(intersection.nrm, w_o) < 0.0f) ? mat.int_medium : mat.ext_medium;
    origin = intersection.pos;
    max_t -= intersection.t;
  }

  return result;
}

}  // namespace etx
