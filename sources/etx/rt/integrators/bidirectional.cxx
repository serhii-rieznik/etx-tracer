#include <etx/rt/integrators/bidirectional.hxx>

namespace etx {

struct CPUBidirectionalImpl : public Task {
  char status[2048] = {};
  Raytracing& rt;
  uint2 film_dimensions = {};
  std::vector<float4> camera_image;
  std::vector<float4> light_image;

  CPUBidirectionalImpl(Raytracing& r)
    : rt(r) {
  }

  void execute_range(uint32_t begin, uint32_t end, uint32_t thread_id) {
  }
};

CPUBidirectional::CPUBidirectional(Raytracing& rt)
  : Integrator(rt) {
  ETX_PIMPL_INIT(CPUBidirectional, rt);
}

CPUBidirectional::~CPUBidirectional() {
  if (current_state != State::Stopped) {
    stop(false);
  }
  ETX_PIMPL_CLEANUP(CPUBidirectional);
}

void CPUBidirectional::preview(const Options&) {
  stop(false);
}

void CPUBidirectional::run(const Options&) {
  stop(false);
  if (rt.has_scene() == false) {
    return
  }
}

void CPUBidirectional::update() {
}

void CPUBidirectional::stop(bool /* wait for completion */) {
}

Options CPUBidirectional::options() const {
  Options result = {};
  return result;
}

void CPUBidirectional::update_options(const Options&) {
}

void CPUBidirectional::set_output_size(const uint2& dim) {
  if (current_state != State::Stopped) {
    stop(false);
  }
  _private->film_dimensions = dim;
  _private->camera_image.resize(1llu * dim.x * dim.y);
  _private->light_image.resize(1llu * dim.x * dim.y);
}

float4* CPUBidirectional::get_updated_camera_image() {
  return _private->camera_image.data();
}

float4* CPUBidirectional::get_updated_light_image() {
  return _private->light_image.data();
}

const char* CPUBidirectional::status() const {
  return _private->status;
}

}  // namespace etx
