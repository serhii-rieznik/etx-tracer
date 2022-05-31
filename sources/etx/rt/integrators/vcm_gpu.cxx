#include <etx/rt/integrators/vcm_gpu.hxx>

namespace etx {

struct GPUVCMImpl {};

GPUVCM::GPUVCM(Raytracing& r)
  : Integrator(r) {
  ETX_PIMPL_INIT(GPUVCM);
}

GPUVCM::~GPUVCM() {
  ETX_PIMPL_CLEANUP(GPUVCM);
}

bool GPUVCM::enabled() const {
  return true;
}

const char* GPUVCM::status() const {
  return "Hello world!";
}

Options GPUVCM::options() const {
  return {};
}

void GPUVCM::set_output_size(const uint2&) {
}

void GPUVCM::preview(const Options&) {
}

void GPUVCM::run(const Options&) {
}

void GPUVCM::update() {
}

void GPUVCM::stop(Stop) {
}

void GPUVCM::update_options(const Options&) {
}

bool GPUVCM::have_updated_camera_image() const {
  return false;
}

const float4* GPUVCM::get_camera_image(bool /* force update */) {
  return nullptr;
}

bool GPUVCM::have_updated_light_image() const {
  return false;
}

const float4* GPUVCM::get_light_image(bool /* force update */) {
  return nullptr;
}

void GPUVCM::reload() {
}

}  // namespace etx
