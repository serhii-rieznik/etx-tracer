#include <etx/gpu/optix.hxx>

namespace etx {

GPUDevice* GPUDevice::create_optix_device() {
  return new GPUOptixImpl();
}

void GPUDevice::free_device(GPUDevice* device) {
  delete device;
}

}  // namespace etx
