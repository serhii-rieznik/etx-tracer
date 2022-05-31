#include <etx/rt/integrators/vcm_gpu.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

namespace etx {

struct GPUVCMImpl {
  GPUVCMImpl(Raytracing& art)
    : rt(art) {
  }

  ~GPUVCMImpl() {
    destroy_output();
    destroy_pipelines();
  }

  enum : uint32_t {
    CameraGen,
    CameraMain,
    LightGen,
    LightMain,
    PipelineCount,
  };

  Raytracing& rt;
  GPUBuffer camera_image = {};
  GPUBuffer light_image = {};
  GPUBuffer light_iteration_image = {};
  std::vector<float4> local_camera_image = {};
  std::vector<float4> local_light_image = {};
  uint2 dimemsions = {};

  GPUCameraLaunchParams camera_launch_params = {};
  device_pointer_t camera_launch_params_ptr = {};

  std::pair<GPUPipeline, const char*> pipelines[PipelineCount] = {
    {{}, "optix/vcm/camera-gen.json"},
    {{}, "optix/vcm/camera-main.json"},
    {{}, "optix/vcm/light-gen.json"},
    {{}, "optix/vcm/light-main.json"},
  };

  bool build_pipelines() {
    for (auto& p : pipelines) {
      if (p.first.handle == kInvalidHandle) {
        auto handle = rt.gpu()->create_pipeline_from_file(env().file_in_data(p.second), true);
        if (handle.handle == kInvalidHandle) {
          return false;
        }
        p.first = handle;
      }
    }

    return true;
  }

  void destroy_pipelines() {
    for (auto& p : pipelines) {
      rt.gpu()->destroy_pipeline(p.first);
      p.first = {};
    }
  }

  void build_output(const uint2& size) {
    destroy_output();

    dimemsions = size;
    local_camera_image.resize(1llu * size.x * size.y);
    local_light_image.resize(1llu * size.x * size.y);
    camera_image = rt.gpu()->create_buffer({sizeof(float4) * size.x * size.y});
    light_image = rt.gpu()->create_buffer({sizeof(float4) * size.x * size.y});
    light_iteration_image = rt.gpu()->create_buffer({sizeof(float4) * size.x * size.y});
  }

  void destroy_output() {
    rt.gpu()->destroy_buffer(camera_image);
    rt.gpu()->destroy_buffer(light_image);
    rt.gpu()->destroy_buffer(light_iteration_image);
  }

  bool run(const Options& opt) {
    if (build_pipelines() == false)
      return false;

    camera_launch_params.camera_image = make_array_view<float4>(reinterpret_cast<void*>(rt.gpu()->get_buffer_device_pointer(camera_image)), 1llu * dimemsions.x * dimemsions.y);
    camera_launch_params_ptr = rt.gpu()->upload_to_shared_buffer(camera_launch_params_ptr, &camera_launch_params, sizeof(camera_launch_params));
    if (rt.gpu()->launch(pipelines[CameraGen].first, dimemsions.x, dimemsions.y, camera_launch_params_ptr, sizeof(camera_launch_params)) == false) {
      return false;
    }

    return true;
  }
};

GPUVCM::GPUVCM(Raytracing& r)
  : Integrator(r) {
  ETX_PIMPL_INIT(GPUVCM, r);
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

void GPUVCM::set_output_size(const uint2& size) {
  _private->build_output(size);
}

void GPUVCM::preview(const Options& opt) {
  if (_private->build_pipelines() == false)
    return;

  _private->run(opt);
}

void GPUVCM::run(const Options& opt) {
  if (_private->build_pipelines() == false)
    return;

  if (_private->run(opt)) {
    current_state = State::Running;
  }
}

void GPUVCM::update() {
}

void GPUVCM::stop(Stop) {
}

void GPUVCM::update_options(const Options&) {
}

bool GPUVCM::have_updated_camera_image() const {
  return true;
}

const float4* GPUVCM::get_camera_image(bool /* force update */) {
  rt.gpu()->copy_from_buffer(_private->camera_image, _private->local_camera_image.data(), 0llu, _private->local_camera_image.size() * sizeof(float4));
  return _private->local_camera_image.data();
}

bool GPUVCM::have_updated_light_image() const {
  return true;
}

const float4* GPUVCM::get_light_image(bool /* force update */) {
  rt.gpu()->copy_from_buffer(_private->light_image, _private->local_light_image.data(), 0llu, _private->local_light_image.size() * sizeof(float4));
  return _private->local_light_image.data();
}

void GPUVCM::reload() {
  stop(Stop::Immediate);
  _private->destroy_pipelines();
  preview({});
}

}  // namespace etx
