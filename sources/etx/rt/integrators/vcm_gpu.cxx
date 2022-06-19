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
  GPUBuffer global_data = {};
  GPUBuffer state_buffers[2] = {};
  std::vector<float4> local_camera_image = {};
  std::vector<float4> local_light_image = {};
  uint2 dimemsions = {};

  VCMState vcm_state = VCMState::Stopped;
  VCMGlobal gpu_data = {};
  VCMIteration iteration = {};
  device_pointer_t iteration_ptr = {};
  device_pointer_t gpu_data_ptr = {};
  uint32_t current_buffer = 0;

  std::pair<GPUPipeline, const char*> pipelines[PipelineCount] = {
    {{}, "optix/vcm/camera-gen.json"},
    {{}, "optix/vcm/camera-main.json"},
    {{}, "optix/vcm/light-gen.json"},
    {{}, "optix/vcm/light-main.json"},
  };

  bool build_pipelines() {
    for (auto& p : pipelines) {
      if (p.first.handle == kInvalidHandle) {
        auto handle = rt.gpu()->create_pipeline_from_file(env().file_in_data(p.second), false);
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
    rt.gpu()->destroy_buffer(state_buffers[0]);
    state_buffers[0].handle = kInvalidIndex;
    
    rt.gpu()->destroy_buffer(state_buffers[1]);
    state_buffers[1].handle = kInvalidIndex;
    
    rt.gpu()->destroy_buffer(global_data);
    global_data.handle = kInvalidIndex;
    
    rt.gpu()->destroy_buffer(camera_image);
    camera_image.handle = kInvalidIndex;
    
    rt.gpu()->destroy_buffer(light_image);
    light_image.handle = kInvalidIndex;
    
    rt.gpu()->destroy_buffer(light_iteration_image);
    light_iteration_image.handle = kInvalidIndex;
  }

  bool run(const Options& opt) {
    if (build_pipelines() == false)
      return false;

    rt.gpu()->destroy_buffer(global_data);
    global_data = rt.gpu()->create_buffer({align_up(sizeof(VCMIteration), 16llu) + align_up(sizeof(VCMGlobal), 16llu)});

    rt.gpu()->destroy_buffer(state_buffers[0]);
    state_buffers[0] = rt.gpu()->create_buffer({sizeof(VCMPathState) * dimemsions.x * dimemsions.y});

    rt.gpu()->destroy_buffer(state_buffers[1]);
    state_buffers[1] = rt.gpu()->create_buffer({sizeof(VCMPathState) * dimemsions.x * dimemsions.y});

    gpu_data.scene = rt.gpu_scene();
    gpu_data.camera_image = make_array_view<float4>(rt.gpu()->get_buffer_device_pointer(camera_image), 1llu * dimemsions.x * dimemsions.y);
    gpu_data.light_image = make_array_view<float4>(rt.gpu()->get_buffer_device_pointer(light_image), 1llu * dimemsions.x * dimemsions.y);

    current_buffer = 0;
    iteration.iteration = 0;
    return start_next_iteration();
  }

  bool start_next_iteration() {
    constexpr float opt_radius_decay = 256.0f;
    constexpr float opt_radius = 0.0f;

    float used_radius = opt_radius;
    if (used_radius == 0.0f) {
      used_radius = 5.0f * rt.scene().bounding_sphere_radius * min(1.0f / float(dimemsions.x), 1.0f / float(dimemsions.y));
    }

    current_buffer = 0;

    float radius_scale = 1.0f / (1.0f + float(iteration.iteration) / float(opt_radius_decay));
    iteration.current_radius = used_radius * radius_scale;

    float eta_vcm = kPi * sqr(iteration.current_radius) * float(dimemsions.x) * float(dimemsions.y);
    iteration.vm_weight = eta_vcm;  // _vcm_options.merge_vertices() ? eta_vcm : 0.0f;
    iteration.vc_weight = 1.0f / eta_vcm;
    iteration.vm_normalization = 1.0f / eta_vcm;
    iteration.active_camera_paths = 0;
    iteration.active_light_paths = 0;
    iteration_ptr = rt.gpu()->copy_to_buffer(global_data, &iteration, 0, sizeof(VCMIteration));

    gpu_data.iteration = reinterpret_cast<VCMIteration*>(iteration_ptr);
    gpu_data.input_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[current_buffer]), 1llu * dimemsions.x * dimemsions.y);
    gpu_data.output_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[1 - current_buffer]), 1llu * dimemsions.x * dimemsions.y);
    gpu_data_ptr = rt.gpu()->copy_to_buffer(global_data, &gpu_data, sizeof(VCMIteration), sizeof(VCMGlobal));

    if (rt.gpu()->launch(pipelines[LightGen].first, dimemsions.x, dimemsions.y, gpu_data_ptr, sizeof(VCMGlobal)) == false) {
      vcm_state = VCMState::Stopped;
      return false;
    }

    vcm_state = VCMState::GatheringLightVertices;
    return true;
  }

  void update() {
    if (vcm_state == VCMState::Stopped)
      return;

    if (vcm_state == VCMState::GatheringLightVertices) {
      VCMIteration it = {};
      rt.gpu()->copy_from_buffer(global_data, &it, 0, sizeof(VCMIteration));
      if (it.active_light_paths > 0) {
        log::info("active_light_paths = %u", it.active_light_paths);
        // update iteration
        iteration.active_light_paths = 0;
        rt.gpu()->copy_to_buffer(global_data, &iteration, 0, sizeof(VCMIteration));

        // update buffers
        gpu_data.input_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[current_buffer]), 1llu * dimemsions.x * dimemsions.y);
        gpu_data.output_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[1 - current_buffer]), 1llu * dimemsions.x * dimemsions.y);
        gpu_data_ptr = rt.gpu()->copy_to_buffer(global_data, &gpu_data, sizeof(VCMIteration), sizeof(VCMGlobal));

        // launch main
        rt.gpu()->launch(pipelines[LightMain].first, it.active_light_paths, 1, gpu_data_ptr, sizeof(VCMGlobal));

        current_buffer = 1u - current_buffer;
      } else {
        log::info("active_light_paths = %u, FINISHED!", it.active_light_paths);
        // light gathering finished
        // build grid
        // etc...
        // ...
        vcm_state = VCMState::GatheringCameraVertices;
      }

    } else if (vcm_state == VCMState::GatheringCameraVertices) {
    }
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
  _private->update();
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
