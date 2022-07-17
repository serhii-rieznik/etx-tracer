#include <etx/rt/integrators/vcm_gpu.hxx>
#include <etx/rt/shared/vcm_shared.hxx>

#include <cuda.h>

namespace etx {

CUstream cuda_stream();

struct CUDATimer {
  void begin() {
    _started = true;
    cuEventRecord(_begin, cuda_stream());
  }

  float retreive() {
    ETX_ASSERT(_started);
    cuEventRecord(_end, cuda_stream());
    cuEventSynchronize(_end);
    cuEventElapsedTime(&_duration, _begin, _end);
    return _duration;
  }

  CUDATimer() {
    cuEventCreate(&_begin, CU_EVENT_DEFAULT);
    cuEventCreate(&_end, CU_EVENT_DEFAULT);
  }

  ~CUDATimer() {
    cuEventDestroy(_end);
    cuEventDestroy(_begin);
  }

  CUDATimer(const CUDATimer&) = delete;
  CUDATimer& operator=(const CUDATimer&) = delete;
  CUDATimer(CUDATimer&&) = delete;
  CUDATimer& operator=(CUDATimer&&) = delete;

 private:
  CUevent _begin = {};
  CUevent _end = {};
  float _duration = 0.0f;
  bool _started = false;
};

struct GPUVCMImpl {
  GPUVCMImpl(Raytracing& art, std::atomic<Integrator::State>& st)
    : rt(art)
    , state(st)
    , options(VCMOptions::default_values()) {
  }

  ~GPUVCMImpl() {
    destroy_output();
    destroy_pipelines();
  }

  enum : uint32_t {
    VCMLibrary,
    // CameraGen,
    CameraMain,
    // LightGen,
    LightMain,
    PipelineCount,
  };

  enum : uint32_t {
    LightBufferCapacity = 3840u * 2160u,
    LightBufferDataSize = LightBufferCapacity * sizeof(VCMLightVertex),
  };

  Raytracing& rt;
  GPUBuffer camera_image = {};
  GPUBuffer light_final_image = {};
  GPUBuffer light_iteration_image = {};
  GPUBuffer light_vertex_buffer = {};
  GPUBuffer global_data = {};
  GPUBuffer state_buffers[2] = {};
  GPUBuffer spatial_grid_indices = {};
  GPUBuffer spatial_grid_cells = {};
  GPUBuffer light_paths_buffer = {};
  std::vector<float4> local_camera_image = {};
  std::vector<float4> local_light_image = {};
  std::vector<VCMLightVertex> light_vertices = {};
  std::vector<VCMLightPath> light_paths;
  uint2 dimemsions = {};

  Integrator::DebugInfo debug_info[2] = {
    {"GPU light time (ms)  "},
    {"GPU camera time (ms) "},
  };
  CUDATimer gpu_light_timer = {};
  CUDATimer gpu_camera_timer = {};

  std::atomic<Integrator::State>& state;

  VCMSpatialGrid grid_builder = {};
  VCMState vcm_state = VCMState::Stopped;
  VCMGlobal gpu_data = {};
  VCMIteration iteration = {};
  VCMOptions options = {};
  device_pointer_t iteration_ptr = {};
  device_pointer_t gpu_data_ptr = {};
  uint32_t current_buffer = 0;
  uint32_t iteration_frame = 0;
  uint32_t update_frame = 0;

  std::pair<GPUPipeline, const char*> pipelines[PipelineCount] = {
    {{}, "optix/vcm/vcm.json"},
    {{}, "optix/vcm/camera-main.json"},
    {{}, "optix/vcm/light-main.json"},
  };
  bool reload_pipelines[PipelineCount] = {};

  char status[2048] = {};

  void update_status(const char* tag, uint32_t active_paths) {
    if (state == Integrator::State::Stopped) {
      snprintf(status, sizeof(status), "Not running");
    } else {
      double completed = 100.0 * double(dimemsions.x * dimemsions.y - active_paths) / double(dimemsions.x * dimemsions.y);
      snprintf(status, sizeof(status), "[%u / %u] %s: %u frame, launching %u threads (%.2f percent done)...",  //
        iteration.iteration, options.max_samples, tag, iteration_frame, active_paths, completed);
    }
  }

  void stop() {
    vcm_state = VCMState::Stopped;
    state = Integrator::State::Stopped;

    update_status("Stopped", 0);
  }

  bool build_pipelines(const Options& opt) {
    for (auto& p : pipelines) {
      bool force_reload = opt.get(p.second, false).to_bool();
      if (force_reload || (p.first.handle == kInvalidHandle)) {
        auto handle = rt.gpu()->create_pipeline_from_file(env().file_in_data(p.second), force_reload);
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
    light_final_image = rt.gpu()->create_buffer({sizeof(float4) * size.x * size.y});
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

    rt.gpu()->destroy_buffer(light_final_image);
    light_final_image.handle = kInvalidIndex;

    rt.gpu()->destroy_buffer(light_iteration_image);
    light_iteration_image.handle = kInvalidIndex;

    rt.gpu()->destroy_buffer(light_vertex_buffer);
    light_vertex_buffer.handle = kInvalidHandle;

    rt.gpu()->destroy_buffer(spatial_grid_indices);
    spatial_grid_indices.handle = kInvalidIndex;

    rt.gpu()->destroy_buffer(spatial_grid_cells);
    spatial_grid_cells.handle = kInvalidIndex;

    rt.gpu()->destroy_buffer(light_paths_buffer);
    light_paths_buffer.handle = kInvalidHandle;
  }

  bool run(const Options& opt) {
    if (build_pipelines(opt) == false)
      return false;

    rt.gpu()->destroy_buffer(global_data);
    global_data = rt.gpu()->create_buffer({align_up(sizeof(VCMIteration), 16llu) + align_up(sizeof(VCMGlobal), 16llu)});

    rt.gpu()->destroy_buffer(state_buffers[0]);
    state_buffers[0] = rt.gpu()->create_buffer({sizeof(VCMPathState) * dimemsions.x * dimemsions.y});

    rt.gpu()->destroy_buffer(state_buffers[1]);
    state_buffers[1] = rt.gpu()->create_buffer({sizeof(VCMPathState) * dimemsions.x * dimemsions.y});

    light_vertices.reserve(LightBufferCapacity);

    light_paths.resize(1llu * dimemsions.x * dimemsions.y);
    light_paths_buffer = rt.gpu()->create_buffer({sizeof(VCMLightPath) * dimemsions.x * dimemsions.y});

    options.load(opt);

    gpu_data.options = options;
    gpu_data.scene = rt.gpu_scene();
    gpu_data.camera_image = make_array_view<float4>(rt.gpu()->get_buffer_device_pointer(camera_image), 1llu * dimemsions.x * dimemsions.y);
    gpu_data.light_final_image = make_array_view<float4>(rt.gpu()->get_buffer_device_pointer(light_final_image), 1llu * dimemsions.x * dimemsions.y);
    gpu_data.light_iteration_image = make_array_view<float4>(rt.gpu()->get_buffer_device_pointer(light_iteration_image), 1llu * dimemsions.x * dimemsions.y);

    rt.gpu()->clear_buffer(camera_image);
    rt.gpu()->clear_buffer(light_final_image);

    update_frame = 0;
    iteration_frame = 0;
    current_buffer = 0;
    iteration.iteration = 0;
    return start_next_iteration();
  }

  bool start_next_iteration() {
    if (iteration.iteration >= options.max_samples) {
      stop();
      return false;
    }

    float used_radius = options.initial_radius;
    if (used_radius == 0.0f) {
      used_radius = 5.0f * rt.scene().bounding_sphere_radius * min(1.0f / float(dimemsions.x), 1.0f / float(dimemsions.y));
    }

    current_buffer = 0;

    float radius_scale = 1.0f / (1.0f + float(iteration.iteration) / float(options.radius_decay));
    iteration.current_radius = used_radius * radius_scale;

    float eta_vcm = kPi * sqr(iteration.current_radius) * float(dimemsions.x) * float(dimemsions.y);
    iteration.vm_weight = eta_vcm;
    iteration.vc_weight = 1.0f / eta_vcm;
    iteration.vm_normalization = 1.0f / eta_vcm;
    iteration.active_paths = 0;
    iteration.light_vertices = 0;
    iteration_ptr = rt.gpu()->copy_to_buffer(global_data, &iteration, 0, sizeof(VCMIteration));

    rt.gpu()->destroy_buffer(light_vertex_buffer);
    light_vertex_buffer = rt.gpu()->create_buffer({LightBufferDataSize});
    light_vertices.clear();

    gpu_data.iteration = reinterpret_cast<VCMIteration*>(iteration_ptr);
    gpu_data.input_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[current_buffer]), 1llu * dimemsions.x * dimemsions.y);
    gpu_data.output_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[1 - current_buffer]), 1llu * dimemsions.x * dimemsions.y);
    gpu_data.light_vertices = make_array_view<VCMLightVertex>(rt.gpu()->get_buffer_device_pointer(light_vertex_buffer), LightBufferCapacity);
    gpu_data_ptr = rt.gpu()->copy_to_buffer(global_data, &gpu_data, sizeof(VCMIteration), sizeof(VCMGlobal));

    iteration_frame = 0;

    if (rt.gpu()->launch(pipelines[VCMLibrary].first, "gen_light_rays", dimemsions.x, dimemsions.y, gpu_data_ptr, sizeof(VCMGlobal)) == false) {
      stop();
      return false;
    }

    vcm_state = VCMState::GatheringLightVertices;
    return true;
  }

  void download_light_vertices(uint32_t count) {
    TimeMeasure tm = {};
    log::info("Downloading %u light vertices...", count);
    uint64_t light_buffer_pos = light_vertices.size();
    light_vertices.resize(light_buffer_pos + count);
    rt.gpu()->copy_from_buffer(light_vertex_buffer, light_vertices.data() + light_buffer_pos, 0, sizeof(VCMLightVertex) * count);
    log::info("Completed in %.2f sec", tm.measure());
  }

  uint32_t build_spatial_grid_data(const Scene& scene, VCMLightVertex* light_vertices_ptr, uint64_t light_vertex_count, VCMLightPath* paths_ptr, uint64_t paths_count,
    const VCMIteration& iteration, VCMSpatialGrid& grid_builder) {
    TimeMeasure tm_total = {};
    TimeMeasure tm = {};

    log::info(" - processing CPU data...");
    std::sort(light_vertices_ptr, light_vertices_ptr + light_vertex_count, [](const VCMLightVertex& a, const VCMLightVertex& b) {
      return (a.path_index == b.path_index) ? (a.path_length < b.path_length) : (a.path_index < b.path_index);
    });
    log::info(" --- sorting light vertices: %.2f sec", tm.lap());

    grid_builder.construct(scene, light_vertices_ptr, light_vertex_count, iteration.current_radius);
    log::info(" --- constructing grid: %.2f sec", tm.lap());

    memset(paths_ptr, 0, paths_count * sizeof(VCMLightPath));
    uint32_t max_path_length = 1;
    uint32_t index = 0u;
    uint32_t path_index = light_vertices_ptr ? light_vertices_ptr[0].path_index : 0u;
    for (uint32_t i = 0; i < light_vertex_count; ++i) {
      const auto& v = light_vertices_ptr[i];

      if (v.path_index != path_index) {
        index = static_cast<uint32_t>(i);
        path_index = v.path_index;
      }

      auto& path = paths_ptr[path_index];
      path.count += 1;
      path.index = index;
      path.spect = {v.throughput.wavelength};
      max_path_length = max(max_path_length, path.count);
    }

    log::info(
      " --- building paths: %.2f sec\n"
      " - completed in %.2f sec",
      tm.measure(), tm_total.measure());

    return max_path_length;
  }

  void build_spatial_grid() {
    TimeMeasure tm_total = {};
    log::info("Building spatial grid...");

    uint32_t max_path_length = build_spatial_grid_data(rt.scene(), light_vertices.data(), light_vertices.size(), light_paths.data(), light_paths.size(), iteration, grid_builder);
    gpu_data.spatial_grid = grid_builder.data;

    TimeMeasure tm = {};
    rt.gpu()->destroy_buffer(light_vertex_buffer);
    light_vertex_buffer = rt.gpu()->create_buffer({light_vertices.size() * sizeof(VCMLightVertex), light_vertices.data()});
    gpu_data.light_vertices = make_array_view<VCMLightVertex>(rt.gpu()->get_buffer_device_pointer(light_vertex_buffer), light_vertices.size());

    rt.gpu()->destroy_buffer(spatial_grid_indices);
    spatial_grid_indices = rt.gpu()->create_buffer({sizeof(uint32_t) * grid_builder.data.indices.count, gpu_data.spatial_grid.indices.a});
    gpu_data.spatial_grid.indices = make_array_view<uint32_t>(rt.gpu()->get_buffer_device_pointer(spatial_grid_indices), grid_builder.data.indices.count);

    rt.gpu()->destroy_buffer(spatial_grid_cells);
    spatial_grid_cells = rt.gpu()->create_buffer({sizeof(uint32_t) * grid_builder.data.cell_ends.count, gpu_data.spatial_grid.cell_ends.a});
    gpu_data.spatial_grid.cell_ends = make_array_view<uint32_t>(rt.gpu()->get_buffer_device_pointer(spatial_grid_cells), grid_builder.data.cell_ends.count);

    auto light_paths_ptr = rt.gpu()->copy_to_buffer(light_paths_buffer, light_paths.data(), 0, sizeof(VCMLightPath) * dimemsions.x * dimemsions.y);
    gpu_data.light_paths = make_array_view<VCMLightPath>(light_paths_ptr, 1llu * dimemsions.x * dimemsions.y);

    log::info(
      " - updating GPU buffers: %.2f sec (max path: %u)\n"
      "Completed: %.2f sec\n",
      tm.measure(), max_path_length, tm_total.measure());
  }

  void update_light_vertices() {
    VCMIteration it = {};
    rt.gpu()->copy_from_buffer(global_data, &it, 0, sizeof(VCMIteration));

    if ((it.active_paths == 0) || (it.light_vertices + it.active_paths > LightBufferCapacity)) {
      download_light_vertices(it.light_vertices);
      it.light_vertices = 0;
    }

    if (it.active_paths == 0) {
      rt.gpu()->launch(pipelines[VCMLibrary].first, "merge_light_image", dimemsions.x, dimemsions.y, gpu_data_ptr, sizeof(VCMGlobal));
      auto task = rt.scheduler().schedule(1u, [this](uint32_t, uint32_t, uint32_t) {
        build_spatial_grid();
      });
      rt.scheduler().wait(task);

      if (generate_camera_vertices() == false) {
        stop();
        return;
      }

      vcm_state = VCMState::GatheringCameraVertices;
      return;
    }

    // update iteration
    iteration.active_paths = 0;
    iteration.light_vertices = it.light_vertices;
    rt.gpu()->copy_to_buffer(global_data, &iteration, 0, sizeof(VCMIteration));

    // update buffers
    gpu_data.input_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[current_buffer]), 1llu * dimemsions.x * dimemsions.y);
    gpu_data.output_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[1 - current_buffer]), 1llu * dimemsions.x * dimemsions.y);
    gpu_data_ptr = rt.gpu()->copy_to_buffer(global_data, &gpu_data, sizeof(VCMIteration), sizeof(VCMGlobal));

    // launch main
    update_status("camera", it.active_paths);

    gpu_light_timer.begin();
    if (rt.gpu()->launch(pipelines[LightMain].first, it.active_paths, 1u, gpu_data_ptr, sizeof(VCMGlobal)) == false) {
      stop();
    }
    debug_info[0].value += gpu_light_timer.retreive();

    current_buffer = 1u - current_buffer;
    ++iteration_frame;
  }

  bool generate_camera_vertices() {
    iteration.active_paths = 0;
    rt.gpu()->copy_to_buffer(global_data, &iteration, 0, sizeof(VCMIteration));

    current_buffer = 0;
    iteration_frame = 0;

    gpu_data.input_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[current_buffer]), 1llu * dimemsions.x * dimemsions.y);
    gpu_data.output_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[1 - current_buffer]), 1llu * dimemsions.x * dimemsions.y);
    gpu_data_ptr = rt.gpu()->copy_to_buffer(global_data, &gpu_data, sizeof(VCMIteration), sizeof(VCMGlobal));

    return rt.gpu()->launch(pipelines[VCMLibrary].first, "gen_camera_rays", dimemsions.x, dimemsions.y, gpu_data_ptr, sizeof(VCMGlobal));
  }

  void update_camera_vertices() {
    VCMIteration it = {};
    rt.gpu()->copy_from_buffer(global_data, &it, 0, sizeof(VCMIteration));

    if (it.active_paths == 0) {
      iteration.iteration += 1;
      start_next_iteration();
      return;
    }

    iteration.active_paths = 0;
    rt.gpu()->copy_to_buffer(global_data, &iteration, 0, sizeof(VCMIteration));

    gpu_data.input_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[current_buffer]), 1llu * dimemsions.x * dimemsions.y);
    gpu_data.output_state = make_array_view<VCMPathState>(rt.gpu()->get_buffer_device_pointer(state_buffers[1 - current_buffer]), 1llu * dimemsions.x * dimemsions.y);
    gpu_data_ptr = rt.gpu()->copy_to_buffer(global_data, &gpu_data, sizeof(VCMIteration), sizeof(VCMGlobal));

    update_status("camera", it.active_paths);

    gpu_camera_timer.begin();
    if (rt.gpu()->launch(pipelines[CameraMain].first, it.active_paths, 1u, gpu_data_ptr, sizeof(VCMGlobal)) == false) {
      stop();
    }
    debug_info[1].value += gpu_camera_timer.retreive();

    current_buffer = 1u - current_buffer;
    ++iteration_frame;
  }

  void update() {
    if ((state == Integrator::State::Stopped) || (vcm_state == VCMState::Stopped)) {
      snprintf(status, sizeof(status), "Stopped");
      state = Integrator::State::Stopped;
      return;
    }

    if (vcm_state == VCMState::GatheringLightVertices) {
      update_light_vertices();
    } else if (vcm_state == VCMState::GatheringCameraVertices) {
      update_camera_vertices();
    }
  }
};

GPUVCM::GPUVCM(Raytracing& r)
  : Integrator(r) {
  ETX_PIMPL_INIT(GPUVCM, r, current_state);
}

GPUVCM::~GPUVCM() {
  ETX_PIMPL_CLEANUP(GPUVCM);
}

bool GPUVCM::enabled() const {
  return true;
}

const char* GPUVCM::status() const {
  return _private->status;
}

Options GPUVCM::options() const {
  Options opt;
  _private->options.store(opt);
  opt.add("reload", "Reload pipelines on next launch:");
  for (uint32_t i = 0; i < GPUVCMImpl::PipelineCount; ++i) {
    opt.add(_private->reload_pipelines[i], _private->pipelines[i].second, _private->pipelines[i].second + 10);
  }
  return opt;
}

void GPUVCM::set_output_size(const uint2& size) {
  _private->build_output(size);
}

void GPUVCM::preview(const Options& opt) {
  if (_private->build_pipelines(opt) == false)
    return;

  if (current_state == State::Preview) {
    stop(Stop::Immediate);
  }

  if (_private->run(opt)) {
    current_state = State::Preview;
  }
}

void GPUVCM::run(const Options& opt) {
  if (_private->build_pipelines(opt) == false)
    return;

  if ((current_state != State::Running) && _private->run(opt)) {
    current_state = State::Running;
  }
}

void GPUVCM::update() {
  constexpr float kDeltaTime = 1.0f / 30.0f;

  TimeMeasure tm = {};

  _private->debug_info[0].value = 0.0f;
  _private->debug_info[1].value = 0.0f;
  while ((current_state != State::Stopped) && (tm.measure() < kDeltaTime)) {
    _private->update();
  }

  ++_private->update_frame;
}

void GPUVCM::stop(Stop stop) {
  current_state = State::Stopped;
}

void GPUVCM::update_options(const Options&) {
}

bool GPUVCM::have_updated_camera_image() const {
  return true;
}

const float4* GPUVCM::get_camera_image(bool /* force update */) {
  if (_private->camera_image.handle == kInvalidHandle)
    return nullptr;

  rt.gpu()->copy_from_buffer(_private->camera_image, _private->local_camera_image.data(), 0llu, _private->local_camera_image.size() * sizeof(float4));
  return _private->local_camera_image.data();
}

bool GPUVCM::have_updated_light_image() const {
  return true;
}

const float4* GPUVCM::get_light_image(bool /* force update */) {
  if (_private->light_final_image.handle == kInvalidHandle)
    return nullptr;

  rt.gpu()->copy_from_buffer(_private->light_final_image, _private->local_light_image.data(), 0llu, _private->local_light_image.size() * sizeof(float4));
  return _private->local_light_image.data();
}

void GPUVCM::reload() {
  stop(Stop::Immediate);
  _private->destroy_pipelines();
  preview({});
}

uint64_t GPUVCM::debug_info_count() const {
  return std::size(_private->debug_info);
}

Integrator::DebugInfo* GPUVCM::debug_info() const {
  return _private->debug_info;
}

}  // namespace etx
