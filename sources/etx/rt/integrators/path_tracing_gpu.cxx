#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/log/log.hxx>

#include <etx/rt/integrators/path_tracing_gpu.hxx>
#include <etx/rt/shared/path_tracing_shared.hxx>

#include <etx/gpu/gpu.hxx>

namespace etx {

struct ETX_ALIGNED GPUPathTracingImpl {
  PTGPUData gpu_data = {};

  GPUBuffer camera_image = {};
  GPUBuffer payload_buffer = {};
  GPUPipeline raygen_pipeline = {};
  GPUPipeline main_pipeline = {};
  std::vector<float4> local_camera_image = {};
  uint2 output_size = {};
  device_pointer_t gpu_launch_params = {};
  float frame_time_ms = 30.0f;

  bool start(const Options& opt, Raytracing& rt) {
    frame_time_ms = opt.get("framems", frame_time_ms).to_float();
    gpu_data.options.iterations = opt.get("spp", gpu_data.options.iterations).to_integer();
    gpu_data.options.max_depth = opt.get("pathlen", gpu_data.options.max_depth).to_integer();
    gpu_data.options.rr_start = opt.get("rrstart", gpu_data.options.rr_start).to_integer();
    gpu_data.options.path_per_iteration = opt.get("plen", gpu_data.options.path_per_iteration).to_integer();
    return start(rt, false);
  }

  bool start(Raytracing& rt, bool recompile) {
    if (main_pipeline.handle == kInvalidHandle) {
      main_pipeline = rt.gpu()->create_pipeline_from_file(env().file_in_data("optix/pt/main.json"), recompile);
      if (main_pipeline.handle == kInvalidHandle) {
        log::error("GPU Path Tracing failed to compile main pipeline");
        return false;
      }
    }

    if (raygen_pipeline.handle == kInvalidHandle) {
      raygen_pipeline = rt.gpu()->create_pipeline_from_file(env().file_in_data("optix/pt/raygen.json"), recompile);
      if (raygen_pipeline.handle == kInvalidHandle) {
        log::error("GPU Path Tracing failed to compile raygen pipeline");
        return false;
      }
    }

    gpu_data.payloads = reinterpret_cast<PTRayPayload*>(rt.gpu()->get_buffer_device_pointer(payload_buffer));
    gpu_data.output = reinterpret_cast<float4*>(rt.gpu()->get_buffer_device_pointer(camera_image));
    gpu_data.scene = rt.gpu_scene();

    gpu_launch_params = rt.gpu()->upload_to_shared_buffer(gpu_launch_params, &gpu_data, sizeof(gpu_data));
    return rt.gpu()->launch(raygen_pipeline, output_size.x, output_size.y, gpu_launch_params, sizeof(gpu_data));
  }

  void frame(Raytracing& rt) {
    TimeMeasure tm;
    while (tm.measure() <= frame_time_ms / 1000.0f) {
      rt.gpu()->launch(main_pipeline, output_size.x, output_size.y, gpu_launch_params, sizeof(gpu_data));
    }
  }
};

GPUPathTracing::GPUPathTracing(Raytracing& r)
  : Integrator(r) {
  ETX_PIMPL_INIT(GPUPathTracing);
}

GPUPathTracing::~GPUPathTracing() {
  rt.gpu()->destroy_buffer(_private->camera_image);
  rt.gpu()->destroy_buffer(_private->payload_buffer);
  rt.gpu()->destroy_pipeline(_private->raygen_pipeline);
  rt.gpu()->destroy_pipeline(_private->main_pipeline);
  ETX_PIMPL_CLEANUP(GPUPathTracing);
}

Options GPUPathTracing::options() const {
  Options result = {};
  result.add(1u, _private->gpu_data.options.iterations, 0xffffu, "spp", "Samples per Pixel");
  result.add(1u, _private->gpu_data.options.max_depth, 65536u, "pathlen", "Maximal Path Length");
  result.add(1u, _private->gpu_data.options.rr_start, 65536u, "rrstart", "Start Russian Roulette at");
  result.add(1u, _private->gpu_data.options.path_per_iteration, 32u, "plen", "Path Length per iteration");
  result.add(1.0f, _private->frame_time_ms, 1000.0f, "framems", "CPU frame time (ms)");
  return result;
}

void GPUPathTracing::set_output_size(const uint2& size) {
  rt.gpu()->destroy_buffer(_private->camera_image);
  rt.gpu()->destroy_buffer(_private->payload_buffer);
  _private->output_size = size;
  _private->payload_buffer = rt.gpu()->create_buffer({size.x * size.y * sizeof(PTRayPayload)});
  _private->camera_image = rt.gpu()->create_buffer({size.x * size.y * sizeof(float4)});
  _private->local_camera_image.resize(1llu * size.x * size.y);
}

void GPUPathTracing::preview(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = _private->start(opt, rt) ? State::Preview : State::Stopped;
  }
}

void GPUPathTracing::run(const Options& opt) {
  stop(Stop::Immediate);

  if (rt.has_scene()) {
    current_state = _private->start(opt, rt) ? State::Running : State::Stopped;
  }
}

void GPUPathTracing::update() {
  if ((current_state != State::Preview) && (current_state != State::Running)) {
    return;
  }

  _private->frame(rt);
}

void GPUPathTracing::stop(Stop mode) {
  if (current_state == State::Stopped) {
    return;
  }

  current_state = State::Stopped;
}

void GPUPathTracing::update_options(const Options& opt) {
  if (current_state == State::Preview) {
    preview(opt);
  }
}

const float4* GPUPathTracing::get_camera_image(bool /* force */) {
  if (_private->camera_image.handle == kInvalidHandle)
    return nullptr;

  rt.gpu()->copy_from_buffer(_private->camera_image, _private->local_camera_image.data(), 0llu, _private->local_camera_image.size() * sizeof(float4));
  return _private->local_camera_image.data();
}

void GPUPathTracing::reload() {
  stop(Stop::Immediate);

  rt.gpu()->destroy_pipeline(_private->raygen_pipeline);
  _private->raygen_pipeline = {};

  rt.gpu()->destroy_pipeline(_private->main_pipeline);
  _private->main_pipeline = {};

  _private->start(rt, true);
}

}  // namespace etx