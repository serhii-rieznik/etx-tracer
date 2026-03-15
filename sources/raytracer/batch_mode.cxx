#include "batch_mode.hxx"

#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/host/scene_global.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/rt/integrators/integrator.hxx>
#include <etx/rt/rt.hxx>

#include "cpu_renderer.hxx"
#include "headless_render_context.hxx"
#include "image_output.hxx"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <thread>

namespace etx {

namespace {

bool parse_u32_argument(const char* value, uint32_t& result) {
  if ((value == nullptr) || (value[0] == 0)) {
    return false;
  }

  char* end_ptr = nullptr;
  unsigned long parsed_value = std::strtoul(value, &end_ptr, 10);
  if ((end_ptr == nullptr) || (end_ptr[0] != 0)) {
    return false;
  }
  if (parsed_value > 0xFFFFFFFFul) {
    return false;
  }

  result = static_cast<uint32_t>(parsed_value);
  return true;
}

bool parse_f32_argument(const char* value, float& result) {
  if ((value == nullptr) || (value[0] == 0)) {
    return false;
  }

  char* end_ptr = nullptr;
  result = std::strtof(value, &end_ptr);
  if ((end_ptr == nullptr) || (end_ptr[0] != 0)) {
    return false;
  }

  return true;
}

const char* batch_usage_string() {
  return
    "Usage:\n"
    "  raytracer --render --scene <scene-file> --output <output-file> [options]\n"
    "\n"
    "Options:\n"
    "  --integrator <debug|pt|bdpt|vcm|bdpt_distilled>\n"
    "  --renderer <cpu>\n"
    "  --samples <count>\n"
    "  --reference <reference-image>\n"
    "  --compare <render>\n"
    "  --denoise\n"
    "  --exposure <value>\n"
    "  --help\n";
}

std::string resolve_input_path(const std::string& path) {
  if (path.empty()) {
    return {};
  }

  std::filesystem::path input_path(path);
  std::error_code ec = {};
  if (input_path.is_absolute()) {
    const std::filesystem::path canonical_path = std::filesystem::weakly_canonical(input_path, ec);
    if (ec.value() == 0) {
      return canonical_path.generic_string();
    }
    const std::filesystem::path absolute_path = std::filesystem::absolute(input_path, ec);
    return absolute_path.generic_string();
  }

  if (std::filesystem::exists(input_path, ec)) {
    const std::filesystem::path canonical_path = std::filesystem::weakly_canonical(input_path, ec);
    if (ec.value() == 0) {
      return canonical_path.generic_string();
    }
    const std::filesystem::path absolute_path = std::filesystem::absolute(input_path, ec);
    return absolute_path.generic_string();
  }

  return env().resolve_to_absolute(path);
}

struct BatchRenderSession {
  BatchRenderSession()
    : film(scheduler)
    , rt(scheduler, film)
    , scene(scheduler, ior_database)
    , cpu_renderer(rt, scene) {
  }

  ~BatchRenderSession() {
    cleanup();
  }

  bool init() {
    scene_global_init();
    render_context.init();
    if (render_context.context().valid() == false) {
      log::error("Failed to initialize headless RHI context");
      return false;
    }

    scene.set_scattering_rhi(render_context.context());

    std::string ior_folder = env().file_in_data("./spectrum/");
    ior_database.load(ior_folder.c_str());

    cpu_renderer.init(render_context.context(), scene);
    return true;
  }

  void cleanup() {
    if (render_context.context().valid()) {
      cpu_renderer.cleanup(render_context.context());
    }
    scene_global_deinit();
    render_context.cleanup();
  }

  TaskScheduler scheduler;
  Film film;
  Raytracing rt;
  IORDatabase ior_database;
  SceneRepresentation scene;
  HeadlessRenderContext render_context;
  CPURaytracingRenderer cpu_renderer;
};

Integrator* select_integrator(const BatchRenderOptions& options, BatchRenderSession& session, const SceneRepresentation::IntegratorData& integrator_data) {
  Integrator* selected_integrator = nullptr;

  if (options.integrator.empty() == false) {
    const Integrator::Type requested_type = integrator_id_to_type(options.integrator.c_str());
    selected_integrator = integrator_type_to_instance(requested_type, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
    if (selected_integrator == nullptr) {
      log::error("Unknown integrator: %s", options.integrator.c_str());
      return nullptr;
    }
  }

  if ((selected_integrator == nullptr) && (integrator_data.selected != Integrator::Type::Invalid)) {
    selected_integrator =
      integrator_type_to_instance(integrator_data.selected, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
  }

  if ((selected_integrator == nullptr) && (session.cpu_renderer.integrator_count() > 1u)) {
    selected_integrator = session.cpu_renderer.integrator_list()[1];
  }

  return selected_integrator;
}

bool load_scene_for_batch(const BatchRenderOptions& options, BatchRenderSession& session, Integrator*& selected_integrator) {
  SceneRepresentation::IntegratorData integrator_data = {};
  const std::string absolute_scene_path = resolve_input_path(options.scene_file);
  if (session.scene.load_from_file(absolute_scene_path.c_str(), SceneRepresentation::LoadEverything, &integrator_data) == false) {
    log::error("Failed to load scene from file: %s", absolute_scene_path.c_str());
    return false;
  }

  if (session.scene.valid() == false) {
    log::error("Scene is invalid after loading: %s", absolute_scene_path.c_str());
    return false;
  }

  session.cpu_renderer.set_output_dimensions(session.render_context.context(), session.scene.camera().film_size);

  for (const auto& [type, options_data] : integrator_data.settings) {
    Integrator* integrator =
      integrator_type_to_instance(type, session.cpu_renderer.integrator_list(), session.cpu_renderer.integrator_count());
    if (integrator != nullptr) {
      integrator->sync_from_options(options_data);
      integrator->update_options();
    }
  }

  selected_integrator = select_integrator(options, session, integrator_data);
  if (selected_integrator == nullptr) {
    log::error("Failed to select an integrator for batch rendering");
    return false;
  }

  session.cpu_renderer.set_integrator(selected_integrator);

  if (options.samples > 0u) {
    session.scene.data().options.samples = options.samples;
  }

  return true;
}

bool run_cpu_batch_render(const BatchRenderOptions& options, BatchRenderSession& session) {
  if (options.renderer != "cpu") {
    log::error("Unsupported batch renderer: %s", options.renderer.c_str());
    return false;
  }

  Integrator* selected_integrator = nullptr;
  if (load_scene_for_batch(options, session, selected_integrator) == false) {
    return false;
  }

  ETX_ASSERT(selected_integrator != nullptr);

  session.cpu_renderer.film().clear(Film::ClearEverything);
  session.cpu_renderer.start();

  uint32_t last_completed_iterations = 0u;
  const uint32_t target_iterations = session.scene.data().options.samples;

  while (true) {
    session.cpu_renderer.integrator_thread().update();

    const Integrator::Status& status = session.cpu_renderer.integrator_thread().status();
    if (status.completed_iterations != last_completed_iterations) {
      log::info("Rendering progress: %u / %u", status.completed_iterations, target_iterations);
      last_completed_iterations = status.completed_iterations;
    }

    const bool render_finished = (session.cpu_renderer.is_running() == false) && (status.completed_iterations >= target_iterations);
    if (render_finished) {
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  uint32_t output_layer = ViewLayer::Result;
  if (options.denoise) {
    session.cpu_renderer.film().denoise(ViewLayer::Result, session.rt.scene().options.radiance_clamp);
    output_layer = ViewLayer::Denoised;
  }

  const float4* output = session.cpu_renderer.film().layer(output_layer, session.rt.scene().options.radiance_clamp);
  const uint2 image_size = session.scene.camera().film_size;
  ImageOutputParameters output_params = {
    .mode = save_image_mode_from_file_name(options.output_file),
    .exposure = options.exposure,
  };
  if (save_image_to_file(options.output_file, output, image_size, output_params) == false) {
    return false;
  }

  if (options.compare_mode == "render") {
    std::vector<float4> reference_image = {};
    uint2 reference_image_size = {};
    if (load_hdr_image_from_file(options.reference_file, reference_image, reference_image_size) == false) {
      return false;
    }

    if ((reference_image_size.x != image_size.x) || (reference_image_size.y != image_size.y)) {
      log::error("Reference image size mismatch: reference=%ux%u, render=%ux%u", reference_image_size.x, reference_image_size.y, image_size.x, image_size.y);
      return false;
    }

    std::vector<float4> difference_image = {};
    ImageComparisonResult comparison = {};
    if (compare_images(reference_image.data(), output, image_size, difference_image, comparison) == false) {
      return false;
    }

    ImageOutputParameters difference_params = {
      .mode = SaveImageMode::RGB,
      .exposure = 1.0f,
    };
    const std::string difference_file_name = comparison_file_name_from_output(options.output_file);
    if (save_image_to_file(difference_file_name, difference_image.data(), image_size, difference_params) == false) {
      return false;
    }

    log::info("Image comparison: similarity=%.2f%%, rmse=%.6f, mae=%.6f, relative_rmse=%.6f, max_abs=%.6f", comparison.similarity,
      comparison.root_mean_squared_error, comparison.mean_absolute_error, comparison.relative_root_mean_squared_error, comparison.max_absolute_error);
  }

  return true;
}

}  // namespace

BatchModeCommand parse_batch_command_line(int argc, char* argv[], BatchRenderOptions& options, std::string& message) {
  bool render_requested = false;
  bool batch_argument_seen = false;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];

    if ((argument == "--help") || (argument == "-h")) {
      message = batch_usage_string();
      return BatchModeCommand::Help;
    }

    if (argument == "--render") {
      render_requested = true;
      batch_argument_seen = true;
      continue;
    }

    if (argument == "--scene") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --scene\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.scene_file = argv[++i];
      continue;
    }

    if (argument == "--output") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --output\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.output_file = argv[++i];
      continue;
    }

    if (argument == "--reference") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --reference\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.reference_file = argv[++i];
      continue;
    }

    if (argument == "--integrator") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --integrator\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.integrator = argv[++i];
      continue;
    }

    if (argument == "--renderer") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --renderer\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      options.renderer = argv[++i];
      continue;
    }

    if (argument == "--samples") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --samples\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (parse_u32_argument(argv[i + 1], options.samples) == false) {
        message = "Invalid value for --samples\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      i += 1;
      continue;
    }

    if (argument == "--exposure") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --exposure\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      if (parse_f32_argument(argv[i + 1], options.exposure) == false) {
        message = "Invalid value for --exposure\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      i += 1;
      continue;
    }

    if (argument == "--compare") {
      batch_argument_seen = true;
      if ((i + 1) >= argc) {
        message = "Missing value for --compare\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }

      options.compare_mode = argv[++i];
      if (options.compare_mode != "render") {
        message = "Unsupported value for --compare. Expected: render\n\n";
        message += batch_usage_string();
        return BatchModeCommand::Error;
      }
      continue;
    }

    if (argument == "--denoise") {
      batch_argument_seen = true;
      options.denoise = true;
      continue;
    }

    message = "Unknown argument: " + argument + "\n\n";
    message += batch_usage_string();
    return BatchModeCommand::Error;
  }

  if (render_requested == false) {
    if (batch_argument_seen) {
      message = "Batch render options require --render\n\n";
      message += batch_usage_string();
      return BatchModeCommand::Error;
    }
    return BatchModeCommand::None;
  }

  if ((options.scene_file.empty()) || (options.output_file.empty())) {
    message = "Batch rendering requires --scene and --output\n\n";
    message += batch_usage_string();
    return BatchModeCommand::Error;
  }

  if ((options.compare_mode.empty() == false) && (options.reference_file.empty())) {
    message = "Image comparison requires --reference\n\n";
    message += batch_usage_string();
    return BatchModeCommand::Error;
  }

  return BatchModeCommand::Run;
}

int run_batch_render(const BatchRenderOptions& options) {
  BatchRenderSession session = {};
  if (session.init() == false) {
    return 1;
  }

  if (run_cpu_batch_render(options, session) == false) {
    return 1;
  }

  return 0;
}

}  // namespace etx
