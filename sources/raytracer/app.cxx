#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>

#include <etx/render/host/scene_global.hxx>
#include <etx/render/interop/bsdf_energy_compensation_constants_shared.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/rt/integrators/integrator.hxx>

#include "app.hxx"
#include "image_output.hxx"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

#if defined(ETX_PLATFORM_WINDOWS)
# define WIN32_LEAN_AND_MEAN 1
# include <Windows.h>
#endif

#include <interop/render_options.hxx>

namespace etx {

namespace {

constexpr uint32_t kRecentFileLimit = 8u;
constexpr size_t kRetainedApplicationCommandResultLimit = 256u;

bool energy_compensation_cache_matches_render_mode(const SceneData& data) {
  const uint32_t expected_cache_mode =
    data.options.properties[Scene::Properties::Spectral] ? kBSDFEnergyCompensationCacheModeSpectralScalar : kBSDFEnergyCompensationCacheModeIntegratedRGB;
  for (const Scene::EnergyCompensationInterface& interface_data : data.energy_compensation_interfaces) {
    if (interface_data.cache_mode != expected_cache_mode) {
      return false;
    }
  }
  return true;
}

bool read_texture_to_float4_buffer(RHIContext& ctx, RHITexture texture, const uint2 image_size, std::vector<float4>& output) {
  output.clear();
  if ((texture.valid() == false) || (image_size.x == 0u) || (image_size.y == 0u)) {
    log::warning("No renderer output image is available for capture");
    return false;
  }

  const uint64_t pixel_count = static_cast<uint64_t>(image_size.x) * static_cast<uint64_t>(image_size.y);
  const uint64_t buffer_size = pixel_count * sizeof(float4);
  RHIBufferDesc readback_desc = {};
  readback_desc.size = buffer_size;
  readback_desc.usage = RHIBufferUsage::TransferDst;
  readback_desc.host_visible = true;

  auto readback_result = ctx.device().create_buffer(readback_desc);
  if ((readback_result.result != RHIResult::Success) || (readback_result.handle.valid() == false)) {
    log::error("Failed to create renderer image capture buffer (%u)", static_cast<uint32_t>(readback_result.result));
    return false;
  }

  const RHIResult idle_wait = ctx.wait_idle();
  if (idle_wait != RHIResult::Success) {
    log::error("Failed to wait before renderer image capture (%u)", static_cast<uint32_t>(idle_wait));
    ctx.device().destroy_buffer(readback_result.handle);
    return false;
  }

  RHICommandBuffer cmd = ctx.get_command_buffer();
  if (cmd.valid() == false) {
    log::error("Failed to get command buffer for renderer image capture");
    ctx.device().destroy_buffer(readback_result.handle);
    return false;
  }

  ctx.command_buffer_begin(cmd);
  ctx.cmd_texture_barrier(cmd, texture, RHIResourceState::ShaderReadOnly, RHIResourceState::TransferSrc);
  ctx.cmd_copy_texture_to_buffer(cmd, texture, readback_result.handle, image_size.x, image_size.y, 0u);
  ctx.cmd_texture_barrier(cmd, texture, RHIResourceState::TransferSrc, RHIResourceState::ShaderReadOnly);
  ctx.command_buffer_end(cmd);
  ctx.submit_command_buffer({cmd});

  const RHIResult capture_wait = ctx.wait_for_command_buffer(cmd);
  if (capture_wait != RHIResult::Success) {
    log::error("Renderer image capture wait failed (%u)", static_cast<uint32_t>(capture_wait));
    ctx.destroy_command_buffer(cmd);
    ctx.device().destroy_buffer(readback_result.handle);
    return false;
  }
  ctx.destroy_command_buffer(cmd);

  output.resize(static_cast<size_t>(pixel_count));
  const RHIResult read_result = ctx.device().read_buffer(readback_result.handle, output.data(), buffer_size, 0u);
  const RHIResult destroy_result = ctx.device().destroy_buffer(readback_result.handle);
  if (destroy_result != RHIResult::Success) {
    log::warning("Failed to destroy renderer image capture buffer (%u)", static_cast<uint32_t>(destroy_result));
  }

  if (read_result != RHIResult::Success) {
    log::error("Failed to read renderer image capture buffer (%u)", static_cast<uint32_t>(read_result));
    output.clear();
    return false;
  }

  return true;
}

std::string normalized_existing_scene_path(const std::string& value) {
  if (value.empty()) {
    return {};
  }

  std::filesystem::path path(env().resolve_to_absolute(value));
  if (path.empty()) {
    return {};
  }

  std::error_code ec = {};
  const bool scene_file_exists = std::filesystem::is_regular_file(path, ec);
  if (scene_file_exists == false) {
    static constexpr const char* recovery_suffixes[] = {".save-preparing", ".save-pending", ".save-committed", ".save-backup"};
    bool recovery_file_exists = false;
    for (const char* suffix : recovery_suffixes) {
      ec.clear();
      recovery_file_exists = std::filesystem::is_regular_file(path.string() + suffix, ec);
      if (recovery_file_exists) {
        break;
      }
    }
    if (recovery_file_exists == false) {
      return {};
    }
  }

  const std::filesystem::path canonical_path = std::filesystem::weakly_canonical(path, ec);
  if (ec.value() == 0) {
    path = canonical_path;
  }

  return path.generic_string();
}

std::string portable_scene_path(const std::string& value) {
  if (!env().bundled() || value.empty()) {
    return value;
  }

  std::error_code ec = {};
  const std::filesystem::path resource_root = std::filesystem::weakly_canonical(env().data_folder(), ec);
  if (ec) {
    return value;
  }

  const std::filesystem::path scene_path = std::filesystem::weakly_canonical(value, ec);
  if (ec) {
    return value;
  }

  const std::filesystem::path relative_path = scene_path.lexically_relative(resource_root);
  if (relative_path.empty() || (*relative_path.begin() == "..")) {
    return value;
  }
  return relative_path.generic_string();
}

}  // namespace

RTApplication::RTApplication()
  : rt(scheduler, film)
  , film(scheduler)
  , scene(scheduler, _ior_database)
  , render_context(scheduler)
  , cpu_renderer(rt, scene)
  , raster_renderer(scheduler)
  , gpu_renderer(scheduler) {
  _active_renderer = &cpu_renderer;
}

RTApplication::~RTApplication() {
  if (_initialized) {
    save_options();
  }
}

uint64_t RTApplication::submit_command(ApplicationCommand command) {
  std::lock_guard<std::mutex> lock(_application_control_mutex);
  command.id = _next_application_command_id.fetch_add(1u);
  const uint64_t command_id = command.id;
  _pending_application_commands.push_back(std::move(command));
  return command_id;
}

ApplicationStateSnapshot RTApplication::state_snapshot() const {
  std::lock_guard<std::mutex> lock(_application_control_mutex);
  return _application_state;
}

void RTApplication::drain_command_results(std::vector<ApplicationCommandResult>& output) {
  std::lock_guard<std::mutex> lock(_application_control_mutex);
  output = std::move(_application_command_results);
  _application_command_results.clear();
}

bool RTApplication::quit_requested() const {
  std::lock_guard<std::mutex> lock(_application_control_mutex);
  return _application_quit_requested;
}

bool RTApplication::initialized() const {
  return _initialized.load();
}

bool RTApplication::capture_output_png(std::vector<uint8_t>& png_data, uint32_t& width, uint32_t& height) {
  if (_current_scene_file.empty() || (_active_renderer == nullptr) || (_active_renderer->display_texture().valid() == false)) {
    png_data.clear();
    width = 0u;
    height = 0u;
    return false;
  }
  return render_context.capture_output_png(png_data, width, height);
}

void RTApplication::prepare_startup() {
  platform_ui().show_startup();
  if (!platform_ui().defers_initialization()) {
    init();
  }
}

void RTApplication::init(const ApplicationConfig& config) {
  ETX_PROFILER_SCOPE();

  _application_config = config;
  _initialization_started = true;
  scene_global_init();
  _scene_global_initialized = true;

  {
    ETX_PROFILER_NAMED_SCOPE("app_load_options");
    const std::string options_file = env().file_in_config("options.json");
    if (std::filesystem::exists(options_file)) {
      _options.load_from_file(options_file);
    } else if (env().bundled()) {
      _options.load_from_file(env().file_in_data("DefaultOptions.json"));
    }
  }

  {
    ETX_PROFILER_NAMED_SCOPE("app_init_render_context_and_ior");
    render_context.init({
      .mode = config.runtime_mode,
      .width = config.width,
      .height = config.height,
      .enable_imgui = config.enable_imgui,
    });
    if (render_context.valid() == false) {
      log::error("Failed to initialize rendering context");
      return;
    }
    if (config.enable_platform_ui) {
      sync_platform_color_scheme();
    }
    scene.set_scattering_rhi(render_context.get_context());
    _gpu_renderer_supported = render_context.get_context().capabilities().supports_ray_tracing;
    ui.set_gpu_renderer_available(_gpu_renderer_supported);
    sync_ui_renderer_state();
    if (_gpu_renderer_supported == false) {
      log::warning("GPU ray tracing is not supported by the active RHI backend; falling back to CPU or raster rendering");
    }
    std::string ior_folder = env().file_in_data("./spectrum/");
    _ior_database.load(ior_folder.c_str());
  }

  {
    ETX_PROFILER_NAMED_SCOPE("app_init_renderers");
    ui.set_integrator_list(cpu_renderer.integrator_list(), cpu_renderer.integrator_count());

    cpu_renderer.init(render_context.get_context(), scene);
    raster_renderer.init(render_context.get_context(), scene);
  }

  gpu_renderer.set_wavefront_auto_tuning(true);

  RendererMode mode = config.override_renderer ? config.renderer : RendererMode::CPURaytracing;
  if (!config.override_renderer) {
    auto renderer_name = _options.get_string("renderer", "cpu");
    if (renderer_name == "gpu") {
      mode = RendererMode::GPURaytracing;
    } else if (renderer_name == "raster") {
      mode = RendererMode::Rasterization;
    }
  }
  {
    ETX_PROFILER_NAMED_SCOPE("app_bind_ui_callbacks");
    ui.callbacks.quit_selected = [this]() {
      _scene_dirty = false;
      submit_command({.type = ApplicationCommandType::Quit});
    };
    ui.callbacks.reference_image_selected = [this](std::string path) {
      submit_command({.type = ApplicationCommandType::LoadReferenceImage, .path = std::move(path)});
    };
    ui.callbacks.save_image_selected = [this](std::string path, SaveImageMode mode) {
      submit_command({.type = ApplicationCommandType::SaveImage, .path = std::move(path), .save_image_mode = mode});
    };
    ui.callbacks.scene_file_selected = [this](std::string path) {
      submit_command({.type = ApplicationCommandType::LoadScene, .path = std::move(path)});
    };
    ui.callbacks.save_scene_file_selected = [this](std::string path) {
      std::string message = {};
      return execute_application_command({.type = ApplicationCommandType::SaveScene, .path = std::move(path)}, message);
    };
    ui.callbacks.render_configuration_selected = [this](RendererMode mode, Integrator::Type integrator_type) {
      submit_command({.type = ApplicationCommandType::SetRenderConfiguration, .renderer = mode, .integrator = integrator_type});
    };
    ui.callbacks.run_selected = [this]() {
      submit_command({.type = ApplicationCommandType::Run});
    };
    ui.callbacks.stop_selected = [this](bool finish) {
      submit_command({.type = finish ? ApplicationCommandType::Finish : ApplicationCommandType::Stop});
    };
    ui.callbacks.restart_selected = [this]() {
      submit_command({.type = ApplicationCommandType::Restart});
    };
    ui.callbacks.reload_scene_selected = [this]() {
      submit_command({.type = ApplicationCommandType::ReloadScene});
    };
    ui.callbacks.reload_geometry_selected = [this]() {
      submit_command({.type = ApplicationCommandType::ReloadGeometry});
    };
    ui.callbacks.options_changed = std::bind(&RTApplication::on_options_changed, this);
    ui.callbacks.reload_shaders_selected = [this]() {
      submit_command({.type = ApplicationCommandType::ReloadShaders});
    };
    ui.callbacks.cancel_renderer_preparation_selected = [this]() {
      submit_command({.type = ApplicationCommandType::CancelPreparation});
    };
    ui.callbacks.use_image_as_reference = std::bind(&RTApplication::on_use_image_as_reference, this);
    ui.callbacks.material_added = std::bind(&RTApplication::on_material_added, this);
    ui.callbacks.material_duplicated = std::bind(&RTApplication::on_material_duplicated, this, std::placeholders::_1);
    ui.callbacks.material_deleted = std::bind(&RTApplication::on_material_deleted, this, std::placeholders::_1);
    ui.callbacks.material_renamed = std::bind(&RTApplication::on_material_renamed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.material_changed = std::bind(&RTApplication::on_material_changed, this, std::placeholders::_1);
    ui.callbacks.material_interaction_started = std::bind(&RTApplication::on_material_interaction_started, this);
    ui.callbacks.material_interaction_finished = std::bind(&RTApplication::on_material_interaction_finished, this, std::placeholders::_1);
    ui.callbacks.medium_added = std::bind(&RTApplication::on_medium_added, this);
    ui.callbacks.medium_duplicated = std::bind(&RTApplication::on_medium_duplicated, this, std::placeholders::_1);
    ui.callbacks.medium_deleted = std::bind(&RTApplication::on_medium_deleted, this, std::placeholders::_1);
    ui.callbacks.medium_renamed = std::bind(&RTApplication::on_medium_renamed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.medium_changed = std::bind(&RTApplication::on_medium_changed, this, std::placeholders::_1);
    ui.callbacks.medium_interaction_started = std::bind(&RTApplication::on_medium_interaction_started, this);
    ui.callbacks.medium_interaction_finished = std::bind(&RTApplication::on_medium_interaction_finished, this, std::placeholders::_1);
    ui.callbacks.mesh_material_changed = std::bind(&RTApplication::on_mesh_material_changed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.mesh_material_made_unique = std::bind(&RTApplication::on_make_mesh_material_unique, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.emitter_changed = std::bind(&RTApplication::on_emitter_changed, this, std::placeholders::_1);
    ui.callbacks.emitter_interaction_started = std::bind(&RTApplication::on_emitter_interaction_started, this);
    ui.callbacks.emitter_interaction_finished = std::bind(&RTApplication::on_emitter_interaction_finished, this, std::placeholders::_1);
    ui.callbacks.emitter_added = std::bind(&RTApplication::on_emitter_added, this, std::placeholders::_1);
    ui.callbacks.emitter_duplicated = std::bind(&RTApplication::on_emitter_duplicated, this, std::placeholders::_1);
    ui.callbacks.emitter_deleted = std::bind(&RTApplication::on_emitter_deleted, this, std::placeholders::_1);
    ui.callbacks.emitter_renamed = std::bind(&RTApplication::on_emitter_renamed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.camera_added = std::bind(&RTApplication::on_camera_added, this);
    ui.callbacks.camera_duplicated = std::bind(&RTApplication::on_camera_duplicated, this, std::placeholders::_1);
    ui.callbacks.camera_deleted = std::bind(&RTApplication::on_camera_deleted, this, std::placeholders::_1);
    ui.callbacks.camera_renamed = std::bind(&RTApplication::on_camera_renamed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.empty_node_added = std::bind(&RTApplication::on_empty_node_added, this);
    ui.callbacks.primitive_added = std::bind(&RTApplication::on_primitive_added, this, std::placeholders::_1);
    ui.callbacks.node_duplicated = std::bind(&RTApplication::on_node_duplicated, this, std::placeholders::_1);
    ui.callbacks.node_deleted = std::bind(&RTApplication::on_node_deleted, this, std::placeholders::_1);
    ui.callbacks.node_reparented = std::bind(&RTApplication::on_node_reparented, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.node_enabled_changed = std::bind(&RTApplication::on_node_enabled_changed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.node_transform_changed = std::bind(&RTApplication::on_node_transform_changed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.node_geometry_edited = std::bind(&RTApplication::on_node_geometry_edited, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.node_resource_attached = std::bind(&RTApplication::on_node_resource_attached, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    ui.callbacks.node_resource_detached = std::bind(&RTApplication::on_node_resource_detached, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.node_renamed = std::bind(&RTApplication::on_node_renamed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.camera_changed = std::bind(&RTApplication::on_camera_changed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.scene_settings_changed = std::bind(&RTApplication::on_scene_settings_changed, this);
    ui.callbacks.scene_modified = std::bind(&RTApplication::mark_scene_dirty, this);
    ui.callbacks.scene_transforms_changed = std::bind(&RTApplication::on_scene_transforms_changed, this);
    ui.callbacks.scene_transform_interaction_started = std::bind(&RTApplication::on_scene_transform_interaction_started, this);
    ui.callbacks.scene_transform_interaction_finished = std::bind(&RTApplication::on_scene_transform_interaction_finished, this);
    ui.callbacks.denoise_selected = [this]() {
      submit_command({.type = ApplicationCommandType::Denoise});
    };
    ui.callbacks.view_scene = std::bind(&RTApplication::on_view_scene, this, std::placeholders::_1);
    ui.callbacks.clear_recent_files = std::bind(&RTApplication::on_clear_recent_files, this);
    ui.callbacks.camera_activated = std::bind(&RTApplication::on_camera_activated, this, std::placeholders::_1);
    ui.callbacks.gpu_kernel_timing_enabled_changed = [this](bool value) {
      gpu_renderer.set_kernel_timing_enabled(value);
    };
    ui.callbacks.exposure_changed = [this](float value) {
      submit_command({.type = ApplicationCommandType::SetExposure, .float_value = value});
    };
    ui.callbacks.view_layer_changed = [this](uint32_t value) {
      submit_command({.type = ApplicationCommandType::SetViewLayer, .unsigned_value = value});
    };
    ui.callbacks.output_view_changed = [this](uint32_t value) {
      submit_command({.type = ApplicationCommandType::SetOutputView, .unsigned_value = value});
    };
    ui.callbacks.display_transform_changed = [this](uint32_t value) {
      submit_command({.type = ApplicationCommandType::SetDisplayTransform, .unsigned_value = value});
    };
  }

  {
    ETX_PROFILER_NAMED_SCOPE("app_restore_recent_files");
    for (uint32_t i = 0; i < kRecentFileLimit; ++i) {
      const auto name = "recent-" + std::to_string(i);
      if (_options.has(name, Option::Class::String)) {
        const std::string restored_path = _options.get_string(name, {});
        if (restored_path.empty() == false) {
          add_to_recent(restored_path);
        }
      }
    }
  }

  {
    ETX_PROFILER_NAMED_SCOPE("app_set_initial_renderer_mode");
    set_renderer_mode(mode);
  }

#if defined(ETX_PLATFORM_WINDOWS)
  if (GetAsyncKeyState(VK_ESCAPE)) {
    _options.set_string("integrator", {}, "Integrator");
  }
#endif

  Integrator* integrator = nullptr;

  {
    ETX_PROFILER_NAMED_SCOPE("app_select_integrator");
    const std::string selected_integrator = _options.get_string("integrator", {});
    for (uint64_t i = 0; (selected_integrator.empty() == false) && (i < (uint64_t)cpu_renderer.integrator_count()); ++i) {
      Integrator* it = cpu_renderer.integrator_list()[i];
      ETX_ASSERT(it != nullptr);
      if (selected_integrator == it->name()) {
        integrator = it;
      }
    }

    if (integrator == nullptr && cpu_renderer.integrator_count() > 0) {
      integrator = cpu_renderer.integrator_list()[1];  // pt
    }
  }

  cpu_renderer.set_integrator(integrator);
  sync_scene_integrator_data_from_current_integrator();
  ui.set_current_integrator(integrator);

  if ((config.runtime_mode == RuntimeMode::Desktop) && config.persist_options) {
    ETX_PROFILER_NAMED_SCOPE("app_restore_last_scene");
    std::string restored_scene = _options.get_string("scene", {});
    if (restored_scene.empty() && !_recent_files.empty()) {
      restored_scene = _recent_files.back();
    }
    if ((restored_scene.empty() == false) && (load_scene_file(restored_scene, SceneRepresentation::LoadEverything | SceneRepresentation::PreferRecoveredSave, false) == false)) {
      _options.remove("scene");
    }
  }

  if ((config.runtime_mode == RuntimeMode::Desktop) && config.persist_options) {
    ETX_PROFILER_NAMED_SCOPE("app_restore_reference");
    const std::string ref = _options.get_string("ref", {});
    std::error_code error = {};
    if (!ref.empty() && std::filesystem::is_regular_file(ref, error)) {
      on_referenece_image_selected(ref);
    } else if (!ref.empty()) {
      _options.remove("ref");
    }
  }

  save_options();

  if (config.enable_platform_ui) {
    platform_ui().setup(ui);
    platform_ui().update(ui, _recent_files);
  }
  _initialized = true;
  publish_application_state();
}

void RTApplication::save_options() {
  ETX_PROFILER_SCOPE();

  if (!_application_config.persist_options) {
    return;
  }

  for (uint32_t idx = 0; idx < kRecentFileLimit; ++idx) {
    _options.remove("recent-" + std::to_string(idx));
  }
  uint32_t i = 0;
  for (const auto& recent : _recent_files) {
    _options.set_string("recent-" + std::to_string(i++), portable_scene_path(recent), "Recent File");
  }
  if ((_application_config.runtime_mode == RuntimeMode::Desktop) && !_current_scene_file.empty()) {
    _options.set_string("scene", portable_scene_path(_current_scene_file), "Scene");
  }
  _options.save_to_file(env().file_in_config("options.json"));
}

bool RTApplication::ensure_gpu_renderer_initialized() {
  ETX_PROFILER_SCOPE();

  if (_gpu_renderer_supported == false) {
    log::warning("GPU ray tracing is unavailable for the active RHI backend");
    return false;
  }

  if (_gpu_renderer_initialized) {
    return true;
  }

  gpu_renderer.init(render_context.get_context(), scene);
  if (gpu_renderer.runtime_failed()) {
    log::warning("GPU ray tracing initialization failed: %s", gpu_renderer.runtime_failure_reason().c_str());
    return false;
  }
  _gpu_renderer_initialized = true;
  return true;
}

void RTApplication::set_renderer_mode(RendererMode mode) {
  ETX_PROFILER_SCOPE();

  if ((mode == RendererMode::GPURaytracing) && (_gpu_renderer_supported == false)) {
    log::warning("GPU ray tracing is unavailable for the active RHI backend; using CPU ray tracing instead");
    mode = RendererMode::CPURaytracing;
  }

  Renderer* next_renderer = nullptr;
  std::string renderer_name;
  switch (mode) {
    case RendererMode::Rasterization:
      next_renderer = &raster_renderer;
      renderer_name = "raster";
      break;

    case RendererMode::GPURaytracing:
      next_renderer = &gpu_renderer;
      renderer_name = "gpu";
      break;

    default:
      next_renderer = &cpu_renderer;
      renderer_name = "cpu";
      break;
  }

  if ((next_renderer == &gpu_renderer) && !_current_scene_file.empty()) {
    if (ensure_gpu_renderer_initialized() == false) {
      log::warning("GPU ray tracing is unavailable for the active RHI backend; using CPU ray tracing instead");
      next_renderer = &cpu_renderer;
      renderer_name = "cpu";
      mode = RendererMode::CPURaytracing;
    }
  }

  _options.set_string("renderer", renderer_name, "Renderer");
  save_options();

  ui.set_current_renderer_mode(mode);

  if (next_renderer == _active_renderer) {
    sync_ui_renderer_state();
    return;
  }

  if (_scene_transform_interaction_active && (_scene_transform_interaction_renderer != next_renderer)) {
    on_scene_transform_interaction_finished();
  }

  if (_active_renderer != nullptr) {
    _active_renderer->stop();
  }

  if (next_renderer != &cpu_renderer) {
    cpu_renderer.film().release();
    cpu_renderer.integrator_thread().reset_scene_hashes();
  }

  _active_renderer = next_renderer;
  if ((_active_renderer == &cpu_renderer) && (_current_scene_file.empty() == false) && scene.valid()) {
    cpu_renderer.set_output_dimensions(render_context.get_context(), scene.camera().film_size);
  }

  if ((_active_renderer != nullptr) && !_current_scene_file.empty() && scene.valid()) {
    _active_renderer->start();
  }

  sync_ui_renderer_state();
}

bool RTApplication::set_render_configuration(RendererMode mode, Integrator::Type integrator_type) {
  if (mode == RendererMode::Rasterization) {
    set_renderer_mode(mode);
    return (_active_renderer != nullptr) && (_active_renderer->mode() == mode);
  }
  if ((mode != RendererMode::CPURaytracing) && (mode != RendererMode::GPURaytracing)) {
    return false;
  }
  if ((mode == RendererMode::GPURaytracing) && (_gpu_renderer_supported == false)) {
    return false;
  }
  if ((mode == RendererMode::GPURaytracing) && (UI::gpu_integrator_supported(integrator_type) == false)) {
    return false;
  }

  Integrator* const integrator = integrator_type_to_instance(integrator_type, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
  if ((integrator == nullptr) || (integrator->enabled() == false)) {
    return false;
  }

  Renderer* const target_renderer = (mode == RendererMode::GPURaytracing) ? static_cast<Renderer*>(&gpu_renderer) : static_cast<Renderer*>(&cpu_renderer);
  const bool renderer_changes = _active_renderer != target_renderer;
  const bool integrator_changes = cpu_renderer.current_integrator() != integrator;
  if (integrator_changes && (renderer_changes == false)) {
    on_integrator_selected(integrator_type);
  } else if (integrator_changes) {
    cpu_renderer.set_integrator(integrator);
    ui.set_current_integrator(integrator);
    sync_scene_integrator_data_from_current_integrator();
    _options.set_string("integrator", integrator->name(), "Integrator");
    notify_scene_might_have_changed();
  }

  if (renderer_changes) {
    set_renderer_mode(mode);
  } else {
    sync_ui_renderer_state();
  }
  ui.set_current_integrator(integrator);
  return (_active_renderer == target_renderer) && (cpu_renderer.current_integrator() == integrator);
}

void RTApplication::sync_platform_color_scheme() {
  if (!_application_config.enable_platform_ui || !render_context.valid() || !render_context.rhi_ui().initialized()) {
    return;
  }

  const PlatformColorScheme color_scheme = platform_ui().color_scheme();
  if (!_platform_color_scheme_initialized || (_platform_color_scheme != color_scheme)) {
    _platform_color_scheme_initialized = true;
    _platform_color_scheme = color_scheme;
    const RHIImGuiTheme theme = color_scheme == PlatformColorScheme::Dark ? RHIImGuiTheme::Dark : RHIImGuiTheme::Light;
    render_context.set_ui_theme(theme);
    ui.set_theme(theme);
  }
}

void RTApplication::frame() {
  ETX_PROFILER_SCOPE();

  if (!_initialization_started) {
    if (!_startup_frame_presented) {
      _startup_frame_presented = true;
      return;
    }
    init();
    if (_application_config.enable_platform_ui) {
      platform_ui().finish_startup(_initialized);
    }
  }

  if (!_initialized) {
    return;
  }

  sync_platform_color_scheme();
  process_application_commands();
  ui.set_view_options(_view_parameters);
  if (_gpu_renderer_initialized && (_active_renderer != &gpu_renderer)) {
    gpu_renderer.poll_preparation(render_context.get_context());
  }

  auto thread = _active_renderer && (_active_renderer->mode() == RendererMode::CPURaytracing) ? &cpu_renderer.integrator_thread() : nullptr;

  RenderContext::FrameData render_frame_data = {
    .dt = float(time_measure.lap()),
    .sample_count = thread ? thread->status().current_iteration : 0u,
    .view_parameters = _view_parameters,
  };

  UI::FrameData ui_frame_data = {
    .ior_database = _ior_database,
    .recent_files = _recent_files,
    .film = film,
    .output_size = _active_renderer ? _active_renderer->output_size() : uint2{},
    .dt = render_frame_data.dt,
    .scene_loaded = (_current_scene_file.empty() == false) && scene.valid(),
  };
  sync_ui_renderer_state();
  process_pending_image_requests();

  {
    ETX_PROFILER_NAMED_SCOPE("app_render_context_start_frame");
    Renderer* frame_renderer = _current_scene_file.empty() ? nullptr : _active_renderer;
    render_context.start_frame(frame_renderer, scene, render_frame_data);
  }
  sync_ui_renderer_state();
  if (_application_config.enable_platform_ui) {
    platform_ui().update(ui, _recent_files);
  }
  if (render_context.valid() && render_context.rhi_ui().initialized()) {
    ETX_PROFILER_NAMED_SCOPE("app_ui_build");
    ui.set_scene_dirty(_scene_dirty);
    ui.build(scene, ui_frame_data);
    const UI::ViewportGeometry& geometry = ui.viewport_geometry();
    RenderContext::PresentationViewport viewport = {};
    if (geometry.valid) {
      const int32_t left = static_cast<int32_t>(std::lround(geometry.logical_position.x * geometry.framebuffer_scale.x));
      const int32_t top = static_cast<int32_t>(std::lround(geometry.logical_position.y * geometry.framebuffer_scale.y));
      const int32_t right = static_cast<int32_t>(std::lround((geometry.logical_position.x + geometry.logical_size.x) * geometry.framebuffer_scale.x));
      const int32_t bottom = static_cast<int32_t>(std::lround((geometry.logical_position.y + geometry.logical_size.y) * geometry.framebuffer_scale.y));
      const uint32_t display_width = static_cast<uint32_t>(std::max(1l, std::lround(geometry.image_size.x * geometry.framebuffer_scale.x)));
      const uint32_t display_height = static_cast<uint32_t>(std::max(1l, std::lround(geometry.image_size.y * geometry.framebuffer_scale.y)));
      viewport = {
        .x = left,
        .y = top,
        .width = static_cast<uint32_t>(std::max(0, right - left)),
        .height = static_cast<uint32_t>(std::max(0, bottom - top)),
        .display_width = display_width,
        .display_height = display_height,
        .valid = (right > left) && (bottom > top),
      };
    }
    render_context.set_presentation_viewport(viewport);
  }
  poll_material_render_resource_preparation();
  {
    ETX_PROFILER_NAMED_SCOPE("app_render_context_end_frame");
    render_context.end_frame();
  }
  publish_application_state();
}

void RTApplication::cleanup() {
  ETX_PROFILER_SCOPE();

  if (_initialized.exchange(false)) {
    save_options();
  }

  if (_application_config.enable_platform_ui) {
    platform_ui().shutdown();
  }

  bool device_already_idle = false;

  if ((_active_renderer != nullptr) && !_current_scene_file.empty()) {
    _active_renderer->stop();
  }

  if (render_context.valid()) {
    auto& ctx = render_context.get_context();
    cpu_renderer.cleanup(ctx);
    raster_renderer.cleanup(ctx);

    if (_gpu_renderer_initialized) {
      gpu_renderer.cleanup(ctx);
      device_already_idle = gpu_renderer.cleanup_wait_succeeded();
    }

    scene.cancel_energy_compensation_interface_preparation();
  }

  scheduler.shutdown();
  ShaderCompiler::instance().shutdown();
  if (_scene_global_initialized) {
    scene_global_deinit();
    _scene_global_initialized = false;
  }

  render_context.cleanup(device_already_idle);
}

void RTApplication::process_event(const sapp_event* e) {
  ETX_PROFILER_SCOPE();

  if (!_initialized) {
    return;
  }

  if ((e != nullptr) && (e->type == SAPP_EVENTTYPE_QUIT_REQUESTED) && _scene_dirty) {
    sapp_cancel_quit();
    ui.request_quit_confirmation();
    return;
  }

  if ((e != nullptr) && (e->type == SAPP_EVENTTYPE_QUIT_REQUESTED) && (_quit_preparation_cancel_requested == false)) {
    if (_gpu_renderer_initialized) {
      gpu_renderer.cancel_preparation();
    }
    _quit_preparation_cancel_requested = true;
  }

  {
    ETX_PROFILER_NAMED_SCOPE("app_process_event_imgui");
    if (render_context.imgui_enabled() && render_context.rhi_ui().handle_event(e)) {
      return;
    }
  }

  if (_application_config.enable_imgui || _application_config.enable_platform_ui) {
    ETX_PROFILER_NAMED_SCOPE("app_process_event_ui");
    if (ui.handle_event(e)) {
      return;
    }
  }

  if ((_active_renderer != nullptr) && !_current_scene_file.empty()) {
    ETX_PROFILER_NAMED_SCOPE("app_process_event_renderer");
    const bool pointer_camera_input = (e->type == SAPP_EVENTTYPE_MOUSE_DOWN) || (e->type == SAPP_EVENTTYPE_MOUSE_SCROLL);
    const bool keyboard_camera_input =
      (e->type == SAPP_EVENTTYPE_KEY_DOWN) && ((e->key_code == SAPP_KEYCODE_W) || (e->key_code == SAPP_KEYCODE_A) || (e->key_code == SAPP_KEYCODE_S) ||
                                                (e->key_code == SAPP_KEYCODE_D) || (e->key_code == SAPP_KEYCODE_Q) || (e->key_code == SAPP_KEYCODE_E));
    if (pointer_camera_input || keyboard_camera_input) {
      mark_scene_dirty();
    }
    _active_renderer->process_event(e);
  }
}

void RTApplication::add_to_recent(const std::string& value) {
  ETX_PROFILER_SCOPE();

  const std::string absolute_path = normalized_existing_scene_path(value);
  if (absolute_path.empty()) {
    return;
  }

  auto e = std::remove_if(_recent_files.begin(), _recent_files.end(), [&](const std::string& entry) {
    const std::string normalized_entry = normalized_existing_scene_path(entry);
    return (normalized_entry.empty() || (normalized_entry == absolute_path));
  });
  _recent_files.erase(e, _recent_files.end());

  _recent_files.emplace_back(absolute_path);

  if (_recent_files.size() > kRecentFileLimit) {
    _recent_files.erase(_recent_files.begin());
  }
}

bool RTApplication::load_scene_file(const std::string& file_name, uint32_t options, bool start_rendering) {
  ETX_PROFILER_SCOPE();

  const std::string scene_file = normalized_existing_scene_path(file_name);
  if (scene_file.empty()) {
    log::error("Scene file does not exist: %s", file_name.c_str());
    auto e = std::remove_if(_recent_files.begin(), _recent_files.end(), [&](const std::string& entry) {
      return env().resolve_to_absolute(entry) == env().resolve_to_absolute(file_name);
    });
    _recent_files.erase(e, _recent_files.end());
    save_options();
    return false;
  }

  log::warning("Loading scene %s...", scene_file.c_str());
  SceneRepresentation::IntegratorData integrator_data;
  SceneRepresentation loaded_scene(scheduler, _ior_database);
  loaded_scene.set_scattering_rhi(render_context.get_context());
  {
    ETX_PROFILER_NAMED_SCOPE("app_scene_load_from_file");
    if ((loaded_scene.load_from_file(scene_file.c_str(), options, &integrator_data) == false) || (loaded_scene.valid() == false)) {
      log::error("Failed to load scene from file: %s", scene_file.c_str());
      return false;
    }
  }

  if (_scene_transform_interaction_active) {
    on_scene_transform_interaction_finished();
  }
  if (_active_renderer != nullptr) {
    _active_renderer->stop();
  }
  scene.replace_loaded_scene(loaded_scene);
  if ((_active_renderer != nullptr) && (_active_renderer->camera_controller() != nullptr)) {
    _active_renderer->camera_controller()->sync_from_camera();
  }
  ui.reset_scene_state();
  _material_interaction_active = false;
  _material_interaction_cpu_was_running = false;
  _medium_interaction_active = false;
  _medium_interaction_cpu_was_active = false;
  _emitter_interaction_active = false;
  _emitter_interaction_cpu_was_running = false;
  _material_render_resource_preparation_active = false;
  _restart_cpu_after_material_resource_preparation = false;
  _restart_gpu_after_material_resource_preparation = false;
  _current_scene_file = scene.data().json_file_name.empty() ? scene_file : env().resolve_to_absolute(scene.data().json_file_name);
  log::warning("Setting output dimensions...");
  {
    ETX_PROFILER_NAMED_SCOPE("app_scene_set_output_dimensions");
    cpu_renderer.set_output_dimensions(render_context.get_context(), scene.camera().film_size);
  }

  if ((_active_renderer == &gpu_renderer) && !ensure_gpu_renderer_initialized()) {
    set_renderer_mode(RendererMode::CPURaytracing);
  }

  if (_application_config.persist_options == false) {
    cpu_renderer.integrator_thread().suppress_next_scene_commit_run();
  }
  notify_scene_might_have_changed();

  {
    ETX_PROFILER_NAMED_SCOPE("app_scene_sync_integrator_settings");
    for (const auto& [type, options] : integrator_data.settings) {
      Integrator* integrator = integrator_type_to_instance(type, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
      if (integrator != nullptr) {
        integrator->sync_from_options(options);
        integrator->update_options();
      }
    }
  }

  Integrator* integrator = nullptr;
  {
    ETX_PROFILER_NAMED_SCOPE("app_scene_select_integrator");
    if (integrator_data.selected != Integrator::Type::Invalid) {
      integrator = integrator_type_to_instance(integrator_data.selected, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
    }

    if (integrator == nullptr && cpu_renderer.integrator_count() > 0) {
      integrator = cpu_renderer.integrator_list()[1];  // pt
    }
  }

  cpu_renderer.set_integrator(integrator);
  sync_scene_integrator_data_from_current_integrator();
  ui.set_current_integrator(integrator);
  notify_scene_might_have_changed();

  add_to_recent(_current_scene_file);
  _scene_dirty = false;
  save_options();

  if (start_rendering && (_active_renderer != nullptr)) {
    ETX_PROFILER_NAMED_SCOPE("app_scene_start_renderer");
    _active_renderer->start();
  }
  return true;
}

std::string RTApplication::save_scene_file(const std::string& file_name) {
  ETX_PROFILER_SCOPE();

  log::info("Saving %s..", file_name.c_str());
  Integrator* current = cpu_renderer.current_integrator();
  Integrator::Type selected_type = integrator_to_type(current);
  std::string saved_path = scene.save_to_file(file_name.c_str(), selected_type, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
  if (saved_path.empty()) {
    log::error("Failed to save scene to %s", file_name.c_str());
    return {};
  }

  _current_scene_file = env().resolve_to_absolute(saved_path);
  add_to_recent(_current_scene_file);
  _scene_dirty = false;
  save_options();

  return _current_scene_file;
}

void RTApplication::mark_scene_dirty() {
  if ((_current_scene_file.empty() == false) && scene.valid()) {
    _scene_dirty = true;
  }
}

void RTApplication::on_referenece_image_selected(std::string file_name) {
  ETX_PROFILER_SCOPE();

  _options.set_string("ref", file_name, "Reference");
  save_options();

  _pending_reference_file = file_name;
  _pending_reference_file_load = true;
}

bool RTApplication::read_active_renderer_output(std::vector<float4>& output, uint2& image_size) {
  output.clear();
  image_size = {};

  if (render_context.valid() == false) {
    log::warning("Cannot capture renderer output: render context is not initialized");
    return false;
  }

  if (_active_renderer == nullptr) {
    log::warning("Cannot capture renderer output: no renderer is active");
    return false;
  }

  const RHITexture texture = _active_renderer->output_texture();
  image_size = _active_renderer->output_size();
  return read_texture_to_float4_buffer(render_context.get_context(), texture, image_size, output);
}

void RTApplication::process_pending_image_requests() {
  if ((_pending_reference_file_load == false) && (_pending_current_image_reference_capture == false) && (_pending_gpu_save_image == false)) {
    return;
  }

  if (_pending_reference_file_load) {
    log::warning("Loading reference image %s...", _pending_reference_file.c_str());
    render_context.set_reference_image(_pending_reference_file.c_str());
    _pending_reference_file.clear();
    _pending_reference_file_load = false;
  }

  std::vector<float4> output = {};
  uint2 image_size = {};
  bool capture_succeeded = false;
  bool capture_attempted = false;
  const bool gpu_renderer_active = (_active_renderer != nullptr) && (_active_renderer->mode() == RendererMode::GPURaytracing);
  if (_pending_current_image_reference_capture && (_pending_reference_capture_renderer != _active_renderer)) {
    log::warning("Reference capture canceled because the active renderer changed");
    _pending_current_image_reference_capture = false;
    _pending_reference_capture_renderer = nullptr;
  }

  const bool reference_output_available = _pending_current_image_reference_capture && (_active_renderer != nullptr) && _active_renderer->output_texture().valid();
  const bool gpu_save_output_available = gpu_renderer_active && _pending_gpu_save_image && _active_renderer->output_texture().valid();
  const bool needs_renderer_capture = reference_output_available || gpu_save_output_available;
  if (needs_renderer_capture) {
    capture_attempted = true;
    capture_succeeded = read_active_renderer_output(output, image_size);
  }
  const bool reference_capture_attempted = reference_output_available && capture_attempted;

  if (_pending_current_image_reference_capture) {
    if (reference_capture_attempted) {
      if (capture_succeeded) {
        render_context.set_reference_image(output.data(), image_size);
        _options.set_string("ref", {}, "Reference");
        save_options();
      } else {
        log::warning("Failed to use the current renderer output as a reference image");
      }
      _pending_current_image_reference_capture = false;
      _pending_reference_capture_renderer = nullptr;
    }
  }

  if (_pending_gpu_save_image) {
    if (gpu_renderer_active == false) {
      log::warning("GPU image save canceled because the active renderer changed");
      _pending_gpu_save_image_file.clear();
      _pending_gpu_save_image = false;
    } else if (gpu_save_output_available) {
      if (capture_attempted == false) {
        capture_succeeded = read_active_renderer_output(output, image_size);
      }
      if (capture_succeeded && (_pending_gpu_save_image_file.empty() == false)) {
        ImageOutputParameters params = {
          .mode = _pending_gpu_save_image_mode,
          .exposure = _view_parameters.exposure,
        };
        save_image_to_file(_pending_gpu_save_image_file, output.data(), image_size, params);
      }
      _pending_gpu_save_image_file.clear();
      _pending_gpu_save_image = false;
    }
  }
}

void RTApplication::on_use_image_as_reference() {
  ETX_PROFILER_SCOPE();

  if ((_active_renderer == nullptr) || ((_active_renderer->mode() != RendererMode::CPURaytracing) && (_active_renderer->mode() != RendererMode::GPURaytracing))) {
    log::warning("A CPU or GPU ray-tracing output is required for reference capture");
    return;
  }

  _pending_current_image_reference_capture = true;
  _pending_reference_capture_renderer = _active_renderer;
  if (_active_renderer->output_texture().valid() == false) {
    log::info("Reference capture is waiting for the active renderer to complete an output image");
  }
}

void RTApplication::on_save_image_selected(std::string file_name, SaveImageMode mode) {
  ETX_PROFILER_SCOPE();

  if ((_active_renderer != nullptr) && (_active_renderer->mode() == RendererMode::GPURaytracing)) {
    _pending_gpu_save_image_file = file_name;
    _pending_gpu_save_image_mode = mode;
    _pending_gpu_save_image = true;
    return;
  }

  uint2 image_size = {scene.camera().film_size.x, scene.camera().film_size.y};
  const float4* output = cpu_renderer.film().layer(_view_parameters.view_layer, cpu_renderer.scene().options.radiance_clamp);
  ImageOutputParameters params = {
    .mode = mode,
    .exposure = _view_parameters.exposure,
  };
  save_image_to_file(file_name, output, image_size, params);
}

void RTApplication::on_integrator_selected(Integrator::Type itype) {
  ETX_PROFILER_SCOPE();

  Integrator* i = integrator_type_to_instance(itype, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
  if (i == nullptr) {
    return;
  }

  cpu_renderer.set_integrator(i);
  ui.set_current_integrator(i);
  if (_active_renderer == &gpu_renderer) {
    gpu_renderer.invalidate_output();
  }
  sync_scene_integrator_data_from_current_integrator();
  _options.set_string("integrator", i->name(), "Integrator");
  save_options();

  notify_scene_might_have_changed();

  if ((_active_renderer == &cpu_renderer) && !_current_scene_file.empty() && scene.valid()) {
    cpu_renderer.film().clear(Film::ClearEverything);
    cpu_renderer.start();
  }
}

void RTApplication::on_run_selected() {
  ETX_PROFILER_SCOPE();

  if (_view_parameters.view_layer == ViewLayer::Denoised) {
    _view_parameters.view_layer = ViewLayer::Result;
  }
  if (_active_renderer != nullptr) {
    _active_renderer->start();
  }
}

void RTApplication::on_stop_selected(bool wait_for_completion) {
  ETX_PROFILER_SCOPE();
  if (_active_renderer == nullptr) {
    return;
  }

  if (wait_for_completion) {
    _active_renderer->finish();
  } else {
    _active_renderer->stop();
  }
}

void RTApplication::on_restart_selected() {
  ETX_PROFILER_SCOPE();
  if (_view_parameters.view_layer == ViewLayer::Denoised) {
    _view_parameters.view_layer = ViewLayer::Result;
  }
  if (_active_renderer != nullptr) {
    _active_renderer->restart();
  }
}

void RTApplication::on_options_changed() {
  ETX_PROFILER_SCOPE();
  mark_scene_dirty();
  const bool cpu_renderer_active = _active_renderer == &cpu_renderer;
  if (cpu_renderer_active) {
    cpu_renderer.stop();
  }

  sync_scene_integrator_data_from_current_integrator();
  notify_scene_might_have_changed();
  if (cpu_renderer_active) {
    cpu_renderer.restart();
  }
}

SceneResourceEditResult RTApplication::on_material_added() {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneResourceEditResult result = scene.create_material(nullptr);
  if (result.succeeded()) {
    mark_scene_dirty();
    notify_scene_might_have_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneResourceEditResult RTApplication::on_material_duplicated(uint32_t index) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneResourceEditResult result = scene.duplicate_material(index);
  if (result.succeeded()) {
    _restart_cpu_after_material_resource_preparation = _restart_cpu_after_material_resource_preparation || cpu_was_running;
    on_material_changed(result.resource_index);
  } else if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneResourceEditResult RTApplication::on_material_deleted(uint32_t index) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneResourceEditResult result = scene.delete_material(index);
  if (result.succeeded()) {
    mark_scene_dirty();
    notify_scene_might_have_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

std::string RTApplication::on_material_renamed(uint32_t index, const std::string& name) {
  std::string current_name;
  for (const auto& [resource_name, resource_index] : scene.material_mapping()) {
    if (resource_index == index) {
      current_name = resource_name;
      break;
    }
  }
  const std::string renamed = scene.rename_material(index, name.c_str());
  if ((renamed.empty() == false) && (renamed != current_name)) {
    const bool cpu_was_running = cpu_renderer.is_running();
    if (cpu_was_running) {
      cpu_renderer.stop();
    }
    mark_scene_dirty();
    scene.create_area_emitters_from_materials();
    notify_scene_might_have_changed();
    if (cpu_was_running) {
      cpu_renderer.restart();
    }
  }
  return renamed;
}

void RTApplication::on_material_changed(uint32_t index) {
  (void)index;
  mark_scene_dirty();
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  scene.create_area_emitters_from_materials();
  _restart_cpu_after_material_resource_preparation = _restart_cpu_after_material_resource_preparation || cpu_was_running;
  if (scene.begin_energy_compensation_interface_preparation()) {
    _material_render_resource_preparation_active = true;
    return;
  }

  _material_render_resource_preparation_active = false;
  if (rebuild_material_render_resources() == false) {
    finish_material_render_resource_preparation(false);
    return;
  }
  finish_material_render_resource_preparation(true);
}

void RTApplication::on_material_interaction_started() {
  if (_material_interaction_active) {
    return;
  }

  _material_interaction_active = true;
  _material_interaction_cpu_was_running = cpu_renderer.is_running();
  if (_material_interaction_cpu_was_running) {
    cpu_renderer.stop();
  }
}

void RTApplication::on_material_interaction_finished(const std::vector<uint32_t>& material_indices) {
  if (_material_interaction_active == false) {
    return;
  }

  _material_interaction_active = false;
  const bool cpu_was_running = _material_interaction_cpu_was_running;
  _material_interaction_cpu_was_running = false;
  if (material_indices.empty()) {
    _restart_cpu_after_material_resource_preparation = _restart_cpu_after_material_resource_preparation || cpu_was_running;
    if (_restart_cpu_after_material_resource_preparation && (_material_render_resource_preparation_active == false)) {
      cpu_renderer.restart();
      _restart_cpu_after_material_resource_preparation = false;
    }
    return;
  }

  _restart_cpu_after_material_resource_preparation = _restart_cpu_after_material_resource_preparation || cpu_was_running;
  on_material_changed(material_indices.front());
}

SceneResourceEditResult RTApplication::on_medium_added() {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneResourceEditResult result = scene.create_medium(nullptr);
  if (result.succeeded()) {
    mark_scene_dirty();
    scene.update_medium_bounds();
    notify_scene_might_have_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneResourceEditResult RTApplication::on_medium_duplicated(uint32_t index) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneResourceEditResult result = scene.duplicate_medium(index);
  if (result.succeeded()) {
    mark_scene_dirty();
    scene.update_medium_bounds();
    notify_scene_might_have_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneResourceEditResult RTApplication::on_medium_deleted(uint32_t index) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneResourceEditResult result = scene.delete_medium(index);
  if (result.succeeded()) {
    mark_scene_dirty();
    scene.update_medium_bounds();
    notify_scene_might_have_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

std::string RTApplication::on_medium_renamed(uint32_t index, const std::string& name) {
  std::string current_name;
  for (const auto& [resource_name, resource_index] : scene.medium_mapping()) {
    if (resource_index == index) {
      current_name = resource_name;
      break;
    }
  }
  const std::string renamed = scene.rename_medium(index, name.c_str());
  if ((renamed.empty() == false) && (renamed != current_name)) {
    mark_scene_dirty();
  }
  return renamed;
}

void RTApplication::on_medium_changed(uint32_t index) {
  (void)index;
  mark_scene_dirty();
  scene.update_medium_bounds();
  notify_scene_might_have_changed();
}

void RTApplication::on_medium_interaction_started() {
  if (_medium_interaction_active) {
    return;
  }

  _medium_interaction_active = true;
  _medium_interaction_cpu_was_active = cpu_renderer.control_state().can_stop;
  if (_medium_interaction_cpu_was_active) {
    cpu_renderer.stop();
  }
}

void RTApplication::on_medium_interaction_finished(const std::vector<uint32_t>& medium_indices) {
  if (_medium_interaction_active == false) {
    return;
  }

  _medium_interaction_active = false;
  const bool cpu_was_active = _medium_interaction_cpu_was_active;
  _medium_interaction_cpu_was_active = false;
  if (medium_indices.empty() == false) {
    on_medium_changed(medium_indices.front());
  }
  if (cpu_was_active) {
    cpu_renderer.restart();
  }
}

void RTApplication::on_mesh_material_changed(uint32_t mesh_index, uint32_t material_index) {
  mark_scene_dirty();
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  scene.set_mesh_material(mesh_index, material_index);
  scene.create_area_emitters_from_materials();
  notify_scene_might_have_changed();

  if (cpu_was_running) {
    cpu_renderer.restart();
  }
}

uint32_t RTApplication::on_make_mesh_material_unique(uint32_t mesh_index, uint32_t material_index) {
  SceneData& scene_data = scene.data();
  if ((mesh_index >= scene_data.meshes.size()) || (material_index >= scene_data.materials.size())) {
    return kInvalidIndex;
  }

  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
    _restart_cpu_after_material_resource_preparation = true;
  }

  Material material = scene_data.materials[material_index];
  const auto clone_spectrum = [&](uint32_t spectrum_index) {
    return spectrum_index < scene_data.spectrum_values.size() ? scene_data.add_spectrum(scene_data.spectrum_values[spectrum_index]) : kInvalidIndex;
  };
  material.reflectance.spectrum_index = clone_spectrum(material.reflectance.spectrum_index);
  material.scattering.spectrum_index = clone_spectrum(material.scattering.spectrum_index);
  material.emission.spectrum_index = clone_spectrum(material.emission.spectrum_index);
  material.subsurface.spectrum_index = clone_spectrum(material.subsurface.spectrum_index);
  material.thinfilm.ior.eta_index = clone_spectrum(material.thinfilm.ior.eta_index);
  material.thinfilm.ior.k_index = clone_spectrum(material.thinfilm.ior.k_index);
  material.ext_ior.eta_index = clone_spectrum(material.ext_ior.eta_index);
  material.ext_ior.k_index = clone_spectrum(material.ext_ior.k_index);
  material.int_ior.eta_index = clone_spectrum(material.int_ior.eta_index);
  material.int_ior.k_index = clone_spectrum(material.int_ior.k_index);
  material.energy_compensation_interface_index = kInvalidIndex;
  material.conductor_energy_compensation_interface_index = kInvalidIndex;

  std::string source_name = "material";
  for (const auto& [name, index] : scene_data.material_mapping) {
    if (index == material_index) {
      source_name = name;
      break;
    }
  }
  const std::string clone_name = source_name + " copy";
  const uint32_t clone_index = scene_data.clone_material(material, clone_name.c_str());
  scene.set_mesh_material(mesh_index, clone_index);
  on_material_changed(clone_index);
  return clone_index;
}

void RTApplication::on_emitter_changed(uint32_t index) {
  mark_scene_dirty();
  const bool cpu_was_running = cpu_renderer.is_running();
  bool atmosphere_related = false;
  bool rebuild_all_atmospheres = false;
  uint32_t atmosphere_emitter_index = kInvalidIndex;
  if (index < scene.data().emitter_profiles.size()) {
    const auto& emitter = scene.data().emitter_profiles[index];
    if ((emitter.cls == EmitterProfile::Class::Environment) && ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0u)) {
      atmosphere_related = true;
      atmosphere_emitter_index = index;
    } else if (emitter.cls == EmitterProfile::Class::Directional) {
      atmosphere_related = true;
      rebuild_all_atmospheres = true;
    }

    if (atmosphere_related) {
      if (cpu_was_running) {
        cpu_renderer.stop();
      }
      if (rebuild_all_atmospheres) {
        rebuild_all_atmosphere_emitters();
      } else {
        scene.rebuild_atmosphere_emitter(atmosphere_emitter_index);
      }
    }
  }

  notify_scene_might_have_changed();
  if (atmosphere_related && cpu_was_running) {
    cpu_renderer.restart();
  }
}

void RTApplication::on_emitter_interaction_started() {
  if (_emitter_interaction_active) {
    return;
  }

  _emitter_interaction_active = true;
  _emitter_interaction_cpu_was_running = cpu_renderer.is_running();
  if (_emitter_interaction_cpu_was_running) {
    cpu_renderer.stop();
  }
}

void RTApplication::on_emitter_interaction_finished(uint32_t index) {
  if (_emitter_interaction_active == false) {
    return;
  }

  _emitter_interaction_active = false;
  const bool cpu_was_running = _emitter_interaction_cpu_was_running;
  _emitter_interaction_cpu_was_running = false;
  if (index != kInvalidIndex) {
    on_emitter_changed(index);
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
}

SceneResourceEditResult RTApplication::on_emitter_added(uint32_t type) {
  ETX_PROFILER_SCOPE();
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  SceneResourceEditResult result = {.status = SceneResourceEditStatus::InvalidResource};
  switch (type) {
    case 0: {
      result = {.resource_index = scene.add_environment_emitter({1.0f, 1.0f, 1.0f}, kInvalidIndex)};
      break;
    }
    case 1: {
      result = {.resource_index = scene.add_directional_emitter({0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, 0.5422f, kInvalidIndex)};
      break;
    }
    case 2: {
      const uint32_t emitter_index = static_cast<uint32_t>(scene.data().emitter_profiles.size());
      scene.add_atmosphere_emitter({
        .scattering = {.altitude = 1000.0f, .anisotropy = 0.825f, .rayleigh_scale = 1.0f, .mie_scale = 1.0f, .ozone_scale = 1.0f},
        .quality = 0.125f,
      });
      result = (scene.data().emitter_profiles.size() > emitter_index) ? SceneResourceEditResult{.resource_index = emitter_index}
                                                                      : SceneResourceEditResult{.status = SceneResourceEditStatus::ResourceUpdateFailed};
      break;
    }
    default:
      break;
  }

  if (result.succeeded()) {
    mark_scene_dirty();
    notify_scene_might_have_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneResourceEditResult RTApplication::on_emitter_duplicated(uint32_t index) {
  ETX_PROFILER_SCOPE();
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const bool directional_emitter = (index < scene.data().emitter_profiles.size()) && (scene.data().emitter_profiles[index].cls == EmitterProfile::Class::Directional);
  const bool atmosphere_emitter = (index < scene.data().emitter_profiles.size()) && (scene.data().emitter_profiles[index].cls == EmitterProfile::Class::Environment) &&
                                  ((scene.data().emitter_profiles[index].meta & EmitterProfile::Meta::Atmosphere) != 0u);
  const SceneResourceEditResult result = scene.duplicate_emitter(index);
  if (result.succeeded()) {
    if (directional_emitter) {
      rebuild_all_atmosphere_emitters();
    } else if (atmosphere_emitter) {
      scene.rebuild_atmosphere_emitter(result.resource_index);
    }
    mark_scene_dirty();
    notify_scene_might_have_changed();
  }

  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneResourceEditResult RTApplication::on_emitter_deleted(uint32_t index) {
  ETX_PROFILER_SCOPE();
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const bool directional_emitter = (index < scene.data().emitter_profiles.size()) && (scene.data().emitter_profiles[index].cls == EmitterProfile::Class::Directional);
  const SceneResourceEditResult result = scene.delete_emitter_profile(index);
  if (result.succeeded()) {
    if (directional_emitter) {
      rebuild_all_atmosphere_emitters();
    }
    mark_scene_dirty();
    notify_scene_might_have_changed();
  }

  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

std::string RTApplication::on_emitter_renamed(uint32_t index, const std::string& name) {
  const std::vector<std::string>& emitter_names = scene.emitter_names();
  const std::string current_name = index < emitter_names.size() ? emitter_names[index] : std::string();
  const std::string renamed = scene.rename_emitter(index, name.c_str());
  if ((renamed.empty() == false) && (renamed != current_name)) {
    mark_scene_dirty();
  }
  return renamed;
}

SceneResourceEditResult RTApplication::on_camera_added() {
  const SceneResourceEditResult result = scene.create_camera(nullptr);
  if (result.succeeded()) {
    mark_scene_dirty();
    notify_scene_might_have_changed();
  }
  return result;
}

SceneResourceEditResult RTApplication::on_camera_duplicated(uint32_t index) {
  const SceneResourceEditResult result = scene.duplicate_camera(index);
  if (result.succeeded()) {
    mark_scene_dirty();
    notify_scene_might_have_changed();
  }
  return result;
}

SceneResourceEditResult RTApplication::on_camera_deleted(uint32_t index) {
  const SceneResourceEditResult result = scene.delete_camera(index);
  if (result.succeeded()) {
    mark_scene_dirty();
    notify_scene_might_have_changed();
  }
  return result;
}

std::string RTApplication::on_camera_renamed(uint32_t index, const std::string& name) {
  const std::string current_name = index < scene.data().cameras.size() ? scene.data().cameras[index].id : std::string();
  const std::string renamed = scene.rename_camera(index, name.c_str());
  if ((renamed.empty() == false) && (renamed != current_name)) {
    mark_scene_dirty();
  }
  return renamed;
}

SceneEditResult RTApplication::on_empty_node_added() {
  const SceneEditResult result = scene.create_empty_node();
  if (result.succeeded()) {
    handle_scene_hierarchy_changed();
  }
  return result;
}

SceneEditResult RTApplication::on_primitive_added(ScenePrimitive primitive) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneEditResult result = scene.create_primitive(primitive);
  if (result.succeeded()) {
    scene.create_area_emitters_from_materials();
    handle_scene_hierarchy_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneEditResult RTApplication::on_node_duplicated(uint32_t node_index) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneEditResult result = scene.duplicate_node_subtree(node_index);
  if (result.succeeded()) {
    handle_scene_hierarchy_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneEditResult RTApplication::on_node_deleted(uint32_t node_index) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneEditResult result = scene.delete_node_subtree(node_index);
  if (result.succeeded()) {
    handle_scene_hierarchy_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneEditResult RTApplication::on_node_reparented(uint32_t node_index, uint32_t parent_index) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneEditResult result = scene.reparent_node(node_index, parent_index);
  if (result.succeeded()) {
    handle_scene_hierarchy_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneEditResult RTApplication::on_node_enabled_changed(uint32_t node_index, bool enabled) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneEditResult result = scene.set_node_enabled(node_index, enabled);
  if (result.succeeded()) {
    handle_scene_hierarchy_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneEditResult RTApplication::on_node_transform_changed(uint32_t node_index, const AffineTransform& transform) {
  const SceneEditResult result = scene.set_node_local_transform(node_index, transform);
  if (result.succeeded()) {
    on_scene_transforms_changed();
  }
  return result;
}

NodeGeometryEditResult RTApplication::on_node_geometry_edited(uint32_t node_index, NodeGeometryOperation operation) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const NodeGeometryEditResult result = scene.edit_node_geometry(node_index, operation);
  if (result == NodeGeometryEditResult::Success) {
    scene.update_medium_bounds();
    scene.update_active_camera();
    mark_scene_dirty();
    notify_scene_might_have_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneEditResult RTApplication::on_node_resource_attached(uint32_t node_index, SceneAttachment::Type type, uint32_t resource_index) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneEditResult result = scene.attach_node_resource(node_index, type, resource_index);
  if (result.succeeded()) {
    handle_scene_hierarchy_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

SceneEditResult RTApplication::on_node_resource_detached(uint32_t node_index, uint32_t local_attachment_index) {
  const bool cpu_was_running = cpu_renderer.is_running();
  if (cpu_was_running) {
    cpu_renderer.stop();
  }

  const SceneEditResult result = scene.detach_node_resource(node_index, local_attachment_index);
  if (result.succeeded()) {
    handle_scene_hierarchy_changed();
  }
  if (cpu_was_running) {
    cpu_renderer.restart();
  }
  return result;
}

std::string RTApplication::on_node_renamed(uint32_t node_index, const std::string& name) {
  const std::string current_name = node_index < scene.data().hierarchy.node_names.size() ? scene.data().hierarchy.node_names[node_index] : std::string();
  const std::string renamed = scene.rename_node(node_index, name.c_str());
  if ((renamed.empty() == false) && (renamed != current_name)) {
    mark_scene_dirty();
  }
  return renamed;
}

void RTApplication::on_camera_changed(uint2 viewport, uint32_t pixel_size) {
  ETX_PROFILER_SCOPE();
  mark_scene_dirty();

  scene.update_active_camera();
  if ((_active_renderer != nullptr) && (_active_renderer->camera_controller() != nullptr)) {
    _active_renderer->camera_controller()->sync_from_camera();
  }
  if ((_active_renderer == &cpu_renderer) && ((viewport != film.base_dimensions()) || (pixel_size != film.pixel_size()))) {
    cpu_renderer.set_output_dimensions(render_context.get_context(), scene.camera().film_size);
  }
  notify_scene_might_have_changed();
  if (_active_renderer == &cpu_renderer) {
    cpu_renderer.restart();
  }
}

void RTApplication::on_scene_settings_changed() {
  mark_scene_dirty();
  if ((_active_renderer != nullptr) && (_active_renderer->camera_controller() != nullptr)) {
    _active_renderer->camera_controller()->sync_from_camera();
  }

  if (energy_compensation_cache_matches_render_mode(scene.data()) == false) {
    const bool cpu_was_running = (_active_renderer == &cpu_renderer) && cpu_renderer.is_running();
    const bool gpu_was_running = (_active_renderer == &gpu_renderer) && gpu_renderer.is_running();
    if (cpu_was_running) {
      cpu_renderer.stop();
    }
    if (gpu_was_running) {
      gpu_renderer.stop();
    }
    _restart_cpu_after_material_resource_preparation = _restart_cpu_after_material_resource_preparation || cpu_was_running;
    _restart_gpu_after_material_resource_preparation = _restart_gpu_after_material_resource_preparation || gpu_was_running;
    if (scene.begin_energy_compensation_interface_preparation()) {
      _material_render_resource_preparation_active = true;
      return;
    }

    _material_render_resource_preparation_active = false;
    if (rebuild_material_render_resources() == false) {
      finish_material_render_resource_preparation(false);
      return;
    }
    finish_material_render_resource_preparation(true);
    return;
  }

  notify_scene_might_have_changed();
}

void RTApplication::on_scene_transforms_changed() {
  mark_scene_dirty();
  if ((_active_renderer != nullptr) && (_active_renderer->camera_controller() != nullptr)) {
    _active_renderer->camera_controller()->sync_from_camera();
  }
  notify_scene_transforms_changed();
}

void RTApplication::handle_scene_hierarchy_changed() {
  mark_scene_dirty();
  if ((_active_renderer != nullptr) && (_active_renderer->camera_controller() != nullptr)) {
    _active_renderer->camera_controller()->sync_from_camera();
  }
  notify_scene_might_have_changed();
}

void RTApplication::rebuild_all_atmosphere_emitters() {
  for (uint32_t emitter_index = 0u; emitter_index < scene.data().emitter_profiles.size(); ++emitter_index) {
    const EmitterProfile& emitter = scene.data().emitter_profiles[emitter_index];
    if ((emitter.cls == EmitterProfile::Class::Environment) && ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0u)) {
      scene.rebuild_atmosphere_emitter(emitter_index);
    }
  }
}

void RTApplication::on_scene_transform_interaction_started() {
  if (_scene_transform_interaction_active) {
    return;
  }

  if (_active_renderer == nullptr) {
    return;
  }

  _scene_transform_interaction_active = true;
  _scene_transform_interaction_renderer = _active_renderer;
  _scene_transform_interaction_renderer->on_scene_transform_interaction_started(scene);
}

void RTApplication::on_scene_transform_interaction_finished() {
  if (_scene_transform_interaction_active == false) {
    return;
  }

  _scene_transform_interaction_active = false;
  Renderer* const interaction_renderer = _scene_transform_interaction_renderer;
  _scene_transform_interaction_renderer = nullptr;
  scene.update_medium_bounds();
  if (interaction_renderer != nullptr) {
    interaction_renderer->on_scene_transform_interaction_finished(scene);
  }
  notify_scene_might_have_changed();
}

void RTApplication::on_denoise_selected() {
  ETX_PROFILER_SCOPE();
  cpu_renderer.film().denoise(_view_parameters.view_layer, cpu_renderer.scene().options.radiance_clamp);
  _view_parameters.view_layer = ViewLayer::Denoised;
}

void RTApplication::update_camera_to_fit_scene(const float3& view_direction) {
  ETX_PROFILER_SCOPE();
  float3 position = {};
  float3 target = {};
  compute_camera_position_to_fit_scene(scene.data(), scene.camera(), view_direction, position, target);
  if (_active_renderer && _active_renderer->camera_controller()) {
    _active_renderer->camera_controller()->schedule(position, target);
  }
}

void RTApplication::on_view_scene(uint32_t direction) {
  mark_scene_dirty();
  ETX_PROFILER_SCOPE();
  constexpr float3 directions[] = {
    {1.0f, 1.0f, 1.0f},
    kWorldRight,
    -kWorldRight,
    kWorldUp,
    -kWorldUp,
    -kWorldForward,
    kWorldForward,
  };
  constexpr uint32_t direction_count = uint32_t(sizeof(directions) / sizeof(directions[0]));
  direction = clamp(direction, 0u, direction_count - 1u);
  update_camera_to_fit_scene(directions[direction]);
}

void RTApplication::on_clear_recent_files() {
  ETX_PROFILER_SCOPE();
  _recent_files.clear();
  save_options();
}

void RTApplication::on_camera_activated(uint32_t camera_index) {
  mark_scene_dirty();
  ETX_PROFILER_SCOPE();
  if (camera_index >= (uint32_t)scene.data().cameras.size()) {
    return;
  }

  for (auto& cam : scene.data().cameras) {
    cam.active = false;
  }

  scene.data().cameras[camera_index].active = true;

  on_camera_changed(scene.data().cameras[camera_index].cam.film_size, film.pixel_size());
  if (_active_renderer != nullptr) {
    _active_renderer->on_camera_changed(scene);
  }
}

void RTApplication::notify_scene_might_have_changed() {
  cpu_renderer.on_scene_changed(scene);
  raster_renderer.on_scene_changed(scene);
  gpu_renderer.on_scene_changed(scene);
}

void RTApplication::notify_scene_transforms_changed() {
  if (_scene_transform_interaction_active == false) {
    scene.update_medium_bounds();
  }
  cpu_renderer.on_scene_transforms_changed(scene);
  raster_renderer.on_scene_transforms_changed(scene);
  gpu_renderer.on_scene_transforms_changed(scene);
}

void RTApplication::sync_ui_renderer_state() {
  RendererPreparationStatus preparation = _active_renderer ? _active_renderer->preparation_status() : RendererPreparationStatus{};
  if (_material_render_resource_preparation_active) {
    const EnergyCompensationPreparationStatus material_status = scene.energy_compensation_interface_preparation_status();
    preparation = {
      .state = RendererPreparationState::Preparing,
      .phase = "Building material energy tables",
      .message = "Generating spectral energy-compensation interfaces",
      .completed_steps = material_status.completed_steps,
      .total_steps = material_status.total_steps,
      .elapsed_seconds = material_status.elapsed_seconds,
      .remaining_seconds = material_status.remaining_seconds,
      .remaining_available = material_status.remaining_available,
      .cancelable = false,
    };
  }
  ui.set_current_renderer_preparation(preparation);
  ui.set_current_renderer_status(_active_renderer ? _active_renderer->status() : RendererStatus{.mode = ui.current_renderer_mode()});
  ui.set_memory_stats(render_context.get_context().device().get_memory_statistics(), _active_renderer ? _active_renderer->memory_stats() : RendererMemoryStats{});
  ui.set_current_renderer_controls((_active_renderer && !_current_scene_file.empty()) ? _active_renderer->control_state() : RendererControlState{});
  ui.set_gpu_kernel_timing_stats((_active_renderer == &gpu_renderer) ? gpu_renderer.kernel_timing_stats() : RendererKernelTimingStats{});
  ui.set_gpu_wavefront_schedule(gpu_renderer.wavefront_steps_per_render(), gpu_renderer.wavefront_last_batch_ms(), gpu_renderer.wavefront_auto_tuning_enabled());
}

void RTApplication::process_application_commands() {
  std::vector<ApplicationCommand> commands = {};
  {
    std::lock_guard<std::mutex> lock(_application_control_mutex);
    commands = std::move(_pending_application_commands);
    _pending_application_commands.clear();
  }

  if (commands.empty()) {
    return;
  }

  std::vector<ApplicationCommandResult> results = {};
  results.reserve(commands.size());
  for (const ApplicationCommand& command : commands) {
    std::string message = {};
    const bool success = execute_application_command(command, message);
    results.push_back({
      .command_id = command.id,
      .success = success,
      .message = std::move(message),
    });
  }

  sync_ui_renderer_state();
  publish_application_state();

  std::lock_guard<std::mutex> lock(_application_control_mutex);
  for (ApplicationCommandResult& result : results) {
    _application_command_results.push_back(std::move(result));
  }
  if (_application_command_results.size() > kRetainedApplicationCommandResultLimit) {
    const size_t excess = _application_command_results.size() - kRetainedApplicationCommandResultLimit;
    _application_command_results.erase(_application_command_results.begin(), _application_command_results.begin() + static_cast<std::ptrdiff_t>(excess));
  }
}

bool RTApplication::execute_application_command(const ApplicationCommand& command, std::string& message) {
  switch (command.type) {
    case ApplicationCommandType::LoadScene:
      if (command.path.empty()) {
        message = "Scene path is required";
        return false;
      }
      if (!load_scene_file(command.path, SceneRepresentation::LoadEverything, false)) {
        message = "Scene loading failed";
        return false;
      }
      message = "Scene loaded";
      return true;

    case ApplicationCommandType::SaveScene:
      if (command.path.empty() && _current_scene_file.empty()) {
        message = "No scene path is available";
        return false;
      }
      {
        std::string path = command.path.empty() ? _current_scene_file : command.path;
        if (std::strlen(get_file_ext(path.c_str())) == 0u) {
          path += ".json";
        }
        if (!save_scene_file(path).empty()) {
          message = "Scene saved";
          return true;
        }
        message = "Scene save failed";
        return false;
      }

    case ApplicationCommandType::LoadReferenceImage: {
      std::error_code error = {};
      if (command.path.empty() || !std::filesystem::is_regular_file(command.path, error)) {
        message = "Reference image does not exist";
        return false;
      }
    }
      on_referenece_image_selected(command.path);
      message = "Reference image load requested";
      return true;

    case ApplicationCommandType::SaveImage:
      if (command.path.empty() || _current_scene_file.empty() || (scene.valid() == false)) {
        message = "A scene and output path are required";
        return false;
      }
      on_save_image_selected(command.path, command.save_image_mode);
      message = "Image save requested";
      return true;

    case ApplicationCommandType::Denoise: {
      if (_current_scene_file.empty() || (scene.valid() == false)) {
        message = "No scene is loaded";
        return false;
      }
      if (_active_renderer != &cpu_renderer) {
        message = "Denoising requires stopped CPU output with at least one rendered sample";
        return false;
      }
      const RendererStatus renderer_status = _active_renderer->status();
      if ((_active_renderer->control_state().can_run == false) || (renderer_status.completed_units == 0u)) {
        message = "Denoising requires stopped CPU output with at least one rendered sample";
        return false;
      }
      on_denoise_selected();
      message = "Image denoised";
      return true;
    }

    case ApplicationCommandType::SetRenderConfiguration:
      if (set_render_configuration(command.renderer, command.integrator) == false) {
        message = "Requested integrator is unavailable";
        return false;
      }
      message = "Integrator changed";
      return true;

    case ApplicationCommandType::SetRenderer:
      if ((command.renderer == RendererMode::GPURaytracing) && (_gpu_renderer_supported == false)) {
        message = "GPU ray tracing is unavailable";
        return false;
      }
      if ((command.renderer == RendererMode::GPURaytracing) &&
          ((cpu_renderer.current_integrator() == nullptr) || (UI::gpu_integrator_supported(cpu_renderer.current_integrator()->type()) == false))) {
        message = "Current integrator is unavailable on GPU";
        return false;
      }
      set_renderer_mode(command.renderer);
      if ((_active_renderer == nullptr) || (_active_renderer->mode() != command.renderer)) {
        message = "Requested renderer could not be initialized";
        return false;
      }
      message = "Renderer changed";
      return true;

    case ApplicationCommandType::SetIntegrator: {
      Integrator* integrator = integrator_type_to_instance(command.integrator, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
      if ((integrator == nullptr) || (integrator->enabled() == false)) {
        message = "Integrator is unavailable";
        return false;
      }
      if ((_active_renderer == &gpu_renderer) && (UI::gpu_integrator_supported(command.integrator) == false)) {
        message = "Integrator is unavailable on GPU";
        return false;
      }
      on_integrator_selected(command.integrator);
      message = "Integrator changed";
      return true;
    }

    case ApplicationCommandType::Run:
      if (_current_scene_file.empty() || (_active_renderer == nullptr) || (_active_renderer->control_state().can_run == false)) {
        message = "Renderer cannot start in its current state";
        return false;
      }
      on_run_selected();
      message = "Renderer started";
      return true;

    case ApplicationCommandType::Finish:
      if (_current_scene_file.empty() || (_active_renderer == nullptr) || (_active_renderer->control_state().can_finish == false)) {
        message = "Renderer cannot finish in its current state";
        return false;
      }
      on_stop_selected(true);
      message = "Renderer will stop after the current iteration";
      return true;

    case ApplicationCommandType::Stop:
      if (_current_scene_file.empty() || (_active_renderer == nullptr) || (_active_renderer->control_state().can_stop == false)) {
        message = "Renderer cannot stop in its current state";
        return false;
      }
      on_stop_selected(false);
      message = "Renderer stopped";
      return true;

    case ApplicationCommandType::Restart:
      if (_current_scene_file.empty() || (_active_renderer == nullptr) || (_active_renderer->control_state().can_restart == false)) {
        message = "Renderer cannot restart in its current state";
        return false;
      }
      on_restart_selected();
      message = "Renderer restarted";
      return true;

    case ApplicationCommandType::ReloadScene:
      if (_current_scene_file.empty()) {
        message = "No scene is loaded";
        return false;
      }
      {
        const bool was_running = (_active_renderer != nullptr) && _active_renderer->is_running();
        const bool reloaded = load_scene_file(_current_scene_file, SceneRepresentation::LoadEverything, was_running);
        message = reloaded ? "Scene reloaded" : "Scene reload failed";
        return reloaded;
      }

    case ApplicationCommandType::ReloadGeometry:
      if (_current_scene_file.empty()) {
        message = "No scene is loaded";
        return false;
      }
      {
        const bool was_running = (_active_renderer != nullptr) && _active_renderer->is_running();
        const bool reloaded = load_scene_file(_current_scene_file, SceneRepresentation::LoadGeometry, was_running);
        message = reloaded ? "Geometry reloaded" : "Geometry reload failed";
        return reloaded;
      }

    case ApplicationCommandType::ReloadShaders:
      if ((_active_renderer != &gpu_renderer) || !_gpu_renderer_initialized || _current_scene_file.empty()) {
        message = "GPU renderer is not active";
        return false;
      }
      on_reload_shaders_selected();
      message = "Shader reload requested";
      return true;

    case ApplicationCommandType::CancelPreparation:
      if ((_active_renderer != &gpu_renderer) || (gpu_renderer.preparation_status().state != RendererPreparationState::Preparing)) {
        message = "Renderer preparation is not active";
        return false;
      }
      on_cancel_renderer_preparation_selected();
      message = "Renderer preparation canceled";
      return true;

    case ApplicationCommandType::SetExposure:
      if (!std::isfinite(command.float_value)) {
        message = "Exposure must be finite";
        return false;
      }
      _view_parameters.exposure = std::clamp(command.float_value, 1.0f / 1024.0f, 1024.0f);
      message = "Exposure changed";
      return true;

    case ApplicationCommandType::SetViewLayer:
      if (command.unsigned_value >= ViewLayer::Count) {
        message = "Invalid view layer";
        return false;
      }
      _view_parameters.view_layer = command.unsigned_value;
      message = "View layer changed";
      return true;

    case ApplicationCommandType::SetOutputView:
      if (command.unsigned_value >= static_cast<uint32_t>(OutputView::Count)) {
        message = "Invalid output view";
        return false;
      }
      _view_parameters.view_image = command.unsigned_value;
      message = "Output view changed";
      return true;

    case ApplicationCommandType::SetDisplayTransform:
      if (command.unsigned_value >= static_cast<uint32_t>(ViewOptions::Count)) {
        message = "Invalid display transform";
        return false;
      }
      _view_parameters.view_option = command.unsigned_value;
      message = "Display transform changed";
      return true;

    case ApplicationCommandType::Quit: {
      std::lock_guard<std::mutex> lock(_application_control_mutex);
      _application_quit_requested = true;
    }
      if (_application_config.runtime_mode == RuntimeMode::Desktop) {
        sapp_request_quit();
      }
      message = "Quit requested";
      return true;
  }

  message = "Unknown command";
  return false;
}

void RTApplication::publish_application_state() {
  ApplicationStateSnapshot state = {};
  state.initialized = _initialized.load();
  state.scene_loaded = !_current_scene_file.empty() && scene.valid();
  state.gpu_renderer_available = _gpu_renderer_supported;
  state.scene_file = _current_scene_file;
  state.renderer_mode = _active_renderer ? _active_renderer->mode() : RendererMode::CPURaytracing;
  state.renderer_name = _active_renderer ? _active_renderer->name() : "None";
  state.preparation = _active_renderer ? _active_renderer->preparation_status() : RendererPreparationStatus{};
  state.status = _active_renderer ? _active_renderer->status() : RendererStatus{.mode = state.renderer_mode};
  state.controls = state.scene_loaded && _active_renderer ? _active_renderer->control_state() : RendererControlState{};
  state.can_denoise = state.scene_loaded && (_active_renderer == &cpu_renderer) && state.controls.can_run && (state.status.progress_kind == RendererProgressKind::Samples) &&
                      (state.status.completed_units > 0u);
  state.view = _view_parameters;
  if (Integrator* integrator = cpu_renderer.current_integrator()) {
    state.integrator_type = integrator->type();
    state.integrator_name = integrator->name();
  }
  for (uint64_t index = 0u; index < cpu_renderer.integrator_count(); ++index) {
    Integrator* integrator = cpu_renderer.integrator_list()[index];
    if (integrator == nullptr) {
      continue;
    }
    const char* id = integrator_type_to_id(integrator->type());
    state.integrators.push_back({
      .value = static_cast<uint32_t>(integrator->type()),
      .id = id ? id : "",
      .name = integrator->name(),
      .enabled = integrator->enabled(),
    });
  }

  std::lock_guard<std::mutex> lock(_application_control_mutex);
  state.revision = _application_state.revision + 1u;
  state.quit_requested = _application_quit_requested;
  _application_state = std::move(state);
}

void RTApplication::sync_scene_integrator_data_from_current_integrator() {
  Integrator* current = cpu_renderer.current_integrator();
  if (current == nullptr) {
    return;
  }

  current->sync_from_options(current->options());

  SceneRepresentation::IntegratorData integrator_data = scene.integrator_data();
  integrator_data.selected = current->type();
  integrator_data.settings[current->type()] = current->options();
  scene.set_integrator_data(integrator_data);
}

bool RTApplication::rebuild_material_render_resources() {
  if (scene.ensure_energy_compensation_interfaces() == false) {
    log::error("Failed to rebuild material energy-compensation interfaces");
    return false;
  }
  return true;
}

void RTApplication::poll_material_render_resource_preparation() {
  if (_material_render_resource_preparation_active == false) {
    return;
  }

  const EnergyCompensationPreparationState state = scene.poll_energy_compensation_interface_preparation();
  if (state == EnergyCompensationPreparationState::Preparing) {
    return;
  }

  if (state == EnergyCompensationPreparationState::Failed) {
    log::error("Failed to prepare material energy-compensation interfaces");
    finish_material_render_resource_preparation(false);
    return;
  }

  finish_material_render_resource_preparation(true);
}

void RTApplication::finish_material_render_resource_preparation(bool resources_ready) {
  _material_render_resource_preparation_active = false;
  if (resources_ready == false) {
    return;
  }
  notify_scene_might_have_changed();
  if (_restart_cpu_after_material_resource_preparation && (_material_interaction_active == false)) {
    cpu_renderer.restart();
    _restart_cpu_after_material_resource_preparation = false;
  }
  if (_restart_gpu_after_material_resource_preparation && (_material_interaction_active == false)) {
    gpu_renderer.restart();
    _restart_gpu_after_material_resource_preparation = false;
  }
}

void RTApplication::on_reload_shaders_selected() {
  ETX_PROFILER_SCOPE();
  if (_active_renderer == &gpu_renderer) {
    gpu_renderer.reload_shaders(render_context.get_context(), scene);
  }
}

void RTApplication::on_cancel_renderer_preparation_selected() {
  ETX_PROFILER_SCOPE();
  if (_active_renderer == &gpu_renderer) {
    set_renderer_mode(RendererMode::CPURaytracing);
  }
}

}  // namespace etx
