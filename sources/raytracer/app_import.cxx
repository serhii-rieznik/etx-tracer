#include "app.hxx"
#include <etx/render/host/scene_document_merge.hxx>

namespace etx {

void RTApplication::cancel_scene_import() {
  if (_import_job == nullptr)
    return;
  _import_job->request_cancel();
  if (_import_job->worker.joinable())
    _import_job->worker.join();
  _import_job.reset();
}

bool RTApplication::begin_scene_import(const ApplicationCommand& command, std::string& error) {
  if (_import_job != nullptr) {
    error = "An import is already running.";
    return false;
  }
  if (command.path.empty()) {
    error = "Import source path is required.";
    return false;
  }
  if ((command.type == ApplicationCommandType::ImportIntoScene) && (scene.valid() == false)) {
    error = "Open a native document before adding an import.";
    return false;
  }
  if ((command.type == ApplicationCommandType::ConvertScene) && command.output_path.empty()) {
    error = "Conversion destination path is required.";
    return false;
  }
  _import_job = std::make_unique<SceneImportJob>();
  auto& job = *_import_job;
  job.command = command;
  job.target_identity = _document_identity;
  // Discover modules on the application thread; the worker retains the service until conversion returns.
  auto& service = import_service();
  job.worker = std::thread([&job, &service] {
    try {
      job.artifact = service.convert(
        std::filesystem::u8path(job.command.path), job.command.importer_id,
        [&job](uint32_t completed, uint32_t total, const char* stage) {
          std::lock_guard<std::mutex> lock(job.mutex);
          job.completed = completed;
          job.total = total;
          job.stage = stage;
          return job.cancelled() == false;
        },
        job.error);
      if ((job.artifact != nullptr) && (job.command.type == ApplicationCommandType::ConvertScene)) {
        std::string output;
        auto expected = SceneImportJob::State::Converting;
        if (job.state.compare_exchange_strong(expected, SceneImportJob::State::Publishing) == false) {
          job.error = "Import cancelled.";
          job.artifact.reset();
        } else {
          {
            std::lock_guard<std::mutex> lock(job.mutex);
            job.stage = "Publishing native document";
          }
          if (publish_import_document(*job.artifact, std::filesystem::u8path(job.command.output_path), output, job.error) == false)
            job.artifact.reset();
          else {
            std::lock_guard<std::mutex> lock(job.mutex);
            job.stage = output;
          }
        }
      }
    } catch (const std::exception& failure) {
      job.error = failure.what();
    }
    job.done.store(true);
  });
  return true;
}

void RTApplication::poll_scene_import() {
  if ((_import_job == nullptr) || (_import_job->done.load() == false))
    return;
  auto job = std::move(_import_job);
  job->worker.join();
  bool success = false;
  std::string message = job->error;
  if (job->cancelled())
    message = "Import cancelled.";
  else if (job->artifact != nullptr) {
    if ((job->command.type != ApplicationCommandType::ConvertScene) && (job->target_identity != _document_identity)) {
      message = "The target document changed while importing; the result was discarded.";
    } else if (job->command.type == ApplicationCommandType::ImportIntoScene) {
      success = add_scene_file(path_to_utf8(job->artifact->document), message, true);
      if (success)
        _document_imports.push_back(job->artifact);
    } else if (job->command.type == ApplicationCommandType::ConvertScene) {
      success = true;
      message = job->stage;
    } else {
      success = load_scene_file(path_to_utf8(job->artifact->document), SceneRepresentation::LoadEverything, false, true);
      if (success) {
        _document_imports.push_back(job->artifact);
        _scene_dirty = true;
        message = "Imported into an unsaved native document.";
        save_options();
      } else
        message = "The converted document could not be prepared.";
    }
  }
  if (success && message.empty())
    message = "Imported into the current document.";
  _command_failure_reason = success ? std::string{} : message;
  std::lock_guard<std::mutex> lock(_application_control_mutex);
  _application_command_results.push_back({job->command.id, success, std::move(message)});
}

bool RTApplication::add_scene_file(const std::string& path, std::string& error, bool owned_source) {
  if (scene.valid() == false) {
    error = "Open a native document before adding another document.";
    return false;
  }
  if (ui.commit_pending_edits(scene) == false) {
    error = "Pending scene edits could not be applied.";
    return false;
  }
  SceneRepresentation source(scheduler, _ior_database);
  if (source.load_from_file(path.c_str(), SceneRepresentation::DocumentOnly) == false) {
    error = "Cannot read the native source document.";
    return false;
  }
  if (owned_source)
    source.data().owns_assets = true;
  SceneRepresentation staged(scheduler, _ior_database);
  if ((append_scene_document(staged.data(), scene.data(), error) == false) || (append_scene_document(staged.data(), source.data(), error) == false))
    return false;
  staged.mutable_camera() = scene.camera();
  staged.store_active_camera();
  staged.set_integrator_data(scene.integrator_data());
  staged.set_scattering_rhi(render_context.get_context());
  if (staged.prepare_document(SceneRepresentation::LoadGeometry) == false) {
    error = "Merged document preparation failed.";
    return false;
  }
  cancel_preview();
  if (_active_renderer != nullptr)
    _active_renderer->stop();
  scene.replace_loaded_scene(staged);
  ui.invalidate_scene_resources();
  if ((_active_renderer != nullptr) && (_active_renderer->camera_controller() != nullptr))
    _active_renderer->camera_controller()->sync_from_camera();
  _material_render_resource_preparation_active = false;
  _material_render_resource_preparation_failed = false;
  _restart_cpu_after_material_resource_preparation = false;
  _restart_gpu_after_material_resource_preparation = false;
  notify_scene_might_have_changed();
  mark_scene_dirty();
  return true;
}

}  // namespace etx
