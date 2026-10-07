bool SceneRepresentation::load_from_file(const char* filename, uint32_t options, IntegratorData* out_integrator) {
  try {
    return load_document(filename, nullptr, false, "", options, out_integrator);
  } catch (const std::exception& error) {
    log::error("Invalid native document: %s", error.what());
    return false;
  }
}

bool SceneRepresentation::prepare_document(uint32_t options) {
  return _private->finalize_scene_loading(options, "", SceneLoadSucceeded, get_camera_fov(_private->active_camera), false, 50.0f, false,
    _private->data.options.properties[Scene::Properties::Spectral], _private->data.pixel_filter.radius, true);
}

bool SceneRepresentation::load_document(const char* filename, SourceDecoder decoder, bool decode_root, const char* source_assets, uint32_t options,
  IntegratorData* out_integrator) {
  if ((filename == nullptr) || (filename[0] == 0)) {
    return false;
  }
  std::string committed_scene_file;
  if (decoder == nullptr) {
    const auto input_path = std::filesystem::u8path(filename);
    std::string error;
    if (validate_native_document(input_path, error) == false) {
      std::error_code file_error;
      const bool missing_json = (_stricmp(get_file_ext(filename), ".json") == 0) && (std::filesystem::exists(input_path, file_error) == false) && (file_error.value() == 0);
      if (missing_json == false) {
        log::error("%s", error.c_str());
        return false;
      }
    }
    const SceneSavePaths interrupted_save_paths = scene_save_paths(input_path);
    std::array<StagedSceneFile, 3u> interrupted_save_files = staged_scene_files(interrupted_save_paths);
    bool committed_scene_available = false;
    if (recover_interrupted_scene_save(interrupted_save_files, committed_scene_available) == false)
      return false;
    if (committed_scene_available && ((options & PreferRecoveredSave) != 0u)) {
      committed_scene_file = path_to_utf8(interrupted_save_paths.json);
      filename = committed_scene_file.c_str();
    }
    if (validate_native_document(std::filesystem::u8path(filename), error) == false) {
      log::error("%s", error.c_str());
      return false;
    }
  }

  IntegratorData parsed_integrator_data = {};
  IntegratorData* integrator_data = out_integrator;
  if (integrator_data == nullptr) {
    integrator_data = &parsed_integrator_data;
  } else {
    *integrator_data = {};
  }

  char base_folder[2048] = {};
  get_file_folder(filename, base_folder, sizeof(base_folder));

  _private->cleanup();
  _private->data.source_asset_directory = source_assets;
  _private->data.json_file_name = {};
  _private->data.materials_file_name = {};
  _private->data.geometry_file_name = filename;
  _private->active_camera.lens_radius = 0.0f;
  _private->active_camera.focal_distance = 0.0f;
  _private->active_camera.lens_image = kInvalidIndex;
  _private->active_camera.medium_index = kInvalidIndex;
  _private->active_camera.up = kWorldUp;

  Camera default_camera = {};
  default_camera.lens_image = kInvalidIndex;
  default_camera.medium_index = kInvalidIndex;
  default_camera.up = kWorldUp;
  default_camera.cls = Camera::Class::Perspective;

  float3 camera_target = default_camera.position + default_camera.direction;
  bool has_target = false;
  bool has_direction = false;
  float camera_focal_len = 50.0f;
  float camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
  bool use_focal_len = false;
  bool force_tangents = false;
  bool spectral_scene = false;
  float pixel_filter_radius = 1.5f;
  nlohmann::json hierarchy_json;
  nlohmann::json material_names_json;
  nlohmann::json medium_names_json;
  nlohmann::json camera_names_json;
  nlohmann::json spectral_overrides_json;

  if (decode_root) {
    const uint32_t load_result = decoder(filename, "", _private->data, _private->ior_database, _private->scheduler, _private->active_camera);
    if ((load_result & SceneLoadSucceeded) == 0)
      return false;
    _private->integrator_data = *integrator_data;
    return _private->finalize_scene_loading(options, base_folder, load_result, camera_fov, use_focal_len, camera_focal_len, force_tangents, spectral_scene, pixel_filter_radius,
      false);
  }

  const bool raw_model_file = (_stricmp(get_file_ext(filename), ".json") != 0);

  if (raw_model_file == false) {
    std::string json_content;
    if (auto f = fopen_utf8(filename, "rb")) {
      size_t file_size = get_file_size(f);
      if (file_size > 0) {
        json_content.resize(file_size);
        size_t read_bytes = fread(json_content.data(), 1, json_content.size(), f);
        json_content.resize(read_bytes);
      }
      fclose(f);
    }

    nlohmann::json js = nlohmann::json::parse(json_content, nullptr, false);
    bool parsed = js.is_discarded() == false;
    const bool is_native = parsed && js.is_object() && (js.contains("geometry") || js.contains("materials") || js.contains("integrator"));

    if (parsed == false) {
      log::error("Failed to parse JSON scene %s", filename);
      return false;
    }

    if (is_native) {
      _private->data.geometry_file_name.clear();
    }

    for (auto i = js.begin(), e = js.end(); i != e; ++i) {
      const auto& key = i.key();
      const auto& obj = i.value();
      std::string str_value = {};
      float float_value = 0.0f;
      int64_t int_value = 0;
      bool bool_value = false;
      if (key == "etx_document") {
        _private->data.owns_assets = obj.value("assets", "linked") == "owned";
      } else if (json_get_int(i, "samples", int_value)) {
        _private->data.options.samples = static_cast<uint32_t>(max(int64_t(1), int_value));
      } else if (json_get_int(i, "random-termination-start", int_value)) {
        _private->data.options.random_path_termination = static_cast<uint32_t>(max(int64_t(1), int_value));
      } else if (json_get_int(i, "max-path-length", int_value)) {
        _private->data.options.max_path_length = static_cast<uint32_t>(max(int64_t(1), int_value));
      } else if (json_get_int(i, "min-path-length", int_value)) {
        _private->data.options.min_path_length = static_cast<uint32_t>(max(int64_t(1), int_value));

      } else if ((json_get_float(i, "radiance-clamp", float_value)) && (std::isfinite(float_value))) {
        _private->data.options.radiance_clamp = max(float_value, 0.0f);
      } else if ((json_get_float(i, "pixel-filter-radius", float_value)) && (std::isfinite(float_value))) {
        pixel_filter_radius = clamp(float_value, 0.0f, 32.0f);
      } else if (json_get_string(i, "geometry", str_value)) {
        _private->data.geometry_file_name = path_to_utf8((std::filesystem::u8path(base_folder) / std::filesystem::u8path(str_value)).lexically_normal());
      } else if (json_get_string(i, "materials", str_value)) {
        _private->data.materials_file_name = path_to_utf8((std::filesystem::u8path(base_folder) / std::filesystem::u8path(str_value)).lexically_normal());
      } else if (json_get_bool(i, "spectral", bool_value)) {
        spectral_scene = bool_value;
      } else if (json_get_bool(i, "energy_compensated_specular", bool_value)) {
        (void)bool_value;
      } else if (json_get_bool(i, "multiple_importance_sampling", bool_value)) {
        _private->data.options.properties[Scene::Properties::MultipleImportanceSampling] = bool_value;
      } else if (json_get_bool(i, "blue_noise", bool_value)) {
        _private->data.options.properties[Scene::Properties::BlueNoise] = bool_value;
      } else if ((key == "scene_hierarchy") && obj.is_object()) {
        hierarchy_json = obj;
      } else if (key == "material_names") {
        material_names_json = obj;
      } else if (key == "medium_names") {
        medium_names_json = obj;
      } else if (key == "camera_names") {
        camera_names_json = obj;
      } else if (key == "spectral_overrides") {
        spectral_overrides_json = obj;
      } else if ((key == "emitter_names") && obj.is_array()) {
        _private->data.emitter_names.clear();
        for (const nlohmann::json& emitter_name : obj) {
          if (emitter_name.is_string() == false) {
            _private->data.emitter_names.clear();
            break;
          }
          _private->data.emitter_names.push_back(emitter_name.get<std::string>());
        }
      } else if (json_get_string(i, "light_sampling", str_value)) {
        if (str_value == "uniform") {
          _private->data.options.light_sampling = Scene::LightSampling::Uniform;
        } else if (str_value == "from_distribution") {
          _private->data.options.light_sampling = Scene::LightSampling::FromDistribution;
        } else if (str_value == "ris_uniform") {
          _private->data.options.light_sampling = Scene::LightSampling::RIS_Uniform;
        } else if (str_value == "ris_from_distribution") {
          _private->data.options.light_sampling = Scene::LightSampling::RIS_FromDistribution;
        }
      } else if (key == "strategies" && obj.is_object()) {
        uint32_t strategy_flags = Scene::Strategy::Default;
        for (auto strat_it = obj.begin(); strat_it != obj.end(); ++strat_it) {
          const std::string& strat_key = strat_it.key();
          if (strat_it.value().is_boolean() == false) {
            continue;
          }
          bool strat_value = strat_it.value().get<bool>();
          if (strat_key == "direct_hit") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::DirectHit)) | (strat_value ? Scene::Strategy::DirectHit : 0u);
          } else if (strat_key == "next_event_estimation") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectToLight)) | (strat_value ? Scene::Strategy::ConnectToLight : 0u);
          } else if (strat_key == "connect_to_light") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectToLight)) | (strat_value ? Scene::Strategy::ConnectToLight : 0u);
          } else if (strat_key == "connect_to_camera") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectToCamera)) | (strat_value ? Scene::Strategy::ConnectToCamera : 0u);
          } else if (strat_key == "connect_vertices") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectVertices)) | (strat_value ? Scene::Strategy::ConnectVertices : 0u);
          } else if (strat_key == "merge_vertices") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::MergeVertices)) | (strat_value ? Scene::Strategy::MergeVertices : 0u);
          } else if (strat_key == "multiple_importance_sampling") {
            _private->data.options.properties[Scene::Properties::MultipleImportanceSampling] = strat_value;
          } else if (strat_key == "blue_noise") {
            _private->data.options.properties[Scene::Properties::BlueNoise] = strat_value;
          }
        }
        _private->data.options.strategy_flags = strategy_flags;
      } else if (json_get_bool(i, "force-tangents", bool_value)) {
        force_tangents = bool_value;
      } else if ((key == "camera") && obj.is_object()) {
        for (auto ci = obj.begin(), ce = obj.end(); ci != ce; ++ci) {
          const auto& ckey = ci.key();
          const auto& cobj = ci.value();
          if (json_get_string(ci, "class", str_value)) {
            default_camera.cls = str_value == "eq" ? Camera::Class::Equirectangular : Camera::Class::Perspective;
          } else if (json_get_float(ci, "fov", float_value)) {
            camera_fov = float_value;
          } else if (json_get_float(ci, "focal-length", float_value)) {
            camera_focal_len = float_value;
            use_focal_len = true;
          } else if (json_get_float(ci, "lens-radius", float_value)) {
            default_camera.lens_radius = float_value;
          } else if (json_get_float(ci, "focal-distance", float_value)) {
            default_camera.focal_distance = float_value;
          } else if (json_get_float(ci, "clip-near", float_value)) {
            default_camera.clip_near = float_value;
          } else if (json_get_float(ci, "clip-far", float_value)) {
            default_camera.clip_far = float_value;
          } else if (cobj.is_array()) {
            if (ckey == "origin") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &default_camera.position.x, 3llu);
            } else if (ckey == "target") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &camera_target.x, 3llu);
              has_target = true;
            } else if (ckey == "direction") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &default_camera.direction.x, 3llu);
              has_direction = true;
            } else if (ckey == "up") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &default_camera.up.x, 3llu);
            } else if (ckey == "viewport") {
              auto values = cobj.get<std::vector<uint32_t>>();
              get_values(values, &default_camera.film_size.x, 2llu);
            } else {
              log::warning("Unhandled value in camera description : %s", key.c_str());
            }
          }
        }

        if (has_direction) {
          default_camera.direction = normalize(default_camera.direction);
        } else if (has_target) {
          default_camera.direction = normalize(camera_target - default_camera.position);
        } else {
          default_camera.direction = kWorldForward;
        }
      } else if ((key == "integrator") && obj.is_object()) {
        if (integrator_data != nullptr) {
          std::string selected_id_str;
          if (obj.contains("selected") && obj["selected"].is_string()) {
            selected_id_str = obj["selected"].get<std::string>();
            integrator_data->selected = legacy_integrator_selection_to_type(selected_id_str);
          }
          if (integrator_data->selected == Integrator::Type::Invalid) {
            if (obj.contains("type") && obj["type"].is_string()) {
              integrator_data->selected = legacy_integrator_selection_to_type(obj["type"].get<std::string>());
            } else if (obj.contains("name") && obj["name"].is_string()) {
              std::string name = obj["name"].get<std::string>();
              if (name.find("Path Tracing") != std::string::npos) {
                integrator_data->selected = Integrator::Type::PathTracing;
              } else if (name.find("Bidirectional") != std::string::npos) {
                integrator_data->selected = Integrator::Type::Bidirectional;
              } else if (name.find("Distilled") != std::string::npos) {
                integrator_data->selected = Integrator::Type::Bidirectional;
              } else if (name.find("VCM") != std::string::npos) {
                integrator_data->selected = Integrator::Type::VCM;
              } else if (name.find("UPBP") != std::string::npos) {
                integrator_data->selected = Integrator::Type::UPBP;
              } else if (name.find("Debug") != std::string::npos) {
                integrator_data->selected = Integrator::Type::Debug;
              }
            }
          }

          if (obj.contains("settings") && obj["settings"].is_object()) {
            const auto& settings_obj = obj["settings"];
            for (auto it = settings_obj.begin(); it != settings_obj.end(); ++it) {
              const std::string& type_id = it.key();
              const auto& options_array = it.value();

              Integrator::Type type = integrator_id_to_type(type_id.c_str());
              if (type == Integrator::Type::Invalid)
                continue;

              if (options_array.is_array()) {
                Options options;
                if (options.deserialize_from_json(options_array)) {
                  integrator_data->settings[type] = std::move(options);
                }
              }
            }
          }

          if ((integrator_data->selected != Integrator::Type::Invalid) && obj.contains("options") && obj["options"].is_array()) {
            Options options;
            if (options.deserialize_from_json(obj["options"])) {
              integrator_data->settings[integrator_data->selected] = std::move(options);
            }
          }
        }
      } else {
        log::warning("Unhandled value in scene description : %s", key.c_str());
      }
    }
    _private->data.json_file_name = filename;
  }

  _private->integrator_data = *integrator_data;
  _private->integrator_data_revision += 1u;

  uint32_t load_result = SceneLoadFailed;

  const char* materials_file_name = _private->data.materials_file_name.c_str();
  if (_private->data.geometry_file_name.empty()) {
    if ((materials_file_name == nullptr) || (materials_file_name[0] == 0)) {
      log::error("Scene %s does not provide geometry or materials", filename);
      return false;
    }

    char materials_base_dir[2048] = {};
    get_file_folder(materials_file_name, materials_base_dir, sizeof(materials_base_dir));
    SceneSerialization loader;
    if (loader.parse_materials_file(std::filesystem::u8path(materials_file_name), materials_base_dir, _private->data, _private->ior_database, _private->scheduler) == false) {
      log::error("Failed to load materials from %s", materials_file_name);
      return false;
    }

    load_result = SceneLoadSucceeded;
  } else {
    const char* geometry_file_name = _private->data.geometry_file_name.c_str();
    auto ext = get_file_ext(geometry_file_name);
    if (_stricmp(ext, ".etx") == 0) {
      SceneSerialization loader;
      if (loader.load_from_file(std::filesystem::u8path(geometry_file_name), _private->data, materials_file_name, _private->ior_database, _private->scheduler) == false) {
        log::error("Failed to load ETX file from %s", geometry_file_name);
        return false;
      }
      load_result = SceneLoadSucceeded;
    } else if (decoder != nullptr) {
      load_result = decoder(geometry_file_name, materials_file_name, _private->data, _private->ior_database, _private->scheduler, _private->active_camera);
    }
  }

  if ((load_result & SceneLoadSucceeded) == 0) {
    return false;
  }

  if ((spectral_overrides_json.is_null() == false) && (apply_scene_spectral_overrides(spectral_overrides_json, _private->data) == false)) {
    log::error("Failed to restore spectral scene data from %s", filename);
    return false;
  }
  if ((medium_names_json.is_null() == false) && (restore_scene_medium_names(medium_names_json, _private->data) == false)) {
    log::error("Failed to restore medium names from %s", filename);
    return false;
  }
  if ((material_names_json.is_null() == false) && (restore_scene_material_names(material_names_json, _private->data) == false)) {
    log::error("Failed to restore material names from %s", filename);
    return false;
  }
  if ((camera_names_json.is_null() == false) && (restore_scene_camera_names(camera_names_json, _private->data) == false)) {
    log::error("Failed to restore camera names from %s", filename);
    return false;
  }

  if (hierarchy_json.is_null() == false) {
    _private->data.hierarchy.clear();
    if (deserialize_scene_hierarchy(hierarchy_json, _private->data) == false) {
      log::error("Failed to deserialize scene hierarchy from %s", filename);
      return false;
    }
  } else {
    synthesize_identity_scene_hierarchy(_private->data);
  }

  if (has_target || has_direction || default_camera.film_size.x > 0 || default_camera.lens_radius > 0.0f) {
    if (use_focal_len) {
      camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
    }

    if (default_camera.film_size.x * default_camera.film_size.y == 0) {
      default_camera.film_size = {1280, 720};
    }

    auto& entry = _private->data.cameras.emplace_back();
    entry.id = "default";
    entry.active = _private->data.cameras.size() == 1;

    build_camera(entry.cam, default_camera.position, default_camera.direction, default_camera.up, default_camera.film_size, camera_fov);

    entry.cam.cls = default_camera.cls;
    entry.cam.lens_radius = default_camera.lens_radius;
    entry.cam.focal_distance = default_camera.focal_distance;
    entry.cam.clip_near = default_camera.clip_near;
    entry.cam.clip_far = default_camera.clip_far;
  }

  return _private->finalize_scene_loading(options, base_folder, load_result, camera_fov, use_focal_len, camera_focal_len, force_tangents, spectral_scene, pixel_filter_radius,
    hierarchy_json.is_null() == false);
}
