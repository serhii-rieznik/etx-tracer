std::string SceneRepresentation::save_to_file(const char* filename, Integrator::Type selected_type, Integrator* integrator_array[], size_t integrator_count) {
  auto save_start = std::chrono::high_resolution_clock::now();

  auto impl = _private;

  std::string base_file = {};
  if ((filename != nullptr) && (filename[0] != 0)) {
    base_file = filename;
  } else if (impl->data.json_file_name.empty() == false) {
    base_file = impl->data.json_file_name;
  } else if (impl->data.geometry_file_name.empty() == false) {
    base_file = impl->data.geometry_file_name;
  }

  if (base_file.empty()) {
    log::error("Unable to determine base file for saving scene");
    return {};
  }

  const SceneSavePaths paths = scene_save_paths(std::filesystem::u8path(base_file));
  const std::filesystem::path& json_path = paths.json;
  const std::filesystem::path& materials_path = paths.materials;
  const std::filesystem::path& geometry_path = paths.geometry;
  const std::filesystem::path asset_directory = path_with_suffix(geometry_path, ".assets");
  if (json_path.parent_path().empty() == false) {
    std::error_code directory_error;
    std::filesystem::create_directories(json_path.parent_path(), directory_error);
    if (directory_error) {
      log::error("Could not create native document directory: %s", directory_error.message().c_str());
      return {};
    }
  }
  std::array<StagedSceneFile, 3u> scene_files = staged_scene_files(paths);
  bool committed_scene_available = false;
  if (recover_interrupted_scene_save(scene_files, committed_scene_available) == false) {
    return {};
  }

  std::vector<SerializedMaterialEntry> serialized_material_entries;
  if (build_serialized_material_entries(impl->data, serialized_material_entries) == false) {
    return {};
  }
  std::vector<SerializedMediumEntry> serialized_medium_entries;
  if (build_serialized_medium_entries(impl->data, serialized_medium_entries) == false) {
    return {};
  }
  const std::vector<SerializedCameraEntry> serialized_camera_entries = build_serialized_camera_entries(impl->data);
  SceneSerialization::MaterialNameMapping serialized_material_names;
  serialized_material_names.reserve(serialized_material_entries.size());
  for (const SerializedMaterialEntry& entry : serialized_material_entries) {
    serialized_material_names[entry.material_index] = entry.id;
  }

  auto to_relative = [](const std::filesystem::path& target, const std::filesystem::path& base_folder) {
    std::error_code ec = {};
    auto relative_path = std::filesystem::relative(target, base_folder, ec);
    if (ec.value() == 0) {
      std::string result = path_to_utf8(relative_path);
      if (result.empty()) {
        result = path_to_utf8(target);
      }
      return result;
    }

    return path_to_utf8(target);
  };

  std::string geometry_ref = to_relative(geometry_path, json_path.parent_path());
  std::string materials_ref = to_relative(materials_path, json_path.parent_path());

  nlohmann::json js = nlohmann::json::object();
  js["etx_document"] = {{"format", "etx"}, {"version", 1u}, {"assets", impl->data.owns_assets ? "owned" : "linked"}};
  js["samples"] = impl->data.options.samples;
  js["random-termination-start"] = impl->data.options.random_path_termination;
  js["max-path-length"] = impl->data.options.max_path_length;
  js["min-path-length"] = impl->data.options.min_path_length;
  js["radiance-clamp"] = impl->data.options.radiance_clamp;
  js["pixel-filter-radius"] = impl->data.pixel_filter.radius;
  js["geometry"] = geometry_ref;
  if (materials_ref.empty() == false) {
    js["materials"] = materials_ref;
  }
  js["spectral"] = impl->data.options.properties[Scene::Properties::Spectral];
  js["multiple_importance_sampling"] = impl->data.options.properties[Scene::Properties::MultipleImportanceSampling];
  js["blue_noise"] = impl->data.options.properties[Scene::Properties::BlueNoise];
  ensure_emitter_names(impl->data);
  js["scene_hierarchy"] = serialize_scene_hierarchy(impl->data);
  nlohmann::json material_names_json = nlohmann::json::array();
  for (const SerializedMaterialEntry& entry : serialized_material_entries) {
    material_names_json.push_back({{"id", entry.id}, {"names", entry.authored_names}});
  }
  js["material_names"] = std::move(material_names_json);
  nlohmann::json medium_names_json = nlohmann::json::array();
  for (const SerializedMediumEntry& entry : serialized_medium_entries) {
    medium_names_json.push_back({{"id", entry.id}, {"name", entry.authored_name}});
  }
  js["medium_names"] = std::move(medium_names_json);
  nlohmann::json camera_names_json = nlohmann::json::array();
  for (const SerializedCameraEntry& entry : serialized_camera_entries) {
    camera_names_json.push_back({{"id", entry.id}, {"name", entry.authored_name}});
  }
  if (camera_names_json.empty() == false) {
    js["camera_names"] = std::move(camera_names_json);
  }
  bool spectral_overrides_valid = false;
  nlohmann::json spectral_overrides = serialize_scene_spectral_overrides(impl->data, serialized_material_entries, serialized_medium_entries, spectral_overrides_valid);
  if (spectral_overrides_valid == false) {
    return {};
  }
  js["spectral_overrides"] = std::move(spectral_overrides);
  nlohmann::json emitter_names_json = nlohmann::json::array();
  for (uint32_t emitter_index : serialized_emitter_indices(impl->data)) {
    emitter_names_json.push_back(impl->data.emitter_names[emitter_index]);
  }
  js["emitter_names"] = std::move(emitter_names_json);

  switch (impl->data.options.light_sampling) {
    case Scene::LightSampling::Uniform:
      js["light_sampling"] = "uniform";
      break;
    case Scene::LightSampling::FromDistribution:
      js["light_sampling"] = "from_distribution";
      break;
    case Scene::LightSampling::RIS_Uniform:
      js["light_sampling"] = "ris_uniform";
      break;
    case Scene::LightSampling::RIS_FromDistribution:
      js["light_sampling"] = "ris_from_distribution";
      break;
    default:
      js["light_sampling"] = "ris_from_distribution";
      break;
  }

  nlohmann::json strategies = nlohmann::json::object();
  strategies["direct_hit"] = ((impl->data.options.strategy_flags & Scene::Strategy::DirectHit) == Scene::Strategy::DirectHit);
  strategies["connect_to_light"] = ((impl->data.options.strategy_flags & Scene::Strategy::ConnectToLight) == Scene::Strategy::ConnectToLight);
  strategies["connect_to_camera"] = ((impl->data.options.strategy_flags & Scene::Strategy::ConnectToCamera) == Scene::Strategy::ConnectToCamera);
  strategies["connect_vertices"] = ((impl->data.options.strategy_flags & Scene::Strategy::ConnectVertices) == Scene::Strategy::ConnectVertices);
  strategies["merge_vertices"] = ((impl->data.options.strategy_flags & Scene::Strategy::MergeVertices) == Scene::Strategy::MergeVertices);
  js["strategies"] = strategies;

  if (selected_type != Integrator::Type::Invalid && integrator_array != nullptr && integrator_count > 0) {
    nlohmann::json integrator_json;

    const char* selected_id = integrator_type_to_id(selected_type);
    if (selected_id != nullptr) {
      integrator_json["selected"] = selected_id;
    }

    nlohmann::json settings_json = nlohmann::json::object();

    for (size_t i = 0; i < integrator_count; ++i) {
      Integrator* integrator = integrator_array[i];
      if (integrator == nullptr)
        continue;

      Integrator::Type type = integrator_to_type(integrator);
      if (type == Integrator::Type::Invalid)
        continue;

      const char* type_id = integrator_type_to_id(type);
      if (type_id == nullptr)
        continue;

      nlohmann::json options_json;
      integrator->options().serialize_to_json(options_json);

      if (options_json.is_array() && options_json.size() > 0) {
        settings_json[type_id] = options_json;
      }
    }

    if (settings_json.empty() == false) {
      integrator_json["settings"] = settings_json;
    }

    if (integrator_json.empty() == false) {
      js["integrator"] = integrator_json;
    }
  }

  if ((js.contains("integrator") == false) && (impl->integrator_data.selected != Integrator::Type::Invalid)) {
    js["integrator"]["selected"] = integrator_type_to_id(impl->integrator_data.selected);
    for (const auto& [type, settings] : impl->integrator_data.settings) {
      nlohmann::json serialized;
      settings.serialize_to_json(serialized);
      js["integrator"]["settings"][integrator_type_to_id(type)] = std::move(serialized);
    }
  }

  auto sanitize_name = [](const std::string& value) {
    std::string result = value;
    for (char& ch : result) {
      if (std::isalnum(static_cast<unsigned char>(ch)) == 0) {
        ch = '_';
      }
    }
    return result;
  };

  std::unordered_map<uint32_t, std::string> medium_names;
  medium_names.reserve(serialized_medium_entries.size());
  for (const SerializedMediumEntry& entry : serialized_medium_entries) {
    medium_names[entry.medium_index] = entry.id;
  }

  auto spectrum_rgb = [&](uint32_t index) -> float3 {
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return {0.0f, 0.0f, 0.0f};
    }
    return impl->data.spectrum_values[index].integrated();
  };

  auto spectrum_scalar = [&](uint32_t index, float fallback) -> float {
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return fallback;
    }
    float3 rgb = impl->data.spectrum_values[index].integrated();
    return (rgb.x + rgb.y + rgb.z) / 3.0f;
  };

  auto spectrum_by_index = [&](uint32_t index) -> const SpectralDistribution& {
    static const SpectralDistribution null_spectrum = SpectralDistribution::constant(0.0f);
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return null_spectrum;
    }
    return impl->data.spectrum_values[index];
  };

  auto is_white_fallback_image = [&](uint32_t image_index) {
    if (image_index >= impl->data.images.array_size()) {
      return false;
    }
    const Image& image = impl->data.images.get(image_index);
    if ((image.isize.x != 1u) || (image.isize.y != 1u) || (image.isize.z != 1u) || (image.format != Image::Format::RGBA32F) || (image.pixels.f32.a == nullptr) ||
        (image.pixels.f32.count == 0u)) {
      return false;
    }
    const float4& pixel = image.pixels.f32.a[0u];
    return (pixel.x == 1.0f) && (pixel.y == 1.0f) && (pixel.z == 1.0f) && (pixel.w == 1.0f);
  };

  auto ensure_asset_directory = [&]() {
    std::error_code error;
    std::filesystem::create_directories(asset_directory, error);
    if (error) {
      log::error("Failed to create scene asset directory: %s", path_to_utf8(asset_directory).c_str());
      return false;
    }
    return true;
  };

  auto install_staged_asset = [&](const std::filesystem::path& staged, const std::filesystem::path& destination) {
    std::error_code error;
    std::filesystem::rename(staged, destination, error);
    if (error) {
      log::error("Failed to install scene asset: %s", path_to_utf8(destination).c_str());
      (void)remove_scene_save_file(staged, "staged scene asset");
      return false;
    }
    return true;
  };

  auto publish_staged_asset = [&](const std::filesystem::path& staged, const std::string& file_name) -> std::string {
    const std::filesystem::path destination = asset_directory / file_name;
    bool destination_exists = false;
    if (inspect_scene_asset_path(destination, destination_exists) == false) {
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }
    if (destination_exists) {
      const bool matches = binary_files_match(destination, staged);
      (void)remove_scene_save_file(staged, "staged scene asset");
      if (matches == false) {
        log::error("Existing scene asset does not match its content hash: %s", path_to_utf8(destination).c_str());
        return {};
      }
      return to_relative(destination, materials_path.parent_path());
    }
    if (install_staged_asset(staged, destination) == false) {
      return {};
    }
    return to_relative(destination, materials_path.parent_path());
  };

  auto managed_source_path = [&](const std::filesystem::path& source) {
    return impl->data.owns_assets || (impl->data.owned_asset_paths.count(path_to_utf8(source)) > 0u) || managed_scene_asset_path(source);
  };

  auto persist_managed_image = [&](uint32_t image_index) -> std::string {
    if (ensure_asset_directory() == false) {
      return {};
    }

    const Image& image = impl->data.images.get(image_index);
    const uint64_t pixel_count = 1ull * image.isize.x * image.isize.y;
    bool pixel_data_available = false;
    if (image.format == Image::Format::RGBA32F) {
      pixel_data_available = (image.pixels.f32.a != nullptr) && (pixel_count <= image.pixels.f32.count);
    } else if (image.format == Image::Format::RGBA8) {
      pixel_data_available = (image.pixels.u8.a != nullptr) && (pixel_count <= image.pixels.u8.count);
    } else if (image.format == Image::Format::R32F) {
      pixel_data_available = (image.pixels.r32.a != nullptr) && (pixel_count <= image.pixels.r32.count);
    } else if (Image::is_compressed_bc_format(image.format)) {
      pixel_data_available = (image.pixels.compressed.a != nullptr) && (image.pixels.compressed.count > 0u);
    }
    if ((image.isize.x == 0u) || (image.isize.y == 0u) || (image.isize.z != 1u) || (pixel_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) ||
        (pixel_data_available == false)) {
      log::error("Cannot persist managed image %u: unsupported or incomplete pixel data", image_index);
      return {};
    }

    std::vector<float4> converted_pixels;
    const float4* pixels = image.format == Image::Format::RGBA32F ? image.pixels.f32.a : nullptr;
    if (pixels == nullptr) {
      converted_pixels.resize(static_cast<size_t>(pixel_count));
      pixels = converted_pixels.data();
    }
    for (uint32_t pixel_index = 0u; pixel_index < static_cast<uint32_t>(pixel_count); ++pixel_index) {
      const float4 pixel = image.pixel(pixel_index);
      if ((std::isfinite(pixel.x) == false) || (std::isfinite(pixel.y) == false) || (std::isfinite(pixel.z) == false) || (std::isfinite(pixel.w) == false) || (pixel.x < 0.0f) ||
          (pixel.y < 0.0f) || (pixel.z < 0.0f) || (pixel.w < 0.0f)) {
        log::error("Cannot persist managed image %u: EXR requires finite non-negative pixels", image_index);
        return {};
      }
      if (converted_pixels.empty() == false) {
        converted_pixels[pixel_index] = pixel;
      }
    }

    const std::filesystem::path staged = asset_directory / "image-save-staged.exr";
    bool staged_exists = false;
    if (inspect_scene_asset_path(staged, staged_exists) == false) {
      return {};
    }
    if (staged_exists && (remove_scene_save_file(staged, "staged scene asset") == false)) {
      return {};
    }

    std::string encode_error;
    if (save_exr_image(path_to_utf8(staged).c_str(), pixels, {image.isize.x, image.isize.y}, &encode_error) == false) {
      log::error("Failed to encode managed scene image %u: %s", image_index, encode_error.c_str());
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }

    uint64_t hash = 0u;
    uint64_t encoded_size = 0u;
    if (hash_binary_file(staged, hash, encoded_size) == false) {
      log::error("Failed to hash managed scene image %u", image_index);
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }
    return publish_staged_asset(staged, "image-" + content_hash_string(hash) + ".exr");
  };

  auto persist_managed_file = [&](const std::filesystem::path& source, const char* prefix, const char* extension) -> std::string {
    if (ensure_asset_directory() == false) {
      return {};
    }

    const std::filesystem::path staged = asset_directory / (std::string(prefix) + "save-staged");
    bool staged_exists = false;
    if (inspect_scene_asset_path(staged, staged_exists) == false) {
      return {};
    }
    if (staged_exists && (remove_scene_save_file(staged, "staged scene asset") == false)) {
      return {};
    }

    std::ifstream input(source, std::ios::binary);
    std::ofstream output(staged, std::ios::binary | std::ios::trunc);
    if ((input.is_open() == false) || (output.is_open() == false)) {
      log::error("Failed to open scene dependency for persistence: %s", path_to_utf8(source).c_str());
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }

    std::array<uint8_t, 64u * 1024u> buffer = {};
    uint64_t hash = 0u;
    uint64_t total_size = 0u;
    while (true) {
      input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
      const std::streamsize bytes_read = input.gcount();
      if (bytes_read > 0) {
        output.write(reinterpret_cast<const char*>(buffer.data()), bytes_read);
        if (output.good() == false) {
          break;
        }
        hash = etx_hash64_continue(buffer.data(), static_cast<uint64_t>(bytes_read), hash);
        total_size += static_cast<uint64_t>(bytes_read);
      }
      if (bytes_read < static_cast<std::streamsize>(buffer.size())) {
        break;
      }
    }
    output.flush();
    const bool streams_succeeded = (input.bad() == false) && output.good() && (total_size > 0u);
    input.close();
    output.close();
    if ((streams_succeeded == false) || output.fail()) {
      log::error("Failed to persist scene dependency: %s", path_to_utf8(source).c_str());
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }

    hash = etx_hash64_continue(&total_size, sizeof(total_size), hash);
    const std::filesystem::path destination = asset_directory / (std::string(prefix) + content_hash_string(hash) + extension);
    bool destination_exists = false;
    if (inspect_scene_asset_path(destination, destination_exists) == false) {
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }
    if (destination_exists) {
      const bool matches = binary_files_match(staged, destination);
      (void)remove_scene_save_file(staged, "staged scene asset");
      if (matches == false) {
        log::error("Existing scene asset does not match its content hash: %s", path_to_utf8(destination).c_str());
        return {};
      }
      return to_relative(destination, materials_path.parent_path());
    }
    if (install_staged_asset(staged, destination) == false) {
      return {};
    }
    return to_relative(destination, materials_path.parent_path());
  };

  auto texture_path = [&](uint32_t image_index, bool omit_white_fallback) -> std::string {
    if ((image_index == kInvalidIndex) || (image_index >= impl->data.images.array_size())) {
      return {};
    }
    const std::string stored = impl->data.images.path(image_index);
    if ((stored.compare(0, 2, "##") == 0) || stored.empty()) {
      if (omit_white_fallback && is_white_fallback_image(image_index)) {
        return {};
      }
      return persist_managed_image(image_index);
    }
    const std::filesystem::path source = std::filesystem::u8path(stored).lexically_normal();
    if (managed_source_path(source)) {
      return persist_managed_file(source, "image-", source.extension().string().c_str());
    }
    return to_relative(source, materials_path.parent_path());
  };

  auto write_path_token = [&](std::ostringstream& stream, const std::string& path, const char* context) {
    for (const unsigned char character : path) {
      if (character < 0x20u) {
        log::error("Cannot save %s path containing control characters", context);
        return false;
      }
    }
    stream << std::quoted(path);
    return true;
  };

  auto write_texture_line = [&](std::ostringstream& stream, const char* label, uint32_t image_index, uint32_t channel) {
    if (image_index == kInvalidIndex) {
      return true;
    }
    const std::string path = texture_path(image_index, false);
    if (path.empty()) {
      log::error("Cannot save %s texture: its source file is unavailable", label);
      return false;
    }
    stream << label << " ";
    if (write_path_token(stream, path, label) == false) {
      return false;
    }
    if (channel != kInvalidIndex) {
      stream << " channel " << channel;
    }
    stream << "\n";
    return true;
  };

  auto write_spectrum_line = [&](std::ostringstream& stream, const char* label, uint32_t index, bool use_gamma) {
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return;
    }
    float3 value = spectrum_rgb(index);
    if (use_gamma) {
      value = linear_to_gamma(value);
    }
    stream << label << " " << value.x << " " << value.y << " " << value.z << "\n";
  };

  std::ostringstream materials_stream;
  materials_stream << std::setprecision(std::numeric_limits<float>::max_digits10);

  const IORDatabase& database = impl->ior_database;
  std::vector<std::pair<uint32_t, std::filesystem::path>> persisted_volume_paths;

  for (const SerializedMediumEntry& entry : serialized_medium_entries) {
    const uint32_t pool_index = entry.medium_index;
    const Medium& medium = impl->data.mediums.get(pool_index);
    materials_stream << "newmtl et::medium\n";
    materials_stream << "id " << entry.id << "\n";
    float3 absorption = impl->data.spectrum_values[medium.absorption_index].integrated();
    if ((std::fabs(absorption.x) >= kEpsilon) || (std::fabs(absorption.y) >= kEpsilon) || (std::fabs(absorption.z) >= kEpsilon)) {
      materials_stream << "absorption " << absorption.x << " " << absorption.y << " " << absorption.z << "\n";
    }
    float3 scattering = impl->data.spectrum_values[medium.scattering_index].integrated();
    if ((std::fabs(scattering.x) >= kEpsilon) || (std::fabs(scattering.y) >= kEpsilon) || (std::fabs(scattering.z) >= kEpsilon)) {
      materials_stream << "scattering " << scattering.x << " " << scattering.y << " " << scattering.z << "\n";
    }
    if (medium.emission_index != kInvalidIndex) {
      const float3 emission = impl->data.spectrum_values[medium.emission_index].integrated();
      if ((emission.x > 0.0f) || (emission.y > 0.0f) || (emission.z > 0.0f)) {
        materials_stream << "emission " << emission.x << " " << emission.y << " " << emission.z << "\n";
      }
    }
    if (std::fabs(medium.phase_function_g) >= kEpsilon) {
      materials_stream << "anisotropy " << medium.phase_function_g << "\n";
    }
    if (medium.enable_explicit_connections == false) {
      materials_stream << "enclosed 1\n";
    }
    const std::string& source_volume_path = impl->data.mediums.volume_path(pool_index);
    if (source_volume_path.empty() == false) {
      const std::filesystem::path volume_source = std::filesystem::u8path(source_volume_path).lexically_normal();
      const bool persist_volume = managed_source_path(volume_source);
      const std::string saved_volume_path = persist_volume ? persist_managed_file(volume_source, "volume-", ".nvdb") : to_relative(volume_source, materials_path.parent_path());
      if (saved_volume_path.empty()) {
        log::error("Cannot save volume dependency: %s", source_volume_path.c_str());
        return {};
      }
      if (persist_volume) {
        std::filesystem::path persisted_path = std::filesystem::u8path(saved_volume_path);
        if (persisted_path.is_relative()) {
          persisted_path = materials_path.parent_path() / persisted_path;
        }
        persisted_volume_paths.emplace_back(pool_index, persisted_path.lexically_normal());
      }
      materials_stream << "volume ";
      if (write_path_token(materials_stream, saved_volume_path, "volume") == false) {
        return {};
      }
      materials_stream << "\n";
    } else if ((medium.cls == Medium::Heterogeneous) && (medium.grid_type_enum() == DensityGrid::Type::Texture3D)) {
      log::error("Cannot save file-backed medium %s: its source volume path is unavailable", entry.authored_name.c_str());
      return {};
    } else if ((medium.cls == Medium::Heterogeneous) && (medium.grid_type_enum() == DensityGrid::Type::NoiseFunction)) {
      materials_stream << "noise type " << static_cast<uint32_t>(medium.noise_type_enum()) << " scale " << medium.grid.noise_scale << " octaves " << medium.grid.noise_octaves
                       << " lacunarity " << medium.grid.noise_lacunarity << " persistence " << medium.grid.noise_persistence << " seed " << medium.grid.noise_seed << " power "
                       << medium.grid.noise_power << " sharpness " << medium.grid.noise_sharpness << " offset " << medium.grid.noise_offset.x << " " << medium.grid.noise_offset.y
                       << " " << medium.grid.noise_offset.z << " border_fade " << medium.grid.noise_enable_border_fade << " border_fade_distance "
                       << medium.grid.noise_border_fade_distance << "\n";
    }
    materials_stream << "\n";
  }

  const auto write_camera = [&](const Camera& camera, const std::string& camera_id, bool active) {
    if ((camera.film_size.x == 0u) || (camera.film_size.y == 0u)) {
      if (camera_id.empty()) {
        return true;
      }
      log::error("Cannot save camera %s: its viewport is invalid", camera_id.c_str());
      return false;
    }
    const float3 target = camera.position + camera.direction;
    materials_stream << "newmtl et::camera\n";
    materials_stream << "class " << ((camera.cls == Camera::Class::Equirectangular) ? "eq" : "perspective") << "\n";
    materials_stream << "viewport " << camera.film_size.x << " " << camera.film_size.y << "\n";
    materials_stream << "origin " << camera.position.x << " " << camera.position.y << " " << camera.position.z << "\n";
    materials_stream << "target " << target.x << " " << target.y << " " << target.z << "\n";
    materials_stream << "up " << camera.up.x << " " << camera.up.y << " " << camera.up.z << "\n";
    materials_stream << "fov " << get_camera_fov(camera) << "\n";
    const float fov_from_focal = focal_length_to_fov(get_camera_focal_length(camera)) * 180.0f / kPi;
    if (std::fabs(fov_from_focal - get_camera_fov(camera)) > 0.01f) {
      materials_stream << "focal-length " << get_camera_focal_length(camera) << "\n";
    }
    if (camera.lens_radius > 0.0f) {
      materials_stream << "lens-radius " << camera.lens_radius << "\n";
    }
    if (camera.focal_distance > 0.0f) {
      materials_stream << "focal-distance " << camera.focal_distance << "\n";
    }
    materials_stream << "clip-near " << camera.clip_near << "\n";
    materials_stream << "clip-far " << camera.clip_far << "\n";
    const std::string lens_shape = texture_path(camera.lens_image, false);
    if (lens_shape.empty() == false) {
      materials_stream << "shape ";
      if (write_path_token(materials_stream, lens_shape, "camera lens") == false) {
        return false;
      }
      materials_stream << "\n";
    } else if (camera.lens_image != kInvalidIndex) {
      log::error("Cannot save camera %s: its lens image source file is unavailable", camera_id.empty() ? "camera" : camera_id.c_str());
      return false;
    }
    const bool camera_medium_valid = (camera.medium_index != kInvalidIndex) && (medium_names.count(camera.medium_index) > 0);
    if (camera_medium_valid) {
      materials_stream << "ext_medium " << medium_names[camera.medium_index] << "\n";
    }
    if (camera_id.empty() == false) {
      materials_stream << "id " << camera_id << "\n";
    }
    materials_stream << "active " << (active ? 1 : 0) << "\n";
    materials_stream << "\n";
    return true;
  };

  {
    for (const SerializedCameraEntry& serialized_entry : serialized_camera_entries) {
      const SceneData::CameraInfo& camera_entry = impl->data.cameras[serialized_entry.camera_index];
      const Camera* camera_to_save = &camera_entry.cam;
      if (camera_entry.active) {
        AttachmentTransform transform = {};
        if (find_attachment_transform(impl->data, SceneAttachment::Type::Camera, serialized_entry.camera_index, transform) == false) {
          camera_to_save = &impl->active_camera;
        }
      }
      if (write_camera(*camera_to_save, serialized_entry.id, camera_entry.active) == false) {
        return {};
      }
    }
  }

  std::vector<uint32_t> atmosphere_emitter_indices;

  for (uint32_t i = 0; i < impl->data.emitter_profiles.size(); ++i) {
    const auto& profile = impl->data.emitter_profiles[i];
    if ((profile.meta & EmitterProfile::Meta::Atmosphere) && (profile.cls == EmitterProfile::Class::Environment)) {
      atmosphere_emitter_indices.push_back(i);
    }
  }

  for (uint32_t emitter_index : atmosphere_emitter_indices) {
    const auto& env_profile = impl->data.emitter_profiles[emitter_index];
    float3 env_color = spectrum_rgb(env_profile.emission.spectrum_index);
    const auto& scattering = env_profile.atmosphere.scattering;
    materials_stream << "newmtl et::atmosphere\n";
    materials_stream << "anisotropy " << scattering.anisotropy << "\n";
    materials_stream << "altitude " << scattering.altitude << "\n";
    materials_stream << "rayleigh " << scattering.rayleigh_scale << "\n";
    materials_stream << "mie " << scattering.mie_scale << "\n";
    materials_stream << "ozone " << scattering.ozone_scale << "\n";
    if (scattering.primary_scattering == 0u) {
      materials_stream << "primary-scattering 0\n";
    }
    if (scattering.secondary_scattering == 0u) {
      materials_stream << "secondary-scattering 0\n";
    }
    materials_stream << "quality " << env_profile.atmosphere.quality << "\n";
    materials_stream << "color " << env_color.x << " " << env_color.y << " " << env_color.z << "\n";
    const bool atmosphere_medium_valid = (env_profile.medium_index != kInvalidIndex) && (medium_names.count(env_profile.medium_index) > 0u);
    if (atmosphere_medium_valid) {
      materials_stream << "ext_medium " << medium_names[env_profile.medium_index] << "\n";
    }
    materials_stream << "\n";
  }

  for (uint32_t i = 0; i < impl->data.emitter_profiles.size(); ++i) {
    const auto& profile = impl->data.emitter_profiles[i];
    if (profile.cls != EmitterProfile::Class::Environment) {
      continue;
    }
    if (profile.meta & EmitterProfile::Meta::Atmosphere) {
      continue;
    }

    materials_stream << "newmtl et::env\n";
    std::string env_path = texture_path(profile.emission.image_index, true);
    if (env_path.empty() == false) {
      materials_stream << "image ";
      if (write_path_token(materials_stream, env_path, "environment image") == false) {
        return {};
      }
      materials_stream << "\n";
    } else if (is_white_fallback_image(profile.emission.image_index) == false) {
      log::error("Cannot save environment light: its image source file is unavailable");
      return {};
    }
    float3 env_color = spectrum_rgb(profile.emission.spectrum_index);
    materials_stream << "color " << env_color.x << " " << env_color.y << " " << env_color.z << "\n";
    float env_rotation_offset = 0.0f;
    float env_scale_u = 1.0f;
    if (profile.emission.image_index != kInvalidIndex) {
      const Image& env_image = impl->data.images.get(profile.emission.image_index);
      env_rotation_offset = env_image.offset.x;
      env_scale_u = env_image.scale.x;
    }
    if (std::fabs(env_rotation_offset) >= kEpsilon) {
      materials_stream << "rotation " << (-env_rotation_offset * 360.0f) << "\n";
    }
    if (std::fabs(env_scale_u - 1.0f) >= kEpsilon) {
      materials_stream << "scale " << env_scale_u << "\n";
    }
    bool env_medium_valid = (profile.medium_index != kInvalidIndex) && (medium_names.count(profile.medium_index) > 0);
    if (env_medium_valid) {
      materials_stream << "ext_medium " << medium_names[profile.medium_index] << "\n";
    }
    materials_stream << "\n";
  }

  for (uint32_t i = 0; i < impl->data.emitter_profiles.size(); ++i) {
    const auto& profile = impl->data.emitter_profiles[i];
    if (profile.cls != EmitterProfile::Class::Directional) {
      continue;
    }

    materials_stream << "newmtl et::dir\n";
    float3 dir_color = spectrum_rgb(profile.emission.spectrum_index);
    materials_stream << "color " << dir_color.x << " " << dir_color.y << " " << dir_color.z << "\n";
    materials_stream << "direction " << profile.directional.direction.x << " " << profile.directional.direction.y << " " << profile.directional.direction.z << "\n";
    if (profile.directional.angular_size >= kEpsilon) {
      materials_stream << "angular_diameter " << (profile.directional.angular_size * 180.0f / kPi) << "\n";
    }
    const bool references_atmosphere = (profile.reference_emitter_index != kInvalidIndex) && (profile.reference_emitter_index < impl->data.emitter_profiles.size()) &&
                                       (impl->data.emitter_profiles[profile.reference_emitter_index].cls == EmitterProfile::Class::Environment) &&
                                       ((impl->data.emitter_profiles[profile.reference_emitter_index].meta & EmitterProfile::Meta::Atmosphere) != 0u);
    if (references_atmosphere) {
      materials_stream << "use_as_sun 1\n";
      const auto atmosphere_position = std::find(atmosphere_emitter_indices.begin(), atmosphere_emitter_indices.end(), profile.reference_emitter_index);
      if (atmosphere_position != atmosphere_emitter_indices.end()) {
        materials_stream << "atmosphere_index " << std::distance(atmosphere_emitter_indices.begin(), atmosphere_position) << "\n";
      }
    }
    std::string dir_path;
    if (references_atmosphere == false) {
      dir_path = texture_path(profile.emission.image_index, false);
    }
    if (dir_path.empty() == false) {
      materials_stream << "image ";
      if (write_path_token(materials_stream, dir_path, "directional image") == false) {
        return {};
      }
      materials_stream << "\n";
    } else if ((profile.emission.image_index != kInvalidIndex) && (references_atmosphere == false)) {
      log::error("Cannot save directional light: its image source file is unavailable");
      return {};
    }
    bool dir_medium_valid = (profile.medium_index != kInvalidIndex) && (medium_names.count(profile.medium_index) > 0);
    if (dir_medium_valid) {
      materials_stream << "ext_medium " << medium_names[profile.medium_index] << "\n";
    }
    materials_stream << "\n";
  }

  for (const SerializedMaterialEntry& entry : serialized_material_entries) {
    const std::string& serialized_name = entry.id;
    const std::string& display_name = entry.authored_names.front();
    const uint32_t index = entry.material_index;
    const Material& material = impl->data.materials[index];

    materials_stream << "newmtl " << serialized_name << "\n";
    materials_stream << "material class " << material_class_to_string(material.cls) << "\n";
    if (material.temperature_kelvin > 0.0f) {
      materials_stream << "temperature " << material.temperature_kelvin << "\n";
    }

    write_spectrum_line(materials_stream, "Kd", material.scattering.spectrum_index, true);
    if ((material.cls == MaterialClass::Dielectric) || (material.cls == MaterialClass::Translucent) || (material.transmission.value.x > kEpsilon)) {
      write_spectrum_line(materials_stream, "Kt", material.scattering.spectrum_index, true);
    }
    write_spectrum_line(materials_stream, "Ks", material.reflectance.spectrum_index, true);

    float rough_u = material.roughness.value.x;
    float rough_v = material.roughness.value.y;
    if ((rough_u >= kEpsilon) || (rough_v >= kEpsilon)) {
      float value_u = std::sqrt(max(0.0f, rough_u));
      float value_v = std::sqrt(max(0.0f, rough_v));
      if (std::fabs(value_u - value_v) < kEpsilon) {
        materials_stream << "Pr " << value_u << "\n";
      } else {
        materials_stream << "Pr " << value_u << " " << value_v << "\n";
      }
    }

    if (material.metalness.value.x >= kEpsilon) {
      materials_stream << "metalness " << material.metalness.value.x << "\n";
    }
    if (material.transmission.value.x >= kEpsilon) {
      materials_stream << "transmission " << material.transmission.value.x << "\n";
    }

    materials_stream << "bump_strength " << material.bump.value.x << "\n";
    if (write_texture_line(materials_stream, "map_bump", material.bump.image_index, material.bump.channel) == false)
      return {};

    materials_stream << "alpha_mask " << material.alpha_mask.value.x << "\n";
    if ((write_texture_line(materials_stream, "map_d", material.alpha_mask.image_index, material.alpha_mask.channel) == false) ||
        (write_texture_line(materials_stream, "map_Kd", material.scattering.image_index, kInvalidIndex) == false) ||
        (write_texture_line(materials_stream, "map_Ks", material.reflectance.image_index, kInvalidIndex) == false) ||
        (write_texture_line(materials_stream, "map_Kt", material.scattering.image_index, kInvalidIndex) == false) ||
        (write_texture_line(materials_stream, "map_Pr", material.roughness.image_index, material.roughness.channel) == false) ||
        (write_texture_line(materials_stream, "map_Ml", material.metalness.image_index, material.metalness.channel) == false) ||
        (write_texture_line(materials_stream, "map_Tm", material.transmission.image_index, material.transmission.channel) == false)) {
      return {};
    }

    if ((material.normal_image_index != kInvalidIndex) || (std::fabs(material.normal_scale - 1.0f) >= kEpsilon)) {
      std::string normal_path = texture_path(material.normal_image_index, false);
      materials_stream << "normalmap";
      if (normal_path.empty() == false) {
        materials_stream << " image ";
        if (write_path_token(materials_stream, normal_path, "normal map") == false) {
          return {};
        }
      } else if (material.normal_image_index != kInvalidIndex) {
        log::error("Cannot save material %s: its normal map source file is unavailable", display_name.c_str());
        return {};
      }
      materials_stream << " scale " << material.normal_scale << "\n";
    }

    int matched_int_index = -1;
    const auto internal_source = impl->data.spectrum_sources.find(material.int_ior.eta_index);
    const bool temperature_profile = (internal_source != impl->data.spectrum_sources.end()) && (internal_source->second.temperature_profile != nullptr);
    if ((material.int_ior.cls != SpectralDistribution::Invalid) && (temperature_profile == false)) {
      matched_int_index = database.find_matching_index(spectrum_by_index(material.int_ior.eta_index), spectrum_by_index(material.int_ior.k_index), material.int_ior.cls);
    }
    if ((matched_int_index >= 0) && (matched_int_index < static_cast<int>(database.definitions.size()))) {
      const IORDefinition& def = database.definitions[static_cast<size_t>(matched_int_index)];
      materials_stream << "int_ior " << def.name << "\n";
    } else if ((material.int_ior.eta_index != kInvalidIndex) && (material.int_ior.cls != SpectralDistribution::Invalid)) {
      float eta_value = spectrum_scalar(material.int_ior.eta_index, 1.0f);
      if (material.int_ior.cls == SpectralDistribution::Dielectric) {
        materials_stream << "int_ior " << eta_value << "\n";
      } else if (material.int_ior.cls == SpectralDistribution::Conductor) {
        float k_value = spectrum_scalar(material.int_ior.k_index, 0.0f);
        materials_stream << "int_ior " << eta_value << " " << k_value << "\n";
      }
    }

    int matched_ext_index = -1;
    if (material.ext_ior.cls != SpectralDistribution::Invalid) {
      matched_ext_index = database.find_matching_index(spectrum_by_index(material.ext_ior.eta_index), spectrum_by_index(material.ext_ior.k_index), material.ext_ior.cls);
    }
    if ((matched_ext_index >= 0) && (matched_ext_index < static_cast<int>(database.definitions.size()))) {
      const IORDefinition& def = database.definitions[static_cast<size_t>(matched_ext_index)];
      materials_stream << "ext_ior " << def.name << "\n";
    } else {
      float ext_eta_value = spectrum_scalar(material.ext_ior.eta_index, 1.0f);
      if ((material.ext_ior.eta_index != kInvalidIndex) && (material.ext_ior.cls != SpectralDistribution::Invalid) &&
          (material.ext_ior.cls != SpectralDistribution::Dielectric || std::fabs(ext_eta_value - 1.0f) >= kEpsilon)) {
        if (material.ext_ior.cls == SpectralDistribution::Dielectric) {
          materials_stream << "ext_ior " << ext_eta_value << "\n";
        } else if (material.ext_ior.cls == SpectralDistribution::Conductor) {
          float ext_k_value = spectrum_scalar(material.ext_ior.k_index, 0.0f);
          materials_stream << "ext_ior " << ext_eta_value << " " << ext_k_value << "\n";
        }
      } else {
        materials_stream << "ext_ior 1.0\n";
      }
    }

    if (medium_names.count(material.int_medium) > 0u) {
      materials_stream << "int_medium " << medium_names[material.int_medium] << "\n";
    }
    if (medium_names.count(material.ext_medium) > 0u) {
      materials_stream << "ext_medium " << medium_names[material.ext_medium] << "\n";
    }

    if (material.two_sided != 0u) {
      materials_stream << "two_sided 1\n";
    }
    if (std::fabs(material.opacity - 1.0f) >= kEpsilon) {
      materials_stream << "opacity " << material.opacity << "\n";
    }

    bool has_emission_texture = (material.emission.image_index != kInvalidIndex);
    bool has_emission_spectrum = (material.emission.spectrum_index != kInvalidIndex) && (material.emission.spectrum_index < impl->data.spectrum_values.size());
    if (has_emission_texture || has_emission_spectrum) {
      materials_stream << "emitter";
      if (has_emission_texture) {
        std::string emission_path = texture_path(material.emission.image_index, false);
        if (emission_path.empty() == false) {
          materials_stream << " image ";
          if (write_path_token(materials_stream, emission_path, "emission texture") == false) {
            return {};
          }
        } else {
          log::error("Cannot save material %s: its emission texture source file is unavailable", display_name.c_str());
          return {};
        }
      }
      if (has_emission_spectrum) {
        float3 emission_value = spectrum_rgb(material.emission.spectrum_index);
        materials_stream << " color " << emission_value.x << " " << emission_value.y << " " << emission_value.z;
      }
      if (material.two_sided != 0u) {
        materials_stream << " twosided";
      }
      if (material.emission_collimation >= kEpsilon) {
        materials_stream << " collimated " << material.emission_collimation;
      }
      materials_stream << "\n";
    }

    if (material.subsurface_cls != SubsurfaceMaterial::Disabled) {
      materials_stream << "subsurface";
      if (material.subsurface.image_index != kInvalidIndex) {
        const std::string subsurface_path = texture_path(material.subsurface.image_index, false);
        if (subsurface_path.empty()) {
          log::error("Cannot save material %s: its subsurface texture has no source file", display_name.c_str());
          return {};
        }
        materials_stream << " image ";
        if (write_path_token(materials_stream, subsurface_path, "subsurface texture") == false) {
          return {};
        }
      }
      if (material.subsurface_path == SubsurfaceMaterial::RefractedPath) {
        materials_stream << " path refracted";
      }
      float3 subsurface_color = spectrum_rgb(material.subsurface.spectrum_index);
      materials_stream << " distances " << subsurface_color.x << " " << subsurface_color.y << " " << subsurface_color.z;
      materials_stream << " packing " << material.subsurface_packing << " anisotropy " << material.subsurface_anisotropy;
      materials_stream << "\n";
    }

    if ((material.thinfilm.thinkness_image != kInvalidIndex) || (std::fabs(material.thinfilm.min_thickness) >= kEpsilon) ||
        (std::fabs(material.thinfilm.max_thickness) >= kEpsilon) || (std::fabs(material.thinfilm.weight - 1.0f) >= kEpsilon)) {
      materials_stream << "thinfilm";
      std::string thinfilm_path = texture_path(material.thinfilm.thinkness_image, false);
      if (thinfilm_path.empty() == false) {
        materials_stream << " image ";
        if (write_path_token(materials_stream, thinfilm_path, "thin-film texture") == false) {
          return {};
        }
      } else if (material.thinfilm.thinkness_image != kInvalidIndex) {
        log::error("Cannot save material %s: its thin-film texture source file is unavailable", display_name.c_str());
        return {};
      }
      materials_stream << " range " << material.thinfilm.min_thickness << " " << material.thinfilm.max_thickness;
      materials_stream << " weight " << clamp(material.thinfilm.weight, 0.0f, 1.0f);
      int matched_thinfilm_index = -1;
      if (material.thinfilm.ior.cls != SpectralDistribution::Invalid) {
        matched_thinfilm_index =
          database.find_matching_index(spectrum_by_index(material.thinfilm.ior.eta_index), spectrum_by_index(material.thinfilm.ior.k_index), material.thinfilm.ior.cls);
      }
      if ((matched_thinfilm_index >= 0) && (matched_thinfilm_index < static_cast<int>(database.definitions.size()))) {
        const IORDefinition& def = database.definitions[static_cast<size_t>(matched_thinfilm_index)];
        materials_stream << " ior " << def.name << "\n";
      } else {
        float thinfilm_eta = spectrum_scalar(material.thinfilm.ior.eta_index, 1.0f);
        materials_stream << " ior " << thinfilm_eta << "\n";
      }
    }

    if (material.cls == MaterialClass::DiffractionGrating) {
      materials_stream << "diffraction_grating period_nm " << material.diffraction_grating.period_nm;
      materials_stream << " optical_path_difference_nm " << material.diffraction_grating.optical_path_difference_nm;
      materials_stream << " duty_cycle " << material.diffraction_grating.duty_cycle;
      materials_stream << " rotation_degrees " << material.diffraction_grating.rotation * 180.0f / kPi;
      materials_stream << "\n";
    }

    materials_stream << "\n";
  }

  const std::string materials_string = materials_stream.str();

  std::string json_string;
  try {
    json_string = js.dump(2);
  } catch (const nlohmann::json::exception& error) {
    log::error("Failed to serialize scene config: %s", error.what());
    return {};
  }

  const auto geometry_export_start = std::chrono::high_resolution_clock::now();
  SceneSerialization archive;
  if (archive.save_to_file(impl->data, scene_files[0].staged, serialized_material_names) == false) {
    log::error("Failed to stage geometry for %s", path_to_utf8(geometry_path).c_str());
    discard_staged_scene_files(scene_files);
    return {};
  }
  const auto geometry_export_end = std::chrono::high_resolution_clock::now();
  const auto geometry_export_duration = std::chrono::duration_cast<std::chrono::milliseconds>(geometry_export_end - geometry_export_start);
  log::info("Geometry export: %lld ms", geometry_export_duration.count());

  const auto materials_write_start = std::chrono::high_resolution_clock::now();
  if (write_file_contents(scene_files[1].staged, materials_string) == false) {
    discard_staged_scene_files(scene_files);
    return {};
  }
  const auto materials_write_end = std::chrono::high_resolution_clock::now();
  const auto materials_write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(materials_write_end - materials_write_start);
  log::info("Materials file write: %lld ms (%zu bytes)", materials_write_duration.count(), materials_string.size());

  const auto json_write_start = std::chrono::high_resolution_clock::now();
  if (write_file_contents(scene_files[2].staged, json_string) == false) {
    discard_staged_scene_files(scene_files);
    return {};
  }
  const auto json_write_end = std::chrono::high_resolution_clock::now();
  const auto json_write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(json_write_end - json_write_start);
  log::info("JSON config write: %lld ms", json_write_duration.count());

  if (commit_staged_scene_files(scene_files) == false) {
    return {};
  }

  for (const auto& [medium_index, volume_path] : persisted_volume_paths) {
    impl->data.mediums.set_volume_path(medium_index, path_to_utf8(volume_path));
  }
  impl->data.geometry_file_name = path_to_utf8(geometry_path);
  impl->data.json_file_name = path_to_utf8(json_path);
  impl->data.materials_file_name = path_to_utf8(materials_path);

  auto save_end = std::chrono::high_resolution_clock::now();
  auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(save_end - save_start);
  log::info("Scene save total: %lld ms", save_duration.count());

  return path_to_utf8(json_path);
}
