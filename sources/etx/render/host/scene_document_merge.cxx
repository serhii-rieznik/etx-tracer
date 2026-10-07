#include <etx/render/host/scene_document_merge.hxx>
#include <stdexcept>
#include <unordered_set>

namespace etx {

bool append_scene_document(SceneData& destination, const SceneData& source, std::string& error) {
  try {
    const bool first = destination.spectrum_values.empty();
    auto offset = [](size_t base, size_t count) {
      if ((base + count) >= kInvalidIndex)
        throw std::runtime_error("Merged document exceeds the resource index limit.");
      return static_cast<uint32_t>(base);
    };
    const uint32_t spectrum_base = offset(destination.spectrum_values.size(), source.spectrum_values.size());
    const uint32_t material_base = offset(destination.materials.size(), source.materials.size());
    const uint32_t medium_base = offset(destination.mediums_vector.size(), source.mediums_vector.size());
    const uint32_t image_base = offset(destination.images_vector.size(), source.images_vector.size());
    const uint32_t emitter_base = offset(destination.emitter_profiles.size(), source.emitter_profiles.size());
    const uint32_t vertex_base = offset(destination.vertices.pos.size(), source.vertices.pos.size());
    const uint32_t triangle_base = offset(destination.triangles.size(), source.triangles.size());
    const uint32_t mesh_base = offset(destination.meshes.size(), source.meshes.size());
    const uint32_t camera_base = offset(destination.cameras.size(), source.cameras.size());
    const uint32_t node_base = offset(destination.hierarchy.nodes.size(), source.hierarchy.nodes.size());
    auto remap = [](uint32_t value, uint32_t base, size_t count) {
      if (value == kInvalidIndex)
        return value;
      if (value >= count)
        throw std::runtime_error("Source document contains an invalid resource reference.");
      return base + value;
    };
    auto spectrum = [&](uint32_t value) {
      return remap(value, spectrum_base, source.spectrum_values.size());
    };
    auto image = [&](uint32_t value) {
      return remap(value, image_base, source.images_vector.size());
    };
    auto medium = [&](uint32_t value) {
      return remap(value, medium_base, source.mediums_vector.size());
    };
    auto emitter = [&](uint32_t value) {
      return remap(value, emitter_base, source.emitter_profiles.size());
    };
    destination.spectrum_values.insert(destination.spectrum_values.end(), source.spectrum_values.begin(), source.spectrum_values.end());
    destination.spectrum_names.resize(destination.spectrum_values.size());
    for (uint32_t index = 0u; index < source.spectrum_names.size(); ++index)
      destination.spectrum_names[spectrum_base + index] = source.spectrum_names[index];
    for (const auto& [index, metadata] : source.spectrum_sources)
      destination.spectrum_sources.emplace(spectrum(index), metadata);
    for (uint32_t index = 0u; index < source.images_vector.size(); ++index) {
      if (destination.images.add_copy(source.images_vector[index], source.images.path(index)) != (image_base + index)) {
        throw std::runtime_error("Failed to copy a document image.");
      }
    }
    auto unique_name = [](const auto& mapping, const std::string& desired) {
      std::string name = desired;
      for (uint32_t suffix = 2u; mapping.count(name) > 0u; ++suffix)
        name = desired + " " + std::to_string(suffix);
      return name;
    };
    std::vector<std::string> medium_names(source.mediums_vector.size());
    for (const auto& [name, index] : source.mediums.mapping())
      medium_names.at(index) = name;
    for (uint32_t index = 0u; index < source.mediums_vector.size(); ++index) {
      const auto& original = source.mediums_vector[index];
      const std::string name = unique_name(destination.mediums.mapping(), medium_names[index]);
      const uint32_t added = destination.mediums.add(Medium::Homogeneous, name, "", spectrum(original.absorption_index), spectrum(original.scattering_index),
        original.phase_function_g, original.enable_explicit_connections != 0u);
      if (added != (medium_base + index))
        throw std::runtime_error("Failed to copy a document medium.");
      auto& copy = destination.mediums.get(added);
      copy = original;
      copy.bounds = original.local_bounds;
      copy.absorption_index = spectrum(original.absorption_index);
      copy.scattering_index = spectrum(original.scattering_index);
      copy.emission_index = spectrum(original.emission_index);
      copy.thermal_source_index = spectrum(original.thermal_source_index);
      copy.grid.density_image_index = image(original.grid.density_image_index);
      copy.density_buffer = {};
      copy.density_data = {};
      copy.density_view = {};
      copy.emission_failure = nullptr;
      if (copy.grid.density_image_index != kInvalidIndex) {
        copy.density_view = destination.images.get(copy.grid.density_image_index).pixels.r32;
      } else if (original.density_data.valid()) {
        copy.density_buffer = destination.buffer_pool.create(original.density_data.byte_size, "document density");
        copy.density_data = destination.buffer_pool.allocate(copy.density_buffer, original.density_data.byte_size, alignof(float));
        if (destination.buffer_pool.write(copy.density_data, source.buffer_pool.map(original.density_data), original.density_data.byte_size) == false) {
          throw std::runtime_error("Failed to copy document density data.");
        }
        copy.density_view = destination.buffer_pool.view_as_array<float>(copy.density_data);
      }
      destination.mediums.set_volume_path(added, source.mediums.volume_path(index));
    }
    auto spectral_image = [&](SpectralImage& value) {
      value.spectrum_index = spectrum(value.spectrum_index);
      value.image_index = image(value.image_index);
    };
    auto ior = [&](RefractiveIndex& value) {
      value.eta_index = spectrum(value.eta_index);
      value.k_index = spectrum(value.k_index);
    };
    for (auto material : source.materials) {
      spectral_image(material.reflectance);
      spectral_image(material.scattering);
      spectral_image(material.emission);
      spectral_image(material.subsurface);
      ior(material.ext_ior);
      ior(material.int_ior);
      ior(material.thinfilm.ior);
      material.roughness.image_index = image(material.roughness.image_index);
      material.metalness.image_index = image(material.metalness.image_index);
      material.transmission.image_index = image(material.transmission.image_index);
      material.thinfilm.thinkness_image = image(material.thinfilm.thinkness_image);
      material.normal_image_index = image(material.normal_image_index);
      material.alpha_mask.image_index = image(material.alpha_mask.image_index);
      material.bump.image_index = image(material.bump.image_index);
      material.int_medium = medium(material.int_medium);
      material.ext_medium = medium(material.ext_medium);
      material.energy_compensation_interface_index = kInvalidIndex;
      material.conductor_energy_compensation_interface_index = kInvalidIndex;
      material.thermal_rgb_image_index = kInvalidIndex;
      material.thermal_conductor_image_index = kInvalidIndex;
      material.thermal_energy_compensation_interface_index = kInvalidIndex;
      material.thermal_int_medium_index = kInvalidIndex;
      material.thermal_emission_weight = 0.0f;
      destination.materials.push_back(material);
    }
    for (const auto& [name, index] : source.material_mapping)
      destination.material_mapping.emplace(unique_name(destination.material_mapping, name), remap(index, material_base, source.materials.size()));
    auto append = [](auto& dst, const auto& src) {
      dst.insert(dst.end(), src.begin(), src.end());
    };
    append(destination.vertices.pos, source.vertices.pos);
    append(destination.vertices.nrm, source.vertices.nrm);
    append(destination.vertices.tan, source.vertices.tan);
    append(destination.vertices.btn, source.vertices.btn);
    append(destination.vertices.tex, source.vertices.tex);
    for (auto triangle : source.triangles) {
      for (auto& index : triangle.i)
        index = remap(index, vertex_base, source.vertices.pos.size());
      triangle.material_index = remap(triangle.material_index, material_base, source.materials.size());
      triangle.emitter_index = emitter(triangle.emitter_index);
      destination.triangles.push_back(triangle);
    }
    for (auto mesh : source.meshes) {
      if ((static_cast<uint64_t>(mesh.triangle_offset) + mesh.triangle_count) > source.triangles.size())
        throw std::runtime_error("Invalid mesh triangle range.");
      mesh.triangle_offset += triangle_base;
      destination.meshes.push_back(mesh);
    }
    for (const auto& [name, index] : source.mesh_mapping)
      destination.mesh_mapping.emplace(unique_name(destination.mesh_mapping, name), remap(index, mesh_base, source.meshes.size()));
    std::unordered_set<std::string> emitter_names(destination.emitter_names.begin(), destination.emitter_names.end());
    for (uint32_t index = 0u; index < source.emitter_profiles.size(); ++index) {
      auto profile = source.emitter_profiles[index];
      spectral_image(profile.emission);
      profile.medium_index = medium(profile.medium_index);
      profile.reference_emitter_index = emitter(profile.reference_emitter_index);
      destination.emitter_profiles.push_back(profile);
      const std::string desired = index < source.emitter_names.size() ? source.emitter_names[index] : "Emitter " + std::to_string(index);
      const auto name = unique_name(emitter_names, desired);
      emitter_names.insert(name);
      destination.emitter_names.push_back(name);
    }
    for (const auto& [material_index, emitter_index] : source.material_to_emitter_profile) {
      destination.material_to_emitter_profile.emplace(remap(material_index, material_base, source.materials.size()), emitter(emitter_index));
    }
    std::unordered_set<std::string> camera_names;
    for (const auto& camera : destination.cameras)
      camera_names.insert(camera.id);
    for (auto camera : source.cameras) {
      camera.cam.medium_index = medium(camera.cam.medium_index);
      camera.cam.lens_image = image(camera.cam.lens_image);
      camera.id = unique_name(camera_names, camera.id);
      camera_names.insert(camera.id);
      if (first == false)
        camera.active = false;
      destination.cameras.push_back(camera);
    }
    std::unordered_set<std::string> node_names(destination.hierarchy.node_names.begin(), destination.hierarchy.node_names.end());
    for (uint32_t index = 0u; index < source.hierarchy.nodes.size(); ++index) {
      const auto& node = source.hierarchy.nodes[index];
      const auto name = unique_name(node_names, source.hierarchy.node_names.at(index));
      node_names.insert(name);
      if (destination.hierarchy.add_node(name.c_str(), kInvalidIndex, node.local_transform) != (node_base + index))
        throw std::runtime_error("Invalid node transform.");
      destination.hierarchy.nodes.back().flags = node.flags;
    }
    for (uint32_t index = 0u; index < source.hierarchy.nodes.size(); ++index) {
      const auto& node = source.hierarchy.nodes[index];
      if (destination.hierarchy.set_parent(node_base + index, remap(node.parent_index, node_base, source.hierarchy.nodes.size())) == false)
        throw std::runtime_error("Invalid hierarchy parent.");
      for (uint32_t local = 0u; local < node.attachment_count; ++local) {
        auto attachment = source.hierarchy.attachments.at(node.attachment_offset + local);
        switch (attachment.type) {
          case SceneAttachment::Type::Mesh:
            attachment.resource_index = remap(attachment.resource_index, mesh_base, source.meshes.size());
            break;
          case SceneAttachment::Type::Camera:
            attachment.resource_index = remap(attachment.resource_index, camera_base, source.cameras.size());
            break;
          case SceneAttachment::Type::Emitter:
            attachment.resource_index = emitter(attachment.resource_index);
            break;
          case SceneAttachment::Type::Medium:
            attachment.resource_index = medium(attachment.resource_index);
            break;
          default:
            throw std::runtime_error("Unknown hierarchy attachment type.");
        }
        if (destination.hierarchy.add_attachment(node_base + index, attachment) == false)
          throw std::runtime_error("Could not copy a hierarchy attachment.");
      }
    }
    if (first) {
      destination.defaults = source.defaults;
      destination.options = source.options;
      destination.pixel_filter = source.pixel_filter;
      destination.json_file_name = source.json_file_name;
      destination.geometry_file_name = source.geometry_file_name;
      destination.materials_file_name = source.materials_file_name;
      destination.owns_assets = source.owns_assets;
    }
    destination.owned_asset_paths.insert(source.owned_asset_paths.begin(), source.owned_asset_paths.end());
    if (source.owns_assets) {
      for (uint32_t index = 0u; index < source.images_vector.size(); ++index)
        destination.owned_asset_paths.insert(source.images.path(index));
      for (uint32_t index = 0u; index < source.mediums_vector.size(); ++index)
        destination.owned_asset_paths.insert(source.mediums.volume_path(index));
    }
    if (destination.resolve_hierarchy() == false)
      throw std::runtime_error("Merged hierarchy validation failed.");
    return true;
  } catch (const std::exception& failure) {
    error = failure.what();
    return false;
  }
}

}  // namespace etx
