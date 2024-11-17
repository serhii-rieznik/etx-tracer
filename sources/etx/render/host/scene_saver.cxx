#include <etx/render/host/scene_saver.hxx>

namespace etx {

void write_materials(const SceneRepresentation& scene_rep, const char* filename) {
  FILE* fout = fopen(filename, "w");
  if (fout == nullptr) {
    log::error("Failed to write materials file: %s", filename);
    return;
  }

  const auto& scene = scene_rep.scene();

  for (const auto& em : scene.emitters) {
    switch (em.cls) {
      case Emitter::Class::Directional: {
        float3 e = em.emission.spectrum.integrate_to_xyz();
        fprintf(fout, "newmtl et::dir\n");
        fprintf(fout, "color %.3f %.3f %.3f\n", e.x, e.y, e.z);
        fprintf(fout, "direction %.3f %.3f %.3f\n", em.direction.x, em.direction.y, em.direction.z);
        fprintf(fout, "angular_diameter %.3f\n", em.angular_size * 180.0f / kPi);
        fprintf(fout, "\n");
        break;
      }
      case Emitter::Class::Environment: {
        float3 e = em.emission.spectrum.integrate_to_xyz();
        fprintf(fout, "newmtl et::env\n");
        fprintf(fout, "color %.3f %.3f %.3f\n", e.x, e.y, e.z);
        fprintf(fout, "\n");
        break;
      }
      default:
        break;
    }
  }

  for (const auto& mmap : scene_rep.material_mapping()) {
    const auto& material = scene.materials[mmap.second];

    fprintf(fout, "newmtl %s\n", mmap.first.c_str());
    fprintf(fout, "material class %s\n", material_class_to_string(material.cls));
    // TODO : support anisotripic roughness
    fprintf(fout, "Pr %.3f\n", sqrtf(0.5f * (sqr(material.roughness.x) + sqr(material.roughness.y))));
    {
      float3 ks = spectrum::xyz_to_rgb(material.reflectance.spectrum.integrate_to_xyz());
      fprintf(fout, "Ks %.3f %.3f %.3f\n", ks.x, ks.y, ks.z);
    }
    {
      float3 kt = spectrum::xyz_to_rgb(material.transmittance.spectrum.integrate_to_xyz());
      fprintf(fout, "Kt %.3f %.3f %.3f\n", kt.x, kt.y, kt.z);
    }

    if (material.emission.spectrum.is_zero() == false) {
      float3 ke = spectrum::xyz_to_rgb(material.emission.spectrum.integrate_to_xyz());
      fprintf(fout, "Ke %.3f %.3f %.3f\n", ke.x, ke.y, ke.z);
    }

    if (material.subsurface.scattering_distance.is_zero() == false) {
      float3 ss = spectrum::xyz_to_rgb(material.subsurface.scattering_distance.integrate_to_xyz());
      fprintf(fout, "subsurface %.3f %.3f %.3f\n", ss.x, ss.y, ss.z);
    }

    fprintf(fout, "\n");
  }

  fclose(fout);
}

void save_scene_to_file(const SceneRepresentation& scene, const char* filename) {
#if (0)
  if (_private->geometry_file_name.empty())
    return;

  FILE* fout = fopen(filename, "w");
  if (fout == nullptr)
    return;

  auto materials_file = _private->geometry_file_name + ".materials";
  auto relative_obj_file = std::filesystem::relative(_private->geometry_file_name, std::filesystem::path(filename).parent_path()).string();
  auto relative_mtl_file = std::filesystem::relative(materials_file, std::filesystem::path(filename).parent_path()).string();
  write_materials(materials_file.c_str());

  auto j = json_object();
  json_object_set(j, "geometry", json_string(relative_obj_file.c_str()));
  json_object_set(j, "materials", json_string(relative_mtl_file.c_str()));

  {
    auto camera = json_object();
    json_object_set(camera, "viewport", json_uint2_to_array(_private->scene.camera.image_size));
    json_object_set(camera, "origin", json_float3_to_array(_private->scene.camera.position));
    json_object_set(camera, "target", json_float3_to_array(_private->scene.camera.position + _private->scene.camera.direction));
    json_object_set(camera, "up", json_float3_to_array(_private->scene.camera.up));
    json_object_set(camera, "lens-radius", json_real(_private->scene.camera.lens_radius));
    json_object_set(camera, "focal-distance", json_real(_private->scene.camera.focal_distance));
    json_object_set(camera, "focal-length", json_real(get_camera_focal_length(_private->scene.camera)));
    json_object_set(j, "camera", camera);
  }

  json_dumpf(j, fout, JSON_INDENT(2));
  json_decref(j);

  fclose(fout);
#endif
}

}  // namespace etx
