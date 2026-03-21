#pragma once

#include <etx/render/access/material_access_shared.hxx>

struct Scene;

struct MaterialAccessCPUContext {
  const Scene* scene = nullptr;
};

ETX_SHARED_INLINE MaterialAccessCPUContext make_material_access_cpu_context(const Scene& scene) {
  MaterialAccessCPUContext result = {};
  result.scene = &scene;
  return result;
}

ETX_SHARED_INLINE bool material_access_can_load(ETX_IN(MaterialAccessCPUContext, context), uint32_t material_index) {
  return (context.scene != nullptr) && (material_index < context.scene->materials.count);
}

ETX_SHARED_INLINE MaterialAccess material_access_cpu_make(ETX_IN(Material, material)) {
  MaterialAccess result = {};
  result.material_class = material.cls;
  result.int_medium_index = material.int_medium;
  result.ext_medium_index = material.ext_medium;
  result.scattering_spectrum_index = material.scattering.spectrum_index;
  result.scattering_image_index = material.scattering.image_index;
  result.opacity = material.opacity;
  return result;
}

ETX_SHARED_INLINE bool material_access_try_load(
  ETX_IN(MaterialAccessCPUContext, context), uint32_t material_index, ETX_OUT(MaterialAccess, access)) {
  access = {};
  if (material_access_can_load(context, material_index) == false) {
    return false;
  }

  access = material_access_cpu_make(context.scene->materials[material_index]);
  return true;
}

ETX_SHARED_INLINE bool material_access_try_load_full(
  ETX_IN(MaterialAccessCPUContext, context), uint32_t material_index, ETX_OUT(Material, material)) {
  material = {};
  if (material_access_can_load(context, material_index) == false) {
    return false;
  }

  material = context.scene->materials[material_index];
  return true;
}
