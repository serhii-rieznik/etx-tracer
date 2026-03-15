#pragma once

#include <access/material_access_shared.hxx>
#include <interop/gpu_abi_access_shared.hxx>
#include <interop/scene_gpu_access_shared.hxx>

struct MaterialAccessGPUContext {
  uint materials_descriptor_index;
};

bool material_access_can_load(MaterialAccessGPUContext context, uint material_index) {
  return scene_gpu_has_descriptor(context.materials_descriptor_index) && (material_index != kInvalidIndex);
}

bool material_access_try_load(MaterialAccessGPUContext context, uint material_index, out MaterialAccess access) {
  access = ETX_ZERO(MaterialAccess);
  if (material_access_can_load(context, material_index) == false) {
    return false;
  }

  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(context.materials_descriptor_index)];
  GPUMaterialABIData material_data = gpu_abi_load_material(material_buffer, material_index);
  access.material_class = material_data.material_class;
  access.int_medium_index = material_data.int_medium_index;
  access.ext_medium_index = material_data.ext_medium_index;
  access.scattering_spectrum_index = material_data.scattering_spectrum_index;
  access.scattering_image_index = material_data.scattering_image_index;
  access.opacity = material_data.opacity;
  return true;
}
