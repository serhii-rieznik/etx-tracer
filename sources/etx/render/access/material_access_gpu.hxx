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
  access.alpha_mask_image_index = material_data.alpha_mask_image_index;
  access.alpha_mask_channel = material_data.alpha_mask_channel;
  return true;
}

bool material_access_try_load_full(MaterialAccessGPUContext context, uint material_index, out Material material) {
  material = ETX_ZERO(Material);
  if (material_access_can_load(context, material_index) == false) {
    return false;
  }

  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(context.materials_descriptor_index)];
  material = gpu_abi_load_material_full(material_buffer, material_index);
  return true;
}

bool material_access_try_load_bump(MaterialAccessGPUContext context, uint material_index, out SampledImage bump) {
  bump = ETX_ZERO(SampledImage);
  if (material_access_can_load(context, material_index) == false)
    return false;
  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(context.materials_descriptor_index)];
  const uint base_offset = material_index * kMaterialStride;
  bump.value.x = asfloat(material_buffer.Load(base_offset + kMaterialBumpValueOffset));
  bump.image_index = gpu_abi_load_u32(material_buffer, base_offset + kMaterialBumpImageIndexOffset);
  bump.channel = gpu_abi_load_u32(material_buffer, base_offset + kMaterialBumpChannelOffset);
  return true;
}
