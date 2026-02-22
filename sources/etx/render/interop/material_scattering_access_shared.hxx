#pragma once

#ifndef ETX_MATERIAL_SCATTERING_ACCESS_SHARED_CONTEXT_TYPE
# error "ETX_MATERIAL_SCATTERING_ACCESS_SHARED_CONTEXT_TYPE must be defined before including material_scattering_access_shared.hxx"
#endif

#ifndef ETX_MATERIAL_SCATTERING_ACCESS_SHARED_HAS_REQUIRED_SCENE_BUFFERS
# error "ETX_MATERIAL_SCATTERING_ACCESS_SHARED_HAS_REQUIRED_SCENE_BUFFERS must be defined before including material_scattering_access_shared.hxx"
#endif

#ifndef ETX_MATERIAL_SCATTERING_ACCESS_SHARED_LOAD_SCATTERING_SPECTRUM_INDEX
# error "ETX_MATERIAL_SCATTERING_ACCESS_SHARED_LOAD_SCATTERING_SPECTRUM_INDEX must be defined before including material_scattering_access_shared.hxx"
#endif

#ifndef ETX_MATERIAL_SCATTERING_ACCESS_SHARED_LOAD_SCATTERING_IMAGE_INDEX
# error "ETX_MATERIAL_SCATTERING_ACCESS_SHARED_LOAD_SCATTERING_IMAGE_INDEX must be defined before including material_scattering_access_shared.hxx"
#endif

ETX_SHARED_INLINE bool material_scattering_access_shared_try_load(ETX_IN(ETX_MATERIAL_SCATTERING_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t material_index,
  ETX_OUT(uint32_t, scattering_spectrum_index), ETX_OUT(uint32_t, scattering_image_index)) {
  scattering_spectrum_index = kInvalidIndex;
  scattering_image_index = kInvalidIndex;
  if (ETX_MATERIAL_SCATTERING_ACCESS_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) == false) {
    return false;
  }

  scattering_spectrum_index = ETX_MATERIAL_SCATTERING_ACCESS_SHARED_LOAD_SCATTERING_SPECTRUM_INDEX(context, material_index);
  scattering_image_index = ETX_MATERIAL_SCATTERING_ACCESS_SHARED_LOAD_SCATTERING_IMAGE_INDEX(context, material_index);
  return scattering_spectrum_index != kInvalidIndex;
}
