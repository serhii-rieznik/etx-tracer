#pragma once

struct ETX_ALIGNED MaterialAccess {
  uint32_t material_class ETX_INIT(kInvalidIndex);
  uint32_t int_medium_index ETX_INIT(kInvalidIndex);
  uint32_t ext_medium_index ETX_INIT(kInvalidIndex);
  uint32_t scattering_spectrum_index ETX_INIT(kInvalidIndex);
  uint32_t scattering_image_index ETX_INIT(kInvalidIndex);
  float opacity ETX_INIT(1.0f);
};

ETX_SHARED_INLINE bool material_access_has_scattering(ETX_IN(MaterialAccess, access)) {
  return access.scattering_spectrum_index != kInvalidIndex;
}
