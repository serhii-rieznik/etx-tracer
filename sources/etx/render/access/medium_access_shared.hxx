#pragma once

struct ETX_ALIGNED MediumAccess {
  MediumDensitySharedGrid grid ETX_INIT({});
  float3 bounds_min ETX_INIT({});
  float3 bounds_max ETX_INIT({});
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  uint32_t density_payload_descriptor_index ETX_INIT(kInvalidIndex);
  uint32_t medium_class ETX_INIT(Medium::Homogeneous);
  uint32_t absorption_spectrum_index ETX_INIT(kInvalidIndex);
  uint32_t scattering_spectrum_index ETX_INIT(kInvalidIndex);
};

ETX_SHARED_INLINE BoundingBox medium_access_bounds(ETX_IN(MediumAccess, access)) {
  BoundingBox result;
  result.p_min = access.bounds_min;
  result.p_max = access.bounds_max;
  return result;
}

ETX_SHARED_INLINE bool medium_access_supported_class(uint32_t medium_class) {
  return (medium_class == Medium::Homogeneous) || (medium_class == Medium::Heterogeneous);
}
