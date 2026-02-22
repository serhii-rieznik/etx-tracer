#pragma once

#include "medium.hxx"

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_CONTEXT_TYPE
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_CONTEXT_TYPE must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_ACCESS_TYPE
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_ACCESS_TYPE must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_RESPONSE_TYPE
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_RESPONSE_TYPE must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_QUERY_TYPE
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_QUERY_TYPE must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_REQUIRED_SCENE_BUFFERS
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_REQUIRED_SCENE_BUFFERS must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRY_LOAD_ACCESS
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRY_LOAD_ACCESS must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_MEDIUM_CLASS
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_MEDIUM_CLASS must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_GRID_DATA
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_GRID_DATA must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_INTEGRATED
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_INTEGRATED must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_SPECTRAL
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_SPECTRAL must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_INTEGRATED
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_INTEGRATED must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_SPECTRAL
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_SPECTRAL must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_INTEGRATED
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_INTEGRATED must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_SPECTRAL
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_SPECTRAL must be defined before including medium_segment_transmittance_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_ONE
# error "ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_ONE must be defined before including medium_segment_transmittance_shared.hxx"
#endif

ETX_SHARED_INLINE bool medium_segment_transmittance_shared_supported_class(uint32_t medium_class) {
  return (medium_class == Medium::Homogeneous) || (medium_class == Medium::Heterogeneous);
}

ETX_SHARED_INLINE float3 medium_segment_transmittance_shared_integrated(ETX_INOUT(ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_CONTEXT_TYPE, context), uint32_t medium_index,
  ETX_IN(float3, origin), ETX_IN(float3, direction), float distance) {
  float3 one = float3(1.0f, 1.0f, 1.0f);
  if ((distance <= 0.0f) || (ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) == false)) {
    return one;
  }

  ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_ACCESS_TYPE medium_access;
  if (ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRY_LOAD_ACCESS(context, medium_index, medium_access) == false) {
    return one;
  }

  uint32_t medium_class = ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_MEDIUM_CLASS(context, medium_access);
  if (medium_segment_transmittance_shared_supported_class(medium_class) == false) {
    return one;
  }

  float3 extinction = ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_INTEGRATED(context, medium_access);
  if (medium_class == Medium::Homogeneous) {
    return ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_INTEGRATED(context, medium_access, extinction, distance);
  }

  if (ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_GRID_DATA(context, medium_access) == false) {
    return one;
  }

  return ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_INTEGRATED(context, medium_access, extinction, origin, direction, distance);
}

ETX_SHARED_INLINE ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_RESPONSE_TYPE medium_segment_transmittance_shared_spectral(
  ETX_INOUT(ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_CONTEXT_TYPE, context), uint32_t medium_index, ETX_IN(float3, origin), ETX_IN(float3, direction), float distance,
  ETX_IN(ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_QUERY_TYPE, spect)) {
  ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_RESPONSE_TYPE one = ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_ONE(spect);
  if ((distance <= 0.0f) || (ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) == false)) {
    return one;
  }

  ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_ACCESS_TYPE medium_access;
  if (ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRY_LOAD_ACCESS(context, medium_index, medium_access) == false) {
    return one;
  }

  uint32_t medium_class = ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_MEDIUM_CLASS(context, medium_access);
  if (medium_segment_transmittance_shared_supported_class(medium_class) == false) {
    return one;
  }

  ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_RESPONSE_TYPE extinction =
    ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_SPECTRAL(context, medium_access, spect);
  if (medium_class == Medium::Homogeneous) {
    return ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_SPECTRAL(context, medium_access, extinction, distance, spect);
  }

  if (ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_GRID_DATA(context, medium_access) == false) {
    return one;
  }

  return ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_SPECTRAL(context, medium_access, extinction, origin, direction, distance, spect);
}
