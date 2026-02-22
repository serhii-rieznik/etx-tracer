#pragma once

#include "medium_density_shared.hxx"

#ifndef ETX_MEDIUM_GRID_POLICY_SHARED_CONTEXT_TYPE
# error "ETX_MEDIUM_GRID_POLICY_SHARED_CONTEXT_TYPE must be defined before including medium_grid_policy_shared.hxx"
#endif

#ifndef ETX_MEDIUM_GRID_POLICY_SHARED_GRID
# error "ETX_MEDIUM_GRID_POLICY_SHARED_GRID must be defined before including medium_grid_policy_shared.hxx"
#endif

#ifndef ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_NOISE
# error "ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_NOISE must be defined before including medium_grid_policy_shared.hxx"
#endif

#ifndef ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_TEXTURE
# error "ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_TEXTURE must be defined before including medium_grid_policy_shared.hxx"
#endif

#ifndef ETX_MEDIUM_GRID_POLICY_SHARED_TEXTURE_READY
# error "ETX_MEDIUM_GRID_POLICY_SHARED_TEXTURE_READY must be defined before including medium_grid_policy_shared.hxx"
#endif

ETX_SHARED_INLINE float medium_grid_policy_shared_sample_density(
  ETX_INOUT(ETX_MEDIUM_GRID_POLICY_SHARED_CONTEXT_TYPE, context), ETX_IN(float3, local_coord)) {
  MediumDensitySharedGrid grid = ETX_MEDIUM_GRID_POLICY_SHARED_GRID(context);
  float value = 0.0f;
  if (grid.type == MediumGridType::NoiseFunction) {
    value = ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_NOISE(context, local_coord);
  } else if (grid.type == MediumGridType::Texture3D) {
    value = ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_TEXTURE(context, local_coord);
  }

  return medium_density_shared_apply_shape(value, grid.noise_power, grid.noise_sharpness);
}

ETX_SHARED_INLINE bool medium_grid_policy_shared_has_grid_data(ETX_INOUT(ETX_MEDIUM_GRID_POLICY_SHARED_CONTEXT_TYPE, context)) {
  MediumDensitySharedGrid grid = ETX_MEDIUM_GRID_POLICY_SHARED_GRID(context);
  if (medium_density_shared_has_grid_data(grid.type, grid.dimensions, grid.density_count) == false) {
    return false;
  }

  if (grid.type == MediumGridType::NoiseFunction) {
    return true;
  }

  return ETX_MEDIUM_GRID_POLICY_SHARED_TEXTURE_READY(context);
}
