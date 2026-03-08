#pragma once

#include "medium_density_shared.hxx"

ETX_SHARED_INLINE float medium_texture_sample_shared_3d(
  ETX_INOUT(MediumTextureSampleContext, context), ETX_IN(float3, local_coord), ETX_IN(uint3, dimensions)) {
  MediumDensitySharedTextureSample3D sample = medium_density_shared_zero_texture_sample_3d();
  if (medium_density_shared_prepare_texture_sample_3d(local_coord, dimensions, sample) == false) {
    return 0.0f;
  }

  float d000 = medium_texture_sample_density(context, dimensions, sample.ix, sample.iy, sample.iz);
  float d001 = medium_texture_sample_density(context, dimensions, sample.nx, sample.iy, sample.iz);
  float d010 = medium_texture_sample_density(context, dimensions, sample.ix, sample.ny, sample.iz);
  float d011 = medium_texture_sample_density(context, dimensions, sample.nx, sample.ny, sample.iz);
  float d100 = medium_texture_sample_density(context, dimensions, sample.ix, sample.iy, sample.nz);
  float d101 = medium_texture_sample_density(context, dimensions, sample.nx, sample.iy, sample.nz);
  float d110 = medium_texture_sample_density(context, dimensions, sample.ix, sample.ny, sample.nz);
  float d111 = medium_texture_sample_density(context, dimensions, sample.nx, sample.ny, sample.nz);
  return medium_density_shared_trilerp(d000, d001, d010, d011, d100, d101, d110, d111, sample.dx, sample.dy, sample.dz);
}
