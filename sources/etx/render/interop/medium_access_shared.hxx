#pragma once

#include "gpu_abi_constants.hxx"

#ifndef ETX_MEDIUM_ACCESS_SHARED_CONTEXT_TYPE
# error "ETX_MEDIUM_ACCESS_SHARED_CONTEXT_TYPE must be defined before including medium_access_shared.hxx"
#endif

#ifndef ETX_MEDIUM_ACCESS_SHARED_ACCESS_TYPE
# error "ETX_MEDIUM_ACCESS_SHARED_ACCESS_TYPE must be defined before including medium_access_shared.hxx"
#endif

#ifndef ETX_MEDIUM_ACCESS_SHARED_LOAD_U32
# error "ETX_MEDIUM_ACCESS_SHARED_LOAD_U32 must be defined before including medium_access_shared.hxx"
#endif

#ifndef ETX_MEDIUM_ACCESS_SHARED_LOAD_U16
# error "ETX_MEDIUM_ACCESS_SHARED_LOAD_U16 must be defined before including medium_access_shared.hxx"
#endif

#ifndef ETX_MEDIUM_ACCESS_SHARED_LOAD_U32X3
# error "ETX_MEDIUM_ACCESS_SHARED_LOAD_U32X3 must be defined before including medium_access_shared.hxx"
#endif

#ifndef ETX_MEDIUM_ACCESS_SHARED_LOAD_F32
# error "ETX_MEDIUM_ACCESS_SHARED_LOAD_F32 must be defined before including medium_access_shared.hxx"
#endif

#ifndef ETX_MEDIUM_ACCESS_SHARED_LOAD_F32X3
# error "ETX_MEDIUM_ACCESS_SHARED_LOAD_F32X3 must be defined before including medium_access_shared.hxx"
#endif

#ifndef ETX_MEDIUM_ACCESS_SHARED_CHUNK_DESCRIPTOR_INDEX
# error "ETX_MEDIUM_ACCESS_SHARED_CHUNK_DESCRIPTOR_INDEX must be defined before including medium_access_shared.hxx"
#endif

ETX_SHARED_INLINE void medium_access_shared_load(
  ETX_IN(ETX_MEDIUM_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t medium_desc_offset, ETX_OUT(ETX_MEDIUM_ACCESS_SHARED_ACCESS_TYPE, access)) {
  access.grid.dimensions = ETX_MEDIUM_ACCESS_SHARED_LOAD_U32X3(context, medium_desc_offset + kMediumGridDimensionsOffset);
  access.grid.type = ETX_MEDIUM_ACCESS_SHARED_LOAD_U32(context, medium_desc_offset + kMediumGridTypeOffset);
  access.grid.noise_type = ETX_MEDIUM_ACCESS_SHARED_LOAD_U32(context, medium_desc_offset + kMediumGridNoiseTypeOffset);
  access.grid.density_data_offset = ETX_MEDIUM_ACCESS_SHARED_LOAD_U32(context, medium_desc_offset + kMediumGridDensityDataOffsetOffset);
  access.grid.density_count = ETX_MEDIUM_ACCESS_SHARED_LOAD_U32(context, medium_desc_offset + kMediumGridDensityCountOffset);
  access.grid.noise_seed = ETX_MEDIUM_ACCESS_SHARED_LOAD_U32(context, medium_desc_offset + kMediumGridNoiseSeedOffset);
  access.grid.noise_offset = ETX_MEDIUM_ACCESS_SHARED_LOAD_F32X3(context, medium_desc_offset + kMediumGridNoiseOffsetOffset);
  access.grid.noise_enable_border_fade = ETX_MEDIUM_ACCESS_SHARED_LOAD_U32(context, medium_desc_offset + kMediumGridNoiseEnableBorderFadeOffset);
  access.grid.noise_octaves = ETX_MEDIUM_ACCESS_SHARED_LOAD_U32(context, medium_desc_offset + kMediumGridNoiseOctavesOffset);
  access.grid.noise_scale = ETX_MEDIUM_ACCESS_SHARED_LOAD_F32(context, medium_desc_offset + kMediumGridNoiseScaleOffset);
  access.grid.noise_lacunarity = ETX_MEDIUM_ACCESS_SHARED_LOAD_F32(context, medium_desc_offset + kMediumGridNoiseLacunarityOffset);
  access.grid.noise_persistence = ETX_MEDIUM_ACCESS_SHARED_LOAD_F32(context, medium_desc_offset + kMediumGridNoisePersistenceOffset);
  access.grid.noise_power = ETX_MEDIUM_ACCESS_SHARED_LOAD_F32(context, medium_desc_offset + kMediumGridNoisePowerOffset);
  access.grid.noise_sharpness = ETX_MEDIUM_ACCESS_SHARED_LOAD_F32(context, medium_desc_offset + kMediumGridNoiseSharpnessOffset);
  access.grid.noise_border_fade_distance = ETX_MEDIUM_ACCESS_SHARED_LOAD_F32(context, medium_desc_offset + kMediumGridNoiseBorderFadeDistanceOffset);
  access.grid.density_data_chunk_index = ETX_MEDIUM_ACCESS_SHARED_LOAD_U32(context, medium_desc_offset + kMediumGridDensityDataChunkIndexOffset);
  access.bounds_min = ETX_MEDIUM_ACCESS_SHARED_LOAD_F32X3(context, medium_desc_offset + kMediumBoundsMinOffset);
  access.bounds_max = ETX_MEDIUM_ACCESS_SHARED_LOAD_F32X3(context, medium_desc_offset + kMediumBoundsMaxOffset);
  access.absorption_spectrum_index = ETX_MEDIUM_ACCESS_SHARED_LOAD_U32(context, medium_desc_offset + kMediumAbsorptionIndexOffset);
  access.scattering_spectrum_index = ETX_MEDIUM_ACCESS_SHARED_LOAD_U32(context, medium_desc_offset + kMediumScatteringIndexOffset);
  access.medium_class = ETX_MEDIUM_ACCESS_SHARED_LOAD_U16(context, medium_desc_offset + kMediumClassOffset);
  access.density_payload_descriptor_index = ETX_MEDIUM_ACCESS_SHARED_CHUNK_DESCRIPTOR_INDEX(context, access.grid.density_data_chunk_index);
}
