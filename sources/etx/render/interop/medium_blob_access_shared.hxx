#pragma once

#include "gpu_abi_constants.hxx"

#ifndef ETX_MEDIUM_BLOB_ACCESS_SHARED_CONTEXT_TYPE
# error "ETX_MEDIUM_BLOB_ACCESS_SHARED_CONTEXT_TYPE must be defined before including medium_blob_access_shared.hxx"
#endif

#ifndef ETX_MEDIUM_BLOB_ACCESS_SHARED_LOAD_U32
# error "ETX_MEDIUM_BLOB_ACCESS_SHARED_LOAD_U32 must be defined before including medium_blob_access_shared.hxx"
#endif

ETX_SHARED_INLINE uint32_t medium_blob_access_shared_medium_count(ETX_IN(ETX_MEDIUM_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_MEDIUM_BLOB_ACCESS_SHARED_LOAD_U32(context, kMediumBlobHeaderMediumCountOffset);
}

ETX_SHARED_INLINE uint32_t medium_blob_access_shared_mediums_offset(ETX_IN(ETX_MEDIUM_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_MEDIUM_BLOB_ACCESS_SHARED_LOAD_U32(context, kMediumBlobHeaderMediumsOffset);
}

ETX_SHARED_INLINE uint32_t medium_blob_access_shared_data_chunk_count(ETX_IN(ETX_MEDIUM_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_MEDIUM_BLOB_ACCESS_SHARED_LOAD_U32(context, kMediumBlobHeaderDataChunkCountOffset);
}

ETX_SHARED_INLINE uint32_t medium_blob_access_shared_data_chunk_indices_offset(ETX_IN(ETX_MEDIUM_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_MEDIUM_BLOB_ACCESS_SHARED_LOAD_U32(context, kMediumBlobHeaderDataChunkIndicesOffset);
}

ETX_SHARED_INLINE uint32_t medium_blob_access_shared_desc_offset(ETX_IN(ETX_MEDIUM_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t medium_index) {
  uint32_t mediums_offset = medium_blob_access_shared_mediums_offset(context);
  return mediums_offset + medium_index * kMediumStride;
}

ETX_SHARED_INLINE uint32_t medium_blob_access_shared_chunk_descriptor_index(ETX_IN(ETX_MEDIUM_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t chunk_index) {
  uint32_t data_chunk_count = medium_blob_access_shared_data_chunk_count(context);
  uint32_t data_chunk_indices_offset = medium_blob_access_shared_data_chunk_indices_offset(context);
  if ((chunk_index == kInvalidIndex) || (chunk_index >= data_chunk_count) || (data_chunk_indices_offset == kInvalidIndex)) {
    return kInvalidIndex;
  }

  return ETX_MEDIUM_BLOB_ACCESS_SHARED_LOAD_U32(context, data_chunk_indices_offset + chunk_index * 4u);
}

ETX_SHARED_INLINE bool medium_blob_access_shared_try_get_desc_offset(
  ETX_IN(ETX_MEDIUM_BLOB_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t medium_index, ETX_OUT(uint32_t, medium_desc_offset)) {
  medium_desc_offset = kInvalidIndex;
  uint32_t medium_count = medium_blob_access_shared_medium_count(context);
  if (medium_index >= medium_count) {
    return false;
  }

  uint32_t mediums_offset = medium_blob_access_shared_mediums_offset(context);
  if (mediums_offset == kInvalidIndex) {
    return false;
  }

  medium_desc_offset = medium_blob_access_shared_desc_offset(context, medium_index);
  return true;
}
