#pragma once

#include "spectrum.hxx"

#ifndef ETX_SPECTRUM_ACCESS_SHARED_CONTEXT_TYPE
# error "ETX_SPECTRUM_ACCESS_SHARED_CONTEXT_TYPE must be defined before including spectrum_access_shared.hxx"
#endif

#ifndef ETX_SPECTRUM_ACCESS_SHARED_INTEGRATED
# error "ETX_SPECTRUM_ACCESS_SHARED_INTEGRATED must be defined before including spectrum_access_shared.hxx"
#endif

#ifndef ETX_SPECTRUM_ACCESS_SHARED_ENTRY_COUNT
# error "ETX_SPECTRUM_ACCESS_SHARED_ENTRY_COUNT must be defined before including spectrum_access_shared.hxx"
#endif

#ifndef ETX_SPECTRUM_ACCESS_SHARED_ENTRY_WAVELENGTH
# error "ETX_SPECTRUM_ACCESS_SHARED_ENTRY_WAVELENGTH must be defined before including spectrum_access_shared.hxx"
#endif

#ifndef ETX_SPECTRUM_ACCESS_SHARED_ENTRY_POWER
# error "ETX_SPECTRUM_ACCESS_SHARED_ENTRY_POWER must be defined before including spectrum_access_shared.hxx"
#endif

#ifndef ETX_SPECTRUM_ACCESS_SHARED_QUERY_TYPE
# define ETX_SPECTRUM_ACCESS_SHARED_QUERY_TYPE SpectralQuery
#endif

#ifndef ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_TYPE
# define ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_TYPE SpectralResponse
#endif

#ifndef ETX_SPECTRUM_ACCESS_SHARED_QUERY_IS_SPECTRAL
# define ETX_SPECTRUM_ACCESS_SHARED_QUERY_IS_SPECTRAL(query) spectral_query_is_spectral(query)
#endif

#ifndef ETX_SPECTRUM_ACCESS_SHARED_QUERY_WAVELENGTH
# define ETX_SPECTRUM_ACCESS_SHARED_QUERY_WAVELENGTH(query) query.wavelength
#endif

#ifndef ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_MAKE_SCALAR
# define ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_MAKE_SCALAR(query, value) spectral_response_make(query, value)
#endif

#ifndef ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_MAKE_INTEGRATED
# define ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_MAKE_INTEGRATED(query, value) spectral_response_make(query, value)
#endif

ETX_SHARED_INLINE float3 spectrum_access_shared_integrated(ETX_IN(ETX_SPECTRUM_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t spectrum_index) {
  return ETX_SPECTRUM_ACCESS_SHARED_INTEGRATED(context, spectrum_index);
}

ETX_SHARED_INLINE ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_TYPE spectrum_access_shared_query(ETX_IN(ETX_SPECTRUM_ACCESS_SHARED_CONTEXT_TYPE, context), uint32_t spectrum_index,
  ETX_IN(ETX_SPECTRUM_ACCESS_SHARED_QUERY_TYPE, query)) {
  if (ETX_SPECTRUM_ACCESS_SHARED_QUERY_IS_SPECTRAL(query) == false) {
    return ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_MAKE_INTEGRATED(query, ETX_SPECTRUM_ACCESS_SHARED_INTEGRATED(context, spectrum_index));
  }

  uint32_t entry_count = ETX_SPECTRUM_ACCESS_SHARED_ENTRY_COUNT(context, spectrum_index);
  if (entry_count == 0u) {
    return ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_MAKE_SCALAR(query, 0.0f);
  }

  uint32_t begin = 0u;
  uint32_t end = entry_count;
  while ((end - begin) > 1u) {
    uint32_t middle = begin + ((end - begin) / 2u);
    float middle_wavelength = ETX_SPECTRUM_ACCESS_SHARED_ENTRY_WAVELENGTH(context, spectrum_index, middle);
    if (middle_wavelength > ETX_SPECTRUM_ACCESS_SHARED_QUERY_WAVELENGTH(query)) {
      end = middle;
    } else {
      begin = middle;
    }
  }

  uint32_t i = begin;
  if (i >= entry_count) {
    return ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_MAKE_SCALAR(query, 0.0f);
  }

  float wi = ETX_SPECTRUM_ACCESS_SHARED_ENTRY_WAVELENGTH(context, spectrum_index, i);
  if ((i == 0u) && (ETX_SPECTRUM_ACCESS_SHARED_QUERY_WAVELENGTH(query) < wi)) {
    return ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_MAKE_SCALAR(query, 0.0f);
  }

  if (((i + 1u) == entry_count) && (ETX_SPECTRUM_ACCESS_SHARED_QUERY_WAVELENGTH(query) > wi)) {
    return ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_MAKE_SCALAR(query, 0.0f);
  }

  uint32_t j = min(i + 1u, entry_count - 1u);
  float wj = ETX_SPECTRUM_ACCESS_SHARED_ENTRY_WAVELENGTH(context, spectrum_index, j);
  float pi = ETX_SPECTRUM_ACCESS_SHARED_ENTRY_POWER(context, spectrum_index, i);
  float pj = ETX_SPECTRUM_ACCESS_SHARED_ENTRY_POWER(context, spectrum_index, j);
  float t = (i == j) ? 0.0f : ((ETX_SPECTRUM_ACCESS_SHARED_QUERY_WAVELENGTH(query) - wi) / (wj - wi));
  return ETX_SPECTRUM_ACCESS_SHARED_RESPONSE_MAKE_SCALAR(query, lerp(pi, pj, t));
}
