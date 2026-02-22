#pragma once

#ifndef ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE
# error "ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE must be defined before including medium_extinction_shared.hxx"
#endif

#ifndef ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE
# error "ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE must be defined before including medium_extinction_shared.hxx"
#endif

#ifndef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_ABSORPTION_INDEX
# error "ETX_MEDIUM_EXTINCTION_SHARED_LOAD_ABSORPTION_INDEX must be defined before including medium_extinction_shared.hxx"
#endif

#ifndef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SCATTERING_INDEX
# error "ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SCATTERING_INDEX must be defined before including medium_extinction_shared.hxx"
#endif

#ifndef ETX_MEDIUM_EXTINCTION_SHARED_CAN_SAMPLE_SPECTRUM
# error "ETX_MEDIUM_EXTINCTION_SHARED_CAN_SAMPLE_SPECTRUM must be defined before including medium_extinction_shared.hxx"
#endif

#ifndef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_INTEGRATED
# error "ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_INTEGRATED must be defined before including medium_extinction_shared.hxx"
#endif

#ifndef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_SPECTRAL
# error "ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_SPECTRAL must be defined before including medium_extinction_shared.hxx"
#endif

#ifndef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_RESPONSE_TYPE
# error "ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_RESPONSE_TYPE must be defined before including medium_extinction_shared.hxx"
#endif

#ifndef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_QUERY_TYPE
# error "ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_QUERY_TYPE must be defined before including medium_extinction_shared.hxx"
#endif

#ifndef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ZERO
# error "ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ZERO must be defined before including medium_extinction_shared.hxx"
#endif

#ifndef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ADD
# error "ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ADD must be defined before including medium_extinction_shared.hxx"
#endif

ETX_SHARED_INLINE float3 medium_extinction_shared_load_absorption_integrated(
  ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE, context), ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE, access)) {
  uint32_t absorption_index = ETX_MEDIUM_EXTINCTION_SHARED_LOAD_ABSORPTION_INDEX(context, access);
  if (ETX_MEDIUM_EXTINCTION_SHARED_CAN_SAMPLE_SPECTRUM(context, absorption_index) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  return ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_INTEGRATED(context, absorption_index);
}

ETX_SHARED_INLINE float3 medium_extinction_shared_load_scattering_integrated(
  ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE, context), ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE, access)) {
  uint32_t scattering_index = ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SCATTERING_INDEX(context, access);
  if (ETX_MEDIUM_EXTINCTION_SHARED_CAN_SAMPLE_SPECTRUM(context, scattering_index) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  return ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_INTEGRATED(context, scattering_index);
}

ETX_SHARED_INLINE float3 medium_extinction_shared_load_integrated(
  ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE, context), ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE, access)) {
  return medium_extinction_shared_load_absorption_integrated(context, access) + medium_extinction_shared_load_scattering_integrated(context, access);
}

ETX_SHARED_INLINE ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_RESPONSE_TYPE medium_extinction_shared_load_absorption_spectral(
  ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE, context), ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE, access),
  ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_QUERY_TYPE, spect)) {
  uint32_t absorption_index = ETX_MEDIUM_EXTINCTION_SHARED_LOAD_ABSORPTION_INDEX(context, access);
  if (ETX_MEDIUM_EXTINCTION_SHARED_CAN_SAMPLE_SPECTRUM(context, absorption_index) == false) {
    return ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ZERO(spect);
  }

  return ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_SPECTRAL(context, absorption_index, spect);
}

ETX_SHARED_INLINE ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_RESPONSE_TYPE medium_extinction_shared_load_scattering_spectral(
  ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE, context), ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE, access),
  ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_QUERY_TYPE, spect)) {
  uint32_t scattering_index = ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SCATTERING_INDEX(context, access);
  if (ETX_MEDIUM_EXTINCTION_SHARED_CAN_SAMPLE_SPECTRUM(context, scattering_index) == false) {
    return ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ZERO(spect);
  }

  return ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_SPECTRAL(context, scattering_index, spect);
}

ETX_SHARED_INLINE ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_RESPONSE_TYPE medium_extinction_shared_load_spectral(
  ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE, context), ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE, access),
  ETX_IN(ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_QUERY_TYPE, spect)) {
  ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_RESPONSE_TYPE absorption = medium_extinction_shared_load_absorption_spectral(context, access, spect);
  ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_RESPONSE_TYPE scattering = medium_extinction_shared_load_scattering_spectral(context, access, spect);
  return ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ADD(absorption, scattering);
}
