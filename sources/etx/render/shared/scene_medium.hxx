#pragma once

#include <etx/render/interop/scene_resource_shared.hxx>
#include <etx/render/interop/medium_phase_shared.hxx>
#include <etx/render/shared/medium.hxx>

namespace etx {

struct MediumTransmittanceSharedContext {
  const Medium& medium;
  BoundingBox bounds = {};
  Sampler& sampler;
};

ETX_SHARED_INLINE float medium_transmittance_shared_rnd(ETX_INOUT(MediumTransmittanceSharedContext, context)) {
  return context.sampler.next();
}

ETX_SHARED_INLINE float medium_transmittance_shared_density(ETX_INOUT(MediumTransmittanceSharedContext, context), ETX_IN(float3, local_pos)) {
  return context.medium.sample_density(local_pos, context.bounds);
}

ETX_SHARED_INLINE MediumTransmittanceSharedContext make_medium_transmittance_shared_context(
  const Medium& medium, ETX_IN(BoundingBox, bounds), Sampler& sampler) {
  return {medium, bounds, sampler};
}

ETX_SHARED_INLINE SpectralResponse medium_transmittance_shared_to_spectral_response(ETX_IN(::SpectralResponse, response)) {
  SpectralQuery query = {response.wavelength, response.flags};
  if (::spectral_response_is_spectral(response)) {
    return {query, response.value};
  }
  return {query, response.integrated};
}

#define ETX_MEDIUM_SHARED_CONTEXT_TYPE MediumTransmittanceSharedContext
#define ETX_MEDIUM_SHARED_RND(context) medium_transmittance_shared_rnd(context)
#define ETX_MEDIUM_SHARED_DENSITY(context, local_pos) medium_transmittance_shared_density(context, local_pos)
#include <etx/render/interop/medium_transmittance_shared.hxx>
#undef ETX_MEDIUM_SHARED_DENSITY
#undef ETX_MEDIUM_SHARED_RND
#undef ETX_MEDIUM_SHARED_CONTEXT_TYPE

ETX_SHARED_INLINE uint32_t medium_sample_shared_cpu_load_medium_class(ETX_IN(MediumTransmittanceSharedContext, context)) {
  return static_cast<uint32_t>(context.medium.cls);
}

ETX_SHARED_INLINE bool medium_sample_shared_cpu_has_grid_data(ETX_IN(MediumTransmittanceSharedContext, context)) {
  return context.medium.has_grid_data();
}

ETX_SHARED_INLINE float3 medium_sample_shared_cpu_bounds_min(ETX_IN(MediumTransmittanceSharedContext, context)) {
  return context.bounds.p_min;
}

ETX_SHARED_INLINE float3 medium_sample_shared_cpu_bounds_max(ETX_IN(MediumTransmittanceSharedContext, context)) {
  return context.bounds.p_max;
}

#define ETX_MEDIUM_SAMPLE_SHARED_CONTEXT_TYPE MediumTransmittanceSharedContext
#define ETX_MEDIUM_SAMPLE_SHARED_RND(context) medium_transmittance_shared_rnd(context)
#define ETX_MEDIUM_SAMPLE_SHARED_DENSITY(context, local_pos) medium_transmittance_shared_density(context, local_pos)
#define ETX_MEDIUM_SAMPLE_SHARED_LOAD_MEDIUM_CLASS(context) medium_sample_shared_cpu_load_medium_class(context)
#define ETX_MEDIUM_SAMPLE_SHARED_HAS_GRID_DATA(context) medium_sample_shared_cpu_has_grid_data(context)
#define ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MIN(context) medium_sample_shared_cpu_bounds_min(context)
#define ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MAX(context) medium_sample_shared_cpu_bounds_max(context)
#include <etx/render/interop/medium_sample_shared.hxx>
#undef ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MAX
#undef ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MIN
#undef ETX_MEDIUM_SAMPLE_SHARED_HAS_GRID_DATA
#undef ETX_MEDIUM_SAMPLE_SHARED_LOAD_MEDIUM_CLASS
#undef ETX_MEDIUM_SAMPLE_SHARED_DENSITY
#undef ETX_MEDIUM_SAMPLE_SHARED_RND
#undef ETX_MEDIUM_SAMPLE_SHARED_CONTEXT_TYPE

struct MediumSpectrumCPUSharedAccess {
  uint32_t absorption_index = kInvalidIndex;
  uint32_t scattering_index = kInvalidIndex;
};

struct MediumSpectrumCPUSharedContext {
  const Scene& scene;
};

ETX_SHARED_INLINE MediumSpectrumCPUSharedContext make_medium_spectrum_shared_context(const Scene& scene) {
  return {scene};
}

ETX_SHARED_INLINE MediumSpectrumCPUSharedAccess make_medium_spectrum_shared_access(const Medium& medium) {
  return {medium.absorption_index, medium.scattering_index};
}

ETX_SHARED_INLINE uint32_t medium_extinction_shared_cpu_load_absorption_index(
  ETX_IN(MediumSpectrumCPUSharedContext, context), ETX_IN(MediumSpectrumCPUSharedAccess, access)) {
  (void)context;
  return access.absorption_index;
}

ETX_SHARED_INLINE uint32_t medium_extinction_shared_cpu_load_scattering_index(
  ETX_IN(MediumSpectrumCPUSharedContext, context), ETX_IN(MediumSpectrumCPUSharedAccess, access)) {
  (void)context;
  return access.scattering_index;
}

ETX_SHARED_INLINE bool medium_extinction_shared_cpu_can_sample_spectrum(ETX_IN(MediumSpectrumCPUSharedContext, context), uint32_t spectrum_index) {
  return (spectrum_index != kInvalidIndex) && (spectrum_index < context.scene.spectrums.count);
}

ETX_SHARED_INLINE float3 medium_extinction_shared_cpu_load_spectrum_integrated(ETX_IN(MediumSpectrumCPUSharedContext, context), uint32_t spectrum_index) {
  return context.scene.spectrums[spectrum_index].integrated_value;
}

ETX_SHARED_INLINE ::SpectralResponse medium_extinction_shared_cpu_load_spectrum_spectral(
  ETX_IN(MediumSpectrumCPUSharedContext, context), uint32_t spectrum_index, ETX_IN(::SpectralQuery, spect)) {
  SpectralQuery query = {spect.wavelength, spect.flags};
  const SpectralResponse result = context.scene.spectrums[spectrum_index](query);
  return static_cast<const ::SpectralResponse&>(result);
}

#define ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE MediumSpectrumCPUSharedContext
#define ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE MediumSpectrumCPUSharedAccess
#define ETX_MEDIUM_EXTINCTION_SHARED_LOAD_ABSORPTION_INDEX(context, access) medium_extinction_shared_cpu_load_absorption_index(context, access)
#define ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SCATTERING_INDEX(context, access) medium_extinction_shared_cpu_load_scattering_index(context, access)
#define ETX_MEDIUM_EXTINCTION_SHARED_CAN_SAMPLE_SPECTRUM(context, spectrum_index) medium_extinction_shared_cpu_can_sample_spectrum(context, spectrum_index)
#define ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_INTEGRATED(context, spectrum_index) medium_extinction_shared_cpu_load_spectrum_integrated(context, spectrum_index)
#define ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_SPECTRAL(context, spectrum_index, spect) \
  medium_extinction_shared_cpu_load_spectrum_spectral(context, spectrum_index, spect)
#define ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_RESPONSE_TYPE ::SpectralResponse
#define ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_QUERY_TYPE ::SpectralQuery
#define ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ZERO(spect) ::spectral_response_zero(spect)
#define ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ADD(a, b) ::spectral_response_add(a, b)
#include <etx/render/interop/medium_extinction_shared.hxx>
#undef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ADD
#undef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_ZERO
#undef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_QUERY_TYPE
#undef ETX_MEDIUM_EXTINCTION_SHARED_SPECTRAL_RESPONSE_TYPE
#undef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_SPECTRAL
#undef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SPECTRUM_INTEGRATED
#undef ETX_MEDIUM_EXTINCTION_SHARED_CAN_SAMPLE_SPECTRUM
#undef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_SCATTERING_INDEX
#undef ETX_MEDIUM_EXTINCTION_SHARED_LOAD_ABSORPTION_INDEX
#undef ETX_MEDIUM_EXTINCTION_SHARED_ACCESS_TYPE
#undef ETX_MEDIUM_EXTINCTION_SHARED_CONTEXT_TYPE

struct MediumSegmentTransmittanceCPUSharedAccess {
  uint32_t medium_class = Medium::Homogeneous;
  BoundingBox bounds = {};
  MediumSpectrumCPUSharedAccess spectrum = {};
};

struct MediumSegmentTransmittanceCPUSharedContext {
  const Scene& scene;
  const Medium& medium;
  Sampler& sampler;
};

ETX_SHARED_INLINE bool medium_segment_transmittance_shared_cpu_has_required_scene_buffers(ETX_IN(MediumSegmentTransmittanceCPUSharedContext, context)) {
  uint32_t medium_descriptor_index = 0u;
  uint32_t spectrums_descriptor_index = (context.scene.spectrums.count > 0u) ? 0u : kInvalidIndex;
  return scene_resource_shared_has_medium_spectrum_buffers(medium_descriptor_index, spectrums_descriptor_index);
}

ETX_SHARED_INLINE bool medium_segment_transmittance_shared_cpu_try_load_access(
  ETX_IN(MediumSegmentTransmittanceCPUSharedContext, context), uint32_t medium_index, ETX_OUT(MediumSegmentTransmittanceCPUSharedAccess, access)) {
  (void)medium_index;
  access.medium_class = static_cast<uint32_t>(context.medium.cls);
  access.bounds = context.medium.bounds;
  access.spectrum.absorption_index = context.medium.absorption_index;
  access.spectrum.scattering_index = context.medium.scattering_index;
  return true;
}

ETX_SHARED_INLINE uint32_t medium_segment_transmittance_shared_cpu_load_medium_class(
  ETX_IN(MediumSegmentTransmittanceCPUSharedContext, context), ETX_IN(MediumSegmentTransmittanceCPUSharedAccess, access)) {
  (void)context;
  return access.medium_class;
}

ETX_SHARED_INLINE bool medium_segment_transmittance_shared_cpu_has_grid_data(
  ETX_IN(MediumSegmentTransmittanceCPUSharedContext, context), ETX_IN(MediumSegmentTransmittanceCPUSharedAccess, access)) {
  (void)access;
  return context.medium.has_grid_data();
}

ETX_SHARED_INLINE float3 medium_segment_transmittance_shared_cpu_load_extinction_integrated(
  ETX_IN(MediumSegmentTransmittanceCPUSharedContext, context), ETX_IN(MediumSegmentTransmittanceCPUSharedAccess, access)) {
  MediumSpectrumCPUSharedContext spectrum_context = make_medium_spectrum_shared_context(context.scene);
  return medium_extinction_shared_load_integrated(spectrum_context, access.spectrum);
}

ETX_SHARED_INLINE SpectralResponse medium_segment_transmittance_shared_cpu_load_extinction_spectral(
  ETX_IN(MediumSegmentTransmittanceCPUSharedContext, context), ETX_IN(MediumSegmentTransmittanceCPUSharedAccess, access), ETX_IN(SpectralQuery, spect)) {
  MediumSpectrumCPUSharedContext spectrum_context = make_medium_spectrum_shared_context(context.scene);
  const ::SpectralResponse extinction = medium_extinction_shared_load_spectral(spectrum_context, access.spectrum, static_cast<const ::SpectralQuery&>(spect));
  return medium_transmittance_shared_to_spectral_response(extinction);
}

ETX_SHARED_INLINE MediumTransmittanceSharedContext make_medium_transmittance_shared_context(
  ETX_IN(MediumSegmentTransmittanceCPUSharedContext, context), ETX_IN(MediumSegmentTransmittanceCPUSharedAccess, access)) {
  return make_medium_transmittance_shared_context(context.medium, access.bounds, context.sampler);
}

ETX_SHARED_INLINE float3 medium_segment_transmittance_shared_cpu_transmittance_homogeneous_integrated(ETX_IN(MediumSegmentTransmittanceCPUSharedContext, context),
  ETX_IN(MediumSegmentTransmittanceCPUSharedAccess, access), ETX_IN(float3, extinction), float distance) {
  (void)context;
  (void)access;
  return medium_shared_transmittance_homogeneous_integrated(extinction, distance);
}

ETX_SHARED_INLINE SpectralResponse medium_segment_transmittance_shared_cpu_transmittance_homogeneous_spectral(ETX_IN(MediumSegmentTransmittanceCPUSharedContext, context),
  ETX_IN(MediumSegmentTransmittanceCPUSharedAccess, access), ETX_IN(SpectralResponse, extinction), float distance, ETX_IN(SpectralQuery, spect)) {
  (void)context;
  (void)access;
  (void)spect;
  const ::SpectralResponse transmittance = medium_shared_transmittance_homogeneous_spectral(static_cast<const ::SpectralResponse&>(extinction), distance);
  return medium_transmittance_shared_to_spectral_response(transmittance);
}

ETX_SHARED_INLINE float3 medium_segment_transmittance_shared_cpu_transmittance_heterogeneous_integrated(ETX_INOUT(MediumSegmentTransmittanceCPUSharedContext, context),
  ETX_IN(MediumSegmentTransmittanceCPUSharedAccess, access), ETX_IN(float3, extinction), ETX_IN(float3, origin), ETX_IN(float3, direction), float distance) {
  MediumTransmittanceSharedContext medium_context = make_medium_transmittance_shared_context(context, access);
  return medium_shared_transmittance_heterogeneous_integrated(extinction, origin, direction, distance, access.bounds.p_min, access.bounds.p_max, medium_context);
}

ETX_SHARED_INLINE SpectralResponse medium_segment_transmittance_shared_cpu_transmittance_heterogeneous_spectral(ETX_INOUT(MediumSegmentTransmittanceCPUSharedContext, context),
  ETX_IN(MediumSegmentTransmittanceCPUSharedAccess, access), ETX_IN(SpectralResponse, extinction), ETX_IN(float3, origin), ETX_IN(float3, direction), float distance,
  ETX_IN(SpectralQuery, spect)) {
  MediumTransmittanceSharedContext medium_context = make_medium_transmittance_shared_context(context, access);
  const ::SpectralResponse transmittance = medium_shared_transmittance_heterogeneous_spectral(static_cast<const ::SpectralResponse&>(extinction), origin, direction, distance,
    access.bounds.p_min, access.bounds.p_max, medium_context, static_cast<const ::SpectralQuery&>(spect));
  return medium_transmittance_shared_to_spectral_response(transmittance);
}

ETX_SHARED_INLINE SpectralResponse medium_segment_transmittance_shared_cpu_spectral_one(ETX_IN(SpectralQuery, spect)) {
  return {spect, 1.0f};
}

#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_CONTEXT_TYPE MediumSegmentTransmittanceCPUSharedContext
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_ACCESS_TYPE MediumSegmentTransmittanceCPUSharedAccess
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_RESPONSE_TYPE SpectralResponse
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_QUERY_TYPE SpectralQuery
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) medium_segment_transmittance_shared_cpu_has_required_scene_buffers(context)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRY_LOAD_ACCESS(context, medium_index, access) medium_segment_transmittance_shared_cpu_try_load_access(context, medium_index, access)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_MEDIUM_CLASS(context, access) medium_segment_transmittance_shared_cpu_load_medium_class(context, access)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_GRID_DATA(context, access) medium_segment_transmittance_shared_cpu_has_grid_data(context, access)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_INTEGRATED(context, access) \
  medium_segment_transmittance_shared_cpu_load_extinction_integrated(context, access)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_SPECTRAL(context, access, spect) \
  medium_segment_transmittance_shared_cpu_load_extinction_spectral(context, access, spect)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_INTEGRATED(context, access, extinction, distance) \
  medium_segment_transmittance_shared_cpu_transmittance_homogeneous_integrated(context, access, extinction, distance)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_SPECTRAL(context, access, extinction, distance, spect) \
  medium_segment_transmittance_shared_cpu_transmittance_homogeneous_spectral(context, access, extinction, distance, spect)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_INTEGRATED(context, access, extinction, origin, direction, distance) \
  medium_segment_transmittance_shared_cpu_transmittance_heterogeneous_integrated(context, access, extinction, origin, direction, distance)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_SPECTRAL(context, access, extinction, origin, direction, distance, spect) \
  medium_segment_transmittance_shared_cpu_transmittance_heterogeneous_spectral(context, access, extinction, origin, direction, distance, spect)
#define ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_ONE(spect) medium_segment_transmittance_shared_cpu_spectral_one(spect)
#include <etx/render/interop/medium_segment_transmittance_shared.hxx>
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_ONE
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_SPECTRAL
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HETEROGENEOUS_INTEGRATED
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_SPECTRAL
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRANSMITTANCE_HOMOGENEOUS_INTEGRATED
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_SPECTRAL
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_EXTINCTION_INTEGRATED
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_GRID_DATA
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_LOAD_MEDIUM_CLASS
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_TRY_LOAD_ACCESS
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_HAS_REQUIRED_SCENE_BUFFERS
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_QUERY_TYPE
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_SPECTRAL_RESPONSE_TYPE
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_ACCESS_TYPE
#undef ETX_MEDIUM_SEGMENT_TRANSMITTANCE_SHARED_CONTEXT_TYPE

ETX_SHARED_INLINE uint32_t sample_spectrum_component(const SpectralQuery spect, const SpectralResponse& albedo, const SpectralResponse& throughput, const float rnd,
  SpectralResponse& pdf) {
  ::SpectralResponse shared_pdf = {};
  uint32_t result = medium_sample_shared_sample_spectrum_component(
    static_cast<const ::SpectralQuery&>(spect), static_cast<const ::SpectralResponse&>(albedo), static_cast<const ::SpectralResponse&>(throughput), rnd, shared_pdf);
  pdf = medium_transmittance_shared_to_spectral_response(shared_pdf);
  return result;
}

ETX_SHARED_INLINE SpectralResponse calculate_albedo(const SpectralQuery spect, const SpectralResponse& scattering, const SpectralResponse& extinction) {
  const ::SpectralResponse albedo = medium_sample_shared_calculate_albedo(
    static_cast<const ::SpectralQuery&>(spect), static_cast<const ::SpectralResponse&>(scattering), static_cast<const ::SpectralResponse&>(extinction));
  return medium_transmittance_shared_to_spectral_response(albedo);
}

ETX_SHARED_INLINE float phase_function(const float3& w_i, const float3& w_o, const float g) {
  return medium_phase_shared_henyey_greenstein(w_i, w_o, g);
}

ETX_SHARED_INLINE float3 sample_phase_function(const float3& w_i, const float g, const float2& smp_rnd) {
  return medium_phase_shared_sample_henyey_greenstein(w_i, g, smp_rnd);
}

ETX_SHARED_INLINE SpectralResponse medium_absorption(const Scene& scene, const Medium& medium, const SpectralQuery spect) {
  MediumSpectrumCPUSharedContext spectrum_context = make_medium_spectrum_shared_context(scene);
  MediumSpectrumCPUSharedAccess spectrum_access = make_medium_spectrum_shared_access(medium);
  const ::SpectralResponse response = medium_extinction_shared_load_absorption_spectral(spectrum_context, spectrum_access, static_cast<const ::SpectralQuery&>(spect));
  return medium_transmittance_shared_to_spectral_response(response);
}

ETX_SHARED_INLINE SpectralResponse medium_scattering(const Scene& scene, const Medium& medium, const SpectralQuery spect) {
  MediumSpectrumCPUSharedContext spectrum_context = make_medium_spectrum_shared_context(scene);
  MediumSpectrumCPUSharedAccess spectrum_access = make_medium_spectrum_shared_access(medium);
  const ::SpectralResponse response = medium_extinction_shared_load_scattering_spectral(spectrum_context, spectrum_access, static_cast<const ::SpectralQuery&>(spect));
  return medium_transmittance_shared_to_spectral_response(response);
}

ETX_SHARED_INLINE SpectralResponse medium_extinction(const Scene& scene, const Medium& medium, const SpectralQuery spect) {
  MediumSpectrumCPUSharedContext spectrum_context = make_medium_spectrum_shared_context(scene);
  MediumSpectrumCPUSharedAccess spectrum_access = make_medium_spectrum_shared_access(medium);
  const ::SpectralResponse response = medium_extinction_shared_load_spectral(spectrum_context, spectrum_access, static_cast<const ::SpectralQuery&>(spect));
  return medium_transmittance_shared_to_spectral_response(response);
}

ETX_SHARED_INLINE MediumInstance make_medium_instance(const Scene& scene, const Medium& medium, const SpectralQuery spect, uint32_t index) {
  MediumInstance result = {};
  result.extinction = medium_extinction(scene, medium, spect);
  result.anisotropy = medium.phase_function_g;
  result.index = index;
  return result;
}

ETX_SHARED_INLINE SpectralResponse medium_transmittance(const Scene& scene, const Medium& medium, const SpectralQuery spect, Sampler& smp, const float3& pos,
  const float3& direction, float distance) {
  MediumSegmentTransmittanceCPUSharedContext context = {scene, medium, smp};
  return medium_segment_transmittance_shared_spectral(context, 0u, pos, direction, distance, spect);
}

ETX_SHARED_INLINE MediumSample sample_medium(const Scene& scene, const Medium& medium, const SpectralQuery spect, const SpectralResponse& throughput, Sampler& smp,
  const float3& pos, const float3& w_i, float max_t) {
  ETX_CRITICAL(max_t > 0.0f);

  const SpectralResponse scattering_value = medium_scattering(scene, medium, spect);
  ETX_VALIDATE(scattering_value);
  const SpectralResponse absorption_value = medium_absorption(scene, medium, spect);
  ETX_VALIDATE(absorption_value);

  MediumTransmittanceSharedContext context = make_medium_transmittance_shared_context(medium, medium.bounds, smp);
  return medium_sample_shared_sample(
    context, static_cast<const ::SpectralQuery&>(spect), static_cast<const ::SpectralResponse&>(throughput), static_cast<const ::SpectralResponse&>(scattering_value),
    static_cast<const ::SpectralResponse&>(absorption_value), pos, w_i, max_t);
}

ETX_SHARED_INLINE float medium_phase_function(const Medium& medium, const float3& w_i, const float3& w_o) {
  return phase_function(w_i, w_o, medium.phase_function_g);
}

ETX_SHARED_INLINE float3 medium_sample_phase_function(const Medium& medium, const float2& smp_rnd, const float3& w_i) {
  return sample_phase_function(w_i, medium.phase_function_g, smp_rnd);
}

}  // namespace etx
