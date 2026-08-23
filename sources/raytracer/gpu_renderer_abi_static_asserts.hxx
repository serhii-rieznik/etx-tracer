#pragma once

#include <interop/gpu_wavefront_abi.hxx>

namespace etx {
namespace {
static_assert(std::is_standard_layout_v<GPUWavefrontQueueHeader>, "GPUWavefrontQueueHeader must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontQueueHeader>, "GPUWavefrontQueueHeader must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontQueueHeader) == kGPUWavefrontQueueHeaderSize, "GPUWavefrontQueueHeader size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontQueueHeader, count) == kGPUWavefrontQueueCountOffset, "GPUWavefrontQueueHeader::count offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontQueueHeader, max_path_length) == kGPUWavefrontQueueMaxPathLengthOffset,
  "GPUWavefrontQueueHeader::max_path_length offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontQueueHeader, pad1) == kGPUWavefrontQueuePad1Offset, "GPUWavefrontQueueHeader::pad1 offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontQueueHeader, pad2) == kGPUWavefrontQueuePad2Offset, "GPUWavefrontQueueHeader::pad2 offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontPathState>, "GPUWavefrontPathState must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontPathState>, "GPUWavefrontPathState must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontPathState) == kGPUWavefrontPathStateStride, "GPUWavefrontPathState size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, ray) == kGPUWavefrontPathStateRayOffset, "GPUWavefrontPathState::ray offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, throughput) == kGPUWavefrontPathStateThroughputOffset, "GPUWavefrontPathState::throughput offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, eta) == kGPUWavefrontPathStateEtaOffset, "GPUWavefrontPathState::eta offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, eta_scale) == kGPUWavefrontPathStateEtaScaleOffset, "GPUWavefrontPathState::eta_scale offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, forward_pdf) == kGPUWavefrontPathStateForwardPdfOffset,
  "GPUWavefrontPathState::forward_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, reverse_pdf) == kGPUWavefrontPathStateReversePdfOffset,
  "GPUWavefrontPathState::reverse_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, sampled_bsdf_pdf) == kGPUWavefrontPathStateSampledBsdfPdfOffset,
  "GPUWavefrontPathState::sampled_bsdf_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, last_emitter_pdf) == kGPUWavefrontPathStateLastEmitterPdfOffset,
  "GPUWavefrontPathState::last_emitter_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, medium_index) == kGPUWavefrontPathStateMediumIndexOffset,
  "GPUWavefrontPathState::medium_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, path_length) == kGPUWavefrontPathStatePathLengthOffset,
  "GPUWavefrontPathState::path_length offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, pixel_index) == kGPUWavefrontPathStatePixelIndexOffset,
  "GPUWavefrontPathState::pixel_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, flags) == kGPUWavefrontPathStateFlagsOffset, "GPUWavefrontPathState::flags offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, path_source) == kGPUWavefrontPathStatePathSourceOffset,
  "GPUWavefrontPathState::path_source offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, sampler_seed) == kGPUWavefrontPathStateSamplerSeedOffset,
  "GPUWavefrontPathState::sampler_seed offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, pixel) == kGPUWavefrontPathStatePixelOffset, "GPUWavefrontPathState::pixel offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, spect) == kGPUWavefrontPathStateSpectralQueryOffset, "GPUWavefrontPathState::spect offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, film_uv) == kGPUWavefrontPathStateFilmUvOffset, "GPUWavefrontPathState::film_uv offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, last_vertex_index) == kGPUWavefrontPathStateLastVertexIndexOffset,
  "GPUWavefrontPathState::last_vertex_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, d_vm) == kGPUWavefrontPathStateDVmOffset, "GPUWavefrontPathState::d_vm offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathState, reserved0) == kGPUWavefrontPathStateReserved0Offset, "GPUWavefrontPathState::reserved0 offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontHit>, "GPUWavefrontHit must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontHit>, "GPUWavefrontHit must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontHit) == kGPUWavefrontHitStride, "GPUWavefrontHit size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontHit, transmittance) == kGPUWavefrontHitTransmittanceOffset, "GPUWavefrontHit::transmittance offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontHit, vertex) == kGPUWavefrontHitVertexOffset, "GPUWavefrontHit::vertex offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontHit, geo_normal) == kGPUWavefrontHitGeoNormalOffset, "GPUWavefrontHit::geo_normal offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontHit, hit_t) == kGPUWavefrontHitHitTOffset, "GPUWavefrontHit::hit_t offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontHit, triangle_index) == kGPUWavefrontHitTriangleIndexOffset, "GPUWavefrontHit::triangle_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontHit, material_index) == kGPUWavefrontHitMaterialIndexOffset, "GPUWavefrontHit::material_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontHit, emitter_index) == kGPUWavefrontHitEmitterIndexOffset, "GPUWavefrontHit::emitter_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontHit, medium_index) == kGPUWavefrontHitMediumIndexOffset, "GPUWavefrontHit::medium_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontHit, flags) == kGPUWavefrontHitFlagsOffset, "GPUWavefrontHit::flags offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontHit, barycentric) == kGPUWavefrontHitBarycentricOffset, "GPUWavefrontHit::barycentric offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontHit, instance_index) == kGPUWavefrontHitInstanceIndexOffset, "GPUWavefrontHit::instance_index offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontPathVertex>, "GPUWavefrontPathVertex must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontPathVertex>, "GPUWavefrontPathVertex must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontPathVertex) == kGPUWavefrontPathVertexStride, "GPUWavefrontPathVertex size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, throughput) == kGPUWavefrontPathVertexThroughputOffset,
  "GPUWavefrontPathVertex::throughput offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, position) == kGPUWavefrontPathVertexPositionOffset, "GPUWavefrontPathVertex::position offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, triangle_index) == kGPUWavefrontPathVertexTriangleIndexOffset,
  "GPUWavefrontPathVertex::triangle_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, normal) == kGPUWavefrontPathVertexNormalOffset, "GPUWavefrontPathVertex::normal offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, material_index) == kGPUWavefrontPathVertexMaterialIndexOffset,
  "GPUWavefrontPathVertex::material_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, geo_normal) == kGPUWavefrontPathVertexGeoNormalOffset,
  "GPUWavefrontPathVertex::geo_normal offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, medium_index) == kGPUWavefrontPathVertexMediumIndexOffset,
  "GPUWavefrontPathVertex::medium_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, w_i) == kGPUWavefrontPathVertexWiOffset, "GPUWavefrontPathVertex::w_i offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, emitter_index) == kGPUWavefrontPathVertexEmitterIndexOffset,
  "GPUWavefrontPathVertex::emitter_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, texcoord) == kGPUWavefrontPathVertexTexcoordOffset, "GPUWavefrontPathVertex::texcoord offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, forward_pdf) == kGPUWavefrontPathVertexForwardPdfOffset,
  "GPUWavefrontPathVertex::forward_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, reverse_pdf) == kGPUWavefrontPathVertexReversePdfOffset,
  "GPUWavefrontPathVertex::reverse_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, sampled_bsdf_pdf) == kGPUWavefrontPathVertexSampledBsdfPdfOffset,
  "GPUWavefrontPathVertex::sampled_bsdf_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, eta_scale) == kGPUWavefrontPathVertexEtaScaleOffset, "GPUWavefrontPathVertex::eta_scale offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, path_length) == kGPUWavefrontPathVertexPathLengthOffset,
  "GPUWavefrontPathVertex::path_length offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, pixel_index) == kGPUWavefrontPathVertexPixelIndexOffset,
  "GPUWavefrontPathVertex::pixel_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, flags) == kGPUWavefrontPathVertexFlagsOffset, "GPUWavefrontPathVertex::flags offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, pdf_from_prev) == kGPUWavefrontPathVertexPdfFromPrevOffset,
  "GPUWavefrontPathVertex::pdf_from_prev offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, pdf_from_next) == kGPUWavefrontPathVertexPdfFromNextOffset,
  "GPUWavefrontPathVertex::pdf_from_next offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, pdf_accumulated) == kGPUWavefrontPathVertexPdfAccumulatedOffset,
  "GPUWavefrontPathVertex::pdf_accumulated offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, pdf_history) == kGPUWavefrontPathVertexPdfHistoryOffset,
  "GPUWavefrontPathVertex::pdf_history offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, pdf_ratio) == kGPUWavefrontPathVertexPdfRatioOffset, "GPUWavefrontPathVertex::pdf_ratio offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, barycentric) == kGPUWavefrontPathVertexBarycentricOffset,
  "GPUWavefrontPathVertex::barycentric offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, instance_index) == kGPUWavefrontPathVertexInstanceIndexOffset,
  "GPUWavefrontPathVertex::instance_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, reserved0) == kGPUWavefrontPathVertexReserved0Offset,
  "GPUWavefrontPathVertex::reserved0 offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathVertex, d_vm) == kGPUWavefrontPathVertexDVmOffset, "GPUWavefrontPathVertex::d_vm offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontLightPathVertex>, "GPUWavefrontLightPathVertex must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontLightPathVertex>, "GPUWavefrontLightPathVertex must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontLightPathVertex) == kGPUWavefrontLightPathVertexStride, "GPUWavefrontLightPathVertex size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, throughput) == kGPUWavefrontLightPathVertexThroughputOffset,
  "GPUWavefrontLightPathVertex::throughput offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, inline_medium_extinction) == kGPUWavefrontLightPathVertexInlineMediumExtinctionOffset,
  "GPUWavefrontLightPathVertex::inline_medium_extinction offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, position) == kGPUWavefrontLightPathVertexPositionOffset,
  "GPUWavefrontLightPathVertex::position offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, triangle_index) == kGPUWavefrontLightPathVertexTriangleIndexOffset,
  "GPUWavefrontLightPathVertex::triangle_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, normal) == kGPUWavefrontLightPathVertexNormalOffset,
  "GPUWavefrontLightPathVertex::normal offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, material_index) == kGPUWavefrontLightPathVertexMaterialIndexOffset,
  "GPUWavefrontLightPathVertex::material_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, geo_normal) == kGPUWavefrontLightPathVertexGeoNormalOffset,
  "GPUWavefrontLightPathVertex::geo_normal offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, medium_index) == kGPUWavefrontLightPathVertexMediumIndexOffset,
  "GPUWavefrontLightPathVertex::medium_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, w_i) == kGPUWavefrontLightPathVertexWiOffset, "GPUWavefrontLightPathVertex::w_i offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, emitter_index) == kGPUWavefrontLightPathVertexEmitterIndexOffset,
  "GPUWavefrontLightPathVertex::emitter_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, texcoord) == kGPUWavefrontLightPathVertexTexcoordOffset,
  "GPUWavefrontLightPathVertex::texcoord offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, forward_pdf) == kGPUWavefrontLightPathVertexForwardPdfOffset,
  "GPUWavefrontLightPathVertex::forward_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, reverse_pdf) == kGPUWavefrontLightPathVertexReversePdfOffset,
  "GPUWavefrontLightPathVertex::reverse_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, sampled_bsdf_pdf) == kGPUWavefrontLightPathVertexSampledBsdfPdfOffset,
  "GPUWavefrontLightPathVertex::sampled_bsdf_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, path_length) == kGPUWavefrontLightPathVertexPathLengthOffset,
  "GPUWavefrontLightPathVertex::path_length offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, flags) == kGPUWavefrontLightPathVertexFlagsOffset,
  "GPUWavefrontLightPathVertex::flags offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, pdf_from_prev) == kGPUWavefrontLightPathVertexPdfFromPrevOffset,
  "GPUWavefrontLightPathVertex::pdf_from_prev offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, pdf_from_next) == kGPUWavefrontLightPathVertexPdfFromNextOffset,
  "GPUWavefrontLightPathVertex::pdf_from_next offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, pdf_accumulated) == kGPUWavefrontLightPathVertexPdfAccumulatedOffset,
  "GPUWavefrontLightPathVertex::pdf_accumulated offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, pdf_history) == kGPUWavefrontLightPathVertexPdfHistoryOffset,
  "GPUWavefrontLightPathVertex::pdf_history offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, pdf_ratio) == kGPUWavefrontLightPathVertexPdfRatioOffset,
  "GPUWavefrontLightPathVertex::pdf_ratio offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, barycentric) == kGPUWavefrontLightPathVertexBarycentricOffset,
  "GPUWavefrontLightPathVertex::barycentric offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, previous_vertex_index) == kGPUWavefrontLightPathVertexPreviousVertexIndexOffset,
  "GPUWavefrontLightPathVertex::previous_vertex_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, instance_index) == kGPUWavefrontLightPathVertexInstanceIndexOffset,
  "GPUWavefrontLightPathVertex::instance_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontLightPathVertex, d_vm) == kGPUWavefrontLightPathVertexDVmOffset,
  "GPUWavefrontLightPathVertex::d_vm offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontFastLightEndpoint>, "GPUWavefrontFastLightEndpoint must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontFastLightEndpoint>, "GPUWavefrontFastLightEndpoint must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontFastLightEndpoint) == kGPUWavefrontFastLightEndpointStride, "GPUWavefrontFastLightEndpoint size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontFastLightEndpoint, emitter_pdf_from_prev) == kGPUWavefrontFastLightEndpointEmitterPdfFromPrevOffset,
  "GPUWavefrontFastLightEndpoint::emitter_pdf_from_prev offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontFastLightEndpoint, emitter_pdf_from_next) == kGPUWavefrontFastLightEndpointEmitterPdfFromNextOffset,
  "GPUWavefrontFastLightEndpoint::emitter_pdf_from_next offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontFastLightEndpoint, emitter_flags) == kGPUWavefrontFastLightEndpointEmitterFlagsOffset,
  "GPUWavefrontFastLightEndpoint::emitter_flags offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontFastLightEndpoint, first_vertex_flags) == kGPUWavefrontFastLightEndpointFirstVertexFlagsOffset,
  "GPUWavefrontFastLightEndpoint::first_vertex_flags offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontPathMeta>, "GPUWavefrontPathMeta must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontPathMeta>, "GPUWavefrontPathMeta must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontPathMeta) == kGPUWavefrontPathMetaStride, "GPUWavefrontPathMeta size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathMeta, camera_path_length) == kGPUWavefrontPathMetaCameraPathLengthOffset,
  "GPUWavefrontPathMeta::camera_path_length offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathMeta, light_path_length) == kGPUWavefrontPathMetaLightPathLengthOffset,
  "GPUWavefrontPathMeta::light_path_length offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathMeta, flags) == kGPUWavefrontPathMetaFlagsOffset, "GPUWavefrontPathMeta::flags offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathMeta, reserved0) == kGPUWavefrontPathMetaReserved0Offset, "GPUWavefrontPathMeta::reserved0 offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathMeta, camera_mis_history) == kGPUWavefrontPathMetaCameraMisHistoryOffset,
  "GPUWavefrontPathMeta::camera_mis_history offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathMeta, light_mis_history) == kGPUWavefrontPathMetaLightMisHistoryOffset,
  "GPUWavefrontPathMeta::light_mis_history offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathMeta, from_delta) == kGPUWavefrontPathMetaFromDeltaOffset, "GPUWavefrontPathMeta::from_delta offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontPathMeta, reserved1) == kGPUWavefrontPathMetaReserved1Offset, "GPUWavefrontPathMeta::reserved1 offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontSubsurfaceState>, "GPUWavefrontSubsurfaceState must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontSubsurfaceState>, "GPUWavefrontSubsurfaceState must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontSubsurfaceState) == kGPUWavefrontSubsurfaceStateStride, "GPUWavefrontSubsurfaceState size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontSubsurfaceState, extinction) == kGPUWavefrontSubsurfaceStateExtinctionOffset,
  "GPUWavefrontSubsurfaceState::extinction offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontSubsurfaceState, scattering) == kGPUWavefrontSubsurfaceStateScatteringOffset,
  "GPUWavefrontSubsurfaceState::scattering offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontSubsurfaceState, albedo) == kGPUWavefrontSubsurfaceStateAlbedoOffset,
  "GPUWavefrontSubsurfaceState::albedo offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontSubsurfaceState, material_index) == kGPUWavefrontSubsurfaceStateMaterialIndexOffset,
  "GPUWavefrontSubsurfaceState::material_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontSubsurfaceState, medium_index) == kGPUWavefrontSubsurfaceStateMediumIndexOffset,
  "GPUWavefrontSubsurfaceState::medium_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontSubsurfaceState, scatter_material_index) == kGPUWavefrontSubsurfaceStateScatterMaterialIndexOffset,
  "GPUWavefrontSubsurfaceState::scatter_material_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontSubsurfaceState, flags) == kGPUWavefrontSubsurfaceStateFlagsOffset,
  "GPUWavefrontSubsurfaceState::flags offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontSubsurfaceState, phase_function_g) == kGPUWavefrontSubsurfaceStatePhaseFunctionGOffset,
  "GPUWavefrontSubsurfaceState::phase_function_g offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontDirectLightSample>, "GPUWavefrontDirectLightSample must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontDirectLightSample>, "GPUWavefrontDirectLightSample must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontDirectLightSample) == kGPUWavefrontDirectLightSampleStride, "GPUWavefrontDirectLightSample size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, value) == kGPUWavefrontDirectLightSampleValueOffset,
  "GPUWavefrontDirectLightSample::value offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, origin) == kGPUWavefrontDirectLightSampleOriginOffset,
  "GPUWavefrontDirectLightSample::origin offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, pdf_sample) == kGPUWavefrontDirectLightSamplePdfSampleOffset,
  "GPUWavefrontDirectLightSample::pdf_sample offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, direction) == kGPUWavefrontDirectLightSampleDirectionOffset,
  "GPUWavefrontDirectLightSample::direction offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, pdf_area) == kGPUWavefrontDirectLightSamplePdfAreaOffset,
  "GPUWavefrontDirectLightSample::pdf_area offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, normal) == kGPUWavefrontDirectLightSampleNormalOffset,
  "GPUWavefrontDirectLightSample::normal offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, pdf_dir) == kGPUWavefrontDirectLightSamplePdfDirOffset,
  "GPUWavefrontDirectLightSample::pdf_dir offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, texcoord) == kGPUWavefrontDirectLightSampleTexcoordOffset,
  "GPUWavefrontDirectLightSample::texcoord offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, emitter_index) == kGPUWavefrontDirectLightSampleEmitterIndexOffset,
  "GPUWavefrontDirectLightSample::emitter_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, triangle_index) == kGPUWavefrontDirectLightSampleTriangleIndexOffset,
  "GPUWavefrontDirectLightSample::triangle_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, flags) == kGPUWavefrontDirectLightSampleFlagsOffset,
  "GPUWavefrontDirectLightSample::flags offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightSample, pdf_dir_out) == kGPUWavefrontDirectLightSamplePdfDirOutOffset,
  "GPUWavefrontDirectLightSample::pdf_dir_out offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontDirectLightTask>, "GPUWavefrontDirectLightTask must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontDirectLightTask>, "GPUWavefrontDirectLightTask must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontDirectLightTask) == kGPUWavefrontDirectLightTaskStride, "GPUWavefrontDirectLightTask size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightTask, shadow_ray) == kGPUWavefrontDirectLightTaskShadowRayOffset,
  "GPUWavefrontDirectLightTask::shadow_ray offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightTask, shadow_target) == kGPUWavefrontDirectLightTaskShadowTargetOffset,
  "GPUWavefrontDirectLightTask::shadow_target offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightTask, contribution) == kGPUWavefrontDirectLightTaskContributionOffset,
  "GPUWavefrontDirectLightTask::contribution offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightTask, mis_weight) == kGPUWavefrontDirectLightTaskMisWeightOffset,
  "GPUWavefrontDirectLightTask::mis_weight offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightTask, pixel_index) == kGPUWavefrontDirectLightTaskPixelIndexOffset,
  "GPUWavefrontDirectLightTask::pixel_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightTask, medium_index) == kGPUWavefrontDirectLightTaskMediumIndexOffset,
  "GPUWavefrontDirectLightTask::medium_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightTask, flags) == kGPUWavefrontDirectLightTaskFlagsOffset,
  "GPUWavefrontDirectLightTask::flags offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightTask, path_index) == kGPUWavefrontDirectLightTaskPathIndexOffset,
  "GPUWavefrontDirectLightTask::path_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightTask, sampler_seed) == kGPUWavefrontDirectLightTaskSamplerSeedOffset,
  "GPUWavefrontDirectLightTask::sampler_seed offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontDirectLightResult>, "GPUWavefrontDirectLightResult must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontDirectLightResult>, "GPUWavefrontDirectLightResult must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontDirectLightResult) == kGPUWavefrontDirectLightResultStride, "GPUWavefrontDirectLightResult size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightResult, transmittance) == kGPUWavefrontDirectLightResultTransmittanceOffset,
  "GPUWavefrontDirectLightResult::transmittance offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontDirectLightResult, visible) == kGPUWavefrontDirectLightResultVisibleOffset,
  "GPUWavefrontDirectLightResult::visible offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontConnectLightCandidate>, "GPUWavefrontConnectLightCandidate must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontConnectLightCandidate>, "GPUWavefrontConnectLightCandidate must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontConnectLightCandidate) == 64u, "GPUWavefrontConnectLightCandidate size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightCandidate, camera_contribution) == kGPUWavefrontConnectLightCandidateCameraContributionOffset,
  "GPUWavefrontConnectLightCandidate::camera_contribution offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightCandidate, camera_pdf) == kGPUWavefrontConnectLightCandidateCameraPdfOffset,
  "GPUWavefrontConnectLightCandidate::camera_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightCandidate, camera_reverse_area_pdf) == kGPUWavefrontConnectLightCandidateCameraReverseAreaPdfOffset,
  "GPUWavefrontConnectLightCandidate::camera_reverse_area_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightCandidate, camera_reverse_direction_pdf) == kGPUWavefrontConnectLightCandidateCameraReverseDirectionPdfOffset,
  "GPUWavefrontConnectLightCandidate::camera_reverse_direction_pdf offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightCandidate, light_vertex_index) == kGPUWavefrontConnectLightCandidateLightVertexIndexOffset,
  "GPUWavefrontConnectLightCandidate::light_vertex_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightCandidate, previous_light_vertex_index) == kGPUWavefrontConnectLightCandidatePreviousLightVertexIndexOffset,
  "GPUWavefrontConnectLightCandidate::previous_light_vertex_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightCandidate, flags) == kGPUWavefrontConnectLightCandidateFlagsOffset,
  "GPUWavefrontConnectLightCandidate::flags offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightCandidate, sampler_seed) == kGPUWavefrontConnectLightCandidateSamplerSeedOffset,
  "GPUWavefrontConnectLightCandidate::sampler_seed offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontConnectLightTask>, "GPUWavefrontConnectLightTask must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontConnectLightTask>, "GPUWavefrontConnectLightTask must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontConnectLightTask) == kGPUWavefrontConnectLightTaskStride, "GPUWavefrontConnectLightTask size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightTask, shadow_origin) == kGPUWavefrontConnectLightTaskShadowOriginOffset,
  "GPUWavefrontConnectLightTask::shadow_origin offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightTask, shadow_target) == kGPUWavefrontConnectLightTaskShadowTargetOffset,
  "GPUWavefrontConnectLightTask::shadow_target offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightTask, contribution) == kGPUWavefrontConnectLightTaskContributionOffset,
  "GPUWavefrontConnectLightTask::contribution offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightTask, pixel_index) == kGPUWavefrontConnectLightTaskPixelIndexOffset,
  "GPUWavefrontConnectLightTask::pixel_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightTask, medium_index) == kGPUWavefrontConnectLightTaskMediumIndexOffset,
  "GPUWavefrontConnectLightTask::medium_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightTask, sampler_seed) == kGPUWavefrontConnectLightTaskSamplerSeedOffset,
  "GPUWavefrontConnectLightTask::sampler_seed offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightTask, inline_medium_extinction) == kGPUWavefrontConnectLightTaskInlineMediumExtinctionOffset,
  "GPUWavefrontConnectLightTask::inline_medium_extinction offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectLightTask, inline_medium_flags) == kGPUWavefrontConnectLightTaskInlineMediumFlagsOffset,
  "GPUWavefrontConnectLightTask::inline_medium_flags offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontConnectCameraTask>, "GPUWavefrontConnectCameraTask must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontConnectCameraTask>, "GPUWavefrontConnectCameraTask must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontConnectCameraTask) == kGPUWavefrontConnectCameraTaskStride, "GPUWavefrontConnectCameraTask size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectCameraTask, shadow_ray) == kGPUWavefrontConnectCameraTaskShadowRayOffset,
  "GPUWavefrontConnectCameraTask::shadow_ray offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectCameraTask, shadow_target) == kGPUWavefrontConnectCameraTaskShadowTargetOffset,
  "GPUWavefrontConnectCameraTask::shadow_target offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectCameraTask, contribution) == kGPUWavefrontConnectCameraTaskContributionOffset,
  "GPUWavefrontConnectCameraTask::contribution offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectCameraTask, mis_weight) == kGPUWavefrontConnectCameraTaskMisWeightOffset,
  "GPUWavefrontConnectCameraTask::mis_weight offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectCameraTask, pixel_index) == kGPUWavefrontConnectCameraTaskPixelIndexOffset,
  "GPUWavefrontConnectCameraTask::pixel_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectCameraTask, medium_index) == kGPUWavefrontConnectCameraTaskMediumIndexOffset,
  "GPUWavefrontConnectCameraTask::medium_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectCameraTask, flags) == kGPUWavefrontConnectCameraTaskFlagsOffset,
  "GPUWavefrontConnectCameraTask::flags offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectCameraTask, path_index) == kGPUWavefrontConnectCameraTaskPathIndexOffset,
  "GPUWavefrontConnectCameraTask::path_index offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectCameraTask, sampler_seed) == kGPUWavefrontConnectCameraTaskSamplerSeedOffset,
  "GPUWavefrontConnectCameraTask::sampler_seed offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontConnectCameraResult>, "GPUWavefrontConnectCameraResult must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontConnectCameraResult>, "GPUWavefrontConnectCameraResult must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontConnectCameraResult) == kGPUWavefrontConnectCameraResultStride, "GPUWavefrontConnectCameraResult size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectCameraResult, transmittance) == kGPUWavefrontConnectCameraResultTransmittanceOffset,
  "GPUWavefrontConnectCameraResult::transmittance offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontConnectCameraResult, visible) == kGPUWavefrontConnectCameraResultVisibleOffset,
  "GPUWavefrontConnectCameraResult::visible offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<GPUWavefrontResources>, "GPUWavefrontResources must stay standard layout for GPU wavefront ABI");
static_assert(std::is_trivially_copyable_v<GPUWavefrontResources>, "GPUWavefrontResources must stay trivially copyable for GPU wavefront ABI");
static_assert(sizeof(GPUWavefrontResources) == kGPUWavefrontResourcesStride, "GPUWavefrontResources size changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, camera_state_buffer) == kGPUWavefrontResourcesCameraStateBufferOffset,
  "GPUWavefrontResources::camera_state_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, light_state_buffer) == kGPUWavefrontResourcesLightStateBufferOffset,
  "GPUWavefrontResources::light_state_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, camera_hit_buffer) == kGPUWavefrontResourcesCameraHitBufferOffset,
  "GPUWavefrontResources::camera_hit_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, light_hit_buffer) == kGPUWavefrontResourcesLightHitBufferOffset,
  "GPUWavefrontResources::light_hit_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, camera_queue_a_buffer) == kGPUWavefrontResourcesCameraQueueABufferOffset,
  "GPUWavefrontResources::camera_queue_a_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, camera_queue_b_buffer) == kGPUWavefrontResourcesCameraQueueBBufferOffset,
  "GPUWavefrontResources::camera_queue_b_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, light_queue_a_buffer) == kGPUWavefrontResourcesLightQueueABufferOffset,
  "GPUWavefrontResources::light_queue_a_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, light_queue_b_buffer) == kGPUWavefrontResourcesLightQueueBBufferOffset,
  "GPUWavefrontResources::light_queue_b_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, camera_vertex_buffer) == kGPUWavefrontResourcesCameraVertexBufferOffset,
  "GPUWavefrontResources::camera_vertex_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, light_vertex_buffer) == kGPUWavefrontResourcesLightVertexBufferOffset,
  "GPUWavefrontResources::light_vertex_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, film_buffer) == kGPUWavefrontResourcesFilmBufferOffset,
  "GPUWavefrontResources::film_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, path_meta_buffer) == kGPUWavefrontResourcesPathMetaBufferOffset,
  "GPUWavefrontResources::path_meta_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, direct_light_sample_buffer) == kGPUWavefrontResourcesDirectLightSampleBufferOffset,
  "GPUWavefrontResources::direct_light_sample_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, direct_light_task_buffer) == kGPUWavefrontResourcesDirectLightTaskBufferOffset,
  "GPUWavefrontResources::direct_light_task_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, direct_light_result_buffer) == kGPUWavefrontResourcesDirectLightResultBufferOffset,
  "GPUWavefrontResources::direct_light_result_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, connect_light_task_buffer) == kGPUWavefrontResourcesConnectLightTaskBufferOffset,
  "GPUWavefrontResources::connect_light_task_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, connect_light_result_buffer) == kGPUWavefrontResourcesConnectLightResultBufferOffset,
  "GPUWavefrontResources::connect_light_result_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, connect_camera_task_buffer) == kGPUWavefrontResourcesConnectCameraTaskBufferOffset,
  "GPUWavefrontResources::connect_camera_task_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, connect_camera_result_buffer) == kGPUWavefrontResourcesConnectCameraResultBufferOffset,
  "GPUWavefrontResources::connect_camera_result_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, camera_subsurface_state_buffer) == kGPUWavefrontResourcesCameraSubsurfaceStateBufferOffset,
  "GPUWavefrontResources::camera_subsurface_state_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, light_subsurface_state_buffer) == kGPUWavefrontResourcesLightSubsurfaceStateBufferOffset,
  "GPUWavefrontResources::light_subsurface_state_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, path_capacity) == kGPUWavefrontResourcesPathCapacityOffset,
  "GPUWavefrontResources::path_capacity offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, max_path_length) == kGPUWavefrontResourcesMaxPathLengthOffset,
  "GPUWavefrontResources::max_path_length offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, camera_vertex_capacity) == kGPUWavefrontResourcesCameraVertexCapacityOffset,
  "GPUWavefrontResources::camera_vertex_capacity offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, light_vertex_capacity) == kGPUWavefrontResourcesLightVertexCapacityOffset,
  "GPUWavefrontResources::light_vertex_capacity offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, camera_fixed_max_bounces) == kGPUWavefrontResourcesCameraFixedMaxBouncesOffset,
  "GPUWavefrontResources::camera_fixed_max_bounces offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, light_fixed_max_bounces) == kGPUWavefrontResourcesLightFixedMaxBouncesOffset,
  "GPUWavefrontResources::light_fixed_max_bounces offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, dispatch_args_buffer) == kGPUWavefrontResourcesDispatchArgsBufferOffset,
  "GPUWavefrontResources::dispatch_args_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, material_queue_buffer) == kGPUWavefrontResourcesMaterialQueueBufferOffset,
  "GPUWavefrontResources::material_queue_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, shadow_queue_buffer) == kGPUWavefrontResourcesShadowQueueBufferOffset,
  "GPUWavefrontResources::shadow_queue_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, light_vertex_counter_buffer) == kGPUWavefrontResourcesLightVertexCounterBufferOffset,
  "GPUWavefrontResources::light_vertex_counter_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, fast_light_endpoint_buffer) == kGPUWavefrontResourcesFastLightEndpointBufferOffset,
  "GPUWavefrontResources::fast_light_endpoint_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, vcm_grid_heads_buffer) == kGPUWavefrontResourcesVCMGridHeadsBufferOffset,
  "GPUWavefrontResources::vcm_grid_heads_buffer offset changed; update GPU wavefront ABI");
static_assert(offsetof(GPUWavefrontResources, vcm_grid_next_buffer) == kGPUWavefrontResourcesVCMGridNextBufferOffset,
  "GPUWavefrontResources::vcm_grid_next_buffer offset changed; update GPU wavefront ABI");

static_assert(std::is_standard_layout_v<float2>, "float2 must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<float2>, "float2 must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(float2) == 8u, "float2 size changed; update GPU upload ABI");

static_assert(std::is_standard_layout_v<float3>, "float3 must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<float3>, "float3 must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(float3) == 12u, "float3 size changed; update GPU upload ABI");

static_assert(std::is_standard_layout_v<Triangle>, "Triangle must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Triangle>, "Triangle must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Triangle) == kTriangleStride, "Triangle size changed; update GPU shader decode stride");
static_assert(offsetof(Triangle, i) == 0u, "Triangle::i offset changed; update GPU shader decode");
static_assert(offsetof(Triangle, material_index) == 12u, "Triangle::material_index offset changed; update GPU shader decode");
static_assert(offsetof(Triangle, geo_n) == 16u, "Triangle::geo_n offset changed; update GPU shader decode");
static_assert(offsetof(Triangle, emitter_index) == 28u, "Triangle::emitter_index offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<Mesh>, "Mesh must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Mesh>, "Mesh must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Mesh) == 32u, "Mesh size changed; update GPU upload ABI");

static_assert(std::is_standard_layout_v<EmitterProfile>, "EmitterProfile must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<EmitterProfile>, "EmitterProfile must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(EmitterProfile) == kEmitterProfileStride, "EmitterProfile size changed; update GPU upload ABI");
static_assert(offsetof(EmitterProfile, emission) == kEmitterProfileEmissionSpectrumIndexOffset, "EmitterProfile::emission offset changed; update GPU shader decode");
static_assert(offsetof(EmitterProfile, cls) == kEmitterProfileClassOffset, "EmitterProfile::cls offset changed; update GPU shader decode");
static_assert((offsetof(EmitterProfile, directional) + offsetof(EmitterProfile::DirectionalData, direction)) == kEmitterProfileDirectionalDirectionOffset,
  "EmitterProfile::directional.direction offset changed; update GPU shader decode");
static_assert((offsetof(EmitterProfile, directional) + offsetof(EmitterProfile::DirectionalData, angular_size)) == kEmitterProfileDirectionalAngularSizeOffset,
  "EmitterProfile::directional.angular_size offset changed; update GPU shader decode");
static_assert((offsetof(EmitterProfile, directional) + offsetof(EmitterProfile::DirectionalData, angular_size_cosine)) == kEmitterProfileDirectionalAngularSizeCosineOffset,
  "EmitterProfile::directional.angular_size_cosine offset changed; update GPU shader decode");
static_assert(offsetof(EmitterProfile, medium_index) == kEmitterProfileMediumIndexOffset, "EmitterProfile::medium_index offset changed; update GPU shader decode");
static_assert(offsetof(EmitterProfile, meta) == kEmitterProfileMetaOffset, "EmitterProfile::meta offset changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(EmitterProfile::Class::Area) == EmitterClass::Area, "EmitterProfile::Class::Area changed; update GPU shader decode");
static_assert(EmitterProfile::Meta::Atmosphere == EmitterProfileMeta::Atmosphere, "EmitterProfile::Meta::Atmosphere changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<Emitter>, "Emitter must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Emitter>, "Emitter must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Emitter) == kEmitterStride, "Emitter size changed; update GPU upload ABI");
static_assert(offsetof(Emitter, cls) == kEmitterClassOffset, "Emitter::cls offset changed; update GPU shader decode");
static_assert(offsetof(Emitter, profile) == kEmitterProfileOffset, "Emitter::profile offset changed; update GPU shader decode");
static_assert(offsetof(Emitter, triangle_index) == kEmitterTriangleIndexOffset, "Emitter::triangle_index offset changed; update GPU shader decode");
static_assert(offsetof(Emitter, spectrum_weight) == kEmitterSpectrumWeightOffset, "Emitter::spectrum_weight offset changed; update GPU shader decode");
static_assert(offsetof(Emitter, additional_weight) == kEmitterAdditionalWeightOffset, "Emitter::additional_weight offset changed; update GPU shader decode");
static_assert(offsetof(Emitter, triangle_area) == kEmitterTriangleAreaOffset, "Emitter::triangle_area offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<Material>, "Material must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Material>, "Material must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Material) == kMaterialStride, "Material size changed; update GPU shader decode stride");
static_assert((offsetof(Material, diffraction_grating) + offsetof(DiffractionGrating, period_nm)) == kMaterialDiffractionGratingPeriodNmOffset,
  "Material::diffraction_grating.period_nm offset changed; update GPU shader decode");
static_assert((offsetof(Material, diffraction_grating) + offsetof(DiffractionGrating, optical_path_difference_nm)) == kMaterialDiffractionGratingOpticalPathDifferenceNmOffset,
  "Material::diffraction_grating.optical_path_difference_nm offset changed; update GPU shader decode");
static_assert((offsetof(Material, diffraction_grating) + offsetof(DiffractionGrating, duty_cycle)) == kMaterialDiffractionGratingDutyCycleOffset,
  "Material::diffraction_grating.duty_cycle offset changed; update GPU shader decode");
static_assert((offsetof(Material, diffraction_grating) + offsetof(DiffractionGrating, rotation)) == kMaterialDiffractionGratingRotationOffset,
  "Material::diffraction_grating.rotation offset changed; update GPU shader decode");
static_assert(offsetof(Material, scattering) == kMaterialScatteringSpectrumIndexOffset, "Material::scattering offset changed; update GPU shader decode");
static_assert((offsetof(Material, scattering) + offsetof(SpectralImage, image_index)) == kMaterialScatteringImageIndexOffset,
  "Material::scattering.image_index offset changed; update GPU shader decode");
static_assert(offsetof(Material, cls) == kMaterialClassOffset, "Material::cls offset changed; update GPU shader decode");
static_assert(offsetof(Material, int_medium) == kMaterialIntMediumOffset, "Material::int_medium offset changed; update GPU shader decode");
static_assert(offsetof(Material, ext_medium) == kMaterialExtMediumOffset, "Material::ext_medium offset changed; update GPU shader decode");
static_assert(offsetof(Material, opacity) == kMaterialOpacityOffset, "Material::opacity offset changed; update GPU shader decode");
static_assert(offsetof(Material, emission_collimation) == kMaterialEmissionCollimationOffset, "Material::emission_collimation offset changed; update GPU shader decode");
static_assert(offsetof(Material, energy_compensation_interface_index) == kMaterialEnergyCompensationInterfaceIndexOffset,
  "Material::energy_compensation_interface_index offset changed; update GPU shader decode");
static_assert(offsetof(Material, conductor_energy_compensation_interface_index) == kMaterialConductorEnergyCompensationInterfaceIndexOffset,
  "Material::conductor_energy_compensation_interface_index offset changed; update GPU shader decode");
static_assert(std::is_standard_layout_v<SpectralImage>, "SpectralImage must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<SpectralImage>, "SpectralImage must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(SpectralImage) == 16u, "SpectralImage size changed; update material ABI");
static_assert(offsetof(SpectralImage, spectrum_index) == 0u, "SpectralImage::spectrum_index offset changed; update material ABI");
static_assert(offsetof(SpectralImage, image_index) == 4u, "SpectralImage::image_index offset changed; update material ABI");

static_assert(std::is_standard_layout_v<SampledImage>, "SampledImage must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<SampledImage>, "SampledImage must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(SampledImage) == 32u, "SampledImage size changed; update material ABI");
static_assert(offsetof(SampledImage, value) == 0u, "SampledImage::value offset changed; update material ABI");
static_assert(offsetof(SampledImage, image_index) == 16u, "SampledImage::image_index offset changed; update material ABI");
static_assert(offsetof(SampledImage, channel) == 20u, "SampledImage::channel offset changed; update material ABI");

static_assert(std::is_standard_layout_v<Thinfilm>, "Thinfilm must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Thinfilm>, "Thinfilm must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Thinfilm) == 32u, "Thinfilm size changed; update material ABI");

static_assert(std::is_standard_layout_v<RefractiveIndex>, "RefractiveIndex must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<RefractiveIndex>, "RefractiveIndex must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(RefractiveIndex) == 16u, "RefractiveIndex size changed; update material/spectrum ABI");

static_assert(std::is_standard_layout_v<SpectralDistribution>, "SpectralDistribution must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<SpectralDistribution>, "SpectralDistribution must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(SpectralDistribution) == kSpectralDistributionStride, "SpectralDistribution size changed; update GPU shader decode stride");
static_assert(offsetof(SpectralDistribution, integrated_value) == kSpectralDistributionIntegratedOffset,
  "SpectralDistribution::integrated_value offset changed; update GPU shader decode");

static_assert(offsetof(GPUSceneGlobals, vertex_count) == kSceneGlobalsVertexCountOffset, "GPUSceneGlobals::vertex_count offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, triangle_count) == kSceneGlobalsTriangleCountOffset, "GPUSceneGlobals::triangle_count offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, environment_emitter_count) == kSceneGlobalsEnvironmentEmitterCountOffset,
  "GPUSceneGlobals::environment_emitter_count offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, bounding_sphere_center) == kSceneGlobalsBoundingSphereCenterOffset,
  "GPUSceneGlobals::bounding_sphere_center offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, bounding_sphere_radius) == kSceneGlobalsBoundingSphereRadiusOffset,
  "GPUSceneGlobals::bounding_sphere_radius offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, environment_emitters) == kSceneGlobalsEnvironmentEmittersOffset,
  "GPUSceneGlobals::environment_emitters offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, pixel_filter_image_index) == kSceneGlobalsPixelFilterImageIndexOffset,
  "GPUSceneGlobals::pixel_filter_image_index offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, pixel_filter_radius) == kSceneGlobalsPixelFilterRadiusOffset,
  "GPUSceneGlobals::pixel_filter_radius offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<GPUSceneOptions>, "GPUSceneOptions must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<GPUSceneOptions>, "GPUSceneOptions must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(GPUSceneOptions) == 48u, "GPUSceneOptions size changed; update GPU shader decode stride");
static_assert(offsetof(GPUSceneOptions, min_path_length) == kSceneOptionsMinPathLengthOffset, "GPUSceneOptions::min_path_length offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, max_path_length) == kSceneOptionsMaxPathLengthOffset, "GPUSceneOptions::max_path_length offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, samples) == kSceneOptionsSamplesOffset, "GPUSceneOptions::samples offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, random_path_termination) == kSceneOptionsRandomPathTerminationOffset,
  "GPUSceneOptions::random_path_termination offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, noise_threshold) == kSceneOptionsNoiseThresholdOffset, "GPUSceneOptions::noise_threshold offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, radiance_clamp) == kSceneOptionsRadianceClampOffset, "GPUSceneOptions::radiance_clamp offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, strategy_flags) == kSceneOptionsStrategyFlagsOffset, "GPUSceneOptions::strategy_flags offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, light_sampling) == kSceneOptionsLightSamplingOffset, "GPUSceneOptions::light_sampling offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, properties_flags) == kSceneOptionsPropertiesFlagsOffset, "GPUSceneOptions::properties_flags offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, path_mode) == kSceneOptionsPathModeOffset, "GPUSceneOptions::path_mode offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, random_seed) == kSceneOptionsRandomSeedOffset, "GPUSceneOptions::random_seed offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<Camera>, "Camera must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Camera>, "Camera must stay trivially copyable for GPU upload ABI");
static_assert(offsetof(Camera, position) == kCameraPositionOffset, "Camera::position offset changed; update GPU shader decode");
static_assert(offsetof(Camera, direction) == kCameraDirectionOffset, "Camera::direction offset changed; update GPU shader decode");
static_assert(offsetof(Camera, aspect) == kCameraAspectOffset, "Camera::aspect offset changed; update GPU shader decode");
static_assert(offsetof(Camera, side) == kCameraSideOffset, "Camera::side offset changed; update GPU shader decode");
static_assert(offsetof(Camera, tan_half_fov) == kCameraTanHalfFovOffset, "Camera::tan_half_fov offset changed; update GPU shader decode");
static_assert(offsetof(Camera, up) == kCameraUpOffset, "Camera::up offset changed; update GPU shader decode");
static_assert(offsetof(Camera, film_size) == kCameraFilmSizeOffset, "Camera::film_size offset changed; update GPU shader decode");
static_assert(offsetof(Camera, clip_near) == kCameraClipNearOffset, "Camera::clip_near offset changed; update GPU shader decode");
static_assert(offsetof(Camera, clip_far) == kCameraClipFarOffset, "Camera::clip_far offset changed; update GPU shader decode");
static_assert(offsetof(Camera, medium_index) == kCameraMediumIndexOffset, "Camera::medium_index offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<GPUImageBlobHeader>, "GPUImageBlobHeader must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<GPUImageBlobHeader>, "GPUImageBlobHeader must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(GPUImageBlobHeader) == 16u, "GPUImageBlobHeader size changed; update GPU image blob ABI");
static_assert(offsetof(GPUImageBlobHeader, image_count) == kImageBlobHeaderImageCountOffset, "GPUImageBlobHeader::image_count offset changed; update GPU image blob ABI");
static_assert(offsetof(GPUImageBlobHeader, images_offset) == kImageBlobHeaderImagesOffset, "GPUImageBlobHeader::images_offset offset changed; update GPU image blob ABI");
static_assert(offsetof(GPUImageBlobHeader, data_chunk_count) == kImageBlobHeaderDataChunkCountOffset,
  "GPUImageBlobHeader::data_chunk_count offset changed; update GPU image blob ABI");
static_assert(offsetof(GPUImageBlobHeader, data_chunk_indices_offset) == kImageBlobHeaderDataChunkIndicesOffset,
  "GPUImageBlobHeader::data_chunk_indices_offset offset changed; update GPU image blob ABI");

static_assert(std::is_standard_layout_v<::Image>, "Interop Image must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<::Image>, "Interop Image must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(::Image) == kImageDescStride, "Interop Image size changed; update GPU image blob ABI");
static_assert(offsetof(::Image, fsize) == kImageDescFSizeOffset, "Image::fsize offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, offset) == kImageDescOffsetOffset, "Image::offset offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, scale) == kImageDescScaleOffset, "Image::scale offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, isize) == kImageDescISizeOffset, "Image::isize offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, options) == kImageDescOptionsOffset, "Image::options offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, format) == kImageDescFormatOffset, "Image::format offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, data_size) == kImageDescDataSizeOffset, "Image::data_size offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, pixel_data_offset) == kImageDescPixelDataOffset, "Image::pixel_data_offset offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, pixel_data_stride) == kImageDescPixelDataStrideOffset, "Image::pixel_data_stride offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, pixel_data_chunk_index) == kImageDescPixelDataChunkIndexOffset, "Image::pixel_data_chunk_index offset changed; update GPU image blob ABI");

static_assert(std::is_standard_layout_v<GPUMediumBlobHeader>, "GPUMediumBlobHeader must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<GPUMediumBlobHeader>, "GPUMediumBlobHeader must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(GPUMediumBlobHeader) == 16u, "GPUMediumBlobHeader size changed; update GPU medium blob ABI");
static_assert(offsetof(GPUMediumBlobHeader, medium_count) == kMediumBlobHeaderMediumCountOffset, "GPUMediumBlobHeader::medium_count offset changed; update GPU medium blob ABI");
static_assert(offsetof(GPUMediumBlobHeader, mediums_offset) == kMediumBlobHeaderMediumsOffset, "GPUMediumBlobHeader::mediums_offset offset changed; update GPU medium blob ABI");
static_assert(offsetof(GPUMediumBlobHeader, data_chunk_count) == kMediumBlobHeaderDataChunkCountOffset,
  "GPUMediumBlobHeader::data_chunk_count offset changed; update GPU medium blob ABI");
static_assert(offsetof(GPUMediumBlobHeader, data_chunk_indices_offset) == kMediumBlobHeaderDataChunkIndicesOffset,
  "GPUMediumBlobHeader::data_chunk_indices_offset offset changed; update GPU medium blob ABI");
static_assert(SceneLimits::MaxEnvironmentEmitters == std::extent_v<decltype(GPUSceneGlobals::environment_emitters)>,
  "SceneLimits::MaxEnvironmentEmitters changed; update GPUSceneGlobals::environment_emitters size to keep CPU/GPU emitter lists aligned");

static_assert(std::is_standard_layout_v<::Medium>, "Interop Medium must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<::Medium>, "Interop Medium must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(::Medium) == kMediumStride, "Interop Medium size changed; update GPU medium blob ABI");
static_assert(sizeof(::Medium) == kMediumStride, "Medium stride changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, dimensions)) == kMediumGridDimensionsOffset, "Medium::grid.dimensions offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, type)) == kMediumGridTypeOffset, "Medium::grid.type offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_type)) == kMediumGridNoiseTypeOffset, "Medium::grid.noise_type offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, density_data_offset)) == kMediumGridDensityDataOffsetOffset,
  "Medium::grid.density_data_offset offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, density_count)) == kMediumGridDensityCountOffset,
  "Medium::grid.density_count offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_seed)) == kMediumGridNoiseSeedOffset, "Medium::grid.noise_seed offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_offset)) == kMediumGridNoiseOffsetOffset,
  "Medium::grid.noise_offset offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_enable_border_fade)) == kMediumGridNoiseEnableBorderFadeOffset,
  "Medium::grid.noise_enable_border_fade offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_octaves)) == kMediumGridNoiseOctavesOffset,
  "Medium::grid.noise_octaves offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_scale)) == kMediumGridNoiseScaleOffset, "Medium::grid.noise_scale offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_lacunarity)) == kMediumGridNoiseLacunarityOffset,
  "Medium::grid.noise_lacunarity offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_persistence)) == kMediumGridNoisePersistenceOffset,
  "Medium::grid.noise_persistence offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_power)) == kMediumGridNoisePowerOffset, "Medium::grid.noise_power offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_sharpness)) == kMediumGridNoiseSharpnessOffset,
  "Medium::grid.noise_sharpness offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_border_fade_distance)) == kMediumGridNoiseBorderFadeDistanceOffset,
  "Medium::grid.noise_border_fade_distance offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, density_data_chunk_index)) == kMediumGridDensityDataChunkIndexOffset,
  "Medium::grid.density_data_chunk_index offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, density_image_index)) == kMediumGridDensityImageIndexOffset,
  "Medium::grid.density_image_index offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, bounds) + offsetof(BoundingBox, p_min)) == kMediumBoundsMinOffset, "Medium::bounds.p_min offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, bounds) + offsetof(BoundingBox, p_max)) == kMediumBoundsMaxOffset, "Medium::bounds.p_max offset changed; update GPU shader decode");
static_assert(offsetof(::Medium, absorption_index) == kMediumAbsorptionIndexOffset, "Medium::absorption_index offset changed; update GPU shader decode");
static_assert(offsetof(::Medium, scattering_index) == kMediumScatteringIndexOffset, "Medium::scattering_index offset changed; update GPU shader decode");
static_assert(offsetof(::Medium, phase_function_g) == kMediumPhaseFunctionGOffset, "Medium::phase_function_g offset changed; update GPU shader decode");
static_assert(offsetof(::Medium, enable_explicit_connections) == kMediumEnableExplicitConnectionsOffset,
  "Medium::enable_explicit_connections offset changed; update GPU shader decode");
static_assert(offsetof(::Medium, cls) == kMediumClassOffset, "Medium::cls offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, world_to_object) + offsetof(AffineTransform, rows)) == kMediumWorldToObjectRow0Offset,
  "Medium::world_to_object.rows[0] offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, world_to_object) + offsetof(AffineTransform, rows) + sizeof(float4)) == kMediumWorldToObjectRow1Offset,
  "Medium::world_to_object.rows[1] offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, world_to_object) + offsetof(AffineTransform, rows) + 2u * sizeof(float4)) == kMediumWorldToObjectRow2Offset,
  "Medium::world_to_object.rows[2] offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, local_bounds) + offsetof(BoundingBox, p_min)) == kMediumLocalBoundsMinOffset,
  "Medium::local_bounds.p_min offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, local_bounds) + offsetof(BoundingBox, p_max)) == kMediumLocalBoundsMaxOffset,
  "Medium::local_bounds.p_max offset changed; update GPU shader decode");
static_assert(::Medium::Homogeneous == Medium::Homogeneous, "Medium::Homogeneous changed; update GPU shader decode");
static_assert(::Medium::Heterogeneous == Medium::Heterogeneous, "Medium::Heterogeneous changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(DensityGrid::Type::Texture3D) == MediumGridType::Texture3D, "DensityGrid::Type::Texture3D changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(DensityGrid::Type::NoiseFunction) == MediumGridType::NoiseFunction, "DensityGrid::Type::NoiseFunction changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Perlin) == MediumNoiseType::Perlin, "NoiseFunction::Perlin changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Worley) == MediumNoiseType::Worley, "NoiseFunction::Worley changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Billow) == MediumNoiseType::Billow, "NoiseFunction::Billow changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Voronoi) == MediumNoiseType::Voronoi, "NoiseFunction::Voronoi changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Lattice) == MediumNoiseType::Lattice, "NoiseFunction::Lattice changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Uniform) == MediumNoiseType::Uniform, "NoiseFunction::Uniform changed; update GPU shader decode");
}  // namespace
}  // namespace etx
