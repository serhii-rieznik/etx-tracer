#define ETX_EMISSION_STAGE_wavefront_emission_sample_begin_main   1
#define ETX_EMISSION_STAGE_wavefront_emission_sample_delta_main   2
#define ETX_EMISSION_STAGE_wavefront_emission_sample_resolve_main 3
#define ETX_EMISSION_STAGE_wavefront_emission_advance_main        4
#define ETX_EMISSION_STAGE_wavefront_emission_phase_main          5
#define ETX_EMISSION_STAGE_wavefront_emission_base_sample_main    6
#define ETX_EMISSION_STAGE_wavefront_emission_velvet_sample_main  7
#define ETX_EMISSION_STAGE_wavefront_emission_base_emission_main  8
#define ETX_EMISSION_STAGE_wavefront_emission_velvet_escape_main  9
#define ETX_EMISSION_STAGE_wavefront_emission_base_continue_main  10
#define ETX_EMISSION_STAGE_EXPAND(entry)                          ETX_EMISSION_STAGE_##entry
#define ETX_EMISSION_STAGE_SELECT(entry)                          ETX_EMISSION_STAGE_EXPAND(entry)
#ifdef ETX_STAGE_ENTRY
# define ETX_EMISSION_STAGE ETX_EMISSION_STAGE_SELECT(ETX_STAGE_ENTRY)
#else
# define ETX_EMISSION_STAGE 0
#endif

#include "gpu_rt_wavefront_transport_work_storage.hlsl"
#include "gpu_rt_wavefront_emission_storage.hlsl"
#include <interop/gpu_emission_consumer_abi.hxx>
#include <access/material_access_gpu.hxx>
#include <interop/thermal_surface_shared.hxx>

void wavefront_emission_store_interface(RWByteAddressBuffer buffer, uint base_offset, BSDFEnergyCompensatedDielectricSampleState state) {
  buffer.Store2(base_offset + 0u, asuint(state.prepared.roughness));
  buffer.Store(base_offset + 8u, asuint(state.prepared.alpha));
  buffer.Store3(base_offset + 12u, asuint(state.prepared.ext_ior.eta.integrated));
  buffer.Store(base_offset + 24u, asuint(state.prepared.ext_ior.eta.value));
  buffer.Store(base_offset + 28u, asuint(state.prepared.ext_ior.eta.wavelength));
  buffer.Store(base_offset + 32u, state.prepared.ext_ior.eta.flags);
  buffer.Store3(base_offset + 36u, asuint(state.prepared.ext_ior.k.integrated));
  buffer.Store(base_offset + 48u, asuint(state.prepared.ext_ior.k.value));
  buffer.Store(base_offset + 52u, asuint(state.prepared.ext_ior.k.wavelength));
  buffer.Store(base_offset + 56u, state.prepared.ext_ior.k.flags);
  buffer.Store(base_offset + 60u, state.prepared.ext_ior.cls);
  buffer.Store3(base_offset + 64u, asuint(state.prepared.int_ior.eta.integrated));
  buffer.Store(base_offset + 76u, asuint(state.prepared.int_ior.eta.value));
  buffer.Store(base_offset + 80u, asuint(state.prepared.int_ior.eta.wavelength));
  buffer.Store(base_offset + 84u, state.prepared.int_ior.eta.flags);
  buffer.Store3(base_offset + 88u, asuint(state.prepared.int_ior.k.integrated));
  buffer.Store(base_offset + 100u, asuint(state.prepared.int_ior.k.value));
  buffer.Store(base_offset + 104u, asuint(state.prepared.int_ior.k.wavelength));
  buffer.Store(base_offset + 108u, state.prepared.int_ior.k.flags);
  buffer.Store(base_offset + 112u, state.prepared.int_ior.cls);
  buffer.Store3(base_offset + 116u, asuint(state.prepared.thinfilm.ior.eta.integrated));
  buffer.Store(base_offset + 128u, asuint(state.prepared.thinfilm.ior.eta.value));
  buffer.Store(base_offset + 132u, asuint(state.prepared.thinfilm.ior.eta.wavelength));
  buffer.Store(base_offset + 136u, state.prepared.thinfilm.ior.eta.flags);
  buffer.Store3(base_offset + 140u, asuint(state.prepared.thinfilm.ior.k.integrated));
  buffer.Store(base_offset + 152u, asuint(state.prepared.thinfilm.ior.k.value));
  buffer.Store(base_offset + 156u, asuint(state.prepared.thinfilm.ior.k.wavelength));
  buffer.Store(base_offset + 160u, state.prepared.thinfilm.ior.k.flags);
  buffer.Store(base_offset + 164u, state.prepared.thinfilm.ior.cls);
  buffer.Store3(base_offset + 168u, asuint(state.prepared.thinfilm.rgb_wavelengths));
  buffer.Store(base_offset + 180u, asuint(state.prepared.thinfilm.thickness));
  buffer.Store(base_offset + 184u, asuint(state.prepared.thinfilm.weight));
  buffer.Store3(base_offset + 188u, asuint(state.prepared.thinfilm.coating_ior.eta.integrated));
  buffer.Store(base_offset + 200u, asuint(state.prepared.thinfilm.coating_ior.eta.value));
  buffer.Store(base_offset + 204u, asuint(state.prepared.thinfilm.coating_ior.eta.wavelength));
  buffer.Store(base_offset + 208u, state.prepared.thinfilm.coating_ior.eta.flags);
  buffer.Store3(base_offset + 212u, asuint(state.prepared.thinfilm.coating_ior.k.integrated));
  buffer.Store(base_offset + 224u, asuint(state.prepared.thinfilm.coating_ior.k.value));
  buffer.Store(base_offset + 228u, asuint(state.prepared.thinfilm.coating_ior.k.wavelength));
  buffer.Store(base_offset + 232u, state.prepared.thinfilm.coating_ior.k.flags);
  buffer.Store(base_offset + 236u, state.prepared.thinfilm.coating_ior.cls);
  buffer.Store(base_offset + 240u, asuint(state.prepared.thinfilm.coating_thickness_nm));
  buffer.Store(base_offset + 244u, state.prepared.thinfilm.incident_inside);
  buffer.Store2(base_offset + 248u, asuint(state.prepared.interface_lut_value));
  buffer.Store(base_offset + 256u, (state.prepared.supported ? 1u : 0u));
  buffer.Store(base_offset + 260u, (state.prepared.conductor_delta ? 1u : 0u));
  buffer.Store(base_offset + 264u, (state.prepared.dielectric_delta ? 1u : 0u));
  buffer.Store4(base_offset + 268u, asuint(state.incident_pair.outside_albedo));
  buffer.Store4(base_offset + 284u, asuint(state.incident_pair.inside_albedo));
  buffer.Store(base_offset + 300u, asuint(state.incident_pair.outside_probability));
  buffer.Store(base_offset + 304u, asuint(state.incident_pair.inside_probability));
  buffer.Store3(base_offset + 308u, asuint(state.proposal.sample.weight.integrated));
  buffer.Store(base_offset + 320u, asuint(state.proposal.sample.weight.value));
  buffer.Store(base_offset + 324u, asuint(state.proposal.sample.weight.wavelength));
  buffer.Store(base_offset + 328u, state.proposal.sample.weight.flags);
  buffer.Store3(base_offset + 332u, asuint(state.proposal.sample.w_o));
  buffer.Store(base_offset + 344u, asuint(state.proposal.sample.pdf));
  buffer.Store(base_offset + 348u, asuint(state.proposal.sample.eta));
  buffer.Store(base_offset + 352u, state.proposal.sample.properties);
  buffer.Store(base_offset + 356u, state.proposal.sample.medium_index);
  buffer.Store3(base_offset + 360u, asuint(state.proposal.local_w_o));
  buffer.Store(base_offset + 372u, state.proposal.kind);
  buffer.Store(base_offset + 376u, (state.proposal.wavelength_dependent_direction ? 1u : 0u));
  buffer.Store3(base_offset + 380u, asuint(state.w_i_local));
  buffer.Store(base_offset + 392u, state.stage);
}

BSDFEnergyCompensatedDielectricSampleState wavefront_emission_load_interface(RWByteAddressBuffer buffer, uint base_offset) {
  BSDFEnergyCompensatedDielectricSampleState state = (BSDFEnergyCompensatedDielectricSampleState)0;
  state.prepared.roughness = asfloat(buffer.Load2(base_offset + 0u));
  state.prepared.alpha = asfloat(buffer.Load(base_offset + 8u));
  state.prepared.ext_ior.eta.integrated = asfloat(buffer.Load3(base_offset + 12u));
  state.prepared.ext_ior.eta.value = asfloat(buffer.Load(base_offset + 24u));
  state.prepared.ext_ior.eta.wavelength = asfloat(buffer.Load(base_offset + 28u));
  state.prepared.ext_ior.eta.flags = buffer.Load(base_offset + 32u);
  state.prepared.ext_ior.k.integrated = asfloat(buffer.Load3(base_offset + 36u));
  state.prepared.ext_ior.k.value = asfloat(buffer.Load(base_offset + 48u));
  state.prepared.ext_ior.k.wavelength = asfloat(buffer.Load(base_offset + 52u));
  state.prepared.ext_ior.k.flags = buffer.Load(base_offset + 56u);
  state.prepared.ext_ior.cls = buffer.Load(base_offset + 60u);
  state.prepared.int_ior.eta.integrated = asfloat(buffer.Load3(base_offset + 64u));
  state.prepared.int_ior.eta.value = asfloat(buffer.Load(base_offset + 76u));
  state.prepared.int_ior.eta.wavelength = asfloat(buffer.Load(base_offset + 80u));
  state.prepared.int_ior.eta.flags = buffer.Load(base_offset + 84u);
  state.prepared.int_ior.k.integrated = asfloat(buffer.Load3(base_offset + 88u));
  state.prepared.int_ior.k.value = asfloat(buffer.Load(base_offset + 100u));
  state.prepared.int_ior.k.wavelength = asfloat(buffer.Load(base_offset + 104u));
  state.prepared.int_ior.k.flags = buffer.Load(base_offset + 108u);
  state.prepared.int_ior.cls = buffer.Load(base_offset + 112u);
  state.prepared.thinfilm.ior.eta.integrated = asfloat(buffer.Load3(base_offset + 116u));
  state.prepared.thinfilm.ior.eta.value = asfloat(buffer.Load(base_offset + 128u));
  state.prepared.thinfilm.ior.eta.wavelength = asfloat(buffer.Load(base_offset + 132u));
  state.prepared.thinfilm.ior.eta.flags = buffer.Load(base_offset + 136u);
  state.prepared.thinfilm.ior.k.integrated = asfloat(buffer.Load3(base_offset + 140u));
  state.prepared.thinfilm.ior.k.value = asfloat(buffer.Load(base_offset + 152u));
  state.prepared.thinfilm.ior.k.wavelength = asfloat(buffer.Load(base_offset + 156u));
  state.prepared.thinfilm.ior.k.flags = buffer.Load(base_offset + 160u);
  state.prepared.thinfilm.ior.cls = buffer.Load(base_offset + 164u);
  state.prepared.thinfilm.rgb_wavelengths = asfloat(buffer.Load3(base_offset + 168u));
  state.prepared.thinfilm.thickness = asfloat(buffer.Load(base_offset + 180u));
  state.prepared.thinfilm.weight = asfloat(buffer.Load(base_offset + 184u));
  state.prepared.thinfilm.coating_ior.eta.integrated = asfloat(buffer.Load3(base_offset + 188u));
  state.prepared.thinfilm.coating_ior.eta.value = asfloat(buffer.Load(base_offset + 200u));
  state.prepared.thinfilm.coating_ior.eta.wavelength = asfloat(buffer.Load(base_offset + 204u));
  state.prepared.thinfilm.coating_ior.eta.flags = buffer.Load(base_offset + 208u);
  state.prepared.thinfilm.coating_ior.k.integrated = asfloat(buffer.Load3(base_offset + 212u));
  state.prepared.thinfilm.coating_ior.k.value = asfloat(buffer.Load(base_offset + 224u));
  state.prepared.thinfilm.coating_ior.k.wavelength = asfloat(buffer.Load(base_offset + 228u));
  state.prepared.thinfilm.coating_ior.k.flags = buffer.Load(base_offset + 232u);
  state.prepared.thinfilm.coating_ior.cls = buffer.Load(base_offset + 236u);
  state.prepared.thinfilm.coating_thickness_nm = asfloat(buffer.Load(base_offset + 240u));
  state.prepared.thinfilm.incident_inside = buffer.Load(base_offset + 244u);
  state.prepared.interface_lut_value = asfloat(buffer.Load2(base_offset + 248u));
  state.prepared.supported = (buffer.Load(base_offset + 256u) != 0u);
  state.prepared.conductor_delta = (buffer.Load(base_offset + 260u) != 0u);
  state.prepared.dielectric_delta = (buffer.Load(base_offset + 264u) != 0u);
  state.incident_pair.outside_albedo = asfloat(buffer.Load4(base_offset + 268u));
  state.incident_pair.inside_albedo = asfloat(buffer.Load4(base_offset + 284u));
  state.incident_pair.outside_probability = asfloat(buffer.Load(base_offset + 300u));
  state.incident_pair.inside_probability = asfloat(buffer.Load(base_offset + 304u));
  state.proposal.sample.weight.integrated = asfloat(buffer.Load3(base_offset + 308u));
  state.proposal.sample.weight.value = asfloat(buffer.Load(base_offset + 320u));
  state.proposal.sample.weight.wavelength = asfloat(buffer.Load(base_offset + 324u));
  state.proposal.sample.weight.flags = buffer.Load(base_offset + 328u);
  state.proposal.sample.w_o = asfloat(buffer.Load3(base_offset + 332u));
  state.proposal.sample.pdf = asfloat(buffer.Load(base_offset + 344u));
  state.proposal.sample.eta = asfloat(buffer.Load(base_offset + 348u));
  state.proposal.sample.properties = buffer.Load(base_offset + 352u);
  state.proposal.sample.medium_index = buffer.Load(base_offset + 356u);
  state.proposal.local_w_o = asfloat(buffer.Load3(base_offset + 360u));
  state.proposal.kind = buffer.Load(base_offset + 372u);
  state.proposal.wavelength_dependent_direction = (buffer.Load(base_offset + 376u) != 0u);
  state.w_i_local = asfloat(buffer.Load3(base_offset + 380u));
  state.stage = buffer.Load(base_offset + 392u);
  return state;
}

void wavefront_emission_store_velvet(RWByteAddressBuffer buffer, uint offset, BSDFVelvetWalk state) {
  wavefront_emission_store_response(buffer, offset + 0u, state.value);
  wavefront_emission_store_response(buffer, offset + 24u, state.absorption);
  wavefront_emission_store_response(buffer, offset + 48u, state.throughput);
  wavefront_emission_store_response(buffer, offset + 72u, state.transmission);
  wavefront_emission_store_response(buffer, offset + 96u, state.multiple);
  buffer.Store3(offset + 120u, asuint(state.direction));
  buffer.Store3(offset + 132u, asuint(state.outgoing));
  buffer.Store(offset + 144u, asuint(state.exponent));
  buffer.Store(offset + 148u, state.depth);
  buffer.Store(offset + 152u, state.operation);
}

BSDFVelvetWalk wavefront_emission_load_velvet(RWByteAddressBuffer buffer, uint offset) {
  BSDFVelvetWalk state = (BSDFVelvetWalk)0;
  state.value = wavefront_emission_load_response(buffer, offset + 0u);
  state.absorption = wavefront_emission_load_response(buffer, offset + 24u);
  state.throughput = wavefront_emission_load_response(buffer, offset + 48u);
  state.transmission = wavefront_emission_load_response(buffer, offset + 72u);
  state.multiple = wavefront_emission_load_response(buffer, offset + 96u);
  state.direction = asfloat(buffer.Load3(offset + 120u));
  state.outgoing = asfloat(buffer.Load3(offset + 132u));
  state.exponent = asfloat(buffer.Load(offset + 144u));
  state.depth = buffer.Load(offset + 148u);
  state.operation = buffer.Load(offset + 152u);
  return state;
}

void wavefront_emission_store_coating(RWByteAddressBuffer buffer, uint offset, BSDFCoatingEscapeState state) {
  wavefront_emission_store_response(buffer, offset + 0u, state.source);
  wavefront_emission_store_response(buffer, offset + 24u, state.absorption);
  wavefront_emission_store_response(buffer, offset + 48u, state.throughput);
  wavefront_emission_store_response(buffer, offset + 72u, state.base.weight);
  buffer.Store3(offset + 96u, asuint(state.base.w_o));
  buffer.Store(offset + 108u, asuint(state.base.pdf));
  buffer.Store(offset + 112u, asuint(state.base.eta));
  buffer.Store(offset + 116u, state.base.properties);
  buffer.Store(offset + 120u, state.base.medium_index);
  buffer.Store3(offset + 124u, asuint(state.direction));
  buffer.Store(offset + 136u, asuint(state.height));
  buffer.Store(offset + 140u, state.depth);
  buffer.Store(offset + 144u, state.operation);
  buffer.Store(offset + 148u, state.sample_target);
}

BSDFCoatingEscapeState wavefront_emission_load_coating(RWByteAddressBuffer buffer, uint offset) {
  BSDFCoatingEscapeState state = (BSDFCoatingEscapeState)0;
  state.source = wavefront_emission_load_response(buffer, offset + 0u);
  state.absorption = wavefront_emission_load_response(buffer, offset + 24u);
  state.throughput = wavefront_emission_load_response(buffer, offset + 48u);
  state.base.weight = wavefront_emission_load_response(buffer, offset + 72u);
  state.base.w_o = asfloat(buffer.Load3(offset + 96u));
  state.base.pdf = asfloat(buffer.Load(offset + 108u));
  state.base.eta = asfloat(buffer.Load(offset + 112u));
  state.base.properties = buffer.Load(offset + 116u);
  state.base.medium_index = buffer.Load(offset + 120u);
  state.direction = asfloat(buffer.Load3(offset + 124u));
  state.height = asfloat(buffer.Load(offset + 136u));
  state.depth = buffer.Load(offset + 140u);
  state.operation = buffer.Load(offset + 144u);
  state.sample_target = buffer.Load(offset + 148u);
  return state;
}

BSDFResourceContext wavefront_emission_context() {
  return make_bsdf_resource_gpu_context(constants.scene.images, constants.scene.spectrums, constants.scene.spectral_values, constants.scene.energy_compensation_interfaces,
    constants.scene.scene_globals);
}

BSDFData wavefront_emission_data(RWByteAddressBuffer buffer, uint offset) {
  BSDFData data = (BSDFData)0;
  const uint2 spect = buffer.Load2(offset + kGPUEmissionRequestSpectrumOffset);
  data.spectrum_sample.wavelength = asfloat(spect.x);
  data.spectrum_sample.flags = spect.y;
  data.tex = asfloat(buffer.Load2(offset + kGPUEmissionRequestUVOffset));
  data.nrm = float3(0.0f, 0.0f, 1.0f);
  data.tan = float3(1.0f, 0.0f, 0.0f);
  data.btn = float3(0.0f, 1.0f, 0.0f);
  data.w_i = -asfloat(buffer.Load3(offset + kGPUEmissionRequestOutgoingOffset));
  data.path_source = PathSource::Camera;
  return data;
}

Material wavefront_emission_material(RWByteAddressBuffer buffer, uint offset) {
  return gpu_abi_load_material_full(WAVEFRONT_RO_BUFFER(constants.scene.materials), buffer.Load(offset + kGPUEmissionRequestMaterialOffset));
}

bool wavefront_emission_active(RWByteAddressBuffer buffer, uint index, uint stride, out uint offset) {
  offset = 0u;
  if (index >= buffer.Load(kGPUEmissionRequestCapacityOffset)) {
    return false;
  }
  offset = wavefront_emission_request_offset(index, stride);
  return buffer.Load(offset + kGPUEmissionRequestActiveOffset) != 0u;
}

void wavefront_emission_publish(RWByteAddressBuffer buffer, uint offset, SpectralResponse radiance, Sampler sampler) {
  wavefront_emission_store_response(buffer, offset + kGPUEmissionRequestResultOffset, radiance);
  wavefront_emission_store_sampler(buffer, offset + kGPUEmissionRequestSamplerOffset, sampler);
  buffer.Store(offset + kGPUEmissionRequestActiveOffset, 0u);
  uint unused;
  buffer.InterlockedAdd(kGPUEmissionRequestPendingOffset, 0xffffffffu, unused);
}

[numthreads(64, 1, 1)] void wavefront_emission_clear_main(uint3 id : SV_DispatchThreadID) {
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_request_buffer);
  RWByteAddressBuffer consumers = WAVEFRONT_RW_BUFFER(resources.emission_consumer_buffer);
  if ((constants.dispatch_item_offset + id.x) == 0u) {
    buffer.Store(kGPUEmissionRequestPendingOffset, 0u);
    buffer.Store(kGPUEmissionRequestCapacityOffset, resources.path_capacity);
    consumers.Store(kGPUEmissionConsumerPendingOffset, 0u);
    consumers.Store(kGPUEmissionConsumerCapacityOffset, resources.path_capacity);
  }
  if (id.x < constants.dispatch_item_count) {
    buffer.Store(wavefront_emission_request_offset((constants.dispatch_item_offset + id.x), resources.emission_request_stride) + kGPUEmissionRequestActiveOffset, 0u);
    consumers.Store(kGPUEmissionConsumerHeaderSize + (constants.dispatch_item_offset + id.x) * kGPUEmissionConsumerStride, 0u);
  }
}

  [numthreads(64, 1, 1)] void wavefront_emission_classify_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_request_buffer);
  uint offset;
  if (wavefront_emission_active(buffer, request_index, resources.emission_request_stride, offset) == false) {
    return;
  }
  if ((buffer.Load(offset + kGPUEmissionRequestOperationOffset) != GPUEmissionRequestOperation::Classify)) {
    return;
  }
  const Material material = wavefront_emission_material(buffer, offset);
  const uint operation = (thermal_surface_uses_substrate_escape(material) == false)
                           ? GPUEmissionRequestOperation::Simple
                           : (bsdf_conductor_has_finite_coating(material) ? GPUEmissionRequestOperation::FiniteConductor : GPUEmissionRequestOperation::Begin);
  buffer.Store(offset + kGPUEmissionRequestOperationOffset, operation);
}

[numthreads(64, 1, 1)] void wavefront_emission_simple_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_request_buffer);
  uint offset;
  if (wavefront_emission_active(buffer, request_index, resources.emission_request_stride, offset) == false) {
    return;
  }
  if ((buffer.Load(offset + kGPUEmissionRequestOperationOffset) != GPUEmissionRequestOperation::Simple)) {
    return;
  }
  const BSDFResourceContext context = wavefront_emission_context();
  const BSDFData data = wavefront_emission_data(buffer, offset);
  const Material material = wavefront_emission_material(buffer, offset);
  const float geometric_cosine = asfloat(buffer.Load(offset + kGPUEmissionRequestGeometricCosineOffset));
  const SpectralResponse result = thermal_surface_combined_radiance(context, data.spectrum_sample, material, data.tex, geometric_cosine, -data.w_i.z);
  wavefront_emission_publish(buffer, offset, result, wavefront_emission_load_sampler(buffer, offset + kGPUEmissionRequestSamplerOffset));
}

  [numthreads(64, 1, 1)] void wavefront_emission_finite_conductor_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_request_buffer);
  uint offset;
  if (wavefront_emission_active(buffer, request_index, resources.emission_request_stride, offset) == false) {
    return;
  }
  if ((buffer.Load(offset + kGPUEmissionRequestOperationOffset) != GPUEmissionRequestOperation::FiniteConductor)) {
    return;
  }
  const BSDFResourceContext context = wavefront_emission_context();
  const BSDFData data = wavefront_emission_data(buffer, offset);
  const Material material = wavefront_emission_material(buffer, offset);
  const float geometric_cosine = asfloat(buffer.Load(offset + kGPUEmissionRequestGeometricCosineOffset));
  Sampler sampler = wavefront_emission_load_sampler(buffer, offset + kGPUEmissionRequestSamplerOffset);
  const SpectralResponse result = thermal_conductor_radiance(context, data.spectrum_sample, material, data.tex, geometric_cosine, -data.w_i.z, sampler);
  wavefront_emission_publish(buffer, offset, result, sampler);
}

[numthreads(64, 1, 1)] void wavefront_emission_begin_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_request_buffer);
  uint offset;
  if (wavefront_emission_active(buffer, request_index, resources.emission_request_stride, offset) == false) {
    return;
  }
  if ((buffer.Load(offset + kGPUEmissionRequestOperationOffset) != GPUEmissionRequestOperation::Begin)) {
    return;
  }
  const BSDFResourceContext context = wavefront_emission_context();
  const BSDFData data = wavefront_emission_data(buffer, offset);
  const Material material = wavefront_emission_material(buffer, offset);
  const float geometric_cosine = asfloat(buffer.Load(offset + kGPUEmissionRequestGeometricCosineOffset));
  const bool transmitting_base = (material.cls == MaterialClass::CoatedConductor) && material_supports_surface_transmission(material);
  const float shading_cosine = transmitting_base ? abs(data.w_i.z) : -data.w_i.z;
  if ((shading_cosine <= kEpsilon) || ((transmitting_base == false) && (material.two_sided == 0u) && (geometric_cosine <= 0.0f)) ||
      (spectral_response_is_zero(bsdf_resource_apply_image(context, data.spectrum_sample, material.emission, data.tex)) &&
        (material.coating.thermal_spectrum_index == kInvalidIndex))) {
    wavefront_emission_publish(buffer, offset, spectral_response_zero(data.spectrum_sample), wavefront_emission_load_sampler(buffer, offset + kGPUEmissionRequestSamplerOffset));
    return;
  }
  if (material.cls == MaterialClass::Velvet) {
    const BSDFVelvetPrepared prepared = bsdf_velvet_prepare(context, data, material);
    const BSDFVelvetWalk child = bsdf_velvet_walk_begin_escape(context, data.spectrum_sample, prepared, -data.w_i, material.emission_collimation);
    wavefront_emission_store_velvet(buffer, offset + kGPUEmissionRequestChildOffset, child);
    buffer.Store(offset + kGPUEmissionRequestOperationOffset, uint(GPUEmissionRequestOperation::Velvet));
    return;
  }
  const BSDFCoatedConductor layer = bsdf_coated_conductor_prepare(context, data, material);
  wavefront_emission_store_coating(buffer, offset + kGPUEmissionRequestCoatingOffset, bsdf_coating_escape_begin(context, layer));
  buffer.Store(offset + kGPUEmissionRequestOperationOffset, uint(GPUEmissionRequestOperation::Coating));
}

void wavefront_emission_coating_step(uint index, uint operation) {
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_request_buffer);
  uint offset;
  if (wavefront_emission_active(buffer, index, resources.emission_request_stride, offset) == false) {
    return;
  }
  if ((buffer.Load(offset + kGPUEmissionRequestOperationOffset) != GPUEmissionRequestOperation::Coating) ||
      (buffer.Load(offset + kGPUEmissionRequestCoatingOffset + kGPUEmissionCoatingOperationOffset) != operation)) {
    return;
  }
  const BSDFResourceContext context = wavefront_emission_context();
  const BSDFData data = wavefront_emission_data(buffer, offset);
  const Material material = wavefront_emission_material(buffer, offset);
  const BSDFCoatedConductor layer = bsdf_coated_conductor_prepare(context, data, material);
  BSDFCoatingEscapeState state = wavefront_emission_load_coating(buffer, offset + kGPUEmissionRequestCoatingOffset);
  Sampler sampler = wavefront_emission_load_sampler(buffer, offset + kGPUEmissionRequestSamplerOffset);
  switch (operation) {
#if (ETX_EMISSION_STAGE == 0) || (ETX_EMISSION_STAGE == ETX_EMISSION_STAGE_wavefront_emission_sample_begin_main)
    case BSDFCoatingEscapeState::SampleBegin: {
      BSDFEnergyCompensatedDielectricSampleState sample_state;
      bsdf_coating_escape_sample_begin(context, layer, state, sample_state, sampler);
      wavefront_emission_store_interface(buffer, offset + kGPUEmissionRequestChildOffset, sample_state);
      break;
    }
#endif
#if (ETX_EMISSION_STAGE == 0) || (ETX_EMISSION_STAGE == ETX_EMISSION_STAGE_wavefront_emission_sample_delta_main)
    case BSDFCoatingEscapeState::SampleDelta:
      bsdf_coating_escape_sample_delta(context, layer, state, sampler);
      break;
#endif
#if (ETX_EMISSION_STAGE == 0) || (ETX_EMISSION_STAGE == ETX_EMISSION_STAGE_wavefront_emission_sample_resolve_main)
    case BSDFCoatingEscapeState::SampleResolve: {
      const BSDFEnergyCompensatedDielectricSampleState sample_state = wavefront_emission_load_interface(buffer, offset + kGPUEmissionRequestChildOffset);
      bsdf_coating_escape_sample_resolve(context, layer, state, sample_state);
      break;
    }
#endif
#if (ETX_EMISSION_STAGE == 0) || (ETX_EMISSION_STAGE == ETX_EMISSION_STAGE_wavefront_emission_advance_main)
    case BSDFCoatingEscapeState::Advance:
      bsdf_coating_escape_advance(layer, state, sampler);
      break;
#endif
#if (ETX_EMISSION_STAGE == 0) || (ETX_EMISSION_STAGE == ETX_EMISSION_STAGE_wavefront_emission_phase_main)
    case BSDFCoatingEscapeState::Phase:
      bsdf_coating_escape_phase(layer, state, sampler);
      break;
#endif
#if (ETX_EMISSION_STAGE == 0) || (ETX_EMISSION_STAGE == ETX_EMISSION_STAGE_wavefront_emission_base_sample_main)
    case BSDFCoatingEscapeState::BaseSample: {
      BSDFVelvetWalk child;
      bsdf_coating_escape_base_sample(context, layer, state, child, sampler);
      wavefront_emission_store_velvet(buffer, offset + kGPUEmissionRequestChildOffset, child);
      break;
    }
#endif
#if (ETX_EMISSION_STAGE == 0) || (ETX_EMISSION_STAGE == ETX_EMISSION_STAGE_wavefront_emission_velvet_sample_main)
    case BSDFCoatingEscapeState::VelvetSample: {
      BSDFVelvetWalk child = wavefront_emission_load_velvet(buffer, offset + kGPUEmissionRequestChildOffset);
      bsdf_coating_escape_velvet_sample(context, layer, state, child, sampler);
      wavefront_emission_store_velvet(buffer, offset + kGPUEmissionRequestChildOffset, child);
      break;
    }
#endif
#if (ETX_EMISSION_STAGE == 0) || (ETX_EMISSION_STAGE == ETX_EMISSION_STAGE_wavefront_emission_base_emission_main)
    case BSDFCoatingEscapeState::BaseEmission: {
      BSDFCoatingRenewalPrepared renewal;
      bool renewed = false;
      if (bsdf_coating_renewal_prepare(context, layer, sampler, renewal)) {
        renewed = bsdf_coating_renewal_escape_step(renewal, layer, state, sampler);
      }
      if (renewed == false) {
        BSDFVelvetWalk child;
        bsdf_coating_escape_base_emission(context, layer, state, child, sampler);
        wavefront_emission_store_velvet(buffer, offset + kGPUEmissionRequestChildOffset, child);
      }
      break;
    }
#endif
#if (ETX_EMISSION_STAGE == 0) || (ETX_EMISSION_STAGE == ETX_EMISSION_STAGE_wavefront_emission_velvet_escape_main)
    case BSDFCoatingEscapeState::VelvetEscape: {
      BSDFVelvetWalk child = wavefront_emission_load_velvet(buffer, offset + kGPUEmissionRequestChildOffset);
      bsdf_coating_escape_velvet_escape(context, layer, state, child, sampler);
      wavefront_emission_store_velvet(buffer, offset + kGPUEmissionRequestChildOffset, child);
      break;
    }
#endif
#if (ETX_EMISSION_STAGE == 0) || (ETX_EMISSION_STAGE == ETX_EMISSION_STAGE_wavefront_emission_base_continue_main)
    case BSDFCoatingEscapeState::BaseContinue:
      bsdf_coating_escape_base_continue(layer, state);
      break;
#endif
  }
  wavefront_emission_store_coating(buffer, offset + kGPUEmissionRequestCoatingOffset, state);
  wavefront_emission_store_sampler(buffer, offset + kGPUEmissionRequestSamplerOffset, sampler);
}

[numthreads(64, 1, 1)] void wavefront_emission_sample_begin_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  wavefront_emission_coating_step(request_index, BSDFCoatingEscapeState::SampleBegin);
}

  [numthreads(64, 1, 1)] void wavefront_emission_sample_delta_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  wavefront_emission_coating_step(request_index, BSDFCoatingEscapeState::SampleDelta);
}

[numthreads(64, 1, 1)] void wavefront_emission_sample_resolve_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  wavefront_emission_coating_step(request_index, BSDFCoatingEscapeState::SampleResolve);
}

  [numthreads(64, 1, 1)] void wavefront_emission_advance_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  wavefront_emission_coating_step(request_index, BSDFCoatingEscapeState::Advance);
}

[numthreads(64, 1, 1)] void wavefront_emission_phase_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  wavefront_emission_coating_step(request_index, BSDFCoatingEscapeState::Phase);
}

  [numthreads(64, 1, 1)] void wavefront_emission_base_sample_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  wavefront_emission_coating_step(request_index, BSDFCoatingEscapeState::BaseSample);
}

[numthreads(64, 1, 1)] void wavefront_emission_velvet_sample_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  wavefront_emission_coating_step(request_index, BSDFCoatingEscapeState::VelvetSample);
}

  [numthreads(64, 1, 1)] void wavefront_emission_base_emission_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  wavefront_emission_coating_step(request_index, BSDFCoatingEscapeState::BaseEmission);
}

[numthreads(64, 1, 1)] void wavefront_emission_base_continue_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  wavefront_emission_coating_step(request_index, BSDFCoatingEscapeState::BaseContinue);
}

  [numthreads(64, 1, 1)] void wavefront_emission_velvet_escape_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_request_buffer);
  uint offset;
  if (wavefront_emission_active(buffer, request_index, resources.emission_request_stride, offset) == false) {
    return;
  }
  const uint operation = buffer.Load(offset + kGPUEmissionRequestOperationOffset);
  if (operation == GPUEmissionRequestOperation::Coating) {
    wavefront_emission_coating_step(request_index, BSDFCoatingEscapeState::VelvetEscape);
    return;
  }
  if ((operation != GPUEmissionRequestOperation::Velvet) ||
      (buffer.Load(offset + kGPUEmissionRequestChildOffset + kGPUEmissionVelvetOperationOffset) == BSDFVelvetWalk::Complete)) {
    return;
  }
  const BSDFResourceContext context = wavefront_emission_context();
  const BSDFData data = wavefront_emission_data(buffer, offset);
  const Material material = wavefront_emission_material(buffer, offset);
  const BSDFVelvetPrepared prepared = bsdf_velvet_prepare(context, data, material);
  BSDFVelvetWalk child = wavefront_emission_load_velvet(buffer, offset + kGPUEmissionRequestChildOffset);
  Sampler sampler = wavefront_emission_load_sampler(buffer, offset + kGPUEmissionRequestSamplerOffset);
  bsdf_velvet_walk_escape_step(context, data.spectrum_sample, prepared, child, sampler);
  wavefront_emission_store_velvet(buffer, offset + kGPUEmissionRequestChildOffset, child);
  wavefront_emission_store_sampler(buffer, offset + kGPUEmissionRequestSamplerOffset, sampler);
}

[numthreads(64, 1, 1)] void wavefront_emission_finish_main(uint3 id : SV_DispatchThreadID) {
  uint request_index;
  if (wavefront_transport_work_request(id, request_index) == false) {
    return;
  }
  const GPUWavefrontResources resources = wavefront_load_resources();
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(resources.emission_request_buffer);
  uint offset;
  if (wavefront_emission_active(buffer, request_index, resources.emission_request_stride, offset) == false) {
    return;
  }
  const uint operation = buffer.Load(offset + kGPUEmissionRequestOperationOffset);
  SpectralResponse source;
  SpectralResponse absorption;
  if (operation == GPUEmissionRequestOperation::Coating) {
    if (buffer.Load(offset + kGPUEmissionRequestCoatingOffset + kGPUEmissionCoatingOperationOffset) != BSDFCoatingEscapeState::Complete) {
      return;
    }
    source = wavefront_emission_load_response(buffer, offset + kGPUEmissionRequestCoatingOffset);
    absorption = wavefront_emission_load_response(buffer, offset + kGPUEmissionRequestCoatingOffset + 24u);
  } else if (operation == GPUEmissionRequestOperation::Velvet) {
    if (buffer.Load(offset + kGPUEmissionRequestChildOffset + kGPUEmissionVelvetOperationOffset) != BSDFVelvetWalk::Complete) {
      return;
    }
    source = wavefront_emission_load_response(buffer, offset + kGPUEmissionRequestChildOffset);
    absorption = wavefront_emission_load_response(buffer, offset + kGPUEmissionRequestChildOffset + 24u);
  } else {
    return;
  }
  const BSDFResourceContext context = wavefront_emission_context();
  const BSDFData data = wavefront_emission_data(buffer, offset);
  const Material material = wavefront_emission_material(buffer, offset);
  const SpectralResponse authored = spectral_response_mul(source, bsdf_resource_apply_image(context, data.spectrum_sample, material.emission, data.tex));
  const SpectralResponse thermal = spectral_response_mul(absorption, bsdf_resource_load_spectrum(context, material.coating.thermal_spectrum_index, data.spectrum_sample));
  wavefront_emission_publish(buffer, offset, spectral_response_add(authored, thermal), wavefront_emission_load_sampler(buffer, offset + kGPUEmissionRequestSamplerOffset));
}
