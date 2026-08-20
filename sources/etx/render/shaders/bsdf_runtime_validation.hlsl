#include "bindless.hlsl"

#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>
#include <access/material_access_gpu.hxx>

#ifndef ETX_BSDF_RUNTIME_VALIDATION_MODE
  #define ETX_BSDF_RUNTIME_VALIDATION_MODE 0
#endif

#ifndef ETX_BSDF_RUNTIME_VALIDATION_OPERATION
  #define ETX_BSDF_RUNTIME_VALIDATION_OPERATION 0
#endif

#define ETX_BSDF_RUNTIME_VALIDATION_KIND_PLASTIC     1
#define ETX_BSDF_RUNTIME_VALIDATION_KIND_CONDUCTOR   2
#define ETX_BSDF_RUNTIME_VALIDATION_KIND_DIELECTRIC  3
#define ETX_BSDF_RUNTIME_VALIDATION_KIND_DIFFRACTION 4

#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 3
  #include <interop/bsdf_energy_compensated_shared.hxx>
  #include <interop/bsdf_diffraction_grating_shared.hxx>
#elif ETX_BSDF_RUNTIME_VALIDATION_MODE == 2
  #include <interop/bsdf_openpbr_shared.hxx>
#elif ETX_BSDF_RUNTIME_VALIDATION_MODE == 1
  #include <interop/bsdf_plastic_shared.hxx>
#else
  #include <interop/bsdf_energy_compensated_shared.hxx>
#endif

struct BSDFRuntimeValidationPushConstants {
  uint case_buffer_index;
  uint output_buffer_index;
  uint materials_descriptor_index;
  uint images_descriptor_index;
  uint spectrums_descriptor_index;
  uint energy_compensation_interfaces_descriptor_index;
  uint scene_globals_descriptor_index;
  uint case_count;
};

[[vk::push_constant]] BSDFRuntimeValidationPushConstants constants;

static const uint kCaseStride = 32u;
static const uint kOutputStride = 148u;

uint validation_load_u32(ByteAddressBuffer buffer, uint case_index, uint offset) {
  return buffer.Load(case_index * kCaseStride + offset);
}

float validation_load_f32(ByteAddressBuffer buffer, uint case_index, uint offset) {
  return asfloat(buffer.Load(case_index * kCaseStride + offset));
}

void validation_store_f32(RWByteAddressBuffer buffer, uint base_offset, uint offset, float value) {
  buffer.Store(base_offset + offset, asuint(value));
}

void validation_store_f32x3(RWByteAddressBuffer buffer, uint base_offset, uint offset, float3 value) {
  buffer.Store3(base_offset + offset, asuint(value));
}

void validation_store_u32(RWByteAddressBuffer buffer, uint base_offset, uint offset, uint value) {
  buffer.Store(base_offset + offset, value);
}

BSDFData validation_data() {
  BSDFData result = (BSDFData)0;
  result.pos = float3(0.0f, 0.0f, 0.0f);
  result.nrm = float3(0.0f, 0.0f, 1.0f);
  result.tan = float3(1.0f, 0.0f, 0.0f);
  result.btn = float3(0.0f, 1.0f, 0.0f);
  result.tex = float2(0.5f, 0.5f);
  result.w_i = float3(0.0f, 0.0f, -1.0f);
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 3
  result.spectrum_sample.wavelength = 550.0f;
  result.spectrum_sample.flags = SpectralFlags::Spectral;
#else
  result.spectrum_sample = spectral_query_sample();
#endif
  result.path_source = PathSource::Camera;
  result.current_medium = kInvalidIndex;
  return result;
}

Sampler validation_sampler(uint seed, float fixed_u, float fixed_v, float fixed_w) {
  Sampler result = (Sampler)0;
  result.seed = seed;
  result.fixed_u = fixed_u;
  result.fixed_v = fixed_v;
  result.fixed_w = fixed_w;
  return result;
}

#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 1)
BSDFSample validation_sample(BSDFResourceContext context, BSDFData data, Material material, inout Sampler sampler) {
#if defined(ETX_BSDF_RUNTIME_VALIDATION_KIND)
# if ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_CONDUCTOR
  return bsdf_conductor_energy_compensated_sample(context, data, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_DIELECTRIC
  return bsdf_dielectric_energy_compensated_sample(context, data, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_PLASTIC
  return bsdf_plastic_sample(context, data, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_DIFFRACTION
  return bsdf_diffraction_grating_sample(context, data, material, sampler);
# endif
#else
  if (material.cls == MaterialClass::Conductor) {
    return bsdf_conductor_energy_compensated_sample(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::Dielectric) {
    return bsdf_dielectric_energy_compensated_sample(context, data, material, sampler);
  }
#if (ETX_BSDF_RUNTIME_VALIDATION_MODE == 1) || (ETX_BSDF_RUNTIME_VALIDATION_MODE == 2)
  if (material.cls == MaterialClass::Plastic) {
    return bsdf_plastic_sample(context, data, material, sampler);
  }
#endif
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 2
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_sample(context, data, material, sampler);
  }
#endif
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 3
  if (material.cls == MaterialClass::DiffractionGrating) {
    return bsdf_diffraction_grating_sample(context, data, material, sampler);
  }
#endif
  return bsdf_sample_zero(data.spectrum_sample);
#endif
}
#endif

#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 2)
BSDFEval validation_evaluate(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
#if defined(ETX_BSDF_RUNTIME_VALIDATION_KIND)
# if ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_CONDUCTOR
  return bsdf_conductor_energy_compensated_evaluate(context, data, outgoing_direction, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_DIELECTRIC
  return bsdf_dielectric_energy_compensated_evaluate(context, data, outgoing_direction, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_PLASTIC
  return bsdf_plastic_evaluate(context, data, outgoing_direction, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_DIFFRACTION
  return bsdf_diffraction_grating_evaluate(context, data, outgoing_direction, material, sampler);
# endif
#else
  if (material.cls == MaterialClass::Conductor) {
    return bsdf_conductor_energy_compensated_evaluate(context, data, outgoing_direction, material, sampler);
  }
  if (material.cls == MaterialClass::Dielectric) {
    return bsdf_dielectric_energy_compensated_evaluate(context, data, outgoing_direction, material, sampler);
  }
#if (ETX_BSDF_RUNTIME_VALIDATION_MODE == 1) || (ETX_BSDF_RUNTIME_VALIDATION_MODE == 2)
  if (material.cls == MaterialClass::Plastic) {
    return bsdf_plastic_evaluate(context, data, outgoing_direction, material, sampler);
  }
#endif
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 2
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_evaluate(context, data, outgoing_direction, material, sampler);
  }
#endif
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 3
  if (material.cls == MaterialClass::DiffractionGrating) {
    return bsdf_diffraction_grating_evaluate(context, data, outgoing_direction, material, sampler);
  }
#endif
  return bsdf_eval_zero(data.spectrum_sample);
#endif
}
#endif

#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 3)
float validation_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
#if defined(ETX_BSDF_RUNTIME_VALIDATION_KIND)
# if ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_CONDUCTOR
  return bsdf_conductor_energy_compensated_pdf(context, data, outgoing_direction, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_DIELECTRIC
  return bsdf_dielectric_energy_compensated_pdf(context, data, outgoing_direction, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_PLASTIC
  return bsdf_plastic_pdf(context, data, outgoing_direction, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_DIFFRACTION
  return bsdf_diffraction_grating_pdf(context, data, outgoing_direction, material, sampler);
# endif
#else
  if (material.cls == MaterialClass::Conductor) {
    return bsdf_conductor_energy_compensated_pdf(context, data, outgoing_direction, material, sampler);
  }
  if (material.cls == MaterialClass::Dielectric) {
    return bsdf_dielectric_energy_compensated_pdf(context, data, outgoing_direction, material, sampler);
  }
#if (ETX_BSDF_RUNTIME_VALIDATION_MODE == 1) || (ETX_BSDF_RUNTIME_VALIDATION_MODE == 2)
  if (material.cls == MaterialClass::Plastic) {
    return bsdf_plastic_pdf(context, data, outgoing_direction, material, sampler);
  }
#endif
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 2
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_pdf(context, data, outgoing_direction, material, sampler);
  }
#endif
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 3
  if (material.cls == MaterialClass::DiffractionGrating) {
    return bsdf_diffraction_grating_pdf(context, data, outgoing_direction, material, sampler);
  }
#endif
  return 0.0f;
#endif
}

float validation_reverse_pdf(BSDFResourceContext context, BSDFData input_data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  float3 reverse_w_o = -input_data.w_i;
  BSDFData reverse_data = input_data;
  reverse_data.w_i = -outgoing_direction;
  if (input_data.path_source == PathSource::Camera) {
    reverse_data.path_source = PathSource::Light;
  } else if (input_data.path_source == PathSource::Light) {
    reverse_data.path_source = PathSource::Camera;
  }
  return validation_pdf(context, reverse_data, reverse_w_o, material, sampler);
}
#endif

#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 4)
bool validation_is_delta(BSDFResourceContext context, Material material, float2 tex, inout Sampler sampler) {
#if defined(ETX_BSDF_RUNTIME_VALIDATION_KIND)
# if ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_CONDUCTOR
  return bsdf_conductor_energy_compensated_is_delta_with_context(context, material, tex);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_DIELECTRIC
  return bsdf_dielectric_energy_compensated_is_delta_with_context(context, material, tex);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_PLASTIC
  return bsdf_plastic_is_delta(material, tex, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_DIFFRACTION
  return bsdf_diffraction_grating_is_delta(material, tex, sampler);
# endif
#else
  if (material.cls == MaterialClass::Conductor) {
    return bsdf_conductor_energy_compensated_is_delta_with_context(context, material, tex);
  }
  if (material.cls == MaterialClass::Dielectric) {
    return bsdf_dielectric_energy_compensated_is_delta_with_context(context, material, tex);
  }
#if (ETX_BSDF_RUNTIME_VALIDATION_MODE == 1) || (ETX_BSDF_RUNTIME_VALIDATION_MODE == 2)
  if (material.cls == MaterialClass::Plastic) {
    return bsdf_plastic_is_delta(material, tex, sampler);
  }
#endif
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 2
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_is_delta(material, tex, sampler);
  }
#endif
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 3
  if (material.cls == MaterialClass::DiffractionGrating) {
    return bsdf_diffraction_grating_is_delta(material, tex, sampler);
  }
#endif
  return false;
#endif
}

SpectralResponse validation_albedo(BSDFResourceContext context, BSDFData data, Material material, inout Sampler sampler) {
#if defined(ETX_BSDF_RUNTIME_VALIDATION_KIND)
# if ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_CONDUCTOR
  return bsdf_conductor_energy_compensated_albedo(context, data, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_DIELECTRIC
  return bsdf_dielectric_energy_compensated_albedo(context, data, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_PLASTIC
  return bsdf_plastic_albedo(context, data, material, sampler);
# elif ETX_BSDF_RUNTIME_VALIDATION_KIND == ETX_BSDF_RUNTIME_VALIDATION_KIND_DIFFRACTION
  return bsdf_diffraction_grating_albedo(context, data, material, sampler);
# endif
#else
  if (material.cls == MaterialClass::Conductor) {
    return bsdf_conductor_energy_compensated_albedo(context, data, material, sampler);
  }
  if (material.cls == MaterialClass::Dielectric) {
    return bsdf_dielectric_energy_compensated_albedo(context, data, material, sampler);
  }
#if (ETX_BSDF_RUNTIME_VALIDATION_MODE == 1) || (ETX_BSDF_RUNTIME_VALIDATION_MODE == 2)
  if (material.cls == MaterialClass::Plastic) {
    return bsdf_plastic_albedo(context, data, material, sampler);
  }
#endif
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 2
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_albedo(context, data, material, sampler);
  }
#endif
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 3
  if (material.cls == MaterialClass::DiffractionGrating) {
    return bsdf_diffraction_grating_albedo(context, data, material, sampler);
  }
#endif
  return spectral_response_zero(data.spectrum_sample);
#endif
}
#endif

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  const uint case_index = id.x;
  if (case_index >= constants.case_count) {
    return;
  }

  ByteAddressBuffer case_buffer = bindless_buffers[NonUniformResourceIndex(constants.case_buffer_index)];
  RWByteAddressBuffer output_buffer = bindless_rw_buffers[NonUniformResourceIndex(constants.output_buffer_index)];
  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(constants.materials_descriptor_index)];

  const uint material_index = validation_load_u32(case_buffer, case_index, 0u);
  const uint seed = validation_load_u32(case_buffer, case_index, 4u);
  const float fixed_u = validation_load_f32(case_buffer, case_index, 8u);
  const float fixed_v = validation_load_f32(case_buffer, case_index, 12u);
  const float fixed_w = validation_load_f32(case_buffer, case_index, 16u);

  Material material = gpu_abi_load_material_full(material_buffer, material_index);
  BSDFData data = validation_data();
#if ETX_BSDF_RUNTIME_VALIDATION_MODE == 3
  const float diffraction_tangent = data.spectrum_sample.wavelength / material.diffraction_grating.period_nm;
  const float3 outgoing_direction = float3(diffraction_tangent, 0.0f, sqrt(max(0.0f, 1.0f - diffraction_tangent * diffraction_tangent)));
#else
  const float3 outgoing_direction = normalize(float3(0.35f, 0.0f, 0.9367497f));
#endif
  BSDFResourceContext context = make_bsdf_resource_gpu_context(constants.images_descriptor_index, constants.spectrums_descriptor_index,
    constants.energy_compensation_interfaces_descriptor_index, constants.scene_globals_descriptor_index);

#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 1)
  Sampler sample_sampler = validation_sampler(seed, fixed_u, fixed_v, fixed_w);
  const BSDFSample sample = validation_sample(context, data, material, sample_sampler);
#endif

#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 2)
  Sampler eval_sampler = validation_sampler(seed + 1u, fixed_u, fixed_v, fixed_w);
  const BSDFEval eval = validation_evaluate(context, data, outgoing_direction, material, eval_sampler);
#endif

#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 3)
  Sampler pdf_sampler = validation_sampler(seed + 2u, fixed_u, fixed_v, fixed_w);
  const float pdf = validation_pdf(context, data, outgoing_direction, material, pdf_sampler);

  Sampler reverse_pdf_sampler = validation_sampler(seed + 3u, fixed_u, fixed_v, fixed_w);
  const float reverse_pdf = validation_reverse_pdf(context, data, outgoing_direction, material, reverse_pdf_sampler);
#endif

#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 4)
  Sampler delta_sampler = validation_sampler(seed + 4u, fixed_u, fixed_v, fixed_w);
  const bool is_delta = validation_is_delta(context, material, data.tex, delta_sampler);

  Sampler albedo_sampler = validation_sampler(seed + 5u, fixed_u, fixed_v, fixed_w);
  const SpectralResponse albedo = validation_albedo(context, data, material, albedo_sampler);
#endif

  const uint base_offset = case_index * kOutputStride;
#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 1)
  validation_store_f32x3(output_buffer, base_offset, 0u, sample.weight.integrated);
  validation_store_f32(output_buffer, base_offset, 12u, sample.weight.value);
  validation_store_f32x3(output_buffer, base_offset, 16u, sample.w_o);
  validation_store_f32(output_buffer, base_offset, 28u, sample.pdf);
  validation_store_f32(output_buffer, base_offset, 32u, sample.eta);
  validation_store_u32(output_buffer, base_offset, 36u, sample.properties);
  validation_store_u32(output_buffer, base_offset, 40u, sample.medium_index);
  validation_store_u32(output_buffer, base_offset, 44u, sample_sampler.seed);
#endif

#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 2)
  validation_store_f32x3(output_buffer, base_offset, 48u, eval.func.integrated);
  validation_store_f32(output_buffer, base_offset, 60u, eval.func.value);
  validation_store_f32x3(output_buffer, base_offset, 64u, eval.bsdf.integrated);
  validation_store_f32(output_buffer, base_offset, 76u, eval.bsdf.value);
  validation_store_f32(output_buffer, base_offset, 80u, eval.pdf);
  validation_store_f32(output_buffer, base_offset, 84u, eval.eta);
  validation_store_u32(output_buffer, base_offset, 88u, eval.properties);
  validation_store_u32(output_buffer, base_offset, 92u, eval.medium_index);
  validation_store_u32(output_buffer, base_offset, 96u, eval_sampler.seed);
#endif

#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 3)
  validation_store_f32(output_buffer, base_offset, 100u, pdf);
  validation_store_u32(output_buffer, base_offset, 104u, pdf_sampler.seed);
  validation_store_f32(output_buffer, base_offset, 108u, reverse_pdf);
  validation_store_u32(output_buffer, base_offset, 112u, reverse_pdf_sampler.seed);
#endif

#if (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 0) || (ETX_BSDF_RUNTIME_VALIDATION_OPERATION == 4)
  validation_store_u32(output_buffer, base_offset, 116u, is_delta ? 1u : 0u);
  validation_store_u32(output_buffer, base_offset, 120u, delta_sampler.seed);

  validation_store_f32x3(output_buffer, base_offset, 128u, albedo.integrated);
  validation_store_f32(output_buffer, base_offset, 140u, albedo.value);
  validation_store_u32(output_buffer, base_offset, 144u, albedo_sampler.seed);
#endif
}
