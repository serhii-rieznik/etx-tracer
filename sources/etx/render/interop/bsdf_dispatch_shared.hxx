#pragma once

#include "bsdf_conductor_shared.hxx"
#include "bsdf_dielectric_shared.hxx"
#if (ETX_CPP)
#include "bsdf_energy_compensated_shared.hxx"
#endif
#include "bsdf_plastic_shared.hxx"
#include "bsdf_various_shared.hxx"
#include "bsdf_velvet_shared.hxx"

ETX_SHARED_INLINE bool bsdf_gpu_supported_class(uint32_t material_class) {
  switch (material_class) {
    case MaterialClass::Diffuse:
    case MaterialClass::Translucent:
    case MaterialClass::Mirror:
    case MaterialClass::Boundary:
    case MaterialClass::Conductor:
    case MaterialClass::Dielectric:
    case MaterialClass::Plastic:
    case MaterialClass::Thinfilm:
    case MaterialClass::Velvet:
    case MaterialClass::Void: {
      return true;
    }

    default: {
      return false;
    }
  }
}

ETX_SHARED_INLINE BSDFSample bsdf_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const Material effective_material = material;
  switch (effective_material.cls) {
    case MaterialClass::Diffuse:
      return bsdf_diffuse_sample(context, data, effective_material, sampler);
    case MaterialClass::Translucent:
      return bsdf_translucent_sample(context, data, effective_material, sampler);
    case MaterialClass::Conductor:
#if (ETX_CPP)
      return bsdf_conductor_energy_compensated_sample(context, data, effective_material, sampler);
#else
      return bsdf_diffuse_sample(context, data, effective_material, sampler);
#endif
    case MaterialClass::Dielectric:
#if (ETX_CPP)
      return bsdf_dielectric_energy_compensated_sample(context, data, effective_material, sampler);
#else
      return bsdf_diffuse_sample(context, data, effective_material, sampler);
#endif
    case MaterialClass::Plastic:
      return bsdf_plastic_sample(context, data, effective_material, sampler);
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_sample(context, data, effective_material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_sample(context, data, effective_material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_sample(context, data, effective_material, sampler);
    case MaterialClass::Velvet:
      return bsdf_velvet_sample(context, data, effective_material, sampler);
    case MaterialClass::Void:
      return bsdf_void_sample(context, data, effective_material, sampler);
    default:
      return bsdf_sample_zero(data.spectrum_sample);
  }
}

ETX_SHARED_INLINE BSDFEval bsdf_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const Material effective_material = material;
  switch (effective_material.cls) {
    case MaterialClass::Diffuse:
      return bsdf_diffuse_evaluate(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Translucent:
      return bsdf_translucent_evaluate(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Conductor:
#if (ETX_CPP)
      return bsdf_conductor_energy_compensated_evaluate(context, data, outgoing_direction, effective_material, sampler);
#else
      return bsdf_diffuse_evaluate(context, data, outgoing_direction, effective_material, sampler);
#endif
    case MaterialClass::Dielectric:
#if (ETX_CPP)
      return bsdf_dielectric_energy_compensated_evaluate(context, data, outgoing_direction, effective_material, sampler);
#else
      return bsdf_diffuse_evaluate(context, data, outgoing_direction, effective_material, sampler);
#endif
    case MaterialClass::Plastic:
      return bsdf_plastic_evaluate(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_evaluate(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_evaluate(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_evaluate(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Velvet:
      return bsdf_velvet_evaluate(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Void:
      return bsdf_void_evaluate(context, data, outgoing_direction, effective_material, sampler);
    default:
      return bsdf_eval_zero(data.spectrum_sample);
  }
}

ETX_SHARED_INLINE float bsdf_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const Material effective_material = material;
  switch (effective_material.cls) {
    case MaterialClass::Diffuse:
      return bsdf_diffuse_pdf(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Translucent:
      return bsdf_translucent_pdf(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Conductor:
#if (ETX_CPP)
      return bsdf_conductor_energy_compensated_pdf(context, data, outgoing_direction, effective_material, sampler);
#else
      return bsdf_diffuse_pdf(context, data, outgoing_direction, effective_material, sampler);
#endif
    case MaterialClass::Dielectric:
#if (ETX_CPP)
      return bsdf_dielectric_energy_compensated_pdf(context, data, outgoing_direction, effective_material, sampler);
#else
      return bsdf_diffuse_pdf(context, data, outgoing_direction, effective_material, sampler);
#endif
    case MaterialClass::Plastic:
      return bsdf_plastic_pdf(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_pdf(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_pdf(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_pdf(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Velvet:
      return bsdf_velvet_pdf(context, data, outgoing_direction, effective_material, sampler);
    case MaterialClass::Void:
      return bsdf_void_pdf(context, data, outgoing_direction, effective_material, sampler);
    default:
      return 0.0f;
  }
}

ETX_SHARED_INLINE float bsdf_reverse_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, input_data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  float3 reverse_w_o = -input_data.w_i;
  BSDFData reverse_data = input_data;
  reverse_data.w_i = -outgoing_direction;
  if (input_data.path_source == PathSource::Camera) {
    reverse_data.path_source = PathSource::Light;
  } else if (input_data.path_source == PathSource::Light) {
    reverse_data.path_source = PathSource::Camera;
  }
  return bsdf_pdf(context, reverse_data, reverse_w_o, material, sampler);
}

ETX_SHARED_INLINE bool bsdf_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  switch (material.cls) {
    case MaterialClass::Diffuse:
      return bsdf_diffuse_is_delta(material, tex, sampler);
    case MaterialClass::Translucent:
      return bsdf_translucent_is_delta(material, tex, sampler);
    case MaterialClass::Conductor:
#if (ETX_CPP)
      return bsdf_conductor_is_delta(material, tex, sampler);
#else
      return bsdf_diffuse_is_delta(material, tex, sampler);
#endif
    case MaterialClass::Dielectric:
#if (ETX_CPP)
      return bsdf_dielectric_is_delta(material, tex, sampler);
#else
      return bsdf_diffuse_is_delta(material, tex, sampler);
#endif
    case MaterialClass::Plastic:
      return bsdf_plastic_is_delta(material, tex, sampler);
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_is_delta(material, tex, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_is_delta(material, tex, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_is_delta(material, tex, sampler);
    case MaterialClass::Velvet:
      return bsdf_velvet_is_delta(material, tex, sampler);
    case MaterialClass::Void:
      return bsdf_void_is_delta(material, tex, sampler);
    default:
      return false;
  }
}

ETX_SHARED_INLINE bool bsdf_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  const Material effective_material = material;
  switch (effective_material.cls) {
    case MaterialClass::Diffuse:
      return bsdf_diffuse_is_delta(effective_material, tex, sampler);
    case MaterialClass::Translucent:
      return bsdf_translucent_is_delta(effective_material, tex, sampler);
    case MaterialClass::Conductor:
#if (ETX_CPP)
      return bsdf_conductor_energy_compensated_is_delta_with_context(context, effective_material, tex);
#else
      return bsdf_diffuse_is_delta(effective_material, tex, sampler);
#endif
    case MaterialClass::Dielectric:
#if (ETX_CPP)
      return bsdf_dielectric_energy_compensated_is_delta_with_context(context, effective_material, tex);
#else
      return bsdf_diffuse_is_delta(effective_material, tex, sampler);
#endif
    case MaterialClass::Plastic:
      return bsdf_plastic_is_delta(effective_material, tex, sampler);
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_is_delta(effective_material, tex, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_is_delta(effective_material, tex, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_is_delta(effective_material, tex, sampler);
    case MaterialClass::Velvet:
      return bsdf_velvet_is_delta(effective_material, tex, sampler);
    case MaterialClass::Void:
      return bsdf_void_is_delta(effective_material, tex, sampler);
    default:
      return false;
  }
}

ETX_SHARED_INLINE SpectralResponse bsdf_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const Material effective_material = material;
  switch (effective_material.cls) {
    case MaterialClass::Diffuse:
      return bsdf_diffuse_albedo(context, data, effective_material, sampler);
    case MaterialClass::Translucent:
      return bsdf_translucent_albedo(context, data, effective_material, sampler);
    case MaterialClass::Conductor:
#if (ETX_CPP)
      return bsdf_conductor_energy_compensated_albedo(context, data, effective_material, sampler);
#else
      return bsdf_diffuse_albedo(context, data, effective_material, sampler);
#endif
    case MaterialClass::Dielectric:
#if (ETX_CPP)
      return bsdf_dielectric_energy_compensated_albedo(context, data, effective_material, sampler);
#else
      return bsdf_diffuse_albedo(context, data, effective_material, sampler);
#endif
    case MaterialClass::Plastic:
      return bsdf_plastic_albedo(context, data, effective_material, sampler);
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_albedo(context, data, effective_material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_albedo(context, data, effective_material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_albedo(context, data, effective_material, sampler);
    case MaterialClass::Velvet:
      return bsdf_velvet_albedo(context, data, effective_material, sampler);
    case MaterialClass::Void:
      return bsdf_void_albedo(context, data, effective_material, sampler);
    default:
      return spectral_response_zero(data.spectrum_sample);
  }
}
