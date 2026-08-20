#include "bindless.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>

#define ETX_WAVEFRONT_BSDF_KIND_DIFFUSE    1
#define ETX_WAVEFRONT_BSDF_KIND_PLASTIC    2
#define ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR  3
#define ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC 4

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
# if ETX_ENABLE_VELVET_STAGE
#  include <interop/bsdf_velvet_shared.hxx>
# else
#  include <interop/bsdf_various_shared.hxx>
# endif
# if ETX_ENABLE_THINFILM_STAGE
#  include <interop/bsdf_dielectric_shared.hxx>
# endif
# define ETX_STAGE_BSDF_EVAL wavefront_vcm_merge_various_eval
# define ETX_STAGE_BSDF_PDF wavefront_vcm_merge_various_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_PLASTIC)
# include <interop/bsdf_plastic_shared.hxx>
# define ETX_STAGE_BSDF_EVAL bsdf_plastic_evaluate
# define ETX_STAGE_BSDF_PDF bsdf_plastic_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
# if ETX_ENABLE_OPENPBR_STAGE
#  include <interop/bsdf_openpbr_shared.hxx>
# else
#  include <interop/bsdf_energy_compensated_shared.hxx>
# endif
# define ETX_STAGE_BSDF_EVAL wavefront_vcm_merge_conductor_eval
# define ETX_STAGE_BSDF_PDF wavefront_vcm_merge_conductor_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)
# include <interop/bsdf_energy_compensated_shared.hxx>
# define ETX_STAGE_BSDF_EVAL bsdf_dielectric_energy_compensated_evaluate
# define ETX_STAGE_BSDF_PDF bsdf_dielectric_energy_compensated_pdf
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
BSDFEval wavefront_vcm_merge_various_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  switch (material.cls) {
    case MaterialClass::Translucent: return bsdf_translucent_evaluate(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Mirror: return bsdf_mirror_evaluate(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Boundary: return bsdf_boundary_evaluate(context, data, outgoing_direction, material, sampler);
# if ETX_ENABLE_THINFILM_STAGE
    case MaterialClass::Thinfilm: return bsdf_thinfilm_evaluate(context, data, outgoing_direction, material, sampler);
# endif
# if ETX_ENABLE_VELVET_STAGE
    case MaterialClass::Velvet: return bsdf_velvet_evaluate(context, data, outgoing_direction, material, sampler);
# endif
    case MaterialClass::Void: return bsdf_void_evaluate(context, data, outgoing_direction, material, sampler);
    case MaterialClass::DiffractionGrating: return bsdf_diffraction_grating_evaluate(context, data, outgoing_direction, material, sampler);
    default: return bsdf_diffuse_evaluate(context, data, outgoing_direction, material, sampler);
  }
}
float wavefront_vcm_merge_various_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  switch (material.cls) {
    case MaterialClass::Translucent: return bsdf_translucent_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Mirror: return bsdf_mirror_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Boundary: return bsdf_boundary_pdf(context, data, outgoing_direction, material, sampler);
# if ETX_ENABLE_THINFILM_STAGE
    case MaterialClass::Thinfilm: return bsdf_thinfilm_pdf(context, data, outgoing_direction, material, sampler);
# endif
# if ETX_ENABLE_VELVET_STAGE
    case MaterialClass::Velvet: return bsdf_velvet_pdf(context, data, outgoing_direction, material, sampler);
# endif
    case MaterialClass::Void: return bsdf_void_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::DiffractionGrating: return bsdf_diffraction_grating_pdf(context, data, outgoing_direction, material, sampler);
    default: return bsdf_diffuse_pdf(context, data, outgoing_direction, material, sampler);
  }
}
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
BSDFEval wavefront_vcm_merge_conductor_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
# if ETX_ENABLE_OPENPBR_STAGE
  if (material.cls == MaterialClass::OpenPBR) return bsdf_openpbr_evaluate(context, data, outgoing_direction, material, sampler);
# endif
  return bsdf_conductor_energy_compensated_evaluate(context, data, outgoing_direction, material, sampler);
}
float wavefront_vcm_merge_conductor_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
# if ETX_ENABLE_OPENPBR_STAGE
  if (material.cls == MaterialClass::OpenPBR) return bsdf_openpbr_pdf(context, data, outgoing_direction, material, sampler);
# endif
  return bsdf_conductor_energy_compensated_pdf(context, data, outgoing_direction, material, sampler);
}
#endif

BSDFEval wavefront_vcm_merge_stage_bsdf_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_EVAL(context, data, outgoing_direction, material, sampler);
}
float wavefront_vcm_merge_stage_reverse_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  float3 reverse_outgoing = -data.w_i;
  data.w_i = -outgoing_direction;
  data.path_source = PathSource::Light;
  return ETX_STAGE_BSDF_PDF(context, data, reverse_outgoing, material, sampler);
}
bool wavefront_vcm_merge_stage_matches_material(uint material_class) {
#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
  return (material_class == MaterialClass::Diffuse) || (material_class == MaterialClass::Translucent) || (material_class == MaterialClass::Mirror) ||
         (material_class == MaterialClass::Boundary) || (material_class == MaterialClass::Void) || (material_class == MaterialClass::DiffractionGrating)
# if ETX_ENABLE_THINFILM_STAGE
         || (material_class == MaterialClass::Thinfilm)
# endif
# if ETX_ENABLE_VELVET_STAGE
         || (material_class == MaterialClass::Velvet)
# endif
         ;
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
# if ETX_ENABLE_OPENPBR_STAGE
  return (material_class == MaterialClass::Conductor) || (material_class == MaterialClass::OpenPBR);
# else
  return material_class == MaterialClass::Conductor;
# endif
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_PLASTIC)
  return material_class == MaterialClass::Plastic;
#else
  return material_class == MaterialClass::Dielectric;
#endif
}

#include "gpu_rt_wavefront_vcm_merge_common.hlsl"

[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 dtid : SV_DispatchThreadID) {
  wavefront_vcm_merge(dtid.x);
}
