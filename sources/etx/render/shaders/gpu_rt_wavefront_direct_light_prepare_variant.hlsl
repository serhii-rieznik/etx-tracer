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
# define ETX_STAGE_BSDF_CLASS MaterialClass::Diffuse
# define ETX_STAGE_BSDF_EVAL  wavefront_direct_light_stage_various_eval
# define ETX_STAGE_BSDF_PDF   wavefront_direct_light_stage_various_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_PLASTIC)
# include <interop/bsdf_plastic_shared.hxx>
# define ETX_STAGE_BSDF_CLASS MaterialClass::Plastic
# define ETX_STAGE_BSDF_EVAL  bsdf_plastic_evaluate
# define ETX_STAGE_BSDF_PDF   bsdf_plastic_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
# if ETX_ENABLE_OPENPBR_STAGE
#  include <interop/bsdf_openpbr_shared.hxx>
# else
#  include <interop/bsdf_energy_compensated_shared.hxx>
# endif
# define ETX_STAGE_BSDF_CLASS MaterialClass::Conductor
# define ETX_STAGE_BSDF_EVAL  wavefront_direct_light_stage_conductor_eval
# define ETX_STAGE_BSDF_PDF   wavefront_direct_light_stage_conductor_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)
# include <interop/bsdf_energy_compensated_shared.hxx>
# define ETX_STAGE_BSDF_CLASS MaterialClass::Dielectric
# define ETX_STAGE_BSDF_EVAL  bsdf_dielectric_energy_compensated_evaluate
# define ETX_STAGE_BSDF_PDF   bsdf_dielectric_energy_compensated_pdf
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
BSDFEval wavefront_direct_light_stage_various_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  switch (material.cls) {
    case MaterialClass::Translucent:
      return bsdf_translucent_evaluate(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_evaluate(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_evaluate(context, data, outgoing_direction, material, sampler);
# if ETX_ENABLE_THINFILM_STAGE
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_evaluate(context, data, outgoing_direction, material, sampler);
# endif
# if ETX_ENABLE_VELVET_STAGE
    case MaterialClass::Velvet:
      return bsdf_velvet_evaluate(context, data, outgoing_direction, material, sampler);
# endif
    case MaterialClass::Void:
      return bsdf_void_evaluate(context, data, outgoing_direction, material, sampler);
    case MaterialClass::DiffractionGrating:
      return bsdf_diffraction_grating_evaluate(context, data, outgoing_direction, material, sampler);
    default:
      return bsdf_diffuse_evaluate(context, data, outgoing_direction, material, sampler);
  }
}

#if ETX_WAVEFRONT_PATH_TRACING_ONLY == 0
float wavefront_direct_light_stage_various_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  switch (material.cls) {
    case MaterialClass::Translucent:
      return bsdf_translucent_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_pdf(context, data, outgoing_direction, material, sampler);
# if ETX_ENABLE_THINFILM_STAGE
    case MaterialClass::Thinfilm:
      return bsdf_thinfilm_pdf(context, data, outgoing_direction, material, sampler);
# endif
# if ETX_ENABLE_VELVET_STAGE
    case MaterialClass::Velvet:
      return bsdf_velvet_pdf(context, data, outgoing_direction, material, sampler);
# endif
    case MaterialClass::Void:
      return bsdf_void_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::DiffractionGrating:
      return bsdf_diffraction_grating_pdf(context, data, outgoing_direction, material, sampler);
    default:
      return bsdf_diffuse_pdf(context, data, outgoing_direction, material, sampler);
  }
}
#endif
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
BSDFEval wavefront_direct_light_stage_conductor_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
# if ETX_ENABLE_OPENPBR_STAGE
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_evaluate(context, data, outgoing_direction, material, sampler);
  }
# endif
  return bsdf_conductor_energy_compensated_evaluate(context, data, outgoing_direction, material, sampler);
}

#if ETX_WAVEFRONT_PATH_TRACING_ONLY == 0
float wavefront_direct_light_stage_conductor_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
# if ETX_ENABLE_OPENPBR_STAGE
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_pdf(context, data, outgoing_direction, material, sampler);
  }
# endif
  return bsdf_conductor_energy_compensated_pdf(context, data, outgoing_direction, material, sampler);
}
#endif
#endif

BSDFEval wavefront_direct_light_stage_bsdf_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_EVAL(context, data, outgoing_direction, material, sampler);
}

#if ETX_WAVEFRONT_PATH_TRACING_ONLY == 0
float wavefront_direct_light_stage_bsdf_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return ETX_STAGE_BSDF_PDF(context, data, outgoing_direction, material, sampler);
}
#endif

bool wavefront_direct_light_stage_matches_material(uint material_class) {
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
#else
  return material_class == ETX_STAGE_BSDF_CLASS;
#endif
}

# include "gpu_rt_wavefront_direct_light_prepare_common.hlsl"

[numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 dtid : SV_DispatchThreadID) {
  WavefrontDirectLightPrepareInput input_value = (WavefrontDirectLightPrepareInput)0;
  if (wavefront_load_direct_light_prepare_input(dtid.x, input_value) == false) {
    return;
  }
  if (wavefront_direct_light_stage_matches_material(input_value.material.cls) == false) {
    return;
  }

  Sampler bsdf_sampler = wavefront_make_bsdf_sampler(input_value.state.sampler_seed);
  bsdf_sampler_push_fixed(bsdf_sampler, input_value.state.film_uv.x, input_value.state.film_uv.y, input_value.state.last_emitter_pdf);
  BSDFData bsdf_data = wavefront_make_surface_bsdf_data(input_value.hit.vertex, input_value.state.spect, input_value.current_vertex.medium_index, input_value.current_vertex.w_i);
  BSDFEval bsdf_eval =
    wavefront_direct_light_stage_bsdf_eval(wavefront_make_scene_bsdf_resource_gpu_context(), bsdf_data, input_value.sample_value.direction, input_value.material, bsdf_sampler);
  wavefront_store_direct_light_prepare_task(dtid.x, input_value, bsdf_eval, bsdf_sampler);
}
