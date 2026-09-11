#include "bindless.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>

#define ETX_WAVEFRONT_BSDF_KIND_DIFFUSE    1
#define ETX_WAVEFRONT_BSDF_KIND_PLASTIC    2
#define ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR  3
#define ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC 4

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
# define ETX_UPBP_SURFACE_QUERY_FAMILY GPUUPBPSurfaceQueryFamily::Various
# if ETX_ENABLE_VELVET_STAGE
#  include <interop/bsdf_velvet_shared.hxx>
# else
#  include <interop/bsdf_various_shared.hxx>
# endif
# define ETX_UPBP_SURFACE_EVAL upbp_surface_various_eval
# define ETX_UPBP_SURFACE_PDF  upbp_surface_various_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_PLASTIC)
# define ETX_UPBP_SURFACE_QUERY_FAMILY GPUUPBPSurfaceQueryFamily::Plastic
# include <interop/bsdf_plastic_shared.hxx>
# define ETX_UPBP_SURFACE_EVAL bsdf_plastic_evaluate
# define ETX_UPBP_SURFACE_PDF  bsdf_plastic_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
# define ETX_UPBP_SURFACE_QUERY_FAMILY GPUUPBPSurfaceQueryFamily::Conductor
# if ETX_ENABLE_OPENPBR_STAGE
#  include <interop/bsdf_openpbr_shared.hxx>
# else
#  include <interop/bsdf_energy_compensated_shared.hxx>
# endif
# define ETX_UPBP_SURFACE_EVAL upbp_surface_conductor_eval
# define ETX_UPBP_SURFACE_PDF  upbp_surface_conductor_pdf
#elif (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)
# define ETX_UPBP_SURFACE_QUERY_FAMILY GPUUPBPSurfaceQueryFamily::Dielectric
# include <interop/bsdf_energy_compensated_shared.hxx>
# define ETX_UPBP_SURFACE_EVAL upbp_surface_dielectric_eval
# define ETX_UPBP_SURFACE_PDF  upbp_surface_dielectric_pdf
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
BSDFEval upbp_surface_various_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  switch (material.cls) {
    case MaterialClass::Translucent:
      return bsdf_translucent_evaluate(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_evaluate(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_evaluate(context, data, outgoing_direction, material, sampler);
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

float upbp_surface_various_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  switch (material.cls) {
    case MaterialClass::Translucent:
      return bsdf_translucent_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Mirror:
      return bsdf_mirror_pdf(context, data, outgoing_direction, material, sampler);
    case MaterialClass::Boundary:
      return bsdf_boundary_pdf(context, data, outgoing_direction, material, sampler);
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

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)
BSDFEval upbp_surface_dielectric_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
# if ETX_ENABLE_THINFILM_STAGE
  if (material.cls == MaterialClass::Thinfilm) {
    return bsdf_thinfilm_evaluate(context, data, outgoing_direction, material, sampler);
  }
# endif
  return bsdf_dielectric_energy_compensated_evaluate(context, data, outgoing_direction, material, sampler);
}

float upbp_surface_dielectric_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
# if ETX_ENABLE_THINFILM_STAGE
  if (material.cls == MaterialClass::Thinfilm) {
    return bsdf_thinfilm_pdf(context, data, outgoing_direction, material, sampler);
  }
# endif
  return bsdf_dielectric_energy_compensated_pdf(context, data, outgoing_direction, material, sampler);
}
#endif

#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR)
BSDFEval upbp_surface_conductor_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
# if ETX_ENABLE_OPENPBR_STAGE
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_evaluate(context, data, outgoing_direction, material, sampler);
  }
# endif
  return bsdf_conductor_energy_compensated_evaluate(context, data, outgoing_direction, material, sampler);
}

float upbp_surface_conductor_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
# if ETX_ENABLE_OPENPBR_STAGE
  if (material.cls == MaterialClass::OpenPBR) {
    return bsdf_openpbr_pdf(context, data, outgoing_direction, material, sampler);
  }
# endif
  return bsdf_conductor_energy_compensated_pdf(context, data, outgoing_direction, material, sampler);
}
#endif

BSDFEval upbp_surface_stage_bsdf_eval(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return ETX_UPBP_SURFACE_EVAL(context, data, outgoing_direction, material, sampler);
}

float upbp_surface_stage_reverse_pdf(BSDFResourceContext context, BSDFData input_data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  const float3 reverse_outgoing = -input_data.w_i;
  input_data.w_i = -outgoing_direction;
  input_data.path_source = PathSource::Light;
  return ETX_UPBP_SURFACE_PDF(context, input_data, reverse_outgoing, material, sampler);
}

bool upbp_surface_stage_matches_material(uint material_class) {
#if (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIFFUSE)
  return (material_class == MaterialClass::Diffuse) || (material_class == MaterialClass::Translucent) || (material_class == MaterialClass::Mirror) ||
         (material_class == MaterialClass::Boundary) || (material_class == MaterialClass::Void) || (material_class == MaterialClass::DiffractionGrating)
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
  return (material_class == MaterialClass::Dielectric)
# if ETX_ENABLE_THINFILM_STAGE
         || (material_class == MaterialClass::Thinfilm)
# endif
    ;
#endif
}

struct UPBPSurfacePreparedQuery {
  Material material;
  BSDFData data;
  BSDFResourceContext context;
  bool valid;
#if ((ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR) || (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)) && (ETX_SPECTRAL_MODE != ETX_SPECTRAL_MODE_RUNTIME)
  bool use_prepared_material;
  BSDFEnergyCompensatedPreparedMaterial prepared_material;
# if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR
  bool use_prepared_conductor;
  LocalFrame frame;
  float3 local_w_i;
  SpectralResponse reflectance;
  BSDFEnergyCompensatedConductorIncident incident;
# endif
#endif
};

UPBPSurfacePreparedQuery upbp_surface_prepare_query(BSDFResourceContext context, Material material, BSDFData data) {
  UPBPSurfacePreparedQuery result = (UPBPSurfacePreparedQuery)0;
  result.material = material;
  result.data = data;
  result.context = context;
  result.valid = upbp_surface_stage_matches_material(material.cls);
#if ((ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR) || (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)) && (ETX_SPECTRAL_MODE != ETX_SPECTRAL_MODE_RUNTIME)
  result.use_prepared_material = (material.cls == MaterialClass::Conductor) || (material.cls == MaterialClass::Dielectric);
  if (result.use_prepared_material) {
    Sampler sampler = (Sampler)0;
    result.prepared_material = bsdf_energy_compensated_prepare_material(result.context, data.spectrum_sample, material, data.tex, sampler);
# if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR
    result.frame = bsdf_data_get_normal_frame(data, material);
    result.local_w_i = local_frame_to_local(result.frame, -data.w_i);
    result.use_prepared_conductor = (result.prepared_material.conductor_delta == false) && (result.local_w_i.z > kEpsilon) &&
                                    bsdf_energy_compensated_material_interface_valid(result.context, material, MaterialClass::Conductor);
    if (result.use_prepared_conductor) {
      result.reflectance = bsdf_resource_apply_image(result.context, data.spectrum_sample, material.reflectance, data.tex);
      result.incident = bsdf_energy_compensated_conductor_prepare_incident(result.context, data.spectrum_sample, material, result.local_w_i.z, result.prepared_material);
    }
# endif
  }
#endif
  return result;
}

[noinline] BSDFEval upbp_surface_evaluate_prepared(UPBPSurfacePreparedQuery query, float3 outgoing_direction, inout Sampler sampler, out float reverse_pdf) {
  BSDFData reverse_data = query.data;
  reverse_data.w_i = -outgoing_direction;
  reverse_data.path_source = PathSource::Light;
#if ((ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR) || (ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_DIELECTRIC)) && (ETX_SPECTRAL_MODE != ETX_SPECTRAL_MODE_RUNTIME)
  if (query.use_prepared_material) {
# if ETX_BSDF_KIND == ETX_WAVEFRONT_BSDF_KIND_CONDUCTOR
    reverse_pdf = bsdf_conductor_energy_compensated_pdf_prepared(query.context, reverse_data, -query.data.w_i, query.material, query.prepared_material);
    if (query.use_prepared_conductor) {
      return bsdf_conductor_energy_compensated_evaluate_prepared_local(query.context, query.data, query.material, query.local_w_i,
        local_frame_to_local(query.frame, outgoing_direction), query.prepared_material, query.reflectance, query.incident);
    }
    return bsdf_conductor_energy_compensated_evaluate_prepared(query.context, query.data, outgoing_direction, query.material, query.prepared_material);
# else
    reverse_pdf = bsdf_dielectric_energy_compensated_pdf_prepared(query.context, reverse_data, -query.data.w_i, query.material, query.prepared_material);
    return bsdf_dielectric_energy_compensated_evaluate_prepared(query.context, query.data, outgoing_direction, query.material, query.prepared_material);
# endif
  }
#endif
  const BSDFEval evaluation = upbp_surface_stage_bsdf_eval(query.context, query.data, outgoing_direction, query.material, sampler);
  reverse_pdf = upbp_surface_stage_reverse_pdf(query.context, query.data, outgoing_direction, query.material, sampler);
  return evaluation;
}

#define ETX_UPBP_SURFACE_VARIANT 1
#include "gpu_rt_wavefront_upbp_density.hlsl"

  [numthreads(64, 1, 1)] void ETX_STAGE_ENTRY(uint3 group_id : SV_GroupID, uint group_thread_index : SV_GroupIndex) {
  if (group_id.x < constants.dispatch_item_count) {
    upbp_evaluate_surface_point_merge_group(constants.dispatch_item_offset + group_id.x, group_thread_index);
  }
}
