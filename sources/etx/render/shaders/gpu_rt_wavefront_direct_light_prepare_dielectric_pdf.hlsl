#include "bindless.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>
#include <interop/bsdf_dielectric_shared.hxx>

float wavefront_direct_light_stage_bsdf_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  return bsdf_dielectric_pdf(context, data, outgoing_direction, material, sampler);
}

#include "gpu_rt_wavefront_direct_light_prepare_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_camera_direct_light_prepare_dielectric_pdf_main(uint3 dtid : SV_DispatchThreadID) {
  WavefrontDirectLightPrepareInput input_value = (WavefrontDirectLightPrepareInput)0;
  if (wavefront_load_direct_light_prepare_input(dtid.x, input_value) == false) {
    return;
  }
  if (input_value.material.cls != MaterialClass::Dielectric) {
    return;
  }

  GPUWavefrontDirectLightTask partial_task = wavefront_load_direct_light_task(input_value.resources.direct_light_task_buffer, dtid.x);
  if (partial_task.flags != 2u) {
    return;
  }

  GPUWavefrontDirectLightTask cleared_task = (GPUWavefrontDirectLightTask)0;
  cleared_task.medium_index = kInvalidIndex;
  wavefront_store_direct_light_task(input_value.resources.direct_light_task_buffer, dtid.x, cleared_task);

  Sampler bsdf_sampler = wavefront_make_bsdf_sampler(partial_task.sampler_seed);
  bsdf_sampler_push_fixed(bsdf_sampler, input_value.state.film_uv.x, input_value.state.film_uv.y, input_value.state.last_emitter_pdf);
  BSDFData bsdf_data = wavefront_make_surface_bsdf_data(input_value.hit.vertex, input_value.state.spect, input_value.current_vertex.medium_index, input_value.current_vertex.w_i);

  BSDFEval bsdf_eval = (BSDFEval)0;
  bsdf_eval.bsdf = partial_task.contribution;
  bsdf_eval.pdf = bsdf_dielectric_pdf(
    wavefront_make_scene_bsdf_resource_gpu_context(), bsdf_data, input_value.sample_value.direction, input_value.material, bsdf_sampler);
  bsdf_eval.eta = 1.0f;

  wavefront_store_direct_light_prepare_task(dtid.x, input_value, bsdf_eval, bsdf_sampler);
}
