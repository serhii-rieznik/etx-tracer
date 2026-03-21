#include "bindless.hlsl"
#include <interop/image_filter_shared.hxx>
#include <access/bsdf_resource_gpu.hxx>
#include <interop/bsdf_external_shared.hxx>

float wavefront_direct_light_stage_bsdf_pdf(BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  (void)context;
  (void)data;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;
  return 0.0f;
}

#include "gpu_rt_wavefront_direct_light_prepare_common.hlsl"

BSDFEval wavefront_direct_light_dielectric_eval_external(
  BSDFResourceContext context, BSDFData data, float3 outgoing_direction, Material material, inout Sampler sampler) {
  LocalFrame local_frame = ETX_ZERO(LocalFrame);
  local_frame.tan = data.tan;
  local_frame.btn = data.btn;
  local_frame.nrm = data.nrm;

  float3 w_i = local_frame_to_local(local_frame, -data.w_i);
  if (abs(LocalFrame::cos_theta(w_i)) <= kEpsilon) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  float3 w_o = local_frame_to_local(local_frame, outgoing_direction);
  if (abs(LocalFrame::cos_theta(w_o)) <= kEpsilon) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);

  SpectralResponse value = spectral_response_make(data.spectrum_sample, 0.0f);
  if (LocalFrame::cos_theta(w_i) > 0.0f) {
    if (LocalFrame::cos_theta(w_o) >= 0.0f) {
      value = bsdf_external_eval_dielectric(data.spectrum_sample, sampler, w_i, w_o, true, roughness, ext_ior, int_ior, thinfilm);
    } else {
      value = bsdf_external_eval_dielectric(data.spectrum_sample, sampler, w_i, w_o, false, roughness, ext_ior, int_ior, thinfilm);
    }
  } else if (LocalFrame::cos_theta(w_o) <= 0.0f) {
    value = bsdf_external_eval_dielectric(data.spectrum_sample, sampler, -w_i, -w_o, true, roughness, int_ior, ext_ior, thinfilm);
  } else {
    value = bsdf_external_eval_dielectric(data.spectrum_sample, sampler, -w_i, -w_o, false, roughness, int_ior, ext_ior, thinfilm);
  }

  if (spectral_response_is_zero(value)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  bool reflection = (LocalFrame::cos_theta(w_i) * LocalFrame::cos_theta(w_o)) > 0.0f;
  SpectralImage scattering_image = material.scattering;
  if (reflection) {
    scattering_image = material.reflectance;
  }

  BSDFEval eval = ETX_ZERO(BSDFEval);
  eval.func = spectral_response_mul(spectral_response_mul(value, 2.0f), bsdf_resource_apply_image(context, data.spectrum_sample, scattering_image, data.tex));
  eval.bsdf = spectral_response_mul(eval.func, abs(LocalFrame::cos_theta(w_o)));
  eval.pdf = 0.0f;
  eval.eta = 1.0f;
  return eval;
}

void wavefront_store_direct_light_dielectric_partial_task(
  uint dispatch_index, WavefrontDirectLightPrepareInput input_value, ETX_IN(BSDFEval, bsdf_eval), ETX_IN(Sampler, sampler)) {
  if ((bsdf_eval_valid(bsdf_eval) == false) || (wavefront_valid_spectral_response(bsdf_eval.bsdf) == false)) {
    return;
  }

  GPUWavefrontDirectLightTask task = (GPUWavefrontDirectLightTask)0;
  task.contribution = bsdf_eval.bsdf;
  task.pixel_index = input_value.current_vertex.pixel_index;
  task.medium_index = input_value.current_vertex.medium_index;
  task.sampler_seed = sampler.seed;
  task.flags = 2u;
  wavefront_store_direct_light_task(input_value.resources.direct_light_task_buffer, dispatch_index, task);
}

[numthreads(64, 1, 1)] void wavefront_camera_direct_light_prepare_dielectric_eval_main(uint3 dtid : SV_DispatchThreadID) {
  WavefrontDirectLightPrepareInput input_value = (WavefrontDirectLightPrepareInput)0;
  if (wavefront_load_direct_light_prepare_input(dtid.x, input_value) == false) {
    return;
  }
  if (input_value.material.cls != MaterialClass::Dielectric) {
    return;
  }

  Sampler bsdf_sampler = wavefront_make_bsdf_sampler(input_value.state.sampler_seed);
  bsdf_sampler_push_fixed(bsdf_sampler, input_value.state.film_uv.x, input_value.state.film_uv.y, input_value.state.last_emitter_pdf);
  BSDFData bsdf_data = wavefront_make_surface_bsdf_data(input_value.hit.vertex, input_value.state.spect, input_value.current_vertex.medium_index, input_value.current_vertex.w_i);
  BSDFEval bsdf_eval =
    wavefront_direct_light_dielectric_eval_external(wavefront_make_scene_bsdf_resource_gpu_context(), bsdf_data, input_value.sample_value.direction, input_value.material, bsdf_sampler);
  wavefront_store_direct_light_dielectric_partial_task(dtid.x, input_value, bsdf_eval, bsdf_sampler);
}
