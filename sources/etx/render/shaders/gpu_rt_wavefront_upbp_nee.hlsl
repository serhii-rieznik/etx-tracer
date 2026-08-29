#pragma once

bool upbp_bpt_nee_competitor_terms(float pdf_sample, float pdf_dir, float pdf_dir_out, bool delta, float scattering_pdf_forward, float camera_cosine, float light_cosine,
  out float w_light, out float emission_to_direct_ratio) {
  w_light = 0.0f;
  emission_to_direct_ratio = 0.0f;
  if ((pdf_sample <= 0.0f) || (pdf_dir <= 0.0f) || (pdf_dir_out <= 0.0f) || (scattering_pdf_forward <= 0.0f) || (camera_cosine <= 0.0f) || (light_cosine <= 0.0f)) {
    return false;
  }
  const float direct_density = pdf_sample * pdf_dir;
  w_light = delta ? 0.0f : scattering_pdf_forward / direct_density;
  emission_to_direct_ratio = pdf_dir_out * camera_cosine / (pdf_dir * light_cosine);
  return isfinite(w_light) && (w_light >= 0.0f) && isfinite(emission_to_direct_ratio) && (emission_to_direct_ratio > 0.0f);
}
