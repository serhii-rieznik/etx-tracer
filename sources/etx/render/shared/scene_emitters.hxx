#pragma once

#include <etx/render/interop/projection.hxx>

namespace etx {

bool try_load_emitter_instance_count_shared(uint32_t& emitter_instance_count);
uint32_t environment_emitter_shared_count();
bool environment_emitter_shared_try_load_index(uint32_t local_index, uint32_t& emitter_index);
bool try_select_environment_emitter_random(Sampler& smp, uint32_t& emitter_index, uint32_t& emitter_count);

ETX_SHARED_INLINE float emitter_pdf_area_local(const Emitter& em) {
  ETX_ASSERT(em.is_local());
  return 1.0f / em.triangle_area;
}

uint32_t emitter_external_medium_index(const Emitter& em_inst);

SpectralResponse emitter_evaluate_out_local(const Emitter& em_inst, SpectralQuery spect, const Vertex& vertex, const float3& emitter_normal, const float3& direction,
  float& pdf_area, float& pdf_dir, float& pdf_dir_out);

SpectralResponse emitter_get_radiance(const Emitter& em_inst, SpectralQuery spect, const EmitterRadianceQuery& query, float& pdf_area, float& pdf_dir, float& pdf_dir_out);

SpectralResponse emitter_evaluate_out_dist(const Emitter& em_inst, SpectralQuery spect, const float3& in_direction, float& pdf_area, float& pdf_dir);
bool emitter_distribution_has_values();
uint32_t sample_emitter_distribution(Sampler& smp, float& pdf_sample);
bool try_load_emitter_instance(uint32_t emitter_index, Emitter& emitter);
EmitterSample emitter_sample_in(const Emitter& em_inst, SpectralQuery spect, const float3& from_point, const float2& smp);
EmitterSample sample_emission_from_emitter(const Emitter& em_inst, SpectralQuery spect, Sampler& smp);
float emitter_discrete_pdf(const Emitter& emitter);

float emitter_sample_pdf(const Emitter& em_inst, ETX_IN(float3, in_direction));

float2 emitter_environment_pdf(ETX_IN(float3, in_direction), bool target_is_surface, uint32_t target_triangle_index, uint32_t target_instance_index);

ETX_SHARED_INLINE float emitter_ris_candidate_weight(const EmitterSample& emitter_sample, const EmitterSampleQuery& query) {
  const float radiance_weight = emitter_sample.value.luminance();
  const float proposal_pdf = emitter_sample.pdf_sample * emitter_sample.pdf_dir;
  if ((radiance_weight <= 0.0f) || (proposal_pdf <= 0.0f)) {
    return 0.0f;
  }

  // Use incoming solid angle: an area light's directional PDF already contains its geometry Jacobian.
  const float source_alignment = (query.source_type == InteractionType::Surface) ? fabsf(dot(query.source_normal, emitter_sample.direction)) : 1.0f;
  return (radiance_weight * source_alignment) / proposal_pdf;
}

ETX_SHARED_INLINE bool scene_has_only_environment_emitters() {
  uint32_t emitter_count = 0u;
  if (try_load_emitter_instance_count_shared(emitter_count) == false) {
    return false;
  }

  uint32_t environment_emitter_count = environment_emitter_shared_count();
  return (emitter_count > 0u) && (environment_emitter_count == emitter_count);
}

ETX_SHARED_INLINE EmitterSample sample_emitter(Scene::LightSampling sampling_method, const EmitterSampleQuery& query, Sampler& smp) {
  if (emitter_distribution_has_values() == false) {
    return {};
  }

  uint32_t emitter_instance_count = 0u;
  if (try_load_emitter_instance_count_shared(emitter_instance_count) == false) {
    return {};
  }

  if ((sampling_method == Scene::LightSampling::RIS_Uniform) || (sampling_method == Scene::LightSampling::RIS_FromDistribution)) {
    uint32_t candidate_count = min(4u * emitter_instance_count, 16u);

    float weight_sum = 0.0f;
    float selected_weight = 0.0f;
    EmitterSample selected_sample = {};

    for (uint32_t i = 0; i < candidate_count; ++i) {
      float pdf_sample = 0.0f;
      uint32_t emitter_index = kInvalidIndex;

      if (sampling_method == Scene::LightSampling::RIS_FromDistribution) {
        emitter_index = sample_emitter_distribution(smp, pdf_sample);
        if (emitter_index == kInvalidIndex) {
          continue;
        }
      } else {
        if (scene_has_only_environment_emitters()) {
          uint32_t emitter_count = 0u;
          if (try_select_environment_emitter_random(smp, emitter_index, emitter_count) == false) {
            continue;
          }
          pdf_sample = 1.0f / float(emitter_count);
        } else {
          emitter_index = uint32_t(smp.next() * float(emitter_instance_count));
          pdf_sample = 1.0f / float(emitter_instance_count);
        }
      }

      Emitter emitter = {};
      if (try_load_emitter_instance(emitter_index, emitter) == false) {
        continue;
      }

      EmitterSample sample = emitter_sample_in(emitter, query.spect, query.source_position, smp.next_2d());
      sample.pdf_sample = pdf_sample;
      sample.emitter_index = emitter_index;
      sample.triangle_index = emitter.triangle_index;
      sample.is_delta = emitter.is_delta();
      sample.is_sample_only = emitter.is_sample_only();
      sample.is_distant = emitter.is_distant();

      const float weight = emitter_ris_candidate_weight(sample, query);

      weight_sum += weight;
      float reservoir_weight = smp.next() * weight_sum;
      if (reservoir_weight < weight) {
        selected_sample = sample;
        selected_weight = weight;
      }
    }

    if (selected_weight <= 0.0f) {
      return {};
    }

    // Retain the proposal PDFs for MIS; this scale preserves the average candidate estimator.
    float reservoir_scale = weight_sum / (float(candidate_count) * selected_weight);
    selected_sample.value *= reservoir_scale;
    return selected_sample;
  }

  float pdf_sample = 0.0f;
  uint32_t emitter_index = kInvalidIndex;

  if (sampling_method == Scene::LightSampling::FromDistribution) {
    emitter_index = sample_emitter_distribution(smp, pdf_sample);
    if (emitter_index == kInvalidIndex) {
      return {};
    }
  } else {
    if (scene_has_only_environment_emitters()) {
      uint32_t emitter_count = 0u;
      if (try_select_environment_emitter_random(smp, emitter_index, emitter_count) == false) {
        return {};
      }
      pdf_sample = 1.0f / float(emitter_count);
    } else {
      emitter_index = uint32_t(smp.next() * float(emitter_instance_count));
      pdf_sample = 1.0f / float(emitter_instance_count);
    }
  }

  Emitter emitter = {};
  if (try_load_emitter_instance(emitter_index, emitter) == false) {
    return {};
  }

  EmitterSample sample = emitter_sample_in(emitter, query.spect, query.source_position, smp.next_2d());
  sample.pdf_sample = pdf_sample;
  sample.emitter_index = emitter_index;
  sample.triangle_index = emitter.triangle_index;
  sample.is_delta = emitter.is_delta();
  sample.is_sample_only = emitter.is_sample_only();
  sample.is_distant = emitter.is_distant();
  return sample;
}

ETX_SHARED_INLINE const EmitterSample sample_emission(SpectralQuery spect, Sampler& smp) {
  if (emitter_distribution_has_values() == false) {
    return {};
  }

  float pdf_sample = 0.0f;
  uint32_t emitter_index = sample_emitter_distribution(smp, pdf_sample);
  if (emitter_index == kInvalidIndex) {
    return {};
  }

  Emitter emitter = {};
  if (try_load_emitter_instance(emitter_index, emitter) == false) {
    return {};
  }

  EmitterSample result = sample_emission_from_emitter(emitter, spect, smp);
  result.pdf_sample = pdf_sample;
  result.emitter_index = emitter_index;
  result.triangle_index = emitter.triangle_index;
  result.is_delta = emitter.is_delta();
  result.is_sample_only = emitter.is_sample_only();
  result.is_distant = emitter.is_distant();
  return result;
}

}  // namespace etx
