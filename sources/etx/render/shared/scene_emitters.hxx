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

SpectralResponse emitter_evaluate_out_local(const Emitter& em_inst, SpectralQuery spect, const float2& uv, const float3& emitter_normal, const float3& direction, float& pdf_area,
  float& pdf_dir, float& pdf_dir_out);

SpectralResponse emitter_get_radiance(const Emitter& em_inst, SpectralQuery spect, const EmitterRadianceQuery& query, float& pdf_area, float& pdf_dir, float& pdf_dir_out);

SpectralResponse emitter_evaluate_out_dist(const Emitter& em_inst, SpectralQuery spect, const float3& in_direction, float& pdf_area, float& pdf_dir);
bool emitter_distribution_has_values();
uint32_t sample_emitter_distribution(Sampler& smp, float& pdf_sample);
bool try_load_emitter_instance(uint32_t emitter_index, Emitter& emitter);
EmitterSample emitter_sample_in(const Emitter& em_inst, SpectralQuery spect, const float3& from_point, const float2& smp);
EmitterSample sample_emission_from_emitter(const Emitter& em_inst, SpectralQuery spect, Sampler& smp);
float emitter_discrete_pdf(const Emitter& emitter);

float emitter_sample_pdf(const Emitter& em_inst, ETX_IN(float3, in_direction));

float2 emitter_environment_pdf(ETX_IN(float3, in_direction), bool target_is_surface, uint32_t target_triangle_index);

ETX_SHARED_INLINE float emitter_ris_candidate_weight(const EmitterSample& emitter_sample, const EmitterSampleQuery& query) {
  float radiance_weight = emitter_sample.value.luminance();
  if (radiance_weight <= 0.0f) {
    return 0.0f;
  }

  float source_alignment = 1.0f;
  float3 to_emitter = emitter_sample.origin - query.source_position;
  float len_sq = dot(to_emitter, to_emitter);

  if (query.source_type == InteractionType::Surface) {
    source_alignment = fabsf(dot(query.source_normal, to_emitter) / sqrtf(len_sq));
  }

  if (emitter_sample.is_distant) {
    return radiance_weight * source_alignment;
  }

  float emitter_orientation = dot(emitter_sample.normal, -to_emitter);
  if ((emitter_orientation <= 0.0f) || (len_sq <= kEpsilon)) {
    return 0.0f;
  }

  float distance_weight = 1.0f / fmaxf(1.0f, len_sq);
  return radiance_weight * distance_weight * (emitter_orientation / sqrtf(len_sq)) * source_alignment;
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
      sample.is_distant = emitter.is_distant();

      float candidate_weight = emitter_ris_candidate_weight(sample, query);
      float weight = candidate_weight / pdf_sample;

      weight_sum += weight;
      if (smp.next() * weight_sum < weight) {
        selected_sample = sample;
        selected_weight = weight;
      }
    }

    if (selected_weight <= 0.0f) {
      return {};
    }

    selected_sample.value *= weight_sum / (float(candidate_count) * selected_weight);
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
  result.is_distant = emitter.is_distant();
  return result;
}

}  // namespace etx
