#pragma once

#include <etx/rt/integrators/upbp_point_merge.hxx>
#include <etx/rt/integrators/upbp_spatial.hxx>

#if defined(_M_X64) || defined(__SSE2__)
# include <emmintrin.h>
# define ETX_UPBP_HAS_SSE2 1
#else
# define ETX_UPBP_HAS_SSE2 0
#endif

namespace etx {

#if defined(_MSC_VER)
# define ETX_UPBP_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
# define ETX_UPBP_FORCE_INLINE inline __attribute__((always_inline))
#else
# define ETX_UPBP_FORCE_INLINE inline
#endif

struct UPBPBeamContribution {
  SpectralResponse contribution;
  bool applicable;
};

struct UPBPPreparedBeam {
  const UPBPMediumTrackingEventRecord* tracking_events = nullptr;
  SpectralResponse source_throughput = {};
  UPBPBeamTransportPrefix transport_at_origin = {};
  double d_shared = 0.0;
  double d_pde_reverse_coefficient = 0.0;
  double d_pde_constant = 0.0;
  double d_surface_constant = 0.0;
  double source_event_density = 1.0;
  float interval_distance = 0.0f;
  uint32_t tracking_event_count = 0u;
  bool scale_d_shared_by_distance = false;
  bool previous_delta = false;
  bool valid = false;
};

struct UPBPPartialPreparedBeamVertex {
  UPBPPointMergeMISInput::Weights weights;
  SpectralResponse throughput;
  float medium_density;
};

struct UPBPPreparedNoiseOctave {
  std::vector<uint64_t> cell_gradients = {};
  int32_t minimum_x = 0;
  int32_t minimum_y = 0;
  int32_t minimum_z = 0;
  uint32_t dimension_x = 0u;
  uint32_t dimension_y = 0u;
  uint32_t dimension_z = 0u;
  float frequency = 0.0f;
  float amplitude = 0.0f;

  uint64_t storage_bytes() const {
    return static_cast<uint64_t>(cell_gradients.capacity()) * sizeof(uint64_t);
  }
};

struct UPBPPreparedMedium {
  SpectralResponse scattering = {};
  std::array<UPBPPreparedNoiseOctave, 16u> noise_octaves = {};
  double real_event_density_scale = 0.0;
  float phase_constant = 0.0f;
  float phase_one_plus_g_squared = 0.0f;
  float phase_two_g = 0.0f;
  float noise_amplitude_sum = 0.0f;
  uint32_t noise_octave_count = 0u;
  uint32_t noise_type = MediumNoiseType::Perlin;
  bool noise_prepared = false;
  bool valid = false;

  uint64_t storage_bytes() const {
    uint64_t result = 0u;
    for (const UPBPPreparedNoiseOctave& octave : noise_octaves) {
      result += octave.storage_bytes();
    }
    return result;
  }
};

struct UPBPPreparedBB1D {
  double radius_squared = 0.0;
  double inverse_radius_squared = 0.0;
  double kernel_normalization = 0.0;
  double estimator_normalization = 0.0;
  UPBPKernel kernel = UPBPKernel::Epanechnikov;
  bool valid = false;
};

inline bool upbp_prepare_noise_octave(const Medium& medium, const uint32_t octave_index, const float frequency, const float amplitude, UPBPPreparedNoiseOctave& result) {
  result = {};
  const float3 bounds_size = medium.local_bounds.p_max - medium.local_bounds.p_min;
  const float maximum_dimension = max(bounds_size.x, max(bounds_size.y, bounds_size.z));
  if ((maximum_dimension <= kEpsilon) || (std::isfinite(maximum_dimension) == false) || (frequency <= 0.0f) || (std::isfinite(frequency) == false)) {
    return false;
  }
  const float3 normalized_minimum = (medium.local_bounds.p_min + medium.grid.noise_offset) / maximum_dimension;
  const float3 normalized_maximum = (medium.local_bounds.p_max + medium.grid.noise_offset) / maximum_dimension;
  const float3 sample_minimum = min(normalized_minimum * frequency, normalized_maximum * frequency);
  const float3 sample_maximum = max(normalized_minimum * frequency, normalized_maximum * frequency);
  const double minimum_x_value = std::floor(static_cast<double>(sample_minimum.x)) - 1.0;
  const double minimum_y_value = std::floor(static_cast<double>(sample_minimum.y)) - 1.0;
  const double minimum_z_value = std::floor(static_cast<double>(sample_minimum.z)) - 1.0;
  const double maximum_x_value = std::floor(static_cast<double>(sample_maximum.x)) + 1.0;
  const double maximum_y_value = std::floor(static_cast<double>(sample_maximum.y)) + 1.0;
  const double maximum_z_value = std::floor(static_cast<double>(sample_maximum.z)) + 1.0;
  if ((std::isfinite(minimum_x_value) == false) || (std::isfinite(minimum_y_value) == false) || (std::isfinite(minimum_z_value) == false) ||
      (std::isfinite(maximum_x_value) == false) || (std::isfinite(maximum_y_value) == false) || (std::isfinite(maximum_z_value) == false) ||
      (minimum_x_value < std::numeric_limits<int32_t>::min()) || (minimum_y_value < std::numeric_limits<int32_t>::min()) ||
      (minimum_z_value < std::numeric_limits<int32_t>::min()) || (maximum_x_value > std::numeric_limits<int32_t>::max()) ||
      (maximum_y_value > std::numeric_limits<int32_t>::max()) || (maximum_z_value > std::numeric_limits<int32_t>::max())) {
    return false;
  }
  const int64_t minimum_x = static_cast<int64_t>(minimum_x_value);
  const int64_t minimum_y = static_cast<int64_t>(minimum_y_value);
  const int64_t minimum_z = static_cast<int64_t>(minimum_z_value);
  const int64_t maximum_x = static_cast<int64_t>(maximum_x_value);
  const int64_t maximum_y = static_cast<int64_t>(maximum_y_value);
  const int64_t maximum_z = static_cast<int64_t>(maximum_z_value);
  const uint64_t dimension_x = static_cast<uint64_t>(maximum_x - minimum_x + 1ll);
  const uint64_t dimension_y = static_cast<uint64_t>(maximum_y - minimum_y + 1ll);
  const uint64_t dimension_z = static_cast<uint64_t>(maximum_z - minimum_z + 1ll);
  uint64_t cell_count = 0u;
  constexpr uint64_t maximum_cached_cell_count = 16ull * 1024ull * 1024ull;
  if ((dimension_x > std::numeric_limits<uint32_t>::max()) || (dimension_y > std::numeric_limits<uint32_t>::max()) || (dimension_z > std::numeric_limits<uint32_t>::max()) ||
      (dimension_y > std::numeric_limits<uint64_t>::max() / dimension_x)) {
    return false;
  }
  cell_count = dimension_x * dimension_y;
  if ((dimension_z > std::numeric_limits<uint64_t>::max() / cell_count) || ((cell_count *= dimension_z) > maximum_cached_cell_count)) {
    return false;
  }

  result.minimum_x = static_cast<int32_t>(minimum_x);
  result.minimum_y = static_cast<int32_t>(minimum_y);
  result.minimum_z = static_cast<int32_t>(minimum_z);
  result.dimension_x = static_cast<uint32_t>(dimension_x);
  result.dimension_y = static_cast<uint32_t>(dimension_y);
  result.dimension_z = static_cast<uint32_t>(dimension_z);
  result.frequency = frequency;
  result.amplitude = amplitude;
  result.cell_gradients.resize(static_cast<size_t>(cell_count));
  const uint32_t seed = medium.grid.noise_seed + octave_index;
  for (uint32_t z = 0u; z < result.dimension_z; ++z) {
    const uint32_t wrapped_z = medium_density_shared_wrap_255(result.minimum_z + static_cast<int32_t>(z));
    for (uint32_t y = 0u; y < result.dimension_y; ++y) {
      const uint32_t wrapped_y = medium_density_shared_wrap_255(result.minimum_y + static_cast<int32_t>(y));
      for (uint32_t x = 0u; x < result.dimension_x; ++x) {
        const uint32_t wrapped_x = medium_density_shared_wrap_255(result.minimum_x + static_cast<int32_t>(x));
        uint64_t gradients = medium_density_shared_hash3d(wrapped_x, wrapped_y, wrapped_z, seed) & 15u;
        gradients |= static_cast<uint64_t>(medium_density_shared_hash3d(wrapped_x + 1u, wrapped_y, wrapped_z, seed) & 15u) << 8u;
        gradients |= static_cast<uint64_t>(medium_density_shared_hash3d(wrapped_x, wrapped_y + 1u, wrapped_z, seed) & 15u) << 16u;
        gradients |= static_cast<uint64_t>(medium_density_shared_hash3d(wrapped_x + 1u, wrapped_y + 1u, wrapped_z, seed) & 15u) << 24u;
        gradients |= static_cast<uint64_t>(medium_density_shared_hash3d(wrapped_x, wrapped_y, wrapped_z + 1u, seed) & 15u) << 32u;
        gradients |= static_cast<uint64_t>(medium_density_shared_hash3d(wrapped_x + 1u, wrapped_y, wrapped_z + 1u, seed) & 15u) << 40u;
        gradients |= static_cast<uint64_t>(medium_density_shared_hash3d(wrapped_x, wrapped_y + 1u, wrapped_z + 1u, seed) & 15u) << 48u;
        gradients |= static_cast<uint64_t>(medium_density_shared_hash3d(wrapped_x + 1u, wrapped_y + 1u, wrapped_z + 1u, seed) & 15u) << 56u;
        result.cell_gradients[x + y * result.dimension_x + z * result.dimension_x * result.dimension_y] = gradients;
      }
    }
  }
  return true;
}

inline UPBPPreparedMedium upbp_prepare_medium(const Medium& medium, const SpectralQuery spect) {
  UPBPPreparedMedium result = {};
  result.scattering = medium_scattering(medium, spect);
  const SpectralResponse extinction = result.scattering + medium_absorption(medium, spect);
  result.real_event_density_scale = static_cast<double>(extinction.average());
  result.phase_constant = (1.0f / (4.0f * kPi)) * (1.0f - medium.phase_function_g * medium.phase_function_g);
  result.phase_one_plus_g_squared = 1.0f + medium.phase_function_g * medium.phase_function_g;
  result.phase_two_g = 2.0f * medium.phase_function_g;
  result.valid = result.scattering.valid() && extinction.valid() && (result.scattering.minimum() >= 0.0f) && (extinction.minimum() >= 0.0f) &&
                 (result.real_event_density_scale > 0.0) && std::isfinite(result.real_event_density_scale);
  const bool supported_noise = (medium.grid.noise_type == MediumNoiseType::Perlin) || (medium.grid.noise_type == MediumNoiseType::Billow);
  if ((medium.cls != Medium::Heterogeneous) || (medium.grid.type != MediumGridType::NoiseFunction) || (supported_noise == false) ||
      (medium.grid.noise_octaves > result.noise_octaves.size())) {
    return result;
  }
  float frequency = medium.grid.noise_scale;
  float amplitude = 1.0f;
  constexpr float amplitude_threshold = 0.001f;
  const uint32_t octave_count = min(medium.grid.noise_octaves, static_cast<uint32_t>(result.noise_octaves.size()));
  for (uint32_t octave_index = 0u; octave_index < octave_count; ++octave_index) {
    if (amplitude < amplitude_threshold) {
      break;
    }
    if (upbp_prepare_noise_octave(medium, octave_index, frequency, amplitude, result.noise_octaves[result.noise_octave_count]) == false) {
      return result;
    }
    result.noise_amplitude_sum += amplitude;
    ++result.noise_octave_count;
    amplitude *= medium.grid.noise_persistence;
    frequency *= medium.grid.noise_lacunarity;
  }
  result.noise_type = medium.grid.noise_type;
  result.noise_prepared = (result.noise_octave_count > 0u) && (result.noise_octave_count == min(medium.grid.noise_octaves, static_cast<uint32_t>(result.noise_octaves.size())));
  return result;
}

ETX_UPBP_FORCE_INLINE uint64_t upbp_prepared_noise_gradients(const UPBPPreparedNoiseOctave& octave, const int32_t x, const int32_t y, const int32_t z) {
  const uint32_t local_x = static_cast<uint32_t>(x - octave.minimum_x);
  const uint32_t local_y = static_cast<uint32_t>(y - octave.minimum_y);
  const uint32_t local_z = static_cast<uint32_t>(z - octave.minimum_z);
  ETX_ASSERT((local_x < octave.dimension_x) && (local_y < octave.dimension_y) && (local_z < octave.dimension_z));
  const uint64_t index =
    static_cast<uint64_t>(local_x) + static_cast<uint64_t>(local_y) * octave.dimension_x + static_cast<uint64_t>(local_z) * octave.dimension_x * octave.dimension_y;
  return octave.cell_gradients[static_cast<size_t>(index)];
}

#if ETX_UPBP_HAS_SSE2
ETX_UPBP_FORCE_INLINE __m128 upbp_prepared_gradient4(const __m128i hashes, const __m128 x, const __m128 y, const __m128 z) {
  const __m128 less_than_eight = _mm_castsi128_ps(_mm_cmplt_epi32(hashes, _mm_set1_epi32(8)));
  const __m128 less_than_four = _mm_castsi128_ps(_mm_cmplt_epi32(hashes, _mm_set1_epi32(4)));
  const __m128 is_twelve = _mm_castsi128_ps(_mm_cmpeq_epi32(hashes, _mm_set1_epi32(12)));
  const __m128 is_fourteen = _mm_castsi128_ps(_mm_cmpeq_epi32(hashes, _mm_set1_epi32(14)));
  const __m128 use_x_for_v = _mm_or_ps(is_twelve, is_fourteen);
  __m128 u = _mm_or_ps(_mm_and_ps(less_than_eight, x), _mm_andnot_ps(less_than_eight, y));
  const __m128 x_or_z = _mm_or_ps(_mm_and_ps(use_x_for_v, x), _mm_andnot_ps(use_x_for_v, z));
  __m128 v = _mm_or_ps(_mm_and_ps(less_than_four, y), _mm_andnot_ps(less_than_four, x_or_z));
  const __m128 u_sign = _mm_castsi128_ps(_mm_slli_epi32(_mm_and_si128(hashes, _mm_set1_epi32(1)), 31));
  const __m128 v_sign = _mm_castsi128_ps(_mm_slli_epi32(_mm_and_si128(hashes, _mm_set1_epi32(2)), 30));
  u = _mm_xor_ps(u, u_sign);
  v = _mm_xor_ps(v, v_sign);
  return _mm_add_ps(u, v);
}

#endif

ETX_UPBP_FORCE_INLINE float upbp_prepared_perlin_noise(const UPBPPreparedNoiseOctave& octave, const float3& position) {
  const int32_t i = medium_density_shared_floor_to_int(position.x);
  const int32_t j = medium_density_shared_floor_to_int(position.y);
  const int32_t k = medium_density_shared_floor_to_int(position.z);
  const uint64_t gradients = upbp_prepared_noise_gradients(octave, i, j, k);
  const float x = position.x - static_cast<float>(i);
  const float y = position.y - static_cast<float>(j);
  const float z = position.z - static_cast<float>(k);
#if ETX_UPBP_HAS_SSE2
  const __m128 coordinates = _mm_setr_ps(x, y, z, 0.0f);
  const __m128 coordinate_squared = _mm_mul_ps(coordinates, coordinates);
  const __m128 coordinate_cubed = _mm_mul_ps(coordinate_squared, coordinates);
  const __m128 fade_polynomial = _mm_add_ps(_mm_mul_ps(coordinates, _mm_sub_ps(_mm_mul_ps(coordinates, _mm_set1_ps(6.0f)), _mm_set1_ps(15.0f))), _mm_set1_ps(10.0f));
  float fade_values[4u] = {};
  _mm_storeu_ps(fade_values, _mm_mul_ps(coordinate_cubed, fade_polynomial));
  const float u = fade_values[0u];
  const float v = fade_values[1u];
  const float w = fade_values[2u];
  const __m128i gradient_bytes = _mm_cvtsi64_si128(static_cast<int64_t>(gradients));
  const __m128i gradient_words = _mm_unpacklo_epi8(gradient_bytes, _mm_setzero_si128());
  const __m128i bottom_hashes = _mm_unpacklo_epi16(gradient_words, _mm_setzero_si128());
  const __m128i top_hashes = _mm_unpackhi_epi16(gradient_words, _mm_setzero_si128());
  const __m128 corner_x = _mm_setr_ps(x, x - 1.0f, x, x - 1.0f);
  const __m128 corner_y = _mm_setr_ps(y, y, y - 1.0f, y - 1.0f);
  float gradient_values[8u] = {};
  _mm_storeu_ps(gradient_values, upbp_prepared_gradient4(bottom_hashes, corner_x, corner_y, _mm_set1_ps(z)));
  _mm_storeu_ps(gradient_values + 4u, upbp_prepared_gradient4(top_hashes, corner_x, corner_y, _mm_set1_ps(z - 1.0f)));
  const float g000 = gradient_values[0u];
  const float g100 = gradient_values[1u];
  const float g010 = gradient_values[2u];
  const float g110 = gradient_values[3u];
  const float g001 = gradient_values[4u];
  const float g101 = gradient_values[5u];
  const float g011 = gradient_values[6u];
  const float g111 = gradient_values[7u];
#else
  const float u = medium_density_shared_fade(x);
  const float v = medium_density_shared_fade(y);
  const float w = medium_density_shared_fade(z);
  const float g000 = medium_density_shared_gradient3d(gradients & 15u, x, y, z);
  const float g100 = medium_density_shared_gradient3d((gradients >> 8u) & 15u, x - 1.0f, y, z);
  const float g010 = medium_density_shared_gradient3d((gradients >> 16u) & 15u, x, y - 1.0f, z);
  const float g110 = medium_density_shared_gradient3d((gradients >> 24u) & 15u, x - 1.0f, y - 1.0f, z);
  const float g001 = medium_density_shared_gradient3d((gradients >> 32u) & 15u, x, y, z - 1.0f);
  const float g101 = medium_density_shared_gradient3d((gradients >> 40u) & 15u, x - 1.0f, y, z - 1.0f);
  const float g011 = medium_density_shared_gradient3d((gradients >> 48u) & 15u, x, y - 1.0f, z - 1.0f);
  const float g111 = medium_density_shared_gradient3d((gradients >> 56u) & 15u, x - 1.0f, y - 1.0f, z - 1.0f);
#endif
  const float c000 = medium_density_shared_lerp(medium_density_shared_lerp(g000, g100, u), medium_density_shared_lerp(g010, g110, u), v);
  const float c001 = medium_density_shared_lerp(medium_density_shared_lerp(g001, g101, u), medium_density_shared_lerp(g011, g111, u), v);
  return medium_density_shared_lerp(c000, c001, w);
}

ETX_UPBP_FORCE_INLINE float upbp_prepared_medium_density(const Medium& medium, const UPBPPreparedMedium& prepared, const float3& world_position) {
  if (prepared.noise_prepared == false) {
    return medium.sample_density_world(world_position);
  }
  const float3 local_coord = medium_world_to_local(medium.world_to_object, medium.local_bounds, world_position);
  if (medium_local_coordinate_valid(local_coord) == false) {
    return 0.0f;
  }
  const float3 normalized_world_position = medium_density_shared_normalized_world_pos(local_coord, medium.local_bounds.p_min, medium.local_bounds.p_max, medium.grid.noise_offset);
  float value = 0.0f;
  for (uint32_t octave_index = 0u; octave_index < prepared.noise_octave_count; ++octave_index) {
    const UPBPPreparedNoiseOctave& octave = prepared.noise_octaves[octave_index];
    float noise = upbp_prepared_perlin_noise(octave, normalized_world_position * octave.frequency);
    if (prepared.noise_type == MediumNoiseType::Billow) {
      noise = medium_density_shared_abs(noise);
    }
    value += noise * octave.amplitude;
  }
  value = prepared.noise_amplitude_sum > 0.0f ? saturate(value / prepared.noise_amplitude_sum) : 0.0f;
  value = medium_density_shared_apply_border_fade(value, local_coord, medium.grid.noise_enable_border_fade, medium.grid.noise_border_fade_distance);
  value = saturate(value);
  return medium_density_shared_apply_shape(value, medium.grid.noise_power, medium.grid.noise_sharpness);
}

inline UPBPPreparedBB1D upbp_prepare_bb1d(const UPBPKernel kernel, const double radius, const uint64_t light_subpath_count, const double beam_selection_probability) {
  UPBPPreparedBB1D result = {};
  result.kernel = kernel;
  if (((kernel != UPBPKernel::TopHat) && (kernel != UPBPKernel::Epanechnikov)) || (radius <= 0.0) || (std::isfinite(radius) == false) || (light_subpath_count == 0u) ||
      (beam_selection_probability <= 0.0) || (beam_selection_probability > 1.0)) {
    return result;
  }
  result.radius_squared = radius * radius;
  result.inverse_radius_squared = 1.0 / result.radius_squared;
  result.kernel_normalization = kernel == UPBPKernel::TopHat ? 1.0 / (2.0 * radius) : 3.0 / (4.0 * radius);
  result.estimator_normalization = 1.0 / (static_cast<double>(light_subpath_count) * beam_selection_probability);
  result.valid = (result.radius_squared > 0.0) && std::isfinite(result.radius_squared) && std::isfinite(result.inverse_radius_squared) &&
                 std::isfinite(result.kernel_normalization) && std::isfinite(result.estimator_normalization);
  return result;
}

ETX_UPBP_FORCE_INLINE double upbp_evaluate_prepared_bb1d_kernel(const UPBPPreparedBB1D& prepared, const double distance_squared, const double sin_theta) {
  if ((prepared.valid == false) || (distance_squared < 0.0) || (distance_squared >= prepared.radius_squared) || (std::isfinite(distance_squared) == false) || (sin_theta <= 0.0) ||
      (std::isfinite(sin_theta) == false)) {
    return 0.0;
  }
  const double profile = prepared.kernel == UPBPKernel::TopHat ? 1.0 : 1.0 - distance_squared * prepared.inverse_radius_squared;
  return prepared.kernel_normalization * profile / sin_theta;
}

ETX_UPBP_FORCE_INLINE float upbp_prepared_phase_function_from_cosine(const UPBPPreparedMedium& prepared, const float cosine) {
  const float denominator = prepared.phase_one_plus_g_squared - prepared.phase_two_g * cosine;
  return prepared.phase_constant / (denominator * sqrtf(denominator));
}

inline bool upbp_prepare_beam(const UPBPPathRecord& path, const UPBPRecursivePathWeights& path_weights, const UPBPBeamReference& beam,
  const UPBPDensityMISConfiguration& configuration, UPBPPreparedBeam& result) {
  result = {};
  if ((beam.source_vertex_index >= path.vertices.size()) || (beam.source_vertex_index >= path_weights.departures.size()) ||
      (path_weights.has_departure[beam.source_vertex_index] == false)) {
    return false;
  }
  const UPBPTransportSegmentRecord* segment = nullptr;
  if (beam.transport_segment_index < path.segments.size()) {
    segment = &path.segments[beam.transport_segment_index];
  } else if (path.has_terminal_segment && (beam.transport_segment_index == path.segments.size())) {
    segment = &path.terminal_segment;
  }
  if ((segment == nullptr) || (beam.transport_interval_index >= segment->intervals.size())) {
    return false;
  }

  const UPBPPathVertexRecord& source = path.vertices[beam.source_vertex_index];
  const UPBPSegmentRecord& interval = segment->intervals[beam.transport_interval_index];
  if ((interval.valid() == false) || (interval.complete == false) || (interval.medium_index == kInvalidIndex) || interval.events.empty() ||
      (interval.events.size() > std::numeric_limits<uint32_t>::max())) {
    return false;
  }
  result.tracking_events = interval.events.data();
  result.tracking_event_count = static_cast<uint32_t>(interval.events.size());
  result.interval_distance = interval.distance;
  const UPBPRecursiveState& departure = path_weights.departures[beam.source_vertex_index];
  result.d_shared = departure.weights.d_shared;
  result.previous_delta = departure.weights.previous_delta;
  result.source_throughput = source.outgoing_throughput;
  result.transport_at_origin.weight = SpectralResponse{segment->weight.as_query(), 1.0f};
  for (uint32_t interval_index = 0u; interval_index < beam.transport_interval_index; ++interval_index) {
    const UPBPSegmentRecord& interval = segment->intervals[interval_index];
    const bool terminal_medium_event = (interval.terminal_event == MediumTrackingEventType::Scatter) || (interval.terminal_event == MediumTrackingEventType::Absorb);
    if ((interval.valid() == false) || (interval.complete == false) || terminal_medium_event) {
      return false;
    }
    result.transport_at_origin.weight *= interval.weight;
    result.transport_at_origin.log_transport_pdf_forward += interval.log_transport_pdf_forward;
    result.transport_at_origin.log_transport_pdf_reverse += interval.log_transport_pdf_reverse;
    result.transport_at_origin.distance += interval.distance;
  }
  result.source_event_density = source.cls == UPBPVertexClass::Medium ? std::exp(source.log_medium_event_density) : 1.0;
  const double source_ray_ratio = source.cls == UPBPVertexClass::Medium ? 1.0 / result.source_event_density : 0.0;
  result.scale_d_shared_by_distance = (beam.source_vertex_index > 0u) || (path.vertices.front().distant_endpoint == false);
  if (beam.source_vertex_index > 0u) {
    const UPBPRecursiveLocalPDEAffine local = upbp_recursive_local_pde_affine(configuration, source.cls, source.delta, source.density_connectible, departure.weights,
      source_ray_ratio, departure.last_sin_theta, source.source);
    result.d_pde_reverse_coefficient = departure.d_bpt_a * local.reverse_pdf_inverse_coefficient;
    result.d_pde_constant = departure.d_bpt_a * local.constant + departure.d_pde_b;
    result.d_surface_constant = departure.d_bpt_a * local.surface_coefficient + departure.d_surface_b;
  } else {
    result.d_pde_constant = departure.weights.d_pde_base;
    result.d_surface_constant = departure.weights.d_surface;
  }
  result.valid = (interval.medium_index == beam.medium_index) && (result.source_event_density > 0.0) && std::isfinite(result.source_event_density) &&
                 std::isfinite(source_ray_ratio) && std::isfinite(result.d_shared) && std::isfinite(result.d_pde_reverse_coefficient) &&
                 (std::isfinite(result.d_pde_constant) && std::isfinite(result.d_surface_constant));
  if (result.valid == false) {
    result = {};
  }
  return result.valid;
}

ETX_UPBP_FORCE_INLINE bool upbp_prepared_medium_interval_prefix(const UPBPPreparedBeam& prepared, const float prefix_distance, UPBPBeamTransportPrefix& result) {
  if ((prepared.tracking_events == nullptr) || (prepared.tracking_event_count == 0u) || (prefix_distance <= 0.0f) || (prefix_distance >= prepared.interval_distance)) {
    return false;
  }
  const UPBPMediumTrackingEventRecord* prefix = prepared.tracking_events;
  const UPBPMediumTrackingEventRecord* const end = prepared.tracking_events + prepared.tracking_event_count;
  constexpr uint32_t linear_search_limit = 8u;
  if (prepared.tracking_event_count <= linear_search_limit) {
    while ((prefix != end) && (prefix->end_distance < prefix_distance)) {
      ++prefix;
    }
  } else {
    prefix = std::lower_bound(prefix, end, prefix_distance, [](const UPBPMediumTrackingEventRecord& record, const float value) {
      return record.end_distance < value;
    });
  }
  if (prefix == end) {
    return false;
  }
  result.weight = prefix->weight_before;
  result.log_transport_pdf_forward = prefix->log_transport_pdf_forward_before;
  result.log_transport_pdf_reverse = prefix->log_transport_pdf_reverse_before;
  const float remaining_distance = prefix_distance - prefix->distance_before;
  const double log_transmittance = -static_cast<double>(prefix->majorant) * static_cast<double>(remaining_distance);
  if (std::isfinite(log_transmittance) == false) {
    return false;
  }
  result.log_transport_pdf_forward += log_transmittance;
  result.log_transport_pdf_reverse += log_transmittance;
  result.distance = prefix_distance;
  return true;
}

ETX_UPBP_FORCE_INLINE bool upbp_complete_prepared_partial_medium_arrival(const UPBPPreparedBeam& prepared, const double log_transport_pdf_forward,
  const double log_transport_pdf_reverse, const float transport_distance, const double query_real_event_density, UPBPPointMergeMISInput::Weights& weights) {
  if ((transport_distance <= 0.0f) || (query_real_event_density <= 0.0)) {
    return false;
  }
  const double forward_transport_pdf = std::exp(log_transport_pdf_forward);
  const double reverse_transport_pdf = log_transport_pdf_reverse == log_transport_pdf_forward ? forward_transport_pdf : std::exp(log_transport_pdf_reverse);
  const double forward_pdf = forward_transport_pdf * query_real_event_density;
  const double reverse_pdf = reverse_transport_pdf * prepared.source_event_density;
  if ((forward_pdf <= 0.0) || (reverse_pdf <= 0.0) || (std::isfinite(forward_pdf) == false) || (std::isfinite(reverse_pdf) == false)) {
    return false;
  }
  const double reverse_pdf_inverse = 1.0 / reverse_pdf;
  weights.d_pde_base = (prepared.d_pde_reverse_coefficient * reverse_pdf_inverse + prepared.d_pde_constant) / forward_pdf;
  weights.d_surface = prepared.d_surface_constant / forward_pdf;
  weights.d_shared = prepared.d_shared / forward_pdf;
  if (prepared.scale_d_shared_by_distance) {
    weights.d_shared *= static_cast<double>(transport_distance) * transport_distance;
  }
  weights.ray_sample_forward_pdf_inverse = 1.0 / forward_pdf;
  weights.ray_sample_reverse_pdf_inverse = reverse_pdf_inverse;
  weights.ray_sample_forward_ratio = 1.0 / query_real_event_density;
  weights.previous_delta = prepared.previous_delta;
  return std::isfinite(weights.d_shared) && (std::isfinite(weights.d_pde_base) && std::isfinite(weights.d_surface));
}

ETX_UPBP_FORCE_INLINE bool upbp_partial_prepared_beam_vertex(const Medium& medium, const UPBPPreparedMedium& prepared_medium, const UPBPPreparedBeam& prepared,
  const UPBPBeamReference& beam, const float distance, UPBPPartialPreparedBeamVertex& result) {
  if ((prepared.valid == false) || (prepared_medium.valid == false)) {
    return false;
  }
  UPBPBeamTransportPrefix partial_interval;
  if (upbp_prepared_medium_interval_prefix(prepared, distance, partial_interval) == false) {
    return false;
  }
  if (medium.cls == Medium::Homogeneous) {
    result.medium_density = 1.0f;
  } else {
    const float3 position = beam.origin + beam.direction * distance;
    result.medium_density = upbp_prepared_medium_density(medium, prepared_medium, position);
  }
  if ((result.medium_density < 0.0f) || (result.medium_density > medium_tracking_density_majorant(medium)) || (std::isfinite(result.medium_density) == false)) {
    return false;
  }
  const double real_event_density = prepared_medium.real_event_density_scale * result.medium_density;
  if (real_event_density <= 0.0) {
    return false;
  }
  if (upbp_complete_prepared_partial_medium_arrival(prepared, prepared.transport_at_origin.log_transport_pdf_forward + partial_interval.log_transport_pdf_forward,
        prepared.transport_at_origin.log_transport_pdf_reverse + partial_interval.log_transport_pdf_reverse, prepared.transport_at_origin.distance + partial_interval.distance,
        real_event_density, result.weights) == false) {
    return false;
  }
  result.throughput = prepared.source_throughput * (prepared.transport_at_origin.weight * partial_interval.weight);
  return result.throughput.is_zero() == false;
}

inline double upbp_medium_phase_sine(const float3& incoming_direction, const float3& outgoing_direction) {
  const double cosine = static_cast<double>(dot(incoming_direction, outgoing_direction));
  return std::sqrt(fmax(0.0, 1.0 - cosine * cosine));
}

ETX_UPBP_FORCE_INLINE bool upbp_evaluate_prepared_pb2d(const UPBPPathVertexRecord& light_vertex, const UPBPRecursiveVertexWeights& light_weights,
  const SpectralResponse& light_throughput, const Medium& medium, const UPBPPreparedMedium& prepared_medium, const UPBPPreparedBeam& prepared_camera_beam,
  const UPBPBeamReference& camera_beam, const UPBPPointBeamIntersection& intersection, const UPBPDensityMISConfiguration& configuration, const UPBPKernel kernel,
  const double radius, const uint64_t light_subpath_count, const uint64_t bpt_sample_count, UPBPBeamContribution& result) {
  result.applicable = false;
  if ((prepared_medium.valid == false) || (light_subpath_count == 0u)) {
    return false;
  }
  UPBPPartialPreparedBeamVertex partial_camera;
  if (upbp_partial_prepared_beam_vertex(medium, prepared_medium, prepared_camera_beam, camera_beam, intersection.beam_distance, partial_camera) == false) {
    return true;
  }
  const SpectralResponse scattering = prepared_medium.scattering * partial_camera.medium_density;
  const float3 outgoing_direction = -light_vertex.intersection.w_i;
  const float cosine = dot(camera_beam.direction, outgoing_direction);
  const double phase = upbp_prepared_phase_function_from_cosine(prepared_medium, cosine);
  const double sin_theta = std::sqrt(fmax(0.0, 1.0 - static_cast<double>(cosine) * cosine));
  const double kernel_value = upbp_kernel_value(kernel, 2u, radius, intersection.distance_squared);
  const double mis_weight = upbp_point_merge_mis_weight({
    UPBPTechnique::PB2D,
    UPBPVertexClass::Medium,
    upbp_point_merge_weights(light_weights),
    partial_camera.weights,
    configuration,
    phase,
    phase,
    sin_theta,
    bpt_sample_count,
  });
  if ((kernel_value <= 0.0) || (mis_weight <= 0.0) || scattering.is_zero()) {
    return true;
  }
  const double estimator_scale = kernel_value / static_cast<double>(light_subpath_count);
  result.contribution = light_throughput * partial_camera.throughput * scattering * static_cast<float>(phase * estimator_scale * mis_weight);
  result.applicable = true;
  return true;
}

ETX_UPBP_FORCE_INLINE bool upbp_evaluate_prepared_bp2d(const Medium& medium, const UPBPPreparedMedium& prepared_medium, const UPBPPreparedBeam& prepared_light_beam,
  const UPBPBeamReference& light_beam, const UPBPPointBeamIntersection& intersection, const float3& camera_incoming_direction,
  const UPBPPointMergeMISInput::Weights& camera_weights, const SpectralResponse& camera_throughput, const SpectralResponse& scattering,
  const UPBPDensityMISConfiguration& configuration, const UPBPKernel kernel, const double radius, const uint64_t light_subpath_count, const uint64_t bpt_sample_count,
  UPBPBeamContribution& result) {
  result.applicable = false;
  if ((prepared_medium.valid == false) || (light_subpath_count == 0u)) {
    return false;
  }
  UPBPPartialPreparedBeamVertex partial_light;
  if (upbp_partial_prepared_beam_vertex(medium, prepared_medium, prepared_light_beam, light_beam, intersection.beam_distance, partial_light) == false) {
    return true;
  }
  const float cosine = dot(camera_incoming_direction, -light_beam.direction);
  const double phase = upbp_prepared_phase_function_from_cosine(prepared_medium, cosine);
  const double sin_theta = std::sqrt(fmax(0.0, 1.0 - static_cast<double>(cosine) * cosine));
  const double kernel_value = upbp_kernel_value(kernel, 2u, radius, intersection.distance_squared);
  const double mis_weight = upbp_point_merge_mis_weight({
    UPBPTechnique::BP2D,
    UPBPVertexClass::Medium,
    partial_light.weights,
    camera_weights,
    configuration,
    phase,
    phase,
    sin_theta,
    bpt_sample_count,
  });
  if ((kernel_value <= 0.0) || (mis_weight <= 0.0) || scattering.is_zero()) {
    return true;
  }
  const double estimator_scale = kernel_value / static_cast<double>(light_subpath_count);
  result.contribution = partial_light.throughput * camera_throughput * scattering * static_cast<float>(phase * estimator_scale * mis_weight);
  result.applicable = true;
  return true;
}

ETX_UPBP_FORCE_INLINE bool upbp_evaluate_bb1d(const Medium& medium, const UPBPPreparedMedium& prepared_medium, const UPBPPreparedBeam& prepared_light_beam,
  const UPBPBeamReference& light_beam, const UPBPPreparedBeam& prepared_camera_beam, const UPBPBeamReference& camera_beam, const UPBPBeamBeamIntersection& intersection,
  const UPBPDensityMISConfiguration& configuration, const UPBPPreparedBB1D& prepared_bb1d, const uint64_t bpt_sample_count, UPBPBeamContribution& result) {
  result.applicable = false;
  if (prepared_bb1d.valid == false) {
    return false;
  }
  UPBPPartialPreparedBeamVertex partial_light;
  UPBPPartialPreparedBeamVertex partial_camera;
  if ((upbp_partial_prepared_beam_vertex(medium, prepared_medium, prepared_light_beam, light_beam, intersection.first_distance, partial_light) == false) ||
      (upbp_partial_prepared_beam_vertex(medium, prepared_medium, prepared_camera_beam, camera_beam, intersection.second_distance, partial_camera) == false)) {
    return true;
  }
  const SpectralResponse scattering = prepared_medium.scattering * partial_camera.medium_density;
  const double phase = upbp_prepared_phase_function_from_cosine(prepared_medium, -intersection.direction_dot);
  const double kernel_value = upbp_evaluate_prepared_bb1d_kernel(prepared_bb1d, intersection.distance_squared, intersection.sin_theta);
  const double mis_weight = upbp_point_merge_mis_weight({
    UPBPTechnique::BB1D,
    UPBPVertexClass::Medium,
    partial_light.weights,
    partial_camera.weights,
    configuration,
    phase,
    phase,
    intersection.sin_theta,
    bpt_sample_count,
  });
  if ((kernel_value <= 0.0) || (mis_weight <= 0.0) || scattering.is_zero()) {
    return true;
  }
  const double estimator_scale = kernel_value * prepared_bb1d.estimator_normalization;
  result.contribution = partial_light.throughput * partial_camera.throughput * scattering * static_cast<float>(phase * estimator_scale * mis_weight);
  result.applicable = true;
  return true;
}

#undef ETX_UPBP_FORCE_INLINE
#undef ETX_UPBP_HAS_SSE2

}  // namespace etx
