#pragma once

#include <etx/rt/integrators/upbp_core.hxx>
#include <etx/render/host/tasks.hxx>

#include <cmath>
#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace etx {

struct UPBPBeamReference;
struct UPBPPointBeamIntersection;

struct UPBPSpatialQueryState {
  void begin(const uint32_t item_count) {
    if (_visited.size() != item_count) {
      _visited.assign(item_count, 0u);
      _generation = 1u;
      return;
    }
    if (_generation == std::numeric_limits<uint32_t>::max()) {
      std::fill(_visited.begin(), _visited.end(), 0u);
      _generation = 1u;
    } else {
      ++_generation;
    }
  }

  bool mark(const uint32_t item_index) {
    if (item_index >= _visited.size()) {
      return false;
    }
    if (_visited[item_index] == _generation) {
      return false;
    }
    _visited[item_index] = _generation;
    return true;
  }

 private:
  std::vector<uint32_t> _visited = {};
  uint32_t _generation = 0u;
};

struct UPBPPointReference {
  float3 position = {};
  uint32_t path_index = 0u;
  uint32_t vertex_index = 0u;
};

struct UPBPPointIndex {
  struct Cell {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;

    bool operator==(const Cell&) const = default;
  };

  struct Entry {
    Cell cell = {};
    UPBPPointReference point = {};
  };

  struct CellRange {
    Cell cell = {};
    uint32_t first = 0u;
    uint32_t count = 0u;
  };

  bool build(const UPBPPointReference* points, const uint32_t point_count, const float cell_size) {
    clear();
    if ((points == nullptr) || (point_count == 0u) || (cell_size <= 0.0f) || (std::isfinite(cell_size) == false)) {
      return false;
    }

    _cell_size = cell_size;
    _inverse_cell_size = 1.0f / cell_size;
    _entries.resize(point_count);
    for (uint32_t index = 0u; index < point_count; ++index) {
      Cell point_cell = {};
      if (make_cell(points[index].position, point_cell) == false) {
        clear();
        return false;
      }
      _entries[index] = {point_cell, points[index]};
    }
    std::sort(_entries.begin(), _entries.end(), [](const Entry& first, const Entry& second) {
      if (cell_less(first.cell, second.cell)) {
        return true;
      }
      if (cell_less(second.cell, first.cell)) {
        return false;
      }
      if (first.point.path_index != second.point.path_index) {
        return first.point.path_index < second.point.path_index;
      }
      return first.point.vertex_index < second.point.vertex_index;
    });
    if (build_cell_ranges() == false) {
      clear();
      return false;
    }
    return true;
  }

  static bool beam_query_cell_size(const UPBPPointReference* points, const uint32_t point_count, const float radius, float& result) {
    if ((points == nullptr) || (point_count == 0u) || (radius <= 0.0f) || (std::isfinite(radius) == false)) {
      return false;
    }
    float3 minimum = {kMaxFloat, kMaxFloat, kMaxFloat};
    float3 maximum = {-kMaxFloat, -kMaxFloat, -kMaxFloat};
    for (uint32_t point_index = 0u; point_index < point_count; ++point_index) {
      const float3& position = points[point_index].position;
      if ((std::isfinite(position.x) == false) || (std::isfinite(position.y) == false) || (std::isfinite(position.z) == false)) {
        return false;
      }
      minimum = min(minimum, position);
      maximum = max(maximum, position);
    }
    const float3 extent = maximum - minimum;
    const float maximum_extent = max(extent.x, max(extent.y, extent.z));
    if ((maximum_extent < 0.0f) || (std::isfinite(maximum_extent) == false)) {
      return false;
    }
    constexpr double target_points_per_cell = 4.0;
    const double resolution = cbrt(max(1.0, static_cast<double>(point_count) / target_points_per_cell));
    const float population_cell_size = static_cast<float>(static_cast<double>(maximum_extent) / resolution);
    result = max(radius, std::nextafter(population_cell_size, std::numeric_limits<float>::infinity()));
    return (result > 0.0f) && std::isfinite(result);
  }

  void clear() {
    _entries.clear();
    _cell_ranges.clear();
    _cell_slots.clear();
    _cell_size = 0.0f;
    _inverse_cell_size = 0.0f;
  }

  template <typename Visitor>
  bool query(const float3& position, const float radius, Visitor&& visitor) const {
    if ((_cell_size <= 0.0f) || (radius <= 0.0f) || (radius > _cell_size) || (std::isfinite(radius) == false)) {
      return false;
    }

    Cell center = {};
    if (make_cell(position, center) == false) {
      return false;
    }
    const float radius_squared = radius * radius;
    for (int32_t z = center.z - 1; z <= center.z + 1; ++z) {
      for (int32_t y = center.y - 1; y <= center.y + 1; ++y) {
        for (int32_t x = center.x - 1; x <= center.x + 1; ++x) {
          visit_cell({x, y, z}, [&position, radius_squared, &visitor](const UPBPPointReference& point, const uint32_t) {
            const float3 delta = point.position - position;
            const float distance_squared = dot(delta, delta);
            if (distance_squared < radius_squared) {
              visitor(point, distance_squared);
            }
          });
        }
      }
    }
    return true;
  }

  template <typename Visitor>
  bool query_beam(const UPBPBeamReference& beam, float radius, UPBPSpatialQueryState& query_state, Visitor&& visitor) const;

  uint32_t size() const {
    return static_cast<uint32_t>(_entries.size());
  }

  uint64_t storage_bytes() const {
    return static_cast<uint64_t>(_entries.capacity()) * sizeof(Entry) + static_cast<uint64_t>(_cell_ranges.capacity()) * sizeof(CellRange) +
           static_cast<uint64_t>(_cell_slots.capacity()) * sizeof(uint32_t);
  }

  bool projected_storage_bytes(const uint32_t point_count, uint64_t& result) const {
    const uint64_t capacity = max(static_cast<uint64_t>(_entries.capacity()), static_cast<uint64_t>(point_count));
    if (capacity > std::numeric_limits<uint64_t>::max() / sizeof(Entry)) {
      return false;
    }
    result = capacity * sizeof(Entry);
    uint64_t slot_count = 0u;
    if (projected_cell_slot_count(point_count, slot_count) == false) {
      return false;
    }
    const uint64_t cell_range_capacity = max(static_cast<uint64_t>(_cell_ranges.capacity()), static_cast<uint64_t>(point_count));
    const uint64_t cell_slot_capacity = max(static_cast<uint64_t>(_cell_slots.capacity()), slot_count);
    if ((cell_range_capacity > std::numeric_limits<uint64_t>::max() / sizeof(CellRange)) || (cell_slot_capacity > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t))) {
      return false;
    }
    const uint64_t cell_range_bytes = cell_range_capacity * sizeof(CellRange);
    const uint64_t cell_slot_bytes = cell_slot_capacity * sizeof(uint32_t);
    if ((result > std::numeric_limits<uint64_t>::max() - cell_range_bytes) || ((result + cell_range_bytes) > std::numeric_limits<uint64_t>::max() - cell_slot_bytes)) {
      return false;
    }
    result += cell_range_bytes + cell_slot_bytes;
    return true;
  }

 private:
  static bool cell_less(const Cell& first, const Cell& second) {
    if (first.z != second.z) {
      return first.z < second.z;
    }
    if (first.y != second.y) {
      return first.y < second.y;
    }
    return first.x < second.x;
  }

  static uint64_t cell_hash(const Cell& cell) {
    uint64_t value = static_cast<uint32_t>(cell.x);
    value = (value ^ (static_cast<uint64_t>(static_cast<uint32_t>(cell.y)) + 0x9e3779b97f4a7c15ull + (value << 6u) + (value >> 2u)));
    value = (value ^ (static_cast<uint64_t>(static_cast<uint32_t>(cell.z)) + 0x9e3779b97f4a7c15ull + (value << 6u) + (value >> 2u)));
    value ^= value >> 30u;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27u;
    value *= 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
  }

  static bool projected_cell_slot_count(const uint64_t cell_count, uint64_t& result) {
    if (cell_count == 0u) {
      result = 0u;
      return true;
    }
    if (cell_count > std::numeric_limits<uint64_t>::max() / 2u) {
      return false;
    }
    const uint64_t required = cell_count * 2u;
    result = 1u;
    while (result < required) {
      if (result > std::numeric_limits<uint64_t>::max() / 2u) {
        return false;
      }
      result *= 2u;
    }
    return true;
  }

  bool build_cell_ranges() {
    if (_entries.empty()) {
      return false;
    }
    _cell_ranges.reserve(_entries.size());
    uint32_t first = 0u;
    while (first < _entries.size()) {
      uint32_t end = first + 1u;
      while ((end < _entries.size()) && (_entries[end].cell == _entries[first].cell)) {
        ++end;
      }
      _cell_ranges.emplace_back(CellRange{_entries[first].cell, first, end - first});
      first = end;
    }
    if (_cell_ranges.size() >= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      return false;
    }
    uint64_t slot_count = 0u;
    if ((projected_cell_slot_count(_cell_ranges.size(), slot_count) == false) || (slot_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))) {
      return false;
    }
    _cell_slots.assign(static_cast<size_t>(slot_count), 0u);
    const size_t mask = _cell_slots.size() - 1u;
    for (uint32_t cell_index = 0u; cell_index < _cell_ranges.size(); ++cell_index) {
      size_t slot = static_cast<size_t>(cell_hash(_cell_ranges[cell_index].cell)) & mask;
      while (_cell_slots[slot] != 0u) {
        slot = (slot + 1u) & mask;
      }
      _cell_slots[slot] = cell_index + 1u;
    }
    return true;
  }

  template <typename Visitor>
  void visit_cell(const Cell& cell, Visitor&& visitor) const {
    if (_cell_slots.empty()) {
      return;
    }
    const size_t mask = _cell_slots.size() - 1u;
    size_t slot = static_cast<size_t>(cell_hash(cell)) & mask;
    for (;;) {
      const uint32_t encoded_index = _cell_slots[slot];
      if (encoded_index == 0u) {
        return;
      }
      const CellRange& range = _cell_ranges[encoded_index - 1u];
      if (range.cell == cell) {
        for (uint32_t offset = 0u; offset < range.count; ++offset) {
          const uint32_t entry_index = range.first + offset;
          visitor(_entries[entry_index].point, entry_index);
        }
        return;
      }
      slot = (slot + 1u) & mask;
    }
  }

  bool make_cell(const float3& position, Cell& result) const {
    constexpr double minimum_cell = static_cast<double>(std::numeric_limits<int32_t>::min()) + 1.0;
    constexpr double maximum_cell = static_cast<double>(std::numeric_limits<int32_t>::max()) - 1.0;
    const double x = floor(static_cast<double>(position.x) * static_cast<double>(_inverse_cell_size));
    const double y = floor(static_cast<double>(position.y) * static_cast<double>(_inverse_cell_size));
    const double z = floor(static_cast<double>(position.z) * static_cast<double>(_inverse_cell_size));
    if ((std::isfinite(x) == false) || (std::isfinite(y) == false) || (std::isfinite(z) == false) || (x < minimum_cell) || (x > maximum_cell) || (y < minimum_cell) ||
        (y > maximum_cell) || (z < minimum_cell) || (z > maximum_cell)) {
      return false;
    }

    result = {static_cast<int32_t>(x), static_cast<int32_t>(y), static_cast<int32_t>(z)};
    return true;
  }

  std::vector<Entry> _entries = {};
  std::vector<CellRange> _cell_ranges = {};
  std::vector<uint32_t> _cell_slots = {};
  float _cell_size = 0.0f;
  float _inverse_cell_size = 0.0f;
};

struct UPBPBeamReference {
  float3 origin = {};
  float3 direction = {};
  float length = 0.0f;
  uint32_t medium_index = kInvalidIndex;
  uint32_t path_index = 0u;
  uint32_t segment_index = 0u;
  uint32_t source_vertex_index = 0u;
  uint32_t transport_segment_index = 0u;
  uint32_t transport_interval_index = 0u;
  SpectralResponse throughput_at_origin = {};
};

inline bool upbp_collect_medium_beams(const UPBPPathRecord& path, const uint32_t path_index, std::vector<UPBPBeamReference>& result) {
  if (path.valid() == false) {
    return false;
  }

  result.clear();
  uint32_t interval_index = 0u;
  auto append_segment = [&result, &interval_index, &path, path_index](const UPBPTransportSegmentRecord& segment, const uint32_t source_vertex_index,
                          const uint32_t transport_segment_index) {
    SpectralResponse throughput = path.vertices[source_vertex_index].outgoing_throughput;
    for (uint32_t transport_interval_index = 0u; transport_interval_index < segment.intervals.size(); ++transport_interval_index) {
      const UPBPSegmentRecord& interval = segment.intervals[transport_interval_index];
      if (interval.medium_index != kInvalidIndex) {
        const float3 delta = interval.end_position - interval.start_position;
        const float distance_squared = dot(delta, delta);
        if ((interval.distance <= 0.0f) || (std::isfinite(interval.distance) == false) || (distance_squared < 0.0f) || (std::isfinite(distance_squared) == false)) {
          return false;
        }
        if (distance_squared > 0.0f) {
          const float beam_length = sqrtf(distance_squared);
          result.emplace_back(UPBPBeamReference{
            interval.start_position,
            delta / beam_length,
            beam_length,
            interval.medium_index,
            path_index,
            interval_index,
            source_vertex_index,
            transport_segment_index,
            transport_interval_index,
            throughput,
          });
        }
      }
      throughput *= interval.weight;
      ++interval_index;
    }
    return true;
  };

  for (uint32_t segment_index = 0u; segment_index < path.segments.size(); ++segment_index) {
    if (append_segment(path.segments[segment_index], segment_index, segment_index) == false) {
      result.clear();
      return false;
    }
  }
  if (path.has_terminal_segment &&
      (append_segment(path.terminal_segment, static_cast<uint32_t>(path.vertices.size() - 1u), static_cast<uint32_t>(path.segments.size())) == false)) {
    result.clear();
    return false;
  }
  return true;
}

inline const UPBPSegmentRecord* upbp_beam_interval(const UPBPPathRecord& path, const UPBPBeamReference& beam) {
  const UPBPTransportSegmentRecord* segment = nullptr;
  if (beam.transport_segment_index < path.segments.size()) {
    segment = &path.segments[beam.transport_segment_index];
  } else if (path.has_terminal_segment && (beam.transport_segment_index == path.segments.size())) {
    segment = &path.terminal_segment;
  }
  if ((segment == nullptr) || (beam.transport_interval_index >= segment->intervals.size())) {
    return nullptr;
  }
  return &segment->intervals[beam.transport_interval_index];
}

struct UPBPBeamTransportPrefix {
  SpectralResponse weight = {};
  double log_transport_pdf_forward = 0.0;
  double log_transport_pdf_reverse = 0.0;
  float distance = 0.0f;
};

inline bool upbp_medium_interval_prefix(const UPBPSegmentRecord& interval, const float prefix_distance, UPBPBeamTransportPrefix& result) {
  if ((interval.valid() == false) || (interval.complete == false) || (interval.medium_index == kInvalidIndex) || (prefix_distance <= 0.0f) ||
      (prefix_distance >= interval.distance)) {
    return false;
  }

  result = {};
  auto prefix = interval.events.begin();
  constexpr size_t linear_search_limit = 8u;
  if (interval.events.size() <= linear_search_limit) {
    while ((prefix != interval.events.end()) && (prefix->end_distance < prefix_distance)) {
      ++prefix;
    }
  } else {
    prefix = std::lower_bound(interval.events.begin(), interval.events.end(), prefix_distance, [](const UPBPMediumTrackingEventRecord& record, const float value) {
      return record.end_distance < value;
    });
  }
  if (prefix == interval.events.end()) {
    return false;
  }
  if ((prefix->majorant <= 0.0f) || (std::isfinite(prefix->majorant) == false)) {
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

inline bool upbp_beam_transport_prefix(const UPBPPathRecord& path, const UPBPBeamReference& beam, const float interval_distance, UPBPBeamTransportPrefix& result) {
  const UPBPTransportSegmentRecord* segment = nullptr;
  if (beam.transport_segment_index < path.segments.size()) {
    segment = &path.segments[beam.transport_segment_index];
  } else if (path.has_terminal_segment && (beam.transport_segment_index == path.segments.size())) {
    segment = &path.terminal_segment;
  }
  if ((segment == nullptr) || (beam.transport_interval_index >= segment->intervals.size())) {
    return false;
  }

  result = {};
  result.weight = SpectralResponse{segment->weight.as_query(), 1.0f};
  for (uint32_t interval_index = 0u; interval_index < beam.transport_interval_index; ++interval_index) {
    const UPBPSegmentRecord& interval = segment->intervals[interval_index];
    const bool terminal_medium_event = (interval.terminal_event == MediumTrackingEventType::Scatter) || (interval.terminal_event == MediumTrackingEventType::Absorb);
    if ((interval.valid() == false) || (interval.complete == false) || terminal_medium_event) {
      return false;
    }
    result.weight *= interval.weight;
    result.log_transport_pdf_forward += interval.log_transport_pdf_forward;
    result.log_transport_pdf_reverse += interval.log_transport_pdf_reverse;
    result.distance += interval.distance;
  }

  UPBPBeamTransportPrefix partial_interval = {};
  if (upbp_medium_interval_prefix(segment->intervals[beam.transport_interval_index], interval_distance, partial_interval) == false) {
    return false;
  }
  result.weight *= partial_interval.weight;
  result.log_transport_pdf_forward += partial_interval.log_transport_pdf_forward;
  result.log_transport_pdf_reverse += partial_interval.log_transport_pdf_reverse;
  result.distance += partial_interval.distance;
  return true;
}

struct UPBPPointBeamIntersection {
  float beam_distance = 0.0f;
  float distance_squared = 0.0f;
};

inline bool upbp_intersect_point_beam(const float3& point, const UPBPBeamReference& beam, const float radius, UPBPPointBeamIntersection& result) {
  if ((beam.length <= 0.0f) || (radius <= 0.0f) || (fabsf(dot(beam.direction, beam.direction) - 1.0f) > 1.0e-4f)) {
    return false;
  }

  result.beam_distance = dot(point - beam.origin, beam.direction);
  if ((result.beam_distance < 0.0f) || (result.beam_distance >= beam.length)) {
    return false;
  }

  const float3 delta = point - (beam.origin + beam.direction * result.beam_distance);
  result.distance_squared = dot(delta, delta);
  return result.distance_squared < radius * radius;
}

template <typename Visitor>
bool UPBPPointIndex::query_beam(const UPBPBeamReference& beam, const float radius, UPBPSpatialQueryState& query_state, Visitor&& visitor) const {
  if ((_cell_size <= 0.0f) || _cell_slots.empty() || (radius <= 0.0f) || (radius > _cell_size) || (std::isfinite(radius) == false) || (beam.length <= 0.0f) ||
      (std::isfinite(beam.length) == false) || (fabsf(dot(beam.direction, beam.direction) - 1.0f) > 1.0e-4f)) {
    return false;
  }

  const float3 end = beam.origin + beam.direction * beam.length;
  Cell cell = {};
  Cell end_cell = {};
  if ((make_cell(beam.origin, cell) == false) || (make_cell(end, end_cell) == false)) {
    return false;
  }
  query_state.begin(static_cast<uint32_t>(_entries.size()));

  std::array<int32_t, 3u> step = {};
  std::array<double, 3u> next_distance = {};
  std::array<double, 3u> distance_step = {};
  const double infinity = std::numeric_limits<double>::infinity();
  const std::array<int32_t, 3u> initial_cell = {cell.x, cell.y, cell.z};
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    const double direction = static_cast<double>(axis == 0u ? beam.direction.x : (axis == 1u ? beam.direction.y : beam.direction.z));
    const double origin = static_cast<double>(axis == 0u ? beam.origin.x : (axis == 1u ? beam.origin.y : beam.origin.z));
    if (direction > 0.0) {
      step[axis] = 1;
      const double boundary = static_cast<double>(initial_cell[axis] + 1) * static_cast<double>(_cell_size);
      next_distance[axis] = (boundary - origin) / direction;
      distance_step[axis] = static_cast<double>(_cell_size) / direction;
    } else if (direction < 0.0) {
      step[axis] = -1;
      const double boundary = static_cast<double>(initial_cell[axis]) * static_cast<double>(_cell_size);
      next_distance[axis] = (boundary - origin) / direction;
      distance_step[axis] = -static_cast<double>(_cell_size) / direction;
    } else {
      next_distance[axis] = infinity;
      distance_step[axis] = infinity;
    }
    while (next_distance[axis] < 0.0) {
      next_distance[axis] += distance_step[axis];
    }
  }

  // With cells at least as wide as the query radius, the neighboring cells conservatively cover the beam support.
  for (;;) {
    for (int32_t z = cell.z - 1; z <= cell.z + 1; ++z) {
      for (int32_t y = cell.y - 1; y <= cell.y + 1; ++y) {
        for (int32_t x = cell.x - 1; x <= cell.x + 1; ++x) {
          visit_cell({x, y, z}, [&beam, radius, &query_state, &visitor](const UPBPPointReference& point, const uint32_t entry_index) {
            if (query_state.mark(entry_index) == false) {
              return;
            }
            UPBPPointBeamIntersection intersection = {};
            if (upbp_intersect_point_beam(point.position, beam, radius, intersection)) {
              visitor(point, intersection);
            }
          });
        }
      }
    }

    const double next = min(next_distance[0u], min(next_distance[1u], next_distance[2u]));
    if (next >= static_cast<double>(beam.length)) {
      break;
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
      if (next_distance[axis] != next) {
        continue;
      }
      if (axis == 0u) {
        cell.x += step[axis];
      } else if (axis == 1u) {
        cell.y += step[axis];
      } else {
        cell.z += step[axis];
      }
      next_distance[axis] += distance_step[axis];
    }
  }
  return true;
}

struct UPBPBeamBeamIntersection {
  float first_distance = 0.0f;
  float second_distance = 0.0f;
  float distance_squared = 0.0f;
  float sin_theta = 0.0f;
  float direction_dot = 0.0f;
};

inline bool upbp_intersect_valid_beams_squared(const UPBPBeamReference& first, const UPBPBeamReference& second, const float radius_squared, UPBPBeamBeamIntersection& result) {
  const float3 direction_cross = cross(first.direction, second.direction);
  const float sin_theta_squared = dot(direction_cross, direction_cross);
  constexpr float degeneracy_threshold = 16.0f * std::numeric_limits<float>::epsilon();
  if (sin_theta_squared <= degeneracy_threshold) {
    return false;
  }

  const float3 origin_delta = first.origin - second.origin;
  const float scaled_distance = dot(origin_delta, direction_cross);
  const float scaled_distance_squared = scaled_distance * scaled_distance;
  if (scaled_distance_squared >= radius_squared * sin_theta_squared) {
    return false;
  }

  const float direction_dot = dot(first.direction, second.direction);
  const float first_projection = dot(first.direction, origin_delta);
  const float second_projection = dot(second.direction, origin_delta);
  result.first_distance = (direction_dot * second_projection - first_projection) / sin_theta_squared;
  result.second_distance = (second_projection - direction_dot * first_projection) / sin_theta_squared;
  if ((result.first_distance < 0.0f) || (result.first_distance >= first.length) || (result.second_distance < 0.0f) || (result.second_distance >= second.length)) {
    return false;
  }

  result.distance_squared = scaled_distance_squared / sin_theta_squared;
  result.sin_theta = sqrtf(sin_theta_squared);
  result.direction_dot = direction_dot;
  return true;
}

inline bool upbp_intersect_beams(const UPBPBeamReference& first, const UPBPBeamReference& second, const float radius, UPBPBeamBeamIntersection& result) {
  if ((first.length <= 0.0f) || (second.length <= 0.0f) || (radius <= 0.0f) || (fabsf(dot(first.direction, first.direction) - 1.0f) > 1.0e-4f) ||
      (fabsf(dot(second.direction, second.direction) - 1.0f) > 1.0e-4f)) {
    return false;
  }
  return upbp_intersect_valid_beams_squared(first, second, radius * radius, result);
}

struct UPBPBeamGrid {
  struct Bounds {
    float3 minimum = {kMaxFloat, kMaxFloat, kMaxFloat};
    float3 maximum = {-kMaxFloat, -kMaxFloat, -kMaxFloat};
  };

  struct Description {
    Bounds bounds = {};
    float cell_size = 0.0f;
    float inverse_cell_size = 0.0f;
    std::array<uint32_t, 3u> resolution = {};
  };

  static constexpr uint32_t maximum_resolution = 32u;

  bool build(const UPBPBeamReference* beams, const uint32_t beam_count, const float radius) {
    return build_with_executor(beams, beam_count, radius, 1u, [](const uint32_t range, auto&& function) {
      function(0u, range, 0u);
    });
  }

  bool build(const UPBPBeamReference* beams, const uint32_t beam_count, const float radius, TaskScheduler& scheduler) {
    return build_with_executor(beams, beam_count, radius, min(beam_count, max(1u, scheduler.max_thread_count())), [&scheduler](const uint32_t range, auto&& function) {
      scheduler.execute(range, std::forward<decltype(function)>(function));
    });
  }

 private:
  template <typename Execute>
  bool build_with_executor(const UPBPBeamReference* beams, const uint32_t beam_count, const float radius, const uint32_t shard_count, Execute&& execute) {
    clear();
    Description description = {};
    if ((describe(beams, beam_count, radius, description) == false) || (shard_count == 0u) || (shard_count > beam_count)) {
      return false;
    }

    const uint32_t cell_count = grid_cell_count(description.resolution);
    if (static_cast<uint64_t>(shard_count) * static_cast<uint64_t>(cell_count) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return false;
    }
    const size_t shard_cell_count = static_cast<size_t>(shard_count) * static_cast<size_t>(cell_count);
    std::vector<uint32_t> shard_cell_counts(shard_cell_count, 0u);
    std::vector<uint32_t> shard_cell_marks(shard_cell_count, kInvalidIndex);
    std::vector<uint8_t> shard_validity(shard_count, 1u);

    execute(shard_count, [beams, beam_count, radius, description, cell_count, shard_count, &shard_cell_counts, &shard_cell_marks, &shard_validity](const uint32_t begin,
                           const uint32_t end, const uint32_t) {
      for (uint32_t shard_index = begin; shard_index < end; ++shard_index) {
        uint32_t* counts = shard_cell_counts.data() + static_cast<size_t>(shard_index) * cell_count;
        uint32_t* marks = shard_cell_marks.data() + static_cast<size_t>(shard_index) * cell_count;
        const uint32_t first_beam = static_cast<uint32_t>((static_cast<uint64_t>(beam_count) * shard_index) / shard_count);
        const uint32_t end_beam = static_cast<uint32_t>((static_cast<uint64_t>(beam_count) * (shard_index + 1u)) / shard_count);
        for (uint32_t beam_index = first_beam; beam_index < end_beam; ++beam_index) {
          if (enumerate_beam_cells(beams[beam_index], radius, description, [beam_index, counts, marks](const uint32_t cell_index) {
                if (marks[cell_index] != beam_index) {
                  marks[cell_index] = beam_index;
                  ++counts[cell_index];
                }
              }) == false) {
            shard_validity[shard_index] = 0u;
            break;
          }
        }
      }
    });
    if (std::find(shard_validity.begin(), shard_validity.end(), 0u) != shard_validity.end()) {
      return false;
    }

    _cell_offsets.resize(static_cast<size_t>(cell_count) + 1u);
    uint64_t entry_count = 0u;
    for (uint32_t cell_index = 0u; cell_index < cell_count; ++cell_index) {
      _cell_offsets[cell_index] = static_cast<uint32_t>(entry_count);
      for (uint32_t shard_index = 0u; shard_index < shard_count; ++shard_index) {
        const uint32_t count = shard_cell_counts[static_cast<size_t>(shard_index) * cell_count + cell_index];
        if (entry_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - count) {
          clear();
          return false;
        }
        entry_count += count;
      }
    }
    if ((entry_count == 0u) || (entry_count >= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))) {
      clear();
      return false;
    }
    _cell_offsets[cell_count] = static_cast<uint32_t>(entry_count);
    _beam_indices.resize(static_cast<size_t>(entry_count));

    for (uint32_t cell_index = 0u; cell_index < cell_count; ++cell_index) {
      uint32_t offset = _cell_offsets[cell_index];
      for (uint32_t shard_index = 0u; shard_index < shard_count; ++shard_index) {
        const size_t shard_cell_index = static_cast<size_t>(shard_index) * cell_count + cell_index;
        const uint32_t count = shard_cell_counts[shard_cell_index];
        shard_cell_counts[shard_cell_index] = offset;
        offset += count;
      }
    }
    std::fill(shard_cell_marks.begin(), shard_cell_marks.end(), kInvalidIndex);

    _beams.assign(beams, beams + beam_count);
    _bounds = description.bounds;
    _cell_size = description.cell_size;
    _inverse_cell_size = description.inverse_cell_size;
    _resolution = description.resolution;
    _maximum_radius = radius;
    std::fill(shard_validity.begin(), shard_validity.end(), 1u);
    execute(shard_count, [this, beam_count, radius, description, cell_count, shard_count, &shard_cell_counts, &shard_cell_marks, &shard_validity](const uint32_t begin,
                           const uint32_t end, const uint32_t) {
      for (uint32_t shard_index = begin; shard_index < end; ++shard_index) {
        uint32_t* offsets = shard_cell_counts.data() + static_cast<size_t>(shard_index) * cell_count;
        uint32_t* marks = shard_cell_marks.data() + static_cast<size_t>(shard_index) * cell_count;
        const uint32_t first_beam = static_cast<uint32_t>((static_cast<uint64_t>(beam_count) * shard_index) / shard_count);
        const uint32_t end_beam = static_cast<uint32_t>((static_cast<uint64_t>(beam_count) * (shard_index + 1u)) / shard_count);
        for (uint32_t beam_index = first_beam; beam_index < end_beam; ++beam_index) {
          if (enumerate_beam_cells(_beams[beam_index], radius, description, [this, beam_index, offsets, marks](const uint32_t cell_index) {
                if (marks[cell_index] != beam_index) {
                  marks[cell_index] = beam_index;
                  _beam_indices[offsets[cell_index]++] = beam_index;
                }
              }) == false) {
            shard_validity[shard_index] = 0u;
            break;
          }
        }
      }
    });
    if (std::find(shard_validity.begin(), shard_validity.end(), 0u) != shard_validity.end()) {
      clear();
      return false;
    }
    return true;
  }

 public:
  void clear() {
    _beams.clear();
    _beam_indices.clear();
    _cell_offsets.clear();
    _bounds = {};
    _cell_size = 0.0f;
    _inverse_cell_size = 0.0f;
    _resolution = {};
    _maximum_radius = 0.0f;
  }

  template <typename Candidate, typename Visitor>
  bool query_point_intersections(const float3& point, const float radius, Candidate&& candidate, Visitor&& visitor) const {
    if (_cell_offsets.empty() || (radius <= 0.0f) || (radius > _maximum_radius) || (std::isfinite(point.x) == false) || (std::isfinite(point.y) == false) ||
        (std::isfinite(point.z) == false)) {
      return false;
    }
    if ((point.x < _bounds.minimum.x) || (point.x > _bounds.maximum.x) || (point.y < _bounds.minimum.y) || (point.y > _bounds.maximum.y) || (point.z < _bounds.minimum.z) ||
        (point.z > _bounds.maximum.z)) {
      return true;
    }

    std::array<int32_t, 3u> cell = {};
    if (make_cell(point, cell) == false) {
      return false;
    }
    const uint32_t cell_index = linear_cell_index(static_cast<uint32_t>(cell[0u]), static_cast<uint32_t>(cell[1u]), static_cast<uint32_t>(cell[2u]), _resolution);
    visit_cell(cell_index, [this, &point, radius, &candidate, &visitor](const uint32_t beam_index) {
      const UPBPBeamReference& indexed_beam = _beams[beam_index];
      if (candidate(indexed_beam, beam_index) == false) {
        return;
      }
      UPBPPointBeamIntersection intersection = {};
      if (upbp_intersect_point_beam(point, indexed_beam, radius, intersection)) {
        visitor(indexed_beam, beam_index, intersection);
      }
    });
    return true;
  }

  template <typename Candidate, typename Visitor>
  bool query_beam_intersections(const UPBPBeamReference& beam, const float radius, UPBPSpatialQueryState& query_state, Candidate&& candidate, Visitor&& visitor) const {
    if (_cell_offsets.empty() || (beam_valid(beam) == false) || (radius <= 0.0f) || (radius > _maximum_radius) || (std::isfinite(radius) == false)) {
      return false;
    }
    query_state.begin(static_cast<uint32_t>(_beams.size()));

    double minimum_distance = 0.0;
    double maximum_distance = static_cast<double>(beam.length);
    if (intersect_bounds(beam, minimum_distance, maximum_distance) == false) {
      return true;
    }

    const double start_distance = max(0.0, minimum_distance);
    const double end_distance = min(static_cast<double>(beam.length), maximum_distance);
    if (start_distance >= end_distance) {
      return true;
    }

    const float3 start = beam.origin + beam.direction * static_cast<float>(start_distance);
    std::array<int32_t, 3u> cell = {};
    if (make_cell(start, cell) == false) {
      return false;
    }

    std::array<int32_t, 3u> step = {};
    std::array<double, 3u> next_distance = {};
    std::array<double, 3u> distance_step = {};
    const double infinity = std::numeric_limits<double>::infinity();
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
      const double direction = static_cast<double>(component(beam.direction, axis));
      if (direction > 0.0) {
        step[axis] = 1;
        const double boundary = static_cast<double>(component(_bounds.minimum, axis)) + static_cast<double>(cell[axis] + 1) * static_cast<double>(_cell_size);
        next_distance[axis] = (boundary - static_cast<double>(component(beam.origin, axis))) / direction;
        distance_step[axis] = static_cast<double>(_cell_size) / direction;
      } else if (direction < 0.0) {
        step[axis] = -1;
        const double boundary = static_cast<double>(component(_bounds.minimum, axis)) + static_cast<double>(cell[axis]) * static_cast<double>(_cell_size);
        next_distance[axis] = (boundary - static_cast<double>(component(beam.origin, axis))) / direction;
        distance_step[axis] = -static_cast<double>(_cell_size) / direction;
      } else {
        next_distance[axis] = infinity;
        distance_step[axis] = infinity;
      }
      while (next_distance[axis] < start_distance) {
        next_distance[axis] += distance_step[axis];
      }
    }

    const float radius_squared = radius * radius;
    for (;;) {
      const uint32_t cell_index = linear_cell_index(static_cast<uint32_t>(cell[0u]), static_cast<uint32_t>(cell[1u]), static_cast<uint32_t>(cell[2u]), _resolution);
      visit_cell(cell_index, [this, &beam, radius_squared, &query_state, &candidate, &visitor](const uint32_t beam_index) {
        if (query_state.mark(beam_index) == false) {
          return;
        }
        const UPBPBeamReference& indexed_beam = _beams[beam_index];
        if (candidate(indexed_beam, beam_index) == false) {
          return;
        }
        UPBPBeamBeamIntersection intersection = {};
        if (upbp_intersect_valid_beams_squared(indexed_beam, beam, radius_squared, intersection)) {
          visitor(indexed_beam, beam_index, intersection);
        }
      });

      const double next = min(next_distance[0u], min(next_distance[1u], next_distance[2u]));
      if (next >= end_distance) {
        break;
      }
      bool inside = true;
      for (uint32_t axis = 0u; axis < 3u; ++axis) {
        if (next_distance[axis] != next) {
          continue;
        }
        cell[axis] += step[axis];
        next_distance[axis] += distance_step[axis];
        inside = inside && (cell[axis] >= 0) && (cell[axis] < static_cast<int32_t>(_resolution[axis]));
      }
      if (inside == false) {
        break;
      }
    }
    return true;
  }

  uint32_t size() const {
    return static_cast<uint32_t>(_beams.size());
  }

  uint64_t storage_bytes() const {
    return static_cast<uint64_t>(_beams.capacity()) * sizeof(UPBPBeamReference) + static_cast<uint64_t>(_beam_indices.capacity()) * sizeof(uint32_t) +
           static_cast<uint64_t>(_cell_offsets.capacity()) * sizeof(uint32_t);
  }

  bool projected_storage_bytes(const UPBPBeamReference* beams, const uint32_t beam_count, const float radius, uint64_t& result) const {
    if (beam_count == 0u) {
      result = storage_bytes();
      return true;
    }
    Description description = {};
    if (describe(beams, beam_count, radius, description) == false) {
      return false;
    }
    uint64_t entry_count = 0u;
    if (count_entries(beams, beam_count, radius, description, entry_count) == false) {
      return false;
    }
    if (entry_count >= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      return false;
    }
    const uint64_t cell_count = grid_cell_count(description.resolution);

    const uint64_t beam_capacity = max(static_cast<uint64_t>(_beams.capacity()), static_cast<uint64_t>(beam_count));
    const uint64_t index_capacity = max(static_cast<uint64_t>(_beam_indices.capacity()), entry_count);
    const uint64_t offset_capacity = max(static_cast<uint64_t>(_cell_offsets.capacity()), cell_count + 1u);
    if ((beam_capacity > std::numeric_limits<uint64_t>::max() / sizeof(UPBPBeamReference)) || (index_capacity > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t)) ||
        (offset_capacity > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t))) {
      return false;
    }
    const std::array<uint64_t, 3u> storage = {
      beam_capacity * sizeof(UPBPBeamReference),
      index_capacity * sizeof(uint32_t),
      offset_capacity * sizeof(uint32_t),
    };
    result = 0u;
    for (const uint64_t bytes : storage) {
      if (result > std::numeric_limits<uint64_t>::max() - bytes) {
        return false;
      }
      result += bytes;
    }
    return true;
  }

 private:
  static float component(const float3& value, const uint32_t axis) {
    return axis == 0u ? value.x : (axis == 1u ? value.y : value.z);
  }

  static bool beam_valid(const UPBPBeamReference& beam) {
    const float direction_length_squared = dot(beam.direction, beam.direction);
    return (beam.length > 0.0f) && std::isfinite(beam.length) && std::isfinite(beam.origin.x) && std::isfinite(beam.origin.y) && std::isfinite(beam.origin.z) &&
           std::isfinite(direction_length_squared) && (fabsf(direction_length_squared - 1.0f) <= 1.0e-4f);
  }

  static void extend(Bounds& target, const Bounds& source) {
    target.minimum = min(target.minimum, source.minimum);
    target.maximum = max(target.maximum, source.maximum);
  }

  static Bounds support_bounds(const UPBPBeamReference& beam, const float radius) {
    const float3 end = beam.origin + beam.direction * beam.length;
    const float3 extent = {radius, radius, radius};
    return {min(beam.origin, end) - extent, max(beam.origin, end) + extent};
  }

  static bool describe(const UPBPBeamReference* beams, const uint32_t beam_count, const float radius, Description& result) {
    if ((beams == nullptr) || (beam_count == 0u) || (radius <= 0.0f) || (std::isfinite(radius) == false)) {
      return false;
    }
    result = {};
    for (uint32_t beam_index = 0u; beam_index < beam_count; ++beam_index) {
      if (beam_valid(beams[beam_index]) == false) {
        return false;
      }
      extend(result.bounds, support_bounds(beams[beam_index], radius));
    }
    const float3 extent = result.bounds.maximum - result.bounds.minimum;
    const float maximum_extent = max(extent.x, max(extent.y, extent.z));
    if ((maximum_extent <= 0.0f) || (std::isfinite(maximum_extent) == false)) {
      return false;
    }
    result.cell_size = std::nextafter(maximum_extent / static_cast<float>(maximum_resolution), std::numeric_limits<float>::infinity());
    if ((result.cell_size <= 0.0f) || (std::isfinite(result.cell_size) == false)) {
      return false;
    }
    result.inverse_cell_size = 1.0f / result.cell_size;
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
      const double axis_extent = static_cast<double>(component(result.bounds.maximum - result.bounds.minimum, axis));
      const uint32_t axis_resolution = static_cast<uint32_t>(ceil(axis_extent / static_cast<double>(result.cell_size)));
      result.resolution[axis] = min(maximum_resolution, max(1u, axis_resolution));
    }
    result.bounds.maximum = result.bounds.minimum + float3{static_cast<float>(result.resolution[0u]) * result.cell_size,
                                                      static_cast<float>(result.resolution[1u]) * result.cell_size, static_cast<float>(result.resolution[2u]) * result.cell_size};
    return true;
  }

  static uint32_t linear_cell_index(const uint32_t x, const uint32_t y, const uint32_t z, const std::array<uint32_t, 3u>& resolution) {
    return x + resolution[0u] * (y + resolution[1u] * z);
  }

  static uint32_t grid_cell_count(const std::array<uint32_t, 3u>& resolution) {
    return resolution[0u] * resolution[1u] * resolution[2u];
  }

  static bool make_cell(const float3& position, const Description& description, std::array<uint32_t, 3u>& result) {
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
      const double value = floor(
        (static_cast<double>(component(position, axis)) - static_cast<double>(component(description.bounds.minimum, axis))) * static_cast<double>(description.inverse_cell_size));
      if (std::isfinite(value) == false) {
        return false;
      }
      result[axis] = static_cast<uint32_t>(min(static_cast<double>(description.resolution[axis] - 1u), max(0.0, value)));
    }
    return true;
  }

  bool make_cell(const float3& position, std::array<int32_t, 3u>& result) const {
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
      const double value =
        floor((static_cast<double>(component(position, axis)) - static_cast<double>(component(_bounds.minimum, axis))) * static_cast<double>(_inverse_cell_size));
      if (std::isfinite(value) == false) {
        return false;
      }
      result[axis] = static_cast<int32_t>(min(static_cast<double>(_resolution[axis] - 1u), max(0.0, value)));
    }
    return true;
  }

  template <typename Visitor>
  static bool enumerate_beam_cells(const UPBPBeamReference& beam, const float radius, const Description& description, Visitor&& visitor) {
    // Chopped radius-inflated bounds conservatively cover the beam support; queries still perform the exact intersection test.
    const float3 absolute_direction = abs(beam.direction);
    const float dominant_direction = max(absolute_direction.x, max(absolute_direction.y, absolute_direction.z));
    const double projected_cell_count = ceil(static_cast<double>(beam.length) * static_cast<double>(dominant_direction) / static_cast<double>(description.cell_size));
    if ((std::isfinite(projected_cell_count) == false) || (projected_cell_count > static_cast<double>(std::numeric_limits<uint32_t>::max()))) {
      return false;
    }
    const uint32_t segment_count = max(1u, static_cast<uint32_t>(projected_cell_count));
    const float inverse_segment_count = 1.0f / static_cast<float>(segment_count);
    const float3 support_extent = {radius, radius, radius};
    for (uint32_t segment_index = 0u; segment_index < segment_count; ++segment_index) {
      const float first_distance = beam.length * (static_cast<float>(segment_index) * inverse_segment_count);
      const float second_distance = beam.length * (static_cast<float>(segment_index + 1u) * inverse_segment_count);
      const float3 first = beam.origin + beam.direction * first_distance;
      const float3 second = beam.origin + beam.direction * second_distance;
      std::array<uint32_t, 3u> minimum_cell = {};
      std::array<uint32_t, 3u> maximum_cell = {};
      if ((make_cell(min(first, second) - support_extent, description, minimum_cell) == false) ||
          (make_cell(max(first, second) + support_extent, description, maximum_cell) == false)) {
        return false;
      }
      for (uint32_t z = minimum_cell[2u]; z <= maximum_cell[2u]; ++z) {
        for (uint32_t y = minimum_cell[1u]; y <= maximum_cell[1u]; ++y) {
          for (uint32_t x = minimum_cell[0u]; x <= maximum_cell[0u]; ++x) {
            visitor(linear_cell_index(x, y, z, description.resolution));
          }
        }
      }
    }
    return true;
  }

  static bool count_entries(const UPBPBeamReference* beams, const uint32_t beam_count, const float radius, const Description& description, uint64_t& result) {
    try {
      std::vector<uint32_t> cell_marks(grid_cell_count(description.resolution), kInvalidIndex);
      result = 0u;
      for (uint32_t beam_index = 0u; beam_index < beam_count; ++beam_index) {
        uint64_t beam_entry_count = 0u;
        if (enumerate_beam_cells(beams[beam_index], radius, description, [beam_index, &beam_entry_count, &cell_marks](const uint32_t cell_index) {
              if (cell_marks[cell_index] != beam_index) {
                cell_marks[cell_index] = beam_index;
                ++beam_entry_count;
              }
            }) == false) {
          return false;
        }
        if (result > std::numeric_limits<uint64_t>::max() - beam_entry_count) {
          return false;
        }
        result += beam_entry_count;
      }
    } catch (const std::bad_alloc&) {
      return false;
    } catch (const std::length_error&) {
      return false;
    }
    return true;
  }

  template <typename Visitor>
  void visit_cell(const uint32_t cell_index, Visitor&& visitor) const {
    for (uint32_t index = _cell_offsets[cell_index]; index < _cell_offsets[cell_index + 1u]; ++index) {
      visitor(_beam_indices[index]);
    }
  }

  bool intersect_bounds(const UPBPBeamReference& beam, double& minimum_distance, double& maximum_distance) const {
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
      const double origin = static_cast<double>(component(beam.origin, axis));
      const double direction = static_cast<double>(component(beam.direction, axis));
      const double minimum = static_cast<double>(component(_bounds.minimum, axis));
      const double maximum = static_cast<double>(component(_bounds.maximum, axis));
      if (direction == 0.0) {
        if ((origin < minimum) || (origin > maximum)) {
          return false;
        }
        continue;
      }
      double first = (minimum - origin) / direction;
      double second = (maximum - origin) / direction;
      if (first > second) {
        std::swap(first, second);
      }
      minimum_distance = max(minimum_distance, first);
      maximum_distance = min(maximum_distance, second);
      if (minimum_distance > maximum_distance) {
        return false;
      }
    }
    return true;
  }

  std::vector<UPBPBeamReference> _beams = {};
  std::vector<uint32_t> _beam_indices = {};
  std::vector<uint32_t> _cell_offsets = {};
  Bounds _bounds = {};
  float _cell_size = 0.0f;
  float _inverse_cell_size = 0.0f;
  std::array<uint32_t, 3u> _resolution = {};
  float _maximum_radius = 0.0f;
};

}  // namespace etx
