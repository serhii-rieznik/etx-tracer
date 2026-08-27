#pragma once

#include <etx/rt/integrators/upbp_core.hxx>

#include <cmath>
#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace etx {

struct UPBPBeamReference;
struct UPBPPointBeamIntersection;

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

  struct Bounds {
    float3 minimum = {kMaxFloat, kMaxFloat, kMaxFloat};
    float3 maximum = {-kMaxFloat, -kMaxFloat, -kMaxFloat};
  };

  struct Node {
    Bounds bounds = {};
    uint32_t first = 0u;
    uint32_t count = 0u;
    uint32_t left = kInvalidIndex;
    uint32_t right = kInvalidIndex;
  };

  bool build(const UPBPPointReference* points, const uint32_t point_count, const float cell_size, const bool build_beam_acceleration) {
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
    if (build_beam_acceleration) {
      _indices.resize(point_count);
      for (uint32_t index = 0u; index < point_count; ++index) {
        _indices[index] = index;
      }
      _nodes.reserve(static_cast<size_t>(point_count) * 2u);
      build_node(0u, point_count);
    }
    return true;
  }

  void clear() {
    _entries.clear();
    _cell_ranges.clear();
    _cell_slots.clear();
    _indices.clear();
    _nodes.clear();
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
          visit_cell({x, y, z}, [&position, radius_squared, &visitor](const UPBPPointReference& point) {
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
  bool query_beam(const UPBPBeamReference& beam, float radius, Visitor&& visitor) const;

  uint32_t size() const {
    return static_cast<uint32_t>(_entries.size());
  }

  uint64_t storage_bytes() const {
    return static_cast<uint64_t>(_entries.capacity()) * sizeof(Entry) + static_cast<uint64_t>(_cell_ranges.capacity()) * sizeof(CellRange) +
           static_cast<uint64_t>(_cell_slots.capacity()) * sizeof(uint32_t) + static_cast<uint64_t>(_indices.capacity()) * sizeof(uint32_t) +
           static_cast<uint64_t>(_nodes.capacity()) * sizeof(Node);
  }

  bool projected_storage_bytes(const uint32_t point_count, const bool include_beam_acceleration, uint64_t& result) const {
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
    if (include_beam_acceleration == false) {
      return true;
    }

    const uint64_t index_capacity = max(static_cast<uint64_t>(_indices.capacity()), static_cast<uint64_t>(point_count));
    const uint64_t node_count = static_cast<uint64_t>(point_count) * 2u;
    const uint64_t node_capacity = max(static_cast<uint64_t>(_nodes.capacity()), node_count);
    if ((index_capacity > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t)) || (node_capacity > std::numeric_limits<uint64_t>::max() / sizeof(Node))) {
      return false;
    }
    const uint64_t index_bytes = index_capacity * sizeof(uint32_t);
    const uint64_t node_bytes = node_capacity * sizeof(Node);
    if ((result > std::numeric_limits<uint64_t>::max() - index_bytes) || ((result + index_bytes) > std::numeric_limits<uint64_t>::max() - node_bytes)) {
      return false;
    }
    result += index_bytes + node_bytes;
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

  static void extend(Bounds& target, const Bounds& source) {
    target.minimum = min(target.minimum, source.minimum);
    target.maximum = max(target.maximum, source.maximum);
  }

  static bool overlaps(const Bounds& first, const Bounds& second) {
    return (first.minimum.x <= second.maximum.x) && (first.maximum.x >= second.minimum.x) && (first.minimum.y <= second.maximum.y) && (first.maximum.y >= second.minimum.y) &&
           (first.minimum.z <= second.maximum.z) && (first.maximum.z >= second.minimum.z);
  }

  float point_component(const uint32_t entry_index, const uint32_t axis) const {
    const float3& position = _entries[entry_index].point.position;
    return axis == 0u ? position.x : (axis == 1u ? position.y : position.z);
  }

  uint32_t build_node(const uint32_t first, const uint32_t count) {
    const uint32_t node_index = static_cast<uint32_t>(_nodes.size());
    _nodes.emplace_back();
    Bounds bounds = {};
    for (uint32_t offset = 0u; offset < count; ++offset) {
      const float3& position = _entries[_indices[first + offset]].point.position;
      extend(bounds, {position, position});
    }

    _nodes[node_index].bounds = bounds;
    _nodes[node_index].first = first;
    _nodes[node_index].count = count;
    if (count <= 8u) {
      return node_index;
    }

    const float3 extent = bounds.maximum - bounds.minimum;
    const uint32_t axis = (extent.x >= extent.y) && (extent.x >= extent.z) ? 0u : ((extent.y >= extent.z) ? 1u : 2u);
    const uint32_t middle = first + count / 2u;
    std::nth_element(_indices.begin() + first, _indices.begin() + middle, _indices.begin() + first + count, [this, axis](const uint32_t first_index, const uint32_t second_index) {
      const float first_position = point_component(first_index, axis);
      const float second_position = point_component(second_index, axis);
      return first_position == second_position ? first_index < second_index : first_position < second_position;
    });
    const uint32_t left = build_node(first, middle - first);
    const uint32_t right = build_node(middle, first + count - middle);
    _nodes[node_index].left = left;
    _nodes[node_index].right = right;
    _nodes[node_index].count = 0u;
    return node_index;
  }

  template <typename Visitor>
  bool query_bounds(const Bounds& bounds, Visitor&& visitor) const {
    if (_nodes.empty()) {
      return false;
    }

    std::array<uint32_t, 128u> stack = {};
    uint32_t stack_size = 1u;
    stack[0u] = 0u;
    while (stack_size > 0u) {
      const Node& node = _nodes[stack[--stack_size]];
      if (overlaps(node.bounds, bounds) == false) {
        continue;
      }
      if (node.count > 0u) {
        for (uint32_t offset = 0u; offset < node.count; ++offset) {
          visitor(_entries[_indices[node.first + offset]].point);
        }
        continue;
      }
      if (stack_size + 2u > stack.size()) {
        return false;
      }
      stack[stack_size++] = node.left;
      stack[stack_size++] = node.right;
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
          visitor(_entries[range.first + offset].point);
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
  std::vector<uint32_t> _indices = {};
  std::vector<Node> _nodes = {};
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
bool UPBPPointIndex::query_beam(const UPBPBeamReference& beam, const float radius, Visitor&& visitor) const {
  if ((_cell_size <= 0.0f) || _nodes.empty() || (radius <= 0.0f) || (radius > _cell_size) || (std::isfinite(radius) == false) || (beam.length <= 0.0f) ||
      (std::isfinite(beam.length) == false) || (fabsf(dot(beam.direction, beam.direction) - 1.0f) > 1.0e-4f)) {
    return false;
  }

  const float3 end = beam.origin + beam.direction * beam.length;
  const float3 extent = {radius, radius, radius};
  const Bounds bounds = {min(beam.origin, end) - extent, max(beam.origin, end) + extent};
  return query_bounds(bounds, [&beam, radius, &visitor](const UPBPPointReference& point) {
    UPBPPointBeamIntersection intersection = {};
    if (upbp_intersect_point_beam(point.position, beam, radius, intersection)) {
      visitor(point, intersection);
    }
  });
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

struct UPBPBeamIndex {
  struct Bounds {
    float3 minimum = {kMaxFloat, kMaxFloat, kMaxFloat};
    float3 maximum = {-kMaxFloat, -kMaxFloat, -kMaxFloat};
  };

  struct Node {
    Bounds bounds = {};
    uint32_t first = 0u;
    uint32_t count = 0u;
    uint32_t escape = kInvalidIndex;
  };

  struct BeamBoundsQuery {
    double origin_x = 0.0;
    double origin_y = 0.0;
    double origin_z = 0.0;
    double direction_x = 0.0;
    double direction_y = 0.0;
    double direction_z = 0.0;
    double inverse_direction_x = 0.0;
    double inverse_direction_y = 0.0;
    double inverse_direction_z = 0.0;
    double maximum_distance = 0.0;
    double radius = 0.0;
    float radius_squared = 0.0f;
  };

  bool build(const UPBPBeamReference* beams, const uint32_t beam_count, const float maximum_radius) {
    clear();
    if ((beams == nullptr) || (beam_count == 0u) || (maximum_radius <= 0.0f) || (std::isfinite(maximum_radius) == false)) {
      return false;
    }

    _maximum_radius = maximum_radius;
    _beams.assign(beams, beams + beam_count);
    _beam_bounds.resize(beam_count);
    _indices.resize(beam_count);
    for (uint32_t index = 0u; index < beam_count; ++index) {
      _indices[index] = index;
      if (beam_valid(_beams[index]) == false) {
        clear();
        return false;
      }
      _beam_bounds[index] = beam_bounds(_beams[index]);
    }
    _nodes.reserve(static_cast<size_t>(beam_count) * 2u);
    build_node(0u, beam_count, 0u);
    _source_indices = _indices;
    for (uint32_t index = 0u; index < beam_count; ++index) {
      _indices[_source_indices[index]] = index;
    }
    for (uint32_t index = 0u; index < beam_count; ++index) {
      while (_indices[index] != index) {
        const uint32_t destination = _indices[index];
        std::swap(_beams[index], _beams[destination]);
        std::swap(_beam_bounds[index], _beam_bounds[destination]);
        std::swap(_indices[index], _indices[destination]);
      }
    }
    _indices = {};
    return true;
  }

  void clear() {
    _beams.clear();
    _beam_bounds.clear();
    _source_indices.clear();
    _indices.clear();
    _nodes.clear();
    _maximum_radius = 0.0f;
  }

  template <typename Visitor>
  bool query_point(const float3& point, Visitor&& visitor) const {
    return query_point(point, _maximum_radius, std::forward<Visitor>(visitor));
  }

  template <typename Visitor>
  bool query_point(const float3& point, const float radius, Visitor&& visitor) const {
    if (_nodes.empty() || (radius <= 0.0f) || (radius > _maximum_radius) || (std::isfinite(point.x) == false) || (std::isfinite(point.y) == false) ||
        (std::isfinite(point.z) == false)) {
      return false;
    }
    return traverse(
      [&point, radius](const Bounds& bounds) {
        return overlaps_point(bounds, point, radius);
      },
      [this, &point, radius, &visitor](const uint32_t beam_index) {
        if (overlaps_point(_beam_bounds[beam_index], point, radius)) {
          visitor(_beams[beam_index]);
        }
      });
  }

  template <typename Candidate, typename Visitor>
  bool query_point_intersections(const float3& point, const float radius, Candidate&& candidate, Visitor&& visitor) const {
    if (_nodes.empty() || (radius <= 0.0f) || (radius > _maximum_radius) || (std::isfinite(point.x) == false) || (std::isfinite(point.y) == false) ||
        (std::isfinite(point.z) == false)) {
      return false;
    }
    return traverse(
      [&point, radius](const Bounds& bounds) {
        return overlaps_point(bounds, point, radius);
      },
      [this, &point, radius, &candidate, &visitor](const uint32_t storage_index) {
        const UPBPBeamReference& indexed_beam = _beams[storage_index];
        if ((overlaps_point(_beam_bounds[storage_index], point, radius) == false) || (candidate(indexed_beam, storage_index) == false)) {
          return;
        }
        UPBPPointBeamIntersection intersection = {};
        if (upbp_intersect_point_beam(point, indexed_beam, radius, intersection)) {
          visitor(indexed_beam, storage_index, intersection);
        }
      });
  }

  template <typename Visitor>
  bool query_beam(const UPBPBeamReference& beam, const float radius, Visitor&& visitor) const {
    if (_nodes.empty() || (beam_valid(beam) == false) || (radius <= 0.0f) || (radius > _maximum_radius)) {
      return false;
    }
    const BeamBoundsQuery query = make_beam_bounds_query(beam, radius);
    return traverse(
      [&query](const Bounds& bounds) {
        return overlaps_beam(bounds, query);
      },
      [this, &query, &visitor](const uint32_t beam_index) {
        if (overlaps_beam(_beam_bounds[beam_index], query)) {
          visitor(_beams[beam_index]);
        }
      });
  }

  template <typename Candidate, typename Visitor>
  bool query_beam_intersections(const UPBPBeamReference& beam, const float radius, Candidate&& candidate, Visitor&& visitor) const {
    if (_nodes.empty() || (beam_valid(beam) == false) || (radius <= 0.0f) || (radius > _maximum_radius)) {
      return false;
    }
    const BeamBoundsQuery query = make_beam_bounds_query(beam, radius);
    return traverse(
      [&query](const Bounds& bounds) {
        return overlaps_beam(bounds, query);
      },
      [this, &beam, &query, &candidate, &visitor](const uint32_t storage_index) {
        const UPBPBeamReference& indexed_beam = _beams[storage_index];
        if (candidate(indexed_beam, storage_index) == false) {
          return;
        }
        UPBPBeamBeamIntersection intersection = {};
        if (upbp_intersect_valid_beams_squared(indexed_beam, beam, query.radius_squared, intersection)) {
          visitor(indexed_beam, storage_index, intersection);
        }
      });
  }

  template <typename T>
  bool reorder_by_leaf_layout(std::vector<T>& values) const {
    if (values.size() != _beams.size()) {
      return false;
    }
    std::vector<uint32_t> destinations(values.size());
    for (uint32_t storage_index = 0u; storage_index < _beams.size(); ++storage_index) {
      destinations[_source_indices[storage_index]] = storage_index;
    }
    for (uint32_t source_index = 0u; source_index < destinations.size(); ++source_index) {
      while (destinations[source_index] != source_index) {
        const uint32_t destination = destinations[source_index];
        std::swap(values[source_index], values[destination]);
        std::swap(destinations[source_index], destinations[destination]);
      }
    }
    return true;
  }

  uint32_t size() const {
    return static_cast<uint32_t>(_beams.size());
  }

  uint64_t storage_bytes() const {
    return static_cast<uint64_t>(_beams.capacity()) * sizeof(UPBPBeamReference) + static_cast<uint64_t>(_beam_bounds.capacity()) * sizeof(Bounds) +
           static_cast<uint64_t>(_source_indices.capacity()) * sizeof(uint32_t) + static_cast<uint64_t>(_indices.capacity()) * sizeof(uint32_t) +
           static_cast<uint64_t>(_nodes.capacity()) * sizeof(Node);
  }

  bool projected_storage_bytes(const uint32_t beam_count, uint64_t& result) const {
    const uint64_t beam_capacity = max(static_cast<uint64_t>(_beams.capacity()), static_cast<uint64_t>(beam_count));
    const uint64_t bounds_capacity = max(static_cast<uint64_t>(_beam_bounds.capacity()), static_cast<uint64_t>(beam_count));
    const uint64_t source_index_capacity = max(static_cast<uint64_t>(_source_indices.capacity()), static_cast<uint64_t>(beam_count));
    const uint64_t index_capacity = max(static_cast<uint64_t>(_indices.capacity()), static_cast<uint64_t>(beam_count));
    const uint64_t node_count = static_cast<uint64_t>(beam_count) * 2u;
    const uint64_t node_capacity = max(static_cast<uint64_t>(_nodes.capacity()), node_count);
    if ((beam_capacity > std::numeric_limits<uint64_t>::max() / sizeof(UPBPBeamReference)) || (bounds_capacity > std::numeric_limits<uint64_t>::max() / sizeof(Bounds)) ||
        (source_index_capacity > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t)) || (index_capacity > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t)) ||
        (node_capacity > std::numeric_limits<uint64_t>::max() / sizeof(Node))) {
      return false;
    }
    const uint64_t beam_bytes = beam_capacity * sizeof(UPBPBeamReference);
    const uint64_t bounds_bytes = bounds_capacity * sizeof(Bounds);
    const uint64_t source_index_bytes = source_index_capacity * sizeof(uint32_t);
    const uint64_t index_bytes = index_capacity * sizeof(uint32_t);
    const uint64_t node_bytes = node_capacity * sizeof(Node);
    if ((beam_bytes > std::numeric_limits<uint64_t>::max() - bounds_bytes) || ((beam_bytes + bounds_bytes) > std::numeric_limits<uint64_t>::max() - source_index_bytes) ||
        ((beam_bytes + bounds_bytes + source_index_bytes) > std::numeric_limits<uint64_t>::max() - index_bytes) ||
        ((beam_bytes + bounds_bytes + source_index_bytes + index_bytes) > std::numeric_limits<uint64_t>::max() - node_bytes)) {
      return false;
    }
    result = beam_bytes + bounds_bytes + source_index_bytes + index_bytes + node_bytes;
    return true;
  }

 private:
  static bool beam_valid(const UPBPBeamReference& beam) {
    const float direction_length_squared = dot(beam.direction, beam.direction);
    return (beam.length > 0.0f) && std::isfinite(beam.length) && std::isfinite(beam.origin.x) && std::isfinite(beam.origin.y) && std::isfinite(beam.origin.z) &&
           std::isfinite(direction_length_squared) && (fabsf(direction_length_squared - 1.0f) <= 1.0e-4f);
  }

  static Bounds beam_bounds(const UPBPBeamReference& beam) {
    const float3 end = beam.origin + beam.direction * beam.length;
    const float3 minimum = min(beam.origin, end);
    const float3 maximum = max(beam.origin, end);
    return {
      {
        std::nextafter(minimum.x, -std::numeric_limits<float>::infinity()),
        std::nextafter(minimum.y, -std::numeric_limits<float>::infinity()),
        std::nextafter(minimum.z, -std::numeric_limits<float>::infinity()),
      },
      {
        std::nextafter(maximum.x, std::numeric_limits<float>::infinity()),
        std::nextafter(maximum.y, std::numeric_limits<float>::infinity()),
        std::nextafter(maximum.z, std::numeric_limits<float>::infinity()),
      },
    };
  }

  static void extend(Bounds& target, const Bounds& source) {
    target.minimum = min(target.minimum, source.minimum);
    target.maximum = max(target.maximum, source.maximum);
  }

  static bool overlaps_point(const Bounds& bounds, const float3& point, const float radius) {
    return (point.x >= bounds.minimum.x - radius) && (point.x <= bounds.maximum.x + radius) && (point.y >= bounds.minimum.y - radius) && (point.y <= bounds.maximum.y + radius) &&
           (point.z >= bounds.minimum.z - radius) && (point.z <= bounds.maximum.z + radius);
  }

  static float surface_area(const Bounds& bounds) {
    const float3 extent = max(bounds.maximum - bounds.minimum, float3{0.0f, 0.0f, 0.0f});
    return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
  }

  static BeamBoundsQuery make_beam_bounds_query(const UPBPBeamReference& beam, const float radius) {
    BeamBoundsQuery result = {};
    result.origin_x = static_cast<double>(beam.origin.x);
    result.origin_y = static_cast<double>(beam.origin.y);
    result.origin_z = static_cast<double>(beam.origin.z);
    result.direction_x = static_cast<double>(beam.direction.x);
    result.direction_y = static_cast<double>(beam.direction.y);
    result.direction_z = static_cast<double>(beam.direction.z);
    result.inverse_direction_x = beam.direction.x == 0.0f ? 0.0 : 1.0 / result.direction_x;
    result.inverse_direction_y = beam.direction.y == 0.0f ? 0.0 : 1.0 / result.direction_y;
    result.inverse_direction_z = beam.direction.z == 0.0f ? 0.0 : 1.0 / result.direction_z;
    result.maximum_distance = static_cast<double>(beam.length);
    result.radius = static_cast<double>(radius);
    result.radius_squared = radius * radius;
    return result;
  }

  static bool overlaps_beam(const Bounds& bounds, const BeamBoundsQuery& query) {
    double minimum_distance = 0.0;
    double maximum_distance = query.maximum_distance;
    auto overlaps_axis = [&minimum_distance, &maximum_distance](const double origin, const double direction, const double inverse_direction, const double expanded_minimum,
                           const double expanded_maximum) {
      if (direction == 0.0) {
        return (origin >= expanded_minimum) && (origin <= expanded_maximum);
      }
      double first_distance = (expanded_minimum - origin) * inverse_direction;
      double second_distance = (expanded_maximum - origin) * inverse_direction;
      if (first_distance > second_distance) {
        std::swap(first_distance, second_distance);
      }
      minimum_distance = max(minimum_distance, first_distance);
      maximum_distance = min(maximum_distance, second_distance);
      return minimum_distance <= maximum_distance;
    };
    return overlaps_axis(query.origin_x, query.direction_x, query.inverse_direction_x, static_cast<double>(bounds.minimum.x) - query.radius,
             static_cast<double>(bounds.maximum.x) + query.radius) &&
           overlaps_axis(query.origin_y, query.direction_y, query.inverse_direction_y, static_cast<double>(bounds.minimum.y) - query.radius,
             static_cast<double>(bounds.maximum.y) + query.radius) &&
           overlaps_axis(query.origin_z, query.direction_z, query.inverse_direction_z, static_cast<double>(bounds.minimum.z) - query.radius,
             static_cast<double>(bounds.maximum.z) + query.radius);
  }

  float centroid_component(const uint32_t beam_index, const uint32_t axis) const {
    const Bounds& bounds = _beam_bounds[beam_index];
    const float3 centroid = 0.5f * (bounds.minimum + bounds.maximum);
    return axis == 0u ? centroid.x : (axis == 1u ? centroid.y : centroid.z);
  }

  void build_node(const uint32_t first, const uint32_t count, const uint32_t depth) {
    const uint32_t node_index = static_cast<uint32_t>(_nodes.size());
    _nodes.emplace_back();
    Bounds bounds = {};
    Bounds centroid_bounds = {};
    for (uint32_t offset = 0u; offset < count; ++offset) {
      const uint32_t beam_index = _indices[first + offset];
      const Bounds& beam_bounds = _beam_bounds[beam_index];
      extend(bounds, beam_bounds);
      const float3 centroid = 0.5f * (beam_bounds.minimum + beam_bounds.maximum);
      extend(centroid_bounds, {centroid, centroid});
    }

    _nodes[node_index].bounds = bounds;
    _nodes[node_index].first = first;
    _nodes[node_index].count = count;
    if (count <= 16u) {
      _nodes[node_index].escape = static_cast<uint32_t>(_nodes.size());
      return;
    }

    const float3 extent = centroid_bounds.maximum - centroid_bounds.minimum;
    constexpr uint32_t bin_count = 16u;
    struct Bin {
      Bounds bounds = {};
      uint32_t count = 0u;
    };
    float best_cost = kMaxFloat;
    uint32_t best_axis = kInvalidIndex;
    uint32_t best_split = 0u;
    constexpr uint32_t maximum_sah_depth = 64u;
    if (depth < maximum_sah_depth) {
      for (uint32_t axis = 0u; axis < 3u; ++axis) {
        const float axis_extent = axis == 0u ? extent.x : (axis == 1u ? extent.y : extent.z);
        if (axis_extent <= 0.0f) {
          continue;
        }
        const float axis_minimum = axis == 0u ? centroid_bounds.minimum.x : (axis == 1u ? centroid_bounds.minimum.y : centroid_bounds.minimum.z);
        const float bin_scale = static_cast<float>(bin_count) / axis_extent;
        std::array<Bin, bin_count> bins = {};
        for (uint32_t offset = 0u; offset < count; ++offset) {
          const uint32_t beam_index = _indices[first + offset];
          const float centroid = centroid_component(beam_index, axis);
          const uint32_t bin_index = min(static_cast<uint32_t>((centroid - axis_minimum) * bin_scale), bin_count - 1u);
          extend(bins[bin_index].bounds, _beam_bounds[beam_index]);
          ++bins[bin_index].count;
        }

        std::array<Bounds, bin_count - 1u> left_bounds = {};
        std::array<Bounds, bin_count - 1u> right_bounds = {};
        std::array<uint32_t, bin_count - 1u> left_counts = {};
        std::array<uint32_t, bin_count - 1u> right_counts = {};
        Bounds left_bounds_accumulated = {};
        Bounds right_bounds_accumulated = {};
        uint32_t left_count = 0u;
        uint32_t right_count = 0u;
        for (uint32_t split = 0u; split + 1u < bin_count; ++split) {
          extend(left_bounds_accumulated, bins[split].bounds);
          left_count += bins[split].count;
          left_bounds[split] = left_bounds_accumulated;
          left_counts[split] = left_count;

          const uint32_t reverse_bin = bin_count - 1u - split;
          extend(right_bounds_accumulated, bins[reverse_bin].bounds);
          right_count += bins[reverse_bin].count;
          right_bounds[bin_count - 2u - split] = right_bounds_accumulated;
          right_counts[bin_count - 2u - split] = right_count;
        }
        for (uint32_t split = 0u; split + 1u < bin_count; ++split) {
          if ((left_counts[split] == 0u) || (right_counts[split] == 0u)) {
            continue;
          }
          const float cost =
            surface_area(left_bounds[split]) * static_cast<float>(left_counts[split]) + surface_area(right_bounds[split]) * static_cast<float>(right_counts[split]);
          if (cost < best_cost) {
            best_cost = cost;
            best_axis = axis;
            best_split = split;
          }
        }
      }
    }

    uint32_t middle = first + count / 2u;
    if (best_axis != kInvalidIndex) {
      const float axis_extent = best_axis == 0u ? extent.x : (best_axis == 1u ? extent.y : extent.z);
      const float axis_minimum = best_axis == 0u ? centroid_bounds.minimum.x : (best_axis == 1u ? centroid_bounds.minimum.y : centroid_bounds.minimum.z);
      const float bin_scale = static_cast<float>(bin_count) / axis_extent;
      const auto middle_iterator =
        std::partition(_indices.begin() + first, _indices.begin() + first + count, [this, best_axis, best_split, axis_minimum, bin_scale](const uint32_t beam_index) {
          const float centroid = centroid_component(beam_index, best_axis);
          const uint32_t bin_index = min(static_cast<uint32_t>((centroid - axis_minimum) * bin_scale), bin_count - 1u);
          return bin_index <= best_split;
        });
      middle = static_cast<uint32_t>(middle_iterator - _indices.begin());
    }
    if ((middle == first) || (middle == first + count)) {
      const uint32_t axis = (extent.x >= extent.y) && (extent.x >= extent.z) ? 0u : ((extent.y >= extent.z) ? 1u : 2u);
      middle = first + count / 2u;
      std::nth_element(_indices.begin() + first, _indices.begin() + middle, _indices.begin() + first + count,
        [this, axis](const uint32_t first_index, const uint32_t second_index) {
          const float first_centroid = centroid_component(first_index, axis);
          const float second_centroid = centroid_component(second_index, axis);
          return first_centroid == second_centroid ? first_index < second_index : first_centroid < second_centroid;
        });
    }
    build_node(middle, first + count - middle, depth + 1u);
    build_node(first, middle - first, depth + 1u);
    _nodes[node_index].count = 0u;
    _nodes[node_index].escape = static_cast<uint32_t>(_nodes.size());
  }

  template <typename Predicate, typename Visitor>
  bool traverse(Predicate&& predicate, Visitor&& visitor) const {
    uint32_t node_index = 0u;
    while (node_index < _nodes.size()) {
      const Node& node = _nodes[node_index];
      if (predicate(node.bounds) == false) {
        node_index = node.escape;
        continue;
      }

      if (node.count > 0u) {
        for (uint32_t offset = 0u; offset < node.count; ++offset) {
          visitor(node.first + offset);
        }
        node_index = node.escape;
      } else {
        ++node_index;
      }
    }
    return true;
  }

  std::vector<UPBPBeamReference> _beams = {};
  std::vector<Bounds> _beam_bounds = {};
  std::vector<uint32_t> _source_indices = {};
  std::vector<uint32_t> _indices = {};
  std::vector<Node> _nodes = {};
  float _maximum_radius = 0.0f;
};

}  // namespace etx
