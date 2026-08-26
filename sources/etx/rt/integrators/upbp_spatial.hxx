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
    return static_cast<uint64_t>(_entries.capacity()) * sizeof(Entry) + static_cast<uint64_t>(_indices.capacity()) * sizeof(uint32_t) +
           static_cast<uint64_t>(_nodes.capacity()) * sizeof(Node);
  }

  bool projected_storage_bytes(const uint32_t point_count, const bool include_beam_acceleration, uint64_t& result) const {
    const uint64_t capacity = max(static_cast<uint64_t>(_entries.capacity()), static_cast<uint64_t>(point_count));
    if (capacity > std::numeric_limits<uint64_t>::max() / sizeof(Entry)) {
      return false;
    }
    result = capacity * sizeof(Entry);
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
    const auto begin = std::lower_bound(_entries.begin(), _entries.end(), cell, [](const Entry& entry, const Cell& value) {
      return cell_less(entry.cell, value);
    });
    for (auto entry = begin; (entry != _entries.end()) && (entry->cell == cell); ++entry) {
      visitor(entry->point);
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

  const SpectralQuery spect = interval.weight.as_query();
  result = {};
  result.weight = SpectralResponse{spect, 1.0f};
  float covered_distance = 0.0f;
  float majorant = 0.0f;
  for (const MediumTrackingEvent& event : interval.events) {
    majorant = event.majorant;
    if (covered_distance + event.distance >= prefix_distance) {
      break;
    }
    if ((event.valid() == false) || (event.type != MediumTrackingEventType::Null) || (event.pdf_forward <= 0.0f) || (event.pdf_reverse <= 0.0f) ||
        (std::isfinite(event.pdf_forward) == false) || (std::isfinite(event.pdf_reverse) == false)) {
      return false;
    }
    result.weight *= event.weight;
    result.log_transport_pdf_forward += std::log(static_cast<double>(event.pdf_forward));
    result.log_transport_pdf_reverse += std::log(static_cast<double>(event.pdf_reverse));
    covered_distance += event.distance;
  }
  if ((majorant <= 0.0f) || (std::isfinite(majorant) == false)) {
    return false;
  }

  const float remaining_distance = prefix_distance - covered_distance;
  const float transmittance = expf(-majorant * remaining_distance);
  if ((transmittance <= 0.0f) || (std::isfinite(transmittance) == false)) {
    return false;
  }
  const double log_transmittance = std::log(static_cast<double>(transmittance));
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
};

inline bool upbp_intersect_beams(const UPBPBeamReference& first, const UPBPBeamReference& second, const float radius, UPBPBeamBeamIntersection& result) {
  if ((first.length <= 0.0f) || (second.length <= 0.0f) || (radius <= 0.0f) || (fabsf(dot(first.direction, first.direction) - 1.0f) > 1.0e-4f) ||
      (fabsf(dot(second.direction, second.direction) - 1.0f) > 1.0e-4f)) {
    return false;
  }

  const float direction_dot = dot(first.direction, second.direction);
  const float denominator = 1.0f - direction_dot * direction_dot;
  constexpr float degeneracy_threshold = 16.0f * std::numeric_limits<float>::epsilon();
  if (denominator <= degeneracy_threshold) {
    return false;
  }

  const float3 origin_delta = first.origin - second.origin;
  const float first_projection = dot(first.direction, origin_delta);
  const float second_projection = dot(second.direction, origin_delta);
  result.first_distance = (direction_dot * second_projection - first_projection) / denominator;
  result.second_distance = (second_projection - direction_dot * first_projection) / denominator;
  if ((result.first_distance < 0.0f) || (result.first_distance >= first.length) || (result.second_distance < 0.0f) || (result.second_distance >= second.length)) {
    return false;
  }

  const float3 first_point = first.origin + first.direction * result.first_distance;
  const float3 second_point = second.origin + second.direction * result.second_distance;
  const float3 delta = first_point - second_point;
  result.distance_squared = dot(delta, delta);
  result.sin_theta = sqrtf(denominator);
  return result.distance_squared < radius * radius;
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
    uint32_t left = kInvalidIndex;
    uint32_t right = kInvalidIndex;
  };

  bool build(const UPBPBeamReference* beams, const uint32_t beam_count, const float maximum_radius) {
    clear();
    if ((beams == nullptr) || (beam_count == 0u) || (maximum_radius <= 0.0f) || (std::isfinite(maximum_radius) == false)) {
      return false;
    }

    _maximum_radius = maximum_radius;
    _beams.assign(beams, beams + beam_count);
    _indices.resize(beam_count);
    for (uint32_t index = 0u; index < beam_count; ++index) {
      _indices[index] = index;
      if (beam_valid(_beams[index]) == false) {
        clear();
        return false;
      }
    }
    _nodes.reserve(static_cast<size_t>(beam_count) * 2u);
    build_node(0u, beam_count);
    return true;
  }

  void clear() {
    _beams.clear();
    _indices.clear();
    _nodes.clear();
    _maximum_radius = 0.0f;
  }

  template <typename Visitor>
  bool query_point(const float3& point, Visitor&& visitor) const {
    if (_nodes.empty() || (std::isfinite(point.x) == false) || (std::isfinite(point.y) == false) || (std::isfinite(point.z) == false)) {
      return false;
    }
    const float3 extent = {_maximum_radius, _maximum_radius, _maximum_radius};
    const Bounds query_bounds = {point - extent, point + extent};
    return traverse(
      [&query_bounds](const Bounds& bounds) {
        return overlaps(bounds, query_bounds);
      },
      visitor);
  }

  template <typename Visitor>
  bool query_beam(const UPBPBeamReference& beam, const float radius, Visitor&& visitor) const {
    if (_nodes.empty() || (beam_valid(beam) == false) || (radius <= 0.0f) || (radius > _maximum_radius)) {
      return false;
    }
    const Bounds query_bounds = beam_bounds(beam, radius);
    return traverse(
      [&query_bounds](const Bounds& bounds) {
        return overlaps(bounds, query_bounds);
      },
      visitor);
  }

  uint32_t size() const {
    return static_cast<uint32_t>(_beams.size());
  }

  uint64_t storage_bytes() const {
    return static_cast<uint64_t>(_beams.capacity()) * sizeof(UPBPBeamReference) + static_cast<uint64_t>(_indices.capacity()) * sizeof(uint32_t) +
           static_cast<uint64_t>(_nodes.capacity()) * sizeof(Node);
  }

  bool projected_storage_bytes(const uint32_t beam_count, uint64_t& result) const {
    const uint64_t beam_capacity = max(static_cast<uint64_t>(_beams.capacity()), static_cast<uint64_t>(beam_count));
    const uint64_t index_capacity = max(static_cast<uint64_t>(_indices.capacity()), static_cast<uint64_t>(beam_count));
    const uint64_t node_count = static_cast<uint64_t>(beam_count) * 2u;
    const uint64_t node_capacity = max(static_cast<uint64_t>(_nodes.capacity()), node_count);
    if ((beam_capacity > std::numeric_limits<uint64_t>::max() / sizeof(UPBPBeamReference)) || (index_capacity > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t)) ||
        (node_capacity > std::numeric_limits<uint64_t>::max() / sizeof(Node))) {
      return false;
    }
    const uint64_t beam_bytes = beam_capacity * sizeof(UPBPBeamReference);
    const uint64_t index_bytes = index_capacity * sizeof(uint32_t);
    const uint64_t node_bytes = node_capacity * sizeof(Node);
    if ((beam_bytes > std::numeric_limits<uint64_t>::max() - index_bytes) || ((beam_bytes + index_bytes) > std::numeric_limits<uint64_t>::max() - node_bytes)) {
      return false;
    }
    result = beam_bytes + index_bytes + node_bytes;
    return true;
  }

 private:
  static bool beam_valid(const UPBPBeamReference& beam) {
    const float direction_length_squared = dot(beam.direction, beam.direction);
    return (beam.length > 0.0f) && std::isfinite(beam.length) && std::isfinite(beam.origin.x) && std::isfinite(beam.origin.y) && std::isfinite(beam.origin.z) &&
           std::isfinite(direction_length_squared) && (fabsf(direction_length_squared - 1.0f) <= 1.0e-4f);
  }

  static Bounds beam_bounds(const UPBPBeamReference& beam, const float radius) {
    const float3 end = beam.origin + beam.direction * beam.length;
    return {
      min(beam.origin, end) - float3{radius, radius, radius},
      max(beam.origin, end) + float3{radius, radius, radius},
    };
  }

  static void extend(Bounds& target, const Bounds& source) {
    target.minimum = min(target.minimum, source.minimum);
    target.maximum = max(target.maximum, source.maximum);
  }

  static bool overlaps(const Bounds& first, const Bounds& second) {
    return (first.minimum.x <= second.maximum.x) && (first.maximum.x >= second.minimum.x) && (first.minimum.y <= second.maximum.y) && (first.maximum.y >= second.minimum.y) &&
           (first.minimum.z <= second.maximum.z) && (first.maximum.z >= second.minimum.z);
  }

  float centroid_component(const uint32_t beam_index, const uint32_t axis) const {
    const UPBPBeamReference& beam = _beams[beam_index];
    const float3 centroid = beam.origin + beam.direction * (0.5f * beam.length);
    return axis == 0u ? centroid.x : (axis == 1u ? centroid.y : centroid.z);
  }

  uint32_t build_node(const uint32_t first, const uint32_t count) {
    const uint32_t node_index = static_cast<uint32_t>(_nodes.size());
    _nodes.emplace_back();
    Bounds bounds = {};
    Bounds centroid_bounds = {};
    for (uint32_t offset = 0u; offset < count; ++offset) {
      const UPBPBeamReference& beam = _beams[_indices[first + offset]];
      extend(bounds, beam_bounds(beam, 0.0f));
      const float3 centroid = beam.origin + beam.direction * (0.5f * beam.length);
      extend(centroid_bounds, {centroid, centroid});
    }

    _nodes[node_index].bounds = bounds;
    _nodes[node_index].first = first;
    _nodes[node_index].count = count;
    if (count <= 8u) {
      return node_index;
    }

    const float3 extent = centroid_bounds.maximum - centroid_bounds.minimum;
    const uint32_t axis = (extent.x >= extent.y) && (extent.x >= extent.z) ? 0u : ((extent.y >= extent.z) ? 1u : 2u);
    const uint32_t middle = first + count / 2u;
    std::nth_element(_indices.begin() + first, _indices.begin() + middle, _indices.begin() + first + count, [this, axis](const uint32_t first_index, const uint32_t second_index) {
      const float first_centroid = centroid_component(first_index, axis);
      const float second_centroid = centroid_component(second_index, axis);
      return first_centroid == second_centroid ? first_index < second_index : first_centroid < second_centroid;
    });
    const uint32_t left = build_node(first, middle - first);
    const uint32_t right = build_node(middle, first + count - middle);
    _nodes[node_index].left = left;
    _nodes[node_index].right = right;
    _nodes[node_index].count = 0u;
    return node_index;
  }

  template <typename Predicate, typename Visitor>
  bool traverse(Predicate&& predicate, Visitor&& visitor) const {
    std::array<uint32_t, 128u> stack = {};
    uint32_t stack_size = 1u;
    stack[0u] = 0u;
    while (stack_size > 0u) {
      const Node& node = _nodes[stack[--stack_size]];
      if (predicate(node.bounds) == false) {
        continue;
      }

      if (node.count > 0u) {
        for (uint32_t offset = 0u; offset < node.count; ++offset) {
          visitor(_beams[_indices[node.first + offset]]);
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

  std::vector<UPBPBeamReference> _beams = {};
  std::vector<uint32_t> _indices = {};
  std::vector<Node> _nodes = {};
  float _maximum_radius = 0.0f;
};

}  // namespace etx
