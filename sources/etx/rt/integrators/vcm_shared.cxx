#include <etx/rt/integrators/vcm_spatial_grid.hxx>

namespace etx {

VCMOptions VCMOptions::default_values() {
  VCMOptions options = {};
  options.options = DefaultOptions;
  options.max_samples = 128u;
  options.max_depth = 65536u;
  options.rr_start = 6u;
  options.radius_decay = 256u;
  options.initial_radius = 0.0f;
  return options;
}

void VCMOptions::load(const Options& opt) {
  max_samples = opt.get("spp", max_samples).to_integer();
  max_depth = opt.get("max_depth", max_depth).to_integer();
  initial_radius = opt.get("initial_radius", initial_radius).to_float();
  radius_decay = opt.get("radius_decay", radius_decay).to_integer();
  rr_start = opt.get("rr_start", max_depth).to_integer();

  options = opt.get("direct_hit", direct_hit()).to_bool() ? (options | DirectHit) : (options & ~DirectHit);
  options = opt.get("connect_to_light", connect_to_light()).to_bool() ? (options | ConnectToLight) : (options & ~ConnectToLight);
  options = opt.get("connect_to_camera", connect_to_camera()).to_bool() ? (options | ConnectToCamera) : (options & ~ConnectToCamera);
  options = opt.get("connect_vertices", connect_vertices()).to_bool() ? (options | ConnectVertices) : (options & ~ConnectVertices);
  options = opt.get("merge_vertices", merge_vertices()).to_bool() ? (options | MergeVertices) : (options & ~MergeVertices);
  options = opt.get("mis", enable_mis()).to_bool() ? (options | EnableMis) : (options & ~EnableMis);
  options = opt.get("merging", enable_merging()).to_bool() ? (options | EnableMerging) : (options & ~EnableMerging);
}

void VCMOptions::store(Options& opt) {
  opt.add(1u, max_samples, 0xffffu, "spp", "Max Iterations");
  opt.add(1u, max_depth, 65536u, "max_depth", "Max Path Length");
  opt.add(1u, rr_start, 65536u, "rr_start", "RR Start Length");
  opt.add(0.0f, initial_radius, 10.0f, "initial_radius", "Initial Radius");
  opt.add(1u, uint32_t(radius_decay), 65536u, "radius_decay", "Radius Decay");
  opt.add("debug", "Compute:");
  opt.add(direct_hit(), "direct_hit", "Direct Hits");
  opt.add(connect_to_light(), "connect_to_light", "Connect to Lights");
  opt.add(connect_to_camera(), "connect_to_camera", "Connect to Camera");
  opt.add(connect_vertices(), "connect_vertices", "Connect Vertices");
  opt.add(merge_vertices(), "merge_vertices", "Merge Vertices");
  opt.add(enable_mis(), "mis", "Multiple Importance Sampling");
  opt.add(enable_merging(), "merging", "Enable Merging");
}

void VCMSpatialGrid::construct(const Scene& scene, const VCMLightVertex* samples, uint64_t sample_count, float radius, TaskScheduler& scheduler) {

  data = {};
  if (sample_count == 0) {
    return;
  }

  TimeMeasure time_measure = {};

  data.radius_squared = radius * radius;
  data.cell_size = 2.0f * radius;
  data.bounding_box = {{kMaxFloat, kMaxFloat, kMaxFloat}, {-kMaxFloat, -kMaxFloat, -kMaxFloat}};

  std::vector<BoundingBox> thread_boxes(scheduler.max_thread_count(), {{kMaxFloat, kMaxFloat, kMaxFloat}, {-kMaxFloat, -kMaxFloat, -kMaxFloat}});
  scheduler.execute(uint32_t(sample_count), [&scene, &thread_boxes, samples](uint32_t begin, uint32_t end, uint32_t thread_id) {
    for (uint32_t i = begin; i < end; ++i) {
      const auto& p = samples[i];
      thread_boxes[thread_id].p_min = min(thread_boxes[thread_id].p_min, p.position(scene));
      thread_boxes[thread_id].p_max = max(thread_boxes[thread_id].p_max, p.position(scene));
    }
  });
  for (const auto& bbox : thread_boxes) {
    data.bounding_box.p_min = min(data.bounding_box.p_min, bbox.p_min);
    data.bounding_box.p_max = max(data.bounding_box.p_max, bbox.p_max);
  }

  uint32_t hash_table_size = static_cast<uint32_t>(next_power_of_two(sample_count));
  data.hash_table_mask = hash_table_size - 1u;

  _position_to_index.resize(sample_count);
  _indices.resize(sample_count);

  _cell_ends.resize(hash_table_size);
  memset(_cell_ends.data(), 0, sizeof(uint32_t) * hash_table_size);

  static_assert(sizeof(std::atomic_int) == sizeof(uint32_t));

  auto ptr = reinterpret_cast<volatile long*>(_cell_ends.data());
  // auto ptr = reinterpret_cast<std::atomic_int*>(_cell_ends.data());
  scheduler.execute(uint32_t(sample_count), [&scene, &samples, this, ptr](uint32_t begin, uint32_t end, uint32_t thread_id) {
    for (uint32_t i = begin; i < end; ++i) {
      uint32_t index = data.position_to_index(samples[i].position(scene));
      _position_to_index[i] = index;
      _InterlockedIncrement(ptr + index);
      // atomic_fetch_add_explicit(ptr + index, 1u, std::memory_order_relaxed);
    }
  });

  uint32_t sum = 0;
  for (auto& cell_end : _cell_ends) {
    uint32_t t = cell_end;
    cell_end = sum;
    sum += t;
  }

  ptr = reinterpret_cast<volatile long*>(_cell_ends.data());
  // ptr = reinterpret_cast<std::atomic_int*>(_cell_ends.data());
  scheduler.execute(uint32_t(sample_count), [this, ptr](uint32_t begin, uint32_t end, uint32_t thread_id) {
    for (uint32_t i = begin; i < end; ++i) {
      uint32_t index = _position_to_index[i];
      uint32_t target_cell = _InterlockedIncrement(ptr + index);
      // uint32_t target_cell = atomic_fetch_add_explicit(ptr + index, 1u, std::memory_order_relaxed);
      _indices[target_cell] = i;
    }
  });

  data.indices = make_array_view<uint32_t>(_indices.data(), _indices.size());
  data.cell_ends = make_array_view<uint32_t>(_cell_ends.data(), _cell_ends.size());
}

}  // namespace etx
