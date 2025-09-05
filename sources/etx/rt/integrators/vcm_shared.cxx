#include <etx/rt/integrators/vcm_spatial_grid.hxx>
#include <etx/util/options.hxx>

namespace etx {

VCMOptions VCMOptions::default_values() {
  VCMOptions options = {};
  options.options = DefaultOptions;
  options.radius_decay = 256u;
  options.initial_radius = 0.0f;
  options.kernel = VCMOptions::Epanechnikov;
  return options;
}

void VCMOptions::load(const Options& opt) {
  initial_radius = opt.get("vcm-initial_radius", initial_radius).to_float();
  radius_decay = opt.get("vcm-radius_decay", radius_decay).to_integer();
  blue_noise = opt.get("vcm-blue_noise", blue_noise).to_bool();
  kernel = opt.get("vcm-kernel", kernel).to_integer();

  options = opt.get("vcm-direct_hit", direct_hit()).to_bool() ? (options | DirectHit) : (options & ~DirectHit);
  options = opt.get("vcm-connect_to_light", connect_to_light()).to_bool() ? (options | ConnectToLight) : (options & ~ConnectToLight);
  options = opt.get("vcm-connect_to_camera", connect_to_camera()).to_bool() ? (options | ConnectToCamera) : (options & ~ConnectToCamera);
  options = opt.get("vcm-connect_vertices", connect_vertices()).to_bool() ? (options | ConnectVertices) : (options & ~ConnectVertices);
  options = opt.get("vcm-merge_vertices", merge_vertices()).to_bool() ? (options | MergeVertices) : (options & ~MergeVertices);
  options = opt.get("vcm-mis", enable_mis()).to_bool() ? (options | EnableMis) : (options & ~EnableMis);
  options = opt.get("vcm-merging", enable_merging()).to_bool() ? (options | EnableMerging) : (options & ~EnableMerging);
}

void VCMOptions::store(Options& opt) const {
  opt.add("compute", "Connections:");
  opt.add(direct_hit(), "vcm-direct_hit", "Direct Hits");
  opt.add(connect_to_camera(), "vcm-connect_to_camera", "Light Path to Camera");
  opt.add(connect_to_light(), "vcm-connect_to_light", "Camera Path to Light");
  opt.add(connect_vertices(), "vcm-connect_vertices", "Camera Path to Light Path");
  if (enable_merging()) {
    opt.add(merge_vertices(), "vcm-merge_vertices", "Merge Light Vertices");
  }
  opt.add("vcm-opt", "VCM Options");
  opt.add(enable_merging(), "vcm-merging", "Enable Merging");
  opt.add(enable_mis(), "vcm-mis", "Multiple Importance Sampling");
  opt.add(blue_noise, "vcm-blue_noise", "Blue Noise");
  opt.add(smooth_kernel(), "vcm-kernel", "Smooth Merging Kernel");
  opt.add(0.0f, initial_radius, 10.0f, "vcm-initial_radius", "Initial Radius");
  opt.add(1u, uint32_t(radius_decay), 65536u, "vcm-radius_decay", "Radius Decay");
}

void VCMSpatialGrid::construct(const Scene& scene, const VCMLightVertex* samples, uint64_t sample_count, float radius, TaskScheduler& scheduler) {
  data = {};
  if (sample_count == 0) {
    return;
  }

  TimeMeasure time_measure = {};

  data.radius_squared = radius * radius;
  if (data.radius_squared > 0.0f) {
    data.inv_radius_squared = 1.0f / data.radius_squared;
  } else {
    data.inv_radius_squared = 0.0f;
  }
  data.cell_size = 2.0f * radius;
  data.bounding_box = {{kMaxFloat, kMaxFloat, kMaxFloat}, {-kMaxFloat, -kMaxFloat, -kMaxFloat}};

  std::vector<BoundingBox> thread_boxes(scheduler.max_thread_count(), {{kMaxFloat, kMaxFloat, kMaxFloat}, {-kMaxFloat, -kMaxFloat, -kMaxFloat}});
  scheduler.execute(uint32_t(sample_count), [&scene, &thread_boxes, samples](uint32_t begin, uint32_t end, uint32_t thread_id) {
    for (uint32_t i = begin; i < end; ++i) {
      const auto& p = samples[i];
      if (p.is_medium) {
        continue;
      }
      thread_boxes[thread_id].p_min = min(thread_boxes[thread_id].p_min, p.pos);
      thread_boxes[thread_id].p_max = max(thread_boxes[thread_id].p_max, p.pos);
    }
  });
  for (const auto& bbox : thread_boxes) {
    data.bounding_box.p_min = min(data.bounding_box.p_min, bbox.p_min);
    data.bounding_box.p_max = max(data.bounding_box.p_max, bbox.p_max);
  }

  uint32_t hash_table_size = static_cast<uint32_t>(next_power_of_two(sample_count));
  data.hash_table_mask = hash_table_size - 1u;

  _positions.clear();
  _normals.clear();
  _w_in.clear();
  _d_vcm.clear();
  _d_vm.clear();
  _path_lengths.clear();
  _throughput_rgb_div_pdf.clear();
  _cell_ends.resize(hash_table_size);
  memset(_cell_ends.data(), 0, sizeof(uint32_t) * hash_table_size);

  static_assert(sizeof(std::atomic_int) == sizeof(uint32_t));

  auto ptr = reinterpret_cast<int32_t*>(_cell_ends.data());
  scheduler.execute(uint32_t(sample_count), [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    for (uint32_t i = begin; i < end; ++i) {
      const auto& s = samples[i];
      if (s.is_medium) {
        continue;
      }
      uint32_t index = data.position_to_index(s.pos);
      atomic_inc(ptr + index);
    }
  });

  uint32_t sum = 0;
  for (auto& cell_end : _cell_ends) {
    uint32_t t = cell_end;
    cell_end = sum;
    sum += t;
  }

  uint32_t total = _cell_ends.back();
  _positions.resize(total);
  _normals.resize(total);
  _w_in.resize(total);
  _d_vcm.resize(total);
  _d_vm.resize(total);
  _path_lengths.resize(total);
  _throughput_rgb_div_pdf.resize(total);

  ptr = reinterpret_cast<int32_t*>(_cell_ends.data());
  scheduler.execute(uint32_t(sample_count), [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    for (uint32_t i = begin; i < end; ++i) {
      const auto& s = samples[i];
      if (s.is_medium) {
        continue;
      }
      uint32_t cell = data.position_to_index(s.pos);
      uint32_t dst = atomic_inc(ptr + cell) - 1u;
      _positions[dst] = s.pos;
      _normals[dst] = s.nrm;
      _w_in[dst] = s.w_i;
      _d_vcm[dst] = s.d_vcm;
      _d_vm[dst] = s.d_vm;
      _path_lengths[dst] = s.path_length;
      _throughput_rgb_div_pdf[dst] = (s.throughput / s.throughput.sampling_pdf()).to_rgb();
    }
  });

  data.cell_ends = make_array_view<uint32_t>(_cell_ends.data(), _cell_ends.size());
  data.positions = make_array_view<float3>(_positions.data(), _positions.size());
  data.normals = make_array_view<float3>(_normals.data(), _normals.size());
  data.w_in = make_array_view<float3>(_w_in.data(), _w_in.size());
  data.d_vcm = make_array_view<float>(_d_vcm.data(), _d_vcm.size());
  data.d_vm = make_array_view<float>(_d_vm.data(), _d_vm.size());
  data.path_lengths = make_array_view<uint32_t>(_path_lengths.data(), _path_lengths.size());
  data.throughput_rgb_div_pdf = make_array_view<float3>(_throughput_rgb_div_pdf.data(), _throughput_rgb_div_pdf.size());
}

}  // namespace etx
