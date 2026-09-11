#include <etx/rt/integrators/vcm_spatial_grid.hxx>
#include <etx/engine/options.hxx>
#include <etx/render/shared/scene.hxx>

#include <atomic>

namespace etx {

VCMOptions VCMOptions::default_values() {
  VCMOptions options = {};
  options.options = DefaultOptions;
  options.initial_radius = 0.0f;
  options.kernel = VCMOptions::Epanechnikov;
  return options;
}

void VCMOptions::load(const Options& opt, const Scene& scene) {
  initial_radius = opt.get_float("vcm-initial_radius", initial_radius);
  kernel = opt.get_bool("vcm-kernel", smooth_kernel()) ? Epanechnikov : Tophat;

  blue_noise = scene.blue_noise();
  set_option(DirectHit, scene.strategy_enabled(Scene::Strategy::DirectHit));
  set_option(ConnectToLight, scene.strategy_enabled(Scene::Strategy::ConnectToLight));
  set_option(ConnectToCamera, scene.strategy_enabled(Scene::Strategy::ConnectToCamera));
  set_option(ConnectVertices, scene.strategy_enabled(Scene::Strategy::ConnectVertices));
  set_option(MergeVertices, scene.strategy_enabled(Scene::Strategy::MergeVertices));
  set_option(EnableMis, scene.multiple_importance_sampling());

  bool enable_merging_flag = opt.get_bool("vcm-merging", enable_merging());
  set_option(EnableMerging, enable_merging_flag);
}

void VCMOptions::store(Options& opt) const {
  opt.options.clear();
  opt.set_string("vcm-opt", "VCM Options", "VCM Options");
  opt.set_bool("vcm-merging", enable_merging(), "Enable Merging");
  opt.set_bool("vcm-kernel", smooth_kernel(), "Smooth Merging Kernel");
  opt.set_float("vcm-initial_radius", initial_radius, "Initial radius", {0.0f, 10.0f});
}

void VCMSpatialGrid::construct(const Scene& scene, const VCMLightVertex* samples, uint64_t sample_count, float radius, TaskScheduler& scheduler) {
  data = {};
  if (sample_count == 0) {
    return;
  }

  TimeMeasure time_measure = {};

  data.cell_size = 2.0f * radius;
  data.bounding_box = {{kMaxFloat, kMaxFloat, kMaxFloat}, 0.0f, {-kMaxFloat, -kMaxFloat, -kMaxFloat}, 0.0f};

  std::vector<BoundingBox> thread_boxes(scheduler.max_thread_count(), data.bounding_box);
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
  _d_vm_base.clear();
  _d_surface.clear();
  _path_lengths.clear();
  _throughputs.clear();
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

  const uint32_t total = sum;
  _positions.resize(total);
  _normals.resize(total);
  _w_in.resize(total);
  _d_vcm.resize(total);
  _d_vm_base.resize(total);
  _d_surface.resize(total);
  _path_lengths.resize(total);
  _throughputs.resize(total);

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
      _d_vm_base[dst] = s.d_vm_base;
      _d_surface[dst] = s.d_surface;
      _path_lengths[dst] = s.path_length;
      _throughputs[dst] = s.throughput;
    }
  });

  data.cell_ends = make_array_view<uint32_t>(_cell_ends.data(), _cell_ends.size());
  data.positions = make_array_view<float3>(_positions.data(), _positions.size());
  data.normals = make_array_view<float3>(_normals.data(), _normals.size());
  data.w_in = make_array_view<float3>(_w_in.data(), _w_in.size());
  data.d_vcm = make_array_view<float>(_d_vcm.data(), _d_vcm.size());
  data.d_vm_base = make_array_view<float>(_d_vm_base.data(), _d_vm_base.size());
  data.d_surface = make_array_view<float>(_d_surface.data(), _d_surface.size());
  data.path_lengths = make_array_view<uint32_t>(_path_lengths.data(), _path_lengths.size());
  data.throughputs = make_array_view<SpectralResponse>(_throughputs.data(), _throughputs.size());
}

}  // namespace etx
