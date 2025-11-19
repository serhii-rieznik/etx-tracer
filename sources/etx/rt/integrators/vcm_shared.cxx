#include <etx/rt/integrators/vcm_spatial_grid.hxx>
#include <etx/util/options.hxx>
#include <etx/render/shared/scene.hxx>

namespace etx {

VCMOptions VCMOptions::default_values() {
  VCMOptions options = {};
  options.options = DefaultOptions;
  options.radius_decay = 256u;
  options.initial_radius = 0.0f;
  options.kernel = VCMOptions::Epanechnikov;
  return options;
}

void VCMOptions::load(const Options& opt, const Scene& scene) {
  initial_radius = opt.get_float("vcm-initial_radius", initial_radius);
  radius_decay = opt.get_integral("vcm-radius_decay", radius_decay);
  kernel = opt.get_integral("vcm-kernel", kernel);

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
  opt.set_float("vcm-initial_radius", initial_radius, "Initial Radius", {0.0f, 10.0f});
  opt.set_integral("vcm-radius_decay", radius_decay, "Radius Decay", 0, {1u, 65536u});
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
