#include <etx/core/environment.hxx>

#include <etx/render/host/medium_pool.hxx>

#include <nanovdb/util/IO.h>

#include <algorithm>
#include <vector>

namespace etx {

struct MediumPoolImpl {
  void init(uint32_t capacity) {
    mediums.reserve(capacity);
    mapping.reserve(capacity);
  }

  void cleanup() {
    remove_all();
    mapping.clear();
  }

  uint32_t add(Medium::Class cls, const std::string& id, const char* volume_file, uint32_t absorption_index, uint32_t scattering_index, float max_sigma, float g,
    bool explicit_connections) {
    auto existing = mapping.find(id);
    if (existing != mapping.end()) {
      return existing->second;
    }

    uint32_t handle = static_cast<uint32_t>(mediums.size());
    mediums.emplace_back();

    Medium& medium = mediums[handle];
    medium.cls = cls;
    medium.absorption_index = absorption_index;
    medium.scattering_index = scattering_index;
    medium.max_sigma = max_sigma;
    medium.phase_function_g = g;
    medium.enable_explicit_connections = explicit_connections;

    if ((volume_file != nullptr) && (strlen(volume_file) > 0)) {
      float max_density = 0.0f;
      auto density = load_density_grid(volume_file, medium.dimensions);
      for (auto f : density) {
        max_density = max(max_density, f);
      }
      if (max_density > 0.0f) {
        for (auto& f : density) {
          f /= max_density;
        }
        medium.density.count = density.size();
        medium.density.a = reinterpret_cast<float*>(malloc(medium.density.count * sizeof(float)));
        memcpy(medium.density.a, density.data(), sizeof(float) * medium.density.count);
        medium.cls = Medium::Class::Heterogeneous;
      } else {
        medium.cls = Medium::Class::Homogeneous;
      }
    }

    mapping[id] = handle;
    return handle;
  }

  Medium& get_mutable(uint32_t handle) {
    ETX_CRITICAL(handle < mediums.size());
    return mediums[handle];
  }

  const Medium& get(uint32_t handle) const {
    ETX_CRITICAL(handle < mediums.size());
    return mediums[handle];
  }

  void remove_all() {
    for (auto& medium : mediums) {
      free_medium(medium);
    }
    mediums.clear();
    mapping.clear();
  }

  void free_medium(Medium& m) {
    if (m.density.count > 0) {
      free(m.density.a);
    }
    m = {};
  }

  std::vector<float> load_density_grid(const char* file_name, uint3& d) {
    std::vector<float> density;

    const char* ext = get_file_ext(file_name);
    if (_stricmp(ext, ".nvdb") == 0) {
      load_nvdb(file_name, density, d);
    } else {
      log::error("Only NVDB volumetric data format is supported at the moment");
    }

    return density;
  }

  void load_nvdb(const char* file_name, std::vector<float>& density, uint3& d) {
    d = {};
    density.clear();

    auto handle = nanovdb::io::readGrid(file_name);
    auto grid = handle.grid<float>(0);
    if (grid == nullptr) {
      return;
    }

    auto accessor = grid->getAccessor();
    const auto& grid_bbox = grid->indexBBox();
    const auto& min = grid_bbox.min();
    const auto& max = grid_bbox.max();
    auto dim = max - min;
    d.x = static_cast<uint32_t>(dim.x());
    d.y = static_cast<uint32_t>(dim.y());
    d.z = static_cast<uint32_t>(dim.z());
    uint32_t dmax = std::max(d.x, std::max(d.y, d.z));
    float3 fd = {float(d.x) / float(dmax), float(d.y) / float(dmax), float(d.z) / float(dmax)};

    log::info("Medium bounding box: [%d %d %d]...[%d %d %d] : [%d %d %d] (%.4f %.4f %.4f)",  //
      grid_bbox.min().x(), grid_bbox.min().y(), grid_bbox.min().z(),                         //
      grid_bbox.max().x(), grid_bbox.max().y(), grid_bbox.max().z(),                         //
      d.x, d.y, d.z, fd.x, fd.y, fd.z);

    density.resize(1llu * d.x * d.y * d.z, 0.0f);

    float min_val = kMaxFloat;
    float max_val = -kMaxFloat;
    double avg_val = 0.0f;
    uint64_t value_count = 0;
    nanovdb::Coord c = {};
    for (c.z() = min.z(); c.z() < max.z(); ++c.z()) {
      for (c.y() = min.y(); c.y() < max.y(); ++c.y()) {
        for (c.x() = min.x(); c.x() < max.x(); ++c.x()) {
          float val = accessor.getValue(c);
          if (val > 0.0f) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            nanovdb::Coord cr = c - min;
            density[cr.x() + 1llu * cr.y() * d.x + 1llu * cr.z() * d.x * d.y] = val;
            value_count += 1u;
            avg_val += val;
          }
        }
      }
    }
    avg_val /= float(value_count);

    log::info("Density values range: %.5f ... %.5f ... %.5f", min_val, avg_val, max_val);
    if ((value_count == 0) || (min_val == kMaxFloat) || ((max_val - min_val) <= kEpsilon) || (avg_val <= kEpsilon)) {
      log::warning("Density is zero or too small, clearing...");
      d = {};
      density.clear();
      density.shrink_to_fit();
    }
  }

  std::vector<Medium> mediums;
  MediumPool::Mapping mapping;
};

ETX_PIMPL_IMPLEMENT_ALL(MediumPool, Impl);

void MediumPool::init(uint32_t capacity) {
  _private->init(capacity);
}

void MediumPool::cleanup() {
  _private->cleanup();
}

uint32_t MediumPool::add(Medium::Class cls, const std::string& id, const char* volume, uint32_t absorption_index, uint32_t scattering_index, float max_sigma, float g,
  bool explicit_connections) {
  return _private->add(cls, id, volume, absorption_index, scattering_index, max_sigma, g, explicit_connections);
}

Medium& MediumPool::get(uint32_t handle) {
  return _private->get_mutable(handle);
}

const Medium& MediumPool::get(uint32_t handle) const {
  return _private->get(handle);
}

void MediumPool::remove_all() {
  _private->remove_all();
}

Medium* MediumPool::as_array() {
  return _private->mediums.empty() ? nullptr : _private->mediums.data();
}

uint64_t MediumPool::array_size() {
  return _private->mediums.size();
}

uint32_t MediumPool::find(const char* id) {
  auto i = _private->mapping.find(id);
  return (i == _private->mapping.end()) ? kInvalidIndex : i->second;
}

const MediumPool::Mapping& MediumPool::mapping() const {
  return _private->mapping;
}

}  // namespace etx
