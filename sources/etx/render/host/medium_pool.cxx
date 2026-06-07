#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>

#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/buffer_pool.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/shared/medium.hxx>
#include <etx/render/shared/math.hxx>

#include <nanovdb/util/IO.h>
namespace etx {

struct MediumPoolImpl {
  MediumPoolImpl(std::vector<Medium>& external_mediums, BufferPool& external_buffer_pool, ImagePool& external_image_pool)
    : mediums(external_mediums)
    , buffer_pool(external_buffer_pool)
    , image_pool(external_image_pool) {
  }

  void init(uint32_t capacity) {
    mediums.reserve(capacity);
    mapping.reserve(capacity);
  }

  void cleanup() {
    remove_all();
    mapping.clear();
  }

  uint32_t add(Medium::Class cls, const std::string& id, const char* volume_file, uint32_t absorption_index, uint32_t scattering_index, float g, bool explicit_connections) {
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
    medium.phase_function_g = g;
    medium.enable_explicit_connections = explicit_connections;
    medium.bounds = BoundingBox{{-1.0f, -1.0f, -1.0f}, 0.0f, {1.0f, 1.0f, 1.0f}, 0.0f};

    if ((volume_file != nullptr) && (strlen(volume_file) > 0)) {
      float max_density = 0.0f;
      uint3 dimensions = {};
      auto density = load_density_grid(volume_file, dimensions);
      for (auto f : density) {
        max_density = max(max_density, f);
      }
      if (max_density > 0.0f) {
        for (auto& f : density) {
          f /= max_density;
        }
        medium.set_grid_type(DensityGrid::Type::Texture3D);
        medium.grid.density_image_index = image_pool.add_from_data_3d_r32(density.data(), dimensions, 0u, float3{0.0f, 0.0f, 0.0f}, float3{1.0f, 1.0f, 1.0f});
        const Image& density_image = image_pool.get(medium.grid.density_image_index);
        medium.density_view = density_image.pixels.r32;
        ETX_CRITICAL(medium.density_view.a != nullptr);
        medium.grid.dimensions = dimensions;
        medium.cls = Medium::Heterogeneous;
      } else {
        medium.cls = Medium::Homogeneous;
      }
    } else {
      medium.set_grid_type(DensityGrid::Type::Texture3D);
    }

    mapping[id] = handle;
    return handle;
  }

  Medium& get_mutable(uint32_t handle) {
    ETX_CRITICAL(handle < mediums.size());
    return mediums[handle];
  }

  std::string rename(uint32_t index, const std::string& desired) {
    if (index >= mediums.size()) {
      return {};
    }

    auto current = mapping.end();
    for (auto it = mapping.begin(); it != mapping.end(); ++it) {
      if (it->second == index) {
        current = it;
        break;
      }
    }
    if (current == mapping.end()) {
      return {};
    }

    auto strip_prefix = [](const std::string& s) -> std::string {
      if (s.starts_with("etx::")) {
        return s.substr(5);
      }
      if (s.starts_with("et::")) {
        return s.substr(4);
      }
      return s;
    };

    std::string base = strip_prefix(desired);
    if (base.empty()) {
      base = current->first;
    }
    if (base.empty()) {
      base = "medium-" + std::to_string(index);
    }

    std::string final = base;
    uint32_t suffix = 1;
    while (true) {
      auto found = mapping.find(final);
      if ((found == mapping.end()) || (found->second == index)) {
        break;
      }
      final = base + "#" + std::to_string(suffix++);
    }

    if (final != current->first) {
      mapping.erase(current);
      mapping.emplace(final, index);
    }
    return final;
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
    m.density_view = {};
    m.density_data = {};
    m.density_buffer = {};
    m.grid.density_image_index = kInvalidIndex;
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
    const auto& box_min = grid_bbox.min();
    const auto& box_max = grid_bbox.max();
    auto dim = box_max - box_min;
    d.x = static_cast<uint32_t>(dim.x());
    d.y = static_cast<uint32_t>(dim.y());
    d.z = static_cast<uint32_t>(dim.z());
    uint32_t dmax = max(d.x, max(d.y, d.z));
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
    for (c.z() = box_min.z(); c.z() < box_max.z(); ++c.z()) {
      for (c.y() = box_min.y(); c.y() < box_max.y(); ++c.y()) {
        for (c.x() = box_min.x(); c.x() < box_max.x(); ++c.x()) {
          float val = accessor.getValue(c);
          if (val > 0.0f) {
            min_val = min(min_val, val);
            max_val = max(max_val, val);
            nanovdb::Coord cr = c - box_min;
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

  std::vector<Medium>& mediums;
  BufferPool& buffer_pool;
  ImagePool& image_pool;
  MediumPool::Mapping mapping;
};

MediumPool::MediumPool(std::vector<Medium>& external_mediums, BufferPool& buffer_pool, ImagePool& image_pool) {
  ETX_PIMPL_CREATE(MediumPool, Impl, external_mediums, buffer_pool, image_pool);
}

MediumPool::~MediumPool() {
  ETX_PIMPL_DESTROY(MediumPool, Impl);
}

ETX_PIMPL_IMPLEMENT(MediumPool, Impl);

void MediumPool::init(uint32_t capacity) {
  _private->init(capacity);
}

void MediumPool::cleanup() {
  _private->cleanup();
}

uint32_t MediumPool::add(Medium::Class cls, const std::string& id, const char* volume, uint32_t absorption_index, uint32_t scattering_index, float g, bool explicit_connections) {
  return _private->add(cls, id, volume, absorption_index, scattering_index, g, explicit_connections);
}

uint32_t MediumPool::add_noise(Medium::Class cls, const std::string& id, NoiseFunction noise_type, uint32_t absorption_index, uint32_t scattering_index, float anisotropy,
  bool explicit_connections, float noise_scale, uint32_t noise_octaves, float noise_lacunarity, float noise_persistence, uint32_t noise_seed, float noise_power,
  const float3& noise_offset) {
  auto existing = _private->mapping.find(id);
  if (existing != _private->mapping.end()) {
    return existing->second;
  }

  uint32_t handle = static_cast<uint32_t>(_private->mediums.size());
  _private->mediums.emplace_back();

  Medium& medium = _private->mediums[handle];
  medium.cls = cls;
  medium.absorption_index = absorption_index;
  medium.scattering_index = scattering_index;
  medium.phase_function_g = anisotropy;
  medium.enable_explicit_connections = explicit_connections;
  medium.bounds = BoundingBox{{-1.0f, -1.0f, -1.0f}, 0.0f, {1.0f, 1.0f, 1.0f}, 0.0f};

  medium.set_grid_type(DensityGrid::Type::NoiseFunction);
  medium.set_noise_type(noise_type);
  medium.grid.noise_scale = noise_scale;
  medium.grid.noise_octaves = noise_octaves;
  medium.grid.noise_lacunarity = noise_lacunarity;
  medium.grid.noise_persistence = noise_persistence;
  medium.grid.noise_seed = noise_seed;
  medium.grid.noise_power = noise_power;
  medium.grid.noise_sharpness = 1.0f;
  medium.grid.noise_offset = noise_offset;
  medium.grid.noise_enable_border_fade = 0u;
  medium.grid.noise_border_fade_distance = 0.1f;
  medium.cls = Medium::Heterogeneous;

  _private->mapping[id] = handle;
  return handle;
}

Medium& MediumPool::get(uint32_t handle) {
  return _private->get_mutable(handle);
}

const Medium& MediumPool::get(uint32_t handle) const {
  return _private->get(handle);
}

std::string MediumPool::rename(uint32_t index, const std::string& desired_name) {
  return _private->rename(index, desired_name);
}

void MediumPool::remove_all() {
  _private->remove_all();
}

const Medium* MediumPool::as_array() const {
  return _private->mediums.empty() ? nullptr : _private->mediums.data();
}

const uint64_t MediumPool::array_size() const {
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
