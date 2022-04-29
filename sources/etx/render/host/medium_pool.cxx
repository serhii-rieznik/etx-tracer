#include <etx/core/environment.hxx>
#include <etx/log/log.hxx>

#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/pool.hxx>

#if (ETX_HAVE_OPENVDB)
#include <openvdb/openvdb.h>
#endif

#include <unordered_map>
#include <functional>

namespace etx {

struct MediumPoolImpl {
  void init(uint32_t capacity) {
    medium_pool.init(capacity);
  }

  void cleanup() {
    ETX_ASSERT(medium_pool.count_alive() == 0);
    medium_pool.cleanup();
  }

  uint32_t add(Medium::Class cls, const std::string& id, const char* volume_file, const SpectralDistribution& s_a, const SpectralDistribution& s_o, float g,
    const Pointer<Spectrums> s) {
    auto i = mapping.find(id);
    if (i != mapping.end()) {
      return i->second;
    }

    auto handle = medium_pool.alloc();

    auto& medium = medium_pool.get(handle);
    medium.cls = cls;
    medium.s_absorption = s_a;
    medium.s_outscattering = s_o;
    medium.max_sigma = s_a.maximum_power() + s_o.maximum_power();
    medium.phase_function_g = g;

    if (strlen(volume_file) > 0) {
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

    if (s_a.is_zero() && s_o.is_zero()) {
      medium.cls = Medium::Class::Vacuum;
    }

    mapping[id] = handle;
    return handle;
  }

  Medium& get(uint32_t handle) {
    return medium_pool.get(handle);
  }

  void remove(uint32_t handle) {
    if (handle == kInvalidIndex) {
      return;
    }

    free_medium(medium_pool.get(handle));
    medium_pool.free(handle);

    for (auto i = mapping.begin(), e = mapping.end(); i != e; ++i) {
      if (i->second == handle) {
        mapping.erase(i);
        break;
      }
    }
  }

  void remove_all() {
    medium_pool.free_all(std::bind(&MediumPoolImpl::free_medium, this, std::placeholders::_1));
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
    if (_stricmp(ext, ".vdb") == 0) {
      load_vdb(file_name, density, d);
    } else {
      log::error("Only VDB volumetric data format is supported at the moment");
    }

    return density;
  }

  void load_vdb(const char* file_name, std::vector<float>& density, uint3& d) {
    d = {};
    density.clear();
#if (ETX_HAVE_OPENVDB)
    static bool openvdb_initialized = false;

    if (openvdb_initialized == false) {
      openvdb::initialize();
      openvdb_initialized = true;
    }

    openvdb::io::File in_file(file_name);
    if (in_file.open() == false) {
      return;
    }

    auto grids = in_file.getGrids();

    for (const auto& base_grid : *grids) {
      openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base_grid);
      if (grid) {
        auto grid_bbox = grid->evalActiveVoxelBoundingBox();
        auto grid_dim = grid->evalActiveVoxelDim().asVec3i();

        d.x = grid_dim.x();
        d.y = grid_dim.y();
        d.z = grid_dim.z();
        density.resize(1llu * d.x * d.y * d.z, 0.0f);
        log::info("Loaded VDB grid dimensions: %u x %u x %u", d.x, d.y, d.z);

        for (auto i = grid->beginValueOn(); i; ++i) {
          auto pos = (i.getCoord() - grid_bbox.getStart()).asVec3i();
          density[pos.x() + 1llu * pos.y() * d.x + 1llu * pos.z() * d.x * d.y] = i.getValue();
        }
        break;
      }
    }
#else
    log::error("Loading from VDB is disabled. Generate project using CMake with `-DWITH_OPENVDB=1` option to enable support.");
#endif
  }

  ObjectIndexPool<Medium> medium_pool;
  std::unordered_map<std::string, uint32_t> mapping;
};

ETX_PIMPL_IMPLEMENT_ALL(MediumPool, Impl);

void MediumPool::init(uint32_t capacity) {
  _private->init(capacity);
}

void MediumPool::cleanup() {
  _private->cleanup();
}

uint32_t MediumPool::add(Medium::Class cls, const std::string& id, const char* volume, const SpectralDistribution& s_a, const SpectralDistribution& s_o, float g,
  const Pointer<Spectrums> s) {
  return _private->add(cls, id, volume, s_a, s_o, g, s);
}

Medium& MediumPool::get(uint32_t handle) {
  return _private->get(handle);
}

const Medium& MediumPool::get(uint32_t handle) const {
  return _private->get(handle);
}

void MediumPool::remove(uint32_t handle) {
  _private->remove(handle);
}

void MediumPool::remove_all() {
  _private->remove_all();
}

Medium* MediumPool::as_array() {
  return _private->medium_pool.alive_objects_count() > 0 ? _private->medium_pool.data() : nullptr;
}

uint64_t MediumPool::array_size() {
  return _private->medium_pool.alive_objects_count() > 0 ? 1llu + _private->medium_pool.latest_alive_index() : 0;
}

uint32_t MediumPool::find(const char* id) {
  auto i = _private->mapping.find(id);
  return (i == _private->mapping.end()) ? kInvalidIndex : i->second;
}

}  // namespace etx
