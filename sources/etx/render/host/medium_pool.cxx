#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/pool.hxx>

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

  uint32_t add_homogenous(const char* id, const SpectralDistribution& s_a, const SpectralDistribution& s_o, float g) {
    auto i = _mapping.find(id);
    if (i != _mapping.end()) {
      return i->second;
    }

    auto handle = medium_pool.alloc();
    auto& medium = medium_pool.get(handle);

    if (medium.s_absorption.is_zero() && medium.s_outscattering.is_zero()) {
      medium.cls = Medium::Class::Vacuum;
    } else {
      medium.cls = Medium::Class::Homogeneous;
      medium.s_absorption = s_a;
      medium.s_outscattering = s_o;
      medium.phase_function_g = g;
    }

    return handle;
  }

  uint32_t add_heterogenous(const char* id, const char* volume_file, const SpectralDistribution& s_a, const SpectralDistribution& s_o, float g) {
    auto i = _mapping.find(id);
    if (i != _mapping.end()) {
      return i->second;
    }

    auto handle = medium_pool.alloc();
    auto& medium = medium_pool.get(handle);

    if (s_a.is_zero() && s_o.is_zero()) {
      medium.cls = Medium::Class::Vacuum;
    } else {
      medium.cls = Medium::Class::Heterogeneous;
      medium.s_absorption = s_a;
      medium.s_outscattering = s_o;
      medium.phase_function_g = g;
      medium.max_sigma = medium.s_outscattering.maximum_power() + medium.s_absorption.maximum_power();

      auto density = load_density_grid(volume_file, medium.dimensions.x, medium.dimensions.y, medium.dimensions.z, medium.max_density);
      medium.density.count = density.size();
      medium.density.a = reinterpret_cast<float*>(malloc(medium.density.count * sizeof(float)));
    }
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
  }

  void remove_all() {
    medium_pool.free_all(std::bind(&MediumPoolImpl::free_medium, this, std::placeholders::_1));
  }

  void free_medium(Medium& m) {
    if (m.density.count > 0) {
      free(m.density.a);
    }
    m = {};
  }

  std::vector<float> load_density_grid(const char* file_name, uint32_t& dx, uint32_t& dy, uint32_t& dz, float& max_density) {
    // TODO : check validity
    char buffer[2048] = {};
    int last_char = snprintf(buffer, sizeof(buffer), "%s", file_name);
    while ((last_char > 0) && (buffer[--last_char] != '.')) {
    }
    auto ext = buffer + last_char;

    std::vector<float> density;

#if (ETX_HAVE_OPENVDB)
    if (strcmp(ext, ".vdb") == 0) {
      load_vdb(file_name, density, dx, dy, dz);
      snprintf(ext, sizeof(buffer) - last_char, "%s", ".et-vdb");
      save_raw(buffer, density, dx, dy, dz);
    } else
#endif
      if (strcmp(ext, ".et-vdb") == 0) {
      load_raw(file_name, density, dx, dy, dz);
    }

    max_density = 0.0f;
    for (auto f : density) {
      max_density = max(max_density, f);
    }

    return density;
  }

  void load_raw(const char* file_name, std::vector<float>& density, uint32_t& dx, uint32_t& dy, uint32_t& dz) {
    auto fin = fopen(file_name, "rb");
    if (fin == nullptr) {
      return;
    }
    uint32_t d[3] = {};
    fread(d, sizeof(d), 1, fin);
    dx = d[0];
    dy = d[1];
    dz = d[2];

    density.resize(1llu * dx * dy * dz);
    fread(density.data(), sizeof(float), 1llu * dx * dy * dz, fin);
    fclose(fin);
  }

  ObjectIndexPool<Medium> medium_pool;
  std::unordered_map<const char*, uint32_t> _mapping;
};

ETX_PIMPL_IMPLEMENT_ALL(MediumPool, Impl);

void MediumPool::init(uint32_t capacity) {
  _private->init(capacity);
}

void MediumPool::cleanup() {
  _private->cleanup();
}

uint32_t MediumPool::add_homogenous(const char* id, const SpectralDistribution& s_a, const SpectralDistribution& s_o, float g) {
  return _private->add_homogenous(id, s_a, s_o, g);
}

uint32_t MediumPool::add_heterogenous(const char* id, const char* volume, const SpectralDistribution& s_a, const SpectralDistribution& s_o, float g) {
  return _private->add_heterogenous(id, volume, s_a, s_o, g);
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
  return _private->medium_pool.data();
}

uint64_t MediumPool::array_size() {
  return 1llu + _private->medium_pool.latest_alive_index();
}

}  // namespace etx
