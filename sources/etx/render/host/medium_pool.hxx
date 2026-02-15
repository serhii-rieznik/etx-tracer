#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/medium.hxx>
namespace etx {

struct BufferPool;

struct MediumPool {
  using Mapping = std::unordered_map<std::string, uint32_t>;

  MediumPool(std::vector<Medium>&, BufferPool&);
  ~MediumPool();

  void init(uint32_t capacity);
  void cleanup();

  uint32_t add(Medium::Class cls, const std::string&, const char* volume, uint32_t absorption_index, uint32_t scattering_index, float anisotropy, bool explicit_connections);

  uint32_t add_noise(Medium::Class cls, const std::string&, NoiseFunction noise_type, uint32_t absorption_index, uint32_t scattering_index, float anisotropy,
    bool explicit_connections, float noise_scale, uint32_t noise_octaves, float noise_lacunarity, float noise_persistence, uint32_t noise_seed, float noise_power,
    const float3& noise_offset);

  uint32_t find(const char* id);

  void remove_all();

  Medium& get(uint32_t);
  const Medium& get(uint32_t) const;

  const Mapping& mapping() const;

  const Medium* as_array() const;
  const uint64_t array_size() const;

  std::string rename(uint32_t index, const std::string& desired_name);

  ETX_DECLARE_PIMPL(MediumPool, 256);
};

}  // namespace etx
