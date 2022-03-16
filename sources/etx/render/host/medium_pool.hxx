#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/medium.hxx>

#include <string>

namespace etx {

struct MediumPool {
  MediumPool();
  ~MediumPool();

  void init(uint32_t capacity);
  void cleanup();

  uint32_t add(Medium::Class cls, const std::string&, const char* volume, const SpectralDistribution& s_a, const SpectralDistribution& s_o, float g, const Spectrums*);

  uint32_t find(const char* id);

  void remove(uint32_t handle);
  void remove_all();

  Medium& get(uint32_t);
  const Medium& get(uint32_t) const;

  Medium* as_array();
  uint64_t array_size();

  ETX_DECLARE_PIMPL(MediumPool, 256);
};

}  // namespace etx
