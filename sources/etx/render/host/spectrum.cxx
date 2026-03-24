#include <etx/core/log.hxx>
#include <etx/render/shared/spectrum.hxx>
namespace etx {

SpectralDistribution SpectralDistribution::from_samples(const float2 wavelengths_power[], uint64_t count) {
  SpectralDistribution result;

  if ((wavelengths_power == nullptr) || (count == 0)) {
    result.spectral_entry_count = 0u;
    result.integrated_value = {};
    return result;
  }

  float value = wavelengths_power[0].x;
  float wavelength_scale = 1.0f;
  while ((value * wavelength_scale) < 100.0f) {
    wavelength_scale *= 10.0f;
  }

  std::vector<float2> samples;
  samples.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    float wavelength = wavelengths_power[i].x * wavelength_scale;
    float power = wavelengths_power[i].y;
    if (valid_value(power)) {
      wavelength = std::clamp(wavelength, kShortestWavelength, kLongestWavelength);
      samples.emplace_back(float2{wavelength, power});
    }
  }

  if (samples.empty()) {
    result.spectral_entry_count = 0u;
    result.integrated_value = {};
    return result;
  }

  std::sort(samples.begin(), samples.end(), [](const float2& a, const float2& b) {
    return a.x < b.x;
  });

  std::vector<float2> unique_samples;
  unique_samples.reserve(samples.size());
  for (const auto& s : samples) {
    if (unique_samples.empty() || fabsf(s.x - unique_samples.back().x) > 1.0e-4f) {
      unique_samples.emplace_back(s);
    } else {
      unique_samples.back().y = s.y;
    }
  }

  if (unique_samples.size() == 1u) {
    unique_samples.emplace_back(float2{unique_samples.front().x, unique_samples.front().y});
  }

  result.spectral_entry_count = WavelengthCount;

  size_t segment = 0u;
  for (uint32_t i = 0; i < WavelengthCount; ++i) {
    float wavelength = float(ShortestWavelength + i);

    float power = 0.0f;
    if (wavelength <= unique_samples.front().x) {
      power = unique_samples.front().y;
      ETX_VALIDATE(power);
    } else if (wavelength >= unique_samples.back().x) {
      power = unique_samples.back().y;
      ETX_VALIDATE(power);
    } else {
      while ((segment + 1u < unique_samples.size()) && (unique_samples[segment + 1u].x < wavelength)) {
        ++segment;
      }
      const float2& s0 = unique_samples[segment];
      const float2& s1 = unique_samples[segment + 1u];
      float denom = s1.x - s0.x;
      float t = (denom > 0.0f) ? (wavelength - s0.x) / denom : 0.0f;
      power = lerp(s0.y, s1.y, saturate(t));
      ETX_VALIDATE(power);
    }
    result.spectral_entries[i] = {wavelength, power};
  }
  ETX_ASSERT(result.valid());

  float3 xyz = result.integrate_to_xyz();
  result.integrated_value = xyz_to_rgb(xyz);
  result.integrated_value = max(result.integrated_value, float3{0.0f, 0.0f, 0.0f});
  return result;
}

void SpectralDistribution::scale(float factor) {
  ETX_ASSERT(factor >= 0.0f);
  for (uint32_t i = 0; i < spectral_entry_count; ++i) {
    spectral_entries[i].power *= factor;
  }
  integrated_value *= factor;
  ETX_ASSERT(valid());
}

SpectralDistribution SpectralDistribution::constant(float value) {
  float2 samples[2] = {
    {kShortestWavelength, value},
    {kLongestWavelength, value},
  };
  SpectralDistribution spd = from_samples(samples, 2);
  spd.integrated_value = {value, value, value};
  return spd;
}

SpectralDistribution SpectralDistribution::from_black_body(float temperature, float scale) {
  float2 samples[WavelengthCount] = {};
  for (uint32_t i = 0; i < WavelengthCount; ++i) {
    float wl = float(i + ShortestWavelength);
    samples[i] = {wl, black_body_radiation(wl, temperature) * scale};
  }
  return from_samples(samples, WavelengthCount);
}

SpectralDistribution SpectralDistribution::from_normalized_black_body(float t, float scale) {
  float w = black_body_radiation_maximum_wavelength(t);
  float r = black_body_radiation(w, t);
  auto spd = SpectralDistribution::from_black_body(t, 1.0f / r);
  spd.scale(scale / spd.luminance());
  return spd;
}

SpectralDistribution SpectralDistribution::rgb_reflectance(const float3& rgb) {
  if (::luminance(rgb) == 0.0f)
    return SpectralDistribution::constant(0.0f);

  ETX_VALIDATE(rgb);

  float2 samples[RGBResponseWavelengthCount] = {};
  for (uint32_t i = RGBResponseShortestWavelength; i <= RGBResponseLongestWavelength; ++i) {
    auto p = rgb_response({float(i), SpectralFlags::Spectral}, rgb);
    samples[i - RGBResponseShortestWavelength] = {float(i), p.value};
  }

  SpectralDistribution spd = from_samples(samples, RGBResponseWavelengthCount);
  spd.integrated_value = rgb;
  return spd;
}

SpectralDistribution SpectralDistribution::rgb_luminance(const float3& rgb) {
  SpectralDistribution result = rgb_reflectance(rgb * kRGBLuminanceScale);
  result.integrated_value = rgb;
  return result;
}

SpectralDistribution::Class SpectralDistribution::load_from_file(const char* file_name, SpectralDistribution& values0, SpectralDistribution* values1, bool extend_range,
  std::string& out_title) {
  auto file = fopen(file_name, "r");
  if (file == nullptr) {
    log::error("Failed to load SpectralDistribution from file: %s\n", file_name);
    return SpectralDistribution::Invalid;
  }

  fseek(file, 0, SEEK_END);
  uint64_t file_size = ftell(file);
  fseek(file, 0, SEEK_SET);

  std::vector<char> data(file_size + 1, 0);
  fread(data.data(), 1, file_size, file);
  fclose(file);

  struct Sample {
    float wavelength = 0.0f;
    float values[2] = {};
    bool operator<(const Sample& other) const {
      return wavelength < other.wavelength;
    }
  };

  Class cls = SpectralDistribution::Invalid;
  std::vector<Sample> samples;
  samples.reserve(WavelengthCount);

  char* begin = data.data();
  char* end = data.data() + file_size;
  while (begin < end) {
    auto line_end = begin;
    while ((line_end < end) && (*line_end != '\n')) {
      ++line_end;
    }
    *line_end = 0;

    if (begin[0] == '#') {
      if (strstr(begin, "#class") == begin) {
        const char* colon = strchr(begin, ':');
        if (colon != nullptr) {
          ++colon;
          while ((*colon == ' ') || (*colon == '\t')) {
            ++colon;
          }
          const char* cls_begin = colon;
          while ((*colon != 0) && (*colon != ' ') && (*colon != '\t')) {
            ++colon;
          }
          std::string cls_name(cls_begin, colon - cls_begin);
          if (cls_name == "conductor") {
            cls = SpectralDistribution::Conductor;
          } else if (cls_name == "dielectric") {
            cls = SpectralDistribution::Dielectric;
          } else if (cls_name == "illuminant") {
            cls = SpectralDistribution::Illuminant;
          } else if (cls_name.empty() == false) {
            cls = SpectralDistribution::Reflectance;
          }
        }
      } else if (strstr(begin, "#title") == begin) {
        const char* title_begin = strchr(begin, ':');
        if (title_begin != nullptr) {
          ++title_begin;
          while ((*title_begin == ' ') || (*title_begin == '\t')) {
            ++title_begin;
          }
          std::string title = title_begin;
          while (!title.empty() && ((title.back() == '\r') || (title.back() == '\n') || (title.back() == ' ') || (title.back() == '\t'))) {
            title.pop_back();
          }
          if (title.empty() == false) {
            out_title = title;
          }
        }
      }
    } else {
      float wavelength = 0.0f;
      float v0 = 1.0f;
      float v1 = 0.0f;

      int args_read = sscanf(begin, "%f %f %f", &wavelength, &v0, &v1);
      if (args_read >= 2) {
        samples.emplace_back(Sample{wavelength, {v0, v1}});
      }
    }

    begin = line_end + 1;
  }

  if (samples.empty()) {
    log::error("Failed to load SpectralDistribution from file: %s\n", file_name);
    return SpectralDistribution::Invalid;
  }

  std::sort(samples.begin(), samples.end());

  float scale = 1.0f;
  float min_value = samples.front().wavelength;
  while (min_value < 100.0f) {
    min_value *= 10.0f;
    scale *= 10.0f;
  }

  std::vector<float2> samples0;
  samples0.reserve(WavelengthCount);
  std::vector<float2> samples1;
  samples1.reserve(WavelengthCount);
  for (auto& sample : samples) {
    float w = sample.wavelength * scale;
    if ((w >= kShortestWavelength) && (w <= kLongestWavelength)) {
      samples0.emplace_back(float2{w, sample.values[0]});
      samples1.emplace_back(float2{w, sample.values[1]});
    }
  }

  if (samples0.empty()) {
    float fallback = samples.front().values[0];
    samples0.emplace_back(float2{kShortestWavelength, fallback});
    samples0.emplace_back(float2{kLongestWavelength, fallback});
  }

  if ((values1 != nullptr) && samples1.empty() && (cls == SpectralDistribution::Conductor)) {
    float fallback = samples.front().values[1];
    samples1.emplace_back(float2{kShortestWavelength, fallback});
    samples1.emplace_back(float2{kLongestWavelength, fallback});
  }

  if (extend_range) {
    if ((samples0.size() < WavelengthCount) && (samples0.front().x > kShortestWavelength)) {
      samples0.insert(samples0.begin(), samples0.front())->x = kShortestWavelength;
    }
    if ((samples0.size() < WavelengthCount) && (samples0.back().x < kLongestWavelength)) {
      samples0.emplace_back(samples0.back()).x = kLongestWavelength;
    }

    if (samples1.empty() == false) {
      if ((samples1.size() < WavelengthCount) && (samples1.front().x > kShortestWavelength)) {
        samples1.insert(samples1.begin(), samples1.front())->x = kShortestWavelength;
      }
      if ((samples1.size() < WavelengthCount) && (samples1.back().x < kLongestWavelength)) {
        samples1.emplace_back(samples1.back()).x = kLongestWavelength;
      }
    }
  }

  values0 = from_samples(samples0.data(), samples0.size());

  if (values1) {
    if (samples1.empty()) {
      float2 zero_samples[2] = {
        {kShortestWavelength, 0.0f},
        {kLongestWavelength, 0.0f},
      };
      *values1 = from_samples(zero_samples, 2);
    } else {
      *values1 = from_samples(samples1.data(), samples1.size());
    }
  }

  if (cls == SpectralDistribution::Illuminant) {
    float lum = values0.luminance();
    if (lum > 0.0f) {
      values0.scale(1.0f / lum);
    }
  }

  return cls;
}

SpectralDistribution::Class SpectralDistribution::load_refractive_index(const char* file_name, SpectralDistribution& out_eta, SpectralDistribution& out_k, std::string& out_title) {
  SpectralDistribution::Class cls = SpectralDistribution::load_from_file(file_name, out_eta, &out_k, true, out_title);
  if (cls != SpectralDistribution::Invalid) {
    out_eta.integrated_value = rgb_to_xyz(out_eta.integrated_value);
    out_k.integrated_value = rgb_to_xyz(out_k.integrated_value);
  } else {
    out_eta = SpectralDistribution::constant(1.0f);
    out_k = SpectralDistribution::constant(0.0f);
  }
  return cls;
}

bool SpectralDistribution::valid() const {
  for (uint32_t i = 0; i < spectral_entry_count; ++i) {
    if (valid_value(spectral_entries[i].power) == false) {
      return false;
    }
  }
  return true;
}

float SpectralDistribution::maximum_spectral_power() const {
  float result = spectral_entries[0].power;
  for (uint32_t i = 0; i < spectral_entry_count; ++i) {
    result = max(result, spectral_entries[i].power);
  }
  return result;
}

float3 SpectralDistribution::integrate_to_xyz() const {
  float3 result = {};
  SpectralResponse s_begin = {{0.0f, SpectralFlags::Spectral}};
  SpectralResponse s_end = {{0.0f, SpectralFlags::Spectral}};

  for (uint32_t index = 0; index + 1 < spectral_entry_count; ++index) {
    float l0 = spectral_entries[index + 0].wavelength;
    float l1 = spectral_entries[index + 1].wavelength;
    float p0 = spectral_entries[index + 0].power;
    float p1 = spectral_entries[index + 1].power;
    s_begin.wavelength = l0;
    while (s_end.wavelength < l1) {
      float t0 = (s_begin.wavelength - l0) / (l1 - l0);
      s_begin.value = lerp(p0, p1, t0);

      s_end.wavelength = min(l1, s_begin.wavelength + 1.0f);
      float t1 = (s_end.wavelength - l0) / (l1 - l0);
      s_end.value = lerp(p0, p1, t1);

      auto v_begin = s_begin.to_xyz();
      auto v_end = s_end.to_xyz();

      result += (s_end.wavelength - s_begin.wavelength) * (v_begin + 0.5f * (v_end - v_begin));

      s_begin.wavelength = s_end.wavelength;
    }
  }

  return result;
}

float SpectralDistribution::luminance() const {
  return ::luminance(integrated_value);
}

const float3& SpectralDistribution::integrated() const {
  return integrated_value;
}

SpectralResponse rgb_response(const SpectralQuery spect, const float3& rgb) {
  const ::SpectralResponse result = ::spectral_rgb_response(spect, rgb);

  SpectralResponse wrapped = {};
  wrapped.integrated = result.integrated;
  wrapped.value = result.value;
  wrapped.wavelength = result.wavelength;
  wrapped.flags = result.flags;
  return wrapped;
}

}  // namespace etx
