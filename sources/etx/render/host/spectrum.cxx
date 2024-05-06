#include <etx/render/shared/spectrum.hxx>

#include <vector>
#include <algorithm>

namespace etx {

SpectralDistribution SpectralDistribution::from_samples(const float2 wavelengths_power[], uint64_t count, Mapping mapping) {
  float value = (count > 0) ? wavelengths_power[0].x : 100.0f;
  float wavelength_scale = 1.0f;
  while (value < 100.0f) {
    wavelength_scale *= 10.0f;
    value *= 10.0f;
  }

  SpectralDistribution result;
  result.spectral_entry_count = static_cast<uint32_t>(count);
  for (uint32_t i = 0; i < result.spectral_entry_count; ++i) {
    result.spectral_entries[i] = {wavelengths_power[i].x * wavelength_scale, wavelengths_power[i].y};
  }

  if (count == 1) {
    result.spectral_entries[1] = result.spectral_entries[0];
    result.spectral_entries[0].wavelength -= 0.5f;
    result.spectral_entries[1].wavelength += 0.5f;
    result.spectral_entry_count = 2;
  }

  for (uint32_t i = 0; i < result.spectral_entry_count; ++i) {
    for (uint32_t j = i + 1; j < result.spectral_entry_count; ++j) {
      if (result.spectral_entries[i].wavelength > result.spectral_entries[j].wavelength) {
        auto t = result.spectral_entries[i];
        result.spectral_entries[i] = result.spectral_entries[j];
        result.spectral_entries[j] = t;
      }
    }
  }

  float3 xyz = result.integrate_to_xyz();
  float3 rgb = spectrum::xyz_to_rgb(xyz);

  result.integrated_value = mapping == Mapping::Color ? rgb : xyz;

  return result;
}

void SpectralDistribution::scale(float factor) {
  for (uint32_t i = 0; i < spectral_entry_count; ++i) {
    spectral_entries[i].power *= factor;
  }
  integrated_value *= factor;
}

SpectralDistribution SpectralDistribution::null() {
  return constant(0.0f);
}

SpectralDistribution SpectralDistribution::constant(float value) {
  float2 samples[2] = {
    {spectrum::kShortestWavelength, value},
    {spectrum::kLongestWavelength, value},
  };
  SpectralDistribution spd = from_samples(samples, 2, Mapping::Direct);
  spd.integrated_value = {value, value, value};
  return spd;
}

SpectralDistribution SpectralDistribution::from_black_body(float temperature, float scale) {
  float2 samples[spectrum::WavelengthCount] = {};
  for (uint32_t i = 0; i < spectrum::WavelengthCount; ++i) {
    float wl = float(i + spectrum::ShortestWavelength);
    samples[i] = {wl, spectrum::black_body_radiation(wl, temperature) * scale};
  }
  return from_samples(samples, spectrum::WavelengthCount, Mapping::Color);
}

SpectralDistribution SpectralDistribution::from_normalized_black_body(float t, float scale) {
  float w = spectrum::black_body_radiation_maximum_wavelength(t);
  float r = spectrum::black_body_radiation(w, t);
  auto spd = SpectralDistribution::from_black_body(t, 1.0f / r);
  spd.scale(scale / spd.luminance());
  return spd;
}

SpectralDistribution SpectralDistribution::rgb(const float3& rgb) {
  float2 samples[spectrum::RGBResponseWavelengthCount] = {};
  for (uint32_t i = spectrum::RGBResponseShortestWavelength; i <= spectrum::RGBResponseLongestWavelength; ++i) {
    auto p = rgb::query_spd({float(i), SpectralQuery::Spectral}, rgb);
    samples[i - spectrum::RGBResponseShortestWavelength] = {float(i), p.components.w};
  }

  SpectralDistribution spd = from_samples(samples, spectrum::RGBResponseWavelengthCount, Mapping::Color);
  spd.integrated_value = rgb;
  return spd;
}

SpectralDistribution::Class SpectralDistribution::load_from_file(const char* file_name, SpectralDistribution& values0, SpectralDistribution* values1, bool extend_range,
  Mapping mapping) {
  auto file = fopen(file_name, "r");
  if (file == nullptr) {
    printf("Failed to load SpectralDistribution from file: %s\n", file_name);
    return SpectralDistribution::Class::Invalid;
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

  Class cls = Class::Invalid;
  std::vector<Sample> samples;
  samples.reserve(spectrum::WavelengthCount);

  char* begin = data.data();
  char* end = data.data() + file_size;
  while (begin < end) {
    auto line_end = begin;
    while ((line_end < end) && (*line_end != '\n')) {
      ++line_end;
    }
    *line_end = 0;

    if (begin[0] == '#') {
      if (strstr(begin, "#class:") == begin) {
        const char* cls_name = begin + 7;
        if (strcmp(cls_name, "conductor") == 0) {
          cls = Class::Conductor;
        } else if (strcmp(cls_name, "dielectric") == 0) {
          cls = Class::Dielectric;
        } else if (strcmp(cls_name, "illuminant") == 0) {
          cls = Class::Illuminant;
        } else {
          cls = Class::Reflectance;
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

  std::sort(samples.begin(), samples.end());

  float scale = 1.0f;
  float min_value = samples.front().wavelength;
  while (min_value < 100.0f) {
    min_value *= 10.0f;
    scale *= 10.0f;
  }

  std::vector<float2> samples0;
  samples0.reserve(spectrum::WavelengthCount);
  std::vector<float2> samples1;
  samples1.reserve(spectrum::WavelengthCount);
  for (auto& sample : samples) {
    float w = sample.wavelength * scale;
    if ((w >= spectrum::kShortestWavelength) && (w <= spectrum::kLongestWavelength)) {
      samples0.emplace_back(float2{w, sample.values[0]});
      samples1.emplace_back(float2{w, sample.values[1]});
    }
  }

  if (extend_range) {
    if ((samples0.size() < spectrum::WavelengthCount) && (samples0.front().x > spectrum::kShortestWavelength)) {
      samples0.insert(samples0.begin(), samples0.front())->x = spectrum::kShortestWavelength;
    }
    if ((samples0.size() < spectrum::WavelengthCount) && (samples0.back().x < spectrum::kLongestWavelength)) {
      samples0.emplace_back(samples0.back()).x = spectrum::kLongestWavelength;
    }

    if (samples1.empty() == false) {
      if ((samples1.size() < spectrum::WavelengthCount) && (samples1.front().x > spectrum::kShortestWavelength)) {
        samples1.insert(samples1.begin(), samples1.front())->x = spectrum::kShortestWavelength;
      }
      if ((samples1.size() < spectrum::WavelengthCount) && (samples1.back().x < spectrum::kLongestWavelength)) {
        samples1.emplace_back(samples1.back()).x = spectrum::kLongestWavelength;
      }
    }
  }

  values0 = from_samples(samples0.data(), samples0.size(), mapping);

  if (values1) {
    *values1 = from_samples(samples1.data(), samples1.size(), mapping);
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
  SpectralResponse s_begin = {{0.0f, SpectralQuery::Spectral}};
  SpectralResponse s_end = {{0.0f, SpectralQuery::Spectral}};

  for (uint32_t index = 0; index + 1 < spectral_entry_count; ++index) {
    float l0 = spectral_entries[index + 0].wavelength;
    float l1 = spectral_entries[index + 1].wavelength;
    float p0 = spectral_entries[index + 0].power;
    float p1 = spectral_entries[index + 1].power;
    s_begin.wavelength = l0;
    while (s_end.wavelength < l1) {
      float t0 = (s_begin.wavelength - l0) / (l1 - l0);
      s_begin.components.w = lerp(p0, p1, t0);

      s_end.wavelength = min(l1, s_begin.wavelength + 1.0f);
      float t1 = (s_end.wavelength - l0) / (l1 - l0);
      s_end.components.w = lerp(p0, p1, t1);

      auto v_begin = s_begin.to_xyz();
      auto v_end = s_end.to_xyz();

      result += (s_end.wavelength - s_begin.wavelength) * (v_begin + 0.5f * (v_end - v_begin));

      s_begin.wavelength = s_end.wavelength;
    }
  }

  return result;
}

float SpectralDistribution::luminance() const {
  return etx::luminance(integrated_value);
}

const float3& SpectralDistribution::integrated() const {
  return integrated_value;
}

}  // namespace etx
