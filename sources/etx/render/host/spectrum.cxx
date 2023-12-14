#include <etx/render/shared/spectrum.hxx>

#include <vector>

namespace etx {

SpectralDistribution SpectralDistribution::from_black_body(float temperature, Pointer<Spectrums> spectrums) {
  SpectralDistribution result;
  result.count = spectrum::WavelengthCount;
  for (uint32_t i = 0; i < spectrum::WavelengthCount; ++i) {
    float wl = float(i + spectrum::ShortestWavelength);
    result.entries[i] = {wl, spectrum::black_body_radiation(wl, temperature)};
  }

  if constexpr (spectrum::kSpectralRendering == false) {
    float3 xyz = result.integrate_to_xyz();
    result = rgb::make_spd(spectrum::xyz_to_rgb(xyz), spectrums->rgb_illuminant);
  }
  return result;
}

SpectralDistribution::Class SpectralDistribution::load_from_file(const char* file_name, SpectralDistribution& values0, SpectralDistribution* values1,
  Pointer<Spectrums> spectrums) {
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

  for (auto& sample : samples) {
    sample.wavelength *= scale;
  }

  values0.count = 0;

  for (const auto& sample : samples) {
    if ((sample.wavelength >= spectrum::kShortestWavelength) && (sample.wavelength <= spectrum::kLongestWavelength)) {
      values0.entries[values0.count++] = {sample.wavelength, sample.values[0]};
      if (values0.count >= spectrum::kWavelengthCount) {
        break;
      }
    }
  }

  if (values0.count == 0)
    return Class::Invalid;

  if constexpr (spectrum::kSpectralRendering == false) {
    float3 xyz = values0.integrate_to_xyz();
    values0 = rgb::make_spd(spectrum::xyz_to_rgb(xyz), (cls == Class::Illuminant) ? spectrums->rgb_illuminant : spectrums->rgb_reflection);
  }

  if (values1 != nullptr) {
    values1->count = 0;
    for (const auto& sample : samples) {
      if ((sample.wavelength >= spectrum::kShortestWavelength) && (sample.wavelength <= spectrum::kLongestWavelength)) {
        values1->entries[values1->count++] = {sample.wavelength, sample.values[1]};
        if (values1->count >= spectrum::kWavelengthCount) {
          break;
        }
      }
    }

    if constexpr (spectrum::kSpectralRendering == false) {
      float3 xyz = values1->integrate_to_xyz();
      *values1 = rgb::make_spd(spectrum::xyz_to_rgb(xyz), (cls == Class::Illuminant) ? spectrums->rgb_illuminant : spectrums->rgb_reflection);
    }
  }

  return cls;
}

bool SpectralDistribution::valid() const {
  for (uint32_t i = 0; i < count; ++i) {
    if (valid_value(entries[i].power) == false) {
      return false;
    }
  }
  return true;
}

float3 SpectralDistribution::to_xyz() const {
  if constexpr (spectrum::kSpectralRendering) {
    return integrate_to_xyz();
  } else {
    return spectrum::rgb_to_xyz({entries[0].power, entries[1].power, entries[2].power});
  }
}

float SpectralDistribution::maximum_power() const {
  float result = entries[0].power;
  for (uint32_t i = 0; i < count; ++i) {
    result = max(result, entries[i].power);
  }
  return result;
}

float3 SpectralDistribution::integrate_to_xyz() const {
  auto xyz_at = [](float wl) -> float3 {
    uint32_t i = static_cast<uint32_t>(clamp(wl, spectrum::kShortestWavelength, spectrum::kLongestWavelength) - spectrum::ShortestWavelength);
    uint32_t j = min(i + 1u, spectrum::WavelengthCount - 1);
    auto v0 = spectrum::spectral_xyz(i);
    auto v1 = spectrum::spectral_xyz(j);
    float dw = wl - floorf(wl);
    return lerp(v0, v1, dw);
  };

  auto integrate = [entries = entries, xyz_at](uint32_t index) -> float3 {
    float3 result = {};

    float l0 = entries[index + 0].wavelength;
    float l1 = entries[index + 1].wavelength;
    float p0 = entries[index + 0].power;
    float p1 = entries[index + 1].power;
    float begin = l0;
    for (;;) {
      float end = min(l1, begin + 1.0f);
      float t0 = (begin - l0) / (l1 - l0);
      float t1 = (end - l0) / (l1 - l0);
      float p_begin = lerp(p0, p1, t0);
      float p_end = lerp(p0, p1, t1);

      auto v0 = xyz_at(begin) * p_begin;
      auto v1 = xyz_at(end) * p_end;

      result += (end - begin) * (v0 + 0.5f * (v1 - v0));
      if (end == l1) {
        break;
      }
      begin = end;
    }
    return result / spectrum::kYIntegral;
  };

  float3 result = {};
  for (uint32_t i = 0; i + 1 < count; ++i) {
    result += integrate(i);
  }
  return result;
}

float SpectralDistribution::total_power() const {
  if constexpr (spectrum::kSpectralRendering) {
    return integrate_to_xyz().y;
  } else {
    return entries[0].power * 0.2627f + entries[1].power * 0.678f + entries[2].power * 0.0593f;
  }
}

SpectralDistribution SpectralDistribution::from_samples(const float wavelengths[], const float power[], uint32_t count) {
  float value = (count > 0) ? wavelengths[0] : 100.0f;
  float wavelength_scale = 1.0f;
  while (value < 100.0f) {
    wavelength_scale *= 10.0f;
    value *= 10.0f;
  }

  SpectralDistribution result;
  result.count = count;
  for (uint32_t i = 0; i < count; ++i) {
    result.entries[i] = {wavelengths[i] * wavelength_scale, power ? power[i] : 0.0f};
  }

  for (uint32_t i = 0; i < count; ++i) {
    for (uint32_t j = i + 1; j < count; ++j) {
      if (result.entries[i].wavelength > result.entries[j].wavelength) {
        auto t = result.entries[i];
        result.entries[i] = result.entries[j];
        result.entries[j] = t;
      }
    }
  }

  return result;
}

SpectralDistribution SpectralDistribution::from_samples(const float wavelengths[], const float power[], uint32_t count, Class cls, Pointer<Spectrums> spectrums) {
  auto result = from_samples(wavelengths, power, count);
  if constexpr (spectrum::kSpectralRendering == false) {
    float3 xyz = result.integrate_to_xyz();
    result = rgb::make_spd(spectrum::xyz_to_rgb(xyz), (cls == Class::Reflectance) ? spectrums->rgb_reflection : spectrums->rgb_illuminant);
  }
  return result;
}

SpectralDistribution SpectralDistribution::from_samples(const float2 wavelengths_power[], uint32_t count, Class cls, Pointer<Spectrums> spectrums) {
  float value = (count > 0) ? wavelengths_power[0].x : 100.0f;
  float wavelength_scale = 1.0f;
  while (value < 100.0f) {
    wavelength_scale *= 10.0f;
    value *= 10.0f;
  }

  SpectralDistribution result;
  result.count = count;
  for (uint32_t i = 0; i < count; ++i) {
    result.entries[i] = {wavelengths_power[i].x * wavelength_scale, wavelengths_power[i].y};
  }

  for (uint32_t i = 0; i < count; ++i) {
    for (uint32_t j = i + 1; j < count; ++j) {
      if (result.entries[i].wavelength > result.entries[j].wavelength) {
        auto t = result.entries[i];
        result.entries[i] = result.entries[j];
        result.entries[j] = t;
      }
    }
  }

  if constexpr (spectrum::kSpectralRendering == false) {
    float3 xyz = result.integrate_to_xyz();
    result = rgb::make_spd(spectrum::xyz_to_rgb(xyz), (cls == Class::Reflectance) ? spectrums->rgb_reflection : spectrums->rgb_illuminant);
  }

  return result;
}

namespace rgb {

constexpr float kRGBLambda[SampleCount] = {380.000000f, 390.967743f, 401.935486f, 412.903229f, 423.870972f, 434.838715f, 445.806458f, 456.774200f, 467.741943f, 478.709686f,
  489.677429f, 500.645172f, 511.612915f, 522.580627f, 533.548340f, 544.516052f, 555.483765f, 566.451477f, 577.419189f, 588.386902f, 599.354614f, 610.322327f, 621.290039f,
  632.257751f, 643.225464f, 654.193176f, 665.160889f, 676.128601f, 687.096313f, 698.064026f, 709.031738f, 720.000000f};

constexpr float kRGBRefectionWhite[SampleCount] = {1.0618958571272863e+00f, 1.0615019980348779e+00f, 1.0614335379927147e+00f, 1.0622711654692485e+00f, 1.0622036218416742e+00f,
  1.0625059965187085e+00f, 1.0623938486985884e+00f, 1.0624706448043137e+00f, 1.0625048144827762e+00f, 1.0624366131308856e+00f, 1.0620694238892607e+00f, 1.0613167586932164e+00f,
  1.0610334029377020e+00f, 1.0613868564828413e+00f, 1.0614215366116762e+00f, 1.0620336151299086e+00f, 1.0625497454805051e+00f, 1.0624317487992085e+00f, 1.0625249140554480e+00f,
  1.0624277664486914e+00f, 1.0624749854090769e+00f, 1.0625538581025402e+00f, 1.0625326910104864e+00f, 1.0623922312225325e+00f, 1.0623650980354129e+00f, 1.0625256476715284e+00f,
  1.0612277619533155e+00f, 1.0594262608698046e+00f, 1.0599810758292072e+00f, 1.0602547314449409e+00f, 1.0601263046243634e+00f, 1.0606565756823634e+00f};

constexpr float kRGBRefectionCyan[SampleCount] = {1.0414628021426751e+00f, 1.0328661533771188e+00f, 1.0126146228964314e+00f, 1.0350460524836209e+00f, 1.0078661447098567e+00f,
  1.0422280385081280e+00f, 1.0442596738499825e+00f, 1.0535238290294409e+00f, 1.0180776226938120e+00f, 1.0442729908727713e+00f, 1.0529362541920750e+00f, 1.0537034271160244e+00f,
  1.0533901869215969e+00f, 1.0537782700979574e+00f, 1.0527093770467102e+00f, 1.0530449040446797e+00f, 1.0550554640191208e+00f, 1.0553673610724821e+00f, 1.0454306634683976e+00f,
  6.2348950639230805e-01f, 1.8038071613188977e-01f, -7.6303759201984539e-03f, -1.5217847035781367e-04f, -7.5102257347258311e-03f, -2.1708639328491472e-03f, 6.5919466602369636e-04f,
  1.2278815318539780e-02f, -4.4669775637208031e-03f, 1.7119799082865147e-02f, 4.9211089759759801e-03f, 5.8762925143334985e-03f, 2.5259399415550079e-02f};

constexpr float kRGBRefectionMagenta[SampleCount] = {9.9422138151236850e-01f, 9.8986937122975682e-01f, 9.8293658286116958e-01f, 9.9627868399859310e-01f, 1.0198955019000133e+00f,
  1.0166395501210359e+00f, 1.0220913178757398e+00f, 9.9651666040682441e-01f, 1.0097766178917882e+00f, 1.0215422470827016e+00f, 6.4031953387790963e-01f, 2.5012379477078184e-03f,
  6.5339939555769944e-03f, 2.8334080462675826e-03f, -5.1209675389074505e-11, -9.0592291646646381e-03f, 3.3936718323331200e-03f, -3.0638741121828406e-03f, 2.2203936168286292e-01f,
  6.3141140024811970e-01f, 9.7480985576500956e-01f, 9.7209562333590571e-01f, 1.0173770302868150e+00f, 9.9875194322734129e-01f, 9.4701725739602238e-01f, 8.5258623154354796e-01f,
  9.4897798581660842e-01f, 9.4751876096521492e-01f, 9.9598944191059791e-01f, 8.6301351503809076e-01f, 8.9150987853523145e-01f, 8.4866492652845082e-01f};

constexpr float kRGBRefectionYellow[SampleCount] = {5.5740622924920873e-03f, -4.7982831631446787e-03f, -5.2536564298613798e-03f, -6.4571480044499710e-03f, -5.9693514658007013e-03f,
  -2.1836716037686721e-03f, 1.6781120601055327e-02f, 9.6096355429062641e-02f, 2.1217357081986446e-01f, 3.6169133290685068e-01f, 5.3961011543232529e-01f, 7.4408810492171507e-01f,
  9.2209571148394054e-01f, 1.0460304298411225e+00f, 1.0513824989063714e+00f, 1.0511991822135085e+00f, 1.0510530911991052e+00f, 1.0517397230360510e+00f, 1.0516043086790485e+00f,
  1.0511944032061460e+00f, 1.0511590325868068e+00f, 1.0516612465483031e+00f, 1.0514038526836869e+00f, 1.0515941029228475e+00f, 1.0511460436960840e+00f, 1.0515123758830476e+00f,
  1.0508871369510702e+00f, 1.0508923708102380e+00f, 1.0477492815668303e+00f, 1.0493272144017338e+00f, 1.0435963333422726e+00f, 1.0392280772051465e+00f};

constexpr float kRGBRefectionRed[SampleCount] = {1.6575604867086180e-01f, 1.1846442802747797e-01f, 1.2408293329637447e-01f, 1.1371272058349924e-01f, 7.8992434518899132e-02f,
  3.2205603593106549e-02f, -1.0798365407877875e-02f, 1.8051975516730392e-02f, 5.3407196598730527e-03f, 1.3654918729501336e-02f, -5.9564213545642841e-03f, -1.8444365067353252e-03f,
  -1.0571884361529504e-02f, -2.9375521078000011e-03f, -1.0790476271835936e-02f, -8.0224306697503633e-03f, -2.2669167702495940e-03f, 7.0200240494706634e-03f,
  -8.1528469000299308e-03f, 6.0772866969252792e-01f, 9.8831560865432400e-01f, 9.9391691044078823e-01f, 1.0039338994753197e+00f, 9.9234499861167125e-01f, 9.9926530858855522e-01f,
  1.0084621557617270e+00f, 9.8358296827441216e-01f, 1.0085023660099048e+00f, 9.7451138326568698e-01f, 9.8543269570059944e-01f, 9.3495763980962043e-01f, 9.8713907792319400e-01f};

constexpr float kRGBRefectionGreen[SampleCount] = {2.6494153587602255e-03f, -5.0175013429732242e-03f, -1.2547236272489583e-02f, -9.4554964308388671e-03f, -1.2526086181600525e-02f,
  -7.9170697760437767e-03f, -7.9955735204175690e-03f, -9.3559433444469070e-03f, 6.5468611982999303e-02f, 3.9572875517634137e-01f, 7.5244022299886659e-01f, 9.6376478690218559e-01f,
  9.9854433855162328e-01f, 9.9992977025287921e-01f, 9.9939086751140449e-01f, 9.9994372267071396e-01f, 9.9939121813418674e-01f, 9.9911237310424483e-01f, 9.6019584878271580e-01f,
  6.3186279338432438e-01f, 2.5797401028763473e-01f, 9.4014888527335638e-03f, -3.0798345608649747e-03f, -4.5230367033685034e-03f, -6.8933410388274038e-03f, -9.0352195539015398e-03f,
  -8.5913667165340209e-03f, -8.3690869120289398e-03f, -7.8685832338754313e-03f, -8.3657578711085132e-06f, 5.4301225442817177e-03f, -2.7745589759259194e-03f};

constexpr float kRGBRefectionBlue[SampleCount] = {9.9209771469720676e-01f, 9.8876426059369127e-01f, 9.9539040744505636e-01f, 9.9529317353008218e-01f, 9.9181447411633950e-01f,
  1.0002584039673432e+00f, 9.9968478437342512e-01f, 9.9988120766657174e-01f, 9.8504012146370434e-01f, 7.9029849053031276e-01f, 5.6082198617463974e-01f, 3.3133458513996528e-01f,
  1.3692410840839175e-01f, 1.8914906559664151e-02f, -5.1129770932550889e-06f, -4.2395493167891873e-04f, -4.1934593101534273e-04f, 1.7473028136486615e-03f, 3.7999160177631316e-03f,
  -5.5101474906588642e-04f, -4.3716662898480967e-05f, 7.5874501748732798e-03f, 2.5795650780554021e-02f, 3.8168376532500548e-02f, 4.9489586408030833e-02f, 4.9595992290102905e-02f,
  4.9814819505812249e-02f, 3.9840911064978023e-02f, 3.0501024937233868e-02f, 2.1243054765241080e-02f, 6.9596532104356399e-03f, 4.1733649330980525e-03f};

constexpr float kRGBIlluminantWhite[SampleCount] = {1.1565232050369776e+00f, 1.1567225000119139e+00f, 1.1566203150243823e+00f, 1.1555782088080084e+00f, 1.1562175509215700e+00f,
  1.1567674012207332e+00f, 1.1568023194808630e+00f, 1.1567677445485520e+00f, 1.1563563182952830e+00f, 1.1567054702510189e+00f, 1.1565134139372772e+00f, 1.1564336176499312e+00f,
  1.1568023181530034e+00f, 1.1473147688514642e+00f, 1.1339317140561065e+00f, 1.1293876490671435e+00f, 1.1290515328639648e+00f, 1.0504864823782283e+00f, 1.0459696042230884e+00f,
  9.9366687168595691e-01f, 9.5601669265393940e-01f, 9.2467482033511805e-01f, 9.1499944702051761e-01f, 8.9939467658453465e-01f, 8.9542520751331112e-01f, 8.8870566693814745e-01f,
  8.8222843814228114e-01f, 8.7998311373826676e-01f, 8.7635244612244578e-01f, 8.8000368331709111e-01f, 8.8065665428441120e-01f, 8.8304706460276905e-01f};

constexpr float kRGBIlluminantCyan[SampleCount] = {1.1334479663682135e+00f, 1.1266762330194116e+00f, 1.1346827504710164e+00f, 1.1357395805744794e+00f, 1.1356371830149636e+00f,
  1.1361152989346193e+00f, 1.1362179057706772e+00f, 1.1364819652587022e+00f, 1.1355107110714324e+00f, 1.1364060941199556e+00f, 1.1360363621722465e+00f, 1.1360122641141395e+00f,
  1.1354266882467030e+00f, 1.1363099407179136e+00f, 1.1355450412632506e+00f, 1.1353732327376378e+00f, 1.1349496420726002e+00f, 1.1111113947168556e+00f, 9.0598740429727143e-01f,
  6.1160780787465330e-01f, 2.9539752170999634e-01f, 9.5954200671150097e-02f, -1.1650792030826267e-02f, -1.2144633073395025e-02f, -1.1148167569748318e-02f, -1.1997606668458151e-02f,
  -5.0506855475394852e-03f, -7.9982745819542154e-03f, -9.4722817708236418e-03f, -5.5329541006658815e-03f, -4.5428914028274488e-03f, -1.2541015360921132e-02f};

constexpr float kRGBIlluminantMagenta[SampleCount] = {1.0371892935878366e+00f, 1.0587542891035364e+00f, 1.0767271213688903e+00f, 1.0762706844110288e+00f, 1.0795289105258212e+00f,
  1.0743644742950074e+00f, 1.0727028691194342e+00f, 1.0732447452056488e+00f, 1.0823760816041414e+00f, 1.0840545681409282e+00f, 9.5607567526306658e-01f, 5.5197896855064665e-01f,
  8.4191094887247575e-02f, 8.7940070557041006e-05f, -2.3086408335071251e-03f, -1.1248136628651192e-03f, -7.7297612754989586e-11, -2.7270769006770834e-04f, 1.4466473094035592e-02f,
  2.5883116027169478e-01f, 5.2907999827566732e-01f, 9.0966624097105164e-01f, 1.0690571327307956e+00f, 1.0887326064796272e+00f, 1.0637622289511852e+00f, 1.0201812918094260e+00f,
  1.0262196688979945e+00f, 1.0783085560613190e+00f, 9.8333849623218872e-01f, 1.0707246342802621e+00f, 1.0634247770423768e+00f, 1.0150875475729566e+00f};

constexpr float kRGBIlluminantYellow[SampleCount] = {2.7756958965811972e-03f, 3.9673820990646612e-03f, -1.4606936788606750e-04f, 3.6198394557748065e-04f, -2.5819258699309733e-04f,
  -5.0133191628082274e-05f, -2.4437242866157116e-04f, -7.8061419948038946e-05f, 4.9690301207540921e-02f, 4.8515973574763166e-01f, 1.0295725854360589e+00f, 1.0333210878457741e+00f,
  1.0368102644026933e+00f, 1.0364884018886333e+00f, 1.0365427939411784e+00f, 1.0368595402854539e+00f, 1.0365645405660555e+00f, 1.0363938240707142e+00f, 1.0367205578770746e+00f,
  1.0365239329446050e+00f, 1.0361531226427443e+00f, 1.0348785007827348e+00f, 1.0042729660717318e+00f, 8.4218486432354278e-01f, 7.3759394894801567e-01f, 6.5853154500294642e-01f,
  6.0531682444066282e-01f, 5.9549794132420741e-01f, 5.9419261278443136e-01f, 5.6517682326634266e-01f, 5.6061186014968556e-01f, 5.8228610381018719e-01f};

constexpr float kRGBIlluminantRed[SampleCount] = {5.4711187157291841e-02f, 5.5609066498303397e-02f, 6.0755873790918236e-02f, 5.6232948615962369e-02f, 4.6169940535708678e-02f,
  3.8012808167818095e-02f, 2.4424225756670338e-02f, 3.8983580581592181e-03f, -5.6082252172734437e-04f, 9.6493871255194652e-04f, 3.7341198051510371e-04f, -4.3367389093135200e-04f,
  -9.3533962256892034e-05f, -1.2354967412842033e-04f, -1.4524548081687461e-04f, -2.0047691915543731e-04f, -4.9938587694693670e-04f, 2.7255083540032476e-02f,
  1.6067405906297061e-01f, 3.5069788873150953e-01f, 5.7357465538418961e-01f, 7.6392091890718949e-01f, 8.9144466740381523e-01f, 9.6394609909574891e-01f, 9.8879464276016282e-01f,
  9.9897449966227203e-01f, 9.8605140403564162e-01f, 9.9532502805345202e-01f, 9.7433478377305371e-01f, 9.9134364616871407e-01f, 9.8866287772174755e-01f, 9.9713856089735531e-01f};

constexpr float kRGBIlluminantGreen[SampleCount] = {2.5168388755514630e-02f, 3.9427438169423720e-02f, 6.2059571596425793e-03f, 7.1120859807429554e-03f, 2.1760044649139429e-04f,
  7.3271839984290210e-12, -2.1623066217181700e-02f, 1.5670209409407512e-02f, 2.8019603188636222e-03f, 3.2494773799897647e-01f, 1.0164917292316602e+00f, 1.0329476657890369e+00f,
  1.0321586962991549e+00f, 1.0358667411948619e+00f, 1.0151235476834941e+00f, 1.0338076690093119e+00f, 1.0371372378155013e+00f, 1.0361377027692558e+00f, 1.0229822432557210e+00f,
  9.6910327335652324e-01f, -5.1785923899878572e-03f, 1.1131261971061429e-03f, 6.6675503033011771e-03f, 7.4024315686001957e-04f, 2.1591567633473925e-02f, 5.1481620056217231e-03f,
  1.4561928645728216e-03f, 1.6414511045291513e-04f, -6.4630764968453287e-03f, 1.0250854718507939e-02f, 4.2387394733956134e-02f, 2.1252716926861620e-02f};

constexpr float kRGBIlluminantBlue[SampleCount] = {1.0570490759328752e+00f, 1.0538466912851301e+00f, 1.0550494258140670e+00f, 1.0530407754701832e+00f, 1.0579930596460185e+00f,
  1.0578439494812371e+00f, 1.0583132387180239e+00f, 1.0579712943137616e+00f, 1.0561884233578465e+00f, 1.0571399285426490e+00f, 1.0425795187752152e+00f, 3.2603084374056102e-01f,
  -1.9255628442412243e-03f, -1.2959221137046478e-03f, -1.4357356276938696e-03f, -1.2963697250337886e-03f, -1.9227081162373899e-03f, 1.2621152526221778e-03f,
  -1.6095249003578276e-03f, -1.3029983817879568e-03f, -1.7666600873954916e-03f, -1.2325281140280050e-03f, 1.0316809673254932e-02f, 3.1284512648354357e-02f, 8.8773879881746481e-02f,
  1.3873621740236541e-01f, 1.5535067531939065e-01f, 1.4878477178237029e-01f, 1.6624255403475907e-01f, 1.6997613960634927e-01f, 1.5769743995852967e-01f, 1.9069090525482305e-01f};

inline const float* power(uint32_t cls, uint32_t clr) {
  constexpr const float* ptr[2][7] = {
    {kRGBRefectionRed, kRGBRefectionGreen, kRGBRefectionBlue, kRGBRefectionYellow, kRGBRefectionMagenta, kRGBRefectionCyan, kRGBRefectionWhite},
    {kRGBIlluminantRed, kRGBIlluminantGreen, kRGBIlluminantBlue, kRGBIlluminantYellow, kRGBIlluminantMagenta, kRGBIlluminantCyan, kRGBIlluminantWhite},
  };
  return ptr[cls][clr];
}

void init_spectrums(Spectrums& spectrums) {
  for (uint32_t i = 0; i < SampleCount; ++i) {
    for (uint32_t c = 0; c < Color::Count; ++c) {
      spectrums.rgb_reflection.values[c][i] = 0.941f * power(0u, c)[i];
      spectrums.rgb_illuminant.values[c][i] = 0.86445f * power(1u, c)[i];
    }
  }
}

}  // namespace rgb

}  // namespace etx
