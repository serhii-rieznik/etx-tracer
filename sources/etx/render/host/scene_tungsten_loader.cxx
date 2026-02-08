#include <etx/render/interop/interop.hxx>

#include <etx/core/environment.hxx>
#include <etx/core/json.hxx>

#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scattering.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/host/scene_tungsten_loader.hxx>
#include <etx/render/host/scene_obj_loader.hxx>
#include <etx/render/host/scene_gltf_loader.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <etx/render/shared/ior_database.hxx>
namespace etx {

namespace {

float3 json_to_float3(const nlohmann::json& arr, const float3& fallback) {
  if (arr.is_array() && arr.size() >= 3) {
    return {float(arr[0].get<double>()), float(arr[1].get<double>()), float(arr[2].get<double>())};
  }
  return fallback;
}

std::string resolve_path(const std::string& base_dir, const std::string& path) {
  if (path.empty())
    return {};
  std::filesystem::path p(path);
  if (p.is_absolute())
    return p.lexically_normal().string();
  return (std::filesystem::path(base_dir) / p).lexically_normal().string();
}

const char* get_ext(const std::string& path) {
  auto pos = path.find_last_of('.');
  if (pos == std::string::npos)
    return "";
  return path.c_str() + pos;
}

SpectralDistribution json_to_rgb_spectrum(const nlohmann::json& v, float def_value) {
  if (v.is_array() && v.size() >= 3) {
    float3 c = {def_value, def_value, def_value};
    c.x = v[0].is_number() ? float(v[0].get<double>()) : c.x;
    c.y = v[1].is_number() ? float(v[1].get<double>()) : c.y;
    c.z = v[2].is_number() ? float(v[2].get<double>()) : c.z;
    return SpectralDistribution::rgb_luminance(c);
  }
  if (v.is_number()) {
    float s = float(v.get<double>());
    return SpectralDistribution::rgb_luminance({s, s, s});
  }
  return SpectralDistribution::rgb_luminance({def_value, def_value, def_value});
}

float3 rotate_yxz_deg(const float3& v, const float3& rot_deg) {
  const float3 r = rot_deg * (kPi / 180.0f);
  const float c0 = cosf(r.x);
  const float s0 = sinf(r.x);
  const float c1 = cosf(r.y);
  const float s1 = sinf(r.y);
  const float c2 = cosf(r.z);
  const float s2 = sinf(r.z);

  float3 out;
  out.x = (c1 * c2 - s1 * s0 * s2) * v.x + (-c1 * s2 - s1 * s0 * c2) * v.y + (-s1 * c0) * v.z;
  out.y = (c0 * s2) * v.x + (c0 * c2) * v.y + (-s0) * v.z;
  out.z = (s1 * c2 + c1 * s0 * s2) * v.x + (-s1 * s2 + c1 * s0 * c2) * v.y + (c1 * c0) * v.z;
  return out;
}

uint32_t resolve_tungsten_texture(const nlohmann::json& v, SceneData& data, const char* base_dir) {
  if (v.is_string() == false)
    return kInvalidIndex;
  std::string p = resolve_path(base_dir, v.get<std::string>());
  if (p.empty())
    return kInvalidIndex;
  return data.add_image(p.c_str(), Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
}

nlohmann::json tungsten_albedo_json(const nlohmann::json& src) {
  return src.contains("albedo") ? src["albedo"] : nlohmann::json();
}

void tungsten_set_albedo(Material& mtl, const nlohmann::json& val, SceneData& data, const char* base_dir) {
  if (val.is_string()) {
    uint32_t tex_idx = resolve_tungsten_texture(val, data, base_dir);
    if (tex_idx != kInvalidIndex) {
      mtl.scattering.image_index = tex_idx;
      mtl.reflectance.image_index = tex_idx;
      mtl.scattering.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
      mtl.reflectance.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
      return;
    }
  }

  if (val.is_array() && val.size() >= 3) {
    float3 c = {};
    c.x = val[0].is_number() ? float(val[0].get<double>()) : 1.0f;
    c.y = val[1].is_number() ? float(val[1].get<double>()) : 1.0f;
    c.z = val[2].is_number() ? float(val[2].get<double>()) : 1.0f;
    uint32_t sp_s = data.add_spectrum(SpectralDistribution::rgb_reflectance(c));
    uint32_t sp_r = data.add_spectrum(SpectralDistribution::rgb_reflectance(c));
    mtl.scattering.spectrum_index = sp_s;
    mtl.reflectance.spectrum_index = sp_r;
    return;
  }

  if (val.is_number()) {
    float s = float(val.get<double>());
    uint32_t sp_s = data.add_spectrum(SpectralDistribution::rgb_reflectance({s, s, s}));
    uint32_t sp_r = data.add_spectrum(SpectralDistribution::rgb_reflectance({s, s, s}));
    mtl.scattering.spectrum_index = sp_s;
    mtl.reflectance.spectrum_index = sp_r;
    return;
  }

  mtl.scattering.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
  mtl.reflectance.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
}

void tungsten_set_transmission(Material& mtl, const nlohmann::json& val, SceneData& data, const char* base_dir) {
  if (val.is_string()) {
    uint32_t tex_idx = resolve_tungsten_texture(val, data, base_dir);
    if (tex_idx != kInvalidIndex) {
      mtl.transmission.image_index = tex_idx;
      mtl.transmission.channel = 0u;
      mtl.transmission.value = {1.0f, 1.0f, 1.0f, 1.0f};
      return;
    }
  }

  if (val.is_array() && val.size() >= 3) {
    float3 c = {};
    c.x = val[0].is_number() ? float(val[0].get<double>()) : 1.0f;
    c.y = val[1].is_number() ? float(val[1].get<double>()) : 1.0f;
    c.z = val[2].is_number() ? float(val[2].get<double>()) : 1.0f;
    mtl.transmission.image_index = kInvalidIndex;
    mtl.transmission.value = {c.x, c.y, c.z, 1.0f};
    return;
  }

  if (val.is_number()) {
    float s = float(val.get<double>());
    mtl.transmission.image_index = kInvalidIndex;
    mtl.transmission.value = {s, s, s, 1.0f};
    return;
  }

  mtl.transmission.image_index = kInvalidIndex;
  mtl.transmission.value = {1.0f, 1.0f, 1.0f, 1.0f};
}

struct TungstenConductorIOR {
  const char* name;
  float3 eta;
  float3 k;
};

struct PrimitiveLoadResult {
  bool loaded = false;
  uint32_t flags = 0u;
};

bool add_builtin_quad(const float3& translate, const float3& scale, const float3& rotation_deg, uint32_t material_index, SceneData& data, const char* name);
bool add_builtin_cube(const float3& translate, const float3& scale, const float3& rotation_deg, uint32_t material_index, SceneData& data, const char* name);
bool add_builtin_disk(const float3& translate, const float3& scale, const float3& rotation_deg, uint32_t material_index, SceneData& data, const char* name);
bool add_builtin_sphere(const float3& translate, const float3& scale, const float3& rotation_deg, uint32_t material_index, SceneData& data, const char* name);
bool load_wo3_mesh(const std::string& resolved, const float3& translate, const float3& scale, const float3& rotation_deg, uint32_t material_index, SceneData& data,
  bool recompute_normals);
void transform_vertices(SceneData& data, uint32_t vertex_start, uint32_t vertex_end, const float3& rotation_deg, const float3& scale, const float3& translate);
void recompute_mesh_bounds(SceneData& data, uint32_t mesh_start, uint32_t mesh_end);

static const TungstenConductorIOR kTungstenConductors[] = {
  {"a-C", {2.9440999183f, 2.2271502925f, 1.9681668794f}, {0.8874329109f, 0.7993216383f, 0.8152862927f}},
  {"Ag", {0.1552646489f, 0.1167232965f, 0.1383806959f}, {4.8283433224f, 3.1222459278f, 2.1469504455f}},
  {"Al", {1.6574599595f, 0.8803689579f, 0.5212287346f}, {9.2238691996f, 6.2695232477f, 4.8370012281f}},
  {"AlAs", {3.6051023902f, 3.2329365777f, 2.2175611545f}, {0.0006670247f, -0.00049994f, 0.0074261204f}},
  {"AlSb", {-0.0485225705f, 4.1427547893f, 4.6697691348f}, {-0.0363741915f, 0.0937665154f, 1.3007390124f}},
  {"Au", {0.1431189557f, 0.3749570432f, 1.4424785571f}, {3.9831604247f, 2.3857207478f, 1.6032152899f}},
  {"Be", {4.1850592788f, 3.1850604423f, 2.7840913457f}, {3.8354398268f, 3.0101260162f, 2.8690088743f}},
  {"Cr", {4.3696828663f, 2.9167024892f, 1.6547005413f}, {5.2064337956f, 4.2313645277f, 3.7549467933f}},
  {"CsI", {2.1449030413f, 1.7023164587f, 1.6624194173f}, {0.0f, 0.0f, 0.0f}},
  {"Cu", {0.2004376970f, 0.9240334304f, 1.1022119527f}, {3.9129485033f, 2.4528477015f, 2.1421879552f}},
  {"Cu2O", {3.5492833755f, 2.9520622449f, 2.7369202137f}, {0.1132179294f, 0.1946659670f, 0.6001681264f}},
  {"CuO", {3.2453822204f, 2.4496293965f, 2.1974114493f}, {0.5202739621f, 0.5707372756f, 0.7172250613f}},
  {"d-C", {2.7112524747f, 2.3185812849f, 2.2288565009f}, {0.0f, 0.0f, 0.0f}},
  {"Hg", {2.3989314904f, 1.4400254917f, 0.9095512090f}, {6.3276269444f, 4.3719414152f, 3.4217899270f}},
  {"HgTe", {4.7795267752f, 3.2309984581f, 2.6600252401f}, {1.6319827058f, 1.5808189339f, 1.7295753852f}},
  {"Ir", {3.0864098394f, 2.0821938440f, 1.6178866805f}, {5.5921510077f, 4.0671757150f, 3.2672611269f}},
  {"K", {0.0640493070f, 0.0464100621f, 0.0381842017f}, {2.1042155920f, 1.3489364357f, 0.9132113889f}},
  {"Li", {0.2657871942f, 0.1956102432f, 0.2209198538f}, {3.5401743407f, 2.3111306542f, 1.6685930000f}},
  {"MgO", {2.0895885542f, 1.6507224525f, 1.5948759692f}, {0.0f, -0.0f, 0.0f}},
  {"Mo", {4.4837010280f, 3.5254578255f, 2.7760769438f}, {4.1111307988f, 3.4208716252f, 3.1506031404f}},
  {"Na", {0.0602665320f, 0.0561412435f, 0.0619909494f}, {3.1792906496f, 2.1124800781f, 1.5790940266f}},
  {"Nb", {3.4201353595f, 2.7901921379f, 2.3955856658f}, {3.4413817900f, 2.7376437930f, 2.5799132708f}},
  {"Ni", {2.3672753521f, 1.6633583302f, 1.4670554172f}, {4.4988329911f, 3.0501643957f, 2.3454274399f}},
  {"Rh", {2.5857954933f, 1.8601866068f, 1.5544279524f}, {6.7822927110f, 4.7029501026f, 3.9760892461f}},
  {"Se-e", {5.7242724833f, 4.1653992967f, 4.0816099264f}, {0.8713747439f, 1.1052845009f, 1.5647788766f}},
  {"Se", {4.0592611085f, 2.8426947380f, 2.8207582835f}, {0.7543791750f, 0.6385150558f, 0.5215872029f}},
  {"SiC", {3.1723450205f, 2.5259677964f, 2.4793623897f}, {0.0000007284f, -0.0000006859f, 0.0000100150f}},
  {"SnTe", {4.5251865890f, 1.9811525984f, 1.2816819226f}, {0.0f, 0.0f, 0.0f}},
  {"Ta", {2.0625846607f, 2.3930915569f, 2.6280684948f}, {2.4080467973f, 1.7413705864f, 1.9470377016f}},
  {"Te-e", {7.5090397678f, 4.2964603080f, 2.3698732430f}, {5.5842076830f, 4.9476231084f, 3.9975145063f}},
  {"Te", {7.3908396088f, 4.4821028985f, 2.6370708478f}, {3.2561412892f, 3.5273908133f, 3.2921683116f}},
  {"ThF4", {1.8307187117f, 1.4422274283f, 1.3876488528f}, {0.0f, 0.0f, 0.0f}},
  {"TiC", {3.7004673762f, 2.8374356509f, 2.5823030278f}, {3.2656905818f, 2.3515586388f, 2.1727857800f}},
  {"TiN", {1.6484691607f, 1.1504482522f, 1.3797795097f}, {3.3684596226f, 1.9434888540f, 1.1020123347f}},
  {"TiO2-e", {3.1065574823f, 2.5131551146f, 2.5823844157f}, {0.0000289537f, -0.0000251484f, 0.0001775555f}},
  {"TiO2", {3.4566203131f, 2.8017076558f, 2.9051485020f}, {0.0001026662f, -0.0000897534f, 0.0006356902f}},
  {"VC", {3.6575665991f, 2.7527298065f, 2.5326814570f}, {3.0683516659f, 2.1986687713f, 1.9631816252f}},
  {"VN", {2.8656011588f, 2.1191817791f, 1.9400767149f}, {3.0323264950f, 2.0561075580f, 1.6162930914f}},
  {"V", {4.2775126218f, 3.5131538236f, 2.7611257461f}, {3.4911844504f, 2.8893580874f, 3.1116965117f}},
  {"W", {4.3707029924f, 3.3002972445f, 2.9982666528f}, {3.5006778591f, 2.6048652781f, 2.2731930614f}},
};

bool find_tungsten_conductor(const std::string& name, TungstenConductorIOR& out) {
  if (name.empty())
    return false;
  std::string key = name;
  for (char& c : key)
    c = char(::tolower(static_cast<unsigned char>(c)));
  for (const auto& item : kTungstenConductors) {
    std::string item_key = item.name;
    for (char& c : item_key)
      c = char(::tolower(static_cast<unsigned char>(c)));
    if (item_key == key) {
      out = item;
      return true;
    }
  }
  return false;
}

PrimitiveLoadResult handle_infinite_sphere(const nlohmann::json& prim, const char* base_dir, SceneData& data) {
  PrimitiveLoadResult r = {};
  std::string emission = prim.value("emission", "");
  bool sample = prim.value("sample", true);
  if (emission.empty() == false && sample) {
    std::string img_path = resolve_path(base_dir, emission);
    uint32_t img_idx = data.add_image(img_path.c_str(), Image::BuildSamplingTable | Image::RepeatU, {}, {1.0f, 1.0f});
    uint32_t sp_white = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));

    auto& profile = data.emitter_profiles.emplace_back(EmitterProfile::Class::Environment);
    profile.emission.spectrum_index = sp_white;
    profile.emission.image_index = img_idx;
    profile.medium_index = kInvalidIndex;

    r.loaded = true;
  } else {
    log::warning("Skipping Tungsten infinite_sphere without emission or sample=false");
  }
  return r;
}

PrimitiveLoadResult handle_infinite_sphere_cap(const nlohmann::json& prim, SceneData& data) {
  PrimitiveLoadResult r = {};
  bool sample = prim.value("sample", true);
  if (sample == false) {
    log::warning("Skipping Tungsten infinite_sphere_cap with sample=false");
    return r;
  }

  float power = prim.value("power", 1.0f);
  float cap_angle = prim.value("cap_angle", 0.0f);

  float3 rotation = {};
  if (prim.contains("transform") && prim["transform"].is_object()) {
    const auto& tr = prim["transform"];
    if (tr.contains("rotation"))
      rotation = json_to_float3(tr["rotation"], {});
  }

  float3 direction = rotate_yxz_deg(float3{0.0f, 1.0f, 0.0f}, rotation);
  direction = normalize(direction);

  auto& d = data.emitter_profiles.emplace_back(EmitterProfile::Class::Directional);
  d.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_luminance({power, power, power}));
  d.emission.image_index = kInvalidIndex;
  d.directional.direction = direction;
  d.directional.angular_size = 2.0f * cap_angle * kPi / 180.0f;

  r.loaded = true;
  return r;
}

PrimitiveLoadResult handle_skydome(const nlohmann::json& prim, SceneData& data, TaskScheduler& scheduler) {
  PrimitiveLoadResult r = {};

  float temperature = prim.value("temperature", 5777.0f);
  float intensity = prim.value("intensity", 2.0f);
  float turbidity = prim.value("turbidity", 3.0f);
  float3 rotation = {};
  if (prim.contains("transform") && prim["transform"].is_object()) {
    const auto& tr = prim["transform"];
    if (tr.contains("rotation"))
      rotation = json_to_float3(tr["rotation"], {});
  }

  float3 sun_dir = normalize(rotate_yxz_deg(float3{0.0f, 1.0f, 0.0f}, rotation));
  const float sun_angular_diameter_deg = 0.53f;

  scattering::Parameters scattering_params = {};
  scattering_params.mie_scale = turbidity / 3.0f;
  scattering_params.altitude = 1000.0f;
  scattering_params.rayleigh_scale = 1.0f;
  scattering_params.ozone_scale = 1.0f;

  constexpr float kDefaultQuality = 0.25f;
  data.add_atmosphere_emitter({scattering_params, kDefaultQuality});
  r.loaded = true;
  return r;
}

PrimitiveLoadResult handle_builtin_primitive(const std::string& type, const float3& translate, const float3& scale, const float3& rotation, uint32_t material_index,
  SceneData& data) {
  PrimitiveLoadResult r = {};
  std::string mesh_name = type + "#" + std::to_string(data.meshes.size());
  if (type == "quad") {
    r.loaded = add_builtin_quad(translate, scale, rotation, material_index, data, mesh_name.c_str());
  } else if (type == "cube") {
    r.loaded = add_builtin_cube(translate, scale, rotation, material_index, data, mesh_name.c_str());
  } else if (type == "disk") {
    r.loaded = add_builtin_disk(translate, scale, rotation, material_index, data, mesh_name.c_str());
  } else if (type == "sphere") {
    r.loaded = add_builtin_sphere(translate, scale, rotation, material_index, data, mesh_name.c_str());
  } else if (type != "mesh") {
    log::warning("Unsupported Tungsten primitive type: %s", type.c_str());
  }
  return r;
}

PrimitiveLoadResult handle_mesh_primitive(const nlohmann::json& prim, const char* base_dir, const std::string& type, const float3& translate, const float3& scale,
  const float3& rotation, uint32_t material_index, SceneData& data, const IORDatabase& database, TaskScheduler& scheduler, Camera& active_camera) {
  PrimitiveLoadResult r = {};

  if (type != "mesh")
    return r;

  uint32_t vertex_start = static_cast<uint32_t>(data.vertices.pos.size());
  uint32_t triangle_start = static_cast<uint32_t>(data.triangles.size());
  uint32_t mesh_start = static_cast<uint32_t>(data.meshes.size());
  (void)triangle_start;

  std::string fname = prim.value("filename", "");
  if (fname.empty() && prim.contains("file"))
    fname = prim["file"].get<std::string>();
  std::string resolved = resolve_path(base_dir, fname);
  if (resolved.empty()) {
    log::warning("Tungsten mesh has no filename, skipping");
    return r;
  }

  const char* ext = get_ext(resolved);
  bool loader_applied_transform = false;
  if (_stricmp(ext, ".obj") == 0) {
    uint32_t flags = load_from_obj_file(resolved.c_str(), "", data, database, scheduler);
    r.loaded = (flags & SceneLoadSucceeded) != 0u;
    r.flags |= (flags & ~SceneLoadSucceeded);
  } else if (_stricmp(ext, ".gltf") == 0) {
    uint32_t flags = load_from_gltf_file(resolved.c_str(), false, data, scheduler, active_camera);
    r.loaded = (flags & SceneLoadSucceeded) != 0u;
    r.flags |= (flags & ~SceneLoadSucceeded);
  } else if (_stricmp(ext, ".glb") == 0) {
    uint32_t flags = load_from_gltf_file(resolved.c_str(), true, data, scheduler, active_camera);
    r.loaded = (flags & SceneLoadSucceeded) != 0u;
    r.flags |= (flags & ~SceneLoadSucceeded);
  } else if (_stricmp(ext, ".wo3") == 0) {
    bool recompute_normals = prim.value("recompute_normals", false);
    loader_applied_transform = true;
    r.loaded = load_wo3_mesh(resolved, translate, scale, rotation, material_index, data, recompute_normals);
  } else {
    log::warning("Unsupported Tungsten mesh format: %s", resolved.c_str());
  }

  if (r.loaded) {
    uint32_t vertex_end = static_cast<uint32_t>(data.vertices.pos.size());
    if (loader_applied_transform == false)
      transform_vertices(data, vertex_start, vertex_end, rotation, scale, translate);

    uint32_t mesh_end = static_cast<uint32_t>(data.meshes.size());
    if (mesh_end > mesh_start)
      recompute_mesh_bounds(data, mesh_start, mesh_end);
  }

  return r;
}

void set_conductor_ior(Material& mtl, const std::string& material_name, SceneData& data, const IORDatabase& database) {
  const IORDefinition* def = database.find_by_name(material_name.c_str(), SpectralDistribution::Conductor);
  if (def != nullptr) {
    mtl.int_ior.cls = def->cls;
    mtl.int_ior.eta_index = data.add_spectrum(def->title.c_str(), def->eta);
    mtl.int_ior.k_index = data.add_spectrum(def->title.c_str(), def->k);
    return;
  }

  TungstenConductorIOR t = {};
  if (find_tungsten_conductor(material_name, t)) {
    float3 eta_rgb = max(float3{}, rgb_to_xyz(t.eta));
    float3 k_rgb = max(float3{}, rgb_to_xyz(t.k));
    SpectralDistribution eta_spd = SpectralDistribution::rgb_reflectance(eta_rgb);
    SpectralDistribution k_spd = SpectralDistribution::rgb_reflectance(k_rgb);
    mtl.int_ior.cls = SpectralDistribution::Conductor;
    mtl.int_ior.eta_index = data.add_spectrum(eta_spd);
    mtl.int_ior.k_index = data.add_spectrum(k_spd);
    return;
  }
}

void set_dielectric_ior(Material& mtl, const std::string& bsdf_name, const nlohmann::json& b, SceneData& data, const IORDatabase& database) {
  auto try_name = [&](const std::string& n) -> bool {
    if (n.empty())
      return false;
    const IORDefinition* def = database.find_by_name(n.c_str(), SpectralDistribution::Dielectric);
    if (def == nullptr)
      return false;
    mtl.int_ior.cls = def->cls;
    mtl.int_ior.eta_index = data.add_spectrum(def->title.c_str(), def->eta);
    mtl.int_ior.k_index = data.add_spectrum(def->title.c_str(), def->k);
    return true;
  };

  if (b.contains("material") && b["material"].is_string()) {
    if (try_name(b["material"].get<std::string>()))
      return;
  }

  if (b.contains("ior") && b["ior"].is_string()) {
    if (try_name(b["ior"].get<std::string>()))
      return;
  }

  float ior = b.value("ior", 1.5f);
  mtl.int_ior.cls = SpectralDistribution::Dielectric;
  mtl.int_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(ior));
  mtl.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
}

uint32_t add_tungsten_material(const std::string& name, const nlohmann::json& b, SceneData& data, const char* base_dir, const IORDatabase& database, bool force_two_sided) {
  uint32_t mat_idx = data.add_material(name.c_str());
  auto& mtl = data.materials[mat_idx];
  bool two_sided = b.value("two_sided", b.value("twoSided", false));
  std::string type = b.value("type", "lambert");
  mtl.ext_ior.cls = SpectralDistribution::Dielectric;
  mtl.ext_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
  mtl.ext_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
  mtl.int_ior.cls = SpectralDistribution::Dielectric;
  mtl.int_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.5f));
  mtl.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));

  if (type == "smooth_coat") {
    auto find_material_index = [&](const std::string& id) -> uint32_t {
      auto it = data.material_mapping.find(id);
      if (it != data.material_mapping.end())
        return it->second;
      return kInvalidIndex;
    };

    uint32_t base_material = kInvalidIndex;
    if (b.contains("substrate")) {
      const auto& coat_base = b["substrate"];
      if (coat_base.is_string()) {
        base_material = find_material_index(coat_base.get<std::string>());
      } else if (coat_base.is_object()) {
        std::string base_name = name.empty() ? std::string("__coat_base_") + std::to_string(data.materials.size()) : name + "__coat_base";
        base_material = add_tungsten_material(base_name, coat_base, data, base_dir, database, force_two_sided);
      }
    }

    if (base_material != kInvalidIndex) {
      data.materials[mat_idx] = data.materials[base_material];
    }

    float coat_thickness = b.value("thickness", 0.0f);
    float coat_ior = b.value("ior", 1.5f);
    float thickness_nm = coat_thickness * 100.0f;
    if (thickness_nm > 0.0f) {
      mtl.thinfilm.min_thickness = thickness_nm;
      mtl.thinfilm.max_thickness = thickness_nm;
      mtl.thinfilm.ior.cls = SpectralDistribution::Dielectric;
      mtl.thinfilm.ior.eta_index = data.add_spectrum(SpectralDistribution::constant(coat_ior));
      mtl.thinfilm.ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
    }

    if (two_sided || force_two_sided)
      mtl.two_sided = 1u;
    return mat_idx;
  }

  if (type == "lambert" || type == "oren_nayar") {
    mtl.cls = Material::Class::Diffuse;
    tungsten_set_albedo(mtl, tungsten_albedo_json(b), data, base_dir);
    float rough = b.value("roughness", 0.0f);
    mtl.roughness.value = {rough, rough};
  } else if ((type == "plastic") || (type == "rough_plastic")) {
    mtl.cls = Material::Class::Plastic;
    tungsten_set_albedo(mtl, tungsten_albedo_json(b), data, base_dir);
    float rough = b.value("roughness", 0.0f);
    mtl.roughness.value = {rough, rough};
    mtl.metalness.value = {0.0f, 0.0f};
    mtl.reflectance.image_index = kInvalidIndex;
    mtl.reflectance.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
    set_dielectric_ior(mtl, name, b, data, database);
  } else if (type == "mirror") {
    mtl.cls = Material::Class::Mirror;
    tungsten_set_albedo(mtl, tungsten_albedo_json(b), data, base_dir);
    mtl.roughness.value = {0.0f, 0.0f};
    mtl.metalness.value = {1.0f, 1.0f};
  } else if (type == "thinsheet") {
    mtl.cls = Material::Class::Thinfilm;
    tungsten_set_albedo(mtl, tungsten_albedo_json(b), data, base_dir);
    set_dielectric_ior(mtl, name, b, data, database);
  } else if ((type == "conductor") || (type == "rough_conductor")) {
    mtl.cls = Material::Class::Conductor;
    tungsten_set_albedo(mtl, tungsten_albedo_json(b), data, base_dir);
    float rough = b.value("roughness", 0.0f);
    mtl.roughness.value = {rough, rough};
    mtl.metalness.value = {1.0f, 1.0f};
    std::string mat_name = b.value("material", "");
    if (mat_name.empty() == false)
      set_conductor_ior(mtl, mat_name, data, database);
  } else if ((type == "dielectric") || (type == "rough_dielectric")) {
    mtl.cls = Material::Class::Dielectric;
    set_dielectric_ior(mtl, name, b, data, database);
    mtl.transmission.value = {1.0f, 1.0f, 1.0f, 1.0f};
    mtl.roughness.value = {b.value("roughness", 0.0f), b.value("roughness", 0.0f)};
    mtl.reflectance.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
    mtl.reflectance.image_index = kInvalidIndex;
    mtl.scattering.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
    mtl.scattering.image_index = kInvalidIndex;
    tungsten_set_transmission(mtl, tungsten_albedo_json(b), data, base_dir);
  } else if (type == "transparency") {
    mtl.cls = Material::Class::Boundary;
    mtl.transmission.value = {1.0f, 1.0f, 1.0f, 1.0f};
    mtl.reflectance.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
    mtl.reflectance.image_index = kInvalidIndex;
    mtl.scattering.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
    mtl.scattering.image_index = kInvalidIndex;
    tungsten_set_transmission(mtl, tungsten_albedo_json(b), data, base_dir);
    if (b.contains("alpha")) {
      uint32_t alpha_tex = resolve_tungsten_texture(b["alpha"], data, base_dir);
      if (alpha_tex != kInvalidIndex) {
        mtl.transmission.image_index = alpha_tex;
        mtl.transmission.channel = 0u;
      }
    }
  } else {
    mtl.cls = Material::Class::Diffuse;
    tungsten_set_albedo(mtl, tungsten_albedo_json(b), data, base_dir);
  }

  if (two_sided || force_two_sided)
    mtl.two_sided = 1u;

  return mat_idx;
}

float triangle_area(const Triangle& t, const std::vector<float3>& pos) {
  const float3& p0 = pos[t.i[0]];
  const float3& p1 = pos[t.i[1]];
  const float3& p2 = pos[t.i[2]];
  return 0.5f * length(cross(p1 - p0, p2 - p0));
}

float triangles_area(const SceneData& data, uint32_t tri_start, uint32_t tri_end) {
  float area = 0.0f;
  uint32_t end = tri_end;
  uint32_t count = static_cast<uint32_t>(data.triangles.size());
  if (end > count)
    end = count;
  for (uint32_t i = tri_start; i < end; ++i)
    area += triangle_area(data.triangles[i], data.vertices.pos);
  return area;
}

void transform_vertices(SceneData& data, uint32_t vertex_start, uint32_t vertex_end, const float3& rotation_deg, const float3& scale, const float3& translate) {
  if (vertex_end <= vertex_start)
    return;

  bool has_rotation = (rotation_deg.x != 0.0f) || (rotation_deg.y != 0.0f) || (rotation_deg.z != 0.0f);
  bool has_scale = (scale.x != 1.0f) || (scale.y != 1.0f) || (scale.z != 1.0f);
  bool has_translate = (translate.x != 0.0f) || (translate.y != 0.0f) || (translate.z != 0.0f);
  if ((has_rotation == false) && (has_scale == false) && (has_translate == false))
    return;

  float3 inv_scale = {
    scale.x != 0.0f ? (1.0f / scale.x) : 0.0f,
    scale.y != 0.0f ? (1.0f / scale.y) : 0.0f,
    scale.z != 0.0f ? (1.0f / scale.z) : 0.0f,
  };

  for (uint32_t i = vertex_start; i < vertex_end; ++i) {
    float3 p = data.vertices.pos[i];
    if (has_scale) {
      p.x *= scale.x;
      p.y *= scale.y;
      p.z *= scale.z;
    }
    if (has_rotation)
      p = rotate_yxz_deg(p, rotation_deg);
    if (has_translate)
      p += translate;
    data.vertices.pos[i] = p;

    float3 n = data.vertices.nrm[i];
    if (has_scale) {
      n *= inv_scale;
    }
    if (has_rotation)
      n = rotate_yxz_deg(n, rotation_deg);
    float ln = length(n);
    if (ln > kEpsilon)
      n /= ln;
    data.vertices.nrm[i] = normalize(n);

    float3 t = data.vertices.tan[i];
    if (has_scale) {
      t.x *= scale.x;
      t.y *= scale.y;
      t.z *= scale.z;
    }
    if (has_rotation)
      t = rotate_yxz_deg(t, rotation_deg);
    float lt = length(t);
    if (lt > kEpsilon)
      t /= lt;
    data.vertices.tan[i] = normalize(t);

    float3 b = data.vertices.btn[i];
    if (has_scale) {
      b.x *= scale.x;
      b.y *= scale.y;
      b.z *= scale.z;
    }
    if (has_rotation)
      b = rotate_yxz_deg(b, rotation_deg);
    float lb = length(b);
    if (lb > kEpsilon)
      b /= lb;
    data.vertices.btn[i] = normalize(b);
  }
}

void reserve_mesh_vertices(SceneData& data, uint32_t extra_vertices) {
  size_t required = data.vertices.pos.size() + static_cast<size_t>(extra_vertices);
  if (data.vertices.pos.capacity() < required)
    data.vertices.pos.reserve(required);
  if (data.vertices.nrm.capacity() < required)
    data.vertices.nrm.reserve(required);
  if (data.vertices.tan.capacity() < required)
    data.vertices.tan.reserve(required);
  if (data.vertices.btn.capacity() < required)
    data.vertices.btn.reserve(required);
  if (data.vertices.tex.capacity() < required)
    data.vertices.tex.reserve(required);
}

void reserve_mesh_triangles(SceneData& data, uint32_t extra_triangles) {
  size_t required = data.triangles.size() + static_cast<size_t>(extra_triangles);
  if (data.triangles.capacity() < required)
    data.triangles.reserve(required);
}

void recompute_mesh_bounds(SceneData& data, uint32_t mesh_start, uint32_t mesh_end) {
  uint32_t mesh_count = static_cast<uint32_t>(data.meshes.size());
  if (mesh_start >= mesh_count)
    return;
  if (mesh_end > mesh_count)
    mesh_end = mesh_count;

  for (uint32_t m = mesh_start; m < mesh_end; ++m) {
    auto& mesh = data.meshes[m];
    uint32_t tri_begin = mesh.triangle_offset;
    uint32_t tri_end = tri_begin + mesh.triangle_count;
    uint32_t tri_count = static_cast<uint32_t>(data.triangles.size());
    if (tri_end > tri_count)
      tri_end = tri_count;

    float3 bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
    float3 bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};

    for (uint32_t t = tri_begin; t < tri_end; ++t) {
      const Triangle& tri = data.triangles[t];
      for (uint32_t k = 0; k < 3; ++k) {
        const float3& p = data.vertices.pos[tri.i[k]];
        bbox_min = min(bbox_min, p);
        bbox_max = max(bbox_max, p);
      }
    }

    mesh.bbox_min = bbox_min;
    mesh.bbox_max = bbox_max;
  }
}

void set_emission_from_json(Material& mtl, const nlohmann::json& v, SceneData& data, const char* base_dir, float scale = 1.0f) {
  float3 scaled_white = {scale, scale, scale};
  if (v.is_string()) {
    uint32_t tex_idx = resolve_tungsten_texture(v, data, base_dir);
    if (tex_idx != kInvalidIndex) {
      mtl.emission.image_index = tex_idx;
      mtl.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance(scaled_white));
      return;
    }
  }
  if (v.is_array() && v.size() >= 3) {
    float3 c = {float(v[0].get<double>()), float(v[1].get<double>()), float(v[2].get<double>())};
    c *= scale;
    mtl.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance(c));
    mtl.emission.image_index = kInvalidIndex;
    return;
  }
  if (v.is_number()) {
    float s = float(v.get<double>()) * scale;
    mtl.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({s, s, s}));
    mtl.emission.image_index = kInvalidIndex;
    return;
  }
}

bool add_builtin_quad(const float3& translate, const float3& scale, const float3& rotation_deg, uint32_t material_index, SceneData& data, const char* name) {
  const uint32_t vertex_offset = static_cast<uint32_t>(data.vertices.pos.size());
  float3 edge0 = {scale.x, 0.0f, 0.0f};
  float3 edge1 = {0.0f, 0.0f, scale.z};
  edge0 = rotate_yxz_deg(edge0, rotation_deg);
  edge1 = rotate_yxz_deg(edge1, rotation_deg);

  float3 nrm = cross(edge1, edge0);
  float len = length(nrm);
  if (len > 0.0f)
    nrm /= len;
  else
    nrm = {0.0f, 1.0f, 0.0f};

  float3 tan = normalize(edge0);
  float3 btn = normalize(cross(nrm, tan));

  float3 base = translate - 0.5f * edge0 - 0.5f * edge1;

  std::array<float3, 4> final_pos = {
    base,
    base + edge0,
    base + edge0 + edge1,
    base + edge1,
  };
  std::array<float2, 4> uv = {float2{0.0f, 1.0f}, float2{1.0f, 1.0f}, float2{1.0f, 0.0f}, float2{0.0f, 0.0f}};

  float3 bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
  float3 bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};

  for (size_t i = 0; i < final_pos.size(); ++i) {
    const float3& p = final_pos[i];
    data.vertices.pos.emplace_back(p);
    data.vertices.nrm.emplace_back(nrm);
    data.vertices.tan.emplace_back(tan);
    data.vertices.btn.emplace_back(btn);
    data.vertices.tex.emplace_back(uv[i]);
    bbox_min = min(bbox_min, p);
    bbox_max = max(bbox_max, p);
  }

  Triangle t0{};
  t0.i[0] = vertex_offset + 0;
  t0.i[1] = vertex_offset + 2;
  t0.i[2] = vertex_offset + 1;
  t0.material_index = material_index;

  Triangle t1{};
  t1.i[0] = vertex_offset + 0;
  t1.i[1] = vertex_offset + 3;
  t1.i[2] = vertex_offset + 2;
  t1.material_index = material_index;

  if (!validate_triangle(t0, data.vertices.pos) || !validate_triangle(t1, data.vertices.pos)) {
    data.vertices.pos.resize(vertex_offset);
    data.vertices.nrm.resize(vertex_offset);
    data.vertices.tan.resize(vertex_offset);
    data.vertices.btn.resize(vertex_offset);
    data.vertices.tex.resize(vertex_offset);
    return false;
  }

  const uint32_t tri_start = static_cast<uint32_t>(data.triangles.size());
  data.triangles.emplace_back(t0);
  data.triangles.emplace_back(t1);

  const char* mesh_name = (name != nullptr) && (name[0] != 0) ? name : "quad";
  data.add_mesh(mesh_name, tri_start, 2, bbox_min, bbox_max);
  return true;
}

bool add_builtin_cube(const float3& translate, const float3& scale, const float3& rotation_deg, uint32_t material_index, SceneData& data, const char* name) {
  const float3 h = 0.5f * scale;
  const uint32_t vertex_offset = static_cast<uint32_t>(data.vertices.pos.size());

  struct VDef {
    float3 p;
    float3 n;
    float2 uv;
  };

  const float3 px = {1.0f, 0.0f, 0.0f};
  const float3 nx = {-1.0f, 0.0f, 0.0f};
  const float3 py = {0.0f, 1.0f, 0.0f};
  const float3 ny = {0.0f, -1.0f, 0.0f};
  const float3 pz = {0.0f, 0.0f, 1.0f};
  const float3 nz = {0.0f, 0.0f, -1.0f};

  std::vector<VDef> verts = {
    // +X
    {{h.x, -h.y, -h.z}, px, {0.0f, 1.0f}},
    {{h.x, h.y, -h.z}, px, {0.0f, 0.0f}},
    {{h.x, h.y, h.z}, px, {1.0f, 0.0f}},
    {{h.x, -h.y, h.z}, px, {1.0f, 1.0f}},
    // -X
    {{-h.x, -h.y, h.z}, nx, {0.0f, 1.0f}},
    {{-h.x, h.y, h.z}, nx, {0.0f, 0.0f}},
    {{-h.x, h.y, -h.z}, nx, {1.0f, 0.0f}},
    {{-h.x, -h.y, -h.z}, nx, {1.0f, 1.0f}},
    // +Y
    {{-h.x, h.y, -h.z}, py, {0.0f, 1.0f}},
    {{-h.x, h.y, h.z}, py, {0.0f, 0.0f}},
    {{h.x, h.y, h.z}, py, {1.0f, 0.0f}},
    {{h.x, h.y, -h.z}, py, {1.0f, 1.0f}},
    // -Y
    {{-h.x, -h.y, h.z}, ny, {0.0f, 1.0f}},
    {{-h.x, -h.y, -h.z}, ny, {0.0f, 0.0f}},
    {{h.x, -h.y, -h.z}, ny, {1.0f, 0.0f}},
    {{h.x, -h.y, h.z}, ny, {1.0f, 1.0f}},
    // +Z
    {{h.x, -h.y, h.z}, pz, {0.0f, 1.0f}},
    {{h.x, h.y, h.z}, pz, {0.0f, 0.0f}},
    {{-h.x, h.y, h.z}, pz, {1.0f, 0.0f}},
    {{-h.x, -h.y, h.z}, pz, {1.0f, 1.0f}},
    // -Z
    {{-h.x, -h.y, -h.z}, nz, {0.0f, 1.0f}},
    {{-h.x, h.y, -h.z}, nz, {0.0f, 0.0f}},
    {{h.x, h.y, -h.z}, nz, {1.0f, 0.0f}},
    {{h.x, -h.y, -h.z}, nz, {1.0f, 1.0f}},
  };

  float3 bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
  float3 bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};

  for (const auto& v : verts) {
    float3 p = rotate_yxz_deg(v.p, rotation_deg);
    p += translate;
    float3 n = rotate_yxz_deg(v.n, rotation_deg);
    float ln = length(n);
    if (ln > 0.0f)
      n /= ln;
    data.vertices.pos.emplace_back(p);
    data.vertices.nrm.emplace_back(n);
    data.vertices.tan.emplace_back(float3{0.0f, 0.0f, 0.0f});
    data.vertices.btn.emplace_back(float3{0.0f, 0.0f, 0.0f});
    data.vertices.tex.emplace_back(v.uv);
    bbox_min = min(bbox_min, p);
    bbox_max = max(bbox_max, p);
  }

  static const uint32_t idx[] = {
    0, 1, 2, 0, 2, 3,        // +X
    4, 5, 6, 4, 6, 7,        // -X
    8, 9, 10, 8, 10, 11,     // +Y
    12, 13, 14, 12, 14, 15,  // -Y
    16, 17, 18, 16, 18, 19,  // +Z
    20, 21, 22, 20, 22, 23   // -Z
  };

  uint32_t tri_start = static_cast<uint32_t>(data.triangles.size());
  for (size_t i = 0; i < sizeof(idx) / sizeof(idx[0]); i += 3) {
    Triangle& tri = data.triangles.emplace_back();
    tri.i[0] = vertex_offset + idx[i + 0];
    tri.i[1] = vertex_offset + idx[i + 1];
    tri.i[2] = vertex_offset + idx[i + 2];
    tri.material_index = material_index;
    if (validate_triangle(tri, data.vertices.pos) == false) {
      data.triangles.pop_back();
      continue;
    }
  }

  uint32_t tri_end = static_cast<uint32_t>(data.triangles.size());
  if (tri_end == tri_start)
    return false;

  const char* mesh_name = (name != nullptr) && (name[0] != 0) ? name : "cube";
  data.add_mesh(mesh_name, tri_start, tri_end - tri_start, bbox_min, bbox_max);
  return true;
}

bool add_builtin_sphere(const float3& translate, const float3& scale, const float3& rotation_deg, uint32_t material_index, SceneData& data, const char* name) {
  float radius = max(max(fabsf(scale.x), fabsf(scale.y)), fabsf(scale.z));
  if (radius <= kEpsilon)
    return false;

  float target_edge = 0.025f;
  int subdiv = static_cast<int>(ceilf(log2f(max(radius / target_edge, 1.0f))));
  subdiv = clamp(subdiv, 0, 6);

  struct EdgeKey {
    uint32_t a;
    uint32_t b;
    bool operator==(const EdgeKey& other) const {
      return (a == other.a) && (b == other.b);
    }
  };
  struct EdgeHash {
    size_t operator()(const EdgeKey& k) const {
      return (static_cast<size_t>(k.a) << 32) ^ static_cast<size_t>(k.b);
    }
  };

  std::vector<float3> verts = {
    {-1, kGoldenRatio, 0},
    {1, kGoldenRatio, 0},
    {-1, -kGoldenRatio, 0},
    {1, -kGoldenRatio, 0},
    {0, -1, kGoldenRatio},
    {0, 1, kGoldenRatio},
    {0, -1, -kGoldenRatio},
    {0, 1, -kGoldenRatio},
    {kGoldenRatio, 0, -1},
    {kGoldenRatio, 0, 1},
    {-kGoldenRatio, 0, -1},
    {-kGoldenRatio, 0, 1},
  };
  for (auto& v : verts)
    v = normalize(v);

  std::vector<uint3> faces = {
    {0, 11, 5},
    {0, 5, 1},
    {0, 1, 7},
    {0, 7, 10},
    {0, 10, 11},
    {1, 5, 9},
    {5, 11, 4},
    {11, 10, 2},
    {10, 7, 6},
    {7, 1, 8},
    {3, 9, 4},
    {3, 4, 2},
    {3, 2, 6},
    {3, 6, 8},
    {3, 8, 9},
    {4, 9, 5},
    {2, 4, 11},
    {6, 2, 10},
    {8, 6, 7},
    {9, 8, 1},
  };

  auto midpoint = [&](uint32_t a, uint32_t b, std::unordered_map<EdgeKey, uint32_t, EdgeHash>& cache) -> uint32_t {
    EdgeKey key{min(a, b), max(a, b)};
    auto it = cache.find(key);
    if (it != cache.end())
      return it->second;
    float3 m = normalize(verts[a] + verts[b]);
    uint32_t idx = static_cast<uint32_t>(verts.size());
    verts.emplace_back(m);
    cache[key] = idx;
    return idx;
  };

  for (int s = 0; s < subdiv; ++s) {
    std::unordered_map<EdgeKey, uint32_t, EdgeHash> cache;
    std::vector<uint3> new_faces;
    new_faces.reserve(faces.size() * 4);
    for (const auto& f : faces) {
      uint32_t a = f.x;
      uint32_t b = f.y;
      uint32_t c = f.z;
      uint32_t ab = midpoint(a, b, cache);
      uint32_t bc = midpoint(b, c, cache);
      uint32_t ca = midpoint(c, a, cache);
      new_faces.push_back({a, ab, ca});
      new_faces.push_back({b, bc, ab});
      new_faces.push_back({c, ca, bc});
      new_faces.push_back({ab, bc, ca});
    }
    faces.swap(new_faces);
  }

  uint32_t vertex_offset = static_cast<uint32_t>(data.vertices.pos.size());
  float3 bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
  float3 bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};

  for (const auto& v : verts) {
    float3 p = v * radius;
    p = rotate_yxz_deg(p, rotation_deg);
    p += translate;
    float3 n = rotate_yxz_deg(v, rotation_deg);
    n = (length(n) > kEpsilon) ? (n / length(n)) : float3{0.0f, 1.0f, 0.0f};
    data.vertices.pos.emplace_back(p);
    data.vertices.nrm.emplace_back(n);
    data.vertices.tan.emplace_back(float3{0.0f, 0.0f, 0.0f});
    data.vertices.btn.emplace_back(float3{0.0f, 0.0f, 0.0f});
    data.vertices.tex.emplace_back(float2{0.0f, 0.0f});
    bbox_min = min(bbox_min, p);
    bbox_max = max(bbox_max, p);
  }

  uint32_t tri_start = static_cast<uint32_t>(data.triangles.size());
  for (const auto& f : faces) {
    Triangle t{};
    t.i[0] = vertex_offset + f.x;
    t.i[1] = vertex_offset + f.y;
    t.i[2] = vertex_offset + f.z;
    t.material_index = material_index;
    if (validate_triangle(t, data.vertices.pos) == false)
      continue;
    data.triangles.emplace_back(t);
  }

  uint32_t tri_end = static_cast<uint32_t>(data.triangles.size());
  if (tri_end == tri_start)
    return false;

  const char* mesh_name = (name != nullptr) && (name[0] != 0) ? name : "sphere";
  data.add_mesh(mesh_name, tri_start, tri_end - tri_start, bbox_min, bbox_max);
  return true;
}
bool add_builtin_disk(const float3& translate, const float3& scale, const float3& rotation_deg, uint32_t material_index, SceneData& data, const char* name) {
  float radius = 0.5f * max(fabsf(scale.x), fabsf(scale.z));
  if (radius <= kEpsilon)
    return false;

  float circumference = 2.0f * kPi * radius;
  float target_edge = 0.025f;
  uint32_t segments = static_cast<uint32_t>(ceilf(circumference / target_edge));
  segments = clamp(segments, 12u, 256u);

  const uint32_t vertex_offset = static_cast<uint32_t>(data.vertices.pos.size());
  float3 up = rotate_yxz_deg(float3{0.0f, 1.0f, 0.0f}, rotation_deg);
  float3 tan = rotate_yxz_deg(float3{1.0f, 0.0f, 0.0f}, rotation_deg);
  float3 btn = rotate_yxz_deg(float3{0.0f, 0.0f, 1.0f}, rotation_deg);
  up = (length(up) > kEpsilon) ? (up / length(up)) : float3{0.0f, 1.0f, 0.0f};
  tan = (length(tan) > kEpsilon) ? (tan / length(tan)) : float3{1.0f, 0.0f, 0.0f};
  btn = (length(btn) > kEpsilon) ? (btn / length(btn)) : float3{0.0f, 0.0f, 1.0f};

  data.vertices.pos.emplace_back(translate);
  data.vertices.nrm.emplace_back(up);
  data.vertices.tan.emplace_back(tan);
  data.vertices.btn.emplace_back(btn);
  data.vertices.tex.emplace_back(float2{0.5f, 0.5f});

  float3 bbox_min = translate;
  float3 bbox_max = translate;

  for (uint32_t i = 0; i < segments; ++i) {
    float angle = (2.0f * kPi * float(i)) / float(segments);
    float x = radius * cosf(angle);
    float z = radius * sinf(angle);
    float3 p = rotate_yxz_deg(float3{x, 0.0f, z}, rotation_deg) + translate;

    data.vertices.pos.emplace_back(p);
    data.vertices.nrm.emplace_back(up);
    data.vertices.tan.emplace_back(tan);
    data.vertices.btn.emplace_back(btn);
    data.vertices.tex.emplace_back(float2{0.5f + 0.5f * (x / radius), 0.5f + 0.5f * (z / radius)});

    bbox_min = min(bbox_min, p);
    bbox_max = max(bbox_max, p);
  }

  const uint32_t tri_start = static_cast<uint32_t>(data.triangles.size());
  for (uint32_t i = 0; i < segments; ++i) {
    uint32_t i0 = vertex_offset;  // center
    uint32_t i1 = vertex_offset + 1 + i;
    uint32_t i2 = vertex_offset + 1 + ((i + 1) % segments);

    Triangle t{};
    t.i[0] = i0;
    t.i[1] = i1;
    t.i[2] = i2;
    t.material_index = material_index;

    if (validate_triangle(t, data.vertices.pos) == false)
      continue;

    data.triangles.emplace_back(t);
  }

  uint32_t tri_end = static_cast<uint32_t>(data.triangles.size());
  if (tri_end == tri_start)
    return false;

  const char* mesh_name = (name != nullptr) && (name[0] != 0) ? name : "disk";
  data.add_mesh(mesh_name, tri_start, tri_end - tri_start, bbox_min, bbox_max);
  return true;
}

bool load_tungsten_camera(const nlohmann::json& js, SceneData& data, Camera& active_camera) {
  if (js.contains("camera") == false || js["camera"].is_object() == false)
    return false;

  const auto& cam = js["camera"];
  float3 pos = active_camera.position;
  float3 target = pos + active_camera.direction;
  float3 up = kWorldUp;
  float fov = get_camera_fov(active_camera);
  uint2 film = active_camera.film_size;

  if (cam.contains("transform") && cam["transform"].is_object()) {
    const auto& tr = cam["transform"];
    if (tr.contains("position"))
      pos = json_to_float3(tr["position"], pos);
    if (tr.contains("target"))
      target = json_to_float3(tr["target"], target);
    if (tr.contains("look_at"))
      target = json_to_float3(tr["look_at"], target);
    if (tr.contains("up"))
      up = json_to_float3(tr["up"], up);
  }

  if (cam.contains("position"))
    pos = json_to_float3(cam["position"], pos);
  if (cam.contains("target"))
    target = json_to_float3(cam["target"], target);
  if (cam.contains("look_at"))
    target = json_to_float3(cam["look_at"], target);
  if (cam.contains("up"))
    up = json_to_float3(cam["up"], up);
  if (cam.contains("fov") && cam["fov"].is_number())
    fov = float(cam["fov"].get<double>());
  if (cam.contains("resolution")) {
    const auto& res = cam["resolution"];
    if (res.is_array() && res.size() >= 2) {
      film.x = static_cast<uint32_t>(res[0].get<int64_t>());
      film.y = static_cast<uint32_t>(res[1].get<int64_t>());
    } else if (res.is_number_integer()) {
      uint32_t r = static_cast<uint32_t>(std::max<int64_t>(1, res.get<int64_t>()));
      film = {r, r};
    }
  }

  if (up.x == 0.0f && up.y == 0.0f && up.z == 0.0f) {
    up = kWorldUp;
  }

  if ((film.x == 0u) || (film.y == 0u)) {
    film = {1280u, 720u};
  }

  auto& entry = data.cameras.emplace_back();
  entry.id = "tungsten_camera";
  entry.active = data.cameras.size() == 1;
  build_camera(entry.cam, pos, normalize(target - pos), normalize(up), film, fov);
  active_camera = entry.cam;
  return true;
}

void recompute_vertex_normals(SceneData& data, uint32_t vertex_start, uint32_t vertex_end, uint32_t tri_start, uint32_t tri_end) {
  if (vertex_end <= vertex_start)
    return;
  std::vector<float3> accum(vertex_end - vertex_start, float3{});
  for (uint32_t t = tri_start; t < tri_end; ++t) {
    const Triangle& tri = data.triangles[t];
    const float3& p0 = data.vertices.pos[tri.i[0]];
    const float3& p1 = data.vertices.pos[tri.i[1]];
    const float3& p2 = data.vertices.pos[tri.i[2]];
    float3 n = cross(p1 - p0, p2 - p0);
    if (length(n) <= kEpsilon)
      continue;
    accum[tri.i[0] - vertex_start] += n;
    accum[tri.i[1] - vertex_start] += n;
    accum[tri.i[2] - vertex_start] += n;
  }
  for (uint32_t i = vertex_start; i < vertex_end; ++i) {
    float3 n = accum[i - vertex_start];
    float ln = length(n);
    if (ln > kEpsilon)
      n /= ln;
    else
      n = float3{0.0f, 1.0f, 0.0f};
    data.vertices.nrm[i] = n;
  }
}

bool load_wo3_mesh(const std::string& resolved, const float3& translate, const float3& scale, const float3& rotation_deg, uint32_t material_index, SceneData& data,
  bool recompute_normals) {
  std::ifstream fin(resolved, std::ios::binary);
  if (fin.good() == false) {
    log::warning("Failed to open Tungsten mesh %s", resolved.c_str());
    return false;
  }

  struct Wo3Vertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
  };
  struct Wo3Triangle {
    uint32_t v0, v1, v2;
    int32_t material;
  };

  uint64_t vert_count = 0;
  if (!fin.read(reinterpret_cast<char*>(&vert_count), sizeof(uint64_t))) {
    log::warning("Failed to read vertex count from %s", resolved.c_str());
    return false;
  }

  std::vector<Wo3Vertex> vbuf(vert_count);
  if (!fin.read(reinterpret_cast<char*>(vbuf.data()), vbuf.size() * sizeof(Wo3Vertex))) {
    log::warning("Failed to read vertices from %s", resolved.c_str());
    return false;
  }

  uint64_t tri_count = 0;
  if (!fin.read(reinterpret_cast<char*>(&tri_count), sizeof(uint64_t))) {
    log::warning("Failed to read triangle count from %s", resolved.c_str());
    return false;
  }

  std::vector<Wo3Triangle> tbuf(tri_count);
  if (!fin.read(reinterpret_cast<char*>(tbuf.data()), tbuf.size() * sizeof(Wo3Triangle))) {
    log::warning("Failed to read triangles from %s", resolved.c_str());
    return false;
  }

  uint32_t triangle_start = static_cast<uint32_t>(data.triangles.size());
  float3 bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
  float3 bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};

  uint32_t vertex_offset = static_cast<uint32_t>(data.vertices.pos.size());

  data.vertices.pos.reserve(vertex_offset + static_cast<uint32_t>(vbuf.size()));
  data.vertices.nrm.reserve(vertex_offset + static_cast<uint32_t>(vbuf.size()));
  data.vertices.tan.reserve(vertex_offset + static_cast<uint32_t>(vbuf.size()));
  data.vertices.btn.reserve(vertex_offset + static_cast<uint32_t>(vbuf.size()));
  data.vertices.tex.reserve(vertex_offset + static_cast<uint32_t>(vbuf.size()));

  for (const auto& v : vbuf) {
    float3 pos = {v.px * scale.x, v.py * scale.y, v.pz * scale.z};
    pos = rotate_yxz_deg(pos, rotation_deg);
    pos += translate;
    float3 nrm = {v.nx, v.ny, v.nz};
    nrm = rotate_yxz_deg(nrm, rotation_deg);
    float ln = length(nrm);
    if (ln > kEpsilon)
      nrm /= ln;
    float2 uv = {v.u, 1.0f - v.v};
    data.vertices.pos.emplace_back(pos);
    data.vertices.nrm.emplace_back(nrm);
    data.vertices.tan.emplace_back(float3{0.0f, 0.0f, 0.0f});
    data.vertices.btn.emplace_back(float3{0.0f, 0.0f, 0.0f});
    data.vertices.tex.emplace_back(uv);
    bbox_min = min(bbox_min, pos);
    bbox_max = max(bbox_max, pos);
  }

  data.triangles.reserve(data.triangles.size() + tbuf.size());

  for (const auto& tri_in : tbuf) {
    Triangle& tri = data.triangles.emplace_back();
    tri.i[0] = vertex_offset + tri_in.v0;
    tri.i[1] = vertex_offset + tri_in.v1;
    tri.i[2] = vertex_offset + tri_in.v2;
    tri.material_index = material_index;
    if (validate_triangle(tri, data.vertices.pos) == false) {
      data.triangles.pop_back();
      continue;
    }
  }

  uint32_t triangle_end = static_cast<uint32_t>(data.triangles.size());
  uint32_t triangle_count = triangle_end - triangle_start;
  if (triangle_count == 0)
    return false;

  if (recompute_normals) {
    uint32_t vertex_end = static_cast<uint32_t>(data.vertices.pos.size());
    recompute_vertex_normals(data, vertex_offset, vertex_end, triangle_start, triangle_end);
  }

  std::string mesh_name = std::filesystem::path(resolved).stem().string();
  data.add_mesh(mesh_name.c_str(), triangle_start, triangle_count, bbox_min, bbox_max);
  return true;
}

void load_tungsten_media(const nlohmann::json& js, SceneData& data) {
  if (js.contains("media") == false || js["media"].is_array() == false)
    return;

  uint32_t loaded = 0;
  for (const auto& m : js["media"]) {
    if (m.is_object() == false)
      continue;
    std::string name = m.value("name", "");
    if (name.empty())
      name = std::string("medium-") + std::to_string(data.mediums.array_size());

    std::string type = m.value("type", "homogeneous");
    if (type != "homogeneous") {
      log::warning("Unsupported Tungsten medium type: %s", type.c_str());
      continue;
    }

    float g = 0.0f;
    if (m.contains("phase_function") && m["phase_function"].is_object()) {
      const auto& pf = m["phase_function"];
      std::string pf_type = pf.value("type", "isotropic");
      if (pf_type == "henyey_greenstein")
        g = pf.value("g", 0.0f);
    }

    SpectralDistribution s_a = json_to_rgb_spectrum(m["sigma_a"], 0.0f);
    SpectralDistribution s_s = json_to_rgb_spectrum(m["sigma_s"], 0.0f);

    data.add_medium(Medium::Class::Homogeneous, name.c_str(), nullptr, s_a, s_s, g, true);
    ++loaded;
  }
}

uint32_t count_tungsten_meshes(const nlohmann::json& js) {
  if (js.contains("primitives") == false || js["primitives"].is_array() == false)
    return 0u;
  uint32_t count = 0;
  for (const auto& prim : js["primitives"]) {
    if (prim.is_object() == false)
      continue;
    std::string type = prim.value("type", "");
    if (type == "mesh")
      ++count;
  }
  return count;
}

uint32_t load_tungsten_primitives(const nlohmann::json& js, const char* base_dir, const std::unordered_map<std::string, uint32_t>& bsdf_to_mat, SceneData& data,
  const IORDatabase& database, TaskScheduler& scheduler, Camera& active_camera, bool force_two_sided) {
  uint32_t load_flags = SceneLoadFailed;
  bool primitives_loaded = false;

  if (js.contains("primitives") == false || js["primitives"].is_array() == false)
    return load_flags;

  for (const auto& prim : js["primitives"]) {
    if (prim.is_object() == false)
      continue;

    std::string type = prim.value("type", "");
    if (type == "infinite_sphere") {
      PrimitiveLoadResult r = handle_infinite_sphere(prim, base_dir, data);
      primitives_loaded = primitives_loaded || r.loaded;
      load_flags |= r.flags;
      continue;
    }

    if (type == "infinite_sphere_cap") {
      PrimitiveLoadResult r = handle_infinite_sphere_cap(prim, data);
      primitives_loaded = primitives_loaded || r.loaded;
      load_flags |= r.flags;
      continue;
    }

    if (type == "skydome") {
      PrimitiveLoadResult r = handle_skydome(prim, data, scheduler);
      primitives_loaded = primitives_loaded || r.loaded;
      load_flags |= r.flags;
      continue;
    }

    uint32_t material_index = data.defaults.missing_material;
    bool bsdf_is_string = false;
    std::string bsdf_name;
    const nlohmann::json* bsdf_node = prim.contains("bsdf") ? &prim["bsdf"] : nullptr;
    if (bsdf_node != nullptr) {
      if (bsdf_node->is_string()) {
        bsdf_is_string = true;
        bsdf_name = bsdf_node->get<std::string>();
        auto it = bsdf_to_mat.find(bsdf_name);
        if (it != bsdf_to_mat.end()) {
          material_index = it->second;
        }
      } else if (bsdf_node->is_object()) {
        std::string mat_name = std::string("__prim_bsdf_") + std::to_string(data.materials.size());
        material_index = add_tungsten_material(mat_name, *bsdf_node, data, base_dir, database, force_two_sided);
      }
    }

    if (material_index == data.defaults.missing_material) {
      std::string mat_name = std::string("__prim_bsdf_") + std::to_string(data.materials.size());
      material_index = add_tungsten_material(mat_name, nlohmann::json::object(), data, base_dir, database, force_two_sided);
    }

    bool two_sided = prim.value("two_sided", prim.value("twoSided", false));
    if ((bsdf_node != nullptr) && bsdf_node->is_object()) {
      two_sided = two_sided || bsdf_node->value("two_sided", bsdf_node->value("twoSided", false));
    }
    if (force_two_sided)
      two_sided = true;

    bool has_emission = prim.contains("emission");
    bool has_power = prim.contains("power");
    float emission_scale = 1.0f;
    if (prim.contains("scale") && prim["scale"].is_number())
      emission_scale = float(prim["scale"].get<double>());

    bool wants_emission = has_power || has_emission;
    if (wants_emission && bsdf_is_string && (material_index != data.defaults.missing_material)) {
      std::string clone_name = bsdf_name.empty() ? std::string{} : bsdf_name + "__emitter_" + std::to_string(data.materials.size());
      material_index = data.clone_material(data.materials[material_index], clone_name.c_str());
    }

    auto& mtl = data.materials[material_index];
    if (two_sided)
      mtl.two_sided = 1u;

    float3 translate = {};
    float3 scale = {1.0f, 1.0f, 1.0f};
    float3 rotation = {};
    if (prim.contains("transform") && prim["transform"].is_object()) {
      const auto& tr = prim["transform"];
      if (tr.contains("position")) {
        translate = json_to_float3(tr["position"], {});
      }
      if (tr.contains("scale")) {
        const auto& sc = tr["scale"];
        if (sc.is_array() && sc.size() >= 3) {
          scale = {float(sc[0].get<double>()), float(sc[1].get<double>()), float(sc[2].get<double>())};
        } else if (sc.is_number()) {
          float v = float(sc.get<double>());
          scale = {v, v, v};
        }
      }
      if (tr.contains("rotation")) {
        rotation = json_to_float3(tr["rotation"], {});
      }
    }

    uint32_t tri_start = static_cast<uint32_t>(data.triangles.size());
    bool prim_loaded = false;

    PrimitiveLoadResult builtin_result = {};
    if (type != "mesh") {
      builtin_result = handle_builtin_primitive(type, translate, scale, rotation, material_index, data);
      prim_loaded = builtin_result.loaded;
    }

    PrimitiveLoadResult mesh_result = handle_mesh_primitive(prim, base_dir, type, translate, scale, rotation, material_index, data, database, scheduler, active_camera);
    prim_loaded = prim_loaded || mesh_result.loaded;
    load_flags |= mesh_result.flags;

    if (prim_loaded) {
      primitives_loaded = true;
      uint32_t tri_end = static_cast<uint32_t>(data.triangles.size());
      float area = triangles_area(data, tri_start, tri_end);
      if (has_power) {
        if (area <= 0.0f) {
          log::warning("Tungsten emitter has zero area, skipping power");
        } else {
          float power_scale = emission_scale / (area * kPi);
          if (two_sided)
            power_scale *= 0.5f;
          set_emission_from_json(mtl, prim["power"], data, base_dir, power_scale);
        }
      } else if (has_emission) {
        set_emission_from_json(mtl, prim["emission"], data, base_dir, emission_scale);
      }
    }
  }

  if (primitives_loaded)
    load_flags |= SceneLoadSucceeded;
  return load_flags;
}

}  // namespace

uint32_t load_from_tungsten_file(const char* file_name, SceneData& data, const IORDatabase& database, TaskScheduler& scheduler, Camera& active_camera) {
  if ((file_name == nullptr) || (file_name[0] == 0))
    return SceneLoadFailed;

  std::ifstream in(file_name, std::ios::binary);
  if (in.is_open() == false) {
    log::error("Failed to open Tungsten scene: %s", file_name);
    return SceneLoadFailed;
  }

  nlohmann::json js;
  try {
    in >> js;
  } catch (const std::exception& e) {
    log::error("Failed to parse Tungsten scene %s: %s", file_name, e.what());
    return SceneLoadFailed;
  }

  constexpr uint32_t buffer_size = 2048;
  char base_dir[buffer_size] = {};
  get_base_directory(file_name, base_dir, sizeof(base_dir));

  bool force_two_sided = js.value("enable_two_sided_shading", false);
  if (js.contains("renderer") && js["renderer"].is_object()) {
    force_two_sided = force_two_sided || js["renderer"].value("enable_two_sided_shading", false);
  }

  std::unordered_map<std::string, uint32_t> bsdf_to_mat;
  if (js.contains("bsdfs") && js["bsdfs"].is_array()) {
    for (const auto& b : js["bsdfs"]) {
      if (b.is_object() == false)
        continue;
      std::string name = b.value("name", "");
      if (name.empty())
        continue;
      if (bsdf_to_mat.count(name) > 0)
        continue;
      uint32_t mat_idx = add_tungsten_material(name, b, data, base_dir, database, force_two_sided);
      bsdf_to_mat[name] = mat_idx;
    }
  }

  bool camera_loaded = load_tungsten_camera(js, data, active_camera);

  load_tungsten_media(js, data);

  uint32_t mesh_count = count_tungsten_meshes(js);
  if (mesh_count > 0) {
    constexpr uint32_t kMeshVertexReserve = 1024u;
    constexpr uint32_t kMeshTriangleReserve = 2048u;
    reserve_mesh_vertices(data, mesh_count * kMeshVertexReserve);
    reserve_mesh_triangles(data, mesh_count * kMeshTriangleReserve);
  }

  uint32_t load_result = load_tungsten_primitives(js, base_dir, bsdf_to_mat, data, database, scheduler, active_camera, force_two_sided);

  if ((load_result & SceneLoadSucceeded) == 0u) {
    return camera_loaded ? SceneLoadCameraInfo : SceneLoadFailed;
  }

  if (camera_loaded) {
    load_result |= SceneLoadCameraInfo;
  }

  return load_result;
}

}  // namespace etx
