#include "gpu_renderer.hxx"
#include <interop/gpu_rt_shared.hxx>
#include <etx/core/profiler.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <cstddef>
#include <type_traits>

namespace etx {
namespace {
constexpr uint32_t kInvalidDescriptorIndex = ~0u;

constexpr uint32_t kShaderTriangleStride = 32u;
constexpr uint32_t kShaderMaterialStride = 272u;
constexpr uint32_t kShaderMaterialScatteringSpectrumOffset = 16u;
constexpr uint32_t kShaderSpectralDistributionStride = 3552u;
constexpr uint32_t kShaderSpectralDistributionIntegratedOffset = 0u;
constexpr uint32_t kShaderSceneGlobalsVertexCountOffset = 0u;
constexpr uint32_t kShaderSceneGlobalsTriangleCountOffset = 4u;
constexpr uint32_t kShaderSceneGlobalsBoundingSphereRadiusOffset = 44u;

static_assert(std::is_standard_layout_v<float2>, "float2 must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<float2>, "float2 must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(float2) == 8u, "float2 size changed; update GPU upload ABI");

static_assert(std::is_standard_layout_v<float3>, "float3 must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<float3>, "float3 must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(float3) == 12u, "float3 size changed; update GPU upload ABI");

static_assert(std::is_standard_layout_v<Triangle>, "Triangle must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Triangle>, "Triangle must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Triangle) == kShaderTriangleStride, "Triangle size changed; update GPU shader decode stride");
static_assert(offsetof(Triangle, i) == 0u, "Triangle::i offset changed; update GPU shader decode");
static_assert(offsetof(Triangle, material_index) == 12u, "Triangle::material_index offset changed; update GPU shader decode");
static_assert(offsetof(Triangle, geo_n) == 16u, "Triangle::geo_n offset changed; update GPU shader decode");
static_assert(offsetof(Triangle, emitter_index) == 28u, "Triangle::emitter_index offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<Mesh>, "Mesh must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Mesh>, "Mesh must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Mesh) == 32u, "Mesh size changed; update GPU upload ABI");

static_assert(std::is_standard_layout_v<EmitterProfile>, "EmitterProfile must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<EmitterProfile>, "EmitterProfile must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(EmitterProfile) == 80u, "EmitterProfile size changed; update GPU upload ABI");

static_assert(std::is_standard_layout_v<Emitter>, "Emitter must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Emitter>, "Emitter must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Emitter) == 32u, "Emitter size changed; update GPU upload ABI");

static_assert(std::is_standard_layout_v<Material>, "Material must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Material>, "Material must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Material) == kShaderMaterialStride, "Material size changed; update GPU shader decode stride");
static_assert(offsetof(Material, scattering) == kShaderMaterialScatteringSpectrumOffset, "Material::scattering offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<SpectralImage>, "SpectralImage must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<SpectralImage>, "SpectralImage must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(SpectralImage) == 16u, "SpectralImage size changed; update material ABI");
static_assert(offsetof(SpectralImage, spectrum_index) == 0u, "SpectralImage::spectrum_index offset changed; update material ABI");
static_assert(offsetof(SpectralImage, image_index) == 4u, "SpectralImage::image_index offset changed; update material ABI");

static_assert(std::is_standard_layout_v<SampledImage>, "SampledImage must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<SampledImage>, "SampledImage must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(SampledImage) == 32u, "SampledImage size changed; update material ABI");
static_assert(offsetof(SampledImage, value) == 0u, "SampledImage::value offset changed; update material ABI");
static_assert(offsetof(SampledImage, image_index) == 16u, "SampledImage::image_index offset changed; update material ABI");
static_assert(offsetof(SampledImage, channel) == 20u, "SampledImage::channel offset changed; update material ABI");

static_assert(std::is_standard_layout_v<Thinfilm>, "Thinfilm must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Thinfilm>, "Thinfilm must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Thinfilm) == 32u, "Thinfilm size changed; update material ABI");

static_assert(std::is_standard_layout_v<RefractiveIndex>, "RefractiveIndex must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<RefractiveIndex>, "RefractiveIndex must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(RefractiveIndex) == 16u, "RefractiveIndex size changed; update material/spectrum ABI");

static_assert(std::is_standard_layout_v<SpectralDistribution>, "SpectralDistribution must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<SpectralDistribution>, "SpectralDistribution must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(SpectralDistribution) == kShaderSpectralDistributionStride, "SpectralDistribution size changed; update GPU shader decode stride");
static_assert(offsetof(SpectralDistribution, integrated_value) == kShaderSpectralDistributionIntegratedOffset,
  "SpectralDistribution::integrated_value offset changed; update GPU shader decode");

static_assert(offsetof(GPUSceneGlobals, vertex_count) == kShaderSceneGlobalsVertexCountOffset, "GPUSceneGlobals::vertex_count offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, triangle_count) == kShaderSceneGlobalsTriangleCountOffset, "GPUSceneGlobals::triangle_count offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, bounding_sphere_radius) == kShaderSceneGlobalsBoundingSphereRadiusOffset,
  "GPUSceneGlobals::bounding_sphere_radius offset changed; update GPU shader decode");

GPUScene make_invalid_gpu_scene() {
  return {
    .vertex_positions = kInvalidDescriptorIndex,
    .vertex_normals = kInvalidDescriptorIndex,
    .vertex_tangents = kInvalidDescriptorIndex,
    .vertex_bitangents = kInvalidDescriptorIndex,
    .vertex_texcoords = kInvalidDescriptorIndex,
    .triangles = kInvalidDescriptorIndex,
    .meshes = kInvalidDescriptorIndex,
    .emitter_profiles = kInvalidDescriptorIndex,
    .emitter_instances = kInvalidDescriptorIndex,
    .scene_globals = kInvalidDescriptorIndex,
    .materials = kInvalidDescriptorIndex,
    .spectrums = kInvalidDescriptorIndex,
    .images = kInvalidDescriptorIndex,
    .mediums = kInvalidDescriptorIndex,
    .emitters_distribution = kInvalidDescriptorIndex,
    .scene_options = kInvalidDescriptorIndex,
  };
}

void destroy_linear_scene_buffer(RHIDevice& device, RHIBindlessHandle& buffer, uint64_t& buffer_size, uint32_t& descriptor_index) {
  ETX_PROFILER_NAMED_SCOPE("gpu_rt_destroy_linear_scene_buffer");

  if (buffer.valid()) {
    device.destroy_buffer(buffer);
  }
  buffer = {};
  buffer_size = 0;
  descriptor_index = kInvalidDescriptorIndex;
}

template <typename T>
void upload_or_update_linear_scene_buffer(RHIDevice& device, const T* data, size_t count, RHIBufferUsage usage, RHIBindlessHandle& buffer, uint64_t& buffer_size,
  uint32_t& descriptor_index) {
  ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_or_update_linear_scene_buffer");

  if ((data == nullptr) || (count == 0u)) {
    destroy_linear_scene_buffer(device, buffer, buffer_size, descriptor_index);
    return;
  }

  const uint64_t required_size = static_cast<uint64_t>(count) * sizeof(T);
  if (buffer.valid() && (buffer_size == required_size)) {
    device.update_buffer(buffer, data, required_size);
    descriptor_index = get_bindless_descriptor_index(buffer);
    return;
  }

  RHIBufferDesc desc = {};
  desc.size = required_size;
  desc.usage = usage;

  auto result = device.create_buffer(desc);
  if ((result.result != RHIResult::Success) || (result.handle.valid() == false)) {
    descriptor_index = buffer.valid() ? get_bindless_descriptor_index(buffer) : kInvalidDescriptorIndex;
    return;
  }

  device.update_buffer(result.handle, data, required_size);
  if (buffer.valid()) {
    device.destroy_buffer(buffer);
  }

  buffer = result.handle;
  buffer_size = required_size;
  descriptor_index = get_bindless_descriptor_index(buffer);
}

struct PackedEmitterData {
  std::vector<Triangle> triangles;
  std::vector<Emitter> emitter_instances;
};

float safe_spectrum_luminance(const SceneData& scene_data, uint32_t spectrum_index) {
  if (spectrum_index >= static_cast<uint32_t>(scene_data.spectrum_values.size())) {
    return 0.0f;
  }
  return scene_data.spectrum_values[spectrum_index].luminance();
}

PackedEmitterData build_packed_emitters(const SceneData& scene_data) {
  ETX_PROFILER_SCOPE();

  PackedEmitterData result = {};
  result.triangles = scene_data.triangles;

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_reset_triangle_emitter_indices");
    for (auto& tri : result.triangles) {
      tri.emitter_index = kInvalidIndex;
    }
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_pack_non_area_emitters");
    for (uint32_t i = 0u; i < static_cast<uint32_t>(scene_data.emitter_profiles.size()); ++i) {
      const auto& profile = scene_data.emitter_profiles[i];
      if (profile.cls == EmitterProfile::Class::Area) {
        continue;
      }

      Emitter emitter(profile.cls);
      emitter.profile = i;
      emitter.triangle_index = kInvalidIndex;
      emitter.spectrum_weight = (profile.emission.spectrum_index != kInvalidIndex) ? safe_spectrum_luminance(scene_data, profile.emission.spectrum_index) : 0.0f;
      emitter.additional_weight = (profile.cls == EmitterProfile::Class::Directional) ? kPi : (4.0f * kPi);
      result.emitter_instances.push_back(emitter);
    }
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_pack_area_emitters");
    for (uint32_t tri_index = 0u; tri_index < static_cast<uint32_t>(scene_data.triangles.size()); ++tri_index) {
      const Triangle& tri = scene_data.triangles[tri_index];
      if (tri.emitter_index == kInvalidIndex) {
        continue;
      }
      if (tri.emitter_index >= static_cast<uint32_t>(scene_data.emitter_profiles.size())) {
        continue;
      }

      const auto& profile = scene_data.emitter_profiles[tri.emitter_index];
      if (profile.cls != EmitterProfile::Class::Area) {
        continue;
      }

      Emitter emitter(EmitterProfile::Class::Area);
      emitter.profile = tri.emitter_index;
      emitter.triangle_index = tri_index;

      if ((tri.material_index < static_cast<uint32_t>(scene_data.materials.size())) && (tri.i[0] < scene_data.vertices.pos.size()) && (tri.i[1] < scene_data.vertices.pos.size()) &&
          (tri.i[2] < scene_data.vertices.pos.size())) {
        const auto& mtl = scene_data.materials[tri.material_index];
        emitter.spectrum_weight = (profile.emission.spectrum_index != kInvalidIndex) ? safe_spectrum_luminance(scene_data, profile.emission.spectrum_index) : 0.0f;

        const float3& v0 = scene_data.vertices.pos[tri.i[0]];
        const float3& v1 = scene_data.vertices.pos[tri.i[1]];
        const float3& v2 = scene_data.vertices.pos[tri.i[2]];
        float triangle_area = 0.5f * length(cross(v1 - v0, v2 - v0));

        emitter.triangle_area = triangle_area;
        emitter.additional_weight = (mtl.two_sided ? 2.0f : 1.0f) * triangle_area * kPi;
      }

      result.emitter_instances.push_back(emitter);
      result.triangles[tri_index].emitter_index = static_cast<uint32_t>(result.emitter_instances.size() - 1u);
    }
  }

  return result;
}

GPUSceneGlobals build_scene_globals(const SceneData& scene_data, const std::vector<Emitter>& emitter_instances) {
  ETX_PROFILER_SCOPE();

  BoundingBox bbox = {};
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_compute_scene_bounds");
    bbox = scene_data.compute_bounding_volumes();
  }

  const float3 sphere_center = 0.5f * (bbox.p_min + bbox.p_max);
  const float sphere_radius = length(bbox.p_max - sphere_center);

  GPUSceneGlobals globals = {};
  globals.vertex_count = static_cast<uint32_t>(scene_data.vertices.pos.size());
  globals.triangle_count = static_cast<uint32_t>(scene_data.triangles.size());
  globals.mesh_count = static_cast<uint32_t>(scene_data.meshes.size());
  globals.emitter_profile_count = static_cast<uint32_t>(scene_data.emitter_profiles.size());
  globals.emitter_instance_count = static_cast<uint32_t>(emitter_instances.size());

  globals.bounding_sphere_center = sphere_center;
  globals.bounding_sphere_radius = sphere_radius;
  globals.bounding_box_min = bbox.p_min;
  globals.bounding_box_max = bbox.p_max;

  globals.default_black_spectrum = scene_data.defaults.black_spectrum;
  globals.default_white_spectrum = scene_data.defaults.white_spectrum;
  globals.default_rayleigh_spectrum = scene_data.defaults.rayleigh_spectrum;
  globals.default_mie_spectrum = scene_data.defaults.mie_spectrum;
  globals.default_ozone_spectrum = scene_data.defaults.ozone_spectrum;
  globals.default_subsurface_scatter_material = scene_data.defaults.subsurface_scatter_material;
  globals.default_subsurface_exit_material = scene_data.defaults.subsurface_exit_material;
  globals.default_missing_material = scene_data.defaults.missing_material;
  globals.default_dielectric_eta = scene_data.defaults.dielectric_eta;
  globals.default_conductor_eta = scene_data.defaults.conductor_eta;
  globals.default_conductor_k = scene_data.defaults.conductor_k;

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_collect_environment_emitters");
    uint32_t environment_count = 0u;
    for (uint32_t i = 0, e = static_cast<uint32_t>(emitter_instances.size()); i < e; ++i) {
      const auto& emitter = emitter_instances[i];
      const bool is_environment = (emitter.cls == EmitterProfile::Class::Environment);
      const bool is_directional = (emitter.cls == EmitterProfile::Class::Directional);
      if ((is_environment || is_directional) && (environment_count < GPUSceneGlobals::MaxEnvironmentEmitters)) {
        globals.environment_emitters[environment_count++] = i;
      }
    }
    globals.environment_emitter_count = environment_count;
  }

  return globals;
}
}  // namespace

GPURaytracingRenderer::GPURaytracingRenderer(TaskScheduler& s)
  : Renderer(s) {
  _gpu_scene = make_invalid_gpu_scene();
}

GPURaytracingRenderer::~GPURaytracingRenderer() {
}

void GPURaytracingRenderer::init(RHIContext& ctx, SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();

  Renderer::init(ctx, scene);

  create_pipelines(ctx);
  _initialized = true;
}

void GPURaytracingRenderer::create_pipelines(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();

  if (_pipeline.valid()) {
    device.destroy_pipeline(_pipeline);
    _pipeline = {};
  }

  auto& compiler = ShaderCompiler::instance();

  ShaderCompiler::MultiShaderCompilationResult result = {};
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_compile_compute_shader");
    result = compiler.compile("shaders/gpu_rt.hlsl", {{"compute_main", RHIShaderStage::Compute}});
  }

  if (result.result != RHIResult::Success) {
    log::error("Failed to compile GPU RT shader: %s", result.error_message.c_str());
    return;
  }

  RHIComputePipelineDesc desc = device.make_compute_pipeline_desc(result.binaries[0]);
  _pipeline = device.create_compute_pipeline(desc).handle;
}

void GPURaytracingRenderer::reload_shaders(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();
  create_pipelines(ctx);
}

void GPURaytracingRenderer::set_scene_updates_locked(bool locked) {
  ETX_PROFILER_SCOPE();
  _scene_updates_locked = locked;
}

void GPURaytracingRenderer::destroy_scene_buffers(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  destroy_linear_scene_buffer(device, _vertex_normals_buffer, _vertex_normals_buffer_size, _gpu_scene.vertex_normals);
  destroy_linear_scene_buffer(device, _vertex_tangents_buffer, _vertex_tangents_buffer_size, _gpu_scene.vertex_tangents);
  destroy_linear_scene_buffer(device, _vertex_bitangents_buffer, _vertex_bitangents_buffer_size, _gpu_scene.vertex_bitangents);
  destroy_linear_scene_buffer(device, _vertex_texcoords_buffer, _vertex_texcoords_buffer_size, _gpu_scene.vertex_texcoords);
  destroy_linear_scene_buffer(device, _triangles_buffer, _triangles_buffer_size, _gpu_scene.triangles);
  destroy_linear_scene_buffer(device, _meshes_buffer, _meshes_buffer_size, _gpu_scene.meshes);
  destroy_linear_scene_buffer(device, _emitter_profiles_buffer, _emitter_profiles_buffer_size, _gpu_scene.emitter_profiles);
  destroy_linear_scene_buffer(device, _emitter_instances_buffer, _emitter_instances_buffer_size, _gpu_scene.emitter_instances);
  destroy_linear_scene_buffer(device, _materials_buffer, _materials_buffer_size, _gpu_scene.materials);
  destroy_linear_scene_buffer(device, _spectrums_buffer, _spectrums_buffer_size, _gpu_scene.spectrums);
  destroy_linear_scene_buffer(device, _scene_globals_buffer, _scene_globals_buffer_size, _gpu_scene.scene_globals);

  _gpu_scene.images = kInvalidDescriptorIndex;
  _gpu_scene.mediums = kInvalidDescriptorIndex;
  _gpu_scene.emitters_distribution = kInvalidDescriptorIndex;
  _gpu_scene.scene_options = kInvalidDescriptorIndex;
}

void GPURaytracingRenderer::destroy_acceleration_structures(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();

  if (_tlas.valid()) {
    device.destroy_acceleration_structure(_tlas);
    _tlas = {};
  }

  for (auto blas : _blas) {
    device.destroy_acceleration_structure(blas);
  }
  _blas.clear();

  for (auto buf : _blas_buffers) {
    device.destroy_buffer(buf);
  }
  _blas_buffers.clear();
  _vertex_positions_buffer = {};
  _gpu_scene.vertex_positions = kInvalidDescriptorIndex;
}

void GPURaytracingRenderer::render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& frame_data) {
  ETX_PROFILER_SCOPE();

  Renderer::update_camera(scene, frame_data.dt);

  if ((_initialized == false) || (_pipeline.valid() == false))
    return;

  SceneHashes new_hashes = {};
  UpdateFlags changes = {};
  bool scene_changed = false;
  if (_scene_updates_locked == false) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_scene_hashes_and_changes");
    new_hashes = scene.data().compute_hashes();
    changes = new_hashes.compare(_current_scene_hashes);
    scene_changed = changes.any();
  } else {
    new_hashes = _current_scene_hashes;
  }

  const auto& camera = scene.camera();
  const uint64_t new_camera_hash = xxh64(&camera, sizeof(camera));
  const bool camera_changed = (new_camera_hash != _current_camera_hash);

  bool geometry_structure_changed = changes[UpdateFlags::AnyGeometryStructure];
  bool needs_full_rebuild = (_scene_updates_locked == false) && (_scene_dirty || geometry_structure_changed);
  const bool needs_scene_data_reupload = (_scene_updates_locked == false) && scene_changed && (geometry_structure_changed == false);

  if (needs_scene_data_reupload && (_vertex_positions_buffer.valid() == false)) {
    needs_full_rebuild = true;
  }

  if (scene_changed || camera_changed || ((_scene_updates_locked == false) && _scene_dirty)) {
    _frame_index = 0u;
    _sample_index = 0u;
  }

  if (needs_full_rebuild) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_full_rebuild_resources");
    destroy_scene_buffers(ctx);
    destroy_acceleration_structures(ctx);
    _scene_dirty = false;
  }

  if (_tlas.valid() == false) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_acceleration_structures");
    build_acceleration_structures(ctx, scene);
  } else if (needs_scene_data_reupload) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_scene_update");
    update_scene_data_partial(ctx, scene, changes);
  }

  if (_tlas.valid() == false) {
    return;
  }

  if (_scene_updates_locked == false) {
    _current_scene_hashes = new_hashes;
  }
  _current_camera_hash = new_camera_hash;

  uint2 current_dim = scene.camera().film_size;
  if (_output_dimensions.x != current_dim.x || _output_dimensions.y != current_dim.y) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_recreate_output_texture");
    if (_output_texture.valid()) {
      ctx.device().destroy_texture(_output_texture);
    }
    RHITextureDesc desc = {};
    desc.width = current_dim.x;
    desc.height = current_dim.y;
    desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
    desc.usage = RHITextureUsage::Storage | RHITextureUsage::Sampled | RHITextureUsage::TransferSrc;
    _output_texture = ctx.device().create_texture(desc).handle;
    _output_dimensions = current_dim;
  }

  GPURTConstants constants = {
    .camera = scene.camera(),
    .as_index = get_bindless_descriptor_index(_tlas),
    .output_image_index = get_bindless_descriptor_index(_output_texture),
    .frame_index = _frame_index,
    .sample_index = _sample_index,
    .scene = _gpu_scene,
  };

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_dispatch_and_submit");
    auto cmd = ctx.get_command_buffer();
    ctx.command_buffer_begin(cmd);
    ctx.cmd_texture_barrier(cmd, _output_texture, RHIResourceState::Undefined, RHIResourceState::General);
    ctx.cmd_set_pipeline(cmd, _pipeline);
    ctx.cmd_push_constants(cmd, &constants, sizeof(constants));
    ctx.cmd_dispatch(cmd, {(current_dim.x + 7u) / 8u, (current_dim.y + 7u) / 8u, 1u});
    ctx.cmd_texture_barrier(cmd, _output_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
    ctx.command_buffer_end(cmd);
    ctx.submit_command_buffer({cmd});
  }

  _frame_index += 1u;
  _sample_index += 1u;
}

void GPURaytracingRenderer::cleanup(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  if (_pipeline.value != 0) {
    device.destroy_pipeline(_pipeline);
    _pipeline = {};
  }

  destroy_scene_buffers(ctx);
  destroy_acceleration_structures(ctx);

  if (_output_texture.valid()) {
    device.destroy_texture(_output_texture);
    _output_texture = {};
  }

  _initialized = false;
  _scene_dirty = false;
  _scene_updates_locked = false;
  _current_scene_hashes = {};
  _current_camera_hash = 0;
  _frame_index = 0u;
  _sample_index = 0u;
}

void GPURaytracingRenderer::on_camera_changed(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
}

void GPURaytracingRenderer::on_scene_changed(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();

  _scene_dirty = true;
  _current_scene_hashes = {};
  _current_camera_hash = 0;
  _frame_index = 0u;
  _sample_index = 0u;
}

void GPURaytracingRenderer::build_acceleration_structures(RHIContext& ctx, SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();

  // 1. Create BLAS
  const auto& s = scene.data();

  std::vector<uint32_t> indices;
  RHIBufferDesc vb_desc = {};
  auto vb_res = RHICreateBindlessResult{};
  RHIBufferDesc ib_desc = {};
  auto ib_res = RHICreateBindlessResult{};

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_as_upload_geometry_buffers");

    // Vertex Buffer
    vb_desc.size = s.vertices.pos.size() * sizeof(float3);
    vb_desc.usage = RHIBufferUsage::Vertex | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress | RHIBufferUsage::TransferDst;
    vb_res = device.create_buffer(vb_desc);
    device.update_buffer(vb_res.handle, s.vertices.pos.data(), vb_desc.size);
    _blas_buffers.push_back(vb_res.handle);
    _vertex_positions_buffer = vb_res.handle;

    // Index Buffer (Repack from Triangle to uint32 stream)
    indices.reserve(s.triangles.size() * 3);
    for (const auto& tri : s.triangles) {
      indices.push_back(tri.i[0]);
      indices.push_back(tri.i[1]);
      indices.push_back(tri.i[2]);
    }

    ib_desc.size = indices.size() * sizeof(uint32_t);
    ib_desc.usage = RHIBufferUsage::Index | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress | RHIBufferUsage::TransferDst;
    ib_res = device.create_buffer(ib_desc);
    device.update_buffer(ib_res.handle, indices.data(), ib_desc.size);
    _blas_buffers.push_back(ib_res.handle);
  }

  RHIAccelerationStructureGeometry geometry = {};
  geometry.is_opaque = true;
  geometry.triangles.vertex_buffer = vb_res.handle;
  geometry.triangles.vertex_count = static_cast<uint32_t>(s.vertices.pos.size());
  geometry.triangles.vertex_stride = sizeof(float3);
  geometry.triangles.vertex_format = RHIVertexFormat::Float3;
  geometry.triangles.index_buffer = ib_res.handle;
  geometry.triangles.index_count = static_cast<uint32_t>(indices.size());
  geometry.triangles.index_type = RHIIndexType::UInt32;

  RHIAccelerationStructureDesc blas_desc = {};
  blas_desc.type = RHIAccelerationStructureType::BottomLevel;
  blas_desc.geometry_count = 1;
  blas_desc.geometries = &geometry;

  auto blas_result = device.create_acceleration_structure(blas_desc);  // This creates the backing buffer for AS
  if (blas_result.result != RHIResult::Success) {
    log::error("Failed to create BLAS");
    return;
  }
  _blas.push_back(blas_result.handle);

  // TODO: Query AS build size from RHI
  uint64_t scratch_size = 64 * 1024 * 1024;
  RHIBufferDesc scratch_desc = {};
  scratch_desc.size = scratch_size;
  scratch_desc.usage = RHIBufferUsage::Storage | RHIBufferUsage::ShaderDeviceAddress;  // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
  auto scratch_res = device.create_buffer(scratch_desc);
  _blas_buffers.push_back(scratch_res.handle);

  RHIAccelerationStructureBuildDesc build_desc = {};
  build_desc.as_handle = blas_result.handle;
  build_desc.type = RHIAccelerationStructureType::BottomLevel;
  build_desc.geometry_count = 1;
  build_desc.geometries = &geometry;

  auto cmd = ctx.get_command_buffer();
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_blas");
    ctx.command_buffer_begin(cmd);
    ctx.cmd_build_acceleration_structure(cmd, build_desc, scratch_res.handle, 0);
    ctx.cmd_buffer_barrier(cmd, scratch_res.handle, RHIResourceState::AccelerationStructure, RHIResourceState::AccelerationStructure);
  }

  // 2. Create TLAS
  // Create Instance Buffer
  RHIAccelerationStructureInstance instance = {};
  instance.transform[0] = 1.0f;
  instance.transform[5] = 1.0f;
  instance.transform[10] = 1.0f;
  instance.instance_custom_index = 0;
  instance.mask = 0xFF;
  instance.instance_shader_binding_table_record_offset = 0;
  instance.flags = 0;  // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR etc.
  instance.acceleration_structure_reference = device.get_acceleration_structure_device_address(blas_result.handle);

  RHIBufferDesc inst_buf_desc = {};
  inst_buf_desc.size = sizeof(RHIAccelerationStructureInstance);
  inst_buf_desc.usage = RHIBufferUsage::ShaderDeviceAddress | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::TransferDst;  // Input to build
  // Wait, usually it's `VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR`
  auto inst_res = device.create_buffer(inst_buf_desc);
  device.update_buffer(inst_res.handle, &instance, sizeof(instance));
  _blas_buffers.push_back(inst_res.handle);

  RHIAccelerationStructureDesc tlas_desc = {};
  tlas_desc.type = RHIAccelerationStructureType::TopLevel;
  tlas_desc.instance_count = 1;

  auto tlas_result = device.create_acceleration_structure(tlas_desc);
  _tlas = tlas_result.handle;

  RHIAccelerationStructureBuildDesc tlas_build_desc = {};
  tlas_build_desc.as_handle = _tlas;
  tlas_build_desc.type = RHIAccelerationStructureType::TopLevel;
  tlas_build_desc.instance_count = 1;
  tlas_build_desc.instance_buffer = inst_res.handle;

  // Re-use scratch buffer (assuming enough size and barriers)
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_tlas");
    ctx.cmd_build_acceleration_structure(cmd, tlas_build_desc, scratch_res.handle, 32 * 1024 * 1024);  // Offset 32MB just in case
    ctx.command_buffer_end(cmd);
    ctx.submit_command_buffer({cmd});
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_scene_after_as_build");
    upload_scene_data(ctx, scene, _vertex_positions_buffer);
  }
}

void GPURaytracingRenderer::upload_scene_data(RHIContext& ctx, SceneRepresentation& scene, RHIBindlessHandle vertex_positions_buffer) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  const auto& data = scene.data();

  _gpu_scene = make_invalid_gpu_scene();

  if (vertex_positions_buffer.valid()) {
    _gpu_scene.vertex_positions = get_bindless_descriptor_index(vertex_positions_buffer);
  }

  const RHIBufferUsage scene_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
  PackedEmitterData packed_emitters = {};
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_packed_emitters_full_upload");
    packed_emitters = build_packed_emitters(data);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_vertex_and_geometry_buffers");
    upload_or_update_linear_scene_buffer(device, data.vertices.nrm.data(), data.vertices.nrm.size(), scene_buffer_usage, _vertex_normals_buffer, _vertex_normals_buffer_size,
      _gpu_scene.vertex_normals);
    upload_or_update_linear_scene_buffer(device, data.vertices.tan.data(), data.vertices.tan.size(), scene_buffer_usage, _vertex_tangents_buffer, _vertex_tangents_buffer_size,
      _gpu_scene.vertex_tangents);
    upload_or_update_linear_scene_buffer(device, data.vertices.btn.data(), data.vertices.btn.size(), scene_buffer_usage, _vertex_bitangents_buffer, _vertex_bitangents_buffer_size,
      _gpu_scene.vertex_bitangents);
    upload_or_update_linear_scene_buffer(device, data.vertices.tex.data(), data.vertices.tex.size(), scene_buffer_usage, _vertex_texcoords_buffer, _vertex_texcoords_buffer_size,
      _gpu_scene.vertex_texcoords);
    upload_or_update_linear_scene_buffer(device, packed_emitters.triangles.data(), packed_emitters.triangles.size(), scene_buffer_usage, _triangles_buffer, _triangles_buffer_size,
      _gpu_scene.triangles);
    upload_or_update_linear_scene_buffer(device, data.meshes.data(), data.meshes.size(), scene_buffer_usage, _meshes_buffer, _meshes_buffer_size, _gpu_scene.meshes);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_material_and_emitter_buffers");
    upload_or_update_linear_scene_buffer(device, data.emitter_profiles.data(), data.emitter_profiles.size(), scene_buffer_usage, _emitter_profiles_buffer,
      _emitter_profiles_buffer_size, _gpu_scene.emitter_profiles);
    upload_or_update_linear_scene_buffer(device, packed_emitters.emitter_instances.data(), packed_emitters.emitter_instances.size(), scene_buffer_usage, _emitter_instances_buffer,
      _emitter_instances_buffer_size, _gpu_scene.emitter_instances);
    upload_or_update_linear_scene_buffer(device, data.materials.data(), data.materials.size(), scene_buffer_usage, _materials_buffer, _materials_buffer_size, _gpu_scene.materials);
    upload_or_update_linear_scene_buffer(device, data.spectrum_values.data(), data.spectrum_values.size(), scene_buffer_usage, _spectrums_buffer, _spectrums_buffer_size,
      _gpu_scene.spectrums);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_scene_globals");
    GPUSceneGlobals globals = build_scene_globals(data, packed_emitters.emitter_instances);
    upload_or_update_linear_scene_buffer(device, &globals, size_t(1), scene_buffer_usage, _scene_globals_buffer, _scene_globals_buffer_size, _gpu_scene.scene_globals);
  }

  // TODO: upload packed images metadata and raw image/distribution tables.
  // TODO: upload packed mediums metadata and density grids.
  // TODO: upload packed emitters distribution data.
  // TODO: upload packed scene options buffer.
}

void GPURaytracingRenderer::update_scene_data_partial(RHIContext& ctx, SceneRepresentation& scene, const UpdateFlags& changes) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  const auto& data = scene.data();
  const RHIBufferUsage scene_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_direct_buffer_updates");
    if (changes[UpdateFlags::VerticesNrm]) {
      upload_or_update_linear_scene_buffer(device, data.vertices.nrm.data(), data.vertices.nrm.size(), scene_buffer_usage, _vertex_normals_buffer, _vertex_normals_buffer_size,
        _gpu_scene.vertex_normals);
    }
    if (changes[UpdateFlags::VerticesTan]) {
      upload_or_update_linear_scene_buffer(device, data.vertices.tan.data(), data.vertices.tan.size(), scene_buffer_usage, _vertex_tangents_buffer, _vertex_tangents_buffer_size,
        _gpu_scene.vertex_tangents);
    }
    if (changes[UpdateFlags::VerticesBtn]) {
      upload_or_update_linear_scene_buffer(device, data.vertices.btn.data(), data.vertices.btn.size(), scene_buffer_usage, _vertex_bitangents_buffer,
        _vertex_bitangents_buffer_size, _gpu_scene.vertex_bitangents);
    }
    if (changes[UpdateFlags::VerticesTex]) {
      upload_or_update_linear_scene_buffer(device, data.vertices.tex.data(), data.vertices.tex.size(), scene_buffer_usage, _vertex_texcoords_buffer, _vertex_texcoords_buffer_size,
        _gpu_scene.vertex_texcoords);
    }
    if (changes[UpdateFlags::Meshes]) {
      upload_or_update_linear_scene_buffer(device, data.meshes.data(), data.meshes.size(), scene_buffer_usage, _meshes_buffer, _meshes_buffer_size, _gpu_scene.meshes);
    }
    if (changes[UpdateFlags::Materials]) {
      upload_or_update_linear_scene_buffer(device, data.materials.data(), data.materials.size(), scene_buffer_usage, _materials_buffer, _materials_buffer_size,
        _gpu_scene.materials);
    }
    if (changes[UpdateFlags::Spectra]) {
      upload_or_update_linear_scene_buffer(device, data.spectrum_values.data(), data.spectrum_values.size(), scene_buffer_usage, _spectrums_buffer, _spectrums_buffer_size,
        _gpu_scene.spectrums);
    }
    if (changes[UpdateFlags::Emitters]) {
      upload_or_update_linear_scene_buffer(device, data.emitter_profiles.data(), data.emitter_profiles.size(), scene_buffer_usage, _emitter_profiles_buffer,
        _emitter_profiles_buffer_size, _gpu_scene.emitter_profiles);
    }
  }

  const bool packed_emitters_changed = changes[UpdateFlags::Emitters] || changes[UpdateFlags::Materials] || changes[UpdateFlags::Spectra];
  const bool scene_globals_changed = changes[UpdateFlags::Meshes] || changes[UpdateFlags::Emitters] || changes[UpdateFlags::Defaults];

  PackedEmitterData packed_emitters = {};
  if (packed_emitters_changed || scene_globals_changed) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_build_packed_emitters");
    packed_emitters = build_packed_emitters(data);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_dependent_buffer_updates");
    if (changes[UpdateFlags::Emitters]) {
      upload_or_update_linear_scene_buffer(device, packed_emitters.triangles.data(), packed_emitters.triangles.size(), scene_buffer_usage, _triangles_buffer,
        _triangles_buffer_size, _gpu_scene.triangles);
    }

    if (packed_emitters_changed) {
      upload_or_update_linear_scene_buffer(device, packed_emitters.emitter_instances.data(), packed_emitters.emitter_instances.size(), scene_buffer_usage,
        _emitter_instances_buffer, _emitter_instances_buffer_size, _gpu_scene.emitter_instances);
    }
  }

  if (scene_globals_changed) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_update_scene_globals");
    GPUSceneGlobals globals = build_scene_globals(data, packed_emitters.emitter_instances);
    upload_or_update_linear_scene_buffer(device, &globals, size_t(1), scene_buffer_usage, _scene_globals_buffer, _scene_globals_buffer_size, _gpu_scene.scene_globals);
  }
}

}  // namespace etx
