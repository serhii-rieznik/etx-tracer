#include "gpu_renderer.hxx"
#include <interop/gpu_rt_shared.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/render/host/scene_representation.hxx>

namespace etx {
namespace {
constexpr uint32_t kInvalidDescriptorIndex = ~0u;

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

template <typename T>
RHIBindlessHandle upload_linear_buffer(RHIDevice& device, const T* data, size_t count, std::vector<RHIBindlessHandle>& owner, RHIBufferUsage usage) {
  if ((data == nullptr) || (count == 0)) {
    return {};
  }

  RHIBufferDesc desc = {};
  desc.size = count * sizeof(T);
  desc.usage = usage;

  auto result = device.create_buffer(desc);
  if ((result.result != RHIResult::Success) || (result.handle.valid() == false)) {
    return {};
  }

  device.update_buffer(result.handle, data, desc.size);
  owner.push_back(result.handle);
  return result.handle;
}

GPUSceneGlobals build_scene_globals(const SceneData& scene_data) {
  auto bbox = scene_data.compute_bounding_volumes();
  const float3 sphere_center = 0.5f * (bbox.p_min + bbox.p_max);
  const float sphere_radius = length(bbox.p_max - sphere_center);

  GPUSceneGlobals globals = {};
  globals.vertex_count = static_cast<uint32_t>(scene_data.vertices.pos.size());
  globals.triangle_count = static_cast<uint32_t>(scene_data.triangles.size());
  globals.mesh_count = static_cast<uint32_t>(scene_data.meshes.size());
  globals.emitter_profile_count = static_cast<uint32_t>(scene_data.emitter_profiles.size());
  globals.emitter_instance_count = 0u;  // TODO: add packed emitter instances for GPU path.

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

  uint32_t environment_count = 0u;
  for (uint32_t i = 0, e = static_cast<uint32_t>(scene_data.emitter_profiles.size()); i < e; ++i) {
    const auto& emitter = scene_data.emitter_profiles[i];
    const bool is_environment = (emitter.cls == EmitterProfile::Class::Environment);
    const bool is_directional = (emitter.cls == EmitterProfile::Class::Directional);
    if ((is_environment || is_directional) && (environment_count < GPUSceneGlobals::MaxEnvironmentEmitters)) {
      globals.environment_emitters[environment_count++] = i;
    }
  }
  globals.environment_emitter_count = environment_count;

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
  Renderer::init(ctx, scene);

  create_pipelines(ctx);
  _initialized = true;
}

void GPURaytracingRenderer::create_pipelines(RHIContext& ctx) {
  auto& device = ctx.device();

  if (_pipeline.valid()) {
    device.destroy_pipeline(_pipeline);
    _pipeline = {};
  }

  auto& compiler = ShaderCompiler::instance();

  auto result = compiler.compile("shaders/gpu_rt.hlsl", {{"compute_main", RHIShaderStage::Compute}});
  if (result.result != RHIResult::Success) {
    log::error("Failed to compile GPU RT shader: %s", result.error_message.c_str());
    return;
  }

  RHIComputePipelineDesc desc = device.make_compute_pipeline_desc(result.binaries[0]);
  _pipeline = device.create_compute_pipeline(desc).handle;
}

void GPURaytracingRenderer::reload_shaders(RHIContext& ctx) {
  create_pipelines(ctx);
}

void GPURaytracingRenderer::render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& frame_data) {
  Renderer::update_camera(scene, frame_data.dt);

  if ((_initialized == false) || (_pipeline.valid() == false))
    return;

  if (_scene_dirty) {
    if (_tlas.valid()) {
      ctx.device().destroy_acceleration_structure(_tlas);
      _tlas = {};
    }
    for (auto blas : _blas) {
      ctx.device().destroy_acceleration_structure(blas);
    }
    _blas.clear();
    for (auto buf : _blas_buffers) {
      ctx.device().destroy_buffer(buf);
    }
    _blas_buffers.clear();
    for (auto buf : _scene_buffers) {
      ctx.device().destroy_buffer(buf);
    }
    _scene_buffers.clear();
    _gpu_scene = make_invalid_gpu_scene();
    _scene_dirty = false;
  }

  if (_tlas.valid() == false) {
    build_acceleration_structures(ctx, scene);
  }

  if (_tlas.valid() == false) {
    return;
  }

  uint2 current_dim = scene.camera().film_size;
  if (_output_dimensions.x != current_dim.x || _output_dimensions.y != current_dim.y) {
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

  auto cmd = ctx.get_command_buffer();
  ctx.command_buffer_begin(cmd);
  ctx.cmd_texture_barrier(cmd, _output_texture, RHIResourceState::Undefined, RHIResourceState::General);
  ctx.cmd_set_pipeline(cmd, _pipeline);
  ctx.cmd_push_constants(cmd, &constants, sizeof(constants));
  ctx.cmd_dispatch(cmd, {(current_dim.x + 7u) / 8u, (current_dim.y + 7u) / 8u, 1u});
  ctx.cmd_texture_barrier(cmd, _output_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
  ctx.command_buffer_end(cmd);
  ctx.submit_command_buffer({cmd});

  _frame_index += 1u;
  _sample_index += 1u;
}

void GPURaytracingRenderer::cleanup(RHIContext& ctx) {
  auto& device = ctx.device();
  if (_pipeline.value != 0) {
    device.destroy_pipeline(_pipeline);
    _pipeline = {};
  }

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
  for (auto buf : _scene_buffers) {
    device.destroy_buffer(buf);
  }
  _scene_buffers.clear();
  _gpu_scene = make_invalid_gpu_scene();

  if (_output_texture.valid()) {
    device.destroy_texture(_output_texture);
    _output_texture = {};
  }

  _initialized = false;
  _frame_index = 0u;
  _sample_index = 0u;
}

void GPURaytracingRenderer::on_camera_changed(SceneRepresentation& scene) {
}

void GPURaytracingRenderer::on_scene_changed(SceneRepresentation& scene) {
  _scene_dirty = true;
  _frame_index = 0u;
  _sample_index = 0u;
}

void GPURaytracingRenderer::build_acceleration_structures(RHIContext& ctx, SceneRepresentation& scene) {
  auto& device = ctx.device();

  // 1. Create BLAS
  const auto& s = scene.data();

  // Vertex Buffer
  RHIBufferDesc vb_desc = {};
  vb_desc.size = s.vertices.pos.size() * sizeof(float3);
  vb_desc.usage = RHIBufferUsage::Vertex | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress | RHIBufferUsage::TransferDst;
  auto vb_res = device.create_buffer(vb_desc);
  device.update_buffer(vb_res.handle, s.vertices.pos.data(), vb_desc.size);
  _blas_buffers.push_back(vb_res.handle);

  // Index Buffer (Repack from Triangle to uint32 stream)
  std::vector<uint32_t> indices;
  indices.reserve(s.triangles.size() * 3);
  for (const auto& tri : s.triangles) {
    indices.push_back(tri.i[0]);
    indices.push_back(tri.i[1]);
    indices.push_back(tri.i[2]);
  }

  RHIBufferDesc ib_desc = {};
  ib_desc.size = indices.size() * sizeof(uint32_t);
  ib_desc.usage = RHIBufferUsage::Index | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress | RHIBufferUsage::TransferDst;
  auto ib_res = device.create_buffer(ib_desc);
  device.update_buffer(ib_res.handle, indices.data(), ib_desc.size);
  _blas_buffers.push_back(ib_res.handle);

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
  ctx.command_buffer_begin(cmd);
  ctx.cmd_build_acceleration_structure(cmd, build_desc, scratch_res.handle, 0);
  ctx.cmd_buffer_barrier(cmd, scratch_res.handle, RHIResourceState::AccelerationStructure, RHIResourceState::AccelerationStructure);

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
  ctx.cmd_build_acceleration_structure(cmd, tlas_build_desc, scratch_res.handle, 32 * 1024 * 1024);  // Offset 32MB just in case
  ctx.command_buffer_end(cmd);
  ctx.submit_command_buffer({cmd});

  upload_scene_data(ctx, scene, vb_res.handle);
}

void GPURaytracingRenderer::upload_scene_data(RHIContext& ctx, SceneRepresentation& scene, RHIBindlessHandle vertex_positions_buffer) {
  auto& device = ctx.device();
  const auto& data = scene.data();

  _gpu_scene = make_invalid_gpu_scene();

  if (vertex_positions_buffer.valid()) {
    _gpu_scene.vertex_positions = get_bindless_descriptor_index(vertex_positions_buffer);
  }

  const RHIBufferUsage scene_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;

  auto normals = upload_linear_buffer(device, data.vertices.nrm.data(), data.vertices.nrm.size(), _scene_buffers, scene_buffer_usage);
  if (normals.valid()) {
    _gpu_scene.vertex_normals = get_bindless_descriptor_index(normals);
  }

  auto tangents = upload_linear_buffer(device, data.vertices.tan.data(), data.vertices.tan.size(), _scene_buffers, scene_buffer_usage);
  if (tangents.valid()) {
    _gpu_scene.vertex_tangents = get_bindless_descriptor_index(tangents);
  }

  auto bitangents = upload_linear_buffer(device, data.vertices.btn.data(), data.vertices.btn.size(), _scene_buffers, scene_buffer_usage);
  if (bitangents.valid()) {
    _gpu_scene.vertex_bitangents = get_bindless_descriptor_index(bitangents);
  }

  auto texcoords = upload_linear_buffer(device, data.vertices.tex.data(), data.vertices.tex.size(), _scene_buffers, scene_buffer_usage);
  if (texcoords.valid()) {
    _gpu_scene.vertex_texcoords = get_bindless_descriptor_index(texcoords);
  }

  auto triangles = upload_linear_buffer(device, data.triangles.data(), data.triangles.size(), _scene_buffers, scene_buffer_usage);
  if (triangles.valid()) {
    _gpu_scene.triangles = get_bindless_descriptor_index(triangles);
  }

  auto meshes = upload_linear_buffer(device, data.meshes.data(), data.meshes.size(), _scene_buffers, scene_buffer_usage);
  if (meshes.valid()) {
    _gpu_scene.meshes = get_bindless_descriptor_index(meshes);
  }

  auto emitter_profiles = upload_linear_buffer(device, data.emitter_profiles.data(), data.emitter_profiles.size(), _scene_buffers, scene_buffer_usage);
  if (emitter_profiles.valid()) {
    _gpu_scene.emitter_profiles = get_bindless_descriptor_index(emitter_profiles);
  }

  GPUSceneGlobals globals = build_scene_globals(data);
  auto scene_globals = upload_linear_buffer(device, &globals, size_t(1), _scene_buffers, scene_buffer_usage);
  if (scene_globals.valid()) {
    _gpu_scene.scene_globals = get_bindless_descriptor_index(scene_globals);
  }

  // TODO: upload packed emitter instances buffer.
  // TODO: upload packed materials buffer.
  // TODO: upload packed spectrums buffer.
  // TODO: upload packed images metadata and raw image/distribution tables.
  // TODO: upload packed mediums metadata and density grids.
  // TODO: upload packed emitters distribution data.
  // TODO: upload packed scene options buffer.
}

}  // namespace etx
