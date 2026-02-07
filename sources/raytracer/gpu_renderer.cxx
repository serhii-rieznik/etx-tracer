#include "gpu_renderer.hxx"
#include <interop/gpu_rt_shared.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/render/host/scene_representation.hxx>

namespace etx {

GPURaytracingRenderer::GPURaytracingRenderer(TaskScheduler& s)
  : Renderer(s) {
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

  if (_output_texture.valid()) {
    device.destroy_texture(_output_texture);
    _output_texture = {};
  }

  _initialized = false;
}

void GPURaytracingRenderer::on_camera_changed(SceneRepresentation& scene) {
}

void GPURaytracingRenderer::on_scene_changed(SceneRepresentation& scene) {
  _scene_dirty = true;
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
}

}  // namespace etx
