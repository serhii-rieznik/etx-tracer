﻿#include "render.hxx"

#include <sokol_app.h>
#include <sokol_gfx.h>

namespace etx {

extern const char* shader_source;
constexpr uint32_t image_size[2] = {640u, 480u};

struct ShaderConstants {
  float transform[4] = {};
  float dimensions[4] = {};
};

struct RenderContextPrivate {
  sg_shader output_shader = {};
  sg_pipeline output_pipeline = {};
  sg_image sample_image = {};
  sg_image light_image = {};
  sg_image reference_image = {};
  ShaderConstants constants;
};

ETX_PIMPL_IMPLEMENT_ALL(RenderContext, Private);

void RenderContext::init() {
  sg_desc context = {};
  context.context.d3d11.device = sapp_d3d11_get_device();
  context.context.d3d11.device_context = sapp_d3d11_get_device_context();
  context.context.d3d11.depth_stencil_view_cb = sapp_d3d11_get_depth_stencil_view;
  context.context.d3d11.render_target_view_cb = sapp_d3d11_get_render_target_view;
  context.context.depth_format = SG_PIXELFORMAT_NONE;
  sg_setup(context);

  sg_shader_desc shader_desc = {};
  shader_desc.vs.source = shader_source;
  shader_desc.vs.entry = "vertex_main";
  shader_desc.vs.uniform_blocks[0].size = sizeof(ShaderConstants);

  shader_desc.fs.source = shader_source;
  shader_desc.fs.entry = "fragment_main";
  shader_desc.fs.images[0].image_type = SG_IMAGETYPE_2D;
  shader_desc.fs.images[0].name = "sample_image";
  shader_desc.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
  shader_desc.fs.images[1].image_type = SG_IMAGETYPE_2D;
  shader_desc.fs.images[1].name = "light_image";
  shader_desc.fs.images[1].sampler_type = SG_SAMPLERTYPE_FLOAT;
  shader_desc.fs.images[2].image_type = SG_IMAGETYPE_2D;
  shader_desc.fs.images[2].name = "reference_image";
  shader_desc.fs.images[2].sampler_type = SG_SAMPLERTYPE_FLOAT;
  shader_desc.fs.uniform_blocks[0].size = sizeof(ShaderConstants);
  _private->output_shader = sg_make_shader(shader_desc);

  sg_pipeline_desc pipeline_desc = {};
  pipeline_desc.shader = _private->output_shader;
  _private->output_pipeline = sg_make_pipeline(pipeline_desc);

  sg_image_desc sample_image_desc = {};
  sample_image_desc.type = SG_IMAGETYPE_2D;
  sample_image_desc.pixel_format = SG_PIXELFORMAT_RGBA32F;
  sample_image_desc.width = image_size[0];
  sample_image_desc.height = image_size[1];
  sample_image_desc.mag_filter = SG_FILTER_NEAREST;
  sample_image_desc.min_filter = SG_FILTER_NEAREST;
  sample_image_desc.num_mipmaps = 1;
  sample_image_desc.usage = SG_USAGE_STREAM;
  _private->sample_image = sg_make_image(sample_image_desc);

  sample_image_desc.data.subimage[0][0].size = image_size[0] * image_size[1] * sizeof(float) * 4;
  sample_image_desc.data.subimage[0][0].ptr = malloc(sample_image_desc.data.subimage[0][0].size);
  float* ptr = (float*)sample_image_desc.data.subimage[0][0].ptr;
  for (uint32_t y = 0; y < image_size[1]; ++y) {
    for (uint32_t x = 0; x < image_size[0]; ++x) {
      uint32_t i = x + y * image_size[0];
      if ((x % 2) == (y % 2)) {
        ptr[4 * i + 0u] = 1.0f;
        ptr[4 * i + 1u] = 0.5f;
        ptr[4 * i + 2u] = 0.25f;
        ptr[4 * i + 3u] = 1.0f;
      } else {
        ptr[4 * i + 0u] = 1.0f - 1.0f;
        ptr[4 * i + 1u] = 1.0f - 0.5f;
        ptr[4 * i + 2u] = 1.0f - 0.25f;
        ptr[4 * i + 3u] = 1.0f;
      }
    }
  }
  sg_update_image(_private->sample_image, sample_image_desc.data);

  sg_image_desc light_image_desc = {};
  light_image_desc.type = SG_IMAGETYPE_2D;
  light_image_desc.pixel_format = SG_PIXELFORMAT_RGBA32F;
  light_image_desc.width = image_size[0];
  light_image_desc.height = image_size[1];
  light_image_desc.mag_filter = SG_FILTER_NEAREST;
  light_image_desc.min_filter = SG_FILTER_NEAREST;
  light_image_desc.num_mipmaps = 1;
  light_image_desc.usage = SG_USAGE_STREAM;
  _private->light_image = sg_make_image(light_image_desc);

  light_image_desc.data.subimage[0][0].size = image_size[0] * image_size[1] * sizeof(float) * 4;
  light_image_desc.data.subimage[0][0].ptr = malloc(light_image_desc.data.subimage[0][0].size);
  ptr = (float*)light_image_desc.data.subimage[0][0].ptr;
  for (uint32_t y = 0; y < image_size[1]; ++y) {
    for (uint32_t x = 0; x < image_size[0]; ++x) {
      uint32_t i = x + y * image_size[0];
      if ((x % 2) == (y % 2)) {
        ptr[4 * i + 0u] = 0.5f;
        ptr[4 * i + 1u] = 1.0f;
        ptr[4 * i + 2u] = 0.25f;
        ptr[4 * i + 3u] = 1.0f;
      } else {
        ptr[4 * i + 0u] = 1.0f - 0.5f;
        ptr[4 * i + 1u] = 1.0f - 1.0f;
        ptr[4 * i + 2u] = 1.0f - 0.25f;
        ptr[4 * i + 3u] = 1.0f;
      }
    }
  }
  sg_update_image(_private->light_image, light_image_desc.data);

  sg_image_desc ref_image_desc = {};
  ref_image_desc.type = SG_IMAGETYPE_2D;
  ref_image_desc.pixel_format = SG_PIXELFORMAT_RGBA32F;
  ref_image_desc.width = image_size[0];
  ref_image_desc.height = image_size[1];
  ref_image_desc.mag_filter = SG_FILTER_NEAREST;
  ref_image_desc.min_filter = SG_FILTER_NEAREST;
  ref_image_desc.num_mipmaps = 1;
  ref_image_desc.usage = SG_USAGE_STREAM;
  _private->reference_image = sg_make_image(ref_image_desc);

  ref_image_desc.data.subimage[0][0].size = image_size[0] * image_size[1] * sizeof(float) * 4;
  ref_image_desc.data.subimage[0][0].ptr = malloc(ref_image_desc.data.subimage[0][0].size);
  ptr = (float*)ref_image_desc.data.subimage[0][0].ptr;
  for (uint32_t y = 0; y < image_size[1]; ++y) {
    for (uint32_t x = 0; x < image_size[0]; ++x) {
      uint32_t i = x + y * image_size[0];
      if ((x % 2) == (y % 2)) {
        ptr[4 * i + 0u] = 0.25f;
        ptr[4 * i + 1u] = 0.5f;
        ptr[4 * i + 2u] = 1.0f;
        ptr[4 * i + 3u] = 1.0f;
      } else {
        ptr[4 * i + 0u] = 1.0f - 0.25f;
        ptr[4 * i + 1u] = 1.0f - 0.5f;
        ptr[4 * i + 2u] = 1.0f - 1.0f;
        ptr[4 * i + 3u] = 1.0f;
      }
    }
  }
  sg_update_image(_private->reference_image, ref_image_desc.data);
}

void RenderContext::cleanup() {
  sg_destroy_pipeline(_private->output_pipeline);
  sg_destroy_shader(_private->output_shader);
  sg_destroy_image(_private->sample_image);
  sg_destroy_image(_private->light_image);
  sg_destroy_image(_private->reference_image);
  sg_shutdown();
}

void RenderContext::start_frame() {
  sg_pass_action pass_action = {};
  pass_action.colors[0].action = SG_ACTION_CLEAR;
  pass_action.colors[0].value = {0.05f, 0.07f, 0.1f, 1.0f};
  sg_apply_viewport(0, 0, sapp_width(), sapp_height(), true);
  sg_begin_default_pass(&pass_action, sapp_width(), sapp_height());

  _private->constants = {
    {1.0f, 1.0f, 0.0f, 0.0f},
    {sapp_widthf(), sapp_heightf(), float(image_size[0]), float(image_size[1])},
  };

  sg_range uniform_data = {
    .ptr = &_private->constants,
    .size = sizeof(ShaderConstants),
  };

  sg_bindings bindings = {};
  bindings.fs_images[0] = _private->sample_image;
  bindings.fs_images[1] = _private->light_image;
  bindings.fs_images[2] = _private->reference_image;

  sg_apply_pipeline(_private->output_pipeline);
  sg_apply_bindings(bindings);
  sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, uniform_data);
  sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, uniform_data);
  sg_apply_scissor_rect((sapp_width() - image_size[0]) / 2, (sapp_height() - image_size[1]) / 2, image_size[0], image_size[1], true);
  sg_draw(0, 3, 1);
}

void RenderContext::end_frame() {
  sg_end_pass();
  sg_commit();
}

const char* shader_source = R"(

cbuffer Constants : register(b0) {
  float4 transform;
  float4 dimensions;
}

Texture2D<float4> sample_image : register(t0);
Texture2D<float4> light_image : register(t1);
Texture2D<float4> reference_image : register(t2);

struct VSOutput {
  float4 pos : SV_Position;
  float2 uv : TEXCOORD0;
};

VSOutput vertex_main(uint vertexIndex : SV_VertexID) {
  float2 pos = float2((vertexIndex << 1u) & 2u, vertexIndex & 2u);

  VSOutput output = (VSOutput)0;
  output.pos = float4((pos * 2.0f - 1.0f) * dimensions.zw / dimensions.xy, 0.0f, 1.0f);
  output.pos.xy = floor(output.pos.xy * dimensions.xy) / dimensions.xy;
  output.uv = pos;
  return output;
}

float4 fragment_main(in VSOutput input) : SV_Target0 {
  if (any(saturate(input.uv) != input.uv)) {
    return float4(1.0f, 0.0f, 1.0f, 1.0f);
  }

  int px = input.uv.x * dimensions.z;
  int py = input.uv.y * dimensions.w;
  int3 coord = int3(px, py, 0);

  if (input.uv.x < 1.0f / 3.0f) {
    return sample_image.Load(coord);
  }

  if (input.uv.x < 2.0f / 3.0f) {
    return light_image.Load(coord);
  }

  return reference_image.Load(coord);
}

)";

}  // namespace etx
