#include <etx/core/profiler.hxx>
#include <etx/render/host/image_pool.hxx>

#include "render.hxx"

#include <sokol_app.h>
#include <sokol_gfx.h>

#include <vector>

namespace etx {

extern const char* shader_source_hlsl;
extern const char* shader_source_metal;

struct ShaderConstants {
  float4 dimensions = {};
  float exposure = 1.0f;
  uint32_t image_view = 0;
  uint32_t options = ViewOptions::ToneMapping;
  uint32_t sample_count = 0;
};

struct RenderContextImpl {
  RenderContextImpl(TaskScheduler& s)
    : image_pool(s) {
  }

  sg_shader output_shader = {};
  sg_pipeline output_pipeline = {};
  sg_image sample_image = {};
  sg_image reference_image = {};
  ShaderConstants constants;
  uint32_t def_image_handle = kInvalidIndex;
  uint32_t ref_image_handle = kInvalidIndex;
  uint2 output_dimensions = {};
  ImagePool image_pool;

  std::vector<float4> black_image;
};

ETX_PIMPL_IMPLEMENT(RenderContext, Impl);

RenderContext::RenderContext(TaskScheduler& s) {
  ETX_PIMPL_INIT(RenderContext, s);
}

RenderContext::~RenderContext() {
  ETX_PIMPL_CLEANUP(RenderContext);
}

void RenderContext::init() {
  _private->image_pool.init(1024u);
  _private->def_image_handle = _private->image_pool.add_from_file("##default", Image::RepeatU | Image::RepeatV, {});

  sg_desc context = {};
  context.context.d3d11.device = sapp_d3d11_get_device();
  context.context.d3d11.device_context = sapp_d3d11_get_device_context();
  context.context.d3d11.depth_stencil_view_cb = sapp_d3d11_get_depth_stencil_view;
  context.context.d3d11.render_target_view_cb = sapp_d3d11_get_render_target_view;

  context.context.metal.device = sapp_metal_get_device();
  context.context.metal.drawable_cb = []() {
    return sapp_metal_get_drawable();
  };
  context.context.metal.renderpass_descriptor_cb = []() {
    return reinterpret_cast<const void*>(sapp_metal_get_renderpass_descriptor());
  };

  context.context.depth_format = SG_PIXELFORMAT_NONE;
  sg_setup(context);

  sg_shader_desc shader_desc = {};
  shader_desc.vs.entry = "vertex_main";
  shader_desc.vs.uniform_blocks[0].size = sizeof(ShaderConstants);

  shader_desc.fs.entry = "fragment_main";
  shader_desc.fs.images[0].image_type = SG_IMAGETYPE_2D;
  shader_desc.fs.images[0].name = "sample_image";
  shader_desc.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
  shader_desc.fs.images[1].image_type = SG_IMAGETYPE_2D;
  shader_desc.fs.images[1].name = "reference_image";
  shader_desc.fs.images[1].sampler_type = SG_SAMPLERTYPE_FLOAT;
  shader_desc.fs.uniform_blocks[0].size = sizeof(ShaderConstants);

#if (ETX_PLATFORM_WINDOWS)
  shader_desc.vs.source = shader_source_hlsl;
  shader_desc.fs.source = shader_source_hlsl;
#elif (ETX_PLATFORM_APPLE)
  shader_desc.vs.source = shader_source_metal;
  shader_desc.fs.source = shader_source_metal;
#endif

  _private->output_shader = sg_make_shader(shader_desc);

  sg_pipeline_desc pipeline_desc = {};
  pipeline_desc.shader = _private->output_shader;
  _private->output_pipeline = sg_make_pipeline(pipeline_desc);

  apply_reference_image(_private->def_image_handle);

#if (ETX_PLATFORM_WINDOWS)
  set_output_dimensions({16, 16});
  float4 c_image[256] = {};
  for (uint32_t y = 0; y < 16u; ++y) {
    for (uint32_t x = 0; x < 16u; ++x) {
      uint32_t i = x + y * 16u;
      c_image[i] = {1.0f, 0.5f, 0.25f, 1.0f};
    }
  }
  update_image(c_image);
  sg_commit();
#endif
}

void RenderContext::cleanup() {
  sg_destroy_pipeline(_private->output_pipeline);
  sg_destroy_shader(_private->output_shader);
  sg_destroy_image(_private->sample_image);
  sg_destroy_image(_private->reference_image);
  sg_shutdown();

  _private->image_pool.remove(_private->ref_image_handle);
  _private->image_pool.remove(_private->def_image_handle);
  _private->image_pool.cleanup();
}

void RenderContext::start_frame(uint32_t sample_count, const ViewOptions& view_options) {
  ETX_FUNCTION_SCOPE();
  sg_pass_action pass_action = {};
  pass_action.colors[0].load_action = SG_LOADACTION_CLEAR;
  pass_action.colors[0].store_action = SG_STOREACTION_STORE;
  pass_action.colors[0].clear_value = {0.05f, 0.07f, 0.1f, 1.0f};
  sg_apply_viewport(0, 0, sapp_width(), sapp_height(), sg_features().origin_top_left);
  sg_begin_default_pass(&pass_action, sapp_width(), sapp_height());

  _private->constants = {
    {
      sapp_widthf(),
      sapp_heightf(),
      float(_private->output_dimensions.x),
      float(_private->output_dimensions.y),
    },
    view_options.exposure,
    uint32_t(view_options.view),
    view_options.options,
    sample_count,
  };

  sg_range uniform_data = {
    .ptr = &_private->constants,
    .size = sizeof(ShaderConstants),
  };

  sg_bindings bindings = {};
  bindings.fs_images[0] = _private->sample_image;
  bindings.fs_images[1] = _private->reference_image;

  sg_apply_pipeline(_private->output_pipeline);
  sg_apply_bindings(bindings);
  sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, uniform_data);
  sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, uniform_data);
  sg_draw(0, 3, 1);
}

void RenderContext::end_frame() {
  ETX_FUNCTION_SCOPE();
  sg_end_pass();
  sg_commit();
}

void RenderContext::apply_reference_image(uint32_t handle) {
  const auto& img = _private->image_pool.get(handle);

  sg_destroy_image(_private->reference_image);

  sg_image_desc ref_image_desc = {};
  ref_image_desc.type = SG_IMAGETYPE_2D;
  ref_image_desc.width = img.isize.x;
  ref_image_desc.height = img.isize.y;
  ref_image_desc.mag_filter = SG_FILTER_NEAREST;
  ref_image_desc.min_filter = SG_FILTER_NEAREST;
  ref_image_desc.num_mipmaps = 1;
  ref_image_desc.usage = SG_USAGE_STREAM;

  if (img.format == Image::Format::RGBA32F) {
    ref_image_desc.pixel_format = SG_PIXELFORMAT_RGBA32F;
  } else {
    ref_image_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
  }
  _private->reference_image = sg_make_image(ref_image_desc);

  if (img.format == Image::Format::RGBA32F) {
    ref_image_desc.data.subimage[0][0].ptr = img.pixels.f32.a;
    ref_image_desc.data.subimage[0][0].size = sizeof(float4) * img.pixels.f32.count;
  } else {
    ref_image_desc.data.subimage[0][0].ptr = img.pixels.u8.a;
    ref_image_desc.data.subimage[0][0].size = sizeof(ubyte4) * img.pixels.u8.count;
  }
  sg_update_image(_private->reference_image, ref_image_desc.data);
}

void RenderContext::set_reference_image(const char* file_name) {
  _private->image_pool.remove(_private->ref_image_handle);
  _private->ref_image_handle = _private->image_pool.add_from_file(file_name, 0, {});
  apply_reference_image(_private->ref_image_handle);
}

void RenderContext::set_reference_image(const float4 data[], const uint2 dimensions) {
  _private->image_pool.remove(_private->ref_image_handle);
  _private->ref_image_handle = _private->image_pool.add_from_data(data, dimensions, 0u, {});
  apply_reference_image(_private->ref_image_handle);
}

void RenderContext::set_output_dimensions(const uint2& dim) {
  if ((_private->sample_image.id != 0) && (_private->output_dimensions == dim)) {
    return;
  }

  _private->output_dimensions = dim;
  sg_destroy_image(_private->sample_image);

  sg_image_desc desc = {};
  desc.type = SG_IMAGETYPE_2D;
  desc.pixel_format = SG_PIXELFORMAT_RGBA32F;
  desc.width = _private->output_dimensions.x;
  desc.height = _private->output_dimensions.y;
  desc.mag_filter = SG_FILTER_NEAREST;
  desc.min_filter = SG_FILTER_NEAREST;
  desc.num_mipmaps = 1;
  desc.usage = SG_USAGE_STREAM;
  _private->sample_image = sg_make_image(desc);

  _private->black_image.resize(dim.x * dim.y);
  std::fill(_private->black_image.begin(), _private->black_image.end(), float4{});
}

void RenderContext::update_image(const float4* camera) {
  ETX_ASSERT(_private->sample_image.id != 0);
  ETX_FUNCTION_SCOPE();

  sg_image_data data = {};
  data.subimage[0][0].size = sizeof(float4) * _private->output_dimensions.x * _private->output_dimensions.y;
  data.subimage[0][0].ptr = camera ? camera : _private->black_image.data();
  sg_update_image(_private->sample_image, data);
}

const char* shader_source_hlsl = R"(

cbuffer Constants : register(b0) {
  float4 dimensions;
  float exposure;
  uint image_view;
  uint options;
  uint sample_count;
}

Texture2D<float4> sample_image : register(t0);
Texture2D<float4> reference_image : register(t1);

struct VSOutput {
  float4 pos : SV_Position;
  float2 uv : TEXCOORD0;
};

VSOutput vertex_main(uint vertexIndex : SV_VertexID) {
  float2 pos = float2((vertexIndex << 1u) & 2u, vertexIndex & 2u);
  float2 snapped_pos = floor(pos * 2.0f * dimensions.zw - dimensions.zw) / dimensions.xy;

  VSOutput output = (VSOutput)0;
  output.pos = float4(snapped_pos, 0.0f, 1.0f);
  output.uv = pos;
  return output;
}

static const uint kViewResult = 0;
static const uint kViewAlpha = 1;
static const uint kViewOriginal = 2;
static const uint kViewReferenceImage = 3;
static const uint kViewRelativeDifference = 4;
static const uint kViewAbsoluteDifference = 5;

static const uint ToneMapping = 1u << 0u;
static const uint sRGB = 1u << 1u;
static const uint SkipColorConversion = 1u << 2u;

static const float3 lum = float3(0.2627, 0.6780, 0.0593);

float sqr(float t) { 
  return t * t; 
}

float4 validate(in float4 xyz) {
  if (any(isnan(xyz))) {
    return float4(123456.0, 0.0, 123456.0, 1.0);
  }
  if (any(isinf(xyz))) {
    return float4(0.0, 123456.0, 123456.0, 1.0);
  }
  return max(0.0f, xyz);
}

float linear_to_gamma(float value) {
  return value <= 0.0031308f ? (12.92f * value) : (1.055f * pow(abs(value), 1.0f / 2.4f) - 0.055f);
}

float4 tonemap(float4 value) {
  if (options & ToneMapping) {
    value = 1.0f - exp(-exposure * value);
  }

  if (options & sRGB) {
    value.x = linear_to_gamma(value.x);
    value.y = linear_to_gamma(value.y);
    value.z = linear_to_gamma(value.z);
  }

  return value;
}

float4 fragment_main(in VSOutput input) : SV_Target0 {
  float2 offset = 0.5f * (dimensions.xy - dimensions.zw);

  int2 coord = int2(floor(input.pos.xy - offset));
  int2 clamped = clamp(coord.xy, int2(0, 0), int2(dimensions.zw) - 1);
  clip(any(clamped != coord.xy) ? -1 : 1);

  if (any(clamped != coord.xy)) {
    return float4(1.0f, 0.0f, 1.0f, 1.0f);
  }

  int3 load_coord = int3(clamped, 0);

  float4 c_image = sample_image.Load(load_coord);

  if (image_view == kViewAlpha)
    return exposure * c_image.w;

  if (options & SkipColorConversion)
    return exposure * c_image;

  {
    float4 v_image = validate(c_image);
    if (any(v_image != c_image)) {
      return v_image;
    }
  }

  float4 r_image = reference_image.Load(load_coord);
  float r_lum = dot(r_image.xyz, lum);

  float c_lum = dot(c_image.xyz, lum);
  const float c_treshold = 1.0f / 8192.0f;

  float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);
  switch (image_view) {
    case kViewResult: {
      result = tonemap(c_image);
      break;
    }
    case kViewOriginal: {
      result = exposure * c_image;
      break;
    }
    case kViewReferenceImage: {
      result = tonemap(r_image);
      break;
    }
    case kViewRelativeDifference: {
      result.x = exposure * max(0.0f, r_lum - c_lum);
      result.y = exposure * max(0.0f, c_lum - r_lum);
      break;
    }
    case kViewAbsoluteDifference: {
      result.x = float(max(0.0f, r_lum - c_lum) > c_treshold);
      result.y = float(max(0.0f, c_lum - r_lum) > c_treshold);
      break;
    }
    default:
      break;
  };

  return result;
}

)";

const char* shader_source_metal = R"(
using namespace metal;

struct Constants {
  float4 dimensions;
  float exposure;
  uint image_view;
  uint options;
  uint sample_count;
};

struct VSOutput {
  float4 pos [[position]];
  float2 uv;
};

constant constexpr uint ToneMapping = 1u << 0u;
constant constexpr uint sRGB = 1u << 1u;
constant constexpr float c_treshold = 1.0f / 8192.0f;
constant constexpr float3 lum = {0.2627, 0.6780, 0.0593};

constant constexpr uint kViewResult = 0;
constant constexpr uint kViewCameraImage = 1;
constant constexpr uint kViewLightImage = 2;
constant constexpr uint kViewReferenceImage = 3;
constant constexpr uint kViewRelativeDifference = 4;
constant constexpr uint kViewAbsoluteDifference = 5;

float linear_to_gamma(float value) {
  return value <= 0.0031308f ? 12.92f * value : 1.055f * pow(value, 1.0f / 2.4f) - 0.055f;
}

float4 tonemap(float4 value, constant const Constants& params) {
  if (params.options & ToneMapping) {
    value = 1.0f - exp(-params.exposure * value);
  }

  if (params.options & sRGB) {
    value.x = linear_to_gamma(value.x);
    value.y = linear_to_gamma(value.y);
    value.z = linear_to_gamma(value.z);
  }

  return value;
}

vertex VSOutput vertex_main(constant Constants& params [[buffer(0)]], uint vertexIndex [[vertex_id]]) {
  float2 pos = float2((vertexIndex << 1u) & 2u, vertexIndex & 2u);
  float2 snapped_pos = floor(pos * 2.0f * params.dimensions.zw - params.dimensions.zw) / params.dimensions.xy;

  VSOutput output = {};
  output.pos = float4(snapped_pos, 0.0f, 1.0f);
  output.uv = {pos.x, 1.0f - pos.y};
  return output;
}

fragment float4 fragment_main(VSOutput input [[stage_in]],
  constant Constants& params [[buffer(0)]],
  texture2d<float> sample_image [[texture(0)]],
  texture2d<float> reference_image [[texture(1)]]
) {
  constexpr sampler linear_sampler(coord::normalized, address::clamp_to_edge, filter::linear);

  float2 offset = 0.5f * (params.dimensions.xy - params.dimensions.zw);

  int2 coord = int2(floor(input.pos.xy - offset));
  int2 clamped = clamp(coord.xy, int2(0, 0), int2(params.dimensions.zw) - 1);
  if (any(clamped != coord.xy)) {
    discard_fragment();
  }

  float4 sampled_color_xyz = sample_image.sample(linear_sampler, input.uv);

  float4 r_image = reference_image.sample(linear_sampler, input.uv);
  float r_lum = dot(r_image.xyz, lum);

  float4 c_image = sampled_color_xyz;
  float4 v_image = c_image;
  float v_lum = dot(v_image.xyz, lum);

  float4 result = {};

  switch (params.image_view) {
    case kViewResult: {
      result = tonemap(v_image, params);
      break;
    }
    case kViewCameraImage: {
      result = tonemap(c_image, params);
      break;
    }
    case kViewReferenceImage: {
      result = tonemap(r_image, params);
      break;
    }
    case kViewRelativeDifference: {
      result.x = params.exposure * max(0.0f, r_lum - v_lum);
      result.y = params.exposure * max(0.0f, v_lum - r_lum);
      break;
    }
    case kViewAbsoluteDifference: {
      result.x = float(max(0.0f, r_lum - v_lum) > c_treshold);
      result.y = float(max(0.0f, v_lum - r_lum) > c_treshold);
      break;
    }
    default:
      break;
  };

  return result;
}

)";

}  // namespace etx
