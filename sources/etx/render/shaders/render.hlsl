#include "bindless.hlsl"
#include <interop/render_options.hxx>

[[vk::push_constant]] RenderParameters options;

static const float3 lum = float3(0.2627, 0.6780, 0.0593);

float4 validate(in float4 xyz) {
  if (any(isnan(xyz))) {
    return float4(123456.0, 0.0, 123456.0, 1.0);
  }
  if (any(isinf(xyz))) {
    return float4(0.0, 123456.0, 123456.0, 1.0);
  }
  return xyz;
}

float linear_to_gamma(float value) {
  return value <= 0.0031308f ? (12.92f * value) : (1.055f * pow(abs(value), 1.0f / 2.4f) - 0.055f);
}

float4 tonemap(float4 value) {
  switch (options.view.view_option) {
    case ViewOptions::Tonemapped: {
      value = 1.0f - exp(-options.view.exposure * value);
#if !defined(ETX_PRESENT_SRGB_TARGET)
      value.x = linear_to_gamma(value.x);
      value.y = linear_to_gamma(value.y);
      value.z = linear_to_gamma(value.z);
#endif
      break;
    }
    case ViewOptions::HDR: {
      value = options.view.exposure * value;
      break;
    }
    case ViewOptions::Normalized: {
      value.xyz = normalize(value.xyz) * 0.5f + 0.5f;
      break;
    }
    default:
      break;
  }
  return value;
}

struct VSOutput {
  float4 pos : SV_Position;
  float2 uv : TEXCOORD0;
};

VSOutput vertex_main(uint vertexIndex : SV_VertexID) {
  float2 pos = float2((vertexIndex << 1u) & 2u, vertexIndex & 2u);
  float2 snapped_pos = floor(pos * 2.0f * options.dimensions.zw - options.dimensions.zw) / options.dimensions.xy;

  VSOutput output = (VSOutput)0;
  output.pos = float4(snapped_pos, 0.0f, 1.0f);
  output.uv = pos;
  return output;
}

float4 fragment_main(in VSOutput input)
  : SV_Target0 {
  float2 offset = 0.5f * (options.dimensions.xy - options.dimensions.zw);

  const float2 viewport_position = input.pos.xy - options.viewport.xy;
  int2 coord = int2(floor(viewport_position - offset));
  int2 clamped = clamp(coord.xy, int2(0, 0), int2(options.dimensions.zw) - 1);
  clip(any(clamped != coord.xy) ? -1 : 1);

  if (any(clamped != coord.xy)) {
    return float4(1.0f, 0.0f, 1.0f, 1.0f);
  }

  const Texture2D sample_image = bindless_textures[options.sample_image_index];
  uint sample_width = 0;
  uint sample_height = 0;
  sample_image.GetDimensions(sample_width, sample_height);
  float2 sample_uv = (float2(clamped) + 0.5f) / options.dimensions.zw;
  int2 sample_coord = min(int2(sample_uv * float2(sample_width, sample_height)), int2(sample_width, sample_height) - 1);
  int3 load_coord = int3(sample_coord, 0);
  float4 c_image = sample_image.Load(load_coord);

  if (options.view.view_image == OutputView::AlphaChannel)
    return options.view.exposure * c_image.w;

  float4 v_image = validate(c_image);
  if (any(v_image != c_image)) {
    return v_image;
  }

  float c_lum = dot(c_image.xyz, lum);
  const float c_treshold = 1.0f / 8192.0f;

  float4 result = float4(0.0f, 0.0f, 0.0f, 1.0f);
  switch (options.view.view_image) {
    case OutputView::OutputImage: {
      result = tonemap(c_image);
      break;
    }
    case OutputView::ReferenceImage: {
      const Texture2D reference_image = bindless_textures[options.reference_image_index];
      float4 r_image = reference_image.Load(int3(clamped, 0));
      result = tonemap(r_image);
      break;
    }
    case OutputView::RelativeDifference: {
      const Texture2D reference_image = bindless_textures[options.reference_image_index];
      float4 r_image = reference_image.Load(int3(clamped, 0));
      float r_lum = dot(r_image.xyz, lum);
      result.x = options.view.exposure * max(0.0f, r_lum - c_lum);
      result.y = options.view.exposure * max(0.0f, c_lum - r_lum);
      result = tonemap(result);
      break;
    }
    case OutputView::AbsoluteDifference: {
      const Texture2D reference_image = bindless_textures[options.reference_image_index];
      float4 r_image = reference_image.Load(int3(clamped, 0));
      float r_lum = dot(r_image.xyz, lum);
      result.x = float(max(0.0f, r_lum - c_lum) > c_treshold);
      result.y = float(max(0.0f, c_lum - r_lum) > c_treshold);
      break;
    }
    default:
      break;
  };

  return result;
}
