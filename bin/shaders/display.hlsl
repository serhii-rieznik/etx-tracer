
struct ViewParameters {
  float exposure;
  uint view_option;
  uint view_image;
  uint view_layer;
};

struct RenderParameters {
  ViewParameters view;
  float4 dimensions;
  uint sample_count;
  uint sample_image_index;
  uint reference_image_index;
  uint pad;
};

[[vk::push_constant]]
RenderParameters params;

Texture2D GlobalTextures[] : register(t1, space0);

struct VSOutput {
  float4 pos : SV_Position;
  float2 uv : TEXCOORD0;
};

VSOutput vertex_main(uint vertexIndex : SV_VertexID) {
  float2 pos = float2((vertexIndex << 1u) & 2u, vertexIndex & 2u);
  float2 snapped_pos = floor(pos * 2.0f * params.dimensions.zw - params.dimensions.zw) / params.dimensions.xy;

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

static const uint ToneMapping = 0; // ViewOptions::Tonemapped
static const uint HDR = 1;         // ViewOptions::HDR
static const uint Normalized = 2;  // ViewOptions::Normalized

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
  if (params.view.view_option == ToneMapping) {
    value = 1.0f - exp(-params.view.exposure * value);
  }

  if (params.view.view_option != HDR) {
    value.x = linear_to_gamma(value.x);
    value.y = linear_to_gamma(value.y);
    value.z = linear_to_gamma(value.z);
  }

  return value;
}

float4 fragment_main(in VSOutput input) : SV_Target0 {
  float2 offset = 0.5f * (params.dimensions.xy - params.dimensions.zw);

  int2 coord = int2(floor(input.pos.xy - offset));
  int2 clamped = clamp(coord.xy, int2(0, 0), int2(params.dimensions.zw) - 1);
  clip(any(clamped != coord.xy) ? -1 : 1);

  if (any(clamped != coord.xy)) {
    return float4(1.0f, 0.0f, 1.0f, 1.0f);
  }

  int3 load_coord = int3(clamped, 0);

  Texture2D sample_image = GlobalTextures[params.sample_image_index];
  float4 c_image = sample_image.Load(load_coord);

  if (params.view.view_image == kViewAlpha)
    return params.view.exposure * c_image.w;

  if (params.view.view_option == Normalized)
    return params.view.exposure * c_image;

  {
    float4 v_image = validate(c_image);
    if (any(v_image != c_image)) {
      return v_image;
    }
  }

  Texture2D reference_image = GlobalTextures[params.reference_image_index];
  float4 r_image = reference_image.Load(load_coord);
  float r_lum = dot(r_image.xyz, lum);

  float c_lum = dot(c_image.xyz, lum);
  const float c_treshold = 1.0f / 8192.0f;

  float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);
  switch (params.view.view_image) {
    case kViewResult: {
      result = tonemap(c_image);
      break;
    }
    case kViewOriginal: {
      result = params.view.exposure * c_image;
      break;
    }
    case kViewReferenceImage: {
      result = tonemap(r_image);
      break;
    }
    case kViewRelativeDifference: {
      result.x = params.view.exposure * max(0.0f, r_lum - c_lum);
      result.y = params.view.exposure * max(0.0f, c_lum - r_lum);
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
