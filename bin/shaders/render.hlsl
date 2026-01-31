#include "bindless.hlsl"
#include "shared/render_options.hxx"

[[vk::push_constant]] ShaderConstants options;

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

// Texture2D<float4> sample_image : register(t0);
// Texture2D<float4> reference_image : register(t1);

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
    if (options.options & ToneMapping) {
        value = 1.0f - exp(-options.exposure * value);
    }

    if (options.options & sRGB) {
        value.x = linear_to_gamma(value.x);
        value.y = linear_to_gamma(value.y);
        value.z = linear_to_gamma(value.z);
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

float4 fragment_main(in VSOutput input) : SV_Target0 {
    float2 offset = 0.5f * (options.dimensions.xy - options.dimensions.zw);

    int2 coord = int2(floor(input.pos.xy - offset));
    int2 clamped = clamp(coord.xy, int2(0, 0), int2(options.dimensions.zw) - 1);
    clip(any(clamped != coord.xy) ? -1 : 1);

    if (any(clamped != coord.xy)) {
        return float4(1.0f, 0.0f, 1.0f, 1.0f);
    }

    int3 load_coord = int3(clamped, 0);

    const Texture2D sample_image = bindless_textures[options.sample_image_index];
    float4 c_image = sample_image.Load(load_coord);

    if (options.image_view == kViewAlpha)
        return options.exposure * c_image.w;

    if (options.options & SkipColorConversion)
        return options.exposure * c_image;

  {
        float4 v_image = validate(c_image);
        if (any(v_image != c_image)) {
            return v_image;
        }
    }

    const Texture2D reference_image = bindless_textures[options.reference_image_index];
    float4 r_image = reference_image.Load(load_coord);
    float r_lum = dot(r_image.xyz, lum);

    float c_lum = dot(c_image.xyz, lum);
    const float c_treshold = 1.0f / 8192.0f;

    float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);
    switch (options.image_view) {
        case kViewResult:{
                result = tonemap(c_image);
                break;
            }
        case kViewOriginal:{
                result = options.exposure * c_image;
                break;
            }
        case kViewReferenceImage:{
                result = tonemap(r_image);
                break;
            }
        case kViewRelativeDifference:{
                result.x = options.exposure * max(0.0f, r_lum - c_lum);
                result.y = options.exposure * max(0.0f, c_lum - r_lum);
                break;
            }
        case kViewAbsoluteDifference:{
                result.x = float(max(0.0f, r_lum - c_lum) > c_treshold);
                result.y = float(max(0.0f, c_lum - r_lum) > c_treshold);
                break;
            }
        default:
            break;
    };

    return result;
}