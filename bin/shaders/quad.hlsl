#include "bindless.hlsl"

// Vertex structure matching CPU side
struct Vertex {
  float2 position;
  uint quad_index;
};

// Vertex to pixel structure (shared between vertex and fragment shaders)
struct VertexToPixel {
  float4 position : SV_Position;
  nointerpolation uint quad_index : BLENDINDICES0;
};

// Push constants - shared between graphics and compute pipelines
struct PushConstants {
  uint texture_index;
  uint vertex_buffer_index;
  uint viewport_width;
  uint viewport_height;
  uint texture_width;
  uint texture_height;

  // Compute-specific fields
  uint noise_seed;
  float noise_scale;

  float3 quad_color;
};

[[vk::push_constant]] PushConstants pushConstants;

// Vertex shader that reads from bindless vertex buffer
VertexToPixel vs_main(uint vertex_id : SV_VertexID) {
  VertexToPixel output;

  // Read vertex data from bindless buffer using raw byte access
  ByteAddressBuffer vertex_buffer = bindless_buffers[pushConstants.vertex_buffer_index];
  Vertex vertex = vertex_buffer.Load<Vertex>(vertex_id * sizeof(Vertex));

  output.position = float4(vertex.position.x, vertex.position.y, 0.0f, 1.0f);
  output.quad_index = vertex.quad_index;

  return output;
}

//==============================================================================
// Fragment Shader
//==============================================================================

// Fragment output
struct FSOutput {
  float4 color : SV_Target0;
};

// Simple fragment shader that displays full texture (0,0 to 1,1) in each quad
FSOutput fs_main(VertexToPixel input) {
  FSOutput output;

  // Convert pixel coordinates back to NDC for proper UV calculation
  float ndc_x = (input.position.x * 2.0 / pushConstants.viewport_width) - 1.0;
  float ndc_y = (input.position.y * 2.0 / pushConstants.viewport_height) - 1.0;

  // Each quad displays UV gradient from 0 to 1 in both U and V directions
  // Map each quad's NDC range to UV (0,0) to (1,1)
  float2 uv;

  if (input.quad_index == 0) {
    // Top-left quad: NDC x(-1,0) y(0,1) -> UV (0,0) to (1,1)
    uv = float2((ndc_x + 1.0) / 1.0,  // (-1,0) -> (0,1)
      (ndc_y - 0.0) / 1.0             // (0,1) -> (0,1)
    );
  } else if (input.quad_index == 1) {
    // Top-right quad: NDC x(0,1) y(0,1) -> UV (0,0) to (1,1)
    uv = float2((ndc_x - 0.0) / 1.0,  // (0,1) -> (0,1)
      (ndc_y - 0.0) / 1.0             // (0,1) -> (0,1)
    );
  } else if (input.quad_index == 2) {
    // Bottom-left quad: NDC x(-1,0) y(-1,0) -> UV (0,0) to (1,1)
    uv = float2((ndc_x + 1.0) / 1.0,  // (-1,0) -> (0,1)
      (ndc_y + 1.0) / 1.0             // (-1,0) -> (0,1)
    );
  } else {
    // Bottom-right quad: NDC x(0,1) y(-1,0) -> UV (0,0) to (1,1)
    uv = float2((ndc_x - 0.0) / 1.0,  // (0,1) -> (0,1)
      (ndc_y + 1.0) / 1.0             // (-1,0) -> (0,1)
    );
  }

  // Determine which sampler to use based on quad index
  uint sampler_index;
  if (input.quad_index == 0) {
    sampler_index = SamplerType::LinearRepeat;
  } else if (input.quad_index == 1) {
    sampler_index = SamplerType::LinearClamp;
  } else if (input.quad_index == 2) {
    sampler_index = SamplerType::NearestRepeat;
  } else {
    sampler_index = SamplerType::NearestClamp;
  }

  // Sample the noise texture with UV coordinates
  float2 sample_uv = 0.5f * uv * float2(pushConstants.viewport_width, pushConstants.viewport_height) / float2(pushConstants.texture_width, pushConstants.texture_height);
  float4 sampled_color = SampleTexture(pushConstants.texture_index, sampler_index, sample_uv);
  float noise = sampled_color.r;

  // Display UV gradient: Red=U, Green=V, Blue=0, modulated by noise and quad color
  float3 color = float3(uv.x, uv.y, 0.0) * (noise * 0.8 + 0.2) * pushConstants.quad_color;

  output.color = float4(color, 1.0);
  return output;
}

//==============================================================================
// Compute Shader for Noise Generation
//==============================================================================

// Simple hash function for noise generation
uint hash(uint x, uint y, uint seed) {
  uint h = x * 73856093 ^ y * 19349663 ^ seed * 83492791;
  h = (h >> 13) ^ h;
  return (h * (h * h * 15731 + 789221) + 1376312589) & 0x7fffffff;
}

// Generate procedural noise value
float generate_noise(uint x, uint y, uint seed) {
  uint h = hash(uint(float(x) * pushConstants.noise_scale), uint(float(y) * pushConstants.noise_scale), seed);
  return float(h) / float(0x7fffffff);
}

// Compute shader entry point
[numthreads(8, 8, 1)] void cs_main(uint3 dispatch_id
                                   : SV_DispatchThreadID) {
  if (dispatch_id.x >= pushConstants.texture_width || dispatch_id.y >= pushConstants.texture_height) {
    return;
  }

  // Generate noise value
  float noise = generate_noise(dispatch_id.x, dispatch_id.y, pushConstants.noise_seed);

  // Write to output texture (assuming it's bound as RWTexture2D)
  RWTexture2D<float4> output_texture = bindless_storage_textures[pushConstants.texture_index];
  output_texture[dispatch_id.xy] = float4(noise, noise, noise, 1.0f);
}