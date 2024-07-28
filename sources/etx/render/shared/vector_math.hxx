#if !defined(ETX_MATH_INCLUDES)
# error Do not include this file directly
#endif

ETX_GPU_CODE float4x4 operator*(const float4x4& m1, const float4x4& m2) {
  float4 srca0 = m1.col[0];
  float4 srca1 = m1.col[1];
  float4 srca2 = m1.col[2];
  float4 srca3 = m1.col[3];
  float4 srcb0 = m2.col[0];
  float4 srcb1 = m2.col[1];
  float4 srcb2 = m2.col[2];
  float4 srcb3 = m2.col[3];

  float4x4 result;
  result.col[0] = srca0 * srcb0.x + srca1 * srcb0.y + srca2 * srcb0.z + srca3 * srcb0.w;
  result.col[1] = srca0 * srcb1.x + srca1 * srcb1.y + srca2 * srcb1.z + srca3 * srcb1.w;
  result.col[2] = srca0 * srcb2.x + srca1 * srcb2.y + srca2 * srcb2.z + srca3 * srcb2.w;
  result.col[3] = srca0 * srcb3.x + srca1 * srcb3.y + srca2 * srcb3.z + srca3 * srcb3.w;
  return result;
}

ETX_GPU_CODE float3 operator*(const float3x3& m, const float3& v) {
  return float3{
    m.col[0].x * v.x + m.col[1].x * v.y + m.col[2].x * v.z,
    m.col[0].y * v.x + m.col[1].y * v.y + m.col[2].y * v.z,
    m.col[0].z * v.x + m.col[1].z * v.y + m.col[2].z * v.z,
  };
}

ETX_GPU_CODE float4 operator*(const float4x4& m, const float4& v) {
  return float4{
    m.col[0].x * v.x + m.col[1].x * v.y + m.col[2].x * v.z + m.col[3].x * v.w,
    m.col[0].y * v.x + m.col[1].y * v.y + m.col[2].y * v.z + m.col[3].y * v.w,
    m.col[0].z * v.x + m.col[1].z * v.y + m.col[2].z * v.z + m.col[3].z * v.w,
    m.col[0].w * v.x + m.col[1].w * v.y + m.col[2].w * v.z + m.col[3].w * v.w,
  };
}

ETX_GPU_CODE float4x4 inverse(const float4x4& m) {
  float coef00 = m.col[2].z * m.col[3].w - m.col[3].z * m.col[2].w;
  float coef02 = m.col[1].z * m.col[3].w - m.col[3].z * m.col[1].w;
  float coef03 = m.col[1].z * m.col[2].w - m.col[2].z * m.col[1].w;
  float coef04 = m.col[2].y * m.col[3].w - m.col[3].y * m.col[2].w;
  float coef06 = m.col[1].y * m.col[3].w - m.col[3].y * m.col[1].w;
  float coef07 = m.col[1].y * m.col[2].w - m.col[2].y * m.col[1].w;
  float coef08 = m.col[2].y * m.col[3].z - m.col[3].y * m.col[2].z;
  float coef10 = m.col[1].y * m.col[3].z - m.col[3].y * m.col[1].z;
  float coef11 = m.col[1].y * m.col[2].z - m.col[2].y * m.col[1].z;
  float coef12 = m.col[2].x * m.col[3].w - m.col[3].x * m.col[2].w;
  float coef14 = m.col[1].x * m.col[3].w - m.col[3].x * m.col[1].w;
  float coef15 = m.col[1].x * m.col[2].w - m.col[2].x * m.col[1].w;
  float coef16 = m.col[2].x * m.col[3].z - m.col[3].x * m.col[2].z;
  float coef18 = m.col[1].x * m.col[3].z - m.col[3].x * m.col[1].z;
  float coef19 = m.col[1].x * m.col[2].z - m.col[2].x * m.col[1].z;
  float coef20 = m.col[2].x * m.col[3].y - m.col[3].x * m.col[2].y;
  float coef22 = m.col[1].x * m.col[3].y - m.col[3].x * m.col[1].y;
  float coef23 = m.col[1].x * m.col[2].y - m.col[2].x * m.col[1].y;

  float4 fac0 = {coef00, coef00, coef02, coef03};
  float4 fac1 = {coef04, coef04, coef06, coef07};
  float4 fac2 = {coef08, coef08, coef10, coef11};
  float4 fac3 = {coef12, coef12, coef14, coef15};
  float4 fac4 = {coef16, coef16, coef18, coef19};
  float4 fac5 = {coef20, coef20, coef22, coef23};

  float4 vec0 = {m.col[1].x, m.col[0].x, m.col[0].x, m.col[0].x};
  float4 vec1 = {m.col[1].y, m.col[0].y, m.col[0].y, m.col[0].y};
  float4 vec2 = {m.col[1].z, m.col[0].z, m.col[0].z, m.col[0].z};
  float4 vec3 = {m.col[1].w, m.col[0].w, m.col[0].w, m.col[0].w};

  float4 inv0 = {vec1 * fac0 - vec2 * fac1 + vec3 * fac2};
  float4 inv1 = {vec0 * fac0 - vec2 * fac3 + vec3 * fac4};
  float4 inv2 = {vec0 * fac1 - vec1 * fac3 + vec3 * fac5};
  float4 inv3 = {vec0 * fac2 - vec1 * fac4 + vec2 * fac5};

  float4 signa = {+1.0f, -1.0f, +1.0f, -1.0f};
  float4 signb = {-1.0f, +1.0f, -1.0f, +1.0f};

  float4x4 inverse = {inv0 * signa, inv1 * signb, inv2 * signa, inv3 * signb};
  float4 row0 = {inverse.col[0].x, inverse.col[1].x, inverse.col[2].x, inverse.col[3].x};
  float4 dot0 = {m.col[0] * row0};
  float dot1 = (dot0.x + dot0.y) + (dot0.z + dot0.w);
  inverse.col[0] /= dot1;
  inverse.col[1] /= dot1;
  inverse.col[2] /= dot1;
  inverse.col[3] /= dot1;
  return inverse;
}

ETX_GPU_CODE float4x4 look_at(const float3& origin, const float3& target, const float3& up) {
  auto f = normalize(target - origin);
  auto s = normalize(cross(f, up));
  auto u = cross(s, f);

  float4x4 result = {};
  result.col[0].x = s.x;
  result.col[1].x = s.y;
  result.col[2].x = s.z;
  result.col[0].y = u.x;
  result.col[1].y = u.y;
  result.col[2].y = u.z;
  result.col[0].z = -f.x;
  result.col[1].z = -f.y;
  result.col[2].z = -f.z;
  result.col[3].x = -dot(s, origin);
  result.col[3].y = -dot(u, origin);
  result.col[3].z = dot(f, origin);
  result.col[3].w = 1.0f;
  return result;
}

ETX_GPU_CODE float4x4 perspective(float fov, uint32_t width, uint32_t height, float z_near, float z_far) {
  ETX_ASSERT(width > 0);
  ETX_ASSERT(height > 0);
  ETX_ASSERT(fov > 0);

  float h = cosf(0.5f * fov) / sinf(0.5f * fov);
  float aspect = float(width) / float(height);

  float4x4 result = {};
  result.col[0].x = h / aspect;
  result.col[1].y = h;
  result.col[2].z = z_far / (z_near - z_far);
  result.col[2].w = -1.0f;
  result.col[3].z = -(z_far * z_near) / (z_far - z_near);
  return result;
}
