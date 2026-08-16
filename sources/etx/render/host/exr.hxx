#pragma once

#include <etx/render/shared/math.hxx>

#include <string>
#include <vector>

namespace etx {

bool load_exr_image(const char* path, std::vector<float4>& pixels, uint2& dimensions, std::string* error = nullptr);
bool save_exr_image(const char* path, const float4* pixels, uint2 dimensions, std::string* error = nullptr);

}  // namespace etx
