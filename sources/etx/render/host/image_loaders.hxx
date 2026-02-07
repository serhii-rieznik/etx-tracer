#pragma once

#include <etx/render/shared/image.hxx>
namespace etx {

Image::Format load_data(const char* source, std::vector<uint8_t>& data, uint2& dimensions);
Image::Format load_dds(const char* source, std::vector<uint8_t>& data, uint2& dimensions);
bool load_pfm(const char* path, uint2& size, std::vector<uint8_t>& data);

}  // namespace etx
