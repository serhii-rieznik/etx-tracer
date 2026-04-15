#pragma once

#include <cstddef>
#include <cstdint>

#include <vector>

namespace etx {

struct H264EncodedFrame {
  std::vector<uint8_t> data = {};
  uint64_t timestamp_us = 0u;
  bool keyframe = false;
};

struct MediaFoundationH264EncoderConfig {
  uint32_t width = 1920u;
  uint32_t height = 1080u;
  uint32_t fps = 60u;
  uint32_t target_bitrate = 20u * 1024u * 1024u;
};

struct MediaFoundationH264Encoder {
  bool init(const MediaFoundationH264EncoderConfig& config);
  void shutdown();

  void request_keyframe();
  bool encode_bgra_frame(const uint8_t* bgra_data, size_t data_size, uint64_t timestamp_us, std::vector<H264EncodedFrame>& output_frames);

  bool initialized() const {
    return _initialized;
  }

private:
  bool configure_encoder();
  bool create_input_sample(const uint8_t* bgra_data, uint64_t timestamp_us, void** out_sample);
  bool drain_output(uint64_t timestamp_us, std::vector<H264EncodedFrame>& output_frames);

private:
  MediaFoundationH264EncoderConfig _config = {};
  void* _encoder = nullptr;
  void* _output_type = nullptr;
  bool _initialized = false;
  bool _mf_started = false;
  bool _com_initialized = false;
  bool _pending_keyframe = false;
  std::vector<uint8_t> _input_nv12 = {};
  std::vector<uint8_t> _parameter_sets = {};
};

}  // namespace etx
