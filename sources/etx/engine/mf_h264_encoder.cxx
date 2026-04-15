#include "mf_h264_encoder.hxx"

#include <etx/core/log.hxx>

#include <algorithm>
#include <cstring>
#include <limits>

#if ETX_PLATFORM_WINDOWS
# include <Windows.h>
# include <codecapi.h>
# include <mfapi.h>
# include <mferror.h>
# include <mfidl.h>
# include <mfobjects.h>
# include <mftransform.h>
# include <propvarutil.h>
# include <wrl/client.h>
# include <wmcodecdsp.h>
#endif

namespace etx {
namespace {

#if ETX_PLATFORM_WINDOWS
using Microsoft::WRL::ComPtr;

constexpr uint32_t k_h264_payload_type = 96u;
constexpr uint64_t k_hns_per_second = 10'000'000ull;
constexpr uint32_t k_encoder_peak_bitrate_num = 3u;
constexpr uint32_t k_encoder_peak_bitrate_den = 2u;

template <typename T>
T* as_interface(void* value) {
  return reinterpret_cast<T*>(value);
}

bool starts_with_annex_b(const uint8_t* data, size_t data_size) {
  if (data_size < 4u) {
    return false;
  }

  if ((data[0] == 0u) && (data[1] == 0u) && (data[2] == 1u)) {
    return true;
  }

  return (data[0] == 0u) && (data[1] == 0u) && (data[2] == 0u) && (data[3] == 1u);
}

void append_start_code(std::vector<uint8_t>& output) {
  output.push_back(0u);
  output.push_back(0u);
  output.push_back(0u);
  output.push_back(1u);
}

uint8_t clamp_u8(int32_t value) {
  if (value < 0) {
    return 0u;
  }

  if (value > 255) {
    return 255u;
  }

  return static_cast<uint8_t>(value);
}

void convert_bgra_to_nv12(const uint8_t* bgra_data, uint32_t width, uint32_t height, std::vector<uint8_t>& output) {
  const size_t y_plane_size = static_cast<size_t>(width) * static_cast<size_t>(height);
  const size_t uv_plane_size = y_plane_size / 2u;
  output.resize(y_plane_size + uv_plane_size);

  uint8_t* y_plane = output.data();
  uint8_t* uv_plane = output.data() + y_plane_size;

  for (uint32_t y = 0u; y < height; y += 2u) {
    for (uint32_t x = 0u; x < width; x += 2u) {
      int32_t u_sum = 0;
      int32_t v_sum = 0;

      for (uint32_t local_y = 0u; local_y < 2u; ++local_y) {
        for (uint32_t local_x = 0u; local_x < 2u; ++local_x) {
          const uint32_t pixel_x = x + local_x;
          const uint32_t pixel_y = y + local_y;
          const size_t input_index = (static_cast<size_t>(pixel_y) * static_cast<size_t>(width) + static_cast<size_t>(pixel_x)) * 4u;
          const int32_t b = bgra_data[input_index + 0u];
          const int32_t g = bgra_data[input_index + 1u];
          const int32_t r = bgra_data[input_index + 2u];

          const int32_t y_value = ((66 * r) + (129 * g) + (25 * b) + 128) >> 8;
          const int32_t u_value = (((-38 * r) - (74 * g) + (112 * b) + 128) >> 8) + 128;
          const int32_t v_value = (((112 * r) - (94 * g) - (18 * b) + 128) >> 8) + 128;

          y_plane[static_cast<size_t>(pixel_y) * static_cast<size_t>(width) + static_cast<size_t>(pixel_x)] = clamp_u8(y_value + 16);
          u_sum += u_value;
          v_sum += v_value;
        }
      }

      const size_t uv_index = (static_cast<size_t>(y) / 2u) * static_cast<size_t>(width) + static_cast<size_t>(x);
      uv_plane[uv_index + 0u] = clamp_u8(u_sum / 4);
      uv_plane[uv_index + 1u] = clamp_u8(v_sum / 4);
    }
  }
}

bool convert_length_prefixed_to_annex_b(const uint8_t* data, size_t data_size, std::vector<uint8_t>& output) {
  output.clear();
  size_t offset = 0u;
  while ((offset + 4u) <= data_size) {
    const uint32_t nal_size = (static_cast<uint32_t>(data[offset + 0u]) << 24u) | (static_cast<uint32_t>(data[offset + 1u]) << 16u) |
                              (static_cast<uint32_t>(data[offset + 2u]) << 8u) | static_cast<uint32_t>(data[offset + 3u]);
    offset += 4u;
    if ((nal_size == 0u) || ((offset + nal_size) > data_size)) {
      output.clear();
      return false;
    }

    append_start_code(output);
    output.insert(output.end(), data + offset, data + offset + nal_size);
    offset += nal_size;
  }

  return output.empty() == false;
}

bool normalize_h264_stream(const uint8_t* data, size_t data_size, std::vector<uint8_t>& output) {
  output.clear();
  if ((data == nullptr) || (data_size == 0u)) {
    return false;
  }

  if (starts_with_annex_b(data, data_size)) {
    output.assign(data, data + data_size);
    return true;
  }

  return convert_length_prefixed_to_annex_b(data, data_size, output);
}

size_t find_start_code(const std::vector<uint8_t>& data, size_t begin, size_t& start_code_size) {
  for (size_t i = begin; (i + 3u) < data.size(); ++i) {
    if ((data[i + 0u] == 0u) && (data[i + 1u] == 0u)) {
      if (data[i + 2u] == 1u) {
        start_code_size = 3u;
        return i;
      }
      if (((i + 4u) < data.size()) && (data[i + 2u] == 0u) && (data[i + 3u] == 1u)) {
        start_code_size = 4u;
        return i;
      }
    }
  }

  start_code_size = 0u;
  return data.size();
}

void extract_h264_parameter_sets(const std::vector<uint8_t>& data, std::vector<uint8_t>& parameter_sets, bool& has_idr) {
  parameter_sets.clear();
  has_idr = false;

  size_t position = 0u;
  while (position < data.size()) {
    size_t start_code_size = 0u;
    const size_t nal_begin = find_start_code(data, position, start_code_size);
    if (nal_begin == data.size()) {
      break;
    }

    const size_t payload_begin = nal_begin + start_code_size;
    size_t next_start_code_size = 0u;
    const size_t next_nal_begin = find_start_code(data, payload_begin, next_start_code_size);
    const size_t payload_end = (next_nal_begin == data.size()) ? data.size() : next_nal_begin;
    if (payload_begin < payload_end) {
      const uint8_t nal_type = data[payload_begin] & 0x1fu;
      if ((nal_type == 7u) || (nal_type == 8u)) {
        append_start_code(parameter_sets);
        parameter_sets.insert(parameter_sets.end(), data.begin() + static_cast<int64_t>(payload_begin), data.begin() + static_cast<int64_t>(payload_end));
      } else if (nal_type == 5u) {
        has_idr = true;
      }
    }

    position = payload_end;
  }
}

bool set_codec_bool(ICodecAPI* codec_api, const GUID& property_id, bool value) {
  if (codec_api == nullptr) {
    return false;
  }

  VARIANT property_value = {};
  VariantInit(&property_value);
  property_value.vt = VT_BOOL;
  property_value.boolVal = value ? VARIANT_TRUE : VARIANT_FALSE;
  const HRESULT set_result = codec_api->SetValue(&property_id, &property_value);
  VariantClear(&property_value);
  return SUCCEEDED(set_result);
}

bool set_codec_u32(ICodecAPI* codec_api, const GUID& property_id, uint32_t value) {
  if (codec_api == nullptr) {
    return false;
  }

  VARIANT property_value = {};
  VariantInit(&property_value);
  property_value.vt = VT_UI4;
  property_value.ulVal = value;
  const HRESULT set_result = codec_api->SetValue(&property_id, &property_value);
  VariantClear(&property_value);
  return SUCCEEDED(set_result);
}

uint32_t scaled_bitrate(uint32_t bitrate) {
  const uint64_t scaled = (static_cast<uint64_t>(bitrate) * static_cast<uint64_t>(k_encoder_peak_bitrate_num)) /
                          static_cast<uint64_t>(k_encoder_peak_bitrate_den);
  const uint64_t max_u32 = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
  if (scaled > max_u32) {
    return std::numeric_limits<uint32_t>::max();
  }

  return static_cast<uint32_t>(scaled);
}

bool allocate_output_sample(IMFTransform* encoder, ComPtr<IMFSample>& output_sample) {
  MFT_OUTPUT_STREAM_INFO stream_info = {};
  const HRESULT stream_info_result = encoder->GetOutputStreamInfo(0u, &stream_info);
  if (FAILED(stream_info_result)) {
    return false;
  }

  if ((stream_info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES) != 0u) {
    output_sample.Reset();
    return true;
  }

  ComPtr<IMFSample> sample = {};
  HRESULT result = MFCreateSample(&sample);
  if (FAILED(result)) {
    return false;
  }

  ComPtr<IMFMediaBuffer> buffer = {};
  result = MFCreateMemoryBuffer(stream_info.cbSize, &buffer);
  if (FAILED(result)) {
    return false;
  }

  result = sample->AddBuffer(buffer.Get());
  if (FAILED(result)) {
    return false;
  }

  output_sample = sample;
  return true;
}
#endif

}  // namespace

bool MediaFoundationH264Encoder::init(const MediaFoundationH264EncoderConfig& config) {
  shutdown();

  _config = config;

#if ETX_PLATFORM_WINDOWS
  HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  if ((result == S_OK) || (result == S_FALSE)) {
    _com_initialized = true;
  } else if (result == RPC_E_CHANGED_MODE) {
    _com_initialized = false;
  } else {
    log::error("MF H264 encoder: CoInitializeEx failed (0x%08x)", static_cast<uint32_t>(result));
    return false;
  }

  result = MFStartup(MF_VERSION, MFSTARTUP_LITE);
  if (FAILED(result)) {
    log::error("MF H264 encoder: MFStartup failed (0x%08x)", static_cast<uint32_t>(result));
    shutdown();
    return false;
  }
  _mf_started = true;

  MFT_REGISTER_TYPE_INFO input_info = {};
  input_info.guidMajorType = MFMediaType_Video;
  input_info.guidSubtype = MFVideoFormat_NV12;

  MFT_REGISTER_TYPE_INFO output_info = {};
  output_info.guidMajorType = MFMediaType_Video;
  output_info.guidSubtype = MFVideoFormat_H264;

  const DWORD enumeration_flags[2] = {
    MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_SORTANDFILTER,
    MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_LOCALMFT,
  };

  bool encoder_configured = false;
  HRESULT last_enum_result = S_OK;
  for (uint32_t flag_index = 0u; (flag_index < 2u) && (encoder_configured == false); ++flag_index) {
    IMFActivate** activations = nullptr;
    UINT32 activation_count = 0u;
    last_enum_result = MFTEnumEx(MFT_CATEGORY_VIDEO_ENCODER, enumeration_flags[flag_index], &input_info, &output_info, &activations, &activation_count);
    if ((FAILED(last_enum_result)) || (activation_count == 0u)) {
      if (activations != nullptr) {
        CoTaskMemFree(activations);
      }
      continue;
    }

    for (UINT32 activation_index = 0u; (activation_index < activation_count) && (encoder_configured == false); ++activation_index) {
      ComPtr<IMFTransform> encoder = {};
      result = activations[activation_index]->ActivateObject(IID_PPV_ARGS(&encoder));
      if (FAILED(result) || (encoder.Get() == nullptr)) {
        continue;
      }

      _encoder = encoder.Detach();
      if (configure_encoder()) {
        encoder_configured = true;
      } else {
        as_interface<IMFTransform>(_encoder)->Release();
        _encoder = nullptr;
      }
    }

    for (UINT32 i = 0u; i < activation_count; ++i) {
      if (activations[i] != nullptr) {
        activations[i]->Release();
      }
    }
    CoTaskMemFree(activations);
  }

  if (encoder_configured == false) {
    log::error("MF H264 encoder: no compatible H264 encoder MFT found (0x%08x)", static_cast<uint32_t>(last_enum_result));
    shutdown();
    return false;
  }

  _initialized = true;
  return true;
#else
  (void) config;
  log::error("MF H264 encoder: this build only supports Windows");
  return false;
#endif
}

void MediaFoundationH264Encoder::shutdown() {
#if ETX_PLATFORM_WINDOWS
  if (_encoder != nullptr) {
    as_interface<IMFTransform>(_encoder)->Release();
    _encoder = nullptr;
  }
  if (_output_type != nullptr) {
    as_interface<IMFMediaType>(_output_type)->Release();
    _output_type = nullptr;
  }
  if (_mf_started) {
    MFShutdown();
    _mf_started = false;
  }
  if (_com_initialized) {
    CoUninitialize();
    _com_initialized = false;
  }
#endif

  _initialized = false;
  _pending_keyframe = false;
  _input_nv12.clear();
  _parameter_sets.clear();
  _config = {};
}

void MediaFoundationH264Encoder::request_keyframe() {
  _pending_keyframe = true;
}

bool MediaFoundationH264Encoder::configure_encoder() {
#if ETX_PLATFORM_WINDOWS
  IMFTransform* encoder = as_interface<IMFTransform>(_encoder);
  if (encoder == nullptr) {
    return false;
  }

  ComPtr<ICodecAPI> codec_api = {};
  encoder->QueryInterface(IID_PPV_ARGS(&codec_api));
  if (codec_api.Get() != nullptr) {
    set_codec_bool(codec_api.Get(), CODECAPI_AVLowLatencyMode, true);
    set_codec_bool(codec_api.Get(), CODECAPI_AVEncCommonRealTime, true);
    set_codec_u32(codec_api.Get(), CODECAPI_AVEncCommonRateControlMode, eAVEncCommonRateControlMode_LowDelayVBR);
    set_codec_u32(codec_api.Get(), CODECAPI_AVEncCommonMeanBitRate, _config.target_bitrate);
    set_codec_u32(codec_api.Get(), CODECAPI_AVEncCommonMaxBitRate, scaled_bitrate(_config.target_bitrate));
    set_codec_u32(codec_api.Get(), CODECAPI_AVEncMPVGOPSize, std::max(1u, _config.fps));
  }

  ComPtr<IMFMediaType> output_type = {};
  HRESULT result = MFCreateMediaType(&output_type);
  if (FAILED(result)) {
    log::error("MF H264 encoder: MFCreateMediaType output failed (0x%08x)", static_cast<uint32_t>(result));
    return false;
  }

  result = output_type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
  result = FAILED(result) ? result : output_type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
  result = FAILED(result) ? result : output_type->SetUINT32(MF_MT_AVG_BITRATE, _config.target_bitrate);
  result = FAILED(result) ? result : output_type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
  result = FAILED(result) ? result : output_type->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE);
  result = FAILED(result) ? result : output_type->SetUINT32(MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_ConstrainedBase);
  result = FAILED(result) ? result : MFSetAttributeSize(output_type.Get(), MF_MT_FRAME_SIZE, _config.width, _config.height);
  result = FAILED(result) ? result : MFSetAttributeRatio(output_type.Get(), MF_MT_FRAME_RATE, _config.fps, 1u);
  result = FAILED(result) ? result : MFSetAttributeRatio(output_type.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1u, 1u);
  if (FAILED(result)) {
    log::error("MF H264 encoder: configuring output type failed (0x%08x)", static_cast<uint32_t>(result));
    return false;
  }

  result = encoder->SetOutputType(0u, output_type.Get(), 0u);
  if (FAILED(result)) {
    log::error("MF H264 encoder: SetOutputType failed (0x%08x)", static_cast<uint32_t>(result));
    return false;
  }

  ComPtr<IMFMediaType> input_type = {};
  result = MFCreateMediaType(&input_type);
  if (FAILED(result)) {
    log::error("MF H264 encoder: MFCreateMediaType input failed (0x%08x)", static_cast<uint32_t>(result));
    return false;
  }

  result = input_type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
  result = FAILED(result) ? result : input_type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
  result = FAILED(result) ? result : input_type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
  result = FAILED(result) ? result : MFSetAttributeSize(input_type.Get(), MF_MT_FRAME_SIZE, _config.width, _config.height);
  result = FAILED(result) ? result : MFSetAttributeRatio(input_type.Get(), MF_MT_FRAME_RATE, _config.fps, 1u);
  result = FAILED(result) ? result : MFSetAttributeRatio(input_type.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1u, 1u);
  result = FAILED(result) ? result : input_type->SetUINT32(MF_MT_DEFAULT_STRIDE, _config.width);
  result = FAILED(result) ? result : input_type->SetUINT32(MF_MT_SAMPLE_SIZE, (_config.width * _config.height * 3u) / 2u);
  if (FAILED(result)) {
    log::error("MF H264 encoder: configuring input type failed (0x%08x)", static_cast<uint32_t>(result));
    return false;
  }

  result = encoder->SetInputType(0u, input_type.Get(), 0u);
  if (FAILED(result)) {
    log::error("MF H264 encoder: SetInputType failed (0x%08x)", static_cast<uint32_t>(result));
    return false;
  }

  encoder->ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0u);
  encoder->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0u);
  encoder->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0u);

  _output_type = output_type.Detach();
  return true;
#else
  return false;
#endif
}

bool MediaFoundationH264Encoder::create_input_sample(const uint8_t* bgra_data, uint64_t timestamp_us, void** out_sample) {
  if ((bgra_data == nullptr) || (out_sample == nullptr)) {
    return false;
  }

#if ETX_PLATFORM_WINDOWS
  *out_sample = nullptr;
  convert_bgra_to_nv12(bgra_data, _config.width, _config.height, _input_nv12);
  const DWORD data_size = static_cast<DWORD>(_input_nv12.size());
  ComPtr<IMFSample> sample = {};
  HRESULT result = MFCreateSample(&sample);
  if (FAILED(result)) {
    return false;
  }

  ComPtr<IMFMediaBuffer> buffer = {};
  result = MFCreateMemoryBuffer(static_cast<DWORD>(data_size), &buffer);
  if (FAILED(result)) {
    return false;
  }

  BYTE* destination = nullptr;
  DWORD max_length = 0u;
  DWORD current_length = 0u;
  result = buffer->Lock(&destination, &max_length, &current_length);
  if (FAILED(result) || (destination == nullptr)) {
    return false;
  }

  std::memcpy(destination, _input_nv12.data(), data_size);
  buffer->Unlock();
  buffer->SetCurrentLength(data_size);

  result = sample->AddBuffer(buffer.Get());
  if (FAILED(result)) {
    return false;
  }

  const LONGLONG sample_time = static_cast<LONGLONG>(timestamp_us * 10u);
  const LONGLONG sample_duration = static_cast<LONGLONG>(k_hns_per_second / std::max(1u, _config.fps));
  sample->SetSampleTime(sample_time);
  sample->SetSampleDuration(sample_duration);

  if (_pending_keyframe) {
    ComPtr<ICodecAPI> codec_api = {};
    as_interface<IMFTransform>(_encoder)->QueryInterface(IID_PPV_ARGS(&codec_api));
    if (codec_api.Get() != nullptr) {
      set_codec_bool(codec_api.Get(), CODECAPI_AVEncVideoForceKeyFrame, true);
    }
    _pending_keyframe = false;
  }

  *out_sample = sample.Detach();
  return true;
#else
  (void) timestamp_us;
  return false;
#endif
}

bool MediaFoundationH264Encoder::drain_output(uint64_t timestamp_us, std::vector<H264EncodedFrame>& output_frames) {
#if ETX_PLATFORM_WINDOWS
  IMFTransform* encoder = as_interface<IMFTransform>(_encoder);
  if (encoder == nullptr) {
    return false;
  }

  while (true) {
    ComPtr<IMFSample> output_sample = {};
    if (allocate_output_sample(encoder, output_sample) == false) {
      return false;
    }

    MFT_OUTPUT_DATA_BUFFER output_data = {};
    output_data.dwStreamID = 0u;
    output_data.pSample = output_sample.Get();

    DWORD status = 0u;
    const HRESULT result = encoder->ProcessOutput(0u, 1u, &output_data, &status);
    if (result == MF_E_TRANSFORM_NEED_MORE_INPUT) {
      break;
    }
    if (result == MF_E_TRANSFORM_STREAM_CHANGE) {
      continue;
    }
    if (FAILED(result)) {
      log::error("MF H264 encoder: ProcessOutput failed (0x%08x)", static_cast<uint32_t>(result));
      return false;
    }

    ComPtr<IMFSample> sample = output_sample;
    if ((sample.Get() == nullptr) && (output_data.pSample != nullptr)) {
      sample.Attach(output_data.pSample);
    }
    if (sample.Get() == nullptr) {
      continue;
    }

    ComPtr<IMFMediaBuffer> contiguous_buffer = {};
    const HRESULT buffer_result = sample->ConvertToContiguousBuffer(&contiguous_buffer);
    if (FAILED(buffer_result) || (contiguous_buffer.Get() == nullptr)) {
      continue;
    }

    BYTE* encoded_data = nullptr;
    DWORD max_length = 0u;
    DWORD current_length = 0u;
    HRESULT lock_result = contiguous_buffer->Lock(&encoded_data, &max_length, &current_length);
    if (FAILED(lock_result) || (encoded_data == nullptr) || (current_length == 0u)) {
      continue;
    }

    std::vector<uint8_t> annex_b_data = {};
    const bool normalized = normalize_h264_stream(encoded_data, current_length, annex_b_data);
    contiguous_buffer->Unlock();
    if (normalized == false) {
      continue;
    }

    bool has_idr = false;
    std::vector<uint8_t> parameter_sets = {};
    extract_h264_parameter_sets(annex_b_data, parameter_sets, has_idr);
    if (parameter_sets.empty() == false) {
      _parameter_sets = parameter_sets;
    }
    if (has_idr && parameter_sets.empty() && (_parameter_sets.empty() == false)) {
      std::vector<uint8_t> prefixed = _parameter_sets;
      prefixed.insert(prefixed.end(), annex_b_data.begin(), annex_b_data.end());
      annex_b_data = std::move(prefixed);
    }

    UINT32 clean_point = 0u;
    const HRESULT clean_point_result = sample->GetUINT32(MFSampleExtension_CleanPoint, &clean_point);

    H264EncodedFrame frame = {};
    frame.data = std::move(annex_b_data);
    frame.timestamp_us = timestamp_us;
    frame.keyframe = (clean_point_result == S_OK) ? (clean_point != 0u) : has_idr;
    output_frames.push_back(std::move(frame));
  }

  return true;
#else
  (void) timestamp_us;
  (void) output_frames;
  return false;
#endif
}

bool MediaFoundationH264Encoder::encode_bgra_frame(const uint8_t* bgra_data, size_t data_size, uint64_t timestamp_us, std::vector<H264EncodedFrame>& output_frames) {
  output_frames.clear();
  if ((_initialized == false) || (bgra_data == nullptr)) {
    return false;
  }

  const size_t expected_bgra_size = static_cast<size_t>(_config.width) * static_cast<size_t>(_config.height) * 4u;
  if (data_size < expected_bgra_size) {
    return false;
  }

  void* input_sample = nullptr;
  if (create_input_sample(bgra_data, timestamp_us, &input_sample) == false) {
    return false;
  }

#if ETX_PLATFORM_WINDOWS
  ComPtr<IMFSample> sample = {};
  sample.Attach(as_interface<IMFSample>(input_sample));
  const HRESULT input_result = as_interface<IMFTransform>(_encoder)->ProcessInput(0u, sample.Get(), 0u);
  if (FAILED(input_result)) {
    log::error("MF H264 encoder: ProcessInput failed (0x%08x)", static_cast<uint32_t>(input_result));
    return false;
  }
#else
  (void) input_sample;
#endif

  return drain_output(timestamp_us, output_frames);
}

}  // namespace etx
