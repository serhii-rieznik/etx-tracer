#pragma once

#include <etx/render/interop/interop.hxx>
#include <etx/render/interop/distribution.hxx>
#include <etx/render/shared/buffer_view.hxx>

namespace etx {

struct DistributionSharedCPUContext {
  ArrayView<DistributionEntry> values ETX_EMPTY_INIT;
};

ETX_SHARED_INLINE float distribution_shared_cpu_cdf(ETX_IN(DistributionSharedCPUContext, context), uint32_t index) {
  return context.values[index].cdf;
}

ETX_SHARED_INLINE float distribution_shared_cpu_pdf(ETX_IN(DistributionSharedCPUContext, context), uint32_t index) {
  return context.values[index].pdf;
}

#define ETX_DISTRIBUTION_SHARED_CONTEXT_TYPE DistributionSharedCPUContext
#define ETX_DISTRIBUTION_SHARED_CDF(context, index) distribution_shared_cpu_cdf(context, index)
#define ETX_DISTRIBUTION_SHARED_PDF(context, index) distribution_shared_cpu_pdf(context, index)
#include <etx/render/interop/distribution_sample_shared.hxx>
#undef ETX_DISTRIBUTION_SHARED_PDF
#undef ETX_DISTRIBUTION_SHARED_CDF
#undef ETX_DISTRIBUTION_SHARED_CONTEXT_TYPE

ETX_SHARED_INLINE uint32_t sample_distribution(ETX_IN(ArrayView<DistributionEntry>, values), float rnd, ETX_OUT(float, pdf)) {
  if ((values.count == 0) || (values.a == nullptr)) {
    pdf = 0.0f;
    return kInvalidIndex;
  }

  DistributionSharedCPUContext context = {};
  context.values = values;
  return distribution_shared_sample(context, static_cast<uint32_t>(values.count), rnd, pdf);
}

ETX_SHARED_INLINE uint32_t sample_distribution(ETX_IN(ArrayView<DistributionEntry>, values), float rnd) {
  float pdf = 0.0f;
  return sample_distribution(values, rnd, pdf);
}

struct ETX_ALIGNED Distribution {
  using Entry = ::DistributionEntry;
  ArrayView<Entry> values ETX_EMPTY_INIT;
  float total_weight ETX_EMPTY_INIT;
  BufferHandle values_buffer = {};
  BufferView values_storage = {};

  ETX_SHARED_INLINE uint32_t sample(float rnd, float& pdf) const {
    return sample_distribution(values, rnd, pdf);
  }

  ETX_SHARED_INLINE uint32_t sample(float rnd) const {
    return sample_distribution(values, rnd);
  }

  static ETX_SHARED_INLINE Distribution build(Distribution::Entry* entries, uint32_t count, BufferHandle values_buffer = {}, BufferView values_storage = {}) {
    float total_weight = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
      entries[i].cdf = total_weight;
      total_weight += entries[i].value;
    }

    if (total_weight == 0.0f) {
      for (uint32_t i = 0; i < count; ++i) {
        entries[i].value = 1.0f;
        entries[i].pdf = 1.0f / float(count);
        entries[i].cdf = float(i) / float(count);
      }
    } else {
      for (uint32_t i = 0; i < count; ++i) {
        entries[i].pdf = entries[i].value / total_weight;
        entries[i].cdf /= total_weight;
      }
    }
    entries[count] = {0.0f, 0.0f, 1.0f};

    Distribution result;
    result.values = {entries, count};
    result.total_weight = total_weight;
    result.values_buffer = values_buffer;
    result.values_storage = values_storage;
    return result;
  }
};

}  // namespace etx
