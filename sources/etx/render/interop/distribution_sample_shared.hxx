#pragma once

#include "distribution.hxx"

#ifndef ETX_DISTRIBUTION_SHARED_CONTEXT_TYPE
# error "ETX_DISTRIBUTION_SHARED_CONTEXT_TYPE must be defined before including distribution_sample_shared.hxx"
#endif

#ifndef ETX_DISTRIBUTION_SHARED_CDF
# error "ETX_DISTRIBUTION_SHARED_CDF must be defined before including distribution_sample_shared.hxx"
#endif

#ifndef ETX_DISTRIBUTION_SHARED_PDF
# error "ETX_DISTRIBUTION_SHARED_PDF must be defined before including distribution_sample_shared.hxx"
#endif

ETX_SHARED_INLINE uint32_t distribution_shared_sample(ETX_INOUT(ETX_DISTRIBUTION_SHARED_CONTEXT_TYPE, context), uint32_t count, float rnd, ETX_OUT(float, pdf)) {
  if (count == 0u) {
    pdf = 0.0f;
    return kInvalidIndex;
  }

  DistributionSearchRange search = distribution_search_begin(count);
  while (distribution_search_active(search)) {
    uint32_t middle = distribution_search_middle(search);
    float middle_cdf = ETX_DISTRIBUTION_SHARED_CDF(context, middle);
    distribution_search_update(search, middle, middle_cdf, rnd);
  }

  pdf = ETX_DISTRIBUTION_SHARED_PDF(context, search.begin);
  return search.begin;
}
