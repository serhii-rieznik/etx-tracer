#pragma once

static const uint kUPBPRandomDomainFilmSample = 0x0d86c4fu;
static const uint kUPBPRandomDomainCameraPath = 0x17b42d1u;
static const uint kUPBPRandomDomainLightPath = 0x2eb7193u;
static const uint kUPBPRandomDomainCameraMediumTracking = 0x43ca905u;
static const uint kUPBPRandomDomainLightMediumTracking = 0x4d2c6dfu;
static const uint kUPBPRandomDomainConnectionTransmittance = 0x59d03b7u;
static const uint kUPBPRandomDomainEmitterConnection = 0x91f02a5u;
static const uint kUPBPRandomDomainScatteringEvaluation = 0xa70c3d9u;
static const uint kUPBPRandomDomainIntersectionTraversal = 0xb86e14fu;
static const uint kUPBPRandomDomainFilmConnection = 0xc49a721u;
static const uint kUPBPRandomDomainCameraRussianRoulette = 0xe2874adu;
static const uint kUPBPRandomDomainLightRussianRoulette = 0xf1bbcd9u;

uint upbp_deterministic_seed(uint global_path_index, uint physical_depth, uint segment_index, uint domain) {
  uint result = sampler_random_seed(load_scene_options_random_seed(), domain);
  result = sampler_random_seed(result, constants.sample_index);
  result = sampler_random_seed(result, 0u);
  result = sampler_random_seed(result, global_path_index);
  result = sampler_random_seed(result, 0u);
  result = sampler_random_seed(result, physical_depth);
  return sampler_random_seed(result, segment_index);
}
