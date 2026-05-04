#pragma once

#include "interop.hxx"

struct EmitterClass {
  enum : uint32_t {
    Area,
    Environment,
    Directional,

    Undefined = kInvalidIndex,
  };
};

struct EmitterProfileMeta {
  enum : uint32_t {
    None = 0u,
    Atmosphere = 1u << 0u,
  };
};

struct SceneProperty {
  enum : uint32_t {
    Committed,
    Spectral,
    MultipleImportanceSampling,
    BlueNoise,

    Count,
  };
};

struct SceneLimits {
  enum : uint32_t {
    MaxEnvironmentEmitters = 63u,
  };
};

ETX_STATIC_CONST uint32_t kTriangleStride = 32u;

ETX_STATIC_CONST uint32_t kMaterialStride = 272u;
ETX_STATIC_CONST uint32_t kMaterialReflectanceSpectrumIndexOffset = 0u;
ETX_STATIC_CONST uint32_t kMaterialReflectanceImageIndexOffset = 4u;
ETX_STATIC_CONST uint32_t kMaterialScatteringSpectrumIndexOffset = 16u;
ETX_STATIC_CONST uint32_t kMaterialScatteringImageIndexOffset = 20u;
ETX_STATIC_CONST uint32_t kMaterialEmissionSpectrumIndexOffset = 32u;
ETX_STATIC_CONST uint32_t kMaterialEmissionImageIndexOffset = 36u;
ETX_STATIC_CONST uint32_t kMaterialSubsurfaceSpectrumIndexOffset = 48u;
ETX_STATIC_CONST uint32_t kMaterialSubsurfaceImageIndexOffset = 52u;
ETX_STATIC_CONST uint32_t kMaterialRoughnessValueOffset = 64u;
ETX_STATIC_CONST uint32_t kMaterialRoughnessImageIndexOffset = 80u;
ETX_STATIC_CONST uint32_t kMaterialRoughnessChannelOffset = 84u;
ETX_STATIC_CONST uint32_t kMaterialMetalnessValueOffset = 96u;
ETX_STATIC_CONST uint32_t kMaterialMetalnessImageIndexOffset = 112u;
ETX_STATIC_CONST uint32_t kMaterialMetalnessChannelOffset = 116u;
ETX_STATIC_CONST uint32_t kMaterialTransmissionValueOffset = 128u;
ETX_STATIC_CONST uint32_t kMaterialTransmissionImageIndexOffset = 144u;
ETX_STATIC_CONST uint32_t kMaterialTransmissionChannelOffset = 148u;
ETX_STATIC_CONST uint32_t kMaterialThinfilmIorClassOffset = 160u;
ETX_STATIC_CONST uint32_t kMaterialThinfilmIorEtaIndexOffset = 164u;
ETX_STATIC_CONST uint32_t kMaterialThinfilmIorKIndexOffset = 168u;
ETX_STATIC_CONST uint32_t kMaterialThinfilmThicknessImageOffset = 176u;
ETX_STATIC_CONST uint32_t kMaterialThinfilmMinThicknessOffset = 180u;
ETX_STATIC_CONST uint32_t kMaterialThinfilmMaxThicknessOffset = 184u;
ETX_STATIC_CONST uint32_t kMaterialExtIorClassOffset = 192u;
ETX_STATIC_CONST uint32_t kMaterialExtIorEtaIndexOffset = 196u;
ETX_STATIC_CONST uint32_t kMaterialExtIorKIndexOffset = 200u;
ETX_STATIC_CONST uint32_t kMaterialIntIorClassOffset = 208u;
ETX_STATIC_CONST uint32_t kMaterialIntIorEtaIndexOffset = 212u;
ETX_STATIC_CONST uint32_t kMaterialIntIorKIndexOffset = 216u;
ETX_STATIC_CONST uint32_t kMaterialSubsurfaceClassOffset = 224u;
ETX_STATIC_CONST uint32_t kMaterialSubsurfacePathOffset = 228u;
ETX_STATIC_CONST uint32_t kMaterialClassOffset = 232u;
ETX_STATIC_CONST uint32_t kMaterialIntMediumOffset = 236u;
ETX_STATIC_CONST uint32_t kMaterialExtMediumOffset = 240u;
ETX_STATIC_CONST uint32_t kMaterialNormalImageIndexOffset = 244u;
ETX_STATIC_CONST uint32_t kMaterialTwoSidedOffset = 248u;
ETX_STATIC_CONST uint32_t kMaterialNormalScaleOffset = 252u;
ETX_STATIC_CONST uint32_t kMaterialOpacityOffset = 256u;
ETX_STATIC_CONST uint32_t kMaterialEmissionCollimationOffset = 260u;
ETX_STATIC_CONST uint32_t kMaterialEnergyCompensationInterfaceIndexOffset = 264u;
ETX_STATIC_CONST uint32_t kMaterialConductorEnergyCompensationInterfaceIndexOffset = 268u;

ETX_STATIC_CONST uint32_t kEmitterStride = 32u;
ETX_STATIC_CONST uint32_t kEmitterClassOffset = 0u;
ETX_STATIC_CONST uint32_t kEmitterProfileOffset = 4u;
ETX_STATIC_CONST uint32_t kEmitterTriangleIndexOffset = 8u;
ETX_STATIC_CONST uint32_t kEmitterSpectrumWeightOffset = 12u;
ETX_STATIC_CONST uint32_t kEmitterAdditionalWeightOffset = 16u;
ETX_STATIC_CONST uint32_t kEmitterTriangleAreaOffset = 20u;

ETX_STATIC_CONST uint32_t kEmitterProfileStride = 96u;
ETX_STATIC_CONST uint32_t kEmitterProfileEmissionSpectrumIndexOffset = 0u;
ETX_STATIC_CONST uint32_t kEmitterProfileEmissionImageIndexOffset = 4u;
ETX_STATIC_CONST uint32_t kEmitterProfileClassOffset = 16u;
ETX_STATIC_CONST uint32_t kEmitterProfileDirectionalDirectionOffset = 20u;
ETX_STATIC_CONST uint32_t kEmitterProfileDirectionalAngularSizeCosineOffset = 40u;
ETX_STATIC_CONST uint32_t kEmitterProfileMediumIndexOffset = 76u;
ETX_STATIC_CONST uint32_t kEmitterProfileMetaOffset = 84u;

ETX_STATIC_CONST uint32_t kSpectralDistributionStride = 3552u;
ETX_STATIC_CONST uint32_t kSpectralDistributionIntegratedOffset = 0u;
ETX_STATIC_CONST uint32_t kSpectralDistributionEntryCountOffset = 12u;
ETX_STATIC_CONST uint32_t kSpectralDistributionEntriesOffset = 16u;
ETX_STATIC_CONST uint32_t kSpectralDistributionEntryStride = 8u;

ETX_STATIC_CONST uint32_t kImageBlobHeaderImageCountOffset = 0u;
ETX_STATIC_CONST uint32_t kImageBlobHeaderImagesOffset = 4u;
ETX_STATIC_CONST uint32_t kImageBlobHeaderDataChunkCountOffset = 8u;
ETX_STATIC_CONST uint32_t kImageBlobHeaderDataChunkIndicesOffset = 12u;

ETX_STATIC_CONST uint32_t kImageDescStride = 96u;
ETX_STATIC_CONST uint32_t kImageDescFSizeOffset = 0u;
ETX_STATIC_CONST uint32_t kImageDescOffsetOffset = 8u;
ETX_STATIC_CONST uint32_t kImageDescScaleOffset = 16u;
ETX_STATIC_CONST uint32_t kImageDescNormalizationOffset = 24u;
ETX_STATIC_CONST uint32_t kImageDescISizeOffset = 28u;
ETX_STATIC_CONST uint32_t kImageDescOptionsOffset = 36u;
ETX_STATIC_CONST uint32_t kImageDescFormatOffset = 40u;
ETX_STATIC_CONST uint32_t kImageDescPixelDataOffset = 48u;
ETX_STATIC_CONST uint32_t kImageDescXDistributionEntriesOffset = 52u;
ETX_STATIC_CONST uint32_t kImageDescYDistributionEntriesOffset = 56u;
ETX_STATIC_CONST uint32_t kImageDescXEntriesStrideOffset = 60u;
ETX_STATIC_CONST uint32_t kImageDescXDistributionCountOffset = 64u;
ETX_STATIC_CONST uint32_t kImageDescYEntriesCountOffset = 68u;
ETX_STATIC_CONST uint32_t kImageDescYDistributionTotalWeightOffset = 72u;
ETX_STATIC_CONST uint32_t kImageDescPixelDataStrideOffset = 76u;
ETX_STATIC_CONST uint32_t kImageDescPixelDataChunkIndexOffset = 80u;
ETX_STATIC_CONST uint32_t kImageDescXDistributionChunkIndexOffset = 84u;
ETX_STATIC_CONST uint32_t kImageDescYDistributionChunkIndexOffset = 88u;

ETX_STATIC_CONST uint32_t kDistributionEntryStride = 16u;

ETX_STATIC_CONST uint32_t kCameraPositionOffset = 0u;
ETX_STATIC_CONST uint32_t kCameraClassOffset = 12u;
ETX_STATIC_CONST uint32_t kCameraDirectionOffset = 16u;
ETX_STATIC_CONST uint32_t kCameraAspectOffset = 28u;
ETX_STATIC_CONST uint32_t kCameraSideOffset = 32u;
ETX_STATIC_CONST uint32_t kCameraTanHalfFovOffset = 44u;
ETX_STATIC_CONST uint32_t kCameraUpOffset = 48u;
ETX_STATIC_CONST uint32_t kCameraFilmSizeOffset = 64u;
ETX_STATIC_CONST uint32_t kCameraLensRadiusOffset = 72u;
ETX_STATIC_CONST uint32_t kCameraFocalDistanceOffset = 76u;
ETX_STATIC_CONST uint32_t kCameraClipNearOffset = 80u;
ETX_STATIC_CONST uint32_t kCameraClipFarOffset = 84u;
ETX_STATIC_CONST uint32_t kCameraLensImageOffset = 88u;
ETX_STATIC_CONST uint32_t kCameraMediumIndexOffset = 92u;
ETX_STATIC_CONST uint32_t kCameraViewProjOffset = 96u;
ETX_STATIC_CONST uint32_t kCameraAreaOffset = 160u;

ETX_STATIC_CONST uint32_t kMediumBlobHeaderMediumCountOffset = 0u;
ETX_STATIC_CONST uint32_t kMediumBlobHeaderMediumsOffset = 4u;
ETX_STATIC_CONST uint32_t kMediumBlobHeaderDataChunkCountOffset = 8u;
ETX_STATIC_CONST uint32_t kMediumBlobHeaderDataChunkIndicesOffset = 12u;

ETX_STATIC_CONST uint32_t kMediumStride = 128u;
ETX_STATIC_CONST uint32_t kMediumGridDimensionsOffset = 0u;
ETX_STATIC_CONST uint32_t kMediumGridTypeOffset = 12u;
ETX_STATIC_CONST uint32_t kMediumGridNoiseTypeOffset = 16u;
ETX_STATIC_CONST uint32_t kMediumGridDensityDataOffsetOffset = 20u;
ETX_STATIC_CONST uint32_t kMediumGridDensityCountOffset = 24u;
ETX_STATIC_CONST uint32_t kMediumGridNoiseSeedOffset = 28u;
ETX_STATIC_CONST uint32_t kMediumGridNoiseOffsetOffset = 32u;
ETX_STATIC_CONST uint32_t kMediumGridNoiseEnableBorderFadeOffset = 44u;
ETX_STATIC_CONST uint32_t kMediumGridNoiseOctavesOffset = 48u;
ETX_STATIC_CONST uint32_t kMediumGridNoiseScaleOffset = 52u;
ETX_STATIC_CONST uint32_t kMediumGridNoiseLacunarityOffset = 56u;
ETX_STATIC_CONST uint32_t kMediumGridNoisePersistenceOffset = 60u;
ETX_STATIC_CONST uint32_t kMediumGridNoisePowerOffset = 64u;
ETX_STATIC_CONST uint32_t kMediumGridNoiseSharpnessOffset = 68u;
ETX_STATIC_CONST uint32_t kMediumGridNoiseBorderFadeDistanceOffset = 72u;
ETX_STATIC_CONST uint32_t kMediumGridDensityDataChunkIndexOffset = 76u;
ETX_STATIC_CONST uint32_t kMediumBoundsMinOffset = 80u;
ETX_STATIC_CONST uint32_t kMediumBoundsMaxOffset = 96u;
ETX_STATIC_CONST uint32_t kMediumAbsorptionIndexOffset = 112u;
ETX_STATIC_CONST uint32_t kMediumScatteringIndexOffset = 116u;
ETX_STATIC_CONST uint32_t kMediumPhaseFunctionGOffset = 120u;
ETX_STATIC_CONST uint32_t kMediumEnableExplicitConnectionsOffset = 124u;
ETX_STATIC_CONST uint32_t kMediumClassOffset = 126u;

ETX_STATIC_CONST uint32_t kSceneOptionsMinPathLengthOffset = 0u;
ETX_STATIC_CONST uint32_t kSceneOptionsMaxPathLengthOffset = 4u;
ETX_STATIC_CONST uint32_t kSceneOptionsSamplesOffset = 8u;
ETX_STATIC_CONST uint32_t kSceneOptionsRandomPathTerminationOffset = 12u;
ETX_STATIC_CONST uint32_t kSceneOptionsNoiseThresholdOffset = 16u;
ETX_STATIC_CONST uint32_t kSceneOptionsRadianceClampOffset = 20u;
ETX_STATIC_CONST uint32_t kSceneOptionsStrategyFlagsOffset = 24u;
ETX_STATIC_CONST uint32_t kSceneOptionsLightSamplingOffset = 28u;
ETX_STATIC_CONST uint32_t kSceneOptionsPropertiesFlagsOffset = 32u;
ETX_STATIC_CONST uint32_t kSceneOptionsPathModeOffset = 36u;
ETX_STATIC_CONST uint32_t kSceneOptionsRandomSeedOffset = 40u;

ETX_STATIC_CONST uint32_t kSceneGlobalsVertexCountOffset = 0u;
ETX_STATIC_CONST uint32_t kSceneGlobalsTriangleCountOffset = 4u;
ETX_STATIC_CONST uint32_t kSceneGlobalsEmitterProfileCountOffset = 12u;
ETX_STATIC_CONST uint32_t kSceneGlobalsEmitterInstanceCountOffset = 16u;
ETX_STATIC_CONST uint32_t kSceneGlobalsEnvironmentEmitterCountOffset = 20u;
ETX_STATIC_CONST uint32_t kSceneGlobalsActiveEmitterCountOffset = 24u;
ETX_STATIC_CONST uint32_t kSceneGlobalsBoundingSphereCenterOffset = 32u;
ETX_STATIC_CONST uint32_t kSceneGlobalsBoundingSphereRadiusOffset = 44u;
ETX_STATIC_CONST uint32_t kSceneGlobalsEnvironmentEmittersOffset = 80u;
ETX_STATIC_CONST uint32_t kSceneGlobalsPixelFilterImageIndexOffset = 384u;
ETX_STATIC_CONST uint32_t kSceneGlobalsPixelFilterRadiusOffset = 388u;
