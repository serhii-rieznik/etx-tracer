#include <cstddef>
#include <type_traits>

#include <etx/render/interop/interop.hxx>
#include <etx/render/interop/gpu_abi_constants.hxx>

#define ETX_RENDER_BASE_INCLUDED 1
#include <etx/render/interop/ray.hxx>
#undef ETX_RENDER_BASE_INCLUDED

#include <etx/render/interop/atmosphere_scattering_shared.hxx>
#include <etx/render/interop/camera.hxx>
#include <etx/render/interop/geometry.hxx>
#include <etx/render/interop/gpu_rt_shared.hxx>
#include <etx/render/interop/gpu_scene_shared.hxx>
#include <etx/render/interop/image.hxx>
#include <etx/render/interop/imgui_shared.hxx>
#include <etx/render/interop/material.hxx>
#include <etx/render/interop/math_shared.hxx>
#include <etx/render/interop/medium.hxx>
#include <etx/render/interop/projection.hxx>
#include <etx/render/interop/render_options.hxx>
#include <etx/render/shared/emitter.hxx>
#include <etx/render/shared/spectrum.hxx>

static_assert((sizeof(::SpectralDistribution) == kSpectralDistributionStride), "SpectralDistribution ABI size mismatch");
static_assert((offsetof(::SpectralDistribution, integrated_value) == kSpectralDistributionIntegratedOffset), "SpectralDistribution.integrated_value ABI mismatch");
static_assert((offsetof(::SpectralDistribution, spectral_entry_count) == kSpectralDistributionEntryCountOffset), "SpectralDistribution.spectral_entry_count ABI mismatch");
static_assert((offsetof(::SpectralDistribution, spectral_entries) == kSpectralDistributionEntriesOffset), "SpectralDistribution.spectral_entries ABI mismatch");
static_assert((sizeof(::SpectralDistribution::Entry) == kSpectralDistributionEntryStride), "SpectralDistribution::Entry ABI size mismatch");
static_assert((sizeof(::DistributionEntry) == kDistributionEntryStride), "DistributionEntry ABI size mismatch");

static_assert((ETX_ENUM_U32_TO_UINT32(ProjectionType::EqualArea) == kProjectionEqualArea), "ProjectionType::EqualArea must match kProjectionEqualArea");
static_assert((ETX_ENUM_U32_TO_UINT32(ProjectionType::Equirectangular) == Projection::Equirectangular),
  "Projection::Equirectangular changed; update ProjectionType::Equirectangular to keep CPU/GPU projection ABI aligned");
static_assert((ETX_ENUM_U32_TO_UINT32(ProjectionType::EqualArea) == Projection::EqualArea),
  "Projection::EqualArea changed; update ProjectionType::EqualArea to keep CPU/GPU projection ABI aligned");

static_assert((sizeof(AtmosphereSkyGpuParameters) == 32u), "AtmosphereSkyGpuParameters must stay 32 bytes");
static_assert((sizeof(AtmosphereSkyGpuLight) == 32u), "AtmosphereSkyGpuLight must stay 32 bytes");
static_assert((sizeof(AtmosphereSkyPushConstants) == 88u), "AtmosphereSkyPushConstants must stay 88 bytes");
static_assert((sizeof(AtmosphereSunPushConstants) == 48u), "AtmosphereSunPushConstants must stay 48 bytes");

static_assert(std::is_standard_layout_v<Vertex>, "Vertex must stay standard layout for C++/HLSL interop");
static_assert(std::is_trivially_copyable_v<Vertex>, "Vertex must stay trivially copyable for C++/HLSL interop");
static_assert(sizeof(Vertex) == 56, "Vertex size changed; update shared ABI");
static_assert(offsetof(Vertex, pos) == 0, "Vertex::pos offset changed; update shared ABI");
static_assert(offsetof(Vertex, nrm) == 12, "Vertex::nrm offset changed; update shared ABI");
static_assert(offsetof(Vertex, tan) == 24, "Vertex::tan offset changed; update shared ABI");
static_assert(offsetof(Vertex, btn) == 36, "Vertex::btn offset changed; update shared ABI");
static_assert(offsetof(Vertex, tex) == 48, "Vertex::tex offset changed; update shared ABI");

static_assert(std::is_standard_layout_v<Triangle>, "Triangle must stay standard layout for C++/HLSL interop");
static_assert(std::is_trivially_copyable_v<Triangle>, "Triangle must stay trivially copyable for C++/HLSL interop");
static_assert(sizeof(Triangle) == 32, "Triangle size changed; update shared ABI");
static_assert(offsetof(Triangle, i) == 0, "Triangle::i offset changed; update shared ABI");
static_assert(offsetof(Triangle, material_index) == 12, "Triangle::material_index offset changed; update shared ABI");
static_assert(offsetof(Triangle, geo_n) == 16, "Triangle::geo_n offset changed; update shared ABI");
static_assert(offsetof(Triangle, emitter_index) == 28, "Triangle::emitter_index offset changed; update shared ABI");

static_assert(std::is_standard_layout_v<ImGuiPushConstants>, "ImGuiPushConstants must stay standard layout for C++/HLSL interop");
static_assert(alignof(ImGuiPushConstants) == 16, "ImGuiPushConstants alignment must match HLSL packing");
static_assert(sizeof(ImGuiPushConstants) == 32, "ImGuiPushConstants size changed; update shared ABI or padding");
static_assert(offsetof(ImGuiPushConstants, vertex_buffer_index) == 16, "ImGuiPushConstants::vertex_buffer_index offset changed");

static_assert(std::is_standard_layout_v<Ray>, "Ray must stay standard layout for C++/HLSL interop");
static_assert(alignof(Ray) == 16, "Ray alignment must match HLSL packing");
static_assert(sizeof(Ray) == 32, "Ray size changed; update shared ABI or padding");
static_assert(offsetof(Ray, d) == 16, "Ray::d offset changed");

static_assert(std::is_standard_layout_v<ViewParameters>, "ViewParameters must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<RenderParameters>, "RenderParameters must stay standard layout for C++/HLSL interop");
static_assert(sizeof(ViewParameters) == 16, "ViewParameters size changed; update shared ABI or padding");
static_assert(sizeof(RenderParameters) == 48, "RenderParameters size changed; update shared ABI or padding");
static_assert(offsetof(RenderParameters, dimensions) == 16, "RenderParameters::dimensions offset changed");
static_assert(offsetof(RenderParameters, sample_count) == 32, "RenderParameters::sample_count offset changed");
static_assert(sizeof(ShaderConstants) == 32, "ShaderConstants size changed; update shared ABI or padding");

static_assert(std::is_standard_layout_v<GPURTConstants>, "GPURTConstants must stay standard layout for C++/HLSL interop");
static_assert(alignof(GPURTConstants) == 16, "GPURTConstants alignment must match HLSL packing");
static_assert(sizeof(GPURTConstants) == 112, "GPURTConstants size changed; update shared ABI or padding");
static_assert(offsetof(GPURTConstants, as_index) == 4, "GPURTConstants::as_index offset changed");
static_assert(offsetof(GPURTConstants, blue_noise_buffer_index) == 20, "GPURTConstants::blue_noise_buffer_index offset changed");
static_assert(offsetof(GPURTConstants, render_window_origin_x) == 32, "GPURTConstants::render_window_origin_x offset changed");
static_assert(offsetof(GPURTConstants, render_window_origin_y) == 36, "GPURTConstants::render_window_origin_y offset changed");
static_assert(offsetof(GPURTConstants, render_window_width) == 40, "GPURTConstants::render_window_width offset changed");
static_assert(offsetof(GPURTConstants, render_window_height) == 44, "GPURTConstants::render_window_height offset changed");
static_assert(offsetof(GPURTConstants, scene) == 48, "GPURTConstants::scene offset changed");

static_assert(std::is_standard_layout_v<SpectralImage>, "SpectralImage must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<Material>, "Material must stay standard layout for C++/HLSL interop");
static_assert(alignof(Material) == 16, "Material alignment must match HLSL packing");
static_assert(sizeof(Material) == kMaterialStride, "Material size changed; update shared ABI");
static_assert(offsetof(Material, scattering) == kMaterialScatteringSpectrumIndexOffset, "Material::scattering offset changed");
static_assert((offsetof(Material, scattering) + offsetof(SpectralImage, image_index)) == kMaterialScatteringImageIndexOffset, "Material::scattering.image_index offset changed");
static_assert(offsetof(Material, cls) == kMaterialClassOffset, "Material::cls offset changed");
static_assert(offsetof(Material, int_medium) == kMaterialIntMediumOffset, "Material::int_medium offset changed");
static_assert(offsetof(Material, ext_medium) == kMaterialExtMediumOffset, "Material::ext_medium offset changed");
static_assert(offsetof(Material, opacity) == kMaterialOpacityOffset, "Material::opacity offset changed");
static_assert(offsetof(Material, emission_collimation) == kMaterialEmissionCollimationOffset, "Material::emission_collimation offset changed");
static_assert(offsetof(Material, energy_compensation_interface_index) == kMaterialEnergyCompensationInterfaceIndexOffset,
  "Material::energy_compensation_interface_index offset changed");

static_assert(std::is_standard_layout_v<Camera>, "Camera must stay standard layout for C++/HLSL interop");
static_assert(alignof(Camera) == 16, "Camera alignment must match HLSL packing");
static_assert(sizeof(Camera) == 176, "Camera size changed; update shared ABI or padding");
static_assert(offsetof(Camera, position) == kCameraPositionOffset, "Camera::position offset changed");
static_assert(offsetof(Camera, cls) == kCameraClassOffset, "Camera::cls offset changed");
static_assert(offsetof(Camera, direction) == kCameraDirectionOffset, "Camera::direction offset changed");
static_assert(offsetof(Camera, aspect) == kCameraAspectOffset, "Camera::aspect offset changed");
static_assert(offsetof(Camera, side) == kCameraSideOffset, "Camera::side offset changed");
static_assert(offsetof(Camera, tan_half_fov) == kCameraTanHalfFovOffset, "Camera::tan_half_fov offset changed");
static_assert(offsetof(Camera, up) == kCameraUpOffset, "Camera::up offset changed");
static_assert(offsetof(Camera, film_size) == 64, "Camera::film_size offset changed");
static_assert(offsetof(Camera, film_size) == kCameraFilmSizeOffset, "Camera::film_size offset changed");
static_assert(offsetof(Camera, lens_radius) == kCameraLensRadiusOffset, "Camera::lens_radius offset changed");
static_assert(offsetof(Camera, focal_distance) == kCameraFocalDistanceOffset, "Camera::focal_distance offset changed");
static_assert(offsetof(Camera, clip_near) == kCameraClipNearOffset, "Camera::clip_near offset changed");
static_assert(offsetof(Camera, clip_far) == kCameraClipFarOffset, "Camera::clip_far offset changed");
static_assert(offsetof(Camera, lens_image) == kCameraLensImageOffset, "Camera::lens_image offset changed");
static_assert(offsetof(Camera, medium_index) == kCameraMediumIndexOffset, "Camera::medium_index offset changed");
static_assert(offsetof(Camera, view_proj) == 96, "Camera::view_proj offset changed");
static_assert(offsetof(Camera, view_proj) == kCameraViewProjOffset, "Camera::view_proj offset changed");
static_assert(offsetof(Camera, area) == kCameraAreaOffset, "Camera::area offset changed");

static_assert(std::is_standard_layout_v<Image>, "Image must stay standard layout for C++/HLSL interop");
static_assert(alignof(Image) == 16, "Image alignment must match HLSL packing");
static_assert(sizeof(Image) == kImageDescStride, "Image size changed; update shared ABI");
static_assert(offsetof(Image, fsize) == kImageDescFSizeOffset, "Image::fsize offset changed");
static_assert(offsetof(Image, offset) == kImageDescOffsetOffset, "Image::offset offset changed");
static_assert(offsetof(Image, scale) == kImageDescScaleOffset, "Image::scale offset changed");
static_assert(offsetof(Image, normalization) == kImageDescNormalizationOffset, "Image::normalization offset changed");
static_assert(offsetof(Image, isize) == kImageDescISizeOffset, "Image::isize offset changed");
static_assert(offsetof(Image, options) == kImageDescOptionsOffset, "Image::options offset changed");
static_assert(offsetof(Image, format) == kImageDescFormatOffset, "Image::format offset changed");
static_assert(offsetof(Image, pixel_data_offset) == kImageDescPixelDataOffset, "Image::pixel_data_offset offset changed");
static_assert(offsetof(Image, x_distribution_entries_offset) == kImageDescXDistributionEntriesOffset, "Image::x_distribution_entries_offset offset changed");
static_assert(offsetof(Image, y_distribution_entries_offset) == kImageDescYDistributionEntriesOffset, "Image::y_distribution_entries_offset offset changed");
static_assert(offsetof(Image, x_entries_stride) == kImageDescXEntriesStrideOffset, "Image::x_entries_stride offset changed");
static_assert(offsetof(Image, x_distribution_count) == kImageDescXDistributionCountOffset, "Image::x_distribution_count offset changed");
static_assert(offsetof(Image, y_entries_count) == kImageDescYEntriesCountOffset, "Image::y_entries_count offset changed");
static_assert(offsetof(Image, y_distribution_total_weight) == kImageDescYDistributionTotalWeightOffset, "Image::y_distribution_total_weight offset changed");
static_assert(offsetof(Image, pixel_data_stride) == kImageDescPixelDataStrideOffset, "Image::pixel_data_stride offset changed");
static_assert(offsetof(Image, pixel_data_chunk_index) == kImageDescPixelDataChunkIndexOffset, "Image::pixel_data_chunk_index offset changed");
static_assert(offsetof(Image, x_distribution_chunk_index) == kImageDescXDistributionChunkIndexOffset, "Image::x_distribution_chunk_index offset changed");
static_assert(offsetof(Image, y_distribution_chunk_index) == kImageDescYDistributionChunkIndexOffset, "Image::y_distribution_chunk_index offset changed");

static_assert(std::is_standard_layout_v<MediumGrid>, "MediumGrid must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<Medium>, "Medium must stay standard layout for C++/HLSL interop");
static_assert(alignof(MediumGrid) == 16, "MediumGrid alignment must match HLSL packing");
static_assert(alignof(Medium) == 16, "Medium alignment must match HLSL packing");
static_assert(sizeof(Medium) == kMediumStride, "Medium size changed; update shared ABI");
static_assert(offsetof(MediumGrid, dimensions) == kMediumGridDimensionsOffset, "MediumGrid::dimensions offset changed");
static_assert(offsetof(MediumGrid, type) == kMediumGridTypeOffset, "MediumGrid::type offset changed");
static_assert(offsetof(MediumGrid, noise_type) == kMediumGridNoiseTypeOffset, "MediumGrid::noise_type offset changed");
static_assert(offsetof(MediumGrid, density_data_offset) == kMediumGridDensityDataOffsetOffset, "MediumGrid::density_data_offset offset changed");
static_assert(offsetof(MediumGrid, density_count) == kMediumGridDensityCountOffset, "MediumGrid::density_count offset changed");
static_assert(offsetof(MediumGrid, noise_seed) == kMediumGridNoiseSeedOffset, "MediumGrid::noise_seed offset changed");
static_assert(offsetof(MediumGrid, noise_offset) == kMediumGridNoiseOffsetOffset, "MediumGrid::noise_offset offset changed");
static_assert(offsetof(MediumGrid, noise_enable_border_fade) == kMediumGridNoiseEnableBorderFadeOffset, "MediumGrid::noise_enable_border_fade offset changed");
static_assert(offsetof(MediumGrid, noise_octaves) == kMediumGridNoiseOctavesOffset, "MediumGrid::noise_octaves offset changed");
static_assert(offsetof(MediumGrid, noise_scale) == kMediumGridNoiseScaleOffset, "MediumGrid::noise_scale offset changed");
static_assert(offsetof(MediumGrid, noise_lacunarity) == kMediumGridNoiseLacunarityOffset, "MediumGrid::noise_lacunarity offset changed");
static_assert(offsetof(MediumGrid, noise_persistence) == kMediumGridNoisePersistenceOffset, "MediumGrid::noise_persistence offset changed");
static_assert(offsetof(MediumGrid, noise_power) == kMediumGridNoisePowerOffset, "MediumGrid::noise_power offset changed");
static_assert(offsetof(MediumGrid, noise_sharpness) == kMediumGridNoiseSharpnessOffset, "MediumGrid::noise_sharpness offset changed");
static_assert(offsetof(MediumGrid, noise_border_fade_distance) == kMediumGridNoiseBorderFadeDistanceOffset, "MediumGrid::noise_border_fade_distance offset changed");
static_assert(offsetof(MediumGrid, density_data_chunk_index) == kMediumGridDensityDataChunkIndexOffset, "MediumGrid::density_data_chunk_index offset changed");
static_assert((offsetof(Medium, bounds) + offsetof(BoundingBox, p_min)) == kMediumBoundsMinOffset, "Medium::bounds.p_min offset changed");
static_assert((offsetof(Medium, bounds) + offsetof(BoundingBox, p_max)) == kMediumBoundsMaxOffset, "Medium::bounds.p_max offset changed");
static_assert(offsetof(Medium, absorption_index) == kMediumAbsorptionIndexOffset, "Medium::absorption_index offset changed");
static_assert(offsetof(Medium, scattering_index) == kMediumScatteringIndexOffset, "Medium::scattering_index offset changed");
static_assert(offsetof(Medium, cls) == kMediumClassOffset, "Medium::cls offset changed");

static_assert(std::is_standard_layout_v<GPUSceneGlobals>, "GPUSceneGlobals must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<GPUSceneOptions>, "GPUSceneOptions must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<GPUImageBlobHeader>, "GPUImageBlobHeader must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<GPUMediumBlobHeader>, "GPUMediumBlobHeader must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<GPUScene>, "GPUScene must stay standard layout for C++/HLSL interop");
static_assert(alignof(GPUSceneGlobals) == 16, "GPUSceneGlobals alignment must match HLSL packing");
static_assert(alignof(GPUSceneOptions) == 16, "GPUSceneOptions alignment must match HLSL packing");
static_assert(alignof(GPUImageBlobHeader) == 16, "GPUImageBlobHeader alignment must match HLSL packing");
static_assert(alignof(GPUMediumBlobHeader) == 16, "GPUMediumBlobHeader alignment must match HLSL packing");
static_assert(alignof(GPUScene) == 16, "GPUScene alignment must match HLSL packing");
static_assert(sizeof(GPUSceneGlobals) == 400, "GPUSceneGlobals size changed; update shared ABI");
static_assert(sizeof(GPUSceneOptions) == 48, "GPUSceneOptions size changed; update shared ABI");
static_assert(sizeof(GPUImageBlobHeader) == 16, "GPUImageBlobHeader size changed; update shared ABI");
static_assert(sizeof(GPUMediumBlobHeader) == 16, "GPUMediumBlobHeader size changed; update shared ABI");
static_assert(sizeof(GPUScene) == 64, "GPUScene size changed; update shared ABI");
static_assert(offsetof(GPUSceneGlobals, vertex_count) == kSceneGlobalsVertexCountOffset, "GPUSceneGlobals::vertex_count offset changed");
static_assert(offsetof(GPUSceneGlobals, triangle_count) == kSceneGlobalsTriangleCountOffset, "GPUSceneGlobals::triangle_count offset changed");
static_assert(offsetof(GPUSceneGlobals, emitter_profile_count) == kSceneGlobalsEmitterProfileCountOffset, "GPUSceneGlobals::emitter_profile_count offset changed");
static_assert(offsetof(GPUSceneGlobals, emitter_instance_count) == kSceneGlobalsEmitterInstanceCountOffset, "GPUSceneGlobals::emitter_instance_count offset changed");
static_assert(offsetof(GPUSceneGlobals, environment_emitter_count) == kSceneGlobalsEnvironmentEmitterCountOffset, "GPUSceneGlobals::environment_emitter_count offset changed");
static_assert(offsetof(GPUSceneGlobals, active_emitter_count) == kSceneGlobalsActiveEmitterCountOffset, "GPUSceneGlobals::active_emitter_count offset changed");
static_assert(offsetof(GPUSceneGlobals, bounding_sphere_radius) == kSceneGlobalsBoundingSphereRadiusOffset, "GPUSceneGlobals::bounding_sphere_radius offset changed");
static_assert(offsetof(GPUSceneGlobals, environment_emitters) == kSceneGlobalsEnvironmentEmittersOffset, "GPUSceneGlobals::environment_emitters offset changed");
static_assert(offsetof(GPUSceneGlobals, pixel_filter_image_index) == kSceneGlobalsPixelFilterImageIndexOffset, "GPUSceneGlobals::pixel_filter_image_index offset changed");
static_assert(offsetof(GPUSceneGlobals, pixel_filter_radius) == kSceneGlobalsPixelFilterRadiusOffset, "GPUSceneGlobals::pixel_filter_radius offset changed");
static_assert(offsetof(GPUSceneOptions, min_path_length) == kSceneOptionsMinPathLengthOffset, "GPUSceneOptions::min_path_length offset changed");
static_assert(offsetof(GPUSceneOptions, max_path_length) == kSceneOptionsMaxPathLengthOffset, "GPUSceneOptions::max_path_length offset changed");
static_assert(offsetof(GPUSceneOptions, samples) == kSceneOptionsSamplesOffset, "GPUSceneOptions::samples offset changed");
static_assert(offsetof(GPUSceneOptions, random_path_termination) == kSceneOptionsRandomPathTerminationOffset, "GPUSceneOptions::random_path_termination offset changed");
static_assert(offsetof(GPUSceneOptions, noise_threshold) == kSceneOptionsNoiseThresholdOffset, "GPUSceneOptions::noise_threshold offset changed");
static_assert(offsetof(GPUSceneOptions, radiance_clamp) == kSceneOptionsRadianceClampOffset, "GPUSceneOptions::radiance_clamp offset changed");
static_assert(offsetof(GPUSceneOptions, strategy_flags) == kSceneOptionsStrategyFlagsOffset, "GPUSceneOptions::strategy_flags offset changed");
static_assert(offsetof(GPUSceneOptions, light_sampling) == kSceneOptionsLightSamplingOffset, "GPUSceneOptions::light_sampling offset changed");
static_assert(offsetof(GPUSceneOptions, properties_flags) == kSceneOptionsPropertiesFlagsOffset, "GPUSceneOptions::properties_flags offset changed");
static_assert(offsetof(GPUSceneOptions, path_mode) == kSceneOptionsPathModeOffset, "GPUSceneOptions::path_mode offset changed");
static_assert(offsetof(GPUSceneOptions, random_seed) == kSceneOptionsRandomSeedOffset, "GPUSceneOptions::random_seed offset changed");
static_assert(offsetof(GPUImageBlobHeader, image_count) == kImageBlobHeaderImageCountOffset, "GPUImageBlobHeader::image_count offset changed");
static_assert(offsetof(GPUImageBlobHeader, images_offset) == kImageBlobHeaderImagesOffset, "GPUImageBlobHeader::images_offset offset changed");
static_assert(offsetof(GPUImageBlobHeader, data_chunk_count) == kImageBlobHeaderDataChunkCountOffset, "GPUImageBlobHeader::data_chunk_count offset changed");
static_assert(offsetof(GPUImageBlobHeader, data_chunk_indices_offset) == kImageBlobHeaderDataChunkIndicesOffset, "GPUImageBlobHeader::data_chunk_indices_offset offset changed");
static_assert(offsetof(GPUMediumBlobHeader, medium_count) == kMediumBlobHeaderMediumCountOffset, "GPUMediumBlobHeader::medium_count offset changed");
static_assert(offsetof(GPUMediumBlobHeader, mediums_offset) == kMediumBlobHeaderMediumsOffset, "GPUMediumBlobHeader::mediums_offset offset changed");
static_assert(offsetof(GPUMediumBlobHeader, data_chunk_count) == kMediumBlobHeaderDataChunkCountOffset, "GPUMediumBlobHeader::data_chunk_count offset changed");
static_assert(offsetof(GPUMediumBlobHeader, data_chunk_indices_offset) == kMediumBlobHeaderDataChunkIndicesOffset, "GPUMediumBlobHeader::data_chunk_indices_offset offset changed");

static_assert(std::is_standard_layout_v<etx::EmitterProfile>, "EmitterProfile must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<etx::Emitter>, "Emitter must stay standard layout for C++/HLSL interop");
static_assert(alignof(etx::EmitterProfile) == 16, "EmitterProfile alignment must match HLSL packing");
static_assert(sizeof(etx::Emitter) == kEmitterStride, "Emitter size changed; update shared ABI");
static_assert(offsetof(etx::Emitter, cls) == kEmitterClassOffset, "Emitter::cls offset changed");
static_assert(offsetof(etx::Emitter, profile) == kEmitterProfileOffset, "Emitter::profile offset changed");
static_assert(sizeof(etx::EmitterProfile) == kEmitterProfileStride, "EmitterProfile size changed; update shared ABI");
static_assert(offsetof(etx::EmitterProfile, emission) == kEmitterProfileEmissionSpectrumIndexOffset, "EmitterProfile::emission offset changed");
static_assert((offsetof(etx::EmitterProfile, emission) + offsetof(SpectralImage, image_index)) == kEmitterProfileEmissionImageIndexOffset,
  "EmitterProfile::emission.image_index offset changed");
static_assert(offsetof(etx::EmitterProfile, cls) == kEmitterProfileClassOffset, "EmitterProfile::cls offset changed");
static_assert((offsetof(etx::EmitterProfile, directional) + offsetof(etx::EmitterProfile::DirectionalData, direction)) == kEmitterProfileDirectionalDirectionOffset,
  "EmitterProfile::directional.direction offset changed");
static_assert(
  (offsetof(etx::EmitterProfile, directional) + offsetof(etx::EmitterProfile::DirectionalData, angular_size_cosine)) == kEmitterProfileDirectionalAngularSizeCosineOffset,
  "EmitterProfile::directional.angular_size_cosine offset changed");
static_assert(offsetof(etx::EmitterProfile, meta) == kEmitterProfileMetaOffset, "EmitterProfile::meta offset changed");
