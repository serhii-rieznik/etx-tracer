#pragma once

namespace etx {
namespace {
static_assert(std::is_standard_layout_v<float2>, "float2 must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<float2>, "float2 must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(float2) == 8u, "float2 size changed; update GPU upload ABI");

static_assert(std::is_standard_layout_v<float3>, "float3 must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<float3>, "float3 must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(float3) == 12u, "float3 size changed; update GPU upload ABI");

static_assert(std::is_standard_layout_v<Triangle>, "Triangle must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Triangle>, "Triangle must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Triangle) == kTriangleStride, "Triangle size changed; update GPU shader decode stride");
static_assert(offsetof(Triangle, i) == 0u, "Triangle::i offset changed; update GPU shader decode");
static_assert(offsetof(Triangle, material_index) == 12u, "Triangle::material_index offset changed; update GPU shader decode");
static_assert(offsetof(Triangle, geo_n) == 16u, "Triangle::geo_n offset changed; update GPU shader decode");
static_assert(offsetof(Triangle, emitter_index) == 28u, "Triangle::emitter_index offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<Mesh>, "Mesh must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Mesh>, "Mesh must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Mesh) == 32u, "Mesh size changed; update GPU upload ABI");

static_assert(std::is_standard_layout_v<EmitterProfile>, "EmitterProfile must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<EmitterProfile>, "EmitterProfile must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(EmitterProfile) == kEmitterProfileStride, "EmitterProfile size changed; update GPU upload ABI");
static_assert(offsetof(EmitterProfile, emission) == kEmitterProfileEmissionSpectrumIndexOffset, "EmitterProfile::emission offset changed; update GPU shader decode");
static_assert(offsetof(EmitterProfile, cls) == kEmitterProfileClassOffset, "EmitterProfile::cls offset changed; update GPU shader decode");
static_assert((offsetof(EmitterProfile, directional) + offsetof(EmitterProfile::DirectionalData, direction)) == kEmitterProfileDirectionalDirectionOffset,
  "EmitterProfile::directional.direction offset changed; update GPU shader decode");
static_assert((offsetof(EmitterProfile, directional) + offsetof(EmitterProfile::DirectionalData, angular_size_cosine)) ==
    kEmitterProfileDirectionalAngularSizeCosineOffset,
  "EmitterProfile::directional.angular_size_cosine offset changed; update GPU shader decode");
static_assert(offsetof(EmitterProfile, meta) == kEmitterProfileMetaOffset, "EmitterProfile::meta offset changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(EmitterProfile::Class::Area) == EmitterClass::Area, "EmitterProfile::Class::Area changed; update GPU shader decode");
static_assert(EmitterProfile::Meta::Atmosphere == EmitterProfileMeta::Atmosphere, "EmitterProfile::Meta::Atmosphere changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<Emitter>, "Emitter must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Emitter>, "Emitter must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Emitter) == kEmitterStride, "Emitter size changed; update GPU upload ABI");
static_assert(offsetof(Emitter, cls) == kEmitterClassOffset, "Emitter::cls offset changed; update GPU shader decode");
static_assert(offsetof(Emitter, profile) == kEmitterProfileOffset, "Emitter::profile offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<Material>, "Material must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Material>, "Material must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Material) == kMaterialStride, "Material size changed; update GPU shader decode stride");
static_assert(offsetof(Material, scattering) == kMaterialScatteringSpectrumIndexOffset, "Material::scattering offset changed; update GPU shader decode");
static_assert((offsetof(Material, scattering) + offsetof(SpectralImage, image_index)) == kMaterialScatteringImageIndexOffset,
  "Material::scattering.image_index offset changed; update GPU shader decode");
static_assert(offsetof(Material, cls) == kMaterialClassOffset, "Material::cls offset changed; update GPU shader decode");
static_assert(offsetof(Material, int_medium) == kMaterialIntMediumOffset, "Material::int_medium offset changed; update GPU shader decode");
static_assert(offsetof(Material, ext_medium) == kMaterialExtMediumOffset, "Material::ext_medium offset changed; update GPU shader decode");
static_assert(offsetof(Material, opacity) == kMaterialOpacityOffset, "Material::opacity offset changed; update GPU shader decode");
static_assert(std::is_standard_layout_v<SpectralImage>, "SpectralImage must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<SpectralImage>, "SpectralImage must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(SpectralImage) == 16u, "SpectralImage size changed; update material ABI");
static_assert(offsetof(SpectralImage, spectrum_index) == 0u, "SpectralImage::spectrum_index offset changed; update material ABI");
static_assert(offsetof(SpectralImage, image_index) == 4u, "SpectralImage::image_index offset changed; update material ABI");

static_assert(std::is_standard_layout_v<SampledImage>, "SampledImage must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<SampledImage>, "SampledImage must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(SampledImage) == 32u, "SampledImage size changed; update material ABI");
static_assert(offsetof(SampledImage, value) == 0u, "SampledImage::value offset changed; update material ABI");
static_assert(offsetof(SampledImage, image_index) == 16u, "SampledImage::image_index offset changed; update material ABI");
static_assert(offsetof(SampledImage, channel) == 20u, "SampledImage::channel offset changed; update material ABI");

static_assert(std::is_standard_layout_v<Thinfilm>, "Thinfilm must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Thinfilm>, "Thinfilm must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(Thinfilm) == 32u, "Thinfilm size changed; update material ABI");

static_assert(std::is_standard_layout_v<RefractiveIndex>, "RefractiveIndex must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<RefractiveIndex>, "RefractiveIndex must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(RefractiveIndex) == 16u, "RefractiveIndex size changed; update material/spectrum ABI");

static_assert(std::is_standard_layout_v<SpectralDistribution>, "SpectralDistribution must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<SpectralDistribution>, "SpectralDistribution must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(SpectralDistribution) == kSpectralDistributionStride, "SpectralDistribution size changed; update GPU shader decode stride");
static_assert(offsetof(SpectralDistribution, integrated_value) == kSpectralDistributionIntegratedOffset,
  "SpectralDistribution::integrated_value offset changed; update GPU shader decode");

static_assert(offsetof(GPUSceneGlobals, vertex_count) == kSceneGlobalsVertexCountOffset, "GPUSceneGlobals::vertex_count offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, triangle_count) == kSceneGlobalsTriangleCountOffset, "GPUSceneGlobals::triangle_count offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, environment_emitter_count) == kSceneGlobalsEnvironmentEmitterCountOffset,
  "GPUSceneGlobals::environment_emitter_count offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, bounding_sphere_radius) == kSceneGlobalsBoundingSphereRadiusOffset,
  "GPUSceneGlobals::bounding_sphere_radius offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneGlobals, environment_emitters) == kSceneGlobalsEnvironmentEmittersOffset,
  "GPUSceneGlobals::environment_emitters offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<GPUSceneOptions>, "GPUSceneOptions must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<GPUSceneOptions>, "GPUSceneOptions must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(GPUSceneOptions) == 48u, "GPUSceneOptions size changed; update GPU shader decode stride");
static_assert(offsetof(GPUSceneOptions, min_path_length) == kSceneOptionsMinPathLengthOffset, "GPUSceneOptions::min_path_length offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, max_path_length) == kSceneOptionsMaxPathLengthOffset, "GPUSceneOptions::max_path_length offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, samples) == kSceneOptionsSamplesOffset, "GPUSceneOptions::samples offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, random_path_termination) == kSceneOptionsRandomPathTerminationOffset,
  "GPUSceneOptions::random_path_termination offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, noise_threshold) == kSceneOptionsNoiseThresholdOffset, "GPUSceneOptions::noise_threshold offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, radiance_clamp) == kSceneOptionsRadianceClampOffset, "GPUSceneOptions::radiance_clamp offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, strategy_flags) == kSceneOptionsStrategyFlagsOffset, "GPUSceneOptions::strategy_flags offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, light_sampling) == kSceneOptionsLightSamplingOffset, "GPUSceneOptions::light_sampling offset changed; update GPU shader decode");
static_assert(offsetof(GPUSceneOptions, properties_flags) == kSceneOptionsPropertiesFlagsOffset,
  "GPUSceneOptions::properties_flags offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<Camera>, "Camera must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<Camera>, "Camera must stay trivially copyable for GPU upload ABI");
static_assert(offsetof(Camera, position) == kCameraPositionOffset, "Camera::position offset changed; update GPU shader decode");
static_assert(offsetof(Camera, direction) == kCameraDirectionOffset, "Camera::direction offset changed; update GPU shader decode");
static_assert(offsetof(Camera, aspect) == kCameraAspectOffset, "Camera::aspect offset changed; update GPU shader decode");
static_assert(offsetof(Camera, side) == kCameraSideOffset, "Camera::side offset changed; update GPU shader decode");
static_assert(offsetof(Camera, tan_half_fov) == kCameraTanHalfFovOffset, "Camera::tan_half_fov offset changed; update GPU shader decode");
static_assert(offsetof(Camera, up) == kCameraUpOffset, "Camera::up offset changed; update GPU shader decode");
static_assert(offsetof(Camera, film_size) == kCameraFilmSizeOffset, "Camera::film_size offset changed; update GPU shader decode");
static_assert(offsetof(Camera, clip_near) == kCameraClipNearOffset, "Camera::clip_near offset changed; update GPU shader decode");
static_assert(offsetof(Camera, clip_far) == kCameraClipFarOffset, "Camera::clip_far offset changed; update GPU shader decode");
static_assert(offsetof(Camera, medium_index) == kCameraMediumIndexOffset, "Camera::medium_index offset changed; update GPU shader decode");

static_assert(std::is_standard_layout_v<GPUImageBlobHeader>, "GPUImageBlobHeader must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<GPUImageBlobHeader>, "GPUImageBlobHeader must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(GPUImageBlobHeader) == 16u, "GPUImageBlobHeader size changed; update GPU image blob ABI");
static_assert(offsetof(GPUImageBlobHeader, image_count) == kImageBlobHeaderImageCountOffset,
  "GPUImageBlobHeader::image_count offset changed; update GPU image blob ABI");
static_assert(offsetof(GPUImageBlobHeader, images_offset) == kImageBlobHeaderImagesOffset,
  "GPUImageBlobHeader::images_offset offset changed; update GPU image blob ABI");
static_assert(offsetof(GPUImageBlobHeader, data_chunk_count) == kImageBlobHeaderDataChunkCountOffset,
  "GPUImageBlobHeader::data_chunk_count offset changed; update GPU image blob ABI");
static_assert(offsetof(GPUImageBlobHeader, data_chunk_indices_offset) == kImageBlobHeaderDataChunkIndicesOffset,
  "GPUImageBlobHeader::data_chunk_indices_offset offset changed; update GPU image blob ABI");

static_assert(std::is_standard_layout_v<::Image>, "Interop Image must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<::Image>, "Interop Image must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(::Image) == kImageDescStride, "Interop Image size changed; update GPU image blob ABI");
static_assert(offsetof(::Image, fsize) == kImageDescFSizeOffset, "Image::fsize offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, offset) == kImageDescOffsetOffset, "Image::offset offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, scale) == kImageDescScaleOffset, "Image::scale offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, isize) == kImageDescISizeOffset, "Image::isize offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, options) == kImageDescOptionsOffset, "Image::options offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, format) == kImageDescFormatOffset, "Image::format offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, pixel_data_offset) == kImageDescPixelDataOffset, "Image::pixel_data_offset offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, pixel_data_stride) == kImageDescPixelDataStrideOffset, "Image::pixel_data_stride offset changed; update GPU image blob ABI");
static_assert(offsetof(::Image, pixel_data_chunk_index) == kImageDescPixelDataChunkIndexOffset,
  "Image::pixel_data_chunk_index offset changed; update GPU image blob ABI");

static_assert(std::is_standard_layout_v<GPUMediumBlobHeader>, "GPUMediumBlobHeader must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<GPUMediumBlobHeader>, "GPUMediumBlobHeader must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(GPUMediumBlobHeader) == 16u, "GPUMediumBlobHeader size changed; update GPU medium blob ABI");
static_assert(offsetof(GPUMediumBlobHeader, medium_count) == kMediumBlobHeaderMediumCountOffset,
  "GPUMediumBlobHeader::medium_count offset changed; update GPU medium blob ABI");
static_assert(offsetof(GPUMediumBlobHeader, mediums_offset) == kMediumBlobHeaderMediumsOffset,
  "GPUMediumBlobHeader::mediums_offset offset changed; update GPU medium blob ABI");
static_assert(offsetof(GPUMediumBlobHeader, data_chunk_count) == kMediumBlobHeaderDataChunkCountOffset,
  "GPUMediumBlobHeader::data_chunk_count offset changed; update GPU medium blob ABI");
static_assert(offsetof(GPUMediumBlobHeader, data_chunk_indices_offset) == kMediumBlobHeaderDataChunkIndicesOffset,
  "GPUMediumBlobHeader::data_chunk_indices_offset offset changed; update GPU medium blob ABI");
static_assert(SceneLimits::MaxEnvironmentEmitters == std::extent_v<decltype(GPUSceneGlobals::environment_emitters)>,
  "SceneLimits::MaxEnvironmentEmitters changed; update GPUSceneGlobals::environment_emitters size to keep CPU/GPU emitter lists aligned");

static_assert(std::is_standard_layout_v<::Medium>, "Interop Medium must stay standard layout for GPU upload ABI");
static_assert(std::is_trivially_copyable_v<::Medium>, "Interop Medium must stay trivially copyable for GPU upload ABI");
static_assert(sizeof(::Medium) == 128u, "Interop Medium size changed; update GPU medium blob ABI");
static_assert(sizeof(::Medium) == kMediumStride, "Medium stride changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, dimensions)) == kMediumGridDimensionsOffset,
  "Medium::grid.dimensions offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, type)) == kMediumGridTypeOffset,
  "Medium::grid.type offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_type)) == kMediumGridNoiseTypeOffset,
  "Medium::grid.noise_type offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, density_data_offset)) == kMediumGridDensityDataOffsetOffset,
  "Medium::grid.density_data_offset offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, density_count)) == kMediumGridDensityCountOffset,
  "Medium::grid.density_count offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_seed)) == kMediumGridNoiseSeedOffset,
  "Medium::grid.noise_seed offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_offset)) == kMediumGridNoiseOffsetOffset,
  "Medium::grid.noise_offset offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_enable_border_fade)) == kMediumGridNoiseEnableBorderFadeOffset,
  "Medium::grid.noise_enable_border_fade offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_octaves)) == kMediumGridNoiseOctavesOffset,
  "Medium::grid.noise_octaves offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_scale)) == kMediumGridNoiseScaleOffset,
  "Medium::grid.noise_scale offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_lacunarity)) == kMediumGridNoiseLacunarityOffset,
  "Medium::grid.noise_lacunarity offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_persistence)) == kMediumGridNoisePersistenceOffset,
  "Medium::grid.noise_persistence offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_power)) == kMediumGridNoisePowerOffset,
  "Medium::grid.noise_power offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_sharpness)) == kMediumGridNoiseSharpnessOffset,
  "Medium::grid.noise_sharpness offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, noise_border_fade_distance)) == kMediumGridNoiseBorderFadeDistanceOffset,
  "Medium::grid.noise_border_fade_distance offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, grid) + offsetof(::MediumGrid, density_data_chunk_index)) == kMediumGridDensityDataChunkIndexOffset,
  "Medium::grid.density_data_chunk_index offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, bounds) + offsetof(BoundingBox, p_min)) == kMediumBoundsMinOffset, "Medium::bounds.p_min offset changed; update GPU shader decode");
static_assert((offsetof(::Medium, bounds) + offsetof(BoundingBox, p_max)) == kMediumBoundsMaxOffset, "Medium::bounds.p_max offset changed; update GPU shader decode");
static_assert(offsetof(::Medium, absorption_index) == kMediumAbsorptionIndexOffset, "Medium::absorption_index offset changed; update GPU shader decode");
static_assert(offsetof(::Medium, scattering_index) == kMediumScatteringIndexOffset, "Medium::scattering_index offset changed; update GPU shader decode");
static_assert(offsetof(::Medium, cls) == kMediumClassOffset, "Medium::cls offset changed; update GPU shader decode");
static_assert(::Medium::Homogeneous == Medium::Homogeneous, "Medium::Homogeneous changed; update GPU shader decode");
static_assert(::Medium::Heterogeneous == Medium::Heterogeneous, "Medium::Heterogeneous changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(DensityGrid::Type::Texture3D) == MediumGridType::Texture3D, "DensityGrid::Type::Texture3D changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(DensityGrid::Type::NoiseFunction) == MediumGridType::NoiseFunction, "DensityGrid::Type::NoiseFunction changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Perlin) == MediumNoiseType::Perlin, "NoiseFunction::Perlin changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Worley) == MediumNoiseType::Worley, "NoiseFunction::Worley changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Billow) == MediumNoiseType::Billow, "NoiseFunction::Billow changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Voronoi) == MediumNoiseType::Voronoi, "NoiseFunction::Voronoi changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Lattice) == MediumNoiseType::Lattice, "NoiseFunction::Lattice changed; update GPU shader decode");
static_assert(static_cast<uint32_t>(NoiseFunction::Uniform) == MediumNoiseType::Uniform, "NoiseFunction::Uniform changed; update GPU shader decode");
}  // namespace
}  // namespace etx
