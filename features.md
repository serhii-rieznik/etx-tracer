## etx-tracer: Feature Overview

Physically based ray tracer focusing on spectral rendering, participating media, and modern light transport algorithms.

### Rendering algorithms

- **CPU Path Tracing**: classic path tracing with media support.
- **CPU Bidirectional Path Tracing (BDPT)**: modes for Path Tracing, Light Tracing, BDPT Fast, BDPT Full; options for direct hits, connect-to-camera, connect-to-light, connect vertices, MIS, blue noise.
- **CPU Vertex Connection and Merging (VCM)**: merging via spatial grid; kernels (Tophat/Epanechnikov), initial radius and decay; connection options; MIS; blue noise.
- **GPU backend (OptiX)**: device, pipelines, buffers; experimental PT/VCM kernels and OptiX denoiser.

### Spectral pipeline

- **Full-spectral rendering** by single-wavelength sampling; runtime switch between spectral and RGB.
- **Spectral sources**: sample pairs, RGB reflectance/luminance, blackbody and normalized blackbody, SPD files.
- **Color science**: XYZ↔RGB conversion; shared IOR datasets for conductor/dielectric/thin-film.

### Materials and BSDFs

- **BSDFs**: Diffuse, Translucent, Plastic, Conductor, Dielectric, Mirror, Boundary, Velvet, Principled (metal/roughness with optional transmission).
- **Microfacet multiple scattering** for conductor/dielectric; rough diffuse based on vMF model.
- **Thin-film interference** (thickness map/range, custom IOR) over any base, including conductors.
- **Normal mapping** with adjustable scale.

### Subsurface scattering

- **Random-walk SSS** and **Christensen–Burley approximation**.
- Applicable to materials with diffuse layer (diffuse, plastic, velvet); selectable path: Diffuse or Refracted.

### Participating media and volumes

- **Medium types**: Vacuum, Homogeneous, Heterogeneous.
- **Heterogeneous volumes** via NanoVDB (.nvdb); densities normalized from file range.
- **Phase function**: Henyey–Greenstein with anisotropy g; sampling and evaluation integrated.
- **Explicit connections toggle** for enclosed media; integrated transmittance queries.

### Emitters and lighting

- **Area lights**: one-sided, two-sided, or omni; textured emission with importance sampling.
- **Collimated emission** (laser-like) control.
- **Directional light** with finite angular diameter (sun disk); optional texture.
- **Environment maps** (HDRI) with rotation and importance sampling tables.
- **Atmospheric scattering**: physically-based sun and sky generation (Rayleigh/Mie/Ozone, altitude, anisotropy).

### Cameras

- **Classes**: Perspective, Equirectangular.
- **Depth of field**: lens radius, focal distance; arbitrary aperture via `lens_image`.
- **Camera medium** support.

### Textures and images

- **Formats**: PNG, EXR; RGBA8 and RGBA32F.
- **Sampling**: per-image importance sampling (row/column distributions), uniform sampling tables.
- **Addressing**: RepeatU/RepeatV; optional sRGB skip.
- **Mappings**: roughness/metalness channel packing; thin-film thickness maps; normal maps.

### Scene I/O and geometry

- **OBJ/MTL loader with extensions**:
  - Scene directives: `et::env`, `et::dir`, `et::atmosphere`, `et::medium`, `et::spectrum`, `et::camera` (see below).
  - Material params: class, IOR (constants or SPD), int/ext media, emitters, thin-film, normal maps, roughness, diffuse variation.
- **glTF/GLB loader**:
  - PBR Metallic-Roughness, base color textures.
  - MetallicRoughness packed textures; normal maps.
  - KHR extensions: emissive strength, transmission.
- **Geometry prep**: normal validation/fixup; MikkTSpace tangents; emitter distribution; environment emitter set.

### Blender export plugin

- Location: `blender/etx_tracer_exporter` (installable ZIP included next to the folder).
- Exports Blender scenes into etx-tracer format (geometry, materials, cameras, emitters, media, environment, textures).
- Maps Blender Principled BSDF to etx materials with consistent defaults; Principled is exported as `Plastic` (no metallic/conductor split) for stable behavior.
- Supports exporting normal maps, thin-film parameters, emission, and environment HDRI.
- Encodes lights/media/spectra using the same `et::...` scene directives consumed by the OBJ/MTL loader.

### OBJ/MTL scene directives (not materials)

Names starting with `et::` inside MTL files are not surface materials. They are scene directives consumed by the loader to create lights, media, spectra, or camera extras. Geometry should not reference these as BSDFs.

- **`et::env` (environment map)**

  - **image**: path to HDRI/EXR
  - **rotation**: degrees (yaw)
  - **color**: SPD name or RGB triplet (acts as illuminant)

- **`et::dir` (directional/sun light)**

  - **direction**: x y z
  - **angular_diameter**: degrees for finite sun disk
  - **image**: optional texture
  - **color**: SPD name or RGB triplet

- **`et::atmosphere` (physical sun+sky generator)**

  - **direction**, **angular_diameter**, **quality**
  - **scale**, **sun_scale**, **sky_scale**
  - **anisotropy**, **altitude**, **rayleigh**, **mie**, **ozone**

- **`et::medium` (volume definition)**

  - **id**: medium name (required)
  - **absorption/absorbtion**: r g b or scalar
  - **scattering**: r g b or scalar
  - **g / anisotropy**: Henyey–Greenstein parameter
  - **parametric**: color r g b, distance d or distances r g b, scale s
  - **volume**: path to .nvdb (heterogeneous)
  - **enclosed**: mark as enclosed (disables explicit connections)

- **`et::spectrum` (named SPD)**

  - **id**: spectrum name
  - **rgb** r g b, **blackbody** T, **nblackbody** T, or **samples** λ v ...
  - **normalize** [luminance], **scale** s, **illuminant** flag

- **`et::camera` (camera extras)**
  - **shape**: aperture image path
  - **ext_medium**: medium name

### Sampling, MIS, and controls

- **MIS** toggles (BDPT/VCM); vertex merging with spatial hash grid (VCM).
- **Blue-noise** per-pixel RNG for early iterations.
- **Pixel filtering**: Blackman–Harris with precomputed filter image; configurable radius.
- **Path controls**: min/max path length, RR start (random termination), radiance clamp; adaptive sampling noise threshold.

### Outputs, AOVs, and denoising

- **Film layers (AOVs)**: CameraImage, LightImage, LightIteration, Normals, Albedo, Result, Denoised, Adaptive, Internal, Debug.
- **Combined result** = CameraImage + LightImage.
- **Adaptive sampling**: per-pixel noise estimation and active-pixel tracking; progressive preview via pixel downscaling.
- **Denoisers**:
  - Intel OpenImageDenoise (CPU) with albedo/normal guides.
  - OptiX denoiser (GPU).

### Backends, infrastructure, and UI

- **CPU tracing** via Embree; **GPU** via OptiX (optional).
- **Task scheduler** for multithreaded rendering.
- **Sokol + ImGui UI**: integrator selection, options (BDPT/VCM), denoise, camera/material/medium controls, scene load/save.

### References

- **PBRT book**: [pbrt-v3/v4](https://www.pbr-book.org/)
- **VCM paper/implementation**: [Georgiev et al. 2012](https://cgg.mff.cuni.cz/~jaroslav/papers/2012-vcm/)
- **Heitz et al.** Multiple-Scattering Microfacet BSDFs with the Smith Model: [paper](https://eheitzresearch.wordpress.com/240-2/)
