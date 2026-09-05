# ETX Tracer Project Context

## Product and correctness

- ETX Tracer is a C++23 physically based renderer for Windows and macOS. It provides CPU, GPU, and raster renderers; RGB and full-spectral modes; VCM and volumetric bidirectional integration; participating media, subsurface scattering, and layered BSDFs.
- Rendering changes must preserve the estimator rather than only produce visually similar output. Keep wavelength and sampling-PDF corrections, throughput, BSDF and phase-function PDFs, reciprocity, MIS weights, and VCM connection/merge semantics consistent across affected CPU/GPU and RGB/spectral paths.
- Trace the real runtime path before changing renderer behavior. An implementation in a shared layer is not sufficient evidence that every renderer, preparation path, mode switch, or application entry point uses it.

## Source map

- `sources/raytracer/` owns the executable application, CPU/GPU/raster renderer adapters, native and ImGui UI, headless and batch modes, browser-control transport, and shader-package orchestration.
- `sources/etx/render/` is the canonical shared rendering layer for scene data, materials, spectra, host/GPU accessors, GPU ABI definitions, and shader sources. Shared HLSL and validation shaders belong under `sources/etx/render/shaders/`; keep matching host/shader ABI changes synchronized.
- `sources/etx/rhi/` contains the Metal and Vulkan backends. `sources/etx/rt/` contains Embree-backed CPU ray tracing and integrators. `sources/etx/core/` and `sources/etx/engine/` provide common runtime and application infrastructure.
- `sources/tests/` contains maintained standalone validation executables. `bin/` contains runtime assets, sample data, executable output, caches, and generated shader packages; do not treat the whole directory as disposable build output.
- `thirdparty/` contains vendored dependencies. Avoid changing vendored code unless the task specifically requires it.

## Build and validation workflow

- Use the repository CMake workflow and an existing configured build directory when suitable. Configuration requires Embree and DXC; see `docs/BUILDING.md` for current platform setup.
- On Windows, build `raytracer`; production Release validation must also build `raytracer_shader_package`. On macOS, build `raytracer_app`, which depends on the shader package.
- Debug and RelWithDebInfo development builds compile the copied runtime shader sources. After shader changes, rebuild and restart the application. Release builds consume `shaders.etxpack`; missing, corrupt, or incomplete packages are errors.
- Run the narrowest relevant maintained validation target, then build the affected application target. Renderer, mode-switch, asynchronous preparation, presentation, and UI changes also require a representative runtime smoke test; successful compilation alone is not runtime validation.
- Use `docs/APPLICATION_CONTROL.md` for supported desktop, offline-render, and browser-control behavior. Keep renderer and scene mutations on the application command path described there.

# Project Coding Rules

- Do not implement hacks, workarounds, empirical fitting, special-case compensation, or data-matching patches unless the user explicitly asks for that approach and confirms it after the tradeoffs are stated.
- Do not commit one-off validation or development artifacts. This includes temporary test cases added only to validate the current change, ad hoc test programs, test scenes, validation scripts, rendered images, comparison reports, and generated test executables. Keep these local and ignored, and remove temporary changes to tracked test files before committing.
- Maintained, reusable validation infrastructure may be committed when it belongs to the project proper. Shared validation shaders must live in the common `sources/etx/render/shaders/` directory; they must not be placed in a task-specific validation or assets folder.
- Permanent regression tests are appropriate only when they are intended to remain maintained as part of the regular test suite. Do not turn a one-time implementation check into a permanent test without explicit user approval.
