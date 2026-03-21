# Offline Comparison

The batch runner supports CPU/GPU offline comparison from the command line:

```text
raytracer --full-comparison --scene <scene-file> [options]
```

## Behavior

- The scene is loaded once for full-comparison mode.
- Each comparison technique reuses that loaded scene, applies the technique-specific settings, then runs CPU and GPU before comparing the results.
- Both renders must finish before image comparison starts.
- Full-comparison outputs are written into a dedicated folder next to the scene file:
  - `<scene-stem>.comparison/`
- Each comparison writes into that folder:
  - `*.exr` outputs for CPU and GPU
  - `*.png` outputs for CPU and GPU
  - `comparison.html` report with aligned CPU/GPU layer viewing
  - `comparison.ai.json` machine-readable summary for agents/tests
  - `*.cmp.exr` heatmap difference images
- Console output includes a stable machine-readable line per comparison result:

```text
AI_IMAGE_COMPARISON {"schema":"etx.image_comparison.v1", ...}
```

The same metrics are also written to:

```text
<scene-stem>.comparison/comparison.ai.json
```

## Agent Input

For future automated analysis or follow-up work, prefer reading:

```text
<scene-stem>.comparison/comparison.ai.json
```

instead of scraping console output or parsing `comparison.html`.

The file is intended to be the stable machine-readable artifact for agents and tests:

- schema: `etx.full_comparison.v1`
- one `results` array entry per comparison technique
- each entry includes:
  - scene path
  - CPU reference image path
  - GPU output image path
  - compare-space metrics
  - linear-space metrics

## Useful Options

- `--samples <count>`: override samples per pixel.
- `--max-path-length <count>`: clamp maximum path length for both CPU and GPU.
- `--random-seed <value>`: apply the same deterministic base seed to CPU and GPU sampling.
- `--resolution <width>x<height>`: override output resolution before rendering.
- `--crop <x>,<y>,<width>,<height>`: keep full-frame camera/output coordinates, render with crop-aware scheduling, then save and compare only that cropped window.
- `--strategy-flags <flag[,flag...]>`: override transport strategy flags for both renderers.

Supported `--strategy-flags` values:

- `direct_hit`
- `connect_to_light`
- `connect_to_camera`
- `connect_vertices`
- `merge_vertices`
- `none`

## Example

```text
raytracer --full-comparison --scene scenes/example.json --samples 64 --max-path-length 4 --random-seed 1337 --resolution 256x256 --crop 64,64,128,128 --strategy-flags direct_hit,connect_to_light
```

For `scenes/example.json`, the outputs are written under:

```text
scenes/example.comparison/
```

## Notes

- `--crop` keeps the scene and camera at full resolution, and the saved `EXR`/`PNG` outputs and comparison metrics use only the cropped window, without black borders.
- `pt` uses true crop-only tracing on both CPU and GPU.
- Light-path techniques preserve full-frame light-path equivalence so a cropped render matches a direct crop of the corresponding full-frame render:
  - GPU keeps full-frame light-path bootstrap and saves only the requested crop
  - CPU falls back to full-frame work for the non-`pt` bidirectional modes and saves only the requested crop
- `--resolution` still changes the actual full-frame render resolution before cropping.
