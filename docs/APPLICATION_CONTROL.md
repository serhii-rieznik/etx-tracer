# Application control

ETX Tracer exposes core renderer and view operations through typed commands and immutable state snapshots. The native macOS UI, ImGui UI, command-line runner, and browser client use the same command path for those operations. Detailed scene editing remains an in-process ImGui feature for now.

Windows pickup notes, platform-sensitive areas, and the validation matrix are documented in [WINDOWS_HANDOVER.md](WINDOWS_HANDOVER.md).

## Scene editing

The scene tree offers **Duplicate Linked** and **Duplicate Independent**. Linked duplicates share meshes and materials. Independent duplicates copy meshes, materials, and their spectra while preserving mesh/material sharing within the copied assembly. Texture assets remain shared; replacing a texture assignment changes only its material. Both modes copy camera, emitter, and attached-medium resources. Copied cameras start inactive. References to lights and node-owned volumes outside the copied assembly remain scene-level references, preserving their external ownership and transforms.

A linked assembly containing a volume needs private material assignments wherever its boundary materials reference that volume. Those meshes and materials are copied so the new boundaries refer to the copied volume. Other meshes remain linked. Failed duplication restores the original hierarchy and resource storage.

Changing **Assigned Material** in an object's inspector isolates its shared mesh before assigning the new material. **Make Unique** isolates that object's mesh and material spectra. The sharing indicator counts objects, including instances of the same mesh. Direct edits to a shared material still affect its users.

**Bake Transform** and **Center Pivot** copy geometry when another object or mesh shares its storage. Otherwise they update the existing mesh. Copies are ordinary meshes, and subsequent edits follow the same sharing check after save/load. Identity bakes and exactly centered pivots do not create geometry.

Workspace panels and resource-category headers use subtle background tints in both themes. In the resource list, resources used by the scene have yellow-tinted text; unused resources have gray text. Both remain selectable. Usage follows object attachments, active cameras, used mesh materials, and their medium references. Disabled objects still reference their assets. Standalone environment/directional lights count as used because they contribute without a node attachment; derived area lights follow their materials.

### Editing contracts for subsequent milestones

Viewport picking will select the nearest eligible surface and resolve instances to nodes; transparent/opacity handling must be specified against the picking backend before implementation. Navigation, gizmos, and selection will each own complete gestures. Escape will cancel the current gesture. Group transforms will act once on selected hierarchy roots, and editor navigation will be separated from authored render cameras. These interactions are not implemented yet.

## Spectrum editor

Material, light, and medium properties use a compact spectrum control: a label above one row containing source mode, source value, strength multiplier, and a **...** editor button. **...** opens the same state in the bottom **Spectrum** panel. IOR properties offer **Edit eta** and **Edit k**; their paired preset/file picker applies both channels together. Opening a control preserves the existing rendered spectrum.

Choose **Color**, **Temperature**, or **Spectrum** from the mode dropdown. Click the color swatch to open its picker; temperature is edited inline. Color uses sRGB input for reflectance and emission, and linear RGB for medium coefficients and subsurface distance. Temperature produces normalized blackbody emission, or a temperature-derived RGB tint for reflectance. Coefficients offer Color and Spectrum; IOR channels offer a wavelength-independent **Value** and Spectrum. Switching to a generated source changes the spectrum; switching back to Spectrum restores the session's custom curve.

**Strength** multiplies the source without changing it. Values below one and zero are supported; increasing strength after zero restores the source. Scale uses smaller text and four decimal places. Double-click or Ctrl-click the field to type a value; stored precision is retained. The compact control has no preview graph. Labels turn amber for invalid output or reflectance above one; hover for details. Mixed selections show **Mixed** in the mode dropdown.

In Spectrum mode, click the source name to load a two-column SPD file or search matching built-in presets. Imported points retain their values and precision; legacy wavelength units are converted to nanometres. Invalid files leave the current spectrum unchanged and display an error. Built-in spectrum files are read-only. Paired IOR files remain available through the IOR picker.

The expanded source curve is edited before strength. Click the curve or use **Add point** to insert a wavelength/value pair. Drag points to move them. Enter a wavelength to select a point, then edit **Value**; numeric fields commit on Enter or focus loss. Adding an existing wavelength selects it. **Delete point** removes interior points. Endpoints are protected. Resize the bottom panel with its upper divider.

**Simplify** retains original points within an explicitly chosen error tolerance, expressed as a fraction of the curve's value range. **Normalize peak to 1** rescales any nonzero curve to a peak of one while preserving its wavelengths. Neither operation happens automatically. The graph is always visible, including Color and Temperature modes. Mode and parameter changes update it immediately. Editing generated points directly switches the source to Spectrum mode.

Edits apply to the selected material set; shared materials affect all their users. Ordinary spectra update immediately after UI construction. IOR changes wait for interaction release, and renderer resource preparation completes before rendering resumes. Target bindings are checked before a batch is applied.

Scene saves embed the active source mode, parameters, unscaled spectrum, and authored points. Reloading does not require the original SPD file. Independent material copies retain this state. Legacy scenes keep their exact existing spectrum until edited.

## Launch modes

Normal desktop behavior is unchanged:

```text
raytracer
```

Run without a window or ImGui and expose the local browser client:

```text
raytracer --control-server
```

Open `http://127.0.0.1:1654`. The server binds only to the loopback interface unless another IPv4 address is explicitly supplied with `--bind`.
The server has no authentication. Only use a non-loopback bind address on a trusted network, because control commands can read and write files with the renderer process's permissions.

Run an offline render without any UI or server:

```text
raytracer --render --scene /path/to/scene.etx.json --output /path/to/output.exr --samples 64 --renderer cpu
```

Offline rendering runs to its configured sample or time target, saves the requested output, and exits. Control-server mode leaves the renderer stopped until a `run` command is sent. Neither workflow overwrites saved desktop preferences.
Headless output has no independent presentation size. `GET /api/image` follows the active renderer's scene/film dimensions. The built-in browser client polls state and command results without overlapping requests. It refreshes the PNG blob when rendering or view state changes, once per second while rendering, and every five seconds while idle; its Refresh button remains available for an explicit update.

## Pixel reconstruction

CPU and GPU rendering use uniform sampling to the configured sample or time limit. The former `noise-threshold` scene field is ignored on load and omitted on save. Noise-based pixel stopping and its diagnostic views have been removed. View-layer IDs remain Result=0, Denoised=1, Current Frame=2, Accumulation=3, Albedo=5, and Normals=6; retired IDs 4 and 7 are rejected by `set_view_layer`.

Scene JSON `pixel-filter-radius` specifies additional smoothing in film-pixel units. Zero gives a one-pixel box response; it does not select a point sample at the pixel centre. Positive values convolve that footprint with the normalized Blackman-Harris sampling table. The default remains 1.5.

CPU and GPU camera paths sample the pixel area plus a filter offset. Light-tracing splats use the opposite offset before pixel assignment, including contributions projected just outside the unfiltered frame. Both estimate the same filtered image without changing exposure or normalizing by the number of accepted splats. The first accumulated sample uses the same reconstruction as subsequent samples.

This convention changes edge response relative to older builds that used point sampling at radius zero, ignored the filter in CPU VCM, or deposited unfiltered light splats. The smoothing radius is independent of VCM/UPBP merging radii.

## HTTP API

The built-in page is one client of the API; another UI can use the endpoints directly.

- `GET /api/state` returns the current immutable application snapshot.
- `GET /api/results` returns recently completed command results. Pass the last observed command ID as `?after=<command_id>` to receive only newer results; results are retained so multiple clients do not consume each other's notifications.
- `GET /api/image` returns the latest headless presentation image as a PNG blob.
- `POST /api/commands` queues a command and returns its command ID.

Commands are JSON objects with a `type` field. Supported types are:

- `load_scene`, `save_scene`, and `load_reference` with `path`
- `save_image` with `path` and `format` (`png` or `exr`)
- `denoise`
- `set_renderer` with `renderer` (`cpu`, `raster`, or `gpu`)
- `set_integrator` with the numeric integrator `value`
- `run`, `finish`, `stop`, `restart`
- `reload_scene`, `reload_geometry`, `reload_shaders`, `cancel_preparation`
- `set_exposure` with floating-point `value`
- `set_view_layer`, `set_output_view`, and `set_display_transform` with numeric `value`
- `quit`

For example:

```text
POST /api/commands
Content-Type: application/json

{"type":"set_renderer","renderer":"gpu"}
```

The response confirms queueing, not execution:

```json
{"accepted":true,"command_id":12}
```

The corresponding result is later returned by `/api/results` (or `/api/results?after=11`):

```json
[{"command_id":12,"success":true,"message":"Renderer changed"}]
```

The built-in page associates that result with the submitted command ID. It keeps the corresponding control pending after acceptance, then presents the renderer's success or failure message and reconciles the controls with the next state snapshot. State and result polling runs sequentially at approximately 500 ms intervals so a slow scene operation cannot accumulate overlapping browser requests.

The server is deliberately a transport adapter. Renderer and scene mutations execute on the application thread when its command queue is drained, so HTTP handling does not mutate rendering state concurrently.

## Remote scene upload

The browser can load a scene from its own computer. Browser security requires the user to grant access to a containing folder; that permission is not an instruction to upload the folder. The page sends a manifest of relative paths and sizes, uploads the selected scene, and asks the renderer to inspect its dependencies. Only the resulting dependency closure is transferred. Relative paths are reconstructed in a process-owned temporary directory so scene loaders see the same layout as on the client.

The upload protocol is intentionally chunked so large meshes do not block one HTTP request:

- `POST /api/uploads` creates a session from an `entry` path and a `files` manifest. Its response lists the initially requested file and the maximum chunk size.
- `PUT /api/uploads/{id}/files/{index}?offset={bytes}` appends one requested binary chunk. Chunks must be sequential.
- `POST /api/uploads/{id}/resolve` inspects completed descriptors and returns the next referenced files to upload. Repeat until `complete` is true.
- `POST /api/uploads/{id}/commit` validates the resolved dependency set and queues the normal `load_scene` application command.
- `DELETE /api/uploads/{id}` cancels and removes a session. An upload currently used by the renderer cannot be removed.

Upload paths are restricted to portable relative paths; absolute paths, traversal, name conflicts, unrequested files, out-of-order chunks, and incomplete commits are rejected. A session supports up to 8,192 manifest entries, 8 GiB per required file, and 32 GiB of required scene data. JSON dependency documents and GLB JSON chunks are limited to 256 MiB during inspection. Files referenced by a scene but absent from the selected folder are reported as unavailable; the scene loader still decides whether such an optional resource is fatal. Uploaded data is removed when the application shuts down, after the active scene has been released, or immediately if its load command fails.
