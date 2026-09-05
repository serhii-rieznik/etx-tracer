# Windows handover: application control and remote rendering

Date: 2026-08-17

Source branch: `gpu`

## Purpose

This change set separates common application actions from any particular user interface and adds UI-independent renderer modes. It is intended to be taken to Windows as one unit.

The existing Windows desktop application and its ImGui interface are still the default. No native Windows UI is introduced by this work. The new command/state layer lets the existing ImGui UI, the macOS native UI, command-line modes, and the built-in browser page drive the same renderer behavior.

The browser client does not stream video and does not use WebSockets. It polls immutable state, submits commands over HTTP, and retrieves the current rendered image as a PNG blob.

## What is included

### New files

- `sources/raytracer/application_control.hxx`
  - UI-neutral command, command-result, integrator, and application-state types.
- `sources/raytracer/application_runner.hxx`
- `sources/raytracer/application_runner.cxx`
  - Runtime-mode command-line parsing and common desktop/headless lifecycle.
- `sources/raytracer/application_control_server.hxx`
- `sources/raytracer/application_control_server.cxx`
  - Embedded HTTP server, JSON API, embedded HTML/CSS/JavaScript client, PNG endpoint, and selective scene upload sessions.
- `sources/raytracer/scene_dependencies.hxx`
- `sources/raytracer/scene_dependencies.cxx`
  - Dependency discovery for ETX JSON, Tungsten JSON, glTF/GLB, OBJ, and material descriptors.
- `docs/APPLICATION_CONTROL.md`
  - User-facing launch modes, API, and upload protocol reference.
- `docs/WINDOWS_HANDOVER.md`
  - This integration and validation guide.

### Modified areas

- `sources/raytracer/app.hxx` and `app.cxx`
  - Add `ApplicationConfig`, thread-safe command submission, command execution on the application thread, immutable state snapshots, retained results, and PNG capture.
  - Route standardized ImGui/native actions through the same application commands.
  - Keep detailed scene editing in-process and ImGui-specific for now.
- `sources/raytracer/main.cxx`
  - Select the UI-independent runtime before falling back to the existing batch parser and normal desktop startup.
- `sources/raytracer/renderer.hxx`
  - Add normalized renderer run states and control availability.
- `sources/raytracer/cpu_renderer.*` and `gpu_renderer.*`
  - Implement normalized `run`, `finish`, `stop`, and `restart` semantics and runtime statistics.
  - GPU rendering now has an explicit lifecycle instead of rendering unconditionally whenever it is initialized.
- `sources/raytracer/render_context.*`
  - Support desktop/headless configuration, optional ImGui, output sizing from the renderer in headless mode, and PNG readback.
- `sources/raytracer/ui.*`
  - Use normalized renderer control state and submit standardized actions through callbacks.
  - The Windows ImGui toolbar and menus remain enabled in normal desktop mode.
- `sources/raytracer/platform_ui_macos.mm`
  - Consume the normalized control state for macOS toolbar enablement. This does not change Windows UI behavior.
- `sources/raytracer/batch_mode.cxx`
  - Explicitly start the GPU renderer after the new lifecycle was introduced.
- `sources/raytracer/CMakeLists.txt`, `README.md`, and `.gitignore`
  - Register sources, link the documentation, and ignore copied runtime artifacts.

No files are deleted by this change set.

## Architecture

The key rule is that a UI or network thread never mutates renderer state directly.

```text
ImGui / macOS toolbar / HTTP / startup options
                       |
                       v
              ApplicationCommand queue
                       |
              application frame thread
                       |
                       v
       RTApplication::execute_application_command()
                       |
          scene / renderer / view operations
                       |
             immutable state snapshot
                       |
          every UI and HTTP client reads it
```

`RTApplication::submit_command()` assigns an ID and appends to a mutex-protected queue. `process_application_commands()` drains that queue from the application frame, executes each command, publishes a fresh state snapshot, and records an `ApplicationCommandResult`.

This is important on Windows because the HTTP server may receive work independently of the desktop event source, while RHI renderer mutations must remain serialized with the application frame.

### Standardized command surface

The shared command layer currently covers:

- scene load/save and reference-image load;
- image save and CPU-output denoising;
- renderer and integrator selection;
- run, finish-current-iteration, immediate stop, and restart;
- scene, geometry, and shader reload;
- renderer-preparation cancellation;
- exposure, view layer, output view, display transform, and GPU wavefront-step changes;
- application quit.

Material editing, emitter editing, camera manipulation, and other detailed scene tools remain local ImGui operations. They can be moved onto the same layer incrementally if a future remote UI needs them.

### State and results

`ApplicationStateSnapshot` exposes initialization, scene, renderer, integrator, preparation, run-state, timing/sample statistics, action availability, view settings, and quit state. Clients should use `controls.can_run`, `can_finish`, `can_stop`, and `can_restart`, plus `can_denoise`; they should not infer button availability from renderer names or sample counts.

Command acceptance and command completion are separate:

1. `POST /api/commands` returns HTTP 202 with a command ID after queueing.
2. `/api/results?after=<id>` later returns execution success or failure.

Results are retained rather than consumed by one client, so several controllers can poll independently. Both the application and server retention limits are currently 256 results.

## Runtime modes and command-line behavior

Normal desktop behavior remains:

```text
raytracer.exe
```

The control-server parser activates only when its mode flag is present. Existing batch-render and comparison command lines continue through the batch parser.

| Workflow | Window | ImGui/native controls | HTTP server | Lifetime |
| --- | --- | --- | --- | --- |
| Normal desktop | Yes | Yes | No | User-controlled |
| `--control-server` | No | No | Yes | API Quit or process signal |
| `--render` | No | No | No | Render target reached, output saved, then exit |

Examples:

```text
raytracer.exe --control-server
raytracer.exe --control-server --bind 192.168.1.20 --port 1654
raytracer.exe --render --scene C:\scenes\room.etx.json --output C:\renders\room.exr --samples 64 --renderer cpu
```

Relevant rules:

- The default HTTP port is `1654`.
- The default bind address is `127.0.0.1`.
- `--browser-control` is accepted as a compatibility alias for `--control-server` but is not the preferred public spelling.
- `--bind` accepts a numeric IPv4 address; host names and IPv6 are not implemented.
- A control-server session does not start rendering after scene load; the client must send `run`.
- Offline rendering and control-server sessions do not persist renderer selections into desktop preferences.
- The persistent native desktop restores its last valid scene from `options.json`.
- UI-independent modes do not restore or persist that scene; they require an explicit scene command or `--scene` option.

Headless rendering has no presentation-window resolution. The render output follows the active scene/film dimensions. The initial `1 x 1` runtime target is only a bootstrap allocation and is resized from `Renderer::output_size()` when a scene is available.

## HTTP control server

The implementation is a deliberately small HTTP/1.1 adapter embedded into `application_control_server.cxx`. It currently handles one accepted request per application poll and closes the connection after the response. Uploads use 512 KiB chunks so a large asset does not occupy one request.

Endpoints:

- `GET /` — embedded browser client.
- `GET /api/state` — current immutable state as JSON.
- `GET /api/results?after=<command-id>` — newer command results.
- `GET /api/image` — current output as an `image/png` blob.
- `POST /api/commands` — queue an application command.
- `POST /api/uploads` — create an upload session.
- `PUT /api/uploads/{id}/files/{index}?offset={bytes}` — append a requested chunk.
- `POST /api/uploads/{id}/resolve` — inspect uploaded descriptors and request the next dependencies.
- `POST /api/uploads/{id}/commit` — validate and queue scene loading.
- `DELETE /api/uploads/{id}` — cancel and clean up a session.

A PowerShell smoke test after launching `--control-server`:

```powershell
Invoke-RestMethod http://127.0.0.1:1654/api/state

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:1654/api/commands `
  -ContentType application/json `
  -Body '{"type":"set_renderer","renderer":"gpu"}'
```

The full command schema is in `docs/APPLICATION_CONTROL.md`.

### Rendered image behavior

`GET /api/image` performs a renderer-output readback, converts BGRA to RGBA when needed, and encodes PNG in memory. The built-in client treats the response as a blob and replaces the displayed object URL.

The page polls state every 500 ms and automatically refreshes the image when scene, renderer, run state, completed sample count, or view settings change. Automatic readback is throttled to at most once per second. The manual Refresh button remains available.

This is intentionally image transfer, not video streaming. PNG capture currently waits for the RHI to become idle, so frequent requests can temporarily stall rendering. This is acceptable for the current control/preview use case but should be revisited before increasing the refresh rate or supporting many simultaneous viewers.

## Browser client

The page is self-contained in the executable; there are no external HTML, CSS, JavaScript, font, or icon files to package.

The current page provides:

- system light/dark appearance through `prefers-color-scheme`;
- responsive desktop and narrow-window layouts;
- renderer and integrator selection;
- correctly state-driven Run, Finish, Stop, and Restart buttons;
- CPU-output denoising;
- pending, success, timeout, and failure feedback tied to each submitted command ID;
- live renderer name, state, samples, elapsed time, and command feedback;
- automatic image polling while rendering and idle, plus manual refresh;
- local host-path loading for trusted/local operation;
- selective remote scene upload.

The client has been visually checked on macOS in the in-app Chromium browser at desktop and compact widths. It has no observed browser console warnings or errors. On Windows, Edge/Chrome should be checked because folder selection relies on `webkitdirectory`.

## Selective remote scene upload

The browser asks the user to choose a containing folder so it can see relative file names. It does **not** upload that whole folder.

The workflow is:

1. The browser sends a manifest containing relative paths and sizes only.
2. The server requests the selected scene entry.
3. After receiving it, the server inspects references.
4. The server requests only newly discovered dependencies.
5. Steps 3-4 repeat until the dependency closure is complete.
6. The server reconstructs the required relative layout in a process-owned temporary directory and queues the normal `load_scene` command.

Dependency inspection currently understands:

- ETX JSON `geometry` and `materials` references;
- Tungsten JSON meshes, textures, environment/emission resources, and nested materials;
- glTF JSON/GLB buffers and images, excluding `data:` URIs;
- OBJ material libraries;
- ETX/MTL-style material texture, spectrum, shape, and volume references.

References absent from the selected folder are returned as `unavailable_references`. They are not fabricated or searched elsewhere; the actual scene loader determines whether a missing optional resource is fatal.

### Upload safety and lifecycle

- Paths must be portable relative paths.
- Absolute paths, root names, traversal, empty components, conflicting case-insensitive names, control characters, backslashes in manifest paths, and Windows-invalid punctuation are rejected.
- Backslashes found inside scene references are normalized before lookup.
- Files cannot be uploaded until dependency discovery explicitly requests them.
- Chunks must arrive sequentially at the expected offset.
- Committing an incomplete dependency set is rejected.
- Manifest limit: 8,192 files.
- Required-file limit: 8 GiB per file.
- Required-scene limit: 32 GiB per session.
- All active sessions may reserve at most 64 GiB in total.
- At most 16 upload sessions may exist at once.
- JSON documents and GLB JSON chunks are limited to 256 MiB during inspection.
- Incomplete sessions expire after 10 minutes.
- Committed sessions that never become active expire after one hour.
- Active uploaded data remains until another scene replaces it or the application exits.

On Windows, verify the temporary directory behavior under the account running the renderer. Reserved DOS device names such as `CON` or `NUL` are not proactively identified by the portable-path validator; filesystem creation should fail the upload cleanly, but this is an edge case worth testing if untrusted clients are expected.

## Renderer lifecycle changes

### CPU renderer

The CPU renderer maps integrator state to the shared run state:

- stopped: Run enabled;
- running: Finish, Stop, and Restart enabled;
- waiting for completion: Stop enabled;
- statistics: completed/target samples, elapsed time, and estimated time remaining.

Finish uses the existing `Integrator::Stop::WaitForCompletion`; Stop uses `Integrator::Stop::Immediate`.

### GPU renderer

GPU rendering now tracks `Stopped`, `Running`, `Finishing`, and `Completed` explicitly.

- Run resets progress and starts rendering once scene and pipelines are ready.
- Finish completes the current sample/iteration and then stops.
- Stop cancels preparation or active work immediately and returns to `Stopped`.
- Restart resets progress and begins again.
- Reaching the target sample count transitions to `Completed`.
- Pipeline preparation, scene validity, runtime failure, and pipeline availability determine action enablement.

The previous GPU batch path depended on implicit rendering. `batch_mode.cxx` now calls `gpu_renderer.start()` explicitly; preserve that line when resolving Windows-side conflicts.

## Windows-specific implementation notes

The server already contains conditional WinSock support:

- `WSAStartup(MAKEWORD(2, 2))` during initialization;
- `SOCKET`, `INVALID_SOCKET`, `closesocket`, and `WSAGetLastError` handling;
- `ioctlsocket(..., FIONBIO, ...)` on the listening socket;
- Windows receive/send timeout types;
- `WSACleanup()` during shutdown;
- `#pragma comment(lib, "ws2_32.lib")` for MSVC.

The project documentation currently assumes Visual Studio/MSVC. If another Windows toolchain is introduced, add `ws2_32` explicitly in CMake because the MSVC pragma will not provide linkage there.

When binding to a LAN address, Windows Defender Firewall may prompt for network access or block incoming connections. Only allow the executable on a trusted private network. The server has no authentication, authorization, encryption, or TLS.

The default loopback bind is safe for single-machine testing. For another computer on the LAN, bind the renderer to the Windows machine's numeric LAN IPv4 address and browse to `http://<that-address>:1654/` from the client computer.

Do not expose this server directly to the internet. Remote commands include reading and writing paths using the renderer process's permissions.

## Windows pickup procedure

1. Take the complete commit containing this document. Avoid copying only the HTTP files because the renderer lifecycle, application queue, render context, UI wiring, and batch fix are interdependent.
2. Configure the existing Visual Studio build as documented in `docs/BUILDING.md`.
3. Build at least the `raytracer` target in Debug and Release.
4. Confirm `ws2_32.lib` is present in the final link and all new files appear in the generated project.
5. Run the validation matrix below before making Windows-specific UI changes.
6. Keep the normal Windows desktop path on ImGui. Offline rendering and control-server execution use headless runtime output.
7. If Windows fixes are needed, keep them in the platform socket/RHI/runtime boundary. Do not fork the command schema or duplicate application action logic.

## Validation matrix

### 1. Build and legacy regression

- Build `raytracer` in Debug and Release.
- Run the existing offline/batch workflows to confirm the explicit GPU `start()` preserved them.
- Run `raytracer.exe` with no arguments.
- Confirm the normal Windows window, ImGui menu, and ImGui toolbar remain present.
- On a clean configuration, confirm no scene loads until one is selected.
- Reopen the desktop app and confirm the last valid scene is restored without starting a render.
- Confirm options still come from the shared application configuration location.

### 2. Desktop action wiring

With a normal local scene:

- switch among CPU, raster, and GPU where supported;
- change CPU integrators;
- exercise Run, Finish, Stop, and Restart in ImGui;
- confirm disabled/enabled states match actual behavior;
- confirm GPU buttons remain disabled while preparation is incomplete, then become usable;
- verify denoising becomes available only for stopped CPU output and updates the presentation;
- verify scene/geometry/shader reload actions still work.

### 3. Local headless server

Run:

```text
raytracer.exe --control-server
```

Then verify:

- startup without a visible window;
- `http://127.0.0.1:1654/` loads;
- `/api/state` reports `initialized: true`;
- a local host path can be loaded;
- renderer/integrator selection returns an accepted command and a successful result;
- Run, Finish, Stop, and Restart work for CPU and GPU;
- rendered image dimensions match the scene film, not a window size;
- the browser image updates during rendering without manual refresh;
- Ctrl+C exits and removes temporary upload data;
- a second process on port 1654 fails cleanly rather than hanging.

### 4. Remote LAN workflow

Run with the machine's private IPv4 address:

```text
raytracer.exe --control-server --bind 192.168.1.20 --port 1654
```

From a second computer on the trusted LAN:

- open the browser page;
- choose a folder containing a scene and unrelated large files;
- select the scene entry;
- confirm only the entry and discovered dependencies are uploaded;
- test spaces and non-ASCII characters in valid file names;
- test glTF/GLB, OBJ/material, ETX JSON, and Tungsten scenes used by the project;
- verify a missing dependency produces a visible warning/result;
- load a second uploaded scene and verify the first temporary scene is removed;
- cancel an in-progress upload and verify it can be restarted.

### 5. Offline rendering

Run:

```text
raytracer.exe --render --scene C:\scenes\room.etx.json --output C:\renders\room.exr --samples 64 --renderer cpu
```

Confirm:

- no window or server is opened;
- the requested sample target is completed;
- the output file is written and valid;
- the process exits with a successful status.

Repeat with GPU if the Windows RHI supports the selected scene.

### 6. Failure and security cases

- invalid bind address and occupied port;
- missing scene path and unsupported renderer/integrator;
- malformed command JSON;
- upload traversal and absolute-path attempts;
- duplicate manifest paths differing only by case;
- interrupted/out-of-order chunks;
- oversized manifest, file, and dependency document;
- client disconnect during a request;
- Windows Firewall denied and allowed cases.

## Validation completed before handover

The following checks were completed on Apple Silicon/macOS:

- `raytracer` compiled successfully;
- the native `raytracer_app` bundle compiled, bundled, and code-signed successfully;
- `playground` compiled successfully;
- control server started in headless mode and served the embedded page;
- desktop and compact browser layouts were visually inspected;
- connection/status updates worked and the browser reported no console warnings or errors;
- source formatting check passed for `application_control_server.cxx`.

Windows compilation and runtime validation have **not** been performed in this workspace. The WinSock path is implemented, but Windows RHI headless output sizing, synchronous PNG readback, temporary-file behavior, and Firewall/LAN access remain the highest-priority platform checks.

## Known limitations and deferred work

- No authentication, TLS, user accounts, or permission model.
- Numeric IPv4 only; no IPv6 or DNS-name bind parsing.
- No WebSocket transport or image streaming.
- No CORS support for a separately hosted browser frontend; the supplied page is same-origin.
- The small server is single-process and handles one accepted request per application poll.
- PNG capture is synchronous and can stall the renderer while the GPU becomes idle.
- The control API is not versioned yet.
- Detailed scene editing remains ImGui-only.
- Dependency discovery is format-aware but not a replacement for every loader; new scene/resource formats must add inspectors deliberately.
- Remote host-path commands are intentionally powerful and unsafe on an untrusted network.

## Recommended next steps after Windows validation

1. Fix only genuine Windows portability issues found by the matrix above.
2. Add an API version before publishing an external frontend.
3. Move synchronous PNG capture to a staged/asynchronous readback if preview traffic becomes a performance problem.
4. Add authentication before any use outside a trusted LAN.
5. Expand the shared command/state layer only when a second UI needs specific scene-editing operations.
