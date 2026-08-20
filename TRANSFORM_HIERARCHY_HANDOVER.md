# Transform and Scene Hierarchy Production Handover

Date: 2026-08-20

Target branch: `transform`

Repository: `F:\Projects\etx-tracer`

Status: implementation complete and validated on Windows/Vulkan and macOS/Metal. The macOS continuation added orientation-only hierarchy propagation, transform-aware editor windows, Metal runtime fixes, and bounded CPU/GPU validation.

## 1. Executive summary

This change introduces a production-oriented object hierarchy and transform pipeline across scene import, native persistence, CPU ray tracing, GPU ray tracing, lighting, media, cameras, and editor UI.

The central design decision is that the authoring hierarchy is a set of flat contiguous arrays. It is not a recursive tree of heap-allocated nodes. Parent/child relationships are expressed by integer indices, attachments are compact records referencing existing resource arrays, and evaluation uses precomputed iterative traversal order. The hierarchy never crosses the CPU/GPU boundary. Before rendering, it resolves into a dense array of minimal render instances containing only the transforms and metadata needed by Embree and GPU shaders.

The implementation supports:

- arbitrary-depth hierarchies without recursion;
- full affine local transforms represented as row-major 3x4 matrices;
- shared mesh assets referenced by multiple transformed instances;
- inherited node enable/disable state;
- negative-determinant/mirrored transforms;
- singular-transform handling that preserves stable instance indices while masking invalid instances;
- transformed cameras, directional emitters, area emitters, and media;
- Embree BLAS sharing and transform-only instance commits;
- GPU BLAS sharing and TLAS refit/update for transform-only changes;
- explicit, compile-time-checked CPU/GPU ABI layouts;
- native hierarchy round trips and backward-compatible geometry loading;
- iterative glTF hierarchy import with mesh instancing, cameras, punctual directional lights, and `doubleSided` material behavior;
- separate Scene Tree and Node Properties windows for hierarchy selection, reparenting, enable state, and local TRS editing, with non-TRS affine matrices preserved until an explicit reset.

The implementation intentionally does not upload nodes, names, parents, traversal order, or attachment records to the GPU. GPU memory contains the dense render-facing `SceneInstance` array only.

## 2. Goals and non-goals

### 2.1 Goals delivered

1. Preserve object hierarchy from supported scene import paths.
2. Make local-to-world transform propagation correct for deep parent chains.
3. Avoid recursive traversal and pointer-linked node graphs.
4. Share mesh geometry across instances on both CPU and GPU.
5. Keep transform-only updates out of BLAS rebuild paths.
6. Keep instance indices stable when a node is disabled or its transform becomes singular.
7. Apply transforms consistently to positions, normals, tangent frames, lights, cameras, and media.
8. Protect all modified CPU/GPU layouts with shared constants and compile-time assertions.
9. Validate without depending on the interactive window.
10. Keep one-off scenes, rendered images, and comparison artifacts out of source control.

### 2.2 Explicit non-goals

- No recursive scene-node ownership API was introduced.
- No GPU-side hierarchy evaluation was introduced.
- No animation system, skinning system, or transform keyframe format was added.
- No general DAG with multiple parents is supported. The runtime hierarchy is a forest: every node has zero or one parent.
- No effort was made to solve pre-existing CPU/GPU normalization differences in light tracing.
- No environment-only acceleration-structure redesign was included.
- No speculative job-system or GPU transform-propagation abstraction was added.
- No one-off validation scenes or render outputs are committed.

## 3. Architectural overview

The data flow is:

```text
scene file / importer
        |
        v
mesh assets + cameras + emitter profiles + media
        |
        v
flat SceneHierarchy
  nodes[]
  node_names[]
  attachments[]
        |
        | rebuild_topology() when parent structure changes
        | update_world_transforms() for dirty subtree intervals
        | resolve_mesh_instances()
        v
resolved host state
  evaluation_order[]
  world_transforms[]
  effective_enabled[]
  mesh_instances[]
        |
        +-------------------------------+
        |                               |
        v                               v
CPU packed Scene                    GPU packed scene
SceneInstance[]                     SceneInstance raw buffer
Emitter[]                           Emitter raw buffer
Medium[]                            Medium blob
        |                               |
        v                               v
Embree mesh BLAS +                 RHI mesh BLAS +
instance geometry                  TLAS instances
```

There are three separate representations for deliberate reasons:

1. `SceneHierarchy` is authoring/import/editor state. It retains names, parent indices, local transforms, and resource attachments.
2. `ResolvedMeshInstance` is a host-only intermediate. It adds world bounds and the source node index needed while producing render state.
3. `SceneInstance` is the CPU/GPU rendering ABI. It contains two affine matrices plus four 32-bit metadata values and nothing hierarchy-specific.

This separation keeps hierarchy ergonomics and traversal metadata off the GPU while retaining a compact, sequential render path.

## 4. Flat hierarchy data model

Primary files:

- `sources/etx/render/host/scene_hierarchy.hxx`
- `sources/etx/render/host/scene_hierarchy.cxx`
- `sources/etx/render/interop/math_shared.hxx`

### 4.1 `AffineTransform`

`AffineTransform` is three `float4` rows:

```text
[ m00 m01 m02 tx ]
[ m10 m11 m12 ty ]
[ m20 m21 m22 tz ]
```

Properties:

- row-major;
- 48 bytes;
- 16-byte aligned through the shared interop alignment contract;
- directly consumable by Embree as `RTC_FORMAT_FLOAT3X4_ROW_MAJOR`;
- copied directly into `RHIAccelerationStructureInstance::transform`;
- identity by default;
- represents translation, rotation, non-uniform scale, shear, and reflection;
- cannot represent perspective, which is correct for object transforms.

Transform composition uses:

```text
world(node) = world(parent(node)) * local(node)
```

Points use the full affine transform. Vectors omit translation. Normals use the transpose of the stored world-to-object linear transform, followed by normalization.

### 4.2 `SceneNode`

`SceneNode` is exactly 64 bytes:

| Field | Meaning |
| --- | --- |
| `local_transform` | 48-byte local affine transform |
| `parent_index` | Index into `nodes`, or `kInvalidIndex` for a root |
| `attachment_offset` | Start of this node's contiguous range in `attachments` |
| `attachment_count` | Number of attachments owned by the node |
| `flags` | Currently contains the local `Enabled` flag |

There are no child vectors, pointers, virtual functions, per-node allocations, or recursive ownership objects.

### 4.3 `SceneAttachment`

`SceneAttachment` is exactly 16 bytes:

| Field | Meaning |
| --- | --- |
| `type` | `Mesh`, `Camera`, `Emitter`, or `Medium` |
| `resource_index` | Index into the corresponding resource array |
| `flags` | Reserved for attachment behavior |
| `reserved` | Keeps the record compact and layout-stable |

Attachments are references. They do not own or duplicate mesh, camera, emitter, or medium data. A node owns a contiguous attachment range, and insert/remove operations repair offsets for later nodes.

### 4.4 Derived arrays

`SceneHierarchy` stores the following dense derived arrays:

| Array | Purpose |
| --- | --- |
| `evaluation_order` | Parent-before-child iterative traversal order |
| `order_position` | Node index to position in `evaluation_order` |
| `subtree_end_position` | Exclusive end of each node's contiguous DFS subtree interval |
| `world_transforms` | Resolved object-to-world transform for every node |
| `world_orientations` | Rotation-only world transform for cameras and infinite emitters |
| `orientation_valid` | Whether the local-to-world chain can be represented without shear/singularity |
| `effective_enabled` | Local enabled state combined with all ancestors |
| `mesh_instances` | Dense renderable mesh attachment results |

`evaluation_order`, `order_position`, and `subtree_end_position` are rebuilt only when topology changes. The subtree interval representation allows a local transform change to mark one contiguous evaluation range dirty.

### 4.5 Public mutation contract

Use the hierarchy methods instead of mutating topology fields directly:

- `add_node(name, parent_index, local_transform)`
- `set_parent(node_index, parent_index)`
- `set_local_transform(node_index, local_transform)`
- `set_enabled(node_index, enabled)`
- `add_attachment(node_index, attachment)`
- `remove_attachment(node_index, local_attachment_index)`
- `rebuild_topology()`
- `update_world_transforms()`
- `resolve_mesh_instances(meshes)`

Directly changing `parent_index`, `local_transform`, or attachment ranges would bypass dirty/topology bookkeeping. New editor or scripting APIs should route mutations through these methods.

## 5. Topology and transform evaluation

### 5.1 Forest validation

The topology builder accepts multiple roots and produces one flat forest traversal. It validates indices and rejects cycles. `set_parent` rejects:

- a node parenting itself;
- a parent outside the node array;
- a reparent operation that would make the node an ancestor of itself.

The glTF loader also prevents a source node from being instantiated under more than one parent. If malformed or DAG-like input references a node more than once, the first parent wins and a warning is emitted.

### 5.2 No recursion

Topology construction and world-transform propagation use explicit dense vectors/stacks. The maintained regression test creates a chain of 100,000 nodes and resolves it successfully. This prevents stack exhaustion and makes traversal memory visible and bounded.

### 5.3 Dirty ranges

Each node's descendants occupy a contiguous interval in `evaluation_order`. A local-transform change calls `mark_subtree_dirty(node_index)` and unions that interval into the current dirty range. Enabled state does not make matrices dirty; effective enabled state is recomputed sequentially during mesh-instance resolution.

On the next update:

1. topology is rebuilt first if required;
2. nodes in the dirty evaluation interval are visited parent-before-child;
3. each world transform is recomputed from its parent's current world transform and its local transform;
4. each decomposable local rotation is composed with its parent's rotation-only world orientation;
5. effective enabled state is recomputed from local and parent state;
6. unaffected derived entries remain unchanged.

The full affine transform remains authoritative for positions, meshes, media, and area emitters. Rotation-only propagation prevents inherited non-uniform scale from skewing camera bases and infinite-light directions. A sheared or singular local transform invalidates the orientation chain instead of silently inventing a rotation.

This is partial transform propagation. The subsequent render-instance resolution currently scans mesh attachments to regenerate the dense instance view. That scan is simple and sequential; further optimization should be measurement-driven rather than adding speculative indexing structures.

### 5.4 Invalid and singular transforms

Non-finite authoring transforms are rejected.

A finite but singular transform is allowed to remain in authoring state. During instance resolution its inverse cannot be generated, so the instance remains in the dense array but loses `Enabled`. This is intentional:

- indices after it do not shift;
- Embree instance geometry can be masked instead of removed;
- GPU TLAS instance custom indices remain stable;
- wavefront state cannot accidentally refer to a newly remapped object.

Negative determinants are valid. They set `Mirrored`, enabling correct normal/tangent-frame orientation handling.

## 6. Resolved render instances

### 6.1 Host intermediate: `ResolvedMeshInstance`

For every mesh attachment, resolution produces:

- `object_to_world`;
- `world_to_object`;
- transformed world-space `bbox_min` and `bbox_max`;
- source `node_index`;
- `mesh_index`;
- `Mirrored` and `Enabled` flags.

This record is host-only. Bounds are useful for scene bounds and validation but are not uploaded in every GPU instance record.

### 6.2 CPU/GPU ABI: `SceneInstance`

`SceneInstance` is exactly 112 bytes and 16-byte aligned:

| Byte offset | Size | Field |
| ---: | ---: | --- |
| 0 | 48 | `object_to_world` |
| 48 | 48 | `world_to_object` |
| 96 | 4 | `mesh_index` |
| 100 | 4 | `flags` |
| 104 | 4 | `emitter_offset` |
| 108 | 4 | `emitter_count` |

Flags:

- bit 0: `Mirrored`;
- bit 1: `Enabled`.

The two matrices are stored rather than inverting in shaders. This costs 96 matrix bytes per instance but avoids repeated inverse work at hit frequency and gives a stable normal-transform path. For path tracing workloads this is the correct CPU/GPU tradeoff.

`emitter_offset` and `emitter_count` identify the sorted per-instance area-emitter subrange. They make hit-to-emitter mapping possible without a global linear scan.

## 7. Import paths

### 7.1 glTF

Primary file: `sources/etx/render/host/scene_gltf_loader.cxx`.

The importer now separates mesh assets from node instances:

- a glTF mesh/primitive is loaded once into shared vertex/index/mesh arrays;
- every node mesh reference becomes a hierarchy mesh attachment;
- multiple nodes can therefore reference the same mesh asset without duplicating geometry;
- node traversal is iterative;
- named nodes retain names;
- unnamed nodes use deterministic `gltf-node-<index>` names.

Local transform behavior:

- a 16-element glTF matrix is honored when present;
- otherwise translation, quaternion rotation, and scale are composed according to glTF semantics;
- the result is stored as the node's local 3x4 affine transform;
- child transforms are not baked into vertices.

Additional glTF behavior included in this change:

- cameras are imported as camera resources attached to their source nodes;
- `KHR_lights_punctual` directional lights are imported as emitter profiles attached to nodes;
- glTF's local light direction convention is converted so the renderer receives its expected toward-light direction;
- material `doubleSided` maps to the renderer's `two_sided` flag;
- invalid child indices are warned and skipped;
- duplicate source-node references keep the first parent and warn.

Point and spot punctual lights are not newly generalized by this hierarchy work. Directional support is the transform-relevant path implemented here.

### 7.2 OBJ and procedural geometry

OBJ and procedural sources do not carry a general object-node hierarchy. They receive deterministic identity nodes and mesh attachments so all render paths use one hierarchy/instance model.

This compatibility synthesis happens only when no hierarchy was loaded. It prevents older and simpler scene formats from bypassing instancing logic.

### 7.3 Tungsten

The Tungsten loader routes object transforms and supported camera/emitter relationships into the same hierarchy representation. Directional and area-light behavior is resolved through the common emitter packing path rather than maintaining a parallel transform system.

### 7.4 Native ETX scenes

Native scenes load `scene_hierarchy` when present. Older native scenes without it synthesize identity nodes after geometry load.

All import paths converge on `SceneData::resolve_hierarchy()` before the scene is committed for rendering.

## 8. Native persistence and compatibility

### 8.1 JSON hierarchy format

The native scene JSON contains:

```json
"scene_hierarchy": {
  "version": 1,
  "nodes": [
    {
      "name": "example",
      "parent": null,
      "flags": 1,
      "transform": [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0
      ],
      "attachments": [
        {"type": "mesh", "index": 0, "flags": 0}
      ]
    }
  ]
}
```

The example above matches the version-1 writer. The authoritative reader/writer remains `scene_representation.cxx`.

Important persistence rules:

- transforms are full 12-component local affine matrices;
- roots serialize `parent` as `null`;
- node order is stable and parent references are indices;
- flags are persisted;
- mesh, camera, emitter, and medium attachment resource indices are persisted;
- all cameras are preserved, including active state and hierarchy attachment relationships;
- load validates version, node types, transform component count/type, parent range, attachments, and topology.

### 8.2 Binary geometry version

The binary geometry format writes version 2:

- `kBinaryGeometryLegacyVersion = 1`
- `kBinaryGeometryVersion = 2`

The loader accepts both versions 1 and 2. Version 2 supports the current mesh-asset representation while preserving compatibility with existing version-1 geometry. Hierarchy authoring data itself is stored in native scene JSON, not as GPU-ready instance blobs in the geometry file.

### 8.3 Save semantics for transformed resources

Camera resource values are stored in their local/resource form, while hierarchy transforms remain in `scene_hierarchy`. The save path uses attachment transforms where needed to avoid double-applying world state. Multiple attached cameras survive a save/load round trip.

## 9. CPU/Embree rendering path

Primary file: `sources/etx/rt/rt.cxx`.

### 9.1 Acceleration-structure organization

Embree uses a two-level structure:

1. one Embree mesh scene per mesh asset;
2. one top-level Embree scene containing instance geometry.

Each mesh scene shares the renderer's vertex data and points to the mesh's triangle range. Each resolved `SceneInstance` creates an `RTC_GEOMETRY_TYPE_INSTANCE` referencing the corresponding mesh scene.

The instance is configured with:

- `RTC_FORMAT_FLOAT3X4_ROW_MAJOR` transform;
- geometry ID equal to the stable instance index;
- full mask for enabled instances;
- zero mask for disabled or singular instances.

### 9.2 Hit mapping

Embree returns the top-level instance/geometry ID and the mesh-local primitive ID. The renderer maps these to:

- stable `instance_index`;
- `SceneInstance::mesh_index`;
- global triangle index using the mesh's triangle offset;
- material and emitter lookup;
- transformed surface data.

This mapping is essential because geometry is no longer baked uniquely per object.

### 9.3 Transform-only commit

`SceneHashes` distinguishes topology/geometry structure from transform content. On a transform-only change:

1. render-facing instance matrices are refreshed;
2. existing Embree instance geometries receive new transforms and masks;
3. each changed instance geometry is committed;
4. the top-level Embree scene is committed;
5. mesh scenes and triangle geometry are retained.

A geometry-structure change still rebuilds the relevant Embree scene organization. Transform updates do not rebuild mesh BLAS-equivalent scenes.

### 9.4 Surface frames

Positions are transformed with `object_to_world`.

Normals are transformed with the inverse-transpose linear transform derived from `world_to_object`, then normalized. Mirrored instances multiply the orientation by `-1` so the geometric and shading frame follow transformed winding.

Tangents and bitangents preserve source tangent-frame handedness under reflection. This is covered by a maintained regression test.

Before BSDF construction, CPU and GPU hit paths run the same shared frame finalization. It normalizes and hemisphere-aligns shading normals against the world-space geometric normal and incident ray, projects both tangent hints, preserves handedness, and falls back to a deterministic orthonormal basis for missing or degenerate tangents. Normal maps then rebuild the complete frame through the same helper. Stored VCM/BDPT path vertices also restore their world-space tangent frame before connection evaluation.

## 10. GPU rendering path

Primary files:

- `sources/raytracer/gpu_renderer.cxx`
- `sources/raytracer/gpu_renderer.hxx`
- `sources/etx/rhi/rhi_types.hxx`
- `sources/etx/rhi/vulkan/vk_rhi.cxx`
- `sources/etx/rhi/metal/mt_rhi.mm`
- `sources/etx/render/shaders/gpu_rt_shared.hlsl`

### 10.1 GPU scene data

The `GPUScene` bindless-handle block now exposes `instances` at byte offset 28. `GPUScene` remains 80 bytes.

The instance data is uploaded as a raw buffer of 112-byte `SceneInstance` records. HLSL uses `Load4` operations and shared offsets from `gpu_abi_constants.hxx`; it does not depend on an independently declared HLSL struct packing layout.

This is deliberate: raw-buffer decoding plus shared constants makes byte layout explicit across C++ and shader compilation.

### 10.2 GPU acceleration structures

The GPU path also uses two levels:

1. one BLAS per mesh asset;
2. one TLAS instance per resolved mesh attachment.

Each `RHIAccelerationStructureInstance` receives:

- the row-major 3x4 object-to-world matrix;
- the stable instance index as the 24-bit custom index;
- mask `0xff` for enabled instances or `0x00` for disabled instances;
- the device address/reference of the mesh BLAS.

Index buffers now support an explicit byte offset in RHI acceleration-structure geometry. This allows each mesh BLAS to reference its range in the shared triangle/index buffer without duplicating an index buffer per mesh.

### 10.3 Transform-only TLAS refit

RHI descriptions now carry:

- `allow_update` when the acceleration structure is created/built;
- `update` when an existing acceleration structure is refitted/updated.

The TLAS is created with update support. A transform-only scene change:

1. regenerates RHI instance descriptors in `_tlas_instance_staging`;
2. uploads the instance descriptor buffer;
3. records a TLAS update/refit command;
4. keeps all mesh BLAS objects intact.

`_tlas_instance_staging` is a renderer member and is reused across refits. It is cleared/reserved instead of allocating a fresh temporary vector every transform update.

Vulkan sets the update build flags and uses the existing TLAS as the source acceleration structure. Metal uses refit usage and `refitAccelerationStructure` when requested.

### 10.4 Scratch sizing

An update-capable TLAS may require different build and refit scratch sizes. The RHI allocation records the maximum required size so the same scratch allocation is safe for initial build and later updates.

### 10.5 24-bit instance-index limit

Hardware TLAS custom indices are 24-bit in the common instance descriptor layout. The build path guards the resolved instance count against this limit. Any future design requiring more than 16,777,215 instances must introduce an additional indirection scheme rather than silently truncating indices.

## 11. Shader-side instance behavior

### 11.1 Instance decode

Shared constants define:

- stride 112;
- object-to-world row offsets 0, 16, and 32;
- world-to-object row offsets 48, 64, and 80;
- mesh index offset 96;
- flags offset 100;
- emitter offset 104;
- emitter count 108.

`gpu_rt_shared.hlsl` decodes these values into a local shader-side representation. Any future field addition must update the shared constants and C++ assertions before shader code is changed.

### 11.2 Hit state

The hardware ray query's instance custom index becomes the renderer `instance_index`. That index is carried through:

- immediate hit processing;
- `GPUWavefrontHit`;
- `GPUWavefrontPathVertex`;
- `GPUWavefrontLightPathVertex`;
- direct-light and connection paths that reconstruct surface state.

Without this propagation, later wavefront stages would have only a global triangle index and would incorrectly shade repeated mesh assets as if they were untransformed.

### 11.3 World-space shading

At hit reconstruction:

- object-space interpolated positions are transformed by `object_to_world`;
- normals use inverse transpose through `world_to_object`;
- geometric normals include mirrored orientation correction;
- tangent frames retain handedness;
- the instance index is used for area-emitter resolution.

Both the modular active wavefront shaders and the monolithic fallback/reference shader declaration were made ABI-consistent. The active renderer pipeline compiles the modular `gpu_rt_wavefront_*` shader set.

## 12. Lights and emitters

Primary files:

- `sources/etx/render/host/emitter_packing.cxx`
- `sources/etx/render/shared/emitter.hxx`
- `sources/etx/render/shared/scene_emitters.hxx`
- `sources/etx/render/shared/scene.hxx`

### 12.1 Resource profiles versus packed emitter instances

`EmitterProfile` remains the reusable resource-level description. Packing creates render-facing `Emitter` instances after hierarchy resolution.

This matters because a single emissive mesh asset can appear under multiple transforms. Each transformed occurrence needs different world-space triangle positions, normals, area, sampling weight, and instance identity, even though it shares mesh/material resources.

### 12.2 Area emitters

For every enabled mesh instance with emissive triangles:

1. vertices are transformed to world space;
2. world-space triangle area is computed;
3. an `Emitter` record is created with global triangle index and instance index;
4. the instance's emitter records occupy a sorted contiguous range;
5. `SceneInstance::emitter_offset/count` records the range.

When a ray hits an emissive triangle, emitter lookup binary-searches only that instance's sorted range by triangle index. It does not scan every light in the scene.

### 12.3 Directional emitters

A directional emitter attached to an enabled node receives the node's rotation-only world orientation. Environment profile rotation is composed with the same orientation. This prevents nested non-uniform scale from skewing infinite-light directions; invalid orientation chains are rejected from active packing.

Unattached global directional/environment emitters retain their previous global behavior. An emitter explicitly attached to a disabled node must not reappear as an unattached global emitter; this is covered by a regression test.

### 12.4 Emitter ABI

`Emitter` remains 32 bytes. `instance_index` occupies byte offset 24, which was existing padding before this change. Therefore the stride did not grow.

This is an important performance and compatibility result: instance-aware lighting was added without increasing the emitter buffer stride.

## 13. Cameras

Cameras are resources referenced by `Camera` attachments.

The resolved camera transform applies:

- affine point transform to camera position;
- rotation-only transform and normalization to direction;
- rotation-only transform and normalization to up;
- camera rebuild from transformed basis and existing optical settings.

Multiple cameras persist through native save/load. The active camera is updated after hierarchy transform, parent, or enabled-state edits.

Camera resources are currently constrained to one effective enabled attachment when resolving an active transform. Duplicating one camera resource under multiple active nodes would be ambiguous because the existing scene API selects camera resources rather than camera instances. If true camera instancing is required later, it should receive an explicit camera-instance model rather than an arbitrary winner.

## 14. Media

Primary files:

- `sources/etx/render/shared/medium.hxx`
- `sources/etx/render/interop/medium.hxx`
- `sources/etx/render/access/medium_access_cpu.hxx`
- `sources/etx/render/access/medium_access_gpu.hxx`
- `sources/etx/render/interop/medium_sample_shared.hxx`
- `sources/etx/render/interop/medium_transmittance_shared.hxx`

Each medium now carries:

- `world_to_object`;
- `local_bounds`;
- derived world-space `bounds`.

The world-space bounds are used for ray interval intersection, so entry/exit distances remain in world ray units. Density lookup converts each world position through `world_to_object`, then maps from `local_bounds` to normalized medium coordinates.

This convention is used by CPU and GPU sampling/transmittance paths. It avoids the incorrect alternative of transforming ray distances into object space and then mixing them with world-space integration distances.

A medium resource supports one enabled transform attachment. Multiple enabled attachments for the same medium resource are rejected because the medium buffer stores one transform. If true medium instancing is needed, introduce a separate `MediumInstance` resource array rather than copying an arbitrary attachment.

The GPU medium stride is 224 bytes and is compile-time checked.

## 15. Update classification and hashing

Primary files:

- `sources/etx/render/host/scene_data.hxx`
- `sources/etx/render/host/scene_data.cxx`
- `sources/raytracer/gpu_renderer.cxx`

`SceneHashes` now independently hashes:

- hierarchy structure;
- local transform content;
- attachments;
- existing geometry, materials, emitters, media, images, options, and defaults.

Hashing uses actual content. It does not rely on mutation revision counters that could miss edits made through legacy/direct access.

Relevant update aggregation:

| Change | Consequence |
| --- | --- |
| local transform content | transform resolution, instance upload, Embree instance update, GPU TLAS refit |
| node enabled flags | effective visibility recomputation, instance-mask upload, Embree instance update, GPU TLAS refit |
| parent/topology structure | hierarchy rebuild and acceleration-structure organization rebuild |
| attachment structure | instance/emitter/resource resolution and acceleration-structure organization rebuild |
| vertex positions or triangle indices | BLAS/Embree geometry rebuild |
| normals/tangents/UVs only | shading-buffer update without geometry-structure rebuild |
| emitter/material changes | repack/update shading resources as required |

`UpdateFlags::AnyGeometryStructure` includes vertex positions, triangle-index changes, hierarchy structure, and attachments. `UpdateFlags::Transforms` is deliberately separate, enabling the refit path.

Enabled flags are part of the transform/instance-state hash rather than the hierarchy-structure hash. Visibility changes therefore update masks through the existing Embree/TLAS refit paths instead of rebuilding BLAS resources. Hierarchy resolution tracks whether derived state is current; failed validation remains dirty, and attachment-transform queries only resolve again when needed.

## 16. CPU/GPU ABI audit

The user's alignment concern was valid: adding fields to shared structures without exhaustive layout checks would be unsafe. Inspection showed that the actual modified layouts remained correctly aligned, but coverage was previously insufficient. This change hardens the contract.

### 16.1 Scene instance ABI

| Record | Stride | Alignment-sensitive additions |
| --- | ---: | --- |
| `AffineTransform` | 48 | three aligned `float4` rows |
| `SceneInstance` | 112 | two transforms plus four 32-bit values |
| `Emitter` | 32 | `instance_index` at offset 24, consuming padding |
| `Medium` | 224 | transform and local bounds included |
| `GPUScene` | 80 | `instances` handle at offset 28 |

### 16.2 Wavefront ABI

Instance fields consume existing tail/padding space; these record strides do not increase:

| Record | Stride | Instance index offset |
| --- | ---: | ---: |
| `GPUWavefrontHit` | 144 | 132 |
| `GPUWavefrontPathVertex` | 208 | 196 |
| `GPUWavefrontLightPathVertex` | 192 | 188 |

### 16.3 Protection mechanisms

`gpu_abi_constants.hxx` is the source of shared byte constants. C++ compilation verifies:

- standard-layout/trivially-copyable requirements where raw upload depends on them;
- `sizeof`;
- `alignof` where required;
- `offsetof` for every new/critical field;
- correspondence between nested matrix rows and shader byte offsets;
- unchanged buffer block sizes.

Relevant assertion files:

- `sources/etx/render/host/interop_static_asserts.cxx`
- `sources/raytracer/gpu_renderer_abi_static_asserts.hxx`

The shader decode path uses the same constants. Do not introduce literal `112`, `96`, `100`, `104`, or `108` offsets in new HLSL call sites.

### 16.4 ABI change checklist

When changing any shared record:

1. Decide whether the field is genuinely required at hit frequency.
2. Place it deliberately with alignment/padding in mind.
3. Update the shared stride/offset constants.
4. Update every C++ `sizeof`/`offsetof` assertion.
5. Update raw HLSL load/store helpers.
6. Compile every GPU integrator specialization, not only path tracing.
7. Check GPU memory impact against a measured scene baseline.

## 17. Performance characteristics

### 17.1 Deliberately efficient properties

- Nodes are 64-byte dense records.
- Attachments are 16-byte dense records.
- Rendering instances are 112-byte dense records.
- Mesh data is shared by all instances.
- There are no per-node heap objects or child vectors.
- Topology and transform traversal are iterative.
- Parent-before-child order is precomputed.
- Dirty transform subtrees are contiguous intervals.
- Disabled/singular objects preserve stable slots and use masks.
- CPU instance transforms update without rebuilding mesh scenes.
- GPU transform updates refit TLAS without rebuilding BLAS.
- TLAS staging storage is reused.
- Camera/medium attachment-validation scratch storage is reused across hierarchy resolutions.
- Medium bound and attachment scratch storage is reused across transform edits.
- Area-emitter lookup is restricted to an instance range and binary-searched.
- Hierarchy authoring data is not uploaded to GPU memory.
- Inverse matrices are computed once during resolution, not per shader hit.

### 17.2 Current cost centers

The current transform-edit path still performs some scene-wide sequential work after dirty subtree propagation:

- render-instance resolution visits mesh attachments;
- inverses and transformed bounds are regenerated for resolved instances;
- content hashes are recomputed when the renderer checks scene state;
- emitter repacking may revisit emissive instances after transform changes.

These costs are predictable and allocation-light. Do not add per-node indirection or a complex incremental dependency graph without profiling a representative high-instance scene. A likely next optimization, only if measurements justify it, would be explicit node-to-resolved-instance ranges or an instance dirty bitset derived from subtree intervals.

### 17.3 Stability versus compaction

Disabled and singular instances are not removed. This slightly increases TLAS/instance-buffer occupancy relative to aggressive compaction, but it prevents index churn and avoids rebuilding dependent light and wavefront references. For interactive edits and production correctness, stable IDs are the better default.

## 18. Editor/UI integration

Primary files:

- `sources/raytracer/ui.cxx`
- `sources/raytracer/ui.hxx`

The Scene Tree window presents the flat `evaluation_order` as an interactive hierarchy beneath a virtual `Scene` root. Expansion skips closed subtree intervals without building a recursive view model. Cameras and emitters use compact markers from the existing bundled UI font; mesh and empty nodes remain unmarked. Selection is shared with a separate Node Properties window.

Node properties expose:

- name;
- enabled state;
- parent selection that preserves the node's current world transform and rejects cycles or singular parents;
- translation, quaternion-derived Euler rotation, and scale controls when the local affine is losslessly decomposable as TRS;
- a viewport transform gizmo located at the resolved node pivot;
- direct properties for attached meshes, cameras, emitters, and media;
- explicit `Bake Transform` and `Center Pivot` geometry operations.

Geometry edits make attached mesh assets single-user before changing vertices, so other instances retain their source geometry. Baking applies the selected node's local affine to positions, normals, tangent frames, bounds, and geometric normals before resetting the local transform to identity. Centering uses the area-weighted center of the attached triangle surfaces. Both operations compensate direct child transforms to preserve descendant world placement and reject nodes that directly mix mesh attachments with cameras, emitters, or media.

After hierarchy mutations, the UI resolves hierarchy state and refreshes transformed medium bounds and the active camera.

Visibility edits are transactional: if enabling a node would create an invalid duplicate camera or medium attachment, the edit is reverted and the previously valid hierarchy is restored. Editing an active attached camera first converts the world-space editor camera back into the resource's local space, avoiding a second application of its node transform.

Sheared or otherwise non-TRS matrices are shown as non-editable affine state and remain byte-for-byte authoritative. The user must explicitly reset such a node to TRS before decomposed editing can replace it. Negative and non-uniform scale are supported.

## 19. Maintained regression tests

### 19.1 `scene_hierarchy`

File: `sources/tests/scene_hierarchy.cxx`

Twenty-six maintained tests cover hierarchy and transform behavior. In addition to the original coverage, the suite now includes world-preserving reparenting, robust frame finalization, update classification, resolution recovery, attached-medium lifecycle checks, and transactional node-geometry editing.

1. `deep_hierarchy_is_iterative` — 100,000-node chain, no recursion/stack overflow.
2. `reparent_rejects_cycles` — cycle and self-parent rejection.
3. `transform_resolution_and_instances` — composition, inverse, bounds, mirrored flag, and singular masking.
4. `attachment_ranges_remain_valid` — contiguous insertion and offset repair.
5. `visibility_is_inherited` — ancestor disable/enable behavior.
6. `mirrored_tangent_frame_handedness` — reflection-safe shading frame.
7. `disabled_attached_emitter_is_not_global` — attachment enable semantics.
8. `hierarchy_hashes_content` — transform content changes produce different hashes.
9. `attachment_removal_repairs_ranges` — removal and later range repair.
10. `transformed_medium_coordinates_and_bounds` — world/local density mapping and world-distance bounds intersection.
11. `embree_transform_only_commit` — old location stops hitting and transformed location begins hitting without a geometry rebuild.

The additional macOS-continuation coverage is:

12. `affine_trs_round_trip_and_shear_detection` — stable TRS decomposition and explicit shear rejection.
13. `affine_inverse_is_scale_aware` — robust inverses across very small and large finite scales.
14. `vcm_vertex_restores_world_tangent_frame` — world-space tangent-frame restoration for CPU/GPU connection paths.
15. `equirectangular_camera_uses_orientation` — camera profile orientation is honored.
16. `directional_and_area_emitters_use_node_transforms` — finite and infinite emitter transform semantics.
17. `environment_emitter_uses_node_rotation` — environment profile and hierarchy rotations compose correctly.
18. `active_camera_uses_node_transform` — nested non-uniform scale does not skew camera orientation and store/rebuild round-trips.
19. `reparent_preserves_world_transform` — reparenting across rotated/non-uniform parents keeps world affine state and rejects singular parents atomically.
20. `shading_frame_finalization` — opposite/degenerate normals and tangent hints recover to finite orthonormal frames with preserved handedness.
21. `enabled_state_is_an_instance_update` — visibility changes select instance refit/mask updates rather than geometry-structure rebuilds.
22. `failed_resolution_remains_dirty_and_recovers` — duplicate enabled resource attachments cannot mark invalid derived state current.
23. `attached_medium_bounds_follow_visibility_and_transform` — medium mapping/bounds follow node transforms and restore authored state when disabled.
24. `bake_node_transform_isolates_geometry_and_preserves_children` — baking creates private geometry, applies mirrored/non-uniform transforms correctly, and keeps descendants stable.
25. `center_node_pivot_preserves_geometry_and_children` — the area-weighted pivot moves while rendered vertices and descendants remain fixed.
26. `node_geometry_edits_reject_mixed_attachments` — geometry-only edits reject camera/emitter/medium mixtures without mutation.

Latest result: 26/26 passed on macOS.

### 19.2 `procedural_geometry`

File: `sources/tests/procedural_geometry.cxx`

The existing suite was extended with:

- glTF `doubleSided` material import validation;
- native hierarchy/local-affine save/load round trip;
- parent relationship persistence;
- multiple camera persistence;
- resolved transformed bounds after reload.

Latest result: 14/14 passed.

Total maintained executable-test result: 37/37 passed.

## 20. Offline render validation

All validation used non-interactive command-line rendering. One-off source scenes, EXR/PNG outputs, logs, and comparison artifacts were placed under the ignored build tree and are intentionally not committed.

### 20.1 GPU shader compilation matrix

The following GPU modes completed compile-only shader preparation:

| Mode | Compiled stages | Result |
| --- | ---: | --- |
| Path tracing | 16 | passed |
| Light tracing | 15 | passed |
| BDPT fast | 27 | passed |
| BDPT full | 34 | passed |

This matters because instance indices were added to wavefront records used beyond the basic path-tracing pipeline.

### 20.2 CPU/GPU image comparisons

Offline comparisons were rendered at 128 samples per pixel and 128x128 resolution.

| Scene/feature | Similarity | Low-frequency similarity | GPU/CPU brightness ratio |
| --- | ---: | ---: | ---: |
| repeated and mirrored glTF shared mesh | 99.932648% | 99.982361% | 1.000065 |
| transformed medium | 99.280724% | 99.804649% | 1.000221 |
| nested non-uniform transformed Cornell area light, PT | 98.101006% | 99.174225% | 1.049960 |

The repeated/mirrored scene exercises shared geometry instancing, reflections, normal/tangent handling, and CPU/GPU transform agreement. The transformed-medium scene exercises world-to-local density lookup. The transformed-area-light scene exercises world triangle area and light PDFs under nested non-uniform transforms.

### 20.3 Light tracing interpretation

For the transformed validation scene, GPU/CPU light-tracing brightness ratio was 0.615490. The untransformed Cornell baseline was 0.622018. Because the transformed case tracks the pre-existing untransformed baseline, this is evidence that the large LT CPU/GPU discrepancy is not introduced by hierarchy transforms.

Do not treat that ratio as an accepted renderer-quality target. It is a separate pre-existing CPU/GPU normalization or implementation discrepancy and should be investigated in a dedicated task with a baseline-first methodology.

### 20.4 macOS/Metal continuation

The macOS continuation used deliberately bounded jobs after an earlier large validation path made the machine unresponsive:

- full native build, including the signed `raytracer_app` bundle: passed;
- Metal path-tracing compile-only preparation: 16/16 stages passed;
- Metal BDPT-full compile-only preparation: 34/34 stages passed (including the light-path and connection-stage superset);
- actual CPU render: 8x8, 1 spp, maximum path length 2: passed;
- actual Metal render: 8x8, 1 spp, maximum path length 2, one wavefront step per frame: passed in 3 frames with readback;
- one-sample CPU/GPU smoke comparisons completed successfully; their stochastic metrics are recorded only as runtime/readback evidence, not as renderer-quality thresholds;
- GPU energy-compensation LUT parity: conductor/dielectric RGB and spectral families passed;
- material-specialized BSDF runtime harness: conductor, dielectric, plastic, and diffraction sample/evaluate/PDF/albedo operations passed with 1x1x1 dispatches;
- a bounded 1x1 OpenPBR scene check was rejected during GPU renderer preparation with the explicit CPU-fallback diagnostic, before shader compilation or rendering.

The validation scene reports a missing `image-0` asset on both CPU and GPU. This is an existing validation-asset issue rather than a transform or backend divergence.

## 21. Validation commands

Run commands from `F:\Projects\etx-tracer` in PowerShell.

### 21.1 Configure, if the existing build directory is unavailable

Use a Visual Studio generator, consistent with the project workflow:

```powershell
cmake -S . -B build/windows-implementation-vs17 -G "Visual Studio 17 2022" -A x64
```

### 21.2 Full build

```powershell
cmake --build build/windows-implementation-vs17 --config RelWithDebInfo -j 8
```

The last full RelWithDebInfo build passed. After the final TLAS staging-vector reuse adjustment, the affected `raytracer`, `scene_hierarchy`, and `procedural_geometry` targets were rebuilt successfully.

### 21.3 Maintained tests

```powershell
Push-Location bin
.\scene_hierarchy.exe
.\procedural_geometry.exe
Pop-Location
```

Some existing test paths may create local `build` or BSDF cache data relative to the process working directory. Treat those outputs as local artifacts and do not commit them.

### 21.4 GPU compile-only smoke test

```powershell
.\bin\raytracer.exe --render --scene .\bin\assets\cornellbox\cornellbox.etx.json --output .\build\windows-implementation-vs17\hierarchy-validation\shader-pt.exr --renderer gpu --integrator pt --gpu-compile-only --resolution 16x16 --samples 1
```

Repeat with the project's supported bidirectional mode selection, including light tracing, BDPT fast, and BDPT full, whenever wavefront ABI or shared shader records change.

### 21.5 Diff hygiene

```powershell
git diff --check HEAD
git status --short
```

Line-ending conversion warnings may be printed on Windows; they are not `diff --check` whitespace errors.

### 21.6 macOS bounded validation

Build and maintained tests:

```bash
cmake --build build -j 6
./bin/scene_hierarchy
./bin/procedural_geometry
```

When exercising Metal runtime rendering, begin with an 8x8, 1-spp job and always bound wavefront progress with `--gpu-wavefront-steps-per-frame 1`. Compile-only checks should precede actual dispatch whenever shader or ABI code changed.

## 22. Known boundaries and follow-up candidates

### 22.1 Metal runtime status

The Metal RHI implements and has now runtime-validated:

- update-capable TLAS sizing;
- refit usage flags;
- maximum build/refit scratch sizing;
- transform conversion;
- refit command encoding;
- user-ID TLAS descriptors backed by Metal-owned translated storage;
- bounded path-tracing execution and readback on Apple M2 Pro.

Keep actual GPU validation jobs small until their compilation and dispatch behavior is known. A 1x1x1 BSDF validation dispatch is safe, but a monolithic all-material/all-operation shader can overwhelm DXC or Apple's Metal compiler before dispatch.

### 22.2 Light-tracing CPU/GPU baseline gap

The approximately 0.62 GPU/CPU brightness ratio exists in the untransformed baseline. Investigate separately. Start by locking scene, sample count, strategy flags, spectral mode, and normalization, then compare individual path contributions. Do not fold an empirical multiplier into transform or emitter code.

### 22.3 Environment-only GPU scenes

The current GPU acceleration-structure builder may still require at least one renderable mesh instance even when an environment emitter alone could conceptually render. This is outside the hierarchy change. If environment-only rendering is required, handle it as a separate AS/renderer lifecycle feature.

### 22.4 Single-transform camera and medium resources

Meshes support many instances. Camera and medium resources currently resolve one enabled attachment each. True camera/medium instancing requires separate render-instance records and API semantics.

### 22.5 General DAGs

The hierarchy is a forest, not a multi-parent DAG. glTF duplicate references keep the first parent. Supporting a DAG would require either logical node duplication at import or a separate instance-edge model; do not add multiple parent arrays to `SceneNode`.

### 22.6 Transform-update granularity

World propagation is dirty-subtree incremental; instance resolution/hashing remains broader. Profile before extending incrementality. Representative measurements should include:

- at least 100k nodes with sparse transform edits;
- many nodes with no attachments;
- many instances of a few shared meshes;
- emissive and non-emissive mixes;
- CPU commit time, GPU upload time, TLAS refit time, and total frame stall;
- memory overhead of any new reverse mapping.

### 22.7 Animation and motion blur

There is no keyframe or motion-transform system in this change. A future animation layer should write local transforms through `set_local_transform` and batch resolution once per update. Motion blur would require explicit time-sampled transforms in Embree/RHI and is not implied by the current static affine ABI.

### 22.8 OpenPBR GPU boundary

OpenPBR is fully exercised by the CPU BSDF suite, but its generic mixed-component GPU functions remain intentionally excluded from production wavefront stages. Even after separating validation by operation and material kind, DXC/SPIR-V legalization or the Apple Metal compiler expands those kernels pathologically. The GPU renderer now rejects any scene containing OpenPBR during initialization or pipeline reconfiguration and tells the user to select the CPU renderer; batch mode preserves that specific reason instead of reporting a generic preparation failure. This prevents silent path loss and avoids starting the pathological shader compile. Do not claim OpenPBR GPU parity until the production path is split into smaller shader stages. The bounded GPU runtime harness validates only material classes enabled by the production wavefront renderer.

## 23. File map by subsystem

### 23.1 Hierarchy and host scene state

- `sources/etx/render/host/scene_hierarchy.hxx`
- `sources/etx/render/host/scene_hierarchy.cxx`
- `sources/etx/render/host/scene_data.hxx`
- `sources/etx/render/host/scene_data.cxx`
- `sources/etx/render/host/scene.cxx`
- `sources/etx/render/host/scene_representation.cxx`

### 23.2 Import and persistence

- `sources/etx/render/host/scene_gltf_loader.cxx`
- `sources/etx/render/host/scene_tungsten_loader.cxx`
- `sources/etx/render/host/scene_serialization.hxx`
- `sources/etx/render/host/scene_serialization.cxx`

### 23.3 Shared render/ABI data

- `sources/etx/render/interop/math_shared.hxx`
- `sources/etx/render/interop/gpu_abi_constants.hxx`
- `sources/etx/render/interop/gpu_scene_shared.hxx`
- `sources/etx/render/interop/gpu_wavefront_abi.hxx`
- `sources/etx/render/interop/gpu_wavefront_shared.hxx`
- `sources/etx/render/shared/scene.hxx`
- `sources/etx/render/shared/emitter.hxx`
- `sources/etx/render/shared/medium.hxx`

### 23.4 ABI verification

- `sources/etx/render/host/interop_static_asserts.cxx`
- `sources/raytracer/gpu_renderer_abi_static_asserts.hxx`

### 23.5 Emitters and media

- `sources/etx/render/host/emitter_packing.hxx`
- `sources/etx/render/host/emitter_packing.cxx`
- `sources/etx/render/host/medium_pool.cxx`
- `sources/etx/render/shared/scene_emitters.hxx`
- `sources/etx/render/shared/scene_medium.hxx`
- `sources/etx/render/access/medium_access_cpu.hxx`
- `sources/etx/render/access/medium_access_gpu.hxx`
- `sources/etx/render/access/medium_access_shared.hxx`
- `sources/etx/render/interop/medium.hxx`
- `sources/etx/render/interop/medium_sample_shared.hxx`
- `sources/etx/render/interop/medium_transmittance_shared.hxx`

### 23.6 CPU ray tracing and integrators

- `sources/etx/rt/rt.cxx`
- `sources/etx/rt/integrators/integrator.cxx`
- `sources/etx/rt/integrators/bidirectional.cxx`
- `sources/etx/rt/shared/path_tracing_shared.hxx`
- `sources/etx/rt/shared/vcm_shared.hxx`

### 23.7 GPU renderer and shaders

- `sources/raytracer/gpu_renderer.hxx`
- `sources/raytracer/gpu_renderer.cxx`
- `sources/etx/render/shaders/gpu_rt.hlsl`
- `sources/etx/render/shaders/gpu_rt_preview_trace.hlsl`
- `sources/etx/render/shaders/gpu_rt_shared.hlsl`
- `sources/etx/render/shaders/gpu_rt_wavefront.hlsl`
- modified modular `gpu_rt_wavefront_*` shader files.

### 23.8 RHI acceleration structures

- `sources/etx/rhi/rhi_types.hxx`
- `sources/etx/rhi/vulkan/vk_device.cxx`
- `sources/etx/rhi/vulkan/vk_rhi.cxx`
- `sources/etx/rhi/metal/mt_rhi.mm`

### 23.9 UI and tests

- `sources/raytracer/ui.hxx`
- `sources/raytracer/ui.cxx`
- `sources/tests/CMakeLists.txt`
- `sources/tests/scene_hierarchy.cxx`
- `sources/tests/procedural_geometry.cxx`

## 24. How to extend the system safely

### 24.1 Add a mesh instance programmatically

1. Ensure the mesh asset exists in `SceneData::meshes` and references shared geometry ranges.
2. Add or select a hierarchy node.
3. Add a `Mesh` attachment with the mesh resource index.
4. Call `SceneData::resolve_hierarchy()` once after the mutation batch.
5. Let normal scene hashing/commit code classify the change.

Do not append directly to render-facing `SceneInstance` arrays; those are derived and repacked.

### 24.2 Update a transform

1. Call `set_local_transform`.
2. Resolve hierarchy once after all edits for the frame/batch.
3. Refresh dependent medium bounds and camera state if operating below the normal scene-representation UI path.
4. Allow the renderer's hash comparison to select Embree update/TLAS refit.

Do not bake the new matrix into vertices for an ordinary object transform. Use the explicit node-geometry edit API only when the user requests a destructive `Bake Transform` operation; it isolates shared mesh assets and compensates child transforms before changing geometry.

### 24.3 Add a new attachment type

Before extending `SceneAttachment::Type`, define:

- the resource array it indexes;
- whether the resource can be attached more than once;
- whether it produces one or many render instances;
- its disabled/singular semantics;
- serialization string and validation;
- import behavior;
- hashing behavior;
- CPU render behavior;
- GPU data requirements.

Only upload a derived instance record if rendering needs it. Do not upload generic attachment records by default.

### 24.4 Add decomposed transform editing

Keep `AffineTransform` authoritative. Decomposition should be an editor operation, not the stored runtime representation. Account for:

- negative scale/reflection;
- non-uniform scale;
- shear;
- quaternion sign ambiguity;
- decomposition failure near singular matrices.

If a matrix cannot be losslessly represented as TRS, retain matrix-row editing or require an explicit user-confirmed conversion.

## 25. Production review checklist

Before merging or extending this branch, verify:

- [ ] Full RelWithDebInfo Visual Studio build passes.
- [ ] `scene_hierarchy` reports 23/23.
- [ ] `procedural_geometry` reports 14/14.
- [ ] PT, LT, BDPT-fast, and BDPT-full compile-only shader modes pass.
- [ ] No literal replacement for shared GPU ABI offsets was introduced.
- [ ] `SceneInstance`, `Emitter`, `Medium`, `GPUScene`, and wavefront static assertions pass.
- [ ] Transform-only edits do not recreate mesh BLAS.
- [ ] Disabled/singular instances retain stable indices and zero masks.
- [ ] Mirrored and non-uniform transforms are tested on visible and emissive meshes.
- [ ] Camera and medium attachment behavior remains unambiguous.
- [ ] Native version-1 geometry still loads.
- [ ] Native hierarchy/cameras save and reload.
- [ ] No one-off scene, image, executable, comparison report, or cache is staged.
- [ ] Metal compile-only and a bounded 8x8 runtime smoke test pass before claiming Apple runtime support.
- [ ] Any performance claim includes a measured baseline.

## 26. Workspace and artifact policy

The ignored local validation directory is:

```text
build/windows-implementation-vs17/hierarchy-validation
```

It may contain generated scenes, shader logs, offline renders, and comparisons. These are evidence used during development, not maintained project assets.

The following unrelated untracked directories were present and intentionally excluded from the transform commit:

```text
thirdparty/libdatachannel/
thirdparty/mbedtls/
```

Do not infer that these directories belong to this implementation. Preserve or handle them according to their owner's separate work.

## 27. Recommended continuation sequence

When resuming work:

1. Check out `transform`.
2. Read this document and inspect `scene_hierarchy.hxx/.cxx` first.
3. Run the two maintained test executables.
4. Run a full RelWithDebInfo build.
5. If modifying ABI or shader transport, compile all four GPU mode families.
6. If modifying performance behavior, capture CPU commit, GPU upload, and TLAS timing before changing code.
7. If preparing a merge, repeat the bounded Metal compile/runtime smoke checks after any RHI, ABI, or shader change.
8. Keep local validation assets under the build tree and out of commits.

Useful Git commands:

```powershell
git switch transform
git log -1 --oneline
git status --short
git show --stat --oneline HEAD
```

## 28. Final implementation state

At handover time:

- the production implementation is complete across import, persistence, host hierarchy, CPU rendering, GPU rendering, RHI, shaders, emitters, media, cameras, UI, and maintained tests;
- the hierarchy is flat and iterative;
- render geometry is instanced rather than transform-baked;
- CPU and GPU transform-only acceleration updates are implemented;
- GPU layout alignment and offsets are explicitly protected;
- 37/37 maintained tests pass;
- all tested GPU shader mode families compile;
- representative offline CPU/GPU transform comparisons are close for PT and transformed media;
- the known LT CPU/GPU gap is baseline-confirmed as pre-existing;
- Metal path tracing, readback, BSDF LUT parity, and production-enabled BSDF runtime operations have bounded runtime coverage;
- generic OpenPBR GPU wavefront sampling remains an explicit pre-existing compiler/architecture boundary;
- one-off validation artifacts and unrelated third-party directories are not part of the commit.
