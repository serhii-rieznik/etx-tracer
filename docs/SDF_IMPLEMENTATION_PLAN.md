# GPU SDF mesh baking

3 October 2026. Target: responsive GPU baking with declared world-space accuracy. The application uses CPU baking and native triangles. GPU meshes are diagnostic and never published.

## Current implementation

Interval-pruned CSG and surface-band specialization feed adaptive conforming tetrahedral extraction. Faces, hanging edge points, point/edge deduplication, samples, roots and triangle emission now run on the GPU. Root orientation is independent of parallel IDs. Regional grouping keeps evaluation coherent. CPU topology maps and sorting were removed from the GPU path; classification/subdivision control and mesh readback remain on the host.

Complete exported geometry matches the experimental CPU topology reference for the cavity, edited pair, edited Kakapo and 1,000 mixed primitives. This does not qualify accuracy. The cavity is recovered; original-field distance lower bounds prove tolerance violations in cavity, Kakapo and mixed meshes. Mixed output also contains collapsed faces and raw connectivity defects. Diagnostic subdivision lacks error-driven refinement of mixed-sign cells and stops at a fixed depth.

Full-bake observations are about 30 ms for a pair, 173 ms for 1,000 spheres, 347 ms for 1,000 mixed primitives and 344 ms for 10,000 spheres. External GPU load is uncontrolled. These are unqualified, full-container bakes, with transfers and waits included.

## Remaining implementation

1. **Geometry admission and refinement.** Port the accepted CPU baker's interval/sign, directional derivative, error and coverage rules. Retain triangle-to-cell ownership for feedback. Subdivide failing cells with shared face/edge decisions, including neighbors; resolve precision and zero-endpoint degeneracies. Require declared geometric error, coverage, closed oriented manifold topology and nonintersection before publication. Warn explicitly for permitted sub-tolerance feature loss. Sampling and mesh equivalence cannot approve geometry.
2. **Resident local updates.** Move classification/subdivision control onto the GPU. Stabilize grid origin/cell size; retain programs, samples, roots and mesh chunks. Invalidate old/new influence bounds and ancestor Boolean/blend dependencies, including pruned branches. Compare local and complete bakes. Reuse allocations and command buffers; retire by completion. Account for all live generations and verify hash capacity, memory limits and cancellation.
3. **Production integration.** Move qualified kernels into common shader sources and the Release package. Connect asynchronous preparation, cancellation and newest-generation publication. Preserve stochastic materials, container media, disabled SDF emission, normals and CPU/GPU/RGB/spectral estimators. Use resident meshes for GPU acceleration and host meshes for CPU rendering. Failures retain the accepted generation and report once.
4. **Qualification and removal.** Run maintained validations, development and Release/package builds, offline renderer/mode-switch/edit/failure tests and Metal validation. Remove superseded experimental extraction and adoption paths only after the replacement passes.

## Acceptance gates

- Geometry: supported primitives/CSG, Round/C2 blends, pairs, corners, cavities, thin features, grazing views and 10x/nonuniform/sheared transforms. Every published mesh meets declared error and topology rules, with explicit warnings for allowed feature loss.
- Editing: target at most 33 ms for ordinary pairs and localized 1,000-primitive edits at accepted quality. Report broad edits, complete builds and 10,000-primitive scaling separately.
- Performance: compare complete CPU/GPU bakes at matched accepted quality, including preparation and synchronization; target at least 2x dense full-bake improvement. Report per-case time, bake-process CPU RAM and device RAM after performance changes.
- Memory/failure: account for current, queued, staging and retiring allocations; verify reclamation and stale/cancelled/failed generation handling. The CLI's 2 GiB buffer cap is diagnostic, not a production scene budget.
- Integration: verify all affected runtime paths, Release shaders and Metal. Existing CPU-backed application checks do not qualify GPU baking.

Current measurements and replay artifacts: build/sdf-bake-experiment/production_evaluation.html.
