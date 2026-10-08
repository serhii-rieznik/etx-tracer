# GPU acceleration-structure capacity

Open infrastructure task: remove the fixed 512-slot acceleration-structure limit.

`sources/etx/rhi/rhi_bindless.hxx` sets `kDefaultMaxAccelerationStructures` to 512. The GPU renderer consumes slots for mesh BLAS resources and the scene TLAS. During PBRT v4 San Miguel validation on 2026-10-07, Vulkan failed at mesh 511 with `No free slots available for resource type 3`, followed by `GPU acceleration-structure build failed`. Imported San Miguel views contain 812–837 meshes; Zero Day frames contain 9,847–9,877 meshes. Both use CPU rendering for the current comparisons. Transparent Machines was previously observed to hit this limit as well.

Size the resource capacity from the scene's required BLAS/TLAS count within device limits, or revise resource allocation so BLAS storage does not exhaust the shader-visible acceleration-structure table. Check Vulkan and Metal allocation, descriptor or argument-buffer limits, scene replacement, and resource cleanup together.

Validate San Miguel, Zero Day and Transparent Machines on the GPU, including repeated scene switching and recovery after allocation failure.
