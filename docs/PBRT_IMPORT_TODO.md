# Deferred PBRT import features

- Coated conductor BSDF: preserve the conductor, dielectric interface, thickness, and coating absorption/scattering. The importer currently uses a native conductor and reports the omitted coating.
- Arbitrary material mixtures: retain both child materials and their mask with consistent sampling, evaluation, and PDFs. The importer currently uses native plastic with the first child's base color and roughness; the second child and mixing mask are omitted.
- Cylindrical and spherical texture coordinates: preserve these mappings and the transform at texture definition. The importer currently uses mesh UVs and reports the substitution. Planar mappings are converted to native mesh UVs during import.
- GPU acceleration-structure capacity: remove the fixed 512-slot limit across allocation, binding, scene replacement, and cleanup. See [the existing infrastructure task](GPU_ACCELERATION_STRUCTURE_LIMIT.md).

Portal lighting is intentionally excluded. Portal-authored lights become ordinary environment lights; identical definitions with the same spectrum, transform, and medium are collapsed to avoid duplicating the environment.
