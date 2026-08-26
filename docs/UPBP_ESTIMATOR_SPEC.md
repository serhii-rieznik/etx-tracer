# CPU UPBP estimator contract

## Purpose

The CPU UPBP integrator combines bidirectional path sampling with surface and
volumetric density estimators. It is a reference implementation for ETX and
must preserve the renderer's RGB and spectral transport semantics. Guiding and
GPU execution are outside this integrator's scope.

The production estimator contains all of the following techniques:

- bidirectional path tracing (BPT), including direct hits, next-event
  estimation, light tracing, and vertex connection;
- surface vertex merging (SURF, a two-dimensional kernel);
- point--point volumetric merging (PP3D, a three-dimensional kernel);
- point--beam estimation (PB2D, a two-dimensional kernel);
- beam--point estimation (BP2D, a two-dimensional kernel); and
- beam--beam estimation (BB1D, a one-dimensional kernel).

Every technique is individually selectable for validation. A normal UPBP
render enables the complete set and combines all applicable techniques with
the balance heuristic.

## Event model

A transport path contains physical vertices and segment sampling events.
Physical vertices are camera and emitter endpoints, surface interactions, and
real medium-scattering interactions. Segment events are medium boundaries and
null collisions.

An event has one of these kinds:

- `Escape`: no majorant collision occurs before the end of the interval;
- `Scatter`: an extinction-sampled real collision is represented as scattering
  with a spectral albedo weight;
- `Absorb`: a real collision in a purely absorbing medium terminates the
  subpath;
- `Null`: a null collision preserves direction and continues the segment; or
- `Boundary`: the active medium changes without a scattering interaction.

Only `Scatter` and non-delta surface interactions are eligible density
estimator endpoints. Null collisions and boundaries:

- do not increment physical path depth;
- do not sample a phase function or BSDF;
- are not stored in point or beam acceleration structures;
- are not connection endpoints; and
- do contribute to spectral throughput and forward/reverse sampling density.

A physical beam is the portion of a path edge within one medium, bounded by a
physical vertex or medium boundary. Null collisions do not split the beam. The
interior of every non-zero-length physical beam is eligible for BP2D and BB1D,
including beams whose far endpoint is a surface, boundary, absorption, or
escape event.

## Medium coefficients and majorants

At position `x` and active spectral query, ETX evaluates non-negative
coefficients

```
sigma_t(x) = sigma_s(x) + sigma_a(x)
```

component-wise. Heterogeneous ETX density is constrained to `[0, 1]`, so a
valid scalar majorant for an interval is

```
mu_bar = max_component(base_sigma_t) * density_majorant
```

where `density_majorant` is one for the current texture and procedural density
representations. A future localized majorant may replace this scalar without
changing the event contract.

For every component `j`, the null coefficient is

```
sigma_n[j](x) = mu_bar - sigma_t[j](x).
```

The tracker rejects non-finite or negative coefficients and any sampled density
outside its guaranteed majorant. Runtime code must not clamp an invalid
majorant into apparent validity.

## Spectral null tracking

The tracker uses one proposal component in spectral mode and a uniform mixture
of the three integrated RGB components in RGB mode. The mixture is independent
of path throughput, which makes its forward and reverse probability locally
evaluable.

For each majorant candidate at distance `t`:

1. sample the free-flight distance from `mu_bar`;
2. choose proposal component `c` uniformly from the active components;
3. classify the candidate as real or null using

```
p_real(c) = sigma_t[c](x) / mu_bar
p_null(c) = sigma_n[c](x) / mu_bar;
```

4. retain the marginal event coefficient

```
sigma_hat_t(x) = average_component(sigma_t(x));
```

5. multiply spectral throughput by

```
sigma_s(x) / sigma_hat_t(x)
```

for a sampled real event, or by `sigma_n(x) / average_component(sigma_n(x))`
for a sampled null event. A real event with zero scattering terminates as
`Absorb`.

The scalar proposal density of a candidate event is

```
p_real(t) = exp(-mu_bar * t) * sigma_hat_t(x),
```

and the probability of escaping an interval of length `L` is

```
p_escape(L) = exp(-mu_bar * L).
```

The event record stores the scalar forward and reverse proposal densities and
the spectral numerator separately. Products are accumulated in a numerically
stable representation. Russian roulette, when applied to a null chain, is an
explicit sampling decision and its survival probability is part of the stored
density.

## Path measures and densities

Each vertex records forward and reverse density in the measure appropriate to
that vertex:

- solid angle at sampled directions;
- surface area at surface vertices; and
- volume at real medium vertices.

The standard geometry conversion is applied only when measures change across
an edge. A medium vertex has no surface cosine. A surface vertex uses the
geometric normal for measure conversion and the shading frame only for BSDF
evaluation. Delta events have zero alternative-technique density.

ETX camera directional PDFs are normalized over the complete film, rather
than expressed conditionally per pixel. The camera endpoint recurrence
therefore starts with the inverse camera directional PDF directly. It does not
contain a pixel-count or light-subpath-count multiplier. Conversely, the
light-tracing MIS ratio uses the complete-film camera area density directly;
the light-tracing contribution uses the same per-iteration scale as ETX BDPT
and VCM when `N_L = N_C`. When the retained light population is smaller,
projected splats are scaled by `N_C / N_L` before they are accumulated into
the iteration buffer. No additional division by `N_L` is applied.

Each physical segment records:

- ordered active-medium intervals;
- physical distance;
- spectral null-tracking weight;
- scalar forward and reverse null-path density;
- the number of transient null events; and
- the medium at each beam portion.

The probability of an augmented path includes all stored null-event and escape
densities. Alternative BPT and density-estimation techniques evaluate the same
augmented path; hidden transmittance randomness is not omitted from MIS.

## Estimator definitions

Let `N_L` be the number of light subpaths generated in the iteration and `r_j`
the current radius of technique `j`. Every kernel is normalized to integrate
to one in its dimensional measure.

### BPT

BPT uses every valid `(s, t)` connection strategy permitted by path depth and
delta-event constraints. Visibility segments are evaluated with explicit null
tracking. Emitter selection, endpoint, BSDF, phase, wavelength, and null-event
probabilities are included in forward and reverse densities.

### SURF

A non-delta camera surface vertex merges with compatible non-delta light
surface vertices within radius `r_surf`. Compatibility includes material-side,
geometric-normal, and medium-boundary constraints. Its density factor is

```
eta_surf = N_L * pi * r_surf^2.
```

### PP3D

A real camera medium vertex merges with real light medium vertices in the same
medium within radius `r_pp3d`. The isotropic top-hat support volume is

```
V = 4/3 * pi * r_pp3d^3
eta_pp3d = N_L * V.
```

The estimator evaluates the phase function between the camera and light path
directions and the scattering coefficient at the camera merge point. Because
ETX samples real scatter events directly, both stored endpoint throughputs
contain their terminal collision coefficient divided by its scalar event
density. PP3D converts both endpoints back to pre-collision throughput before
evaluating the single physical collision represented by the merge.

### PB2D

A long camera beam queries real light medium points. A point contributes when
its perpendicular distance to the camera beam is within `r_pb2d`, the projected
position lies inside the beam interval, and both belong to the same medium.
The transverse support area is

```
A = pi * r_pb2d^2
eta_pb2d = N_L * A.
```

Attenuation from the camera-beam origin to the projected point is evaluated as
an explicit null path. The stored light-point throughput is converted back to
pre-collision throughput before the scattering coefficient is evaluated at the
projected point.

### BP2D

A real camera medium vertex queries short light beams in the same medium. A
beam contributes when the camera point is inside its transverse support and
the projected position lies inside the beam interval. The kernel and density
factor are two-dimensional:

```
eta_bp2d = N_L * pi * r_bp2d^2.
```

Attenuation along the selected portion of the light beam is evaluated as an
explicit null path. The stored camera-point throughput is converted back to
pre-collision throughput before the scattering coefficient is evaluated at the
camera point.

### BB1D

A long camera beam queries short light beams in the same medium. A pair
contributes when their closest points lie inside both intervals and their
separation is within `r_bb1d`. The one-dimensional density factor includes the
beam-selection probability `p_beam`:

```
eta_bb1d = N_L * p_beam * C_bb * r_bb1d,
```

with `C_bb = 1/2`. For perpendicular separation `h` and beam angle `theta`, the
Epanechnikov contribution kernel is

```
K_bb1d(h, theta) = 3 / (4 * r_bb1d * sin(theta))
                   * (1 - h^2 / r_bb1d^2).
```

The technique is applicable only when both closest points are internal to
their beams and `sin(theta) > 0`. Exactly parallel beams have zero BB1D
technique density and are handled by the other UPBP techniques. An empirical
angular clamp is not permitted; numerical evaluation uses a stable geometric
predicate whose rejection threshold represents floating-point degeneracy, not
an estimator parameter.

## Multiple importance sampling

UPBP uses the balance heuristic over all enabled techniques capable of
generating the augmented path:

```
w_i(x) = n_i p_i(x) / sum_k(n_k p_k(x)).
```

Density-estimation MIS probabilities include the corresponding support-measure
factor `eta`. The normalized reconstruction-kernel profile affects the
estimator contribution, while `eta` uses the support measure, as in the UPBP
recurrence; changing between top-hat and Epanechnikov therefore does not alter
the technique-selection probability. BPT probabilities include all endpoint,
directional, free-flight, event, and null-chain probabilities. Technique
selection and beam-selection probabilities are included in both the estimator
normalization and MIS denominator.

Production evaluation uses recursive endpoint quantities so a weight is
computed from the two joined subpath ends. Maintained validation also contains
an exhaustive evaluator that enumerates all applicable strategies for the same
path. Recursive and exhaustive weights must agree before an estimator is
enabled in the integrator.

An unweighted render may enable exactly one technique. Enabling more than one
UPBP technique requires the scene's multiple-importance-sampling option;
invalid combinations fail before iteration storage is allocated.

## Progressive radii

For iteration `i`, technique dimensionality `d`, initial radius `r_0`, and
reduction parameter `alpha`, the radius is

```
r_i = r_0 * (i + 1)^((alpha - 1) / d).
```

`d` is two for SURF, PB2D, and BP2D; three for PP3D; and one for BB1D. The
consistent default is `alpha = 0.75`. `alpha = 1` is permitted only as an
explicit fixed-radius mode and is reported as a finite-radius biased render.
Normalization always uses the actual number of generated light subpaths.

## Sampling independence and determinism

Random streams are domain-separated by render seed, iteration, pixel or light
path index, physical depth, segment index, and purpose. At minimum, separate
domains exist for:

- camera and light subpath construction;
- camera and light Russian-roulette continuation;
- camera and light medium-event tracking;
- connection transmittance;
- PB2D queries;
- BP2D queries; and
- BB1D queries.

Russian roulette uses the scene's configured termination depth and the same
eta-aware continuation rule as the existing CPU bidirectional and VCM
integrators. Survival scaling is applied to outgoing throughput before a
departing segment is stored.

When BPT and any density technique are enabled together with MIS, all four BPT
endpoint strategy families must be enabled. This keeps the recursive
cross-technique density sum exact. BPT-only renders support each scene strategy
switch independently through exhaustive endpoint-strategy weights. Invalid
mixed configurations fail before allocating iteration resources instead of
producing a biased image.

Changing the enabled estimator mask must not change already sampled camera or
light subpaths. Acceleration structures are built in deterministic light-path
order, and parallel film contributions use deterministic per-worker storage
followed by ordered reduction.

## Resource and failure contract

UPBP chooses the retained light-subpath population independently from the
camera-subpath population. The configured light-path count is an optional
upper bound; zero selects the largest population admitted by the retained-light
storage budget and the fixed-depth structural per-path bound. Every active
pixel still receives one camera sample. When fewer light paths than camera paths are retained, each
camera path deterministically selects one retained light path uniformly. BPT
light tracing is scaled by the camera-to-light population ratio, density MIS
uses the retained population, and each automatically derived density radius is
scaled by the corresponding dimensional root of that ratio. These
normalizations preserve the estimator while allowing image resolution and
retained light-path storage to be decoupled.

The UPBP budget covers retained light paths, light-derived primitives, and
their spatial indices. Scene, film, common renderer, and transient camera-worker
allocations remain part of the process footprint but are outside this budget.
UPBP preflights fixed outer path-container storage against the configured
retained-light budget. Light paths reserve no worst-case depth capacity: worker threads
grow each path to its sampled length and atomically charge its published vertex,
segment, null-chain, recursive-weight, and light-splat capacity against the
budget. Exact point and beam counts and conservative spatial-index capacities
are preflighted at deterministic stage boundaries before their allocation
phases. Transient
camera-path storage uses a small per-worker reserve and is not retained across
pixels. Worker allocation and platform container-limit failures are caught and
stop the render with a specific diagnostic. The renderer reports the retained
light-path count and required and configured storage when a budget checkpoint
is exceeded. Retained light storage is released after a failed render, an
immediate stop, an integrator switch, or a restart.

Physical path length limits only physical vertices. Boundary traversal has a
separate validity limit. Null tracking is not silently truncated; an invalid
majorant or exceeded numerical safety limit fails the render with the medium,
segment, and event count identified.

Surface arrivals must have finite projected measure against both the shading
and geometric normals. A numerically grazing arrival is a zero-contribution
sample: the surface vertex is rejected while the preceding terminal medium
segment remains available to beam estimators. A generated surface direction
must also have positive projected measure and must stay on the same geometric
and shading-normal side for reflection or cross both sides for transmission;
an invalid continuation terminates at the otherwise valid surface vertex.
Invalid finite-probability path records and non-finite MIS recurrences remain
render failures.

Rays originating at sampled medium events use a zero geometric near distance;
there is no source surface to self-intersect, and a positive epsilon would skip
a real boundary when the event lies within that epsilon of it. Subsurface-entry
rays also use zero after their origin has been offset into the closed object.
Other surface-origin rays retain the renderer's offset and near-distance
policy. If the primary
medium-origin query misses, a deterministic retry retreats the origin by the
renderer epsilon, excludes the retreated interval, and subtracts that exact
distance from the reported hit. This recovers watertight boundary hits near
triangle edges without changing the sampled event or geometric endpoint.

## Product integration contract

- `Integrator::Type::UPBP` is appended after `VCM`; existing numeric values do
  not change.
- The persistent identifier is `upbp` and the display name is `UPBP`.
- CPU UI, scene serialization, and batch selection address the same options.
- Selecting UPBP on GPU returns an unsupported-integrator error. There is no
  fallback to VCM or BDPT.
- RGB and full-spectral modes are supported before the integrator is exposed.
- All existing surface materials, phase functions, local medium boundaries,
  dielectric enclosures, area lights, directional lights, and environment
  emitters retain their current physical semantics.
- Medium emission remains unsupported because it is not part of the current
  ETX medium model.

## Release evidence

The integrator is complete only after all of the following are available:

- analytic and statistical null-tracking validation;
- recursive-versus-exhaustive MIS validation;
- accelerated-versus-brute-force point and beam validation;
- isolated validation of every estimator technique;
- no-medium agreement with BDPT Full and surface VCM;
- homogeneous comparison with the published UPBP reference implementation;
- heterogeneous convergence to an independent high-sample path reference;
- RGB and spectral render coverage;
- null-boundary and dielectric-enclosed medium coverage;
- equal-time, equal-memory comparison against VCM on the locked medium suite;
- Release builds of `raytracer` and `raytracer_shader_package`; and
- representative application, cancellation, restart, serialization, and batch
  runtime smoke tests.
