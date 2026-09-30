# Shared physical-law calculators (055–154)

These calculators evaluate physical equations from raw trajectory fields. They do
not compare a simulation with a scripted scene trajectory. All seven IDs are
registered alongside the existing calculators; the previous IDs keep their
implementations. No additional runtime package is needed beyond NumPy and SciPy
already declared by the project.

## Contracts and equations

| ID | Equation / residual | Required additional exports |
| --- | --- | --- |
| `dynamics.momentum_budget.v1` | `ΔP − integral(F_external dt) − ΔJ − ΔP_in + ΔP_out` | Domain membership weights; per-domain external forces; cumulative impulses and incoming/outgoing momentum |
| `dynamics.angular_momentum_budget.v1` | `ΔH − integral(tau_external dt) − ΔJ_angular − ΔH_in + ΔH_out` | Above plus world-space force application points, free couples, world inertia tensors and angular velocities |
| `energy.work_balance.v1` | `Δ(K + U_elastic + U_gravity) + integral(D_power dt) − integral(P_external dt) − ΔE_in + ΔE_out` | Elastic energy, independent external/dissipative powers, cumulative advected energy and interval applicability |
| `contact.coulomb_law.v1` | Stick: `|F_t| <= mu_s F_n`, `v_t ≈ 0`; slip: `F_t = −mu_k F_n v_t/|v_t|`; dissipative sign | Aligned per-contact normal/tangent forces, normals, relative velocities, coefficients and applicability |
| `material.elastic_law.v1` | Volume `P = dPsi/dF`; rod/shell `f = −dU/dx`, rod `couple = −dU/d(world_rotation)` | Deformation gradient and actual elastic stress, or reference/current geometry, topology, model data and actual nodal elastic forces/couples |
| `fluid.newtonian_stress.v1` | Grid `tau = 2 mu sym(grad(u))`; slender thread `N_v = 3 mu A dot_epsilon`; dissipative sign | Grid velocity and actual viscous stress; or centerline positions/velocities, edge areas and actual axial viscous forces |
| `porous.darcy_balance.v1` | `q = −K/mu (grad(p)−rho*g)` and `d(phi*S)/dt + div(phi*S*v_s + q) = s` | Collocated porosity, saturation, pressure, solid velocity, relative Darcy flux, permeability and volume sources |

Input role names and trajectory paths are those in each `metric.yaml`.
`contracts/shared_law_fields.py` adds the new fields to `STANDARD_FIELDS` without
redefining old fields. Shapes and SI units are part of the public contract.

### Mechanical budgets

Each row of `[G,N]` weights describes an independent control volume. Rows may
overlap (a single body and the whole system), but they are never summed together.
Sample order and count may change if all membership weights follow the samples.
Domain row identity must remain fixed. Domain weights lie in `[0,1]`; mass is
nonnegative, and every domain needs positive sampled mass at some time.

External forces include cross-domain material/contact tractions and support or
driver reactions. Internal pairs cancel only in a domain containing both sides.
Continuous forces are trapezoidally integrated; cumulative impulses must exclude
those already represented by that force integral. Cumulative incoming/outgoing
momentum includes material transfers and uses the same time origin. Closed
domains export explicit zero source arrays. An empty force-term axis is allowed
for an unforced dynamic domain, not as a quasistatic equilibrium check.

`H = sum(w * ((x × m*v) + I_world*omega))` uses the fixed origin `[0,0,0]`.
Point samples use zero intrinsic inertia and spin. Do not add intrinsic rigid
rotation if the distributed masses and velocities already resolve it. World
inertia must be symmetric positive semidefinite.

`mode: quasistatic` checks net force or torque directly; no fictitious mass is
introduced for massless films. Only equilibrated samples belong in these tasks'
quasistatic diagnostic trajectories. Transient rupture dynamics are not justified
by a zero-inertia equilibrium law.

Energy includes translational and unresolved intrinsic rotational kinetic energy,
elastic energy and `−sum(m*g dot x)`. Gravity is frozen. If gravity potential is
included, external power must omit gravity. Unknown plastic/fracture/latent/surface
energy is not silently assigned zero. Elastic energies and nonnegative dissipative
powers must be independently auditable from the chosen model; **this budget
calculator does not reconstruct those terms from their raw constituent forces**.
It must never be fed dissipation inferred from the observed energy difference.

Dynamic budget errors use the Euclidean norm (absolute value for energy), divided
by the corresponding frozen positive momentum/angular-momentum/energy scale. The
maximum over independent budget rows is returned for each adjacent interval.
Quasistatic residuals use frozen force/torque scales. These scales do not depend on
the net initial momentum, which may be zero.

### Contact, fluids and pores

Coulomb input normals are unit vectors; normal force is compressive and nonnegative,
and `0 <= mu_k <= mu_s`. Tangential forces and relative velocities use the same
action side. Sticking/sliding is classified with the frozen positive speed
tolerance. The maximum normalized force, sticking-speed and dissipative-sign
violation is returned over active contacts. Impulses and wet adhesion are outside
this sustained dry-contact law. Contact arrays can be empty before contact;
at least one valid contact must exist somewhere in the trajectory.

Grid calculations require at least three cells per axis, uniform frozen positive
spacing, and collocated physical data. Centered differences score interior
centers only. Masks certify valid complete stencils, not merely the center's
phase: a true center may read unscored boundary/ghost neighbors. The caller must
exclude stencils crossing invalid interfaces or sources. There is no implicit
periodic wrapping or boundary extrapolation. The pre-existing
`field.divergence.v1` retains its original differencing and normalization.

Newtonian grid errors use spatial RMS of the Frobenius stress residual divided by
the frozen stress scale, with an additional nonnegative-dissipation check.
The slender-thread strain rate is `((v_j-v_i) dot (x_j-x_i))/|x_j-x_i|²`.
That axial specialization does not claim to evaluate all bending/twisting viscous
stresses in a thread. Grid viscosity is positive and uniform in this v1 interface;
non-Newtonian materials need a different law.

Darcy `q` is volumetric liquid flux relative to the moving solid, not the intrinsic
liquid velocity. `K` is the effective symmetric positive-semidefinite permeability
tensor, including any saturation factor selected by the frozen porous model.
The equation uses incompressible constant-density liquid. Both `phi` and `S` lie
in `[0,1]`. Storage is differentiated between adjacent times on a fixed grid;
the source-minus-divergence term uses a trapezoidal average of interval endpoints.
Only centers active at both endpoints contribute to storage. The storage residual
is a rate (`1/s`), normalized by the positive storage-rate scale; flux residuals
use the positive velocity scale. At each active endpoint the returned error is
the maximum of flux RMS and the available adjacent storage RMS. Both components
and their valid counts remain separately visible in details. At least one valid
storage interval is required. This v1 grid has three spatial dimensions.

### Elastic model selection

See [elastic_model_registry.md](elastic_model_registry.md) for exact energies,
numeric model IDs, parameter-column order, shape constraints and differentiation.
The registry supports volume neo-Hookean/StVK/linear elasticity, a Cosserat rod,
and a StVK membrane with hinge bending. These are explicit shared models, not
aliases for every constitutive model in the cited papers. A solver must use a
matching frozen model or receive a separate, reviewed calculator extension.

Material tables and reference data are checked for consistency across the
trajectory. Volume stresses, shell membrane forces and rod translational forces
are analytical. Rod couples use the energy chain rule with local five-point
strain derivatives; shell hinges use local five-point energy derivatives. This is a
reference evaluator, not an optimized real-time implementation. Large meshes
should use a fixed, representative diagnostic subset with complete elastic
neighborhoods. Model choice and sampling must be fixed before observing errors.

## Temporal alignment and error handling

`CalculatorResult.frame_indices` is an optional, strictly increasing integer array
aligned to `normalized_errors`. It allows interval laws to report their endpoint
indices and applicability-aware laws to omit unscored frames. The evaluator
reduces only valid errors and maps `worst_frame_index` back to the trajectory.
Legacy results without indices still require exactly one error per frame.
No initial zero is prepended to an interval law and no inactive frame is scored
as zero. An energy mask at frame `i` describes `(i-1,i]`; frame zero's mask has no
preceding interval. Every energy domain requires at least one active interval.

Missing fields, malformed shapes, nonfinite data, invalid times, unsupported model
IDs, nonpositive scales, drifting frozen inputs, or an entirely unevaluable
dataset raise a measurement error. They do not count as a physical pass. Raw
normalized errors are not clipped to percentages or bounded by one.

## What still requires a solver/scaffold

The callable calculators and their analytical/interface tests can be completed
before a solver exists. Real scene execution still requires a scaffold to freeze
models, parameters, scales, domain identities and applicability rules, and an
exporter to supply the declared physical data. `StateFrame` contains no unit or
provenance metadata. Cross-frame consistency checks cannot authenticate a forged
first-frame parameter table, mask, stress, source or power against a scaffold
that was not passed to `compute`. These inputs require platform-side provenance
and exporter validation; registration is not an anti-cheating guarantee.

Analytical tests establish equation and error-handling behavior. Equilibrium
fixtures exercising all 404 YAML bindings and HDF5 round trips establish interface
compatibility. They do not establish that the 100 SIGGRAPH scenes have been
simulated, that all solver exporters are correct, or that normalization scales
have been calibrated for those scenes.
