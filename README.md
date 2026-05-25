# LiftingSurfaces.jl

Lifting-surface aerodynamics (vortex-lattice method) for the Julia
ship-CFD stack on top of [WaterLily.jl](https://github.com/pankgeorg/WaterLily.jl).
Wraps [VortexLattice.jl](https://github.com/byuflowlab/VortexLattice.jl)
with WaterLily-friendly types and the two primitives needed to couple
back into a WaterLily simulation:

- `Rudder` and `BladedRotor` — VLM-resolved rudder and propeller
- `smear_force!` — Gaussian deposition of a point force onto a
  WaterLily face-staggered force array (force-conserving)
- `trilinear_inflow` — closure factory that samples WaterLily's
  velocity field for VortexLattice's `additional_velocity` hook

The package is part of a six-package private stack:

| Package | Role |
|---|---|
| WaterLily.jl | Cartesian NS + BDIM substrate |
| Turbulence.jl | LES sub-grid (Smagorinsky, WALE) |
| VoF.jl | Free-surface (vanLeer + MULES) |
| ShipShapes.jl | Hull geometry (Wigley analytic, TabulatedHull) |
| Propellers.jl | Cheap surrogate propellers (ActuatorDisk, SwirlingDisk) |
| **LiftingSurfaces.jl** | **VLM-resolved rudder and propeller (this repo)** |

## Quick start

```julia
using LiftingSurfaces

# AR=2 rectangular flat-plate rudder, 16 spanwise × 8 chordwise panels.
rudder = Rudder(; chord=1.0, span=2.0, ns=16, nc=8)

# Solve VLM at δ=5° in a 1.0 freestream.
r = rudder_forces(rudder, deg2rad(5.0), 1.0)
@show r.CL r.CD r.CY        # CL ≈ +0.215, CD ≈ +0.007
```

```julia
# 3-blade propeller, J=0.7
rotor = BladedRotor(; N_blades=3, R=1.0, R_hub=0.2,
    chord = (0.25, 0.18),
    twist = (deg2rad(35.0), deg2rad(15.0)),
    ns = 12, nc = 4)

V∞ = 1.0
Ω  = π * V∞ / (0.7 * rotor.R)
r  = rotor_forces(rotor, V∞, Ω)
@show r.CT r.CQ r.η_VLM
```

## WaterLily coupling

```julia
using WaterLily, LiftingSurfaces
using StaticArrays: SVector

sim = Simulation((128, 64, 32), (1f0, 0f0, 0f0), 32f0; T=Float32)
rudder = Rudder(; chord=4.0, span=8.0)
δ = deg2rad(10)
rud_pos = SVector(40f0, 32f0, 16f0)

function rudder_udf(flow, t; kwargs...)
    # Two-way: sample WaterLily's local flow at panel control points.
    u_sample = trilinear_inflow(flow.u)
    inflow = (xv) -> let v = u_sample(SVector(xv[1] + rud_pos[1],
                                              xv[2] + rud_pos[2],
                                              xv[3] + rud_pos[3]))
        # VortexLattice treats this as a perturbation on Vinf, so
        # subtract Vinf=1 from the x-component.
        SVector(v[1] - 1f0, v[2], v[3])
    end
    r = rudder_forces(rudder, δ, 1f0; inflow=inflow)
    q = 0.5f0 * rudder.chord * rudder.span         # ρ=1, V=1
    F = SVector(-r.CD * q, 0f0, r.CL * q)
    smear_force!(flow.f, F, rud_pos; ε=2.0)
    return nothing
end

sim_step!(sim; udf=rudder_udf)
```

## Performance

VLM solve cost per step (single Julia 1.12 thread, median over 200 calls):

| Configuration | Panels | Time / call (ms) |
|---|---:|---:|
| Rudder, 16×8                              |   128 |   5.6 |
| BladedRotor, 3 blades, 12×4               |   144 |   9.5 |
| BladedRotor, 3 blades, 24×8               |   576 |   149 |

For a typical WaterLily step of ~1500 ms at 192×96×48 with VoF, the
low-resolution lifting-surface tier is ~1 % overhead. See
`../ShipFlow.jl/RESULTS-vlm-cost.md`.

## Limitations

VortexLattice.jl is inviscid + thin-surface + linear, so:

- No cavitation, no thick-body separation, no stall past ~15°
  without an airfoil-polar table (`nonlinear_analysis!`)
- Free-surface mirroring is approximate (no actual VoF coupling
  on the VLM side)
- Hull boundary-layer wake fraction at the propeller plane must
  come from WaterLily, not VLM

For applications where these matter, stick with `Propellers.SwirlingDisk`
or develop a thicker propeller model.

## Tests

```
$ julia +1.12 --project=. -e 'using Pkg; Pkg.test()'
```

Currently 21 tests covering: rudder polar sign + magnitude,
linear-range dCL/dα against rectangular-LLT, BladedRotor smoke,
`smear_force!` force-conservation + peak, `trilinear_inflow`
uniform + linear field recovery.
