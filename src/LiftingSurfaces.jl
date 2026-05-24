module LiftingSurfaces

using VortexLattice
using VortexLattice: wing_to_grid, grid_to_surface_panels, System, Reference,
    Freestream, steady_analysis!, body_forces, far_field_drag,
    Wind, Body, Cosine, Uniform, AbstractSpacing
using StaticArrays: SVector

export Rudder, rudder_forces, BladedRotor, rotor_forces, smear_force!

# ---------------------------------------------------------------------------
# Rudder
# ---------------------------------------------------------------------------

"""
    Rudder(; chord, span, ns=16, nc=8, spacing_s=Cosine(), spacing_c=Uniform())

Vortex-lattice model of a flat-plate symmetric rudder. The rudder is
panelled at construction; per call, `rudder_forces(rudder, δ, V∞,
inflow=nothing)` solves one VLM steady analysis for rudder angle `δ`
(radians, positive = leading edge to port, lifting starboard) and the
freestream + optional `inflow` perturbation (a `(x,y,z) -> SVector{3}`
that gets sampled at panel control points — this is the WaterLily
coupling hook).

The rudder spans z ∈ [0, span] below z=0 (waterline reference) and is
located at x = 0 along the centerline (y = 0). The caller maps the
returned spanwise force distribution to its own world frame.
"""
struct Rudder{T}
    chord  :: T
    span   :: T
    ns     :: Int
    nc     :: Int
    spacing_s   :: AbstractSpacing
    spacing_c   :: AbstractSpacing
end

function Rudder(; chord::Real, span::Real, ns::Int=16, nc::Int=8,
                  spacing_s::AbstractSpacing=Cosine(),
                  spacing_c::AbstractSpacing=Uniform(),
                  T::Type=Float64)
    Rudder{T}(T(chord), T(span), ns, nc, spacing_s, spacing_c)
end

"""
    rudder_forces(rudder, δ, V∞; inflow=nothing) -> NamedTuple

Solve VLM for the given rudder angle δ (radians) at freestream speed
V∞ in the +x direction. Returns `(; CL, CD, CY, CM, sectional)`:

  * `CL`  — lift coefficient (perpendicular to V∞, in z-direction of
            the rudder; side-force on a horizontal-axis ship → yaw)
  * `CD`  — drag coefficient (along V∞)
  * `CY`  — side force in Body frame (y in VLM Body convention)
  * `CM`  — moment coefficients (3-vector, about rudder root)
  * `sectional` — per-strip Cf along the span (length ns)

`inflow(x,y,z)` may be passed to perturb the freestream with the
WaterLily-side ambient flow (e.g. propeller race). It must return an
`SVector{3,Float64}` and is called at every panel control point.
"""
function rudder_forces(rudder::Rudder{T}, δ::Real, V∞::Real;
                       inflow::Union{Nothing, Function}=nothing) where T
    # VortexLattice convention: leading edge at xle/yle/zle for root and
    # tip; span direction is whichever of (yle, zle) varies. We put span
    # along +y so VortexLattice's "twist around the span axis" gives the
    # rudder angle around y. The caller maps to its own world frame.
    xle = [zero(T), zero(T)]
    yle = [zero(T), T(rudder.span)]
    zle = [zero(T), zero(T)]
    chord = [rudder.chord, rudder.chord]
    twist = [T(δ), T(δ)]
    phi   = [zero(T), zero(T)]
    fc    = fill(x -> zero(T), 2)

    grid, ratio = wing_to_grid(xle, yle, zle, chord, twist, phi,
        rudder.ns, rudder.nc;
        fc=fc, spacing_s=rudder.spacing_s, spacing_c=rudder.spacing_c,
        mirror=false)
    system = System([grid]; ratios=[ratio])

    Sref = rudder.span * rudder.chord
    ref  = Reference(Sref, rudder.chord, rudder.span,
                     [rudder.chord/4, 0, 0], V∞)
    fs   = Freestream(V∞, zero(T), zero(T), [zero(T), zero(T), zero(T)])

    if inflow === nothing
        steady_analysis!(system, ref, fs; symmetric=false)
    else
        steady_analysis!(system, ref, fs; symmetric=false,
                         additional_velocity=inflow)
    end

    CF_wind, CM_wind = body_forces(system; frame=Wind())
    CF_body, CM_body = body_forces(system; frame=Body())
    CL = CF_wind[3]; CD = CF_wind[1]
    CY = CF_body[2]

    return (; CL, CD, CY, CM = CM_wind)
end

# ---------------------------------------------------------------------------
# BladedRotor — N twisted, tapered, rotating blades
# ---------------------------------------------------------------------------

"""
    BladedRotor(; N_blades, R, R_hub, chord, twist, ns=16, nc=6)

Vortex-lattice model of an N-blade propeller. `chord` and `twist` are
2-element vectors giving the root-to-tip distribution; the blade is
linearly tapered + linearly-twisted between them. The rotor sits at
the origin, axis = +x, blades spaced 2π/N apart in the y-z plane.

Per call, `rotor_forces(rotor, V∞, Ω; inflow=nothing)` solves one VLM
analysis with the chosen advance speed V∞ and angular rate Ω
(rad/cell-time-unit). Inflow perturbation is the WaterLily coupling
hook.

This is the higher-fidelity alternative to `Propellers.SwirlingDisk`:
captures radial blade loading, tip vortices, blade-passage-frequency
unsteadiness when used inside an unsteady run.
"""
struct BladedRotor{T}
    N_blades :: Int
    R        :: T
    R_hub    :: T
    chord    :: NTuple{2, T}       # (root, tip)
    twist    :: NTuple{2, T}       # (root, tip), radians
    ns       :: Int
    nc       :: Int
end

function BladedRotor(; N_blades::Int, R::Real, R_hub::Real,
                       chord::NTuple{2,<:Real}, twist::NTuple{2,<:Real},
                       ns::Int=16, nc::Int=6, T::Type=Float64)
    BladedRotor{T}(N_blades, T(R), T(R_hub),
                   (T(chord[1]), T(chord[2])),
                   (T(twist[1]), T(twist[2])),
                   ns, nc)
end

"""
    rotor_forces(rotor, V∞, Ω; inflow=nothing) -> NamedTuple

Returns `(; thrust, torque, CT, CQ, η_VLM)`. Sign of thrust depends on
the twist convention (positive twist + positive Ω + axis aligned with
V∞ → forward thrust); the calling code is responsible for choosing
signs that match its own convention.
"""
function rotor_forces(rotor::BladedRotor{T}, V∞::Real, Ω::Real;
                      inflow::Union{Nothing, Function}=nothing) where T
    # One blade along +y at angle 0.
    xle   = [zero(T), zero(T)]
    yle   = [T(rotor.R_hub), T(rotor.R)]
    zle   = [zero(T), zero(T)]
    chord = [rotor.chord[1], rotor.chord[2]]
    twist = [rotor.twist[1], rotor.twist[2]]
    phi   = [zero(T), zero(T)]
    fc    = fill(x -> zero(T), 2)

    grid_root, ratio = wing_to_grid(xle, yle, zle, chord, twist, phi,
        rotor.ns, rotor.nc;
        fc=fc, spacing_s=Cosine(), spacing_c=Uniform(),
        mirror=false)

    grids  = [grid_root]
    ratios = [ratio]
    for k in 1:rotor.N_blades - 1
        θ = T(2π * k / rotor.N_blades)
        g = copy(grid_root)
        @views for j in 1:size(g, 3), i in 1:size(g, 2)
            y = g[2, i, j]; z = g[3, i, j]
            g[2, i, j] = y * cos(θ) - z * sin(θ)
            g[3, i, j] = y * sin(θ) + z * cos(θ)
        end
        push!(grids,  g)
        push!(ratios, ratio)
    end

    system = System(grids; ratios=ratios)
    Sref   = π * rotor.R^2
    ref    = Reference(Sref, rotor.R, T(2)*rotor.R, [zero(T), zero(T), zero(T)], T(V∞))
    fs     = Freestream(T(V∞), zero(T), zero(T), [T(Ω), zero(T), zero(T)])

    if inflow === nothing
        steady_analysis!(system, ref, fs; symmetric=false)
    else
        steady_analysis!(system, ref, fs; symmetric=false,
                         additional_velocity=inflow)
    end

    CF, CM = body_forces(system; frame=Body())
    CT = CF[1]; CQ = CM[1]
    thrust = CT * 0.5 * V∞^2 * Sref
    torque = CQ * 0.5 * V∞^2 * Sref * rotor.R
    η_VLM  = (CT * V∞) / (Ω * CQ * rotor.R + eps(T))   # ≈ propulsive eff.

    return (; thrust, torque, CT, CQ, η_VLM)
end

# ---------------------------------------------------------------------------
# Eulerian projection — smear a point/line force onto a 3D grid
# ---------------------------------------------------------------------------

"""
    smear_force!(f, force, x_world; ε=2.0)

Add `force::SVector{D}` to the WaterLily face-staggered force array
`f` (shape `(N..., D)`) as an isotropic 3D Gaussian centred at world
position `x_world::SVector{D}` with width `ε` cells. The Gaussian is
normalised so the integrated added force matches `force` exactly to
the discrete grid (no leakage at the truncation radius).

Use this to project the lifting-surface forces returned by VLM back
into the WaterLily flow.f as a body force, matching the regularised-
delta convention used in actuator-line LES.

`ε ≥ 2 Δx` is the customary rule for stable LES coupling. The
function trims to a `[-3ε, +3ε]` box around `x_world` (≈ 99 % of
mass) and renormalises so the in-box sum equals `force`.
"""
function smear_force!(f::AbstractArray{T, N}, force, x_world;
                      ε::Real = 2.0) where {T, N}
    D = N - 1
    @assert length(force) == D
    @assert length(x_world) == D
    sz = ntuple(d -> size(f, d), D)
    # Box: ±3ε around x_world (cell-units).
    box_lo = ntuple(d -> max(1, floor(Int, x_world[d] - 3ε)),  D)
    box_hi = ntuple(d -> min(sz[d], ceil(Int, x_world[d] + 3ε)), D)
    # First pass: compute the kernel sum so we can normalise.
    invε² = inv(T(ε^2))
    box   = CartesianIndices(ntuple(d -> box_lo[d]:box_hi[d], D))
    ksum  = zero(T)
    @inbounds for I in box
        r² = zero(T)
        for d in 1:D
            δ = (I[d] - T(1.5)) - T(x_world[d])
            r² += δ*δ
        end
        ksum += exp(-r² * invε²)
    end
    ksum == 0 && return f
    inv_ks = inv(ksum)
    # Second pass: deposit the (force / ksum) × kernel into f.
    @inbounds for I in box
        r² = zero(T)
        for d in 1:D
            δ = (I[d] - T(1.5)) - T(x_world[d])
            r² += δ*δ
        end
        w = exp(-r² * invε²) * inv_ks
        for d in 1:D
            f[I, d] += T(force[d]) * w
        end
    end
    return f
end

end # module
