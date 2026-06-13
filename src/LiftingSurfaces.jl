module LiftingSurfaces

using VortexLattice
using VortexLattice: wing_to_grid, grid_to_surface_panels, System, Reference,
    Freestream, steady_analysis!, body_forces, far_field_drag,
    lifting_line_coefficients, lifting_line_geometry,
    Wind, Body, Cosine, Uniform, AbstractSpacing
using StaticArrays: SVector

export Rudder, rudder_forces, BladedRotor, rotor_forces, smear_force!,
       smear_torque!, smear_blades!, trilinear_inflow,
       Wing, wing_forces

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
    # Lazy cache for the VortexLattice System (same pattern as
    # BladedRotor). The rudder angle δ changes the panel geometry each
    # call, so the System's *storage* is reused (AIC, Γ, panel arrays)
    # while the surface panels are rewritten in place via
    # `update_surface_panels!` — the influence matrix is recomputed by
    # `steady_analysis!` (its default), so results are unchanged.
    _system_cache :: Base.RefValue{Any}
end

function Rudder(; chord::Real, span::Real, ns::Int=16, nc::Int=8,
                  spacing_s::AbstractSpacing=Cosine(),
                  spacing_c::AbstractSpacing=Uniform(),
                  T::Type=Float64)
    Rudder{T}(T(chord), T(span), ns, nc, spacing_s, spacing_c,
              Base.RefValue{Any}(nothing))
end

"""
    rudder_forces(rudder, δ, V∞; inflow=nothing) -> NamedTuple

Solve VLM for the given rudder angle δ (radians) at freestream speed
V∞ in the +x direction. Returns `(; CL, CD, CY, CM)`:

  * `CL`  — lift coefficient (perpendicular to V∞, in z-direction of
            the rudder; side-force on a horizontal-axis ship → yaw)
  * `CD`  — drag coefficient (along V∞)
  * `CY`  — side force in Body frame (y in VLM Body convention)
  * `CM`  — moment coefficients (3-vector, about the quarter-chord
            reference position set in `Reference(...)`)

All coefficients are normalised to V∞ and reference area `chord·span`,
even when an `inflow` perturbation is supplied — VortexLattice absorbs
the inflow into the circulation distribution but keeps the
normalisation against the freestream reference.

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
    if rudder._system_cache[] === nothing
        rudder._system_cache[] = System([grid]; ratios=[ratio])
    else
        # Reuse the System storage. `steady_analysis!` regenerates the
        # surface panels from `system.grids` on every call
        # (analyses.jl: update_surface_panels!(surfaces[i], grids[i])),
        # so copying the δ-dependent grid in is all an update needs.
        sys = rudder._system_cache[]
        sys.grids[1] .= grid
        sys.ratios[1] .= ratio
    end
    system = rudder._system_cache[]

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
# Wing — a generalized finite wing (taper / sweep / twist / dihedral)
# ---------------------------------------------------------------------------

"""
    Wing(; chord_root, chord_tip, span,
           sweep=0.0, twist_root=0.0, twist_tip=0.0, dihedral=0.0,
           ns=20, nc=8, spacing_s=Cosine(), spacing_c=Uniform())

Vortex-lattice model of a flat (zero-camber) finite wing — the
generalization of [`Rudder`](@ref) to taper, quarter-chord sweep,
linear geometric twist (washout) and dihedral. It is the lifting-device
backend for a rigid sail (8401 Q4) and any finite-wing study.

This is a **full free-ended wing** of geometric span `span`, panelled
directly over `yle ∈ [0, span]` (`mirror=false`, no symmetry plane —
same convention as `Rudder`). Both ends are free tips, so the spanwise
loading is symmetric and peaks at mid-span (verified: an elliptic-like
distribution falling to zero at both `y=0` and `y=span`). The reference
area is the trapezoidal planform `S = (chord_root+chord_tip)/2 · span`
and the aspect ratio reported by [`wing_forces`](@ref) is the geometric
`AR = span²/S` of this whole wing — *not* a half-wing reflected about a
root plane. Lifting-line slope comparisons (`dCL/dα → 2π·AR/(AR+2)`)
therefore use this AR directly, and the validation confirms VLM lands
~86–92 % of the lifting-line slope, rising toward it as AR grows.

Angles in **degrees**. `sweep`/`dihedral` are applied to the tip
leading-edge offset; per-station geometric twist is linear from
`twist_root` (root) to `twist_tip` (tip). The angle of attack `α` is
added to the twist at solve time by [`wing_forces`](@ref).

Geometry → the five VortexLattice `wing_to_grid` arrays:

```
xle = [0, span·tand(sweep)]      yle = [0, span]
zle = [0, span·tand(dihedral)]   chord = [chord_root, chord_tip]
twist = [α+twist_root, α+twist_tip]  (deg→rad)   phi = [0, 0]
```

The `System` storage is cached (same pattern as `Rudder`): geometry
changes per `α`, so the new grid is copied into `system.grids[1]` and
`steady_analysis!` regenerates the panels — results are bit-identical to
a freshly built `System`.
"""
struct Wing{T}
    chord_root :: T
    chord_tip  :: T
    span       :: T
    sweep      :: T        # deg, quarter-chord
    twist_root :: T        # deg
    twist_tip  :: T        # deg
    dihedral   :: T        # deg
    ns         :: Int
    nc         :: Int
    spacing_s  :: AbstractSpacing
    spacing_c  :: AbstractSpacing
    _system_cache :: Base.RefValue{Any}
end

function Wing(; chord_root::Real, chord_tip::Real, span::Real,
                sweep::Real=0.0, twist_root::Real=0.0, twist_tip::Real=0.0,
                dihedral::Real=0.0, ns::Int=20, nc::Int=8,
                spacing_s::AbstractSpacing=Cosine(),
                spacing_c::AbstractSpacing=Uniform(), T::Type=Float64)
    Wing{T}(T(chord_root), T(chord_tip), T(span), T(sweep),
            T(twist_root), T(twist_tip), T(dihedral), ns, nc,
            spacing_s, spacing_c, Base.RefValue{Any}(nothing))
end

"Reference planform area / aspect ratio for a `Wing` (trapezoidal, cantilever)."
function wing_reference(w::Wing{T}) where T
    S  = (w.chord_root + w.chord_tip) / 2 * w.span
    AR = w.span^2 / S
    cref = (w.chord_root + w.chord_tip) / 2     # mean aerodynamic-ish chord
    return (; S, AR, cref)
end

"""
    wing_forces(wing, α, V∞; inflow=nothing)
        -> (; CL, CDi, CD, CM, CY, cl_span, y_span, AR, e)

Solve one VLM steady analysis for the [`Wing`](@ref) at angle of attack
`α` (**radians**) and freestream speed `V∞` (in +x). Returns:

  * `CL`  — lift coefficient (Wind frame z), normalised to the
            trapezoidal `S` and `V∞`.
  * `CDi` — **induced** drag from the Trefftz plane (`far_field_drag`),
            the quantity that should follow `CL²/(π·e·AR)`.
  * `CD`  — near-field total drag (Wind frame x) — for a flat plate this
            is essentially the induced drag plus numerical leading-edge
            suction error; `CDi` is the reference-grade value.
  * `CM`  — moment-coefficient 3-vector (Wind frame) about the
            quarter-root-chord reference point.
  * `CY`  — side-force coefficient (Body frame y).
  * `cl_span`, `y_span` — sectional lift coefficient and spanwise station
            (`y/ (span)` normalised? no — physical y of each segment
            midpoint), for plotting the spanwise loading.
  * `AR`  — the cantilever aspect ratio `span²/S`.
  * `e`   — implied span efficiency `CL²/(π·AR·CDi)` (NaN if CL≈0).

`inflow(x,y,z)->SVector{3}` perturbs the freestream (WaterLily coupling
hook), identical semantics to `rudder_forces`.
"""
function wing_forces(wing::Wing{T}, α::Real, V∞::Real;
                     inflow::Union{Nothing,Function}=nothing) where T
    αT = T(α)
    tr = αT + deg2rad(wing.twist_root)
    tt = αT + deg2rad(wing.twist_tip)
    xle   = [zero(T), T(wing.span * tand(wing.sweep))]
    yle   = [zero(T), T(wing.span)]
    zle   = [zero(T), T(wing.span * tand(wing.dihedral))]
    chord = [wing.chord_root, wing.chord_tip]
    twist = [tr, tt]
    phi   = [zero(T), zero(T)]
    fc    = fill(x -> zero(T), 2)

    grid, ratio = wing_to_grid(xle, yle, zle, chord, twist, phi,
        wing.ns, wing.nc;
        fc=fc, spacing_s=wing.spacing_s, spacing_c=wing.spacing_c,
        mirror=false)
    if wing._system_cache[] === nothing
        wing._system_cache[] = System([grid]; ratios=[ratio])
    else
        sys = wing._system_cache[]
        sys.grids[1] .= grid
        sys.ratios[1] .= ratio
    end
    system = wing._system_cache[]

    rp = wing_reference(wing)
    ref = Reference(rp.S, rp.cref, wing.span,
                    [wing.chord_root/4, 0, 0], V∞)
    fs  = Freestream(V∞, zero(T), zero(T), [zero(T), zero(T), zero(T)])

    if inflow === nothing
        steady_analysis!(system, ref, fs; symmetric=false)
    else
        steady_analysis!(system, ref, fs; symmetric=false,
                         additional_velocity=inflow)
    end

    CF_wind, CM_wind = body_forces(system; frame=Wind())
    CF_body, _       = body_forces(system; frame=Body())
    CL = CF_wind[3]; CD = CF_wind[1]; CY = CF_body[2]
    CDi = far_field_drag(system)

    # Spanwise sectional loading (per-segment, Wind frame): cl is the
    # z-component of the per-span force coefficient; y is the segment
    # midpoint along the span. Robust to the VortexLattice version that
    # may not export `lifting_line_coefficients` — fall back to empties.
    cl_span = T[]; y_span = T[]
    try
        r, c = lifting_line_geometry(system.grids, 0.25)
        cf, _ = lifting_line_coefficients(system, r, c; frame=Wind())
        cz = cf[1]                                  # (3, ns)
        ns = size(cz, 2)
        cl_span = T[cz[3, k] for k in 1:ns]
        # segment midpoint y from the lifting-line geometry (3, ns+1)
        ry = r[1]
        y_span = T[(ry[2, k] + ry[2, k+1]) / 2 for k in 1:ns]
    catch
    end

    e = abs(CL) > 1e-8 ? CL^2 / (π * rp.AR * CDi) : T(NaN)
    return (; CL, CDi, CD, CM = CM_wind, CY, cl_span, y_span, AR = rp.AR, e)
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
    # Lazy cache for the VortexLattice System (per-process, per-rotor).
    # The geometry is fixed at construction so System can be built
    # once and reused across `rotor_forces` calls — saves the bulk of
    # the per-step allocation in the integrated stack (J5 finding).
    _system_cache :: Base.RefValue{Any}
end

function BladedRotor(; N_blades::Int, R::Real, R_hub::Real,
                       chord::NTuple{2,<:Real}, twist::NTuple{2,<:Real},
                       ns::Int=16, nc::Int=6, T::Type=Float64)
    BladedRotor{T}(N_blades, T(R), T(R_hub),
                   (T(chord[1]), T(chord[2])),
                   (T(twist[1]), T(twist[2])),
                   ns, nc,
                   Ref{Any}(nothing))
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
    # Cache hit? VortexLattice System depends only on the rotor's
    # geometry (fixed at construction), so we can build it once and
    # reuse it across calls. Saves the bulk of per-step allocations.
    if rotor._system_cache[] === nothing
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
            c, s = cos(θ), sin(θ)
            g = copy(grid_root)
            @inbounds for j in 1:size(g, 3), i in 1:size(g, 2)
                y = g[2, i, j]; z = g[3, i, j]
                g[2, i, j] = y * c - z * s
                g[3, i, j] = y * s + z * c
            end
            push!(grids,  g)
            push!(ratios, ratio)
        end
        rotor._system_cache[] = System(grids; ratios=ratios)
    end
    system = rotor._system_cache[]
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
    # Box: ±3ε around x_world (cell-units). Note the box is built in the
    # cell-index frame; the per-component face offsets only shift the
    # kernel by 0.5 cell.
    box_lo = ntuple(d -> max(1, floor(Int, x_world[d] - 3ε)),  D)
    box_hi = ntuple(d -> min(sz[d], ceil(Int, x_world[d] + 3ε)), D)
    inv2ε² = inv(T(2 * ε^2))     # Gaussian variance = ε²
    box    = CartesianIndices(ntuple(d -> box_lo[d]:box_hi[d], D))
    # Component-aware deposition. flow.f is FACE-staggered: component d
    # lives at face d (offset -0.5 from cell centre along axis d only).
    # For each component d we therefore compute a separate kernel sum
    # ksum_d and renormalise so the integrated component-d force equals
    # `force[d]` exactly.
    @inbounds for d in 1:D
        ksum_d = zero(T)
        for I in box
            r² = zero(T)
            for k in 1:D
                # Face d sits at I[k] - (k == d ? 1.0 : 1.5)
                δ = (I[k] - (k == d ? T(1.0) : T(1.5))) - T(x_world[k])
                r² += δ*δ
            end
            ksum_d += exp(-r² * inv2ε²)
        end
        ksum_d == 0 && continue
        inv_ks_d = inv(ksum_d)
        for I in box
            r² = zero(T)
            for k in 1:D
                δ = (I[k] - (k == d ? T(1.0) : T(1.5))) - T(x_world[k])
                r² += δ*δ
            end
            w = exp(-r² * inv2ε²) * inv_ks_d
            f[I, d] += T(force[d]) * w
        end
    end
    return f
end

"""
    smear_torque!(f, torque, center, axis, R; N=8, ε=2.0)

Deposit an axial torque `torque` (scalar, signed) about `axis` into
the 3D face-staggered force array `f` as a ring of `N` tangential
point-forces at radius `R` around `center`. The forces sum to zero
(no net force) but produce a net moment of `torque` about `axis`.

Use to add the propeller swirl to a thrust-only `smear_force!` call
when the integrated demo wants both axial and rotational effects.
"""
function smear_torque!(f::AbstractArray{T, 4}, torque::Real,
                       center, axis, R::Real;
                       N::Int = 8, ε::Real = 2.0) where T
    # Build an orthonormal basis (e1, e2) perpendicular to axis.
    a = SVector{3, T}(axis[1], axis[2], axis[3])
    a = a ./ sqrt(sum(abs2, a))
    e1 = if abs(a[1]) < 0.9
        SVector{3, T}(1, 0, 0) - (a[1]) .* a
    else
        SVector{3, T}(0, 1, 0) - (a[2]) .* a
    end
    e1 = e1 ./ sqrt(sum(abs2, e1))
    e2 = SVector{3, T}(
        a[2]*e1[3] - a[3]*e1[2],
        a[3]*e1[1] - a[1]*e1[3],
        a[1]*e1[2] - a[2]*e1[1],
    )
    # Each point gets F_tan such that ∑ r × F = torque ⇒ N · R · F_tan = torque.
    F_tan = T(torque) / (T(N) * T(R))
    c = SVector{3, T}(center[1], center[2], center[3])
    for k in 0:N-1
        θ = T(2π * k / N)
        cθ, sθ = cos(θ), sin(θ)
        # Radial direction at this θ
        r_hat = cθ .* e1 .+ sθ .* e2
        # Tangential direction: axis × r_hat
        t_hat = SVector{3, T}(
            a[2]*r_hat[3] - a[3]*r_hat[2],
            a[3]*r_hat[1] - a[1]*r_hat[3],
            a[1]*r_hat[2] - a[2]*r_hat[1],
        )
        x_point = c .+ T(R) .* r_hat
        smear_force!(f, F_tan .* t_hat, x_point; ε=ε)
    end
    return f
end

"""
    smear_blades!(f, thrust, torque, center, axis, R, R_hub;
                  N_blades=3, N_sections=4, ε=1.5)

Spread `thrust` and `torque` across `N_blades × N_sections`
deposition points arranged as radial blade lines around the rotor
axis, instead of as a single point smear + a tangential ring.

Each blade is a line at angular position `2π·k/N_blades`,
`N_sections` evenly-spaced sections along its span from `R_hub` to
`R`. At each section:

  - axial force = uniform fraction `thrust / (N_blades·N_sections)`
  - tangential force linear-in-r (canonical propeller loading),
    normalised so the discrete `Σ(r × f_t)` matches `torque`

Closer to an actuator-line method than a single-point smear; the
distributed thrust footprint matches the blade swept area rather
than a hot-spot at the centre. Use as a drop-in replacement for
`smear_force!` + `smear_torque!` on BladedRotor's
`(thrust, torque)` outputs.
"""
function smear_blades!(f::AbstractArray{T, 4}, thrust::Real, torque::Real,
                       center, axis, R::Real, R_hub::Real;
                       N_blades::Int = 3, N_sections::Int = 4,
                       ε::Real = 1.5) where T
    a = SVector{3, T}(axis[1], axis[2], axis[3])
    a = a ./ sqrt(sum(abs2, a))
    e1 = if abs(a[1]) < 0.9
        SVector{3, T}(1, 0, 0) - (a[1]) .* a
    else
        SVector{3, T}(0, 1, 0) - (a[2]) .* a
    end
    e1 = e1 ./ sqrt(sum(abs2, e1))
    e2 = SVector{3, T}(
        a[2]*e1[3] - a[3]*e1[2],
        a[3]*e1[1] - a[1]*e1[3],
        a[1]*e1[2] - a[2]*e1[1],
    )
    c = SVector{3, T}(center[1], center[2], center[3])
    # Per-blade axial thrust
    f_axial_per_section = T(thrust) / T(N_blades * N_sections)
    # Tangential force normalisation: ∑(r × f_t) = torque
    # With f_t(r) = K·r over N_blades·N_sections sections each at r_k:
    # ∑ r_k · K r_k = K · ∑ r_k² = torque ⇒ K = torque / ∑ r_k²
    sum_r² = zero(T)
    for k in 1:N_sections
        r_k = T(R_hub) + (T(k) - T(0.5)) / T(N_sections) * T(R - R_hub)
        sum_r² += r_k * r_k
    end
    K_τ = sum_r² > 0 ? T(torque) / (T(N_blades) * sum_r²) : zero(T)
    # Deposit
    for b in 0:N_blades-1
        θ = T(2π * b / N_blades)
        cθ, sθ = cos(θ), sin(θ)
        # Blade radial direction at this θ; tangential = a × r_hat
        r_hat = cθ .* e1 .+ sθ .* e2
        t_hat = SVector{3, T}(
            a[2]*r_hat[3] - a[3]*r_hat[2],
            a[3]*r_hat[1] - a[1]*r_hat[3],
            a[1]*r_hat[2] - a[2]*r_hat[1],
        )
        for k in 1:N_sections
            r_k = T(R_hub) + (T(k) - T(0.5)) / T(N_sections) * T(R - R_hub)
            pos = c .+ r_k .* r_hat
            f_tan = K_τ * r_k
            F = f_axial_per_section .* a .+ f_tan .* t_hat
            smear_force!(f, F, pos; ε=ε)
        end
    end
    return f
end

"""
    trilinear_inflow(u_field; offset=SVector(0,0,0))

Build an `(x,y,z) -> SVector{3}` closure that returns the local
velocity at world position `(x,y,z)` by trilinear interpolation of
the WaterLily face-staggered velocity array `u_field` (shape `(N...,
D)`). World position is in WaterLily cell-coordinates; pass `offset`
to translate (e.g. world ↔ rudder-frame).

Use as VortexLattice's `additional_velocity` argument:

```julia
u_at = trilinear_inflow(sim.flow.u)
r = rudder_forces(rudder, δ, V∞; inflow=u_at)
```

The returned velocity is the **perturbation** on top of the freestream
— *not* the absolute velocity — because that's the VortexLattice
convention. If you want the absolute velocity to be `u_field`, set the
VLM freestream Vinf to 0 (rare in ship CFD; usually freestream is V∞
and the WaterLily field is the perturbation w.r.t. V∞).
"""
function trilinear_inflow(u_field::AbstractArray{T, N}; offset=nothing) where {T, N}
    D = N - 1
    sz = ntuple(d -> size(u_field, d), D)
    off = offset === nothing ? SVector(ntuple(_ -> zero(T), D)...) : offset
    return (xv) -> begin
        # VortexLattice passes a single SVector{3} position.
        # Pull it into cell-index coordinates with offset applied.
        p = (T(xv[1]), T(xv[2]), T(xv[3])) .+ Tuple(off)
        idx = ntuple(d -> p[d] + T(1.5), D)
        # Clamp to interior so we don't sample ghosts. Interior is
        # `2:sz[d]-1`; with the +1 access in the trilinear stencil the
        # safe upper bound for `i` is `sz[d]-1`.
        i = ntuple(d -> clamp(floor(Int, idx[d]), 2, sz[d] - 1), D)
        fr = ntuple(d -> idx[d] - i[d], D)
        # Trilinear over 8 corners (3D only — generic-D is a small loop).
        u = SVector{D, T}(ntuple(D) do d
            v = zero(T)
            for k in 0:1, j in 0:1, ii in 0:1
                w = (ii == 0 ? (one(T) - fr[1]) : fr[1]) *
                    (j  == 0 ? (one(T) - fr[2]) : fr[2]) *
                    (k  == 0 ? (one(T) - fr[3]) : fr[3])
                @inbounds v += w * u_field[i[1] + ii, i[2] + j, i[3] + k, d]
            end
            v
        end)
        return u
    end
end

end # module
