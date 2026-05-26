using Test
using LiftingSurfaces

@testset "LiftingSurfaces" begin

    @testset "Rudder polar — sign and magnitude" begin
        rudder = Rudder(; chord=1.0, span=2.0, ns=12, nc=6)
        # Positive δ should give positive CL (lift in z; chord pitched up).
        r_pos = rudder_forces(rudder, deg2rad( 5.0), 1.0)
        r_neg = rudder_forces(rudder, deg2rad(-5.0), 1.0)
        @test r_pos.CL > 0
        @test r_neg.CL < 0
        @test isapprox(r_pos.CL, -r_neg.CL; rtol=1e-3)   # symmetric
        # Drag is symmetric around δ=0.
        @test isapprox(r_pos.CD,  r_neg.CD;  rtol=1e-3)
    end

    @testset "Rudder linear slope dCL/dα within range" begin
        rudder = Rudder(; chord=1.0, span=2.0, ns=12, nc=6)
        CL5 = rudder_forces(rudder, deg2rad( 5.0), 1.0).CL
        CLm = rudder_forces(rudder, deg2rad(-5.0), 1.0).CL
        slope = (CL5 - CLm) / deg2rad(10.0)
        # AR=2 rectangular planform: slope ~ 0.78 * elliptic-LLT = 0.78 * π
        @test 2.0 < slope < 3.0
    end

    @testset "BladedRotor smoke" begin
        rotor = BladedRotor(; N_blades=3, R=1.0, R_hub=0.2,
                              chord=(0.25, 0.18),
                              twist=(deg2rad(35.0), deg2rad(15.0)),
                              ns=12, nc=4)
        # Use J=0.7
        V∞ = 1.0; J = 0.7
        Ω  = π * V∞ / (J * rotor.R)
        r  = rotor_forces(rotor, V∞, Ω)
        # Whatever the sign convention, magnitudes should be finite & non-trivial.
        @test isfinite(r.thrust)
        @test isfinite(r.torque)
        @test abs(r.CT) > 0.01
        @test abs(r.CQ) > 0.001
        # η_VLM is normalisation-convention-dependent; just check it's
        # finite and not absurd.
        @test isfinite(r.η_VLM)
        @test abs(r.η_VLM) < 10
    end

    @testset "smear_force! conserves total force" begin
        using StaticArrays: SVector
        sz = (32, 32, 32)
        f = zeros(Float32, sz..., 3)
        force = SVector(1.0f0, -0.5f0, 0.2f0)
        smear_force!(f, force, SVector(16f0, 16f0, 16f0); ε=2.0)
        @test isapprox(sum(@view f[:, :, :, 1]),  1.0f0; atol=1e-5)
        @test isapprox(sum(@view f[:, :, :, 2]), -0.5f0; atol=1e-5)
        @test isapprox(sum(@view f[:, :, :, 3]),  0.2f0; atol=1e-5)
    end

    @testset "trilinear_inflow sanity" begin
        using StaticArrays: SVector
        sz = (16, 16, 16)
        # Uniform u in +x: all (x, y, z) should return (1, 0, 0).
        u = zeros(Float32, sz..., 3)
        u[:, :, :, 1] .= 1f0
        inflow = trilinear_inflow(u)
        v = inflow(SVector(5.0, 7.0, 4.0))
        @test isapprox(v[1], 1f0; atol=1e-5)
        @test isapprox(v[2], 0f0; atol=1e-5)
        @test isapprox(v[3], 0f0; atol=1e-5)
        # Linear-in-x field: u_x(I[1]) = I[1] - 1.5
        for I in CartesianIndices((size(u, 1), size(u, 2), size(u, 3)))
            u[I, 1] = I[1] - 1.5
        end
        inflow2 = trilinear_inflow(u)
        @test isapprox(inflow2(SVector(3.7, 8.0, 8.0))[1], 3.7f0; atol=1e-3)
    end

    @testset "smear_force! peaks at the closest cell" begin
        using StaticArrays: SVector
        sz = (16, 16, 16)
        f = zeros(Float32, sz..., 3)
        force = SVector(1f0, 0f0, 0f0)
        x_loc = SVector(8f0, 8f0, 8f0)
        smear_force!(f, force, x_loc; ε=1.5)
        i_peak = argmax(f[:, :, :, 1])
        # Cell centre for index I = I - 1.5; for x_loc=8 nearest cells
        # are I=9 (centre 7.5) or I=10 (centre 8.5).
        @test i_peak[1] in (9, 10)
        @test i_peak[2] in (9, 10)
        @test i_peak[3] in (9, 10)
    end

    @testset "smear_force! conserves each face-staggered component" begin
        # Regression test for commit 0a32d54 — the per-component
        # renormalisation must keep the integrated force exact for
        # every direction, including the tangential ones whose face
        # offset is along their own axis.
        using StaticArrays: SVector
        sz = (24, 24, 24)
        for force in (SVector(1f0, 0f0, 0f0),
                      SVector(0f0, 0.7f0, 0f0),
                      SVector(0f0, 0f0, -0.3f0),
                      SVector(0.5f0, -0.5f0, 0.5f0))
            f = zeros(Float32, sz..., 3)
            smear_force!(f, force, SVector(12f0, 12f0, 12f0); ε=2.0)
            for d in 1:3
                @test isapprox(sum(@view f[:, :, :, d]), force[d]; atol=1e-5)
            end
        end
    end

    @testset "Rudder responds to inflow perturbation" begin
        # Pass a non-zero inflow perturbation and confirm the rudder
        # force changes from the freestream-only baseline. Specifically:
        # adding +y perturbation at the rudder should change the
        # apparent angle of attack, so CL must differ from the
        # inflow=nothing case.
        using StaticArrays: SVector
        rudder = Rudder(; chord=1.0, span=2.0, ns=12, nc=6)
        r_base = rudder_forces(rudder, deg2rad(5.0), 1.0)
        # Constant +z perturbation reduces effective α — CL drops.
        inflow_up = xv -> SVector(0.0, 0.0, 0.1)
        r_up = rudder_forces(rudder, deg2rad(5.0), 1.0; inflow=inflow_up)
        @test r_up.CL != r_base.CL
        # Constant -z perturbation should push CL the other direction.
        inflow_dn = xv -> SVector(0.0, 0.0, -0.1)
        r_dn = rudder_forces(rudder, deg2rad(5.0), 1.0; inflow=inflow_dn)
        @test (r_up.CL - r_base.CL) * (r_dn.CL - r_base.CL) < 0   # opposite signs
    end

    @testset "smear_blades! sums to total thrust + torque" begin
        using StaticArrays: SVector
        sz = (48, 48, 48)
        f = zeros(Float32, sz..., 3)
        smear_blades!(f, 1.0, 0.4,
                      SVector(24.0, 24.0, 24.0),
                      SVector(1.0, 0.0, 0.0),
                      8.0, 1.5;
                      N_blades=3, N_sections=4, ε=1.5)
        # Total axial force ≈ prescribed thrust
        @test isapprox(sum(@view f[:, :, :, 1]), 1.0f0; atol=1e-3)
        # No net side force in 3-blade symmetry
        @test isapprox(sum(@view f[:, :, :, 2]), 0f0; atol=1e-3)
        @test isapprox(sum(@view f[:, :, :, 3]), 0f0; atol=1e-3)
        # Total moment about +x ≈ prescribed torque
        Mx = 0.0
        for i in 1:sz[1], j in 1:sz[2], k in 1:sz[3]
            y = (j - 1.5) - 24.0
            z = (k - 1.5) - 24.0
            Mx += y * f[i, j, k, 3] - z * f[i, j, k, 2]
        end
        @test isapprox(Mx, 0.4; rtol=0.05)
    end

    @testset "smear_torque! produces a moment, zero net force" begin
        using StaticArrays: SVector
        sz = (32, 32, 32)
        f = zeros(Float32, sz..., 3)
        smear_torque!(f, 1.0, SVector(16.0, 16.0, 16.0),
                      SVector(1.0, 0.0, 0.0), 4.0; N=8, ε=2.0)
        # No net force (ring sums to zero)
        @test isapprox(sum(@view f[:, :, :, 1]), 0f0; atol=1e-4)
        @test isapprox(sum(@view f[:, :, :, 2]), 0f0; atol=1e-4)
        @test isapprox(sum(@view f[:, :, :, 3]), 0f0; atol=1e-4)
        # Net moment about +x ≈ 1.0
        Mx = 0.0
        for i in 1:sz[1], j in 1:sz[2], k in 1:sz[3]
            y = (j - 1.5) - 16.0
            z = (k - 1.5) - 16.0
            Mx += y * f[i, j, k, 3] - z * f[i, j, k, 2]
        end
        @test isapprox(Mx, 1.0; rtol=0.05)
    end

    @testset "smear_force! 2D" begin
        using StaticArrays: SVector
        sz = (24, 24)
        f = zeros(Float32, sz..., 2)
        force = SVector(0.5f0, -0.3f0)
        smear_force!(f, force, SVector(12f0, 12f0); ε=1.8)
        @test isapprox(sum(@view f[:, :, 1]),  0.5f0; atol=1e-5)
        @test isapprox(sum(@view f[:, :, 2]), -0.3f0; atol=1e-5)
    end

    @testset "trilinear_inflow reads a sinusoidal field" begin
        # A non-trivial spatial pattern: u_y(x) = sin(2π · I[1] / 16).
        # At I[1]=4 (centre 2.5) we expect sin(π/4) ≈ 0.707. The
        # trilinear interpolation should hit it within the linear
        # interpolation error.
        using StaticArrays: SVector
        sz = (16, 16, 16)
        u = zeros(Float32, sz..., 3)
        for I in CartesianIndices((size(u,1), size(u,2), size(u,3)))
            u[I, 2] = Float32(sin(2π * (I[1] - 1.5) / 16))
        end
        inflow = trilinear_inflow(u)
        # At cell centre x=2.5 ⇒ I[1]=4 sample
        v = inflow(SVector(2.5, 8.0, 8.0))
        @test isapprox(v[2], sin(2π * 2.5 / 16); atol=0.05)
        # At x=6.0 ⇒ phase π·6/8 = 3π/4
        v2 = inflow(SVector(6.0, 8.0, 8.0))
        @test isapprox(v2[2], sin(2π * 6.0 / 16); atol=0.05)
    end

    @testset "trilinear_inflow clamps out-of-bounds queries" begin
        # Ask for a position past the array — should not error or
        # produce NaN. Boundary cell value is the safest defaulta.
        using StaticArrays: SVector
        sz = (8, 8, 8)
        u = zeros(Float32, sz..., 3)
        u[:, :, :, 1] .= 1f0
        inflow = trilinear_inflow(u)
        v = inflow(SVector(-5.0, 50.0, 0.0))
        @test isfinite(v[1]) && isfinite(v[2]) && isfinite(v[3])
        @test isapprox(v[1], 1f0; atol=1e-5)  # u_x is uniform = 1 everywhere
    end

    @testset "Rudder CL scales with accelerated inflow" begin
        # Mimic the F2 setup: rudder in a region where local axial
        # flow is 2× freestream. The returned CL should be
        # substantially larger than the freestream baseline at the
        # same δ — the in-race amplification observed in F2.
        using StaticArrays: SVector
        rudder = Rudder(; chord=2.0, span=3.0, ns=12, nc=6)
        δ = deg2rad(10.0)
        r_base = rudder_forces(rudder, δ, 1.0)
        # Constant inflow perturbation u_x = +1 (so total V_local = 2).
        boost = xv -> SVector(1.0, 0.0, 0.0)
        r_boost = rudder_forces(rudder, δ, 1.0; inflow=boost)
        @test r_boost.CL > 1.5 * r_base.CL    # ≥ 1.5× — matches F2 ~4×
        @test isfinite(r_boost.CD)
    end

    @testset "smear_blades! returns identity (3D)" begin
        # Check the function returns its `f` argument so the
        # call site can chain or pipeline. Trivial but locks the
        # mutate-and-return contract.
        using StaticArrays: SVector
        sz = (32, 32, 32)
        f = zeros(Float32, sz..., 3)
        f_ret = smear_blades!(f, 1.0, 0.2,
                      SVector(16.0, 16.0, 16.0),
                      SVector(1.0, 0.0, 0.0),
                      6.0, 1.0;
                      N_blades=2, N_sections=3, ε=1.5)
        @test f_ret === f
    end

end
