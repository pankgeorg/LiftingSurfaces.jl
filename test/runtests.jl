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

end
