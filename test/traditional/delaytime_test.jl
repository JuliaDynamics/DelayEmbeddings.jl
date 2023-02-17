using DelayEmbeddings, Test
using DelayEmbeddings: exponential_decay_fit
using DynamicalSystemsBase

testval = (val, vmin, vmax) -> @test vmin ≤ val ≤ vmax
diffeq = (atol = 1e-9, rtol = 1e-9)

@testset "Estimate Delay" begin
    # test exponential decay fit
    x = 0:200
    y = @. exp(-x/5)
    @test exponential_decay_fit(x, y, :equal) ≈ 5
    @test exponential_decay_fit(x, y, :small) ≈ 5

    henon_rule(x, p, n) = SVector(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
    ds = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])

    data = trajectory(ds, 1000)[1]
    x = data[:,1]
    @test estimate_delay(x,"ac_zero", 0:10) ≤ 2
    @test estimate_delay(x,"ac_min", 0:10)  ≤ 2
    @test estimate_delay(x,"exp_extrema", 0:10)  ≤ 4
    @test estimate_delay(x,"mi_min", 0:10) == 10

    # Chaotic Roessler timeseries
    function roessler_rule(u, p, t)
        @inbounds begin
        a, b, c = p
        du1 = -u[2]-u[3]
        du2 = u[1] + a*u[2]
        du3 = b + u[3]*(u[1] - c)
        return SVector{3}(du1, du2, du3)
        end
    end
    ro = CoupledODEs(roessler_rule, ones(3), [0.2, 0.2, 5.7]; diffeq)
    Δt = 0.02
    data = trajectory(ro,500.0; Δt)[1]
    x = data[:,1]

    @test 1.3 ≤ estimate_delay(x,"ac_zero", 1:2:500)*Δt ≤ 1.7
    @test 2.6 ≤ estimate_delay(x,"ac_min", 1:2:500)*Δt  ≤ 3.4
    @test 1.0 ≤ estimate_delay(x,"mi_min", 1:2:500)*Δt ≤ 1.6

    Δt = 0.1
    data = trajectory(ro,2000.0; Δt)[1]
    x = data[:,1]
    @test 1.3 ≤ estimate_delay(x,"ac_zero", 1:1:50)*Δt ≤ 1.7
    @test 2.6 ≤ estimate_delay(x,"ac_min", 1:1:50)*Δt  ≤ 3.4
    @test 1.15 ≤ estimate_delay(x,"mi_min", 1:1:50)*Δt ≤ 1.6

end
