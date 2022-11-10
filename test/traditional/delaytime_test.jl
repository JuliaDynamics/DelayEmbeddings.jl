using DelayEmbeddings
using DelayEmbeddings: exponential_decay_fit
using DynamicalSystemsBase, Test

println("\nTesting delay time estimation...")

testval = (val, vmin, vmax) -> @test vmin ≤ val ≤ vmax
diffeq = (atol = 1e-9, rtol = 1e-9, maxiters = typemax(Int))

@testset "Estimate Delay" begin
    # test exponential decay fit
    x = 0:200
    y = @. exp(-x/5)
    @test exponential_decay_fit(x, y, :equal) ≈ 5
    @test exponential_decay_fit(x, y, :small) ≈ 5

    ds = Systems.henon()
    data = trajectory(ds, 1000)
    x = data[:,1]
    @test estimate_delay(x,"ac_zero", 0:10) ≤ 2
    @test estimate_delay(x,"ac_min", 0:10)  ≤ 2
    @test estimate_delay(x,"exp_extrema", 0:10)  ≤ 4
    @test estimate_delay(x,"mi_min", 0:10) == 10

    ds = Systems.roessler(ones(3))
    Δt = 0.02
    data = trajectory(ds,500.0;Δt,diffeq)
    x = data[:,1]
    @test 1.3 ≤ estimate_delay(x,"ac_zero", 1:2:500)*Δt ≤ 1.7
    @test 2.6 ≤ estimate_delay(x,"ac_min", 1:2:500)*Δt  ≤ 3.4
    @test 1.0 ≤ estimate_delay(x,"mi_min", 1:2:500)*Δt ≤ 1.6

    Δt = 0.1
    data = trajectory(ds,2000.0; Δt,diffeq)
    x = data[:,1]
    @test 1.3 ≤ estimate_delay(x,"ac_zero", 1:1:50)*Δt ≤ 1.7
    @test 2.6 ≤ estimate_delay(x,"ac_min", 1:1:50)*Δt  ≤ 3.4
    @test 1.15 ≤ estimate_delay(x,"mi_min", 1:1:50)*Δt ≤ 1.6

    ds = Systems.lorenz()
    Δt = 0.1
    data = trajectory(ds,5000;Δt,diffeq)
    x = data[500:end,1]
    @test 0 < estimate_delay(x,"exp_extrema", 0:2:200)  < 200

    # Issue #18
    ds = Systems.gissinger(ones(3)) # 3D continuous chaotic system, also shown in orbit diagrams tutorial
    Δt = 0.05
    data = trajectory(ds, 1000.0, Δt = Δt)
    s = data[:, 1]

    τ = estimate_delay(s, "mi_min", 0:1:400) # this was the erroring line
    @test τ > 0
end
