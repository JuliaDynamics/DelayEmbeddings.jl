using ChaosTools, DelayEmbeddings
using Test

println("\nTesting delay estimation...")

testval = (val, vmin, vmax) -> @test vmin ≤ val ≤ vmax

@testset "Estimate Delay" begin

    ds = Systems.henon()
    data = trajectory(ds,1000;dt=1)
    x = data[:,1]
    @test estimate_delay(x,"ac_zero", 0:10) ≤ 2
    @test estimate_delay(x,"ac_min", 0:10)  ≤ 2
    # @test estimate_delay(x,"exp_decay")  ≤ 2
    @test estimate_delay(x,"mi_min", 0:10) == 10

    ds = Systems.roessler(ones(3))
    dt = 0.02
    data = trajectory(ds,200,dt=dt)
    x = data[:,1]
    @test 1.3 ≤ estimate_delay(x,"ac_zero", 1:2:500)*dt ≤ 1.7
    @test 2.6 ≤ estimate_delay(x,"ac_min", 1:2:500)*dt  ≤ 3.4
    @test 1.15 ≤ estimate_delay(x,"mi_min", 1:2:500)*dt ≤ 1.6

    dt = 0.1
    data = trajectory(ds,2000,dt=dt)
    x = data[:,1]
    @test 1.3 ≤ estimate_delay(x,"ac_zero", 1:1:50)*dt ≤ 1.7
    @test 2.6 ≤ estimate_delay(x,"ac_min", 1:1:50)*dt  ≤ 3.4
    @test 1.15 ≤ estimate_delay(x,"mi_min", 1:1:50)*dt ≤ 1.6

    ds = Systems.lorenz()
    dt = 0.05
    data = trajectory(ds,1000;dt=dt)
    x = data[500:end,1]
    @test 0.1 ≤ estimate_delay(x,"mi_min")*dt ≤ 0.4
    # @test 0.1 ≤ estimate_delay(x,"exp_decay")*dt  ≤ 0.4
end
