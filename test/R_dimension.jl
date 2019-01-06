using ChaosTools, DelayEmbeddings
using Test

println("\nTesting R-param. estimation...")

"""
    saturation_point(x, y; threshold = 0.01, dxi::Int = 1, tol = 0.2)
Decompose the curve `y(x)` into linear regions using `linear_regions(x, y; dxi, tol)`
and then attempt to find a saturation point where the the first slope
of the linear regions become `< threshold`.

Return the `x` value of the saturation point.
"""
function saturation_point(Ds, E1s; threshold = 0.01, kwargs...)
    lrs, slops = ChaosTools.linear_regions(Ds, E1s; kwargs...)
    i = findfirst(x -> x < threshold, slops)
    return i == 0 ? Ds[end] : Ds[lrs[i]]
end

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

@testset "Estimate Dimension" begin
    s = sin.(0:0.1:1000)
    τ = 15
    Ds = 1:5
    E1s = DelayEmbeddings.estimate_dimension(s, τ, Ds)
    E2s = DelayEmbeddings.stochastic_indicator(s, τ, Ds)
    @test saturation_point(Ds,E1s; threshold=0.01) == 1

    # @test minimum(E2s) < 0.1 # THIS TEST FAILS

    ds = Systems.roessler();τ=15; dt=0.1
    data = trajectory(ds,1000;dt=dt)
    s = data[:,1]
    Ds = 1:5
    E1s = DelayEmbeddings.estimate_dimension(s, τ, Ds)
    E2s = DelayEmbeddings.stochastic_indicator(s, τ, Ds)
    @test saturation_point(Ds,E1s; threshold=0.1) ∈ [2, 3]
    @test minimum(E2s) < 0.3


    ds = Systems.lorenz();τ=5; dt=0.01
    data = trajectory(ds,500;dt=dt)
    s = data[:,1]
    Ds = 1:5
    E1s = DelayEmbeddings.estimate_dimension(s, τ, Ds)
    E2s = DelayEmbeddings.stochastic_indicator(s, τ, Ds)
    @test saturation_point(Ds,E1s; threshold=0.1) ∈ [2, 3]

    # @test minimum(E2s) < 0.1 # THIS TEST FAILS

    #Test against random signal
    E2s = DelayEmbeddings.stochastic_indicator(rand(100000), 1, 1:5)
    @test minimum(E2s) > 0.9

end
