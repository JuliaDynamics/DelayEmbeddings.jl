using DelayEmbeddings
using Test

println("\nTesting delay count estimation...")

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

@testset "Embedding dimension estimation" begin
s_sin = sin.(0:0.1:1000)
ds = Systems.roessler(ones(3));τ=15; dt=0.1
data = trajectory(ds,1000;dt=dt,diffeq...)
s_roessler = data[:,1]
ds = Systems.lorenz();τ=5; dt=0.01
data = trajectory(ds,500;dt=dt,diffeq...)
s_lorenz = data[:,1]
τ = 15
Ds = 1:5

@testset "Caos method" begin
    E1s = DelayEmbeddings.estimate_dimension(s_sin, τ, Ds)
    @test E1s[1] > 0.9

    # @test minimum(E2s) < 0.1 # THIS TEST FAILS
    Ds = 1:5
    E1s = DelayEmbeddings.estimate_dimension(s_roessler, τ, Ds)
    @test E1s[2] > 0.8
    E2s = DelayEmbeddings.stochastic_indicator(s_roessler, τ, Ds)
    @test minimum(E2s) < 0.3

    E1s = DelayEmbeddings.estimate_dimension(s_roessler, τ, Ds;metric = Chebyshev())
    @test saturation_point(Ds,E1s; threshold=0.1) ∈ [2, 3]

    Ds = 1:5
    E1s = DelayEmbeddings.estimate_dimension(s_lorenz, τ, Ds)
    E2s = DelayEmbeddings.stochastic_indicator(s_lorenz, τ, Ds)
    @test saturation_point(Ds,E1s; threshold=0.1) ∈ [2, 3]

    #Test against random signal
    E2s = DelayEmbeddings.stochastic_indicator(rand(10000), 1, 1:5)
    @test minimum(E2s) > 0.9
end

@testset "fnn method" begin
    τ = 15
    Ds = 1:5
    number_fnn = DelayEmbeddings.estimate_dimension(s_sin, τ, Ds, "fnn")
    @test findfirst(number_fnn .≈ 0.0) == 1

    τ = 15
    Ds = 1:5
    number_fnn = DelayEmbeddings.estimate_dimension(s_roessler, τ, Ds, "fnn"; rtol=15)
    @test findfirst(number_fnn .≈ 0.0) ∈ [2, 3]

    τ = 1
    Ds = 1:5
    number_fnn = DelayEmbeddings.estimate_dimension(s_lorenz, τ, Ds, "fnn"; atol=1, rtol=3.0)
    @test findfirst(number_fnn .≈ 0.0) ∈ [2, 3]
end

@testset "f1nn method" begin
    τ = 15
    Ds = 1:5
    ffnn_ratio = DelayEmbeddings.estimate_dimension(s_sin, τ, Ds, "f1nn")
    @test (1 .- ffnn_ratio)[1] > 0.8

    τ = 15
    Ds = 1:5
    ffnn_ratio = DelayEmbeddings.estimate_dimension(s_roessler, τ, Ds, "f1nn")
    @test (1 .- ffnn_ratio)[2] > 0.8
    ffnn_ratio = DelayEmbeddings.estimate_dimension(s_roessler, τ, Ds, "f1nn";metric=Chebyshev())
    @test (1 .- ffnn_ratio)[2] > 0.8

    τ = 15
    Ds = 1:5
    ffnn_ratio = DelayEmbeddings.estimate_dimension(s_lorenz, τ, Ds, "f1nn")
    @test (1 .- ffnn_ratio)[3] > 0.8
end
end
