using DelayEmbeddings
using Test
using DynamicalSystemsBase

println("\nTesting delay count estimation...")

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

@testset "Embedding dimension estimation" begin
s_sin = sin.(0:0.1:1000)
γs = Systems.roessler(ones(3));
data = trajectory(γs,1000;dt=0.1,diffeq...)
s_roessler = data[:,1]
γs = Systems.lorenz();
data = trajectory(γs,500;dt=0.01,diffeq...)
s_lorenz = data[:,1]
τ = 15
γs = 1:8

@testset "Caos method" begin
    E1s = afnn(s_sin, τ, γs) # call afnn directly
    @test E1s[1] > 0.9 # should already converge for dimension 2

    𝒟, τ, x = optimal_traditional_de(s_roessler, "afnn")
    @test 3 ≤ size(𝒟, 2) ≤ 5
    E2s = DelayEmbeddings.stochastic_indicator(s_roessler, τ, γs)
    @test minimum(E2s) < 0.3

    𝒟, τ, x = optimal_traditional_de(s_roessler, "afnn"; metric = Chebyshev())
    @test 3 ≤ size(𝒟, 2) ≤ 5

    𝒟, τ, x = optimal_traditional_de(s_lorenz, "afnn")
    @test 3 ≤ size(𝒟, 2) ≤ 5

    #Test against random signal
    E2s = DelayEmbeddings.stochastic_indicator(rand(10000), 1, 1:5)
    @test minimum(E2s) > 0.9
end

@testset "fnn method" begin
    τ = 15
    number_fnn = DelayEmbeddings.estimate_dimension(s_sin, τ, γs, "fnn")
    @test findfirst(number_fnn .≈ 0.0) == 1

    τ = 15
    γs = 1:5
    number_fnn = DelayEmbeddings.estimate_dimension(s_roessler, τ, γs, "fnn"; rtol=15)
    @test findfirst(number_fnn .≈ 0.0) ∈ [2, 3]

    τ = 1
    number_fnn = DelayEmbeddings.estimate_dimension(s_lorenz, τ, γs, "fnn"; atol=1, rtol=3.0)
    @test findfirst(number_fnn .≈ 0.0) ∈ [2, 3]
end

@testset "f1nn method" begin
    ffnn_ratio = DelayEmbeddings.estimate_dimension(s_sin, τ, γs, "f1nn")
    @test (1 .- ffnn_ratio)[1] > 0.8

    ffnn_ratio = DelayEmbeddings.estimate_dimension(s_roessler, τ, γs, "f1nn")
    @test (1 .- ffnn_ratio)[2] > 0.8
    ffnn_ratio = DelayEmbeddings.estimate_dimension(s_roessler, τ, γs, "f1nn";metric=Chebyshev())
    @test (1 .- ffnn_ratio)[2] > 0.8

    ffnn_ratio = DelayEmbeddings.estimate_dimension(s_lorenz, τ, γs, "f1nn")
    @test (1 .- ffnn_ratio)[3] > 0.8
end
end
