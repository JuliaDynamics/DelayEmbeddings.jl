using DelayEmbeddings
using Test
using DynamicalSystemsBase

println("\nTesting traditional optimal embedding dimension...")
test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

@testset "Embedding dimension estimation" begin
diffeq = (atol = 1e-9, rtol = 1e-9, maxiters = typemax(Int))
s_sin = sin.(0:0.1:1000)
ro = Systems.roessler(ones(3));
data = trajectory(ro,1000;Δt=0.1,diffeq)
s_roessler = data[:,1]
lo = Systems.lorenz96(4, [0.1, 0.2, 0.5, 0.1]; F = 16.0);
data = trajectory(lo,5000.0;Δt=0.05, Ttr = 100.0, diffeq)
s_lorenz = data[:,1]

@testset "Caos method" begin
    𝒟, τ, x = optimal_traditional_de(s_roessler, "afnn")
    @test 3 ≤ size(𝒟, 2) ≤ 5

    E2s = DelayEmbeddings.stochastic_indicator(s_roessler, τ, 1:6)
    @test minimum(E2s) < 0.3

    𝒟, τ, x = optimal_traditional_de(s_roessler, "afnn"; metric = Chebyshev())
    @test 3 ≤ size(𝒟, 2) ≤ 5

    𝒟, τ, x = optimal_traditional_de(s_lorenz, "afnn")
    @test 4 ≤ size(𝒟, 2) ≤ 8

    #Test against random signal
    E2s = DelayEmbeddings.stochastic_indicator(rand(10000), 1, 1:5)
    @test minimum(E2s) > 0.9
end

@testset "fnn method" begin
    𝒟, τ, x = optimal_traditional_de(s_sin, "fnn")
    @test 1 ≤ size(𝒟, 2) ≤ 3

    𝒟, τ, x = optimal_traditional_de(s_roessler, "fnn")
    @test 3 ≤ size(𝒟, 2) ≤ 5

    𝒟, τ, x = optimal_traditional_de(s_lorenz, "fnn")
    @test 4 ≤ size(𝒟, 2) ≤ 8
end

@testset "ifnn method" begin
    𝒟, τ, x = optimal_traditional_de(s_sin, "ifnn")
    @test 1 ≤ size(𝒟, 2) ≤ 4

    𝒟, τ, x = optimal_traditional_de(s_roessler, "ifnn")
    @test 3 ≤ size(𝒟, 2) ≤ 5

    𝒟, τ, x = optimal_traditional_de(s_roessler, "ifnn"; metric = Chebyshev())
    @test 3 ≤ size(𝒟, 2) ≤ 5

    𝒟, τ, x = optimal_traditional_de(s_lorenz, "ifnn")
    @test 4 ≤ size(𝒟, 2) ≤ 8
end

end
