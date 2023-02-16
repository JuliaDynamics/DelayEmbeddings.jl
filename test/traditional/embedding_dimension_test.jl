using DelayEmbeddings
using Test
using DynamicalSystemsBase

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

@testset "Embedding dimension estimation" begin

s_sin = sin.(0:0.1:1000)

diffeq = (atol = 1e-9, rtol = 1e-9)

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
data = trajectory(ro, 1000; Δt=0.1)[1]
s_roessler = data[:,1]

# Chaotic Lorenz96 timeseries
struct Lorenz96{N} end # Structure for size type
function (obj::Lorenz96{N})(dx, x, p, t) where {N}
    F = p[1]
    # 3 edge cases
    @inbounds dx[1] = (x[2] - x[N - 1]) * x[N] - x[1] + F
    @inbounds dx[2] = (x[3] - x[N]) * x[1] - x[2] + F
    @inbounds dx[N] = (x[1] - x[N - 2]) * x[N - 1] - x[N] + F
    # then the general case
    for n in 3:(N - 1)
      @inbounds dx[n] = (x[n + 1] - x[n - 2]) * x[n - 1] - x[n] + F
    end
    return nothing
end
lo = CoupledODEs(Lorenz96{4}(), [0.1, 0.2, 0.5, 0.1], [32.0]; diffeq)
data = trajectory(lo, 1000.0; Δt=0.05, Ttr = 100.0)[1]
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
