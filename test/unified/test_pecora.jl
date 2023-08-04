using DelayEmbeddings, DynamicalSystemsBase
using Test
using Random

@testset "Pecora" begin
# Generate data
function lorenz(u0=[0.0, 10.0, 0.0]; σ = 10.0, ρ = 28.0, β = 8/3)
    return CoupledODEs(lorenz_rule, u0, [σ, ρ, β])
end
@inbounds function lorenz_rule(u, p, t)
    du1 = p[1]*(u[2]-u[1])
    du2 = u[1]*(p[2]-u[3]) - u[2]
    du3 = u[1]*u[2] - p[3]*u[3]
    return SVector{3}(du1, du2, du3)
end
lor = lorenz([0.0, 1.0, 0.0]; ρ=60)
data, = trajectory(lor, 200; Δt=0.02, Ttr = 10)
metric = Chebyshev()

@testset "Pecora univariate" begin
    s = data[:, 2] # input timeseries = first entry of lorenz
    optimal_τ = estimate_delay(s, "mi_min")
    Tmax = 100
    K = 14
    samplesize = 1

    τs = (0,)
    Random.seed!(123)
    es_ref, Γs = pecora(s, τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric)
    (max1,_) = DelayEmbeddings.findlocalextrema(vec(es_ref))
    @test optimal_τ - 2 ≤ max1[1]-1 ≤ optimal_τ + 2
    maxi = maximum(es_ref)
    @test maxi ≤ 1.3

    Random.seed!(123)
    es_ref2, Γs = pecora(s, τs; delays = 0:Tmax, w = optimal_τ, K = K, metric = metric)
    (max2,_) = DelayEmbeddings.findlocalextrema(vec(es_ref2))
    @test optimal_τ - 2 ≤ max2[1]-1 ≤ optimal_τ + 2
    maxi2 = maximum(es_ref2)
    @test maxi - .1 ≤ maxi2 ≤ maxi + .1

    τs = (0, max1[1]-1,)
    Random.seed!(123)
    es, Γs = pecora(s, τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric)
    (max2,min1) = DelayEmbeddings.findlocalextrema(vec(es))
    @test 1 ≤ max2[1]-1 ≤ 4
    @test optimal_τ - 2 ≤ min1[2]-1 ≤ optimal_τ + 2

    τs = (0, max1[1]-1, max2[1]-1)
    Random.seed!(123)
    es, Γs = pecora(s, τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric)
    (_,min2) = DelayEmbeddings.findlocalextrema(vec(es))
    @test max2[1]-2 ≤ min2[1]-1 ≤ max2[1]

end

@testset "Pecora multivariate" begin
## %% Trajectory case
    s = StateSpaceSet(data)
    optimal_τ = estimate_delay(s[:,2], "mi_min")
    Tmax = 100
    K = 14
    samplesize = 1

    js = (2,)
    τs = (0,)
    Random.seed!(123)
    es_ref, _ = pecora(s[:,2], τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric)
    Random.seed!(123)
    es_ref2, _ = pecora(s, τs, js; delays = 0:Tmax, w = optimal_τ, samplesize = .2, K = K, metric = metric)
    Random.seed!(123)
    es, _= pecora(s, τs, js; delays = 0:Tmax, samplesize = samplesize, w = optimal_τ, K = K, metric = metric)


    @test round.(es[:,2], digits = 4) == round.(vec(es_ref),digits = 4)
    (x_maxi,_) = DelayEmbeddings.findlocalextrema(es[:,1])
    @test 8 ≤ x_maxi[2]-1 ≤ 10
    (z_maxi,_) = DelayEmbeddings.findlocalextrema(es[:,3])
    @test 13 ≤ z_maxi[2]-1 ≤ 15
    (x_maxi_ref,_) = DelayEmbeddings.findlocalextrema(es[:,2])
    (x_maxi_ref2,_) = DelayEmbeddings.findlocalextrema(es_ref2[:,2])

    # TODO:
    # This test fails:
    # @test x_maxi_ref[1:4] == x_maxi_ref2[1:4]
    @test es[x_maxi_ref[1:4],2] .- .1 ≤ es_ref2[x_maxi_ref2[1:4],2] ≤ es[x_maxi_ref[1:4],2] .+ .1

end

end
