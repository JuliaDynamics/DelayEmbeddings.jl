using DelayEmbeddings, DynamicalSystemsBase
using Test, Random

println("\nTesting pecuzal_method.jl...")
@testset "PECUZAL" begin

lo = Systems.lorenz([1.0, 1.0, 50.0])
tr = trajectory(lo, 100; dt = 0.01, Ttr = 10)

@testset "Univariate example" begin

    s = vec(tr[:, 1]) # input timeseries = x component of lorenz
    w = estimate_delay(s, "mi_min")
    Tmax = 100

    @time Y, τ_vals, ts_vals, Ls , εs = pecuzal_embedding(s[1:5000];
                                        τs = 0:Tmax , w = w)

    @test -0.728 < Ls[1] < -0.727
    @test -0.324 < Ls[2] < -0.323

    @test τ_vals[2] == 18
    @test τ_vals[3] == 9
    @test length(ts_vals) == 3

    @time Y, τ_vals, ts_vals, Ls , εs = pecuzal_embedding(s[1:5000];
                                        τs = 0:Tmax , w = w, econ = true)
    @test -0.728 < Ls[1] < -0.726
    @test -0.322 < Ls[2] < -0.321

    @test τ_vals[2] == 18
    @test τ_vals[3] == 9
    @test length(ts_vals) == 3

    YY1 = DelayEmbeddings.hcat_lagged_values(Dataset(s), s, 21)
    YY2 = DelayEmbeddings.hcat_lagged_values(YY1, s, 13)
    @test length(YY1) == length(YY2)
    @test YY1 == YY2[:,1:2]

end


@testset "Multivariate example" begin
## Test of the proposed Pecora-Uzal-embedding-method (multivariate case)

    w1 = estimate_delay(tr[:,1], "mi_min")
    w2 = estimate_delay(tr[:,2], "mi_min")
    w3 = estimate_delay(tr[:,3], "mi_min")
    w = w1
    Tmax = 100

    @time Y, τ_vals, ts_vals, Ls , ε★ = pecuzal_embedding(tr[1:5000,:];
                                        τs = 0:Tmax , w = w, econ = true)

    @test length(ts_vals) == 5
    @test ts_vals[3] == ts_vals[4] == ts_vals[5] == 1
    @test ts_vals[1] == 3
    @test ts_vals[2] == 2
    @test τ_vals[1] == 0
    @test τ_vals[2] == 0
    @test τ_vals[3] == 62
    @test τ_vals[4] == 48
    @test τ_vals[5] == 0
    @test -0.9338 < Ls[1] < -0.9337
    @test -0.356 < Ls[2] < -0.355
    @test -0.1279 < Ls[3] < -0.1278
    @test -0.015 < Ls[4] < -0.014

    @time Y, τ_vals, ts_vals, Ls , ε★ = pecuzal_embedding(tr[1:5000,:];
                                        τs = 0:Tmax , w = w, econ = true, L_threshold = 0.2)
    @test -0.9338 < Ls[1] < -0.9337
    @test -0.356 < Ls[2] < -0.355
    @test size(Y,2) == 3


end

@testset "Dummy example" begin
    # Dummy input
    Random.seed!(1234)
    d1 = randn(1000)
    d2 = rand(1000)
    Tmax = 50
    dummy_set = Dataset(hcat(d1,d2))

    w1 = estimate_delay(d1, "mi_min")
    w2 = estimate_delay(d2, "mi_min")
    w = w1

    @time Y, τ_vals, ts_vals, Ls , ε★ = pecuzal_embedding(dummy_set;
                                    τs = 0:Tmax , w = w, econ = true)

    @test size(Y,2) == 1
end

end
