using DelayEmbeddings, DynamicalSystemsBase
using Test

println("\nTesting pecuzal_method.jl...")
@testset "PECUZAL" begin

lo = Systems.lorenz([1.0, 1.0, 50.0])
tr = trajectory(lo, 100; dt = 0.01, Ttr = 10)

@testset "Univariate example" begin

    s = vec(tr[:, 1]) # input timeseries = x component of lorenz
    w = estimate_delay(s, "mi_min")
    Tmax = 100
    K = 14
    samplesize = 1
    KNN = 3
    Tw = 56

    @time Y, τ_vals, ts_vals, Ls , εs = pecuzal_embedding(s[1:5000];
                                        τs = 0:Tmax , w = w, samplesize = samplesize,
                                        K = K, KNN = KNN, Tw = Tw)

    @test -2.698 < Ls[1] < -2.697
    @test -2.7795 < Ls[2] < -2.7794
    @test -2.73 < Ls[3] < -2.72

    @test τ_vals[2] == 18
    @test τ_vals[3] == 9

    @test length(ts_vals) == 3

end


@testset "Multivariate example" begin
## Test of the proposed Pecora-Uzal-embedding-method (multivariate case)

    w1 = estimate_delay(tr[:,1], "mi_min")
    w2 = estimate_delay(tr[:,2], "mi_min")
    w3 = estimate_delay(tr[:,3], "mi_min")
    w = w1
    Tmax = 100
    samplesize = 1

    @time Y, τ_vals, ts_vals, Ls , ε★ = pecuzal_embedding(tr[1:5000,:];
                                        τs = 0:Tmax , w = w, samplesize = samplesize)


    @test length(ts_vals) == 4
    @test ts_vals[3] == ts_vals[4] == 1
    @test ts_vals[1] == 2
    @test ts_vals[2] == 3
    @test τ_vals[2] == 0
    @test τ_vals[3] == 62
    @test τ_vals[4] == 0
    @test -2.19 < Ls[1] < -2.18
    @test -2.49 < Ls[2] < -2.48
    @test -2.57 < Ls[3] < -2.56

end

@testset "Dummy example" begin
    # Dummy input
    d1 = randn(1000)
    d2 = rand(1000)
    Tmax = 100
    samplesize = 1
    dummy_set = Dataset(hcat(d1,d2))

    w1 = estimate_delay(d1, "mi_min")
    w2 = estimate_delay(d2, "mi_min")
    w = w1

    @time Y, τ_vals, ts_vals, Ls , ε★ = pecuzal_embedding(dummy_set;
                                    τs = 0:Tmax , w = w, samplesize = samplesize)

    @test size(Y,2) == 1
end

end
