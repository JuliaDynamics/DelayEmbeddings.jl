using DelayEmbeddings
using Test, Random, DelimitedFiles

println("\nTesting pecuzal_method.jl...")
@testset "PECUZAL" begin

## Test Lorenz example
# For comparison reasons using Travis CI we carry out the integration on a UNIX
# OS and save the resulting time series
# See https://github.com/JuliaDynamics/JuliaDynamics for the storage of
# the time series used for testing
#
# u0 = [0, 10.0, 0.0]
# lo = Systems.lorenz(u0; σ=10, ρ=28, β=8/3)
# tr = trajectory(lo, 100; Δt = 0.01, Ttr = 100)
tr = readdlm(joinpath(tsfolder, "test_time_series_lorenz_standard_N_10000_multivariate.csv"))
tr = StateSpaceSet(tr)

@testset "Univariate example" begin

    s = vec(tr[:, 1]) # input timeseries = x component of lorenz
    w = estimate_delay(s, "mi_min")
    Tmax = 100

    @time Y, τ_vals, ts_vals, Ls , εs = pecuzal_embedding(s[1:5000];
                                        τs = 0:Tmax , w = w, L_threshold = 0.05, econ=true)

    @test -1.08 < Ls[1] < -1.04
    @test -0.51 < Ls[2] < -0.47
    @test -0.18 < Ls[3] < -0.14

    @test τ_vals[2] == 19
    @test τ_vals[3] == 96
    @test τ_vals[4] == 80
    @test length(ts_vals) == 4

    # lower samplesize
    Random.seed!(123)
    @time Y2, τ_vals2, ts_vals2, Ls2 , εs2 = pecuzal_embedding(s[1:5000];
                                        τs = 0:Tmax , w = w, L_threshold = 0.05, econ=true, samplesize = 0.5)
    @test Ls[1] - .1 < Ls2[1] < Ls[1] + .1
    @test Ls[2] - .1 < Ls2[2] < Ls[2] + .1
    @test Ls[3] - .1 < Ls2[3] < Ls[3] + .1

    @test τ_vals2[2] == 19
    @test τ_vals2[3] == 97
    @test τ_vals2[4] == 62
    @test length(ts_vals2) == 4

    @time Y, τ_vals, ts_vals, Ls , εs = pecuzal_embedding(s;
                                        τs = 0:Tmax , w = w, L_threshold = 0.05)
    @test -0.95 < Ls[1] < -0.91
    @test -0.48 < Ls[2] < -0.44

    @test τ_vals[2] == 18
    @test τ_vals[3] == 9
    @test length(ts_vals) == 3

    YY1 = DelayEmbeddings.hcat_lagged_values(StateSpaceSet(s), s, 21)
    YY2 = DelayEmbeddings.hcat_lagged_values(YY1, s, 13)
    @test length(YY1) == length(YY2)
    @test YY1 == YY2[:,1:2]

end


@testset "Multivariate example" begin
    ## Test of the proposed PECUZAL-embedding-method (multivariate case)

    w1 = estimate_delay(tr[:,1], "mi_min")
    w2 = estimate_delay(tr[:,2], "mi_min")
    w3 = estimate_delay(tr[:,3], "mi_min")
    w = w1
    Tmax = 100

    @time Y, τ_vals, ts_vals, Ls , ε★ = pecuzal_embedding(tr[1:5000,:];
                                        τs = 0:Tmax , w = w, econ = true)

    @test length(ts_vals) == 4
    @test ts_vals[2] == ts_vals[3] == ts_vals[4] == 1
    @test ts_vals[1] == 3
    @test τ_vals[1] == 0
    @test τ_vals[2] == 9
    @test τ_vals[3] == 64
    @test τ_vals[4] == 53
    @test -1.40 < Ls[1] < -1.36
    @test -0.76 < Ls[2] < -0.72
    @test -0.1 < Ls[3] < -0.06

    # less fiducial points for computation
    Random.seed!(123)
    @time Y2, τ_vals2, ts_vals2, Ls2 , ε★2 = pecuzal_embedding(tr[1:5000,:];
                                        τs = 0:Tmax , w = w, econ = true, samplesize = 0.5)

    @test length(ts_vals2) == 4
    @test ts_vals2[2] == ts_vals2[3] == ts_vals2[4] == 1
    @test ts_vals2[1] == 3
    @test τ_vals2[1] == 0
    @test τ_vals2[2] == 7
    @test τ_vals2[3] == 61
    @test τ_vals2[4] == 51
    @test Ls[1] - .1 < Ls2[1] < Ls[1] + .1
    @test Ls[2] - .1 < Ls2[2] < Ls[2] + .1
    @test Ls[3] - .1 < Ls2[3] < Ls[3] + .1

    # L-threshold
    @time Y, τ_vals, ts_vals, Ls , ε★ = pecuzal_embedding(tr[1:5000,:];
                                        τs = 0:Tmax , w = w, econ = true, L_threshold = 0.2)
    @test -1.40 < Ls[1] < -1.36
    @test -0.76 < Ls[2] < -0.72
    @test size(Y,2) == 3

end

@testset "Dummy example" begin
    # Dummy input
    Random.seed!(1234)
    d1 = randn(1000)
    d2 = rand(1000)
    Tmax = 50
    dummy_set = StateSpaceSet(hcat(d1,d2))

    w1 = estimate_delay(d1, "mi_min")
    w2 = estimate_delay(d2, "mi_min")
    w = w1

    @time Y, τ_vals, ts_vals, Ls , ε★ = pecuzal_embedding(dummy_set;
                                    τs = 0:Tmax , w = w, econ = true)

    @test size(Y,2) == 1
end

end
