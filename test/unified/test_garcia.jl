using DynamicalSystemsBase
using DelayEmbeddings
using Test, DelimitedFiles

println("\nTesting garcia_almeida.jl...")
@testset "Garcia & Almeida method" begin

# Test Lorenz example
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

x = tr[:, 1]
Y = StateSpaceSet(x)
Y = standardize(Y)

@testset "N-statistic" begin
    τs = 0:100
    N , NN_distances = n_statistic(Y, x, w=0, T=1, τs = τs)

    T = 17
    N2 , NN_distances2 = n_statistic(Y, x, w=0, T=T, τs = τs)

    # check whether the `d_E1`-statistic is the same
    @test NN_distances[1][1:50] == NN_distances2[1][1:50]
    @test NN_distances[5][1:100] == NN_distances2[5][1:100]

    (max_1_idx,_) = DelayEmbeddings.findlocalextrema(N)
    (max_2_idx,min_2_idx) = DelayEmbeddings.findlocalextrema(N2)

    @test 22 ≤ max_1_idx[5] ≤ 24
    @test 35 ≤ max_1_idx[6] ≤ 37
    @test 79 ≤ max_1_idx[12] ≤ 81

    @test 11 ≤ min_2_idx[1] ≤ 13
    @test 36 ≤ max_2_idx[3] ≤ 38
    @test 79 ≤ max_2_idx[7] ≤ 81

    # # plot N-Statistic for the Lorenz system as in Fig. 2(a) in [^Garcia2005a]
    # using Plots
    # plot(N,linewidth = 2, label = "T=1 (as in the Paper)", xaxis=:log)
    # plot!(N2,linewidth = 2, label = "T=$T", xaxis=:log)
    # plot!(N,seriestype = :scatter, label = "",xaxis=:log)
    # plot!(N2,seriestype = :scatter, label = "",xaxis=:log)
    # plot!(title = "Lorenz System as in Fig. 2(a) in the Paper")
    # xlabel!("τ")
    # ylabel!("N-Statistic")
end

@testset "garcia_embed univariate" begin
    Y_act, τ_vals, ts_vals, FNNs, NS = garcia_almeida_embedding(x; τs=0:100,  w = 17, T = 17)
    Y_act2, τ_vals2, ts_vals2, FNNs2, NS2 = garcia_almeida_embedding(x; τs=0:100,  w = 1, T = 1)
    Y_act3, τ_vals3, ts_vals3, FNNs3, NS3 = garcia_almeida_embedding(x; τs=0:100,  w = 17, T = 17, fnn_thres=0.1)

    @test size(Y_act,2) == 3
    @test size(Y_act2,2) == 3
    @test 15 ≤ τ_vals[2] ≤ 17
    @test 1 ≤ τ_vals2[2] ≤ 3
    @test FNNs[1] ≥ FNNs[2] ≤ FNNs[3]
    @test FNNs2[end]>FNNs2[end-1]
    @test size(Y_act3,2) == 2
    @test τ_vals3 == τ_vals[1:2]

    # using Plots
    # plot3d(Y_act[:,1],Y_act[:,2],Y_act[:,3], marker=2, camera = (6, 4))
    # plot!(title = "Reconstruction of Lorenz attractor using Garcia&Almeida univariate embedding")
    #
    # plot3d(Y_act2[:,1],Y_act2[:,2],Y_act2[:,3], marker=2, camera = (6, 4))
    # plot!(title = "Reconstruction of Lorenz attractor using Garcia&Almeida univariate embedding as in the paper")
end


@testset "garcia_embed multivariate" begin

    Y_act, τ_vals, ts_vals, FNNs, NS = garcia_almeida_embedding(tr; τs=0:100,  w = 17, T = 17)
    Y_act2, τ_vals2, ts_vals2, FNNs2, NS2 = garcia_almeida_embedding(tr; τs=0:100,  w = 1, T = 1)

    @test size(Y_act,2) == 5
    @test size(Y_act2,2) == 3
    @test ts_vals[1] == ts_vals[4] == 1
    @test ts_vals[2] == ts_vals[5] == 3
    @test ts_vals[3] == 2
    @test τ_vals == [0, 5, 5, 38, 55]

    @test ts_vals2 == [1, 1, 1]
    @test τ_vals2 == [0, 1, 2]

    # try to reproduce Fig.2a in [^Garcia2005b]
    tra = StateSpaceSet(hcat(tr[:,1], tr[:,3]))
    tra = standardize(tra)
    taus = 0:100

    N , _ = n_statistic(tra, tra[:,1], w=17, T=17, τs = taus)
    N2 , _ = n_statistic(tra, tra[:,2], w=17, T=17, τs = taus)

    _, minis2 = findmin(N2)

    @test 20 ≤ taus[minis2] ≤ 30

    # using Plots
    # NN = hcat(N,N2)
    # plot(NN, xaxis=:log)
    # plot(NN)

end


end
