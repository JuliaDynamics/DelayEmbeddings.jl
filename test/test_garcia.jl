using DynamicalSystemsBase
using DelayEmbeddings
using Test
import Peaks

println("\nTesting garcia_almeida.jl...")
@testset "Garcia & Almeida method" begin

lo = Systems.lorenz()

tr = trajectory(lo, 80; dt = 0.01, Ttr = 10)

x = tr[:, 1]
Y = Dataset(x)
Y = regularize(Y)

@testset "N-statistic" begin
    τs = 0:100
    N , NN_distances = n_statistic(Y, x, w=0, T=1, τs = τs)

    T = 17
    N2 , NN_distances2 = n_statistic(Y, x, w=0, T=T, τs = τs)

    # check whether the `d_E1`-statistic is the same
    @test NN_distances[1][1:50] == NN_distances2[1][1:50]
    @test NN_distances[5][1:100] == NN_distances2[5][1:100]

    min_dist = 12
    (max_1_idx,_) = Peaks.findmaxima(N,min_dist)
    (max_2_idx,_) = Peaks.findmaxima(N2,min_dist)

    @test 30 ≤ max_1_idx[1] ≤ 35
    @test 30 ≤ max_2_idx[1] ≤ 35
    @test 69 ≤ max_1_idx[2] ≤ 75
    @test 69 ≤ max_2_idx[2] ≤ 75

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

    @test size(Y_act,2) == 2
    @test size(Y_act2,2) == 3
    @test 10 ≤ τ_vals[2] ≤ 18
    @test 1 ≤ τ_vals2[2] ≤ 3
    @test FNNs[end]<=0.05
    @test FNNs2[end]>FNNs2[end-1]

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

    @test size(Y_act,2) == size(Y_act2,2) == 3
    @test ts_vals[1] == ts_vals2[1] == ts_vals2[2] == ts_vals2[3] ==1
    @test ts_vals[2] == 2
    @test ts_vals[3] == 3
    @test 10 ≤ τ_vals[3] ≤ 18
    @test 1 ≤ τ_vals2[3] ≤ 3

    # try to reproduce Fig.2a in [^Garcia2005b]
    tra = Dataset(hcat(tr[:,1], tr[:,3]))
    tra = regularize(tra)
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
