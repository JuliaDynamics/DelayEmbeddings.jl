using DynamicalSystemsBase
using DelayEmbeddings
using Test
import Peaks

println("\nTesting garcia_almeida.jl...")
@testset "Garcia & Almeida method" begin

lo = Systems.lorenz()

tr = trajectory(lo, 60; dt = 0.01, Ttr = 10)

x = tr[:, 1]
Y = Dataset(x)

@testset "N-statistic" begin
    τs = 0:50
    N , NN_distances = n_statistic(Y, x, w=0, T=1, τs = τs)

    T = 17
    N2 , NN_distances2 = n_statistic(Y, x, w=0, T=T, τs = τs)

    # check whether the `d_E1`-statistic is the same
    @test NN_distances[1][1:50] == NN_distances2[1][1:50]
    @test NN_distances[5][1:100] == NN_distances2[5][1:100]

    min_dist = 12
    max_1_idx = Peaks.maxima(N,min_dist)
    max_2_idx = Peaks.maxima(N2,min_dist)

    @test max_1_idx == max_2_idx

    # # plot N-Statistic for the Lorenz system as in Fig. 2(a) in [^Garcia2005b]
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

    @test size(Y_act,2)==size(Y_act2,2)
    @test 10 ≤ τ_vals[2] ≤ 18
    @test 1 ≤ τ_vals2[2] ≤ 3
    @test FNNs[end]<=0.05
    @test FNNs2[end]<=0.05

    # using Plots
    # plot3d(Y_act[:,1],Y_act[:,2],Y_act[:,3], marker=2, camera = (6, 4))
    # plot!(title = "Reconstruction of Lorenz attractor using Garcia&Almeida univariate embedding")
    #
    # plot3d(Y_act2[:,1],Y_act2[:,2],Y_act2[:,3], marker=2, camera = (6, 4))
    # plot!(title = "Reconstruction of Lorenz attractor using Garcia&Almeida univariate embedding as in the paper")
end


#@testset "garcia_embed multivariate" begin




#end


end
