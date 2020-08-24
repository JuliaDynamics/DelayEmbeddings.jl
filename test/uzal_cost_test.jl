using DynamicalSystemsBase
using DelayEmbeddings
using Test
using Random
using DelimitedFiles

println("\nTesting uzal_cost.jl...")
@testset "Uzal cost" begin
@testset "Random vectors" begin
    ## Check on random vector
    Random.seed!(1516578735)
    tr = randn(10000)
    tr = Dataset(tr)
    L = uzal_cost(tr;
        Tw = 60, K= 3, w = 1, samplesize = 1.0,
        metric = Euclidean()
    )

    L_max = -2.059
    @test L < L_max


    ## check on clean step function (0-distances)
    tr = zeros(10000)
    tr[1:5000] = zeros(5000)
    tr[5001:end] = ones(5000)
    tr = Dataset(tr)
    L = uzal_cost(tr;
        Tw = 60, K= 3, w = 1, samplesize = 1.0,
        metric = Euclidean()
    )
    @test isnan(L)

    ## check on noisy step function (should yield a proper value)
    Random.seed!(1516578735)
    tr = zeros(10000)
    tr[1:5000] = zeros(5000) .+ 0.001 .* randn(5000)
    tr[5001:end] = ones(5000) .+ 0.001 .* randn(5000)
    tr = Dataset(tr)
    L = uzal_cost(tr;
        Tw = 60, K= 3, w = 1, samplesize = 1.0,
        metric = Euclidean()
    )
    @test L<-3
end

@testset "Lorenz system" begin
    ## Simple Check on Lorenz System

    # For comparison reasons using Travis CI we carry out the integration on a UNIX
    # OS and save the resulting time series
    # lo = Systems.lorenz([0, 10.0, 0.0])
    # tr = trajectory(lo, 10; dt = 0.01, Ttr = 10)
    # writedlm("lorenz_uzal_1.csv", tr)

    tr = readdlm(joinpath(tsfolder, "3.csv"))
    tr = Dataset(tr)

    # check Euclidean metric
    L = uzal_cost(tr;
        Tw = 60, K= 3, w = 12, samplesize = 1.0,
        metric = Euclidean()
    )
    L_max = -1.8
    L_min = -2
    @test L_min < L < L_max
    @test L ≈ -1.8390494512811177

    # check Maximum metric
    L = uzal_cost(tr;
        Tw = 60, K= 3, w = 12, samplesize = 1.0,
        metric = Chebyshev()
    )
    @test L_min < L < L_max
    # using BenchmarkTools
    # @btime uzal_cost($tr; Tw = 60, K = 3, w = 12, samplesize = 1.0, metric = Euclidean())
end

@testset "Uzal local cost (Lorenz)" begin
    tr = readdlm(joinpath(tsfolder, "3.csv"))
    tr = Dataset(tr)

    # check local cost function output
    Tw = 60
    L = uzal_cost(tr;
        Tw = Tw, K= 3, w = 12, samplesize = 1.0,
        metric = Euclidean()
    )
    L_local= uzal_cost_local(tr; Tw = Tw, K= 3, samplesize = 1.0,w = 12, metric = Euclidean())
    @test length(L_local) == length(tr)-Tw
    @test maximum(L_local)>L
    @test minimum(L_local)<L
end

@testset "Roessler system" begin
    # For comparison reasons using Travis CI we carry out the integration on a UNIX
    # OS and save the resulting time series

    ## Test Roessler example as in Fig. 7 in the paper with internal data
    # ro = Systems.roessler([1.0, 1.0, 1.0], a=0.15, b = 0.2, c=10)
    # tr = trajectory(ro, 1250; dt = 0.2, Ttr = 100)
    # writedlm("4.csv", tr)

    tr = readdlm(joinpath(tsfolder, "4.csv"))
    tr = Dataset(tr)
    x = tr[:, 1]

    # theiler window
    w  = 12
    # Time horizon
    Tw = 80
    # sample size
    SampleSize = .5
    # metric
    metric = Euclidean()
    # embedding dimension
    m = 3
    # maximum neighbours
    k_max = 4
    # number of total trials
    trials = 8

    # preallocation
    L = zeros(k_max,trials)
    tw_max = zeros(trials)

    for K = 1:k_max
        for i = 1:trials
            tw_max[i] = i*(m-1)
            Y = embed(x,m,i)
            L[K,i] = uzal_cost(Y; Tw=Tw, K=K, w=w, samplesize=SampleSize, metric=metric)
        end
    end

    tau_min = 4
    tau_max = 8

    min1_idx = sortperm(L[1,:])
    min1 = tw_max[min1_idx[1]]
    @test tau_min < min1 < tau_max
    L_min = -2.7
    L_max = -2.3
    @test L_min < L[1,min1_idx[1]] < L_max


    min2_idx = sortperm(L[2,:])
    min2 = tw_max[min2_idx[1]]
    @test tau_min < min2 < tau_max
    L_min = -2.3
    L_max = -2.1
    @test L_min < L[2,min2_idx[1]] < L_max


    min3_idx = sortperm(L[3,:])
    min3 = tw_max[min3_idx[1]]
    @test tau_min < min3 < tau_max
    L_min = -2.1
    L_max = -2.0
    @test L_min < L[3,min3_idx[1]] < L_max


    min4_idx = sortperm(L[4,:])
    min4 = tw_max[min4_idx[1]]
    @test tau_min < min4 < tau_max
    L_min = -2.0
    L_max = -1.9
    @test L_min < L[4,min4_idx[1]] < L_max

    L_local = uzal_cost_local(tr;
        Tw = Tw, K= 3, w = 12, samplesize=1.0,
        metric = Chebyshev()
    )
    L = uzal_cost(tr;
        Tw = Tw, K= 3, w = 12,
        metric = Chebyshev(), samplesize=1.0
    )
    @test length(L_local) == length(tr)-Tw
    @test maximum(L_local) > L
    @test minimum(L_local) < L

    # # plot results using PyPlot
    # using PyPlot
    # pygui(true)
    # labels = ["k=1", "k=2", "k=3", "k=4"]
    # figure()
    # plot(tw_max,L[1,:], linewidth = 2)
    # plot(tw_max,L[2,:], linewidth = 2)
    # plot(tw_max,L[3,:], linewidth = 2)
    # plot(tw_max,L[4,:], linewidth = 2)
    # xlabel(L"$t_w$")
    # ylabel(L"$L_k$")
    # legend(labels)
    # xticks(0:1:16)
    # #yscale("symlog")
    # title("Roessler System as in Fig. 7(b) in the Uzal Paper")
    # grid()

    # # Display local cost function for two different embeddings
    # tau_value_1 = 8
    # tau_value_2 = 20
    # m = 2
    #
    # # embedding in two dimensions
    # Y_1 = embed(x,m,tau_value_1)
    # Y_2 = embed(x,m,tau_value_2)
    #
    # # compute local cost functions
    # L_local_1= uzal_cost_local(Y_1;
    #     Tw = Tw, K= 3, w = 12, metric = Euclidean()
    # )
    # L1= uzal_cost(Y_1;
    #     Tw = Tw, K= 3, w = 12, metric = Euclidean(), samplesize=1.0
    # )
    #
    # L_local_2= uzal_cost_local(Y_2;
    #     Tw = Tw, K= 3, w = 12, metric = Euclidean()
    # )
    # L2= uzal_cost(Y_2;
    #     Tw = Tw, K= 3, w = 12, metric = Euclidean(), samplesize=1.0
    # )
    # # plot results using PyPlot
    # using PyPlot
    # pygui(true)
    #
    # x_val1 = Y_1[1:length(L_local_1),1]
    # y_val1 = Y_1[1:length(L_local_1),2]
    #
    # x_val2 = Y_2[1:length(L_local_2),1]
    # y_val2 = Y_2[1:length(L_local_2),2]
    #
    # markersize = 10
    #
    # figure()
    # subplot(1,2,1)
    # scatter(x_val1,y_val1,c=L_local_1,s=markersize)
    # plot(x_val1,y_val1,linewidth=0.1)
    # xlabel("x(t)")
    # ylabel("x(t+$tau_value_1)")
    # title("Local L for Roessler System first embedding cycle (τ=$tau_value_1)")
    # grid()
    #
    # subplot(1,2,2)
    # scatter(x_val2,y_val2,c=L_local_2,s=markersize)
    # plot(x_val2,y_val2,linewidth=0.1)
    # xlabel("x(t)")
    # ylabel("x(t+$tau_value_2)")
    # title("Local L for Roessler System first embedding cycle (τ=$tau_value_2)")
    # grid()
    # cbar = colorbar()
    # cbar.set_label("Local cost function")

end

## Test Lorenz example as in Fig. 7 in the paper with internal data
@testset "Lorenz fig. 7." begin
    lo = Systems.lorenz([1.0, 1.0, 50.0])
    tr = trajectory(lo, 100; dt = 0.01, Ttr = 10)
    x = tr[:, 1]

    # theiler window
    w  = 12
    # Time horizon
    Tw = 80
    # sample size
    SampleSize = 1.0
    # metric
    metric = Euclidean()
    # embedding dimension
    m = 3
    # maximum neighbours
    k_max = 4
    # number of total trials
    trials = 16

    # preallocation
    L = zeros(k_max,trials)
    tw_max = zeros(trials)

    for K = 1:k_max
        for i = 1:trials
            tw_max[i] = i*(m-1)
            Y = embed(x,m,i)
            L[K,i] = uzal_cost(Y; Tw=Tw, K=K, w=w, samplesize=SampleSize, metric=metric)
        end
    end

    tau_min = 9
    tau_max = 11
    min1_idx = sortperm(L[1,:])
    min1 = tw_max[min1_idx[1]]
    @test tau_min < min1 < tau_max
    L_max = -2.3
    @test L[1,min1_idx[1]] < L_max

    tau_min = 22
    tau_max = 25
    min2_idx = sortperm(L[2,:])
    min2 = tw_max[min2_idx[1]]
    @test tau_min < min2 < tau_max
    L_max = -2.0
    @test L[2,min2_idx[1]] < L_max

    tau_min = 19
    tau_max = 22
    min3_idx = sortperm(L[3,:])
    min3 = tw_max[min3_idx[1]]
    @test tau_min < min3 < tau_max
    L_max = -1.9
    @test L[3,min3_idx[1]] < L_max

    tau_min = 20
    tau_max = 24
    min4_idx = sortperm(L[4,:])
    min4 = tw_max[min4_idx[1]]
    @test tau_min < min4 < tau_max
    L_max = -1.8
    @test L[4,min4_idx[1]] < L_max

    # # plot results using PyPlot
    # using PyPlot
    # pygui(true)
    # labels = ["k=1", "k=2", "k=3", "k=4"]
    # figure()
    # plot(tw_max,L[1,:], linewidth = 2)
    # plot(tw_max,L[2,:], linewidth = 2)
    # plot(tw_max,L[3,:], linewidth = 2)
    # plot(tw_max,L[4,:], linewidth = 2)
    # xlabel(L"$t_w$")
    # ylabel(L"$L_k$")
    # legend(labels)
    # xticks(0:2:32)
    # #yscale("symlog")
    # title("Lorenz System as in Fig. 7(a) in the Uzal Paper")
    # grid()
end


end # Uzal cost testset
