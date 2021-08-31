using DelayEmbeddings
using Test
using Random
using DelimitedFiles

println("\nTesting uzal_cost.jl...")
@testset "Uzal cost" begin
@testset "Random vectors" begin
    ## Check on random vector
    Random.seed!(1516578735)
    tr1 = randn(10000)
    tr2 = randn(5000)
    tr1 = Dataset(tr1)
    tr2 = Dataset(tr2)
    L = uzal_cost(tr1;
        Tw = 60, K= 3, w = 1, samplesize = 1.0,
        metric = Euclidean()
    )
    L2 = uzal_cost(tr2;
        Tw = 60, K= 3, w = 1, samplesize = 1.0,
        metric = Euclidean()
    )

    L_max = -0.06
    L_min = -0.08
    @test L_min < L < L_max
    @test L_min < L2 < L_max


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
    @test -1.6 < L < -1.5
end

@testset "Uzal local cost (Lorenz)" begin
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
    @test -0.3 < L < -0.28

end

@testset "Roessler system" begin
    ## Test Roessler example as in Fig. 7 in the paper with internal data
    # For comparison reasons using Travis CI we carry out the integration on a UNIX
    # OS and save the resulting time series
    # See https://github.com/JuliaDynamics/JuliaDynamics for the storage of
    # the time series used for testing
    #
    # u0 = [1.0, 1.0, 1.0]
    # ro = Systems.roessler(u0; a=0.15, b = 0.2, c=10)
    # tr = trajectory(ro, 1000; Δt = 0.1, Ttr = 100)
    tr = readdlm(joinpath(tsfolder, "test_time_series_roessler_N_10000_multivariate.csv"))
    tr = Dataset(tr)

    x = tr[:, 1]

    # theiler window
    w  = 12
    # Time horizon
    Tw = 50
    # sample size
    SampleSize = .5
    # metric
    metric = Euclidean()
    # embedding dimension
    m = 3
    # maximum neighbours
    k_max = 5
    # number of considered tw-values
    tw_max = 15 # corresponds to a maximum of tw = 30

    # preallocation
    L = zeros(k_max,tw_max)
    tws = zeros(tw_max)

    for K = 1:k_max
        for i = 1:tw_max
            tws[i] = i*(m-1)
            Y = embed(x,m,i)
            L[K,i] = uzal_cost(Y; Tw=Tw, K=K, w=w, samplesize=SampleSize, metric=metric)
        end
    end

    tau_min = 7
    tau_max = 12

    min1_idx = sortperm(L[1,:])
    min1 = tws[min1_idx[1]]
    @test tau_min < min1 < tau_max
    L_min = -1.2
    L_max = -0.8
    @test L_min < L[1,min1_idx[1]] < L_max


    min2_idx = sortperm(L[2,:])
    min2 = tws[min2_idx[1]]
    @test tau_min < min2 < tau_max
    L_min = -0.79
    L_max = -0.75
    @test L_min < L[2,min2_idx[1]] < L_max


    min3_idx = sortperm(L[3,:])
    min3 = tws[min3_idx[1]]
    @test tau_min < min3 < tau_max
    L_min = -0.62
    L_max = -0.58
    @test L_min < L[3,min3_idx[1]] < L_max


    min4_idx = sortperm(L[4,:])
    min4 = tws[min4_idx[1]]
    @test tau_min < min4 < tau_max
    L_min = -0.51
    L_max = -0.47
    @test L_min < L[4,min4_idx[1]] < L_max

    min5_idx = sortperm(L[5,:])
    min5 = tws[min5_idx[1]]
    @test tau_min < min5 < tau_max
    L_min = -0.43
    L_max = -0.39
    @test L_min < L[5,min5_idx[1]] < L_max

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
    # labels = ["k=1", "k=2", "k=3", "k=4", "k=5"]
    # figure()
    # plot(tws,L[1,:], linewidth = 2)
    # plot(tws,L[2,:], linewidth = 2)
    # plot(tws,L[3,:], linewidth = 2)
    # plot(tws,L[4,:], linewidth = 2)
    # plot(tws,L[5,:], linewidth = 2)
    # xlabel(L"$t_w$")
    # ylabel(L"$L_k$")
    # legend(labels)
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

end # Uzal cost testset
