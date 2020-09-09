using DynamicalSystemsBase
using DelayEmbeddings
using Test
using DelimitedFiles

println("\nTesting mdop_embedding.jl...")
@testset "Nichkawde method MDOP" begin

# For comparison reasons using Travis CI we carry out the integration on a UNIX
# OS and save the resulting time series
# # solve Mackey-Glass-Delay Diff.Eq. as in the Paper
# function mackey_glass(du,u,h,p,t)
#   beta,n,gamma,tau = p
#   hist = h(p, t-tau)[1]
#   du[1] = (beta*hist)/(1+hist^n) - gamma * u[1]
# end
# # set parameters
# h(p,t) = 0
# tau_d = 44
# n = 10
# β = 0.2
# γ = 0.1
# δt = 0.5
# p = (β,n,γ,tau_d)
#
# # time span
# tspan = (0.0, 12000.0)
# u0 = [1.0]
#
# prob = DDEProblem(mackey_glass,u0,h,tspan,p; constant_lags=tau_d)
# alg = MethodOfSteps(Tsit5())
# sol = solve(prob,alg; adaptive=false, dt=δt)
#
# s = [u[1] for u in sol.u]
# s = s[4001:end]

s = readdlm(joinpath(tsfolder, "1.csv"))
s = vec(s)
Y = Dataset(s)

theiler = 57

@testset "beta statistic" begin
    ## Test beta_statistic (core algorithm of mdop_embedding)

    taus = 0:100
    β = @inferred DelayEmbeddings.beta_statistic(Y, s, taus, theiler)
    maxi, max_idx = findmax(β)

    @test maxi>4.1
    @test taus[max_idx]>=50

    # # display results as in Fig. 3 of the paper
    # using Plots
    # plot(taus, β, linewidth = 3, label = "1st embedding cycle")
    # plot!(title = "β-statistic for Mackey Glass System as in Fig. 3 in the Paper")
    # xlabel!("τ")
    # ylabel!("β-Statistic")

    # # display results as in Fig. 3 of the paper
    # using PyPlot
    # pygui(true)
    # figure()
    # plot(taus, β, linewidth = 3, label = "1st embedding cycle")
    # scatter(max_idx-1,maxi, c="red")
    # title("β-statistic for Mackey Glass System as in Fig. 3 in the Paper")
    # xlabel("τ")
    # ylabel("β-Statistic")
    # xticks(0:10:100)
    # grid()

    # test different tau range
    taus2 = 1:4:100
    β2 = @inferred beta_statistic(Y, s, taus2, theiler)
    maxi2, max_idx2 = findmax(β2)

    @test maxi2>4.1
    @test taus2[max_idx2]>=40

    # # display results as in Fig. 3 of the paper
    # using Plots
    # plot(taus2, β2, linewidth = 3, label = "1st embedding cycle")
    # plot!(title = "coarse β-statistic for Mackey Glass System as in Fig. 3 in the Paper")
    # xlabel!("τ")
    # ylabel!("β-Statistic")

end

@testset "mdop_embedding univariate" begin

    taus = 0:100
    β = DelayEmbeddings.beta_statistic(Y, s, taus, theiler)
    Y, τ_vals, ts_vals, FNNs, betas = mdop_embedding(s; τs = taus, w = theiler)
    # for different τs
    taus2 = 1:4:100
    Y2, τ_vals2, ts_vals2, FNNs2, betas2 = mdop_embedding(s; τs = taus2, w = theiler)

    @test round.(β, digits=6) == round.(betas[:,1], digits=6)
    @test size(Y,2) == 5
    @test size(Y,2) == size(Y2,2)
    @test sum(findall(x -> x != 1, ts_vals))==0
    @test sum(abs.(diff(τ_vals)) .< 10) == 0


    # # display results as in Fig. 3 of the paper
    # using Plots
    # # Figure as in Fig.3 in the paper
    # plot(taus, betas[:,1], linewidth = 3, label = "embedding cycle 1")
    # plot!([taus[τ_vals[2]+1]],[betas[τ_vals[2]+1,1]], seriestype = :scatter, color="red", label = "")
    # for i = 2:size(betas,2)
    #   plot!(taus, betas[:,i], linewidth = 3, label = "embedding cycle $i")
    #   plot!([taus[τ_vals[i+1]+1]],[betas[τ_vals[i+1]+1,i]], seriestype = :scatter, color="red", label = "")
    # end
    # plot!(title = "β-statistic's for each embedding cycle of Mackey Glass System as in Fig. 3")
    # xlabel!("delay τ")
    # ylabel!("log10 β(τ)")
    #
    # # Figure of coarse grained analysis
    # taus22 = zeros(length(taus2))
    # [taus22[i]=taus2[i] for i = 1: length(taus2)]
    # plot(taus22, betas2[:,1], linewidth = 3, label = "embedding cycle 1")
    # trueind = findall(x -> x == τ_vals2[2], taus22)
    # plot!(taus22[trueind],[betas2[trueind,1]], seriestype = :scatter, color="red", label = "")
    # for i = 2:size(betas2,2)
    #   plot!(taus22, betas2[:,i], linewidth = 3, label = "embedding cycle $i")
    #   trueind = findall(x -> x == τ_vals2[i+1], taus22)
    #   plot!(taus22[trueind],[betas2[trueind,i]], seriestype = :scatter, color="red", label = "")
    # end
    # plot!(title = "β-statistic's for each embedding cycle of Mackey Glass System as in Fig. 3")
    # xlabel!("delay τ")
    # ylabel!("log10 β(τ)")

end

@testset "estimate τ max (Roessler)" begin
    # For comparison reasons using Travis CI we carry out the integration on a UNIX
    # OS and save the resulting time series
    # roe = Systems.roessler([1.0, 0, 0]; a=0.2, b=0.2, c=5.7)
    # sroe = trajectory(roe, 500; dt = 0.05, Ttr = 100.0)
    # writedlm("2.csv", sroe)

    sroe = readdlm(joinpath(tsfolder, "2.csv"))
    tws = 25:32

    τ_m, L = @inferred mdop_maximum_delay(sroe[:, 2], tws)
    @test τ_m == 26
    τ_m, Ls = @inferred mdop_maximum_delay(Dataset(sroe[:, 1:2]), tws)
    @test τ_m == 26

    # # reproduce Fig.2 of the paper
    # tws = 1:2:101
    # τ_m, L = DelayEmbeddings.mdop_maximum_delay(s[:,2]; tw = tws, samplesize=1.0)
    #
    # using Plots
    # twss = zeros(length(tws))
    # [twss[cnt] = i for (cnt,i) in enumerate(tws)]
    # plot(twss,L, label="")
    # xlabel!("time window")
    # ylabel!("L")

    # tw=1:4:200
    # tau_max, LL = DelayEmbeddings.mdop_maximum_delay(sroe; tw=tw)
    #
    # using Plots
    # gui()
    # twss = zeros(length(tw))
    # [twss[cnt] = i for (cnt,i) in enumerate(tw)]
    # plot(twss,LL, label="")
    # xlabel!("time window")
    # ylabel!("L")
end

@testset "mdop_embedding multivariate" begin
    # For comparison reasons using Travis CI we carry out the integration on a UNIX
    # OS and save the resulting time series
    # roe = Systems.roessler([1.0, 0, 0]; a=0.2, b=0.2, c=5.7)
    # sroe = trajectory(roe, 500; dt = 0.05, Ttr = 100.0)
    # writedlm("2.csv", sroe)

    sroe = readdlm(joinpath(tsfolder, "2.csv"))
    tra = Dataset(sroe)
    w1 = estimate_delay(sroe[:,1], "mi_min")
    w2 = estimate_delay(sroe[:,2], "mi_min")

    theiler = w2
    taus = 0:26
    mc = 10

    Y, τ_vals, ts_vals, FNNs, betas =  mdop_embedding(sroe[:,1]; τs = taus, w = theiler, max_num_of_cycles = mc)

    max_idx, ts_number = @inferred DelayEmbeddings.choose_optimal_tau2(betas)
    @test ts_number == 1
    @test taus[max_idx] == τ_vals[2]

    Y2, τ_vals2, ts_vals2, FNNs2, betas2 = mdop_embedding(tra; τs = taus, w = theiler, max_num_of_cycles = mc)
    ttra = regularize(tra)
    b1 = DelayEmbeddings.beta_statistic(Dataset(ttra[:,ts_vals2[1]]), ttra[:,1], taus, theiler)
    b2 = DelayEmbeddings.beta_statistic(Dataset(ttra[:,ts_vals2[1]]), ttra[:,2], taus, theiler)
    b3 = DelayEmbeddings.beta_statistic(Dataset(ttra[:,ts_vals2[1]]), ttra[:,3], taus, theiler)

    @test betas2[1][:,1] == b1
    @test betas2[1][:,2] == b2
    @test betas2[1][:,3] == b3

    @test size(Y2,2) == 3
    @test τ_vals2[1] == τ_vals2[2] == 0
    @test τ_vals2[3] == maximum(taus)

    @test ts_vals2[2] == ts_vals2[3] == 2
    @test ts_vals2[1] == 3

end

end
