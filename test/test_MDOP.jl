# test MDOP
using DynamicalSystemsBase
using DelayEmbeddings
using StatsBase
using Statistics
using Random
using Test
using Peaks
using DelimitedFiles
using DifferentialEquations

println("\nTesting MDOP.jl...")
@testset "Nichkawde method" begin

# solve Mackey-Glass-Delay Diff.Eq. as in the Paper
function mackey_glass(du,u,h,p,t)
  beta,n,gamma,tau = p
  hist = h(p, t-tau)[1]
  du[1] = (beta*hist)/(1+hist^n) - gamma * u[1]
end
# set parameters
h(p,t) = 0
tau_d = 44
n = 10
β = 0.2
γ = 0.1
δt = 0.5
p = (β,n,γ,tau_d)

# time span
tspan = (0.0, 12000.0)
u0 = [1.0]

prob = DDEProblem(mackey_glass,u0,h,tspan,p; constant_lags=tau_d)
alg = MethodOfSteps(Tsit5())
sol = solve(prob,alg; adaptive=false, dt=δt)

s = sol.u
s = s[4001:end]
ss = zeros(length(s))
[ss[i] = s[i][1] for i in 1:length(s)]

Y = Dataset(s)
theiler = 57

@testset "beta statistic" begin
## Test beta_statistic (core algorithm of MDOP)


taus = 0:100
β = DelayEmbeddings.beta_statistic(Y, ss; τs = taus, w = theiler)

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
β2 = DelayEmbeddings.beta_statistic(Y, ss; τs = taus2, w = theiler)

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


@testset "estimate tau max for MDOP" begin
roe = Systems.roessler([0.1;0;0])
s = trajectory(roe, 500; dt = 0.05, Ttr = 10.0)

tws = 32:36
τ_m, L = DelayEmbeddings.estimate_maximum_delay(s[:,2]; tw = tws, samplesize=1.0)
@test τ_m == 34
τ_m, Ls = DelayEmbeddings.estimate_maximum_delay(s[:,1:2]; tw = tws, samplesize=1.0)
@test τ_m == 34

# # reproduce Fig.2 of the paper
# tws = 1:2:101
# τ_m, L = DelayEmbeddings.estimate_maximum_delay(s[:,2]; tw = tws, samplesize=1.0)
#
# using Plots
# twss = zeros(length(tws))
# [twss[cnt] = i for (cnt,i) in enumerate(tws)]
# plot(twss,L, label="")
# xlabel!("time window")
# ylabel!("L")
end


@testset "MDOP univariate" begin
## test MDOP() univariate

# tw=1:4:200
# tau_max, LL = DelayEmbeddings.estimate_maximum_delay(ss; tw=tw)
#
# using Plots
# gui()
# twss = zeros(length(tw))
# [twss[cnt] = i for (cnt,i) in enumerate(tw)]
# plot(twss,LL, label="")
# xlabel!("time window")
# ylabel!("L")

taus = 0:100
Y, τ_vals, ts_vals, FNNs, betas = MDOP(ss; τs = taus, w = theiler, βs=true)
# for different τs
taus2 = 1:4:100
Y2, τ_vals2, ts_vals2, FNNs2, betas2 = MDOP(ss; τs = taus2, w = theiler, βs=true)

@test round.(β,digits=7) == round.(betas[:,1],digits=7)
@test size(Y,2) == 6
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


@testset "MDOP multivariate" begin
## test MDOP() multivariate




end
end
