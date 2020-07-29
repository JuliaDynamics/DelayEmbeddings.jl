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

using Revise

println("\nTesting MDOP.jl...")

# test beta_statistic()

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

# theiler window
theiler = 57
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

# test MDOP()
