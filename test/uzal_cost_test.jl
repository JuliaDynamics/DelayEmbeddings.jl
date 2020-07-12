using DynamicalSystemsBase
using DelayEmbeddings
using StatsBase
using Test
using Statistics
using Random

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

L_max = 2.059
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
lo = Systems.lorenz()

tr = trajectory(lo, 10; dt = 0.01, Ttr = 10)

# check Euclidean metric
L = uzal_cost(tr;
    Tw = 60, K= 3, w = 12, samplesize = 1.0,
    metric = Euclidean()
)
L_max = -2.411
L_min = -2.412
@test L_min < L < L_max

# check Maximum metric
L = uzal_cost(tr;
    Tw = 60, K= 3, w = 12, samplesize = 1.0,
    metric = Chebyshev()
)
L_max = -2.475
L_min = -2.485
@test L_min < L < L_max
end

@testset "Roessler system" begin

## Test Roessler example as in Fig. 7 in the paper with internal data

ro = Systems.roessler([1.0, 1.0, 1.0], a=0.15, b = 0.2, c=10)

tr = trajectory(ro, 1250; dt = 0.125, Ttr = 10)

x = tr[:,1]

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

tau_min = 9
tau_max = 11

min1_idx = sortperm(L[1,:])
min1 = tw_max[min1_idx[1]]
@test tau_min < min1 < tau_max
L_min = -2.96
L_max = -2.8
@test L_min < L[1,min1_idx[1]] < L_max


min2_idx = sortperm(L[2,:])
min2 = tw_max[min2_idx[1]]
@test tau_min < min2 < tau_max
L_min = -2.64
L_max = -2.5
@test L_min < L[2,min2_idx[1]] < L_max


min3_idx = sortperm(L[3,:])
min3 = tw_max[min3_idx[1]]
@test tau_min < min3 < tau_max
L_min = -2.46
L_max = -2.3
@test L_min < L[3,min3_idx[1]] < L_max


min4_idx = sortperm(L[4,:])
min4 = tw_max[min4_idx[1]]
@test tau_min < min4 < tau_max
L_min = -2.36
L_max = -2.2
@test L_min < L[4,min4_idx[1]] < L_max

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
trials = 12

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

tau_min = 17
tau_max = 21

min1_idx = sortperm(L[1,:])
min1 = tw_max[min1_idx[1]]
@test tau_min < min1 < tau_max
L_max = -2.3
@test L[1,min1_idx[1]] < L_max


min2_idx = sortperm(L[2,:])
min2 = tw_max[min2_idx[1]]
@test tau_min < min2 < tau_max
L_max = -2.0
@test L[2,min2_idx[1]] < L_max


min3_idx = sortperm(L[3,:])
min3 = tw_max[min3_idx[1]]
@test tau_min < min3 < tau_max
L_max = -1.95
@test L[3,min3_idx[1]] < L_max


min4_idx = sortperm(L[4,:])
min4 = tw_max[min4_idx[1]]
@test tau_min < min4 < tau_max
L_max = -1.9
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
# xticks(0:2:24)
# #yscale("symlog")
# title("Lorenz System as in Fig. 7(a) in the Uzal Paper")
# grid()
end
end
