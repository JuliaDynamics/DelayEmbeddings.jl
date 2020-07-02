using DynamicalSystemsBase
using DelayEmbeddings
using StatsBase
using DelimitedFiles
using Revise
using Random
#using Plots
using PyPlot

pygui(true)

## Simple Check on Lorenz System
lo = Systems.lorenz()

tr = trajectory(lo, 100; dt = 0.1, Ttr = 10)

#writedlm("lorenz_trajectory.csv", tr)
x = tr[:, 1]


@time begin
L = uzal_cost(tr; Tw = 40, K= 3, w = 1, SampleSize = 1.0,
    metric = Euclidean())
end

## Test Roessler example as in Fig. 7 in the paper

# load time series generated in Matlab for comparison reasons
ts = readdlm("roess_ts.txt")
# theiler window
w  = 12
# Time horizon
Tw = 80
# sample size
SampleSize = .1
# metric
metric = Euclidean()
# embedding dimension
m = 3
# maximum neighbours
k_max = 4
# number of total trials
trials = 50

# preallocation
L = zeros(k_max,trials)
tw_max = zeros(trials)

@time begin
for K = 1:k_max
    display(K)
    for i = 1:trials
        tw_max[i] = i*(m-1)
        Y = embed(transpose(ts),m,i)
        L[K,i] = uzal_cost(Y; Tw=Tw, K=K, w=w, SampleSize=SampleSize, metric=metric)
    end
end
end

# # plot results using Plots
# plot(tw_max,L[1,:], linewidth = 2, label = "k=1")
# plot!(tw_max,L[2,:], linewidth = 2,label = "k=2")
# plot!(tw_max,L[3,:], linewidth = 2,label = "k=3")
# plot!(tw_max,L[4,:], linewidth = 2,label = "k=4")
# plot!(xticks = 0:10:100)
# plot!(title = "Roessler System as in Fig. 7(b) in the Uzal Paper")

# plot results using PyPlot
labels = ["k=1", "k=2", "k=3", "k=4"]
figure()
plot(tw_max,L[1,:], linewidth = 2)
plot(tw_max,L[2,:], linewidth = 2)
plot(tw_max,L[3,:], linewidth = 2)
plot(tw_max,L[4,:], linewidth = 2)
xlabel(L"$t_w$")
ylabel(L"$L_k$")
legend(labels)
xticks(0:10:100)
#yscale("symlog")
title("Roessler System as in Fig. 7(b) in the Uzal Paper")
grid()

## Test Roessler example as in Fig. 7 in the paper with internal data

ro = Systems.roessler([1.0, 1.0, 1.0], a=0.15, b = 0.2, c=10)

tr = trajectory(ro, 1250; dt = 0.125, Ttr = 10)

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
trials = 50

# preallocation
L = zeros(k_max,trials)
tw_max = zeros(trials)

@time begin
for K = 1:k_max
    display(K)
    for i = 1:trials
        tw_max[i] = i*(m-1)
        Y = embed(x,m,i)
        L[K,i] = uzal_cost(Y; Tw=Tw, K=K, w=w, SampleSize=SampleSize, metric=metric)
    end
end
end

# # plot results using Plots
# plot(tw_max,L[1,:], linewidth = 2, label = "k=1")
# plot!(tw_max,L[2,:], linewidth = 2,label = "k=2")
# plot!(tw_max,L[3,:], linewidth = 2,label = "k=3")
# plot!(tw_max,L[4,:], linewidth = 2,label = "k=4")
# plot!(xticks = 0:10:100)
# plot!(title = "Roessler System as in Fig. 7(b) in the Uzal Paper")

# plot results using PyPlot
labels = ["k=1", "k=2", "k=3", "k=4"]
figure()
plot(tw_max,L[1,:], linewidth = 2)
plot(tw_max,L[2,:], linewidth = 2)
plot(tw_max,L[3,:], linewidth = 2)
plot(tw_max,L[4,:], linewidth = 2)
xlabel(L"$t_w$")
ylabel(L"$L_k$")
legend(labels)
xticks(0:10:100)
#yscale("symlog")
title("Roessler System as in Fig. 7(b) in the Uzal Paper")
grid()

## Test Lorenz example as in Fig. 7 in the paper with internal data

lo = Systems.lorenz([1.0, 1.0, 50.0])

tr = trajectory(lo, 100; dt = 0.01, Ttr = 10)

x = tr[:, 1]

# theiler window
w  = 12
# Time horizon
Tw = 80
# sample size
SampleSize = .1
# metric
metric = Euclidean()
# embedding dimension
m = 3
# maximum neighbours
k_max = 4
# number of total trials
trials = 50

# preallocation
L = zeros(k_max,trials)
tw_max = zeros(trials)

@time begin
for K = 1:k_max
    display(K)
    for i = 1:trials
        tw_max[i] = i*(m-1)
        Y = embed(x,m,i)
        L[K,i] = uzal_cost(Y; Tw=Tw, K=K, w=w, SampleSize=SampleSize, metric=metric)
    end
end
end

# # plot results
# plot(tw_max,L[1,:], linewidth = 2, label = "k=1")
# plot!(tw_max,L[2,:], linewidth = 2, label = "k=2")
# plot!(tw_max,L[3,:], linewidth = 2, label = "k=3")
# plot!(tw_max,L[4,:], linewidth = 2, label = "k=4")
# plot!(xticks = 0:10:100)
# plot!(title = "Lorenz System as in Fig. 7(a) in the Uzal Paper")

labels = ["k=1", "k=2", "k=3", "k=4"]
figure()
plot(tw_max,L[1,:], linewidth = 2)
plot(tw_max,L[2,:], linewidth = 2)
plot(tw_max,L[3,:], linewidth = 2)
plot(tw_max,L[4,:], linewidth = 2)
xlabel(L"$t_w$")
ylabel(L"$L_k$")
legend(labels)
xticks(0:10:100)
#yscale("symlog")
title("Lorenz System as in Fig. 7(a) in the Uzal Paper")
grid()
