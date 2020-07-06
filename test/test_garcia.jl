# test garcia-almeida-method
using DynamicalSystemsBase
using DelayEmbeddings
using StatsBase
using DelimitedFiles
using Revise
using Random
using Plots
# using PyPlot
#
# pygui(true)

## Simple Check on Lorenz System
lo = Systems.lorenz()

tr = trajectory(lo, 60; dt = 0.01, Ttr = 10)

#writedlm("lorenz_trajectory.csv", tr)
x = tr[:, 1]
Y = Dataset(x)

τ_max = 50

@time begin
N , NN_distances = DelayEmbeddings.garcia_embedding_cycle(Y, x, w=0, T=1, τ_max = τ_max)
end

T = 17
@time begin
N2 , NN_distances2 = DelayEmbeddings.garcia_embedding_cycle(Y, x, w=0, T=T, τ_max = τ_max)
end

plot(N,linewidth = 2, label = "T=1 (as in the Paper)", xaxis=:log)
plot!(N2,linewidth = 2, label = "T=$T", xaxis=:log)
plot!(title = "Lorenz System as in Fig. 2(a) in the Paper")
xlabel!("τ")
ylabel!("N-Statistic")


# ts = readdlm("lorenz_matlab.txt")
#
# Y = Dataset(ts)
# @time begin
# N , NN_dkistances = DelayEmbeddings.garcia_embedding_cycle(Y, ts, w=0, T=1)
# end
