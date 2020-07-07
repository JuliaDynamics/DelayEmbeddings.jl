# test garcia-almeida-method
using DynamicalSystemsBase
using DelayEmbeddings
using StatsBase
using Statistics
using Random
using Test
using Peaks

println("\nTesting garcia_almeida.jl...")


## Check on Lorenz System
lo = Systems.lorenz()

tr = trajectory(lo, 60; dt = 0.01, Ttr = 10)

x = tr[:, 1]
Y = Dataset(x)

τ_max = 50

@time begin
N , NN_distances = garcia_embedding_cycle(Y, x, w=0, T=1, τ_max = τ_max)
end

T = 17
@time begin
N2 , NN_distances2 = garcia_embedding_cycle(Y, x, w=0, T=T, τ_max = τ_max)
end

# check whether the `d_E1`-statistic is the same
@test NN_distances[1][1][1:100] == NN_distances2[1][1][1:100]
@test NN_distances[5][1][1:100] == NN_distances2[5][1][1:100]

min_dist = 4
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
