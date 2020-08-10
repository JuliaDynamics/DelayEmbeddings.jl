# test garcia-almeida-method
using DynamicalSystemsBase
using DelayEmbeddings
using StatsBase
using Statistics
using Random
using Test
using Peaks
using Revise
using BenchmarkTools

println("\nTesting garcia_almeida.jl...")

#@testset "G&A method" begin

## Check on Lorenz System
lo = Systems.lorenz()

tr = trajectory(lo, 60; dt = 0.01, Ttr = 10)

x = tr[:, 1]
Y = Dataset(x)


#@testset "N-statistic" begin
τs = 0:50
@code_warntype(garcia_embedding_cycle(Y, x, w=0, T=1, τs = τs))
N , NN_distances = garcia_embedding_cycle(Y, x, w=0, T=1, τs = τs)

T = 17
N2 , NN_distances2 = garcia_embedding_cycle(Y, x, w=0, T=T, τs = τs)

# check whether the `d_E1`-statistic is the same
@test NN_distances[1][1:50] == NN_distances2[1][1:50]
@test NN_distances[5][1:100] == NN_distances2[5][1:100]

min_dist = 12
max_1_idx = Peaks.maxima(N,min_dist)
max_2_idx = Peaks.maxima(N2,min_dist)

@test max_1_idx == max_2_idx

# plot N-Statistic for the Lorenz system as in Fig. 2(a) in [^Garcia2005b]
# using Plots
# plot(N,linewidth = 2, label = "T=1 (as in the Paper)", xaxis=:log)
# plot!(N2,linewidth = 2, label = "T=$T", xaxis=:log)
# plot!(N,seriestype = :scatter, label = "",xaxis=:log)
# plot!(N2,seriestype = :scatter, label = "",xaxis=:log)
# plot!(title = "Lorenz System as in Fig. 2(a) in the Paper")
# xlabel!("τ")
# ylabel!("N-Statistic")

#end

#@testset "garcia_embed univariate" begin
Y_act, τ_vals, ts_vals, FNNs, NS = garcia_almeida_embed(x; τs=0:100,  w = 17,  Ns=true)



#end


#@testset "garcia_embed multivariate" begin




#end


#end
