using DynamicalSystemsBase
using DelayEmbeddings
using StatsBase
using Revise

lo = Systems.lorenz()

tr = trajectory(lo, 100; dt = 0.1, Ttr = 10)

x = tr[:, 1]

L = uzal_cost(tr)
