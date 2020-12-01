using DelayEmbeddings, DynamicalSystemsBase
using Test
using Random
import Peaks

@testset "Pecora" begin
# %% Generate data
lor = Systems.lorenz([0.0;1.0;0.0];ρ=60)
data = trajectory(lor, 200; dt=0.02, Ttr = 10)
metric = Chebyshev()

UNDERSAMPLING = false

@testset "Pecora univariate" begin
# %% Timeseries case

s = data[:, 2] # input timeseries = first entry of lorenz
optimal_τ = estimate_delay(s, "mi_min")
Tmax = 100
K = 14
samplesize = 1

# using PyPlot
# pygui(true)
# figure()
# ax1 = subplot(211)
# ylabel("⟨ε★⟩")
# ax2 = subplot(212)
# xlabel("τ (index units)")
# ylabel("⟨Γ⟩")

τs = (0,)
Random.seed!(123)
es_ref, Γs = pecora(s, τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric, undersampling = UNDERSAMPLING)
(max1,_) = Peaks.findmaxima(vec(es_ref))
@test optimal_τ - 2 ≤ max1[1]-1 ≤ optimal_τ + 2
maxi = maximum(es_ref)
@test maxi ≤ 1.3
# ax1.plot(es_ref, label = "τs = $(τs)")
# ax2.plot(Γs)

τs = (0, max1[1]-1,)
Random.seed!(123)
es, Γs = pecora(s, τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric, undersampling = UNDERSAMPLING)
(max2,_) = Peaks.findmaxima(vec(es))
(min1,_) = Peaks.findminima(vec(es))
@test 1 ≤ max2[1]-1 ≤ 4
@test optimal_τ - 2 ≤ min1[1]-1 ≤ optimal_τ + 2
# ax1.plot(es, label = "τs = $(τs)")
# ax2.plot(Γs)

τs = (0, max1[1]-1, max2[1]-1)
Random.seed!(123)
es, Γs = pecora(s, τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric, undersampling = UNDERSAMPLING)
(min2,_) = Peaks.findminima(vec(es))
@test max2[1]-2 ≤ min2[1]-1 ≤ max2[1]
# ax1.plot(es, label = "τs = $(τs)")
# ax2.plot(Γs)
# ax1.legend()
# ax1.set_title("lorenz data")

end

@testset "Pecora multivariate" begin
## %% Trajectory case
s = Dataset(data)
optimal_τ = estimate_delay(s[:,2], "mi_min")
Tmax = 100
K = 14
samplesize = 1

js = (2,)
τs = (0,)
Random.seed!(123)
es_ref, Γs = pecora(s[:,2], τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric, undersampling = UNDERSAMPLING)
Random.seed!(123)
es, Γs = pecora(s, τs, js; delays = 0:Tmax, samplesize = samplesize, w = optimal_τ, K = K, metric = metric, undersampling = UNDERSAMPLING)

@test round.(es[:,2], digits = 4) == round.(vec(es_ref),digits = 4)
(x_maxi,_) = Peaks.findmaxima(es[:,1])
@test 8 ≤ x_maxi[1]-1 ≤ 10
(z_maxi,_) = Peaks.findmaxima(es[:,3])
@test 13 ≤ z_maxi[1]-1 ≤ 15

# using PyPlot
# pygui(true)
# figure()
# subplot(211)
# for i in 1:3
#     plot(es[:, i], label = "τs = $(τs), js = $(js), J = $(i)")
# end
# ylabel("⟨ε★⟩")
# title("lorenz system, multiple timeseries")
# legend()
# grid()
# subplot(212)
# plot(es_ref, label = "τs = $(τs), js = $(js), J = 2")
# ylabel("⟨ε★⟩-ref")
# xlabel("τ (index units)")
# grid()

end
end
