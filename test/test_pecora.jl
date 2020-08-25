using DelayEmbeddings, DynamicalSystemsBase
using DifferentialEquations
using Random
using Test
import Peaks


@testset "Pecora" begin
# %% Generate data
Random.seed!(414515)
lor = Systems.lorenz(ρ=60)
data = trajectory(lor, 1280; dt=0.02, Ttr = 10)
metric = Chebyshev()

UNDERSAMPLING = false

@testset "Pecora univariate" begin
# %% Timeseries case

s = data[:, 2] # input timeseries = first entry of lorenz
optimal_τ = estimate_delay(s, "mi_min")
Tmax = 100
K = 14
samplesize = 0.05


# using PyPlot
# pygui(true)
# figure()
# ax1 = subplot(211)
# ylabel("⟨ε★⟩")
# ax2 = subplot(212)
# xlabel("τ (index units)")
# ylabel("⟨Γ⟩")


τs = (0,)
es_ref, Γs = pecora(s, τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric, undersampling = UNDERSAMPLING)
max1 = Peaks.maxima(vec(es_ref))
@test max1[1]-1 == optimal_τ
maxi = maximum(es_ref)
@test maxi ≤ 1.35
# ax1.plot(es_ref, label = "τs = $(τs)")
# ax2.plot(Γs)

τs = (0, 5,)
es, Γs = pecora(s, τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric, undersampling = UNDERSAMPLING)
max2 = Peaks.maxima(vec(es))
min1 = Peaks.minima(vec(es))
@test max2[2]-1 == 21
@test min1[1]-1 == optimal_τ
# ax1.plot(es, label = "τs = $(τs)")
# ax2.plot(Γs)

τs = (0, 6, 21)
es, Γs = pecora(s, τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric, undersampling = UNDERSAMPLING)
min2 = Peaks.minima(vec(es))
@test min2[3]-1 == 21
# ax1.plot(es, label = "τs = $(τs)")
# ax2.plot(Γs)
# ax1.legend()
# ax1.set_title("lorenz data")

end

@testset "Pecora multivariate" begin
# %% Trajectory case
s = Dataset(data)
optimal_τ = estimate_delay(s[:,2], "mi_min")
Tmax = 100
K = 14
samplesize = 0.05

js = (2,)
τs = (0,)
Random.seed!(414515)
es_ref, Γs = pecora(s[:,2], τs; delays = 0:Tmax, w = optimal_τ, samplesize = samplesize, K = K, metric = metric, undersampling = UNDERSAMPLING)
Random.seed!(414515)
es, Γs = pecora(s, τs, js; delays = 0:100, samplesize = samplesize, w = optimal_τ, K = K)

@test round.(es[:,2], digits = 5) == round.(vec(es_ref),digits = 5)
x_maxi = Peaks.maxima(es[:,1])
@test x_maxi[1]-1 == 9
z_maxi = Peaks.maxima(es[:,3])
@test z_maxi[1]-1 == 14

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
# subplot(212)
# for i in 1:3
#     plot(Γs[:, i], label = "τs = $(τs), js = $(js), J = $(i)")
# end
# ylabel("⟨Γ⟩")
# xlabel("τ (index units)")

end
end
