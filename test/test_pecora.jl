using DelayEmbeddings, PyPlot, DynamicalSystemsBase, DelimitedFiles
using Distances
using Random
UNDERSAMPLING = false

# %% Timeseries case
Random.seed!(414515)
desktop!()
data = readdlm("data-lorenz.txt")
s = data[:, 2] # input timeseries = first entry of lorenz
metric = Chebyshev()
figure()
ax1 = subplot(211)
ylabel("⟨ε★⟩")
ax2 = subplot(212)
xlabel("τ (index units)")
ylabel("⟨Γ⟩")

optimal_τ = estimate_delay(s, "mi_min")

Tmax = 100

τs = (0,)
@time es, Γs = pecora(s, τs; T = 1:Tmax, N = 100, metric = metric, undersampling = UNDERSAMPLING)
ax1.plot(es, label = "τs = $(τs)")
ax2.plot(Γs)

τs = (0, 6,)
@time es, Γs = pecora(s, τs; T = 1:Tmax, N = 100, metric = metric, undersampling = UNDERSAMPLING)
ax1.plot(es, label = "τs = $(τs)")
ax2.plot(Γs)

τs = (0, 6, 34)
@time es, Γs = pecora(s, τs; T = 1:Tmax, N = 100, metric = metric, undersampling = UNDERSAMPLING)
ax1.plot(es, label = "τs = $(τs)")
ax2.plot(Γs)

ax1.legend()
ax1.set_title("lorenz text data")

# %% Trajectory case
using Statistics
lo = Systems.lorenz()
s = trajectory(lo, 1280; dt = 0.02, Ttr = 10.0)
x, y, z = columns(s)
s = Dataset(x, y, z)

js = (1, 2)
τs = (0, 5)
@time es, Γs = pecora(s, τs, js; T = 1:100, N = 1000)

figure()
subplot(211)
for i in 1:3
    plot(es[:, i], label = "τs = $(τs), js = $(js), J = $(i)")
end
ylabel("⟨ε★⟩")
title("lorenz system, multiple timeseries")
legend()
subplot(212)
for i in 1:3
    plot(Γs[:, i], label = "τs = $(τs), js = $(js), J = $(i)")
end
ylabel("⟨Γ⟩")
xlabel("τ (index units)")
