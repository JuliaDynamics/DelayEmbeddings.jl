using DelayEmbeddings, PyPlot, DynamicalSystemsBase, DelimitedFiles

# %% Timeseries case
desktop!()
data = readdlm("data-lorenz.txt")
s = data[:, 2] # input timeseries = first entry of lorenz

figure()
ylabel("⟨ε★⟩")
xlabel("τ (index units)")
τs = (0, )
@time es = continuity_statistic(s, τs; T = 1:200, N = 1000)
plot(es, label = "τs = $(τs)")

optimal_τ = estimate_delay(s, "mi_min")
axvline(optimal_τ, color = "k", ls = "dashed", label = "mut. inf. optimal τ = $(optimal_τ)")
legend()

τs = (0, 6,)
@time es = continuity_statistic(s, τs; T = 1:200, N = 1000)
plot(es, label = "τs = $(τs)")
legend()

τs = (0, 6, 34)
@time es = continuity_statistic(s, τs; T = 1:200, N = 1000)
plot(es, label = "τs = $(τs)")
legend()

τs = (0, 32, 89, 53)
@time es = continuity_statistic(s, τs; T = 1:200, N = 1000)
plot(es, label = "τs = $(τs)")
legend()

# %% Trajectory case
lo = Systems.lorenz()
s = trajectory(lo, 1280; dt = 0.02, Ttr = 10.0)

x, y, z = columns(s)
#
# x = (x .- mean(x)) ./ std(x)
# y = (y .- mean(y)) ./ std(y)
# z = (z .- mean(z)) ./ std(z)

s = Dataset(x, y, z)

figure()
ylabel("⟨ε★⟩")
xlabel("τ (index units)")
τs = (0, )
js = (1, )

@time es = continuity_statistic(s, τs, js; T = 1:200, N = 1000)

for i in 1:3
    plot(es[:, i], label = "τs = $(τs), js = $(js), J = $(i)")
end
legend()

js = (1, 2)
τs = (0, 3)
@time es = continuity_statistic(s, τs, js; T = 1:200, N = 1000)

for i in 1:3
    plot(es[:, i], label = "τs = $(τs), js = $(js), J = $(i)")
end
legend()

τs = (0, 0, 21)
js = (1, 3, 3)
@time es = continuity_statistic(s, τs, js; T = 1:200, N = 1000)
for i in 1:3
    plot(es[:, i], label = "τs = $(τs), js = $(js), J = $(i)")
end
legend()
