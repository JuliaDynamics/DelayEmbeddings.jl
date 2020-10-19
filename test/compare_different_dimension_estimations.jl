using DelayEmbeddings, DynamicalSystemsBase, PyPlot

pygui(true)
# Generate the first timeseries (use a bunch of different systems)
#ds = Systems.lorenz96(5; F = 8.0)
ds = Systems.lorenz()
tr = trajectory(ds, 1000.0; Ttr = 100.0, dt = 0.05)
s = tr[:, 1]
τ = estimate_delay(s, "mi_min")

# do a plot of the differently found methods
dmax = 10
γs = 0:dmax-1
dims = γs .+ 1

fs = (afnn, fnn, f1nn, ifnn)
fig, axs  = subplots(2, 1; sharex = true)
for (i, f) in enumerate(fs)
    x = f(s, τ, γs)
    @show f, x
    if f == afnn
        x .= 1 .- x
    end
    axs[1].plot(dims, x, label = string(f); color = "C$(i)")
    axs[2].plot(dims[1:end-1], diff(x); color = "C$(i)")
    # lrs, tans = linear_regions(dims, x, tol = 0.5)
    # axs[1].scatter(dims[lrs], x[lrs], color = "C$(i)")
    z = 0.2
    axs[2].axhspan(-z, z; color = "0.5", alpha = 0.2, label = "±$(z)")

    # Estimations of condition for convergence:
    # earliest point where x is both less than a thresohold, 0.2, and local slope is
    # between ± 0.2
    z = 0.2 # threshold of slopes
    r = 0.2 # threshold of convergence near 0
    y = diff(x)
    k = 0
    for j in 1:length(x)-1
        k = j
        x[k] < r && y[k] < z && break
    end
    axs[1].scatter(dims[k], x[k]; color = "C$(i)", edgecolors  = "k")
end
axs[1].grid(); axs[2].grid()
axs[1].legend(loc = "upper right", ncol = 2)
axs[1].set_ylabel("estimator")
axs[2].set_xlabel("dimension")
axs[2].set_ylabel("local slope")

axs[1].set_title("5-dim Lorenz96")
