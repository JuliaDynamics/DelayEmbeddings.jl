using DelayEmbeddings, DynamicalSystemsBase, PyPlot, Test
pygui(true)

function ar_process(u0::T, α::T, p::T, N::Int) where {T<:Real}
    x = zeros(T, N+10)
    x[1] = u0
    for i = 2:N+10
        x[i] = α*x[i-1] + p*randn()
    end
    return x[11:end]
end

trajectories = Dict(
    "White noise" => Dataset(randn(5000)),
    "AR(1)" => Dataset(ar_process(.2,.9,.1,5000)),
    "Hénon (chaotic)" => trajectory(Systems.henon(a=1.4, b=0.3), 199, Ttr=1000),
    "Hénon (periodic)" => trajectory(Systems.henon(a=1.054, b=0.3), 199, Ttr=1000),
    "Lorenz" => trajectory(Systems.lorenz(), 1000.0; Ttr=1000, dt = 0.05),
    "Lorenz96" => trajectory(Systems.lorenz96(5; F = 8.0), 1000.0; Ttr=1000, dt = 0.05),
    "Roessler" => trajectory(Systems.roessler(), 1000.0; Ttr=1000, dt = 0.05),
    "Sine wave" => Dataset(map(x->[sin.(x) cos.(x)], StepRangeLen(0.0,0.2,200)))
)
dict_keys = ["White noise","AR(1)","Hénon (chaotic)","Hénon (periodic)","Lorenz","Lorenz96","Roessler","Sine wave"]

# set noise level
σ = 0.01
# set maximum encountered dimension
dmax = 15
γs = 0:dmax-1
dims = γs .+ 1

slope_thres = .1

for names in dict_keys
    tr = trajectories[names]
    tr = regularize(tr)
    s = tr[:, 1].+ σ.*randn(length(tr))
    τ = estimate_delay(s, "mi_min")

    # do a plot of the differently found methods

    fs = (afnn, fnn, f1nn, ifnn)
    methods = ["afnn", "fnn", "f1nn", "ifnn"]
    fig, axs  = subplots(2, 1; sharex = true)
    for (i, f) in enumerate(fs)
        x = f(s, τ, γs)

        if i == 1
            x .= 1 .- x
            E2 = stochastic_indicator(s, τ, γs)
        end

        axs[1].plot(dims, x, label = string(f); color = "C$(i)")
        axs[2].plot(dims[1:end-1], diff(x); color = "C$(i)")
        # lrs, tans = linear_regions(dims, x, tol = 0.5)
        # axs[1].scatter(dims[lrs], x[lrs], color = "C$(i)")
        z = slope_thres # threshold of slopes
        axs[2].axhspan(-z, z; color = "0.5", alpha = 0.2, label = "±$(z)")

        # Estimations of condition for convergence:
        # earliest point where x is both less than a thresohold, 0.05, and local slope is
        # between ± 0.05
        r = 0.05 # threshold of convergence near 0
        d_max = dmax
        Y, _, rat =  optimal_traditional_de(s, "mi_min", methods[i]; dmax = d_max, slope_thres = slope_thres)
        println("$(methods[i]) embedding dimension for $names: $(size(Y,2)) ")
        axs[1].scatter(dims[size(Y,2)], x[size(Y,2)]; color = "C$(i)", edgecolors  = "k")
    end
    axs[1].grid(); axs[2].grid()
    axs[1].legend(loc = "upper right", ncol = 2)
    axs[1].set_ylabel("estimator")
    axs[2].set_xlabel("dimension")
    axs[2].set_ylabel("local slope")
    axs[1].set_title(names)
end
