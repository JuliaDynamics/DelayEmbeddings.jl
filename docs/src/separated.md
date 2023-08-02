# Separated optimal embedding
This page discusses and provides algorithms for estimating optimal parameters to do Delay Coordinates Embedding (DCE) with using the separated approach.

## Optimal delay time
```@docs
estimate_delay
exponential_decay_fit
```
### Self Mutual Information

```@docs
selfmutualinfo
```

Notice that mutual information between two *different* timeseries x, y exists in JuliaDynamics as well, but in the package [CausalityTools.jl](https://github.com/JuliaDynamics/CausalityTools.jl).
It is also trivial to define it yourself using `entropy` from `ComplexityMeasures`.

## Optimal embedding dimension
```@docs
optimal_separated_de
delay_afnn
delay_ifnn
delay_fnn
delay_f1nn
DelayEmbeddings.stochastic_indicator
```

## Example
```@example MAIN
using DelayEmbeddings, CairoMakie
using DynamicalSystemsBase

function roessler_rule(u, p, t)
    a, b, c = p
    du1 = -u[2]-u[3]
    du2 = u[1] + a*u[2]
    du3 = b + u[3]*(u[1] - c)
    return SVector(du1, du2, du3)
end
ds = CoupledODEs(roessler_rule, [1, -2, 0.1], [0.2, 0.2, 5.7])

# This trajectory is a chaotic attractor with fractal dim ≈ 2
# therefore the set needs at least embedding dimension of 3
X, tvec = trajectory(ds, 1000.0; Δt = 0.05)
x = X[:, 1]

dmax = 7
fig = Figure()
ax = Axis(fig[1,1]; xlabel = "embedding dimension", ylabel = "estimator")
for (i, method) in enumerate(["afnn", "fnn", "f1nn", "ifnn"])
    # Plot statistic used to estimate optimal embedding
    # as well as the automated output embedding
    𝒟, τ, E = optimal_separated_de(x, method; dmax)
    lines!(ax, 1:dmax, E; label = method, marker = :circle, color = Cycled(i))
    optimal_d = size(𝒟, 2)
    ## Scatter the optimal embedding dimension as a lager marker
    scatter!(ax, [optimal_d], [E[optimal_d]];
        color = Cycled(i), markersize = 30
    )
end
axislegend(ax)
fig
```
