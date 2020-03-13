using Distances




"""
    pecora(x::Union{AbstractVector, Dataset}; kwargs...) -> vals
Attempt to estimate optimal delay embedding parameters for `x` using the unified approach
of Pecora et al. [1]. `x` can be anything accepted by [`embed`](@ref).

## Keywords
* `τs = 1:50` :

"""
function pecora(x::Union{AbstractVector, Dataset};
    τ_max = 50, ε_tries = 20, sample_size = 0.5, theiler = 1,
    metric = Chebyshev(), break_percentage = 0.1, α = 0.05,
    )

    #TODO: Optimize τ choice by using estimate_delay

    y₀ = x
    # initial tau value for no embedding
    tau_vals = 0;
    for τ in 1:τmax
        yn = embed(x, )

    # core
end
