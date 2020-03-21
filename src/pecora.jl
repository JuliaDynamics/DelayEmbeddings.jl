using Distances

#=
# Description of the algorithm and definition of used symbols

## 0. Input embedding
s is the input timeseries or Dataset.
On s one performs a d-dimensional embedding, that uses a combination of d timeseries
(with arbitrary amount of repetitions) and d delay times.

I.e. input is `s, ds, τs`, with `ds, τs` being tuples.

## 1. Core loop given a specific input embedding
Let 𝐯 be a d-dimensional embedding, whose each entry is an arbitrary choice out of
the available timeseries (if we have multiple input timeseries, otherwise they are all the
same) and each entry is also defined with respect to an arbitrary delay.

Define a radius δ around 𝐯 in R^d space (d-dimensional embedding). k points are inside
the δ-ball (with respect to some metric) around 𝐯. For simplicity, the time index of
𝐯 is t0. The other poinds inside the δ_ball have indices ti (with several i).

We want to check if we can add an additional dimension to the embedding, using the j-th
timeseries. We check with continuity statistic of Pecora et al.

let x(t+τ) ≡ s_j(t+τ) be this extra dimension we add into the embedding. Out of the
k points in the δ-ball, we count l of them that land into a range ε around x.  Notice that
"points" is a confusing term that should not be used interchange-bly. Here in truth we
refer to **indices** not points. Because of delay embedding, all points are mapped 1-to-1
to a unique time idex. We count the x points with the same time indices ti, if they
are around the original x point with index t0.

Now, if l ≥ δ_to_ε_amount[k] (where δ_to_ε_amount a dictionary defined below), we can
reject the null hypothesis (that the point mapping was by chance), and thus we satisfy
the continuity criterion.

## 2. Finding minimum ε
we repeat 1. while decreasing ε and varying δ, until the null cannot be rejected.
We record the minimum value of ε that can reject the null, and we call that
ε⋆ (\epsilon\star). This ε⋆ is the "continuity statistic"

## 3. Averaging ε
We repeat step 1 and 2 for several different input points 𝐯 and average the result in
ε⋆_avg ≡ ⟨ε⋆⟩

The larger ε⋆_avg, the more functionaly independent is the new d+1 entry to the rest
d entries of the embedding.

## Creating a proper embedding
The Pecora embedding is a sequential process. This means that one should start with
a 1-dimensional embedding, with delay time 0. Then, one performs steps 1-3 for a
choice of one more embedded dimension, i.e. the j-th timeseries and a delay τ.
The optimal choice for the second dimension of the embedding is the j entry with highest
ε⋆_avg and τ in a local maximum of ε⋆_avg.

Then these two dimensions are used again as input to the algorithm, and sequentially
the third optimal entry for the embedding is chosen. Each added entry successfully
reduces ε⋆_avg for the next entry.

This process continues until ε cannot be reduced further, in which scenario the
process terminates and we have found an optimal embedding that maximizes
functional independence among the dimensions of the embedding.

## The undersampling statistic
Because real world data are finite, the aforementioned process (of seeing when ε⋆_avg
will saturate) isn't very accurate because as the dimension of 𝐯 increases, we are
undersampling a high-dimensional object.

# TODO: Understand, describe, and implement the undersampling statistic
=#


# Questions for Hauke:
# - How do you choose the δ ? The size δ?
# - What is the general formula to obtain the number l (i.e. the general form
#   of table 1)?

# do it for many `k`, then take the maximum ε⋆ over `k`.

# Perforamance notes:
# for fnding points within ε, do y = sort!(x) and optimized count starting from index
# of x and going up and down


function continuity_statistic(s, js, τs)


end


"""
    pecora(x::Union{AbstractVector, Dataset}; kwargs...) -> vals
Attempt to estimate optimal delay embedding parameters for `x` using the unified approach
of Pecora et al. [1]. `x` can be anything accepted by [`embed`](@ref).

## Keywords
* `τs = 1:50` : what delay times to use in the approach
*

"""
function pecora(s::Union{AbstractVector, Dataset};
    τs = 1:50, ε_tries = 20, sample_size = 0.5, theiler = 1,
    metric = Chebyshev(), break_percentage = 0.1, β = 0.05,
    )

    #TODO: Optimize τ choice by using estimate_delay

    y₀ = x
    # initial tau value for no embedding
    tau_vals = 0;
    for τ in 1:τmax
        yn = embed(x, )

    # core
end

"""
Table 1 of Pecora (2007), i.e. the necessary amount of points for given δ points
that *must* be mapped into the ε set to reject the null hypothesis.
"""
const δ_to_ε_amount = Dict(
    5=>5,
    6=>6,
    7=>7,
    8=>7,
    9=>8,
    10=>9,
    11=>9,
    12=>9,
    13=>10,
)

)δ_points = [5 6 7 8 9 10 11 12 13];
ε_points = [5 6 7 7 8 9 9 9 10];
