using Neighborhood
using StatsBase
using Distances

export beta_statistic
export MDOP


"""
    MDOP(....)
method by [^Nichkawde2013]


[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
"""
function MDOP(Y...)
    #tbd

end


"""
    beta_statistic(Y::Dataset, s::Vector); kwargs...) → β
Compute the β-statistic `β` for input state space trajectory `Y` and a timeseries `s`
according to Nichkawde [^Nichkawde2013],
based on estimating derivatives on a projected manifold. For a range of delay
values `τs`, `β` gets computed and its maximum over all considered `τs` serves
as the optimal delay considered in this embedding cycle.

## Keyword arguments

* `τs= 0:50`: Considered delay values `τs` (in sampling time units). For each of
  the `τs`'s the β-statistic gets computed.
* `w = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.

## Description
The `β`-statistic is based on the geometrical idea of maximal unfolding of the
reconstructed attractor and is tightly related to the False Nearest Neighbor
method ([^Kennel1992]). In fact the method eliminates the maximum amount of
false nearest neighbors in each embedding cycle. The idea is to estimate the
absolute value of the directional derivative with respect to a possible new
dimension in the reconstruction process, and with respect to the nearest
neighbor, for all points of the state space trajectory:

ϕ'(τ) = Δϕ_d(τ) / Δx_d

Δx_d is simply the Euclidean nearest neighbor distance for a reference point
with respect to the given Theiler window `w`. Δϕ_d(τ) is the distance of the
reference point to its nearest neighbor in the one dimensional time series `s`,
for the specific τ. Δϕ_d(τ) = |s(i+τ)-s(j+τ)|, with i being the index of the
considered reference point and j the index of its nearest neighbor.

Finally,

`β` = log β(τ) = ⟨log₁₀ ϕ'(τ)⟩ ,

with ⟨.⟩ being the mean over all reference points. When one chooses the maximum
of `β` over all considered τ's, one obtains the optimal delay value for this
embedding cycle. Note that in the first embedding cycle, the input state space
trajectory `Y` can also be just a univariate time series.

[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
[^Kennel1992]: Kennel, M. B., Brown, R., Abarbanel, H. D. I. (1992). [Determining embedding dimension for state-space reconstruction using a geometrical construction. Phys. Rev. A 45, 3403] (https://doi.org/10.1103/PhysRevA.45.3403).
"""

function beta_statistic(Y::Dataset, s::Vector; τs = 0:50 , w::Int = 1)
    @assert length(s) ≥ length(Y) "The length of the input time series `s` must be at least the length of the input trajectory `Y` "
    @assert all(x -> x ≥ 0, τs)
    τ_max = maximum(τs)
    metric = Euclidean()    # consider only Euclidean norm
    K = 1                   # consider only first nearest neighbor
    N = length(Y)           # length of the state space trajectory
    NN = N - τ_max          # allowed length of the trajectory w.r.t. τ_max
    vtree = KDTree(Y[1:NN], metric)
    allNNidxs, Δx = all_neighbors(vtree, Y[1:NN], 1:NN, K, w)   # Eq. 12

    Δϕ = zeros(NN, length(τs))
    @inbounds for j = 1:NN # loop over all state space points
        for (i,τ) in enumerate(τs) # loop over all considered τ's
            Δϕ[j,i] = abs(s[j+τ][1]-s[allNNidxs[j][1]+τ][1]) / Δx[j][1] # Eq. 14 & 15
        end
    end

    # final β statistic is the average of the logarithms, see Eq. 16
    β = mean(log10.(Δϕ); dims=1)
    return vec(β)
end
