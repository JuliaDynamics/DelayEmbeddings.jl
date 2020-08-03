using Neighborhood
using StatsBase
using Distances

export beta_statistic
export MDOP


"""
    MDOP(s::Vector; kwargs...) → Y, τ_vals, ts_vals FNNs
MDOP is a unified approach to properly embed a time series (`Vector` type) or a
set of time series (`Dataset` type) based on the paper of Chetan Nichkawde
[^Nichkawde2013].

## Keyword arguments

* `τs= 0:50`: Considered delay values `τs` (in sampling time units). For each of
  the `τs`'s the β-statistic gets computed.
* `w = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `fnn_thres = 0.05`: A threshold value defining a sufficiently small fraction
  of false nearest neighbors, in order to the let algorithm terminate and stop
  the embedding procedure (`0 > fnn_thres > 1).
* `r = 2`: The threshold for the tolerable relative increase of the distance
  between the nearest neighbors, when increasing the embedding dimension.

## Description
The method works iteratively and gradually builds the final embedding vectors
`Y`. Based on the [`beta_statistic`](@ref) the algorithm picks an optimal delay
value `τ` for each embedding cycle as the global maximum of `β`. In case of
multivariate embedding, i.e. when embedding a set of time series (`s::Dataset`),
the optimal delay value `τ` is chosen as the maximum from all maxima's of all
considered `β`-statistics for each embedding cycle. The range of
considered delay values is determined in `τs` and for the nearest neighbor
search we respect the Theiler window `w`. After each embedding cycle the FNN-
statistic `FNNs` [^Hegger1999] is being checked and as soon as this statistic
drops below the threshold `fnn_thres`, the algorithm breaks. In order to
increase the  practability of the method the algorithm also breaks, when the
FNN-statistic `FNNs` increases . The final embedding vector is stored in `Y`
(`Dataset`). The chosen delay values for each embedding cycle are stored in the
`τ_vals` and the according time series number chosen for the according delay
value in `τ_vals` is stored in `ts_vals`. For univariate embedding (`s::Vector`)
`ts_vals` is a vector of ones of length `τ_vals`, because there is simply just
time series to choose from.

[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
[^Hegger1999]: Hegger, Rainer and Kantz, Holger (1999). [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
function MDOP(s::Vector; τs = 0:50 , w::Int = 1, fnn_thres::float = 0.05, r = 2)
    @assert 0 < fnn_thres < 1 "Please select a valid breaking criterion, i.e. a threshold value `fnn_thres` ∈ (0 1)"
    @assert all(x -> x ≥ 0, τs)

    # normalize input time series
    s = (s.-mean(s))./std(s)
    # define actual phase space trajectory
    Y_act = s

    # set a flag, in order to tell the while loop when to stop. Each loop
    # stands for encountering a new embedding dimension
    flag = true

    # set index-counter for the while loop
    cnt = 1;

    # initial tau value for no embedding. This is trivial 0, when there is no
    # embedding
    τ_vals = zeros(Int,1)
    ts_vals = ones(Int,1)

    # loop over increasing embedding dimensions until some break criterion will
    # tell the loop to stop/break
    while flag
        # get the β-statistic
        β = beta_statistic(Y_act, s; τs = τs , w::Int = w)

        # determine the optimal tau value from the β-statistic
        maxi, max_idx = findmax(β)
        # store chosen delay (and chosen time series)
        push!(τ_vals,τs[max_idx])
        push!(ts_vals,1)

        # create phase space vector for this embedding cycle

    end

end

"""
    MDOP(s::Dataset; kwargs...) → Y, τ_vals, ts_vals, FNNs
MDOP is a unified approach to properly embed a time series based on the paper of
Chetan Nichkawde [^Nichkawde2013].
[`uzal_cost`](@ref)


[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
"""
function MDOP(s::Vector; τs = 0:50 , w::Int = 1, fnn_thres::float = 0.05, r = 2)
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
