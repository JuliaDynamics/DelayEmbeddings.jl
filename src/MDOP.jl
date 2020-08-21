using Neighborhood
using StatsBase
using Distances

export beta_statistic
export mdop_embedding
export mdop_maximum_delay


"""
    mdop_embedding(s::Vector; kwargs...) → Y, τ_vals, ts_vals, FNNs, βS
MDOP (for "maximizing derivatives on projection") is a unified approach to properly
embed a timeseries or a set of timeseries (`Dataset`)
based on the paper of Chetan Nichkawde [^Nichkawde2013].

## Keyword arguments

* `τs= 0:50`: Possible delay values `τs`. For each of
  the `τs`'s the β-statistic gets computed.
* `w::Int = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `fnn_thres::Real= 0.05`: A threshold value defining a sufficiently small fraction
  of false nearest neighbors, in order to the let algorithm terminate and stop
  the embedding procedure (`0 ≤ fnn_thres < 1).
* `r::Real = 2`: The threshold for the tolerable relative increase of the distance
  between the nearest neighbors, when increasing the embedding dimension.
* `max_num_of_cycles = 50`: The algorithm will stop after that many cycles no matter what.

## Description
The method works iteratively and gradually builds the final embedding `Y`.
Based on the [`beta_statistic`](@ref) the algorithm picks an optimal delay
value `τ` for each embedding cycle as the global maximum of `β`. In case of
multivariate embedding, i.e. when embedding a set of time series (`s::Dataset`),
the optimal delay value `τ` is chosen as the maximum from all maxima's of all
considered `β`-statistics for each possible timeseries. The range of
considered delay values is determined in `τs` and for the nearest neighbor
search we respect the Theiler window `w`.

After each embedding cycle the FNN-statistic `FNNs` [^Hegger1999][^Kennel1992] is being
checked and as soon as this statistic drops below the threshold `fnn_thres`,
the algorithm terminates. In order to increase the practability of the method the
algorithm also terminates when the FNN-statistic `FNNs` increases.

The final embedding is returned as `Y`. The chosen delay values for each embedding
cycle are stored in the `τ_vals` and the according timeseries index chosen for the
the respective according delay value in `τ_vals` is stored in `ts_vals`.
`βS, FNNs` are returned for clarity and double-checking, since they are computed anyway.

[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
[^Hegger1999]: Hegger, Rainer and Kantz, Holger (1999). [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
[^Kennel1992]: Kennel, M. B., Brown, R., Abarbanel, H. D. I. (1992). [Determining embedding dimension for state-space reconstruction using a geometrical construction. Phys. Rev. A 45, 3403] (https://doi.org/10.1103/PhysRevA.45.3403).
"""
function mdop_embedding(s::Vector{T};
        τs = 0:50 , w::Int = 1, fnn_thres::Real = 0.05,
        max_num_of_cycles = 50,
        r::Real = 2) where {T<:Real}

    @assert 0 ≤ fnn_thres < 1 "Please select a valid breaking criterion, `fnn_thres` ∈ [0 1)"
    @assert all(x -> x ≥ 0, τs)
    metric = Euclidean()
    s = regularize(s) # especially important for fnn-computation
    # define actual phase space trajectory
    Y_act = Dataset(s)

    # compute nearest neighbor distances in the first embedding dimension for
    # FNN statistic
    vtree = KDTree(Y_act, metric)
    allNNidxs, NNdist_old = all_neighbors(vtree, Y_act, 1:length(Y_act), 1, w)

    # set a flag, in order to tell the while loop when to stop. Each loop
    # stands for encountering a new embedding dimension
    flag, counter = true, 1

    τ_vals = Int64[0]
    ts_vals = Int64[1]
    FNNs = Float64[]
    βS = Array{T}(undef, length(τs), max_num_of_cycles)

    # loop over increasing embedding dimensions until some break criterion will
    # tell the loop to stop/break
    while flag
        Y_act, NNdist_new = mdop_embedding_cycle!(
            Y_act, flag, s, τs, w, counter, βS, τ_vals, metric, r,
            FNNs, fnn_thres, ts_vals, NNdist_old
        )
        flag = mdop_break_criterion(FNNs, counter, fnn_thres, max_num_of_cycles)
        counter += 1
        # for the next embedding cycle save the distances of NN for this dimension
        NNdist_old = NNdist_new
    end
    return Y_act[:,1:counter-1], τ_vals[1:counter-1], ts_vals[1:counter-1], FNNs[1:counter-1], βS[:,1:counter-1]
end

# Here we separate the inner core loop of the mdop_embedding into another function
# not only for clarity of source code but also to introduce a function barrier
# for the type instability of `Y → Y_act`
function mdop_embedding_cycle!(
        Y, flag, s, τs, w, counter, βS, τ_vals, metric, r,
        FNNs, fnn_thres, ts_vals, NNdist_old
    )

    β = beta_statistic(Y, s, τs, w)
    βS[:, counter] = β

    # determine the optimal tau value from the β-statistic
    maxi, max_idx = findmax(β)
    # store chosen delay (and chosen time series)
    push!(τ_vals, τs[max_idx])
    push!(ts_vals, 1)

    # create phase space vector for this embedding cycle
    Y_act = DelayEmbeddings.hcat_lagged_values(Y,s,τ_vals[counter+1])
    vtree = KDTree(Y_act, metric)
    allNNidxs, NNdist_new = all_neighbors(vtree, Y_act, 1:length(Y_act), 1, w)

    # compute FNN-statistic and store vals, while also
    # truncating distance-vector to the "new" length of the actual embedding vector
    fnns = fnn_embedding_cycle(view(NNdist_old, 1:length(Y_act)), NNdist_new, r)
    push!(FNNs,fnns)
    return Y_act, NNdist_new
end

function mdop_break_criterion(FNNs, counter, fnn_thres, max_num_of_cycles)
    flag = true
    if FNNs[counter] ≤ fnn_thres
        flag = false
        println("Algorithm stopped due to sufficiently small FNNs. "*
                "Valid embedding achieved ✓.")
    end
    if counter > 1 && FNNs[counter] > FNNs[counter-1]
        flag = false
        println("Algorithm stopped due to rising FNNs. "*
                "Valid embedding achieved ✓.")
    end
    if max_num_of_cycles == counter
        println("Algorithm stopped due to hitting max cycle number. "*
                "Valid embedding NOT achieved ⨉.")
        flag = false
    end
    return flag
end


"""
    beta_statistic(Y::Dataset, s::Vector) [, τs, w]) → β
Compute the β-statistic `β` for input state space trajectory `Y` and a timeseries `s`
according to Nichkawde [^Nichkawde2013], based on estimating derivatives on a
projected manifold. For a range of delay values `τs`, `β` gets computed and its
maximum over all considered `τs` serves as the optimal delay considered in this
embedding cycle.

Arguments `τs, w` as in [`mdop_embedding`](@ref).

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
function beta_statistic(Y::Dataset{D,T}, s::Vector{T}, τs = 0:50, w::Int = 1) where {D, T<:Real}
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

    β = mean(log10.(Δϕ); dims=1) # β statistic is the average of the logarithms, see Eq. 16
    return vec(β)
end


"""
    hcat_lagged_values(Y, s::Vector, τ::Int) -> Z
Add the `τ` lagged values of the timeseries `s` as additional component to `Y`
(`Vector` or `Dataset`), in order to form a higher embedded
dataset `Z`. The dimensionality of `Z` is thus equal to that of `Y` + 1.
"""
function hcat_lagged_values(Y::Dataset{D,T}, s::Vector{T}, τ::Int) where {D, T<:Real}
    N = length(Y)
    @assert N ≤ length(s)
    M = N - τ
    data = Vector{SVector{D+1, T}}(undef, M)
    @inbounds for i in 1:M
        data[i] = SVector{D+1, T}(Y[i]..., s[i+τ])
    end
    return Dataset{D+1, T}(data)
end

function hcat_lagged_values(Y::Vector{T}, s::Vector{T}, τ::Int) where {T<:Real}
    N = length(Y)
    @assert N ≤ length(s)
    M = N - τ
    return Dataset(view(s, 1:M), view(s, τ+1:N))
end



"""
    mdop_maximum_delay(s, tw = 1:50, samplesize = 1.0)) -> τ_max, L
Compute an upper bound for the search of optimal delays, when using `mdop_embedding`
[`mdop_embedding`](@ref) or `beta_statistic` [`beta_statistic`](@ref).

## Description
The input time series `s` gets embedded with unit lag and increasing dimension,
for dimensions (or time windows) `tw` (`RangeObject`).
For each of such a time window the `L`-statistic from Uzal et al. [^Uzal2011]
will be computed. `samplesize` determines the fraction of points to be
considered in the computation of `L` (see [`uzal_cost`](@ref)). When this
statistic reaches its global minimum the maximum delay value `τ_max` gets
returned. When `s` is a multivariate `Dataset`, `τ_max` will becomputed for all
timeseries of that Dataset and the maximum value will be returned. The returned
`L`-statistic has size `(length(tw), size(s,2))`.

[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
[^Uzal2011]: Uzal, L. C., Grinblat, G. L., Verdes, P. F. (2011). [Optimal reconstruction of dynamical systems: A noise amplification approach. Physical Review E 84, 016223](https://doi.org/10.1103/PhysRevE.84.016223).
"""
function mdop_maximum_delay(s::Vector{T}, tw=1:50, samplesize::Real=1) where {T<:Real}
    @assert all(x -> x ≥ 0, tw)
    L = zeros(T, length(tw))
    counter = 1
    for i in tw
        i==1 ? Y_act = Dataset(s) : Y_act = embed(s,i,1)
        L[counter] = uzal_cost(Y_act; samplesize = samplesize)
        counter +=1
    end
    _, τ_max = findmin(L)
    return tw[τ_max], L
end

function mdop_maximum_delay(s::Dataset{D,T}, tw=1:50, samplesize::Real=1) where {D, T<:Real}
    τs_max = zeros(Int,D)
    Ls = zeros(T, length(tw), D)
    @inbounds for i = 1:D
        τs_max[i], Ls[:, i] = mdop_maximum_delay(vec(s[:,i]), tw, samplesize)
    end
    return maximum(τs_max), Ls
end
