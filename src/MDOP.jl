using Neighborhood
using StatsBase
using Distances

export beta_statistic
export MDOP
export estimate_maximum_delay


"""
    MDOP(s::Vector; kwargs...) → Y, τ_vals, ts_vals, FNNs [,βS]
MDOP is a unified approach to properly embed a time series (`Vector` type) or a
set of time series (`Dataset` type) based on the paper of Chetan Nichkawde
[^Nichkawde2013].

## Keyword arguments

* `τs= 0:50`: Considered delay values `τs` (in sampling time units). For each of
  the `τs`'s the β-statistic gets computed.
* `w::Int = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `fnn_thres::Real= 0.05`: A threshold value defining a sufficiently small fraction
  of false nearest neighbors, in order to the let algorithm terminate and stop
  the embedding procedure (`0 <= fnn_thres < 1).
* `r::Real = 2`: The threshold for the tolerable relative increase of the distance
  between the nearest neighbors, when increasing the embedding dimension.
* `β_condition::Bool = false`: When set `true`, the function also returnes the `β`-statistics
  of all embedding cycles.

## Description
The method works iteratively and gradually builds the final embedding vectors
`Y`. Based on the [`beta_statistic`](@ref) the algorithm picks an optimal delay
value `τ` for each embedding cycle as the global maximum of `β`. In case of
multivariate embedding, i.e. when embedding a set of time series (`s::Dataset`),
the optimal delay value `τ` is chosen as the maximum from all maxima's of all
considered `β`-statistics for each embedding cycle. The range of
considered delay values is determined in `τs` and for the nearest neighbor
search we respect the Theiler window `w`. After each embedding cycle the FNN-
statistic `FNNs` [^Hegger1999][^Kennel1992] is being checked and as soon as this statistic
drops below the threshold `fnn_thres`, the algorithm breaks. In order to
increase the  practability of the method the algorithm also breaks, when the
FNN-statistic `FNNs` increases . The final embedding vector is stored in `Y`
(`Dataset`). The chosen delay values for each embedding cycle are stored in the
`τ_vals` and the according time series number chosen for the according delay
value in `τ_vals` is stored in `ts_vals`. For univariate embedding (`s::Vector`)
`ts_vals` is a vector of ones of length `τ_vals`, because there is simply just
one `β`-statistic for each embedding cycle as an `Array` of `Vector`s.

[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
[^Hegger1999]: Hegger, Rainer and Kantz, Holger (1999). [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
[^Kennel1992]: Kennel, M. B., Brown, R., Abarbanel, H. D. I. (1992). [Determining embedding dimension for state-space reconstruction using a geometrical construction. Phys. Rev. A 45, 3403] (https://doi.org/10.1103/PhysRevA.45.3403).
"""
function MDOP(s::Vector{<:Real}; τs = 0:50 , w::Int = 1, fnn_thres::Real = 0.05,
    r::Real = 2, β_condition::Bool=false)
    @assert 0 <= fnn_thres < 1 "Please select a valid breaking criterion, i.e. a threshold value `fnn_thres` ∈ [0 1)"
    @assert all(x -> x ≥ 0, τs)

    T = eltype(s)

    max_num_of_cycles = 50 # assure that the algorithm will break after 50 embedding cycles

    metric = Euclidean()
    # normalize input time series (especially important for fnn-computation)
    s .= (s.-mean(s))./std(s)
    # define actual phase space trajectory
    Y_act = Dataset(s)

    # compute nearest neighbor distances in the first embedding dimension for
    # FNN statistic
    vtree = KDTree(Y_act, metric)
    allNNidxs, NNdist_old = all_neighbors(vtree, Y_act, 1:length(Y_act), 1, w)

    # set a flag, in order to tell the while loop when to stop. Each loop
    # stands for encountering a new embedding dimension
    flag = true

    # set index-counter for the while loop
    cnt = 1;

    # preallocate output variables
    τ_vals = zeros(Int,1)
    ts_vals = ones(Int,1)
    FNNs = zeros(Float64,1)
    if β_condition; βS = Array{T}(undef, length(τs), max_num_of_cycles); end

    # loop over increasing embedding dimensions until some break criterion will
    # tell the loop to stop/break
    while flag

        # get the β-statistic
        β = beta_statistic(Y_act, s; τs = τs , w = w)
        if β_condition; βS[:, cnt] = β; end

        # determine the optimal tau value from the β-statistic
        maxi, max_idx = findmax(β)
        # store chosen delay (and chosen time series)
        push!(τ_vals, τs[max_idx])
        push!(ts_vals, 1)

        # create phase space vector for this embedding cycle
        Y_act = DelayEmbeddings.embed_one_cycle(Y_act,s,τ_vals[cnt+1])

        # compute nearest neighbor distances in the new embedding dimension for
        # FNN statistic
        vtree = KDTree(Y_act, metric)
        allNNidxs, NNdist_new = all_neighbors(vtree, Y_act, 1:length(Y_act), 1, w)

        # truncate distance-vector to the "new" length of the actual embedding vector
        NNdist_old = NNdist_old[1:length(Y_act)]

        # compute FNN-statistic and store vals
        fnns = fnn_embedding_cycle(NNdist_old,NNdist_new;r=r)
        cnt == 1 ? FNNs[1] = fnns : push!(FNNs,fnns)

        # break criterion 1
        if FNNs[cnt] <= fnn_thres
            flag = false
            display("Algorithm stopped due to sufficiently small FNNs. VALID embedding achieved.")
        end
        # break criterion 2
        if cnt > 1 && FNNs[cnt]>FNNs[cnt-1]
            flag = false
            display("Algorithm stopped due to rising FNNs.")
        end
        # break criterion 3 (maximum embedding cycles reached)
        if max_num_of_cycles == cnt; flag = false; end

        cnt += 1
        # for the next embedding cycle save the distances of NN for this dimension
        NNdist_old = NNdist_new
    end

    if β_condition
        return Y_act[:,1:cnt-1], τ_vals[1:cnt-1], ts_vals[1:cnt-1], FNNs[1:cnt-1], βS[:,1:cnt-1]
    else
        return Y_act[:,1:cnt-1], τ_vals[1:cnt-1], ts_vals[1:cnt-1], FNNs[1:cnt-1]
    end
end

"""
    MDOP(s::Dataset; kwargs...) → Y, τ_vals, ts_vals, FNNs
MDOP is a unified approach to properly embed a time series based on the paper of
Chetan Nichkawde [^Nichkawde2013].
[`uzal_cost`](@ref)


[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
"""
function MDOP()
    #tbd

end


"""
    beta_statistic(Y::Dataset, s::Vector); kwargs...) → β
Compute the β-statistic `β` for input state space trajectory `Y` and a timeseries `s`
according to Nichkawde [^Nichkawde2013], based on estimating derivatives on a
projected manifold. For a range of delay values `τs`, `β` gets computed and its
maximum over all considered `τs` serves as the optimal delay considered in this
embedding cycle.

## Keyword arguments

* `τs= 0:50`: Considered delay values `τs` (in sampling time units). For each of
  the `τs`'s the β-statistic gets computed.
* `w::Int= 1`: Theiler window (neighbors in time with index `w` close to the point,
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
function beta_statistic(Y::Dataset{D,T}, s::Vector{T}; τs = 0:50 , w::Int = 1) where {D, T<:Real}
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


"""
    embed_one_cycle(Y, s::Vector, τ::Int) -> Z
Add the `τ` lagged values of the time series `s` as additional component to `Y`
(`Vector` or `Dataset`), in order to form a higher embedded
dataset `Z`. The dimensionality of `Z` is thus equal to that of `Y` + 1.
"""
function embed_one_cycle(Y::Dataset{D,T}, s::Vector{T}, τ::Int) where {D, T<:Real}
    N = length(Y)
    @assert N ≤ length(s)
    M = N - τ
    data = Vector{SVector{D+1, T}}(undef, M)
    @inbounds for i in 1:M
        data[i] = SVector{D+1, T}(Y[i]..., s[i+τ])
    end
    return Dataset(data)
end

function embed_one_cycle(Y::Vector{T}, s::Vector{T}, τ::Int) where {T<:Real}
    N = length(Y)
    @assert N ≤ length(s)
    M = N - τ
    return Dataset(view(s, 1:M), view(s, τ+1:N))
end



"""
    estimate_maximum_delay(s::Vector; tw=1:50, samplesize::Float=1.0)) -> τ_max, L
Compute an upper bound for the search of optimal delays, when using `MDOP`
[`MDOP`](@ref) or `beta_statistic` [`beta_statistic`](@ref).

## Description
The input time series `s` gets embedded with unit lag and increasing dimension,
for dimensions (or time windows) `tw` (`RangeObject`).
For each of such a time window the `L`-statistic from Uzal et al. [^Uzal2011]
will be computed. `samplesize` determines the fraction of points to be
considered in the computation of `L` (see [`uzal_cost`](@ref)). When this
statistic reaches its global minimum the maximum delay value `τ_max` gets
returned. When `s` is a multivariate `Dataset`, `τ_max` will becomputed for all
time series of that Dataset and the maximum value will be returned. The returned
`L`-statistic is a Dataset of size (length(tw)*size(s,2)).

[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
[^Uzal2011]: Uzal, L. C., Grinblat, G. L., Verdes, P. F. (2011). [Optimal reconstruction of dynamical systems: A noise amplification approach. Physical Review E 84, 016223](https://doi.org/10.1103/PhysRevE.84.016223).
"""
function estimate_maximum_delay(s::Vector{T}; tw=1:50, samplesize::Real=1) where {T<:Real}
    L = zeros(T,length(tw))
    # loop over time windows
    cnt = 1
    for i in tw
        Y_act = embed(s,i,1)
        L[cnt] = uzal_cost(Y_act; samplesize = samplesize)
        cnt +=1
    end
    ~, τ_max = findmin(L)
    return tw[τ_max], L
end

function estimate_maximum_delay(s::Dataset{D,T}; tw=1:50, samplesize::Real=1) where {D, T<:Real}
    #M = size(s,2)
    τs_max = zeros(Int,D)
    Ls = zeros(T,length(tw),D)
    for i = 1:D
        τs_max[i], L = estimate_maximum_delay(vec(s[:,i]); tw = tw, samplesize = samplesize)
        Ls[:,i] = L
    end
    return maximum(τs_max), Dataset(Ls)
end
