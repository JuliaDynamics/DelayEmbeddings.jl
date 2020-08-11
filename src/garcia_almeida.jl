using Neighborhood
using NearestNeighbors
using StatsBase
using Distances
using Peaks

export garcia_almeida_embed
export garcia_embedding_cycle


"""
    garcia_almeida_embed(s; kwargs...) → Y, τ_vals, ts_vals, FNNs [,NS]
is a unified approach to properly embed a time series (`Vector` type) or a
set of time series (`Dataset` type) based on the papers of Garcia & Almeida
[^Garcia2005a],[^Garcia2005b].

## Keyword arguments

* `τs= 0:50`: Considered delay values `τs` (in sampling time units). For each of
  the `τs`'s the N-statistic gets computed.
* `w::Int = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `r1::Float64 = 10.0`: The threshold, which defines the factor of tolerable stretching for
  the d_E1-statistic (see algorithm description in [`garcia_embedding_cycle`](@ref)).
* `r2::Float64 = 2.0`: The threshold for the tolerable relative increase of the distance
  between the nearest neighbors, when increasing the embedding dimension.
* `fnn_thres::Float64= 0.05`: A threshold value defining a sufficiently small fraction
  of false nearest neighbors, in order to the let algorithm terminate and stop
  the embedding procedure (`0 > fnn_thres > 1).
* `T::Int = 1`: The forward time step (in sampling units) in order to compute the
  `d_E2`-statistic (see algorithm description). Note that in the paper this is
  not a free parameter and always set to `T=1`.
* `metric = Euclidean()`: metric used for finding nearest neigbhors in the input
  phase space trajectory `Y`.
* `Ns:Bool = false`: When set `true`, the function also returnes the `N`-statistics
  of all embedding cycles.


## Description
The method works iteratively and gradually builds the final embedding vectors
`Y`. Based on the `N`-statistic [`garcia_embedding_cycle`](@ref) the algorithm
picks an optimal delay value `τ` for each embedding cycle as the first local
minimum of `N`. In case of multivariate embedding, i.e. when embedding a set of
time series (`s::Dataset`), the optimal delay value `τ` is chosen as the first
minimum from all minimum's of all considered `N`-statistics for each embedding
cycle. The range of considered delay values is determined in `τs` and for the
nearest neighbor search we respect the Theiler window `w`. After each embedding
cycle the FNN-statistic `FNNs` [^Hegger1999][^Kennel1992] is being checked and
as soon as this statistic drops below the threshold `fnn_thres`, the algorithm
breaks. In order to increase the  practability of the method the algorithm also
breaks, when the FNN-statistic `FNNs` increases . The final embedding vector is
stored in `Y` (`Dataset`). The chosen delay values for each embedding cycle are
stored in the `τ_vals` and the according time series number chosen for the
according delay value in `τ_vals` is stored in `ts_vals`. For univariate
embedding (`s::Vector`) `ts_vals` is a vector of ones of length `τ_vals`,
because there is simply just one time series to choose from. If `Ns=true` then
the function also returns the `N`-statistic `NS` for each embedding cycle as an
`Array` of `Vector`s.

[^Garcia2005a]: Garcia, S. P., Almeida, J. S. (2005). [Nearest neighbor embedding with different time delays. Physical Review E 71, 037204](https://doi.org/10.1103/PhysRevE.71.037204).
[^Garcia2005b]: Garcia, S. P., Almeida, J. S. (2005). [Multivariate phase space reconstruction by nearest neighbor embedding with different time delays. Physical Review E 72, 027205](https://doi.org/10.1103/PhysRevE.72.027205).
"""
function garcia_almeida_embed(s::Vector{Float64}; τs = 0:50 , w::Int = 1,
    r1::Float64 = 10.0, r2::Float64 = 2.0, fnn_thres::Float64 = 0.05,
    T::Int = 1, metric = Euclidean(), Ns::Bool=false)
    @assert 0 < fnn_thres < 1 "Please select a valid breaking criterion, i.e. a threshold value `fnn_thres` ∈ (0 1)"
    @assert all(x -> x ≥ 0, τs)

    max_num_of_cycles = 50 # assure that the algorithm will break after 50 embedding cycles

    # normalize input time series (especially important for fnn-computation)
    s = (s.-mean(s))./std(s)
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
    Ns ? NS = Array{Float64, 2}(undef, length(τs), max_num_of_cycles) : nothing

    # loop over increasing embedding dimensions until some break criterion will
    # tell the loop to stop/break
    while flag

        # get the N-statistic
        N, NNdistances = garcia_embedding_cycle(Y_act, s; τs = τs , r = r1,
            T = T, w = w, metric = metric)
        Ns ? NS[:,cnt] = N : nothing

        # determine the optimal tau value from the N-statistic
        min_idx = Peaks.minima(N)

        if length(min_idx) == 0
            flag = false
            display("Algorithm could not pick a delay value from N-statistic. Increase the considered delays in `τs`-input. NO valid embedding achieved")
            continue
        end
        # store chosen delay (and chosen time series)
        push!(τ_vals,τs[min_idx[1]])
        push!(ts_vals,1)

        # create phase space vector for this embedding cycle
        Y_act = embed_one_cycle(Y_act,s,τ_vals[cnt+1])

        # compute nearest neighbor distances in the new embedding dimension for
        # FNN statistic
        NNdist_new = NNdistances[τ_vals[cnt+1]+1]

        NNdist_new = [NNdist_new[i][1] for i = 1:length(NNdist_new)]
        # truncate distance-vector to the "new" length of the actual embedding vector
        NNdist_old = NNdist_old[1:length(Y_act)-T]
        NNdist_old = [NNdist_old[i][1] for i = 1:length(NNdist_old)]

        # compute FNN-statistic and store vals
        fnns = fnn(NNdist_old,NNdist_new;r=r2)
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
        max_num_of_cycles == cnt ? flag = false : nothing

        cnt += 1
        # for the next embedding cycle save the distances of NN for this dimension
        NNdist_old = NNdist_new
    end

    if Ns
        return Y_act[:,1:cnt-1], τ_vals[1:cnt-1], ts_vals[1:cnt-1], FNNs[1:cnt-1], NS[:,1:cnt-1]
    else
        return Y_act[:,1:cnt-1], τ_vals[1:cnt-1], ts_vals[1:cnt-1], FNNs[1:cnt-1]
    end
end




## core: Garcia-Almeida-method for one arbitrary embedding cycle

"""
    garcia_embedding_cycle(Y,s; kwargs...) → N, d_E1 (`Array`, `Array of Arrays`)
Performs one embedding cycle according to the method proposed in [^Garcia2005a]
for a given phase space trajectory `Y` (of type `Dataset`) and a time series `s
(of type `Vector`). Returns the proposed N-Statistic `N` and all nearest
neighbor distances `d_E1` for each point of the input phase space trajectory
`Y`. Note that `Y` is a single time series in case of the first embedding cycle.

## Keyword arguments
* `τs= 0:50`: Considered delay values `τs` (in sampling time units). For each of
  the `τs`'s the N-statistic gets computed.
* `r::Float64 = 10.0`: The threshold, which defines the factor of tolerable stretching for
  the d_E1-statistic (see algorithm description).
* `T::Int = 1`: The forward time step (in sampling units) in order to compute the
  `d_E2`-statistic (see algorithm description). Note that in the paper this is
  not a free parameter and always set to `T=1`.
* `w::Int = 0`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors. Note that in the paper this is not
  a free parameter and always `w=0`.
* `metric = Euclidean()`: metric used for finding nearest neigbhors in the input
  phase space trajectory `Y`.

## Description
For a range of possible delay values `τs` one constructs a temporary embedding
matrix. That is, one concatenates the input phase space trajectory `Y` with the
`τ`-lagged input time series `s`. For each point on the temporary trajectory one
computes its nearest neighbor, which is denoted as the `d_E1`-statistic for a
specific `τ`. Now one considers the distance between the reference point and its
nearest neighbor `T` sampling units ahead and calls this statistic `d_E2`.
[^Garcia2005a] strictly use `T=1`, so they forward each reference point and its
corresponding nearest neighbor just by one (!) sampling unit. Here it is a free
parameter.

The `N`-statistic is then the fraction of `d_E2`/`d_E1`-pairs which exceed a
threshold `r`.

Plotted vs. the considered `τs`-values it is proposed to pick the `τ`-value for
this embedding cycle as the value, where `N` has its first local minimum.

[^Garcia2005a]: Garcia, S. P., Almeida, J. S. (2005). [Nearest neighbor embedding with different time delays. Physical Review E 71, 037204](https://doi.org/10.1103/PhysRevE.71.037204).
"""
function garcia_embedding_cycle(Y::Vector{Float64}, s::Vector{Float64}; τs = 0:50 , r::Float64 = 10.0,
    T::Int = 1, w::Int = 1, metric = Euclidean())

    # assert a minimum length of the input time series
    @assert length(s) ≥ length(Y) "The length of the input time series `s` must be at least the length of the input trajectory `Y` "
    @assert all(x -> x ≥ 0, τs)
    τ_max = maximum(τs)

    # preallocation of output
    N_stat = zeros(length(τs))
    NN_distances = [AbstractArray[] for j in τs]

    for (i,τ) in enumerate(τs)
        # build temporary embedding matrix Y_temp
        Y_temp = embed_one_cycle(Y,s,τ)
        NN = length(Y_temp)     # length of the temporary phase space vector
        N = NN-T               # accepted length w.r.t. the time horizon `T

        vtree = KDTree(Y_temp[1:N], metric) # tree for input data
        # nearest neighbors (d_E1)
        allNNidxs, d_E1 = all_neighbors(vtree, Y_temp[1:N], 1:N, 1, w)
        # save d_E1-statistic
        #push!(NN_distances[i],hcat(d_E1...))
        NN_distances[i] = d_E1

        # for each point, consider its next iterate and compute the distance to
        # its former(!) nearest neighbour iterate
        newfididxs = [j .+ T for j = 1:N]
        newNNidxs = [j .+ T for j in allNNidxs]

        # compute d_E2
        d_E2 = [evaluate(metric,Y_temp[newfididxs[i]],Y_temp[newNNidxs[i][1]]) for i = 1:N]

        # look at ratio d_E2/d_E1
        distance_ratios = d_E2./vcat(d_E1...)

        # number of distance ratio larger than r
        N_stat[i] = sum(distance_ratios.>r)/NN

    end

    return N_stat, NN_distances
end


"""
    embed_one_cycle(Y, s::Vector, τ::Int) -> Y_new
Add the `τ` lagged values of the time series `s` as additional component to the
time series `Y` (`Vector` or `Dataset`), in order to form a higher embedded
vector `Y_new`. The dimensionality of `Y_new` , thus, equals the dimensionality
of `Y+1`.
"""
function embed_one_cycle(Y::Dataset{D,T}, s::Vector{T}, τ::Int) where {D, T<:Real}
    N = length(Y)
    @assert N <= length(s)
    M = N - τ
    return Dataset(hcat(view(Matrix(Y), 1:M, :), view(s, τ+1:N)))
end

function embed_one_cycle(Y::Vector{T}, s::Vector{T}, τ::Int) where {D, T<:Real}
    N = length(Y)
    @assert N <= length(s)
    M = N - τ
    return Dataset(hcat(view(s, 1:M), view(s, τ+1:N)))
end


"""
    fnn(NNdist,NNdistnew,r::Float64=2.0) -> FNNs
fnn computes the amount of false nearest neighbors when adding another component
to a given (vector-) time series. This new component is the `τ`-lagged version
of a univariate time series. 'NNdist' is storing the distances of the nearest
neighbor for all considered fiducial points and 'NNdistnew' is storing the
distances of the nearest neighbor for each fiducial point in one embedding
dimension higher using a given `τ`. The obligatory threshold `r` is by default
set to 2.
[^Hegger1999]: Hegger, Rainer and Kantz, Holger (1999). [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
function fnn(NNdist::T,NNdistnew::T;r::Float64=2.0) where {T}
    @assert length(NNdist) == length(NNdistnew) "Both input vectors need to store the same number of distances."

    # convert array of arrays into simple vectors, since we only look at K=1
    NN_old = zeros(length(NNdist))
    NN_new = zeros(length(NNdistnew))
    for i = 1:length(NNdist)
        NN_old[i]=NNdist[i][1]
        NN_new[i]=NNdistnew[i][1]
    end

    # first ratio
    ratio1 = 1/r    # since we input only z-standardized data with unit
    # construct statistic
    ratio2 = NN_new./NN_old

    cond1 = ratio2 .> r
    cond2 = NN_old .< ratio1

    fnns = sum(cond1.*cond2)
    fnns2 = sum(NN_old .< ratio1)

    # store fraction of valid nearest neighbors
    if fnns2 == 0
        FNN = NaN;
    else
        FNN = fnns/fnns2
    end

    return FNN
end
