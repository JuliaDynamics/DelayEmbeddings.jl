using Neighborhood
using Distances

export garcia_almeida_embedding
export n_statistic


"""
    garcia_almeida_embedding(s; kwargs...) → Y, τ_vals, ts_vals, FNNs ,NS
A unified approach to properly embed a time series (`Vector` type) or a
set of time series (`StateSpaceSet` type) based on the papers of Garcia & Almeida
[^Garcia2005a],[^Garcia2005b].

## Keyword arguments

* `τs= 0:50`: Possible delay values `τs` (in sampling time units). For each of
  the `τs`'s the N-statistic gets computed.
* `w::Int = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `r1 = 10`: The threshold, which defines the factor of tolerable stretching for
  the d_E1-statistic.
* `r2 = 2`: The threshold for the tolerable relative increase of the distance
  between the nearest neighbors, when increasing the embedding dimension.
* `fnn_thres= 0.05`: A threshold value defining a sufficiently small fraction
  of false nearest neighbors, in order to the let algorithm terminate and stop
  the embedding procedure (`0 ≤ fnn_thres < 1).
* `T::Int = 1`: The forward time step (in sampling units) in order to compute the
  `d_E2`-statistic (see algorithm description). Note that in the paper this is
  not a free parameter and always set to `T=1`.
* `metric = Euclidean()`: metric used for finding nearest neigbhors in the input
  phase space trajectory `Y`.
* `max_num_of_cycles = 50`: The algorithm will stop after that many cycles no matter what.


## Description
The method works iteratively and gradually builds the final embedding vectors
`Y`. Based on the `N`-statistic the algorithm
picks an optimal delay value `τ` for each embedding cycle as the first local
minimum of `N`. In case of multivariate embedding, i.e. when embedding a set of
time series (`s::StateSpaceSet`), the optimal delay value `τ` is chosen as the first
minimum from all minimum's of all considered `N`-statistics for each embedding
cycle. The range of considered delay values is determined in `τs` and for the
nearest neighbor search we respect the Theiler window `w`. After each embedding
cycle the FNN-statistic `FNNs` [^Hegger1999][^Kennel1992] is being checked and
as soon as this statistic drops below the threshold `fnn_thres`, the algorithm
breaks. In order to increase the  practability of the method the algorithm also
breaks, when the FNN-statistic `FNNs` increases . The final embedding vector is
stored in `Y` (`StateSpaceSet`). The chosen delay values for each embedding cycle are
stored in the `τ_vals` and the according time series number chosen for the
according delay value in `τ_vals` is stored in `ts_vals`. For univariate
embedding (`s::Vector`) `ts_vals` is a vector of ones of length `τ_vals`,
because there is simply just one time series to choose from. The function also
returns the `N`-statistic `NS` for each embedding cycle as an `Array` of
`Vector`s.

Notice that we were *not* able to reproduce the figures from the papers with our
implementation (which nevertheless we believe is the correct one).

[^Garcia2005a]: Garcia, S. P., Almeida, J. S. (2005). [Nearest neighbor embedding with different time delays. Physical Review E 71, 037204](https://doi.org/10.1103/PhysRevE.71.037204).
[^Garcia2005b]: Garcia, S. P., Almeida, J. S. (2005). [Multivariate phase space reconstruction by nearest neighbor embedding with different time delays. Physical Review E 72, 027205](https://doi.org/10.1103/PhysRevE.72.027205).
"""
function garcia_almeida_embedding(s::Vector{F}; τs = 0:50 , w::Int = 1,
    r1::Real = 10, r2::Real = 2, fnn_thres::Real = 0.05,
    T::Int = 1, metric = Euclidean(), max_num_of_cycles = 50) where {F<:Real}

    @assert 0 ≤ fnn_thres < 1 "Please select a valid breaking criterion, i.e. a threshold value `fnn_thres` ∈ [0 1)"
    @assert all(x -> x ≥ 0, τs)
    s_orig = s
    s = standardize(s) # especially important for fnn-computation
    # define actual phase space trajectory
    Y_act = StateSpaceSet(s)

    # compute nearest neighbor distances in the first embedding dimension for
    # FNN statistic
    vtree = KDTree(Y_act, metric)
    allNNidxs, NNdist_old = all_neighbors(vtree, Y_act, 1:length(Y_act), 1, w)

    # set a flag, in order to tell the while loop when to stop. Each loop
    # stands for encountering a new embedding dimension
    flag, counter = true, 1

    # preallocate output variables
    τ_vals = Int64[0]
    ts_vals = Int64[1]
    FNNs = Float64[]
    NS = Array{F}(undef, length(τs), max_num_of_cycles)

    # loop over increasing embedding dimensions until some break criterion will
    # tell the loop to stop/break
    while flag
        Y_act, NNdist_new = garcia_embedding_cycle!(
            Y_act, flag, s, τs, w, counter, NS, τ_vals, metric, r1, r2, T,
            FNNs, fnn_thres, ts_vals, NNdist_old
        )

        flag = garcia_almeida_break_criterion(FNNs, counter, fnn_thres, max_num_of_cycles)
        counter += 1
        # for the next embedding cycle save the distances of NN for this dimension
        NNdist_old = NNdist_new
    end

    # construct final reconstruction vector
    Y_final = s_orig
    for i = 2:length(τ_vals[1:counter-1])
        Y_final = DelayEmbeddings.hcat_lagged_values(Y_final,s_orig,τ_vals[i])
    end

    return Y_final, τ_vals[1:counter-1], ts_vals[1:counter-1], FNNs, NS[:,1:counter-1]

end


function garcia_almeida_embedding(Y::AbstractStateSpaceSet{D, F}; τs = 0:50 , w::Int = 1,
    r1::Real = 10, r2::Real = 2, fnn_thres::Real = 0.05,
    T::Int = 1, metric = Euclidean(), max_num_of_cycles = 50) where {D, F<:Real}

    @assert 0 ≤ fnn_thres < 1 "Please select a valid breaking criterion, i.e. a threshold value `fnn_thres` ∈ [0 1)"
    @assert all(x -> x ≥ 0, τs)
    Y_orig = Y
    Y = standardize(Y) # especially important for fnn-computation
    # define actual phase space trajectory and NN-distances
    Y_act = []
    NNdist_old = []

    # set a flag, in order to tell the while loop when to stop. Each loop
    # stands for encountering a new embedding dimension
    flag, counter = true, 1

    # preallocate output variables
    τ_vals = Int64[0]
    ts_vals = Int64[1]
    FNNs = Float64[]
    NS = fill(zeros(F, length(τs), oldsize(Y,2)), 1, max_num_of_cycles)

    # loop over increasing embedding dimensions until some break criterion will
    # tell the loop to stop/break
    while flag
        Y_act, NNdist_new = garcia_multivariate_embedding_cycle!(
            Y_act, flag, Y, τs, w, counter, NS, τ_vals, metric, r1, r2, T,
            FNNs, fnn_thres, ts_vals, NNdist_old
        )

        flag = garcia_almeida_break_criterion(FNNs, counter, fnn_thres, max_num_of_cycles)
        counter += 1
        # for the next embedding cycle save the distances of NN for this dimension
        NNdist_old = NNdist_new
    end

    # construct final reconstruction vector
    Y_final = Y_orig[:,ts_vals[1]]
    for i = 2:length(τ_vals[1:counter-1])
        Y_final = DelayEmbeddings.hcat_lagged_values(Y_final,Y_orig[:,ts_vals[i]],τ_vals[i])
    end

    return Y_final, τ_vals[1:counter-1], ts_vals[1:counter-1], FNNs, NS[:,1:counter-1]

end




## core: Garcia-Almeida-method for one arbitrary embedding cycle

"""
    n_statistic(Y, s; kwargs...) → N, d_E1
Perform one embedding cycle according to the method proposed in [^Garcia2005a]
for a given phase space trajectory `Y` (of type `StateSpaceSet`) and a time series `s
(of type `Vector`). Return the proposed N-Statistic `N` and all nearest
neighbor distances `d_E1` for each point of the input phase space trajectory
`Y`. Note that `Y` is a single time series in case of the first embedding cycle.

## Keyword arguments
* `τs= 0:50`: Considered delay values `τs` (in sampling time units). For each of
  the `τs`'s the N-statistic gets computed.
* `r = 10`: The threshold, which defines the factor of tolerable stretching for
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
function n_statistic(Y::AbstractStateSpaceSet{D, F}, s::Vector{F}; τs = 0:50 , r::Real = 10,
    T::Int = 1, w::Int = 1, metric = Euclidean()) where {D, F<:Real}

    # assert a minimum length of the input time series
    @assert length(s) ≥ length(Y) "The length of the input time series `s` must be at least the length of the input trajectory `Y` "
    @assert all(x -> x ≥ 0, τs)
    τ_max = maximum(τs)

    # preallocation of output
    N_stat = zeros(length(τs))
    NN_distances = [AbstractArray[] for j in τs]

    for (i,τ) in enumerate(τs)
        # build temporary embedding matrix Y_temp
        Y_temp = hcat_lagged_values(Y,s,τ)
        NN = length(Y_temp)     # length of the temporary phase space vector
        N = NN-T               # accepted length w.r.t. the time horizon `T

        vtree = KDTree(Y_temp[1:N], metric) # tree for input data
        # nearest neighbors (d_E1)
        allNNidxs, d_E1 = all_neighbors(vtree, Y_temp[1:N], 1:N, 1, w)
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

# Here we separate the inner core loop of the mdop_embedding into another function
# not only for clarity of source code but also to introduce a function barrier
# for the type instability of `Y → Y_act`
function garcia_embedding_cycle!(
    Y_act, flag, s, τs, w, counter, NS, τ_vals, metric, r1, r2, T,
    FNNs, fnn_thres, ts_vals, NNdist_old)

    N, NNdistances = n_statistic(Y_act, s; τs = τs , r = r1,
        T = T, w = w, metric = metric)
    NS[:, counter] = N

    # determine the optimal tau value from the N-statistic
    min_idx = findlocalminima(N)

    # store chosen delay (and chosen time series)
    push!(τ_vals, τs[min_idx[1]])
    push!(ts_vals, 1)

    # create phase space vector for this embedding cycle
    Y_act = DelayEmbeddings.hcat_lagged_values(Y_act,s,τ_vals[counter+1])
    vtree = KDTree(Y_act, metric)
    allNNidxs, NNdist_new = all_neighbors(vtree, Y_act, 1:length(Y_act), 1, w)

    # compute FNN-statistic and store vals, while also
    # truncating distance-vector to the "new" length of the actual embedding vector
    fnns = fnn_embedding_cycle(view(NNdist_old, 1:length(Y_act)), NNdist_new, r2)
    push!(FNNs,fnns)
    return Y_act, NNdist_new
end


function garcia_multivariate_embedding_cycle!(
    Y_act, flag, Ys, τs, w, counter, NS, τ_vals, metric, r1, r2, T,
    FNNs, fnn_thres, ts_vals, NNdist_old
    )

    M = oldsize(Ys,2)
    # in the 1st cycle we have to check all (oldsize(Y,2)^2 combinations
    if counter == 1
        Y_act, NNdist_new = first_embedding_cycle!(M, Ys, τs, r1, r2, T, w,
                                                metric, τ_vals, ts_vals,
                                                NS, FNNs)

    # in all other cycles we just have to check (oldsize(Y,2)) combinations
    else
        Y_act, NNdist_new = embedding_cycle!(Y_act, M, counter, Ys, τs, r1, r2, T, w,
                                            metric, NNdist_old, τ_vals, ts_vals,
                                            NS, FNNs)
    end
    return Y_act, NNdist_new
end

"""
Perform the first embedding cycle of the multivariate embedding. Return the
actual reconstruction vector `Y_act` and all nearest neighbor distances
`NNdist_new` of `Y_act`.
"""
function first_embedding_cycle!(M, Ys, τs, r1, r2, T, w, metric, τ_vals, ts_vals,
                                NS, FNNs)
    counter = 1
    Ns = zeros(length(τs), M*M)
    n_counter = 1
    for ts = 1:M
        for ts2 = 1:M
            Ns[:,n_counter], _ = n_statistic(StateSpaceSet(Ys[:,ts]), Ys[:,ts2];
                                            τs = τs , r = r1, T = T, w = w,
                                            metric = metric)
            n_counter += 1
        end
    end
    # determine the optimal tau value from the N-statistic
    τ_opt, ts_opt1, ts_opt2 = choose_optimal_tau1_garcia(Ns, M)
    # store chosen delay, time series and according N-statistic
    push!(τ_vals, τs[τ_opt])
    push!(ts_vals, ts_opt1)
    push!(ts_vals, ts_opt2)
    NS[counter] = Ns[:,(ts_vals[counter]-1)*M+1:(ts_vals[counter]-1)*M+M]

    # create phase space vector for this embedding cycle
    Y_act = DelayEmbeddings.hcat_lagged_values(Ys[:,ts_vals[counter]],
                                Ys[:,ts_vals[counter+1]],τ_vals[counter+1])

    # compute nearest neighbor distances in the first embedding dimension for
    # FNN statistic
    Y_old = StateSpaceSet(Ys[:,ts_vals[counter]])
    vtree = KDTree(Y_old, metric)
    _, NNdist_old = all_neighbors(vtree, Y_old, 1:length(Y_old), 1, w)

    vtree = KDTree(Y_act, metric)
    _, NNdist_new = all_neighbors(vtree, Y_act, 1:length(Y_act), 1, w)

    # compute FNN-statistic and store vals, while also
    # truncating distance-vector to the "new" length of the actual embedding vector
    fnns = fnn_embedding_cycle(view(NNdist_old, 1:length(Y_act)), NNdist_new, r2)
    push!(FNNs,fnns)

    return Y_act, NNdist_new
end

"""
Perform an embedding cycle of the multivariate embedding, but the 1st. Return the
actual reconstruction vector `Y_act` and all nearest neighbor distances
`NNdist_new` of `Y_act`.
"""
function embedding_cycle!(Y_act, M, counter, Ys, τs, r1, r2, T, w, metric, NNdist_old,
                                            τ_vals, ts_vals, NS, FNNs)
    Ns = zeros(length(τs), M)
    for ts = 1:M
        Ns[:,ts], _ = n_statistic(Y_act, Ys[:,ts]; τs = τs , r = r1, T = T,
                                    w = w, metric = metric)
    end

    # determine the optimal tau value from the N-statistic
    τ_opt, ts_opt = choose_optimal_tau2_garcia(Ns)
    # store chosen delay, time series and according n-statistic
    push!(τ_vals, τs[τ_opt])
    push!(ts_vals, ts_opt)
    NS[counter] = Ns

    # create phase space vector for this embedding cycle
    Y_act = DelayEmbeddings.hcat_lagged_values(Y_act,
                                Ys[:,ts_vals[counter+1]],τ_vals[counter+1])
    vtree = KDTree(Y_act, metric)
    _, NNdist_new = all_neighbors(vtree, Y_act, 1:length(Y_act), 1, w)

    # compute FNN-statistic and store vals, while also
    # truncating distance-vector to the "new" length of the actual embedding vector
    fnns = fnn_embedding_cycle(view(NNdist_old, 1:length(Y_act)), NNdist_new, r2)
    push!(FNNs,fnns)

    return Y_act, NNdist_new
end

"""
This function chooses the optimal τ value as the 1st minimum of all the
N-statistics 1st minima. It returns the index of this 1st minimum as well as the
time series number corresponding to that 1st minimum of all 1st minima. In this
first embedding cycle it also returns the time series number, which act as Y_act.
"""
function choose_optimal_tau1_garcia(Ns::Array{T, 2}, M::Int) where {T}
    NN = oldsize(Ns,2)
    min_idx = zeros(Int, NN)
    for i = 1:NN
        # determine optimal tau value and minimum vals from all N-statistic's
        minimas = findlocalminima(Ns[:,i])
        min_idx[i] = minimas[1]
    end
    # take the 1st of all 1st minimas
    _, idx = findmin(min_idx)
    # determine the first time series and the according second one
    a = idx/M
    ts_1 = Int(ceil(a))
    ts_2 = idx-M*(ts_1-1)
    return min_idx[idx], ts_1, ts_2
end

"""
This function chooses the optimal τ value as the 1st minimum of the 1st minima
of the N-statistics. It returns the index of this 1st minimum as well as the
time series number corresponding to that 1st minimum of the 1st minima.
"""
function choose_optimal_tau2_garcia(Ns::Array{T, 2}) where {T}
    NN = oldsize(Ns,2)
    min_idx = zeros(Int, NN)
    for i = 1:NN
        # determine optimal tau value and minimum vals from all N-statistic's
        minimas = findlocalminima(Ns[:,i])
        min_idx[i] = minimas[1]
    end
    # take the 1st of all 1st minimas
    _, idx = findmin(min_idx)
    return min_idx[idx], idx
end


function garcia_almeida_break_criterion(FNNs, counter, fnn_thres, max_num_of_cycles)
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
