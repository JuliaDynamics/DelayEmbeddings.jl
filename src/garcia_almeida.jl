using Neighborhood
using NearestNeighbors
using StatsBase
using Distances

export garcia_almeida_embed
export garcia_embedding_cycle


"""
    garcia_almeida_embed(s; kwargs...) → Y, N, FNN
...Garcia & Almeida [^Garcia2005a]
(Nearest neighbor embedding with different time delays),
Garcia & Almeida [^Garcia2005b] (Multivariate phase space reconstruction by
nearest neighbor embedding with different time delays)
for a phase space trajectory `Y` (timeseries or `Dataset`). ...

## Keyword arguments

* `SampleSize = .5`: Number of considered fiducial points v as a fraction of
  input phase space trajectory `Y`'s length, in order to average the conditional
  variances and neighborhood sizes (read algorithm description) to produce `L`.
* `K = 3`: the amount of nearest neighbors considered, in order to compute σ_k^2
  (read algorithm description).
  If given a vector, minimum result over all `k ∈ K` is returned.
* `metric = Euclidean()`: metric used for finding nearest neigbhors in the input
  phase space trajectory `Y.
* `w = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `Tw = 40`: The time horizon (in sampling units) up to which E_k^2 gets computed
  and averaged over (read algorithm description).

## Description
tbd.

[^Garcia2005a]: Garcia, S. P., Almeida, J. S. (2005). [Nearest neighbor embedding with different time delays. Physical Review E 71, 037204](https://doi.org/10.1103/PhysRevE.71.037204).
[^Garcia2005b]: Garcia, S. P., Almeida, J. S. (2005). [Multivariate phase space reconstruction by nearest neighbor embedding with different time delays. Physical Review E 72, 027205](https://doi.org/10.1103/PhysRevE.72.027205).
"""

function garcia_almeida_embed()
    ####
    ####
end




## core: Garcia-Almeida-method for one arbitrary embedding cycle

"""
    garcia_embedding_cycle(Y,s; kwargs...) → N, d_E1
Performs one embedding cycle according to the method proposed in [^Garcia2005a]
for a given input `Y`(phase space trajectory or time series) and `s`
(timeseries or `Dataset`). Returns the proposed N-Statistic `N` and all nearest
neighbor distances 'd_E1' for each point of the input phase space
trajectory `Y`. Note that `Y` is a single time series in case of the first
embedding cycle.

## Keyword arguments
* `τ_max = 50`: Maximum of the τ-values the algorithm checkes (0:τ_max).
* `r = 10`: The threshold, which defines the factor of tolerable stretching for
  the d_E1-statistic (see algorithm description).
* `T = 1`: The forward time step (in sampling units) in order to compute the
  d_E2-statistic (see algorithm description). Note that in the paper this is
  not a free parameter and always set to `T=1`.
* `w = 0`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors. Note that in the paper this is not
  a free parameter and always `w=0`.
* `metric = Euclidean()`: metric used for finding nearest neigbhors in the input
  phase space trajectory `Y`.

## Description
tbd

[^Garcia2005a]: Garcia, S. P., Almeida, J. S. (2005). [Nearest neighbor embedding with different time delays. Physical Review E 71, 037204](https://doi.org/10.1103/PhysRevE.71.037204).
"""

function garcia_embedding_cycle(Y::AbstractDataset, s::AbstractArray; τ_max::Int = 50 , r::Int = 10,
    T::Int = 1, w::Int = 1, metric = Euclidean())

    # preallocation of output
    N_stat = zeros(τ_max+1)
    NN_distances = [AbstractArray[] for i=1, j=1:τ_max+1]

    for (i,τ) in enumerate(0:τ_max)
        # build temporary embedding matrix Y_temp
        Y_temp = embed2(Y,s,τ)
        NN = length(Y_temp)     # length of the temporary phase space vector
        N = NN-T               # accepted length w.r.t. the time horizon `T

        vtree = KDTree(Y_temp[1:N], metric) # tree for input data
        # nearest neighbors (d_E1)
        allNNidxs, d_E1 = all_neighbors(vtree, Y_temp[1:N], 1:N, 1, w)
        # save d_E1-statistic
        push!(NN_distances[i],hcat(d_E1...))

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
    embed2(Y,s,τ) → Dataset
takes a phase space trajectory 'Y' containing all phase space vectors, a
univariate time series 's' and a delay value 'τ' as input. embed2 then expands
the input phase space vectors by an additional component consisting of the
τ-shifted values of the input time series `s` and outputs the new phase space
trajectory `Y_new`.
"""

function embed2(Y,s,τ::Int)
    N = size(Y,1)   # length of input trajectory
    NN = size(Y,2)  # dimensionality of input trajectory
    M = N - τ       # length of output trajectory

    # preallocation
    Y_new = zeros(M,NN+1)
    # fill vector up until index M
    for i = 1:NN
        Y_new[:,i] = Y[1:M,i]
    end
    # add lagged component of s
    Y_new[:,NN+1] = s[1+τ:N]

    return Dataset(Y_new)

end