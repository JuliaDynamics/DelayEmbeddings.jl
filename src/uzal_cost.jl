using Neighborhood
using StatsBase
using Distances

export uzal_cost


"""
    uzal_cost(Y; kwargs...) → L
Compute the L-statistic `L` according to Uzal et al.[^Uzal2011],
for a phase space trajectory `Y` (`Dataset`). To convert an `Array` into Dataset
format, simply use `Dataset(Array)`. The L-statistic is
based on theoretical arguments on noise amplification, the complexity of the
reconstructed attractor and a direct measure of local stretch which constitutes
an irrelevance measure.

## Keyword arguments

* `samplesize = .5`: Number of considered fiducial points v as a fraction of
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
The `L`-statistic based on theoretical arguments on noise amplification, the
complexity of the reconstructed attractor and a direct measure of local stretch
which constitutes an irrelevance measure. Technically, it is the logarithm of
the product of `σ`-statistic and a normalization statistic `α`:

L = log10(σ*α)

The `σ`-statistic is computed as follows. `σ`=√`σ²` and `σ²`=`E²`/`ϵ²`.
`E²` approximates the conditional variance at each point in phase space and
for a time horizon `T`∈`Tw`, using `K` nearest neighbors. For each reference
point of the phase space trajectory, the neighborhood consists of the reference
point itself and its `K`+1 nearest neighbors. `E²` measures how strong
a neighborhood expands during `T` time steps. `E²` is averaged over many time
horizons `T`=1:`Tw`. Consequently, `ϵ²` is the size of the neighborhood at the
reference point itself and is defined as the mean pairwise distance of the
neighborhood. Finally, `σ²` gets averaged over a range of reference points on
the attractor, which is controlled by `samplesize`. This is just for performance
reasons and the most accurate result will obviously be gained when setting
`samplesize=1.0`

The `α`-statistic is a normalization factor, such that `σ`'s from different
reconstructions can be compared. `α²` is defined as the inverse of the sum of
the inverse of all `ϵ²`'s for all considered reference points.

[^Uzal2011]: Uzal, L. C., Grinblat, G. L., Verdes, P. F. (2011). [Optimal reconstruction of dynamical systems: A noise amplification approach. Physical Review E 84, 016223](https://doi.org/10.1103/PhysRevE.84.016223).
"""
function uzal_cost(Y::AbstractDataset; Tw::Int = 40, K::Int = 3, w::Int = 1, samplesize::Float64 = .5,
    metric = Euclidean())

    # select a random phase space vector sample according to input `SampleSize
    NN = length(Y)-Tw;
    NNN = floor(Int,samplesize*NN)
    ns = sample(1:NN, NNN; replace=false) # the fiducial point indices
    vs = Y[ns] # the fiducial points in the data set

    # tree for input data
    vtree = KDTree(Y[1:end-Tw], metric)
    allNNidxs, allNNdist = all_neighbors(vtree, vs, ns, K, w)

    # preallocation
    ϵ² = zeros(NNN)             # neighborhood size
    E²_avrg = zeros(NNN)        # averaged conditional variance

    # loop over each fiducial point
    for (i,v) in enumerate(vs)
        NNidxs = allNNidxs[i] # indices of k nearest neighbors to v
        NNdist = allNNdist[i] # k nearest neighbor distances to v

        # construct neighborhood for this fiducial point
        neighborhood = zeros(K+1,size(Y,2))
        neighborhood[1,:] = v  # the fiducial point is included in the neighborhood
        neighborhood[2:K+1,:] = vcat(transpose(Y[NNidxs])...)

        # estimate size of the neighborhood
        pd = pairwise(metric,neighborhood, dims = 1)
        ϵ²[i] = (2/(K*(K+1))) * sum(pd.^2)  # Eq. 16

        # estimate E²[T]
        E² = zeros(Tw)   # preallocation
        # loop over the different time horizons
        for T = 1:Tw
            E²[T] = comp_Ek2(Y,ns[i],NNidxs,T,K,metric) # Eqs. 13 & 14
        end
        # Average E²[T] over all prediction horizons
        E²_avrg[i] = mean(E²)                   # Eq. 15

    end

    # compute the noise amplification σ²
    σ² = E²_avrg ./ ϵ²                          # Eq. 17

    # compute the averaged value of the noise amplification
    σ²_avrg = mean(σ²)                          # Eq. 18

    # compute α² for normalization
    α² = 1 / sum(ϵ².^(-1))                      # Eq. 21

    # compute the final L-Statistic
    L = log10(sqrt(σ²_avrg)*sqrt(α²))

    return L
end


"""
    comp_Ek2(Y,v,NNidxs,T,K,metric) → E²(T)
Returns the approximated conditional variance for a specific point in phase space
`ns` (index value) with its `K`-nearest neighbors, which indices are stored in
`NNidxs`, for a time horizon `T`. This corresponds to Eqs. 13 & 14 in [^Uzal2011].
The norm specified in input parameter `metric` is used for distance computations.
"""
function comp_Ek2(Y, ns::Int, NNidxs, T::Int, K::Int, metric)
    # determine neighborhood `T` time steps ahead
    ϵ_ball = zeros(K+1, size(Y,2)) # preallocation
    ϵ_ball[1,:] = Y[ns+T]
    ϵ_ball[2:K+1,:] = vcat(transpose(Y[NNidxs.+T])...)

    # compute center of mass
    u_k = sum(ϵ_ball,dims=1) ./ (K+1) # Eq. 14

    E²_sum = 0
    for j = 1:K+1
        E²_sum += (evaluate(metric,ϵ_ball[j,:],u_k))^2
    end
    E² = E²_sum / (K+1)         # Eq. 13
end
