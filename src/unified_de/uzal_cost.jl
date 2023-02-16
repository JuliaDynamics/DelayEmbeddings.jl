using Neighborhood
using Distances

export uzal_cost
export uzal_cost_local

"""
    uzal_cost(Y::StateSpaceSet; kwargs...) → L
Compute the L-statistic `L` for input dataset `Y` according to Uzal et al.[^Uzal2011], based on
theoretical arguments on noise amplification, the complexity of the
reconstructed attractor and a direct measure of local stretch which constitutes
an irrelevance measure. It serves as a cost function of a state space
trajectory/embedding and therefore allows to estimate a "goodness of a
embedding" and also to choose proper embedding parameters, while minimizing
`L` over the parameter space. For receiving the local cost function `L_local`
(for each point in state space - not averaged), use `uzal_cost_local(...)`.

## Keyword arguments

* `samplesize = 0.5`: Number of considered fiducial points v as a fraction of
  input state space trajectory `Y`'s length, in order to average the conditional
  variances and neighborhood sizes (read algorithm description) to produce `L`.
* `K = 3`: the amount of nearest neighbors considered, in order to compute σ_k^2
  (read algorithm description).
  If given a vector, minimum result over all `k ∈ K` is returned.
* `metric = Euclidean()`: metric used for finding nearest neigbhors in the input
  state space trajectory `Y.
* `w = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `Tw = 40`: The time horizon (in sampling units) up to which E_k^2 gets computed
  and averaged over (read algorithm description).

## Description
The `L`-statistic is based on theoretical arguments on noise amplification, the
complexity of the reconstructed attractor and a direct measure of local stretch
which constitutes an irrelevance measure. Technically, it is the logarithm of
the product of `σ`-statistic and a normalization statistic `α`:

L = log10(σ*α)

The `σ`-statistic is computed as follows. `σ = √σ² = √(E²/ϵ²)`.
`E²` approximates the conditional variance at each point in state space and
for a time horizon `T ∈ Tw`, using `K` nearest neighbors. For each reference
point of the state space trajectory, the neighborhood consists of the reference
point itself and its `K+1` nearest neighbors. `E²` measures how strong
a neighborhood expands during `T` time steps. `E²` is averaged over many time
horizons `T = 1:Tw`. Consequently, `ϵ²` is the size of the neighborhood at the
reference point itself and is defined as the mean pairwise distance of the
neighborhood. Finally, `σ²` gets averaged over a range of reference points on
the attractor, which is controlled by `samplesize`. This is just for performance
reasons and the most accurate result will obviously be gained when setting
`samplesize=1.0`

The `α`-statistic is a normalization factor, such that `σ`'s from different
embeddings can be compared. `α²` is defined as the inverse of the sum of
the inverse of all `ϵ²`'s for all considered reference points.

[^Uzal2011]: Uzal, L. C., Grinblat, G. L., Verdes, P. F. (2011). [Optimal reconstruction of dynamical systems: A noise amplification approach. Physical Review E 84, 016223](https://doi.org/10.1103/PhysRevE.84.016223).
"""
function uzal_cost(Y::AbstractStateSpaceSet{D, ET};
        Tw::Int = 40, K::Int = 3, w::Int = 1, samplesize::Real = 0.5,
        metric = Euclidean()
    ) where {D, ET}

    # select a random state space vector sample according to input samplesize
    NN = length(Y)-Tw;
    NNN = floor(Int, samplesize*NN)
    ns = sample(1:NN, NNN; replace=false) # the fiducial point indices

    vs = Y[ns] # the fiducial points in the data set

    vtree = KDTree(Y[1:end-Tw], metric)
    allNNidxs, allNNdist = all_neighbors(vtree, vs, ns, K, w)
    ϵ² = zeros(NNN)             # neighborhood size
    E²_avrg = zeros(NNN)        # averaged conditional variance
    E² = zeros(Tw)
    ϵ_ball = zeros(ET, K+1, D) # preallocation
    u_k = zeros(ET, D)

    # loop over each fiducial point
    for (i,v) in enumerate(vs)
        NNidxs = allNNidxs[i] # indices of k nearest neighbors to v
        # pairwise distance of fiducial points and `v`
        pdsqrd = fiducial_pairwise_dist_sqrd(view(Y.data, NNidxs), v, metric)
        ϵ²[i] = (2/(K*(K+1))) * pdsqrd  # Eq. 16
        # loop over the different time horizons
        for T = 1:Tw
            E²[T] = comp_Ek2!(ϵ_ball, u_k, Y, ns[i], NNidxs, T, K, metric) # Eqs. 13 & 14
        end
        # Average E²[T] over all prediction horizons
        E²_avrg[i] = mean(E²)                   # Eq. 15
    end
    σ² = E²_avrg ./ ϵ² # noise amplification σ², Eq. 17
    σ²_avrg = mean(σ²) # averaged value of the noise amplification, Eq. 18
    α² = 1 / mean(ϵ².^(-1)) # for normalization, Eq. 21
    L = log10(sqrt(σ²_avrg)*sqrt(α²))
end

function fiducial_pairwise_dist_sqrd(fiducials, v, metric)
    pd = zero(eltype(fiducials[1]))
    pd += evaluate(metric, v, v)^2
    for (i, v1) in enumerate(fiducials)
        pd += evaluate(metric, v1, v)^2
        for j in i+1:length(fiducials)
            @inbounds pd += evaluate(metric, v1, fiducials[j])^2
        end
    end
    return sum(pd)
end

"""
    comp_Ek2!(ϵ_ball,u_k,Y,v,NNidxs,T,K,metric) → E²(T)
Returns the approximated conditional variance for a specific point in state space
`ns` (index value) with its `K`-nearest neighbors, which indices are stored in
`NNidxs`, for a time horizon `T`. This corresponds to Eqs. 13 & 14 in [^Uzal2011].
The norm specified in input parameter `metric` is used for distance computations.
"""
function comp_Ek2!(ϵ_ball, u_k, Y, ns::Int, NNidxs, T::Int, K::Int, metric)
    # determine neighborhood `T` time steps ahead
    ϵ_ball[1, :] .= Y[ns+T]
    @inbounds for (i, j) in enumerate(NNidxs)
        ϵ_ball[i+1, :] .= Y[j + T]
    end

    # compute center of mass
    @inbounds for i in 1:size(Y)[2]; u_k[i] = sum(view(ϵ_ball, :, i))/(K+1); end # Eq. 14

    E²_sum = 0
    @inbounds for j = 1:K+1
        E²_sum += (evaluate(metric, view(ϵ_ball, j, :), u_k))^2
    end
    E² = E²_sum / (K+1)         # Eq. 13
end


"""
    uzal_cost_local(Y::StateSpaceSet; kwargs...) → L_local
Compute the local L-statistic `L_local` for input dataset `Y` according to
Uzal et al.[^Uzal2011]. The length of `L_local` is `length(Y)-Tw` and
denotes a value of the local cost-function to each of the points of the
state space trajectory.

In contrast to [`uzal_cost`](@ref) `σ²` here does not get averaged over all the
state space reference points on the attractor. Therefore, the mean of 'L_local'
is different to `L`, when calling `uzal_cost`, since the averaging is performed
before logarithmizing.

Keywords as in [`uzal_cost`](@ref).

[^Uzal2011]: Uzal, L. C., Grinblat, G. L., Verdes, P. F. (2011). [Optimal reconstruction of dynamical systems: A noise amplification approach. Physical Review E 84, 016223](https://doi.org/10.1103/PhysRevE.84.016223).
"""
function uzal_cost_local(Y::AbstractStateSpaceSet{D, ET};
        Tw::Int = 40, K::Int = 3, w::Int = 1, samplesize::Real = 0.5,
        metric = Euclidean()
    ) where {D, ET}

    # select a random state space vector sample according to input samplesize
    NN = length(Y)-Tw;
    NNN = floor(Int, samplesize*NN)
    ns = sample(1:NN, NNN; replace=false) # the fiducial point indices

    vs = Y[ns] # the fiducial points in the data set

    vtree = KDTree(Y[1:end-Tw], metric)
    allNNidxs, allNNdist = all_neighbors(vtree, vs, ns, K, w)
    ϵ² = zeros(NNN)             # neighborhood size
    E²_avrg = zeros(NNN)        # averaged conditional variance
    E² = zeros(Tw)
    ϵ_ball = zeros(ET, K+1, D) # preallocation
    u_k = zeros(ET, D)

    # loop over each fiducial point
    for (i,v) in enumerate(vs)
        NNidxs = allNNidxs[i] # indices of k nearest neighbors to v
        # pairwise distance of fiducial points and `v`
        pdsqrd = fiducial_pairwise_dist_sqrd(view(Y.data, NNidxs), v, metric)
        ϵ²[i] = (2/(K*(K+1))) * pdsqrd  # Eq. 16
        # loop over the different time horizons
        for T = 1:Tw
            E²[T] = comp_Ek2!(ϵ_ball, u_k, Y, ns[i], NNidxs, T, K, metric) # Eqs. 13 & 14
        end
        # Average E²[T] over all prediction horizons
        E²_avrg[i] = mean(E²)                   # Eq. 15
    end
    σ² = E²_avrg ./ ϵ² # noise amplification σ², Eq. 17
    α² = 1 / mean(ϵ².^(-1)) # for normalization, Eq. 21
    L_local = log10.(sqrt.(σ²).*sqrt(α²))
    return L_local
end
