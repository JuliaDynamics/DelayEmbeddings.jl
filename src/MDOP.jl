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
    beta_statistic(Y::Dataset; kwargs...) → β
Compute the L-statistic `L` for input dataset `Y` according to Uzal et al.[^Nichkawde2013], based on
theoretical arguments on noise amplification, the complexity of the
reconstructed attractor and a direct measure of local stretch which constitutes
an irrelevance measure. It serves as a cost function of a phase space
trajectory/embedding and therefore allows to estimate a "goodness of a
embedding" and also to choose proper embedding parameters, while minimizing
`L` over the parameter space.

## Keyword arguments

* `samplesize = 0.5`: Number of considered fiducial points v as a fraction of
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

[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
"""

function beta_statistic()
    # tbd

end
