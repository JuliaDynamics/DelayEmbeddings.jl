using Neighborhood
using NearestNeighbors
using StatsBase

knn = Neighborhood.knn

export uzal_cost


"""
    uzal_cost(Y; kwargs...) → ⟨L⟩
Compute the (average) L-statistic `⟨ε★⟩` according to Uzal et al.[^Uzal2011]
(Optimal reconstruction of dynamical systems: A noise amplification approach),
for a phase space trajectory `Y` (timeseries or `Dataset`). The L-statistic is
based on theoretical arguments on noise amplification, the complexity of the
reconstructed attractor and a direct measure of local stretch which constitutes
an irrelevance measure. The returned result is a scalar.

## Keyword arguments

* `SampleSize = .5`: Number of considered fiducial points v as a fraction of input phase
  space trajectory `Y` length, in order to average L to produce `⟨L⟩`
* `K = 3`: the amount of nearest neighbors considered, in order to compute σ_k^2
  (read algorithm description).
  If given a vector, minimum result over all `k ∈ K` is returned.
* `metric = Chebyshev()`: metric used for finding nearest neigbhors in the input
  phase space trajectory `Y.
* `w = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.
* `T = 40`: The time horizon (in sampling units) up to which E_k^2 gets computed
  and averaged over (read algorithm description).

## Description
tbd.

[^Uzal2011]: Uzal, L. C., Grinblat, G. L., Verdes, P. F. (2011). [Optimal reconstruction of dynamical systems: A noise amplification approach. Physical Review E 84, 016223](https://doi.org/10.1103/PhysRevE.84.016223).
"""


function uzal_cost(Y; T::Int = 40, K::Int = 3, w::Int = 1, SampleSize::Float64 = .5,
    metric = Chebyshev())

    # select a random phase space vector sample according to
    NN = length(Y)-T;
    NNN = floor(Int,SampleSize*NN)
    data_sample = sample(1:NN, NNN; replace=false)

    # tree = KDTree(original_points)
    # queries = all x-bars
    # indxs = bulkisearch(tree, queries, NeighborNumber(k))
    # for i in 1:length(queries)
    #     x_bar = queries[i]
    #     neighbors = indxs[i]
    #     # do stuff , e.g. calculate (13)
    #     approximate_conditional_variance()
    # end
    display("hello world 333")
    return data_sample
end

# function approximate_conditional_variance()
#     # code
# end
