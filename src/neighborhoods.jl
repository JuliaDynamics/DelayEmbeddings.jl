using NearestNeighbors, StaticArrays
using Distances: Euclidean, Metric
import NearestNeighbors: KDTree

export AbstractNeighborhood
export FixedMassNeighborhood, FixedSizeNeighborhood
export neighborhood, KDTree

#####################################################################################
#                              Neighborhoods n stuff                                #
#####################################################################################
"""
    AbstractNeighborhood
Supertype of methods for deciding the neighborhood of points for a given point.

Concrete subtypes:
* `FixedMassNeighborhood(K::Int)` :
  The neighborhood of a point consists of the `K`
  nearest neighbors of the point.
* `FixedSizeNeighborhood(ε::Real)` :
  The neighborhood of a point consists of all
  neighbors that have distance < `ε` from the point.

See [`neighborhood`](@ref) for more.
"""
abstract type AbstractNeighborhood end

struct FixedMassNeighborhood <: AbstractNeighborhood
    K::Int
end
FixedMassNeighborhood() = FixedMassNeighborhood(1)

struct FixedSizeNeighborhood <: AbstractNeighborhood
    ε::Float64
end
FixedSizeNeighborhood() = FixedSizeNeighborhood(0.01)

"""
    neighborhood(point, tree, ntype)
    neighborhood(point, tree, ntype, n::Int, w::Int = 1)

Return a vector of indices which are the neighborhood of `point` in some
`data`, where the `tree` was created using `tree = KDTree(data [, metric])`.
The `ntype` is the type of neighborhood and can be any subtype
of [`AbstractNeighborhood`](@ref).

Use the second method when the `point` belongs in the data,
i.e. `point = data[n]`. Then `w` stands for the Theiler window (positive integer).
Only points that have index
`abs(i - n) ≥ w` are returned as a neighborhood, to exclude close temporal neighbors.
The default `w=1` is the case of excluding the `point` itself.

## References

`neighborhood` simply interfaces the functions
`NearestNeighbors.knn` and `inrange` from
[NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) by using
the argument `ntype`.
"""
function neighborhood(point::AbstractVector, tree,
                      ntype::FixedMassNeighborhood, n::Int, w::Int = 1)
    idxs, = NearestNeighbors.knn(tree, point, ntype.K, false, i -> abs(i-n) < w)
    return idxs
end
function neighborhood(point::AbstractVector, tree, ntype::FixedMassNeighborhood)
    idxs, = NearestNeighbors.knn(tree, point, ntype.K, false)
    return idxs
end

function neighborhood(point::AbstractVector, tree,
                      ntype::FixedSizeNeighborhood, n::Int, w::Int = 1)
    idxs = NearestNeighbors.inrange(tree, point, ntype.ε)
    filter!((el) -> abs(el - n) ≥ w, idxs)
    return idxs
end
function neighborhood(point::AbstractVector, tree, ntype::FixedSizeNeighborhood)
    idxs = NearestNeighbors.inrange(tree, point, ntype.ε)
    return idxs
end

KDTree(D::AbstractDataset, metric::Metric = Euclidean()) = KDTree(D.data, metric)


# TODO: This must use the new Neighborhood.jl
# TODO: Function returns 0-indices and -Inf values, when w is too high compared
#       to the length of vtree. This should be fixed, i.e. throw an error
"""
    all_neighbors(vtree, vs, ns, K, w)
Return the `maximum(K)`-th nearest neighbors for all input points `vs`, with indices `ns` in
original data, while respecting the theiler window `w`.
"""
function all_neighbors(vtree, vs, ns, K, w)
    k, sortres, N = maximum(K), true, length(vs)
    dists = [Vector{eltype(vs[1])}(undef, k) for _ in 1:N]
    idxs = [Vector{Int}(undef, k) for _ in 1:N]
    for i in 1:N
        # The skip predicate also skips the point itself for w ≥ 0
        skip = j -> ns[i] - w ≤ j ≤ ns[i] + w
        NearestNeighbors.knn_point!(vtree, vs[i], sortres, dists[i], idxs[i], skip)
    end
    return idxs, dists
end
