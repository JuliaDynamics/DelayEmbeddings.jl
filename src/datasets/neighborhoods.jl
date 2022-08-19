#####################################################################################
#                   Neighborhood.jl Interface & convenience functions               #
#####################################################################################
using Neighborhood, Distances

export WithinRange, NeighborNumber
export Euclidean, Chebyshev, Cityblock

Neighborhood.KDTree(D::AbstractDataset, metric::Metric = Euclidean(); kwargs...) =
KDTree(D.data, metric; kwargs...)

# Convenience extensions for ::Dataset in bulksearches
for f ∈ (:bulkisearch, :bulksearch)
    for nt ∈ (:NeighborNumber, :WithinRange)
        @eval Neighborhood.$(f)(ss::KDTree, D::AbstractDataset, st::$nt, args...; kwargs...) =
        $(f)(ss, D.data, st, args...; kwargs...)
    end
end

#=
    all_neighbors(vtree, vs, ns, K, w)
Return the `maximum(K)`-th nearest neighbors for all input points `vs`,
with indices `ns` in original data, while respecting the theiler window `w`.

This function is nothing more than a convinience call to `Neighborhood.bulksearch`.

It is internal, convenience function.
=#
function all_neighbors(vtree, vs, ns, K, w)
    w ≥ length(vtree.data)-1 && error("Theiler window larger than the entire data span!")
    k = maximum(K)
    tw = Theiler(w, ns)
    idxs, dists = bulksearch(vtree, vs, NeighborNumber(k), tw)
end

"""
    all_neighbors(A::Dataset, stype, w = 0) → idxs, dists
Find the neighbors of all points in `A` using search type `stype` (either
[`NeighborNumber`](@ref) or [`WithinRange`](@ref)) and `w` the [Theiler window](@ref).

This function is nothing more than a convinience call to `Neighborhood.bulksearch`.
"""
function all_neighbors(A::AbstractDataset, stype, w::Int = 0)
    theiler = Theiler(w)
    tree = KDTree(A)
    idxs, dists = bulksearch(tree, A, stype, theiler)
end
