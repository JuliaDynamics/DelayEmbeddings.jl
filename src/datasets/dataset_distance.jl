export dataset_distance, Hausdorff
###########################################################################################
# Dataset distance
###########################################################################################
"""
    dataset_distance(dataset1, dataset2 [, method])
Calculate a distance between two `AbstractDatasets`,
i.e., a distance defined between sets of points, as dictated by `method`.
`method` defaults to `Euclidean()`.

## Description
If `method isa Metric` from Distances.jl, then the distance is the minimum
distance of all the distances from a point in one set to the closest point in
the other set. The distance is calculated with the given metric.

Else, `method` can be [`Hausdorff`](@ref) (a name provided by this module).
"""
function dataset_distance(d1::AbstractDataset, d2, metric::Metric = Euclidean())
    # this is the tree-based method
    # TODO: Check whether the pairwise method leads to better performance
    tree = KDTree(d2, metric)
    dist, idx, ε = [Inf], [0], Inf
    for p in d1 # iterate over all points of attractor
        Neighborhood.NearestNeighbors.knn_point!(
            tree, p, false, dist, idx, Neighborhood.alwaysfalse
        )
        @inbounds dist[1] < ε && (ε = dist[1])
    end
    return ε
end

"""
    Hausdorff(metric = Euclidean())
A type used in specifying dataset distances in [`dataset_distance`](@ref).
The [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance) is the
greatest of all the distances from a point in one set to the closest point in the other set.
The distance is calculated with the metric given to `Hausdorff` which defaults to Euclidean.
"""
struct Hausdorff{M<:Metric}
    metric::M
end
Hausdorff() = Hausdorff(Euclidean())

function dataset_distance(d1::AbstractDataset, d2, h::Hausdorff)
    tree1 = KDTree(d1, h.metric)
    tree2 = KDTree(d2, h.metric)
    # This yields the minimum distance between each point
    _, vec_of_distances12 = bulksearch(tree1, vec(d2), NeighborNumber(1))
    _, vec_of_distances21 = bulksearch(tree2, vec(d1), NeighborNumber(1))
    # Cast distances in vector (they are vectors of vectors)
    vec_of_distances12 = reduce(vcat, vec_of_distances12)
    vec_of_distances21 = reduce(vcat, vec_of_distances21)
    return max(maximum(vec_of_distances12), maximum(vec_of_distances21))
end

###########################################################################################
# Sets of datasets distance
###########################################################################################
function datasets_sets_distances(a₊, a₋, metric::Metric = Euclidean())
    ids₊, ids₋ = keys(a₊), keys(a₋)
    # TODO: Deduce distance type instead of Float64
    distances = Dict{eltype(ids₊), Dict{eltype(ids₋), Float64}}()
    search_trees = Dict(k => KDTree(vec(att), metric) for (k, att) in pairs(a₋))
    for i in ids₊
        d = valtype(distances)()
        for j in ids₋
            # TODO: create and use `dataset_distance` function in delay embeddings.jl
            # TODO: Use KD-trees or `pairwise`
            d[j] = minimum(metric(x, y) for x ∈ a₊[i] for y ∈ a₋[j])
        end
        distances[i] = d
    end
    return distances
end

# code from attractors via proximity
# This is the non-allocating version, but not sure if it is faster
# than just passing in the entire dataset into a `bulksearch` call
# TODO: Test it.
dist, idx = [Inf], [0]
minε = Inf
for (k, A) in attractors
    for (m, tree) in search_trees
        k == m && continue
        for p in A # iterate over all points of attractor
            Neighborhood.NearestNeighbors.knn_point!(
                tree, p, false, dist, idx, Neighborhood.alwaysfalse
            )
            dist[1] < minε && (minε = dist[1])
        end
    end
end