export dataset_distance, Hausdorff, datasets_sets_distances
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
function dataset_distance(d1, d2::AbstractDataset, metric::Metric = Euclidean())
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
A dataset distance that can be used in [`dataset_distance`](@ref).
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
"""
    datasets_sets_distances(a₊, a₋ [, metric/method]) → distances
Calculate distances between sets of `Dataset`s. Here  `a₊, a₋` are containers of
`Dataset`s, and the returned distances are dictionaries of
of distances. Specifically, `distances[i][j]` is the distance of the dataset in
the `i` key of `a₊` to the `j` key of `a₋`. Notice that distances from `a₋` to
`a₊` are not computed at all (assumming symmetry in the distance function).

The `metric/method` is exactly as in [`dataset_distance`](@ref).
"""
function datasets_sets_distances(a₊, a₋, metric::Metric = Euclidean())
    @assert keytype(a₊) == keytype(a₋)
    ids₊, ids₋ = keys(a₊), keys(a₋)
    # TODO: Deduce distance type instead of Float64
    distances = Dict{eltype(ids₊), Dict{eltype(ids₋), Float64}}()
    # Brute force version:
    # for i in ids₊
    #     d = valtype(distances)()
    #     for j in ids₋
    #         # TODO: create and use `dataset_distance` function in delay embeddings.jl
    #         # TODO: Use KD-trees or `pairwise`
    #         d[j] = minimum(metric(x, y) for x ∈ a₊[i] for y ∈ a₋[j])
    #     end
    #     distances[i] = d
    # end
    # Non-allocating seacrh trees version
    search_trees = Dict(m => KDTree(vec(att), metric) for (m, att) in pairs(a₋))
    dist, idx = [Inf], [0]
    for (k, A) in pairs(a₊)
        distances[k] = valtype(distances)()
        for (m, tree) in search_trees
            # TODO: This should call `dataset_distance` and propagate the tree
            minε = Inf
            for p in A # iterate over all points of attractor
                Neighborhood.NearestNeighbors.knn_point!(
                    tree, p, false, dist, idx, Neighborhood.alwaysfalse
                )
                dist[1] < minε && (minε = dist[1])
            end
            # assume symmetry
            distances[k][m] = minε
        end
    end
    return distances
end
