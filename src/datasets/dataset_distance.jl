"""
    dataset_distance(d1, d2 [, method])
Calculate a distance between two `AbstractDatasets` `d1, d2`,
i.e., a distance defined between sets of points, as dictated by `method`.

If `method isa Metric` from Distances.jl, then the distance is the minimum
distance of distances between the points of the two datasets.
"""
function dataset_distance(d1::AbstractDataset, d2, metric::Metric = Euclidean())
    # this is the tree-based method
    # TODO: Check the pairwise method
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
