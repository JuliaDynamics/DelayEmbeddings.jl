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
In this case there is an internal heuristic: if `length(dataset1)*length(dataset2) ≥ 1000`
the algorithm switches to a KDTree-based version, otherwise it uses brute force.
You can overwrite this by explicitly setting the `brute` boolean keyword
or calling `dataset_distance_brute` directly.

`method` can also be [`Hausdorff`](@ref) (a name provided by this module).
"""
function dataset_distance(d1, d2::AbstractDataset, metric::Metric = Euclidean();
    brute = length(d1)*length(d2) < 1000)
    if brute
        return dataset_distance_brute(d1, d2, metric)
    else
        tree = KDTree(d2, metric)
        return dataset_distance(d1, tree)
    end
end

function dataset_distance(d1, tree::KDTree)
    # Notice that it is faster to do a bulksearch of all points in `d1`
    # rather than use the internal inplace method `NearestNeighbors.knn_point!`.
    _, vec_of_ds = bulksearch(tree, d1, NeighborNumber(1))
    return minimum(vec_of_ds)[1]
    # For future benchmarking reasons, we leave the code here
    ε = eltype(d2)(Inf)
    dist, idx = [ε], [0]
    for p in d1 # iterate over all points of dataset
        Neighborhood.NearestNeighbors.knn_point!(
            tree, p, false, dist, idx, Neighborhood.alwaysfalse
        )
        @inbounds dist[1] < ε && (ε = dist[1])
    end
    return ε
end

function dataset_distance_brute(d1, d2::AbstractDataset, metric = Euclidean())
    ε = eltype(d2)(Inf)
    for x ∈ d1
        for y ∈ d2
            εnew = metric(x, y)
            εnew < ε && (ε = εnew)
        end
    end
    return ε
end


"""
    Hausdorff(metric = Euclidean())
A dataset distance that can be used in [`dataset_distance`](@ref).
The [Hausdorff distance](https://en.wikipedia.org/wiki/Hausdorff_distance) is the
greatest of all the distances from a point in one set to the closest point in the other set.
The distance is calculated with the metric given to `Hausdorff` which defaults to Euclidean.

`Hausdorff` is a proper metric in the space of sets of datasets.
"""
struct Hausdorff{M<:Metric}
    metric::M
end
Hausdorff() = Hausdorff(Euclidean())

function dataset_distance(d1::AbstractDataset, d2, h::Hausdorff,
        # trees given for a natural way to call this function in `datasets_sets_distances`
        tree1 = KDTree(d1, h.metric),
        tree2 = KDTree(d2, h.metric),
    )
    # This yields the minimum distance between each point
    _, vec_of_distances12 = bulksearch(tree1, vec(d2), NeighborNumber(1))
    _, vec_of_distances21 = bulksearch(tree2, vec(d1), NeighborNumber(1))
    # get max of min distances (they are vectors of length-1 vectors, hence the [1])
    max_d12 = maximum(vec_of_distances12)[1]
    max_d21 = maximum(vec_of_distances21)[1]
    return max(max_d12, max_d21)
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

The `metric/method` can be as in [`dataset_distance`](@ref), in which case
both sets must have equal-dimension datasets.
However, `method` can also be any arbitrary user function that takes as input
two datasets and returns any positive-definite number as their "distance".
"""
function datasets_sets_distances(a₊, a₋, method = Euclidean())
    ids₊, ids₋ = keys(a₊), keys(a₋)
    gettype = a -> eltype(first(values(a)))
    T = promote_type(gettype(a₊), gettype(a₋))
    distances = Dict{eltype(ids₊), Dict{eltype(ids₋), T}}()
    _datasets_sets_distances!(distances, a₊, a₋, method)
end

function _datasets_sets_distances!(distances, a₊, a₋, metric::Metric)
    @assert keytype(a₊) == keytype(a₋)
    search_trees = Dict(m => KDTree(vec(att), metric) for (m, att) in pairs(a₋))
    @inbounds for (k, A) in pairs(a₊)
        distances[k] = pairs(valtype(distances)())
        for (m, tree) in search_trees
            # Internal method of `dataset_distance` for non-brute way
            d = dataset_distance(A, tree)
            distances[k][m] = d
        end
    end
    return distances
end


function _datasets_sets_distances!(distances, a₊, a₋, method::Hausdorff)
    @assert keytype(a₊) == keytype(a₋)
    metric = method.metric
    trees₊ = Dict(m => KDTree(vec(att), metric) for (m, att) in pairs(a₊))
    trees₋ = Dict(m => KDTree(vec(att), metric) for (m, att) in pairs(a₋))
    @inbounds for (k, A) in pairs(a₊)
        distances[k] = pairs(valtype(distances)())
        tree1 = trees₊[k]
        for (m, tree2) in trees₋
            # Internal method of `dataset_distance` for non-brute way
            d = dataset_distance(A, a₋[m], method, tree1, tree2)
            distances[k][m] = d
        end
    end
    return distances
end

function _datasets_sets_distances!(distances, a₊, a₋, f::Function)
    @inbounds for (k, A) in pairs(a₊)
        distances[k] = pairs(valtype(distances)())
        for (m, B) in pairs(a₋)
            distances[k][m] = f(A, B)
        end
    end
    return distances
end