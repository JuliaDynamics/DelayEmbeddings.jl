using Neighborhood
using NearestNeighbors

knn = Neighborhood.knn

export uzal_cost

function uzal_cost()
    tree = KDTree(original_points)
    queries = all x-bars
    indxs = bulkisearch(tree, queries, NeighborNumber(k))
    for i in 1:length(queries)
        x_bar = queries[i]
        neighbors = indxs[i]
        # do stuff , e.g. calculate (13)
        approximate_conditional_variance()
    end
end

function approximate_conditional_variance()
    # code
end
