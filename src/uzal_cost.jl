using Neighborhood
using NearestNeighbors

knn = Neighborhood.knn

export uzal_cost

function uzal_cost()
    tree = KDTree(original_points)
    indxs = bulksearch(tree, queries, NeighborNumber(k))
    approximate_conditional_variance()
end

function approximate_conditional_variance()
    # code
end
