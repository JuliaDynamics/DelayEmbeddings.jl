export findlocalminima, findlocalextrema

"""
    findlocalminima(s)
Return the indices of the local minima the timeseries `s`. If none exist,
return the index of the minimum (as a vector).
Starts of plateaus are also considered local minima.
"""
function findlocalminima(s::Vector{<:Real})::Vector{Int}
    minimas = Int[]
    N = length(s)
    flag = false
    first_point = 0
    for i = 2:N-1
        if s[i-1] > s[i] && s[i+1] > s[i]
            flag = false
            push!(minimas, i)
        end
        # handling constant values
        if flag
            if s[i+1] > s[first_point]
                flag = false
                push!(minimas, first_point)
            elseif s[i+1] < s[first_point]
                flag = false
            end
        end
        if s[i-1] > s[i] && s[i+1] == s[i]
            flag = true
            first_point = i
        end
    end
    # make sure there is no empty vector returned
    if isempty(minimas)
        _, mini = findmin(s)
        return [mini]
    else
        return minimas
    end
end


"""
    findlocalextrema(y) -> max_ind, min_ind
Find the local extrema of given array `y`, by scanning point-by-point. Return the
indices of the maxima (`max_ind`) and the indices of the minima (`min_ind`).
"""
function findlocalextrema(y)
    @inbounds begin
        l = length(y)
        i = 1
        maxargs = Int[]
        minargs = Int[]
        if y[1] > y[2]
            push!(maxargs, 1)
        elseif y[1] < y[2]
            push!(minargs, 1)
        end

        for i in 2:l-1
            left = i-1
            right = i+1
            if  y[left] < y[i] > y[right]
                push!(maxargs, i)
            elseif y[left] > y[i] < y[right]
                push!(minargs, i)
            end
        end

        if y[l] > y[l-1]
            push!(maxargs, l)
        elseif y[l] < y[l-1]
            push!(minargs, l)
        end
        return maxargs, minargs
    end
end

"""
    all_neighbors(A::Dataset, stype, w = 0) → idxs, dists
    all_neighbors(vtree, vs, ns, K, w)
Return the `maximum(K)`-th nearest neighbors for all input points `vs`,
with indices `ns` in original data, while respecting the theiler window `w`.

This function is nothing more than a convinience call to `Neighborhood.bulksearch`.

It is an internal, convenience function.
"""
function all_neighbors(vtree, vs, ns, K, w)
    w ≥ length(vtree.data)-1 && error("Theiler window larger than the entire data span!")
    k = maximum(K)
    tw = Theiler(w, ns)
    idxs, dists = bulksearch(vtree, vs, NeighborNumber(k), tw)
end

function all_neighbors(A::AbstractDataset, stype, w::Int = 0)
    theiler = Theiler(w)
    tree = KDTree(A)
    idxs, dists = bulksearch(tree, A, stype, theiler)
end
