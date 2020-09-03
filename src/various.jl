#####################################################################################
#                                Minima Maxima                                      #
#####################################################################################
export findlocalminima

"""
    findlocalminima(s)
Return the indices of the local minima the timeseries `s`. If none exist,
return the index of the minimum (as a vector).
Starts of plateaus are also considered local minima.
"""
function findlocalminima(s::Vector{T})::Vector{Int} where {T}
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


#####################################################################################
#                                Pairwse Distance                                   #
#####################################################################################
using NearestNeighbors, StaticArrays, LinearAlgebra
export min_pairwise_distance, orthonormal

# min_pairwise_distance contributed by Kristoffer Carlsson
"""
    min_pairwise_distance(data) -> (min_pair, min_dist)
Calculate the minimum pairwise distance in the data (`Matrix`, `Vector{Vector}` or
`Dataset`). Return the index pair
of the datapoints that have the minimum distance, as well as its value.
"""
function min_pairwise_distance(cts::AbstractMatrix)
    if size(cts, 1) > size(cts, 2)
        error("Points must be close (transpose the Matrix)")
    end
    tree = KDTree(cts)
    min_d = Inf
    min_pair = (0, 0)
    for p in 1:size(cts, 2)
        inds, dists = NearestNeighbors.knn(tree, view(cts, :, p), 1, false, i -> i == p)
        ind, dist = inds[1], dists[1]
        if dist < min_d
            min_d = dist
            min_pair = (p, ind)
        end
    end
    return min_pair, min_d
end

min_pairwise_distance(d::AbstractDataset) = min_pairwise_distance(d.data)

function min_pairwise_distance(
    pts::Vector{SVector{D,T}}) where {D,T<:Real}
    tree = KDTree(pts)
    min_d = eltype(pts[1])(Inf)
    min_pair = (0, 0)
    for p in 1:length(pts)
        inds, dists = NearestNeighbors.knn(tree, pts[p], 1, false, i -> i == p)
        ind, dist = inds[1], dists[1]
        if dist < min_d
            min_d = dist
            min_pair = (p, ind)
        end
    end
    return min_pair, min_d
end

#####################################################################################
#                                Conversions                                        #
#####################################################################################
to_matrix(a::AbstractVector{<:AbstractVector}) = cat(2, a...)
to_matrix(a::AbstractMatrix) = a
function to_Smatrix(m)
    M = to_matrix(m)
    a, b = size(M)
    return SMatrix{a, b}(M)
end
to_vectorSvector(a::AbstractVector{<:SVector}) = a
function to_vectorSvector(a::AbstractMatrix)
    S = eltype(a)
    D, k = size(a)
    ws = Vector{SVector{D, S}}(k)
    for i in 1:k
        ws[i] = SVector{D, S}(a[:, i])
    end
    return ws
end

"""
    orthonormal(D, k) -> ws
Return a matrix `ws` with `k` columns, each being
an `D`-dimensional orthonormal vector.

Always returns `SMatrix` for stability reasons.
"""
function orthonormal end

@inline function orthonormal(D::Int, k::Int)
    k > D && throw(ArgumentError("k must be ≤ D"))
    q = qr(rand(SMatrix{D, k})).Q
end
