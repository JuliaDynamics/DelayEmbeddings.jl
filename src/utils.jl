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

@deprecate localextrema findlocalextrema

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


"""
    hcat_lagged_values(Y, s::Vector, τ::Int) -> Z
Add the `τ` lagged values of the timeseries `s` as additional component to `Y`
(`Vector` or `Dataset`), in order to form a higher embedded
dataset `Z`. The dimensionality of `Z` is thus equal to that of `Y` + 1.
"""
function hcat_lagged_values(Y::Dataset{D,T}, s::Vector{T}, τ::Int) where {D, T<:Real}
    N = length(Y)
    @assert N ≤ length(s)
    M = N - τ
    data = Vector{SVector{D+1, T}}(undef, M)
    @inbounds for i in 1:M
        data[i] = SVector{D+1, T}(Y[i]..., s[i+τ])
    end
    return Dataset{D+1, T}(data)
end

function hcat_lagged_values(Y::Vector{T}, s::Vector{T}, τ::Int) where {T<:Real}
    N = length(Y)
    @assert N ≤ length(s)
    M = N - τ
    return Dataset(view(Y, 1:M), view(s, τ+1:N))
end
