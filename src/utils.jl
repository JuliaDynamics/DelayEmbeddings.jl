export orthonormal, findlocalminima
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
    orthonormal([T,] D, k) -> ws
Return a matrix `ws` with `k` columns, each being
an `D`-dimensional orthonormal vector.

Default: returns SMatrix{D, k} if D*k < 100, otherwise Matrix
"""
function orthonormal end

orthonormal(D, k) = D*k < 100 ? orthonormal(SMatrix, D, k) : orthonormal(Matrix, D, k)

@inline function orthonormal(T::Type, D::Int, k::Int)
    k > D && throw(ArgumentError("k must be ≤ D"))
    if T == SMatrix
        q = qr(rand(SMatrix{D, k})).Q
    elseif T == Matrix
        q = Matrix(qr(rand(Float64, D, k)).Q)
    end
    q
end

"""
    hcat_lagged_values(Y, s::Vector, τ::Int) -> Z
Add the `τ` lagged values of the timeseries `s` as additional component to `Y`
(`Vector` or `Dataset`), in order to form a higher embedded
dataset `Z`. The dimensionality of `Z` is thus equal to that of `Y` + 1.
"""
function hcat_lagged_values(Y::AbstractDataset{D,T}, s::Vector{T}, τ::Int) where {D, T<:Real}
    N = length(Y)
    MM = length(s)
    @assert N ≤ MM

    MMM = MM - τ
    M = min(N, MMM)
    data = Vector{SVector{D+1, T}}(undef, M)
    @inbounds for i in 1:M
        data[i] = SVector{D+1, T}(Y[i]..., s[i+τ])
    end
    return Dataset{D+1, T}(data)
end

function hcat_lagged_values(Y::Vector{T}, s::Vector{T}, τ::Int) where {T<:Real}
    N = length(Y)
    MM = length(s)
    @assert N ≤ MM
    MMM = MM - τ
    M = min(N, MMM)
    return Dataset(view(Y, 1:M), view(s, τ+1:τ+M))
end
