using NearestNeighbors
using Distances: chebyshev
using SpecialFunctions: digamma
using StatsBase: autocor
using KernelDensity

export mutinfo, mutinfo_delaycurve
#####################################################################################
#                                Mutual Information                                 #
#####################################################################################
"""
    mutinfo(k, X1, X2[, ..., Xm]) -> MI

Calculate the mutual information `MI` of the given vectors
`X1, X2, ...`, using `k` nearest-neighbors.

The method follows the second algorithm ``I^{(2)}`` outlined by Kraskov in [1].

## References
[1] : A. Kraskov *et al.*, [Phys. Rev. E **69**, pp 066138 (2004)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138)

## Performance Notes
This functin gets very slow for large `k`.

See also [`estimate_delay`](@ref) and [`mutinfo_delaycurve`](@ref).
"""
function mutinfo(k, Xm::Vararg{<:AbstractVector,M}) where M
    @assert M > 1
    @assert (size.(Xm,1) .== size(Xm[1],1)) |> prod
    k += 1
    N = size(Xm[1],1)
    invN = 1/N

    d = StateSpaceSet(Xm...)
    tree = KDTree(d.data, Chebyshev())

    n_x_m = zeros(M)

    Xm_sp = zeros(Int, N, M)
    Xm_revsp = zeros(Int, N, M)
    for m in 1:M
        Xm_sp[:,m] .= sortperm(Xm[m]; alg=QuickSort)
        Xm_revsp[:,m] .= sortperm(Xm_sp[:,m]; alg=QuickSort)
    end

    I = digamma(k) - (M-1)/k + (M-1)*digamma(N)

    nns = (x = knn(tree, d.data, k)[1]; [ind[1] for ind in x])

    I_itr = zeros(M)
    # Makes more sense computationally to loop over N rather than M
    for i in 1:N
        ϵ = abs.(d[nns[i]] - d[i])./2

        for m in 1:M # this loop takes 8% of time
            hb = lb = Xm_revsp[i,m]
            while abs(Xm[m][Xm_sp[hb,m]] - Xm[m][i]) <= ϵ[m] && hb < N
                hb += 1
            end
            while abs(Xm[m][Xm_sp[lb,m]] - Xm[m][i]) <= ϵ[m] && lb > 1
                lb -= 1
            end
            n_x_m[m] = hb - lb
        end

        I_itr .+= digamma.(n_x_m)
    end

    I_itr .*= invN

    I -= sum(I_itr)

    return max(0, I)
end

"""
    mutinfo_delaycurve(x; maxtau=100, k=1)

Return the [`mutinfo`](@ref) between `x` and itself for delays of `1:maxtau`.
"""
function mutinfo_delaycurve(X::AbstractVector; maxtau=100, k=1)
    I = zeros(maxtau)

    @views for τ in 1:maxtau
        I[τ] = mutinfo(k, X[1:end-τ],X[τ+1:end])
    end

    return I
end

### Fraser's and Swinney's algorithm ###

# Minimum number of points in the margin of a node
const minnodesize = 6

"""
    FSNode

Tree structure for 4^m partitions of a plane:
`low`, `high` are the corners of a node of the partition.
`middle` is the middle point that defines the subpartition.
`children` contains the four sub-partitions.
"""
struct FSNode{T}
    low::Tuple{T,T}
    high::Tuple{T,T}
    middle::Tuple{T,T}
    children::Array{FSNode,1}
end

# Recursive creation of partitions
FSNode(low::Int, high::Int) = FSNode((low,low), (high,high))
function FSNode(low::Tuple{T,T}, high::Tuple{T,T}) where {T}
    half = [div(high[1]-low[1], 2), div(high[2]-low[2],2)]
    middle = (low[1]+half[1], low[2]+half[2])
    node = FSNode(low, high, middle, FSNode[])
    if maximum(half) > minnodesize # or another condition
        append!(node.children,
                [FSNode((low[1],     low[2]),      (middle[1],middle[2])),
                 FSNode((middle[1]+1,low[2]),      (high[1],  middle[2])),
                 FSNode((low[1],     middle[2]+1), (middle[1],high[2])),
                 FSNode((middle[1]+1,middle[2]+1), (high[1],  high[2]))])
    end
    return node
end

"""
    mi_fs(s,τs)

Compute the mutual information between the series `s` and various
images of the same series delayed by `τ` ∈ `τs`, according to
Fraser's and Swinney's algorithm [1].

## References:
[1]: Fraser A.M. & Swinney H.L. "Independent coordinates for strange attractors
from mutual information" *Phys. Rev. A 33*(2), 1986, 1134:1140.
"""
function mi_fs(s::AbstractVector,τs)
    n = length(s)
    nτ = n-maximum(τs)       # trim the series for the maximum delay
    tree = FSNode(1, nτ)      # tree structure of the subpartitions
    perm = sortperm(s[1:nτ])  # order of the original series
    mi = zeros(length(τs))
    for (i,τ) in enumerate(τs)
        indices = sortperm(sortperm(s[τ+1:n])) # rank values of the delayed image of `s`
        mi[i] = _recursive_mi(tree, indices[perm])/nτ - log2(nτ) # eq. (19) of [1]
    end
    return mi
end

# Recursive implementation of equations (20a) and (20b) in [1]
function _recursive_mi(node::FSNode, s)
    # Get view of the `s` within the first dimension of the node
    v = @view s[node.low[1]:node.high[1]]
    if isempty(node.children) # terminal node, no possible substructure
        # Count points in the range of the second dimension
        n = _count(v, node.low[2], node.high[2])
        mi_value = (n==0) ? 0 : n*log2(n)    # eq. (20a) in [1]
    else
        # Split the view in two halves of the first dimension
        # and count point in the two subranges of the second
        n1 = node.middle[1] - node.low[1] + 1
        vhalf = @view v[1:n1]
        n11, n21 = _count(vhalf, node.low[2], node.middle[2], node.high[2])
        vhalf = @view v[n1+1:end]
        n12, n22 = _count(vhalf, node.low[2], node.middle[2], node.high[2])
        n = n11+n12+n21+n22
        if _homogeneous(n11,n12,n21,n22) # no substructure (homogeneous table)
            mi_value = n*log2(n) # eq. (20a) in [1]
        else
            mi_value = 2.0*n     # eq. (20b) in [1]
            for (i, ni) in enumerate((n11,n12,n21,n22))
                (ni > 0) && (mi_value += _recursive_mi(node.children[i], s))
            end
        end
    end
    return mi_value
end

# count values of `s` inside `low:high`
function _count(s, low, high)
    n = 0
    @inbounds for i in s
        (low <= i <= high) && (n+= 1)
    end
    return n
end
# count values of `s` inside `low:middle` and `low:high`
function _count(s, low, middle, high)
    n1 = 0
    n2 = 0
    @inbounds for i in s
        if low <= i
            if i <= middle
                n1 += 1
            elseif i <= high
                n2 += 1
            end
        end
    end
    return n1, n2
end

# homogeneity test - equation (22) in [1]
function _homogeneous(n11,n12,n21,n22)
    n = n11+n12+n21+n22
    nh = n/4
    χ2 = 4/3/n*((n11-nh)^2+(n12-nh)^2+(n21-nh)^2+(n22-nh)^2)
    return χ2 < 1.547
end


function ami_kde(x::AbstractVector{T}, τs::AbstractVector{Int}) where {T}
    n = length(x)
    np = min(2048, prevpow(2, n))
    dx = kde(x, npoints=np)
    #
    tails_width = 4.0*KernelDensity.default_bandwidth(x)
    tp = round(Int, np*tails_width/(dx.x[end]-dx.x[1]))
    #
    # dx.density .*= step(dx.x)
    px = @view dx.density[tp+1:end-tp]
    hx = -sum(px.*log2.(px))*step(dx.x)
    ami = zeros(T, length(τs))
    #
    np2 = min(256, 2^floor(Int, log2(n)))
    tp = div(tp*np2,np)
    for (i,τ) in enumerate(τs)
        hxy = zero(T)
        dxy = kde((x[1:n-τ], x[τ+1:n]), npoints=(np2,np2))
        # dxy.density .*= (step(dxy.x) * step(dxy.y))
        pxy = @view dxy.density[tp+1:end-tp, tp+1:end-tp]
        for p in pxy
            if p ≉ 0
                hxy -= p*log2(p)
            end
        end
        hxy *= (step(dxy.x)*step(dxy.y))
        mi[i] = 2hx - hxy
    end
    return ami
end



