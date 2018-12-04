using StatsBase: autocor
using KernelDensity
# using LsqFit: curve_fit
export estimate_delay

#####################################################################################
#                               Estimate Delay Times                                #
#####################################################################################
"""
    estimate_delay(s, method::String) -> τ

Estimate an optimal delay to be used in [`reconstruct`](@ref) or [`embed`](@ref).
Return the exponential decay time `τ` rounded to an integer.

The `method` can be one of the following:

* `"first_zero"` : find first delay at which the auto-correlation function becomes 0.
* `"first_min"` : return delay of first minimum of the auto-correlation function.
"""
function estimate_delay(x::AbstractVector, method::String; maxtau=100, k=1)
    method ∈ ("first_zero", "first_min") ||
        throw(ArgumentError("Unknown method"))

    if method=="first_zero"
        c = autocor(x, 0:length(x)÷10; demean=true)
        i = 1
        # Find 0 crossing:
        while c[i] > 0
            i += 1
            i == length(c) && break
        end
        return i

    elseif method=="first_min"
        c = autocor(x, 0:length(x)÷10, demean=true)
        i = 1
        # Find min crossing:
        while c[i+1] < c[i]
            i+= 1
            i == length(c)-1 && break
        end
        return i
    # elseif method=="exp_decay"
    #     c = autocor(x, demean=true)
    #     # Find exponential fit:
    #     τ = exponential_decay(c)
    #     return round(Int,τ)
    # elseif method=="mutual_inf"
    #     m = mutinfo(k, x, x)
    #     L = length(x)
    #     for i=1:maxtau
    #         n = mutinfo(k, view(x, 1:L-i), view(x, 1+i:L))
    #         n > m && return i
    #         m = n
    #     end
    end
end


# Here is the code that does exponential decay:
# """
#     localextrema(y) -> max_ind, min_ind
# Find the local extrema of given array `y`, by scanning point-by-point. Return the
# indices of the maxima (`max_ind`) and the indices of the minima (`min_ind`).
# """
# function localextrema end
# @inbounds function localextrema(y)
#     l = length(y)
#     i = 1
#     maxargs = Int[]
#     minargs = Int[]
#     if y[1] > y[2]
#         push!(maxargs, 1)
#     elseif y[1] < y[2]
#         push!(minargs, 1)
#     end
#
#     for i in 2:l-1
#         left = i-1
#         right = i+1
#         if  y[left] < y[i] > y[right]
#             push!(maxargs, i)
#         elseif y[left] > y[i] < y[right]
#             push!(minargs, i)
#         end
#     end
#
#     if y[l] > y[l-1]
#         push!(maxargs, l)
#     elseif y[l] < y[l-1]
#         push!(minargs, l)
#     end
#     return maxargs, minargs
# end

# function exponential_decay_extrema(c::AbstractVector)
#     ac = abs.(c)
#     ma, mi = localextrema(ac)
#     # ma start from 1 but correlation is expected to start from x=0
#     ydat = ac[ma]; xdat = ma .- 1
#     # Do curve fit from LsqFit
#     model(x, p) = @. exp(-x/p[1])
#     decay = curve_fit(model, xdat, ydat, [1.0]).param[1]
#     return decay
# end
#
# function exponential_decay(c::AbstractVector)
#     # Do curve fit from LsqFit
#     model(x, p) = @. exp(-x/p[1])
#     decay = curve_fit(model, 0:length(c)-1, abs.(c), [1.0]).param[1]
#     return decay
# end

## Average Mutual Information

"""
    ami(x, τs, ε)
"""
function ami_hist(x::AbstractVector{T}, τs::AbstractVector, ε) where {T}
    n = length(x)
    xs = sort(x)
    # Set up the histogram structure
    offset = ε - rem(xs[end]-xs[1], ε)
    start = xs[1]   - offset
    stop  = xs[end] + offset
    edges = start:ε:stop
    nbins = length(edges) - 1
    f = zeros(typeof(0.0), nbins)
    # Calculate the marginal entropy of `x`
    marginal_entropy = _marginalentropy!(f, xs, edges)
    # `perm` can be used to sort other series along ascending values of `x`
    nτ = n-maximum(τs)
    perm = sortperm(x[1:nτ])
    # `hist_x` contains the histogram of the trimmed `x`.
    hist_x = zeros(Int, nbins)
    _frequencies!(hist_x, x[perm], edges)
    # Calculate the joint entropy and the AMI for each `τ`
    ami_values = zeros(T, length(τs))
    for (i, τ) in enumerate(τs)
        xτ = view(x, τ+1:n)[perm] # delayed and reordered time series
        joint_entropy = _jointentropy!(f, xτ, edges, hist_x)
        ami_values[i] = (2*marginal_entropy - joint_entropy)
    end
    return ami_values
end

"""
    _frequencies!(f, x, edges)

Calculate a histogram of values of `x` along the bins defined by `edges`.
Both `x` and `edges` must be sorted ascendingly. The frequencies (counts)
of `x` in each bien will be stored in the pre-allocated vector `f`. 
""" 
function _frequencies!(f::AbstractVector{T}, x::AbstractVector, edges::AbstractVector) where {T} 
    # Initialize the array of frequencies to zero
    fill!(f, zero(T))
    n = length(x)
    nbins = length(edges) - 1
    b = 1 # start in the first bin
    for i = 1:n
        if b ≥ nbins # the last bin is filled with the rest of points
            f[end] += n-i+1
            break
        end
        # when `x[i]` goes after theupper limit of the current bin,
        # move to the next bin and repeat until `x[i]` is found 
        while x[i] > edges[b+1]
            b += 1
        end
        f[b] += 1 # add point to the current bin
    end
end

_binsize(edges::AbstractRange, i) = step(edges)
_binsize(edges::AbstractVector, i::Int) = -(edges[i] - edges[i+1]) 

"""
    _marginalentropy!(f, x, edges)
    
Calculate the entropy of the distribution of `x` in the bins defined by `edges`.
The vector `f` is used as a placeholder to pre-allocate the histogram.
`x` must be sorted ascendingly, and `edges` must be a `Range` object with
homogeneous spacing. 
"""
function _marginalentropy!(f::AbstractVector, x::AbstractVector, edges::AbstractVector)
    # Initialize values
    h = 0.0                     # `h` will contain the result
    _frequencies!(f, x, edges)  # store the values of the histogram in `f`
    # Update the entropy with the values of nonzero bins
    n = length(x)
    for (i,g) in enumerate(f)
        ε = _binsize(edges, i)
        (g != 0) && (h += -g/n*(log2(g/n) - log2(ε)))
    end
    return h
end


"""
    _jointentropy!(f, x, edges, bins0)
    
Calculate the jointy entropy of the distribution of `x` in the bins defined by
`edges` and pre-calculated `bins0` that refer to the order of the elements of `x`.
`x` must be sorted in such a way that all the points of the bin `(1,j)` are
contained in the first `bins0[1]` data points of `x`, the points of the bin `(2,j)
are contained in the following `bins[2]` data points, etc.
  
The vector `f` is used as a placeholder to pre-allocate the histogram.
"""
function _jointentropy!(f::AbstractVector, x::AbstractVector,
    edges::AbstractVector, bins0::AbstractVector{I}) where {I<:Int}

    # Initialize values
    h = 0.0                     # `h` will contain the result
    processed = 0               # number of points of `x` that have been used
    # Go through `bins0`
    n = length(x)
    for b in bins0
        xb = view(x, processed.+(1:b)) # fragment of `x` in the `b`-th bin
        processed += b
        _frequencies!(f, sort(xb), edges) # store the values of the histogram in `f`
        for (i,g) in enumerate(f)
            ε = _binsize(edges, i)
            (g != 0) && (h += -g/n*(log2(g/n) - 2log2(ε)))
        end
    end
    return h
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
        ami[i] = 2hx - hxy
    end
    return ami
end

# Histogram tree based on Fraser's and Sweeney's algorithm

mutable struct FSHistogram{T}
    data::AbstractVector{T}
    bins::AbstractVector{Int}
    edges::AbstractVector{T}
    entropy::Float64
end
struct FSNode
    tree::FSHistogram
    level::Int
    data::SubArray
end
FSNode(tree::FSHistogram) = FSNode(tree, 1, tree.data)
function FSHistogram(data::AbstractVector)
    data = sort(data)
    bins = Int[]
    edges = data[1:1]
    entropy = log2(data[end]-data[1])
    h = FSHistogram(data, bins, edges, entropy)
    bin0 = @view data[:]
    node = FSNode(h, 0, bin0)
    # _branch!(node)
    return h
end

function _branch!(node::FSNode)
    # `x` must be sorted 
    if _uniformtest(node.data)
        push!(node.tree.bins, length(node.data))
        push!(node.tree.edges, node.data[end])
    else
        n = length(node.data)
        half = div(n, 2)
        range_data = node.data[end] - node.data[1]
        ratio_left  = (node.data[half] - node.data[1])/range_data
        ratio_right = (node.data[end] - node.data[half+1])/range_data
        # update entropy
        if ratio_left ≈ 0 || ratio_right ≈ 0
            @warn "many coincident points; calculations may be biased."
            push!(node.tree.bins, length(node.data))
            push!(node.tree.edges, node.data[end])
            return
        end
        node.tree.entropy += 2.0^(-node.level)*(1 + log2(ratio_left*ratio_right)/2)
        # first half
        bin1 = @view node.data[1:half]
        node1 = FSNode(node.tree, node.level+1, bin1)
        _branch!(node1)
        # second half
        bin2 = @view node.data[half+1:end]
        node2 = FSNode(node.tree, node.level+1, bin2)
        _branch!(node2)
    end
    return
end

function _uniformtest(x) # `x` is assumed to be sorted 
    n = length(x)
    if n <  10
        return true
    else
        if n < 20
            nbins = 2
            critical_chisq = 2.706 # df=1, p=0.1
        else
            nbins = 4
            critical_chisq = 6.251 # df=3, p=0.1
        end
    end
    binwidth = (x[end]-x[1])/nbins
    expected = round(Int, n/nbins)
    chisq = 0.0
    # set up first iteration
    first_item = 1
    for i = 1:nbins
        next_first = searchsortedfirst(x, x[1]+i*binwidth; lt=≤)
        observed = next_first - first_item
        chisq += (observed-expected)^2/expected
        (next_first > n) && break # stop if arrived at the end of `x`
        first_item = next_first
    end
    return (chisq < critical_chisq)
end    





