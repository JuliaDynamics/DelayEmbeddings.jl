using StatsBase: autocor
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
    mutualinformation(s, τs[; nbins, binwidth])
    
Calculate the mutual information between the time series `s` and its images
delayed by `τ` points for `τ` ∈ `τs`.

## Description:

The joint space of `s` and its `τ`-delayed image (`sτ`) is partitioned as a
rectangular grid, and the mutual information is computed from the joint and
marginal frequencies of `s` and `sτ` in the grid as defined in [1].
The mutual information values are returned in a vector of the same length
as `τs`.

If any of the optional keyword parameters is given, the grid will be a
homogeneous partition of the space where `s` and `sτ` are defined.
The margins of that partition will be divided in a number of bins equal
to `nbins`, such that the width of each bin will be `binwidth`, and the range
of nonzero values of `s` will be in the centre. If only of those two parameters
is given, the other will be automatically calculated to adjust the size of the
grid to the area where `s` and `sτ` are nonzero.

If no parameter is given, the space will be partitioned by a recursive bisection
algorithm based on the method given in [1].

Notice that the recursive method of [1] evaluates the joint frequencies of `s`
and `sτ` in each cell resulting from a partition, and stops when the data points
are uniformly distributed across the sub-partitions of the following levels.
For performance and stability reasons, the automatic partition method implemented
in this function is only used to divide the axes of the grid, using the marginal
frequencies of `s`.

## References:
[1]: Fraser A.M. & Swinney H.L. "Independent coordinates for strange attractors
from mutual information" *Phys. Rev. A 33*(2), 1986, 1134:1140.
"""
function mutualinformation(s::AbstractVector{T}, τs::AbstractVector{Int}; kwargs...) where {T}
    n = length(x)
    nτ = n-maximum(τs)
    perm = sortperm(s[1:nτ])
    # Choose partition method
    (bins, edges) = isempty(kwargs) ? _bisect(s[perm]) : _equalbins(s[perm]; kwargs...)
    f = zeros(length(bins))
    # Calculate the MI for each `τ`
    mi_values = zeros(length(τs))
    for (i, τ) ∈ enumerate(τs)
        sτ = view(s, τ+1:n)[perm] # delayed and reordered time series
        mi_values[i] = _mutualinfo!(f, sτ, bins, edges)
    end
    return mi_values
end


"""
    _mutualinfo!(f, sτ, edges, bins0)
    
Calculate the mutual information between the distribution of the delayed time
series `sτ` and its original image.

The two series are partitioned in a joint histogram with axes divided by the
points given in `edges`; the distribution of the original image is given by
`bins0`.
The values of `sτ` must be arranged such that all the points of the bin `(1,j)`
are contained in the first `bins0[1]` positions, the points of the bin `(2,j)
are contained in the following `bins[2]` positions, etc.
  
The vector `f` is used as a placeholder to pre-allocate the histogram.
"""
function _mutualinfo!(f::AbstractVector, sτ::AbstractVector,
    bins0::AbstractVector{<:Integer}, edges::AbstractVector)

    # Initialize values
    mi = 0.0                    # `h` will contain the result
    processed = 0               # number of points of `sτ` that have been used
    # Go through `bins0`
    n = length(sτ)
    logn = log2(n)
    for b in bins0
        frag = view(sτ, processed.+(1:b)) # fragment of `sτ` in the `b`-th bin
        processed += b
        _frequencies!(f, edges, sort(frag)) # store the values of the histogram in `f`
        for (i,g) in enumerate(f)
            (g != 0) && (mi += g/n*(log2(g/b/bins0[i])+logn))
        end
    end
    return mi
end

"""
    _frequencies!(f, s, edges)

Calculate a histogram of values of `s` along the bins defined by `edges`.
Both `s` and `edges` must be sorted ascendingly. The frequencies (counts)
of `s` in each bin will be stored in the pre-allocated vector `f`. 
""" 
function _frequencies!(f::AbstractVector{T}, edges::AbstractVector, s::AbstractVector) where {T} 
    # Initialize the array of frequencies to zero
    fill!(f, zero(T))
    n = length(s)
    nbins = length(edges) - 1
    b = 1 # start in the first bin
    for i = 1:n
        if b ≥ nbins # the last bin is filled with the rest of points
            f[end] += n-i+1
            break
        end
        # when `s[i]` goes after the upper limit of the current bin,
        # move to the next bin and repeat until `s[i]` is found 
        while s[i] > edges[b+1]
            b += 1
        end
        f[b] += 1 # add point to the current bin
    end
end

### Functions for creating histograms ###

# For type stability, all must return the same type of tuple
Histogram{T} = Tuple{Vector{<:Integer}, Vector{T}} where {T}

"""
    _equalbins(s[; nbins, binwidth])
    
Create a histogram of the sorted series `s` with bins of the same width.
Either the number of bins (`nbins`) or their width (`binwidth`) must be
given as keyword argument (but not both).
"""
function _equalbins(s::AbstractVector{T}; kwargs...)::Histogram{T} where {T}
    # only one of `nbins` or `binwidth` can be passed 
    if length(kwargs) > 1
        throw(ArgumentError("the keyword argument can only be either `nbins` or `binwidth`"))
    elseif haskey(kwargs, :nbins)     # fixed number of bins
        nbins = Int(kwargs[:nbins])
        r = range(s[1], stop=s[end], length=nbins+1)
    elseif haskey(kwargs, :binwidth)
        binwidth = T(kwargs[:binwidth])        
        nbins = Int(div(s[end]-s[1],binwidth)+1)
        start = (s[1]+s[end]-nbins*binwidth)/2
        r = range(start, step=binwidth, length=nbins+1)
    else
        throw(ArgumentError("`nbins` or `binwidth` keyword argument required"))
    end
    bins = zeros(Int, nbins)
    _frequencies!(bins, r, s)
    edges = T[minimum(s); s[cumsum(bins)]]
    return (bins, edges)
end

"""
    _bisect(s)
    
Create a partition histogram of the sorted series `s` with a partition of its
space defined by a recursive bisection method. The first level partition
divides `s` in two segments with equal number of points; each
partition is divided into two further sub-pantitions, etc.,
until the distribution of the points in the highest level
subpartition is homogeneous.
"""
function _bisect(s::AbstractVector{T})::Histogram{T} where {T}
    # set up the vectors to return
    bins = Int[]
    edges = T[minimum(s)]
    _bisect!(bins, edges, s) # recursive step
    return (bins, edges)
end

function _bisect!(bins, edges, s)
    if _uniformtest(s) # no further partitions if it is uniform
        push!(bins, length(s))
        push!(edges, s[end])
    else
        n = length(s)
        half = div(n, 2)
        # first half
        _bisect!(bins, edges, view(s, 1:half))
        # second half
        _bisect!(bins, edges, view(s, half+1:n))
    end
    return
end

"""
    _uniformtest(s)
    
Test uniformity in the values of the sorted vector `s`.
"""
function _uniformtest(s) 
    n = length(s)
    # less than 10 points is not enough to test
    # between 10 and 19 points, divide the range of values in two sub-ranges
    # for 20 or more, divide the range in four sub-ranges
    # and make a chi-squared test
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
    binwidth = (s[end]-s[1])/nbins
    expected = round(Int, n/nbins)
    chisq = 0.0
    # set up first iteration
    first_item = 1
    for i = 1:nbins
        next_first = searchsortedfirst(s, s[1]+i*binwidth; lt=≤)
        observed = next_first - first_item
        chisq += (observed-expected)^2/expected
        (next_first > n) && break # stop if arrived at the end of `s`
        first_item = next_first
    end
    return (chisq < critical_chisq)
end

