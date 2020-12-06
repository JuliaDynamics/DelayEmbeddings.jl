using StatsBase: autocor
export estimate_delay, exponential_decay_fit, autocor

#####################################################################################
#                               Estimate Delay Times                                #
#####################################################################################
"""
    estimate_delay(s, method::String [, τs = 1:100]; kwargs...) -> τ

Estimate an optimal delay to be used in [`embed`](@ref).
The `method` can be one of the following:

* `"ac_zero"` : first delay at which the auto-correlation function becomes <0.
* `"ac_min"` : delay of first minimum of the auto-correlation function.
* `"mi_min"` : delay of first minimum of mutual information of `s` with itself
  (shifted for various `τs`).
  Keywords `nbins, binwidth` are propagated into [`selfmutualinfo`](@ref).
* `"exp_decay"` : [`exponential_decay_fit`](@ref) of the correlation function rounded
   to an integer (uses least squares on `c(t) = exp(-t/τ)` to find `τ`).
* `"exp_extrema"` : same as above but the exponential fit is done to the
  absolute value of the local extrema of the correlation function.

Both the mutual information and correlation function (`autocor`) are computed _only_
for delays `τs`. This means that the `min` methods can never return the first value
of `τs`!

The method `mi_min` is significantly more accurate than the others and also returns
good results for most timeseries. It is however the slowest method (but still quite fast!).
"""
function estimate_delay(x::AbstractVector, method::String,
    τs = 1:min(100, length(x)); kwargs...)

    issorted(τs) || error("`τs` must be sorted")

    if method=="ac_zero"
        c = autocor(x, τs; demean=true)
        i = 1
        while c[i] > 0 # Find 0 crossing
            i += 1
            if i == length(c)
                @warn "Did not cross 0 value, returning last `τ`."
                return τs[end]
            end
        end
        return τs[i]
    elseif method=="ac_min"
        c = autocor(x, τs, demean=true)
        return mincrossing(c, τs)
    elseif method=="mi_min"
        c = selfmutualinfo(x, τs; kwargs...)
        return mincrossing(c, τs)
    elseif method=="exp_decay"
        c = autocor(x, τs; demean=true)
        if any(x -> x ≤ 0, c)
            error("The correlation function has elements that are ≤ 0. "*
            "We can't fit an exponential to it. Please choose another method.")
        end
        τ = exponential_decay_fit(τs, c)
        return round(Int,τ)
    elseif method=="exp_extrema"
        c = autocor(x, τs; demean=true)
        max_ind, min_ind = findlocalextrema(c)
        idxs = sort!(append!(max_ind, min_ind))
        ca = abs.(c[idxs])
        τa = τs[idxs]
        if any(x -> x ≤ 0, ca)
            error("The absolute correlation function has elements that are = 0. "*
            "We can't fit an exponential to it. Please choose another method.")
        end
        τ = exponential_decay_fit(τa, ca)
        return round(Int,τ)
    else
        throw(ArgumentError("Unknown method for `estimate_delay`."))
    end
end

function mincrossing(c, τs)
    i = 1
    while c[i+1] < c[i]
        i+= 1
        if i == length(c)-1
            @warn "Did not encounter a minimum, returning last `τ`."
            return τs[end]
        end
    end
    return τs[i]
end

"""
    exponential_decay_fit(x, y, weight = :equal) -> τ
Perform a least square fit of the form `y = exp(-x/τ)` and return `τ`.
Taken from:  http://mathworld.wolfram.com/LeastSquaresFittingExponential.html.
Assumes equal lengths of `x, y` and that `y ≥ 0`.

To use the method that gives more weight to small values of `y`, use `weight = :small`.
"""
function exponential_decay_fit(X, Y, weight = :equal)
    @inbounds begin
        L = length(X)
        b = if weight == :equal
            sy = sum(Y); sx = sum(X)
            a1 = sum(X[i]*Y[i]*log(Y[i]) for i in 1:L)
            a2 = sum(X[i]*Y[i] for i in 1:L)
            a3 = sum(Y[i]*log(Y[i]) for i in 1:L)
            a4 = sum(X[i]*X[i]*Y[i] for i in 1:L)
            b = (sy*a1 - a2*a3) / (sy*a4 - a2*a2)
        elseif weight == :small
            sx = sum(X)
            c1 = sum(X[i]*log(Y[i]) for i in 1:L)
            c2 = sum(log(y) for y in Y)
            c3 = sum(x*x for x in X)
            b = (L*c1 - sx*c2)/(L*c3 - sx*sx)
        end
        return -1.0/b
    end
end


#####################################################################################
#                               Mutual information                                  #
#####################################################################################
export selfmutualinfo, mutualinformation
function mutualinformation(args...; kwargs...)
    @warn "`mutualinformation` is deprecated in favor of `selfmutualinfo`."
    selfmutualinfo(args...; kwargs...)
end

"""
    selfmutualinfo(s, τs; kwargs...) → m

Calculate the mutual information between the time series `s` and itself
delayed by `τ` points for `τ` ∈ `τs`, using an _improvement_ of the method
outlined by Fraser & Swinney in[^Fraser1986].

## Description

The joint space of `s` and its `τ`-delayed image (`sτ`) is partitioned as a
rectangular grid, and the mutual information is computed from the joint and
marginal frequencies of `s` and `sτ` in the grid as defined in [1].
The mutual information values are returned in a vector `m` of the same length
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

[^Fraser1986]: Fraser A.M. & Swinney H.L. "Independent coordinates for strange attractors from mutual information" *Phys. Rev. A 33*(2), 1986, 1134:1140.
"""
function selfmutualinfo(s::AbstractVector{T}, τs::AbstractVector{Int};
    kwargs...) where {T}
    n = length(s)
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
function _frequencies!(f::AbstractVector{T}, edges::AbstractVector,
    s::AbstractVector) where {T}
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
        while s[i] > edges[b+1] && b < nbins
            b += 1
        end
        f[b] += 1 # add point to the current bin
    end
end

### Functions for creating histograms ###

# For type stability, all must return the same type of tuple
MIHistogram{T} = Tuple{Vector{<:Integer}, Vector{T}} where {T}

"""
    _equalbins(s[; nbins, binwidth])

Create a histogram of the sorted series `s` with bins of the same width.
Either the number of bins (`nbins`) or their width (`binwidth`) must be
given as keyword argument (**but not both**).
"""
function _equalbins(s::AbstractVector{T}; kwargs...)::MIHistogram{T} where {T}
    # only one of `nbins` or `binwidth` can be passed
    if length(kwargs) > 1
        throw(ArgumentError("the keyword argument can only be either `nbins` "*
        "or `binwidth`"))
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
function _bisect(s::AbstractVector{T})::MIHistogram{T} where {T}
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
