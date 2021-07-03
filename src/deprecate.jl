@deprecate regularize standardize

export mutualinformation
function mutualinformation(args...; kwargs...)
    @warn "`mutualinformation` is deprecated in favor of `selfmutualinfo`."
    selfmutualinfo(args...; kwargs...)
end

function delay_afnn(s::AbstractVector, τ::Int, ds, metric)
    return delay_afnn(s, τ, ds; metric, w = 0)
end


#####################################################################################
#                               OLD method TODO: remove                             #
#####################################################################################
export estimate_dimension
export afnn, fnn, ifnn, f1nn

# Deprecations for new syntax and for removing `reconstruct`.
for f in (:afnn, :fnn, :ifnn, :f1nn)
    q = quote
        function $(f)(s, τ, γs = 1:6, args...; kwargs...)
            dep = """
            function `$($(f))` is deprecated because it uses "γ" (amount of temporal
            neighbors in delay vector) and `reconstruct`. These syntaxes are being phased
            out in favor of `embed` and using `d` directly, the embedding dimension.

            Use instead `$($(Symbol(:delay_, f)))` and replace given `γs` with `γs.+1`.
            """
            @warn dep
            return $(Symbol(:delay_, f))(s, τ, γs .+ 1, args...; kwargs...)
        end
    end
    @eval $q
end

"""
    estimate_dimension(s::AbstractVector, τ::Int, γs = 1:5, method = "afnn"; kwargs...)

Compute a quantity that can estimate an optimal amount of
temporal neighbors `γ` to be used in [`reconstruct`](@ref) or [`embed`](@ref).

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute a quantity
for each `γ ∈ γs` based on the "nearest neighbors" in the embedded time series.

The quantity that is calculated depends on the algorithm defined by the string `method`:

* `"afnn"` (default) is Cao's "Averaged False Nearest Neighbors" method[^Cao1997], which
    gives a ratio of distances between nearest neighbors. This ratio saturates
    around `1.0` near the optimal value of `γ` (see [`afnn`](@ref)).
* `"fnn"` is Kennel's "False Nearest Neighbors" method[^Kennel1992], which gives the
    number of points that cease to be "nearest neighbors" when the dimension
    increases. This number drops down to zero near the optimal value of `γ`.
    This method accepts the keyword arguments `rtol` and `atol`, which stand
    for the "tolerances" required by Kennel's algorithm (see [`fnn`](@ref)).
* `"f1nn"` is Krakovská's "False First Nearest Neighbors" method[^Krakovská2015], which
    gives the ratio of pairs of points that cease to be "nearest neighbors"
    when the dimension increases. This number drops down to zero near the
    optimal value of `γ` (see [`f1nn`](@ref)). This is the worse method of all.

`"afnn"` and `"f1nn"` also support the `metric` keyword, which can be any of
`Cityblock(), Euclidean(), Chebyshev()`. This metric is used both
for computing the nearest neighbors (`KDTree`s) as well as the distances necessary for
Cao's method (eqs. (2, 3) of [1]). Defaults to `Euclidean()` (note that [1] used
`Chebyshev`).

Please be aware that in **DynamicalSystems.jl** `γ` stands for the amount of temporal
neighbors and not the embedding dimension (`D = γ + 1`, see also [`embed`](@ref)).

[^Cao1997]: Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)

[^Kennel1992]: M. Kennel *et al.*, [Phys. Review A **45**(6), (1992)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.3403).

[^Krakovská2015]: Anna Krakovská *et al.*, [J. Complex Sys. 932750 (2015)](https://doi.org/10.1155/2015/932750)
"""
function estimate_dimension(s::AbstractVector, τ::Int, γs = 1:5, method = "afnn";
    metric = Euclidean(), kwargs...)
    @warn """
    Using `estimate_dimension` is deprecated in favor of either calling `delay_afnn, delay_fnn, ...`
    directly or using the function `optimal_traditional_de`.
    """
    if method == "afnn"
        return afnn(s, τ, γs, metric)
    elseif method == "fnn"
        return fnn(s, τ, γs; kwargs...)
    elseif method == "ifnn"
        return ifnn(s, τ, γs; kwargs...)
    elseif method == "f1nn"
        return f1nn(s, τ, γs, metric)
    else
        error("unrecognized method")
    end
end

#####################################################################################
#                                Pairwse Distance                                   #
#####################################################################################
using NearestNeighbors, StaticArrays, LinearAlgebra

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
#                Old Neighborhood Interface, deprecated                             #
#####################################################################################
import NearestNeighbors
using StaticArrays
using Distances: Euclidean, Metric

export AbstractNeighborhood
export FixedMassNeighborhood, FixedSizeNeighborhood
export neighborhood, KDTree

"""
    AbstractNeighborhood
Supertype of methods for deciding the neighborhood of points for a given point.

Concrete subtypes:
* `FixedMassNeighborhood(K::Int)` :
  The neighborhood of a point consists of the `K`
  nearest neighbors of the point.
* `FixedSizeNeighborhood(ε::Real)` :
  The neighborhood of a point consists of all
  neighbors that have distance < `ε` from the point.

See [`neighborhood`](@ref) for more.
"""
abstract type AbstractNeighborhood end

struct FixedMassNeighborhood <: AbstractNeighborhood
    K::Int
    function FixedMassNeighborhood(k::Int = 1)
        @warn "FixedMassNeighborhood is deprecated in favor of NeighborNumber."
        return new(k)
    end
end

struct FixedSizeNeighborhood <: AbstractNeighborhood
    ε::Float64
    function FixedSizeNeighborhood(ε::Real = 0.01)
        @warn "FixedSizeNeighborhood is deprecated in favor of WithinRange."
        return new(ε)
    end
end

"""
    neighborhood(point, tree, ntype)
    neighborhood(point, tree, ntype, n::Int, w::Int = 1)

Return a vector of indices which are the neighborhood of `point` in some
`data`, where the `tree` was created using `tree = KDTree(data [, metric])`.
The `ntype` is the type of neighborhood and can be any subtype
of [`AbstractNeighborhood`](@ref).

Use the second method when the `point` belongs in the data,
i.e. `point = data[n]`. Then `w` stands for the Theiler window (positive integer).
Only points that have index
`abs(i - n) ≥ w` are returned as a neighborhood, to exclude close temporal neighbors.
The default `w=1` is the case of excluding the `point` itself.

## References

`neighborhood` simply interfaces the functions
`NearestNeighbors.knn` and `inrange` from
[NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) by using
the argument `ntype`.
"""
function neighborhood(point::AbstractVector, tree,
                      ntype::FixedMassNeighborhood, n::Int, w::Int = 1)
    @warn "`neighborhood` is deprecated in favor of using `search` and Neighborhood.jl."
    idxs, = NearestNeighbors.knn(tree, point, ntype.K, false, i -> abs(i-n) < w)
    return idxs
end
function neighborhood(point::AbstractVector, tree, ntype::FixedMassNeighborhood)
    @warn "`neighborhood` is deprecated in favor of using `search` and Neighborhood.jl."
    idxs, = NearestNeighbors.knn(tree, point, ntype.K, false)
    return idxs
end

function neighborhood(point::AbstractVector, tree,
                      ntype::FixedSizeNeighborhood, n::Int, w::Int = 1)

@warn "`neighborhood` is deprecated in favor of using `search` and Neighborhood.jl."
    idxs = NearestNeighbors.inrange(tree, point, ntype.ε)
    filter!((el) -> abs(el - n) ≥ w, idxs)
    return idxs
end
function neighborhood(point::AbstractVector, tree, ntype::FixedSizeNeighborhood)
    @warn "`neighborhood` is deprecated in favor of using `search` and Neighborhood.jl."
    idxs = NearestNeighbors.inrange(tree, point, ntype.ε)
    return idxs
end
