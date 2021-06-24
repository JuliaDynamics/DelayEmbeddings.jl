#=
In this file a bunch of methods are collected that allow estimating the appropriate
embedding dimension in the scenario of traditional delay embeddings, where the delay
time is estimated first, and then the embedding dimension, while only constant delay
time is allowed.

All methods in this file are based on the idea of "false nearest neighbors".
=#
using Statistics, Distances
export delay_afnn, delay_fnn, delay_ifnn, delay_f1nn, stochastic_indicator

#####################################################################################
#                                Cao's method                                       #
#####################################################################################
"""
    delay_afnn(s::AbstractVector, τ:Int, ds = 2:6; metric=Euclidean(), w = 0) → E₁

Compute the parameter E₁ of Cao's "averaged false nearest neighbors" method for
determining the minimum embedding dimension of the time series `s`, with
a sequence of `τ`-delayed temporal neighbors.

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute the
values of `E₁` for each embedding dimension `d ∈ ds`, according to Cao's Method
(eq. 3 of[^Cao1997]).

This quantity is a ratio of the averaged distances between the nearest neighbors
of the reconstructed time series, which quantifies the increment of those
distances when the embedding dimension changes from `d` to `d+1`.

Return the vector of all computed `E₁`s. To estimate a good value for `d` from this,
find `d` for which the value `E₁` saturates at some value around 1.

*Note: This method does not work for datasets with perfectly periodic signals.*

`w` is the [Theiler window](@ref).

See also: [`optimal_traditional_de`](@ref) and [`stochastic_indicator`](@ref).
"""
function delay_afnn(s::AbstractVector{T}, τ::Int, ds = 2:6; metric=Euclidean(), w=0) where {T}
    E1s = zeros(length(ds))
    aafter = 0.0
    theiler = Theiler(w)
    aprev = _average_a(s, ds[1], τ, metric, theiler)
    for (i, d) ∈ enumerate(ds)
        aafter = _average_a(s, d+1, τ, metric, theiler)
        E1s[i] = aafter/aprev
        aprev = aafter
    end
    return E1s
end

function _average_a(s::AbstractVector{T},d,τ,metric,theiler) where {T}
    #Sum over all a(i,d) of the Ddim Reconstructed space, equation (2)
    Rd = embed(s[1:end-τ],d,τ)
    tree = KDTree(Rd, metric)
    _nind = bulkisearch(tree, Rd, NeighborNumber(1), theiler)
    nind = (x[1] for x in _nind) # bulksearch always returns vectors of vectors
    e = 0.0
    @inbounds for (i,j) ∈ enumerate(nind)
        δ = evaluate(metric, Rd[i], Rd[j])
        #If Rγ[i] and Rγ[j] are still identical, choose the next nearest neighbor
        if δ == 0.0
            j = isearch(tree, Rd[i], NeighborNumber(1), theiler(i))[end]
            δ = evaluate(metric, Rd[i], Rd[j])
        end
        e += _increase_distance(δ,s,i,j,d-1,τ,metric)/δ
    end
    return e / (length(Rd)-1)
end

# Function to increase the distance (p-norm) between two points `(i,j)` of
# the embedded time `s`series, by adding one temporal neighbor
_increase_distance(δ, s, i::Int, j::Int, γ::Int, τ::Int, ::Chebyshev) =
    max(δ, abs(s[i+γ*τ+τ] - s[j+γ*τ+τ]))
_increase_distance(δ, s, i::Int, j::Int, γ::Int, τ::Int, ::Euclidean) =
    sqrt(δ*δ + abs2(s[i+γ*τ+τ] - s[j+γ*τ+τ]) )
_increase_distance(δ, s, i::Int, j::Int, γ::Int, τ::Int, ::Cityblock) =
    δ + abs(s[i+γ*τ+τ] - s[j+γ*τ+τ])


function stochastic_indicator(s::AbstractVector{T}, τ) where T # E2, equation (5)
    @warn """
    The third argument of `stochastic_indicator` is now a range of embedding dimensions
    `d` instead of temporal entries `γ`, and `ds = γs .+ 1`.
    """
    stochastic_indicator(s, τ, 2:5)
end

"""
    stochastic_indicator(s::AbstractVector, τ:Int, ds = 2:5) -> E₂s

Compute an estimator for apparent randomness in a delay embedding with `ds` dimensions.

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute the
values of `E₂` for each `d ∈ ds`, according to Cao's Method (eq. 5 of [^Cao1997]).

Use this function to confirm that the
input signal is not random and validate the results of [`estimate_dimension`](@ref).
In the case of random signals, it should be `E₂ ≈ 1 ∀ d`.
"""
function stochastic_indicator(s::AbstractVector{T}, τ, ds) where T # E2, equation (5)
    #This function tries to tell the difference between deterministic
    #and stochastic signals
    #Calculate E* for Dimension γ+1
    E2s = Float64[]
    for d ∈ ds
        Rγ1 = embed(s,d+1,τ)
        tree1 = KDTree(Rγ1[1:end-1-τ])
        Es1 = 0.0
        _nind = bulkisearch(tree1, Rγ1[1:end-τ], NeighborNumber(1), Theiler(0))
        nind = (x[1] for x in _nind) # bulksearch always returns vectors of vectors
        for  (i,j) ∈ enumerate(nind)
            Es1 += abs(Rγ1[i+τ][end] - Rγ1[j+τ][end]) / length(Rγ1)
        end

        #Calculate E* for Dimension d
        Rγ = embed(s,d,τ)
        tree2 = KDTree(Rγ[1:end-1-τ])
        Es2 = 0.0
        _nind = bulkisearch(tree2, Rγ[1:end-τ], NeighborNumber(1), Theiler(0))
        nind = (x[1] for x in _nind)
        for  (i,j) ∈ enumerate(nind)
            Es2 += abs(Rγ[i+τ][end] - Rγ[j+τ][end]) / length(Rγ)
        end
        push!(E2s, Es1/Es2)
    end
    return E2s
end

#####################################################################################
#                                FNN / F1NN                                         #
#####################################################################################
"""
    delay_fnn(s::AbstractVector, τ:Int, ds = 2:6; rtol=10.0, atol=2.0) → FNNs

Calculate the number of "false nearest neighbors" (FNNs) of the datasets created
from `s` with `embed(s, d, τ) for d ∈ ds`.

## Description
Given a dataset made by `embed(s, d, τ)`
the "false nearest neighbors" (FNN) are the pairs of points that are nearest to
each other at dimension `d`, but are separated at dimension `d+1`. Kennel's
criteria for detecting FNN are based on a threshold for the relative increment
of the distance between the nearest neighbors (`rtol`, eq. 4 in[^Kennel1992]), and
another threshold for the ratio between the increased distance and the
"size of the attractor" (`atol`, eq. 5 in[^Kennel1992]). These thresholds are given
as keyword arguments.

The returned value is a vector with the number of FNN for each `γ ∈ γs`. The
optimal value for `γ` is found at the point where the number of FNN approaches
zero.

See also: [`optimal_traditional_de`](@ref).
"""
function delay_fnn(s::AbstractVector, τ::Int, ds = 2:6; rtol=10.0, atol=2.0)
    rtol2 = rtol^2
    Ra = std(s, corrected=false)
    nfnn = zeros(length(ds))
    @inbounds for (k, d) ∈ enumerate(ds)
        y = embed(s[1:end-τ],d,τ)
        tree = KDTree(y)
        _nind = bulkisearch(tree, y, NeighborNumber(1), Theiler(0))
        nind = (x[1] for x in _nind) # bulksearch always returns vectors of vectors
        # nind = (x = NearestNeighbors.knn(tree, y.data, 2)[1]; [ind[1] for ind in x])
        for (i,j) ∈ enumerate(nind)
            δ = norm(y[i]-y[j], 2)
            # If y[i] and y[j] are still identical, choose the next nearest neighbor
            # as in Cao's algorithm (not suggested by Kennel, but still advisable)
            if δ == 0.0
                j = Neighborhood.knn(tree, y[i], 2, Theiler(0)(i))[end]
                δ = norm(y[i]-y[j])
            end
            δ1 = _increase_distance(δ,s,i,j,d-1,τ,Euclidean())
            cond_1 = ((δ1/δ)^2 - 1 > rtol2) # equation (4) of Kennel
            cond_2 = (δ1/Ra > atol)         # equation (5) of Kennel
            if cond_1 | cond_2
                nfnn[k] += 1
            end
        end
        nfnn[k] /= length(nind)
    end
    return nfnn
end

"""
    delay_f1nn(s::AbstractVector, τ::Int, ds = 2:6, metric = Euclidean())

Calculate the ratio of "false first nearest neighbors" (FFNN) of the datasets created
from `s` with `embed(s, d, τ) for d ∈ ds`.

## Description
Given a dataset made by `embed(s, d, τ)`
the "false first nearest neighbors" (FFNN) are the pairs of points that are nearest to
each other at dimension `d` that cease to be nearest neighbors at dimension
`d+1`.

The returned value is a vector with the ratio between the number of FFNN and
the number of points in the dataset for each `d ∈ ds`. The optimal value for `d`
is found at the point where this ratio approaches zero.

See also: [`optimal_traditional_de`](@ref).
"""
function delay_f1nn(s::AbstractVector, τ::Int, ds = 2:6, metric = Euclidean())
    f1nn_ratio = zeros(length(ds))
    γ_prev = 0 # to recall what γ has been analyzed before
    Rγ = embed(s[1:end-τ],ds[1],τ) # this is for the first iteration
    for (i, γ) ∈ enumerate(ds)
        if i>1 && γ!=γ_prev+1
            # Re-calculate the series with γ delayed dims if γ does not follow
            # the dimension of the previous iteration
            Rγ = embed(s[1:end-τ],γ,τ)
        end
        (nf1nn, Rγ) = _compare_first_nn(s,γ,τ,Rγ,metric)
        f1nn_ratio[i] = nf1nn/length(Rγ)
        # Trim Rγ for the next iteration
        Rγ = Rγ[1:end-τ,:]
        γ_prev = γ
    end
    return f1nn_ratio
end

function _compare_first_nn(s, γ::Int, τ::Int, Rγ::AbstractDataset{D,T}, metric) where {D,T}
    # This function compares the first nearest neighbors of `s`
    # embedded with Dimensions `γ` and `γ+1` (the former given as input)
    tree = KDTree(Rγ,metric)
    Rγ1 = embed(s,γ+1,τ)
    tree1 = KDTree(Rγ1,metric)
    nf1nn = 0
    # For each point `i`, the fnn of `Rγ` is `j`, and the fnn of `Rγ1` is `k`
    theiler = Theiler(0)
    _nind = bulkisearch(tree, Rγ, NeighborNumber(1), theiler)
    nind = (x[1] for x in _nind) # bulksearch always returns vectors of vectors
    # nind = (x = NearestNeighbors.knn(tree, Rγ.data, 2)[1]; [ind[1] for ind in x])
    @inbounds for (i,j) ∈ enumerate(nind)
        k = Neighborhood.knn(tree1, Rγ1.data[i], 1, theiler(i))[1][1]
        if j != k
            nf1nn += 1
        end
    end
    # `Rγ1` is returned to re-use it if necessary
    return (nf1nn, Rγ1)
end


#####################################################################################
#                               IFNN (Hegger & Kantz)                               #
#####################################################################################
"""
    delay_ifnn(s::Vector, τ::Int, ds = 1:10; kwargs...) → `FNNs`
Compute and return the `FNNs`-statistic for the time series `s` and a uniform
time delay `τ` and embedding dimensions `ds` after [^Hegger1999].
In this notation `γ ∈ γs = d-1`, if `d` is the embedding dimension. This fraction
tends to 0 when the optimal embedding dimension with an appropriate lag is
reached.

## Keywords
*`r = 2`: Obligatory threshold, which determines the maximum tolerable spreading
    of trajectories in the reconstruction space.
*`metric = Euclidean`: The norm used for distance computations.
*`w = 1` = The [Theiler window](@ref).

See also: [`optimal_traditional_de`](@ref).
"""
function delay_ifnn(s::AbstractVector{T}, τ::Int, ds = 1:10;
            r::Real = 2, w::Int = 1, metric = Euclidean()) where {T}
    @assert all(x -> x ≥ 0, ds)
    s = (s .- mean(s)) ./ std(s)
    Y_act = embed(s[1:end-τ],ds[1],τ)

    vtree = KDTree(Y_act, metric)
    _, NNdist_old = all_neighbors(vtree, Y_act, 1:length(Y_act), 1, w)

    FNNs = zeros(length(ds))
    bm = 0
    for (i, d) ∈ enumerate(ds)
        Y_act = embed(s[1:end-τ],d+1,τ)
        Y_act = standardize(Y_act)
        vtree = KDTree(Y_act, metric)
        _, NNdist_new = all_neighbors(vtree, Y_act, 1:length(Y_act), 1, w)

        FNNs[i] = fnn_embedding_cycle(
            view(NNdist_old, 1:length(Y_act)), NNdist_new, r
        )

        NNdist_old = NNdist_new
    end
    return FNNs
end


"""
    fnn_embedding_cycle(NNdist, NNdistnew, r=2) -> FNNs
Compute the amount of false nearest neighbors `FNNs`, when adding another component
to a given (vector-) time series. This new component is the `τ`-lagged version
of a univariate time series. `NNdist` is storing the distances of the nearest
neighbor for all considered fiducial points and `NNdistnew` is storing the
distances of the nearest neighbor for each fiducial point in one embedding
dimension higher using a given `τ`. The obligatory threshold `r` is by default
set to 2.
"""
function fnn_embedding_cycle(NNdist, NNdistnew, r::Real=2)
    @assert length(NNdist) == length(NNdistnew) "Both input vectors need to store the same number of distances."
    N = length(NNdist)
    fnns = 0
    fnns2= 0
    inverse_r = 1/r
    @inbounds for i = 1:N
        if NNdistnew[i][1]/NNdist[i][1] > r && NNdist[i][1] < inverse_r
            fnns +=1
        end
        if NNdist[i][1] < inverse_r
            fnns2 +=1
        end
    end
    if fnns==0
        return 1
    else
        return fnns/fnns2
    end
end
