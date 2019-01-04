using NearestNeighbors, Statistics, Distances

export estimate_dimension, stochastic_indicator
export Euclidean, Chebyshev, Cityblock

#####################################################################################
#                                Estimate Dimension                                 #
#####################################################################################
"""
    estimate_dimension(s::AbstractVector, τ::Int, γs = 1:5, method = "afnn"; kwargs...)

Compute a quantity that can estimate an optimal amount of
temporal neighbors `γ` to be used in [`reconstruct`](@ref) or [`embed`](@ref).

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute a quantity
for each `γ ∈ γs` based on the "nearest neighbors" in the embedded time series.

The quantity that is calculated depends on the algorithm defined by the string `method`:

* `"afnn"` (default) is Cao's "Averaged False Nearest Neighbors" method [1], which
    gives a ratio of distances between nearest neighbors. This ratio saturates
    around `1.0` near the optimal value of `γ` (see [`afnn`](@ref)).
* `"fnn"` is Kennel's "False Nearest Neighbors" method [2], which gives the
    number of points that cease to be "nearest neighbors" when the dimension
    increases. This number drops down to zero near the optimal value of `γ`.
    This method accepts the keyword arguments `rtol` and `atol`, which stand
    for the "tolerances" required by Kennel's algorithm (see [`fnn`](@ref)).
* `"f1nn"` is Krakovská's "False First Nearest Neighbors" method [3], which
    gives the ratio of pairs of points that cease to be "nearest neighbors"
    when the dimension increases. This number drops down to zero near the
    optimal value of `γ` (see [`f1nn`](@ref)).

`"afnn"` and `"f1nn"` also support the `metric` keyword, which can be any of
`Cityblock(), Euclidean(), Chebyshev()`. This metric is used both
for computing the nearest neighbors (`KDTree`s) as well as the distances necessary for
Cao's method (eqs. (2, 3) of [1]). Defaults to `Euclidean()`.

Please be aware that in **DynamicalSystems.jl** `γ` stands for the amount of temporal
neighbors and not the embedding dimension (`D = γ + 1`, see also [`embed`](@ref)).

## References

[1] : Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)

[2] : M. Kennel *et al.*, [Phys. Review A **45**(6), 3403-3411](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.3403)
(1992).

[3] : Anna Krakovská *et al.*, [J. Complex Sys. 932750 (2015)](https://doi.org/10.1155/2015/932750)
"""
function estimate_dimension(s::AbstractVector, τ::Int, γs = 1:5, method = "afnn";
    metric = Euclidean(), kwargs...)

    if method == "afnn"
        return afnn(s, τ, γs, metric)
    elseif method == "fnn"
        return fnn(s, τ, γs; kwargs...)
    elseif method == "f1nn"
        return f1nn(s, τ, γs, metric)
    end
end


"""
    afnn(s::AbstractVector, τ:Int, γs = 1:5, metric=Chebyshev())

Compute the parameter E₁ of Cao's "averaged false nearest neighbors" method for
determining the minimum embedding dimension of the time series `s`, with
a sequence of `τ`-delayed temporal neighbors [1].

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute the
values of `E₁` for each `γ ∈ γs`, according to Cao's Method (eq. 3 of [1]).

This quantity is a ratio of the averaged distances between the nearest neighbors
of the reconstructed time series, which quantifies the increment of those
distances when the number of temporal neighbors changes from `γ` to `γ+1`.

Please be aware that in **DynamicalSystems.jl** `γ` stands for the amount of temporal
neighbors and not the embedding dimension (`D = γ + 1`, see also [`embed`](@ref)).

Return the vector of all computed `E₁`s. To estimate a good value for `γ` from this,
find `γ` for which the value `E₁` saturates at some value around 1.

*Note: This method does not work for datasets with perfectly periodic signals.*

See also: [`estimate_dimension`](@ref), [`fnn`](@ref), [`f1nn`](@ref).

## References

[1] : Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)
"""
function afnn(s::AbstractVector{T}, τ::Int, γs = 1:5, metric=Chebyshev()) where {T}
    E1s = zeros(length(γs))
    aafter = 0.0
    aprev = _average_a(s, γs[1], τ, metric)
    for (i, γ) ∈ enumerate(γs)
        aafter = _average_a(s, γ+1, τ, metric)
        E1s[i] = aafter/aprev
        aprev = aafter
    end
    return E1s
end
# then use function `saturation_point(γs, E1s)` from ChaosTools

function _average_a(s::AbstractVector{T},γ,τ,metric) where {T}
    #Sum over all a(i,d) of the Ddim Reconstructed space, equation (2)
    Rγ = reconstruct(s[1:end-τ],γ,τ)
    tree2 = KDTree(Rγ, metric)
    nind = (x = knn(tree2, Rγ.data, 2)[1]; [ind[1] for ind in x])
    e = 0.0
    @inbounds for (i,j) ∈ enumerate(nind)
        δ = evaluate(metric, Rγ[i], Rγ[j])
        #If Rγ[i] and Rγ[j] are still identical, choose the next nearest neighbor
        if δ == 0.0
            j = knn(tree2, Rγ[i], 3, true)[1][end]
            δ = evaluate(metric, Rγ[i], Rγ[j])
        end
        e += _increase_distance(δ,s,i,j,γ,τ,metric)/δ
    end
    return e / (length(Rγ)-1)
end

# Function to increase the distance (p-norm) between two points `(i,j)` of
# the embedded time `s`series, by adding one temporal neighbor
_increase_distance(δ, s, i::Int, j::Int, γ::Int, τ::Int, ::Chebyshev) =
    max(δ, abs(s[i+γ*τ+τ] - s[j+γ*τ+τ]))
_increase_distance(δ, s, i::Int, j::Int, γ::Int, τ::Int, ::Euclidean) =
    sqrt(δ*δ + abs2(s[i+γ*τ+τ] - s[j+γ*τ+τ]) )
_increase_distance(δ, s, i::Int, j::Int, γ::Int, τ::Int, ::Cityblock) =
    δ + abs(s[i+γ*τ+τ] - s[j+γ*τ+τ])

"""
    fnn(s::AbstractVector, τ:Int, γs = 1:5; rtol=10.0, atol=2.0)

Calculate the number of "false nearest neighbors" (FNN) of the datasets created
from `s` with a sequence of `τ`-delayed temporal neighbors.

## Description
Given a dataset made by embedding `s` with `γ` temporal neighbors and delay `τ`,
the "false nearest neighbors" (FNN) are the pairs of points that are nearest to
each other at dimension `γ`, but are separated at dimension `γ+1`. Kennel's
criteria for detecting FNN are based on a threshold for the relative increment
of the distance between the nearest neighbors (`rtol`, eq. 4 in [1]), and
another threshold for the ratio between the increased distance and the
"size of the attractor" (`atol`, eq. 5 in [1]). These thresholds are given
as keyword arguments.

The returned value is a vector with the number of FNN for each `γ ∈ γs`. The
optimal value for `γ` is found at the point where the number of FNN approaches
zero.

Please be aware that in **DynamicalSystems.jl** `γ` stands for the amount of temporal
neighbors and not the embedding dimension (`D = γ + 1`, see also [`embed`](@ref)).

See also: [`estimate_dimension`](@ref), [`afnn`](@ref), [`f1nn`](@ref).

## References

[1] : M. Kennel *et al.*, "Determining embedding dimension for phase-space
reconstruction using a geometrical construction", *Phys. Review A 45*(6), 3403-3411
(1992).
"""
function fnn(s::AbstractVector, τ::Int, γs = 1:5; rtol=10.0, atol=2.0)
    rtol2 = rtol^2
    Ra = std(s, corrected=false)
    nfnn = zeros(length(γs))
    @inbounds for (k, γ) ∈ enumerate(γs)
        y = reconstruct(s[1:end-τ],γ,τ)
        tree = KDTree(y)
        nind = (x = knn(tree, y.data, 2)[1]; [ind[1] for ind in x])
        for (i,j) ∈ enumerate(nind)
            δ = norm(y[i]-y[j], 2)
            # If y[i] and y[j] are still identical, choose the next nearest neighbor
            # as in Cao's algorithm (not suggested by Kennel, but still advisable)
            if δ == 0.0
                j = knn(tree, y[i], 3, true)[1][end]
                δ = norm(y[i]-y[j])
            end
            δ1 = _increase_distance(δ,s,i,j,γ,τ,Euclidean())
            cond_1 = ((δ1/δ)^2 - 1 > rtol2) # equation (4) of Kennel
            cond_2 = (δ1/Ra > atol)         # equation (5) of Kennel
            if cond_1 | cond_2
                nfnn[k] += 1
            end
        end
    end
    return nfnn
end

"""
    f1nn(s::AbstractVector, τ:Int, γs = 1:5, metric = Euclidean())

Calculate the ratio of "false first nearest neighbors" (FFNN) of the datasets created
from `s` with a sequence of `τ`-delayed temporal neighbors.

## Description
Given a dataset made by embedding `s` with `γ` temporal neighbors and delay `τ`,
the "first nearest neighbors" (FFNN) are the pairs of points that are nearest to
each other at dimension `γ` that cease to be nearest neighbors at dimension
`γ+1` [1].

The returned value is a vector with the ratio between the number of FFNN and
the number of points in the dataset for each `γ ∈ γs`. The optimal value for `γ`
is found at the point where this ratio approaches zero.

Please be aware that in **DynamicalSystems.jl** `γ` stands for the amount of temporal
neighbors and not the embedding dimension (`D = γ + 1`, see also [`embed`](@ref)).

See also: [`estimate_dimension`](@ref), [`afnn`](@ref), [`fnn`](@ref).

## References

[1] : Anna Krakovská *et al.*, "Use of false nearest neighbours for selecting
variables and embedding parameters for state space reconstruction", *J Complex
Sys* 932750 (2015), DOI: 10.1155/2015/932750
"""
function f1nn(s::AbstractVector, τ::Int, γs = 1:5, metric = Euclidean())
    f1nn_ratio = zeros(length(γs))
    γ_prev = 0 # to recall what γ has been analyzed before
    Rγ = reconstruct(s[1:end-τ],γs[1],τ) # this is for the first iteration
    for (i, γ) ∈ enumerate(γs)
        if i>1 && γ!=γ_prev+1
            # Re-calculate the series with γ delayed dims if γ does not follow
            # the dimension of the previous iteration
            Rγ = reconstruct(s[1:end-τ],γ,τ)
        end
        (nf1nn, Rγ) = _compare_first_nn(s,γ,τ,Rγ,metric)
        f1nn_ratio[i] = nf1nn/length(Rγ)
        # Trim Rγ for the next iteration
        Rγ = Rγ[1:end-τ,:]
        γ_prev = γ
    end
    return f1nn_ratio
end

function _compare_first_nn(s, γ::Int, τ::Int, Rγ::Dataset{D,T}, metric) where {D,T}
    # This function compares the first nearest neighbors of `s`
    # embedded with Dimensions `γ` and `γ+1` (the former given as input)
    tree = KDTree(Rγ,metric)
    Rγ1 = reconstruct(s,γ+1,τ)
    tree1 = KDTree(Rγ1,metric)
    nf1nn = 0
    # For each point `i`, the fnn of `Rγ` is `j`, and the fnn of `Rγ1` is `k`
    nind = (x = knn(tree, Rγ.data, 2)[1]; [ind[1] for ind in x])
    @inbounds for  (i,j) ∈ enumerate(nind)
        k = knn(tree1, Rγ1.data[i], 2, true)[1][end]
        if j != k
            nf1nn += 1
        end
    end
    # `Rγ1` is returned to re-use it if necessary
    return (nf1nn, Rγ1)
end


"""
    stochastic_indicator(s::AbstractVector, τ:Int, γs = 1:4) -> E₂s

Compute an estimator for apparent randomness in a reconstruction with `γs` temporal
neighbors.

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute the
values of `E₂` for each `γ ∈ γs`, according to Cao's Method (eq. 5 of [1]).

Use this function to confirm that the
input signal is not random and validate the results of [`estimate_dimension`](@ref).
In the case of random signals, it should be `E₂ ≈ 1 ∀ γ`.

## References

[1] : Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)
"""
function stochastic_indicator(s::AbstractVector{T},τ, γs=1:4) where T # E2, equation (5)
    #This function tries to tell the difference between deterministic
    #and stochastic signals
    #Calculate E* for Dimension γ+1
    E2s = Float64[]
    for γ ∈ γs
        Rγ1 = reconstruct(s,γ+1,τ)
        tree1 = KDTree(Rγ1[1:end-1-τ])
        method = FixedMassNeighborhood(2)

        Es1 = 0.
        nind = (x = neighborhood(Rγ1[1:end-τ], tree1, method); [ind[1] for ind in x])
        for  (i,j) ∈ enumerate(nind)
            Es1 += abs(Rγ1[i+τ][end] - Rγ1[j+τ][end]) / length(Rγ1)
        end

        #Calculate E* for Dimension γ
        Rγ = reconstruct(s,γ,τ)
        tree2 = KDTree(Rγ[1:end-1-τ])
        Es2 = 0.
        nind = (x = neighborhood(Rγ[1:end-τ], tree2, method); [ind[1] for ind in x])
        for  (i,j) ∈ enumerate(nind)
            Es2 += abs(Rγ[i+τ][end] - Rγ[j+τ][end]) / length(Rγ)
        end
        push!(E2s, Es1/Es2)
    end
    return E2s
end
