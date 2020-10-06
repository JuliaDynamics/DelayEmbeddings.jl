export optimal_traditional_de

# TODO: Perhaps what should happen with each quantitity should be documented at the
# appropriate functin instead of here.

"""
    optimal_traditional_de(s, method = "ifnn", dmethod = "mi_min"; kwargs...) → 𝒟, τ

Produce an optimal delay embedding `𝒟` of the given timeseries `s` by
using the traditional approach of first finding an optimal (and constant) delay time using
[`estimate_delay`](@ref) with the given `dmethod`, and then an optimal embedding dimension.
Return the embedding `𝒟` and the optimal delay time `τ` (the optimal embedding dimension
is just `size(𝒟, 2)`).

For estimating the dimension we use the given `method`, which can be:

* `"ifnn"` is the "Improved False Nearest Neighbors" from Hegger & Kantz[^Hegger1999],
    which gives the fraction of false nearest neighbors. This fraction goes to 0
    after the optimal embedding dimension. This is the best method.
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
    optimal embedding dimension (see [`f1nn`](@ref)). This is the worse method.

`"afnn"` and `"f1nn"` also support the `metric` keyword, which can be any of
`Cityblock(), Euclidean(), Chebyshev()`. This metric is used both
for computing the nearest neighbors (`KDTree`s) as well as the distances necessary for
Cao's method (eqs. (2, 3) of [1]). Defaults to `Euclidean()` (note that [1] used
`Chebyshev`).

## Keywords

[^Cao1997]: Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)

[^Kennel1992]: M. Kennel *et al.*, [Phys. Review A **45**(6), (1992)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.3403).

[^Krakovská2015]: Anna Krakovská *et al.*, [J. Complex Sys. 932750 (2015)](https://doi.org/10.1155/2015/932750)

[^Hegger1999]: Hegger & Kantz, [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
function optimal_traditional_de(s::AbstractVector, method::String = "ifnn", ;
    delay::String = "mi_min", τs = 1:min(100, length(s)), kwargs...)

Compute the reconstructed trajectory `Y` from a time series `s` using the
standard time delay embedding. You can choose various combinations of the delay
estimators `delaymethod` (Default is "mi_min") and embedding dimension estimators
(Default is "afnn").

You can choose delay estimators from [`estimate_delay`](@ref):
* `"mi_min"` : delay of first minimum of mutual information of `s` with itself
  (shifted for various `τs`). <- Default
* `"ac_zero"` : first delay at which the auto-correlation function becomes <0.
* `"ac_min"` : delay of first minimum of the auto-correlation function.

You can choose embedding estimators from [`estimate_dimension`](@ref):
* `"afnn"` : Cao's method [`afnn`](@ref). <- Default
* `"fnn"`  : Kennel et al.'s traditional method [`fnn`](@ref)
* `"ifnn"` : improved fnn-method by Hegger & Kantz [`ifnn`](@ref)
* `"f1nn"` : delay of first minimum of the auto-correlation function.

Keyword arguments:
*`thres = 0.05`: This threshold determines the tolerable deviation of the
    proposed statistic from the optimal value of 1 (in case of "afnn") or 0 (in
    case of the other dimension estimators), for breaking the algorithm.
*`dmax = 10`: The maximum embedding dimension, which is encountered by the
    algorithm.
*`w = 1`: The Theiler window, which excludes temporally correlated points from
    the nearest neighbor search (does only work for "afnn", yet).
*`rtol = 10.`: threshold for the relative increment of the distance between the
    nearest neighbors in case of "fnn"
*`atol = 2.`: another threshold for the ratio between the increased distance and
    the "size of the attractor" in case of "fnn"
*`r = 2`:  Obligatory threshold, which determines the maximum tolerable spreading
    of trajectories in the reconstruction space in case of "ifnn".

## Description
Given the scalar timeseries `s` and the requested delay and dimension estimator
methods, this function returns the reconstruction vectors `Y`, given the threshold
`thres`. When this threshold is exceeded, the algorithm breaks. In case of "afnn"
also the [`stochastic_indicator`](@ref) gets checked. Also returns the picked
delay `τ` and the statistic `d_statistic`, according to the chosen dimension
estimation method.

*Note that `thres` needs to be adapted in case of noise contamination of the
input data, otherwise it will not terminate/ give a valid embedding*

"""
function optimal_traditional_de(s::AbstractVector, delaymethod::String= "mi_min",
    dimensionmethod::String = "afnn"; thres::Real = 0.05, dmax::Int = 10,
    w::Int =1, rtol=10.0, atol=2.0, )

    @assert delaymethod=="ac_zero" || delaymethod=="mi_min" || delaymethod=="ac_min"
    @assert dimensionmethod=="afnn" || dimensionmethod=="fnn" || dimensionmethod=="ifnn" || dimensionmethod=="f1nn"

    τ = estimate_delay(s, delaymethod)

    m, Y = 0, nothing

    if dimensionmethod=="ifnn"
        dimension_statistic = estimate_dimension(s, τ, γs = 1:dmax, method = dimensionmethod; r = r, w = w)
        Y, τ = fnn_embed(s, τ, dimension_statistic, thres)
    elseif dimensionmethod=="fnn"
        dimension_statistic = estimate_dimension(s, τ, γs = 1:dmax, method = dimensionmethod; rtol = rtol, atol = atol)
        Y, τ = fnn_embed(s, τ, dimension_statistic, thres)
    elseif dimensionmethod=="afnn"
        dimension_statistic = estimate_dimension(s, τ, γs = 1:dmax, method = dimensionmethod)
        Y, τ = cao_embed(s, τ, dimension_statistic, thres)
        E2 = stochastic_indicator(s, τ, γs = 1:dmax)
        flag = is_stochastic(E2, thres)
        if flag
            println("Stochastic signal."*
                    "Valid embedding NOT achieved ⨉.")
        end
    elseif dimensionmethod=="f1nn"
        dimension_statistic = estimate_dimension(s, τ, γs = 1:dmax, method = dimensionmethod)
        Y, τ = fnn_embed(s, τ, dimension_statistic, thres)
    end
    return Y, τ, dimension_statistic
end


"""
Helper function for selecting the appropriate embedding dimension from the
statistic when using Kennel', Hegger's or Krakovskas's method.
"""
function fnn_embed(s::Vector{T}, τ::Int, rat::Vector, thres::Real) where {T}
    @assert length(rat) > 1
    flag = false
    m, Y = 0, nothing
    for i = 2:length(rat)
        if rat[i] ≤ thres
            m = i
            break
        elseif rat[i] > rat[i-1]
            flag = true
            m = i
            break
        end
    end
    if m == length(rat)
        Y = s
        println("Sufficiently small FNNs NOT reached."*
                "Valid embedding NOT achieved ⨉.")
    else
        Y =  m > 1 ? embed(s, m, τ) : s
        println("Algorithm stopped due to sufficiently small FNNs. "*
                "Valid embedding achieved ✓.")
        if flag
            println("Algorithm stopped due to FNNs. "*
                    "Double-check the FNN-statistic.")
        end
    end
    return Y, τ
end


"""
Helper function for selecting the appropriate embedding dimension from the
statistic when using Cao's method.
"""
function cao_embed(s::Vector{T}, τ::Int, rat::Vector, thres::Real) where {T}
    m, Y = 0, nothing
    for i = 1:length(rat)
        if abs(1-rat[i]) < thres
            m = i
            break
        end
    end
    if m == length(rat)
        Y = s
        println("NO convergence of E₁-statistic."*
                "Valid embedding NOT achieved ⨉.")
    else
        Y =  m > 1 ? embed(s, m, τ) : s
        println("Algorithm stopped due to convergence of E₁-statistic. "*
                "Valid embedding achieved ✓.")
    end
    return Y, τ
end

"""
Helper function for Cao's method, whether to decide if the input signal is
stochastic or not.
"""
function is_stochastic(rat::Vector{T}, thres::Real) where {T}
    cnt = 0
    for i = 1:length(rat)
        if abs(1-rat[i]) > thres
            cnt += 1
        end
    end
    if cnt == 0
        flag = true
    else
        flag = false
    end
    return flag
end
