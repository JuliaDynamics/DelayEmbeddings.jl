export optimal_traditional_de

# TODO: Perhaps what should happen with each quantitity should be documented at the
# appropriate functin instead of here.

"""
    optimal_traditional_de(s, dmethod = "mi_min", method = "afnn"; kwargs...) → 𝒟, τ, x

Produce an optimal delay embedding `𝒟` of the given timeseries `s` by
using the traditional approach of first finding an optimal (and constant) delay time using
[`estimate_delay`](@ref) with the given `dmethod`, and then an optimal embedding dimension.
Return the embedding `𝒟` and the optimal delay time `τ` (the optimal embedding dimension `d`
is just `size(𝒟, 2)`) and the actual statistic `x` used to estimate optimal `d`.

For estimating the dimension we use the given `method`, which can be:

* `"afnn"` (default) is Cao's "Averaged False Nearest Neighbors" method[^Cao1997], which
    gives a ratio of distances between nearest neighbors. This ratio saturates
    around `1.0` near the optimal value of `γ` (see [`afnn`](@ref)).
* `"ifnn"` is the "Improved False Nearest Neighbors" from Hegger & Kantz[^Hegger1999],
    which gives the fraction of false nearest neighbors. This fraction goes to 0
    after the optimal embedding dimension.
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
```
thres::Real = 0.05, dmax::Int = 10,
w::Int=1, rtol=10.0, atol=2.0, τs = 1:100, metric = Euclidean()
```

[^Cao1997]: Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)

[^Kennel1992]: M. Kennel *et al.*, [Phys. Review A **45**(6), (1992)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.3403).

[^Krakovská2015]: Anna Krakovská *et al.*, [J. Complex Sys. 932750 (2015)](https://doi.org/10.1155/2015/932750)

[^Hegger1999]: Hegger & Kantz, [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
function optimal_traditional_de(s::AbstractVector, delaymethod::String= "mi_min",
    dimensionmethod::String = "afnn"; thres::Real = 0.05, dmax::Int = 10,
    w::Int=1, rtol=10.0, atol=2.0, τs = 1:100, metric = Euclidean())

    @assert dimensionmethod ∈ ("afnn", "fnn", "ifnn", "f1nn")

    τ = estimate_delay(s, delaymethod, τs)
    γs = 1:dmax-1 # TODO: This must be updated to just dimension only

    if dimensionmethod=="afnn"
        dimension_statistic = estimate_dimension(s, τ, γs, dimensionmethod)
        Y, τ = cao_embed(s, τ, dimension_statistic, thres)
        E2 = stochastic_indicator(s, τ, γs)
        flag = is_stochastic(E2, thres)
        flag && println("Stochastic signal, valid embedding NOT achieved ⨉.")
    elseif dimensionmethod=="ifnn"
        dimension_statistic = estimate_dimension(s, τ, γs)
        Y, τ = fnn_embed(s, τ, dimension_statistic, thres)
    elseif dimensionmethod=="fnn"
        dimension_statistic = estimate_dimension(s, τ, γs, dimensionmethod; rtol = rtol, atol = atol)
        Y, τ = fnn_embed(s, τ, dimension_statistic, thres)
    elseif dimensionmethod=="f1nn"
        dimension_statistic = estimate_dimension(s, τ, γs, dimensionmethod)
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
