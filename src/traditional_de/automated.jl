export optimal_traditional_de

"""
    optimal_traditional_de(s, dmethod = "mi_min", method = "afnn"; kwargs...) → 𝒟, τ, x

Produce an optimal delay embedding `𝒟` of the given timeseries `s` by
using the traditional approach of first finding an optimal (and constant) delay
time using [`estimate_delay`](@ref) with the given `dmethod`, and then an optimal
embedding dimension. Return the embedding `𝒟` and the optimal delay time `τ`
(the optimal embedding dimension `d` is just `size(𝒟, 2)`) and the actual
statistic `x` used to estimate optimal `d`.

For estimating the dimension we use the given `method`, which can be:

* `"afnn"` (default) is Cao's "Averaged False Nearest Neighbors" method[^Cao1997],
    which gives a ratio of distances between nearest neighbors.
* `"ifnn"` is the "Improved False Nearest Neighbors" from Hegger & Kantz[^Hegger1999],
    which gives the fraction of false nearest neighbors.
* `"fnn"` is Kennel's "False Nearest Neighbors" method[^Kennel1992], which gives
    the number of points that cease to be "nearest neighbors" when the dimension
    increases.
* `"f1nn"` is Krakovská's "False First Nearest Neighbors" method[^Krakovská2015],
    which gives the ratio of pairs of points that cease to be "nearest neighbors"
    when the dimension increases. This is the worse method.

For more details, see individual methods: [`afnn`](@ref), [`ifnn`](@ref),
[`fnn`](@ref), [`f1nn`](@ref).

## Keywords
All keywords are propagated to the low level functions like `afnn` (except `τs`).
```
fnn_thres::Real = 0.05, slope_thres::Real= 0.2, dmax::Int = 10, w::Int=1,
rtol=10.0, atol=2.0, τs = 1:100, metric = Euclidean(), r::Real=2.0
```

## Description
We estimate the optimal embedding dimension based on the given delay time gained
from `dmethod` as follows: For Cao's method the optimal dimension is reached,
when the slope of the `E₁`-statistic (output from `"afnn"`) falls below the
threshold `slope_thres` (Default is .05) and the according stochastic test turns
out to be false, i.e. when the `E₂`-statistic is not "equal" to 1 for all en-
countered dimensions. We treat `E₂`-values as equal to 1, when `1-E₂ ≤ fnn_thres`.
For all the other methods we return the optimal embedding dimension
when the corresponding FNN-statistic (output from `"ifnn"`, `"fnn"` or `"f1nn"`)
falls below the fnn-threshold `fnn_thres` (Default is .05) AND the slope of the
statistic falls below the threshold `slope_thres`. Note that with noise
contaminated time series, one might need to adjust `fnn_thres` according to the
noise level.

[^Cao1997]: Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)

[^Kennel1992]: M. Kennel *et al.*, [Phys. Review A **45**(6), (1992)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.3403).

[^Krakovská2015]: Anna Krakovská *et al.*, [J. Complex Sys. 932750 (2015)](https://doi.org/10.1155/2015/932750)

[^Hegger1999]: Hegger & Kantz, [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
function optimal_traditional_de(s::AbstractVector, delaymethod::String= "mi_min",
        dimensionmethod::String = "afnn";
        fnn_thres::Real = 0.05, slope_thres::Real = .05, dmax::Int = 10, w::Int=1,
        rtol=10.0, atol=2.0, τs = 1:100, metric = Euclidean(), r::Real=2.0
    )

    @assert dimensionmethod ∈ ("afnn", "fnn", "ifnn", "f1nn")
    τ = estimate_delay(s, delaymethod, τs)
    γs = 0:dmax-1 # TODO: This must be updated to dimension in 2.0

    if dimensionmethod=="afnn"
        dimension_statistic = afnn(s, τ, γs, metric)
        Y, τ = cao_embed(s, τ, dimension_statistic, slope_thres)
        E2 = stochastic_indicator(s, τ, γs)
        flag = is_stochastic(E2, fnn_thres)
        flag && println("Stochastic signal, valid embedding NOT achieved ⨉.")
    elseif dimensionmethod=="fnn"
        dimension_statistic = fnn(s, τ, γs; rtol, atol)
        Y, τ = fnn_embed(s, τ, dimension_statistic, fnn_thres, slope_thres)
    elseif dimensionmethod=="ifnn"
        dimension_statistic = ifnn(s, τ, γs; r, w, metric)
        Y, τ = fnn_embed(s, τ, dimension_statistic, fnn_thres, slope_thres)
    elseif dimensionmethod=="f1nn"
        dimension_statistic = f1nn(s, τ, γs, metric)
        Y, τ = fnn_embed(s, τ, dimension_statistic, fnn_thres, slope_thres)
    end
    return Y, τ, dimension_statistic
end


"""
Helper function for selecting the appropriate embedding dimension from the
statistic when using Kennel's, Hegger's or Krakovskas's method.
"""
function fnn_embed(s::Vector{T}, τ::Int, rat::Vector, fnn_thres::Real,
                                                    slope_thres::Real) where {T}
    @assert length(rat) > 1
    y = abs.(diff(rat))
    flag = false
    m, Y = 0, nothing
    for i = 2:length(rat)
        if rat[i] ≤ fnn_thres && y[i-1] ≤ slope_thres
            m = i
            break
        elseif rat[i] > rat[i-1]
            flag = true
            m = i-1
            break
        end
    end
    if m == length(rat) || m == 0
        Y = s
        println("Sufficiently small FNNs NOT reached."*
                "Valid embedding NOT achieved ⨉.")
    else
        Y =  m > 1 ? embed(s, m, τ) : s
        println("Algorithm stopped due to sufficiently small FNNs. "*
                "Valid embedding achieved ✓.")
        if flag
            println("Algorithm stopped due to increasing FNNs. "*
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
    y = abs.(diff(rat))
    for i = 1:length(rat)-1
        if y[i] ≤ thres
            m = i
            break
        end
    end
    if m == 0
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
        if abs(1-rat[i]) ≥ thres
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
