export optimal_traditional_de

"""
    optimal_traditional_de(s, method = "afnn", dmethod = "mi_min"; kwargs...) → 𝒟, τ, E

Produce an optimal delay embedding `𝒟` of the given timeseries `s` by
using the traditional approach of first finding an optimal (and constant) delay
time using [`estimate_delay`](@ref) with the given `dmethod`, and then an optimal
embedding dimension, by calculating an appropriate statistic for each dimension `d ∈ 1:dmax`.
Return the embedding `𝒟`, the optimal delay time `τ`
(the optimal embedding dimension `d` is just `size(𝒟, 2)`) and the actual
statistic `E` used to estimate optimal `d`.

Notice that `E` is a function of the embedding dimension, which ranges from 1 to `dmax`.

For calculating `E` to estimate the dimension we use the given `method` which can be:

* `"afnn"` (default) is Cao's "Averaged False Nearest Neighbors" method[^Cao1997],
    which gives a ratio of distances between nearest neighbors.
* `"ifnn"` is the "Improved False Nearest Neighbors" from Hegger & Kantz[^Hegger1999],
    which gives the fraction of false nearest neighbors.
* `"fnn"` is Kennel's "False Nearest Neighbors" method[^Kennel1992], which gives
    the number of points that cease to be "nearest neighbors" when the dimension
    increases.
* `"f1nn"` is Krakovská's "False First Nearest Neighbors" method[^Krakovská2015],
    which gives the ratio of pairs of points that cease to be "nearest neighbors"
    when the dimension increases.

For more details, see individual methods: [`delay_afnn`](@ref), [`delay_ifnn`](@ref),
[`delay_fnn`](@ref), [`delay_f1nn`](@ref).

!!! warn "Careful in automated methods"
    While this method is automated if you want to be **really sure** of the results,
    you should directly calculate the statistic and plot its values versus the
    dimensions.

## Keyword Arguments
The keywords
```julia
τs = 1:100, dmax = 10
```
denote which delay times and embedding dimensions `ds ∈ 1:dmax` to consider when calculating
optimal embedding. The keywords 
```julia
slope_thres = 0.05, stoch_thres = 0.1, fnn_thres = 0.05
```
are specific to this function, see Description below.
All remaining keywords are propagated to the low level functions:
```
w, rtol, atol, τs, metric, r
```

## Description
We estimate the optimal embedding dimension based on the given delay time gained
from `dmethod` as follows: For Cao's method the optimal dimension is reached,
when the slope of the `E₁`-statistic (output from `"afnn"`) falls below the
threshold `slope_thres` and the according stochastic test turns
out to be false, i.e. if the `E₂`-statistic's first value is `< 1 - stoch_thres`.

For all the other methods we return the optimal embedding dimension
when the corresponding FNN-statistic (output from `"ifnn"`, `"fnn"` or `"f1nn"`)
falls below the fnn-threshold `fnn_thres` AND the slope of the
statistic falls below the threshold `slope_thres`. Note that with noise
contaminated time series, one might need to adjust `fnn_thres` according to the
noise level.

See also the file `test/compare_different_dimension_estimations.jl` for a comparison.

[^Cao1997]: Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)

[^Kennel1992]: M. Kennel *et al.*, [Phys. Review A **45**(6), (1992)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.3403).

[^Krakovská2015]: Anna Krakovská *et al.*, [J. Complex Sys. 932750 (2015)](https://doi.org/10.1155/2015/932750)

[^Hegger1999]: Hegger & Kantz, [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
function optimal_traditional_de(s::AbstractVector, dimensionmethod::String = "afnn",
        delaymethod::String= "mi_min";
        fnn_thres::Real = 0.05, slope_thres::Real = .05, dmax::Int = 10, w::Int=1,
        rtol=10.0, atol=2.0, τs = 1:100, metric = Euclidean(), r::Real=2.0,
        stoch_thres = 0.1,
    )

    @assert dimensionmethod ∈ ("afnn", "fnn", "ifnn", "f1nn")
    τ = estimate_delay(s, delaymethod, τs)
    ds = 1:dmax
    γs = ds .- 1 # TODO: This must be updated to dimension in 2.0

    if dimensionmethod=="afnn"
        dimension_statistic = delay_afnn(s, τ, ds; metric, w)
        Y, τ = find_cao_optimal(s, τ, dimension_statistic, slope_thres)
        flag = is_stochastic(s, τ, 1:3, stoch_thres)
        flag && println("Stochastic signal, valid embedding NOT achieved ⨉.")
    elseif dimensionmethod=="fnn"
        dimension_statistic = delay_fnn(s, τ, ds; rtol, atol)
        Y, τ = find_fnn_optimal(s, τ, dimension_statistic, fnn_thres, slope_thres)
    elseif dimensionmethod=="ifnn"
        dimension_statistic = delay_ifnn(s, τ, ds; r, w, metric)
        Y, τ = find_fnn_optimal(s, τ, dimension_statistic, fnn_thres, slope_thres)
    elseif dimensionmethod=="f1nn"
        dimension_statistic = delay_f1nn(s, τ, ds; metric)
        Y, τ = find_fnn_optimal(s, τ, dimension_statistic, fnn_thres, slope_thres)
    end
    return Y, τ, dimension_statistic
end


"""
Helper function for selecting the appropriate embedding dimension from the
statistic when using Kennel's, Hegger's or Krakovskas's method.
"""
function find_fnn_optimal(s::AbstractVector{T}, τ::Int, rat::Vector, fnn_thres::Real,
                                                    slope_thres::Real) where {T}
    @assert length(rat) > 1
    y = abs.(diff(rat))
    flag = false
    m = 0
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
        println("Sufficiently small FNNs NOT reached. "*
                "Valid embedding NOT achieved ⨉.")
    elseif flag
        println("Algorithm stopped due to increasing FNNs. "*
        "Double-check the FNN-statistic.")
    else
        println("Algorithm stopped due to sufficiently small FNNs. "*
                "Valid embedding achieved ✓.")
    end
    Y = embed(s, max(1, m), τ) # you can embed in 1 dimension in latest version
    return Y, τ
end


"""
Helper function for selecting the appropriate embedding dimension from the
statistic when using Cao's method.
"""
function find_cao_optimal(s::AbstractVector{T}, τ::Int, rat::Vector, thres::Real) where {T}
    m = 0
    y = abs.(diff(rat))
    for i = 1:length(rat)-1
        if y[i] ≤ thres && rat[i] > 0.5
            m = i
            break
        end
    end
    if m == 0
        println("NO convergence of E₁-statistic. "*
                "Valid embedding NOT achieved ⨉.")
    else
        Y =  m > 1 ? embed(s, m, τ) : s
        println("Algorithm stopped due to convergence of E₁-statistic. "*
                "Valid embedding achieved ✓.")
    end
    Y = embed(s, max(1, m), τ) # you can embed in 1 dimension in latest version
    return Y, τ
end

"""
Helper function for Cao's method, whether to decide if the input signal is
stochastic or not.
"""
function is_stochastic(s, τ, ds, stoch_thres) where {T}
    E2 = stochastic_indicator(s, τ, ds)
    E2[1] > 1 - stoch_thres
end
