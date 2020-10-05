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

    τ = estimate_delay(s, delay, τs; kwargs...)
    # Code here that automatically does cao/hegger stuff

end


"""
    standard_embedding_hegger(s::Vector; kwargs...) → `Y`, `τ`
Compute the reconstructed trajectory from a time series using the standard time
delay embedding. The delay `τ` is taken as the 1st minimum of the mutual
information [`estimate_dimension`](@ref) and the embedding dimension `m` is
estimated by using an FNN method from [^Hegger1999] [`ifnn`](@ref).
Return the reconstructed trajectory `Y` and the delay `τ`.

Keyword arguments:

*`fnn_thres = 0.05`: a threshold defining at which fraction of FNNs the search
    should break.
* The `method` can be one of the following:
* `"ac_zero"` : first delay at which the auto-correlation function becomes <0.
* `"ac_min"` : delay of first minimum of the auto-correlation function.
* `"mi_min"` : delay of first minimum of mutual information of `s` with itself
  (shifted for various `τs`). <- Default

[^Hegger1999]: Hegger and Kantz (1999). [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
function standard_embedding_hegger(s::Vector{T}; method::String = "mi_min",
                                            fnn_thres::Real = 0.05) where {T}
    @assert method=="ac_zero" || method=="mi_min" || method=="ac_min"
    "The absolute correlation function has elements that are = 0. "*
    "We can't fit an exponential to it. Please choose another method."

    τ = estimate_delay(s, method)
    _, _, Y = ifnn(s, τ; fnn_thres = fnn_thres)
    return Y, τ
end


"""
    standard_embedding_cao(s::Vector; kwargs...) → `Y`, `τ`
Compute the reconstructed trajectory from a time series using the standard time
delay embedding. The delay `τ` is taken as the 1st minimum of the mutual
information [`estimate_dimension`](@ref) and the embedding dimension `m` is
estimated by using an FNN method from Cao [`estimate_dimension`](@ref), with the
threshold parameter `cao_thres`.
Return the reconstructed trajectory `Y` and the delay `τ`.

Keyword arguments:
*`cao_thres = 0.05`: This threshold determines the tolerable deviation of the
    proposed statistic from the optimal value of 1, for breaking the algorithm.
*`m_max = 10`: The maximum embedding dimension, which is encountered by the
    algorithm.
* The `method` can be one of the following:
* `"ac_zero"` : first delay at which the auto-correlation function becomes <0.
* `"ac_min"` : delay of first minimum of the auto-correlation function.
* `"mi_min"` : delay of first minimum of mutual information of `s` with itself
  (shifted for various `τs`). <- Default

[^Hegger1999]: Hegger and Kantz (1999). [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
function standard_embedding_cao(s::Vector{T}; cao_thres::Real = 0.05,
                        method::String = "mi_min", m_max::Int = 10) where {T}
    @assert method=="ac_zero" || method=="mi_min" || method=="ac_min"
    "The absolute correlation function has elements that are = 0. "*
    "We can't fit an exponential to it. Please choose another method."

    τ = estimate_delay(s, method)
    rat = estimate_dimension(s, τ, 1:m_max, "afnn")
    m, Y = 0, nothing
    for i = 1:m_max
        if abs(1-rat[i]) < cao_thres
            m = i
            break
        end
    end
    Y =  m > 1 ? embed(s, m, τ) : s
    return Y, τ
end
