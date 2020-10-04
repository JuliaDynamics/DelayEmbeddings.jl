
"""
    standard_embedding_hegger(s::Vector; kwargs...) → `Y`, `τ`
Compute the reconstructed trajectory from a time series using the standard time
delay embedding. The delay `τ` is taken as the 1st minimum of the mutual
information [`estimate_dimension`](@ref) and the embedding dimension `m` is
estimated by using an FNN method from [^Hegger1999] [`fnn_uniform_hegger`](@ref).
Return the reconstructed trajectory `Y` and the delay `τ`.

Keyword arguments:

*`fnn_thres = 0.05`: a threshold defining at which fraction of FNNs the search
    should break.
* The `method` can be one of the following:
* `"ac_zero"` : first delay at which the auto-correlation function becomes <0.
* `"ac_min"` : delay of first minimum of the auto-correlation function.
* `"mi_min"` : delay of first minimum of mutual information of `s` with itself
  (shifted for various `τs`). <- Default

[^Hegger1999]: Hegger, Rainer and Kantz, Holger (1999). [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
function standard_embedding_hegger(s::Vector{T}; method::String = "mi_min",
                                            fnn_thres::Real = 0.05) where {T}
    @assert method=="ac_zero" || method=="mi_min" || method=="ac_min"
    "The absolute correlation function has elements that are = 0. "*
    "We can't fit an exponential to it. Please choose another method."

    τ = estimate_delay(s, method)
    _, _, Y = fnn_uniform_hegger(s, τ; fnn_thres = fnn_thres)
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

[^Hegger1999]: Hegger, Rainer and Kantz, Holger (1999). [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
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
