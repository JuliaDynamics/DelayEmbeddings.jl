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
    optimal value of `γ` (see [`f1nn`](@ref)).

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

    if method == "afnn"
        return afnn(s, τ, γs, metric)
    elseif method == "fnn"
        return fnn(s, τ, γs; kwargs...)
    elseif method == "f1nn"
        return f1nn(s, τ, γs, metric)
    end
end


"""
    afnn(s::AbstractVector, τ:Int, γs = 1:5, metric=Euclidean())

Compute the parameter E₁ of Cao's "averaged false nearest neighbors" method for
determining the minimum embedding dimension of the time series `s`, with
a sequence of `τ`-delayed temporal neighbors.

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute the
values of `E₁` for each `γ ∈ γs`, according to Cao's Method (eq. 3 of [1]).

This quantity is a ratio of the averaged distances between the nearest neighbors
of the reconstructed time series, which quantifies the increment of those
distances when the number of temporal neighbors changes from `γ` to `γ+1`.

Return the vector of all computed `E₁`s. To estimate a good value for `γ` from this,
find `γ` for which the value `E₁` saturates at some value around 1.

*Note: This method does not work for datasets with perfectly periodic signals.*

See also: [`estimate_dimension`](@ref), [`fnn`](@ref), [`f1nn`](@ref).
"""
function afnn(s::AbstractVector{T}, τ::Int, γs = 1:5, metric=Euclidean()) where {T}
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

# TODO: This algorithm needs to be re-written on Neighborhood.jl.
function _average_a(s::AbstractVector{T},γ,τ,metric) where {T}
    #Sum over all a(i,d) of the Ddim Reconstructed space, equation (2)
    Rγ = reconstruct(s[1:end-τ],γ,τ)
    tree2 = KDTree(Rγ, metric)
    nind = (x = NearestNeighbors.knn(tree2, Rγ.data, 2)[1]; [ind[1] for ind in x])
    e = 0.0
    @inbounds for (i,j) ∈ enumerate(nind)
        δ = evaluate(metric, Rγ[i], Rγ[j])
        #If Rγ[i] and Rγ[j] are still identical, choose the next nearest neighbor
        if δ == 0.0
            j = NearestNeighbors.knn(tree2, Rγ[i], 3, true)[1][end]
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
of the distance between the nearest neighbors (`rtol`, eq. 4 in[^Kennel1992]), and
another threshold for the ratio between the increased distance and the
"size of the attractor" (`atol`, eq. 5 in[^Kennel1992]). These thresholds are given
as keyword arguments.

The returned value is a vector with the number of FNN for each `γ ∈ γs`. The
optimal value for `γ` is found at the point where the number of FNN approaches
zero.

See also: [`estimate_dimension`](@ref), [`afnn`](@ref), [`f1nn`](@ref).
"""
function fnn(s::AbstractVector, τ::Int, γs = 1:5; rtol=10.0, atol=2.0)
    rtol2 = rtol^2
    Ra = std(s, corrected=false)
    nfnn = zeros(length(γs))
    @inbounds for (k, γ) ∈ enumerate(γs)
        y = reconstruct(s[1:end-τ],γ,τ)
        tree = KDTree(y)
        nind = (x = NearestNeighbors.knn(tree, y.data, 2)[1]; [ind[1] for ind in x])
        for (i,j) ∈ enumerate(nind)
            δ = norm(y[i]-y[j], 2)
            # If y[i] and y[j] are still identical, choose the next nearest neighbor
            # as in Cao's algorithm (not suggested by Kennel, but still advisable)
            if δ == 0.0
                j = NearestNeighbors.knn(tree, y[i], 3, true)[1][end]
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
the "false first nearest neighbors" (FFNN) are the pairs of points that are nearest to
each other at dimension `γ` that cease to be nearest neighbors at dimension
`γ+1`.

The returned value is a vector with the ratio between the number of FFNN and
the number of points in the dataset for each `γ ∈ γs`. The optimal value for `γ`
is found at the point where this ratio approaches zero.

See also: [`estimate_dimension`](@ref), [`afnn`](@ref), [`fnn`](@ref).
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
    nind = (x = NearestNeighbors.knn(tree, Rγ.data, 2)[1]; [ind[1] for ind in x])
    @inbounds for  (i,j) ∈ enumerate(nind)
        k = NearestNeighbors.knn(tree1, Rγ1.data[i], 2, true)[1][end]
        if j != k
            nf1nn += 1
        end
    end
    # `Rγ1` is returned to re-use it if necessary
    return (nf1nn, Rγ1)
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
[^Hegger1999]: Hegger, Rainer and Kantz, Holger (1999). [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
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


"""
    stochastic_indicator(s::AbstractVector, τ:Int, γs = 1:4) -> E₂s

Compute an estimator for apparent randomness in a reconstruction with `γs` temporal
neighbors.

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute the
values of `E₂` for each `γ ∈ γs`, according to Cao's Method (eq. 5 of [^Cao1997]).

Use this function to confirm that the
input signal is not random and validate the results of [`estimate_dimension`](@ref).
In the case of random signals, it should be `E₂ ≈ 1 ∀ γ`.
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
    for i = 1:m_max
        if abs(1-rat[i]) < cao_thres
            global m = i
            break
        end
    end
    try
        if m > 1
            global Y = embed(s, m, τ)
        else
            global Y = s
        end
    catch
        global Y = s
    end
    return Y, τ
end


"""
    fnn_uniform_hegger(s::Vector, τ::Int; kwargs...) →  `m`, `FNNs`, `Y`
Compute and return the optimal embedding dimension `m` for the time series `s`
and a uniform time delay `τ` after [^Hegger1999]. Return the optimal `m` and the
corresponding reconstruction vector `Y` according to that `m` and the input `τ`.
The optimal `m` is chosen, when the fraction of `FNNs` falls below the threshold
`fnn_thres` or when fraction of FNN's increases.

Keyword argument:
*`fnn_thres = 0.05`: Threshold, which defines the tolerable fraction of FNN's
    for which the algorithm breaks.
*`max_dimension = 10`: The maximum dimension which is encountered by the
    algorithm and after which it breaks, if the breaking criterion has not been
    met yet.
*`r = 2`: Obligatory threshold, which determines the maximum tolerable spreading
    of trajectories in the reconstruction space.
*`metric = Euclidean`: The norm used for distance computations.
*`w = 1` = The Theiler window, which excludes temporally correlated points from
    the nearest neighbor search.

[^Hegger1999]: Hegger, Rainer and Kantz, Holger (1999). [Improved false nearest neighbor method to detect determinism in time series data. Physical Review E 60, 4970](https://doi.org/10.1103/PhysRevE.60.4970).
"""
function fnn_uniform_hegger(s::Vector{T}, τ::Int; max_dimension::Int = 10,
            r::Real = 2, w::Int = 1, fnn_thres::Real = 0.05, metric = Euclidean()) where {T}
    @assert max_dimension > 0
    s = (s .- mean(s)) ./ std(s)
    Y_act = s

    vtree = KDTree(Dataset(s), metric)
    _, NNdist_old = DelayEmbeddings.all_neighbors(vtree, Dataset(s), 1:length(s), 1, w)

    FNNs = zeros(max_dimension)
    for m = 2:max_dimension+1
        Y_act = DelayEmbeddings.hcat_lagged_values(Y_act, s, m*τ)
        Y_act = regularize(Y_act)
        vtree = KDTree(Y_act, metric)
        _, NNdist_new = DelayEmbeddings.all_neighbors(vtree, Y_act, 1:length(Y_act), 1, w)

        FNNs[m-1] = DelayEmbeddings.fnn_embedding_cycle(view(NNdist_old,
                                            1:length(Y_act)), NNdist_new, r)

        flag = fnn_break_criterion(FNNs[1:m-1], fnn_thres)
        if flag
            global bm = m
            break
        else
            global bm = m
        end

        NNdist_old = NNdist_new
    end

    if bm>2
        Y_final = embed(s, bm-1, τ)
    else
        Y_final = s
    end
    return bm, FNNs[1:bm-1], Y_final
end

"""
Determines the break criterion for the Hegger-FNN-estimation
"""
function fnn_break_criterion(FNNs, fnn_thres)
    flag = false
    if FNNs[end] ≤ fnn_thres
        flag = true
        println("Algorithm stopped due to sufficiently small FNNs. "*
                "Valid embedding achieved ✓.")
    end
    if length(FNNs) > 1 && FNNs[end] > FNNs[end-1]
        flag = true
        println("Algorithm stopped due to rising FNNs. "*
                "Valid embedding achieved ✓.")
    end
    return flag
end
