using NearestNeighbors

export estimate_dimension, stochastic_indicator
#####################################################################################
#                                Estimate Dimension                                 #
#####################################################################################
"""
    estimate_dimension(s::AbstractVector, τ:Int, γs = 1:5) -> E₁s

Compute a quantity that can estimate an optimal amount of
temporal neighbors `γ` to be used in [`reconstruct`](@ref) or [`embed`](@ref).

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute the
values of `E₁` for each `γ ∈ γs`, according to Cao's Method (eq. 3 of [1]).
Please be aware that in **DynamicalSystems.jl** `γ` stands for the amount of temporal
neighbors and not the embedding dimension (`D = γ + 1`, see also [`embed`](@ref)).

Return the vector of all computed `E₁`s. To estimate a good value for `γ` from this,
find `γ` for which the value `E₁` saturates at some value around 1.

*Note: This method does not work for datasets with perfectly periodic signals.*

## References

[1] : Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)
"""
function estimate_dimension(s::AbstractVector{T}, τ::Int, γs = 1:5) where {T}
    E1s = zeros(T, length(γs))
    aafter = zero(T)
    aprev = _average_a(s, γs[1], τ)
    for (i, γ) ∈ enumerate(γs)
        aafter = _average_a(s, γ+1, τ)
        E1s[i] = aafter/aprev
        aprev = aafter
    end
    return E1s
end
# then use function `saturation_point(γs, E1s)` from ChaosTools

function _average_a(s::AbstractVector{T},γ,τ) where T
    #Sum over all a(i,d) of the Ddim Reconstructed space, equation (2)
    R1 = reconstruct(s,γ+1,τ)
    R2 = reconstruct(s[1:end-τ],γ,τ)
    tree2 = KDTree(R2)
    nind = (x = knn(tree2, R2.data, 2)[1]; [ind[1] for ind in x])
    e=0.
    for (i,j) in enumerate(nind)
        δ = norm(R2[i]-R2[j], Inf)
        #If R2[i] and R2[j] are still identical, choose the next nearest neighbor
        if δ == 0.
            j = knn(tree2, R2[i], 3, true)[1][end]
            δ = norm(R2[i]-R2[j], Inf)
        end
        e += norm(R1[i]-R1[j], Inf) / δ
    end
    return e / length(R1)
end

function dimension_indicator(s,γ,τ) #this is E1, equation (3) of Cao
    return _average_a(s,γ+1,τ)/_average_a(s,γ,τ)
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
        R1 = reconstruct(s,γ+1,τ)
        tree1 = KDTree(R1[1:end-1-τ])
        method = FixedMassNeighborhood(2)

        Es1 = 0.
        nind = (x = neighborhood(R1[1:end-τ], tree1, method); [ind[1] for ind in x])
        for  (i,j) ∈ enumerate(nind)
            Es1 += abs(R1[i+τ][end] - R1[j+τ][end]) / length(R1)
        end

        #Calculate E* for Dimension γ
        R2 = reconstruct(s,γ,τ)
        tree2 = KDTree(R2[1:end-1-τ])
        Es2 = 0.
        nind = (x = neighborhood(R2[1:end-τ], tree2, method); [ind[1] for ind in x])
        for  (i,j) ∈ enumerate(nind)
            Es2 += abs(R2[i+τ][end] - R2[j+τ][end]) / length(R2)
        end
        push!(E2s, Es1/Es2)
    end
    return E2s
end
