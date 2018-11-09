using NearestNeighbors
using Distances: chebyshev
using SpecialFunctions: digamma
using StatsBase: autocor

#####################################################################################
#                                Mutual Information                                 #
#####################################################################################
"""
    mutinfo(k, X1, X2[, ..., Xm]) -> MI

Calculate the mutual information `MI` of the given vectors
`X1, X2, ...`, using `k` nearest-neighbors.

The method follows the second algorithm ``I^{(2)}`` outlined by Kraskov in [1].

## References
[1] : A. Kraskov *et al.*, [Phys. Rev. E **69**, pp 066138 (2004)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138)

## Performance Notes
This functin gets very slow for large `k`.

See also [`estimate_delay`](@ref) and [`mutinfo_delaycurve`](@ref).
"""
function mutinfo(k, Xm::Vararg{<:AbstractVector,M}) where M
    @assert M > 1
    @assert (size.(Xm,1) .== size(Xm[1],1)) |> prod
    k += 1
    N = size(Xm[1],1)
    invN = 1/N

    d = Dataset(Xm...)
    tree = KDTree(d.data, Chebyshev())

    n_x_m = zeros(M)

    Xm_sp = zeros(Int, N, M)
    Xm_revsp = zeros(Int, N, M)
    for m in 1:M
        Xm_sp[:,m] .= sortperm(Xm[m]; alg=QuickSort)
        Xm_revsp[:,m] .= sortperm(Xm_sp[:,m]; alg=QuickSort)
    end

    I = digamma(k) - (M-1)/k + (M-1)*digamma(N)

    nns = (x = knn(tree, d.data, k)[1]; [ind[1] for ind in x])

    I_itr = zeros(M)
    # Makes more sense computationally to loop over N rather than M
    for i in 1:N
        ϵ = abs.(d[nns[i]] - d[i])./2

        for m in 1:M # this loop takes 8% of time
            hb = lb = Xm_revsp[i,m]
            while abs(Xm[m][Xm_sp[hb,m]] - Xm[m][i]) <= ϵ[m] && hb < N
                hb += 1
            end
            while abs(Xm[m][Xm_sp[lb,m]] - Xm[m][i]) <= ϵ[m] && lb > 1
                lb -= 1
            end
            n_x_m[m] = hb - lb
        end

        I_itr .+= digamma.(n_x_m)
    end

    I_itr .*= invN

    I -= sum(I_itr)

    return max(0, I)
end

"""
    mutinfo_delaycurve(x; maxtau=100, k=1)

Return the [`mutinfo`](@ref) between `x` and itself for delays of `1:maxtau`.
"""
function mutinfo_delaycurve(X::AbstractVector; maxtau=100, k=1)
    I = zeros(maxtau)

    @views for τ in 1:maxtau
        I[τ] = mutinfo(k, X[1:end-τ],X[τ+1:end])
    end

    return I
end
