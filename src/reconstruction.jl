using StaticArrays
using Base: @_inline_meta
export reconstruct, DelayEmbedding, AbstractEmbedding, MTDelayEmbedding, embed

#####################################################################################
#                        Delay Embedding Reconstruction                             #
#####################################################################################
"""
    AbstractEmbedding
Super-type of embedding methods. Use `subtypes(AbstractEmbedding)` for available
methods.
"""
abstract type AbstractEmbedding <: Function end

"""
    DelayEmbedding(γ, τ) -> `embedding`
Return a delay coordinates embedding structure to be used as a functor,
given a timeseries and some index. Calling
```julia
embedding(s, n)
```
will create the `n`-th reconstructed vector of the embedded space, which has `γ`
temporal neighbors with delay(s) `τ`. See [`reconstruct`](@ref) for more.

*Be very careful when choosing `n`, because `@inbounds` is used internally.*
"""
struct DelayEmbedding{γ} <: AbstractEmbedding
    delays::SVector{γ, Int}
end

@inline DelayEmbedding(γ, τ) = DelayEmbedding(Val{γ}(), τ)
@inline function DelayEmbedding(::Val{γ}, τ::Int) where {γ}
    idxs = [k*τ for k in 1:γ]
    return DelayEmbedding{γ}(SVector{γ, Int}(idxs...))
end
@inline function DelayEmbedding(::Val{γ}, τ::AbstractVector) where {γ}
    γ != length(τ) && throw(ArgumentError(
    "Delay time vector length must equal the number of temporal neighbors."
    ))
    return DelayEmbedding{γ}(SVector{γ, Int}(τ...))
end

@generated function (r::DelayEmbedding{γ})(s::AbstractArray{T}, i) where {γ, T}
    gens = [:(s[i + r.delays[$k]]) for k=1:γ]
    quote
        @_inline_meta
        @inbounds return SVector{$γ+1,T}(s[i], $(gens...))
    end
end

"""
    reconstruct(s, γ, τ)
Reconstruct `s` using the delay coordinates embedding with `γ` temporal neighbors
and delay `τ` and return the result as a [`Dataset`](@ref).

See [`embed`](@ref) for the version that accepts the embedding dimension `D = γ+1`
directly.

## Description
### Single Timeseries
If `τ` is an integer, then the ``n``-th entry of the embedded space is
```math
(s(n), s(n+\\tau), s(n+2\\tau), \\dots, s(n+γ\\tau))
```
If instead `τ` is a vector of integers, so that `length(τ) == γ`,
then the ``n``-th entry is
```math
(s(n), s(n+\\tau[1]), s(n+\\tau[2]), \\dots, s(n+\\tau[γ]))
```

The reconstructed dataset can have same
invariant quantities (like e.g. lyapunov exponents) with the original system
that the timeseries were recorded from, for proper `γ` and `τ`.
This is known as the Takens embedding theorem [1, 2].
The case of different delay times allows reconstructing systems with many time scales,
see [3].

*Notice* - The dimension of the returned dataset (i.e. embedding dimension) is `γ+1`!

### Multiple Timeseries
To make a reconstruction out of a multiple timeseries (i.e. trajectory) the number
of timeseries must be known by type, so `s` can be either:

* `s::AbstractDataset{B}`
* `s::SizedAray{A, B}`

If the trajectory is for example ``(x, y)`` and `τ` is integer, then the ``n``-th
entry of the embedded space is
```math
(x(n), y(n), x(n+\\tau), y(n+\\tau), \\dots, x(n+γ\\tau), y(n+γ\\tau))
```
If `τ` is an `AbstractMatrix{Int}`, so that `size(τ) == (γ, B)`,
then we have
```math
(x(n), y(n), x(n+\\tau[1, 1]), y(n+\\tau[1, 2]), \\dots, x(n+\\tau[γ, 1]), y(n+\\tau[γ, 2]))
```

*Notice* - The dimension of the returned dataset is `(γ+1)*B`!

## References
[1] : F. Takens, *Detecting Strange Attractors in Turbulence — Dynamical
Systems and Turbulence*, Lecture Notes in Mathematics **366**, Springer (1981)

[2] : T. Sauer *et al.*, J. Stat. Phys. **65**, pp 579 (1991)

[3] : K. Judd & A. Mees, [Physica D **120**, pp 273 (1998)](https://www.sciencedirect.com/science/article/pii/S0167278997001188)
"""
function reconstruct(s::AbstractVector{T}, γ, τ) where {T}
    if γ == 0
        return Dataset{1, T}(s)
    end
    de::DelayEmbedding{γ} = DelayEmbedding(Val{γ}(), τ)
    return reconstruct(s, de)
end
@inline function reconstruct(s::AbstractVector{T}, de::DelayEmbedding{γ}) where {T, γ}
    L = length(s) - maximum(de.delays)
    data = Vector{SVector{γ+1, T}}(undef, L)
    @inbounds for i in 1:L
        data[i] = de(s, i)
    end
    return Dataset{γ+1, T}(data)
end

"""
    embed(s, D, τ)
Perform a delay coordinates embedding on signal `s` with embedding dimension `D`
and delay time `τ`. The result is returned as a [`Dataset`](@ref), which is a
vector of static vectors.

See [`reconstruct`](@ref) for an advanced version that supports multiple delay
times and can reconstruct multiple timeseries efficiently.
"""
embed(s, D, τ) = reconstruct(s, D-1, τ)


#####################################################################################
#                              MultiDimensional R                                   #
#####################################################################################
"""
    MTDelayEmbedding(γ, τ, B) -> `embedding`
Return a delay coordinates embedding structure to be used as a functor,
given multiple timeseries (`B` in total), either as a [`Dataset`](@ref) or a
`SizedArray` (see [`reconstruct`](@ref)), and some index.
Calling
```julia
embedding(s, n)
```
will create the `n`-th reconstructed vector of the embedded space, which has `γ`
temporal neighbors with delay(s) `τ`. See [`reconstruct`](@ref) for more.

*Be very careful when choosing `n`, because `@inbounds` is used internally.*
"""
struct MTDelayEmbedding{γ, B, X} <: AbstractEmbedding
    delays::SMatrix{γ, B, Int, X} # X = γ*B = total dimension number
end

@inline MTDelayEmbedding(γ, τ, B) = MTDelayEmbedding(Val{γ}(), τ, Val{B}())
@inline function MTDelayEmbedding(::Val{γ}, τ::Int, ::Val{B}) where {γ, B}
    X = γ*B
    idxs = SMatrix{γ,B,Int,X}([k*τ for k in 1:γ, j in 1:B])
    return MTDelayEmbedding{γ, B, X}(idxs)
end
@inline function MTDelayEmbedding(
    ::Val{γ}, τ::AbstractMatrix{<:Integer}, ::Val{B}) where {γ, B}
    X = γ*B
    γ != size(τ)[1] && throw(ArgumentError(
    "`size(τ)[1]` must equal the number of spatial neighbors."
    ))
    B != size(τ)[2] && throw(ArgumentError(
    "`size(τ)[2]` must equal the number of timeseries."
    ))
    return MTDelayEmbedding{γ, B, X}(SMatrix{γ, B, Int, X}(τ))
end
function MTDelayEmbedding(
    ::Val{γ}, τ::AbstractVector{<:Integer}, ::Val{B}) where {γ, B}
    error("Does not work with vector τ, only matrix or integer!")
end

@generated function (r::MTDelayEmbedding{γ, B, X})(
    s::Union{AbstractDataset{B, T}, SizedArray{Tuple{A, B}, T, 2, M}},
    i) where {γ, A, B, T, M, X}
    gensprev = [:(s[i, $d]) for d=1:B]
    gens = [:(s[i + r.delays[$k, $d], $d]) for k=1:γ for d=1:B]
    quote
        @_inline_meta
        @inbounds return SVector{$(γ+1)*$B,T}($(gensprev...), $(gens...))
    end
end

@inline function reconstruct(
    s::Union{AbstractDataset{B, T}, SizedArray{Tuple{A, B}, T, 2, M}},
    γ, τ) where {A, B, T, M}

    de::MTDelayEmbedding{γ, B, γ*B} = MTDelayEmbedding(γ, τ, B)
    reconstruct(s, de)
end
@inline function reconstruct(
    s::Union{AbstractDataset{B, T}, SizedArray{Tuple{A, B}, T, 2, M}},
    de::MTDelayEmbedding{γ, B, F}) where {A, B, T, M, γ, F}

    if length(de.delays) == 0
        return Dataset(s)
    else
        L = size(s)[1] - maximum(de.delays)
        X = (γ+1)*B
        data = Vector{SVector{X, T}}(undef, L)
        @inbounds for i in 1:L
            data[i] = de(s, i)
        end
        return Dataset{X, T}(data)
    end
end
