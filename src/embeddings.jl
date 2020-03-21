using StaticArrays
using Base: @_inline_meta
export reconstruct, DelayEmbedding, MTDelayEmbedding, embed, τrange
export WeightedDelayEmbedding

#####################################################################################
# Univariate Delay Coordinates
#####################################################################################
"""
    AbstractEmbedding
Super-type of embedding methods.
"""
abstract type AbstractEmbedding end

"""
    DelayEmbedding(γ, τ) → `embedding`
Return a delay coordinates embedding structure to be used as a functor,
given a timeseries and some index. Calling
```julia
embedding(s, n)
```
will create the `n`-th delay vector of the embedded space, which has `γ`
temporal neighbors with delay(s) `τ`. See [`reconstruct`](@ref) for more.

**Be very careful when choosing `n`, because `@inbounds` is used internally.
It must be that `n ≤ length(s) - maximum(τ)`.**

Convience function [`τrange`](@ref) gives all valid `n` indices.
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

# Weighted version
export WeightedDelayEmbedding
"""
    WeightedDelayEmbedding(γ, τ, w) → `embedding`
Similar with [`DelayEmbedding`](@ref), but the entries of the
embedded vector are further weighted with `w^γ`.
See [`reconstruct`](@ref) for more.

**Be very careful when choosing `n`, because `@inbounds` is used internally.
It must be that `n ≤ length(s) - maximum(τ)`.**

Convience function [`τrange`](@ref) gives all valid `n` indices.
"""
struct WeightedDelayEmbedding{γ, T<:Real} <: AbstractEmbedding
    delays::SVector{γ, Int}
    w::T
end

@inline WeightedDelayEmbedding(γ, τ, w) = WeightedDelayEmbedding(Val{γ}(), τ, w)
@inline function WeightedDelayEmbedding(::Val{γ}, τ::Int, w::T) where {γ, T}
    idxs = [k*τ for k in 1:γ]
    return WeightedDelayEmbedding{γ, T}(SVector{γ, Int}(idxs...), w)
end

@generated function (r::WeightedDelayEmbedding{γ, T})(s::AbstractArray{X}, i) where {γ, T, X}
    gens = [:(r.w^($k) * s[i + r.delays[$k]]) for k=1:γ]
    quote
        @_inline_meta
        @inbounds return SVector{$γ+1,X}(s[i], $(gens...))
    end
end

"""
    reconstruct(s, γ, τ [, w])
Reconstruct `s` using the delay coordinates embedding with `γ` temporal neighbors
and delay `τ` and return the result as a [`Dataset`](@ref). Optionally use weight `w`.

Use [`embed`](@ref) for the version that accepts the embedding dimension `D = γ+1`
instead. Here `τ` is strictly positive, use [`genembed`](@ref) for a generalized
version.

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

If `w` (a "weight") is provided as an extra argument, then the entries
of the embedded vector are further weighted with ``w^\\gamma``, like so
```math
(s(n), w*s(n+\\tau), w^2*s(n+2\\tau), \\dots,w^\\gamma * s(n+γ\\tau))
```

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
function reconstruct(s::AbstractVector{T}, γ, τ, w) where {T}
    if γ == 0
        return Dataset{1, T}(s)
    end
    de = WeightedDelayEmbedding(Val{γ}(), τ, w)
    return reconstruct(s, de)
end
@inline function reconstruct(s::AbstractVector{T},
    de::Union{WeightedDelayEmbedding{γ}, DelayEmbedding{γ}}) where {T, γ}
    r = τrange(s, de)
    data = Vector{SVector{γ+1, T}}(undef, length(r))
    @inbounds for i in r
        data[i] = de(s, i)
    end
    return Dataset{γ+1, T}(data)
end

"""
    τrange(s, de::AbstractEmbedding)
Return the range `r` of valid indices `n` to create delay vectors out of `s` using `de`.
"""
τrange(s, de::AbstractEmbedding) = 1:(length(s) - maximum(de.delays))

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
# Multiple timeseries
#####################################################################################
"""
    MTDelayEmbedding(γ, τ, B) -> `embedding`
Return a delay coordinates embedding structure to be used as a functor,
given multiple timeseries (`B` in total), either as a [`Dataset`](@ref) or a
`SizedArray`, and some index.
Calling
```julia
embedding(s, n)
```
will create the `n`-th delay vector of the embedded space, which has `γ`
temporal neighbors with delay(s) `τ`. See [`reconstruct`](@ref) for more.

**Be very careful when choosing `n`, because `@inbounds` is used internally.
It must be that `n ≤ length(s) - maximum(τ)`.**

Convience function [`τrange`](@ref) gives all valid `n` indices.
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

    if γ == 0
        return Dataset{B, T}(s)
    end
    de::MTDelayEmbedding{γ, B, γ*B} = MTDelayEmbedding(γ, τ, B)
    reconstruct(s, de)
end
@inline function reconstruct(
    s::Union{AbstractDataset{B, T}, SizedArray{Tuple{A, B}, T, 2, M}},
    de::MTDelayEmbedding{γ, B, F}) where {A, B, T, M, γ, F}

    r = τrange(s, de)
    X = (γ+1)*B
    data = Vector{SVector{X, T}}(undef, length(r))
    @inbounds for i in r
        data[i] = de(s, i)
    end
    return Dataset{X, T}(data)
end

reconstruct(s::AbstractMatrix, args...) = reconstruct(Dataset(s), args...)

#####################################################################################
# Generalized embedding (arbitrary combination of timeseries and delays)
#####################################################################################
export GeneralizedEmbedding, genembed

"""
    GeneralizedEmbedding(τs, js) -> `embedding`
Return a delay coordinates embedding structure to be used as a functor.
Given a timeseries *or* trajectory (i.e. `Dataset`) `s` and calling
```julia
embedding(s, n)
```
will create the `n`-th delay vector of `s` in the embedded space using
`generalized` embedding (see [`genembed`](@ref).

`js` is ignored for timeseries input `s` (since all entries of `js` must be `1` in
this case).

**Be very careful when choosing `n`, because `@inbounds` is used internally.
It must be that `minimum(τs) + 1 ≤ n ≤ length(s) - maximum(τs)`.
In addition please ensure that all entries of `js` are valid dimensions of `s`.**

Convience function [`τrange`](@ref) gives all valid `n` indices.
"""
struct GeneralizedEmbedding{D} <: AbstractEmbedding
    τs::NTuple{D, Int}
    js::NTuple{D, Int}
end

function Base.show(io::IO, g::GeneralizedEmbedding{D}) where {D}
    print(io, "$D-dimensional generalized embedding\n")
    print(io, "  τs: $(g.τs)\n")
    print(io, "  js: $(g.js)")
end

# timeseries input
@generated function (g::GeneralizedEmbedding{D})(s::AbstractArray{T}, i::Int) where {D, T}
    gens = [:(s[i + g.τs[$k]]) for k=1:D]
    quote
        @_inline_meta
        @inbounds return SVector{$D,T}($(gens...))
    end
end

# dataset input
@generated function (g::GeneralizedEmbedding{D})(s::Dataset{X, T}, i::Int) where {D, X, T}
    gens = [:(s[i + g.τs[$k], g.js[$k]]) for k=1:D]
    quote
        @_inline_meta
        @inbounds return SVector{$D,T}($(gens...))
    end
end

τrange(s, ge::GeneralizedEmbedding) =
max(1, (-minimum(ge.τs) + 1)):min(length(s), length(s) - maximum(ge.τs))


"""
    genembed(s, τs, js = ones(...)) → dataset
Create a generalized embedding of `s` which can be a timeseries or arbitrary `Dataset`
and return the result as a new `dataset`.

The generalized embedding works as follows:
- `τs::NTuple{D, Int}` denotes what delay times will be used for each of the entries
  of the delay vector. It is strongly recommended that `τs[1] = 0`.
  `τs` is allowed to have *negative entries* as well.
- `js::NTuple{D, Int}` denotes which of the timeseries contained in `s`
  will be used for the entries of the delay vector. `js` can contain duplicate indices.

For example, imagine input trajectory ``s = [x, y, z]`` where ``x, y, z`` are timeseries
(the columns of the `Dataset`).
If `js = (1, 3, 2)` and `τs = (0, 2, -7)` the created delay vector at
each step ``n`` will be
```math
(x(n), z(n+2), y(n-7))
```

`js` can be skipped, defaulting to index 1 (first timeseries) for all delay entries.

See also [`reconstruct`](@ref). Internally uses [`GeneralizedEmbedding`](@ref).
"""
function genembed(s, τs::NTuple{D, Int}, js::NTuple{D, Int}) where {D}
    ge::GeneralizedEmbedding{D} = GeneralizedEmbedding(τs, js)
    r = τrange(s, ge)
    T = eltype(s)
    data = Vector{SVector{D, T}}(undef, length(r))
    @inbounds for (i, n) in enumerate(r)
        data[i] = ge(s, n)
    end
    return Dataset{D, T}(data)
end

genembed(s, τs::NTuple{D, Int}) where {D} = genembed(s, τs, NTuple{D, Int}(ones(D)))
