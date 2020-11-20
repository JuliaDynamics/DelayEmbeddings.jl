using StaticArrays
using Base: @_inline_meta
export reconstruct, DelayEmbedding, MTDelayEmbedding, embed, τrange
export WeightedDelayEmbedding, AbstractEmbedding

#####################################################################################
# Univariate Delay Coordinates
#####################################################################################
"Super-type of embedding methods."
abstract type AbstractEmbedding end

"""
    DelayEmbedding(γ, τ, h = nothing) → `embedding`
Return a delay coordinates embedding structure to be used as a function-like-object,
given a timeseries and some index. Calling
```julia
embedding(s, n)
```
will create the `n`-th delay vector of the embedded space, which has `γ`
temporal neighbors with delay(s) `τ`.
`γ` is the embedding dimension minus 1, `τ` is the delay time(s) while `h` are
extra weights, as in [`embed`](@ref) for more.

**Be very careful when choosing `n`, because `@inbounds` is used internally.**
Use [`τrange`](@ref)!
"""
struct DelayEmbedding{γ, H} <: AbstractEmbedding
    delays::SVector{γ, Int}
    h::SVector{γ, H}
end
@inline DelayEmbedding(γ, τ, h = nothing) = DelayEmbedding(Val{γ}(), τ, h)
@inline function DelayEmbedding(g::Val{γ}, τ::Int, h::H) where {γ, H}
    idxs = [k*τ for k in 1:γ]
    hs = hweights(g, h)
    htype = H <: Union{Nothing, Real} ? H : eltype(H)
    return DelayEmbedding{γ, htype}(SVector{γ, Int}(idxs...), hs)
end
@inline function DelayEmbedding(g::Val{γ}, τ::AbstractVector, h::H) where {γ, H}
    γ != length(τ) && throw(ArgumentError(
        "Delay time vector length must equal the embedding dimension minus 1."
    ))
    hs = hweights(g, h)
    htype = H <: Union{Nothing, Real} ? H : eltype(H)
    return DelayEmbedding{γ, htype}(SVector{γ, Int}(τ...), hs)
end
hweights(::Val{γ}, h::Nothing) where {γ} = SVector{γ, Nothing}(fill(nothing, γ))
hweights(::Val{γ}, h::T) where {γ, T<:Real} = SVector{γ, T}([h^b for b in 1:γ]...)
hweights(::Val{γ}, h::AbstractVector) where {γ} = SVector{γ}(h)

@generated function (r::DelayEmbedding{γ, Nothing})(s::AbstractArray{T}, i) where {γ, T}
    gens = [:(s[i + r.delays[$k]]) for k=1:γ]
    quote
        @_inline_meta
        @inbounds return SVector{$γ+1,T}(s[i], $(gens...))
    end
end

# This is the version with weights
@generated function (r::DelayEmbedding{γ})(s::AbstractArray{T}, i) where {γ, T, R<:Real}
    gens = [:(r.h[$k]*(s[i + r.delays[$k]])) for k=1:γ]
    quote
        @_inline_meta
        @inbounds return SVector{$γ+1,T}(s[i], $(gens...))
    end
end

function WeightedDelayEmbedding(γ, τ, w)
    @warn "`WeightedDelayEmbedding` is deprecated in favor of `DelayEmbedding`."
    return DelayEmbedding(γ, τ, w)
end

"""
    embed(s, d, τ [, h])
Embed `s` using delay coordinates with embedding dimension `d` and delay time `τ`
and return the result as a [`Dataset`](@ref). Optionally use weight `h`, see below.

Here `τ > 0`, use [`genembed`](@ref) for a generalized version.

## Description
If `τ` is an integer, then the ``n``-th entry of the embedded space is
```math
(s(n), s(n+\\tau), s(n+2\\tau), \\dots, s(n+(d-1)\\tau))
```
If instead `τ` is a vector of integers, so that `length(τ) == d-1`,
then the ``n``-th entry is
```math
(s(n), s(n+\\tau[1]), s(n+\\tau[2]), \\dots, s(n+\\tau[d-1]))
```

The resulting set can have same
invariant quantities (like e.g. lyapunov exponents) with the original system
that the timeseries were recorded from, for proper `d` and `τ`.
This is known as the Takens embedding theorem [^Takens1981] [^Sauer1991].
The case of different delay times allows embedding systems with many time scales,
see[^Judd1998].

If provided, `h` can be weights to multiply the entries of the embedded space.
If `h isa Real` then the embedding is
```math
(s(n), h \\cdot s(n+\\tau), w^2 \\cdot s(n+2\\tau), \\dots,w^{d-1} \\cdot s(n+γ\\tau))
```
Otherwise `h` can be a vector of length `d-1`, which the decides the weights of each
entry directly.

## References
[^Takens1981] : F. Takens, *Detecting Strange Attractors in Turbulence — Dynamical
Systems and Turbulence*, Lecture Notes in Mathematics **366**, Springer (1981)

[^Sauer1991] : T. Sauer *et al.*, J. Stat. Phys. **65**, pp 579 (1991)

[^Judd1998]: K. Judd & A. Mees, [Physica D **120**, pp 273 (1998)](https://www.sciencedirect.com/science/article/pii/S0167278997001188)

[^Farmer1988]: Farmer & Sidorowich, [Exploiting Chaos to Predict the Future and Reduce Noise"](http://www.nzdl.org/gsdlmod?e=d-00000-00---off-0cltbibZz-e--00-1----0-10-0---0---0direct-10---4-------0-1l--11-en-50---20-home---00-3-1-00-0--4--0--0-0-11-10-0utfZz-8-00&a=d&cl=CL3.16&d=HASH013b29ffe107dba1e52f1a0c_1245)

"""
function embed(s::AbstractVector{T}, d, τ, h::H = nothing) where {T, H}
    if d == 1
        return Dataset(s)
    end
    htype = H <: Union{Nothing, Real} ? H : eltype(H)
    de::DelayEmbedding{d-1, htype} = DelayEmbedding(d-1, τ, h)
    return embed(s, de)
end

@inline function embed(s::AbstractVector{T}, de::DelayEmbedding{γ}) where {T, γ}
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
    reconstruct(s, γ, τ) = embed(s, γ+1, τ)
"""
function reconstruct(s, γ, τ)
    @warn "`reconstruct` is deprecated in favor of `embed`. In general, the old argument "*
    "`γs` is being phased out throughout the library."
    return embed(s, γ+1, τ)
end


#####################################################################################
# Multiple timeseries
#####################################################################################
"""
    MTDelayEmbedding(γ, τ, B) -> `embedding`
Return a delay coordinates embedding structure to be used as a functor,
that embeds multiple timeseries (`B` in total) given in the form of a [`Dataset`](@ref).

Calling
```julia
embedding(s, n)
```
where `s` is a `Dataset` will create the `n`-th delay vector of the embedded space,
which has `γ` temporal neighbors with delay(s) `τ`. See [`reconstruct`](@ref) for more.

**Be very careful when choosing `n`, because `@inbounds` is used internally.**
Use [`τrange`](@ref)!
"""
struct MTDelayEmbedding{γ, B, X} <: AbstractEmbedding
    delays::SMatrix{γ, B, Int, X} # X = γ*B = total dimension number
end

@inline MTDelayEmbedding(γ, τ, B) = MTDelayEmbedding(Val{γ}(), τ, Val{B}())
@inline function MTDelayEmbedding(::Val{γ}, τ::Int, ::Val{B}) where {γ, B}
    @warn "Multi-timeseries delay embedding via `reconstruct` is deprecated in favor of "*
    "using `genembed` directly."
    X = γ*B
    idxs = SMatrix{γ,B,Int,X}([k*τ for k in 1:γ, j in 1:B])
    return MTDelayEmbedding{γ, B, X}(idxs)
end
@inline function MTDelayEmbedding(
    ::Val{γ}, τ::AbstractMatrix{<:Integer}, ::Val{B}) where {γ, B}
    @warn "Multi-timeseries delay embedding via `reconstruct` is deprecated in favor of "*
    "using `genembed` directly."
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
    @warn "Multi-timeseries delay embedding via `reconstruct` is deprecated in favor of "*
    "using `genembed` directly."
    typeof(s) <: SizedArray && @warn "Using SizedArrays is deprecated. Use Dataset instead."
    gensprev = [:(s[i, $d]) for d=1:B]
    gens = [:(s[i + r.delays[$k, $d], $d]) for k=1:γ for d=1:B]
    quote
        @_inline_meta
        @inbounds return SVector{$(γ+1)*$B,T}($(gensprev...), $(gens...))
    end
end

τrange(s, de::MTDelayEmbedding) = 1:(size(s)[1] - maximum(de.delays))

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
    GeneralizedEmbedding(τs [, js]) -> `embedding`
Return a delay coordinates embedding structure to be used as a functor.
Given a timeseries *or* trajectory (i.e. `Dataset`) `s` and calling
```julia
embedding(s, n)
```
will create the delay vector of the `n`-th point of `s` in the embedded space using
generalized embedding (see [`genembed`](@ref)).

`js` is ignored for timeseries input `s` (since all entries of `js` must be `1` in
this case) and in addition `js` defaults to `(1, ..., 1)` for all `τ`.

**Be very careful when choosing `n`, because `@inbounds` is used internally.**
Use [`τrange`](@ref)!
"""
struct GeneralizedEmbedding{D} <: AbstractEmbedding
    τs::NTuple{D, Int}
    js::NTuple{D, Int}
end
GeneralizedEmbedding(τs::NTuple{D, Int}) where {D} =
GeneralizedEmbedding{D}(τs, NTuple{D, Int}(ones(D)))

function GeneralizedEmbedding(τs, js::AbstractVector = ones(length(τs)))
    D = length(τs)
    a = NTuple{D, Int}(τs)
    b = NTuple{D, Int}(js)
    return GeneralizedEmbedding{D}(a, b)
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
Create a generalized embedding of `s` which can be a timeseries or arbitrary `Dataset`,
and return the result as a new `dataset`.

The generalized embedding works as follows:
- `τs` denotes what delay times will be used for each of the entries
  of the delay vector. It is recommended that `τs[1] = 0`.
  `τs` is allowed to have *negative entries* as well.
- `js` denotes which of the timeseries contained in `s`
  will be used for the entries of the delay vector. `js` can contain duplicate indices.

`τs, js` are tuples (or vectors) of length `D`, which also coincides with the embedding
dimension. For example, imagine input trajectory ``s = [x, y, z]`` where ``x, y, z`` are
timeseries (the columns of the `Dataset`).
If `js = (1, 3, 2)` and `τs = (0, 2, -7)` the created delay vector at
each step ``n`` will be
```math
(x(n), z(n+2), y(n-7))
```

`js` can be skipped, defaulting to index 1 (first timeseries) for all delay entries.

See also [`embed`](@ref). Internally uses [`GeneralizedEmbedding`](@ref).
"""
function genembed(s, τs, js = ones(length(τs)))
    D = length(τs)
    ge::GeneralizedEmbedding{D} = GeneralizedEmbedding(τs, js)
    return genembed(s, ge)
end

function genembed(s, ge::GeneralizedEmbedding{D}) where {D}
    r = τrange(s, ge)
    T = eltype(s)
    data = Vector{SVector{D, T}}(undef, length(r))
    @inbounds for (i, n) in enumerate(r)
        data[i] = ge(s, n)
    end
    return Dataset{D, T}(data)
end
