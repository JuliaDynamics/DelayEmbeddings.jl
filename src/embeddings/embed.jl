using Base: @_inline_meta
export DelayEmbedding, embed, τrange
export AbstractEmbedding

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
@generated function (r::DelayEmbedding{γ})(s::AbstractArray{T}, i) where {γ, T}
    gens = [:(r.h[$k]*(s[i + r.delays[$k]])) for k=1:γ]
    quote
        @_inline_meta
        @inbounds return SVector{$γ+1,T}(s[i], $(gens...))
    end
end


"""
    embed(s, d, τ [, h])

Embed `s` using delay coordinates with embedding dimension `d` and delay time `τ`
and return the result as a [`StateSpaceSet`](@ref). Optionally use weight `h`, see below.

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
invariant quantities (like e.g. Lyapunov exponents) with the original system
that the timeseries were recorded from, for proper `d` and `τ`.
This is known as the Takens embedding theorem [^Takens1981] [^Sauer1991].
The case of different delay times allows embedding systems with many time scales,
see[^Judd1998].

If provided, `h` can be weights to multiply the entries of the embedded space.
If `h isa Real` then the embedding is
```math
(s(n), h \\cdot s(n+\\tau), h^2 \\cdot s(n+2\\tau), \\dots,h^{d-1} \\cdot s(n+γ\\tau))
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
        return StateSpaceSet(s)
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
    return StateSpaceSet{γ+1, T}(data)
end

"""
    τrange(s, de::AbstractEmbedding)
Return the range `r` of valid indices `n` to create delay vectors out of `s` using `de`.
"""
τrange(s, de::AbstractEmbedding) = 1:(length(s) - maximum(de.delays))
