#####################################################################################
# Generalized embedding (arbitrary combination of timeseries and delays)
#####################################################################################
export GeneralizedEmbedding, genembed

"""
    GeneralizedEmbedding(τs, js = ones(length(τs)), ws = nothing) -> `embedding`
Return a delay coordinates embedding structure to be used as a function.
Given a timeseries *or* trajectory (i.e. `StateSpaceSet`) `s` and calling
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
struct GeneralizedEmbedding{D, W} <: AbstractEmbedding
    τs::NTuple{D, Int}
    js::NTuple{D, Int}
    ws::NTuple{D, W}
end
function GeneralizedEmbedding(τs::NTuple{D, Int}) where {D} # type stable version
    GeneralizedEmbedding{D, Nothing}(
        τs, NTuple{D, Int}(ones(D)), NTuple{D, Nothing}(fill(nothing, D))
    )
end

function GeneralizedEmbedding(τs, js = ones(length(τs)), ws = nothing)
    D = length(τs)
    a = NTuple{D, Int}(τs)
    b = NTuple{D, Int}(js)
    W = isnothing(ws) ? Nothing : typeof(promote(ws...)[1])
    c = isnothing(ws) ? NTuple{D, W}(fill(nothing, D)) : NTuple{D, W}(promote(ws...))
    return GeneralizedEmbedding{D, W}(a, b, c)
end

function Base.show(io::IO, g::GeneralizedEmbedding{D, W}) where {D, W}
    print(io, "$D-dimensional generalized embedding\n")
    print(io, "  τs: $(g.τs)\n")
    print(io, "  js: $(g.js)\n")
    wp = W == Nothing ? "nothing" : g.ws
    print(io, "  ws: $(wp)")
end

const Data{T} = Union{StateSpaceSet{D, T}, AbstractVector{T}} where {D}

# dataset version
@generated function (g::GeneralizedEmbedding{D, W})(s::AbstractStateSpaceSet{Z, T}, i::Int) where {D,W,Z,T}
    gens = if W == Nothing
        [:(s[i + g.τs[$k], g.js[$k]]) for k=1:D]
    else
        [:(g.ws[$k]*s[i + g.τs[$k], g.js[$k]]) for k=1:D]
    end
    X = W == Nothing ? T : promote_type(T, W)
    quote
        @_inline_meta
        @inbounds return SVector{$D,$X}($(gens...))
    end
end

# vector version
@generated function (g::GeneralizedEmbedding{D, W})(s::AbstractVector{T}, i::Int) where {D,W,T}
    gens = if W == Nothing
        [:(s[i + g.τs[$k]]) for k=1:D]
    else
        [:(g.ws[$k]*s[i + g.τs[$k]]) for k=1:D]
    end
    X = W == Nothing ? T : promote_type(T, W)
    quote
        @_inline_meta
        @inbounds return SVector{$D,$X}($(gens...))
    end
end

τrange(s, ge::GeneralizedEmbedding) =
max(1, (-minimum(ge.τs) + 1)):min(length(s), length(s) - maximum(ge.τs))


"""
    genembed(s, τs, js = ones(...); ws = nothing) → dataset
Create a generalized embedding of `s` which can be a timeseries or arbitrary `StateSpaceSet`,
and return the result as a new `StateSpaceSet`.

The generalized embedding works as follows:
- `τs` denotes what delay times will be used for each of the entries
  of the delay vector. It is recommended that `τs[1] = 0`.
  `τs` is allowed to have *negative entries* as well.
- `js` denotes which of the timeseries contained in `s`
  will be used for the entries of the delay vector. `js` can contain duplicate indices.
- `ws` are optional weights that weight each embedded entry (the i-th entry of the
    delay vector is weighted by `ws[i]`). If provided, it is recommended that `ws[1] == 1`.

`τs, js, ws` are tuples (or vectors) of length `D`, which also coincides with the embedding
dimension. For example, imagine input trajectory ``s = [x, y, z]`` where ``x, y, z`` are
timeseries (the columns of the `StateSpaceSet`).
If `js = (1, 3, 2)` and `τs = (0, 2, -7)` the created delay vector at
each step ``n`` will be
```math
(x(n), z(n+2), y(n-7))
```
Using `ws = (1, 0.5, 0.25)` as well would create
```math
(x(n), \\frac{1}{2} z(n+2), \\frac{1}{4} y(n-7))
```

`js` can be skipped, defaulting to index 1 (first timeseries) for all delay entries, while
it has no effect if `s` is a timeseries instead of a `StateSpaceSet`.

See also [`embed`](@ref). Internally uses [`GeneralizedEmbedding`](@ref).
"""
function genembed(s, τs, js = ones(length(τs)); ws = nothing)
    D = length(τs)
    W = isnothing(ws) ? Nothing : typeof(promote(ws...)[1])
    ge::GeneralizedEmbedding{D, W} = GeneralizedEmbedding(τs, js, ws)
    return genembed(s, ge)
end

function genembed(s, ge::GeneralizedEmbedding{D, W}) where {D, W}
    r = τrange(s, ge)
    T = eltype(s)
    X = W == Nothing ? T : promote_type(T, W)
    data = Vector{SVector{D, X}}(undef, length(r))
    @inbounds for (i, n) in enumerate(r)
        data[i] = ge(s, n)
    end
    return StateSpaceSet{D, X}(data)
end
