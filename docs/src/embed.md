# [Delay coordinates embedding](@id embedding)

A timeseries recorded in some manner from a dynamical system can be used to gain information about the dynamics of the entire state space of the system. This can be done by constructing a new state space from the timeseries. One method that can do this is what is known as [delay coordinates embedding](https://en.wikipedia.org/wiki/Takens%27_theorem) or delay coordinates reconstruction.

The main functions to use for embedding some input data are [`embed`](@ref) or [`genembed`](@ref). Both functions return a [`StateSpaceSet`](@ref).

## Timeseries embedding

```@docs
embed
```

!!! note "Embedding discretized data values"
    If the data values are very strongly discretized (e.g., integers or floating-point numbers with very small bits), this can result to distances between points in the embedded space being 0. This is problematic for several library functions. Best practice here is to add noise to your original timeseries _before_ embedding, e.g., `s = s .+ 1e-15randn(length(s))`.

---

Here are some examples of embedding a 3D continuous chaotic system:
```@example MAIN
using DelayEmbeddings

x = cos.(0:0.1:1)
```

```@example MAIN
embed(x, 3, 1)
```

!!! note "`τ` and `Δt`"
    Keep in mind that whether a value of `τ` is "reasonable" for continuous time systems depends on the sampling time `Δt`.

### Embedding Structs
The high level function [`embed`](@ref) utilizes a low-level interface for creating embedded vectors on-the-fly. The high level interface simply loops over the low level interface.
```@docs
DelayEmbedding
τrange
```

## Generalized embeddings
```@docs
genembed
GeneralizedEmbedding
```

## StateSpaceSet reference
```@docs
StateSpaceSet
```