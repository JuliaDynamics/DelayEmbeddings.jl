# DelayEmbeddings.jl

```@docs
DelayEmbeddings
```

## Overview

!!! note
    The documentation and the code of this package is parallelizing Chapter 6 of [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7), Datseris & Parlitz, Springer 2022.


The package provides an interface to perform delay coordinates embeddings, as explained in [homonymous page](@ref embedding).

There are two approaches for estimating optimal parameters to do delay embeddings:
1. **Separated**, where one tries to find the best value for a delay time `τ` and then an optimal embedding dimension `d`.
2. **Unified**, where at the same time an optimal combination of `τ, d` is found.

The separated approach is something "old school", while recent scientific research has shifted almost exclusively to unified approaches. This page describes algorithms belonging to the separated approach, which is mainly done by the function [`optimal_separated_de`](@ref).

The unified approach is discussed in the [Unified optimal embedding](@ref) page.