# DelayEmbeddings.jl
This repo is a minimal package used throughout JuliaDynamics. The features are:

1. Defines the `Dataset` structure, which is a wrapper of `Vector{SVector}`, used in **DynamicalSystems.jl**.
2. Implements methods for delay coordinates embedding (Taken's theorem) with high performance and many features.
3. Algorithms for estimating optimal delay embedding parameters, the delay time and the number of temporal neighbors (generalization of the "embedding dimension").
3. Provides a unified `neighborhood` function that works across different kinds of nearest neighbor searching packages.
