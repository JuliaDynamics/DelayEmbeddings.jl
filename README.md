# DelayEmbeddings.jl

![DynamicalSystems.jl logo: The Double Pendulum](https://i.imgur.com/nFQFdB0.gif)

| **Documentation**   |  **Travis**     | **AppVeyor** | Gitter |
|:--------:|:-------------------:|:-----:|:-----:|
|[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaDynamics.github.io/DynamicalSystems.jl/latest) | [![Build Status](https://travis-ci.org/JuliaDynamics/DelayEmbeddings.jl.svg?branch=master)](https://travis-ci.org/JuliaDynamics/DelayEmbeddings.jl) | [![Build status](https://ci.appveyor.com/api/projects/status/1vstt1c39gv8e4sl/branch/master?svg=true)](https://ci.appveyor.com/project/JuliaDynamics/delayembeddings-jl/branch/master) | [![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/JuliaDynamics/Lobby)


This repo is a minimal package used throughout JuliaDynamics. The features are:

1. Defines the `Dataset` structure, which is a wrapper of `Vector{SVector}`, used in **DynamicalSystems.jl**.
2. Implements methods for delay coordinates embedding (Takens' theorem) with high performance and many features.
3. Algorithms for estimating optimal delay embedding parameters, the delay time and the number of temporal neighbors (generalization of the "embedding dimension").
3. Provides a unified `neighborhood` function that works across different kinds of nearest neighbor searching packages.
