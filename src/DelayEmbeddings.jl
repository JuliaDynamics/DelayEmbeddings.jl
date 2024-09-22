module DelayEmbeddings

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end DelayEmbeddings

using Reexport
@reexport using StateSpaceSets

# for backwards compatibility with SSSets.jl v1
# TODO: Cleanup: go through the instances in the codebase
# where `oldsize` is called: replace it with `size` if the
# input is a `Matrix`, and with `dimension` if the input is a `StateSpaceSet`.
oldsize(s::AbstractStateSpaceSet) = (length(s), dimension(s))
oldsize(s) = size(s)
oldsize(s, i::Int) = oldsize(s)[i]

include("embeddings/embed.jl")
include("embeddings/genembed.jl")
include("utils.jl")
include("separated_de/estimate_delay.jl")
include("separated_de/estimate_dimension.jl")
include("separated_de/automated.jl")

include("unified_de/pecora.jl")
include("unified_de/uzal_cost.jl")
include("unified_de/MDOP.jl")
include("unified_de/garcia_almeida.jl")
include("unified_de/pecuzal.jl")

include("deprecate.jl")

end
