module DelayEmbeddings

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end DelayEmbeddings

using Reexport
@reexport using StateSpaceSets

include("embeddings/embed.jl")
include("embeddings/genembed.jl")
include("utils.jl")
include("traditional_de/estimate_delay.jl")
include("traditional_de/estimate_dimension.jl")
include("traditional_de/automated.jl")

include("unified_de/pecora.jl")
include("unified_de/uzal_cost.jl")
include("unified_de/MDOP.jl")
include("unified_de/garcia_almeida.jl")
include("unified_de/pecuzal.jl")

include("deprecate.jl")

end
