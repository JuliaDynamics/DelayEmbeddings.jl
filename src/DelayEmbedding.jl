"""
Delay coordinates embedding, `Dataset` structure and neibhborhoods.
Basic package used in the ecosystem of DynamicalSystems.jl.
"""
module DelayEmbedding

include("dataset.jl")
include("reconstruction.jl")
include("various.jl")
include("neighborhoods.jl")

end
