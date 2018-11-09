"""
Delay coordinates embedding, `Dataset` structure and neighhborhoods.
Basic package used in the ecosystem of DynamicalSystems.jl.
"""
module DelayEmbeddings

include("dataset.jl")
include("reconstruction.jl")
include("various.jl")
include("neighborhoods.jl")

include("estimate_delay.jl")
include("estimate_dimension.jl")

end
