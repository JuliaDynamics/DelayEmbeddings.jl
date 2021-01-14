"""
Delay coordinates embedding, `Dataset` structure and neighhborhoods.
Basic package used in the ecosystem of DynamicalSystems.jl.
"""
module DelayEmbeddings

include("dataset.jl")
include("subdataset.jl")
include("embeddings.jl")
include("utils.jl")
include("neighborhoods.jl")

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
