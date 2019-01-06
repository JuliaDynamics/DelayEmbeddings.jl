using DelayEmbeddings

ti = time()

include("dataset_tests.jl")
include("reconstruction_tests.jl")
include("R_delay.jl")
include("R_dimension.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits=3), " seconds or ", round(ti/60, digits=3), " minutes")
