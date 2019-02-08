using DelayEmbeddings, OrdinaryDiffEq

ti = time()

const diffeq = (alg = Vern9(), atol = 1e-9, rtol = 1e-9, maxiters = typemax(Int))

include("dataset_tests.jl")
include("reconstruction_tests.jl")
include("R_delay.jl")
include("R_dimension.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits=3), " seconds or ", round(ti/60, digits=3), " minutes")
