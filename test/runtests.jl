using DelayEmbeddings

ti = time()

const diffeq = (atol = 1e-9, rtol = 1e-9, maxiters = typemax(Int))

include("dataset_tests.jl")
include("embedding_tests.jl")
include("delaytime_test.jl")
include("delaycount_test.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits=3), " seconds or ", round(ti/60, digits=3), " minutes")
