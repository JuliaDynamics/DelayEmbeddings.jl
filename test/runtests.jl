using DelayEmbeddings
using StaticArrays
using Test
import Downloads

# Download some test timeseries
tsfolder = joinpath(@__DIR__, "timeseries")
todownload1 = ["$n.csv" for n in 1:4]
todownload = ["test_time_series_lorenz_standard_N_10000_multivariate.csv", "test_time_series_roessler_N_10000_multivariate.csv"]
append!(todownload, todownload1)
repo = "https://raw.githubusercontent.com/JuliaDynamics/JuliaDynamics/master/timeseries"
mkpath(tsfolder)
for a in todownload
    Downloads.download(repo*"/"*a, joinpath(tsfolder, a))
end

ti = time()

diffeq = (atol = 1e-9, rtol = 1e-9, maxiters = typemax(Int))

@testset "DelayEmbeddings tests" begin
    include("dataset_tests.jl")
    include("embedding_tests.jl")
    include("traditional/delaytime_test.jl")
    include("traditional/embedding_dimension_test.jl")
    include("unified/test_pecora.jl")
    include("unified/uzal_cost_test.jl")
    include("unified/test_pecuzal_embedding.jl")

    # The following methods have been tested throughly and also published in research.
    # We know that they work, but unfortunately the tests we have written
    # about them are not good.
    # See https://github.com/JuliaDynamics/DelayEmbeddings.jl/issues/95
    # include("unified/mdop_tests.jl")
    # include("unified/test_garcia.jl")

end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, sigdigits=3), " seconds or ", round(ti/60, sigdigits=3), " minutes")
