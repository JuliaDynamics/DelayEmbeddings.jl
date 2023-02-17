using DelayEmbeddings
using Test

function testfile(file, testname=defaultname(file))
    println("running test file $(file)")
    @testset "$testname" begin; include(file); end
    return
end
defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))

# Download some test timeseries
import Downloads
tsfolder = joinpath(@__DIR__, "timeseries")
todownload1 = ["$n.csv" for n in 1:4]
todownload = ["test_time_series_lorenz_standard_N_10000_multivariate.csv", "test_time_series_roessler_N_10000_multivariate.csv"]
append!(todownload, todownload1)
repo = "https://raw.githubusercontent.com/JuliaDynamics/JuliaDynamics/master/timeseries"
mkpath(tsfolder)
for a in todownload
    Downloads.download(repo*"/"*a, joinpath(tsfolder, a))
end

@testset "DelayEmbeddings tests" begin
    testfile("embedding_tests.jl")
    testfile("traditional/delaytime_test.jl")
    testfile("traditional/embedding_dimension_test.jl")
    # TODO: All of these tests need to be re-written to be "good" tests,
    # and not just test the output the functions have had in the past in some
    # pre-existing data.
    # include("unified/test_pecora.jl")
    # include("unified/uzal_cost_test.jl")
    # include("unified/test_pecuzal_embedding.jl")
    # include("unified/test_garcia.jl")
    # include("unified/mdop_tests.jl")
end
