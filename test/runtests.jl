using DelayEmbeddings
using StaticArrays

# Download some test timeseries
repo = "https://raw.githubusercontent.com/JuliaDynamics/ExercisesRepo/master/timeseries"
tsfolder = joinpath(@__DIR__, "timeseries")
todownload = ["$n.csv" for n in 1:4]

mkpath(tsfolder)
for a in todownload
    download(repo*"/"*a, joinpath(tsfolder, a))
end

ti = time()

diffeq = (atol = 1e-9, rtol = 1e-9, maxiters = typemax(Int))

include("dataset_tests.jl")
include("embedding_tests.jl")
include("delaytime_test.jl")
include("embedding_dimension_test.jl")
include("test_pecora.jl")
include("uzal_cost_test.jl")
include("mdop_tests.jl")
include("test_garcia.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, sigdigits=3), " seconds or ", round(ti/60, sigdigits=3), " minutes")
