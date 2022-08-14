using Test, DelayEmbeddings

println("\nTesting Dataset distance...")

@testset "Dataset Distances" begin
    d1 = range(1, 10; length = 11)
    d2 = d1 .+ 10
    @test dataset_distance(Dataset(d1), Dataset(d2)) == 1.0
    @test dataset_distance(Dataset(d1), Dataset(d2), Hausdorff()) == 10.0

    d1 = Dataset([SVector(0, 1)])
    d2 = Dataset([SVector(1, 2)])
    @test dataset_distance(d1, d2, Chebyshev()) == 1.0
    d2 = Dataset([SVector(1, 2), SVector(1, 3)])
    @test dataset_distance(d1, d2, Hausdorff(Chebyshev())) == 2.0
end