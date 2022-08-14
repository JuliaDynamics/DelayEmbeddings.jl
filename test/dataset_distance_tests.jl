using Test, DelayEmbeddings

println("\nTesting Dataset distance...")

@testset "Dataset Distances" begin
  d1 = range(1, 10; length = 11)
  d2 = d1 .+ 10
  @test dataset_distance(Dataset(d1), Dataset(d2)) == 1.0
  @test dataset_distance(Dataset(d1), Dataset(d2), Hausdorff(Euclidean())) == 10.0
end