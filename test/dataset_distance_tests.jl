using Test, DelayEmbeddings

println("\nTesting Dataset distance...")

@testset "Dataset Distances" begin
    d1 = range(1, 10; length = 11)
    d2 = d1 .+ 10
    @test dataset_distance(Dataset(d1), Dataset(d2)) == 1.0
    @test dataset_distance(Dataset(d1), Dataset(d2); brute = false) == 1.0
    @test dataset_distance(Dataset(d1), Dataset(d2); brute = true) == 1.0
    @test dataset_distance(Dataset(d1), Dataset(d2), Hausdorff()) == 10.0

    d1 = Dataset([SVector(0, 1)])
    d2 = Dataset([SVector(1, 2)])
    @test dataset_distance(d1, d2, Chebyshev()) == 1.0
    d2 = Dataset([SVector(1, 2), SVector(1, 3)])
    @test dataset_distance(d1, d2, Hausdorff(Chebyshev())) == 2.0
end

@testset "Sets of Dataset Distances" begin
    d1 = range(1, 10; length = 11)
    d2 = d1 .+ 10
    d3 = d2 .+ 10
    set1 = Dataset.([d1, d2, d3])
    r = range(1, 10; length = 11)
    D = 6
    makedata = x -> (y = zeros(D); y[1] = x; y)
    d1 = Dataset([makedata(x) for x in r])
    d2 = Dataset([makedata(x+10) for x in r])
    d3 = Dataset([makedata(x+20) for x in r])
    set2 = Dataset.([d1, d2, d3])

    for set in (set1, set2)
        dsds = datasets_sets_distances(set, set)
        for i in 1:3
            @test dsds[i][i] == 0
        end
        @test dsds[2][1] == dsds[2][3] == 1
        @test dsds[3][1] == dsds[1][3] == 11
    end

end