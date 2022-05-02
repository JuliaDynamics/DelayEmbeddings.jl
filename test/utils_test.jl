using Test, DelayEmbeddings

println("\nTesting utils...")

@testset "utils" begin

    A = orthonormal(50, 50)
    @test size(A) == (50, 50)

end
