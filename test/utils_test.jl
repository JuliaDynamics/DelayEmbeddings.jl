using Test, DelayEmbeddings

println("\nTesting utils...")

@testset "utils" begin

    A = orthonormal(50, 50)
    @test size(A) == (50, 50)
    @test A isa Matrix

    B = orthonormal(3, 3)
    @test B isa SMatrix
end
