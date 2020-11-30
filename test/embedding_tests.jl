using Test, StaticArrays, DelayEmbeddings

println("\nTesting delay embeddings...")

@testset "embedding" begin

    data = Dataset(rand(10001,3))
    s = data[:, 1]; N = length(s)

    @testset "standard" begin

    	@testset "D = $(D), τ = $(τ)" for D in [2,3], τ in [2,3]
    		R = embed(s, D, τ)
    		@test R[(1+τ):end, 1] == R[1:end-τ, 2]
    		@test size(R) == (length(s) - τ*(D-1), D)
    	end
    end

    @testset "weighted" begin
    	@testset "D = $(D), τ = $(τ)" for D in [2,3], τ in [2,3]

            w = 0.1
    		R1 = embed(s, D, τ)
    		R2 = embed(s, D, τ, w)
            for γ in 0:D-1
                @test  (w^γ) * R1[1, γ+1][1] == R2[1, γ+1]
                @test  (w^γ) * R1[5, γ+1][1] == R2[5, γ+1]
            end
    	end
    end

    @testset "multi-time" begin
        D = 3
        τ1 = [2, 4]
        τ2 = [4, 8]

        R0 = embed(s, D, 2)
        R1 = embed(s, D, τ1)
        R2 = embed(s, D, τ2)

        @test R1 == R0

        R2y = R2[:, 2]
        @test R2y == R0[5:end, 1]
        @test R2[:, 1] == R0[1:end-4, 1]
        @test size(R2) == (N-maximum(τ2), 3)

        @test_throws ArgumentError embed(s, 4, τ1)
    end

end

println("\nTesting generalized embedding...")
@testset "genembed" begin
    τs = (0, 2, -7)
    js = (1, 3, 2)
    ge = GeneralizedEmbedding(τs, js)
    τr = τrange(s, ge)
    @testset "univariate" begin
        x = rand(20)
        em = genembed(x, τs, js)
        @test em == genembed(x, τs)
        @test em[1:3, 3] == x[1:3]
        @test em[1:3, 1] == x[1+7:3+7]
        @test em[1:3, 2] == x[1+7+2:3+7+2]
    end
    @testset "multivariate" begin
        s = Dataset(rand(20, 3))
        em = genembed(s, τs, js)
        x, y, z = columns(s)
        @test em[1:3, 1] == x[1+7:3+7]
        @test em[1:3, 2] == z[1+7+2:3+7+2]
        @test em[1:3, 3] == y[1:3]
    end
    @testset "weighted" begin
        s = Dataset(rand(20, 3))
        x, y, z = columns(s)
        ws = (1, 0, 0.1)
        em = genembed(s, τs, js; ws)
        @test em[1:3, 1] == x[1+7:3+7]
        @test em[1:3, 2] == 0 .* z[1+7+2:3+7+2]
        @test em[1:3, 3] == 0.1 .* y[1:3]
end
