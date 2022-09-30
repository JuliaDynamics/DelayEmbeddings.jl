using Test, DelayEmbeddings
using Statistics

println("\nTesting Dataset...")

@testset "Dataset" begin
  data = Dataset(rand(1001,3))
  xs = columns(data)

  @testset "Concatenation/Append" begin
    x, y, z = Dataset(rand(10, 2)), Dataset(rand(10, 2)), rand(10)
    @test Dataset(x) == x
    @test Dataset(x, y) isa Dataset
    @test size(Dataset(x, y)) == (10, 4)
    @test hcat(x, y) isa Dataset
    @test Dataset(x, z) isa Dataset
    @test Dataset(z, x) isa Dataset

    append!(x, y)
    @test length(x) == 20
    w = hcat(x, rand(20))
    @test size(w) == (20, 3)
  end

  @testset "Methods & Indexing" begin
    a = data[:, 1]
    b = data[:, 2]
    c = data[:, 3]

    @test data[1,1] isa Float64
    @test a isa Vector{Float64}
    @test Dataset(a, b, c) == data
    @test size(Dataset(a, b)) == (1001, 2)

    @test data[5] isa SVector{3, Float64}
    @test data[11:20] isa Dataset
    @test data[:, 2:3][:, 1] == data[:, 2]

    @test size(data[1:10,1:2]) == (10,2)
    @test data[1:10,1:2] == Dataset(a[1:10], b[1:10])
    @test data[1:10, SVector(1, 2)] == data[1:10, 1:2]
    e = data[5, SVector(1,2)]
    @test e isa SVector{2, Float64}

    sub = @view data[11:20]
    @test sub isa DelayEmbeddings.SubDataset
    @test sub[2] == data[12]
    @test dimension(sub) == dimension(data)
    d = sub[:, 1]
    @test d isa Vector{Float64}
    e = sub[5, 1:2]
    @test e isa Vector{Float64}
    @test length(e) == 2
    e = sub[5, SVector(1,2)]
    @test e isa SVector{2, Float64}
    f = sub[5:8, 1:2]
    @test f isa Dataset

    # setindex
    data[1] = SVector(0.1,0.1,0.1)
    @test data[1] == SVector(0.1,0.1,0.1)
    @test_throws ErrorException (data[:,1] .= 0)
  end

  @testset "copy" begin
    d = Dataset(rand(10, 2))
    v = vec(d)
    d2 = copy(d)
    d2[1] == d[1]
    d2[1] = SVector(5.0, 5.0)
    @test d2[1] != d[1]
  end

  @testset "minmax" begin
    mi = minima(data)
    ma = maxima(data)
    mimi, mama = minmaxima(data)
    @test mimi == mi
    @test mama == ma
    for i in 1:3
      @test mi[i] < ma[i]
      a,b = extrema(xs[i])
      @test a == mi[i]
      @test b == ma[i]
    end
  end

  @testset "Conversions" begin
    m = Matrix(data)
    @test Dataset(m) == data

    m = rand(1000, 4)
    @test Matrix(Dataset(m)) == m
  end

  @testset "standardize" begin
    r = standardize(data)
    rs = columns(r)
    for x in rs
      m, s = mean(x), std(x)
      @test abs(m) < 1e-8
      @test abs(s - 1) < 1e-8
    end
  end
end
