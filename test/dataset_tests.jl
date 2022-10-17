using Test, DelayEmbeddings
using Statistics

println("\nTesting Dataset...")

@testset "Dataset" begin
  data = Dataset(rand(1001,3))
  xs = columns(data)

  @testset "Basics" begin
    x, y, z = Dataset(rand(10, 2)), Dataset(rand(10, 2)), rand(10)
    @test Dataset(x) == x # identity
    @test Dataset(x, y,) isa Dataset
    @test Dataset(x, y, y) isa Dataset
    @test size(Dataset(x, y)) == (10, 4)
  end

  @testset "Concatenation/Append" begin

    types = [Int, Float64]
    @testset "hcat with identical element type ($(T))" for T in types
        x1, x2, x3 = T.([1:5 2:6]), T.([3:7 4:8]), T.(5:9)
        D1, D2, D3 = Dataset(x1), Dataset(x2), Dataset(x3)
        y = T.(1:5) |> collect
        @test hcat(D1, y) == Dataset([1:5 2:6 1:5])
        @test hcat(D1, D2) == Dataset([1:5 2:6 3:7 4:8])
        @test hcat(D1, D2, D3) == Dataset([1:5 2:6 3:7 4:8 5:9])
        @test hcat(D1, y) |> size == (5, 3)
        @test hcat(y, D1) |> size == (5, 3)
        @test hcat(D1, y) == Dataset(([1:5 2:6 y]))
        @test hcat(y, D1) == Dataset(([y 1:5 2:6]))
    end

    # TODO: By construction, these errors will occur, because the type constraints are
    # not imposed on the vector inputs, only the dataset input. In contrast, for
    # hcat on datasets only, the we force all datasets to have the same element type.
    #
    # Should we force the element types of the dataset and vector to be identical and
    # throw an informative error message if they are not?
    @testset "hcat with nonidentical element types" begin
        D = Dataset([1:5 2:6]) # Dataset{2, Int}
        x = rand(length(D))    # Vector{Float64}
        @test_throws InexactError hcat(D, x)
        @test_throws InexactError hcat(x, D)
    end
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
