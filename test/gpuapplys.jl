using Yao, Yao.Boost, Yao.Intrinsics
using Test
# using CuYao
include("../src/gpuapplys.jl")

@testset "gpu unapply!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 3)

    for U1 in [mat(H), mat(Y), mat(Z), mat(I2), mat(P0)]
        @test unapply!(v1 |> cu, U1, (3,)) |> Vector ≈ unapply!(v1 |> copy, U1, (3,))
        @test unapply!(vn |> cu, U1, (3,)) |> Matrix ≈ unapply!(vn |> copy, U1, (3,))
    end
    # sparse matrix like P0, P1 et. al. are not implemented.
end

@testset "gpu u1apply!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 3)

    for U1 in [mat(H), mat(Y), mat(Z), mat(I2), mat(P0)]
        @test u1apply!(v1 |> cu, U1, 3) |> Vector ≈ u1apply!(v1 |> copy, U1, 3)
        @test u1apply!(vn |> cu, U1, 3) |> Matrix ≈ u1apply!(vn |> copy, U1, 3)
    end
    # sparse matrix like P0, P1 et. al. are not implemented.
end

@testset "gpu xyz-apply!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 3)

    for func in [xapply!, yapply!, zapply!, tapply!, tdagapply!, sapply!, sdagapply!][1:3]
        @test func(v1 |> cu, 3) |> Vector ≈ func(v1 |> copy, 3)
        @test func(vn |> cu, 3) |> Matrix ≈ func(vn |> copy, 3)
        @test func(v1 |> cu, [1,3,4]) |> Vector ≈ func(v1 |> copy, [1,3,4])
        @test func(vn |> cu, [1,3,4]) |> Matrix ≈ func(vn |> copy, [1,3,4])
    end
end

@testset "gpu cn-xyz-apply!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 3)

    for func in [cxapply!, cyapply!, czapply!, ctapply!, ctdagapply!, csapply!, csdagapply!][1:3]
        @test func(v1 |> cu, (4,5), (0, 1), 3) |> Vector ≈ func(v1 |> copy, (4,5), (0, 1), 3)
        @test func(vn |> cu, (4,5), (0, 1), 3) |> Matrix ≈ func(vn |> copy, (4,5), (0, 1), 3)
        @test func(v1 |> cu, 1, 1, 3) |> Vector ≈ func(v1 |> copy, 1, 1,3)
        @test func(vn |> cu, 1, 1, 3) |> Matrix ≈ func(vn |> copy, 1, 1,3)
    end
end
