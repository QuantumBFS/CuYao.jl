using LinearAlgebra, Yao.ConstGate
using Test
using CuYao
using StaticArrays
using Yao.ConstGate: SWAPGate
using CuArrays

@testset "gpu unapply!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF64, N)
    vn = randn(ComplexF64, N, 333)

    for U1 in [mat(H), mat(Y), mat(Z), mat(I2), mat(P0)]
        @test instruct!(v1 |> cu, U1, (3,)) |> Vector ≈ instruct!(v1 |> copy, U1, (3,))
        @test instruct!(vn |> cu, U1, (3,)) |> Matrix ≈ instruct!(vn |> copy, U1, (3,))
    end
    # sparse matrix like P0, P1 et. al. are not implemented.
end

@testset "gpu swapapply!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF64, N)
    vn = randn(ComplexF64, N, 333)

    @test instruct!(v1 |> cu, Val(:SWAP), (3,5)) |> Vector ≈ instruct!(v1 |> copy, Val(:SWAP), (3,5))
    @test instruct!(vn |> cu, Val(:SWAP), (3,5)) |> Matrix ≈ instruct!(vn |> copy, Val(:SWAP), (3,5))
end

@testset "gpu instruct!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF64, N)
    vn = randn(ComplexF64, N, 333)

    for U1 in [mat(H), mat(Y), mat(Z), mat(I2), mat(P0)]
        @test instruct!(v1 |> cu, U1, (3,)) |> Vector ≈ instruct!(v1 |> copy, U1, (3,))
        @test instruct!(vn |> cu, U1, (3,)) |> Matrix ≈ instruct!(vn |> copy, U1, (3,))
    end
    # sparse matrix like P0, P1 et. al. are not implemented.
end

@testset "gpu xyz-instruct!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF64, N)
    vn = randn(ComplexF64, N, 333)

    for G in [:X, :Y, :Z, :T, :Tdag]
        @test instruct!(v1 |> CuArray, Val(G), (3,)) |> Vector ≈ instruct!(v1 |> copy, Val(G), (3,))
        @test instruct!(vn |> CuArray, Val(G), (3,)) |> Matrix ≈ instruct!(vn |> copy, Val(G), (3,))
        @test instruct!(v1 |> CuArray, Val(G), (1,3,4)) |> Vector ≈ instruct!(v1 |> copy, Val(G), (1,3,4))
        @test instruct!(vn |> CuArray,  Val(G),(1,3,4)) |> Matrix ≈ instruct!(vn |> copy, Val(G), (1,3,4))
    end
    # why the precision is low?
    for G in [:S, :Sdag]
        @test isapprox(Vector(instruct!(v1 |> CuArray, Val(G), (3,))), instruct!(v1 |> copy, Val(G), (3,)); atol=1e-5)
        @test isapprox(Matrix(instruct!(vn |> CuArray, Val(G), (3,))), instruct!(vn |> copy, Val(G), (3,)); atol=1e-5)
        @test isapprox(Vector(instruct!(v1 |> CuArray, Val(G), (1,3,4))), instruct!(v1 |> copy, Val(G), (1,3,4)); atol=1e-5)
        @test isapprox(Matrix(instruct!(vn |> CuArray,  Val(G),(1,3,4))), instruct!(vn |> copy, Val(G), (1,3,4)); atol=1e-5)
    end
end

@testset "gpu cn-xyz-instruct!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF64, N)
    vn = randn(ComplexF64, N, 333)

    for G in [:X, :Y, :Z, :T, :Tdag, :S, :Sdag]
        @test instruct!(v1 |> cu, Val(G), (3,), (4,5), (0, 1)) |> Vector ≈ instruct!(v1 |> copy, Val(G), (3,), (4,5), (0, 1))
        @test instruct!(vn |> cu, Val(G), (3,), (4,5), (0, 1)) |> Matrix ≈ instruct!(vn |> copy, Val(G), (3,), (4,5), (0, 1))
        @test instruct!(v1 |> cu, Val(G), (3,), (1,), (1,)) |> Vector ≈ instruct!(v1 |> copy, Val(G),(3,), (1,), (1,))
        @test instruct!(vn |> cu, Val(G), (3,), (1,), (1,)) |> Matrix ≈ instruct!(vn |> copy, Val(G),(3,), (1,), (1,))
    end
end

@testset "pswap" begin
    ps = put(6, (2,4)=>rot(SWAP, π/2))
    reg = rand_state(6; nbatch=10)
    @test apply!(reg |> cu, ps) ≈ apply!(reg, ps)
    @test apply!(reg |> cu, ps).state isa CuArray
end
