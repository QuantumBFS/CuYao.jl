using LinearAlgebra, Yao.ConstGate
using Test, Random
using CuYao
using StaticArrays
using Yao.ConstGate: SWAPGate
using CUDA

@testset "gpu instruct nbit!" begin
    Random.seed!(3)
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 333)

    for UN in [
            rand_unitary(ComplexF32, 4),
            mat(ComplexF32, CNOT),
            mat(ComplexF32, control(2,2,1=>Z)),
            mat(ComplexF32, put(2,2=>I2)),
            mat(ComplexF32, put(2,2=>P0))
            ]

        @test instruct!(v1 |> CuArray, UN, (3,1)) |> Vector ≈ instruct!(v1 |> copy, UN, (3,1))
        @test instruct!(vn |> CuArray, UN, (3,1)) |> Matrix ≈ instruct!(vn |> copy, UN, (3,1))
    end
    # sparse matrix like P0, P1 et. al. are not implemented.
end

@testset "gpu swapapply!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 333)

    @test instruct!(v1 |> CuArray, Val(:SWAP), (3,5)) |> Vector ≈ instruct!(v1 |> copy, Val(:SWAP), (3,5))
    @test instruct!(vn |> CuArray, Val(:SWAP), (3,5)) |> Matrix ≈ instruct!(vn |> copy, Val(:SWAP), (3,5))
end

@testset "gpu instruct! 1bit" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 333)

    for U1 in [mat(H), mat(Z), mat(I2), mat(ConstGate.P0), mat(X), mat(Y)]
        @test instruct!(v1 |> CuArray, U1, (3,)) |> Vector ≈ instruct!(v1 |> copy, U1, (3,))
        @test instruct!(vn |> CuArray, U1, (3,)) |> Matrix ≈ instruct!(vn |> copy, U1, (3,))
    end
    # sparse matrix like P0, P1 et. al. are not implemented.
end

@testset "gpu xyz-instruct!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 333)

    for G in [:X, :Y, :Z, :T, :H, :Tdag, :S, :Sdag]
        @test instruct!(v1 |> CuArray, Val(G), (3,)) |> Vector ≈ instruct!(v1 |> copy, Val(G), (3,))
        @test instruct!(vn |> CuArray, Val(G), (3,)) |> Matrix ≈ instruct!(vn |> copy, Val(G), (3,))
        if G != :H
            @test instruct!(v1 |> CuArray, Val(G), (1,3,4)) |> Vector ≈ instruct!(v1 |> copy, Val(G), (1,3,4))
            @test instruct!(vn |> CuArray,  Val(G),(1,3,4)) |> Matrix ≈ instruct!(vn |> copy, Val(G), (1,3,4))
        end
    end
end

@testset "gpu cn-xyz-instruct!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 333)

    for G in [:X, :Y, :Z, :T, :Tdag, :S, :Sdag]
        @test instruct!(v1 |> CuArray, Val(G), (3,), (4,5), (0, 1)) |> Vector ≈ instruct!(v1 |> copy, Val(G), (3,), (4,5), (0, 1))
        @test instruct!(vn |> CuArray, Val(G), (3,), (4,5), (0, 1)) |> Matrix ≈ instruct!(vn |> copy, Val(G), (3,), (4,5), (0, 1))
        @test instruct!(v1 |> CuArray, Val(G), (3,), (1,), (1,)) |> Vector ≈ instruct!(v1 |> copy, Val(G),(3,), (1,), (1,))
        @test instruct!(vn |> CuArray, Val(G), (3,), (1,), (1,)) |> Matrix ≈ instruct!(vn |> copy, Val(G),(3,), (1,), (1,))
    end
end

@testset "pswap" begin
    ps = put(6, (2,4)=>rot(SWAP, π/2))
    reg = rand_state(6; nbatch=10)
    @test apply!(reg |> cu, ps) |> cpu ≈ apply!(copy(reg), ps)
    @test apply!(reg |> cu, ps).state isa CuArray
end

@testset "regression test: Rx, Ry, Rz, CPHASE" begin
    Random.seed!(3)
    nbit = 6
    for ps in [put(6, (2,)=>Rx(π/2)), put(6, 2=>Ry(0.5)),  put(6, 2=>Rz(0.4))]
        reg = rand_state(6; nbatch=10)
        @test apply!(reg |> cu, ps) |> cpu ≈ apply!(copy(reg), ps)
        @test apply!(reg |> cu, ps).state isa CuArray
    end
end