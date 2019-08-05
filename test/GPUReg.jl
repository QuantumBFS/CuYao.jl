using Test
using CuYao
using CuYao: tri2ij
using LinearAlgebra
using BitBasis
using Statistics: mean
using StaticArrays
using CuArrays
CuArrays.allowscalar(false)

@testset "basics" begin
    a = randn(ComplexF64, 50, 20)
    ca = a|>cu
    batch_normalize!(ca)
    batch_normalize!(a)
    @test ca |> Matrix ≈ a

    for l = 1:100
        i, j = tri2ij(l)
        @test (i-1)*(i-2)÷2+j == l
    end
end

@testset "constructor an measure" begin
    reg = rand_state(10)
    greg = reg |> cu
    @test greg isa GPUReg
    for reg in [rand_state(10, nbatch=333), rand_state(10)]
        greg = reg |> cu
        @test size(measure(greg |> copy, nshots=10)) == size(measure(reg, nshots=10))
        @test size(measure!(greg |> copy)) == size(measure!(reg |> copy))
        @test size(measure_collapseto!(greg |> copy)) == size(measure_collapseto!(reg |> copy))
        @test size(measure_remove!(greg |> copy)) == size(measure_remove!(reg |> copy))
        @test select(greg |> copy, BitStr{10}(12)) ≈ select(reg, BitStr{10}(12))
        @test size(measure!(greg |> copy |> focus!(3,4,1))) == size(measure!(reg |> copy |> focus!(3,4,1)))
        @test greg |> copy |> focus!(3,4,1) |> relax!(3,4,1) |> cpu ≈ reg

        if nbatch(greg) == 1
            greg1 = greg |> copy |> focus!(1,4,3)
            greg0 = copy(greg1)
            res = measure_remove!(greg1)
            @test select(greg0, res |> Vector) |> normalize! |> cpu ≈ greg1 |> cpu
        end

        greg1 = greg |> copy |> focus!(1,4,3)
        greg0 = copy(greg1)
        res = measure_collapseto!(greg1; config=3)
        @test all(measure(greg1, nshots=10) .== 3)
        @test greg1 |> isnormalized
        @test all(select.(greg0 |> cpu, res |> Vector) .|> normalize! .≈ select.(greg1 |> cpu, BitStr{10}(3)))

        greg1 = greg |> copy |> focus!(1,4,3)
        greg0 = copy(greg1)
        res = measure!(greg1)
        @test all(select.(greg0 |> cpu, res |> Vector) .|> normalize! .≈ select.(greg1 |> cpu, res|>Vector))
    end

    @test join(rand_state(3) |> cu, rand_state(3) |> cu) |> nactive == 6
    @test join(rand_state(3, nbatch=10) |> cu, rand_state(3, nbatch=10) |> cu) |> nactive == 6
end

@testset "insert_qubits!" begin
    reg = rand_state(5; nbatch=10)
    res = insert_qubits!(reg |> cu, 3; nqubits=2)
    @test insert_qubits!(reg, 3; nqubits=2) ≈ res

    reg = rand_state(5, nbatch=10) |>focus!(2,3)
    res = insert_qubits!(reg |> cu, 3; nqubits=2)
    @test insert_qubits!(reg, 3; nqubits=2) ≈ res
end

@testset "cuda-op-measures" begin
    reg = rand_state(8; nbatch=32) |> cu
    op = repeat(5, X, 1:5)

    # measure!
    reg2 = reg |> copy
    res = measure!(op, reg2, 2:6)
    res2 = measure!(op, reg2, 2:6)
    @test size(res) == (32,)
    @test res2 == res

    # measure_collapseto!
    reg2 = reg |> copy
    res = measure_collapseto!(op, reg2, 2:6; config=0)
    reg2 |> repeat(8, H, 2:6)
    res2 = measure_collapseto!(op, reg2, 2:6)
    @test size(res) == (32,) == size(res2)
    @test all(res2 .== 1)

    # measure_remove!
    reg2 = reg |> copy
    res = measure_remove!(op, reg2, 2:6)
    @test size(res) == (32,)

    reg = repeat(ArrayReg([1,-1+0im]/sqrt(2.0)), 10) |> cu
    @test measure!(X, reg) |> mean ≈ -1
    reg = repeat(ArrayReg([1.0,0+0im]), 1000)
    @test abs(measure!(X, reg) |> mean) < 0.1
end

@testset "cuda kron getindex" begin
    a = randn(3,4)
    b = randn(4,2)
    ca, cb = cu(a), cu(b)
    @test kron(ca, cb) |> Array ≈ kron(a, b)

    v = randn(100) |> cu
    inds = [3,5,2,1,7,1]
    @test v[inds] ≈ v[inds |> CuVector]
end
