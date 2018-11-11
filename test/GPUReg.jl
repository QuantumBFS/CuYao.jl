using Yao
using Test
using CuYao
using LinearAlgebra
using Yao.Intrinsics

@testset "basics" begin
    a = randn(ComplexF64, 50, 20)
    ca = a|>cu
    batch_normalize!(ca)
    batch_normalize!(a)
    @test ca |> Matrix ≈ a
end

@testset "constructor an measure" begin
    reg = rand_state(10)
    greg = reg |> cu
    @test greg isa GPUReg
    for reg in [rand_state(10, 3), rand_state(10)]
        greg = reg |> cu
        @test size(measure(greg |> copy, nshot=10)) == size(measure(reg, nshot=10))
        @test size(measure!(greg |> copy)) == size(measure!(reg |> copy))
        @test size(measure_reset!(greg |> copy)) == size(measure_reset!(reg |> copy))
        @test size(measure_remove!(greg |> copy)) == size(measure_remove!(reg |> copy))
        @test select(greg |> copy, 12) ≈ select(reg, 12)
        @test size(measure!(greg |> copy |> focus!(3,4,1))) == size(measure!(reg |> copy |> focus!(3,4,1)))
        @test greg |> copy |> focus!(3,4,1) |> relax!(3,4,1) |> cpu ≈ reg

        if nbatch(greg) == 1
            greg1 = greg |> copy |> focus!(1,4,3)
            greg0 = copy(greg1)
            res = measure_remove!(greg1)
            @test select(greg0, res) |> normalize! |> cpu ≈ greg1 |> cpu
        end

        greg1 = greg |> copy
        greg1 = rand_state(10,3)
        res = measure_reset!(greg1, val=3)
        @test all(measure(greg1, nshot=10) .== 3)
    end
end
