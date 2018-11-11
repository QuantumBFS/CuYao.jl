using CuYao
using Yao, Test
using LinearAlgebra

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
        @test greg |> copy |> focus!(3,4,1) |> relax!(3,4,1) |> cpu ≈ reg |> cpu

        reg = rand_state(6) |> cu |> focus(1,4,3)
        reg0 = copy(reg)
        res = measure_remove!(reg)
        @test select(reg0, res) |> normalize! |> cpu ≈ reg |> cpu
    end
end
