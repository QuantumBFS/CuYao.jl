include("../src/CuYao.jl")
using .CuYao
using Yao, Test

@testset "constructor an measure" begin
    reg = rand_state(10)
    greg = reg |> cu
    @test greg isa GPUReg
    for reg in [rand_state(10, 3), rand_state(10)]
        greg = reg |> cu
        @test size(measure(greg |> copy, 10)) == size(measure(reg, 10))
        @test size(measure!(greg |> copy)) == size(measure!(reg |> copy))
        @test size(measure_reset!(greg |> copy)) == size(measure_reset!(reg |> copy))
        @test size(measure_remove!(greg |> copy)) == size(measure_remove!(reg |> copy))
        @test select(greg |> copy, 12) ≈ select(reg, 12)
        @test size(measure!(greg |> copy |> focus!(3,4,1))) == size(measure!(reg |> copy |> focus!(3,4,1)))
        @test greg |> copy |> focus!(3,4,1) |> relax!(3,4,1) ≈ reg
    end
end
