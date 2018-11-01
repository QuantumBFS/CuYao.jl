include("../src/GPUReg.jl")
using Yao, Test

@testset "constructor" begin
    reg = rand_state(10)
    greg = reg |> cu
    @test greg isa GPUReg
    for reg in [rand_state(10, 3), rand_state(10)]
        @test size(measure(reg |> cu, 10)) == size(measure(reg, 10))
        @test size(measure!(reg |> cu)) == size(measure!(reg |> copy))
        @test size(measure_reset!(reg |> cu)) == size(measure_reset!(reg |> copy))
        @test size(measure_remove!(reg |> cu)) == size(measure_remove!(reg |> copy))
        @test select(reg |> cu, 12) ≈ select(reg, 12)
    end
end
