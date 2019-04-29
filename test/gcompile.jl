using StaticArrays
using Test
using CuYao

@testset "gcompile" begin
    c = chain(put(12, 2=>X), put(12, 2=>rot(X, 0.2)), control(12, 3, 2=>rot(X,0.3)))
    cc = c |> KernelCompiled
    reg = rand_state(12) |> cu
    @test reg |> copy |> c ≈ reg |> cpu |> c
    @test reg |> copy |> cc ≈ reg |> cpu |> cc
end
