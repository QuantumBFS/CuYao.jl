include("../src/CuYao.jl")
using CuArrays, GPUArrays
using Test

@testset "isapprox-complex" begin
    ca = CuArray(randn(ComplexF64,3,3))
    cb = copy(ca)
    #@test ca ≈ cb still error!
    cb[1:1, 1:1] .+= 1e-7im
    @test isapprox(ca, cb, atol=1e-5)
    @test !isapprox(ca, cb, atol=1e-9)
end

@testset "view general" begin
    a = randn(5,6,8) |> CuArray
    @test view(a, 2:4, 4, [1,4,3]) |> size == (3, 3)
end
