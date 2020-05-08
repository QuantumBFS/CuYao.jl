using CuYao
using CuArrays, GPUArrays
using Test
using CUDAnative

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

@testset "permutedims vector" begin
    ca = randn(ComplexF64,3,4,5,1)
    @test permutedims(CuArray(ca), [2,1,4,3]) |> Array ≈ permutedims(ca, [2,1,4,3])
end

@testset "Complex pow" begin
    for T in [ComplexF64, ComplexF32]
        a = CuArray(randn(T, 4, 4))
        @test Array(CUDAnative.pow.(a, Int32(3))) ≈ Array(a).^3
        @test Array(CUDAnative.pow.(a, real(T)(3))) ≈ Array(a).^3
    end
end
