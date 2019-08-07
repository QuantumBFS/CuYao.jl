using Test, CuYao, CuArrays, SparseArrays, LinearAlgebra

@testset "cuda cache" begin
    T = ComplexF64
    for A in [randn(T,4,4), sprand(T,4,4,0.5)]
        B = randn(T,4,2)
        v = randn(T,4)
        c = randn(T,4,1)
        cA = A isa Array ? CuArray(A) : CuSparseMatrixCSC(A)
        cB = B |> CuArray
        cv = v |> CuArray
        cc = c |> CuArray

        v .= (A*v)
        mul!(cv,cA,cv)
        @test cv ≈ v

        c .= (A*c)
        mul!(vec(cc),cA,vec(cc))
        @test cc ≈ c

        B .= (A*B)
        cBout = copy(cB)
        mul!(cBout,cA,cB)
        @test cBout ≈ B
    end
end

@testset "cuda cache" begin
    c = cache(X)
    reg = zero_state(1)
    @test copy(reg) |> c ≈ copy(reg) |> X
    cc = copy(c)
    creg = zero_state(1) |> cu
    #@show cumat(ComplexF64,cc) |> typeof
    @test Matrix(state(copy(creg) |> c)) ≈ Matrix(state(copy(creg) |> X))
    @test (copy(creg) |> c).state |> Matrix ≈ state(copy(reg) |> X)
end
