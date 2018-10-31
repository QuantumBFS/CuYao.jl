using CuYao, Test

@testset "gpu unapply!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 3)

    for U1 in [mat(H), mat(Y), mat(Z), mat(I2), mat(P0)]
        @test unapply!(v1 |> cu, U1, (3,)) |> Vector ≈ unapply!(v1 |> copy, U1, (3,))
        @test unapply!(vn |> cu, U1, (3,)) |> Matrix ≈ unapply!(vn |> copy, U1, (3,))
    end
    # sparse matrix like P0, P1 et. al. are not implemented.
end

@testset "gpu u1apply!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 3)

    for U1 in [mat(H), mat(Y), mat(Z), mat(I2), mat(P0)]
        @test u1apply!(v1 |> cu, U1, 3) |> Vector ≈ u1apply!(v1 |> copy, U1, 3)
        @test u1apply!(vn |> cu, U1, 3) |> Matrix ≈ u1apply!(vn |> copy, U1, 3)
    end
    # sparse matrix like P0, P1 et. al. are not implemented.
end


