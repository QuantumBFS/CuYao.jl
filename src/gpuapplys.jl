using Test
using LuxurySparse
using Yao.Intrinsics
using Yao.Boost
using StaticArrays
import Yao.Intrinsics: unrows!, u1apply!, _unapply!, swaprows!
import Yao.Boost: zapply!, xapply!, yapply!, cxapply!, cyapply!, czapply!

###################### unapply! ############################
function _unapply!(state::CuVecOrMat, U::AbstractMatrix, locs_raw::SDVector, ic::IterControl)
    function kernel(state, U, locs_raw)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unrows!(state, ic[i] + locs_raw, U)
        return
    end
    @cuda blocks=length(ic) kernel(state, U, locs_raw)
    state
end

function _unapply!(state::CuVecOrMat, U::SDSparseMatrixCSC, locs_raw::SDVector, ic::IterControl)
    return _unapply!(state, SMatrix{size(U, 1), size(U,2)}(U), locs_raw, ic)
    wsz = (i==0 ? size(U, 1) : size(state, i) for i in ndims(state))
    function kernel(state, U, locs_raw)
        work = similar(state, wsz...)
        locs = locs_raw+ic[i]
        unrows!(state, locs, U, work)
        return
    end
    @cuda blocks=length(ic) kernel(state, U, locs_raw)
    state
end

################## General U1 apply! ###################
# diagonal
diag_widget_1(mask, a, d) = (v, b)->testany(b, mask) ? d*v : a*v
function u1apply!(state::CuVecOrMat, U1::SDDiagonal, ibit::Int)
    mask = bmask(ibit)
    a, d = U1.diag
    state .= diag_widget_1(mask, a, d).(state, basis(state))
end

function u1apply!(state::CuVecOrMat, U1::AbstractMatrix, ibit::Int)
    # reorder a unirary matrix.
    a, c, b, d = U1
    step = 1<<(ibit-1)
    ic = itercontrol(nactive(state), [ibit], [0])
    function kernel(state, ic, step, a, b, c, d)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        x = ic[i]+1
        u1rows!(state, x, x+step, a, b, c, d)
        return
    end
    @cuda blocks=length(ic) kernel(state, ic, step, a, b, c, d)
    state
end

function u1apply!(state::CuVecOrMat{T}, U1::SDPermMatrix, ibit::Int) where T
    U1.perm[1] == 1 && return u1apply!(state, Diagonal(U1), ibit)

    mask = bmask(ibit)
    b, c = T(U1.vals[1]), T(U1.vals[2])
    step = 1<<(ibit-1)
    ic = itercontrol(nactive(state), [ibit], [0])
    function kernel(state, ic, step, c, b)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = ic[i] + 1
        swaprows!(state, j, j+step, c, b)
        return
    end
    @cuda blocks=length(ic) kernel(state, ic, step, c, b)
    state
end

u1apply!(state::CuVecOrMat, U1::IMatrix, ibit::Int) = state

function xapply!(state::CuVecOrMat{T}, bits::Ints) where T
    length(bits) == 0 && return state
    mask = bmask(bits...)
    do_mask = bmask(bits[1])

    function kernel(state, mask, do_mask)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        b = i-1
        if testany(b, do_mask)
            i_ = flip(b, mask) + 1
            swaprows!(state, i, i_)
        end
        return
    end
    @cuda blocks=size(state, 1) kernel(state, mask, do_mask)
    state
end

function yapply!(state::CuVecOrMat{T}, bits::Ints{Int}) where T
    length(bits) == 0 && return state
    mask = bmask(Int, bits...)
    do_mask = bmask(Int, bits[1])
    bit_parity = length(bits)%2 == 0 ? 1 : -1
    factor = T(-im)^length(bits)

    function kernel(state, mask, do_mask, factor)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        b = i-1
        if testany(b, do_mask)
            i_ = flip(b, mask) + 1
            factor1 = count_ones(b&mask)%2 == 1 ? -factor : factor
            factor2 = factor1*bit_parity
            swaprows!(state, i, i_, factor2, factor1)
        end
        return
    end
    @cuda blocks=size(state, 1) kernel(state, mask, do_mask, factor)
    state
end

z_widget_n(mask) = (v, b)->count_ones(b&mask)%2==1 ? -v : v
z_widget_1(mask) = (v, b)->testany(b, mask) ? -v : v

function zapply!(state::CuVecOrMat{T}, bits::Ints{Int}) where T
    mask = bmask(Int, bits...)
    state .= z_widget_n(mask).(state, basis(state))
end

function zapply!(state::CuVecOrMat{T}, bit::Int) where T
    mask = bmask(Int, bit)
    state .= z_widget_1(mask).(state, basis(state))
end

@testset "gpu xyz-apply!" begin
    nbit = 6
    N = 1<<nbit
    LOC1 = SVector{2}([0, 1])
    v1 = randn(ComplexF32, N)
    vn = randn(ComplexF32, N, 3)

    for func in [xapply!, yapply!, zapply!]
        @test func(v1 |> cu, 3) |> Vector ≈ func(v1 |> copy, 3)
        @test func(vn |> cu, 3) |> Matrix ≈ func(vn |> copy, 3)
        @test func(v1 |> cu, [1,3,4]) |> Vector ≈ func(v1 |> copy, [1,3,4])
        @test func(vn |> cu, [1,3,4]) |> Matrix ≈ func(vn |> copy, [1,3,4])
    end
    # sparse matrix like P0, P1 et. al. are not implemented.
end

for (G, FACTOR) in zip([:s, :t, :sdag, :tdag], [:(im), :($(exp(im*π/4))), :(-im), :($(exp(-im*π/4)))])
    FUNC = Symbol(G, :apply!)
    @eval function $FUNC(state::CuVecOrMat{T}, bits::Ints{Int}) where T
        mask = bmask(Int, bits...)
        for b in basis(Int, state)
            mulrow!(state, b+1, $FACTOR^count_ones(b&mask))
        end
        state
    end
end

for (G, FACTOR) in zip([:z, :s, :t, :sdag, :tdag], [:(-1), :(im), :($(exp(im*π/4))), :(-im), :($(exp(-im*π/4)))])
FUNC = Symbol(G, :apply!)
@eval function $FUNC(state::CuVecOrMat{T}, ibit::Int) where T
    mask = bmask(ibit)
    step = 1<<(ibit-1)
    step_2 = 1<<ibit
    for j = 0:step_2:size(state, 1)-step
        for i = j+step+1:j+step_2
            mulrow!(state, i, $FACTOR)
        end
    end
    state
end
end

################### Multi Controlled Version ####################
for (G, FACTOR) in zip([:z, :s, :t, :sdag, :tdag], [:(-1), :(im), :($(exp(im*π/4))), :(-im), :($(exp(-im*π/4)))])
    FUNC = Symbol(:c, G, :apply!)
    @eval function $FUNC(state::CuVecOrMat{T}, cbits, cvals, b2::Int) where T
        c = controller([cbits..., b2[1]], [cvals..., 1])
        for b = basis(state)
            if b |> c
                mulrow!(state, b+1, $FACTOR)
            end
        end
        state
    end
end

function cyapply!(state::CuVecOrMat{T}, cbits, cvals, b2::Int) where T
    c = controller([cbits..., b2[1]], [cvals..., 0])
    mask2 = bmask(b2...)
    @simd for b = basis(state)
        local i_::Int
        if b |> c
            i = b+1
            i_ = flip(b, mask2) + 1
            swaprows!(state, i, i_, im, -im)
        end
    end
    state
end


function cxapply!(state::CuVecOrMat{T}, cbits, cvals, b2) where T
    c = controller([cbits..., b2[1]], [cvals..., 0])
    mask2 = bmask(b2...)

    @simd for b = basis(state)
        local i_::Int
        if b |> c
            i = b+1
            i_ = flip(b, mask2) + 1
            swaprows!(state, i, i_)
        end
    end
    state
end

################### Single Controlled Version ####################
for (G, FACTOR) in zip([:z, :s, :t, :sdag, :tdag], [:(-1), :(im), :($(exp(im*π/4))), :(-im), :($(exp(-im*π/4)))])
    FUNC = Symbol(:c, G, :apply!)
    @eval function $FUNC(state::CuVecOrMat{T}, cbit::Int, cval::Int, b2::Int) where T
        mask2 = bmask(b2)
        step = 1<<(cbit-1)
        step_2 = 1<<cbit
        start = cval==1 ? step : 0
        for j = start:step_2:size(state, 1)-step+start
            for i = j+1:j+step
                if testall(i-1, mask2)
                    mulrow!(state, i, $FACTOR)
                end
            end
        end
        state
    end
end

function cyapply!(state::CuVecOrMat{T}, cbit::Int, cval::Int, b2::Int) where T
    mask2 = bmask(b2)
    mask = bmask(cbit, b2)

    step = 1<<(cbit-1)
    step_2 = 1<<cbit
    start = cval==1 ? step : 0
    for j = start:step_2:size(state, 1)-step+start
        local i_::Int
        @simd for b = j:j+step-1
            @inbounds if testall(b, mask2)
                i = b+1
                i_ = flip(b, mask2) + 1
                if testall(b, mask2)
                    factor = T(im)
                else
                    factor = T(-im)
                end
                swaprows!(state, i, i_, -factor, factor)
            end
        end
    end
    state
end

function cxapply!(state::CuVecOrMat{T}, cbit::Int, cval::Int, b2::Int) where T
    mask2 = bmask(b2)
    mask = bmask(cbit, b2)

    step = 1<<(cbit-1)
    step_2 = 1<<cbit
    start = cval==1 ? step : 0
    for j = start:step_2:size(state, 1)-step+start
        local i_::Int
        @simd for b = j:j+step-1
            @inbounds if testall(b, mask2)
                i = b+1
                i_ = flip(b, mask2) + 1
                swaprows!(state, i, i_)
            end
        end
    end
    state
end
