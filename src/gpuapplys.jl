#=
using Test
using Yao, Yao.Blocks, Yao.Intrinsics, Yao.Boost

using GPUArrays, CuArrays, CUDAnative
CuArrays.allowscalar(false)

using LuxurySparse, StaticArrays, LinearAlgebra
=#

import Yao.Intrinsics: unrows!, u1apply!, _unapply!, swaprows!, cunapply!, autostatic
import Yao.Boost: zapply!, xapply!, yapply!, cxapply!, cyapply!, czapply!, sapply!, sdagapply!, tapply!, tdagapply!
import Yao.Blocks: swapapply!

include("kernels.jl")

autostatic(A::AbstractVecOrMat) = A |> staticize

###################### unapply! ############################
function cunapply!(state::CuVecOrMat, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix, locs::NTuple{M, Int}) where {C, M}
    # reorder a unirary matrix.
    kf = un_kernel(nactive(state), cbits, cvals, U0, locs)

    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end
cunapply!(state::CuVecOrMat, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::IMatrix, locs::NTuple{M, Int}) where {C, M} = state
cunapply!(state::CuVecOrMat, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::SDSparseMatrixCSC, locs::NTuple{M, Int}) where {C, M} = cunapply!(state, cbits, cvals, U0 |> Matrix, locs)

################## General U1 apply! ###################
for MT in [:SDDiagonal, :SDPermMatrix, :SDMatrix, :IMatrix, :SDSparseMatrixCSC]
@eval function u1apply!(state::CuVecOrMat, U1::$MT, ibit::Int)
    kf = u1_kernel(U1, ibit::Int)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end
end

################## XYZ #############
for G in [:x, :y, :z, :s, :t, :sdag, :tdag]
    KERNEL = Symbol(G, :_kernel)
    FUNC = Symbol(G, :apply!)
    @eval function $FUNC(state::CuVecOrMat, bits::Ints{Int})
        length(bits) == 0 && return state

        kf = $KERNEL(bits)
        X, Y = cudiv(size(state)...)
        @cuda threads=X blocks=Y simple_kernel(kf, state)
        state
    end
    @eval $FUNC(state::CuVecOrMat, bit::Int) = invoke($FUNC, Tuple{CuVecOrMat, Ints{Int}}, state, bit)

    CFUNC = Symbol(:c, FUNC)
    CKERNEL = Symbol(:c, KERNEL)
    @eval function $CFUNC(state::CuVecOrMat, cbits, cvals, bits::Int)
        kf = $CKERNEL(cbits, cvals, bits)
        X, Y = cudiv(size(state)...)
        @cuda threads=X blocks=Y simple_kernel(kf, state)
        state
    end
    @eval $CFUNC(state::CuVecOrMat, cbit::Int, cval::Int, ibit::Int) = invoke($CFUNC, Tuple{CuVecOrMat, Any, Any, Int}, state, cbit, cval, ibit)
end

function Yao.Blocks.swapapply!(state::CuVecOrMat, b1::Int, b2::Int)
    mask1 = bmask(b1)
    mask2 = bmask(b2)

    X, Y = cudiv(size(state)...)
    function kf(state, mask1, mask2)
        inds = ((blockIdx().x-1) * blockDim().x + threadIdx().x,
                       (blockIdx().y-1) * blockDim().y + threadIdx().y)
        b = inds[1]-1
        c = inds[2]
        c <= size(state, 2) || return nothing
        if b&mask1==0 && b&mask2==mask2
            i = b+1
            i_ = b ⊻ (mask1|mask2) + 1
            temp = state[i, c]
            state[i, c] = state[i_, c]
            state[i_, c] = temp
        end
        nothing
    end
    @cuda threads=X blocks=Y kf(state, mask1, mask2)
    state
end

#=
nbit = 6
N = 1<<nbit
LOC1 = SVector{2}([0, 1])
vn = randn(ComplexF32, N, 3)
v1 = randn(ComplexF32, N)
unapply!(vn |> cu, mat(H), (3,)) |> Matrix ≈ unapply!(vn |> copy, mat(H), (3,))
u1apply!(vn |> cu, mat(H), 3) |> Matrix ≈ u1apply!(vn |> copy, mat(H), 3)
czapply!(vn |> cu, [5], [1], 3) |> Matrix ≈ czapply!(vn |> copy, [5], [1], 3)
czapply!(v1 |> cu, [5], [1], 3) |> Vector ≈ czapply!(v1 |> copy, [5], [1], 3)
zapply!(vn |> cu, 3) |> Matrix ≈ zapply!(vn |> copy, 3)
zapply!(v1 |> cu, [3,1,4]) |> Vector ≈ zapply!(v1 |> copy, [3,1,4])
@device_code_warntype unapply!(vn |> cu, mat(H), (3,))

# =#
################### Multi Controlled Version ####################

#= Testing Code
using BenchmarkTools
@device_code_warntype zapply!(cv1, 3)

nbit = 10
N = 1<<nbit
LOC1 = SVector{2}([0, 1])
v1 = randn(ComplexF32, N)
vn = randn(ComplexF32, N, 3)
cv1 = v1 |> cu

#@benchmark cyapply!(cv1, (4,5), (0, 1), 3)

@device_code_warntype cyapply!(cv1, (4,5), (0, 1), 3)
=#
