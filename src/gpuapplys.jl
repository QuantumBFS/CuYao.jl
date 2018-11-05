#=
using Test
using Yao, Yao.Blocks, Yao.Intrinsics, Yao.Boost

using GPUArrays, CuArrays, CUDAnative
CuArrays.allowscalar(false)

using LuxurySparse, StaticArrays, LinearAlgebra
=#

import Yao.Intrinsics: unrows!, u1apply!, _unapply!, swaprows!, cunapply!
import Yao.Boost: zapply!, xapply!, yapply!, cxapply!, cyapply!, czapply!, sapply!, sdagapply!, tapply!, tdagapply!

include("kernels.jl")

###################### unapply! ############################
function _unapply!(state::CuVecOrMat, U::AbstractMatrix, locs_raw::SDVector, ctrl)
    X, Y = cudiv(size(state)...)
    kf = un_kernel(ctrl, U, locs_raw)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end
@eval function _unapply!(state::CuVecOrMat, U::SDSparseMatrixCSC, locs_raw::SDVector, ctrl)
    _unapply!(state, SMatrix{size(U, 1), size(U,2)}(U), locs_raw, ctrl)
end

function cunapply!(state::CuVecOrMat, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix, locs::NTuple{M, Int}) where {C, M}
    # reorder a unirary matrix.
    U = all(diff(locs).>0) ? U0 : reorder(U0, collect(locs)|>sortperm)
    N, MM = nactive(state), size(U0, 1)
    locked_bits = [cbits..., locs...]
    locked_vals = [cvals..., zeros(Int, M)...]
    locs_raw = [i+1 for i in itercontrol(N, setdiff(1:N, locs), zeros(Int, N-M))]
    ctrl = controller(locked_bits, locked_vals)

    _unapply!(state, U |> autostatic, locs_raw |> autostatic, ctrl)
end
cunapply!(state::CuVecOrMat, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::IMatrix, locs::NTuple{M, Int}) where {C, M} = state

################## General U1 apply! ###################
# diagonal
function u1apply!(state::CuVecOrMat, U1::SDDiagonal, ibit::Int)
    mask = bmask(ibit)
    a, d = U1.diag
    kf = u1diag_kernel(mask, a, d)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end

# sparse
u1apply!(state::CuVecOrMat, U1::SDSparseMatrixCSC, ibit::Int) = u1apply!(state, U1|>Matrix, ibit)

# dense
function u1apply!(state::CuVecOrMat, U1::SDMatrix, ibit::Int)
    # reorder a unirary matrix.
    a, c, b, d = U1
    step = 1<<(ibit-1)
    ctrl = controller([ibit], [0])
    kf = u1_kernel(ctrl, step, a, b, c, d)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end

# perm
function u1apply!(state::CuVecOrMat{T}, U1::SDPermMatrix, ibit::Int) where T
    U1.perm[1] == 1 && return u1apply!(state, Diagonal(U1), ibit)

    mask = bmask(ibit)
    b, c = T(U1.vals[1]), T(U1.vals[2])
    step = 1<<(ibit-1)
    ctrl = controller([ibit], [0])
    kf = u1pm_kernel(ctrl, step, c, b)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end

# identity
u1apply!(state::CuVecOrMat, U1::IMatrix, ibit::Int) = state

################## XYZ #############
xapply!(state::CuVecOrMat, bits::Ints) = cxapply!(state, (), (), bits)
function cxapply!(state::CuVecOrMat{T}, cbits, cvals, bits::Ints) where T
    length(bits) == 0 && return state
    c = controller([cbits..., bits[1]], [cvals..., 0])
    mask = bmask(bits...)

    kf = x_kernel(mask, c)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end
cxapply!(state::CuVecOrMat, cbit::Int, cval::Int, b2::Int) = cxapply!(state, [cbit], cval, b2)

function yapply!(state::CuVecOrMat{T}, bits::Ints{Int}) where T
    length(bits) == 0 && return state
    mask = bmask(Int, bits...)
    c = controller(bits[1], 0)
    bit_parity = length(bits)%2 == 0 ? 1 : -1
    factor = T(-im)^length(bits)

    kf = y_kernel(mask, c, factor, bit_parity)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end

function cyapply!(state::CuVecOrMat, cbits, cvals, bit::Int)
    c = controller([cbits..., bit], [cvals..., 0])
    mask = bmask(bit)

    kf = cy_kernel(mask, c)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end
cyapply!(state::CuVecOrMat, cbit::Int, cval::Int, b2::Int) = cyapply!(state, [cbit], cval, b2)

function zapply!(state::CuVecOrMat, bits::Ints{Int})
    length(bits) == 0 && return state
    mask = bmask(bits...)

    kf = z_kernel(mask)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end

function zapply!(state::CuVecOrMat, bit::Int)
    mask = bmask(bit)

    kf = u1diag_kernel(mask, 1, -1)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end

function _zlike_apply!(state::CuVecOrMat, bits::Ints; factor)
    length(bits) == 0 && return state
    mask = bmask(bits...)

    kf = zlike_kernel(mask, factor)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end

function _czapply!(state::CuVecOrMat{T}, cbits, cvals, b2::Int; factor) where T
    c = controller([cbits..., b2[1]], [cvals..., 1])

    kf = cdg_kernel(c, 1, factor)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end
czapply!(state::CuVecOrMat, cbits, cvals, b2::Int) = _czapply!(state, cbits, cvals, b2, factor=-1)
czapply!(state::CuVecOrMat, cbit::Int, cval::Int, b2::Int) = czapply!(state, [cbit], cval, b2)

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
for (G, FACTOR) in zip([:s, :t, :sdag, :tdag], [:(im), :($(exp(im*π/4))), :(-im), :($(exp(-im*π/4)))])
    FUNC = Symbol(G, :apply!)
    CFUNC = Symbol(:c, G, :apply!)
    @eval $FUNC(state::CuVecOrMat, bits::Ints{Int}) = _zlike_apply!(state, bits, factor=$FACTOR)
    @eval $FUNC(state::CuVecOrMat, bit::Int) = _zlike_apply!(state, bit, factor=$FACTOR)
    @eval $CFUNC(state::CuVecOrMat, cbits, cvals, bit::Int) = _czapply!(state, cbits, cvals, bit, factor=$FACTOR)
end

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
