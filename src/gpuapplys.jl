using Test
using Yao, Yao.Blocks, Yao.Intrinsics, Yao.Boost

using GPUArrays
using CuArrays
using CUDAnative
CuArrays.allowscalar(false)

using LuxurySparse, StaticArrays, LinearAlgebra

import Yao.Intrinsics: unrows!, u1apply!, _unapply!, swaprows!
import Yao.Boost: zapply!, xapply!, yapply!, cxapply!, cyapply!, czapply!, sapply!, sdagapply!, tapply!, tdagapply!

include("GeneralApply.jl")

function _xkernel(state, mask, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    b = i-1
    if c(b)
        i_ = flip(b, mask) + 1
        swaprows!(state, i, i_)
    end
    return
end

function _ykernel(state, mask, c, factor, bit_parity)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    b = i-1
    if c(b)
        i_ = flip(b, mask) + 1
        factor1 = count_ones(b&mask)%2 == 1 ? -factor : factor
        factor2 = factor1*bit_parity
        swaprows!(state, i, i_, factor2, factor1)
    end
    return
end

function xapply!(state::CuVecOrMat{T}, bits::Ints) where T
    length(bits) == 0 && return state
    mask = bmask(bits...)
    c = controller(bits[1], 0)
    @cuda blocks=size(state, 1) _xkernel(state, mask, c)
    state
end

function yapply!(state::CuVecOrMat{T}, bits::Ints{Int}) where T
    length(bits) == 0 && return state
    mask = bmask(Int, bits...)
    c = controller(bits[1], 0)
    bit_parity = length(bits)%2 == 0 ? 1 : -1
    factor = T(-im)^length(bits)
    @cuda blocks=size(state, 1) _ykernel(state, mask, c, factor, bit_parity)
    state
end

function _zlike_apply!(widget, state::CuVecOrMat, bits::Ints{Int})
    mask = bmask(Int, bits...)
    state .= widget.(mask, state, basis(state))
end

function _czlike_apply!(widget, state::CuVecOrMat{T}, cbits, cvals, b2::Int) where T
    c = controller([cbits..., b2[1]], [cvals..., 1])
    f = (v, b) -> c(b) ? widget(v) : v
    state .= f.(state, basis(state))
    state
end

for (G, FACTOR) in zip([:z, :s, :t, :sdag, :tdag], [:(-1), :(im), :($(exp(im*π/4))), :(-im), :($(exp(-im*π/4)))])
    FUNC = Symbol(G, :apply!)
    CFUNC = Symbol(:c, G, :apply!)
    WIDGET_N = Symbol(G, :_widget_n)
    WIDGET_1 = Symbol(G, :_widget_1)
    if G == :z
        WIDGET_N = :((mask, v, b)->count_ones(b&mask)%2==1 ? -v : v)
        WIDGET_1 = :((mask, v, b)->testany(b, mask) ? -v : v)
        WIDGET_C = :(v->-v)
    else
        WIDGET_N = :((mask, v, b)-> $FACTOR^count_ones(b&mask) * v)
        WIDGET_1 = :((mask, v, b)->testany(b, mask) ? $FACTOR*v : v)
        WIDGET_C = :(v->v*$FACTOR)
    end
    @eval $FUNC(state::CuVecOrMat, bits::Ints{Int}) = _zlike_apply!($WIDGET_N, state, bits)
    @eval $FUNC(state::CuVecOrMat, bit::Int) = _zlike_apply!($WIDGET_1, state, bit)
    @eval $CFUNC(state::CuVecOrMat, cbits, cvals, bit::Int) = _czlike_apply!($WIDGET_C, state, cbits, cvals, bit)
end

################### Multi Controlled Version ####################
function cxapply!(state::CuVecOrMat{T}, cbits, cvals, bits::Ints) where T
    length(bits) == 0 && return state
    c = controller([cbits..., bits[1]], [cvals..., 0])
    mask = bmask(bits...)
    @cuda blocks=size(state, 1) _xkernel(state, mask, c)
    state
end

function cyapply!(state::CuVecOrMat, cbits, cvals, bit::Int)
    c = controller([cbits..., bit], [cvals..., 0])
    mask = bmask(bit)
    function kernel(state, mask, c)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        b = i-1
        if c(b)
            i_ = flip(b, mask) + 1
            swaprows!(state, i, i_, im, -im)
        end
        return
    end
    @cuda blocks=size(state, 1) kernel(state, mask, c)
    state
end

cyapply!(state::CuVecOrMat, cbit::Int, cval::Int, b2::Int) = cyapply!(state, [cbit], cval, b2)
cxapply!(state::CuVecOrMat, cbit::Int, cval::Int, b2::Int) = cxapply!(state, [cbit], cval, b2)
czapply!(state::CuVecOrMat, cbit::Int, cval::Int, b2::Int) = czapply!(state, [cbit], cval, b2)

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
