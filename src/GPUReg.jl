# TODO fix focus!(reg, non-continuous bits)
#=
using BenchmarkTools, Test
using Yao
using Yao.Blocks
using LinearAlgebra, LuxurySparse
using StatsBase

using GPUArrays
using CuArrays
using CUDAnative
CuArrays.allowscalar(false)
include("CUDApatch.jl")
# =#

import CuArrays: cu
import Yao.Registers: _measure, measure, measure!, measure_reset!, measure_remove!
import Yao.Intrinsics: batch_normalize!

export cpu, cu, GPUReg

cu(reg::DefaultRegister{B}) where B = DefaultRegister{B}(cu(reg.state))
cpu(reg::DefaultRegister{B}) where B = DefaultRegister{B}(collect(reg.state))
const GPUReg{B, T, MT} = DefaultRegister{B, T, MT} where MT<:GPUArray

function batch_normalize!(s::CuSubArr, p::Real=2)
    p!=2 && throw(ArgumentError("p must be 2!"))
    s./=norm2(s, dims=1)
    s
end

############### MEASURE ##################
measure(reg::GPUReg{1}; nshot::Int=1) = _measure(reg |> probs |> Vector, nshot)
# TODO: optimize the batch dimension using parallel sampling
function measure(reg::GPUReg{B}; nshot::Int=1) where B
    regm = reg |> rank3
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    _measure(pl |> Matrix, nshot)
end

function measure_remove!(reg::GPUReg{B}) where B
    regm = reg |> rank3
    nregm = similar(regm, 1<<nremain(reg), B)
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    pl_cpu = pl |> Matrix
    res_cpu = map(ib->_measure(view(pl_cpu, :, ib), 1)[], 1:B)
    res = CuArray(res_cpu)
    gpu_call(nregm, (nregm, regm, res, pl)) do state, nregm, regm, res, pl
        i,j = @cartesianidx nregm state
        @inbounds nregm[i,j] = regm[res[j]+1,i,j]/CUDAnative.sqrt(pl[res[j]+1, j])
        return
    end
    reg.state = reshape(nregm,1,:)
    res
end

function measure!(reg::GPUReg{B, T}) where {B, T}
    regm = reg |> rank3
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    pl_cpu = pl |> Matrix
    res_cpu = map(ib->_measure(view(pl_cpu, :, ib), 1)[], 1:B)
    res = CuArray(res_cpu)

    @inline function kernel(regm, res, pl)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        k,i,j = GPUArrays.gpu_ind2sub(regm, state)
        rind = res[j] + 1
        regm[k,i,j] = k==rind ? regm[k,i,j]/CUDAnative.sqrt(pl[k, j]) : T(0)
        return
    end

    X, Y = cudiv(length(regm))
    @cuda threads=X blocks=Y kernel(regm, res, pl)
    res
end

function measure_reset!(reg::GPUReg{B, T}; val=0) where {B, T}
    regm = reg |> rank3
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    pl_cpu = pl |> Matrix
    res_cpu = map(ib->_measure(view(pl_cpu, :, ib), 1)[], 1:B)
    res = CuArray(res_cpu)

    @inline function kernel(regm, res, pl)
        k,i,j = @cartesianidx regm
        @inbounds rind = res[j] + 1
        @inbounds k==val+1 && (regm[k,i,j] = regm[rind,i,j]/CUDAnative.sqrt(pl[rind, j]))
        CuArrays.sync_threads()
        @inbounds k!=val+1 && (regm[k,i,j] = 0)
        return
    end

    X, Y = cudiv(length(regm))
    @cuda threads=X blocks=Y kernel(regm, res, pl)
    res
end

#=function kernel(regm, res, pl)
    ki = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    K = size(regm,1)
    k = (ki-1)%K + 1
    i = (ki-1)÷K + 1
    rind = res[j] + 1
    regm[k,i,j] = k==rind ? regm[k,i,j]/sqrt(pl[rind, j]) : T(0)
    return
end
threads, blocks = cudiv(size(regm,1)*size(regm,2), size(regm, 3))
@cuda threads=threads blocks=blocks kernel(regm, res, pl)
# =#

#=
function _measure(pl::CuVector, ntimes::Int)
    sample(0:length(pl)-1, Weights(pl), ntimes)
end
_measure(pl::AbstractVector, ntimes::Int) = sample(0:length(pl)-1, Weights(pl), ntimes)

function _measure(pl::AbstractMatrix, ntimes::Int)
    B = size(pl, 2)
    res = Matrix{Int}(undef, ntimes, B)
   # CuArrays.QURAND.randn()
    for ib=1:B
        @inbounds res[:,ib] = _measure(view(pl,:,ib), ntimes)
    end
    res
end

pl = rand_state(16, 3) |> probs |> cu
=#
