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
# =#

import CuArrays: cu
import Yao.Registers: _measure, measure, measure!, measure_reset!, measure_remove!

include("CUDApack.jl")

export cpu, cu, GPUReg

cu(reg::DefaultRegister{B}) where B = DefaultRegister{B}(cu(reg.state))
cpu(reg::DefaultRegister{B}) where B = DefaultRegister{B}(collect(reg.state))
const GPUReg{B, T, MT} = DefaultRegister{B, T, MT} where MT<:GPUArray

############### MEASURE ##################
measure(reg::GPUReg{1}, nshot::Int=1) = _measure(reg |> probs |> Vector, nshot)
# TODO: optimize the batch dimension using parallel sampling
function measure(reg::GPUReg{B}, nshot::Int=1) where B
    pl = dropdims(sum(reg |> rank3 .|> abs2 |> Array, dims=2), dims=2)
    _measure(pl, nshot)
end

function measure_remove!(reg::GPUReg{B}) where B
    state = reg |> rank3
    nstate = similar(reg.state, 1<<nremain(reg), B)
    pl = dropdims(sum(state .|> abs2, dims=2), dims=2)
    pl_cpu = pl |> Matrix
    res = Vector{Int}(undef, B)
    @inbounds for ib = 1:B
        ires = _measure(view(pl_cpu, :, ib), 1)[]
        nstate[:,ib] = state[ires+1,:,ib]./sqrt(pl_cpu[ires+1, ib])
        res[ib] = ires
    end
    reg.state = reshape(nstate,1,:)
    res
end

#=
function measure!(reg::GPUReg{B}) where B
    state = reg |> rank3
    nstate = zero(state)
    res = measure_remove!(reg)
    _nstate = reshape(reg.state, :, B)
    for ib in 1:B
        @inbounds nstate[res[ib]+1, :, ib] = _nstate[:,ib]
    end
    reg.state = reshape(nstate, size(state, 1), :)
    res
end

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
