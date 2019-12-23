import CuArrays: cu
import Yao.YaoArrayRegister: _measure, measure, measure!, measure_collapseto!, measure_remove!
import Yao.YaoBase: batch_normalize!
import Yao: expect

export cpu, cu, GPUReg

cu(reg::ArrayReg{B}) where B = ArrayReg{B}(CuArray(reg.state))
cpu(reg::ArrayReg{B}) where B = ArrayReg{B}(collect(reg.state))
const GPUReg{B, T, MT} = ArrayReg{B, T, MT} where MT<:GPUArray

function batch_normalize!(s::CuSubArr, p::Real=2)
    p!=2 && throw(ArgumentError("p must be 2!"))
    s./=norm2(s, dims=1)
    s
end

@inline function tri2ij(l::Int)
    i = ceil(Int, sqrt(2*l+0.25)-0.5)
    j = l-i*(i-1)÷2
    i+1,j
end

############### MEASURE ##################
function measure(::ComputationalBasis, reg::GPUReg{1}, ::AllLocs; rng::AbstractRNG=Random.GLOBAL_RNG, nshots::Int=1)
    _measure(rng, reg |> probs |> Vector, nshots)
end

# TODO: optimize the batch dimension using parallel sampling
function measure(::ComputationalBasis, reg::GPUReg{B}, ::AllLocs; rng::AbstractRNG=Random.GLOBAL_RNG, nshots::Int=1) where B
    regm = reg |> rank3
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    _measure(rng, pl |> Matrix, nshots)
end

function measure!(::RemoveMeasured, ::ComputationalBasis, reg::GPUReg{B}, ::AllLocs; rng::AbstractRNG=Random.GLOBAL_RNG) where B
    regm = reg |> rank3
    nregm = similar(regm, 1<<nremain(reg), B)
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    pl_cpu = pl |> Matrix
    res_cpu = map(ib->_measure(rng, view(pl_cpu, :, ib), 1)[], 1:B)
    res = CuArray(res_cpu)
    @inline function kernel(nregm, regm, res, pl)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if state <= length(nregm)
            i,j = GPUArrays.gpu_ind2sub(nregm, state)
            r = Int(res[j])+1
            @inbounds nregm[i,j] = regm[r,i,j]/CUDAnative.sqrt(pl[r, j])
        end
        return
    end
    X, Y = cudiv(length(nregm))
    @cuda threads=X blocks=Y kernel(nregm, regm, res, pl)
    reg.state = reshape(nregm,1,:)
    B == 1 ? Array(res)[] : res
end

function measure!(::NoPostProcess, ::ComputationalBasis, reg::GPUReg{B, T}, ::AllLocs; rng::AbstractRNG=Random.GLOBAL_RNG) where {B, T}
    regm = reg |> rank3
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    pl_cpu = pl |> Matrix
    res_cpu = map(ib->_measure(rng, view(pl_cpu, :, ib), 1)[], 1:B)
    res = CuArray(res_cpu)

    @inline function kernel(regm, res, pl)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if state <= length(regm)
            k,i,j = GPUArrays.gpu_ind2sub(regm, state)
            @inbounds rind = Int(res[j]) + 1
            @inbounds regm[k,i,j] = k==rind ? regm[k,i,j]/CUDAnative.sqrt(pl[k, j]) : T(0)
        end
        return
    end

    X, Y = cudiv(length(regm))
    @cuda threads=X blocks=Y kernel(regm, res, pl)
    B == 1 ? Array(res)[] : res
end

function measure!(rst::ResetTo, ::ComputationalBasis, reg::GPUReg{B, T}, ::AllLocs; rng::AbstractRNG=Random.GLOBAL_RNG) where {B, T}
    regm = reg |> rank3
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    pl_cpu = pl |> Matrix
    res_cpu = map(ib->_measure(rng, view(pl_cpu, :, ib), 1)[], 1:B)
    res = CuArray(res_cpu)

    @inline function kernel(regm, res, pl, val)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if state <= length(regm)
            k,i,j = GPUArrays.gpu_ind2sub(regm, state)
            @inbounds rind = Int(res[j]) + 1
            @inbounds k==val+1 && (regm[k,i,j] = regm[rind,i,j]/CUDAnative.sqrt(pl[rind, j]))
            CuArrays.sync_threads()
            @inbounds k!=val+1 && (regm[k,i,j] = 0)
        end
        return
    end

    X, Y = cudiv(length(regm))
    @cuda threads=X blocks=Y kernel(regm, res, pl, rst.x)
    B == 1 ? Array(res)[] : res
end

import Yao.YaoArrayRegister: insert_qubits!, join
function batched_kron(A::Union{CuArray{T1, 3}, Adjoint{<:Any, <:CuArray{T1, 3}}}, B::Union{CuArray{T2, 3}, Adjoint{<:Any, <:CuArray{T2, 3}}}) where {T1 ,T2}
    res = CuArrays.zeros(promote_type(T1,T2), size(A,1)*size(B, 1), size(A,2)*size(B,2), size(A, 3))
    @inline function kernel(res, A, B)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        i,j,b = GPUArrays.gpu_ind2sub(res, state)
        i_A = (i-1) ÷ size(B,1) + 1
        j_A = (j-1) ÷ size(B,2) + 1
        i_B = (i-1) % size(B,1) + 1
        j_B = (j-1) % size(B,2) + 1
        state <= length(res) && (@inbounds res[state] = A[i_A, j_A, b]*B[i_B, j_B, b])
        return
    end

    X, Y = cudiv(length(res))
    @cuda threads=X blocks=Y kernel(res, A, B)
    res
end

function join(reg1::GPUReg{B}, reg2::GPUReg{B}) where {B}
    s1 = reg1 |> rank3
    s2 = reg2 |> rank3
    state = batched_kron(s1, s2)
    ArrayReg{B}(copy(reshape(state, size(state, 1), :)))
end

export insert_qubits!
function insert_qubits!(reg::GPUReg{B}, loc::Int; nqubits::Int=1) where B
    na = nactive(reg)
    focus!(reg, 1:loc-1)
    reg2 = join(zero_state(nqubits; nbatch=B) |> cu, reg) |> relax! |> focus!((1:na+nqubits)...)
    reg.state = reg2.state
    reg
end

#=
for FUNC in [:measure!, :measure!]
    @eval function $FUNC(rng::AbstractRNG, op::AbstractBlock, reg::GPUReg, al::AllLocs; kwargs...) where B
        E, V = eigen!(mat(op) |> Matrix)
        ei = Eigen(E|>cu, V|>cu)
        $FUNC(rng::AbstractRNG, ei, reg, al; kwargs...)
    end
end
=#
