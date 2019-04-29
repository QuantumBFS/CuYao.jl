import CuArrays: cu
import YaoArrayRegister: _measure, measure, measure!, measure_collapseto!, measure_remove!
import YaoBase: batch_normalize!
import Yao: expect

export cpu, cu, GPUReg

cu(reg::ArrayReg{B}) where B = ArrayReg{B}(cu(reg.state))
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
measure(::ComputationalBasis, reg::GPUReg{1}, ::AllLocs; nshots::Int=1) = _measure(reg |> probs |> Vector, nshots)
# TODO: optimize the batch dimension using parallel sampling
function measure(::ComputationalBasis, reg::GPUReg{B}, ::AllLocs; nshots::Int=1) where B
    regm = reg |> rank3
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    _measure(pl |> Matrix, nshots)
end

function measure_remove!(::ComputationalBasis, reg::GPUReg{B}, ::AllLocs) where B
    regm = reg |> rank3
    nregm = similar(regm, 1<<nremain(reg), B)
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    pl_cpu = pl |> Matrix
    res_cpu = map(ib->_measure(view(pl_cpu, :, ib), 1)[], 1:B)
    res = CuArray(res_cpu)
    @inline function kernel(nregm, regm, res, pl)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if state <= length(nregm)
            i,j = GPUArrays.gpu_ind2sub(nregm, state)
            @inbounds nregm[i,j] = regm[res[j]+1,i,j]/CUDAnative.sqrt(pl[res[j]+1, j])
        end
        return
    end
    X, Y = cudiv(length(nregm))
    @cuda threads=X blocks=Y kernel(nregm, regm, res, pl)
    reg.state = reshape(nregm,1,:)
    res
end

function measure!(::ComputationalBasis, reg::GPUReg{B, T}, ::AllLocs) where {B, T}
    regm = reg |> rank3
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    pl_cpu = pl |> Matrix
    res_cpu = map(ib->_measure(view(pl_cpu, :, ib), 1)[], 1:B)
    res = CuArray(res_cpu)

    @inline function kernel(regm, res, pl)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if state <= length(regm)
            k,i,j = GPUArrays.gpu_ind2sub(regm, state)
            @inbounds rind = res[j] + 1
            @inbounds regm[k,i,j] = k==rind ? regm[k,i,j]/CUDAnative.sqrt(pl[k, j]) : T(0)
        end
        return
    end

    X, Y = cudiv(length(regm))
    @cuda threads=X blocks=Y kernel(regm, res, pl)
    res
end

function measure_collapseto!(::ComputationalBasis, reg::GPUReg{B, T}, ::AllLocs; config=0) where {B, T}
    regm = reg |> rank3
    pl = dropdims(mapreduce(abs2, +, regm, dims=2), dims=2)
    pl_cpu = pl |> Matrix
    res_cpu = map(ib->_measure(view(pl_cpu, :, ib), 1)[], 1:B)
    res = CuArray(res_cpu)

    @inline function kernel(regm, res, pl, val)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        if state <= length(regm)
            k,i,j = GPUArrays.gpu_ind2sub(regm, state)
            @inbounds rind = res[j] + 1
            @inbounds k==val+1 && (regm[k,i,j] = regm[rind,i,j]/CUDAnative.sqrt(pl[rind, j]))
            CuArrays.sync_threads()
            @inbounds k!=val+1 && (regm[k,i,j] = 0)
        end
        return
    end

    X, Y = cudiv(length(regm))
    @cuda threads=X blocks=Y kernel(regm, res, pl, config)
    res
end

import YaoArrayRegister: insert_qubits!, join
function batched_kron(A::Union{CuArray{T1, 3}, Adjoint{<:Any, <:CuArray{T1, 3}}}, B::Union{CuArray{T2, 3}, Adjoint{<:Any, <:CuArray{T2, 3}}}) where {T1 ,T2}
    res = cuzeros(promote_type(T1,T2), size(A,1)*size(B, 1), size(A,2)*size(B,2), size(A, 3))
    @inline function kernel(res, A, B)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        inds = GPUArrays.gpu_ind2sub(res, state)
        i_A = (inds[1]-1) ÷ size(B,1) + 1
        j_A = (inds[2]-1) ÷ size(B,2) + 1
        i_B = (inds[1]-1) % size(B,1) + 1
        j_B = (inds[2]-1) % size(B,2) + 1
        b = inds[3]
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
    ArrayReg{B}(reshape(state, size(state, 1), :))
end

export insert_qubits!
function insert_qubits!(reg::GPUReg{B}, loc::Int; nqubits::Int=1) where B
    na = nactive(reg)
    focus!(reg, 1:loc-1)
    reg2 = join(zero_state(nqubits; nbatch=B) |> cu, reg) |> relax! |> focus!((1:na+nqubits)...)
    reg.state = reg2.state
    reg
end

for FUNC in [:measure!, :measure_collapseto!, :measure_remove!]
    @eval function $FUNC(op::AbstractBlock, reg::GPUReg, al::AllLocs; kwargs...) where B
        E, V = eigen!(mat(op) |> Matrix)
        ei = Eigen(E|>cu, V|>cu)
        $FUNC(ei, reg, al; kwargs...)
    end
end
