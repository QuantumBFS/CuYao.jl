import CuArrays: cu
import Yao.Registers: _measure, measure, measure!, measure_reset!, measure_remove!
import Yao.Intrinsics: batch_normalize!
import Yao: expect

export cpu, cu, GPUReg

cu(reg::DefaultRegister{B}) where B = DefaultRegister{B}(cu(reg.state))
cpu(reg::DefaultRegister{B}) where B = DefaultRegister{B}(collect(reg.state))
const GPUReg{B, T, MT} = DefaultRegister{B, T, MT} where MT<:GPUArray

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

function expect(stat::StatFunctional{2, <:Function}, xs::CuVector{T}) where T
    N = length(xs)
    s = reduce(+, stat.data.(xs', xs))
    d = mapreduce(xi->stat.data(xi, xi), +, xs)
    (s-d)/(N*(N-1))
end

function expect(stat::StatFunctional{2, <:Function}, xs::CuVector, ys::CuVector)
    M = length(xs)
    N = length(ys)
    reduce(+, stat.data.(xs', ys))/M/N
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

function measure!(reg::GPUReg{B, T}) where {B, T}
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

function measure_reset!(reg::GPUReg{B, T}; val=0) where {B, T}
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
    @cuda threads=X blocks=Y kernel(regm, res, pl, val)
    res
end
