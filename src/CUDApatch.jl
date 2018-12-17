import GPUArrays: fast_isapprox, to_index
to_index(a::A, x::Array{ET}) where {A, ET} = copyto!(similar(a, ET, size(x)...), x)

Base.isapprox(A::GPUArray{T1}, B::GPUArray{T2}, rtol::Real = Base.rtoldefault(T1, T2, 0), atol::Real=0) where {T1, T2} = all(fast_isapprox.(A, B, T1(rtol)|>real, T1(atol)|>real))
Base.isapprox(A::AbstractArray{T1}, B::GPUArray{T2}, rtol::Real = Base.rtoldefault(T1, T2, 0), atol::Real=0) where {T1, T2} = all(fast_isapprox.(A, Array(B), T1(rtol)|>real, T1(atol)|>real))
Base.isapprox(A::GPUArray{T1}, B::AbstractArray{T2}, rtol::Real = Base.rtoldefault(T1, T2, 0), atol::Real=0) where {T1, T2} = all(fast_isapprox.(Array(A), B, T1(rtol)|>real, T1(atol)|>real))

import CuArrays: _cuview, ViewIndex, NonContiguous
using GPUArrays: genperm
# fallback to SubArray when the view is not contiguous
_cuview(A, I::Tuple, ::NonContiguous) = invoke(view, Tuple{AbstractArray, typeof(I).parameters...}, A, I...)

function LinearAlgebra.permutedims!(dest::GPUArray, src::GPUArray, perm) where N
    perm isa Tuple || (perm = Tuple(perm))
    gpu_call(dest, (dest, src, perm)) do state, dest, src, perm
        I = @cartesianidx src state
        @inbounds dest[genperm(I, perm)...] = src[I...]
        return
    end
    return dest
end

# TODO
# support norm(view(reshape(A, m, n), :, 1))
# support view(A, :, 1, :)[:,1]
# k,i,j = GPUArrays.gpu_ind2sub(regm, state), @cuprinf don't work
using LinearAlgebra
import LinearAlgebra: norm
const CuSubArr{T, N} = Union{CuArray{T, N}, SubArray{T, N, <:CuArray}}
norm2(A::CuSubArr; dims=1) = mapreduce(abs2, +, A, dims=dims) .|> sqrt

export piecewise, cudiv
@inline function cudiv(x::Int)
    max_threads = 256
    threads_x = min(max_threads, x)
    threads_x, ceil(Int, x/threads_x)
end

@inline function cudiv(x::Int, y::Int)
    max_threads = 256
    threads_x = min(max_threads, x)
    threads_y = min(max_threads ÷ threads_x, y)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (x, y) ./ threads)
    threads, blocks
end

piecewise(state::AbstractVector, inds) = state
piecewise(state::AbstractMatrix, inds) = @inbounds view(state,:,inds[2])

import Base: kron, getindex
function kron(A::Union{CuArray{T1}, Adjoint{<:Any, <:CuArray{T1}}}, B::Union{CuArray{T2}, Adjoint{<:Any, <:CuArray{T2}}}) where {T1, T2}
    res = cuzeros(promote_type(T1,T2), (size(A).*size(B))...)
    @inline function kernel(res, A, B)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        inds = GPUArrays.gpu_ind2sub(res, state)
        inds_A = (inds.-1) .÷ size(B) .+ 1
        inds_B = (inds.-1) .% size(B) .+ 1
        state <= length(res) && (@inbounds res[state] = A[inds_A...]*B[inds_B...])
        return
    end

    X, Y = cudiv(length(res))
    @cuda threads=X blocks=Y kernel(res, A, B)
    res
end

function getindex(A::CuVector{T}, B::CuArray{<:Integer}) where T
    res = cuzeros(T, size(B)...)
    @inline function kernel(res, A, B)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        state <= length(res) && (@inbounds res[state] = A[B[state]])
        return
    end

    X, Y = cudiv(length(B))
    @cuda threads=X blocks=Y kernel(res, A, B)
    res
end
