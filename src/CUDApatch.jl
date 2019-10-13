#import CuArrays: _cuview, ViewIndex, NonContiguous
#using GPUArrays: genperm
# fallback to SubArray when the view is not contiguous

#=
function LinearAlgebra.permutedims!(dest::GPUArray, src::GPUArray, perm) where N
    perm isa Tuple || (perm = Tuple(perm))
    gpu_call(dest, (dest, src, perm)) do state, dest, src, perm
        I = @cartesianidx src state
        @inbounds dest[genperm(I, perm)...] = src[I...]
        return
    end
    return dest
end
=#

import CUDAnative: pow, abs, angle
for (RT, CT) in [(:Float64, :ComplexF64), (:Float32, :ComplexF32)]
    @eval cp2c(d::$RT, a::$RT) = CUDAnative.ComplexF64(d*CUDAnative.cos(a), d*CUDAnative.sin(a))
    for NT in [RT, :Int32]
        @eval CUDAnative.pow(z::$CT, n::$NT) = CUDAnative.ComplexF64((CUDAnative.pow(CUDAnative.abs(z), n)*CUDAnative.cos(n*CUDAnative.angle(z))), (CUDAnative.pow(CUDAnative.abs(z), n)*CUDAnative.sin(n*CUDAnative.angle(z))))
    end
end

@inline function bit_count(x::UInt32)
    x = ((x >> 1) & 0b01010101010101010101010101010101) + (x & 0b01010101010101010101010101010101)
    x = ((x >> 2) & 0b00110011001100110011001100110011) + (x & 0b00110011001100110011001100110011)
    x = ((x >> 4) & 0b00001111000011110000111100001111) + (x & 0b00001111000011110000111100001111)
    x = ((x >> 8) & 0b00000000111111110000000011111111) + (x & 0b00000000111111110000000011111111)
    x = ((x >> 16)& 0b00000000000000001111111111111111) + (x & 0b00000000000000001111111111111111)
    return x
end

@inline function bit_count(x::Int32)
    x = ((x >> 1) & Int32(0b01010101010101010101010101010101)) + (x & Int32(0b01010101010101010101010101010101))
    x = ((x >> 2) & Int32(0b00110011001100110011001100110011)) + (x & Int32(0b00110011001100110011001100110011))
    x = ((x >> 4) & Int32(0b00001111000011110000111100001111)) + (x & Int32(0b00001111000011110000111100001111))
    x = ((x >> 8) & Int32(0b00000000111111110000000011111111)) + (x & Int32(0b00000000111111110000000011111111))
    x = ((x >> 16)& Int32(0b00000000000000001111111111111111)) + (x & Int32(0b00000000000000001111111111111111))
    return x
end

bit_count(UInt32(0b11111))

# TODO
# support norm(view(reshape(A, m, n), :, 1))
using LinearAlgebra
import LinearAlgebra: norm
const CuSubArr{T, N} = Union{CuArray{T, N}, SubArray{T, N, <:CuArray}}
norm2(A::CuSubArr; dims=1) = mapreduce(abs2, +, A, dims=dims) .|> CUDAnative.sqrt

export piecewise, cudiv
@inline function cudiv(x::Int)
    max_threads = 256
    threads_x = min(max_threads, x)
    threads_x, ceil(Int, x/threads_x)
end

# NOTE: the maximum block size is 65535
@inline function cudiv(x::Int, y::Int)
    max_threads = 256
    threads_x = min(max_threads, x)
    threads_y = min(max_threads ÷ threads_x, y)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (x, y) ./ threads)
    threads, blocks
end

fix_cudiv(A::AbstractVector, firstdim::Int) = cudiv(firstdim)
fix_cudiv(A::AbstractMatrix, firstdim::Int) = cudiv(firstdim, size(A,2))

piecewise(state::AbstractVector, inds) = state
piecewise(state::AbstractMatrix, inds) = @inbounds view(state,:,inds[2])

import Base: kron, getindex
function kron(A::Union{CuArray{T1}, Adjoint{<:Any, <:CuArray{T1}}}, B::Union{CuArray{T2}, Adjoint{<:Any, <:CuArray{T2}}}) where {T1, T2}
    res = CuArrays.zeros(promote_type(T1,T2), (size(A).*size(B))...)
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
    res = CuArrays.zeros(T, size(B)...)
    @inline function kernel(res, A, B)
        state = (blockIdx().x-1) * blockDim().x + threadIdx().x
        state <= length(res) && (@inbounds res[state] = A[B[state]])
        return
    end

    X, Y = cudiv(length(B))
    @cuda threads=X blocks=Y kernel(res, A, B)
    res
end
