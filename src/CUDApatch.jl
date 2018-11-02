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
