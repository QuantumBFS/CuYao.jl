using Yao.YaoBase
import Yao.YaoArrayRegister: u1rows!, unrows!, autostatic, instruct!, swaprows!

include("kernels.jl")

gpu_compatible(A::AbstractVecOrMat) = A |> staticize
gpu_compatible(A::StaticArray) = A

###################### unapply! ############################
function instruct!(::Val{2}, state::DenseCuVecOrMat, U0::AbstractMatrix, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M}
    @debug "The generic U(N) matrix of size ($(size(U0))), on: GPU, locations: $(locs), controlled by: $(clocs) = $(cvals)."
    U0 = gpu_compatible(U0)
    # reorder a unirary matrix.
    U = (all(TupleTools.diff(locs).>0) ? U0 : reorder(U0, collect(locs)|>sortperm)) |> staticize
    locked_bits = [cbits..., locs...]
    locked_vals = [cvals..., zeros(Int, M)...]
    locs_raw = [i+1 for i in itercontrol(nbit, setdiff(1:nbit, locs), zeros(Int, nbit-M))] |> staticize
    configs = itercontrol(nbit, locked_bits, locked_vals)

    len = length(configs)
    @inline function kernel(ctx, state, locs_raw, U, configs, len)
        inds = @idx replace_first(size(state), len)
        x = @inbounds configs[inds[1]]
        @inbounds unrows!(piecewise(state, inds), x .+ locs_raw, U)
        return
    end

    gpu_call(kernel, state, locs_raw, U, configs, len; elements=len*size(state,2))
    state
end
instruct!(::Val{2}, state::DenseCuVecOrMat, U0::IMatrix, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M} = state
instruct!(::Val{2}, state::DenseCuVecOrMat, U0::SDSparseMatrixCSC, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M} = instruct!(Val(2), state, U0 |> Matrix, locs, clocs, cvals)

################## General U1 apply! ###################
function instruct!(::Val{2}, state::DenseCuVecOrMat, U1::SDSparseMatrixCSC, ibit::Tuple{Int}, clocs::Tuple{}, cvals::Tuple{})
    instruct!(Val(2), state, Matrix(U1), ibit, clocs, cval)
end
function instruct!(::Val{2}, state::DenseCuVecOrMat, U1::AbstractMatrix, ibit::Tuple{Int}, clocs::Tuple{}, cvals::Tuple{})
    @debug "The generic U(2) matrix of size ($(size(U1))), on: GPU, locations: $(ibit), controlled by: $(clocs) = $(cvals)."
    a, c, b, d = U1
    nbit = log2dim1(state)
    step = 1<<(ibit-1)
    configs = itercontrol(nbit, [ibit], [0])

    len = length(configs)
    @inline function kernel(ctx, state, a, b, c, d, len)
        inds = @idx replace_first(size(state), len)
        i = @inbounds configs[inds[1]]+1
        @inbounds u1rows!(piecewise(state, inds), i, i+step, a, b, c, d)
        return
    end
    gpu_call(kf, state, a, b, c, d, len; elements=len*size(state,2))
    return state
end

function instruct!(::Val{2}, state::DenseCuVecOrMat, U1::SDPermMatrix, ibit::Tuple{Int}, clocs::Tuple{}, cvals::Tuple{})
    @debug "The single qubit permutation matrix of size ($(size(U1))), on: GPU, locations: $(ibit), controlled by: $(clocs) = $(cvals)."
    nbit = log2dim1(state)
    b, c = U1.vals
    step = 1<<(ibit-1)
    configs = itercontrol(nbit, [ibit], [0])

    len = length(configs)
    function kernel(ctx, state, b, c, step, len, configs)
        inds = @idx replace_first(size(state), len)
        x = @inbounds configs[inds[1]] + 1
        @inbounds swaprows!(piecewise(state, inds), x, x+step, c, b)
        return
    end
    gpu_call(kernel, state, b, c, step, len, configs; elements=len*size(state,2))
    return state
end

function instruct!(::Val{2}, state::DenseCuVecOrMat, U1::SDDiagonal, ibit::Tuple{Int}, clocs::Tuple{}, cvals::Tuple{})
    @debug "The single qubit diagonal matrix of size ($(size(U1))), on: GPU, locations: $(ibit), controlled by: $(clocs) = $(cvals)."
    a, d = U1.diag
    nbit = log2dim1(state)
    mask = bmask(ibit)
    @inline function kernel(ctx, state, a, d, mask)
        inds = @cartesianidx state
        i = inds[1]
        piecewise(state, inds)[i] *= anyone(i-1, mask) ? d : a
        return
    end
    gpu_call(kernel, state, a, d, mask; elements=length(state))
    return state
end

instruct!(::Val{2}, state::DenseCuVecOrMat, U::IMatrix, locs::Tuple{Int}) = state

################## XYZ #############
using Yao.ConstGate: S, T, Sdag, Tdag

function _instruct!(state::DenseCuVecOrMat, ::Val{:X}, locs::NTuple{L,Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {L,C}
    length(locs) == 0 && return state
    nbit = log2dim1(state)
    configs = itercontrol(nbit, [cbits..., locs[1]], [cvals..., 0])
    mask = bmask(locs...)
    len = length(configs)
    @inline function kernel(ctx, state, mask, len, configs)
        inds = @idx replace_first(size(state), len)
        b = @inbounds configs[inds[1]]
        @inbounds swaprows!(piecewise(state, inds), b+1, flip(b, mask) + 1)
        return
    end
    gpu_call(kernel, state, mask, len, configs; elements=len*size(state,2))
    return state
end

function _instruct!(state::DenseCuVecOrMat, ::Val{:Y}, locs::NTuple{C,Int}, clocs::NTuple{}, cvals::NTuple{}) where C
    length(locs) == 0 && return state
    nbit = log2dim1(state)
    configs = itercontrol(nbit, [locs[1]], [0])
    bit_parity = length(locs)%2 == 0 ? 1 : -1
    factor = (-im)^length(locs)
    len = length(configs)
    @inline function kernel(ctx, state, factor1, factor2, mask, bit_par, configs, len)
        inds = @idx replace_first(size(state), len)
        b = @inbounds configs[inds[1]]
        i_ = flip(b, mask) + 1
        factor1 = count_ones(b&mask)%2 == 1 ? -factor : factor
        factor2 = factor1*bit_parity
        @inbounds swaprows!(piecewise(state, inds), b+1, i_, factor2, factor1)
        return
    end
    gpu_call(kernel, state, factor1, factor2, mask, bit_par, configs, len; elements=len*size(state,2))
    return state
end

function _instruct!(state::DenseCuVecOrMat, ::Val{:Y}, loc::Tuple{Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where C
    length(locs) == 0 && return state
    nbit = log2dim1(state)
    configs = itercontrol(nbit, [cbits..., bit], [cvals..., 0])
    mask = bmask(bit)
    len = length(configs)
    @inline function kernel(ctx, state, configs, mask, len)
        inds = @idx replace_first(size(state), len)
        b = @inbounds configs[inds[1]]
        @inbounds swaprows!(piecewise(state, inds), b+1, flip(b, mask) + 1, im, -im)
        return
    end
    gpu_call(kernel, state, configs, mask, len; elements=len*size(state,2))
    return state
end

function _instruct!(state::DenseCuVecOrMat, ::Val{:Z}, locs::NTuple{Int}, clocs::Tuple{}, cvals::Tuple{})
    length(locs) == 0 && return state
    nbit = log2dim1(state)
    mask = bmask(locs...)
    @inline function kernel(ctx, state, mask)
        inds = @cartesianidx state
        i = inds[1]
        piecewise(state, inds)[i] *= count_ones((i-1)&mask)%2==1 ? -1 : 1
        return
    end
    gpu_call(kernel, state, mask; elements=length(state))
    return state
end


for (G, FACTOR) in zip([:S, :T, :Sdag, :Tdag], [:(-one(Int32)), :(1f0im), :($(exp(im*π/4))), :(-1f0im), :($(exp(-im*π/4)))])
    @eval function _instruct!(state::DenseCuVecOrMat, ::Val{:Z}, locs::NTuple{Int}, clocs::Tuple{}, cvals::Tuple{})
        length(locs) == 0 && return state
        nbit = log2dim1(state)
        mask = bmask(Int32, locs...)
        @inline function kernel(ctx, state)
            inds = @cartesianidx state
            i = inds[1]
            piecewise(state, inds)[i] *= $d ^ count_ones(Int32(i-1)&mask)
            return
        end
        gpu_call(kernel, state, mask; elements=length(state))
        return state
    end
end


for G in [:X, :Y, :Z, :S, :T, :Sdag, :Tdag]
    KERNEL = Symbol(G |> string |> lowercase, :_kernel)

    @eval function _instruct!(state::DenseCuVecOrMat, ::Val{$(QuoteNode(G))}, locs::NTuple{C,Int}) where C
        length(locs) == 0 && return state

        D, kf = $KERNEL(log2dim1(state), locs)
        gpu_call(kf, state; elements=D*size(state,2))
        state
    end

    CKERNEL = Symbol(:c, KERNEL)
    @eval function _instruct!(state::DenseCuVecOrMat, ::Val{$(QuoteNode(G))}, loc::Tuple{Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where C
        D,kf = $CKERNEL(log2dim1(state), clocs, cvals, loc...)
        gpu_call(kf, state; elements=D*size(state,2))
        state
    end

    @eval begin
        function YaoBase.instruct!(::Val{2}, state::DenseCuVecOrMat, g::Val{$(QuoteNode(G))}, locs::NTuple{C,Int}) where C
            _instruct!(state, g, locs, (), ())
        end

        function YaoBase.instruct!(::Val{2}, state::DenseCuVecOrMat, g::Val{$(QuoteNode(G))}, locs::Tuple{Int})
            _instruct!(state, g, locs, (), ())
        end

        function YaoBase.instruct!(::Val{2}, state::DenseCuVecOrMat, g::Val{$(QuoteNode(G))}, loc::Tuple{Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where C
            _instruct!(state, g, loc, clocs, cvals)
        end

        function YaoBase.instruct!(::Val{2}, state::DenseCuVecOrMat, vg::Val{$(QuoteNode(G))}, loc::Tuple{Int}, cloc::Tuple{Int}, cval::Tuple{Int})
            _instruct!(state, vg, loc, cloc, cval)
        end
    end

end

function instruct!(::Val{2}, state::DenseCuVecOrMat, ::Val{:SWAP}, locs::Tuple{Int,Int})
    b1, b2 = locs
    mask1 = bmask(b1)
    mask2 = bmask(b2)

    configs = itercontrol(log2dim1(state), [locs...], [1,0])
    function kf(ctx, state, mask1, mask2)
        inds = @idx replace_first(size(state), length(configs))

        b = configs[inds[1]]
        i = b+1
        i_ = b ⊻ (mask1|mask2) + 1
        swaprows!(piecewise(state, inds), i, i_)
        nothing
    end
    gpu_call(kf, state, mask1, mask2; elements=length(configs)*size(state,2))
    state
end

############## other gates ################
# parametrized swap gate
using Yao.ConstGate: SWAPGate

function instruct!(::Val{2}, state::DenseCuVecOrMat, ::Val{:PSWAP}, locs::Tuple{Int, Int}, θ::Real)
    nbit = log2dim1(state)
    mask1 = bmask(m)
    mask2 = bmask(n)
    mask12 = mask1|mask2
    a, c, b_, d = mat(Rx(theta))
    e = exp(-im/2*theta)
    configs = itercontrol(nbit, [m,n], [0,0])
    len = length(configs)
    @inline function kernel(ctx, state, mask2, mask12, configs, a, b_, c, d)
        inds = @idx replace_first(size(state), len)
        @inbounds x = configs[inds[1]]
        piecewise(state, inds)[x+1] *= e
        piecewise(state, inds)[x⊻mask12+1] *= e
        y = x ⊻ mask2
        @inbounds u1rows!(piecewise(state, inds), y+1, y⊻mask12+1, a, b_, c, d)
        return
    end
    gpu_call(kernel, state, mask2, mask12, configs, a, b_, c, d; elements=len*size(state,2))
    state
end

using Yao.YaoBlocks
function YaoBlocks._apply_fallback!(r::GPUReg{B,T}, b::AbstractBlock) where {B,T}
    YaoBlocks._check_size(r, b)
    r.state .= CUDA.adapt(CuArray{T}, mat(T, b)) * r.state
    return r
end

for RG in [:Rx, :Ry, :Rz]
    @eval function instruct!(
            ::Val{2}, 
            state::DenseCuVecOrMat{T},
            ::Val{$(QuoteNode(RG))},
            locs::Tuple{Int},
            theta::Number
        ) where {T, N}
        YaoArrayRegister.instruct!(Val(2), state, Val($(QuoteNode(RG))), locs, (), (), theta)
        return state
    end
end
