using Yao.YaoBase
import Yao.YaoArrayRegister: u1rows!, unrows!, autostatic, instruct!, swaprows!

include("kernels.jl")

gpu_compatible(A::AbstractVecOrMat) = A |> staticize
gpu_compatible(A::StaticArray) = A

###################### unapply! ############################
function instruct!(::Val{2}, state::DenseCuVecOrMat, U0::AbstractMatrix, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M}
    U0 = gpu_compatible(U0)
    # reorder a unirary matrix.
    D, kf = un_kernel(log2dim1(state), clocs, cvals, U0, locs)

    gpu_call(kf, state; elements=D*size(state,2))
    state
end
instruct!(::Val{2}, state::DenseCuVecOrMat, U0::IMatrix, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M} = state
instruct!(::Val{2}, state::DenseCuVecOrMat, U0::SDSparseMatrixCSC, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M} = instruct!(Val(2), state, U0 |> Matrix, locs, clocs, cvals)

################## General U1 apply! ###################
for MT in [:SDDiagonal, :SDPermMatrix, :AbstractMatrix, :SDSparseMatrixCSC]
    @eval function instruct!(::Val{2}, state::DenseCuVecOrMat, U1::$MT, ibit::Tuple{Int})
        D,kf = u1_kernel(log2dim1(state), U1, ibit...)
        gpu_call(kf, state; elements=D*size(state,2))
        state
    end
end
instruct!(::Val{2}, state::DenseCuVecOrMat, U::IMatrix, locs::Tuple{Int}) = state

################## XYZ #############
using Yao.ConstGate: S, T, Sdag, Tdag

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
            _instruct!(state, g, locs)
        end

        function YaoBase.instruct!(::Val{2}, state::DenseCuVecOrMat, g::Val{$(QuoteNode(G))}, locs::Tuple{Int})
            _instruct!(state, g, locs)
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
    D, kf = pswap_kernel(log2dim1(state),locs..., θ)
    gpu_call(kf, state; elements=D*size(state,2))
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
        YaoArrayRegister.instruct!(state, Val($(QuoteNode(RG))), locs, (), (), theta)
        return state
    end
end