import YaoArrayRegister: u1rows!, unrows!, autostatic, instruct!, swaprows!

include("kernels.jl")

gpu_compatible(A::AbstractVecOrMat) = A |> staticize
gpu_compatible(A::StaticArray) = A

###################### unapply! ############################
function instruct!(state::CuVecOrMat, U0::AbstractMatrix, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M}
    U0 = gpu_compatible(U0)
    # reorder a unirary matrix.
    kf = un_kernel(log2dim1(state), clocs, cvals, U0, locs)

    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end
instruct!(state::CuVecOrMat, U0::IMatrix, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M} = state
instruct!(state::CuVecOrMat, U0::SDSparseMatrixCSC, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M} = instruct!(state, U0 |> Matrix, locs, clocs, cvals)

################## General U1 apply! ###################
for MT in [:SDDiagonal, :SDPermMatrix, :AbstractMatrix, :IMatrix, :SDSparseMatrixCSC]
    @eval function instruct!(state::CuVecOrMat, U1::$MT, ibit::Tuple{Int})
        kf = u1_kernel(U1, ibit...)
        X, Y = cudiv(size(state)...)
        @cuda threads=X blocks=Y simple_kernel(kf, state)
        state
    end
end

################## XYZ #############
using Yao.ConstGate: S, T, Sdag, Tdag
for G in [:X, :Y, :Z, :S, :T, :Sdag, :Tdag]
    KERNEL = Symbol(G |> string |> lowercase, :_kernel)
    @eval function instruct!(state::CuVecOrMat, ::Val{$(QuoteNode(G))}, bits::NTuple{<:Any,Int})
        length(bits) == 0 && return state

        kf = $KERNEL(bits)
        X, Y = cudiv(size(state)...)
        @cuda threads=X blocks=Y simple_kernel(kf, state)
        state
    end

    CKERNEL = Symbol(:c, KERNEL)
    @eval function instruct!(state::CuVecOrMat, ::Val{$(QuoteNode(G))}, loc::Tuple{Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where C
        kf = $CKERNEL(clocs, cvals, loc...)
        X, Y = cudiv(size(state)...)
        @cuda threads=X blocks=Y simple_kernel(kf, state)
        state
    end
    @eval instruct!(state::CuVecOrMat, vg::Val{$(QuoteNode(G))}, loc::Tuple{Int}, cloc::Tuple{Int}, cval::Tuple{Int}) = invoke(instruct!, Tuple{CuVecOrMat, Val{$(QuoteNode(G))}, Tuple{Int}, NTuple{C, Int}, NTuple{C, Int}} where C, state, vg, loc, cloc, cval)
    @eval instruct!(state::CuVecOrMat, vg::Val{$(QuoteNode(G))}, loc::Tuple{Int}) = invoke(instruct!, Tuple{CuVecOrMat, Val{$(QuoteNode(G))}, NTuple{<:Any, Int}} where C, state, vg, loc)
end

function instruct!(state::CuVecOrMat, ::Val{:SWAP}, locs::Tuple{Int,Int})
    b1, b2 = locs
    mask1 = bmask(b1)
    mask2 = bmask(b2)

    X, Y = cudiv(size(state)...)
    function kf(state, mask1, mask2)
        inds = ((blockIdx().x-1) * blockDim().x + threadIdx().x,
                       (blockIdx().y-1) * blockDim().y + threadIdx().y)
        b = inds[1]-1
        c = inds[2]
        c <= size(state, 2) || return nothing
        if b&mask1==0 && b&mask2==mask2
            i = b+1
            i_ = b ⊻ (mask1|mask2) + 1
            temp = state[i, c]
            state[i, c] = state[i_, c]
            state[i_, c] = temp
        end
        nothing
    end
    @cuda threads=X blocks=Y kf(state, mask1, mask2)
    state
end

############## other gates ################
# parametrized swap gate
using Yao.ConstGate: SWAPGate

function instruct!(state::CuVecOrMat, ::Val{:PSWAP}, locs::Tuple{Int, Int}, θ::Real)
    kf = pswap_kernel(locs..., θ)
    X, Y = cudiv(size(state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, state)
    state
end
