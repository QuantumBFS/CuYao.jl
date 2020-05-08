using Yao.YaoBase
import Yao.YaoArrayRegister: instruct!

using Yao, KernelAbstractions
using LuxurySparse, TupleTools, StaticArrays, CuArrays, BitBasis

macro cuda256(fcall, len)
    kernel = fcall.args[1]
    state = fcall.args[2]
    args = fcall.args[3:end]
    event = gensym()
    esc(quote
        $event = $kernel(CUDA(), 256)($state, $(args...); ndrange=($len, size($state,2)))
        wait($event)
        $state
    end)
end

###################### unapply! ############################
@kernel function un_kernel(state, U, configs, locs_raw) where {C, M}
    i, j = @index(Global, NTuple)
    x = @inbounds configs[i]
    inds = x .+ locs_raw
    @inbounds state[inds, j] = U * state[inds, j]
end

gpu_compatible(A::AbstractVecOrMat) = A |> staticize
gpu_compatible(A::StaticArray) = A

function Yao.instruct!(state::CuVecOrMat, U0::AbstractMatrix, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M}
    nbit = log2dim1(state)
    # reorder a unirary matrix.
    U = (all(TupleTools.diff(locs).>0) ? U0 : reorder(U0, collect(locs)|>sortperm)) |> gpu_compatible
    MM = size(U0, 1)
    locked_bits = [clocs..., locs...]
    locked_vals = [cvals..., zeros(Int, M)...]
    locs_raw = [i+1 for i in itercontrol(nbit, setdiff(1:nbit, locs), zeros(Int, nbit-M))] |> staticize
    configs = itercontrol(nbit, locked_bits, locked_vals)
    @cuda256 un_kernel(state, U, configs, locs_raw) length(configs)
end
instruct!(state::CuVecOrMat, U0::IMatrix, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M} = state
instruct!(state::CuVecOrMat, U0::SDSparseMatrixCSC, locs::NTuple{M, Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C, M} = instruct!(state, U0 |> Matrix, locs, clocs, cvals)

################## General U1 apply! ###################
@kernel function u1_kernel(state, config, step, a, b, c, d)
    i, j = @index(Global, NTuple)
    x = @inbounds config[i] + 1
    mu1rows!(state, x, x+step, j, a, b, c, d)
end
function Yao.instruct!(state::CuVecOrMat, U1::AbstractMatrix, locs::Tuple{Int})
    a, c, b, d = U1
    ibit = locs[1]
    nbit = log2dim1(state)
    step = 1<<(ibit-1)
    configs = itercontrol(nbit, [ibit], [0])
    @cuda256 u1_kernel(state, configs, step, a, b, c, d) length(configs)
end

Yao.instruct!(state::CuVecOrMat, U1::SDSparseMatrixCSC, locs::Tuple{Int}) = instruct!(state, U1|>Matrix, locs)

@kernel function u1_kernel_pm(state, config, step, b, c)
    i, j = @index(Global, NTuple)
    x = @inbounds config[i] + 1
    mswaprows!(state, x, x+step, j, c, b)
end

function Yao.instruct!(state::CuVecOrMat, U1::SDPermMatrix, locs::Tuple{Int})
    U1.perm[1] == 1 && return instruct!(state, Diagonal(U1), locs)
    nbit = log2dim1(state)
    ibit = locs[1]
    mask = bmask(ibit)
    b, c = U1.vals[1], U1.vals[2]
    step = 1<<(ibit-1)
    configs = itercontrol(nbit, [ibit], [0])
    @cuda256 u1_kernel_pm(state, configs, step, b, c) length(configs)
end

@kernel function u1_kernel_dg(state, ibit, a, d)
    i, j = @index(Global, NTuple)
    mask = bmask(ibit)
    state[i, j] *= anyone(i-1, mask) ? d : a
end

function Yao.instruct!(state::CuVecOrMat, U1::SDDiagonal, locs::Tuple{Int})
    a, d = U1.diag
    nbit = log2dim1(state)
    @cuda256 u1_kernel_dg(state, locs[1], a, d) 1<<nbit
end

@kernel function cdg_kernel(state, ctrl, a, d)
    i, j = @index(Global, NTuple)
    state[i, j] *= ctrl(i-1) ? d : a
end

function Yao.instruct!(state::CuVecOrMat, U1::SDDiagonal, locs::Tuple{Int}, clocs::NTuple{C,Int}, cvals::NTuple{C,Int}) where C
    a, d = U1.diag
    nbit = log2dim1(state)
    ctrl = controller((clocs..., locs[1]), (cvals..., 1))
    @cuda256 cdg_kernel(state, ctrl, a, d) 1<<nbit
end

instruct!(state::CuVecOrMat, U::IMatrix, locs::Tuple{Int}) = state

################## XYZ #############
using Yao.ConstGate: S, T, Sdag, Tdag

@kernel function x_kernel(state, configs, mask)
    i, j = @index(Global, NTuple)
    b = @inbounds configs[i]
    mswaprows!(state, b+1, flip(b, mask) + 1, j)
end

function instruct!(state::CuVecOrMat, ::Val{:X}, locs::NTuple{M,Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C,M}
    length(locs) == 0 && return state
    nbit = log2dim1(state)
    configs = itercontrol(nbit, [clocs..., locs[1]], [cvals..., 0])
    mask = bmask(locs...)
    @cuda256 x_kernel(state, configs, mask) length(configs)
end

@kernel function y_kernel(state, configs, mask, factor, bit_parity)
    i, j = @index(Global, NTuple)
    b = configs[i]
    i_ = flip(b, mask) + 1
    factor1 = count_ones(b&mask)%2 == 1 ? -factor : factor
    factor2 = factor1*bit_parity
    mswaprows!(state, i, i_, j, factor2, factor1)
end

function instruct!(state::CuVecOrMat, ::Val{:Y}, locs::NTuple{M,Int}) where {M}
    nbit = log2dim1(state)
    mask = bmask(Int, locs...)
    configs = itercontrol(nbit, [locs[1]], [0])
    bit_parity = length(locs)%2 == 0 ? 1 : -1
    factor = (-im)^length(locs)
    @cuda256 y_kernel(state, configs, mask) length(configs)
end

@kernel function z_kernel(state, mask)
    i, j = @index(Global, NTuple)
    state[i, j] *= count_ones((i-1)&mask)%2==1 ? -1 : 1
end

function instruct!(state::CuVecOrMat, ::Val{:Z}, locs::NTuple{M,Int}) where {M}
    mask = bmask(locs...)
    @cuda256 z_kernel(state, mask) size(state, 1)
end

@kernel function cy_kernel(state, configs, mask)
    i, j = @index(Global, NTuple)
    b = @inbounds configs[i]
    mswaprows!(state, b+1, flip(b, mask) + 1, j, im, -im)
end

function instruct!(state::CuVecOrMat, ::Val{:Y}, locs::NTuple{M,Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C,M}
    nbit = log2dim1(state)
    configs = itercontrol(nbit, [clocs..., bit], [cvals..., 0])
    mask = bmask(bit)
    @cuda256 cy_kernel(state, configs, mask) length(configs)
end

@kernel function swap_kernel(state, configs, mask1, mask2)
    i, j = @index(Global, NTuple)
    b = configs[i]
    i_ = b ⊻ (mask1|mask2) + 1
    mswaprows!(state, b + 1, i_, j)
end

function instruct!(state::CuVecOrMat, ::Val{:SWAP}, locs::Tuple{Int,Int})
    b1, b2 = locs
    mask1 = bmask(b1)
    mask2 = bmask(b2)
    configs = itercontrol(log2dim1(state), [locs...], [1,0])
    @cuda256 swap_kernel(state, configs, mask1, mask2) length(configs)
end

############## other gates ################
# parametrized swap gate
using Yao.ConstGate: SWAPGate

@kernel function pswap_kernel(state, configs, mask1, mask2, mask12, a, b_, c, d, e)
    i, j = @index(Global, NTuple)
    @inbounds x = configs[i]
    state[x+1, j] *= e
    state[x⊻mask12+1, j] *= e
    y = x ⊻ mask2
    mu1rows!(state, y+1, y⊻mask12+1, j, a, b_, c, d)
end

function instruct!(state::CuVecOrMat, ::Val{:PSWAP}, locs::Tuple{Int, Int}, θ::Real)
    nbit = log2dim1(state)
    m, n = locs
    mask1 = bmask(m)
    mask2 = bmask(n)
    mask12 = mask1|mask2
    a, c, b_, d = mat(Rx(θ))
    e = exp(-im/2*θ)
    configs = itercontrol(nbit, [m,n], [0,0])
    @cuda256 pswap_kernel(state, configs, mask1, mask2, mask12, a, b_, c, d, e) length(configs)
end

using Yao.YaoBlocks
function YaoBlocks._apply_fallback!(r::GPUReg{B,T}, b::AbstractBlock) where {B,T}
    YaoBlocks._check_size(r, b)
    r.state .= CuArrays.adapt(CuArray{T}, mat(T, b)) * r.state
    return r
end

for G in [:T, :Tdag, :S, :Sdag]
    @eval function instruct!(state::CuVecOrMat, ::Val{$(QuoteNode(G))}, locs::NTuple{M,Int}, clocs::Tuple{Int}, cvals::Tuple{Int}) where {M}
        invoke(instruct!, Tuple{CuVecOrMat, SDDiagonal, Tuple{Int}, NTuple{C,Int}, NTuple{C,Int}} where C,
            state, mat($G), locs, clocs, cvals)
    end
    @eval function instruct!(state::CuVecOrMat, ::Val{$(QuoteNode(G))}, locs::NTuple{M,Int}, clocs::NTuple{C, Int}, cvals::NTuple{C, Int}) where {C,M}
        invoke(instruct!, Tuple{CuVecOrMat, SDDiagonal, Tuple{Int}, NTuple{C,Int}, NTuple{C,Int}} where C,
            state, mat($G), locs, clocs, cvals)
    end
    @eval function instruct!(state::CuVecOrMat, ::Val{$(QuoteNode(G))}, locs::NTuple{M,Int}) where {M}
        invoke(instruct!, Tuple{CuVecOrMat, SDDiagonal, Tuple{Int}, NTuple{C,Int}, NTuple{C,Int}} where C,
            state, mat($G), locs, (), ())
    end
end
