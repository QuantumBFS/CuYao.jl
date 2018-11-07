export kernel, KernelCompiled

"""
    kernel(blk::MatrixBlock) -> Function

Get the kernel of a block.
"""
function kernel end

function kernel(blk::Union{ChainBlock, Sequential})
    ks = Tuple(kernel(bi) for bi in subblocks(blk))
    ex = :()
    for kfi in ks
        ex = :($ex; $kfi(state, inds); CuArrays.sync_threads())
    end
        #ks[2](state, inds)
        #CuArrays.sync_threads()
        #ks[3](state, inds)
        #CuArrays.sync_threads()
        #end
    :(function kf(state, inds); $ex; end)
end

kernel(blk::PutBlock{N}) where N = :(un_kernel($N, (), (), $(mat(blk.block)), $(blk.addrs)))
kernel(blk::PutBlock{1}) = :(u1_kernel($(mat(blk.block)), $(blk.addrs[1])))

kernel(blk::ControlBlock{N}) where N = :(un_kernel($N, $(blk.ctrl_qubits), $(blk.vals), $(mat(blk.block)), $(blk.addrs)))

for G in [:X, :Y, :Z]#, :S, :T, :Sdag, :Tdag]
    GATE = Symbol(G, :Gate)
    KERNEL = Symbol(G |> string |> lowercase, :_kernel)
    CKERNEL = Symbol(:c, KERNEL)
    @eval function kernel(ctrl::ControlBlock{N, <:$GATE}) where N
        :($($CKERNEL)($(ctrl.ctrl_qubits), $(ctrl.vals), $(ctrl.addrs)...))
    end
    @eval function kernel(ctrl::PutBlock{N, 1, <:$GATE}) where N
        :($($KERNEL)($(ctrl.addrs)...))
    end
    @eval function kernel(ctrl::RepeatedBlock{N, 1, <:$GATE}) where N
        :($($KERNEL)($(ctrl.addrs)...))
    end
end

kernel(blk::Union{QDiff, CachedBlock}) = :((state, inds) -> nothing)

#################### KernelCompiled Block #################
import Yao.Blocks: istraitkeeper, chblock, usedbits, apply!, mat, print_block, adjoint, parent
"""
    KernelCompiled{GT, N, T} <: TagBlock{N, T}
    KernelCompiled(block) -> KernelCompiled

Mark a block as a CUDA-Kernel Compilable block.
"""
struct KernelCompiled{GT, N, T} <: TagBlock{N, T}
    block::GT
    KernelCompiled(block::MatrixBlock{N, T}) where {N, T} = new{typeof(block), N, T}(block)
end
kernel(kc::KernelCompiled) = kernel(parent(kc))
function apply!(reg::GPUReg, kc::KernelCompiled)
    println(kernel(kc))
    kf = @eval $(kernel(kc))
    X, Y = cudiv(size(reg.state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, reg.state)
    reg
end

istraitkeeper(::KernelCompiled) = Val(true)
parent(df::KernelCompiled) = df.block
chblock(kc::KernelCompiled, blk::MatrixBlock) = KernelCompiled(blk)

adjoint(df::KernelCompiled) = KernelCompiled(parent(df)')
mat(df::KernelCompiled) = mat(parent(df))
usedbits(df::KernelCompiled) = usedbits(parent(df))

function print_block(io::IO, df::KernelCompiled)
    printstyled(io, "[G] "; bold=true, color=:yellow)
    print(io, parent(df))
end
