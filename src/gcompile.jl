export kernel, KernelCompiled

"""
    kernel(blk::MatrixBlock) -> Function

Get the kernel of a block.
"""
function kernel end

function kernel(blk::Union{ChainBlock, Sequential})
    #ks = Tuple(kernel(bi) for bi in subblocks(blk))
    ks = Tuple(kernel(bi) for bi in subblocks(blk)[1:1])
    function kf(state, inds)
        ks[1](state, inds)#; CuArrays.sync_threads()
        #ks[2](state, inds); CuArrays.sync_threads()
        #ks[3](state, inds); CuArrays.sync_threads()
        #for kfi in ks
        #    kfi(state, inds); CuArrays.sync_threads()
        #end
    end
end

kernel(blk::PutBlock{N}) where N = un_kernel(N, (), (), mat(blk.block), blk.addrs)
kernel(blk::PutBlock{1}) = u1_kernel(mat(blk.block), blk.addrs[1])

kernel(blk::ControlBlock{N}) where N = un_kernel(N, blk.ctrl_qubits, blk.vals, mat(blk.block), blk.addrs)

for G in [:X, :Y, :Z]#, :S, :T, :Sdag, :Tdag]
    GATE = Symbol(G, :Gate)
    KERNEL = Symbol(G |> string |> lowercase, :_kernel)
    CKERNEL = Symbol(:c, KERNEL)
    @eval function kernel(ctrl::ControlBlock{N, <:$GATE}) where N
        $CKERNEL(ctrl.ctrl_qubits, ctrl.vals, ctrl.addrs...)
    end
    @eval function kernel(ctrl::PutBlock{N, 1, <:$GATE}) where N
        $KERNEL(ctrl.addrs...)
    end
    @eval function kernel(ctrl::RepeatedBlock{N, 1, <:$GATE}) where N
        $KERNEL(ctrl.addrs...)
    end
end

kernel(blk::Union{QDiff, CachedBlock}) = (state, inds) -> nothing

#################### KernelCompiled Block #################
import YaoBlocks: istraitkeeper, chcontent, occupied_locs, apply!, mat, print_block, content
"""
    KernelCompiled{GT, N, T} <: TagBlock{N, T}
    KernelCompiled(block) -> KernelCompiled

Mark a block as a CUDA-Kernel Compilable block.
"""
struct KernelCompiled{GT, N, T} <: TagBlock{N, T}
    block::GT
    KernelCompiled(block::MatrixBlock{N, T}) where {N, T} = new{typeof(block), N, T}(block)
end
kernel(kc::KernelCompiled) = kernel(content(kc))
function apply!(reg::GPUReg, kc::KernelCompiled)
    kf = kernel(kc)
    X, Y = cudiv(size(reg.state)...)
    @cuda threads=X blocks=Y simple_kernel(kf, reg.state)
    reg
end

istraitkeeper(::KernelCompiled) = Val(true)
content(df::KernelCompiled) = df.block
chcontent(kc::KernelCompiled, blk::MatrixBlock) = KernelCompiled(blk)

Base.adjoint(df::KernelCompiled) = KernelCompiled(content(df)')
mat(df::KernelCompiled) = mat(content(df))
occupied_locs(df::KernelCompiled) = occupied_locs(content(df))

function print_block(io::IO, df::KernelCompiled)
    printstyled(io, "[G] "; bold=true, color=:yellow)
    #print(io, content(df))
end
