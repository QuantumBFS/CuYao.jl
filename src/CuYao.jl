module CuYao
using LuxurySparse, StaticArrays, LinearAlgebra, Base.Cartesian
using StatsBase
using BitBasis
using Reexport
import TupleTools
using Random

using Yao.YaoArrayRegister
using CUDA
using CUDA.GPUArrays: gpu_call, @linearidx, @cartesianidx, linear_index
@reexport using Yao
using Yao.YaoArrayRegister
using Yao.YaoBlocks
using Yao.ConstGate: SWAPGate
using Yao.ConstGate: S, T, Sdag, Tdag

import Yao.YaoArrayRegister: insert_qudits!, join
import CUDA: cu
import Yao.YaoArrayRegister: _measure, measure, measure!
import Yao.YaoArrayRegister: batch_normalize!
import Yao.YaoBlocks: BlockedBasis, nblocks, subblock
import Yao: expect
import Yao.YaoArrayRegister: u1rows!, unrows!, autostatic, instruct!, swaprows!
import LinearAlgebra: norm
import Base: kron, getindex

export cpu, cu, AbstractCuArrayReg, CuArrayReg, CuBatchedArrayReg, CuDensityMatrix,
    cuzero_state, cuuniform_state, curand_state, cuproduct_state, cughz_state

const Ints = NTuple{<:Any, Int}

include("CUDApatch.jl")
include("register.jl")
include("instructs.jl")

function __init__()
    CUDA.allowscalar(false)
end

end
