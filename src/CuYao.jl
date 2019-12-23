module CuYao
using LuxurySparse, StaticArrays, LinearAlgebra, Base.Cartesian
using StatsBase
using BitBasis
using Reexport
import TupleTools
using Random

using Yao.YaoArrayRegister
using GPUArrays, CuArrays, CUDAnative
@reexport using Yao

const Ints = NTuple{<:Any, Int}

include("CUDApatch.jl")
include("GPUReg.jl")
include("gpuapplys.jl")
#include("gcompile.jl")

function __init__()
    CuArrays.allowscalar(false)
end

end
