module CuYao
using LuxurySparse, StaticArrays, LinearAlgebra
using StatsBase

using Yao, Yao.Blocks, Yao.Intrinsics, Yao.Boost
using Yao.Intrinsics: autostatic

using GPUArrays, CuArrays, CUDAnative
CuArrays.allowscalar(false)

include("CUDApatch.jl")
include("GPUReg.jl")
include("gpuapplys.jl")
end
