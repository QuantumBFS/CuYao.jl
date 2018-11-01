module CuYao
using LuxurySparse, StaticArrays, LinearAlgebra
using StatsBase

using Yao, Yao.Blocks, Yao.Intrinsics, Yao.Boost

using GPUArrays, CuArrays, CUDAnative
CuArrays.allowscalar(false)

include("CUDApack.jl")
include("GPUReg.jl")
include("gpuapplys.jl")
end
