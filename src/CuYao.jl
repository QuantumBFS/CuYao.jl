module CuYao
using LuxurySparse, StaticArrays, LinearAlgebra, Base.Cartesian
using StatsBase

using Yao, Yao.Blocks, Yao.Intrinsics, Yao.Boost
using Yao.Intrinsics: autostatic, staticize
using GPUArrays, CuArrays, CUDAnative
CuArrays.allowscalar(false)

include("CUDApatch.jl")
include("GPUReg.jl")
include("gpuapplys.jl")
include("gcompile.jl")
end
