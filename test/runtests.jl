using CUDA
CUDA.allowscalar(false)
include("CUDApatch.jl")
include("GPUReg.jl")
include("gpuapplys.jl")
#include("gcompile.jl")
