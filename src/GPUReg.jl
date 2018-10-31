using Yao
using Yao.Blocks

using GPUArrays
using CuArrays
using CUDAnative
CuArrays.allowscalar(false)

import CuArrays: cu

cu(reg::DefaultRegister{B}) where B = DefaultRegister{B}(cu(reg.state))
const GPUReg{B, T, MT} = DefaultRegister{B, T, MT} where MT<:GPUArray

reg = rand_state(10)
greg = reg |> cu
