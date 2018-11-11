using Yao, CuYao
using BenchmarkTools

import Yao.Intrinsics: controller
using Yao.Intrinsics

reg = rand_state(9, 1000) |> cu
@benchmark reg |> put(12, 3=>Z)
@benchmark reg |> control(12, 6, 3=>X)
@benchmark reg |> put(12, 3=>rot(X, 0.3))
